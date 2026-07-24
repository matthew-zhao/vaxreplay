"""Narrow host/guest RPC for an untrusted Agentic Replay harness.

The guest gets one ordered request/response stream.  It can page the logical workspace, ask the
host-owned provider gateway for a model completion, and submit exactly one final answer.  It never
receives a provider capability secret, a filesystem path, an HTTP proxy, or a general command API.

The HMAC on the terminal artifact uses a host-only key.  It authenticates the supervisor's exact
RPC observations; it is deliberately separate from the provider-gateway session seal and from the
eventual run receipt.

This channel alone is not proof of complete harness tracing: local computation and direct writes to
the guest scratch drive do not traverse it, and its retry cache is process-local.  A host crash must
therefore fail the run instead of resuming the RPC session.  Official admission still needs the
microVM lifecycle evidence and a run-finalization schema which cross-binds this terminal artifact.
"""

from __future__ import annotations

import base64
import enum
import hashlib
import hmac
import socket
import struct
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, Self, TypeVar

from pydantic import BaseModel, Field, JsonValue, ValidationError, field_validator, model_validator

from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest, AgenticModelResponse
from vaxreplay.agentic.gateway_auth import gateway_capability_id
from vaxreplay.agentic.protocol import AgenticTool
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedProviderGateway,
    GatewayCapabilityGrant,
    build_gateway_request_frame,
    gateway_capability_grant_sha256,
    parse_gateway_response_frame,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.run_artifact import AgenticToolEvent
from vaxreplay.agentic.task_protocol import (
    AgenticRuntimeSubmission,
    AgenticTaskInvocation,
    agentic_task_invocation_sha256,
    validate_submission_for_invocation,
)
from vaxreplay.agentic.workspace import AgenticLogicalWorkspaceBroker, AgenticWorkspaceError
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

GUEST_RPC_POLICY_SCHEMA_VERSION = 'vaxreplay.guest-rpc-policy.v0.1'
GUEST_RPC_REQUEST_SCHEMA_VERSION = 'vaxreplay.guest-rpc-request.v0.1'
GUEST_RPC_RESPONSE_SCHEMA_VERSION = 'vaxreplay.guest-rpc-response.v0.1'
GUEST_RPC_ATTEMPT_SCHEMA_VERSION = 'vaxreplay.guest-rpc-attempt.v0.1'
GUEST_RPC_SESSION_SEAL_SCHEMA_VERSION = 'vaxreplay.guest-rpc-session-seal.v0.2'
AUTHENTICATED_GUEST_RPC_SESSION_SCHEMA_VERSION = 'vaxreplay.authenticated-guest-rpc-session.v0.2'
GUEST_RPC_SESSION_AUTHENTICATION = 'hmac-sha256-domain-separated'

DEFAULT_MAX_GUEST_RPC_BODY_BYTES = 4 * 1024 * 1024
_MAX_RPC_BODY_BYTES = 64 * 1024 * 1024
_FRAME_HEADER = struct.Struct('>I')
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_SESSION_ID_PATTERN = r'^[0-9a-f]{32}$'
_SEAL_KEY_ID_DOMAIN = b'vaxreplay.guest-rpc-session-key-id.v0.1\x00'
_SEAL_HMAC_DOMAIN = b'vaxreplay.guest-rpc-session-seal.v0.2\x00'
_EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()

ModelT = TypeVar('ModelT', bound=BaseModel)


class GuestRpcError(ValueError):
    """Malformed framing, noncanonical JSON, or a locally rejected RPC exchange."""


class GuestRpcRemoteError(RuntimeError):
    """Stable body-free error returned to the guest."""

    def __init__(self, code: GuestRpcErrorCode):
        super().__init__(code.value)
        self.code = code


class GuestRpcMethod(str, enum.Enum):
    LIST = 'list_workspace'
    READ = 'read_workspace'
    SEARCH = 'search_workspace'
    MODEL_GENERATE = 'model_generate'
    SUBMIT = 'submit'


class GuestRpcErrorCode(str, enum.Enum):
    WRONG_SESSION = 'wrong_session'
    OUT_OF_ORDER = 'out_of_order'
    REPLAY_CONFLICT = 'replay_conflict'
    UNKNOWN_METHOD = 'unknown_method'
    INVALID_BODY = 'invalid_body'
    WORKSPACE_REJECTED = 'workspace_rejected'
    GATEWAY_REJECTED = 'gateway_rejected'
    SUBMISSION_REJECTED = 'submission_rejected'
    LIMIT_EXCEEDED = 'limit_exceeded'
    CONCURRENT_REQUEST = 'concurrent_request'
    CONNECTION_CLOSED = 'connection_closed'
    TERMINAL = 'terminal'
    INTERNAL = 'internal'


class GuestRpcTerminalStatus(str, enum.Enum):
    COMPLETED = 'completed'
    FAILED = 'failed'
    ABORTED = 'aborted'


class GuestRpcPolicy(StrictModel):
    """Organizer-owned bounds and identity for the RPC endpoint."""

    schema_version: Literal['vaxreplay.guest-rpc-policy.v0.1'] = GUEST_RPC_POLICY_SCHEMA_VERSION
    rpc_server_id: str = Field(min_length=1, max_length=200)
    rpc_server_version: str = Field(min_length=1, max_length=200)
    rpc_server_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    maximum_frame_body_bytes: int = Field(default=DEFAULT_MAX_GUEST_RPC_BODY_BYTES, ge=1024, le=_MAX_RPC_BODY_BYTES)
    maximum_session_wire_bytes: int = Field(default=64 * 1024 * 1024, ge=4096, le=1024 * 1024 * 1024)
    maximum_requests: int = Field(default=10_000, ge=1, le=100_000)
    maximum_list_entries: int = Field(default=100, ge=1, le=1000)
    maximum_read_bytes: int = Field(default=1024 * 1024, ge=1, le=16 * 1024 * 1024)
    maximum_search_results: int = Field(default=100, ge=1, le=1000)
    maximum_submission_bytes: int = Field(default=1024 * 1024, ge=1, le=16 * 1024 * 1024)
    one_request_at_a_time: Literal[True] = True
    contiguous_sequences_required: Literal[True] = True
    general_network_methods_exposed: Literal[False] = False
    shell_methods_exposed: Literal[False] = False
    provider_credentials_exposed: Literal[False] = False
    exactly_one_final_submission: Literal[True] = True

    @model_validator(mode='after')
    def validate_bounds(self) -> Self:
        if self.maximum_session_wire_bytes < 2 * self.maximum_frame_body_bytes:
            raise ValueError('RPC session wire budget must hold at least one maximum request and response')
        # Base64 expands by 4/3.  The factor of two leaves deterministic room for the envelope.
        if self.maximum_read_bytes > self.maximum_frame_body_bytes // 2:
            raise ValueError('RPC read limit is too large for a bounded base64 response')
        if self.maximum_submission_bytes > self.maximum_frame_body_bytes // 2:
            raise ValueError('RPC submission limit is too large for its bounded request envelope')
        return self


class GuestRpcRequest(StrictModel):
    schema_version: Literal['vaxreplay.guest-rpc-request.v0.1'] = GUEST_RPC_REQUEST_SCHEMA_VERSION
    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    sequence: int = Field(ge=0, le=2**63 - 1)
    # A string, rather than an enum, lets the host return and authenticate a stable rejection for an
    # unknown method.  Dispatch still uses the closed GuestRpcMethod enum below.
    method: str = Field(min_length=1, max_length=100, pattern=r'^[a-z][a-z0-9_]*$')
    body: dict[str, JsonValue]


class GuestRpcResponse(StrictModel):
    schema_version: Literal['vaxreplay.guest-rpc-response.v0.1'] = GUEST_RPC_RESPONSE_SCHEMA_VERSION
    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    sequence: int = Field(ge=0, le=2**63 - 1)
    succeeded: bool
    result: dict[str, JsonValue] | None = None
    error_code: GuestRpcErrorCode | None = None
    error_message: Literal['rpc request rejected'] | None = None

    @model_validator(mode='after')
    def validate_outcome(self) -> Self:
        if self.succeeded:
            if self.result is None or self.error_code is not None or self.error_message is not None:
                raise ValueError('successful RPC responses require only a result body')
        elif self.result is not None or self.error_code is None or self.error_message is None:
            raise ValueError('failed RPC responses require only a stable body-free error')
        return self


class ListWorkspaceRequest(StrictModel):
    cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)


class LogicalFileResult(StrictModel):
    path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)


class ListWorkspaceResult(StrictModel):
    files: tuple[LogicalFileResult, ...]
    next_cursor: int | None = Field(default=None, ge=0)


class ReadWorkspaceRequest(StrictModel):
    path: str = Field(min_length=1, max_length=4096)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(gt=0, le=16 * 1024 * 1024)


class ReadWorkspaceResult(StrictModel):
    content_base64: str
    offset: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    eof: bool

    @model_validator(mode='after')
    def validate_content(self) -> Self:
        try:
            decoded = base64.b64decode(self.content_base64, validate=True)
        except ValueError as error:
            raise ValueError('read response must contain canonical base64') from error
        if base64.b64encode(decoded).decode('ascii') != self.content_base64 or len(decoded) != self.byte_count:
            raise ValueError('read response base64 does not match its byte count')
        return self

    @property
    def content(self) -> bytes:
        return base64.b64decode(self.content_base64, validate=True)


class SearchWorkspaceRequest(StrictModel):
    needle: str = Field(min_length=1)
    paths: tuple[str, ...] | None = None
    max_results: int = Field(default=100, ge=1, le=1000)

    @field_validator('needle')
    @classmethod
    def validate_needle(cls, value: str) -> str:
        if len(value.encode('utf-8')) > 4096:
            raise ValueError('search needle cannot exceed 4096 UTF-8 bytes')
        return value


class LogicalSearchHitResult(StrictModel):
    path: str = Field(min_length=1)
    start_byte: int = Field(ge=0)
    end_byte: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.end_byte <= self.start_byte:
            raise ValueError('search hit must use a nonempty byte interval')
        return self


class SearchWorkspaceResult(StrictModel):
    hits: tuple[LogicalSearchHitResult, ...]


class ModelGenerateRequest(StrictModel):
    messages: tuple[AgenticModelMessage, ...] = Field(min_length=1, max_length=1000)
    max_output_tokens: int = Field(gt=0)
    response_schema_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_first_message(self) -> Self:
        if self.messages[0].role != 'system':
            raise ValueError('the first model message must be a system message')
        return self


class ModelGenerateResult(StrictModel):
    response: AgenticModelResponse


class SubmitRequest(StrictModel):
    submission: AgenticRuntimeSubmission


class SubmitResult(StrictModel):
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    submission_bytes: int = Field(gt=0)


class GuestRpcAttempt(StrictModel):
    """One host-observed non-replay request and its exact canonical response."""

    schema_version: Literal['vaxreplay.guest-rpc-attempt.v0.1'] = GUEST_RPC_ATTEMPT_SCHEMA_VERSION
    attempt_index: int = Field(ge=0)
    sequence_accepted: bool
    request: GuestRpcRequest
    response: GuestRpcResponse
    started_at: datetime
    finished_at: datetime
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    request_bytes: int = Field(gt=0)
    response_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_bytes: int = Field(gt=0)
    tool: AgenticTool | None = None
    gateway_call_index: int | None = Field(default=None, ge=0)
    tool_request_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    tool_request_bytes: int | None = Field(default=None, ge=0)
    tool_response_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    tool_response_bytes: int | None = Field(default=None, ge=0)
    projected_tool_event_index: int | None = Field(default=None, ge=0)

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_attempt(self) -> Self:
        request_bytes = canonical_json_bytes(self.request)
        response_bytes = canonical_json_bytes(self.response)
        if self.finished_at < self.started_at:
            raise ValueError('RPC attempt cannot finish before it starts')
        if (self.request.session_id, self.request.sequence) != (
            self.response.session_id,
            self.response.sequence,
        ):
            raise ValueError('RPC response identity must echo its request')
        if (
            self.request_sha256,
            self.request_bytes,
            self.response_sha256,
            self.response_bytes,
        ) != (
            _sha256(request_bytes),
            len(request_bytes),
            _sha256(response_bytes),
            len(response_bytes),
        ):
            raise ValueError('RPC attempt hashes or byte counts do not match its exact bodies')
        tool_fields = (
            self.tool_request_sha256,
            self.tool_request_bytes,
            self.tool_response_sha256,
            self.tool_response_bytes,
        )
        if self.tool is None:
            if self.gateway_call_index is not None or any(value is not None for value in tool_fields):
                raise ValueError('non-tool RPC attempts cannot carry tool evidence')
            if self.projected_tool_event_index is not None:
                raise ValueError('non-tool RPC attempts cannot project a tool event')
        elif any(value is None for value in tool_fields):
            raise ValueError('tool RPC attempts require exact tool request and response bindings')
        if self.tool != AgenticTool.MODEL_GENERATE and self.gateway_call_index is not None:
            raise ValueError('only model RPC attempts may carry a gateway call index')
        if self.tool == AgenticTool.MODEL_GENERATE and self.response.succeeded and self.gateway_call_index is None:
            raise ValueError('successful model RPC attempts must carry a gateway call index')
        if self.projected_tool_event_index is not None and self.tool is None:
            raise ValueError('projected events require a tool')
        return self


class GuestRpcSessionSeal(StrictModel):
    schema_version: Literal['vaxreplay.guest-rpc-session-seal.v0.2'] = GUEST_RPC_SESSION_SEAL_SCHEMA_VERSION
    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_protocol: AgenticResponseProtocol
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    rpc_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_broker_contract_version: str = Field(min_length=1)
    workspace_broker_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_capability_id: str = Field(pattern=_SHA256_PATTERN)
    gateway_grant_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_peer_cid: int = Field(ge=3, le=2**32 - 1)
    observed_peer_cid: int = Field(ge=3, le=2**32 - 1)
    rpc_port: int = Field(ge=1, le=2**32 - 1)
    receipt_authentication: Literal['hmac-sha256-domain-separated'] = GUEST_RPC_SESSION_AUTHENTICATION
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    attempt_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    projected_tool_events_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_count: int = Field(ge=0)
    accepted_sequence_count: int = Field(ge=0)
    projected_tool_event_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    exact_replay_count: int = Field(ge=0)
    wire_bytes: int = Field(ge=0)
    terminal_status: GuestRpcTerminalStatus
    terminal_error_code: GuestRpcErrorCode | None = None
    submit_attempted: bool
    submit_accepted: bool
    final_submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    final_submission_bytes: int = Field(ge=0)
    started_at: datetime
    sealed_at: datetime

    @field_validator('started_at', 'sealed_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_terminal(self) -> Self:
        if self.sealed_at < self.started_at:
            raise ValueError('RPC session cannot be sealed before it starts')
        if self.expected_peer_cid != self.observed_peer_cid:
            raise ValueError('RPC session peer CID does not match the pinned guest CID')
        if self.accepted_sequence_count > self.attempt_count:
            raise ValueError('accepted RPC sequence count cannot exceed attempt count')
        if self.projected_tool_event_count > self.attempt_count:
            raise ValueError('projected tool event count cannot exceed attempt count')
        if self.terminal_status == GuestRpcTerminalStatus.COMPLETED:
            if self.terminal_error_code is not None or not self.submit_attempted or not self.submit_accepted:
                raise ValueError('completed RPC sessions require exactly one accepted final submission')
            if self.final_submission_bytes == 0:
                raise ValueError('completed RPC sessions require nonempty final submission bytes')
        else:
            if self.terminal_error_code is None or self.submit_accepted:
                raise ValueError('failed or aborted RPC sessions require a terminal error and no accepted submission')
            if self.final_submission_bytes != 0 or self.final_submission_sha256 != _EMPTY_SHA256:
                raise ValueError('unsuccessful RPC sessions cannot carry a final submission')
        return self


class AuthenticatedGuestRpcSession(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-guest-rpc-session.v0.2'] = (
        AUTHENTICATED_GUEST_RPC_SESSION_SCHEMA_VERSION
    )
    policy: GuestRpcPolicy
    gateway_grant: GatewayCapabilityGrant
    task_invocation: AgenticTaskInvocation
    attempts: tuple[GuestRpcAttempt, ...]
    projected_tool_events: tuple[AgenticToolEvent, ...]
    submission: AgenticRuntimeSubmission | None = None
    seal: GuestRpcSessionSeal
    seal_hmac: str = Field(pattern=_SHA256_PATTERN)


class _DispatchFailure(RuntimeError):
    def __init__(self, code: GuestRpcErrorCode, *, fatal: bool):
        super().__init__(code.value)
        self.code = code
        self.fatal = fatal


def guest_rpc_policy_sha256(policy: GuestRpcPolicy) -> str:
    return _sha256(canonical_json_bytes(policy))


def guest_rpc_session_key_id(key: bytes) -> str:
    _validate_receipt_key(key)
    return _sha256(_SEAL_KEY_ID_DOMAIN + key)


def guest_rpc_session_seal_hmac(seal: GuestRpcSessionSeal, key: bytes) -> str:
    _validate_receipt_key(key)
    return hmac.new(key, _SEAL_HMAC_DOMAIN + canonical_json_bytes(seal), hashlib.sha256).hexdigest()


def encode_guest_rpc_frame(payload: BaseModel, *, maximum_body_bytes: int) -> bytes:
    _validate_maximum_body(maximum_body_bytes)
    body = canonical_json_bytes(payload)
    if not body or len(body) > maximum_body_bytes:
        raise GuestRpcError('RPC frame body exceeds its byte limit')
    return _FRAME_HEADER.pack(len(body)) + body


def decode_guest_rpc_frame(
    frame: bytes,
    model_type: type[ModelT],
    *,
    maximum_body_bytes: int,
) -> tuple[ModelT, bytes]:
    _validate_maximum_body(maximum_body_bytes)
    if len(frame) < _FRAME_HEADER.size + 1:
        raise GuestRpcError('RPC frame is truncated')
    (body_length,) = _FRAME_HEADER.unpack(frame[: _FRAME_HEADER.size])
    if body_length == 0 or body_length > maximum_body_bytes:
        raise GuestRpcError('RPC frame header is invalid')
    if len(frame) != _FRAME_HEADER.size + body_length:
        raise GuestRpcError('RPC frame has truncated or trailing bytes')
    body = frame[_FRAME_HEADER.size :]
    try:
        value = model_type.model_validate_json(body)
    except ValidationError as error:
        raise GuestRpcError('RPC frame body does not match its strict schema') from error
    if canonical_json_bytes(value) != body:
        raise GuestRpcError('RPC frame body must use canonical JSON')
    return value, body


def receive_guest_rpc_frame(connection: socket.socket, *, maximum_body_bytes: int) -> bytes:
    _validate_maximum_body(maximum_body_bytes)
    header = _receive_exact(connection, _FRAME_HEADER.size)
    (body_length,) = _FRAME_HEADER.unpack(header)
    if body_length == 0 or body_length > maximum_body_bytes:
        raise GuestRpcError('RPC frame header is invalid')
    return header + _receive_exact(connection, body_length)


def send_guest_rpc_frame(connection: socket.socket, frame: bytes) -> None:
    view = memoryview(frame)
    while view:
        sent = connection.send(view)
        if sent <= 0:
            raise GuestRpcError('RPC connection closed during frame write')
        view = view[sent:]


class GuestRpcHostSession:
    """Single-use host-side RPC state machine for one microVM guest CID."""

    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        workspace_manifest_sha256: str,
        workspace_tree_sha256: str,
        model_visible_surface_sha256: str,
        task_invocation: AgenticTaskInvocation,
        expected_response_protocol: AgenticResponseProtocol,
        worker_spec_sha256: str,
        execution_policy_sha256: str,
        broker: AgenticLogicalWorkspaceBroker,
        gateway: AuthenticatedProviderGateway,
        gateway_grant: GatewayCapabilityGrant,
        gateway_secret: bytes,
        observed_peer_cid: int,
        rpc_port: int,
        policy: GuestRpcPolicy,
        receipt_key: bytes,
        expected_receipt_key_id: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        _validate_hex(session_id, 32, 'RPC session ID')
        _validate_hex(run_id, 32, 'run ID')
        for value, label in (
            (workspace_manifest_sha256, 'workspace manifest SHA-256'),
            (workspace_tree_sha256, 'workspace tree SHA-256'),
            (model_visible_surface_sha256, 'model-visible surface SHA-256'),
            (worker_spec_sha256, 'worker spec SHA-256'),
            (execution_policy_sha256, 'execution policy SHA-256'),
        ):
            _validate_hex(value, 64, label)
        _validate_receipt_key(receipt_key)
        if guest_rpc_session_key_id(receipt_key) != expected_receipt_key_id:
            raise ValueError('guest RPC receipt key does not match the pinned key ID')
        if gateway_grant.run_id != run_id:
            raise ValueError('gateway grant is bound to a different run')
        if gateway_grant.workspace_manifest_sha256 != workspace_manifest_sha256:
            raise ValueError('gateway grant is bound to a different workspace')
        task_invocation = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(task_invocation))
        if task_invocation.workspace_manifest_sha256 != workspace_manifest_sha256:
            raise ValueError('task invocation is bound to a different workspace')
        if task_invocation.response_protocol != expected_response_protocol:
            raise ValueError('task invocation response protocol differs from the execution policy')
        if gateway_grant.execution_policy_sha256 != execution_policy_sha256:
            raise ValueError('gateway grant is bound to a different execution policy')
        if gateway_grant.expected_peer_cid != observed_peer_cid:
            raise ValueError('observed RPC peer CID does not match the gateway grant')
        if not 1 <= rpc_port <= 2**32 - 1:
            raise ValueError('guest RPC port is outside the vsock port range')
        # build_gateway_request_frame also checks this before any provider call.  Validate now so a
        # miswired host cannot start a session which will fail only after the guest is running.
        if gateway_capability_id(gateway_secret) != gateway_grant.capability_id:
            raise ValueError('host-owned gateway secret does not match its capability grant')

        self.session_id = session_id
        self.run_id = run_id
        self.workspace_manifest_sha256 = workspace_manifest_sha256
        self.workspace_tree_sha256 = workspace_tree_sha256
        self.model_visible_surface_sha256 = model_visible_surface_sha256
        self.task_invocation = task_invocation
        self.worker_spec_sha256 = worker_spec_sha256
        self.execution_policy_sha256 = execution_policy_sha256
        self.broker = broker
        self.gateway = gateway
        self.gateway_grant = gateway_grant
        self._gateway_secret = bytes(gateway_secret)
        self.observed_peer_cid = observed_peer_cid
        self.rpc_port = rpc_port
        self.policy = policy
        self._receipt_key = bytes(receipt_key)
        self._receipt_key_id = expected_receipt_key_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self._started_at = self._now()
        self._lock = threading.Lock()
        self._attempts: list[GuestRpcAttempt] = []
        self._tool_events: list[AgenticToolEvent] = []
        self._expected_sequence = 0
        self._model_call_count = 0
        self._wire_bytes = 0
        self._exact_replay_count = 0
        self._last_request_body: bytes | None = None
        self._last_response_frame: bytes | None = None
        self._submission: AgenticRuntimeSubmission | None = None
        self._terminal_status: GuestRpcTerminalStatus | None = None
        self._terminal_error: GuestRpcErrorCode | None = None
        self._sealed_artifact: AuthenticatedGuestRpcSession | None = None

    @property
    def terminal(self) -> bool:
        return self._terminal_status is not None

    @property
    def attempts(self) -> tuple[GuestRpcAttempt, ...]:
        return tuple(self._attempts)

    @property
    def projected_tool_events(self) -> tuple[AgenticToolEvent, ...]:
        return tuple(self._tool_events)

    @property
    def final_submission_bytes(self) -> bytes:
        return b'' if self._submission is None else canonical_json_bytes(self._submission)

    def handle_frame(self, frame: bytes) -> bytes:
        """Handle one complete frame; exact last-request retries return the cached response."""

        # This lock is part of the protocol boundary, not just an implementation convenience:
        # even if a supervisor accidentally feeds two connections into one session, dispatch is
        # serialized and the sequence/replay checks below run before either second action executes.
        self._lock.acquire()
        try:
            try:
                request, request_body = decode_guest_rpc_frame(
                    frame,
                    GuestRpcRequest,
                    maximum_body_bytes=self.policy.maximum_frame_body_bytes,
                )
            except GuestRpcError:
                self._fail_unlogged(GuestRpcErrorCode.INVALID_BODY)
                raise

            if self._sealed_artifact is not None:
                return self._unrecorded_error_frame(request, GuestRpcErrorCode.TERMINAL)

            if request.sequence == self._expected_sequence - 1 and self._last_request_body is not None:
                if request_body == self._last_request_body and self._last_response_frame is not None:
                    replay_wire = len(frame) + len(self._last_response_frame)
                    if self._wire_bytes + replay_wire > self.policy.maximum_session_wire_bytes:
                        self._fail_unlogged(GuestRpcErrorCode.LIMIT_EXCEEDED)
                        raise GuestRpcError('RPC session wire budget exhausted during exact replay')
                    self._wire_bytes += replay_wire
                    self._exact_replay_count += 1
                    return self._last_response_frame
                if self.terminal:
                    # A conflicting retry cannot rewrite an already accepted final submission.
                    return self._unrecorded_error_frame(request, GuestRpcErrorCode.TERMINAL)
                return self._record_protocol_rejection(
                    request=request,
                    request_body=request_body,
                    frame_bytes=len(frame),
                    code=GuestRpcErrorCode.REPLAY_CONFLICT,
                )

            if self.terminal:
                return self._unrecorded_error_frame(request, GuestRpcErrorCode.TERMINAL)
            if request.session_id != self.session_id:
                return self._record_protocol_rejection(
                    request=request,
                    request_body=request_body,
                    frame_bytes=len(frame),
                    code=GuestRpcErrorCode.WRONG_SESSION,
                )
            if request.sequence != self._expected_sequence:
                return self._record_protocol_rejection(
                    request=request,
                    request_body=request_body,
                    frame_bytes=len(frame),
                    code=GuestRpcErrorCode.OUT_OF_ORDER,
                )
            started_at = self._now()
            try:
                method = GuestRpcMethod(request.method)
            except ValueError:
                return self._record_attempt(
                    request=request,
                    request_body=request_body,
                    frame_bytes=len(frame),
                    sequence_accepted=True,
                    started_at=started_at,
                    response=self._error_response(request, GuestRpcErrorCode.UNKNOWN_METHOD),
                    fatal=True,
                )

            if len(self._attempts) >= self.policy.maximum_requests:
                response = self._error_response(request, GuestRpcErrorCode.LIMIT_EXCEEDED)
                return self._record_attempt(
                    request=request,
                    request_body=request_body,
                    frame_bytes=len(frame),
                    sequence_accepted=True,
                    started_at=started_at,
                    response=response,
                    fatal=True,
                    tool_material=self._failed_tool_material(
                        method,
                        request.body,
                        response,
                        gateway_call_index=None,
                    ),
                )

            # MODEL_GENERATE can dispatch an irreversible provider call and SUBMIT can accept the
            # final answer.  Reserve enough room for the largest legal RPC response before either
            # action.  This deliberately favors a clean, authenticated limit failure over making
            # an action whose result cannot be recorded in the terminal artifact.
            if method in {GuestRpcMethod.MODEL_GENERATE, GuestRpcMethod.SUBMIT}:
                maximum_response_frame_bytes = _FRAME_HEADER.size + self.policy.maximum_frame_body_bytes
                if (
                    self._wire_bytes + len(frame) + maximum_response_frame_bytes
                    > self.policy.maximum_session_wire_bytes
                ):
                    response = self._error_response(request, GuestRpcErrorCode.LIMIT_EXCEEDED)
                    return self._record_attempt(
                        request=request,
                        request_body=request_body,
                        frame_bytes=len(frame),
                        sequence_accepted=True,
                        started_at=started_at,
                        response=response,
                        fatal=True,
                        tool_material=self._failed_tool_material(
                            method,
                            request.body,
                            response,
                            gateway_call_index=None,
                        ),
                    )

            model_call_count_before = self._model_call_count
            accepted_submission: AgenticRuntimeSubmission | None = None
            try:
                result, tool_material, accepted_submission = self._dispatch(
                    method,
                    request.body,
                    started_at=started_at,
                )
                response = GuestRpcResponse(
                    session_id=request.session_id,
                    sequence=request.sequence,
                    succeeded=True,
                    result=result.model_dump(mode='json'),
                )
                fatal = method == GuestRpcMethod.SUBMIT
            except _DispatchFailure as error:
                response = self._error_response(request, error.code)
                dispatched_call_index = (
                    model_call_count_before
                    if method == GuestRpcMethod.MODEL_GENERATE and self._model_call_count > model_call_count_before
                    else None
                )
                tool_material = self._failed_tool_material(
                    method,
                    request.body,
                    response,
                    gateway_call_index=dispatched_call_index,
                )
                fatal = error.fatal
            except BaseException:
                response = self._error_response(request, GuestRpcErrorCode.INTERNAL)
                dispatched_call_index = (
                    model_call_count_before
                    if method == GuestRpcMethod.MODEL_GENERATE and self._model_call_count > model_call_count_before
                    else None
                )
                tool_material = self._failed_tool_material(
                    method,
                    request.body,
                    response,
                    gateway_call_index=dispatched_call_index,
                )
                fatal = True

            return self._record_attempt(
                request=request,
                request_body=request_body,
                frame_bytes=len(frame),
                sequence_accepted=True,
                started_at=started_at,
                response=response,
                fatal=fatal,
                tool_material=tool_material,
                completed=method == GuestRpcMethod.SUBMIT and response.succeeded,
                accepted_submission=accepted_submission,
            )
        finally:
            self._lock.release()

    def abort(self, code: GuestRpcErrorCode = GuestRpcErrorCode.CONNECTION_CLOSED) -> None:
        """Terminate an incomplete connection without accepting a submission."""

        if code == GuestRpcErrorCode.TERMINAL:
            raise ValueError('terminal is a response code, not an abort reason')
        with self._lock:
            if not self.terminal:
                self._terminal_status = GuestRpcTerminalStatus.ABORTED
                self._terminal_error = code

    def seal(self, *, sealed_at: datetime | None = None) -> AuthenticatedGuestRpcSession:
        """Return one immutable host-authenticated terminal artifact."""

        with self._lock:
            if self._sealed_artifact is not None:
                return self._sealed_artifact
            if not self.terminal or self._terminal_status is None:
                raise ValueError('cannot seal a nonterminal guest RPC session')
            final_bytes = self.final_submission_bytes
            seal = GuestRpcSessionSeal(
                session_id=self.session_id,
                run_id=self.run_id,
                attempt_reservation_sha256=self.gateway_grant.attempt_reservation_sha256,
                execution_policy_sha256=self.execution_policy_sha256,
                workspace_manifest_sha256=self.workspace_manifest_sha256,
                workspace_tree_sha256=self.workspace_tree_sha256,
                model_visible_surface_sha256=self.model_visible_surface_sha256,
                task_invocation_sha256=agentic_task_invocation_sha256(self.task_invocation),
                response_protocol=self.task_invocation.response_protocol,
                worker_spec_sha256=self.worker_spec_sha256,
                rpc_policy_sha256=guest_rpc_policy_sha256(self.policy),
                workspace_broker_contract_version=self.broker.contract_version,
                workspace_broker_contract_sha256=self.broker.contract_sha256,
                gateway_capability_id=self.gateway_grant.capability_id,
                gateway_grant_sha256=gateway_capability_grant_sha256(self.gateway_grant),
                expected_peer_cid=self.gateway_grant.expected_peer_cid,
                observed_peer_cid=self.observed_peer_cid,
                rpc_port=self.rpc_port,
                receipt_key_id=self._receipt_key_id,
                attempt_log_sha256=_sha256(
                    canonical_json_bytes([item.model_dump(mode='json') for item in self._attempts])
                ),
                projected_tool_events_sha256=_sha256(
                    canonical_json_bytes([item.model_dump(mode='json') for item in self._tool_events])
                ),
                attempt_count=len(self._attempts),
                accepted_sequence_count=self._expected_sequence,
                projected_tool_event_count=len(self._tool_events),
                model_call_count=self._model_call_count,
                exact_replay_count=self._exact_replay_count,
                wire_bytes=self._wire_bytes,
                terminal_status=self._terminal_status,
                terminal_error_code=self._terminal_error,
                submit_attempted=any(item.request.method == GuestRpcMethod.SUBMIT.value for item in self._attempts),
                submit_accepted=self._submission is not None,
                final_submission_sha256=_sha256(final_bytes),
                final_submission_bytes=len(final_bytes),
                started_at=self._started_at,
                sealed_at=self._aware(sealed_at or self._now(), 'sealed_at'),
            )
            artifact = AuthenticatedGuestRpcSession(
                policy=self.policy,
                gateway_grant=self.gateway_grant,
                task_invocation=self.task_invocation,
                attempts=tuple(self._attempts),
                projected_tool_events=tuple(self._tool_events),
                submission=self._submission,
                seal=seal,
                seal_hmac=guest_rpc_session_seal_hmac(seal, self._receipt_key),
            )
            verify_authenticated_guest_rpc_session(
                artifact,
                receipt_key=self._receipt_key,
                expected_receipt_key_id=self._receipt_key_id,
                expected_run_id=self.run_id,
                expected_workspace_manifest_sha256=self.workspace_manifest_sha256,
                expected_execution_policy_sha256=self.execution_policy_sha256,
                expected_task_invocation_sha256=agentic_task_invocation_sha256(self.task_invocation),
                expected_response_protocol=self.task_invocation.response_protocol,
                expected_peer_cid=self.observed_peer_cid,
                expected_rpc_port=self.rpc_port,
            )
            self._sealed_artifact = artifact
            self._gateway_secret = b''
            self._receipt_key = b''
            return artifact

    def _dispatch(
        self,
        method: GuestRpcMethod,
        body: dict[str, JsonValue],
        *,
        started_at: datetime,
    ) -> tuple[
        BaseModel,
        tuple[AgenticTool, int | None, bytes, bytes] | None,
        AgenticRuntimeSubmission | None,
    ]:
        if method == GuestRpcMethod.LIST:
            request = _body_model(ListWorkspaceRequest, body)
            if request.limit > self.policy.maximum_list_entries:
                raise _DispatchFailure(GuestRpcErrorCode.INVALID_BODY, fatal=False)
            files = self.broker.list_files()
            if request.cursor > len(files):
                raise _DispatchFailure(GuestRpcErrorCode.WORKSPACE_REJECTED, fatal=False)
            selected = files[request.cursor : request.cursor + request.limit]
            next_cursor = request.cursor + len(selected)
            result = ListWorkspaceResult(
                files=tuple(
                    LogicalFileResult(
                        path=item.path,
                        media_type=item.media_type.value,
                        sha256=item.sha256,
                        byte_count=item.byte_count,
                    )
                    for item in selected
                ),
                next_cursor=next_cursor if next_cursor < len(files) else None,
            )
            return result, _tool_material(AgenticTool.LIST_WORKSPACE, None, request, result), None

        if method == GuestRpcMethod.READ:
            request = _body_model(ReadWorkspaceRequest, body)
            if request.limit > self.policy.maximum_read_bytes:
                raise _DispatchFailure(GuestRpcErrorCode.INVALID_BODY, fatal=False)
            try:
                content = self.broker.read(request.path, offset=request.offset, limit=request.limit)
                metadata = {item.path: item for item in self.broker.list_files()}[request.path]
            except (AgenticWorkspaceError, KeyError):
                raise _DispatchFailure(GuestRpcErrorCode.WORKSPACE_REJECTED, fatal=False) from None
            result = ReadWorkspaceResult(
                content_base64=base64.b64encode(content).decode('ascii'),
                offset=request.offset,
                byte_count=len(content),
                eof=request.offset + len(content) >= metadata.byte_count,
            )
            return result, _tool_material(AgenticTool.READ_WORKSPACE, None, request, result), None

        if method == GuestRpcMethod.SEARCH:
            request = _body_model(SearchWorkspaceRequest, body)
            if request.max_results > self.policy.maximum_search_results:
                raise _DispatchFailure(GuestRpcErrorCode.INVALID_BODY, fatal=False)
            try:
                hits = self.broker.search(
                    request.needle,
                    paths=request.paths,
                    max_results=request.max_results,
                )
            except AgenticWorkspaceError:
                raise _DispatchFailure(GuestRpcErrorCode.WORKSPACE_REJECTED, fatal=False) from None
            result = SearchWorkspaceResult(
                hits=tuple(
                    LogicalSearchHitResult(
                        path=hit.path,
                        start_byte=hit.start_byte,
                        end_byte=hit.end_byte,
                    )
                    for hit in hits
                )
            )
            return result, _tool_material(AgenticTool.SEARCH_WORKSPACE, None, request, result), None

        if method == GuestRpcMethod.MODEL_GENERATE:
            request = _body_model(ModelGenerateRequest, body)
            call_index = self._model_call_count
            model_request = AgenticModelRequest(
                run_id=self.run_id,
                call_index=call_index,
                messages=request.messages,
                max_output_tokens=request.max_output_tokens,
                response_schema_sha256=request.response_schema_sha256,
            )
            self._model_call_count += 1
            try:
                gateway_request = build_gateway_request_frame(
                    self.gateway_grant,
                    model_request,
                    secret=self._gateway_secret,
                    maximum_body_bytes=self.gateway.policy.maximum_frame_body_bytes,
                )
                gateway_response_frame = self.gateway.handle_frame(
                    gateway_request,
                    peer_cid=self.observed_peer_cid,
                    observed_at=started_at,
                )
                wire_response = parse_gateway_response_frame(
                    gateway_response_frame,
                    self.gateway_grant,
                    secret=self._gateway_secret,
                    maximum_body_bytes=self.gateway.policy.maximum_frame_body_bytes,
                )
            except BaseException:
                raise _DispatchFailure(GuestRpcErrorCode.GATEWAY_REJECTED, fatal=True) from None
            if not wire_response.succeeded or wire_response.response is None:
                raise _DispatchFailure(GuestRpcErrorCode.GATEWAY_REJECTED, fatal=True)
            result = ModelGenerateResult(response=wire_response.response)
            return (
                result,
                _tool_material(
                    AgenticTool.MODEL_GENERATE,
                    call_index,
                    model_request,
                    wire_response.response,
                ),
                None,
            )

        if method == GuestRpcMethod.SUBMIT:
            try:
                request = _body_model(SubmitRequest, body)
                validate_submission_for_invocation(self.task_invocation, request.submission)
            except (_DispatchFailure, ValueError):
                raise _DispatchFailure(GuestRpcErrorCode.SUBMISSION_REJECTED, fatal=True) from None
            submission_bytes = canonical_json_bytes(request.submission)
            if not submission_bytes or len(submission_bytes) > self.policy.maximum_submission_bytes:
                raise _DispatchFailure(GuestRpcErrorCode.SUBMISSION_REJECTED, fatal=True)
            return (
                SubmitResult(
                    submission_sha256=_sha256(submission_bytes),
                    submission_bytes=len(submission_bytes),
                ),
                None,
                request.submission,
            )

        raise AssertionError('closed RPC method dispatch became nonexhaustive')

    def _failed_tool_material(
        self,
        method: GuestRpcMethod,
        body: dict[str, JsonValue],
        response: GuestRpcResponse,
        *,
        gateway_call_index: int | None,
    ) -> tuple[AgenticTool, int | None, bytes, bytes] | None:
        mapping = {
            GuestRpcMethod.LIST: AgenticTool.LIST_WORKSPACE,
            GuestRpcMethod.READ: AgenticTool.READ_WORKSPACE,
            GuestRpcMethod.SEARCH: AgenticTool.SEARCH_WORKSPACE,
            GuestRpcMethod.MODEL_GENERATE: AgenticTool.MODEL_GENERATE,
        }
        tool = mapping.get(method)
        if tool is None:
            return None
        # Failed model attempts cannot be projected into AgenticToolEvent v0.1.  These exact hashes
        # remain in GuestRpcAttempt and the authenticated terminal attempt log.
        request_bytes = canonical_json_bytes(body)
        response_bytes = canonical_json_bytes(response)
        return tool, gateway_call_index, request_bytes, response_bytes

    def _record_protocol_rejection(
        self,
        *,
        request: GuestRpcRequest,
        request_body: bytes,
        frame_bytes: int,
        code: GuestRpcErrorCode,
    ) -> bytes:
        return self._record_attempt(
            request=request,
            request_body=request_body,
            frame_bytes=frame_bytes,
            sequence_accepted=False,
            started_at=self._now(),
            response=self._error_response(request, code),
            fatal=True,
        )

    def _record_attempt(
        self,
        *,
        request: GuestRpcRequest,
        request_body: bytes,
        frame_bytes: int,
        sequence_accepted: bool,
        started_at: datetime,
        response: GuestRpcResponse,
        fatal: bool,
        tool_material: tuple[AgenticTool, int | None, bytes, bytes] | None = None,
        completed: bool = False,
        accepted_submission: AgenticRuntimeSubmission | None = None,
    ) -> bytes:
        try:
            response_frame = encode_guest_rpc_frame(
                response,
                maximum_body_bytes=self.policy.maximum_frame_body_bytes,
            )
        except GuestRpcError:
            response = self._error_response(request, GuestRpcErrorCode.LIMIT_EXCEEDED)
            response_frame = encode_guest_rpc_frame(
                response,
                maximum_body_bytes=self.policy.maximum_frame_body_bytes,
            )
            fatal = True
            completed = False
            accepted_submission = None
            tool_material = self._failed_tool_material_for_existing(
                tool_material,
                request.body,
                response,
            )
        next_wire = self._wire_bytes + frame_bytes + len(response_frame)
        if next_wire > self.policy.maximum_session_wire_bytes:
            self._fail_unlogged(GuestRpcErrorCode.LIMIT_EXCEEDED)
            raise GuestRpcError('RPC session wire budget exhausted')

        finished_at = self._now()
        response_body = response_frame[_FRAME_HEADER.size :]
        tool: AgenticTool | None = None
        gateway_call_index: int | None = None
        tool_request_sha256: str | None = None
        tool_request_bytes: int | None = None
        tool_response_sha256: str | None = None
        tool_response_bytes: int | None = None
        projected_event_index: int | None = None
        if tool_material is not None:
            tool, gateway_call_index, tool_request, tool_response = tool_material
            tool_request_sha256 = _sha256(tool_request)
            tool_request_bytes = len(tool_request)
            tool_response_sha256 = _sha256(tool_response)
            tool_response_bytes = len(tool_response)
            # AgenticToolEvent v0.1 cannot represent a failed model attempt.  All other tool
            # attempts can be projected without losing their exact RPC receipt.
            if tool != AgenticTool.MODEL_GENERATE or response.succeeded:
                projected_event_index = len(self._tool_events)
                self._tool_events.append(
                    AgenticToolEvent(
                        event_index=projected_event_index,
                        tool=tool,
                        gateway_call_index=gateway_call_index,
                        started_at=started_at,
                        finished_at=finished_at,
                        request_sha256=tool_request_sha256,
                        request_bytes=tool_request_bytes,
                        response_sha256=tool_response_sha256,
                        response_bytes=tool_response_bytes,
                        succeeded=response.succeeded,
                    )
                )

        attempt = GuestRpcAttempt(
            attempt_index=len(self._attempts),
            sequence_accepted=sequence_accepted,
            request=request,
            response=response,
            started_at=started_at,
            finished_at=finished_at,
            request_sha256=_sha256(request_body),
            request_bytes=len(request_body),
            response_sha256=_sha256(response_body),
            response_bytes=len(response_body),
            tool=tool,
            gateway_call_index=gateway_call_index,
            tool_request_sha256=tool_request_sha256,
            tool_request_bytes=tool_request_bytes,
            tool_response_sha256=tool_response_sha256,
            tool_response_bytes=tool_response_bytes,
            projected_tool_event_index=projected_event_index,
        )
        self._attempts.append(attempt)
        self._wire_bytes = next_wire
        if sequence_accepted:
            self._expected_sequence += 1
            self._last_request_body = request_body
            self._last_response_frame = response_frame
        if completed:
            if accepted_submission is None:
                raise AssertionError('completed submit RPC is missing its staged submission')
            self._submission = accepted_submission
            self._terminal_status = GuestRpcTerminalStatus.COMPLETED
            self._terminal_error = None
        elif fatal:
            self._terminal_status = GuestRpcTerminalStatus.FAILED
            self._terminal_error = response.error_code or GuestRpcErrorCode.INTERNAL
        return response_frame

    def _failed_tool_material_for_existing(
        self,
        material: tuple[AgenticTool, int | None, bytes, bytes] | None,
        request_body: dict[str, JsonValue],
        response: GuestRpcResponse,
    ) -> tuple[AgenticTool, int | None, bytes, bytes] | None:
        if material is None:
            return None
        tool, call_index, _, _ = material
        return tool, call_index, canonical_json_bytes(request_body), canonical_json_bytes(response)

    def _unrecorded_error_frame(self, request: GuestRpcRequest, code: GuestRpcErrorCode) -> bytes:
        return encode_guest_rpc_frame(
            self._error_response(request, code),
            maximum_body_bytes=self.policy.maximum_frame_body_bytes,
        )

    def _error_response(self, request: GuestRpcRequest, code: GuestRpcErrorCode) -> GuestRpcResponse:
        return GuestRpcResponse(
            session_id=request.session_id,
            sequence=request.sequence,
            succeeded=False,
            error_code=code,
            error_message='rpc request rejected',
        )

    def _fail_unlogged(self, code: GuestRpcErrorCode) -> None:
        if not self.terminal:
            self._terminal_status = GuestRpcTerminalStatus.FAILED
            self._terminal_error = code

    def _now(self) -> datetime:
        return self._aware(self._clock(), 'host clock')

    @staticmethod
    def _aware(value: datetime, field_name: str) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{field_name} must include a UTC offset')
        return value.astimezone(UTC)


class GuestRpcHostServer:
    """Sequential server loop over a preconnected Firecracker-vsock socket."""

    def __init__(self, session: GuestRpcHostSession) -> None:
        self.session = session

    def serve(self, connection: socket.socket) -> None:
        try:
            while not self.session.terminal:
                frame = receive_guest_rpc_frame(
                    connection,
                    maximum_body_bytes=self.session.policy.maximum_frame_body_bytes,
                )
                response = self.session.handle_frame(frame)
                send_guest_rpc_frame(connection, response)
        except (GuestRpcError, OSError):
            self.session.abort(GuestRpcErrorCode.CONNECTION_CLOSED)


class GuestRpcClient:
    """Tiny guest client; it exposes no general network, filesystem, or shell operation."""

    def __init__(
        self,
        connection: socket.socket,
        *,
        session_id: str,
        task_invocation: AgenticTaskInvocation,
        maximum_body_bytes: int = DEFAULT_MAX_GUEST_RPC_BODY_BYTES,
    ) -> None:
        _validate_hex(session_id, 32, 'RPC session ID')
        _validate_maximum_body(maximum_body_bytes)
        self._connection = connection
        self._session_id = session_id
        self._task_invocation = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(task_invocation))
        self._maximum_body_bytes = maximum_body_bytes
        self._sequence = 0
        self._terminal = False
        self._lock = threading.Lock()

    def list_workspace(self, *, cursor: int = 0, limit: int = 100) -> ListWorkspaceResult:
        return self._exchange(
            GuestRpcMethod.LIST, ListWorkspaceRequest(cursor=cursor, limit=limit), ListWorkspaceResult
        )

    def read_workspace(self, path: str, *, offset: int = 0, limit: int) -> ReadWorkspaceResult:
        return self._exchange(
            GuestRpcMethod.READ,
            ReadWorkspaceRequest(path=path, offset=offset, limit=limit),
            ReadWorkspaceResult,
        )

    def search_workspace(
        self,
        needle: str,
        *,
        paths: tuple[str, ...] | None = None,
        max_results: int = 100,
    ) -> SearchWorkspaceResult:
        return self._exchange(
            GuestRpcMethod.SEARCH,
            SearchWorkspaceRequest(needle=needle, paths=paths, max_results=max_results),
            SearchWorkspaceResult,
        )

    def model_generate(
        self,
        *,
        messages: tuple[AgenticModelMessage, ...],
        max_output_tokens: int,
        response_schema_sha256: str | None = None,
    ) -> AgenticModelResponse:
        result = self._exchange(
            GuestRpcMethod.MODEL_GENERATE,
            ModelGenerateRequest(
                messages=messages,
                max_output_tokens=max_output_tokens,
                response_schema_sha256=response_schema_sha256,
            ),
            ModelGenerateResult,
        )
        return result.response

    def submit(self, submission: AgenticRuntimeSubmission) -> SubmitResult:
        try:
            validate_submission_for_invocation(self._task_invocation, submission)
        except ValueError as error:
            raise GuestRpcError('submission does not match the task invocation') from error
        result = self._exchange(GuestRpcMethod.SUBMIT, SubmitRequest(submission=submission), SubmitResult)
        self._terminal = True
        return result

    def _exchange(self, method: GuestRpcMethod, body: BaseModel, result_type: type[ModelT]) -> ModelT:
        with self._lock:
            if self._terminal:
                raise GuestRpcError('guest RPC client is terminal')
            request = GuestRpcRequest(
                session_id=self._session_id,
                sequence=self._sequence,
                method=method.value,
                body=body.model_dump(mode='json'),
            )
            frame = encode_guest_rpc_frame(request, maximum_body_bytes=self._maximum_body_bytes)
            send_guest_rpc_frame(self._connection, frame)
            response_frame = receive_guest_rpc_frame(
                self._connection,
                maximum_body_bytes=self._maximum_body_bytes,
            )
            response, _ = decode_guest_rpc_frame(
                response_frame,
                GuestRpcResponse,
                maximum_body_bytes=self._maximum_body_bytes,
            )
            if (response.session_id, response.sequence) != (self._session_id, self._sequence):
                self._terminal = True
                raise GuestRpcError('RPC response does not match its request')
            self._sequence += 1
            if not response.succeeded or response.result is None:
                self._terminal = True
                raise GuestRpcRemoteError(response.error_code or GuestRpcErrorCode.INTERNAL)
            try:
                result = result_type.model_validate_json(canonical_json_bytes(response.result))
            except ValidationError as error:
                self._terminal = True
                raise GuestRpcError('RPC response result does not match its strict method schema') from error
            if canonical_json_bytes(result) != canonical_json_bytes(response.result):
                self._terminal = True
                raise GuestRpcError('RPC response result is not canonical for its method schema')
            return result


def verify_authenticated_guest_rpc_session(
    artifact: AuthenticatedGuestRpcSession,
    *,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    expected_run_id: str,
    expected_workspace_manifest_sha256: str,
    expected_execution_policy_sha256: str,
    expected_task_invocation_sha256: str,
    expected_response_protocol: AgenticResponseProtocol,
    expected_peer_cid: int,
    expected_rpc_port: int,
) -> None:
    """Authenticate and fully rederive a terminal RPC session artifact."""

    if guest_rpc_session_key_id(receipt_key) != expected_receipt_key_id:
        raise ValueError('guest RPC receipt key does not match the pinned key ID')
    if artifact.seal.receipt_key_id != expected_receipt_key_id:
        raise ValueError('guest RPC session uses an unexpected receipt key')
    expected_hmac = guest_rpc_session_seal_hmac(artifact.seal, receipt_key)
    if not hmac.compare_digest(artifact.seal_hmac, expected_hmac):
        raise ValueError('guest RPC session authentication failed')
    seal = artifact.seal
    observed_invocation_sha256 = agentic_task_invocation_sha256(artifact.task_invocation)
    if (
        seal.run_id,
        seal.workspace_manifest_sha256,
        seal.execution_policy_sha256,
        seal.task_invocation_sha256,
        seal.response_protocol,
        seal.expected_peer_cid,
        seal.observed_peer_cid,
        seal.rpc_port,
    ) != (
        expected_run_id,
        expected_workspace_manifest_sha256,
        expected_execution_policy_sha256,
        expected_task_invocation_sha256,
        expected_response_protocol,
        expected_peer_cid,
        expected_peer_cid,
        expected_rpc_port,
    ):
        raise ValueError('guest RPC session does not match the expected run boundary')
    if (
        observed_invocation_sha256 != seal.task_invocation_sha256
        or artifact.task_invocation.workspace_manifest_sha256 != seal.workspace_manifest_sha256
        or artifact.task_invocation.response_protocol != seal.response_protocol
    ):
        raise ValueError('guest RPC session does not bind its exact task invocation')
    grant = artifact.gateway_grant
    if (
        grant.run_id,
        grant.attempt_reservation_sha256,
        grant.execution_policy_sha256,
        grant.workspace_manifest_sha256,
        grant.capability_id,
        grant.expected_peer_cid,
        gateway_capability_grant_sha256(grant),
    ) != (
        seal.run_id,
        seal.attempt_reservation_sha256,
        seal.execution_policy_sha256,
        seal.workspace_manifest_sha256,
        seal.gateway_capability_id,
        seal.expected_peer_cid,
        seal.gateway_grant_sha256,
    ):
        raise ValueError('guest RPC session does not bind its gateway capability grant')
    if seal.rpc_policy_sha256 != guest_rpc_policy_sha256(artifact.policy):
        raise ValueError('guest RPC session does not bind its RPC policy')
    attempt_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in artifact.attempts])
    tool_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in artifact.projected_tool_events])
    if seal.attempt_log_sha256 != _sha256(attempt_bytes):
        raise ValueError('guest RPC session does not bind its exact attempt log')
    if seal.projected_tool_events_sha256 != _sha256(tool_bytes):
        raise ValueError('guest RPC session does not bind its projected tool events')
    if (seal.attempt_count, seal.projected_tool_event_count) != (
        len(artifact.attempts),
        len(artifact.projected_tool_events),
    ):
        raise ValueError('guest RPC session counts do not match its evidence')
    if tuple(item.attempt_index for item in artifact.attempts) != tuple(range(len(artifact.attempts))):
        raise ValueError('guest RPC attempt indexes must be contiguous')
    if tuple(item.event_index for item in artifact.projected_tool_events) != tuple(
        range(len(artifact.projected_tool_events))
    ):
        raise ValueError('guest RPC projected tool event indexes must be contiguous')

    next_sequence = 0
    projected: list[AgenticToolEvent] = []
    model_call_indexes: list[int] = []
    for attempt in artifact.attempts:
        if attempt.started_at < seal.started_at or attempt.finished_at > seal.sealed_at:
            raise ValueError('guest RPC attempt timestamp is outside the authenticated session')
        if attempt.sequence_accepted:
            if attempt.request.session_id != seal.session_id or attempt.request.sequence != next_sequence:
                raise ValueError('guest RPC accepted request sequences must be contiguous')
            next_sequence += 1
        elif attempt.response.error_code not in {
            GuestRpcErrorCode.WRONG_SESSION,
            GuestRpcErrorCode.OUT_OF_ORDER,
            GuestRpcErrorCode.REPLAY_CONFLICT,
        }:
            raise ValueError('only protocol identity/order faults may be outside the accepted sequence')
        if attempt.projected_tool_event_index is not None:
            if (
                attempt.tool is None
                or attempt.tool_request_sha256 is None
                or attempt.tool_request_bytes is None
                or attempt.tool_response_sha256 is None
                or attempt.tool_response_bytes is None
            ):
                raise ValueError('projected guest RPC event is missing exact tool evidence')
            projected.append(
                AgenticToolEvent(
                    event_index=attempt.projected_tool_event_index,
                    tool=attempt.tool,
                    gateway_call_index=attempt.gateway_call_index,
                    started_at=attempt.started_at,
                    finished_at=attempt.finished_at,
                    request_sha256=attempt.tool_request_sha256,
                    request_bytes=attempt.tool_request_bytes,
                    response_sha256=attempt.tool_response_sha256,
                    response_bytes=attempt.tool_response_bytes,
                    succeeded=attempt.response.succeeded,
                )
            )
        if attempt.tool == AgenticTool.MODEL_GENERATE and attempt.gateway_call_index is not None:
            model_call_indexes.append(attempt.gateway_call_index)
        _verify_attempt_method_binding(attempt)
        _verify_tool_material(attempt, seal.run_id)
    if next_sequence != seal.accepted_sequence_count:
        raise ValueError('guest RPC accepted sequence count does not match its attempts')
    if tuple(model_call_indexes) != tuple(range(len(model_call_indexes))):
        raise ValueError('guest RPC gateway call indexes must be contiguous')
    if tuple(projected) != artifact.projected_tool_events:
        raise ValueError('guest RPC projected tool events do not match authenticated attempts')
    if seal.model_call_count != len(model_call_indexes):
        raise ValueError('guest RPC model call count does not match authenticated attempts')

    submission_bytes = b'' if artifact.submission is None else canonical_json_bytes(artifact.submission)
    if (seal.final_submission_sha256, seal.final_submission_bytes) != (
        _sha256(submission_bytes),
        len(submission_bytes),
    ):
        raise ValueError('guest RPC session does not bind its exact final submission')
    successful_submits = [
        item
        for item in artifact.attempts
        if item.request.method == GuestRpcMethod.SUBMIT.value and item.response.succeeded
    ]
    submit_attempted = any(item.request.method == GuestRpcMethod.SUBMIT.value for item in artifact.attempts)
    if seal.submit_attempted != submit_attempted:
        raise ValueError('guest RPC submit-attempt flag does not match authenticated attempts')
    if seal.terminal_status == GuestRpcTerminalStatus.COMPLETED:
        if len(successful_submits) != 1 or successful_submits[0] is not artifact.attempts[-1]:
            raise ValueError('completed guest RPC session must end with exactly one successful submission')
        if artifact.submission is None:
            raise ValueError('completed guest RPC session is missing its final submission')
        try:
            validate_submission_for_invocation(artifact.task_invocation, artifact.submission)
        except ValueError as error:
            raise ValueError('authenticated final submission violates its task invocation') from error
        submit_attempt = successful_submits[0]
        try:
            submit_body = _body_model(SubmitRequest, submit_attempt.request.body)
            submit_result = _body_model(SubmitResult, submit_attempt.response.result or {})
        except _DispatchFailure as error:
            raise ValueError('successful submit RPC attempt has invalid method bodies') from error
        if submit_body.submission != artifact.submission:
            raise ValueError('authenticated final submission does not match the accepted submit request')
        if (submit_result.submission_sha256, submit_result.submission_bytes) != (
            seal.final_submission_sha256,
            seal.final_submission_bytes,
        ):
            raise ValueError('submit response does not match the authenticated final submission')
    elif artifact.submission is not None or successful_submits:
        raise ValueError('unsuccessful guest RPC session cannot contain an accepted submission')


def _verify_attempt_method_binding(attempt: GuestRpcAttempt) -> None:
    mapping = {
        GuestRpcMethod.LIST.value: AgenticTool.LIST_WORKSPACE,
        GuestRpcMethod.READ.value: AgenticTool.READ_WORKSPACE,
        GuestRpcMethod.SEARCH.value: AgenticTool.SEARCH_WORKSPACE,
        GuestRpcMethod.MODEL_GENERATE.value: AgenticTool.MODEL_GENERATE,
    }
    expected_tool = mapping.get(attempt.request.method) if attempt.sequence_accepted else None
    if attempt.tool != expected_tool:
        raise ValueError('guest RPC attempt tool does not match its closed method mapping')
    if expected_tool is None and attempt.request.method == GuestRpcMethod.SUBMIT.value and attempt.tool is not None:
        raise ValueError('submission cannot project a tool event')


def _verify_successful_model_tool_material(attempt: GuestRpcAttempt, run_id: str) -> None:
    if attempt.gateway_call_index is None:
        raise ValueError('successful model RPC attempt is missing its gateway call index')
    try:
        body = _body_model(ModelGenerateRequest, attempt.request.body)
        result = _body_model(ModelGenerateResult, attempt.response.result or {})
    except _DispatchFailure as error:
        raise ValueError('successful model RPC attempt has invalid method bodies') from error
    request = AgenticModelRequest(
        run_id=run_id,
        call_index=attempt.gateway_call_index,
        messages=body.messages,
        max_output_tokens=body.max_output_tokens,
        response_schema_sha256=body.response_schema_sha256,
    )
    request_bytes = canonical_json_bytes(request)
    response_bytes = canonical_json_bytes(result.response)
    if (
        attempt.tool_request_sha256,
        attempt.tool_request_bytes,
        attempt.tool_response_sha256,
        attempt.tool_response_bytes,
    ) != (_sha256(request_bytes), len(request_bytes), _sha256(response_bytes), len(response_bytes)):
        raise ValueError('model tool projection does not bind the inner gateway request and response')


def _verify_tool_material(attempt: GuestRpcAttempt, run_id: str) -> None:
    if attempt.tool is None:
        return
    if attempt.tool == AgenticTool.MODEL_GENERATE and attempt.response.succeeded:
        _verify_successful_model_tool_material(attempt, run_id)
        return
    if attempt.response.succeeded:
        method_models: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
            GuestRpcMethod.LIST.value: (ListWorkspaceRequest, ListWorkspaceResult),
            GuestRpcMethod.READ.value: (ReadWorkspaceRequest, ReadWorkspaceResult),
            GuestRpcMethod.SEARCH.value: (SearchWorkspaceRequest, SearchWorkspaceResult),
        }
        model_types = method_models.get(attempt.request.method)
        if model_types is None or attempt.response.result is None:
            raise ValueError('successful tool RPC attempt has no closed method schema')
        try:
            request = _body_model(model_types[0], attempt.request.body)
            response = _body_model(model_types[1], attempt.response.result)
        except _DispatchFailure as error:
            raise ValueError('successful tool RPC attempt has invalid method bodies') from error
        request_bytes = canonical_json_bytes(request)
        response_bytes = canonical_json_bytes(response)
    else:
        request_bytes = canonical_json_bytes(attempt.request.body)
        response_bytes = canonical_json_bytes(attempt.response)
    if (
        attempt.tool_request_sha256,
        attempt.tool_request_bytes,
        attempt.tool_response_sha256,
        attempt.tool_response_bytes,
    ) != (_sha256(request_bytes), len(request_bytes), _sha256(response_bytes), len(response_bytes)):
        raise ValueError('tool projection does not bind its exact method request and response')


def _tool_material(
    tool: AgenticTool,
    gateway_call_index: int | None,
    request: BaseModel,
    response: BaseModel,
) -> tuple[AgenticTool, int | None, bytes, bytes]:
    return tool, gateway_call_index, canonical_json_bytes(request), canonical_json_bytes(response)


def _body_model(model_type: type[ModelT], body: dict[str, JsonValue]) -> ModelT:
    encoded = canonical_json_bytes(body)
    try:
        value = model_type.model_validate_json(encoded)
    except ValidationError as error:
        raise _DispatchFailure(GuestRpcErrorCode.INVALID_BODY, fatal=True) from error
    if canonical_json_bytes(value) != encoded:
        raise _DispatchFailure(GuestRpcErrorCode.INVALID_BODY, fatal=True)
    return value


def _receive_exact(connection: socket.socket, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = connection.recv(byte_count - len(chunks))
        if not chunk:
            raise GuestRpcError('RPC connection closed during frame read')
        chunks.extend(chunk)
    return bytes(chunks)


def _validate_maximum_body(maximum_body_bytes: int) -> None:
    if maximum_body_bytes <= 0 or maximum_body_bytes > _MAX_RPC_BODY_BYTES:
        raise ValueError('RPC maximum frame body must be between 1 byte and 64 MiB')


def _validate_hex(value: str, length: int, label: str) -> None:
    if len(value) != length or any(character not in '0123456789abcdef' for character in value):
        raise ValueError(f'{label} must contain exactly {length} lowercase hexadecimal characters')


def _validate_receipt_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError('guest RPC receipt key must contain at least 32 bytes')


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
