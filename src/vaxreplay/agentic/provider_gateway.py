"""Replay-safe authenticated provider gateway with durable call reservations.

The hostile worker knows its short-lived capability secret, so capability-frame HMACs provide
channel integrity but are not organizer evidence.  A separate gateway receipt key authenticates a
closed session seal which the trusted run finalizer can verify.
"""

from __future__ import annotations

import enum
import fcntl
import hashlib
import hmac
import os
import sqlite3
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.gateway import (
    AgenticGatewayTranscript,
    AgenticModelCallReceipt,
    AgenticModelExchange,
    AgenticModelRequest,
    AgenticModelResponse,
)
from vaxreplay.agentic.gateway_auth import (
    DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES,
    GatewayFrameError,
    GatewaySecretResolver,
    RevocableGatewaySecretResolver,
    decode_gateway_frame,
    encode_gateway_frame,
    gateway_capability_id,
    maximum_gateway_frame_bytes,
    peek_gateway_frame_capability_id,
)
from vaxreplay.agentic.protocol import AgenticRunLimits
from vaxreplay.agentic.provider_adapter import (
    ProviderAdapter,
    ProviderAdapterDescriptor,
    ProviderCallFailure,
    ProviderCallResult,
    ProviderFailureCode,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

GATEWAY_MODEL_ROUTE_SCHEMA_VERSION = 'vaxreplay.gateway-model-route.v0.4'
AUTHENTICATED_GATEWAY_POLICY_SCHEMA_VERSION = 'vaxreplay.authenticated-gateway-policy.v0.2'
GATEWAY_CAPABILITY_GRANT_SCHEMA_VERSION = 'vaxreplay.gateway-capability-grant.v0.1'
GATEWAY_CAPABILITY_BINDING_SCHEMA_VERSION = 'vaxreplay.gateway-capability-binding.v0.1'
GATEWAY_CAPABILITY_REVOCATION_SCHEMA_VERSION = 'vaxreplay.gateway-capability-revocation.v0.1'
GATEWAY_LEDGER_IDENTITY_SCHEMA_VERSION = 'vaxreplay.gateway-ledger-identity.v0.1'
GATEWAY_WIRE_REQUEST_SCHEMA_VERSION = 'vaxreplay.gateway-wire-request.v0.1'
GATEWAY_WIRE_RESPONSE_SCHEMA_VERSION = 'vaxreplay.gateway-wire-response.v0.1'
GATEWAY_ATTEMPT_RECEIPT_SCHEMA_VERSION = 'vaxreplay.gateway-attempt-receipt.v0.3'
GATEWAY_SESSION_SEAL_SCHEMA_VERSION = 'vaxreplay.gateway-session-seal.v0.3'
AUTHENTICATED_GATEWAY_SESSION_SCHEMA_VERSION = 'vaxreplay.authenticated-gateway-session.v0.3'
GATEWAY_SESSION_AUTHENTICATION = 'hmac-sha256-domain-separated'

_SESSION_HMAC_DOMAIN = b'vaxreplay.gateway-session-seal.v0.3\x00'
_SESSION_KEY_ID_DOMAIN = b'vaxreplay.gateway-session-key-id.v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class GatewayErrorCode(str, enum.Enum):
    UNAUTHORIZED = 'unauthorized'
    EXPIRED = 'expired'
    WRONG_RUN = 'wrong_run'
    WRONG_PEER = 'wrong_peer'
    OUT_OF_ORDER = 'out_of_order'
    REPLAY_CONFLICT = 'replay_conflict'
    INVALID_REQUEST = 'invalid_request'
    MODEL_FORBIDDEN = 'model_forbidden'
    BUDGET_EXHAUSTED = 'budget_exhausted'
    PROVIDER_TIMEOUT = 'provider_timeout'
    PROVIDER_RATE_LIMIT = 'provider_rate_limit'
    PROVIDER_REJECTED = 'provider_rejected'
    PROVIDER_PROTOCOL = 'provider_protocol'
    AMBIGUOUS_IN_FLIGHT = 'ambiguous_in_flight'
    INTERNAL = 'internal'


class GatewayTerminalReason(str, enum.Enum):
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'
    FAILED = 'failed'


class GatewayCapabilityRevocationReason(str, enum.Enum):
    """Why this gateway stopped admitting the local per-run capability."""

    SESSION_SEALED = 'session_sealed'
    RUNTIME_CLEANUP = 'runtime_cleanup'
    STARTUP_REAPER = 'startup_reaper'
    OPERATOR_CANCELLED = 'operator_cancelled'


class AuthenticatedGatewayError(RuntimeError):
    """Stable gateway error which never contains provider or credential text."""

    def __init__(self, code: GatewayErrorCode):
        super().__init__(code.value)
        self.code = code


class GatewayModelRoute(StrictModel):
    """Exact organizer-selected provider route and separately evidenced data-control claims.

    v0.4 deliberately rejects older documents instead of silently reinterpreting their committed
    storage or evidence claims. Routes are hash-bound into grants and receipts, so a legacy
    document cannot be safely upgraded in place. A non-default data-control SHA-256 is only a
    commitment here; the gateway does not load or semantically verify the external artifact.
    """

    schema_version: Literal['vaxreplay.gateway-model-route.v0.4'] = GATEWAY_MODEL_ROUTE_SCHEMA_VERSION
    route_id: str = Field(min_length=1)
    logical_model_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    provider_model_id: str = Field(min_length=1)
    resolved_model_id: str = Field(min_length=1)
    accepted_provider_model_ids: tuple[str, ...] = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    adapter_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    endpoint_origin: str = Field(min_length=1)
    endpoint_path: str = Field(min_length=1)
    fixed_parameters_sha256: str = Field(pattern=_SHA256_PATTERN)
    max_context_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    input_preflight: Literal['authoritative_tokenizer', 'conservative_upper_bound']
    reasoning_accounting: Literal['reported', 'not_applicable']
    provider_usage_authoritative: Literal[True] = True
    provider_storage_disabled: bool = True
    provider_data_control: Literal['default', 'modified_abuse_monitoring', 'zero_data_retention']
    provider_data_control_attested: bool = False
    provider_data_control_attestation_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    provider_data_control_evidence_verified_by_route_schema: Literal[False] = False
    provider_data_control_evidence_requires_operator_artifact: Literal[True] = True
    provider_model_snapshot_attested: Literal[False] = False
    request_body_logging: Literal[False] = False
    response_body_logging: Literal[False] = False
    redirects_allowed: Literal[False] = False
    ambient_proxy_configuration_allowed: Literal[False] = False

    @field_validator('accepted_provider_model_ids')
    @classmethod
    def validate_accepted_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('accepted provider model IDs must be unique and sorted')
        return value

    @field_validator('endpoint_origin')
    @classmethod
    def validate_origin(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != 'https'
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {'', '/'}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError('provider endpoint origin must be a bare HTTPS origin')
        if parsed.port not in {None, 443}:
            raise ValueError('provider endpoint origin must use the default HTTPS port')
        return value.removesuffix('/')

    @field_validator('endpoint_path')
    @classmethod
    def validate_endpoint_path(cls, value: str) -> str:
        if not value.startswith('/') or value.startswith('//') or '?' in value or '#' in value or '\\' in value:
            raise ValueError('provider endpoint path must be a fixed absolute path without a query')
        return value

    @model_validator(mode='after')
    def validate_route(self) -> Self:
        if self.provider_model_id not in self.accepted_provider_model_ids:
            raise ValueError('accepted provider model IDs must include the requested provider model')
        if self.resolved_model_id not in self.accepted_provider_model_ids:
            raise ValueError('accepted provider model IDs must include the pinned resolved model')
        if self.max_output_tokens >= self.max_context_tokens:
            raise ValueError('route output limit must be smaller than its context window')
        externally_controlled = self.provider_data_control != 'default'
        if externally_controlled != self.provider_data_control_attested or externally_controlled != (
            self.provider_data_control_attestation_sha256 is not None
        ):
            raise ValueError('non-default provider data control requires one external attestation commitment')
        return self


class AuthenticatedGatewayPolicy(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-gateway-policy.v0.2'] = AUTHENTICATED_GATEWAY_POLICY_SCHEMA_VERSION
    gateway_id: str = Field(min_length=1)
    gateway_version: str = Field(min_length=1)
    gateway_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    maximum_frame_body_bytes: int = Field(default=DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES, ge=1, le=64 * 1024 * 1024)
    maximum_session_wire_bytes: int = Field(default=64 * 1024 * 1024, ge=1, le=1024 * 1024 * 1024)
    maximum_provider_call_seconds: int = Field(default=300, ge=1, le=3600)
    synchronous_non_streaming_only: Literal[True] = True
    automatic_provider_retries: Literal[False] = False
    worker_controls_provider_route: Literal[False] = False
    worker_receives_provider_credentials: Literal[False] = False

    @model_validator(mode='after')
    def validate_wire_capacity(self) -> Self:
        if self.maximum_frame_body_bytes < _maximum_gateway_error_body_bytes():
            raise ValueError('gateway frame body limit cannot hold a terminal error response')
        minimum_terminal_exchange = 2 * maximum_gateway_frame_bytes(self.maximum_frame_body_bytes)
        if self.maximum_session_wire_bytes < minimum_terminal_exchange:
            raise ValueError('gateway session wire budget must hold one maximum request and rejection frame')
        return self


class GatewayCapabilityGrant(StrictModel):
    schema_version: Literal['vaxreplay.gateway-capability-grant.v0.1'] = GATEWAY_CAPABILITY_GRANT_SCHEMA_VERSION
    capability_id: str = Field(pattern=_SHA256_PATTERN)
    audience: Literal['vaxreplay-provider-gateway'] = 'vaxreplay-provider-gateway'
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_route_sha256: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    expires_at: datetime
    expected_peer_cid: int = Field(ge=3, le=2**32 - 1)
    limits: AgenticRunLimits
    maximum_request_bytes: int = Field(ge=1, le=16 * 1024 * 1024)
    maximum_session_wire_bytes: int = Field(ge=1, le=1024 * 1024 * 1024)
    one_session: Literal[True] = True

    @field_validator('issued_at', 'expires_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_window(self) -> Self:
        if self.expires_at <= self.issued_at:
            raise ValueError('gateway capability must expire after it is issued')
        return self


class GatewayCapabilityBinding(StrictModel):
    """Restart-visible identity of one local gateway capability, without its secret."""

    schema_version: Literal['vaxreplay.gateway-capability-binding.v0.1'] = GATEWAY_CAPABILITY_BINDING_SCHEMA_VERSION
    capability_id: str = Field(pattern=_SHA256_PATTERN)
    grant_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_route_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_peer_cid: int = Field(ge=3, le=2**32 - 1)
    issued_at: datetime
    expires_at: datetime

    @field_validator('issued_at', 'expires_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_window(self) -> Self:
        if self.expires_at <= self.issued_at:
            raise ValueError('gateway capability binding has an invalid time window')
        return self


class GatewayCapabilityRevocation(StrictModel):
    """Durable local admission tombstone; this is not provider-key revocation."""

    schema_version: Literal['vaxreplay.gateway-capability-revocation.v0.1'] = (
        GATEWAY_CAPABILITY_REVOCATION_SCHEMA_VERSION
    )
    capability_id: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_route_sha256: str = Field(pattern=_SHA256_PATTERN)
    registered_binding: GatewayCapabilityBinding | None = None
    reason: GatewayCapabilityRevocationReason
    revoked_at: datetime
    local_gateway_admission_revoked: Literal[True] = True
    provider_api_credentials_revoked: Literal[False] = False
    already_dispatched_provider_requests_cancelled: Literal[False] = False

    @field_validator('revoked_at')
    @classmethod
    def validate_revoked_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('gateway capability revocation time must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_binding(self) -> Self:
        if self.registered_binding is not None and (
            self.registered_binding.capability_id,
            self.registered_binding.run_id,
            self.registered_binding.attempt_reservation_sha256,
            self.registered_binding.model_route_sha256,
        ) != (
            self.capability_id,
            self.run_id,
            self.attempt_reservation_sha256,
            self.model_route_sha256,
        ):
            raise ValueError('gateway capability revocation differs from its registered binding')
        if self.registered_binding is not None and self.revoked_at < self.registered_binding.issued_at:
            raise ValueError('gateway capability cannot be revoked before issuance')
        return self


class GatewayLedgerIdentity(StrictModel):
    """Pinned filesystem identity shared by the gateway and managed startup reaper."""

    schema_version: Literal['vaxreplay.gateway-ledger-identity.v0.1'] = GATEWAY_LEDGER_IDENTITY_SCHEMA_VERSION
    resolved_path: str
    device_id: int = Field(ge=0, le=2**63 - 1)
    inode: int = Field(gt=0, le=2**63 - 1)
    admission_lock_resolved_path: str
    admission_lock_device_id: int = Field(ge=0, le=2**63 - 1)
    admission_lock_inode: int = Field(gt=0, le=2**63 - 1)

    @field_validator('resolved_path', 'admission_lock_resolved_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('gateway ledger identity paths must be normalized and absolute')
        return value


class GatewayWireRequest(StrictModel):
    schema_version: Literal['vaxreplay.gateway-wire-request.v0.1'] = GATEWAY_WIRE_REQUEST_SCHEMA_VERSION
    grant_sha256: str = Field(pattern=_SHA256_PATTERN)
    request: AgenticModelRequest


class GatewayWireResponse(StrictModel):
    schema_version: Literal['vaxreplay.gateway-wire-response.v0.1'] = GATEWAY_WIRE_RESPONSE_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    call_index: int = Field(ge=0)
    succeeded: bool
    response: AgenticModelResponse | None = None
    error_code: GatewayErrorCode | None = None
    error_message: Literal['gateway request rejected'] | None = None

    @model_validator(mode='after')
    def validate_outcome(self) -> Self:
        if self.succeeded:
            if self.response is None or self.error_code is not None or self.error_message is not None:
                raise ValueError('successful gateway responses require only a model response')
            if (self.response.run_id, self.response.call_index) != (self.run_id, self.call_index):
                raise ValueError('gateway wire response identity does not match its model response')
        elif self.response is not None or self.error_code is None or self.error_message is None:
            raise ValueError('failed gateway responses require only a stable error')
        return self


def _maximum_gateway_error_body_bytes() -> int:
    return max(
        len(
            canonical_json_bytes(
                GatewayWireResponse(
                    run_id='0' * 32,
                    call_index=0,
                    succeeded=False,
                    error_code=code,
                    error_message='gateway request rejected',
                )
            )
        )
        for code in GatewayErrorCode
    )


class GatewayAttemptReceipt(StrictModel):
    schema_version: Literal['vaxreplay.gateway-attempt-receipt.v0.3'] = GATEWAY_ATTEMPT_RECEIPT_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    capability_id: str = Field(pattern=_SHA256_PATTERN)
    call_index: int = Field(ge=0)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    request_bytes: int = Field(gt=0)
    succeeded: bool
    response_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    response_bytes: int = Field(default=0, ge=0)
    error_code: GatewayErrorCode | None = None
    # ``None`` is reserved for a crash-left reservation: the restarted gateway can prove that the
    # call was reserved, but cannot safely claim whether the old process reached the provider.
    provider_dispatched: bool | None
    provider_result: ProviderCallResult | None = None
    exchange_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    admitted_wire_bytes: int = Field(ge=0)
    terminal_budget_rejection_wire_bytes: int = Field(default=0, ge=0)
    exact_replay_count: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def validate_attempt(self) -> Self:
        if self.succeeded:
            if (
                self.response_sha256 is None
                or self.response_bytes == 0
                or self.error_code is not None
                or self.provider_dispatched is not True
                or self.provider_result is None
                or self.exchange_sha256 is None
            ):
                raise ValueError('successful gateway attempts require complete provider and exchange evidence')
        elif self.response_sha256 is not None or self.response_bytes != 0 or self.error_code is None:
            raise ValueError('failed gateway attempts require an error and no model response binding')
        if self.provider_result is not None and self.provider_dispatched is not True:
            raise ValueError('provider result evidence requires a dispatched provider call')
        if self.provider_dispatched is None and (
            self.succeeded or self.error_code != GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
        ):
            raise ValueError('unknown provider dispatch state is reserved for ambiguous in-flight failures')
        if self.admitted_wire_bytes == 0 and self.terminal_budget_rejection_wire_bytes == 0:
            raise ValueError('gateway attempt must account for admitted or terminal rejection wire bytes')
        if (
            self.terminal_budget_rejection_wire_bytes
            and not self.succeeded
            and (
                self.error_code != GatewayErrorCode.BUDGET_EXHAUSTED
                or self.provider_dispatched is not False
                or self.exact_replay_count != 0
                or self.admitted_wire_bytes != 0
            )
        ):
            raise ValueError('terminal budget-rejection bytes require a no-dispatch budget failure')
        return self


class GatewaySessionSeal(StrictModel):
    schema_version: Literal['vaxreplay.gateway-session-seal.v0.3'] = GATEWAY_SESSION_SEAL_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    capability_id: str = Field(pattern=_SHA256_PATTERN)
    grant_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_route_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_id: str = Field(min_length=1)
    gateway_version: str = Field(min_length=1)
    gateway_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    receipt_authentication: Literal['hmac-sha256-domain-separated'] = GATEWAY_SESSION_AUTHENTICATION
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    transcript_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    terminal_reason: GatewayTerminalReason
    terminal_error_code: GatewayErrorCode | None = None
    attempt_count: int = Field(ge=0)
    successful_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    provider_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    admitted_wire_bytes: int = Field(ge=0)
    terminal_budget_rejection_wire_bytes: int = Field(ge=0)
    terminal_observed_overage_bytes: int = Field(ge=0)
    exact_replay_count: int = Field(ge=0)
    issued_at: datetime
    sealed_at: datetime

    @field_validator('issued_at', 'sealed_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_terminal(self) -> Self:
        if self.sealed_at < self.issued_at:
            raise ValueError('gateway session cannot be sealed before issuance')
        if self.successful_call_count > self.attempt_count:
            raise ValueError('successful calls cannot exceed gateway attempts')
        if (self.terminal_reason == GatewayTerminalReason.FAILED) != (self.terminal_error_code is not None):
            raise ValueError('only failed gateway sessions require a terminal error code')
        return self


class AuthenticatedGatewaySession(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-gateway-session.v0.3'] = (
        AUTHENTICATED_GATEWAY_SESSION_SCHEMA_VERSION
    )
    grant: GatewayCapabilityGrant
    route: GatewayModelRoute
    policy: AuthenticatedGatewayPolicy
    transcript: AgenticGatewayTranscript
    attempts: tuple[GatewayAttemptReceipt, ...]
    seal: GatewaySessionSeal
    seal_hmac: str = Field(pattern=_SHA256_PATTERN)


def gateway_model_route_sha256(route: GatewayModelRoute) -> str:
    return _sha256(canonical_json_bytes(route))


def authenticated_gateway_policy_sha256(policy: AuthenticatedGatewayPolicy) -> str:
    return _sha256(canonical_json_bytes(policy))


def gateway_capability_grant_sha256(grant: GatewayCapabilityGrant) -> str:
    return _sha256(canonical_json_bytes(grant))


def gateway_capability_binding(grant: GatewayCapabilityGrant) -> GatewayCapabilityBinding:
    """Project the exact non-secret identity retained by the revocation authority."""

    return GatewayCapabilityBinding(
        capability_id=grant.capability_id,
        grant_sha256=gateway_capability_grant_sha256(grant),
        run_id=grant.run_id,
        attempt_reservation_sha256=grant.attempt_reservation_sha256,
        execution_policy_sha256=grant.execution_policy_sha256,
        workspace_manifest_sha256=grant.workspace_manifest_sha256,
        gateway_policy_sha256=grant.gateway_policy_sha256,
        model_route_sha256=grant.model_route_sha256,
        expected_peer_cid=grant.expected_peer_cid,
        issued_at=grant.issued_at,
        expires_at=grant.expires_at,
    )


def gateway_session_key_id(key: bytes) -> str:
    _validate_receipt_key(key)
    return _sha256(_SESSION_KEY_ID_DOMAIN + key)


def gateway_session_seal_hmac(seal: GatewaySessionSeal, key: bytes) -> str:
    _validate_receipt_key(key)
    return hmac.new(key, _SESSION_HMAC_DOMAIN + canonical_json_bytes(seal), hashlib.sha256).hexdigest()


def issue_gateway_capability(
    *,
    secret: bytes,
    run_id: str,
    attempt_reservation_sha256: str,
    execution_policy_sha256: str,
    workspace_manifest_sha256: str,
    policy: AuthenticatedGatewayPolicy,
    route: GatewayModelRoute,
    issued_at: datetime,
    expires_at: datetime,
    expected_peer_cid: int,
    limits: AgenticRunLimits,
) -> GatewayCapabilityGrant:
    maximum_request = min(policy.maximum_frame_body_bytes, 16 * 1024 * 1024)
    return GatewayCapabilityGrant(
        capability_id=gateway_capability_id(secret),
        run_id=run_id,
        attempt_reservation_sha256=attempt_reservation_sha256,
        execution_policy_sha256=execution_policy_sha256,
        workspace_manifest_sha256=workspace_manifest_sha256,
        gateway_policy_sha256=authenticated_gateway_policy_sha256(policy),
        model_route_sha256=gateway_model_route_sha256(route),
        issued_at=issued_at,
        expires_at=expires_at,
        expected_peer_cid=expected_peer_cid,
        limits=limits,
        maximum_request_bytes=maximum_request,
        maximum_session_wire_bytes=policy.maximum_session_wire_bytes,
    )


@dataclass(frozen=True)
class _LedgerSession:
    grant: GatewayCapabilityGrant
    route: GatewayModelRoute
    policy: AuthenticatedGatewayPolicy
    status: str
    next_index: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    provider_cost_usd: float | None
    admitted_wire_bytes: int
    terminal_budget_rejection_wire_bytes: int
    dispatch_reserved_wire_bytes: int
    exact_replay_count: int
    terminal_error_code: GatewayErrorCode | None
    terminal_reason: GatewayTerminalReason | None


@dataclass(frozen=True)
class _Reservation:
    cached_response: GatewayWireResponse | None
    dispatch_ambiguous: bool = False


class SqliteGatewayLedger:
    """Durable at-most-once call and local capability-revocation authority.

    Capability secrets remain outside SQLite.  The sibling admission lock serializes a complete
    provider call against a restart/startup-reaper revocation, including across processes which
    open the same ledger.  Thus a revocation linearizes either before a call is admitted or after
    that already-admitted call finishes locally; it does not claim to cancel a remote request.
    """

    def __init__(self, path: Path) -> None:
        supplied = path.expanduser()
        if supplied.is_symlink():
            raise ValueError('gateway ledger path cannot be a symlink')
        parent = supplied.parent
        if not parent.exists():
            try:
                parent.mkdir(mode=0o700)
            except OSError as error:
                raise ValueError('gateway ledger parent cannot be created privately') from error
        try:
            parent_stat = parent.lstat()
        except OSError as error:
            raise ValueError('gateway ledger parent is unavailable') from error
        if (
            stat.S_ISLNK(parent_stat.st_mode)
            or not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != os.geteuid()
            or stat.S_IMODE(parent_stat.st_mode) & 0o077
        ):
            raise ValueError('gateway ledger parent must be a private, owned, non-symlink directory')
        self.path = parent.resolve() / supplied.name
        self._admission_lock_path = self.path.with_name(f'{self.path.name}.admission.lock')
        if self.path.exists():
            existing = self.path.lstat()
            if (
                not stat.S_ISREG(existing.st_mode)
                or existing.st_nlink != 1
                or existing.st_uid != os.geteuid()
                or stat.S_IMODE(existing.st_mode) != 0o600
            ):
                raise ValueError('existing gateway ledger must be a private, owned, single-link regular file')
        self._initialize_admission_lock()
        self._initialize()
        ledger_metadata = self.path.lstat()
        lock_metadata = self._admission_lock_path.lstat()
        self._identity = GatewayLedgerIdentity(
            resolved_path=str(self.path),
            device_id=ledger_metadata.st_dev,
            inode=ledger_metadata.st_ino,
            admission_lock_resolved_path=str(self._admission_lock_path),
            admission_lock_device_id=lock_metadata.st_dev,
            admission_lock_inode=lock_metadata.st_ino,
        )

    @property
    def identity(self) -> GatewayLedgerIdentity:
        """Return the originally pinned DB/lock inodes after checking they remain named."""

        self._require_pinned_identity()
        return self._identity

    def _initialize_admission_lock(self) -> None:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        try:
            descriptor = os.open(self._admission_lock_path, flags, 0o600)
        except OSError as error:
            raise ValueError('gateway admission lock could not be opened safely') from error
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or stat.S_IMODE(opened.st_mode) != 0o600
            ):
                raise ValueError('gateway admission lock must be a private, owned, single-link regular file')
        finally:
            os.close(descriptor)

    @contextmanager
    def capability_admission_guard(self) -> Iterator[None]:
        """Serialize complete local call admission/execution against durable revocation."""

        before = self._admission_lock_path.lstat()
        flags = os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        try:
            descriptor = os.open(self._admission_lock_path, flags)
        except OSError as error:
            raise ValueError('gateway admission lock could not be reopened safely') from error
        locked = False
        try:
            opened = os.fstat(descriptor)
            if (
                (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or stat.S_IMODE(opened.st_mode) != 0o600
            ):
                raise ValueError('gateway admission lock changed while opening')
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            after = self._admission_lock_path.lstat()
            if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError('gateway admission lock changed while acquiring it')
            yield
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _connect(self) -> sqlite3.Connection:
        if hasattr(self, '_identity'):
            self._require_pinned_identity()
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('PRAGMA trusted_schema=OFF')
        return connection

    def _require_pinned_identity(self) -> None:
        try:
            ledger = self.path.lstat()
            lock = self._admission_lock_path.lstat()
        except OSError as error:
            raise ValueError('gateway ledger or admission lock is unavailable') from error
        expected = self._identity
        if (
            (ledger.st_dev, ledger.st_ino) != (expected.device_id, expected.inode)
            or not stat.S_ISREG(ledger.st_mode)
            or ledger.st_uid != os.geteuid()
            or ledger.st_nlink != 1
            or stat.S_IMODE(ledger.st_mode) != 0o600
            or (lock.st_dev, lock.st_ino) != (expected.admission_lock_device_id, expected.admission_lock_inode)
            or not stat.S_ISREG(lock.st_mode)
            or lock.st_uid != os.geteuid()
            or lock.st_nlink != 1
            or stat.S_IMODE(lock.st_mode) != 0o600
        ):
            raise ValueError('gateway ledger or admission lock changed filesystem identity')

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    capability_id TEXT PRIMARY KEY,
                    grant_json BLOB NOT NULL,
                    route_json BLOB NOT NULL,
                    policy_json BLOB NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('open', 'failed', 'closed')),
                    next_index INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    provider_cost_usd REAL,
                    admitted_wire_bytes INTEGER NOT NULL,
                    terminal_budget_rejection_wire_bytes INTEGER NOT NULL,
                    dispatch_reserved_wire_bytes INTEGER NOT NULL,
                    exact_replay_count INTEGER NOT NULL,
                    terminal_error_code TEXT,
                    terminal_reason TEXT CHECK(terminal_reason IN ('completed', 'cancelled', 'failed'))
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    capability_id TEXT NOT NULL,
                    call_index INTEGER NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    request_bytes INTEGER NOT NULL CHECK(request_bytes > 0),
                    request_frame_bytes INTEGER NOT NULL CHECK(request_frame_bytes > 0),
                    state TEXT NOT NULL CHECK(state IN ('reserved', 'completed')),
                    wire_response_json BLOB,
                    attempt_json BLOB,
                    exchange_json BLOB,
                    PRIMARY KEY(capability_id, call_index),
                    FOREIGN KEY(capability_id) REFERENCES sessions(capability_id)
                );
                CREATE TABLE IF NOT EXISTS session_seals (
                    capability_id TEXT PRIMARY KEY,
                    artifact_json BLOB NOT NULL,
                    FOREIGN KEY(capability_id) REFERENCES sessions(capability_id)
                );
                CREATE TABLE IF NOT EXISTS capability_revocations (
                    capability_id TEXT PRIMARY KEY,
                    revocation_json BLOB NOT NULL
                );
                """
            )
            session_columns = {str(row[1]) for row in connection.execute('PRAGMA table_info(sessions)').fetchall()}
            attempt_columns = {str(row[1]) for row in connection.execute('PRAGMA table_info(attempts)').fetchall()}
            if (
                not {
                    'terminal_reason',
                    'exact_replay_count',
                    'admitted_wire_bytes',
                    'terminal_budget_rejection_wire_bytes',
                    'dispatch_reserved_wire_bytes',
                }
                <= session_columns
                or not {
                    'request_bytes',
                    'request_frame_bytes',
                }
                <= attempt_columns
            ):
                raise ValueError('legacy gateway ledger schema is unsupported; create a new ledger')
        self.path.chmod(0o600)

    def register(
        self,
        grant: GatewayCapabilityGrant,
        route: GatewayModelRoute,
        policy: AuthenticatedGatewayPolicy,
    ) -> None:
        if grant.gateway_policy_sha256 != authenticated_gateway_policy_sha256(policy):
            raise ValueError('gateway grant does not bind the registered policy')
        if grant.model_route_sha256 != gateway_model_route_sha256(route):
            raise ValueError('gateway grant does not bind the registered model route')
        with self.capability_admission_guard():
            with self._connect() as connection:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    revoked = connection.execute(
                        'SELECT 1 FROM capability_revocations WHERE capability_id=?',
                        (grant.capability_id,),
                    ).fetchone()
                    if revoked is not None:
                        raise ValueError('gateway capability has a durable revocation tombstone')
                    connection.execute(
                        """
                        INSERT INTO sessions(
                            capability_id, grant_json, route_json, policy_json, status, next_index,
                            input_tokens, output_tokens, reasoning_tokens, provider_cost_usd,
                            admitted_wire_bytes, terminal_budget_rejection_wire_bytes,
                            dispatch_reserved_wire_bytes, exact_replay_count,
                            terminal_error_code, terminal_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            grant.capability_id,
                            canonical_json_bytes(grant),
                            canonical_json_bytes(route),
                            canonical_json_bytes(policy),
                            'open',
                            0,
                            0,
                            0,
                            0,
                            None,
                            0,
                            0,
                            0,
                            0,
                            None,
                            None,
                        ),
                    )
                    connection.execute('COMMIT')
                except sqlite3.IntegrityError as error:
                    if connection.in_transaction:
                        connection.execute('ROLLBACK')
                    raise ValueError('gateway capability session already exists') from error
                except BaseException:
                    if connection.in_transaction:
                        connection.execute('ROLLBACK')
                    raise

    def load(self, capability_id: str) -> _LedgerSession:
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM sessions WHERE capability_id=?', (capability_id,)).fetchone()
        if row is None:
            raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
        return _session_from_row(row)

    def capability_binding(self, capability_id: str) -> GatewayCapabilityBinding:
        """Return the immutable exact grant projection, whether active or revoked."""

        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM sessions WHERE capability_id=?',
                (capability_id,),
            ).fetchone()
        if row is None:
            raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
        return gateway_capability_binding(_session_from_row(row).grant)

    def unrevoked_capability_bindings(self) -> tuple[GatewayCapabilityBinding, ...]:
        """Enumerate every session row which lacks an explicit durable tombstone."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sessions.*
                FROM sessions
                LEFT JOIN capability_revocations USING(capability_id)
                WHERE capability_revocations.capability_id IS NULL
                ORDER BY sessions.capability_id
                """
            ).fetchall()
        return tuple(gateway_capability_binding(_session_from_row(row).grant) for row in rows)

    def capability_revocation(
        self,
        capability_id: str,
    ) -> GatewayCapabilityRevocation | None:
        """Load and strictly revalidate the canonical durable tombstone, if one exists."""

        with self._connect() as connection:
            row = connection.execute(
                'SELECT revocation_json FROM capability_revocations WHERE capability_id=?',
                (capability_id,),
            ).fetchone()
        if row is None:
            return None
        payload = bytes(row['revocation_json'])
        revocation = GatewayCapabilityRevocation.model_validate_json(payload)
        if canonical_json_bytes(revocation) != payload or revocation.capability_id != capability_id:
            raise ValueError('gateway capability revocation is noncanonical or misbound')
        if revocation.registered_binding is not None and revocation.registered_binding != self.capability_binding(
            capability_id
        ):
            raise ValueError('gateway capability revocation differs from its immutable grant')
        return revocation

    def require_capability_admission(self, capability_id: str) -> GatewayCapabilityBinding:
        """Fail closed unless the session exists and has no durable revocation tombstone.

        Ordinary failed/closed sessions still reach the existing ledger preflight so the gateway
        can return its stable authenticated terminal response; those paths cannot dispatch.
        """

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT sessions.*, capability_revocations.capability_id AS revoked_capability_id
                FROM sessions
                LEFT JOIN capability_revocations USING(capability_id)
                WHERE sessions.capability_id=?
                """,
                (capability_id,),
            ).fetchone()
        if row is None or row['revoked_capability_id'] is not None:
            raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
        return gateway_capability_binding(_session_from_row(row).grant)

    def revoke_capability(
        self,
        *,
        capability_id: str,
        expected_run_id: str,
        expected_attempt_reservation_sha256: str,
        expected_model_route_sha256: str,
        reason: GatewayCapabilityRevocationReason,
        revoked_at: datetime,
    ) -> GatewayCapabilityRevocation:
        """Atomically tombstone exact local admission; repeated exact revocation is safe.

        The admission file lock is also held by ``AuthenticatedProviderGateway.handle_frame`` for
        the complete local call.  A revocation therefore cannot slip between dispatch admission
        and the provider call.  A request dispatched by a process which crashed before this lock
        was acquired remains intentionally outside the cancellation claim.
        """

        revoked_at = _aware(revoked_at, 'capability revocation timestamp')
        with self.capability_admission_guard():
            with self._connect() as connection:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    row = connection.execute(
                        'SELECT * FROM sessions WHERE capability_id=?',
                        (capability_id,),
                    ).fetchone()
                    if row is None:
                        raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
                    session = _session_from_row(row)
                    binding = gateway_capability_binding(session.grant)
                    if (
                        binding.run_id,
                        binding.attempt_reservation_sha256,
                        binding.model_route_sha256,
                    ) != (
                        expected_run_id,
                        expected_attempt_reservation_sha256,
                        expected_model_route_sha256,
                    ):
                        raise ValueError('gateway capability revocation differs from its exact binding')
                    existing_row = connection.execute(
                        'SELECT revocation_json FROM capability_revocations WHERE capability_id=?',
                        (capability_id,),
                    ).fetchone()
                    if existing_row is not None:
                        payload = bytes(existing_row['revocation_json'])
                        existing = GatewayCapabilityRevocation.model_validate_json(payload)
                        if canonical_json_bytes(existing) != payload or existing.registered_binding != binding:
                            raise ValueError('gateway capability has a conflicting revocation tombstone')
                        connection.execute('COMMIT')
                        return existing
                    revocation = GatewayCapabilityRevocation(
                        capability_id=binding.capability_id,
                        run_id=binding.run_id,
                        attempt_reservation_sha256=binding.attempt_reservation_sha256,
                        model_route_sha256=binding.model_route_sha256,
                        registered_binding=binding,
                        reason=reason,
                        revoked_at=max(revoked_at, binding.issued_at),
                    )
                    connection.execute(
                        'INSERT INTO capability_revocations(capability_id, revocation_json) VALUES (?, ?)',
                        (capability_id, canonical_json_bytes(revocation)),
                    )
                    if session.status == 'open':
                        connection.execute(
                            """
                            UPDATE sessions
                            SET status='closed', terminal_reason='cancelled'
                            WHERE capability_id=? AND status='open'
                            """,
                            (capability_id,),
                        )
                    connection.execute('COMMIT')
                    return revocation
                except BaseException:
                    if connection.in_transaction:
                        connection.execute('ROLLBACK')
                    raise

    def revoke_unregistered_capability(
        self,
        *,
        capability_id: str,
        expected_run_id: str,
        expected_attempt_reservation_sha256: str,
        expected_model_route_sha256: str,
        reason: GatewayCapabilityRevocationReason,
        revoked_at: datetime,
    ) -> GatewayCapabilityRevocation:
        """Tombstone a start-bound capability which crashed before gateway registration.

        Registration takes the same cross-process admission lock and rejects any tombstone, so a
        later process cannot resurrect this exact capability.  The exact run, redeemed-start hash,
        and pinned model route are retained even though no grant/session row was ever observed.
        """

        revoked_at = _aware(revoked_at, 'capability revocation timestamp')
        candidate = GatewayCapabilityRevocation(
            capability_id=capability_id,
            run_id=expected_run_id,
            attempt_reservation_sha256=expected_attempt_reservation_sha256,
            model_route_sha256=expected_model_route_sha256,
            registered_binding=None,
            reason=reason,
            revoked_at=revoked_at,
        )
        with self.capability_admission_guard():
            with self._connect() as connection:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    session = connection.execute(
                        'SELECT 1 FROM sessions WHERE capability_id=?',
                        (capability_id,),
                    ).fetchone()
                    if session is not None:
                        raise ValueError('registered gateway capability requires its exact persisted grant binding')
                    existing_row = connection.execute(
                        'SELECT revocation_json FROM capability_revocations WHERE capability_id=?',
                        (capability_id,),
                    ).fetchone()
                    if existing_row is not None:
                        payload = bytes(existing_row['revocation_json'])
                        existing = GatewayCapabilityRevocation.model_validate_json(payload)
                        if canonical_json_bytes(existing) != payload or (
                            existing.capability_id,
                            existing.run_id,
                            existing.attempt_reservation_sha256,
                            existing.model_route_sha256,
                            existing.registered_binding,
                        ) != (
                            candidate.capability_id,
                            candidate.run_id,
                            candidate.attempt_reservation_sha256,
                            candidate.model_route_sha256,
                            None,
                        ):
                            raise ValueError('gateway capability has a conflicting pre-registration tombstone')
                        connection.execute('COMMIT')
                        return existing
                    connection.execute(
                        'INSERT INTO capability_revocations(capability_id, revocation_json) VALUES (?, ?)',
                        (capability_id, canonical_json_bytes(candidate)),
                    )
                    connection.execute('COMMIT')
                    return candidate
                except BaseException:
                    if connection.in_transaction:
                        connection.execute('ROLLBACK')
                    raise

    def reserve(
        self,
        capability_id: str,
        request: AgenticModelRequest,
        *,
        request_frame_bytes: int,
    ) -> _Reservation:
        if request_frame_bytes <= 0:
            raise ValueError('gateway request frame byte count must be positive')
        request_bytes = canonical_json_bytes(request)
        request_sha256 = _sha256(request_bytes)
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                session = connection.execute(
                    'SELECT status, next_index FROM sessions WHERE capability_id=?', (capability_id,)
                ).fetchone()
                if session is None:
                    raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
                if session['status'] != 'open':
                    raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
                existing = connection.execute(
                    'SELECT * FROM attempts WHERE capability_id=? AND call_index=?',
                    (capability_id, request.call_index),
                ).fetchone()
                if existing is not None:
                    if existing['request_sha256'] != request_sha256:
                        raise AuthenticatedGatewayError(GatewayErrorCode.REPLAY_CONFLICT)
                    if existing['state'] == 'reserved':
                        connection.execute('COMMIT')
                        return _Reservation(cached_response=None, dispatch_ambiguous=True)
                    if request.call_index != int(session['next_index']) - 1:
                        raise AuthenticatedGatewayError(GatewayErrorCode.OUT_OF_ORDER)
                    response = GatewayWireResponse.model_validate_json(bytes(existing['wire_response_json']))
                    connection.execute('COMMIT')
                    return _Reservation(cached_response=response)
                if request.call_index != session['next_index']:
                    raise AuthenticatedGatewayError(GatewayErrorCode.OUT_OF_ORDER)
                connection.execute(
                    """
                    INSERT INTO attempts(
                        capability_id, call_index, request_sha256, request_bytes, request_frame_bytes, state
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        capability_id,
                        request.call_index,
                        request_sha256,
                        len(request_bytes),
                        request_frame_bytes,
                        'reserved',
                    ),
                )
                connection.execute('COMMIT')
            except BaseException:
                if connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise
        return _Reservation(cached_response=None)

    def reserve_provider_dispatch(
        self,
        *,
        capability_id: str,
        call_index: int,
        maximum_exchange_wire_bytes: int,
    ) -> bool:
        """Atomically reserve worst-case admitted capacity before provider side effects."""

        if maximum_exchange_wire_bytes <= 0:
            raise ValueError('provider dispatch reservation must be positive')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                session = connection.execute(
                    'SELECT * FROM sessions WHERE capability_id=?',
                    (capability_id,),
                ).fetchone()
                attempt = connection.execute(
                    'SELECT state FROM attempts WHERE capability_id=? AND call_index=?',
                    (capability_id, call_index),
                ).fetchone()
                if (
                    session is None
                    or session['status'] != 'open'
                    or session['next_index'] != call_index
                    or attempt is None
                    or attempt['state'] != 'reserved'
                    or int(session['dispatch_reserved_wire_bytes']) != 0
                ):
                    raise AuthenticatedGatewayError(GatewayErrorCode.AMBIGUOUS_IN_FLIGHT)
                loaded_session = _session_from_row(session)
                if (
                    loaded_session.admitted_wire_bytes + maximum_exchange_wire_bytes
                    > loaded_session.grant.maximum_session_wire_bytes
                ):
                    connection.execute('COMMIT')
                    return False
                connection.execute(
                    'UPDATE sessions SET dispatch_reserved_wire_bytes=? WHERE capability_id=?',
                    (maximum_exchange_wire_bytes, capability_id),
                )
                connection.execute('COMMIT')
                return True
            except BaseException:
                if connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise

    def charge_cached_replay(
        self,
        *,
        capability_id: str,
        request: AgenticModelRequest,
        admitted_wire_bytes_delta: int,
        terminal_rejection_wire_bytes: int,
    ) -> bool:
        """Atomically charge an exact cached response or fail the session at its wire limit."""

        if admitted_wire_bytes_delta <= 0 or terminal_rejection_wire_bytes <= 0:
            raise ValueError('cached replay wire-byte charges must be positive')
        request_sha256 = _sha256(canonical_json_bytes(request))
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                session = connection.execute(
                    'SELECT * FROM sessions WHERE capability_id=?',
                    (capability_id,),
                ).fetchone()
                if session is None or session['status'] != 'open':
                    raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
                attempt = connection.execute(
                    'SELECT * FROM attempts WHERE capability_id=? AND call_index=?',
                    (capability_id, request.call_index),
                ).fetchone()
                if attempt is None or attempt['state'] != 'completed':
                    raise AuthenticatedGatewayError(GatewayErrorCode.AMBIGUOUS_IN_FLIGHT)
                if attempt['request_sha256'] != request_sha256:
                    raise AuthenticatedGatewayError(GatewayErrorCode.REPLAY_CONFLICT)
                loaded_session = _session_from_row(session)
                if loaded_session.dispatch_reserved_wire_bytes:
                    raise AuthenticatedGatewayError(GatewayErrorCode.OUT_OF_ORDER)
                grant = loaded_session.grant
                next_wire_bytes = int(session['admitted_wire_bytes']) + admitted_wire_bytes_delta
                if next_wire_bytes > grant.maximum_session_wire_bytes:
                    receipt = GatewayAttemptReceipt.model_validate_json(bytes(attempt['attempt_json']))
                    if receipt.terminal_budget_rejection_wire_bytes:
                        raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
                    rejection_fits = (
                        loaded_session.admitted_wire_bytes + terminal_rejection_wire_bytes
                        <= grant.maximum_session_wire_bytes
                    )
                    receipt = receipt.model_copy(
                        update=(
                            {'admitted_wire_bytes': receipt.admitted_wire_bytes + terminal_rejection_wire_bytes}
                            if rejection_fits
                            else {'terminal_budget_rejection_wire_bytes': terminal_rejection_wire_bytes}
                        )
                    )
                    connection.execute(
                        'UPDATE attempts SET attempt_json=? WHERE capability_id=? AND call_index=?',
                        (canonical_json_bytes(receipt), capability_id, request.call_index),
                    )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status='failed', admitted_wire_bytes=?,
                            terminal_budget_rejection_wire_bytes=?,
                            terminal_error_code=?, terminal_reason='failed'
                        WHERE capability_id=?
                        """,
                        (
                            loaded_session.admitted_wire_bytes
                            + (terminal_rejection_wire_bytes if rejection_fits else 0),
                            0 if rejection_fits else terminal_rejection_wire_bytes,
                            GatewayErrorCode.BUDGET_EXHAUSTED.value,
                            capability_id,
                        ),
                    )
                    connection.execute('COMMIT')
                    return False
                connection.execute(
                    """
                    UPDATE sessions
                    SET admitted_wire_bytes=?, exact_replay_count=exact_replay_count+1
                    WHERE capability_id=?
                    """,
                    (next_wire_bytes, capability_id),
                )
                receipt = GatewayAttemptReceipt.model_validate_json(bytes(attempt['attempt_json']))
                receipt = receipt.model_copy(
                    update={
                        'admitted_wire_bytes': receipt.admitted_wire_bytes + admitted_wire_bytes_delta,
                        'exact_replay_count': receipt.exact_replay_count + 1,
                    }
                )
                connection.execute(
                    'UPDATE attempts SET attempt_json=? WHERE capability_id=? AND call_index=?',
                    (canonical_json_bytes(receipt), capability_id, request.call_index),
                )
                connection.execute('COMMIT')
                return True
            except BaseException:
                if connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise

    def complete(
        self,
        *,
        capability_id: str,
        wire_response: GatewayWireResponse,
        attempt: GatewayAttemptReceipt,
        exchange: AgenticModelExchange | None,
    ) -> None:
        if attempt.exact_replay_count != 0:
            raise ValueError('initial gateway attempt receipt has inconsistent wire accounting')
        provider_result = attempt.provider_result
        input_tokens = provider_result.usage.input_tokens if provider_result is not None else 0
        output_tokens = provider_result.usage.output_tokens if provider_result is not None else 0
        reasoning_tokens = (provider_result.usage.reasoning_tokens or 0) if provider_result is not None else 0
        provider_cost = provider_result.provider_cost_usd if provider_result is not None else None
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                row = connection.execute(
                    'SELECT * FROM attempts WHERE capability_id=? AND call_index=?',
                    (capability_id, attempt.call_index),
                ).fetchone()
                if row is None or row['state'] != 'reserved' or row['request_sha256'] != attempt.request_sha256:
                    raise AuthenticatedGatewayError(GatewayErrorCode.AMBIGUOUS_IN_FLIGHT)
                session = connection.execute(
                    'SELECT * FROM sessions WHERE capability_id=?', (capability_id,)
                ).fetchone()
                if session is None or session['status'] != 'open' or session['next_index'] != attempt.call_index:
                    raise AuthenticatedGatewayError(GatewayErrorCode.AMBIGUOUS_IN_FLIGHT)
                loaded_session = _session_from_row(session)
                next_admitted_wire_bytes = loaded_session.admitted_wire_bytes + attempt.admitted_wire_bytes
                if attempt.provider_dispatched is True:
                    if (
                        loaded_session.dispatch_reserved_wire_bytes == 0
                        or attempt.admitted_wire_bytes > loaded_session.dispatch_reserved_wire_bytes
                        or attempt.terminal_budget_rejection_wire_bytes != 0
                    ):
                        raise AuthenticatedGatewayError(GatewayErrorCode.AMBIGUOUS_IN_FLIGHT)
                elif loaded_session.dispatch_reserved_wire_bytes != 0:
                    raise AuthenticatedGatewayError(GatewayErrorCode.AMBIGUOUS_IN_FLIGHT)
                if next_admitted_wire_bytes > loaded_session.grant.maximum_session_wire_bytes:
                    raise AuthenticatedGatewayError(GatewayErrorCode.AMBIGUOUS_IN_FLIGHT)
                existing_cost = session['provider_cost_usd']
                next_cost = (
                    None
                    if existing_cost is None and provider_cost is None
                    else float(existing_cost or 0.0) + float(provider_cost or 0.0)
                )
                connection.execute(
                    """
                    UPDATE attempts
                    SET state='completed', wire_response_json=?, attempt_json=?, exchange_json=?
                    WHERE capability_id=? AND call_index=?
                    """,
                    (
                        canonical_json_bytes(wire_response),
                        canonical_json_bytes(attempt),
                        canonical_json_bytes(exchange) if exchange is not None else None,
                        capability_id,
                        attempt.call_index,
                    ),
                )
                connection.execute(
                    """
                    UPDATE sessions
                    SET status=?, next_index=?, input_tokens=?, output_tokens=?, reasoning_tokens=?,
                        provider_cost_usd=?, admitted_wire_bytes=?,
                        terminal_budget_rejection_wire_bytes=?, terminal_error_code=?, terminal_reason=?,
                        dispatch_reserved_wire_bytes=0
                    WHERE capability_id=?
                    """,
                    (
                        'open' if attempt.succeeded else 'failed',
                        attempt.call_index + 1,
                        session['input_tokens'] + input_tokens,
                        session['output_tokens'] + output_tokens,
                        session['reasoning_tokens'] + reasoning_tokens,
                        next_cost,
                        next_admitted_wire_bytes,
                        session['terminal_budget_rejection_wire_bytes'] + attempt.terminal_budget_rejection_wire_bytes,
                        None if attempt.error_code is None else attempt.error_code.value,
                        None if attempt.succeeded else GatewayTerminalReason.FAILED.value,
                        capability_id,
                    ),
                )
                connection.execute('COMMIT')
            except BaseException:
                if connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise

    def fail_session(self, capability_id: str, code: GatewayErrorCode) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET status='failed', terminal_error_code=?, terminal_reason='failed'
                WHERE capability_id=? AND status='open'
                """,
                (code.value, capability_id),
            )

    def close(self, capability_id: str, terminal_reason: GatewayTerminalReason) -> _LedgerSession:
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                row = connection.execute('SELECT * FROM sessions WHERE capability_id=?', (capability_id,)).fetchone()
                if row is None:
                    raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
                _session_from_row(row)
                pending = connection.execute(
                    "SELECT * FROM attempts WHERE capability_id=? AND state='reserved' ORDER BY call_index",
                    (capability_id,),
                ).fetchall()
                if pending:
                    grant = GatewayCapabilityGrant.model_validate_json(bytes(row['grant_json']))
                    pending_wire_bytes = sum(int(item['request_frame_bytes']) for item in pending)
                    if int(row['admitted_wire_bytes']) + pending_wire_bytes > grant.maximum_session_wire_bytes:
                        raise ValueError('gateway cannot seal an ambiguous reservation beyond its admitted wire budget')
                    for reservation in pending:
                        request_bytes = reservation['request_bytes']
                        request_frame_bytes = reservation['request_frame_bytes']
                        if (
                            request_bytes is None
                            or int(request_bytes) <= 0
                            or request_frame_bytes is None
                            or int(request_frame_bytes) <= 0
                        ):
                            raise ValueError('gateway cannot seal an unaccounted legacy call reservation')
                        attempt = GatewayAttemptReceipt(
                            run_id=grant.run_id,
                            capability_id=capability_id,
                            call_index=int(reservation['call_index']),
                            request_sha256=str(reservation['request_sha256']),
                            request_bytes=int(request_bytes),
                            succeeded=False,
                            error_code=GatewayErrorCode.AMBIGUOUS_IN_FLIGHT,
                            provider_dispatched=None,
                            admitted_wire_bytes=int(request_frame_bytes),
                        )
                        response = GatewayWireResponse(
                            run_id=grant.run_id,
                            call_index=int(reservation['call_index']),
                            succeeded=False,
                            error_code=GatewayErrorCode.AMBIGUOUS_IN_FLIGHT,
                            error_message='gateway request rejected',
                        )
                        connection.execute(
                            """
                            UPDATE attempts
                            SET state='completed', wire_response_json=?, attempt_json=?, exchange_json=NULL
                            WHERE capability_id=? AND call_index=? AND state='reserved'
                            """,
                            (
                                canonical_json_bytes(response),
                                canonical_json_bytes(attempt),
                                capability_id,
                                int(reservation['call_index']),
                            ),
                        )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status='failed', next_index=?,
                            admitted_wire_bytes=admitted_wire_bytes+?,
                            dispatch_reserved_wire_bytes=0,
                            terminal_error_code=?, terminal_reason='failed'
                        WHERE capability_id=?
                        """,
                        (
                            max(int(row['next_index']), max(int(item['call_index']) for item in pending) + 1),
                            pending_wire_bytes,
                            GatewayErrorCode.AMBIGUOUS_IN_FLIGHT.value,
                            capability_id,
                        ),
                    )
                    row = connection.execute(
                        'SELECT * FROM sessions WHERE capability_id=?',
                        (capability_id,),
                    ).fetchone()
                    if row is None:
                        raise AuthenticatedGatewayError(GatewayErrorCode.UNAUTHORIZED)
                actual_reason = GatewayTerminalReason.FAILED if row['status'] == 'failed' else terminal_reason
                stored_reason = row['terminal_reason']
                if stored_reason is not None and stored_reason != actual_reason.value:
                    raise ValueError('gateway session already has a different terminal reason')
                if row['status'] == 'open':
                    status = 'closed' if actual_reason != GatewayTerminalReason.FAILED else 'failed'
                    connection.execute(
                        'UPDATE sessions SET status=?, terminal_reason=? WHERE capability_id=?',
                        (status, actual_reason.value, capability_id),
                    )
                elif stored_reason is None:
                    connection.execute(
                        'UPDATE sessions SET terminal_reason=? WHERE capability_id=?',
                        (actual_reason.value, capability_id),
                    )
                connection.execute('COMMIT')
            except BaseException:
                if connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise
        return self.load(capability_id)

    def sealed_session(self, capability_id: str) -> AuthenticatedGatewaySession | None:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT artifact_json FROM session_seals WHERE capability_id=?',
                (capability_id,),
            ).fetchone()
            if row is None:
                return None
            pending = connection.execute(
                "SELECT 1 FROM attempts WHERE capability_id=? AND state='reserved' LIMIT 1",
                (capability_id,),
            ).fetchone()
            if pending is not None:
                raise ValueError('gateway cannot load a sealed session with an unaccounted call reservation')
        return AuthenticatedGatewaySession.model_validate_json(bytes(row['artifact_json']))

    def store_sealed_session(self, artifact: AuthenticatedGatewaySession) -> None:
        payload = canonical_json_bytes(artifact)
        with self._connect() as connection:
            try:
                connection.execute(
                    'INSERT INTO session_seals(capability_id, artifact_json) VALUES (?, ?)',
                    (artifact.grant.capability_id, payload),
                )
            except sqlite3.IntegrityError:
                row = connection.execute(
                    'SELECT artifact_json FROM session_seals WHERE capability_id=?',
                    (artifact.grant.capability_id,),
                ).fetchone()
                if row is None or not hmac.compare_digest(bytes(row['artifact_json']), payload):
                    raise ValueError('gateway session was already sealed with different evidence') from None

    def attempts(self, capability_id: str) -> tuple[tuple[GatewayAttemptReceipt, AgenticModelExchange | None], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM attempts WHERE capability_id=? ORDER BY call_index',
                (capability_id,),
            ).fetchall()
        values: list[tuple[GatewayAttemptReceipt, AgenticModelExchange | None]] = []
        for row in rows:
            if row['state'] != 'completed' or row['attempt_json'] is None:
                raise ValueError('gateway attempt log contains an unaccounted call reservation')
            attempt = GatewayAttemptReceipt.model_validate_json(bytes(row['attempt_json']))
            exchange = (
                None
                if row['exchange_json'] is None
                else AgenticModelExchange.model_validate_json(bytes(row['exchange_json']))
            )
            values.append((attempt, exchange))
        return tuple(values)


class AuthenticatedProviderGateway:
    """Host-side gateway core. Transport code supplies the observed vsock peer CID."""

    def __init__(
        self,
        *,
        policy: AuthenticatedGatewayPolicy,
        ledger: SqliteGatewayLedger,
        secret_resolver: GatewaySecretResolver,
        adapters: tuple[ProviderAdapter, ...],
        receipt_key: bytes,
    ) -> None:
        if gateway_session_key_id(receipt_key) != policy.receipt_key_id:
            raise ValueError('gateway receipt key does not match the pinned policy key ID')
        descriptors = tuple(adapter.descriptor for adapter in adapters)
        identities = tuple((item.provider, item.adapter_id, item.adapter_version) for item in descriptors)
        if len(identities) != len(set(identities)):
            raise ValueError('provider adapters must use unique provider/adapter identities')
        self.policy = policy
        self.ledger = ledger
        self.secret_resolver = secret_resolver
        self._adapters = {
            (adapter.descriptor.provider, adapter.descriptor.adapter_id, adapter.descriptor.adapter_version): adapter
            for adapter in adapters
        }
        self._receipt_key = bytes(receipt_key)
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    @property
    def provider_calls_forcibly_cancellable(self) -> bool:
        """Whether every configured adapter uses the qualified one-shot child boundary."""

        return bool(self._adapters) and all(
            getattr(adapter, 'forcibly_cancellable_provider_calls', False) is True
            and getattr(adapter, 'provider_credentials_child_side', False) is True
            for adapter in self._adapters.values()
        )

    def register_session(
        self,
        *,
        grant: GatewayCapabilityGrant,
        route: GatewayModelRoute,
        secret: bytes,
    ) -> None:
        if gateway_capability_id(secret) != grant.capability_id:
            raise ValueError('gateway capability secret does not match the grant')
        if grant.gateway_policy_sha256 != authenticated_gateway_policy_sha256(self.policy):
            raise ValueError('gateway capability grant does not bind this gateway policy')
        if grant.model_route_sha256 != gateway_model_route_sha256(route):
            raise ValueError('gateway capability grant does not bind the supplied model route')
        if grant.maximum_session_wire_bytes != self.policy.maximum_session_wire_bytes:
            raise ValueError('gateway capability wire budget does not match gateway policy')
        adapter = self._adapter(route)
        _validate_adapter_descriptor(adapter.descriptor, route)
        resolved_secret = self.secret_resolver.resolve(grant.capability_id)
        if not hmac.compare_digest(resolved_secret, secret):
            raise ValueError('gateway secret resolver returned a different capability secret')
        self.ledger.register(grant, route, self.policy)

    def handle_frame(
        self,
        frame: bytes,
        *,
        peer_cid: int,
        observed_at: datetime | None = None,
    ) -> bytes:
        capability_id = peek_gateway_frame_capability_id(frame, maximum_body_bytes=self.policy.maximum_frame_body_bytes)
        with self.ledger.capability_admission_guard():
            return self._handle_frame_under_admission_guard(
                frame,
                capability_id=capability_id,
                peer_cid=peer_cid,
                observed_at=observed_at,
            )

    def _handle_frame_under_admission_guard(
        self,
        frame: bytes,
        *,
        capability_id: str,
        peer_cid: int,
        observed_at: datetime | None,
    ) -> bytes:
        """Admit and finish one local call while durable revocation is excluded."""

        self.ledger.require_capability_admission(capability_id)
        secret = self.secret_resolver.resolve(capability_id)
        wire_request = decode_gateway_frame(
            frame,
            GatewayWireRequest,
            secret=secret,
            direction='request',
            expected_capability_id=capability_id,
            maximum_body_bytes=self.policy.maximum_frame_body_bytes,
        )
        request = wire_request.request
        now = _aware(observed_at or datetime.now(UTC), 'observed_at')
        lock = self._session_lock(capability_id)
        with lock:
            try:
                session = self.ledger.load(capability_id)
                rejection = self._authenticated_preflight_error(
                    wire_request=wire_request,
                    session=session,
                    peer_cid=peer_cid,
                    now=now,
                )
                if rejection is not None:
                    return self._complete_authenticated_rejection(
                        frame=frame,
                        secret=secret,
                        session=session,
                        request=request,
                        code=rejection,
                    )
                return self._handle_authenticated_request(
                    frame=frame,
                    secret=secret,
                    session=session,
                    request=request,
                    now=now,
                )
            except AuthenticatedGatewayError as error:
                if error.code in {GatewayErrorCode.REPLAY_CONFLICT, GatewayErrorCode.AMBIGUOUS_IN_FLIGHT}:
                    self.ledger.fail_session(capability_id, error.code)
                return self._error_frame(secret, request, error.code)

    def _authenticated_preflight_error(
        self,
        *,
        wire_request: GatewayWireRequest,
        session: _LedgerSession,
        peer_cid: int,
        now: datetime,
    ) -> GatewayErrorCode | None:
        if wire_request.grant_sha256 != gateway_capability_grant_sha256(session.grant):
            return GatewayErrorCode.UNAUTHORIZED
        if peer_cid != session.grant.expected_peer_cid:
            return GatewayErrorCode.WRONG_PEER
        if now < session.grant.issued_at or now >= session.grant.expires_at:
            return GatewayErrorCode.EXPIRED
        if wire_request.request.run_id != session.grant.run_id:
            return GatewayErrorCode.WRONG_RUN
        if len(canonical_json_bytes(wire_request.request)) > session.grant.maximum_request_bytes:
            return GatewayErrorCode.INVALID_REQUEST
        return None

    def _complete_authenticated_rejection(
        self,
        *,
        frame: bytes,
        secret: bytes,
        session: _LedgerSession,
        request: AgenticModelRequest,
        code: GatewayErrorCode,
    ) -> bytes:
        reservation = self.ledger.reserve(
            session.grant.capability_id,
            request,
            request_frame_bytes=len(frame),
        )
        if reservation.cached_response is not None:
            self.ledger.fail_session(session.grant.capability_id, code)
            return self._error_frame(secret, request, code)
        request_bytes = canonical_json_bytes(request)
        return self._complete_failure(
            secret=secret,
            session=session,
            request=request,
            request_sha256=_sha256(request_bytes),
            request_bytes=len(request_bytes),
            request_frame_bytes=len(frame),
            code=GatewayErrorCode.AMBIGUOUS_IN_FLIGHT if reservation.dispatch_ambiguous else code,
            provider_result=None,
            provider_dispatched=None if reservation.dispatch_ambiguous else False,
        )

    def _handle_authenticated_request(
        self,
        *,
        frame: bytes,
        secret: bytes,
        session: _LedgerSession,
        request: AgenticModelRequest,
        now: datetime,
    ) -> bytes:
        reservation = self.ledger.reserve(
            session.grant.capability_id,
            request,
            request_frame_bytes=len(frame),
        )
        if reservation.cached_response is not None:
            encoded = self._encode_response(secret, session.grant.capability_id, reservation.cached_response)
            terminal_error = self._error_frame(secret, request, GatewayErrorCode.BUDGET_EXHAUSTED)
            if not self.ledger.charge_cached_replay(
                capability_id=session.grant.capability_id,
                request=request,
                admitted_wire_bytes_delta=len(frame) + len(encoded),
                terminal_rejection_wire_bytes=len(frame) + len(terminal_error),
            ):
                return terminal_error
            return encoded
        request_bytes = canonical_json_bytes(request)
        request_sha256 = _sha256(request_bytes)
        if reservation.dispatch_ambiguous:
            return self._complete_failure(
                secret=secret,
                session=session,
                request=request,
                request_sha256=request_sha256,
                request_bytes=len(request_bytes),
                request_frame_bytes=len(frame),
                code=GatewayErrorCode.AMBIGUOUS_IN_FLIGHT,
                provider_result=None,
                provider_dispatched=None,
            )
        maximum_response_frame_bytes = maximum_gateway_frame_bytes(self.policy.maximum_frame_body_bytes)
        if (
            session.admitted_wire_bytes + len(frame) + maximum_response_frame_bytes
            > session.grant.maximum_session_wire_bytes
        ):
            return self._complete_failure(
                secret=secret,
                session=session,
                request=request,
                request_sha256=request_sha256,
                request_bytes=len(request_bytes),
                request_frame_bytes=len(frame),
                code=GatewayErrorCode.BUDGET_EXHAUSTED,
                provider_result=None,
                provider_dispatched=False,
            )
        adapter = self._adapter(session.route)
        provider_dispatched = False
        try:
            estimated_input = adapter.estimate_input_tokens(request, session.route)
            if estimated_input < 0:
                raise AuthenticatedGatewayError(GatewayErrorCode.INTERNAL)
            if (
                request.max_output_tokens > session.route.max_output_tokens
                or request.max_output_tokens > session.grant.limits.max_output_tokens
                or session.input_tokens + estimated_input > session.grant.limits.max_input_tokens
                or estimated_input + request.max_output_tokens > session.route.max_context_tokens
                or session.output_tokens + request.max_output_tokens > session.grant.limits.max_output_tokens
                or request.call_index >= session.grant.limits.max_model_calls
            ):
                return self._complete_failure(
                    secret=secret,
                    session=session,
                    request=request,
                    request_sha256=request_sha256,
                    request_bytes=len(request_bytes),
                    request_frame_bytes=len(frame),
                    code=GatewayErrorCode.BUDGET_EXHAUSTED,
                    provider_result=None,
                    provider_dispatched=False,
                )
            if not self.ledger.reserve_provider_dispatch(
                capability_id=session.grant.capability_id,
                call_index=request.call_index,
                maximum_exchange_wire_bytes=len(frame) + maximum_response_frame_bytes,
            ):
                return self._complete_failure(
                    secret=secret,
                    session=session,
                    request=request,
                    request_sha256=request_sha256,
                    request_bytes=len(request_bytes),
                    request_frame_bytes=len(frame),
                    code=GatewayErrorCode.BUDGET_EXHAUSTED,
                    provider_result=None,
                    provider_dispatched=False,
                )
            remaining_seconds = min(
                self.policy.maximum_provider_call_seconds,
                max(0.0, (session.grant.expires_at - now).total_seconds()),
            )
            provider_dispatched = True
            result = adapter.generate(request, session.route, timeout_seconds=remaining_seconds)
        except ProviderCallFailure as error:
            return self._complete_failure(
                secret=secret,
                session=session,
                request=request,
                request_sha256=request_sha256,
                request_bytes=len(request_bytes),
                request_frame_bytes=len(frame),
                code=_map_provider_failure(error.code),
                provider_result=None,
                provider_dispatched=True,
            )
        except AuthenticatedGatewayError:
            raise
        except BaseException:
            return self._complete_failure(
                secret=secret,
                session=session,
                request=request,
                request_sha256=request_sha256,
                request_bytes=len(request_bytes),
                request_frame_bytes=len(frame),
                code=GatewayErrorCode.INTERNAL,
                provider_result=None,
                provider_dispatched=provider_dispatched,
            )

        validation_error = _validate_provider_result(
            result=result,
            request=request,
            route=session.route,
            session=session,
        )
        if validation_error is not None:
            return self._complete_failure(
                secret=secret,
                session=session,
                request=request,
                request_sha256=request_sha256,
                request_bytes=len(request_bytes),
                request_frame_bytes=len(frame),
                code=validation_error,
                provider_result=result,
                provider_dispatched=True,
            )
        response = AgenticModelResponse(
            run_id=request.run_id,
            call_index=request.call_index,
            resolved_model_id=session.route.resolved_model_id,
            content=result.content,
            stop_reason=result.stop_reason,
            usage=result.usage,
        )
        response_bytes = canonical_json_bytes(response)
        receipt = AgenticModelCallReceipt(
            run_id=request.run_id,
            call_index=request.call_index,
            request_sha256=request_sha256,
            response_sha256=_sha256(response_bytes),
            resolved_model_id=session.route.resolved_model_id,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            reasoning_tokens=result.usage.reasoning_tokens,
            stop_reason=result.stop_reason,
        )
        exchange = AgenticModelExchange(request=request, response=response, receipt=receipt)
        wire_response = GatewayWireResponse(
            run_id=request.run_id,
            call_index=request.call_index,
            succeeded=True,
            response=response,
        )
        try:
            encoded = self._encode_response(secret, session.grant.capability_id, wire_response)
        except GatewayFrameError:
            return self._complete_failure(
                secret=secret,
                session=session,
                request=request,
                request_sha256=request_sha256,
                request_bytes=len(request_bytes),
                request_frame_bytes=len(frame),
                code=GatewayErrorCode.PROVIDER_PROTOCOL,
                provider_result=result,
                provider_dispatched=True,
            )
        attempt = GatewayAttemptReceipt(
            run_id=session.grant.run_id,
            capability_id=session.grant.capability_id,
            call_index=request.call_index,
            request_sha256=request_sha256,
            request_bytes=len(request_bytes),
            succeeded=True,
            response_sha256=_sha256(response_bytes),
            response_bytes=len(response_bytes),
            provider_dispatched=True,
            provider_result=result,
            exchange_sha256=_sha256(canonical_json_bytes(exchange)),
            admitted_wire_bytes=len(frame) + len(encoded),
        )
        self.ledger.complete(
            capability_id=session.grant.capability_id,
            wire_response=wire_response,
            attempt=attempt,
            exchange=exchange,
        )
        return encoded

    def _complete_failure(
        self,
        *,
        secret: bytes,
        session: _LedgerSession,
        request: AgenticModelRequest,
        request_sha256: str,
        request_bytes: int,
        request_frame_bytes: int,
        code: GatewayErrorCode,
        provider_result: ProviderCallResult | None,
        provider_dispatched: bool | None,
    ) -> bytes:
        def rejection(error_code: GatewayErrorCode) -> tuple[GatewayWireResponse, bytes]:
            response = GatewayWireResponse(
                run_id=request.run_id,
                call_index=request.call_index,
                succeeded=False,
                error_code=error_code,
                error_message='gateway request rejected',
            )
            return response, self._encode_response(secret, session.grant.capability_id, response)

        wire_response, encoded = rejection(code)
        exchange_wire_bytes = request_frame_bytes + len(encoded)
        terminal_rejection_wire_bytes = 0
        admitted_wire_bytes = exchange_wire_bytes
        if session.admitted_wire_bytes + exchange_wire_bytes > session.grant.maximum_session_wire_bytes:
            if provider_dispatched is not False:
                raise AuthenticatedGatewayError(GatewayErrorCode.AMBIGUOUS_IN_FLIGHT)
            code = GatewayErrorCode.BUDGET_EXHAUSTED
            wire_response, encoded = rejection(code)
            exchange_wire_bytes = request_frame_bytes + len(encoded)
            admitted_wire_bytes = 0
            terminal_rejection_wire_bytes = exchange_wire_bytes
        attempt = GatewayAttemptReceipt(
            run_id=session.grant.run_id,
            capability_id=session.grant.capability_id,
            call_index=request.call_index,
            request_sha256=request_sha256,
            request_bytes=request_bytes,
            succeeded=False,
            error_code=code,
            provider_dispatched=provider_dispatched,
            provider_result=provider_result,
            admitted_wire_bytes=admitted_wire_bytes,
            terminal_budget_rejection_wire_bytes=terminal_rejection_wire_bytes,
        )
        self.ledger.complete(
            capability_id=session.grant.capability_id,
            wire_response=wire_response,
            attempt=attempt,
            exchange=None,
        )
        return encoded

    def seal_session(
        self,
        capability_id: str,
        *,
        terminal_reason: GatewayTerminalReason,
        sealed_at: datetime | None = None,
        revoke_secret: bool = True,
    ) -> AuthenticatedGatewaySession:
        with self._session_lock(capability_id):
            return self._seal_session_locked(
                capability_id,
                terminal_reason=terminal_reason,
                sealed_at=sealed_at,
                revoke_secret=revoke_secret,
            )

    def revoke_capability(
        self,
        capability_id: str,
        *,
        reason: GatewayCapabilityRevocationReason,
        revoked_at: datetime | None = None,
    ) -> GatewayCapabilityRevocation:
        """Durably stop local admission, then clear the configured volatile secret.

        Callers must first stop/reap the worker which can originate new requests.  This method
        does not revoke the provider adapter's API credential or cancel a remote request which a
        crashed gateway process may already have dispatched.
        """

        binding = self.ledger.capability_binding(capability_id)
        revocation = self.ledger.revoke_capability(
            capability_id=capability_id,
            expected_run_id=binding.run_id,
            expected_attempt_reservation_sha256=(binding.attempt_reservation_sha256),
            expected_model_route_sha256=binding.model_route_sha256,
            reason=reason,
            revoked_at=revoked_at or datetime.now(UTC),
        )
        if isinstance(self.secret_resolver, RevocableGatewaySecretResolver):
            self.secret_resolver.revoke(capability_id)
        return revocation

    def revoke_unregistered_capability(
        self,
        capability_id: str,
        *,
        run_id: str,
        attempt_reservation_sha256: str,
        model_route_sha256: str,
        reason: GatewayCapabilityRevocationReason,
        revoked_at: datetime | None = None,
    ) -> GatewayCapabilityRevocation:
        """Durably tombstone an exact capability intent which never registered a session."""

        revocation = self.ledger.revoke_unregistered_capability(
            capability_id=capability_id,
            expected_run_id=run_id,
            expected_attempt_reservation_sha256=attempt_reservation_sha256,
            expected_model_route_sha256=model_route_sha256,
            reason=reason,
            revoked_at=revoked_at or datetime.now(UTC),
        )
        if isinstance(self.secret_resolver, RevocableGatewaySecretResolver):
            self.secret_resolver.revoke(capability_id)
        return revocation

    def _seal_session_locked(
        self,
        capability_id: str,
        *,
        terminal_reason: GatewayTerminalReason,
        sealed_at: datetime | None,
        revoke_secret: bool,
    ) -> AuthenticatedGatewaySession:
        existing = self.ledger.sealed_session(capability_id)
        if existing is not None:
            if (
                existing.seal.terminal_reason != GatewayTerminalReason.FAILED
                and existing.seal.terminal_reason != terminal_reason
            ):
                raise ValueError('gateway session was already sealed with a different terminal reason')
            verify_authenticated_gateway_session(
                existing,
                receipt_key=self._receipt_key,
                expected_receipt_key_id=self.policy.receipt_key_id,
            )
            if revoke_secret:
                self._revoke_sealed_capability(existing)
            return existing
        session = self.ledger.close(capability_id, terminal_reason)
        pairs = self.ledger.attempts(capability_id)
        attempts = tuple(attempt for attempt, _ in pairs)
        exchanges = tuple(exchange for _, exchange in pairs if exchange is not None)
        transcript = AgenticGatewayTranscript(
            run_id=session.grant.run_id,
            resolved_model_id=session.route.resolved_model_id if exchanges else None,
            exchanges=exchanges,
            input_tokens=sum(exchange.response.usage.input_tokens for exchange in exchanges),
            output_tokens=sum(exchange.response.usage.output_tokens for exchange in exchanges),
            reasoning_tokens=sum((exchange.response.usage.reasoning_tokens or 0) for exchange in exchanges),
        )
        actual_terminal = session.terminal_reason
        if actual_terminal is None:
            raise ValueError('gateway ledger did not persist a terminal reason')
        if session.dispatch_reserved_wire_bytes:
            raise ValueError('gateway cannot seal with outstanding provider dispatch capacity')
        terminal_error = session.terminal_error_code
        if actual_terminal == GatewayTerminalReason.FAILED:
            if terminal_error is None:
                terminal_error = GatewayErrorCode.INTERNAL
        if actual_terminal == GatewayTerminalReason.COMPLETED and any(not attempt.succeeded for attempt in attempts):
            raise ValueError('completed gateway sessions cannot contain failed attempts')
        seal = GatewaySessionSeal(
            run_id=session.grant.run_id,
            capability_id=capability_id,
            grant_sha256=gateway_capability_grant_sha256(session.grant),
            attempt_reservation_sha256=session.grant.attempt_reservation_sha256,
            execution_policy_sha256=session.grant.execution_policy_sha256,
            workspace_manifest_sha256=session.grant.workspace_manifest_sha256,
            gateway_policy_sha256=authenticated_gateway_policy_sha256(session.policy),
            model_route_sha256=gateway_model_route_sha256(session.route),
            gateway_id=session.policy.gateway_id,
            gateway_version=session.policy.gateway_version,
            gateway_executable_sha256=session.policy.gateway_executable_sha256,
            gateway_config_sha256=session.policy.gateway_config_sha256,
            model_registry_sha256=session.policy.model_registry_sha256,
            receipt_key_id=session.policy.receipt_key_id,
            transcript_sha256=_sha256(canonical_json_bytes(transcript)),
            attempt_log_sha256=_sha256(canonical_json_bytes([attempt.model_dump(mode='json') for attempt in attempts])),
            terminal_reason=actual_terminal,
            terminal_error_code=terminal_error,
            attempt_count=len(attempts),
            successful_call_count=len(exchanges),
            input_tokens=session.input_tokens,
            output_tokens=session.output_tokens,
            reasoning_tokens=session.reasoning_tokens,
            provider_cost_usd=session.provider_cost_usd,
            admitted_wire_bytes=session.admitted_wire_bytes,
            terminal_budget_rejection_wire_bytes=session.terminal_budget_rejection_wire_bytes,
            terminal_observed_overage_bytes=max(
                0,
                session.admitted_wire_bytes
                + session.terminal_budget_rejection_wire_bytes
                - session.grant.maximum_session_wire_bytes,
            ),
            exact_replay_count=session.exact_replay_count,
            issued_at=session.grant.issued_at,
            sealed_at=_aware(sealed_at or datetime.now(UTC), 'sealed_at'),
        )
        artifact = AuthenticatedGatewaySession(
            grant=session.grant,
            route=session.route,
            policy=session.policy,
            transcript=transcript,
            attempts=attempts,
            seal=seal,
            seal_hmac=gateway_session_seal_hmac(seal, self._receipt_key),
        )
        verify_authenticated_gateway_session(
            artifact,
            receipt_key=self._receipt_key,
            expected_receipt_key_id=session.policy.receipt_key_id,
        )
        self.ledger.store_sealed_session(artifact)
        if revoke_secret:
            self._revoke_sealed_capability(artifact)
        return artifact

    def _revoke_sealed_capability(self, artifact: AuthenticatedGatewaySession) -> None:
        self.revoke_capability(
            artifact.grant.capability_id,
            reason=GatewayCapabilityRevocationReason.SESSION_SEALED,
            revoked_at=artifact.seal.sealed_at,
        )

    def _adapter(self, route: GatewayModelRoute) -> ProviderAdapter:
        key = (route.provider, route.adapter_id, route.adapter_version)
        adapter = self._adapters.get(key)
        if adapter is None:
            raise ValueError('gateway model route references an unavailable provider adapter')
        return adapter

    def _session_lock(self, capability_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(capability_id, threading.Lock())

    def _error_frame(
        self,
        secret: bytes,
        request: AgenticModelRequest,
        code: GatewayErrorCode,
    ) -> bytes:
        return self._encode_response(
            secret,
            gateway_capability_id(secret),
            GatewayWireResponse(
                run_id=request.run_id,
                call_index=request.call_index,
                succeeded=False,
                error_code=code,
                error_message='gateway request rejected',
            ),
        )

    def _encode_response(
        self,
        secret: bytes,
        capability_id: str,
        response: GatewayWireResponse,
    ) -> bytes:
        return encode_gateway_frame(
            response,
            capability_id=capability_id,
            secret=secret,
            direction='response',
            maximum_body_bytes=self.policy.maximum_frame_body_bytes,
        )


def verify_authenticated_gateway_session(
    artifact: AuthenticatedGatewaySession,
    *,
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> None:
    if gateway_session_key_id(receipt_key) != expected_receipt_key_id:
        raise ValueError('gateway receipt key does not match the expected key ID')
    if artifact.policy.receipt_key_id != expected_receipt_key_id:
        raise ValueError('gateway session policy uses a different receipt key')
    if artifact.grant.gateway_policy_sha256 != authenticated_gateway_policy_sha256(artifact.policy):
        raise ValueError('gateway capability grant does not bind the authenticated gateway policy')
    if artifact.grant.model_route_sha256 != gateway_model_route_sha256(artifact.route):
        raise ValueError('gateway capability grant does not bind the authenticated model route')
    if artifact.grant.maximum_session_wire_bytes != artifact.policy.maximum_session_wire_bytes:
        raise ValueError('gateway capability grant wire budget does not match authenticated policy')
    if not hmac.compare_digest(artifact.seal_hmac, gateway_session_seal_hmac(artifact.seal, receipt_key)):
        raise ValueError('gateway session seal authentication failed')
    expected_bindings = (
        artifact.grant.run_id,
        artifact.grant.capability_id,
        gateway_capability_grant_sha256(artifact.grant),
        artifact.grant.attempt_reservation_sha256,
        artifact.grant.execution_policy_sha256,
        artifact.grant.workspace_manifest_sha256,
        authenticated_gateway_policy_sha256(artifact.policy),
        gateway_model_route_sha256(artifact.route),
        artifact.policy.gateway_id,
        artifact.policy.gateway_version,
        artifact.policy.gateway_executable_sha256,
        artifact.policy.gateway_config_sha256,
        artifact.policy.model_registry_sha256,
        artifact.policy.receipt_key_id,
        _sha256(canonical_json_bytes(artifact.transcript)),
        _sha256(canonical_json_bytes([attempt.model_dump(mode='json') for attempt in artifact.attempts])),
    )
    actual_bindings = (
        artifact.seal.run_id,
        artifact.seal.capability_id,
        artifact.seal.grant_sha256,
        artifact.seal.attempt_reservation_sha256,
        artifact.seal.execution_policy_sha256,
        artifact.seal.workspace_manifest_sha256,
        artifact.seal.gateway_policy_sha256,
        artifact.seal.model_route_sha256,
        artifact.seal.gateway_id,
        artifact.seal.gateway_version,
        artifact.seal.gateway_executable_sha256,
        artifact.seal.gateway_config_sha256,
        artifact.seal.model_registry_sha256,
        artifact.seal.receipt_key_id,
        artifact.seal.transcript_sha256,
        artifact.seal.attempt_log_sha256,
    )
    if actual_bindings != expected_bindings:
        raise ValueError('gateway session seal does not bind its exact grant, route, policy, and logs')
    if artifact.seal.issued_at != artifact.grant.issued_at:
        raise ValueError('gateway session seal issuance does not match its capability grant')
    indices = tuple(attempt.call_index for attempt in artifact.attempts)
    if indices != tuple(range(len(indices))):
        raise ValueError('gateway attempts must use contiguous call indexes')
    if any(
        (attempt.run_id, attempt.capability_id) != (artifact.grant.run_id, artifact.grant.capability_id)
        for attempt in artifact.attempts
    ):
        raise ValueError('gateway attempt belongs to a different session')
    successful = tuple(attempt for attempt in artifact.attempts if attempt.succeeded)
    if len(successful) != len(artifact.transcript.exchanges):
        raise ValueError('gateway transcript does not cover every successful attempt')
    for attempt, exchange in zip(successful, artifact.transcript.exchanges, strict=True):
        if (
            attempt.call_index,
            attempt.request_sha256,
            attempt.response_sha256,
            attempt.exchange_sha256,
        ) != (
            exchange.request.call_index,
            exchange.receipt.request_sha256,
            exchange.receipt.response_sha256,
            _sha256(canonical_json_bytes(exchange)),
        ):
            raise ValueError('gateway successful attempt does not match its transcript exchange')
        result = attempt.provider_result
        if result is None or (
            result.resolved_model_id != artifact.route.resolved_model_id
            or result.provider_reported_model_id != artifact.route.resolved_model_id
        ):
            raise ValueError('gateway provider did not attest the pinned resolved model')
        if not (
            artifact.grant.issued_at <= result.started_at < artifact.grant.expires_at
            and result.finished_at <= artifact.grant.expires_at
            and result.finished_at <= artifact.seal.sealed_at
            and (result.finished_at - result.started_at).total_seconds()
            <= artifact.policy.maximum_provider_call_seconds
        ):
            raise ValueError('gateway provider evidence falls outside the authenticated session interval')
    totals = (
        len(artifact.attempts),
        len(successful),
        sum(
            (attempt.provider_result.usage.input_tokens if attempt.provider_result else 0)
            for attempt in artifact.attempts
        ),
        sum(
            (attempt.provider_result.usage.output_tokens if attempt.provider_result else 0)
            for attempt in artifact.attempts
        ),
        sum(
            ((attempt.provider_result.usage.reasoning_tokens or 0) if attempt.provider_result else 0)
            for attempt in artifact.attempts
        ),
    )
    if totals != (
        artifact.seal.attempt_count,
        artifact.seal.successful_call_count,
        artifact.seal.input_tokens,
        artifact.seal.output_tokens,
        artifact.seal.reasoning_tokens,
    ):
        raise ValueError('gateway session totals do not match authenticated attempts')
    costs = [attempt.provider_result.provider_cost_usd for attempt in artifact.attempts if attempt.provider_result]
    expected_cost = None if all(cost is None for cost in costs) else sum(cost or 0.0 for cost in costs)
    if artifact.seal.provider_cost_usd != expected_cost:
        raise ValueError('gateway session provider cost does not match authenticated attempts')
    admitted_wire_bytes = sum(attempt.admitted_wire_bytes for attempt in artifact.attempts)
    terminal_rejection_wire_bytes = sum(attempt.terminal_budget_rejection_wire_bytes for attempt in artifact.attempts)
    expected_overage = max(
        0,
        admitted_wire_bytes + terminal_rejection_wire_bytes - artifact.grant.maximum_session_wire_bytes,
    )
    if (
        artifact.seal.admitted_wire_bytes != admitted_wire_bytes
        or artifact.seal.terminal_budget_rejection_wire_bytes != terminal_rejection_wire_bytes
        or artifact.seal.terminal_observed_overage_bytes != expected_overage
        or artifact.seal.exact_replay_count != sum(attempt.exact_replay_count for attempt in artifact.attempts)
    ):
        raise ValueError('gateway session wire totals do not match authenticated attempts')
    if admitted_wire_bytes > artifact.grant.maximum_session_wire_bytes:
        raise ValueError('gateway admitted wire bytes exceed the authenticated dispatch budget')
    terminal_rejections = tuple(
        attempt for attempt in artifact.attempts if attempt.terminal_budget_rejection_wire_bytes
    )
    if terminal_rejections:
        if (
            len(terminal_rejections) != 1
            or terminal_rejections[0].call_index != artifact.attempts[-1].call_index
            or artifact.seal.terminal_reason != GatewayTerminalReason.FAILED
            or artifact.seal.terminal_error_code != GatewayErrorCode.BUDGET_EXHAUSTED
            or expected_overage <= 0
            or terminal_rejection_wire_bytes > 2 * maximum_gateway_frame_bytes(artifact.policy.maximum_frame_body_bytes)
        ):
            raise ValueError('gateway terminal wire overage is not one bounded budget rejection')
    elif artifact.seal.terminal_observed_overage_bytes != 0:
        raise ValueError('gateway seal claims terminal wire overage without a bounded rejection')
    if artifact.seal.terminal_reason == GatewayTerminalReason.COMPLETED and any(
        not attempt.succeeded for attempt in artifact.attempts
    ):
        raise ValueError('completed gateway session contains a failed attempt')
    if any(attempt.provider_dispatched is None for attempt in artifact.attempts) and (
        artifact.seal.terminal_reason != GatewayTerminalReason.FAILED
        or artifact.seal.terminal_error_code != GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
    ):
        raise ValueError('ambiguous in-flight evidence requires an ambiguous failed gateway session')


def build_gateway_request_frame(
    grant: GatewayCapabilityGrant,
    request: AgenticModelRequest,
    *,
    secret: bytes,
    maximum_body_bytes: int = DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES,
) -> bytes:
    return encode_gateway_frame(
        GatewayWireRequest(grant_sha256=gateway_capability_grant_sha256(grant), request=request),
        capability_id=grant.capability_id,
        secret=secret,
        direction='request',
        maximum_body_bytes=maximum_body_bytes,
    )


def parse_gateway_response_frame(
    frame: bytes,
    grant: GatewayCapabilityGrant,
    *,
    secret: bytes,
    maximum_body_bytes: int = DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES,
) -> GatewayWireResponse:
    return decode_gateway_frame(
        frame,
        GatewayWireResponse,
        secret=secret,
        direction='response',
        expected_capability_id=grant.capability_id,
        maximum_body_bytes=maximum_body_bytes,
    )


def _session_from_row(row: sqlite3.Row) -> _LedgerSession:
    terminal = None if row['terminal_error_code'] is None else GatewayErrorCode(row['terminal_error_code'])
    terminal_reason = None if row['terminal_reason'] is None else GatewayTerminalReason(row['terminal_reason'])
    session = _LedgerSession(
        grant=GatewayCapabilityGrant.model_validate_json(bytes(row['grant_json'])),
        route=GatewayModelRoute.model_validate_json(bytes(row['route_json'])),
        policy=AuthenticatedGatewayPolicy.model_validate_json(bytes(row['policy_json'])),
        status=str(row['status']),
        next_index=int(row['next_index']),
        input_tokens=int(row['input_tokens']),
        output_tokens=int(row['output_tokens']),
        reasoning_tokens=int(row['reasoning_tokens']),
        provider_cost_usd=None if row['provider_cost_usd'] is None else float(row['provider_cost_usd']),
        admitted_wire_bytes=int(row['admitted_wire_bytes']),
        terminal_budget_rejection_wire_bytes=int(row['terminal_budget_rejection_wire_bytes']),
        dispatch_reserved_wire_bytes=int(row['dispatch_reserved_wire_bytes']),
        exact_replay_count=int(row['exact_replay_count']),
        terminal_error_code=terminal,
        terminal_reason=terminal_reason,
    )
    _validate_loaded_session_bindings(session)
    return session


def _validate_loaded_session_bindings(session: _LedgerSession) -> None:
    if session.grant.gateway_policy_sha256 != authenticated_gateway_policy_sha256(session.policy):
        raise ValueError('gateway ledger grant does not bind its stored policy')
    if session.grant.model_route_sha256 != gateway_model_route_sha256(session.route):
        raise ValueError('gateway ledger grant does not bind its stored model route')
    if session.grant.maximum_session_wire_bytes != session.policy.maximum_session_wire_bytes:
        raise ValueError('gateway ledger grant wire budget does not match its stored policy')


def _validate_adapter_descriptor(descriptor: ProviderAdapterDescriptor, route: GatewayModelRoute) -> None:
    if (
        descriptor.provider,
        descriptor.adapter_id,
        descriptor.adapter_version,
        descriptor.executable_sha256,
        descriptor.config_sha256,
    ) != (
        route.provider,
        route.adapter_id,
        route.adapter_version,
        route.adapter_executable_sha256,
        route.adapter_config_sha256,
    ):
        raise ValueError('provider adapter does not match the committed model route')


def _validate_provider_result(
    *,
    result: ProviderCallResult,
    request: AgenticModelRequest,
    route: GatewayModelRoute,
    session: _LedgerSession,
) -> GatewayErrorCode | None:
    if (
        result.started_at < session.grant.issued_at
        or result.started_at >= session.grant.expires_at
        or result.finished_at > session.grant.expires_at
        or (result.finished_at - result.started_at).total_seconds() > session.policy.maximum_provider_call_seconds
    ):
        return GatewayErrorCode.PROVIDER_PROTOCOL
    if (
        result.resolved_model_id != route.resolved_model_id
        or result.provider_reported_model_id != route.resolved_model_id
    ):
        return GatewayErrorCode.MODEL_FORBIDDEN
    usage = result.usage
    if usage.output_tokens > request.max_output_tokens:
        return GatewayErrorCode.PROVIDER_PROTOCOL
    if route.reasoning_accounting == 'reported' and usage.reasoning_tokens is None:
        return GatewayErrorCode.PROVIDER_PROTOCOL
    if route.reasoning_accounting == 'not_applicable' and usage.reasoning_tokens not in {None, 0}:
        return GatewayErrorCode.PROVIDER_PROTOCOL
    if (
        session.input_tokens + usage.input_tokens > session.grant.limits.max_input_tokens
        or session.output_tokens + usage.output_tokens > session.grant.limits.max_output_tokens
        or session.reasoning_tokens + (usage.reasoning_tokens or 0) > session.grant.limits.max_reasoning_tokens
        or usage.input_tokens + request.max_output_tokens > route.max_context_tokens
    ):
        return GatewayErrorCode.BUDGET_EXHAUSTED
    return None


def _map_provider_failure(code: ProviderFailureCode) -> GatewayErrorCode:
    return {
        ProviderFailureCode.TIMEOUT: GatewayErrorCode.PROVIDER_TIMEOUT,
        ProviderFailureCode.RATE_LIMIT: GatewayErrorCode.PROVIDER_RATE_LIMIT,
        ProviderFailureCode.REJECTED: GatewayErrorCode.PROVIDER_REJECTED,
        ProviderFailureCode.PROTOCOL: GatewayErrorCode.PROVIDER_PROTOCOL,
        ProviderFailureCode.INTERNAL: GatewayErrorCode.INTERNAL,
    }[code]


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(UTC)


def _validate_receipt_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError('gateway receipt key must contain at least 32 bytes')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
