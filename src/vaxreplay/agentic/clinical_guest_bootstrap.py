"""Signed one-message bootstrap for the Lane A guest harness.

The transport is an already-connected, peer-scoped socket supplied by the microVM runtime.  This
module does not open a socket, launch a worker, read a file, or contact a provider.  After the
one-time start redemption, the launcher signs one exact canonical hello.  Before ordinary guest RPC
begins, the host sends that bounded authorization and the guest verifies it against an independently
supplied public-key trust anchor and static image pins.  The guest returns one canonical
acknowledgement bound to the authorization, and the same connection is handed to ``GuestRpcClient``.

The authenticated receipt is a host observation, not remote attestation and not proof that this
code was baked into a qualified image.  Production composition must separately bind it into the
worker and outer run receipts.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import socket
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal, Protocol, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_ACTION_SCHEMA_SHA256,
    LANE_A_GUEST_HARNESS_POLICY,
    LANE_A_GUEST_HARNESS_POLICY_ID,
    LaneAGuestHarnessResult,
    run_lane_a_guest_harness,
)
from vaxreplay.agentic.guest_rpc import (
    GuestRpcClient,
    GuestRpcError,
    decode_guest_rpc_frame,
    encode_guest_rpc_frame,
    receive_guest_rpc_frame,
    send_guest_rpc_frame,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.task_protocol import AgenticTaskInvocation, agentic_task_invocation_sha256
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_task import ExecutionTask
from vaxreplay.operations.signing import Ed25519Signer, checked_signer

CLINICAL_GUEST_BOOTSTRAP_PROTOCOL_ID = 'lane-a-clinical-guest-bootstrap-dev-v0.3'
CLINICAL_GUEST_BOOTSTRAP_HELLO_SCHEMA_VERSION = 'vaxreplay.clinical-guest-bootstrap-hello.dev-v0.3'
CLINICAL_GUEST_BOOTSTRAP_SIGNED_HELLO_SCHEMA_VERSION = 'vaxreplay.clinical-guest-bootstrap-signed-hello.dev-v0.3'
CLINICAL_GUEST_BOOTSTRAP_TRUST_ANCHOR_SCHEMA_VERSION = 'vaxreplay.clinical-guest-bootstrap-trust-anchor.dev-v0.3'
CLINICAL_GUEST_BOOTSTRAP_ACK_SCHEMA_VERSION = 'vaxreplay.clinical-guest-bootstrap-ack.dev-v0.3'
CLINICAL_GUEST_BOOTSTRAP_CONTEXT_SCHEMA_VERSION = 'vaxreplay.clinical-guest-bootstrap-context.dev-v0.3'
CLINICAL_GUEST_BOOTSTRAP_RECEIPT_SCHEMA_VERSION = 'vaxreplay.clinical-guest-bootstrap-receipt.dev-v0.3'
AUTHENTICATED_CLINICAL_GUEST_BOOTSTRAP_SCHEMA_VERSION = 'vaxreplay.authenticated-clinical-guest-bootstrap.dev-v0.3'

CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES = 4 * 1024 * 1024
CLINICAL_GUEST_BOOTSTRAP_DEFAULT_TIMEOUT_SECONDS = 5.0
CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_TIMEOUT_SECONDS = 30.0
CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_VALIDITY_SECONDS = 300.0
CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256 = hashlib.sha256(
    canonical_json_bytes(LANE_A_GUEST_HARNESS_POLICY)
).hexdigest()

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_NONCE_PATTERN = r'^[0-9a-f]{64}$'
_RECEIPT_KEY_ID_DOMAIN = b'vaxreplay.clinical-guest-bootstrap-receipt-key-id.dev-v0.3\x00'
_RECEIPT_HMAC_DOMAIN = b'vaxreplay.clinical-guest-bootstrap-receipt.dev-v0.3\x00'
_AUTHORIZATION_KEY_ID_DOMAIN = b'vaxreplay.clinical-guest-bootstrap-authorization-key-id.dev-v0.3\x00'
_AUTHORIZATION_SIGNATURE_DOMAIN = b'vaxreplay.clinical-guest-bootstrap-authorization.dev-v0.3\x00'
_MINIMUM_RPC_FRAME_BODY_BYTES = 256 * 1024
_MAXIMUM_RPC_FRAME_BODY_BYTES = 64 * 1024 * 1024
_MAXIMUM_RPC_SESSION_WIRE_BYTES = 1024 * 1024 * 1024
_MINIMUM_RPC_REQUESTS = 2 * LANE_A_GUEST_HARNESS_POLICY.maximum_model_calls


class ClinicalGuestBootstrapFailureCode(str, enum.Enum):
    """Content-free terminal reasons safe to expose across the bootstrap boundary."""

    INVALID_CONFIGURATION = 'invalid_configuration'
    TIMEOUT = 'timeout'
    CONNECTION_FAILED = 'connection_failed'
    FRAME_REJECTED = 'frame_rejected'
    OUTSIDE_VALIDITY_WINDOW = 'outside_validity_window'
    WRONG_RUN = 'wrong_run'
    HELLO_BINDING_INVALID = 'hello_binding_invalid'
    AUTHORIZATION_INVALID = 'authorization_invalid'
    TRUST_ANCHOR_MISMATCH = 'trust_anchor_mismatch'
    ACK_BINDING_INVALID = 'ack_binding_invalid'
    REPLAY_REJECTED = 'replay_rejected'
    RECEIPT_AUTHENTICATION_FAILED = 'receipt_authentication_failed'


class ClinicalGuestBootstrapError(RuntimeError):
    """A stable error code without task, frame, socket, or exception details."""

    def __init__(self, code: ClinicalGuestBootstrapFailureCode):
        super().__init__(code.value)
        self.code = code


class ClinicalGuestRpcLimits(StrictModel):
    """Exact guest-visible subset of the host RPC policy required by the fixed Lane A loop."""

    maximum_frame_body_bytes: int = Field(
        ge=_MINIMUM_RPC_FRAME_BODY_BYTES,
        le=_MAXIMUM_RPC_FRAME_BODY_BYTES,
    )
    maximum_session_wire_bytes: int = Field(
        ge=2 * _MINIMUM_RPC_FRAME_BODY_BYTES,
        le=_MAXIMUM_RPC_SESSION_WIRE_BYTES,
    )
    maximum_requests: int = Field(ge=_MINIMUM_RPC_REQUESTS, le=100_000)
    maximum_list_entries: int = Field(
        ge=LANE_A_GUEST_HARNESS_POLICY.maximum_list_entries_per_action,
        le=1000,
    )
    maximum_read_bytes: int = Field(
        ge=LANE_A_GUEST_HARNESS_POLICY.maximum_read_bytes_per_action,
        le=16 * 1024 * 1024,
    )
    maximum_search_results: int = Field(
        ge=LANE_A_GUEST_HARNESS_POLICY.maximum_search_results_per_action,
        le=1000,
    )
    maximum_submission_bytes: int = Field(
        ge=LANE_A_GUEST_HARNESS_POLICY.maximum_action_response_bytes,
        le=16 * 1024 * 1024,
    )

    @model_validator(mode='after')
    def validate_envelope_capacity(self) -> Self:
        if self.maximum_session_wire_bytes < 2 * self.maximum_frame_body_bytes:
            raise ValueError('RPC session wire budget cannot hold one maximum request and response')
        if self.maximum_read_bytes > self.maximum_frame_body_bytes // 2:
            raise ValueError('RPC read limit cannot fit its bounded response envelope')
        if self.maximum_submission_bytes > self.maximum_frame_body_bytes // 2:
            raise ValueError('RPC submission limit cannot fit its bounded request envelope')
        return self


class ClinicalGuestBootstrapHello(StrictModel):
    """The host's complete, canonical authorization for one guest RPC session."""

    schema_version: Literal['vaxreplay.clinical-guest-bootstrap-hello.dev-v0.3'] = (
        CLINICAL_GUEST_BOOTSTRAP_HELLO_SCHEMA_VERSION
    )
    protocol_id: Literal['lane-a-clinical-guest-bootstrap-dev-v0.3'] = CLINICAL_GUEST_BOOTSTRAP_PROTOCOL_ID
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    start_redemption_sha256: str = Field(pattern=_SHA256_PATTERN)
    session_id: str = Field(pattern=_RUN_ID_PATTERN)
    task_invocation: AgenticTaskInvocation
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_bootstrap_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_policy_id: Literal['lane-a-benchmark-native-retrieval-agent-v0.1'] = LANE_A_GUEST_HARNESS_POLICY_ID
    harness_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    action_schema_sha256: str = Field(pattern=_SHA256_PATTERN)
    rpc_limits: ClinicalGuestRpcLimits
    nonce: str = Field(pattern=_NONCE_PATTERN)
    valid_from: datetime
    expires_at: datetime
    one_hello: Literal[True] = True
    one_ack: Literal[True] = True
    guest_rpc_before_ack_permitted: Literal[False] = False
    development_only: Literal[True] = True

    @field_validator('valid_from', 'expires_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode='after')
    def validate_bindings(self) -> Self:
        if not isinstance(self.task_invocation.task, ExecutionTask):
            raise ValueError('clinical guest bootstrap requires a Lane A execution task')
        if self.task_invocation.response_protocol != AgenticResponseProtocol.CLINICAL_EXECUTION:
            raise ValueError('clinical guest bootstrap requires the clinical response protocol')
        if agentic_task_invocation_sha256(self.task_invocation) != self.task_invocation_sha256:
            raise ValueError('hello task invocation hash does not bind the exact invocation')
        if self.task_invocation.workspace_manifest_sha256 != self.workspace_manifest_sha256:
            raise ValueError('hello workspace manifest differs from the task invocation')
        if self.harness_policy_sha256 != CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256:
            raise ValueError('hello does not pin the exact built-in harness policy')
        if self.action_schema_sha256 != LANE_A_GUEST_ACTION_SCHEMA_SHA256:
            raise ValueError('hello does not pin the exact built-in action schema')
        if self.expires_at < self.valid_from:
            raise ValueError('hello validity interval is inverted')
        if (self.expires_at - self.valid_from).total_seconds() > CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_VALIDITY_SECONDS:
            raise ValueError('hello validity interval exceeds the fixed maximum')
        return self


class ClinicalGuestBootstrapTrustAnchor(StrictModel):
    """Static guest trust supplied independently of the per-run wire authorization.

    A qualified image should bake this record (or byte-identical values) into its measured harness
    image.  The development module verifies the values but does not claim that image baking or
    remote attestation has occurred.
    """

    schema_version: Literal['vaxreplay.clinical-guest-bootstrap-trust-anchor.dev-v0.3'] = (
        CLINICAL_GUEST_BOOTSTRAP_TRUST_ANCHOR_SCHEMA_VERSION
    )
    authorization_key_id: str = Field(pattern=_SHA256_PATTERN)
    ed25519_public_key_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_bootstrap_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_policy_id: Literal['lane-a-benchmark-native-retrieval-agent-v0.1'] = LANE_A_GUEST_HARNESS_POLICY_ID
    harness_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    action_schema_sha256: str = Field(pattern=_SHA256_PATTERN)
    rpc_limits: ClinicalGuestRpcLimits
    response_protocol: Literal[AgenticResponseProtocol.CLINICAL_EXECUTION] = AgenticResponseProtocol.CLINICAL_EXECUTION
    dynamic_run_values_accepted_only_under_launcher_signature: Literal[True] = True
    development_only: Literal[True] = True
    measured_guest_image_bake_attested: Literal[False] = False
    remote_guest_identity_attested: Literal[False] = False

    @model_validator(mode='after')
    def validate_static_pins(self) -> Self:
        public_key = bytes.fromhex(self.ed25519_public_key_hex)
        if self.authorization_key_id != clinical_guest_bootstrap_authorization_key_id(public_key):
            raise ValueError('guest trust anchor key ID does not bind its Ed25519 public key')
        if self.harness_policy_sha256 != CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256:
            raise ValueError('guest trust anchor does not pin the exact built-in harness policy')
        if self.action_schema_sha256 != LANE_A_GUEST_ACTION_SCHEMA_SHA256:
            raise ValueError('guest trust anchor does not pin the exact built-in action schema')
        return self


class SignedClinicalGuestBootstrapHello(StrictModel):
    """One launcher-signed post-redemption authorization sent over the guest wire."""

    schema_version: Literal['vaxreplay.clinical-guest-bootstrap-signed-hello.dev-v0.3'] = (
        CLINICAL_GUEST_BOOTSTRAP_SIGNED_HELLO_SCHEMA_VERSION
    )
    protocol_id: Literal['lane-a-clinical-guest-bootstrap-dev-v0.3'] = CLINICAL_GUEST_BOOTSTRAP_PROTOCOL_ID
    authorization_key_id: str = Field(pattern=_SHA256_PATTERN)
    hello: ClinicalGuestBootstrapHello
    hello_sha256: str = Field(pattern=_SHA256_PATTERN)
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    signature_hex: str = Field(pattern=r'^[0-9a-f]{128}$')
    post_redemption_authorization: Literal[True] = True
    development_only: Literal[True] = True

    @model_validator(mode='after')
    def validate_hello_hash(self) -> Self:
        if self.hello_sha256 != clinical_guest_bootstrap_hello_sha256(self.hello):
            raise ValueError('signed bootstrap authorization does not bind its exact hello')
        return self


class ClinicalGuestBootstrapAck(StrictModel):
    """The guest's sole pre-RPC response, bound to the signed authorization."""

    schema_version: Literal['vaxreplay.clinical-guest-bootstrap-ack.dev-v0.3'] = (
        CLINICAL_GUEST_BOOTSTRAP_ACK_SCHEMA_VERSION
    )
    protocol_id: Literal['lane-a-clinical-guest-bootstrap-dev-v0.3'] = CLINICAL_GUEST_BOOTSTRAP_PROTOCOL_ID
    signed_hello_sha256: str = Field(pattern=_SHA256_PATTERN)
    hello_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    session_id: str = Field(pattern=_RUN_ID_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    nonce: str = Field(pattern=_NONCE_PATTERN)
    accepted_at: datetime
    accepted: Literal[True] = True
    guest_rpc_started: Literal[False] = False

    @field_validator('accepted_at')
    @classmethod
    def validate_accepted_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'accepted_at')


class ClinicalGuestBootstrapContext(StrictModel):
    """Guest-local validated handoff used to construct the first ``GuestRpcClient``."""

    schema_version: Literal['vaxreplay.clinical-guest-bootstrap-context.dev-v0.3'] = (
        CLINICAL_GUEST_BOOTSTRAP_CONTEXT_SCHEMA_VERSION
    )
    signed_hello: SignedClinicalGuestBootstrapHello
    signed_hello_sha256: str = Field(pattern=_SHA256_PATTERN)
    ack: ClinicalGuestBootstrapAck

    @model_validator(mode='after')
    def validate_context(self) -> Self:
        expected_signed_sha256 = clinical_guest_bootstrap_signed_hello_sha256(self.signed_hello)
        if expected_signed_sha256 != self.signed_hello_sha256:
            raise ValueError('bootstrap context hash does not bind the exact signed hello')
        if not _ack_matches_signed_hello(self.ack, self.signed_hello, expected_signed_sha256):
            raise ValueError('bootstrap context acknowledgement does not bind its signed hello')
        return self

    @property
    def hello(self) -> ClinicalGuestBootstrapHello:
        return self.signed_hello.hello


class ClinicalGuestBootstrapReceipt(StrictModel):
    """Host-authenticated, content-free record of one successful bootstrap exchange."""

    schema_version: Literal['vaxreplay.clinical-guest-bootstrap-receipt.dev-v0.3'] = (
        CLINICAL_GUEST_BOOTSTRAP_RECEIPT_SCHEMA_VERSION
    )
    protocol_id: Literal['lane-a-clinical-guest-bootstrap-dev-v0.3'] = CLINICAL_GUEST_BOOTSTRAP_PROTOCOL_ID
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    authorization_key_id: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    start_redemption_sha256: str = Field(pattern=_SHA256_PATTERN)
    session_id: str = Field(pattern=_RUN_ID_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_bootstrap_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    action_schema_sha256: str = Field(pattern=_SHA256_PATTERN)
    rpc_limits_sha256: str = Field(pattern=_SHA256_PATTERN)
    nonce_sha256: str = Field(pattern=_SHA256_PATTERN)
    hello_sha256: str = Field(pattern=_SHA256_PATTERN)
    hello_bytes: int = Field(gt=0, le=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES)
    signed_hello_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_hello_bytes: int = Field(gt=0, le=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES)
    ack_sha256: str = Field(pattern=_SHA256_PATTERN)
    ack_bytes: int = Field(gt=0, le=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES)
    valid_from: datetime
    expires_at: datetime
    hello_sent_at: datetime
    ack_received_at: datetime
    guest_accepted_at: datetime
    bootstrap_succeeded: Literal[True] = True
    guest_rpc_started_before_ack: Literal[False] = False
    task_content_retained_in_receipt: Literal[False] = False
    guest_ack_protocol_requires_launcher_signature_verification: Literal[True] = True
    guest_signature_verification_remotely_attested: Literal[False] = False
    dynamic_values_copied_from_wire_as_trust_anchor: Literal[False] = False
    development_only: Literal[True] = True
    outer_binding_claimed_by_bootstrap_layer: Literal[False] = False

    @field_validator(
        'valid_from',
        'expires_at',
        'hello_sent_at',
        'ack_received_at',
        'guest_accepted_at',
    )
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode='after')
    def validate_times(self) -> Self:
        if not (
            self.valid_from <= self.hello_sent_at <= self.ack_received_at <= self.expires_at
            and self.valid_from <= self.guest_accepted_at <= self.expires_at
        ):
            raise ValueError('bootstrap receipt timestamps lie outside the hello validity interval')
        return self


class AuthenticatedClinicalGuestBootstrap(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-clinical-guest-bootstrap.dev-v0.3'] = (
        AUTHENTICATED_CLINICAL_GUEST_BOOTSTRAP_SCHEMA_VERSION
    )
    signed_hello: SignedClinicalGuestBootstrapHello
    receipt: ClinicalGuestBootstrapReceipt
    receipt_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


class ClinicalGuestBootstrapReplayGuard(Protocol):
    """Single-use nonce boundary. A durable production implementation may replace this one."""

    def consume(self, *, nonce: str, hello_sha256: str) -> bool: ...


class InMemoryClinicalGuestBootstrapReplayGuard:
    """Thread-safe fail-closed guard for one guest process lifetime."""

    def __init__(self) -> None:
        self._consumed_nonces: dict[str, str] = {}
        self._lock = threading.Lock()

    def consume(self, *, nonce: str, hello_sha256: str) -> bool:
        with self._lock:
            if nonce in self._consumed_nonces:
                return False
            self._consumed_nonces[nonce] = hello_sha256
            return True


def clinical_guest_bootstrap_hello_sha256(hello: ClinicalGuestBootstrapHello) -> str:
    canonical = ClinicalGuestBootstrapHello.model_validate_json(canonical_json_bytes(hello))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def clinical_guest_bootstrap_authorization_key_id(public_key: bytes) -> str:
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError('clinical guest bootstrap authorization key must contain 32 bytes')
    try:
        Ed25519PublicKey.from_public_bytes(public_key)
    except ValueError as error:
        raise ValueError('clinical guest bootstrap authorization key is invalid') from error
    return hashlib.sha256(_AUTHORIZATION_KEY_ID_DOMAIN + public_key).hexdigest()


def sign_clinical_guest_bootstrap_hello(
    hello: ClinicalGuestBootstrapHello,
    *,
    signer: Ed25519Signer,
) -> SignedClinicalGuestBootstrapHello:
    """Create the one launcher authorization accepted by the guest."""

    canonical_hello = _canonical_model(hello, ClinicalGuestBootstrapHello)
    try:
        checked = checked_signer(signer)
        public_key = checked.public_key_bytes()
        key_id = clinical_guest_bootstrap_authorization_key_id(public_key)
        signature = checked.sign(_AUTHORIZATION_SIGNATURE_DOMAIN + canonical_json_bytes(canonical_hello))
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise ValueError('clinical guest bootstrap signer returned an invalid signature')
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            _AUTHORIZATION_SIGNATURE_DOMAIN + canonical_json_bytes(canonical_hello),
        )
    except BaseException:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.INVALID_CONFIGURATION) from None
    return SignedClinicalGuestBootstrapHello(
        authorization_key_id=key_id,
        hello=canonical_hello,
        hello_sha256=clinical_guest_bootstrap_hello_sha256(canonical_hello),
        signature_hex=signature.hex(),
    )


def clinical_guest_bootstrap_signed_hello_sha256(
    signed_hello: SignedClinicalGuestBootstrapHello,
) -> str:
    canonical = SignedClinicalGuestBootstrapHello.model_validate_json(canonical_json_bytes(signed_hello))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def verify_signed_clinical_guest_bootstrap_hello(
    signed_hello: SignedClinicalGuestBootstrapHello,
    *,
    trust_anchor: ClinicalGuestBootstrapTrustAnchor,
) -> ClinicalGuestBootstrapHello:
    """Verify launcher authorization and every static image pin before acknowledgement."""

    try:
        canonical_signed = _canonical_model(
            signed_hello,
            SignedClinicalGuestBootstrapHello,
        )
        canonical_trust = _canonical_model(
            trust_anchor,
            ClinicalGuestBootstrapTrustAnchor,
        )
        public_key = bytes.fromhex(canonical_trust.ed25519_public_key_hex)
    except (ClinicalGuestBootstrapError, TypeError, ValueError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.TRUST_ANCHOR_MISMATCH) from None
    hello = canonical_signed.hello
    if (
        canonical_signed.authorization_key_id != canonical_trust.authorization_key_id
        or clinical_guest_bootstrap_authorization_key_id(public_key) != canonical_trust.authorization_key_id
        or hello.execution_policy_sha256 != canonical_trust.execution_policy_sha256
        or hello.worker_bootstrap_profile_sha256 != canonical_trust.worker_bootstrap_profile_sha256
        or hello.harness_policy_id != canonical_trust.harness_policy_id
        or hello.harness_policy_sha256 != canonical_trust.harness_policy_sha256
        or hello.action_schema_sha256 != canonical_trust.action_schema_sha256
        or hello.rpc_limits != canonical_trust.rpc_limits
        or hello.task_invocation.response_protocol != canonical_trust.response_protocol
    ):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.TRUST_ANCHOR_MISMATCH)
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(canonical_signed.signature_hex),
            _AUTHORIZATION_SIGNATURE_DOMAIN + canonical_json_bytes(hello),
        )
    except (InvalidSignature, ValueError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.AUTHORIZATION_INVALID) from None
    return hello


def clinical_guest_bootstrap_receipt_key_id(key: bytes) -> str:
    _require_receipt_key(key)
    return hashlib.sha256(_RECEIPT_KEY_ID_DOMAIN + key).hexdigest()


def clinical_guest_bootstrap_receipt_hmac(receipt: ClinicalGuestBootstrapReceipt, key: bytes) -> str:
    _require_receipt_key(key)
    return hmac.new(key, _RECEIPT_HMAC_DOMAIN + canonical_json_bytes(receipt), hashlib.sha256).hexdigest()


def perform_host_clinical_guest_bootstrap(
    connection: socket.socket,
    *,
    hello: ClinicalGuestBootstrapHello,
    authorization_signer: Ed25519Signer,
    expected_authorization_key_id: str,
    receipt_key: bytes,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    timeout_seconds: float = CLINICAL_GUEST_BOOTSTRAP_DEFAULT_TIMEOUT_SECONDS,
) -> AuthenticatedClinicalGuestBootstrap:
    """Send one hello, require its one exact ack, and authenticate a content-free receipt."""

    canonical_hello = _canonical_model(hello, ClinicalGuestBootstrapHello)
    _validate_timeout(timeout_seconds)
    try:
        _require_receipt_key(receipt_key)
        signed_hello = sign_clinical_guest_bootstrap_hello(
            canonical_hello,
            signer=authorization_signer,
        )
    except (TypeError, ValueError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.INVALID_CONFIGURATION) from None
    if not hmac.compare_digest(
        signed_hello.authorization_key_id,
        expected_authorization_key_id,
    ):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.INVALID_CONFIGURATION)
    hello_body = canonical_json_bytes(canonical_hello)
    hello_sha256 = hashlib.sha256(hello_body).hexdigest()
    signed_hello_body = canonical_json_bytes(signed_hello)
    signed_hello_sha256 = hashlib.sha256(signed_hello_body).hexdigest()
    sent_at = _clock_now(clock)
    _require_in_window(canonical_hello, sent_at)

    try:
        frame = encode_guest_rpc_frame(
            signed_hello,
            maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
        )
        with _temporary_socket_timeout(connection, timeout_seconds):
            send_guest_rpc_frame(connection, frame)
            ack_frame = receive_guest_rpc_frame(
                connection,
                maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
            )
    except TimeoutError:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.TIMEOUT) from None
    except (GuestRpcError, OSError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.CONNECTION_FAILED) from None

    try:
        ack, ack_body = decode_guest_rpc_frame(
            ack_frame,
            ClinicalGuestBootstrapAck,
            maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
        )
    except GuestRpcError:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.FRAME_REJECTED) from None
    received_at = _clock_now(clock)
    _require_in_window(canonical_hello, received_at)
    if not _ack_matches_signed_hello(ack, signed_hello, signed_hello_sha256):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.ACK_BINDING_INVALID)
    if not canonical_hello.valid_from <= ack.accepted_at <= canonical_hello.expires_at:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.OUTSIDE_VALIDITY_WINDOW)

    try:
        receipt = ClinicalGuestBootstrapReceipt(
            receipt_key_id=clinical_guest_bootstrap_receipt_key_id(receipt_key),
            authorization_key_id=signed_hello.authorization_key_id,
            run_id=canonical_hello.run_id,
            start_redemption_sha256=canonical_hello.start_redemption_sha256,
            session_id=canonical_hello.session_id,
            task_invocation_sha256=canonical_hello.task_invocation_sha256,
            workspace_manifest_sha256=canonical_hello.workspace_manifest_sha256,
            workspace_tree_sha256=canonical_hello.workspace_tree_sha256,
            model_visible_surface_sha256=canonical_hello.model_visible_surface_sha256,
            execution_policy_sha256=canonical_hello.execution_policy_sha256,
            worker_bootstrap_profile_sha256=(canonical_hello.worker_bootstrap_profile_sha256),
            worker_spec_sha256=canonical_hello.worker_spec_sha256,
            harness_policy_sha256=canonical_hello.harness_policy_sha256,
            action_schema_sha256=canonical_hello.action_schema_sha256,
            rpc_limits_sha256=hashlib.sha256(canonical_json_bytes(canonical_hello.rpc_limits)).hexdigest(),
            nonce_sha256=hashlib.sha256(canonical_hello.nonce.encode('ascii')).hexdigest(),
            hello_sha256=hello_sha256,
            hello_bytes=len(hello_body),
            signed_hello_sha256=signed_hello_sha256,
            signed_hello_bytes=len(signed_hello_body),
            ack_sha256=hashlib.sha256(ack_body).hexdigest(),
            ack_bytes=len(ack_body),
            valid_from=canonical_hello.valid_from,
            expires_at=canonical_hello.expires_at,
            hello_sent_at=sent_at,
            ack_received_at=received_at,
            guest_accepted_at=ack.accepted_at,
        )
    except ValueError:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.INVALID_CONFIGURATION) from None
    return AuthenticatedClinicalGuestBootstrap(
        signed_hello=signed_hello,
        receipt=receipt,
        receipt_hmac_sha256=clinical_guest_bootstrap_receipt_hmac(receipt, receipt_key),
    )


def perform_guest_clinical_bootstrap(
    connection: socket.socket,
    *,
    trust_anchor: ClinicalGuestBootstrapTrustAnchor,
    replay_guard: ClinicalGuestBootstrapReplayGuard,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    timeout_seconds: float = CLINICAL_GUEST_BOOTSTRAP_DEFAULT_TIMEOUT_SECONDS,
) -> ClinicalGuestBootstrapContext:
    """Verify launcher authority and static pins before acknowledgement or ordinary guest RPC."""

    canonical_trust = _canonical_model(trust_anchor, ClinicalGuestBootstrapTrustAnchor)
    _validate_timeout(timeout_seconds)
    try:
        with _temporary_socket_timeout(connection, timeout_seconds):
            frame = receive_guest_rpc_frame(
                connection,
                maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
            )
    except TimeoutError:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.TIMEOUT) from None
    except (GuestRpcError, OSError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.FRAME_REJECTED) from None

    try:
        signed_hello, signed_hello_body = decode_guest_rpc_frame(
            frame,
            SignedClinicalGuestBootstrapHello,
            maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
        )
    except GuestRpcError:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.FRAME_REJECTED) from None
    hello = verify_signed_clinical_guest_bootstrap_hello(
        signed_hello,
        trust_anchor=canonical_trust,
    )
    signed_hello_sha256 = hashlib.sha256(signed_hello_body).hexdigest()
    accepted_at = _clock_now(clock)
    _require_in_window(hello, accepted_at)
    try:
        consumed = replay_guard.consume(nonce=hello.nonce, hello_sha256=signed_hello_sha256)
    except Exception:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.REPLAY_REJECTED) from None
    if not consumed:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.REPLAY_REJECTED)

    ack = ClinicalGuestBootstrapAck(
        signed_hello_sha256=signed_hello_sha256,
        hello_sha256=signed_hello.hello_sha256,
        run_id=hello.run_id,
        session_id=hello.session_id,
        task_invocation_sha256=hello.task_invocation_sha256,
        nonce=hello.nonce,
        accepted_at=accepted_at,
    )
    try:
        ack_frame = encode_guest_rpc_frame(
            ack,
            maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
        )
        with _temporary_socket_timeout(connection, timeout_seconds):
            send_guest_rpc_frame(connection, ack_frame)
    except TimeoutError:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.TIMEOUT) from None
    except (GuestRpcError, OSError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.CONNECTION_FAILED) from None
    return ClinicalGuestBootstrapContext(
        signed_hello=signed_hello,
        signed_hello_sha256=signed_hello_sha256,
        ack=ack,
    )


def run_lane_a_clinical_guest_entry(
    connection: socket.socket,
    *,
    trust_anchor: ClinicalGuestBootstrapTrustAnchor,
    replay_guard: ClinicalGuestBootstrapReplayGuard,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    timeout_seconds: float = CLINICAL_GUEST_BOOTSTRAP_DEFAULT_TIMEOUT_SECONDS,
) -> LaneAGuestHarnessResult:
    """Bootstrap, then run the fixed Lane A harness over the very same connected socket."""

    context = perform_guest_clinical_bootstrap(
        connection,
        trust_anchor=trust_anchor,
        replay_guard=replay_guard,
        clock=clock,
        timeout_seconds=timeout_seconds,
    )
    client = GuestRpcClient(
        connection,
        session_id=context.hello.session_id,
        task_invocation=context.hello.task_invocation,
        maximum_body_bytes=context.hello.rpc_limits.maximum_frame_body_bytes,
    )
    return run_lane_a_guest_harness(client, task_invocation=context.hello.task_invocation)


def verify_authenticated_clinical_guest_bootstrap(
    artifact: AuthenticatedClinicalGuestBootstrap,
    *,
    key: bytes,
    expected_key_id: str,
    expected_hello: ClinicalGuestBootstrapHello,
    trust_anchor: ClinicalGuestBootstrapTrustAnchor,
) -> ClinicalGuestBootstrapReceipt:
    """Authenticate the receipt and bind signed authorization to caller-owned run state."""

    try:
        canonical_artifact = _canonical_model(artifact, AuthenticatedClinicalGuestBootstrap)
        canonical_expected_hello = _canonical_model(expected_hello, ClinicalGuestBootstrapHello)
        key_id = clinical_guest_bootstrap_receipt_key_id(key)
    except (ClinicalGuestBootstrapError, TypeError, ValueError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.RECEIPT_AUTHENTICATION_FAILED) from None
    try:
        canonical_hello = verify_signed_clinical_guest_bootstrap_hello(
            canonical_artifact.signed_hello,
            trust_anchor=trust_anchor,
        )
    except ClinicalGuestBootstrapError:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.RECEIPT_AUTHENTICATION_FAILED) from None
    if canonical_hello != canonical_expected_hello:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.RECEIPT_AUTHENTICATION_FAILED)
    receipt = canonical_artifact.receipt
    if not hmac.compare_digest(key_id, expected_key_id) or not hmac.compare_digest(
        receipt.receipt_key_id,
        expected_key_id,
    ):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.RECEIPT_AUTHENTICATION_FAILED)
    expected_hmac = clinical_guest_bootstrap_receipt_hmac(receipt, key)
    if not hmac.compare_digest(canonical_artifact.receipt_hmac_sha256, expected_hmac):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.RECEIPT_AUTHENTICATION_FAILED)
    expected_fields = (
        canonical_artifact.signed_hello.authorization_key_id,
        canonical_hello.run_id,
        canonical_hello.start_redemption_sha256,
        canonical_hello.session_id,
        canonical_hello.task_invocation_sha256,
        canonical_hello.workspace_manifest_sha256,
        canonical_hello.workspace_tree_sha256,
        canonical_hello.model_visible_surface_sha256,
        canonical_hello.execution_policy_sha256,
        canonical_hello.worker_bootstrap_profile_sha256,
        canonical_hello.worker_spec_sha256,
        canonical_hello.harness_policy_sha256,
        canonical_hello.action_schema_sha256,
        hashlib.sha256(canonical_json_bytes(canonical_hello.rpc_limits)).hexdigest(),
        hashlib.sha256(canonical_hello.nonce.encode('ascii')).hexdigest(),
        clinical_guest_bootstrap_hello_sha256(canonical_hello),
        len(canonical_json_bytes(canonical_hello)),
        clinical_guest_bootstrap_signed_hello_sha256(canonical_artifact.signed_hello),
        len(canonical_json_bytes(canonical_artifact.signed_hello)),
        canonical_hello.valid_from,
        canonical_hello.expires_at,
    )
    actual_fields = (
        receipt.authorization_key_id,
        receipt.run_id,
        receipt.start_redemption_sha256,
        receipt.session_id,
        receipt.task_invocation_sha256,
        receipt.workspace_manifest_sha256,
        receipt.workspace_tree_sha256,
        receipt.model_visible_surface_sha256,
        receipt.execution_policy_sha256,
        receipt.worker_bootstrap_profile_sha256,
        receipt.worker_spec_sha256,
        receipt.harness_policy_sha256,
        receipt.action_schema_sha256,
        receipt.rpc_limits_sha256,
        receipt.nonce_sha256,
        receipt.hello_sha256,
        receipt.hello_bytes,
        receipt.signed_hello_sha256,
        receipt.signed_hello_bytes,
        receipt.valid_from,
        receipt.expires_at,
    )
    if actual_fields != expected_fields:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.RECEIPT_AUTHENTICATION_FAILED)
    return receipt


def _ack_matches_signed_hello(
    ack: ClinicalGuestBootstrapAck,
    signed_hello: SignedClinicalGuestBootstrapHello,
    signed_hello_sha256: str,
) -> bool:
    hello = signed_hello.hello
    return (
        ack.signed_hello_sha256,
        ack.hello_sha256,
        ack.run_id,
        ack.session_id,
        ack.task_invocation_sha256,
        ack.nonce,
    ) == (
        signed_hello_sha256,
        signed_hello.hello_sha256,
        hello.run_id,
        hello.session_id,
        hello.task_invocation_sha256,
        hello.nonce,
    )


def _require_in_window(hello: ClinicalGuestBootstrapHello, now: datetime) -> None:
    if not hello.valid_from <= now <= hello.expires_at:
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.OUTSIDE_VALIDITY_WINDOW)


def _canonical_model[ModelT: StrictModel](value: ModelT, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate_json(canonical_json_bytes(value))
    except (TypeError, ValueError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.INVALID_CONFIGURATION) from None


def _clock_now(clock: Callable[[], datetime]) -> datetime:
    try:
        return _aware_utc(clock(), 'bootstrap clock')
    except (TypeError, ValueError):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.INVALID_CONFIGURATION) from None


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(UTC)


def _validate_timeout(timeout_seconds: float) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_TIMEOUT_SECONDS
    ):
        raise ClinicalGuestBootstrapError(ClinicalGuestBootstrapFailureCode.INVALID_CONFIGURATION)


@contextmanager
def _temporary_socket_timeout(connection: socket.socket, timeout_seconds: float):
    previous = connection.gettimeout()
    connection.settimeout(timeout_seconds)
    try:
        yield
    finally:
        connection.settimeout(previous)


def _require_receipt_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError('clinical guest bootstrap receipt key must contain at least 32 bytes')


__all__ = [
    'AUTHENTICATED_CLINICAL_GUEST_BOOTSTRAP_SCHEMA_VERSION',
    'CLINICAL_GUEST_BOOTSTRAP_ACK_SCHEMA_VERSION',
    'CLINICAL_GUEST_BOOTSTRAP_CONTEXT_SCHEMA_VERSION',
    'CLINICAL_GUEST_BOOTSTRAP_DEFAULT_TIMEOUT_SECONDS',
    'CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256',
    'CLINICAL_GUEST_BOOTSTRAP_HELLO_SCHEMA_VERSION',
    'CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES',
    'CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_TIMEOUT_SECONDS',
    'CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_VALIDITY_SECONDS',
    'CLINICAL_GUEST_BOOTSTRAP_PROTOCOL_ID',
    'CLINICAL_GUEST_BOOTSTRAP_RECEIPT_SCHEMA_VERSION',
    'CLINICAL_GUEST_BOOTSTRAP_SIGNED_HELLO_SCHEMA_VERSION',
    'CLINICAL_GUEST_BOOTSTRAP_TRUST_ANCHOR_SCHEMA_VERSION',
    'AuthenticatedClinicalGuestBootstrap',
    'ClinicalGuestBootstrapAck',
    'ClinicalGuestBootstrapContext',
    'ClinicalGuestBootstrapError',
    'ClinicalGuestBootstrapFailureCode',
    'ClinicalGuestBootstrapHello',
    'ClinicalGuestBootstrapReceipt',
    'ClinicalGuestBootstrapReplayGuard',
    'ClinicalGuestBootstrapTrustAnchor',
    'ClinicalGuestRpcLimits',
    'InMemoryClinicalGuestBootstrapReplayGuard',
    'SignedClinicalGuestBootstrapHello',
    'clinical_guest_bootstrap_authorization_key_id',
    'clinical_guest_bootstrap_hello_sha256',
    'clinical_guest_bootstrap_receipt_hmac',
    'clinical_guest_bootstrap_receipt_key_id',
    'clinical_guest_bootstrap_signed_hello_sha256',
    'perform_guest_clinical_bootstrap',
    'perform_host_clinical_guest_bootstrap',
    'run_lane_a_clinical_guest_entry',
    'sign_clinical_guest_bootstrap_hello',
    'verify_authenticated_clinical_guest_bootstrap',
    'verify_signed_clinical_guest_bootstrap_hello',
]
