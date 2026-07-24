"""Strict wire and persistence schemas for the external witness service."""

from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import datetime, timezone
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.schema import LEDGER_CHECKPOINT_SCHEMA_VERSION, SAFE_ID_PATTERN
from vaxreplay.operations.witness import (
    CheckpointWitnessRequest,
    ExternalWitnessMethod,
    RegistryCheckpointWitnessRequest,
)

WITNESS_SERVICE_POLICY_SCHEMA_VERSION = 'vaxreplay.witness-service-policy.v0.3'
WITNESS_SERVICE_TRUST_POLICY_SCHEMA_VERSION = 'vaxreplay.witness-service-trust-policy.v0.1'
WITNESS_SERVICE_SUBMISSION_SCHEMA_VERSION = 'vaxreplay.witness-service-submission.v0.1'
WITNESS_SERVICE_LOG_ENTRY_SCHEMA_VERSION = 'vaxreplay.witness-service-log-entry.v0.1'
WITNESS_SERVICE_LOG_CHECKPOINT_SCHEMA_VERSION = 'vaxreplay.witness-service-log-checkpoint.v0.1'
WITNESS_SERVICE_RECEIPT_STATEMENT_SCHEMA_VERSION = 'vaxreplay.witness-service-receipt-statement.v0.1'
WITNESS_SERVICE_PROOF_SCHEMA_VERSION = 'vaxreplay.witness-service-proof.v0.1'
WITNESS_SERVICE_SIGNED_CHECKPOINT_SCHEMA_VERSION = 'vaxreplay.witness-service-signed-checkpoint.v0.1'
WITNESS_SERVICE_VERIFICATION_REPORT_SCHEMA_VERSION = 'vaxreplay.witness-service-verification-report.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_NONCE_PATTERN = r'^[0-9a-f]{64}$'
_RECEIPT_ID_PATTERN = r'^receipt-[0-9a-f]{32}$'
_ZERO_SHA256 = '0' * 64
_EMPTY_TREE_SHA256 = hashlib.sha256(b'').hexdigest()


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _canonical_base64(value: str, *, expected_bytes: int, field_name: str) -> str:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f'{field_name} must be valid base64') from error
    if len(decoded) != expected_bytes:
        raise ValueError(f'{field_name} must encode exactly {expected_bytes} bytes')
    if base64.b64encode(decoded).decode('ascii') != value:
        raise ValueError(f'{field_name} must use canonical base64 encoding')
    return value


class WitnessRegistrySigningKey(StrictModel):
    """Registry checkpoint key accepted by this independent monitor."""

    key_id: str = Field(pattern=SAFE_ID_PATTERN)
    public_key_base64: str
    valid_from: datetime
    valid_until: datetime | None = None

    @field_validator('public_key_base64')
    @classmethod
    def validate_public_key(cls, value: str) -> str:
        return _canonical_base64(value, expected_bytes=32, field_name='registry public key')

    @field_validator('valid_from', 'valid_until')
    @classmethod
    def validate_key_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_utc(value, 'registry signing-key validity time')

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError('registry signing-key valid_until must follow valid_from')
        return self


class WitnessRegistryMonitor(StrictModel):
    """One registry identity and key ring monitored for non-equivocation."""

    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_keys: tuple[WitnessRegistrySigningKey, ...] = Field(min_length=1, max_length=64)
    require_sequential_tree_sizes: Literal[True] = True
    require_rfc6962_consistency: Literal[True] = True

    @model_validator(mode='after')
    def validate_keys(self) -> Self:
        key_ids = [key.key_id for key in self.signing_keys]
        if len(key_ids) != len(set(key_ids)):
            raise ValueError('registry monitor signing key IDs must be unique')
        return self


class WitnessedRegistryCheckpoint(StrictModel):
    """Wire-compatible signed registry tree head parsed by the monitor."""

    schema_version: Literal['vaxreplay.plan-selection-registry-checkpoint.v0.1']
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    tree_size: int = Field(ge=0)
    root_sha256: str = Field(pattern=_SHA256_PATTERN)
    issued_at_upper_bound: datetime
    signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    previous_checkpoint_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator('issued_at_upper_bound')
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'registry checkpoint issuance time')

    @model_validator(mode='after')
    def validate_genesis(self) -> Self:
        if self.tree_size == 0 and (
            self.root_sha256 != _EMPTY_TREE_SHA256 or self.previous_checkpoint_sha256 is not None
        ):
            raise ValueError('registry genesis must be the canonical empty-tree head')
        if self.tree_size > 0 and self.previous_checkpoint_sha256 is None:
            raise ValueError('non-genesis registry checkpoint must bind its exact predecessor')
        return self


class WitnessedSignedRegistryCheckpoint(StrictModel):
    """Exact envelope schema accepted by the stateful registry monitor."""

    schema_version: Literal['vaxreplay.signed-plan-selection-registry-checkpoint.v0.1']
    checkpoint: WitnessedRegistryCheckpoint
    signature_base64: str

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        return _canonical_base64(value, expected_bytes=64, field_name='registry checkpoint signature')


class WitnessServicePolicy(StrictModel):
    """Public, immutable behavior policy for one separately deployed service."""

    schema_version: Literal['vaxreplay.witness-service-policy.v0.3'] = WITNESS_SERVICE_POLICY_SCHEMA_VERSION
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    endpoint_uri: str = Field(min_length=1, max_length=4096)
    method: Literal[ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG] = ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    digest_algorithm: Literal['sha256'] = 'sha256'
    client_nonce_bytes: Literal[32] = 32
    clock_source: Literal['service-host-system-utc-no-client-time'] = 'service-host-system-utc-no-client-time'
    clock_rollback_behavior: Literal['fail-closed-no-time-synthesis'] = 'fail-closed-no-time-synthesis'
    storage_profile: Literal['sqlite-wal-full-sync-hash-chain'] = 'sqlite-wal-full-sync-hash-chain'
    write_authentication_required: Literal[True] = True
    max_submission_bytes: int = Field(ge=512, le=1024 * 1024)
    max_proof_bytes: int = Field(ge=1024, le=16 * 1024 * 1024)
    client_timeout_seconds: float = Field(ge=0.1, le=120.0)
    registry_monitors: tuple[WitnessRegistryMonitor, ...] = Field(default=(), max_length=256)
    clock_health_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    clock_health_process_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    external_signer_process_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator('endpoint_uri')
    @classmethod
    def validate_endpoint_uri(cls, value: str) -> str:
        if value.strip() != value or any(character in value for character in '\x00\r\n'):
            raise ValueError('endpoint_uri must be trimmed and contain no control separators')
        parsed = urlsplit(value)
        if (
            parsed.scheme != 'https'
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path != '/v1/witness'
        ):
            raise ValueError('endpoint_uri must be an exact HTTPS /v1/witness endpoint without credentials or query')
        return value

    @model_validator(mode='after')
    def validate_registry_monitors(self) -> Self:
        identities = [(monitor.registry_id, monitor.authority_id) for monitor in self.registry_monitors]
        if len(identities) != len(set(identities)):
            raise ValueError('witness policy contains duplicate registry monitor identities')
        bindings = (
            self.clock_health_policy_sha256,
            self.clock_health_process_sha256,
            self.external_signer_process_sha256,
        )
        if any(value is None for value in bindings) and any(value is not None for value in bindings):
            raise ValueError('witness runtime-trust digests must be all present or all null')
        return self


class WitnessServiceTrustPolicy(StrictModel):
    """Out-of-band Ed25519 public-key pin and validity window."""

    schema_version: Literal['vaxreplay.witness-service-trust-policy.v0.1'] = WITNESS_SERVICE_TRUST_POLICY_SCHEMA_VERSION
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=SAFE_ID_PATTERN)
    trust_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    service_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    public_key_base64: str
    public_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    key_valid_from: datetime
    key_valid_until: datetime | None = None
    revocation_status: Literal['not-revoked'] = 'not-revoked'

    @field_validator('public_key_base64')
    @classmethod
    def validate_public_key(cls, value: str) -> str:
        return _canonical_base64(value, expected_bytes=32, field_name='public_key_base64')

    @field_validator('key_valid_from', 'key_valid_until')
    @classmethod
    def validate_key_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_utc(value, 'key validity time')

    @model_validator(mode='after')
    def validate_key_binding(self) -> Self:
        public_key = base64.b64decode(self.public_key_base64, validate=True)
        if hashlib.sha256(public_key).hexdigest() != self.public_key_sha256:
            raise ValueError('public_key_base64 does not match public_key_sha256')
        if self.key_valid_until is not None and self.key_valid_until <= self.key_valid_from:
            raise ValueError('key_valid_until must be later than key_valid_from')
        return self


class WitnessServiceSubmission(StrictModel):
    """Authenticated-write body; deliberately contains no client-provided time."""

    schema_version: Literal['vaxreplay.witness-service-submission.v0.1'] = WITNESS_SERVICE_SUBMISSION_SCHEMA_VERSION
    witness_request: CheckpointWitnessRequest | RegistryCheckpointWitnessRequest
    client_nonce: str = Field(pattern=_NONCE_PATTERN)

    @model_validator(mode='after')
    def validate_profile(self) -> Self:
        if self.witness_request.method is not ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG:
            raise ValueError('the Ed25519 witness service only accepts public-transparency-log requests')
        return self


class WitnessServiceLogEntry(StrictModel):
    """One exact hash-chain entry persisted before a receipt is returned."""

    schema_version: Literal['vaxreplay.witness-service-log-entry.v0.1'] = WITNESS_SERVICE_LOG_ENTRY_SCHEMA_VERSION
    sequence: int = Field(gt=0)
    previous_entry_sha256: str = Field(pattern=_SHA256_PATTERN)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_schema_version: Literal[
        'vaxreplay.operations-ledger-checkpoint.v0.1',
        'vaxreplay.signed-plan-selection-registry-checkpoint.v0.1',
    ] = LEDGER_CHECKPOINT_SCHEMA_VERSION
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_bytes: int = Field(gt=0)
    client_nonce: str = Field(pattern=_NONCE_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    witnessed_at: datetime

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'witnessed_at')


class WitnessServiceLogCheckpoint(StrictModel):
    """Signed service-log head issued atomically with one receipt."""

    schema_version: Literal['vaxreplay.witness-service-log-checkpoint.v0.1'] = (
        WITNESS_SERVICE_LOG_CHECKPOINT_SCHEMA_VERSION
    )
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    tree_size: int = Field(gt=0)
    through_entry_sha256: str = Field(pattern=_SHA256_PATTERN)
    previous_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime

    @field_validator('issued_at')
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'issued_at')


class WitnessServiceReceiptStatement(StrictModel):
    """Complete statement protected by the receipt signature."""

    schema_version: Literal['vaxreplay.witness-service-receipt-statement.v0.1'] = (
        WITNESS_SERVICE_RECEIPT_STATEMENT_SCHEMA_VERSION
    )
    submission: WitnessServiceSubmission
    entry: WitnessServiceLogEntry
    checkpoint: WitnessServiceLogCheckpoint


class WitnessServiceProof(StrictModel):
    """Portable receipt plus independent signature over its included log checkpoint."""

    schema_version: Literal['vaxreplay.witness-service-proof.v0.1'] = WITNESS_SERVICE_PROOF_SCHEMA_VERSION
    receipt_id: str = Field(pattern=_RECEIPT_ID_PATTERN)
    statement: WitnessServiceReceiptStatement
    receipt_signature_base64: str
    checkpoint_signature_base64: str

    @field_validator('receipt_signature_base64', 'checkpoint_signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        return _canonical_base64(value, expected_bytes=64, field_name='Ed25519 signature')


class WitnessServiceSignedCheckpoint(StrictModel):
    """Public latest-log-checkpoint response."""

    schema_version: Literal['vaxreplay.witness-service-signed-checkpoint.v0.1'] = (
        WITNESS_SERVICE_SIGNED_CHECKPOINT_SCHEMA_VERSION
    )
    checkpoint: WitnessServiceLogCheckpoint
    signature_base64: str

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        return _canonical_base64(value, expected_bytes=64, field_name='checkpoint signature')


class WitnessServiceVerificationReport(StrictModel):
    """Result of a complete local database replay and signature audit."""

    schema_version: Literal['vaxreplay.witness-service-verification-report.v0.1'] = (
        WITNESS_SERVICE_VERIFICATION_REPORT_SCHEMA_VERSION
    )
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=SAFE_ID_PATTERN)
    entry_count: int = Field(ge=0)
    through_entry_sha256: str = Field(pattern=_SHA256_PATTERN)
    through_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    signatures_verified: Literal[True] = True
    hash_chain_verified: Literal[True] = True
    client_time_accepted: Literal[False] = False


ZERO_SHA256 = _ZERO_SHA256
