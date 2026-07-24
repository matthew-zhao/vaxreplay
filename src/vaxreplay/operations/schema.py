"""Strict schemas for the prospective capture operations ledger.

The operations layer deliberately stores only opaque bytes and scheduling metadata.  It
does not interpret vaccine, immune, sequence, or clinical content.  All hashes in this
module are lowercase SHA-256 digests of canonical JSON or exact artifact bytes.
"""

from __future__ import annotations

import enum
import hashlib
from datetime import datetime, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

OPERATIONS_JOB_SCHEMA_VERSION = 'vaxreplay.operations-job.v0.1'
LEDGER_EVENT_SCHEMA_VERSION = 'vaxreplay.operations-ledger-event.v0.1'
LEDGER_CHECKPOINT_SCHEMA_VERSION = 'vaxreplay.operations-ledger-checkpoint.v0.1'

SHA256_PATTERN = r'^[0-9a-f]{64}$'
SAFE_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$'
ARTIFACT_ROLE_PATTERN = r'^[a-z][a-z0-9._-]{0,127}$'


def aware_utc(value: datetime, field_name: str) -> datetime:
    """Normalize an offset-aware timestamp to UTC, rejecting naive datetimes."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class AttemptState(str, enum.Enum):
    STARTED = 'started'
    FAILED = 'failed'
    ABANDONED = 'abandoned'
    SUCCEEDED = 'succeeded'


class LogicalRunState(str, enum.Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'


class LedgerEventType(str, enum.Enum):
    STORE_INITIALIZED = 'store_initialized'
    ARTIFACT_STORED = 'artifact_stored'
    JOB_REGISTERED = 'job_registered'
    LOGICAL_RUN_REGISTERED = 'logical_run_registered'
    ATTEMPT_STARTED = 'attempt_started'
    ATTEMPT_LEASE_RENEWED = 'attempt_lease_renewed'
    ATTEMPT_ARTIFACT_ATTACHED = 'attempt_artifact_attached'
    ATTEMPT_FAILED = 'attempt_failed'
    ATTEMPT_ABANDONED = 'attempt_abandoned'
    ATTEMPT_SUCCEEDED = 'attempt_succeeded'


type LedgerScalar = str | int | bool | None


class CaptureJobSpec(StrictModel):
    """Immutable registration for one periodically scheduled opaque collector.

    ``configuration`` is intentionally a flat map of public commitments.  The generic
    key-name checks are defense in depth, not a secret-detection boundary; each
    collector must parse this map with its own exact allowlisted schema.  Secrets and
    secret identifiers belong outside the immutable ledger.
    """

    schema_version: Literal['vaxreplay.operations-job.v0.1'] = OPERATIONS_JOB_SCHEMA_VERSION
    job_id: str = Field(pattern=SAFE_ID_PATTERN)
    collector_id: str = Field(pattern=SAFE_ID_PATTERN)
    schedule_anchor_at: datetime
    schedule_interval_seconds: int = Field(ge=1, le=366 * 24 * 60 * 60)
    configuration: dict[str, str | int | bool] = Field(default_factory=dict)

    @field_validator('schedule_anchor_at')
    @classmethod
    def validate_schedule_anchor_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'schedule_anchor_at')

    @field_validator('configuration')
    @classmethod
    def validate_configuration(cls, value: dict[str, str | int | bool]) -> dict[str, str | int | bool]:
        for key, item in value.items():
            if not key or len(key) > 200 or key.strip() != key or any(character in key for character in '\x00\r\n'):
                raise ValueError('configuration keys must be nonempty, trimmed, and contain no control separators')
            normalized_key = ''.join(character for character in key.lower() if character.isalnum())
            secret_tokens = (
                'apikey',
                'accesskey',
                'auth',
                'authorization',
                'bearer',
                'cookie',
                'credential',
                'password',
                'privatekey',
                'secret',
                'token',
            )
            if any(token in normalized_key for token in secret_tokens):
                raise ValueError('configuration cannot contain secret-bearing keys; use an external secret identifier')
            if isinstance(item, str) and (len(item) > 4096 or '\x00' in item):
                raise ValueError('configuration string values must be at most 4096 characters and contain no NUL')
        return value


def job_spec_sha256(spec: CaptureJobSpec) -> str:
    return hashlib.sha256(canonical_json_bytes(spec)).hexdigest()


def scheduled_logical_run_id(spec_sha256: str, scheduled_for: datetime) -> str:
    """Return the deterministic identity for a job-spec version and schedule slot."""

    if len(spec_sha256) != 64 or any(character not in '0123456789abcdef' for character in spec_sha256):
        raise ValueError('spec_sha256 must be a lowercase SHA-256 digest')
    scheduled_for = aware_utc(scheduled_for, 'scheduled_for')
    preimage = {
        'job_spec_sha256': spec_sha256,
        'scheduled_for': scheduled_for.isoformat().replace('+00:00', 'Z'),
        'schema_version': 'vaxreplay.operations-logical-run-id.v0.1',
    }
    return f'run-{hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()}'


class RegisteredJob(StrictModel):
    spec: CaptureJobSpec
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    registered_at: datetime

    @field_validator('registered_at')
    @classmethod
    def validate_registered_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'registered_at')

    @model_validator(mode='after')
    def validate_spec_digest(self) -> Self:
        if job_spec_sha256(self.spec) != self.spec_sha256:
            raise ValueError('spec_sha256 does not bind spec')
        return self


class LogicalRunRecord(StrictModel):
    logical_run_id: str = Field(pattern=r'^run-[0-9a-f]{64}$')
    job_id: str = Field(pattern=SAFE_ID_PATTERN)
    job_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    scheduled_for: datetime
    state: LogicalRunState
    successful_attempt_id: str | None = Field(default=None, pattern=r'^attempt-[0-9a-f]{32}$')

    @field_validator('scheduled_for')
    @classmethod
    def validate_scheduled_for(cls, value: datetime) -> datetime:
        return aware_utc(value, 'scheduled_for')

    @model_validator(mode='after')
    def validate_success(self) -> Self:
        if (self.state is LogicalRunState.SUCCEEDED) != (self.successful_attempt_id is not None):
            raise ValueError('only a succeeded logical run may identify a successful attempt')
        return self


class AttemptLease(StrictModel):
    attempt_id: str = Field(pattern=r'^attempt-[0-9a-f]{32}$')
    logical_run_id: str = Field(pattern=r'^run-[0-9a-f]{64}$')
    attempt_number: int = Field(ge=1)
    owner_id: str = Field(pattern=SAFE_ID_PATTERN)
    state: AttemptState
    started_at: datetime
    lease_expires_at: datetime
    finished_at: datetime | None = None
    terminal_code: str | None = Field(default=None, max_length=200)

    @field_validator('started_at', 'lease_expires_at')
    @classmethod
    def validate_required_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'attempt timestamp')

    @field_validator('finished_at')
    @classmethod
    def validate_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else aware_utc(value, 'finished_at')

    @model_validator(mode='after')
    def validate_lifecycle(self) -> Self:
        if self.lease_expires_at <= self.started_at:
            raise ValueError('lease_expires_at must be after started_at')
        if self.state is AttemptState.STARTED:
            if self.finished_at is not None or self.terminal_code is not None:
                raise ValueError('started attempts cannot contain terminal fields')
        elif self.finished_at is None:
            raise ValueError('terminal attempts require finished_at')
        return self


class StoredArtifact(StrictModel):
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(ge=0)
    relative_path: str = Field(pattern=r'^objects/sha256/[0-9a-f]{2}/[0-9a-f]{64}$')
    first_recorded_at: datetime

    @field_validator('first_recorded_at')
    @classmethod
    def validate_first_recorded_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'first_recorded_at')

    @model_validator(mode='after')
    def validate_path(self) -> Self:
        parts = self.relative_path.split('/')
        if len(parts) != 4 or parts[2] != self.sha256[:2] or parts[3] != self.sha256:
            raise ValueError('relative_path must be derived from sha256')
        return self


class LedgerEvent(StrictModel):
    schema_version: Literal['vaxreplay.operations-ledger-event.v0.1'] = LEDGER_EVENT_SCHEMA_VERSION
    sequence: int = Field(ge=1)
    event_type: LedgerEventType
    occurred_at: datetime
    previous_event_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    payload: dict[str, LedgerScalar]
    event_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator('occurred_at')
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'occurred_at')

    @model_validator(mode='after')
    def validate_event_hash(self) -> Self:
        if ledger_event_sha256(self) != self.event_sha256:
            raise ValueError('event_sha256 does not bind the canonical event preimage')
        if (self.sequence == 1) != (self.previous_event_sha256 is None):
            raise ValueError('only the first ledger event may omit previous_event_sha256')
        return self


def ledger_event_sha256(event: LedgerEvent) -> str:
    preimage = event.model_dump(mode='json', exclude={'event_sha256'})
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


class LedgerCheckpoint(StrictModel):
    """Small canonical target intended for an RFC 3161 or transparency witness."""

    schema_version: Literal['vaxreplay.operations-ledger-checkpoint.v0.1'] = LEDGER_CHECKPOINT_SCHEMA_VERSION
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    created_at: datetime
    through_sequence: int = Field(ge=1)
    through_event_sha256: str = Field(pattern=SHA256_PATTERN)
    object_count: int = Field(ge=0)
    object_inventory_sha256: str = Field(pattern=SHA256_PATTERN)
    external_timestamp_required: Literal[True] = True

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'created_at')


def checkpoint_bytes(checkpoint: LedgerCheckpoint) -> bytes:
    return canonical_json_bytes(checkpoint)


def checkpoint_sha256(checkpoint: LedgerCheckpoint) -> str:
    return hashlib.sha256(checkpoint_bytes(checkpoint)).hexdigest()


class StoreVerificationReport(StrictModel):
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    verified_at: datetime
    event_count: int = Field(ge=1)
    ledger_head_sha256: str = Field(pattern=SHA256_PATTERN)
    object_count: int = Field(ge=0)
    job_count: int = Field(ge=0)
    logical_run_count: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    checkpoint_verified: bool

    @field_validator('verified_at')
    @classmethod
    def validate_verified_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'verified_at')
