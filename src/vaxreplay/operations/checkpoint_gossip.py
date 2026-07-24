"""Durable multi-party gossip for registry and witness signed checkpoints.

The selection registry and external witness each protect their current state with
signed, predecessor-linked checkpoints.  A valid older database can nevertheless
be restored without breaking either service's local verification.  This module is
the independently deployed ratchet: every observation is signature checked,
appended to a local hash-chained journal, and compared with signed reports from at
least one other monitor.

The monitor deliberately requires adjacent source heads.  If polling skips a
tree size, operators must retrieve the immutable historical head before advancing
the ratchet.  Silently accepting a gap would turn a predecessor link into an
unchecked assertion.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Final, Literal, Self, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.clock_health import ClockHealthGate, require_clock_health
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc
from vaxreplay.operations.signing import Ed25519Signer, LocalEd25519Signer, checked_signer
from vaxreplay.operations.witness_service import verify_witness_service_signed_checkpoint
from vaxreplay.operations.witness_service_schema import (
    ZERO_SHA256,
    WitnessedSignedRegistryCheckpoint,
    WitnessRegistryMonitor,
    WitnessServicePolicy,
    WitnessServiceSignedCheckpoint,
    WitnessServiceTrustPolicy,
)

GOSSIP_REGISTRY_STREAM_POLICY_SCHEMA_VERSION = 'vaxreplay.registry-gossip-stream-policy.v0.1'
GOSSIP_WITNESS_STREAM_POLICY_SCHEMA_VERSION = 'vaxreplay.witness-gossip-stream-policy.v0.2'
GOSSIP_MONITOR_POLICY_SCHEMA_VERSION = 'vaxreplay.checkpoint-gossip-monitor-policy.v0.3'
GOSSIP_OBSERVATION_SCHEMA_VERSION = 'vaxreplay.checkpoint-gossip-observation.v0.1'
GOSSIP_MONITOR_REPORT_SCHEMA_VERSION = 'vaxreplay.checkpoint-gossip-monitor-report.v0.1'
SIGNED_GOSSIP_MONITOR_REPORT_SCHEMA_VERSION = 'vaxreplay.signed-checkpoint-gossip-monitor-report.v0.1'
GOSSIP_COMPARISON_POLICY_SCHEMA_VERSION = 'vaxreplay.checkpoint-gossip-comparison-policy.v0.2'
GOSSIP_AGREEMENT_REPORT_SCHEMA_VERSION = 'vaxreplay.checkpoint-gossip-agreement-report.v0.1'
GOSSIP_VERIFICATION_REPORT_SCHEMA_VERSION = 'vaxreplay.checkpoint-gossip-verification-report.v0.1'
GOSSIP_DATABASE_SCHEMA_VERSION = 'vaxreplay.checkpoint-gossip-database.v0.1'

_POLICY_PATH: Final = 'policy.json'
_PRIVATE_KEY_PATH: Final = 'report-ed25519-private-key.bin'
_DATABASE_PATH: Final = 'gossip.sqlite3'
_REPORT_SIGNATURE_DOMAIN: Final = b'VaxReplay checkpoint gossip report v0.1\x00'
_SHA256_PATTERN: Final = r'^[0-9a-f]{64}$'
_MAX_CHECKPOINT_BYTES: Final = 16 * 1024 * 1024
_MAX_POLICY_BYTES: Final = 16 * 1024 * 1024
_MAX_REPORT_BYTES: Final = 64 * 1024 * 1024
_DEFAULT_BUSY_TIMEOUT_SECONDS: Final = 30.0


class CheckpointGossipError(ValueError):
    """A gossip policy, source head, journal, report, or comparison failed closed."""


class RegistryGossipStreamPolicy(StrictModel):
    """Exact registry identity/key ring and out-of-band bootstrap head."""

    schema_version: Literal['vaxreplay.registry-gossip-stream-policy.v0.1'] = (
        GOSSIP_REGISTRY_STREAM_POLICY_SCHEMA_VERSION
    )
    stream_id: str = Field(pattern=SAFE_ID_PATTERN)
    source_kind: Literal['selection_registry'] = 'selection_registry'
    registry_monitor: WitnessRegistryMonitor
    bootstrap_tree_size: int = Field(ge=0)
    bootstrap_signed_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    require_adjacent_tree_sizes: Literal[True] = True
    require_rfc6962_consistency: Literal[True] = True


class WitnessGossipStreamPolicy(StrictModel):
    """Exact witness policy/key and out-of-band bootstrap head."""

    schema_version: Literal['vaxreplay.witness-gossip-stream-policy.v0.2'] = GOSSIP_WITNESS_STREAM_POLICY_SCHEMA_VERSION
    stream_id: str = Field(pattern=SAFE_ID_PATTERN)
    source_kind: Literal['witness_service'] = 'witness_service'
    service_policy: WitnessServicePolicy
    service_trust_policy: WitnessServiceTrustPolicy
    bootstrap_tree_size: int = Field(gt=0)
    bootstrap_signed_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    require_adjacent_tree_sizes: Literal[True] = True

    @model_validator(mode='after')
    def validate_policy_binding(self) -> Self:
        policy_sha256 = hashlib.sha256(canonical_json_bytes(self.service_policy)).hexdigest()
        trust = self.service_trust_policy
        if (
            trust.authority_id != self.service_policy.authority_id
            or trust.witness_id != self.service_policy.witness_id
            or trust.service_policy_sha256 != policy_sha256
        ):
            raise ValueError('witness stream trust does not bind the exact service policy')
        return self


type GossipStreamPolicy = Annotated[
    RegistryGossipStreamPolicy | WitnessGossipStreamPolicy,
    Field(discriminator='source_kind'),
]


class CheckpointGossipMonitorPolicy(StrictModel):
    """Public configuration for one independently operated gossip monitor."""

    schema_version: Literal['vaxreplay.checkpoint-gossip-monitor-policy.v0.3'] = GOSSIP_MONITOR_POLICY_SCHEMA_VERSION
    monitor_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    streams: tuple[GossipStreamPolicy, ...] = Field(min_length=1, max_length=256)
    max_observation_age_seconds: int = Field(ge=1, le=31 * 24 * 60 * 60)
    max_future_clock_skew_seconds: int = Field(default=30, ge=0, le=3600)
    report_signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    report_signing_public_key_base64: str
    report_signing_key_valid_from: datetime
    report_signing_key_valid_until: datetime | None = None
    clock_health_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    clock_health_process_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    external_signer_process_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator('report_signing_public_key_base64')
    @classmethod
    def validate_public_key(cls, value: str) -> str:
        _canonical_base64(value, expected_bytes=32, label='gossip report public key')
        return value

    @field_validator('report_signing_key_valid_from', 'report_signing_key_valid_until')
    @classmethod
    def validate_key_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else aware_utc(value, 'gossip report signing-key validity time')

    @model_validator(mode='after')
    def validate_policy(self) -> Self:
        stream_ids = [stream.stream_id for stream in self.streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError('gossip monitor policy contains duplicate stream IDs')
        if (
            self.report_signing_key_valid_until is not None
            and self.report_signing_key_valid_until <= self.report_signing_key_valid_from
        ):
            raise ValueError('gossip report signing-key valid_until must follow valid_from')
        bindings = (
            self.clock_health_policy_sha256,
            self.clock_health_process_sha256,
            self.external_signer_process_sha256,
        )
        if any(value is None for value in bindings) and any(value is not None for value in bindings):
            raise ValueError('gossip runtime-trust digests must be all present or all null')
        return self


class GossipObservation(StrictModel):
    """One immutable local observation of a source-signed head."""

    schema_version: Literal['vaxreplay.checkpoint-gossip-observation.v0.1'] = GOSSIP_OBSERVATION_SCHEMA_VERSION
    monitor_id: str = Field(pattern=SAFE_ID_PATTERN)
    monitor_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    observation_sequence: int = Field(gt=0)
    previous_observation_sha256: str = Field(pattern=_SHA256_PATTERN)
    stream_id: str = Field(pattern=SAFE_ID_PATTERN)
    source_kind: Literal['selection_registry', 'witness_service']
    source_tree_size: int = Field(ge=0)
    source_root_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_checkpoint_bytes: int = Field(gt=0, le=_MAX_CHECKPOINT_BYTES)
    checkpoint_issued_at: datetime
    observed_at: datetime
    transition: Literal['bootstrap', 'successor', 'heartbeat']
    registry_consistency_proof_sha256: tuple[str, ...] = Field(default=(), max_length=256)

    @field_validator('checkpoint_issued_at', 'observed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return aware_utc(value, 'gossip observation time')

    @field_validator('registry_consistency_proof_sha256')
    @classmethod
    def validate_consistency_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(item) != 64 or any(character not in '0123456789abcdef' for character in item) for item in value):
            raise ValueError('registry consistency path must contain lowercase SHA-256 hex digests')
        return value


class GossipStreamHead(StrictModel):
    """Latest exact source head included in a signed monitor report."""

    stream_id: str = Field(pattern=SAFE_ID_PATTERN)
    stream_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_kind: Literal['selection_registry', 'witness_service']
    source_tree_size: int = Field(ge=0)
    source_root_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_checkpoint_base64: str = Field(min_length=1)
    checkpoint_issued_at: datetime
    observed_at: datetime

    @field_validator('signed_checkpoint_base64')
    @classmethod
    def validate_checkpoint_base64(cls, value: str) -> str:
        decoded = _decode_base64(value, 'signed checkpoint')
        if not decoded or len(decoded) > _MAX_CHECKPOINT_BYTES:
            raise ValueError('signed checkpoint has an invalid decoded size')
        if base64.b64encode(decoded).decode('ascii') != value:
            raise ValueError('signed checkpoint must use canonical base64')
        return value

    @field_validator('checkpoint_issued_at', 'observed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return aware_utc(value, 'gossip stream-head time')


class GossipMonitorReport(StrictModel):
    """Fresh monitor view signed by its independently pinned report key."""

    schema_version: Literal['vaxreplay.checkpoint-gossip-monitor-report.v0.1'] = GOSSIP_MONITOR_REPORT_SCHEMA_VERSION
    monitor_id: str = Field(pattern=SAFE_ID_PATTERN)
    monitor_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    monitor_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    generated_at: datetime
    through_observation_sequence: int = Field(gt=0)
    through_observation_sha256: str = Field(pattern=_SHA256_PATTERN)
    stream_heads: tuple[GossipStreamHead, ...] = Field(min_length=1, max_length=256)
    local_signatures_verified: Literal[True] = True
    local_hash_chain_verified: Literal[True] = True
    stale_observations_rejected: Literal[True] = True

    @field_validator('generated_at')
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'gossip report generation time')

    @model_validator(mode='after')
    def validate_stream_ids(self) -> Self:
        stream_ids = [head.stream_id for head in self.stream_heads]
        if stream_ids != sorted(stream_ids) or len(stream_ids) != len(set(stream_ids)):
            raise ValueError('gossip report stream heads must be unique and sorted by stream_id')
        return self


class SignedGossipMonitorReport(StrictModel):
    """Authenticated monitor report suitable for cross-network comparison."""

    schema_version: Literal['vaxreplay.signed-checkpoint-gossip-monitor-report.v0.1'] = (
        SIGNED_GOSSIP_MONITOR_REPORT_SCHEMA_VERSION
    )
    report: GossipMonitorReport
    signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    signature_base64: str

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        _canonical_base64(value, expected_bytes=64, label='gossip report signature')
        return value


class GossipMonitorPolicyPin(StrictModel):
    monitor_id: str = Field(pattern=SAFE_ID_PATTERN)
    monitor_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    monitor_policy: CheckpointGossipMonitorPolicy

    @model_validator(mode='after')
    def validate_pin(self) -> Self:
        if self.monitor_id != self.monitor_policy.monitor_id:
            raise ValueError('gossip comparison pin monitor_id differs from embedded policy')
        if self.monitor_policy_sha256 != hashlib.sha256(canonical_json_bytes(self.monitor_policy)).hexdigest():
            raise ValueError('gossip comparison pin differs from the exact embedded monitor policy')
        return self


class GossipComparisonPolicy(StrictModel):
    """Exact quorum and freshness rules for fail-closed report comparison."""

    schema_version: Literal['vaxreplay.checkpoint-gossip-comparison-policy.v0.2'] = (
        GOSSIP_COMPARISON_POLICY_SCHEMA_VERSION
    )
    comparison_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    monitors: tuple[GossipMonitorPolicyPin, ...] = Field(min_length=2, max_length=32)
    required_stream_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    max_report_age_seconds: int = Field(ge=1, le=31 * 24 * 60 * 60)
    max_observation_age_seconds: int = Field(ge=1, le=31 * 24 * 60 * 60)
    max_future_clock_skew_seconds: int = Field(default=30, ge=0, le=3600)
    require_exact_latest_head_agreement: Literal[True] = True

    @field_validator('required_stream_ids')
    @classmethod
    def validate_required_stream_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('required gossip stream IDs must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_monitor_quorum(self) -> Self:
        monitor_ids = [pin.monitor_id for pin in self.monitors]
        if len(monitor_ids) != len(set(monitor_ids)):
            raise ValueError('gossip comparison policy contains duplicate monitor IDs')
        expected_stream_ids = set(self.required_stream_ids)
        reference: dict[str, bytes] | None = None
        for pin in self.monitors:
            streams = {stream.stream_id: canonical_json_bytes(stream) for stream in pin.monitor_policy.streams}
            if set(streams) != expected_stream_ids:
                raise ValueError('every gossip monitor must configure exactly the required streams')
            if self.max_observation_age_seconds > pin.monitor_policy.max_observation_age_seconds:
                raise ValueError('comparison freshness cannot be weaker than a monitor freshness policy')
            if self.max_future_clock_skew_seconds > pin.monitor_policy.max_future_clock_skew_seconds:
                raise ValueError('comparison clock skew cannot be weaker than a monitor clock-skew policy')
            if reference is None:
                reference = streams
            elif streams != reference:
                raise ValueError('gossip monitors must pin byte-identical source stream policies')
        return self


class GossipAgreedStreamHead(StrictModel):
    stream_id: str = Field(pattern=SAFE_ID_PATTERN)
    source_kind: Literal['selection_registry', 'witness_service']
    source_tree_size: int = Field(ge=0)
    source_root_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)


class GossipAgreementReport(StrictModel):
    """Evidence that every required fresh monitor reported the same latest heads."""

    schema_version: Literal['vaxreplay.checkpoint-gossip-agreement-report.v0.1'] = (
        GOSSIP_AGREEMENT_REPORT_SCHEMA_VERSION
    )
    comparison_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    comparison_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    compared_at: datetime
    monitor_ids: tuple[str, ...] = Field(min_length=2, max_length=32)
    stream_heads: tuple[GossipAgreedStreamHead, ...] = Field(min_length=1, max_length=256)
    report_signatures_verified: Literal[True] = True
    source_signatures_verified: Literal[True] = True
    exact_latest_heads_agree: Literal[True] = True
    stale_reports_rejected: Literal[True] = True

    @field_validator('compared_at')
    @classmethod
    def validate_compared_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'gossip comparison time')


class GossipVerificationReport(StrictModel):
    """Local full-journal replay result."""

    schema_version: Literal['vaxreplay.checkpoint-gossip-verification-report.v0.1'] = (
        GOSSIP_VERIFICATION_REPORT_SCHEMA_VERSION
    )
    monitor_id: str = Field(pattern=SAFE_ID_PATTERN)
    observation_count: int = Field(ge=0)
    through_observation_sha256: str = Field(pattern=_SHA256_PATTERN)
    stream_count: int = Field(ge=0)
    signatures_verified: Literal[True] = True
    hash_chain_verified: Literal[True] = True
    source_transitions_verified: Literal[True] = True


class _VerifiedSourceHead(StrictModel):
    source_kind: Literal['selection_registry', 'witness_service']
    tree_size: int = Field(ge=0)
    root_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    predecessor_sha256: str | None

    @field_validator('issued_at')
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'source checkpoint issuance time')


class _ReplayState:
    def __init__(self) -> None:
        self.observation_count = 0
        self.through_sha256 = ZERO_SHA256
        self.last_observed_at: datetime | None = None
        self.latest_by_stream: dict[str, tuple[GossipObservation, bytes]] = {}


_DATABASE_SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
) STRICT;
CREATE TABLE observations (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    observation_sha256 TEXT NOT NULL UNIQUE,
    previous_observation_sha256 TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    source_tree_size INTEGER NOT NULL CHECK (source_tree_size >= 0),
    signed_checkpoint_sha256 TEXT NOT NULL,
    observation_bytes BLOB NOT NULL,
    signed_checkpoint_bytes BLOB NOT NULL
) STRICT;
CREATE TRIGGER metadata_no_update BEFORE UPDATE ON metadata BEGIN
    SELECT RAISE(ABORT, 'gossip metadata is immutable');
END;
CREATE TRIGGER metadata_no_delete BEFORE DELETE ON metadata BEGIN
    SELECT RAISE(ABORT, 'gossip metadata is immutable');
END;
CREATE TRIGGER observations_no_update BEFORE UPDATE ON observations BEGIN
    SELECT RAISE(ABORT, 'gossip journal is append-only');
END;
CREATE TRIGGER observations_no_delete BEFORE DELETE ON observations BEGIN
    SELECT RAISE(ABORT, 'gossip journal is append-only');
END;
"""


class CheckpointGossipMonitorStore:
    """Owner-only durable append-only checkpoint monitor state."""

    def __init__(
        self,
        root: Path,
        *,
        busy_timeout_seconds: float = _DEFAULT_BUSY_TIMEOUT_SECONDS,
        clock: Callable[[], datetime] | None = None,
        signer: Ed25519Signer | None = None,
        clock_health_gate: ClockHealthGate | None = None,
    ) -> None:
        if (
            not isinstance(busy_timeout_seconds, (int, float))
            or isinstance(busy_timeout_seconds, bool)
            or busy_timeout_seconds <= 0
            or busy_timeout_seconds > 300
        ):
            raise CheckpointGossipError('gossip busy timeout must be numeric and in (0, 300]')
        self.root = _validate_root(root)
        if signer is not None and clock_health_gate is None:
            raise CheckpointGossipError('isolated gossip signers require a fail-closed clock-health gate')
        self.busy_timeout_seconds = float(busy_timeout_seconds)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._clock_health_gate = clock_health_gate
        self.policy_bytes = _read_regular_nofollow(
            self.root / _POLICY_PATH,
            maximum=_MAX_POLICY_BYTES,
            private=False,
        )
        self.policy = _load_canonical_model(
            self.policy_bytes,
            CheckpointGossipMonitorPolicy,
            'gossip monitor policy',
        )
        self.policy_sha256 = hashlib.sha256(self.policy_bytes).hexdigest()
        expected_public_key = _decode_base64(
            self.policy.report_signing_public_key_base64,
            'gossip report public key',
        )
        if signer is None:
            private_key_bytes = _read_regular_nofollow(
                self.root / _PRIVATE_KEY_PATH,
                maximum=32,
                private=True,
            )
            if len(private_key_bytes) != 32:
                raise CheckpointGossipError('gossip report private key must contain exactly 32 bytes')
            try:
                signer = LocalEd25519Signer(Ed25519PrivateKey.from_private_bytes(private_key_bytes))
            except ValueError as error:
                raise CheckpointGossipError('invalid gossip report private key') from error
        try:
            self._signer = checked_signer(signer, expected_public_key=expected_public_key)
        except ValueError as error:
            raise CheckpointGossipError(f'gossip report signer is invalid: {error}') from error
        _validate_database_file(self.root / _DATABASE_PATH)
        with closing(self._connect()) as connection:
            self._verify_metadata(connection)
        self.verify()

    @classmethod
    def initialize(
        cls,
        root: Path,
        *,
        policy: CheckpointGossipMonitorPolicy,
        report_signing_private_key: bytes | None = None,
        busy_timeout_seconds: float = _DEFAULT_BUSY_TIMEOUT_SECONDS,
        clock: Callable[[], datetime] | None = None,
        signer: Ed25519Signer | None = None,
        clock_health_gate: ClockHealthGate | None = None,
    ) -> CheckpointGossipMonitorStore:
        """Create a no-replace monitor root bound to exact policy and key bytes."""

        policy = CheckpointGossipMonitorPolicy.model_validate_json(canonical_json_bytes(policy))
        policy_bytes = canonical_json_bytes(policy)
        if signer is not None and clock_health_gate is None:
            raise CheckpointGossipError('isolated gossip signers require a fail-closed clock-health gate')
        if (report_signing_private_key is None) == (signer is None):
            raise CheckpointGossipError('provide exactly one local private key or isolated signer')
        if report_signing_private_key is not None:
            if not isinstance(report_signing_private_key, bytes) or len(report_signing_private_key) != 32:
                raise CheckpointGossipError('gossip report private key must contain exactly 32 bytes')
            try:
                private_key = Ed25519PrivateKey.from_private_bytes(report_signing_private_key)
            except ValueError as error:
                raise CheckpointGossipError('invalid gossip report private key') from error
            active_signer: Ed25519Signer = LocalEd25519Signer(private_key)
        else:
            if signer is None:  # narrowed above; keep static analyzers fail closed
                raise CheckpointGossipError('isolated signer is missing')
            active_signer = checked_signer(signer)
        public_key = active_signer.public_key_bytes()
        if not hmac.compare_digest(
            public_key,
            _decode_base64(policy.report_signing_public_key_base64, 'gossip report public key'),
        ):
            raise CheckpointGossipError('gossip report private key does not match monitor policy')

        requested = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
        requested.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent = requested.parent.resolve(strict=True)
        target = parent / requested.name
        if os.path.lexists(target):
            raise CheckpointGossipError(f'gossip monitor root already exists: {target}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=parent))
        try:
            staging.chmod(0o700)
            _write_durable_file(staging / _POLICY_PATH, policy_bytes, mode=0o644)
            if report_signing_private_key is not None:
                _write_durable_file(
                    staging / _PRIVATE_KEY_PATH,
                    report_signing_private_key,
                    mode=0o600,
                )
            database_path = staging / _DATABASE_PATH
            connection = sqlite3.connect(database_path, isolation_level=None)
            try:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=FULL')
                connection.executescript(_DATABASE_SCHEMA)
                connection.execute('BEGIN IMMEDIATE')
                metadata = {
                    'schema_version': GOSSIP_DATABASE_SCHEMA_VERSION,
                    'monitor_id': policy.monitor_id,
                    'policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
                    'report_public_key_sha256': hashlib.sha256(public_key).hexdigest(),
                }
                connection.executemany(
                    'INSERT INTO metadata(key, value) VALUES (?, ?)',
                    tuple((key, value.encode('ascii')) for key, value in metadata.items()),
                )
                connection.execute('COMMIT')
                connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            except BaseException:
                if connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise
            finally:
                connection.close()
            database_path.chmod(0o600)
            _fsync_file(database_path)
            fsync_directory(staging)
            try:
                rename_directory_noreplace(staging, target)
            except FileExistsError as error:
                raise CheckpointGossipError(f'gossip monitor root already exists: {target}') from error
            staging = Path()
            fsync_directory(parent)
        finally:
            if staging != Path() and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return cls(
            target,
            busy_timeout_seconds=busy_timeout_seconds,
            clock=clock,
            signer=None if report_signing_private_key is not None else active_signer,
            clock_health_gate=clock_health_gate,
        )

    def observe(
        self,
        stream_id: str,
        signed_checkpoint_bytes: bytes,
        *,
        registry_consistency_proof_sha256: Sequence[str] = (),
    ) -> GossipObservation:
        """Verify and append one latest-head poll using the monitor's own clock."""

        stream = self._stream(stream_id)
        checkpoint_bytes = _exact_bytes(
            signed_checkpoint_bytes,
            'signed checkpoint',
            maximum=_MAX_CHECKPOINT_BYTES,
        )
        proof = tuple(registry_consistency_proof_sha256)
        if isinstance(stream, WitnessGossipStreamPolicy) and proof:
            raise CheckpointGossipError('witness checkpoint observations cannot carry a registry consistency path')

        with closing(self._connect()) as connection:
            try:
                connection.execute('BEGIN IMMEDIATE')
                state = self._replay(connection)
                observed_at = _clock_utc(self._clock)
                require_clock_health(self._clock_health_gate, security_time=observed_at)
                if state.last_observed_at is not None and observed_at < state.last_observed_at:
                    raise CheckpointGossipError('gossip monitor clock moved backwards')
                head = _verify_source_head(stream, checkpoint_bytes)
                if head.issued_at > observed_at + timedelta(seconds=self.policy.max_future_clock_skew_seconds):
                    raise CheckpointGossipError('source checkpoint issuance time is implausibly in the future')
                previous = state.latest_by_stream.get(stream_id)
                transition = _verify_transition(stream, previous, head, checkpoint_bytes, proof)
                sequence = state.observation_count + 1
                observation = GossipObservation(
                    monitor_id=self.policy.monitor_id,
                    monitor_policy_sha256=self.policy_sha256,
                    observation_sequence=sequence,
                    previous_observation_sha256=state.through_sha256,
                    stream_id=stream.stream_id,
                    source_kind=head.source_kind,
                    source_tree_size=head.tree_size,
                    source_root_sha256=head.root_sha256,
                    signed_checkpoint_sha256=head.signed_checkpoint_sha256,
                    signed_checkpoint_bytes=len(checkpoint_bytes),
                    checkpoint_issued_at=head.issued_at,
                    observed_at=observed_at,
                    transition=transition,
                    registry_consistency_proof_sha256=proof,
                )
                observation_bytes = canonical_json_bytes(observation)
                observation_sha256 = hashlib.sha256(observation_bytes).hexdigest()
                connection.execute(
                    'INSERT INTO observations('
                    'sequence, observation_sha256, previous_observation_sha256, stream_id, '
                    'source_tree_size, signed_checkpoint_sha256, observation_bytes, signed_checkpoint_bytes'
                    ') VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        sequence,
                        observation_sha256,
                        state.through_sha256,
                        stream_id,
                        head.tree_size,
                        head.signed_checkpoint_sha256,
                        observation_bytes,
                        checkpoint_bytes,
                    ),
                )
                connection.execute('COMMIT')
                return observation
            except BaseException:
                if connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise

    def verify(self) -> GossipVerificationReport:
        """Replay every source signature, transition, and local journal hash link."""

        with closing(self._connect()) as connection:
            self._verify_metadata(connection)
            state = self._replay(connection)
        return GossipVerificationReport(
            monitor_id=self.policy.monitor_id,
            observation_count=state.observation_count,
            through_observation_sha256=state.through_sha256,
            stream_count=len(state.latest_by_stream),
        )

    def signed_report(self) -> SignedGossipMonitorReport:
        """Build a fresh, source-complete view and sign it with the monitor key."""

        with closing(self._connect()) as connection:
            self._verify_metadata(connection)
            state = self._replay(connection)
        generated_at = _clock_utc(self._clock)
        require_clock_health(self._clock_health_gate, security_time=generated_at)
        if state.last_observed_at is not None and generated_at < state.last_observed_at:
            raise CheckpointGossipError('gossip monitor clock moved backwards')
        if generated_at < self.policy.report_signing_key_valid_from or (
            self.policy.report_signing_key_valid_until is not None
            and generated_at >= self.policy.report_signing_key_valid_until
        ):
            raise CheckpointGossipError('gossip report signing key is not valid at report generation time')
        expected_stream_ids = {stream.stream_id for stream in self.policy.streams}
        if set(state.latest_by_stream) != expected_stream_ids:
            missing = sorted(expected_stream_ids - set(state.latest_by_stream))
            raise CheckpointGossipError(f'gossip report is missing required stream observations: {missing}')
        heads: list[GossipStreamHead] = []
        for stream in sorted(self.policy.streams, key=lambda item: item.stream_id):
            observation, checkpoint_bytes = state.latest_by_stream[stream.stream_id]
            if generated_at - observation.observed_at > timedelta(seconds=self.policy.max_observation_age_seconds):
                raise CheckpointGossipError(f'gossip observation for stream {stream.stream_id!r} is stale')
            if observation.observed_at > generated_at + timedelta(seconds=self.policy.max_future_clock_skew_seconds):
                raise CheckpointGossipError('gossip observation time is implausibly in the future')
            heads.append(
                GossipStreamHead(
                    stream_id=stream.stream_id,
                    stream_policy_sha256=hashlib.sha256(canonical_json_bytes(stream)).hexdigest(),
                    source_kind=observation.source_kind,
                    source_tree_size=observation.source_tree_size,
                    source_root_sha256=observation.source_root_sha256,
                    signed_checkpoint_sha256=observation.signed_checkpoint_sha256,
                    signed_checkpoint_base64=base64.b64encode(checkpoint_bytes).decode('ascii'),
                    checkpoint_issued_at=observation.checkpoint_issued_at,
                    observed_at=observation.observed_at,
                )
            )
        report = GossipMonitorReport(
            monitor_id=self.policy.monitor_id,
            monitor_policy_id=self.policy.policy_id,
            monitor_policy_sha256=self.policy_sha256,
            generated_at=generated_at,
            through_observation_sequence=state.observation_count,
            through_observation_sha256=state.through_sha256,
            stream_heads=tuple(heads),
        )
        signature = self._signer.sign(_REPORT_SIGNATURE_DOMAIN + canonical_json_bytes(report))
        return SignedGossipMonitorReport(
            report=report,
            signing_key_id=self.policy.report_signing_key_id,
            signature_base64=base64.b64encode(signature).decode('ascii'),
        )

    def _connect(self) -> sqlite3.Connection:
        _validate_database_file(self.root / _DATABASE_PATH)
        connection = sqlite3.connect(
            self.root / _DATABASE_PATH,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA trusted_schema=OFF')
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute(f'PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}')
        return connection

    def _verify_metadata(self, connection: sqlite3.Connection) -> None:
        integrity = connection.execute('PRAGMA quick_check').fetchone()
        if integrity is None or integrity[0] != 'ok':
            raise CheckpointGossipError('gossip database failed SQLite integrity verification')
        public_key = self._signer.public_key_bytes()
        expected = {
            'schema_version': GOSSIP_DATABASE_SCHEMA_VERSION,
            'monitor_id': self.policy.monitor_id,
            'policy_sha256': self.policy_sha256,
            'report_public_key_sha256': hashlib.sha256(public_key).hexdigest(),
        }
        rows = connection.execute('SELECT key, value FROM metadata ORDER BY key').fetchall()
        actual = {str(row['key']): bytes(row['value']).decode('ascii') for row in rows}
        if actual != expected:
            raise CheckpointGossipError('gossip database metadata differs from installed policy/key')

    def _replay(self, connection: sqlite3.Connection) -> _ReplayState:
        state = _ReplayState()
        rows = connection.execute(
            'SELECT sequence, observation_sha256, previous_observation_sha256, stream_id, '
            'source_tree_size, signed_checkpoint_sha256, observation_bytes, signed_checkpoint_bytes '
            'FROM observations ORDER BY sequence'
        ).fetchall()
        for expected_sequence, row in enumerate(rows, start=1):
            if int(row['sequence']) != expected_sequence:
                raise CheckpointGossipError('gossip observation sequence is not contiguous')
            observation_bytes = bytes(row['observation_bytes'])
            observation = _load_canonical_model(
                observation_bytes,
                GossipObservation,
                f'gossip observation {expected_sequence}',
            )
            checkpoint_bytes = _exact_bytes(
                bytes(row['signed_checkpoint_bytes']),
                f'gossip checkpoint {expected_sequence}',
                maximum=_MAX_CHECKPOINT_BYTES,
            )
            observation_sha256 = hashlib.sha256(observation_bytes).hexdigest()
            if (
                observation.monitor_id != self.policy.monitor_id
                or observation.monitor_policy_sha256 != self.policy_sha256
                or observation.observation_sequence != expected_sequence
                or observation.previous_observation_sha256 != state.through_sha256
                or str(row['observation_sha256']) != observation_sha256
                or str(row['previous_observation_sha256']) != state.through_sha256
                or str(row['stream_id']) != observation.stream_id
                or int(row['source_tree_size']) != observation.source_tree_size
                or str(row['signed_checkpoint_sha256']) != observation.signed_checkpoint_sha256
                or len(checkpoint_bytes) != observation.signed_checkpoint_bytes
                or hashlib.sha256(checkpoint_bytes).hexdigest() != observation.signed_checkpoint_sha256
            ):
                raise CheckpointGossipError(f'gossip database row {expected_sequence} is internally inconsistent')
            if state.last_observed_at is not None and observation.observed_at < state.last_observed_at:
                raise CheckpointGossipError('gossip observation clock moved backwards in the journal')
            stream = self._stream(observation.stream_id)
            head = _verify_source_head(stream, checkpoint_bytes)
            if (
                head.source_kind != observation.source_kind
                or head.tree_size != observation.source_tree_size
                or head.root_sha256 != observation.source_root_sha256
                or head.signed_checkpoint_sha256 != observation.signed_checkpoint_sha256
                or head.issued_at != observation.checkpoint_issued_at
            ):
                raise CheckpointGossipError('gossip observation metadata differs from its exact source head')
            previous = state.latest_by_stream.get(observation.stream_id)
            transition = _verify_transition(
                stream,
                previous,
                head,
                checkpoint_bytes,
                observation.registry_consistency_proof_sha256,
            )
            if transition != observation.transition:
                raise CheckpointGossipError('gossip observation transition classification is inconsistent')
            state.observation_count = expected_sequence
            state.through_sha256 = observation_sha256
            state.last_observed_at = observation.observed_at
            state.latest_by_stream[observation.stream_id] = (observation, checkpoint_bytes)
        return state

    def _stream(self, stream_id: str) -> GossipStreamPolicy:
        matches = [stream for stream in self.policy.streams if stream.stream_id == stream_id]
        if len(matches) != 1:
            raise CheckpointGossipError(f'gossip stream {stream_id!r} is not uniquely configured')
        return matches[0]


def verify_gossip_agreement(
    signed_report_bytes: Sequence[bytes],
    comparison_policy_bytes: bytes,
    *,
    now: datetime | None = None,
) -> GossipAgreementReport:
    """Authenticate all required monitor reports and require exact latest-head agreement."""

    policy = _load_canonical_model(
        _exact_bytes(
            comparison_policy_bytes,
            'gossip comparison policy',
            maximum=_MAX_POLICY_BYTES,
        ),
        GossipComparisonPolicy,
        'gossip comparison policy',
    )
    compared_at = aware_utc(now or datetime.now(timezone.utc), 'gossip comparison time')
    if len(signed_report_bytes) != len(policy.monitors):
        raise CheckpointGossipError('gossip comparison requires exactly one report from every monitor')
    pins = {pin.monitor_id: pin for pin in policy.monitors}
    reports: dict[str, GossipMonitorReport] = {}
    for index, payload in enumerate(signed_report_bytes, start=1):
        signed = _load_canonical_model(
            _exact_bytes(payload, f'signed gossip report {index}', maximum=_MAX_REPORT_BYTES),
            SignedGossipMonitorReport,
            f'signed gossip report {index}',
        )
        monitor_id = signed.report.monitor_id
        if monitor_id in reports:
            raise CheckpointGossipError('gossip comparison received duplicate monitor reports')
        pin = pins.get(monitor_id)
        if pin is None:
            raise CheckpointGossipError(f'gossip report came from unconfigured monitor {monitor_id!r}')
        _verify_signed_report(signed, pin.monitor_policy, compared_at, policy)
        reports[monitor_id] = signed.report
    if set(reports) != set(pins):
        raise CheckpointGossipError('gossip comparison is missing a required monitor report')

    agreed: list[GossipAgreedStreamHead] = []
    for stream_id in policy.required_stream_ids:
        values: set[tuple[str, int, str, str]] = set()
        for report in reports.values():
            matches = [head for head in report.stream_heads if head.stream_id == stream_id]
            if len(matches) != 1:
                raise CheckpointGossipError('gossip report does not contain exactly one required stream head')
            head = matches[0]
            values.add(
                (
                    head.source_kind,
                    head.source_tree_size,
                    head.source_root_sha256,
                    head.signed_checkpoint_sha256,
                )
            )
        if len(values) != 1:
            raise CheckpointGossipError(f'gossip monitors disagree on latest head for stream {stream_id!r}')
        source_kind, tree_size, root_sha256, checkpoint_sha256 = values.pop()
        if source_kind not in {'selection_registry', 'witness_service'}:
            raise CheckpointGossipError('gossip monitor reported an unknown source kind')
        typed_source_kind = cast(Literal['selection_registry', 'witness_service'], source_kind)
        agreed.append(
            GossipAgreedStreamHead(
                stream_id=stream_id,
                source_kind=typed_source_kind,
                source_tree_size=tree_size,
                source_root_sha256=root_sha256,
                signed_checkpoint_sha256=checkpoint_sha256,
            )
        )
    return GossipAgreementReport(
        comparison_policy_id=policy.comparison_policy_id,
        comparison_policy_sha256=hashlib.sha256(comparison_policy_bytes).hexdigest(),
        compared_at=compared_at,
        monitor_ids=tuple(sorted(reports)),
        stream_heads=tuple(agreed),
    )


def verify_gossip_bootstrap_head(
    stream: RegistryGossipStreamPolicy | WitnessGossipStreamPolicy,
    signed_checkpoint_bytes: bytes,
) -> GossipAgreedStreamHead:
    """Authenticate one exact out-of-band bootstrap under its stream policy."""

    if not isinstance(stream, (RegistryGossipStreamPolicy, WitnessGossipStreamPolicy)):
        raise CheckpointGossipError('gossip bootstrap stream policy has an invalid type')
    canonical_stream = type(stream).model_validate_json(canonical_json_bytes(stream))
    payload = _exact_bytes(
        signed_checkpoint_bytes,
        'gossip bootstrap signed checkpoint',
        maximum=_MAX_CHECKPOINT_BYTES,
    )
    head = _verify_source_head(canonical_stream, payload)
    if (
        head.tree_size != canonical_stream.bootstrap_tree_size
        or head.signed_checkpoint_sha256 != canonical_stream.bootstrap_signed_checkpoint_sha256
    ):
        raise CheckpointGossipError('gossip bootstrap differs from the exact stream-policy pin')
    return GossipAgreedStreamHead(
        stream_id=canonical_stream.stream_id,
        source_kind=head.source_kind,
        source_tree_size=head.tree_size,
        source_root_sha256=head.root_sha256,
        signed_checkpoint_sha256=head.signed_checkpoint_sha256,
    )


def _verify_signed_report(
    signed: SignedGossipMonitorReport,
    monitor_policy: CheckpointGossipMonitorPolicy,
    compared_at: datetime,
    comparison_policy: GossipComparisonPolicy,
) -> None:
    report = signed.report
    monitor_policy_bytes = canonical_json_bytes(monitor_policy)
    if (
        report.monitor_id != monitor_policy.monitor_id
        or report.monitor_policy_id != monitor_policy.policy_id
        or report.monitor_policy_sha256 != hashlib.sha256(monitor_policy_bytes).hexdigest()
        or signed.signing_key_id != monitor_policy.report_signing_key_id
    ):
        raise CheckpointGossipError('gossip report does not match the exact pinned monitor policy')
    if report.generated_at < monitor_policy.report_signing_key_valid_from or (
        monitor_policy.report_signing_key_valid_until is not None
        and report.generated_at >= monitor_policy.report_signing_key_valid_until
    ):
        raise CheckpointGossipError('gossip report was signed outside its pinned key validity interval')
    try:
        Ed25519PublicKey.from_public_bytes(
            _decode_base64(
                monitor_policy.report_signing_public_key_base64,
                'gossip report public key',
            )
        ).verify(
            _decode_base64(signed.signature_base64, 'gossip report signature'),
            _REPORT_SIGNATURE_DOMAIN + canonical_json_bytes(report),
        )
    except (InvalidSignature, ValueError) as error:
        raise CheckpointGossipError('gossip monitor report signature verification failed') from error
    future_limit = compared_at + timedelta(seconds=comparison_policy.max_future_clock_skew_seconds)
    if report.generated_at > future_limit:
        raise CheckpointGossipError('gossip monitor report generation time is implausibly in the future')
    if compared_at - report.generated_at > timedelta(seconds=comparison_policy.max_report_age_seconds):
        raise CheckpointGossipError('gossip monitor report is stale')
    streams = {stream.stream_id: stream for stream in monitor_policy.streams}
    if {head.stream_id for head in report.stream_heads} != set(comparison_policy.required_stream_ids):
        raise CheckpointGossipError('gossip monitor report stream inventory differs from comparison policy')
    for head in report.stream_heads:
        stream = streams.get(head.stream_id)
        if stream is None:
            raise CheckpointGossipError('gossip monitor report contains an unconfigured stream')
        if head.stream_policy_sha256 != hashlib.sha256(canonical_json_bytes(stream)).hexdigest():
            raise CheckpointGossipError('gossip stream head does not bind its exact source policy')
        if head.observed_at > future_limit:
            raise CheckpointGossipError('gossip observation time is implausibly in the future')
        if head.observed_at > report.generated_at:
            raise CheckpointGossipError('gossip observation claims to occur after report generation')
        if compared_at - head.observed_at > timedelta(seconds=comparison_policy.max_observation_age_seconds):
            raise CheckpointGossipError('gossip monitor report contains a stale observation')
        checkpoint_bytes = _decode_base64(head.signed_checkpoint_base64, 'signed checkpoint')
        verified = _verify_source_head(stream, checkpoint_bytes)
        if (
            verified.source_kind != head.source_kind
            or verified.tree_size != head.source_tree_size
            or verified.root_sha256 != head.source_root_sha256
            or verified.signed_checkpoint_sha256 != head.signed_checkpoint_sha256
            or verified.issued_at != head.checkpoint_issued_at
        ):
            raise CheckpointGossipError('reported stream metadata differs from its source-signed checkpoint')
        if verified.issued_at > head.observed_at + timedelta(seconds=monitor_policy.max_future_clock_skew_seconds):
            raise CheckpointGossipError('source checkpoint issuance time is after its monitor observation')


def _verify_source_head(stream: GossipStreamPolicy, payload: bytes) -> _VerifiedSourceHead:
    signed_sha256 = hashlib.sha256(payload).hexdigest()
    if isinstance(stream, RegistryGossipStreamPolicy):
        signed = _load_canonical_model(
            payload,
            WitnessedSignedRegistryCheckpoint,
            'signed registry checkpoint',
        )
        checkpoint = signed.checkpoint
        monitor = stream.registry_monitor
        if checkpoint.registry_id != monitor.registry_id or checkpoint.authority_id != monitor.authority_id:
            raise CheckpointGossipError('registry checkpoint differs from the pinned stream identity')
        keys = [key for key in monitor.signing_keys if key.key_id == checkpoint.signing_key_id]
        if len(keys) != 1:
            raise CheckpointGossipError('registry checkpoint signer is not uniquely pinned')
        key = keys[0]
        if checkpoint.issued_at_upper_bound < key.valid_from or (
            key.valid_until is not None and checkpoint.issued_at_upper_bound >= key.valid_until
        ):
            raise CheckpointGossipError('registry checkpoint signer was not valid at issuance')
        try:
            Ed25519PublicKey.from_public_bytes(
                _decode_base64(key.public_key_base64, 'registry checkpoint public key')
            ).verify(
                _decode_base64(signed.signature_base64, 'registry checkpoint signature'),
                canonical_json_bytes(checkpoint),
            )
        except (InvalidSignature, ValueError) as error:
            raise CheckpointGossipError('registry checkpoint signature verification failed') from error
        return _VerifiedSourceHead(
            source_kind='selection_registry',
            tree_size=checkpoint.tree_size,
            root_sha256=checkpoint.root_sha256,
            signed_checkpoint_sha256=signed_sha256,
            issued_at=checkpoint.issued_at_upper_bound,
            predecessor_sha256=checkpoint.previous_checkpoint_sha256,
        )

    try:
        signed_witness = verify_witness_service_signed_checkpoint(
            payload,
            policy_bytes=canonical_json_bytes(stream.service_policy),
            trust_policy_bytes=canonical_json_bytes(stream.service_trust_policy),
        )
    except ValueError as error:
        raise CheckpointGossipError(f'witness checkpoint verification failed: {error}') from error
    checkpoint = signed_witness.checkpoint
    return _VerifiedSourceHead(
        source_kind='witness_service',
        tree_size=checkpoint.tree_size,
        root_sha256=checkpoint.through_entry_sha256,
        signed_checkpoint_sha256=signed_sha256,
        issued_at=checkpoint.issued_at,
        predecessor_sha256=checkpoint.previous_checkpoint_sha256,
    )


def _verify_transition(
    stream: GossipStreamPolicy,
    previous: tuple[GossipObservation, bytes] | None,
    current: _VerifiedSourceHead,
    current_bytes: bytes,
    consistency_proof_sha256: tuple[str, ...],
) -> Literal['bootstrap', 'successor', 'heartbeat']:
    if previous is None:
        if (
            current.tree_size != stream.bootstrap_tree_size
            or current.signed_checkpoint_sha256 != stream.bootstrap_signed_checkpoint_sha256
        ):
            raise CheckpointGossipError('first gossip observation differs from its out-of-band bootstrap head')
        if consistency_proof_sha256:
            raise CheckpointGossipError('bootstrap observation cannot carry a predecessor consistency path')
        return 'bootstrap'

    previous_observation, previous_bytes = previous
    previous_head = _verify_source_head(stream, previous_bytes)
    if current.tree_size < previous_head.tree_size:
        raise CheckpointGossipError('source checkpoint rollback detected')
    if current.tree_size == previous_head.tree_size:
        if (
            current.root_sha256 != previous_head.root_sha256
            or current.signed_checkpoint_sha256 != previous_head.signed_checkpoint_sha256
            or not hmac.compare_digest(current_bytes, previous_bytes)
        ):
            raise CheckpointGossipError('same-sequence conflicting source checkpoint detected')
        if consistency_proof_sha256:
            raise CheckpointGossipError('heartbeat observation cannot carry a consistency path')
        return 'heartbeat'
    if current.tree_size != previous_head.tree_size + 1:
        raise CheckpointGossipError('source checkpoint predecessor chain has a tree-size gap')
    if current.issued_at < previous_head.issued_at:
        raise CheckpointGossipError('source checkpoint issuance time moved backwards')
    if isinstance(stream, RegistryGossipStreamPolicy):
        previous_signed = _load_canonical_model(
            previous_bytes,
            WitnessedSignedRegistryCheckpoint,
            'previous signed registry checkpoint',
        )
        expected_predecessor = hashlib.sha256(
            canonical_json_bytes(previous_signed.checkpoint)
            + _decode_base64(previous_signed.signature_base64, 'registry checkpoint signature')
        ).hexdigest()
        if current.predecessor_sha256 != expected_predecessor:
            raise CheckpointGossipError('registry checkpoint does not bind the exact predecessor')
        if not _verify_rfc6962_consistency(
            old_size=previous_head.tree_size,
            new_size=current.tree_size,
            old_root=bytes.fromhex(previous_head.root_sha256),
            new_root=bytes.fromhex(current.root_sha256),
            proof=tuple(bytes.fromhex(item) for item in consistency_proof_sha256),
        ):
            raise CheckpointGossipError('registry checkpoint RFC6962 consistency proof is invalid')
    else:
        previous_signed_witness = _load_canonical_model(
            previous_bytes,
            WitnessServiceSignedCheckpoint,
            'previous signed witness checkpoint',
        )
        expected_predecessor = hashlib.sha256(canonical_json_bytes(previous_signed_witness.checkpoint)).hexdigest()
        if current.predecessor_sha256 != expected_predecessor:
            raise CheckpointGossipError('witness checkpoint does not bind the exact predecessor')
        if consistency_proof_sha256:
            raise CheckpointGossipError('witness checkpoint successor cannot carry a registry consistency path')
    return 'successor'


def _verify_rfc6962_consistency(
    *,
    old_size: int,
    new_size: int,
    old_root: bytes,
    new_root: bytes,
    proof: tuple[bytes, ...],
) -> bool:
    """Verify an RFC 6962 consistency path without trusting either service."""

    if old_size < 0 or new_size < old_size or len(old_root) != 32 or len(new_root) != 32:
        return False
    if old_size == 0:
        return not proof and hmac.compare_digest(old_root, hashlib.sha256(b'').digest())
    if old_size == new_size:
        return not proof and hmac.compare_digest(old_root, new_root)
    first = old_size - 1
    second = new_size - 1
    while first & 1:
        first >>= 1
        second >>= 1
    position = 0
    if first == 0:
        old_hash = old_root
        new_hash = old_root
    else:
        if not proof or len(proof[0]) != 32:
            return False
        old_hash = proof[0]
        new_hash = proof[0]
        position = 1
    for sibling in proof[position:]:
        if len(sibling) != 32 or second == 0:
            return False
        if first & 1 or first == second:
            old_hash = _registry_node_hash(sibling, old_hash)
            new_hash = _registry_node_hash(sibling, new_hash)
            while first != 0 and not (first & 1):
                first >>= 1
                second >>= 1
        else:
            new_hash = _registry_node_hash(new_hash, sibling)
        first >>= 1
        second >>= 1
    return second == 0 and hmac.compare_digest(old_hash, old_root) and hmac.compare_digest(new_hash, new_root)


def _registry_node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b'\x01' + left + right).digest()


def _stream_map(policy: CheckpointGossipMonitorPolicy) -> dict[str, GossipStreamPolicy]:
    return {stream.stream_id: stream for stream in policy.streams}


def _clock_utc(clock: Callable[[], datetime]) -> datetime:
    try:
        value = clock()
    except Exception as error:
        raise CheckpointGossipError('gossip monitor clock read failed') from error
    return aware_utc(value, 'gossip monitor clock')


def _canonical_base64(value: str, *, expected_bytes: int, label: str) -> bytes:
    decoded = _decode_base64(value, label)
    if len(decoded) != expected_bytes:
        raise ValueError(f'{label} must encode exactly {expected_bytes} bytes')
    if base64.b64encode(decoded).decode('ascii') != value:
        raise ValueError(f'{label} must use canonical base64')
    return decoded


def _decode_base64(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f'{label} must use valid base64') from error


def _exact_bytes(payload: bytes, label: str, *, maximum: int) -> bytes:
    if not isinstance(payload, bytes) or not payload or len(payload) > maximum:
        raise CheckpointGossipError(f'{label} must be nonempty exact bytes within its bound')
    return payload


def _load_canonical_model[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        parsed = model.model_validate_json(payload)
    except (TypeError, ValueError) as error:
        raise CheckpointGossipError(f'invalid {label}: {error}') from error
    if payload != canonical_json_bytes(parsed):
        raise CheckpointGossipError(f'{label} must use canonical JSON encoding')
    return parsed


def _validate_root(root: Path) -> Path:
    requested = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    try:
        metadata = requested.lstat()
    except OSError as error:
        raise CheckpointGossipError(f'cannot inspect gossip monitor root: {error}') from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise CheckpointGossipError('gossip monitor root must be a real directory')
    if hasattr(os, 'getuid') and metadata.st_uid != os.getuid():
        raise CheckpointGossipError('gossip monitor root must be owned by the service user')
    if metadata.st_mode & 0o077:
        raise CheckpointGossipError('gossip monitor root must be owner-only')
    return requested


def _read_regular_nofollow(path: Path, *, maximum: int, private: bool) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CheckpointGossipError(f'cannot open gossip state file {path.name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CheckpointGossipError(f'gossip state file {path.name} must be regular')
        if private and metadata.st_mode & 0o077:
            raise CheckpointGossipError(f'gossip state file {path.name} must be owner-only')
        if metadata.st_size < 1 or metadata.st_size > maximum:
            raise CheckpointGossipError(f'gossip state file {path.name} has an invalid size')
        payload = os.read(descriptor, maximum + 1)
        if len(payload) != metadata.st_size:
            raise CheckpointGossipError(f'gossip state file {path.name} changed while read')
        return payload
    finally:
        os.close(descriptor)


def _validate_database_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CheckpointGossipError(f'cannot inspect gossip database: {error}') from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise CheckpointGossipError('gossip database must be a regular file')
    if metadata.st_mode & 0o077:
        raise CheckpointGossipError('gossip database must be owner-only')


def _write_durable_file(path: Path, payload: bytes, *, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
        mode,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count < 1:
                raise OSError('short write while creating gossip state')
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    'CheckpointGossipError',
    'CheckpointGossipMonitorPolicy',
    'CheckpointGossipMonitorStore',
    'GossipAgreementReport',
    'GossipComparisonPolicy',
    'GossipMonitorPolicyPin',
    'GossipMonitorReport',
    'GossipObservation',
    'GossipStreamHead',
    'GossipVerificationReport',
    'RegistryGossipStreamPolicy',
    'SignedGossipMonitorReport',
    'WitnessGossipStreamPolicy',
    'verify_gossip_agreement',
    'verify_gossip_bootstrap_head',
]
