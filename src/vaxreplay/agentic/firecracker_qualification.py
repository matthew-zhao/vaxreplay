"""Fail-closed, create-once Firecracker host qualification evidence.

Host preflight is useful evidence, but it is not runtime qualification.  This module keeps that
distinction in the schema: ``qualified`` can be true only when a Linux/KVM preflight and every
required live-runtime drill are present and successful.  The command-line host inspection shipped
with this module retains an authenticated negative or preflight-only result. A positive record can
be produced and loaded only through the separately authenticated raw collector and independent
verifier; legacy caller-authored full-suite inputs remain disabled.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import platform
import shutil
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.agentic.firecracker import (
    FirecrackerHostPreflightReceipt,
    FirecrackerPreflightError,
    FirecrackerWorkerSpec,
    firecracker_model_sha256,
    preflight_firecracker_host,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

FIRECRACKER_HOST_OBSERVATION_SCHEMA_VERSION = 'vaxreplay.firecracker-host-observation.v0.1'
FIRECRACKER_QUALIFICATION_DRILL_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-drill.v0.1'
FIRECRACKER_FULL_SUITE_SCHEMA_VERSION = 'vaxreplay.firecracker-full-suite-evidence.v0.1'
FIRECRACKER_QUALIFICATION_RECORD_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-record.v0.1'
FIRECRACKER_COLLECTOR_VERIFICATION_BINDING_SCHEMA_VERSION = 'vaxreplay.firecracker-collector-verification-binding.v0.2'
AUTHENTICATED_FIRECRACKER_QUALIFICATION_SCHEMA_VERSION = 'vaxreplay.authenticated-firecracker-qualification.v0.1'

QUALIFICATION_FILE = 'qualification.json'
WORKER_SPEC_FILE = 'worker-spec.json'
ARTIFACT_SHA256_FILE = 'QUALIFICATION.sha256'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_QUALIFICATION_ID_PATTERN = r'^[0-9a-f]{32}$'
_HMAC_DOMAIN = b'vaxreplay.firecracker-qualification.v0.1\x00'
_KEY_ID_DOMAIN = b'vaxreplay.firecracker-qualification-key-id.v0.1\x00'
_MAX_KEY_FILE_BYTES = 4096
_MAX_ARTIFACT_FILE_BYTES = 64 * 1024 * 1024
_REQUIRED_CONTROLLERS = frozenset({'cpu', 'memory', 'pids'})
_ARTIFACT_FILES = frozenset({QUALIFICATION_FILE, WORKER_SPEC_FILE, ARTIFACT_SHA256_FILE})
COLLECTOR_EVIDENCE_FILE = 'collector-evidence.json'
_POSITIVE_ARTIFACT_FILES = frozenset(
    {QUALIFICATION_FILE, WORKER_SPEC_FILE, ARTIFACT_SHA256_FILE, COLLECTOR_EVIDENCE_FILE}
)


class FirecrackerQualificationError(ValueError):
    """Qualification input or retained evidence failed closed."""


class FirecrackerQualificationStatus(str, enum.Enum):
    UNSUPPORTED_HOST_OS = 'unsupported_host_os'
    UNSUPPORTED_ARCHITECTURE = 'unsupported_architecture'
    INSUFFICIENT_PRIVILEGES = 'insufficient_privileges'
    KVM_UNAVAILABLE = 'kvm_unavailable'
    CGROUP_V2_UNAVAILABLE = 'cgroup_v2_unavailable'
    PINNED_ARTIFACT_REJECTED = 'pinned_artifact_rejected'
    HOST_PREFLIGHT_PASSED_ONLY = 'host_preflight_passed_only'
    FULL_RUNTIME_FAILED = 'full_runtime_failed'
    FULL_RUNTIME_QUALIFIED = 'full_runtime_qualified'


class FirecrackerQualificationDrillId(str, enum.Enum):
    LIVE_BOOT = 'live_boot'
    VSOCK_ROUND_TRIP = 'vsock_round_trip'
    GUEST_ISOLATION = 'guest_isolation'
    CGROUP_ENFORCEMENT = 'cgroup_enforcement'
    WALL_TIMEOUT = 'wall_timeout'
    TEARDOWN = 'teardown'
    LOAD_CANARY = 'load_canary'


class FirecrackerQualificationClaim(str, enum.Enum):
    FIRECRACKER_PROCESS_STARTED = 'firecracker_process_started'
    GUEST_READY_AUTHENTICATED = 'guest_ready_authenticated'
    CLEAN_GUEST_EXIT = 'clean_guest_exit'
    HOST_VSOCK_HANDSHAKE = 'host_vsock_handshake'
    GUEST_RPC_ROUND_TRIP = 'guest_rpc_round_trip'
    PEER_CID_BOUND = 'peer_cid_bound'
    ROOTFS_WRITE_DENIED = 'rootfs_write_denied'
    HARNESS_WRITE_DENIED = 'harness_write_denied'
    SCRATCH_WRITE_SUCCEEDED = 'scratch_write_succeeded'
    SCRATCH_FRESH = 'scratch_fresh'
    NETWORK_UNREACHABLE = 'network_unreachable'
    MMDS_UNREACHABLE = 'mmds_unreachable'
    CPU_LIMIT_OBSERVED = 'cpu_limit_observed'
    MEMORY_LIMIT_OBSERVED = 'memory_limit_observed'
    SWAP_DISABLED_OBSERVED = 'swap_disabled_observed'
    PIDS_LIMIT_OBSERVED = 'pids_limit_observed'
    WALL_WATCHDOG_TRIGGERED = 'wall_watchdog_triggered'
    PROCESS_GROUP_KILLED = 'process_group_killed'
    DEADLINE_BOUND = 'deadline_bound'
    CGROUP_ABSENT = 'cgroup_absent'
    JAIL_ABSENT = 'jail_absent'
    VSOCK_ABSENT = 'vsock_absent'
    PARALLEL_WORKERS_DISTINCT = 'parallel_workers_distinct'
    ALL_WORKERS_COMPLETED = 'all_workers_completed'
    ALL_WORKERS_TORN_DOWN = 'all_workers_torn_down'


_REQUIRED_CLAIMS: dict[FirecrackerQualificationDrillId, frozenset[FirecrackerQualificationClaim]] = {
    FirecrackerQualificationDrillId.LIVE_BOOT: frozenset(
        {
            FirecrackerQualificationClaim.FIRECRACKER_PROCESS_STARTED,
            FirecrackerQualificationClaim.GUEST_READY_AUTHENTICATED,
            FirecrackerQualificationClaim.CLEAN_GUEST_EXIT,
        }
    ),
    FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP: frozenset(
        {
            FirecrackerQualificationClaim.HOST_VSOCK_HANDSHAKE,
            FirecrackerQualificationClaim.GUEST_RPC_ROUND_TRIP,
            FirecrackerQualificationClaim.PEER_CID_BOUND,
        }
    ),
    FirecrackerQualificationDrillId.GUEST_ISOLATION: frozenset(
        {
            FirecrackerQualificationClaim.ROOTFS_WRITE_DENIED,
            FirecrackerQualificationClaim.HARNESS_WRITE_DENIED,
            FirecrackerQualificationClaim.SCRATCH_WRITE_SUCCEEDED,
            FirecrackerQualificationClaim.SCRATCH_FRESH,
            FirecrackerQualificationClaim.NETWORK_UNREACHABLE,
            FirecrackerQualificationClaim.MMDS_UNREACHABLE,
        }
    ),
    FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT: frozenset(
        {
            FirecrackerQualificationClaim.CPU_LIMIT_OBSERVED,
            FirecrackerQualificationClaim.MEMORY_LIMIT_OBSERVED,
            FirecrackerQualificationClaim.SWAP_DISABLED_OBSERVED,
            FirecrackerQualificationClaim.PIDS_LIMIT_OBSERVED,
        }
    ),
    FirecrackerQualificationDrillId.WALL_TIMEOUT: frozenset(
        {
            FirecrackerQualificationClaim.WALL_WATCHDOG_TRIGGERED,
            FirecrackerQualificationClaim.PROCESS_GROUP_KILLED,
            FirecrackerQualificationClaim.DEADLINE_BOUND,
        }
    ),
    FirecrackerQualificationDrillId.TEARDOWN: frozenset(
        {
            FirecrackerQualificationClaim.CGROUP_ABSENT,
            FirecrackerQualificationClaim.JAIL_ABSENT,
            FirecrackerQualificationClaim.VSOCK_ABSENT,
        }
    ),
    FirecrackerQualificationDrillId.LOAD_CANARY: frozenset(
        {
            FirecrackerQualificationClaim.PARALLEL_WORKERS_DISTINCT,
            FirecrackerQualificationClaim.ALL_WORKERS_COMPLETED,
            FirecrackerQualificationClaim.ALL_WORKERS_TORN_DOWN,
        }
    ),
}


class FirecrackerHostObservation(StrictModel):
    """Small, non-secret host observation made before Firecracker preflight."""

    schema_version: Literal['vaxreplay.firecracker-host-observation.v0.1'] = FIRECRACKER_HOST_OBSERVATION_SCHEMA_VERSION
    collected_at: datetime
    host_os: str = Field(min_length=1, max_length=100)
    host_architecture: str = Field(min_length=1, max_length=100)
    host_kernel_release: str = Field(min_length=1, max_length=500)
    effective_uid: int = Field(ge=0)
    kvm_path_present: bool
    kvm_non_symlink_character_device: bool
    kvm_read_write_access: bool
    cgroup_v2_controller_file_present: bool
    cgroup_controllers: tuple[str, ...]

    @field_validator('collected_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('host observation time must include a UTC offset')
        return value.astimezone(UTC)

    @field_validator('cgroup_controllers')
    @classmethod
    def validate_controllers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError('observed cgroup controllers must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_kvm_observation(self) -> Self:
        if self.kvm_non_symlink_character_device and not self.kvm_path_present:
            raise ValueError('a KVM character device cannot be verified when the path is absent')
        if self.kvm_read_write_access and not self.kvm_non_symlink_character_device:
            raise ValueError('KVM read/write access cannot be claimed for an unsafe device path')
        if self.cgroup_controllers and not self.cgroup_v2_controller_file_present:
            raise ValueError('cgroup controllers cannot be reported without the unified controller file')
        return self


class FirecrackerQualificationDrillEvidence(StrictModel):
    """Hash-bound evidence from one required live-runtime qualification drill.

    The referenced evidence is retained separately by the trusted qualification launcher.  This
    record intentionally cannot be produced by host preflight alone.
    """

    schema_version: Literal['vaxreplay.firecracker-qualification-drill.v0.1'] = (
        FIRECRACKER_QUALIFICATION_DRILL_SCHEMA_VERSION
    )
    drill_id: FirecrackerQualificationDrillId
    passed: bool
    started_at: datetime
    finished_at: datetime
    run_ids: tuple[str, ...] = Field(min_length=1, max_length=1024)
    evidence_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    authenticated_worker_attestation_sha256: str = Field(pattern=_SHA256_PATTERN)
    observer_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    observation_count: int = Field(ge=1, le=1_000_000)
    verified_claims: tuple[FirecrackerQualificationClaim, ...] = Field(min_length=1, max_length=32)
    failed_claims: tuple[FirecrackerQualificationClaim, ...] = Field(default=(), max_length=32)

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @field_validator('run_ids')
    @classmethod
    def validate_run_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError('qualification drill run IDs must be unique')
        for run_id in value:
            if len(run_id) != 32 or any(character not in '0123456789abcdef' for character in run_id):
                raise ValueError('qualification drill run IDs must be 32 lowercase hexadecimal characters')
        return value

    @field_validator('verified_claims', 'failed_claims')
    @classmethod
    def validate_claim_order(
        cls, value: tuple[FirecrackerQualificationClaim, ...], info
    ) -> tuple[FirecrackerQualificationClaim, ...]:
        if value != tuple(sorted(set(value), key=lambda item: item.value)):
            raise ValueError(f'{info.field_name} must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError('qualification drill cannot finish before it starts')
        if self.observation_count < len(self.run_ids):
            raise ValueError('qualification drill must contain at least one observation per run')
        required = _REQUIRED_CLAIMS[self.drill_id]
        verified = frozenset(self.verified_claims)
        failed = frozenset(self.failed_claims)
        if verified & failed or verified | failed != required:
            raise ValueError('qualification drill must dispose every required live claim exactly once')
        if self.passed != (verified == required and not failed):
            raise ValueError('qualification drill pass state must exactly match its verified live claims')
        return self


class FirecrackerFullSuiteEvidence(StrictModel):
    """Complete live suite required for a positive runtime qualification."""

    schema_version: Literal['vaxreplay.firecracker-full-suite-evidence.v0.1'] = FIRECRACKER_FULL_SUITE_SCHEMA_VERSION
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    host_preflight_sha256: str = Field(pattern=_SHA256_PATTERN)
    collected_on_linux_kvm: bool
    live_boot: FirecrackerQualificationDrillEvidence
    vsock_round_trip: FirecrackerQualificationDrillEvidence
    guest_isolation: FirecrackerQualificationDrillEvidence
    cgroup_enforcement: FirecrackerQualificationDrillEvidence
    wall_timeout: FirecrackerQualificationDrillEvidence
    teardown: FirecrackerQualificationDrillEvidence
    load_canary: FirecrackerQualificationDrillEvidence

    @model_validator(mode='after')
    def validate_drills(self) -> Self:
        expected = (
            (self.live_boot, FirecrackerQualificationDrillId.LIVE_BOOT),
            (self.vsock_round_trip, FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP),
            (self.guest_isolation, FirecrackerQualificationDrillId.GUEST_ISOLATION),
            (self.cgroup_enforcement, FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT),
            (self.wall_timeout, FirecrackerQualificationDrillId.WALL_TIMEOUT),
            (self.teardown, FirecrackerQualificationDrillId.TEARDOWN),
            (self.load_canary, FirecrackerQualificationDrillId.LOAD_CANARY),
        )
        if any(drill.drill_id != drill_id for drill, drill_id in expected):
            raise ValueError('full-suite evidence fields must contain their corresponding drill IDs')
        if len(self.load_canary.run_ids) < 2 or self.load_canary.observation_count < 2:
            raise ValueError('load-canary evidence must cover at least two distinct worker runs')
        return self

    @property
    def all_required_drills_passed(self) -> bool:
        return self.collected_on_linux_kvm and all(
            drill.passed
            for drill in (
                self.live_boot,
                self.vsock_round_trip,
                self.guest_isolation,
                self.cgroup_enforcement,
                self.wall_timeout,
                self.teardown,
                self.load_canary,
            )
        )


class FirecrackerCollectorVerificationBinding(StrictModel):
    """Cryptographic and release pins produced only after independent raw-evidence verification."""

    schema_version: Literal['vaxreplay.firecracker-collector-verification-binding.v0.2'] = (
        FIRECRACKER_COLLECTOR_VERIFICATION_BINDING_SCHEMA_VERSION
    )
    collector_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_key_id: str = Field(pattern=_SHA256_PATTERN)
    collector_public_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    driver_runtime_closure_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    driver_runtime_closure_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    driver_runtime_closure_sha256: str = Field(pattern=_SHA256_PATTERN)
    verifier_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_full_suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    production_linux_kvm_raw_evidence_verified: Literal[True] = True
    development_or_simulated_evidence_accepted: Literal[False] = False
    caller_authored_claims_accepted: Literal[False] = False


class FirecrackerQualificationRecord(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-record.v0.1'] = (
        FIRECRACKER_QUALIFICATION_RECORD_SCHEMA_VERSION
    )
    qualification_id: str = Field(pattern=_QUALIFICATION_ID_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_bytes: int = Field(gt=0, le=_MAX_ARTIFACT_FILE_BYTES)
    qualifier_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    host_observation: FirecrackerHostObservation
    status: FirecrackerQualificationStatus
    preflight: FirecrackerHostPreflightReceipt | None
    preflight_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    full_suite_evidence: FirecrackerFullSuiteEvidence | None
    collector_verification: FirecrackerCollectorVerificationBinding | None = None
    qualified: bool
    failure_summary: str | None = Field(default=None, min_length=1, max_length=300)
    recorded_at: datetime
    preflight_alone_is_full_runtime_qualification: Literal[False] = False
    provider_or_model_execution_qualified: Literal[False] = False
    official_leaderboard_execution_qualified: Literal[False] = False

    @field_validator('recorded_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('qualification record time must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_claim(self) -> Self:
        if (self.preflight is None) != (self.preflight_sha256 is None):
            raise ValueError('host preflight and its digest must either both be present or both be absent')
        if self.preflight is not None:
            if firecracker_model_sha256(self.preflight) != self.preflight_sha256:
                raise ValueError('host preflight digest does not match the embedded observation')
            if self.preflight.worker_spec_sha256 != self.worker_spec_sha256:
                raise ValueError('host preflight is bound to a different worker specification')
        if self.full_suite_evidence is not None:
            if self.preflight_sha256 is None:
                raise ValueError('full runtime evidence requires successful host preflight')
            if (
                self.full_suite_evidence.worker_spec_sha256 != self.worker_spec_sha256
                or self.full_suite_evidence.host_preflight_sha256 != self.preflight_sha256
            ):
                raise ValueError('full runtime evidence is not bound to this worker specification and preflight')
        if self.collector_verification is not None:
            if self.full_suite_evidence is None:
                raise ValueError('collector verification requires full runtime evidence')
            if self.collector_verification.verified_full_suite_sha256 != firecracker_model_sha256(
                self.full_suite_evidence
            ):
                raise ValueError('collector verification differs from the embedded full runtime evidence')

        fully_supported = (
            self.preflight is not None
            and self.full_suite_evidence is not None
            and self.full_suite_evidence.all_required_drills_passed
        )
        if self.qualified != fully_supported:
            raise ValueError('qualified must exactly reflect successful preflight and every live-runtime drill')
        if self.qualified:
            if self.status != FirecrackerQualificationStatus.FULL_RUNTIME_QUALIFIED or self.failure_summary is not None:
                raise ValueError('a qualified record must have qualified status and no failure summary')
            observation = self.host_observation
            normalized_architecture = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(
                observation.host_architecture.lower(), observation.host_architecture.lower()
            )
            if (
                observation.host_os != 'Linux'
                or normalized_architecture not in {'x86_64', 'aarch64'}
                or observation.effective_uid != 0
                or not observation.kvm_non_symlink_character_device
                or not observation.kvm_read_write_access
                or not observation.cgroup_v2_controller_file_present
                or not _REQUIRED_CONTROLLERS.issubset(observation.cgroup_controllers)
            ):
                raise ValueError('qualified runtime evidence requires a compatible observed Linux/KVM host')
        elif self.status == FirecrackerQualificationStatus.FULL_RUNTIME_QUALIFIED:
            raise ValueError('full-runtime-qualified status cannot be used for an unqualified record')
        if self.status == FirecrackerQualificationStatus.HOST_PREFLIGHT_PASSED_ONLY:
            if self.preflight is None or self.full_suite_evidence is not None or self.failure_summary is not None:
                raise ValueError('preflight-only status must contain only a successful host preflight')
        elif not self.qualified and self.failure_summary is None:
            raise ValueError('an unqualified non-preflight-only record requires a bounded failure summary')
        return self


class AuthenticatedFirecrackerQualification(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-firecracker-qualification.v0.1'] = (
        AUTHENTICATED_FIRECRACKER_QUALIFICATION_SCHEMA_VERSION
    )
    record: FirecrackerQualificationRecord
    authentication: Literal['hmac-sha256-domain-separated'] = 'hmac-sha256-domain-separated'
    qualification_key_id: str = Field(pattern=_SHA256_PATTERN)
    qualification_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


class LoadedFirecrackerQualification(StrictModel):
    root: str
    authenticated: AuthenticatedFirecrackerQualification
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)


def firecracker_qualification_key_id(key: bytes) -> str:
    _require_key(key)
    return _sha256(_KEY_ID_DOMAIN + key)


def required_firecracker_qualification_claims(
    drill_id: FirecrackerQualificationDrillId,
) -> tuple[FirecrackerQualificationClaim, ...]:
    return tuple(sorted(_REQUIRED_CLAIMS[drill_id], key=lambda item: item.value))


def firecracker_qualification_hmac(record: FirecrackerQualificationRecord, key: bytes) -> str:
    _require_key(key)
    return hmac.new(key, _HMAC_DOMAIN + canonical_json_bytes(record), hashlib.sha256).hexdigest()


def inspect_and_retain_firecracker_host(
    *,
    worker_spec_path: Path,
    expected_worker_spec_sha256: str,
    output_root: Path,
    qualification_key: bytes,
    expected_qualification_key_id: str,
    qualification_id: str | None = None,
    full_suite_evidence: FirecrackerFullSuiteEvidence | None = None,
) -> LoadedFirecrackerQualification:
    """Inspect this host and immutably retain authenticated success or failure evidence.

    ``FirecrackerFullSuiteEvidence`` is currently a contract schema, not authenticated proof that
    its claims came from live drills.  Until a collector-authentication boundary exists, accepting
    such an object here would let a caller turn self-asserted JSON into ``qualified=true``.  The
    parameter remains only to fail closed for callers of the earlier development API.
    """

    if full_suite_evidence is not None:
        raise FirecrackerQualificationError(
            'unauthenticated full-suite evidence is disabled; an authenticated live collector is required'
        )

    key_id = firecracker_qualification_key_id(qualification_key)
    if not hmac.compare_digest(key_id, expected_qualification_key_id):
        raise FirecrackerQualificationError('qualification key does not match the externally pinned key ID')
    spec, spec_bytes = load_pinned_firecracker_worker_spec(
        worker_spec_path,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
    )
    observation = observe_firecracker_host()
    preflight: FirecrackerHostPreflightReceipt | None = None
    status: FirecrackerQualificationStatus
    failure_summary: str | None
    try:
        preflight = preflight_firecracker_host(spec)
    except FirecrackerPreflightError:
        status, failure_summary = _classify_preflight_failure(observation, spec)
    else:
        status = FirecrackerQualificationStatus.HOST_PREFLIGHT_PASSED_ONLY
        failure_summary = None

    preflight_sha256 = None if preflight is None else firecracker_model_sha256(preflight)
    record = FirecrackerQualificationRecord(
        qualification_id=qualification_id or os.urandom(16).hex(),
        worker_spec_sha256=expected_worker_spec_sha256,
        worker_spec_bytes=len(spec_bytes),
        qualifier_source_sha256=_qualifier_source_sha256(),
        host_observation=observation,
        status=status,
        preflight=preflight,
        preflight_sha256=preflight_sha256,
        full_suite_evidence=None,
        qualified=False,
        failure_summary=failure_summary,
        recorded_at=datetime.now(UTC),
    )
    authenticated = AuthenticatedFirecrackerQualification(
        record=record,
        qualification_key_id=key_id,
        qualification_hmac_sha256=firecracker_qualification_hmac(record, qualification_key),
    )
    return _publish_qualification(
        output_root=output_root,
        spec_bytes=spec_bytes,
        authenticated=authenticated,
        qualification_key=qualification_key,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
        expected_qualification_key_id=expected_qualification_key_id,
    )


def verify_and_retain_firecracker_live_qualification(
    *,
    collector_evidence_root: Path,
    expected_collector_evidence_sha256: str,
    worker_spec_path: Path,
    expected_worker_spec_sha256: str,
    expected_probe_manifest_sha256: str,
    expected_driver_runtime_closure_manifest_sha256: str,
    expected_driver_runtime_closure_receipt_sha256: str,
    expected_driver_runtime_closure_sha256: str,
    expected_host_preflight_sha256: str,
    expected_collector_public_key_hex: str,
    expected_collector_key_id: str,
    expected_verifier_source_sha256: str,
    output_root: Path,
    qualification_key: bytes,
    expected_qualification_key_id: str,
    qualification_id: str | None = None,
) -> LoadedFirecrackerQualification:
    """Independently verify signed raw drills, then publish the only accepted positive record."""

    from vaxreplay.agentic.firecracker_qualification_collector import (
        independently_verify_firecracker_qualification_collector_evidence,
        load_firecracker_qualification_collector_evidence,
    )

    key_id = firecracker_qualification_key_id(qualification_key)
    if not hmac.compare_digest(key_id, expected_qualification_key_id):
        raise FirecrackerQualificationError('qualification key does not match the externally pinned key ID')
    spec, spec_bytes = load_pinned_firecracker_worker_spec(
        worker_spec_path,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
    )
    try:
        loaded_collector = load_firecracker_qualification_collector_evidence(
            collector_evidence_root,
            expected_evidence_sha256=expected_collector_evidence_sha256,
            expected_worker_spec_sha256=expected_worker_spec_sha256,
            expected_probe_manifest_sha256=expected_probe_manifest_sha256,
            expected_collector_public_key_hex=expected_collector_public_key_hex,
            expected_collector_key_id=expected_collector_key_id,
            expected_driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
            expected_driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
            expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
        )
        verified = independently_verify_firecracker_qualification_collector_evidence(
            collector_evidence_root,
            expected_evidence_sha256=expected_collector_evidence_sha256,
            expected_worker_spec_sha256=expected_worker_spec_sha256,
            expected_probe_manifest_sha256=expected_probe_manifest_sha256,
            expected_driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
            expected_driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
            expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
            expected_host_preflight_sha256=expected_host_preflight_sha256,
            expected_collector_public_key_hex=expected_collector_public_key_hex,
            expected_collector_key_id=expected_collector_key_id,
            expected_verifier_source_sha256=expected_verifier_source_sha256,
        )
    except ValueError as error:
        raise FirecrackerQualificationError(str(error)) from error
    collection = loaded_collector.authenticated.collection
    if verified.authenticated != loaded_collector.authenticated:
        raise FirecrackerQualificationError('independent verifier returned a different collector artifact')
    public_key_bytes = _decode_pinned_ed25519_public_key(expected_collector_public_key_hex)
    binding = FirecrackerCollectorVerificationBinding(
        collector_evidence_sha256=expected_collector_evidence_sha256,
        collector_key_id=expected_collector_key_id,
        collector_public_key_sha256=_sha256(public_key_bytes),
        probe_manifest_sha256=expected_probe_manifest_sha256,
        driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
        driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
        driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
        verifier_source_sha256=expected_verifier_source_sha256,
        verified_full_suite_sha256=firecracker_model_sha256(verified.full_suite_evidence),
    )
    record = FirecrackerQualificationRecord(
        qualification_id=qualification_id or os.urandom(16).hex(),
        worker_spec_sha256=expected_worker_spec_sha256,
        worker_spec_bytes=len(spec_bytes),
        qualifier_source_sha256=_qualifier_source_sha256(),
        host_observation=collection.host_observation,
        status=FirecrackerQualificationStatus.FULL_RUNTIME_QUALIFIED,
        preflight=collection.host_preflight,
        preflight_sha256=collection.host_preflight_sha256,
        full_suite_evidence=verified.full_suite_evidence,
        collector_verification=binding,
        qualified=True,
        failure_summary=None,
        recorded_at=datetime.now(UTC),
    )
    authenticated = AuthenticatedFirecrackerQualification(
        record=record,
        qualification_key_id=key_id,
        qualification_hmac_sha256=firecracker_qualification_hmac(record, qualification_key),
    )
    collector_evidence_bytes = canonical_json_bytes(loaded_collector.authenticated)
    return _publish_qualification(
        output_root=output_root,
        spec_bytes=spec_bytes,
        authenticated=authenticated,
        qualification_key=qualification_key,
        expected_worker_spec_sha256=firecracker_model_sha256(spec),
        expected_qualification_key_id=expected_qualification_key_id,
        collector_evidence_bytes=collector_evidence_bytes,
        expected_collector_evidence_sha256=expected_collector_evidence_sha256,
        expected_probe_manifest_sha256=expected_probe_manifest_sha256,
        expected_driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
        expected_driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
        expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
        expected_collector_public_key_hex=expected_collector_public_key_hex,
        expected_collector_key_id=expected_collector_key_id,
        expected_verifier_source_sha256=expected_verifier_source_sha256,
    )


def load_firecracker_qualification(
    root: Path,
    *,
    qualification_key: bytes,
    expected_qualification_key_id: str,
    expected_worker_spec_sha256: str,
    expected_artifact_sha256: str,
    expected_collector_evidence_sha256: str | None = None,
    expected_probe_manifest_sha256: str | None = None,
    expected_driver_runtime_closure_manifest_sha256: str | None = None,
    expected_driver_runtime_closure_receipt_sha256: str | None = None,
    expected_driver_runtime_closure_sha256: str | None = None,
    expected_collector_public_key_hex: str | None = None,
    expected_collector_key_id: str | None = None,
    expected_verifier_source_sha256: str | None = None,
) -> LoadedFirecrackerQualification:
    """Reload and independently verify an exact create-once qualification artifact."""

    resolved = root.expanduser()
    if resolved.is_symlink():
        raise FirecrackerQualificationError('qualification artifact root cannot be a symbolic link')
    resolved = resolved.resolve(strict=True)
    root_stat = resolved.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise FirecrackerQualificationError('qualification artifact root must be a private directory')
    observed = {entry.name for entry in os.scandir(resolved)}
    if frozenset(observed) not in {_ARTIFACT_FILES, _POSITIVE_ARTIFACT_FILES}:
        raise FirecrackerQualificationError('qualification artifact has an unexpected file inventory')

    qualification_bytes = _read_private_file(resolved / QUALIFICATION_FILE, _MAX_ARTIFACT_FILE_BYTES)
    spec_bytes = _read_private_file(resolved / WORKER_SPEC_FILE, _MAX_ARTIFACT_FILE_BYTES)
    digest_bytes = _read_private_file(resolved / ARTIFACT_SHA256_FILE, 65)
    try:
        authenticated = AuthenticatedFirecrackerQualification.model_validate_json(qualification_bytes)
    except ValueError as error:
        raise FirecrackerQualificationError('qualification record is invalid') from error
    if qualification_bytes != canonical_json_bytes(authenticated):
        raise FirecrackerQualificationError('qualification record is not canonical JSON')
    artifact_sha256 = _sha256(qualification_bytes)
    if not hmac.compare_digest(artifact_sha256, expected_artifact_sha256) or not hmac.compare_digest(
        digest_bytes, (artifact_sha256 + '\n').encode('ascii')
    ):
        raise FirecrackerQualificationError('qualification artifact digest does not match its external pin')

    key_id = firecracker_qualification_key_id(qualification_key)
    if (
        not hmac.compare_digest(key_id, expected_qualification_key_id)
        or not hmac.compare_digest(authenticated.qualification_key_id, expected_qualification_key_id)
        or not hmac.compare_digest(
            authenticated.qualification_hmac_sha256,
            firecracker_qualification_hmac(authenticated.record, qualification_key),
        )
    ):
        raise FirecrackerQualificationError('qualification record authentication failed')

    spec, canonical_spec_bytes = _parse_worker_spec(spec_bytes)
    if canonical_spec_bytes != spec_bytes:
        raise FirecrackerQualificationError('retained worker specification is not canonical JSON')
    observed_spec_sha256 = firecracker_model_sha256(spec)
    if (
        not hmac.compare_digest(observed_spec_sha256, expected_worker_spec_sha256)
        or authenticated.record.worker_spec_sha256 != expected_worker_spec_sha256
        or authenticated.record.worker_spec_bytes != len(spec_bytes)
    ):
        raise FirecrackerQualificationError('qualification record does not bind the externally pinned worker spec')
    if authenticated.record.qualified:
        _verify_positive_collector_binding(
            root=resolved,
            authenticated=authenticated,
            spec=spec,
            observed_files=observed,
            expected_collector_evidence_sha256=expected_collector_evidence_sha256,
            expected_probe_manifest_sha256=expected_probe_manifest_sha256,
            expected_driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
            expected_driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
            expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
            expected_collector_public_key_hex=expected_collector_public_key_hex,
            expected_collector_key_id=expected_collector_key_id,
            expected_verifier_source_sha256=expected_verifier_source_sha256,
        )
    elif observed != _ARTIFACT_FILES or authenticated.record.collector_verification is not None:
        raise FirecrackerQualificationError('unqualified artifact cannot contain collector-verification evidence')
    return LoadedFirecrackerQualification(
        root=str(resolved),
        authenticated=authenticated,
        artifact_sha256=artifact_sha256,
    )


def _verify_positive_collector_binding(
    *,
    root: Path,
    authenticated: AuthenticatedFirecrackerQualification,
    spec: FirecrackerWorkerSpec,
    observed_files: set[str],
    expected_collector_evidence_sha256: str | None,
    expected_probe_manifest_sha256: str | None,
    expected_driver_runtime_closure_manifest_sha256: str | None,
    expected_driver_runtime_closure_receipt_sha256: str | None,
    expected_driver_runtime_closure_sha256: str | None,
    expected_collector_public_key_hex: str | None,
    expected_collector_key_id: str | None,
    expected_verifier_source_sha256: str | None,
) -> None:
    from vaxreplay.agentic.firecracker_qualification_probe import (
        AuthenticatedFirecrackerQualificationRawCollection,
        FirecrackerQualificationProbeError,
        verify_authenticated_firecracker_qualification_collection,
    )

    record = authenticated.record
    binding = record.collector_verification
    required_external_pins = (
        expected_collector_evidence_sha256,
        expected_probe_manifest_sha256,
        expected_driver_runtime_closure_manifest_sha256,
        expected_driver_runtime_closure_receipt_sha256,
        expected_driver_runtime_closure_sha256,
        expected_collector_public_key_hex,
        expected_collector_key_id,
        expected_verifier_source_sha256,
    )
    if binding is None or any(value is None for value in required_external_pins):
        raise FirecrackerQualificationError(
            'positive qualification requires externally pinned authenticated live collector evidence'
        )
    if observed_files != _POSITIVE_ARTIFACT_FILES:
        raise FirecrackerQualificationError('legacy positive qualification lacks raw collector evidence')
    assert expected_collector_evidence_sha256 is not None
    assert expected_probe_manifest_sha256 is not None
    assert expected_driver_runtime_closure_manifest_sha256 is not None
    assert expected_driver_runtime_closure_receipt_sha256 is not None
    assert expected_driver_runtime_closure_sha256 is not None
    assert expected_collector_public_key_hex is not None
    assert expected_collector_key_id is not None
    assert expected_verifier_source_sha256 is not None
    public_key_bytes = _decode_pinned_ed25519_public_key(expected_collector_public_key_hex)
    if (
        binding.collector_evidence_sha256 != expected_collector_evidence_sha256
        or binding.probe_manifest_sha256 != expected_probe_manifest_sha256
        or binding.driver_runtime_closure_manifest_sha256 != expected_driver_runtime_closure_manifest_sha256
        or binding.driver_runtime_closure_receipt_sha256 != expected_driver_runtime_closure_receipt_sha256
        or binding.driver_runtime_closure_sha256 != expected_driver_runtime_closure_sha256
        or binding.collector_key_id != expected_collector_key_id
        or binding.verifier_source_sha256 != expected_verifier_source_sha256
        or binding.collector_public_key_sha256 != _sha256(public_key_bytes)
        or record.full_suite_evidence is None
        or record.preflight_sha256 is None
    ):
        raise FirecrackerQualificationError('positive qualification differs from an external collector pin')
    evidence_bytes = _read_private_file(root / COLLECTOR_EVIDENCE_FILE, _MAX_ARTIFACT_FILE_BYTES)
    if not hmac.compare_digest(_sha256(evidence_bytes), expected_collector_evidence_sha256):
        raise FirecrackerQualificationError('embedded collector evidence differs from its external digest pin')
    try:
        collector = AuthenticatedFirecrackerQualificationRawCollection.model_validate_json(evidence_bytes)
    except ValueError as error:
        raise FirecrackerQualificationError('embedded collector evidence is invalid') from error
    if canonical_json_bytes(collector) != evidence_bytes:
        raise FirecrackerQualificationError('embedded collector evidence is not canonical JSON')
    try:
        verified = verify_authenticated_firecracker_qualification_collection(
            collector,
            worker_spec=spec,
            expected_collector_public_key_hex=expected_collector_public_key_hex,
            expected_collector_key_id=expected_collector_key_id,
            expected_worker_spec_sha256=record.worker_spec_sha256,
            expected_probe_manifest_sha256=expected_probe_manifest_sha256,
            expected_driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
            expected_driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
            expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
            expected_host_preflight_sha256=record.preflight_sha256,
            verifier_source_sha256=expected_verifier_source_sha256,
        )
    except FirecrackerQualificationProbeError as error:
        raise FirecrackerQualificationError(str(error)) from error
    if (
        verified.full_suite_evidence != record.full_suite_evidence
        or binding.verified_full_suite_sha256 != firecracker_model_sha256(verified.full_suite_evidence)
    ):
        raise FirecrackerQualificationError('independent collector result differs from the qualification record')


def _decode_pinned_ed25519_public_key(value: str) -> bytes:
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise FirecrackerQualificationError('collector public key must be lowercase hexadecimal') from error
    if len(decoded) != 32 or value != decoded.hex():
        raise FirecrackerQualificationError('collector public key must contain exactly 32 lowercase hexadecimal bytes')
    return decoded


def load_pinned_firecracker_worker_spec(
    path: Path,
    *,
    expected_worker_spec_sha256: str,
) -> tuple[FirecrackerWorkerSpec, bytes]:
    if len(expected_worker_spec_sha256) != 64 or any(
        character not in '0123456789abcdef' for character in expected_worker_spec_sha256
    ):
        raise FirecrackerQualificationError('expected worker-spec pin must be a lowercase SHA-256 digest')
    content = _read_stable_regular_file(path, maximum_bytes=_MAX_ARTIFACT_FILE_BYTES, private=False)
    spec, canonical = _parse_worker_spec(content)
    if canonical != content:
        raise FirecrackerQualificationError('externally pinned worker specification must be canonical JSON')
    if not hmac.compare_digest(firecracker_model_sha256(spec), expected_worker_spec_sha256):
        raise FirecrackerQualificationError('worker specification does not match its external SHA-256 pin')
    return spec, content


def load_firecracker_full_suite_evidence(path: Path) -> FirecrackerFullSuiteEvidence:
    """Load the canonical contract shape; this does not authenticate how its claims were observed."""

    content = _read_stable_regular_file(path, maximum_bytes=_MAX_ARTIFACT_FILE_BYTES, private=False)
    try:
        evidence = FirecrackerFullSuiteEvidence.model_validate_json(content)
    except ValueError as error:
        raise FirecrackerQualificationError('full-suite evidence input is invalid') from error
    if canonical_json_bytes(evidence) != content:
        raise FirecrackerQualificationError('full-suite evidence input must be canonical JSON')
    return evidence


def observe_firecracker_host() -> FirecrackerHostObservation:
    kvm = Path('/dev/kvm')
    kvm_present = False
    kvm_character = False
    try:
        metadata = kvm.lstat()
    except OSError:
        pass
    else:
        kvm_present = True
        kvm_character = not stat.S_ISLNK(metadata.st_mode) and stat.S_ISCHR(metadata.st_mode)

    controllers_path = Path('/sys/fs/cgroup/cgroup.controllers')
    controllers_present = False
    controllers: tuple[str, ...] = ()
    try:
        metadata = controllers_path.lstat()
        if not stat.S_ISLNK(metadata.st_mode) and stat.S_ISREG(metadata.st_mode):
            raw = controllers_path.read_bytes()
            if len(raw) <= 4096:
                decoded = raw.decode('ascii')
                controllers = tuple(sorted(set(decoded.split())))
                controllers_present = True
    except (OSError, UnicodeDecodeError):
        pass
    return FirecrackerHostObservation(
        collected_at=datetime.now(UTC),
        host_os=platform.system() or 'unknown',
        host_architecture=platform.machine() or 'unknown',
        host_kernel_release=platform.release() or 'unknown',
        effective_uid=os.geteuid(),
        kvm_path_present=kvm_present,
        kvm_non_symlink_character_device=kvm_character,
        kvm_read_write_access=kvm_character and os.access(kvm, os.R_OK | os.W_OK),
        cgroup_v2_controller_file_present=controllers_present,
        cgroup_controllers=controllers,
    )


def read_firecracker_qualification_key_file(path: Path) -> bytes:
    content = _read_stable_regular_file(path, maximum_bytes=_MAX_KEY_FILE_BYTES, private=True)
    return decode_firecracker_qualification_key(content)


def read_firecracker_qualification_key_fd(file_descriptor: int) -> bytes:
    if file_descriptor < 0:
        raise FirecrackerQualificationError('qualification key file descriptor must be nonnegative')
    try:
        duplicate = os.dup(file_descriptor)
    except OSError as error:
        raise FirecrackerQualificationError('cannot duplicate qualification key file descriptor') from error
    try:
        chunks = bytearray()
        while len(chunks) <= _MAX_KEY_FILE_BYTES:
            chunk = os.read(duplicate, min(4096, _MAX_KEY_FILE_BYTES + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > _MAX_KEY_FILE_BYTES:
            raise FirecrackerQualificationError('qualification key input exceeds its byte limit')
        return decode_firecracker_qualification_key(bytes(chunks))
    except OSError as error:
        raise FirecrackerQualificationError('cannot read qualification key file descriptor') from error
    finally:
        os.close(duplicate)


def _publish_qualification(
    *,
    output_root: Path,
    spec_bytes: bytes,
    authenticated: AuthenticatedFirecrackerQualification,
    qualification_key: bytes,
    expected_worker_spec_sha256: str,
    expected_qualification_key_id: str,
    collector_evidence_bytes: bytes | None = None,
    expected_collector_evidence_sha256: str | None = None,
    expected_probe_manifest_sha256: str | None = None,
    expected_driver_runtime_closure_manifest_sha256: str | None = None,
    expected_driver_runtime_closure_receipt_sha256: str | None = None,
    expected_driver_runtime_closure_sha256: str | None = None,
    expected_collector_public_key_hex: str | None = None,
    expected_collector_key_id: str | None = None,
    expected_verifier_source_sha256: str | None = None,
) -> LoadedFirecrackerQualification:
    target = output_root.expanduser()
    if target.is_symlink():
        raise FirecrackerQualificationError('qualification output cannot be a symbolic link')
    target = target.absolute()
    if target.exists():
        raise FirecrackerQualificationError('qualification output already exists and cannot be replaced')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    staging.chmod(0o700)
    qualification_bytes = canonical_json_bytes(authenticated)
    artifact_sha256 = _sha256(qualification_bytes)
    try:
        files = [
            (QUALIFICATION_FILE, qualification_bytes),
            (WORKER_SPEC_FILE, spec_bytes),
            (ARTIFACT_SHA256_FILE, (artifact_sha256 + '\n').encode('ascii')),
        ]
        if collector_evidence_bytes is not None:
            files.append((COLLECTOR_EVIDENCE_FILE, collector_evidence_bytes))
        for name, content in files:
            _write_private_file(staging / name, content)
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except FileExistsError as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise FirecrackerQualificationError('qualification output already exists and cannot be replaced') from error
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_firecracker_qualification(
        target,
        qualification_key=qualification_key,
        expected_qualification_key_id=expected_qualification_key_id,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
        expected_artifact_sha256=artifact_sha256,
        expected_collector_evidence_sha256=expected_collector_evidence_sha256,
        expected_probe_manifest_sha256=expected_probe_manifest_sha256,
        expected_driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
        expected_driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
        expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
        expected_collector_public_key_hex=expected_collector_public_key_hex,
        expected_collector_key_id=expected_collector_key_id,
        expected_verifier_source_sha256=expected_verifier_source_sha256,
    )


def _classify_preflight_failure(
    observation: FirecrackerHostObservation,
    spec: FirecrackerWorkerSpec,
) -> tuple[FirecrackerQualificationStatus, str]:
    if observation.host_os != 'Linux':
        return FirecrackerQualificationStatus.UNSUPPORTED_HOST_OS, 'Firecracker requires a Linux host with KVM'
    normalized_architecture = observation.host_architecture.lower()
    architecture = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(normalized_architecture, normalized_architecture)
    if architecture not in {'x86_64', 'aarch64'} or architecture != spec.runtime.architecture:
        return (
            FirecrackerQualificationStatus.UNSUPPORTED_ARCHITECTURE,
            'host architecture does not match the externally pinned Firecracker runtime',
        )
    if observation.effective_uid != 0:
        return FirecrackerQualificationStatus.INSUFFICIENT_PRIVILEGES, 'Firecracker host inspection requires UID 0'
    if not observation.kvm_non_symlink_character_device or not observation.kvm_read_write_access:
        return FirecrackerQualificationStatus.KVM_UNAVAILABLE, 'a readable and writable /dev/kvm device is unavailable'
    if not observation.cgroup_v2_controller_file_present or not _REQUIRED_CONTROLLERS.issubset(
        observation.cgroup_controllers
    ):
        return (
            FirecrackerQualificationStatus.CGROUP_V2_UNAVAILABLE,
            'cgroup v2 does not expose the required cpu, memory, and pids controllers',
        )
    return (
        FirecrackerQualificationStatus.PINNED_ARTIFACT_REJECTED,
        'one or more externally pinned runtime artifacts or trusted paths failed preflight',
    )


def _parse_worker_spec(content: bytes) -> tuple[FirecrackerWorkerSpec, bytes]:
    try:
        spec = FirecrackerWorkerSpec.model_validate_json(content)
    except ValueError as error:
        raise FirecrackerQualificationError('worker specification is invalid') from error
    return spec, canonical_json_bytes(spec)


def decode_firecracker_qualification_key(content: bytes) -> bytes:
    """Decode the one-value ASCII-hex format shared by qualification secret inputs."""

    try:
        encoded = content.decode('ascii')
    except UnicodeDecodeError as error:
        raise FirecrackerQualificationError('qualification key must be ASCII hexadecimal') from error
    if encoded != encoded.strip() + ('\n' if encoded.endswith('\n') else ''):
        raise FirecrackerQualificationError('qualification key file must contain one trimmed hexadecimal value')
    encoded = encoded.strip()
    if not encoded or len(encoded) % 2 or any(character not in '0123456789abcdefABCDEF' for character in encoded):
        raise FirecrackerQualificationError('qualification key is not valid hexadecimal')
    try:
        key = bytes.fromhex(encoded)
    except ValueError as error:
        raise FirecrackerQualificationError('qualification key is not valid hexadecimal') from error
    _require_key(key)
    return key


def _require_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32 or len(key) > 512:
        raise FirecrackerQualificationError('qualification HMAC key must contain 32 to 512 bytes')


def _read_stable_regular_file(path: Path, *, maximum_bytes: int, private: bool) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FirecrackerQualificationError('cannot open pinned qualification input') from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum_bytes:
            raise FirecrackerQualificationError('pinned qualification input is unsafe or oversized')
        if private and (stat.S_IMODE(before.st_mode) != 0o600 or before.st_uid not in {0, os.geteuid()}):
            raise FirecrackerQualificationError('qualification key file must be owner-only and owned by root or caller')
        content = bytearray()
        while len(content) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > maximum_bytes:
            raise FirecrackerQualificationError('pinned qualification input exceeds its byte limit')
        after = os.fstat(descriptor)
        stable = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise FirecrackerQualificationError('pinned qualification input changed while being read')
        return bytes(content)
    except OSError as error:
        raise FirecrackerQualificationError('cannot read pinned qualification input') from error
    finally:
        os.close(descriptor)


def _read_private_file(path: Path, maximum_bytes: int) -> bytes:
    return _read_stable_regular_file(path, maximum_bytes=maximum_bytes, private=True)


def _write_private_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short write while retaining qualification evidence')
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _qualifier_source_sha256() -> str:
    return _sha256(_read_stable_regular_file(Path(__file__), maximum_bytes=4 * 1024 * 1024, private=False))


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
