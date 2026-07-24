"""Fail-closed planning and authenticated raw-evidence collection for Firecracker qualification.

The historical plan artifact remains non-qualifying.  The live path accepts observations only from
an injected, pinned probe boundary, binds every drill to a fresh challenge, and publishes an
Ed25519-authenticated create-once raw bundle.  It never sets ``qualified`` itself; the independent
verifier in :mod:`firecracker_qualification_probe` is the sole positive decision point.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import platform
import selectors
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal, Self, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.agentic.firecracker import (
    FirecrackerHostPreflightReceipt,
    FirecrackerWorkerSpec,
    firecracker_model_sha256,
    preflight_firecracker_host,
)
from vaxreplay.agentic.firecracker_qualification import (
    FirecrackerHostObservation,
    FirecrackerQualificationClaim,
    FirecrackerQualificationDrillId,
    FirecrackerQualificationError,
    load_pinned_firecracker_worker_spec,
    observe_firecracker_host,
    required_firecracker_qualification_claims,
)
from vaxreplay.agentic.firecracker_qualification_driver import FirecrackerQualificationDriverRequest
from vaxreplay.agentic.firecracker_qualification_probe import (
    AuthenticatedFirecrackerQualificationRawCollection,
    FirecrackerQualificationBoundaryIdentity,
    FirecrackerQualificationBoundaryKind,
    FirecrackerQualificationChallenge,
    FirecrackerQualificationCollectionMode,
    FirecrackerQualificationProbeBoundary,
    FirecrackerQualificationProbeError,
    FirecrackerQualificationProbeManifest,
    FirecrackerQualificationRawCollection,
    FirecrackerQualificationRawDrillObservation,
    VerifiedFirecrackerQualificationCollection,
    authenticate_firecracker_qualification_collection,
    derive_firecracker_full_suite_evidence,
    ed25519_public_key_bytes,
    firecracker_live_collector_key_id,
    firecracker_qualification_probe_manifest_sha256,
    validate_probe_manifest_for_worker,
    verify_authenticated_firecracker_qualification_collection,
    verify_firecracker_qualification_collection_authentication,
)
from vaxreplay.agentic.firecracker_qualification_runtime_closure import (
    LoadedQualificationDriverRuntimeClosure,
    QualificationDriverRuntimeClosureError,
    verify_qualification_driver_runtime_closure,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

FIRECRACKER_COLLECTOR_DRILL_PLAN_SCHEMA_VERSION = 'vaxreplay.firecracker-collector-drill-plan.v0.1'
FIRECRACKER_COLLECTOR_PLAN_SCHEMA_VERSION = 'vaxreplay.firecracker-collector-plan.v0.1'
FIRECRACKER_DRIVER_FAILURE_DIAGNOSTIC_SCHEMA_VERSION = 'vaxreplay.firecracker-driver-failure-diagnostic.v0.1'

COLLECTOR_PLAN_FILE = 'collector-plan.json'
WORKER_SPEC_FILE = 'worker-spec.json'
COLLECTOR_PLAN_SHA256_FILE = 'COLLECTOR-PLAN.sha256'
COLLECTOR_EVIDENCE_FILE = 'collector-evidence.json'
PROBE_MANIFEST_FILE = 'probe-manifest.json'
COLLECTOR_EVIDENCE_SHA256_FILE = 'COLLECTOR-EVIDENCE.sha256'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_PLAN_ID_PATTERN = r'^[0-9a-f]{32}$'
_MAX_FILE_BYTES = 64 * 1024 * 1024
_ARTIFACT_FILES = frozenset({COLLECTOR_PLAN_FILE, WORKER_SPEC_FILE, COLLECTOR_PLAN_SHA256_FILE})
_EVIDENCE_ARTIFACT_FILES = frozenset(
    {COLLECTOR_EVIDENCE_FILE, WORKER_SPEC_FILE, PROBE_MANIFEST_FILE, COLLECTOR_EVIDENCE_SHA256_FILE}
)
_REQUIRED_CONTROLLERS = frozenset({'cpu', 'memory', 'pids'})
_MAX_DRIVER_OUTPUT_BYTES = 16 * 1024 * 1024
_MAX_DRIVER_STDERR_BYTES = 1024 * 1024
_DRIVER_STREAM_CHUNK_BYTES = 64 * 1024
_DRIVER_TERMINATION_GRACE_SECONDS = 5
_MAX_COLLECTOR_KEY_BYTES = 256


class FirecrackerQualificationCollectorError(ValueError):
    """A collector plan was unsafe, changed, or did not match its external pins."""


class FirecrackerDriverFailureKind(str, enum.Enum):
    PROCESS_EXIT = 'process_exit'
    STDOUT_LIMIT_EXCEEDED = 'stdout_limit_exceeded'
    STDERR_LIMIT_EXCEEDED = 'stderr_limit_exceeded'
    EXECUTION_TIMEOUT = 'execution_timeout'
    EXECUTION_NOT_STARTED = 'execution_not_started'
    EXECUTION_ERROR = 'execution_error'
    INVALID_RAW_EVIDENCE = 'invalid_raw_evidence'
    NON_CANONICAL_RAW_EVIDENCE = 'non_canonical_raw_evidence'


class FirecrackerDriverFailureDiagnostic(StrictModel):
    """Content-free, bounded metadata suitable for an operator log after a failed drill."""

    schema_version: Literal['vaxreplay.firecracker-driver-failure-diagnostic.v0.1'] = (
        FIRECRACKER_DRIVER_FAILURE_DIAGNOSTIC_SCHEMA_VERSION
    )
    drill_id: FirecrackerQualificationDrillId
    failure_kind: FirecrackerDriverFailureKind
    exit_status: int | Literal['not_started', 'timed_out', 'unknown']
    stdout_byte_count: int = Field(ge=0, le=2**63 - 1)
    stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    stderr_byte_count: int = Field(ge=0, le=2**63 - 1)
    stderr_sha256: str = Field(pattern=_SHA256_PATTERN)
    stderr_content_retained: Literal[False] = False

    @field_validator('exit_status')
    @classmethod
    def validate_exit_status(cls, value: int | str) -> int | str:
        if isinstance(value, int) and not -255 <= value <= 255:
            raise ValueError('driver exit status is out of range')
        return value


@dataclass(frozen=True)
class _PinnedDriverProcessResult:
    exit_status: int | Literal['not_started', 'timed_out', 'unknown']
    stdout: bytes
    stdout_byte_count: int
    stdout_sha256: str
    stderr_byte_count: int
    stderr_sha256: str
    failure_kind: FirecrackerDriverFailureKind | None


class _BoundedDriverStreamCapture:
    """Drain and hash one child stream while retaining at most a fixed byte prefix."""

    def __init__(self, *, byte_limit: int, retained_byte_limit: int) -> None:
        if byte_limit < 1 or retained_byte_limit < 0 or retained_byte_limit > byte_limit:
            raise ValueError('driver stream capture limits are invalid')
        self.byte_limit = byte_limit
        self.retained_byte_limit = retained_byte_limit
        self.byte_count = 0
        self.digest = hashlib.sha256()
        self.retained = bytearray()
        self.limit_exceeded = False

    def consume(self, chunk: bytes) -> bool:
        self.byte_count += len(chunk)
        self.digest.update(chunk)
        remaining = self.retained_byte_limit - len(self.retained)
        if remaining > 0:
            self.retained.extend(chunk[:remaining])
        first_excess = self.byte_count > self.byte_limit and not self.limit_exceeded
        self.limit_exceeded = self.limit_exceeded or first_excess
        return first_excess


class FirecrackerQualificationCollectorStatus(str, enum.Enum):
    BLOCKED_MISSING_MEASURED_PROBES = 'blocked_missing_measured_probes'


class FirecrackerCollectorHostPrimitive(str, enum.Enum):
    """Checked-in host mechanisms that can be reused, but are not live observations."""

    PINNED_WORKER_PREFLIGHT = 'pinned_worker_preflight'
    EXACT_JAIL_PREPARATION = 'exact_jail_preparation'
    FOREGROUND_PROCESS_GROUP = 'foreground_process_group'
    WALL_WATCHDOG = 'wall_watchdog'
    EXACT_CGROUP_JAIL_VSOCK_TEARDOWN = 'exact_cgroup_jail_vsock_teardown'
    AUTHENTICATED_WORKER_ATTESTATION_SCHEMA = 'authenticated_worker_attestation_schema'


class FirecrackerCollectorMissingCapability(str, enum.Enum):
    """Capabilities that must exist before any positive collector is implemented."""

    PINNED_SEPARATE_QUALIFICATION_GUEST = 'pinned_separate_qualification_guest'
    GUEST_READY_CHALLENGE = 'guest_ready_challenge'
    VSOCK_NONCE_ROUND_TRIP = 'vsock_nonce_round_trip'
    VSOCK_PEER_PROCESS_TREE_BINDING = 'vsock_peer_process_tree_binding'
    GUEST_MOUNT_AND_SCRATCH_PROBES = 'guest_mount_and_scratch_probes'
    GUEST_NETWORK_AND_MMDS_PROBES = 'guest_network_and_mmds_probes'
    LIVE_CGROUP_STRESS_AND_COUNTER_PROBES = 'live_cgroup_stress_and_counter_probes'
    INTENTIONAL_GUEST_HANG_PROBE = 'intentional_guest_hang_probe'
    SEVEN_DRILL_ORCHESTRATOR = 'seven_drill_orchestrator'
    TWO_WORKER_PARALLEL_CANARY = 'two_worker_parallel_canary'
    CREATE_ONCE_RAW_OBSERVATION_BUNDLE = 'create_once_raw_observation_bundle'
    AUTHENTICATED_COLLECTOR_OUTPUT = 'authenticated_collector_output'


_ALL_HOST_PRIMITIVES = tuple(FirecrackerCollectorHostPrimitive)
_ALL_MISSING_CAPABILITIES = tuple(FirecrackerCollectorMissingCapability)

_COMMON_GAPS = (
    FirecrackerCollectorMissingCapability.SEVEN_DRILL_ORCHESTRATOR,
    FirecrackerCollectorMissingCapability.CREATE_ONCE_RAW_OBSERVATION_BUNDLE,
    FirecrackerCollectorMissingCapability.AUTHENTICATED_COLLECTOR_OUTPUT,
)

_DRILL_GAPS: dict[FirecrackerQualificationDrillId, tuple[FirecrackerCollectorMissingCapability, ...]] = {
    FirecrackerQualificationDrillId.LIVE_BOOT: (
        FirecrackerCollectorMissingCapability.PINNED_SEPARATE_QUALIFICATION_GUEST,
        FirecrackerCollectorMissingCapability.GUEST_READY_CHALLENGE,
        FirecrackerCollectorMissingCapability.VSOCK_PEER_PROCESS_TREE_BINDING,
        *_COMMON_GAPS,
    ),
    FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP: (
        FirecrackerCollectorMissingCapability.PINNED_SEPARATE_QUALIFICATION_GUEST,
        FirecrackerCollectorMissingCapability.VSOCK_NONCE_ROUND_TRIP,
        FirecrackerCollectorMissingCapability.VSOCK_PEER_PROCESS_TREE_BINDING,
        *_COMMON_GAPS,
    ),
    FirecrackerQualificationDrillId.GUEST_ISOLATION: (
        FirecrackerCollectorMissingCapability.PINNED_SEPARATE_QUALIFICATION_GUEST,
        FirecrackerCollectorMissingCapability.GUEST_MOUNT_AND_SCRATCH_PROBES,
        FirecrackerCollectorMissingCapability.GUEST_NETWORK_AND_MMDS_PROBES,
        *_COMMON_GAPS,
    ),
    FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT: (
        FirecrackerCollectorMissingCapability.PINNED_SEPARATE_QUALIFICATION_GUEST,
        FirecrackerCollectorMissingCapability.LIVE_CGROUP_STRESS_AND_COUNTER_PROBES,
        *_COMMON_GAPS,
    ),
    FirecrackerQualificationDrillId.WALL_TIMEOUT: (
        FirecrackerCollectorMissingCapability.PINNED_SEPARATE_QUALIFICATION_GUEST,
        FirecrackerCollectorMissingCapability.INTENTIONAL_GUEST_HANG_PROBE,
        *_COMMON_GAPS,
    ),
    FirecrackerQualificationDrillId.TEARDOWN: _COMMON_GAPS,
    FirecrackerQualificationDrillId.LOAD_CANARY: (
        FirecrackerCollectorMissingCapability.PINNED_SEPARATE_QUALIFICATION_GUEST,
        FirecrackerCollectorMissingCapability.GUEST_READY_CHALLENGE,
        FirecrackerCollectorMissingCapability.VSOCK_NONCE_ROUND_TRIP,
        FirecrackerCollectorMissingCapability.VSOCK_PEER_PROCESS_TREE_BINDING,
        FirecrackerCollectorMissingCapability.TWO_WORKER_PARALLEL_CANARY,
        *_COMMON_GAPS,
    ),
}


class FirecrackerCollectorDrillPlan(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-collector-drill-plan.v0.1'] = (
        FIRECRACKER_COLLECTOR_DRILL_PLAN_SCHEMA_VERSION
    )
    drill_id: FirecrackerQualificationDrillId
    required_claims: tuple[FirecrackerQualificationClaim, ...] = Field(min_length=1, max_length=32)
    reusable_host_primitives: tuple[FirecrackerCollectorHostPrimitive, ...] = Field(min_length=1, max_length=16)
    missing_capabilities: tuple[FirecrackerCollectorMissingCapability, ...] = Field(min_length=1, max_length=32)
    ready_to_collect: Literal[False] = False
    caller_assertions_can_satisfy: Literal[False] = False

    @field_validator('required_claims', 'reusable_host_primitives', 'missing_capabilities')
    @classmethod
    def validate_unique_ordered_tuple(cls, value: tuple[enum.Enum, ...], info) -> tuple[enum.Enum, ...]:
        if value != tuple(sorted(set(value), key=lambda item: str(item.value))):
            raise ValueError(f'{info.field_name} must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_drill_contract(self) -> Self:
        if self.required_claims != required_firecracker_qualification_claims(self.drill_id):
            raise ValueError('collector drill plan must contain the exact qualification claims')
        expected_gaps = tuple(sorted(_DRILL_GAPS[self.drill_id], key=lambda item: item.value))
        if self.missing_capabilities != expected_gaps:
            raise ValueError('collector drill plan must retain every currently missing capability')
        return self


class FirecrackerQualificationCollectorPlan(StrictModel):
    """Non-evidence inventory; its literals make positive qualification unrepresentable."""

    schema_version: Literal['vaxreplay.firecracker-collector-plan.v0.1'] = FIRECRACKER_COLLECTOR_PLAN_SCHEMA_VERSION
    plan_id: str = Field(pattern=_PLAN_ID_PATTERN)
    worker_id: str = Field(min_length=1, max_length=200)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_runtime_architecture: Literal['x86_64', 'aarch64']
    collector_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    host_observation: FirecrackerHostObservation
    host_linux_kvm_cgroup_prerequisites_observed: bool
    status: Literal[FirecrackerQualificationCollectorStatus.BLOCKED_MISSING_MEASURED_PROBES] = (
        FirecrackerQualificationCollectorStatus.BLOCKED_MISSING_MEASURED_PROBES
    )
    reusable_host_primitives: tuple[FirecrackerCollectorHostPrimitive, ...] = Field(min_length=1, max_length=16)
    missing_capabilities: tuple[FirecrackerCollectorMissingCapability, ...] = Field(min_length=1, max_length=32)
    drills: tuple[FirecrackerCollectorDrillPlan, ...] = Field(min_length=7, max_length=7)
    recorded_at: datetime
    guest_probe_protocol_id: None = None
    task_guest_protocol_reused_for_qualification: Literal[False] = False
    live_vm_launched: Literal[False] = False
    host_preflight_executed: Literal[False] = False
    raw_observation_bundle_emitted: Literal[False] = False
    full_suite_evidence_emitted: Literal[False] = False
    remote_guest_or_process_attestation_claimed: Literal[False] = False
    caller_supplied_drill_evidence_accepted: Literal[False] = False
    collector_source_is_transitive_attestation: Literal[False] = False
    qualified: Literal[False] = False
    provider_or_model_execution_qualified: Literal[False] = False
    official_leaderboard_execution_qualified: Literal[False] = False

    @field_validator('recorded_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('collector plan time must include a UTC offset')
        return value.astimezone(UTC)

    @field_validator('reusable_host_primitives', 'missing_capabilities')
    @classmethod
    def validate_unique_ordered_tuple(cls, value: tuple[enum.Enum, ...], info) -> tuple[enum.Enum, ...]:
        if value != tuple(sorted(set(value), key=lambda item: str(item.value))):
            raise ValueError(f'{info.field_name} must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_complete_blocked_plan(self) -> Self:
        expected_drill_ids = tuple(FirecrackerQualificationDrillId)
        if tuple(drill.drill_id for drill in self.drills) != expected_drill_ids:
            raise ValueError('collector plan must contain every required drill exactly once in canonical order')
        expected_primitives = tuple(sorted(_ALL_HOST_PRIMITIVES, key=lambda item: item.value))
        expected_gaps = tuple(sorted(_ALL_MISSING_CAPABILITIES, key=lambda item: item.value))
        if self.reusable_host_primitives != expected_primitives or self.missing_capabilities != expected_gaps:
            raise ValueError('collector plan must enumerate the complete checked-in and missing capability sets')
        expected_host = _host_prerequisites_observed(
            self.host_observation,
            worker_runtime_architecture=self.worker_runtime_architecture,
        )
        if self.host_linux_kvm_cgroup_prerequisites_observed != expected_host:
            raise ValueError('collector host-prerequisite status must be derived from the retained observation')
        return self


class LoadedFirecrackerQualificationCollectorPlan(StrictModel):
    root: str
    plan: FirecrackerQualificationCollectorPlan
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)


class LoadedFirecrackerQualificationCollectorEvidence(StrictModel):
    root: str
    authenticated: AuthenticatedFirecrackerQualificationRawCollection
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec: FirecrackerWorkerSpec
    probe_manifest: FirecrackerQualificationProbeManifest


class PinnedLinuxKvmQualificationDriver:
    """Exact executable boundary for the host-specific Firecracker probe driver.

    The create-once runtime closure is offline-reverified before every drill. The exact interpreter
    and driver are then opened without symlink traversal, checked as root-owned immutable regular
    files, hashed through their descriptors, and invoked through ``/proc/self/fd`` in isolated
    Python mode. Its stdout is an untrusted raw-drill object which the collector validates and later
    authenticates. The driver owns the concrete Firecracker boot, cgroupfs reads, watchdog,
    teardown, and parallel canary.
    """

    def __init__(
        self,
        *,
        driver_id: str,
        executable_path: Path,
        expected_executable_sha256: str,
        runtime_closure: LoadedQualificationDriverRuntimeClosure,
        worker_spec: FirecrackerWorkerSpec,
        probe_manifest: FirecrackerQualificationProbeManifest,
    ) -> None:
        _require_sha256(expected_executable_sha256, label='expected qualification-driver pin')
        if not driver_id or len(driver_id) > 200:
            raise FirecrackerQualificationCollectorError('qualification driver ID is invalid')
        self._driver_id = driver_id
        self._path = executable_path.expanduser().absolute()
        self._sha256 = expected_executable_sha256
        try:
            verified_closure = verify_qualification_driver_runtime_closure(
                Path(runtime_closure.root),
                expected_manifest_sha256=runtime_closure.manifest_sha256,
                expected_receipt_sha256=runtime_closure.receipt_sha256,
                expected_closure_sha256=runtime_closure.closure_sha256,
                require_root_owned=True,
            )
        except QualificationDriverRuntimeClosureError as error:
            raise FirecrackerQualificationCollectorError(str(error)) from error
        if verified_closure != runtime_closure:
            raise FirecrackerQualificationCollectorError('qualification-driver runtime closure changed while loading')
        if (
            Path(runtime_closure.manifest.driver_entrypoint_path) != self._path
            or runtime_closure.manifest.driver_entrypoint_sha256 != expected_executable_sha256
        ):
            raise FirecrackerQualificationCollectorError(
                'qualification driver differs from its transitive runtime closure'
            )
        self._runtime_closure = runtime_closure
        self._spec = worker_spec
        self._manifest = probe_manifest

    @property
    def identity(self) -> FirecrackerQualificationBoundaryIdentity:
        return FirecrackerQualificationBoundaryIdentity(
            boundary_id=self._driver_id,
            kind=FirecrackerQualificationBoundaryKind.PINNED_LINUX_KVM_DRIVER,
            executable_sha256=self._sha256,
            external_executable_pin_enforced=True,
            direct_linux_kvm_launch=True,
            injected_test_boundary=False,
            runtime_closure_manifest_sha256=self._runtime_closure.manifest_sha256,
            runtime_closure_receipt_sha256=self._runtime_closure.receipt_sha256,
            runtime_closure_sha256=self._runtime_closure.closure_sha256,
            transitive_runtime_pin_enforced=True,
        )

    def live_boot(self, challenge: FirecrackerQualificationChallenge) -> FirecrackerQualificationRawDrillObservation:
        return self._run(challenge)

    def vsock_round_trip(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation:
        return self._run(challenge)

    def guest_isolation(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation:
        return self._run(challenge)

    def cgroup_enforcement(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation:
        return self._run(challenge)

    def wall_timeout(self, challenge: FirecrackerQualificationChallenge) -> FirecrackerQualificationRawDrillObservation:
        return self._run(challenge)

    def teardown(self, challenge: FirecrackerQualificationChallenge) -> FirecrackerQualificationRawDrillObservation:
        return self._run(challenge)

    def load_canary(self, challenge: FirecrackerQualificationChallenge) -> FirecrackerQualificationRawDrillObservation:
        return self._run(challenge)

    def _run(self, challenge: FirecrackerQualificationChallenge) -> FirecrackerQualificationRawDrillObservation:
        if platform.system() != 'Linux' or os.geteuid() != 0:
            raise FirecrackerQualificationCollectorError('live qualification driver requires root on Linux/KVM')
        try:
            runtime_closure = verify_qualification_driver_runtime_closure(
                Path(self._runtime_closure.root),
                expected_manifest_sha256=self._runtime_closure.manifest_sha256,
                expected_receipt_sha256=self._runtime_closure.receipt_sha256,
                expected_closure_sha256=self._runtime_closure.closure_sha256,
                require_root_owned=True,
            )
        except QualificationDriverRuntimeClosureError as error:
            raise FirecrackerQualificationCollectorError(str(error)) from error
        descriptor = _open_pinned_driver(self._path, expected_sha256=self._sha256)
        try:
            interpreter_descriptor = _open_pinned_driver(
                Path(runtime_closure.manifest.interpreter_path),
                expected_sha256=runtime_closure.manifest.interpreter_sha256,
            )
        except BaseException:
            os.close(descriptor)
            raise
        request = canonical_json_bytes(
            FirecrackerQualificationDriverRequest(
                challenge=challenge,
                worker_spec=self._spec,
                probe_manifest=self._manifest,
            )
        )
        try:
            result = _run_bounded_driver_process(
                argv=(
                    f'/proc/self/fd/{interpreter_descriptor}',
                    '-I',
                    '-B',
                    f'/proc/self/fd/{descriptor}',
                    'run-drill',
                    '--protocol',
                    'vaxreplay.firecracker-qualification-driver.v0.1',
                ),
                request=request,
                cwd=Path('/'),
                env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'},
                pass_fds=(descriptor, interpreter_descriptor),
                timeout_seconds=self._spec.limits.wall_seconds + 60,
            )
        finally:
            os.close(descriptor)
            os.close(interpreter_descriptor)
        diagnostic_arguments = {
            'challenge': challenge,
            'exit_status': result.exit_status,
            'stdout_byte_count': result.stdout_byte_count,
            'stdout_sha256': result.stdout_sha256,
            'stderr_byte_count': result.stderr_byte_count,
            'stderr_sha256': result.stderr_sha256,
        }
        if result.failure_kind is not None:
            diagnostic = _driver_failure_diagnostic(
                failure_kind=result.failure_kind,
                **diagnostic_arguments,
            )
            raise FirecrackerQualificationCollectorError(
                _driver_failure_message(_driver_failure_label(result.failure_kind), diagnostic)
            )
        try:
            drill = FirecrackerQualificationRawDrillObservation.model_validate_json(result.stdout)
        except ValueError as error:
            diagnostic = _driver_failure_diagnostic(
                failure_kind=FirecrackerDriverFailureKind.INVALID_RAW_EVIDENCE,
                **diagnostic_arguments,
            )
            raise FirecrackerQualificationCollectorError(
                _driver_failure_message('pinned qualification driver returned invalid raw evidence', diagnostic)
            ) from error
        if canonical_json_bytes(drill) != result.stdout:
            diagnostic = _driver_failure_diagnostic(
                failure_kind=FirecrackerDriverFailureKind.NON_CANONICAL_RAW_EVIDENCE,
                **diagnostic_arguments,
            )
            raise FirecrackerQualificationCollectorError(
                _driver_failure_message('pinned qualification driver output is not canonical JSON', diagnostic)
            )
        return drill


def build_firecracker_qualification_collector_plan(
    spec: FirecrackerWorkerSpec,
    *,
    plan_id: str | None = None,
    host_observation: FirecrackerHostObservation | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> FirecrackerQualificationCollectorPlan:
    """Build a complete blocked plan without preflight, VM launch, or guest interaction."""

    observation = host_observation or observe_firecracker_host()
    drills = tuple(
        FirecrackerCollectorDrillPlan(
            drill_id=drill_id,
            required_claims=required_firecracker_qualification_claims(drill_id),
            reusable_host_primitives=tuple(sorted(_ALL_HOST_PRIMITIVES, key=lambda item: item.value)),
            missing_capabilities=tuple(sorted(_DRILL_GAPS[drill_id], key=lambda item: item.value)),
        )
        for drill_id in FirecrackerQualificationDrillId
    )
    return FirecrackerQualificationCollectorPlan(
        plan_id=plan_id or os.urandom(16).hex(),
        worker_id=spec.worker_id,
        worker_spec_sha256=firecracker_model_sha256(spec),
        worker_runtime_architecture=spec.runtime.architecture,
        collector_source_sha256=_collector_source_sha256(),
        host_observation=observation,
        host_linux_kvm_cgroup_prerequisites_observed=_host_prerequisites_observed(
            observation,
            worker_runtime_architecture=spec.runtime.architecture,
        ),
        reusable_host_primitives=tuple(sorted(_ALL_HOST_PRIMITIVES, key=lambda item: item.value)),
        missing_capabilities=tuple(sorted(_ALL_MISSING_CAPABILITIES, key=lambda item: item.value)),
        drills=drills,
        recorded_at=clock(),
    )


def retain_firecracker_qualification_collector_plan(
    *,
    worker_spec_path: Path,
    expected_worker_spec_sha256: str,
    output_root: Path,
    plan_id: str | None = None,
) -> LoadedFirecrackerQualificationCollectorPlan:
    """Create and immutably retain a non-qualifying collector plan."""

    try:
        spec, spec_bytes = load_pinned_firecracker_worker_spec(
            worker_spec_path,
            expected_worker_spec_sha256=expected_worker_spec_sha256,
        )
    except FirecrackerQualificationError as error:
        raise FirecrackerQualificationCollectorError(str(error)) from error
    plan = build_firecracker_qualification_collector_plan(spec, plan_id=plan_id)
    return _publish_plan(
        output_root=output_root,
        plan=plan,
        spec_bytes=spec_bytes,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
    )


def collect_and_retain_firecracker_qualification_evidence(
    *,
    worker_spec_path: Path,
    expected_worker_spec_sha256: str,
    probe_manifest_path: Path,
    expected_probe_manifest_sha256: str,
    boundary: FirecrackerQualificationProbeBoundary,
    mode: FirecrackerQualificationCollectionMode,
    collector_private_key: Ed25519PrivateKey,
    expected_collector_key_id: str,
    expected_collector_source_sha256: str,
    output_root: Path,
    development_host_observation: FirecrackerHostObservation | None = None,
    development_host_preflight: FirecrackerHostPreflightReceipt | None = None,
    development_collection_id: str | None = None,
    development_random_bytes: Callable[[int], bytes] | None = None,
    development_clock: Callable[[], datetime] | None = None,
) -> LoadedFirecrackerQualificationCollectorEvidence:
    """Run all seven drills and retain their exact raw observations.

    Production never accepts injected host state, entropy, IDs, or clocks.  Development must inject
    its host observation and preflight explicitly and remains structurally non-qualifying.
    """

    try:
        spec, spec_bytes = load_pinned_firecracker_worker_spec(
            worker_spec_path,
            expected_worker_spec_sha256=expected_worker_spec_sha256,
        )
        manifest, manifest_bytes = load_pinned_firecracker_qualification_probe_manifest(
            probe_manifest_path,
            expected_probe_manifest_sha256=expected_probe_manifest_sha256,
        )
        validate_probe_manifest_for_worker(manifest, spec)
    except (FirecrackerQualificationError, FirecrackerQualificationProbeError) as error:
        raise FirecrackerQualificationCollectorError(str(error)) from error
    source_sha256 = _collector_source_sha256()
    _require_sha256(expected_collector_source_sha256, label='expected collector-source pin')
    if not hmac.compare_digest(source_sha256, expected_collector_source_sha256):
        raise FirecrackerQualificationCollectorError('collector source differs from its external release pin')
    collector_public_key = ed25519_public_key_bytes(collector_private_key)
    if not hmac.compare_digest(firecracker_live_collector_key_id(collector_public_key), expected_collector_key_id):
        raise FirecrackerQualificationCollectorError('collector signing key differs from its external key-ID pin')

    production = mode == FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM
    injected = (
        development_host_observation,
        development_host_preflight,
        development_collection_id,
        development_random_bytes,
        development_clock,
    )
    if production:
        if any(item is not None for item in injected):
            raise FirecrackerQualificationCollectorError('production collection forbids injected development state')
        if boundary.identity.kind != FirecrackerQualificationBoundaryKind.PINNED_LINUX_KVM_DRIVER:
            raise FirecrackerQualificationCollectorError('production collection requires the pinned Linux/KVM driver')
        observation = observe_firecracker_host()
        if not _host_prerequisites_observed(observation, worker_runtime_architecture=spec.runtime.architecture):
            raise FirecrackerQualificationCollectorError('production collection requires root Linux/KVM and cgroup v2')
        try:
            preflight = preflight_firecracker_host(spec)
        except Exception:
            raise FirecrackerQualificationCollectorError('production Firecracker host preflight failed') from None
        _validate_pinned_qualification_artifact(
            Path(manifest.qualification_rootfs_path),
            expected_sha256=manifest.qualification_rootfs_sha256,
        )
        _validate_pinned_qualification_artifact(
            Path(manifest.qualification_harness_path),
            expected_sha256=manifest.qualification_harness_sha256,
        )
        collection_id = os.urandom(16).hex()
        random_bytes = os.urandom
        clock: Callable[[], datetime] = _utc_now
    else:
        if boundary.identity.kind != FirecrackerQualificationBoundaryKind.DETERMINISTIC_DEVELOPMENT:
            raise FirecrackerQualificationCollectorError(
                'development collection requires an explicit simulated boundary'
            )
        if development_host_observation is None or development_host_preflight is None:
            raise FirecrackerQualificationCollectorError(
                'development collection requires injected host and preflight data'
            )
        observation = development_host_observation
        preflight = development_host_preflight
        collection_id = development_collection_id or '0' * 32
        random_bytes = development_random_bytes or _deterministic_development_random_bytes()
        clock = development_clock or (lambda: datetime(2000, 1, 1, tzinfo=UTC))

    if preflight.worker_spec_sha256 != expected_worker_spec_sha256:
        raise FirecrackerQualificationCollectorError('collector preflight differs from the pinned worker specification')
    manifest_sha256 = firecracker_qualification_probe_manifest_sha256(manifest)
    drills: list[FirecrackerQualificationRawDrillObservation] = []
    dispatch = {
        FirecrackerQualificationDrillId.LIVE_BOOT: boundary.live_boot,
        FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP: boundary.vsock_round_trip,
        FirecrackerQualificationDrillId.GUEST_ISOLATION: boundary.guest_isolation,
        FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT: boundary.cgroup_enforcement,
        FirecrackerQualificationDrillId.WALL_TIMEOUT: boundary.wall_timeout,
        FirecrackerQualificationDrillId.TEARDOWN: boundary.teardown,
        FirecrackerQualificationDrillId.LOAD_CANARY: boundary.load_canary,
    }
    for drill_id in FirecrackerQualificationDrillId:
        run_count = 2 if drill_id == FirecrackerQualificationDrillId.LOAD_CANARY else 1
        challenge = FirecrackerQualificationChallenge(
            collection_id=collection_id,
            challenge_id=_fresh_hex(random_bytes, 16, label='challenge ID'),
            nonce_hex=_fresh_hex(random_bytes, 32, label='challenge nonce'),
            drill_id=drill_id,
            run_ids=tuple(_fresh_hex(random_bytes, 16, label='worker run ID') for _ in range(run_count)),
            worker_spec_sha256=expected_worker_spec_sha256,
            probe_manifest_sha256=manifest_sha256,
            issued_at=clock(),
        )
        try:
            raw_drill = dispatch[drill_id](challenge)
        except FirecrackerQualificationCollectorError:
            raise
        except Exception:
            raise FirecrackerQualificationCollectorError('qualification probe boundary failed') from None
        if raw_drill.challenge != challenge or raw_drill.drill_id != drill_id:
            raise FirecrackerQualificationCollectorError('probe boundary returned evidence for a different challenge')
        _validate_raw_drill_release_bindings(raw_drill, spec)
        drills.append(raw_drill)

    collection = FirecrackerQualificationRawCollection(
        collection_id=collection_id,
        mode=mode,
        worker_spec_sha256=expected_worker_spec_sha256,
        probe_manifest=manifest,
        probe_manifest_sha256=manifest_sha256,
        boundary_identity=boundary.identity,
        driver_runtime_closure_sha256=boundary.identity.runtime_closure_sha256,
        collector_executable_sha256=source_sha256,
        host_observation=observation,
        host_preflight=preflight,
        host_preflight_sha256=firecracker_model_sha256(preflight),
        drills=tuple(drills),
        collected_at=clock(),
        development_simulated=not production,
        production_qualification_eligible=production,
    )
    # Exercise the independent derivation before signing.  Failed claim measurements remain valid
    # raw evidence and will later produce a non-qualifying suite; malformed cross-bindings do not.
    derive_firecracker_full_suite_evidence(collection, worker_spec=spec)
    authenticated = authenticate_firecracker_qualification_collection(
        collection,
        private_key=collector_private_key,
    )
    return _publish_collector_evidence(
        output_root=output_root,
        authenticated=authenticated,
        spec_bytes=spec_bytes,
        manifest_bytes=manifest_bytes,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
        expected_probe_manifest_sha256=expected_probe_manifest_sha256,
        expected_collector_public_key_hex=collector_public_key.hex(),
        expected_collector_key_id=expected_collector_key_id,
        expected_driver_runtime_closure_manifest_sha256=(
            boundary.identity.runtime_closure_manifest_sha256 if production else None
        ),
        expected_driver_runtime_closure_receipt_sha256=(
            boundary.identity.runtime_closure_receipt_sha256 if production else None
        ),
        expected_driver_runtime_closure_sha256=(boundary.identity.runtime_closure_sha256 if production else None),
    )


def load_pinned_firecracker_qualification_probe_manifest(
    path: Path,
    *,
    expected_probe_manifest_sha256: str,
) -> tuple[FirecrackerQualificationProbeManifest, bytes]:
    _require_sha256(expected_probe_manifest_sha256, label='expected probe-manifest pin')
    content = _read_safe_regular_file(path, _MAX_FILE_BYTES)
    try:
        manifest = FirecrackerQualificationProbeManifest.model_validate_json(content)
    except ValueError as error:
        raise FirecrackerQualificationCollectorError('qualification probe manifest is invalid') from error
    if content != canonical_json_bytes(manifest):
        raise FirecrackerQualificationCollectorError('qualification probe manifest must be canonical JSON')
    if not hmac.compare_digest(
        firecracker_qualification_probe_manifest_sha256(manifest), expected_probe_manifest_sha256
    ):
        raise FirecrackerQualificationCollectorError('qualification probe manifest differs from its external pin')
    return manifest, content


def load_firecracker_qualification_collector_evidence(
    root: Path,
    *,
    expected_evidence_sha256: str,
    expected_worker_spec_sha256: str,
    expected_probe_manifest_sha256: str,
    expected_collector_public_key_hex: str,
    expected_collector_key_id: str,
    expected_driver_runtime_closure_manifest_sha256: str | None = None,
    expected_driver_runtime_closure_receipt_sha256: str | None = None,
    expected_driver_runtime_closure_sha256: str | None = None,
) -> LoadedFirecrackerQualificationCollectorEvidence:
    """Authenticate an exact raw bundle without treating development evidence as production."""

    _require_sha256(expected_evidence_sha256, label='expected collector-evidence pin')
    _require_sha256(expected_worker_spec_sha256, label='expected worker-spec pin')
    _require_sha256(expected_probe_manifest_sha256, label='expected probe-manifest pin')
    resolved = _validate_private_artifact_root(root, expected_files=_EVIDENCE_ARTIFACT_FILES)
    evidence_bytes = _read_private_file(resolved / COLLECTOR_EVIDENCE_FILE, _MAX_FILE_BYTES)
    spec_bytes = _read_private_file(resolved / WORKER_SPEC_FILE, _MAX_FILE_BYTES)
    manifest_bytes = _read_private_file(resolved / PROBE_MANIFEST_FILE, _MAX_FILE_BYTES)
    digest_bytes = _read_private_file(resolved / COLLECTOR_EVIDENCE_SHA256_FILE, 65)
    try:
        authenticated = AuthenticatedFirecrackerQualificationRawCollection.model_validate_json(evidence_bytes)
        spec = FirecrackerWorkerSpec.model_validate_json(spec_bytes)
        manifest = FirecrackerQualificationProbeManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise FirecrackerQualificationCollectorError('collector evidence artifact contains invalid data') from error
    if (
        canonical_json_bytes(authenticated) != evidence_bytes
        or canonical_json_bytes(spec) != spec_bytes
        or canonical_json_bytes(manifest) != manifest_bytes
    ):
        raise FirecrackerQualificationCollectorError('collector evidence artifact contains non-canonical JSON')
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    if not hmac.compare_digest(evidence_sha256, expected_evidence_sha256) or not hmac.compare_digest(
        digest_bytes, (evidence_sha256 + '\n').encode('ascii')
    ):
        raise FirecrackerQualificationCollectorError('collector evidence digest differs from its external pin')
    if (
        firecracker_model_sha256(spec) != expected_worker_spec_sha256
        or firecracker_qualification_probe_manifest_sha256(manifest) != expected_probe_manifest_sha256
        or authenticated.collection.worker_spec_sha256 != expected_worker_spec_sha256
        or authenticated.collection.probe_manifest_sha256 != expected_probe_manifest_sha256
        or authenticated.collection.probe_manifest != manifest
    ):
        raise FirecrackerQualificationCollectorError('collector evidence differs from its externally pinned release')
    _verify_raw_collection_runtime_closure_pins(
        authenticated,
        expected_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
        expected_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
        expected_closure_sha256=expected_driver_runtime_closure_sha256,
    )
    try:
        verify_firecracker_qualification_collection_authentication(
            authenticated,
            expected_collector_public_key_hex=expected_collector_public_key_hex,
            expected_collector_key_id=expected_collector_key_id,
        )
    except FirecrackerQualificationProbeError as error:
        raise FirecrackerQualificationCollectorError(str(error)) from error
    return LoadedFirecrackerQualificationCollectorEvidence(
        root=str(resolved),
        authenticated=authenticated,
        evidence_sha256=evidence_sha256,
        worker_spec=spec,
        probe_manifest=manifest,
    )


def independently_verify_firecracker_qualification_collector_evidence(
    root: Path,
    *,
    expected_evidence_sha256: str,
    expected_worker_spec_sha256: str,
    expected_probe_manifest_sha256: str,
    expected_host_preflight_sha256: str,
    expected_collector_public_key_hex: str,
    expected_collector_key_id: str,
    expected_verifier_source_sha256: str,
    expected_driver_runtime_closure_manifest_sha256: str | None = None,
    expected_driver_runtime_closure_receipt_sha256: str | None = None,
    expected_driver_runtime_closure_sha256: str | None = None,
) -> VerifiedFirecrackerQualificationCollection:
    """The sole collector API that can return a production-eligible full suite."""

    loaded = load_firecracker_qualification_collector_evidence(
        root,
        expected_evidence_sha256=expected_evidence_sha256,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
        expected_probe_manifest_sha256=expected_probe_manifest_sha256,
        expected_collector_public_key_hex=expected_collector_public_key_hex,
        expected_collector_key_id=expected_collector_key_id,
        expected_driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
        expected_driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
        expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
    )
    if loaded.authenticated.collection.mode != FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM:
        raise FirecrackerQualificationCollectorError('development/simulated collector evidence can never qualify')
    assert expected_driver_runtime_closure_manifest_sha256 is not None
    assert expected_driver_runtime_closure_receipt_sha256 is not None
    assert expected_driver_runtime_closure_sha256 is not None
    try:
        return verify_authenticated_firecracker_qualification_collection(
            loaded.authenticated,
            worker_spec=loaded.worker_spec,
            expected_collector_public_key_hex=expected_collector_public_key_hex,
            expected_collector_key_id=expected_collector_key_id,
            expected_worker_spec_sha256=expected_worker_spec_sha256,
            expected_probe_manifest_sha256=expected_probe_manifest_sha256,
            expected_driver_runtime_closure_manifest_sha256=expected_driver_runtime_closure_manifest_sha256,
            expected_driver_runtime_closure_receipt_sha256=expected_driver_runtime_closure_receipt_sha256,
            expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
            expected_host_preflight_sha256=expected_host_preflight_sha256,
            verifier_source_sha256=expected_verifier_source_sha256,
        )
    except FirecrackerQualificationProbeError as error:
        raise FirecrackerQualificationCollectorError(str(error)) from error


def load_firecracker_qualification_collector_plan(
    root: Path,
    *,
    expected_plan_sha256: str,
    expected_worker_spec_sha256: str,
) -> LoadedFirecrackerQualificationCollectorPlan:
    """Verify the exact private inventory, canonical bytes, and external plan/spec pins."""

    _require_sha256(expected_plan_sha256, label='expected collector-plan pin')
    _require_sha256(expected_worker_spec_sha256, label='expected worker-spec pin')
    resolved = root.expanduser()
    if resolved.is_symlink():
        raise FirecrackerQualificationCollectorError('collector-plan artifact root cannot be a symbolic link')
    try:
        resolved = resolved.resolve(strict=True)
        root_stat = resolved.lstat()
    except OSError as error:
        raise FirecrackerQualificationCollectorError('collector-plan artifact root is unavailable') from error
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise FirecrackerQualificationCollectorError('collector-plan artifact root must be a private directory')
    try:
        observed = {entry.name for entry in os.scandir(resolved)}
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot inventory collector-plan artifact') from error
    if observed != _ARTIFACT_FILES:
        raise FirecrackerQualificationCollectorError('collector-plan artifact has an unexpected file inventory')

    plan_bytes = _read_private_file(resolved / COLLECTOR_PLAN_FILE, _MAX_FILE_BYTES)
    spec_bytes = _read_private_file(resolved / WORKER_SPEC_FILE, _MAX_FILE_BYTES)
    digest_bytes = _read_private_file(resolved / COLLECTOR_PLAN_SHA256_FILE, 65)
    try:
        plan = FirecrackerQualificationCollectorPlan.model_validate_json(plan_bytes)
    except ValueError as error:
        raise FirecrackerQualificationCollectorError('collector plan is invalid') from error
    if plan_bytes != canonical_json_bytes(plan):
        raise FirecrackerQualificationCollectorError('collector plan is not canonical JSON')
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    if not hmac.compare_digest(plan_sha256, expected_plan_sha256) or not hmac.compare_digest(
        digest_bytes,
        (plan_sha256 + '\n').encode('ascii'),
    ):
        raise FirecrackerQualificationCollectorError('collector-plan digest does not match its external pin')
    try:
        spec = FirecrackerWorkerSpec.model_validate_json(spec_bytes)
    except ValueError as error:
        raise FirecrackerQualificationCollectorError('retained worker specification is invalid') from error
    if spec_bytes != canonical_json_bytes(spec):
        raise FirecrackerQualificationCollectorError('retained worker specification is not canonical JSON')
    if (
        not hmac.compare_digest(firecracker_model_sha256(spec), expected_worker_spec_sha256)
        or plan.worker_spec_sha256 != expected_worker_spec_sha256
        or plan.worker_id != spec.worker_id
        or plan.worker_runtime_architecture != spec.runtime.architecture
    ):
        raise FirecrackerQualificationCollectorError(
            'collector plan does not bind the externally pinned worker specification'
        )
    return LoadedFirecrackerQualificationCollectorPlan(
        root=str(resolved),
        plan=plan,
        plan_sha256=plan_sha256,
    )


def _publish_plan(
    *,
    output_root: Path,
    plan: FirecrackerQualificationCollectorPlan,
    spec_bytes: bytes,
    expected_worker_spec_sha256: str,
) -> LoadedFirecrackerQualificationCollectorPlan:
    target = output_root.expanduser()
    if target.is_symlink():
        raise FirecrackerQualificationCollectorError('collector-plan output cannot be a symbolic link')
    target = target.absolute()
    if target.exists():
        raise FirecrackerQualificationCollectorError('collector-plan output already exists and cannot be replaced')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    staging.chmod(0o700)
    plan_bytes = canonical_json_bytes(plan)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    try:
        for name, content in (
            (COLLECTOR_PLAN_FILE, plan_bytes),
            (WORKER_SPEC_FILE, spec_bytes),
            (COLLECTOR_PLAN_SHA256_FILE, (plan_sha256 + '\n').encode('ascii')),
        ):
            _write_private_file(staging / name, content)
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except FileExistsError as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise FirecrackerQualificationCollectorError(
            'collector-plan output already exists and cannot be replaced'
        ) from error
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_firecracker_qualification_collector_plan(
        target,
        expected_plan_sha256=plan_sha256,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
    )


def _publish_collector_evidence(
    *,
    output_root: Path,
    authenticated: AuthenticatedFirecrackerQualificationRawCollection,
    spec_bytes: bytes,
    manifest_bytes: bytes,
    expected_worker_spec_sha256: str,
    expected_probe_manifest_sha256: str,
    expected_collector_public_key_hex: str,
    expected_collector_key_id: str,
    expected_driver_runtime_closure_manifest_sha256: str | None,
    expected_driver_runtime_closure_receipt_sha256: str | None,
    expected_driver_runtime_closure_sha256: str | None,
) -> LoadedFirecrackerQualificationCollectorEvidence:
    target = output_root.expanduser()
    if target.is_symlink():
        raise FirecrackerQualificationCollectorError('collector-evidence output cannot be a symbolic link')
    target = target.absolute()
    if target.exists():
        raise FirecrackerQualificationCollectorError('collector-evidence output already exists and cannot be replaced')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    staging.chmod(0o700)
    evidence_bytes = canonical_json_bytes(authenticated)
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    try:
        for name, content in (
            (COLLECTOR_EVIDENCE_FILE, evidence_bytes),
            (WORKER_SPEC_FILE, spec_bytes),
            (PROBE_MANIFEST_FILE, manifest_bytes),
            (COLLECTOR_EVIDENCE_SHA256_FILE, (evidence_sha256 + '\n').encode('ascii')),
        ):
            _write_private_file(staging / name, content)
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except FileExistsError as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise FirecrackerQualificationCollectorError(
            'collector-evidence output already exists and cannot be replaced'
        ) from error
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_firecracker_qualification_collector_evidence(
        target,
        expected_evidence_sha256=evidence_sha256,
        expected_worker_spec_sha256=expected_worker_spec_sha256,
        expected_probe_manifest_sha256=expected_probe_manifest_sha256,
        expected_collector_public_key_hex=expected_collector_public_key_hex,
        expected_collector_key_id=expected_collector_key_id,
        expected_driver_runtime_closure_manifest_sha256=(expected_driver_runtime_closure_manifest_sha256),
        expected_driver_runtime_closure_receipt_sha256=(expected_driver_runtime_closure_receipt_sha256),
        expected_driver_runtime_closure_sha256=expected_driver_runtime_closure_sha256,
    )


def read_firecracker_live_collector_private_key_file(path: Path) -> Ed25519PrivateKey:
    content = _read_private_file(path.expanduser().resolve(strict=True), _MAX_COLLECTOR_KEY_BYTES)
    return _decode_collector_private_key(content)


def read_firecracker_live_collector_private_key_fd(file_descriptor: int) -> Ed25519PrivateKey:
    if file_descriptor < 0:
        raise FirecrackerQualificationCollectorError('collector key file descriptor must be nonnegative')
    try:
        duplicate = os.dup(file_descriptor)
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot duplicate collector key file descriptor') from error
    try:
        content = bytearray()
        while len(content) <= _MAX_COLLECTOR_KEY_BYTES:
            chunk = os.read(duplicate, min(256, _MAX_COLLECTOR_KEY_BYTES + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > _MAX_COLLECTOR_KEY_BYTES:
            raise FirecrackerQualificationCollectorError('collector key input exceeds its byte limit')
        return _decode_collector_private_key(bytes(content))
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot read collector key file descriptor') from error
    finally:
        os.close(duplicate)


def _decode_collector_private_key(content: bytes) -> Ed25519PrivateKey:
    if content.endswith(b'\n'):
        content = content[:-1]
    if len(content) != 64:
        raise FirecrackerQualificationCollectorError('collector key must be one 32-byte lowercase hexadecimal seed')
    try:
        seed = bytes.fromhex(content.decode('ascii'))
    except (UnicodeDecodeError, ValueError) as error:
        raise FirecrackerQualificationCollectorError('collector key must be lowercase hexadecimal') from error
    if content != seed.hex().encode('ascii'):
        raise FirecrackerQualificationCollectorError('collector key must use canonical lowercase hexadecimal')
    return Ed25519PrivateKey.from_private_bytes(seed)


def _validate_private_artifact_root(root: Path, *, expected_files: frozenset[str]) -> Path:
    resolved = root.expanduser()
    if resolved.is_symlink():
        raise FirecrackerQualificationCollectorError('collector artifact root cannot be a symbolic link')
    try:
        resolved = resolved.resolve(strict=True)
        metadata = resolved.lstat()
        observed = {entry.name for entry in os.scandir(resolved)}
    except OSError as error:
        raise FirecrackerQualificationCollectorError('collector artifact root is unavailable') from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise FirecrackerQualificationCollectorError('collector artifact root must be a private directory')
    if observed != expected_files:
        raise FirecrackerQualificationCollectorError('collector artifact has an unexpected file inventory')
    return resolved


def _read_safe_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path.expanduser(), flags)
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot open pinned collector input') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > maximum_bytes:
            raise FirecrackerQualificationCollectorError('pinned collector input is unsafe or oversized')
        content = bytearray()
        while len(content) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > maximum_bytes:
            raise FirecrackerQualificationCollectorError('pinned collector input exceeds its byte limit')
        return bytes(content)
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot read pinned collector input') from error
    finally:
        os.close(descriptor)


def _validate_raw_drill_release_bindings(
    drill: FirecrackerQualificationRawDrillObservation,
    spec: FirecrackerWorkerSpec,
) -> None:
    executable_name = Path(spec.runtime.firecracker.source_path).name
    for binding in drill.worker_bindings:
        expected_cgroup = Path('/sys/fs/cgroup').joinpath(*spec.cgroup_parent.split('/'), binding.run_id)
        expected_jail_root = Path(spec.chroot_base_dir) / executable_name / binding.run_id / 'root'
        if (
            binding.worker_uid != spec.worker_uid
            or binding.worker_gid != spec.worker_gid
            or binding.guest_cid != spec.guest_cid
            or binding.firecracker_executable_sha256 != spec.runtime.firecracker.sha256
            or Path(binding.cgroup_path) != expected_cgroup
            or Path(binding.jail_root) != expected_jail_root
            or Path(binding.vsock_uds_path) != expected_jail_root / 'run' / 'vsock.sock'
            or Path(binding.firecracker_pid_file_path) != expected_jail_root / f'{executable_name}.pid'
            or binding.firecracker_pid == binding.jailer_pid
            or binding.process_group_id != binding.jailer_pid
            or binding.jailer_process_group_id != binding.jailer_pid
        ):
            raise FirecrackerQualificationCollectorError('raw drill worker binding differs from the pinned release')
    if drill.drill_id == FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT:
        expected_memory_bytes = spec.limits.memory_mib * 1024 * 1024
        for snapshot in drill.cgroup_snapshots:
            if (
                snapshot.cpu_max_quota_us != spec.limits.cpu_quota_us
                or snapshot.cpu_max_period_us != spec.limits.cpu_period_us
                or snapshot.memory_max_bytes != expected_memory_bytes
                or snapshot.memory_swap_max_bytes != 0
                or snapshot.pids_max != spec.limits.pids
            ):
                raise FirecrackerQualificationCollectorError('raw cgroupfs counters differ from pinned resource limits')


def _verify_raw_collection_runtime_closure_pins(
    authenticated: AuthenticatedFirecrackerQualificationRawCollection,
    *,
    expected_manifest_sha256: str | None,
    expected_receipt_sha256: str | None,
    expected_closure_sha256: str | None,
) -> None:
    collection = authenticated.collection
    supplied = (
        expected_manifest_sha256,
        expected_receipt_sha256,
        expected_closure_sha256,
    )
    production = collection.mode == FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM
    supplied_count = sum(value is not None for value in supplied)
    if supplied_count != (3 if production else 0):
        raise FirecrackerQualificationCollectorError(
            'production collector evidence requires all external driver runtime-closure pins'
        )
    if not production:
        return
    assert expected_manifest_sha256 is not None
    assert expected_receipt_sha256 is not None
    assert expected_closure_sha256 is not None
    for value, label in (
        (expected_manifest_sha256, 'driver runtime-closure manifest pin'),
        (expected_receipt_sha256, 'driver runtime-closure receipt pin'),
        (expected_closure_sha256, 'driver runtime-closure pin'),
    ):
        _require_sha256(value, label=label)
    boundary = collection.boundary_identity
    if (
        not boundary.transitive_runtime_pin_enforced
        or boundary.runtime_closure_manifest_sha256 != expected_manifest_sha256
        or boundary.runtime_closure_receipt_sha256 != expected_receipt_sha256
        or boundary.runtime_closure_sha256 != expected_closure_sha256
        or collection.driver_runtime_closure_sha256 != expected_closure_sha256
    ):
        raise FirecrackerQualificationCollectorError(
            'collector evidence differs from the externally pinned qualification-driver runtime closure'
        )


def _validate_pinned_qualification_artifact(path: Path, *, expected_sha256: str) -> None:
    _require_sha256(expected_sha256, label='qualification-guest artifact pin')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot open pinned qualification-guest artifact') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise FirecrackerQualificationCollectorError('qualification-guest artifact is not root-owned and immutable')
        observed = _sha256_descriptor(descriptor)
        if not hmac.compare_digest(observed, expected_sha256):
            raise FirecrackerQualificationCollectorError('qualification-guest artifact differs from its digest pin')
    finally:
        os.close(descriptor)


def _open_pinned_driver(path: Path, *, expected_sha256: str) -> int:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot open pinned qualification driver') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != 0
            or not stat.S_IMODE(metadata.st_mode) & 0o100
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise FirecrackerQualificationCollectorError('qualification driver is not root-owned and immutable')
        if not hmac.compare_digest(_sha256_descriptor(descriptor), expected_sha256):
            raise FirecrackerQualificationCollectorError('qualification driver differs from its external digest pin')
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _bounded_driver_exit_status(value: int) -> int | Literal['unknown']:
    return value if -255 <= value <= 255 else 'unknown'


def _run_bounded_driver_process(
    *,
    argv: tuple[str, ...],
    request: bytes,
    cwd: Path,
    env: dict[str, str],
    pass_fds: tuple[int, ...],
    timeout_seconds: int,
    stdout_byte_limit: int = _MAX_DRIVER_OUTPUT_BYTES,
    stderr_byte_limit: int = _MAX_DRIVER_STDERR_BYTES,
) -> _PinnedDriverProcessResult:
    """Run without a shell while retaining only bounded stdout and no stderr content."""

    if not argv or timeout_seconds < 1:
        raise ValueError('bounded driver invocation is invalid')
    stdout_capture = _BoundedDriverStreamCapture(
        byte_limit=stdout_byte_limit,
        retained_byte_limit=stdout_byte_limit,
    )
    stderr_capture = _BoundedDriverStreamCapture(
        byte_limit=stderr_byte_limit,
        retained_byte_limit=0,
    )

    def result(
        *,
        exit_status: int | Literal['not_started', 'timed_out', 'unknown'],
        failure_kind: FirecrackerDriverFailureKind | None,
    ) -> _PinnedDriverProcessResult:
        return _PinnedDriverProcessResult(
            exit_status=exit_status,
            stdout=bytes(stdout_capture.retained),
            stdout_byte_count=stdout_capture.byte_count,
            stdout_sha256=stdout_capture.digest.hexdigest(),
            stderr_byte_count=stderr_capture.byte_count,
            stderr_sha256=stderr_capture.digest.hexdigest(),
            failure_kind=failure_kind,
        )

    try:
        request_file = tempfile.TemporaryFile(mode='w+b')
    except OSError:
        return result(
            exit_status='not_started',
            failure_kind=FirecrackerDriverFailureKind.EXECUTION_NOT_STARTED,
        )
    with request_file:
        try:
            os.fchmod(request_file.fileno(), 0o600)
            request_file.write(request)
            request_file.flush()
            request_file.seek(0)
            process = subprocess.Popen(
                argv,
                stdin=request_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                pass_fds=pass_fds,
                close_fds=True,
            )
        except OSError:
            return result(
                exit_status='not_started',
                failure_kind=FirecrackerDriverFailureKind.EXECUTION_NOT_STARTED,
            )
        except subprocess.SubprocessError:
            return result(
                exit_status='unknown',
                failure_kind=FirecrackerDriverFailureKind.EXECUTION_ERROR,
            )

        assert process.stdout is not None
        assert process.stderr is not None
        selector = selectors.DefaultSelector()
        registered_streams: dict[int, BinaryIO] = {}
        failure_kind: FirecrackerDriverFailureKind | None = None
        drain_deadline: float | None = None

        def mark_failure(kind: FirecrackerDriverFailureKind) -> None:
            nonlocal failure_kind, drain_deadline
            if failure_kind is None:
                failure_kind = kind
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            drain_deadline = drain_deadline or time.monotonic() + _DRIVER_TERMINATION_GRACE_SECONDS

        def close_registered_streams() -> None:
            for key in tuple(selector.get_map().values()):
                try:
                    selector.unregister(key.fd)
                except (KeyError, ValueError):
                    pass
                stream = registered_streams.pop(key.fd, None)
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

        try:
            for stream, capture, limit_kind in (
                (process.stdout, stdout_capture, FirecrackerDriverFailureKind.STDOUT_LIMIT_EXCEEDED),
                (process.stderr, stderr_capture, FirecrackerDriverFailureKind.STDERR_LIMIT_EXCEEDED),
            ):
                descriptor = stream.fileno()
                os.set_blocking(descriptor, False)
                registered_streams[descriptor] = cast(BinaryIO, stream)
                selector.register(descriptor, selectors.EVENT_READ, (capture, limit_kind))

            deadline = time.monotonic() + timeout_seconds
            while selector.get_map():
                now = time.monotonic()
                if process.poll() is None and now >= deadline:
                    mark_failure(FirecrackerDriverFailureKind.EXECUTION_TIMEOUT)
                elif process.poll() is not None and drain_deadline is None:
                    drain_deadline = now + _DRIVER_TERMINATION_GRACE_SECONDS
                if drain_deadline is not None and now >= drain_deadline:
                    if failure_kind is None:
                        failure_kind = FirecrackerDriverFailureKind.EXECUTION_ERROR
                    close_registered_streams()
                    break
                next_deadline = deadline if drain_deadline is None else drain_deadline
                events = selector.select(max(0.0, min(0.1, next_deadline - now)))
                for key, _ in events:
                    capture, limit_kind = key.data
                    while True:
                        try:
                            chunk = os.read(key.fd, _DRIVER_STREAM_CHUNK_BYTES)
                        except BlockingIOError:
                            break
                        if not chunk:
                            selector.unregister(key.fd)
                            registered_streams.pop(key.fd).close()
                            break
                        if capture.consume(chunk):
                            mark_failure(limit_kind)
        except (OSError, ValueError):
            mark_failure(FirecrackerDriverFailureKind.EXECUTION_ERROR)
            close_registered_streams()
        finally:
            selector.close()

        if failure_kind is None and process.poll() is None:
            try:
                returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                mark_failure(FirecrackerDriverFailureKind.EXECUTION_TIMEOUT)
                returncode = None
        else:
            returncode = process.poll()
        if returncode is None:
            mark_failure(FirecrackerDriverFailureKind.EXECUTION_TIMEOUT)
            try:
                returncode = process.wait(timeout=_DRIVER_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                return result(exit_status='timed_out', failure_kind=failure_kind)
        if failure_kind is None and returncode != 0:
            failure_kind = FirecrackerDriverFailureKind.PROCESS_EXIT
        return result(
            exit_status=_bounded_driver_exit_status(returncode),
            failure_kind=failure_kind,
        )


def _driver_failure_diagnostic(
    *,
    challenge: FirecrackerQualificationChallenge,
    failure_kind: FirecrackerDriverFailureKind,
    exit_status: int | Literal['not_started', 'timed_out', 'unknown'],
    stdout_byte_count: int,
    stdout_sha256: str,
    stderr_byte_count: int,
    stderr_sha256: str,
) -> FirecrackerDriverFailureDiagnostic:
    return FirecrackerDriverFailureDiagnostic(
        drill_id=challenge.drill_id,
        failure_kind=failure_kind,
        exit_status=exit_status,
        stdout_byte_count=stdout_byte_count,
        stdout_sha256=stdout_sha256,
        stderr_byte_count=stderr_byte_count,
        stderr_sha256=stderr_sha256,
    )


def _driver_failure_label(kind: FirecrackerDriverFailureKind) -> str:
    return {
        FirecrackerDriverFailureKind.PROCESS_EXIT: 'pinned qualification driver returned a failed result',
        FirecrackerDriverFailureKind.STDOUT_LIMIT_EXCEEDED: ('pinned qualification driver stdout exceeded its limit'),
        FirecrackerDriverFailureKind.STDERR_LIMIT_EXCEEDED: ('pinned qualification driver stderr exceeded its limit'),
        FirecrackerDriverFailureKind.EXECUTION_TIMEOUT: 'pinned qualification driver timed out',
        FirecrackerDriverFailureKind.EXECUTION_NOT_STARTED: ('pinned qualification driver execution failed'),
        FirecrackerDriverFailureKind.EXECUTION_ERROR: 'pinned qualification driver execution failed',
        FirecrackerDriverFailureKind.INVALID_RAW_EVIDENCE: (
            'pinned qualification driver returned invalid raw evidence'
        ),
        FirecrackerDriverFailureKind.NON_CANONICAL_RAW_EVIDENCE: (
            'pinned qualification driver output is not canonical JSON'
        ),
    }[kind]


def _driver_failure_message(label: str, diagnostic: FirecrackerDriverFailureDiagnostic) -> str:
    return f'{label}; diagnostic={canonical_json_bytes(diagnostic).decode("ascii")}'


def _fresh_hex(random_bytes: Callable[[int], bytes], byte_count: int, *, label: str) -> str:
    value = random_bytes(byte_count)
    if not isinstance(value, bytes) or len(value) != byte_count:
        raise FirecrackerQualificationCollectorError(f'{label} entropy source returned the wrong byte count')
    return value.hex()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _deterministic_development_random_bytes() -> Callable[[int], bytes]:
    counter = 0

    def generate(byte_count: int) -> bytes:
        nonlocal counter
        counter += 1
        return hashlib.sha256(f'vaxreplay-development-qualification-{counter}'.encode()).digest()[:byte_count]

    return generate


def _host_prerequisites_observed(
    observation: FirecrackerHostObservation,
    *,
    worker_runtime_architecture: str,
) -> bool:
    architecture = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(
        observation.host_architecture.lower(),
        observation.host_architecture.lower(),
    )
    return (
        observation.host_os == 'Linux'
        and architecture == worker_runtime_architecture
        and observation.effective_uid == 0
        and observation.kvm_non_symlink_character_device
        and observation.kvm_read_write_access
        and observation.cgroup_v2_controller_file_present
        and _REQUIRED_CONTROLLERS.issubset(observation.cgroup_controllers)
    )


def _read_private_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot open collector-plan artifact file') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum_bytes
        ):
            raise FirecrackerQualificationCollectorError('collector-plan artifact file is unsafe or oversized')
        content = bytearray()
        while len(content) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > maximum_bytes:
            raise FirecrackerQualificationCollectorError('collector-plan artifact file exceeds its byte limit')
        return bytes(content)
    except OSError as error:
        raise FirecrackerQualificationCollectorError('cannot read collector-plan artifact file') from error
    finally:
        os.close(descriptor)


def _write_private_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short write')
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _collector_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise FirecrackerQualificationCollectorError(f'{label} must be a lowercase SHA-256 digest')


__all__ = [
    'COLLECTOR_EVIDENCE_FILE',
    'COLLECTOR_EVIDENCE_SHA256_FILE',
    'COLLECTOR_PLAN_FILE',
    'COLLECTOR_PLAN_SHA256_FILE',
    'PROBE_MANIFEST_FILE',
    'WORKER_SPEC_FILE',
    'FirecrackerCollectorDrillPlan',
    'FirecrackerCollectorHostPrimitive',
    'FirecrackerCollectorMissingCapability',
    'FirecrackerDriverFailureDiagnostic',
    'FirecrackerDriverFailureKind',
    'FirecrackerQualificationCollectorError',
    'FirecrackerQualificationCollectorPlan',
    'FirecrackerQualificationCollectorStatus',
    'LoadedFirecrackerQualificationCollectorEvidence',
    'LoadedFirecrackerQualificationCollectorPlan',
    'PinnedLinuxKvmQualificationDriver',
    'build_firecracker_qualification_collector_plan',
    'collect_and_retain_firecracker_qualification_evidence',
    'independently_verify_firecracker_qualification_collector_evidence',
    'load_firecracker_qualification_collector_evidence',
    'load_firecracker_qualification_collector_plan',
    'load_pinned_firecracker_qualification_probe_manifest',
    'read_firecracker_live_collector_private_key_fd',
    'read_firecracker_live_collector_private_key_file',
    'retain_firecracker_qualification_collector_plan',
]
