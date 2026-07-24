"""Managed startup reconciliation for the Lane A Firecracker worker.

The generic runtime can authenticate its retained guest-bootstrap journals, but it cannot infer
which Linux processes, cgroups, jail directories, vsock endpoints, or gateway capabilities still
exist after a host crash.  This module is the concrete fail-closed composition boundary for that
work.  Host and ledger access are injected so the ordering can be tested without killing local
processes; a Linux deployment adapter must only return artifacts carrying an authenticated
VaxReplay ownership record.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import AtomicDirectoryPublicationError, rename_file_noreplace
from vaxreplay.agentic.clinical_production_registry import (
    SqliteClinicalProductionRegistry,
    clinical_production_start_redemption_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalRuntime,
    FirecrackerClinicalStartupCleanupReceipt,
    FirecrackerClinicalStartupReconciliationReport,
    FirecrackerClinicalStartupReconciliationRequest,
    firecracker_clinical_startup_reconciliation_request_sha256,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

MANAGED_CLINICAL_STARTUP_CONFIG_SCHEMA_VERSION = 'vaxreplay.managed-clinical-startup-config.dev-v0.2'
MANAGED_CLINICAL_ATTEMPT_INVENTORY_SCHEMA_VERSION = 'vaxreplay.managed-clinical-attempt-inventory.dev-v0.1'
MANAGED_CLINICAL_HOST_ARTIFACT_SCHEMA_VERSION = 'vaxreplay.managed-clinical-host-artifact.dev-v0.3'
AUTHENTICATED_MANAGED_CLINICAL_CLEANUP_SCHEMA_VERSION = 'vaxreplay.authenticated-managed-clinical-cleanup.dev-v0.2'
MANAGED_CLINICAL_STARTUP_ADMISSION_SCHEMA_VERSION = 'vaxreplay.managed-clinical-startup-admission.dev-v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
_CLEANUP_KEY_ID_DOMAIN = b'vaxreplay.managed-clinical-cleanup-key-id.dev-v0.1\x00'
_CLEANUP_HMAC_DOMAIN = b'vaxreplay.authenticated-managed-clinical-cleanup.dev-v0.1\x00'
_OWNERSHIP_HMAC_DOMAIN = b'vaxreplay.managed-clinical-artifact-ownership.dev-v0.2\x00'
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024


class ManagedClinicalStartupError(RuntimeError):
    """Startup state was incomplete, ambiguous, unowned, or could not be cleaned safely."""


class ManagedClinicalStartupConfig(StrictModel):
    """Deployment-owned roots and identities which a run caller cannot override."""

    schema_version: Literal['vaxreplay.managed-clinical-startup-config.dev-v0.2'] = (
        MANAGED_CLINICAL_STARTUP_CONFIG_SCHEMA_VERSION
    )
    reconciler_id: str = Field(pattern=_ID_PATTERN)
    reconciler_version: str = Field(min_length=1, max_length=200)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    runtime_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    cleanup_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    cgroup_root: str
    jail_root: str
    vsock_root: str
    receipt_root: str
    vsock_namespace_layout: Literal['run-container-root-run-vsock.dev-v0.1'] = 'run-container-root-run-vsock.dev-v0.1'
    vsock_root_is_jail_namespace_root: Literal[True] = True
    cleanup_grace_seconds: float = Field(gt=0, le=60)
    complete_process_group_scan_required: Literal[True] = True
    complete_cgroup_scan_required: Literal[True] = True
    complete_jail_scan_required: Literal[True] = True
    complete_vsock_scan_required: Literal[True] = True
    complete_capability_scan_required: Literal[True] = True
    exact_owned_paths_only: Literal[True] = True
    cross_host_consensus_claimed: Literal[False] = False

    @field_validator('cgroup_root', 'jail_root', 'vsock_root', 'receipt_root')
    @classmethod
    def validate_root(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value or value == '/':
            raise ValueError('managed startup roots must be normalized absolute non-root paths')
        return value

    @model_validator(mode='after')
    def validate_namespace_layout(self) -> Self:
        if self.vsock_root != self.jail_root:
            raise ValueError('managed vsock root must be the jail namespace root for the pinned layout')
        roots = (self.cgroup_root, self.jail_root, self.receipt_root)
        if len(set(roots)) != len(roots):
            raise ValueError('managed cgroup, jail, and receipt roots must be distinct')
        return self


def managed_clinical_cleanup_key_id(key: bytes) -> str:
    _require_key(key)
    return hashlib.sha256(_CLEANUP_KEY_ID_DOMAIN + key).hexdigest()


class ManagedClinicalAttemptInventoryRecord(StrictModel):
    """Authoritative registry projection used to decide whether a host artifact is owned."""

    schema_version: Literal['vaxreplay.managed-clinical-attempt-inventory.dev-v0.1'] = (
        MANAGED_CLINICAL_ATTEMPT_INVENTORY_SCHEMA_VERSION
    )
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    launch_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_redemption_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    episode_id: str = Field(min_length=1, max_length=500)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    state: Literal['launched', 'succeeded', 'failed']


class ManagedClinicalArtifactKind(str):
    """String constants avoid accepting an open-ended artifact namespace."""

    PROCESS_GROUP = 'process_group'
    CGROUP = 'cgroup'
    JAIL_ROOT = 'jail_root'
    VSOCK_ENDPOINT = 'vsock_endpoint'


class ManagedClinicalHostArtifact(StrictModel):
    """One discovered item plus the authenticated ownership binding supplied by the adapter."""

    schema_version: Literal['vaxreplay.managed-clinical-host-artifact.dev-v0.3'] = (
        MANAGED_CLINICAL_HOST_ARTIFACT_SCHEMA_VERSION
    )
    artifact_kind: Literal['process_group', 'cgroup', 'jail_root', 'vsock_endpoint']
    artifact_id: str = Field(min_length=1, max_length=4096)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    launch_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_redemption_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    ownership_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    ownership_authentication_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)
    process_group_leader_start_time_ticks: int | None = Field(
        default=None,
        gt=0,
        le=2**63 - 1,
    )
    process_group_session_id: int | None = Field(default=None, gt=1, le=2**31 - 1)
    process_identity_source: Literal['durable-jailer-group', 'recovered-firecracker-child'] | None = None
    process_witness_pid: int | None = Field(default=None, gt=1, le=2**31 - 1)
    process_witness_start_time_ticks: int | None = Field(
        default=None,
        gt=0,
        le=2**63 - 1,
    )
    path_device_id: int | None = Field(default=None, ge=0, le=2**63 - 1)
    path_inode: int | None = Field(default=None, gt=0, le=2**63 - 1)
    process_cgroup_device_id: int | None = Field(default=None, ge=0, le=2**63 - 1)
    process_cgroup_inode: int | None = Field(default=None, gt=0, le=2**63 - 1)
    ownership_record_authenticated: Literal[True] = True

    @model_validator(mode='after')
    def validate_kernel_identity(self) -> Self:
        if self.artifact_kind == ManagedClinicalArtifactKind.PROCESS_GROUP:
            if (
                self.process_group_session_id is None
                or self.process_identity_source is None
                or self.process_witness_pid is None
                or self.process_witness_start_time_ticks is None
                or self.path_device_id is not None
                or self.path_inode is not None
                or self.process_cgroup_device_id is None
                or self.process_cgroup_inode is None
                or (
                    self.process_identity_source == 'durable-jailer-group'
                    and self.process_group_leader_start_time_ticks is None
                )
                or (
                    self.process_identity_source == 'recovered-firecracker-child'
                    and self.process_group_leader_start_time_ticks is not None
                )
            ):
                raise ValueError('process-group ownership must carry one truthful durable or recovered identity')
        elif (
            self.process_group_leader_start_time_ticks is not None
            or self.process_group_session_id is not None
            or self.process_identity_source is not None
            or self.process_witness_pid is not None
            or self.process_witness_start_time_ticks is not None
            or self.path_device_id is None
            or self.path_inode is None
            or self.process_cgroup_device_id is not None
            or self.process_cgroup_inode is not None
        ):
            raise ValueError('path artifacts must bind device/inode and no process start time')
        return self


class ManagedClinicalCapability(StrictModel):
    capability_id: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    launch_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_redemption_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    ownership_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    ownership_authentication_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)
    ownership_record_authenticated: Literal[True] = True


type ManagedClinicalOwnedItem = ManagedClinicalHostArtifact | ManagedClinicalCapability


def managed_clinical_ownership_hmac(
    item: ManagedClinicalOwnedItem,
    *,
    key: bytes,
) -> str:
    """Authenticate one exact host artifact/capability, not merely its run label."""

    _require_key(key)
    if isinstance(item, ManagedClinicalHostArtifact):
        canonical: ManagedClinicalOwnedItem = ManagedClinicalHostArtifact.model_validate_json(
            canonical_json_bytes(item)
        )
    elif isinstance(item, ManagedClinicalCapability):
        canonical = ManagedClinicalCapability.model_validate_json(canonical_json_bytes(item))
    else:
        raise TypeError('managed ownership HMAC requires a typed owned item')
    unsigned = canonical.model_copy(update={'ownership_authentication_hmac_sha256': '0' * 64})
    return hmac.new(
        key,
        _OWNERSHIP_HMAC_DOMAIN + canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()


class ManagedClinicalHostAdapter(Protocol):
    """Complete Linux inventory and exact-artifact destructive operations."""

    def owned_run_ids(self) -> tuple[str, ...]: ...

    def scan_process_groups(self) -> tuple[ManagedClinicalHostArtifact, ...]: ...

    def scan_cgroups(self) -> tuple[ManagedClinicalHostArtifact, ...]: ...

    def scan_jail_roots(self) -> tuple[ManagedClinicalHostArtifact, ...]: ...

    def scan_vsock_endpoints(self) -> tuple[ManagedClinicalHostArtifact, ...]: ...

    def terminate_process_group(self, artifact: ManagedClinicalHostArtifact, *, grace_seconds: float) -> None: ...

    def reap_process_group(self, artifact: ManagedClinicalHostArtifact) -> None: ...

    def remove_vsock_endpoint(self, artifact: ManagedClinicalHostArtifact) -> None: ...

    def remove_cgroup(self, artifact: ManagedClinicalHostArtifact) -> None: ...

    def remove_jail_root(self, artifact: ManagedClinicalHostArtifact) -> None: ...

    def finalize_reconciled_run(self, run_id: str) -> None: ...


class ManagedClinicalCapabilityLedger(Protocol):
    """Durable capability inventory; an in-memory secret map is insufficient at restart."""

    def inventory(self) -> tuple[ManagedClinicalCapability, ...]: ...

    def revoke(self, capability: ManagedClinicalCapability) -> None: ...


class ManagedClinicalAttemptInventory(Protocol):
    """Read-only projection from the single managed registry authority."""

    @property
    def authority_id(self) -> str: ...

    def inventory(self) -> tuple[ManagedClinicalAttemptInventoryRecord, ...]: ...


class ManagedSqliteClinicalAttemptInventory:
    """Complete adapter over the one stopped/root-owned managed SQLite authority."""

    def __init__(self, registry: SqliteClinicalProductionRegistry) -> None:
        self._registry = registry

    @property
    def authority_id(self) -> str:
        return self._registry.authority_id

    def inventory(self) -> tuple[ManagedClinicalAttemptInventoryRecord, ...]:
        values: list[ManagedClinicalAttemptInventoryRecord] = []
        for reservation_sha256 in self._registry.reservation_hashes():
            context = self._registry.reservation_context(reservation_sha256)
            reservation = context.reservation
            for record in self._registry.task_records(reservation_sha256):
                launch = record.launch
                if launch is None:
                    continue
                if record.state == 'reserved':
                    raise ManagedClinicalStartupError(
                        'authoritative attempt inventory has launch data in a reserved record'
                    )
                redemption = record.start_redemption
                values.append(
                    ManagedClinicalAttemptInventoryRecord(
                        registry_authority_id=reservation.registry_authority_id,
                        reservation_sha256=reservation_sha256,
                        launch_sha256=clinical_production_task_launch_sha256(launch),
                        start_redemption_sha256=(
                            None if redemption is None else clinical_production_start_redemption_sha256(redemption)
                        ),
                        run_id=launch.run_id,
                        episode_id=record.episode_id,
                        worker_spec_sha256=reservation.system.worker_spec_sha256,
                        state=record.state,
                    )
                )
        return tuple(sorted(values, key=lambda item: (item.run_id, item.reservation_sha256)))


class AuthenticatedManagedClinicalStartupCleanup(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-managed-clinical-cleanup.dev-v0.2'] = (
        AUTHENTICATED_MANAGED_CLINICAL_CLEANUP_SCHEMA_VERSION
    )
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    reconciliation_request: FirecrackerClinicalStartupReconciliationRequest
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    process_group_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    cgroup_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    jail_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    vsock_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    capability_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    cleanup_receipt: FirecrackerClinicalStartupCleanupReceipt
    cleanup_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    cleanup_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)
    persisted_path: str
    persisted_create_once: Literal[True] = True
    file_fsync_complete: Literal[True] = True
    parent_directory_fsync_complete: Literal[True] = True
    process_groups_terminated_before_capability_revocation: Literal[True] = True
    process_groups_reaped_before_capability_revocation: Literal[True] = True
    capabilities_revoked_before_artifact_removal: Literal[True] = True
    post_cleanup_inventory_empty: Literal[True] = True
    independent_host_attestation: Literal[False] = False
    cross_host_consensus_claimed: Literal[False] = False

    @field_validator('persisted_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed startup receipt path must be normalized and absolute')
        return value

    @model_validator(mode='after')
    def validate_request_binding(self) -> Self:
        if self.request_sha256 != firecracker_clinical_startup_reconciliation_request_sha256(
            self.reconciliation_request
        ):
            raise ValueError('managed startup request differs from its retained digest')
        return self


class ManagedClinicalStartupAdmissionReport(StrictModel):
    """Managed wrapper which upgrades the generic adapter assertion with durable authentication."""

    schema_version: Literal['vaxreplay.managed-clinical-startup-admission.dev-v0.1'] = (
        MANAGED_CLINICAL_STARTUP_ADMISSION_SCHEMA_VERSION
    )
    runtime_report: FirecrackerClinicalStartupReconciliationReport
    authenticated_cleanup: AuthenticatedManagedClinicalStartupCleanup
    global_startup_reconciliation_required: Literal[True] = True
    cleanup_receipt_cryptographically_authenticated: Literal[True] = True
    cleanup_receipt_durably_persisted: Literal[True] = True
    runtime_preparation_admitted: Literal[True] = True
    independent_host_attestation: Literal[False] = False
    linux_kvm_cleanup_qualified: Literal[False] = False
    official_execution_qualified: Literal[False] = False

    @model_validator(mode='after')
    def validate_binding(self) -> Self:
        if (
            self.authenticated_cleanup.request_sha256
            != firecracker_clinical_startup_reconciliation_request_sha256(self.runtime_report.request)
            or self.authenticated_cleanup.cleanup_receipt != self.runtime_report.cleanup_receipt
        ):
            raise ValueError('managed admission wrapper differs from the runtime reconciliation')
        return self


def managed_clinical_startup_config_sha256(config: ManagedClinicalStartupConfig) -> str:
    return _sha256(canonical_json_bytes(config))


def verify_authenticated_managed_cleanup(
    artifact: AuthenticatedManagedClinicalStartupCleanup,
    *,
    key: bytes,
    expected_key_id: str,
    expected_config_sha256: str,
    expected_request_sha256: str,
) -> FirecrackerClinicalStartupCleanupReceipt:
    _require_key(key)
    canonical = AuthenticatedManagedClinicalStartupCleanup.model_validate_json(canonical_json_bytes(artifact))
    if (
        canonical.cleanup_receipt_key_id != expected_key_id
        or managed_clinical_cleanup_key_id(key) != expected_key_id
        or canonical.config_sha256 != expected_config_sha256
        or canonical.request_sha256 != expected_request_sha256
    ):
        raise ManagedClinicalStartupError('managed cleanup receipt differs from its deployment pins')
    unsigned = canonical.model_copy(update={'cleanup_hmac_sha256': '0' * 64})
    expected_hmac = hmac.new(
        key,
        _CLEANUP_HMAC_DOMAIN + canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hmac, canonical.cleanup_hmac_sha256):
        raise ManagedClinicalStartupError('managed cleanup receipt authentication failed')
    return canonical.cleanup_receipt


def load_authenticated_managed_cleanup(
    path: Path,
    *,
    expected_root: Path,
) -> AuthenticatedManagedClinicalStartupCleanup:
    """Reload exact create-once cleanup bytes from the private durable receipt directory."""

    try:
        root = expected_root.resolve(strict=True)
        supplied = path.resolve(strict=True)
    except OSError:
        raise ManagedClinicalStartupError('managed cleanup receipt is unavailable') from None
    if supplied.parent != root or supplied.name != path.name:
        raise ManagedClinicalStartupError('managed cleanup receipt escaped its configured root')
    try:
        before = supplied.lstat()
    except OSError:
        raise ManagedClinicalStartupError('managed cleanup receipt is unavailable') from None
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
    ):
        raise ManagedClinicalStartupError('managed cleanup receipt must be one owned mode-0600 regular file')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(supplied, flags)
    except OSError:
        raise ManagedClinicalStartupError('managed cleanup receipt could not be opened safely') from None
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ManagedClinicalStartupError('managed cleanup receipt changed while opening')
        content = bytearray()
        while len(content) <= _MAX_RECEIPT_BYTES:
            block = os.read(descriptor, min(1024 * 1024, _MAX_RECEIPT_BYTES + 1 - len(content)))
            if not block:
                break
            content.extend(block)
        if not content or len(content) > _MAX_RECEIPT_BYTES:
            raise ManagedClinicalStartupError('managed cleanup receipt has invalid size')
    finally:
        os.close(descriptor)
    try:
        after = supplied.lstat()
    except OSError:
        raise ManagedClinicalStartupError('managed cleanup receipt disappeared while reading') from None
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        raise ManagedClinicalStartupError('managed cleanup receipt changed while reading')
    try:
        artifact = AuthenticatedManagedClinicalStartupCleanup.model_validate_json(bytes(content))
    except ValueError:
        raise ManagedClinicalStartupError('managed cleanup receipt has an invalid strict schema') from None
    if canonical_json_bytes(artifact) != bytes(content) or artifact.persisted_path != str(supplied):
        raise ManagedClinicalStartupError('managed cleanup receipt is non-canonical or names a different path')
    return artifact


def reconcile_canonical_managed_runtime_startup(
    runtime: FirecrackerClinicalRuntime,
    *,
    reconciler: ManagedClinicalStartupReconciler,
) -> ManagedClinicalStartupAdmissionReport:
    """Mandatory startup gate for the managed operator composition.

    A runtime constructed without ``require_global_startup_reconciliation=True`` is rejected,
    even if its current journal directory happens to be empty.  Admission happens only after the
    reconciler's independently persisted HMAC wrapper is verified again.
    """

    if not runtime.startup_reconciliation_required:
        raise ManagedClinicalStartupError(
            'canonical managed runtime was not configured for global startup reconciliation'
        )
    report = runtime.reconcile_startup(reconciler=reconciler)
    artifact = reconciler.last_authenticated_receipt
    if artifact is None:
        raise ManagedClinicalStartupError('managed reconciler did not retain authenticated evidence')
    persisted = load_authenticated_managed_cleanup(
        Path(artifact.persisted_path),
        expected_root=Path(reconciler.config.receipt_root),
    )
    if persisted != artifact:
        raise ManagedClinicalStartupError('runtime cleanup evidence differs from the durably persisted receipt')
    receipt = verify_authenticated_managed_cleanup(
        persisted,
        key=reconciler._key,
        expected_key_id=reconciler.config.cleanup_receipt_key_id,
        expected_config_sha256=managed_clinical_startup_config_sha256(reconciler.config),
        expected_request_sha256=(firecracker_clinical_startup_reconciliation_request_sha256(report.request)),
    )
    if receipt != report.cleanup_receipt or runtime.startup_reconciliation_required:
        raise ManagedClinicalStartupError('runtime admitted a different or incomplete startup cleanup')
    return ManagedClinicalStartupAdmissionReport(
        runtime_report=report,
        authenticated_cleanup=artifact,
    )


class ManagedClinicalStartupReconciler:
    """Scan every managed namespace, clean exact owned artifacts, and retain an HMAC receipt."""

    def __init__(
        self,
        *,
        config: ManagedClinicalStartupConfig,
        host: ManagedClinicalHostAdapter,
        capabilities: ManagedClinicalCapabilityLedger,
        attempts: ManagedClinicalAttemptInventory,
        cleanup_receipt_key: bytes,
        clock: Callable[[], datetime] | None = None,
        reconciliation_complete: Callable[[AuthenticatedManagedClinicalStartupCleanup], None] | None = None,
    ) -> None:
        _require_key(cleanup_receipt_key)
        if managed_clinical_cleanup_key_id(cleanup_receipt_key) != config.cleanup_receipt_key_id:
            raise ValueError('managed cleanup key differs from its deployment key ID')
        if attempts.authority_id != config.registry_authority_id:
            raise ValueError('managed attempt inventory belongs to a different authority')
        self.config = config
        self.host = host
        self.capabilities = capabilities
        self.attempts = attempts
        self._key = bytes(cleanup_receipt_key)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._reconciliation_complete = reconciliation_complete
        self.last_authenticated_receipt: AuthenticatedManagedClinicalStartupCleanup | None = None

    def reconcile(
        self,
        request: FirecrackerClinicalStartupReconciliationRequest,
    ) -> FirecrackerClinicalStartupCleanupReceipt:
        request = FirecrackerClinicalStartupReconciliationRequest.model_validate_json(canonical_json_bytes(request))
        if (
            request.runtime_config_sha256 != self.config.runtime_config_sha256
            or request.worker_spec_sha256 != self.config.worker_spec_sha256
        ):
            raise ManagedClinicalStartupError('startup request differs from managed runtime pins')
        before = self._scan()
        attempts = self._canonical_attempts(self.attempts.inventory())
        self._validate_owned_inventory(request=request, inventory=before, attempts=attempts)

        grouped: dict[str, dict[str, list[ManagedClinicalHostArtifact]]] = defaultdict(lambda: defaultdict(list))
        for artifact in before.host_artifacts:
            grouped[artifact.run_id][artifact.artifact_kind].append(artifact)
        capabilities_by_run: dict[str, list[ManagedClinicalCapability]] = defaultdict(list)
        for capability in before.capabilities:
            capabilities_by_run[capability.run_id].append(capability)

        # Safe order is deliberate: stop and reap every execution source, then revoke provider
        # authority, then remove communication and filesystem/cgroup state.
        all_owned_run_ids = set(before.owned_run_ids) | set(grouped) | set(capabilities_by_run)
        for run_id in sorted(all_owned_run_ids):
            run = grouped[run_id]
            for artifact in run[ManagedClinicalArtifactKind.PROCESS_GROUP]:
                self.host.terminate_process_group(
                    artifact,
                    grace_seconds=self.config.cleanup_grace_seconds,
                )
            for artifact in run[ManagedClinicalArtifactKind.PROCESS_GROUP]:
                self.host.reap_process_group(artifact)
        for capability in before.capabilities:
            self.capabilities.revoke(capability)
        for run_id in sorted(grouped):
            run = grouped[run_id]
            for artifact in run[ManagedClinicalArtifactKind.VSOCK_ENDPOINT]:
                self.host.remove_vsock_endpoint(artifact)
            for artifact in run[ManagedClinicalArtifactKind.CGROUP]:
                self.host.remove_cgroup(artifact)
            for artifact in run[ManagedClinicalArtifactKind.JAIL_ROOT]:
                self.host.remove_jail_root(artifact)
        for run_id in sorted(all_owned_run_ids):
            self.host.finalize_reconciled_run(run_id)

        after = self._scan()
        if after.host_artifacts or after.capabilities or after.owned_run_ids:
            raise ManagedClinicalStartupError('managed startup cleanup left surviving artifacts')

        now = self._now()
        if now < request.requested_at:
            now = request.requested_at
        request_sha256 = firecracker_clinical_startup_reconciliation_request_sha256(request)
        receipt = FirecrackerClinicalStartupCleanupReceipt(
            reconciler_id=self.config.reconciler_id,
            reconciler_version=self.config.reconciler_version,
            reconciliation_request_sha256=request_sha256,
            retained_journal_count=len(request.retained_journals),
            worker_inventory_sha256=_sha256(_model_sequence_bytes(before.process_groups)),
            ephemeral_run_artifact_inventory_sha256=_sha256(
                _model_sequence_bytes(before.cgroups + before.jail_roots + before.vsock_endpoints)
            ),
            capability_inventory_sha256=_sha256(_model_sequence_bytes(before.capabilities)),
            attempt_registry_inventory_sha256=_sha256(_model_sequence_bytes(attempts)),
            cleanup_evidence_sha256=_sha256(
                canonical_json_bytes(
                    {
                        'request_sha256': request_sha256,
                        'before_sha256': before.sha256,
                        'after_sha256': after.sha256,
                    }
                )
            ),
            discovered_worker_count=len(before.process_groups),
            terminated_worker_count=len(before.process_groups),
            discovered_ephemeral_run_artifact_count=(
                len(before.cgroups) + len(before.jail_roots) + len(before.vsock_endpoints)
            ),
            removed_ephemeral_run_artifact_count=(
                len(before.cgroups) + len(before.jail_roots) + len(before.vsock_endpoints)
            ),
            discovered_capability_count=len(before.capabilities),
            revoked_capability_count=len(before.capabilities),
            reconciled_at=now,
        )
        authenticated = self._authenticate_and_persist(
            request=request,
            inventory=before,
            attempts=attempts,
            receipt=receipt,
        )
        verify_authenticated_managed_cleanup(
            authenticated,
            key=self._key,
            expected_key_id=self.config.cleanup_receipt_key_id,
            expected_config_sha256=managed_clinical_startup_config_sha256(self.config),
            expected_request_sha256=request_sha256,
        )
        self.last_authenticated_receipt = authenticated
        if self._reconciliation_complete is not None:
            self._reconciliation_complete(authenticated)
        return receipt

    def _validate_owned_inventory(
        self,
        *,
        request: FirecrackerClinicalStartupReconciliationRequest,
        inventory: _StartupInventory,
        attempts: tuple[ManagedClinicalAttemptInventoryRecord, ...],
    ) -> None:
        attempts_by_run = {item.run_id: item for item in attempts}
        if len(attempts_by_run) != len(attempts):
            raise ManagedClinicalStartupError('attempt inventory contains duplicate run IDs')
        journals = {item.run_id: item for item in request.retained_journals}
        if any(run_id not in attempts_by_run for run_id in inventory.owned_run_ids):
            raise ManagedClinicalStartupError('host ownership ledger contains a run absent from the attempt authority')
        for artifact in inventory.host_artifacts:
            self._validate_path(artifact)
            self._validate_authenticated_ownership(artifact)
            attempt = attempts_by_run.get(artifact.run_id)
            if attempt is None or not _artifact_matches_attempt(artifact, attempt):
                raise ManagedClinicalStartupError('host inventory contains an unowned artifact')
        for capability in inventory.capabilities:
            self._validate_authenticated_ownership(capability)
            attempt = attempts_by_run.get(capability.run_id)
            if attempt is None or not _capability_matches_attempt(capability, attempt):
                raise ManagedClinicalStartupError('capability inventory contains an unowned capability')
        for run_id, journal in journals.items():
            attempt = attempts_by_run.get(run_id)
            if (
                attempt is None
                or attempt.start_redemption_sha256 != journal.start_redemption_sha256
                or attempt.worker_spec_sha256 != journal.worker_spec_sha256
            ):
                raise ManagedClinicalStartupError(
                    'retained bootstrap journal is absent from the authoritative attempt inventory'
                )
        identities = [(item.artifact_kind, item.artifact_id) for item in inventory.host_artifacts]
        if len(identities) != len(set(identities)):
            raise ManagedClinicalStartupError('host inventory contains an ambiguous duplicate artifact')
        capability_ids = [item.capability_id for item in inventory.capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ManagedClinicalStartupError('capability inventory contains an ambiguous duplicate')
        ownership_by_run: dict[str, set[str]] = defaultdict(set)
        count_by_run_and_kind: dict[tuple[str, str], int] = defaultdict(int)
        for artifact in inventory.host_artifacts:
            ownership_by_run[artifact.run_id].add(artifact.ownership_record_sha256)
            count_by_run_and_kind[(artifact.run_id, artifact.artifact_kind)] += 1
        for capability in inventory.capabilities:
            ownership_by_run[capability.run_id].add(capability.ownership_record_sha256)
            count_by_run_and_kind[(capability.run_id, 'capability')] += 1
        if any(len(values) != 1 for values in ownership_by_run.values()) or any(
            count != 1 for count in count_by_run_and_kind.values()
        ):
            raise ManagedClinicalStartupError(
                'managed inventory has ambiguous per-run ownership or duplicate artifact kinds'
            )

    def _validate_path(self, artifact: ManagedClinicalHostArtifact) -> None:
        if artifact.artifact_kind == ManagedClinicalArtifactKind.PROCESS_GROUP:
            if not artifact.artifact_id.startswith('pgid:') or not artifact.artifact_id[5:].isdigit():
                raise ManagedClinicalStartupError('process-group inventory has an invalid exact ID')
            process_group = int(artifact.artifact_id[5:])
            if process_group <= 1 or process_group > 2**31 - 1:
                raise ManagedClinicalStartupError('process-group inventory has an unsafe exact ID')
            return
        root_by_kind = {
            ManagedClinicalArtifactKind.CGROUP: self.config.cgroup_root,
            ManagedClinicalArtifactKind.JAIL_ROOT: self.config.jail_root,
            ManagedClinicalArtifactKind.VSOCK_ENDPOINT: self.config.vsock_root,
        }
        root = PurePosixPath(root_by_kind[artifact.artifact_kind])
        path = PurePosixPath(artifact.artifact_id)
        if (
            not path.is_absolute()
            or '..' in path.parts
            or str(path) != artifact.artifact_id
            or path == root
            or root not in path.parents
            or artifact.run_id not in path.parts
        ):
            raise ManagedClinicalStartupError('host inventory contains a path outside its owned root')

    def _validate_authenticated_ownership(self, item: ManagedClinicalOwnedItem) -> None:
        expected = managed_clinical_ownership_hmac(item, key=self._key)
        if not hmac.compare_digest(expected, item.ownership_authentication_hmac_sha256):
            raise ManagedClinicalStartupError('managed inventory contains unauthenticated ownership metadata')

    def _scan(self) -> _StartupInventory:
        supplied_owned_run_ids = self.host.owned_run_ids()
        owned_run_ids = tuple(sorted(set(supplied_owned_run_ids)))
        if len(owned_run_ids) != len(supplied_owned_run_ids):
            raise ManagedClinicalStartupError('host ownership ledger contains duplicate run IDs')
        if any(
            len(run_id) != 32 or any(character not in '0123456789abcdef' for character in run_id)
            for run_id in owned_run_ids
        ):
            raise ManagedClinicalStartupError('host ownership ledger contains an invalid run ID')
        process_groups = self._canonical_host(
            self.host.scan_process_groups(), ManagedClinicalArtifactKind.PROCESS_GROUP
        )
        cgroups = self._canonical_host(self.host.scan_cgroups(), ManagedClinicalArtifactKind.CGROUP)
        jail_roots = self._canonical_host(self.host.scan_jail_roots(), ManagedClinicalArtifactKind.JAIL_ROOT)
        vsock_endpoints = self._canonical_host(
            self.host.scan_vsock_endpoints(), ManagedClinicalArtifactKind.VSOCK_ENDPOINT
        )
        capabilities = tuple(
            sorted(
                (
                    ManagedClinicalCapability.model_validate_json(canonical_json_bytes(item))
                    for item in self.capabilities.inventory()
                ),
                key=lambda item: item.capability_id,
            )
        )
        return _StartupInventory(
            owned_run_ids=owned_run_ids,
            process_groups=process_groups,
            cgroups=cgroups,
            jail_roots=jail_roots,
            vsock_endpoints=vsock_endpoints,
            capabilities=capabilities,
        )

    @staticmethod
    def _canonical_host(
        supplied: tuple[ManagedClinicalHostArtifact, ...],
        kind: str,
    ) -> tuple[ManagedClinicalHostArtifact, ...]:
        values = tuple(ManagedClinicalHostArtifact.model_validate_json(canonical_json_bytes(item)) for item in supplied)
        if any(item.artifact_kind != kind for item in values):
            raise ManagedClinicalStartupError('host adapter returned an artifact in the wrong scan')
        return tuple(sorted(values, key=lambda item: (item.run_id, item.artifact_id)))

    @staticmethod
    def _canonical_attempts(
        supplied: tuple[ManagedClinicalAttemptInventoryRecord, ...],
    ) -> tuple[ManagedClinicalAttemptInventoryRecord, ...]:
        return tuple(
            sorted(
                (
                    ManagedClinicalAttemptInventoryRecord.model_validate_json(canonical_json_bytes(item))
                    for item in supplied
                ),
                key=lambda item: (item.run_id, item.reservation_sha256),
            )
        )

    def _authenticate_and_persist(
        self,
        *,
        request: FirecrackerClinicalStartupReconciliationRequest,
        inventory: _StartupInventory,
        attempts: tuple[ManagedClinicalAttemptInventoryRecord, ...],
        receipt: FirecrackerClinicalStartupCleanupReceipt,
    ) -> AuthenticatedManagedClinicalStartupCleanup:
        request_sha256 = firecracker_clinical_startup_reconciliation_request_sha256(request)
        root = _prepare_private_root(Path(self.config.receipt_root))
        path = root / f'{request_sha256}.json'
        unsigned = AuthenticatedManagedClinicalStartupCleanup(
            config_sha256=managed_clinical_startup_config_sha256(self.config),
            reconciliation_request=request,
            request_sha256=request_sha256,
            process_group_inventory_sha256=_sha256(_model_sequence_bytes(inventory.process_groups)),
            cgroup_inventory_sha256=_sha256(_model_sequence_bytes(inventory.cgroups)),
            jail_inventory_sha256=_sha256(_model_sequence_bytes(inventory.jail_roots)),
            vsock_inventory_sha256=_sha256(_model_sequence_bytes(inventory.vsock_endpoints)),
            capability_inventory_sha256=_sha256(_model_sequence_bytes(inventory.capabilities)),
            attempt_inventory_sha256=_sha256(_model_sequence_bytes(attempts)),
            cleanup_receipt=receipt,
            cleanup_receipt_key_id=self.config.cleanup_receipt_key_id,
            cleanup_hmac_sha256='0' * 64,
            persisted_path=str(path),
        )
        artifact = unsigned.model_copy(
            update={
                'cleanup_hmac_sha256': hmac.new(
                    self._key,
                    _CLEANUP_HMAC_DOMAIN + canonical_json_bytes(unsigned),
                    hashlib.sha256,
                ).hexdigest()
            }
        )
        _write_create_once(path, canonical_json_bytes(artifact))
        persisted = load_authenticated_managed_cleanup(path, expected_root=root)
        if persisted != artifact:
            raise ManagedClinicalStartupError('persisted managed cleanup receipt differs from the authenticated bytes')
        return persisted

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ManagedClinicalStartupError('managed cleanup clock must return an aware time')
        return value.astimezone(UTC)


class _StartupInventory:
    __slots__ = (
        'owned_run_ids',
        'process_groups',
        'cgroups',
        'jail_roots',
        'vsock_endpoints',
        'capabilities',
    )

    def __init__(
        self,
        *,
        owned_run_ids: tuple[str, ...],
        process_groups: tuple[ManagedClinicalHostArtifact, ...],
        cgroups: tuple[ManagedClinicalHostArtifact, ...],
        jail_roots: tuple[ManagedClinicalHostArtifact, ...],
        vsock_endpoints: tuple[ManagedClinicalHostArtifact, ...],
        capabilities: tuple[ManagedClinicalCapability, ...],
    ) -> None:
        self.owned_run_ids = owned_run_ids
        self.process_groups = process_groups
        self.cgroups = cgroups
        self.jail_roots = jail_roots
        self.vsock_endpoints = vsock_endpoints
        self.capabilities = capabilities

    @property
    def host_artifacts(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        return self.process_groups + self.cgroups + self.jail_roots + self.vsock_endpoints

    @property
    def sha256(self) -> str:
        return _sha256(
            canonical_json_bytes(
                {
                    'owned_run_ids': list(self.owned_run_ids),
                    'process_groups': [item.model_dump(mode='json') for item in self.process_groups],
                    'cgroups': [item.model_dump(mode='json') for item in self.cgroups],
                    'jail_roots': [item.model_dump(mode='json') for item in self.jail_roots],
                    'vsock_endpoints': [item.model_dump(mode='json') for item in self.vsock_endpoints],
                    'capabilities': [item.model_dump(mode='json') for item in self.capabilities],
                }
            )
        )


def _artifact_matches_attempt(
    artifact: ManagedClinicalHostArtifact,
    attempt: ManagedClinicalAttemptInventoryRecord,
) -> bool:
    return (
        artifact.registry_authority_id,
        artifact.reservation_sha256,
        artifact.launch_sha256,
        artifact.start_redemption_sha256,
        artifact.worker_spec_sha256,
    ) == (
        attempt.registry_authority_id,
        attempt.reservation_sha256,
        attempt.launch_sha256,
        attempt.start_redemption_sha256,
        attempt.worker_spec_sha256,
    )


def _capability_matches_attempt(
    capability: ManagedClinicalCapability,
    attempt: ManagedClinicalAttemptInventoryRecord,
) -> bool:
    return (
        capability.registry_authority_id,
        capability.reservation_sha256,
        capability.launch_sha256,
        capability.start_redemption_sha256,
        capability.worker_spec_sha256,
    ) == (
        attempt.registry_authority_id,
        attempt.reservation_sha256,
        attempt.launch_sha256,
        attempt.start_redemption_sha256,
        attempt.worker_spec_sha256,
    )


def _prepare_private_root(path: Path) -> Path:
    supplied = path.expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ManagedClinicalStartupError(
            'cleanup receipt root must be absolute and cannot contain symbolic-link components'
        )
    try:
        resolved_before_create = supplied.resolve(strict=False)
    except OSError:
        raise ManagedClinicalStartupError('cleanup receipt root could not be resolved safely') from None
    if resolved_before_create != supplied:
        raise ManagedClinicalStartupError(
            'cleanup receipt root must be absolute and cannot contain symbolic-link components'
        )
    try:
        supplied.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        pass
    except OSError:
        raise ManagedClinicalStartupError('cleanup receipt root could not be created') from None
    resolved = supplied.resolve(strict=True)
    metadata = resolved.lstat()
    if (
        resolved != supplied
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ManagedClinicalStartupError('cleanup receipt root must be symlink-free and owned mode-0700')
    return resolved


def _write_create_once(path: Path, content: bytes) -> None:
    if not content or len(content) > _MAX_RECEIPT_BYTES:
        raise ManagedClinicalStartupError('cleanup receipt has invalid size')
    root = _prepare_private_root(path.parent)
    if path.parent != root or path.name.startswith('.') or '/' in path.name:
        raise ManagedClinicalStartupError('cleanup receipt output escaped its configured root')
    _reap_incomplete_cleanup_receipt_staging(root)
    staging = root / f'.cleanup-stage-{secrets.token_hex(32)}'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    descriptor: int | None = None
    created = False
    staging_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(staging, flags, 0o600)
        created = True
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        staging_identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ManagedClinicalStartupError('cleanup receipt staging file could not be pinned as owned mode-0600')
        written = 0
        while written < len(content):
            try:
                count = os.write(descriptor, content[written:])
            except InterruptedError:
                continue
            if count <= 0:
                raise ManagedClinicalStartupError('cleanup receipt write did not make progress')
            written += count
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_uid != os.geteuid()
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o600
            or final.st_size != len(content)
            or (final.st_dev, final.st_ino) != staging_identity
        ):
            raise ManagedClinicalStartupError('cleanup receipt staging file changed while writing')
        os.close(descriptor)
        descriptor = None
        rename_file_noreplace(staging, path)
        created = False
    except (AtomicDirectoryPublicationError, ManagedClinicalStartupError, OSError):
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created and staging_identity is not None:
            try:
                current = staging.lstat()
                if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == staging_identity:
                    staging.unlink()
                    _fsync_cleanup_receipt_directory(root)
            except OSError:
                pass
        raise ManagedClinicalStartupError('cleanup receipt path is unavailable or already used') from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _fsync_cleanup_receipt_directory(root)


def _reap_incomplete_cleanup_receipt_staging(root: Path) -> None:
    """Discard only unpublished operation-owned receipt stages after a process crash."""

    try:
        entries = tuple(root.iterdir())
    except OSError:
        raise ManagedClinicalStartupError('cleanup receipt staging inventory is unavailable') from None
    removed = False
    prefix = '.cleanup-stage-'
    for path in entries:
        if not path.name.startswith(prefix):
            continue
        nonce = path.name.removeprefix(prefix)
        if len(nonce) != 64 or any(character not in '0123456789abcdef' for character in nonce):
            raise ManagedClinicalStartupError('cleanup receipt root contains an ambiguous staging name')
        try:
            before = path.lstat()
        except OSError:
            raise ManagedClinicalStartupError('cleanup receipt staging file is unavailable') from None
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > _MAX_RECEIPT_BYTES
        ):
            raise ManagedClinicalStartupError('cleanup receipt staging file has unsafe metadata')
        try:
            after = path.lstat()
            if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                raise ManagedClinicalStartupError('cleanup receipt staging file changed before cleanup')
            path.unlink()
        except OSError:
            raise ManagedClinicalStartupError('cleanup receipt staging cleanup failed') from None
        removed = True
    if removed:
        _fsync_cleanup_receipt_directory(root)


def _fsync_cleanup_receipt_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError('managed cleanup authentication key must contain at least 32 bytes')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_sequence_bytes(values: tuple[StrictModel, ...]) -> bytes:
    return canonical_json_bytes([item.model_dump(mode='json') for item in values])


__all__ = [
    'AuthenticatedManagedClinicalStartupCleanup',
    'ManagedClinicalAttemptInventory',
    'ManagedClinicalAttemptInventoryRecord',
    'ManagedClinicalCapability',
    'ManagedClinicalCapabilityLedger',
    'ManagedClinicalHostAdapter',
    'ManagedClinicalHostArtifact',
    'ManagedClinicalStartupConfig',
    'ManagedClinicalStartupAdmissionReport',
    'ManagedClinicalStartupError',
    'ManagedClinicalStartupReconciler',
    'ManagedSqliteClinicalAttemptInventory',
    'load_authenticated_managed_cleanup',
    'managed_clinical_cleanup_key_id',
    'managed_clinical_ownership_hmac',
    'managed_clinical_startup_config_sha256',
    'reconcile_canonical_managed_runtime_startup',
    'verify_authenticated_managed_cleanup',
]
