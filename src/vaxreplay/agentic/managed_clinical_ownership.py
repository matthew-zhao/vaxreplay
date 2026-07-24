"""Durable run ownership and exact Linux orphan cleanup for managed Lane A workers.

The registry says which attempt is allowed.  This ledger says which host objects that attempt
actually created.  Every state transition is append-only, chained, HMAC-authenticated, create-once,
and fsynced.  The Linux adapter inventories dedicated jail/cgroup namespaces, derives live process
identity from ``cgroup.procs`` plus ``/proc/<pid>/stat``, and refuses loose prefix deletion.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import os
import signal
import stat
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import AtomicDirectoryPublicationError, rename_file_noreplace
from vaxreplay.agentic.clinical_launcher import ClinicalRuntimePrepareRequest, ClinicalRuntimeStart
from vaxreplay.agentic.clinical_production_registry import (
    clinical_production_reservation_sha256,
    clinical_production_start_redemption_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.firecracker import (
    FirecrackerCleanupReceipt,
    FirecrackerPreparedWorker,
    FirecrackerWorkerSpec,
    RunningFirecrackerWorker,
    firecracker_model_sha256,
)
from vaxreplay.agentic.managed_clinical_startup import (
    ManagedClinicalCapability,
    ManagedClinicalHostArtifact,
    ManagedClinicalStartupConfig,
    ManagedClinicalStartupError,
    managed_clinical_cleanup_key_id,
    managed_clinical_ownership_hmac,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

MANAGED_CLINICAL_OWNERSHIP_CONFIG_SCHEMA_VERSION = 'vaxreplay.managed-clinical-ownership-config.dev-v0.2'
MANAGED_CLINICAL_OWNERSHIP_RECORD_SCHEMA_VERSION = 'vaxreplay.managed-clinical-ownership-record.dev-v0.2'
AUTHENTICATED_MANAGED_CLINICAL_OWNERSHIP_SCHEMA_VERSION = 'vaxreplay.authenticated-managed-clinical-ownership.dev-v0.2'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
_HMAC_DOMAIN = b'vaxreplay.authenticated-managed-clinical-ownership.dev-v0.2\x00'
_MAX_RECORD_BYTES = 2 * 1024 * 1024
_MAX_PROC_BYTES = 1024 * 1024
_MAX_VSOCK_NAMESPACE_ENTRIES = 100_000
_MAX_VSOCK_NAMESPACE_DEPTH = 32
_CGROUP_CONTROL_FILE_PREFIXES = (
    'cgroup.',
    'cpu.',
    'cpuset.',
    'hugetlb.',
    'io.',
    'memory.',
    'misc.',
    'pids.',
    'rdma.',
)

type ManagedClinicalOwnershipState = Literal[
    'preparing',
    'prepared',
    'start_bound',
    'running',
    'capability_revoked',
    'cleaned',
]


class ManagedClinicalOwnershipError(ManagedClinicalStartupError):
    """The durable ownership chain or exact Linux inventory could not be proved."""


class ManagedClinicalOwnershipConfig(StrictModel):
    schema_version: Literal['vaxreplay.managed-clinical-ownership-config.dev-v0.2'] = (
        MANAGED_CLINICAL_OWNERSHIP_CONFIG_SCHEMA_VERSION
    )
    ledger_id: str = Field(pattern=_ID_PATTERN)
    ledger_version: str = Field(min_length=1, max_length=200)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    firecracker_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    firecracker_executable_name: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
    ownership_key_id: str = Field(pattern=_SHA256_PATTERN)
    ledger_root: str
    jail_namespace_root: str
    cgroup_namespace_root: str
    proc_root: str = '/proc'
    append_only_create_once_records: Literal[True] = True
    record_and_parent_fsync_required: Literal[True] = True
    exact_path_device_inode_required: Literal[True] = True
    process_start_time_required: Literal[True] = True
    complete_dedicated_namespace_scan_required: Literal[True] = True
    unknown_namespace_entries_fail_closed: Literal[True] = True

    @field_validator(
        'ledger_root',
        'jail_namespace_root',
        'cgroup_namespace_root',
        'proc_root',
    )
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value or value == '/':
            raise ValueError('managed ownership roots must be normalized absolute non-root paths')
        return value

    @model_validator(mode='after')
    def validate_roots(self) -> Self:
        roots = (
            self.ledger_root,
            self.jail_namespace_root,
            self.cgroup_namespace_root,
            self.proc_root,
        )
        if len(set(roots)) != len(roots):
            raise ValueError('managed ownership roots must be distinct')
        return self


class ManagedClinicalOwnershipRecord(StrictModel):
    schema_version: Literal['vaxreplay.managed-clinical-ownership-record.dev-v0.2'] = (
        MANAGED_CLINICAL_OWNERSHIP_RECORD_SCHEMA_VERSION
    )
    ledger_id: str = Field(pattern=_ID_PATTERN)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    sequence: int = Field(ge=0, le=100)
    previous_envelope_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    state: ManagedClinicalOwnershipState
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    launch_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_redemption_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1, max_length=500)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepared_worker_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    run_container_path: str
    jail_root_path: str
    vsock_path: str
    cgroup_path: str
    run_container_device_id: int | None = Field(default=None, ge=0, le=2**63 - 1)
    run_container_inode: int | None = Field(default=None, gt=0, le=2**63 - 1)
    capability_id: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    capability_revoked: bool = False
    process_group_id: int | None = Field(default=None, gt=1, le=2**31 - 1)
    process_group_leader_start_time_ticks: int | None = Field(
        default=None,
        gt=0,
        le=2**63 - 1,
    )
    process_group_session_id: int | None = Field(default=None, gt=0, le=2**31 - 1)
    firecracker_pid: int | None = Field(default=None, gt=1, le=2**31 - 1)
    firecracker_start_time_ticks: int | None = Field(default=None, gt=0, le=2**63 - 1)
    firecracker_executable_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    firecracker_pid_file_path: str | None = None
    firecracker_pid_file_device_id: int | None = Field(default=None, ge=0, le=2**63 - 1)
    firecracker_pid_file_inode: int | None = Field(default=None, gt=0, le=2**63 - 1)
    firecracker_pid_file_owner_uid: int | None = Field(default=None, ge=0, le=2**31 - 1)
    firecracker_pid_file_mode: int | None = Field(default=None, ge=0, le=0o7777)
    cgroup_device_id: int | None = Field(default=None, ge=0, le=2**63 - 1)
    cgroup_inode: int | None = Field(default=None, gt=0, le=2**63 - 1)
    cleanup_receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    terminal_reason: Literal['runtime_cleanup', 'startup_reaper', 'preparation_failed'] | None = None
    recorded_at: datetime

    @field_validator('run_container_path', 'jail_root_path', 'vsock_path', 'cgroup_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed ownership paths must be normalized and absolute')
        return value

    @field_validator('firecracker_pid_file_path')
    @classmethod
    def validate_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed ownership pid-file path must be normalized and absolute')
        return value

    @field_validator('recorded_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('managed ownership time must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_state_fields(self) -> Self:
        if self.sequence == 0 and self.previous_envelope_sha256 is not None:
            raise ValueError('first ownership record cannot name a predecessor')
        if self.sequence > 0 and self.previous_envelope_sha256 is None:
            raise ValueError('later ownership records must name their predecessor')
        prepared = self.prepared_worker_sha256 is not None
        bound = self.start_redemption_sha256 is not None and self.capability_id is not None
        process_fields = (
            self.process_group_id,
            self.process_group_leader_start_time_ticks,
            self.process_group_session_id,
            self.firecracker_pid,
            self.firecracker_start_time_ticks,
            self.firecracker_executable_sha256,
            self.firecracker_pid_file_path,
            self.firecracker_pid_file_device_id,
            self.firecracker_pid_file_inode,
            self.firecracker_pid_file_owner_uid,
            self.firecracker_pid_file_mode,
            self.cgroup_device_id,
            self.cgroup_inode,
        )
        running = all(value is not None for value in process_fields)
        if any(value is not None for value in process_fields) and not running:
            raise ValueError('managed process identity fields must be present together')
        if running and (
            self.process_group_id == self.process_group_session_id or self.firecracker_pid == self.process_group_id
        ):
            raise ValueError('managed process identity must distinguish jailer, session, and child')
        if (self.run_container_device_id is None) != (self.run_container_inode is None):
            raise ValueError('run-container device/inode must be present together')
        if self.state == 'preparing' and (prepared or bound or running):
            raise ValueError('preparing ownership cannot claim later lifecycle state')
        if self.state == 'prepared' and (not prepared or bound or running):
            raise ValueError('prepared ownership must bind only the prepared worker')
        if self.state == 'start_bound' and (not prepared or not bound or running):
            raise ValueError('start-bound ownership must bind preparation and capability')
        if self.state == 'capability_revoked' and (not prepared or not bound):
            raise ValueError('capability-revoked ownership must retain preparation and capability')
        if self.state == 'running' and (not prepared or not bound or not running):
            raise ValueError('running ownership must bind preparation, capability, and process')
        if self.state == 'capability_revoked' and not self.capability_revoked:
            raise ValueError('capability-revoked state must record revocation')
        if self.state == 'cleaned' and (
            self.terminal_reason is None or (self.capability_id is not None and not self.capability_revoked)
        ):
            raise ValueError('cleaned ownership requires terminal reason and capability revocation')
        if self.state != 'cleaned' and (self.cleanup_receipt_sha256 is not None or self.terminal_reason is not None):
            raise ValueError('only cleaned ownership can carry terminal fields')
        return self


class AuthenticatedManagedClinicalOwnership(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-managed-clinical-ownership.dev-v0.2'] = (
        AUTHENTICATED_MANAGED_CLINICAL_OWNERSHIP_SCHEMA_VERSION
    )
    record: ManagedClinicalOwnershipRecord
    ownership_key_id: str = Field(pattern=_SHA256_PATTERN)
    ownership_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


def managed_clinical_ownership_config_sha256(config: ManagedClinicalOwnershipConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def authenticated_managed_clinical_ownership_sha256(
    envelope: AuthenticatedManagedClinicalOwnership,
) -> str:
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def _ownership_envelope_hmac(
    envelope: AuthenticatedManagedClinicalOwnership,
    *,
    key: bytes,
) -> str:
    unsigned = envelope.model_copy(update={'ownership_hmac_sha256': '0' * 64})
    return hmac.new(
        key,
        _HMAC_DOMAIN + canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()


class DurableManagedClinicalOwnershipLedger:
    """Append-only, chained, authenticated ownership state for every consumed run ID."""

    def __init__(
        self,
        *,
        config: ManagedClinicalOwnershipConfig,
        ownership_key: bytes,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if managed_clinical_cleanup_key_id(ownership_key) != config.ownership_key_id:
            raise ValueError('managed ownership key differs from its deployment key ID')
        self.config = config
        self._key = bytes(ownership_key)
        self._clock = clock or (lambda: datetime.now(UTC))
        self.root = _prepare_private_root(Path(config.ledger_root))
        self._lock_path = self.root / '.ledger.lock'
        if not self._lock_path.exists():
            _write_create_once(self._lock_path, b'vaxreplay-managed-ownership-lock\n')

    def begin_preparing(
        self,
        request: ClinicalRuntimePrepareRequest,
        *,
        spec: FirecrackerWorkerSpec,
    ) -> AuthenticatedManagedClinicalOwnership:
        if (
            firecracker_model_sha256(spec) != self.config.worker_spec_sha256
            or spec.runtime.firecracker.sha256 != self.config.firecracker_executable_sha256
            or Path(spec.runtime.firecracker.source_path).name != self.config.firecracker_executable_name
        ):
            raise ManagedClinicalOwnershipError('ownership ledger received a different worker spec')
        reservation = request.reservation
        launch = request.launch
        if (
            reservation.registry_authority_id != self.config.registry_authority_id
            or launch.registry_authority_id != self.config.registry_authority_id
            or launch.reservation_sha256 != clinical_production_reservation_sha256(reservation)
            or reservation.system.worker_spec_sha256 != self.config.worker_spec_sha256
            or launch.episode_id != request.binding.episode_id
            or request.binding not in reservation.tasks
            or launch.run_id == ''
        ):
            raise ManagedClinicalOwnershipError('ownership preparation differs from its registry authority')
        run_container, jail_root, vsock, cgroup = _expected_paths(spec, launch.run_id, self.config)
        record = ManagedClinicalOwnershipRecord(
            ledger_id=self.config.ledger_id,
            registry_authority_id=self.config.registry_authority_id,
            sequence=0,
            state='preparing',
            run_id=launch.run_id,
            reservation_sha256=launch.reservation_sha256,
            launch_sha256=clinical_production_task_launch_sha256(launch),
            episode_id=launch.episode_id,
            worker_spec_sha256=self.config.worker_spec_sha256,
            run_container_path=str(run_container),
            jail_root_path=str(jail_root),
            vsock_path=str(vsock),
            cgroup_path=str(cgroup),
            recorded_at=self._now(),
        )
        with self._locked():
            if self._run_root(launch.run_id).exists():
                raise ManagedClinicalOwnershipError('run ID already has a durable ownership chain')
            return self._append(record)

    def record_prepared(
        self,
        worker: FirecrackerPreparedWorker,
    ) -> AuthenticatedManagedClinicalOwnership:
        with self._locked():
            previous = self._latest(worker.run_id)
            if previous.record.state != 'preparing':
                raise ManagedClinicalOwnershipError('prepared transition is out of order')
            record = previous.record
            if (
                worker.worker_spec_sha256 != record.worker_spec_sha256
                or worker.run_id != record.run_id
                or worker.jail_root != record.jail_root_path
                or worker.vsock_uds_path != record.vsock_path
            ):
                raise ManagedClinicalOwnershipError('prepared worker differs from its ownership intent')
            container = Path(record.run_container_path)
            metadata = _require_path_type(container, directory=True)
            updated = record.model_copy(
                update={
                    'sequence': record.sequence + 1,
                    'previous_envelope_sha256': (authenticated_managed_clinical_ownership_sha256(previous)),
                    'state': 'prepared',
                    'prepared_worker_sha256': firecracker_model_sha256(worker),
                    'run_container_device_id': metadata.st_dev,
                    'run_container_inode': metadata.st_ino,
                    'recorded_at': self._now(),
                }
            )
            return self._append(updated)

    def record_start_bound(
        self,
        *,
        run_id: str,
        start: ClinicalRuntimeStart,
        capability_id: str,
    ) -> AuthenticatedManagedClinicalOwnership:
        with self._locked():
            previous = self._latest(run_id)
            record = previous.record
            redemption = start.start_redemption
            if record.state != 'prepared' or (
                redemption.run_id,
                redemption.reservation_sha256,
                redemption.launch_sha256,
                redemption.episode_id,
            ) != (
                record.run_id,
                record.reservation_sha256,
                record.launch_sha256,
                record.episode_id,
            ):
                raise ManagedClinicalOwnershipError('start redemption differs from ownership intent')
            if (
                redemption.gateway_capability_id != capability_id
                or redemption.prepared_worker_sha256 != record.prepared_worker_sha256
                or start.start_redemption_sha256 != clinical_production_start_redemption_sha256(redemption)
            ):
                raise ManagedClinicalOwnershipError('start redemption differs from owned capability')
            updated = record.model_copy(
                update={
                    'sequence': record.sequence + 1,
                    'previous_envelope_sha256': (authenticated_managed_clinical_ownership_sha256(previous)),
                    'state': 'start_bound',
                    'start_redemption_sha256': (clinical_production_start_redemption_sha256(redemption)),
                    'capability_id': capability_id,
                    'recorded_at': self._now(),
                }
            )
            return self._append(updated)

    def record_running(
        self,
        running: RunningFirecrackerWorker,
    ) -> AuthenticatedManagedClinicalOwnership:
        with self._locked():
            previous = self._latest(running.prepared.run_id)
            record = previous.record
            if record.state != 'start_bound' or (
                running.prepared.run_id != record.run_id
                or firecracker_model_sha256(running.prepared) != record.prepared_worker_sha256
            ):
                raise ManagedClinicalOwnershipError('running worker differs from its start-bound intent')
            expected_pid_file = Path(record.jail_root_path) / f'{self.config.firecracker_executable_name}.pid'
            if (
                running.process.pid <= 1
                or running.jailer_process_group_id != running.process.pid
                or running.jailer_session_id == running.process.pid
                or running.firecracker_pid == running.process.pid
                or running.firecracker_process_group_id != running.jailer_process_group_id
                or running.firecracker_session_id != running.jailer_session_id
                or running.firecracker_start_time_ticks < running.jailer_start_time_ticks
                or running.firecracker_executable_sha256 != self.config.firecracker_executable_sha256
                or Path(running.firecracker_pid_file_path) != expected_pid_file
            ):
                raise ManagedClinicalOwnershipError(
                    'managed running worker differs from its stored jailer/Firecracker identity'
                )
            pid_file = _require_path_type(expected_pid_file, directory=False)
            cgroup = _require_path_type(Path(record.cgroup_path), directory=True)
            try:
                opened_cgroup = os.fstat(running.cgroup_descriptor)
            except OSError:
                raise ManagedClinicalOwnershipError(
                    'managed running worker lost its pinned cgroup descriptor'
                ) from None
            if (
                (pid_file.st_dev, pid_file.st_ino)
                != (
                    running.firecracker_pid_file_device_id,
                    running.firecracker_pid_file_inode,
                )
                or _read_pid_file(
                    expected_pid_file,
                    expected_device=running.firecracker_pid_file_device_id,
                    expected_inode=running.firecracker_pid_file_inode,
                    expected_owner_uid=pid_file.st_uid,
                    expected_mode=stat.S_IMODE(pid_file.st_mode),
                )
                != running.firecracker_pid
                or (cgroup.st_dev, cgroup.st_ino) != (running.cgroup_device_id, running.cgroup_inode)
                or (opened_cgroup.st_dev, opened_cgroup.st_ino) != (running.cgroup_device_id, running.cgroup_inode)
                or not stat.S_ISDIR(opened_cgroup.st_mode)
            ):
                raise ManagedClinicalOwnershipError('managed Firecracker pid file differs from its launch identity')
            updated = record.model_copy(
                update={
                    'sequence': record.sequence + 1,
                    'previous_envelope_sha256': (authenticated_managed_clinical_ownership_sha256(previous)),
                    'state': 'running',
                    'process_group_id': running.jailer_process_group_id,
                    'process_group_leader_start_time_ticks': running.jailer_start_time_ticks,
                    'process_group_session_id': running.jailer_session_id,
                    'firecracker_pid': running.firecracker_pid,
                    'firecracker_start_time_ticks': running.firecracker_start_time_ticks,
                    'firecracker_executable_sha256': (running.firecracker_executable_sha256),
                    'firecracker_pid_file_path': running.firecracker_pid_file_path,
                    'firecracker_pid_file_device_id': (running.firecracker_pid_file_device_id),
                    'firecracker_pid_file_inode': running.firecracker_pid_file_inode,
                    'firecracker_pid_file_owner_uid': pid_file.st_uid,
                    'firecracker_pid_file_mode': stat.S_IMODE(pid_file.st_mode),
                    'cgroup_device_id': running.cgroup_device_id,
                    'cgroup_inode': running.cgroup_inode,
                    'recorded_at': self._now(),
                }
            )
            return self._append(updated)

    def record_capability_revoked(
        self,
        *,
        run_id: str,
        capability_id: str,
    ) -> AuthenticatedManagedClinicalOwnership:
        with self._locked():
            previous = self._latest(run_id)
            record = previous.record
            if record.state not in {'start_bound', 'running'} or record.capability_id != capability_id:
                raise ManagedClinicalOwnershipError('capability revocation differs from active ownership')
            updated = record.model_copy(
                update={
                    'sequence': record.sequence + 1,
                    'previous_envelope_sha256': (authenticated_managed_clinical_ownership_sha256(previous)),
                    'state': 'capability_revoked',
                    'capability_revoked': True,
                    'recorded_at': self._now(),
                }
            )
            return self._append(updated)

    def record_cleaned(
        self,
        *,
        run_id: str,
        terminal_reason: Literal['runtime_cleanup', 'startup_reaper', 'preparation_failed'],
        cleanup_receipt: FirecrackerCleanupReceipt | None = None,
    ) -> AuthenticatedManagedClinicalOwnership:
        with self._locked():
            previous = self._latest(run_id)
            record = previous.record
            if record.state == 'cleaned':
                raise ManagedClinicalOwnershipError('ownership run is already terminal')
            if record.capability_id is not None and not record.capability_revoked:
                raise ManagedClinicalOwnershipError('owned capability must be revoked before cleanup')
            self._require_owned_state_absent(record)
            cleanup_sha256 = None
            if cleanup_receipt is not None:
                if cleanup_receipt.run_id != run_id:
                    raise ManagedClinicalOwnershipError('cleanup receipt belongs to a different run')
                cleanup_sha256 = firecracker_model_sha256(cleanup_receipt)
            updated = record.model_copy(
                update={
                    'sequence': record.sequence + 1,
                    'previous_envelope_sha256': (authenticated_managed_clinical_ownership_sha256(previous)),
                    'state': 'cleaned',
                    'cleanup_receipt_sha256': cleanup_sha256,
                    'terminal_reason': terminal_reason,
                    'recorded_at': self._now(),
                }
            )
            return self._append(updated)

    def active(self) -> tuple[AuthenticatedManagedClinicalOwnership, ...]:
        with self._locked():
            values = []
            for run_id in self._run_ids():
                latest = self._latest(run_id)
                if latest.record.state != 'cleaned':
                    values.append(latest)
            return tuple(values)

    def run_ids(self) -> tuple[str, ...]:
        """Return the canonical durable run-directory inventory after crash recovery."""

        with self._locked():
            return self._run_ids()

    def latest(self, run_id: str) -> AuthenticatedManagedClinicalOwnership:
        with self._locked():
            return self._latest(run_id)

    def chain(
        self,
        run_id: str,
    ) -> tuple[AuthenticatedManagedClinicalOwnership, ...]:
        """Reload one complete canonical HMAC/predecessor/transition chain.

        This is intentionally read-only.  It exposes the same fully verified sequence used by
        ``latest`` so an independent collector can compare every persisted envelope, rather than
        trusting a caller-selected lifecycle summary.
        """

        with self._locked():
            return self._load_chain(run_id)

    def _append(self, record: ManagedClinicalOwnershipRecord) -> AuthenticatedManagedClinicalOwnership:
        run_root = self._run_root(record.run_id)
        if record.sequence == 0:
            try:
                run_root.mkdir(mode=0o700)
            except OSError:
                raise ManagedClinicalOwnershipError('ownership run directory could not be created') from None
            _fsync_directory(self.root)
        else:
            _require_private_directory(run_root)
        unsigned = AuthenticatedManagedClinicalOwnership(
            record=record,
            ownership_key_id=self.config.ownership_key_id,
            ownership_hmac_sha256='0' * 64,
        )
        envelope = unsigned.model_copy(
            update={'ownership_hmac_sha256': _ownership_envelope_hmac(unsigned, key=self._key)}
        )
        path = run_root / f'{record.sequence:04d}.json'
        _write_create_once(path, canonical_json_bytes(envelope))
        loaded = self._load_envelope(path)
        if loaded != envelope:
            raise ManagedClinicalOwnershipError('persisted ownership record differs from signed bytes')
        return loaded

    def _latest(self, run_id: str) -> AuthenticatedManagedClinicalOwnership:
        chain = self._load_chain(run_id)
        if not chain:
            raise ManagedClinicalOwnershipError('ownership run has no records')
        return chain[-1]

    def _load_chain(self, run_id: str) -> tuple[AuthenticatedManagedClinicalOwnership, ...]:
        root = self._run_root(run_id)
        _require_private_directory(root)
        _recover_pending_record_publications(root)
        paths = tuple(sorted(root.iterdir(), key=lambda item: item.name))
        if not paths or any(path.name != f'{index:04d}.json' for index, path in enumerate(paths)):
            raise ManagedClinicalOwnershipError('ownership chain inventory is incomplete or non-canonical')
        values = tuple(self._load_envelope(path) for path in paths)
        previous_sha256: str | None = None
        previous_state: ManagedClinicalOwnershipState | None = None
        previous_recorded_at: datetime | None = None
        previous_record: ManagedClinicalOwnershipRecord | None = None
        allowed = {
            None: {'preparing'},
            'preparing': {'prepared', 'cleaned'},
            'prepared': {'start_bound', 'cleaned'},
            'start_bound': {'running', 'capability_revoked'},
            'running': {'capability_revoked'},
            'capability_revoked': {'cleaned'},
            'cleaned': set(),
        }
        first = values[0].record
        immutable = (
            first.ledger_id,
            first.registry_authority_id,
            first.run_id,
            first.reservation_sha256,
            first.launch_sha256,
            first.episode_id,
            first.worker_spec_sha256,
            first.run_container_path,
            first.jail_root_path,
            first.vsock_path,
            first.cgroup_path,
        )
        for sequence, envelope in enumerate(values):
            record = envelope.record
            self._require_record_matches_config(record, requested_run_id=run_id)
            if (
                record.sequence != sequence
                or record.previous_envelope_sha256 != previous_sha256
                or record.state not in allowed[previous_state]
                or (previous_recorded_at is not None and record.recorded_at < previous_recorded_at)
                or immutable
                != (
                    record.ledger_id,
                    record.registry_authority_id,
                    record.run_id,
                    record.reservation_sha256,
                    record.launch_sha256,
                    record.episode_id,
                    record.worker_spec_sha256,
                    record.run_container_path,
                    record.jail_root_path,
                    record.vsock_path,
                    record.cgroup_path,
                )
                or (previous_record is not None and _ownership_binding_regressed(previous_record, record))
            ):
                raise ManagedClinicalOwnershipError('ownership chain transition or binding is invalid')
            previous_sha256 = authenticated_managed_clinical_ownership_sha256(envelope)
            previous_state = record.state
            previous_recorded_at = record.recorded_at
            previous_record = record
        return values

    def _require_record_matches_config(
        self,
        record: ManagedClinicalOwnershipRecord,
        *,
        requested_run_id: str,
    ) -> None:
        run_container = Path(self.config.jail_namespace_root) / requested_run_id
        jail_root = run_container / 'root'
        expected = (
            requested_run_id,
            self.config.ledger_id,
            self.config.registry_authority_id,
            self.config.worker_spec_sha256,
            str(run_container),
            str(jail_root),
            str(jail_root / 'run' / 'vsock.sock'),
            str(Path(self.config.cgroup_namespace_root) / requested_run_id),
        )
        observed = (
            record.run_id,
            record.ledger_id,
            record.registry_authority_id,
            record.worker_spec_sha256,
            record.run_container_path,
            record.jail_root_path,
            record.vsock_path,
            record.cgroup_path,
        )
        if (
            observed != expected
            or (
                record.firecracker_executable_sha256 is not None
                and record.firecracker_executable_sha256 != self.config.firecracker_executable_sha256
            )
            or (
                record.firecracker_pid_file_path is not None
                and record.firecracker_pid_file_path
                != str(jail_root / f'{self.config.firecracker_executable_name}.pid')
            )
        ):
            raise ManagedClinicalOwnershipError(
                'ownership record differs from its requested run or configured namespace'
            )

    def _load_envelope(self, path: Path) -> AuthenticatedManagedClinicalOwnership:
        content = _read_stable_private_file(path)
        try:
            envelope = AuthenticatedManagedClinicalOwnership.model_validate_json(content)
        except ValueError:
            raise ManagedClinicalOwnershipError('ownership record has an invalid strict schema') from None
        if canonical_json_bytes(envelope) != content:
            raise ManagedClinicalOwnershipError('ownership record must use exact canonical JSON')
        if envelope.ownership_key_id != self.config.ownership_key_id or not hmac.compare_digest(
            envelope.ownership_hmac_sha256,
            _ownership_envelope_hmac(envelope, key=self._key),
        ):
            raise ManagedClinicalOwnershipError('ownership record authentication failed')
        return envelope

    def _run_ids(self) -> tuple[str, ...]:
        names = []
        for entry in self.root.iterdir():
            if entry.name == '.ledger.lock':
                continue
            if (
                len(entry.name) != 32
                or any(character not in '0123456789abcdef' for character in entry.name)
                or entry.is_symlink()
                or not entry.is_dir()
            ):
                raise ManagedClinicalOwnershipError('ownership ledger contains an unexpected entry')
            _recover_pending_record_publications(entry)
            if not any(entry.iterdir()):
                try:
                    entry.rmdir()
                except OSError:
                    raise ManagedClinicalOwnershipError(
                        'empty interrupted ownership run could not be recovered'
                    ) from None
                _fsync_directory(self.root)
                continue
            names.append(entry.name)
        return tuple(sorted(names))

    def _run_root(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(character not in '0123456789abcdef' for character in run_id):
            raise ManagedClinicalOwnershipError('ownership run ID is invalid')
        return self.root / run_id

    def _require_owned_state_absent(self, record: ManagedClinicalOwnershipRecord) -> None:
        for value in (record.run_container_path, record.cgroup_path, record.vsock_path):
            try:
                Path(value).lstat()
            except FileNotFoundError:
                continue
            except OSError:
                raise ManagedClinicalOwnershipError('owned path could not be inspected') from None
            raise ManagedClinicalOwnershipError('owned path remains present during terminalization')
        if record.firecracker_pid is not None:
            observed = _optional_process_identity(
                record.firecracker_pid,
                proc_root=Path(self.config.proc_root),
            )
            if observed is not None and observed.start_time_ticks == record.firecracker_start_time_ticks:
                raise ManagedClinicalOwnershipError('owned Firecracker child remains live during terminalization')

    @contextmanager
    def _locked(self):
        descriptor = os.open(
            self._lock_path,
            os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ManagedClinicalOwnershipError('ownership clock must return an aware time')
        return value.astimezone(UTC)


class LinuxProcessIdentity(StrictModel):
    pid: int = Field(gt=1, le=2**31 - 1)
    process_group_id: int = Field(gt=1, le=2**31 - 1)
    session_id: int = Field(gt=1, le=2**31 - 1)
    start_time_ticks: int = Field(gt=0, le=2**63 - 1)
    process_state: str = Field(min_length=1, max_length=1)


class ManagedProcessGroupIdentity(StrictModel):
    """Truthful group binding plus one actually observed live-process witness."""

    process_group_id: int = Field(gt=1, le=2**31 - 1)
    session_id: int = Field(gt=1, le=2**31 - 1)
    identity_source: Literal['durable-jailer-group', 'recovered-firecracker-child']
    process_group_leader_start_time_ticks: int | None = Field(
        default=None,
        gt=0,
        le=2**63 - 1,
    )
    witness_pid: int = Field(gt=1, le=2**31 - 1)
    witness_start_time_ticks: int = Field(gt=0, le=2**63 - 1)

    @model_validator(mode='after')
    def validate_source(self) -> Self:
        if (self.identity_source == 'durable-jailer-group') != (self.process_group_leader_start_time_ticks is not None):
            raise ValueError('only a durable jailer record may claim a group-leader start counter')
        return self


def _same_linux_process_kernel_identity(
    first: LinuxProcessIdentity,
    second: LinuxProcessIdentity,
) -> bool:
    return (
        first.pid,
        first.process_group_id,
        first.session_id,
        first.start_time_ticks,
    ) == (
        second.pid,
        second.process_group_id,
        second.session_id,
        second.start_time_ticks,
    )


def _same_managed_process_group(
    expected: ManagedProcessGroupIdentity,
    observed: ManagedProcessGroupIdentity,
) -> bool:
    if (
        expected.process_group_id,
        expected.session_id,
        expected.identity_source,
        expected.process_group_leader_start_time_ticks,
    ) != (
        observed.process_group_id,
        observed.session_id,
        observed.identity_source,
        observed.process_group_leader_start_time_ticks,
    ):
        return False
    if expected.identity_source == 'recovered-firecracker-child':
        return (
            expected.witness_pid,
            expected.witness_start_time_ticks,
        ) == (
            observed.witness_pid,
            observed.witness_start_time_ticks,
        )
    return True


def read_linux_process_identity(pid: int, *, proc_root: Path = Path('/proc')) -> LinuxProcessIdentity:
    try:
        content = _read_stable_nofollow_file(
            proc_root / str(pid) / 'stat',
            maximum_bytes=_MAX_PROC_BYTES,
            label='owned process stat',
        )
    except OSError:
        raise ManagedClinicalOwnershipError('owned process stat is unavailable') from None
    if not content or len(content) > _MAX_PROC_BYTES or b'\x00' in content:
        raise ManagedClinicalOwnershipError('owned process stat has invalid size or encoding')
    close = content.rfind(b')')
    if close <= 0:
        raise ManagedClinicalOwnershipError('owned process stat has invalid framing')
    fields = content[close + 2 :].split()
    if len(fields) < 20:
        raise ManagedClinicalOwnershipError('owned process stat is incomplete')
    try:
        observed_pid = int(content[: content.find(b' ')])
        process_state = fields[0].decode('ascii')
        process_group_id = int(fields[2])
        session_id = int(fields[3])
        start_time_ticks = int(fields[19])
    except (UnicodeDecodeError, ValueError):
        raise ManagedClinicalOwnershipError('owned process stat contains invalid integers') from None
    if observed_pid != pid:
        raise ManagedClinicalOwnershipError('owned process stat names a different PID')
    return LinuxProcessIdentity(
        pid=pid,
        process_group_id=process_group_id,
        session_id=session_id,
        start_time_ticks=start_time_ticks,
        process_state=process_state,
    )


class LinuxManagedClinicalHostAdapter:
    """Complete scanner and exact destructive adapter for dedicated Linux namespaces."""

    def __init__(
        self,
        *,
        config: ManagedClinicalStartupConfig,
        ownership: DurableManagedClinicalOwnershipLedger,
        ownership_key: bytes,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if managed_clinical_cleanup_key_id(ownership_key) != config.cleanup_receipt_key_id:
            raise ValueError('Linux adapter ownership key differs from startup cleanup key')
        if ownership.config.registry_authority_id != config.registry_authority_id:
            raise ValueError('Linux adapter ownership ledger belongs to a different authority')
        if (
            ownership.config.jail_namespace_root != config.jail_root
            or ownership.config.cgroup_namespace_root != config.cgroup_root
            or ownership.config.jail_namespace_root != config.vsock_root
        ):
            raise ValueError('Linux adapter roots differ from startup reconciliation roots')
        self.config = config
        self.ownership = ownership
        self._key = bytes(ownership_key)
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep

    def owned_run_ids(self) -> tuple[str, ...]:
        return tuple(item.record.run_id for item in self.ownership.active())

    def scan_process_groups(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        values = []
        for envelope in self.ownership.active():
            record = envelope.record
            cgroup_path = Path(record.cgroup_path)
            try:
                cgroup_metadata = cgroup_path.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                raise ManagedClinicalOwnershipError('owned cgroup is unavailable') from None
            if stat.S_ISLNK(cgroup_metadata.st_mode) or not stat.S_ISDIR(cgroup_metadata.st_mode):
                raise ManagedClinicalOwnershipError('owned cgroup changed type')
            identity = self._live_process_identity(
                record,
                expected_cgroup_device=cgroup_metadata.st_dev,
                expected_cgroup_inode=cgroup_metadata.st_ino,
            )
            if identity is None:
                continue
            values.append(
                self._host_artifact(
                    envelope,
                    kind='process_group',
                    artifact_id=f'pgid:{identity.process_group_id}',
                    process_identity=identity,
                    process_cgroup_metadata=cgroup_metadata,
                )
            )
        return tuple(values)

    def scan_cgroups(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        return self._scan_namespace('cgroup')

    def scan_jail_roots(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        return self._scan_namespace('jail_root')

    def scan_vsock_endpoints(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        active = {item.record.run_id: item for item in self.ownership.active()}
        root = Path(self.config.vsock_root)
        root_metadata = _require_path_type(root, directory=True)
        discovered = _scan_complete_vsock_namespace(
            root,
            active_run_containers={
                run_id: Path(envelope.record.run_container_path) for run_id, envelope in active.items()
            },
        )
        expected_by_path = {Path(envelope.record.vsock_path): envelope for envelope in active.values()}
        discovered_paths = {path for path, _metadata in discovered}
        for expected in expected_by_path:
            try:
                metadata = expected.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                raise ManagedClinicalOwnershipError(
                    'owned vsock path is unavailable during complete inventory'
                ) from None
            if not stat.S_ISSOCK(metadata.st_mode) or expected not in discovered_paths:
                raise ManagedClinicalOwnershipError('owned vsock path changed type or identity')
        unknown = tuple(path for path, _metadata in discovered if path not in expected_by_path)
        if unknown:
            raise ManagedClinicalOwnershipError('managed vsock namespace contains an unrepresentable socket')
        values = []
        for path, metadata in discovered:
            envelope = expected_by_path[path]
            current = path.lstat()
            if not stat.S_ISSOCK(current.st_mode) or (current.st_dev, current.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise ManagedClinicalOwnershipError('owned vsock changed identity during complete inventory')
            values.append(
                self._host_artifact(
                    envelope,
                    kind='vsock_endpoint',
                    artifact_id=str(path),
                    path_metadata=metadata,
                )
            )
        final_root = root.lstat()
        if (
            (final_root.st_dev, final_root.st_ino) != (root_metadata.st_dev, root_metadata.st_ino)
            or stat.S_ISLNK(final_root.st_mode)
            or not stat.S_ISDIR(final_root.st_mode)
        ):
            raise ManagedClinicalOwnershipError('managed vsock namespace changed during complete inventory')
        return tuple(sorted(values, key=lambda item: (item.run_id, item.artifact_id)))

    def terminate_process_group(
        self,
        artifact: ManagedClinicalHostArtifact,
        *,
        grace_seconds: float,
    ) -> None:
        identity, record = self._require_process_artifact(artifact)
        self._signal_exact_firecracker_child(record, artifact, signal.SIGTERM)
        deadline = self._monotonic() + grace_seconds
        while self._owned_process_group_alive(identity, record, artifact) and self._monotonic() < deadline:
            self._sleep(0.02)
        if self._owned_process_group_alive(identity, record, artifact):
            self._kill_exact_owned_cgroup(record, artifact)
            deadline = self._monotonic() + grace_seconds
            while self._owned_process_group_alive(identity, record, artifact) and self._monotonic() < deadline:
                self._sleep(0.02)
        if self._owned_process_group_alive(identity, record, artifact):
            raise ManagedClinicalOwnershipError('owned cgroup survived cgroup.kill')

    def reap_process_group(self, artifact: ManagedClinicalHostArtifact) -> None:
        identity, record = self._require_process_artifact(artifact, allow_exited_leader=True)
        if self._owned_process_group_alive(identity, record, artifact):
            raise ManagedClinicalOwnershipError('owned process group remains live after termination')

    def remove_vsock_endpoint(self, artifact: ManagedClinicalHostArtifact) -> None:
        self._require_owned_artifact(artifact, expected_kind='vsock_endpoint')
        _unlink_exact(
            Path(artifact.artifact_id),
            expected_device=artifact.path_device_id,
            expected_inode=artifact.path_inode,
            expected_socket=True,
        )

    def remove_cgroup(self, artifact: ManagedClinicalHostArtifact) -> None:
        self._require_owned_artifact(artifact, expected_kind='cgroup')
        if artifact.path_device_id is None or artifact.path_inode is None:
            raise ManagedClinicalOwnershipError('owned cgroup lacks an exact inode binding')
        path = Path(artifact.artifact_id)
        try:
            if _read_cgroup_process_ids(
                path / 'cgroup.procs',
                expected_parent_device=artifact.path_device_id,
                expected_parent_inode=artifact.path_inode,
            ):
                raise ManagedClinicalOwnershipError('owned cgroup remains nonempty')
        except FileNotFoundError:
            pass
        _rmdir_exact(
            path,
            expected_device=artifact.path_device_id,
            expected_inode=artifact.path_inode,
        )

    def remove_jail_root(self, artifact: ManagedClinicalHostArtifact) -> None:
        self._require_owned_artifact(artifact, expected_kind='jail_root')
        path = Path(artifact.artifact_id)
        _remove_tree_exact_fd(
            path,
            expected_device=artifact.path_device_id,
            expected_inode=artifact.path_inode,
        )

    def finalize_reconciled_run(self, run_id: str) -> None:
        latest = self.ownership.latest(run_id).record
        if latest.capability_id is not None and not latest.capability_revoked:
            raise ManagedClinicalOwnershipError('startup cannot finalize an unrevoked capability')
        self.ownership.record_cleaned(run_id=run_id, terminal_reason='startup_reaper')

    def _scan_namespace(
        self,
        kind: Literal['cgroup', 'jail_root'],
    ) -> tuple[ManagedClinicalHostArtifact, ...]:
        active = {item.record.run_id: item for item in self.ownership.active()}
        root = Path(self.config.cgroup_root if kind == 'cgroup' else self.config.jail_root)
        _require_path_type(root, directory=True)
        try:
            with os.scandir(root) as scanner:
                entries = tuple(sorted(scanner, key=lambda entry: entry.name))
        except OSError:
            raise ManagedClinicalOwnershipError('managed namespace inventory is unavailable') from None
        values = []
        for entry in entries:
            if kind == 'cgroup' and not entry.is_dir(follow_symlinks=False):
                if entry.is_symlink():
                    raise ManagedClinicalOwnershipError('managed cgroup namespace contains an unexpected symlink')
                # A cgroup v2 directory necessarily contains kernel-created control files.  Refuse
                # arbitrary files or special objects rather than treating every non-directory as
                # kernel metadata.
                if not entry.is_file(follow_symlinks=False) or not entry.name.startswith(_CGROUP_CONTROL_FILE_PREFIXES):
                    raise ManagedClinicalOwnershipError('managed cgroup namespace contains an unexpected entry')
                continue
            envelope = active.get(entry.name)
            if envelope is None:
                raise ManagedClinicalOwnershipError('managed namespace contains an unowned entry')
            expected = Path(envelope.record.cgroup_path if kind == 'cgroup' else envelope.record.run_container_path)
            if Path(entry.path) != expected or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise ManagedClinicalOwnershipError('managed namespace entry differs from ownership')
            metadata = entry.stat(follow_symlinks=False)
            values.append(
                self._host_artifact(
                    envelope,
                    kind=kind,
                    artifact_id=str(expected),
                    path_metadata=metadata,
                )
            )
        return tuple(values)

    def _live_process_identity(
        self,
        record: ManagedClinicalOwnershipRecord,
        *,
        expected_cgroup_device: int,
        expected_cgroup_inode: int,
    ) -> ManagedProcessGroupIdentity | None:
        cgroup = Path(record.cgroup_path)
        if record.cgroup_device_id is not None and (
            record.cgroup_inode is None
            or (expected_cgroup_device, expected_cgroup_inode) != (record.cgroup_device_id, record.cgroup_inode)
        ):
            raise ManagedClinicalOwnershipError('owned cgroup differs from its durable launch identity')
        try:
            pids = _read_cgroup_process_ids(
                cgroup / 'cgroup.procs',
                expected_parent_device=expected_cgroup_device,
                expected_parent_inode=expected_cgroup_inode,
            )
        except FileNotFoundError:
            return None
        except OSError:
            raise ManagedClinicalOwnershipError('owned cgroup process inventory is invalid') from None
        if not pids:
            if record.firecracker_pid is not None:
                child = _optional_process_identity(
                    record.firecracker_pid,
                    proc_root=Path(self.ownership.config.proc_root),
                )
                if child is not None and child.start_time_ticks == record.firecracker_start_time_ticks:
                    raise ManagedClinicalOwnershipError('owned Firecracker child is missing from its cgroup inventory')
            return None
        identities = tuple(
            read_linux_process_identity(pid, proc_root=Path(self.ownership.config.proc_root)) for pid in pids
        )
        if record.process_group_id is None:
            pid_file = Path(record.jail_root_path) / f'{self.ownership.config.firecracker_executable_name}.pid'
            metadata = _require_path_type(pid_file, directory=False)
            if metadata.st_uid != 0 or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600:
                raise ManagedClinicalOwnershipError(
                    'recoverable Firecracker pid file lacks the jailer root:0600 identity'
                )
            firecracker_pid = _read_pid_file(
                pid_file,
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
                expected_owner_uid=0,
                expected_mode=0o600,
            )
            child = next((item for item in identities if item.pid == firecracker_pid), None)
            executable_sha256 = (
                None
                if child is None
                else _proc_executable_sha256_or_exact_zombie(
                    child,
                    proc_root=Path(self.ownership.config.proc_root),
                    expected_sha256=(self.ownership.config.firecracker_executable_sha256),
                )
            )
            if (
                child is None
                or child.process_group_id <= 1
                or child.process_group_id == child.pid
                or child.session_id == child.process_group_id
                or any(
                    item.process_group_id != child.process_group_id or item.session_id != child.session_id
                    for item in identities
                )
                or executable_sha256 != self.ownership.config.firecracker_executable_sha256
            ):
                raise ManagedClinicalOwnershipError('live cgroup lacks an authentic recoverable Firecracker child')
            rebound_pids = _read_cgroup_process_ids(
                cgroup / 'cgroup.procs',
                expected_parent_device=expected_cgroup_device,
                expected_parent_inode=expected_cgroup_inode,
            )
            if not rebound_pids:
                if (
                    _optional_process_identity(
                        firecracker_pid,
                        proc_root=Path(self.ownership.config.proc_root),
                    )
                    is not None
                ):
                    raise ManagedClinicalOwnershipError(
                        'recoverable Firecracker child changed while its identity was pinned'
                    )
                return None
            rebound_child = _optional_process_identity(
                firecracker_pid,
                proc_root=Path(self.ownership.config.proc_root),
            )
            if rebound_child is None:
                final_pids = _read_cgroup_process_ids(
                    cgroup / 'cgroup.procs',
                    expected_parent_device=expected_cgroup_device,
                    expected_parent_inode=expected_cgroup_inode,
                )
                if final_pids:
                    raise ManagedClinicalOwnershipError('recoverable Firecracker child vanished from a nonempty cgroup')
                return None
            rebound_metadata = _require_path_type(pid_file, directory=False)
            rebound_pid = _read_pid_file(
                pid_file,
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
                expected_owner_uid=0,
                expected_mode=0o600,
            )
            rebound_executable_sha256 = _proc_executable_sha256_or_exact_zombie(
                rebound_child,
                proc_root=Path(self.ownership.config.proc_root),
                expected_sha256=self.ownership.config.firecracker_executable_sha256,
            )
            if (
                rebound_pids != pids
                or not _same_linux_process_kernel_identity(rebound_child, child)
                or rebound_pid != firecracker_pid
                or (
                    rebound_metadata.st_dev,
                    rebound_metadata.st_ino,
                    rebound_metadata.st_uid,
                    rebound_metadata.st_nlink,
                    stat.S_IMODE(rebound_metadata.st_mode),
                )
                != (metadata.st_dev, metadata.st_ino, 0, 1, 0o600)
                or rebound_executable_sha256 != executable_sha256
            ):
                raise ManagedClinicalOwnershipError(
                    'recoverable Firecracker child changed while its identity was pinned'
                )
            # The start-bound record and fixed paths existed before launch.  Exact cgroup
            # membership, pid-file identity, child start counter, session/group, and executable
            # digest safely recover a crash between process creation and record_running().
            del metadata
            return ManagedProcessGroupIdentity(
                process_group_id=child.process_group_id,
                session_id=child.session_id,
                identity_source='recovered-firecracker-child',
                witness_pid=child.pid,
                witness_start_time_ticks=child.start_time_ticks,
            )
        if (
            record.process_group_leader_start_time_ticks is None
            or record.process_group_session_id is None
            or record.firecracker_pid is None
            or record.firecracker_start_time_ticks is None
            or record.firecracker_executable_sha256 is None
            or record.firecracker_pid_file_path is None
            or record.firecracker_pid_file_device_id is None
            or record.firecracker_pid_file_inode is None
            or record.firecracker_pid_file_owner_uid is None
            or record.firecracker_pid_file_mode is None
        ):
            raise ManagedClinicalOwnershipError('durable Firecracker child identity is incomplete')
        child = next((item for item in identities if item.pid == record.firecracker_pid), None)
        if any(
            item.process_group_id != record.process_group_id or item.session_id != record.process_group_session_id
            for item in identities
        ) or (child is not None and child.start_time_ticks != record.firecracker_start_time_ticks):
            raise ManagedClinicalOwnershipError('owned process identity differs from durable record')
        pid_file = Path(record.firecracker_pid_file_path)
        metadata = _require_path_type(pid_file, directory=False)
        if (
            (metadata.st_dev, metadata.st_ino)
            != (record.firecracker_pid_file_device_id, record.firecracker_pid_file_inode)
            or _read_pid_file(
                pid_file,
                expected_device=record.firecracker_pid_file_device_id,
                expected_inode=record.firecracker_pid_file_inode,
                expected_owner_uid=record.firecracker_pid_file_owner_uid,
                expected_mode=record.firecracker_pid_file_mode,
            )
            != record.firecracker_pid
            or (
                child is not None
                and _proc_executable_sha256_or_exact_zombie(
                    child,
                    proc_root=Path(self.ownership.config.proc_root),
                    expected_sha256=record.firecracker_executable_sha256,
                )
                != record.firecracker_executable_sha256
            )
        ):
            raise ManagedClinicalOwnershipError('owned Firecracker child executable or pid file changed identity')
        witness = child or identities[0]
        return ManagedProcessGroupIdentity(
            process_group_id=record.process_group_id,
            session_id=record.process_group_session_id,
            identity_source='durable-jailer-group',
            process_group_leader_start_time_ticks=(record.process_group_leader_start_time_ticks),
            witness_pid=witness.pid,
            witness_start_time_ticks=witness.start_time_ticks,
        )

    def _host_artifact(
        self,
        envelope: AuthenticatedManagedClinicalOwnership,
        *,
        kind: Literal['process_group', 'cgroup', 'jail_root', 'vsock_endpoint'],
        artifact_id: str,
        process_identity: ManagedProcessGroupIdentity | None = None,
        path_metadata: os.stat_result | None = None,
        process_cgroup_metadata: os.stat_result | None = None,
    ) -> ManagedClinicalHostArtifact:
        record = envelope.record
        unsigned = ManagedClinicalHostArtifact(
            artifact_kind=kind,
            artifact_id=artifact_id,
            run_id=record.run_id,
            registry_authority_id=record.registry_authority_id,
            reservation_sha256=record.reservation_sha256,
            launch_sha256=record.launch_sha256,
            start_redemption_sha256=record.start_redemption_sha256,
            worker_spec_sha256=record.worker_spec_sha256,
            ownership_record_sha256=authenticated_managed_clinical_ownership_sha256(envelope),
            ownership_authentication_hmac_sha256='0' * 64,
            process_group_leader_start_time_ticks=(
                None if process_identity is None else process_identity.process_group_leader_start_time_ticks
            ),
            process_group_session_id=(None if process_identity is None else process_identity.session_id),
            process_identity_source=(None if process_identity is None else process_identity.identity_source),
            process_witness_pid=(None if process_identity is None else process_identity.witness_pid),
            process_witness_start_time_ticks=(
                None if process_identity is None else process_identity.witness_start_time_ticks
            ),
            path_device_id=None if path_metadata is None else path_metadata.st_dev,
            path_inode=None if path_metadata is None else path_metadata.st_ino,
            process_cgroup_device_id=(None if process_cgroup_metadata is None else process_cgroup_metadata.st_dev),
            process_cgroup_inode=(None if process_cgroup_metadata is None else process_cgroup_metadata.st_ino),
        )
        return unsigned.model_copy(
            update={
                'ownership_authentication_hmac_sha256': managed_clinical_ownership_hmac(
                    unsigned,
                    key=self._key,
                )
            }
        )

    def _require_owned_artifact(
        self,
        artifact: ManagedClinicalHostArtifact,
        *,
        expected_kind: Literal['cgroup', 'jail_root', 'vsock_endpoint'],
    ) -> AuthenticatedManagedClinicalOwnership:
        if artifact.artifact_kind != expected_kind or not hmac.compare_digest(
            artifact.ownership_authentication_hmac_sha256,
            managed_clinical_ownership_hmac(artifact, key=self._key),
        ):
            raise ManagedClinicalOwnershipError('destructive operation received unowned artifact')
        latest = self.ownership.latest(artifact.run_id)
        artifact_record_sha256 = artifact.ownership_record_sha256
        latest_sha256 = authenticated_managed_clinical_ownership_sha256(latest)
        exact_revocation_successor = (
            latest.record.state == 'capability_revoked'
            and latest.record.previous_envelope_sha256 == artifact_record_sha256
        )
        if not hmac.compare_digest(artifact_record_sha256, latest_sha256) and not (exact_revocation_successor):
            raise ManagedClinicalOwnershipError('destructive operation received stale ownership')
        if not _artifact_matches_ownership_record(artifact, latest.record):
            raise ManagedClinicalOwnershipError('destructive operation ownership binding changed')
        return latest

    def _artifact_process_identity(
        self,
        artifact: ManagedClinicalHostArtifact,
        record: ManagedClinicalOwnershipRecord,
    ) -> ManagedProcessGroupIdentity:
        del record
        if artifact.artifact_kind != 'process_group':
            raise ManagedClinicalOwnershipError('process operation received a non-process artifact')
        process_group = int(artifact.artifact_id.removeprefix('pgid:'))
        if (
            artifact.process_group_session_id is None
            or artifact.process_identity_source is None
            or artifact.process_witness_pid is None
            or artifact.process_witness_start_time_ticks is None
        ):
            raise ManagedClinicalOwnershipError('process operation lacks a truthful identity binding')
        return ManagedProcessGroupIdentity(
            process_group_id=process_group,
            session_id=artifact.process_group_session_id,
            identity_source=artifact.process_identity_source,
            process_group_leader_start_time_ticks=(artifact.process_group_leader_start_time_ticks),
            witness_pid=artifact.process_witness_pid,
            witness_start_time_ticks=artifact.process_witness_start_time_ticks,
        )

    def _require_process_artifact(
        self,
        artifact: ManagedClinicalHostArtifact,
        *,
        allow_exited_leader: bool = False,
    ) -> tuple[ManagedProcessGroupIdentity, ManagedClinicalOwnershipRecord]:
        if artifact.artifact_kind != 'process_group':
            raise ManagedClinicalOwnershipError('process operation received a non-process artifact')
        if not hmac.compare_digest(
            artifact.ownership_authentication_hmac_sha256,
            managed_clinical_ownership_hmac(artifact, key=self._key),
        ):
            raise ManagedClinicalOwnershipError('process operation received unowned artifact')
        latest = self.ownership.latest(artifact.run_id)
        if artifact.ownership_record_sha256 != authenticated_managed_clinical_ownership_sha256(latest):
            raise ManagedClinicalOwnershipError('process operation received stale ownership')
        if artifact.process_cgroup_device_id is None or artifact.process_cgroup_inode is None:
            raise ManagedClinicalOwnershipError('process operation lacks an exact cgroup binding')
        identity = self._artifact_process_identity(artifact, latest.record)
        observed = self._live_process_identity(
            latest.record,
            expected_cgroup_device=artifact.process_cgroup_device_id,
            expected_cgroup_inode=artifact.process_cgroup_inode,
        )
        if observed is None:
            if not allow_exited_leader:
                raise ManagedClinicalOwnershipError('owned process group exited before signaling')
            return identity, latest.record
        if not _same_managed_process_group(identity, observed):
            raise ManagedClinicalOwnershipError('process identity changed before signaling')
        return observed, latest.record

    def _signal_exact_firecracker_child(
        self,
        record: ManagedClinicalOwnershipRecord,
        artifact: ManagedClinicalHostArtifact,
        signal_number: signal.Signals,
    ) -> bool:
        """Gracefully signal only a pidfd-pinned Firecracker child.

        A numeric PID or PGID can be reused after the process exits.  A pidfd pins the exact
        kernel process object; when pidfds are unavailable, cleanup skips the graceful phase and
        proceeds to the exact cgroup kill path instead of signaling a reusable number.
        """

        if (
            record.firecracker_pid is None
            or record.firecracker_start_time_ticks is None
            or record.firecracker_executable_sha256 is None
            or record.firecracker_pid_file_path is None
            or record.firecracker_pid_file_device_id is None
            or record.firecracker_pid_file_inode is None
            or record.firecracker_pid_file_owner_uid is None
            or record.firecracker_pid_file_mode is None
            or artifact.process_cgroup_device_id is None
            or artifact.process_cgroup_inode is None
        ):
            return False
        pidfd_open = getattr(os, 'pidfd_open', None)
        pidfd_send_signal = getattr(signal, 'pidfd_send_signal', None)
        if pidfd_open is None or pidfd_send_signal is None:
            return False
        child = _optional_process_identity(
            record.firecracker_pid,
            proc_root=Path(self.ownership.config.proc_root),
        )
        if child is None:
            return False
        if (
            child.start_time_ticks != record.firecracker_start_time_ticks
            or child.process_group_id != record.process_group_id
            or child.session_id != record.process_group_session_id
            or _proc_executable_sha256_or_exact_zombie(
                child,
                proc_root=Path(self.ownership.config.proc_root),
                expected_sha256=record.firecracker_executable_sha256,
            )
            != record.firecracker_executable_sha256
            or _read_pid_file(
                Path(record.firecracker_pid_file_path),
                expected_device=record.firecracker_pid_file_device_id,
                expected_inode=record.firecracker_pid_file_inode,
                expected_owner_uid=record.firecracker_pid_file_owner_uid,
                expected_mode=record.firecracker_pid_file_mode,
            )
            != child.pid
        ):
            raise ManagedClinicalOwnershipError('owned Firecracker child changed before pidfd signaling')
        descriptor: int | None = None
        try:
            descriptor = pidfd_open(child.pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            raise ManagedClinicalOwnershipError('owned Firecracker child could not be pinned with pidfd') from None
        try:
            pinned = _optional_process_identity(
                child.pid,
                proc_root=Path(self.ownership.config.proc_root),
            )
            pids = _read_cgroup_process_ids(
                Path(record.cgroup_path) / 'cgroup.procs',
                expected_parent_device=artifact.process_cgroup_device_id,
                expected_parent_inode=artifact.process_cgroup_inode,
            )
            if (
                pinned is None
                or pinned != child
                or child.pid not in pids
                or _proc_executable_sha256_or_exact_zombie(
                    child,
                    proc_root=Path(self.ownership.config.proc_root),
                    expected_sha256=record.firecracker_executable_sha256,
                )
                != record.firecracker_executable_sha256
            ):
                raise ManagedClinicalOwnershipError('owned Firecracker child changed after pidfd acquisition')
            try:
                pidfd_send_signal(descriptor, signal_number, None, 0)
            except ProcessLookupError:
                return False
            except OSError:
                raise ManagedClinicalOwnershipError('pidfd-bound Firecracker signal failed') from None
            return True
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _kill_exact_owned_cgroup(
        self,
        record: ManagedClinicalOwnershipRecord,
        artifact: ManagedClinicalHostArtifact,
    ) -> None:
        if artifact.process_cgroup_device_id is None or artifact.process_cgroup_inode is None:
            raise ManagedClinicalOwnershipError('cgroup kill lacks an exact inode binding')
        directory = _open_exact_directory(
            Path(record.cgroup_path),
            expected_device=artifact.process_cgroup_device_id,
            expected_inode=artifact.process_cgroup_inode,
        )
        kill_descriptor: int | None = None
        try:
            flags = os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
            try:
                kill_descriptor = os.open('cgroup.kill', flags, dir_fd=directory)
            except OSError:
                raise ManagedClinicalOwnershipError('exact owned cgroup does not expose cgroup.kill') from None
            metadata = os.fstat(kill_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ManagedClinicalOwnershipError('owned cgroup.kill changed type')
            _require_open_directory_still_named(
                Path(record.cgroup_path),
                descriptor=directory,
                expected_device=artifact.process_cgroup_device_id,
                expected_inode=artifact.process_cgroup_inode,
            )
            if os.write(kill_descriptor, b'1') != 1:
                raise ManagedClinicalOwnershipError('owned cgroup.kill write was incomplete')
        finally:
            if kill_descriptor is not None:
                os.close(kill_descriptor)
            os.close(directory)

    def _owned_process_group_alive(
        self,
        identity: ManagedProcessGroupIdentity,
        record: ManagedClinicalOwnershipRecord,
        artifact: ManagedClinicalHostArtifact,
    ) -> bool:
        if artifact.process_cgroup_device_id is None or artifact.process_cgroup_inode is None:
            raise ManagedClinicalOwnershipError('process liveness lacks an exact cgroup binding')
        observed = self._live_process_identity(
            record,
            expected_cgroup_device=artifact.process_cgroup_device_id,
            expected_cgroup_inode=artifact.process_cgroup_inode,
        )
        if observed is None:
            return False
        if not _same_managed_process_group(identity, observed):
            raise ManagedClinicalOwnershipError('owned process group changed identity')
        return True


class DurableManagedClinicalCapabilityLedger:
    """Restart-visible capability projection backed by the ownership transition chain."""

    def __init__(
        self,
        *,
        ownership: DurableManagedClinicalOwnershipLedger,
        ownership_key: bytes,
        capability_revoke: Callable[[str], None],
    ) -> None:
        self.ownership = ownership
        self._key = bytes(ownership_key)
        self._capability_revoke = capability_revoke

    def inventory(self) -> tuple[ManagedClinicalCapability, ...]:
        values = []
        for envelope in self.ownership.active():
            record = envelope.record
            if record.capability_id is None or record.capability_revoked:
                continue
            if record.start_redemption_sha256 is None:
                raise ManagedClinicalOwnershipError('active capability lacks its start-redemption binding')
            unsigned = ManagedClinicalCapability(
                capability_id=record.capability_id,
                run_id=record.run_id,
                registry_authority_id=record.registry_authority_id,
                reservation_sha256=record.reservation_sha256,
                launch_sha256=record.launch_sha256,
                start_redemption_sha256=record.start_redemption_sha256,
                worker_spec_sha256=record.worker_spec_sha256,
                ownership_record_sha256=authenticated_managed_clinical_ownership_sha256(envelope),
                ownership_authentication_hmac_sha256='0' * 64,
            )
            values.append(
                unsigned.model_copy(
                    update={
                        'ownership_authentication_hmac_sha256': managed_clinical_ownership_hmac(
                            unsigned,
                            key=self._key,
                        )
                    }
                )
            )
        return tuple(values)

    def revoke(self, capability: ManagedClinicalCapability) -> None:
        if not hmac.compare_digest(
            capability.ownership_authentication_hmac_sha256,
            managed_clinical_ownership_hmac(capability, key=self._key),
        ):
            raise ManagedClinicalOwnershipError('capability revocation received unowned metadata')
        latest = self.ownership.latest(capability.run_id)
        if (
            latest.record.capability_id != capability.capability_id
            or authenticated_managed_clinical_ownership_sha256(latest) != capability.ownership_record_sha256
        ):
            raise ManagedClinicalOwnershipError('capability revocation received stale metadata')
        self._capability_revoke(capability.capability_id)
        self.ownership.record_capability_revoked(
            run_id=capability.run_id,
            capability_id=capability.capability_id,
        )


def _expected_paths(
    spec: FirecrackerWorkerSpec,
    run_id: str,
    config: ManagedClinicalOwnershipConfig,
) -> tuple[Path, Path, Path, Path]:
    executable_name = Path(spec.runtime.firecracker.source_path).name
    expected_jail_namespace = Path(spec.chroot_base_dir) / executable_name
    if expected_jail_namespace != Path(config.jail_namespace_root):
        raise ManagedClinicalOwnershipError('ownership jail namespace differs from worker spec')
    run_container = expected_jail_namespace / run_id
    jail_root = run_container / 'root'
    vsock = jail_root / 'run' / 'vsock.sock'
    cgroup = Path(config.cgroup_namespace_root) / run_id
    return run_container, jail_root, vsock, cgroup


def _scan_complete_vsock_namespace(
    root: Path,
    *,
    active_run_containers: dict[str, Path],
) -> tuple[tuple[Path, os.stat_result], ...]:
    """Enumerate every socket below the pinned jail namespace without following aliases."""

    metadata = root.lstat()
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(root, flags)
    except OSError:
        raise ManagedClinicalOwnershipError('managed vsock namespace could not be opened safely') from None
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino) or not stat.S_ISDIR(opened.st_mode):
            raise ManagedClinicalOwnershipError('managed vsock namespace changed while opening')
        mount_id = _linux_fd_mount_id(descriptor)
        try:
            with os.scandir(descriptor) as scanner:
                entries = tuple(sorted(scanner, key=lambda item: item.name))
        except OSError:
            raise ManagedClinicalOwnershipError('managed vsock namespace inventory is unavailable') from None
        discovered: list[tuple[Path, os.stat_result]] = []
        observed_entries = [0]
        for entry in entries:
            observed_entries[0] += 1
            if observed_entries[0] > _MAX_VSOCK_NAMESPACE_ENTRIES:
                raise ManagedClinicalOwnershipError('managed vsock namespace exceeds its entry limit')
            expected = active_run_containers.get(entry.name)
            if expected is None:
                raise ManagedClinicalOwnershipError('managed vsock namespace contains an unowned top-level entry')
            run_container = root / entry.name
            if expected != run_container or entry.is_symlink():
                raise ManagedClinicalOwnershipError('managed vsock namespace run layout differs from ownership')
            entry_metadata = entry.stat(follow_symlinks=False)
            if not stat.S_ISDIR(entry_metadata.st_mode):
                raise ManagedClinicalOwnershipError('managed vsock namespace run entry is not a directory')
            child = _open_scanned_directory(
                descriptor,
                entry.name,
                expected_metadata=entry_metadata,
                expected_mount_id=mount_id,
                label='managed vsock run container',
            )
            try:
                _scan_vsock_tree_fd(
                    child,
                    path=run_container,
                    expected_mount_id=mount_id,
                    depth=0,
                    observed_entries=observed_entries,
                    discovered=discovered,
                )
                _require_scanned_entry_stable(
                    descriptor,
                    entry.name,
                    child_descriptor=child,
                    expected_metadata=entry_metadata,
                    label='managed vsock run container',
                )
            finally:
                os.close(child)
        _require_open_directory_still_named(
            root,
            descriptor=descriptor,
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
        )
        if _linux_fd_mount_id(descriptor) != mount_id:
            raise ManagedClinicalOwnershipError('managed vsock namespace crossed a mount boundary during inventory')
        return tuple(discovered)
    finally:
        os.close(descriptor)


def _scan_vsock_tree_fd(
    descriptor: int,
    *,
    path: Path,
    expected_mount_id: int,
    depth: int,
    observed_entries: list[int],
    discovered: list[tuple[Path, os.stat_result]],
) -> None:
    if depth > _MAX_VSOCK_NAMESPACE_DEPTH:
        raise ManagedClinicalOwnershipError('managed vsock namespace exceeds its depth limit')
    try:
        with os.scandir(descriptor) as scanner:
            entries = tuple(sorted(scanner, key=lambda item: item.name))
    except OSError:
        raise ManagedClinicalOwnershipError('managed vsock namespace subtree is unavailable') from None
    for entry in entries:
        observed_entries[0] += 1
        if observed_entries[0] > _MAX_VSOCK_NAMESPACE_ENTRIES:
            raise ManagedClinicalOwnershipError('managed vsock namespace exceeds its entry limit')
        if entry.is_symlink():
            raise ManagedClinicalOwnershipError('managed vsock namespace contains an unexpected symlink')
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError:
            raise ManagedClinicalOwnershipError('managed vsock namespace entry is unavailable') from None
        child_path = path / entry.name
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_scanned_directory(
                descriptor,
                entry.name,
                expected_metadata=metadata,
                expected_mount_id=expected_mount_id,
                label='managed vsock namespace directory',
            )
            try:
                _scan_vsock_tree_fd(
                    child,
                    path=child_path,
                    expected_mount_id=expected_mount_id,
                    depth=depth + 1,
                    observed_entries=observed_entries,
                    discovered=discovered,
                )
                _require_scanned_entry_stable(
                    descriptor,
                    entry.name,
                    child_descriptor=child,
                    expected_metadata=metadata,
                    label='managed vsock namespace directory',
                )
            finally:
                os.close(child)
            continue
        if stat.S_ISSOCK(metadata.st_mode):
            discovered.append((child_path, metadata))
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ManagedClinicalOwnershipError('managed vsock namespace contains an unrepresentable special entry')


def _open_scanned_directory(
    parent_descriptor: int,
    name: str,
    *,
    expected_metadata: os.stat_result,
    expected_mount_id: int,
    label: str,
) -> int:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError:
        raise ManagedClinicalOwnershipError(f'{label} could not be opened safely') from None
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (expected_metadata.st_dev, expected_metadata.st_ino)
            or _linux_fd_mount_id(descriptor) != expected_mount_id
        ):
            raise ManagedClinicalOwnershipError(f'{label} changed identity or crossed a mount boundary')
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _require_scanned_entry_stable(
    parent_descriptor: int,
    name: str,
    *,
    child_descriptor: int,
    expected_metadata: os.stat_result,
    label: str,
) -> None:
    try:
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        opened = os.fstat(child_descriptor)
    except OSError:
        raise ManagedClinicalOwnershipError(f'{label} changed during inventory') from None
    if (
        not stat.S_ISDIR(named.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (named.st_dev, named.st_ino) != (expected_metadata.st_dev, expected_metadata.st_ino)
        or (opened.st_dev, opened.st_ino) != (expected_metadata.st_dev, expected_metadata.st_ino)
    ):
        raise ManagedClinicalOwnershipError(f'{label} changed during inventory')


def _optional_process_identity(pid: int, *, proc_root: Path) -> LinuxProcessIdentity | None:
    try:
        return read_linux_process_identity(pid, proc_root=proc_root)
    except ManagedClinicalOwnershipError:
        try:
            (proc_root / str(pid)).lstat()
        except FileNotFoundError:
            return None
        except OSError:
            pass
        raise


def _read_cgroup_process_ids(
    path: Path,
    *,
    expected_parent_device: int,
    expected_parent_inode: int,
) -> tuple[int, ...]:
    directory = _open_exact_directory(
        path.parent,
        expected_device=expected_parent_device,
        expected_inode=expected_parent_inode,
    )
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        descriptor = os.open(path.name, flags, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ManagedClinicalOwnershipError('owned cgroup process inventory changed type')
        content = _read_bounded_descriptor(
            descriptor,
            maximum_bytes=_MAX_PROC_BYTES,
            label='owned cgroup process inventory',
        )
        after = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_mode, before.st_uid) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
        ):
            raise ManagedClinicalOwnershipError('owned cgroup process inventory changed while reading')
        _require_open_directory_still_named(
            path.parent,
            descriptor=directory,
            expected_device=expected_parent_device,
            expected_inode=expected_parent_inode,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)
    if len(content) > _MAX_PROC_BYTES or b'\x00' in content:
        raise ManagedClinicalOwnershipError('owned cgroup process inventory is invalid')
    try:
        values = tuple(sorted({int(value) for value in content.split()}))
    except ValueError:
        raise ManagedClinicalOwnershipError('owned cgroup process inventory is invalid') from None
    if any(value <= 1 or value > 2**31 - 1 for value in values):
        raise ManagedClinicalOwnershipError('owned cgroup process inventory has an invalid PID')
    return values


def _read_pid_file(
    path: Path,
    *,
    expected_device: int,
    expected_inode: int,
    expected_owner_uid: int,
    expected_mode: int,
) -> int:
    try:
        content = _read_stable_nofollow_file(
            path,
            maximum_bytes=128,
            label='Firecracker pid file',
            expected_device=expected_device,
            expected_inode=expected_inode,
            expected_owner_uid=expected_owner_uid,
            expected_mode=expected_mode,
            require_single_link=True,
        )
    except OSError:
        raise ManagedClinicalOwnershipError('Firecracker pid file is unavailable') from None
    if not content or len(content) > 128 or b'\x00' in content:
        raise ManagedClinicalOwnershipError('Firecracker pid file is invalid')
    try:
        value = int(content.strip())
    except ValueError:
        raise ManagedClinicalOwnershipError('Firecracker pid file is invalid') from None
    if value <= 1 or value > 2**31 - 1:
        raise ManagedClinicalOwnershipError('Firecracker pid file has an invalid PID')
    return value


def _read_stable_nofollow_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    expected_owner_uid: int | None = None,
    expected_mode: int | None = None,
    require_single_link: bool = False,
) -> bytes:
    try:
        before = path.lstat()
    except OSError:
        raise ManagedClinicalOwnershipError(f'{label} is unavailable') from None
    _require_expected_file_metadata(
        before,
        label=label,
        expected_device=expected_device,
        expected_inode=expected_inode,
        expected_owner_uid=expected_owner_uid,
        expected_mode=expected_mode,
        require_single_link=require_single_link,
    )
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        _require_expected_file_metadata(
            opened,
            label=label,
            expected_device=expected_device,
            expected_inode=expected_inode,
            expected_owner_uid=expected_owner_uid,
            expected_mode=expected_mode,
            require_single_link=require_single_link,
        )
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ManagedClinicalOwnershipError(f'{label} changed while opening')
        content = _read_bounded_descriptor(
            descriptor,
            maximum_bytes=maximum_bytes,
            label=label,
        )
        after = path.lstat()
        if (after.st_dev, after.st_ino, after.st_mode, after.st_uid, after.st_nlink) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_uid,
            opened.st_nlink,
        ):
            raise ManagedClinicalOwnershipError(f'{label} changed while reading')
        return content
    except OSError:
        raise ManagedClinicalOwnershipError(f'{label} is unavailable') from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_expected_file_metadata(
    metadata: os.stat_result,
    *,
    label: str,
    expected_device: int | None,
    expected_inode: int | None,
    expected_owner_uid: int | None,
    expected_mode: int | None,
    require_single_link: bool,
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or (expected_device is not None and metadata.st_dev != expected_device)
        or (expected_inode is not None and metadata.st_ino != expected_inode)
        or (expected_owner_uid is not None and metadata.st_uid != expected_owner_uid)
        or (expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode)
        or (require_single_link and metadata.st_nlink != 1)
    ):
        raise ManagedClinicalOwnershipError(f'{label} metadata differs from its ownership pin')


def _read_bounded_descriptor(
    descriptor: int,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    while True:
        block = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - observed))
        if not block:
            break
        chunks.append(block)
        observed += len(block)
        if observed > maximum_bytes:
            raise ManagedClinicalOwnershipError(f'{label} exceeds its byte limit')
    return b''.join(chunks)


def _open_exact_directory(
    path: Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> int:
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    except OSError:
        raise ManagedClinicalOwnershipError('owned cgroup directory is unavailable') from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or (before.st_dev, before.st_ino) != (expected_device, expected_inode)
    ):
        raise ManagedClinicalOwnershipError('owned cgroup directory changed identity')
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError:
        raise ManagedClinicalOwnershipError('owned cgroup directory is unavailable') from None
    try:
        _require_open_directory_still_named(
            path,
            descriptor=descriptor,
            expected_device=expected_device,
            expected_inode=expected_inode,
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _require_open_directory_still_named(
    path: Path,
    *,
    descriptor: int,
    expected_device: int,
    expected_inode: int,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
    except OSError:
        raise ManagedClinicalOwnershipError('owned cgroup directory changed identity') from None
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or (opened.st_dev, opened.st_ino) != (expected_device, expected_inode)
        or (named.st_dev, named.st_ino) != (expected_device, expected_inode)
    ):
        raise ManagedClinicalOwnershipError('owned cgroup directory changed identity')


def _proc_executable_sha256(pid: int, *, proc_root: Path) -> str:
    path = proc_root / str(pid) / 'exe'
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
    except OSError:
        raise ManagedClinicalOwnershipError('Firecracker executable identity is unavailable') from None
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ManagedClinicalOwnershipError('Firecracker executable is not a regular file')
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _proc_executable_sha256_or_exact_zombie(
    identity: LinuxProcessIdentity,
    *,
    proc_root: Path,
    expected_sha256: str,
) -> str:
    """Retain a prior executable pin only for one immutable, exact zombie identity."""

    try:
        return _proc_executable_sha256(identity.pid, proc_root=proc_root)
    except ManagedClinicalOwnershipError as error:
        deadline = time.monotonic() + 0.1
        while True:
            rebound = _optional_process_identity(identity.pid, proc_root=proc_root)
            if rebound is None:
                return expected_sha256
            if not _same_linux_process_kernel_identity(identity, rebound):
                raise error
            if rebound.process_state in {'Z', 'X', 'x'}:
                # A zombie cannot execute or change its image.  PID reuse is excluded by the
                # stable start counter, group, session, terminal state, and exact cgroup binding.
                return expected_sha256
            if time.monotonic() >= deadline:
                raise error
            time.sleep(0.001)


def _artifact_matches_ownership_record(
    artifact: ManagedClinicalHostArtifact,
    record: ManagedClinicalOwnershipRecord,
) -> bool:
    return (
        artifact.run_id,
        artifact.registry_authority_id,
        artifact.reservation_sha256,
        artifact.launch_sha256,
        artifact.start_redemption_sha256,
        artifact.worker_spec_sha256,
    ) == (
        record.run_id,
        record.registry_authority_id,
        record.reservation_sha256,
        record.launch_sha256,
        record.start_redemption_sha256,
        record.worker_spec_sha256,
    )


def _ownership_binding_regressed(
    previous: ManagedClinicalOwnershipRecord,
    current: ManagedClinicalOwnershipRecord,
) -> bool:
    """Reject removal or mutation of any kernel/capability identity once first observed."""

    monotonic_fields = (
        'prepared_worker_sha256',
        'run_container_device_id',
        'run_container_inode',
        'start_redemption_sha256',
        'capability_id',
        'process_group_id',
        'process_group_leader_start_time_ticks',
        'process_group_session_id',
        'firecracker_pid',
        'firecracker_start_time_ticks',
        'firecracker_executable_sha256',
        'firecracker_pid_file_path',
        'firecracker_pid_file_device_id',
        'firecracker_pid_file_inode',
        'firecracker_pid_file_owner_uid',
        'firecracker_pid_file_mode',
        'cgroup_device_id',
        'cgroup_inode',
    )
    return any(
        getattr(previous, name) is not None and getattr(current, name) != getattr(previous, name)
        for name in monotonic_fields
    ) or (previous.capability_revoked and not current.capability_revoked)


def _require_path_type(path: Path, *, directory: bool) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError:
        raise ManagedClinicalOwnershipError('owned path is unavailable') from None
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode) or not expected:
        raise ManagedClinicalOwnershipError('owned path changed type')
    return metadata


def _unlink_exact(
    path: Path,
    *,
    expected_device: int | None,
    expected_inode: int | None,
    expected_socket: bool,
) -> None:
    metadata = path.lstat()
    if (
        (metadata.st_dev, metadata.st_ino) != (expected_device, expected_inode)
        or (expected_socket and not stat.S_ISSOCK(metadata.st_mode))
        or stat.S_ISLNK(metadata.st_mode)
    ):
        raise ManagedClinicalOwnershipError('owned path identity changed before unlink')
    parent = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0),
    )
    try:
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def _rmdir_exact(
    path: Path,
    *,
    expected_device: int | None,
    expected_inode: int | None,
) -> None:
    metadata = path.lstat()
    if (
        (metadata.st_dev, metadata.st_ino) != (expected_device, expected_inode)
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ManagedClinicalOwnershipError('owned directory identity changed before removal')
    parent = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0),
    )
    try:
        os.rmdir(path.name, dir_fd=parent)
        # This helper is used for an exact cgroup-v2 directory.  cgroupfs is kernel state rather
        # than durable storage and rejects fsync(2) with EINVAL after a successful rmdir.  The
        # caller binds the inode before removal and the startup reconciler performs a full rescan.
    finally:
        os.close(parent)


def _remove_tree_exact_fd(
    path: Path,
    *,
    expected_device: int | None,
    expected_inode: int | None,
) -> None:
    parent = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0),
    )
    try:
        parent_device = os.fstat(parent).st_dev
        parent_mount_id = _linux_fd_mount_id(parent)
        _remove_directory_entry_fd(
            parent,
            path.name,
            expected_device=expected_device,
            expected_inode=expected_inode,
            expected_filesystem_device=parent_device,
            expected_mount_id=parent_mount_id,
        )
        os.fsync(parent)
    finally:
        os.close(parent)


def _remove_directory_entry_fd(
    parent_fd: int,
    name: str,
    *,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    expected_filesystem_device: int,
    expected_mount_id: int,
) -> None:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        metadata = os.fstat(descriptor)
        if expected_device is not None and (
            metadata.st_dev,
            metadata.st_ino,
        ) != (expected_device, expected_inode):
            raise ManagedClinicalOwnershipError('owned tree identity changed before removal')
        if metadata.st_dev != expected_filesystem_device or _linux_fd_mount_id(descriptor) != expected_mount_id:
            raise ManagedClinicalOwnershipError('owned tree crosses a filesystem or mount boundary')
        with os.scandir(descriptor) as entries:
            inventory = tuple(entries)
        for entry in inventory:
            if entry.is_dir(follow_symlinks=False):
                _remove_directory_entry_fd(
                    descriptor,
                    entry.name,
                    expected_filesystem_device=expected_filesystem_device,
                    expected_mount_id=expected_mount_id,
                )
            else:
                os.unlink(entry.name, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=parent_fd)


def _linux_fd_mount_id(descriptor: int) -> int:
    path = Path('/proc/self/fdinfo') / str(descriptor)
    try:
        content = path.read_bytes()
    except OSError:
        raise ManagedClinicalOwnershipError('Linux mount identity is unavailable') from None
    if not content or len(content) > 64 * 1024 or b'\x00' in content:
        raise ManagedClinicalOwnershipError('Linux mount identity is invalid')
    values = []
    for line in content.splitlines():
        if line.startswith(b'mnt_id:'):
            try:
                values.append(int(line.partition(b':')[2].strip()))
            except ValueError:
                raise ManagedClinicalOwnershipError('Linux mount identity is invalid') from None
    if len(values) != 1 or values[0] <= 0:
        raise ManagedClinicalOwnershipError('Linux mount identity is unavailable')
    return values[0]


def _prepare_private_root(path: Path) -> Path:
    if path.is_symlink():
        raise ManagedClinicalOwnershipError('ownership ledger root cannot be a symlink')
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        pass
    except OSError:
        raise ManagedClinicalOwnershipError('ownership ledger root could not be created') from None
    resolved = path.resolve(strict=True)
    # Darwin's system ``/var -> /private/var`` alias affects every pytest temporary path.  This
    # boundary is executed only on the Linux/KVM qualification host, where any changed component
    # is an unsafe deployment redirect and must fail closed.
    if os.uname().sysname == 'Linux' and resolved != path:
        raise ManagedClinicalOwnershipError('ownership ledger root cannot contain symbolic-link components')
    _require_private_directory(resolved)
    return resolved


def _require_private_directory(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError:
        raise ManagedClinicalOwnershipError('ownership directory is unavailable') from None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ManagedClinicalOwnershipError('ownership directory must be owned mode-0700')
    return metadata


def _write_create_once(path: Path, content: bytes) -> None:
    if not content or len(content) > _MAX_RECORD_BYTES:
        raise ManagedClinicalOwnershipError('ownership record has invalid size')
    parent = path.parent
    _require_private_directory(parent)
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
    )
    pending_name = f'.{path.name}.pending'
    descriptor = -1
    try:
        _recover_one_pending_publication(
            parent_descriptor,
            pending_name=pending_name,
        )
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            raise ManagedClinicalOwnershipError('ownership publication target could not be inspected') from None
        else:
            raise ManagedClinicalOwnershipError('ownership publication target already exists')
        descriptor = os.open(
            pending_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(content):
            count = os.write(descriptor, content[written:])
            if count <= 0:
                raise ManagedClinicalOwnershipError('ownership record write made no progress')
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            rename_file_noreplace(parent / pending_name, path)
        except (AtomicDirectoryPublicationError, OSError):
            raise ManagedClinicalOwnershipError('ownership record could not be atomically published') from None
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _recover_pending_record_publications(root: Path) -> None:
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
    )
    changed = False
    try:
        for name in os.listdir(descriptor):
            if (
                len(name) == len('.0000.json.pending')
                and name.startswith('.')
                and name.endswith('.json.pending')
                and name[1:5].isdigit()
            ):
                _recover_one_pending_publication(
                    descriptor,
                    pending_name=name,
                )
                changed = True
        if changed:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _recover_one_pending_publication(
    parent_descriptor: int,
    *,
    pending_name: str,
) -> None:
    try:
        metadata = os.stat(
            pending_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError:
        raise ManagedClinicalOwnershipError('interrupted ownership publication could not be inspected') from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ManagedClinicalOwnershipError('interrupted ownership publication has unsafe metadata')
    try:
        os.unlink(pending_name, dir_fd=parent_descriptor)
    except OSError:
        raise ManagedClinicalOwnershipError('interrupted ownership publication could not be discarded') from None


def _read_stable_private_file(path: Path) -> bytes:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
    ):
        raise ManagedClinicalOwnershipError('ownership record must be one owned mode-0600 file')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ManagedClinicalOwnershipError('ownership record changed while opening')
        content = bytearray()
        while len(content) <= _MAX_RECORD_BYTES:
            block = os.read(descriptor, min(1024 * 1024, _MAX_RECORD_BYTES + 1 - len(content)))
            if not block:
                break
            content.extend(block)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or not content
        or len(content) > _MAX_RECORD_BYTES
    ):
        raise ManagedClinicalOwnershipError('ownership record changed or has invalid size')
    return bytes(content)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    'AuthenticatedManagedClinicalOwnership',
    'DurableManagedClinicalCapabilityLedger',
    'DurableManagedClinicalOwnershipLedger',
    'LinuxManagedClinicalHostAdapter',
    'LinuxProcessIdentity',
    'ManagedClinicalOwnershipConfig',
    'ManagedClinicalOwnershipError',
    'ManagedClinicalOwnershipRecord',
    'ManagedProcessGroupIdentity',
    'authenticated_managed_clinical_ownership_sha256',
    'managed_clinical_ownership_config_sha256',
    'read_linux_process_identity',
]
