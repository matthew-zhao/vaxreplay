"""Fail-closed Firecracker worker preparation and supervision.

This module is a production-shaped host contract, not a claim that the repository has boot-qualified
Firecracker on every platform.  The live preflight deliberately succeeds only on a root-operated
Linux host with KVM, cgroup v2, and root-owned digest-pinned artifacts.  Unit tests can exercise the
deterministic configuration, copying, cleanup, and vsock protocol on other platforms; an official
deployment still has to boot and attack-test the exact pinned Linux/KVM stack.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import platform
import select
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

FIRECRACKER_ARTIFACT_SCHEMA_VERSION = 'vaxreplay.firecracker-artifact.v0.1'
FIRECRACKER_RUNTIME_SCHEMA_VERSION = 'vaxreplay.firecracker-runtime.v0.1'
FIRECRACKER_IMAGES_SCHEMA_VERSION = 'vaxreplay.firecracker-images.v0.1'
FIRECRACKER_LIMITS_SCHEMA_VERSION = 'vaxreplay.firecracker-limits.v0.1'
FIRECRACKER_WORKER_SPEC_SCHEMA_VERSION = 'vaxreplay.firecracker-worker-spec.v0.1'
FIRECRACKER_GUEST_BOOTSTRAP_PROFILE_SCHEMA_VERSION = 'vaxreplay.firecracker-guest-bootstrap-profile.v0.1'
FIRECRACKER_HOST_PREFLIGHT_SCHEMA_VERSION = 'vaxreplay.firecracker-host-preflight.v0.1'
FIRECRACKER_PREPARED_WORKER_SCHEMA_VERSION = 'vaxreplay.firecracker-prepared-worker.v0.1'
FIRECRACKER_PREBOUND_GUEST_LISTENER_SCHEMA_VERSION = 'vaxreplay.firecracker-prebound-guest-listener.v0.1'
FIRECRACKER_CLEANUP_SCHEMA_VERSION = 'vaxreplay.firecracker-cleanup.v0.3'
FIRECRACKER_WORKER_ATTESTATION_SCHEMA_VERSION = 'vaxreplay.firecracker-worker-attestation.v0.2'
AUTHENTICATED_FIRECRACKER_WORKER_ATTESTATION_SCHEMA_VERSION = (
    'vaxreplay.authenticated-firecracker-worker-attestation.v0.2'
)

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_SAFE_CGROUP_PARENT_PATTERN = r'^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$'
_REQUIRED_CGROUP_CONTROLLERS = frozenset({'cpu', 'memory', 'pids'})
_MIB = 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_VSOCK_HANDSHAKE_BYTES = 128
_MAX_FIRECRACKER_PID_BYTES = 10
_FIRECRACKER_CHILD_OBSERVATION_SECONDS = 10.0
_KERNEL_BOOT_ARGS = 'reboot=k panic=1 pci=off random.trust_cpu=on'
_ATTESTATION_KEY_ID_DOMAIN = b'vaxreplay.firecracker-attestation-key-id.v0.1\x00'
_ATTESTATION_HMAC_DOMAIN = b'vaxreplay.firecracker-worker-attestation.v0.2\x00'
_GUEST_DISK_SHA256_SENTINEL = '0' * 64


class FirecrackerWorkerError(RuntimeError):
    """Base error for a rejected or failed microVM lifecycle operation."""


class FirecrackerPreflightError(FirecrackerWorkerError):
    """Raised before launch when the host or an immutable artifact is unsafe."""


class FirecrackerPreparationError(FirecrackerWorkerError):
    """Raised when a fresh jail cannot be constructed exactly."""


class FirecrackerCleanupError(FirecrackerWorkerError):
    """Raised when process termination or ephemeral-state removal cannot be proved."""


class FirecrackerVsockError(FirecrackerWorkerError):
    """Raised when the Firecracker host-side vsock handshake is invalid."""


class FirecrackerAttestationError(FirecrackerWorkerError):
    """Raised when authenticated worker evidence is incomplete or invalid."""


class FirecrackerArtifactRole(str, enum.Enum):
    FIRECRACKER = 'firecracker'
    JAILER = 'jailer'
    KERNEL = 'kernel'
    ROOTFS = 'rootfs'
    HARNESS = 'harness'
    SCRATCH_TEMPLATE = 'scratch_template'


class FirecrackerArtifactIdentity(StrictModel):
    """Expected bytes for one root-owned host artifact."""

    schema_version: Literal['vaxreplay.firecracker-artifact.v0.1'] = FIRECRACKER_ARTIFACT_SCHEMA_VERSION
    artifact_id: str = Field(min_length=1, max_length=200)
    role: FirecrackerArtifactRole
    source_path: str = Field(min_length=2, max_length=4096)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=1024 * 1024 * 1024 * 1024)

    @field_validator('source_path')
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        return _absolute_normalized_path(value, label='artifact source path')


class FirecrackerRuntimeIdentity(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-runtime.v0.1'] = FIRECRACKER_RUNTIME_SCHEMA_VERSION
    release: str = Field(min_length=1, max_length=200)
    architecture: Literal['x86_64', 'aarch64']
    firecracker: FirecrackerArtifactIdentity
    jailer: FirecrackerArtifactIdentity

    @model_validator(mode='after')
    def validate_runtime(self) -> Self:
        if self.firecracker.role != FirecrackerArtifactRole.FIRECRACKER:
            raise ValueError('firecracker runtime artifact must use the firecracker role')
        if self.jailer.role != FirecrackerArtifactRole.JAILER:
            raise ValueError('jailer runtime artifact must use the jailer role')
        if self.firecracker.source_path == self.jailer.source_path:
            raise ValueError('firecracker and jailer must be distinct pinned files')
        return self


class FirecrackerGuestImages(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-images.v0.1'] = FIRECRACKER_IMAGES_SCHEMA_VERSION
    kernel: FirecrackerArtifactIdentity
    rootfs: FirecrackerArtifactIdentity
    harness: FirecrackerArtifactIdentity
    scratch_template: FirecrackerArtifactIdentity

    @model_validator(mode='after')
    def validate_images(self) -> Self:
        expected = (
            (self.kernel, FirecrackerArtifactRole.KERNEL),
            (self.rootfs, FirecrackerArtifactRole.ROOTFS),
            (self.harness, FirecrackerArtifactRole.HARNESS),
            (self.scratch_template, FirecrackerArtifactRole.SCRATCH_TEMPLATE),
        )
        if any(artifact.role != role for artifact, role in expected):
            raise ValueError('guest image fields must use their corresponding artifact roles')
        paths = tuple(artifact.source_path for artifact, _ in expected)
        if len(paths) != len(set(paths)):
            raise ValueError('kernel, rootfs, harness, and scratch template must be distinct files')
        return self


class FirecrackerResourceLimits(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-limits.v0.1'] = FIRECRACKER_LIMITS_SCHEMA_VERSION
    wall_seconds: int = Field(default=600, ge=1, le=86_400)
    vcpu_count: int = Field(default=4, ge=1, le=32)
    cpu_period_us: int = Field(default=100_000, ge=1_000, le=1_000_000)
    cpu_quota_us: int = Field(default=400_000, ge=1_000, le=32_000_000)
    memory_mib: int = Field(default=8192, ge=128, le=1_048_576)
    pids: int = Field(default=256, ge=16, le=1_048_576)
    open_files: int = Field(default=1024, ge=64, le=1_048_576)
    scratch_bytes: int = Field(default=1024 * _MIB, ge=_MIB, le=1024 * 1024 * _MIB)

    @model_validator(mode='after')
    def validate_cpu_limit(self) -> Self:
        if self.cpu_quota_us > self.vcpu_count * self.cpu_period_us:
            raise ValueError('CPU quota cannot exceed the configured virtual CPU capacity')
        return self


class FirecrackerWorkerSpec(StrictModel):
    """Organizer-owned, immutable description of one allowed worker stack."""

    schema_version: Literal['vaxreplay.firecracker-worker-spec.v0.1'] = FIRECRACKER_WORKER_SPEC_SCHEMA_VERSION
    worker_id: str = Field(min_length=1, max_length=200)
    runtime: FirecrackerRuntimeIdentity
    images: FirecrackerGuestImages
    limits: FirecrackerResourceLimits = Field(default_factory=FirecrackerResourceLimits)
    chroot_base_dir: str = Field(min_length=2, max_length=4096)
    cgroup_parent: str = Field(pattern=_SAFE_CGROUP_PARENT_PATTERN, min_length=1, max_length=500)
    worker_uid: int = Field(ge=1, le=2**31 - 1)
    worker_gid: int = Field(ge=1, le=2**31 - 1)
    guest_cid: int = Field(ge=3, le=2**32 - 1)
    guest_rpc_port: int = Field(ge=1, le=2**32 - 1)
    kernel_boot_args: Literal['reboot=k panic=1 pci=off random.trust_cpu=on'] = _KERNEL_BOOT_ARGS
    network_interfaces_enabled: Literal[False] = False
    mmds_enabled: Literal[False] = False
    api_enabled: Literal[False] = False
    new_pid_namespace: Literal[True] = True
    cgroup_version: Literal[2] = 2
    rootfs_read_only: Literal[True] = True
    harness_read_only: Literal[True] = True
    scratch_fresh_per_run: Literal[True] = True
    scratch_read_only: Literal[False] = False
    smt_enabled: Literal[False] = False
    track_dirty_pages: Literal[False] = False

    @field_validator('chroot_base_dir')
    @classmethod
    def validate_chroot_base_dir(cls, value: str) -> str:
        return _absolute_normalized_path(value, label='chroot base directory')

    @field_validator('cgroup_parent')
    @classmethod
    def validate_cgroup_parent(cls, value: str) -> str:
        if any(part in {'.', '..'} for part in value.split('/')):
            raise ValueError('cgroup parent cannot contain dot path components')
        return value

    @model_validator(mode='after')
    def validate_spec(self) -> Self:
        if self.images.scratch_template.byte_count != self.limits.scratch_bytes:
            raise ValueError('scratch template byte count must equal the fixed scratch limit')
        all_paths = (
            self.runtime.firecracker.source_path,
            self.runtime.jailer.source_path,
            self.images.kernel.source_path,
            self.images.rootfs.source_path,
            self.images.harness.source_path,
            self.images.scratch_template.source_path,
        )
        if len(all_paths) != len(set(all_paths)):
            raise ValueError('all pinned runtime and guest artifacts must use distinct source files')
        return self


class FirecrackerGuestBootstrapProfile(StrictModel):
    """Non-circular static worker projection safe to bake into the guest image.

    The rootfs and harness image hashes cannot be part of a config baked into either image: doing
    so would require finding a cryptographic fixed point.  This profile retains every other worker
    field and replaces only those two self-referential hashes with an explicit fixed sentinel.  A
    launcher-signed hello still carries the exact, unmodified full worker-spec hash separately.
    """

    schema_version: Literal['vaxreplay.firecracker-guest-bootstrap-profile.v0.1'] = (
        FIRECRACKER_GUEST_BOOTSTRAP_PROFILE_SCHEMA_VERSION
    )
    projected_worker_spec: FirecrackerWorkerSpec
    rootfs_sha256_replaced_by_fixed_sentinel: Literal[True] = True
    harness_sha256_replaced_by_fixed_sentinel: Literal[True] = True
    full_worker_spec_sha256_in_signed_hello_required: Literal[True] = True

    @model_validator(mode='after')
    def validate_projection(self) -> Self:
        if (
            self.projected_worker_spec.images.rootfs.sha256,
            self.projected_worker_spec.images.harness.sha256,
        ) != (_GUEST_DISK_SHA256_SENTINEL, _GUEST_DISK_SHA256_SENTINEL):
            raise ValueError('guest bootstrap profile must normalize both self-referential disk hashes')
        return self


def firecracker_guest_bootstrap_profile(
    spec: FirecrackerWorkerSpec,
) -> FirecrackerGuestBootstrapProfile:
    """Project a full worker spec without either config-dependent guest-disk hash."""

    canonical = FirecrackerWorkerSpec.model_validate_json(canonical_json_bytes(spec))
    images = canonical.images.model_copy(
        update={
            'rootfs': canonical.images.rootfs.model_copy(update={'sha256': _GUEST_DISK_SHA256_SENTINEL}),
            'harness': canonical.images.harness.model_copy(update={'sha256': _GUEST_DISK_SHA256_SENTINEL}),
        }
    )
    projected = canonical.model_copy(update={'images': images})
    return FirecrackerGuestBootstrapProfile(projected_worker_spec=projected)


def firecracker_guest_bootstrap_profile_sha256(spec: FirecrackerWorkerSpec) -> str:
    """Hash the versioned non-circular profile used by the baked guest trust anchor."""

    return hashlib.sha256(canonical_json_bytes(firecracker_guest_bootstrap_profile(spec))).hexdigest()


class FirecrackerBootSource(StrictModel):
    kernel_image_path: Literal['/kernel.bin'] = '/kernel.bin'
    boot_args: Literal['reboot=k panic=1 pci=off random.trust_cpu=on'] = _KERNEL_BOOT_ARGS


class FirecrackerDrive(StrictModel):
    drive_id: Literal['rootfs', 'harness', 'scratch']
    path_on_host: Literal['/rootfs.ext4', '/harness.ext4', '/scratch.ext4']
    is_root_device: bool
    is_read_only: bool


class FirecrackerMachineConfig(StrictModel):
    vcpu_count: int = Field(ge=1, le=32)
    mem_size_mib: int = Field(ge=128, le=1_048_576)
    smt: Literal[False] = False
    track_dirty_pages: Literal[False] = False


class FirecrackerVsockDevice(StrictModel):
    guest_cid: int = Field(ge=3, le=2**32 - 1)
    uds_path: Literal['/run/vsock.sock'] = '/run/vsock.sock'


class FirecrackerStaticConfig(StrictModel):
    """The complete no-API boot configuration; extra devices are rejected."""

    boot_source: FirecrackerBootSource = Field(alias='boot-source')
    drives: tuple[FirecrackerDrive, FirecrackerDrive, FirecrackerDrive]
    machine_config: FirecrackerMachineConfig = Field(alias='machine-config')
    vsock: FirecrackerVsockDevice

    @model_validator(mode='after')
    def validate_devices(self) -> Self:
        expected = (
            ('rootfs', '/rootfs.ext4', True, True),
            ('harness', '/harness.ext4', False, True),
            ('scratch', '/scratch.ext4', False, False),
        )
        observed = tuple(
            (drive.drive_id, drive.path_on_host, drive.is_root_device, drive.is_read_only) for drive in self.drives
        )
        if observed != expected:
            raise ValueError('Firecracker drives must be ordered read-only rootfs, read-only harness, writable scratch')
        return self


class FirecrackerHostPreflightReceipt(StrictModel):
    """An observation, not a remotely authenticated platform attestation."""

    schema_version: Literal['vaxreplay.firecracker-host-preflight.v0.1'] = FIRECRACKER_HOST_PREFLIGHT_SCHEMA_VERSION
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    collected_at: datetime
    host_os: Literal['Linux'] = 'Linux'
    host_architecture: Literal['x86_64', 'aarch64']
    host_kernel_release: str = Field(min_length=1, max_length=500)
    effective_uid: Literal[0] = 0
    kvm_character_device_verified: Literal[True] = True
    kvm_read_write_access_verified: Literal[True] = True
    cgroup_version: Literal[2] = 2
    cgroup_controllers: tuple[str, ...] = Field(min_length=3)
    root_owned_artifact_paths_verified: Literal[True] = True
    artifact_digests_verified: Literal[True] = True
    chroot_base_trusted: Literal[True] = True

    @field_validator('collected_at')
    @classmethod
    def validate_collected_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('preflight timestamp must include a UTC offset')
        return value.astimezone(UTC)

    @field_validator('cgroup_controllers')
    @classmethod
    def validate_controllers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('cgroup controllers must be unique and sorted')
        if not _REQUIRED_CGROUP_CONTROLLERS.issubset(value):
            raise ValueError('cgroup v2 must expose cpu, memory, and pids controllers')
        return value


class FirecrackerPreparedWorker(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-prepared-worker.v0.1'] = FIRECRACKER_PREPARED_WORKER_SCHEMA_VERSION
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    host_preflight: FirecrackerHostPreflightReceipt
    host_preflight_sha256: str = Field(pattern=_SHA256_PATTERN)
    jail_root: str = Field(min_length=2, max_length=4096)
    config_path: str = Field(min_length=2, max_length=4096)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    kernel_sha256: str = Field(pattern=_SHA256_PATTERN)
    rootfs_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_sha256: str = Field(pattern=_SHA256_PATTERN)
    initial_scratch_sha256: str = Field(pattern=_SHA256_PATTERN)
    vsock_uds_path: str = Field(min_length=2, max_length=4096)
    created_at: datetime
    rootfs_read_only: Literal[True] = True
    harness_read_only: Literal[True] = True
    scratch_writable: Literal[True] = True
    scratch_fresh_copy: Literal[True] = True
    network_interfaces_absent: Literal[True] = True
    mmds_absent: Literal[True] = True
    api_disabled: Literal[True] = True

    @field_validator('jail_root', 'config_path', 'vsock_uds_path')
    @classmethod
    def validate_host_path(cls, value: str, info) -> str:
        return _absolute_normalized_path(value, label=info.field_name)

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('prepared worker timestamp must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_paths(self) -> Self:
        if self.host_preflight.worker_spec_sha256 != self.worker_spec_sha256:
            raise ValueError('prepared worker and host preflight must bind the same worker specification')
        if firecracker_model_sha256(self.host_preflight) != self.host_preflight_sha256:
            raise ValueError('prepared worker host preflight hash does not match the embedded observation')
        root = PurePosixPath(self.jail_root)
        if PurePosixPath(self.config_path) != root / 'firecracker-config.json':
            raise ValueError('prepared config path must be the fixed file inside the jail root')
        if PurePosixPath(self.vsock_uds_path) != root / 'run' / 'vsock.sock':
            raise ValueError('prepared vsock path must be the fixed socket inside the jail root')
        return self


class FirecrackerPreboundGuestListener(StrictModel):
    """Exact callback socket intentionally added after immutable jail preparation.

    A host listener must exist before Firecracker starts, or a fast guest can race past the
    callback endpoint. This inode-bound record permits only that deliberate mutation; it never
    turns a generally non-empty prepared jail into an accepted launch input.
    """

    schema_version: Literal['vaxreplay.firecracker-prebound-guest-listener.v0.1'] = (
        FIRECRACKER_PREBOUND_GUEST_LISTENER_SCHEMA_VERSION
    )
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    socket_path: str = Field(min_length=2, max_length=4096)
    device_id: int = Field(ge=0)
    inode: int = Field(gt=0)
    worker_uid: int = Field(ge=1, le=2**31 - 1)
    worker_gid: int = Field(ge=1, le=2**31 - 1)
    mode: Literal[384] = 0o600

    @field_validator('socket_path')
    @classmethod
    def validate_socket_path(cls, value: str) -> str:
        return _absolute_normalized_path(value, label='prebound guest-listener socket path')


class FirecrackerCleanupReceipt(StrictModel):
    """Observed cleanup milestones; monotonic time is authoritative for ordering.

    UTC fields correlate the observations with other retained evidence.  They are not synthesized
    from a monotonic deadline and are deliberately not used to calculate elapsed time because the
    realtime clock can be adjusted while a worker is running.
    """

    schema_version: Literal['vaxreplay.firecracker-cleanup.v0.3'] = FIRECRACKER_CLEANUP_SCHEMA_VERSION
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    launched_monotonic_ns: int | None = Field(ge=0)
    wall_deadline_monotonic_ns: int | None = Field(ge=0)
    watchdog_triggered_at: datetime | None
    watchdog_triggered_monotonic_ns: int | None = Field(ge=0)
    jailer_reaped_at: datetime | None
    jailer_reaped_monotonic_ns: int | None = Field(ge=0)
    cgroup_empty_at: datetime | None
    cgroup_empty_monotonic_ns: int | None = Field(ge=0)
    cleanup_finished_at: datetime
    cleanup_finished_monotonic_ns: int = Field(ge=0)
    lifecycle: Literal['terminated', 'never_launched']
    jailer_exit_code: int | None
    wall_watchdog_armed: bool
    wall_timeout_triggered: bool
    process_group_exit_verified: Literal[True] = True
    cgroup_removed: Literal[True] = True
    jail_root_removed: Literal[True] = True
    vsock_removed: Literal[True] = True

    @field_validator(
        'watchdog_triggered_at',
        'jailer_reaped_at',
        'cgroup_empty_at',
        'cleanup_finished_at',
    )
    @classmethod
    def validate_observed_at(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_lifecycle(self) -> Self:
        if self.lifecycle == 'terminated':
            required = (
                self.launched_monotonic_ns,
                self.wall_deadline_monotonic_ns,
                self.jailer_reaped_at,
                self.jailer_reaped_monotonic_ns,
                self.cgroup_empty_at,
                self.cgroup_empty_monotonic_ns,
            )
            if (
                self.jailer_exit_code is None
                or not self.wall_watchdog_armed
                or any(value is None for value in required)
            ):
                raise ValueError('terminated workers require observed launch, reap, cgroup-empty, and cleanup times')
            assert self.launched_monotonic_ns is not None
            assert self.wall_deadline_monotonic_ns is not None
            assert self.jailer_reaped_monotonic_ns is not None
            assert self.cgroup_empty_monotonic_ns is not None
            if not (
                self.launched_monotonic_ns
                <= self.jailer_reaped_monotonic_ns
                <= self.cgroup_empty_monotonic_ns
                <= self.cleanup_finished_monotonic_ns
            ):
                raise ValueError('worker cleanup monotonic observations are out of order')
            if self.wall_deadline_monotonic_ns <= self.launched_monotonic_ns:
                raise ValueError('worker wall deadline must follow launch on the monotonic clock')
            trigger_pair = (self.watchdog_triggered_at, self.watchdog_triggered_monotonic_ns)
            if self.wall_timeout_triggered:
                if any(value is None for value in trigger_pair):
                    raise ValueError('timed-out workers require an observed watchdog trigger')
                assert self.watchdog_triggered_monotonic_ns is not None
                if not (
                    self.wall_deadline_monotonic_ns
                    <= self.watchdog_triggered_monotonic_ns
                    <= self.cgroup_empty_monotonic_ns
                ):
                    raise ValueError(
                        'watchdog trigger must be observed at or after the deadline and before cgroup exit'
                    )
            elif any(value is not None for value in trigger_pair):
                raise ValueError('a worker without a wall timeout cannot report a watchdog trigger')
        elif (
            self.launched_monotonic_ns is not None
            or self.wall_deadline_monotonic_ns is not None
            or self.watchdog_triggered_at is not None
            or self.watchdog_triggered_monotonic_ns is not None
            or self.jailer_reaped_at is not None
            or self.jailer_reaped_monotonic_ns is not None
            or self.cgroup_empty_at is not None
            or self.cgroup_empty_monotonic_ns is not None
            or self.jailer_exit_code is not None
            or self.wall_watchdog_armed
            or self.wall_timeout_triggered
        ):
            raise ValueError('an unlaunched worker cannot report an exit code or wall watchdog activity')
        return self


class FirecrackerWorkerAttestation(StrictModel):
    """Trusted-supervisor account of one complete, cleaned-up microVM execution.

    ``finished_at`` is the UTC observation that the pinned cgroup was empty, not a claim about the
    unobservable instruction at which guest execution ended.  ``duration_ms`` is derived from the
    corresponding monotonic observations, so asynchronous cgroup teardown is never backdated to the
    nominal watchdog deadline.
    """

    schema_version: Literal['vaxreplay.firecracker-worker-attestation.v0.2'] = (
        FIRECRACKER_WORKER_ATTESTATION_SCHEMA_VERSION
    )
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    host_preflight_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepared_worker_sha256: str = Field(pattern=_SHA256_PATTERN)
    cleanup_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    jailer_argv_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_release: str = Field(min_length=1, max_length=200)
    firecracker_sha256: str = Field(pattern=_SHA256_PATTERN)
    jailer_sha256: str = Field(pattern=_SHA256_PATTERN)
    kernel_sha256: str = Field(pattern=_SHA256_PATTERN)
    rootfs_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_sha256: str = Field(pattern=_SHA256_PATTERN)
    initial_scratch_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_cid: int = Field(ge=3, le=2**32 - 1)
    guest_rpc_port: int = Field(ge=1, le=2**32 - 1)
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)
    launched_monotonic_ns: int = Field(ge=0)
    wall_deadline_monotonic_ns: int = Field(gt=0)
    watchdog_triggered_at: datetime | None
    watchdog_triggered_monotonic_ns: int | None = Field(ge=0)
    jailer_reaped_at: datetime
    jailer_reaped_monotonic_ns: int = Field(ge=0)
    cgroup_empty_at: datetime
    cgroup_empty_monotonic_ns: int = Field(ge=0)
    cleanup_finished_at: datetime
    cleanup_finished_monotonic_ns: int = Field(ge=0)
    jailer_exit_code: int
    wall_seconds: int = Field(ge=1, le=86_400)
    wall_timeout_triggered: bool
    cgroup_version: Literal[2] = 2
    cgroup_cpu_memory_pids_enforced: Literal[True] = True
    new_pid_namespace: Literal[True] = True
    no_api: Literal[True] = True
    network_interfaces_absent: Literal[True] = True
    mmds_absent: Literal[True] = True
    rootfs_read_only: Literal[True] = True
    harness_read_only: Literal[True] = True
    fresh_bounded_writable_scratch: Literal[True] = True
    wall_watchdog_armed: Literal[True] = True
    watchdog_stopped_verified: Literal[True] = True
    watchdog_failure_absent: Literal[True] = True
    process_group_exit_verified: Literal[True] = True
    cgroup_removed: Literal[True] = True
    jail_root_removed: Literal[True] = True
    vsock_removed: Literal[True] = True

    @field_validator(
        'started_at',
        'finished_at',
        'watchdog_triggered_at',
        'jailer_reaped_at',
        'cgroup_empty_at',
        'cleanup_finished_at',
    )
    @classmethod
    def validate_time(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.finished_at != self.cgroup_empty_at:
            raise ValueError('worker finish must be the retained cgroup-empty UTC observation')
        if not (
            self.launched_monotonic_ns
            <= self.jailer_reaped_monotonic_ns
            <= self.cgroup_empty_monotonic_ns
            <= self.cleanup_finished_monotonic_ns
        ):
            raise ValueError('worker attestation monotonic observations are out of order')
        if self.wall_deadline_monotonic_ns - self.launched_monotonic_ns != self.wall_seconds * 1_000_000_000:
            raise ValueError('worker monotonic deadline must implement the pinned wall limit exactly')
        expected_duration = round((self.cgroup_empty_monotonic_ns - self.launched_monotonic_ns) / 1_000_000)
        if self.duration_ms != expected_duration:
            raise ValueError('worker attestation duration must match its monotonic observations')
        trigger_pair = (self.watchdog_triggered_at, self.watchdog_triggered_monotonic_ns)
        if self.wall_timeout_triggered:
            if any(value is None for value in trigger_pair):
                raise ValueError('timed-out worker attestation is missing its observed watchdog trigger')
            assert self.watchdog_triggered_monotonic_ns is not None
            if not (
                self.wall_deadline_monotonic_ns
                <= self.watchdog_triggered_monotonic_ns
                <= self.cgroup_empty_monotonic_ns
            ):
                raise ValueError('worker watchdog trigger is outside its observed monotonic interval')
        elif any(value is not None for value in trigger_pair):
            raise ValueError('worker without a wall timeout cannot retain watchdog-trigger observations')
        return self


class AuthenticatedFirecrackerWorkerAttestation(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-firecracker-worker-attestation.v0.2'] = (
        AUTHENTICATED_FIRECRACKER_WORKER_ATTESTATION_SCHEMA_VERSION
    )
    attestation: FirecrackerWorkerAttestation
    authentication: Literal['hmac-sha256-domain-separated'] = 'hmac-sha256-domain-separated'
    attestation_key_id: str = Field(pattern=_SHA256_PATTERN)
    attestation_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True)
class _ProcProcessIdentity:
    pid: int
    state: str
    parent_pid: int
    process_group_id: int
    session_id: int
    start_time_ticks: int


@dataclass(frozen=True)
class _ObservedFirecrackerChild:
    pid: int
    parent_pid_at_observation: int
    process_group_id: int
    session_id: int
    start_time_ticks: int
    executable_sha256: str
    pid_file_path: str
    pid_file_device_id: int
    pid_file_inode: int
    pidfd: int
    cgroup_descriptor: int
    cgroup_device_id: int
    cgroup_inode: int
    jailer_reaped_at: datetime
    jailer_reaped_monotonic_ns: int


@dataclass(frozen=True)
class _LifecycleObservation:
    observed_at: datetime
    monotonic_ns: int


class _WatchdogTiming:
    """Thread-safe, write-once retention of the watchdog's actual trigger observation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._triggered: _LifecycleObservation | None = None

    def record_triggered(self) -> _LifecycleObservation:
        observation = _observe_lifecycle_time()
        with self._lock:
            if self._triggered is None:
                self._triggered = observation
            return self._triggered

    def triggered(self) -> _LifecycleObservation | None:
        with self._lock:
            return self._triggered


@dataclass(frozen=True)
class RunningFirecrackerWorker:
    """Locally tracked jailer bootstrap plus its distinct Firecracker child.

    With ``--new-pid-ns`` the jailer writes ``<exec-file>.pid`` and exits after cloning
    Firecracker.  ``process`` therefore identifies only that short-lived, directly reapable jailer
    parent.  The remaining fields bind the persistent Firecracker child which owns the vsock peer
    and stays in the dedicated process group until the VM exits or the watchdog kills it.
    """

    prepared: FirecrackerPreparedWorker
    process: subprocess.Popen[bytes]
    jailer_start_time_ticks: int
    jailer_process_group_id: int
    jailer_session_id: int
    firecracker_pid: int
    firecracker_parent_pid_at_observation: int
    firecracker_process_group_id: int
    firecracker_session_id: int
    firecracker_start_time_ticks: int
    firecracker_executable_sha256: str
    firecracker_pid_file_path: str
    firecracker_pid_file_device_id: int
    firecracker_pid_file_inode: int
    firecracker_pidfd: int
    cgroup_descriptor: int
    cgroup_device_id: int
    cgroup_inode: int
    jailer_reaped_at: datetime
    jailer_reaped_monotonic_ns: int
    identity_descriptors_closed: threading.Event
    launched_at: datetime
    launched_at_monotonic: float
    launched_at_monotonic_ns: int
    wall_deadline_monotonic: float
    wall_deadline_monotonic_ns: int
    watchdog_stop: threading.Event
    watchdog_timeout_triggered: threading.Event
    watchdog_failure: threading.Event
    watchdog_timing: _WatchdogTiming
    watchdog_thread: threading.Thread


def firecracker_model_sha256(model: StrictModel) -> str:
    return hashlib.sha256(canonical_json_bytes(model)).hexdigest()


def firecracker_attestation_key_id(attestation_key: bytes) -> str:
    _validate_attestation_key(attestation_key)
    return hashlib.sha256(_ATTESTATION_KEY_ID_DOMAIN + attestation_key).hexdigest()


def authenticated_firecracker_worker_attestation_sha256(
    authenticated: AuthenticatedFirecrackerWorkerAttestation,
) -> str:
    return firecracker_model_sha256(authenticated)


def finalize_firecracker_worker_attestation(
    *,
    spec: FirecrackerWorkerSpec,
    running: RunningFirecrackerWorker,
    cleanup: FirecrackerCleanupReceipt,
    attempt_reservation_sha256: str,
    attestation_key: bytes,
    expected_attestation_key_id: str,
) -> AuthenticatedFirecrackerWorkerAttestation:
    """Authenticate the exact worker lifecycle after process-group and jail cleanup."""

    if len(attempt_reservation_sha256) != 64 or any(
        character not in '0123456789abcdef' for character in attempt_reservation_sha256
    ):
        raise FirecrackerAttestationError('attempt reservation commitment must be a lowercase SHA-256 digest')
    key_id = firecracker_attestation_key_id(attestation_key)
    if not hmac.compare_digest(key_id, expected_attestation_key_id):
        raise FirecrackerAttestationError('worker attestation key does not match the release-pinned key ID')

    prepared = running.prepared
    worker_spec_sha256 = firecracker_model_sha256(spec)
    if prepared.worker_spec_sha256 != worker_spec_sha256:
        raise FirecrackerAttestationError('prepared worker is bound to a different worker specification')
    if prepared.host_preflight.worker_spec_sha256 != worker_spec_sha256:
        raise FirecrackerAttestationError('host preflight is bound to a different worker specification')
    if firecracker_model_sha256(prepared.host_preflight) != prepared.host_preflight_sha256:
        raise FirecrackerAttestationError('embedded host preflight does not match its prepared-worker commitment')
    expected_prepared_hashes = (
        (prepared.config_sha256, hashlib.sha256(firecracker_static_config_bytes(spec)).hexdigest()),
        (prepared.kernel_sha256, spec.images.kernel.sha256),
        (prepared.rootfs_sha256, spec.images.rootfs.sha256),
        (prepared.harness_sha256, spec.images.harness.sha256),
        (prepared.initial_scratch_sha256, spec.images.scratch_template.sha256),
    )
    if any(observed != expected for observed, expected in expected_prepared_hashes):
        raise FirecrackerAttestationError('prepared worker artifact commitment differs from the pinned specification')
    if cleanup.run_id != prepared.run_id or cleanup.lifecycle != 'terminated':
        raise FirecrackerAttestationError('cleanup receipt does not terminate this worker run')
    expected_pid_file = Path(prepared.jail_root) / f'{Path(spec.runtime.firecracker.source_path).name}.pid'
    if (
        running.process.pid <= 1
        or running.firecracker_pid <= 1
        or running.process.pid == running.firecracker_pid
        or running.jailer_process_group_id != running.process.pid
        or running.firecracker_process_group_id != running.jailer_process_group_id
        or running.jailer_session_id == running.process.pid
        or running.firecracker_session_id != running.jailer_session_id
        or running.firecracker_start_time_ticks < running.jailer_start_time_ticks
        or running.firecracker_executable_sha256 != spec.runtime.firecracker.sha256
        or Path(running.firecracker_pid_file_path) != expected_pid_file
        or running.firecracker_pid_file_device_id <= 0
        or running.firecracker_pid_file_inode <= 0
    ):
        raise FirecrackerAttestationError('running worker does not bind a distinct pinned Firecracker child')
    jailer_exit_code = cleanup.jailer_exit_code
    if jailer_exit_code is None or running.process.poll() is None or running.process.returncode != jailer_exit_code:
        raise FirecrackerAttestationError('cleanup receipt does not match the reaped jailer exit status')
    cgroup_empty_at = cleanup.cgroup_empty_at
    cgroup_empty_monotonic_ns = cleanup.cgroup_empty_monotonic_ns
    jailer_reaped_at = cleanup.jailer_reaped_at
    jailer_reaped_monotonic_ns = cleanup.jailer_reaped_monotonic_ns
    if cgroup_empty_at is None or cgroup_empty_monotonic_ns is None:
        raise FirecrackerAttestationError('terminated worker cleanup is missing its cgroup-empty observation')
    if jailer_reaped_at is None or jailer_reaped_monotonic_ns is None:
        raise FirecrackerAttestationError('terminated worker cleanup is missing its jailer-reap observation')
    if prepared.created_at > running.launched_at:
        raise FirecrackerAttestationError('worker lifecycle timestamps are inconsistent')
    if (
        running.wall_deadline_monotonic_ns - running.launched_at_monotonic_ns
        != spec.limits.wall_seconds * 1_000_000_000
        or cleanup.launched_monotonic_ns != running.launched_at_monotonic_ns
        or cleanup.wall_deadline_monotonic_ns != running.wall_deadline_monotonic_ns
    ):
        raise FirecrackerAttestationError('worker watchdog deadline differs from the pinned wall limit')
    if jailer_reaped_at != running.jailer_reaped_at or jailer_reaped_monotonic_ns != running.jailer_reaped_monotonic_ns:
        raise FirecrackerAttestationError('cleanup receipt differs from the actual jailer reap observation')

    if not running.watchdog_stop.is_set() or running.watchdog_thread.is_alive():
        raise FirecrackerAttestationError('worker watchdog has not been stopped and joined')
    if running.watchdog_failure.is_set():
        raise FirecrackerAttestationError('worker watchdog recorded a process-group signaling failure')
    if cleanup.wall_timeout_triggered != running.watchdog_timeout_triggered.is_set():
        raise FirecrackerAttestationError('cleanup timeout claim differs from the runtime watchdog event')
    watchdog_triggered = running.watchdog_timing.triggered()
    receipt_trigger = (
        cleanup.watchdog_triggered_at,
        cleanup.watchdog_triggered_monotonic_ns,
    )
    observed_trigger = (
        None if watchdog_triggered is None else watchdog_triggered.observed_at,
        None if watchdog_triggered is None else watchdog_triggered.monotonic_ns,
    )
    if receipt_trigger != observed_trigger:
        raise FirecrackerAttestationError('cleanup receipt differs from the actual watchdog trigger observation')
    if not running.identity_descriptors_closed.is_set():
        raise FirecrackerAttestationError('Firecracker pidfd and cgroup descriptor remain open after cleanup')

    run_container, expected_jail_root = _expected_jail_paths(spec, prepared.run_id)
    expected_cgroup = _expected_cgroup_path(spec, prepared.run_id)
    expected_run_directory = expected_jail_root / 'run'
    expected_vsock = expected_run_directory / 'vsock.sock'
    if Path(prepared.jail_root) != expected_jail_root or Path(prepared.vsock_uds_path) != expected_vsock:
        raise FirecrackerAttestationError('prepared worker cleanup paths differ from the exact run paths')
    for label, path in (
        ('cgroup', expected_cgroup),
        ('run container', run_container),
        ('jail root', expected_jail_root),
        ('run directory', expected_run_directory),
        ('vsock endpoint', expected_vsock),
    ):
        if not _path_absent_exact(path, label=label):
            raise FirecrackerAttestationError(f'{label} remains after claimed worker cleanup')

    duration_ms = round((cgroup_empty_monotonic_ns - running.launched_at_monotonic_ns) / 1_000_000)
    attestation = FirecrackerWorkerAttestation(
        run_id=prepared.run_id,
        attempt_reservation_sha256=attempt_reservation_sha256,
        worker_spec_sha256=worker_spec_sha256,
        host_preflight_sha256=prepared.host_preflight_sha256,
        prepared_worker_sha256=firecracker_model_sha256(prepared),
        cleanup_receipt_sha256=firecracker_model_sha256(cleanup),
        jailer_argv_sha256=hashlib.sha256(
            canonical_json_bytes(list(build_jailer_argv(spec=spec, run_id=prepared.run_id)))
        ).hexdigest(),
        runtime_release=spec.runtime.release,
        firecracker_sha256=spec.runtime.firecracker.sha256,
        jailer_sha256=spec.runtime.jailer.sha256,
        kernel_sha256=spec.images.kernel.sha256,
        rootfs_sha256=spec.images.rootfs.sha256,
        harness_sha256=spec.images.harness.sha256,
        initial_scratch_sha256=spec.images.scratch_template.sha256,
        config_sha256=prepared.config_sha256,
        guest_cid=spec.guest_cid,
        guest_rpc_port=spec.guest_rpc_port,
        started_at=running.launched_at,
        finished_at=cgroup_empty_at,
        duration_ms=duration_ms,
        launched_monotonic_ns=running.launched_at_monotonic_ns,
        wall_deadline_monotonic_ns=running.wall_deadline_monotonic_ns,
        watchdog_triggered_at=cleanup.watchdog_triggered_at,
        watchdog_triggered_monotonic_ns=cleanup.watchdog_triggered_monotonic_ns,
        jailer_reaped_at=jailer_reaped_at,
        jailer_reaped_monotonic_ns=jailer_reaped_monotonic_ns,
        cgroup_empty_at=cgroup_empty_at,
        cgroup_empty_monotonic_ns=cgroup_empty_monotonic_ns,
        cleanup_finished_at=cleanup.cleanup_finished_at,
        cleanup_finished_monotonic_ns=cleanup.cleanup_finished_monotonic_ns,
        jailer_exit_code=jailer_exit_code,
        wall_seconds=spec.limits.wall_seconds,
        wall_timeout_triggered=cleanup.wall_timeout_triggered,
    )
    authentication = hmac.new(
        attestation_key,
        _ATTESTATION_HMAC_DOMAIN + canonical_json_bytes(attestation),
        hashlib.sha256,
    ).hexdigest()
    return AuthenticatedFirecrackerWorkerAttestation(
        attestation=attestation,
        attestation_key_id=key_id,
        attestation_hmac_sha256=authentication,
    )


def verify_firecracker_worker_attestation(
    authenticated: AuthenticatedFirecrackerWorkerAttestation,
    *,
    attestation_key: bytes,
    expected_attestation_key_id: str,
    expected_run_id: str,
    expected_attempt_reservation_sha256: str,
    expected_worker_spec_sha256: str,
) -> FirecrackerWorkerAttestation:
    """Verify HMAC authentication and all caller-required replay bindings."""

    key_id = firecracker_attestation_key_id(attestation_key)
    if not hmac.compare_digest(key_id, expected_attestation_key_id) or not hmac.compare_digest(
        authenticated.attestation_key_id, expected_attestation_key_id
    ):
        raise FirecrackerAttestationError('worker attestation key ID mismatch')
    expected_hmac = hmac.new(
        attestation_key,
        _ATTESTATION_HMAC_DOMAIN + canonical_json_bytes(authenticated.attestation),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hmac, authenticated.attestation_hmac_sha256):
        raise FirecrackerAttestationError('worker attestation HMAC verification failed')
    attestation = authenticated.attestation
    if attestation.run_id != expected_run_id:
        raise FirecrackerAttestationError('worker attestation run ID mismatch')
    if attestation.attempt_reservation_sha256 != expected_attempt_reservation_sha256:
        raise FirecrackerAttestationError('worker attestation attempt reservation mismatch')
    if attestation.worker_spec_sha256 != expected_worker_spec_sha256:
        raise FirecrackerAttestationError('worker attestation specification mismatch')
    return attestation


def _validate_attestation_key(attestation_key: bytes) -> None:
    if not isinstance(attestation_key, bytes) or len(attestation_key) < 32:
        raise FirecrackerAttestationError('worker attestation key must contain at least 32 bytes')


def build_firecracker_static_config(spec: FirecrackerWorkerSpec) -> FirecrackerStaticConfig:
    """Build the complete device allowlist used by ``--no-api`` Firecracker."""

    return FirecrackerStaticConfig.model_validate(
        {
            'boot-source': {
                'kernel_image_path': '/kernel.bin',
                'boot_args': spec.kernel_boot_args,
            },
            'drives': (
                {
                    'drive_id': 'rootfs',
                    'path_on_host': '/rootfs.ext4',
                    'is_root_device': True,
                    'is_read_only': True,
                },
                {
                    'drive_id': 'harness',
                    'path_on_host': '/harness.ext4',
                    'is_root_device': False,
                    'is_read_only': True,
                },
                {
                    'drive_id': 'scratch',
                    'path_on_host': '/scratch.ext4',
                    'is_root_device': False,
                    'is_read_only': False,
                },
            ),
            'machine-config': {
                'vcpu_count': spec.limits.vcpu_count,
                'mem_size_mib': spec.limits.memory_mib,
                'smt': False,
                'track_dirty_pages': False,
            },
            'vsock': {
                'guest_cid': spec.guest_cid,
                'uds_path': '/run/vsock.sock',
            },
        }
    )


def firecracker_static_config_bytes(spec: FirecrackerWorkerSpec) -> bytes:
    config = build_firecracker_static_config(spec)
    return canonical_json_bytes(config.model_dump(mode='json', by_alias=True))


def build_jailer_argv(*, spec: FirecrackerWorkerSpec, run_id: str) -> tuple[str, ...]:
    """Return a shell-free jailer invocation with host and guest resource limits."""

    _validate_run_id(run_id)
    memory_bytes = spec.limits.memory_mib * _MIB
    return (
        spec.runtime.jailer.source_path,
        '--id',
        run_id,
        '--exec-file',
        spec.runtime.firecracker.source_path,
        '--uid',
        str(spec.worker_uid),
        '--gid',
        str(spec.worker_gid),
        '--chroot-base-dir',
        spec.chroot_base_dir,
        '--cgroup-version',
        '2',
        '--parent-cgroup',
        spec.cgroup_parent,
        '--cgroup',
        f'cpu.max={spec.limits.cpu_quota_us} {spec.limits.cpu_period_us}',
        '--cgroup',
        f'memory.max={memory_bytes}',
        '--cgroup',
        'memory.swap.max=0',
        '--cgroup',
        f'pids.max={spec.limits.pids}',
        '--resource-limit',
        f'no-file={spec.limits.open_files}',
        '--resource-limit',
        f'fsize={spec.limits.scratch_bytes}',
        '--new-pid-ns',
        '--',
        '--no-api',
        '--config-file',
        '/firecracker-config.json',
    )


def preflight_firecracker_host(spec: FirecrackerWorkerSpec) -> FirecrackerHostPreflightReceipt:
    """Verify the real host and every source artifact; there is no permissive fallback."""

    host_os = platform.system()
    if host_os != 'Linux':
        raise FirecrackerPreflightError('Firecracker execution requires a Linux host')
    host_architecture = _normalized_architecture(platform.machine())
    if host_architecture != spec.runtime.architecture:
        raise FirecrackerPreflightError('host architecture does not match the pinned Firecracker runtime')
    if os.geteuid() != 0:
        raise FirecrackerPreflightError('Firecracker jail preparation requires effective UID 0')

    kvm = Path('/dev/kvm')
    try:
        kvm_stat = kvm.lstat()
    except OSError as error:
        raise FirecrackerPreflightError('/dev/kvm is unavailable') from error
    if stat.S_ISLNK(kvm_stat.st_mode) or not stat.S_ISCHR(kvm_stat.st_mode):
        raise FirecrackerPreflightError('/dev/kvm must be a non-symlink character device')
    if not os.access(kvm, os.R_OK | os.W_OK):
        raise FirecrackerPreflightError('/dev/kvm must be readable and writable')

    controllers_path = Path('/sys/fs/cgroup/cgroup.controllers')
    try:
        if controllers_path.is_symlink() or not controllers_path.is_file():
            raise FirecrackerPreflightError('the unified cgroup v2 controller file is unavailable')
        controller_bytes = controllers_path.read_bytes()
    except OSError as error:
        raise FirecrackerPreflightError('cannot read cgroup v2 controllers') from error
    if len(controller_bytes) > 4096:
        raise FirecrackerPreflightError('cgroup controller inventory is unexpectedly large')
    try:
        controllers = tuple(sorted(controller_bytes.decode('ascii').split()))
    except UnicodeDecodeError as error:
        raise FirecrackerPreflightError('cgroup controller inventory is not ASCII') from error
    if not _REQUIRED_CGROUP_CONTROLLERS.issubset(controllers):
        raise FirecrackerPreflightError('cgroup v2 must expose cpu, memory, and pids controllers')

    _verify_trusted_directory(Path(spec.chroot_base_dir), expected_owner=0)
    for artifact in _all_artifacts(spec):
        _verify_trusted_artifact(artifact, executable=artifact.role in _EXECUTABLE_ROLES)

    return FirecrackerHostPreflightReceipt(
        worker_spec_sha256=firecracker_model_sha256(spec),
        collected_at=datetime.now(UTC),
        host_architecture=host_architecture,
        host_kernel_release=platform.release(),
        cgroup_controllers=controllers,
    )


class FirecrackerSupervisor:
    """Prepare one fresh jail, launch one foreground jailer, and prove teardown."""

    def __init__(self, spec: FirecrackerWorkerSpec):
        self._spec = spec

    @property
    def spec(self) -> FirecrackerWorkerSpec:
        return self._spec

    def preflight(self) -> FirecrackerHostPreflightReceipt:
        return preflight_firecracker_host(self._spec)

    def prepare(self, *, run_id: str) -> FirecrackerPreparedWorker:
        _validate_run_id(run_id)
        preflight = self.preflight()
        expected_preflight_spec = firecracker_model_sha256(self._spec)
        if preflight.worker_spec_sha256 != expected_preflight_spec:
            raise FirecrackerPreparationError('preflight receipt is bound to a different worker specification')

        run_container, jail_root = _expected_jail_paths(self._spec, run_id)
        created_run_container = False
        try:
            expected_cgroup = _expected_cgroup_path(self._spec, run_id)
            if expected_cgroup.exists() or expected_cgroup.is_symlink():
                raise FirecrackerPreparationError('run cgroup already exists; run IDs are never reused')
            executable_dir = run_container.parent
            if not executable_dir.exists():
                executable_dir.mkdir(mode=0o700)
            _verify_trusted_directory(executable_dir, expected_owner=os.geteuid())
            if run_container.exists() or run_container.is_symlink():
                raise FirecrackerPreparationError('run jail already exists; run IDs are never reused')
            run_container.mkdir(mode=0o700)
            created_run_container = True
            jail_root.mkdir(mode=0o755)
            run_directory = jail_root / 'run'
            run_directory.mkdir(mode=0o700)
            os.chown(run_directory, self._spec.worker_uid, self._spec.worker_gid)

            copied = (
                (self._spec.images.kernel, jail_root / 'kernel.bin', 0o400),
                (self._spec.images.rootfs, jail_root / 'rootfs.ext4', 0o400),
                (self._spec.images.harness, jail_root / 'harness.ext4', 0o400),
                (self._spec.images.scratch_template, jail_root / 'scratch.ext4', 0o600),
            )
            for identity, destination, mode in copied:
                _copy_verified(identity, destination, mode=mode)
                os.chown(destination, self._spec.worker_uid, self._spec.worker_gid)

            config_bytes = firecracker_static_config_bytes(self._spec)
            config_path = jail_root / 'firecracker-config.json'
            _write_new_file(config_path, config_bytes, mode=0o400)
            os.chown(config_path, self._spec.worker_uid, self._spec.worker_gid)

            prepared = FirecrackerPreparedWorker(
                run_id=run_id,
                worker_spec_sha256=expected_preflight_spec,
                host_preflight=preflight,
                host_preflight_sha256=firecracker_model_sha256(preflight),
                jail_root=str(jail_root),
                config_path=str(config_path),
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                kernel_sha256=self._spec.images.kernel.sha256,
                rootfs_sha256=self._spec.images.rootfs.sha256,
                harness_sha256=self._spec.images.harness.sha256,
                initial_scratch_sha256=self._spec.images.scratch_template.sha256,
                vsock_uds_path=str(run_directory / 'vsock.sock'),
                created_at=datetime.now(UTC),
            )
            self.verify_prepared(prepared)
            return prepared
        except Exception as error:
            if created_run_container and not _remove_tree_exact(run_container):
                raise FirecrackerCleanupError('cannot prove cleanup of a partially prepared jail') from error
            if isinstance(error, FirecrackerWorkerError):
                raise
            raise FirecrackerPreparationError('cannot prepare the Firecracker jail') from error

    def verify_prepared(
        self,
        prepared: FirecrackerPreparedWorker,
        *,
        prebound_guest_listener: FirecrackerPreboundGuestListener | None = None,
    ) -> None:
        """Recheck exact pre-launch bytes and inventory; reject caller-created receipts."""

        if prepared.worker_spec_sha256 != firecracker_model_sha256(self._spec):
            raise FirecrackerPreparationError('prepared worker is bound to a different specification')
        expected_container, expected_root = _expected_jail_paths(self._spec, prepared.run_id)
        if Path(prepared.jail_root) != expected_root:
            raise FirecrackerPreparationError('prepared jail root is not the expected run-specific path')
        if not expected_container.is_dir() or expected_container.is_symlink():
            raise FirecrackerPreparationError('prepared run container is missing or unsafe')
        if not expected_root.is_dir() or expected_root.is_symlink():
            raise FirecrackerPreparationError('prepared jail root is missing or unsafe')
        expected_receipt_hashes = (
            (prepared.kernel_sha256, self._spec.images.kernel.sha256),
            (prepared.rootfs_sha256, self._spec.images.rootfs.sha256),
            (prepared.harness_sha256, self._spec.images.harness.sha256),
            (prepared.initial_scratch_sha256, self._spec.images.scratch_template.sha256),
        )
        if any(observed != expected for observed, expected in expected_receipt_hashes):
            raise FirecrackerPreparationError('prepared artifact commitment differs from the worker specification')

        expected_inventory = {
            'firecracker-config.json',
            'harness.ext4',
            'kernel.bin',
            'rootfs.ext4',
            'run',
            'scratch.ext4',
        }
        try:
            observed_inventory = {entry.name for entry in os.scandir(expected_root)}
        except OSError as error:
            raise FirecrackerPreparationError('cannot inventory prepared jail root') from error
        if observed_inventory != expected_inventory:
            raise FirecrackerPreparationError('prepared jail root contains an unexpected or missing entry')
        run_directory = expected_root / 'run'
        if run_directory.is_symlink() or not run_directory.is_dir():
            raise FirecrackerPreparationError('prepared vsock directory must be a real directory')
        run_stat = run_directory.stat()
        if (
            stat.S_IMODE(run_stat.st_mode) != 0o700
            or run_stat.st_uid != self._spec.worker_uid
            or run_stat.st_gid != self._spec.worker_gid
        ):
            raise FirecrackerPreparationError('prepared vsock directory mode and ownership must be pinned')
        if prebound_guest_listener is None:
            if any(os.scandir(run_directory)):
                raise FirecrackerPreparationError('prepared vsock directory must be empty before listener binding')
        else:
            self._verify_prebound_guest_listener(
                prepared,
                prebound_guest_listener,
                run_directory=run_directory,
            )

        expected_files = (
            ('kernel.bin', self._spec.images.kernel.sha256, self._spec.images.kernel.byte_count, 0o400),
            ('rootfs.ext4', self._spec.images.rootfs.sha256, self._spec.images.rootfs.byte_count, 0o400),
            ('harness.ext4', self._spec.images.harness.sha256, self._spec.images.harness.byte_count, 0o400),
            (
                'scratch.ext4',
                self._spec.images.scratch_template.sha256,
                self._spec.images.scratch_template.byte_count,
                0o600,
            ),
        )
        for name, expected_sha256, expected_bytes, expected_mode in expected_files:
            _verify_copied_file(
                expected_root / name,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
                expected_mode=expected_mode,
                expected_uid=self._spec.worker_uid,
                expected_gid=self._spec.worker_gid,
            )
        config_bytes = firecracker_static_config_bytes(self._spec)
        config_path = expected_root / 'firecracker-config.json'
        _verify_copied_file(
            config_path,
            expected_sha256=hashlib.sha256(config_bytes).hexdigest(),
            expected_bytes=len(config_bytes),
            expected_mode=0o400,
            expected_uid=self._spec.worker_uid,
            expected_gid=self._spec.worker_gid,
        )
        if prepared.config_sha256 != hashlib.sha256(config_bytes).hexdigest():
            raise FirecrackerPreparationError('prepared config commitment does not match the pinned specification')

    def _verify_prebound_guest_listener(
        self,
        prepared: FirecrackerPreparedWorker,
        listener: FirecrackerPreboundGuestListener,
        *,
        run_directory: Path,
    ) -> None:
        expected_path = Path(
            firecracker_guest_initiated_uds_path(
                uds_path=prepared.vsock_uds_path,
                port=self._spec.guest_rpc_port,
            )
        )
        if (
            listener.run_id != prepared.run_id
            or listener.worker_spec_sha256 != firecracker_model_sha256(self._spec)
            or listener.socket_path != str(expected_path)
            or expected_path.parent != run_directory
            or listener.worker_uid != self._spec.worker_uid
            or listener.worker_gid != self._spec.worker_gid
        ):
            raise FirecrackerPreparationError('prebound guest listener differs from the prepared worker')
        try:
            inventory = tuple(os.scandir(run_directory))
            metadata = expected_path.lstat()
        except OSError as error:
            raise FirecrackerPreparationError('cannot inspect the prebound guest listener') from error
        if len(inventory) != 1 or inventory[0].name != expected_path.name:
            raise FirecrackerPreparationError('prebound guest listener must be the only run-directory entry')
        if (
            not stat.S_ISSOCK(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (listener.device_id, listener.inode)
            or metadata.st_uid != listener.worker_uid
            or metadata.st_gid != listener.worker_gid
            or stat.S_IMODE(metadata.st_mode) != listener.mode
            or metadata.st_nlink != 1
        ):
            raise FirecrackerPreparationError('prebound guest listener identity changed before launch')

    def launch(
        self,
        prepared: FirecrackerPreparedWorker,
        *,
        prebound_guest_listener: FirecrackerPreboundGuestListener | None = None,
    ) -> RunningFirecrackerWorker:
        """Launch and bind the short-lived jailer parent to its Firecracker child.

        ``--new-pid-ns`` makes the jailer clone Firecracker, write ``firecracker.pid``, and exit.
        A dedicated process group (without making the jailer a session leader) keeps that child in
        one signalable group.  The pid file, procfs identity, executable digest, and cgroup binding
        are all observed before a worker is returned.
        """

        self.preflight()
        self.verify_prepared(prepared, prebound_guest_listener=prebound_guest_listener)
        argv = build_jailer_argv(spec=self._spec, run_id=prepared.run_id)
        jail_root_descriptor = _open_pinned_jail_root(Path(prepared.jail_root))
        process: subprocess.Popen[bytes] | None = None
        launched_at = datetime.now(UTC)
        launched_at_monotonic_ns = time.monotonic_ns()
        launched_at_monotonic = launched_at_monotonic_ns / 1_000_000_000
        try:
            process = subprocess.Popen(  # noqa: S603 - absolute digest-verified argv; never a shell
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd='/',
                env={'LANG': 'C', 'LC_ALL': 'C', 'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'},
                close_fds=True,
                shell=False,
                start_new_session=False,
                process_group=0,
                restore_signals=True,
                umask=0o077,
            )
            jailer_identity = _read_proc_process_identity(process.pid)
            if jailer_identity.process_group_id != process.pid or jailer_identity.session_id == process.pid:
                raise FirecrackerWorkerError(
                    'jailer bootstrap is not the dedicated non-session-leader process-group leader'
                )
            child = _observe_launched_firecracker_child(
                spec=self._spec,
                prepared=prepared,
                jailer_process=process,
                jailer_identity=jailer_identity,
                jail_root_descriptor=jail_root_descriptor,
                timeout_seconds=min(
                    _FIRECRACKER_CHILD_OBSERVATION_SECONDS,
                    float(self._spec.limits.wall_seconds),
                ),
            )
        except (OSError, FirecrackerWorkerError) as error:
            if process is not None:
                _abort_failed_launch(process, spec=self._spec, run_id=prepared.run_id)
            if not _remove_empty_cgroup_exact(_expected_cgroup_path(self._spec, prepared.run_id)):
                raise FirecrackerCleanupError('failed jailer left a nonempty Firecracker cgroup') from error
            if not _remove_tree_exact(_expected_jail_paths(self._spec, prepared.run_id)[0]):
                raise FirecrackerCleanupError('jailer failed and its prepared jail could not be removed') from error
            if isinstance(error, FirecrackerWorkerError):
                raise
            raise FirecrackerWorkerError('cannot launch the pinned Firecracker jailer') from error
        finally:
            os.close(jail_root_descriptor)
        assert process is not None  # narrowed by the successful launch path
        wall_deadline_monotonic_ns = launched_at_monotonic_ns + self._spec.limits.wall_seconds * 1_000_000_000
        wall_deadline_monotonic = wall_deadline_monotonic_ns / 1_000_000_000
        watchdog_stop = threading.Event()
        watchdog_timeout_triggered = threading.Event()
        watchdog_failure = threading.Event()
        watchdog_timing = _WatchdogTiming()
        identity_descriptors_closed = threading.Event()
        try:
            watchdog = threading.Thread(
                target=_watchdog_process_group,
                args=(
                    child.pid,
                    child.start_time_ticks,
                    child.process_group_id,
                    child.pidfd,
                    child.cgroup_descriptor,
                    child.cgroup_device_id,
                    child.cgroup_inode,
                    watchdog_stop,
                    watchdog_timeout_triggered,
                    watchdog_failure,
                    wall_deadline_monotonic_ns,
                    watchdog_timing,
                ),
                name=f'vaxreplay-firecracker-watchdog-{prepared.run_id}',
                daemon=True,
            )
            watchdog.start()
        except RuntimeError as error:
            try:
                _abort_failed_launch(process, spec=self._spec, run_id=prepared.run_id)
            finally:
                os.close(child.pidfd)
                os.close(child.cgroup_descriptor)
            if not _remove_empty_cgroup_exact(_expected_cgroup_path(self._spec, prepared.run_id)):
                raise FirecrackerCleanupError('watchdog startup failure left a nonempty cgroup') from error
            if not _remove_tree_exact(_expected_jail_paths(self._spec, prepared.run_id)[0]):
                raise FirecrackerCleanupError('watchdog startup failure left a worker jail') from error
            raise FirecrackerWorkerError('cannot arm the Firecracker wall-time watchdog') from error
        return RunningFirecrackerWorker(
            prepared=prepared,
            process=process,
            jailer_start_time_ticks=jailer_identity.start_time_ticks,
            jailer_process_group_id=jailer_identity.process_group_id,
            jailer_session_id=jailer_identity.session_id,
            firecracker_pid=child.pid,
            firecracker_parent_pid_at_observation=child.parent_pid_at_observation,
            firecracker_process_group_id=child.process_group_id,
            firecracker_session_id=child.session_id,
            firecracker_start_time_ticks=child.start_time_ticks,
            firecracker_executable_sha256=child.executable_sha256,
            firecracker_pid_file_path=child.pid_file_path,
            firecracker_pid_file_device_id=child.pid_file_device_id,
            firecracker_pid_file_inode=child.pid_file_inode,
            firecracker_pidfd=child.pidfd,
            cgroup_descriptor=child.cgroup_descriptor,
            cgroup_device_id=child.cgroup_device_id,
            cgroup_inode=child.cgroup_inode,
            jailer_reaped_at=child.jailer_reaped_at,
            jailer_reaped_monotonic_ns=child.jailer_reaped_monotonic_ns,
            identity_descriptors_closed=identity_descriptors_closed,
            launched_at=launched_at,
            launched_at_monotonic=launched_at_monotonic,
            launched_at_monotonic_ns=launched_at_monotonic_ns,
            wall_deadline_monotonic=wall_deadline_monotonic,
            wall_deadline_monotonic_ns=wall_deadline_monotonic_ns,
            watchdog_stop=watchdog_stop,
            watchdog_timeout_triggered=watchdog_timeout_triggered,
            watchdog_failure=watchdog_failure,
            watchdog_timing=watchdog_timing,
            watchdog_thread=watchdog,
        )

    def wait_for_exit(self, running: RunningFirecrackerWorker, *, timeout_seconds: float) -> bool:
        """Wait for the bound Firecracker child, never the already-exited jailer parent."""

        if timeout_seconds < 0 or timeout_seconds > self._spec.limits.wall_seconds:
            raise ValueError('Firecracker exit wait is outside the worker wall limit')
        deadline = time.monotonic() + timeout_seconds
        while True:
            if not _bound_firecracker_process_alive(running):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.02, remaining))

    def terminate_and_cleanup(
        self,
        running: RunningFirecrackerWorker,
        *,
        grace_seconds: float = 5.0,
    ) -> FirecrackerCleanupReceipt:
        """Stop the pidfd-bound child and pinned cgroup before removing run state."""

        if running.prepared.worker_spec_sha256 != firecracker_model_sha256(self._spec):
            raise FirecrackerCleanupError('running worker belongs to a different supervisor specification')
        if grace_seconds <= 0 or grace_seconds > 60:
            raise ValueError('cleanup grace period must be greater than zero and at most 60 seconds')
        termination_error: BaseException | None = None
        cgroup_empty: _LifecycleObservation | None = None
        try:
            cgroup_empty = _terminate_process_group(running, grace_seconds=grace_seconds)
        except BaseException as error:
            termination_error = error
        finally:
            running.watchdog_stop.set()
            running.watchdog_thread.join(timeout=1)
        if running.watchdog_thread.is_alive():
            raise FirecrackerCleanupError('cannot prove shutdown of the Firecracker wall-time watchdog')
        if not running.identity_descriptors_closed.is_set():
            try:
                os.close(running.firecracker_pidfd)
                os.close(running.cgroup_descriptor)
            except OSError as error:
                raise FirecrackerCleanupError('cannot close Firecracker identity descriptors') from error
            running.identity_descriptors_closed.set()
        if termination_error is not None:
            if isinstance(termination_error, FirecrackerCleanupError):
                raise termination_error
            raise FirecrackerCleanupError('cannot terminate the pinned Firecracker cgroup') from termination_error
        if running.watchdog_failure.is_set():
            raise FirecrackerCleanupError('Firecracker wall-time watchdog could not kill the pinned cgroup')
        try:
            jailer_exit_code = running.process.wait(timeout=max(0.1, grace_seconds))
        except subprocess.TimeoutExpired as error:
            raise FirecrackerCleanupError('cannot reap the Firecracker jailer bootstrap') from error
        if jailer_exit_code is None:
            raise FirecrackerCleanupError('terminated Firecracker jailer has no exit status')
        wall_timeout_triggered = running.watchdog_timeout_triggered.is_set()
        watchdog_triggered = running.watchdog_timing.triggered()
        if wall_timeout_triggered != (watchdog_triggered is not None):
            raise FirecrackerCleanupError('Firecracker watchdog event is missing its actual trigger observation')
        if cgroup_empty is None:
            raise FirecrackerCleanupError('cannot retain a cgroup-empty observation for the terminated worker')

        if not _remove_empty_cgroup_exact(_expected_cgroup_path(self._spec, running.prepared.run_id)):
            raise FirecrackerCleanupError('cannot prove removal of the Firecracker run cgroup')
        run_container, _ = _expected_jail_paths(self._spec, running.prepared.run_id)
        if not _remove_tree_exact(run_container):
            raise FirecrackerCleanupError('cannot prove removal of the Firecracker run jail')
        cleanup_finished = _observe_lifecycle_time()
        return FirecrackerCleanupReceipt(
            run_id=running.prepared.run_id,
            launched_monotonic_ns=running.launched_at_monotonic_ns,
            wall_deadline_monotonic_ns=running.wall_deadline_monotonic_ns,
            watchdog_triggered_at=None if watchdog_triggered is None else watchdog_triggered.observed_at,
            watchdog_triggered_monotonic_ns=None if watchdog_triggered is None else watchdog_triggered.monotonic_ns,
            jailer_reaped_at=running.jailer_reaped_at,
            jailer_reaped_monotonic_ns=running.jailer_reaped_monotonic_ns,
            cgroup_empty_at=cgroup_empty.observed_at,
            cgroup_empty_monotonic_ns=cgroup_empty.monotonic_ns,
            cleanup_finished_at=cleanup_finished.observed_at,
            cleanup_finished_monotonic_ns=cleanup_finished.monotonic_ns,
            lifecycle='terminated',
            jailer_exit_code=jailer_exit_code,
            wall_watchdog_armed=True,
            wall_timeout_triggered=wall_timeout_triggered,
        )

    def discard_prepared(self, prepared: FirecrackerPreparedWorker) -> FirecrackerCleanupReceipt:
        """Remove an unlaunched jail only while its exact pre-launch inventory is still intact."""

        self.verify_prepared(prepared)
        expected_cgroup = _expected_cgroup_path(self._spec, prepared.run_id)
        if expected_cgroup.exists() or expected_cgroup.is_symlink():
            raise FirecrackerCleanupError('an unlaunched worker cannot have a run cgroup')
        run_container, _ = _expected_jail_paths(self._spec, prepared.run_id)
        if not _remove_tree_exact(run_container):
            raise FirecrackerCleanupError('cannot prove removal of the unlaunched Firecracker jail')
        cleanup_finished = _observe_lifecycle_time()
        return FirecrackerCleanupReceipt(
            run_id=prepared.run_id,
            launched_monotonic_ns=None,
            wall_deadline_monotonic_ns=None,
            watchdog_triggered_at=None,
            watchdog_triggered_monotonic_ns=None,
            jailer_reaped_at=None,
            jailer_reaped_monotonic_ns=None,
            cgroup_empty_at=None,
            cgroup_empty_monotonic_ns=None,
            cleanup_finished_at=cleanup_finished.observed_at,
            cleanup_finished_monotonic_ns=cleanup_finished.monotonic_ns,
            lifecycle='never_launched',
            jailer_exit_code=None,
            wall_watchdog_armed=False,
            wall_timeout_triggered=False,
        )


def connect_firecracker_vsock(*, uds_path: str, port: int, timeout_seconds: float) -> socket.socket:
    """Open a host UDS and complete Firecracker's exact ``CONNECT <port>`` handshake."""

    normalized_path = _absolute_normalized_path(uds_path, label='vsock UDS path')
    if len(os.fsencode(normalized_path)) > 100:
        raise FirecrackerVsockError('vsock UDS path is too long for a portable AF_UNIX address')
    if port < 1 or port > 2**32 - 1:
        raise FirecrackerVsockError('vsock port must be between 1 and 2^32 - 1')
    if timeout_seconds <= 0 or timeout_seconds > 3600:
        raise FirecrackerVsockError('vsock timeout must be greater than zero and at most one hour')
    path = Path(normalized_path)
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise FirecrackerVsockError('vsock UDS does not exist') from error
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISSOCK(path_stat.st_mode):
        raise FirecrackerVsockError('vsock endpoint must be a non-symlink Unix socket')

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout_seconds)
    try:
        connection.connect(normalized_path)
        connection.sendall(f'CONNECT {port}\n'.encode('ascii'))
        response = bytearray()
        while not response.endswith(b'\n'):
            if len(response) >= _MAX_VSOCK_HANDSHAKE_BYTES:
                raise FirecrackerVsockError('vsock handshake exceeded its fixed byte limit')
            chunk = connection.recv(1)
            if not chunk:
                raise FirecrackerVsockError('vsock endpoint closed during handshake')
            response.extend(chunk)
        assigned_port_bytes = bytes(response)[3:-1] if response.startswith(b'OK ') else b''
        if not assigned_port_bytes or not assigned_port_bytes.isdigit() or len(assigned_port_bytes) > 10:
            raise FirecrackerVsockError('vsock endpoint returned an invalid handshake response')
        assigned_hostside_port = int(assigned_port_bytes)
        if assigned_hostside_port < 1 or assigned_hostside_port > 2**32 - 1:
            raise FirecrackerVsockError('vsock endpoint returned an out-of-range host-side port')
    except (OSError, FirecrackerVsockError) as error:
        connection.close()
        if isinstance(error, FirecrackerVsockError):
            raise
        raise FirecrackerVsockError('cannot establish the Firecracker vsock channel') from error
    return connection


def firecracker_guest_initiated_uds_path(*, uds_path: str, port: int) -> str:
    """Return Firecracker's ``<uds_path>_<PORT>`` listener for guest-initiated connections."""

    normalized_path = _absolute_normalized_path(uds_path, label='vsock UDS path')
    if port < 1 or port > 2**32 - 1:
        raise FirecrackerVsockError('vsock port must be between 1 and 2^32 - 1')
    listener_path = f'{normalized_path}_{port}'
    if len(os.fsencode(listener_path)) > 100:
        raise FirecrackerVsockError('guest-initiated vsock UDS path is too long for a portable AF_UNIX address')
    return listener_path


def capture_firecracker_prebound_guest_listener(
    prepared: FirecrackerPreparedWorker,
    *,
    spec: FirecrackerWorkerSpec,
) -> FirecrackerPreboundGuestListener:
    """Snapshot the one exact callback socket that ``launch`` is allowed to preserve."""

    if prepared.worker_spec_sha256 != firecracker_model_sha256(spec):
        raise FirecrackerPreparationError('prebound listener belongs to a different worker specification')
    socket_path = Path(
        firecracker_guest_initiated_uds_path(
            uds_path=prepared.vsock_uds_path,
            port=spec.guest_rpc_port,
        )
    )
    if socket_path.parent != Path(prepared.vsock_uds_path).parent:
        raise FirecrackerPreparationError('prebound listener escaped the prepared run directory')
    try:
        metadata = socket_path.lstat()
    except OSError as error:
        raise FirecrackerPreparationError('prebound guest listener is unavailable') from error
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != spec.worker_uid
        or metadata.st_gid != spec.worker_gid
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise FirecrackerPreparationError('prebound guest listener has unsafe metadata')
    return FirecrackerPreboundGuestListener(
        run_id=prepared.run_id,
        worker_spec_sha256=prepared.worker_spec_sha256,
        socket_path=str(socket_path),
        device_id=metadata.st_dev,
        inode=metadata.st_ino,
        worker_uid=spec.worker_uid,
        worker_gid=spec.worker_gid,
    )


_EXECUTABLE_ROLES = frozenset({FirecrackerArtifactRole.FIRECRACKER, FirecrackerArtifactRole.JAILER})


def _absolute_normalized_path(value: str, *, label: str) -> str:
    if '\x00' in value:
        raise ValueError(f'{label} cannot contain NUL')
    path = PurePosixPath(value)
    if not path.is_absolute() or path.as_posix() != value or '..' in path.parts:
        raise ValueError(f'{label} must be an absolute normalized POSIX path')
    if value == '/':
        raise ValueError(f'{label} cannot be the filesystem root')
    return value


def _validate_run_id(run_id: str) -> None:
    if len(run_id) != 32 or any(character not in '0123456789abcdef' for character in run_id):
        raise ValueError('Firecracker run ID must contain exactly 32 lowercase hexadecimal characters')


def _normalized_architecture(value: str) -> Literal['x86_64', 'aarch64']:
    lowered = value.lower()
    if lowered in {'x86_64', 'amd64'}:
        return 'x86_64'
    if lowered in {'aarch64', 'arm64'}:
        return 'aarch64'
    raise FirecrackerPreflightError(f'unsupported Firecracker host architecture: {value}')


def _all_artifacts(spec: FirecrackerWorkerSpec) -> tuple[FirecrackerArtifactIdentity, ...]:
    return (
        spec.runtime.firecracker,
        spec.runtime.jailer,
        spec.images.kernel,
        spec.images.rootfs,
        spec.images.harness,
        spec.images.scratch_template,
    )


def _verify_trusted_directory(path: Path, *, expected_owner: int) -> None:
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise FirecrackerPreflightError(f'trusted directory is unavailable: {path}') from error
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise FirecrackerPreflightError(f'trusted path must be a non-symlink directory: {path}')
    if path_stat.st_uid != expected_owner or path_stat.st_mode & 0o022:
        raise FirecrackerPreflightError(
            f'trusted directory must have the expected owner and no group/other writes: {path}'
        )


def _verify_trusted_artifact(identity: FirecrackerArtifactIdentity, *, executable: bool) -> None:
    path = Path(identity.source_path)
    current = path.parent
    while current != Path('/'):
        _verify_trusted_directory(current, expected_owner=0)
        current = current.parent
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise FirecrackerPreflightError(f'pinned artifact is unavailable: {identity.artifact_id}') from error
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise FirecrackerPreflightError(f'pinned artifact must be a non-symlink regular file: {identity.artifact_id}')
    if path_stat.st_uid != 0 or path_stat.st_mode & 0o022:
        raise FirecrackerPreflightError(
            f'pinned artifact must be root-owned and immutable to other users: {identity.artifact_id}'
        )
    if executable and not path_stat.st_mode & stat.S_IXUSR:
        raise FirecrackerPreflightError(f'pinned runtime artifact must be owner-executable: {identity.artifact_id}')
    try:
        observed_sha256, observed_bytes = _hash_regular_file(path)
    except OSError as error:
        raise FirecrackerPreflightError(f'cannot hash pinned artifact: {identity.artifact_id}') from error
    if observed_sha256 != identity.sha256 or observed_bytes != identity.byte_count:
        raise FirecrackerPreflightError(f'pinned artifact digest or size mismatch: {identity.artifact_id}')


def _hash_regular_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError('artifact is not a regular file')
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
            raise OSError('artifact changed while it was being hashed')
        return digest.hexdigest(), byte_count
    finally:
        os.close(descriptor)


def _copy_verified(identity: FirecrackerArtifactIdentity, destination: Path, *, mode: int) -> None:
    source_flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    destination_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    )
    try:
        source = os.open(identity.source_path, source_flags)
    except OSError as error:
        raise FirecrackerPreparationError(f'cannot open pinned source artifact: {identity.artifact_id}') from error
    destination_descriptor: int | None = None
    try:
        source_stat = os.fstat(source)
        if not stat.S_ISREG(source_stat.st_mode):
            raise FirecrackerPreparationError(f'pinned source is not a regular file: {identity.artifact_id}')
        destination_descriptor = os.open(destination, destination_flags, mode)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(source, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError('short write while copying pinned artifact')
                view = view[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source)
        if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise FirecrackerPreparationError(f'pinned source changed while copied: {identity.artifact_id}')
        if byte_count != identity.byte_count or digest.hexdigest() != identity.sha256:
            raise FirecrackerPreparationError(f'pinned source digest or size mismatch: {identity.artifact_id}')
        os.fchmod(destination_descriptor, mode)
    except Exception:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
            destination_descriptor = None
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        os.close(source)
        if destination_descriptor is not None:
            os.close(destination_descriptor)


def _write_new_file(path: Path, content: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short write while writing Firecracker config')
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _verify_copied_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    expected_mode: int,
    expected_uid: int,
    expected_gid: int,
) -> None:
    try:
        path_stat = path.lstat()
        observed_sha256, observed_bytes = _hash_regular_file(path)
    except OSError as error:
        raise FirecrackerPreparationError(f'cannot verify prepared file: {path.name}') from error
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise FirecrackerPreparationError(f'prepared entry is not a regular file: {path.name}')
    if stat.S_IMODE(path_stat.st_mode) != expected_mode:
        raise FirecrackerPreparationError(f'prepared file mode is not pinned: {path.name}')
    if path_stat.st_uid != expected_uid or path_stat.st_gid != expected_gid:
        raise FirecrackerPreparationError(f'prepared file ownership is not pinned: {path.name}')
    if observed_sha256 != expected_sha256 or observed_bytes != expected_bytes:
        raise FirecrackerPreparationError(f'prepared file digest or size mismatch: {path.name}')


def _expected_jail_paths(spec: FirecrackerWorkerSpec, run_id: str) -> tuple[Path, Path]:
    _validate_run_id(run_id)
    executable_name = Path(spec.runtime.firecracker.source_path).name
    run_container = Path(spec.chroot_base_dir) / executable_name / run_id
    return run_container, run_container / 'root'


def _expected_cgroup_path(spec: FirecrackerWorkerSpec, run_id: str) -> Path:
    _validate_run_id(run_id)
    return Path('/sys/fs/cgroup').joinpath(*spec.cgroup_parent.split('/'), run_id)


def _path_absent_exact(path: Path, *, label: str) -> bool:
    """Prove absence with ``lstat`` so broken symlinks cannot masquerade as cleanup."""

    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError as error:
        raise FirecrackerAttestationError(f'cannot independently inspect the claimed-removed {label}') from error
    return False


def _remove_empty_cgroup_exact(cgroup_path: Path) -> bool:
    """Remove only the run leaf; cgroupfs rejects removal while tasks remain."""

    try:
        path_stat = cgroup_path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        return False
    try:
        cgroup_path.rmdir()
    except OSError:
        return False
    return not cgroup_path.exists() and not cgroup_path.is_symlink()


def _remove_tree_exact(run_container: Path) -> bool:
    """Remove only a run directory, never a symlink or an ancestor."""

    try:
        path_stat = run_container.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        return False
    try:
        shutil.rmtree(run_container)
    except OSError:
        return False
    return not run_container.exists() and not run_container.is_symlink()


def _open_pinned_jail_root(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise FirecrackerWorkerError('cannot pin the prepared jail root before launch') from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
        os.close(descriptor)
        raise FirecrackerWorkerError('prepared jail root identity or ownership changed before launch')
    return descriptor


def _observe_launched_firecracker_child(
    *,
    spec: FirecrackerWorkerSpec,
    prepared: FirecrackerPreparedWorker,
    jailer_process: subprocess.Popen[bytes],
    jailer_identity: _ProcProcessIdentity,
    jail_root_descriptor: int,
    timeout_seconds: float,
) -> _ObservedFirecrackerChild:
    """Bind the jailer's create-once pid file to one pidfd and pinned cgroup."""

    executable_name = Path(spec.runtime.firecracker.source_path).name
    if not executable_name or '/' in executable_name or executable_name in {'.', '..'}:
        raise FirecrackerWorkerError('Firecracker executable basename is unsafe')
    pid_file_name = f'{executable_name}.pid'
    pid_file_path = str(Path(prepared.jail_root) / pid_file_name)
    deadline = time.monotonic() + timeout_seconds
    pid: int | None = None
    pid_file_identity: tuple[int, int] | None = None
    while time.monotonic() < deadline:
        try:
            descriptor = os.open(
                pid_file_name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=jail_root_descriptor,
            )
        except FileNotFoundError:
            if jailer_process.poll() not in {None, 0}:
                raise FirecrackerWorkerError('jailer failed before publishing its Firecracker child PID')
            time.sleep(0.01)
            continue
        except OSError as error:
            raise FirecrackerWorkerError('cannot open the jailer-published Firecracker PID') from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != 0
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 1
                or before.st_size > _MAX_FIRECRACKER_PID_BYTES
            ):
                raise FirecrackerWorkerError('jailer-published Firecracker PID file is unsafe')
            content = os.read(descriptor, _MAX_FIRECRACKER_PID_BYTES + 1)
            after = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or len(content) != before.st_size
                or not content.isdigit()
            ):
                raise FirecrackerWorkerError('jailer-published Firecracker PID changed while read')
            pid = int(content)
            pid_file_identity = (before.st_dev, before.st_ino)
        finally:
            os.close(descriptor)
        break
    if pid is None or pid_file_identity is None:
        raise FirecrackerWorkerError('jailer did not publish a Firecracker child PID before the launch deadline')
    if pid <= 1 or pid == jailer_process.pid:
        raise FirecrackerWorkerError('jailer PID file does not name a distinct Firecracker child')
    post_setup_root = os.fstat(jail_root_descriptor)
    if (
        not stat.S_ISDIR(post_setup_root.st_mode)
        or post_setup_root.st_uid != spec.worker_uid
        or post_setup_root.st_gid != spec.worker_gid
        or stat.S_IMODE(post_setup_root.st_mode) != 0o700
    ):
        raise FirecrackerWorkerError('jailer did not apply its pinned jail-root ownership transition')

    pidfd: int | None = None
    cgroup_descriptor: int | None = None
    try:
        pidfd_open = getattr(os, 'pidfd_open', None)
        if pidfd_open is None or getattr(signal, 'pidfd_send_signal', None) is None:
            raise FirecrackerWorkerError('host Python lacks the required pidfd signaling support')
        try:
            pidfd = pidfd_open(pid, 0)
        except OSError as error:
            raise FirecrackerWorkerError('cannot pin the jailer-published Firecracker PID') from error
        cgroup_descriptor, cgroup_device_id, cgroup_inode = _open_pinned_worker_cgroup(
            spec=spec,
            run_id=prepared.run_id,
            firecracker_pid=pid,
        )

        observed: _ProcProcessIdentity | None = None
        executable_sha256: str | None = None
        while time.monotonic() < deadline:
            try:
                candidate = _read_proc_process_identity(pid)
                candidate_sha256 = _proc_executable_sha256(pid)
                cgroup_bound = _process_in_pinned_cgroup(pid, cgroup_descriptor, spec, prepared.run_id)
            except (FileNotFoundError, ProcessLookupError):
                candidate = None
                candidate_sha256 = None
                cgroup_bound = False
            if (
                candidate is not None
                and candidate.state != 'Z'
                and candidate.process_group_id == jailer_process.pid
                and candidate.session_id == jailer_identity.session_id
                and candidate.start_time_ticks >= jailer_identity.start_time_ticks
                and candidate_sha256 == spec.runtime.firecracker.sha256
                and cgroup_bound
                and _pidfd_process_alive(pidfd)
            ):
                observed = candidate
                executable_sha256 = candidate_sha256
                break
            if jailer_process.poll() not in {None, 0}:
                raise FirecrackerWorkerError('jailer failed while starting its Firecracker child')
            time.sleep(0.01)
        if observed is None or executable_sha256 is None:
            raise FirecrackerWorkerError('Firecracker child identity never matched the pinned launch')
        try:
            jailer_exit_code = jailer_process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            raise FirecrackerWorkerError('jailer parent did not exit after publishing its child') from error
        jailer_reaped = _observe_lifecycle_time()
        if jailer_exit_code != 0:
            raise FirecrackerWorkerError('jailer parent reported a failed Firecracker launch')
        # Recheck after reaping the parent so a PID switch between observation and return cannot pass.
        rebound = _read_proc_process_identity(pid)
        if (
            rebound.start_time_ticks != observed.start_time_ticks
            or rebound.process_group_id != observed.process_group_id
            or rebound.session_id != observed.session_id
            or _proc_executable_sha256(pid) != executable_sha256
            or not _process_in_pinned_cgroup(pid, cgroup_descriptor, spec, prepared.run_id)
            or not _pidfd_process_alive(pidfd)
        ):
            raise FirecrackerWorkerError('Firecracker child identity changed after jailer exit')
        return _ObservedFirecrackerChild(
            pid=pid,
            parent_pid_at_observation=observed.parent_pid,
            process_group_id=observed.process_group_id,
            session_id=observed.session_id,
            start_time_ticks=observed.start_time_ticks,
            executable_sha256=executable_sha256,
            pid_file_path=pid_file_path,
            pid_file_device_id=pid_file_identity[0],
            pid_file_inode=pid_file_identity[1],
            pidfd=pidfd,
            cgroup_descriptor=cgroup_descriptor,
            cgroup_device_id=cgroup_device_id,
            cgroup_inode=cgroup_inode,
            jailer_reaped_at=jailer_reaped.observed_at,
            jailer_reaped_monotonic_ns=jailer_reaped.monotonic_ns,
        )
    except BaseException:
        if cgroup_descriptor is not None:
            os.close(cgroup_descriptor)
        if pidfd is not None:
            os.close(pidfd)
        raise


def _read_proc_process_identity(pid: int, *, proc_root: Path = Path('/proc')) -> _ProcProcessIdentity:
    path = proc_root / str(pid) / 'stat'
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        content = os.read(descriptor, 64 * 1024 + 1)
    finally:
        os.close(descriptor)
    if not content or len(content) > 64 * 1024 or b'\x00' in content:
        raise FirecrackerWorkerError('Firecracker procfs identity is invalid')
    close = content.rfind(b')')
    if close <= 0:
        raise FirecrackerWorkerError('Firecracker procfs identity is malformed')
    fields = content[close + 2 :].split()
    if len(fields) < 20:
        raise FirecrackerWorkerError('Firecracker procfs identity is incomplete')
    try:
        observed_pid = int(content[: content.find(b' ')])
        state = fields[0].decode('ascii')
        parent_pid = int(fields[1])
        process_group_id = int(fields[2])
        session_id = int(fields[3])
        start_time_ticks = int(fields[19])
    except (UnicodeDecodeError, ValueError) as error:
        raise FirecrackerWorkerError('Firecracker procfs identity contains invalid fields') from error
    if (
        observed_pid != pid
        or len(state) != 1
        or parent_pid < 0
        or process_group_id <= 1
        or session_id <= 0
        or start_time_ticks <= 0
    ):
        raise FirecrackerWorkerError('Firecracker procfs identity is out of range')
    return _ProcProcessIdentity(
        pid=pid,
        state=state,
        parent_pid=parent_pid,
        process_group_id=process_group_id,
        session_id=session_id,
        start_time_ticks=start_time_ticks,
    )


def _proc_executable_sha256(pid: int, *, proc_root: Path = Path('/proc')) -> str:
    descriptor = os.open(proc_root / str(pid) / 'exe', os.O_RDONLY | os.O_CLOEXEC)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FirecrackerWorkerError('Firecracker procfs executable is not a regular file')
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _COPY_CHUNK_BYTES):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _open_pinned_worker_cgroup(
    *,
    spec: FirecrackerWorkerSpec,
    run_id: str,
    firecracker_pid: int,
) -> tuple[int, int, int]:
    cgroup_path = _expected_cgroup_path(spec, run_id)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            cgroup_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        metadata = os.fstat(descriptor)
        kill_metadata = os.stat('cgroup.kill', dir_fd=descriptor, follow_symlinks=False)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise FirecrackerWorkerError('cannot pin the exact worker cgroup with cgroup.kill') from error
    assert descriptor is not None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or not stat.S_ISREG(kill_metadata.st_mode)
        or kill_metadata.st_uid != 0
    ):
        os.close(descriptor)
        raise FirecrackerWorkerError('worker cgroup or cgroup.kill has an unsafe identity')
    if not _process_in_pinned_cgroup(firecracker_pid, descriptor, spec, run_id):
        os.close(descriptor)
        raise FirecrackerWorkerError('Firecracker child is not in the pinned worker cgroup')
    return descriptor, metadata.st_dev, metadata.st_ino


def _read_openat_bounded(descriptor: int, name: str, maximum_bytes: int) -> bytes:
    child = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=descriptor)
    try:
        content = os.read(child, maximum_bytes + 1)
    finally:
        os.close(child)
    if len(content) > maximum_bytes:
        raise FirecrackerWorkerError('worker cgroup evidence file is oversized')
    return content


def _process_in_pinned_cgroup(
    pid: int,
    cgroup_descriptor: int,
    spec: FirecrackerWorkerSpec,
    run_id: str,
) -> bool:
    try:
        members = _read_openat_bounded(cgroup_descriptor, 'cgroup.procs', 64 * 1024)
        proc_descriptor = os.open(
            Path('/proc') / str(pid) / 'cgroup',
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            proc_cgroup = os.read(proc_descriptor, 64 * 1024 + 1)
        finally:
            os.close(proc_descriptor)
    except OSError:
        return False
    if len(proc_cgroup) > 64 * 1024:
        return False
    try:
        member_pids = {int(value) for value in members.split()}
        relative = '/' + _expected_cgroup_path(spec, run_id).relative_to('/sys/fs/cgroup').as_posix()
        lines = proc_cgroup.decode('ascii').splitlines()
    except (UnicodeDecodeError, ValueError):
        return False
    return pid in member_pids and f'0::{relative}' in lines


def _pidfd_process_alive(pidfd: int) -> bool:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
    return not poller.poll(0)


def _pinned_cgroup_member_pids(
    descriptor: int,
    *,
    expected_device_id: int,
    expected_inode: int,
) -> tuple[int, ...]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != (expected_device_id, expected_inode):
        raise FirecrackerCleanupError('pinned Firecracker cgroup descriptor changed identity')
    try:
        members = tuple(int(value) for value in _read_openat_bounded(descriptor, 'cgroup.procs', 64 * 1024).split())
    except (OSError, ValueError, FirecrackerWorkerError) as error:
        raise FirecrackerCleanupError('cannot read the pinned Firecracker cgroup members') from error
    if any(pid <= 1 for pid in members) or len(members) != len(set(members)):
        raise FirecrackerCleanupError('pinned Firecracker cgroup contains invalid member PIDs')
    return tuple(sorted(members))


def _kill_pinned_cgroup(
    descriptor: int,
    *,
    expected_device_id: int,
    expected_inode: int,
) -> None:
    _pinned_cgroup_member_pids(
        descriptor,
        expected_device_id=expected_device_id,
        expected_inode=expected_inode,
    )
    try:
        kill_descriptor = os.open(
            'cgroup.kill',
            os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            if os.write(kill_descriptor, b'1') != 1:
                raise FirecrackerCleanupError('short write to pinned cgroup.kill')
        finally:
            os.close(kill_descriptor)
    except OSError as error:
        raise FirecrackerCleanupError('cannot kill the pinned Firecracker cgroup') from error


def _signal_bound_firecracker(running: RunningFirecrackerWorker, signal_number: int) -> None:
    send = getattr(signal, 'pidfd_send_signal', None)
    if send is None:
        raise FirecrackerCleanupError('host Python lacks required pidfd signaling support')
    try:
        send(running.firecracker_pidfd, signal_number, None, 0)
    except ProcessLookupError:
        return
    except OSError as error:
        raise FirecrackerCleanupError('cannot signal the pidfd-bound Firecracker child') from error


def _bound_firecracker_process_alive(running: RunningFirecrackerWorker) -> bool:
    if not _pidfd_process_alive(running.firecracker_pidfd):
        return False
    try:
        observed = _read_proc_process_identity(running.firecracker_pid)
    except FileNotFoundError:
        return False
    if observed.state == 'Z':
        return False
    if (
        observed.start_time_ticks != running.firecracker_start_time_ticks
        or observed.process_group_id != running.firecracker_process_group_id
        or observed.session_id != running.firecracker_session_id
    ):
        raise FirecrackerCleanupError('bound Firecracker child PID was reused or changed identity')
    return True


def _abort_failed_launch(
    process: subprocess.Popen[bytes],
    *,
    spec: FirecrackerWorkerSpec,
    run_id: str,
) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass
    cgroup_path = _expected_cgroup_path(spec, run_id)
    try:
        descriptor = os.open(
            cgroup_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except FileNotFoundError:
        return
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0:
            raise FirecrackerCleanupError('failed launch cgroup has an unsafe identity')
        _kill_pinned_cgroup(
            descriptor,
            expected_device_id=metadata.st_dev,
            expected_inode=metadata.st_ino,
        )
        deadline = time.monotonic() + 5.0
        while (
            _pinned_cgroup_member_pids(
                descriptor,
                expected_device_id=metadata.st_dev,
                expected_inode=metadata.st_ino,
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        if _pinned_cgroup_member_pids(
            descriptor,
            expected_device_id=metadata.st_dev,
            expected_inode=metadata.st_ino,
        ):
            raise FirecrackerCleanupError('failed launch cgroup remains populated after cgroup.kill')
    finally:
        os.close(descriptor)


def _watchdog_process_group(
    firecracker_pid: int,
    firecracker_start_time_ticks: int,
    process_group_id: int,
    pidfd: int,
    cgroup_descriptor: int,
    cgroup_device_id: int,
    cgroup_inode: int,
    stop: threading.Event,
    timeout_triggered: threading.Event,
    failure: threading.Event,
    deadline_monotonic_ns: int,
    timing: _WatchdogTiming,
) -> None:
    """Request pinned-cgroup kill after the deadline and retain the real trigger time.

    Writing ``cgroup.kill`` starts asynchronous kernel work.  This watchdog therefore records only
    when it observed the expired deadline and issued the request; the cleanup path separately polls
    for and records the later cgroup-empty observation.
    """

    while not stop.is_set():
        if not _pidfd_process_alive(pidfd):
            try:
                members = _pinned_cgroup_member_pids(
                    cgroup_descriptor,
                    expected_device_id=cgroup_device_id,
                    expected_inode=cgroup_inode,
                )
                if members:
                    _kill_pinned_cgroup(
                        cgroup_descriptor,
                        expected_device_id=cgroup_device_id,
                        expected_inode=cgroup_inode,
                    )
            except FirecrackerCleanupError:
                failure.set()
            return
        try:
            observed = _read_proc_process_identity(firecracker_pid)
        except FileNotFoundError:
            return
        except (OSError, FirecrackerWorkerError):
            failure.set()
            return
        if observed.state == 'Z':
            return
        if observed.start_time_ticks != firecracker_start_time_ticks or observed.process_group_id != process_group_id:
            failure.set()
            return
        remaining_ns = deadline_monotonic_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            timing.record_triggered()
            timeout_triggered.set()
            try:
                _kill_pinned_cgroup(
                    cgroup_descriptor,
                    expected_device_id=cgroup_device_id,
                    expected_inode=cgroup_inode,
                )
            except FirecrackerCleanupError:
                failure.set()
            return
        stop.wait(min(remaining_ns / 1_000_000_000, 0.1))


def _terminate_process_group(
    running: RunningFirecrackerWorker,
    *,
    grace_seconds: float,
) -> _LifecycleObservation:
    if _bound_firecracker_process_alive(running):
        _signal_bound_firecracker(running, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while (
        _pinned_cgroup_member_pids(
            running.cgroup_descriptor,
            expected_device_id=running.cgroup_device_id,
            expected_inode=running.cgroup_inode,
        )
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    if _pinned_cgroup_member_pids(
        running.cgroup_descriptor,
        expected_device_id=running.cgroup_device_id,
        expected_inode=running.cgroup_inode,
    ):
        _kill_pinned_cgroup(
            running.cgroup_descriptor,
            expected_device_id=running.cgroup_device_id,
            expected_inode=running.cgroup_inode,
        )
        deadline = time.monotonic() + grace_seconds
        while (
            _pinned_cgroup_member_pids(
                running.cgroup_descriptor,
                expected_device_id=running.cgroup_device_id,
                expected_inode=running.cgroup_inode,
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
    if _pinned_cgroup_member_pids(
        running.cgroup_descriptor,
        expected_device_id=running.cgroup_device_id,
        expected_inode=running.cgroup_inode,
    ):
        raise FirecrackerCleanupError('Firecracker cgroup remains populated after cgroup.kill')
    return _observe_lifecycle_time()


def _observe_lifecycle_time() -> _LifecycleObservation:
    """Capture paired audit (UTC) and ordering (monotonic) clocks without converting either."""

    monotonic_ns = time.monotonic_ns()
    return _LifecycleObservation(observed_at=datetime.now(UTC), monotonic_ns=monotonic_ns)
