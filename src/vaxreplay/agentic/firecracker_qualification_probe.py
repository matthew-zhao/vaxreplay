"""Measured probe protocol for Firecracker runtime qualification.

The collector and verifier intentionally have different jobs.  A collector obtains fresh
challenge-bound guest and host observations and authenticates the exact raw collection.  This
module's verifier independently derives the seven qualification drill outcomes from those
observations.  A development collection uses the same protocol but is cryptographically and
structurally ineligible for production qualification.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.firecracker import (
    FirecrackerArtifactIdentity,
    FirecrackerArtifactRole,
    FirecrackerGuestImages,
    FirecrackerHostPreflightReceipt,
    FirecrackerWorkerSpec,
    firecracker_model_sha256,
    firecracker_static_config_bytes,
)
from vaxreplay.agentic.firecracker_qualification import (
    FirecrackerFullSuiteEvidence,
    FirecrackerHostObservation,
    FirecrackerQualificationClaim,
    FirecrackerQualificationDrillEvidence,
    FirecrackerQualificationDrillId,
    required_firecracker_qualification_claims,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

FIRECRACKER_QUALIFICATION_PROBE_MANIFEST_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-probe-manifest.v0.1'
FIRECRACKER_QUALIFICATION_CHALLENGE_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-challenge.v0.1'
FIRECRACKER_QUALIFICATION_WORKER_BINDING_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-worker-binding.v0.2'
FIRECRACKER_QUALIFICATION_GUEST_RESPONSE_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-guest-response.v0.1'
AUTHENTICATED_FIRECRACKER_QUALIFICATION_GUEST_RESPONSE_SCHEMA_VERSION = (
    'vaxreplay.authenticated-firecracker-qualification-guest-response.v0.1'
)
FIRECRACKER_QUALIFICATION_CGROUP_SNAPSHOT_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-cgroup-snapshot.v0.1'
FIRECRACKER_QUALIFICATION_HOST_CGROUP_CANARY_SCHEMA_VERSION = (
    'vaxreplay.firecracker-qualification-host-cgroup-canary.v0.1'
)
FIRECRACKER_QUALIFICATION_RAW_DRILL_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-raw-drill.v0.1'
FIRECRACKER_QUALIFICATION_RAW_COLLECTION_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-raw-collection.v0.2'
AUTHENTICATED_FIRECRACKER_QUALIFICATION_RAW_COLLECTION_SCHEMA_VERSION = (
    'vaxreplay.authenticated-firecracker-qualification-raw-collection.v0.1'
)

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_HEX_32_PATTERN = r'^[0-9a-f]{32}$'
_HEX_64_PATTERN = r'^[0-9a-f]{64}$'
_SIGNATURE_PATTERN = r'^[0-9a-f]{128}$'
_COLLECTOR_SIGNATURE_DOMAIN = b'vaxreplay.firecracker-live-collector.v0.1\x00'
_GUEST_SIGNATURE_DOMAIN = b'vaxreplay.firecracker-qualification-guest.v0.1\x00'
_COLLECTOR_KEY_ID_DOMAIN = b'vaxreplay.firecracker-live-collector-key-id.v0.1\x00'
_GUEST_KEY_ID_DOMAIN = b'vaxreplay.firecracker-qualification-guest-key-id.v0.1\x00'


class FirecrackerQualificationGuestDiskBuildReceipt(StrictModel):
    """Reproducible, qualification-only rootfs/harness build output embedded in the manifest."""

    schema_version: Literal['vaxreplay.firecracker-qualification-guest-disk-build.v0.1'] = (
        'vaxreplay.firecracker-qualification-guest-disk-build.v0.1'
    )
    profile: Literal['firecracker_qualification_probe_v1'] = 'firecracker_qualification_probe_v1'
    source_date_epoch: int = Field(ge=1, le=2**31 - 1)
    base_rootfs_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    package_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    normalized_rootfs_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    normalized_harness_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    build_recipe_sha256: str = Field(pattern=_SHA256_PATTERN)
    mke2fs_sha256: str = Field(pattern=_SHA256_PATTERN)
    mke2fs_version: str = Field(min_length=1, max_length=200)
    e2fsck_sha256: str = Field(pattern=_SHA256_PATTERN)
    e2fsck_version: str = Field(min_length=1, max_length=200)
    debugfs_sha256: str = Field(pattern=_SHA256_PATTERN)
    debugfs_version: str = Field(min_length=1, max_length=200)
    build_argv_and_env_sha256: str = Field(pattern=_SHA256_PATTERN)
    init_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_probe_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    rootfs_sha256: str = Field(pattern=_SHA256_PATTERN)
    rootfs_byte_count: int = Field(gt=0, le=1024 * 1024 * 1024 * 1024)
    harness_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_byte_count: int = Field(gt=0, le=1024 * 1024 * 1024 * 1024)
    rootfs_uuid: Literal['00000000-0000-4000-8000-000000000101'] = '00000000-0000-4000-8000-000000000101'
    harness_uuid: Literal['00000000-0000-4000-8000-000000000102'] = '00000000-0000-4000-8000-000000000102'
    rootfs_label: Literal['vaxqual-root'] = 'vaxqual-root'
    harness_label: Literal['vaxqual-harness'] = 'vaxqual-harness'
    fixed_init_argv: tuple[
        Literal['/usr/bin/python3'],
        Literal['-I'],
        Literal['/opt/vaxreplay/bin/vaxreplay-firecracker-qualification-probe'],
        Literal['--config'],
        Literal['/opt/vaxreplay/etc/qualification-guest.json'],
    ] = (
        '/usr/bin/python3',
        '-I',
        '/opt/vaxreplay/bin/vaxreplay-firecracker-qualification-probe',
        '--config',
        '/opt/vaxreplay/etc/qualification-guest.json',
    )
    rootfs_and_harness_built_separately: Literal[True] = True
    lane_a_task_guest_reused: Literal[False] = False
    ext4_lazy_initialization_disabled: Literal[True] = True
    source_metadata_normalized: Literal[True] = True


class FirecrackerQualificationProbeError(ValueError):
    """A raw collection or probe response failed closed."""


class FirecrackerQualificationCollectionMode(str, enum.Enum):
    PRODUCTION_LINUX_KVM = 'production_linux_kvm'
    DEVELOPMENT_SIMULATED = 'development_simulated'


class FirecrackerQualificationBoundaryKind(str, enum.Enum):
    PINNED_LINUX_KVM_DRIVER = 'pinned_linux_kvm_driver'
    DETERMINISTIC_DEVELOPMENT = 'deterministic_development'


class FirecrackerQualificationObservationSource(str, enum.Enum):
    GUEST_SIGNED = 'guest_signed'
    HOST_PROCFS = 'host_procfs'
    HOST_CGROUPFS = 'host_cgroupfs'
    HOST_LSTAT = 'host_lstat'
    HOST_MONOTONIC = 'host_monotonic'
    HOST_FIRECRACKER_CONFIG = 'host_firecracker_config'
    HOST_VSOCK = 'host_vsock'


class FirecrackerQualificationGuestCommand(str, enum.Enum):
    BOOT_READY_AND_EXIT = 'boot_ready_and_exit'
    VSOCK_NONCE_ECHO = 'vsock_nonce_echo'
    ISOLATION_PROBES = 'isolation_probes'
    CGROUP_STRESS = 'cgroup_stress'
    INTENTIONAL_HANG = 'intentional_hang'
    LOAD_CANARY = 'load_canary'


_GUEST_COMMAND_FOR_DRILL: dict[FirecrackerQualificationDrillId, FirecrackerQualificationGuestCommand | None] = {
    FirecrackerQualificationDrillId.LIVE_BOOT: FirecrackerQualificationGuestCommand.BOOT_READY_AND_EXIT,
    FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP: FirecrackerQualificationGuestCommand.VSOCK_NONCE_ECHO,
    FirecrackerQualificationDrillId.GUEST_ISOLATION: FirecrackerQualificationGuestCommand.ISOLATION_PROBES,
    FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT: FirecrackerQualificationGuestCommand.CGROUP_STRESS,
    FirecrackerQualificationDrillId.WALL_TIMEOUT: FirecrackerQualificationGuestCommand.INTENTIONAL_HANG,
    FirecrackerQualificationDrillId.TEARDOWN: None,
    FirecrackerQualificationDrillId.LOAD_CANARY: FirecrackerQualificationGuestCommand.LOAD_CANARY,
}

_CLAIM_SOURCE: dict[FirecrackerQualificationClaim, FirecrackerQualificationObservationSource] = {
    FirecrackerQualificationClaim.FIRECRACKER_PROCESS_STARTED: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.GUEST_READY_AUTHENTICATED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.CLEAN_GUEST_EXIT: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.HOST_VSOCK_HANDSHAKE: FirecrackerQualificationObservationSource.HOST_VSOCK,
    FirecrackerQualificationClaim.GUEST_RPC_ROUND_TRIP: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.PEER_CID_BOUND: FirecrackerQualificationObservationSource.HOST_VSOCK,
    FirecrackerQualificationClaim.ROOTFS_WRITE_DENIED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.HARNESS_WRITE_DENIED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.SCRATCH_WRITE_SUCCEEDED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.SCRATCH_FRESH: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.NETWORK_UNREACHABLE: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.MMDS_UNREACHABLE: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.CPU_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.MEMORY_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.SWAP_DISABLED_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.PIDS_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.WALL_WATCHDOG_TRIGGERED: FirecrackerQualificationObservationSource.HOST_MONOTONIC,
    FirecrackerQualificationClaim.PROCESS_GROUP_KILLED: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.DEADLINE_BOUND: FirecrackerQualificationObservationSource.HOST_MONOTONIC,
    FirecrackerQualificationClaim.CGROUP_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.JAIL_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.VSOCK_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.PARALLEL_WORKERS_DISTINCT: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.ALL_WORKERS_COMPLETED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.ALL_WORKERS_TORN_DOWN: FirecrackerQualificationObservationSource.HOST_LSTAT,
}


class FirecrackerQualificationProbeManifest(StrictModel):
    """Externally pinned identity for a guest used only by qualification."""

    schema_version: Literal['vaxreplay.firecracker-qualification-probe-manifest.v0.1'] = (
        FIRECRACKER_QUALIFICATION_PROBE_MANIFEST_SCHEMA_VERSION
    )
    manifest_id: str = Field(min_length=1, max_length=200)
    task_worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_rootfs_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_harness_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_kernel_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_rootfs_path: str = Field(pattern=r'^/[A-Za-z0-9_./-]+$', min_length=2, max_length=4096)
    qualification_rootfs_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_rootfs_byte_count: int = Field(gt=0, le=1024 * 1024 * 1024 * 1024)
    qualification_harness_path: str = Field(pattern=r'^/[A-Za-z0-9_./-]+$', min_length=2, max_length=4096)
    qualification_harness_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_harness_byte_count: int = Field(gt=0, le=1024 * 1024 * 1024 * 1024)
    qualification_disk_build_receipt: FirecrackerQualificationGuestDiskBuildReceipt
    qualification_disk_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_probe_executable_path: Literal['/opt/vaxreplay/bin/vaxreplay-firecracker-qualification-probe'] = (
        '/opt/vaxreplay/bin/vaxreplay-firecracker-qualification-probe'
    )
    guest_probe_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_probe_public_key_hex: str = Field(pattern=_HEX_64_PATTERN)
    guest_probe_key_id: str = Field(pattern=_SHA256_PATTERN)
    guest_protocol_id: Literal['vaxreplay.firecracker-qualification-guest.v0.1'] = (
        'vaxreplay.firecracker-qualification-guest.v0.1'
    )
    qualification_guest_separate_from_task_guest: Literal[True] = True
    task_guest_protocol_reused_for_qualification: Literal[False] = False
    network_interfaces_enabled: Literal[False] = False
    mmds_enabled: Literal[False] = False

    @model_validator(mode='after')
    def validate_separate_guest(self) -> Self:
        for path in (self.qualification_rootfs_path, self.qualification_harness_path):
            if any(part in {'.', '..'} for part in path.split('/')):
                raise ValueError('qualification guest artifact paths cannot contain dot components')
        if self.qualification_rootfs_sha256 == self.task_rootfs_sha256:
            raise ValueError('qualification rootfs must be separate from the task guest rootfs')
        if self.qualification_harness_sha256 == self.task_harness_sha256:
            raise ValueError('qualification harness must be separate from the task harness')
        receipt = self.qualification_disk_build_receipt
        if (
            self.qualification_disk_build_receipt_sha256 != firecracker_model_sha256(receipt)
            or (self.qualification_rootfs_sha256, self.qualification_rootfs_byte_count)
            != (receipt.rootfs_sha256, receipt.rootfs_byte_count)
            or (self.qualification_harness_sha256, self.qualification_harness_byte_count)
            != (receipt.harness_sha256, receipt.harness_byte_count)
            or self.guest_probe_executable_sha256 != receipt.guest_probe_executable_sha256
        ):
            raise ValueError('qualification manifest differs from its embedded disk-build receipt')
        observed_key_id = firecracker_qualification_guest_key_id(bytes.fromhex(self.guest_probe_public_key_hex))
        if not hmac.compare_digest(observed_key_id, self.guest_probe_key_id):
            raise ValueError('qualification guest key ID does not match its public key')
        return self


class FirecrackerQualificationBoundaryIdentity(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-boundary-identity.v0.2'] = (
        'vaxreplay.firecracker-qualification-boundary-identity.v0.2'
    )
    boundary_id: str = Field(min_length=1, max_length=200)
    kind: FirecrackerQualificationBoundaryKind
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    external_executable_pin_enforced: bool
    direct_linux_kvm_launch: bool
    injected_test_boundary: bool
    runtime_closure_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    runtime_closure_receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    runtime_closure_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    transitive_runtime_pin_enforced: bool = False

    @model_validator(mode='after')
    def validate_kind(self) -> Self:
        is_production = self.kind == FirecrackerQualificationBoundaryKind.PINNED_LINUX_KVM_DRIVER
        if (
            self.external_executable_pin_enforced,
            self.direct_linux_kvm_launch,
            self.injected_test_boundary,
        ) != (is_production, is_production, not is_production):
            raise ValueError('qualification boundary flags must exactly match its boundary kind')
        closure_pins_present = all(
            value is not None
            for value in (
                self.runtime_closure_manifest_sha256,
                self.runtime_closure_receipt_sha256,
                self.runtime_closure_sha256,
            )
        )
        if (
            any(
                value is not None
                for value in (
                    self.runtime_closure_manifest_sha256,
                    self.runtime_closure_receipt_sha256,
                    self.runtime_closure_sha256,
                )
            )
            != closure_pins_present
        ):
            raise ValueError('qualification runtime-closure pins must be all present or all absent')
        if (closure_pins_present, self.transitive_runtime_pin_enforced) != (is_production, is_production):
            raise ValueError('only the production boundary may claim a complete transitive runtime pin')
        return self


class FirecrackerQualificationChallenge(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-challenge.v0.1'] = (
        FIRECRACKER_QUALIFICATION_CHALLENGE_SCHEMA_VERSION
    )
    collection_id: str = Field(pattern=_HEX_32_PATTERN)
    challenge_id: str = Field(pattern=_HEX_32_PATTERN)
    nonce_hex: str = Field(pattern=_HEX_64_PATTERN)
    drill_id: FirecrackerQualificationDrillId
    run_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime

    @field_validator('issued_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('qualification challenge time must include a UTC offset')
        return value.astimezone(UTC)

    @field_validator('run_ids')
    @classmethod
    def validate_run_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError('qualification run IDs must be unique')
        if any(len(item) != 32 or any(character not in '0123456789abcdef' for character in item) for item in value):
            raise ValueError('qualification run IDs must be 32 lowercase hexadecimal characters')
        return value

    @model_validator(mode='after')
    def validate_load_shape(self) -> Self:
        expected_count = 2 if self.drill_id == FirecrackerQualificationDrillId.LOAD_CANARY else 1
        if len(self.run_ids) != expected_count:
            raise ValueError('only the load-canary challenge may contain two run IDs')
        return self


class FirecrackerQualificationWorkerBinding(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-worker-binding.v0.2'] = (
        FIRECRACKER_QUALIFICATION_WORKER_BINDING_SCHEMA_VERSION
    )
    run_id: str = Field(pattern=_HEX_32_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_static_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepared_worker_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    firecracker_pid: int = Field(gt=1)
    firecracker_parent_pid_at_observation: int = Field(ge=0)
    firecracker_start_time_ticks: int = Field(gt=0)
    firecracker_session_id: int = Field(gt=0)
    firecracker_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    firecracker_pid_file_path: str = Field(pattern=r'^/[A-Za-z0-9_./-]+\.pid$', min_length=6, max_length=4096)
    firecracker_pid_file_device_id: int = Field(gt=0)
    firecracker_pid_file_inode: int = Field(gt=0)
    firecracker_pid_from_jailer_file_verified: Literal[True] = True
    jailer_pid: int = Field(gt=1)
    jailer_start_time_ticks: int = Field(gt=0)
    jailer_process_group_id: int = Field(gt=1)
    jailer_session_id: int = Field(gt=0)
    process_group_id: int = Field(gt=1)
    worker_uid: int = Field(gt=0)
    worker_gid: int = Field(gt=0)
    cgroup_path: str = Field(pattern=r'^/sys/fs/cgroup/[A-Za-z0-9_./-]+$', min_length=16, max_length=4096)
    cgroup_inode: int = Field(gt=0)
    cgroup_member_pids: tuple[int, ...] = Field(min_length=1, max_length=4096)
    jail_root: str = Field(pattern=r'^/[A-Za-z0-9_./-]+$', min_length=2, max_length=4096)
    vsock_uds_path: str = Field(pattern=r'^/[A-Za-z0-9_./-]+$', min_length=2, max_length=4096)
    guest_cid: int = Field(ge=3, le=2**32 - 1)
    peer_pid: int = Field(gt=1)
    peer_uid: int = Field(gt=0)
    peer_gid: int = Field(gt=0)
    process_tree_verified: bool
    pid_cgroup_binding_verified: bool

    @field_validator('cgroup_member_pids')
    @classmethod
    def validate_pids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != tuple(sorted(set(value))) or any(pid <= 1 for pid in value):
            raise ValueError('cgroup member PIDs must be positive, unique, and sorted')
        return value

    @model_validator(mode='after')
    def validate_pid_binding(self) -> Self:
        if self.firecracker_pid not in self.cgroup_member_pids or self.peer_pid != self.firecracker_pid:
            raise ValueError('worker binding must bind the vsock peer and cgroup to the tracked Firecracker PID')
        if (
            self.firecracker_pid == self.jailer_pid
            or self.jailer_process_group_id != self.jailer_pid
            or self.process_group_id != self.jailer_process_group_id
            or self.firecracker_session_id != self.jailer_session_id
            or self.jailer_session_id == self.jailer_pid
            or self.firecracker_start_time_ticks < self.jailer_start_time_ticks
        ):
            raise ValueError('worker binding does not describe a distinct jailer-created Firecracker child')
        pid_file = Path(self.firecracker_pid_file_path)
        if pid_file.parent != Path(self.jail_root) or pid_file.name in {'.pid', '..pid'}:
            raise ValueError('worker binding PID file must be the jailer-published file in the exact jail root')
        if (self.peer_uid, self.peer_gid) != (self.worker_uid, self.worker_gid):
            raise ValueError('worker binding peer credentials differ from the pinned worker identity')
        return self


class FirecrackerQualificationGuestResponse(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-guest-response.v0.1'] = (
        FIRECRACKER_QUALIFICATION_GUEST_RESPONSE_SCHEMA_VERSION
    )
    challenge_sha256: str = Field(pattern=_SHA256_PATTERN)
    nonce_hex: str = Field(pattern=_HEX_64_PATTERN)
    run_id: str = Field(pattern=_HEX_32_PATTERN)
    worker_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_probe_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    command: FirecrackerQualificationGuestCommand
    verified_guest_claims: tuple[FirecrackerQualificationClaim, ...]
    result_bytes_sha256: str = Field(pattern=_SHA256_PATTERN)
    responded_at: datetime

    @field_validator('responded_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('qualification guest response time must include a UTC offset')
        return value.astimezone(UTC)

    @field_validator('verified_guest_claims')
    @classmethod
    def validate_claims(
        cls, value: tuple[FirecrackerQualificationClaim, ...]
    ) -> tuple[FirecrackerQualificationClaim, ...]:
        if value != tuple(sorted(set(value), key=lambda item: item.value)):
            raise ValueError('guest response claims must be unique and sorted')
        return value


class AuthenticatedFirecrackerQualificationGuestResponse(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-firecracker-qualification-guest-response.v0.1'] = (
        AUTHENTICATED_FIRECRACKER_QUALIFICATION_GUEST_RESPONSE_SCHEMA_VERSION
    )
    response: FirecrackerQualificationGuestResponse
    authentication: Literal['ed25519-domain-separated'] = 'ed25519-domain-separated'
    guest_probe_key_id: str = Field(pattern=_SHA256_PATTERN)
    signature_hex: str = Field(pattern=_SIGNATURE_PATTERN)


class FirecrackerQualificationClaimMeasurement(StrictModel):
    claim: FirecrackerQualificationClaim
    source: FirecrackerQualificationObservationSource
    observed: bool
    raw_observation_sha256: str = Field(pattern=_SHA256_PATTERN)
    observed_at: datetime

    @field_validator('observed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('claim measurement time must include a UTC offset')
        return value.astimezone(UTC)


class FirecrackerQualificationCgroupSnapshot(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-cgroup-snapshot.v0.1'] = (
        FIRECRACKER_QUALIFICATION_CGROUP_SNAPSHOT_SCHEMA_VERSION
    )
    run_id: str = Field(pattern=_HEX_32_PATTERN)
    cgroup_path: str = Field(pattern=r'^/sys/fs/cgroup/[A-Za-z0-9_./-]+$', min_length=16, max_length=4096)
    cgroup_inode: int = Field(gt=0)
    observed_at: datetime
    cpu_max_quota_us: int = Field(gt=0)
    cpu_max_period_us: int = Field(gt=0)
    memory_max_bytes: int = Field(gt=0)
    memory_swap_max_bytes: int = Field(ge=0)
    pids_max: int = Field(gt=0)
    cpu_nr_throttled: int = Field(ge=0)
    cpu_throttled_usec: int = Field(ge=0)
    memory_oom: int = Field(ge=0)
    memory_oom_kill: int = Field(ge=0)
    pids_max_events: int = Field(ge=0)
    member_pids: tuple[int, ...]

    @field_validator('observed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('cgroup snapshot time must include a UTC offset')
        return value.astimezone(UTC)

    @field_validator('member_pids')
    @classmethod
    def validate_pids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != tuple(sorted(set(value))) or any(pid <= 1 for pid in value):
            raise ValueError('cgroup snapshot PIDs must be positive, unique, and sorted')
        return value


class FirecrackerQualificationHostCgroupCanary(StrictModel):
    """Bounded host helpers and the exact signed snapshots that measured them.

    The canary runs only in a qualification VM's already-bound worker cgroup.  Its two direct
    children are retained here so an independent verifier can distinguish controller pressure
    from the guest's unrelated virtual processes and prove that every helper disappeared while
    the original Firecracker process survived.
    """

    schema_version: Literal['vaxreplay.firecracker-qualification-host-cgroup-canary.v0.1'] = (
        FIRECRACKER_QUALIFICATION_HOST_CGROUP_CANARY_SCHEMA_VERSION
    )
    run_id: str = Field(pattern=_HEX_32_PATTERN)
    cgroup_path: str = Field(pattern=r'^/sys/fs/cgroup/[A-Za-z0-9_./-]+$', min_length=16, max_length=4096)
    cgroup_inode: int = Field(gt=0)
    firecracker_pid: int = Field(gt=1)
    memory_helper_pid: int = Field(gt=1)
    pids_helper_pid: int = Field(gt=1)
    pids_helper_descendant_pids: tuple[int, ...] = Field(min_length=1, max_length=512)
    started_monotonic_ns: int = Field(ge=0)
    memory_helper_reaped_monotonic_ns: int = Field(gt=0)
    pids_limit_observed_monotonic_ns: int = Field(gt=0)
    pids_helper_reaped_monotonic_ns: int = Field(gt=0)
    finished_monotonic_ns: int = Field(gt=0)
    allowed_duration_ns: int = Field(gt=0, le=10_000_000_000)
    memory_helper_wait_signal: Literal['SIGKILL'] = 'SIGKILL'
    pids_helper_exit_code: Literal[0] = 0
    baseline_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_pressure_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    memory_armed_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    memory_triggered_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    pids_peak_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    cleanup_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    all_helpers_reaped: Literal[True] = True

    @field_validator('pids_helper_descendant_pids')
    @classmethod
    def validate_descendants(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != tuple(sorted(set(value))) or any(pid <= 1 for pid in value):
            raise ValueError('cgroup canary descendant PIDs must be positive, unique, and sorted')
        return value

    @model_validator(mode='after')
    def validate_canary(self) -> Self:
        helper_pids = {self.memory_helper_pid, self.pids_helper_pid, *self.pids_helper_descendant_pids}
        if len(helper_pids) != len(self.pids_helper_descendant_pids) + 2 or self.firecracker_pid in helper_pids:
            raise ValueError('cgroup canary helper identities overlap')
        if not (
            self.started_monotonic_ns
            < self.memory_helper_reaped_monotonic_ns
            <= self.pids_limit_observed_monotonic_ns
            <= self.pids_helper_reaped_monotonic_ns
            <= self.finished_monotonic_ns
        ):
            raise ValueError('cgroup canary monotonic observations are out of order')
        if self.finished_monotonic_ns - self.started_monotonic_ns > self.allowed_duration_ns:
            raise ValueError('cgroup canary exceeded its retained duration bound')
        return self


class FirecrackerQualificationWallTimeoutMeasurement(StrictModel):
    """Host monotonic timeout observations.

    ``process_group_reaped_monotonic_ns`` is the existing wire-field name for the observation that
    the pinned worker cgroup had become empty. It is not the time of the asynchronous
    ``cgroup.kill`` write and is never synthesized from the nominal deadline.
    """

    run_id: str = Field(pattern=_HEX_32_PATTERN)
    process_group_id: int = Field(gt=1)
    armed_monotonic_ns: int = Field(ge=0)
    deadline_monotonic_ns: int = Field(gt=0)
    watchdog_triggered_monotonic_ns: int = Field(gt=0)
    process_group_reaped_monotonic_ns: int = Field(gt=0)
    allowed_teardown_grace_ns: int = Field(gt=0, le=60_000_000_000)
    kill_signal: Literal['SIGKILL'] = 'SIGKILL'
    member_pids_before_kill: tuple[int, ...] = Field(min_length=1)
    surviving_pids_after_reap: tuple[int, ...] = ()

    @model_validator(mode='after')
    def validate_timeline(self) -> Self:
        if not (
            self.armed_monotonic_ns
            < self.deadline_monotonic_ns
            <= self.watchdog_triggered_monotonic_ns
            <= self.process_group_reaped_monotonic_ns
        ):
            raise ValueError('wall-timeout monotonic observations are out of order')
        if self.process_group_reaped_monotonic_ns > self.deadline_monotonic_ns + self.allowed_teardown_grace_ns:
            raise ValueError('wall-timeout cgroup-empty observation exceeded the permitted grace bound')
        if self.surviving_pids_after_reap:
            raise ValueError('wall-timeout evidence cannot retain surviving process-group members')
        return self


class FirecrackerQualificationTeardownMeasurement(StrictModel):
    run_id: str = Field(pattern=_HEX_32_PATTERN)
    cgroup_path: str = Field(min_length=2, max_length=4096)
    jail_root: str = Field(min_length=2, max_length=4096)
    vsock_uds_path: str = Field(min_length=2, max_length=4096)
    cgroup_lstat_errno: Literal['ENOENT'] = 'ENOENT'
    jail_lstat_errno: Literal['ENOENT'] = 'ENOENT'
    vsock_lstat_errno: Literal['ENOENT'] = 'ENOENT'
    surviving_pids: tuple[int, ...] = ()
    observed_at: datetime

    @field_validator('observed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('teardown observation time must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_absence(self) -> Self:
        if self.surviving_pids:
            raise ValueError('teardown observation cannot retain surviving worker PIDs')
        return self


class FirecrackerQualificationWorkerInterval(StrictModel):
    run_id: str = Field(pattern=_HEX_32_PATTERN)
    started_monotonic_ns: int = Field(ge=0)
    finished_monotonic_ns: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.finished_monotonic_ns <= self.started_monotonic_ns:
            raise ValueError('load-canary worker interval must be positive')
        return self


class FirecrackerQualificationRawDrillObservation(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-raw-drill.v0.1'] = (
        FIRECRACKER_QUALIFICATION_RAW_DRILL_SCHEMA_VERSION
    )
    drill_id: FirecrackerQualificationDrillId
    challenge: FirecrackerQualificationChallenge
    started_at: datetime
    finished_at: datetime
    worker_bindings: tuple[FirecrackerQualificationWorkerBinding, ...] = Field(min_length=1, max_length=2)
    guest_responses: tuple[AuthenticatedFirecrackerQualificationGuestResponse, ...] = Field(max_length=2)
    claim_measurements: tuple[FirecrackerQualificationClaimMeasurement, ...] = Field(min_length=1, max_length=32)
    cgroup_snapshots: tuple[FirecrackerQualificationCgroupSnapshot, ...] = Field(default=(), max_length=16)
    host_cgroup_canary: FirecrackerQualificationHostCgroupCanary | None = None
    wall_timeout: FirecrackerQualificationWallTimeoutMeasurement | None = None
    teardown_measurements: tuple[FirecrackerQualificationTeardownMeasurement, ...] = Field(default=(), max_length=2)
    worker_intervals: tuple[FirecrackerQualificationWorkerInterval, ...] = Field(default=(), max_length=2)

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @field_validator('claim_measurements')
    @classmethod
    def validate_measurements(
        cls, value: tuple[FirecrackerQualificationClaimMeasurement, ...]
    ) -> tuple[FirecrackerQualificationClaimMeasurement, ...]:
        if tuple(item.claim for item in value) != tuple(
            sorted({item.claim for item in value}, key=lambda item: item.value)
        ):
            raise ValueError('claim measurements must be unique and sorted by claim')
        return value

    @model_validator(mode='after')
    def validate_drill_shape(self) -> Self:
        if self.drill_id != self.challenge.drill_id:
            raise ValueError('raw drill differs from its fresh challenge')
        if self.finished_at < self.started_at or self.started_at < self.challenge.issued_at:
            raise ValueError('raw drill time interval is invalid')
        run_ids = tuple(binding.run_id for binding in self.worker_bindings)
        if run_ids != self.challenge.run_ids:
            raise ValueError('raw drill worker bindings differ from challenged run IDs')
        required = required_firecracker_qualification_claims(self.drill_id)
        if tuple(item.claim for item in self.claim_measurements) != required:
            raise ValueError('raw drill must retain one measurement for every required claim')
        if any(item.source != _CLAIM_SOURCE[item.claim] for item in self.claim_measurements):
            raise ValueError('raw drill claim measurement used an untrusted observation source')
        for binding in self.worker_bindings:
            if (
                binding.worker_spec_sha256 != self.challenge.worker_spec_sha256
                or binding.probe_manifest_sha256 != self.challenge.probe_manifest_sha256
            ):
                raise ValueError('raw drill worker binding differs from the challenged release')
        expected_guest_count = 0 if self.drill_id == FirecrackerQualificationDrillId.TEARDOWN else len(run_ids)
        if len(self.guest_responses) != expected_guest_count:
            raise ValueError('raw drill has the wrong number of guest challenge responses')
        if self.drill_id == FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT:
            if len(self.cgroup_snapshots) != 6 or self.host_cgroup_canary is None:
                raise ValueError('cgroup drill requires six bound snapshots and one host-controller canary')
        elif self.cgroup_snapshots or self.host_cgroup_canary is not None:
            raise ValueError('only the cgroup drill may retain cgroup stress evidence')
        if (self.wall_timeout is not None) != (self.drill_id == FirecrackerQualificationDrillId.WALL_TIMEOUT):
            raise ValueError('only the wall-timeout drill may retain watchdog timing')
        expected_teardowns = (
            len(run_ids)
            if self.drill_id in {FirecrackerQualificationDrillId.TEARDOWN, FirecrackerQualificationDrillId.LOAD_CANARY}
            else 0
        )
        if len(self.teardown_measurements) != expected_teardowns:
            raise ValueError('teardown observations are incomplete or attached to the wrong drill')
        expected_intervals = len(run_ids) if self.drill_id == FirecrackerQualificationDrillId.LOAD_CANARY else 0
        if len(self.worker_intervals) != expected_intervals:
            raise ValueError('only the load-canary drill may retain two worker intervals')
        return self


class FirecrackerQualificationRawCollection(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-raw-collection.v0.2'] = (
        'vaxreplay.firecracker-qualification-raw-collection.v0.2'
    )
    collection_id: str = Field(pattern=_HEX_32_PATTERN)
    mode: FirecrackerQualificationCollectionMode
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_manifest: FirecrackerQualificationProbeManifest
    probe_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    boundary_identity: FirecrackerQualificationBoundaryIdentity
    driver_runtime_closure_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    collector_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    host_observation: FirecrackerHostObservation
    host_preflight: FirecrackerHostPreflightReceipt
    host_preflight_sha256: str = Field(pattern=_SHA256_PATTERN)
    drills: tuple[FirecrackerQualificationRawDrillObservation, ...] = Field(min_length=7, max_length=7)
    collected_at: datetime
    caller_supplied_drill_evidence_accepted: Literal[False] = False
    development_simulated: bool
    production_qualification_eligible: bool

    @field_validator('collected_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('raw qualification collection time must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_collection(self) -> Self:
        if self.probe_manifest_sha256 != firecracker_model_sha256(self.probe_manifest):
            raise ValueError('probe manifest digest does not match its embedded manifest')
        if (
            self.worker_spec_sha256 != self.probe_manifest.task_worker_spec_sha256
            or self.host_preflight.worker_spec_sha256 != self.worker_spec_sha256
            or self.host_preflight_sha256 != firecracker_model_sha256(self.host_preflight)
        ):
            raise ValueError('raw collection does not bind one worker specification and preflight')
        expected_ids = tuple(FirecrackerQualificationDrillId)
        if tuple(drill.drill_id for drill in self.drills) != expected_ids:
            raise ValueError('raw collection must contain all seven drills in canonical order')
        challenge_ids = tuple(drill.challenge.challenge_id for drill in self.drills)
        nonces = tuple(drill.challenge.nonce_hex for drill in self.drills)
        run_ids = tuple(run_id for drill in self.drills for run_id in drill.challenge.run_ids)
        if len(challenge_ids) != len(set(challenge_ids)) or len(nonces) != len(set(nonces)):
            raise ValueError('every qualification drill requires a fresh challenge and nonce')
        if len(run_ids) != len(set(run_ids)):
            raise ValueError('every qualification worker run requires a fresh run ID')
        for drill in self.drills:
            if (
                drill.challenge.collection_id != self.collection_id
                or drill.challenge.worker_spec_sha256 != self.worker_spec_sha256
                or drill.challenge.probe_manifest_sha256 != self.probe_manifest_sha256
            ):
                raise ValueError('qualification challenge is not bound to this collection')
        if self.collected_at < max(drill.finished_at for drill in self.drills):
            raise ValueError('raw collection cannot finish before its final drill observation')
        production = self.mode == FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM
        if (self.development_simulated, self.production_qualification_eligible) != (not production, production):
            raise ValueError('collection eligibility flags must exactly match its immutable mode')
        boundary_production = (
            self.boundary_identity.kind == FirecrackerQualificationBoundaryKind.PINNED_LINUX_KVM_DRIVER
        )
        if production != boundary_production:
            raise ValueError('collection mode differs from its concrete probe boundary')
        if self.driver_runtime_closure_sha256 != self.boundary_identity.runtime_closure_sha256:
            raise ValueError('raw collection differs from its boundary runtime-closure digest')
        if production != (self.driver_runtime_closure_sha256 is not None):
            raise ValueError('only production raw evidence may bind a qualification-driver runtime closure')
        if production and not _host_is_production_linux_kvm(self.host_observation):
            raise ValueError('production collection requires a root Linux/KVM/cgroup-v2 host observation')
        return self


class AuthenticatedFirecrackerQualificationRawCollection(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-firecracker-qualification-raw-collection.v0.1'] = (
        AUTHENTICATED_FIRECRACKER_QUALIFICATION_RAW_COLLECTION_SCHEMA_VERSION
    )
    collection: FirecrackerQualificationRawCollection
    authentication: Literal['ed25519-domain-separated'] = 'ed25519-domain-separated'
    collector_public_key_hex: str = Field(pattern=_HEX_64_PATTERN)
    collector_key_id: str = Field(pattern=_SHA256_PATTERN)
    signature_hex: str = Field(pattern=_SIGNATURE_PATTERN)


class VerifiedFirecrackerQualificationCollection(StrictModel):
    authenticated: AuthenticatedFirecrackerQualificationRawCollection
    authenticated_collection_sha256: str = Field(pattern=_SHA256_PATTERN)
    full_suite_evidence: FirecrackerFullSuiteEvidence
    verifier_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    production_qualification_eligible: Literal[True] = True


class FirecrackerQualificationProbeBoundary(Protocol):
    """Deployment boundary that performs the actual guest/host observations for each drill."""

    @property
    def identity(self) -> FirecrackerQualificationBoundaryIdentity: ...

    def live_boot(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation: ...

    def vsock_round_trip(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation: ...

    def guest_isolation(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation: ...

    def cgroup_enforcement(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation: ...

    def wall_timeout(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation: ...

    def teardown(self, challenge: FirecrackerQualificationChallenge) -> FirecrackerQualificationRawDrillObservation: ...

    def load_canary(
        self, challenge: FirecrackerQualificationChallenge
    ) -> FirecrackerQualificationRawDrillObservation: ...


def firecracker_qualification_probe_manifest_sha256(manifest: FirecrackerQualificationProbeManifest) -> str:
    return firecracker_model_sha256(manifest)


def firecracker_qualification_verifier_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def firecracker_qualification_challenge_sha256(challenge: FirecrackerQualificationChallenge) -> str:
    return firecracker_model_sha256(challenge)


def firecracker_qualification_worker_binding_sha256(binding: FirecrackerQualificationWorkerBinding) -> str:
    return firecracker_model_sha256(binding)


def firecracker_qualification_guest_key_id(public_key: bytes) -> str:
    if len(public_key) != 32:
        raise FirecrackerQualificationProbeError('qualification guest public key must be exactly 32 bytes')
    return hashlib.sha256(_GUEST_KEY_ID_DOMAIN + public_key).hexdigest()


def firecracker_live_collector_key_id(public_key: bytes) -> str:
    if len(public_key) != 32:
        raise FirecrackerQualificationProbeError('collector public key must be exactly 32 bytes')
    return hashlib.sha256(_COLLECTOR_KEY_ID_DOMAIN + public_key).hexdigest()


def ed25519_public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def sign_firecracker_qualification_guest_response(
    response: FirecrackerQualificationGuestResponse,
    *,
    private_key: Ed25519PrivateKey,
) -> AuthenticatedFirecrackerQualificationGuestResponse:
    public_key = ed25519_public_key_bytes(private_key)
    return AuthenticatedFirecrackerQualificationGuestResponse(
        response=response,
        guest_probe_key_id=firecracker_qualification_guest_key_id(public_key),
        signature_hex=private_key.sign(_GUEST_SIGNATURE_DOMAIN + canonical_json_bytes(response)).hex(),
    )


def authenticate_firecracker_qualification_collection(
    collection: FirecrackerQualificationRawCollection,
    *,
    private_key: Ed25519PrivateKey,
) -> AuthenticatedFirecrackerQualificationRawCollection:
    public_key = ed25519_public_key_bytes(private_key)
    return AuthenticatedFirecrackerQualificationRawCollection(
        collection=collection,
        collector_public_key_hex=public_key.hex(),
        collector_key_id=firecracker_live_collector_key_id(public_key),
        signature_hex=private_key.sign(_COLLECTOR_SIGNATURE_DOMAIN + canonical_json_bytes(collection)).hex(),
    )


def verify_authenticated_firecracker_qualification_collection(
    authenticated: AuthenticatedFirecrackerQualificationRawCollection,
    *,
    worker_spec: FirecrackerWorkerSpec,
    expected_collector_public_key_hex: str,
    expected_collector_key_id: str,
    expected_worker_spec_sha256: str,
    expected_probe_manifest_sha256: str,
    expected_driver_runtime_closure_manifest_sha256: str,
    expected_driver_runtime_closure_receipt_sha256: str,
    expected_driver_runtime_closure_sha256: str,
    expected_host_preflight_sha256: str | None = None,
    verifier_source_sha256: str,
) -> VerifiedFirecrackerQualificationCollection:
    """Authenticate raw evidence and independently derive the only positive suite result."""

    _require_sha256(expected_worker_spec_sha256, 'worker-spec pin')
    _require_sha256(expected_probe_manifest_sha256, 'probe-manifest pin')
    _require_sha256(expected_driver_runtime_closure_manifest_sha256, 'driver runtime-closure manifest pin')
    _require_sha256(expected_driver_runtime_closure_receipt_sha256, 'driver runtime-closure receipt pin')
    _require_sha256(expected_driver_runtime_closure_sha256, 'driver runtime-closure pin')
    _require_sha256(verifier_source_sha256, 'verifier source pin')
    if not hmac.compare_digest(verifier_source_sha256, firecracker_qualification_verifier_source_sha256()):
        raise FirecrackerQualificationProbeError('verifier source differs from its external release pin')
    if expected_host_preflight_sha256 is not None:
        _require_sha256(expected_host_preflight_sha256, 'host-preflight pin')
    verify_firecracker_qualification_collection_authentication(
        authenticated,
        expected_collector_public_key_hex=expected_collector_public_key_hex,
        expected_collector_key_id=expected_collector_key_id,
    )
    collection = authenticated.collection
    if collection.mode != FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM:
        raise FirecrackerQualificationProbeError('development/simulated collector evidence can never qualify')
    if not collection.production_qualification_eligible or collection.development_simulated:
        raise FirecrackerQualificationProbeError('collector evidence is structurally ineligible for production')
    if (
        collection.worker_spec_sha256 != expected_worker_spec_sha256
        or collection.probe_manifest_sha256 != expected_probe_manifest_sha256
        or firecracker_model_sha256(worker_spec) != expected_worker_spec_sha256
    ):
        raise FirecrackerQualificationProbeError('collector evidence differs from an external release pin')
    boundary = collection.boundary_identity
    if (
        not boundary.transitive_runtime_pin_enforced
        or boundary.runtime_closure_manifest_sha256 != expected_driver_runtime_closure_manifest_sha256
        or boundary.runtime_closure_receipt_sha256 != expected_driver_runtime_closure_receipt_sha256
        or boundary.runtime_closure_sha256 != expected_driver_runtime_closure_sha256
        or collection.driver_runtime_closure_sha256 != expected_driver_runtime_closure_sha256
    ):
        raise FirecrackerQualificationProbeError(
            'collector evidence differs from the externally pinned qualification-driver runtime closure'
        )
    if (
        expected_host_preflight_sha256 is not None
        and collection.host_preflight_sha256 != expected_host_preflight_sha256
    ):
        raise FirecrackerQualificationProbeError('collector evidence differs from the pinned host preflight')
    full_suite = derive_firecracker_full_suite_evidence(collection, worker_spec=worker_spec)
    if not full_suite.all_required_drills_passed:
        raise FirecrackerQualificationProbeError('one or more independently verified live drills failed')
    return VerifiedFirecrackerQualificationCollection(
        authenticated=authenticated,
        authenticated_collection_sha256=hashlib.sha256(canonical_json_bytes(authenticated)).hexdigest(),
        full_suite_evidence=full_suite,
        verifier_source_sha256=verifier_source_sha256,
    )


def verify_firecracker_qualification_collection_authentication(
    authenticated: AuthenticatedFirecrackerQualificationRawCollection,
    *,
    expected_collector_public_key_hex: str,
    expected_collector_key_id: str,
) -> None:
    """Authenticate either production or explicitly simulated raw evidence without qualifying it."""

    try:
        public_key_bytes = bytes.fromhex(expected_collector_public_key_hex)
    except ValueError as error:
        raise FirecrackerQualificationProbeError('collector public key pin is not lowercase hexadecimal') from error
    if len(public_key_bytes) != 32 or expected_collector_public_key_hex != public_key_bytes.hex():
        raise FirecrackerQualificationProbeError('collector public key pin must be 32 lowercase hexadecimal bytes')
    observed_key_id = firecracker_live_collector_key_id(public_key_bytes)
    if (
        not hmac.compare_digest(observed_key_id, expected_collector_key_id)
        or not hmac.compare_digest(authenticated.collector_key_id, expected_collector_key_id)
        or not hmac.compare_digest(authenticated.collector_public_key_hex, expected_collector_public_key_hex)
    ):
        raise FirecrackerQualificationProbeError('collector identity differs from its external public-key pin')
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            bytes.fromhex(authenticated.signature_hex),
            _COLLECTOR_SIGNATURE_DOMAIN + canonical_json_bytes(authenticated.collection),
        )
    except (InvalidSignature, ValueError) as error:
        raise FirecrackerQualificationProbeError('collector signature authentication failed') from error


def derive_firecracker_full_suite_evidence(
    collection: FirecrackerQualificationRawCollection,
    *,
    worker_spec: FirecrackerWorkerSpec | None = None,
) -> FirecrackerFullSuiteEvidence:
    """Derive claim dispositions from authenticated raw measurements, never caller booleans."""

    if worker_spec is not None and firecracker_model_sha256(worker_spec) != collection.worker_spec_sha256:
        raise FirecrackerQualificationProbeError('worker specification differs from the raw collection')
    if worker_spec is not None:
        qualification_spec = derive_firecracker_qualification_worker_spec(
            collection.probe_manifest,
            task_worker_spec=worker_spec,
        )
        qualification_spec_sha256 = firecracker_model_sha256(qualification_spec)
        qualification_config_sha256 = firecracker_qualification_static_config_sha256(qualification_spec)
        if any(
            binding.qualification_worker_spec_sha256 != qualification_spec_sha256
            or binding.qualification_static_config_sha256 != qualification_config_sha256
            or not _worker_binding_matches_spec(binding, qualification_spec)
            for drill in collection.drills
            for binding in drill.worker_bindings
        ):
            raise FirecrackerQualificationProbeError(
                'observed qualification worker differs from the independently reconstructed probe worker'
            )
    derived = tuple(_derive_drill(collection, drill, worker_spec=worker_spec) for drill in collection.drills)
    return FirecrackerFullSuiteEvidence(
        worker_spec_sha256=collection.worker_spec_sha256,
        host_preflight_sha256=collection.host_preflight_sha256,
        collected_on_linux_kvm=(
            collection.mode == FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM
            and _host_is_production_linux_kvm(collection.host_observation)
        ),
        live_boot=derived[0],
        vsock_round_trip=derived[1],
        guest_isolation=derived[2],
        cgroup_enforcement=derived[3],
        wall_timeout=derived[4],
        teardown=derived[5],
        load_canary=derived[6],
    )


def _worker_binding_matches_spec(
    binding: FirecrackerQualificationWorkerBinding,
    spec: FirecrackerWorkerSpec,
) -> bool:
    executable_name = Path(spec.runtime.firecracker.source_path).name
    expected_jail_root = Path(spec.chroot_base_dir) / executable_name / binding.run_id / 'root'
    expected_cgroup = Path('/sys/fs/cgroup').joinpath(*spec.cgroup_parent.split('/'), binding.run_id)
    return (
        binding.worker_uid == spec.worker_uid
        and binding.worker_gid == spec.worker_gid
        and binding.guest_cid == spec.guest_cid
        and binding.firecracker_executable_sha256 == spec.runtime.firecracker.sha256
        and Path(binding.cgroup_path) == expected_cgroup
        and Path(binding.jail_root) == expected_jail_root
        and Path(binding.vsock_uds_path) == expected_jail_root / 'run' / 'vsock.sock'
        and Path(binding.firecracker_pid_file_path) == expected_jail_root / f'{executable_name}.pid'
        and binding.firecracker_pid in binding.cgroup_member_pids
        and binding.peer_pid == binding.firecracker_pid
        and binding.process_group_id == binding.jailer_pid
        and binding.jailer_process_group_id == binding.jailer_pid
    )


def validate_probe_manifest_for_worker(
    manifest: FirecrackerQualificationProbeManifest,
    spec: FirecrackerWorkerSpec,
) -> None:
    if (
        manifest.task_worker_spec_sha256 != firecracker_model_sha256(spec)
        or manifest.task_rootfs_sha256 != spec.images.rootfs.sha256
        or manifest.task_harness_sha256 != spec.images.harness.sha256
        or manifest.qualification_kernel_sha256 != spec.images.kernel.sha256
    ):
        raise FirecrackerQualificationProbeError('qualification probe manifest differs from the task worker release')
    derive_firecracker_qualification_worker_spec(manifest, task_worker_spec=spec)


def derive_firecracker_qualification_worker_spec(
    manifest: FirecrackerQualificationProbeManifest,
    *,
    task_worker_spec: FirecrackerWorkerSpec,
) -> FirecrackerWorkerSpec:
    """Replace only task rootfs/harness bytes with the independently built probe disks.

    Runtime binaries, kernel, scratch template, resource limits, account, cgroup/jail roots,
    guest CID/port, boot arguments, and all isolation flags remain exactly the task release.  The
    returned model is the only worker spec the live qualification driver may hand to the
    ``FirecrackerSupervisor``.
    """

    if (
        manifest.task_worker_spec_sha256 != firecracker_model_sha256(task_worker_spec)
        or manifest.task_rootfs_sha256 != task_worker_spec.images.rootfs.sha256
        or manifest.task_harness_sha256 != task_worker_spec.images.harness.sha256
        or manifest.qualification_kernel_sha256 != task_worker_spec.images.kernel.sha256
    ):
        raise FirecrackerQualificationProbeError('qualification probe manifest differs from the task worker release')
    qualification_images = FirecrackerGuestImages(
        kernel=task_worker_spec.images.kernel,
        rootfs=FirecrackerArtifactIdentity(
            artifact_id=f'{manifest.manifest_id}-rootfs',
            role=FirecrackerArtifactRole.ROOTFS,
            source_path=manifest.qualification_rootfs_path,
            sha256=manifest.qualification_rootfs_sha256,
            byte_count=manifest.qualification_rootfs_byte_count,
        ),
        harness=FirecrackerArtifactIdentity(
            artifact_id=f'{manifest.manifest_id}-harness',
            role=FirecrackerArtifactRole.HARNESS,
            source_path=manifest.qualification_harness_path,
            sha256=manifest.qualification_harness_sha256,
            byte_count=manifest.qualification_harness_byte_count,
        ),
        scratch_template=task_worker_spec.images.scratch_template,
    )
    return task_worker_spec.model_copy(update={'images': qualification_images})


def firecracker_qualification_static_config_sha256(spec: FirecrackerWorkerSpec) -> str:
    return hashlib.sha256(firecracker_static_config_bytes(spec)).hexdigest()


def _derive_drill(
    collection: FirecrackerQualificationRawCollection,
    drill: FirecrackerQualificationRawDrillObservation,
    *,
    worker_spec: FirecrackerWorkerSpec | None,
) -> FirecrackerQualificationDrillEvidence:
    guest_valid = _verify_guest_responses(collection, drill)
    verified = {
        measurement.claim
        for measurement in drill.claim_measurements
        if measurement.observed
        and (measurement.source != FirecrackerQualificationObservationSource.GUEST_SIGNED or guest_valid)
    }
    if any(
        not binding.process_tree_verified or not binding.pid_cgroup_binding_verified
        for binding in drill.worker_bindings
    ):
        verified.clear()
    if drill.drill_id == FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT:
        verified &= _derived_cgroup_claims(
            drill.cgroup_snapshots,
            drill.host_cgroup_canary,
            collection,
            worker_spec=worker_spec,
        )
    elif drill.drill_id == FirecrackerQualificationDrillId.WALL_TIMEOUT:
        if drill.wall_timeout is None:
            verified.clear()
        else:
            verified &= {
                FirecrackerQualificationClaim.WALL_WATCHDOG_TRIGGERED,
                FirecrackerQualificationClaim.PROCESS_GROUP_KILLED,
                FirecrackerQualificationClaim.DEADLINE_BOUND,
            }
    elif drill.drill_id == FirecrackerQualificationDrillId.TEARDOWN:
        verified &= _derived_teardown_claims(drill)
    elif drill.drill_id == FirecrackerQualificationDrillId.LOAD_CANARY:
        verified &= _derived_load_canary_claims(drill)
    required = frozenset(required_firecracker_qualification_claims(drill.drill_id))
    verified &= required
    failed = required - verified
    guest_hash = hashlib.sha256(
        b''.join(canonical_json_bytes(item) for item in drill.guest_responses) or b'no-guest-response'
    ).hexdigest()
    return FirecrackerQualificationDrillEvidence(
        drill_id=drill.drill_id,
        passed=not failed,
        started_at=drill.started_at,
        finished_at=drill.finished_at,
        run_ids=drill.challenge.run_ids,
        evidence_artifact_sha256=firecracker_model_sha256(drill),
        authenticated_worker_attestation_sha256=guest_hash,
        observer_executable_sha256=collection.boundary_identity.executable_sha256,
        observation_count=(
            len(drill.claim_measurements)
            + len(drill.guest_responses)
            + len(drill.cgroup_snapshots)
            + len(drill.teardown_measurements)
            + len(drill.worker_intervals)
            + (1 if drill.host_cgroup_canary is not None else 0)
            + (1 if drill.wall_timeout is not None else 0)
        ),
        verified_claims=tuple(sorted(verified, key=lambda item: item.value)),
        failed_claims=tuple(sorted(failed, key=lambda item: item.value)),
    )


def _verify_guest_responses(
    collection: FirecrackerQualificationRawCollection,
    drill: FirecrackerQualificationRawDrillObservation,
) -> bool:
    expected_command = _GUEST_COMMAND_FOR_DRILL[drill.drill_id]
    if expected_command is None:
        return not drill.guest_responses
    manifest = collection.probe_manifest
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(manifest.guest_probe_public_key_hex))
    except ValueError:
        return False
    by_run = {binding.run_id: binding for binding in drill.worker_bindings}
    expected_guest_claims = {
        measurement.claim
        for measurement in drill.claim_measurements
        if measurement.source == FirecrackerQualificationObservationSource.GUEST_SIGNED
    }
    if drill.drill_id == FirecrackerQualificationDrillId.LOAD_CANARY:
        # For the canary, receiving one valid signed response for each challenged run is itself
        # the completion evidence.  The guest does not self-assert that the host observed both
        # workers concurrently or later tore them down.
        expected_guest_claims.discard(FirecrackerQualificationClaim.ALL_WORKERS_COMPLETED)
    seen: set[str] = set()
    for authenticated in drill.guest_responses:
        response = authenticated.response
        binding = by_run.get(response.run_id)
        if binding is None or response.run_id in seen:
            return False
        seen.add(response.run_id)
        if (
            authenticated.guest_probe_key_id != manifest.guest_probe_key_id
            or response.challenge_sha256 != firecracker_qualification_challenge_sha256(drill.challenge)
            or response.nonce_hex != drill.challenge.nonce_hex
            or response.worker_binding_sha256 != firecracker_qualification_worker_binding_sha256(binding)
            or response.worker_spec_sha256 != collection.worker_spec_sha256
            or response.probe_manifest_sha256 != collection.probe_manifest_sha256
            or response.guest_probe_executable_sha256 != manifest.guest_probe_executable_sha256
            or response.command != expected_command
            or not expected_guest_claims.issubset(response.verified_guest_claims)
        ):
            return False
        try:
            public_key.verify(
                bytes.fromhex(authenticated.signature_hex),
                _GUEST_SIGNATURE_DOMAIN + canonical_json_bytes(response),
            )
        except (InvalidSignature, ValueError):
            return False
    return seen == set(drill.challenge.run_ids)


def _derived_cgroup_claims(
    snapshots: tuple[FirecrackerQualificationCgroupSnapshot, ...],
    canary: FirecrackerQualificationHostCgroupCanary | None,
    collection: FirecrackerQualificationRawCollection,
    *,
    worker_spec: FirecrackerWorkerSpec | None,
) -> set[FirecrackerQualificationClaim]:
    if len(snapshots) != 6 or canary is None:
        return set()
    before, guest_pressure, memory_armed, memory_triggered, pids_peak, after = snapshots
    binding = collection.drills[3].worker_bindings[0]
    expected_snapshot_hashes = (
        canary.baseline_snapshot_sha256,
        canary.guest_pressure_snapshot_sha256,
        canary.memory_armed_snapshot_sha256,
        canary.memory_triggered_snapshot_sha256,
        canary.pids_peak_snapshot_sha256,
        canary.cleanup_snapshot_sha256,
    )
    observed_snapshot_hashes = tuple(firecracker_model_sha256(snapshot) for snapshot in snapshots)
    expected_identity = (binding.run_id, binding.cgroup_path, binding.cgroup_inode)
    snapshot_times = tuple(snapshot.observed_at for snapshot in snapshots)
    if (
        (canary.run_id, canary.cgroup_path, canary.cgroup_inode) != expected_identity
        or canary.firecracker_pid != binding.firecracker_pid
        or any((item.run_id, item.cgroup_path, item.cgroup_inode) != expected_identity for item in snapshots)
        or expected_snapshot_hashes != observed_snapshot_hashes
        or any(later < earlier for earlier, later in zip(snapshot_times, snapshot_times[1:]))
        or after.observed_at <= before.observed_at
    ):
        return set()
    if worker_spec is None:
        return set()
    limits = worker_spec.limits
    expected_memory_bytes = limits.memory_mib * 1024 * 1024
    if any(
        snapshot.cpu_max_quota_us != limits.cpu_quota_us
        or snapshot.cpu_max_period_us != limits.cpu_period_us
        or snapshot.memory_max_bytes != expected_memory_bytes
        or snapshot.memory_swap_max_bytes != 0
        or snapshot.pids_max != limits.pids
        for snapshot in snapshots
    ):
        return set()
    baseline_members = set(binding.cgroup_member_pids)
    pids_descendants = set(canary.pids_helper_descendant_pids)
    if (
        before.member_pids != binding.cgroup_member_pids
        or guest_pressure.member_pids != binding.cgroup_member_pids
        or set(memory_armed.member_pids) != baseline_members | {canary.memory_helper_pid}
        or memory_triggered.member_pids != binding.cgroup_member_pids
        or set(pids_peak.member_pids) != baseline_members | {canary.pids_helper_pid} | pids_descendants
        or after.member_pids != binding.cgroup_member_pids
        or binding.firecracker_pid not in set.intersection(*(set(snapshot.member_pids) for snapshot in snapshots))
    ):
        return set()
    verified: set[FirecrackerQualificationClaim] = set()
    if after.cpu_nr_throttled > before.cpu_nr_throttled and after.cpu_throttled_usec > before.cpu_throttled_usec:
        verified.add(FirecrackerQualificationClaim.CPU_LIMIT_OBSERVED)
    if (
        memory_triggered.memory_oom > memory_armed.memory_oom
        and memory_triggered.memory_oom_kill > memory_armed.memory_oom_kill
        and after.memory_oom >= memory_triggered.memory_oom
        and after.memory_oom_kill >= memory_triggered.memory_oom_kill
    ):
        verified.add(FirecrackerQualificationClaim.MEMORY_LIMIT_OBSERVED)
    if before.memory_swap_max_bytes == 0 and after.memory_swap_max_bytes == 0:
        verified.add(FirecrackerQualificationClaim.SWAP_DISABLED_OBSERVED)
    if (
        pids_peak.pids_max_events > memory_triggered.pids_max_events
        and after.pids_max_events >= pids_peak.pids_max_events
    ):
        verified.add(FirecrackerQualificationClaim.PIDS_LIMIT_OBSERVED)
    return verified


def _derived_teardown_claims(
    drill: FirecrackerQualificationRawDrillObservation,
) -> set[FirecrackerQualificationClaim]:
    if len(drill.teardown_measurements) != 1:
        return set()
    teardown = drill.teardown_measurements[0]
    binding = drill.worker_bindings[0]
    if (
        teardown.run_id,
        teardown.cgroup_path,
        teardown.jail_root,
        teardown.vsock_uds_path,
    ) != (binding.run_id, binding.cgroup_path, binding.jail_root, binding.vsock_uds_path):
        return set()
    return {
        FirecrackerQualificationClaim.CGROUP_ABSENT,
        FirecrackerQualificationClaim.JAIL_ABSENT,
        FirecrackerQualificationClaim.VSOCK_ABSENT,
    }


def _derived_load_canary_claims(
    drill: FirecrackerQualificationRawDrillObservation,
) -> set[FirecrackerQualificationClaim]:
    if len(drill.worker_bindings) != 2 or len(drill.worker_intervals) != 2 or len(drill.teardown_measurements) != 2:
        return set()
    bindings = drill.worker_bindings
    distinct_sets = (
        {item.firecracker_pid for item in bindings},
        {item.process_group_id for item in bindings},
        {item.cgroup_path for item in bindings},
        {item.jail_root for item in bindings},
        {item.vsock_uds_path for item in bindings},
    )
    intervals = drill.worker_intervals
    overlap = max(item.started_monotonic_ns for item in intervals) < min(
        item.finished_monotonic_ns for item in intervals
    )
    teardown_runs = {item.run_id for item in drill.teardown_measurements}
    binding_runs = {item.run_id for item in bindings}
    verified: set[FirecrackerQualificationClaim] = set()
    if all(len(values) == 2 for values in distinct_sets) and overlap:
        verified.add(FirecrackerQualificationClaim.PARALLEL_WORKERS_DISTINCT)
    # ``_verify_guest_responses`` has already authenticated one response for every challenged
    # run.  LOAD_CANARY is a liveness/overlap drill, so the signed completion responses need no
    # additional scientific claim payload.
    if (
        len(drill.guest_responses) == 2
        and {response.response.run_id for response in drill.guest_responses} == binding_runs
    ):
        verified.add(FirecrackerQualificationClaim.ALL_WORKERS_COMPLETED)
    if teardown_runs == binding_runs:
        verified.add(FirecrackerQualificationClaim.ALL_WORKERS_TORN_DOWN)
    return verified


def _host_is_production_linux_kvm(observation: FirecrackerHostObservation) -> bool:
    architecture = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(
        observation.host_architecture.lower(), observation.host_architecture.lower()
    )
    return (
        observation.host_os == 'Linux'
        and architecture in {'x86_64', 'aarch64'}
        and observation.effective_uid == 0
        and observation.kvm_non_symlink_character_device
        and observation.kvm_read_write_access
        and observation.cgroup_v2_controller_file_present
        and {'cpu', 'memory', 'pids'}.issubset(observation.cgroup_controllers)
    )


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise FirecrackerQualificationProbeError(f'{label} must be a lowercase SHA-256 digest')
