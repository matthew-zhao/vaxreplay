"""Same-host reproducible Lane A task rootfs and harness disk construction.

The production boundary in this module is deliberately narrow: source artifacts are externally
SHA-256 pinned, ext4 tools are opened once and executed through ``/proc/self/fd``, and two clean
stagings on one host must produce byte-identical images.  This is not a cross-host or hermetic
reproducibility claim: the receipt says so explicitly and separately commits the declared tool
runtime closure.  A command-executor seam exists so the deterministic orchestration can be tested
away from a Linux/KVM build host; receipts created through that seam are explicitly ineligible for
production.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from vaxreplay.agentic.guest_boot_dispatch import (
    GUEST_CONFIG_DIGEST_FLAG,
    NATIVE_GUEST_CONFIG_PATH,
    NATIVE_GUEST_EXECUTABLE_PATH,
    GuestBootDispatchAdmission,
    GuestBootDispatchError,
    GuestBootDispatchManifest,
    guest_boot_dispatch_manifest_sha256,
    make_native_guest_boot_dispatch_manifest,
    validate_guest_boot_config_bytes,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

LANE_A_GUEST_DISK_BUILD_SCHEMA_VERSION = 'vaxreplay.lane-a-guest-disk-build.dev-v0.3'
LANE_A_GUEST_DISK_BUILD_PROFILE = 'lane_a_manifest_dispatched_task_guest_v2'
LANE_A_TOOL_RUNTIME_CLOSURE_SCHEMA_VERSION = 'vaxreplay.lane-a-tool-runtime-closure.v0.1'

LANE_A_ROOTFS_UUID = '00000000-0000-4000-8000-000000000201'
LANE_A_HARNESS_UUID = '00000000-0000-4000-8000-000000000202'
LANE_A_ROOTFS_LABEL = 'vaxlanea-root'
LANE_A_HARNESS_LABEL = 'vaxlanea-harness'
LANE_A_GUEST_EXECUTABLE_PATH = NATIVE_GUEST_EXECUTABLE_PATH
LANE_A_GUEST_CONFIG_PATH = NATIVE_GUEST_CONFIG_PATH
LANE_A_GUEST_CONFIG_DIGEST_FLAG = GUEST_CONFIG_DIGEST_FLAG

DEFAULT_LANE_A_ROOTFS_BYTES = 512 * 1024 * 1024
DEFAULT_LANE_A_HARNESS_BYTES = 512 * 1024 * 1024

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_TOOL_ENVIRONMENT = {
    'LANG': 'C',
    'LC_ALL': 'C',
    'MKE2FS_CONFIG': '/dev/null',
    'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
}
_MKE2FS_FEATURES = (
    'none,has_journal,ext_attr,resize_inode,dir_index,filetype,extent,64bit,'
    'flex_bg,sparse_super,large_file,huge_file,dir_nlink,extra_isize'
)
_MKE2FS_ARGV_TEMPLATE = (
    '$MKE2FS',
    '-q',
    '-t',
    'ext4',
    '-F',
    '-b',
    '4096',
    '-I',
    '256',
    '-i',
    '16384',
    '-m',
    '0',
    '-U',
    '$UUID',
    '-L',
    '$LABEL',
    '-O',
    _MKE2FS_FEATURES,
    '-E',
    'lazy_itable_init=0,lazy_journal_init=0,root_owner=0:0,hash_seed=$UUID',
    '-d',
    '$TREE',
    '$OUTPUT',
)
_SAFE_PATH_COMPONENT = re.compile(r'^[A-Za-z0-9._+@-]+$')
_MAX_ARCHIVE_MEMBERS = 1_000_000
_MAX_SOURCE_BYTES = 64 * 1024 * 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_TOOL_RUNTIME_CLOSURE_BYTES = 1024 * 1024


class LaneAGuestDiskBuildError(RuntimeError):
    """A source, tool, image, receipt, or reproducibility check failed closed."""


class GuestDiskSourceKind(str, enum.Enum):
    NORMALIZED_TAR = 'normalized_tar'
    TEST_DIRECTORY = 'test_directory'


class GuestDiskExecutionBoundary(str, enum.Enum):
    PINNED_LINUX_PROCFS = 'pinned_linux_procfs'
    TEST_EXECUTOR = 'test_executor'


class GuestDiskToolLinkage(str, enum.Enum):
    STATIC_ELF = 'static_elf'
    DYNAMIC_ELF = 'dynamic_elf'
    TEST_UNKNOWN = 'test_unknown'


class GuestDiskSourceIdentity(StrictModel):
    kind: GuestDiskSourceKind
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0, le=64 * 1024 * 1024 * 1024)
    normalized_tree_sha256: str = Field(pattern=_SHA256_PATTERN)


class GuestDiskToolIdentity(StrictModel):
    name: Literal['mke2fs', 'e2fsck', 'debugfs']
    sha256: str = Field(pattern=_SHA256_PATTERN)
    version: str = Field(min_length=1, max_length=200)
    executed_via_proc_self_fd: bool
    linkage: GuestDiskToolLinkage


class GuestDiskToolRuntimeDependency(StrictModel):
    """One declared non-tool byte dependency in the ext4 tool execution runtime."""

    logical_name: str = Field(min_length=1, max_length=500, pattern=r'^[A-Za-z0-9/._+@:-]+$')
    role: Literal['elf_interpreter', 'shared_library', 'other_runtime_dependency']
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=1024 * 1024 * 1024 * 1024)


class GuestDiskToolRuntimeBinding(StrictModel):
    """Declared runtime closure for one externally pinned tool executable."""

    tool: Literal['mke2fs', 'e2fsck', 'debugfs']
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    linkage: Literal['static_elf', 'dynamic_elf']
    dependencies: tuple[GuestDiskToolRuntimeDependency, ...] = Field(max_length=4096)

    @model_validator(mode='after')
    def validate_dependency_shape(self) -> Self:
        keys = tuple((item.role, item.logical_name, item.sha256) for item in self.dependencies)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('tool runtime dependencies must be unique and canonically sorted')
        interpreter_count = sum(item.role == 'elf_interpreter' for item in self.dependencies)
        shared_library_count = sum(item.role == 'shared_library' for item in self.dependencies)
        if self.linkage == 'static_elf' and (interpreter_count or shared_library_count):
            raise ValueError('a static ELF binding cannot declare an ELF interpreter or shared library')
        if self.linkage == 'dynamic_elf' and (interpreter_count != 1 or shared_library_count < 1):
            raise ValueError('a dynamic ELF binding must pin one interpreter and at least one shared library')
        return self


class GuestDiskToolRuntimeClosureManifest(StrictModel):
    """Externally pinned declaration of the ext4 tools' transitive user-space runtime."""

    schema_version: Literal['vaxreplay.lane-a-tool-runtime-closure.v0.1'] = LANE_A_TOOL_RUNTIME_CLOSURE_SCHEMA_VERSION
    platform_system: Literal['Linux'] = 'Linux'
    platform_machine: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9._+-]+$')
    bindings: tuple[GuestDiskToolRuntimeBinding, GuestDiskToolRuntimeBinding, GuestDiskToolRuntimeBinding]
    complete_transitive_user_space_dependency_inventory_attested: Literal[True] = True

    @model_validator(mode='after')
    def validate_binding_order(self) -> Self:
        if tuple(binding.tool for binding in self.bindings) != ('mke2fs', 'e2fsck', 'debugfs'):
            raise ValueError('tool runtime bindings must be complete and in canonical tool order')
        return self


class GuestDiskOutputIdentity(StrictModel):
    role: Literal['rootfs', 'harness']
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=1024 * 1024 * 1024 * 1024)
    uuid: str = Field(min_length=36, max_length=36)
    label: str = Field(min_length=1, max_length=16)
    normalized_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    ext4_health_check_passed: Literal[True] = True
    expected_entry_inspection_passed: Literal[True] = True


class LaneAGuestDiskBuildReceipt(StrictModel):
    """Canonical public evidence for one same-host, twice-staged disk pair."""

    schema_version: Literal['vaxreplay.lane-a-guest-disk-build.dev-v0.3'] = LANE_A_GUEST_DISK_BUILD_SCHEMA_VERSION
    profile: Literal['lane_a_manifest_dispatched_task_guest_v2'] = LANE_A_GUEST_DISK_BUILD_PROFILE
    execution_boundary: GuestDiskExecutionBoundary
    production_eligible: bool
    source_date_epoch: int = Field(ge=1, le=2**31 - 1)
    base_rootfs_source: GuestDiskSourceIdentity
    harness_payload_source: GuestDiskSourceIdentity
    mke2fs: GuestDiskToolIdentity
    e2fsck: GuestDiskToolIdentity
    debugfs: GuestDiskToolIdentity
    tool_runtime_closure_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    tool_runtime_closure_manifest_byte_count: int | None = Field(
        default=None,
        gt=0,
        le=_MAX_TOOL_RUNTIME_CLOSURE_BYTES,
    )
    tool_runtime_closure_external_pin_checked: bool = False
    tool_runtime_closure_bindings_checked: bool = False
    tool_runtime_closure_contains_dynamic_linkage: bool | None = None
    builder_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    build_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    mke2fs_argv_sha256: str = Field(pattern=_SHA256_PATTERN)
    build_environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    inspection_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    inspection_argv_sha256: str = Field(pattern=_SHA256_PATTERN)
    init_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_boot_dispatch: GuestBootDispatchManifest
    guest_boot_dispatch_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_executable_path: str = Field(min_length=2, max_length=4096)
    guest_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_executable_mode: Literal[365] = 0o555
    guest_config_path: str = Field(min_length=2, max_length=4096)
    guest_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_config_mode: Literal[256] = 0o400
    dependency_closure_sha256: str = Field(pattern=_SHA256_PATTERN)
    fixed_guest_argv: tuple[str, ...] = Field(min_length=3, max_length=3)
    fixed_guest_environment: tuple[str, ...] = Field(default=(), max_length=0)
    pid1_clears_ambient_environment: Literal[True] = True
    pid1_uses_busybox_env_argv_dispatch: Literal[True] = True
    submitted_command_string_or_shell_construction_allowed: Literal[False] = False
    ambient_provider_route_or_credentials_present: Literal[False] = False
    canonical_operator_runtime_supported: bool
    linux_kvm_qualification_claimed_by_build_receipt: Literal[False] = False
    rootfs: GuestDiskOutputIdentity
    harness: GuestDiskOutputIdentity
    reproducibility_scope: Literal['same_host_same_process_two_clean_stagings'] = (
        'same_host_same_process_two_clean_stagings'
    )
    same_host_separately_staged_build_count: Literal[2] = 2
    same_host_byte_identical_rebuild_verified: Literal[True] = True
    cross_host_hermetic_reproducibility_claimed: Literal[False] = False
    source_metadata_normalized: Literal[True] = True
    rootfs_harness_separate: Literal[True] = True
    harness_device: Literal['/dev/vdb'] = '/dev/vdb'
    harness_mountpoint: Literal['/opt/vaxreplay'] = '/opt/vaxreplay'
    harness_mounted_read_only: Literal[True] = True
    scratch_device: Literal['/dev/vdc'] = '/dev/vdc'
    scratch_mountpoint: Literal['/var/lib/vaxreplay/scratch'] = '/var/lib/vaxreplay/scratch'
    scratch_mounted_writable: Literal[True] = True
    workspace_brokered_not_baked: Literal[True] = True
    workspace_disk_or_mount_present: Literal[False] = False
    init_execs_fixed_guest_argv: Literal[True] = True
    receipt_contains_no_secret_material: Literal[True] = True

    @model_validator(mode='after')
    def validate_contract(self) -> Self:
        dispatch = self.guest_boot_dispatch
        expected_dispatch_sha256 = guest_boot_dispatch_manifest_sha256(dispatch)
        expected_launch = (
            dispatch.guest_executable_path,
            dispatch.guest_executable_sha256,
            dispatch.guest_config_path,
            dispatch.guest_config_sha256,
            dispatch.guest_argv,
            dispatch.guest_environment,
            dispatch.admission == GuestBootDispatchAdmission.RUNTIME_INTEGRATED_REQUIRES_EXTERNAL_QUALIFICATION,
        )
        observed_launch = (
            self.guest_executable_path,
            self.guest_executable_sha256,
            self.guest_config_path,
            self.guest_config_sha256,
            self.fixed_guest_argv,
            self.fixed_guest_environment,
            self.canonical_operator_runtime_supported,
        )
        if (
            not hmac.compare_digest(
                self.guest_boot_dispatch_manifest_sha256,
                expected_dispatch_sha256,
            )
            or observed_launch != expected_launch
        ):
            raise ValueError('build receipt does not bind the exact guest boot-dispatch manifest')
        if self.rootfs.role != 'rootfs' or self.harness.role != 'harness':
            raise ValueError('disk roles are inverted')
        if self.rootfs.uuid != LANE_A_ROOTFS_UUID or self.rootfs.label != LANE_A_ROOTFS_LABEL:
            raise ValueError('rootfs UUID or label differs from the fixed recipe')
        if self.harness.uuid != LANE_A_HARNESS_UUID or self.harness.label != LANE_A_HARNESS_LABEL:
            raise ValueError('harness UUID or label differs from the fixed recipe')
        production_boundary = self.execution_boundary == GuestDiskExecutionBoundary.PINNED_LINUX_PROCFS
        closure_fields = (
            self.tool_runtime_closure_manifest_sha256,
            self.tool_runtime_closure_manifest_byte_count,
            self.tool_runtime_closure_contains_dynamic_linkage,
        )
        closure_complete = all(value is not None for value in closure_fields)
        if closure_complete != self.tool_runtime_closure_external_pin_checked or (
            closure_complete != self.tool_runtime_closure_bindings_checked
        ):
            raise ValueError('tool runtime closure fields must be present and checked together')
        if production_boundary and not closure_complete:
            raise ValueError('a pinned Linux build requires a separately pinned tool runtime closure')
        measured_dynamic_linkage = any(
            tool.linkage == GuestDiskToolLinkage.DYNAMIC_ELF for tool in (self.mke2fs, self.e2fsck, self.debugfs)
        )
        if closure_complete and self.tool_runtime_closure_contains_dynamic_linkage != measured_dynamic_linkage:
            raise ValueError('tool runtime closure linkage differs from the measured tool executables')
        production_eligible = (
            production_boundary and closure_complete and not bool(self.tool_runtime_closure_contains_dynamic_linkage)
        )
        if self.production_eligible != production_eligible:
            raise ValueError('production eligibility must fail closed for an unpinned or dynamic tool runtime')
        if production_boundary and (
            self.base_rootfs_source.kind != GuestDiskSourceKind.NORMALIZED_TAR
            or self.harness_payload_source.kind != GuestDiskSourceKind.NORMALIZED_TAR
        ):
            raise ValueError('production receipts require normalized tar sources')
        for tool in (self.mke2fs, self.e2fsck, self.debugfs):
            if tool.executed_via_proc_self_fd != production_boundary:
                raise ValueError('tool execution evidence must reflect the execution boundary')
            if production_boundary and tool.linkage == GuestDiskToolLinkage.TEST_UNKNOWN:
                raise ValueError('production tool linkage cannot be unknown')
        return self


@dataclass(frozen=True)
class Ext4BuildRequest:
    tree: Path
    output: Path
    byte_count: int
    uuid: str
    label: str
    source_date_epoch: int


@dataclass(frozen=True)
class ExpectedGuestTreeEntry:
    path: str
    kind: Literal['directory', 'file', 'symlink']
    mode: int
    uid: int
    gid: int
    content_sha256: str


class GuestDiskCommandExecutor(Protocol):
    """Only effectful boundary used by deterministic orchestration."""

    @property
    def boundary(self) -> GuestDiskExecutionBoundary: ...

    @property
    def tool_identities(self) -> tuple[GuestDiskToolIdentity, GuestDiskToolIdentity, GuestDiskToolIdentity]: ...

    def build_ext4(self, request: Ext4BuildRequest) -> None: ...

    def inspect_ext4(self, image: Path, expected_entries: tuple[ExpectedGuestTreeEntry, ...]) -> None: ...


class PinnedLinuxExt4Executor:
    """Root-only Linux executor that never re-resolves an authenticated tool pathname."""

    def __init__(
        self,
        *,
        mke2fs_path: Path,
        expected_mke2fs_sha256: str,
        e2fsck_path: Path,
        expected_e2fsck_sha256: str,
        debugfs_path: Path,
        expected_debugfs_sha256: str,
    ) -> None:
        if platform.system() != 'Linux' or os.geteuid() != 0 or not Path('/proc/self/fd').is_dir():
            raise LaneAGuestDiskBuildError('production disk building requires root on Linux with procfs mounted')
        self._descriptors: dict[str, int] = {}
        try:
            for name, path, expected in (
                ('mke2fs', mke2fs_path, expected_mke2fs_sha256),
                ('e2fsck', e2fsck_path, expected_e2fsck_sha256),
                ('debugfs', debugfs_path, expected_debugfs_sha256),
            ):
                self._descriptors[name] = _open_pinned_root_executable(path, expected, label=name)
            identities = tuple(
                GuestDiskToolIdentity(
                    name=name,  # type: ignore[arg-type]
                    sha256=expected,
                    version=_probe_tool_version(self._descriptors[name]),
                    executed_via_proc_self_fd=True,
                    linkage=_inspect_elf_linkage(self._descriptors[name]),
                )
                for name, expected in (
                    ('mke2fs', expected_mke2fs_sha256),
                    ('e2fsck', expected_e2fsck_sha256),
                    ('debugfs', expected_debugfs_sha256),
                )
            )
            self._tool_identities = (identities[0], identities[1], identities[2])
        except BaseException:
            self.close()
            raise

    @property
    def boundary(self) -> GuestDiskExecutionBoundary:
        return GuestDiskExecutionBoundary.PINNED_LINUX_PROCFS

    @property
    def tool_identities(self) -> tuple[GuestDiskToolIdentity, GuestDiskToolIdentity, GuestDiskToolIdentity]:
        return self._tool_identities

    def close(self) -> None:
        for descriptor in self._descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._descriptors.clear()

    def __enter__(self) -> PinnedLinuxExt4Executor:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def build_ext4(self, request: Ext4BuildRequest) -> None:
        _validate_image_size(request.byte_count, label='ext4 output')
        with request.output.open('xb') as stream:
            stream.truncate(request.byte_count)
            stream.flush()
            os.fsync(stream.fileno())
        arguments = (
            '-q',
            '-t',
            'ext4',
            '-F',
            '-b',
            '4096',
            '-I',
            '256',
            '-i',
            '16384',
            '-m',
            '0',
            '-U',
            request.uuid,
            '-L',
            request.label,
            '-O',
            _MKE2FS_FEATURES,
            '-E',
            f'lazy_itable_init=0,lazy_journal_init=0,root_owner=0:0,hash_seed={request.uuid}',
            '-d',
            str(request.tree),
            str(request.output),
        )
        environment = _build_environment(source_date_epoch=request.source_date_epoch)
        completed = _run_procfd_tool(
            self._descriptors['mke2fs'],
            arguments,
            timeout=600,
            environment=environment,
        )
        if completed.returncode != 0 or len(completed.stdout) + len(completed.stderr) > 1024 * 1024:
            raise LaneAGuestDiskBuildError('pinned mke2fs rejected the fixed Lane A image recipe')

    def inspect_ext4(self, image: Path, expected_entries: tuple[ExpectedGuestTreeEntry, ...]) -> None:
        checked = _run_procfd_tool(self._descriptors['e2fsck'], ('-fn', str(image)), timeout=300)
        if checked.returncode != 0 or len(checked.stdout) + len(checked.stderr) > 1024 * 1024:
            raise LaneAGuestDiskBuildError('pinned e2fsck rejected the completed Lane A image')
        with tempfile.TemporaryDirectory(prefix='vaxreplay-lane-a-inspect.') as temporary:
            root = Path(temporary)
            for index, expected in enumerate(expected_entries):
                _require_safe_guest_path(expected.path)
                observed = _run_procfd_tool(
                    self._descriptors['debugfs'],
                    ('-R', f'stat {expected.path}', str(image)),
                    timeout=120,
                )
                if observed.returncode != 0 or len(observed.stdout) + len(observed.stderr) > 64 * 1024:
                    raise LaneAGuestDiskBuildError('debugfs could not stat an expected Lane A image entry')
                text = observed.stdout.decode('ascii', errors='strict')
                mode_match = re.search(r'Mode:\s+0*([0-7]{3,4})\b', text)
                type_match = re.search(r'Type:\s+([A-Za-z]+)\b', text)
                owner_match = re.search(r'User:\s+(\d+)\s+Group:\s+(\d+)\b', text)
                observed_kind = {
                    'regular': 'file',
                    'directory': 'directory',
                    'symlink': 'symlink',
                }.get(type_match.group(1).lower() if type_match else '')
                if (
                    mode_match is None
                    or int(mode_match.group(1), 8) != expected.mode
                    or observed_kind != expected.kind
                    or owner_match is None
                    or (int(owner_match.group(1)), int(owner_match.group(2))) != (expected.uid, expected.gid)
                ):
                    raise LaneAGuestDiskBuildError(
                        f'debugfs observed an unexpected Lane A image entry type or mode at {expected.path}'
                    )
                if expected.kind == 'directory':
                    continue
                if expected.kind == 'symlink':
                    # ext4 stores short ("fast") symlink targets in the inode.  debugfs reports
                    # those bytes in ``stat`` but ``dump`` creates an empty file; long symlinks are
                    # block-backed and remain inspectable through the ordinary dump path below.
                    fast_symlink_sha256 = _debugfs_fast_symlink_target_sha256(text)
                    if fast_symlink_sha256 is not None:
                        if not hmac.compare_digest(fast_symlink_sha256, expected.content_sha256):
                            raise LaneAGuestDiskBuildError('debugfs observed unexpected Lane A image symlink bytes')
                        continue
                dumped = root / f'entry-{index}'
                extracted = _run_procfd_tool(
                    self._descriptors['debugfs'],
                    ('-R', f'dump {expected.path} {dumped}', str(image)),
                    timeout=120,
                )
                if extracted.returncode != 0 or not dumped.is_file():
                    raise LaneAGuestDiskBuildError('debugfs could not extract an expected Lane A image entry')
                if not hmac.compare_digest(_file_sha256(dumped), expected.content_sha256):
                    raise LaneAGuestDiskBuildError('debugfs observed unexpected Lane A image entry bytes')


def _debugfs_fast_symlink_target_sha256(stat_output: str) -> str | None:
    """Hash a safe fast-symlink target from the already authenticated debugfs stat output."""

    match = re.search(r'^Fast link dest: "([^"\r\n]*)"$', stat_output, flags=re.MULTILINE)
    if match is None:
        return None
    target = match.group(1)
    # Source admission already rejects non-ASCII and quote-bearing targets.  Reapply the same
    # logical-target validation at the image boundary before interpreting debugfs text as bytes.
    _validate_logical_symlink(PurePosixPath('inspected-link'), target, label='debugfs inspection')
    return hashlib.sha256(target.encode('ascii')).hexdigest()


@dataclass(frozen=True)
class BuiltLaneAGuestDisks:
    rootfs_path: Path
    harness_path: Path
    receipt_path: Path
    receipt: LaneAGuestDiskBuildReceipt
    receipt_sha256: str


@dataclass(frozen=True)
class VerifiedLaneAGuestDisks:
    rootfs_path: Path
    harness_path: Path
    receipt: LaneAGuestDiskBuildReceipt
    receipt_sha256: str


def build_lane_a_guest_disks(
    *,
    base_rootfs_source: Path,
    expected_base_rootfs_source_sha256: str,
    harness_payload_source: Path,
    expected_harness_payload_source_sha256: str,
    expected_guest_executable_sha256: str,
    guest_config_bytes: bytes,
    guest_boot_dispatch: GuestBootDispatchManifest | None = None,
    expected_guest_boot_dispatch_manifest_sha256: str | None = None,
    executor: GuestDiskCommandExecutor,
    tool_runtime_closure_manifest: GuestDiskToolRuntimeClosureManifest | None = None,
    expected_tool_runtime_closure_manifest_sha256: str | None = None,
    output_rootfs_path: Path,
    output_harness_path: Path,
    output_receipt_path: Path,
    source_date_epoch: int,
    rootfs_byte_count: int = DEFAULT_LANE_A_ROOTFS_BYTES,
    harness_byte_count: int = DEFAULT_LANE_A_HARNESS_BYTES,
    testing_only: bool = False,
) -> BuiltLaneAGuestDisks:
    """Build twice on one host, inspect expected entries, and publish the Lane A task disks.

    Production calls must use normalized uncompressed tar inputs and
    :class:`PinnedLinuxExt4Executor`, plus an independently pinned canonical tool-runtime closure
    manifest.  Dynamically linked tools are recorded but fail the disk-build production gate until
    execution from that pinned closure is implemented.  ``testing_only`` permits safe directory
    inputs and a fake executor, but the resulting receipt is structurally ineligible for
    production admission.
    """

    _require_sha256(expected_base_rootfs_source_sha256, label='base rootfs source digest')
    _require_sha256(expected_harness_payload_source_sha256, label='harness payload source digest')
    _require_sha256(expected_guest_executable_sha256, label='guest executable digest')
    boundary = executor.boundary
    if boundary == GuestDiskExecutionBoundary.TEST_EXECUTOR and not testing_only:
        raise LaneAGuestDiskBuildError('a test executor requires an explicit testing-only build')
    if boundary == GuestDiskExecutionBoundary.PINNED_LINUX_PROCFS and testing_only:
        raise LaneAGuestDiskBuildError('the production executor cannot emit a testing-only receipt')
    production_boundary = boundary == GuestDiskExecutionBoundary.PINNED_LINUX_PROCFS
    if source_date_epoch < 1 or source_date_epoch > 2**31 - 1:
        raise LaneAGuestDiskBuildError('SOURCE_DATE_EPOCH is out of range')
    _validate_image_size(rootfs_byte_count, label='rootfs', testing_only=testing_only)
    _validate_image_size(harness_byte_count, label='harness', testing_only=testing_only)
    if not guest_config_bytes or len(guest_config_bytes) > 64 * 1024:
        raise LaneAGuestDiskBuildError('guest configuration is empty or oversized')
    guest_config_sha256 = hashlib.sha256(guest_config_bytes).hexdigest()
    dispatch = _canonical_guest_boot_dispatch(
        guest_boot_dispatch,
        expected_guest_executable_sha256=expected_guest_executable_sha256,
        guest_config_sha256=guest_config_sha256,
        expected_manifest_sha256=expected_guest_boot_dispatch_manifest_sha256,
    )
    _require_canonical_guest_config(dispatch, guest_config_bytes)
    dispatch_manifest_sha256 = guest_boot_dispatch_manifest_sha256(dispatch)
    init_bytes = _lane_a_init_bytes(dispatch)
    if b'/workspace' in init_bytes or b'workspace' in init_bytes.lower():
        raise AssertionError('the fixed init must never create or mount a workspace')

    tool_identities = _validate_tool_identities(
        executor.tool_identities,
        production=production_boundary,
    )
    tool_runtime_closure_identity = _validate_tool_runtime_closure_manifest(
        tool_runtime_closure_manifest,
        expected_sha256=expected_tool_runtime_closure_manifest_sha256,
        tool_identities=tool_identities,
        required=production_boundary,
    )
    production_eligible = (
        production_boundary and tool_runtime_closure_identity is not None and not (tool_runtime_closure_identity[2])
    )

    outputs = tuple(
        path.expanduser().absolute() for path in (output_rootfs_path, output_harness_path, output_receipt_path)
    )
    if len(set(outputs)) != len(outputs):
        raise LaneAGuestDiskBuildError('rootfs, harness, and receipt outputs must be distinct')
    for output in outputs:
        if output.is_symlink() or output.exists():
            raise LaneAGuestDiskBuildError('Lane A guest disk outputs are create-once')
        _validate_output_parent(output, production=production_boundary)

    with tempfile.TemporaryDirectory(prefix='vaxreplay-lane-a-build.') as temporary:
        temporary_root = Path(temporary)
        base_source_tree, base_identity = _materialize_source(
            base_rootfs_source,
            expected_sha256=expected_base_rootfs_source_sha256,
            destination=temporary_root / 'base-source',
            source_date_epoch=source_date_epoch,
            testing_only=testing_only,
            label='base rootfs',
        )
        harness_source_tree, harness_identity = _materialize_source(
            harness_payload_source,
            expected_sha256=expected_harness_payload_source_sha256,
            destination=temporary_root / 'harness-source',
            source_date_epoch=source_date_epoch,
            testing_only=testing_only,
            label='harness payload',
        )
        _validate_base_rootfs_contract(base_source_tree)
        dependency_closure_sha256 = _validate_harness_payload(
            harness_source_tree,
            dispatch=dispatch,
        )

        rootfs_trees = (temporary_root / 'rootfs-tree-a', temporary_root / 'rootfs-tree-b')
        harness_trees = (temporary_root / 'harness-tree-a', temporary_root / 'harness-tree-b')
        for tree in rootfs_trees:
            _prepare_rootfs_tree(
                base_source_tree,
                tree,
                init_bytes=init_bytes,
                source_date_epoch=source_date_epoch,
                production=production_boundary,
            )
        for tree in harness_trees:
            _prepare_harness_tree(
                harness_source_tree,
                tree,
                dispatch=dispatch,
                guest_config_bytes=guest_config_bytes,
                source_date_epoch=source_date_epoch,
                production=production_boundary,
            )
        rootfs_tree_hashes = tuple(_tree_sha256(tree) for tree in rootfs_trees)
        harness_tree_hashes = tuple(_tree_sha256(tree) for tree in harness_trees)
        if len(set(rootfs_tree_hashes)) != 1 or len(set(harness_tree_hashes)) != 1:
            raise LaneAGuestDiskBuildError('independently staged source trees are not identical')

        rootfs_images = (temporary_root / 'rootfs-a.ext4', temporary_root / 'rootfs-b.ext4')
        harness_images = (temporary_root / 'harness-a.ext4', temporary_root / 'harness-b.ext4')
        for image, tree in zip(rootfs_images, rootfs_trees, strict=True):
            executor.build_ext4(
                Ext4BuildRequest(
                    tree=tree,
                    output=image,
                    byte_count=rootfs_byte_count,
                    uuid=LANE_A_ROOTFS_UUID,
                    label=LANE_A_ROOTFS_LABEL,
                    source_date_epoch=source_date_epoch,
                )
            )
        for image, tree in zip(harness_images, harness_trees, strict=True):
            executor.build_ext4(
                Ext4BuildRequest(
                    tree=tree,
                    output=image,
                    byte_count=harness_byte_count,
                    uuid=LANE_A_HARNESS_UUID,
                    label=LANE_A_HARNESS_LABEL,
                    source_date_epoch=source_date_epoch,
                )
            )
        _require_identical_files(rootfs_images[0], rootfs_images[1], label='Lane A rootfs')
        _require_identical_files(harness_images[0], harness_images[1], label='Lane A harness')

        rootfs_entries = _expected_tree_entries(rootfs_trees[0])
        harness_entries = _expected_tree_entries(harness_trees[0])
        if any(entry.path == '/workspace' or entry.path.startswith('/workspace/') for entry in rootfs_entries):
            raise LaneAGuestDiskBuildError('a workspace path survived in the rootfs')
        if any(entry.path == '/workspace' or entry.path.startswith('/workspace/') for entry in harness_entries):
            raise LaneAGuestDiskBuildError('a workspace path survived in the harness')
        for image in rootfs_images:
            executor.inspect_ext4(image, rootfs_entries)
        for image in harness_images:
            executor.inspect_ext4(image, harness_entries)

        builder_source_sha256 = _file_sha256(Path(__file__))
        build_contract_sha256 = hashlib.sha256(
            _build_contract_bytes(source_date_epoch=source_date_epoch, dispatch=dispatch)
        ).hexdigest()
        inspection_contract_sha256 = hashlib.sha256(_inspection_contract_bytes()).hexdigest()
        rootfs_sha256 = _file_sha256(rootfs_images[0])
        harness_sha256 = _file_sha256(harness_images[0])
        receipt = LaneAGuestDiskBuildReceipt(
            execution_boundary=boundary,
            production_eligible=production_eligible,
            source_date_epoch=source_date_epoch,
            base_rootfs_source=base_identity,
            harness_payload_source=harness_identity,
            mke2fs=tool_identities[0],
            e2fsck=tool_identities[1],
            debugfs=tool_identities[2],
            tool_runtime_closure_manifest_sha256=(
                tool_runtime_closure_identity[0] if tool_runtime_closure_identity is not None else None
            ),
            tool_runtime_closure_manifest_byte_count=(
                tool_runtime_closure_identity[1] if tool_runtime_closure_identity is not None else None
            ),
            tool_runtime_closure_external_pin_checked=tool_runtime_closure_identity is not None,
            tool_runtime_closure_bindings_checked=tool_runtime_closure_identity is not None,
            tool_runtime_closure_contains_dynamic_linkage=(
                tool_runtime_closure_identity[2] if tool_runtime_closure_identity is not None else None
            ),
            builder_source_sha256=builder_source_sha256,
            build_contract_sha256=build_contract_sha256,
            mke2fs_argv_sha256=hashlib.sha256(canonical_json_bytes(_MKE2FS_ARGV_TEMPLATE)).hexdigest(),
            build_environment_sha256=hashlib.sha256(
                _build_environment_bytes(source_date_epoch=source_date_epoch)
            ).hexdigest(),
            inspection_contract_sha256=inspection_contract_sha256,
            inspection_argv_sha256=hashlib.sha256(_inspection_argv_bytes()).hexdigest(),
            init_sha256=hashlib.sha256(init_bytes).hexdigest(),
            guest_boot_dispatch=dispatch,
            guest_boot_dispatch_manifest_sha256=dispatch_manifest_sha256,
            guest_executable_path=dispatch.guest_executable_path,
            guest_executable_sha256=expected_guest_executable_sha256,
            guest_config_path=dispatch.guest_config_path,
            guest_config_sha256=guest_config_sha256,
            dependency_closure_sha256=dependency_closure_sha256,
            fixed_guest_argv=dispatch.guest_argv,
            fixed_guest_environment=dispatch.guest_environment,
            canonical_operator_runtime_supported=(
                dispatch.admission == GuestBootDispatchAdmission.RUNTIME_INTEGRATED_REQUIRES_EXTERNAL_QUALIFICATION
            ),
            rootfs=GuestDiskOutputIdentity(
                role='rootfs',
                sha256=rootfs_sha256,
                byte_count=rootfs_byte_count,
                uuid=LANE_A_ROOTFS_UUID,
                label=LANE_A_ROOTFS_LABEL,
                normalized_tree_sha256=rootfs_tree_hashes[0],
            ),
            harness=GuestDiskOutputIdentity(
                role='harness',
                sha256=harness_sha256,
                byte_count=harness_byte_count,
                uuid=LANE_A_HARNESS_UUID,
                label=LANE_A_HARNESS_LABEL,
                normalized_tree_sha256=harness_tree_hashes[0],
            ),
        )
        receipt_bytes = canonical_json_bytes(receipt)
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        _publish_create_once_streaming(outputs[0], rootfs_images[0], mode=0o600)
        _publish_create_once_streaming(outputs[1], harness_images[0], mode=0o600)
        _publish_create_once_bytes(outputs[2], receipt_bytes, mode=0o644)
        if not _regular_file_matches(outputs[0], rootfs_sha256, rootfs_byte_count):
            raise LaneAGuestDiskBuildError('published rootfs differs from the verified temporary image')
        if not _regular_file_matches(outputs[1], harness_sha256, harness_byte_count):
            raise LaneAGuestDiskBuildError('published harness differs from the verified temporary image')
        published_receipt = _read_regular_file_no_follow(outputs[2], maximum_bytes=_MAX_RECEIPT_BYTES)
        if not hmac.compare_digest(published_receipt, receipt_bytes):
            raise LaneAGuestDiskBuildError('published receipt differs from the canonical verified bytes')

    return BuiltLaneAGuestDisks(
        rootfs_path=outputs[0],
        harness_path=outputs[1],
        receipt_path=outputs[2],
        receipt=receipt,
        receipt_sha256=receipt_sha256,
    )


def lane_a_guest_disk_build_receipt_sha256(receipt: LaneAGuestDiskBuildReceipt) -> str:
    canonical = LaneAGuestDiskBuildReceipt.model_validate_json(canonical_json_bytes(receipt))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def compute_testing_guest_disk_source_directory_sha256(path: Path) -> str:
    """Compute the external tree pin accepted for an explicit testing-only directory source."""

    absolute = path.expanduser().absolute()
    _validate_tree(absolute, label='testing guest disk source')
    return _tree_sha256(absolute)


def load_pinned_lane_a_guest_config_bytes(path: Path, *, expected_sha256: str) -> bytes:
    """No-follow load and schema-check the exact canonical configuration supplied to the builder."""

    _require_sha256(expected_sha256, label='guest configuration digest')
    payload = _read_regular_file_no_follow(path, maximum_bytes=64 * 1024)
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_sha256):
        raise LaneAGuestDiskBuildError('guest configuration differs from its external digest pin')
    _require_canonical_guest_config(
        make_native_guest_boot_dispatch_manifest(
            guest_executable_sha256='0' * 64,
            guest_config_sha256=expected_sha256,
        ),
        payload,
    )
    return payload


def load_pinned_guest_disk_tool_runtime_closure_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> GuestDiskToolRuntimeClosureManifest:
    """Load exact canonical tool-runtime closure bytes under an independent digest pin."""

    _require_sha256(expected_sha256, label='tool runtime closure manifest digest')
    payload = _read_regular_file_no_follow(path, maximum_bytes=_MAX_TOOL_RUNTIME_CLOSURE_BYTES)
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_sha256):
        raise LaneAGuestDiskBuildError('tool runtime closure manifest differs from its external digest pin')
    try:
        manifest = GuestDiskToolRuntimeClosureManifest.model_validate_json(payload)
    except (TypeError, ValueError) as error:
        raise LaneAGuestDiskBuildError('tool runtime closure manifest schema is invalid') from error
    if not hmac.compare_digest(payload, canonical_json_bytes(manifest)):
        raise LaneAGuestDiskBuildError('tool runtime closure manifest is not exact canonical JSON')
    return manifest


def load_lane_a_guest_disk_build_receipt(
    path: Path,
    *,
    expected_receipt_sha256: str,
) -> LaneAGuestDiskBuildReceipt:
    """Load exact no-follow canonical receipt bytes under an external digest pin."""

    _require_sha256(expected_receipt_sha256, label='build receipt digest')
    payload = _read_regular_file_no_follow(path, maximum_bytes=_MAX_RECEIPT_BYTES)
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_receipt_sha256):
        raise LaneAGuestDiskBuildError('build receipt differs from its external digest pin')
    try:
        receipt = LaneAGuestDiskBuildReceipt.model_validate_json(payload)
    except (TypeError, ValueError) as error:
        raise LaneAGuestDiskBuildError('build receipt schema is invalid') from error
    if not hmac.compare_digest(payload, canonical_json_bytes(receipt)):
        raise LaneAGuestDiskBuildError('build receipt is not exact canonical JSON')
    return receipt


def verify_lane_a_guest_disk_build(
    *,
    receipt: LaneAGuestDiskBuildReceipt,
    rootfs_path: Path,
    harness_path: Path,
    expected_base_rootfs_source_sha256: str,
    expected_harness_payload_source_sha256: str,
    expected_guest_executable_sha256: str,
    expected_guest_config_sha256: str,
    expected_guest_boot_dispatch: GuestBootDispatchManifest | None = None,
    expected_guest_boot_dispatch_manifest_sha256: str | None = None,
    expected_mke2fs_sha256: str,
    expected_e2fsck_sha256: str,
    expected_debugfs_sha256: str,
    expected_tool_runtime_closure_manifest_sha256: str | None = None,
    expected_builder_source_sha256: str | None = None,
    require_production: bool = True,
) -> VerifiedLaneAGuestDisks:
    """Offline verification of receipt semantics, external pins, and exact output bytes."""

    canonical = LaneAGuestDiskBuildReceipt.model_validate_json(canonical_json_bytes(receipt))
    expected_pins = (
        expected_base_rootfs_source_sha256,
        expected_harness_payload_source_sha256,
        expected_guest_executable_sha256,
        expected_guest_config_sha256,
        expected_mke2fs_sha256,
        expected_e2fsck_sha256,
        expected_debugfs_sha256,
    )
    for index, pin in enumerate(expected_pins):
        _require_sha256(pin, label=f'external verification pin {index}')
    if require_production and not canonical.production_eligible:
        raise LaneAGuestDiskBuildError('a testing-only build receipt cannot authorize a production launch')
    observed_pins = (
        canonical.base_rootfs_source.sha256,
        canonical.harness_payload_source.sha256,
        canonical.guest_executable_sha256,
        canonical.guest_config_sha256,
        canonical.mke2fs.sha256,
        canonical.e2fsck.sha256,
        canonical.debugfs.sha256,
    )
    if not all(hmac.compare_digest(expected, observed) for expected, observed in zip(expected_pins, observed_pins)):
        raise LaneAGuestDiskBuildError('build receipt differs from an external source, guest, or tool pin')
    if expected_guest_boot_dispatch is not None:
        expected_dispatch = GuestBootDispatchManifest.model_validate_json(
            canonical_json_bytes(expected_guest_boot_dispatch)
        )
        if canonical.guest_boot_dispatch != expected_dispatch:
            raise LaneAGuestDiskBuildError(
                'build receipt differs from the externally selected guest boot-dispatch manifest'
            )
    if expected_guest_boot_dispatch_manifest_sha256 is not None:
        _require_sha256(
            expected_guest_boot_dispatch_manifest_sha256,
            label='external guest boot-dispatch manifest pin',
        )
        if not hmac.compare_digest(
            canonical.guest_boot_dispatch_manifest_sha256,
            expected_guest_boot_dispatch_manifest_sha256,
        ):
            raise LaneAGuestDiskBuildError('build receipt differs from the external guest boot-dispatch manifest pin')
    if expected_tool_runtime_closure_manifest_sha256 is not None:
        _require_sha256(
            expected_tool_runtime_closure_manifest_sha256,
            label='external tool runtime closure manifest pin',
        )
        if canonical.tool_runtime_closure_manifest_sha256 is None or not hmac.compare_digest(
            canonical.tool_runtime_closure_manifest_sha256,
            expected_tool_runtime_closure_manifest_sha256,
        ):
            raise LaneAGuestDiskBuildError('build receipt differs from the external tool runtime closure manifest pin')
    builder_source_pin = expected_builder_source_sha256 or _file_sha256(Path(__file__))
    _require_sha256(builder_source_pin, label='builder source digest')
    if not hmac.compare_digest(canonical.builder_source_sha256, builder_source_pin):
        raise LaneAGuestDiskBuildError('build receipt differs from the externally pinned builder source')
    expected_init_sha256 = hashlib.sha256(_lane_a_init_bytes(canonical.guest_boot_dispatch)).hexdigest()
    expected_build_contract_sha256 = hashlib.sha256(
        _build_contract_bytes(
            source_date_epoch=canonical.source_date_epoch,
            dispatch=canonical.guest_boot_dispatch,
        )
    ).hexdigest()
    expected_inspection_contract_sha256 = hashlib.sha256(_inspection_contract_bytes()).hexdigest()
    expected_mke2fs_argv_sha256 = hashlib.sha256(canonical_json_bytes(_MKE2FS_ARGV_TEMPLATE)).hexdigest()
    expected_build_environment_sha256 = hashlib.sha256(
        _build_environment_bytes(source_date_epoch=canonical.source_date_epoch)
    ).hexdigest()
    expected_inspection_argv_sha256 = hashlib.sha256(_inspection_argv_bytes()).hexdigest()
    if (
        not hmac.compare_digest(canonical.init_sha256, expected_init_sha256)
        or not hmac.compare_digest(canonical.build_contract_sha256, expected_build_contract_sha256)
        or not hmac.compare_digest(canonical.mke2fs_argv_sha256, expected_mke2fs_argv_sha256)
        or not hmac.compare_digest(canonical.build_environment_sha256, expected_build_environment_sha256)
        or not hmac.compare_digest(canonical.inspection_contract_sha256, expected_inspection_contract_sha256)
        or not hmac.compare_digest(canonical.inspection_argv_sha256, expected_inspection_argv_sha256)
    ):
        raise LaneAGuestDiskBuildError('build receipt does not bind the current fixed init/build/inspection contract')
    rootfs = rootfs_path.expanduser().absolute()
    harness = harness_path.expanduser().absolute()
    if not _regular_file_matches(rootfs, canonical.rootfs.sha256, canonical.rootfs.byte_count):
        raise LaneAGuestDiskBuildError('rootfs bytes differ from the authenticated build receipt')
    if not _regular_file_matches(harness, canonical.harness.sha256, canonical.harness.byte_count):
        raise LaneAGuestDiskBuildError('harness bytes differ from the authenticated build receipt')
    return VerifiedLaneAGuestDisks(
        rootfs_path=rootfs,
        harness_path=harness,
        receipt=canonical,
        receipt_sha256=lane_a_guest_disk_build_receipt_sha256(canonical),
    )


def verify_lane_a_guest_disk_build_parity(
    first: LaneAGuestDiskBuildReceipt,
    second: LaneAGuestDiskBuildReceipt,
) -> LaneAGuestDiskBuildReceipt:
    """Require two builder receipts to be byte-for-byte canonical equivalents."""

    first_canonical = LaneAGuestDiskBuildReceipt.model_validate_json(canonical_json_bytes(first))
    second_canonical = LaneAGuestDiskBuildReceipt.model_validate_json(canonical_json_bytes(second))
    if not hmac.compare_digest(canonical_json_bytes(first_canonical), canonical_json_bytes(second_canonical)):
        raise LaneAGuestDiskBuildError('independent Lane A build receipts do not have exact parity')
    return first_canonical


def _lane_a_init_bytes(dispatch: GuestBootDispatchManifest | str) -> bytes:
    """Render fixed PID-1 bytes without accepting a command string or inherited environment.

    A digest string retains the native-only construction API used by the local artifact builder.
    New callers should pass the complete manifest.  Every accepted argv element is shell-inert by
    schema, and BusyBox ``env -i`` directly execs the selected executable with an empty environment.
    """

    if isinstance(dispatch, str):
        _require_sha256(dispatch, label='guest configuration digest')
        canonical = make_native_guest_boot_dispatch_manifest(
            guest_executable_sha256='0' * 64,
            guest_config_sha256=dispatch,
        )
    else:
        canonical = GuestBootDispatchManifest.model_validate_json(canonical_json_bytes(dispatch))
    argv = ' '.join(canonical.guest_argv)
    return (
        '#!/bin/sh\n'
        'set -eu\n'
        'if ! /bin/mount -t proc -o nosuid,nodev,noexec proc /proc; then [ -r /proc/1/stat ]; fi\n'
        'if ! /bin/mount -t sysfs -o nosuid,nodev,noexec sysfs /sys; then [ -d /sys/kernel ]; fi\n'
        'if ! /bin/mount -t devtmpfs -o nosuid devtmpfs /dev; then [ -c /dev/null ]; fi\n'
        '/bin/mkdir -p /opt/vaxreplay /var/lib/vaxreplay/scratch\n'
        '/bin/mount -t ext4 -o ro,nodev,nosuid /dev/vdb /opt/vaxreplay\n'
        '/bin/mount -t ext4 -o rw,nodev,nosuid,noexec /dev/vdc /var/lib/vaxreplay/scratch\n'
        f'exec /bin/busybox env -i -- {argv}\n'
    ).encode('ascii')


def _canonical_guest_boot_dispatch(
    dispatch: GuestBootDispatchManifest | None,
    *,
    expected_guest_executable_sha256: str,
    guest_config_sha256: str,
    expected_manifest_sha256: str | None,
) -> GuestBootDispatchManifest:
    canonical = (
        make_native_guest_boot_dispatch_manifest(
            guest_executable_sha256=expected_guest_executable_sha256,
            guest_config_sha256=guest_config_sha256,
        )
        if dispatch is None
        else GuestBootDispatchManifest.model_validate_json(canonical_json_bytes(dispatch))
    )
    if not hmac.compare_digest(
        canonical.guest_executable_sha256,
        expected_guest_executable_sha256,
    ) or not hmac.compare_digest(canonical.guest_config_sha256, guest_config_sha256):
        raise LaneAGuestDiskBuildError('guest boot-dispatch manifest differs from the executable or configuration pin')
    if expected_manifest_sha256 is not None:
        _require_sha256(expected_manifest_sha256, label='guest boot-dispatch manifest digest')
        if not hmac.compare_digest(
            guest_boot_dispatch_manifest_sha256(canonical),
            expected_manifest_sha256,
        ):
            raise LaneAGuestDiskBuildError('guest boot-dispatch manifest differs from its external digest pin')
    return canonical


def _require_canonical_guest_config(
    dispatch: GuestBootDispatchManifest,
    payload: bytes,
) -> None:
    try:
        validate_guest_boot_config_bytes(dispatch, payload)
    except GuestBootDispatchError as error:
        raise LaneAGuestDiskBuildError('guest configuration does not satisfy its dispatch schema') from error


def _validate_tool_identities(
    identities: tuple[GuestDiskToolIdentity, GuestDiskToolIdentity, GuestDiskToolIdentity],
    *,
    production: bool,
) -> tuple[GuestDiskToolIdentity, GuestDiskToolIdentity, GuestDiskToolIdentity]:
    if tuple(identity.name for identity in identities) != ('mke2fs', 'e2fsck', 'debugfs'):
        raise LaneAGuestDiskBuildError('executor tool identities are missing, duplicated, or out of order')
    if any(identity.executed_via_proc_self_fd != production for identity in identities):
        raise LaneAGuestDiskBuildError('executor tool identities misrepresent their process boundary')
    if production and any(identity.linkage == GuestDiskToolLinkage.TEST_UNKNOWN for identity in identities):
        raise LaneAGuestDiskBuildError('production ext4 tools must have measured ELF linkage')
    if not production and any(identity.linkage != GuestDiskToolLinkage.TEST_UNKNOWN for identity in identities):
        raise LaneAGuestDiskBuildError('test executor tool linkage must remain explicitly unknown')
    return identities


def _validate_tool_runtime_closure_manifest(
    manifest: GuestDiskToolRuntimeClosureManifest | None,
    *,
    expected_sha256: str | None,
    tool_identities: tuple[GuestDiskToolIdentity, GuestDiskToolIdentity, GuestDiskToolIdentity],
    required: bool,
) -> tuple[str, int, bool] | None:
    if manifest is None and expected_sha256 is None:
        if required:
            raise LaneAGuestDiskBuildError(
                'a pinned Linux build requires an independently pinned tool runtime closure manifest'
            )
        return None
    if manifest is None or expected_sha256 is None:
        raise LaneAGuestDiskBuildError('tool runtime closure manifest and external digest pin are inseparable')
    _require_sha256(expected_sha256, label='tool runtime closure manifest digest')
    payload = canonical_json_bytes(manifest)
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_sha256):
        raise LaneAGuestDiskBuildError('tool runtime closure manifest differs from its external digest pin')
    if manifest.platform_machine != platform.machine():
        raise LaneAGuestDiskBuildError('tool runtime closure manifest targets a different machine architecture')
    observed = tuple((identity.name, identity.sha256, identity.linkage.value) for identity in tool_identities)
    committed = tuple((binding.tool, binding.executable_sha256, binding.linkage) for binding in manifest.bindings)
    if observed != committed:
        raise LaneAGuestDiskBuildError('tool runtime closure bindings differ from the measured ext4 tools')
    contains_dynamic = any(binding.linkage == 'dynamic_elf' for binding in manifest.bindings)
    return expected_sha256, len(payload), contains_dynamic


def _materialize_source(
    source: Path,
    *,
    expected_sha256: str,
    destination: Path,
    source_date_epoch: int,
    testing_only: bool,
    label: str,
) -> tuple[Path, GuestDiskSourceIdentity]:
    absolute = source.expanduser().absolute()
    try:
        metadata = absolute.lstat()
    except OSError as error:
        raise LaneAGuestDiskBuildError(f'{label} source is unavailable') from error
    if stat.S_ISDIR(metadata.st_mode):
        if not testing_only or absolute.is_symlink():
            raise LaneAGuestDiskBuildError(f'{label} directories are permitted only in explicit tests')
        source_root = absolute.resolve(strict=True)
        _validate_tree(source_root, label=label)
        observed_sha256 = _tree_sha256(source_root)
        if not hmac.compare_digest(observed_sha256, expected_sha256):
            raise LaneAGuestDiskBuildError(f'{label} directory differs from its external tree pin')
        _copy_tree_deterministic(source_root, destination)
        if not hmac.compare_digest(_tree_sha256(source_root), expected_sha256):
            raise LaneAGuestDiskBuildError(f'{label} directory changed while it was copied')
        _normalize_tree(destination, source_date_epoch=source_date_epoch, production=False, harness=False)
        normalized_sha256 = _tree_sha256(destination)
        return destination, GuestDiskSourceIdentity(
            kind=GuestDiskSourceKind.TEST_DIRECTORY,
            sha256=observed_sha256,
            byte_count=_tree_payload_byte_count(source_root),
            normalized_tree_sha256=normalized_sha256,
        )
    if not stat.S_ISREG(metadata.st_mode) or absolute.is_symlink():
        raise LaneAGuestDiskBuildError(f'{label} must be an uncompressed normalized tar archive')
    byte_count, observed_sha256 = _extract_normalized_tar(
        absolute,
        destination,
        expected_sha256=expected_sha256,
        source_date_epoch=source_date_epoch,
        production=not testing_only,
        label=label,
    )
    normalized_sha256 = _tree_sha256(destination)
    return destination, GuestDiskSourceIdentity(
        kind=GuestDiskSourceKind.NORMALIZED_TAR,
        sha256=observed_sha256,
        byte_count=byte_count,
        normalized_tree_sha256=normalized_sha256,
    )


def _extract_normalized_tar(
    archive_path: Path,
    destination: Path,
    *,
    expected_sha256: str,
    source_date_epoch: int,
    production: bool,
    label: str,
) -> tuple[int, str]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(archive_path, flags)
        before = os.fstat(descriptor)
    except OSError as error:
        raise LaneAGuestDiskBuildError(f'{label} archive cannot be opened safely') from error
    try:
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > _MAX_SOURCE_BYTES:
            raise LaneAGuestDiskBuildError(f'{label} archive size is invalid')
        observed_sha256 = _descriptor_sha256(descriptor)
        if not hmac.compare_digest(observed_sha256, expected_sha256):
            raise LaneAGuestDiskBuildError(f'{label} archive differs from its external digest pin')
        destination.mkdir(mode=0o755)
        names: set[str] = set()
        previous_name = ''
        total_payload_bytes = 0
        member_count = 0
        with os.fdopen(os.dup(descriptor), 'rb') as stream:
            stream.seek(0)
            try:
                archive = tarfile.open(fileobj=stream, mode='r:')
            except tarfile.TarError as error:
                raise LaneAGuestDiskBuildError(f'{label} is not an uncompressed tar archive') from error
            with archive:
                try:
                    for member in archive:
                        member_count += 1
                        if member_count > _MAX_ARCHIVE_MEMBERS:
                            raise LaneAGuestDiskBuildError(f'{label} archive has too many entries')
                        relative = _normalized_archive_member_path(member.name, label=label)
                        if relative is None:
                            continue
                        name = relative.as_posix()
                        if name in names or (previous_name and name <= previous_name):
                            raise LaneAGuestDiskBuildError(f'{label} archive entries must be unique and sorted')
                        names.add(name)
                        previous_name = name
                        if (
                            member.uid != 0
                            or member.gid != 0
                            or member.mtime != source_date_epoch
                            or member.pax_headers
                            or member.mode < 0
                            or member.mode > 0o7777
                        ):
                            raise LaneAGuestDiskBuildError(f'{label} tar metadata is not normalized')
                        if member.isfile():
                            total_payload_bytes += member.size
                        if member.size < 0 or total_payload_bytes > _MAX_SOURCE_BYTES:
                            raise LaneAGuestDiskBuildError(f'{label} tar payload is oversized')
                        target = _ensure_real_parent_path(destination, relative)
                        if member.isdir():
                            if target.exists():
                                if target.is_symlink() or not target.is_dir():
                                    raise LaneAGuestDiskBuildError(
                                        f'{label} archive path collides with a non-directory'
                                    )
                            else:
                                target.mkdir(mode=member.mode or 0o755)
                            target.chmod(member.mode or 0o755)
                        elif member.isfile():
                            extracted_stream = archive.extractfile(member)
                            if extracted_stream is None:
                                raise LaneAGuestDiskBuildError(f'{label} archive file body is unavailable')
                            with extracted_stream, target.open('xb') as output:
                                shutil.copyfileobj(extracted_stream, output, length=1024 * 1024)
                            if target.stat().st_size != member.size:
                                raise LaneAGuestDiskBuildError(f'{label} archive file size is inconsistent')
                            target.chmod(member.mode or 0o444)
                        elif member.issym():
                            _validate_logical_symlink(relative, member.linkname, label=label)
                            target.symlink_to(member.linkname)
                        else:
                            raise LaneAGuestDiskBuildError(
                                f'{label} archive may contain only directories, files, and symbolic links'
                            )
                except (OSError, tarfile.TarError) as error:
                    if isinstance(error, LaneAGuestDiskBuildError):
                        raise
                    raise LaneAGuestDiskBuildError(f'{label} archive extraction failed') from error
        if not names:
            raise LaneAGuestDiskBuildError(f'{label} archive is empty')
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise LaneAGuestDiskBuildError(f'{label} archive changed while it was read')
    finally:
        os.close(descriptor)
    _validate_tree(destination, label=label)
    _normalize_tree(
        destination,
        source_date_epoch=source_date_epoch,
        production=production,
        harness=False,
    )
    return before.st_size, observed_sha256


def _normalized_archive_member_path(name: str, *, label: str) -> PurePosixPath | None:
    if name in {'.', './'}:
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or not path.parts or '..' in path.parts:
        raise LaneAGuestDiskBuildError(f'{label} archive contains a non-normalized path')
    for component in path.parts:
        if component in {'', '.', '..'} or _SAFE_PATH_COMPONENT.fullmatch(component) is None:
            raise LaneAGuestDiskBuildError(f'{label} archive contains an unsafe path component')
    return path


def _ensure_real_parent_path(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for component in relative.parent.parts:
        current /= component
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise LaneAGuestDiskBuildError('archive entry has a symbolic or non-directory parent')
        else:
            current.mkdir(mode=0o755)
    return root.joinpath(*relative.parts)


def _validate_base_rootfs_contract(root: Path) -> None:
    for relative in ('bin/busybox', 'bin/sh', 'bin/mount', 'bin/mkdir'):
        path = _resolve_chroot_path(root, PurePosixPath(relative), label='base rootfs')
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or not stat.S_IMODE(metadata.st_mode) & 0o111:
            raise LaneAGuestDiskBuildError(f'base rootfs required path is not executable: {relative}')


def _harness_relative_path(path_text: str) -> PurePosixPath:
    try:
        relative = PurePosixPath(path_text).relative_to('/opt/vaxreplay')
    except ValueError as error:
        raise LaneAGuestDiskBuildError('guest dispatch path is outside the read-only harness mount') from error
    if not relative.parts or relative.as_posix() in {'', '.'} or '..' in relative.parts:
        raise LaneAGuestDiskBuildError('guest dispatch path does not name a harness artifact')
    return relative


def _require_real_harness_path(
    root: Path,
    relative: PurePosixPath,
    *,
    allow_missing_leaf: bool,
    label: str,
) -> Path:
    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        is_leaf = index == len(relative.parts) - 1
        if not current.exists() and not current.is_symlink():
            if allow_missing_leaf:
                return root.joinpath(*relative.parts)
            raise LaneAGuestDiskBuildError(f'harness payload lacks the manifest-declared {label}')
        if current.is_symlink():
            raise LaneAGuestDiskBuildError(f'manifest-declared {label} path cannot contain a symbolic link')
        if not is_leaf and not current.is_dir():
            raise LaneAGuestDiskBuildError(f'manifest-declared {label} path has a non-directory parent')
    return current


def _validate_harness_payload(
    root: Path,
    *,
    dispatch: GuestBootDispatchManifest,
) -> str:
    executable_relative = _harness_relative_path(dispatch.guest_executable_path)
    config_relative = _harness_relative_path(dispatch.guest_config_path)
    executable = _require_real_harness_path(
        root,
        executable_relative,
        allow_missing_leaf=False,
        label='guest executable',
    )
    try:
        metadata = executable.lstat()
    except OSError as error:
        raise LaneAGuestDiskBuildError('harness payload lacks the manifest-declared guest executable') from error
    if not stat.S_ISREG(metadata.st_mode) or not hmac.compare_digest(
        _file_sha256(executable), dispatch.guest_executable_sha256
    ):
        raise LaneAGuestDiskBuildError('harness payload guest executable differs from its external pin')
    config = _require_real_harness_path(
        root,
        config_relative,
        allow_missing_leaf=True,
        label='guest configuration',
    )
    if config.exists() or config.is_symlink():
        raise LaneAGuestDiskBuildError('harness payload must leave the builder-owned config path empty')
    for forbidden in ('workspace', 'opt', 'private', 'gold'):
        path = root / forbidden
        if path.is_symlink() or path.exists():
            raise LaneAGuestDiskBuildError(f'harness payload contains forbidden top-level path: {forbidden}')
    return _tree_sha256(root, excluded_paths={executable_relative.as_posix()})


def _prepare_rootfs_tree(
    source: Path,
    destination: Path,
    *,
    init_bytes: bytes,
    source_date_epoch: int,
    production: bool,
) -> None:
    _copy_tree_deterministic(source, destination)
    for relative in ('workspace', 'opt/vaxreplay', 'var/lib/vaxreplay/scratch'):
        target = _guest_overlay_target(destination, PurePosixPath(relative))
        _remove_path(target)
        if relative != 'workspace':
            target.mkdir(parents=True, mode=0o755)
    for relative in ('proc', 'sys', 'dev'):
        _guest_overlay_target(destination, PurePosixPath(relative)).mkdir(
            parents=True,
            exist_ok=True,
            mode=0o755,
        )
    init = _guest_overlay_target(destination, PurePosixPath('sbin/init'))
    _remove_path(init)
    init.parent.mkdir(parents=True, exist_ok=True)
    init.write_bytes(init_bytes)
    init.chmod(0o755)
    _normalize_tree(
        destination,
        source_date_epoch=source_date_epoch,
        production=production,
        harness=False,
    )
    if _file_sha256(init) != hashlib.sha256(init_bytes).hexdigest() or stat.S_IMODE(init.stat().st_mode) != 0o755:
        raise LaneAGuestDiskBuildError('fixed PID-1 init overlay did not survive normalization')


def _prepare_harness_tree(
    source: Path,
    destination: Path,
    *,
    dispatch: GuestBootDispatchManifest,
    guest_config_bytes: bytes,
    source_date_epoch: int,
    production: bool,
) -> None:
    _copy_tree_deterministic(source, destination)
    executable_relative = _harness_relative_path(dispatch.guest_executable_path)
    config_relative = _harness_relative_path(dispatch.guest_config_path)
    executable = destination.joinpath(*executable_relative.parts)
    executable.chmod(0o555)
    config = destination.joinpath(*config_relative.parts)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_bytes(guest_config_bytes)
    config.chmod(0o400)
    _normalize_tree(
        destination,
        source_date_epoch=source_date_epoch,
        production=production,
        harness=True,
        readonly_config_path=config,
    )
    if stat.S_IMODE(executable.stat().st_mode) != 0o555 or stat.S_IMODE(config.stat().st_mode) != 0o400:
        raise LaneAGuestDiskBuildError('fixed harness executable or configuration mode was not preserved')
    if _file_sha256(config) != hashlib.sha256(guest_config_bytes).hexdigest():
        raise LaneAGuestDiskBuildError('fixed harness configuration bytes were not preserved')


def _copy_tree_deterministic(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise LaneAGuestDiskBuildError('deterministic copy destination already exists')

    def copy_directory(source_directory: Path, destination_directory: Path) -> None:
        destination_directory.mkdir(mode=stat.S_IMODE(source_directory.lstat().st_mode))
        for item in sorted(source_directory.iterdir(), key=lambda path: path.name.encode('utf-8')):
            target = destination_directory / item.name
            metadata = item.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                copy_directory(item, target)
            elif stat.S_ISREG(metadata.st_mode):
                with item.open('rb') as input_stream, target.open('xb') as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                target.chmod(stat.S_IMODE(metadata.st_mode))
            elif stat.S_ISLNK(metadata.st_mode):
                target.symlink_to(os.readlink(item))
            else:
                raise LaneAGuestDiskBuildError('source tree contains an unsupported special file')

    copy_directory(source, destination)


def _normalize_tree(
    root: Path,
    *,
    source_date_epoch: int,
    production: bool,
    harness: bool,
    readonly_config_path: Path | None = None,
) -> None:
    entries = [root, *sorted(root.rglob('*'), key=lambda path: path.as_posix().encode('utf-8'))]
    uid = 0 if production else os.geteuid()
    gid = 0 if production else os.getegid()
    for entry in reversed(entries):
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            os.lchown(entry, uid, gid)
            try:
                os.utime(entry, (source_date_epoch, source_date_epoch), follow_symlinks=False)
            except (NotImplementedError, OSError):
                if production:
                    raise LaneAGuestDiskBuildError('could not normalize symlink metadata') from None
            continue
        if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise LaneAGuestDiskBuildError('guest source tree contains an unsupported special file')
        mode = stat.S_IMODE(metadata.st_mode) & 0o1777
        if harness:
            if stat.S_ISDIR(metadata.st_mode):
                # mke2fs creates the filesystem root inode itself and does not apply ``-d``
                # source-root permissions to it.  Pin that measured behavior explicitly; all
                # descendant harness directories remain non-writable and the device is mounted
                # read-only by PID 1.
                mode = 0o755 if entry == root else 0o555
            elif readonly_config_path is not None and entry == readonly_config_path:
                mode = 0o400
            else:
                mode = 0o555 if mode & 0o111 else 0o444
        entry.chmod(mode)
        os.chown(entry, uid, gid)
        os.utime(entry, (source_date_epoch, source_date_epoch), follow_symlinks=False)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _validate_tree(root: Path, *, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise LaneAGuestDiskBuildError(f'{label} source must be a non-symlink directory')
    for path in sorted(root.rglob('*'), key=lambda item: item.as_posix().encode('utf-8')):
        relative = path.relative_to(root)
        for component in relative.parts:
            if _SAFE_PATH_COMPONENT.fullmatch(component) is None:
                raise LaneAGuestDiskBuildError(f'{label} source has an unsafe path component')
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            _validate_logical_symlink(PurePosixPath(relative.as_posix()), os.readlink(path), label=label)
        elif not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise LaneAGuestDiskBuildError(f'{label} source contains an unsupported special file')


def _validate_logical_symlink(path: PurePosixPath, target: str, *, label: str) -> tuple[str, ...]:
    if not target or '\x00' in target or len(target.encode('utf-8')) > 4096:
        raise LaneAGuestDiskBuildError(f'{label} contains an empty, NUL, or oversized symlink')
    target_path = PurePosixPath(target)
    parts: list[str] = [] if target_path.is_absolute() else list(path.parent.parts)
    for component in target_path.parts:
        if component in {'', '/', '.'}:
            continue
        if component == '..':
            if not parts:
                raise LaneAGuestDiskBuildError(f'{label} contains a symlink escaping the guest root')
            parts.pop()
            continue
        if _SAFE_PATH_COMPONENT.fullmatch(component) is None:
            raise LaneAGuestDiskBuildError(f'{label} contains an unsafe symlink target')
        parts.append(component)
    return tuple(parts)


def _resolve_chroot_path(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    pending = list(relative.parts)
    resolved: list[str] = []
    symlink_count = 0
    while pending:
        component = pending.pop(0)
        candidate = root.joinpath(*resolved, component)
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise LaneAGuestDiskBuildError(f'{label} lacks required {relative.as_posix()}') from error
        if stat.S_ISLNK(metadata.st_mode):
            symlink_count += 1
            if symlink_count > 40:
                raise LaneAGuestDiskBuildError(f'{label} contains a symlink loop')
            link_path = PurePosixPath(*resolved, component)
            pending = [*_validate_logical_symlink(link_path, os.readlink(candidate), label=label), *pending]
            resolved = []
        else:
            resolved.append(component)
    return root.joinpath(*resolved)


def _guest_overlay_target(root: Path, relative: PurePosixPath) -> Path:
    """Resolve existing parents as guest-root symlinks; never follow a host-root absolute link."""

    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise LaneAGuestDiskBuildError('overlay path is not normalized relative to the guest root')
    current = root
    logical_parts: list[str] = []
    for component in relative.parent.parts:
        candidate = current / component
        if candidate.is_symlink():
            guest_link = PurePosixPath(*logical_parts, component)
            resolved_parts = _validate_logical_symlink(
                guest_link,
                os.readlink(candidate),
                label='base rootfs',
            )
            current = root.joinpath(*resolved_parts)
            try:
                metadata = current.lstat()
            except OSError as error:
                raise LaneAGuestDiskBuildError('overlay parent symlink target is unavailable') from error
            if not stat.S_ISDIR(metadata.st_mode):
                raise LaneAGuestDiskBuildError('overlay parent symlink does not resolve to a guest directory')
            logical_parts = list(resolved_parts)
            continue
        if candidate.exists():
            if not candidate.is_dir():
                raise LaneAGuestDiskBuildError('overlay parent collides with a non-directory')
        else:
            candidate.mkdir(mode=0o755)
        current = candidate
        logical_parts.append(component)
    return current / relative.name


def _tree_sha256(root: Path, *, excluded_paths: set[str] | None = None) -> str:
    excluded = excluded_paths or set()
    digest = hashlib.sha256()
    for path in [root, *sorted(root.rglob('*'), key=lambda item: item.as_posix().encode('utf-8'))]:
        relative = path.relative_to(root).as_posix() or '.'
        if relative in excluded:
            continue
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind = 'directory'
            content_sha256 = hashlib.sha256(b'').hexdigest()
        elif stat.S_ISREG(metadata.st_mode):
            kind = 'file'
            content_sha256 = _file_sha256(path)
        elif stat.S_ISLNK(metadata.st_mode):
            kind = 'symlink'
            content_sha256 = hashlib.sha256(os.readlink(path).encode('utf-8')).hexdigest()
        else:
            raise LaneAGuestDiskBuildError('tree hash encountered an unsupported special file')
        digest.update(
            canonical_json_bytes(
                {
                    'path': relative,
                    'kind': kind,
                    'mode': stat.S_IMODE(metadata.st_mode),
                    'uid': metadata.st_uid,
                    'gid': metadata.st_gid,
                    'content_sha256': content_sha256,
                }
            )
        )
        digest.update(b'\n')
    return digest.hexdigest()


def _expected_tree_entries(root: Path) -> tuple[ExpectedGuestTreeEntry, ...]:
    entries: list[ExpectedGuestTreeEntry] = []
    for path in [root, *sorted(root.rglob('*'), key=lambda item: item.as_posix().encode('utf-8'))]:
        relative = path.relative_to(root).as_posix()
        # ``Path.relative_to`` represents the root itself as ``Path('.')``.  Never leak that host
        # spelling into the guest protocol: debugfs and the inspection validator both require the
        # canonical filesystem-root spelling ``/`` rather than ``/.``.
        guest_path = '/' if relative == '.' else f'/{relative}'
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind: Literal['directory', 'file', 'symlink'] = 'directory'
            content_sha256 = hashlib.sha256(b'').hexdigest()
        elif stat.S_ISREG(metadata.st_mode):
            kind = 'file'
            content_sha256 = _file_sha256(path)
        elif stat.S_ISLNK(metadata.st_mode):
            kind = 'symlink'
            content_sha256 = hashlib.sha256(os.readlink(path).encode('utf-8')).hexdigest()
        else:
            raise LaneAGuestDiskBuildError('tree inspection encountered an unsupported special file')
        entries.append(
            ExpectedGuestTreeEntry(
                path=guest_path,
                kind=kind,
                mode=stat.S_IMODE(metadata.st_mode),
                uid=metadata.st_uid,
                gid=metadata.st_gid,
                content_sha256=content_sha256,
            )
        )
    return tuple(entries)


def _tree_payload_byte_count(root: Path) -> int:
    total = 0
    for path in root.rglob('*'):
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
        elif stat.S_ISLNK(metadata.st_mode):
            total += len(os.readlink(path).encode('utf-8'))
        if total > _MAX_SOURCE_BYTES:
            raise LaneAGuestDiskBuildError('source directory payload is oversized')
    return total


def _build_contract_bytes(
    *,
    source_date_epoch: int,
    dispatch: GuestBootDispatchManifest,
) -> bytes:
    canonical_dispatch = GuestBootDispatchManifest.model_validate_json(canonical_json_bytes(dispatch))
    return canonical_json_bytes(
        {
            'schema_version': 'vaxreplay.lane-a-guest-disk-build-contract.v0.2',
            'mke2fs_argv': _MKE2FS_ARGV_TEMPLATE,
            'environment': _build_environment(source_date_epoch=source_date_epoch),
            'rootfs': {
                'uuid': LANE_A_ROOTFS_UUID,
                'label': LANE_A_ROOTFS_LABEL,
                'init_bytes_utf8': _lane_a_init_bytes(canonical_dispatch).decode('ascii'),
                'harness_device': '/dev/vdb',
                'harness_mountpoint': '/opt/vaxreplay',
                'harness_read_only': True,
                'scratch_device': '/dev/vdc',
                'scratch_mountpoint': '/var/lib/vaxreplay/scratch',
                'scratch_writable': True,
                'workspace_path_present': False,
                'guest_environment': (),
                'submitted_command_string_or_shell_construction_allowed': False,
            },
            'harness': {
                'uuid': LANE_A_HARNESS_UUID,
                'label': LANE_A_HARNESS_LABEL,
                'guest_boot_dispatch': canonical_dispatch.model_dump(mode='json'),
                'guest_boot_dispatch_manifest_sha256': (guest_boot_dispatch_manifest_sha256(canonical_dispatch)),
            },
        }
    )


def _build_environment(*, source_date_epoch: int) -> dict[str, str]:
    return {
        **_TOOL_ENVIRONMENT,
        'E2FSPROGS_FAKE_TIME': str(source_date_epoch),
        'SOURCE_DATE_EPOCH': str(source_date_epoch),
    }


def _build_environment_bytes(*, source_date_epoch: int) -> bytes:
    return canonical_json_bytes(_build_environment(source_date_epoch=source_date_epoch))


def _inspection_argv_bytes() -> bytes:
    return canonical_json_bytes(
        {
            'e2fsck': ('$E2FSCK', '-fn', '$IMAGE'),
            'debugfs_stat': ('$DEBUGFS', '-R', 'stat $GUEST_PATH', '$IMAGE'),
            'debugfs_dump': ('$DEBUGFS', '-R', 'dump $GUEST_PATH $OUTPUT', '$IMAGE'),
        }
    )


def _inspection_contract_bytes() -> bytes:
    return canonical_json_bytes(
        {
            'schema_version': 'vaxreplay.lane-a-guest-disk-inspection.v0.2',
            'health': 'pinned e2fsck -fn must exit 0 for both copies of both disks',
            'entries': 'pinned debugfs stat checks every expected staged path type, owner, and mode',
            'content': 'pinned debugfs dump checks every staged non-directory SHA-256',
            'inventory_scope': (
                'expected entries only; extra filesystem-generated or unexpected paths are not enumerated or rejected'
            ),
            'parity': (
                'both separately staged ext4 outputs on the same host and in the same process must be '
                'byte-identical before publication'
            ),
        }
    )


def _require_safe_guest_path(path: str) -> None:
    if path == '/':
        return
    pure = PurePosixPath(path)
    if not pure.is_absolute() or pure.as_posix() != path or '..' in pure.parts:
        raise LaneAGuestDiskBuildError('guest inspection path is not normalized')
    if any(
        component not in {'/', ''} and _SAFE_PATH_COMPONENT.fullmatch(component) is None for component in pure.parts
    ):
        raise LaneAGuestDiskBuildError('guest inspection path has an unsafe component')


def _validate_image_size(value: int, *, label: str, testing_only: bool = False) -> None:
    minimum = 4096 if testing_only else 64 * 1024 * 1024
    if value < minimum or value > 16 * 1024 * 1024 * 1024 or value % 4096:
        qualifier = '4 KiB' if testing_only else '64 MiB'
        raise LaneAGuestDiskBuildError(f'{label} size must be a 4 KiB multiple from {qualifier} to 16 GiB')


def _require_identical_files(first: Path, second: Path, *, label: str) -> None:
    if first.stat().st_size != second.stat().st_size or not hmac.compare_digest(
        _file_sha256(first), _file_sha256(second)
    ):
        raise LaneAGuestDiskBuildError(f'{label} was not byte-identical across two clean builds')


def _validate_output_parent(path: Path, *, production: bool) -> None:
    parent = path.parent
    try:
        metadata = parent.lstat()
        resolved = parent.resolve(strict=True)
    except OSError as error:
        raise LaneAGuestDiskBuildError('output parent must already exist') from error
    if parent.is_symlink() or not stat.S_ISDIR(metadata.st_mode) or resolved != parent:
        raise LaneAGuestDiskBuildError('output parent must be a real non-symlink directory')
    if production and (metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022):
        raise LaneAGuestDiskBuildError('production output parent must be root-owned and not group/world writable')


def _publish_create_once_streaming(path: Path, source: Path, *, mode: int) -> None:
    parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
            dir_fd=parent_descriptor,
        )
        with source.open('rb') as input_stream, os.fdopen(descriptor, 'wb', closefd=False) as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.fsync(parent_descriptor)
    except OSError as error:
        raise LaneAGuestDiskBuildError('create-once output publication failed') from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _publish_create_once_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    with tempfile.NamedTemporaryFile(prefix='.vaxreplay-lane-a-receipt.', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        _publish_create_once_streaming(path, temporary, mode=mode)
    finally:
        temporary.unlink(missing_ok=True)


def _read_regular_file_no_follow(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path.expanduser().absolute(), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum_bytes:
            raise LaneAGuestDiskBuildError('artifact is not a bounded regular file')
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise LaneAGuestDiskBuildError('artifact cannot be opened safely') from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    payload = b''.join(chunks)
    if len(payload) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise LaneAGuestDiskBuildError('artifact changed while it was read')
    return payload


def _regular_file_matches(path: Path, expected_sha256: str, expected_byte_count: int) -> bool:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_byte_count:
            return False
        observed_sha256 = _descriptor_sha256(descriptor)
        after = os.fstat(descriptor)
    except OSError:
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) and hmac.compare_digest(observed_sha256, expected_sha256)


def _file_sha256(path: Path) -> str:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LaneAGuestDiskBuildError('hash input is not a regular file')
        digest = _descriptor_sha256(descriptor)
        after = os.fstat(descriptor)
    except OSError as error:
        raise LaneAGuestDiskBuildError('hash input cannot be opened safely') from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise LaneAGuestDiskBuildError('hash input changed while it was read')
    return digest


def _open_pinned_root_executable(path: Path, expected_sha256: str, *, label: str) -> int:
    _require_sha256(expected_sha256, label=f'{label} digest')
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.expanduser().absolute(), flags)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise LaneAGuestDiskBuildError(f'pinned {label} is unavailable') from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not stat.S_IMODE(metadata.st_mode) & 0o111
        or _descriptor_sha256(descriptor) != expected_sha256
    ):
        os.close(descriptor)
        raise LaneAGuestDiskBuildError(f'pinned {label} identity or ownership is invalid')
    return descriptor


def _inspect_elf_linkage(descriptor: int) -> GuestDiskToolLinkage:
    """Classify a pinned ELF by parsing its program headers without another host tool."""

    try:
        metadata = os.fstat(descriptor)
        header = os.pread(descriptor, min(metadata.st_size, 64), 0)
        if len(header) < 52 or header[:4] != b'\x7fELF' or header[6] != 1:
            raise LaneAGuestDiskBuildError('pinned ext4 tool is not a supported ELF executable')
        elf_class = header[4]
        byte_order = header[5]
        if byte_order == 1:
            endian = '<'
        elif byte_order == 2:
            endian = '>'
        else:
            raise LaneAGuestDiskBuildError('pinned ext4 tool has an unsupported ELF byte order')
        if elf_class == 1:
            program_header_offset = struct.unpack_from(f'{endian}I', header, 28)[0]
            program_header_entry_size = struct.unpack_from(f'{endian}H', header, 42)[0]
            program_header_count = struct.unpack_from(f'{endian}H', header, 44)[0]
            minimum_entry_size = 32
        elif elf_class == 2 and len(header) >= 64:
            program_header_offset = struct.unpack_from(f'{endian}Q', header, 32)[0]
            program_header_entry_size = struct.unpack_from(f'{endian}H', header, 54)[0]
            program_header_count = struct.unpack_from(f'{endian}H', header, 56)[0]
            minimum_entry_size = 56
        else:
            raise LaneAGuestDiskBuildError('pinned ext4 tool has an unsupported ELF class')
        if (
            program_header_count == 0
            or program_header_count == 0xFFFF
            or program_header_entry_size < minimum_entry_size
            or program_header_offset > metadata.st_size
            or program_header_count > (metadata.st_size - program_header_offset) // program_header_entry_size
        ):
            raise LaneAGuestDiskBuildError('pinned ext4 tool has an invalid ELF program-header table')
        for index in range(program_header_count):
            offset = program_header_offset + (index * program_header_entry_size)
            program_type_bytes = os.pread(descriptor, 4, offset)
            if len(program_type_bytes) != 4:
                raise LaneAGuestDiskBuildError('pinned ext4 tool ELF changed during linkage inspection')
            if struct.unpack(f'{endian}I', program_type_bytes)[0] == 3:  # PT_INTERP
                return GuestDiskToolLinkage.DYNAMIC_ELF
    except OSError as error:
        raise LaneAGuestDiskBuildError('pinned ext4 tool ELF linkage cannot be inspected') from error
    return GuestDiskToolLinkage.STATIC_ELF


def _probe_tool_version(descriptor: int) -> str:
    completed = _run_procfd_tool(descriptor, ('-V',), timeout=10)
    output = (completed.stdout + completed.stderr).decode('ascii', errors='strict').strip().replace('\n', ' ')
    if completed.returncode != 0 or not output or len(output) > 200:
        raise LaneAGuestDiskBuildError('pinned ext4 tool version probe failed')
    return output


def _run_procfd_tool(
    descriptor: int,
    arguments: tuple[str, ...],
    *,
    timeout: int,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            (f'/proc/self/fd/{descriptor}', *arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd='/',
            env=environment or _TOOL_ENVIRONMENT,
            pass_fds=(descriptor,),
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise LaneAGuestDiskBuildError('pinned ext4 tool execution failed') from error


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if re.fullmatch(_SHA256_PATTERN, value) is None:
        raise LaneAGuestDiskBuildError(f'{label} must be a lowercase SHA-256 digest')


__all__ = [
    'BuiltLaneAGuestDisks',
    'DEFAULT_LANE_A_HARNESS_BYTES',
    'DEFAULT_LANE_A_ROOTFS_BYTES',
    'ExpectedGuestTreeEntry',
    'Ext4BuildRequest',
    'GuestDiskCommandExecutor',
    'GuestDiskExecutionBoundary',
    'GuestDiskOutputIdentity',
    'GuestDiskSourceIdentity',
    'GuestDiskSourceKind',
    'GuestDiskToolIdentity',
    'GuestDiskToolLinkage',
    'GuestDiskToolRuntimeBinding',
    'GuestDiskToolRuntimeClosureManifest',
    'GuestDiskToolRuntimeDependency',
    'LANE_A_GUEST_CONFIG_PATH',
    'LANE_A_GUEST_EXECUTABLE_PATH',
    'LANE_A_HARNESS_LABEL',
    'LANE_A_HARNESS_UUID',
    'LANE_A_ROOTFS_LABEL',
    'LANE_A_ROOTFS_UUID',
    'LaneAGuestDiskBuildError',
    'LaneAGuestDiskBuildReceipt',
    'PinnedLinuxExt4Executor',
    'VerifiedLaneAGuestDisks',
    'build_lane_a_guest_disks',
    'compute_testing_guest_disk_source_directory_sha256',
    'lane_a_guest_disk_build_receipt_sha256',
    'load_pinned_guest_disk_tool_runtime_closure_manifest',
    'load_pinned_lane_a_guest_config_bytes',
    'load_lane_a_guest_disk_build_receipt',
    'verify_lane_a_guest_disk_build',
    'verify_lane_a_guest_disk_build_parity',
]
