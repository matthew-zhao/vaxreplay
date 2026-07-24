from __future__ import annotations

import hashlib
import os
import platform
import stat
import tarfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.agentic.claude_code_guest_adapter import (
    CLAUDE_CODE_SUPPORTED_VENDOR_VERSION,
    CLAUDE_CODE_VENDOR_EXECUTABLE_PATH,
    claude_code_vendor_argv_template,
)
from vaxreplay.agentic.clinical_guest_bootstrap import (
    CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
    ClinicalGuestBootstrapTrustAnchor,
    ClinicalGuestRpcLimits,
    clinical_guest_bootstrap_authorization_key_id,
)
from vaxreplay.agentic.clinical_guest_executable import LaneAClinicalGuestConfig
from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_ACTION_SCHEMA_SHA256,
    LANE_A_GUEST_HARNESS_POLICY_ID,
)
from vaxreplay.agentic.codex_guest_adapter import (
    CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
    CODEX_VENDOR_EXECUTABLE_PATH,
    codex_vendor_argv_template,
)
from vaxreplay.agentic.guest_boot_dispatch import (
    HEADLESS_GUEST_CONFIG_PATH,
    HEADLESS_GUEST_EXECUTABLE_PATH,
    GuestBootConfigSchema,
    GuestBootDispatchAdmission,
    GuestBootDispatchManifest,
    guest_boot_dispatch_manifest_sha256,
)
from vaxreplay.agentic.guest_disk_build import (
    ExpectedGuestTreeEntry,
    Ext4BuildRequest,
    GuestDiskExecutionBoundary,
    GuestDiskSourceKind,
    GuestDiskToolIdentity,
    GuestDiskToolLinkage,
    GuestDiskToolRuntimeBinding,
    GuestDiskToolRuntimeClosureManifest,
    GuestDiskToolRuntimeDependency,
    LaneAGuestDiskBuildError,
    LaneAGuestDiskBuildReceipt,
    PinnedLinuxExt4Executor,
    _debugfs_fast_symlink_target_sha256,
    _lane_a_init_bytes,
    build_lane_a_guest_disks,
    compute_testing_guest_disk_source_directory_sha256,
    lane_a_guest_disk_build_receipt_sha256,
    load_lane_a_guest_disk_build_receipt,
    load_pinned_guest_disk_tool_runtime_closure_manifest,
    verify_lane_a_guest_disk_build,
    verify_lane_a_guest_disk_build_parity,
)
from vaxreplay.agentic.headless_guest_adapter import (
    HeadlessGuestAdapterConfig,
    HeadlessInvocationProtocol,
    HeadlessResponseChannel,
    headless_guest_adapter_config_sha256,
)
from vaxreplay.agentic.submitted_harness import HarnessFamily, HarnessRuntimeSupport
from vaxreplay.bundle import canonical_json_bytes

_SOURCE_DATE_EPOCH = 1_700_000_000
_IMAGE_BYTES = 4 * 4096


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config_bytes() -> bytes:
    private_key = Ed25519PrivateKey.from_private_bytes(b'\x11' * 32)
    public_key = private_key.public_key().public_bytes_raw()
    limits = ClinicalGuestRpcLimits(
        maximum_frame_body_bytes=1024 * 1024,
        maximum_session_wire_bytes=8 * 1024 * 1024,
        maximum_requests=100,
        maximum_list_entries=100,
        maximum_read_bytes=32_768,
        maximum_search_results=20,
        maximum_submission_bytes=65_536,
    )
    anchor = ClinicalGuestBootstrapTrustAnchor(
        authorization_key_id=clinical_guest_bootstrap_authorization_key_id(public_key),
        ed25519_public_key_hex=public_key.hex(),
        execution_policy_sha256='7' * 64,
        worker_bootstrap_profile_sha256='8' * 64,
        harness_policy_id=LANE_A_GUEST_HARNESS_POLICY_ID,
        harness_policy_sha256=CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
        action_schema_sha256=LANE_A_GUEST_ACTION_SCHEMA_SHA256,
        rpc_limits=limits,
    )
    return canonical_json_bytes(LaneAClinicalGuestConfig(trust_anchor=anchor, guest_rpc_port=7000))


def _write_executable(path: Path, payload: bytes = b'#!/bin/sh\nexit 0\n') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)


def test_debugfs_fast_symlink_target_is_hashed_but_block_backed_symlink_is_not_guessed() -> None:
    output = 'Inode: 15   Type: symlink    Mode:  0777\nFast link dest: "/bin/busybox"\n'
    assert _debugfs_fast_symlink_target_sha256(output) == _sha256(b'/bin/busybox')
    assert _debugfs_fast_symlink_target_sha256('Inode: 15   Type: symlink\nEXTENTS:\n') is None


def _source_trees(root: Path) -> tuple[Path, Path, bytes]:
    base = root / 'base-rootfs'
    (base / 'bin').mkdir(parents=True)
    _write_executable(base / 'bin' / 'busybox', b'fake-static-busybox')
    (base / 'bin' / 'sh').symlink_to('/bin/busybox')
    (base / 'bin' / 'mount').symlink_to('/bin/busybox')
    (base / 'bin' / 'mkdir').symlink_to('/bin/busybox')
    (base / 'sbin').mkdir()

    harness = root / 'harness-payload'
    executable = b'#!/bin/sh\nexit 0\n'
    _write_executable(harness / 'bin' / 'vaxreplay-lane-a-clinical-guest', executable)
    dependency = harness / 'lib' / 'vaxreplay' / 'runtime.dat'
    dependency.parent.mkdir(parents=True)
    dependency.write_bytes(b'pinned dependency closure')
    dependency.chmod(0o444)
    return base, harness, executable


def _development_headless_config(family: HarnessFamily) -> HeadlessGuestAdapterConfig:
    common = {
        'family': family,
        'adapter_executable_sha256': '1' * 64,
        'vendor_executable_sha256': '2' * 64,
        'complete_dependency_closure_sha256': '3' * 64,
        'vendor_version_output_sha256': '4' * 64,
        'vendor_config_template_sha256': '5' * 64,
        'adapter_implementation_checked_in': True,
        'provider_shim_implementation_checked_in': True,
        'workspace_materialization_bridge_implementation_checked_in': True,
    }
    if family == HarnessFamily.CODEX:
        return HeadlessGuestAdapterConfig(
            **common,
            invocation_protocol=HeadlessInvocationProtocol.CODEX_EXEC,
            vendor_executable_path=CODEX_VENDOR_EXECUTABLE_PATH,
            vendor_reported_version=CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
            vendor_argv_template=codex_vendor_argv_template(),
            response_channel=HeadlessResponseChannel.BOUNDED_OUTPUT_FILE,
            local_shell_enabled=True,
        )
    return HeadlessGuestAdapterConfig(
        **common,
        invocation_protocol=HeadlessInvocationProtocol.CLAUDE_PRINT,
        vendor_executable_path=CLAUDE_CODE_VENDOR_EXECUTABLE_PATH,
        vendor_reported_version=CLAUDE_CODE_SUPPORTED_VENDOR_VERSION,
        vendor_argv_template=claude_code_vendor_argv_template(),
        response_channel=HeadlessResponseChannel.BOUNDED_JSON_STDOUT,
        local_shell_enabled=False,
    )


def _write_normalized_tar(source: Path, destination: Path) -> None:
    with tarfile.open(destination, 'w', format=tarfile.GNU_FORMAT) as archive:
        for path in sorted(source.rglob('*'), key=lambda item: item.relative_to(source).as_posix()):
            relative = path.relative_to(source).as_posix()
            metadata = path.lstat()
            entry = tarfile.TarInfo(relative)
            entry.uid = 0
            entry.gid = 0
            entry.uname = ''
            entry.gname = ''
            entry.mtime = _SOURCE_DATE_EPOCH
            entry.mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                entry.type = tarfile.DIRTYPE
                archive.addfile(entry)
            elif stat.S_ISLNK(metadata.st_mode):
                entry.type = tarfile.SYMTYPE
                entry.linkname = os.readlink(path)
                archive.addfile(entry)
            else:
                entry.size = metadata.st_size
                with path.open('rb') as stream:
                    archive.addfile(entry, stream)


class _DeterministicExecutor:
    def __init__(self, *, corrupt_second_rootfs: bool = False) -> None:
        self.corrupt_second_rootfs = corrupt_second_rootfs
        self.requests: list[Ext4BuildRequest] = []
        self.inspections: list[tuple[Path, tuple[ExpectedGuestTreeEntry, ...]]] = []
        self._tools = tuple(
            GuestDiskToolIdentity(
                name=name,
                sha256=_sha256(f'fake-{name}'.encode()),
                version=f'{name} fake-v1',
                executed_via_proc_self_fd=False,
                linkage=GuestDiskToolLinkage.TEST_UNKNOWN,
            )
            for name in ('mke2fs', 'e2fsck', 'debugfs')
        )

    @property
    def boundary(self) -> GuestDiskExecutionBoundary:
        return GuestDiskExecutionBoundary.TEST_EXECUTOR

    @property
    def tool_identities(self) -> tuple[GuestDiskToolIdentity, GuestDiskToolIdentity, GuestDiskToolIdentity]:
        return self._tools[0], self._tools[1], self._tools[2]

    def build_ext4(self, request: Ext4BuildRequest) -> None:
        self.requests.append(request)
        inventory: list[dict[str, object]] = []
        for path in [request.tree, *sorted(request.tree.rglob('*'))]:
            relative = path.relative_to(request.tree).as_posix() or '.'
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                kind = 'directory'
                content = b''
            elif stat.S_ISREG(metadata.st_mode):
                kind = 'file'
                content = path.read_bytes()
            else:
                kind = 'symlink'
                content = os.readlink(path).encode()
            inventory.append(
                {
                    'path': relative,
                    'kind': kind,
                    'mode': stat.S_IMODE(metadata.st_mode),
                    'sha256': _sha256(content),
                }
            )
        seed = canonical_json_bytes(
            {
                'uuid': request.uuid,
                'label': request.label,
                'source_date_epoch': request.source_date_epoch,
                'inventory': inventory,
            }
        )
        block = hashlib.sha256(seed).digest()
        body = (block * (request.byte_count // len(block) + 1))[: request.byte_count]
        if self.corrupt_second_rootfs and request.label == 'vaxlanea-root' and len(self.requests) == 2:
            body = body[:-1] + bytes([body[-1] ^ 1])
        request.output.write_bytes(body)

    def inspect_ext4(self, image: Path, expected_entries: tuple[ExpectedGuestTreeEntry, ...]) -> None:
        assert image.is_file()
        assert expected_entries
        self.inspections.append((image, expected_entries))


def _build(tmp_path: Path, *, executor: _DeterministicExecutor | None = None):
    base, harness, executable = _source_trees(tmp_path)
    config_bytes = _config_bytes()
    executor = executor or _DeterministicExecutor()
    build = build_lane_a_guest_disks(
        base_rootfs_source=base,
        expected_base_rootfs_source_sha256=compute_testing_guest_disk_source_directory_sha256(base),
        harness_payload_source=harness,
        expected_harness_payload_source_sha256=compute_testing_guest_disk_source_directory_sha256(harness),
        expected_guest_executable_sha256=_sha256(executable),
        guest_config_bytes=config_bytes,
        executor=executor,
        output_rootfs_path=tmp_path / 'lane-a-rootfs.ext4',
        output_harness_path=tmp_path / 'lane-a-harness.ext4',
        output_receipt_path=tmp_path / 'lane-a-build-receipt.json',
        source_date_epoch=_SOURCE_DATE_EPOCH,
        rootfs_byte_count=_IMAGE_BYTES,
        harness_byte_count=_IMAGE_BYTES,
        testing_only=True,
    )
    return build, executor, base, harness, executable, config_bytes


def test_two_clean_builds_are_identical_inspected_and_bound_to_fixed_init(tmp_path: Path) -> None:
    build, executor, base, harness, executable, config_bytes = _build(tmp_path)

    assert len(executor.requests) == 4
    assert len(executor.inspections) == 4
    assert build.receipt.production_eligible is False
    assert build.receipt.reproducibility_scope == 'same_host_same_process_two_clean_stagings'
    assert build.receipt.same_host_separately_staged_build_count == 2
    assert build.receipt.same_host_byte_identical_rebuild_verified is True
    assert build.receipt.cross_host_hermetic_reproducibility_claimed is False
    assert build.receipt.tool_runtime_closure_manifest_sha256 is None
    assert build.receipt.base_rootfs_source.kind == GuestDiskSourceKind.TEST_DIRECTORY
    assert build.receipt.harness_payload_source.kind == GuestDiskSourceKind.TEST_DIRECTORY
    assert build.receipt.harness_device == '/dev/vdb'
    assert build.receipt.harness_mountpoint == '/opt/vaxreplay'
    assert build.receipt.harness_mounted_read_only is True
    assert build.receipt.scratch_device == '/dev/vdc'
    assert build.receipt.scratch_mountpoint == '/var/lib/vaxreplay/scratch'
    assert build.receipt.scratch_mounted_writable is True
    assert build.receipt.workspace_disk_or_mount_present is False
    assert len(build.receipt.mke2fs_argv_sha256) == 64
    assert len(build.receipt.build_environment_sha256) == 64
    assert len(build.receipt.inspection_argv_sha256) == 64
    assert build.receipt.rootfs.expected_entry_inspection_passed is True
    assert build.receipt.harness.expected_entry_inspection_passed is True
    assert 'exact_tree_inspection_passed' not in build.receipt.rootfs.model_dump()
    assert build.receipt.fixed_guest_argv == (
        '/opt/vaxreplay/bin/vaxreplay-lane-a-clinical-guest',
        '--expected-config-sha256',
        _sha256(config_bytes),
    )

    root_entries = executor.inspections[0][1]
    harness_entries = executor.inspections[2][1]
    assert root_entries[0].path == '/'
    assert harness_entries[0].path == '/'
    assert harness_entries[0].mode == 0o755
    assert all(entry.mode == 0o555 for entry in harness_entries if entry.kind == 'directory' and entry.path != '/')
    init = next(entry for entry in root_entries if entry.path == '/sbin/init')
    init_bytes = _lane_a_init_bytes(_sha256(config_bytes))
    assert init.mode == 0o755
    assert init.content_sha256 == build.receipt.init_sha256 == _sha256(init_bytes)
    assert b'if ! /bin/mount -t proc ' in init_bytes
    assert b'then [ -r /proc/1/stat ]; fi\n' in init_bytes
    assert b'if ! /bin/mount -t sysfs ' in init_bytes
    assert b'then [ -d /sys/kernel ]; fi\n' in init_bytes
    assert b'if ! /bin/mount -t devtmpfs ' in init_bytes
    assert b'then [ -c /dev/null ]; fi\n' in init_bytes
    assert not any(entry.path == '/workspace' or entry.path.startswith('/workspace/') for entry in root_entries)
    guest = next(entry for entry in harness_entries if entry.path == '/bin/vaxreplay-lane-a-clinical-guest')
    config = next(entry for entry in harness_entries if entry.path == '/etc/lane-a-clinical-guest.json')
    assert (guest.mode, guest.content_sha256) == (0o555, _sha256(executable))
    assert (config.mode, config.content_sha256) == (0o400, _sha256(config_bytes))

    loaded = load_lane_a_guest_disk_build_receipt(
        build.receipt_path,
        expected_receipt_sha256=build.receipt_sha256,
    )
    assert loaded == build.receipt
    verified = verify_lane_a_guest_disk_build(
        receipt=loaded,
        rootfs_path=build.rootfs_path,
        harness_path=build.harness_path,
        expected_base_rootfs_source_sha256=compute_testing_guest_disk_source_directory_sha256(base),
        expected_harness_payload_source_sha256=compute_testing_guest_disk_source_directory_sha256(harness),
        expected_guest_executable_sha256=_sha256(executable),
        expected_guest_config_sha256=_sha256(config_bytes),
        expected_mke2fs_sha256=executor.tool_identities[0].sha256,
        expected_e2fsck_sha256=executor.tool_identities[1].sha256,
        expected_debugfs_sha256=executor.tool_identities[2].sha256,
        require_production=False,
    )
    assert verified.receipt_sha256 == build.receipt_sha256


@pytest.mark.parametrize('family', (HarnessFamily.CODEX, HarnessFamily.CLAUDE_CODE))
def test_checked_in_adapter_can_be_packaged_without_claiming_runtime_or_kvm_qualification(
    tmp_path: Path,
    family: HarnessFamily,
) -> None:
    base, harness, _native_executable = _source_trees(tmp_path)
    (harness / 'bin' / 'vaxreplay-lane-a-clinical-guest').unlink()
    guest_executable = b'#!/usr/bin/python3\nraise SystemExit(70)\n'
    _write_executable(
        harness / 'bin' / 'vaxreplay-headless-guest-adapter',
        guest_executable,
    )
    config = _development_headless_config(family)
    config_bytes = canonical_json_bytes(config)
    config_sha256 = headless_guest_adapter_config_sha256(config)
    dispatch = GuestBootDispatchManifest(
        family=family,
        runtime_support=HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED,
        admission=GuestBootDispatchAdmission.DEVELOPMENT_PACKAGING_ONLY,
        config_schema=GuestBootConfigSchema.HEADLESS_ADAPTER,
        guest_executable_path=HEADLESS_GUEST_EXECUTABLE_PATH,
        guest_executable_sha256=_sha256(guest_executable),
        guest_config_path=HEADLESS_GUEST_CONFIG_PATH,
        guest_config_sha256=config_sha256,
        guest_argv=(
            HEADLESS_GUEST_EXECUTABLE_PATH,
            '--expected-config-sha256',
            config_sha256,
        ),
    )
    # The headless config also pins the outer executable.  Rebuild it with the exact test payload
    # digest so the generic config validator proves the cross-artifact binding.
    config = config.model_copy(update={'adapter_executable_sha256': dispatch.guest_executable_sha256})
    config_bytes = canonical_json_bytes(config)
    config_sha256 = headless_guest_adapter_config_sha256(config)
    dispatch = dispatch.model_copy(
        update={
            'guest_config_sha256': config_sha256,
            'guest_argv': (
                HEADLESS_GUEST_EXECUTABLE_PATH,
                '--expected-config-sha256',
                config_sha256,
            ),
        }
    )
    dispatch = GuestBootDispatchManifest.model_validate(dispatch.model_dump(mode='python'))
    executor = _DeterministicExecutor()

    build = build_lane_a_guest_disks(
        base_rootfs_source=base,
        expected_base_rootfs_source_sha256=compute_testing_guest_disk_source_directory_sha256(base),
        harness_payload_source=harness,
        expected_harness_payload_source_sha256=compute_testing_guest_disk_source_directory_sha256(harness),
        expected_guest_executable_sha256=_sha256(guest_executable),
        guest_config_bytes=config_bytes,
        guest_boot_dispatch=dispatch,
        expected_guest_boot_dispatch_manifest_sha256=(guest_boot_dispatch_manifest_sha256(dispatch)),
        executor=executor,
        output_rootfs_path=tmp_path / 'headless-rootfs.ext4',
        output_harness_path=tmp_path / 'headless-harness.ext4',
        output_receipt_path=tmp_path / 'headless-build-receipt.json',
        source_date_epoch=_SOURCE_DATE_EPOCH,
        rootfs_byte_count=_IMAGE_BYTES,
        harness_byte_count=_IMAGE_BYTES,
        testing_only=True,
    )

    assert build.receipt.guest_boot_dispatch == dispatch
    assert build.receipt.fixed_guest_environment == ()
    assert build.receipt.pid1_clears_ambient_environment is True
    assert build.receipt.submitted_command_string_or_shell_construction_allowed is False
    assert build.receipt.ambient_provider_route_or_credentials_present is False
    assert build.receipt.canonical_operator_runtime_supported is False
    assert build.receipt.linux_kvm_qualification_claimed_by_build_receipt is False
    root_entries = executor.inspections[0][1]
    init = next(item for item in root_entries if item.path == '/sbin/init')
    init_bytes = _lane_a_init_bytes(dispatch)
    assert init.content_sha256 == _sha256(init_bytes)
    assert b'exec /bin/busybox env -i -- ' in init_bytes
    assert b'sh -c' not in init_bytes
    harness_entries = executor.inspections[2][1]
    assert any(
        item.path == '/bin/vaxreplay-headless-guest-adapter' and item.content_sha256 == _sha256(guest_executable)
        for item in harness_entries
    )
    assert any(
        item.path == '/etc/headless-guest-adapter.json' and item.content_sha256 == config_sha256
        for item in harness_entries
    )

    verified = verify_lane_a_guest_disk_build(
        receipt=build.receipt,
        rootfs_path=build.rootfs_path,
        harness_path=build.harness_path,
        expected_base_rootfs_source_sha256=compute_testing_guest_disk_source_directory_sha256(base),
        expected_harness_payload_source_sha256=compute_testing_guest_disk_source_directory_sha256(harness),
        expected_guest_executable_sha256=_sha256(guest_executable),
        expected_guest_config_sha256=config_sha256,
        expected_guest_boot_dispatch=dispatch,
        expected_guest_boot_dispatch_manifest_sha256=(guest_boot_dispatch_manifest_sha256(dispatch)),
        expected_mke2fs_sha256=executor.tool_identities[0].sha256,
        expected_e2fsck_sha256=executor.tool_identities[1].sha256,
        expected_debugfs_sha256=executor.tool_identities[2].sha256,
        require_production=False,
    )
    assert verified.receipt_sha256 == build.receipt_sha256
    with pytest.raises(LaneAGuestDiskBuildError, match='boot-dispatch manifest pin'):
        verify_lane_a_guest_disk_build(
            receipt=build.receipt,
            rootfs_path=build.rootfs_path,
            harness_path=build.harness_path,
            expected_base_rootfs_source_sha256=(compute_testing_guest_disk_source_directory_sha256(base)),
            expected_harness_payload_source_sha256=(compute_testing_guest_disk_source_directory_sha256(harness)),
            expected_guest_executable_sha256=_sha256(guest_executable),
            expected_guest_config_sha256=config_sha256,
            expected_guest_boot_dispatch_manifest_sha256='f' * 64,
            expected_mke2fs_sha256=executor.tool_identities[0].sha256,
            expected_e2fsck_sha256=executor.tool_identities[1].sha256,
            expected_debugfs_sha256=executor.tool_identities[2].sha256,
            require_production=False,
        )


def test_testing_receipt_cannot_authorize_production_and_tampered_image_fails(tmp_path: Path) -> None:
    build, executor, base, harness, executable, config_bytes = _build(tmp_path)

    def verify(*, require_production: bool = True) -> None:
        verify_lane_a_guest_disk_build(
            receipt=build.receipt,
            rootfs_path=build.rootfs_path,
            harness_path=build.harness_path,
            expected_base_rootfs_source_sha256=compute_testing_guest_disk_source_directory_sha256(base),
            expected_harness_payload_source_sha256=compute_testing_guest_disk_source_directory_sha256(harness),
            expected_guest_executable_sha256=_sha256(executable),
            expected_guest_config_sha256=_sha256(config_bytes),
            expected_mke2fs_sha256=executor.tool_identities[0].sha256,
            expected_e2fsck_sha256=executor.tool_identities[1].sha256,
            expected_debugfs_sha256=executor.tool_identities[2].sha256,
            require_production=require_production,
        )

    with pytest.raises(LaneAGuestDiskBuildError, match='testing-only'):
        verify()

    with build.rootfs_path.open('r+b') as stream:
        stream.seek(0)
        stream.write(b'bad!')
    with pytest.raises(LaneAGuestDiskBuildError, match='rootfs bytes'):
        verify(require_production=False)


def test_create_once_noncanonical_loader_and_parity_fail_closed(tmp_path: Path) -> None:
    build, executor, base, harness, executable, config_bytes = _build(tmp_path)
    with pytest.raises(LaneAGuestDiskBuildError, match='create-once'):
        build_lane_a_guest_disks(
            base_rootfs_source=base,
            expected_base_rootfs_source_sha256=compute_testing_guest_disk_source_directory_sha256(base),
            harness_payload_source=harness,
            expected_harness_payload_source_sha256=compute_testing_guest_disk_source_directory_sha256(harness),
            expected_guest_executable_sha256=_sha256(executable),
            guest_config_bytes=config_bytes,
            executor=executor,
            output_rootfs_path=build.rootfs_path,
            output_harness_path=build.harness_path,
            output_receipt_path=build.receipt_path,
            source_date_epoch=_SOURCE_DATE_EPOCH,
            rootfs_byte_count=_IMAGE_BYTES,
            harness_byte_count=_IMAGE_BYTES,
            testing_only=True,
        )

    noncanonical = tmp_path / 'noncanonical.json'
    noncanonical.write_bytes(canonical_json_bytes(build.receipt) + b'\n')
    with pytest.raises(LaneAGuestDiskBuildError, match='canonical'):
        load_lane_a_guest_disk_build_receipt(
            noncanonical,
            expected_receipt_sha256=_sha256(noncanonical.read_bytes()),
        )

    changed_rootfs = build.receipt.rootfs.model_copy(update={'sha256': 'f' * 64})
    changed_receipt = build.receipt.model_copy(update={'rootfs': changed_rootfs})
    with pytest.raises(LaneAGuestDiskBuildError, match='parity'):
        verify_lane_a_guest_disk_build_parity(build.receipt, changed_receipt)
    assert lane_a_guest_disk_build_receipt_sha256(build.receipt) == build.receipt_sha256


def test_rebuild_nondeterminism_and_escaping_symlink_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(LaneAGuestDiskBuildError, match='byte-identical'):
        _build(tmp_path / 'nondeterministic', executor=_DeterministicExecutor(corrupt_second_rootfs=True))

    source = tmp_path / 'unsafe'
    source.mkdir()
    (source / 'escape').symlink_to('../../etc/passwd')
    with pytest.raises(LaneAGuestDiskBuildError, match='escaping'):
        compute_testing_guest_disk_source_directory_sha256(source)


def test_normalized_pinned_tar_sources_are_accepted_by_the_test_seam(tmp_path: Path) -> None:
    base, harness, executable = _source_trees(tmp_path / 'sources')
    base_tar = tmp_path / 'base.tar'
    harness_tar = tmp_path / 'harness.tar'
    _write_normalized_tar(base, base_tar)
    _write_normalized_tar(harness, harness_tar)
    executor = _DeterministicExecutor()
    config_bytes = _config_bytes()

    build = build_lane_a_guest_disks(
        base_rootfs_source=base_tar,
        expected_base_rootfs_source_sha256=_sha256(base_tar.read_bytes()),
        harness_payload_source=harness_tar,
        expected_harness_payload_source_sha256=_sha256(harness_tar.read_bytes()),
        expected_guest_executable_sha256=_sha256(executable),
        guest_config_bytes=config_bytes,
        executor=executor,
        output_rootfs_path=tmp_path / 'rootfs.ext4',
        output_harness_path=tmp_path / 'harness.ext4',
        output_receipt_path=tmp_path / 'receipt.json',
        source_date_epoch=_SOURCE_DATE_EPOCH,
        rootfs_byte_count=_IMAGE_BYTES,
        harness_byte_count=_IMAGE_BYTES,
        testing_only=True,
    )

    assert build.receipt.base_rootfs_source.kind == GuestDiskSourceKind.NORMALIZED_TAR
    assert build.receipt.harness_payload_source.kind == GuestDiskSourceKind.NORMALIZED_TAR
    assert build.receipt.production_eligible is False


def test_production_executor_fails_closed_off_linux_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr('vaxreplay.agentic.guest_disk_build.platform.system', lambda: 'Darwin')
    with pytest.raises(LaneAGuestDiskBuildError, match='root on Linux'):
        PinnedLinuxExt4Executor(
            mke2fs_path=tmp_path / 'mke2fs',
            expected_mke2fs_sha256='1' * 64,
            e2fsck_path=tmp_path / 'e2fsck',
            expected_e2fsck_sha256='2' * 64,
            debugfs_path=tmp_path / 'debugfs',
            expected_debugfs_sha256='3' * 64,
        )


def test_tool_runtime_closure_is_canonical_externally_pinned_and_complete(tmp_path: Path) -> None:
    bindings = tuple(
        GuestDiskToolRuntimeBinding(
            tool=name,  # type: ignore[arg-type]
            executable_sha256=_sha256(f'production-{name}'.encode()),
            linkage='static_elf',
            dependencies=(),
        )
        for name in ('mke2fs', 'e2fsck', 'debugfs')
    )
    manifest = GuestDiskToolRuntimeClosureManifest(
        platform_machine=platform.machine(),
        bindings=(bindings[0], bindings[1], bindings[2]),
    )
    payload = canonical_json_bytes(manifest)
    path = tmp_path / 'tool-runtime-closure.json'
    path.write_bytes(payload)

    assert (
        load_pinned_guest_disk_tool_runtime_closure_manifest(
            path,
            expected_sha256=_sha256(payload),
        )
        == manifest
    )
    with pytest.raises(LaneAGuestDiskBuildError, match='external digest pin'):
        load_pinned_guest_disk_tool_runtime_closure_manifest(
            path,
            expected_sha256='f' * 64,
        )

    noncanonical = tmp_path / 'noncanonical-tool-runtime-closure.json'
    noncanonical.write_bytes(payload + b'\n')
    with pytest.raises(LaneAGuestDiskBuildError, match='canonical'):
        load_pinned_guest_disk_tool_runtime_closure_manifest(
            noncanonical,
            expected_sha256=_sha256(noncanonical.read_bytes()),
        )


def test_dynamic_tool_manifest_pins_loader_and_libraries_but_cannot_claim_production() -> None:
    dependencies = (
        GuestDiskToolRuntimeDependency(
            logical_name='/lib64/ld-linux-x86-64.so.2',
            role='elf_interpreter',
            sha256='1' * 64,
            byte_count=4096,
        ),
        GuestDiskToolRuntimeDependency(
            logical_name='/lib64/libc.so.6',
            role='shared_library',
            sha256='2' * 64,
            byte_count=8192,
        ),
    )
    binding = GuestDiskToolRuntimeBinding(
        tool='mke2fs',
        executable_sha256='3' * 64,
        linkage='dynamic_elf',
        dependencies=dependencies,
    )
    assert binding.dependencies == dependencies

    with pytest.raises(ValueError, match='one interpreter and at least one shared library'):
        GuestDiskToolRuntimeBinding(
            tool='mke2fs',
            executable_sha256='3' * 64,
            linkage='dynamic_elf',
            dependencies=(),
        )


def test_receipt_claims_expected_entries_and_same_host_scope_only(tmp_path: Path) -> None:
    build, _executor, _base, _harness, _executable, _config = _build(tmp_path)
    payload = build.receipt.model_dump(mode='python')
    payload['rootfs']['exact_tree_inspection_passed'] = True
    with pytest.raises(ValueError, match='extra'):
        LaneAGuestDiskBuildReceipt.model_validate(payload)


def test_production_eligibility_requires_static_measured_tools_and_closure(tmp_path: Path) -> None:
    base, harness, executable = _source_trees(tmp_path / 'sources')
    base_tar = tmp_path / 'base.tar'
    harness_tar = tmp_path / 'harness.tar'
    _write_normalized_tar(base, base_tar)
    _write_normalized_tar(harness, harness_tar)
    executor = _DeterministicExecutor()
    build = build_lane_a_guest_disks(
        base_rootfs_source=base_tar,
        expected_base_rootfs_source_sha256=_sha256(base_tar.read_bytes()),
        harness_payload_source=harness_tar,
        expected_harness_payload_source_sha256=_sha256(harness_tar.read_bytes()),
        expected_guest_executable_sha256=_sha256(executable),
        guest_config_bytes=_config_bytes(),
        executor=executor,
        output_rootfs_path=tmp_path / 'rootfs.ext4',
        output_harness_path=tmp_path / 'harness.ext4',
        output_receipt_path=tmp_path / 'receipt.json',
        source_date_epoch=_SOURCE_DATE_EPOCH,
        rootfs_byte_count=_IMAGE_BYTES,
        harness_byte_count=_IMAGE_BYTES,
        testing_only=True,
    )
    payload = build.receipt.model_dump(mode='python')
    payload['execution_boundary'] = GuestDiskExecutionBoundary.PINNED_LINUX_PROCFS
    payload['production_eligible'] = True
    for name in ('mke2fs', 'e2fsck', 'debugfs'):
        payload[name]['executed_via_proc_self_fd'] = True
        payload[name]['linkage'] = GuestDiskToolLinkage.STATIC_ELF
    payload['tool_runtime_closure_manifest_sha256'] = 'a' * 64
    payload['tool_runtime_closure_manifest_byte_count'] = 4096
    payload['tool_runtime_closure_external_pin_checked'] = True
    payload['tool_runtime_closure_bindings_checked'] = True
    payload['tool_runtime_closure_contains_dynamic_linkage'] = False
    assert LaneAGuestDiskBuildReceipt.model_validate(payload).production_eligible is True

    payload['tool_runtime_closure_contains_dynamic_linkage'] = True
    for name in ('mke2fs', 'e2fsck', 'debugfs'):
        payload[name]['linkage'] = GuestDiskToolLinkage.DYNAMIC_ELF
    with pytest.raises(ValueError, match='fail closed'):
        LaneAGuestDiskBuildReceipt.model_validate(payload)

    payload['production_eligible'] = False
    assert LaneAGuestDiskBuildReceipt.model_validate(payload).production_eligible is False
