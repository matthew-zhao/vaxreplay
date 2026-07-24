"""Reproducible builder for the qualification-only Firecracker rootfs and harness disks.

This profile is intentionally distinct from the Lane A task guest.  A Linux/root build copies a
caller-pinned base rootfs, installs a fixed PID-1 script, installs the exact qualification probe and
package tree on a separate read-only harness, and invokes a digest-pinned ``mke2fs`` twice.  The
build is accepted only when both independent ext4 outputs are byte-identical.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.agentic.firecracker import FirecrackerWorkerSpec, firecracker_model_sha256
from vaxreplay.agentic.firecracker_qualification_guest import (
    QUALIFICATION_GUEST_CONFIG_PATH,
    QUALIFICATION_GUEST_EXECUTABLE_PATH,
    FirecrackerQualificationGuestConfig,
)
from vaxreplay.agentic.firecracker_qualification_probe import (
    FirecrackerQualificationGuestDiskBuildReceipt,
    FirecrackerQualificationProbeManifest,
    ed25519_public_key_bytes,
    firecracker_qualification_guest_key_id,
)
from vaxreplay.bundle import canonical_json_bytes

_ROOTFS_UUID = '00000000-0000-4000-8000-000000000101'
_HARNESS_UUID = '00000000-0000-4000-8000-000000000102'
_ROOTFS_LABEL = 'vaxqual-root'
_HARNESS_LABEL = 'vaxqual-harness'
_DEFAULT_ROOTFS_BYTES = 512 * 1024 * 1024
_DEFAULT_HARNESS_BYTES = 256 * 1024 * 1024
_SHA256_PATTERN = frozenset('0123456789abcdef')

_INIT_BYTES = b"""#!/bin/sh
set -eu
if ! /bin/mount -t proc -o nosuid,nodev,noexec proc /proc; then [ -r /proc/1/stat ]; fi
if ! /bin/mount -t sysfs -o nosuid,nodev,noexec sysfs /sys; then [ -d /sys/kernel ]; fi
if ! /bin/mount -t devtmpfs -o nosuid devtmpfs /dev; then [ -c /dev/null ]; fi
/bin/mkdir -p /opt/vaxreplay /workspace
/bin/mount -o ro /dev/vdb /opt/vaxreplay
/bin/mount -o rw /dev/vdc /workspace
status=0
/usr/bin/python3 -I \
  /opt/vaxreplay/bin/vaxreplay-firecracker-qualification-probe \
  --config /opt/vaxreplay/etc/qualification-guest.json || status=$?
/bin/sync
/sbin/poweroff -f
while true; do /bin/sleep 3600; done
"""


class FirecrackerQualificationGuestDiskBuildError(RuntimeError):
    """The qualification guest disk inputs or deterministic outputs failed closed."""


@dataclass(frozen=True)
class BuiltFirecrackerQualificationGuestDisks:
    rootfs_path: Path
    harness_path: Path
    receipt_path: Path
    manifest_path: Path
    receipt: FirecrackerQualificationGuestDiskBuildReceipt
    manifest: FirecrackerQualificationProbeManifest


def build_firecracker_qualification_guest_disks(
    *,
    task_worker_spec: FirecrackerWorkerSpec,
    manifest_id: str,
    base_rootfs_tree: Path,
    package_tree: Path,
    guest_private_key: Ed25519PrivateKey,
    mke2fs_path: Path,
    expected_mke2fs_sha256: str,
    e2fsck_path: Path,
    expected_e2fsck_sha256: str,
    debugfs_path: Path,
    expected_debugfs_sha256: str,
    output_rootfs_path: Path,
    output_harness_path: Path,
    output_receipt_path: Path,
    output_manifest_path: Path,
    source_date_epoch: int,
    qualification_rootfs_install_path: str,
    qualification_harness_install_path: str,
    rootfs_byte_count: int = _DEFAULT_ROOTFS_BYTES,
    harness_byte_count: int = _DEFAULT_HARNESS_BYTES,
) -> BuiltFirecrackerQualificationGuestDisks:
    """Build and independently reproduce both ext4 images on a qualified Linux host."""

    _require_linux_root()
    if source_date_epoch < 1 or source_date_epoch > 2**31 - 1:
        raise FirecrackerQualificationGuestDiskBuildError('SOURCE_DATE_EPOCH is out of range')
    _validate_image_size(rootfs_byte_count, label='rootfs')
    _validate_image_size(harness_byte_count, label='harness')
    mke2fs = _open_pinned_executable(mke2fs_path, expected_sha256=expected_mke2fs_sha256)
    e2fsck = _open_pinned_executable(e2fsck_path, expected_sha256=expected_e2fsck_sha256)
    debugfs = _open_pinned_executable(debugfs_path, expected_sha256=expected_debugfs_sha256)
    tool_versions = (
        _tool_version(mke2fs),
        _tool_version(e2fsck),
        _tool_version(debugfs),
    )
    base_root = _safe_source_directory(base_rootfs_tree, label='base rootfs')
    package_root = _safe_source_directory(package_tree, label='package tree')
    _validate_base_rootfs_contract(base_root)
    guest_source = Path(__file__).with_name('firecracker_qualification_guest.py').read_bytes()
    guest_source_sha256 = hashlib.sha256(guest_source).hexdigest()
    public_key = ed25519_public_key_bytes(guest_private_key)
    guest_config = FirecrackerQualificationGuestConfig(
        rpc_port=task_worker_spec.guest_rpc_port,
        guest_probe_private_key_hex=guest_private_key.private_bytes_raw().hex(),
        guest_probe_key_id=firecracker_qualification_guest_key_id(public_key),
        guest_probe_executable_sha256=guest_source_sha256,
    )
    config_bytes = canonical_json_bytes(guest_config)
    base_tree_sha256 = _tree_sha256(base_root)
    recipe_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    outputs = (
        output_rootfs_path,
        output_harness_path,
        output_receipt_path,
        output_manifest_path,
    )
    if any(path.expanduser().absolute().exists() or path.expanduser().absolute().is_symlink() for path in outputs):
        raise FirecrackerQualificationGuestDiskBuildError('qualification guest outputs are create-once')
    for output in outputs:
        _validate_output_parent(output)

    try:
        with tempfile.TemporaryDirectory(prefix='vaxreplay-qualification-guest-build.') as temporary:
            temporary_root = Path(temporary)
            rootfs_trees = (temporary_root / 'rootfs-tree-a', temporary_root / 'rootfs-tree-b')
            harness_trees = (temporary_root / 'harness-tree-a', temporary_root / 'harness-tree-b')
            for rootfs_tree in rootfs_trees:
                _copy_tree_normalized(base_root, rootfs_tree, source_date_epoch=source_date_epoch)
                _install_rootfs_init(rootfs_tree, source_date_epoch=source_date_epoch)
            for harness_tree in harness_trees:
                _install_harness(
                    harness_tree,
                    package_root=package_root,
                    guest_source=guest_source,
                    config_bytes=config_bytes,
                    source_date_epoch=source_date_epoch,
                )
            if _tree_sha256(rootfs_trees[0]) != _tree_sha256(rootfs_trees[1]) or _tree_sha256(
                harness_trees[0]
            ) != _tree_sha256(harness_trees[1]):
                raise FirecrackerQualificationGuestDiskBuildError('normalized source trees are not reproducible')
            rootfs_tree_sha256 = _tree_sha256(rootfs_trees[0])
            harness_tree_sha256 = _tree_sha256(harness_trees[0])
            rootfs_a = temporary_root / 'rootfs-a.ext4'
            rootfs_b = temporary_root / 'rootfs-b.ext4'
            harness_a = temporary_root / 'harness-a.ext4'
            harness_b = temporary_root / 'harness-b.ext4'
            for output, tree, byte_count, uuid, label in (
                (rootfs_a, rootfs_trees[0], rootfs_byte_count, _ROOTFS_UUID, _ROOTFS_LABEL),
                (rootfs_b, rootfs_trees[1], rootfs_byte_count, _ROOTFS_UUID, _ROOTFS_LABEL),
                (harness_a, harness_trees[0], harness_byte_count, _HARNESS_UUID, _HARNESS_LABEL),
                (harness_b, harness_trees[1], harness_byte_count, _HARNESS_UUID, _HARNESS_LABEL),
            ):
                _build_ext4(
                    mke2fs_fd=mke2fs,
                    tree=tree,
                    output=output,
                    byte_count=byte_count,
                    uuid=uuid,
                    label=label,
                    source_date_epoch=source_date_epoch,
                )
            _require_identical_rebuild(rootfs_a, rootfs_b, label='qualification rootfs')
            _require_identical_rebuild(harness_a, harness_b, label='qualification harness')
            _inspect_ext4(
                image=rootfs_a,
                e2fsck_fd=e2fsck,
                debugfs_fd=debugfs,
                expected_files={'/sbin/init': (_INIT_BYTES, 0o755)},
                expected_directories={
                    '/opt/vaxreplay': 0o755,
                    '/workspace': 0o755,
                },
                temporary_root=temporary_root,
            )
            package_init = (package_root / '__init__.py').read_bytes()
            _inspect_ext4(
                image=harness_a,
                e2fsck_fd=e2fsck,
                debugfs_fd=debugfs,
                expected_files={
                    '/bin/vaxreplay-firecracker-qualification-probe': (guest_source, 0o555),
                    '/etc/qualification-guest.json': (config_bytes, 0o400),
                    '/lib/vaxreplay/__init__.py': (
                        package_init,
                        stat.S_IMODE((package_root / '__init__.py').stat().st_mode),
                    ),
                },
                expected_directories={},
                temporary_root=temporary_root,
            )
            rootfs_sha256 = _file_sha256(rootfs_a)
            harness_sha256 = _file_sha256(harness_a)
            build_contract = _build_contract_bytes(source_date_epoch=source_date_epoch)
            receipt = FirecrackerQualificationGuestDiskBuildReceipt(
                source_date_epoch=source_date_epoch,
                base_rootfs_tree_sha256=base_tree_sha256,
                package_tree_sha256=_tree_sha256(package_root),
                normalized_rootfs_tree_sha256=rootfs_tree_sha256,
                normalized_harness_tree_sha256=harness_tree_sha256,
                build_recipe_sha256=recipe_sha256,
                mke2fs_sha256=expected_mke2fs_sha256,
                mke2fs_version=tool_versions[0],
                e2fsck_sha256=expected_e2fsck_sha256,
                e2fsck_version=tool_versions[1],
                debugfs_sha256=expected_debugfs_sha256,
                debugfs_version=tool_versions[2],
                build_argv_and_env_sha256=hashlib.sha256(build_contract).hexdigest(),
                init_sha256=hashlib.sha256(_INIT_BYTES).hexdigest(),
                guest_probe_executable_sha256=guest_source_sha256,
                guest_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                rootfs_sha256=rootfs_sha256,
                rootfs_byte_count=rootfs_byte_count,
                harness_sha256=harness_sha256,
                harness_byte_count=harness_byte_count,
            )
            manifest = FirecrackerQualificationProbeManifest(
                manifest_id=manifest_id,
                task_worker_spec_sha256=firecracker_model_sha256(task_worker_spec),
                task_rootfs_sha256=task_worker_spec.images.rootfs.sha256,
                task_harness_sha256=task_worker_spec.images.harness.sha256,
                qualification_kernel_sha256=task_worker_spec.images.kernel.sha256,
                qualification_rootfs_path=qualification_rootfs_install_path,
                qualification_rootfs_sha256=rootfs_sha256,
                qualification_rootfs_byte_count=rootfs_byte_count,
                qualification_harness_path=qualification_harness_install_path,
                qualification_harness_sha256=harness_sha256,
                qualification_harness_byte_count=harness_byte_count,
                qualification_disk_build_receipt=receipt,
                qualification_disk_build_receipt_sha256=firecracker_model_sha256(receipt),
                guest_probe_executable_sha256=guest_source_sha256,
                guest_probe_public_key_hex=public_key.hex(),
                guest_probe_key_id=firecracker_qualification_guest_key_id(public_key),
            )
            _publish_new_file(output_rootfs_path, rootfs_a, mode=0o600)
            _publish_new_file(output_harness_path, harness_a, mode=0o600)
            _publish_new_bytes(output_receipt_path, canonical_json_bytes(receipt), mode=0o600)
            _publish_new_bytes(output_manifest_path, canonical_json_bytes(manifest), mode=0o600)
    finally:
        os.close(mke2fs)
        os.close(e2fsck)
        os.close(debugfs)
    return BuiltFirecrackerQualificationGuestDisks(
        rootfs_path=output_rootfs_path.absolute(),
        harness_path=output_harness_path.absolute(),
        receipt_path=output_receipt_path.absolute(),
        manifest_path=output_manifest_path.absolute(),
        receipt=receipt,
        manifest=manifest,
    )


def _require_linux_root() -> None:
    if platform.system() != 'Linux' or os.geteuid() != 0 or not Path('/proc/self/fd').is_dir():
        raise FirecrackerQualificationGuestDiskBuildError(
            'qualification guest disks require root on Linux with procfs mounted'
        )


def _validate_image_size(value: int, *, label: str) -> None:
    if value < 64 * 1024 * 1024 or value > 16 * 1024 * 1024 * 1024 or value % 4096:
        raise FirecrackerQualificationGuestDiskBuildError(
            f'{label} size must be a 4 KiB multiple from 64 MiB to 16 GiB'
        )


def _open_pinned_executable(path: Path, *, expected_sha256: str) -> int:
    _require_sha256(expected_sha256, label='mke2fs')
    try:
        descriptor = os.open(path.expanduser().absolute(), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise FirecrackerQualificationGuestDiskBuildError('digest-pinned mke2fs is unavailable') from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not stat.S_IMODE(metadata.st_mode) & 0o111
        or _descriptor_sha256(descriptor) != expected_sha256
    ):
        os.close(descriptor)
        raise FirecrackerQualificationGuestDiskBuildError('mke2fs identity differs from its root-owned release pin')
    return descriptor


def _build_ext4(
    *,
    mke2fs_fd: int,
    tree: Path,
    output: Path,
    byte_count: int,
    uuid: str,
    label: str,
    source_date_epoch: int,
) -> None:
    with output.open('xb') as stream:
        stream.truncate(byte_count)
    command = (
        f'/proc/self/fd/{mke2fs_fd}',
        '-q',
        '-t',
        'ext4',
        '-F',
        '-b',
        '4096',
        '-U',
        uuid,
        '-L',
        label,
        '-O',
        '^metadata_csum_seed,^orphan_file',
        '-E',
        f'lazy_itable_init=0,lazy_journal_init=0,root_owner=0:0,hash_seed={uuid}',
        '-d',
        str(tree),
        str(output),
    )
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd='/',
            env={
                'E2FSPROGS_FAKE_TIME': str(source_date_epoch),
                'SOURCE_DATE_EPOCH': str(source_date_epoch),
                'LANG': 'C',
                'LC_ALL': 'C',
                'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
            },
            pass_fds=(mke2fs_fd,),
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FirecrackerQualificationGuestDiskBuildError('mke2fs execution failed') from error
    if completed.returncode != 0 or len(completed.stderr) > 64 * 1024:
        raise FirecrackerQualificationGuestDiskBuildError('mke2fs rejected the fixed qualification image recipe')


def _install_rootfs_init(root: Path, *, source_date_epoch: int) -> None:
    # The root disk is mounted read-only.  These directories therefore have to
    # exist in the image before boot; PID 1 cannot create them at runtime.
    for mountpoint in (root / 'opt' / 'vaxreplay', root / 'workspace'):
        if mountpoint.is_symlink() or (mountpoint.exists() and not mountpoint.is_dir()):
            raise FirecrackerQualificationGuestDiskBuildError('qualification mountpoints must be real directories')
        if mountpoint.exists():
            shutil.rmtree(mountpoint)
        mountpoint.mkdir(parents=True, mode=0o755)
        mountpoint.chmod(0o755)
        os.chown(mountpoint, 0, 0)
        os.utime(mountpoint, (source_date_epoch, source_date_epoch), follow_symlinks=False)
    opt = root / 'opt'
    opt.chmod(0o755)
    os.chown(opt, 0, 0)
    os.utime(opt, (source_date_epoch, source_date_epoch), follow_symlinks=False)

    target = root / 'sbin' / 'init'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.write_bytes(_INIT_BYTES)
    target.chmod(0o755)
    os.chown(target, 0, 0)
    os.utime(target, (source_date_epoch, source_date_epoch), follow_symlinks=False)


def _install_harness(
    root: Path,
    *,
    package_root: Path,
    guest_source: bytes,
    config_bytes: bytes,
    source_date_epoch: int,
) -> None:
    executable = root.joinpath(*PurePosixPath(QUALIFICATION_GUEST_EXECUTABLE_PATH).relative_to('/opt/vaxreplay').parts)
    config = root.joinpath(*PurePosixPath(QUALIFICATION_GUEST_CONFIG_PATH).relative_to('/opt/vaxreplay').parts)
    library = root / 'lib' / 'vaxreplay'
    executable.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    _copy_package_tree(package_root, library)
    executable.write_bytes(guest_source)
    executable.chmod(0o555)
    config.write_bytes(config_bytes)
    config.chmod(0o400)
    _normalize_tree(root, source_date_epoch=source_date_epoch)


def _copy_package_tree(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name == '__pycache__' or name.endswith(('.pyc', '.pyo'))}

    shutil.copytree(source, destination, symlinks=False, ignore=ignore)


def _copy_tree_normalized(source: Path, destination: Path, *, source_date_epoch: int) -> None:
    shutil.copytree(source, destination, symlinks=True)
    _normalize_tree(destination, source_date_epoch=source_date_epoch)


def _normalize_tree(root: Path, *, source_date_epoch: int) -> None:
    entries = [root, *sorted(root.rglob('*'), key=lambda item: item.as_posix())]
    for entry in entries:
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            os.lchown(entry, 0, 0)
            os.utime(entry, (source_date_epoch, source_date_epoch), follow_symlinks=False)
            continue
        if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise FirecrackerQualificationGuestDiskBuildError(
                'guest source trees may contain only files, directories, and symlinks'
            )
        os.chown(entry, 0, 0)
        os.utime(entry, (source_date_epoch, source_date_epoch), follow_symlinks=False)


def _validate_base_rootfs_contract(root: Path) -> None:
    required = (
        ('bin/sh', True),
        ('bin/mount', True),
        ('bin/mkdir', True),
        ('bin/sync', True),
        ('bin/sleep', True),
        ('sbin/poweroff', True),
        ('usr/bin/python3', True),
    )
    for relative, executable in required:
        try:
            path = _resolve_guest_tree_path(root, PurePosixPath('/') / relative)
            metadata = path.lstat()
        except OSError as error:
            raise FirecrackerQualificationGuestDiskBuildError(f'base rootfs lacks required {relative}') from error
        if not stat.S_ISREG(metadata.st_mode) or (executable and not stat.S_IMODE(metadata.st_mode) & 0o111):
            raise FirecrackerQualificationGuestDiskBuildError(
                f'base rootfs required path is not executable: {relative}'
            )


def _resolve_guest_tree_path(root: Path, guest_path: PurePosixPath) -> Path:
    """Resolve symlinks as the guest would, never against the host's ``/`` tree."""

    pending = [part for part in guest_path.parts if part not in {'', '/', '.'}]
    resolved: list[str] = []
    symlink_count = 0
    while pending:
        part = pending.pop(0)
        if part == '..':
            if not resolved:
                raise FirecrackerQualificationGuestDiskBuildError('base rootfs required path escapes guest root')
            resolved.pop()
            continue
        candidate = root.joinpath(*resolved, part)
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            symlink_count += 1
            if symlink_count > 40:
                raise FirecrackerQualificationGuestDiskBuildError('base rootfs required path has a symlink cycle')
            target = PurePosixPath(os.readlink(candidate))
            if target.is_absolute():
                resolved.clear()
            pending = [item for item in target.parts if item not in {'', '/', '.'}] + pending
        else:
            resolved.append(part)
    return root.joinpath(*resolved)


def _safe_source_directory(path: Path, *, label: str) -> Path:
    absolute = path.expanduser().absolute()
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise FirecrackerQualificationGuestDiskBuildError(f'{label} is unavailable') from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FirecrackerQualificationGuestDiskBuildError(f'{label} must be a non-symlink directory')
    _validate_symlinks_within(resolved, label=label)
    return resolved


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in [root, *sorted(root.rglob('*'), key=lambda item: item.as_posix())]:
        relative = path.relative_to(root).as_posix() or '.'
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            kind = b'l'
            content = os.readlink(path).encode('utf-8')
        elif stat.S_ISDIR(metadata.st_mode):
            kind = b'd'
            content = b''
        elif stat.S_ISREG(metadata.st_mode):
            kind = b'f'
            content = path.read_bytes()
        else:
            raise FirecrackerQualificationGuestDiskBuildError('source tree contains an unsupported special file')
        digest.update(kind + b'\0' + relative.encode('utf-8') + b'\0')
        digest.update(str(stat.S_IMODE(metadata.st_mode)).encode('ascii') + b'\0')
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _require_identical_rebuild(first: Path, second: Path, *, label: str) -> None:
    if first.stat().st_size != second.stat().st_size or not hmac.compare_digest(
        _file_sha256(first), _file_sha256(second)
    ):
        raise FirecrackerQualificationGuestDiskBuildError(f'{label} was not byte-identical across two clean builds')


def _validate_symlinks_within(root: Path, *, label: str) -> None:
    for path in sorted(root.rglob('*'), key=lambda item: item.as_posix()):
        if not path.is_symlink():
            continue
        target = PurePosixPath(os.readlink(path))
        parts: list[str] = [] if target.is_absolute() else list(path.relative_to(root).parent.parts)
        for part in target.parts:
            if part in {'', '/', '.'}:
                continue
            if part == '..':
                if not parts:
                    raise FirecrackerQualificationGuestDiskBuildError(f'{label} contains an escaping symlink')
                parts.pop()
            else:
                parts.append(part)


def _validate_output_parent(path: Path) -> None:
    parent = path.expanduser().absolute().parent
    try:
        metadata = parent.lstat()
    except OSError as error:
        raise FirecrackerQualificationGuestDiskBuildError('output parent must already exist') from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise FirecrackerQualificationGuestDiskBuildError(
            'output parent must be a root-owned non-writable real directory'
        )


def _publish_new_file(path: Path, source: Path, *, mode: int) -> None:
    target = path.expanduser().absolute()
    parent_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    descriptor = os.open(
        target.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        mode,
        dir_fd=parent_descriptor,
    )
    try:
        with source.open('rb') as input_stream, os.fdopen(descriptor, 'wb', closefd=False) as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.fsync(parent_descriptor)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _publish_new_bytes(path: Path, content: bytes, *, mode: int) -> None:
    with tempfile.NamedTemporaryFile(prefix='.qualification-metadata.', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        _publish_new_file(path, temporary, mode=mode)
    finally:
        temporary.unlink(missing_ok=True)


def _tool_version(descriptor: int) -> str:
    try:
        completed = subprocess.run(
            (f'/proc/self/fd/{descriptor}', '-V'),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd='/',
            env={'LANG': 'C', 'LC_ALL': 'C', 'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'},
            pass_fds=(descriptor,),
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FirecrackerQualificationGuestDiskBuildError('ext4 tool version probe failed') from error
    output = (completed.stdout + completed.stderr).decode('ascii', errors='strict').strip().replace('\n', ' ')
    if completed.returncode != 0 or not output or len(output) > 200:
        raise FirecrackerQualificationGuestDiskBuildError('ext4 tool version is unavailable or oversized')
    return output


def _inspect_ext4(
    *,
    image: Path,
    e2fsck_fd: int,
    debugfs_fd: int,
    expected_files: dict[str, tuple[bytes, int]],
    expected_directories: dict[str, int],
    temporary_root: Path,
) -> None:
    checked = _run_ext4_tool(e2fsck_fd, ('-fn', str(image)), accepted_codes={0})
    if len(checked.stdout) + len(checked.stderr) > 1024 * 1024:
        raise FirecrackerQualificationGuestDiskBuildError('e2fsck output is oversized')
    for guest_path, expected_mode in sorted(expected_directories.items()):
        stat_result = _run_ext4_tool(
            debugfs_fd,
            ('-R', f'stat {guest_path}', str(image)),
            accepted_codes={0},
        )
        stat_text = stat_result.stdout.decode('ascii', errors='strict')
        mode_match = re.search(r'Mode:\s+0*([0-7]{3,4})\b', stat_text)
        if (
            re.search(r'Type:\s+directory\b', stat_text) is None
            or mode_match is None
            or int(mode_match.group(1), 8) != expected_mode
        ):
            raise FirecrackerQualificationGuestDiskBuildError('debugfs observed an unexpected guest directory')
    for index, (guest_path, (expected_bytes, expected_mode)) in enumerate(sorted(expected_files.items())):
        stat_result = _run_ext4_tool(
            debugfs_fd,
            ('-R', f'stat {guest_path}', str(image)),
            accepted_codes={0},
        )
        stat_text = stat_result.stdout.decode('ascii', errors='strict')
        match = re.search(r'Mode:\s+0*([0-7]{3,4})\b', stat_text)
        if match is None or int(match.group(1), 8) != expected_mode:
            raise FirecrackerQualificationGuestDiskBuildError('debugfs observed an unexpected guest file mode')
        dumped = temporary_root / f'inspected-{image.name}-{index}'
        _run_ext4_tool(
            debugfs_fd,
            ('-R', f'dump {guest_path} {dumped}', str(image)),
            accepted_codes={0},
        )
        try:
            observed_bytes = dumped.read_bytes()
        except OSError as error:
            raise FirecrackerQualificationGuestDiskBuildError(
                'debugfs did not extract an expected guest file'
            ) from error
        if not hmac.compare_digest(hashlib.sha256(observed_bytes).digest(), hashlib.sha256(expected_bytes).digest()):
            raise FirecrackerQualificationGuestDiskBuildError('debugfs observed unexpected guest file bytes')


def _run_ext4_tool(
    descriptor: int, arguments: tuple[str, ...], *, accepted_codes: set[int]
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            (f'/proc/self/fd/{descriptor}', *arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd='/',
            env={'LANG': 'C', 'LC_ALL': 'C', 'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'},
            pass_fds=(descriptor,),
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FirecrackerQualificationGuestDiskBuildError('pinned ext4 inspection tool failed') from error
    if completed.returncode not in accepted_codes:
        raise FirecrackerQualificationGuestDiskBuildError('pinned ext4 inspection rejected the completed image')
    return completed


def _build_contract_bytes(*, source_date_epoch: int) -> bytes:
    return canonical_json_bytes(
        {
            'mke2fs_argv': [
                '$MKE2FS',
                '-q',
                '-t',
                'ext4',
                '-F',
                '-b',
                '4096',
                '-U',
                '$UUID',
                '-L',
                '$LABEL',
                '-O',
                '^metadata_csum_seed,^orphan_file',
                '-E',
                'lazy_itable_init=0,lazy_journal_init=0,root_owner=0:0,hash_seed=$UUID',
                '-d',
                '$TREE',
                '$OUTPUT',
            ],
            'environment': {
                'E2FSPROGS_FAKE_TIME': str(source_date_epoch),
                'SOURCE_DATE_EPOCH': str(source_date_epoch),
                'LANG': 'C',
                'LC_ALL': 'C',
                'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
            },
            'inspection': ['e2fsck -fn $IMAGE', 'debugfs stat+dump fixed paths'],
        }
    )


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or not set(value) <= _SHA256_PATTERN:
        raise FirecrackerQualificationGuestDiskBuildError(f'{label} pin must be a lowercase SHA-256 digest')


__all__ = [
    'BuiltFirecrackerQualificationGuestDisks',
    'FirecrackerQualificationGuestDiskBuildError',
    'build_firecracker_qualification_guest_disks',
]
