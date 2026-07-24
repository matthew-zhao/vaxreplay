"""Linux-only CLI for the qualification guest's reproducible disk profile."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.agentic.firecracker import FirecrackerWorkerSpec, firecracker_model_sha256
from vaxreplay.agentic.firecracker_qualification_guest_disk_build import (
    FirecrackerQualificationGuestDiskBuildError,
    build_firecracker_qualification_guest_disks,
)
from vaxreplay.bundle import canonical_json_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='build the separate reproducible Firecracker qualification guest disks'
    )
    parser.add_argument('--task-worker-spec', required=True, type=Path)
    parser.add_argument('--expected-task-worker-spec-sha256', required=True)
    parser.add_argument('--manifest-id', required=True)
    parser.add_argument('--base-rootfs-tree', required=True, type=Path)
    parser.add_argument('--package-tree', required=True, type=Path)
    parser.add_argument('--guest-private-key-file', required=True, type=Path)
    for tool in ('mke2fs', 'e2fsck', 'debugfs'):
        parser.add_argument(f'--{tool}', required=True, type=Path)
        parser.add_argument(f'--expected-{tool}-sha256', required=True)
    parser.add_argument('--output-rootfs', required=True, type=Path)
    parser.add_argument('--output-harness', required=True, type=Path)
    parser.add_argument('--output-receipt', required=True, type=Path)
    parser.add_argument('--output-manifest', required=True, type=Path)
    parser.add_argument('--source-date-epoch', required=True, type=int)
    parser.add_argument('--qualification-rootfs-install-path', required=True)
    parser.add_argument('--qualification-harness-install-path', required=True)
    parser.add_argument('--rootfs-bytes', type=int, default=512 * 1024 * 1024)
    parser.add_argument('--harness-bytes', type=int, default=256 * 1024 * 1024)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    try:
        task_spec = _load_task_spec(
            arguments.task_worker_spec,
            expected_sha256=arguments.expected_task_worker_spec_sha256,
        )
        built = build_firecracker_qualification_guest_disks(
            task_worker_spec=task_spec,
            manifest_id=arguments.manifest_id,
            base_rootfs_tree=arguments.base_rootfs_tree,
            package_tree=arguments.package_tree,
            guest_private_key=_load_guest_key(arguments.guest_private_key_file),
            mke2fs_path=arguments.mke2fs,
            expected_mke2fs_sha256=arguments.expected_mke2fs_sha256,
            e2fsck_path=arguments.e2fsck,
            expected_e2fsck_sha256=arguments.expected_e2fsck_sha256,
            debugfs_path=arguments.debugfs,
            expected_debugfs_sha256=arguments.expected_debugfs_sha256,
            output_rootfs_path=arguments.output_rootfs,
            output_harness_path=arguments.output_harness,
            output_receipt_path=arguments.output_receipt,
            output_manifest_path=arguments.output_manifest,
            source_date_epoch=arguments.source_date_epoch,
            qualification_rootfs_install_path=arguments.qualification_rootfs_install_path,
            qualification_harness_install_path=arguments.qualification_harness_install_path,
            rootfs_byte_count=arguments.rootfs_bytes,
            harness_byte_count=arguments.harness_bytes,
        )
        sys.stdout.buffer.write(
            canonical_json_bytes(
                {
                    'rootfs_path': str(built.rootfs_path),
                    'rootfs_sha256': built.receipt.rootfs_sha256,
                    'harness_path': str(built.harness_path),
                    'harness_sha256': built.receipt.harness_sha256,
                    'receipt_path': str(built.receipt_path),
                    'receipt_sha256': firecracker_model_sha256(built.receipt),
                    'manifest_path': str(built.manifest_path),
                    'manifest_sha256': firecracker_model_sha256(built.manifest),
                    'qualification_guest_separate_from_task_guest': True,
                }
            )
        )
    except (FirecrackerQualificationGuestDiskBuildError, ValueError) as error:
        sys.stderr.write(f'qualification guest disk build rejected: {error}\n')
        raise SystemExit(64) from error


def _load_task_spec(path: Path, *, expected_sha256: str) -> FirecrackerWorkerSpec:
    content = _read_private_file(path, maximum=8 * 1024 * 1024)
    try:
        spec = FirecrackerWorkerSpec.model_validate_json(content)
    except ValueError as error:
        raise FirecrackerQualificationGuestDiskBuildError('task worker spec is invalid') from error
    if canonical_json_bytes(spec) != content or firecracker_model_sha256(spec) != expected_sha256:
        raise FirecrackerQualificationGuestDiskBuildError('task worker spec differs from its canonical release pin')
    return spec


def _load_guest_key(path: Path) -> Ed25519PrivateKey:
    content = _read_private_file(path, maximum=65).strip()
    try:
        private_bytes = bytes.fromhex(content.decode('ascii'))
    except (UnicodeDecodeError, ValueError) as error:
        raise FirecrackerQualificationGuestDiskBuildError('guest key must be 32 bytes of lowercase hex') from error
    if len(private_bytes) != 32 or content.decode('ascii') != private_bytes.hex():
        raise FirecrackerQualificationGuestDiskBuildError('guest key must be 32 bytes of lowercase hex')
    return Ed25519PrivateKey.from_private_bytes(private_bytes)


def _read_private_file(path: Path, *, maximum: int) -> bytes:
    try:
        descriptor = os.open(path.expanduser().absolute(), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise FirecrackerQualificationGuestDiskBuildError('private build input is unavailable') from error
    try:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > maximum
        ):
            raise FirecrackerQualificationGuestDiskBuildError('private build input has unsafe metadata')
        content = bytearray()
        while chunk := os.read(descriptor, min(64 * 1024, maximum + 1 - len(content))):
            content.extend(chunk)
            if len(content) > maximum:
                raise FirecrackerQualificationGuestDiskBuildError('private build input is oversized')
        return bytes(content)
    finally:
        os.close(descriptor)


__all__ = ['main']
