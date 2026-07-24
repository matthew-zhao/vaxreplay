"""Root-only production CLI for Lane A task guest disk builds and offline verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vaxreplay.agentic.guest_boot_dispatch import (
    GuestBootDispatchError,
    load_and_validate_guest_boot_config,
    load_pinned_guest_boot_dispatch_manifest,
)
from vaxreplay.agentic.guest_disk_build import (
    DEFAULT_LANE_A_HARNESS_BYTES,
    DEFAULT_LANE_A_ROOTFS_BYTES,
    LaneAGuestDiskBuildError,
    PinnedLinuxExt4Executor,
    build_lane_a_guest_disks,
    lane_a_guest_disk_build_receipt_sha256,
    load_lane_a_guest_disk_build_receipt,
    load_pinned_guest_disk_tool_runtime_closure_manifest,
    load_pinned_lane_a_guest_config_bytes,
    verify_lane_a_guest_disk_build,
    verify_lane_a_guest_disk_build_parity,
)
from vaxreplay.bundle import canonical_json_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='vaxreplay-lane-a-disk-build')
    commands = parser.add_subparsers(dest='command', required=True)

    build = commands.add_parser('build', help='build, reproduce, inspect, and publish both ext4 images')
    build.add_argument('--base-rootfs-tar', type=Path, required=True)
    build.add_argument('--base-rootfs-sha256', required=True)
    build.add_argument('--harness-payload-tar', type=Path, required=True)
    build.add_argument('--harness-payload-sha256', required=True)
    build.add_argument('--guest-executable-sha256', required=True)
    build.add_argument('--guest-config', type=Path, required=True)
    build.add_argument('--guest-config-sha256', required=True)
    build.add_argument('--guest-boot-dispatch-manifest', type=Path)
    build.add_argument('--guest-boot-dispatch-manifest-sha256')
    build.add_argument('--tool-runtime-closure-manifest', type=Path, required=True)
    build.add_argument('--tool-runtime-closure-manifest-sha256', required=True)
    _add_tool_arguments(build)
    build.add_argument('--output-rootfs', type=Path, required=True)
    build.add_argument('--output-harness', type=Path, required=True)
    build.add_argument('--output-receipt', type=Path, required=True)
    build.add_argument('--source-date-epoch', type=int, required=True)
    build.add_argument('--rootfs-bytes', type=int, default=DEFAULT_LANE_A_ROOTFS_BYTES)
    build.add_argument('--harness-bytes', type=int, default=DEFAULT_LANE_A_HARNESS_BYTES)

    verify = commands.add_parser('verify', help='offline-check one receipt and the exact published image bytes')
    verify.add_argument('--receipt', type=Path, required=True)
    verify.add_argument('--receipt-sha256', required=True)
    verify.add_argument('--rootfs', type=Path, required=True)
    verify.add_argument('--harness', type=Path, required=True)
    verify.add_argument('--base-rootfs-sha256', required=True)
    verify.add_argument('--harness-payload-sha256', required=True)
    verify.add_argument('--guest-executable-sha256', required=True)
    verify.add_argument('--guest-config-sha256', required=True)
    verify.add_argument('--guest-boot-dispatch-manifest-sha256')
    verify.add_argument('--mke2fs-sha256', required=True)
    verify.add_argument('--e2fsck-sha256', required=True)
    verify.add_argument('--debugfs-sha256', required=True)
    verify.add_argument('--tool-runtime-closure-manifest-sha256', required=True)
    verify.add_argument('--builder-source-sha256')

    parity = commands.add_parser('parity', help='compare canonical receipts from independent builders')
    parity.add_argument('--first-receipt', type=Path, required=True)
    parity.add_argument('--first-receipt-sha256', required=True)
    parity.add_argument('--second-receipt', type=Path, required=True)
    parity.add_argument('--second-receipt-sha256', required=True)
    return parser


def _add_tool_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ('mke2fs', 'e2fsck', 'debugfs'):
        parser.add_argument(f'--{name}', type=Path, required=True)
        parser.add_argument(f'--{name}-sha256', required=True)


def _run_build(arguments: argparse.Namespace) -> dict[str, object]:
    dispatch_path = arguments.guest_boot_dispatch_manifest
    dispatch_sha256 = arguments.guest_boot_dispatch_manifest_sha256
    if (dispatch_path is None) != (dispatch_sha256 is None):
        raise LaneAGuestDiskBuildError('guest boot-dispatch manifest path and external digest pin are inseparable')
    if dispatch_path is None:
        dispatch = None
        config_bytes = load_pinned_lane_a_guest_config_bytes(
            arguments.guest_config,
            expected_sha256=arguments.guest_config_sha256,
        )
    else:
        dispatch = load_pinned_guest_boot_dispatch_manifest(
            dispatch_path,
            expected_sha256=dispatch_sha256,
        )
        config_bytes = load_and_validate_guest_boot_config(
            arguments.guest_config,
            dispatch=dispatch,
        )
        if dispatch.guest_config_sha256 != arguments.guest_config_sha256:
            raise LaneAGuestDiskBuildError('guest config CLI pin differs from the boot-dispatch manifest')
    tool_runtime_closure_manifest = load_pinned_guest_disk_tool_runtime_closure_manifest(
        arguments.tool_runtime_closure_manifest,
        expected_sha256=arguments.tool_runtime_closure_manifest_sha256,
    )
    with PinnedLinuxExt4Executor(
        mke2fs_path=arguments.mke2fs,
        expected_mke2fs_sha256=arguments.mke2fs_sha256,
        e2fsck_path=arguments.e2fsck,
        expected_e2fsck_sha256=arguments.e2fsck_sha256,
        debugfs_path=arguments.debugfs,
        expected_debugfs_sha256=arguments.debugfs_sha256,
    ) as executor:
        build = build_lane_a_guest_disks(
            base_rootfs_source=arguments.base_rootfs_tar,
            expected_base_rootfs_source_sha256=arguments.base_rootfs_sha256,
            harness_payload_source=arguments.harness_payload_tar,
            expected_harness_payload_source_sha256=arguments.harness_payload_sha256,
            expected_guest_executable_sha256=arguments.guest_executable_sha256,
            guest_config_bytes=config_bytes,
            guest_boot_dispatch=dispatch,
            expected_guest_boot_dispatch_manifest_sha256=dispatch_sha256,
            executor=executor,
            tool_runtime_closure_manifest=tool_runtime_closure_manifest,
            expected_tool_runtime_closure_manifest_sha256=(arguments.tool_runtime_closure_manifest_sha256),
            output_rootfs_path=arguments.output_rootfs,
            output_harness_path=arguments.output_harness,
            output_receipt_path=arguments.output_receipt,
            source_date_epoch=arguments.source_date_epoch,
            rootfs_byte_count=arguments.rootfs_bytes,
            harness_byte_count=arguments.harness_bytes,
        )
    return {
        'schema_version': 'vaxreplay.lane-a-guest-disk-build-cli-result.v0.3',
        'receipt_sha256': build.receipt_sha256,
        'rootfs_sha256': build.receipt.rootfs.sha256,
        'harness_sha256': build.receipt.harness.sha256,
        'reproducibility_scope': build.receipt.reproducibility_scope,
        'cross_host_hermetic_reproducibility_claimed': (build.receipt.cross_host_hermetic_reproducibility_claimed),
        'production_eligible': build.receipt.production_eligible,
        'guest_boot_dispatch_manifest_sha256': (build.receipt.guest_boot_dispatch_manifest_sha256),
        'canonical_operator_runtime_supported': (build.receipt.canonical_operator_runtime_supported),
    }


def _run_verify(arguments: argparse.Namespace) -> dict[str, object]:
    receipt = load_lane_a_guest_disk_build_receipt(
        arguments.receipt,
        expected_receipt_sha256=arguments.receipt_sha256,
    )
    verified = verify_lane_a_guest_disk_build(
        receipt=receipt,
        rootfs_path=arguments.rootfs,
        harness_path=arguments.harness,
        expected_base_rootfs_source_sha256=arguments.base_rootfs_sha256,
        expected_harness_payload_source_sha256=arguments.harness_payload_sha256,
        expected_guest_executable_sha256=arguments.guest_executable_sha256,
        expected_guest_config_sha256=arguments.guest_config_sha256,
        expected_guest_boot_dispatch_manifest_sha256=(arguments.guest_boot_dispatch_manifest_sha256),
        expected_mke2fs_sha256=arguments.mke2fs_sha256,
        expected_e2fsck_sha256=arguments.e2fsck_sha256,
        expected_debugfs_sha256=arguments.debugfs_sha256,
        expected_tool_runtime_closure_manifest_sha256=(arguments.tool_runtime_closure_manifest_sha256),
        expected_builder_source_sha256=arguments.builder_source_sha256,
    )
    return {
        'schema_version': 'vaxreplay.lane-a-guest-disk-verify-cli-result.v0.3',
        'receipt_sha256': verified.receipt_sha256,
        'rootfs_sha256': verified.receipt.rootfs.sha256,
        'harness_sha256': verified.receipt.harness.sha256,
        'reproducibility_scope': verified.receipt.reproducibility_scope,
        'verified': True,
        'guest_boot_dispatch_manifest_sha256': (verified.receipt.guest_boot_dispatch_manifest_sha256),
    }


def _run_parity(arguments: argparse.Namespace) -> dict[str, object]:
    first = load_lane_a_guest_disk_build_receipt(
        arguments.first_receipt,
        expected_receipt_sha256=arguments.first_receipt_sha256,
    )
    second = load_lane_a_guest_disk_build_receipt(
        arguments.second_receipt,
        expected_receipt_sha256=arguments.second_receipt_sha256,
    )
    receipt = verify_lane_a_guest_disk_build_parity(first, second)
    return {
        'schema_version': 'vaxreplay.lane-a-guest-disk-parity-cli-result.v0.1',
        'receipt_sha256': lane_a_guest_disk_build_receipt_sha256(receipt),
        'parity_verified': True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'build':
            result = _run_build(arguments)
        elif arguments.command == 'verify':
            result = _run_verify(arguments)
        else:
            result = _run_parity(arguments)
    except (GuestBootDispatchError, LaneAGuestDiskBuildError, OSError, ValueError) as error:
        print(f'lane-a guest disk operation rejected: {error}', file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode('utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
