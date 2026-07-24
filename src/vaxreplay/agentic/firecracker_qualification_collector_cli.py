"""CLI for raw Firecracker qualification collection and independent verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vaxreplay.agentic.firecracker_qualification import (
    FirecrackerQualificationError,
    load_pinned_firecracker_worker_spec,
)
from vaxreplay.agentic.firecracker_qualification_collector import (
    FirecrackerQualificationCollectorError,
    PinnedLinuxKvmQualificationDriver,
    collect_and_retain_firecracker_qualification_evidence,
    independently_verify_firecracker_qualification_collector_evidence,
    load_firecracker_qualification_collector_evidence,
    load_firecracker_qualification_collector_plan,
    load_pinned_firecracker_qualification_probe_manifest,
    read_firecracker_live_collector_private_key_fd,
    read_firecracker_live_collector_private_key_file,
    retain_firecracker_qualification_collector_plan,
)
from vaxreplay.agentic.firecracker_qualification_probe import (
    FirecrackerQualificationCollectionMode,
    ed25519_public_key_bytes,
    firecracker_live_collector_key_id,
)
from vaxreplay.agentic.firecracker_qualification_runtime_closure import (
    QualificationDriverRuntimeClosureError,
    build_and_retain_qualification_driver_runtime_closure,
    verify_qualification_driver_runtime_closure,
)
from vaxreplay.bundle import canonical_json_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Retain plans, collect signed raw Linux/KVM evidence, or independently verify all seven drills',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    plan = subparsers.add_parser(
        'plan',
        help='retain the exact missing-probe inventory (always exits 2 after successful retention)',
    )
    plan.add_argument('--worker-spec', required=True, type=Path)
    plan.add_argument('--expected-worker-spec-sha256', required=True)
    plan.add_argument('--output', required=True, type=Path)
    plan.add_argument('--plan-id')

    verify = subparsers.add_parser('verify-plan', help='verify an exact retained non-qualifying plan')
    verify.add_argument('--artifact', required=True, type=Path)
    verify.add_argument('--expected-plan-sha256', required=True)
    verify.add_argument('--expected-worker-spec-sha256', required=True)

    closure_build = subparsers.add_parser(
        'build-driver-runtime-closure',
        help='inventory and create-once publish the installed qualification-driver runtime',
    )
    closure_build.add_argument('--closure-id', required=True)
    closure_build.add_argument('--driver', required=True, type=Path)
    closure_build.add_argument('--interpreter', required=True, type=Path)
    closure_build.add_argument('--runtime-root', required=True, action='append', type=Path)
    closure_build.add_argument('--source-date-epoch', required=True, type=int)
    closure_build.add_argument('--output', required=True, type=Path)

    closure_verify = subparsers.add_parser(
        'verify-driver-runtime-closure',
        help='offline-verify an exact installed qualification-driver runtime closure',
    )
    _add_runtime_closure_inputs(closure_verify)

    collect = subparsers.add_parser(
        'collect-live',
        help='run the exact pinned Linux/KVM driver for all seven drills and retain signed raw evidence',
    )
    collect.add_argument('--worker-spec', required=True, type=Path)
    collect.add_argument('--expected-worker-spec-sha256', required=True)
    collect.add_argument('--probe-manifest', required=True, type=Path)
    collect.add_argument('--expected-probe-manifest-sha256', required=True)
    collect.add_argument('--driver', required=True, type=Path)
    collect.add_argument('--driver-id', required=True)
    collect.add_argument('--expected-driver-sha256', required=True)
    _add_runtime_closure_inputs(collect)
    collect.add_argument('--expected-collector-source-sha256', required=True)
    collect.add_argument('--expected-collector-key-id', required=True)
    collect.add_argument('--output', required=True, type=Path)
    _add_collector_key_source(collect)

    authenticate = subparsers.add_parser(
        'verify-authentication',
        help='verify exact pins and collector signature without qualifying simulated evidence',
    )
    _add_evidence_inputs(authenticate, require_runtime_closure=False)

    qualify = subparsers.add_parser(
        'verify-live',
        help='independently derive every drill from production Linux/KVM raw evidence',
    )
    _add_evidence_inputs(qualify, require_runtime_closure=True)
    qualify.add_argument('--expected-host-preflight-sha256', required=True)
    qualify.add_argument('--expected-verifier-source-sha256', required=True)

    key_id = subparsers.add_parser('collector-key-id', help='derive the non-secret Ed25519 collector key identity')
    _add_collector_key_source(key_id)
    return parser


def _add_evidence_inputs(parser: argparse.ArgumentParser, *, require_runtime_closure: bool) -> None:
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--expected-evidence-sha256', required=True)
    parser.add_argument('--expected-worker-spec-sha256', required=True)
    parser.add_argument('--expected-probe-manifest-sha256', required=True)
    parser.add_argument('--expected-collector-public-key-hex', required=True)
    parser.add_argument('--expected-collector-key-id', required=True)
    parser.add_argument(
        '--expected-driver-runtime-closure-manifest-sha256',
        required=require_runtime_closure,
    )
    parser.add_argument(
        '--expected-driver-runtime-closure-receipt-sha256',
        required=require_runtime_closure,
    )
    parser.add_argument('--expected-driver-runtime-closure-sha256', required=require_runtime_closure)


def _add_collector_key_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--collector-key-file', type=Path)
    source.add_argument('--collector-key-fd', type=int)


def _add_runtime_closure_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--driver-runtime-closure', required=True, type=Path)
    parser.add_argument('--expected-driver-runtime-closure-manifest-sha256', required=True)
    parser.add_argument('--expected-driver-runtime-closure-receipt-sha256', required=True)
    parser.add_argument('--expected-driver-runtime-closure-sha256', required=True)


def main() -> None:
    arguments = _parser().parse_args()
    try:
        if arguments.command == 'plan':
            loaded = retain_firecracker_qualification_collector_plan(
                worker_spec_path=arguments.worker_spec,
                expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
                output_root=arguments.output,
                plan_id=arguments.plan_id,
            )
            _write_plan_summary(loaded)
            raise SystemExit(2)
        if arguments.command == 'verify-plan':
            loaded = load_firecracker_qualification_collector_plan(
                arguments.artifact,
                expected_plan_sha256=arguments.expected_plan_sha256,
                expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
            )
            _write_plan_summary(loaded)
            return
        if arguments.command == 'build-driver-runtime-closure':
            loaded = build_and_retain_qualification_driver_runtime_closure(
                closure_id=arguments.closure_id,
                driver_entrypoint_path=arguments.driver,
                interpreter_path=arguments.interpreter,
                runtime_roots=arguments.runtime_root,
                output_root=arguments.output,
                source_date_epoch=arguments.source_date_epoch,
            )
            _write_runtime_closure_summary(loaded)
            return
        if arguments.command == 'verify-driver-runtime-closure':
            loaded = _load_runtime_closure(arguments)
            _write_runtime_closure_summary(loaded)
            return
        if arguments.command == 'collector-key-id':
            private_key = _read_collector_key(arguments)
            public_key = ed25519_public_key_bytes(private_key)
            _write_json(
                {
                    'collector_public_key_hex': public_key.hex(),
                    'collector_key_id': firecracker_live_collector_key_id(public_key),
                }
            )
            return
        if arguments.command == 'collect-live':
            spec, _ = load_pinned_firecracker_worker_spec(
                arguments.worker_spec,
                expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
            )
            manifest, _ = load_pinned_firecracker_qualification_probe_manifest(
                arguments.probe_manifest,
                expected_probe_manifest_sha256=arguments.expected_probe_manifest_sha256,
            )
            runtime_closure = _load_runtime_closure(arguments)
            boundary = PinnedLinuxKvmQualificationDriver(
                driver_id=arguments.driver_id,
                executable_path=arguments.driver,
                expected_executable_sha256=arguments.expected_driver_sha256,
                runtime_closure=runtime_closure,
                worker_spec=spec,
                probe_manifest=manifest,
            )
            loaded = collect_and_retain_firecracker_qualification_evidence(
                worker_spec_path=arguments.worker_spec,
                expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
                probe_manifest_path=arguments.probe_manifest,
                expected_probe_manifest_sha256=arguments.expected_probe_manifest_sha256,
                boundary=boundary,
                mode=FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM,
                collector_private_key=_read_collector_key(arguments),
                expected_collector_key_id=arguments.expected_collector_key_id,
                expected_collector_source_sha256=arguments.expected_collector_source_sha256,
                output_root=arguments.output,
            )
            _write_evidence_summary(loaded, qualified=False)
            return
        if arguments.command == 'verify-authentication':
            loaded = load_firecracker_qualification_collector_evidence(
                arguments.artifact,
                expected_evidence_sha256=arguments.expected_evidence_sha256,
                expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
                expected_probe_manifest_sha256=arguments.expected_probe_manifest_sha256,
                expected_collector_public_key_hex=arguments.expected_collector_public_key_hex,
                expected_collector_key_id=arguments.expected_collector_key_id,
                expected_driver_runtime_closure_manifest_sha256=(
                    arguments.expected_driver_runtime_closure_manifest_sha256
                ),
                expected_driver_runtime_closure_receipt_sha256=(
                    arguments.expected_driver_runtime_closure_receipt_sha256
                ),
                expected_driver_runtime_closure_sha256=arguments.expected_driver_runtime_closure_sha256,
            )
            _write_evidence_summary(loaded, qualified=False)
            return
        verified = independently_verify_firecracker_qualification_collector_evidence(
            arguments.artifact,
            expected_evidence_sha256=arguments.expected_evidence_sha256,
            expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
            expected_probe_manifest_sha256=arguments.expected_probe_manifest_sha256,
            expected_driver_runtime_closure_manifest_sha256=(arguments.expected_driver_runtime_closure_manifest_sha256),
            expected_driver_runtime_closure_receipt_sha256=(arguments.expected_driver_runtime_closure_receipt_sha256),
            expected_driver_runtime_closure_sha256=arguments.expected_driver_runtime_closure_sha256,
            expected_host_preflight_sha256=arguments.expected_host_preflight_sha256,
            expected_collector_public_key_hex=arguments.expected_collector_public_key_hex,
            expected_collector_key_id=arguments.expected_collector_key_id,
            expected_verifier_source_sha256=arguments.expected_verifier_source_sha256,
        )
        _write_json(
            {
                'authenticated_collection_sha256': verified.authenticated_collection_sha256,
                'collector_key_id': verified.authenticated.collector_key_id,
                'mode': verified.authenticated.collection.mode,
                'all_required_drills_passed': verified.full_suite_evidence.all_required_drills_passed,
                'qualified': True,
                'official_leaderboard_execution_qualified': False,
            }
        )
    except (
        FirecrackerQualificationCollectorError,
        FirecrackerQualificationError,
        QualificationDriverRuntimeClosureError,
    ) as error:
        sys.stderr.write(f'firecracker qualification collector rejected: {error}\n')
        raise SystemExit(64) from error


def _read_collector_key(arguments: argparse.Namespace):
    if arguments.collector_key_file is not None:
        return read_firecracker_live_collector_private_key_file(arguments.collector_key_file)
    return read_firecracker_live_collector_private_key_fd(arguments.collector_key_fd)


def _load_runtime_closure(arguments: argparse.Namespace):
    return verify_qualification_driver_runtime_closure(
        arguments.driver_runtime_closure,
        expected_manifest_sha256=arguments.expected_driver_runtime_closure_manifest_sha256,
        expected_receipt_sha256=arguments.expected_driver_runtime_closure_receipt_sha256,
        expected_closure_sha256=arguments.expected_driver_runtime_closure_sha256,
        require_root_owned=True,
    )


def _write_runtime_closure_summary(loaded) -> None:
    _write_json(
        {
            'artifact_root': loaded.root,
            'closure_id': loaded.manifest.closure_id,
            'driver_entrypoint_sha256': loaded.manifest.driver_entrypoint_sha256,
            'interpreter_sha256': loaded.manifest.interpreter_sha256,
            'runtime_entry_count': loaded.manifest.entry_count,
            'driver_runtime_closure_manifest_sha256': loaded.manifest_sha256,
            'driver_runtime_closure_receipt_sha256': loaded.receipt_sha256,
            'driver_runtime_closure_sha256': loaded.closure_sha256,
            'self_contained_executable_claimed': False,
            'reproducible_build_claimed': False,
        }
    )


def _write_plan_summary(loaded) -> None:
    plan = loaded.plan
    _write_json(
        {
            'artifact_root': loaded.root,
            'collector_plan_sha256': loaded.plan_sha256,
            'plan_id': plan.plan_id,
            'status': plan.status,
            'required_drill_count': len(plan.drills),
            'missing_capability_count': len(plan.missing_capabilities),
            'host_prerequisites_observed': plan.host_linux_kvm_cgroup_prerequisites_observed,
            'live_vm_launched': plan.live_vm_launched,
            'full_suite_evidence_emitted': plan.full_suite_evidence_emitted,
            'qualified': plan.qualified,
            'official_leaderboard_execution_qualified': plan.official_leaderboard_execution_qualified,
        }
    )


def _write_evidence_summary(loaded, *, qualified: bool) -> None:
    collection = loaded.authenticated.collection
    _write_json(
        {
            'artifact_root': loaded.root,
            'collector_evidence_sha256': loaded.evidence_sha256,
            'collection_id': collection.collection_id,
            'mode': collection.mode,
            'required_drill_count': len(collection.drills),
            'development_simulated': collection.development_simulated,
            'production_qualification_eligible': collection.production_qualification_eligible,
            'driver_runtime_closure_sha256': collection.driver_runtime_closure_sha256,
            'qualified': qualified,
            'official_leaderboard_execution_qualified': False,
        }
    )


def _write_json(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
