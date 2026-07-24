"""Operator CLI for immutable Firecracker host-qualification evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vaxreplay.agentic.firecracker_qualification import (
    FirecrackerFullSuiteEvidence,
    FirecrackerQualificationError,
    firecracker_qualification_key_id,
    inspect_and_retain_firecracker_host,
    load_firecracker_qualification,
    read_firecracker_qualification_key_fd,
    read_firecracker_qualification_key_file,
    verify_and_retain_firecracker_live_qualification,
)
from vaxreplay.bundle import canonical_json_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Retain or verify fail-closed Linux/KVM qualification evidence',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    inspect = subparsers.add_parser(
        'inspect-host',
        help='run exact host preflight and retain an authenticated result (never full qualification by itself)',
    )
    inspect.add_argument('--worker-spec', required=True, type=Path)
    inspect.add_argument('--expected-worker-spec-sha256', required=True)
    inspect.add_argument('--expected-qualification-key-id', required=True)
    inspect.add_argument('--output', required=True, type=Path)
    inspect.add_argument('--qualification-id')
    inspect.add_argument(
        '--full-suite-evidence',
        type=Path,
        help='disabled: caller-authored evidence cannot establish runtime qualification',
    )
    _add_key_source(inspect)

    verify = subparsers.add_parser('verify', help='verify an exact retained qualification artifact')
    verify.add_argument('--artifact', required=True, type=Path)
    verify.add_argument('--expected-artifact-sha256', required=True)
    verify.add_argument('--expected-worker-spec-sha256', required=True)
    verify.add_argument('--expected-qualification-key-id', required=True)
    _add_optional_collector_pins(verify)
    _add_key_source(verify)

    live = subparsers.add_parser(
        'verify-live-collector',
        help='independently verify signed raw Linux/KVM drills and retain the only accepted positive artifact',
    )
    live.add_argument('--collector-evidence', required=True, type=Path)
    live.add_argument('--expected-collector-evidence-sha256', required=True)
    live.add_argument('--worker-spec', required=True, type=Path)
    live.add_argument('--expected-worker-spec-sha256', required=True)
    live.add_argument('--expected-probe-manifest-sha256', required=True)
    live.add_argument('--expected-driver-runtime-closure-manifest-sha256', required=True)
    live.add_argument('--expected-driver-runtime-closure-receipt-sha256', required=True)
    live.add_argument('--expected-driver-runtime-closure-sha256', required=True)
    live.add_argument('--expected-host-preflight-sha256', required=True)
    live.add_argument('--expected-collector-public-key-hex', required=True)
    live.add_argument('--expected-collector-key-id', required=True)
    live.add_argument('--expected-verifier-source-sha256', required=True)
    live.add_argument('--expected-qualification-key-id', required=True)
    live.add_argument('--output', required=True, type=Path)
    live.add_argument('--qualification-id')
    _add_key_source(live)

    key_id = subparsers.add_parser('key-id', help='derive the non-secret qualification key identifier')
    _add_key_source(key_id)
    return parser


def _add_key_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--key-file', type=Path)
    source.add_argument('--key-fd', type=int)


def _add_optional_collector_pins(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--expected-collector-evidence-sha256')
    parser.add_argument('--expected-probe-manifest-sha256')
    parser.add_argument('--expected-driver-runtime-closure-manifest-sha256')
    parser.add_argument('--expected-driver-runtime-closure-receipt-sha256')
    parser.add_argument('--expected-driver-runtime-closure-sha256')
    parser.add_argument('--expected-collector-public-key-hex')
    parser.add_argument('--expected-collector-key-id')
    parser.add_argument('--expected-verifier-source-sha256')


def main() -> None:
    arguments = _parser().parse_args()
    try:
        key = _read_key(arguments)
        if arguments.command == 'key-id':
            _write_summary({'qualification_key_id': firecracker_qualification_key_id(key)})
            return
        if arguments.command == 'inspect-host':
            full_suite = _load_full_suite(arguments.full_suite_evidence)
            loaded = inspect_and_retain_firecracker_host(
                worker_spec_path=arguments.worker_spec,
                expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
                output_root=arguments.output,
                qualification_key=key,
                expected_qualification_key_id=arguments.expected_qualification_key_id,
                qualification_id=arguments.qualification_id,
                full_suite_evidence=full_suite,
            )
            _write_loaded_summary(loaded)
            if not loaded.authenticated.record.qualified:
                raise SystemExit(2)
            return
        if arguments.command == 'verify-live-collector':
            loaded = verify_and_retain_firecracker_live_qualification(
                collector_evidence_root=arguments.collector_evidence,
                expected_collector_evidence_sha256=arguments.expected_collector_evidence_sha256,
                worker_spec_path=arguments.worker_spec,
                expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
                expected_probe_manifest_sha256=arguments.expected_probe_manifest_sha256,
                expected_driver_runtime_closure_manifest_sha256=(
                    arguments.expected_driver_runtime_closure_manifest_sha256
                ),
                expected_driver_runtime_closure_receipt_sha256=(
                    arguments.expected_driver_runtime_closure_receipt_sha256
                ),
                expected_driver_runtime_closure_sha256=arguments.expected_driver_runtime_closure_sha256,
                expected_host_preflight_sha256=arguments.expected_host_preflight_sha256,
                expected_collector_public_key_hex=arguments.expected_collector_public_key_hex,
                expected_collector_key_id=arguments.expected_collector_key_id,
                expected_verifier_source_sha256=arguments.expected_verifier_source_sha256,
                output_root=arguments.output,
                qualification_key=key,
                expected_qualification_key_id=arguments.expected_qualification_key_id,
                qualification_id=arguments.qualification_id,
            )
            _write_loaded_summary(loaded)
            return
        loaded = load_firecracker_qualification(
            arguments.artifact,
            qualification_key=key,
            expected_qualification_key_id=arguments.expected_qualification_key_id,
            expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
            expected_artifact_sha256=arguments.expected_artifact_sha256,
            expected_collector_evidence_sha256=arguments.expected_collector_evidence_sha256,
            expected_probe_manifest_sha256=arguments.expected_probe_manifest_sha256,
            expected_driver_runtime_closure_manifest_sha256=(arguments.expected_driver_runtime_closure_manifest_sha256),
            expected_driver_runtime_closure_receipt_sha256=(arguments.expected_driver_runtime_closure_receipt_sha256),
            expected_driver_runtime_closure_sha256=arguments.expected_driver_runtime_closure_sha256,
            expected_collector_public_key_hex=arguments.expected_collector_public_key_hex,
            expected_collector_key_id=arguments.expected_collector_key_id,
            expected_verifier_source_sha256=arguments.expected_verifier_source_sha256,
        )
        _write_loaded_summary(loaded)
    except (FirecrackerQualificationError, ValueError) as error:
        # Errors are deliberately bounded and never include key bytes or key paths.
        sys.stderr.write(f'firecracker qualification rejected: {error}\n')
        raise SystemExit(64) from error


def _read_key(arguments: argparse.Namespace) -> bytes:
    if arguments.key_file is not None:
        return read_firecracker_qualification_key_file(arguments.key_file)
    return read_firecracker_qualification_key_fd(arguments.key_fd)


def _load_full_suite(path: Path | None) -> FirecrackerFullSuiteEvidence | None:
    if path is None:
        return None
    raise FirecrackerQualificationError(
        'unauthenticated full-suite evidence is disabled; an authenticated live collector is required'
    )


def _write_loaded_summary(loaded) -> None:
    record = loaded.authenticated.record
    collector_binding = record.collector_verification
    _write_summary(
        {
            'artifact_root': loaded.root,
            'artifact_sha256': loaded.artifact_sha256,
            'qualification_id': record.qualification_id,
            'status': record.status.value,
            'qualified': record.qualified,
            'host_preflight_present': record.preflight is not None,
            'full_runtime_evidence_present': record.full_suite_evidence is not None,
            'authenticated_collector_evidence_present': record.collector_verification is not None,
            'driver_runtime_closure_sha256': (
                collector_binding.driver_runtime_closure_sha256 if collector_binding is not None else None
            ),
            'preflight_alone_is_full_runtime_qualification': False,
        }
    )


def _write_summary(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
