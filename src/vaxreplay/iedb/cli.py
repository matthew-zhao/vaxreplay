"""Command-line interface for pinned IEDB snapshot validation and episode building."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split
from vaxreplay.iedb.adapter import audit_episode, build_episode, export_public_episode, load_snapshot
from vaxreplay.iedb.live_capture import (
    IedbApiCaptureSpec,
    IedbFullExportCaptureSpec,
    build_api_capture,
    build_full_export_capture,
    write_capture_manifest,
)
from vaxreplay.iedb.production_plan import (
    compile_iedb_production_plan,
    discover_and_compile_iedb_production_plan,
    read_iedb_production_plan_input,
    write_compiled_iedb_production_plan,
)
from vaxreplay.iedb.raw_schema import IedbEpisodeSpec
from vaxreplay.release import build_synthetic_integration_release
from vaxreplay.runner.orchestrator import load_receipt_key, receipt_key_id
from vaxreplay.runner.schema import IsolationTier, RunnerPolicy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Normalize pinned IEDB snapshots into VaxReplay')
    subparsers = parser.add_subparsers(dest='command', required=True)

    validate = subparsers.add_parser('validate-snapshot')
    validate.add_argument('--snapshot-dir', required=True)

    for command in ('capture-api', 'capture-full-export'):
        capture = subparsers.add_parser(command)
        capture.add_argument('--capture-dir', required=True)
        capture.add_argument('--spec', required=True)
        capture.add_argument('--manifest-output', required=True)

    for command in ('compile-production-plan', 'discover-production-plan'):
        planning = subparsers.add_parser(command)
        planning.add_argument('--policy', required=True)
        planning.add_argument('--expected-policy-sha256', required=True)
        planning.add_argument('--output-dir', required=True)
        if command == 'compile-production-plan':
            planning.add_argument('--metrics-body', required=True)
            planning.add_argument('--metrics-receipt', required=True)

    build = subparsers.add_parser('build')
    build.add_argument('--spec', required=True)
    build.add_argument('--snapshot-dir', action='append', required=True)
    build.add_argument('--output-dir', required=True)
    build.add_argument('--label-commitment-key-file')

    audit = subparsers.add_parser('audit')
    audit.add_argument('--episode-dir', required=True)

    public_export = subparsers.add_parser('export-public')
    public_export.add_argument('--episode-dir', required=True)
    public_export.add_argument('--output-dir', required=True)

    pilot = subparsers.add_parser('build-synthetic-pilot')
    pilot.add_argument('--spec', required=True)
    pilot.add_argument('--snapshot-dir', action='append', required=True)
    pilot.add_argument('--receipt-key', required=True)
    pilot.add_argument('--public-output-dir', required=True)
    pilot.add_argument('--private-output-dir', required=True)
    pilot.add_argument('--label-commitment-key-file')
    pilot.add_argument('--policy')
    pilot.add_argument('--release-id', default='iedb-synthetic-pilot-v0')
    pilot.add_argument('--challenge-id', default='iedb-synthetic-pilot-challenge-v0')
    pilot.add_argument('--suite-id', default='iedb-synthetic-pilot-suite-v0')
    pilot.add_argument('--episode-id')
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command in {'compile-production-plan', 'discover-production-plan'}:
        policy_bytes = read_iedb_production_plan_input(
            Path(args.policy),
            kind='compiler_policy',
        )
        if args.command == 'compile-production-plan':
            compiled = compile_iedb_production_plan(
                policy_bytes=policy_bytes,
                expected_policy_sha256=args.expected_policy_sha256,
                discovery_metrics_bytes=read_iedb_production_plan_input(
                    Path(args.metrics_body),
                    kind='discovery_metrics',
                ),
                discovery_receipt_bytes=read_iedb_production_plan_input(
                    Path(args.metrics_receipt),
                    kind='discovery_receipt',
                ),
            )
        else:
            compiled = discover_and_compile_iedb_production_plan(
                policy_bytes=policy_bytes,
                expected_policy_sha256=args.expected_policy_sha256,
            )
        output = write_compiled_iedb_production_plan(
            compiled,
            Path(args.output_dir),
        )
        _write_json(
            {
                'compiler_policy_sha256': compiled.compilation.compiler_policy_sha256,
                'discovery_is_planning_input_only': True,
                'discovery_metrics_sha256': (compiled.compilation.discovery_metrics_sha256),
                'output_dir': str(output),
                'plan_compilation_sha256': hashlib.sha256(canonical_json_bytes(compiled.compilation)).hexdigest(),
                'plan_requires_pre_capture_selection_registry_commitment': True,
                'source_verifier_policy_sha256': (compiled.compilation.source_verifier_policy_sha256),
                'static_collection_plan_sha256': (compiled.compilation.static_collection_plan_sha256),
                'status': 'planned_not_tier_a_release_ready',
                'tier_a_release_ready': False,
            }
        )
    elif args.command in {'capture-api', 'capture-full-export'}:
        if args.command == 'capture-api':
            spec = IedbApiCaptureSpec.model_validate_json(Path(args.spec).read_bytes())
            capture = build_api_capture(Path(args.capture_dir), spec)
        else:
            spec = IedbFullExportCaptureSpec.model_validate_json(Path(args.spec).read_bytes())
            capture = build_full_export_capture(Path(args.capture_dir), spec)
        output = write_capture_manifest(capture, Path(args.manifest_output))
        binding = capture.manifest.source_binding
        completeness_scope = binding.completeness_scope
        _write_json(
            {
                'capture_id': capture.manifest.capture_id,
                'source_mode': capture.manifest.source_binding.source_mode,
                'source_build_at': capture.manifest.source_build_at.isoformat(),
                'retrieved_at': capture.manifest.retrieved_at.isoformat(),
                'capture_manifest_sha256': capture.manifest_sha256,
                'capture_manifest_path': str(output),
                'completeness_scope': completeness_scope,
                'complete_within_scope': True,
                'source_authenticity_verified': capture.manifest.source_authenticity_verified,
                'tier_a_eligible': capture.manifest.tier_a_eligible,
                'external_timestamp_required': capture.manifest.external_timestamp_required,
                'next_action': (
                    'independently verify source provenance, bind this manifest into a decision package, '
                    'and obtain external timestamp proofs'
                ),
            }
        )
    elif args.command == 'validate-snapshot':
        snapshot = load_snapshot(Path(args.snapshot_dir))
        _write_json(
            {
                'snapshot_id': snapshot.manifest.snapshot_id,
                'manifest_sha256': snapshot.manifest_sha256,
                'source_build_at': snapshot.manifest.source_build_at.isoformat(),
                'tables': {endpoint.value: len(rows) for endpoint, rows in snapshot.rows_by_endpoint.items()},
            }
        )
    elif args.command == 'build':
        spec = IedbEpisodeSpec.model_validate_json(Path(args.spec).read_bytes())
        commitment_key = None
        if args.label_commitment_key_file:
            commitment_key = bytes.fromhex(Path(args.label_commitment_key_file).read_text(encoding='ascii').strip())
        bundle = build_episode(
            spec=spec,
            snapshot_roots=[Path(path) for path in args.snapshot_dir],
            output_root=Path(args.output_dir),
            label_commitment_key=commitment_key,
        )
        _write_json(
            {
                'episode_id': bundle.manifest.episode_id,
                'manifest_sha256': bundle.manifest_sha256,
                'candidate_count': len(bundle.manifest.candidate_ids),
                'visible_evidence_count': len(bundle.visible_evidence),
                'total_evidence_count': len(bundle.evidence),
            }
        )
    elif args.command == 'audit':
        _write_json(audit_episode(Path(args.episode_dir)))
    elif args.command == 'export-public':
        bundle = export_public_episode(Path(args.episode_dir), Path(args.output_dir))
        _write_json(
            {
                'episode_id': bundle.manifest.episode_id,
                'manifest_sha256': bundle.manifest_sha256,
                'candidate_count': len(bundle.manifest.candidate_ids),
                'evidence_count': len(bundle.evidence),
            }
        )
    elif args.command == 'build-synthetic-pilot':
        spec = IedbEpisodeSpec.model_validate_json(Path(args.spec).read_bytes())
        if not spec.synthetic:
            raise ValueError('the synthetic pilot builder refuses real IEDB episode specs')
        pilot_spec = spec.model_copy(
            update={
                'episode_id': args.episode_id or f'{spec.episode_id}-sealed-pilot-v0',
                'split': Split.TEST,
            }
        )
        commitment_key = None
        if args.label_commitment_key_file:
            commitment_key = bytes.fromhex(Path(args.label_commitment_key_file).read_text(encoding='ascii').strip())
        policy = (
            RunnerPolicy.model_validate_json(Path(args.policy).read_bytes())
            if args.policy
            else RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
        )
        run_receipt_key = load_receipt_key(Path(args.receipt_key))
        with tempfile.TemporaryDirectory(prefix='vaxreplay-iedb-synthetic-pilot-') as temporary_directory:
            episode_root = Path(temporary_directory) / 'episode'
            build_episode(
                spec=pilot_spec,
                snapshot_roots=[Path(path) for path in args.snapshot_dir],
                output_root=episode_root,
                label_commitment_key=commitment_key,
            )
            release = build_synthetic_integration_release(
                release_id=args.release_id,
                challenge_id=args.challenge_id,
                suite_id=args.suite_id,
                episode_dirs=(episode_root,),
                policy=policy,
                receipt_key_id=receipt_key_id(run_receipt_key),
                public_output_dir=Path(args.public_output_dir),
                private_output_dir=Path(args.private_output_dir),
            )
        _write_json(
            {
                'release_id': release.public_manifest.release_id,
                'episode_id': pilot_spec.episode_id,
                'release_purpose': release.public_manifest.purpose,
                'source_tier': 'tier_c',
                'split_inventory_complete': release.private_manifest.split_inventory_complete,
                'sealed_eligible': release.public_manifest.sealed_eligible,
                'public_release_sha256': release.public_manifest_sha256,
                'challenge_bundle_sha256': release.challenge.manifest_sha256,
                'receipt_key_id': release.public_manifest.receipt_key_id,
            }
        )


def _write_json(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
