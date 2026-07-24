"""Local validation and scoring CLI for VaxReplay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.aggregation import (
    SuiteManifest,
    aggregate_scores,
    make_suite_manifest,
    suite_manifest_sha256,
)
from vaxreplay.baselines import oracle_submission, uniform_submission
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes, resolve_episode_root
from vaxreplay.case_schema import Submission
from vaxreplay.prompt import build_episode_prompt
from vaxreplay.release import build_synthetic_integration_release, load_release
from vaxreplay.release_schema import ReleasePurpose
from vaxreplay.runner.challenge import build_challenge_bundle, load_challenge_bundle
from vaxreplay.runner.orchestrator import load_receipt_key, load_run_artifact, receipt_key_id
from vaxreplay.runner.schema import RunnerPolicy, SystemSubmissionManifest
from vaxreplay.scoring import make_submission_evaluator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate and score VaxReplay episodes')
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('validate', 'prompt'):
        subparser = subparsers.add_parser(command)
        subparser.add_argument('--episode-dir', required=True)

    baseline = subparsers.add_parser('baseline')
    baseline.add_argument('--episode-dir', required=True)
    baseline.add_argument('--kind', choices=('uniform', 'oracle'), default='uniform')

    score = subparsers.add_parser('score')
    score.add_argument('--episode-dir', required=True)
    score.add_argument('--submission', required=True)

    score_suite = subparsers.add_parser('score-suite')
    score_suite.add_argument('--suite-manifest', required=True)
    score_suite.add_argument('--expected-suite-sha256', required=True)
    score_suite.add_argument('--episode-dir', action='append', required=True)
    score_suite.add_argument('--responses-jsonl', required=True)

    make_suite = subparsers.add_parser('make-suite')
    make_suite.add_argument('--suite-id', required=True)
    make_suite.add_argument('--episode-dir', action='append', required=True)

    suite_hash = subparsers.add_parser('suite-hash')
    suite_hash.add_argument('--suite-manifest', required=True)

    make_challenge = subparsers.add_parser('make-challenge')
    make_challenge.add_argument('--challenge-id', required=True)
    make_challenge.add_argument('--suite-id', required=True)
    make_challenge.add_argument('--episode-dir', action='append', required=True)
    make_challenge.add_argument('--sample-index', type=int, default=0)
    make_challenge.add_argument('--output-dir', required=True)

    challenge_hash = subparsers.add_parser('challenge-hash')
    challenge_hash.add_argument('--challenge-dir', required=True)

    verify_challenge = subparsers.add_parser('verify-challenge')
    verify_challenge.add_argument('--challenge-dir', required=True)
    verify_challenge.add_argument('--expected-challenge-sha256', required=True)

    score_run = subparsers.add_parser('score-run')
    score_run.add_argument('--challenge-dir', required=True)
    score_run.add_argument('--expected-challenge-sha256', required=True)
    score_run.add_argument('--run-dir', required=True)
    score_run.add_argument('--system-manifest', required=True)
    score_run.add_argument('--policy', required=True)
    score_run.add_argument('--receipt-key', required=True)
    score_run.add_argument('--expected-receipt-key-id', required=True)
    score_run.add_argument('--episode-dir', action='append', required=True)
    score_run.add_argument('--allow-development-run', action='store_true')

    package_release = subparsers.add_parser('package-release')
    package_release.add_argument('--release-id', required=True)
    package_release.add_argument('--challenge-id', required=True)
    package_release.add_argument('--suite-id', required=True)
    package_release.add_argument('--episode-dir', action='append', required=True)
    package_release.add_argument('--policy', required=True)
    package_release.add_argument('--receipt-key', required=True)
    package_release.add_argument('--public-output-dir', required=True)
    package_release.add_argument('--private-output-dir', required=True)

    verify_release = subparsers.add_parser('verify-release')
    verify_release.add_argument('--public-release-dir', required=True)
    verify_release.add_argument('--private-release-dir', required=True)
    verify_release.add_argument('--expected-release-sha256', required=True)

    score_release_run = subparsers.add_parser('score-release-run')
    score_release_run.add_argument('--public-release-dir', required=True)
    score_release_run.add_argument('--private-release-dir', required=True)
    score_release_run.add_argument('--expected-release-sha256', required=True)
    score_release_run.add_argument('--run-dir', required=True)
    score_release_run.add_argument('--system-manifest', required=True)
    score_release_run.add_argument('--receipt-key', required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == 'package-release':
        policy = RunnerPolicy.model_validate_json(Path(args.policy).read_bytes())
        receipt_key = load_receipt_key(Path(args.receipt_key))
        release = build_synthetic_integration_release(
            release_id=args.release_id,
            challenge_id=args.challenge_id,
            suite_id=args.suite_id,
            episode_dirs=tuple(Path(path) for path in args.episode_dir),
            policy=policy,
            receipt_key_id=receipt_key_id(receipt_key),
            public_output_dir=Path(args.public_output_dir),
            private_output_dir=Path(args.private_output_dir),
        )
        _write_json(
            {
                'release_id': release.public_manifest.release_id,
                'release_purpose': release.public_manifest.purpose,
                'public_release_sha256': release.public_manifest_sha256,
                'challenge_bundle_sha256': release.challenge.manifest_sha256,
                'receipt_key_id': release.public_manifest.receipt_key_id,
                'sealed_eligible': release.public_manifest.sealed_eligible,
                'episode_count': release.public_manifest.episode_count,
            }
        )
        return
    if args.command in {'verify-release', 'score-release-run'}:
        release = load_release(
            Path(args.public_release_dir),
            Path(args.private_release_dir),
            expected_public_release_sha256=args.expected_release_sha256,
        )
        if args.command == 'verify-release':
            _write_json(
                {
                    'release_id': release.public_manifest.release_id,
                    'release_purpose': release.public_manifest.purpose,
                    'public_release_sha256': release.public_manifest_sha256,
                    'challenge_bundle_sha256': release.challenge.manifest_sha256,
                    'source_tiers': sorted({admission.source_tier for admission in release.temporal_admissions}),
                    'split_inventory_complete': release.private_manifest.split_inventory_complete,
                    'sealed_eligible': release.public_manifest.sealed_eligible,
                    'episode_count': release.public_manifest.episode_count,
                }
            )
            return
        system = SystemSubmissionManifest.model_validate_json(Path(args.system_manifest).read_bytes())
        receipt_key = load_receipt_key(Path(args.receipt_key))
        run = load_run_artifact(
            Path(args.run_dir),
            challenge=release.challenge,
            system=system,
            policy=release.policy,
            receipt_key=receipt_key,
            expected_receipt_key_id=release.public_manifest.receipt_key_id,
            require_sealed=release.public_manifest.purpose != ReleasePurpose.SYNTHETIC_INTEGRATION,
        )
        response_lines = [record.decode('utf-8') for record in run.response_records]
        _write_json(
            _score_suite(release.challenge.suite, list(release.bundles), response_lines).model_dump(mode='json')
        )
        return
    if args.command == 'make-challenge':
        challenge = build_challenge_bundle(
            Path(args.output_dir),
            challenge_id=args.challenge_id,
            suite_id=args.suite_id,
            episode_dirs=(Path(path) for path in args.episode_dir),
            sample_index=args.sample_index,
        )
        _write_json(
            {
                'challenge_id': challenge.manifest.challenge_id,
                'challenge_bundle_sha256': challenge.manifest_sha256,
                'suite_manifest_sha256': challenge.manifest.suite_manifest_sha256,
                'episode_count': len(challenge.envelopes),
            }
        )
        return
    if args.command in {'challenge-hash', 'verify-challenge'}:
        challenge = load_challenge_bundle(Path(args.challenge_dir))
        if args.command == 'verify-challenge' and challenge.manifest_sha256 != args.expected_challenge_sha256:
            raise ValueError('challenge bundle does not match expected_challenge_sha256')
        _write_json(
            {
                'challenge_id': challenge.manifest.challenge_id,
                'challenge_bundle_sha256': challenge.manifest_sha256,
                'suite_manifest_sha256': challenge.manifest.suite_manifest_sha256,
                'episode_count': len(challenge.envelopes),
            }
        )
        return
    if args.command == 'make-suite':
        bundles = [EpisodeBundle.load(resolve_episode_root(path)) for path in args.episode_dir]
        _write_json(make_suite_manifest(args.suite_id, bundles).model_dump(mode='json'))
        return
    if args.command == 'suite-hash':
        manifest = SuiteManifest.model_validate_json(Path(args.suite_manifest).read_text(encoding='utf-8'))
        _write_json({'suite_manifest_sha256': suite_manifest_sha256(manifest)})
        return
    if args.command == 'score-suite':
        manifest = SuiteManifest.model_validate_json(Path(args.suite_manifest).read_text(encoding='utf-8'))
        actual_suite_sha256 = suite_manifest_sha256(manifest)
        if actual_suite_sha256 != args.expected_suite_sha256:
            raise ValueError('suite manifest does not match expected_suite_sha256')
        bundles = [EpisodeBundle.load(resolve_episode_root(path), include_private=True) for path in args.episode_dir]
        response_lines = _load_response_lines(Path(args.responses_jsonl))
        _write_json(_score_suite(manifest, bundles, response_lines).model_dump(mode='json'))
        return
    if args.command == 'score-run':
        challenge = load_challenge_bundle(Path(args.challenge_dir))
        if challenge.manifest_sha256 != args.expected_challenge_sha256:
            raise ValueError('challenge bundle does not match expected_challenge_sha256')
        system = SystemSubmissionManifest.model_validate_json(Path(args.system_manifest).read_bytes())
        policy = RunnerPolicy.model_validate_json(Path(args.policy).read_bytes())
        receipt_key = load_receipt_key(Path(args.receipt_key))
        run = load_run_artifact(
            Path(args.run_dir),
            challenge=challenge,
            system=system,
            policy=policy,
            receipt_key=receipt_key,
            expected_receipt_key_id=args.expected_receipt_key_id,
            require_sealed=not args.allow_development_run,
        )
        bundles = [EpisodeBundle.load(resolve_episode_root(path), include_private=True) for path in args.episode_dir]
        response_lines = [record.decode('utf-8') for record in run.response_records]
        _write_json(_score_suite(challenge.suite, bundles, response_lines).model_dump(mode='json'))
        return

    root = resolve_episode_root(args.episode_dir)
    include_private = args.command == 'score' or (args.command == 'baseline' and args.kind == 'oracle')
    bundle = EpisodeBundle.load(root, include_private=include_private)

    if args.command == 'validate':
        _write_json(
            {
                'episode_id': bundle.manifest.episode_id,
                'manifest_sha256': bundle.manifest_sha256,
                'visible_evidence_count': len(bundle.visible_evidence),
                'total_evidence_count': len(bundle.evidence),
            }
        )
    elif args.command == 'prompt':
        sys.stdout.write(build_episode_prompt(bundle) + '\n')
    elif args.command == 'baseline':
        submission = oracle_submission(bundle) if args.kind == 'oracle' else uniform_submission(bundle)
        sys.stdout.write(submission.model_dump_json(indent=2) + '\n')
    elif args.command == 'score':
        submission = Submission.model_validate_json(Path(args.submission).read_text(encoding='utf-8'))
        _write_json(make_submission_evaluator(bundle).score(submission).model_dump(mode='json'))


def _write_json(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


def _load_response_lines(path: Path) -> list[str]:
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as error:
        raise ValueError(f'cannot read response JSONL {path}: {error}') from error
    response_lines: list[str] = []
    for raw_line in raw_lines:
        try:
            response_lines.append(raw_line.decode('utf-8'))
        except UnicodeDecodeError:
            response_lines.append('')
    return response_lines


def _score_suite(
    manifest: SuiteManifest,
    bundles: list[EpisodeBundle],
    response_lines: list[str],
):
    derived_manifest = make_suite_manifest(manifest.suite_id, bundles)
    if derived_manifest != manifest:
        raise ValueError('suite manifest does not match the supplied episode bundles')
    if len(response_lines) > len(manifest.episodes):
        raise ValueError('responses JSONL contains more rows than the suite manifest')
    bundle_by_id = {bundle.manifest.episode_id: bundle for bundle in bundles}
    scores = []
    for binding, response_text in zip(manifest.episodes, response_lines, strict=False):
        try:
            submission = Submission.model_validate_json(response_text)
        except ValidationError:
            continue
        bundle = bundle_by_id[binding.episode_id]
        scores.append(make_submission_evaluator(bundle, allow_sealed_test=True).score(submission))
    return aggregate_scores(manifest, scores)


if __name__ == '__main__':
    main()
