"""CLI for public challenge execution. This process never loads labels or scores."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.runner.challenge import load_challenge_bundle
from vaxreplay.runner.oci import OciDevelopmentBackend
from vaxreplay.runner.orchestrator import load_receipt_key, receipt_key_id, run_challenge_bundle
from vaxreplay.runner.schema import RunnerPolicy, SystemSubmissionManifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a VaxReplay challenge without labels or scoring access')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_oci = subparsers.add_parser('run-oci')
    run_oci.add_argument('--challenge-dir', required=True)
    run_oci.add_argument('--expected-challenge-sha256', required=True)
    run_oci.add_argument('--system-manifest', required=True)
    run_oci.add_argument('--policy', required=True)
    run_oci.add_argument('--receipt-key', required=True)
    run_oci.add_argument('--expected-receipt-key-id', required=True)
    run_oci.add_argument('--output-dir', required=True)
    run_oci.add_argument('--runtime', default='docker')

    for command, argument in (('system-hash', '--system-manifest'), ('policy-hash', '--policy')):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(argument, required=True)
    receipt_key = subparsers.add_parser('receipt-key-id')
    receipt_key.add_argument('--receipt-key', required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == 'system-hash':
        system = _load_model(Path(args.system_manifest), SystemSubmissionManifest)
        _write_json({'system_manifest_sha256': _object_sha256(system)})
        return
    if args.command == 'policy-hash':
        policy = _load_model(Path(args.policy), RunnerPolicy)
        _write_json({'policy_sha256': _object_sha256(policy)})
        return
    if args.command == 'receipt-key-id':
        _write_json({'receipt_key_id': receipt_key_id(load_receipt_key(Path(args.receipt_key)))})
        return

    challenge = load_challenge_bundle(Path(args.challenge_dir))
    system = _load_model(Path(args.system_manifest), SystemSubmissionManifest)
    policy = _load_model(Path(args.policy), RunnerPolicy)
    receipt_key = load_receipt_key(Path(args.receipt_key))
    run = run_challenge_bundle(
        challenge,
        expected_challenge_sha256=args.expected_challenge_sha256,
        system=system,
        policy=policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=args.expected_receipt_key_id,
        output_dir=Path(args.output_dir),
        backend=OciDevelopmentBackend(args.runtime),
    )
    status_counts = Counter(receipt.status.value for receipt in run.receipt.episodes)
    _write_json(
        {
            'run_receipt_sha256': run.receipt_sha256,
            'responses_sha256': run.receipt.responses_sha256,
            'receipt_key_id': run.receipt.receipt_key_id,
            'sealed': run.receipt.sealed,
            'status_counts': {key: status_counts[key] for key in sorted(status_counts)},
        }
    )


def _load_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except OSError as error:
        raise ValueError(f'cannot read {path}: {error}') from error


def _object_sha256(value: object) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_json(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
