"""One-command operator entrypoint for the canonical development Lane A composition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vaxreplay.agentic.clinical_launcher import ClinicalLauncherFailure, ClinicalLauncherSuccess
from vaxreplay.agentic.clinical_operator import (
    ClinicalOperatorError,
    dry_run_report,
    execute_operator_task,
    load_canonical_clinical_operator_manifest,
    validate_operator_inputs,
)
from vaxreplay.bundle import canonical_json_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run one externally pinned Lane A task on an authenticated Linux/KVM deployment',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    run = subparsers.add_parser(
        'run-task',
        help='run the explicit development-only local-SQLite composition',
    )
    run.add_argument('--manifest', required=True, type=Path)
    run.add_argument('--expected-manifest-sha256', required=True)
    run.add_argument(
        '--secret-root',
        required=True,
        type=Path,
        help='owned mode-0700 directory containing the documented fixed-name mode-0600 secret files',
    )
    run.add_argument(
        '--allow-development-local-sqlite',
        action='store_true',
        required=True,
        help=(
            'acknowledge that this command is not the managed authority; managed manifests are '
            'rejected instead of falling back to their registry_path'
        ),
    )
    run.add_argument(
        '--dry-run',
        action='store_true',
        help='authenticate config, qualification, current host, secrets, and workspace without opening SQLite',
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    inputs = None
    try:
        manifest, manifest_sha256 = load_canonical_clinical_operator_manifest(
            arguments.manifest,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
        )
        if manifest.registry_execution_mode != 'development-local-sqlite':
            raise ClinicalOperatorError('managed-authority manifests require the deployment-owned managed entrypoint')
        inputs = validate_operator_inputs(
            manifest,
            manifest_sha256=manifest_sha256,
            secret_root=arguments.secret_root,
        )
        if arguments.dry_run:
            _write(dry_run_report(inputs))
            return
        result = execute_operator_task(inputs)
        if isinstance(result, ClinicalLauncherSuccess):
            _write(
                {
                    'status': 'succeeded',
                    'run_id': result.launch.run_id,
                    'reservation_sha256': manifest.reservation_sha256,
                    'episode_id': manifest.episode_id,
                    'evidence_sha256': result.record.evidence_sha256,
                    'attempt_consumed': True,
                    'retry_permitted': False,
                    'development_only': True,
                    'official_execution_qualified': False,
                    'leaderboard_admitted': False,
                }
            )
            return
        if not isinstance(result, ClinicalLauncherFailure):
            raise ClinicalOperatorError('canonical launcher returned an unknown result type')
        _write(
            {
                'status': 'failed',
                'run_id': result.launch.run_id,
                'reservation_sha256': manifest.reservation_sha256,
                'episode_id': manifest.episode_id,
                'failure_code': result.failure_code.value,
                'terminal_code': result.record.terminal_code.value if result.record.terminal_code else None,
                'attempt_consumed': True,
                'retry_permitted': False,
                'development_only': True,
                'official_execution_qualified': False,
                'leaderboard_admitted': False,
            }
        )
        raise SystemExit(1)
    except ClinicalOperatorError as error:
        sys.stderr.write(f'canonical clinical operator rejected: {error}\n')
        raise SystemExit(64) from error
    except (OSError, ValueError) as error:
        # Never print nested provider, task-content, or secret-bearing exception details.
        sys.stderr.write('canonical clinical operator rejected: authenticated configuration is invalid\n')
        raise SystemExit(64) from error
    except Exception as error:
        # Runtime-boundary exceptions are deliberately collapsed so a traceback cannot disclose
        # task, provider, filesystem, or secret-adjacent state to the invoking environment.
        sys.stderr.write('canonical clinical operator rejected: bounded execution failed\n')
        raise SystemExit(70) from error
    finally:
        if inputs is not None:
            inputs.secrets.close()


def _write(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
