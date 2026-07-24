"""No-argument entrypoint for the fixed managed Lane A one-shot service."""

from __future__ import annotations

import sys

from vaxreplay.agentic.clinical_launcher import ClinicalLauncherFailure, ClinicalLauncherSuccess
from vaxreplay.agentic.managed_clinical_deployment import (
    ManagedClinicalDeploymentError,
    execute_fixed_managed_clinical_task,
)
from vaxreplay.bundle import canonical_json_bytes


def main() -> None:
    """Run only the compiled-in root-owned deployment; no caller path overrides exist."""

    if len(sys.argv) != 1:
        sys.stderr.write('managed clinical deployment rejected: this entrypoint accepts no arguments\n')
        raise SystemExit(64)
    try:
        result = execute_fixed_managed_clinical_task()
        if isinstance(result, ClinicalLauncherSuccess):
            _write(
                {
                    'status': 'succeeded',
                    'run_id': result.launch.run_id,
                    'reservation_sha256': result.launch.reservation_sha256,
                    'episode_id': result.launch.episode_id,
                    'evidence_sha256': result.record.evidence_sha256,
                    'attempt_consumed': True,
                    'retry_permitted': False,
                    'managed_one_host_authority': True,
                    'live_deployment_qualification_claimed': False,
                    'leaderboard_admitted': False,
                }
            )
            return
        if not isinstance(result, ClinicalLauncherFailure):
            raise ManagedClinicalDeploymentError('managed operator returned an unknown result type')
        _write(
            {
                'status': 'failed',
                'run_id': result.launch.run_id,
                'reservation_sha256': result.launch.reservation_sha256,
                'episode_id': result.launch.episode_id,
                'failure_code': result.failure_code.value,
                'terminal_code': (
                    result.record.terminal_code.value if result.record.terminal_code is not None else None
                ),
                'attempt_consumed': True,
                'retry_permitted': False,
                'managed_one_host_authority': True,
                'live_deployment_qualification_claimed': False,
                'leaderboard_admitted': False,
            }
        )
        raise SystemExit(1)
    except ManagedClinicalDeploymentError as error:
        sys.stderr.write(f'managed clinical deployment rejected: {error}\n')
        raise SystemExit(64) from error
    except (OSError, ValueError) as error:
        sys.stderr.write('managed clinical deployment rejected: root-owned configuration is invalid\n')
        raise SystemExit(64) from error
    except Exception as error:
        # Do not disclose task, provider, secret, or filesystem-adjacent nested exceptions.
        sys.stderr.write('managed clinical deployment rejected: bounded execution failed\n')
        raise SystemExit(70) from error


def _write(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
