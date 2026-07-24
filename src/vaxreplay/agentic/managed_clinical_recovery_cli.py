"""No-argument cleanup-only entrypoint for managed Lane A crash recovery."""

from __future__ import annotations

import sys

from vaxreplay.agentic.managed_clinical_deployment import (
    ManagedClinicalDeploymentError,
    execute_fixed_managed_clinical_recovery,
)
from vaxreplay.bundle import canonical_json_bytes


def main() -> None:
    """Recover only the compiled-in deployment; no task or path arguments are accepted."""

    if len(sys.argv) != 1:
        sys.stderr.write('managed clinical recovery rejected: this entrypoint accepts no arguments\n')
        raise SystemExit(64)
    try:
        result = execute_fixed_managed_clinical_recovery()
        sys.stdout.write(canonical_json_bytes(result).decode('utf-8') + '\n')
    except ManagedClinicalDeploymentError as error:
        sys.stderr.write(f'managed clinical recovery rejected: {error}\n')
        raise SystemExit(64) from error
    except (OSError, ValueError) as error:
        sys.stderr.write('managed clinical recovery rejected: root-owned configuration is invalid\n')
        raise SystemExit(64) from error
    except Exception as error:
        # Do not disclose task, provider, secret, or filesystem-adjacent nested exceptions.
        sys.stderr.write('managed clinical recovery rejected: bounded cleanup failed\n')
        raise SystemExit(70) from error


if __name__ == '__main__':
    main()
