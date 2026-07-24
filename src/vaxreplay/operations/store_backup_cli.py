"""Operator CLI for OperationalStore backup, restore, and orphan inventory."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.schema import LedgerCheckpoint
from vaxreplay.operations.store import OperationalStore
from vaxreplay.operations.store_backup import (
    audit_store_orphans,
    create_store_backup,
    restore_store_backup,
    verify_store_backup,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Back up and recover a VaxReplay OperationalStore')
    subparsers = parser.add_subparsers(dest='command', required=True)

    create = subparsers.add_parser('create', help='create one consistent immutable backup directory')
    create.add_argument('--root', required=True)
    create.add_argument('--output', required=True)
    create.add_argument('--backup-id', required=True)
    create.add_argument('--checkpoint')

    verify = subparsers.add_parser('verify', help='verify exact bytes and exercise a clean-root restore')
    verify.add_argument('--backup', required=True)

    restore = subparsers.add_parser('restore', help='restore into a nonexistent root and fully replay it')
    restore.add_argument('--backup', required=True)
    restore.add_argument('--root', required=True)

    audit = subparsers.add_parser('orphan-audit', help='read-only CAS-to-ledger inventory; never deletes files')
    audit.add_argument('--root', required=True)
    audit.add_argument('--max-entries', type=int, default=10_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == 'create':
            checkpoint = None
            if arguments.checkpoint:
                checkpoint = LedgerCheckpoint.model_validate_json(Path(arguments.checkpoint).read_bytes())
            result = create_store_backup(
                OperationalStore(Path(arguments.root)),
                Path(arguments.output),
                backup_id=arguments.backup_id,
                checkpoint=checkpoint,
            )
        elif arguments.command == 'verify':
            result = verify_store_backup(Path(arguments.backup))
        elif arguments.command == 'restore':
            result = restore_store_backup(Path(arguments.backup), Path(arguments.root))
        else:
            result = audit_store_orphans(
                OperationalStore(Path(arguments.root)),
                max_entries=arguments.max_entries,
            )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f'operations-store {arguments.command} failed: {error}', file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result) + b'\n')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
