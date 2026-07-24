"""CLI for deterministic offline ImmPort/ClinicalTrials.gov feasibility inventories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.feasibility.inventory import (
    audit_inventory,
    build_inventory,
    export_public_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Inventory pinned ImmPort and ClinicalTrials.gov metadata')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build = subparsers.add_parser('build')
    build.add_argument('--spec', required=True)
    build.add_argument('--immport-root', required=True)
    build.add_argument('--ctgov-root', required=True)
    build.add_argument('--output-dir', required=True)

    audit = subparsers.add_parser('audit')
    audit.add_argument('--inventory-dir', required=True)

    public = subparsers.add_parser('export-summary')
    public.add_argument('--inventory-dir', required=True)
    public.add_argument('--output', required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == 'build':
        report = build_inventory(
            spec_path=Path(args.spec),
            immport_root=Path(args.immport_root),
            ctgov_root=Path(args.ctgov_root),
            output_root=Path(args.output_dir),
        )
    elif args.command == 'audit':
        report = audit_inventory(Path(args.inventory_dir))
    else:
        report = export_public_summary(Path(args.inventory_dir), Path(args.output))
    sys.stdout.write(canonical_json_bytes(report).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
