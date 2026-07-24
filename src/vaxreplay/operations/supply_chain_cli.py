"""Offline CLI for retained source-worker supply-chain bundles."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.supply_chain import verify_source_worker_supply_chain

_MAX_MATERIAL_BYTES = 8 * 1024 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Verify a retained VaxReplay source-worker OCI build offline')
    parser.add_argument('--source-archive', required=True)
    parser.add_argument('--primary-package', required=True)
    parser.add_argument('--runtime-lock', required=True)
    parser.add_argument('--build-recipe', required=True)
    parser.add_argument('--sbom', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--oci-manifest', required=True)
    parser.add_argument('--oci-config', required=True)
    parser.add_argument(
        '--component',
        action='append',
        default=[],
        metavar='SHA256=PATH',
        help='exact runtime distribution; repeat for every SBOM component',
    )
    parser.add_argument(
        '--layer',
        action='append',
        default=[],
        metavar='SHA256=PATH',
        help='exact compressed OCI layer; repeat for every manifest layer',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = verify_source_worker_supply_chain(
            source_archive_bytes=_read_regular(Path(arguments.source_archive)),
            primary_package_bytes=_read_regular(Path(arguments.primary_package)),
            runtime_lock_bytes=_read_regular(Path(arguments.runtime_lock)),
            build_recipe_bytes=_read_regular(Path(arguments.build_recipe)),
            sbom_bytes=_read_regular(Path(arguments.sbom)),
            provenance_bytes=_read_regular(Path(arguments.provenance)),
            oci_manifest_bytes=_read_regular(Path(arguments.oci_manifest)),
            oci_config_bytes=_read_regular(Path(arguments.oci_config)),
            component_distribution_bytes=_load_digest_paths(arguments.component, 'component'),
            oci_layer_bytes=_load_digest_paths(arguments.layer, 'layer'),
        )
    except (OSError, TypeError, ValueError) as error:
        print(f'source-worker supply-chain verification failed: {error}', file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(report) + b'\n')
    return 0


def _load_digest_paths(values: list[str], label: str) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for value in values:
        digest, separator, path = value.partition('=')
        if not separator or not path or digest in result:
            raise ValueError(f'{label} arguments must use unique SHA256=PATH bindings')
        result[digest] = _read_regular(Path(path))
    return result


def _read_regular(path: Path) -> bytes:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f'input cannot be a symbolic link: {requested}')
    descriptor = os.open(requested, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > _MAX_MATERIAL_BYTES:
            raise ValueError(f'input must be a bounded regular file: {requested}')
        with os.fdopen(descriptor, 'rb', closefd=False) as source:
            payload = source.read(_MAX_MATERIAL_BYTES + 1)
        if len(payload) > _MAX_MATERIAL_BYTES:
            raise ValueError(f'input exceeds {_MAX_MATERIAL_BYTES} bytes: {requested}')
        final = os.fstat(descriptor)
        if (
            len(payload) != initial.st_size
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
            or final.st_ctime_ns != initial.st_ctime_ns
        ):
            raise ValueError(f'input changed size or metadata while being read: {requested}')
        return payload
    finally:
        os.close(descriptor)


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
