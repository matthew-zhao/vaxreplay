"""Offline CLI for machine-checked Tier A release readiness."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.release_readiness import (
    TierAReleaseReadinessManifest,
    TierAReleaseScope,
    applicable_gate_ids,
    verify_tier_a_release_readiness,
)

_MAX_EVIDENCE_BYTES = 8 * 1024 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Inspect or verify applicable Tier A release gates')
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify', help='verify signed evidence for every applicable gate')
    verify.add_argument('--policy', required=True)
    verify.add_argument('--expected-policy-sha256', required=True)
    verify.add_argument('--manifest', required=True)
    verify.add_argument('--verified-at', required=True)
    verify.add_argument('--verification-time-evidence', required=True)
    verify.add_argument('--verification-time-public-key', required=True)
    verify.add_argument(
        '--subject',
        action='append',
        default=[],
        metavar='ROLE=PATH',
        help='exact release subject; repeat for every manifest subject',
    )
    verify.add_argument(
        '--evidence-root',
        required=True,
        help='flat directory whose filenames are exact evidence SHA-256 digests',
    )
    verify.add_argument(
        '--authority-key',
        action='append',
        default=[],
        metavar='AUTHORITY_ID=PATH',
        help='raw 32-byte Ed25519 public key; repeat for every policy authority',
    )
    list_gates = commands.add_parser(
        'list-gates',
        help='list the exact code-derived gate inventory for a canonical scope',
    )
    list_gates.add_argument('--scope', required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == 'list-gates':
        return _list_gates(Path(arguments.scope))
    try:
        policy_bytes = _read_regular(Path(arguments.policy), maximum=8 * 1024 * 1024)
        manifest_bytes = _read_regular(Path(arguments.manifest), maximum=64 * 1024 * 1024)
        manifest = TierAReleaseReadinessManifest.model_validate_json(manifest_bytes)
        subjects = _read_named_paths(arguments.subject, 'release subject')
        evidence_digests = {item.statement.evidence_artifact.sha256 for item in manifest.evidence}
        evidence = _read_evidence_root(Path(arguments.evidence_root), evidence_digests)
        keys = _read_authority_keys(arguments.authority_key)
        verification_time_evidence = _read_regular(
            Path(arguments.verification_time_evidence),
            maximum=1024 * 1024,
        )
        verification_time_public_key = _read_regular(
            Path(arguments.verification_time_public_key),
            maximum=32,
        )
        report = verify_tier_a_release_readiness(
            policy_bytes=policy_bytes,
            expected_policy_sha256=arguments.expected_policy_sha256,
            manifest_bytes=manifest_bytes,
            release_subject_bytes=subjects,
            evidence_artifact_bytes=evidence,
            authority_public_key_bytes=keys,
            verification_time_evidence_bytes=verification_time_evidence,
            verification_time_public_key_bytes=verification_time_public_key,
            verified_at=_parse_time(arguments.verified_at),
        )
    except (OSError, TypeError, ValueError) as error:
        print(f'Tier A release readiness verification failed: {error}', file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(report) + b'\n')
    return 0


def _list_gates(scope_path: Path) -> int:
    try:
        scope_bytes = _read_regular(scope_path, maximum=1024 * 1024)
        scope = TierAReleaseScope.model_validate_json(scope_bytes)
        if scope_bytes != canonical_json_bytes(scope):
            raise ValueError('release scope must use exact canonical JSON bytes')
        gate_ids = applicable_gate_ids(scope)
    except (OSError, TypeError, ValueError) as error:
        print(f'Tier A release gate listing failed: {error}', file=sys.stderr)
        return 2
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                'applicable_gate_count': len(gate_ids),
                'gate_ids': gate_ids,
                'scope': scope.model_dump(mode='json'),
                'scope_sha256': hashlib.sha256(scope_bytes).hexdigest(),
            }
        )
        + b'\n'
    )
    return 0


def _read_authority_keys(values: list[str]) -> dict[str, bytes]:
    return _read_named_paths(values, 'authority key', maximum=32)


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('verified-at must be an RFC 3339 timestamp') from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('verified-at must include a UTC offset')
    return parsed


def _read_named_paths(
    values: list[str],
    label: str,
    *,
    maximum: int = _MAX_EVIDENCE_BYTES,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for item in values:
        name, separator, path = item.partition('=')
        if not separator or not name or name in result:
            raise ValueError(f'{label} arguments must use unique NAME=PATH bindings')
        result[name] = _read_regular(Path(path), maximum=maximum)
    return result


def _read_evidence_root(root: Path, expected_digests: set[str]) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError('evidence root must be a non-symlink directory')
    result: dict[str, bytes] = {}
    for entry in os.scandir(root):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError('evidence root may contain only regular non-symlink files')
        if re.fullmatch(r'[0-9a-f]{64}', entry.name) is None or entry.name in result:
            raise ValueError('evidence filenames must be unique lowercase SHA-256 digests')
        result[entry.name] = _read_regular(Path(entry.path), maximum=_MAX_EVIDENCE_BYTES)
    if set(result) != expected_digests:
        raise ValueError('evidence-root inventory differs from the readiness manifest')
    return result


def _read_regular(path: Path, *, maximum: int) -> bytes:
    if path.is_symlink():
        raise ValueError(f'input cannot be a symbolic link: {path}')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum:
            raise ValueError(f'input must be a bounded regular file: {path}')
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f'input changed while being read: {path}')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f'input changed while being read: {path}')
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f'input changed while being read: {path}')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
