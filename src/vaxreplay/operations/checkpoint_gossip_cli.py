"""Operator CLI for independently deployed checkpoint gossip monitors."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.checkpoint_gossip import (
    CheckpointGossipError,
    CheckpointGossipMonitorPolicy,
    CheckpointGossipMonitorStore,
    verify_gossip_agreement,
)
from vaxreplay.operations.clock_health import ClockHealthGate
from vaxreplay.operations.operator_trust import (
    OperatorTrustError,
    load_clock_health_gate,
    load_external_signer,
)
from vaxreplay.operations.signing import Ed25519Signer

_MAX_POLICY_BYTES = 16 * 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Ratchet and compare selection-registry and witness signed checkpoints'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    generate_key = subparsers.add_parser(
        'generate-report-key',
        help='generate an owner-only Ed25519 monitor-report key',
    )
    generate_key.add_argument('--output', required=True)

    initialize = subparsers.add_parser(
        'init',
        help='create a no-replace monitor root from exact policy and private-key bytes',
    )
    initialize.add_argument('--root', required=True)
    initialize.add_argument('--policy', required=True)
    signer_mode = initialize.add_mutually_exclusive_group(required=True)
    signer_mode.add_argument(
        '--dev-report-signing-private-key',
        help='development-only owner-protected raw 32-byte Ed25519 key',
    )
    signer_mode.add_argument('--external-signer-process')
    _external_trust_arguments(initialize)

    observe = subparsers.add_parser(
        'observe',
        help='verify and append one latest source-signed checkpoint',
    )
    observe.add_argument('--root', required=True)
    observe.add_argument('--stream-id', required=True)
    observe.add_argument('--checkpoint', required=True)
    observe.add_argument(
        '--registry-consistency-hash',
        action='append',
        default=[],
        metavar='LOWERCASE_SHA256',
        help='ordered RFC6962 consistency path; repeat once per proof node',
    )
    _stored_runtime_trust_arguments(observe)

    verify = subparsers.add_parser(
        'verify',
        help='replay every signature, source transition, and local journal link',
    )
    verify.add_argument('--root', required=True)
    _stored_runtime_trust_arguments(verify)

    report = subparsers.add_parser(
        'report',
        help='write one fresh signed all-stream monitor report',
    )
    report.add_argument('--root', required=True)
    report.add_argument('--output', required=True)
    _stored_runtime_trust_arguments(report)

    compare = subparsers.add_parser(
        'compare',
        help='authenticate a complete monitor quorum and require exact latest-head agreement',
    )
    compare.add_argument('--comparison-policy', required=True)
    compare.add_argument('--report', action='append', required=True)
    return parser


def _external_trust_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--external-signer-public-key')
    parser.add_argument('--external-signer-process-sha256')
    parser.add_argument('--clock-health-policy')
    parser.add_argument('--clock-health-policy-sha256')
    parser.add_argument('--clock-health-process')
    parser.add_argument('--clock-health-process-sha256')


def _stored_runtime_trust_arguments(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--dev-local-root-key',
        action='store_true',
        help='development-only: use the private key persisted in the monitor root',
    )
    mode.add_argument('--external-signer-process')
    _external_trust_arguments(parser)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == 'generate-report-key':
            private_key = Ed25519PrivateKey.generate()
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            output = _write_exclusive(Path(args.output), private_bytes, mode=0o600, protected_parent=True)
            public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            _write_stdout(
                {
                    'output': str(output),
                    'public_key_base64': base64.b64encode(public_key).decode('ascii'),
                    'public_key_sha256': hashlib.sha256(public_key).hexdigest(),
                }
            )
            return
        if args.command == 'init':
            policy_bytes = _read_bounded(Path(args.policy), _MAX_POLICY_BYTES, private=False)
            policy = CheckpointGossipMonitorPolicy.model_validate_json(policy_bytes)
            if policy_bytes != canonical_json_bytes(policy):
                raise CheckpointGossipError('gossip monitor policy must use canonical JSON')
            _require_runtime_trust_binding(policy, args)
            signer, clock_health_gate = _runtime_trust(args)
            private_key_bytes = None
            if args.dev_report_signing_private_key is not None:
                private_key_bytes = _read_bounded(
                    Path(args.dev_report_signing_private_key),
                    32,
                    private=True,
                )
            store = CheckpointGossipMonitorStore.initialize(
                Path(args.root),
                policy=policy,
                report_signing_private_key=private_key_bytes,
                signer=signer,
                clock_health_gate=clock_health_gate,
            )
            _write_stdout(
                {
                    'monitor_id': store.policy.monitor_id,
                    'policy_sha256': store.policy_sha256,
                    'root': str(store.root),
                    'signer_mode': 'external' if signer is not None else 'development-local',
                    'clock_health_policy_sha256': args.clock_health_policy_sha256,
                    **store.verify().model_dump(mode='json'),
                }
            )
            return

        store = None
        if args.command in {'observe', 'verify', 'report'}:
            signer, clock_health_gate = _runtime_trust(args)
            store = CheckpointGossipMonitorStore(
                Path(args.root),
                signer=signer,
                clock_health_gate=clock_health_gate,
            )
            _require_runtime_trust_binding(store.policy, args)
        if args.command == 'observe':
            assert store is not None
            observation = store.observe(
                args.stream_id,
                _read_bounded(Path(args.checkpoint), 16 * 1024 * 1024, private=False),
                registry_consistency_proof_sha256=tuple(args.registry_consistency_hash),
            )
            _write_stdout(observation.model_dump(mode='json'))
        elif args.command == 'verify':
            assert store is not None
            _write_stdout(store.verify().model_dump(mode='json'))
        elif args.command == 'report':
            assert store is not None
            payload = canonical_json_bytes(store.signed_report())
            output = _write_exclusive(Path(args.output), payload, mode=0o644)
            _write_stdout(
                {
                    'output': str(output),
                    'sha256': hashlib.sha256(payload).hexdigest(),
                }
            )
        elif args.command == 'compare':
            comparison_policy_bytes = _read_bounded(
                Path(args.comparison_policy),
                _MAX_POLICY_BYTES,
                private=False,
            )
            report_bytes = tuple(_read_bounded(Path(path), _MAX_REPORT_BYTES, private=False) for path in args.report)
            _write_stdout(
                verify_gossip_agreement(
                    report_bytes,
                    comparison_policy_bytes,
                ).model_dump(mode='json')
            )
    except (OSError, ValueError, CheckpointGossipError, OperatorTrustError) as error:
        _write_stderr({'error': str(error), 'status': 'failed'})
        raise SystemExit(2) from error


def _runtime_trust(args: argparse.Namespace) -> tuple[Ed25519Signer | None, ClockHealthGate | None]:
    clock_selected = any(
        value is not None
        for value in (
            args.clock_health_policy,
            args.clock_health_policy_sha256,
            args.clock_health_process,
            args.clock_health_process_sha256,
        )
    )
    if clock_selected and (
        args.clock_health_policy is None
        or args.clock_health_policy_sha256 is None
        or args.clock_health_process is None
        or args.clock_health_process_sha256 is None
    ):
        raise CheckpointGossipError('clock-health policy and process configuration require both trusted digests')
    gate = None
    if clock_selected:
        gate = load_clock_health_gate(
            policy_path=Path(args.clock_health_policy),
            policy_sha256=args.clock_health_policy_sha256,
            process_config=Path(args.clock_health_process),
            process_config_sha256=args.clock_health_process_sha256,
        )
    if args.external_signer_process is not None:
        if args.external_signer_public_key is None or args.external_signer_process_sha256 is None:
            raise CheckpointGossipError('external signer public key and trusted process digest are required')
        if gate is None:
            raise CheckpointGossipError('external signer mode requires a clock-health gate')
        return (
            load_external_signer(
                process_config=Path(args.external_signer_process),
                process_config_sha256=args.external_signer_process_sha256,
                public_key=Path(args.external_signer_public_key),
            ),
            gate,
        )
    local_selected = bool(
        getattr(args, 'dev_report_signing_private_key', None) or getattr(args, 'dev_local_root_key', False)
    )
    if not local_selected:
        raise CheckpointGossipError('an explicit gossip signer mode is required')
    if args.external_signer_public_key is not None or args.external_signer_process_sha256 is not None:
        raise CheckpointGossipError('external signer bindings require external signer mode')
    return None, gate


def _require_runtime_trust_binding(
    policy: CheckpointGossipMonitorPolicy,
    args: argparse.Namespace,
) -> None:
    stored = (
        policy.clock_health_policy_sha256,
        policy.clock_health_process_sha256,
        policy.external_signer_process_sha256,
    )
    supplied = (
        args.clock_health_policy_sha256,
        args.clock_health_process_sha256,
        args.external_signer_process_sha256,
    )
    if stored != supplied:
        raise CheckpointGossipError('supplied runtime-trust digests differ from the persisted gossip policy')


def _read_bounded(path: Path, maximum: int, *, private: bool) -> bytes:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(requested, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CheckpointGossipError(f'input must be a regular file: {requested}')
        if private and metadata.st_mode & 0o077:
            raise CheckpointGossipError(f'private-key input must be owner-only: {requested}')
        if metadata.st_size < 1 or metadata.st_size > maximum:
            raise CheckpointGossipError(f'input has an invalid size: {requested}')
        payload = os.read(descriptor, maximum + 1)
        if len(payload) != metadata.st_size:
            raise CheckpointGossipError(f'input changed while being read: {requested}')
        return payload
    finally:
        os.close(descriptor)


def _write_exclusive(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    protected_parent: bool = False,
) -> Path:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    requested.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = requested.parent.resolve(strict=True)
    metadata = parent.stat()
    if protected_parent and (
        not stat.S_ISDIR(metadata.st_mode)
        or (hasattr(os, 'getuid') and metadata.st_uid != os.getuid())
        or metadata.st_mode & 0o022
    ):
        raise CheckpointGossipError('private-key output parent must be owner-controlled')
    target = parent / requested.name
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
        mode,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count < 1:
                raise OSError('short write while exporting gossip artifact')
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def _write_stdout(value: object) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b'\n')


def _write_stderr(value: object) -> None:
    sys.stderr.buffer.write(canonical_json_bytes(value) + b'\n')


if __name__ == '__main__':
    main()
