"""Operator CLI for the separately deployable checkpoint witness service."""

from __future__ import annotations

import argparse
import hashlib
import os
import ssl
import sys
from collections.abc import Sequence
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.clock_health import ClockHealthGate
from vaxreplay.operations.operator_trust import (
    OperatorTrustError,
    load_clock_health_gate,
    load_external_signer,
)
from vaxreplay.operations.signing import Ed25519Signer
from vaxreplay.operations.witness_service import (
    WitnessServiceError,
    WitnessServiceStore,
    build_witness_http_server,
)
from vaxreplay.operations.witness_service_schema import WitnessRegistryMonitor, WitnessServicePolicy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Operate an external VaxReplay checkpoint witness service')
    subparsers = parser.add_subparsers(dest='command', required=True)

    initialize = subparsers.add_parser('init', help='create a no-replace service root and signing key')
    initialize.add_argument('--root', required=True)
    initialize.add_argument('--authority-id', required=True)
    initialize.add_argument('--witness-id', required=True)
    initialize.add_argument('--policy-id', required=True)
    initialize.add_argument('--trust-policy-id', required=True)
    initialize.add_argument('--endpoint-uri', required=True)
    initialize.add_argument('--max-submission-bytes', type=int, default=64 * 1024)
    initialize.add_argument('--max-proof-bytes', type=int, default=1024 * 1024)
    initialize.add_argument('--client-timeout-seconds', type=float, default=15.0)
    initialize.add_argument(
        '--registry-monitor',
        action='append',
        default=[],
        metavar='CANONICAL_JSON',
        help='repeatable pinned registry identity/key-ring monitor configuration',
    )
    _runtime_trust_arguments(initialize, initialize=True)

    issue = subparsers.add_parser('issue', help='issue from an exact canonical submission on the service host')
    issue.add_argument('--root', required=True)
    issue.add_argument('--submission', required=True)
    issue.add_argument('--output', required=True)
    _runtime_trust_arguments(issue)

    checkpoint = subparsers.add_parser('checkpoint', help='export the latest public signed log checkpoint')
    checkpoint.add_argument('--root', required=True)
    checkpoint.add_argument('--output', required=True)
    _runtime_trust_arguments(checkpoint)

    verify = subparsers.add_parser('verify', help='replay every durable row, signature, and hash-chain link')
    verify.add_argument('--root', required=True)
    _runtime_trust_arguments(verify)

    serve = subparsers.add_parser('serve', help='serve authenticated bounded POSTs and public proof reads')
    serve.add_argument('--root', required=True)
    serve.add_argument('--host', required=True)
    serve.add_argument('--port', type=int, required=True)
    serve.add_argument(
        '--write-token-env',
        required=True,
        help='environment variable holding the bearer token; its value is removed from this process environment',
    )
    serve.add_argument('--tls-cert')
    serve.add_argument('--tls-key')
    serve.add_argument(
        '--allow-insecure-loopback',
        action='store_true',
        help='explicit development-only switch; never permits plaintext on a non-loopback bind',
    )
    _runtime_trust_arguments(serve)
    return parser


def _runtime_trust_arguments(parser: argparse.ArgumentParser, *, initialize: bool = False) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--dev-local-root-key',
        action='store_true',
        help=(
            'development-only: generate and persist a local key in the root'
            if initialize
            else 'development-only: use the private key persisted in the root'
        ),
    )
    mode.add_argument('--external-signer-process')
    parser.add_argument('--external-signer-public-key')
    parser.add_argument('--external-signer-process-sha256')
    parser.add_argument('--clock-health-policy')
    parser.add_argument('--clock-health-policy-sha256')
    parser.add_argument('--clock-health-process')
    parser.add_argument('--clock-health-process-sha256')


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        signer, clock_health_gate = _runtime_trust(args)
        if args.command == 'init':
            registry_monitors = tuple(_load_canonical_registry_monitor(Path(path)) for path in args.registry_monitor)
            store = WitnessServiceStore.initialize(
                Path(args.root),
                authority_id=args.authority_id,
                witness_id=args.witness_id,
                policy_id=args.policy_id,
                trust_policy_id=args.trust_policy_id,
                endpoint_uri=args.endpoint_uri,
                max_submission_bytes=args.max_submission_bytes,
                max_proof_bytes=args.max_proof_bytes,
                client_timeout_seconds=args.client_timeout_seconds,
                registry_monitors=registry_monitors,
                clock_health_policy_sha256=args.clock_health_policy_sha256,
                clock_health_process_sha256=args.clock_health_process_sha256,
                external_signer_process_sha256=args.external_signer_process_sha256,
                signer=signer,
                clock_health_gate=clock_health_gate,
            )
            _require_runtime_trust_binding(store.policy, args)
            report = store.verify()
            _write_stdout(
                {
                    'authority_id': store.policy.authority_id,
                    'endpoint_uri': store.policy.endpoint_uri,
                    'policy_path': str(store.root / 'policy.json'),
                    'root': str(store.root),
                    'trust_policy_path': str(store.root / 'trust-policy.json'),
                    'witness_id': store.policy.witness_id,
                    **report.model_dump(mode='json'),
                    'signer_mode': 'external' if signer is not None else 'development-local',
                    'clock_health_policy_sha256': args.clock_health_policy_sha256,
                }
            )
            return
        store = WitnessServiceStore(
            Path(args.root),
            signer=signer,
            clock_health_gate=clock_health_gate,
        )
        _require_runtime_trust_binding(store.policy, args)
        if args.command == 'issue':
            issuance = store.issue(Path(args.submission).read_bytes())
            output = _write_exclusive(Path(args.output), issuance.proof_bytes)
            _write_stdout(
                {
                    'created': issuance.created,
                    'output': str(output),
                    'receipt_id': issuance.receipt_id,
                    'sequence': issuance.sequence,
                }
            )
        elif args.command == 'checkpoint':
            payload = store.latest_signed_checkpoint_bytes()
            output = _write_exclusive(Path(args.output), payload)
            _write_stdout({'output': str(output), 'sha256': hashlib.sha256(payload).hexdigest()})
        elif args.command == 'verify':
            _write_stdout(store.verify().model_dump(mode='json'))
        elif args.command == 'serve':
            _serve(store, args)
    except (OSError, ValueError, WitnessServiceError, OperatorTrustError) as error:
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
        raise WitnessServiceError('clock-health policy and process configuration require both trusted digests')
    clock_health_gate = None
    if clock_selected:
        clock_health_gate = load_clock_health_gate(
            policy_path=Path(args.clock_health_policy),
            policy_sha256=args.clock_health_policy_sha256,
            process_config=Path(args.clock_health_process),
            process_config_sha256=args.clock_health_process_sha256,
        )
    if args.external_signer_process is not None:
        if args.external_signer_public_key is None or args.external_signer_process_sha256 is None:
            raise WitnessServiceError('external signer public key and trusted process digest are required')
        if clock_health_gate is None:
            raise WitnessServiceError('external signer mode requires a clock-health gate')
        return (
            load_external_signer(
                process_config=Path(args.external_signer_process),
                process_config_sha256=args.external_signer_process_sha256,
                public_key=Path(args.external_signer_public_key),
            ),
            clock_health_gate,
        )
    if not args.dev_local_root_key:
        raise WitnessServiceError('an explicit witness signer mode is required')
    if args.external_signer_public_key is not None or args.external_signer_process_sha256 is not None:
        raise WitnessServiceError('external signer bindings require external signer mode')
    return None, clock_health_gate


def _require_runtime_trust_binding(
    policy: WitnessServicePolicy,
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
        raise WitnessServiceError('supplied runtime-trust digests differ from the persisted witness policy')


def _serve(store: WitnessServiceStore, args: argparse.Namespace) -> None:
    if not args.write_token_env or any(character in args.write_token_env for character in '\x00=\r\n'):
        raise WitnessServiceError('write-token environment-variable name is invalid')
    token_text = os.environ.pop(args.write_token_env, None)
    if token_text is None:
        raise WitnessServiceError('write-token environment variable is not set')
    try:
        token = token_text.encode('ascii')
    except UnicodeEncodeError as error:
        raise WitnessServiceError('write-token environment variable must contain visible ASCII') from error
    tls_context: ssl.SSLContext | None = None
    if (args.tls_cert is None) != (args.tls_key is None):
        raise WitnessServiceError('--tls-cert and --tls-key must be supplied together')
    if args.tls_cert is not None:
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        tls_context.load_cert_chain(certfile=args.tls_cert, keyfile=args.tls_key)
    server = build_witness_http_server(
        store,
        host=args.host,
        port=args.port,
        authorization_bearer_token=token,
        tls_context=tls_context,
        allow_insecure_loopback=args.allow_insecure_loopback,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def _write_exclusive(path: Path, payload: bytes) -> Path:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count < 1:
                raise OSError('short write while exporting witness artifact')
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def _load_canonical_registry_monitor(path: Path) -> WitnessRegistryMonitor:
    payload = path.read_bytes()
    try:
        monitor = WitnessRegistryMonitor.model_validate_json(payload)
    except ValueError as error:
        raise WitnessServiceError(f'invalid registry monitor file {path}: {error}') from error
    if payload != canonical_json_bytes(monitor):
        raise WitnessServiceError(f'registry monitor file must use canonical JSON: {path}')
    return monitor


def _write_stdout(value: object) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b'\n')


def _write_stderr(value: object) -> None:
    sys.stderr.buffer.write(canonical_json_bytes(value) + b'\n')


if __name__ == '__main__':
    main()
