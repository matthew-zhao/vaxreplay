"""Authenticated bounded HTTP service and operator CLI for the selection registry."""

from __future__ import annotations

import argparse
import base64
import binascii
import http.client
import os
import re
import ssl
import stat
import sys
from collections.abc import Sequence
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlsplit

from vaxreplay._atomic import fsync_directory
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.operator_trust import (
    add_signer_arguments,
    load_clock_health_gate,
    load_external_signer,
)
from vaxreplay.operations.plan_selection import PlanSelectionPolicyBinding, PlanSelectionRequest
from vaxreplay.operations.schema import SAFE_ID_PATTERN
from vaxreplay.operations.selection_registry import (
    RegistryConflictError,
    RegistrySelectionResponse,
    RegistryWitnessUnavailableError,
    SelectionRegistryError,
    SelectionRegistryPolicy,
    SelectionRegistryTrustPolicy,
    SQLitePlanSelectionRegistry,
    ed25519_public_key_base64,
    generate_ed25519_private_key,
    load_ed25519_private_key,
    verify_registry_selection,
    verify_service_bearer_token,
)
from vaxreplay.operations.witness_service import Ed25519WitnessServiceProvider

_SAFE_ID = re.compile(SAFE_ID_PATTERN)
_MAX_PATH_BYTES = 4096


class SelectionRegistryHTTPServer(ThreadingHTTPServer):
    """Threaded server holding a process-local registry and no secret token."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        server_address: tuple[str, int],
        registry: SQLitePlanSelectionRegistry,
        *,
        write_token_sha256: str,
    ) -> None:
        self.registry = registry
        # Validate once at startup. Only the digest is retained by the server.
        verify_service_bearer_token(write_token_sha256, None)
        self.write_token_sha256 = write_token_sha256
        super().__init__(server_address, SelectionRegistryRequestHandler)


class HttpsPlanSelectionRegistryProvider:
    """No-redirect HTTPS client implementing the ``PlanSelectionProvider`` ABI."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str,
        ca_file: Path | None = None,
        client_certificate: Path | None = None,
        client_private_key: Path | None = None,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 24 * 1024 * 1024,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != 'https'
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or base_url.endswith('/')
        ):
            raise SelectionRegistryError('registry provider base URL must be a credential-free HTTPS URL')
        if len(bearer_token) < 32 or len(bearer_token) > 4096 or bearer_token != bearer_token.strip():
            raise SelectionRegistryError('registry bearer token must contain 32 to 4096 trimmed characters')
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise SelectionRegistryError('registry provider timeout must be in (0, 300] seconds')
        if max_response_bytes < 4096 or max_response_bytes > 96 * 1024 * 1024:
            raise SelectionRegistryError('registry provider response limit is outside the safe range')
        if (client_certificate is None) != (client_private_key is None):
            raise SelectionRegistryError('mTLS client certificate and private key must be provided together')
        checked_ca_file = None if ca_file is None else _require_nonwritable_regular_file(ca_file, 'TLS CA file')
        context = ssl.create_default_context(cafile=None if checked_ca_file is None else str(checked_ca_file))
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if client_certificate is not None and client_private_key is not None:
            checked_client_key = _require_owner_only_regular_file(
                client_private_key,
                'mTLS client private key',
            )
            context.load_cert_chain(certfile=client_certificate, keyfile=checked_client_key)
        self._hostname = parsed.hostname
        self._port = parsed.port or 443
        self._path = f'{parsed.path}/v1/selections'
        self._bearer_token = bearer_token
        self._context = context
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    def __call__(self, request: PlanSelectionRequest):
        request = PlanSelectionRequest.model_validate_json(canonical_json_bytes(request))
        payload = canonical_json_bytes(request)
        connection = http.client.HTTPSConnection(
            self._hostname,
            self._port,
            timeout=self._timeout_seconds,
            context=self._context,
        )
        try:
            connection.request(
                'POST',
                self._path,
                body=payload,
                headers={
                    'Authorization': f'Bearer {self._bearer_token}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
            )
            response = connection.getresponse()
            content_length = response.getheader('Content-Length')
            if content_length is None or not content_length.isdecimal():
                raise SelectionRegistryError('registry response omitted a valid Content-Length')
            if int(content_length) <= 0 or int(content_length) > self._max_response_bytes:
                raise SelectionRegistryError('registry response exceeds the configured byte limit')
            body = response.read(self._max_response_bytes + 1)
            if len(body) != int(content_length) or len(body) > self._max_response_bytes:
                raise SelectionRegistryError('registry response was truncated or oversized')
            if response.status != HTTPStatus.OK:
                raise SelectionRegistryError(f'registry service returned HTTP {response.status}')
            parsed = RegistrySelectionResponse.model_validate_json(body)
            if body != canonical_json_bytes(parsed):
                raise SelectionRegistryError('registry service response must use canonical JSON')
            try:
                proof = base64.b64decode(parsed.proof_base64, validate=True)
            except (binascii.Error, ValueError) as error:
                raise SelectionRegistryError('registry response proof is not canonical base64') from error
            if base64.b64encode(proof).decode('ascii') != parsed.proof_base64:
                raise SelectionRegistryError('registry response proof is not canonical base64')
            return parsed.claim, proof
        except (OSError, http.client.HTTPException) as error:
            raise SelectionRegistryError(f'registry HTTPS request failed: {type(error).__name__}') from error
        finally:
            connection.close()


class SelectionRegistryRequestHandler(BaseHTTPRequestHandler):
    """Strict API: authenticated writes and public immutable proof reads."""

    server: SelectionRegistryHTTPServer
    protocol_version = 'HTTP/1.1'
    server_version = 'VaxReplaySelectionRegistry/1'
    sys_version = ''
    _json_content_type: ClassVar[str] = 'application/json'

    def do_POST(self) -> None:  # noqa: N802
        if self._path() != '/v1/selections':
            self._error(HTTPStatus.NOT_FOUND, 'not_found')
            return
        if not verify_service_bearer_token(
            self.server.write_token_sha256,
            self.headers.get('Authorization'),
        ):
            self._error(HTTPStatus.UNAUTHORIZED, 'authentication_required')
            return
        if self.headers.get_content_type() != self._json_content_type:
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, 'application_json_required')
            return
        body = self._read_bounded_body(self.server.registry.policy.max_request_bytes)
        if body is None:
            return
        try:
            request = PlanSelectionRequest.model_validate_json(body)
            if body != canonical_json_bytes(request):
                raise ValueError('request must use canonical JSON')
            claim, proof = self.server.registry.provider(request)
            response = RegistrySelectionResponse(
                claim=claim,
                proof_base64=base64.b64encode(proof).decode('ascii'),
            )
            self._response(HTTPStatus.OK, canonical_json_bytes(response))
        except RegistryConflictError:
            self._error(HTTPStatus.CONFLICT, 'selection_key_already_assigned')
        except RegistryWitnessUnavailableError:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, 'checkpoint_witness_unavailable')
        except (SelectionRegistryError, ValueError):
            self._error(HTTPStatus.BAD_REQUEST, 'invalid_selection_request')

    def do_GET(self) -> None:  # noqa: N802
        path = self._path()
        if path == '/healthz':
            checkpoint = self.server.registry.tree_head()
            self._response(
                HTTPStatus.OK,
                canonical_json_bytes(
                    {
                        'registry_id': self.server.registry.policy.registry_id,
                        'status': 'ok',
                        'tree_size': 0 if checkpoint is None else checkpoint.tree_size,
                    }
                ),
            )
            return
        if path == '/v1/checkpoints/latest':
            checkpoint = self.server.registry.signed_tree_head()
            if checkpoint is None:
                self._error(HTTPStatus.NOT_FOUND, 'checkpoint_not_found')
                return
            self._response(HTTPStatus.OK, canonical_json_bytes(checkpoint))
            return
        entry_prefix = '/v1/entries/'
        if path.startswith(entry_prefix):
            raw_sequence = path[len(entry_prefix) :]
            if (
                not raw_sequence.isascii()
                or not raw_sequence.isdecimal()
                or (len(raw_sequence) > 1 and raw_sequence.startswith('0'))
            ):
                self._error(HTTPStatus.NOT_FOUND, 'entry_not_found')
                return
            try:
                entry_bytes = self.server.registry.registry_entry_bytes(int(raw_sequence))
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, 'entry_not_found')
                return
            except SelectionRegistryError:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, 'registry_integrity_failure')
                return
            self._response(
                HTTPStatus.OK,
                entry_bytes,
                content_type='application/vnd.vaxreplay.registry-entry+json',
                cache_control='public, immutable, max-age=31536000',
            )
            return
        prefix = '/v1/proofs/'
        if not path.startswith(prefix):
            self._error(HTTPStatus.NOT_FOUND, 'not_found')
            return
        components = path[len(prefix) :].split('/')
        if len(components) != 2:
            self._error(HTTPStatus.NOT_FOUND, 'not_found')
            return
        try:
            campaign_id, selection_key = (unquote(component, errors='strict') for component in components)
        except UnicodeError:
            self._error(HTTPStatus.BAD_REQUEST, 'invalid_selection_key')
            return
        if not _SAFE_ID.fullmatch(campaign_id) or not _SAFE_ID.fullmatch(selection_key):
            self._error(HTTPStatus.BAD_REQUEST, 'invalid_selection_key')
            return
        try:
            proof = self.server.registry.proof_for(campaign_id, selection_key)
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, 'proof_not_found')
            return
        except SelectionRegistryError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, 'registry_integrity_failure')
            return
        self._response(HTTPStatus.OK, proof, content_type='application/vnd.vaxreplay.registry-proof+json')

    def do_PUT(self) -> None:  # noqa: N802
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, 'method_not_allowed')

    do_DELETE = do_PUT
    do_PATCH = do_PUT

    def log_message(self, format: str, *args: object) -> None:
        # Deliberately suppress BaseHTTPRequestHandler's raw request logging.
        # A front proxy may log method/path/status, but must redact Authorization.
        return

    def _path(self) -> str:
        if len(self.path.encode('utf-8', errors='replace')) > _MAX_PATH_BYTES:
            return '/__path_too_long__'
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            return '/__query_forbidden__'
        return parsed.path

    def _read_bounded_body(self, maximum: int) -> bytes | None:
        transfer_encoding = self.headers.get('Transfer-Encoding')
        if transfer_encoding is not None:
            self._error(HTTPStatus.BAD_REQUEST, 'chunked_requests_not_supported')
            return None
        raw_length = self.headers.get('Content-Length')
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            length = -1
        if length <= 0 or length > maximum:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, 'invalid_content_length')
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._error(HTTPStatus.BAD_REQUEST, 'truncated_request')
            return None
        return body

    def _error(self, status: HTTPStatus, code: str) -> None:
        self._response(status, canonical_json_bytes({'error': code, 'status': int(status)}))

    def _response(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str = _json_content_type,
        cache_control: str = 'no-store',
    ) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.send_header('Cache-Control', cache_control)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True


def serve_registry(
    registry: SQLitePlanSelectionRegistry,
    *,
    host: str,
    port: int,
    tls_certificate: Path,
    tls_private_key: Path,
    write_token_sha256: str,
) -> None:
    """Serve HTTPS until interrupted; TLS termination is mandatory here."""

    checked_tls_key = _require_owner_only_regular_file(tls_private_key, 'TLS private key')
    server = SelectionRegistryHTTPServer(
        (host, port),
        registry,
        write_token_sha256=write_token_sha256,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=tls_certificate, keyfile=checked_tls_key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Operate the VaxReplay signed plan-selection registry')
    commands = parser.add_subparsers(dest='command', required=True)

    keygen = commands.add_parser('keygen', help='create an owner-only Ed25519 private key file')
    keygen.add_argument('--output', required=True)

    initialize = commands.add_parser('init', help='initialize an exclusive append-only registry database')
    _registry_arguments(initialize, include_database=True)

    serve = commands.add_parser('serve', help='serve authenticated HTTPS writes and public proof reads')
    _registry_arguments(serve, include_database=True)
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, required=True)
    serve.add_argument('--tls-certificate', required=True)
    serve.add_argument('--tls-private-key', required=True)
    serve.add_argument('--write-token-file', required=True)
    serve.add_argument('--witness-write-token-file', required=True)

    register = commands.add_parser('register', help='perform one authenticated-equivalent local assignment')
    _registry_arguments(register, include_database=True)
    register.add_argument('--request', required=True)
    register.add_argument('--proof-output', required=True)
    register.add_argument('--witness-write-token-file', required=True)

    anchor = commands.add_parser(
        'anchor-checkpoint',
        help='retry independent witnessing for a durable signed checkpoint',
    )
    _registry_arguments(anchor, include_database=True)
    anchor.add_argument('--tree-size', type=int, required=True)
    anchor.add_argument('--witness-write-token-file', required=True)
    anchor.add_argument('--proof-output')

    proof = commands.add_parser('proof', help='export an existing public raw proof')
    _registry_arguments(proof, include_database=True)
    proof.add_argument('--campaign-id', required=True)
    proof.add_argument('--selection-key', required=True)
    proof.add_argument('--proof-output', required=True)

    entry = commands.add_parser('entry', help='export one immutable public registry leaf')
    _registry_arguments(entry, include_database=True)
    entry.add_argument('--sequence', type=int, required=True)
    entry.add_argument('--output', required=True)

    add_key = commands.add_parser('add-signing-key', help='append a trust-authorized rotation public key')
    _registry_arguments(add_key, include_database=True)
    add_key.add_argument('--new-signing-key-id', required=True)
    add_key.add_argument('--new-public-key-base64', required=True)
    add_key.add_argument('--registered-at', required=True)

    status = commands.add_parser('status', help='show the latest public signed tree head')
    _registry_arguments(status, include_database=True)

    verify = commands.add_parser('verify-proof', help='offline-verify a raw proof against pinned materials')
    verify.add_argument('--commitment', required=True)
    verify.add_argument('--proof', required=True)
    verify.add_argument('--policy-binding', required=True)
    verify.add_argument('--registry-policy', required=True)
    verify.add_argument('--trust-policy', required=True)
    return parser


def _registry_arguments(parser: argparse.ArgumentParser, *, include_database: bool) -> None:
    if include_database:
        parser.add_argument('--database', required=True)
    add_signer_arguments(parser, dev_required=True)
    parser.add_argument('--signing-key-id', required=True)
    parser.add_argument('--registry-policy', required=True)
    parser.add_argument('--trust-policy', required=True)
    parser.add_argument('--public-base-url', required=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == 'keygen':
        path = generate_ed25519_private_key(Path(args.output))
        _stdout(
            {
                'private_key_path': str(path),
                'private_key_printed': False,
                'public_key_base64': ed25519_public_key_base64(load_ed25519_private_key(path)),
            }
        )
        return
    if args.command == 'verify-proof':
        commitment_bytes = Path(args.commitment).read_bytes()
        proof_bytes = Path(args.proof).read_bytes()
        binding = PlanSelectionPolicyBinding.model_validate_json(Path(args.policy_binding).read_bytes())
        facts = verify_registry_selection(
            commitment_bytes,
            proof_bytes,
            binding,
            Path(args.registry_policy).read_bytes(),
            Path(args.trust_policy).read_bytes(),
        )
        _stdout(facts.model_dump(mode='json'))
        return

    registry_policy_bytes = Path(args.registry_policy).read_bytes()
    registry_policy = SelectionRegistryPolicy.model_validate_json(registry_policy_bytes)
    _require_runtime_trust_binding(registry_policy, args)
    trust_policy_bytes = Path(args.trust_policy).read_bytes()
    clock_health_gate = None
    clock_selected = any(
        value is not None
        for value in (
            args.clock_health_policy,
            args.clock_health_policy_sha256,
            args.clock_health_process,
            args.clock_health_process_sha256,
        )
    )
    if clock_selected:
        if (
            args.clock_health_policy is None
            or args.clock_health_policy_sha256 is None
            or args.clock_health_process is None
            or args.clock_health_process_sha256 is None
        ):
            raise SelectionRegistryError('clock-health policy and process configuration require both trusted digests')
        clock_health_gate = load_clock_health_gate(
            policy_path=Path(args.clock_health_policy),
            policy_sha256=args.clock_health_policy_sha256,
            process_config=Path(args.clock_health_process),
            process_config_sha256=args.clock_health_process_sha256,
        )
    if args.external_signer_process is not None:
        if args.external_signer_public_key is None or args.external_signer_process_sha256 is None:
            raise SelectionRegistryError('external signer public key and trusted process digest are required')
        if clock_health_gate is None:
            raise SelectionRegistryError('external signer mode requires a clock-health gate')
        signing_key = load_external_signer(
            process_config=Path(args.external_signer_process),
            process_config_sha256=args.external_signer_process_sha256,
            public_key=Path(args.external_signer_public_key),
        )
    else:
        if args.external_signer_public_key is not None or args.external_signer_process_sha256 is not None:
            raise SelectionRegistryError('external signer bindings require external signer mode')
        signing_key = load_ed25519_private_key(Path(args.dev_signing_private_key))
    if args.command == 'init':
        registry = SQLitePlanSelectionRegistry.initialize(
            Path(args.database),
            signing_key=signing_key,
            signing_key_id=args.signing_key_id,
            registry_policy_bytes=registry_policy_bytes,
            trust_policy_bytes=trust_policy_bytes,
            public_base_url=args.public_base_url,
            clock_health_gate=clock_health_gate,
        )
        _stdout(
            {
                'database': str(registry.database_path),
                'private_key_persisted_in_database': False,
                'registry_id': registry.policy.registry_id,
                'tree_size': 0,
                'clock_health_policy_sha256': args.clock_health_policy_sha256,
                'signer_mode': 'external' if args.external_signer_process else 'development-local',
            }
        )
        return
    checkpoint_witness_provider = None
    if args.command in {'serve', 'register', 'anchor-checkpoint'}:
        witness_token = read_owner_only_secret(Path(args.witness_write_token_file)).encode('ascii')
        parsed_trust = SelectionRegistryTrustPolicy.model_validate_json(trust_policy_bytes)
        checkpoint_witness_provider = Ed25519WitnessServiceProvider(
            canonical_json_bytes(parsed_trust.checkpoint_witness_policy),
            authorization_bearer_token=witness_token,
        )
    registry = SQLitePlanSelectionRegistry(
        Path(args.database),
        signing_key=signing_key,
        signing_key_id=args.signing_key_id,
        registry_policy_bytes=registry_policy_bytes,
        trust_policy_bytes=trust_policy_bytes,
        public_base_url=args.public_base_url,
        checkpoint_witness_provider=checkpoint_witness_provider,
        clock_health_gate=clock_health_gate,
    )
    if args.command == 'serve':
        serve_registry(
            registry,
            host=args.host,
            port=args.port,
            tls_certificate=Path(args.tls_certificate),
            tls_private_key=Path(args.tls_private_key),
            write_token_sha256=_sha256(read_owner_only_secret(Path(args.write_token_file)).encode('utf-8')),
        )
    elif args.command == 'register':
        request = PlanSelectionRequest.model_validate_json(Path(args.request).read_bytes())
        claim, proof = registry.provider(request)
        output = _write_exclusive(Path(args.proof_output), proof)
        _stdout({'proof_path': str(output), 'proof_sha256': _sha256(proof), **claim.model_dump(mode='json')})
    elif args.command == 'anchor-checkpoint':
        witness_proof = registry.anchor_checkpoint(args.tree_size)
        result: dict[str, object] = {
            'anchored': True,
            'tree_size': args.tree_size,
            'witness_proof_sha256': _sha256(witness_proof),
        }
        if args.proof_output:
            output = _write_exclusive(Path(args.proof_output), witness_proof)
            result['proof_path'] = str(output)
        _stdout(result)
    elif args.command == 'proof':
        proof = registry.proof_for(args.campaign_id, args.selection_key)
        output = _write_exclusive(Path(args.proof_output), proof)
        _stdout({'proof_path': str(output), 'proof_sha256': _sha256(proof)})
    elif args.command == 'entry':
        entry_bytes = registry.registry_entry_bytes(args.sequence)
        output = _write_exclusive(Path(args.output), entry_bytes)
        _stdout({'entry_path': str(output), 'entry_sha256': _sha256(entry_bytes), 'sequence': args.sequence})
    elif args.command == 'add-signing-key':
        try:
            registered_at = datetime.fromisoformat(args.registered_at.replace('Z', '+00:00'))
        except ValueError as error:
            raise SelectionRegistryError('registered-at must be an ISO 8601 timestamp with UTC offset') from error
        registry.register_signing_key(
            key_id=args.new_signing_key_id,
            public_key_base64=args.new_public_key_base64,
            registered_at=registered_at,
        )
        _stdout({'registered': True, 'signing_key_id': args.new_signing_key_id})
    elif args.command == 'status':
        checkpoint = registry.signed_tree_head()
        _stdout(
            {
                'registry_id': registry.policy.registry_id,
                'signed_checkpoint': None if checkpoint is None else checkpoint.model_dump(mode='json'),
                'tree_size': 0 if checkpoint is None else checkpoint.checkpoint.tree_size,
            }
        )


def _require_runtime_trust_binding(
    policy: SelectionRegistryPolicy,
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
        raise SelectionRegistryError('supplied runtime-trust digests differ from the persisted registry policy')


def _write_exclusive(path: Path, payload: bytes) -> Path:
    target = Path(path).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short write while persisting registry proof')
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(target.parent)
    return target


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _stdout(value: object) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b'\n')


def read_owner_only_secret(path: Path) -> str:
    """Read a high-entropy service token without accepting links or loose mode."""

    source = _require_owner_only_regular_file(path, 'secret')
    metadata = source.stat()
    if metadata.st_size < 32 or metadata.st_size > 4096:
        raise SelectionRegistryError('secret must contain 32 to 4096 bytes')
    value = source.read_text(encoding='utf-8')
    if value.endswith('\n'):
        value = value[:-1]
    if len(value) < 32 or value != value.strip():
        raise SelectionRegistryError('secret must be at least 32 trimmed characters')
    return value


def _require_owner_only_regular_file(path: Path, label: str) -> Path:
    source = _require_nonwritable_regular_file(path, label)
    if source.stat().st_mode & 0o077:
        raise SelectionRegistryError(f'{label} must be an owner-only regular file')
    return source


def _require_nonwritable_regular_file(path: Path, label: str) -> Path:
    request = Path(path).expanduser().absolute()
    parent = request.parent.resolve(strict=True)
    parent_metadata = parent.stat()
    allowed_owners = {os.getuid()} if hasattr(os, 'getuid') else set()
    allowed_owners.add(0)
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_mode & 0o022
        or (allowed_owners and parent_metadata.st_uid not in allowed_owners)
    ):
        raise SelectionRegistryError(f'{label} parent directory is not protected')
    source = parent / request.name
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o022:
        raise SelectionRegistryError(f'{label} must be a regular file not writable by group or other users')
    return source


if __name__ == '__main__':
    main()
