"""Separately deployable Ed25519 checkpoint witness and timestamp service.

The service accepts either a minimal operational-ledger checkpoint commitment or
an explicit signed-registry-checkpoint request containing the exact envelope and
RFC 6962 consistency path.  Registry requests are checked against an immutable
registry identity/key ring and a stateful monotonic-head index before signing.  It
obtains time from the service host after acquiring the append-only database write
transaction; callers cannot supply or override that time.  Each response contains
two Ed25519 signatures: one over the complete receipt statement and one over the
service's current hash-chain checkpoint.

The implementation is compatible with :mod:`vaxreplay.operations.witness`: the
HTTPS client implements ``ExternalCheckpointWitnessProvider`` and the offline
verifier implements ``TrustedCheckpointWitnessVerifier``.  RFC 3161 remains an
independent standards-backed alternative in :mod:`vaxreplay.operations.rfc3161`.

Running this code in the benchmark operator's own trust domain does *not* make it an
external witness.  Deploy it under an independent operator, pin its policy and
public key out of band, protect its host clock, and gossip/anchor signed checkpoints
if resistance to service equivocation or database rollback is required.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.server
import ipaddress
import os
import re
import secrets
import shutil
import sqlite3
import ssl
import stat
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.clock_health import ClockHealthGate, require_clock_health
from vaxreplay.operations.schema import SAFE_ID_PATTERN
from vaxreplay.operations.signing import Ed25519Signer, LocalEd25519Signer, checked_signer
from vaxreplay.operations.witness import (
    AuthenticatedExternalWitnessFacts,
    CheckpointWitnessRequest,
    ExternalWitnessClaim,
    ExternalWitnessMethod,
    RegistryCheckpointWitnessRequest,
    WitnessPolicyBinding,
)
from vaxreplay.operations.witness_service_schema import (
    ZERO_SHA256,
    WitnessedSignedRegistryCheckpoint,
    WitnessRegistryMonitor,
    WitnessServiceLogCheckpoint,
    WitnessServiceLogEntry,
    WitnessServicePolicy,
    WitnessServiceProof,
    WitnessServiceReceiptStatement,
    WitnessServiceSignedCheckpoint,
    WitnessServiceSubmission,
    WitnessServiceTrustPolicy,
    WitnessServiceVerificationReport,
)

WITNESS_SERVICE_DATABASE_SCHEMA_VERSION = 'vaxreplay.witness-service-database.v0.2'

_POLICY_PATH = 'policy.json'
_TRUST_POLICY_PATH = 'trust-policy.json'
_PRIVATE_KEY_PATH = 'ed25519-private-key.bin'
_DATABASE_PATH = 'witness.sqlite3'
_RECEIPT_SIGNATURE_DOMAIN = b'VaxReplay witness receipt v0.1\x00'
_CHECKPOINT_SIGNATURE_DOMAIN = b'VaxReplay witness log checkpoint v0.1\x00'
_MAX_KEY_BYTES = 32
_DEFAULT_BUSY_TIMEOUT_SECONDS = 30.0
_PROOF_MEDIA_TYPE = 'application/vnd.vaxreplay.witness-proof+json'
_CHECKPOINT_MEDIA_TYPE = 'application/vnd.vaxreplay.witness-checkpoint+json'
_JSON_MEDIA_TYPE = 'application/json'


class WitnessServiceError(ValueError):
    """The witness service, transport, store, or proof failed closed validation."""


class WitnessServiceDependencyError(ImportError):
    """Reserved for deployments missing the optional witness dependencies."""


@dataclass(frozen=True)
class WitnessServiceIssuance:
    """Result of one durable issuance or an exact idempotent retry."""

    receipt_id: str
    proof_bytes: bytes
    sequence: int
    created: bool


@dataclass(frozen=True)
class WitnessServiceTransportRequest:
    """Exact bounded HTTPS request made by the provider adapter."""

    endpoint_uri: str
    body: bytes
    timeout_seconds: float
    max_response_bytes: int
    authorization_bearer_token: bytes


@dataclass(frozen=True)
class WitnessServiceTransportResponse:
    """HTTP response facts required for fail-closed client validation."""

    status_code: int
    content_type: str | None
    body: bytes
    final_uri: str
    content_encoding: str | None = None
    content_length: int | None = None


type WitnessServiceTransport = Callable[[WitnessServiceTransportRequest], WitnessServiceTransportResponse]


def _security_time() -> datetime:
    """Read the service-host wall clock; intentionally has no caller input."""

    return datetime.now(timezone.utc)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _registry_node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b'\x01' + left + right).digest()


def _verify_registry_consistency(
    *,
    old_size: int,
    new_size: int,
    old_root: bytes,
    new_root: bytes,
    proof: tuple[bytes, ...],
) -> bool:
    """Verify one RFC6962 append-only path without importing registry code."""

    if old_size < 0 or new_size < old_size or len(old_root) != 32 or len(new_root) != 32:
        return False
    if old_size == 0:
        return not proof and hmac.compare_digest(old_root, hashlib.sha256(b'').digest())
    if old_size == new_size:
        return not proof and hmac.compare_digest(old_root, new_root)
    first = old_size - 1
    second = new_size - 1
    while first & 1:
        first >>= 1
        second >>= 1
    position = 0
    if first == 0:
        old_hash = old_root
        new_hash = old_root
    else:
        if not proof or len(proof[0]) != 32:
            return False
        old_hash = proof[0]
        new_hash = proof[0]
        position = 1
    for sibling in proof[position:]:
        if len(sibling) != 32 or second == 0:
            return False
        if first & 1 or first == second:
            old_hash = _registry_node_hash(sibling, old_hash)
            new_hash = _registry_node_hash(sibling, new_hash)
            while first != 0 and not (first & 1):
                first >>= 1
                second >>= 1
        else:
            new_hash = _registry_node_hash(new_hash, sibling)
        first >>= 1
        second >>= 1
    return second == 0 and hmac.compare_digest(old_hash, old_root) and hmac.compare_digest(new_hash, new_root)


def _exact_bytes(payload: bytes, label: str, *, maximum: int) -> bytes:
    if not isinstance(payload, bytes):
        raise WitnessServiceError(f'{label} must be exact bytes')
    if not payload or len(payload) > maximum:
        raise WitnessServiceError(f'{label} has an invalid size')
    return payload


def _load_canonical_model[ModelT: StrictModel](payload: bytes, model: type[ModelT], label: str) -> ModelT:
    try:
        parsed = model.model_validate_json(payload)
    except (TypeError, ValueError) as error:
        raise WitnessServiceError(f'invalid {label}: {error}') from error
    if payload != canonical_json_bytes(parsed):
        raise WitnessServiceError(f'{label} must use canonical JSON encoding')
    return parsed


def _aware_security_time(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise WitnessServiceError('service-host clock returned a timezone-naive value')
    return value.astimezone(timezone.utc)


def _signature_payload(domain: bytes, model: object) -> bytes:
    return domain + canonical_json_bytes(model)


def _receipt_id(statement: WitnessServiceReceiptStatement) -> str:
    return f'receipt-{_sha256(canonical_json_bytes(statement))[:32]}'


def _public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _private_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def make_witness_service_policy_binding(
    policy_bytes: bytes,
    trust_policy_bytes: bytes,
    verifier_implementation_bytes: bytes,
    *,
    verifier_id: str,
) -> WitnessPolicyBinding:
    """Construct the exact generic witness binding from three pinned artifacts.

    ``verifier_implementation_bytes`` must be the actual deployed verifier artifact
    (for example, a hermetic executable or wheel), not merely a version string.
    """

    policy_bytes = _exact_bytes(policy_bytes, 'witness service policy', maximum=1024 * 1024)
    trust_policy_bytes = _exact_bytes(trust_policy_bytes, 'witness service trust policy', maximum=1024 * 1024)
    verifier_implementation_bytes = _exact_bytes(
        verifier_implementation_bytes,
        'witness service verifier implementation',
        maximum=16 * 1024 * 1024,
    )
    policy = _load_canonical_model(policy_bytes, WitnessServicePolicy, 'witness service policy')
    trust = _load_canonical_model(
        trust_policy_bytes,
        WitnessServiceTrustPolicy,
        'witness service trust policy',
    )
    _validate_policy_trust_binding(policy, policy_bytes, trust)
    if not isinstance(verifier_id, str) or not re.fullmatch(SAFE_ID_PATTERN, verifier_id):
        raise WitnessServiceError('verifier_id is invalid')
    return WitnessPolicyBinding(
        authority_id=policy.authority_id,
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id=policy.policy_id,
        policy_sha256=_sha256(policy_bytes),
        trust_policy_id=trust.trust_policy_id,
        trust_policy_sha256=_sha256(trust_policy_bytes),
        verifier_id=verifier_id,
        verifier_implementation_sha256=_sha256(verifier_implementation_bytes),
    )


def _validate_policy_trust_binding(
    policy: WitnessServicePolicy,
    policy_bytes: bytes,
    trust: WitnessServiceTrustPolicy,
) -> None:
    if (
        trust.authority_id != policy.authority_id
        or trust.witness_id != policy.witness_id
        or trust.service_policy_sha256 != _sha256(policy_bytes)
    ):
        raise WitnessServiceError('witness service trust policy does not bind the exact service policy')


def _parse_public_key(trust: WitnessServiceTrustPolicy) -> Ed25519PublicKey:
    try:
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(trust.public_key_base64, validate=True))
    except (TypeError, ValueError) as error:
        raise WitnessServiceError('witness service trust policy has an invalid Ed25519 public key') from error


def _verify_proof_commitment(
    proof_bytes: bytes,
    *,
    policy: WitnessServicePolicy,
    policy_bytes: bytes,
    trust: WitnessServiceTrustPolicy,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
) -> WitnessServiceProof:
    proof_bytes = _exact_bytes(proof_bytes, 'witness service proof', maximum=policy.max_proof_bytes)
    proof = _load_canonical_model(proof_bytes, WitnessServiceProof, 'witness service proof')
    statement = proof.statement
    submission = statement.submission
    request = submission.witness_request
    entry = statement.entry
    checkpoint = statement.checkpoint
    policy_sha256 = _sha256(policy_bytes)

    if proof.receipt_id != _receipt_id(statement):
        raise WitnessServiceError('witness receipt_id does not match its signed statement')
    if (
        request.authority_id != policy.authority_id
        or request.method is not ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG
        or request.policy_id != policy.policy_id
        or request.policy_sha256 != policy_sha256
    ):
        raise WitnessServiceError('witness proof request does not match the exact service policy')
    if request.checkpoint_sha256 != checkpoint_sha256 or request.checkpoint_bytes != checkpoint_bytes:
        raise WitnessServiceError('witness proof binds a different checkpoint commitment')
    submission_bytes = canonical_json_bytes(submission)
    if (
        entry.submission_sha256 != _sha256(submission_bytes)
        or entry.checkpoint_schema_version != request.checkpoint_schema_version
        or entry.checkpoint_sha256 != request.checkpoint_sha256
        or entry.checkpoint_bytes != request.checkpoint_bytes
        or entry.client_nonce != submission.client_nonce
        or entry.authority_id != policy.authority_id
        or entry.witness_id != policy.witness_id
        or entry.policy_id != policy.policy_id
        or entry.policy_sha256 != policy_sha256
    ):
        raise WitnessServiceError('witness log entry does not bind the exact submission and policy')
    entry_sha256 = _sha256(canonical_json_bytes(entry))
    if (
        checkpoint.authority_id != policy.authority_id
        or checkpoint.witness_id != policy.witness_id
        or checkpoint.policy_id != policy.policy_id
        or checkpoint.policy_sha256 != policy_sha256
        or checkpoint.tree_size != entry.sequence
        or checkpoint.through_entry_sha256 != entry_sha256
        or checkpoint.issued_at != entry.witnessed_at
    ):
        raise WitnessServiceError('signed witness log checkpoint does not bind the receipt entry')
    if entry.witnessed_at < trust.key_valid_from or (
        trust.key_valid_until is not None and entry.witnessed_at >= trust.key_valid_until
    ):
        raise WitnessServiceError('witness receipt time is outside the pinned signing-key validity window')

    public_key = _parse_public_key(trust)
    try:
        public_key.verify(
            base64.b64decode(proof.receipt_signature_base64, validate=True),
            _signature_payload(_RECEIPT_SIGNATURE_DOMAIN, statement),
        )
        public_key.verify(
            base64.b64decode(proof.checkpoint_signature_base64, validate=True),
            _signature_payload(_CHECKPOINT_SIGNATURE_DOMAIN, checkpoint),
        )
    except InvalidSignature as error:
        raise WitnessServiceError('witness service Ed25519 signature verification failed') from error
    return proof


def _build_signed_proof(
    *,
    signer: Ed25519Signer,
    submission: WitnessServiceSubmission,
    entry: WitnessServiceLogEntry,
    checkpoint: WitnessServiceLogCheckpoint,
) -> WitnessServiceProof:
    statement = WitnessServiceReceiptStatement(submission=submission, entry=entry, checkpoint=checkpoint)
    receipt_signature = signer.sign(_signature_payload(_RECEIPT_SIGNATURE_DOMAIN, statement))
    checkpoint_signature = signer.sign(_signature_payload(_CHECKPOINT_SIGNATURE_DOMAIN, checkpoint))
    return WitnessServiceProof(
        receipt_id=_receipt_id(statement),
        statement=statement,
        receipt_signature_base64=base64.b64encode(receipt_signature).decode('ascii'),
        checkpoint_signature_base64=base64.b64encode(checkpoint_signature).decode('ascii'),
    )


_DATABASE_SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
) STRICT;
CREATE TABLE entries (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    receipt_id TEXT NOT NULL UNIQUE,
    submission_sha256 TEXT NOT NULL UNIQUE,
    client_nonce TEXT NOT NULL UNIQUE,
    submission_bytes BLOB NOT NULL,
    entry_bytes BLOB NOT NULL,
    checkpoint_bytes BLOB NOT NULL,
    proof_bytes BLOB NOT NULL,
    entry_sha256 TEXT NOT NULL UNIQUE,
    previous_entry_sha256 TEXT NOT NULL,
    witnessed_at TEXT NOT NULL
) STRICT;
CREATE TABLE registry_heads (
    registry_id TEXT NOT NULL,
    authority_id TEXT NOT NULL,
    tree_size INTEGER NOT NULL CHECK (tree_size >= 0),
    root_sha256 TEXT NOT NULL,
    envelope_sha256 TEXT NOT NULL UNIQUE,
    envelope_bytes BLOB NOT NULL,
    witness_entry_sequence INTEGER NOT NULL UNIQUE REFERENCES entries(sequence),
    PRIMARY KEY (registry_id, authority_id, tree_size)
) STRICT;
CREATE TRIGGER metadata_no_update BEFORE UPDATE ON metadata BEGIN
    SELECT RAISE(ABORT, 'metadata is immutable');
END;
CREATE TRIGGER metadata_no_delete BEFORE DELETE ON metadata BEGIN
    SELECT RAISE(ABORT, 'metadata is immutable');
END;
CREATE TRIGGER entries_no_update BEFORE UPDATE ON entries BEGIN
    SELECT RAISE(ABORT, 'witness log is append-only');
END;
CREATE TRIGGER entries_no_delete BEFORE DELETE ON entries BEGIN
    SELECT RAISE(ABORT, 'witness log is append-only');
END;
CREATE TRIGGER registry_heads_no_update BEFORE UPDATE ON registry_heads BEGIN
    SELECT RAISE(ABORT, 'witnessed registry heads are append-only');
END;
CREATE TRIGGER registry_heads_no_delete BEFORE DELETE ON registry_heads BEGIN
    SELECT RAISE(ABORT, 'witnessed registry heads are append-only');
END;
"""


class WitnessServiceStore:
    """Durable SQLite-backed append-only signer state.

    A new SQLite connection and ``BEGIN IMMEDIATE`` transaction are used for every
    issuance, so independent server workers serialize sequence allocation and clock
    checks.  ``synchronous=FULL`` makes a successful commit the response boundary.
    """

    def __init__(
        self,
        root: Path,
        *,
        busy_timeout_seconds: float = _DEFAULT_BUSY_TIMEOUT_SECONDS,
        signer: Ed25519Signer | None = None,
        clock: Callable[[], datetime] | None = None,
        clock_health_gate: ClockHealthGate | None = None,
    ) -> None:
        if not isinstance(busy_timeout_seconds, (int, float)) or isinstance(busy_timeout_seconds, bool):
            raise WitnessServiceError('busy_timeout_seconds must be numeric')
        if busy_timeout_seconds <= 0 or busy_timeout_seconds > 300:
            raise WitnessServiceError('busy_timeout_seconds is outside the accepted range')
        self.root = _validate_service_root(root)
        if signer is not None and clock_health_gate is None:
            raise WitnessServiceError('isolated witness signers require a fail-closed clock-health gate')
        self.busy_timeout_seconds = float(busy_timeout_seconds)
        self.policy_bytes = _read_regular_nofollow(self.root / _POLICY_PATH, 1024 * 1024, private=False)
        self.trust_policy_bytes = _read_regular_nofollow(
            self.root / _TRUST_POLICY_PATH,
            1024 * 1024,
            private=False,
        )
        self.policy = _load_canonical_model(
            self.policy_bytes,
            WitnessServicePolicy,
            'witness service policy',
        )
        self.trust_policy = _load_canonical_model(
            self.trust_policy_bytes,
            WitnessServiceTrustPolicy,
            'witness service trust policy',
        )
        _validate_policy_trust_binding(self.policy, self.policy_bytes, self.trust_policy)
        expected_public_key = base64.b64decode(self.trust_policy.public_key_base64, validate=True)
        if signer is None:
            private_key_bytes = _read_regular_nofollow(
                self.root / _PRIVATE_KEY_PATH,
                _MAX_KEY_BYTES,
                private=True,
            )
            if len(private_key_bytes) != _MAX_KEY_BYTES:
                raise WitnessServiceError('Ed25519 private key file must contain exactly 32 bytes')
            try:
                signer = LocalEd25519Signer(Ed25519PrivateKey.from_private_bytes(private_key_bytes))
            except ValueError as error:
                raise WitnessServiceError('invalid Ed25519 private key bytes') from error
        try:
            self._signer = checked_signer(signer, expected_public_key=expected_public_key)
        except ValueError as error:
            raise WitnessServiceError(f'witness signer is invalid: {error}') from error
        # Keep ``None`` dynamic so tests and emergency operators can replace the
        # module security-clock implementation before issuance.
        self._clock = clock
        self._clock_health_gate = clock_health_gate
        _validate_database_file(self.root / _DATABASE_PATH)
        with closing(self._connect()) as connection:
            self._verify_metadata(connection)
        # A valid SQLite page structure and immutable metadata do not authenticate
        # the latest predecessor.  Replay every signature and hash-chain link before
        # accepting traffic so a locally altered tail cannot be extended and signed.
        self.verify()

    @classmethod
    def initialize(
        cls,
        root: Path,
        *,
        authority_id: str,
        witness_id: str,
        policy_id: str,
        trust_policy_id: str,
        endpoint_uri: str,
        max_submission_bytes: int = 64 * 1024,
        max_proof_bytes: int = 1024 * 1024,
        client_timeout_seconds: float = 15.0,
        registry_monitors: tuple[WitnessRegistryMonitor, ...] = (),
        clock_health_policy_sha256: str | None = None,
        clock_health_process_sha256: str | None = None,
        external_signer_process_sha256: str | None = None,
        signer: Ed25519Signer | None = None,
        clock: Callable[[], datetime] | None = None,
        clock_health_gate: ClockHealthGate | None = None,
    ) -> WitnessServiceStore:
        """Create a no-replace service root and generate its key on the service host."""

        if signer is not None and clock_health_gate is None:
            raise WitnessServiceError('isolated witness signers require a fail-closed clock-health gate')
        created_at = _aware_security_time((clock or _security_time)())
        require_clock_health(clock_health_gate, security_time=created_at)
        private_key = Ed25519PrivateKey.generate() if signer is None else None
        if private_key is not None:
            active_signer: Ed25519Signer = LocalEd25519Signer(private_key)
        else:
            if signer is None:  # narrowed by private-key selection above
                raise WitnessServiceError('isolated witness signer is missing')
            active_signer = checked_signer(signer)
        public_key = active_signer.public_key_bytes()
        policy = WitnessServicePolicy(
            authority_id=authority_id,
            witness_id=witness_id,
            policy_id=policy_id,
            endpoint_uri=endpoint_uri,
            max_submission_bytes=max_submission_bytes,
            max_proof_bytes=max_proof_bytes,
            client_timeout_seconds=client_timeout_seconds,
            registry_monitors=registry_monitors,
            clock_health_policy_sha256=clock_health_policy_sha256,
            clock_health_process_sha256=clock_health_process_sha256,
            external_signer_process_sha256=external_signer_process_sha256,
        )
        policy_bytes = canonical_json_bytes(policy)
        trust_policy = WitnessServiceTrustPolicy(
            authority_id=authority_id,
            witness_id=witness_id,
            trust_policy_id=trust_policy_id,
            service_policy_sha256=_sha256(policy_bytes),
            public_key_base64=base64.b64encode(public_key).decode('ascii'),
            public_key_sha256=_sha256(public_key),
            key_valid_from=created_at,
        )
        trust_policy_bytes = canonical_json_bytes(trust_policy)

        requested = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
        requested.parent.mkdir(parents=True, exist_ok=True)
        parent = requested.parent.resolve(strict=True)
        target = parent / requested.name
        if os.path.lexists(target):
            raise WitnessServiceError(f'witness service root already exists: {target}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=parent))
        try:
            staging.chmod(0o700)
            _write_durable_file(staging / _POLICY_PATH, policy_bytes, mode=0o644)
            _write_durable_file(staging / _TRUST_POLICY_PATH, trust_policy_bytes, mode=0o644)
            if private_key is not None:
                _write_durable_file(staging / _PRIVATE_KEY_PATH, _private_key_bytes(private_key), mode=0o600)
            database_path = staging / _DATABASE_PATH
            connection = sqlite3.connect(database_path, isolation_level=None)
            try:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=FULL')
                connection.executescript(_DATABASE_SCHEMA)
                connection.execute('BEGIN IMMEDIATE')
                metadata = {
                    'schema_version': WITNESS_SERVICE_DATABASE_SCHEMA_VERSION,
                    'policy_sha256': _sha256(policy_bytes),
                    'trust_policy_sha256': _sha256(trust_policy_bytes),
                    'public_key_sha256': _sha256(public_key),
                }
                connection.executemany(
                    'INSERT INTO metadata(key, value) VALUES (?, ?)',
                    tuple((key, value.encode('ascii')) for key, value in metadata.items()),
                )
                connection.execute('COMMIT')
                connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            except BaseException:
                if connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise
            finally:
                connection.close()
            database_path.chmod(0o600)
            _fsync_file(database_path)
            fsync_directory(staging)
            try:
                rename_directory_noreplace(staging, target)
            except FileExistsError as error:
                raise WitnessServiceError(f'witness service root already exists: {target}') from error
            staging = Path()
            fsync_directory(parent)
        finally:
            if staging != Path() and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return cls(
            target,
            signer=None if private_key is not None else active_signer,
            clock=clock,
            clock_health_gate=clock_health_gate,
        )

    def _connect(self) -> sqlite3.Connection:
        _validate_database_file(self.root / _DATABASE_PATH)
        connection = sqlite3.connect(
            self.root / _DATABASE_PATH,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA trusted_schema=OFF')
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute(f'PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}')
        return connection

    def _verify_metadata(self, connection: sqlite3.Connection) -> None:
        integrity = connection.execute('PRAGMA quick_check').fetchone()
        if integrity is None or integrity[0] != 'ok':
            raise WitnessServiceError('witness service database failed SQLite integrity verification')
        rows = connection.execute('SELECT key, value FROM metadata ORDER BY key').fetchall()
        metadata = {str(row['key']): bytes(row['value']).decode('ascii') for row in rows}
        expected = {
            'schema_version': WITNESS_SERVICE_DATABASE_SCHEMA_VERSION,
            'policy_sha256': _sha256(self.policy_bytes),
            'trust_policy_sha256': _sha256(self.trust_policy_bytes),
            'public_key_sha256': self.trust_policy.public_key_sha256,
        }
        if metadata != expected:
            raise WitnessServiceError('witness service database metadata does not match pinned files')
        trigger_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' ORDER BY name"
            ).fetchall()
        }
        if trigger_names != {
            'entries_no_delete',
            'entries_no_update',
            'metadata_no_delete',
            'metadata_no_update',
            'registry_heads_no_delete',
            'registry_heads_no_update',
        }:
            raise WitnessServiceError('witness service append-only database triggers are missing or unexpected')

    def issue(self, submission_bytes: bytes) -> WitnessServiceIssuance:
        """Sign and commit one canonical submission, then return exact proof bytes.

        Exact retries return the original proof without allocating a new sequence.
        Reusing a nonce for different bytes is rejected.  The response is never
        returned before the ``synchronous=FULL`` SQLite commit succeeds.
        """

        submission_bytes = _exact_bytes(
            submission_bytes,
            'witness service submission',
            maximum=self.policy.max_submission_bytes,
        )
        submission = _load_canonical_model(
            submission_bytes,
            WitnessServiceSubmission,
            'witness service submission',
        )
        self._validate_submission_policy(submission)
        submission_sha256 = _sha256(submission_bytes)

        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            self._verify_metadata(connection)
            self._verify_log(connection)
            duplicate = connection.execute(
                'SELECT sequence, receipt_id, submission_bytes, proof_bytes FROM entries WHERE submission_sha256 = ?',
                (submission_sha256,),
            ).fetchone()
            if duplicate is not None:
                stored_submission = bytes(duplicate['submission_bytes'])
                if not hmac.compare_digest(stored_submission, submission_bytes):
                    raise WitnessServiceError('submission SHA-256 collision detected')
                proof_bytes = bytes(duplicate['proof_bytes'])
                proof = _verify_proof_commitment(
                    proof_bytes,
                    policy=self.policy,
                    policy_bytes=self.policy_bytes,
                    trust=self.trust_policy,
                    checkpoint_sha256=submission.witness_request.checkpoint_sha256,
                    checkpoint_bytes=submission.witness_request.checkpoint_bytes,
                )
                if proof.receipt_id != str(duplicate['receipt_id']):
                    raise WitnessServiceError('stored idempotent receipt has an inconsistent identifier')
                connection.execute('COMMIT')
                return WitnessServiceIssuance(
                    receipt_id=proof.receipt_id,
                    proof_bytes=proof_bytes,
                    sequence=int(duplicate['sequence']),
                    created=False,
                )
            reused_nonce = connection.execute(
                'SELECT submission_sha256 FROM entries WHERE client_nonce = ?',
                (submission.client_nonce,),
            ).fetchone()
            if reused_nonce is not None:
                raise WitnessServiceError('client nonce was already used by a different submission')

            previous = connection.execute(
                'SELECT sequence, entry_sha256, checkpoint_bytes, witnessed_at '
                'FROM entries ORDER BY sequence DESC LIMIT 1'
            ).fetchone()
            if previous is None:
                sequence = 1
                previous_entry_sha256 = ZERO_SHA256
                previous_checkpoint_sha256 = ZERO_SHA256
                previous_time = None
            else:
                sequence = int(previous['sequence']) + 1
                previous_entry_sha256 = str(previous['entry_sha256'])
                previous_checkpoint_sha256 = _sha256(bytes(previous['checkpoint_bytes']))
                previous_time = datetime.fromisoformat(str(previous['witnessed_at']))

            witnessed_at = _aware_security_time((self._clock or _security_time)())
            require_clock_health(self._clock_health_gate, security_time=witnessed_at)
            if previous_time is not None:
                previous_time = _aware_security_time(previous_time)
                if witnessed_at < previous_time:
                    raise WitnessServiceError(
                        'service-host clock moved backwards; issuance failed without synthesizing a timestamp'
                    )
            if witnessed_at < self.trust_policy.key_valid_from or (
                self.trust_policy.key_valid_until is not None and witnessed_at >= self.trust_policy.key_valid_until
            ):
                raise WitnessServiceError('service-host time is outside the signing-key validity window')

            request = submission.witness_request
            registry_head = self._validate_registry_checkpoint_request(
                connection,
                request,
                witnessed_at=witnessed_at,
            )
            entry = WitnessServiceLogEntry(
                sequence=sequence,
                previous_entry_sha256=previous_entry_sha256,
                submission_sha256=submission_sha256,
                checkpoint_schema_version=request.checkpoint_schema_version,
                checkpoint_sha256=request.checkpoint_sha256,
                checkpoint_bytes=request.checkpoint_bytes,
                client_nonce=submission.client_nonce,
                authority_id=self.policy.authority_id,
                witness_id=self.policy.witness_id,
                policy_id=self.policy.policy_id,
                policy_sha256=_sha256(self.policy_bytes),
                witnessed_at=witnessed_at,
            )
            entry_bytes = canonical_json_bytes(entry)
            entry_sha256 = _sha256(entry_bytes)
            checkpoint = WitnessServiceLogCheckpoint(
                authority_id=self.policy.authority_id,
                witness_id=self.policy.witness_id,
                policy_id=self.policy.policy_id,
                policy_sha256=_sha256(self.policy_bytes),
                tree_size=sequence,
                through_entry_sha256=entry_sha256,
                previous_checkpoint_sha256=previous_checkpoint_sha256,
                issued_at=witnessed_at,
            )
            checkpoint_bytes = canonical_json_bytes(checkpoint)
            proof = _build_signed_proof(
                signer=self._signer,
                submission=submission,
                entry=entry,
                checkpoint=checkpoint,
            )
            proof_bytes = canonical_json_bytes(proof)
            if len(proof_bytes) > self.policy.max_proof_bytes:
                raise WitnessServiceError('generated witness proof exceeds the policy byte limit')
            _verify_proof_commitment(
                proof_bytes,
                policy=self.policy,
                policy_bytes=self.policy_bytes,
                trust=self.trust_policy,
                checkpoint_sha256=request.checkpoint_sha256,
                checkpoint_bytes=request.checkpoint_bytes,
            )
            connection.execute(
                """
                INSERT INTO entries(
                    sequence, receipt_id, submission_sha256, client_nonce, submission_bytes,
                    entry_bytes, checkpoint_bytes, proof_bytes, entry_sha256,
                    previous_entry_sha256, witnessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sequence,
                    proof.receipt_id,
                    submission_sha256,
                    submission.client_nonce,
                    submission_bytes,
                    entry_bytes,
                    checkpoint_bytes,
                    proof_bytes,
                    entry_sha256,
                    previous_entry_sha256,
                    witnessed_at.isoformat(),
                ),
            )
            if registry_head is not None:
                envelope, already_recorded = registry_head
                if not already_recorded:
                    envelope_bytes = canonical_json_bytes(envelope)
                    checkpoint = envelope.checkpoint
                    connection.execute(
                        'INSERT INTO registry_heads('
                        'registry_id, authority_id, tree_size, root_sha256, envelope_sha256, '
                        'envelope_bytes, witness_entry_sequence) VALUES (?, ?, ?, ?, ?, ?, ?)',
                        (
                            checkpoint.registry_id,
                            checkpoint.authority_id,
                            checkpoint.tree_size,
                            checkpoint.root_sha256,
                            _sha256(envelope_bytes),
                            envelope_bytes,
                            sequence,
                        ),
                    )
            connection.execute('COMMIT')
            return WitnessServiceIssuance(
                receipt_id=proof.receipt_id,
                proof_bytes=proof_bytes,
                sequence=sequence,
                created=True,
            )
        except BaseException:
            if connection.in_transaction:
                connection.execute('ROLLBACK')
            raise
        finally:
            connection.close()

    def _validate_registry_checkpoint_request(
        self,
        connection: sqlite3.Connection,
        request: CheckpointWitnessRequest | RegistryCheckpointWitnessRequest,
        *,
        witnessed_at: datetime,
    ) -> tuple[WitnessedSignedRegistryCheckpoint, bool] | None:
        if isinstance(request, CheckpointWitnessRequest):
            return None
        try:
            envelope_bytes = base64.b64decode(request.signed_checkpoint_base64, validate=True)
        except (TypeError, ValueError) as error:
            raise WitnessServiceError('registry checkpoint request has invalid base64') from error
        if len(envelope_bytes) != request.checkpoint_bytes or _sha256(envelope_bytes) != request.checkpoint_sha256:
            raise WitnessServiceError('registry checkpoint request digest or size differs from exact envelope')
        envelope = _load_canonical_model(
            envelope_bytes,
            WitnessedSignedRegistryCheckpoint,
            'signed registry checkpoint',
        )
        checkpoint = envelope.checkpoint
        monitors = [
            monitor
            for monitor in self.policy.registry_monitors
            if monitor.registry_id == checkpoint.registry_id and monitor.authority_id == checkpoint.authority_id
        ]
        if len(monitors) != 1:
            raise WitnessServiceError('registry checkpoint identity is not uniquely configured for monitoring')
        monitor = monitors[0]
        signing_keys = [key for key in monitor.signing_keys if key.key_id == checkpoint.signing_key_id]
        if len(signing_keys) != 1:
            raise WitnessServiceError('registry checkpoint signer is not uniquely authorized by witness policy')
        signing_key = signing_keys[0]
        if witnessed_at < signing_key.valid_from or (
            signing_key.valid_until is not None and witnessed_at >= signing_key.valid_until
        ):
            raise WitnessServiceError('registry checkpoint key is not valid at independent witness time')
        if checkpoint.issued_at_upper_bound > witnessed_at:
            raise WitnessServiceError('registry checkpoint claims issuance after independent witness time')
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(signing_key.public_key_base64, validate=True)).verify(
                base64.b64decode(envelope.signature_base64, validate=True),
                canonical_json_bytes(checkpoint),
            )
        except (InvalidSignature, ValueError) as error:
            raise WitnessServiceError('registry checkpoint has an invalid configured-key signature') from error

        prior_same_size = connection.execute(
            'SELECT root_sha256, envelope_bytes FROM registry_heads '
            'WHERE registry_id = ? AND authority_id = ? AND tree_size = ?',
            (checkpoint.registry_id, checkpoint.authority_id, checkpoint.tree_size),
        ).fetchone()
        if prior_same_size is not None:
            if str(prior_same_size['root_sha256']) != checkpoint.root_sha256 or not hmac.compare_digest(
                bytes(prior_same_size['envelope_bytes']), envelope_bytes
            ):
                raise WitnessServiceError(
                    'registry split view rejected: tree size was already witnessed with another root'
                )
            return envelope, True

        previous = connection.execute(
            'SELECT tree_size, root_sha256, envelope_bytes FROM registry_heads '
            'WHERE registry_id = ? AND authority_id = ? ORDER BY tree_size DESC LIMIT 1',
            (checkpoint.registry_id, checkpoint.authority_id),
        ).fetchone()
        if previous is None:
            if (
                checkpoint.tree_size != 0
                or request.consistency_from_tree_size != 0
                or request.consistency_from_root_sha256 != hashlib.sha256(b'').hexdigest()
                or request.consistency_proof_sha256
            ):
                raise WitnessServiceError('first witnessed registry head must be the canonical genesis')
            return envelope, False

        previous_size = int(previous['tree_size'])
        previous_root = str(previous['root_sha256'])
        if checkpoint.tree_size != previous_size + 1:
            raise WitnessServiceError('registry checkpoint size is not the next monotonic tree size')
        if request.consistency_from_tree_size != previous_size or request.consistency_from_root_sha256 != previous_root:
            raise WitnessServiceError('registry consistency path does not start at the last witnessed head')
        previous_envelope = _load_canonical_model(
            bytes(previous['envelope_bytes']),
            WitnessedSignedRegistryCheckpoint,
            'previous witnessed registry checkpoint',
        )
        expected_previous_sha256 = _sha256(
            canonical_json_bytes(previous_envelope.checkpoint)
            + base64.b64decode(previous_envelope.signature_base64, validate=True)
        )
        if checkpoint.previous_checkpoint_sha256 != expected_previous_sha256:
            raise WitnessServiceError('registry checkpoint does not name the exact last witnessed predecessor')
        if checkpoint.issued_at_upper_bound < previous_envelope.checkpoint.issued_at_upper_bound:
            raise WitnessServiceError('registry checkpoint issuance clock regressed')
        proof = tuple(bytes.fromhex(item) for item in request.consistency_proof_sha256)
        if not _verify_registry_consistency(
            old_size=previous_size,
            new_size=checkpoint.tree_size,
            old_root=bytes.fromhex(previous_root),
            new_root=bytes.fromhex(checkpoint.root_sha256),
            proof=proof,
        ):
            raise WitnessServiceError('registry checkpoint has an invalid RFC6962 consistency proof')
        return envelope, False

    def _validate_submission_policy(self, submission: WitnessServiceSubmission) -> None:
        request = submission.witness_request
        if (
            request.authority_id != self.policy.authority_id
            or request.method is not ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG
            or request.policy_id != self.policy.policy_id
            or request.policy_sha256 != _sha256(self.policy_bytes)
        ):
            raise WitnessServiceError('submission does not match the exact witness service policy')

    def receipt_bytes(self, receipt_id: str) -> bytes:
        """Read one immutable public proof by its derived receipt identifier."""

        if not isinstance(receipt_id, str) or len(receipt_id) != 40 or not receipt_id.startswith('receipt-'):
            raise WitnessServiceError('invalid witness receipt identifier')
        with closing(self._connect()) as connection:
            row = connection.execute(
                'SELECT proof_bytes FROM entries WHERE receipt_id = ?',
                (receipt_id,),
            ).fetchone()
        if row is None:
            raise WitnessServiceError('witness receipt was not found')
        proof_bytes = bytes(row['proof_bytes'])
        proof = _load_canonical_model(proof_bytes, WitnessServiceProof, 'stored witness service proof')
        if proof.receipt_id != receipt_id:
            raise WitnessServiceError('stored witness receipt identifier is inconsistent')
        return proof_bytes

    def proof_bytes_at_sequence(self, sequence: int) -> bytes:
        """Return one immutable log proof so monitors can enumerate the public log."""

        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise WitnessServiceError('witness log sequence must be a positive integer')
        with closing(self._connect()) as connection:
            row = connection.execute(
                'SELECT proof_bytes FROM entries WHERE sequence = ?',
                (sequence,),
            ).fetchone()
        if row is None:
            raise WitnessServiceError('witness log entry was not found')
        proof_bytes = bytes(row['proof_bytes'])
        _load_canonical_model(proof_bytes, WitnessServiceProof, 'stored witness service proof')
        return proof_bytes

    def latest_signed_checkpoint_bytes(self) -> bytes:
        """Return the public signed head without disclosing submission data."""

        return self.signed_checkpoint_bytes()

    def signed_checkpoint_bytes(self, sequence: int | None = None) -> bytes:
        """Return a signed historical checkpoint by sequence, or the latest head."""

        if sequence is not None and (not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1):
            raise WitnessServiceError('witness checkpoint sequence must be a positive integer')
        with closing(self._connect()) as connection:
            if sequence is None:
                row = connection.execute('SELECT proof_bytes FROM entries ORDER BY sequence DESC LIMIT 1').fetchone()
            else:
                row = connection.execute(
                    'SELECT proof_bytes FROM entries WHERE sequence = ?',
                    (sequence,),
                ).fetchone()
        if row is None:
            raise WitnessServiceError('requested witness service log checkpoint was not found')
        proof = _load_canonical_model(
            bytes(row['proof_bytes']),
            WitnessServiceProof,
            'stored witness service proof',
        )
        signed = WitnessServiceSignedCheckpoint(
            checkpoint=proof.statement.checkpoint,
            signature_base64=proof.checkpoint_signature_base64,
        )
        return canonical_json_bytes(signed)

    def verify(self) -> WitnessServiceVerificationReport:
        """Replay all rows, signatures, hashes, sequence links, and checkpoints."""

        with closing(self._connect()) as connection:
            self._verify_metadata(connection)
            return self._verify_log(connection)

    def _verify_log(self, connection: sqlite3.Connection) -> WitnessServiceVerificationReport:
        """Replay the complete log using an existing read or write transaction."""

        previous_entry_sha256 = ZERO_SHA256
        previous_checkpoint_sha256 = ZERO_SHA256
        previous_time: datetime | None = None
        entry_count = 0
        registry_replay: list[tuple[int, RegistryCheckpointWitnessRequest, WitnessedSignedRegistryCheckpoint]] = []
        rows = connection.execute(
            """
            SELECT sequence, receipt_id, submission_sha256, client_nonce, submission_bytes,
                   entry_bytes, checkpoint_bytes, proof_bytes, entry_sha256,
                   previous_entry_sha256, witnessed_at
            FROM entries ORDER BY sequence
            """
        ).fetchall()
        for expected_sequence, row in enumerate(rows, start=1):
            sequence = int(row['sequence'])
            if sequence != expected_sequence:
                raise WitnessServiceError('witness service log sequence is not contiguous')
            submission_bytes = bytes(row['submission_bytes'])
            entry_bytes = bytes(row['entry_bytes'])
            checkpoint_bytes = bytes(row['checkpoint_bytes'])
            proof_bytes = bytes(row['proof_bytes'])
            submission = _load_canonical_model(
                submission_bytes,
                WitnessServiceSubmission,
                f'witness submission {sequence}',
            )
            entry = _load_canonical_model(entry_bytes, WitnessServiceLogEntry, f'witness log entry {sequence}')
            checkpoint = _load_canonical_model(
                checkpoint_bytes,
                WitnessServiceLogCheckpoint,
                f'witness log checkpoint {sequence}',
            )
            proof = _verify_proof_commitment(
                proof_bytes,
                policy=self.policy,
                policy_bytes=self.policy_bytes,
                trust=self.trust_policy,
                checkpoint_sha256=submission.witness_request.checkpoint_sha256,
                checkpoint_bytes=submission.witness_request.checkpoint_bytes,
            )
            if (
                proof.statement.submission != submission
                or proof.statement.entry != entry
                or proof.statement.checkpoint != checkpoint
                or str(row['receipt_id']) != proof.receipt_id
                or str(row['submission_sha256']) != _sha256(submission_bytes)
                or str(row['client_nonce']) != submission.client_nonce
                or str(row['entry_sha256']) != _sha256(entry_bytes)
                or str(row['previous_entry_sha256']) != previous_entry_sha256
                or entry.previous_entry_sha256 != previous_entry_sha256
                or checkpoint.previous_checkpoint_sha256 != previous_checkpoint_sha256
                or str(row['witnessed_at']) != entry.witnessed_at.isoformat()
            ):
                raise WitnessServiceError(f'witness service database row {sequence} is internally inconsistent')
            if previous_time is not None and entry.witnessed_at < previous_time:
                raise WitnessServiceError('witness service log time moved backwards')
            if isinstance(submission.witness_request, RegistryCheckpointWitnessRequest):
                validated = self._validate_registry_checkpoint_request(
                    connection,
                    submission.witness_request,
                    witnessed_at=entry.witnessed_at,
                )
                if validated is None:
                    raise WitnessServiceError('registry witness request was not parsed as a registry head')
                registry_replay.append((sequence, submission.witness_request, validated[0]))
            previous_time = entry.witnessed_at
            previous_entry_sha256 = _sha256(entry_bytes)
            previous_checkpoint_sha256 = _sha256(checkpoint_bytes)
            entry_count = sequence
        self._verify_registry_head_replay(connection, registry_replay)
        return WitnessServiceVerificationReport(
            authority_id=self.policy.authority_id,
            witness_id=self.policy.witness_id,
            entry_count=entry_count,
            through_entry_sha256=previous_entry_sha256,
            through_checkpoint_sha256=previous_checkpoint_sha256,
        )

    def _verify_registry_head_replay(
        self,
        connection: sqlite3.Connection,
        replay: list[tuple[int, RegistryCheckpointWitnessRequest, WitnessedSignedRegistryCheckpoint]],
    ) -> None:
        latest: dict[tuple[str, str], WitnessedSignedRegistryCheckpoint] = {}
        expected_rows: dict[tuple[str, str, int], tuple[str, str, bytes, int]] = {}
        for sequence, request, envelope in replay:
            checkpoint = envelope.checkpoint
            identity = (checkpoint.registry_id, checkpoint.authority_id)
            row_key = (*identity, checkpoint.tree_size)
            envelope_bytes = canonical_json_bytes(envelope)
            existing = expected_rows.get(row_key)
            if existing is not None:
                if existing[2] != envelope_bytes:
                    raise WitnessServiceError('witness log contains a registry split view at one tree size')
                continue
            previous = latest.get(identity)
            if previous is None:
                if checkpoint.tree_size != 0:
                    raise WitnessServiceError('witness log registry history does not begin at genesis')
            else:
                previous_checkpoint = previous.checkpoint
                if checkpoint.tree_size != previous_checkpoint.tree_size + 1:
                    raise WitnessServiceError('witness log registry history has a non-monotonic size gap')
                if (
                    request.consistency_from_tree_size != previous_checkpoint.tree_size
                    or request.consistency_from_root_sha256 != previous_checkpoint.root_sha256
                ):
                    raise WitnessServiceError('witness log registry request does not extend its prior head')
                expected_previous_sha256 = _sha256(
                    canonical_json_bytes(previous_checkpoint)
                    + base64.b64decode(previous.signature_base64, validate=True)
                )
                if checkpoint.previous_checkpoint_sha256 != expected_previous_sha256:
                    raise WitnessServiceError('witness log registry predecessor digest is invalid')
                if not _verify_registry_consistency(
                    old_size=previous_checkpoint.tree_size,
                    new_size=checkpoint.tree_size,
                    old_root=bytes.fromhex(previous_checkpoint.root_sha256),
                    new_root=bytes.fromhex(checkpoint.root_sha256),
                    proof=tuple(bytes.fromhex(item) for item in request.consistency_proof_sha256),
                ):
                    raise WitnessServiceError('witness log registry consistency proof is invalid')
            latest[identity] = envelope
            expected_rows[row_key] = (
                checkpoint.root_sha256,
                _sha256(envelope_bytes),
                envelope_bytes,
                sequence,
            )
        stored_rows = connection.execute(
            'SELECT registry_id, authority_id, tree_size, root_sha256, envelope_sha256, '
            'envelope_bytes, witness_entry_sequence FROM registry_heads '
            'ORDER BY registry_id, authority_id, tree_size'
        ).fetchall()
        stored = {
            (str(row['registry_id']), str(row['authority_id']), int(row['tree_size'])): (
                str(row['root_sha256']),
                str(row['envelope_sha256']),
                bytes(row['envelope_bytes']),
                int(row['witness_entry_sequence']),
            )
            for row in stored_rows
        }
        if stored != expected_rows:
            raise WitnessServiceError('witnessed registry-head index differs from the signed log replay')


class Ed25519WitnessServiceVerifier:
    """Offline verifier for exact proof, policy, key, nonce, digest, and time bindings."""

    def __init__(
        self,
        policy_bytes: bytes,
        trust_policy_bytes: bytes,
        verifier_implementation_bytes: bytes,
        *,
        verifier_id: str,
    ) -> None:
        self.policy_bytes = _exact_bytes(policy_bytes, 'witness service policy', maximum=1024 * 1024)
        self.trust_policy_bytes = _exact_bytes(
            trust_policy_bytes,
            'witness service trust policy',
            maximum=1024 * 1024,
        )
        self.verifier_implementation_bytes = _exact_bytes(
            verifier_implementation_bytes,
            'witness service verifier implementation',
            maximum=16 * 1024 * 1024,
        )
        self.policy = _load_canonical_model(
            self.policy_bytes,
            WitnessServicePolicy,
            'witness service policy',
        )
        self.trust_policy = _load_canonical_model(
            self.trust_policy_bytes,
            WitnessServiceTrustPolicy,
            'witness service trust policy',
        )
        _validate_policy_trust_binding(self.policy, self.policy_bytes, self.trust_policy)
        self.binding = make_witness_service_policy_binding(
            self.policy_bytes,
            self.trust_policy_bytes,
            self.verifier_implementation_bytes,
            verifier_id=verifier_id,
        )

    def __call__(
        self,
        checkpoint_bytes: bytes,
        proof_bytes: bytes,
        policy_binding: WitnessPolicyBinding,
    ) -> AuthenticatedExternalWitnessFacts:
        if not isinstance(checkpoint_bytes, bytes) or not checkpoint_bytes:
            raise WitnessServiceError('offline verifier requires nonempty exact checkpoint bytes')
        if policy_binding != self.binding:
            raise WitnessServiceError('offline verifier received a different out-of-band witness policy binding')
        proof = _verify_proof_commitment(
            proof_bytes,
            policy=self.policy,
            policy_bytes=self.policy_bytes,
            trust=self.trust_policy,
            checkpoint_sha256=_sha256(checkpoint_bytes),
            checkpoint_bytes=len(checkpoint_bytes),
        )
        return AuthenticatedExternalWitnessFacts(
            receipt_id=proof.receipt_id,
            authority_id=self.policy.authority_id,
            witness_id=self.policy.witness_id,
            method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
            policy_id=self.policy.policy_id,
            checkpoint_sha256=_sha256(checkpoint_bytes),
            witnessed_at=proof.statement.entry.witnessed_at,
        )


def verify_witness_service_artifact(
    artifact_bytes: bytes,
    proof_bytes: bytes,
    *,
    policy_bytes: bytes,
    trust_policy_bytes: bytes,
    checkpoint_schema_version: str,
) -> AuthenticatedExternalWitnessFacts:
    """Verify a witness proof over arbitrary exact bytes under pinned materials.

    The generic operations-ledger broker fixes one checkpoint schema.  The plan
    selection registry also needs to witness its exact signed tree-head envelope,
    so this lower-level verifier keeps the same cryptographic checks while requiring
    the caller to pin the expected artifact schema explicitly.
    """

    artifact_bytes = _exact_bytes(artifact_bytes, 'witnessed artifact', maximum=16 * 1024 * 1024)
    policy_bytes = _exact_bytes(policy_bytes, 'witness service policy', maximum=1024 * 1024)
    trust_policy_bytes = _exact_bytes(
        trust_policy_bytes,
        'witness service trust policy',
        maximum=1024 * 1024,
    )
    policy = _load_canonical_model(policy_bytes, WitnessServicePolicy, 'witness service policy')
    trust = _load_canonical_model(
        trust_policy_bytes,
        WitnessServiceTrustPolicy,
        'witness service trust policy',
    )
    _validate_policy_trust_binding(policy, policy_bytes, trust)
    proof = _verify_proof_commitment(
        proof_bytes,
        policy=policy,
        policy_bytes=policy_bytes,
        trust=trust,
        checkpoint_sha256=_sha256(artifact_bytes),
        checkpoint_bytes=len(artifact_bytes),
    )
    if proof.statement.submission.witness_request.checkpoint_schema_version != checkpoint_schema_version:
        raise WitnessServiceError('witness proof binds a different artifact schema version')
    request = proof.statement.submission.witness_request
    if isinstance(request, RegistryCheckpointWitnessRequest):
        try:
            embedded = base64.b64decode(request.signed_checkpoint_base64, validate=True)
        except (TypeError, ValueError) as error:
            raise WitnessServiceError('witness proof embeds invalid registry checkpoint bytes') from error
        if not hmac.compare_digest(embedded, artifact_bytes):
            raise WitnessServiceError('witness proof does not embed the exact signed registry checkpoint')
    return AuthenticatedExternalWitnessFacts(
        receipt_id=proof.receipt_id,
        authority_id=policy.authority_id,
        witness_id=policy.witness_id,
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id=policy.policy_id,
        checkpoint_sha256=_sha256(artifact_bytes),
        witnessed_at=proof.statement.entry.witnessed_at,
    )


class Ed25519WitnessServiceProvider:
    """Authenticated HTTPS client implementing the generic witness provider ABI."""

    def __init__(
        self,
        policy_bytes: bytes,
        *,
        authorization_bearer_token: bytes,
        transport: WitnessServiceTransport | None = None,
    ) -> None:
        self.policy_bytes = _exact_bytes(policy_bytes, 'witness service policy', maximum=1024 * 1024)
        self.policy = _load_canonical_model(
            self.policy_bytes,
            WitnessServicePolicy,
            'witness service policy',
        )
        _validate_bearer_token(authorization_bearer_token)
        self._authorization_bearer_token = authorization_bearer_token
        self._transport = transport or https_witness_service_transport

    def __call__(
        self,
        request: CheckpointWitnessRequest | RegistryCheckpointWitnessRequest,
    ) -> tuple[ExternalWitnessClaim, bytes]:
        if not isinstance(request, (CheckpointWitnessRequest, RegistryCheckpointWitnessRequest)):
            raise WitnessServiceError('witness service provider requires a supported witness request')
        if (
            request.authority_id != self.policy.authority_id
            or request.method is not ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG
            or request.policy_id != self.policy.policy_id
            or request.policy_sha256 != _sha256(self.policy_bytes)
        ):
            raise WitnessServiceError('checkpoint request does not match the exact witness service policy')
        submission = WitnessServiceSubmission(
            witness_request=request,
            client_nonce=secrets.token_hex(self.policy.client_nonce_bytes),
        )
        body = canonical_json_bytes(submission)
        if len(body) > self.policy.max_submission_bytes:
            raise WitnessServiceError('witness service submission exceeds the pinned byte limit')
        response = self._transport(
            WitnessServiceTransportRequest(
                endpoint_uri=self.policy.endpoint_uri,
                body=body,
                timeout_seconds=self.policy.client_timeout_seconds,
                max_response_bytes=self.policy.max_proof_bytes,
                authorization_bearer_token=self._authorization_bearer_token,
            )
        )
        if not isinstance(response, WitnessServiceTransportResponse):
            raise WitnessServiceError('witness service transport returned an invalid response object')
        if response.status_code not in {200, 201}:
            raise WitnessServiceError(f'witness service returned HTTP status {response.status_code}')
        if response.final_uri != self.policy.endpoint_uri:
            raise WitnessServiceError('witness service transport redirected away from the pinned endpoint')
        if (response.content_type or '').strip().lower() != _PROOF_MEDIA_TYPE:
            raise WitnessServiceError('witness service returned an invalid Content-Type')
        if response.content_encoding not in (None, '', 'identity'):
            raise WitnessServiceError('witness service returned a content-encoded proof')
        if response.content_length is not None and response.content_length != len(response.body):
            raise WitnessServiceError('witness service Content-Length does not match exact response bytes')
        proof_bytes = _exact_bytes(
            response.body,
            'witness service response proof',
            maximum=self.policy.max_proof_bytes,
        )
        proof = _load_canonical_model(proof_bytes, WitnessServiceProof, 'witness service response proof')
        if proof.statement.submission != submission:
            raise WitnessServiceError('witness service response does not echo the exact nonce-bound submission')
        verification_uri = _receipt_uri(self.policy.endpoint_uri, proof.receipt_id)
        return ExternalWitnessClaim(verification_uri=verification_uri), proof_bytes


def verify_witness_service_signed_checkpoint(
    signed_checkpoint_bytes: bytes,
    *,
    policy_bytes: bytes,
    trust_policy_bytes: bytes,
) -> WitnessServiceSignedCheckpoint:
    """Offline-verify an exact public log-head response under pinned key bytes."""

    policy_bytes = _exact_bytes(policy_bytes, 'witness service policy', maximum=1024 * 1024)
    trust_policy_bytes = _exact_bytes(
        trust_policy_bytes,
        'witness service trust policy',
        maximum=1024 * 1024,
    )
    policy = _load_canonical_model(policy_bytes, WitnessServicePolicy, 'witness service policy')
    trust = _load_canonical_model(
        trust_policy_bytes,
        WitnessServiceTrustPolicy,
        'witness service trust policy',
    )
    _validate_policy_trust_binding(policy, policy_bytes, trust)
    signed = _load_canonical_model(
        _exact_bytes(
            signed_checkpoint_bytes,
            'signed witness service checkpoint',
            maximum=policy.max_proof_bytes,
        ),
        WitnessServiceSignedCheckpoint,
        'signed witness service checkpoint',
    )
    checkpoint = signed.checkpoint
    if (
        checkpoint.authority_id != policy.authority_id
        or checkpoint.witness_id != policy.witness_id
        or checkpoint.policy_id != policy.policy_id
        or checkpoint.policy_sha256 != _sha256(policy_bytes)
    ):
        raise WitnessServiceError('signed log checkpoint does not match the exact service policy')
    if checkpoint.issued_at < trust.key_valid_from or (
        trust.key_valid_until is not None and checkpoint.issued_at >= trust.key_valid_until
    ):
        raise WitnessServiceError('signed log checkpoint is outside the pinned key validity window')
    try:
        _parse_public_key(trust).verify(
            base64.b64decode(signed.signature_base64, validate=True),
            _signature_payload(_CHECKPOINT_SIGNATURE_DOMAIN, checkpoint),
        )
    except InvalidSignature as error:
        raise WitnessServiceError('signed witness log checkpoint signature verification failed') from error
    return signed


def verify_witness_service_checkpoint_successor(
    previous_signed_checkpoint_bytes: bytes,
    current_signed_checkpoint_bytes: bytes,
    *,
    policy_bytes: bytes,
    trust_policy_bytes: bytes,
) -> tuple[WitnessServiceSignedCheckpoint, WitnessServiceSignedCheckpoint]:
    """Verify two adjacent signed heads and their explicit append-only link."""

    previous = verify_witness_service_signed_checkpoint(
        previous_signed_checkpoint_bytes,
        policy_bytes=policy_bytes,
        trust_policy_bytes=trust_policy_bytes,
    )
    current = verify_witness_service_signed_checkpoint(
        current_signed_checkpoint_bytes,
        policy_bytes=policy_bytes,
        trust_policy_bytes=trust_policy_bytes,
    )
    if current.checkpoint.tree_size != previous.checkpoint.tree_size + 1:
        raise WitnessServiceError('signed witness checkpoints are not adjacent')
    if current.checkpoint.previous_checkpoint_sha256 != _sha256(canonical_json_bytes(previous.checkpoint)):
        raise WitnessServiceError('signed witness checkpoint does not link to its declared predecessor')
    if current.checkpoint.issued_at < previous.checkpoint.issued_at:
        raise WitnessServiceError('signed witness checkpoint time moved backwards')
    return previous, current


def https_witness_service_transport(request: WitnessServiceTransportRequest) -> WitnessServiceTransportResponse:
    """Perform one no-redirect, system-trust HTTPS POST with bounded reads."""

    if not isinstance(request, WitnessServiceTransportRequest):
        raise WitnessServiceError('HTTPS transport requires a WitnessServiceTransportRequest')
    parsed = urlsplit(request.endpoint_uri)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise WitnessServiceError('witness service transport requires an HTTPS endpoint')
    _validate_bearer_token(request.authorization_bearer_token)
    body = _exact_bytes(request.body, 'witness service HTTPS body', maximum=1024 * 1024)
    if request.timeout_seconds < 0.1 or request.timeout_seconds > 120:
        raise WitnessServiceError('witness service transport timeout is invalid')
    if request.max_response_bytes < 1 or request.max_response_bytes > 16 * 1024 * 1024:
        raise WitnessServiceError('witness service transport response bound is invalid')
    authorization = b'Bearer ' + request.authorization_bearer_token
    http_request = urllib.request.Request(
        request.endpoint_uri,
        data=body,
        method='POST',
        headers={
            'Accept': _PROOF_MEDIA_TYPE,
            'Authorization': authorization.decode('ascii'),
            'Content-Type': _JSON_MEDIA_TYPE,
            'User-Agent': 'vaxreplay-witness-client/0.1',
        },
    )
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        _NoRedirectHandler(),
    )
    try:
        with opener.open(http_request, timeout=request.timeout_seconds) as response:
            response_body = response.read(request.max_response_bytes + 1)
            if len(response_body) > request.max_response_bytes:
                raise WitnessServiceError('witness service response exceeds the pinned byte limit')
            content_length_header = response.headers.get('Content-Length')
            try:
                content_length = None if content_length_header is None else int(content_length_header)
            except ValueError as error:
                raise WitnessServiceError('witness service returned an invalid Content-Length') from error
            return WitnessServiceTransportResponse(
                status_code=int(response.status),
                content_type=response.headers.get('Content-Type'),
                body=response_body,
                final_uri=response.geturl(),
                content_encoding=response.headers.get('Content-Encoding'),
                content_length=content_length,
            )
    except urllib.error.HTTPError as error:
        raise WitnessServiceError(f'witness service returned HTTP status {error.code}') from error
    except (OSError, urllib.error.URLError) as error:
        raise WitnessServiceError(f'witness service HTTPS transport failed: {error}') from error


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise WitnessServiceError('witness service redirect was rejected')


def build_witness_http_server(
    store: WitnessServiceStore,
    *,
    host: str,
    port: int,
    authorization_bearer_token: bytes,
    tls_context: ssl.SSLContext | None = None,
    allow_insecure_loopback: bool = False,
) -> http.server.ThreadingHTTPServer:
    """Build the bounded authenticated-write/public-read network service.

    Production callers must supply ``tls_context``.  Plain HTTP is available only
    behind the explicit loopback test/development switch.  The bearer token lives in
    process memory only and is never written or logged by this service.
    """

    if not isinstance(store, WitnessServiceStore):
        raise WitnessServiceError('HTTP service requires a WitnessServiceStore')
    _validate_bearer_token(authorization_bearer_token)
    if not isinstance(host, str) or not host or any(character in host for character in '\x00\r\n'):
        raise WitnessServiceError('HTTP service host is invalid')
    if not isinstance(port, int) or isinstance(port, bool) or port < 0 or port > 65535:
        raise WitnessServiceError('HTTP service port is invalid')
    if tls_context is None and (not allow_insecure_loopback or not _is_loopback_host(host)):
        raise WitnessServiceError('TLS is required unless insecure transport is explicitly limited to loopback')
    token = b'Bearer ' + authorization_bearer_token

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = 'VaxReplayWitness/0.1'
        sys_version = ''

        def log_message(self, format: str, *args: object) -> None:
            # Never let default request logging expose authorization or request details.
            del format, args
            return

        def do_POST(self) -> None:  # noqa: N802
            if self.path != '/v1/witness':
                self._error(404, 'not_found')
                return
            authorization_headers = self.headers.get_all('Authorization', failobj=[])
            supplied_authorization = authorization_headers[0] if len(authorization_headers) == 1 else None
            supplied_bytes = b'' if supplied_authorization is None else supplied_authorization.encode('ascii', 'ignore')
            if not hmac.compare_digest(supplied_bytes, token):
                self.close_connection = True
                self._error(401, 'unauthorized', authenticate=True)
                return
            if self.headers.get('Transfer-Encoding') is not None:
                self.close_connection = True
                self._error(400, 'transfer_encoding_forbidden')
                return
            content_type_headers = self.headers.get_all('Content-Type', failobj=[])
            if len(content_type_headers) != 1 or content_type_headers[0].strip().lower() != _JSON_MEDIA_TYPE:
                self.close_connection = True
                self._error(415, 'unsupported_media_type')
                return
            content_length_headers = self.headers.get_all('Content-Length', failobj=[])
            raw_length = content_length_headers[0] if len(content_length_headers) == 1 else None
            try:
                content_length = int(raw_length or '')
            except ValueError:
                self.close_connection = True
                self._error(411, 'content_length_required')
                return
            if content_length < 1 or content_length > store.policy.max_submission_bytes:
                self.close_connection = True
                self._error(413, 'submission_too_large')
                return
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                self.close_connection = True
                self._error(400, 'truncated_submission')
                return
            try:
                issuance = store.issue(body)
            except WitnessServiceError:
                self._error(400, 'invalid_submission')
                return
            except Exception:
                self.close_connection = True
                self._error(503, 'service_unavailable')
                return
            self._send(
                201 if issuance.created else 200,
                issuance.proof_bytes,
                content_type=_PROOF_MEDIA_TYPE,
                cache_control='public, immutable, max-age=31536000',
            )

        def do_GET(self) -> None:  # noqa: N802
            if self.path == '/healthz':
                self._send(200, b'{"status":"ok"}', content_type=_JSON_MEDIA_TYPE, cache_control='no-store')
                return
            if self.path == '/v1/checkpoint':
                try:
                    payload = store.latest_signed_checkpoint_bytes()
                except WitnessServiceError:
                    self._error(404, 'checkpoint_not_found')
                    return
                self._send(200, payload, content_type=_CHECKPOINT_MEDIA_TYPE, cache_control='no-store')
                return
            checkpoint_prefix = '/v1/checkpoints/'
            if self.path.startswith(checkpoint_prefix) and '?' not in self.path and '#' not in self.path:
                raw_sequence = self.path[len(checkpoint_prefix) :]
                if not raw_sequence.isascii() or not raw_sequence.isdecimal() or raw_sequence.startswith('0'):
                    self._error(404, 'checkpoint_not_found')
                    return
                try:
                    payload = store.signed_checkpoint_bytes(int(raw_sequence))
                except WitnessServiceError:
                    self._error(404, 'checkpoint_not_found')
                    return
                self._send(
                    200,
                    payload,
                    content_type=_CHECKPOINT_MEDIA_TYPE,
                    cache_control='public, immutable, max-age=31536000',
                )
                return
            entry_prefix = '/v1/entries/'
            if self.path.startswith(entry_prefix) and '?' not in self.path and '#' not in self.path:
                raw_sequence = self.path[len(entry_prefix) :]
                if not raw_sequence.isascii() or not raw_sequence.isdecimal() or raw_sequence.startswith('0'):
                    self._error(404, 'entry_not_found')
                    return
                try:
                    payload = store.proof_bytes_at_sequence(int(raw_sequence))
                except WitnessServiceError:
                    self._error(404, 'entry_not_found')
                    return
                self._send(
                    200,
                    payload,
                    content_type=_PROOF_MEDIA_TYPE,
                    cache_control='public, immutable, max-age=31536000',
                )
                return
            prefix = '/v1/receipts/'
            if self.path.startswith(prefix) and '?' not in self.path and '#' not in self.path:
                receipt_id = self.path[len(prefix) :]
                try:
                    payload = store.receipt_bytes(receipt_id)
                except WitnessServiceError:
                    self._error(404, 'receipt_not_found')
                    return
                self._send(
                    200,
                    payload,
                    content_type=_PROOF_MEDIA_TYPE,
                    cache_control='public, immutable, max-age=31536000',
                )
                return
            self._error(404, 'not_found')

        def _error(self, status_code: int, code: str, *, authenticate: bool = False) -> None:
            payload = canonical_json_bytes({'error': code})
            extra_headers = {'WWW-Authenticate': 'Bearer'} if authenticate else None
            self._send(
                status_code,
                payload,
                content_type=_JSON_MEDIA_TYPE,
                cache_control='no-store',
                extra_headers=extra_headers,
            )

        def _send(
            self,
            status_code: int,
            payload: bytes,
            *,
            content_type: str,
            cache_control: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status_code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', cache_control)
            self.send_header('X-Content-Type-Options', 'nosniff')
            if extra_headers is not None:
                for name, value in extra_headers.items():
                    self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

    server = _QuietThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    if tls_context is not None:
        server.socket = tls_context.wrap_socket(server.socket, server_side=True)
    return server


class _QuietThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """Suppress default traceback logging, which is inappropriate at a secret boundary."""

    def handle_error(self, request, client_address):  # type: ignore[no-untyped-def]
        del request, client_address


def _receipt_uri(endpoint_uri: str, receipt_id: str) -> str:
    suffix = '/v1/witness'
    if not endpoint_uri.endswith(suffix):
        raise WitnessServiceError('witness endpoint does not have the required /v1/witness path')
    return endpoint_uri[: -len(suffix)] + f'/v1/receipts/{receipt_id}'


def _validate_bearer_token(token: bytes) -> None:
    if not isinstance(token, bytes) or len(token) < 32 or len(token) > 4096:
        raise WitnessServiceError('write bearer token must contain between 32 and 4096 bytes')
    if any(byte < 0x21 or byte > 0x7E for byte in token):
        raise WitnessServiceError('write bearer token must contain only visible ASCII bytes')


def _is_loopback_host(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_service_root(root: Path) -> Path:
    requested = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    try:
        metadata = os.lstat(requested)
    except OSError as error:
        raise WitnessServiceError(f'cannot inspect witness service root: {error}') from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise WitnessServiceError('witness service root must be a real directory, not a symlink')
    if metadata.st_mode & 0o022:
        raise WitnessServiceError('witness service root must not be group- or world-writable')
    if hasattr(os, 'geteuid') and metadata.st_uid != os.geteuid():
        raise WitnessServiceError('witness service root must be owned by the service process user')
    return requested


def _read_regular_nofollow(path: Path, maximum: int, *, private: bool) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise WitnessServiceError(f'cannot open witness service file {path.name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise WitnessServiceError(f'witness service file {path.name} must be a singly linked regular file')
        forbidden_mode = 0o077 if private else 0o022
        if metadata.st_mode & forbidden_mode:
            raise WitnessServiceError(f'witness service file {path.name} has unsafe permissions')
        if metadata.st_size < 1 or metadata.st_size > maximum:
            raise WitnessServiceError(f'witness service file {path.name} has an invalid size')
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != metadata.st_size or len(payload) > maximum:
            raise WitnessServiceError(f'witness service file {path.name} changed or exceeded its byte limit')
        after = os.fstat(descriptor)
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
        ):
            raise WitnessServiceError(f'witness service file {path.name} changed while it was read')
        return bytes(payload)
    finally:
        os.close(descriptor)


def _validate_database_file(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise WitnessServiceError(f'cannot inspect witness service database: {error}') from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise WitnessServiceError('witness service database must be a singly linked regular file')
    if metadata.st_mode & 0o077:
        raise WitnessServiceError('witness service database has unsafe permissions')


def _write_durable_file(path: Path, payload: bytes, *, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0), mode)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count < 1:
                raise OSError('short write while creating witness service state')
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
