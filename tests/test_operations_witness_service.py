from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.schema import LedgerCheckpoint, checkpoint_bytes
from vaxreplay.operations.selection_registry import (
    RegistryCheckpoint,
    SelectionRegistryPolicy,
    SignedRegistryCheckpoint,
    build_signed_registry_genesis_checkpoint,
    checkpoint_witness_request,
)
from vaxreplay.operations.witness import (
    CheckpointWitnessRequest,
    ExternalWitnessMethod,
    WitnessVerificationError,
    broker_witness_checkpoint,
    load_witnessed_checkpoint,
)
from vaxreplay.operations.witness_service import (
    Ed25519WitnessServiceProvider,
    Ed25519WitnessServiceVerifier,
    WitnessServiceError,
    WitnessServiceStore,
    WitnessServiceTransportResponse,
    build_witness_http_server,
    verify_witness_service_checkpoint_successor,
    verify_witness_service_signed_checkpoint,
)
from vaxreplay.operations.witness_service_cli import main as witness_service_main
from vaxreplay.operations.witness_service_schema import (
    WitnessRegistryMonitor,
    WitnessRegistrySigningKey,
    WitnessServiceProof,
    WitnessServiceSubmission,
)

_TOKEN = b'service-write-token-with-at-least-32-bytes'
_VERIFIER_BYTES = b'fictional hermetic ed25519 witness verifier artifact v1'
_PROOF_MEDIA_TYPE = 'application/vnd.vaxreplay.witness-proof+json'


@pytest.fixture
def service(tmp_path: Path) -> WitnessServiceStore:
    return WitnessServiceStore.initialize(
        tmp_path / 'service',
        authority_id='independent-witness-operator',
        witness_id='witness-key-2026-07',
        policy_id='vaxreplay-checkpoint-witness-v1',
        trust_policy_id='witness-root-2026-07',
        endpoint_uri='https://witness.invalid/v1/witness',
    )


def _request(service: WitnessServiceStore, *, digest: str = 'a' * 64, size: int = 123) -> CheckpointWitnessRequest:
    return CheckpointWitnessRequest(
        checkpoint_sha256=digest,
        checkpoint_bytes=size,
        authority_id=service.policy.authority_id,
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id=service.policy.policy_id,
        policy_sha256=hashlib.sha256(service.policy_bytes).hexdigest(),
    )


def _submission(
    service: WitnessServiceStore,
    *,
    ordinal: int = 1,
    nonce: str | None = None,
) -> WitnessServiceSubmission:
    checkpoint_payload = _checkpoint_payload(ordinal)
    return WitnessServiceSubmission(
        witness_request=_request(
            service,
            digest=hashlib.sha256(checkpoint_payload).hexdigest(),
            size=len(checkpoint_payload),
        ),
        client_nonce=nonce or f'{ordinal + 1000:064x}',
    )


def _checkpoint_payload(ordinal: int) -> bytes:
    return f'canonical-fictional-checkpoint-{ordinal}'.encode('ascii')


def _verifier(service: WitnessServiceStore) -> Ed25519WitnessServiceVerifier:
    return Ed25519WitnessServiceVerifier(
        service.policy_bytes,
        service.trust_policy_bytes,
        _VERIFIER_BYTES,
        verifier_id='ed25519-offline-verifier-v1',
    )


def test_service_provider_and_offline_verifier_integrate_with_generic_broker(
    service: WitnessServiceStore,
    tmp_path: Path,
) -> None:
    checkpoint = LedgerCheckpoint(
        store_id='1' * 32,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        through_sequence=4,
        through_event_sha256='2' * 64,
        object_count=2,
        object_inventory_sha256='3' * 64,
    )

    def transport(request):
        issuance = service.issue(request.body)
        return WitnessServiceTransportResponse(
            status_code=201 if issuance.created else 200,
            content_type=_PROOF_MEDIA_TYPE,
            body=issuance.proof_bytes,
            final_uri=request.endpoint_uri,
            content_length=len(issuance.proof_bytes),
        )

    provider = Ed25519WitnessServiceProvider(
        service.policy_bytes,
        authorization_bearer_token=_TOKEN,
        transport=transport,
    )
    verifier = _verifier(service)
    built = broker_witness_checkpoint(
        tmp_path / 'witnessed',
        checkpoint=checkpoint,
        policy=verifier.binding,
        provider=provider,
        verifier=verifier,
    )
    assert built.manifest.receipt.method is ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG
    assert built.manifest.receipt.witness_id == service.policy.witness_id
    assert built.manifest.receipt.verification_uri.endswith(f'/v1/receipts/{built.manifest.receipt.receipt_id}')
    proof = WitnessServiceProof.model_validate_json(built.proof_bytes)
    assert (
        proof.statement.submission.witness_request.checkpoint_sha256
        == hashlib.sha256(checkpoint_bytes(checkpoint)).hexdigest()
    )
    assert not hasattr(proof.statement.submission, 'checkpoint_body')
    loaded = load_witnessed_checkpoint(
        built.root,
        verifier=verifier,
        expected_policy=verifier.binding,
        expected_checkpoint_sha256=hashlib.sha256(checkpoint_bytes(checkpoint)).hexdigest(),
    )
    assert loaded == built
    assert service.verify().entry_count == 1


def test_nonce_digest_policy_and_signature_are_strictly_bound(service: WitnessServiceStore) -> None:
    submission = _submission(service)
    issued = service.issue(canonical_json_bytes(submission))
    verifier = _verifier(service)
    fake_checkpoint = b'x' * submission.witness_request.checkpoint_bytes
    with pytest.raises(WitnessServiceError, match='different checkpoint commitment'):
        verifier(fake_checkpoint, issued.proof_bytes, verifier.binding)

    proof = WitnessServiceProof.model_validate_json(issued.proof_bytes)
    changed_submission = proof.statement.submission.model_copy(update={'client_nonce': 'f' * 64})
    changed_statement = proof.statement.model_copy(update={'submission': changed_submission})
    changed_proof = proof.model_copy(update={'statement': changed_statement})
    with pytest.raises(WitnessServiceError, match='receipt_id|submission'):
        verifier(
            _checkpoint_payload(1),
            canonical_json_bytes(changed_proof),
            verifier.binding,
        )

    altered_signature = bytearray(issued.proof_bytes)
    signature_location = issued.proof_bytes.index(proof.receipt_signature_base64.encode('ascii'))
    altered_signature[signature_location] = ord('A') if altered_signature[signature_location] != ord('A') else ord('B')
    with pytest.raises(WitnessServiceError, match='signature'):
        # Internal commitment verification is reached through a structurally valid proof.
        verifier(
            _checkpoint_payload(1),
            bytes(altered_signature),
            verifier.binding,
        )


def test_concurrent_issuance_is_contiguous_and_exact_retry_is_idempotent(service: WitnessServiceStore) -> None:
    submissions = [canonical_json_bytes(_submission(service, ordinal=index)) for index in range(1, 17)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        issued = list(executor.map(service.issue, submissions))
    assert sorted(result.sequence for result in issued) == list(range(1, 17))
    assert len({result.receipt_id for result in issued}) == 16
    report = service.verify()
    assert report.entry_count == 16

    retry = service.issue(submissions[4])
    original = issued[4]
    assert retry.created is False
    assert retry.sequence == original.sequence
    assert retry.receipt_id == original.receipt_id
    assert retry.proof_bytes == original.proof_bytes
    assert service.verify().entry_count == 16


def test_nonce_reuse_with_different_commitment_fails_without_append(service: WitnessServiceStore) -> None:
    nonce = 'd' * 64
    service.issue(canonical_json_bytes(_submission(service, ordinal=1, nonce=nonce)))
    with pytest.raises(WitnessServiceError, match='nonce was already used'):
        service.issue(canonical_json_bytes(_submission(service, ordinal=2, nonce=nonce)))
    assert service.verify().entry_count == 1


def test_clock_rollback_fails_closed_without_synthesizing_time(service: WitnessServiceStore) -> None:
    first_time = datetime.now(timezone.utc) + timedelta(seconds=2)
    with patch(
        'vaxreplay.operations.witness_service._security_time',
        side_effect=[first_time, first_time - timedelta(microseconds=1)],
    ):
        first = service.issue(canonical_json_bytes(_submission(service, ordinal=1)))
        with pytest.raises(WitnessServiceError, match='clock moved backwards'):
            service.issue(canonical_json_bytes(_submission(service, ordinal=2)))
    proof = WitnessServiceProof.model_validate_json(first.proof_bytes)
    assert proof.statement.entry.witnessed_at == first_time
    assert service.verify().entry_count == 1


def test_signing_failure_rolls_back_the_transaction(service: WitnessServiceStore) -> None:
    with patch('vaxreplay.operations.witness_service._build_signed_proof', side_effect=RuntimeError('signer failed')):
        with pytest.raises(RuntimeError, match='signer failed'):
            service.issue(canonical_json_bytes(_submission(service)))
    assert service.verify().entry_count == 0


def test_database_triggers_reject_update_and_delete(service: WitnessServiceStore) -> None:
    service.issue(canonical_json_bytes(_submission(service)))
    connection = sqlite3.connect(service.root / 'witness.sqlite3')
    try:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            connection.execute("UPDATE entries SET witnessed_at = '2000-01-01T00:00:00+00:00' WHERE sequence = 1")
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            connection.execute('DELETE FROM entries WHERE sequence = 1')
    finally:
        connection.close()
    assert service.verify().entry_count == 1


def test_startup_and_every_issuance_replay_a_tampered_predecessor(
    service: WitnessServiceStore,
) -> None:
    service.issue(canonical_json_bytes(_submission(service, ordinal=1)))
    service.issue(canonical_json_bytes(_submission(service, ordinal=2)))
    connection = sqlite3.connect(service.root / 'witness.sqlite3')
    try:
        connection.execute('DROP TRIGGER entries_no_update')
        connection.execute(
            'UPDATE entries SET previous_entry_sha256 = ? WHERE sequence = 2',
            ('f' * 64,),
        )
        connection.execute(
            'CREATE TRIGGER entries_no_update BEFORE UPDATE ON entries BEGIN '
            "SELECT RAISE(ABORT, 'witness log is append-only'); END"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(WitnessServiceError, match='internally inconsistent'):
        service.issue(canonical_json_bytes(_submission(service, ordinal=3)))
    with pytest.raises(WitnessServiceError, match='internally inconsistent'):
        WitnessServiceStore(service.root)


def test_registry_monitor_rejects_same_size_split_view_and_bad_consistency(tmp_path: Path) -> None:
    registry_key = Ed25519PrivateKey.generate()
    registry_public = registry_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    now = datetime.now(timezone.utc)
    registry_policy = SelectionRegistryPolicy(
        registry_id='monitored-selection-registry',
        authority_id='monitored-registry-authority',
        policy_id='monitored-registry-policy',
    )
    monitor = WitnessRegistryMonitor(
        registry_id=registry_policy.registry_id,
        authority_id=registry_policy.authority_id,
        signing_keys=(
            WitnessRegistrySigningKey(
                key_id='registry-key-1',
                public_key_base64=base64.b64encode(registry_public).decode('ascii'),
                valid_from=now - timedelta(days=1),
            ),
        ),
    )
    monitored = WitnessServiceStore.initialize(
        tmp_path / 'monitored-witness',
        authority_id='independent-monitor-authority',
        witness_id='independent-monitor-key',
        policy_id='stateful-registry-monitor-v1',
        trust_policy_id='stateful-registry-monitor-trust-v1',
        endpoint_uri='https://monitor.invalid/v1/witness',
        registry_monitors=(monitor,),
    )

    def transport(request):
        issued = monitored.issue(request.body)
        return WitnessServiceTransportResponse(
            status_code=201,
            content_type=_PROOF_MEDIA_TYPE,
            body=issued.proof_bytes,
            final_uri=request.endpoint_uri,
            content_length=len(issued.proof_bytes),
        )

    provider = Ed25519WitnessServiceProvider(
        monitored.policy_bytes,
        authorization_bearer_token=_TOKEN,
        transport=transport,
    )
    genesis = build_signed_registry_genesis_checkpoint(
        registry_policy=registry_policy,
        signing_key=registry_key,
        signing_key_id='registry-key-1',
        issued_at=now - timedelta(seconds=2),
    )
    genesis_bytes = canonical_json_bytes(genesis)
    provider(
        checkpoint_witness_request(
            genesis_bytes,
            monitored.policy,
            consistency_from_tree_size=0,
            consistency_from_root_sha256=hashlib.sha256(b'').hexdigest(),
            consistency_proof_sha256=(),
        )
    )
    previous_sha256 = hashlib.sha256(
        canonical_json_bytes(genesis.checkpoint) + base64.b64decode(genesis.signature_base64, validate=True)
    ).hexdigest()

    def signed_head(root_sha256: str, *, tree_size: int, previous: str) -> SignedRegistryCheckpoint:
        checkpoint = RegistryCheckpoint(
            registry_id=registry_policy.registry_id,
            authority_id=registry_policy.authority_id,
            tree_size=tree_size,
            root_sha256=root_sha256,
            issued_at_upper_bound=now - timedelta(seconds=1),
            signing_key_id='registry-key-1',
            previous_checkpoint_sha256=previous,
        )
        return SignedRegistryCheckpoint(
            checkpoint=checkpoint,
            signature_base64=base64.b64encode(registry_key.sign(canonical_json_bytes(checkpoint))).decode('ascii'),
        )

    first = signed_head('1' * 64, tree_size=1, previous=previous_sha256)
    first_bytes = canonical_json_bytes(first)
    provider(
        checkpoint_witness_request(
            first_bytes,
            monitored.policy,
            consistency_from_tree_size=0,
            consistency_from_root_sha256=hashlib.sha256(b'').hexdigest(),
            consistency_proof_sha256=(),
        )
    )
    conflicting = signed_head('2' * 64, tree_size=1, previous=previous_sha256)
    with pytest.raises(WitnessServiceError, match='split view'):
        provider(
            checkpoint_witness_request(
                canonical_json_bytes(conflicting),
                monitored.policy,
                consistency_from_tree_size=0,
                consistency_from_root_sha256=hashlib.sha256(b'').hexdigest(),
                consistency_proof_sha256=(),
            )
        )

    first_sha256 = hashlib.sha256(
        canonical_json_bytes(first.checkpoint) + base64.b64decode(first.signature_base64, validate=True)
    ).hexdigest()
    second = signed_head('3' * 64, tree_size=2, previous=first_sha256)
    with pytest.raises(WitnessServiceError, match='consistency'):
        provider(
            checkpoint_witness_request(
                canonical_json_bytes(second),
                monitored.policy,
                consistency_from_tree_size=1,
                consistency_from_root_sha256=first.checkpoint.root_sha256,
                consistency_proof_sha256=('0' * 64,),
            )
        )
    assert monitored.verify().entry_count == 2
    assert monitored.proof_bytes_at_sequence(2)


def test_signed_checkpoint_is_publicly_offline_verifiable_and_tamper_evident(
    service: WitnessServiceStore,
) -> None:
    service.issue(canonical_json_bytes(_submission(service)))
    service.issue(canonical_json_bytes(_submission(service, ordinal=2)))
    first_payload = service.signed_checkpoint_bytes(1)
    payload = service.latest_signed_checkpoint_bytes()
    signed = verify_witness_service_signed_checkpoint(
        payload,
        policy_bytes=service.policy_bytes,
        trust_policy_bytes=service.trust_policy_bytes,
    )
    assert signed.checkpoint.tree_size == 2
    previous, current = verify_witness_service_checkpoint_successor(
        first_payload,
        payload,
        policy_bytes=service.policy_bytes,
        trust_policy_bytes=service.trust_policy_bytes,
    )
    assert previous.checkpoint.tree_size == 1
    assert current.checkpoint.tree_size == 2
    altered = payload.replace(b'"tree_size":2', b'"tree_size":3')
    with pytest.raises(WitnessServiceError, match='signature'):
        verify_witness_service_signed_checkpoint(
            altered,
            policy_bytes=service.policy_bytes,
            trust_policy_bytes=service.trust_policy_bytes,
        )


def test_wrong_trust_key_and_wrong_generic_binding_fail_closed(
    service: WitnessServiceStore,
    tmp_path: Path,
) -> None:
    submission = _submission(service)
    issued = service.issue(canonical_json_bytes(submission))
    verifier = _verifier(service)
    with pytest.raises(WitnessServiceError, match='different out-of-band'):
        wrong_binding = verifier.binding.model_copy(update={'trust_policy_sha256': 'f' * 64})
        verifier(b'x', issued.proof_bytes, wrong_binding)

    other = WitnessServiceStore.initialize(
        tmp_path / 'other',
        authority_id=service.policy.authority_id,
        witness_id=service.policy.witness_id,
        policy_id=service.policy.policy_id,
        trust_policy_id='other-trust-root',
        endpoint_uri=service.policy.endpoint_uri,
    )
    wrong_trust = other.trust_policy.model_copy(update={'key_valid_from': service.trust_policy.key_valid_from})
    wrong_key_verifier = Ed25519WitnessServiceVerifier(
        service.policy_bytes,
        canonical_json_bytes(wrong_trust),
        _VERIFIER_BYTES,
        verifier_id='ed25519-offline-verifier-v1',
    )
    with pytest.raises(WitnessServiceError, match='signature'):
        wrong_key_verifier(
            _checkpoint_payload(1),
            issued.proof_bytes,
            wrong_key_verifier.binding,
        )


def test_http_service_authenticates_bounded_writes_and_exposes_public_proofs(
    service: WitnessServiceStore,
) -> None:
    server = build_witness_http_server(
        service,
        host='127.0.0.1',
        port=0,
        authorization_bearer_token=_TOKEN,
        allow_insecure_loopback=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    assert isinstance(host, str)
    assert isinstance(port, int)
    body = canonical_json_bytes(_submission(service))
    try:
        unauthorized = http.client.HTTPConnection(host, port, timeout=5)
        unauthorized.request(
            'POST',
            '/v1/witness',
            body=body,
            headers={'Content-Type': 'application/json'},
        )
        response = unauthorized.getresponse()
        assert response.status == 401
        response.read()
        unauthorized.close()

        oversized = http.client.HTTPConnection(host, port, timeout=5)
        oversized.putrequest('POST', '/v1/witness')
        oversized.putheader('Authorization', f'Bearer {_TOKEN.decode("ascii")}')
        oversized.putheader('Content-Type', 'application/json')
        oversized.putheader('Content-Length', str(service.policy.max_submission_bytes + 1))
        oversized.endheaders()
        response = oversized.getresponse()
        assert response.status == 413
        response.read()
        oversized.close()

        authorized = http.client.HTTPConnection(host, port, timeout=5)
        authorized.request(
            'POST',
            '/v1/witness',
            body=body,
            headers={
                'Authorization': f'Bearer {_TOKEN.decode("ascii")}',
                'Content-Type': 'application/json',
            },
        )
        response = authorized.getresponse()
        proof_bytes = response.read()
        assert response.status == 201
        assert response.getheader('Content-Type') == _PROOF_MEDIA_TYPE
        proof = WitnessServiceProof.model_validate_json(proof_bytes)
        authorized.close()

        public = http.client.HTTPConnection(host, port, timeout=5)
        public.request('GET', f'/v1/receipts/{proof.receipt_id}')
        response = public.getresponse()
        assert response.status == 200
        assert response.read() == proof_bytes
        public.request('GET', '/v1/entries/1')
        response = public.getresponse()
        assert response.status == 200
        assert response.read() == proof_bytes
        public.request('GET', '/v1/checkpoint')
        response = public.getresponse()
        assert response.status == 200
        verify_witness_service_signed_checkpoint(
            response.read(),
            policy_bytes=service.policy_bytes,
            trust_policy_bytes=service.trust_policy_bytes,
        )
        public.request('GET', '/v1/checkpoints/1')
        response = public.getresponse()
        assert response.status == 200
        verify_witness_service_signed_checkpoint(
            response.read(),
            policy_bytes=service.policy_bytes,
            trust_policy_bytes=service.trust_policy_bytes,
        )
        public.close()
        for path in service.root.iterdir():
            if path.is_file():
                assert _TOKEN not in path.read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_plaintext_nonloopback_is_forbidden_and_private_state_permissions_are_strict(
    service: WitnessServiceStore,
) -> None:
    with pytest.raises(WitnessServiceError, match='TLS is required'):
        build_witness_http_server(
            service,
            host='0.0.0.0',
            port=0,
            authorization_bearer_token=_TOKEN,
            allow_insecure_loopback=True,
        )
    assert stat_mode(service.root / 'ed25519-private-key.bin') & 0o077 == 0
    assert stat_mode(service.root / 'witness.sqlite3') & 0o077 == 0
    with pytest.raises(WitnessServiceError, match='already exists'):
        WitnessServiceStore.initialize(
            service.root,
            authority_id='different',
            witness_id='different',
            policy_id='different',
            trust_policy_id='different',
            endpoint_uri='https://different.invalid/v1/witness',
        )


def test_service_state_loader_rejects_symlinked_key_and_root(tmp_path: Path) -> None:
    service = WitnessServiceStore.initialize(
        tmp_path / 'service-for-symlink-test',
        authority_id='symlink-test-authority',
        witness_id='symlink-test-witness',
        policy_id='symlink-test-policy',
        trust_policy_id='symlink-test-trust',
        endpoint_uri='https://symlink-test.invalid/v1/witness',
    )
    key_path = service.root / 'ed25519-private-key.bin'
    external_key = tmp_path / 'external-key.bin'
    external_key.write_bytes(key_path.read_bytes())
    external_key.chmod(0o600)
    key_path.unlink()
    key_path.symlink_to(external_key)
    with pytest.raises(WitnessServiceError, match='cannot open witness service file'):
        WitnessServiceStore(service.root)

    root_link = tmp_path / 'root-link'
    root_link.symlink_to(service.root, target_is_directory=True)
    with pytest.raises(WitnessServiceError, match='real directory'):
        WitnessServiceStore(root_link)


def test_outer_witness_rejects_service_time_after_local_verification(service: WitnessServiceStore) -> None:
    checkpoint = LedgerCheckpoint(
        store_id='1' * 32,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        through_sequence=1,
        through_event_sha256='2' * 64,
        object_count=0,
        object_inventory_sha256='3' * 64,
    )
    verifier = _verifier(service)

    def transport(request):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        with patch('vaxreplay.operations.witness_service._security_time', return_value=future):
            issuance = service.issue(request.body)
        return WitnessServiceTransportResponse(
            status_code=201,
            content_type=_PROOF_MEDIA_TYPE,
            body=issuance.proof_bytes,
            final_uri=request.endpoint_uri,
        )

    provider = Ed25519WitnessServiceProvider(
        service.policy_bytes,
        authorization_bearer_token=_TOKEN,
        transport=transport,
    )
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(WitnessVerificationError, match='verification time predates'):
            broker_witness_checkpoint(
                Path(directory) / 'future',
                checkpoint=checkpoint,
                policy=verifier.binding,
                provider=provider,
                verifier=verifier,
                verified_at=datetime.now(timezone.utc),
            )


def test_cli_initializes_and_verifies_service_state(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / 'cli-service'
    witness_service_main(
        [
            'init',
            '--root',
            str(root),
            '--authority-id',
            'independent-cli-authority',
            '--witness-id',
            'cli-witness-key',
            '--policy-id',
            'cli-policy',
            '--trust-policy-id',
            'cli-trust-policy',
            '--endpoint-uri',
            'https://cli-witness.invalid/v1/witness',
            '--dev-local-root-key',
        ]
    )
    initialized = json.loads(capfd.readouterr().out)
    assert initialized['entry_count'] == 0
    assert initialized['client_time_accepted'] is False
    witness_service_main(['verify', '--root', str(root), '--dev-local-root-key'])
    verified = json.loads(capfd.readouterr().out)
    assert verified['signatures_verified'] is True
    assert verified['hash_chain_verified'] is True


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode
