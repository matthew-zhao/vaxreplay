from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import os
import sqlite3
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
from cryptography.x509.oid import NameOID

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.plan_selection import (
    PlanSelectionCommitment,
    PlanSelectionIntegrityError,
    PlanSelectionRequest,
    broker_plan_selection,
)
from vaxreplay.operations.selection_registry import (
    RegistryConflictError,
    RegistryLogEntry,
    RegistryPinnedCheckpoint,
    RegistrySelectionProof,
    RegistryTrustedSigningKey,
    RegistryWitnessUnavailableError,
    SelectionRegistryError,
    SelectionRegistryPolicy,
    SelectionRegistryTrustPolicy,
    SQLitePlanSelectionRegistry,
    build_plan_selection_policy_binding,
    build_signed_registry_genesis_checkpoint,
    checkpoint_witness_request,
    consistency_proof,
    ed25519_public_key_base64,
    generate_ed25519_private_key,
    inclusion_proof,
    load_ed25519_private_key,
    merkle_root,
    production_plan_selection_materials,
    selection_registry_verifier_implementation_bytes,
    verify_consistency_proof,
    verify_inclusion_proof,
    verify_signed_registry_checkpoint,
)
from vaxreplay.operations.selection_registry_service import (
    HttpsPlanSelectionRegistryProvider,
    SelectionRegistryHTTPServer,
)
from vaxreplay.operations.witness_service import (
    Ed25519WitnessServiceProvider,
    WitnessServiceStore,
    WitnessServiceTransportResponse,
)
from vaxreplay.operations.witness_service_schema import (
    WitnessRegistryMonitor,
    WitnessRegistrySigningKey,
    WitnessServicePolicy,
    WitnessServiceProof,
    WitnessServiceTrustPolicy,
)

_T0 = datetime.now(timezone.utc).replace(microsecond=0)
_TOKEN = 'registry-test-token-' + ('x' * 64)


def _policy() -> SelectionRegistryPolicy:
    return SelectionRegistryPolicy(
        registry_id='independent-vax-plan-registry',
        authority_id='vax-benchmark-registry-authority',
        policy_id='atomic-fww-policy-v1',
    )


def _trust(
    key: Ed25519PrivateKey,
    *,
    witness_policy: WitnessServicePolicy,
    witness_trust: WitnessServiceTrustPolicy,
    signed_checkpoint_bytes: bytes,
    witness_proof_bytes: bytes,
    key_id: str = 'registry-key-2026',
    tree_size: int = 0,
    root_sha256: str | None = None,
    valid_from: datetime = _T0 - timedelta(days=1),
    valid_until: datetime | None = None,
) -> SelectionRegistryTrustPolicy:
    return SelectionRegistryTrustPolicy(
        trust_policy_id='registry-trust-v1',
        registry_id=_policy().registry_id,
        authority_id=_policy().authority_id,
        pinned_checkpoint=RegistryPinnedCheckpoint(
            tree_size=tree_size,
            root_sha256=root_sha256 or hashlib.sha256(b'').hexdigest(),
            signed_checkpoint_base64=base64.b64encode(signed_checkpoint_bytes).decode('ascii'),
            witness_proof_base64=base64.b64encode(witness_proof_bytes).decode('ascii'),
        ),
        signing_keys=(
            RegistryTrustedSigningKey(
                key_id=key_id,
                public_key_base64=ed25519_public_key_base64(key),
                valid_from=valid_from,
                valid_until=valid_until,
            ),
        ),
        checkpoint_witness_policy=witness_policy,
        checkpoint_witness_trust_policy=witness_trust,
    )


def _registry(
    tmp_path: Path,
    *,
    key: Ed25519PrivateKey | None = None,
    clock=_T0,
    future_monitor_keys: tuple[WitnessRegistrySigningKey, ...] = (),
    runtime_trust_digests: tuple[str, str, str] | None = None,
) -> tuple[SQLitePlanSelectionRegistry, Ed25519PrivateKey, bytes, bytes]:
    signing_key = key or Ed25519PrivateKey.generate()
    runtime_digests: tuple[str | None, str | None, str | None] = runtime_trust_digests or (
        None,
        None,
        None,
    )
    policy = _policy().model_copy(
        update={
            'clock_health_policy_sha256': runtime_digests[0],
            'clock_health_process_sha256': runtime_digests[1],
            'external_signer_process_sha256': runtime_digests[2],
        }
    )
    policy_bytes = canonical_json_bytes(policy)
    monitor = WitnessRegistryMonitor(
        registry_id=policy.registry_id,
        authority_id=policy.authority_id,
        signing_keys=(
            WitnessRegistrySigningKey(
                key_id='registry-key-2026',
                public_key_base64=ed25519_public_key_base64(signing_key),
                valid_from=_T0 - timedelta(days=1),
            ),
            *future_monitor_keys,
        ),
    )
    witness = WitnessServiceStore.initialize(
        tmp_path / 'checkpoint-witness',
        authority_id='independent-selection-checkpoint-witness',
        witness_id='selection-witness-key-2026',
        policy_id='selection-checkpoint-witness-v1',
        trust_policy_id='selection-witness-trust-v1',
        endpoint_uri='https://selection-witness.invalid/v1/witness',
        registry_monitors=(monitor,),
        clock_health_policy_sha256=runtime_digests[0],
        clock_health_process_sha256=runtime_digests[1],
        external_signer_process_sha256=runtime_digests[2],
    )

    def transport(request):
        issuance = witness.issue(request.body)
        return WitnessServiceTransportResponse(
            status_code=201 if issuance.created else 200,
            content_type='application/vnd.vaxreplay.witness-proof+json',
            body=issuance.proof_bytes,
            final_uri=request.endpoint_uri,
            content_length=len(issuance.proof_bytes),
        )

    provider = Ed25519WitnessServiceProvider(
        witness.policy_bytes,
        authorization_bearer_token=b'test-selection-witness-token-at-least-32-bytes',
        transport=transport,
    )
    genesis = build_signed_registry_genesis_checkpoint(
        registry_policy=policy,
        signing_key=signing_key,
        signing_key_id='registry-key-2026',
        issued_at=_T0 - timedelta(seconds=2),
    )
    genesis_bytes = canonical_json_bytes(genesis)
    _claim, genesis_proof = provider(
        checkpoint_witness_request(
            genesis_bytes,
            witness.policy,
            consistency_from_tree_size=0,
            consistency_from_root_sha256=hashlib.sha256(b'').hexdigest(),
            consistency_proof_sha256=(),
        )
    )
    bootstrap_trust = _trust(
        signing_key,
        witness_policy=witness.policy,
        witness_trust=witness.trust_policy,
        signed_checkpoint_bytes=genesis_bytes,
        witness_proof_bytes=genesis_proof,
    )
    bootstrap_trust_bytes = canonical_json_bytes(bootstrap_trust)
    registry = SQLitePlanSelectionRegistry.initialize(
        tmp_path / 'selection-registry.sqlite',
        signing_key=signing_key,
        signing_key_id='registry-key-2026',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=bootstrap_trust_bytes,
        public_base_url='https://registry.example',
        checkpoint_witness_provider=provider,
        clock=lambda: clock,
    )
    bootstrap_request = PlanSelectionRequest(
        commitment_sha256='1' * 64,
        commitment_bytes=1,
        campaign_id='registry-bootstrap-campaign',
        selection_key='registry-bootstrap-selection',
        registry_id=policy.registry_id,
        authority_id=policy.authority_id,
        policy_id=policy.policy_id,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
    )
    registry.assign(bootstrap_request)
    bootstrap_head = registry.tree_head()
    assert bootstrap_head is not None
    pinned_envelope, pinned_witness_proof = registry.signed_checkpoint_and_witness(1)
    trust = _trust(
        signing_key,
        witness_policy=witness.policy,
        witness_trust=witness.trust_policy,
        signed_checkpoint_bytes=pinned_envelope,
        witness_proof_bytes=pinned_witness_proof,
        tree_size=1,
        root_sha256=bootstrap_head.root_sha256,
    )
    trust_bytes = canonical_json_bytes(trust)
    production_registry = SQLitePlanSelectionRegistry(
        registry.database_path,
        signing_key=signing_key,
        signing_key_id='registry-key-2026',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
        public_base_url='https://registry.example',
        checkpoint_witness_provider=provider,
        clock=lambda: clock,
    )
    return production_registry, signing_key, policy_bytes, trust_bytes


def _binding(policy_bytes: bytes, trust_bytes: bytes, *, campaign_id: str = 'campaign-2027'):
    return build_plan_selection_policy_binding(
        campaign_id=campaign_id,
        selection_key='antigen-plan',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
    )


def test_registry_trust_composite_schema_version_is_explicit(tmp_path: Path) -> None:
    _registry_store, _key, _policy_bytes, trust_bytes = _registry(tmp_path)
    trust = SelectionRegistryTrustPolicy.model_validate_json(trust_bytes)

    assert trust.schema_version == 'vaxreplay.plan-selection-registry-trust.v0.3'
    assert canonical_json_bytes(trust) == trust_bytes
    assert SelectionRegistryTrustPolicy.model_validate_json(trust_bytes) == trust

    legacy = trust.model_dump(mode='json')
    legacy['schema_version'] = 'vaxreplay.plan-selection-registry-trust.v0.2'
    with pytest.raises(ValueError, match='schema_version'):
        SelectionRegistryTrustPolicy.model_validate(legacy)


def _commitment(policy_bytes: bytes, trust_bytes: bytes, *, campaign_id: str = 'campaign-2027'):
    return PlanSelectionCommitment(
        policy=_binding(policy_bytes, trust_bytes, campaign_id=campaign_id),
        store_id='a' * 32,
        checkpoint_sha256='b' * 64,
        checkpoint_created_at=_T0 - timedelta(seconds=1),
        scope_policy_sha256='c' * 64,
        pre_capture_plan_sha256='d' * 64,
        earliest_scheduled_slot=_T0 + timedelta(hours=1),
    )


def _request(
    commitment: PlanSelectionCommitment,
    policy_bytes: bytes,
) -> PlanSelectionRequest:
    payload = canonical_json_bytes(commitment)
    return PlanSelectionRequest(
        commitment_sha256=hashlib.sha256(payload).hexdigest(),
        commitment_bytes=len(payload),
        campaign_id=commitment.policy.campaign_id,
        selection_key=commitment.policy.selection_key,
        registry_id=commitment.policy.registry_id,
        authority_id=commitment.policy.authority_id,
        policy_id=commitment.policy.policy_id,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
    )


def test_end_to_end_bridge_uses_durable_offline_verifiable_proof(tmp_path: Path) -> None:
    registry, key, policy_bytes, trust_bytes = _registry(tmp_path)
    commitment = _commitment(policy_bytes, trust_bytes)
    materials = production_plan_selection_materials(
        binding=commitment.policy,
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
    )

    built = broker_plan_selection(
        tmp_path / 'selection-sidecar',
        commitment=commitment,
        materials=materials,
        provider=registry.provider,
        verified_at=_T0 + timedelta(minutes=5),
    )

    assert built.manifest.receipt.facts.atomic_first_write_wins_enforced is True
    assert built.manifest.receipt.facts.consistent_from_pinned_trust_checkpoint is True
    assert built.manifest.receipt.facts.signed_checkpoint_size == 2
    reopened = SQLitePlanSelectionRegistry(
        tmp_path / 'selection-registry.sqlite',
        signing_key=key,
        signing_key_id='registry-key-2026',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
        public_base_url='https://registry.example',
        clock=lambda: _T0 + timedelta(minutes=10),
    )
    assert reopened.proof_for('campaign-2027', 'antigen-plan') == built.proof_bytes
    envelope = reopened.signed_tree_head()
    assert envelope is not None
    assert (
        verify_signed_registry_checkpoint(
            canonical_json_bytes(envelope),
            trust_bytes,
        ).tree_size
        == 2
    )
    assert selection_registry_verifier_implementation_bytes()


def test_atomic_first_write_wins_under_cross_connection_race(tmp_path: Path) -> None:
    registry, key, policy_bytes, trust_bytes = _registry(tmp_path)
    second = SQLitePlanSelectionRegistry(
        registry.database_path,
        signing_key=key,
        signing_key_id='registry-key-2026',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
        public_base_url='https://registry.example',
        checkpoint_witness_provider=registry.checkpoint_witness_provider,
        clock=lambda: _T0,
    )
    first_commitment = _commitment(policy_bytes, trust_bytes)
    second_commitment = first_commitment.model_copy(update={'pre_capture_plan_sha256': 'e' * 64})
    gate = threading.Barrier(2)

    def assign(target: SQLitePlanSelectionRegistry, commitment: PlanSelectionCommitment):
        gate.wait()
        try:
            target.assign(_request(commitment, policy_bytes))
            return commitment.pre_capture_plan_sha256
        except RegistryConflictError:
            return 'conflict'

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            future.result()
            for future in (
                executor.submit(assign, registry, first_commitment),
                executor.submit(assign, second, second_commitment),
            )
        )
    assert results.count('conflict') == 1
    assert sum(value in {'d' * 64, 'e' * 64} for value in results) == 1
    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM registry_entries').fetchone()[0] == 2
        assert connection.execute('SELECT COUNT(*) FROM registry_checkpoints').fetchone()[0] == 3


def test_identical_retry_is_idempotent_but_different_request_conflicts(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    commitment = _commitment(policy_bytes, trust_bytes)
    request = _request(commitment, policy_bytes)
    first = registry.assign(request)
    assert registry.assign(request) == first

    conflicting = request.model_copy(update={'commitment_sha256': 'f' * 64})
    with pytest.raises(RegistryConflictError, match='immutably assigned'):
        registry.assign(conflicting)

    wrong_policy = request.model_copy(update={'policy_sha256': '0' * 64})
    with pytest.raises(SelectionRegistryError, match='service policy'):
        registry.assign(wrong_policy)


def test_witness_outage_preserves_reservation_checkpoint_and_retryability(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    commitment = _commitment(policy_bytes, trust_bytes, campaign_id='witness-outage-campaign')
    request = _request(commitment, policy_bytes)
    working_provider = registry.checkpoint_witness_provider

    def unavailable(_request):
        raise OSError('simulated independent witness outage')

    registry.checkpoint_witness_provider = unavailable
    with pytest.raises(RegistryWitnessUnavailableError, match='after durable selection'):
        registry.assign(request)
    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM registry_assignments').fetchone()[0] == 2
        assert connection.execute('SELECT COUNT(*) FROM registry_entries').fetchone()[0] == 2
        assert connection.execute('SELECT COUNT(*) FROM registry_checkpoints').fetchone()[0] == 3
        assert connection.execute('SELECT COUNT(*) FROM registry_checkpoint_witnesses').fetchone()[0] == 2
    with pytest.raises(RegistryWitnessUnavailableError, match='not been independently witnessed'):
        registry.proof_for('witness-outage-campaign', 'antigen-plan')
    conflicting = commitment.model_copy(update={'pre_capture_plan_sha256': 'e' * 64})
    with pytest.raises(RegistryConflictError, match='immutably assigned'):
        registry.assign(_request(conflicting, policy_bytes))

    registry.checkpoint_witness_provider = working_provider
    recovered = RegistrySelectionProof.model_validate_json(registry.assign(request))
    assert recovered.checkpoint.tree_size == 2
    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM registry_checkpoint_witnesses').fetchone()[0] == 3


def test_offline_verifier_uses_authenticated_witness_time_and_rejects_tampering(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    commitment = _commitment(policy_bytes, trust_bytes)
    proof_bytes = registry.assign(_request(commitment, policy_bytes))
    proof = RegistrySelectionProof.model_validate_json(proof_bytes)
    witness_proof_bytes = base64.b64decode(proof.checkpoint_witness_proof_base64, validate=True)
    witness_proof = WitnessServiceProof.model_validate_json(witness_proof_bytes)
    materials = production_plan_selection_materials(
        binding=commitment.policy,
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
    )
    facts = materials.verifier(
        canonical_json_bytes(commitment),
        proof_bytes,
        commitment.policy,
        policy_bytes,
        trust_bytes,
    )
    assert facts.selected_at_upper_bound == witness_proof.statement.entry.witnessed_at
    assert facts.selected_at_upper_bound >= proof.entry.selected_at_upper_bound

    signature = bytearray(base64.b64decode(witness_proof.receipt_signature_base64, validate=True))
    signature[0] ^= 1
    tampered_witness = witness_proof.model_copy(
        update={'receipt_signature_base64': base64.b64encode(signature).decode('ascii')}
    )
    tampered_proof = proof.model_copy(
        update={
            'checkpoint_witness_proof_base64': base64.b64encode(canonical_json_bytes(tampered_witness)).decode('ascii')
        }
    )
    with pytest.raises(PlanSelectionIntegrityError, match='witness'):
        materials.verifier(
            canonical_json_bytes(commitment),
            canonical_json_bytes(tampered_proof),
            commitment.policy,
            policy_bytes,
            trust_bytes,
        )


def test_production_binding_rejects_genesis_only_trust_anchor(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    production_trust = SelectionRegistryTrustPolicy.model_validate_json(trust_bytes)
    genesis_envelope, genesis_witness = registry.signed_checkpoint_and_witness(0)
    genesis_trust = production_trust.model_copy(
        update={
            'pinned_checkpoint': RegistryPinnedCheckpoint(
                tree_size=0,
                root_sha256=hashlib.sha256(b'').hexdigest(),
                signed_checkpoint_base64=base64.b64encode(genesis_envelope).decode('ascii'),
                witness_proof_base64=base64.b64encode(genesis_witness).decode('ascii'),
            )
        }
    )
    with pytest.raises(SelectionRegistryError, match='nonempty checkpoint'):
        build_plan_selection_policy_binding(
            campaign_id='campaign-after-bootstrap',
            selection_key='antigen-plan',
            registry_policy_bytes=policy_bytes,
            trust_policy_bytes=canonical_json_bytes(genesis_trust),
        )


def test_signing_failure_preserves_fww_reservation_and_rolls_back_log_append(tmp_path: Path) -> None:
    registry, signing_key, policy_bytes, trust_bytes = _registry(tmp_path)

    class FailingSigner:
        def sign(self, _payload: bytes) -> bytes:
            raise RuntimeError('simulated signing-device failure')

    registry.signing_key = FailingSigner()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match='signing-device failure'):
        registry.assign(_request(_commitment(policy_bytes, trust_bytes), policy_bytes))
    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM registry_assignments').fetchone()[0] == 2
        assert connection.execute('SELECT COUNT(*) FROM registry_entries').fetchone()[0] == 1
        assert connection.execute('SELECT COUNT(*) FROM registry_checkpoints').fetchone()[0] == 2

    conflicting = _commitment(policy_bytes, trust_bytes).model_copy(update={'pre_capture_plan_sha256': 'e' * 64})
    with pytest.raises(RegistryConflictError, match='immutably assigned'):
        registry.assign(_request(conflicting, policy_bytes))

    registry.signing_key = signing_key
    assert (
        RegistrySelectionProof.model_validate_json(
            registry.assign(_request(_commitment(policy_bytes, trust_bytes), policy_bytes))
        ).checkpoint.tree_size
        == 2
    )


def test_clock_regression_fails_closed_without_appending(tmp_path: Path) -> None:
    registry, key, policy_bytes, trust_bytes = _registry(tmp_path)
    first = _commitment(policy_bytes, trust_bytes, campaign_id='first-campaign')
    registry.assign(_request(first, policy_bytes))
    regressed = SQLitePlanSelectionRegistry(
        registry.database_path,
        signing_key=key,
        signing_key_id='registry-key-2026',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
        public_base_url='https://registry.example',
        checkpoint_witness_provider=registry.checkpoint_witness_provider,
        clock=lambda: _T0 - timedelta(seconds=1),
    )
    second = _commitment(policy_bytes, trust_bytes, campaign_id='second-campaign')
    with pytest.raises(SelectionRegistryError, match='clock regressed'):
        regressed.assign(_request(second, policy_bytes))
    head = registry.tree_head()
    assert head is not None
    assert head.tree_size == 2


def test_signed_upper_bound_is_sampled_after_durable_fww_commit(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    observed_assignment_counts: list[int] = []

    def post_commit_clock() -> datetime:
        with sqlite3.connect(registry.database_path) as connection:
            observed_assignment_counts.append(
                int(connection.execute('SELECT COUNT(*) FROM registry_assignments').fetchone()[0])
            )
        return _T0

    registry.clock = post_commit_clock
    proof = RegistrySelectionProof.model_validate_json(
        registry.assign(_request(_commitment(policy_bytes, trust_bytes), policy_bytes))
    )
    assert observed_assignment_counts == [2]
    assert proof.entry.selected_at_upper_bound == _T0


def test_append_only_database_triggers_reject_mutation(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    registry.assign(_request(_commitment(policy_bytes, trust_bytes), policy_bytes))

    connection = sqlite3.connect(registry.database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            connection.execute('DELETE FROM registry_entries')
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            connection.execute('UPDATE registry_checkpoints SET root_sha256 = ?', ('0' * 64,))
    finally:
        connection.close()


def test_merkle_proofs_cover_non_power_of_two_tree_shapes() -> None:
    for tree_size in range(1, 40):
        leaves = tuple(hashlib.sha256(f'leaf-{index}'.encode()).digest() for index in range(tree_size))
        root = merkle_root(leaves)
        for index, leaf in enumerate(leaves):
            proof = inclusion_proof(leaves, index)
            assert verify_inclusion_proof(
                leaf,
                index=index,
                tree_size=tree_size,
                proof=proof,
                expected_root=root,
            )
        for old_size in range(tree_size + 1):
            proof = consistency_proof(leaves, old_size)
            assert verify_consistency_proof(
                old_size=old_size,
                new_size=tree_size,
                old_root=merkle_root(leaves[:old_size]),
                new_root=root,
                proof=proof,
            )


def test_verifier_rejects_signature_leaf_and_policy_tampering(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    commitment = _commitment(policy_bytes, trust_bytes)
    proof_bytes = registry.assign(_request(commitment, policy_bytes))
    materials = production_plan_selection_materials(
        binding=commitment.policy,
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
    )
    facts = materials.verifier(
        canonical_json_bytes(commitment),
        proof_bytes,
        commitment.policy,
        policy_bytes,
        trust_bytes,
    )
    assert facts.registry_sequence == 1

    proof = RegistrySelectionProof.model_validate_json(proof_bytes)
    signature = bytearray(base64.b64decode(proof.checkpoint_signature_base64))
    signature[0] ^= 1
    bad_signature = proof.model_copy(update={'checkpoint_signature_base64': base64.b64encode(signature).decode()})
    with pytest.raises(PlanSelectionIntegrityError, match='signature'):
        materials.verifier(
            canonical_json_bytes(commitment),
            canonical_json_bytes(bad_signature),
            commitment.policy,
            policy_bytes,
            trust_bytes,
        )

    bad_entry = proof.model_copy(update={'entry': proof.entry.model_copy(update={'commitment_sha256': 'f' * 64})})
    with pytest.raises(PlanSelectionIntegrityError, match='different request'):
        materials.verifier(
            canonical_json_bytes(commitment),
            canonical_json_bytes(bad_entry),
            commitment.policy,
            policy_bytes,
            trust_bytes,
        )

    wrong_policy = _policy().model_copy(update={'max_request_bytes': 8192})
    with pytest.raises(PlanSelectionIntegrityError, match='materials'):
        materials.verifier(
            canonical_json_bytes(commitment),
            proof_bytes,
            commitment.policy,
            canonical_json_bytes(wrong_policy),
            trust_bytes,
        )


def test_nonempty_consistency_proof_extends_pinned_checkpoint(tmp_path: Path) -> None:
    registry, key, policy_bytes, initial_trust_bytes = _registry(tmp_path)
    initial_commitment = _commitment(policy_bytes, initial_trust_bytes, campaign_id='bootstrap-campaign')
    registry.assign(_request(initial_commitment, policy_bytes))
    pinned_head = registry.tree_head()
    assert pinned_head is not None
    existing_trust = SelectionRegistryTrustPolicy.model_validate_json(initial_trust_bytes)
    pinned_envelope, pinned_witness_proof = registry.signed_checkpoint_and_witness(pinned_head.tree_size)
    pinned_trust = _trust(
        key,
        witness_policy=existing_trust.checkpoint_witness_policy,
        witness_trust=existing_trust.checkpoint_witness_trust_policy,
        signed_checkpoint_bytes=pinned_envelope,
        witness_proof_bytes=pinned_witness_proof,
        tree_size=pinned_head.tree_size,
        root_sha256=pinned_head.root_sha256,
    )
    pinned_trust_bytes = canonical_json_bytes(pinned_trust)
    extended = SQLitePlanSelectionRegistry(
        registry.database_path,
        signing_key=key,
        signing_key_id='registry-key-2026',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=pinned_trust_bytes,
        public_base_url='https://registry.example',
        checkpoint_witness_provider=registry.checkpoint_witness_provider,
        clock=lambda: _T0,
    )
    commitment = _commitment(policy_bytes, pinned_trust_bytes, campaign_id='post-pin-campaign')
    proof_bytes = extended.assign(_request(commitment, policy_bytes))
    proof = RegistrySelectionProof.model_validate_json(proof_bytes)
    assert proof.consistency_from_tree_size == 2
    assert proof.consistency_proof_sha256

    materials = production_plan_selection_materials(
        binding=commitment.policy,
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=pinned_trust_bytes,
    )
    facts = materials.verifier(
        canonical_json_bytes(commitment),
        proof_bytes,
        commitment.policy,
        policy_bytes,
        pinned_trust_bytes,
    )
    assert facts.signed_checkpoint_size == 3
    tampered = proof.model_copy(update={'consistency_proof_sha256': ('0' * 64,)})
    with pytest.raises(PlanSelectionIntegrityError, match='consistency'):
        materials.verifier(
            canonical_json_bytes(commitment),
            canonical_json_bytes(tampered),
            commitment.policy,
            policy_bytes,
            pinned_trust_bytes,
        )


def test_signing_key_validity_and_rotation_public_key_registration(tmp_path: Path) -> None:
    next_key = Ed25519PrivateKey.generate()
    next_public = ed25519_public_key_base64(next_key)
    next_valid_from = _T0 + timedelta(days=1)
    registry, _key, policy_bytes, trust_bytes = _registry(
        tmp_path,
        future_monitor_keys=(
            WitnessRegistrySigningKey(
                key_id='registry-key-2027',
                public_key_base64=next_public,
                valid_from=next_valid_from,
            ),
        ),
    )
    current_trust = SelectionRegistryTrustPolicy.model_validate_json(trust_bytes)
    rotated_trust = current_trust.model_copy(
        update={
            'signing_keys': (
                *current_trust.signing_keys,
                RegistryTrustedSigningKey(
                    key_id='registry-key-2027',
                    public_key_base64=next_public,
                    valid_from=next_valid_from,
                ),
            ),
        }
    )
    rotated_bytes = canonical_json_bytes(rotated_trust)
    rotated_registry = SQLitePlanSelectionRegistry(
        registry.database_path,
        signing_key=registry.signing_key,
        signing_key_id='registry-key-2026',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=rotated_bytes,
        public_base_url='https://registry.example',
        clock=lambda: _T0,
    )
    rotated_registry.register_signing_key(
        key_id='registry-key-2027',
        public_key_base64=next_public,
        registered_at=_T0,
    )
    next_registry = SQLitePlanSelectionRegistry(
        registry.database_path,
        signing_key=next_key,
        signing_key_id='registry-key-2027',
        registry_policy_bytes=policy_bytes,
        trust_policy_bytes=rotated_bytes,
        public_base_url='https://registry.example',
        clock=lambda: _T0,
    )
    with pytest.raises(SelectionRegistryError, match='not valid'):
        next_registry.assign(_request(_commitment(policy_bytes, rotated_bytes), policy_bytes))


def test_private_key_file_is_exclusive_owner_only_and_not_in_registry(tmp_path: Path) -> None:
    key_path = generate_ed25519_private_key(tmp_path / 'registry.key')
    assert stat_mode(key_path) == 0o600
    key = load_ed25519_private_key(key_path)
    raw_private = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path / 'state', key=key)
    registry.assign(_request(_commitment(policy_bytes, trust_bytes), policy_bytes))
    assert raw_private not in registry.database_path.read_bytes()
    assert raw_private not in registry.proof_for('campaign-2027', 'antigen-plan')

    os.chmod(key_path, 0o644)
    with pytest.raises(SelectionRegistryError, match='group or other'):
        load_ed25519_private_key(key_path)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_http_service_authenticates_writes_and_keeps_proofs_public(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    commitment = _commitment(policy_bytes, trust_bytes)
    request_body = canonical_json_bytes(_request(commitment, policy_bytes))
    server = SelectionRegistryHTTPServer(
        ('127.0.0.1', 0),
        registry,
        write_token_sha256=hashlib.sha256(_TOKEN.encode()).hexdigest(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request(
            'POST',
            '/v1/selections',
            body=request_body,
            headers={'Content-Type': 'application/json'},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 401

        working_witness = registry.checkpoint_witness_provider

        def unavailable(_request):
            raise OSError('witness temporarily unavailable')

        registry.checkpoint_witness_provider = unavailable
        connection.request(
            'POST',
            '/v1/selections',
            body=request_body,
            headers={'Authorization': f'Bearer {_TOKEN}', 'Content-Type': 'application/json'},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 503
        registry.checkpoint_witness_provider = working_witness

        connection.request(
            'POST',
            '/v1/selections',
            body=request_body,
            headers={'Authorization': f'Bearer {_TOKEN}', 'Content-Type': 'application/json'},
        )
        response = connection.getresponse()
        body = response.read()
        assert response.status == 200
        assert _TOKEN.encode() not in body

        connection.request('GET', '/v1/proofs/campaign-2027/antigen-plan')
        response = connection.getresponse()
        proof = response.read()
        assert response.status == 200
        assert RegistrySelectionProof.model_validate_json(proof).entry.campaign_id == 'campaign-2027'

        connection.request('GET', '/v1/entries/1')
        response = connection.getresponse()
        entry_bytes = response.read()
        assert response.status == 200
        assert response.getheader('Cache-Control') == 'public, immutable, max-age=31536000'
        entry = RegistryLogEntry.model_validate_json(entry_bytes)
        assert entry.registry_sequence == 1
        assert entry.campaign_id == 'campaign-2027'
        assert entry_bytes == registry.registry_entry_bytes(1)

        connection.request('GET', '/v1/entries/01')
        response = connection.getresponse()
        response.read()
        assert response.status == 404

        connection.request('GET', '/v1/checkpoints/latest')
        response = connection.getresponse()
        checkpoint_envelope = response.read()
        assert response.status == 200
        assert registry.signed_tree_head() is not None
        assert checkpoint_envelope == canonical_json_bytes(registry.signed_tree_head())

        connection.request(
            'POST',
            '/v1/selections',
            body=b'x' * (registry.policy.max_request_bytes + 1),
            headers={'Authorization': f'Bearer {_TOKEN}', 'Content-Type': 'application/json'},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 413
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_https_provider_round_trip_uses_tls_and_bearer_auth(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    tls_key = generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    tls_now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(tls_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(tls_now - timedelta(days=1))
        .not_valid_after(tls_now + timedelta(days=30))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName('localhost'), x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),
            critical=False,
        )
        .sign(tls_key, hashes.SHA256())
    )
    cert_path = tmp_path / 'tls-cert.pem'
    tls_key_path = tmp_path / 'tls-key.pem'
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    tls_key_path.write_bytes(
        tls_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = SelectionRegistryHTTPServer(
        ('127.0.0.1', 0),
        registry,
        write_token_sha256=hashlib.sha256(_TOKEN.encode()).hexdigest(),
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_path, keyfile=tls_key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _host, port = server.server_address
    try:
        provider = HttpsPlanSelectionRegistryProvider(
            f'https://localhost:{port}',
            bearer_token=_TOKEN,
            ca_file=cert_path,
        )
        commitment = _commitment(policy_bytes, trust_bytes)
        claim, proof = provider(_request(commitment, policy_bytes))
        assert claim.verification_uri == 'https://registry.example/v1/proofs/campaign-2027/antigen-plan'
        assert (
            RegistrySelectionProof.model_validate_json(proof).entry.commitment_sha256
            == _request(commitment, policy_bytes).commitment_sha256
        )

        unauthorized = HttpsPlanSelectionRegistryProvider(
            f'https://localhost:{port}',
            bearer_token='wrong-' + ('z' * 64),
            ca_file=cert_path,
        )
        with pytest.raises(SelectionRegistryError, match='HTTP 401'):
            unauthorized(_request(commitment, policy_bytes))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
