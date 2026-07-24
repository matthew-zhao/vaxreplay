from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations import plan_selection as plan_selection_module
from vaxreplay.operations.plan_selection import (
    AuthenticatedPlanSelectionFacts,
    PlanSelectionClaim,
    PlanSelectionCommitment,
    PlanSelectionIntegrityError,
    PlanSelectionMaterialSpec,
    PlanSelectionPolicyBinding,
    PlanSelectionRequest,
    broker_plan_selection,
    load_plan_selection,
)

_T0 = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
_FIRST_SLOT = _T0 + timedelta(hours=1)
_SELECTED_AT = _T0 + timedelta(seconds=2)
_VERIFIED_AT = _T0 + timedelta(seconds=3)
_POLICY_BYTES = b'fixture atomic first-write-wins campaign policy'
_TRUST_BYTES = b'fixture plan registry trust roots'
_VERIFIER_BYTES = b'fixture offline plan registry verifier implementation'


class _Registry:
    def __init__(self, *, selected_at: datetime = _SELECTED_AT) -> None:
        self.selected_at = selected_at
        self.assignments: dict[tuple[str, str], str] = {}
        self.requests: list[PlanSelectionRequest] = []
        self.verifier_calls = 0

    def provider(self, request: PlanSelectionRequest):
        self.requests.append(request)
        key = (request.campaign_id, request.selection_key)
        existing = self.assignments.get(key)
        if existing is not None and existing != request.commitment_sha256:
            raise ValueError('selection key is already assigned to another commitment')
        self.assignments[key] = request.commitment_sha256
        proof = self._proof_bytes(
            campaign_id=request.campaign_id,
            selection_key=request.selection_key,
            commitment_sha256=request.commitment_sha256,
        )
        return PlanSelectionClaim(verification_uri='https://registry.invalid/selections/fixture'), proof

    def verifier(
        self,
        commitment_bytes: bytes,
        proof_bytes: bytes,
        policy: PlanSelectionPolicyBinding,
        policy_bytes: bytes,
        trust_policy_bytes: bytes,
    ) -> AuthenticatedPlanSelectionFacts:
        self.verifier_calls += 1
        assert policy_bytes == _POLICY_BYTES
        assert trust_policy_bytes == _TRUST_BYTES
        commitment = PlanSelectionCommitment.model_validate_json(commitment_bytes)
        digest = hashlib.sha256(commitment_bytes).hexdigest()
        key = (policy.campaign_id, policy.selection_key)
        if self.assignments.get(key) != digest:
            raise ValueError('registry does not contain this exact assignment')
        expected_proof = self._proof_bytes(
            campaign_id=policy.campaign_id,
            selection_key=policy.selection_key,
            commitment_sha256=digest,
        )
        if proof_bytes != expected_proof:
            raise ValueError('registry proof bytes differ')
        return AuthenticatedPlanSelectionFacts(
            receipt_id='fixture-plan-selection-receipt',
            registry_id=policy.registry_id,
            authority_id=policy.authority_id,
            campaign_id=policy.campaign_id,
            selection_key=policy.selection_key,
            commitment_sha256=digest,
            store_id=commitment.store_id,
            checkpoint_sha256=commitment.checkpoint_sha256,
            scope_policy_sha256=commitment.scope_policy_sha256,
            pre_capture_plan_sha256=commitment.pre_capture_plan_sha256,
            selected_at_upper_bound=self.selected_at,
            registry_entry_id='fixture-plan-entry-000001',
            registry_sequence=0,
            signed_checkpoint_sha256=hashlib.sha256(b'fixture signed registry checkpoint').hexdigest(),
            signed_checkpoint_size=1,
        )

    def _proof_bytes(self, *, campaign_id: str, selection_key: str, commitment_sha256: str) -> bytes:
        return canonical_json_bytes(
            {
                'campaign_id': campaign_id,
                'selection_key': selection_key,
                'commitment_sha256': commitment_sha256,
                'selected_at_upper_bound': self.selected_at.isoformat().replace('+00:00', 'Z'),
                'first_write_wins': True,
                'final': True,
                'registry_entry_id': 'fixture-plan-entry-000001',
                'registry_sequence': 0,
                'signed_checkpoint_sha256': hashlib.sha256(b'fixture signed registry checkpoint').hexdigest(),
                'signed_checkpoint_size': 1,
                'valid_inclusion_proof': True,
                'consistent_from_pinned_trust_checkpoint': True,
                'selection_key_history_count': 1,
            }
        )


def _policy() -> PlanSelectionPolicyBinding:
    return PlanSelectionPolicyBinding(
        campaign_id='fixture-pandemic-campaign-2027',
        selection_key='antigen-prioritization-plan',
        registry_id='fixture-independent-plan-registry',
        authority_id='fixture-benchmark-authority',
        policy_id='fixture-first-write-wins-policy-v1',
        policy_sha256=hashlib.sha256(_POLICY_BYTES).hexdigest(),
        trust_policy_id='fixture-plan-registry-trust-v1',
        trust_policy_sha256=hashlib.sha256(_TRUST_BYTES).hexdigest(),
        verifier_id='fixture-plan-registry-verifier-v1',
        verifier_implementation_sha256=hashlib.sha256(_VERIFIER_BYTES).hexdigest(),
    )


def _commitment(*, plan_sha256: str = 'd' * 64) -> PlanSelectionCommitment:
    return PlanSelectionCommitment(
        policy=_policy(),
        store_id='a' * 32,
        checkpoint_sha256='b' * 64,
        checkpoint_created_at=_T0 + timedelta(seconds=1),
        scope_policy_sha256='c' * 64,
        pre_capture_plan_sha256=plan_sha256,
        earliest_scheduled_slot=_FIRST_SLOT,
    )


def _materials(registry: _Registry) -> PlanSelectionMaterialSpec:
    return PlanSelectionMaterialSpec(
        policy=_policy(),
        policy_bytes=_POLICY_BYTES,
        trust_policy_bytes=_TRUST_BYTES,
        verifier_implementation_bytes=_VERIFIER_BYTES,
        verifier=registry.verifier,
    )


def _build(root: Path, registry: _Registry, *, commitment: PlanSelectionCommitment | None = None):
    return broker_plan_selection(
        root,
        commitment=commitment or _commitment(),
        materials=_materials(registry),
        provider=registry.provider,
        verified_at=_VERIFIED_AT,
    )


def test_round_trip_is_exact_and_reruns_the_pinned_verifier(tmp_path: Path) -> None:
    registry = _Registry()
    built = _build(tmp_path / 'selection', registry)

    assert {path.name for path in built.root.iterdir()} == {
        'commitment.json',
        'registry-proof.bin',
        'selection.json',
    }
    assert registry.verifier_calls == 2
    request = registry.requests[0]
    request_fields = request.model_dump()
    assert request.commitment_sha256 == hashlib.sha256(built.commitment_bytes).hexdigest()
    assert 'pre_capture_plan_sha256' not in request_fields
    assert 'scope_policy_sha256' not in request_fields

    loaded = load_plan_selection(
        built.root,
        expected_commitment=_commitment(),
        expected_manifest_sha256=built.manifest_sha256,
        materials=_materials(registry),
        verified_at=_VERIFIED_AT + timedelta(seconds=1),
    )
    assert loaded == built
    assert registry.verifier_calls == 3


@pytest.mark.parametrize('selected_at', (_FIRST_SLOT, _FIRST_SLOT + timedelta(microseconds=1)))
def test_selection_must_be_strictly_before_the_first_slot(tmp_path: Path, selected_at: datetime) -> None:
    registry = _Registry(selected_at=selected_at)
    with pytest.raises(PlanSelectionIntegrityError, match='strictly before'):
        _build(tmp_path / 'selection', registry)
    assert not (tmp_path / 'selection').exists()


@pytest.mark.parametrize(
    ('selected_at', 'message'),
    (
        (_T0, 'predates'),
        (_VERIFIED_AT + timedelta(microseconds=1), 'postdates'),
    ),
)
def test_selection_must_follow_the_checkpoint_and_precede_verification(
    tmp_path: Path,
    selected_at: datetime,
    message: str,
) -> None:
    registry = _Registry(selected_at=selected_at)
    with pytest.raises(PlanSelectionIntegrityError, match=message):
        _build(tmp_path / 'selection', registry)


def test_registry_enforces_one_commitment_for_the_stable_selection_key(tmp_path: Path) -> None:
    registry = _Registry()
    _build(tmp_path / 'first', registry, commitment=_commitment(plan_sha256='d' * 64))

    with pytest.raises(PlanSelectionIntegrityError, match='already assigned'):
        _build(tmp_path / 'second', registry, commitment=_commitment(plan_sha256='e' * 64))
    assert not (tmp_path / 'second').exists()


def test_load_rejects_wrong_policy_material_and_expected_commitment(tmp_path: Path) -> None:
    registry = _Registry()
    built = _build(tmp_path / 'selection', registry)
    wrong_materials = PlanSelectionMaterialSpec(
        policy=_policy(),
        policy_bytes=b'different policy bytes',
        trust_policy_bytes=_TRUST_BYTES,
        verifier_implementation_bytes=_VERIFIER_BYTES,
        verifier=registry.verifier,
    )
    with pytest.raises(PlanSelectionIntegrityError, match='pinned digest'):
        load_plan_selection(
            built.root,
            expected_commitment=_commitment(),
            expected_manifest_sha256=built.manifest_sha256,
            materials=wrong_materials,
            verified_at=_VERIFIED_AT,
        )

    for label, wrong_materials in (
        (
            'trust',
            PlanSelectionMaterialSpec(
                policy=_policy(),
                policy_bytes=_POLICY_BYTES,
                trust_policy_bytes=b'different trust checkpoint bytes',
                verifier_implementation_bytes=_VERIFIER_BYTES,
                verifier=registry.verifier,
            ),
        ),
        (
            'verifier',
            PlanSelectionMaterialSpec(
                policy=_policy(),
                policy_bytes=_POLICY_BYTES,
                trust_policy_bytes=_TRUST_BYTES,
                verifier_implementation_bytes=b'different verifier implementation bytes',
                verifier=registry.verifier,
            ),
        ),
    ):
        with pytest.raises(PlanSelectionIntegrityError, match=f'{label}.*pinned digest'):
            load_plan_selection(
                built.root,
                expected_commitment=_commitment(),
                expected_manifest_sha256=built.manifest_sha256,
                materials=wrong_materials,
                verified_at=_VERIFIED_AT,
            )

    with pytest.raises(PlanSelectionIntegrityError, match='different expected commitment'):
        load_plan_selection(
            built.root,
            expected_commitment=_commitment(plan_sha256='e' * 64),
            expected_manifest_sha256=built.manifest_sha256,
            materials=_materials(registry),
            verified_at=_VERIFIED_AT,
        )


def test_load_rejects_tampered_proof_extra_files_and_symlinks(tmp_path: Path) -> None:
    registry = _Registry()
    tampered = _build(tmp_path / 'tampered', registry)
    proof_path = tampered.root / 'registry-proof.bin'
    tampered.root.chmod(0o755)
    proof_path.chmod(0o644)
    proof_path.write_bytes(b'tampered proof')
    with pytest.raises(PlanSelectionIntegrityError, match='proof'):
        load_plan_selection(
            tampered.root,
            expected_commitment=_commitment(),
            expected_manifest_sha256=tampered.manifest_sha256,
            materials=_materials(registry),
            verified_at=_VERIFIED_AT,
        )

    registry_extra = _Registry()
    extra = _build(tmp_path / 'extra', registry_extra)
    extra.root.chmod(0o755)
    (extra.root / 'undeclared.bin').write_bytes(b'extra')
    with pytest.raises(PlanSelectionIntegrityError, match='inventory|file count'):
        load_plan_selection(
            extra.root,
            expected_commitment=_commitment(),
            expected_manifest_sha256=extra.manifest_sha256,
            materials=_materials(registry_extra),
            verified_at=_VERIFIED_AT,
        )

    registry_link = _Registry()
    linked = _build(tmp_path / 'linked', registry_link)
    linked.root.chmod(0o755)
    linked_proof = linked.root / 'registry-proof.bin'
    linked_proof.unlink()
    linked_proof.symlink_to(linked.root / 'commitment.json')
    with pytest.raises(PlanSelectionIntegrityError, match='symbolic link'):
        load_plan_selection(
            linked.root,
            expected_commitment=_commitment(),
            expected_manifest_sha256=linked.manifest_sha256,
            materials=_materials(registry_link),
            verified_at=_VERIFIED_AT,
        )


def test_authenticated_facts_must_assert_first_write_wins_and_finality(tmp_path: Path) -> None:
    registry = _Registry()
    materials = _materials(registry)

    def verifier(
        commitment_bytes: bytes,
        proof_bytes: bytes,
        policy: PlanSelectionPolicyBinding,
        policy_bytes: bytes,
        trust_policy_bytes: bytes,
    ):
        facts = registry.verifier(
            commitment_bytes,
            proof_bytes,
            policy,
            policy_bytes,
            trust_policy_bytes,
        )
        return facts.model_copy(update={'key_previously_unassigned': False})

    invalid = PlanSelectionMaterialSpec(
        policy=materials.policy,
        policy_bytes=materials.policy_bytes,
        trust_policy_bytes=materials.trust_policy_bytes,
        verifier_implementation_bytes=materials.verifier_implementation_bytes,
        verifier=verifier,
    )
    with pytest.raises(PlanSelectionIntegrityError, match='authenticated plan-selection facts'):
        broker_plan_selection(
            tmp_path / 'selection',
            commitment=_commitment(),
            materials=invalid,
            provider=registry.provider,
            verified_at=_VERIFIED_AT,
        )


@pytest.mark.parametrize(
    'uri',
    (
        ' https://registry.invalid/selections/fixture',
        'https://registry.invalid/selections/fixture\n',
        'https://registry.invalid/\x00fixture',
    ),
)
def test_verification_uri_rejects_whitespace_and_controls(uri: str) -> None:
    with pytest.raises(ValueError, match='verification_uri'):
        PlanSelectionClaim(verification_uri=uri)


def test_publication_never_replaces_a_racing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    target = tmp_path / 'selection'
    original_rename = plan_selection_module.rename_directory_noreplace

    def race_at_install(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / 'owner.txt').write_bytes(b'competing publisher')
        original_rename(source, destination)

    monkeypatch.setattr(plan_selection_module, 'rename_directory_noreplace', race_at_install)
    with pytest.raises(PlanSelectionIntegrityError, match='output already exists'):
        _build(target, registry)
    assert (target / 'owner.txt').read_bytes() == b'competing publisher'
    assert not (tmp_path / '.selection.publish.lock').exists()
