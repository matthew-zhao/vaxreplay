from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations import scope_precommit as scope_precommit_module
from vaxreplay.operations.plan_selection import (
    AuthenticatedPlanSelectionFacts,
    PlanSelectionClaim,
    PlanSelectionCommitment,
    PlanSelectionMaterialSpec,
    PlanSelectionPolicyBinding,
    PlanSelectionRequest,
    broker_plan_selection,
)
from vaxreplay.operations.promotion import WitnessMaterialSpec
from vaxreplay.operations.promotion_schema import (
    PreCapturePlan,
    PrecommittedAdapter,
    PrecommittedSourceVerifier,
    PromotionScopePolicy,
    PromotionSourceScope,
)
from vaxreplay.operations.schema import CaptureJobSpec, LedgerCheckpoint, checkpoint_sha256, job_spec_sha256
from vaxreplay.operations.scope_precommit import (
    ScopePrecommitManifest,
    build_scope_precommit,
    derive_plan_selection_commitment,
    load_scope_precommit,
)
from vaxreplay.operations.store import OperationalStore
from vaxreplay.operations.witness import (
    AuthenticatedExternalWitnessFacts,
    ExternalWitnessClaim,
    ExternalWitnessMethod,
    WitnessPolicyBinding,
    broker_witness_checkpoint,
)

_T0 = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
_FIRST_SCHEDULED_SLOT = _T0 + timedelta(hours=1)
_STORE_ID = 'a' * 32
_PROOF = b'fixture transparency-log proof over the scope checkpoint'
_POLICY_BYTES = b'fixture scope-checkpoint witness policy v1'
_TRUST_POLICY_BYTES = b'fixture independent transparency-log trust roots v1'
_VERIFIER_BYTES = b'fixture offline transparency-log verifier v1'
_SELECTION_POLICY_BYTES = b'fixture first-write-wins plan-selection policy v1'
_SELECTION_TRUST_BYTES = b'fixture signed append-only registry trust checkpoint v1'
_SELECTION_VERIFIER_BYTES = b'fixture offline selection inclusion-and-consistency verifier v1'
_SELECTED_AT = _T0 + timedelta(seconds=6)


@dataclass(frozen=True)
class _PrecommitFixture:
    store: OperationalStore
    scope_policy: PromotionScopePolicy
    pre_capture_plan: PreCapturePlan
    checkpoint: LedgerCheckpoint
    witness_root: Path
    witness_materials: WitnessMaterialSpec
    selection_root: Path
    selection_manifest_sha256: str
    selection_materials: PlanSelectionMaterialSpec
    created_at: datetime
    verified_at: datetime


def _witness_policy() -> WitnessPolicyBinding:
    return WitnessPolicyBinding(
        authority_id='fixture-independent-log',
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id='fixture-scope-precommit-policy-v1',
        policy_sha256=hashlib.sha256(_POLICY_BYTES).hexdigest(),
        trust_policy_id='fixture-log-trust-roots-v1',
        trust_policy_sha256=hashlib.sha256(_TRUST_POLICY_BYTES).hexdigest(),
        verifier_id='fixture-offline-log-verifier-v1',
        verifier_implementation_sha256=hashlib.sha256(_VERIFIER_BYTES).hexdigest(),
    )


def _selection_policy() -> PlanSelectionPolicyBinding:
    return PlanSelectionPolicyBinding(
        campaign_id='fixture-pandemic-campaign-2027',
        selection_key='antigen-prioritization-plan',
        registry_id='fixture-independent-plan-registry',
        authority_id='fixture-benchmark-authority',
        policy_id='fixture-first-write-wins-policy-v1',
        policy_sha256=hashlib.sha256(_SELECTION_POLICY_BYTES).hexdigest(),
        trust_policy_id='fixture-registry-trust-checkpoint-v1',
        trust_policy_sha256=hashlib.sha256(_SELECTION_TRUST_BYTES).hexdigest(),
        verifier_id='fixture-registry-proof-verifier-v1',
        verifier_implementation_sha256=hashlib.sha256(_SELECTION_VERIFIER_BYTES).hexdigest(),
    )


class _SelectionRegistry:
    def __init__(self, *, selected_at: datetime) -> None:
        self.selected_at = selected_at
        self.assignments: dict[tuple[str, str], str] = {}

    def provider(self, request: PlanSelectionRequest):
        key = (request.campaign_id, request.selection_key)
        previous = self.assignments.get(key)
        if previous is not None and previous != request.commitment_sha256:
            raise ValueError('selection key already assigned')
        self.assignments[key] = request.commitment_sha256
        proof = canonical_json_bytes(
            {
                'commitment_sha256': request.commitment_sha256,
                'registry_entry_id': 'fixture-plan-entry-000001',
                'registry_sequence': 0,
                'selected_at_upper_bound': self.selected_at.isoformat().replace('+00:00', 'Z'),
                'signed_checkpoint_sha256': hashlib.sha256(b'fixture signed selection checkpoint').hexdigest(),
                'signed_checkpoint_size': 1,
                'valid_inclusion_proof': True,
                'consistent_from_pinned_trust_checkpoint': True,
                'selection_key_history_count': 1,
            }
        )
        return PlanSelectionClaim(
            verification_uri='https://registry.invalid/selections/fixture-plan-entry-000001'
        ), proof

    def verifier(
        self,
        commitment_bytes: bytes,
        proof_bytes: bytes,
        policy: PlanSelectionPolicyBinding,
        policy_bytes: bytes,
        trust_policy_bytes: bytes,
    ) -> AuthenticatedPlanSelectionFacts:
        assert policy_bytes == _SELECTION_POLICY_BYTES
        assert trust_policy_bytes == _SELECTION_TRUST_BYTES
        commitment = PlanSelectionCommitment.model_validate_json(commitment_bytes)
        digest = hashlib.sha256(commitment_bytes).hexdigest()
        assert self.assignments[(policy.campaign_id, policy.selection_key)] == digest
        assert proof_bytes == canonical_json_bytes(
            {
                'commitment_sha256': digest,
                'registry_entry_id': 'fixture-plan-entry-000001',
                'registry_sequence': 0,
                'selected_at_upper_bound': self.selected_at.isoformat().replace('+00:00', 'Z'),
                'signed_checkpoint_sha256': hashlib.sha256(b'fixture signed selection checkpoint').hexdigest(),
                'signed_checkpoint_size': 1,
                'valid_inclusion_proof': True,
                'consistent_from_pinned_trust_checkpoint': True,
                'selection_key_history_count': 1,
            }
        )
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
            signed_checkpoint_sha256=hashlib.sha256(b'fixture signed selection checkpoint').hexdigest(),
            signed_checkpoint_size=1,
        )


def _fixture(
    root: Path,
    *,
    scope_store_id: str = _STORE_ID,
    scope_in_checkpoint: bool = True,
    plan_in_checkpoint: bool = True,
    job_in_checkpoint: bool = True,
    witnessed_at: datetime = _T0 + timedelta(seconds=4),
    selected_at: datetime = _SELECTED_AT,
) -> _PrecommitFixture:
    store = OperationalStore.initialize(
        root / 'operations',
        created_at=_T0,
        store_id=_STORE_ID,
        trusted_lease_clock=None,
    )
    job_spec = CaptureJobSpec(
        job_id='future-daily-release',
        collector_id='fixture-static-https',
        schedule_anchor_at=_FIRST_SCHEDULED_SLOT,
        schedule_interval_seconds=86_400,
        configuration={'source_id': 'publisher:fixture'},
    )
    job_digest = job_spec_sha256(job_spec)
    scope_policy = PromotionScopePolicy(
        policy_id='fixture-promotion-scope-v1',
        store_id=scope_store_id,
        # These bounds apply to the later capture checkpoint, not to this
        # earlier scope-commitment checkpoint.
        checkpoint_created_at_not_before=_FIRST_SCHEDULED_SLOT + timedelta(minutes=30),
        checkpoint_created_at_not_after=_FIRST_SCHEDULED_SLOT + timedelta(hours=2),
        sources=(
            PromotionSourceScope(
                source_id='publisher:fixture',
                job_spec_sha256s=(job_digest,),
                scheduled_from=_FIRST_SCHEDULED_SLOT,
                scheduled_through=_FIRST_SCHEDULED_SLOT,
            ),
        ),
    )
    pre_capture_plan = PreCapturePlan(
        scope_policy=scope_policy,
        selection_policy=_selection_policy(),
        capture_witness_policy=WitnessPolicyBinding(
            authority_id='fixture-capture-tsa',
            method=ExternalWitnessMethod.RFC3161_TIMESTAMP,
            policy_id='fixture-capture-witness-policy-v1',
            policy_sha256='1' * 64,
            trust_policy_id='fixture-capture-trust-v1',
            trust_policy_sha256='2' * 64,
            verifier_id='fixture-capture-verifier-v1',
            verifier_implementation_sha256='3' * 64,
        ),
        source_verifiers=(
            PrecommittedSourceVerifier(
                source_id='publisher:fixture',
                verifier_id='fixture-source-verifier',
                verifier_version='v1',
                implementation_sha256='4' * 64,
                policy_sha256='5' * 64,
                execution_environment_sha256='9' * 64,
            ),
        ),
        adapter=PrecommittedAdapter(
            adapter_id='fixture-normalizer',
            adapter_version='v1',
            implementation_sha256='6' * 64,
            policy_sha256='7' * 64,
            execution_environment_sha256='8' * 64,
            allowed_exclusion_reason_codes=('not_relevant',),
        ),
    )

    if job_in_checkpoint:
        store.register_job(job_spec, registered_at=_T0 + timedelta(seconds=1))
    if scope_in_checkpoint:
        store.put_bytes(canonical_json_bytes(scope_policy), recorded_at=_T0 + timedelta(seconds=2))
    if plan_in_checkpoint:
        store.put_bytes(canonical_json_bytes(pre_capture_plan), recorded_at=_T0 + timedelta(seconds=2, microseconds=1))

    checkpoint = store.checkpoint(created_at=_T0 + timedelta(seconds=3))

    # Keep the objects available in the live store while proving that the builder
    # accepts only their registration/storage events from the witnessed prefix.
    if not job_in_checkpoint:
        store.register_job(job_spec, registered_at=_T0 + timedelta(seconds=3, microseconds=1))
    if not scope_in_checkpoint:
        store.put_bytes(
            canonical_json_bytes(scope_policy),
            recorded_at=_T0 + timedelta(seconds=3, microseconds=2),
        )
    if not plan_in_checkpoint:
        store.put_bytes(
            canonical_json_bytes(pre_capture_plan),
            recorded_at=_T0 + timedelta(seconds=3, microseconds=3),
        )

    policy = _witness_policy()

    def verifier(target_bytes: bytes, proof_bytes: bytes, expected_policy: WitnessPolicyBinding):
        assert target_bytes == canonical_json_bytes(checkpoint)
        assert proof_bytes == _PROOF
        assert expected_policy == policy
        return AuthenticatedExternalWitnessFacts(
            receipt_id='fixture-scope-checkpoint-receipt',
            authority_id=policy.authority_id,
            witness_id='fixture-independent-log-key-v1',
            method=policy.method,
            policy_id=policy.policy_id,
            checkpoint_sha256=checkpoint_sha256(checkpoint),
            witnessed_at=witnessed_at,
        )

    witness_root = root / 'witness'
    witness_verified_at = witnessed_at + timedelta(seconds=1)
    broker_witness_checkpoint(
        witness_root,
        checkpoint=checkpoint,
        policy=policy,
        provider=lambda _request: (
            ExternalWitnessClaim(verification_uri='https://log.invalid/fixture-scope-checkpoint'),
            _PROOF,
        ),
        verifier=verifier,
        verified_at=witness_verified_at,
    )
    registry = _SelectionRegistry(selected_at=selected_at)
    selection_materials = PlanSelectionMaterialSpec(
        policy=pre_capture_plan.selection_policy,
        policy_bytes=_SELECTION_POLICY_BYTES,
        trust_policy_bytes=_SELECTION_TRUST_BYTES,
        verifier_implementation_bytes=_SELECTION_VERIFIER_BYTES,
        verifier=registry.verifier,
    )
    selection_root = root / 'selection'
    selected_plan = broker_plan_selection(
        selection_root,
        commitment=derive_plan_selection_commitment(scope_policy, pre_capture_plan, checkpoint),
        materials=selection_materials,
        provider=registry.provider,
        verified_at=max(witness_verified_at, selected_at + timedelta(seconds=1)),
    )
    archive_created_at = max(witnessed_at, selected_at) + timedelta(seconds=2)
    return _PrecommitFixture(
        store=store,
        scope_policy=scope_policy,
        pre_capture_plan=pre_capture_plan,
        checkpoint=checkpoint,
        witness_root=witness_root,
        witness_materials=WitnessMaterialSpec(
            policy=policy,
            policy_bytes=_POLICY_BYTES,
            trust_policy_bytes=_TRUST_POLICY_BYTES,
            verifier_implementation_bytes=_VERIFIER_BYTES,
            verifier=verifier,
        ),
        selection_root=selection_root,
        selection_manifest_sha256=selected_plan.manifest_sha256,
        selection_materials=selection_materials,
        created_at=archive_created_at,
        verified_at=max(witness_verified_at, archive_created_at + timedelta(seconds=1)),
    )


def _archive_sha256(value: object) -> str:
    archive_sha256 = getattr(value, 'archive_sha256', None)
    if isinstance(archive_sha256, str):
        return archive_sha256
    manifest_sha256 = getattr(value, 'manifest_sha256')
    assert isinstance(manifest_sha256, str)
    return manifest_sha256


def _build(
    root: Path,
    fixture: _PrecommitFixture,
    *,
    selection_root: Path | None = None,
    expected_selection_manifest_sha256: str | None = None,
    selection_materials: PlanSelectionMaterialSpec | None = None,
):
    return build_scope_precommit(
        root,
        store=fixture.store,
        scope_policy=fixture.scope_policy,
        pre_capture_plan=fixture.pre_capture_plan,
        witness_root=fixture.witness_root,
        witness_materials=fixture.witness_materials,
        selection_root=selection_root or fixture.selection_root,
        expected_selection_manifest_sha256=(expected_selection_manifest_sha256 or fixture.selection_manifest_sha256),
        selection_materials=selection_materials or fixture.selection_materials,
        created_at=fixture.created_at,
        verified_at=fixture.verified_at,
    )


def test_scope_precommit_round_trip_is_portable_and_offline_verifiable(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / 'scope-precommit'

    built = _build(output, fixture)
    archive_sha256 = _archive_sha256(built)
    loaded = load_scope_precommit(
        output,
        expected_archive_sha256=archive_sha256,
        expected_scope_policy=fixture.scope_policy,
        expected_pre_capture_plan=fixture.pre_capture_plan,
        witness_materials=fixture.witness_materials,
        expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
        selection_materials=fixture.selection_materials,
        verified_at=fixture.verified_at,
    )

    assert built.root == output.resolve()
    assert loaded.root == built.root
    assert loaded.manifest == built.manifest
    assert _archive_sha256(loaded) == archive_sha256
    assert loaded.scope_policy == fixture.scope_policy
    assert loaded.pre_capture_plan == fixture.pre_capture_plan
    assert loaded.checkpoint == fixture.checkpoint
    assert loaded.ledger_events == built.ledger_events
    assert loaded.ledger_events[-1].event_sha256 == fixture.checkpoint.through_event_sha256
    assert any(
        event.payload.get('artifact_sha256') == hashlib.sha256(canonical_json_bytes(fixture.scope_policy)).hexdigest()
        for event in loaded.ledger_events
    )
    assert any(
        event.payload.get('artifact_sha256')
        == hashlib.sha256(canonical_json_bytes(fixture.pre_capture_plan)).hexdigest()
        for event in loaded.ledger_events
    )


def test_scope_precommit_rejects_a_witness_after_the_first_scheduled_slot(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        witnessed_at=_FIRST_SCHEDULED_SLOT + timedelta(microseconds=1),
    )

    with pytest.raises(ValueError, match='witness|schedule|window|pre-capture'):
        _build(tmp_path / 'scope-precommit', fixture)


def test_scope_precommit_rejects_a_witness_at_the_first_scheduled_slot(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, witnessed_at=_FIRST_SCHEDULED_SLOT)

    with pytest.raises(ValueError, match='strictly before'):
        _build(tmp_path / 'scope-precommit', fixture)


def test_scope_precommit_rejects_a_scope_for_another_store(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, scope_store_id='b' * 32)

    with pytest.raises(ValueError, match='store'):
        _build(tmp_path / 'scope-precommit', fixture)


def test_scope_precommit_load_requires_the_exact_out_of_band_scope(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    wrong_scope = fixture.scope_policy.model_copy(update={'policy_id': 'different-scope-policy'})

    with pytest.raises(ValueError, match='scope'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256=_archive_sha256(built),
            expected_scope_policy=wrong_scope,
            expected_pre_capture_plan=fixture.pre_capture_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_load_requires_the_exact_archive_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)

    with pytest.raises(ValueError, match='digest|sha256|archive'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256='0' * 64,
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=fixture.pre_capture_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_requires_the_exact_selected_plan_sidecar(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / 'fixture')

    with pytest.raises(ValueError, match='plan-selection|unsafe|artifact'):
        _build(
            tmp_path / 'missing-selection-archive',
            fixture,
            selection_root=tmp_path / 'absent-selection',
        )

    with pytest.raises(ValueError, match='plan-selection|manifest|digest'):
        _build(
            tmp_path / 'wrong-selection-digest-archive',
            fixture,
            expected_selection_manifest_sha256='0' * 64,
        )


def test_scope_precommit_rejects_changed_selection_key_policy_and_material(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    wrong_policy = fixture.selection_materials.policy.model_copy(
        update={'selection_key': 'different-antigen-prioritization-plan'}
    )
    changed_key_materials = PlanSelectionMaterialSpec(
        policy=wrong_policy,
        policy_bytes=fixture.selection_materials.policy_bytes,
        trust_policy_bytes=fixture.selection_materials.trust_policy_bytes,
        verifier_implementation_bytes=fixture.selection_materials.verifier_implementation_bytes,
        verifier=fixture.selection_materials.verifier,
    )
    with pytest.raises(ValueError, match='policy|selection'):
        _build(
            tmp_path / 'changed-key-archive',
            fixture,
            selection_materials=changed_key_materials,
        )

    changed_policy_bytes = PlanSelectionMaterialSpec(
        policy=fixture.selection_materials.policy,
        policy_bytes=b'different selected-plan policy bytes',
        trust_policy_bytes=fixture.selection_materials.trust_policy_bytes,
        verifier_implementation_bytes=fixture.selection_materials.verifier_implementation_bytes,
        verifier=fixture.selection_materials.verifier,
    )
    with pytest.raises(ValueError, match='policy|pinned digest'):
        _build(
            tmp_path / 'changed-policy-material-archive',
            fixture,
            selection_materials=changed_policy_bytes,
        )

    malformed_materials: Any = object()
    with pytest.raises(ValueError, match='materials|interface|selection'):
        _build(
            tmp_path / 'malformed-material-archive',
            fixture,
            selection_materials=malformed_materials,
        )


def test_scope_precommit_derives_the_real_checkpoint_and_first_slot(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / 'fixture')
    registry = _SelectionRegistry(selected_at=_SELECTED_AT)
    materials = PlanSelectionMaterialSpec(
        policy=fixture.pre_capture_plan.selection_policy,
        policy_bytes=_SELECTION_POLICY_BYTES,
        trust_policy_bytes=_SELECTION_TRUST_BYTES,
        verifier_implementation_bytes=_SELECTION_VERIFIER_BYTES,
        verifier=registry.verifier,
    )
    actual = derive_plan_selection_commitment(
        fixture.scope_policy,
        fixture.pre_capture_plan,
        fixture.checkpoint,
    )
    false_commitment = actual.model_copy(
        update={'earliest_scheduled_slot': actual.earliest_scheduled_slot + timedelta(days=1)}
    )
    false_selection = broker_plan_selection(
        tmp_path / 'false-slot-selection',
        commitment=false_commitment,
        materials=materials,
        provider=registry.provider,
        verified_at=fixture.verified_at,
    )

    with pytest.raises(ValueError, match='commitment|selection'):
        _build(
            tmp_path / 'false-slot-archive',
            fixture,
            selection_root=false_selection.root,
            expected_selection_manifest_sha256=false_selection.manifest_sha256,
            selection_materials=materials,
        )


def test_scope_precommit_rejects_late_plan_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='strictly before'):
        _fixture(tmp_path, selected_at=_FIRST_SCHEDULED_SLOT)


@pytest.mark.parametrize(
    ('relative_path', 'replacement'),
    (
        ('selection/sidecar/registry-proof.bin', b'tampered selected-plan proof'),
        ('selection/materials/policy.bin', b'tampered selected-plan policy material'),
    ),
)
def test_scope_precommit_load_rejects_tampered_nested_selection_bytes(
    tmp_path: Path,
    relative_path: str,
    replacement: bytes,
) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    target = built.root / relative_path
    target.chmod(0o644)
    target.write_bytes(replacement)

    with pytest.raises(ValueError, match='selection|payload|binding|digest|proof|policy'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256=built.archive_sha256,
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=fixture.pre_capture_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_load_requires_exact_out_of_band_witness_materials(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    wrong_materials = WitnessMaterialSpec(
        policy=fixture.witness_materials.policy,
        policy_bytes=b'different witness policy bytes',
        trust_policy_bytes=fixture.witness_materials.trust_policy_bytes,
        verifier_implementation_bytes=fixture.witness_materials.verifier_implementation_bytes,
        verifier=fixture.witness_materials.verifier,
    )

    with pytest.raises(ValueError, match='witness policy|pinned digest'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256=built.archive_sha256,
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=fixture.pre_capture_plan,
            witness_materials=wrong_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_load_rejects_a_tampered_portable_ledger_prefix(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    ledger_path = built.root / 'ledger' / 'events.jsonl'
    lines = ledger_path.read_bytes().splitlines(keepends=True)
    assert len(lines) >= 2
    ledger_path.chmod(0o644)
    ledger_path.write_bytes(b''.join(lines[:-1]))

    with pytest.raises(ValueError, match='ledger|prefix|file|digest|inventory'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256=_archive_sha256(built),
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=fixture.pre_capture_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_and_job_must_both_be_committed_inside_the_witnessed_prefix(tmp_path: Path) -> None:
    scope_late = _fixture(tmp_path / 'scope-late', scope_in_checkpoint=False)
    with pytest.raises(ValueError, match='scope|artifact|prefix'):
        _build(tmp_path / 'scope-late-archive', scope_late)

    job_late = _fixture(tmp_path / 'job-late', job_in_checkpoint=False)
    with pytest.raises(ValueError, match='job|register|prefix'):
        _build(tmp_path / 'job-late-archive', job_late)

    plan_late = _fixture(tmp_path / 'plan-late', plan_in_checkpoint=False)
    with pytest.raises(ValueError, match='plan|artifact|prefix'):
        _build(tmp_path / 'plan-late-archive', plan_late)


def test_scope_precommit_commits_the_exact_adapter_exclusion_reason_allowlist(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    wrong_adapter = fixture.pre_capture_plan.adapter.model_copy(
        update={'allowed_exclusion_reason_codes': ('different_reason',)}
    )
    wrong_plan = fixture.pre_capture_plan.model_copy(update={'adapter': wrong_adapter})

    with pytest.raises(ValueError, match='pre-capture plan'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256=built.archive_sha256,
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=wrong_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_commits_the_exact_source_verifier_environment(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    verifier = fixture.pre_capture_plan.source_verifiers[0]
    wrong_verifier = verifier.model_copy(update={'execution_environment_sha256': '0' * 64})
    wrong_plan = fixture.pre_capture_plan.model_copy(update={'source_verifiers': (wrong_verifier,)})

    with pytest.raises(ValueError, match='pre-capture plan'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256=built.archive_sha256,
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=wrong_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_rejects_stale_outer_and_inner_schema_versions(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)

    stale_outer = built.manifest.model_dump(mode='json')
    stale_outer['schema_version'] = 'vaxreplay.scope-precommit.v0.2'
    with pytest.raises(ValueError, match='schema_version'):
        ScopePrecommitManifest.model_validate(stale_outer)

    stale_inner = fixture.pre_capture_plan.model_dump(mode='json')
    stale_inner['schema_version'] = 'vaxreplay.pre-capture-plan.v0.2'
    with pytest.raises(ValueError, match='schema_version'):
        PreCapturePlan.model_validate(stale_inner)


def test_scope_precommit_load_rejects_an_extra_unbound_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    built.root.chmod(0o755)
    (built.root / 'unbound.bin').write_bytes(b'not in the manifest')

    with pytest.raises(ValueError, match='inventory'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256=built.archive_sha256,
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=fixture.pre_capture_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_load_rejects_a_symlinked_bound_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    scope_directory = built.root / 'scope'
    policy_path = scope_directory / 'policy.json'
    scope_directory.chmod(0o755)
    policy_path.unlink()
    policy_path.symlink_to(built.root / 'scope-precommit.json')

    with pytest.raises(ValueError, match='symbolic link'):
        load_scope_precommit(
            built.root,
            expected_archive_sha256=built.archive_sha256,
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=fixture.pre_capture_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_load_rejects_symlink_in_root_parent_chain(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _build(tmp_path / 'scope-precommit', fixture)
    linked_parent = tmp_path / 'linked-parent'
    linked_parent.symlink_to(built.root.parent, target_is_directory=True)

    with pytest.raises(ValueError, match='unsafe scope precommit archive'):
        load_scope_precommit(
            linked_parent / built.root.name,
            expected_archive_sha256=built.archive_sha256,
            expected_scope_policy=fixture.scope_policy,
            expected_pre_capture_plan=fixture.pre_capture_plan,
            witness_materials=fixture.witness_materials,
            expected_selection_manifest_sha256=fixture.selection_manifest_sha256,
            selection_materials=fixture.selection_materials,
            verified_at=fixture.verified_at,
        )


def test_scope_precommit_publication_never_replaces_a_racing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / 'fixture')
    target = tmp_path / 'scope-precommit'
    original_rename = scope_precommit_module.rename_directory_noreplace

    def race_at_install(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / 'owner.txt').write_bytes(b'created by competing publisher')
        original_rename(source, destination)

    monkeypatch.setattr(scope_precommit_module, 'rename_directory_noreplace', race_at_install)
    with pytest.raises(ValueError, match='output already exists'):
        _build(target, fixture)

    assert (target / 'owner.txt').read_bytes() == b'created by competing publisher'
    assert not (tmp_path / '.scope-precommit.publish.lock').exists()
    assert not any(path.name.startswith('.scope-precommit.') for path in tmp_path.iterdir())
