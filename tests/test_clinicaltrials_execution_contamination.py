from __future__ import annotations

import hashlib
from datetime import date

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_contamination import (
    ExecutionCaseContaminationEvidence,
    ExecutionCaseRiskStratum,
    ExecutionCaseSurfaceBinding,
    ExecutionContaminationControlError,
    ExecutionProbeKind,
    ExecutionProbeResponse,
    ExecutionProbeSurfaceVariant,
    ExecutionSystemExposureStatus,
    LaterRegistryRecallClaim,
    ProbeClaimBasis,
    WorkspaceLeakageStatus,
    accept_execution_probe_response,
    assess_execution_case_contamination,
    build_execution_contamination_strata_manifest,
    build_execution_system_probe_manifest,
    evaluate_execution_probe_response,
    execution_probe_challenge_sha256,
    execution_probe_prompt_bytes,
    make_execution_probe_challenge,
    make_execution_probe_private_gold,
)
from vaxreplay.clinicaltrials.execution_schema import ObservationState, RegistryOutcomeClass
from vaxreplay.clinicaltrials.execution_task import (
    ContinuousForecastSpec,
    ExecutionPrivateGold,
    ExecutionTaskContext,
    build_execution_task,
    execution_task_context_sha256,
)

_EXECUTION_GOLD_KEY = bytes(range(32))
_PROBE_GOLD_KEY = bytes(range(32, 64))
_SYSTEM_A = 'a' * 64
_SYSTEM_B = 'b' * 64
_ORGANIZER_ATTACKERS = (_SYSTEM_A, _SYSTEM_B)
_WORKSPACE_AUDIT = 'c' * 64
_PROBE_BATCH = 'd' * 64
_SESSION = 'e' * 64


def _case():
    context = ExecutionTaskContext(
        episode_id='execution-dev-00112233445566778899aabb',
        target_trial_id='trial-target',
        decision_snapshot_id='aact-2020-03-02',
        anchor_date=date(2020, 3, 2),
        label_snapshot_id='aact-2024-03-02',
        label_archive_date=date(2024, 3, 2),
        planned_enrollment=100,
        planned_primary_completion_date=date(2021, 1, 1),
        enrollment_ratio_spec=ContinuousForecastSpec(
            forecast_kind='point',
            lower_bound=0.0,
            upper_bound=2.0,
        ),
        primary_completion_slippage_days_spec=ContinuousForecastSpec(
            forecast_kind='point',
            lower_bound=-365.0,
            upper_bound=730.0,
        ),
    )
    gold = ExecutionPrivateGold(
        episode_id=context.episode_id,
        target_trial_id=context.target_trial_id,
        organizer_private_nct_id='NCT00000001',
        organizer_private_decision_record_sha256='f' * 64,
        task_context_sha256=execution_task_context_sha256(context),
        registry_outcome_class=RegistryOutcomeClass.COMPLETED,
        enrollment_observation=ObservationState.OBSERVED_ACTUAL,
        enrollment_ratio=0.83,
        primary_completion_observation=ObservationState.OBSERVED_ACTUAL,
        primary_completion_slippage_days=47,
    )
    task = build_execution_task(
        context=context,
        gold=gold,
        private_gold_key=_EXECUTION_GOLD_KEY,
    )
    probe_gold = make_execution_probe_private_gold(
        task=task,
        execution_gold=gold,
        execution_gold_key=_EXECUTION_GOLD_KEY,
        identity_aliases=('Example Vaccine Study',),
    )
    public_surface = canonical_json_bytes(
        {
            'task': task.model_dump(mode='json'),
            'decision_time_metadata': {'phase': 'phase_1', 'condition': 'infectious disease'},
        }
    )
    return task, probe_gold, public_surface


def _challenge(
    *,
    system: str = _SYSTEM_A,
    kind: ExecutionProbeKind = ExecutionProbeKind.REIDENTIFICATION,
):
    task, probe_gold, public_surface = _case()
    challenge = make_execution_probe_challenge(
        task=task,
        probe_gold=probe_gold,
        probe_gold_key=_PROBE_GOLD_KEY,
        public_surface=public_surface,
        probe_kind=kind,
        surface_variant=ExecutionProbeSurfaceVariant.RELEASED_TASK,
        system_manifest_sha256=system,
    )
    return challenge, probe_gold, public_surface


def _response(
    challenge,
    *,
    registry_id: str | None = None,
    aliases: tuple[str, ...] = (),
    later: LaterRegistryRecallClaim | None = None,
    basis: ProbeClaimBasis = ProbeClaimBasis.UNCERTAIN,
):
    abstained = registry_id is None and not aliases and later is None
    return ExecutionProbeResponse(
        challenge_id=challenge.challenge_id,
        challenge_sha256=execution_probe_challenge_sha256(challenge),
        probe_kind=challenge.probe_kind,
        system_manifest_sha256=challenge.system_manifest_sha256,
        session_isolation_receipt_sha256=_SESSION,
        abstained=abstained,
        declared_basis=ProbeClaimBasis.UNKNOWN if abstained else basis,
        registry_identifier_claim=registry_id,
        identity_name_claims=aliases,
        later_registry_claim=later,
    )


def _evaluation(
    *,
    system: str,
    registry_id: str | None,
    kind: ExecutionProbeKind = ExecutionProbeKind.REIDENTIFICATION,
    later: LaterRegistryRecallClaim | None = None,
):
    challenge, probe_gold, _ = _challenge(system=system, kind=kind)
    response = _response(challenge, registry_id=registry_id, later=later)
    return evaluate_execution_probe_response(
        challenge=challenge,
        probe_gold=probe_gold,
        probe_gold_key=_PROBE_GOLD_KEY,
        response=response,
    )


def test_public_probe_is_outcome_blind_and_receipt_returns_no_correctness() -> None:
    challenge, _, _ = _challenge()
    prompt = execution_probe_prompt_bytes(challenge)

    assert b'NCT00000001' not in prompt
    assert b'Example Vaccine Study' not in prompt
    assert b'"completed"' not in prompt
    assert b'0.83' not in prompt
    assert b'"primary_completion_slippage_days":47' not in prompt
    assert b'"correctness_feedback":null' in prompt

    response = _response(challenge, registry_id='NCT99999999')
    receipt = accept_execution_probe_response(challenge, response)
    receipt_bytes = canonical_json_bytes(receipt)

    assert b'NCT99999999' not in receipt_bytes
    assert not receipt.correctness_evaluated_in_this_receipt
    assert not receipt.correctness_feedback_returned
    assert not receipt.private_identity_returned
    assert not receipt.private_outcome_returned


def test_reidentification_signal_is_not_labeled_parametric_memory() -> None:
    exact = _evaluation(system=_SYSTEM_A, registry_id='nct00000001')

    assert exact.identity_recovered
    assert exact.exact_registry_identifier_match
    assert exact.matched_future_fields == ()
    assert exact.exposure_status == ExecutionSystemExposureStatus.IDENTITY_SIGNAL_ONLY
    assert not exact.proves_training_membership
    assert not exact.score_denominator_changed


def test_strong_memory_signal_requires_identity_two_future_matches_and_quantitative_detail() -> None:
    strong = _evaluation(
        system=_SYSTEM_A,
        registry_id='NCT00000001',
        kind=ExecutionProbeKind.PARAMETRIC_RECALL,
        later=LaterRegistryRecallClaim(
            registry_outcome_class=RegistryOutcomeClass.COMPLETED,
            enrollment_ratio=0.83,
        ),
    )
    qualitative_only = _evaluation(
        system=_SYSTEM_A,
        registry_id='NCT00000001',
        kind=ExecutionProbeKind.PARAMETRIC_RECALL,
        later=LaterRegistryRecallClaim(
            registry_outcome_class=RegistryOutcomeClass.COMPLETED,
            enrollment_observation=ObservationState.OBSERVED_ACTUAL,
        ),
    )
    forecast_like = _evaluation(
        system=_SYSTEM_A,
        registry_id=None,
        kind=ExecutionProbeKind.PARAMETRIC_RECALL,
        later=LaterRegistryRecallClaim(
            registry_outcome_class=RegistryOutcomeClass.COMPLETED,
            enrollment_ratio=0.83,
        ),
    )

    assert strong.exposure_status == ExecutionSystemExposureStatus.PROBE_POSITIVE_MEMORY_SIGNAL
    assert strong.matched_future_fields == ('registry_outcome_class', 'enrollment_ratio')
    assert strong.high_specificity_future_match
    assert qualitative_only.exposure_status == ExecutionSystemExposureStatus.IDENTITY_SIGNAL_ONLY
    assert forecast_like.exposure_status == (ExecutionSystemExposureStatus.FUTURE_MATCH_WITHOUT_IDENTITY_AMBIGUOUS)
    assert not forecast_like.proves_training_membership


def test_case_fingerprinting_is_global_and_does_not_change_historical_all() -> None:
    challenge, _, _ = _challenge()
    recovered = _evaluation(system=_SYSTEM_A, registry_id='NCT00000001')
    missed = _evaluation(system=_SYSTEM_B, registry_id='NCT99999999')
    evidence = ExecutionCaseContaminationEvidence(
        episode_id=challenge.episode_id,
        task_context_sha256=challenge.task_context_sha256,
        public_surface_sha256=challenge.public_surface_sha256,
        workspace_leakage_status=WorkspaceLeakageStatus.NO_SIGNAL_UNDER_PINNED_AUDIT,
        workspace_audit_receipt_sha256=_WORKSPACE_AUDIT,
        organizer_probe_batch_sha256=_PROBE_BATCH,
        precommitted_organizer_attacker_system_manifest_sha256s=_ORGANIZER_ATTACKERS,
        organizer_reidentification_evaluations=(recovered, missed),
    )

    risk = assess_execution_case_contamination(evidence)

    assert risk.stratum == ExecutionCaseRiskStratum.IDENTITY_FINGERPRINTABLE
    assert risk.included_in_historical_all
    assert not risk.included_in_historical_common_low_risk
    assert not risk.target_system_results_used_for_case_selection


def test_partial_precommitted_attacker_batch_cannot_create_low_risk_case() -> None:
    challenge, _, _ = _challenge()
    first = _evaluation(system=_SYSTEM_A, registry_id='NCT99999999')
    second = _evaluation(system=_SYSTEM_B, registry_id='NCT99999998')
    evidence = ExecutionCaseContaminationEvidence(
        episode_id=challenge.episode_id,
        task_context_sha256=challenge.task_context_sha256,
        public_surface_sha256=challenge.public_surface_sha256,
        workspace_leakage_status=WorkspaceLeakageStatus.NO_SIGNAL_UNDER_PINNED_AUDIT,
        workspace_audit_receipt_sha256=_WORKSPACE_AUDIT,
        organizer_probe_batch_sha256=_PROBE_BATCH,
        precommitted_organizer_attacker_system_manifest_sha256s=(
            _SYSTEM_A,
            _SYSTEM_B,
            'f' * 64,
        ),
        organizer_reidentification_evaluations=(first, second),
    )

    risk = assess_execution_case_contamination(evidence)

    assert risk.independent_organizer_attacker_count == 2
    assert risk.stratum == ExecutionCaseRiskStratum.REIDENTIFICATION_UNMEASURED
    assert risk.included_in_historical_all
    assert not risk.included_in_historical_common_low_risk


@pytest.mark.parametrize(
    ('status', 'expected_stratum'),
    (
        (WorkspaceLeakageStatus.LEAK_DETECTED, ExecutionCaseRiskStratum.WORKSPACE_LEAK_EXCLUDED),
        (
            WorkspaceLeakageStatus.AUDIT_INCOMPLETE,
            ExecutionCaseRiskStratum.WORKSPACE_AUDIT_INCOMPLETE,
        ),
    ),
)
def test_workspace_leak_or_incomplete_audit_is_the_only_global_exclusion(
    status: WorkspaceLeakageStatus,
    expected_stratum: ExecutionCaseRiskStratum,
) -> None:
    challenge, _, _ = _challenge()
    evidence = ExecutionCaseContaminationEvidence(
        episode_id=challenge.episode_id,
        task_context_sha256=challenge.task_context_sha256,
        public_surface_sha256=challenge.public_surface_sha256,
        workspace_leakage_status=status,
        workspace_audit_receipt_sha256=None if status == WorkspaceLeakageStatus.AUDIT_INCOMPLETE else _WORKSPACE_AUDIT,
        organizer_probe_batch_sha256=_PROBE_BATCH,
        precommitted_organizer_attacker_system_manifest_sha256s=_ORGANIZER_ATTACKERS,
    )

    risk = assess_execution_case_contamination(evidence)

    assert risk.stratum == expected_stratum
    assert not risk.included_in_historical_all
    assert not risk.included_in_historical_common_low_risk


def test_no_signal_remains_residual_risk_not_a_clean_weights_claim() -> None:
    challenge, probe_gold, _ = _challenge(kind=ExecutionProbeKind.PARAMETRIC_RECALL)
    evaluation = evaluate_execution_probe_response(
        challenge=challenge,
        probe_gold=probe_gold,
        probe_gold_key=_PROBE_GOLD_KEY,
        response=_response(challenge, registry_id='NCT99999999'),
    )

    assert evaluation.exposure_status == ExecutionSystemExposureStatus.NO_SIGNAL
    assert evaluation.no_signal_is_not_proof_of_clean_weights
    assert evaluation.residual_model_weight_contamination_possible
    assert not evaluation.proves_absence_of_contamination


def test_missing_organizer_probe_cannot_count_toward_low_risk_threshold() -> None:
    challenge, probe_gold, _ = _challenge()
    unknown = evaluate_execution_probe_response(
        challenge=challenge,
        probe_gold=probe_gold,
        probe_gold_key=_PROBE_GOLD_KEY,
        response=None,
    )
    with pytest.raises(ValueError, match='cannot count'):
        ExecutionCaseContaminationEvidence(
            episode_id=challenge.episode_id,
            task_context_sha256=challenge.task_context_sha256,
            public_surface_sha256=challenge.public_surface_sha256,
            workspace_leakage_status=WorkspaceLeakageStatus.NO_SIGNAL_UNDER_PINNED_AUDIT,
            workspace_audit_receipt_sha256=_WORKSPACE_AUDIT,
            organizer_probe_batch_sha256=_PROBE_BATCH,
            precommitted_organizer_attacker_system_manifest_sha256s=_ORGANIZER_ATTACKERS,
            organizer_reidentification_evaluations=(unknown,),
        )


def test_probe_gold_surface_and_system_tampering_fail_closed() -> None:
    challenge, probe_gold, public_surface = _challenge()
    task, _, _ = _case()
    with pytest.raises(ExecutionContaminationControlError, match='exposes an NCT'):
        make_execution_probe_challenge(
            task=task,
            probe_gold=probe_gold,
            probe_gold_key=_PROBE_GOLD_KEY,
            public_surface=public_surface + b' NCT00000001',
            probe_kind=ExecutionProbeKind.REIDENTIFICATION,
            surface_variant=ExecutionProbeSurfaceVariant.RELEASED_TASK,
            system_manifest_sha256=_SYSTEM_A,
        )

    response = _response(challenge, registry_id='NCT00000001').model_copy(update={'system_manifest_sha256': _SYSTEM_B})
    with pytest.raises(ExecutionContaminationControlError, match='different system'):
        accept_execution_probe_response(challenge, response)

    tampered_gold = probe_gold.model_copy(update={'organizer_private_nct_id': 'NCT00000002'})
    with pytest.raises(ExecutionContaminationControlError, match='does not match the challenge HMAC'):
        evaluate_execution_probe_response(
            challenge=challenge,
            probe_gold=tampered_gold,
            probe_gold_key=_PROBE_GOLD_KEY,
            response=None,
        )


def test_complete_case_manifest_and_system_probe_keep_one_fixed_denominator() -> None:
    challenge, _, _ = _challenge()
    first = _evaluation(system=_SYSTEM_A, registry_id='NCT99999999')
    second = _evaluation(system=_SYSTEM_B, registry_id='NCT99999998')
    evidence = ExecutionCaseContaminationEvidence(
        episode_id=challenge.episode_id,
        task_context_sha256=challenge.task_context_sha256,
        public_surface_sha256=challenge.public_surface_sha256,
        workspace_leakage_status=WorkspaceLeakageStatus.NO_SIGNAL_UNDER_PINNED_AUDIT,
        workspace_audit_receipt_sha256=_WORKSPACE_AUDIT,
        organizer_probe_batch_sha256=_PROBE_BATCH,
        precommitted_organizer_attacker_system_manifest_sha256s=_ORGANIZER_ATTACKERS,
        organizer_reidentification_evaluations=(first, second),
    )
    risk = assess_execution_case_contamination(evidence)
    universe = ExecutionCaseSurfaceBinding(
        episode_id=risk.episode_id,
        task_context_sha256=risk.task_context_sha256,
        public_surface_sha256=risk.public_surface_sha256,
    )
    strata = build_execution_contamination_strata_manifest(
        manifest_id='fictional-conformance-contamination-strata-v0.1',
        case_universe=(universe,),
        case_risks=(risk,),
    )

    assert risk.stratum == ExecutionCaseRiskStratum.NO_IDENTITY_SIGNAL
    assert strata.historical_all_count == 1
    assert strata.historical_common_low_risk_count == 1

    system_evaluation = _evaluation(
        system=_SYSTEM_A,
        registry_id='NCT00000001',
        kind=ExecutionProbeKind.PARAMETRIC_RECALL,
        later=LaterRegistryRecallClaim(
            registry_outcome_class=RegistryOutcomeClass.COMPLETED,
            enrollment_ratio=0.83,
        ),
    )
    system_manifest = build_execution_system_probe_manifest(
        case_strata_manifest=strata,
        system_manifest_sha256=_SYSTEM_A,
        evaluations=(system_evaluation,),
    )

    assert system_manifest.evaluated_case_count == strata.historical_all_count
    assert not system_manifest.score_denominator_changed
    assert not system_manifest.score_corrected_using_probe
    assert system_manifest.probe_reported_beside_score

    with pytest.raises(ExecutionContaminationControlError, match='exact task context and surface'):
        build_execution_system_probe_manifest(
            case_strata_manifest=strata,
            system_manifest_sha256=_SYSTEM_A,
            evaluations=(system_evaluation.model_copy(update={'public_surface_sha256': '0' * 64}),),
        )

    with pytest.raises(ExecutionContaminationControlError, match='exactly cover'):
        build_execution_contamination_strata_manifest(
            manifest_id='incomplete',
            case_universe=(universe,),
            case_risks=(),
        )


def test_probe_construction_and_evaluation_are_deterministic() -> None:
    first_challenge, first_gold, _ = _challenge(kind=ExecutionProbeKind.PARAMETRIC_RECALL)
    second_challenge, second_gold, _ = _challenge(kind=ExecutionProbeKind.PARAMETRIC_RECALL)
    assert canonical_json_bytes(first_challenge) == canonical_json_bytes(second_challenge)
    assert canonical_json_bytes(first_gold) == canonical_json_bytes(second_gold)

    response = _response(
        first_challenge,
        aliases=('example vaccine study',),
        later=LaterRegistryRecallClaim(
            registry_outcome_class=RegistryOutcomeClass.COMPLETED,
            primary_completion_slippage_days=47,
        ),
    )
    first = evaluate_execution_probe_response(
        challenge=first_challenge,
        probe_gold=first_gold,
        probe_gold_key=_PROBE_GOLD_KEY,
        response=response,
    )
    second = evaluate_execution_probe_response(
        challenge=second_challenge,
        probe_gold=second_gold,
        probe_gold_key=_PROBE_GOLD_KEY,
        response=response,
    )

    assert hashlib.sha256(canonical_json_bytes(first)).digest() == hashlib.sha256(canonical_json_bytes(second)).digest()
    assert first.exposure_status == ExecutionSystemExposureStatus.PROBE_POSITIVE_MEMORY_SIGNAL
