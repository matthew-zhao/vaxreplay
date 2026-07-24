from __future__ import annotations

from datetime import date

import pytest

from vaxreplay.case_schema import Split
from vaxreplay.clinicaltrials.execution_contamination import (
    ExecutionCaseContaminationRisk,
    ExecutionCaseRiskStratum,
    ExecutionCaseSurfaceBinding,
    ExecutionPrivateProbeEvaluation,
    ExecutionProbeKind,
    ExecutionProbeSurfaceVariant,
    ExecutionSystemExposureStatus,
    WorkspaceLeakageStatus,
    build_execution_contamination_strata_manifest,
    build_execution_system_probe_manifest,
)
from vaxreplay.clinicaltrials.execution_contamination_admission import (
    ExecutionCaseSplitBinding,
    ExecutionContaminationAdmissionError,
    ExecutionModelWeightDeclaration,
    ExposureDeclaration,
    build_execution_contamination_admission_manifest,
    build_execution_system_contamination_report,
)

_ATTACKERS = ('a' * 64, 'b' * 64)
_PROBE_BATCH = 'c' * 64
_AUDIT = 'd' * 64
_TARGET_SYSTEM = 'e' * 64


def _risk(index: int, stratum: ExecutionCaseRiskStratum) -> ExecutionCaseContaminationRisk:
    episode_id = f'execution-case-{index:02d}'
    context = f'{index + 1:064x}'
    surface = f'{index + 101:064x}'
    if stratum == ExecutionCaseRiskStratum.WORKSPACE_LEAK_EXCLUDED:
        workspace_status = WorkspaceLeakageStatus.LEAK_DETECTED
        completed, recovery_count = _ATTACKERS, 0
        historical_all, low_risk, reason = False, False, 'workspace_leak_detected'
    elif stratum == ExecutionCaseRiskStratum.WORKSPACE_AUDIT_INCOMPLETE:
        workspace_status = WorkspaceLeakageStatus.AUDIT_INCOMPLETE
        completed, recovery_count = _ATTACKERS, 0
        historical_all, low_risk, reason = False, False, 'workspace_audit_incomplete'
    elif stratum == ExecutionCaseRiskStratum.IDENTITY_FINGERPRINTABLE:
        workspace_status = WorkspaceLeakageStatus.NO_SIGNAL_UNDER_PINNED_AUDIT
        completed, recovery_count = _ATTACKERS, 1
        historical_all, low_risk, reason = True, False, None
    elif stratum == ExecutionCaseRiskStratum.NO_IDENTITY_SIGNAL:
        workspace_status = WorkspaceLeakageStatus.NO_SIGNAL_UNDER_PINNED_AUDIT
        completed, recovery_count = _ATTACKERS, 0
        historical_all, low_risk, reason = True, True, None
    else:
        workspace_status = WorkspaceLeakageStatus.NO_SIGNAL_UNDER_PINNED_AUDIT
        completed, recovery_count = (_ATTACKERS[0],), 0
        historical_all, low_risk, reason = True, False, None
    return ExecutionCaseContaminationRisk(
        episode_id=episode_id,
        task_context_sha256=context,
        public_surface_sha256=surface,
        workspace_leakage_status=workspace_status,
        workspace_audit_receipt_sha256=(
            None if workspace_status == WorkspaceLeakageStatus.AUDIT_INCOMPLETE else _AUDIT
        ),
        organizer_probe_batch_sha256=_PROBE_BATCH,
        precommitted_organizer_attacker_system_manifest_sha256s=_ATTACKERS,
        completed_organizer_attacker_system_manifest_sha256s=completed,
        independent_organizer_attacker_count=len(completed),
        identity_recovery_count=recovery_count,
        stratum=stratum,
        included_in_historical_all=historical_all,
        included_in_historical_common_low_risk=low_risk,
        exclusion_reason=reason,
    )


def _strata_and_splits():
    definitions = (
        (ExecutionCaseRiskStratum.WORKSPACE_LEAK_EXCLUDED, Split.TRAIN),
        (ExecutionCaseRiskStratum.IDENTITY_FINGERPRINTABLE, Split.TRAIN),
        (ExecutionCaseRiskStratum.REIDENTIFICATION_UNMEASURED, Split.DEV),
        (ExecutionCaseRiskStratum.NO_IDENTITY_SIGNAL, Split.DEV),
        (ExecutionCaseRiskStratum.IDENTITY_FINGERPRINTABLE, Split.TEST),
        (ExecutionCaseRiskStratum.REIDENTIFICATION_UNMEASURED, Split.TEST),
        (ExecutionCaseRiskStratum.NO_IDENTITY_SIGNAL, Split.TEST),
        (ExecutionCaseRiskStratum.WORKSPACE_AUDIT_INCOMPLETE, Split.TRAIN),
    )
    risks = tuple(_risk(index, stratum) for index, (stratum, _) in enumerate(definitions))
    universe = tuple(
        ExecutionCaseSurfaceBinding(
            episode_id=risk.episode_id,
            task_context_sha256=risk.task_context_sha256,
            public_surface_sha256=risk.public_surface_sha256,
        )
        for risk in risks
    )
    strata = build_execution_contamination_strata_manifest(
        manifest_id='fixture-strata',
        case_universe=universe,
        case_risks=risks,
    )
    splits = tuple(
        ExecutionCaseSplitBinding(episode_id=risk.episode_id, split=split)
        for risk, (_, split) in zip(risks, definitions, strict=True)
    )
    return strata, splits


def _admission():
    strata, splits = _strata_and_splits()
    return strata, build_execution_contamination_admission_manifest(
        manifest_id='fixture-admission',
        contamination_strata=strata,
        split_bindings=splits,
    )


def _declaration(
    *,
    private: ExposureDeclaration = ExposureDeclaration.UNKNOWN,
    test_tuning: ExposureDeclaration = ExposureDeclaration.UNKNOWN,
    public: ExposureDeclaration = ExposureDeclaration.UNKNOWN,
) -> ExecutionModelWeightDeclaration:
    return ExecutionModelWeightDeclaration(
        system_manifest_sha256=_TARGET_SYSTEM,
        declared_model_id='fixture-model',
        declared_training_data_cutoff=date(2025, 1, 1),
        public_aact_or_linked_publication_exposure=public,
        organizer_private_eval_material_exposure=private,
        benchmark_specific_test_tuning=test_tuning,
        machine_unlearning_attempted=False,
        declaration_basis='Provider training corpus is not fully enumerated.',
        submitted_by='fixture-submitter',
    )


def _system_evaluation(
    risk: ExecutionCaseContaminationRisk,
    *,
    index: int,
    positive: bool,
) -> ExecutionPrivateProbeEvaluation:
    future_fields = ('enrollment_ratio', 'primary_completion_slippage_days') if positive else ()
    return ExecutionPrivateProbeEvaluation(
        challenge_id=f'probe-{index + 1:024x}',
        challenge_sha256=f'{index + 201:064x}',
        response_sha256=f'{index + 301:064x}',
        episode_id=risk.episode_id,
        task_context_sha256=risk.task_context_sha256,
        public_surface_sha256=risk.public_surface_sha256,
        system_manifest_sha256=_TARGET_SYSTEM,
        session_isolation_receipt_sha256=f'{index + 401:064x}',
        probe_kind=ExecutionProbeKind.PARAMETRIC_RECALL,
        surface_variant=ExecutionProbeSurfaceVariant.IDENTITY_SCRUBBED,
        exact_registry_identifier_match=positive,
        exact_identity_alias_match_count=0,
        identity_recovered=positive,
        matched_future_fields=future_fields,
        high_specificity_future_match=positive,
        exposure_status=(
            ExecutionSystemExposureStatus.PROBE_POSITIVE_MEMORY_SIGNAL
            if positive
            else ExecutionSystemExposureStatus.NO_SIGNAL
        ),
    )


def test_case_policy_is_permissive_for_train_and_strict_for_primary_test_view() -> None:
    _, admission = _admission()

    assert admission.case_count == 8
    assert admission.train_eligible_count == 1
    assert admission.dev_historical_all_count == 2
    assert admission.dev_common_low_risk_count == 1
    assert admission.test_historical_all_count == 3
    assert admission.test_common_low_risk_count == 1
    assert admission.primary_leaderboard_count == 1
    assert admission.globally_excluded_count == 2

    fingerprintable_train = admission.cases[1]
    assert fingerprintable_train.stratum == ExecutionCaseRiskStratum.IDENTITY_FINGERPRINTABLE
    assert fingerprintable_train.eligible_for_train_use
    unmeasured_test = admission.cases[5]
    assert unmeasured_test.eligible_for_test_historical_all
    assert not unmeasured_test.eligible_for_primary_leaderboard


def test_split_bindings_must_cover_the_complete_frozen_case_universe() -> None:
    strata, splits = _strata_and_splits()
    with pytest.raises(ExecutionContaminationAdmissionError, match='exactly cover'):
        build_execution_contamination_admission_manifest(
            manifest_id='incomplete',
            contamination_strata=strata,
            split_bindings=splits[:-1],
        )


def test_unknown_or_public_model_exposure_is_labeled_without_forcing_unlearning() -> None:
    _, admission = _admission()
    report = build_execution_system_contamination_report(
        case_admission=admission,
        declaration=_declaration(public=ExposureDeclaration.YES),
        system_probe=None,
    )

    assert report.eligible_for_held_out_leaderboard
    assert report.public_source_exposure == ExposureDeclaration.YES
    assert report.probe_coverage_status == 'not_run'
    assert not report.machine_unlearning_required
    assert not report.declaration_proves_clean_weights
    assert report.residual_model_weight_contamination_possible


@pytest.mark.parametrize(
    ('private', 'test_tuning', 'reason'),
    (
        (
            ExposureDeclaration.YES,
            ExposureDeclaration.NO,
            'known_organizer_private_eval_material_exposure',
        ),
        (
            ExposureDeclaration.NO,
            ExposureDeclaration.YES,
            'known_benchmark_specific_test_tuning',
        ),
    ),
)
def test_known_private_eval_access_or_test_tuning_disqualifies_only_held_out_leaderboard(
    private: ExposureDeclaration,
    test_tuning: ExposureDeclaration,
    reason: str,
) -> None:
    _, admission = _admission()
    report = build_execution_system_contamination_report(
        case_admission=admission,
        declaration=_declaration(private=private, test_tuning=test_tuning),
        system_probe=None,
    )

    assert not report.eligible_for_held_out_leaderboard
    assert report.eligible_for_train_and_dev_experimentation
    assert report.held_out_ineligibility_reasons == (reason,)


def test_positive_memory_probe_is_reported_beside_score_without_changing_denominator() -> None:
    strata, admission = _admission()
    included = tuple(risk for risk in strata.cases if risk.included_in_historical_all)
    evaluations = tuple(
        _system_evaluation(risk, index=index, positive=index == 0) for index, risk in enumerate(included)
    )
    probe = build_execution_system_probe_manifest(
        case_strata_manifest=strata,
        system_manifest_sha256=_TARGET_SYSTEM,
        evaluations=evaluations,
    )
    report = build_execution_system_contamination_report(
        case_admission=admission,
        declaration=_declaration(),
        system_probe=probe,
    )

    assert report.eligible_for_held_out_leaderboard
    assert not report.case_denominator_changed
    assert dict(report.probe_status_counts)[ExecutionSystemExposureStatus.PROBE_POSITIVE_MEMORY_SIGNAL] == 1
    assert report.no_signal_proves_clean_weights is False
