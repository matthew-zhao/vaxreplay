from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tests.test_clinicaltrials_execution_scoring import _context, _gold, _submission
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split
from vaxreplay.clinicaltrials.execution_aggregation import (
    ExecutionCohortAggregationError,
    ExecutionCohortEvaluationCase,
    ExecutionCohortEvaluator,
    ExecutionCohortFailureCode,
    ExecutionCohortFailureResult,
    ExecutionCohortManifest,
    ExecutionCohortResult,
    ExecutionCohortSubmission,
    execution_cohort_manifest_sha256,
    make_execution_cohort_manifest,
    make_execution_cohort_submission,
)
from vaxreplay.clinicaltrials.execution_schema import ObservationState, RegistryOutcomeClass
from vaxreplay.clinicaltrials.execution_scoring import ExecutionSubmissionEvaluator
from vaxreplay.clinicaltrials.execution_task import build_execution_task

_KEY = bytes(range(32))
_LINEAGE_SPLIT_SHA256 = 'a' * 64
_WORKSPACE_RECEIPT_SHA256 = 'b' * 64
_GOLD_RECEIPT_SHA256 = 'c' * 64


def _lineage(index: int) -> str:
    return f'lineage-{index:020x}'


def _case(
    index: int,
    *,
    split: Split = Split.TRAIN,
    public_lineage_id: str | None = None,
    registry_outcome_class: RegistryOutcomeClass = RegistryOutcomeClass.COMPLETED,
    enrollment_observation: ObservationState = ObservationState.OBSERVED_ACTUAL,
    enrollment_ratio: float | None = 0.8,
    completion_observation: ObservationState = ObservationState.OBSERVED_ACTUAL,
    slippage_days: int | None = 30,
) -> ExecutionCohortEvaluationCase:
    context = _context(with_fact=False).model_copy(
        update={
            'episode_id': f'execution-cohort-{index:03d}',
            'target_trial_id': 'trial-target',
        }
    )
    gold = _gold(
        context,
        registry_outcome_class=registry_outcome_class,
        enrollment_observation=enrollment_observation,
        enrollment_ratio=enrollment_ratio,
        completion_observation=completion_observation,
        slippage_days=slippage_days,
    ).model_copy(update={'organizer_private_nct_id': f'NCT{index:08d}'})
    task = build_execution_task(context=context, gold=gold, private_gold_key=_KEY)
    return ExecutionCohortEvaluationCase(
        task=task,
        private_gold=gold,
        private_gold_key=_KEY,
        split=split,
        public_lineage_id=public_lineage_id or _lineage(index),
    )


def _manifest(cases: tuple[ExecutionCohortEvaluationCase, ...]) -> ExecutionCohortManifest:
    return make_execution_cohort_manifest(
        cohort_id='fictional-conformance-cohort',
        cases=cases,
        lineage_split_manifest_sha256=_LINEAGE_SPLIT_SHA256,
        workspace_build_receipt_sha256=_WORKSPACE_RECEIPT_SHA256,
        gold_derivation_receipt_sha256=_GOLD_RECEIPT_SHA256,
    )


def test_manifest_is_deterministic_and_binds_task_context_gold_lineage_and_split() -> None:
    cases = (
        _case(2, split=Split.TEST),
        _case(1, split=Split.TEST),
        _case(3, split=Split.TEST),
    )
    first = _manifest(cases)
    second = _manifest(tuple(reversed(cases)))

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert tuple(item.episode_id for item in first.tasks) == (
        'execution-cohort-001',
        'execution-cohort-002',
        'execution-cohort-003',
    )
    assert first.task_count == first.lineage_count == 3
    assert tuple((item.split, item.task_count, item.lineage_count) for item in first.split_counts) == (
        (Split.TRAIN, 0, 0),
        (Split.DEV, 0, 0),
        (Split.TEST, 3, 3),
    )
    assert first.evaluation_split == Split.TEST
    for binding, case in zip(first.tasks, sorted(cases, key=lambda item: item.task.context.episode_id), strict=True):
        assert binding.task_context_sha256 == case.task.context_sha256
        assert binding.private_gold_commitment_sha256 == case.task.private_gold_commitment_sha256
        assert binding.private_gold_commitment_key_id == case.task.private_gold_commitment_key_id
        assert binding.task_sha256
        assert binding.private_gold_sha256
    assert first.organizer_private and first.exact_task_context_and_gold_bindings
    assert first.lineage_split_isolated
    assert first.development_only and not first.leaderboard_admitted
    assert not first.sealed_execution_supported and not first.identity_contamination_controlled
    assert not first.external_receipts_authenticated_for_admission


def test_manifest_rejects_a_lineage_that_crosses_splits_and_private_case_tampering() -> None:
    lineage = _lineage(99)
    with pytest.raises(ExecutionCohortAggregationError, match='exactly one split'):
        _manifest(
            (
                _case(1, split=Split.TRAIN, public_lineage_id=lineage),
                _case(2, split=Split.DEV, public_lineage_id=lineage),
            )
        )

    valid = _case(1)
    wrong_gold = valid.private_gold.model_copy(update={'registry_outcome_class': RegistryOutcomeClass.TERMINATED})
    with pytest.raises(ValueError, match='commitment'):
        _manifest(
            (
                ExecutionCohortEvaluationCase(
                    task=valid.task,
                    private_gold=wrong_gold,
                    private_gold_key=valid.private_gold_key,
                    split=valid.split,
                    public_lineage_id=valid.public_lineage_id,
                ),
            )
        )


def test_evaluator_handles_complete_multi_task_coverage_as_one_macro_result() -> None:
    cases = tuple(_case(index, split=Split.TEST) for index in range(1, 8))
    manifest = _manifest(cases)
    batch = make_execution_cohort_submission(
        manifest=manifest,
        submissions=reversed([_submission(case.task) for case in cases]),
    )

    result = ExecutionCohortEvaluator(manifest=manifest, cases=reversed(cases)).score(batch)

    assert result.task_count == result.valid_task_count == result.metrics.task_count == 7
    assert result.invalid_task_count == 0
    assert tuple(item.task_count for item in result.split_counts) == (0, 0, 7)
    assert result.evaluation_split == Split.TEST
    assert result.metrics.mean_reward == pytest.approx(1.0)
    assert result.metrics.mean_core_reward == pytest.approx(1.0)
    assert result.metrics.enrollment_continuous.applied_task_count == 7
    assert result.metrics.primary_completion_continuous.applied_task_count == 7
    assert result.metrics.cutoff_facts.configured_task_count == 0
    assert result.full_manifest_coverage_verified and result.exactly_one_submission_per_task_verified
    assert result.aggregation_rejects_episode_subset_input
    assert not result.participant_visible_episode_scores_included
    assert result.development_only and not result.leaderboard_admitted
    assert not result.tier_b_admitted and not result.tier_a_official
    assert not result.sealed_execution_supported and not result.identity_contamination_controlled
    assert 'episode_scores' not in result.model_dump()


def test_component_metrics_use_full_macro_and_explicit_conditional_denominators() -> None:
    cases = (
        _case(1),
        _case(
            2,
            enrollment_observation=ObservationState.NOT_ACTUAL,
            enrollment_ratio=None,
        ),
        _case(
            3,
            completion_observation=ObservationState.VALUE_MISSING,
            slippage_days=None,
        ),
    )
    manifest = _manifest(cases)
    submissions = (
        _submission(cases[0].task),
        _submission(
            cases[1].task,
            enrollment_observation=ObservationState.NOT_ACTUAL,
        ),
        _submission(
            cases[2].task,
            registry_outcome=RegistryOutcomeClass.TERMINATED,
            completion_observation=ObservationState.VALUE_MISSING,
            enrollment_point=1.8,
        ),
    )
    individual_scores = tuple(
        ExecutionSubmissionEvaluator(
            task=case.task,
            private_gold=case.private_gold,
            private_gold_key=case.private_gold_key,
        ).score(submission)
        for case, submission in zip(cases, submissions, strict=True)
    )

    result = ExecutionCohortEvaluator(manifest=manifest, cases=cases).score(
        make_execution_cohort_submission(manifest=manifest, submissions=submissions)
    )

    expected_reward = math.fsum(score.reward or 0.0 for score in individual_scores) / 3
    assert result.metrics.mean_reward == pytest.approx(expected_reward)
    assert result.metrics.enrollment_continuous.applied_task_count == 2
    assert result.metrics.enrollment_continuous.applied_rate == pytest.approx(2 / 3)
    assert result.metrics.enrollment_continuous.mean_reward_when_applied == pytest.approx(0.875)
    assert result.metrics.enrollment_continuous.mean_fixed_reward_all_tasks == pytest.approx((1.0 + 1.0 + 0.75) / 3)
    assert result.metrics.primary_completion_continuous.applied_task_count == 2
    assert result.metrics.primary_completion_continuous.applied_rate == pytest.approx(2 / 3)
    assert result.metrics.mean_registry_outcome_reward == pytest.approx(2 / 3)


def test_missing_extra_duplicate_wrong_manifest_and_task_invalid_batches_fail_closed() -> None:
    cases = (_case(1), _case(2))
    manifest = _manifest(cases)
    valid_submissions = tuple(_submission(case.task) for case in cases)
    evaluator = ExecutionCohortEvaluator(manifest=manifest, cases=cases)

    missing = make_execution_cohort_submission(manifest=manifest, submissions=valid_submissions[:1])
    with pytest.raises(ExecutionCohortAggregationError, match='missing='):
        evaluator.score(missing)

    extra_submission = valid_submissions[1].model_copy(update={'episode_id': 'execution-cohort-999'})
    extra = make_execution_cohort_submission(
        manifest=manifest,
        submissions=(*valid_submissions, extra_submission),
    )
    with pytest.raises(ExecutionCohortAggregationError, match='extra='):
        evaluator.score(extra)

    with pytest.raises(ValidationError, match='unique ascending'):
        ExecutionCohortSubmission(
            cohort_id=manifest.cohort_id,
            cohort_manifest_sha256=execution_cohort_manifest_sha256(manifest),
            submissions=(valid_submissions[0], valid_submissions[0]),
        )

    wrong_manifest = make_execution_cohort_submission(manifest=manifest, submissions=valid_submissions).model_copy(
        update={'cohort_manifest_sha256': '0' * 64}
    )
    with pytest.raises(ExecutionCohortAggregationError, match='exact manifest'):
        evaluator.score(wrong_manifest)

    invalid_submission = valid_submissions[0].model_copy(update={'task_context_sha256': '0' * 64})
    task_invalid = make_execution_cohort_submission(
        manifest=manifest,
        submissions=(invalid_submission, valid_submissions[1]),
    )
    with pytest.raises(ExecutionCohortAggregationError, match='TASK_CONTEXT_MISMATCH'):
        evaluator.score(task_invalid)


def test_terminal_scoring_assigns_fixed_zero_without_partial_metrics() -> None:
    cases = (_case(1, split=Split.TEST), _case(2, split=Split.TEST))
    manifest = _manifest(cases)
    evaluator = ExecutionCohortEvaluator(manifest=manifest, cases=cases)

    missing = evaluator.score_terminal(None)
    malformed = evaluator.score_terminal(b'{not-json')
    incomplete = evaluator.score_terminal(
        make_execution_cohort_submission(
            manifest=manifest,
            submissions=(_submission(cases[0].task),),
        )
    )
    valid = evaluator.score_terminal(
        make_execution_cohort_submission(
            manifest=manifest,
            submissions=(_submission(cases[0].task), _submission(cases[1].task)),
        )
    )

    for failure, code in (
        (missing, ExecutionCohortFailureCode.MISSING_BATCH),
        (malformed, ExecutionCohortFailureCode.MALFORMED_BATCH),
        (incomplete, ExecutionCohortFailureCode.INVALID_OR_INCOMPLETE_BATCH),
    ):
        assert isinstance(failure, ExecutionCohortFailureResult)
        assert failure.failure_code == code
        assert failure.terminal_reward == 0.0
        assert failure.penalized_task_count == failure.task_count == 2
        assert failure.partial_metrics_emitted is False
        assert failure.authenticated_attempt_required_for_admission
        assert not failure.authenticated_attempt_present
        assert not failure.leaderboard_admitted
    assert isinstance(valid, ExecutionCohortResult)


def test_manifest_case_mismatch_and_result_admission_or_count_tampering_are_rejected() -> None:
    cases = (_case(1), _case(2))
    manifest = _manifest(cases)
    with pytest.raises(ExecutionCohortAggregationError, match='do not reconstruct'):
        ExecutionCohortEvaluator(manifest=manifest, cases=(cases[0], _case(3)))

    result = ExecutionCohortEvaluator(manifest=manifest, cases=cases).score(
        make_execution_cohort_submission(
            manifest=manifest,
            submissions=(_submission(cases[0].task), _submission(cases[1].task)),
        )
    )
    payload = result.model_dump(mode='python')
    with pytest.raises(ValidationError):
        ExecutionCohortResult.model_validate({**payload, 'leaderboard_admitted': True})
    with pytest.raises(ValidationError, match='full valid task count'):
        ExecutionCohortResult.model_validate({**payload, 'valid_task_count': 1})
    with pytest.raises(ValidationError, match='component weights'):
        ExecutionCohortResult.model_validate(
            {
                **payload,
                'metrics': {**payload['metrics'], 'mean_core_reward': 0.5},
            }
        )
