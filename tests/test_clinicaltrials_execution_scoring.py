from __future__ import annotations

import hashlib
import math
from datetime import date

import pytest
from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import ScoreStatus
from vaxreplay.clinicaltrials.execution_schema import ObservationState, RegistryOutcomeClass
from vaxreplay.clinicaltrials.execution_scoring import (
    ExecutionIssueCode,
    ExecutionScore,
    ExecutionSubmissionEvaluator,
)
from vaxreplay.clinicaltrials.execution_task import (
    ConditionalPointForecast,
    ConditionalQuantileForecast,
    ContinuousForecastSpec,
    CutoffCitation,
    CutoffDocument,
    CutoffFactAnswer,
    CutoffFactQuestion,
    ExecutionPrivateGold,
    ExecutionSubmission,
    ExecutionTask,
    ExecutionTaskContext,
    GoldByteSpan,
    GoldCutoffFact,
    ObservationStateProbabilities,
    QuantilePoint,
    RegistryOutcomeProbabilities,
    build_execution_task,
    execution_task_context_sha256,
    validate_execution_task_gold,
)

_KEY = bytes(range(32))
_BODY = 'Phase: Phase 1. Planned enrollment: 100. Sponsor: Café Labs.'


def _body_sha256(body: str) -> str:
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def _byte_span(text: str, needle: str) -> tuple[int, int]:
    prefix, separator, _ = text.partition(needle)
    assert separator
    start = len(prefix.encode('utf-8'))
    return start, start + len(needle.encode('utf-8'))


def _document(body: str = _BODY) -> CutoffDocument:
    return CutoffDocument(
        document_id='registry-record',
        available_on=date(2020, 3, 2),
        body=body,
        body_sha256=_body_sha256(body),
    )


def _context(*, with_fact: bool = True) -> ExecutionTaskContext:
    documents: tuple[CutoffDocument, ...] = (_document(),) if with_fact else ()
    questions: tuple[CutoffFactQuestion, ...] = (
        (
            CutoffFactQuestion(
                question_id='phase-at-cutoff',
                prompt='Which phase was recorded at the cutoff?',
                answer_choices=('Phase 1', 'Phase 2', 'not reported'),
                allowed_document_ids=('registry-record',),
            ),
        )
        if with_fact
        else ()
    )
    return ExecutionTaskContext(
        episode_id='execution-dev-001',
        target_trial_id='trial-001',
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
            forecast_kind='quantiles',
            lower_bound=-365.0,
            upper_bound=730.0,
            quantile_levels=(0.1, 0.5, 0.9),
        ),
        cutoff_documents=documents,
        fact_questions=questions,
    )


def _gold(
    context: ExecutionTaskContext,
    *,
    registry_outcome_class: RegistryOutcomeClass = RegistryOutcomeClass.COMPLETED,
    enrollment_observation: ObservationState = ObservationState.OBSERVED_ACTUAL,
    enrollment_ratio: float | None = 0.8,
    completion_observation: ObservationState = ObservationState.OBSERVED_ACTUAL,
    slippage_days: int | None = 30,
) -> ExecutionPrivateGold:
    if context.fact_questions:
        start, end = _byte_span(_BODY, 'Phase 1')
        facts = (
            GoldCutoffFact(
                question_id='phase-at-cutoff',
                correct_choice='Phase 1',
                acceptable_citations=(GoldByteSpan(document_id='registry-record', start_byte=start, end_byte=end),),
            ),
        )
    else:
        facts = ()
    return ExecutionPrivateGold(
        episode_id=context.episode_id,
        target_trial_id=context.target_trial_id,
        organizer_private_nct_id='NCT00000001',
        organizer_private_decision_record_sha256='d' * 64,
        task_context_sha256=execution_task_context_sha256(context),
        registry_outcome_class=registry_outcome_class,
        enrollment_observation=enrollment_observation,
        enrollment_ratio=enrollment_ratio,
        primary_completion_observation=completion_observation,
        primary_completion_slippage_days=slippage_days,
        fact_labels=facts,
    )


def _case(
    *,
    with_fact: bool = True,
    registry_outcome_class: RegistryOutcomeClass = RegistryOutcomeClass.COMPLETED,
    enrollment_observation: ObservationState = ObservationState.OBSERVED_ACTUAL,
    enrollment_ratio: float | None = 0.8,
    completion_observation: ObservationState = ObservationState.OBSERVED_ACTUAL,
    slippage_days: int | None = 30,
) -> tuple[ExecutionTask, ExecutionPrivateGold]:
    context = _context(with_fact=with_fact)
    gold = _gold(
        context,
        registry_outcome_class=registry_outcome_class,
        enrollment_observation=enrollment_observation,
        enrollment_ratio=enrollment_ratio,
        completion_observation=completion_observation,
        slippage_days=slippage_days,
    )
    return build_execution_task(context=context, gold=gold, private_gold_key=_KEY), gold


def _one_hot_outcome(value: RegistryOutcomeClass) -> RegistryOutcomeProbabilities:
    values = {item.value: 0.0 for item in RegistryOutcomeClass}
    values[value.value] = 1.0
    return RegistryOutcomeProbabilities.model_validate(values)


def _one_hot_observation(value: ObservationState) -> ObservationStateProbabilities:
    values = {item.value: 0.0 for item in ObservationState}
    values[value.value] = 1.0
    return ObservationStateProbabilities.model_validate(values)


def _citation(needle: str = 'Phase 1') -> CutoffCitation:
    start, end = _byte_span(_BODY, needle)
    return CutoffCitation(document_id='registry-record', start_byte=start, end_byte=end, quote=needle)


def _submission(
    task: ExecutionTask,
    *,
    registry_outcome: RegistryOutcomeClass = RegistryOutcomeClass.COMPLETED,
    enrollment_observation: ObservationState = ObservationState.OBSERVED_ACTUAL,
    completion_observation: ObservationState = ObservationState.OBSERVED_ACTUAL,
    enrollment_point: float = 0.8,
    completion_quantiles: tuple[float, float, float] = (30.0, 30.0, 30.0),
    fact_answers: tuple[CutoffFactAnswer, ...] | None = None,
) -> ExecutionSubmission:
    if fact_answers is None:
        fact_answers = (
            (
                CutoffFactAnswer(
                    question_id='phase-at-cutoff',
                    selected_choice='Phase 1',
                    citations=(_citation(),),
                ),
            )
            if task.context.fact_questions
            else ()
        )
    assert fact_answers is not None
    return ExecutionSubmission(
        episode_id=task.context.episode_id,
        target_trial_id=task.context.target_trial_id,
        task_context_sha256=task.context_sha256,
        registry_outcome_probabilities=_one_hot_outcome(registry_outcome),
        enrollment_observation_probabilities=_one_hot_observation(enrollment_observation),
        primary_completion_observation_probabilities=_one_hot_observation(completion_observation),
        enrollment_ratio_given_observed_actual=ConditionalPointForecast(value=enrollment_point),
        primary_completion_slippage_days_given_observed_actual=ConditionalQuantileForecast(
            values=tuple(
                QuantilePoint(quantile=quantile, value=value)
                for quantile, value in zip((0.1, 0.5, 0.9), completion_quantiles, strict=True)
            )
        ),
        fact_answers=fact_answers,
    )


def _issue_codes(score: ExecutionScore) -> set[ExecutionIssueCode]:
    return {issue.code for issue in score.issues}


def test_perfect_one_trial_score_is_deterministic_hmac_bound_and_explicitly_unadmitted() -> None:
    task, gold = _case()
    evaluator = ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=_KEY)
    submission = _submission(task)

    first = evaluator.score(submission)
    second = evaluator.score(submission)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.status == ScoreStatus.VALID
    assert first.reward == pytest.approx(1.0)
    assert first.core_reward == pytest.approx(1.0)
    assert first.registry_outcome_brier == 0.0
    assert first.enrollment_continuous_applied is True
    assert first.primary_completion_continuous_applied is True
    assert first.applicable_core_weight == pytest.approx(1.0)
    assert first.applicable_component_count == 5
    assert first.cutoff_fact_reward == 1.0
    assert task.private_gold_commitment_scheme == 'hmac-sha256'
    assert task.private_gold_commitment_key_id == hashlib.sha256(_KEY).hexdigest()
    assert task.development_only and not task.leaderboard_admitted and not task.sealed_execution_supported
    assert not task.source_derivation_verified and not task.forecast_spec_preregistered
    assert not task.context.identity_contamination_controlled
    assert not first.identity_contamination_controlled
    assert not first.per_episode_scalar_subset_selection_robust
    assert 'enrollment_continuous_applied' not in first.metrics()
    assert 'primary_completion_continuous_applied' not in first.metrics()


def test_probability_contracts_cover_every_enum_and_reject_malformed_simplexes() -> None:
    assert set(_one_hot_outcome(RegistryOutcomeClass.COMPLETED).as_mapping()) == set(RegistryOutcomeClass)
    assert set(_one_hot_observation(ObservationState.VALUE_MISSING).as_mapping()) == set(ObservationState)

    valid = _one_hot_outcome(RegistryOutcomeClass.COMPLETED).model_dump()
    with pytest.raises(ValidationError, match='sum to 1'):
        RegistryOutcomeProbabilities.model_validate({**valid, 'completed': 0.8})
    with pytest.raises(ValidationError):
        RegistryOutcomeProbabilities.model_validate({key: value for key, value in valid.items() if key != 'suspended'})
    with pytest.raises(ValidationError):
        RegistryOutcomeProbabilities.model_validate({**valid, 'invented': 0.0})
    with pytest.raises(ValidationError):
        RegistryOutcomeProbabilities.model_validate({**valid, 'completed': math.nan})
    with pytest.raises(ValidationError):
        ObservationStateProbabilities.model_validate(
            {
                'observed_actual': math.inf,
                'not_actual': 0.0,
                'value_missing': 0.0,
                'record_missing': 0.0,
            }
        )


def test_multiclass_brier_uses_fixed_outcome_independent_weights() -> None:
    task, gold = _case(with_fact=False)
    score = ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=_KEY).score(
        _submission(task, registry_outcome=RegistryOutcomeClass.TERMINATED)
    )

    assert score.status == ScoreStatus.VALID
    assert score.registry_outcome_brier == pytest.approx(1.0)
    assert score.registry_outcome_reward == pytest.approx(0.0)
    assert score.core_reward == pytest.approx(0.6)
    assert score.reward == pytest.approx(0.6)


def test_inapplicable_continuous_forecasts_are_constant_and_coverage_is_exposed() -> None:
    task, gold = _case(
        with_fact=False,
        registry_outcome_class=RegistryOutcomeClass.RECORD_MISSING,
        enrollment_observation=ObservationState.RECORD_MISSING,
        enrollment_ratio=None,
        completion_observation=ObservationState.RECORD_MISSING,
        slippage_days=None,
    )
    evaluator = ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=_KEY)
    first = evaluator.score(
        _submission(
            task,
            registry_outcome=RegistryOutcomeClass.RECORD_MISSING,
            enrollment_observation=ObservationState.RECORD_MISSING,
            completion_observation=ObservationState.RECORD_MISSING,
            enrollment_point=0.0,
            completion_quantiles=(-365.0, 0.0, 730.0),
        )
    )
    second = evaluator.score(
        _submission(
            task,
            registry_outcome=RegistryOutcomeClass.RECORD_MISSING,
            enrollment_observation=ObservationState.RECORD_MISSING,
            completion_observation=ObservationState.RECORD_MISSING,
            enrollment_point=2.0,
            completion_quantiles=(700.0, 710.0, 720.0),
        )
    )

    assert first.reward == second.reward == 1.0
    assert first.enrollment_continuous_applied is False
    assert first.primary_completion_continuous_applied is False
    assert first.enrollment_continuous_error is None
    assert first.primary_completion_continuous_error is None
    assert first.enrollment_continuous_reward == first.primary_completion_continuous_reward == 1.0
    assert first.applicable_core_weight == pytest.approx(0.8)
    assert first.applicable_component_count == 3


def test_forecast_format_levels_bounds_and_crossing_fail_closed() -> None:
    task, gold = _case()
    evaluator = ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=_KEY)
    base = _submission(task).model_dump()

    wrong_kind = ExecutionSubmission.model_validate(
        {
            **base,
            'enrollment_ratio_given_observed_actual': {
                'kind': 'quantiles',
                'values': ({'quantile': 0.5, 'value': 0.8},),
            },
        }
    )
    assert ExecutionIssueCode.CONTINUOUS_FORECAST_FORMAT in _issue_codes(evaluator.score(wrong_kind))

    wrong_levels = ExecutionSubmission.model_validate(
        {
            **base,
            'primary_completion_slippage_days_given_observed_actual': {
                'kind': 'quantiles',
                'values': (
                    {'quantile': 0.2, 'value': 30.0},
                    {'quantile': 0.5, 'value': 30.0},
                    {'quantile': 0.8, 'value': 30.0},
                ),
            },
        }
    )
    assert ExecutionIssueCode.QUANTILE_LEVELS in _issue_codes(evaluator.score(wrong_levels))

    out_of_bounds = ExecutionSubmission.model_validate(
        {**base, 'enrollment_ratio_given_observed_actual': {'kind': 'point', 'value': 2.1}}
    )
    assert ExecutionIssueCode.CONTINUOUS_FORECAST_BOUNDS in _issue_codes(evaluator.score(out_of_bounds))

    with pytest.raises(ValidationError, match='cannot cross'):
        ConditionalQuantileForecast(
            values=(
                QuantilePoint(quantile=0.1, value=20.0),
                QuantilePoint(quantile=0.5, value=10.0),
            )
        )


def test_quantile_pinball_fixed_scaling_has_unit_worst_case_and_preserves_loss_ratios() -> None:
    worst_task, worst_gold = _case(with_fact=False, slippage_days=-365)
    worst = ExecutionSubmissionEvaluator(task=worst_task, private_gold=worst_gold, private_gold_key=_KEY).score(
        _submission(worst_task, completion_quantiles=(730.0, 730.0, 730.0))
    )
    assert worst.primary_completion_continuous_error == pytest.approx(1.0)
    assert worst.primary_completion_continuous_reward == pytest.approx(0.0)

    task, gold = _case(with_fact=False)
    evaluator = ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=_KEY)
    near = evaluator.score(_submission(task, completion_quantiles=(40.0, 40.0, 40.0)))
    far = evaluator.score(_submission(task, completion_quantiles=(130.0, 130.0, 130.0)))
    assert near.primary_completion_continuous_error is not None
    assert far.primary_completion_continuous_error is not None
    assert far.primary_completion_continuous_error == pytest.approx(10.0 * near.primary_completion_continuous_error)


def test_private_gold_observation_missingness_invariants_are_strict() -> None:
    context = _context(with_fact=False)
    base = _gold(context).model_dump()

    with pytest.raises(ValidationError, match='record-missing outcome'):
        ExecutionPrivateGold.model_validate(
            {
                **base,
                'enrollment_observation': ObservationState.RECORD_MISSING,
                'enrollment_ratio': None,
            }
        )
    with pytest.raises(ValidationError, match='record-missing outcome'):
        ExecutionPrivateGold.model_validate(
            {
                **base,
                'registry_outcome_class': RegistryOutcomeClass.RECORD_MISSING,
                'enrollment_observation': ObservationState.RECORD_MISSING,
                'enrollment_ratio': None,
            }
        )
    with pytest.raises(ValidationError, match='enrollment ratio exists exactly'):
        ExecutionPrivateGold.model_validate(
            {
                **base,
                'enrollment_observation': ObservationState.NOT_ACTUAL,
            }
        )


def test_hmac_rejects_tampered_gold_wrong_key_and_forged_model_copy_context() -> None:
    task, gold = _case()
    wrong_key = b'x' * 32
    with pytest.raises(ValueError, match='key ID'):
        ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=wrong_key)

    tampered = gold.model_copy(update={'registry_outcome_class': RegistryOutcomeClass.TERMINATED})
    with pytest.raises(ValueError, match='HMAC commitment'):
        ExecutionSubmissionEvaluator(task=task, private_gold=tampered, private_gold_key=_KEY)

    forged_context = task.context.model_copy(update={'planned_enrollment': 9_999})
    forged_task = task.model_copy(update={'context': forged_context})
    with pytest.raises(ValidationError, match='context'):
        validate_execution_task_gold(forged_task, gold, _KEY)

    with pytest.raises(ValueError, match='at least 32 bytes'):
        build_execution_task(context=task.context, gold=gold, private_gold_key=b'short')


def test_public_contract_uses_opaque_identity_and_rejects_nct_in_released_text() -> None:
    task, gold = _case()
    public_bytes = canonical_json_bytes(task)

    assert b'NCT00000001' not in public_bytes
    assert task.context.target_trial_id == 'trial-001'
    assert gold.organizer_private_nct_id == 'NCT00000001'
    assert task.context.span_mapped_identity_mask_receipt_present is False
    assert task.context.identity_contamination_controlled is False

    leaking_body = 'This is prefixNCT12345678_suffix.'
    with pytest.raises(ValidationError, match='cannot expose an NCT'):
        _document(leaking_body)
    with pytest.raises(ValidationError, match='cannot expose an NCT'):
        CutoffFactQuestion(
            question_id='leak',
            prompt='What happened in nct12345678?',
            answer_choices=('yes', 'no'),
            allowed_document_ids=('registry-record',),
        )
    context_payload = _context().model_dump()
    with pytest.raises(ValidationError, match='cannot expose an NCT'):
        ExecutionTaskContext.model_validate({**context_payload, 'episode_id': 'NCT12345678'})
    with pytest.raises(ValidationError, match='exactly 48 calendar months'):
        ExecutionTaskContext.model_validate({**context_payload, 'label_archive_date': date(2023, 2, 1)})


def test_fact_citations_are_exact_utf8_spans_and_spam_is_penalized() -> None:
    task, gold = _case()
    evaluator = ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=_KEY)

    fabricated = _citation().model_copy(update={'quote': 'Phase 2'})
    fabricated_score = evaluator.score(
        _submission(
            task,
            fact_answers=(
                CutoffFactAnswer(
                    question_id='phase-at-cutoff',
                    selected_choice='Phase 1',
                    citations=(fabricated,),
                ),
            ),
        )
    )
    assert fabricated_score.status == ScoreStatus.INVALID_SCHEMA
    assert ExecutionIssueCode.INVALID_CITATION_QUOTE in _issue_codes(fabricated_score)

    cafe_start, cafe_end = _byte_span(_BODY, 'é')
    split_utf8 = CutoffCitation(
        document_id='registry-record',
        start_byte=cafe_start + 1,
        end_byte=cafe_end,
        quote='x',
    )
    split_score = evaluator.score(
        _submission(
            task,
            fact_answers=(
                CutoffFactAnswer(
                    question_id='phase-at-cutoff',
                    selected_choice='Phase 1',
                    citations=(split_utf8,),
                ),
            ),
        )
    )
    assert ExecutionIssueCode.INVALID_CITATION_SPAN in _issue_codes(split_score)

    spam_score = evaluator.score(
        _submission(
            task,
            fact_answers=(
                CutoffFactAnswer(
                    question_id='phase-at-cutoff',
                    selected_choice='Phase 1',
                    citations=(_citation('Phase 1'), _citation('100')),
                ),
            ),
        )
    )
    assert spam_score.status == ScoreStatus.VALID
    assert spam_score.cutoff_fact_reward == pytest.approx(0.5)
    assert spam_score.reward == pytest.approx(0.925)


def test_fact_coverage_choices_and_private_spans_are_bound() -> None:
    task, gold = _case()
    evaluator = ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=_KEY)

    missing = evaluator.score(_submission(task, fact_answers=()))
    assert ExecutionIssueCode.FACT_COVERAGE in _issue_codes(missing)

    bad_choice = evaluator.score(
        _submission(
            task,
            fact_answers=(
                CutoffFactAnswer(
                    question_id='phase-at-cutoff',
                    selected_choice='invented',
                    citations=(_citation(),),
                ),
            ),
        )
    )
    assert ExecutionIssueCode.FACT_CHOICE in _issue_codes(bad_choice)

    label = gold.fact_labels[0].model_copy(
        update={
            'acceptable_citations': (
                GoldByteSpan(document_id='registry-record', start_byte=0, end_byte=len(_BODY.encode()) + 1),
            )
        }
    )
    forged_gold = gold.model_copy(update={'fact_labels': (label,)})
    forged_task = ExecutionTask.model_validate(task.model_dump())
    with pytest.raises(ValueError, match='outside its cutoff document'):
        build_execution_task(context=forged_task.context, gold=forged_gold, private_gold_key=_KEY)


def test_score_formula_and_submission_schema_cannot_be_bypassed_with_model_copy() -> None:
    task, gold = _case()
    evaluator = ExecutionSubmissionEvaluator(task=task, private_gold=gold, private_gold_key=_KEY)
    score = evaluator.score(_submission(task))
    score_payload = score.model_dump()
    with pytest.raises(ValidationError, match='reward is inconsistent'):
        ExecutionScore.model_validate({**score_payload, 'reward': 0.5})

    valid_submission = _submission(task)
    forged_probabilities = valid_submission.registry_outcome_probabilities.model_copy(update={'completed': 0.5})
    forged_submission = valid_submission.model_copy(update={'registry_outcome_probabilities': forged_probabilities})
    with pytest.raises(ValidationError):
        evaluator.score(forged_submission)

    submission_payload = _submission(task).model_dump()
    with pytest.raises(ValidationError):
        ExecutionSubmission.model_validate({**submission_payload, 'ranking': ('trial-001',)})
