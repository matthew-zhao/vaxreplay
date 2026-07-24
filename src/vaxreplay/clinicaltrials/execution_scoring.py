"""Deterministic development scorer for one-trial clinical execution tasks.

Categorical forecasts use the multiclass Brier loss.  Point forecasts use bounded normalized
squared error and quantile forecasts use bounded normalized pinball loss.  A continuous component
is applied only when its private observation state is ``observed_actual``. Inapplicable components
receive the forecast-independent constant 1.0 so weights never depend on the realized outcome.
Coverage remains explicit because episode scalars are comparable only inside a frozen cohort.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Mapping
from typing import Literal, Self, TypeVar

from pydantic import Field, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import ScoreStatus, StrictModel
from vaxreplay.clinicaltrials.execution_schema import ObservationState
from vaxreplay.clinicaltrials.execution_task import (
    EXECUTION_REWARD_VERSION,
    ConditionalContinuousForecast,
    ConditionalPointForecast,
    ContinuousForecastSpec,
    CutoffCitation,
    ExecutionPrivateGold,
    ExecutionSubmission,
    ExecutionTask,
    GoldByteSpan,
    validate_execution_task_gold,
)

EXECUTION_SCORE_SCHEMA_VERSION = 'vaxreplay.clinical-execution-score.dev-v0.1'
REGISTRY_OUTCOME_WEIGHT = 0.40
ENROLLMENT_OBSERVATION_WEIGHT = 0.20
PRIMARY_COMPLETION_OBSERVATION_WEIGHT = 0.20
ENROLLMENT_CONTINUOUS_WEIGHT = 0.10
PRIMARY_COMPLETION_CONTINUOUS_WEIGHT = 0.10
CUTOFF_FACT_WEIGHT = 0.15
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_LabelT = TypeVar('_LabelT')


class ExecutionIssueCode(str, enum.Enum):
    EPISODE_MISMATCH = 'EPISODE_MISMATCH'
    TARGET_TRIAL_MISMATCH = 'TARGET_TRIAL_MISMATCH'
    TASK_CONTEXT_MISMATCH = 'TASK_CONTEXT_MISMATCH'
    CONTINUOUS_FORECAST_FORMAT = 'CONTINUOUS_FORECAST_FORMAT'
    CONTINUOUS_FORECAST_BOUNDS = 'CONTINUOUS_FORECAST_BOUNDS'
    QUANTILE_LEVELS = 'QUANTILE_LEVELS'
    FACT_COVERAGE = 'FACT_COVERAGE'
    FACT_CHOICE = 'FACT_CHOICE'
    UNKNOWN_DOCUMENT = 'UNKNOWN_DOCUMENT'
    DISALLOWED_DOCUMENT = 'DISALLOWED_DOCUMENT'
    INVALID_CITATION_SPAN = 'INVALID_CITATION_SPAN'
    INVALID_CITATION_QUOTE = 'INVALID_CITATION_QUOTE'


class ExecutionValidationIssue(StrictModel):
    code: ExecutionIssueCode
    detail: str = Field(min_length=1)


class ExecutionScore(StrictModel):
    """Auditable scalar and all of its deterministic components."""

    schema_version: Literal['vaxreplay.clinical-execution-score.dev-v0.1'] = EXECUTION_SCORE_SCHEMA_VERSION
    reward_version: Literal['vaxreplay.clinical-execution-reward.dev-v0.1'] = EXECUTION_REWARD_VERSION
    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_key_id: str = Field(pattern=_SHA256_PATTERN)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    per_episode_scalar_subset_selection_robust: Literal[False] = False
    source_derivation_verified: Literal[False] = False
    forecast_spec_preregistered: Literal[False] = False
    status: ScoreStatus
    reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    core_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    registry_outcome_brier: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    registry_outcome_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    enrollment_observation_brier: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    enrollment_observation_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    primary_completion_observation_brier: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    primary_completion_observation_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    enrollment_continuous_applied: bool | None = None
    enrollment_continuous_error: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    enrollment_continuous_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    primary_completion_continuous_applied: bool | None = None
    primary_completion_continuous_error: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    primary_completion_continuous_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    applicable_core_weight: float | None = Field(default=None, ge=0.8, le=1.0, allow_inf_nan=False)
    applicable_component_count: int | None = Field(default=None, ge=3, le=5)
    cutoff_fact_weight: float | None = Field(default=None, ge=0.0, le=CUTOFF_FACT_WEIGHT, allow_inf_nan=False)
    cutoff_fact_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    issues: tuple[ExecutionValidationIssue, ...] = ()

    @model_validator(mode='after')
    def validate_status_and_formula(self) -> Self:
        always_present_metrics = (
            self.reward,
            self.core_reward,
            self.registry_outcome_brier,
            self.registry_outcome_reward,
            self.enrollment_observation_brier,
            self.enrollment_observation_reward,
            self.primary_completion_observation_brier,
            self.primary_completion_observation_reward,
            self.enrollment_continuous_reward,
            self.primary_completion_continuous_reward,
            self.applicable_core_weight,
            self.applicable_component_count,
            self.cutoff_fact_weight,
        )
        branch_metrics = (
            self.enrollment_continuous_applied,
            self.primary_completion_continuous_applied,
        )
        if self.status != ScoreStatus.VALID:
            optional_metrics = (
                *always_present_metrics,
                *branch_metrics,
                self.enrollment_continuous_error,
                self.primary_completion_continuous_error,
                self.cutoff_fact_reward,
            )
            if any(value is not None for value in optional_metrics) or not self.issues:
                raise ValueError('invalid execution scores require issues and cannot contain metrics')
            return self

        if self.issues or any(value is None for value in (*always_present_metrics, *branch_metrics)):
            raise ValueError('valid execution scores require all fixed metrics and no issues')
        assert self.reward is not None
        assert self.core_reward is not None
        assert self.registry_outcome_brier is not None
        assert self.registry_outcome_reward is not None
        assert self.enrollment_observation_brier is not None
        assert self.enrollment_observation_reward is not None
        assert self.primary_completion_observation_brier is not None
        assert self.primary_completion_observation_reward is not None
        assert self.enrollment_continuous_applied is not None
        assert self.enrollment_continuous_reward is not None
        assert self.primary_completion_continuous_applied is not None
        assert self.primary_completion_continuous_reward is not None
        assert self.applicable_core_weight is not None
        assert self.applicable_component_count is not None
        assert self.cutoff_fact_weight is not None

        checks = (
            (self.registry_outcome_reward, 1.0 - self.registry_outcome_brier, 'registry_outcome_reward'),
            (
                self.enrollment_observation_reward,
                1.0 - self.enrollment_observation_brier,
                'enrollment_observation_reward',
            ),
            (
                self.primary_completion_observation_reward,
                1.0 - self.primary_completion_observation_brier,
                'primary_completion_observation_reward',
            ),
        )
        for observed, expected, name in checks:
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'{name} is inconsistent with its Brier loss')

        conditional = (
            (
                self.enrollment_continuous_applied,
                self.enrollment_continuous_error,
                self.enrollment_continuous_reward,
                'enrollment',
            ),
            (
                self.primary_completion_continuous_applied,
                self.primary_completion_continuous_error,
                self.primary_completion_continuous_reward,
                'primary completion',
            ),
        )
        for applied, error, component_reward, name in conditional:
            if applied:
                if (
                    error is None
                    or component_reward is None
                    or not math.isclose(component_reward, 1.0 - error, rel_tol=0.0, abs_tol=1e-12)
                ):
                    raise ValueError(f'{name} continuous reward is inconsistent with its error')
            elif error is not None or not math.isclose(component_reward, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'inapplicable {name} continuous score must use the fixed N/A constant 1')

        expected_weight = math.fsum(
            (
                REGISTRY_OUTCOME_WEIGHT,
                ENROLLMENT_OBSERVATION_WEIGHT,
                PRIMARY_COMPLETION_OBSERVATION_WEIGHT,
                ENROLLMENT_CONTINUOUS_WEIGHT if self.enrollment_continuous_applied else 0.0,
                PRIMARY_COMPLETION_CONTINUOUS_WEIGHT if self.primary_completion_continuous_applied else 0.0,
            )
        )
        expected_count = 3 + int(self.enrollment_continuous_applied) + int(self.primary_completion_continuous_applied)
        if not math.isclose(self.applicable_core_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('applicable_core_weight is inconsistent with the observed branches')
        if self.applicable_component_count != expected_count:
            raise ValueError('applicable_component_count is inconsistent with the observed branches')
        expected_core = math.fsum(
            (
                REGISTRY_OUTCOME_WEIGHT * self.registry_outcome_reward,
                ENROLLMENT_OBSERVATION_WEIGHT * self.enrollment_observation_reward,
                PRIMARY_COMPLETION_OBSERVATION_WEIGHT * self.primary_completion_observation_reward,
                ENROLLMENT_CONTINUOUS_WEIGHT * self.enrollment_continuous_reward,
                PRIMARY_COMPLETION_CONTINUOUS_WEIGHT * self.primary_completion_continuous_reward,
            )
        )
        if not math.isclose(self.core_reward, expected_core, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('core_reward is inconsistent with fixed outcome-independent component weights')
        if math.isclose(self.cutoff_fact_weight, 0.0, rel_tol=0.0, abs_tol=1e-15):
            if self.cutoff_fact_reward is not None:
                raise ValueError('tasks without cutoff facts cannot contain a cutoff fact reward')
            expected_reward = self.core_reward
        else:
            if not math.isclose(self.cutoff_fact_weight, CUTOFF_FACT_WEIGHT, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError('cutoff fact weight is fixed when cutoff facts are present')
            if self.cutoff_fact_reward is None:
                raise ValueError('fact-augmented scores require cutoff_fact_reward')
            expected_reward = (1.0 - CUTOFF_FACT_WEIGHT) * self.core_reward + (
                CUTOFF_FACT_WEIGHT * self.cutoff_fact_reward
            )
        if not math.isclose(self.reward, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('reward is inconsistent with the fixed execution formula')
        return self

    def metrics(self) -> dict[str, float]:
        values = self.model_dump(
            mode='python',
            exclude={
                'schema_version',
                'reward_version',
                'episode_id',
                'target_trial_id',
                'task_context_sha256',
                'private_gold_commitment_sha256',
                'private_gold_commitment_key_id',
                'submission_sha256',
                'development_only',
                'leaderboard_admitted',
                'sealed_execution_supported',
                'identity_contamination_controlled',
                'per_episode_scalar_subset_selection_robust',
                'source_derivation_verified',
                'forecast_spec_preregistered',
                'enrollment_continuous_applied',
                'primary_completion_continuous_applied',
                'status',
                'issues',
            },
        )
        return {
            name: value
            for name, value in values.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        }


def validate_execution_submission(
    task: ExecutionTask,
    submission: ExecutionSubmission,
) -> tuple[ExecutionValidationIssue, ...]:
    """Validate every public, task-dependent constraint without opening private gold."""

    task = ExecutionTask.model_validate_json(canonical_json_bytes(task))
    submission = ExecutionSubmission.model_validate_json(canonical_json_bytes(submission))
    context = task.context
    documents = {item.document_id: item for item in context.cutoff_documents}
    questions = {item.question_id: item for item in context.fact_questions}
    issues: list[ExecutionValidationIssue] = []
    if submission.episode_id != context.episode_id:
        issues.append(
            ExecutionValidationIssue(
                code=ExecutionIssueCode.EPISODE_MISMATCH,
                detail=f'expected episode {context.episode_id}, got {submission.episode_id}',
            )
        )
    if submission.target_trial_id != context.target_trial_id:
        issues.append(
            ExecutionValidationIssue(
                code=ExecutionIssueCode.TARGET_TRIAL_MISMATCH,
                detail=f'expected target {context.target_trial_id}, got {submission.target_trial_id}',
            )
        )
    if submission.task_context_sha256 != task.context_sha256:
        issues.append(
            ExecutionValidationIssue(
                code=ExecutionIssueCode.TASK_CONTEXT_MISMATCH,
                detail='submission is not bound to the exact public task context',
            )
        )
    issues.extend(
        _forecast_issues(
            submission.enrollment_ratio_given_observed_actual,
            context.enrollment_ratio_spec,
            'enrollment ratio',
        )
    )
    issues.extend(
        _forecast_issues(
            submission.primary_completion_slippage_days_given_observed_actual,
            context.primary_completion_slippage_days_spec,
            'primary-completion slippage days',
        )
    )

    fact_ids = tuple(answer.question_id for answer in submission.fact_answers)
    if set(fact_ids) != set(questions):
        issues.append(
            ExecutionValidationIssue(
                code=ExecutionIssueCode.FACT_COVERAGE,
                detail='fact answers must cover every configured cutoff question exactly once',
            )
        )
    for answer in submission.fact_answers:
        question = questions.get(answer.question_id)
        if question is None:
            continue
        if answer.selected_choice not in question.answer_choices:
            issues.append(
                ExecutionValidationIssue(
                    code=ExecutionIssueCode.FACT_CHOICE,
                    detail=f'fact answer {answer.question_id} is outside its public answer choices',
                )
            )
        for citation in answer.citations:
            document = documents.get(citation.document_id)
            if document is None:
                issues.append(
                    ExecutionValidationIssue(
                        code=ExecutionIssueCode.UNKNOWN_DOCUMENT,
                        detail=f'citation references unknown cutoff document {citation.document_id}',
                    )
                )
                continue
            if citation.document_id not in question.allowed_document_ids:
                issues.append(
                    ExecutionValidationIssue(
                        code=ExecutionIssueCode.DISALLOWED_DOCUMENT,
                        detail=f'fact {answer.question_id} cannot cite cutoff document {citation.document_id}',
                    )
                )
                continue
            body = document.body.encode('utf-8')
            if citation.end_byte > len(body):
                issues.append(
                    ExecutionValidationIssue(
                        code=ExecutionIssueCode.INVALID_CITATION_SPAN,
                        detail=f'fact {answer.question_id} citation lies outside {citation.document_id}',
                    )
                )
                continue
            selected = body[citation.start_byte : citation.end_byte]
            try:
                decoded = selected.decode('utf-8')
            except UnicodeDecodeError:
                issues.append(
                    ExecutionValidationIssue(
                        code=ExecutionIssueCode.INVALID_CITATION_SPAN,
                        detail=f'fact {answer.question_id} citation splits a UTF-8 character',
                    )
                )
                continue
            if citation.quote != decoded:
                issues.append(
                    ExecutionValidationIssue(
                        code=ExecutionIssueCode.INVALID_CITATION_QUOTE,
                        detail=f'fact {answer.question_id} quote does not match the cited bytes',
                    )
                )
    return tuple(issues)


class ExecutionSubmissionEvaluator:
    """Private deterministic evaluator with no runner, network, or admission integration."""

    def __init__(
        self,
        *,
        task: ExecutionTask,
        private_gold: ExecutionPrivateGold,
        private_gold_key: bytes,
    ):
        task = ExecutionTask.model_validate_json(canonical_json_bytes(task))
        private_gold = ExecutionPrivateGold.model_validate_json(canonical_json_bytes(private_gold))
        validate_execution_task_gold(task, private_gold, private_gold_key)
        self._task = task
        self._gold = private_gold
        self._documents = {item.document_id: item for item in task.context.cutoff_documents}
        self._questions = {item.question_id: item for item in task.context.fact_questions}
        self._fact_labels = {item.question_id: item for item in private_gold.fact_labels}

    def score(self, submission: ExecutionSubmission) -> ExecutionScore:
        submission = ExecutionSubmission.model_validate_json(canonical_json_bytes(submission))
        submission_sha256 = hashlib.sha256(canonical_json_bytes(submission)).hexdigest()
        issues = self._validate_submission(submission)
        if issues:
            return ExecutionScore(
                episode_id=self._task.context.episode_id,
                target_trial_id=self._task.context.target_trial_id,
                task_context_sha256=self._task.context_sha256,
                private_gold_commitment_sha256=self._task.private_gold_commitment_sha256,
                private_gold_commitment_key_id=self._task.private_gold_commitment_key_id,
                submission_sha256=submission_sha256,
                status=ScoreStatus.INVALID_SCHEMA,
                issues=tuple(issues),
            )

        registry_brier = _multiclass_brier(
            submission.registry_outcome_probabilities.as_mapping(), self._gold.registry_outcome_class
        )
        enrollment_brier = _multiclass_brier(
            submission.enrollment_observation_probabilities.as_mapping(), self._gold.enrollment_observation
        )
        completion_brier = _multiclass_brier(
            submission.primary_completion_observation_probabilities.as_mapping(),
            self._gold.primary_completion_observation,
        )
        enrollment_applied = self._gold.enrollment_observation == ObservationState.OBSERVED_ACTUAL
        completion_applied = self._gold.primary_completion_observation == ObservationState.OBSERVED_ACTUAL
        enrollment_error = (
            _continuous_error(
                submission.enrollment_ratio_given_observed_actual,
                self._task.context.enrollment_ratio_spec,
                self._gold.enrollment_ratio,
            )
            if enrollment_applied
            else None
        )
        completion_error = (
            _continuous_error(
                submission.primary_completion_slippage_days_given_observed_actual,
                self._task.context.primary_completion_slippage_days_spec,
                self._gold.primary_completion_slippage_days,
            )
            if completion_applied
            else None
        )
        enrollment_continuous_reward = 1.0 if enrollment_error is None else 1.0 - enrollment_error
        completion_continuous_reward = 1.0 if completion_error is None else 1.0 - completion_error
        registry_reward = 1.0 - registry_brier
        enrollment_reward = 1.0 - enrollment_brier
        completion_reward = 1.0 - completion_brier
        weighted_components = (
            REGISTRY_OUTCOME_WEIGHT * registry_reward,
            ENROLLMENT_OBSERVATION_WEIGHT * enrollment_reward,
            PRIMARY_COMPLETION_OBSERVATION_WEIGHT * completion_reward,
            ENROLLMENT_CONTINUOUS_WEIGHT * enrollment_continuous_reward,
            PRIMARY_COMPLETION_CONTINUOUS_WEIGHT * completion_continuous_reward,
        )
        applicable_core_weight = math.fsum(
            (REGISTRY_OUTCOME_WEIGHT, ENROLLMENT_OBSERVATION_WEIGHT, PRIMARY_COMPLETION_OBSERVATION_WEIGHT)
        )
        applicable_component_count = 3
        if enrollment_applied:
            applicable_core_weight += ENROLLMENT_CONTINUOUS_WEIGHT
            applicable_component_count += 1
        if completion_applied:
            applicable_core_weight += PRIMARY_COMPLETION_CONTINUOUS_WEIGHT
            applicable_component_count += 1
        core_reward = math.fsum(weighted_components)
        if self._questions:
            fact_weight = CUTOFF_FACT_WEIGHT
            fact_reward = self._score_facts(submission)
            reward = (1.0 - fact_weight) * core_reward + fact_weight * fact_reward
        else:
            fact_weight = 0.0
            fact_reward = None
            reward = core_reward
        return ExecutionScore(
            episode_id=self._task.context.episode_id,
            target_trial_id=self._task.context.target_trial_id,
            task_context_sha256=self._task.context_sha256,
            private_gold_commitment_sha256=self._task.private_gold_commitment_sha256,
            private_gold_commitment_key_id=self._task.private_gold_commitment_key_id,
            submission_sha256=submission_sha256,
            status=ScoreStatus.VALID,
            reward=reward,
            core_reward=core_reward,
            registry_outcome_brier=registry_brier,
            registry_outcome_reward=registry_reward,
            enrollment_observation_brier=enrollment_brier,
            enrollment_observation_reward=enrollment_reward,
            primary_completion_observation_brier=completion_brier,
            primary_completion_observation_reward=completion_reward,
            enrollment_continuous_applied=enrollment_applied,
            enrollment_continuous_error=enrollment_error,
            enrollment_continuous_reward=enrollment_continuous_reward,
            primary_completion_continuous_applied=completion_applied,
            primary_completion_continuous_error=completion_error,
            primary_completion_continuous_reward=completion_continuous_reward,
            applicable_core_weight=applicable_core_weight,
            applicable_component_count=applicable_component_count,
            cutoff_fact_weight=fact_weight,
            cutoff_fact_reward=fact_reward,
        )

    def _validate_submission(self, submission: ExecutionSubmission) -> list[ExecutionValidationIssue]:
        return list(validate_execution_submission(self._task, submission))

    def _citation_issues(self, question_id: str, citation: CutoffCitation) -> list[ExecutionValidationIssue]:
        document = self._documents.get(citation.document_id)
        if document is None:
            return [
                ExecutionValidationIssue(
                    code=ExecutionIssueCode.UNKNOWN_DOCUMENT,
                    detail=f'citation references unknown cutoff document {citation.document_id}',
                )
            ]
        question = self._questions[question_id]
        if citation.document_id not in question.allowed_document_ids:
            return [
                ExecutionValidationIssue(
                    code=ExecutionIssueCode.DISALLOWED_DOCUMENT,
                    detail=f'fact {question_id} cannot cite cutoff document {citation.document_id}',
                )
            ]
        body = document.body.encode('utf-8')
        if citation.end_byte > len(body):
            return [
                ExecutionValidationIssue(
                    code=ExecutionIssueCode.INVALID_CITATION_SPAN,
                    detail=f'fact {question_id} citation lies outside {citation.document_id}',
                )
            ]
        selected = body[citation.start_byte : citation.end_byte]
        try:
            decoded = selected.decode('utf-8')
        except UnicodeDecodeError:
            return [
                ExecutionValidationIssue(
                    code=ExecutionIssueCode.INVALID_CITATION_SPAN,
                    detail=f'fact {question_id} citation splits a UTF-8 character',
                )
            ]
        if citation.quote != decoded:
            return [
                ExecutionValidationIssue(
                    code=ExecutionIssueCode.INVALID_CITATION_QUOTE,
                    detail=f'fact {question_id} quote does not match the cited bytes',
                )
            ]
        return []

    def _score_facts(self, submission: ExecutionSubmission) -> float:
        answers = {answer.question_id: answer for answer in submission.fact_answers}
        utilities: list[float] = []
        for question_id, label in sorted(self._fact_labels.items()):
            answer = answers[question_id]
            accepted = {_span_key(span) for span in label.acceptable_citations}
            citation_precision = sum(_citation_key(item) in accepted for item in answer.citations) / len(
                answer.citations
            )
            utilities.append(float(answer.selected_choice == label.correct_choice) * citation_precision)
        return math.fsum(utilities) / len(utilities)


def _multiclass_brier(
    probabilities: Mapping[_LabelT, float | int],
    observed: _LabelT,
) -> float:
    """Brier loss scaled by 1/2 so any one-hot miss has the maximum loss 1."""

    return 0.5 * math.fsum((value - float(label == observed)) ** 2 for label, value in probabilities.items())


def _forecast_issues(
    forecast: ConditionalContinuousForecast,
    spec: ContinuousForecastSpec,
    name: str,
) -> list[ExecutionValidationIssue]:
    if forecast.kind != spec.forecast_kind:
        return [
            ExecutionValidationIssue(
                code=ExecutionIssueCode.CONTINUOUS_FORECAST_FORMAT,
                detail=f'{name} must use the precommitted {spec.forecast_kind} format',
            )
        ]
    if isinstance(forecast, ConditionalPointForecast):
        values = (forecast.value,)
    else:
        levels = tuple(item.quantile for item in forecast.values)
        if levels != spec.quantile_levels:
            return [
                ExecutionValidationIssue(
                    code=ExecutionIssueCode.QUANTILE_LEVELS,
                    detail=f'{name} must contain exactly the precommitted quantile levels',
                )
            ]
        values = tuple(item.value for item in forecast.values)
    if any(value < spec.lower_bound or value > spec.upper_bound for value in values):
        return [
            ExecutionValidationIssue(
                code=ExecutionIssueCode.CONTINUOUS_FORECAST_BOUNDS,
                detail=f'{name} forecasts must stay inside the public clipping interval',
            )
        ]
    return []


def _continuous_error(
    forecast: ConditionalContinuousForecast,
    spec: ContinuousForecastSpec,
    observed: float | int | None,
) -> float:
    if observed is None:
        raise ValueError('an applied continuous score requires an observed value')
    target = min(spec.upper_bound, max(spec.lower_bound, float(observed)))
    width = spec.upper_bound - spec.lower_bound
    if isinstance(forecast, ConditionalPointForecast):
        return ((forecast.value - target) / width) ** 2
    pinball_losses = []
    for item in forecast.values:
        residual = target - item.value
        pinball_losses.append(
            (item.quantile * residual if residual >= 0.0 else (item.quantile - 1.0) * residual) / width
        )
    mean_quantile = math.fsum(item.quantile for item in forecast.values) / len(forecast.values)
    # This positive scale depends only on the public quantile grid. It maps the worst possible
    # mean pinball loss on the bounded interval to 1 without changing the proper-score optimum.
    maximum_mean_pinball = max(mean_quantile, 1.0 - mean_quantile)
    normalized = (math.fsum(pinball_losses) / len(pinball_losses)) / maximum_mean_pinball
    return min(1.0, max(0.0, normalized))


def _span_key(span: GoldByteSpan) -> tuple[str, int, int]:
    return (span.document_id, span.start_byte, span.end_byte)


def _citation_key(citation: CutoffCitation) -> tuple[str, int, int]:
    return (citation.document_id, citation.start_byte, citation.end_byte)


__all__ = [
    'CUTOFF_FACT_WEIGHT',
    'ENROLLMENT_CONTINUOUS_WEIGHT',
    'ENROLLMENT_OBSERVATION_WEIGHT',
    'EXECUTION_SCORE_SCHEMA_VERSION',
    'PRIMARY_COMPLETION_CONTINUOUS_WEIGHT',
    'PRIMARY_COMPLETION_OBSERVATION_WEIGHT',
    'REGISTRY_OUTCOME_WEIGHT',
    'ExecutionIssueCode',
    'ExecutionScore',
    'ExecutionSubmissionEvaluator',
    'ExecutionValidationIssue',
]
