"""Fail-closed cohort aggregation for development-only clinical execution replay.

The one-task scorer is intentionally not enough for a benchmark result: allowing a caller to
submit or report only favorable episodes would make the scalar easy to game.  This module binds a
fixed organizer-private cohort manifest to every task, final context, private gold record,
lineage, and split.  It emits an aggregate only after receiving exactly one valid raw submission
for every bound task.

Missing, duplicate, extra, malformed, and task-invalid submissions reject the whole batch.  No
partial or valid-only result is emitted.  The resulting metric is an equal-task macro average;
conditional continuous metrics additionally expose their fixed gold-defined applicability count.
All release, sealing, contamination-control, and leaderboard-admission claims remain false.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import ScoreStatus, Split, StrictModel
from vaxreplay.clinicaltrials.execution_schema import EXECUTION_TASK_ID
from vaxreplay.clinicaltrials.execution_scoring import (
    CUTOFF_FACT_WEIGHT,
    ENROLLMENT_CONTINUOUS_WEIGHT,
    ENROLLMENT_OBSERVATION_WEIGHT,
    PRIMARY_COMPLETION_CONTINUOUS_WEIGHT,
    PRIMARY_COMPLETION_OBSERVATION_WEIGHT,
    REGISTRY_OUTCOME_WEIGHT,
    ExecutionScore,
    ExecutionSubmissionEvaluator,
)
from vaxreplay.clinicaltrials.execution_task import (
    EXECUTION_REWARD_VERSION,
    ExecutionPrivateGold,
    ExecutionSubmission,
    ExecutionTask,
    validate_execution_task_gold,
)

EXECUTION_COHORT_AGGREGATION_POLICY_SCHEMA_VERSION = 'vaxreplay.clinical-execution-cohort-aggregation-policy.dev-v0.1'
EXECUTION_COHORT_MANIFEST_SCHEMA_VERSION = 'vaxreplay.clinical-execution-cohort-manifest.dev-v0.1'
EXECUTION_COHORT_SUBMISSION_SCHEMA_VERSION = 'vaxreplay.clinical-execution-cohort-submission.dev-v0.1'
EXECUTION_COHORT_RESULT_SCHEMA_VERSION = 'vaxreplay.clinical-execution-cohort-result.dev-v0.1'
EXECUTION_COHORT_FAILURE_RESULT_SCHEMA_VERSION = 'vaxreplay.clinical-execution-cohort-failure.dev-v0.1'
EXECUTION_COHORT_AGGREGATION_POLICY_ID = 'clinical-execution-exact-coverage-macro-v0.1'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SPLIT_ORDER = (Split.TRAIN, Split.DEV, Split.TEST)
_MAX_COHORT_SUBMISSION_BYTES = 64 * 1024 * 1024


class ExecutionCohortAggregationError(ValueError):
    """A closed cohort could not be scored without violating the frozen policy."""


class ExecutionCohortAggregationPolicy(StrictModel):
    """Preregistered behavior for coverage failures and metric denominators."""

    schema_version: Literal['vaxreplay.clinical-execution-cohort-aggregation-policy.dev-v0.1'] = (
        EXECUTION_COHORT_AGGREGATION_POLICY_SCHEMA_VERSION
    )
    policy_id: Literal['clinical-execution-exact-coverage-macro-v0.1'] = EXECUTION_COHORT_AGGREGATION_POLICY_ID
    task_weighting: Literal['equal_weight_macro_over_one_declared_split'] = 'equal_weight_macro_over_one_declared_split'
    cross_split_scalar_emitted: Literal[False] = False
    official_scalar_scope: Literal['held_out_test_split_only'] = 'held_out_test_split_only'
    conditional_metric_denominator: Literal['gold_defined_applicable_tasks_with_count_and_rate'] = (
        'gold_defined_applicable_tasks_with_count_and_rate'
    )
    missing_submission_policy: Literal['fixed_zero_terminal_failure'] = 'fixed_zero_terminal_failure'
    duplicate_submission_policy: Literal['fixed_zero_terminal_failure'] = 'fixed_zero_terminal_failure'
    extra_submission_policy: Literal['fixed_zero_terminal_failure'] = 'fixed_zero_terminal_failure'
    invalid_submission_policy: Literal['fixed_zero_terminal_failure'] = 'fixed_zero_terminal_failure'
    partial_result_emitted: Literal[False] = False
    valid_only_mean_emitted: Literal[False] = False
    episode_subset_result_emitted: Literal[False] = False
    overall_cohort_metric_required: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False


EXECUTION_COHORT_AGGREGATION_POLICY = ExecutionCohortAggregationPolicy()


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def execution_cohort_aggregation_policy_sha256() -> str:
    return _sha256(EXECUTION_COHORT_AGGREGATION_POLICY)


class ExecutionCohortTaskBinding(StrictModel):
    """Organizer-private exact binding for one task in a frozen cohort."""

    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    split: Split
    public_lineage_id: str = Field(pattern=r'^lineage-[0-9a-f]{20}$')
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_key_id: str = Field(pattern=_SHA256_PATTERN)
    cutoff_facts_configured: bool


class ExecutionCohortSplitCount(StrictModel):
    split: Split
    task_count: int = Field(ge=0)
    lineage_count: int = Field(ge=0)


class ExecutionCohortManifest(StrictModel):
    """Frozen organizer-private cohort; upstream hashes are bindings, not admission proof."""

    schema_version: Literal['vaxreplay.clinical-execution-cohort-manifest.dev-v0.1'] = (
        EXECUTION_COHORT_MANIFEST_SCHEMA_VERSION
    )
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    task_type: Literal['registry_observed_trial_execution'] = EXECUTION_TASK_ID
    reward_version: Literal['vaxreplay.clinical-execution-reward.dev-v0.1'] = EXECUTION_REWARD_VERSION
    aggregation_policy_id: Literal['clinical-execution-exact-coverage-macro-v0.1'] = (
        EXECUTION_COHORT_AGGREGATION_POLICY_ID
    )
    aggregation_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    gold_derivation_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_split: Split
    tasks: tuple[ExecutionCohortTaskBinding, ...] = Field(min_length=1)
    task_count: int = Field(gt=0)
    lineage_count: int = Field(gt=0)
    split_counts: tuple[ExecutionCohortSplitCount, ...] = Field(min_length=3, max_length=3)
    organizer_private: Literal[True] = True
    exact_task_context_and_gold_bindings: Literal[True] = True
    lineage_split_isolated: Literal[True] = True
    external_receipts_authenticated_for_admission: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    source_derivation_verified_for_admission: Literal[False] = False

    @field_validator('tasks')
    @classmethod
    def validate_tasks(cls, value: tuple[ExecutionCohortTaskBinding, ...]) -> tuple[ExecutionCohortTaskBinding, ...]:
        episode_ids = tuple(item.episode_id for item in value)
        if episode_ids != tuple(sorted(set(episode_ids))):
            raise ValueError('cohort task bindings must have unique ascending episode IDs')
        for field_name in ('task_sha256', 'task_context_sha256', 'private_gold_sha256'):
            values = tuple(getattr(item, field_name) for item in value)
            if len(values) != len(set(values)):
                raise ValueError(f'cohort task bindings must have unique {field_name} values')
        return value

    @model_validator(mode='after')
    def validate_manifest(self) -> Self:
        if self.aggregation_policy_sha256 != execution_cohort_aggregation_policy_sha256():
            raise ValueError('cohort manifest does not bind the fixed aggregation policy')
        if self.task_count != len(self.tasks):
            raise ValueError('cohort task_count does not equal its task bindings')
        if {task.split for task in self.tasks} != {self.evaluation_split}:
            raise ValueError('one cohort manifest may aggregate exactly one declared split')
        if len({task.cutoff_facts_configured for task in self.tasks}) != 1:
            raise ValueError('mixed fact/non-fact task cohorts are unsupported')

        lineage_splits: dict[str, set[Split]] = {}
        for task in self.tasks:
            lineage_splits.setdefault(task.public_lineage_id, set()).add(task.split)
        if any(len(splits) != 1 for splits in lineage_splits.values()):
            raise ValueError('one public lineage cannot cross cohort splits')
        if self.lineage_count != len(lineage_splits):
            raise ValueError('cohort lineage_count is inconsistent with task bindings')
        if tuple(item.split for item in self.split_counts) != _SPLIT_ORDER:
            raise ValueError('cohort split counts must be ordered train, dev, test')
        for split_count in self.split_counts:
            expected_tasks = sum(task.split == split_count.split for task in self.tasks)
            expected_lineages = sum(splits == {split_count.split} for splits in lineage_splits.values())
            if (split_count.task_count, split_count.lineage_count) != (expected_tasks, expected_lineages):
                raise ValueError('cohort split counts are inconsistent with task bindings')
        return self


def execution_cohort_manifest_sha256(manifest: ExecutionCohortManifest) -> str:
    validated = ExecutionCohortManifest.model_validate_json(canonical_json_bytes(manifest))
    return _sha256(validated)


class ExecutionCohortSubmission(StrictModel):
    """One canonical batch containing every raw per-task submission exactly once."""

    schema_version: Literal['vaxreplay.clinical-execution-cohort-submission.dev-v0.1'] = (
        EXECUTION_COHORT_SUBMISSION_SCHEMA_VERSION
    )
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    cohort_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    submissions: tuple[ExecutionSubmission, ...] = Field(min_length=1)

    @field_validator('submissions')
    @classmethod
    def validate_submissions(cls, value: tuple[ExecutionSubmission, ...]) -> tuple[ExecutionSubmission, ...]:
        episode_ids = tuple(item.episode_id for item in value)
        if episode_ids != tuple(sorted(set(episode_ids))):
            raise ValueError('cohort submissions must have unique ascending episode IDs')
        return value


def make_execution_cohort_submission(
    *,
    manifest: ExecutionCohortManifest,
    submissions: Iterable[ExecutionSubmission],
) -> ExecutionCohortSubmission:
    """Canonicalize participant responses without weakening exact-coverage checks."""

    manifest = ExecutionCohortManifest.model_validate_json(canonical_json_bytes(manifest))
    ordered = tuple(sorted(submissions, key=lambda item: item.episode_id))
    return ExecutionCohortSubmission(
        cohort_id=manifest.cohort_id,
        cohort_manifest_sha256=execution_cohort_manifest_sha256(manifest),
        submissions=ordered,
    )


class ExecutionConditionalMetricSummary(StrictModel):
    """A conditional component with its immutable gold-defined denominator."""

    task_count: int = Field(gt=0)
    applied_task_count: int = Field(ge=0)
    applied_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_error_when_applied: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    mean_reward_when_applied: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    mean_fixed_reward_all_tasks: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_summary(self) -> Self:
        if self.applied_task_count > self.task_count:
            raise ValueError('conditional applied count cannot exceed task_count')
        expected_rate = self.applied_task_count / self.task_count
        if not math.isclose(self.applied_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('conditional applied_rate is inconsistent with counts')
        if self.applied_task_count == 0:
            if self.mean_error_when_applied is not None or self.mean_reward_when_applied is not None:
                raise ValueError('an inapplicable conditional component cannot contain applied-only means')
            expected_all_task_reward = 1.0
        else:
            if self.mean_error_when_applied is None or self.mean_reward_when_applied is None:
                raise ValueError('an applicable conditional component requires both applied-only means')
            if not math.isclose(
                self.mean_reward_when_applied,
                1.0 - self.mean_error_when_applied,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError('conditional applied reward is inconsistent with its error')
            expected_all_task_reward = (
                self.applied_task_count * self.mean_reward_when_applied + (self.task_count - self.applied_task_count)
            ) / self.task_count
        if not math.isclose(
            self.mean_fixed_reward_all_tasks,
            expected_all_task_reward,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('conditional all-task reward is inconsistent with fixed N/A rewards')
        return self


class ExecutionFactMetricSummary(StrictModel):
    """Fact metric coverage; no mean exists when no manifest task asks facts."""

    task_count: int = Field(gt=0)
    configured_task_count: int = Field(ge=0)
    configured_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_reward_when_configured: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_summary(self) -> Self:
        if self.configured_task_count > self.task_count:
            raise ValueError('fact-task count cannot exceed task_count')
        expected_rate = self.configured_task_count / self.task_count
        if not math.isclose(self.configured_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('fact configured_rate is inconsistent with counts')
        if (self.mean_reward_when_configured is not None) != (self.configured_task_count > 0):
            raise ValueError('fact reward exists exactly when at least one task configures facts')
        return self


class ExecutionCohortMetrics(StrictModel):
    """Explicit equal-task component means for one complete cohort."""

    task_count: int = Field(gt=0)
    mean_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_core_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_registry_outcome_brier: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_registry_outcome_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_enrollment_observation_brier: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_enrollment_observation_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_primary_completion_observation_brier: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_primary_completion_observation_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    enrollment_continuous: ExecutionConditionalMetricSummary
    primary_completion_continuous: ExecutionConditionalMetricSummary
    mean_applicable_core_weight: float = Field(ge=0.8, le=1.0, allow_inf_nan=False)
    mean_applicable_component_count: float = Field(ge=3.0, le=5.0, allow_inf_nan=False)
    cutoff_facts: ExecutionFactMetricSummary

    @model_validator(mode='after')
    def validate_metrics(self) -> Self:
        if (
            self.enrollment_continuous.task_count != self.task_count
            or self.primary_completion_continuous.task_count != self.task_count
            or self.cutoff_facts.task_count != self.task_count
        ):
            raise ValueError('component summaries must use the full cohort task_count')
        complements = (
            (self.mean_registry_outcome_brier, self.mean_registry_outcome_reward, 'registry outcome'),
            (
                self.mean_enrollment_observation_brier,
                self.mean_enrollment_observation_reward,
                'enrollment observation',
            ),
            (
                self.mean_primary_completion_observation_brier,
                self.mean_primary_completion_observation_reward,
                'primary-completion observation',
            ),
        )
        for loss, reward, name in complements:
            if not math.isclose(reward, 1.0 - loss, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'mean {name} reward is inconsistent with its mean Brier loss')
        expected_core = math.fsum(
            (
                REGISTRY_OUTCOME_WEIGHT * self.mean_registry_outcome_reward,
                ENROLLMENT_OBSERVATION_WEIGHT * self.mean_enrollment_observation_reward,
                PRIMARY_COMPLETION_OBSERVATION_WEIGHT * self.mean_primary_completion_observation_reward,
                ENROLLMENT_CONTINUOUS_WEIGHT * self.enrollment_continuous.mean_fixed_reward_all_tasks,
                PRIMARY_COMPLETION_CONTINUOUS_WEIGHT * self.primary_completion_continuous.mean_fixed_reward_all_tasks,
            )
        )
        if not math.isclose(self.mean_core_reward, expected_core, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('mean core reward is inconsistent with fixed component weights')
        expected_weight = (
            0.8
            + ENROLLMENT_CONTINUOUS_WEIGHT * self.enrollment_continuous.applied_rate
            + (PRIMARY_COMPLETION_CONTINUOUS_WEIGHT * self.primary_completion_continuous.applied_rate)
        )
        expected_component_count = (
            3.0 + self.enrollment_continuous.applied_rate + self.primary_completion_continuous.applied_rate
        )
        if not math.isclose(self.mean_applicable_core_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('mean applicable core weight is inconsistent with applicability rates')
        if not math.isclose(
            self.mean_applicable_component_count,
            expected_component_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('mean applicable component count is inconsistent with applicability rates')
        if self.cutoff_facts.configured_task_count == 0 and not math.isclose(
            self.mean_reward, self.mean_core_reward, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError('a cohort without fact questions must have reward equal to core reward')
        if self.cutoff_facts.configured_task_count == self.task_count:
            assert self.cutoff_facts.mean_reward_when_configured is not None
            expected_reward = (1.0 - CUTOFF_FACT_WEIGHT) * self.mean_core_reward + (
                CUTOFF_FACT_WEIGHT * self.cutoff_facts.mean_reward_when_configured
            )
            if not math.isclose(self.mean_reward, expected_reward, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError('all-fact cohort mean reward is inconsistent with the fixed fact weight')
        return self


class ExecutionCohortResult(StrictModel):
    """Only produced for a complete valid batch; per-episode scores are not included."""

    schema_version: Literal['vaxreplay.clinical-execution-cohort-result.dev-v0.1'] = (
        EXECUTION_COHORT_RESULT_SCHEMA_VERSION
    )
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    task_type: Literal['registry_observed_trial_execution'] = EXECUTION_TASK_ID
    reward_version: Literal['vaxreplay.clinical-execution-reward.dev-v0.1'] = EXECUTION_REWARD_VERSION
    aggregation_policy_id: Literal['clinical-execution-exact-coverage-macro-v0.1'] = (
        EXECUTION_COHORT_AGGREGATION_POLICY_ID
    )
    aggregation_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    cohort_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    input_submissions_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_episode_scores_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_split: Split
    task_count: int = Field(gt=0)
    valid_task_count: int = Field(gt=0)
    invalid_task_count: Literal[0] = 0
    lineage_count: int = Field(gt=0)
    split_counts: tuple[ExecutionCohortSplitCount, ...] = Field(min_length=3, max_length=3)
    metrics: ExecutionCohortMetrics
    admission_status: Literal['development_only_not_admitted'] = 'development_only_not_admitted'
    full_manifest_coverage_verified: Literal[True] = True
    exactly_one_submission_per_task_verified: Literal[True] = True
    every_task_score_valid: Literal[True] = True
    exact_task_context_and_gold_bindings_verified: Literal[True] = True
    aggregation_rejects_episode_subset_input: Literal[True] = True
    participant_visible_episode_scores_included: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    source_derivation_verified_for_admission: Literal[False] = False

    @model_validator(mode='after')
    def validate_result(self) -> Self:
        if self.aggregation_policy_sha256 != execution_cohort_aggregation_policy_sha256():
            raise ValueError('cohort result does not bind the fixed aggregation policy')
        if self.valid_task_count != self.task_count or self.metrics.task_count != self.task_count:
            raise ValueError('cohort result must contain exactly the full valid task count')
        if tuple(item.split for item in self.split_counts) != _SPLIT_ORDER:
            raise ValueError('cohort result split counts must be ordered train, dev, test')
        if sum(item.task_count for item in self.split_counts) != self.task_count:
            raise ValueError('cohort result split task counts must sum to task_count')
        if sum(item.lineage_count for item in self.split_counts) != self.lineage_count:
            raise ValueError('cohort result split lineage counts must sum to lineage_count')
        selected = next(item for item in self.split_counts if item.split == self.evaluation_split)
        if selected.task_count != self.task_count or any(
            item.task_count != 0 for item in self.split_counts if item.split != self.evaluation_split
        ):
            raise ValueError('cohort result may contain exactly one evaluation split')
        return self


class ExecutionCohortFailureCode(str, enum.Enum):
    MISSING_BATCH = 'missing_batch'
    MALFORMED_BATCH = 'malformed_batch'
    INVALID_OR_INCOMPLETE_BATCH = 'invalid_or_incomplete_batch'


class ExecutionCohortFailureResult(StrictModel):
    """Fixed terminal penalty for a participant-caused incomplete or invalid cohort attempt."""

    schema_version: Literal['vaxreplay.clinical-execution-cohort-failure.dev-v0.1'] = (
        EXECUTION_COHORT_FAILURE_RESULT_SCHEMA_VERSION
    )
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    cohort_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    aggregation_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    input_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_split: Split
    task_count: int = Field(gt=0)
    penalized_task_count: int = Field(gt=0)
    failure_code: ExecutionCohortFailureCode
    terminal_reward: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)
    terminal: Literal[True] = True
    partial_metrics_emitted: Literal[False] = False
    authenticated_attempt_required_for_admission: Literal[True] = True
    authenticated_attempt_present: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False

    @model_validator(mode='after')
    def validate_failure(self) -> Self:
        if self.aggregation_policy_sha256 != execution_cohort_aggregation_policy_sha256():
            raise ValueError('cohort failure does not bind the fixed aggregation policy')
        if self.penalized_task_count != self.task_count:
            raise ValueError('a terminal cohort failure must penalize every task in the split')
        return self


type ExecutionCohortTerminalResult = ExecutionCohortResult | ExecutionCohortFailureResult


@dataclass(frozen=True, slots=True)
class ExecutionCohortEvaluationCase:
    """Private evaluator input.  The HMAC key is validated but never copied into a manifest."""

    task: ExecutionTask
    private_gold: ExecutionPrivateGold
    private_gold_key: bytes
    split: Split
    public_lineage_id: str


def _validate_case(case: ExecutionCohortEvaluationCase) -> ExecutionCohortEvaluationCase:
    task = ExecutionTask.model_validate_json(canonical_json_bytes(case.task))
    gold = ExecutionPrivateGold.model_validate_json(canonical_json_bytes(case.private_gold))
    if not isinstance(case.private_gold_key, bytes):
        raise ExecutionCohortAggregationError('private gold key must be bytes')
    validate_execution_task_gold(task, gold, case.private_gold_key)
    # Reuse the strict binding schema to validate split and public-lineage syntax now.
    ExecutionCohortTaskBinding(
        episode_id=task.context.episode_id,
        target_trial_id=task.context.target_trial_id,
        split=case.split,
        public_lineage_id=case.public_lineage_id,
        task_sha256=_sha256(task),
        task_context_sha256=task.context_sha256,
        private_gold_sha256=_sha256(gold),
        private_gold_commitment_sha256=task.private_gold_commitment_sha256,
        private_gold_commitment_key_id=task.private_gold_commitment_key_id,
        cutoff_facts_configured=bool(task.context.fact_questions),
    )
    return ExecutionCohortEvaluationCase(
        task=task,
        private_gold=gold,
        private_gold_key=case.private_gold_key,
        split=case.split,
        public_lineage_id=case.public_lineage_id,
    )


def _binding(case: ExecutionCohortEvaluationCase) -> ExecutionCohortTaskBinding:
    return ExecutionCohortTaskBinding(
        episode_id=case.task.context.episode_id,
        target_trial_id=case.task.context.target_trial_id,
        split=case.split,
        public_lineage_id=case.public_lineage_id,
        task_sha256=_sha256(case.task),
        task_context_sha256=case.task.context_sha256,
        private_gold_sha256=_sha256(case.private_gold),
        private_gold_commitment_sha256=case.task.private_gold_commitment_sha256,
        private_gold_commitment_key_id=case.task.private_gold_commitment_key_id,
        cutoff_facts_configured=bool(case.task.context.fact_questions),
    )


def make_execution_cohort_manifest(
    *,
    cohort_id: str,
    cases: Iterable[ExecutionCohortEvaluationCase],
    lineage_split_manifest_sha256: str,
    workspace_build_receipt_sha256: str,
    gold_derivation_receipt_sha256: str,
) -> ExecutionCohortManifest:
    """Validate all private bindings and freeze a deterministic cohort manifest."""

    validated_cases = tuple(_validate_case(case) for case in cases)
    if not validated_cases:
        raise ExecutionCohortAggregationError('cannot create an empty execution cohort manifest')
    tasks = tuple(sorted((_binding(case) for case in validated_cases), key=lambda item: item.episode_id))
    lineage_splits: dict[str, set[Split]] = {}
    for task in tasks:
        lineage_splits.setdefault(task.public_lineage_id, set()).add(task.split)
    split_counts = tuple(
        ExecutionCohortSplitCount(
            split=split,
            task_count=sum(task.split == split for task in tasks),
            lineage_count=sum(splits == {split} for splits in lineage_splits.values()),
        )
        for split in _SPLIT_ORDER
    )
    observed_splits = {task.split for task in tasks}
    if len(observed_splits) != 1:
        raise ExecutionCohortAggregationError('one cohort manifest may aggregate exactly one split')
    return ExecutionCohortManifest(
        cohort_id=cohort_id,
        aggregation_policy_sha256=execution_cohort_aggregation_policy_sha256(),
        lineage_split_manifest_sha256=lineage_split_manifest_sha256,
        workspace_build_receipt_sha256=workspace_build_receipt_sha256,
        gold_derivation_receipt_sha256=gold_derivation_receipt_sha256,
        evaluation_split=next(iter(observed_splits)),
        tasks=tasks,
        task_count=len(tasks),
        lineage_count=len(lineage_splits),
        split_counts=split_counts,
    )


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ExecutionCohortAggregationError('cannot average an empty metric collection')
    return math.fsum(materialized) / len(materialized)


def _required_metric(scores: tuple[ExecutionScore, ...], field_name: str) -> tuple[float, ...]:
    values = tuple(getattr(score, field_name) for score in scores)
    if any(value is None or isinstance(value, bool) for value in values):
        raise ExecutionCohortAggregationError(f'valid score is missing required metric {field_name}')
    return tuple(float(value) for value in values)


def _conditional_summary(
    scores: tuple[ExecutionScore, ...],
    *,
    applied_field: str,
    error_field: str,
    reward_field: str,
) -> ExecutionConditionalMetricSummary:
    applied_scores = tuple(score for score in scores if getattr(score, applied_field) is True)
    applied_errors = _required_metric(applied_scores, error_field) if applied_scores else ()
    applied_rewards = _required_metric(applied_scores, reward_field) if applied_scores else ()
    all_rewards = _required_metric(scores, reward_field)
    return ExecutionConditionalMetricSummary(
        task_count=len(scores),
        applied_task_count=len(applied_scores),
        applied_rate=len(applied_scores) / len(scores),
        mean_error_when_applied=_mean(applied_errors) if applied_errors else None,
        mean_reward_when_applied=_mean(applied_rewards) if applied_rewards else None,
        mean_fixed_reward_all_tasks=_mean(all_rewards),
    )


def _aggregate_metrics(scores: tuple[ExecutionScore, ...]) -> ExecutionCohortMetrics:
    if not scores:
        raise ExecutionCohortAggregationError('cannot aggregate an empty execution score set')
    if any(score.status != ScoreStatus.VALID for score in scores):
        raise ExecutionCohortAggregationError('cohort aggregation accepts only valid task scores')
    enrollment = _conditional_summary(
        scores,
        applied_field='enrollment_continuous_applied',
        error_field='enrollment_continuous_error',
        reward_field='enrollment_continuous_reward',
    )
    completion = _conditional_summary(
        scores,
        applied_field='primary_completion_continuous_applied',
        error_field='primary_completion_continuous_error',
        reward_field='primary_completion_continuous_reward',
    )
    fact_scores = tuple(score for score in scores if (score.cutoff_fact_weight or 0.0) > 0.0)
    fact_rewards = _required_metric(fact_scores, 'cutoff_fact_reward') if fact_scores else ()
    return ExecutionCohortMetrics(
        task_count=len(scores),
        mean_reward=_mean(_required_metric(scores, 'reward')),
        mean_core_reward=_mean(_required_metric(scores, 'core_reward')),
        mean_registry_outcome_brier=_mean(_required_metric(scores, 'registry_outcome_brier')),
        mean_registry_outcome_reward=_mean(_required_metric(scores, 'registry_outcome_reward')),
        mean_enrollment_observation_brier=_mean(_required_metric(scores, 'enrollment_observation_brier')),
        mean_enrollment_observation_reward=_mean(_required_metric(scores, 'enrollment_observation_reward')),
        mean_primary_completion_observation_brier=_mean(
            _required_metric(scores, 'primary_completion_observation_brier')
        ),
        mean_primary_completion_observation_reward=_mean(
            _required_metric(scores, 'primary_completion_observation_reward')
        ),
        enrollment_continuous=enrollment,
        primary_completion_continuous=completion,
        mean_applicable_core_weight=_mean(_required_metric(scores, 'applicable_core_weight')),
        mean_applicable_component_count=_mean(_required_metric(scores, 'applicable_component_count')),
        cutoff_facts=ExecutionFactMetricSummary(
            task_count=len(scores),
            configured_task_count=len(fact_scores),
            configured_rate=len(fact_scores) / len(scores),
            mean_reward_when_configured=_mean(fact_rewards) if fact_rewards else None,
        ),
    )


class ExecutionCohortEvaluator:
    """Private evaluator that scores raw responses before producing one closed aggregate."""

    def __init__(self, *, manifest: ExecutionCohortManifest, cases: Iterable[ExecutionCohortEvaluationCase]):
        self._manifest = ExecutionCohortManifest.model_validate_json(canonical_json_bytes(manifest))
        validated_cases = tuple(_validate_case(case) for case in cases)
        rebuilt = make_execution_cohort_manifest(
            cohort_id=self._manifest.cohort_id,
            cases=validated_cases,
            lineage_split_manifest_sha256=self._manifest.lineage_split_manifest_sha256,
            workspace_build_receipt_sha256=self._manifest.workspace_build_receipt_sha256,
            gold_derivation_receipt_sha256=self._manifest.gold_derivation_receipt_sha256,
        )
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(self._manifest):
            raise ExecutionCohortAggregationError(
                'private evaluation cases do not reconstruct the exact cohort manifest'
            )
        self._case_by_episode = {case.task.context.episode_id: case for case in validated_cases}
        self._binding_by_episode = {item.episode_id: item for item in self._manifest.tasks}

    def _terminal_failure(
        self,
        *,
        input_payload: bytes,
        code: ExecutionCohortFailureCode,
    ) -> ExecutionCohortFailureResult:
        return ExecutionCohortFailureResult(
            cohort_id=self._manifest.cohort_id,
            cohort_manifest_sha256=execution_cohort_manifest_sha256(self._manifest),
            aggregation_policy_sha256=execution_cohort_aggregation_policy_sha256(),
            input_sha256=hashlib.sha256(input_payload).hexdigest(),
            evaluation_split=self._manifest.evaluation_split,
            task_count=self._manifest.task_count,
            penalized_task_count=self._manifest.task_count,
            failure_code=code,
        )

    def score_terminal(
        self,
        submission: ExecutionCohortSubmission | bytes | None,
    ) -> ExecutionCohortTerminalResult:
        """Always return one terminal development result; invalid attempts receive fixed zero.

        Admission still requires an outer authenticated one-attempt registry to prove that this
        terminal result was retained rather than retried or discarded.
        """

        if submission is None:
            return self._terminal_failure(input_payload=b'', code=ExecutionCohortFailureCode.MISSING_BATCH)
        if isinstance(submission, bytes):
            payload = submission
            if not payload:
                return self._terminal_failure(input_payload=payload, code=ExecutionCohortFailureCode.MISSING_BATCH)
            if len(payload) > _MAX_COHORT_SUBMISSION_BYTES:
                return self._terminal_failure(
                    input_payload=payload,
                    code=ExecutionCohortFailureCode.MALFORMED_BATCH,
                )
            try:
                parsed = ExecutionCohortSubmission.model_validate_json(payload)
            except ValueError:
                return self._terminal_failure(
                    input_payload=payload,
                    code=ExecutionCohortFailureCode.MALFORMED_BATCH,
                )
            if canonical_json_bytes(parsed) != payload:
                return self._terminal_failure(
                    input_payload=payload,
                    code=ExecutionCohortFailureCode.MALFORMED_BATCH,
                )
        else:
            try:
                parsed = ExecutionCohortSubmission.model_validate_json(canonical_json_bytes(submission))
            except ValueError:
                payload = canonical_json_bytes(submission)
                return self._terminal_failure(
                    input_payload=payload,
                    code=ExecutionCohortFailureCode.MALFORMED_BATCH,
                )
            payload = canonical_json_bytes(parsed)
        try:
            return self.score(parsed)
        except ExecutionCohortAggregationError:
            return self._terminal_failure(
                input_payload=payload,
                code=ExecutionCohortFailureCode.INVALID_OR_INCOMPLETE_BATCH,
            )

    def score(self, submission: ExecutionCohortSubmission) -> ExecutionCohortResult:
        """Return one full-cohort result or fail before emitting any partial aggregate."""

        submission = ExecutionCohortSubmission.model_validate_json(canonical_json_bytes(submission))
        expected_manifest_sha256 = execution_cohort_manifest_sha256(self._manifest)
        if submission.cohort_id != self._manifest.cohort_id:
            raise ExecutionCohortAggregationError('cohort submission uses the wrong cohort_id')
        if submission.cohort_manifest_sha256 != expected_manifest_sha256:
            raise ExecutionCohortAggregationError('cohort submission is not bound to the exact manifest')
        expected_ids = tuple(item.episode_id for item in self._manifest.tasks)
        observed_ids = tuple(item.episode_id for item in submission.submissions)
        if observed_ids != expected_ids:
            missing = sorted(set(expected_ids) - set(observed_ids))
            extra = sorted(set(observed_ids) - set(expected_ids))
            raise ExecutionCohortAggregationError(
                f'cohort submission must cover every task exactly once; missing={missing}, extra={extra}'
            )

        scores: list[ExecutionScore] = []
        for raw_submission in submission.submissions:
            case = self._case_by_episode[raw_submission.episode_id]
            score = ExecutionSubmissionEvaluator(
                task=case.task,
                private_gold=case.private_gold,
                private_gold_key=case.private_gold_key,
            ).score(raw_submission)
            if score.status != ScoreStatus.VALID:
                issue_codes = [issue.code.value for issue in score.issues]
                raise ExecutionCohortAggregationError(
                    f'cohort task {raw_submission.episode_id} is invalid; issues={issue_codes}'
                )
            binding = self._binding_by_episode[raw_submission.episode_id]
            observed_binding = (
                score.episode_id,
                score.target_trial_id,
                score.task_context_sha256,
                score.private_gold_commitment_sha256,
                score.private_gold_commitment_key_id,
            )
            expected_binding = (
                binding.episode_id,
                binding.target_trial_id,
                binding.task_context_sha256,
                binding.private_gold_commitment_sha256,
                binding.private_gold_commitment_key_id,
            )
            if observed_binding != expected_binding:
                raise ExecutionCohortAggregationError('private task score escaped its exact manifest binding')
            scores.append(score)

        ordered_scores = tuple(scores)
        return ExecutionCohortResult(
            cohort_id=self._manifest.cohort_id,
            aggregation_policy_sha256=execution_cohort_aggregation_policy_sha256(),
            cohort_manifest_sha256=expected_manifest_sha256,
            input_submissions_sha256=_sha256(submission),
            private_episode_scores_sha256=_sha256([score.model_dump(mode='json') for score in ordered_scores]),
            evaluation_split=self._manifest.evaluation_split,
            task_count=self._manifest.task_count,
            valid_task_count=len(ordered_scores),
            lineage_count=self._manifest.lineage_count,
            split_counts=self._manifest.split_counts,
            metrics=_aggregate_metrics(ordered_scores),
        )


__all__ = [
    'EXECUTION_COHORT_AGGREGATION_POLICY',
    'EXECUTION_COHORT_AGGREGATION_POLICY_ID',
    'EXECUTION_COHORT_AGGREGATION_POLICY_SCHEMA_VERSION',
    'EXECUTION_COHORT_MANIFEST_SCHEMA_VERSION',
    'EXECUTION_COHORT_RESULT_SCHEMA_VERSION',
    'EXECUTION_COHORT_SUBMISSION_SCHEMA_VERSION',
    'ExecutionCohortAggregationError',
    'ExecutionCohortAggregationPolicy',
    'ExecutionCohortEvaluationCase',
    'ExecutionCohortEvaluator',
    'ExecutionCohortFailureCode',
    'ExecutionCohortFailureResult',
    'ExecutionCohortManifest',
    'ExecutionCohortMetrics',
    'ExecutionCohortResult',
    'ExecutionCohortSplitCount',
    'ExecutionCohortSubmission',
    'ExecutionCohortTaskBinding',
    'ExecutionCohortTerminalResult',
    'ExecutionConditionalMetricSummary',
    'ExecutionFactMetricSummary',
    'execution_cohort_aggregation_policy_sha256',
    'execution_cohort_manifest_sha256',
    'make_execution_cohort_manifest',
    'make_execution_cohort_submission',
]
