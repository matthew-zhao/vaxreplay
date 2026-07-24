"""Development-only task contracts for registry-observed trial-execution replay.

This module is intentionally independent of candidate-ranking submissions and of the agentic
workspace/RPC stack.  It describes one forecast target (one historical trial) and nothing here
claims that the task is sealed, admitted, or suitable for an official leaderboard.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from datetime import date
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_schema import (
    EXECUTION_TASK_ID,
    EXECUTION_TASK_SEMANTICS,
    LABEL_HORIZON_MONTHS,
    ExecutionTaskSemantics,
    ObservationState,
    RegistryOutcomeClass,
    add_calendar_months,
)

EXECUTION_TASK_CONTEXT_SCHEMA_VERSION = 'vaxreplay.clinical-execution-task-context.dev-v0.1'
EXECUTION_TASK_SCHEMA_VERSION = 'vaxreplay.clinical-execution-task.dev-v0.1'
EXECUTION_SUBMISSION_SCHEMA_VERSION = 'vaxreplay.clinical-execution-submission.dev-v0.1'
EXECUTION_PRIVATE_GOLD_SCHEMA_VERSION = 'vaxreplay.clinical-execution-private-gold.dev-v0.1'
EXECUTION_REWARD_VERSION = 'vaxreplay.clinical-execution-reward.dev-v0.1'
MAX_CITATIONS_PER_FACT = 3
MAX_CITATION_BYTES = 1_024
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_PRIVATE_GOLD_HMAC_DOMAIN = b'vaxreplay.clinical-execution-private-gold.dev-v0.1\x00'
_NCT_IDENTIFIER = re.compile(r'NCT\d{8}', re.IGNORECASE)


class RegistryOutcomeProbabilities(StrictModel):
    """A complete probability simplex over every explicit registry outcome class."""

    completed: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    terminated: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    withdrawn: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    suspended: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    non_terminal: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    status_missing: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    record_missing: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_simplex(self) -> Self:
        if not math.isclose(math.fsum(self.as_mapping().values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('registry-outcome probabilities must sum to 1')
        return self

    def as_mapping(self) -> dict[RegistryOutcomeClass, float]:
        return {
            RegistryOutcomeClass.COMPLETED: self.completed,
            RegistryOutcomeClass.TERMINATED: self.terminated,
            RegistryOutcomeClass.WITHDRAWN: self.withdrawn,
            RegistryOutcomeClass.SUSPENDED: self.suspended,
            RegistryOutcomeClass.NON_TERMINAL: self.non_terminal,
            RegistryOutcomeClass.STATUS_MISSING: self.status_missing,
            RegistryOutcomeClass.RECORD_MISSING: self.record_missing,
        }


class ObservationStateProbabilities(StrictModel):
    """A complete probability simplex that makes later registry missingness scoreable."""

    observed_actual: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    not_actual: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    value_missing: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    record_missing: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_simplex(self) -> Self:
        if not math.isclose(math.fsum(self.as_mapping().values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('observation-state probabilities must sum to 1')
        return self

    def as_mapping(self) -> dict[ObservationState, float]:
        return {
            ObservationState.OBSERVED_ACTUAL: self.observed_actual,
            ObservationState.NOT_ACTUAL: self.not_actual,
            ObservationState.VALUE_MISSING: self.value_missing,
            ObservationState.RECORD_MISSING: self.record_missing,
        }


class ContinuousForecastSpec(StrictModel):
    """Precommitted format and clipping interval for one conditional forecast."""

    forecast_kind: Literal['point', 'quantiles']
    lower_bound: float = Field(allow_inf_nan=False)
    upper_bound: float = Field(allow_inf_nan=False)
    quantile_levels: tuple[float, ...] = ()

    @model_validator(mode='after')
    def validate_spec(self) -> Self:
        if self.upper_bound <= self.lower_bound:
            raise ValueError('continuous scoring bounds require upper_bound > lower_bound')
        if self.forecast_kind == 'point':
            if self.quantile_levels:
                raise ValueError('point forecast specs cannot declare quantile levels')
        else:
            if not self.quantile_levels:
                raise ValueError('quantile forecast specs require at least one quantile level')
            if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in self.quantile_levels):
                raise ValueError('quantile levels must be finite and strictly between 0 and 1')
            if self.quantile_levels != tuple(sorted(self.quantile_levels)) or len(self.quantile_levels) != len(
                set(self.quantile_levels)
            ):
                raise ValueError('quantile levels must be unique and strictly increasing')
        return self


class ConditionalPointForecast(StrictModel):
    """Point forecast conditional on the later registry field being marked Actual."""

    kind: Literal['point'] = 'point'
    value: float = Field(allow_inf_nan=False)


class QuantilePoint(StrictModel):
    quantile: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    value: float = Field(allow_inf_nan=False)


class ConditionalQuantileForecast(StrictModel):
    """Quantile forecast conditional on the later registry field being marked Actual."""

    kind: Literal['quantiles'] = 'quantiles'
    values: tuple[QuantilePoint, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_quantiles(self) -> Self:
        levels = tuple(item.quantile for item in self.values)
        predictions = tuple(item.value for item in self.values)
        if levels != tuple(sorted(levels)) or len(levels) != len(set(levels)):
            raise ValueError('submitted quantile levels must be unique and strictly increasing')
        if predictions != tuple(sorted(predictions)):
            raise ValueError('submitted quantile values cannot cross')
        return self


type ConditionalContinuousForecast = Annotated[
    ConditionalPointForecast | ConditionalQuantileForecast,
    Field(discriminator='kind'),
]


class CutoffDocument(StrictModel):
    """Exact UTF-8 text made available at the decision cutoff."""

    document_id: str = Field(min_length=1)
    available_on: date
    body: str = Field(min_length=1)
    body_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_body_hash(self) -> Self:
        if hashlib.sha256(self.body.encode('utf-8')).hexdigest() != self.body_sha256:
            raise ValueError('cutoff document body does not match body_sha256')
        if _NCT_IDENTIFIER.search(self.document_id) or _NCT_IDENTIFIER.search(self.body):
            raise ValueError('public cutoff documents cannot expose an NCT identifier')
        return self


class CutoffFactQuestion(StrictModel):
    """Optional deterministic extraction question over public cutoff documents."""

    question_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    answer_choices: tuple[str, ...] = Field(min_length=2)
    allowed_document_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_question(self) -> Self:
        if len(self.answer_choices) != len(set(self.answer_choices)):
            raise ValueError('fact answer choices must be unique')
        if len(self.allowed_document_ids) != len(set(self.allowed_document_ids)):
            raise ValueError('allowed fact document IDs must be unique')
        public_text = (
            self.question_id,
            self.prompt,
            *self.answer_choices,
            *self.allowed_document_ids,
        )
        if any(_NCT_IDENTIFIER.search(value) for value in public_text):
            raise ValueError('public fact questions and choices cannot expose an NCT identifier')
        return self


class ExecutionTaskContext(StrictModel):
    """Public, outcome-free context for exactly one target trial."""

    schema_version: Literal['vaxreplay.clinical-execution-task-context.dev-v0.1'] = (
        EXECUTION_TASK_CONTEXT_SCHEMA_VERSION
    )
    reward_version: Literal['vaxreplay.clinical-execution-reward.dev-v0.1'] = EXECUTION_REWARD_VERSION
    task_type: Literal['registry_observed_trial_execution'] = EXECUTION_TASK_ID
    task_semantics: ExecutionTaskSemantics = EXECUTION_TASK_SEMANTICS
    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    decision_snapshot_id: str = Field(min_length=1)
    anchor_date: date
    label_snapshot_id: str = Field(min_length=1)
    label_archive_date: date
    planned_enrollment: int = Field(gt=0)
    planned_primary_completion_date: date
    enrollment_ratio_spec: ContinuousForecastSpec
    primary_completion_slippage_days_spec: ContinuousForecastSpec
    cutoff_documents: tuple[CutoffDocument, ...] = ()
    fact_questions: tuple[CutoffFactQuestion, ...] = ()
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    span_mapped_identity_mask_receipt_present: Literal[False] = False
    per_episode_scalar_subset_selection_robust: Literal[False] = False
    source_derivation_verified: Literal[False] = False
    forecast_spec_preregistered: Literal[False] = False

    @model_validator(mode='after')
    def validate_context(self) -> Self:
        public_identifiers = (
            self.episode_id,
            self.target_trial_id,
            self.decision_snapshot_id,
            self.label_snapshot_id,
        )
        if any(_NCT_IDENTIFIER.search(value) for value in public_identifiers):
            raise ValueError('public task identifiers cannot expose an NCT identifier')
        if self.label_archive_date != add_calendar_months(self.anchor_date, LABEL_HORIZON_MONTHS):
            raise ValueError('label archive must be exactly 48 calendar months after the decision anchor')
        if self.planned_primary_completion_date <= self.anchor_date:
            raise ValueError('planned primary completion must be after the decision anchor')
        document_ids = tuple(document.document_id for document in self.cutoff_documents)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError('cutoff document IDs must be unique')
        for document in self.cutoff_documents:
            if document.available_on > self.anchor_date:
                raise ValueError('cutoff documents cannot be available after the decision anchor')
        question_ids = tuple(question.question_id for question in self.fact_questions)
        if len(question_ids) != len(set(question_ids)):
            raise ValueError('fact question IDs must be unique')
        known_documents = set(document_ids)
        for question in self.fact_questions:
            if unknown := set(question.allowed_document_ids) - known_documents:
                raise ValueError(f'fact question references unknown cutoff documents {sorted(unknown)}')
        return self


def execution_task_context_sha256(context: ExecutionTaskContext) -> str:
    validated = ExecutionTaskContext.model_validate_json(canonical_json_bytes(context))
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


class ExecutionTask(StrictModel):
    """Public development task plus a hiding commitment to its private outcome."""

    schema_version: Literal['vaxreplay.clinical-execution-task.dev-v0.1'] = EXECUTION_TASK_SCHEMA_VERSION
    context: ExecutionTaskContext
    context_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_scheme: Literal['hmac-sha256'] = 'hmac-sha256'
    private_gold_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_key_id: str = Field(pattern=_SHA256_PATTERN)
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    source_derivation_verified: Literal[False] = False
    forecast_spec_preregistered: Literal[False] = False

    @model_validator(mode='after')
    def validate_context_binding(self) -> Self:
        if execution_task_context_sha256(self.context) != self.context_sha256:
            raise ValueError('execution task context does not match context_sha256')
        return self


class CutoffCitation(StrictModel):
    """Half-open byte span into one public UTF-8 cutoff document."""

    document_id: str = Field(min_length=1)
    start_byte: int = Field(ge=0)
    end_byte: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_span(self) -> Self:
        if self.end_byte <= self.start_byte:
            raise ValueError('citation requires end_byte > start_byte')
        if self.end_byte - self.start_byte > MAX_CITATION_BYTES:
            raise ValueError(f'citation cannot exceed {MAX_CITATION_BYTES} bytes')
        return self


class CutoffFactAnswer(StrictModel):
    question_id: str = Field(min_length=1)
    selected_choice: str = Field(min_length=1)
    citations: tuple[CutoffCitation, ...] = Field(min_length=1, max_length=MAX_CITATIONS_PER_FACT)

    @model_validator(mode='after')
    def validate_unique_citations(self) -> Self:
        keys = tuple((item.document_id, item.start_byte, item.end_byte) for item in self.citations)
        if len(keys) != len(set(keys)):
            raise ValueError('citations within one fact answer must be unique')
        return self


class ExecutionSubmission(StrictModel):
    """Final machine-scoreable answer for one execution task; no ranking fields exist."""

    schema_version: Literal['vaxreplay.clinical-execution-submission.dev-v0.1'] = EXECUTION_SUBMISSION_SCHEMA_VERSION
    reward_version: Literal['vaxreplay.clinical-execution-reward.dev-v0.1'] = EXECUTION_REWARD_VERSION
    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    registry_outcome_probabilities: RegistryOutcomeProbabilities
    enrollment_observation_probabilities: ObservationStateProbabilities
    primary_completion_observation_probabilities: ObservationStateProbabilities
    enrollment_ratio_given_observed_actual: ConditionalContinuousForecast
    primary_completion_slippage_days_given_observed_actual: ConditionalContinuousForecast
    fact_answers: tuple[CutoffFactAnswer, ...] = ()

    @field_validator('fact_answers')
    @classmethod
    def validate_unique_fact_answers(cls, value: tuple[CutoffFactAnswer, ...]) -> tuple[CutoffFactAnswer, ...]:
        question_ids = tuple(answer.question_id for answer in value)
        if len(question_ids) != len(set(question_ids)):
            raise ValueError('fact answers must have unique question IDs')
        return value


class GoldByteSpan(StrictModel):
    document_id: str = Field(min_length=1)
    start_byte: int = Field(ge=0)
    end_byte: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_span(self) -> Self:
        if self.end_byte <= self.start_byte:
            raise ValueError('gold byte span requires end_byte > start_byte')
        if self.end_byte - self.start_byte > MAX_CITATION_BYTES:
            raise ValueError(f'gold byte span cannot exceed {MAX_CITATION_BYTES} bytes')
        return self


class GoldCutoffFact(StrictModel):
    """One correct choice and alternative exact spans; one accepted span is sufficient.

    Complementary evidence groups that require multiple distinct supporting spans are intentionally
    deferred to a future contract rather than approximated here.
    """

    question_id: str = Field(min_length=1)
    correct_choice: str = Field(min_length=1)
    acceptable_citations: tuple[GoldByteSpan, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_unique_citations(self) -> Self:
        keys = tuple((item.document_id, item.start_byte, item.end_byte) for item in self.acceptable_citations)
        if len(keys) != len(set(keys)):
            raise ValueError('acceptable gold citations must be unique')
        return self


class ExecutionPrivateGold(StrictModel):
    """Hidden later registry observation for one task."""

    schema_version: Literal['vaxreplay.clinical-execution-private-gold.dev-v0.1'] = (
        EXECUTION_PRIVATE_GOLD_SCHEMA_VERSION
    )
    reward_version: Literal['vaxreplay.clinical-execution-reward.dev-v0.1'] = EXECUTION_REWARD_VERSION
    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    organizer_private_nct_id: str = Field(pattern=r'^NCT\d{8}$')
    organizer_private_decision_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    registry_outcome_class: RegistryOutcomeClass
    enrollment_observation: ObservationState
    enrollment_ratio: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    primary_completion_observation: ObservationState
    primary_completion_slippage_days: int | None = None
    fact_labels: tuple[GoldCutoffFact, ...] = ()
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_observability(self) -> Self:
        if (self.enrollment_ratio is not None) != (self.enrollment_observation == ObservationState.OBSERVED_ACTUAL):
            raise ValueError('enrollment ratio exists exactly when actual enrollment is observed')
        if (self.primary_completion_slippage_days is not None) != (
            self.primary_completion_observation == ObservationState.OBSERVED_ACTUAL
        ):
            raise ValueError('completion slippage exists exactly when an actual completion date is observed')
        record_missing = self.registry_outcome_class == RegistryOutcomeClass.RECORD_MISSING
        observation_record_missing = (
            self.enrollment_observation == ObservationState.RECORD_MISSING,
            self.primary_completion_observation == ObservationState.RECORD_MISSING,
        )
        if (record_missing and not all(observation_record_missing)) or (
            not record_missing and any(observation_record_missing)
        ):
            raise ValueError('record-missing outcome and observation states must agree')
        question_ids = tuple(label.question_id for label in self.fact_labels)
        if len(question_ids) != len(set(question_ids)):
            raise ValueError('private fact labels must have unique question IDs')
        return self


def execution_private_gold_commitment(gold: ExecutionPrivateGold, key: bytes) -> str:
    """Return a domain-separated HMAC so low-entropy outcomes cannot be dictionary attacked."""

    if len(key) < 32:
        raise ValueError('execution private-gold HMAC key must contain at least 32 bytes')
    validated = ExecutionPrivateGold.model_validate_json(canonical_json_bytes(gold))
    return hmac.new(key, _PRIVATE_GOLD_HMAC_DOMAIN + canonical_json_bytes(validated), hashlib.sha256).hexdigest()


def _span_bytes(document: CutoffDocument, start_byte: int, end_byte: int) -> bytes:
    body = document.body.encode('utf-8')
    if end_byte > len(body):
        raise ValueError('fact citation is outside its cutoff document')
    selected = body[start_byte:end_byte]
    try:
        selected.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError('fact citation must begin and end on UTF-8 character boundaries') from exc
    return selected


def validate_execution_task_gold(
    task: ExecutionTask,
    gold: ExecutionPrivateGold,
    key: bytes,
) -> None:
    """Verify all public/private bindings before a scorer is allowed to see the gold."""

    task = ExecutionTask.model_validate_json(canonical_json_bytes(task))
    gold = ExecutionPrivateGold.model_validate_json(canonical_json_bytes(gold))
    if execution_task_context_sha256(task.context) != task.context_sha256:
        raise ValueError('execution task context does not match its committed hash')
    if len(key) < 32:
        raise ValueError('execution private-gold HMAC key must contain at least 32 bytes')
    expected_key_id = hashlib.sha256(key).hexdigest()
    if not hmac.compare_digest(expected_key_id, task.private_gold_commitment_key_id):
        raise ValueError('private-gold key does not match the task key ID')
    observed_commitment = execution_private_gold_commitment(gold, key)
    if not hmac.compare_digest(observed_commitment, task.private_gold_commitment_sha256):
        raise ValueError('private gold does not match the task HMAC commitment')
    context = task.context
    if (gold.episode_id, gold.target_trial_id, gold.task_context_sha256) != (
        context.episode_id,
        context.target_trial_id,
        task.context_sha256,
    ):
        raise ValueError('private gold does not bind the exact public task context')

    questions = {question.question_id: question for question in context.fact_questions}
    labels = {label.question_id: label for label in gold.fact_labels}
    if set(labels) != set(questions):
        raise ValueError('private fact labels must exactly cover every public fact question')
    documents = {document.document_id: document for document in context.cutoff_documents}
    for question_id, label in labels.items():
        question = questions[question_id]
        if label.correct_choice not in question.answer_choices:
            raise ValueError(f'private fact label {question_id} is outside the public answer choices')
        for span in label.acceptable_citations:
            if span.document_id not in question.allowed_document_ids:
                raise ValueError(f'private fact label {question_id} cites a disallowed document')
            _span_bytes(documents[span.document_id], span.start_byte, span.end_byte)


def build_execution_task(
    *,
    context: ExecutionTaskContext,
    gold: ExecutionPrivateGold,
    private_gold_key: bytes,
) -> ExecutionTask:
    """Build and immediately audit a development task without writing private labels anywhere."""

    context = ExecutionTaskContext.model_validate_json(canonical_json_bytes(context))
    gold = ExecutionPrivateGold.model_validate_json(canonical_json_bytes(gold))
    context_sha256 = execution_task_context_sha256(context)
    if gold.task_context_sha256 != context_sha256:
        raise ValueError('private gold task_context_sha256 does not match the supplied context')
    task = ExecutionTask(
        context=context,
        context_sha256=context_sha256,
        private_gold_commitment_sha256=execution_private_gold_commitment(gold, private_gold_key),
        private_gold_commitment_key_id=hashlib.sha256(private_gold_key).hexdigest(),
    )
    validate_execution_task_gold(task, gold, private_gold_key)
    return task


__all__ = [
    'EXECUTION_PRIVATE_GOLD_SCHEMA_VERSION',
    'EXECUTION_REWARD_VERSION',
    'EXECUTION_SUBMISSION_SCHEMA_VERSION',
    'EXECUTION_TASK_CONTEXT_SCHEMA_VERSION',
    'EXECUTION_TASK_SCHEMA_VERSION',
    'MAX_CITATIONS_PER_FACT',
    'MAX_CITATION_BYTES',
    'ConditionalContinuousForecast',
    'ConditionalPointForecast',
    'ConditionalQuantileForecast',
    'ContinuousForecastSpec',
    'CutoffCitation',
    'CutoffDocument',
    'CutoffFactAnswer',
    'CutoffFactQuestion',
    'ExecutionPrivateGold',
    'ExecutionSubmission',
    'ExecutionTask',
    'ExecutionTaskContext',
    'GoldByteSpan',
    'GoldCutoffFact',
    'ObservationStateProbabilities',
    'QuantilePoint',
    'RegistryOutcomeProbabilities',
    'build_execution_task',
    'execution_private_gold_commitment',
    'execution_task_context_sha256',
    'validate_execution_task_gold',
]
