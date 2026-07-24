"""Deterministic contracts, scoring, and baselines for Agentic Replay V1.

The agentic evaluator deliberately scores only a canonical final task dossier.  Tool traces,
free-form rationales, and model-provided quote text are never reward inputs.  Citations are byte
offsets into a separately verified, frozen workspace; the evaluator reads and hashes those bytes
itself before it considers a submission.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.admission import AgenticWorkspaceAdmission, require_workspace_admission
from vaxreplay.agentic.schema import (
    AgenticAssuranceProfile,
    AgenticValueType,
    AgenticWorkspaceSource,
    agentic_model_sha256,
)
from vaxreplay.agentic.workspace import LoadedAgenticWorkspace, load_agentic_workspace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import ScoreStatus, StrictModel

AGENTIC_SUBMISSION_SCHEMA_VERSION = 'vaxreplay.agentic-submission-file.v0.1'
AGENTIC_SCORING_CONTRACT_SCHEMA_VERSION = 'vaxreplay.agentic-scoring-contract.v1'
AGENTIC_PRIVATE_GOLD_SCHEMA_VERSION = 'vaxreplay.agentic-private-gold.v1'
AGENTIC_SCORE_SCHEMA_VERSION = 'vaxreplay.agentic-score.v1'
AGENTIC_REWARD_VERSION = 'vaxreplay.agentic-reward.v1.0'

MAX_SOURCE_SPAN_BYTES = 2_048
MAX_GOLD_SPAN_EXPANSION_BYTES = 256
MAX_CITATIONS_PER_FACT = 16
MAX_TOTAL_CITATIONS = 256
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_PRIVATE_GOLD_HMAC_DOMAIN = b'vaxreplay.agentic-private-gold.v1\x00'


def agentic_private_gold_commitment(gold: AgenticPrivateGoldV1, key: bytes) -> str:
    """Return a domain-separated HMAC commitment without exposing low-entropy private labels."""

    if len(key) < 32:
        raise ValueError('Agentic private-gold HMAC key must contain at least 32 bytes')
    return hmac.new(key, _PRIVATE_GOLD_HMAC_DOMAIN + canonical_json_bytes(gold), hashlib.sha256).hexdigest()


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _unique(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f'{field_name} must be unique')
    return values


class TypedValueKind(str, enum.Enum):
    NUMBER = 'number'
    STRING = 'string'
    BOOLEAN = 'boolean'


class TypedValue(StrictModel):
    """A small deterministic value algebra; free-form prose is not a scored value."""

    kind: TypedValueKind
    number: float | int | None = Field(default=None, allow_inf_nan=False)
    text: str | None = None
    boolean: bool | None = None
    unit: str | None = None

    @model_validator(mode='after')
    def validate_value(self) -> Self:
        if self.kind == TypedValueKind.NUMBER:
            if self.number is None or self.text is not None or self.boolean is not None:
                raise ValueError('number values require only number')
            if self.unit is not None and not self.unit.strip():
                raise ValueError('numeric units cannot be blank')
        elif self.kind == TypedValueKind.BOOLEAN:
            if self.boolean is None or self.number is not None or self.text is not None or self.unit is not None:
                raise ValueError('boolean values require only boolean')
        else:
            if (
                self.text is None
                or not self.text.strip()
                or self.number is not None
                or self.boolean is not None
                or self.unit is not None
            ):
                raise ValueError('string values require only nonblank text')
        return self


class SourceSpan(StrictModel):
    """Half-open UTF-8 byte span in one source; no submitted quote is trusted."""

    source_id: str = Field(min_length=1)
    start_byte: int = Field(ge=0)
    end_byte: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_order(self) -> Self:
        if self.end_byte <= self.start_byte:
            raise ValueError('source spans require end_byte > start_byte')
        return self

    @property
    def byte_count(self) -> int:
        return self.end_byte - self.start_byte


class FactAnswerStatus(str, enum.Enum):
    OBSERVED = 'observed'
    NOT_FOUND = 'not_found'
    CONFLICT = 'conflict'


class MetricAnswerStatus(str, enum.Enum):
    COMPUTED = 'computed'
    NOT_COMPUTABLE = 'not_computable'


class DecisionStatus(str, enum.Enum):
    RECOMMEND = 'recommend'
    INSUFFICIENT_EVIDENCE = 'insufficient_evidence'


class FactAnswer(StrictModel):
    query_id: str = Field(min_length=1)
    status: FactAnswerStatus
    value: TypedValue | None = None
    citations: tuple[SourceSpan, ...] = Field(default=(), max_length=MAX_CITATIONS_PER_FACT)

    @model_validator(mode='after')
    def validate_status(self) -> Self:
        if self.status == FactAnswerStatus.OBSERVED:
            if self.value is None or not self.citations:
                raise ValueError('observed facts require a typed value and at least one citation')
        elif self.status == FactAnswerStatus.NOT_FOUND:
            if self.value is not None or self.citations:
                raise ValueError('not_found facts cannot contain a value or citations')
        elif self.value is not None or len(self.citations) < 2:
            raise ValueError('conflict facts require no value and at least two citations')
        return self


class DerivedMetricAnswer(StrictModel):
    metric_id: str = Field(min_length=1)
    status: MetricAnswerStatus
    formula_id: str = Field(min_length=1)
    dependency_query_ids: tuple[str, ...] = Field(min_length=1)
    value: TypedValue | None = None

    @field_validator('dependency_query_ids')
    @classmethod
    def validate_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, 'derived metric dependency_query_ids')

    @model_validator(mode='after')
    def validate_status(self) -> Self:
        if self.status == MetricAnswerStatus.COMPUTED and self.value is None:
            raise ValueError('computed metrics require a typed value')
        if self.status == MetricAnswerStatus.NOT_COMPUTABLE and self.value is not None:
            raise ValueError('not_computable metrics cannot contain a value')
        return self


class CandidateProbability(StrictModel):
    candidate_id: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class AgenticDecision(StrictModel):
    status: DecisionStatus
    ranking: tuple[str, ...] = Field(min_length=2)
    portfolio: tuple[str, ...] = ()
    advancement_probabilities: tuple[CandidateProbability, ...] = Field(min_length=2)

    @field_validator('ranking', 'portfolio')
    @classmethod
    def validate_candidate_lists(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return _unique(value, info.field_name)

    @field_validator('advancement_probabilities')
    @classmethod
    def validate_probabilities(cls, value: tuple[CandidateProbability, ...]) -> tuple[CandidateProbability, ...]:
        _unique(tuple(item.candidate_id for item in value), 'advancement probability candidate IDs')
        return value

    @model_validator(mode='after')
    def validate_status(self) -> Self:
        if self.status == DecisionStatus.RECOMMEND:
            if not self.portfolio or self.portfolio != self.ranking[: len(self.portfolio)]:
                raise ValueError('recommended portfolio must be a nonempty prefix of ranking')
        elif self.portfolio:
            raise ValueError('insufficient_evidence decisions must use an empty portfolio')
        return self


class AgenticSubmissionV1(StrictModel):
    schema_version: Literal['vaxreplay.agentic-submission-file.v0.1'] = AGENTIC_SUBMISSION_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    fact_answers: tuple[FactAnswer, ...] = Field(min_length=1)
    derived_metrics: tuple[DerivedMetricAnswer, ...] = ()
    decision: AgenticDecision

    @model_validator(mode='after')
    def validate_citation_budget(self) -> Self:
        if sum(len(answer.citations) for answer in self.fact_answers) > MAX_TOTAL_CITATIONS:
            raise ValueError(f'submissions cannot contain more than {MAX_TOTAL_CITATIONS} citations')
        return self


class RequiredFactQuery(StrictModel):
    query_id: str = Field(min_length=1)
    value_type: AgenticValueType
    unit: str | None = None

    @model_validator(mode='after')
    def validate_unit(self) -> Self:
        if self.value_type != AgenticValueType.NUMBER and self.unit is not None:
            raise ValueError('only numeric fact queries can declare a unit')
        return self


class RequiredMetric(StrictModel):
    metric_id: str = Field(min_length=1)
    value_type: AgenticValueType
    unit: str | None = None
    formula_id: str = Field(min_length=1)
    dependency_query_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator('dependency_query_ids')
    @classmethod
    def validate_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, 'required metric dependency_query_ids')

    @model_validator(mode='after')
    def validate_unit(self) -> Self:
        if self.value_type != AgenticValueType.NUMBER and self.unit is not None:
            raise ValueError('only numeric derived metrics can declare a unit')
        return self


class AgenticScoringContract(StrictModel):
    schema_version: Literal['vaxreplay.agentic-scoring-contract.v1'] = AGENTIC_SCORING_CONTRACT_SCHEMA_VERSION
    reward_version: Literal['vaxreplay.agentic-reward.v1.0'] = AGENTIC_REWARD_VERSION
    task_id: str = Field(min_length=1)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_at: datetime
    candidate_ids: tuple[str, ...] = Field(min_length=2)
    portfolio_size: int = Field(gt=0)
    fact_queries: tuple[RequiredFactQuery, ...] = Field(min_length=1)
    required_metrics: tuple[RequiredMetric, ...] = ()

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return _aware(value, 'decision_at')

    @field_validator('candidate_ids')
    @classmethod
    def validate_ids(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return _unique(value, info.field_name)

    @field_validator('fact_queries')
    @classmethod
    def validate_queries(cls, value: tuple[RequiredFactQuery, ...]) -> tuple[RequiredFactQuery, ...]:
        _unique(tuple(query.query_id for query in value), 'required fact query IDs')
        return value

    @field_validator('required_metrics')
    @classmethod
    def validate_metrics(cls, value: tuple[RequiredMetric, ...]) -> tuple[RequiredMetric, ...]:
        _unique(tuple(metric.metric_id for metric in value), 'required metric IDs')
        return value

    @model_validator(mode='after')
    def validate_contract(self) -> Self:
        if self.portfolio_size >= len(self.candidate_ids):
            raise ValueError('portfolio_size must be smaller than the candidate count')
        known_queries = {query.query_id for query in self.fact_queries}
        for metric in self.required_metrics:
            if unknown := set(metric.dependency_query_ids) - known_queries:
                raise ValueError(f'metric {metric.metric_id} references unknown fact queries {sorted(unknown)}')
        return self

    @property
    def fact_query_ids(self) -> tuple[str, ...]:
        return tuple(query.query_id for query in self.fact_queries)

    @classmethod
    def from_workspace(cls, workspace: LoadedAgenticWorkspace) -> AgenticScoringContract:
        task = workspace.task
        return cls(
            reward_version=AGENTIC_REWARD_VERSION,
            task_id=task.task_id,
            workspace_manifest_sha256=workspace.manifest_sha256,
            decision_at=task.decision_at,
            candidate_ids=task.candidate_ids,
            portfolio_size=task.portfolio_size,
            fact_queries=tuple(
                RequiredFactQuery(query_id=query.query_id, value_type=query.value_type, unit=query.unit)
                for query in task.fact_queries
            ),
            required_metrics=tuple(
                RequiredMetric(
                    metric_id=metric.metric_id,
                    value_type=metric.value_type,
                    unit=metric.unit,
                    formula_id=metric.formula_id,
                    dependency_query_ids=metric.dependency_query_ids,
                )
                for metric in task.derived_metrics
            ),
        )


class GoldEvidenceGroup(StrictModel):
    group_id: str = Field(min_length=1)
    acceptable_source_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator('acceptable_source_ids')
    @classmethod
    def validate_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, 'gold evidence-group source IDs')


class GoldSupportGroup(StrictModel):
    group_id: str = Field(min_length=1)
    evidence_group_id: str = Field(min_length=1)
    alternatives: tuple[SourceSpan, ...] = Field(min_length=1, max_length=MAX_CITATIONS_PER_FACT)

    @field_validator('alternatives')
    @classmethod
    def validate_alternatives(cls, value: tuple[SourceSpan, ...]) -> tuple[SourceSpan, ...]:
        keys = tuple((span.source_id, span.start_byte, span.end_byte) for span in value)
        if len(keys) != len(set(keys)):
            raise ValueError('gold support alternatives must be unique')
        return value


class GoldFactLabel(StrictModel):
    query_id: str = Field(min_length=1)
    status: FactAnswerStatus
    accepted_values: tuple[TypedValue, ...] = ()
    absolute_tolerance: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    relative_tolerance: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    support_groups: tuple[GoldSupportGroup, ...] = Field(default=(), max_length=MAX_CITATIONS_PER_FACT)

    @model_validator(mode='after')
    def validate_label(self) -> Self:
        group_ids = tuple(group.group_id for group in self.support_groups)
        _unique(group_ids, 'gold fact support-group IDs')
        if self.status == FactAnswerStatus.OBSERVED:
            if not self.accepted_values or not self.support_groups:
                raise ValueError('observed gold facts require accepted values and support groups')
        elif self.status == FactAnswerStatus.NOT_FOUND:
            if self.accepted_values or self.support_groups:
                raise ValueError('not_found gold facts cannot contain values or support groups')
        elif self.accepted_values or len(self.support_groups) < 2:
            raise ValueError('conflict gold facts require no accepted value and at least two support groups')
        if (self.absolute_tolerance or self.relative_tolerance) and any(
            value.kind != TypedValueKind.NUMBER for value in self.accepted_values
        ):
            raise ValueError('numeric tolerances can only be used with numeric accepted values')
        return self


class GoldMetricLabel(StrictModel):
    metric_id: str = Field(min_length=1)
    status: MetricAnswerStatus
    formula_id: str = Field(min_length=1)
    dependency_query_ids: tuple[str, ...] = Field(min_length=1)
    accepted_values: tuple[TypedValue, ...] = ()
    absolute_tolerance: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    relative_tolerance: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @field_validator('dependency_query_ids')
    @classmethod
    def validate_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, 'gold metric dependency_query_ids')

    @model_validator(mode='after')
    def validate_label(self) -> Self:
        if self.status == MetricAnswerStatus.COMPUTED and not self.accepted_values:
            raise ValueError('computed gold metrics require accepted values')
        if self.status == MetricAnswerStatus.NOT_COMPUTABLE and self.accepted_values:
            raise ValueError('not_computable gold metrics cannot contain accepted values')
        if (self.absolute_tolerance or self.relative_tolerance) and any(
            value.kind != TypedValueKind.NUMBER for value in self.accepted_values
        ):
            raise ValueError('numeric tolerances can only be used with numeric accepted values')
        return self


class GoldCandidateDecision(StrictModel):
    candidate_id: str = Field(min_length=1)
    relevance: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)
    utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    historically_advanced: bool


class AgenticPrivateGoldV1(StrictModel):
    schema_version: Literal['vaxreplay.agentic-private-gold.v1'] = AGENTIC_PRIVATE_GOLD_SCHEMA_VERSION
    reward_version: Literal['vaxreplay.agentic-reward.v1.0'] = AGENTIC_REWARD_VERSION
    task_id: str = Field(min_length=1)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    scoring_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    fact_labels: tuple[GoldFactLabel, ...] = Field(min_length=1)
    metric_labels: tuple[GoldMetricLabel, ...] = ()
    evidence_groups: tuple[GoldEvidenceGroup, ...] = ()
    decision_labels: tuple[GoldCandidateDecision, ...] = Field(min_length=2)

    @model_validator(mode='after')
    def validate_unique_keys(self) -> Self:
        _unique(tuple(label.query_id for label in self.fact_labels), 'gold fact query IDs')
        _unique(tuple(label.metric_id for label in self.metric_labels), 'gold metric IDs')
        _unique(tuple(group.group_id for group in self.evidence_groups), 'gold evidence-group IDs')
        _unique(tuple(label.candidate_id for label in self.decision_labels), 'gold decision candidate IDs')
        support_ids = tuple(group.group_id for label in self.fact_labels for group in label.support_groups)
        _unique(support_ids, 'gold support-group IDs')
        return self


class AgenticIssueCode(str, enum.Enum):
    TASK_MISMATCH = 'TASK_MISMATCH'
    WORKSPACE_MISMATCH = 'WORKSPACE_MISMATCH'
    WORKSPACE_INTEGRITY = 'WORKSPACE_INTEGRITY'
    INVALID_FACT_COVERAGE = 'INVALID_FACT_COVERAGE'
    INVALID_FACT_VALUE_TYPE = 'INVALID_FACT_VALUE_TYPE'
    INVALID_METRIC_COVERAGE = 'INVALID_METRIC_COVERAGE'
    INVALID_METRIC_VALUE_TYPE = 'INVALID_METRIC_VALUE_TYPE'
    INVALID_DECISION = 'INVALID_DECISION'
    INVALID_SPAN = 'INVALID_SPAN'
    OVERLONG_SPAN = 'OVERLONG_SPAN'
    DUPLICATE_SPAN = 'DUPLICATE_SPAN'
    LEAK_UNKNOWN_SOURCE = 'LEAK_UNKNOWN_SOURCE'
    LEAK_POST_CUTOFF_SOURCE = 'LEAK_POST_CUTOFF_SOURCE'


class AgenticValidationIssue(StrictModel):
    code: AgenticIssueCode
    detail: str = Field(min_length=1)


class AgenticScoreVectorV1(StrictModel):
    schema_version: Literal['vaxreplay.agentic-score.v1'] = AGENTIC_SCORE_SCHEMA_VERSION
    reward_version: Literal['vaxreplay.agentic-reward.v1.0'] = AGENTIC_REWARD_VERSION
    task_id: str = Field(min_length=1)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    scoring_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_key_id: str = Field(pattern=_SHA256_PATTERN)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    assurance_profile: AgenticAssuranceProfile
    admitted_use: Literal['prospective_research', 'retrospective_research', 'best_effort_research', 'fixture']
    status: ScoreStatus
    reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    retrieval_precision: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    retrieval_recall: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    retrieval_f1: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    extraction_score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    extraction_signed_utility: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    analysis_score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    analysis_signed_utility: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    citation_recall: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    citation_f1: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    ndcg_at_k: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    top_k_utility: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    decision_score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    process_score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    advancement_brier: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    advancement_brier_skill: float | None = Field(default=None, allow_inf_nan=False)
    advancement_prevalence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    issues: tuple[AgenticValidationIssue, ...] = ()

    @model_validator(mode='after')
    def validate_status_and_formulas(self) -> Self:
        required_metrics = (
            self.reward,
            self.retrieval_precision,
            self.retrieval_recall,
            self.retrieval_f1,
            self.extraction_score,
            self.extraction_signed_utility,
            self.analysis_score,
            self.analysis_signed_utility,
            self.citation_precision,
            self.citation_recall,
            self.citation_f1,
            self.ndcg_at_k,
            self.top_k_utility,
            self.decision_score,
            self.process_score,
            self.advancement_brier,
            self.advancement_prevalence,
        )
        all_metrics = (*required_metrics, self.advancement_brier_skill)
        expected_use = {
            AgenticAssuranceProfile.PROSPECTIVE_EXACT: 'prospective_research',
            AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE: 'retrospective_research',
            AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT: 'best_effort_research',
            AgenticAssuranceProfile.FIXTURE: 'fixture',
        }[self.assurance_profile]
        if self.admitted_use != expected_use:
            raise ValueError('score admitted_use must reflect its assurance profile')
        if self.status == ScoreStatus.VALID:
            if any(value is None for value in required_metrics) or self.issues:
                raise ValueError('valid agentic scores require every required metric and no issues')
            assert self.retrieval_precision is not None
            assert self.retrieval_recall is not None
            assert self.retrieval_f1 is not None
            assert self.citation_precision is not None
            assert self.citation_recall is not None
            assert self.citation_f1 is not None
            assert self.extraction_score is not None
            assert self.extraction_signed_utility is not None
            assert self.analysis_score is not None
            assert self.analysis_signed_utility is not None
            assert self.ndcg_at_k is not None
            assert self.top_k_utility is not None
            assert self.decision_score is not None
            assert self.process_score is not None
            assert self.reward is not None
            assert self.advancement_prevalence is not None
            skill_should_be_defined = 0.0 < self.advancement_prevalence < 1.0
            if (self.advancement_brier_skill is not None) != skill_should_be_defined:
                raise ValueError('Brier skill is defined only for nondegenerate empirical prevalence')
            expected_retrieval = _f1(self.retrieval_precision, self.retrieval_recall)
            expected_citation = _f1(self.citation_precision, self.citation_recall)
            expected_decision = 0.5 * self.ndcg_at_k + 0.5 * self.top_k_utility
            expected_process = _harmonic(
                self.retrieval_f1,
                self.extraction_score,
                self.analysis_score,
                self.citation_f1,
            )
            expected_reward = _harmonic(expected_process, expected_decision)
            checks = (
                (self.retrieval_f1, expected_retrieval, 'retrieval_f1'),
                (self.citation_f1, expected_citation, 'citation_f1'),
                (self.decision_score, expected_decision, 'decision_score'),
                (self.process_score, expected_process, 'process_score'),
                (self.reward, expected_reward, 'reward'),
                (
                    self.extraction_score,
                    _bounded_signed_utility(self.extraction_signed_utility),
                    'extraction_score',
                ),
                (
                    self.analysis_score,
                    _bounded_signed_utility(self.analysis_signed_utility),
                    'analysis_score',
                ),
            )
            for observed, expected, name in checks:
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f'{name} is inconsistent with the fixed agentic formula')
        elif any(value is not None for value in all_metrics) or not self.issues:
            raise ValueError('invalid agentic scores require issues and cannot contain scalar metrics')
        return self

    def metrics(self) -> dict[str, float]:
        values = self.model_dump(
            mode='python',
            exclude={
                'schema_version',
                'task_id',
                'workspace_manifest_sha256',
                'workspace_admission_sha256',
                'scoring_contract_sha256',
                'private_gold_commitment_sha256',
                'private_gold_commitment_key_id',
                'submission_sha256',
                'reward_version',
                'assurance_profile',
                'admitted_use',
                'status',
                'issues',
            },
        )
        return {name: value for name, value in values.items() if isinstance(value, (float, int))}


@dataclass(frozen=True)
class _LoadedSource:
    source: AgenticWorkspaceSource
    content: bytes


class AgenticSubmissionEvaluator:
    """Private, deterministic evaluator for one frozen Agentic Replay task."""

    def __init__(
        self,
        workspace: LoadedAgenticWorkspace,
        admission: AgenticWorkspaceAdmission,
        expected_admission_sha256: str,
        gold: AgenticPrivateGoldV1,
        gold_commitment_key: bytes,
        expected_gold_commitment_sha256: str,
    ):
        verified_workspace = load_agentic_workspace(workspace.root)
        require_workspace_admission(
            verified_workspace,
            admission,
            expected_admission_sha256=expected_admission_sha256,
        )
        self._contract = AgenticScoringContract.from_workspace(verified_workspace)
        self._contract_sha256 = agentic_model_sha256(self._contract)
        observed_gold_commitment = agentic_private_gold_commitment(gold, gold_commitment_key)
        if not hmac.compare_digest(observed_gold_commitment, expected_gold_commitment_sha256):
            raise ValueError('private gold does not match its trusted commitment')
        self._gold = gold
        self._workspace = verified_workspace
        self._admission_sha256 = expected_admission_sha256
        self._gold_commitment_sha256 = expected_gold_commitment_sha256
        self._gold_commitment_key_id = hashlib.sha256(gold_commitment_key).hexdigest()
        self._assurance_profile = admission.assurance_profile
        self._admitted_use = admission.admitted_use
        self._workspace_issues, self._sources = _load_workspace_sources(verified_workspace)
        self._validate_gold()

    def score(self, submission: AgenticSubmissionV1) -> AgenticScoreVectorV1:
        submission_sha256 = agentic_model_sha256(submission)
        issues = [*self._workspace_issues, *self._validate_submission(submission)]
        if issues:
            leakage_codes = {
                AgenticIssueCode.LEAK_UNKNOWN_SOURCE,
                AgenticIssueCode.LEAK_POST_CUTOFF_SOURCE,
            }
            return AgenticScoreVectorV1(
                task_id=self._contract.task_id,
                workspace_manifest_sha256=self._contract.workspace_manifest_sha256,
                workspace_admission_sha256=self._admission_sha256,
                scoring_contract_sha256=self._contract_sha256,
                private_gold_commitment_sha256=self._gold_commitment_sha256,
                private_gold_commitment_key_id=self._gold_commitment_key_id,
                submission_sha256=submission_sha256,
                assurance_profile=self._assurance_profile,
                admitted_use=self._admitted_use,
                status=(
                    ScoreStatus.INVALID_LEAKAGE
                    if any(issue.code in leakage_codes for issue in issues)
                    else ScoreStatus.INVALID_SCHEMA
                ),
                issues=tuple(issues),
            )

        retrieval_precision, retrieval_recall, retrieval_f1 = self._retrieval_scores(submission)
        extraction_score, extraction_signed_utility = self._extraction_scores(submission)
        analysis_score, analysis_signed_utility = self._analysis_scores(submission)
        citation_precision, citation_recall, citation_f1 = self._citation_scores(submission)
        ndcg_at_k, top_k_utility = self._decision_scores(submission)
        decision_score = 0.5 * ndcg_at_k + 0.5 * top_k_utility
        process_score = _harmonic(retrieval_f1, extraction_score, analysis_score, citation_f1)
        reward = _harmonic(process_score, decision_score)
        advancement_brier, advancement_brier_skill, advancement_prevalence = self._forecast_diagnostics(submission)
        return AgenticScoreVectorV1(
            task_id=self._contract.task_id,
            workspace_manifest_sha256=self._contract.workspace_manifest_sha256,
            workspace_admission_sha256=self._admission_sha256,
            scoring_contract_sha256=self._contract_sha256,
            private_gold_commitment_sha256=self._gold_commitment_sha256,
            private_gold_commitment_key_id=self._gold_commitment_key_id,
            submission_sha256=submission_sha256,
            assurance_profile=self._assurance_profile,
            admitted_use=self._admitted_use,
            status=ScoreStatus.VALID,
            reward=reward,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            retrieval_f1=retrieval_f1,
            extraction_score=extraction_score,
            extraction_signed_utility=extraction_signed_utility,
            analysis_score=analysis_score,
            analysis_signed_utility=analysis_signed_utility,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            citation_f1=citation_f1,
            ndcg_at_k=ndcg_at_k,
            top_k_utility=top_k_utility,
            decision_score=decision_score,
            process_score=process_score,
            advancement_brier=advancement_brier,
            advancement_brier_skill=advancement_brier_skill,
            advancement_prevalence=advancement_prevalence,
        )

    def _validate_gold(self) -> None:
        contract = self._contract
        gold = self._gold
        if gold.reward_version != contract.reward_version:
            raise ValueError('private gold reward_version does not match the scoring contract')
        if gold.task_id != contract.task_id:
            raise ValueError('private gold task_id does not match the scoring contract')
        if gold.workspace_manifest_sha256 != contract.workspace_manifest_sha256:
            raise ValueError('private gold does not bind the exact scoring workspace')
        if gold.scoring_contract_sha256 != self._contract_sha256:
            raise ValueError('private gold does not bind the exact scoring contract')
        if {label.query_id for label in gold.fact_labels} != set(contract.fact_query_ids):
            raise ValueError('private gold must exactly cover every required fact query')
        required_queries = {query.query_id: query for query in contract.fact_queries}
        for label in gold.fact_labels:
            required = required_queries[label.query_id]
            for value in label.accepted_values:
                if not _value_matches_contract(value, required.value_type, required.unit):
                    raise ValueError(f'gold fact {label.query_id} has the wrong typed value or unit')
        required_metrics = {metric.metric_id: metric for metric in contract.required_metrics}
        if {label.metric_id for label in gold.metric_labels} != set(required_metrics):
            raise ValueError('private gold must exactly cover every required metric')
        for label in gold.metric_labels:
            required = required_metrics[label.metric_id]
            if (label.formula_id, label.dependency_query_ids) != (
                required.formula_id,
                required.dependency_query_ids,
            ):
                raise ValueError(f'gold metric {label.metric_id} disagrees with its public contract')
            for value in label.accepted_values:
                if not _value_matches_contract(value, required.value_type, required.unit):
                    raise ValueError(f'gold metric {label.metric_id} has the wrong typed value or unit')
            if label.status == MetricAnswerStatus.COMPUTED:
                fact_labels = {fact.query_id: fact for fact in gold.fact_labels}
                if any(
                    fact_labels[query_id].status != FactAnswerStatus.OBSERVED for query_id in label.dependency_query_ids
                ):
                    raise ValueError(f'computed gold metric {label.metric_id} requires observed dependency facts')
        if {label.candidate_id for label in gold.decision_labels} != set(contract.candidate_ids):
            raise ValueError('private gold must exactly cover every candidate decision')

        evidence_by_id = {group.group_id: group for group in gold.evidence_groups}
        known_sources = set(self._sources)
        for group in gold.evidence_groups:
            if unknown := set(group.acceptable_source_ids) - known_sources:
                raise ValueError(f'evidence group {group.group_id} references unknown sources {sorted(unknown)}')
        referenced_evidence_groups: set[str] = set()
        for label in gold.fact_labels:
            _choose_unique_support_spans(label.support_groups)
            for support in label.support_groups:
                evidence = evidence_by_id.get(support.evidence_group_id)
                if evidence is None:
                    raise ValueError(f'support group {support.group_id} references an unknown evidence group')
                referenced_evidence_groups.add(support.evidence_group_id)
                acceptable_sources = set(evidence.acceptable_source_ids)
                if any(span.source_id not in acceptable_sources for span in support.alternatives):
                    raise ValueError(f'support group {support.group_id} uses a source outside its evidence group')
                for span in support.alternatives:
                    error = self._span_error(span, gold_span=True)
                    if error is not None:
                        raise ValueError(f'invalid private gold span in {support.group_id}: {error.detail}')
        if referenced_evidence_groups != set(evidence_by_id):
            raise ValueError('every private evidence group must be referenced by a fact support group')

        ideal_ranking = _ideal_ranking(gold.decision_labels)
        relevance = {label.candidate_id: label.relevance for label in gold.decision_labels}
        ideal_dcg = _discounted_gain(ideal_ranking[: contract.portfolio_size], relevance)
        if ideal_dcg <= 0.0:
            raise ValueError('decision gold requires positive ideal DCG')
        utilities = {label.candidate_id: label.utility for label in gold.decision_labels}
        sorted_utilities = sorted(utilities.values())
        worst = sum(sorted_utilities[: contract.portfolio_size])
        best = sum(sorted_utilities[-contract.portfolio_size :])
        if math.isclose(best, worst, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError('decision gold requires nondegenerate top-k utility')
        oracle_selected = sum(utilities[candidate_id] for candidate_id in ideal_ranking[: contract.portfolio_size])
        if not math.isclose(oracle_selected, best, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('relevance and utility gold must admit a common oracle ranking')

    def _validate_submission(self, submission: AgenticSubmissionV1) -> list[AgenticValidationIssue]:
        issues: list[AgenticValidationIssue] = []
        contract = self._contract
        if submission.task_id != contract.task_id:
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.TASK_MISMATCH,
                    detail=f'expected task {contract.task_id}, got {submission.task_id}',
                )
            )
        if submission.workspace_manifest_sha256 != contract.workspace_manifest_sha256:
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.WORKSPACE_MISMATCH,
                    detail='submission is not bound to the frozen workspace manifest',
                )
            )

        fact_keys = tuple(answer.query_id for answer in submission.fact_answers)
        if len(fact_keys) != len(set(fact_keys)) or set(fact_keys) != set(contract.fact_query_ids):
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.INVALID_FACT_COVERAGE,
                    detail='fact answers must cover every required query exactly once',
                )
            )
        required_queries = {query.query_id: query for query in contract.fact_queries}
        for answer in submission.fact_answers:
            required_query = required_queries.get(answer.query_id)
            if (
                required_query is not None
                and answer.value is not None
                and not _value_matches_contract(
                    answer.value,
                    required_query.value_type,
                    required_query.unit,
                )
            ):
                issues.append(
                    AgenticValidationIssue(
                        code=AgenticIssueCode.INVALID_FACT_VALUE_TYPE,
                        detail=f'fact {answer.query_id} does not match its declared value type or unit',
                    )
                )
        metric_keys = tuple(answer.metric_id for answer in submission.derived_metrics)
        required_by_id = {metric.metric_id: metric for metric in contract.required_metrics}
        fact_by_id = {answer.query_id: answer for answer in submission.fact_answers}
        if len(metric_keys) != len(set(metric_keys)) or set(metric_keys) != set(required_by_id):
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.INVALID_METRIC_COVERAGE,
                    detail='derived metrics must cover every required metric exactly once',
                )
            )
        for answer in submission.derived_metrics:
            required = required_by_id.get(answer.metric_id)
            if required is not None and (answer.formula_id, answer.dependency_query_ids) != (
                required.formula_id,
                required.dependency_query_ids,
            ):
                issues.append(
                    AgenticValidationIssue(
                        code=AgenticIssueCode.INVALID_METRIC_COVERAGE,
                        detail=f'derived metric {answer.metric_id} changed its formula or dependencies',
                    )
                )
            if (
                required is not None
                and answer.value is not None
                and not _value_matches_contract(answer.value, required.value_type, required.unit)
            ):
                issues.append(
                    AgenticValidationIssue(
                        code=AgenticIssueCode.INVALID_METRIC_VALUE_TYPE,
                        detail=f'derived metric {answer.metric_id} does not match its declared value type or unit',
                    )
                )
            if (
                required is not None
                and answer.status == MetricAnswerStatus.COMPUTED
                and any(
                    fact_by_id.get(query_id) is None or fact_by_id[query_id].status != FactAnswerStatus.OBSERVED
                    for query_id in required.dependency_query_ids
                )
            ):
                issues.append(
                    AgenticValidationIssue(
                        code=AgenticIssueCode.INVALID_METRIC_COVERAGE,
                        detail=f'computed metric {answer.metric_id} requires observed dependency facts',
                    )
                )

        issues.extend(self._validate_decision(submission.decision))
        for answer in submission.fact_answers:
            seen: set[tuple[str, int, int]] = set()
            for span in answer.citations:
                key = (span.source_id, span.start_byte, span.end_byte)
                if key in seen:
                    issues.append(
                        AgenticValidationIssue(
                            code=AgenticIssueCode.DUPLICATE_SPAN,
                            detail=f'fact {answer.query_id} repeats source span {key}',
                        )
                    )
                    continue
                seen.add(key)
                if error := self._span_error(span, gold_span=False):
                    issues.append(error)
        return issues

    def _validate_decision(self, decision: AgenticDecision) -> list[AgenticValidationIssue]:
        issues: list[AgenticValidationIssue] = []
        candidate_ids = set(self._contract.candidate_ids)
        ranking = tuple(decision.ranking)
        if len(ranking) != len(set(ranking)) or set(ranking) != candidate_ids:
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.INVALID_DECISION,
                    detail='decision ranking must contain every candidate exactly once',
                )
            )
        k = self._contract.portfolio_size
        if decision.status == DecisionStatus.RECOMMEND:
            if len(decision.portfolio) != k or tuple(decision.portfolio) != ranking[:k]:
                issues.append(
                    AgenticValidationIssue(
                        code=AgenticIssueCode.INVALID_DECISION,
                        detail='recommended portfolio must equal the first portfolio_size ranked candidates',
                    )
                )
        elif decision.portfolio:
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.INVALID_DECISION,
                    detail='insufficient_evidence decisions require an empty portfolio',
                )
            )
        probability_ids = tuple(item.candidate_id for item in decision.advancement_probabilities)
        if len(probability_ids) != len(set(probability_ids)) or set(probability_ids) != candidate_ids:
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.INVALID_DECISION,
                    detail='advancement probabilities must cover every candidate exactly once',
                )
            )
        return issues

    def _span_error(self, span: SourceSpan, *, gold_span: bool) -> AgenticValidationIssue | None:
        source = self._sources.get(span.source_id)
        if source is None:
            return AgenticValidationIssue(
                code=AgenticIssueCode.LEAK_UNKNOWN_SOURCE,
                detail=f'citation references unknown source {span.source_id}',
            )
        if source.source.effective_available_at_upper > self._contract.decision_at:
            return AgenticValidationIssue(
                code=AgenticIssueCode.LEAK_POST_CUTOFF_SOURCE,
                detail=f'citation references post-cutoff source {span.source_id}',
            )
        if span.end_byte > len(source.content):
            return AgenticValidationIssue(
                code=AgenticIssueCode.INVALID_SPAN,
                detail=f'span for {span.source_id} exceeds its frozen bytes',
            )
        limit = MAX_SOURCE_SPAN_BYTES
        if span.byte_count > limit:
            return AgenticValidationIssue(
                code=AgenticIssueCode.OVERLONG_SPAN,
                detail=f'span for {span.source_id} exceeds {limit} bytes',
            )
        try:
            source.content[: span.start_byte].decode('utf-8')
            source.content[: span.end_byte].decode('utf-8')
            source.content[span.start_byte : span.end_byte].decode('utf-8')
        except UnicodeDecodeError:
            return AgenticValidationIssue(
                code=AgenticIssueCode.INVALID_SPAN,
                detail=f'span for {span.source_id} is not aligned to UTF-8 byte boundaries',
            )
        _ = gold_span
        return None

    def _retrieval_scores(self, submission: AgenticSubmissionV1) -> tuple[float, float, float]:
        cited_sources = {span.source_id for answer in submission.fact_answers for span in answer.citations}
        if not self._gold.evidence_groups:
            precision = 1.0 if not cited_sources else 0.0
            return precision, 1.0, _f1(precision, 1.0)
        accepted_sources = {
            source_id for group in self._gold.evidence_groups for source_id in group.acceptable_source_ids
        }
        precision = len(cited_sources & accepted_sources) / len(cited_sources) if cited_sources else 0.0
        hit_groups = sum(bool(cited_sources & set(group.acceptable_source_ids)) for group in self._gold.evidence_groups)
        recall = hit_groups / len(self._gold.evidence_groups)
        return precision, recall, _f1(precision, recall)

    def _extraction_scores(self, submission: AgenticSubmissionV1) -> tuple[float, float]:
        answers = {answer.query_id: answer for answer in submission.fact_answers}
        scores = [self._fact_cell_score(answers[label.query_id], label) for label in self._gold.fact_labels]
        signed = _signed_macro_utility(scores)
        return _bounded_signed_utility(signed), signed

    @staticmethod
    def _fact_cell_score(answer: FactAnswer, label: GoldFactLabel) -> float:
        if answer.status == label.status:
            if label.status != FactAnswerStatus.OBSERVED:
                return 1.0
            assert answer.value is not None
            return (
                1.0
                if any(
                    _typed_value_equal(
                        answer.value,
                        accepted,
                        abs_tol=label.absolute_tolerance,
                        rel_tol=label.relative_tolerance,
                    )
                    for accepted in label.accepted_values
                )
                else -1.0
            )
        if answer.status == FactAnswerStatus.NOT_FOUND and label.status != FactAnswerStatus.NOT_FOUND:
            return 0.0
        return -1.0

    def _analysis_scores(self, submission: AgenticSubmissionV1) -> tuple[float, float]:
        metric_answers = {answer.metric_id: answer for answer in submission.derived_metrics}
        fact_answers = {answer.query_id: answer for answer in submission.fact_answers}
        fact_labels = {label.query_id: label for label in self._gold.fact_labels}
        scores: list[float] = []
        for label in self._gold.metric_labels:
            answer = metric_answers[label.metric_id]
            dependencies_correct = all(
                self._fact_cell_score(fact_answers[query_id], fact_labels[query_id]) == 1.0
                for query_id in label.dependency_query_ids
            )
            if label.status == MetricAnswerStatus.COMPUTED and not dependencies_correct:
                scores.append(0.0 if answer.status == MetricAnswerStatus.NOT_COMPUTABLE else -1.0)
            else:
                scores.append(self._metric_cell_score(answer, label))
        signed = _signed_macro_utility(scores)
        return _bounded_signed_utility(signed), signed

    @staticmethod
    def _metric_cell_score(answer: DerivedMetricAnswer, label: GoldMetricLabel) -> float:
        if answer.status == label.status:
            if label.status != MetricAnswerStatus.COMPUTED:
                return 1.0
            assert answer.value is not None
            return (
                1.0
                if any(
                    _typed_value_equal(
                        answer.value,
                        accepted,
                        abs_tol=label.absolute_tolerance,
                        rel_tol=label.relative_tolerance,
                    )
                    for accepted in label.accepted_values
                )
                else -1.0
            )
        if answer.status == MetricAnswerStatus.NOT_COMPUTABLE and label.status != MetricAnswerStatus.NOT_COMPUTABLE:
            return 0.0
        return -1.0

    def _citation_scores(self, submission: AgenticSubmissionV1) -> tuple[float, float, float]:
        predictions: list[tuple[str, SourceSpan]] = [
            (answer.query_id, span) for answer in submission.fact_answers for span in answer.citations
        ]
        answers = {answer.query_id: answer for answer in submission.fact_answers}
        correct_queries = {
            label.query_id
            for label in self._gold.fact_labels
            if self._fact_cell_score(answers[label.query_id], label) == 1.0
        }
        gold_groups: list[tuple[str, GoldSupportGroup]] = [
            (label.query_id, group) for label in self._gold.fact_labels for group in label.support_groups
        ]
        adjacency = [
            [
                index
                for index, (gold_query_id, group) in enumerate(gold_groups)
                if query_id == gold_query_id
                and query_id in correct_queries
                and any(_span_matches(span, accepted) for accepted in group.alternatives)
            ]
            for query_id, span in predictions
        ]
        matched = _maximum_bipartite_matches(adjacency, len(gold_groups))
        precision = matched / len(predictions) if predictions else (1.0 if not gold_groups else 0.0)
        recall = matched / len(gold_groups) if gold_groups else 1.0
        return precision, recall, _f1(precision, recall)

    def _decision_scores(self, submission: AgenticSubmissionV1) -> tuple[float, float]:
        if submission.decision.status == DecisionStatus.INSUFFICIENT_EVIDENCE:
            return 0.0, 0.0
        labels = self._gold.decision_labels
        relevance = {label.candidate_id: label.relevance for label in labels}
        k = self._contract.portfolio_size
        ranking = list(submission.decision.ranking)
        dcg = _discounted_gain(ranking[:k], relevance)
        ideal = _ideal_ranking(labels)
        ideal_dcg = _discounted_gain(ideal[:k], relevance)
        ndcg = min(1.0, max(0.0, dcg / ideal_dcg))

        utility = {label.candidate_id: label.utility for label in labels}
        selected = sum(utility[candidate_id] for candidate_id in submission.decision.portfolio)
        ordered = sorted(utility.values())
        worst = sum(ordered[:k])
        best = sum(ordered[-k:])
        top_k = min(1.0, max(0.0, (selected - worst) / (best - worst)))
        return ndcg, top_k

    def _forecast_diagnostics(self, submission: AgenticSubmissionV1) -> tuple[float, float | None, float]:
        predictions = {item.candidate_id: item.probability for item in submission.decision.advancement_probabilities}
        outcomes = {label.candidate_id: float(label.historically_advanced) for label in self._gold.decision_labels}
        brier = math.fsum(
            (predictions[candidate_id] - outcome) ** 2 for candidate_id, outcome in outcomes.items()
        ) / len(outcomes)
        prevalence = math.fsum(outcomes.values()) / len(outcomes)
        baseline = math.fsum((prevalence - outcome) ** 2 for outcome in outcomes.values()) / len(outcomes)
        skill = (baseline - brier) / baseline if baseline > 0.0 else None
        return brier, skill, prevalence


def score_agentic_submission(
    *,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
    gold: AgenticPrivateGoldV1,
    gold_commitment_key: bytes,
    expected_gold_commitment_sha256: str,
    submission: AgenticSubmissionV1,
) -> AgenticScoreVectorV1:
    """Score one submission only after exact workspace admission is reverified."""

    return AgenticSubmissionEvaluator(
        workspace,
        admission,
        expected_admission_sha256,
        gold,
        gold_commitment_key,
        expected_gold_commitment_sha256,
    ).score(submission)


def all_abstain_submission(contract: AgenticScoringContract) -> AgenticSubmissionV1:
    """Schema-valid no-information baseline; its task reward is necessarily zero."""

    probability = contract.portfolio_size / len(contract.candidate_ids)
    return AgenticSubmissionV1(
        task_id=contract.task_id,
        workspace_manifest_sha256=contract.workspace_manifest_sha256,
        fact_answers=tuple(
            FactAnswer(query_id=query_id, status=FactAnswerStatus.NOT_FOUND) for query_id in contract.fact_query_ids
        ),
        derived_metrics=tuple(
            DerivedMetricAnswer(
                metric_id=metric.metric_id,
                status=MetricAnswerStatus.NOT_COMPUTABLE,
                formula_id=metric.formula_id,
                dependency_query_ids=metric.dependency_query_ids,
            )
            for metric in contract.required_metrics
        ),
        decision=AgenticDecision(
            status=DecisionStatus.INSUFFICIENT_EVIDENCE,
            ranking=contract.candidate_ids,
            advancement_probabilities=tuple(
                CandidateProbability(candidate_id=candidate_id, probability=probability)
                for candidate_id in contract.candidate_ids
            ),
        ),
    )


def random_submission(contract: AgenticScoringContract, *, seed: int = 0) -> AgenticSubmissionV1:
    """No-evidence baseline with a deterministic random complete decision ranking."""

    baseline = all_abstain_submission(contract)
    ranking = list(contract.candidate_ids)
    random.Random(seed).shuffle(ranking)
    return baseline.model_copy(
        update={
            'decision': AgenticDecision(
                status=DecisionStatus.RECOMMEND,
                ranking=tuple(ranking),
                portfolio=tuple(ranking[: contract.portfolio_size]),
                advancement_probabilities=baseline.decision.advancement_probabilities,
            )
        }
    )


def oracle_submission(
    contract: AgenticScoringContract,
    gold: AgenticPrivateGoldV1,
) -> AgenticSubmissionV1:
    """Private solvability baseline; never expose it for a sealed test task."""

    fact_answers: list[FactAnswer] = []
    for label in gold.fact_labels:
        citations = _choose_unique_support_spans(label.support_groups)
        value = label.accepted_values[0] if label.status == FactAnswerStatus.OBSERVED else None
        fact_answers.append(
            FactAnswer(
                query_id=label.query_id,
                status=label.status,
                value=value,
                citations=citations,
            )
        )
    metric_answers = tuple(
        DerivedMetricAnswer(
            metric_id=label.metric_id,
            status=label.status,
            formula_id=label.formula_id,
            dependency_query_ids=label.dependency_query_ids,
            value=label.accepted_values[0] if label.status == MetricAnswerStatus.COMPUTED else None,
        )
        for label in gold.metric_labels
    )
    ranking = _ideal_ranking(gold.decision_labels)
    return AgenticSubmissionV1(
        task_id=contract.task_id,
        workspace_manifest_sha256=contract.workspace_manifest_sha256,
        fact_answers=tuple(fact_answers),
        derived_metrics=metric_answers,
        decision=AgenticDecision(
            status=DecisionStatus.RECOMMEND,
            ranking=ranking,
            portfolio=ranking[: contract.portfolio_size],
            advancement_probabilities=tuple(
                CandidateProbability(
                    candidate_id=candidate_id,
                    probability=float(
                        next(
                            label.historically_advanced
                            for label in gold.decision_labels
                            if label.candidate_id == candidate_id
                        )
                    ),
                )
                for candidate_id in contract.candidate_ids
            ),
        ),
    )


def _load_workspace_sources(
    workspace: LoadedAgenticWorkspace,
) -> tuple[list[AgenticValidationIssue], dict[str, _LoadedSource]]:
    issues: list[AgenticValidationIssue] = []
    loaded: dict[str, _LoadedSource] = {}
    for source in workspace.sources:
        try:
            content = workspace.read_source(source.source_id)
        except ValueError as error:
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.WORKSPACE_INTEGRITY,
                    detail=str(error),
                )
            )
            continue
        if (hashlib.sha256(content).hexdigest(), len(content)) != (source.sha256, source.byte_count):
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.WORKSPACE_INTEGRITY,
                    detail=f'workspace source changed before scoring: {source.source_id}',
                )
            )
            continue
        loaded[source.source_id] = _LoadedSource(source=source, content=content)
        if source.effective_available_at_upper > workspace.task.decision_at:
            issues.append(
                AgenticValidationIssue(
                    code=AgenticIssueCode.LEAK_POST_CUTOFF_SOURCE,
                    detail=f'workspace exposes post-cutoff source {source.source_id}',
                )
            )
    return issues, loaded


def _typed_value_equal(
    observed: TypedValue,
    expected: TypedValue,
    *,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    if observed.kind != expected.kind or observed.unit != expected.unit:
        return False
    if observed.kind == TypedValueKind.NUMBER:
        assert observed.number is not None and expected.number is not None
        return math.isclose(float(observed.number), float(expected.number), rel_tol=rel_tol, abs_tol=abs_tol)
    if observed.kind == TypedValueKind.BOOLEAN:
        return observed.boolean == expected.boolean
    return observed.text == expected.text


def _value_matches_contract(
    value: TypedValue,
    expected_type: AgenticValueType,
    expected_unit: str | None,
) -> bool:
    expected_kind = {
        AgenticValueType.STRING: TypedValueKind.STRING,
        AgenticValueType.NUMBER: TypedValueKind.NUMBER,
        AgenticValueType.BOOLEAN: TypedValueKind.BOOLEAN,
    }[expected_type]
    return value.kind == expected_kind and value.unit == expected_unit


def _span_matches(observed: SourceSpan, accepted: SourceSpan) -> bool:
    if observed.source_id != accepted.source_id:
        return False
    observed_contains_gold = observed.start_byte <= accepted.start_byte and observed.end_byte >= accepted.end_byte
    maximum = min(MAX_SOURCE_SPAN_BYTES, accepted.byte_count + MAX_GOLD_SPAN_EXPANSION_BYTES)
    return observed_contains_gold and observed.byte_count <= maximum


def _maximum_bipartite_matches(adjacency: list[list[int]], gold_count: int) -> int:
    prediction_by_gold = [-1] * gold_count

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for gold_index in adjacency[prediction_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            previous = prediction_by_gold[gold_index]
            if previous == -1 or augment(previous, visited):
                prediction_by_gold[gold_index] = prediction_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(adjacency)))


def _choose_unique_support_spans(groups: tuple[GoldSupportGroup, ...]) -> tuple[SourceSpan, ...]:
    span_owner: dict[tuple[str, int, int], int] = {}
    selection: dict[int, SourceSpan] = {}

    def assign(group_index: int, visited: set[tuple[str, int, int]]) -> bool:
        for span in groups[group_index].alternatives:
            key = (span.source_id, span.start_byte, span.end_byte)
            if key in visited:
                continue
            visited.add(key)
            previous = span_owner.get(key)
            if previous is None or assign(previous, visited):
                span_owner[key] = group_index
                selection[group_index] = span
                return True
        return False

    for index in range(len(groups)):
        if not assign(index, set()):
            raise ValueError('private support groups do not admit unique oracle citations')
    return tuple(selection[index] for index in range(len(groups)))


def _ideal_ranking(labels: tuple[GoldCandidateDecision, ...]) -> tuple[str, ...]:
    return tuple(
        label.candidate_id
        for label in sorted(
            labels,
            key=lambda label: (-label.relevance, -label.utility, label.candidate_id),
        )
    )


def _discounted_gain(ranking: list[str] | tuple[str, ...], relevance: dict[str, float]) -> float:
    return math.fsum(
        math.expm1(math.log(2.0) * relevance[candidate_id]) / math.log2(index + 2)
        for index, candidate_id in enumerate(ranking)
    )


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _signed_macro_utility(values: list[float]) -> float:
    """Macro-average correct/abstain/wrong utilities (+1/0/-1) without erasing ordering."""

    if not values:
        return 1.0
    return min(1.0, max(-1.0, math.fsum(values) / len(values)))


def _bounded_signed_utility(value: float) -> float:
    """Map wrong/abstain/correct utility (-1/0/+1) monotonically onto (0/.5/1)."""

    return (value + 1.0) / 2.0


def _harmonic(*values: float) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / math.fsum(1.0 / value for value in values)
