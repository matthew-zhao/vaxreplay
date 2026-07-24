"""Strict shared schemas for VaxReplay episode bundles and submissions."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 'vaxreplay.v0.1'
REWARD_VERSION = 'v0.1'
RANKING_REWARD_VERSION = 'v1.0'
ANTIGEN_TARGET_PRIORITIZATION_TASK = 'antigen_target_prioritization'
PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK = 'preclinical_candidate_advancement'
EARLY_CLINICAL_ARM_PRIORITIZATION_TASK = 'early_clinical_arm_prioritization'

type RewardVersion = Literal['v0.1', 'v1.0']
type TaskType = Literal[
    'antigen_target_prioritization',
    'preclinical_candidate_advancement',
    'early_clinical_arm_prioritization',
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)


class Split(str, enum.Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'


class SourceType(str, enum.Enum):
    SURVEILLANCE = 'surveillance'
    EXPERIMENTAL = 'experimental'
    JOURNAL_ABSTRACT = 'journal_abstract'
    PUBLIC_HEALTH = 'public_health'
    OTHER = 'other'


class EvidenceStance(str, enum.Enum):
    SUPPORT = 'support'
    CONCERN = 'concern'


class AssessmentConclusion(str, enum.Enum):
    FAVORABLE = 'favorable'
    CONCERN = 'concern'
    MIXED = 'mixed'
    INSUFFICIENT = 'insufficient'


class LabelCommitmentScheme(str, enum.Enum):
    SHA256 = 'sha256'
    HMAC_SHA256 = 'hmac-sha256'


class ForecastTarget(StrictModel):
    target_id: str = Field(min_length=1)
    horizon_days: int = Field(gt=0)


class SourceSnapshotCommitment(StrictModel):
    snapshot_id: str = Field(min_length=1)
    source_build_at: datetime
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_url: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    citation: str = Field(min_length=1)

    @field_validator('source_build_at')
    @classmethod
    def validate_source_build_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('source_build_at must include a UTC offset')
        return value


class AdapterProvenance(StrictModel):
    adapter_id: str = Field(min_length=1)
    episode_spec_commitment: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_snapshot_id: str = Field(min_length=1)
    label_snapshot_id: str = Field(min_length=1)
    snapshot_commitments: list[SourceSnapshotCommitment] = Field(min_length=2)
    private_audit_commitment: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_snapshots(self) -> Self:
        snapshot_ids = [snapshot.snapshot_id for snapshot in self.snapshot_commitments]
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError('source snapshot commitments must be unique')
        if self.decision_snapshot_id not in snapshot_ids:
            raise ValueError('decision_snapshot_id must reference a source snapshot commitment')
        if self.label_snapshot_id not in snapshot_ids:
            raise ValueError('label_snapshot_id must reference a source snapshot commitment')
        return self


class EpisodeManifest(StrictModel):
    schema_version: Literal['vaxreplay.v0.1'] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    lineage_group_id: str = Field(min_length=1)
    synthetic: bool
    task_type: TaskType = ANTIGEN_TARGET_PRIORITIZATION_TASK
    split: Split
    decision_at: datetime
    closed_book: Literal[True] = True
    network_allowed: Literal[False] = False
    portfolio_size: int = Field(gt=0)
    candidate_ids: list[str] = Field(min_length=2)
    forecast_targets: list[ForecastTarget] = Field(min_length=1)
    required_dimensions: list[str] = Field(min_length=1)
    evidence_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    candidates_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    labels_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_commitment_scheme: LabelCommitmentScheme = LabelCommitmentScheme.SHA256
    label_commitment_key_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    adjudication_version: str = Field(min_length=1)
    source_provenance: AdapterProvenance | None = None
    reward_version: RewardVersion = REWARD_VERSION

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('decision_at must include a UTC offset')
        return value

    @model_validator(mode='after')
    def validate_collections(self) -> Self:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError('candidate_ids must be unique')
        if self.portfolio_size > len(self.candidate_ids):
            raise ValueError('portfolio_size cannot exceed the candidate count')
        if len(self.required_dimensions) != len(set(self.required_dimensions)):
            raise ValueError('required_dimensions must be unique')
        target_keys = [(target.target_id, target.horizon_days) for target in self.forecast_targets]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError('forecast target and horizon pairs must be unique')
        if self.label_commitment_scheme == LabelCommitmentScheme.HMAC_SHA256:
            if self.label_commitment_key_id is None:
                raise ValueError('HMAC label commitments require label_commitment_key_id')
        elif self.label_commitment_key_id is not None:
            raise ValueError('SHA-256 label commitments cannot declare label_commitment_key_id')
        if (
            self.reward_version == RANKING_REWARD_VERSION
            and self.split == Split.TEST
            and self.label_commitment_scheme != LabelCommitmentScheme.HMAC_SHA256
        ):
            raise ValueError('sealed V1 test episodes require HMAC label commitments')
        if self.task_type == PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK and self.reward_version != RANKING_REWARD_VERSION:
            raise ValueError('preclinical candidate advancement requires the V1 ranking reward')
        if self.task_type == EARLY_CLINICAL_ARM_PRIORITIZATION_TASK and self.reward_version != RANKING_REWARD_VERSION:
            raise ValueError('early clinical arm prioritization requires the V1 ranking reward')
        return self


class CandidateRecord(StrictModel):
    schema_version: Literal['vaxreplay.v0.1'] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    eligible: bool = True


class EvidenceRecord(StrictModel):
    schema_version: Literal['vaxreplay.v0.1'] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    source_type: SourceType
    collected_at: datetime | None = None
    available_at: datetime
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    body_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    related_candidate_ids: list[str] = Field(default_factory=list)
    provenance_url: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    derivation: str = Field(min_length=1)

    @field_validator('collected_at', 'available_at')
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError('timestamps must include a UTC offset')
        return value

    @model_validator(mode='after')
    def validate_release_order(self) -> Self:
        if self.collected_at is not None and self.collected_at > self.available_at:
            raise ValueError('collected_at cannot be after available_at')
        return self


class OutcomeRecord(StrictModel):
    schema_version: Literal['vaxreplay.v0.1'] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    horizon_days: int = Field(gt=0)
    outcome: Literal[0, 1] | None
    candidate_utility: float = Field(ge=0.0, le=1.0)
    revealed_at: datetime
    censor_reason: str | None = None

    @field_validator('revealed_at')
    @classmethod
    def validate_revealed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('revealed_at must include a UTC offset')
        return value

    @model_validator(mode='after')
    def validate_censoring(self) -> Self:
        if self.outcome is None and not self.censor_reason:
            raise ValueError('censored outcomes require censor_reason')
        if self.outcome is not None and self.censor_reason is not None:
            raise ValueError('observed outcomes cannot have censor_reason')
        return self


class GoldEvidenceRecord(StrictModel):
    schema_version: Literal['vaxreplay.v0.1'] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    stance: EvidenceStance
    quote: str = Field(min_length=12, max_length=400)


class GoldAssessmentRecord(StrictModel):
    schema_version: Literal['vaxreplay.v0.1'] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    conclusion: AssessmentConclusion


class PrivateLabels(StrictModel):
    outcomes: list[OutcomeRecord] = Field(min_length=1)
    assessments_gold: list[GoldAssessmentRecord] = Field(min_length=1)
    evidence_gold: list[GoldEvidenceRecord] = Field(min_length=1)


class Citation(StrictModel):
    evidence_id: str = Field(min_length=1)
    stance: EvidenceStance
    quote: str = Field(min_length=12, max_length=400)


class CandidateForecast(StrictModel):
    candidate_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    horizon_days: int = Field(gt=0)
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class CandidateAssessment(StrictModel):
    candidate_id: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    conclusion: AssessmentConclusion
    citations: list[Citation] = Field(default_factory=list)


class Submission(StrictModel):
    schema_version: Literal['vaxreplay.v0.1'] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    ranking: list[str] = Field(min_length=1)
    forecasts: list[CandidateForecast] = Field(min_length=1)
    assessments: list[CandidateAssessment] = Field(default_factory=list)


class ScoreStatus(str, enum.Enum):
    VALID = 'valid'
    INVALID_SCHEMA = 'invalid_schema'
    INVALID_LEAKAGE = 'invalid_leakage'


class IssueCode(str, enum.Enum):
    EPISODE_MISMATCH = 'EPISODE_MISMATCH'
    MANIFEST_HASH_MISMATCH = 'MANIFEST_HASH_MISMATCH'
    INVALID_RANKING = 'INVALID_RANKING'
    INVALID_FORECASTS = 'INVALID_FORECASTS'
    INVALID_ASSESSMENTS = 'INVALID_ASSESSMENTS'
    INVALID_CITATION_QUOTE = 'INVALID_CITATION_QUOTE'
    INVALID_RUN_RESPONSE = 'INVALID_RUN_RESPONSE'
    RUNNER_FAILURE = 'RUNNER_FAILURE'
    LEAK_NON_MANIFEST_SOURCE = 'LEAK_NON_MANIFEST_SOURCE'
    LEAK_POST_CUTOFF_SOURCE = 'LEAK_POST_CUTOFF_SOURCE'


class ValidationIssue(StrictModel):
    code: IssueCode
    detail: str
    fatal: Literal[True] = True


class ScoreVector(StrictModel):
    episode_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    labels_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reward_version: Literal['v0.1'] = REWARD_VERSION
    status: ScoreStatus
    reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    forecast_brier: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    forecast_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    ndcg_at_k: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_precision: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_recall: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_f1: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    assessment_accuracy: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    issues: list[ValidationIssue] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_status(self) -> Self:
        score_values = (
            self.reward,
            self.forecast_brier,
            self.forecast_reward,
            self.ndcg_at_k,
            self.grounding_precision,
            self.grounding_recall,
            self.grounding_f1,
            self.assessment_accuracy,
            self.grounding_reward,
        )
        if self.status == ScoreStatus.VALID:
            if any(value is None for value in score_values) or self.issues:
                raise ValueError('valid scores require every metric and no validation issues')
        elif any(value is not None for value in score_values) or not self.issues:
            raise ValueError('invalid scores require issues and cannot contain reward metrics')
        return self

    def metrics(self) -> dict[str, float]:
        values = {
            'reward': self.reward,
            'forecast_brier': self.forecast_brier,
            'forecast_reward': self.forecast_reward,
            'ndcg_at_k': self.ndcg_at_k,
            'grounding_precision': self.grounding_precision,
            'grounding_recall': self.grounding_recall,
            'grounding_f1': self.grounding_f1,
            'assessment_accuracy': self.assessment_accuracy,
            'grounding_reward': self.grounding_reward,
        }
        return {name: value for name, value in values.items() if value is not None}
