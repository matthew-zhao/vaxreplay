"""V1 ranking labels and score vector, kept separate from bit-stable V0 models."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, model_validator

from vaxreplay.case_schema import RANKING_REWARD_VERSION, ScoreStatus, StrictModel, ValidationIssue

RANKING_LABEL_SCHEMA_VERSION = 'vaxreplay.ranking.v1.0'

V1_FORECAST_WEIGHT = 0.50
V1_RANKING_WEIGHT = 0.30
V1_GROUNDING_WEIGHT = 0.20

V1_NDCG_WEIGHT = 0.50
V1_PAIRWISE_WEIGHT = 0.25
V1_TOP_K_UTILITY_WEIGHT = 0.25


class RankingLabelV1(StrictModel):
    schema_version: Literal['vaxreplay.ranking.v1.0'] = RANKING_LABEL_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    relevance_grade: int | None = Field(default=None, ge=0, le=4)
    censor_reason: str | None = None

    @model_validator(mode='after')
    def validate_censoring(self) -> Self:
        if self.relevance_grade is None and not self.censor_reason:
            raise ValueError('censored ranking labels require censor_reason')
        if self.relevance_grade is not None and self.censor_reason is not None:
            raise ValueError('observed ranking labels cannot have censor_reason')
        return self


class ScoreVectorV1(StrictModel):
    episode_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    labels_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reward_version: Literal['v1.0'] = RANKING_REWARD_VERSION
    status: ScoreStatus
    reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    forecast_brier: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    forecast_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    ndcg_at_k: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    pairwise_concordance: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    top_k_utility: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    ranking_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_precision: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_recall: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_f1: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    assessment_accuracy: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    issues: list[ValidationIssue] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_status_and_formula(self) -> Self:
        score_values = (
            self.reward,
            self.forecast_brier,
            self.forecast_reward,
            self.ndcg_at_k,
            self.pairwise_concordance,
            self.top_k_utility,
            self.ranking_reward,
            self.grounding_precision,
            self.grounding_recall,
            self.grounding_f1,
            self.assessment_accuracy,
            self.grounding_reward,
        )
        if self.status == ScoreStatus.VALID:
            if any(value is None for value in score_values) or self.issues:
                raise ValueError('valid V1 scores require every metric and no validation issues')
            assert self.reward is not None
            assert self.forecast_brier is not None
            assert self.forecast_reward is not None
            assert self.ndcg_at_k is not None
            assert self.pairwise_concordance is not None
            assert self.top_k_utility is not None
            assert self.ranking_reward is not None
            assert self.grounding_precision is not None
            assert self.grounding_recall is not None
            assert self.grounding_f1 is not None
            assert self.assessment_accuracy is not None
            assert self.grounding_reward is not None
            if not math.isclose(self.forecast_reward, 1.0 - self.forecast_brier, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError('V1 forecast_reward is inconsistent with forecast_brier')
            expected_ranking = (
                V1_NDCG_WEIGHT * self.ndcg_at_k
                + V1_PAIRWISE_WEIGHT * self.pairwise_concordance
                + V1_TOP_K_UTILITY_WEIGHT * self.top_k_utility
            )
            if not math.isclose(self.ranking_reward, expected_ranking, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError('V1 ranking_reward is inconsistent with its components')
            expected_f1 = (
                0.0
                if self.grounding_precision + self.grounding_recall == 0.0
                else (
                    2.0
                    * self.grounding_precision
                    * self.grounding_recall
                    / (self.grounding_precision + self.grounding_recall)
                )
            )
            if not math.isclose(self.grounding_f1, expected_f1, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError('V1 grounding_f1 is inconsistent with precision and recall')
            expected_grounding = self.grounding_f1 * self.assessment_accuracy
            if not math.isclose(self.grounding_reward, expected_grounding, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError('V1 grounding_reward is inconsistent with its components')
            expected_reward = (
                V1_FORECAST_WEIGHT * self.forecast_reward
                + V1_RANKING_WEIGHT * self.ranking_reward
                + V1_GROUNDING_WEIGHT * self.grounding_reward
            )
            if not math.isclose(self.reward, expected_reward, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError('V1 reward is inconsistent with its components')
        elif any(value is not None for value in score_values) or not self.issues:
            raise ValueError('invalid V1 scores require issues and cannot contain reward metrics')
        return self

    def metrics(self) -> dict[str, float]:
        values = {
            'reward': self.reward,
            'forecast_brier': self.forecast_brier,
            'forecast_reward': self.forecast_reward,
            'ndcg_at_k': self.ndcg_at_k,
            'pairwise_concordance': self.pairwise_concordance,
            'top_k_utility': self.top_k_utility,
            'ranking_reward': self.ranking_reward,
            'grounding_precision': self.grounding_precision,
            'grounding_recall': self.grounding_recall,
            'grounding_f1': self.grounding_f1,
            'assessment_accuracy': self.assessment_accuracy,
            'grounding_reward': self.grounding_reward,
        }
        return {name: value for name, value in values.items() if value is not None}
