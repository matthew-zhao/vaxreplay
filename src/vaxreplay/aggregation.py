"""Deterministic, manifest-bound macro-aggregation of episode score vectors."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_schema import (
    RewardVersion,
    ScoreStatus,
    ScoreVector,
    StrictModel,
    TaskType,
)
from vaxreplay.ranking_schema import ScoreVectorV1

AGGREGATION_VERSION = 'vaxreplay.aggregate.v1'
SUITE_MANIFEST_SCHEMA_VERSION = 'vaxreplay.suite.v1'
INVALID_EPISODE_PENALTY = -1.0

type EpisodeScore = ScoreVector | ScoreVectorV1


class SuiteEpisodeBinding(StrictModel):
    episode_id: str = Field(min_length=1)
    task_type: TaskType
    reward_version: RewardVersion
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    labels_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class SuiteManifest(StrictModel):
    """Fixed, task-homogeneous episode set used by official aggregation."""

    schema_version: Literal['vaxreplay.suite.v1'] = SUITE_MANIFEST_SCHEMA_VERSION
    suite_id: str = Field(min_length=1)
    task_type: TaskType
    reward_version: RewardVersion
    episodes: tuple[SuiteEpisodeBinding, ...] = Field(min_length=1)

    @field_validator('episodes')
    @classmethod
    def validate_episodes(cls, value: tuple[SuiteEpisodeBinding, ...]) -> tuple[SuiteEpisodeBinding, ...]:
        episode_ids = tuple(binding.episode_id for binding in value)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('suite episode IDs must be unique')
        if episode_ids != tuple(sorted(episode_ids)):
            raise ValueError('suite episode bindings must be sorted by episode_id')
        return value

    @model_validator(mode='after')
    def validate_episode_profiles(self) -> Self:
        if any(binding.task_type != self.task_type for binding in self.episodes):
            raise ValueError('suite episode task_type must match the suite task_type')
        if any(binding.reward_version != self.reward_version for binding in self.episodes):
            raise ValueError('suite episode reward_version must match the suite reward_version')
        return self


class SuiteScore(StrictModel):
    """Auditable suite result using equal weight for every expected episode."""

    aggregation_version: Literal['vaxreplay.aggregate.v1'] = AGGREGATION_VERSION
    suite_id: str = Field(min_length=1)
    task_type: TaskType
    reward_version: RewardVersion
    suite_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    input_scores_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    episode_ids: tuple[str, ...] = Field(min_length=1)
    missing_episode_ids: tuple[str, ...] = ()
    episode_count: int = Field(gt=0)
    valid_episode_count: int = Field(ge=0)
    invalid_episode_count: int = Field(ge=0)
    invalid_episode_penalty: float = Field(
        default=INVALID_EPISODE_PENALTY,
        ge=INVALID_EPISODE_PENALTY,
        le=INVALID_EPISODE_PENALTY,
        allow_inf_nan=False,
    )
    validity_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    status_counts: dict[str, int]
    valid_metric_means: dict[str, float]
    all_episode_mean_environment_reward: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    valid_only_mean_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator('episode_ids', 'missing_episode_ids')
    @classmethod
    def validate_episode_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not episode_id for episode_id in value):
            raise ValueError('episode IDs cannot be empty')
        if len(value) != len(set(value)):
            raise ValueError('episode IDs must be unique')
        if value != tuple(sorted(value)):
            raise ValueError('episode IDs must be sorted')
        return value

    @field_validator('status_counts')
    @classmethod
    def validate_status_counts(cls, value: dict[str, int]) -> dict[str, int]:
        expected_keys = tuple(sorted(status.value for status in ScoreStatus))
        if tuple(value) != expected_keys:
            raise ValueError('status_counts must contain every status in sorted order')
        if any(count < 0 for count in value.values()):
            raise ValueError('status counts cannot be negative')
        return value

    @field_validator('valid_metric_means')
    @classmethod
    def validate_metric_means(cls, value: dict[str, float]) -> dict[str, float]:
        if tuple(value) != tuple(sorted(value)):
            raise ValueError('valid_metric_means keys must be sorted')
        if any(not math.isfinite(metric) for metric in value.values()):
            raise ValueError('valid_metric_means must be finite')
        return value

    @model_validator(mode='after')
    def validate_summary(self) -> Self:
        if self.episode_count != len(self.episode_ids):
            raise ValueError('episode_count must equal the number of episode_ids')
        if not set(self.missing_episode_ids).issubset(self.episode_ids):
            raise ValueError('missing_episode_ids must be a subset of episode_ids')
        if self.valid_episode_count + self.invalid_episode_count != self.episode_count:
            raise ValueError('valid and invalid counts must sum to episode_count')
        if self.invalid_episode_count < len(self.missing_episode_ids):
            raise ValueError('every missing episode must count as invalid')
        if sum(self.status_counts.values()) != self.episode_count:
            raise ValueError('status counts must sum to episode_count')
        if self.status_counts[ScoreStatus.VALID.value] != self.valid_episode_count:
            raise ValueError('valid status count must equal valid_episode_count')
        if self.status_counts[ScoreStatus.INVALID_SCHEMA.value] < len(self.missing_episode_ids):
            raise ValueError('missing episode scores must count as invalid_schema')

        expected_validity = self.valid_episode_count / self.episode_count
        if not math.isclose(self.validity_rate, expected_validity, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('validity_rate is inconsistent with episode counts')

        reward_mean = self.valid_metric_means.get('reward')
        if self.valid_episode_count == 0:
            if self.valid_metric_means or self.valid_only_mean_reward is not None:
                raise ValueError('an all-invalid suite cannot contain valid-only metrics')
        elif reward_mean is None or self.valid_only_mean_reward is None:
            raise ValueError('a suite with valid episodes requires valid-only reward means')
        elif not math.isclose(self.valid_only_mean_reward, reward_mean, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('valid_only_mean_reward must equal the valid reward metric mean')
        expected_environment_reward = (
            (self.valid_only_mean_reward or 0.0) * self.valid_episode_count
            + self.invalid_episode_penalty * self.invalid_episode_count
        ) / self.episode_count
        if not math.isclose(
            self.all_episode_mean_environment_reward,
            expected_environment_reward,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('all_episode_mean_environment_reward is inconsistent with episode counts')
        return self


def make_suite_manifest(suite_id: str, bundles: Iterable[EpisodeBundle]) -> SuiteManifest:
    """Bind a task-homogeneous suite to episode, manifest, and label commitments."""

    ordered_bundles = tuple(sorted(bundles, key=lambda bundle: bundle.manifest.episode_id))
    if not ordered_bundles:
        raise ValueError('cannot create an empty suite manifest')
    episode_ids = [bundle.manifest.episode_id for bundle in ordered_bundles]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError('suite episode IDs must be unique')
    task_types = {bundle.manifest.task_type for bundle in ordered_bundles}
    if len(task_types) != 1:
        raise ValueError('suite episodes must use one homogeneous task_type')
    reward_versions = {bundle.manifest.reward_version for bundle in ordered_bundles}
    if len(reward_versions) != 1:
        raise ValueError('suite episodes must use one homogeneous reward_version')
    splits = {bundle.manifest.split for bundle in ordered_bundles}
    if len(splits) != 1:
        raise ValueError('suite episodes must belong to one homogeneous split')
    return SuiteManifest(
        suite_id=suite_id,
        task_type=task_types.pop(),
        reward_version=reward_versions.pop(),
        episodes=tuple(
            SuiteEpisodeBinding(
                episode_id=bundle.manifest.episode_id,
                task_type=bundle.manifest.task_type,
                reward_version=bundle.manifest.reward_version,
                manifest_sha256=bundle.manifest_sha256,
                labels_sha256=bundle.manifest.labels_sha256,
            )
            for bundle in ordered_bundles
        ),
    )


def suite_manifest_sha256(manifest: SuiteManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def aggregate_scores(manifest: SuiteManifest, scores: Iterable[EpisodeScore]) -> SuiteScore:
    """Aggregate trusted evaluator scores; absent suite scores receive `-1.0`.

    This function validates bindings and creates audit commitments, but it does not authenticate
    the origin of caller-created score objects. Participant-facing flows must score raw responses
    inside the private evaluator, as the `score-suite` CLI does.
    """

    ordered_scores = tuple(sorted(scores, key=lambda score: score.episode_id))
    score_ids = tuple(score.episode_id for score in ordered_scores)
    if len(score_ids) != len(set(score_ids)):
        raise ValueError('episode score IDs must be unique')

    binding_by_id = {binding.episode_id: binding for binding in manifest.episodes}
    extra_ids = set(score_ids) - binding_by_id.keys()
    if extra_ids:
        raise ValueError(f'episode scores are not present in the suite manifest: {sorted(extra_ids)}')
    score_by_id = {score.episode_id: score for score in ordered_scores}
    for score in ordered_scores:
        binding = binding_by_id[score.episode_id]
        if score.reward_version != manifest.reward_version:
            raise ValueError(f'episode {score.episode_id} has the wrong reward_version')
        if score.manifest_sha256 != binding.manifest_sha256 or score.labels_sha256 != binding.labels_sha256:
            raise ValueError(f'episode {score.episode_id} score is not bound to the suite manifest')

    episode_ids = tuple(binding.episode_id for binding in manifest.episodes)
    missing_episode_ids = tuple(episode_id for episode_id in episode_ids if episode_id not in score_by_id)
    valid_scores = tuple(score for score in ordered_scores if score.status == ScoreStatus.VALID)
    valid_metric_means = _macro_metric_means(valid_scores)
    numeric_valid_rewards = tuple(score.reward for score in valid_scores if score.reward is not None)
    if len(numeric_valid_rewards) != len(valid_scores):
        raise ValueError('valid episode scores must contain rewards')

    invalid_count = len(episode_ids) - len(valid_scores)
    environment_rewards = (*numeric_valid_rewards, *((INVALID_EPISODE_PENALTY,) * invalid_count))
    status_counter = Counter(score.status.value for score in ordered_scores)
    status_counter[ScoreStatus.INVALID_SCHEMA.value] += len(missing_episode_ids)
    status_counts = {
        status: status_counter[status] for status in sorted(known_status.value for known_status in ScoreStatus)
    }
    committed_inputs = [
        {
            'binding': binding.model_dump(mode='json'),
            'score': (
                score_by_id[binding.episode_id].model_dump(mode='json') if binding.episode_id in score_by_id else None
            ),
        }
        for binding in manifest.episodes
    ]

    return SuiteScore(
        suite_id=manifest.suite_id,
        task_type=manifest.task_type,
        reward_version=manifest.reward_version,
        suite_manifest_sha256=suite_manifest_sha256(manifest),
        input_scores_sha256=hashlib.sha256(canonical_json_bytes(committed_inputs)).hexdigest(),
        episode_ids=episode_ids,
        missing_episode_ids=missing_episode_ids,
        episode_count=len(episode_ids),
        valid_episode_count=len(valid_scores),
        invalid_episode_count=invalid_count,
        validity_rate=len(valid_scores) / len(episode_ids),
        status_counts=status_counts,
        valid_metric_means=valid_metric_means,
        all_episode_mean_environment_reward=math.fsum(environment_rewards) / len(episode_ids),
        valid_only_mean_reward=(
            math.fsum(numeric_valid_rewards) / len(numeric_valid_rewards) if numeric_valid_rewards else None
        ),
    )


def _macro_metric_means(scores: tuple[EpisodeScore, ...]) -> dict[str, float]:
    if not scores:
        return {}

    episode_metrics = tuple(score.metrics() for score in scores)
    metric_names = tuple(sorted(episode_metrics[0]))
    if any(tuple(sorted(metrics)) != metric_names for metrics in episode_metrics[1:]):
        raise ValueError('valid episode scores must expose a consistent metric set')

    return {
        metric_name: math.fsum(metrics[metric_name] for metrics in episode_metrics) / len(episode_metrics)
        for metric_name in metric_names
    }
