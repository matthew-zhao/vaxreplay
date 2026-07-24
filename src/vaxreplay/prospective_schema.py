"""Pure contracts for a prospectively sealed, pre-outcome challenge lifecycle.

These models deliberately use a decision-snapshot hash as the stable episode identity.  They do
not accept an episode-manifest hash or any label/outcome commitment because those values cannot
exist when a Tier A challenge and its model responses are sealed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import (
    CandidateAssessment,
    CandidateForecast,
    RewardVersion,
    Split,
    StrictModel,
    TaskType,
)
from vaxreplay.temporal_schema import DecisionSnapshotCommitment, model_sha256

PROSPECTIVE_EPISODE_BINDING_SCHEMA_VERSION = 'vaxreplay.prospective-episode-binding.v0.2'
PROSPECTIVE_SPLIT_INVENTORY_SCHEMA_VERSION = 'vaxreplay.prospective-split-inventory.v0.2'
PROSPECTIVE_SUITE_MANIFEST_SCHEMA_VERSION = 'vaxreplay.prospective-suite.v0.2'
PROSPECTIVE_CHALLENGE_ADMISSION_SCHEMA_VERSION = 'vaxreplay.prospective-challenge-admission.v0.3'
PROSPECTIVE_SUBMISSION_SCHEMA_VERSION = 'vaxreplay.prospective-submission.v0.1'
PROSPECTIVE_FINALIZATION_BINDING_SCHEMA_VERSION = 'vaxreplay.prospective-finalization-binding.v0.3'
PROSPECTIVE_ATTEMPT_POLICY_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-policy.v0.1'
PROSPECTIVE_RESPONSE_PROTOCOL = 'vaxreplay.prospective-submission-json-stdout.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _validate_bindings(
    bindings: tuple[ProspectiveEpisodeBinding, ...],
    *,
    require_homogeneous_profile: bool,
) -> None:
    episode_ids = tuple(binding.episode_id for binding in bindings)
    if episode_ids != tuple(sorted(episode_ids)):
        raise ValueError('prospective episode bindings must be sorted by episode_id')
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError('prospective episode IDs must be unique')

    lineage_ids = tuple(binding.lineage_group_id for binding in bindings)
    if len(lineage_ids) != len(set(lineage_ids)):
        raise ValueError('prospective lineage_group_ids must be unique')

    decision_hashes = tuple(binding.decision_snapshot_sha256 for binding in bindings)
    if len(decision_hashes) != len(set(decision_hashes)):
        raise ValueError('prospective decision snapshot hashes must be unique')

    if require_homogeneous_profile:
        profiles = {(binding.task_type, binding.reward_version, binding.split) for binding in bindings}
        if len(profiles) != 1:
            raise ValueError('prospective suite episodes must use one task, reward version, and split')


class ProspectiveAttemptPolicy(StrictModel):
    """The Tier A execution rule committed before a system sees the cohort."""

    schema_version: Literal['vaxreplay.prospective-attempt-policy.v0.1'] = PROSPECTIVE_ATTEMPT_POLICY_SCHEMA_VERSION
    maximum_attempts: Literal[1] = 1
    admitted_attempt_number: Literal[1] = 1
    selection_rule: Literal['first_and_only_started_run'] = 'first_and_only_started_run'
    retry_on_failure: Literal[False] = False
    failure_disposition: Literal['retain_as_invalid'] = 'retain_as_invalid'
    conflicting_attempt_disposition: Literal['invalidate_system_entry'] = 'invalidate_system_entry'


def prospective_attempt_policy_sha256(policy: ProspectiveAttemptPolicy) -> str:
    return hashlib.sha256(canonical_json_bytes(policy)).hexdigest()


class ProspectiveEpisodeBinding(StrictModel):
    """Stable, label-free identity for one decision-time episode."""

    schema_version: Literal['vaxreplay.prospective-episode-binding.v0.2'] = PROSPECTIVE_EPISODE_BINDING_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    lineage_group_id: str = Field(min_length=1)
    task_type: TaskType
    reward_version: RewardVersion
    split: Split
    decision_at: datetime
    decision_snapshot: DecisionSnapshotCommitment
    decision_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_context_bytes: int = Field(gt=0)

    @classmethod
    def from_decision_snapshot(
        cls,
        snapshot: DecisionSnapshotCommitment,
        *,
        decision_context_sha256: str,
        decision_context_bytes: int,
    ) -> ProspectiveEpisodeBinding:
        config = snapshot.config
        return cls(
            episode_id=config.episode_id,
            lineage_group_id=config.lineage_group_id,
            task_type=config.task_type,
            reward_version=config.reward_version,
            split=config.split,
            decision_at=config.decision_at,
            decision_snapshot=snapshot,
            decision_snapshot_sha256=model_sha256(snapshot),
            decision_context_sha256=decision_context_sha256,
            decision_context_bytes=decision_context_bytes,
        )

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return _aware(value, 'decision_at')

    @model_validator(mode='after')
    def validate_decision_binding(self) -> Self:
        config = self.decision_snapshot.config
        observed_profile = (
            self.episode_id,
            self.lineage_group_id,
            self.task_type,
            self.reward_version,
            self.split,
            self.decision_at,
        )
        expected_profile = (
            config.episode_id,
            config.lineage_group_id,
            config.task_type,
            config.reward_version,
            config.split,
            config.decision_at,
        )
        if observed_profile != expected_profile:
            raise ValueError('prospective episode binding does not match its decision snapshot config')
        if config.synthetic:
            raise ValueError('prospective Tier A episode bindings cannot be synthetic')
        if self.decision_snapshot_sha256 != model_sha256(self.decision_snapshot):
            raise ValueError('decision_snapshot_sha256 does not bind the canonical decision snapshot')
        return self

    @property
    def first_forecast_maturity_at(self) -> datetime:
        return min(
            self.decision_at + timedelta(days=target.horizon_days)
            for target in self.decision_snapshot.config.forecast_targets
        )


class ProspectiveSplitInventory(StrictModel):
    """Complete, pre-outcome lineage-to-split assignment across the release."""

    schema_version: Literal['vaxreplay.prospective-split-inventory.v0.2'] = PROSPECTIVE_SPLIT_INVENTORY_SCHEMA_VERSION
    inventory_id: str = Field(min_length=1)
    inventory_complete: Literal[True] = True
    episodes: tuple[ProspectiveEpisodeBinding, ...] = Field(min_length=1)

    @field_validator('episodes')
    @classmethod
    def validate_episodes(
        cls,
        value: tuple[ProspectiveEpisodeBinding, ...],
    ) -> tuple[ProspectiveEpisodeBinding, ...]:
        _validate_bindings(value, require_homogeneous_profile=False)
        return value


class ProspectiveSuiteManifest(StrictModel):
    """A homogeneous prospective test suite whose identity excludes future labels."""

    schema_version: Literal['vaxreplay.prospective-suite.v0.2'] = PROSPECTIVE_SUITE_MANIFEST_SCHEMA_VERSION
    suite_id: str = Field(min_length=1)
    task_type: TaskType
    reward_version: RewardVersion
    split: Literal[Split.TEST] = Split.TEST
    episodes: tuple[ProspectiveEpisodeBinding, ...] = Field(min_length=1)

    @field_validator('episodes')
    @classmethod
    def validate_episodes(
        cls,
        value: tuple[ProspectiveEpisodeBinding, ...],
    ) -> tuple[ProspectiveEpisodeBinding, ...]:
        _validate_bindings(value, require_homogeneous_profile=True)
        return value

    @model_validator(mode='after')
    def validate_suite_profile(self) -> Self:
        expected = (self.task_type, self.reward_version, self.split)
        if any((binding.task_type, binding.reward_version, binding.split) != expected for binding in self.episodes):
            raise ValueError('prospective episode profile does not match its suite')
        return self


class ProspectiveChallengeAdmission(StrictModel):
    """Public commitment to a complete prospective challenge before outcomes exist.

    ``official_benchmark`` is emitted only by the promotion-backed admission path
    that freshly replays required hermetic evidence. Generic source-verifier
    callbacks produce the explicitly non-Tier-A ``prospective_research`` profile.
    """

    schema_version: Literal['vaxreplay.prospective-challenge-admission.v0.3'] = (
        PROSPECTIVE_CHALLENGE_ADMISSION_SCHEMA_VERSION
    )
    release_id: str = Field(min_length=1)
    purpose: Literal['official_benchmark', 'prospective_research'] = 'prospective_research'
    suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_inventory_complete: Literal[True] = True
    case_universe_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_inventory_complete: Literal[True] = True
    verifier_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_capture_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    eligibility_protocol_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_deadline_at: datetime
    episodes: tuple[ProspectiveEpisodeBinding, ...] = Field(min_length=1)

    @field_validator('run_deadline_at')
    @classmethod
    def validate_run_deadline_at(cls, value: datetime) -> datetime:
        return _aware(value, 'run_deadline_at')

    @field_validator('episodes')
    @classmethod
    def validate_episodes(
        cls,
        value: tuple[ProspectiveEpisodeBinding, ...],
    ) -> tuple[ProspectiveEpisodeBinding, ...]:
        _validate_bindings(value, require_homogeneous_profile=True)
        return value

    @model_validator(mode='after')
    def validate_admission_window(self) -> Self:
        if any(binding.split != Split.TEST for binding in self.episodes):
            raise ValueError('prospective challenge episodes must use the test split')
        if self.run_deadline_at <= max(binding.decision_at for binding in self.episodes):
            raise ValueError('run_deadline_at must be after every episode decision_at')
        if self.run_deadline_at >= min(binding.first_forecast_maturity_at for binding in self.episodes):
            raise ValueError('run_deadline_at must precede every episode first forecast maturity')
        return self


class ProspectiveSubmission(StrictModel):
    """A model response bound to immutable decision state, never to future labels."""

    schema_version: Literal['vaxreplay.prospective-submission.v0.1'] = PROSPECTIVE_SUBMISSION_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    decision_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    ranking: tuple[str, ...] = Field(min_length=1)
    forecasts: tuple[CandidateForecast, ...] = Field(min_length=1)
    assessments: tuple[CandidateAssessment, ...] = ()

    @field_validator('ranking')
    @classmethod
    def validate_ranking(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError('prospective submission ranking candidate IDs must be unique')
        return value

    @model_validator(mode='after')
    def validate_response_keys(self) -> Self:
        ranking_ids = set(self.ranking)
        forecast_keys = tuple(
            (forecast.candidate_id, forecast.target_id, forecast.horizon_days) for forecast in self.forecasts
        )
        if len(forecast_keys) != len(set(forecast_keys)):
            raise ValueError('prospective submission forecasts must use unique candidate/target/horizon keys')
        if any(forecast.candidate_id not in ranking_ids for forecast in self.forecasts):
            raise ValueError('prospective submission forecasts must reference ranked candidates')

        assessment_keys = tuple((assessment.candidate_id, assessment.dimension) for assessment in self.assessments)
        if len(assessment_keys) != len(set(assessment_keys)):
            raise ValueError('prospective submission assessments must use unique candidate/dimension keys')
        if any(assessment.candidate_id not in ranking_ids for assessment in self.assessments):
            raise ValueError('prospective submission assessments must reference ranked candidates')
        return self

    def require_episode(self, binding: ProspectiveEpisodeBinding) -> Self:
        """Fail closed unless the response exactly covers its sealed decision-time contract."""

        if self.episode_id != binding.episode_id:
            raise ValueError('prospective submission episode_id does not match the sealed episode')
        if self.decision_snapshot_sha256 != binding.decision_snapshot_sha256:
            raise ValueError('prospective submission does not match the sealed decision snapshot')

        config = binding.decision_snapshot.config
        expected_candidate_ids = tuple(config.candidate_ids)
        if len(self.ranking) != len(expected_candidate_ids) or set(self.ranking) != set(expected_candidate_ids):
            raise ValueError('prospective submission ranking must cover every sealed candidate exactly once')
        expected_forecasts = {
            (candidate_id, target.target_id, target.horizon_days)
            for candidate_id in expected_candidate_ids
            for target in config.forecast_targets
        }
        actual_forecasts = {
            (forecast.candidate_id, forecast.target_id, forecast.horizon_days) for forecast in self.forecasts
        }
        if actual_forecasts != expected_forecasts or len(actual_forecasts) != len(self.forecasts):
            raise ValueError('prospective submission forecasts must cover every sealed candidate/target pair')
        expected_assessments = {
            (candidate_id, dimension)
            for candidate_id in self.ranking[: config.portfolio_size]
            for dimension in config.required_dimensions
        }
        actual_assessments = {(assessment.candidate_id, assessment.dimension) for assessment in self.assessments}
        if actual_assessments != expected_assessments or len(actual_assessments) != len(self.assessments):
            raise ValueError('prospective submission assessments must cover each top-portfolio dimension')
        return self


class ProspectiveFinalizationBinding(StrictModel):
    """Post-outcome crosswalk without embedding mutable final or outcome material."""

    schema_version: Literal['vaxreplay.prospective-finalization-binding.v0.3'] = (
        PROSPECTIVE_FINALIZATION_BINDING_SCHEMA_VERSION
    )
    release_id: str = Field(min_length=1)
    purpose: Literal['official_benchmark', 'prospective_research']
    episode_id: str = Field(min_length=1)
    prospective_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    temporal_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_selection_audit_sha256: str = Field(pattern=_SHA256_PATTERN)
    finalized_at: datetime

    @field_validator('finalized_at')
    @classmethod
    def validate_finalized_at(cls, value: datetime) -> datetime:
        return _aware(value, 'finalized_at')

    def require_episode(
        self,
        admission: ProspectiveChallengeAdmission,
        binding: ProspectiveEpisodeBinding,
    ) -> Self:
        """Verify the prospective half of a later organizer-controlled finalization."""

        if self.release_id != admission.release_id:
            raise ValueError('finalization release_id does not match the prospective admission')
        if self.purpose != admission.purpose:
            raise ValueError('finalization purpose does not match the prospective admission')
        if self.prospective_admission_sha256 != prospective_challenge_admission_sha256(admission):
            raise ValueError('finalization does not bind the canonical prospective admission')
        admitted_by_id = {episode.episode_id: episode for episode in admission.episodes}
        if admitted_by_id.get(self.episode_id) != binding:
            raise ValueError('finalization episode is absent from the prospective admission')
        if self.episode_id != binding.episode_id:
            raise ValueError('finalization episode_id does not match the prospective episode')
        if self.decision_snapshot_sha256 != binding.decision_snapshot_sha256:
            raise ValueError('finalization changed the sealed decision snapshot identity')
        if self.decision_context_sha256 != binding.decision_context_sha256:
            raise ValueError('finalization changed the sealed decision-context lineage')
        if self.finalized_at <= admission.run_deadline_at:
            raise ValueError('finalization must occur after the prospective run deadline')
        return self


def prospective_split_inventory_sha256(inventory: ProspectiveSplitInventory) -> str:
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def prospective_suite_manifest_sha256(manifest: ProspectiveSuiteManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def prospective_challenge_admission_sha256(admission: ProspectiveChallengeAdmission) -> str:
    return hashlib.sha256(canonical_json_bytes(admission)).hexdigest()
