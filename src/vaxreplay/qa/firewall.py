"""Two-phase reward quarantine and signed release for optimizer consumption."""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_schema import ScoreStatus, StrictModel, Submission
from vaxreplay.prompt import build_episode_prompt, build_system_prompt
from vaxreplay.qa.admission import (
    AdmissionTokenConsumer,
    verify_gradient_admission_token,
)
from vaxreplay.qa.components import ComponentFloor, require_component_floors
from vaxreplay.qa.parsing import SubmissionParseError, parse_submission
from vaxreplay.qa.schema import (
    GradientAdmissionToken,
    RewardContract,
    TrainingRunAdmission,
    reward_contract_sha256,
)
from vaxreplay.qa.score_integrity import (
    AnyScoreVector,
    ScoreEvaluator,
    ScoreIntegrityError,
    differential_score,
    require_score_formula,
    revalidate_score,
    revalidate_submission,
    score_vector_sha256,
)

QUARANTINED_TRAJECTORY_SCHEMA_VERSION = 'vaxreplay.quarantined-trajectory.v0.1'
QUARANTINED_BATCH_SCHEMA_VERSION = 'vaxreplay.quarantined-training-batch.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_AUDIT_TRACE_BYTES = 8 * 1024 * 1024


class RewardFirewallReason(str, enum.Enum):
    STRICT_PARSE = 'strict_parse'
    SCORE_INTEGRITY = 'score_integrity'
    INVALID_SCORE = 'invalid_score'
    COMPONENT_FLOOR = 'component_floor'
    BATCH_BINDING = 'batch_binding'
    ADMISSION = 'admission'


class RewardFirewallQuarantine(ValueError):
    """A trajectory or batch that must not reach an optimizer."""

    def __init__(self, reason: RewardFirewallReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f'{reason.value}: {detail}')


class QuarantinedTrajectoryEnvelope(StrictModel):
    schema_version: Literal['vaxreplay.quarantined-trajectory.v0.1'] = QUARANTINED_TRAJECTORY_SCHEMA_VERSION
    trajectory_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    episode_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    labels_sha256: str = Field(pattern=_SHA256_PATTERN)
    reward_version: str = Field(min_length=1)
    response_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    audit_trace_sha256: str = Field(pattern=_SHA256_PATTERN)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    score_sha256: str = Field(pattern=_SHA256_PATTERN)
    reward_contract_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class QuarantinedTrajectory:
    envelope: QuarantinedTrajectoryEnvelope
    raw_response: bytes
    prompt_artifact: bytes
    audit_trace: bytes
    submission: Submission
    score: AnyScoreVector


@dataclass(frozen=True, slots=True)
class ReplayScorerPair:
    """Exact episode and independently identified scorers used for release replay."""

    bundle: EpisodeBundle
    primary_evaluator: ScoreEvaluator
    reference_evaluator: ScoreEvaluator
    primary_scorer_sha256: str
    reference_scorer_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ('primary_scorer_sha256', self.primary_scorer_sha256),
            ('reference_scorer_sha256', self.reference_scorer_sha256),
        ):
            if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
                raise ValueError(f'{name} must be a lowercase SHA-256 digest')
        if self.primary_scorer_sha256 == self.reference_scorer_sha256:
            raise ValueError('release replay requires distinct primary and reference scorer builds')


class QuarantinedBatchEnvelope(StrictModel):
    schema_version: Literal['vaxreplay.quarantined-training-batch.v0.1'] = QUARANTINED_BATCH_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    trajectories: tuple[QuarantinedTrajectoryEnvelope, ...] = Field(min_length=1)
    trajectory_batch_sha256: str = Field(pattern=_SHA256_PATTERN)
    reward_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    reward_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_manifest_sha256s: tuple[str, ...] = Field(min_length=1)

    @field_validator('trajectories')
    @classmethod
    def validate_trajectories(
        cls,
        value: tuple[QuarantinedTrajectoryEnvelope, ...],
    ) -> tuple[QuarantinedTrajectoryEnvelope, ...]:
        ids = tuple(item.trajectory_id for item in value)
        if len(ids) != len(set(ids)):
            raise ValueError('quarantined trajectory IDs must be unique')
        if ids != tuple(sorted(ids)):
            raise ValueError('quarantined trajectories must be sorted by trajectory_id')
        return value

    @field_validator('episode_manifest_sha256s')
    @classmethod
    def validate_manifests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or value != tuple(sorted(value)):
            raise ValueError('episode manifest hashes must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_bindings(self) -> Self:
        if {item.run_id for item in self.trajectories} != {self.run_id}:
            raise ValueError('every quarantined trajectory must bind the batch run_id')
        if {item.reward_contract_sha256 for item in self.trajectories} != {self.reward_contract_sha256}:
            raise ValueError('every quarantined trajectory must bind the batch reward contract')
        if tuple(sorted({item.episode_manifest_sha256 for item in self.trajectories})) != (
            self.episode_manifest_sha256s
        ):
            raise ValueError('episode manifest inventory does not match quarantined trajectories')
        expected_batch = _trajectory_batch_sha256(self.trajectories)
        if self.trajectory_batch_sha256 != expected_batch:
            raise ValueError('trajectory_batch_sha256 does not bind the trajectory envelopes')
        return self


@dataclass(frozen=True, slots=True)
class QuarantinedBatch:
    envelope: QuarantinedBatchEnvelope
    trajectories: tuple[QuarantinedTrajectory, ...]


class ReleasedTrainingReward(StrictModel):
    trajectory_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    score_sha256: str = Field(pattern=_SHA256_PATTERN)
    gradient_admission_token_id: str = Field(pattern=r'^[0-9a-f]{32}$')


class TrainingRewardFirewall:
    """Prepare exact scored trajectories without exposing optimizer reward."""

    def __init__(
        self,
        *,
        run_id: str,
        bundle: EpisodeBundle,
        primary_evaluator: ScoreEvaluator,
        reference_evaluator: ScoreEvaluator,
        primary_scorer_sha256: str,
        reference_scorer_sha256: str,
        reward_contract: RewardContract,
    ) -> None:
        if not run_id:
            raise ValueError('reward firewall run_id must be non-empty')
        if reward_contract.reward_version != bundle.manifest.reward_version:
            raise ValueError('reward contract version does not match the episode')
        if tuple(sorted(bundle.manifest.required_dimensions)) != reward_contract.required_dimensions:
            raise ValueError('reward contract dimensions do not match the episode')
        if primary_scorer_sha256 != reward_contract.scorer_sha256:
            raise ValueError('primary scorer build does not match the reward contract')
        if reference_scorer_sha256 != reward_contract.reference_scorer_sha256:
            raise ValueError('reference scorer build does not match the reward contract')
        self._run_id = run_id
        self._bundle = bundle
        self._primary = primary_evaluator
        self._reference = reference_evaluator
        self._primary_scorer_sha256 = primary_scorer_sha256
        self._reference_scorer_sha256 = reference_scorer_sha256
        self._contract = reward_contract
        self._contract_sha256 = reward_contract_sha256(reward_contract)
        self._bundle_manifest_sha256 = bundle.manifest_sha256
        self._prompt_bytes = _prompt_artifact(bundle)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def episode_id(self) -> str:
        return self._bundle.manifest.episode_id

    @property
    def episode_manifest_sha256(self) -> str:
        return self._bundle.manifest_sha256

    @property
    def reward_contract_sha256(self) -> str:
        return self._contract_sha256

    def quarantine(
        self,
        *,
        trajectory_id: str,
        response: bytes | str,
        audit_trace: bytes,
    ) -> QuarantinedTrajectory:
        """Strictly parse and score, retaining reward only in quarantine."""

        if not trajectory_id:
            raise ValueError('trajectory_id must be non-empty')
        if not isinstance(audit_trace, bytes) or not audit_trace:
            raise ValueError('audit_trace must be non-empty bytes from the trajectory recorder')
        if len(audit_trace) > _MAX_AUDIT_TRACE_BYTES:
            raise ValueError(f'audit_trace exceeds the {_MAX_AUDIT_TRACE_BYTES}-byte quarantine limit')
        try:
            submission = parse_submission(response)
        except SubmissionParseError as error:
            raise RewardFirewallQuarantine(
                RewardFirewallReason.STRICT_PARSE,
                f'{error.reason.value}: {error.detail}',
            ) from error
        try:
            score = differential_score(
                self._bundle,
                submission,
                self._primary,
                self._reference,
            )
        except ScoreIntegrityError as error:
            raise RewardFirewallQuarantine(
                RewardFirewallReason.SCORE_INTEGRITY,
                f'{error.reason.value}: {error.detail}',
            ) from error
        if (
            self._bundle.manifest_sha256 != self._bundle_manifest_sha256
            or _prompt_artifact(self._bundle) != self._prompt_bytes
        ):
            raise RewardFirewallQuarantine(
                RewardFirewallReason.SCORE_INTEGRITY,
                'a scorer mutated the episode bundle or prompt material',
            )
        if score.status != ScoreStatus.VALID or score.reward is None:
            raise RewardFirewallQuarantine(
                RewardFirewallReason.INVALID_SCORE,
                f'score status {score.status.value} is not training-eligible',
            )
        floors = tuple(
            ComponentFloor(metric=item.metric, minimum=item.minimum) for item in self._contract.component_floors
        )
        try:
            require_component_floors(score, floors)
        except ValueError as error:
            raise RewardFirewallQuarantine(
                RewardFirewallReason.COMPONENT_FLOOR,
                str(error),
            ) from error

        response_bytes = response if isinstance(response, bytes) else response.encode('utf-8')
        prompt_artifact = self._prompt_bytes
        envelope = QuarantinedTrajectoryEnvelope(
            trajectory_id=trajectory_id,
            run_id=self._run_id,
            episode_id=self._bundle.manifest.episode_id,
            episode_manifest_sha256=self._bundle.manifest_sha256,
            labels_sha256=self._bundle.manifest.labels_sha256,
            reward_version=self._bundle.manifest.reward_version,
            response_sha256=hashlib.sha256(response_bytes).hexdigest(),
            prompt_artifact_sha256=hashlib.sha256(prompt_artifact).hexdigest(),
            audit_trace_sha256=hashlib.sha256(audit_trace).hexdigest(),
            submission_sha256=hashlib.sha256(canonical_json_bytes(submission)).hexdigest(),
            score_sha256=score_vector_sha256(score),
            reward_contract_sha256=self._contract_sha256,
        )
        return QuarantinedTrajectory(
            envelope=envelope,
            raw_response=response_bytes,
            prompt_artifact=prompt_artifact,
            audit_trace=audit_trace,
            submission=submission,
            score=score,
        )


def build_quarantined_batch(
    trajectories: Iterable[QuarantinedTrajectory],
) -> QuarantinedBatch:
    ordered = tuple(sorted(trajectories, key=lambda item: item.envelope.trajectory_id))
    if not ordered:
        raise ValueError('a quarantined batch requires at least one trajectory')
    for trajectory in ordered:
        _require_trajectory_artifact_integrity(trajectory)
    envelopes = tuple(item.envelope for item in ordered)
    run_ids = {item.run_id for item in envelopes}
    contract_hashes = {item.reward_contract_sha256 for item in envelopes}
    if len(run_ids) != 1:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            'every trajectory in a batch must bind the same run_id',
        )
    if len(contract_hashes) != 1:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            'every trajectory in a batch must bind the same reward contract',
        )
    reward_artifact_sha256 = _reward_artifact_sha256(ordered)
    envelope = QuarantinedBatchEnvelope(
        run_id=next(iter(run_ids)),
        trajectories=envelopes,
        trajectory_batch_sha256=_trajectory_batch_sha256(envelopes),
        reward_artifact_sha256=reward_artifact_sha256,
        reward_contract_sha256=next(iter(contract_hashes)),
        episode_manifest_sha256s=tuple(sorted({item.episode_manifest_sha256 for item in envelopes})),
    )
    return QuarantinedBatch(envelope=envelope, trajectories=ordered)


def release_training_batch(
    batch: QuarantinedBatch,
    *,
    reward_contract: RewardContract,
    admission: TrainingRunAdmission,
    token: GradientAdmissionToken,
    trusted_public_key_bytes: bytes,
    now: datetime,
    consume_token: AdmissionTokenConsumer,
    expected_model_sha256: str,
    expected_harness_sha256: str,
    expected_tool_policy_sha256: str,
    expected_environment_sha256: str,
    expected_dataset_sha256: str,
    expected_optimizer_config_sha256: str,
    replay_scorers: Mapping[str, ReplayScorerPair],
) -> tuple[ReleasedTrainingReward, ...]:
    """Release rewards only after exact local replay and signed batch admission."""

    replayed_scores = _require_batch_integrity(batch, reward_contract, replay_scorers)
    envelope = batch.envelope
    try:
        verify_gradient_admission_token(
            token,
            admission,
            trusted_public_key_bytes,
            now=now,
            consume_token=consume_token,
            expected_run_id=envelope.run_id,
            expected_trajectory_batch_sha256=envelope.trajectory_batch_sha256,
            expected_reward_artifact_sha256=envelope.reward_artifact_sha256,
            expected_model_sha256=expected_model_sha256,
            expected_harness_sha256=expected_harness_sha256,
            expected_tool_policy_sha256=expected_tool_policy_sha256,
            expected_environment_sha256=expected_environment_sha256,
            expected_dataset_sha256=expected_dataset_sha256,
            expected_optimizer_config_sha256=expected_optimizer_config_sha256,
            expected_reward_contract_sha256=envelope.reward_contract_sha256,
            expected_episode_manifest_sha256s=envelope.episode_manifest_sha256s,
        )
    except ValueError as error:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.ADMISSION,
            str(error),
        ) from error

    released: list[ReleasedTrainingReward] = []
    for trajectory, replayed_score in zip(batch.trajectories, replayed_scores, strict=True):
        assert replayed_score.reward is not None
        released.append(
            ReleasedTrainingReward(
                trajectory_id=trajectory.envelope.trajectory_id,
                episode_id=trajectory.envelope.episode_id,
                reward=replayed_score.reward,
                score_sha256=trajectory.envelope.score_sha256,
                gradient_admission_token_id=token.token_id,
            )
        )
    return tuple(released)


def _require_batch_integrity(
    batch: QuarantinedBatch,
    reward_contract: RewardContract,
    replay_scorers: Mapping[str, ReplayScorerPair],
) -> tuple[AnyScoreVector, ...]:
    try:
        reparsed = QuarantinedBatchEnvelope.model_validate_json(canonical_json_bytes(batch.envelope))
        validated_contract = RewardContract.model_validate_json(canonical_json_bytes(reward_contract))
    except (TypeError, ValueError) as error:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'batch or reward contract failed canonical replay validation: {error}',
        ) from error
    if reparsed != batch.envelope:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            'batch envelope changed during replay validation',
        )
    if reward_contract_sha256(validated_contract) != batch.envelope.reward_contract_sha256:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            'supplied reward contract differs from the quarantined batch',
        )
    actual_envelopes = tuple(item.envelope for item in batch.trajectories)
    if actual_envelopes != batch.envelope.trajectories:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            'batch trajectory objects differ from the bound envelope inventory',
        )
    replayed_scores: list[AnyScoreVector] = []
    for trajectory in batch.trajectories:
        _require_trajectory_artifact_integrity(trajectory)
        envelope = trajectory.envelope
        scorer_pair = replay_scorers.get(envelope.episode_manifest_sha256)
        if scorer_pair is None:
            raise RewardFirewallQuarantine(
                RewardFirewallReason.BATCH_BINDING,
                f'trajectory {envelope.trajectory_id} has no release replay scorer pair',
            )
        if (
            scorer_pair.bundle.manifest_sha256 != envelope.episode_manifest_sha256
            or scorer_pair.bundle.manifest.episode_id != envelope.episode_id
            or scorer_pair.bundle.manifest.labels_sha256 != envelope.labels_sha256
        ):
            raise RewardFirewallQuarantine(
                RewardFirewallReason.BATCH_BINDING,
                f'trajectory {envelope.trajectory_id} replay bundle binding changed',
            )
        if (
            scorer_pair.primary_scorer_sha256 != validated_contract.scorer_sha256
            or scorer_pair.reference_scorer_sha256 != validated_contract.reference_scorer_sha256
        ):
            raise RewardFirewallQuarantine(
                RewardFirewallReason.BATCH_BINDING,
                f'trajectory {envelope.trajectory_id} replay scorer build binding changed',
            )
        expected_prompt = _prompt_artifact(scorer_pair.bundle)
        expected_manifest_sha256 = scorer_pair.bundle.manifest_sha256
        if trajectory.prompt_artifact != expected_prompt:
            raise RewardFirewallQuarantine(
                RewardFirewallReason.BATCH_BINDING,
                f'trajectory {envelope.trajectory_id} prompt artifact differs from the replay bundle',
            )
        try:
            parsed = parse_submission(trajectory.raw_response)
            if canonical_json_bytes(parsed) != canonical_json_bytes(revalidate_submission(trajectory.submission)):
                raise ValueError('raw response parses to a different submission')
            replayed_score = differential_score(
                scorer_pair.bundle,
                parsed,
                scorer_pair.primary_evaluator,
                scorer_pair.reference_evaluator,
            )
            if (
                scorer_pair.bundle.manifest_sha256 != expected_manifest_sha256
                or _prompt_artifact(scorer_pair.bundle) != expected_prompt
            ):
                raise ValueError('a replay scorer mutated the episode bundle or prompt material')
            if canonical_json_bytes(replayed_score) != canonical_json_bytes(revalidate_score(trajectory.score)):
                raise ValueError('independent score replay differs from the quarantined score')
            require_component_floors(
                replayed_score,
                tuple(
                    ComponentFloor(metric=item.metric, minimum=item.minimum)
                    for item in validated_contract.component_floors
                ),
            )
        except (SubmissionParseError, ScoreIntegrityError, ValueError) as error:
            raise RewardFirewallQuarantine(
                RewardFirewallReason.BATCH_BINDING,
                f'trajectory {envelope.trajectory_id} failed score replay: {error}',
            ) from error
        replayed_scores.append(replayed_score)
    if _reward_artifact_sha256(batch.trajectories) != batch.envelope.reward_artifact_sha256:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            'reward artifact hash changed',
        )
    return tuple(replayed_scores)


def _require_trajectory_artifact_integrity(
    trajectory: QuarantinedTrajectory,
) -> None:
    envelope = trajectory.envelope
    try:
        validated_envelope = QuarantinedTrajectoryEnvelope.model_validate_json(canonical_json_bytes(envelope))
        validated_submission = revalidate_submission(trajectory.submission)
        validated_score = revalidate_score(trajectory.score)
    except (TypeError, ValueError) as error:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {getattr(envelope, "trajectory_id", "<unknown>")} failed schema replay',
        ) from error
    if validated_envelope != envelope:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {envelope.trajectory_id} envelope changed during schema replay',
        )
    byte_artifacts = (
        ('raw response', trajectory.raw_response, envelope.response_sha256),
        ('prompt artifact', trajectory.prompt_artifact, envelope.prompt_artifact_sha256),
        ('audit trace', trajectory.audit_trace, envelope.audit_trace_sha256),
    )
    for name, artifact, expected_sha256 in byte_artifacts:
        if not isinstance(artifact, bytes):
            raise RewardFirewallQuarantine(
                RewardFirewallReason.BATCH_BINDING,
                f'trajectory {envelope.trajectory_id} {name} is not immutable bytes',
            )
        if hashlib.sha256(artifact).hexdigest() != expected_sha256:
            raise RewardFirewallQuarantine(
                RewardFirewallReason.BATCH_BINDING,
                f'trajectory {envelope.trajectory_id} {name} hash changed',
            )
    if not trajectory.audit_trace or len(trajectory.audit_trace) > _MAX_AUDIT_TRACE_BYTES:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {envelope.trajectory_id} audit trace violates size constraints',
        )
    try:
        reparsed_submission = parse_submission(trajectory.raw_response)
    except SubmissionParseError as error:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {envelope.trajectory_id} raw response no longer parses',
        ) from error
    if canonical_json_bytes(reparsed_submission) != canonical_json_bytes(validated_submission):
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {envelope.trajectory_id} raw response and submission differ',
        )
    if hashlib.sha256(canonical_json_bytes(validated_submission)).hexdigest() != envelope.submission_sha256:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {envelope.trajectory_id} submission hash changed',
        )
    if score_vector_sha256(validated_score) != envelope.score_sha256:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {envelope.trajectory_id} score hash changed',
        )
    if (
        validated_score.episode_id != envelope.episode_id
        or validated_score.manifest_sha256 != envelope.episode_manifest_sha256
        or validated_score.labels_sha256 != envelope.labels_sha256
        or validated_score.reward_version != envelope.reward_version
    ):
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {envelope.trajectory_id} score binding changed',
        )
    try:
        require_score_formula(validated_score)
    except ValueError as error:
        raise RewardFirewallQuarantine(
            RewardFirewallReason.BATCH_BINDING,
            f'trajectory {envelope.trajectory_id} score formula changed',
        ) from error


def _trajectory_batch_sha256(
    envelopes: Sequence[QuarantinedTrajectoryEnvelope],
) -> str:
    return hashlib.sha256(canonical_json_bytes([item.model_dump(mode='json') for item in envelopes])).hexdigest()


def _reward_artifact_sha256(
    trajectories: Sequence[QuarantinedTrajectory],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    'trajectory_id': item.envelope.trajectory_id,
                    'score': revalidate_score(item.score).model_dump(mode='json'),
                }
                for item in trajectories
            ]
        )
    ).hexdigest()


def _prompt_artifact(bundle: EpisodeBundle) -> bytes:
    return canonical_json_bytes(
        [
            {'role': 'system', 'content': build_system_prompt(bundle)},
            {'role': 'user', 'content': build_episode_prompt(bundle)},
        ]
    )
