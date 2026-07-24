"""Framework-neutral one-turn RL environment for VaxReplay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import ScoreStatus, ScoreVector, Split, Submission
from vaxreplay.prompt import build_episode_prompt, build_system_prompt
from vaxreplay.qa.parsing import SubmissionParseError, parse_submission
from vaxreplay.ranking_schema import ScoreVectorV1


class SubmissionEvaluator(Protocol):
    def score(self, submission: Submission) -> ScoreVector | ScoreVectorV1: ...


@dataclass(frozen=True)
class EnvironmentStep:
    reporting_reward: float
    done: bool
    metrics: dict[str, float]
    info: dict[str, Any]

    @property
    def reward(self) -> float:
        raise RuntimeError(
            'VaxReplayEnvironment is evaluation-only; reporting_reward must never be sent '
            'to an optimizer. Use vaxreplay.qa.release_training_batch for training.'
        )


class VaxReplayEnvironment:
    """Evaluation-only one-shot environment for smoke tests and metric reporting.

    This class deliberately has no usable ``reward`` attribute. Optimizer-facing
    code must use the signed, whole-batch reward firewall.
    """

    def __init__(self, bundle: EpisodeBundle, evaluator: SubmissionEvaluator):
        if bundle.manifest.split == Split.TEST:
            raise ValueError('sealed test episodes require a one-shot external evaluator, not an interactive RL env')
        self._bundle = bundle
        self._evaluator = evaluator

    def reset(self) -> list[dict[str, str]]:
        return [
            {'role': 'system', 'content': build_system_prompt(self._bundle)},
            {'role': 'user', 'content': build_episode_prompt(self._bundle)},
        ]

    def step(self, response_text: bytes | str) -> EnvironmentStep:
        try:
            submission = parse_submission(response_text)
        except SubmissionParseError as error:
            return EnvironmentStep(
                reporting_reward=-1.0,
                done=True,
                metrics={'valid_submission': 0.0},
                info={
                    'episode_id': self._bundle.manifest.episode_id,
                    'status': ScoreStatus.INVALID_SCHEMA.value,
                    'issue_codes': [f'STRICT_PARSE_{error.reason.value.upper()}'],
                    'parse_reason': error.reason.value,
                    **(
                        {'validation_error_count': error.validation_error_count}
                        if error.validation_error_count is not None
                        else {}
                    ),
                },
            )

        score = self._evaluator.score(submission)
        self._validate_score_binding(score)
        return EnvironmentStep(
            reporting_reward=score.reward if score.reward is not None else -1.0,
            done=True,
            metrics={
                'valid_submission': float(score.status == ScoreStatus.VALID),
                **score.metrics(),
            },
            info={
                'episode_id': self._bundle.manifest.episode_id,
                'status': score.status.value,
                'issue_codes': [issue.code.value for issue in score.issues],
            },
        )

    def _validate_score_binding(self, score: ScoreVector | ScoreVectorV1) -> None:
        if (
            score.episode_id != self._bundle.manifest.episode_id
            or score.manifest_sha256 != self._bundle.manifest_sha256
            or score.labels_sha256 != self._bundle.manifest.labels_sha256
            or score.reward_version != self._bundle.manifest.reward_version
        ):
            raise ValueError('evaluator response is not bound to the active episode and label commitment')
