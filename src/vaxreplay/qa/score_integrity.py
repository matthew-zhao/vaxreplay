"""Independent formula, binding, and differential checks for training rewards."""

from __future__ import annotations

import enum
import hashlib
import math
from typing import Protocol

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_schema import ScoreStatus, ScoreVector, Submission
from vaxreplay.ranking_schema import (
    V1_FORECAST_WEIGHT,
    V1_GROUNDING_WEIGHT,
    V1_NDCG_WEIGHT,
    V1_PAIRWISE_WEIGHT,
    V1_RANKING_WEIGHT,
    V1_TOP_K_UTILITY_WEIGHT,
    ScoreVectorV1,
)

_ABS_TOLERANCE = 1e-12

type AnyScoreVector = ScoreVector | ScoreVectorV1


class ScoreIntegrityReason(str, enum.Enum):
    SCHEMA_INVALID = 'schema_invalid'
    BINDING_MISMATCH = 'binding_mismatch'
    FORMULA_MISMATCH = 'formula_mismatch'
    DIFFERENTIAL_MISMATCH = 'differential_mismatch'
    INPUT_MUTATION = 'input_mutation'
    EVALUATOR_FAILURE = 'evaluator_failure'


class ScoreIntegrityError(ValueError):
    """A fail-closed reward-integrity failure with a stable reason."""

    def __init__(self, reason: ScoreIntegrityReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f'{reason.value}: {detail}')


class ScoreEvaluator(Protocol):
    def score(self, submission: Submission) -> AnyScoreVector: ...


def revalidate_submission(submission: Submission) -> Submission:
    """Round-trip a possibly copied/constructed model through its strict schema."""

    try:
        return Submission.model_validate_json(canonical_json_bytes(submission))
    except (TypeError, ValueError) as error:
        raise ScoreIntegrityError(
            ScoreIntegrityReason.SCHEMA_INVALID,
            'submission does not satisfy the strict canonical schema',
        ) from error


def revalidate_score(score: AnyScoreVector) -> AnyScoreVector:
    """Reject Pydantic ``model_copy``/``model_construct`` invariant bypasses."""

    score_type = ScoreVectorV1 if isinstance(score, ScoreVectorV1) else ScoreVector
    try:
        return score_type.model_validate_json(canonical_json_bytes(score))
    except (TypeError, ValueError) as error:
        raise ScoreIntegrityError(
            ScoreIntegrityReason.SCHEMA_INVALID,
            'score does not satisfy its strict canonical schema',
        ) from error


def score_vector_sha256(score: AnyScoreVector) -> str:
    return hashlib.sha256(canonical_json_bytes(revalidate_score(score))).hexdigest()


def require_score_binding(bundle: EpisodeBundle, score: AnyScoreVector) -> None:
    expected = (
        bundle.manifest.episode_id,
        bundle.manifest_sha256,
        bundle.manifest.labels_sha256,
        bundle.manifest.reward_version,
    )
    observed = (
        score.episode_id,
        score.manifest_sha256,
        score.labels_sha256,
        score.reward_version,
    )
    if observed != expected:
        raise ScoreIntegrityError(
            ScoreIntegrityReason.BINDING_MISMATCH,
            'score does not bind the active episode, manifest, labels, and reward version',
        )


def _require_close(observed: float, expected: float, name: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=_ABS_TOLERANCE):
        raise ScoreIntegrityError(
            ScoreIntegrityReason.FORMULA_MISMATCH,
            f'{name} is {observed!r}; independently recomputed value is {expected!r}',
        )


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def require_score_formula(score: AnyScoreVector) -> None:
    """Recompute every published scalar relationship outside the scorer.

    Invalid scores contain no reward and therefore have no scalar formula to
    verify. Their Pydantic contracts already require issues and forbid metrics.
    """

    if score.status != ScoreStatus.VALID:
        return
    if isinstance(score, ScoreVectorV1):
        _require_v1_formula(score)
    else:
        _require_v0_formula(score)


def _require_v0_formula(score: ScoreVector) -> None:
    assert score.forecast_brier is not None
    assert score.forecast_reward is not None
    assert score.ndcg_at_k is not None
    assert score.grounding_precision is not None
    assert score.grounding_recall is not None
    assert score.grounding_f1 is not None
    assert score.assessment_accuracy is not None
    assert score.grounding_reward is not None
    assert score.reward is not None
    _require_close(score.forecast_reward, 1.0 - score.forecast_brier, 'forecast_reward')
    _require_close(
        score.grounding_f1,
        _f1(score.grounding_precision, score.grounding_recall),
        'grounding_f1',
    )
    _require_close(
        score.grounding_reward,
        score.grounding_f1 * score.assessment_accuracy,
        'grounding_reward',
    )
    expected_reward = 0.50 * score.forecast_reward + 0.30 * score.ndcg_at_k + 0.20 * score.grounding_reward
    _require_close(score.reward, min(1.0, max(0.0, expected_reward)), 'reward')


def _require_v1_formula(score: ScoreVectorV1) -> None:
    assert score.forecast_brier is not None
    assert score.forecast_reward is not None
    assert score.ndcg_at_k is not None
    assert score.pairwise_concordance is not None
    assert score.top_k_utility is not None
    assert score.ranking_reward is not None
    assert score.grounding_precision is not None
    assert score.grounding_recall is not None
    assert score.grounding_f1 is not None
    assert score.assessment_accuracy is not None
    assert score.grounding_reward is not None
    assert score.reward is not None
    _require_close(score.forecast_reward, 1.0 - score.forecast_brier, 'forecast_reward')
    _require_close(
        score.ranking_reward,
        (
            V1_NDCG_WEIGHT * score.ndcg_at_k
            + V1_PAIRWISE_WEIGHT * score.pairwise_concordance
            + V1_TOP_K_UTILITY_WEIGHT * score.top_k_utility
        ),
        'ranking_reward',
    )
    _require_close(
        score.grounding_f1,
        _f1(score.grounding_precision, score.grounding_recall),
        'grounding_f1',
    )
    _require_close(
        score.grounding_reward,
        score.grounding_f1 * score.assessment_accuracy,
        'grounding_reward',
    )
    _require_close(
        score.reward,
        (
            V1_FORECAST_WEIGHT * score.forecast_reward
            + V1_RANKING_WEIGHT * score.ranking_reward
            + V1_GROUNDING_WEIGHT * score.grounding_reward
        ),
        'reward',
    )


def differential_score(
    bundle: EpisodeBundle,
    submission: Submission,
    primary: ScoreEvaluator,
    reference: ScoreEvaluator,
) -> AnyScoreVector:
    """Score twice and release only a byte-identical, correctly bound result."""

    canonical_submission = canonical_json_bytes(revalidate_submission(submission))
    primary_input = Submission.model_validate_json(canonical_submission)
    reference_input = Submission.model_validate_json(canonical_submission)
    try:
        primary_score = primary.score(primary_input)
    except Exception as error:
        raise ScoreIntegrityError(
            ScoreIntegrityReason.EVALUATOR_FAILURE,
            'primary scorer failed',
        ) from error
    if canonical_json_bytes(primary_input) != canonical_submission:
        raise ScoreIntegrityError(
            ScoreIntegrityReason.INPUT_MUTATION,
            'primary scorer mutated its submission input',
        )
    try:
        reference_score = reference.score(reference_input)
    except Exception as error:
        raise ScoreIntegrityError(
            ScoreIntegrityReason.EVALUATOR_FAILURE,
            'reference scorer failed',
        ) from error
    if canonical_json_bytes(reference_input) != canonical_submission:
        raise ScoreIntegrityError(
            ScoreIntegrityReason.INPUT_MUTATION,
            'reference scorer mutated its submission input',
        )
    primary_score = revalidate_score(primary_score)
    reference_score = revalidate_score(reference_score)
    require_score_binding(bundle, primary_score)
    require_score_binding(bundle, reference_score)
    require_score_formula(primary_score)
    require_score_formula(reference_score)
    if canonical_json_bytes(primary_score) != canonical_json_bytes(reference_score):
        raise ScoreIntegrityError(
            ScoreIntegrityReason.DIFFERENTIAL_MISMATCH,
            (
                'primary and reference scorers disagree: '
                f'{score_vector_sha256(primary_score)} != {score_vector_sha256(reference_score)}'
            ),
        )
    return primary_score
