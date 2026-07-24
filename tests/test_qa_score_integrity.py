from __future__ import annotations

from pathlib import Path

import pytest

from vaxreplay.baselines import oracle_submission, uniform_submission
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.qa.score_integrity import (
    ScoreIntegrityError,
    ScoreIntegrityReason,
    differential_score,
    require_score_binding,
    require_score_formula,
)
from vaxreplay.scoring import LocalSubmissionEvaluator


def _bundle() -> EpisodeBundle:
    return EpisodeBundle.load(
        Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0',
        include_private=True,
    )


def test_accepts_independently_recomputed_v0_oracle_score() -> None:
    bundle = _bundle()
    score = LocalSubmissionEvaluator(bundle).score(oracle_submission(bundle))

    require_score_binding(bundle, score)
    require_score_formula(score)


@pytest.mark.parametrize(
    ('field', 'value', 'name'),
    [
        ('forecast_reward', 0.9, 'forecast_reward'),
        ('grounding_f1', 0.9, 'grounding_f1'),
        ('grounding_reward', 0.9, 'grounding_reward'),
        ('reward', 0.9, 'reward'),
    ],
)
def test_rejects_v0_scores_with_forged_derived_components(
    field: str,
    value: float,
    name: str,
) -> None:
    bundle = _bundle()
    score = LocalSubmissionEvaluator(bundle).score(oracle_submission(bundle)).model_copy(update={field: value})

    with pytest.raises(ScoreIntegrityError, match=name) as caught:
        require_score_formula(score)
    assert caught.value.reason is ScoreIntegrityReason.FORMULA_MISMATCH


def test_rejects_score_bound_to_different_episode_material() -> None:
    bundle = _bundle()
    score = (
        LocalSubmissionEvaluator(bundle).score(oracle_submission(bundle)).model_copy(update={'labels_sha256': '0' * 64})
    )

    with pytest.raises(ScoreIntegrityError) as caught:
        require_score_binding(bundle, score)
    assert caught.value.reason is ScoreIntegrityReason.BINDING_MISMATCH


def test_differential_scoring_requires_byte_identical_results() -> None:
    bundle = _bundle()
    submission = oracle_submission(bundle)
    primary = LocalSubmissionEvaluator(bundle)

    class ForgedReference:
        def score(self, _submission):
            return primary.score(_submission).model_copy(update={'reward': 0.9})

    with pytest.raises(ScoreIntegrityError) as caught:
        differential_score(bundle, submission, primary, ForgedReference())
    assert caught.value.reason is ScoreIntegrityReason.FORMULA_MISMATCH


def test_differential_scoring_rejects_two_individually_valid_disagreeing_scores() -> None:
    bundle = _bundle()
    submission = oracle_submission(bundle)
    evaluator = LocalSubmissionEvaluator(bundle)

    class DisagreeingReference:
        def score(self, _submission):
            return evaluator.score(uniform_submission(bundle))

    with pytest.raises(ScoreIntegrityError) as caught:
        differential_score(bundle, submission, evaluator, DisagreeingReference())
    assert caught.value.reason is ScoreIntegrityReason.DIFFERENTIAL_MISMATCH


def test_differential_scoring_releases_matching_result() -> None:
    bundle = _bundle()
    submission = oracle_submission(bundle)

    score = differential_score(
        bundle,
        submission,
        LocalSubmissionEvaluator(bundle),
        LocalSubmissionEvaluator(bundle),
    )

    assert score.reward == 1.0


def test_differential_scoring_isolates_and_rejects_input_mutation() -> None:
    bundle = _bundle()
    evaluator = LocalSubmissionEvaluator(bundle)

    class MutatingPrimary:
        def score(self, submission):
            submission.ranking.reverse()
            return evaluator.score(submission)

    with pytest.raises(ScoreIntegrityError) as caught:
        differential_score(
            bundle,
            oracle_submission(bundle),
            MutatingPrimary(),
            evaluator,
        )

    assert caught.value.reason is ScoreIntegrityReason.INPUT_MUTATION


def test_differential_scoring_revalidates_copied_score_schema() -> None:
    bundle = _bundle()
    evaluator = LocalSubmissionEvaluator(bundle)

    class InvalidPrimary:
        def score(self, submission):
            return evaluator.score(submission).model_copy(update={'forecast_brier': -1.0})

    with pytest.raises(ScoreIntegrityError) as caught:
        differential_score(
            bundle,
            oracle_submission(bundle),
            InvalidPrimary(),
            evaluator,
        )

    assert caught.value.reason is ScoreIntegrityReason.SCHEMA_INVALID
