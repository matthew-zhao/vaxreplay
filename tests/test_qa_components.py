from __future__ import annotations

from pathlib import Path

import pytest

from vaxreplay.baselines import oracle_submission, uniform_submission
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.qa.components import (
    ComponentFloor,
    audit_component_floors,
    normalized_component_floors,
    require_component_floors,
    zero_grounding_reward_ceiling,
)
from vaxreplay.scoring import LocalSubmissionEvaluator


def _bundle() -> EpisodeBundle:
    return EpisodeBundle.load(
        Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0',
        include_private=True,
    )


def test_oracle_passes_non_collapsible_scientific_component_floors() -> None:
    bundle = _bundle()
    score = LocalSubmissionEvaluator(bundle).score(oracle_submission(bundle))

    results = audit_component_floors(
        score,
        {'assessment_accuracy': 0.5, 'grounding_f1': 0.5},
    )

    assert all(result.passed for result in results)
    require_component_floors(score, {'assessment_accuracy': 0.5, 'grounding_f1': 0.5})


def test_uniform_policy_is_quarantined_despite_positive_aggregate_reward() -> None:
    bundle = _bundle()
    score = LocalSubmissionEvaluator(bundle).score(uniform_submission(bundle))
    assert score.reward is not None and score.reward > 0.5

    results = audit_component_floors(
        score,
        {'assessment_accuracy': 0.01, 'grounding_f1': 0.01},
    )

    assert {result.metric for result in results if not result.passed} == {
        'assessment_accuracy',
        'grounding_f1',
    }
    with pytest.raises(ValueError, match='non-collapsible component floors'):
        require_component_floors(score, {'assessment_accuracy': 0.01, 'grounding_f1': 0.01})


def test_missing_component_fails_closed() -> None:
    bundle = _bundle()
    score = LocalSubmissionEvaluator(bundle).score(oracle_submission(bundle))

    [result] = audit_component_floors(score, {'nonexistent_metric': 0.1})

    assert not result.passed
    assert result.observed is None


def test_floor_contract_is_unique_sorted_and_bounded() -> None:
    assert normalized_component_floors(
        (ComponentFloor('grounding_f1', 0.5), ComponentFloor('assessment_accuracy', 0.5))
    ) == (
        ComponentFloor('assessment_accuracy', 0.5),
        ComponentFloor('grounding_f1', 0.5),
    )
    with pytest.raises(ValueError, match='unique'):
        normalized_component_floors((ComponentFloor('grounding_f1', 0.1), ComponentFloor('grounding_f1', 0.2)))
    with pytest.raises(ValueError, match='between zero and one'):
        ComponentFloor('grounding_f1', 1.1)


def test_published_zero_grounding_ceiling_is_explicit() -> None:
    assert zero_grounding_reward_ceiling('v0.1') == 0.8
    assert zero_grounding_reward_ceiling('v1.0') == 0.8
