from __future__ import annotations

import itertools
import unittest
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import (
    BundleIntegrityError,
    EpisodeBundle,
    ranking_labels_commitment,
)
from vaxreplay.case_schema import (
    RANKING_REWARD_VERSION,
    CandidateAssessment,
    Citation,
    IssueCode,
    ScoreStatus,
    ScoreVector,
    Submission,
)
from vaxreplay.environment import VaxReplayEnvironment
from vaxreplay.ranking_schema import RankingLabelV1, ScoreVectorV1
from vaxreplay.scoring import RankingSubmissionEvaluator, make_submission_evaluator


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'


def _v1_bundle(
    *,
    grades: dict[str, int | None] | None = None,
    portfolio_size: int = 1,
    validate: bool = True,
) -> EpisodeBundle:
    bundle = EpisodeBundle.load(_fixture_root(), include_private=True)
    private_labels = bundle.private_labels
    assert private_labels is not None
    grades = grades or {
        'target-17': 4,
        'target-42': 0,
        'target-88': 3,
    }
    ranking_labels = tuple(
        RankingLabelV1(
            episode_id=bundle.manifest.episode_id,
            candidate_id=candidate_id,
            relevance_grade=grades[candidate_id],
            censor_reason='outcome unavailable' if grades[candidate_id] is None else None,
        )
        for candidate_id in bundle.manifest.candidate_ids
    )
    commitment = ranking_labels_commitment(
        private_labels,
        ranking_labels,
        bundle.manifest.label_commitment_scheme,
        key=bundle.label_commitment_key,
    )
    manifest = bundle.manifest.model_copy(
        update={
            'reward_version': RANKING_REWARD_VERSION,
            'portfolio_size': portfolio_size,
            'labels_sha256': commitment,
        }
    )
    v1_bundle = replace(bundle, manifest=manifest, ranking_labels=ranking_labels)
    if validate:
        v1_bundle.validate_integrity()
    return v1_bundle


def _submission_for_ranking(bundle: EpisodeBundle, ranking: tuple[str, ...] | list[str]) -> Submission:
    private_labels = bundle.private_labels
    assert private_labels is not None
    conclusion_by_key = {
        (assessment.candidate_id, assessment.dimension): assessment.conclusion
        for assessment in private_labels.assessments_gold
    }
    evidence_by_key = defaultdict(list)
    for record in private_labels.evidence_gold:
        evidence_by_key[(record.candidate_id, record.dimension)].append(record)

    assessments = []
    for candidate_id in ranking[: bundle.manifest.portfolio_size]:
        for dimension in bundle.manifest.required_dimensions:
            assessments.append(
                CandidateAssessment(
                    candidate_id=candidate_id,
                    dimension=dimension,
                    conclusion=conclusion_by_key[(candidate_id, dimension)],
                    citations=[
                        Citation(
                            evidence_id=record.evidence_id,
                            stance=record.stance,
                            quote=record.quote,
                        )
                        for record in evidence_by_key[(candidate_id, dimension)]
                    ],
                )
            )

    oracle = oracle_submission(bundle)
    return oracle.model_copy(
        update={
            'ranking': list(ranking),
            'assessments': assessments,
        }
    )


class RankingSubmissionEvaluatorTest(unittest.TestCase):
    def test_v1_oracle_reaches_every_reward_ceiling(self) -> None:
        bundle = _v1_bundle()
        evaluator = make_submission_evaluator(bundle)

        score = evaluator.score(oracle_submission(bundle))

        self.assertIsInstance(score, ScoreVectorV1)
        assert isinstance(score, ScoreVectorV1)
        self.assertEqual(score.reward_version, RANKING_REWARD_VERSION)
        self.assertEqual(score.status, ScoreStatus.VALID)
        self.assertEqual(score.reward, 1.0)
        self.assertEqual(score.forecast_brier, 0.0)
        self.assertEqual(score.forecast_reward, 1.0)
        self.assertEqual(score.ndcg_at_k, 1.0)
        self.assertEqual(score.pairwise_concordance, 1.0)
        self.assertEqual(score.top_k_utility, 1.0)
        self.assertEqual(score.ranking_reward, 1.0)
        self.assertEqual(score.grounding_f1, 1.0)
        self.assertEqual(score.assessment_accuracy, 1.0)
        self.assertEqual(score.grounding_reward, 1.0)

    def test_invalid_v1_result_keeps_reward_version_and_has_no_metrics(self) -> None:
        bundle = _v1_bundle()
        evaluator = RankingSubmissionEvaluator(bundle)
        invalid = oracle_submission(bundle).model_copy(update={'ranking': ['target-17', 'target-88', 'target-88']})

        score = evaluator.score(invalid)

        self.assertEqual(score.reward_version, RANKING_REWARD_VERSION)
        self.assertEqual(score.status, ScoreStatus.INVALID_SCHEMA)
        self.assertIsNone(score.reward)
        self.assertIsNone(score.ranking_reward)
        self.assertIn(IssueCode.INVALID_RANKING, {issue.code for issue in score.issues})

    def test_environment_rejects_a_v0_score_bound_to_a_v1_episode(self) -> None:
        bundle = _v1_bundle()
        mismatched_score = ScoreVector(
            episode_id=bundle.manifest.episode_id,
            manifest_sha256=bundle.manifest_sha256,
            labels_sha256=bundle.manifest.labels_sha256,
            status=ScoreStatus.VALID,
            reward=1.0,
            forecast_brier=0.0,
            forecast_reward=1.0,
            ndcg_at_k=1.0,
            grounding_precision=1.0,
            grounding_recall=1.0,
            grounding_f1=1.0,
            assessment_accuracy=1.0,
            grounding_reward=1.0,
        )

        class MismatchedEvaluator:
            def score(self, _submission: Submission) -> ScoreVector:
                return mismatched_score

        environment = VaxReplayEnvironment(bundle, MismatchedEvaluator())

        with self.assertRaisesRegex(ValueError, 'not bound'):
            environment.step(oracle_submission(bundle).model_dump_json())

    def test_below_k_unequal_swap_changes_only_pairwise_ranking_component(self) -> None:
        bundle = _v1_bundle(portfolio_size=1)
        evaluator = RankingSubmissionEvaluator(bundle)
        oracle_score = evaluator.score(_submission_for_ranking(bundle, ('target-17', 'target-88', 'target-42')))
        swapped_score = evaluator.score(_submission_for_ranking(bundle, ('target-17', 'target-42', 'target-88')))

        self.assertEqual(swapped_score.status, ScoreStatus.VALID)
        self.assertEqual(swapped_score.ndcg_at_k, oracle_score.ndcg_at_k)
        self.assertEqual(swapped_score.top_k_utility, oracle_score.top_k_utility)
        self.assertEqual(swapped_score.forecast_reward, oracle_score.forecast_reward)
        self.assertEqual(swapped_score.grounding_reward, oracle_score.grounding_reward)
        self.assertEqual(swapped_score.pairwise_concordance, 2 / 3)
        assert swapped_score.ranking_reward is not None
        assert oracle_score.ranking_reward is not None
        assert swapped_score.reward is not None
        assert oracle_score.reward is not None
        self.assertLess(swapped_score.ranking_reward, oracle_score.ranking_reward)
        self.assertLess(swapped_score.reward, oracle_score.reward)

    def test_reordering_same_top_k_set_preserves_top_k_utility(self) -> None:
        bundle = _v1_bundle(portfolio_size=2)
        evaluator = RankingSubmissionEvaluator(bundle)
        oracle_score = evaluator.score(_submission_for_ranking(bundle, ('target-17', 'target-88', 'target-42')))
        reordered_score = evaluator.score(_submission_for_ranking(bundle, ('target-88', 'target-17', 'target-42')))

        self.assertEqual(reordered_score.status, ScoreStatus.VALID)
        self.assertEqual(reordered_score.top_k_utility, 1.0)
        self.assertEqual(reordered_score.top_k_utility, oracle_score.top_k_utility)
        assert reordered_score.ndcg_at_k is not None
        assert oracle_score.ndcg_at_k is not None
        assert reordered_score.pairwise_concordance is not None
        assert oracle_score.pairwise_concordance is not None
        self.assertLess(reordered_score.ndcg_at_k, oracle_score.ndcg_at_k)
        self.assertLess(reordered_score.pairwise_concordance, oracle_score.pairwise_concordance)
        self.assertEqual(reordered_score.forecast_reward, oracle_score.forecast_reward)
        self.assertEqual(reordered_score.grounding_reward, oracle_score.grounding_reward)

    def test_replacing_the_top_candidate_with_a_lower_grade_reduces_ranking_reward(self) -> None:
        bundle = _v1_bundle(portfolio_size=1)
        evaluator = RankingSubmissionEvaluator(bundle)
        oracle_score = evaluator.score(_submission_for_ranking(bundle, ('target-17', 'target-88', 'target-42')))
        lower_top_score = evaluator.score(_submission_for_ranking(bundle, ('target-88', 'target-17', 'target-42')))

        self.assertEqual(lower_top_score.status, ScoreStatus.VALID)
        self.assertLess(lower_top_score.ndcg_at_k, oracle_score.ndcg_at_k)
        self.assertLess(lower_top_score.pairwise_concordance, oracle_score.pairwise_concordance)
        self.assertLess(lower_top_score.top_k_utility, oracle_score.top_k_utility)
        self.assertLess(lower_top_score.ranking_reward, oracle_score.ranking_reward)
        self.assertLess(lower_top_score.reward, oracle_score.reward)

    def test_gold_tie_permutations_are_metric_invariant(self) -> None:
        bundle = _v1_bundle(
            grades={'target-17': 4, 'target-42': 1, 'target-88': 1},
            portfolio_size=1,
        )
        evaluator = RankingSubmissionEvaluator(bundle)

        first = evaluator.score(_submission_for_ranking(bundle, ('target-17', 'target-42', 'target-88')))
        second = evaluator.score(_submission_for_ranking(bundle, ('target-17', 'target-88', 'target-42')))

        self.assertEqual(first.status, ScoreStatus.VALID)
        self.assertEqual(second.status, ScoreStatus.VALID)
        self.assertEqual(first.metrics(), second.metrics())
        self.assertEqual(first.pairwise_concordance, 1.0)

    def test_rejects_all_equal_ranking_grades(self) -> None:
        bundle = _v1_bundle(
            grades={'target-17': 2, 'target-42': 2, 'target-88': 2},
            validate=False,
        )

        with self.assertRaisesRegex(BundleIntegrityError, 'at least two distinct grades'):
            bundle.validate_integrity()

    def test_rejects_censored_ranking_label(self) -> None:
        bundle = _v1_bundle(
            grades={'target-17': 4, 'target-42': None, 'target-88': 3},
            validate=False,
        )

        with self.assertRaisesRegex(BundleIntegrityError, 'cannot contain censored ranking labels'):
            bundle.validate_integrity()

    def test_rejects_portfolio_covering_every_candidate(self) -> None:
        bundle = _v1_bundle(portfolio_size=3, validate=False)

        with self.assertRaisesRegex(BundleIntegrityError, 'must be smaller than the candidate count'):
            bundle.validate_integrity()

    def test_tampered_ranking_grade_breaks_label_commitment(self) -> None:
        bundle = _v1_bundle()
        assert bundle.ranking_labels is not None
        tampered_labels = (
            bundle.ranking_labels[0].model_copy(update={'relevance_grade': 2}),
            *bundle.ranking_labels[1:],
        )
        tampered_bundle = replace(bundle, ranking_labels=tampered_labels)

        with self.assertRaisesRegex(BundleIntegrityError, 'private label hash'):
            tampered_bundle.validate_integrity()

    def test_all_three_candidate_permutations_are_bounded_and_oracle_is_unique_maximum(self) -> None:
        bundle = _v1_bundle(portfolio_size=1)
        evaluator = RankingSubmissionEvaluator(bundle)
        oracle_ranking = ('target-17', 'target-88', 'target-42')
        scores: dict[tuple[str, ...], float] = {}

        for ranking in itertools.permutations(bundle.manifest.candidate_ids):
            score = evaluator.score(_submission_for_ranking(bundle, ranking))
            self.assertEqual(score.status, ScoreStatus.VALID)
            for metric in (
                score.reward,
                score.ndcg_at_k,
                score.pairwise_concordance,
                score.top_k_utility,
                score.ranking_reward,
            ):
                assert metric is not None
                self.assertGreaterEqual(metric, 0.0)
                self.assertLessEqual(metric, 1.0)
            assert score.reward is not None
            scores[ranking] = score.reward

        oracle_reward = scores[oracle_ranking]
        self.assertEqual(oracle_reward, 1.0)
        self.assertTrue(all(reward < oracle_reward for ranking, reward in scores.items() if ranking != oracle_ranking))


if __name__ == '__main__':
    unittest.main()
