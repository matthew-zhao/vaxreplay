from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.aggregation import (
    INVALID_EPISODE_PENALTY,
    SuiteEpisodeBinding,
    SuiteManifest,
    SuiteScore,
    aggregate_scores,
    make_suite_manifest,
)
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import (
    ANTIGEN_TARGET_PRIORITIZATION_TASK,
    PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK,
    IssueCode,
    RewardVersion,
    ScoreStatus,
    ScoreVector,
    ValidationIssue,
)
from vaxreplay.ranking_schema import ScoreVectorV1

MANIFEST_HASH = '1' * 64
LABELS_HASH = '2' * 64


def _suite(*episode_ids: str, reward_version: RewardVersion = 'v0.1') -> SuiteManifest:
    return SuiteManifest(
        suite_id='fixture-suite',
        task_type=ANTIGEN_TARGET_PRIORITIZATION_TASK,
        reward_version=reward_version,
        episodes=tuple(
            SuiteEpisodeBinding(
                episode_id=episode_id,
                task_type=ANTIGEN_TARGET_PRIORITIZATION_TASK,
                reward_version=reward_version,
                manifest_sha256=MANIFEST_HASH,
                labels_sha256=LABELS_HASH,
            )
            for episode_id in sorted(episode_ids)
        ),
    )


def _v0_score(episode_id: str, *, reward: float, metric_offset: float = 0.0) -> ScoreVector:
    return ScoreVector(
        episode_id=episode_id,
        manifest_sha256=MANIFEST_HASH,
        labels_sha256=LABELS_HASH,
        status=ScoreStatus.VALID,
        reward=reward,
        forecast_brier=0.2 + metric_offset,
        forecast_reward=0.8 - metric_offset,
        ndcg_at_k=0.7 - metric_offset,
        grounding_precision=0.6 - metric_offset,
        grounding_recall=0.5 - metric_offset,
        grounding_f1=0.4 - metric_offset,
        assessment_accuracy=0.9 - metric_offset,
        grounding_reward=0.3 - metric_offset,
    )


def _invalid_score(
    episode_id: str,
    *,
    status: ScoreStatus = ScoreStatus.INVALID_SCHEMA,
) -> ScoreVector:
    return ScoreVector(
        episode_id=episode_id,
        manifest_sha256=MANIFEST_HASH,
        labels_sha256=LABELS_HASH,
        status=status,
        issues=[ValidationIssue(code=IssueCode.INVALID_RANKING, detail='invalid test submission')],
    )


def _v1_score(episode_id: str) -> ScoreVectorV1:
    return ScoreVectorV1(
        episode_id=episode_id,
        manifest_sha256=MANIFEST_HASH,
        labels_sha256=LABELS_HASH,
        status=ScoreStatus.VALID,
        reward=0.74,
        forecast_brier=0.2,
        forecast_reward=0.8,
        ndcg_at_k=0.8,
        pairwise_concordance=0.6,
        top_k_utility=1.0,
        ranking_reward=0.8,
        grounding_precision=0.5,
        grounding_recall=0.5,
        grounding_f1=0.5,
        assessment_accuracy=1.0,
        grounding_reward=0.5,
    )


class SuiteAggregationTest(unittest.TestCase):
    def test_macro_averages_valid_and_penalizes_invalid_or_missing_episodes(self) -> None:
        manifest = _suite('episode-a', 'episode-b', 'episode-c', 'episode-d')
        first = _v0_score('episode-b', reward=0.4)
        second = _v0_score('episode-a', reward=0.8, metric_offset=0.2)
        invalid = _invalid_score('episode-c')

        result = aggregate_scores(manifest, [first, invalid, second])

        self.assertEqual(result.episode_ids, ('episode-a', 'episode-b', 'episode-c', 'episode-d'))
        self.assertEqual(result.missing_episode_ids, ('episode-d',))
        self.assertEqual(result.valid_episode_count, 2)
        self.assertEqual(result.invalid_episode_count, 2)
        self.assertEqual(result.invalid_episode_penalty, INVALID_EPISODE_PENALTY)
        self.assertEqual(result.validity_rate, 0.5)
        self.assertEqual(
            result.status_counts,
            {'invalid_leakage': 0, 'invalid_schema': 2, 'valid': 2},
        )
        self.assertTrue(math.isclose(result.valid_metric_means['forecast_brier'], 0.3))
        self.assertTrue(math.isclose(result.valid_metric_means['reward'], 0.6))
        self.assertTrue(math.isclose(result.valid_only_mean_reward or -1.0, 0.6))
        self.assertTrue(math.isclose(result.all_episode_mean_environment_reward, (0.4 + 0.8 - 2.0) / 4))

    def test_all_missing_suite_receives_the_full_invalid_penalty(self) -> None:
        result = aggregate_scores(_suite('episode-a', 'episode-b'), [])

        self.assertEqual(result.missing_episode_ids, ('episode-a', 'episode-b'))
        self.assertEqual(result.validity_rate, 0.0)
        self.assertEqual(result.valid_metric_means, {})
        self.assertIsNone(result.valid_only_mean_reward)
        self.assertEqual(result.all_episode_mean_environment_reward, -1.0)
        self.assertEqual(
            result.status_counts,
            {'invalid_leakage': 0, 'invalid_schema': 2, 'valid': 0},
        )

    def test_v1_metrics_are_macro_averaged_without_pooling_ranking_pairs(self) -> None:
        result = aggregate_scores(_suite('episode-v1', reward_version='v1.0'), [_v1_score('episode-v1')])

        self.assertEqual(result.reward_version, 'v1.0')
        self.assertEqual(result.valid_metric_means['pairwise_concordance'], 0.6)
        self.assertEqual(result.valid_metric_means['top_k_utility'], 1.0)
        self.assertEqual(result.all_episode_mean_environment_reward, 0.74)

    def test_result_is_independent_of_score_input_order(self) -> None:
        manifest = _suite('episode-a', 'episode-b')
        scores = [_v0_score('episode-b', reward=0.3), _v0_score('episode-a', reward=0.7)]

        forward = aggregate_scores(manifest, scores)
        reverse = aggregate_scores(manifest, reversed(scores))

        self.assertEqual(forward.model_dump_json(), reverse.model_dump_json())

    def test_input_score_commitment_changes_with_an_episode_score(self) -> None:
        manifest = _suite('episode-a')
        first = aggregate_scores(manifest, [_v0_score('episode-a', reward=0.5)])
        revised = aggregate_scores(manifest, [_v0_score('episode-a', reward=0.6)])

        self.assertNotEqual(first.input_scores_sha256, revised.input_scores_sha256)

    def test_rejects_duplicate_extra_wrong_version_and_unbound_scores(self) -> None:
        manifest = _suite('episode-a')
        with self.subTest(case='duplicate'), self.assertRaisesRegex(ValueError, 'unique'):
            aggregate_scores(manifest, [_v0_score('episode-a', reward=0.1), _v0_score('episode-a', reward=0.9)])
        with self.subTest(case='extra'), self.assertRaisesRegex(ValueError, 'not present'):
            aggregate_scores(manifest, [_v0_score('episode-extra', reward=0.5)])
        with self.subTest(case='version'), self.assertRaisesRegex(ValueError, 'wrong reward_version'):
            aggregate_scores(manifest, [_v1_score('episode-a')])
        unbound = _v0_score('episode-a', reward=0.5).model_copy(update={'manifest_sha256': '3' * 64})
        with self.subTest(case='binding'), self.assertRaisesRegex(ValueError, 'not bound'):
            aggregate_scores(manifest, [unbound])

    def test_suite_manifest_factory_rejects_mixed_task_types(self) -> None:
        root = Path(__file__).parent / 'fixtures'
        antigen = EpisodeBundle.load(root / 'synthetic_antigen_v0')
        preclinical = EpisodeBundle.load(root / 'synthetic_preclinical_v1')

        manifest = make_suite_manifest('antigen-suite', [antigen])
        self.assertEqual(manifest.episodes[0].manifest_sha256, antigen.manifest_sha256)
        with self.assertRaisesRegex(ValueError, 'task_type'):
            make_suite_manifest('mixed-suite', [antigen, preclinical])

    def test_suite_manifest_factory_rejects_mixed_splits(self) -> None:
        root = Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'
        train = EpisodeBundle.load(root)
        dev = replace(
            train,
            manifest=train.manifest.model_copy(update={'episode_id': 'synthetic-antigen-dev', 'split': 'dev'}),
        )

        with self.assertRaisesRegex(ValueError, 'homogeneous split'):
            make_suite_manifest('mixed-split-suite', [train, dev])

    def test_suite_manifest_rejects_a_binding_from_another_task(self) -> None:
        binding = SuiteEpisodeBinding(
            episode_id='episode-a',
            task_type=PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK,
            reward_version='v1.0',
            manifest_sha256=MANIFEST_HASH,
            labels_sha256=LABELS_HASH,
        )

        with self.assertRaisesRegex(ValidationError, 'task_type'):
            SuiteManifest(
                suite_id='mismatched-suite',
                task_type=ANTIGEN_TARGET_PRIORITIZATION_TASK,
                reward_version='v1.0',
                episodes=(binding,),
            )

    def test_suite_model_forbids_penalty_override_and_extra_fields(self) -> None:
        data = aggregate_scores(_suite('episode-a'), [_v0_score('episode-a', reward=0.5)]).model_dump()

        with self.subTest(case='penalty'), self.assertRaises(ValidationError):
            SuiteScore.model_validate({**data, 'invalid_episode_penalty': 0.0})
        with self.subTest(case='extra'), self.assertRaises(ValidationError):
            SuiteScore.model_validate({**data, 'candidate_weight': 10.0})
        with self.subTest(case='mean'), self.assertRaisesRegex(ValidationError, 'inconsistent'):
            SuiteScore.model_validate({**data, 'all_episode_mean_environment_reward': -0.5})


if __name__ == '__main__':
    unittest.main()
