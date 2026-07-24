from __future__ import annotations

import unittest

from pydantic import ValidationError

from vaxreplay.case_schema import ScoreStatus
from vaxreplay.ranking_schema import ScoreVectorV1


def _valid_score_data() -> dict[str, object]:
    return {
        'episode_id': 'episode-v1',
        'manifest_sha256': '1' * 64,
        'labels_sha256': '2' * 64,
        'status': ScoreStatus.VALID,
        'reward': 0.74,
        'forecast_brier': 0.2,
        'forecast_reward': 0.8,
        'ndcg_at_k': 0.8,
        'pairwise_concordance': 0.6,
        'top_k_utility': 1.0,
        'ranking_reward': 0.8,
        'grounding_precision': 0.5,
        'grounding_recall': 0.5,
        'grounding_f1': 0.5,
        'assessment_accuracy': 1.0,
        'grounding_reward': 0.5,
    }


class ScoreVectorV1Test(unittest.TestCase):
    def test_accepts_internally_consistent_components(self) -> None:
        score = ScoreVectorV1.model_validate(_valid_score_data())

        self.assertEqual(score.reward_version, 'v1.0')

    def test_rejects_inconsistent_derived_components(self) -> None:
        cases = (
            ('forecast_reward', 0.7, 'forecast_reward'),
            ('ranking_reward', 0.7, 'ranking_reward'),
            ('grounding_f1', 0.4, 'grounding_f1'),
            ('grounding_reward', 0.4, 'grounding_reward'),
            ('reward', 0.7, 'reward is inconsistent'),
        )
        for field, value, message in cases:
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, message):
                ScoreVectorV1.model_validate({**_valid_score_data(), field: value})


if __name__ == '__main__':
    unittest.main()
