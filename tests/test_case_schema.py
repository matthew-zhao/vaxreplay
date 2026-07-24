from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.case_schema import CandidateForecast, EpisodeManifest, ScoreStatus, ScoreVector


def _fixture_manifest() -> dict[str, object]:
    path = Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0' / 'manifest.json'
    return json.loads(path.read_text(encoding='utf-8'))


class CaseSchemaTest(unittest.TestCase):
    def test_forecast_rejects_boolean_probability_and_string_horizon(self) -> None:
        with self.assertRaises(ValidationError):
            CandidateForecast.model_validate_json(
                json.dumps(
                    {
                        'candidate_id': 'target-17',
                        'target_id': 'functional_validation',
                        'horizon_days': '180',
                        'probability': True,
                    }
                )
            )

    def test_score_rejects_non_finite_metric(self) -> None:
        with self.assertRaises(ValidationError):
            ScoreVector(
                episode_id='synthetic-antigen-001',
                manifest_sha256='0' * 64,
                labels_sha256='1' * 64,
                status=ScoreStatus.VALID,
                reward=math.nan,
                forecast_brier=0.0,
                forecast_reward=1.0,
                ndcg_at_k=1.0,
                grounding_precision=1.0,
                grounding_recall=1.0,
                grounding_f1=1.0,
                assessment_accuracy=1.0,
                grounding_reward=1.0,
            )

    def test_invalid_score_requires_issue_and_no_metrics(self) -> None:
        with self.assertRaises(ValidationError):
            ScoreVector(
                episode_id='synthetic-antigen-001',
                manifest_sha256='0' * 64,
                labels_sha256='1' * 64,
                status=ScoreStatus.INVALID_SCHEMA,
            )

    def test_sealed_v1_episode_requires_an_hmac_label_commitment(self) -> None:
        manifest = {
            **_fixture_manifest(),
            'split': 'test',
            'reward_version': 'v1.0',
        }

        with self.assertRaisesRegex(ValidationError, 'require HMAC'):
            EpisodeManifest.model_validate_json(json.dumps(manifest))


if __name__ == '__main__':
    unittest.main()
