from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import Split
from vaxreplay.environment import VaxReplayEnvironment
from vaxreplay.scoring import LocalSubmissionEvaluator


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'


class VaxReplayEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EpisodeBundle.load(_fixture_root(), include_private=True)
        self.environment = VaxReplayEnvironment(self.bundle, LocalSubmissionEvaluator(self.bundle))

    def test_reset_contains_only_public_material(self) -> None:
        messages = self.environment.reset()
        rendered = '\n'.join(message['content'] for message in messages)

        self.assertNotIn('POST-CUTOFF CANARY', rendered)
        self.assertNotIn('candidate_utility', rendered)
        self.assertIn('ev-surveillance-1', rendered)

    def test_valid_response_receives_deterministic_reward(self) -> None:
        result = self.environment.step(oracle_submission(self.bundle).model_dump_json())

        self.assertTrue(result.done)
        self.assertEqual(result.reporting_reward, 1.0)
        self.assertEqual(result.metrics['valid_submission'], 1.0)
        self.assertNotIn('candidate_utility', str(result.info))
        with self.assertRaisesRegex(RuntimeError, 'evaluation-only'):
            _ = result.reward

    def test_malformed_response_terminates_with_format_penalty(self) -> None:
        result = self.environment.step('not json')

        self.assertTrue(result.done)
        self.assertEqual(result.reporting_reward, -1.0)
        self.assertEqual(result.metrics, {'valid_submission': 0.0})
        self.assertEqual(result.info['parse_reason'], 'invalid_json')

    def test_duplicate_json_members_fail_closed_before_schema_validation(self) -> None:
        response = oracle_submission(self.bundle).model_dump_json()
        duplicate = response.replace(
            f'"episode_id":"{self.bundle.manifest.episode_id}"',
            (f'"episode_id":"attacker-selected","episode_id":"{self.bundle.manifest.episode_id}"'),
            1,
        )

        result = self.environment.step(duplicate)

        self.assertEqual(result.reporting_reward, -1.0)
        self.assertEqual(result.metrics, {'valid_submission': 0.0})
        self.assertEqual(result.info['parse_reason'], 'duplicate_key')
        self.assertEqual(result.info['issue_codes'], ['STRICT_PARSE_DUPLICATE_KEY'])

    def test_rejects_sealed_test_episode(self) -> None:
        test_bundle = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'split': Split.TEST}),
        )

        with self.assertRaisesRegex(ValueError, 'one-shot external evaluator'):
            VaxReplayEnvironment(test_bundle, LocalSubmissionEvaluator(self.bundle))

    def test_rejects_evaluator_response_bound_to_different_labels(self) -> None:
        submission = oracle_submission(self.bundle)
        score = LocalSubmissionEvaluator(self.bundle).score(submission).model_copy(update={'labels_sha256': '0' * 64})
        evaluator = MagicMock()
        evaluator.score.return_value = score
        environment = VaxReplayEnvironment(self.bundle, evaluator)

        with self.assertRaisesRegex(ValueError, 'not bound to the active episode'):
            environment.step(submission.model_dump_json())


if __name__ == '__main__':
    unittest.main()
