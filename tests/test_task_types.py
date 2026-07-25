from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import (
    EARLY_CLINICAL_ARM_PRIORITIZATION_TASK,
    PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK,
    EpisodeManifest,
)
from vaxreplay.prompt import build_episode_prompt
from vaxreplay.scoring import make_submission_evaluator
from vaxreplay.temporal_schema import DecisionTimeConfig


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'


def _preclinical_fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


def _manifest_data() -> dict[str, object]:
    return json.loads((_fixture_root() / 'manifest.json').read_text(encoding='utf-8'))


class TaskTypeTest(unittest.TestCase):
    def test_preclinical_advancement_requires_v1(self) -> None:
        data = {
            **_manifest_data(),
            'task_type': PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK,
        }

        with self.assertRaisesRegex(ValidationError, 'requires the V1 ranking reward'):
            EpisodeManifest.model_validate_json(json.dumps(data))

    def test_preclinical_prompt_ranks_only_existing_candidates(self) -> None:
        bundle = EpisodeBundle.load(_fixture_root(), include_private=False)
        manifest = bundle.manifest.model_copy(
            update={
                'task_type': PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK,
                'reward_version': 'v1.0',
            }
        )
        stage_bundle = replace(bundle, manifest=manifest)

        prompt = build_episode_prompt(stage_bundle)

        self.assertIn('"task_type": "preclinical_candidate_advancement"', prompt)
        self.assertIn('already-defined candidates for preclinical advancement', prompt)
        self.assertIn('Do not invent or modify candidates', prompt)
        self.assertIn('do not propose experimental procedures', prompt)
        self.assertNotIn('design a new candidate', prompt.lower())

    def test_early_clinical_arm_prioritization_requires_v1(self) -> None:
        data = {
            **_manifest_data(),
            'task_type': EARLY_CLINICAL_ARM_PRIORITIZATION_TASK,
        }

        with self.assertRaisesRegex(ValidationError, 'requires the V1 ranking reward'):
            EpisodeManifest.model_validate_json(json.dumps(data))

        bundle = EpisodeBundle.load(_fixture_root(), include_private=False)
        invalid_manifest = bundle.manifest.model_copy(update={'task_type': EARLY_CLINICAL_ARM_PRIORITIZATION_TASK})
        with self.assertRaisesRegex(ValidationError, 'requires the V1 ranking reward'):
            DecisionTimeConfig.from_manifest(invalid_manifest)

    def test_early_clinical_prompt_defines_blinded_historical_decision(self) -> None:
        bundle = EpisodeBundle.load(_fixture_root(), include_private=False)
        manifest = bundle.manifest.model_copy(
            update={
                'task_type': EARLY_CLINICAL_ARM_PRIORITIZATION_TASK,
                'reward_version': 'v1.0',
                'required_dimensions': ['regimen_definition', 'endpoint_alignment'],
            }
        )
        clinical_bundle = replace(bundle, manifest=manifest)

        prompt = build_episode_prompt(clinical_bundle)

        self.assertIn('"task_type": "early_clinical_arm_prioritization"', prompt)
        self.assertIn('blinded early-clinical vaccine regimens', prompt)
        self.assertIn('frozen pre-results protocol evidence', prompt)
        self.assertIn('episode-defined proxy advancement objective, not by clinical efficacy', prompt)
        self.assertIn('clears the episode-declared threshold', prompt)
        self.assertIn('Apply only the endpoint horizon, control normalization, aggregation', prompt)
        self.assertIn('threshold, and grade bins stated in the visible episode evidence', prompt)
        self.assertIn('"regimen_definition"', prompt)
        self.assertIn('"endpoint_alignment"', prompt)
        self.assertIn('infer unshown results', prompt)

    def test_preclinical_fixture_is_a_solvable_v1_episode(self) -> None:
        bundle = EpisodeBundle.load(_preclinical_fixture_root(), include_private=True)

        self.assertEqual(bundle.manifest.task_type, PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK)
        self.assertEqual(bundle.manifest.reward_version, 'v1.0')
        self.assertEqual(bundle.manifest_sha256, '02bc9fd4ffdbdd1d26f3dcfff25cdfb9995f7c0c43ec5a2c50c46e859fcfea5c')
        self.assertNotIn('POST-CUTOFF CANARY', build_episode_prompt(bundle))
        score = make_submission_evaluator(bundle).score(oracle_submission(bundle))
        self.assertEqual(score.reward, 1.0)
        self.assertEqual(score.metrics()['ranking_reward'], 1.0)


if __name__ == '__main__':
    unittest.main()
