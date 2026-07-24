#!/usr/bin/env python3
"""Run a deterministic end-to-end VaxReplay smoke test on fictional fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from vaxreplay.aggregation import aggregate_scores, make_suite_manifest, suite_manifest_sha256
from vaxreplay.baselines import oracle_submission, uniform_submission
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import ScoreVector
from vaxreplay.environment import VaxReplayEnvironment
from vaxreplay.ranking_schema import ScoreVectorV1
from vaxreplay.scoring import make_submission_evaluator

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _score_summary(score: ScoreVector | ScoreVectorV1) -> dict[str, object]:
    metrics = score.metrics()
    return {
        'status': score.status.value,
        **{name: metrics[name] for name in sorted(metrics)},
    }


def _exercise_episode(relative_path: str) -> tuple[dict[str, object], ScoreVector | ScoreVectorV1]:
    bundle = EpisodeBundle.load(FIXTURES / relative_path, include_private=True)
    evaluator = make_submission_evaluator(bundle)
    uniform_score = evaluator.score(uniform_submission(bundle))
    oracle = oracle_submission(bundle)
    oracle_score = evaluator.score(oracle)
    oracle_environment = VaxReplayEnvironment(bundle, evaluator)
    messages = oracle_environment.reset()
    oracle_step = oracle_environment.step(oracle.model_dump_json())
    malformed_environment = VaxReplayEnvironment(bundle, evaluator)
    malformed_environment.reset()
    malformed_step = malformed_environment.step('{not-json}')

    _require(oracle_score.reward == 1.0, f'{bundle.manifest.episode_id}: oracle did not score 1.0')
    _require(
        uniform_score.reward is not None and uniform_score.reward < oracle_score.reward,
        f'{bundle.manifest.episode_id}: uniform baseline did not score below oracle',
    )
    _require(
        oracle_step.reporting_reward == 1.0 and oracle_step.done,
        f'{bundle.manifest.episode_id}: oracle environment step failed',
    )
    _require(
        malformed_step.reporting_reward == -1.0 and malformed_step.done,
        f'{bundle.manifest.episode_id}: malformed response did not receive -1.0',
    )
    _require(
        malformed_step.info['status'] == 'invalid_schema',
        f'{bundle.manifest.episode_id}: malformed response had the wrong status',
    )

    return (
        {
            'episode_id': bundle.manifest.episode_id,
            'task_type': bundle.manifest.task_type,
            'reward_version': bundle.manifest.reward_version,
            'candidate_count': len(bundle.candidates),
            'visible_evidence_count': len(bundle.visible_evidence),
            'total_evidence_count': len(bundle.evidence),
            'prompt_message_count': len(messages),
            'uniform': _score_summary(uniform_score),
            'oracle': _score_summary(oracle_score),
            'environment': {
                'oracle_reward': oracle_step.reporting_reward,
                'malformed_reward': malformed_step.reporting_reward,
                'malformed_status': malformed_step.info['status'],
            },
        },
        oracle_score,
    )


def main() -> None:
    antigen, _ = _exercise_episode('synthetic_antigen_v0')
    preclinical, preclinical_oracle_score = _exercise_episode('synthetic_preclinical_v1')
    preclinical_bundle = EpisodeBundle.load(FIXTURES / 'synthetic_preclinical_v1', include_private=True)
    suite = make_suite_manifest('smoke-preclinical-v1', (preclinical_bundle,))
    valid_suite = aggregate_scores(suite, (preclinical_oracle_score,))
    missing_suite = aggregate_scores(suite, ())

    _require(valid_suite.all_episode_mean_environment_reward == 1.0, 'oracle suite did not score 1.0')
    _require(valid_suite.validity_rate == 1.0, 'oracle suite validity was not 1.0')
    _require(missing_suite.all_episode_mean_environment_reward == -1.0, 'missing suite did not score -1.0')
    _require(missing_suite.validity_rate == 0.0, 'missing suite validity was not 0.0')

    report = {
        'status': 'pass',
        'episodes': [antigen, preclinical],
        'suite': {
            'suite_id': suite.suite_id,
            'suite_manifest_sha256': suite_manifest_sha256(suite),
            'oracle_environment_reward': valid_suite.all_episode_mean_environment_reward,
            'oracle_validity_rate': valid_suite.validity_rate,
            'missing_environment_reward': missing_suite.all_episode_mean_environment_reward,
            'missing_validity_rate': missing_suite.validity_rate,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
