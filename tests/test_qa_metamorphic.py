from __future__ import annotations

import unittest
from pathlib import Path

from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.qa.metamorphic import (
    DecisionTarget,
    DecisionTargetKind,
    ExpectedDirection,
    MetamorphicRelation,
    MetricExpectation,
    ResponseExpectation,
    audit_candidate_equivariance,
    audit_evidence_intervention,
    audit_nuisance_invariance,
    failed_findings,
)
from vaxreplay.scoring import make_submission_evaluator


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'


class MetamorphicQaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EpisodeBundle.load(_fixture_root(), include_private=True)
        self.submission = oracle_submission(self.bundle)
        self.score = make_submission_evaluator(self.bundle).score(self.submission)

    def _aliased_submission(self):
        reference_to_variant = {
            candidate_id: f'alias-{index}'
            for index, candidate_id in enumerate(self.bundle.manifest.candidate_ids, start=1)
        }
        variant_to_reference = {variant: reference for reference, variant in reference_to_variant.items()}
        aliased = self.submission.model_copy(
            update={
                'episode_id': 'aliased-episode',
                'manifest_sha256': 'a' * 64,
                'ranking': [reference_to_variant[candidate_id] for candidate_id in self.submission.ranking],
                'forecasts': [
                    forecast.model_copy(update={'candidate_id': reference_to_variant[forecast.candidate_id]})
                    for forecast in reversed(self.submission.forecasts)
                ],
                'assessments': [
                    assessment.model_copy(update={'candidate_id': reference_to_variant[assessment.candidate_id]})
                    for assessment in reversed(self.submission.assessments)
                ],
            }
        )
        return aliased, variant_to_reference

    def test_candidate_alias_equivariance_normalizes_structured_candidate_ids(self) -> None:
        aliased, variant_to_reference = self._aliased_submission()

        findings = audit_candidate_equivariance(
            self.submission,
            aliased,
            variant_to_reference=variant_to_reference,
            reference_score=self.score,
            variant_score=self.score,
            audit_id='alias-control-001',
        )

        self.assertTrue(all(finding.passed for finding in findings))
        self.assertEqual({finding.relation for finding in findings}, {MetamorphicRelation.CANDIDATE_EQUIVARIANCE})
        self.assertEqual(len({finding.finding_id for finding in findings}), len(findings))
        self.assertTrue(all(finding.as_dict()['finding_id'] == finding.finding_id for finding in findings))

    def test_candidate_order_shortcut_fails_ranking_equivariance(self) -> None:
        aliased, variant_to_reference = self._aliased_submission()
        shortcut = aliased.model_copy(update={'ranking': sorted(aliased.ranking)})

        findings = audit_candidate_equivariance(
            self.submission,
            shortcut,
            variant_to_reference=variant_to_reference,
        )

        failures = failed_findings(findings)
        self.assertEqual([finding.subject for finding in failures], ['response.ranking'])
        self.assertNotEqual(failures[0].reference_fingerprint, failures[0].variant_fingerprint)

    def test_candidate_map_must_be_a_total_bijection(self) -> None:
        aliased, variant_to_reference = self._aliased_submission()
        variant_to_reference.pop(next(iter(variant_to_reference)))

        with self.assertRaisesRegex(ValueError, 'cover variant IDs exactly'):
            audit_candidate_equivariance(
                self.submission,
                aliased,
                variant_to_reference=variant_to_reference,
            )

    def test_nuisance_invariance_ignores_nonsemantic_list_order_and_small_metric_noise(self) -> None:
        reordered = self.submission.model_copy(
            update={
                'forecasts': list(reversed(self.submission.forecasts)),
                'assessments': list(reversed(self.submission.assessments)),
            }
        )
        reference_score = {'reward': 0.8, 'grounding_reward': 0.7}
        variant_score = {'reward': 0.8 + 5e-10, 'grounding_reward': 0.7}

        findings = audit_nuisance_invariance(
            self.submission,
            reordered,
            reference_score=reference_score,
            variant_score=variant_score,
            tolerance=1e-9,
            audit_id='evidence-order-control',
        )

        self.assertTrue(all(finding.passed for finding in findings))
        self.assertEqual({finding.relation for finding in findings}, {MetamorphicRelation.NUISANCE_INVARIANCE})

    def test_nuisance_invariance_catches_response_and_score_sensitivity(self) -> None:
        changed_forecast = self.submission.forecasts[0].model_copy(update={'probability': 0.25})
        changed = self.submission.model_copy(update={'forecasts': [changed_forecast, *self.submission.forecasts[1:]]})

        findings = audit_nuisance_invariance(
            self.submission,
            changed,
            reference_score={'reward': 1.0},
            variant_score={'reward': 0.9},
        )

        self.assertEqual(
            {finding.subject for finding in failed_findings(findings)},
            {'response.forecasts', 'score.reward'},
        )

    def test_score_metric_disappearance_is_a_failure_not_a_silent_intersection(self) -> None:
        findings = audit_nuisance_invariance(
            self.submission,
            self.submission,
            reference_score={'reward': 1.0, 'grounding_reward': 1.0},
            variant_score={'reward': 1.0},
        )

        failure = failed_findings(findings)
        self.assertEqual(len(failure), 1)
        self.assertEqual(failure[0].subject, 'score.grounding_reward')
        self.assertIn('missing from variant score', failure[0].observed)

    def test_evidence_intervention_requires_decision_change_and_expected_metric_direction(self) -> None:
        targeted_forecast = self.submission.forecasts[0]
        changed_forecast = targeted_forecast.model_copy(update={'probability': 0.2})
        intervention = self.submission.model_copy(
            update={'forecasts': [changed_forecast, *self.submission.forecasts[1:]]}
        )

        findings = audit_evidence_intervention(
            self.submission,
            intervention,
            reference_score={'reward': 1.0, 'grounding_reward': 1.0},
            intervention_score={'reward': 0.7, 'grounding_reward': 0.5},
            expectations=(
                MetricExpectation('reward', ExpectedDirection.DECREASE, minimum_change=0.2),
                MetricExpectation('grounding_reward', ExpectedDirection.DECREASE, minimum_change=0.4),
            ),
            decision_targets=(
                DecisionTarget(
                    kind=DecisionTargetKind.FORECAST_PROBABILITY,
                    candidate_id=targeted_forecast.candidate_id,
                    target_id=targeted_forecast.target_id,
                    horizon_days=targeted_forecast.horizon_days,
                    minimum_change=0.1,
                ),
            ),
            audit_id='remove-decisive-evidence',
        )

        self.assertTrue(all(finding.passed for finding in findings))
        self.assertIn(targeted_forecast.candidate_id, findings[0].subject)
        self.assertEqual(findings[1].delta, -0.30000000000000004)

    def test_unrelated_decision_change_cannot_satisfy_targeted_evidence_sensitivity(self) -> None:
        targeted = self.submission.forecasts[0]
        unrelated = self.submission.forecasts[1]
        changed_unrelated = unrelated.model_copy(update={'probability': 0.0 if unrelated.probability > 0.5 else 1.0})
        intervention = self.submission.model_copy(
            update={
                'forecasts': [
                    targeted,
                    changed_unrelated,
                    *self.submission.forecasts[2:],
                ]
            }
        )

        findings = audit_evidence_intervention(
            self.submission,
            intervention,
            reference_score={'reward': 1.0},
            intervention_score={'reward': 0.5},
            expectations=(MetricExpectation('reward', ExpectedDirection.DECREASE, minimum_change=0.4),),
            decision_targets=(
                DecisionTarget(
                    kind=DecisionTargetKind.FORECAST_PROBABILITY,
                    candidate_id=targeted.candidate_id,
                    target_id=targeted.target_id,
                    horizon_days=targeted.horizon_days,
                    minimum_change=0.1,
                ),
            ),
        )

        self.assertFalse(findings[0].passed)
        self.assertEqual(findings[0].delta, 0.0)
        self.assertTrue(findings[1].passed)

    def test_change_expectation_without_explicit_decision_target_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, 'explicit decision target'):
            audit_evidence_intervention(
                self.submission,
                self.submission,
                reference_score={'reward': 1.0},
                intervention_score={'reward': 0.5},
                expectations=(MetricExpectation('reward', ExpectedDirection.DECREASE, minimum_change=0.4),),
            )

    def test_citation_only_change_does_not_fake_evidence_sensitivity(self) -> None:
        first_assessment = self.submission.assessments[0]
        citation_only = self.submission.model_copy(
            update={
                'assessments': [
                    first_assessment.model_copy(update={'citations': []}),
                    *self.submission.assessments[1:],
                ]
            }
        )

        findings = audit_evidence_intervention(
            self.submission,
            citation_only,
            reference_score={'grounding_reward': 1.0},
            intervention_score={'grounding_reward': 1.0},
            expectations=(MetricExpectation('grounding_reward', ExpectedDirection.DECREASE),),
            decision_targets=(
                DecisionTarget(
                    kind=DecisionTargetKind.ASSESSMENT_CONCLUSION,
                    candidate_id=first_assessment.candidate_id,
                    dimension=first_assessment.dimension,
                ),
            ),
        )

        failures = failed_findings(findings)
        self.assertEqual(len(failures), 2)
        self.assertIn(first_assessment.candidate_id, failures[0].subject)
        self.assertEqual(failures[1].subject, 'score.grounding_reward')

    def test_evidence_negative_control_can_expect_unchanged_decision_and_score(self) -> None:
        findings = audit_evidence_intervention(
            self.submission,
            self.submission,
            reference_score={'reward': 1.0},
            intervention_score={'reward': 1.0},
            expectations=(MetricExpectation('reward', ExpectedDirection.UNCHANGED),),
            response_expectation=ResponseExpectation.UNCHANGED,
        )

        self.assertTrue(all(finding.passed for finding in findings))


if __name__ == '__main__':
    unittest.main()
