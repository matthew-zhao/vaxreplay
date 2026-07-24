from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from vaxreplay.baselines import oracle_submission, uniform_submission
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import (
    AssessmentConclusion,
    CandidateAssessment,
    Citation,
    EvidenceStance,
    IssueCode,
    ScoreStatus,
    Split,
)
from vaxreplay.scoring import LocalSubmissionEvaluator, ScoreWeights, make_submission_evaluator


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'


class LocalSubmissionEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EpisodeBundle.load(_fixture_root(), include_private=True)
        self.evaluator = LocalSubmissionEvaluator(self.bundle)

    def test_oracle_reaches_reward_ceiling(self) -> None:
        score = self.evaluator.score(oracle_submission(self.bundle))

        self.assertEqual(score.status, ScoreStatus.VALID)
        self.assertEqual(score.reward, 1.0)
        self.assertEqual(score.forecast_brier, 0.0)
        self.assertEqual(score.ndcg_at_k, 1.0)
        self.assertEqual(score.grounding_f1, 1.0)
        self.assertEqual(score.assessment_accuracy, 1.0)
        self.assertEqual(score.grounding_reward, 1.0)

    def test_uniform_baseline_is_valid_but_below_oracle(self) -> None:
        score = self.evaluator.score(uniform_submission(self.bundle))

        self.assertEqual(score.status, ScoreStatus.VALID)
        self.assertEqual(score.forecast_brier, 0.25)
        self.assertEqual(score.grounding_f1, 0.0)
        self.assertEqual(score.assessment_accuracy, 0.0)
        self.assertEqual(score.reward, 0.5886746317431102)
        self.assertEqual(
            set(score.model_dump(mode='json')),
            {
                'episode_id',
                'manifest_sha256',
                'labels_sha256',
                'reward_version',
                'status',
                'reward',
                'forecast_brier',
                'forecast_reward',
                'ndcg_at_k',
                'grounding_precision',
                'grounding_recall',
                'grounding_f1',
                'assessment_accuracy',
                'grounding_reward',
                'issues',
            },
        )

    def test_official_v0_weights_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, 'weights are fixed'):
            LocalSubmissionEvaluator(
                self.bundle,
                ScoreWeights(forecast=0.4, ranking=0.4, grounding=0.2),
            )

    def test_post_cutoff_citation_is_a_hard_leakage_failure(self) -> None:
        submission = oracle_submission(self.bundle)
        first = submission.assessments[0]
        leaky_assessment = CandidateAssessment(
            candidate_id=first.candidate_id,
            dimension=first.dimension,
            conclusion=first.conclusion,
            citations=[
                *first.citations,
                Citation(
                    evidence_id='ev-future-canary',
                    stance=EvidenceStance.SUPPORT,
                    quote='POST-CUTOFF CANARY: target 42 later produced the strongest validation result.',
                ),
            ],
        )
        leaky_submission = submission.model_copy(
            update={'assessments': [leaky_assessment, *submission.assessments[1:]]}
        )

        score = self.evaluator.score(leaky_submission)

        self.assertEqual(score.status, ScoreStatus.INVALID_LEAKAGE)
        self.assertIsNone(score.reward)
        self.assertIn(IssueCode.LEAK_POST_CUTOFF_SOURCE, {issue.code for issue in score.issues})

    def test_quote_must_be_an_exact_visible_substring(self) -> None:
        submission = oracle_submission(self.bundle)
        first = submission.assessments[0]
        invalid_assessment = first.model_copy(
            update={
                'citations': [
                    Citation(
                        evidence_id=first.citations[0].evidence_id,
                        stance=first.citations[0].stance,
                        quote='This quote was never present.',
                    )
                ]
            }
        )
        invalid_submission = submission.model_copy(
            update={'assessments': [invalid_assessment, *submission.assessments[1:]]}
        )

        score = self.evaluator.score(invalid_submission)

        self.assertEqual(score.status, ScoreStatus.INVALID_SCHEMA)
        self.assertIn(IssueCode.INVALID_CITATION_QUOTE, {issue.code for issue in score.issues})

    def test_conclusion_must_agree_with_the_grounded_stance(self) -> None:
        submission = oracle_submission(self.bundle)
        first = submission.assessments[0]
        contradictory_assessment = first.model_copy(update={'conclusion': AssessmentConclusion.CONCERN})
        contradictory_submission = submission.model_copy(
            update={'assessments': [contradictory_assessment, *submission.assessments[1:]]}
        )

        score = self.evaluator.score(contradictory_submission)

        self.assertEqual(score.status, ScoreStatus.VALID)
        self.assertLess(score.grounding_f1, 1.0)
        self.assertLess(score.reward, 1.0)

    def test_portfolio_flooding_is_invalid(self) -> None:
        submission = oracle_submission(self.bundle).model_copy(
            update={'ranking': ['target-17', 'target-88', 'target-42', 'target-42']}
        )

        score = self.evaluator.score(submission)

        self.assertEqual(score.status, ScoreStatus.INVALID_SCHEMA)
        self.assertIn(IssueCode.INVALID_RANKING, {issue.code for issue in score.issues})

    def test_non_manifest_source_is_a_hard_leakage_failure(self) -> None:
        submission = oracle_submission(self.bundle)
        first = submission.assessments[0]
        unknown_source_assessment = first.model_copy(
            update={
                'citations': [
                    *first.citations,
                    Citation(
                        evidence_id='ev-outside-corpus',
                        stance=EvidenceStance.SUPPORT,
                        quote='A sufficiently long but unlisted evidence quote.',
                    ),
                ]
            }
        )
        unknown_source_submission = submission.model_copy(
            update={'assessments': [unknown_source_assessment, *submission.assessments[1:]]}
        )

        score = self.evaluator.score(unknown_source_submission)

        self.assertEqual(score.status, ScoreStatus.INVALID_LEAKAGE)
        self.assertIn(IssueCode.LEAK_NON_MANIFEST_SOURCE, {issue.code for issue in score.issues})

    def test_duplicate_citation_variants_do_not_inflate_grounding(self) -> None:
        submission = oracle_submission(self.bundle)
        first = submission.assessments[0]
        flooded_assessment = first.model_copy(
            update={
                'citations': [
                    *first.citations,
                    Citation(
                        evidence_id='ev-surveillance-1',
                        stance=EvidenceStance.SUPPORT,
                        quote=(
                            'Target 17 was unchanged across 96 of 100 sampled isolates. '
                            'Target 42 varied in 38 of 100 isolates.'
                        ),
                    ),
                ]
            }
        )
        flooded_submission = submission.model_copy(
            update={'assessments': [flooded_assessment, *submission.assessments[1:]]}
        )

        score = self.evaluator.score(flooded_submission)

        self.assertEqual(score.status, ScoreStatus.VALID)
        self.assertLess(score.grounding_precision, 1.0)
        self.assertLess(score.reward, 1.0)

    def test_whole_document_citations_do_not_receive_span_credit(self) -> None:
        submission = oracle_submission(self.bundle)
        evidence_by_id = self.bundle.evidence_by_id
        whole_document_submission = submission.model_copy(
            update={
                'assessments': [
                    assessment.model_copy(
                        update={
                            'citations': [
                                citation.model_copy(update={'quote': evidence_by_id[citation.evidence_id].body})
                                for citation in assessment.citations
                            ]
                        }
                    )
                    for assessment in submission.assessments
                ]
            }
        )

        score = self.evaluator.score(whole_document_submission)

        self.assertEqual(score.status, ScoreStatus.VALID)
        self.assertEqual(score.grounding_f1, 0.0)
        self.assertLess(score.reward, 1.0)

    def test_oracle_is_unavailable_for_sealed_test_episode(self) -> None:
        test_bundle = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'split': Split.TEST}),
        )

        with self.assertRaisesRegex(ValueError, 'unavailable for sealed test'):
            oracle_submission(test_bundle)

    def test_sealed_test_scoring_requires_an_explicit_private_evaluator_scope(self) -> None:
        test_bundle = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'split': Split.TEST}),
        )

        with self.assertRaisesRegex(ValueError, 'separate private evaluator service'):
            make_submission_evaluator(test_bundle)

        with self.assertRaisesRegex(ValueError, 'requires HMAC-SHA256'):
            make_submission_evaluator(test_bundle, allow_sealed_test=True)


if __name__ == '__main__':
    unittest.main()
