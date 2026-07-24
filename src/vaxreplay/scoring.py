"""Deterministic, non-LLM scoring for published VaxReplay train/dev episodes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import (
    RANKING_REWARD_VERSION,
    REWARD_VERSION,
    AssessmentConclusion,
    EvidenceStance,
    GoldEvidenceRecord,
    IssueCode,
    LabelCommitmentScheme,
    ScoreStatus,
    ScoreVector,
    Split,
    Submission,
    ValidationIssue,
)
from vaxreplay.ranking_schema import (
    V1_FORECAST_WEIGHT,
    V1_GROUNDING_WEIGHT,
    V1_NDCG_WEIGHT,
    V1_PAIRWISE_WEIGHT,
    V1_RANKING_WEIGHT,
    V1_TOP_K_UTILITY_WEIGHT,
    ScoreVectorV1,
)


@dataclass(frozen=True)
class ScoreWeights:
    forecast: float = 0.5
    ranking: float = 0.3
    grounding: float = 0.2

    def __post_init__(self) -> None:
        if min(self.forecast, self.ranking, self.grounding) < 0:
            raise ValueError('score weights must be non-negative')
        if not math.isclose(self.forecast + self.ranking + self.grounding, 1.0):
            raise ValueError('score weights must sum to 1')


class LocalSubmissionEvaluator:
    """Evaluator for public episodes whose private labels are intentionally local."""

    def __init__(
        self,
        bundle: EpisodeBundle,
        weights: ScoreWeights | None = None,
        *,
        allow_sealed_test: bool = False,
    ):
        labels = bundle.private_labels
        if labels is None:
            raise ValueError('local scoring requires private labels')
        _validate_sealed_evaluator_access(bundle, allow_sealed_test=allow_sealed_test)
        if bundle.manifest.reward_version != REWARD_VERSION:
            raise ValueError('LocalSubmissionEvaluator supports V0 only; use make_submission_evaluator')
        if weights is not None and weights != ScoreWeights():
            raise ValueError('official V0 reward weights are fixed')
        self._bundle = bundle
        self._labels = labels
        self._weights = weights or ScoreWeights()

    def score(self, submission: Submission) -> ScoreVector | ScoreVectorV1:
        issues = self._validate_submission(submission)
        if issues:
            leakage_codes = {
                IssueCode.LEAK_NON_MANIFEST_SOURCE,
                IssueCode.LEAK_POST_CUTOFF_SOURCE,
            }
            return ScoreVector(
                episode_id=self._bundle.manifest.episode_id,
                manifest_sha256=self._bundle.manifest_sha256,
                labels_sha256=self._bundle.manifest.labels_sha256,
                status=(
                    ScoreStatus.INVALID_LEAKAGE
                    if any(issue.code in leakage_codes for issue in issues)
                    else ScoreStatus.INVALID_SCHEMA
                ),
                issues=issues,
            )

        forecast_brier = self._forecast_brier(submission)
        forecast_reward = 1.0 - forecast_brier
        ndcg_at_k = self._ndcg_at_k(submission)
        grounding_precision, grounding_recall, grounding_f1, assessment_accuracy = self._grounding_scores(submission)
        grounding_reward = grounding_f1 * assessment_accuracy
        reward = min(
            1.0,
            max(
                0.0,
                self._weights.forecast * forecast_reward
                + self._weights.ranking * ndcg_at_k
                + self._weights.grounding * grounding_reward,
            ),
        )
        return ScoreVector(
            episode_id=self._bundle.manifest.episode_id,
            manifest_sha256=self._bundle.manifest_sha256,
            labels_sha256=self._bundle.manifest.labels_sha256,
            status=ScoreStatus.VALID,
            reward=reward,
            forecast_brier=forecast_brier,
            forecast_reward=forecast_reward,
            ndcg_at_k=ndcg_at_k,
            grounding_precision=grounding_precision,
            grounding_recall=grounding_recall,
            grounding_f1=grounding_f1,
            assessment_accuracy=assessment_accuracy,
            grounding_reward=grounding_reward,
        )

    def _validate_submission(self, submission: Submission) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        manifest = self._bundle.manifest
        if submission.episode_id != manifest.episode_id:
            issues.append(
                ValidationIssue(
                    code=IssueCode.EPISODE_MISMATCH,
                    detail=f'expected episode {manifest.episode_id}, got {submission.episode_id}',
                )
            )
        if submission.manifest_sha256 != self._bundle.manifest_sha256:
            issues.append(
                ValidationIssue(
                    code=IssueCode.MANIFEST_HASH_MISMATCH,
                    detail='submission manifest hash does not match the loaded episode',
                )
            )
        if submission.ranking != list(dict.fromkeys(submission.ranking)) or set(submission.ranking) != set(
            manifest.candidate_ids
        ):
            issues.append(
                ValidationIssue(
                    code=IssueCode.INVALID_RANKING,
                    detail='ranking must contain every eligible candidate exactly once',
                )
            )

        expected_forecasts = {
            (candidate_id, target.target_id, target.horizon_days)
            for candidate_id in manifest.candidate_ids
            for target in manifest.forecast_targets
        }
        forecast_keys = [
            (forecast.candidate_id, forecast.target_id, forecast.horizon_days) for forecast in submission.forecasts
        ]
        if set(forecast_keys) != expected_forecasts or len(forecast_keys) != len(set(forecast_keys)):
            issues.append(
                ValidationIssue(
                    code=IssueCode.INVALID_FORECASTS,
                    detail='forecasts must cover every candidate and target exactly once',
                )
            )

        top_candidates = set(submission.ranking[: manifest.portfolio_size])
        required_assessments = {
            (candidate_id, dimension) for candidate_id in top_candidates for dimension in manifest.required_dimensions
        }
        assessment_keys = [(assessment.candidate_id, assessment.dimension) for assessment in submission.assessments]
        if set(assessment_keys) != required_assessments or len(assessment_keys) != len(set(assessment_keys)):
            issues.append(
                ValidationIssue(
                    code=IssueCode.INVALID_ASSESSMENTS,
                    detail='top-k candidates require exactly one assessment for every required dimension',
                )
            )

        evidence_by_id = self._bundle.evidence_by_id
        for assessment in submission.assessments:
            for citation in assessment.citations:
                evidence = evidence_by_id.get(citation.evidence_id)
                if evidence is None:
                    issues.append(
                        ValidationIssue(
                            code=IssueCode.LEAK_NON_MANIFEST_SOURCE,
                            detail=f'citation references unknown evidence {citation.evidence_id}',
                        )
                    )
                elif evidence.available_at > manifest.decision_at:
                    issues.append(
                        ValidationIssue(
                            code=IssueCode.LEAK_POST_CUTOFF_SOURCE,
                            detail=f'citation references post-cutoff evidence {citation.evidence_id}',
                        )
                    )
                elif citation.quote not in evidence.body:
                    issues.append(
                        ValidationIssue(
                            code=IssueCode.INVALID_CITATION_QUOTE,
                            detail=f'quote is not present in evidence {citation.evidence_id}',
                        )
                    )
        return issues

    def _forecast_brier(self, submission: Submission) -> float:
        forecast_by_key = {
            (forecast.candidate_id, forecast.target_id, forecast.horizon_days): forecast.probability
            for forecast in submission.forecasts
        }
        squared_errors = [
            (forecast_by_key[(outcome.candidate_id, outcome.target_id, outcome.horizon_days)] - outcome.outcome) ** 2
            for outcome in self._labels.outcomes
            if outcome.outcome is not None
        ]
        if not squared_errors:
            raise ValueError('at least one non-censored outcome is required for scoring')
        return sum(squared_errors) / len(squared_errors)

    def _ndcg_at_k(self, submission: Submission) -> float:
        utility_by_candidate = {outcome.candidate_id: outcome.candidate_utility for outcome in self._labels.outcomes}
        k = self._bundle.manifest.portfolio_size
        dcg = self._discounted_gain(submission.ranking[:k], utility_by_candidate)
        ideal = sorted(utility_by_candidate, key=lambda candidate_id: utility_by_candidate[candidate_id], reverse=True)[
            :k
        ]
        ideal_dcg = self._discounted_gain(ideal, utility_by_candidate)
        return 1.0 if ideal_dcg == 0.0 else min(1.0, max(0.0, dcg / ideal_dcg))

    @staticmethod
    def _discounted_gain(ranking: list[str], utility_by_candidate: Mapping[str, float | int]) -> float:
        return sum(
            (2 ** utility_by_candidate[candidate_id] - 1) / math.log2(index + 2)
            for index, candidate_id in enumerate(ranking)
        )

    def _grounding_scores(self, submission: Submission) -> tuple[float, float, float, float]:
        top_candidates = set(submission.ranking[: self._bundle.manifest.portfolio_size])
        gold = [record for record in self._labels.evidence_gold if record.candidate_id in top_candidates]
        expected_conclusions = {
            (record.candidate_id, record.dimension): record.conclusion
            for record in self._labels.assessments_gold
            if record.candidate_id in top_candidates
        }
        submitted_conclusions = {
            (assessment.candidate_id, assessment.dimension): assessment.conclusion
            for assessment in submission.assessments
        }
        assessment_accuracy = sum(
            submitted_conclusions[key] == expected for key, expected in expected_conclusions.items()
        ) / len(expected_conclusions)
        predictions = {
            (
                assessment.candidate_id,
                assessment.dimension,
                assessment.conclusion,
                citation.evidence_id,
                citation.stance,
                citation.quote.strip(),
            )
            for assessment in submission.assessments
            if assessment.candidate_id in top_candidates
            for citation in assessment.citations
        }
        matched_gold: set[int] = set()
        matched_predictions = 0
        for prediction in sorted(
            predictions,
            key=lambda value: (value[0], value[1], value[2].value, value[3], value[4].value, value[5]),
        ):
            match = self._matching_gold_index(prediction, gold, expected_conclusions, matched_gold)
            if match is not None:
                matched_predictions += 1
                matched_gold.add(match)

        precision = matched_predictions / len(predictions) if predictions else (1.0 if not gold else 0.0)
        recall = len(matched_gold) / len(gold) if gold else 1.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1, assessment_accuracy

    @staticmethod
    def _matching_gold_index(
        prediction: tuple[str, str, AssessmentConclusion, str, EvidenceStance, str],
        gold: list[GoldEvidenceRecord],
        expected_conclusions: dict[tuple[str, str], AssessmentConclusion],
        already_matched: set[int],
    ) -> int | None:
        candidate_id, dimension, conclusion, evidence_id, stance, quote = prediction
        if conclusion != expected_conclusions[(candidate_id, dimension)]:
            return None
        for index, record in enumerate(gold):
            if index not in already_matched and (
                record.candidate_id == candidate_id
                and record.dimension == dimension
                and record.evidence_id == evidence_id
                and record.stance == stance
                and record.quote == quote
            ):
                return index
        return None


class RankingSubmissionEvaluator(LocalSubmissionEvaluator):
    """Fixed-weight V1 scorer with independent, complete integer ranking labels."""

    def __init__(self, bundle: EpisodeBundle, *, allow_sealed_test: bool = False):
        labels = bundle.private_labels
        ranking_labels = bundle.ranking_labels
        if labels is None or ranking_labels is None:
            raise ValueError('V1 local scoring requires private outcome and ranking labels')
        _validate_sealed_evaluator_access(bundle, allow_sealed_test=allow_sealed_test)
        if bundle.manifest.reward_version != RANKING_REWARD_VERSION:
            raise ValueError('RankingSubmissionEvaluator requires reward_version v1.0')
        self._bundle = bundle
        self._labels = labels
        self._ranking_labels = ranking_labels

    def score(self, submission: Submission) -> ScoreVectorV1:
        issues = self._validate_submission(submission)
        if issues:
            leakage_codes = {
                IssueCode.LEAK_NON_MANIFEST_SOURCE,
                IssueCode.LEAK_POST_CUTOFF_SOURCE,
            }
            return ScoreVectorV1(
                episode_id=self._bundle.manifest.episode_id,
                manifest_sha256=self._bundle.manifest_sha256,
                labels_sha256=self._bundle.manifest.labels_sha256,
                status=(
                    ScoreStatus.INVALID_LEAKAGE
                    if any(issue.code in leakage_codes for issue in issues)
                    else ScoreStatus.INVALID_SCHEMA
                ),
                issues=issues,
            )

        forecast_brier = self._forecast_brier(submission)
        forecast_reward = 1.0 - forecast_brier
        ndcg_at_k = self._ranking_ndcg_at_k(submission)
        pairwise_concordance = self._pairwise_concordance(submission)
        top_k_utility = self._top_k_utility(submission)
        ranking_reward = (
            V1_NDCG_WEIGHT * ndcg_at_k
            + V1_PAIRWISE_WEIGHT * pairwise_concordance
            + V1_TOP_K_UTILITY_WEIGHT * top_k_utility
        )
        grounding_precision, grounding_recall, grounding_f1, assessment_accuracy = self._grounding_scores(submission)
        grounding_reward = grounding_f1 * assessment_accuracy
        reward = (
            V1_FORECAST_WEIGHT * forecast_reward
            + V1_RANKING_WEIGHT * ranking_reward
            + V1_GROUNDING_WEIGHT * grounding_reward
        )
        return ScoreVectorV1(
            episode_id=self._bundle.manifest.episode_id,
            manifest_sha256=self._bundle.manifest_sha256,
            labels_sha256=self._bundle.manifest.labels_sha256,
            status=ScoreStatus.VALID,
            reward=reward,
            forecast_brier=forecast_brier,
            forecast_reward=forecast_reward,
            ndcg_at_k=ndcg_at_k,
            pairwise_concordance=pairwise_concordance,
            top_k_utility=top_k_utility,
            ranking_reward=ranking_reward,
            grounding_precision=grounding_precision,
            grounding_recall=grounding_recall,
            grounding_f1=grounding_f1,
            assessment_accuracy=assessment_accuracy,
            grounding_reward=grounding_reward,
        )

    def _ranking_ndcg_at_k(self, submission: Submission) -> float:
        grades = self._grade_by_candidate()
        k = self._bundle.manifest.portfolio_size
        dcg = self._discounted_gain(submission.ranking[:k], grades)
        ideal = sorted(grades, key=lambda candidate_id: (-grades[candidate_id], candidate_id))[:k]
        ideal_dcg = self._discounted_gain(ideal, grades)
        if ideal_dcg <= 0.0:
            raise ValueError('V1 ranking labels require positive ideal DCG')
        return dcg / ideal_dcg

    def _pairwise_concordance(self, submission: Submission) -> float:
        grades = self._grade_by_candidate()
        position = {candidate_id: index for index, candidate_id in enumerate(submission.ranking)}
        comparable_pairs = 0
        concordant_pairs = 0
        candidate_ids = self._bundle.manifest.candidate_ids
        for left_index, left_id in enumerate(candidate_ids):
            for right_id in candidate_ids[left_index + 1 :]:
                left_grade = grades[left_id]
                right_grade = grades[right_id]
                if left_grade == right_grade:
                    continue
                comparable_pairs += 1
                higher_id, lower_id = (left_id, right_id) if left_grade > right_grade else (right_id, left_id)
                concordant_pairs += position[higher_id] < position[lower_id]
        if comparable_pairs == 0:
            raise ValueError('V1 ranking labels require at least one strict pair')
        return concordant_pairs / comparable_pairs

    def _top_k_utility(self, submission: Submission) -> float:
        grades = self._grade_by_candidate()
        k = self._bundle.manifest.portfolio_size
        selected = sum(grades[candidate_id] for candidate_id in submission.ranking[:k])
        sorted_grades = sorted(grades.values())
        worst = sum(sorted_grades[:k])
        best = sum(sorted_grades[-k:])
        if best == worst:
            raise ValueError('V1 ranking labels have a degenerate top-k utility range')
        return (selected - worst) / (best - worst)

    def _grade_by_candidate(self) -> dict[str, int]:
        grades = {label.candidate_id: label.relevance_grade for label in self._ranking_labels}
        if any(grade is None for grade in grades.values()):
            raise ValueError('official V1 scoring cannot use censored ranking labels')
        return {candidate_id: grade for candidate_id, grade in grades.items() if grade is not None}


type SubmissionEvaluator = LocalSubmissionEvaluator | RankingSubmissionEvaluator


def _validate_sealed_evaluator_access(
    bundle: EpisodeBundle,
    *,
    allow_sealed_test: bool,
) -> None:
    if bundle.manifest.split != Split.TEST:
        return
    if not allow_sealed_test:
        raise ValueError('sealed test episodes require a separate private evaluator service')
    if bundle.manifest.label_commitment_scheme != LabelCommitmentScheme.HMAC_SHA256:
        raise ValueError('sealed test scoring requires HMAC-SHA256 label commitments')


def make_submission_evaluator(
    bundle: EpisodeBundle,
    *,
    allow_sealed_test: bool = False,
) -> SubmissionEvaluator:
    if bundle.manifest.reward_version == REWARD_VERSION:
        return LocalSubmissionEvaluator(bundle, allow_sealed_test=allow_sealed_test)
    if bundle.manifest.reward_version == RANKING_REWARD_VERSION:
        return RankingSubmissionEvaluator(bundle, allow_sealed_test=allow_sealed_test)
    raise ValueError(f'unsupported reward version {bundle.manifest.reward_version}')
