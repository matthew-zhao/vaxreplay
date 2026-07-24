"""Deterministic VaxReplay baselines and oracle solvability checks."""

from __future__ import annotations

from collections import defaultdict

from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import (
    RANKING_REWARD_VERSION,
    AssessmentConclusion,
    CandidateAssessment,
    CandidateForecast,
    Citation,
    Split,
    Submission,
)


def uniform_submission(bundle: EpisodeBundle) -> Submission:
    manifest = bundle.manifest
    return Submission(
        episode_id=manifest.episode_id,
        manifest_sha256=bundle.manifest_sha256,
        ranking=list(manifest.candidate_ids),
        forecasts=[
            CandidateForecast(
                candidate_id=candidate_id,
                target_id=target.target_id,
                horizon_days=target.horizon_days,
                probability=0.5,
            )
            for candidate_id in manifest.candidate_ids
            for target in manifest.forecast_targets
        ],
        assessments=[
            CandidateAssessment(
                candidate_id=candidate_id,
                dimension=dimension,
                conclusion=AssessmentConclusion.INSUFFICIENT,
            )
            for candidate_id in manifest.candidate_ids[: manifest.portfolio_size]
            for dimension in manifest.required_dimensions
        ],
    )


def oracle_submission(bundle: EpisodeBundle) -> Submission:
    if bundle.manifest.split == Split.TEST:
        raise ValueError('oracle submission is unavailable for sealed test episodes')
    if bundle.private_labels is None:
        raise ValueError('oracle submission requires private labels')
    manifest = bundle.manifest
    if manifest.reward_version == RANKING_REWARD_VERSION:
        if bundle.ranking_labels is None:
            raise ValueError('V1 oracle submission requires private ranking labels')
        utility_by_candidate = {
            label.candidate_id: label.relevance_grade
            for label in bundle.ranking_labels
            if label.relevance_grade is not None
        }
    else:
        utility_by_candidate = {
            outcome.candidate_id: outcome.candidate_utility for outcome in bundle.private_labels.outcomes
        }
    outcome_by_key = {
        (outcome.candidate_id, outcome.target_id, outcome.horizon_days): outcome
        for outcome in bundle.private_labels.outcomes
    }
    conclusion_by_assessment = {
        (assessment.candidate_id, assessment.dimension): assessment.conclusion
        for assessment in bundle.private_labels.assessments_gold
    }
    if manifest.reward_version == RANKING_REWARD_VERSION:
        ranking = sorted(
            manifest.candidate_ids,
            key=lambda candidate_id: (-utility_by_candidate[candidate_id], candidate_id),
        )
    else:
        ranking = sorted(
            manifest.candidate_ids,
            key=lambda candidate_id: utility_by_candidate[candidate_id],
            reverse=True,
        )
    gold_by_assessment = defaultdict(list)
    for gold in bundle.private_labels.evidence_gold:
        gold_by_assessment[(gold.candidate_id, gold.dimension)].append(gold)

    assessments: list[CandidateAssessment] = []
    for candidate_id in ranking[: manifest.portfolio_size]:
        for dimension in manifest.required_dimensions:
            gold_records = gold_by_assessment[(candidate_id, dimension)]
            assessments.append(
                CandidateAssessment(
                    candidate_id=candidate_id,
                    dimension=dimension,
                    conclusion=conclusion_by_assessment[(candidate_id, dimension)],
                    citations=[
                        Citation(evidence_id=record.evidence_id, stance=record.stance, quote=record.quote)
                        for record in gold_records
                    ],
                )
            )

    forecasts: list[CandidateForecast] = []
    for candidate_id in manifest.candidate_ids:
        for target in manifest.forecast_targets:
            outcome = outcome_by_key[(candidate_id, target.target_id, target.horizon_days)].outcome
            forecasts.append(
                CandidateForecast(
                    candidate_id=candidate_id,
                    target_id=target.target_id,
                    horizon_days=target.horizon_days,
                    probability=0.5 if outcome is None else float(outcome),
                )
            )

    return Submission(
        episode_id=manifest.episode_id,
        manifest_sha256=bundle.manifest_sha256,
        ranking=ranking,
        forecasts=forecasts,
        assessments=assessments,
    )
