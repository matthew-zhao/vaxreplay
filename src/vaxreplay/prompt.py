"""Closed-book VaxReplay prompt construction."""

from __future__ import annotations

import enum
import json

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_schema import (
    ANTIGEN_TARGET_PRIORITIZATION_TASK,
    EARLY_CLINICAL_ARM_PRIORITIZATION_TASK,
    PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK,
    RANKING_REWARD_VERSION,
)

SYSTEM_PROMPT = """You are participating in VaxReplay V0, a closed-book benchmark of
evidence-grounded antigen-target prioritization. Use only the episode evidence in the user message.
Do not use external knowledge, inferred future events, or sources that are not shown. Return exactly
one JSON object and no surrounding prose."""

V1_SYSTEM_PROMPT = """You are participating in VaxReplay V1, a closed-book benchmark of
evidence-grounded candidate ranking. Use only the episode evidence in the user message. Do not use
external knowledge, inferred future events, or sources that are not shown. Return exactly one JSON
object and no surrounding prose."""


class PromptVariant(str, enum.Enum):
    """Fixed model-facing views used for contamination sensitivity analyses."""

    FULL = 'full'
    BIBLIOGRAPHICALLY_SCRUBBED = 'bibliographically_scrubbed'
    NO_EVIDENCE = 'no_evidence'


def build_system_prompt(bundle: EpisodeBundle) -> str:
    if bundle.manifest.reward_version == RANKING_REWARD_VERSION:
        return V1_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def build_episode_prompt(
    bundle: EpisodeBundle,
    *,
    variant: PromptVariant = PromptVariant.FULL,
) -> str:
    manifest = bundle.manifest
    visible_evidence = list(bundle.visible_evidence)
    if variant == PromptVariant.NO_EVIDENCE:
        rendered_evidence: list[dict[str, object]] = []
    else:
        rendered_evidence = [
            {
                'evidence_id': evidence.evidence_id,
                'source_type': evidence.source_type,
                'available_at': evidence.available_at.isoformat(),
                'title': (
                    f'Historical source {index + 1}'
                    if variant == PromptVariant.BIBLIOGRAPHICALLY_SCRUBBED
                    else evidence.title
                ),
                'body': evidence.body,
                'related_candidate_ids': evidence.related_candidate_ids,
            }
            for index, evidence in enumerate(visible_evidence)
        ]
    episode = {
        'episode_id': manifest.episode_id,
        'decision_at': manifest.decision_at.isoformat(),
        'manifest_sha256': bundle.manifest_sha256,
        'portfolio_size': manifest.portfolio_size,
        'required_dimensions': manifest.required_dimensions,
        'forecast_targets': [target.model_dump(mode='json') for target in manifest.forecast_targets],
        'candidate_ids': [candidate.candidate_id for candidate in bundle.candidates if candidate.eligible],
        'evidence': rendered_evidence,
    }
    if variant != PromptVariant.FULL:
        episode['prompt_variant'] = variant.value
    if manifest.reward_version == RANKING_REWARD_VERSION:
        episode['reward_version'] = manifest.reward_version
        episode['ranking_objective'] = {
            'ndcg_at_portfolio_size': 0.50,
            'strict_pairwise_concordance': 0.25,
            'normalized_top_k_set_utility': 0.25,
        }
    if manifest.task_type != ANTIGEN_TARGET_PRIORITIZATION_TASK:
        episode['task_type'] = manifest.task_type
    output_contract = {
        'schema_version': 'vaxreplay.v0.1',
        'episode_id': manifest.episode_id,
        'manifest_sha256': bundle.manifest_sha256,
        'ranking': ['every eligible candidate ID exactly once, best first'],
        'forecasts': [
            {
                'candidate_id': 'candidate ID',
                'target_id': 'forecast target ID',
                'horizon_days': 'forecast horizon from the episode',
                'probability': 'number from 0 to 1',
            }
        ],
        'assessments': [
            {
                'candidate_id': 'top-portfolio candidate ID',
                'dimension': 'one required dimension',
                'conclusion': 'favorable | concern | mixed | insufficient',
                'citations': [
                    {
                        'evidence_id': 'visible evidence ID',
                        'stance': 'support | concern',
                        'quote': 'an exact quote copied from that evidence body',
                    }
                ],
            }
        ],
    }
    if manifest.task_type == PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK:
        task_opening = (
            'Rank only the already-defined candidates for preclinical advancement and forecast later validation. '
            'Do not invent or modify candidates, and do not propose experimental procedures. '
        )
    elif manifest.task_type == EARLY_CLINICAL_ARM_PRIORITIZATION_TASK:
        task_opening = (
            'Prioritize only the already-defined, blinded early-clinical vaccine regimens using only frozen '
            'pre-results protocol evidence. Rank the regimens by the episode-defined proxy advancement '
            'objective, not by clinical efficacy. Forecast the probability that each regimen clears the '
            'episode-declared threshold. Apply only the endpoint horizon, control normalization, aggregation, '
            'threshold, and grade bins stated in the visible episode evidence. Do not invent or modify '
            'regimens, infer unshown results, or propose experimental procedures. '
        )
    else:
        task_opening = (
            'Candidate ranking is the primary V1 decision output. '
            if manifest.reward_version == RANKING_REWARD_VERSION
            else ''
        ) + 'Rank the candidates and forecast functional validation. '
    evidence_instruction = (
        'This diagnostic view intentionally contains no episode evidence. Do not recover or use external sources. '
        'Assessments may use empty citation lists. '
        if variant == PromptVariant.NO_EVIDENCE
        else ''
    )
    citation_instruction = 'substrings of the cited evidence body.'
    if evidence_instruction:
        citation_instruction += f' {evidence_instruction}'
    return (
        task_opening + 'The ranking must contain every candidate exactly '
        f'once. Provide one forecast for every candidate/target pair. For each of the top {manifest.portfolio_size} '
        'candidates, provide exactly one assessment for every required dimension. Citation quotes must be exact '
        f'{citation_instruction}\n\n'
        f'EPISODE\n{json.dumps(episode, ensure_ascii=False, indent=2)}\n\n'
        f'OUTPUT CONTRACT\n{json.dumps(output_contract, ensure_ascii=False, indent=2)}'
    )


def model_facing_payload_bytes(
    bundle: EpisodeBundle,
    *,
    variant: PromptVariant = PromptVariant.FULL,
) -> bytes:
    """Canonical bytes for the exact two messages visible to a worker model."""

    return canonical_json_bytes(
        {
            'messages': [
                {'role': 'system', 'content': build_system_prompt(bundle)},
                {'role': 'user', 'content': build_episode_prompt(bundle, variant=variant)},
            ],
            'prompt_variant': variant.value,
        }
    )
