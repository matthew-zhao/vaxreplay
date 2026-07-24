from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from vaxreplay.agentic.schema import (
    AgenticAliasPermutationReceipt,
    AgenticArtifactKind,
    AgenticBuildPolicy,
    AgenticCandidateAliasAssignment,
    AgenticDiscoveredCandidate,
    AgenticDiscoveredSource,
    AgenticDiscoveryDisposition,
    AgenticDiscoveryManifest,
    AgenticSourceAliasAssignment,
    AgenticTaskEnvelope,
    AgenticTransformationReceipt,
    AgenticTransformCommitment,
    AgenticWorkspaceSource,
    agentic_model_sha256,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import EpisodeManifest, ForecastTarget, Split


def bind_episode_manifest(task: AgenticTaskEnvelope) -> tuple[AgenticTaskEnvelope, EpisodeManifest]:
    episode = EpisodeManifest(
        episode_id=task.episode_id,
        lineage_group_id=f'fixture-lineage-{task.episode_id}',
        synthetic=True,
        task_type='early_clinical_arm_prioritization',
        split=Split.DEV,
        decision_at=task.decision_at,
        portfolio_size=task.portfolio_size,
        candidate_ids=list(task.candidate_ids),
        forecast_targets=[ForecastTarget(target_id='fixture-target', horizon_days=28)],
        required_dimensions=['fixture-evidence'],
        evidence_sha256=hashlib.sha256(b'fixture evidence').hexdigest(),
        candidates_sha256=hashlib.sha256(b'fixture candidates').hexdigest(),
        labels_sha256=hashlib.sha256(b'fixture labels').hexdigest(),
        adjudication_version='fixture-v1',
        reward_version='v1.0',
    )
    rebound = AgenticTaskEnvelope.model_validate(
        {**task.model_dump(), 'episode_manifest_sha256': agentic_model_sha256(episode)}
    )
    return rebound, episode


def selection_contract(
    task: AgenticTaskEnvelope,
    sources: tuple[AgenticWorkspaceSource, ...],
    transformations: tuple[AgenticTransformationReceipt, ...] = (),
    *,
    created_at: datetime | None = None,
) -> tuple[AgenticBuildPolicy, AgenticDiscoveryManifest]:
    """Build an explicitly retrospective fixture selection contract for unit tests."""

    contract_created_at = created_at or datetime(2026, 1, 1, tzinfo=UTC)
    candidate_key_by_id = {
        candidate_id: hashlib.sha256(f'fixture-private:{candidate_id}'.encode()).hexdigest()
        for candidate_id in task.candidate_ids
    }
    alias_scheme_sha256 = hashlib.sha256(b'fixture secret aliases').hexdigest()
    alias_seed_commitment_sha256 = hashlib.sha256(b'fixture secret seed').hexdigest()
    alias_generator_executable_sha256 = hashlib.sha256(b'fixture alias generator executable').hexdigest()
    alias_generator_config_sha256 = hashlib.sha256(b'fixture alias generator config').hexdigest()
    candidate_assignments = tuple(
        AgenticCandidateAliasAssignment(
            candidate_key_commitment_sha256=candidate_key_by_id[candidate_id],
            public_candidate_id=candidate_id,
            presentation_index=index,
        )
        for index, candidate_id in enumerate(task.candidate_ids)
    )
    source_assignments = tuple(
        AgenticSourceAliasAssignment(
            private_source_key_commitment_sha256=hashlib.sha256(
                f'fixture-private-source:{source.source_id}'.encode()
            ).hexdigest(),
            artifact_sha256=source.sha256,
            artifact_bytes=source.byte_count,
            public_source_id=source.source_id,
            public_path=source.path,
            public_title=source.display_title,
            presentation_index=index,
        )
        for index, source in enumerate(sources)
    )
    alias_receipt_payload = b'fixture alias permutation execution receipt'
    alias_receipt = AgenticAliasPermutationReceipt(
        receipt_id=f'fixture-alias-receipt-{task.task_id}',
        task_id=task.task_id,
        alias_scheme_sha256=alias_scheme_sha256,
        alias_seed_commitment_sha256=alias_seed_commitment_sha256,
        permutation_algorithm_id='fixture-keyed-fisher-yates-v1',
        generator_id='fixture-alias-generator',
        generator_version='1',
        generator_executable_sha256=alias_generator_executable_sha256,
        generator_config_sha256=alias_generator_config_sha256,
        execution_receipt_sha256=hashlib.sha256(alias_receipt_payload).hexdigest(),
        execution_receipt_bytes=len(alias_receipt_payload),
        generated_at=contract_created_at,
        candidate_order_sha256=hashlib.sha256(canonical_json_bytes(list(task.candidate_ids))).hexdigest(),
        source_order_sha256=hashlib.sha256(canonical_json_bytes([source.source_id for source in sources])).hexdigest(),
        candidate_assignments=candidate_assignments,
        source_assignments=source_assignments,
    )
    policy = AgenticBuildPolicy(
        policy_id=f'fixture-policy-{task.task_id}',
        task_id=task.task_id,
        decision_at=task.decision_at,
        created_at=contract_created_at,
        discovery_spec_sha256=hashlib.sha256(b'fixture discovery').hexdigest(),
        candidate_rule_sha256=hashlib.sha256(b'fixture candidates').hexdigest(),
        inclusion_rule_sha256=hashlib.sha256(b'fixture inclusion').hexdigest(),
        deduplication_rule_sha256=hashlib.sha256(b'fixture dedup').hexdigest(),
        distractor_rule_sha256=hashlib.sha256(b'fixture distractors').hexdigest(),
        alias_scheme_sha256=alias_scheme_sha256,
        alias_seed_commitment_sha256=alias_seed_commitment_sha256,
        alias_permutation_algorithm_id=alias_receipt.permutation_algorithm_id,
        alias_generator_id=alias_receipt.generator_id,
        alias_generator_version=alias_receipt.generator_version,
        alias_generator_executable_sha256=alias_generator_executable_sha256,
        alias_generator_config_sha256=alias_generator_config_sha256,
        protected_outcome_namespace_sha256=hashlib.sha256(b'fixture protected-corpus selection policy').hexdigest(),
        allowed_transforms=tuple(
            sorted(
                (
                    AgenticTransformCommitment(
                        transform_id=receipt.transform_id,
                        transform_version=receipt.transform_version,
                        executable_sha256=receipt.executable_sha256,
                        config_sha256=receipt.config_sha256,
                    )
                    for receipt in transformations
                ),
                key=lambda value: (value.transform_id, value.transform_version),
            )
        ),
    )
    source_records = tuple(
        sorted(
            (
                AgenticDiscoveredSource(
                    discovery_id=f'discovery-{source.source_id}',
                    artifact_sha256=source.sha256,
                    artifact_bytes=source.byte_count,
                    selected_temporal_proof_id=source.selected_proof_id or 'derived-not-discovered',
                    effective_available_at_upper=source.effective_available_at_upper,
                    disposition=AgenticDiscoveryDisposition.INCLUDED,
                    reason_code='fixture-included',
                    workspace_source_id=source.source_id,
                )
                for source in sources
                if source.artifact_kind == AgenticArtifactKind.RAW
            ),
            key=lambda value: value.discovery_id,
        )
    )
    candidates = tuple(
        sorted(
            (
                AgenticDiscoveredCandidate(
                    candidate_key_commitment_sha256=candidate_key_by_id[candidate_id],
                    disposition=AgenticDiscoveryDisposition.INCLUDED,
                    reason_code='fixture-included',
                    public_candidate_id=candidate_id,
                )
                for candidate_id in task.candidate_ids
            ),
            key=lambda value: value.candidate_key_commitment_sha256,
        )
    )
    discovery = AgenticDiscoveryManifest(
        manifest_id=f'fixture-discovery-{task.task_id}',
        task_id=task.task_id,
        build_policy_sha256=agentic_model_sha256(policy),
        discovery_capture_receipt_sha256=hashlib.sha256(b'fixture capture receipt').hexdigest(),
        alias_permutation_receipt=alias_receipt,
        created_at=contract_created_at,
        sources=source_records,
        candidates=candidates,
    )
    return policy, discovery
