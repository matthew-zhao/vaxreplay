from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.agentic_helpers import bind_episode_manifest, selection_contract
from vaxreplay.agentic.admission import AgenticAdmissionError
from vaxreplay.agentic.contamination import (
    AGENTIC_AUDIT_COMPARISON_PREFIX,
    AGENTIC_AUDIT_IDENTIFIERS_KEY,
    AGENTIC_AUDIT_MANIFEST_KEY,
    AGENTIC_AUDIT_POLICY_KEY,
    AGENTIC_AUDIT_PROTECTED_CORPUS_KEY,
    AgenticProtectedCorpusArtifact,
    AgenticProtectedCorpusManifest,
    agentic_case_universe_sha256,
    make_agentic_audit_input,
    make_agentic_contamination_binding,
    verify_agentic_contamination_audit,
)
from vaxreplay.agentic.schema import (
    AgenticArtifactKind,
    AgenticAssuranceProfile,
    AgenticFactQuery,
    AgenticMediaType,
    AgenticTaskEnvelope,
    AgenticValueType,
    AgenticWorkspaceSource,
    ArtifactTemporalProof,
    AvailabilityInterval,
    TemporalProofKind,
)
from vaxreplay.agentic.workspace import build_agentic_workspace, model_visible_surface_bytes
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.contamination import (
    CalibrationPolicy,
    ContaminationAuditPolicy,
    ExactRetrievalConfig,
    JudgeCalibrationResult,
    JudgeVerdict,
    LlmJudgeOutput,
    PinnedLlmJudge,
    build_contamination_audit,
    make_audit_input,
    make_audit_manifest,
    make_llm_audit_run,
    retrieve_exact_candidates,
)

NOW = datetime(2026, 7, 13, 12, tzinfo=UTC)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _workspace(tmp_path: Path):
    source_bytes = b'candidate|dose\na|120\n'
    interval = AvailabilityInterval(
        lower_at=datetime(2020, 2, 1, tzinfo=UTC),
        upper_at=datetime(2020, 2, 1, 23, 59, 59, tzinfo=UTC),
        precision='day',
        timezone_basis='UTC upper bound',
    )
    proof_bytes = b'proof'
    proof = ArtifactTemporalProof(
        proof_id='proof-source',
        kind=TemporalProofKind.FIXTURE,
        artifact_sha256=_sha(source_bytes),
        artifact_bytes=len(source_bytes),
        witnessed=interval,
        authority_id='fixture',
        proof_sha256=_sha(proof_bytes),
        proof_bytes=len(proof_bytes),
        verification_uri='fixture://proof',
    )
    source = AgenticWorkspaceSource(
        source_id='source-001',
        path='sources/source-001.txt',
        display_title='Source 001',
        artifact_kind=AgenticArtifactKind.RAW,
        media_type=AgenticMediaType.TEXT,
        sha256=_sha(source_bytes),
        byte_count=len(source_bytes),
        source_url='fixture://source',
        license_id='fixture',
        retrieved_at=NOW,
        temporal_proofs=(proof,),
        selected_proof_id=proof.proof_id,
        effective_available_at_upper=interval.upper_at,
    )
    task = AgenticTaskEnvelope(
        task_id='task-a',
        episode_id='episode-a',
        episode_manifest_sha256='e' * 64,
        decision_at=datetime(2020, 2, 2, tzinfo=UTC),
        task_type='early_clinical_arm_prioritization',
        candidate_ids=('candidate-001', 'candidate-002'),
        portfolio_size=1,
        instructions='Extract the dose and rank the candidates.',
        fact_queries=(
            AgenticFactQuery(
                query_id='dose',
                description='Candidate dose',
                value_type=AgenticValueType.NUMBER,
                unit='microgram',
            ),
        ),
        historically_preregistered=False,
    )
    task, episode_manifest = bind_episode_manifest(task)
    build_policy, discovery_manifest = selection_contract(task, (source,))
    return build_agentic_workspace(
        workspace_id='workspace-a',
        task=task,
        episode_manifest=episode_manifest,
        build_policy=build_policy,
        discovery_manifest=discovery_manifest,
        assurance_profile=AgenticAssuranceProfile.FIXTURE,
        sources=(source,),
        transformations=(),
        source_bytes={'source-001': source_bytes},
        output_root=tmp_path / 'workspace',
    )


def _judge(judge_id: str) -> PinnedLlmJudge:
    return PinnedLlmJudge(
        judge_id=judge_id,
        provider='fixture-provider',
        model_id=f'model-{judge_id}',
        model_revision='1',
        system_fingerprint=f'fingerprint-{judge_id}',
        system_manifest_sha256=_sha(f'system-{judge_id}'.encode()),
        prompt_sha256=_sha(f'prompt-{judge_id}'.encode()),
        config_sha256=_sha(f'config-{judge_id}'.encode()),
    )


def _policy() -> ContaminationAuditPolicy:
    return ContaminationAuditPolicy(
        policy_id='agentic-fixture-policy',
        retrieval=ExactRetrievalConfig(ngram_tokens=4, minimum_ngram_bytes=12),
        calibration=CalibrationPolicy(
            minimum_canary_count=2,
            minimum_negative_control_count=2,
            minimum_canary_recall=1,
            maximum_false_positive_rate=0,
        ),
        judges=(_judge('alpha'), _judge('beta')),
    )


def _clear_output() -> LlmJudgeOutput:
    return LlmJudgeOutput(
        verdict=JudgeVerdict.CLEAR,
        calibration=JudgeCalibrationResult(
            canary_count=2,
            canary_detected_count=2,
            negative_control_count=2,
            false_positive_count=0,
        ),
    )


def _protected_corpus(workspace, comparisons: dict[str, bytes]) -> AgenticProtectedCorpusManifest:
    return AgenticProtectedCorpusManifest(
        corpus_id='protected-workspace-a',
        historical_cutoff=workspace.task.decision_at,
        selection_policy_sha256=_sha(b'fixture protected-corpus selection policy'),
        scope_description='All fixture post-cutoff outcome artifacts selected by the pinned fixture policy.',
        coverage_limitations='Fixture-only namespace; this does not establish global literature completeness.',
        artifacts=tuple(
            AgenticProtectedCorpusArtifact(
                artifact_id=artifact_id,
                sha256=_sha(payload),
                byte_count=len(payload),
                category='outcome_or_label',
                source_uri=f'fixture://{artifact_id}',
                acquired_at=NOW,
            )
            for artifact_id, payload in sorted(comparisons.items())
        ),
    )


def _audit_bundle(workspace):
    comparisons = {'future-outcome': b'Unrelated candidate z later had a favorable result.'}
    protected_corpus = _protected_corpus(workspace, comparisons)
    audit_input = make_agentic_audit_input(
        workspace,
        protected_corpus=protected_corpus,
        comparison_payloads=comparisons,
    )
    policy = _policy()
    runs = tuple(
        make_llm_audit_run(
            run_id=f'run-{judge.judge_id}',
            judge=judge,
            audit_input=audit_input,
            output=_clear_output(),
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=index + 1),
        )
        for index, judge in enumerate(policy.judges)
    )
    audit = build_contamination_audit(
        audit_id='audit-workspace-a',
        audit_input=audit_input,
        policy=policy,
        public_payload=workspace.model_visible_surface,
        comparison_payloads=comparisons,
        judge_runs=runs,
        screened_at=NOW,
    )
    manifest = make_audit_manifest(
        manifest_id='agentic-workspace-a',
        case_universe_sha256=agentic_case_universe_sha256(
            workspace_manifest_sha256=workspace.manifest_sha256,
            model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        ),
        policy=policy,
        audits=(audit,),
    )
    artifacts = {
        AGENTIC_AUDIT_MANIFEST_KEY: canonical_json_bytes(manifest),
        AGENTIC_AUDIT_POLICY_KEY: canonical_json_bytes(policy),
        AGENTIC_AUDIT_IDENTIFIERS_KEY: canonical_json_bytes([]),
        AGENTIC_AUDIT_PROTECTED_CORPUS_KEY: canonical_json_bytes(protected_corpus),
        AGENTIC_AUDIT_COMPARISON_PREFIX + 'future-outcome': comparisons['future-outcome'],
    }
    return manifest, policy, protected_corpus, artifacts


def test_exact_surface_audit_verifies_every_visible_file(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, policy, protected_corpus, artifacts = _audit_bundle(workspace)
    binding = make_agentic_contamination_binding(
        workspace,
        manifest=manifest,
        policy=policy,
        protected_corpus=protected_corpus,
    )

    verified = verify_agentic_contamination_audit(
        binding,
        model_visible_surface=workspace.model_visible_surface,
        audit_artifacts=artifacts,
    )

    assert verified.audited_surface_sha256 == workspace.manifest.model_visible_surface_sha256
    assert verified.audited_file_count == len(workspace.manifest.entries)
    assert verified.judge_count == 2


def test_exact_retrieval_sees_raw_ngram_across_source_newline() -> None:
    leaked_phrase = b'future validation\nsignal strongly'
    public_payload = model_visible_surface_bytes({'sources/document-001.txt': leaked_phrase})
    comparisons = {'future-outcome': b'The future validation\nsignal strongly predicted success.'}
    audit_input = make_audit_input(
        case_id='raw-framing-regression',
        episode_id='episode-a',
        decision_package_sha256='d' * 64,
        episode_manifest_sha256='e' * 64,
        public_artifact_id='agentic-model-visible-surface',
        public_payload=public_payload,
        comparison_payloads=comparisons,
    )

    candidates = retrieve_exact_candidates(
        audit_input,
        public_payload=public_payload,
        comparison_payloads=comparisons,
        policy=_policy(),
    )

    assert leaked_phrase in public_payload
    assert any(candidate.public_span.quote == leaked_phrase.decode() for candidate in candidates)


def test_surface_mutation_and_extra_audit_material_fail_closed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, policy, protected_corpus, artifacts = _audit_bundle(workspace)
    binding = make_agentic_contamination_binding(
        workspace,
        manifest=manifest,
        policy=policy,
        protected_corpus=protected_corpus,
    )

    with pytest.raises(AgenticAdmissionError, match='different model-visible surface'):
        verify_agentic_contamination_audit(
            binding,
            model_visible_surface=workspace.model_visible_surface + b' ',
            audit_artifacts=artifacts,
        )

    with pytest.raises(AgenticAdmissionError, match='inventory mismatch'):
        verify_agentic_contamination_audit(
            binding,
            model_visible_surface=workspace.model_visible_surface,
            audit_artifacts={**artifacts, 'unbound-note.txt': b'looks clean'},
        )


def test_nonpassing_manifest_cannot_be_bound(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    comparisons = {'future-outcome': workspace.model_visible_surface}
    protected_corpus = _protected_corpus(workspace, comparisons)
    audit_input = make_agentic_audit_input(
        workspace,
        protected_corpus=protected_corpus,
        comparison_payloads=comparisons,
    )
    policy = _policy()
    runs = tuple(
        make_llm_audit_run(
            run_id=f'run-{judge.judge_id}',
            judge=judge,
            audit_input=audit_input,
            output=_clear_output(),
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=index + 1),
        )
        for index, judge in enumerate(policy.judges)
    )
    audit = build_contamination_audit(
        audit_id='audit-contaminated',
        audit_input=audit_input,
        policy=policy,
        public_payload=workspace.model_visible_surface,
        comparison_payloads=comparisons,
        judge_runs=runs,
        screened_at=NOW,
    )
    manifest = make_audit_manifest(
        manifest_id='agentic-contaminated',
        case_universe_sha256=agentic_case_universe_sha256(
            workspace_manifest_sha256=workspace.manifest_sha256,
            model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        ),
        policy=policy,
        audits=(audit,),
    )
    with pytest.raises(AgenticAdmissionError, match='only a pass audit'):
        make_agentic_contamination_binding(
            workspace,
            manifest=manifest,
            policy=policy,
            protected_corpus=protected_corpus,
        )
