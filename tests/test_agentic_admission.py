from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.agentic_helpers import bind_episode_manifest, selection_contract
from tests.test_agentic_contamination import _audit_bundle
from vaxreplay.agentic.admission import (
    AgenticAdmissionError,
    AgenticContaminationAdmissionPolicy,
    AgenticContaminationBinding,
    AgenticTemporalPolicy,
    AgenticTrustedTemporalVerifier,
    AgenticTrustedTransformationVerifier,
    AgenticWorkspaceAdmission,
    VerifiedTemporalProof,
    VerifiedTransformation,
    admit_workspace_temporally,
    finalize_workspace_admission,
    require_workspace_admission,
)
from vaxreplay.agentic.contamination import (
    AGENTIC_AUDIT_VERIFIER_ID,
    AGENTIC_AUDIT_VERIFIER_VERSION,
    make_agentic_contamination_binding,
)
from vaxreplay.agentic.schema import (
    AgenticArtifactKind,
    AgenticAssuranceProfile,
    AgenticDerivationKind,
    AgenticFactQuery,
    AgenticMediaType,
    AgenticTaskEnvelope,
    AgenticTransformationReceipt,
    AgenticValueType,
    AgenticWorkspaceSource,
    ArtifactAvailabilityClaim,
    ArtifactTemporalProof,
    AvailabilityClaimKind,
    AvailabilityInterval,
    AvailabilityScope,
    TemporalProofKind,
    agentic_model_sha256,
)
from vaxreplay.agentic.span_map import (
    AgenticIdentityMaskSpan,
    AgenticIdentityMaskSpanMap,
    AgenticNeutralAliasBinding,
    AgenticNeutralAliasNamespace,
    AgenticNeutralAliasPolicy,
    AgenticSourceContainerKind,
    AgenticSpanMappingKind,
    AgenticSpanMapSourceArtifact,
    AgenticSpanMapSourceMember,
    verify_agentic_identity_mask_span_map,
)
from vaxreplay.agentic.workspace import build_agentic_workspace
from vaxreplay.bundle import canonical_json_bytes

_PROOF_BYTES = b'authenticated-proof'


def _interval(year: int = 2020) -> AvailabilityInterval:
    return AvailabilityInterval(
        lower_at=datetime(year, 2, 1, tzinfo=UTC),
        upper_at=datetime(year, 2, 1, 23, 59, 59, tzinfo=UTC),
        precision='day',
        timezone_basis='UTC upper bound',
    )


def _task(*, preregistered: bool = False) -> AgenticTaskEnvelope:
    return AgenticTaskEnvelope(
        task_id='task-1',
        episode_id='episode-1',
        episode_manifest_sha256='e' * 64,
        decision_at=datetime(2020, 2, 2, tzinfo=UTC),
        task_type='early_clinical_arm_prioritization',
        candidate_ids=('candidate-001', 'candidate-002'),
        portfolio_size=1,
        instructions='Extract the dose and prioritize one candidate.',
        fact_queries=(
            AgenticFactQuery(
                query_id='dose',
                description='Candidate dose',
                value_type=AgenticValueType.NUMBER,
                unit='microgram',
            ),
        ),
        historically_preregistered=preregistered,
    )


def _proof(
    content: bytes,
    *,
    proof_id: str = 'proof-raw',
    kind: TemporalProofKind = TemporalProofKind.SOURCE_ATTESTED_SNAPSHOT,
    witnessed: AvailabilityInterval | None = None,
) -> ArtifactTemporalProof:
    return ArtifactTemporalProof(
        proof_id=proof_id,
        kind=kind,
        artifact_sha256=hashlib.sha256(content).hexdigest(),
        artifact_bytes=len(content),
        witnessed=witnessed or _interval(),
        authority_id='test-authority',
        proof_sha256=hashlib.sha256(_PROOF_BYTES).hexdigest(),
        proof_bytes=len(_PROOF_BYTES),
        verification_uri='https://example.test/proof',
    )


def _raw_source(
    content: bytes,
    *,
    proof: ArtifactTemporalProof | None = None,
    claims: tuple[ArtifactAvailabilityClaim, ...] = (),
) -> AgenticWorkspaceSource:
    selected = proof or _proof(content)
    return AgenticWorkspaceSource(
        source_id='source-001',
        path='sources/source-001.txt',
        display_title='Source 001',
        artifact_kind=AgenticArtifactKind.RAW,
        media_type=AgenticMediaType.TEXT,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        source_url='https://example.test/source',
        license_id='fixture',
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        availability_claims=claims,
        temporal_proofs=(selected,),
        selected_proof_id=selected.proof_id,
        effective_available_at_upper=selected.witnessed.upper_at,
    )


def _workspace(
    tmp_path: Path,
    *,
    profile: AgenticAssuranceProfile = AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT,
    proof: ArtifactTemporalProof | None = None,
    preregistered: bool = False,
    claims: tuple[ArtifactAvailabilityClaim, ...] = (),
):
    content = b'candidate|dose\na|120\n'
    source = _raw_source(content, proof=proof, claims=claims)
    task, episode_manifest = bind_episode_manifest(_task(preregistered=preregistered))
    contract_created_at = datetime(2020, 2, 1, tzinfo=UTC) if preregistered else None
    build_policy, discovery_manifest = selection_contract(task, (source,), created_at=contract_created_at)
    return build_agentic_workspace(
        workspace_id='workspace-1',
        task=task,
        episode_manifest=episode_manifest,
        build_policy=build_policy,
        discovery_manifest=discovery_manifest,
        assurance_profile=profile,
        sources=(source,),
        transformations=(),
        source_bytes={'source-001': content},
        output_root=tmp_path / 'workspace',
    )


def _verify_proof(proof, *, artifact_bytes: bytes, proof_bytes: bytes) -> VerifiedTemporalProof:
    assert hashlib.sha256(artifact_bytes).hexdigest() == proof.artifact_sha256
    assert proof_bytes == _PROOF_BYTES
    return VerifiedTemporalProof(
        proof_id=proof.proof_id,
        artifact_sha256=proof.artifact_sha256,
        artifact_bytes=proof.artifact_bytes,
        kind=proof.kind,
        witnessed=proof.witnessed,
        verifier_id='test-proof-verifier',
        verifier_version='1',
        verifier_executable_sha256='f' * 64,
    )


def _unused_transform(*_args, **_kwargs):
    raise AssertionError('no transformation should be verified')


def test_source_attested_workspace_is_explicitly_nonofficial(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    temporal = admit_workspace_temporally(
        workspace,
        policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT),
        proof_artifacts={'proof-raw': _PROOF_BYTES},
        proof_verifier=_verify_proof,
        transformation_execution_receipts={},
        transformation_verifier=_unused_transform,
        admitted_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert temporal.strict_exact_byte_provenance is False
    assert temporal.official_temporal_eligible is False
    assert temporal.retrospective_only is True
    assert temporal.proves_absence_of_contamination is False

    manifest, audit_policy, protected_corpus, artifacts = _audit_bundle(workspace)
    binding = make_agentic_contamination_binding(
        workspace,
        manifest=manifest,
        policy=audit_policy,
        protected_corpus=protected_corpus,
    )
    contamination_policy = AgenticContaminationAdmissionPolicy(
        expected_audit_manifest_sha256=binding.contamination_audit_manifest_sha256,
        expected_audit_policy_sha256=binding.contamination_audit_policy_sha256,
        expected_protected_corpus_manifest_sha256=binding.protected_corpus_manifest_sha256,
        expected_verifier_id=AGENTIC_AUDIT_VERIFIER_ID,
        expected_verifier_version=AGENTIC_AUDIT_VERIFIER_VERSION,
    )

    final = finalize_workspace_admission(
        workspace,
        temporal_admission=temporal,
        contamination_policy=contamination_policy,
        contamination_binding=binding,
        audit_artifacts=artifacts,
        expected_temporal_admission_sha256=agentic_model_sha256(temporal),
    )
    assert final.admitted_use == 'best_effort_research'
    assert final.official_release_eligible is False


def test_source_attestation_and_publication_claim_cannot_upgrade_to_strict_tier_b(tmp_path: Path) -> None:
    claim = ArtifactAvailabilityClaim(
        claim_id='publication',
        kind=AvailabilityClaimKind.PUBLICATION,
        scope=AvailabilityScope.WORK,
        interval=_interval(2019),
        issuer='publisher',
        evidence_sha256='d' * 64,
        note='Publication year predates the cutoff but does not bind these bytes.',
    )
    workspace = _workspace(tmp_path, claims=(claim,))

    with pytest.raises(AgenticAdmissionError, match='profile does not match'):
        admit_workspace_temporally(
            workspace,
            policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE),
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={},
            transformation_verifier=_unused_transform,
        )

    strict_workspace = _workspace(
        tmp_path / 'strict',
        profile=AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE,
        claims=(claim,),
    )
    with pytest.raises(AgenticAdmissionError, match='cannot satisfy'):
        admit_workspace_temporally(
            strict_workspace,
            policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE),
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={},
            transformation_verifier=_unused_transform,
        )


def test_postcutoff_exact_byte_witness_is_quarantined(tmp_path: Path) -> None:
    content = b'candidate|dose\na|120\n'
    later = _proof(content, witnessed=_interval(2021))
    workspace = _workspace(tmp_path, proof=later)
    with pytest.raises(AgenticAdmissionError, match='crosses the historical cutoff'):
        admit_workspace_temporally(
            workspace,
            policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT),
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={},
            transformation_verifier=_unused_transform,
        )


def test_prospective_profile_requires_exact_task_proof(tmp_path: Path) -> None:
    content = b'candidate|dose\na|120\n'
    source_proof = _proof(content, kind=TemporalProofKind.RFC3161_TIMESTAMP)
    workspace = _workspace(
        tmp_path,
        profile=AgenticAssuranceProfile.PROSPECTIVE_EXACT,
        proof=source_proof,
        preregistered=True,
    )
    with pytest.raises(AgenticAdmissionError, match='task, build-policy, and discovery proofs'):
        admit_workspace_temporally(
            workspace,
            policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.PROSPECTIVE_EXACT),
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={},
            transformation_verifier=_unused_transform,
        )

    task_bytes = workspace.task.model_dump_json().encode()
    # Proof binding must use the canonical task bytes, not Pydantic's noncanonical JSON text.
    task_proof = _proof(
        task_bytes,
        proof_id='proof-task',
        kind=TemporalProofKind.RFC3161_TIMESTAMP,
    )
    build_policy_proof = _proof(
        canonical_json_bytes(workspace.build_policy),
        proof_id='proof-build-policy',
        kind=TemporalProofKind.RFC3161_TIMESTAMP,
    )
    discovery_proof = _proof(
        canonical_json_bytes(workspace.discovery_manifest),
        proof_id='proof-discovery',
        kind=TemporalProofKind.RFC3161_TIMESTAMP,
    )
    with pytest.raises(AgenticAdmissionError, match='does not bind TASK.json'):
        admit_workspace_temporally(
            workspace,
            policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.PROSPECTIVE_EXACT),
            proof_artifacts={
                'proof-build-policy': _PROOF_BYTES,
                'proof-discovery': _PROOF_BYTES,
                'proof-raw': _PROOF_BYTES,
                'proof-task': _PROOF_BYTES,
            },
            proof_verifier=_verify_proof,
            transformation_execution_receipts={},
            transformation_verifier=_unused_transform,
            task_definition_proof=task_proof,
            build_policy_proof=build_policy_proof,
            discovery_manifest_proof=discovery_proof,
        )


def test_prospective_profile_seals_task_build_policy_and_discovery_capture(tmp_path: Path) -> None:
    content = b'candidate|dose\na|120\n'
    workspace = _workspace(
        tmp_path,
        profile=AgenticAssuranceProfile.PROSPECTIVE_EXACT,
        proof=_proof(content, kind=TemporalProofKind.PUBLIC_TRANSPARENCY_LOG),
        preregistered=True,
    )
    task_proof = _proof(
        canonical_json_bytes(workspace.task),
        proof_id='proof-task',
        kind=TemporalProofKind.PUBLIC_TRANSPARENCY_LOG,
    )
    build_policy_proof = _proof(
        canonical_json_bytes(workspace.build_policy),
        proof_id='proof-build-policy',
        kind=TemporalProofKind.PUBLIC_TRANSPARENCY_LOG,
    )
    discovery_proof = _proof(
        canonical_json_bytes(workspace.discovery_manifest),
        proof_id='proof-discovery',
        kind=TemporalProofKind.PUBLIC_TRANSPARENCY_LOG,
    )

    admission = admit_workspace_temporally(
        workspace,
        policy=AgenticTemporalPolicy(
            required_profile=AgenticAssuranceProfile.PROSPECTIVE_EXACT,
            verifier_mode='release_pinned',
            trusted_verifiers=(
                AgenticTrustedTemporalVerifier(
                    proof_kind=TemporalProofKind.PUBLIC_TRANSPARENCY_LOG,
                    authority_id='test-authority',
                    verifier_id='test-proof-verifier',
                    verifier_version='1',
                    verifier_executable_sha256='f' * 64,
                ),
            ),
        ),
        proof_artifacts={
            'proof-build-policy': _PROOF_BYTES,
            'proof-discovery': _PROOF_BYTES,
            'proof-raw': _PROOF_BYTES,
            'proof-task': _PROOF_BYTES,
        },
        proof_verifier=_verify_proof,
        transformation_execution_receipts={},
        transformation_verifier=_unused_transform,
        task_definition_proof=task_proof,
        build_policy_proof=build_policy_proof,
        discovery_manifest_proof=discovery_proof,
        admitted_at=datetime(2020, 2, 2, tzinfo=UTC),
    )

    assert admission.official_temporal_eligible
    assert admission.selection_precommitted_before_cutoff
    assert admission.residual_retrospective_selection_contamination is False
    assert admission.build_policy_sha256 == workspace.manifest.build_policy_sha256
    assert admission.discovery_manifest_sha256 == workspace.manifest.discovery_manifest_sha256


def test_contamination_binding_must_cover_every_visible_file(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    temporal = admit_workspace_temporally(
        workspace,
        policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT),
        proof_artifacts={'proof-raw': _PROOF_BYTES},
        proof_verifier=_verify_proof,
        transformation_execution_receipts={},
        transformation_verifier=_unused_transform,
    )
    binding = AgenticContaminationBinding(
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        contamination_audit_manifest_sha256='c' * 64,
        contamination_audit_policy_sha256='d' * 64,
        protected_corpus_manifest_sha256='f' * 64,
        protected_outcome_namespace_sha256=workspace.build_policy.protected_outcome_namespace_sha256,
        audited_file_count=len(workspace.manifest.entries) - 1,
    )
    contamination_policy = AgenticContaminationAdmissionPolicy(
        expected_audit_manifest_sha256=binding.contamination_audit_manifest_sha256,
        expected_audit_policy_sha256=binding.contamination_audit_policy_sha256,
        expected_protected_corpus_manifest_sha256=binding.protected_corpus_manifest_sha256,
        expected_verifier_id=AGENTIC_AUDIT_VERIFIER_ID,
        expected_verifier_version=AGENTIC_AUDIT_VERIFIER_VERSION,
    )
    with pytest.raises(AgenticAdmissionError, match='complete visible inventory'):
        finalize_workspace_admission(
            workspace,
            temporal_admission=temporal,
            contamination_policy=contamination_policy,
            contamination_binding=binding,
            audit_artifacts={},
            expected_temporal_admission_sha256=agentic_model_sha256(temporal),
        )


def test_llm_derived_visible_file_is_rejected_even_when_receipted(tmp_path: Path) -> None:
    raw_bytes = b'raw evidence\n'
    derived_bytes = b'semantic summary\n'
    raw = _raw_source(raw_bytes)
    execution_bytes = b'execution receipt'
    receipt = AgenticTransformationReceipt(
        receipt_id='transform-1',
        kind=AgenticDerivationKind.LLM,
        input_source_ids=('source-001',),
        output_source_id='source-002',
        output_sha256=hashlib.sha256(derived_bytes).hexdigest(),
        output_bytes=len(derived_bytes),
        transform_id='summarize',
        transform_version='1',
        executable_sha256='1' * 64,
        config_sha256='2' * 64,
        execution_receipt_sha256=hashlib.sha256(execution_bytes).hexdigest(),
        execution_receipt_bytes=len(execution_bytes),
        executed_at=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_rewrite=True,
        source_span_mapping_complete=False,
    )
    derived = AgenticWorkspaceSource(
        source_id='source-002',
        path='sources/source-002.txt',
        display_title='Source 002',
        artifact_kind=AgenticArtifactKind.DERIVED,
        media_type=AgenticMediaType.TEXT,
        sha256=hashlib.sha256(derived_bytes).hexdigest(),
        byte_count=len(derived_bytes),
        source_url='derived://summary',
        license_id='inherits-parent',
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        effective_available_at_upper=raw.effective_available_at_upper,
        parent_source_ids=('source-001',),
        transformation_receipt_id='transform-1',
    )
    task, episode_manifest = bind_episode_manifest(_task())
    build_policy, discovery_manifest = selection_contract(task, (raw, derived), (receipt,))
    workspace = build_agentic_workspace(
        workspace_id='workspace-llm',
        task=task,
        episode_manifest=episode_manifest,
        build_policy=build_policy,
        discovery_manifest=discovery_manifest,
        assurance_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT,
        sources=(raw, derived),
        transformations=(receipt,),
        source_bytes={'source-001': raw_bytes, 'source-002': derived_bytes},
        output_root=tmp_path / 'workspace',
    )

    def should_not_verify(*_args, **_kwargs) -> VerifiedTransformation:
        raise AssertionError('LLM transformation must fail before trusted re-execution')

    with pytest.raises(AgenticAdmissionError, match='deterministic, extractive'):
        admit_workspace_temporally(
            workspace,
            policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT),
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={'transform-1': execution_bytes},
            transformation_verifier=should_not_verify,
        )


def test_deterministic_identity_mask_requires_and_verifies_exact_private_span_map(tmp_path: Path) -> None:
    raw_bytes = b'row-1|Product-X\n'
    derived_bytes = b'candidate-001'
    raw = _raw_source(raw_bytes)
    execution_bytes = b'deterministic execution receipt'
    alias_policy = AgenticNeutralAliasPolicy(
        policy_id='mask-policy-001',
        bindings=(
            AgenticNeutralAliasBinding(
                source_identity_sha256=hashlib.sha256(raw_bytes[6:15]).hexdigest(),
                namespace=AgenticNeutralAliasNamespace.CANDIDATE,
                ordinal=1,
                alias_token='candidate-001',
            ),
        ),
    )
    span_map = AgenticIdentityMaskSpanMap(
        span_map_id='mask-map-001',
        transformation_receipt_id='transform-1',
        output_source_id='source-002',
        output_sha256=hashlib.sha256(derived_bytes).hexdigest(),
        output_bytes=len(derived_bytes),
        complete_output_coverage=True,
        neutral_alias_policy=alias_policy,
        source_artifacts=(
            AgenticSpanMapSourceArtifact(
                source_artifact_id='source-001',
                container_kind=AgenticSourceContainerKind.FILE,
                sha256=hashlib.sha256(raw_bytes).hexdigest(),
                byte_count=len(raw_bytes),
            ),
        ),
        source_members=(
            AgenticSpanMapSourceMember(
                source_member_id='member-001',
                source_artifact_id='source-001',
                member_path='records.txt',
                sha256=hashlib.sha256(raw_bytes).hexdigest(),
                byte_count=len(raw_bytes),
            ),
        ),
        mappings=(
            AgenticIdentityMaskSpan(
                mapping_id='map-001',
                kind=AgenticSpanMappingKind.MASKED_REPLACEMENT,
                output_start_byte=0,
                output_end_byte=len(derived_bytes),
                output_span_sha256=hashlib.sha256(derived_bytes).hexdigest(),
                source_member_id='member-001',
                source_row_id='row-1',
                source_row_start_byte=0,
                source_row_end_byte=len(raw_bytes),
                source_row_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                source_row_id_start_byte=0,
                source_row_id_end_byte=5,
                source_field_name='product',
                source_field_start_byte=6,
                source_field_end_byte=15,
                source_field_sha256=hashlib.sha256(raw_bytes[6:15]).hexdigest(),
                source_start_byte=6,
                source_end_byte=15,
                source_span_sha256=hashlib.sha256(raw_bytes[6:15]).hexdigest(),
                neutral_alias_token='candidate-001',
            ),
        ),
    )
    span_map_bytes = canonical_json_bytes(span_map)
    receipt = AgenticTransformationReceipt(
        receipt_id='transform-1',
        kind=AgenticDerivationKind.DETERMINISTIC,
        input_source_ids=('source-001',),
        output_source_id='source-002',
        output_sha256=hashlib.sha256(derived_bytes).hexdigest(),
        output_bytes=len(derived_bytes),
        transform_id='identity-mask',
        transform_version='1',
        executable_sha256='1' * 64,
        config_sha256=hashlib.sha256(canonical_json_bytes(alias_policy)).hexdigest(),
        execution_receipt_sha256=hashlib.sha256(execution_bytes).hexdigest(),
        execution_receipt_bytes=len(execution_bytes),
        executed_at=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_rewrite=False,
        source_span_mapping_complete=True,
        span_map_sha256=hashlib.sha256(span_map_bytes).hexdigest(),
        span_map_bytes=len(span_map_bytes),
    )
    derived = AgenticWorkspaceSource(
        source_id='source-002',
        path='sources/source-002.txt',
        display_title='Source 002',
        artifact_kind=AgenticArtifactKind.DERIVED,
        media_type=AgenticMediaType.TEXT,
        sha256=hashlib.sha256(derived_bytes).hexdigest(),
        byte_count=len(derived_bytes),
        source_url='derived://identity-mask',
        license_id='inherits-parent',
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        effective_available_at_upper=raw.effective_available_at_upper,
        parent_source_ids=('source-001',),
        transformation_receipt_id='transform-1',
    )
    task, episode_manifest = bind_episode_manifest(_task())
    build_policy, discovery_manifest = selection_contract(task, (raw, derived), (receipt,))
    workspace = build_agentic_workspace(
        workspace_id='workspace-identity-mask',
        task=task,
        episode_manifest=episode_manifest,
        build_policy=build_policy,
        discovery_manifest=discovery_manifest,
        assurance_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT,
        sources=(raw, derived),
        transformations=(receipt,),
        source_bytes={'source-001': raw_bytes, 'source-002': derived_bytes},
        output_root=tmp_path / 'workspace',
    )

    with pytest.raises(AgenticAdmissionError, match='requires its private span-map artifact'):
        admit_workspace_temporally(
            workspace,
            policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT),
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={'transform-1': execution_bytes},
            transformation_verifier=_unused_transform,
        )

    transform_verifier_sha256 = 'e' * 64

    def verify_transform(
        receipt: AgenticTransformationReceipt,
        *,
        input_bytes: tuple[bytes, ...],
        output_bytes: bytes,
        execution_receipt_bytes: bytes,
        span_map_artifact_bytes: bytes,
    ) -> VerifiedTransformation:
        assert execution_receipt_bytes == execution_bytes
        verified_map = verify_agentic_identity_mask_span_map(
            span_map_artifact_bytes,
            receipt=receipt,
            output_bytes=output_bytes,
            source_artifacts=dict(zip(receipt.input_source_ids, input_bytes, strict=True)),
        )
        return VerifiedTransformation(
            receipt_id=receipt.receipt_id,
            transform_id=receipt.transform_id,
            transform_version=receipt.transform_version,
            transform_executable_sha256=receipt.executable_sha256,
            output_sha256=verified_map.output_sha256,
            output_bytes=verified_map.output_bytes,
            span_map_sha256=verified_map.span_map_sha256,
            span_map_bytes=verified_map.span_map_bytes,
            execution_receipt_sha256=hashlib.sha256(execution_receipt_bytes).hexdigest(),
            execution_receipt_bytes=len(execution_receipt_bytes),
            reexecution_performed=True,
            reexecution_output_sha256=hashlib.sha256(output_bytes).hexdigest(),
            reexecution_output_bytes=len(output_bytes),
            verifier_id='test-identity-mask-verifier',
            verifier_version='1',
            verifier_executable_sha256=transform_verifier_sha256,
        )

    admission = admit_workspace_temporally(
        workspace,
        policy=AgenticTemporalPolicy(required_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT),
        proof_artifacts={'proof-raw': _PROOF_BYTES},
        proof_verifier=_verify_proof,
        transformation_execution_receipts={'transform-1': execution_bytes},
        transformation_span_map_artifacts={'transform-1': span_map_bytes},
        transformation_verifier=verify_transform,
    )
    assert admission.verified_transformations == (
        VerifiedTransformation(
            receipt_id='transform-1',
            transform_id='identity-mask',
            transform_version='1',
            transform_executable_sha256='1' * 64,
            output_sha256=hashlib.sha256(derived_bytes).hexdigest(),
            output_bytes=len(derived_bytes),
            span_map_sha256=hashlib.sha256(span_map_bytes).hexdigest(),
            span_map_bytes=len(span_map_bytes),
            execution_receipt_sha256=hashlib.sha256(execution_bytes).hexdigest(),
            execution_receipt_bytes=len(execution_bytes),
            reexecution_performed=True,
            reexecution_output_sha256=hashlib.sha256(derived_bytes).hexdigest(),
            reexecution_output_bytes=len(derived_bytes),
            verifier_id='test-identity-mask-verifier',
            verifier_version='1',
            verifier_executable_sha256=transform_verifier_sha256,
        ),
    )

    release_policy = AgenticTemporalPolicy(
        required_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT,
        verifier_mode='release_pinned',
        trusted_verifiers=(
            AgenticTrustedTemporalVerifier(
                proof_kind=TemporalProofKind.SOURCE_ATTESTED_SNAPSHOT,
                authority_id='test-authority',
                verifier_id='test-proof-verifier',
                verifier_version='1',
                verifier_executable_sha256='f' * 64,
            ),
        ),
        trusted_transformation_verifiers=(
            AgenticTrustedTransformationVerifier(
                transform_id='identity-mask',
                transform_version='1',
                transform_executable_sha256='1' * 64,
                verifier_id='test-identity-mask-verifier',
                verifier_version='1',
                verifier_executable_sha256=transform_verifier_sha256,
            ),
        ),
    )
    release_admission = admit_workspace_temporally(
        workspace,
        policy=release_policy,
        proof_artifacts={'proof-raw': _PROOF_BYTES},
        proof_verifier=_verify_proof,
        transformation_execution_receipts={'transform-1': execution_bytes},
        transformation_span_map_artifacts={'transform-1': span_map_bytes},
        transformation_verifier=verify_transform,
    )
    assert release_admission.verifier_mode == 'release_pinned'

    without_transform_registry = release_policy.model_copy(update={'trusted_transformation_verifiers': ()})
    with pytest.raises(AgenticAdmissionError, match='absent from the release-pinned transformation-verifier registry'):
        admit_workspace_temporally(
            workspace,
            policy=without_transform_registry,
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={'transform-1': execution_bytes},
            transformation_span_map_artifacts={'transform-1': span_map_bytes},
            transformation_verifier=verify_transform,
        )

    def forged_verifier_identity(
        receipt: AgenticTransformationReceipt,
        *,
        input_bytes: tuple[bytes, ...],
        output_bytes: bytes,
        execution_receipt_bytes: bytes,
        span_map_artifact_bytes: bytes,
    ) -> VerifiedTransformation:
        result = verify_transform(
            receipt,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            execution_receipt_bytes=execution_receipt_bytes,
            span_map_artifact_bytes=span_map_artifact_bytes,
        )
        return result.model_copy(update={'verifier_executable_sha256': '0' * 64})

    with pytest.raises(AgenticAdmissionError, match='release-pinned transform and verifier implementations'):
        admit_workspace_temporally(
            workspace,
            policy=release_policy,
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={'transform-1': execution_bytes},
            transformation_span_map_artifacts={'transform-1': span_map_bytes},
            transformation_verifier=forged_verifier_identity,
        )

    def forged_transform_identity(
        receipt: AgenticTransformationReceipt,
        *,
        input_bytes: tuple[bytes, ...],
        output_bytes: bytes,
        execution_receipt_bytes: bytes,
        span_map_artifact_bytes: bytes,
    ) -> VerifiedTransformation:
        result = verify_transform(
            receipt,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            execution_receipt_bytes=execution_receipt_bytes,
            span_map_artifact_bytes=span_map_artifact_bytes,
        )
        return result.model_copy(update={'transform_version': 'future-version'})

    with pytest.raises(AgenticAdmissionError, match='exact transform, receipt artifacts, and re-executed output'):
        admit_workspace_temporally(
            workspace,
            policy=release_policy,
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={'transform-1': execution_bytes},
            transformation_span_map_artifacts={'transform-1': span_map_bytes},
            transformation_verifier=forged_transform_identity,
        )

    def forged_reexecution_result(
        receipt: AgenticTransformationReceipt,
        *,
        input_bytes: tuple[bytes, ...],
        output_bytes: bytes,
        execution_receipt_bytes: bytes,
        span_map_artifact_bytes: bytes,
    ) -> VerifiedTransformation:
        result = verify_transform(
            receipt,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            execution_receipt_bytes=execution_receipt_bytes,
            span_map_artifact_bytes=span_map_artifact_bytes,
        )
        return result.model_copy(update={'reexecution_output_sha256': '0' * 64})

    with pytest.raises(AgenticAdmissionError, match='exact transform, receipt artifacts, and re-executed output'):
        admit_workspace_temporally(
            workspace,
            policy=release_policy,
            proof_artifacts={'proof-raw': _PROOF_BYTES},
            proof_verifier=_verify_proof,
            transformation_execution_receipts={'transform-1': execution_bytes},
            transformation_span_map_artifacts={'transform-1': span_map_bytes},
            transformation_verifier=forged_reexecution_result,
        )


def test_workspace_admission_requires_profile_and_trusted_release_commitment(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    forged = AgenticWorkspaceAdmission(
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        build_policy_sha256=workspace.manifest.build_policy_sha256,
        discovery_manifest_sha256=workspace.manifest.discovery_manifest_sha256,
        alias_seed_commitment_sha256=workspace.manifest.alias_seed_commitment_sha256,
        temporal_admission_sha256='1' * 64,
        contamination_binding_sha256='2' * 64,
        contamination_admission_policy_sha256='4' * 64,
        contamination_audit_manifest_sha256='3' * 64,
        assurance_profile=AgenticAssuranceProfile.PROSPECTIVE_EXACT,
        admitted_use='prospective_research',
        official_release_eligible=False,
        selection_precommitted_before_cutoff=True,
        residual_retrospective_selection_contamination=False,
    )
    with pytest.raises(ValueError):
        AgenticWorkspaceAdmission.model_validate(
            {
                **forged.model_dump(),
                'admitted_use': 'official_benchmark',
                'official_release_eligible': True,
            }
        )
    with pytest.raises(AgenticAdmissionError, match='current workspace'):
        require_workspace_admission(
            workspace,
            forged,
            expected_admission_sha256=agentic_model_sha256(forged),
        )

    same_profile_forgery = AgenticWorkspaceAdmission(
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        build_policy_sha256=workspace.manifest.build_policy_sha256,
        discovery_manifest_sha256=workspace.manifest.discovery_manifest_sha256,
        alias_seed_commitment_sha256=workspace.manifest.alias_seed_commitment_sha256,
        temporal_admission_sha256='1' * 64,
        contamination_binding_sha256='2' * 64,
        contamination_admission_policy_sha256='4' * 64,
        contamination_audit_manifest_sha256='3' * 64,
        assurance_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT,
        admitted_use='best_effort_research',
        official_release_eligible=False,
        selection_precommitted_before_cutoff=False,
        residual_retrospective_selection_contamination=True,
    )
    with pytest.raises(AgenticAdmissionError, match='trusted release commitment'):
        require_workspace_admission(
            workspace,
            same_profile_forgery,
            expected_admission_sha256='0' * 64,
        )
