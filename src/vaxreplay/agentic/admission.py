"""Fail-closed temporal and contamination admission for Agentic Replay workspaces."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.schema import (
    AGENTIC_CONTAMINATION_ADMISSION_POLICY_SCHEMA_VERSION,
    AGENTIC_CONTAMINATION_BINDING_SCHEMA_VERSION,
    AGENTIC_TEMPORAL_ADMISSION_SCHEMA_VERSION,
    AGENTIC_TEMPORAL_POLICY_SCHEMA_VERSION,
    AGENTIC_WORKSPACE_ADMISSION_SCHEMA_VERSION,
    AgenticArtifactKind,
    AgenticAssuranceProfile,
    AgenticDerivationKind,
    AgenticMediaType,
    AgenticTransformationReceipt,
    ArtifactTemporalProof,
    AvailabilityInterval,
    TemporalProofKind,
    agentic_model_sha256,
)
from vaxreplay.agentic.workspace import LoadedAgenticWorkspace, load_agentic_workspace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel


class AgenticAdmissionError(ValueError):
    """Raised when any visible artifact lacks a closed cutoff-admissible provenance path."""


class AgenticTrustedTemporalVerifier(StrictModel):
    proof_kind: TemporalProofKind
    authority_id: str = Field(min_length=1)
    verifier_id: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    verifier_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class AgenticTrustedTransformationVerifier(StrictModel):
    """Release-pinned transform and verifier implementation identity."""

    transform_id: str = Field(min_length=1)
    transform_version: str = Field(min_length=1)
    transform_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    verifier_id: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    verifier_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class AgenticTemporalPolicy(StrictModel):
    schema_version: Literal['vaxreplay.agentic-temporal-policy.v0.1'] = AGENTIC_TEMPORAL_POLICY_SCHEMA_VERSION
    required_profile: AgenticAssuranceProfile
    verifier_mode: Literal['conformance', 'release_pinned'] = 'conformance'
    trusted_verifiers: tuple[AgenticTrustedTemporalVerifier, ...] = ()
    trusted_transformation_verifiers: tuple[AgenticTrustedTransformationVerifier, ...] = ()
    allowed_media_types: tuple[AgenticMediaType, ...] = tuple(AgenticMediaType)
    exact_byte_scope_required: Literal[True] = True
    publication_date_is_proof: Literal[False] = False
    deterministic_derivations_only: Literal[True] = True
    semantic_rewrite_allowed: Literal[False] = False
    trusted_reexecution_required: Literal[True] = True
    source_span_mapping_required: Literal[True] = True
    label_blind_required: Literal[True] = True
    outcome_namespace_allowed: Literal[False] = False
    transformation_network_allowed: Literal[False] = False

    @field_validator('allowed_media_types')
    @classmethod
    def validate_media(cls, value: tuple[AgenticMediaType, ...]) -> tuple[AgenticMediaType, ...]:
        expected = tuple(media for media in AgenticMediaType if media in value)
        if value != expected or len(value) != len(set(value)):
            raise ValueError('allowed media types must be unique and use canonical order')
        if not value:
            raise ValueError('temporal policy must allow at least one media type')
        return value

    @model_validator(mode='after')
    def validate_verifier_registry(self) -> Self:
        identities = tuple((item.proof_kind.value, item.authority_id) for item in self.trusted_verifiers)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError('trusted temporal verifiers must use unique kind/authority identities in sorted order')
        if (self.verifier_mode == 'release_pinned') != bool(self.trusted_verifiers):
            raise ValueError('release_pinned mode requires a non-empty registry; conformance mode forbids one')
        transform_identities = tuple(
            (item.transform_id, item.transform_version) for item in self.trusted_transformation_verifiers
        )
        if transform_identities != tuple(sorted(transform_identities)) or len(transform_identities) != len(
            set(transform_identities)
        ):
            raise ValueError(
                'trusted transformation verifiers must use unique transform ID/version identities in sorted order'
            )
        if self.verifier_mode == 'conformance' and self.trusted_transformation_verifiers:
            raise ValueError('conformance mode forbids a release-pinned transformation-verifier registry')
        return self


class VerifiedTemporalProof(StrictModel):
    proof_id: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    artifact_bytes: int = Field(gt=0)
    kind: TemporalProofKind
    witnessed: AvailabilityInterval
    verifier_id: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    verifier_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class VerifiedTransformation(StrictModel):
    receipt_id: str = Field(min_length=1)
    transform_id: str = Field(min_length=1)
    transform_version: str = Field(min_length=1)
    transform_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    output_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    output_bytes: int = Field(gt=0)
    span_map_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    span_map_bytes: int = Field(gt=0)
    execution_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    execution_receipt_bytes: int = Field(gt=0)
    reexecution_performed: Literal[True]
    reexecution_output_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reexecution_output_bytes: int = Field(gt=0)
    verifier_id: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    verifier_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_reexecution_result(self) -> Self:
        if (self.reexecution_output_sha256, self.reexecution_output_bytes) != (
            self.output_sha256,
            self.output_bytes,
        ):
            raise ValueError('re-executed transformation output must match the verified public output')
        return self


class TemporalProofVerifier(Protocol):
    def __call__(
        self,
        proof: ArtifactTemporalProof,
        *,
        artifact_bytes: bytes,
        proof_bytes: bytes,
    ) -> VerifiedTemporalProof: ...


class TransformationVerifier(Protocol):
    def __call__(
        self,
        receipt: AgenticTransformationReceipt,
        *,
        input_bytes: tuple[bytes, ...],
        output_bytes: bytes,
        execution_receipt_bytes: bytes,
        span_map_artifact_bytes: bytes,
    ) -> VerifiedTransformation: ...


class AgenticTemporalAdmission(StrictModel):
    schema_version: Literal['vaxreplay.agentic-temporal-admission.v0.1'] = AGENTIC_TEMPORAL_ADMISSION_SCHEMA_VERSION
    workspace_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    workspace_tree_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_visible_surface_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_inventory_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    transformation_inventory_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    build_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    discovery_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    alias_seed_commitment_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    temporal_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    verifier_mode: Literal['conformance', 'release_pinned']
    assurance_profile: AgenticAssuranceProfile
    decision_at: datetime
    admitted_at: datetime
    verified_source_proofs: tuple[VerifiedTemporalProof, ...] = Field(min_length=1)
    task_definition_proof: VerifiedTemporalProof | None = None
    build_policy_proof: VerifiedTemporalProof | None = None
    discovery_manifest_proof: VerifiedTemporalProof | None = None
    verified_transformations: tuple[VerifiedTransformation, ...] = ()
    every_visible_source_has_closed_provenance: Literal[True] = True
    strict_exact_byte_provenance: bool
    official_temporal_eligible: bool
    retrospective_only: bool
    selection_precommitted_before_cutoff: bool
    residual_retrospective_selection_contamination: bool
    proves_global_discovery_complete: Literal[False] = False
    residual_model_weight_contamination: Literal[True] = True
    residual_harness_embedded_knowledge: Literal[True] = True
    proves_absence_of_contamination: Literal[False] = False

    @field_validator('decision_at', 'admitted_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_profile_claim(self) -> Self:
        release_pinned = self.verifier_mode == 'release_pinned'
        expected_strict = release_pinned and self.assurance_profile in {
            AgenticAssuranceProfile.PROSPECTIVE_EXACT,
            AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE,
        }
        expected_official = release_pinned and self.assurance_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT
        if self.strict_exact_byte_provenance != expected_strict:
            raise ValueError('strict_exact_byte_provenance must reflect the assurance profile')
        if self.official_temporal_eligible != expected_official:
            raise ValueError('official_temporal_eligible is true only for prospective_exact')
        expected_retrospective = self.assurance_profile != AgenticAssuranceProfile.PROSPECTIVE_EXACT
        if self.retrospective_only != expected_retrospective:
            raise ValueError('retrospective_only must reflect whether the profile is prospective')
        if self.assurance_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT and self.task_definition_proof is None:
            raise ValueError('prospective_exact admission requires a task-definition proof')
        if self.assurance_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT and (
            self.build_policy_proof is None or self.discovery_manifest_proof is None
        ):
            raise ValueError('prospective_exact admission requires build-policy and discovery proofs')
        if self.selection_precommitted_before_cutoff != expected_official:
            raise ValueError('only prospective_exact can claim pre-cutoff selection commitment')
        if self.residual_retrospective_selection_contamination == expected_official:
            raise ValueError('retrospective selection contamination must remain explicit outside prospective_exact')
        proof_ids = tuple(proof.proof_id for proof in self.verified_source_proofs)
        if proof_ids != tuple(sorted(proof_ids)) or len(proof_ids) != len(set(proof_ids)):
            raise ValueError('verified source proofs must use unique proof IDs in sorted order')
        receipt_ids = tuple(receipt.receipt_id for receipt in self.verified_transformations)
        if receipt_ids != tuple(sorted(receipt_ids)) or len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError('verified transformations must use unique receipt IDs in sorted order')
        return self


class VerifiedContaminationAudit(StrictModel):
    contamination_audit_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contamination_audit_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    protected_corpus_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    audited_surface_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    audited_file_count: int = Field(gt=0)
    disposition: Literal['pass'] = 'pass'
    judge_count: int = Field(ge=2)
    calibration_passed: Literal[True] = True
    inventory_complete: Literal[True] = True
    verifier_id: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)


class ContaminationAuditVerifier(Protocol):
    def __call__(
        self,
        binding: AgenticContaminationBinding,
        *,
        model_visible_surface: bytes,
        audit_artifacts: Mapping[str, bytes],
    ) -> VerifiedContaminationAudit: ...


class AgenticContaminationBinding(StrictModel):
    schema_version: Literal['vaxreplay.agentic-contamination-binding.v0.1'] = (
        AGENTIC_CONTAMINATION_BINDING_SCHEMA_VERSION
    )
    workspace_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    workspace_tree_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_visible_surface_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contamination_audit_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contamination_audit_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    protected_corpus_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    protected_outcome_namespace_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    audited_file_count: int = Field(gt=0)
    disposition: Literal['pass'] = 'pass'
    calibration_passed: Literal[True] = True
    inventory_complete: Literal[True] = True
    residual_contamination_possible: Literal[True] = True
    proves_absence_of_contamination: Literal[False] = False


class AgenticContaminationAdmissionPolicy(StrictModel):
    """Organizer-approved audit and protected-corpus commitments for one release."""

    schema_version: Literal['vaxreplay.agentic-contamination-admission-policy.v0.1'] = (
        AGENTIC_CONTAMINATION_ADMISSION_POLICY_SCHEMA_VERSION
    )
    expected_audit_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    expected_audit_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    expected_protected_corpus_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    expected_verifier_id: str = Field(min_length=1)
    expected_verifier_version: str = Field(min_length=1)
    exact_surface_required: Literal[True] = True
    exact_comparison_inventory_required: Literal[True] = True
    independently_pinned_judges_required: Literal[True] = True
    pass_means_no_signal_under_pinned_screen_only: Literal[True] = True
    proves_protected_corpus_globally_complete: Literal[False] = False
    proves_absence_of_contamination: Literal[False] = False


class AgenticWorkspaceAdmission(StrictModel):
    schema_version: Literal['vaxreplay.agentic-workspace-admission.v0.1'] = AGENTIC_WORKSPACE_ADMISSION_SCHEMA_VERSION
    workspace_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    workspace_tree_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_visible_surface_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    build_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    discovery_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    alias_seed_commitment_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    temporal_admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contamination_binding_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contamination_admission_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contamination_audit_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    assurance_profile: AgenticAssuranceProfile
    admitted_use: Literal[
        'prospective_research',
        'retrospective_research',
        'best_effort_research',
        'fixture',
    ]
    official_release_eligible: Literal[False] = False
    authenticated_release_seal_present: Literal[False] = False
    release_gate_status: Literal['research_only_unsealed'] = 'research_only_unsealed'
    selection_precommitted_before_cutoff: bool
    residual_retrospective_selection_contamination: bool
    one_attempt_required: Literal[True] = True
    delayed_aggregate_feedback_required: Literal[True] = True
    residual_model_weight_contamination: Literal[True] = True
    residual_harness_embedded_knowledge: Literal[True] = True
    proves_absence_of_contamination: Literal[False] = False

    @model_validator(mode='after')
    def validate_use(self) -> Self:
        expected = {
            AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE: ('retrospective_research', False),
            AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT: ('best_effort_research', False),
            AgenticAssuranceProfile.FIXTURE: ('fixture', False),
        }.get(self.assurance_profile)
        if self.assurance_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT:
            expected = ('prospective_research', False)
        if (self.admitted_use, self.official_release_eligible) != expected:
            raise ValueError('workspace admitted use must reflect its assurance profile')
        if self.assurance_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT:
            if self.selection_precommitted_before_cutoff == self.residual_retrospective_selection_contamination:
                raise ValueError('prospective selection commitment and residual-risk flags must be opposites')
        elif self.selection_precommitted_before_cutoff or not self.residual_retrospective_selection_contamination:
            raise ValueError('retrospective profiles must retain selection-contamination risk')
        return self


def admit_workspace_temporally(
    workspace: LoadedAgenticWorkspace,
    *,
    policy: AgenticTemporalPolicy,
    proof_artifacts: Mapping[str, bytes],
    proof_verifier: TemporalProofVerifier,
    transformation_execution_receipts: Mapping[str, bytes],
    transformation_verifier: TransformationVerifier,
    transformation_span_map_artifacts: Mapping[str, bytes] | None = None,
    task_definition_proof: ArtifactTemporalProof | None = None,
    build_policy_proof: ArtifactTemporalProof | None = None,
    discovery_manifest_proof: ArtifactTemporalProof | None = None,
    admitted_at: datetime | None = None,
) -> AgenticTemporalAdmission:
    """Verify every source leaf and deterministic transformation without consulting outcomes."""

    workspace = load_agentic_workspace(workspace.root)
    if workspace.manifest.assurance_profile != policy.required_profile:
        raise AgenticAdmissionError('workspace assurance profile does not match the temporal policy')
    if any(source.media_type not in policy.allowed_media_types for source in workspace.sources):
        raise AgenticAdmissionError('workspace contains a media type forbidden by temporal policy')

    selected_proofs = tuple(
        next(proof for proof in source.temporal_proofs if proof.proof_id == source.selected_proof_id)
        for source in workspace.sources
        if source.artifact_kind == AgenticArtifactKind.RAW
    )
    required_proof_ids = {proof.proof_id for proof in selected_proofs}
    if task_definition_proof is not None:
        required_proof_ids.add(task_definition_proof.proof_id)
    if build_policy_proof is not None:
        required_proof_ids.add(build_policy_proof.proof_id)
    if discovery_manifest_proof is not None:
        required_proof_ids.add(discovery_manifest_proof.proof_id)
    if set(proof_artifacts) != required_proof_ids:
        raise AgenticAdmissionError('proof artifacts must cover every and only selected source/task proof')

    verified_proofs: list[VerifiedTemporalProof] = []
    source_by_id = workspace.source_by_id
    for source, proof in zip(
        (source for source in workspace.sources if source.artifact_kind == AgenticArtifactKind.RAW),
        selected_proofs,
        strict=True,
    ):
        if source.effective_available_at_upper > workspace.task.decision_at:
            raise AgenticAdmissionError(f'workspace source crosses the historical cutoff: {source.source_id}')
        _require_proof_kind(policy.required_profile, proof.kind)
        proof_bytes = proof_artifacts[proof.proof_id]
        _verify_proof_file_binding(proof, proof_bytes)
        verified = proof_verifier(
            proof,
            artifact_bytes=workspace.read_source(source.source_id),
            proof_bytes=proof_bytes,
        )
        _validate_verified_proof(verified, proof, decision_at=workspace.task.decision_at)
        _validate_trusted_verifier(policy, proof, verified)
        verified_proofs.append(verified)

    verified_task: VerifiedTemporalProof | None = None
    verified_build_policy: VerifiedTemporalProof | None = None
    verified_discovery_manifest: VerifiedTemporalProof | None = None
    if policy.required_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT:
        if (
            not workspace.task.historically_preregistered
            or task_definition_proof is None
            or build_policy_proof is None
            or discovery_manifest_proof is None
        ):
            raise AgenticAdmissionError(
                'prospective_exact requires historically preregistered task, build-policy, and discovery proofs'
            )
        if (
            workspace.build_policy.created_at > workspace.task.decision_at
            or workspace.discovery_manifest.created_at > workspace.task.decision_at
        ):
            raise AgenticAdmissionError('prospective selection commitments must be created by the historical cutoff')
        if (task_definition_proof.artifact_sha256, task_definition_proof.artifact_bytes) != (
            workspace.manifest.task_sha256,
            len(canonical_json_bytes(workspace.task)),
        ):
            raise AgenticAdmissionError('task-definition proof does not bind TASK.json')
        _require_proof_kind(AgenticAssuranceProfile.PROSPECTIVE_EXACT, task_definition_proof.kind)
        proof_bytes = proof_artifacts[task_definition_proof.proof_id]
        _verify_proof_file_binding(task_definition_proof, proof_bytes)
        verified_task = proof_verifier(
            task_definition_proof,
            artifact_bytes=canonical_json_bytes(workspace.task),
            proof_bytes=proof_bytes,
        )
        _validate_verified_proof(verified_task, task_definition_proof, decision_at=workspace.task.decision_at)
        _validate_trusted_verifier(policy, task_definition_proof, verified_task)
        verified_build_policy = _verify_workspace_metadata_proof(
            proof=build_policy_proof,
            artifact=canonical_json_bytes(workspace.build_policy),
            expected_sha256=workspace.manifest.build_policy_sha256,
            proof_bytes=proof_artifacts[build_policy_proof.proof_id],
            proof_verifier=proof_verifier,
            temporal_policy=policy,
            decision_at=workspace.task.decision_at,
        )
        verified_discovery_manifest = _verify_workspace_metadata_proof(
            proof=discovery_manifest_proof,
            artifact=canonical_json_bytes(workspace.discovery_manifest),
            expected_sha256=workspace.manifest.discovery_manifest_sha256,
            proof_bytes=proof_artifacts[discovery_manifest_proof.proof_id],
            proof_verifier=proof_verifier,
            temporal_policy=policy,
            decision_at=workspace.task.decision_at,
        )
    elif any(value is not None for value in (task_definition_proof, build_policy_proof, discovery_manifest_proof)):
        raise AgenticAdmissionError('non-prospective profiles cannot imply task or selection preregistration seals')

    required_receipt_ids = {receipt.receipt_id for receipt in workspace.transformations}
    if set(transformation_execution_receipts) != required_receipt_ids:
        raise AgenticAdmissionError('execution receipts must cover every and only declared transformation')
    span_map_artifacts = dict(transformation_span_map_artifacts or {})
    if unknown_span_maps := set(span_map_artifacts) - required_receipt_ids:
        raise AgenticAdmissionError(
            f'span-map artifacts reference undeclared transformations: {sorted(unknown_span_maps)}'
        )
    verified_transformations: list[VerifiedTransformation] = []
    availability_by_source = {
        source.source_id: source.effective_available_at_upper
        for source in workspace.sources
        if source.artifact_kind == AgenticArtifactKind.RAW
    }
    pending = {source.source_id for source in workspace.sources if source.artifact_kind == AgenticArtifactKind.DERIVED}
    receipt_by_id = {receipt.receipt_id: receipt for receipt in workspace.transformations}
    while pending:
        progressed = False
        for source_id in sorted(tuple(pending)):
            source = source_by_id[source_id]
            if any(parent not in availability_by_source for parent in source.parent_source_ids):
                continue
            receipt_id = source.transformation_receipt_id
            if receipt_id is None or receipt_id not in receipt_by_id:
                raise AgenticAdmissionError('derived source has no declared transformation receipt')
            receipt = receipt_by_id[receipt_id]
            _validate_transform_policy(receipt)
            span_map_bytes = span_map_artifacts.get(receipt.receipt_id)
            if span_map_bytes is None:
                raise AgenticAdmissionError('every admitted transformation requires its private span-map artifact')
            if not isinstance(span_map_bytes, bytes):
                raise AgenticAdmissionError('transformation span-map artifact must be bytes')
            if receipt.span_map_sha256 is None or receipt.span_map_bytes is None:
                raise AgenticAdmissionError('transformation receipt lacks its exact private span-map binding')
            if (hashlib.sha256(span_map_bytes).hexdigest(), len(span_map_bytes)) != (
                receipt.span_map_sha256,
                receipt.span_map_bytes,
            ):
                raise AgenticAdmissionError('transformation span-map artifact byte binding mismatch')
            expected_available = max(availability_by_source[parent] for parent in source.parent_source_ids)
            if (
                source.effective_available_at_upper != expected_available
                or expected_available > workspace.task.decision_at
            ):
                raise AgenticAdmissionError('derived source has an invalid inherited availability upper bound')
            execution_bytes = transformation_execution_receipts[receipt.receipt_id]
            if (hashlib.sha256(execution_bytes).hexdigest(), len(execution_bytes)) != (
                receipt.execution_receipt_sha256,
                receipt.execution_receipt_bytes,
            ):
                raise AgenticAdmissionError('transformation execution receipt byte binding mismatch')
            public_output_bytes = workspace.read_source(source_id)
            verified = transformation_verifier(
                receipt,
                input_bytes=tuple(workspace.read_source(parent) for parent in receipt.input_source_ids),
                output_bytes=public_output_bytes,
                execution_receipt_bytes=execution_bytes,
                span_map_artifact_bytes=span_map_bytes,
            )
            _validate_verified_transformation(
                policy,
                receipt,
                verified,
                output_bytes=public_output_bytes,
                execution_receipt_bytes=execution_bytes,
                span_map_artifact_bytes=span_map_bytes,
            )
            verified_transformations.append(verified)
            availability_by_source[source_id] = expected_available
            pending.remove(source_id)
            progressed = True
        if not progressed:
            raise AgenticAdmissionError('derived provenance graph cannot be closed from admitted raw sources')
    if {verified.receipt_id for verified in verified_transformations} != required_receipt_ids:
        raise AgenticAdmissionError('trusted verification did not cover every declared transformation receipt')
    if set(span_map_artifacts) != required_receipt_ids:
        raise AgenticAdmissionError('span-map artifacts must cover every declared transformation receipt')

    now = admitted_at or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise AgenticAdmissionError('admitted_at must include a UTC offset')
    release_pinned = policy.verifier_mode == 'release_pinned'
    strict = release_pinned and policy.required_profile in {
        AgenticAssuranceProfile.PROSPECTIVE_EXACT,
        AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE,
    }
    official = release_pinned and policy.required_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT
    return AgenticTemporalAdmission(
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        source_inventory_sha256=workspace.manifest.source_inventory_sha256,
        transformation_inventory_sha256=workspace.manifest.transformation_inventory_sha256,
        build_policy_sha256=workspace.manifest.build_policy_sha256,
        discovery_manifest_sha256=workspace.manifest.discovery_manifest_sha256,
        alias_seed_commitment_sha256=workspace.manifest.alias_seed_commitment_sha256,
        temporal_policy_sha256=agentic_model_sha256(policy),
        verifier_mode=policy.verifier_mode,
        assurance_profile=policy.required_profile,
        decision_at=workspace.task.decision_at,
        admitted_at=now,
        verified_source_proofs=tuple(sorted(verified_proofs, key=lambda value: value.proof_id)),
        task_definition_proof=verified_task,
        build_policy_proof=verified_build_policy,
        discovery_manifest_proof=verified_discovery_manifest,
        verified_transformations=tuple(sorted(verified_transformations, key=lambda value: value.receipt_id)),
        strict_exact_byte_provenance=strict,
        official_temporal_eligible=official,
        retrospective_only=policy.required_profile != AgenticAssuranceProfile.PROSPECTIVE_EXACT,
        selection_precommitted_before_cutoff=official,
        residual_retrospective_selection_contamination=not official,
    )


def finalize_workspace_admission(
    workspace: LoadedAgenticWorkspace,
    *,
    temporal_admission: AgenticTemporalAdmission,
    contamination_policy: AgenticContaminationAdmissionPolicy,
    contamination_binding: AgenticContaminationBinding,
    audit_artifacts: Mapping[str, bytes],
    expected_temporal_admission_sha256: str,
) -> AgenticWorkspaceAdmission:
    """Bind the built-in exact audit to a research-only, organizer-committed admission.

    This compiler cannot mint an official release.  That requires a separate authenticated
    organizer seal and a production-isolated worker implementation, neither of which V1 claims.
    """

    workspace = load_agentic_workspace(workspace.root)
    if agentic_model_sha256(temporal_admission) != expected_temporal_admission_sha256:
        raise AgenticAdmissionError('temporal admission does not match its organizer commitment')
    _require_temporal_binding(workspace, temporal_admission)
    expected_workspace = (
        workspace.manifest_sha256,
        workspace.manifest.workspace_tree_sha256,
        workspace.manifest.model_visible_surface_sha256,
    )
    actual_workspace = (
        contamination_binding.workspace_manifest_sha256,
        contamination_binding.workspace_tree_sha256,
        contamination_binding.model_visible_surface_sha256,
    )
    if actual_workspace != expected_workspace:
        raise AgenticAdmissionError('contamination binding does not cover the exact workspace')
    if (
        contamination_binding.protected_outcome_namespace_sha256
        != workspace.build_policy.protected_outcome_namespace_sha256
    ):
        raise AgenticAdmissionError('contamination binding does not cover the committed outcome namespace')
    if contamination_binding.audited_file_count != len(workspace.manifest.entries):
        raise AgenticAdmissionError('contamination audit file count does not cover the complete visible inventory')
    if (
        contamination_binding.contamination_audit_manifest_sha256 != contamination_policy.expected_audit_manifest_sha256
        or contamination_binding.contamination_audit_policy_sha256 != contamination_policy.expected_audit_policy_sha256
        or contamination_binding.protected_corpus_manifest_sha256
        != contamination_policy.expected_protected_corpus_manifest_sha256
    ):
        raise AgenticAdmissionError('contamination binding does not use the organizer-pinned audit and corpus')
    # Dynamic import avoids the adapter's schema import cycle while making the verifier itself
    # non-pluggable at this trust boundary.  Caller-supplied callbacks cannot attest themselves.
    from vaxreplay.agentic.contamination import (
        AGENTIC_AUDIT_VERIFIER_ID,
        AGENTIC_AUDIT_VERIFIER_VERSION,
        verify_agentic_contamination_audit,
    )

    if (
        contamination_policy.expected_verifier_id,
        contamination_policy.expected_verifier_version,
    ) != (AGENTIC_AUDIT_VERIFIER_ID, AGENTIC_AUDIT_VERIFIER_VERSION):
        raise AgenticAdmissionError('contamination policy does not pin the built-in trusted verifier')
    verified = verify_agentic_contamination_audit(
        contamination_binding,
        model_visible_surface=workspace.model_visible_surface,
        audit_artifacts=audit_artifacts,
    )
    if (
        verified.contamination_audit_manifest_sha256 != contamination_binding.contamination_audit_manifest_sha256
        or verified.contamination_audit_policy_sha256 != contamination_binding.contamination_audit_policy_sha256
        or verified.protected_corpus_manifest_sha256 != contamination_binding.protected_corpus_manifest_sha256
        or verified.audited_surface_sha256 != workspace.manifest.model_visible_surface_sha256
        or verified.audited_file_count != len(workspace.manifest.entries)
        or verified.verifier_id != contamination_policy.expected_verifier_id
        or verified.verifier_version != contamination_policy.expected_verifier_version
    ):
        raise AgenticAdmissionError('trusted contamination verifier returned a mismatched audit binding')
    if temporal_admission.assurance_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT:
        admitted_use: Literal[
            'prospective_research',
            'retrospective_research',
            'best_effort_research',
            'fixture',
        ] = 'prospective_research'
    elif temporal_admission.assurance_profile == AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE:
        admitted_use = 'retrospective_research'
    elif temporal_admission.assurance_profile == AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT:
        admitted_use = 'best_effort_research'
    else:
        admitted_use = 'fixture'
    return AgenticWorkspaceAdmission(
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        build_policy_sha256=workspace.manifest.build_policy_sha256,
        discovery_manifest_sha256=workspace.manifest.discovery_manifest_sha256,
        alias_seed_commitment_sha256=workspace.manifest.alias_seed_commitment_sha256,
        temporal_admission_sha256=agentic_model_sha256(temporal_admission),
        contamination_binding_sha256=agentic_model_sha256(contamination_binding),
        contamination_admission_policy_sha256=agentic_model_sha256(contamination_policy),
        contamination_audit_manifest_sha256=contamination_binding.contamination_audit_manifest_sha256,
        assurance_profile=temporal_admission.assurance_profile,
        admitted_use=admitted_use,
        official_release_eligible=False,
        selection_precommitted_before_cutoff=temporal_admission.selection_precommitted_before_cutoff,
        residual_retrospective_selection_contamination=(
            temporal_admission.residual_retrospective_selection_contamination
        ),
    )


def require_workspace_admission(
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    *,
    expected_admission_sha256: str,
) -> AgenticWorkspaceAdmission:
    workspace = load_agentic_workspace(workspace.root)
    if agentic_model_sha256(admission) != expected_admission_sha256:
        raise AgenticAdmissionError('workspace admission does not match its trusted release commitment')
    if (
        admission.workspace_manifest_sha256,
        admission.workspace_tree_sha256,
        admission.model_visible_surface_sha256,
        admission.build_policy_sha256,
        admission.discovery_manifest_sha256,
        admission.alias_seed_commitment_sha256,
        admission.assurance_profile,
    ) != (
        workspace.manifest_sha256,
        workspace.manifest.workspace_tree_sha256,
        workspace.manifest.model_visible_surface_sha256,
        workspace.manifest.build_policy_sha256,
        workspace.manifest.discovery_manifest_sha256,
        workspace.manifest.alias_seed_commitment_sha256,
        workspace.manifest.assurance_profile,
    ):
        raise AgenticAdmissionError('workspace admission is not bound to the exact current workspace')
    return admission


def _require_proof_kind(profile: AgenticAssuranceProfile, kind: TemporalProofKind) -> None:
    allowed = {
        AgenticAssuranceProfile.PROSPECTIVE_EXACT: {
            TemporalProofKind.RFC3161_TIMESTAMP,
            TemporalProofKind.PUBLIC_TRANSPARENCY_LOG,
        },
        AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE: {
            TemporalProofKind.RFC3161_TIMESTAMP,
            TemporalProofKind.PUBLIC_TRANSPARENCY_LOG,
            TemporalProofKind.INDEPENDENT_ARCHIVE_EXACT_BYTES,
        },
        AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT: {
            TemporalProofKind.RFC3161_TIMESTAMP,
            TemporalProofKind.PUBLIC_TRANSPARENCY_LOG,
            TemporalProofKind.INDEPENDENT_ARCHIVE_EXACT_BYTES,
            TemporalProofKind.SOURCE_SIGNED_DIGEST,
            TemporalProofKind.SOURCE_ATTESTED_SNAPSHOT,
        },
        AgenticAssuranceProfile.FIXTURE: {TemporalProofKind.FIXTURE},
    }[profile]
    if kind not in allowed:
        raise AgenticAdmissionError(f'{kind.value} cannot satisfy the {profile.value} assurance profile')


def _verify_proof_file_binding(proof: ArtifactTemporalProof, proof_bytes: bytes) -> None:
    if (hashlib.sha256(proof_bytes).hexdigest(), len(proof_bytes)) != (proof.proof_sha256, proof.proof_bytes):
        raise AgenticAdmissionError('temporal proof artifact does not match its exact byte binding')


def _verify_workspace_metadata_proof(
    *,
    proof: ArtifactTemporalProof,
    artifact: bytes,
    expected_sha256: str,
    proof_bytes: bytes,
    proof_verifier: TemporalProofVerifier,
    temporal_policy: AgenticTemporalPolicy,
    decision_at: datetime,
) -> VerifiedTemporalProof:
    if (proof.artifact_sha256, proof.artifact_bytes) != (expected_sha256, len(artifact)):
        raise AgenticAdmissionError('prospective metadata proof does not bind the exact workspace artifact')
    _require_proof_kind(AgenticAssuranceProfile.PROSPECTIVE_EXACT, proof.kind)
    _verify_proof_file_binding(proof, proof_bytes)
    verified = proof_verifier(proof, artifact_bytes=artifact, proof_bytes=proof_bytes)
    _validate_verified_proof(verified, proof, decision_at=decision_at)
    _validate_trusted_verifier(temporal_policy, proof, verified)
    return verified


def _validate_trusted_verifier(
    policy: AgenticTemporalPolicy,
    proof: ArtifactTemporalProof,
    verified: VerifiedTemporalProof,
) -> None:
    if policy.verifier_mode == 'conformance':
        return
    match = next(
        (
            item
            for item in policy.trusted_verifiers
            if item.proof_kind == proof.kind and item.authority_id == proof.authority_id
        ),
        None,
    )
    if match is None:
        raise AgenticAdmissionError('temporal proof authority is absent from the release-pinned verifier registry')
    if (
        verified.verifier_id,
        verified.verifier_version,
        verified.verifier_executable_sha256,
    ) != (
        match.verifier_id,
        match.verifier_version,
        match.verifier_executable_sha256,
    ):
        raise AgenticAdmissionError('temporal proof was not verified by the release-pinned verifier implementation')


def _validate_verified_proof(
    verified: VerifiedTemporalProof,
    proof: ArtifactTemporalProof,
    *,
    decision_at: datetime,
) -> None:
    if (
        verified.proof_id,
        verified.artifact_sha256,
        verified.artifact_bytes,
        verified.kind,
        verified.witnessed,
    ) != (
        proof.proof_id,
        proof.artifact_sha256,
        proof.artifact_bytes,
        proof.kind,
        proof.witnessed,
    ):
        raise AgenticAdmissionError('trusted proof verifier returned a mismatched proof result')
    if verified.witnessed.upper_at > decision_at:
        raise AgenticAdmissionError('verified exact-byte witness crosses the historical cutoff')


def _validate_verified_transformation(
    policy: AgenticTemporalPolicy,
    receipt: AgenticTransformationReceipt,
    verified: VerifiedTransformation,
    *,
    output_bytes: bytes,
    execution_receipt_bytes: bytes,
    span_map_artifact_bytes: bytes,
) -> None:
    """Bind a re-execution result to exact transform, verifier, receipt, and output identities."""

    output_binding = (hashlib.sha256(output_bytes).hexdigest(), len(output_bytes))
    execution_binding = (hashlib.sha256(execution_receipt_bytes).hexdigest(), len(execution_receipt_bytes))
    span_map_binding = (hashlib.sha256(span_map_artifact_bytes).hexdigest(), len(span_map_artifact_bytes))
    if (
        output_binding,
        execution_binding,
        span_map_binding,
    ) != (
        (receipt.output_sha256, receipt.output_bytes),
        (receipt.execution_receipt_sha256, receipt.execution_receipt_bytes),
        (receipt.span_map_sha256, receipt.span_map_bytes),
    ):
        raise AgenticAdmissionError('transformation artifacts do not match their exact receipt bindings')
    if (
        verified.receipt_id,
        verified.transform_id,
        verified.transform_version,
        verified.transform_executable_sha256,
        verified.output_sha256,
        verified.output_bytes,
        verified.span_map_sha256,
        verified.span_map_bytes,
        verified.execution_receipt_sha256,
        verified.execution_receipt_bytes,
        verified.reexecution_output_sha256,
        verified.reexecution_output_bytes,
    ) != (
        receipt.receipt_id,
        receipt.transform_id,
        receipt.transform_version,
        receipt.executable_sha256,
        *output_binding,
        *span_map_binding,
        *execution_binding,
        *output_binding,
    ):
        raise AgenticAdmissionError(
            'transformation verification did not bind the exact transform, receipt artifacts, and re-executed output'
        )
    if policy.verifier_mode == 'conformance':
        return
    match = next(
        (
            item
            for item in policy.trusted_transformation_verifiers
            if item.transform_id == receipt.transform_id and item.transform_version == receipt.transform_version
        ),
        None,
    )
    if match is None:
        raise AgenticAdmissionError('transformation is absent from the release-pinned transformation-verifier registry')
    if (
        receipt.executable_sha256,
        verified.verifier_id,
        verified.verifier_version,
        verified.verifier_executable_sha256,
    ) != (
        match.transform_executable_sha256,
        match.verifier_id,
        match.verifier_version,
        match.verifier_executable_sha256,
    ):
        raise AgenticAdmissionError(
            'transformation was not re-executed by the release-pinned transform and verifier implementations'
        )


def _validate_transform_policy(receipt: AgenticTransformationReceipt) -> None:
    if (
        receipt.kind != AgenticDerivationKind.DETERMINISTIC
        or receipt.semantic_rewrite
        or not receipt.source_span_mapping_complete
        or receipt.network_allowed
        or receipt.outcome_namespace_mounted
        or not receipt.label_blind
    ):
        raise AgenticAdmissionError('Agentic V1 admits only deterministic, extractive, label-blind transformations')


def _require_temporal_binding(
    workspace: LoadedAgenticWorkspace,
    admission: AgenticTemporalAdmission,
) -> None:
    if (
        admission.workspace_manifest_sha256,
        admission.workspace_tree_sha256,
        admission.model_visible_surface_sha256,
        admission.source_inventory_sha256,
        admission.transformation_inventory_sha256,
        admission.build_policy_sha256,
        admission.discovery_manifest_sha256,
        admission.alias_seed_commitment_sha256,
        admission.assurance_profile,
        admission.decision_at,
    ) != (
        workspace.manifest_sha256,
        workspace.manifest.workspace_tree_sha256,
        workspace.manifest.model_visible_surface_sha256,
        workspace.manifest.source_inventory_sha256,
        workspace.manifest.transformation_inventory_sha256,
        workspace.manifest.build_policy_sha256,
        workspace.manifest.discovery_manifest_sha256,
        workspace.manifest.alias_seed_commitment_sha256,
        workspace.manifest.assurance_profile,
        workspace.task.decision_at,
    ):
        raise AgenticAdmissionError('temporal admission is not bound to the exact current workspace')
