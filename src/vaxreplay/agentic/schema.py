"""Strict contracts for Agentic Replay tasks and frozen workspaces."""

from __future__ import annotations

import enum
import hashlib
import unicodedata
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import LabelCommitmentScheme, Split, StrictModel

AGENTIC_TASK_SCHEMA_VERSION = 'vaxreplay.agentic-task.v0.1'
AGENTIC_BUILD_POLICY_SCHEMA_VERSION = 'vaxreplay.agentic-build-policy.v0.2'
AGENTIC_DISCOVERY_MANIFEST_SCHEMA_VERSION = 'vaxreplay.agentic-discovery-manifest.v0.2'
AGENTIC_WORKSPACE_MANIFEST_SCHEMA_VERSION = 'vaxreplay.agentic-workspace-manifest.v0.2'
AGENTIC_TEMPORAL_POLICY_SCHEMA_VERSION = 'vaxreplay.agentic-temporal-policy.v0.1'
AGENTIC_TEMPORAL_ADMISSION_SCHEMA_VERSION = 'vaxreplay.agentic-temporal-admission.v0.1'
AGENTIC_CONTAMINATION_BINDING_SCHEMA_VERSION = 'vaxreplay.agentic-contamination-binding.v0.1'
AGENTIC_CONTAMINATION_ADMISSION_POLICY_SCHEMA_VERSION = 'vaxreplay.agentic-contamination-admission-policy.v0.1'
AGENTIC_WORKSPACE_ADMISSION_SCHEMA_VERSION = 'vaxreplay.agentic-workspace-admission.v0.1'
AGENTIC_ALIAS_PERMUTATION_RECEIPT_SCHEMA_VERSION = 'vaxreplay.agentic-alias-permutation-receipt.v0.1'
AGENTIC_LOGICAL_WORKSPACE_CONTRACT_VERSION = 'vaxreplay.agentic-logical-workspace.v0.1'

_AGENTIC_LOGICAL_WORKSPACE_CONTRACT = {
    'schema_version': AGENTIC_LOGICAL_WORKSPACE_CONTRACT_VERSION,
    'operations': ['list', 'read', 'search'],
    'visible_metadata': ['path', 'media_type', 'sha256', 'byte_count'],
    'filesystem_metadata_visible': False,
    'raw_host_paths_visible': False,
    'private_workspace_visible': False,
}
AGENTIC_LOGICAL_WORKSPACE_CONTRACT_SHA256 = hashlib.sha256(
    canonical_json_bytes(_AGENTIC_LOGICAL_WORKSPACE_CONTRACT)
).hexdigest()

_SHA256_PATTERN = r'^[0-9a-f]{64}$'


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def normalized_relative_path(value: str, *, prefix: str | None = None) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError('workspace paths must be normalized relative POSIX paths')
    if not path.parts or any(part in {'', '.', '..'} or part.startswith('.') for part in path.parts):
        raise ValueError('workspace paths cannot contain empty, dot, or hidden components')
    if unicodedata.normalize('NFC', value) != value:
        raise ValueError('workspace paths must use NFC Unicode normalization')
    if '\x00' in value:
        raise ValueError('workspace paths cannot contain NUL')
    if prefix is not None and (not path.parts or path.parts[0] != prefix):
        raise ValueError(f'workspace path must remain inside {prefix}/')
    return value


class AgenticAssuranceProfile(str, enum.Enum):
    PROSPECTIVE_EXACT = 'prospective_exact'
    INDEPENDENT_EXACT_BYTE = 'independent_exact_byte'
    SOURCE_ATTESTED_BEST_EFFORT = 'source_attested_best_effort'
    FIXTURE = 'fixture'


class AvailabilityClaimKind(str, enum.Enum):
    PUBLICATION = 'publication'
    ONLINE_FIRST = 'online_first'
    DATABASE_ROW_DATE = 'database_row_date'
    SOURCE_ARCHIVE_DATE = 'source_archive_date'
    HTTP_LAST_MODIFIED = 'http_last_modified'
    RETRIEVAL = 'retrieval'


class AvailabilityScope(str, enum.Enum):
    WORK = 'work'
    VERSION = 'version'
    EXACT_BYTES = 'exact_bytes'


class TemporalProofKind(str, enum.Enum):
    RFC3161_TIMESTAMP = 'rfc3161_timestamp'
    PUBLIC_TRANSPARENCY_LOG = 'public_transparency_log'
    INDEPENDENT_ARCHIVE_EXACT_BYTES = 'independent_archive_exact_bytes'
    SOURCE_SIGNED_DIGEST = 'source_signed_digest'
    SOURCE_ATTESTED_SNAPSHOT = 'source_attested_snapshot'
    FIXTURE = 'fixture'


class AgenticArtifactKind(str, enum.Enum):
    RAW = 'raw'
    DERIVED = 'derived'


class AgenticDerivationKind(str, enum.Enum):
    DETERMINISTIC = 'deterministic'
    HUMAN = 'human'
    LLM = 'llm'


class AgenticDiscoveryDisposition(str, enum.Enum):
    INCLUDED = 'included'
    EXCLUDED = 'excluded'
    QUARANTINED = 'quarantined'


class AgenticMediaType(str, enum.Enum):
    TEXT = 'text/plain'
    MARKDOWN = 'text/markdown'
    CSV = 'text/csv'
    TSV = 'text/tab-separated-values'
    JSON = 'application/json'
    JSONL = 'application/x-ndjson'


class AgenticValueType(str, enum.Enum):
    STRING = 'string'
    NUMBER = 'number'
    BOOLEAN = 'boolean'


_MEDIA_EXTENSIONS = {
    AgenticMediaType.TEXT: {'.txt'},
    AgenticMediaType.MARKDOWN: {'.md'},
    AgenticMediaType.CSV: {'.csv'},
    AgenticMediaType.TSV: {'.tsv'},
    AgenticMediaType.JSON: {'.json'},
    AgenticMediaType.JSONL: {'.jsonl'},
}


class AvailabilityInterval(StrictModel):
    lower_at: datetime
    upper_at: datetime
    precision: Literal['instant', 'day', 'month', 'year', 'unknown']
    timezone_basis: str = Field(min_length=1)

    @field_validator('lower_at', 'upper_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _aware(value, info.field_name)

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.upper_at < self.lower_at:
            raise ValueError('availability interval upper_at cannot precede lower_at')
        return self


class ArtifactAvailabilityClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    kind: AvailabilityClaimKind
    scope: AvailabilityScope
    interval: AvailabilityInterval
    issuer: str = Field(min_length=1)
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    note: str = Field(min_length=1)


class ArtifactTemporalProof(StrictModel):
    proof_id: str = Field(min_length=1)
    kind: TemporalProofKind
    scope: Literal['exact_bytes'] = 'exact_bytes'
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_bytes: int = Field(gt=0)
    witnessed: AvailabilityInterval
    authority_id: str = Field(min_length=1)
    proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)


class AgenticFactQuery(StrictModel):
    query_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    value_type: AgenticValueType
    unit: str | None = None
    candidate_id: str | None = None


class AgenticDerivedMetric(StrictModel):
    metric_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    value_type: AgenticValueType
    unit: str | None = None
    formula_id: str = Field(min_length=1)
    dependency_query_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator('dependency_query_ids')
    @classmethod
    def validate_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('derived metric dependencies must be unique and sorted')
        return value


class AgenticTaskEnvelope(StrictModel):
    schema_version: Literal['vaxreplay.agentic-task.v0.1'] = AGENTIC_TASK_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    episode_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_at: datetime
    task_type: str = Field(min_length=1)
    candidate_ids: tuple[str, ...] = Field(min_length=2)
    portfolio_size: int = Field(gt=0)
    instructions: str = Field(min_length=1, max_length=100_000)
    fact_queries: tuple[AgenticFactQuery, ...] = Field(min_length=1)
    derived_metrics: tuple[AgenticDerivedMetric, ...] = ()
    historically_preregistered: bool
    fixed_candidate_universe: Literal[True] = True
    response_protocol: Literal['vaxreplay.agentic-submission-file.v0.1'] = 'vaxreplay.agentic-submission-file.v0.1'

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return _aware(value, 'decision_at')

    @model_validator(mode='after')
    def validate_task(self) -> Self:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError('candidate_ids must be unique')
        if self.portfolio_size >= len(self.candidate_ids):
            raise ValueError('portfolio_size must be smaller than the candidate universe')
        query_ids = tuple(query.query_id for query in self.fact_queries)
        if query_ids != tuple(sorted(query_ids)) or len(query_ids) != len(set(query_ids)):
            raise ValueError('fact queries must use unique query IDs in sorted order')
        metric_ids = tuple(metric.metric_id for metric in self.derived_metrics)
        if metric_ids != tuple(sorted(metric_ids)) or len(metric_ids) != len(set(metric_ids)):
            raise ValueError('derived metrics must use unique metric IDs in sorted order')
        query_id_set = set(query_ids)
        if any(not set(metric.dependency_query_ids).issubset(query_id_set) for metric in self.derived_metrics):
            raise ValueError('derived metric dependencies must reference public fact queries')
        if any(
            query.candidate_id is not None and query.candidate_id not in self.candidate_ids
            for query in self.fact_queries
        ):
            raise ValueError('fact queries cannot reference candidates outside the fixed universe')
        return self


class AgenticTransformCommitment(StrictModel):
    transform_id: str = Field(min_length=1)
    transform_version: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)


class AgenticCandidateAliasAssignment(StrictModel):
    """Organizer-private mapping commitment for one public candidate position."""

    candidate_key_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_candidate_id: str = Field(pattern=r'^candidate-[0-9]{3}$')
    presentation_index: int = Field(ge=0)


class AgenticSourceAliasAssignment(StrictModel):
    """Organizer-private binding for one neutral source presentation."""

    private_source_key_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_bytes: int = Field(gt=0)
    public_source_id: str = Field(pattern=r'^source-[0-9]{3}$')
    public_path: str = Field(min_length=1)
    public_title: str = Field(pattern=r'^Source [0-9]{3}$')
    presentation_index: int = Field(ge=0)

    @field_validator('public_path')
    @classmethod
    def validate_public_path(cls, value: str) -> str:
        normalized_relative_path(value, prefix='sources')
        stem = PurePosixPath(value).stem
        if stem != PurePosixPath(value).name.removesuffix(PurePosixPath(value).suffix) or not stem.startswith(
            'source-'
        ):
            raise ValueError('public source paths must use neutral source-NNN names')
        return value


class AgenticAliasPermutationReceipt(StrictModel):
    """Private, reproducible receipt for alias generation and presentation permutation."""

    schema_version: Literal['vaxreplay.agentic-alias-permutation-receipt.v0.1'] = (
        AGENTIC_ALIAS_PERMUTATION_RECEIPT_SCHEMA_VERSION
    )
    receipt_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    alias_scheme_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_seed_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    permutation_algorithm_id: str = Field(min_length=1)
    generator_id: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    generator_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    generator_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_receipt_bytes: int = Field(gt=0)
    generated_at: datetime
    candidate_order_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_order_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_assignments: tuple[AgenticCandidateAliasAssignment, ...] = Field(min_length=2)
    source_assignments: tuple[AgenticSourceAliasAssignment, ...] = Field(min_length=1)
    labels_mounted: Literal[False] = False
    outcome_namespace_mounted: Literal[False] = False
    identity_or_discovery_order_used: Literal[False] = False
    structured_identifiers_paths_and_titles_neutralized: Literal[True] = True

    @field_validator('generated_at')
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _aware(value, 'generated_at')

    @model_validator(mode='after')
    def validate_assignments(self) -> Self:
        candidate_indices = tuple(item.presentation_index for item in self.candidate_assignments)
        candidate_ids = tuple(item.public_candidate_id for item in self.candidate_assignments)
        candidate_keys = tuple(item.candidate_key_commitment_sha256 for item in self.candidate_assignments)
        if candidate_indices != tuple(range(len(self.candidate_assignments))):
            raise ValueError('candidate alias assignments must use contiguous presentation order')
        if candidate_ids != tuple(f'candidate-{index:03d}' for index in range(1, len(candidate_ids) + 1)):
            raise ValueError('candidate aliases must be contiguous neutral IDs in presentation order')
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError('candidate alias assignments must use unique private key commitments')
        if self.candidate_order_sha256 != hashlib.sha256(canonical_json_bytes(list(candidate_ids))).hexdigest():
            raise ValueError('candidate order commitment does not match the alias assignments')

        source_indices = tuple(item.presentation_index for item in self.source_assignments)
        source_ids = tuple(item.public_source_id for item in self.source_assignments)
        source_keys = tuple(item.private_source_key_commitment_sha256 for item in self.source_assignments)
        if source_indices != tuple(range(len(self.source_assignments))):
            raise ValueError('source alias assignments must use contiguous presentation order')
        if source_ids != tuple(f'source-{index:03d}' for index in range(1, len(source_ids) + 1)):
            raise ValueError('source aliases must be contiguous neutral IDs in presentation order')
        if len(source_keys) != len(set(source_keys)):
            raise ValueError('source alias assignments must use unique private key commitments')
        if self.source_order_sha256 != hashlib.sha256(canonical_json_bytes(list(source_ids))).hexdigest():
            raise ValueError('source order commitment does not match the alias assignments')
        return self


class AgenticBuildPolicy(StrictModel):
    """Organizer-private, outcome-blind rules committed before workspace construction."""

    schema_version: Literal['vaxreplay.agentic-build-policy.v0.2'] = AGENTIC_BUILD_POLICY_SCHEMA_VERSION
    policy_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    decision_at: datetime
    created_at: datetime
    discovery_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_rule_sha256: str = Field(pattern=_SHA256_PATTERN)
    inclusion_rule_sha256: str = Field(pattern=_SHA256_PATTERN)
    deduplication_rule_sha256: str = Field(pattern=_SHA256_PATTERN)
    distractor_rule_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_scheme_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_seed_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_permutation_algorithm_id: str = Field(min_length=1)
    alias_generator_id: str = Field(min_length=1)
    alias_generator_version: str = Field(min_length=1)
    alias_generator_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_generator_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    protected_outcome_namespace_sha256: str = Field(pattern=_SHA256_PATTERN)
    allowed_transforms: tuple[AgenticTransformCommitment, ...] = ()
    labels_mounted: Literal[False] = False
    outcome_namespace_mounted: Literal[False] = False
    outcome_blind_selection_required: Literal[True] = True
    secret_random_aliases_required: Literal[True] = True
    identity_or_discovery_order_aliases_allowed: Literal[False] = False
    neutral_paths_and_titles_required: Literal[True] = True

    @field_validator('decision_at', 'created_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _aware(value, info.field_name)

    @field_validator('allowed_transforms')
    @classmethod
    def validate_transforms(
        cls,
        value: tuple[AgenticTransformCommitment, ...],
    ) -> tuple[AgenticTransformCommitment, ...]:
        identities = tuple((item.transform_id, item.transform_version) for item in value)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError('allowed transforms must use unique identities in sorted order')
        return value

    @model_validator(mode='after')
    def validate_policy_times(self) -> Self:
        if self.created_at > self.decision_at:
            # Retrospective policy creation is allowed but must be represented honestly by using
            # the actual post-cutoff timestamp; only a prospective proof can remove residual risk.
            return self
        return self


class AgenticDiscoveredSource(StrictModel):
    discovery_id: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_bytes: int = Field(gt=0)
    selected_temporal_proof_id: str = Field(min_length=1)
    effective_available_at_upper: datetime
    disposition: AgenticDiscoveryDisposition
    reason_code: str = Field(min_length=1)
    workspace_source_id: str | None = None

    @field_validator('effective_available_at_upper')
    @classmethod
    def validate_available_at(cls, value: datetime) -> datetime:
        return _aware(value, 'effective_available_at_upper')

    @model_validator(mode='after')
    def validate_disposition(self) -> Self:
        if (self.disposition == AgenticDiscoveryDisposition.INCLUDED) != (self.workspace_source_id is not None):
            raise ValueError('only included discovered sources may have a workspace_source_id')
        return self


class AgenticDiscoveredCandidate(StrictModel):
    candidate_key_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    disposition: AgenticDiscoveryDisposition
    reason_code: str = Field(min_length=1)
    public_candidate_id: str | None = None

    @model_validator(mode='after')
    def validate_disposition(self) -> Self:
        if (self.disposition == AgenticDiscoveryDisposition.INCLUDED) != (self.public_candidate_id is not None):
            raise ValueError('only included discovered candidates may have a public candidate ID')
        return self


class AgenticDiscoveryManifest(StrictModel):
    """Complete disposition ledger for one pinned discovery capture."""

    schema_version: Literal['vaxreplay.agentic-discovery-manifest.v0.2'] = AGENTIC_DISCOVERY_MANIFEST_SCHEMA_VERSION
    manifest_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    build_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    discovery_capture_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_permutation_receipt: AgenticAliasPermutationReceipt
    created_at: datetime
    capture_complete_under_policy: Literal[True] = True
    every_discovered_item_dispositioned: Literal[True] = True
    proves_global_source_completeness: Literal[False] = False
    sources: tuple[AgenticDiscoveredSource, ...] = Field(min_length=1)
    candidates: tuple[AgenticDiscoveredCandidate, ...] = Field(min_length=2)

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _aware(value, 'created_at')

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        if self.alias_permutation_receipt.task_id != self.task_id:
            raise ValueError('alias permutation receipt must match the discovery task')
        if self.alias_permutation_receipt.generated_at > self.created_at:
            raise ValueError('alias permutation receipt cannot postdate its discovery manifest')
        discovery_ids = tuple(source.discovery_id for source in self.sources)
        if discovery_ids != tuple(sorted(discovery_ids)) or len(discovery_ids) != len(set(discovery_ids)):
            raise ValueError('discovered sources must use unique discovery IDs in sorted order')
        workspace_ids = tuple(
            source.workspace_source_id for source in self.sources if source.workspace_source_id is not None
        )
        if len(workspace_ids) != len(set(workspace_ids)):
            raise ValueError('included discovered sources must use unique workspace source IDs')
        candidate_keys = tuple(candidate.candidate_key_commitment_sha256 for candidate in self.candidates)
        if candidate_keys != tuple(sorted(candidate_keys)) or len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError('discovered candidates must use unique key commitments in sorted order')
        public_ids = tuple(
            candidate.public_candidate_id for candidate in self.candidates if candidate.public_candidate_id is not None
        )
        if len(public_ids) != len(set(public_ids)):
            raise ValueError('included discovered candidates must use unique public aliases')
        return self


class AgenticWorkspaceSource(StrictModel):
    source_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    display_title: str = Field(min_length=1, max_length=500)
    artifact_kind: AgenticArtifactKind
    media_type: AgenticMediaType
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)
    source_url: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    retrieved_at: datetime
    availability_claims: tuple[ArtifactAvailabilityClaim, ...] = ()
    temporal_proofs: tuple[ArtifactTemporalProof, ...] = ()
    selected_proof_id: str | None = None
    effective_available_at_upper: datetime
    parent_source_ids: tuple[str, ...] = ()
    transformation_receipt_id: str | None = None

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalized_relative_path(value, prefix='sources')

    @field_validator('retrieved_at', 'effective_available_at_upper')
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, info.field_name)

    @model_validator(mode='after')
    def validate_source(self) -> Self:
        suffix = PurePosixPath(self.path).suffix.casefold()
        if suffix not in _MEDIA_EXTENSIONS[self.media_type]:
            raise ValueError('workspace source extension does not match its media type')
        claim_ids = tuple(claim.claim_id for claim in self.availability_claims)
        if claim_ids != tuple(sorted(claim_ids)) or len(claim_ids) != len(set(claim_ids)):
            raise ValueError('availability claims must use unique IDs in sorted order')
        proof_ids = tuple(proof.proof_id for proof in self.temporal_proofs)
        if proof_ids != tuple(sorted(proof_ids)) or len(proof_ids) != len(set(proof_ids)):
            raise ValueError('temporal proofs must use unique IDs in sorted order')
        if self.artifact_kind == AgenticArtifactKind.RAW:
            if not self.temporal_proofs or self.selected_proof_id not in set(proof_ids):
                raise ValueError('raw sources require a selected exact-byte temporal proof')
            if self.parent_source_ids or self.transformation_receipt_id is not None:
                raise ValueError('raw sources cannot declare transformation parents')
            selected = next(proof for proof in self.temporal_proofs if proof.proof_id == self.selected_proof_id)
            if (selected.artifact_sha256, selected.artifact_bytes) != (self.sha256, self.byte_count):
                raise ValueError('selected temporal proof must bind the exact workspace source bytes')
            if selected.witnessed.upper_at != self.effective_available_at_upper:
                raise ValueError('raw source availability must use the selected proof conservative upper bound')
        else:
            if self.temporal_proofs or self.selected_proof_id is not None:
                raise ValueError('derived sources inherit availability from parents rather than temporal proofs')
            if not self.parent_source_ids or self.transformation_receipt_id is None:
                raise ValueError('derived sources require parents and a transformation receipt')
            if self.parent_source_ids != tuple(sorted(self.parent_source_ids)) or len(self.parent_source_ids) != len(
                set(self.parent_source_ids)
            ):
                raise ValueError('derived source parents must be unique and sorted')
        return self


class AgenticTransformationReceipt(StrictModel):
    receipt_id: str = Field(min_length=1)
    kind: AgenticDerivationKind
    input_source_ids: tuple[str, ...] = Field(min_length=1)
    output_source_id: str = Field(min_length=1)
    output_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_bytes: int = Field(gt=0)
    transform_id: str = Field(min_length=1)
    transform_version: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_receipt_bytes: int = Field(gt=0)
    executed_at: datetime
    network_allowed: Literal[False] = False
    outcome_namespace_mounted: Literal[False] = False
    label_blind: Literal[True] = True
    semantic_rewrite: bool
    source_span_mapping_complete: bool
    span_map_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    span_map_bytes: int | None = Field(default=None, gt=0)

    @field_validator('executed_at')
    @classmethod
    def validate_executed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'executed_at')

    @field_validator('input_source_ids')
    @classmethod
    def validate_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('transformation inputs must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_span_map_binding(self) -> Self:
        has_span_map_binding = self.span_map_sha256 is not None and self.span_map_bytes is not None
        if (self.span_map_sha256 is None) != (self.span_map_bytes is None):
            raise ValueError('span-map SHA-256 and byte count must be present or absent together')
        if self.source_span_mapping_complete != has_span_map_binding:
            raise ValueError('complete source-span mapping requires an exact private span-map binding')
        return self


class AgenticSourceCatalogEntry(StrictModel):
    source_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    media_type: AgenticMediaType

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalized_relative_path(value, prefix='sources')


class AgenticWorkspaceEntry(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)
    media_type: AgenticMediaType
    provenance_node_id: str = Field(min_length=1)
    mode: Literal['0444'] = '0444'

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalized_relative_path(value)


class AgenticWorkspaceManifest(StrictModel):
    schema_version: Literal['vaxreplay.agentic-workspace-manifest.v0.2'] = AGENTIC_WORKSPACE_MANIFEST_SCHEMA_VERSION
    workspace_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    episode_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_at: datetime
    assurance_profile: AgenticAssuranceProfile
    historically_preregistered: bool
    build_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    discovery_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_seed_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_permutation_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    transformation_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_synthetic: bool
    episode_split: Split
    episode_label_commitment_scheme: LabelCommitmentScheme
    episode_reward_version: str = Field(min_length=1)
    prospective_input_structurally_eligible: bool
    official_release_ready: Literal[False] = False
    worker_input_contract: Literal['vaxreplay.agentic-logical-workspace.v0.1'] = (
        AGENTIC_LOGICAL_WORKSPACE_CONTRACT_VERSION
    )
    worker_input_contract_sha256: str = Field(
        default=AGENTIC_LOGICAL_WORKSPACE_CONTRACT_SHA256,
        pattern=_SHA256_PATTERN,
    )
    raw_host_filesystem_exposure_sealed: Literal[False] = False
    exact_inventory: Literal[True] = True
    labels_mounted: Literal[False] = False
    outcome_namespace_mounted: Literal[False] = False
    entries: tuple[AgenticWorkspaceEntry, ...] = Field(min_length=4)

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return _aware(value, 'decision_at')

    @field_validator('entries')
    @classmethod
    def validate_entries(cls, value: tuple[AgenticWorkspaceEntry, ...]) -> tuple[AgenticWorkspaceEntry, ...]:
        paths = tuple(entry.path for entry in value)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('workspace entries must use unique paths in sorted order')
        folded = tuple(path.casefold() for path in paths)
        if len(folded) != len(set(folded)):
            raise ValueError('workspace paths cannot collide under Unicode case folding')
        required = {'TASK.json', 'TASK.md', 'source-catalog.json'}
        if not required.issubset(paths) or not any(path.startswith('sources/') for path in paths):
            raise ValueError('workspace must contain task files, a catalog, and at least one source')
        return value

    @model_validator(mode='after')
    def validate_release_properties(self) -> Self:
        if self.worker_input_contract_sha256 != AGENTIC_LOGICAL_WORKSPACE_CONTRACT_SHA256:
            raise ValueError('worker input contract digest does not match the supported logical broker contract')
        structurally_eligible = (
            not self.episode_synthetic
            and self.episode_split == Split.TEST
            and self.episode_label_commitment_scheme == LabelCommitmentScheme.HMAC_SHA256
            and self.assurance_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT
            and self.historically_preregistered
        )
        if self.prospective_input_structurally_eligible != structurally_eligible:
            raise ValueError('prospective input structural eligibility does not match bound episode properties')
        return self


def agentic_model_sha256(value: StrictModel) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
