"""Strict contracts for a Tier B literature replay.

The schemas deliberately distinguish independently archived source bytes from derived benchmark
artifacts.  Archive proofs establish a conservative upper bound on public availability for exact
document bytes.  A separate decision seal binds the complete panel and label-blind extraction
before later outcomes may be joined.
"""

from __future__ import annotations

import enum
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import EvidenceStance, Split, StrictModel
from vaxreplay.temporal_schema import TemporalReceiptAuthority

ARCHIVE_PROOF_SCHEMA_VERSION = 'vaxreplay.literature-archive-proof.v0.1'
ARCHIVED_DOCUMENT_SCHEMA_VERSION = 'vaxreplay.literature-document.v0.1'
CORPUS_SCHEMA_VERSION = 'vaxreplay.literature-corpus.v0.1'
PANEL_SCHEMA_VERSION = 'vaxreplay.literature-panel.v0.1'
EXTRACTION_SCHEMA_VERSION = 'vaxreplay.literature-extraction.v0.1'
DECISION_SEAL_SCHEMA_VERSION = 'vaxreplay.literature-decision-seal.v0.1'
DECISION_PACKAGE_SCHEMA_VERSION = 'vaxreplay.literature-decision-package.v0.1'
OUTCOME_AUDIT_SCHEMA_VERSION = 'vaxreplay.literature-outcome-audit.v0.1'
OUTCOME_PACKAGE_SCHEMA_VERSION = 'vaxreplay.literature-outcome-package.v0.1'
SOURCE_AUDIT_SCHEMA_VERSION = 'vaxreplay.literature-source-audit.v0.1'
EVALUATION_CONFIG_SCHEMA_VERSION = 'vaxreplay.literature-evaluation-config.v0.1'

_INDEPENDENT_AUTHORITIES = {
    TemporalReceiptAuthority.RFC3161_TIMESTAMP,
    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
    TemporalReceiptAuthority.SOURCE_SIGNED_VERSION,
    TemporalReceiptAuthority.INDEPENDENT_ARCHIVE,
}


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _relative_path(value: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError(f'{field_name} must be a normalized relative path')
    return value


class ExtractionKind(str, enum.Enum):
    DETERMINISTIC = 'deterministic'
    HUMAN = 'human'
    LLM = 'llm'


class OutcomeJoinStatus(str, enum.Enum):
    OBSERVED = 'observed'
    CENSORED = 'censored'
    MISSING = 'missing'
    CONFLICT = 'conflict'


class LiteratureEvaluationConfig(StrictModel):
    """Reward-affecting configuration sealed before any outcome join."""

    schema_version: Literal['vaxreplay.literature-evaluation-config.v0.1'] = EVALUATION_CONFIG_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    lineage_group_id: str = Field(min_length=1)
    synthetic: bool
    split: Split = Split.TEST
    portfolio_size: int = Field(gt=0)
    target_id: str = Field(min_length=1)
    horizon_days: int = Field(gt=0)
    required_dimensions: tuple[str, ...] = Field(min_length=1)
    adjudication_version: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_config(self) -> Self:
        if self.split != Split.TEST:
            raise ValueError('literature replay evaluations must use the test split')
        if len(self.required_dimensions) != len(set(self.required_dimensions)):
            raise ValueError('required dimensions must be unique')
        if tuple(sorted(self.required_dimensions)) != self.required_dimensions:
            raise ValueError('required dimensions must be sorted')
        return self


class ArchiveProof(StrictModel):
    """Independent proof over one exact archived document version."""

    schema_version: Literal['vaxreplay.literature-archive-proof.v0.1'] = ARCHIVE_PROOF_SCHEMA_VERSION
    proof_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    artifact_bytes: int = Field(gt=0)
    witnessed_at: datetime
    authority_type: TemporalReceiptAuthority
    authority_id: str = Field(min_length=1)
    proof_path: str = Field(min_length=1)
    proof_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)
    fixture_only: bool = False

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'witnessed_at')

    @field_validator('proof_path')
    @classmethod
    def validate_proof_path(cls, value: str) -> str:
        return _relative_path(value, 'proof_path')

    @model_validator(mode='after')
    def validate_authority(self) -> Self:
        if self.authority_type not in _INDEPENDENT_AUTHORITIES:
            raise ValueError('literature archive proofs require an independent authority')
        return self


class ArchivedLiteratureDocument(StrictModel):
    """Exact raw bytes, deterministic text view, and conservative availability resolution."""

    schema_version: Literal['vaxreplay.literature-document.v0.1'] = ARCHIVED_DOCUMENT_SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    canonical_id: str = Field(min_length=1)
    raw_path: str = Field(min_length=1)
    raw_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    raw_bytes: int = Field(gt=0)
    media_type: str = Field(min_length=1)
    text_path: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    text_bytes: int = Field(gt=0)
    text_derivation: str = Field(min_length=1)
    resolved_available_at: datetime
    selected_proof_id: str = Field(min_length=1)
    archive_proofs: tuple[ArchiveProof, ...] = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    citation: str = Field(min_length=1)

    @field_validator('raw_path', 'text_path')
    @classmethod
    def validate_paths(cls, value: str, info) -> str:
        return _relative_path(value, info.field_name)

    @field_validator('resolved_available_at')
    @classmethod
    def validate_resolved_available_at(cls, value: datetime) -> datetime:
        return _aware(value, 'resolved_available_at')

    @model_validator(mode='after')
    def validate_proofs(self) -> Self:
        proof_ids = tuple(proof.proof_id for proof in self.archive_proofs)
        if len(proof_ids) != len(set(proof_ids)):
            raise ValueError('document archive proof IDs must be unique')
        if proof_ids != tuple(sorted(proof_ids)):
            raise ValueError('document archive proofs must be sorted by proof_id')
        for proof in self.archive_proofs:
            if (proof.document_id, proof.version_id) != (self.document_id, self.version_id):
                raise ValueError('archive proof document/version does not match its document')
            if (proof.artifact_sha256, proof.artifact_bytes) != (self.raw_sha256, self.raw_bytes):
                raise ValueError('archive proof must bind the exact raw document bytes')
        selected = next((proof for proof in self.archive_proofs if proof.proof_id == self.selected_proof_id), None)
        if selected is None:
            raise ValueError('selected_proof_id must reference an archive proof')
        earliest = min(proof.witnessed_at for proof in self.archive_proofs)
        if selected.witnessed_at != earliest or self.resolved_available_at != earliest:
            raise ValueError('availability must use the earliest independent exact-byte witness')
        return self


class CorpusExclusion(StrictModel):
    source_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class LiteratureCorpusManifest(StrictModel):
    schema_version: Literal['vaxreplay.literature-corpus.v0.1'] = CORPUS_SCHEMA_VERSION
    corpus_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    decision_at: datetime
    discovery_protocol_path: str = Field(min_length=1)
    discovery_protocol_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    discovery_protocol_bytes: int = Field(gt=0)
    inventory_complete: Literal[True] = True
    fixture_only: bool = False
    documents: tuple[ArchivedLiteratureDocument, ...] = Field(min_length=1)
    exclusions: tuple[CorpusExclusion, ...] = ()

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return _aware(value, 'decision_at')

    @field_validator('discovery_protocol_path')
    @classmethod
    def validate_discovery_protocol_path(cls, value: str) -> str:
        return _relative_path(value, 'discovery_protocol_path')

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        document_ids = tuple(document.document_id for document in self.documents)
        if document_ids != tuple(sorted(document_ids)) or len(document_ids) != len(set(document_ids)):
            raise ValueError('corpus documents must have unique document IDs in sorted order')
        if any(document.resolved_available_at > self.decision_at for document in self.documents):
            raise ValueError('corpus documents must be independently available by decision_at')
        exclusion_ids = tuple(exclusion.source_id for exclusion in self.exclusions)
        if exclusion_ids != tuple(sorted(exclusion_ids)) or len(exclusion_ids) != len(set(exclusion_ids)):
            raise ValueError('corpus exclusions must have unique source IDs in sorted order')
        if set(document_ids) & set(exclusion_ids):
            raise ValueError('a source cannot be both included and excluded')
        if (
            any(proof.fixture_only for document in self.documents for proof in document.archive_proofs)
            != self.fixture_only
        ):
            raise ValueError('fixture-only archive proofs require a fixture-only corpus')
        return self


class LiteratureByteSpan(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=12, max_length=400)

    @model_validator(mode='after')
    def validate_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError('span end must be greater than span start')
        return self


class PanelCandidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_version_id: str = Field(min_length=1)
    source_row_id: str = Field(min_length=1)
    source_span: LiteratureByteSpan


class PanelExclusion(StrictModel):
    source_document_id: str = Field(min_length=1)
    source_version_id: str = Field(min_length=1)
    source_row_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    source_span: LiteratureByteSpan


class CandidatePanelManifest(StrictModel):
    schema_version: Literal['vaxreplay.literature-panel.v0.1'] = PANEL_SCHEMA_VERSION
    panel_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    candidate_set_definition_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    outcome_sources_permitted: Literal[False] = False
    complete: Literal[True] = True
    matching_source_row_ids: tuple[str, ...] = Field(min_length=2)
    included_source_row_ids: tuple[str, ...] = Field(min_length=2)
    candidates: tuple[PanelCandidate, ...] = Field(min_length=2)
    exclusions: tuple[PanelExclusion, ...] = ()

    @model_validator(mode='after')
    def validate_panel(self) -> Self:
        matching = self.matching_source_row_ids
        included = self.included_source_row_ids
        if matching != tuple(sorted(matching)) or len(matching) != len(set(matching)):
            raise ValueError('matching source rows must be unique and sorted')
        if included != tuple(sorted(included)) or len(included) != len(set(included)):
            raise ValueError('included source rows must be unique and sorted')
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        candidate_rows = tuple(candidate.source_row_id for candidate in self.candidates)
        if candidate_ids != tuple(sorted(candidate_ids)) or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError('panel candidates must have unique candidate IDs in sorted order')
        if tuple(sorted(candidate_rows)) != included or len(candidate_rows) != len(set(candidate_rows)):
            raise ValueError('each included source row must produce exactly one panel candidate')
        excluded_rows = tuple(exclusion.source_row_id for exclusion in self.exclusions)
        if excluded_rows != tuple(sorted(excluded_rows)) or len(excluded_rows) != len(set(excluded_rows)):
            raise ValueError('panel exclusions must have unique source rows in sorted order')
        if set(included) & set(excluded_rows) or tuple(sorted((*included, *excluded_rows))) != matching:
            raise ValueError('included and excluded rows must exactly partition all matching rows')
        return self


class LiteratureClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version_id: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    candidate_id: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    stance: EvidenceStance
    span: LiteratureByteSpan


class ExtractionRunManifest(StrictModel):
    schema_version: Literal['vaxreplay.literature-extraction.v0.1'] = EXTRACTION_SCHEMA_VERSION
    extraction_id: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    panel_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    extractor_kind: ExtractionKind
    extractor_id: str = Field(min_length=1)
    extractor_code_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    prompt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    config_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    runner_receipt_path: str = Field(min_length=1)
    runner_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    runner_receipt_bytes: int = Field(gt=0)
    network_allowed: Literal[False] = False
    may_select_candidates: Literal[False] = False
    label_blind: Literal[True] = True
    outcome_namespace_mounted: Literal[False] = False
    claims: tuple[LiteratureClaim, ...] = Field(min_length=1)

    @field_validator('runner_receipt_path')
    @classmethod
    def validate_runner_receipt_path(cls, value: str) -> str:
        return _relative_path(value, 'runner_receipt_path')

    @model_validator(mode='after')
    def validate_claim_order(self) -> Self:
        keys = tuple(
            (claim.document_id, claim.candidate_id, claim.dimension, claim.span.start) for claim in self.claims
        )
        if keys != tuple(sorted(keys)) or len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError('extracted claims must be uniquely identified and canonically ordered')
        if self.extractor_kind == ExtractionKind.LLM and self.prompt_sha256 == '0' * 64:
            raise ValueError('LLM extraction requires a real pinned prompt commitment')
        return self


def literature_model_sha256(value: StrictModel) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def literature_decision_content_sha256(
    *,
    episode_id: str,
    decision_at: datetime,
    corpus: LiteratureCorpusManifest,
    panel: CandidatePanelManifest,
    extraction: ExtractionRunManifest,
    evaluation_config: LiteratureEvaluationConfig,
    candidate_set_definition_sha256: str,
    evidence_acquisition_spec_sha256: str,
    outcome_adjudication_spec_sha256: str,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                'episode_id': episode_id,
                'decision_at': decision_at.isoformat(),
                'corpus_sha256': literature_model_sha256(corpus),
                'panel_sha256': literature_model_sha256(panel),
                'extraction_sha256': literature_model_sha256(extraction),
                'evaluation_config_sha256': literature_model_sha256(evaluation_config),
                'candidate_set_definition_sha256': candidate_set_definition_sha256,
                'evidence_acquisition_spec_sha256': evidence_acquisition_spec_sha256,
                'outcome_adjudication_spec_sha256': outcome_adjudication_spec_sha256,
            }
        )
    ).hexdigest()


class DecisionSeal(StrictModel):
    schema_version: Literal['vaxreplay.literature-decision-seal.v0.1'] = DECISION_SEAL_SCHEMA_VERSION
    seal_id: str = Field(min_length=1)
    decision_content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    witnessed_at: datetime
    authority_type: TemporalReceiptAuthority
    authority_id: str = Field(min_length=1)
    proof_path: str = Field(min_length=1)
    proof_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)
    fixture_only: bool = False

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'witnessed_at')

    @field_validator('proof_path')
    @classmethod
    def validate_proof_path(cls, value: str) -> str:
        return _relative_path(value, 'proof_path')

    @model_validator(mode='after')
    def validate_authority(self) -> Self:
        if self.authority_type not in _INDEPENDENT_AUTHORITIES:
            raise ValueError('decision seals require an independent authority')
        return self


class LiteratureDecisionPackage(StrictModel):
    schema_version: Literal['vaxreplay.literature-decision-package.v0.1'] = DECISION_PACKAGE_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    decision_at: datetime
    corpus: LiteratureCorpusManifest
    panel: CandidatePanelManifest
    extraction: ExtractionRunManifest
    evaluation_config: LiteratureEvaluationConfig
    candidate_set_definition_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    evidence_acquisition_spec_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    outcome_adjudication_spec_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    seal: DecisionSeal

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return _aware(value, 'decision_at')

    @model_validator(mode='after')
    def validate_package(self) -> Self:
        if (
            self.corpus.episode_id != self.episode_id
            or self.panel.episode_id != self.episode_id
            or self.evaluation_config.episode_id != self.episode_id
        ):
            raise ValueError('corpus, panel, and evaluation config must match the decision package episode')
        if self.corpus.decision_at != self.decision_at:
            raise ValueError('corpus and decision package cutoffs must match')
        corpus_sha = literature_model_sha256(self.corpus)
        panel_sha = literature_model_sha256(self.panel)
        if self.extraction.corpus_sha256 != corpus_sha or self.extraction.panel_sha256 != panel_sha:
            raise ValueError('extraction input commitments do not match the corpus and panel')
        if self.panel.candidate_set_definition_sha256 != self.candidate_set_definition_sha256:
            raise ValueError('panel does not use the committed candidate-set definition')
        if self.evaluation_config.synthetic != self.corpus.fixture_only:
            raise ValueError('evaluation config synthetic flag must match the corpus fixture flag')
        if self.evaluation_config.portfolio_size >= len(self.panel.candidates):
            raise ValueError('evaluation portfolio size must be smaller than the frozen panel')
        documents = {document.document_id: document for document in self.corpus.documents}
        candidates = {candidate.candidate_id for candidate in self.panel.candidates}
        for candidate in self.panel.candidates:
            document = documents.get(candidate.source_document_id)
            if document is None or document.version_id != candidate.source_version_id:
                raise ValueError('panel candidate refers to an unadmitted document version')
        for exclusion in self.panel.exclusions:
            document = documents.get(exclusion.source_document_id)
            if document is None or document.version_id != exclusion.source_version_id:
                raise ValueError('panel exclusion refers to an unadmitted document version')
        for claim in self.extraction.claims:
            document = documents.get(claim.document_id)
            if (
                document is None
                or document.version_id != claim.document_version_id
                or document.text_sha256 != claim.text_sha256
            ):
                raise ValueError('claim refers to an unadmitted document text view')
            if claim.candidate_id not in candidates:
                raise ValueError('claim refers to a candidate outside the frozen panel')
        unknown_dimensions = sorted(
            {claim.dimension for claim in self.extraction.claims} - set(self.evaluation_config.required_dimensions)
        )
        if unknown_dimensions:
            raise ValueError(f'extraction contains unregistered dimensions: {unknown_dimensions}')
        expected_content_sha = literature_decision_content_sha256(
            episode_id=self.episode_id,
            decision_at=self.decision_at,
            corpus=self.corpus,
            panel=self.panel,
            extraction=self.extraction,
            evaluation_config=self.evaluation_config,
            candidate_set_definition_sha256=self.candidate_set_definition_sha256,
            evidence_acquisition_spec_sha256=self.evidence_acquisition_spec_sha256,
            outcome_adjudication_spec_sha256=self.outcome_adjudication_spec_sha256,
        )
        if self.decision_content_sha256 != expected_content_sha:
            raise ValueError('decision content hash does not match the frozen decision artifacts')
        if self.seal.decision_content_sha256 != expected_content_sha:
            raise ValueError('decision seal does not bind the frozen decision content')
        if self.seal.fixture_only != self.corpus.fixture_only:
            raise ValueError('fixture-only corpus and decision seal flags must agree')
        return self


class OutcomeJoinRecord(StrictModel):
    candidate_id: str = Field(min_length=1)
    status: OutcomeJoinStatus
    outcome: Literal[0, 1] | None = None
    candidate_utility: float | None = Field(default=None, ge=0.0, le=1.0)
    relevance_grade: int | None = Field(default=None, ge=0, le=4)
    censor_reason: str | None = None
    source_record_ids: tuple[str, ...] = ()

    @model_validator(mode='after')
    def validate_status(self) -> Self:
        if self.status == OutcomeJoinStatus.OBSERVED:
            if self.outcome is None or self.candidate_utility is None or self.relevance_grade is None:
                raise ValueError('observed joins require outcome, utility, and relevance grade')
            if self.censor_reason is not None or not self.source_record_ids:
                raise ValueError('observed joins require source rows and cannot declare censoring')
        elif any(value is not None for value in (self.outcome, self.candidate_utility, self.relevance_grade)):
            raise ValueError('non-observed joins cannot contain scored values')
        elif not self.censor_reason:
            raise ValueError('non-observed joins require a reason')
        if self.source_record_ids != tuple(sorted(self.source_record_ids)):
            raise ValueError('outcome source record IDs must be sorted')
        return self


class LiteratureOutcomeJoinAudit(StrictModel):
    schema_version: Literal['vaxreplay.literature-outcome-audit.v0.1'] = OUTCOME_AUDIT_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    decision_content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_join_started_at: datetime
    panel_candidate_ids: tuple[str, ...] = Field(min_length=2)
    records: tuple[OutcomeJoinRecord, ...] = Field(min_length=2)
    unmatched_outcome_record_ids: tuple[str, ...] = ()

    @field_validator('label_join_started_at')
    @classmethod
    def validate_label_join_started_at(cls, value: datetime) -> datetime:
        return _aware(value, 'label_join_started_at')

    @model_validator(mode='after')
    def validate_records(self) -> Self:
        candidate_ids = tuple(record.candidate_id for record in self.records)
        if candidate_ids != tuple(sorted(candidate_ids)) or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError('outcome joins must cover unique candidates in sorted order')
        if self.panel_candidate_ids != tuple(sorted(self.panel_candidate_ids)):
            raise ValueError('outcome audit panel candidate IDs must be sorted')
        if candidate_ids != self.panel_candidate_ids:
            raise ValueError('outcome joins must retain every frozen panel candidate exactly once')
        if self.unmatched_outcome_record_ids != tuple(sorted(self.unmatched_outcome_record_ids)):
            raise ValueError('unmatched outcome record IDs must be sorted')
        return self


class LiteratureOutcomePackage(StrictModel):
    schema_version: Literal['vaxreplay.literature-outcome-package.v0.1'] = OUTCOME_PACKAGE_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    decision_package_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    raw_outcome_path: str = Field(min_length=1)
    raw_outcome_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    raw_outcome_bytes: int = Field(gt=0)
    source_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    availability_proof: ArchiveProof
    outcome_available_at: datetime
    target_id: str = Field(min_length=1)
    horizon_days: int = Field(gt=0)
    source_url: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    join_audit: LiteratureOutcomeJoinAudit

    @field_validator('raw_outcome_path')
    @classmethod
    def validate_raw_outcome_path(cls, value: str) -> str:
        return _relative_path(value, 'raw_outcome_path')

    @field_validator('outcome_available_at')
    @classmethod
    def validate_outcome_available_at(cls, value: datetime) -> datetime:
        return _aware(value, 'outcome_available_at')

    @model_validator(mode='after')
    def validate_ids(self) -> Self:
        if self.join_audit.episode_id != self.episode_id:
            raise ValueError('outcome join audit must match the outcome package episode')
        proof = self.availability_proof
        if proof.document_id != self.source_id or proof.version_id != self.version_id:
            raise ValueError('outcome availability proof must identify the exact outcome source version')
        if proof.artifact_sha256 != self.raw_outcome_sha256 or proof.artifact_bytes != self.raw_outcome_bytes:
            raise ValueError('outcome availability proof must bind the exact raw outcome bytes')
        if proof.witnessed_at != self.outcome_available_at:
            raise ValueError('outcome availability must equal its independently verified exact-byte witness')
        return self


def literature_decision_package_sha256(package: LiteratureDecisionPackage) -> str:
    return literature_model_sha256(package)


def literature_outcome_package_sha256(package: LiteratureOutcomePackage) -> str:
    return literature_model_sha256(package)


def validate_outcome_package_against_decision(
    package: LiteratureOutcomePackage,
    decision: LiteratureDecisionPackage,
) -> None:
    """Validate the one-way transition from sealed decision state to later labels."""

    if package.episode_id != decision.episode_id:
        raise ValueError('outcome and decision packages have different episode IDs')
    if package.decision_package_sha256 != literature_decision_package_sha256(decision):
        raise ValueError('outcome package does not bind the sealed decision package')
    if package.join_audit.decision_content_sha256 != decision.decision_content_sha256:
        raise ValueError('outcome join audit does not bind the decision content')
    if package.availability_proof.fixture_only != decision.corpus.fixture_only:
        raise ValueError('outcome proof and decision corpus fixture flags must agree')
    if package.join_audit.label_join_started_at <= decision.seal.witnessed_at:
        raise ValueError('later labels may be joined only after the decision package is sealed')
    if package.join_audit.label_join_started_at < package.outcome_available_at:
        raise ValueError('label joining cannot begin before the declared outcome availability')
    maturity_at = decision.decision_at + timedelta(days=package.horizon_days)
    if package.outcome_available_at < maturity_at:
        raise ValueError('outcomes cannot be available before the forecast horizon matures')
    panel_ids = tuple(candidate.candidate_id for candidate in decision.panel.candidates)
    joined_ids = tuple(record.candidate_id for record in package.join_audit.records)
    if joined_ids != panel_ids:
        raise ValueError('outcome join must retain every frozen panel candidate exactly once')


class LiteratureSourceFileBinding(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(gt=0)

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _relative_path(value, 'path')


class LiteratureSourceAudit(StrictModel):
    """Private, release-bound source inventory for replay re-verification."""

    schema_version: Literal['vaxreplay.literature-source-audit.v0.1'] = SOURCE_AUDIT_SCHEMA_VERSION
    adapter_id: Literal['vaxreplay.literature.v0.1'] = 'vaxreplay.literature.v0.1'
    episode_id: str = Field(min_length=1)
    episode_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    fixture_only: bool = False
    decision_package: LiteratureDecisionPackage
    outcome_package: LiteratureOutcomePackage
    files: tuple[LiteratureSourceFileBinding, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_audit(self) -> Self:
        if self.episode_id != self.decision_package.episode_id or self.episode_id != self.outcome_package.episode_id:
            raise ValueError('source audit episode IDs must agree')
        if self.fixture_only != self.decision_package.corpus.fixture_only:
            raise ValueError('source audit fixture flag must match the decision package')
        paths = tuple(binding.path for binding in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('source audit files must have unique paths in sorted order')
        expected_paths = {
            f'decision/{self.decision_package.corpus.discovery_protocol_path}',
            f'decision/{self.decision_package.seal.proof_path}',
            f'decision/{self.decision_package.extraction.runner_receipt_path}',
            *(f'decision/{document.raw_path}' for document in self.decision_package.corpus.documents),
            *(f'decision/{document.text_path}' for document in self.decision_package.corpus.documents),
            *(
                f'decision/{proof.proof_path}'
                for document in self.decision_package.corpus.documents
                for proof in document.archive_proofs
            ),
            f'outcome/{self.outcome_package.raw_outcome_path}',
            f'outcome/{self.outcome_package.availability_proof.proof_path}',
        }
        if set(paths) != expected_paths:
            raise ValueError('source audit file inventory must exactly match the replay contract')
        validate_outcome_package_against_decision(self.outcome_package, self.decision_package)
        return self
