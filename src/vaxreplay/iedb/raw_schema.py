"""Strict contracts for pinned IEDB snapshots and replay episode specifications."""

from __future__ import annotations

import enum
import hashlib
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, Self
from urllib.parse import parse_qs, urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.case_schema import RANKING_REWARD_VERSION, Split, StrictModel

IEDB_SNAPSHOT_SCHEMA_VERSION = 'vaxreplay.iedb_snapshot.v0.1'
IEDB_EPISODE_SPEC_SCHEMA_VERSION = 'vaxreplay.iedb_episode_spec.v0.1'
IEDB_AUDIT_SCHEMA_VERSION = 'vaxreplay.iedb_audit.v0.1'
IEDB_ADAPTER_ID = 'iedb-iq-api-cohort-v0.1'
IEDB_BINARY_RANKING_RUBRIC_VERSION = 'iedb-qualitative-binary-v1'


class IedbEndpoint(str, enum.Enum):
    TCELL = 'tcell_search'
    BCELL = 'bcell_search'
    MHC = 'mhc_search'


class IedbTableFormat(str, enum.Enum):
    JSON = 'json'
    JSONL = 'jsonl'


class QualitativePolarity(str, enum.Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    UNKNOWN = 'unknown'


class IedbApiMetric(StrictModel):
    search_table_name: str = Field(min_length=1)
    record_count: int = Field(ge=0)
    creation_date: str = Field(min_length=1)

    @field_validator('creation_date')
    @classmethod
    def validate_creation_date(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as error:
            raise ValueError('api_metrics creation_date must be ISO-8601') from error
        return value


class IedbSnapshotTable(StrictModel):
    endpoint: IedbEndpoint
    relative_path: str = Field(min_length=1)
    format: IedbTableFormat
    source_url: str = Field(min_length=1)
    order_by: str = Field(min_length=1)
    request_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    content_range: str = Field(pattern=r'^(?:\d+-\d+/\d+|\*/0)$')
    complete_capture: Literal[True] = True
    source_build_at: datetime
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    columns_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @field_validator('relative_path')
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('relative_path must stay inside the snapshot directory')
        return value

    @field_validator('source_build_at')
    @classmethod
    def validate_source_build_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('source_build_at must include a UTC offset')
        return value

    @model_validator(mode='after')
    def validate_capture_receipt(self) -> Self:
        expected_order = {
            IedbEndpoint.TCELL: 'tcell_id',
            IedbEndpoint.BCELL: 'bcell_id',
            IedbEndpoint.MHC: 'elution_id',
        }[self.endpoint]
        if self.order_by != expected_order:
            raise ValueError(f'{self.endpoint.value} captures must use unique order {expected_order}')
        if hashlib.sha256(self.source_url.encode('utf-8')).hexdigest() != self.request_sha256:
            raise ValueError('request_sha256 must hash source_url exactly')
        query = parse_qs(urlsplit(self.source_url).query)
        if query.get('order') != [self.order_by]:
            raise ValueError('source_url must contain the declared stable order parameter')
        if self.row_count > 10_000:
            raise ValueError('current snapshot tables support one complete IQ-API page of at most 10,000 rows')
        if self.row_count == 0:
            if self.content_range != '*/0':
                raise ValueError('empty complete captures require content_range */0')
            return self
        match = re.fullmatch(r'(\d+)-(\d+)/(\d+)', self.content_range)
        if match is None:
            raise ValueError('non-empty captures require a concrete Content-Range')
        start, end, total = (int(value) for value in match.groups())
        if start != 0 or end - start + 1 != self.row_count or total != self.row_count:
            raise ValueError('Content-Range must prove the file is the complete ordered query result')
        return self


class IedbSnapshotManifest(StrictModel):
    schema_version: Literal['vaxreplay.iedb_snapshot.v0.1'] = IEDB_SNAPSHOT_SCHEMA_VERSION
    snapshot_id: str = Field(min_length=1)
    synthetic: bool
    source_build_at: datetime
    retrieved_at: datetime
    source_base_url: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    terms_url: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    third_party_rights_reviewed: bool
    api_metrics_relative_path: str = Field(min_length=1)
    api_metrics_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    api_metrics_after_relative_path: str = Field(min_length=1)
    api_metrics_after_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    api_metrics_naive_timezone: Literal['UTC'] = 'UTC'
    tables: list[IedbSnapshotTable] = Field(min_length=1)

    @field_validator('api_metrics_relative_path', 'api_metrics_after_relative_path')
    @classmethod
    def validate_api_metrics_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('api_metrics_relative_path must stay inside the snapshot directory')
        return value

    @field_validator('source_build_at', 'retrieved_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('snapshot timestamps must include a UTC offset')
        return value

    @model_validator(mode='after')
    def validate_snapshot(self) -> Self:
        if self.source_build_at > self.retrieved_at:
            raise ValueError('source_build_at cannot be after retrieved_at')
        endpoints = [table.endpoint for table in self.tables]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError('snapshot table endpoints must be unique')
        if any(table.source_build_at != self.source_build_at for table in self.tables):
            raise ValueError('all snapshot tables must come from the same source build')
        base_url = urlsplit(self.source_base_url)
        if (
            base_url.scheme.lower() != 'https'
            or not base_url.netloc
            or base_url.username is not None
            or base_url.password is not None
            or base_url.path not in ('', '/')
            or base_url.query
            or base_url.fragment
        ):
            raise ValueError('source_base_url must be an HTTPS origin without a path, query, or fragment')
        for table in self.tables:
            table_url = urlsplit(table.source_url)
            if (
                table_url.scheme.lower() != base_url.scheme.lower()
                or table_url.netloc.lower() != base_url.netloc.lower()
            ):
                raise ValueError('snapshot table URLs must use the exact source_base_url origin')
            if table_url.path != f'/{table.endpoint.value}' or table_url.fragment:
                raise ValueError('snapshot table URL path must exactly match its declared endpoint')
        if not self.synthetic:
            if self.source_base_url != 'https://query-api.iedb.org':
                raise ValueError('real IEDB snapshots must come from the official IEDB IQ-API')
            if self.license_id != 'CC-BY-4.0':
                raise ValueError('real IEDB snapshots must preserve the declared CC-BY-4.0 license')
            if not self.third_party_rights_reviewed:
                raise ValueError('real snapshots require explicit third-party-rights review')
            if 'IEDB' not in self.citation:
                raise ValueError('real snapshot citation must explicitly attribute IEDB')
        return self


class IedbCandidateSpec(StrictModel):
    candidate_id: str = Field(pattern=r'^cand-[a-z0-9][a-z0-9-]*$')
    structure_iri: str = Field(min_length=1)


class IedbEpisodeSpec(StrictModel):
    schema_version: Literal['vaxreplay.iedb_episode_spec.v0.1'] = IEDB_EPISODE_SPEC_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    lineage_group_id: str = Field(min_length=1)
    synthetic: bool
    split: Split
    decision_at: datetime
    outcome_as_of: datetime
    portfolio_size: int = Field(gt=0)
    candidates: list[IedbCandidateSpec] = Field(min_length=2)
    evidence_endpoints: list[IedbEndpoint] = Field(min_length=1)
    label_endpoint: IedbEndpoint
    label_reference_iri: str = Field(min_length=1)
    label_assay_iri: str = Field(min_length=1)
    label_mhc_restriction: str | None = None
    label_host_organism_iri: str = Field(min_length=1)
    label_source_organism_iri: str = Field(min_length=1)
    min_positive_candidates: int = Field(ge=1)
    min_negative_candidates: int = Field(ge=1)
    reward_version: Literal['v1.0'] = RANKING_REWARD_VERSION
    ranking_rubric_version: Literal['iedb-qualitative-binary-v1']
    require_visible_evidence_for_all: Literal[True] = True

    @field_validator('decision_at', 'outcome_as_of')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('episode timestamps must include a UTC offset')
        return value

    @model_validator(mode='after')
    def validate_episode(self) -> Self:
        if self.outcome_as_of <= self.decision_at:
            raise ValueError('outcome_as_of must be after decision_at')
        horizon_seconds = (self.outcome_as_of - self.decision_at).total_seconds()
        if horizon_seconds % 86_400 != 0:
            raise ValueError('decision_at to outcome_as_of must be a whole number of days')
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        structure_iris = [candidate.structure_iri for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError('candidate IDs must be unique')
        if len(structure_iris) != len(set(structure_iris)):
            raise ValueError('candidate structure IRIs must be unique')
        if self.portfolio_size > len(self.candidates):
            raise ValueError('portfolio_size cannot exceed the candidate count')
        if len(self.evidence_endpoints) != len(set(self.evidence_endpoints)):
            raise ValueError('evidence_endpoints must be unique')
        if self.label_endpoint not in self.evidence_endpoints:
            raise ValueError('label_endpoint must also be an evidence endpoint')
        if self.min_positive_candidates + self.min_negative_candidates > len(self.candidates):
            raise ValueError('minimum class counts cannot exceed the candidate count')
        if not self.synthetic:
            if len(self.candidates) < 10:
                raise ValueError('real cohort replay episodes require at least 10 candidates')
            if self.min_positive_candidates < 2 or self.min_negative_candidates < 2:
                raise ValueError('real cohort replay episodes require at least two candidates per class')
        return self


class NormalizedIedbAssay(StrictModel):
    endpoint: IedbEndpoint
    assay_iri: str = Field(min_length=1)
    structure_iri: str = Field(min_length=1)
    qualitative_measure: str | None = None
    polarity: QualitativePolarity
    assay_iris: list[str] = Field(default_factory=list)
    assay_names: list[str] = Field(default_factory=list)
    reference_iri: str = Field(min_length=1)
    pubmed_id: str | None = None
    reference_titles: list[str] = Field(default_factory=list)
    reference_dates: list[str] = Field(default_factory=list)
    parent_source_antigen_iri: str | None = None
    parent_source_antigen_name: str | None = None
    curated_source_antigen_name: str | None = None
    region_start: int | None = None
    region_end: int | None = None
    source_organism_iri: str | None = None
    source_organism_name: str | None = None
    host_organism_iri: str | None = None
    host_organism_name: str | None = None
    mhc_allele_iri: str | None = None
    mhc_allele_name: str | None = None


class IedbAuditSource(StrictModel):
    endpoint: IedbEndpoint
    assay_iri: str = Field(min_length=1)
    normalized_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    first_seen_at: datetime
    first_seen_snapshot_id: str = Field(min_length=1)
    reference_iri: str = Field(min_length=1)
    pubmed_id: str | None = None

    @field_validator('first_seen_at')
    @classmethod
    def validate_first_seen_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('first_seen_at must include a UTC offset')
        return value


class IedbOutcomeAudit(StrictModel):
    candidate_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    outcome: Literal[0, 1]
    sources: list[IedbAuditSource] = Field(min_length=1)


class IedbCandidateMap(StrictModel):
    candidate_id: str = Field(min_length=1)
    structure_iri: str = Field(min_length=1)


class IedbEvidenceAudit(StrictModel):
    candidate_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    source: IedbAuditSource


class IedbPrivateAudit(StrictModel):
    schema_version: Literal['vaxreplay.iedb_audit.v0.1'] = IEDB_AUDIT_SCHEMA_VERSION
    adapter_id: Literal['iedb-iq-api-cohort-v0.1'] = IEDB_ADAPTER_ID
    episode_id: str = Field(min_length=1)
    candidate_map: list[IedbCandidateMap] = Field(min_length=2)
    evidence: list[IedbEvidenceAudit] = Field(min_length=1)
    outcomes: list[IedbOutcomeAudit] = Field(min_length=2)
