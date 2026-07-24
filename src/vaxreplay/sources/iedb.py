"""Production IEDB IQ-API source verification for witnessed promotions.

This module is deliberately offline.  It consumes exact bodies and HTTPS receipts
already retained by the operational collector, enforces an independently committed
IQ-API layout, reconstructs :mod:`vaxreplay.iedb.live_capture`, and emits the exact
source-record inventory required by the promotion bridge.

The functions here are worker cores, not an execution boundary.  Tier A callers must
run them through the hermetic execution service and persist its signed receipt.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, quote, urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord, SourceType, StrictModel
from vaxreplay.iedb.adapter import IedbAdapterError, normalize_assay
from vaxreplay.iedb.live_capture import (
    IedbApiCaptureSpec,
    IedbApiExchangeReceipt,
    IedbApiPageSpec,
    IedbHttpHeader,
    build_api_capture,
)
from vaxreplay.iedb.raw_schema import IedbApiMetric, IedbEndpoint, NormalizedIedbAssay
from vaxreplay.operations.http_capture import HttpsCaptureReceipt
from vaxreplay.operations.promotion import (
    AdapterRunResult,
    AdapterSourceInput,
    SourceVerificationInput,
    SourceVerifierRunResult,
)
from vaxreplay.operations.promotion_schema import (
    AuthoritativeReleaseBasis,
    AuthoritativeSourceRelease,
    NormalizedRecordReference,
    SourceRecordBinding,
    SourceRecordDisposition,
    SourceVerificationResult,
    SourceVerifierIdentity,
)
from vaxreplay.operations.schema import aware_utc

IEDB_PROMOTION_LAYOUT_SCHEMA_VERSION = 'vaxreplay.iedb-promotion-layout.v0.1'
IEDB_SOURCE_VERIFIER_POLICY_SCHEMA_VERSION = 'vaxreplay.iedb-source-verifier-policy.v0.2'
IEDB_SOURCE_VERIFIER_ID = 'iedb-iq-api-offline-verifier'
IEDB_SOURCE_VERIFIER_VERSION = 'v0.2'
IEDB_ANTIGEN_ADAPTER_POLICY_SCHEMA_VERSION = 'vaxreplay.iedb-antigen-adapter-policy.v0.2'
IEDB_ANTIGEN_CANDIDATE_MAP_SCHEMA_VERSION = 'vaxreplay.iedb-antigen-candidate-map.v0.1'
IEDB_ANTIGEN_ADAPTER_ID = 'iedb-antigen-target-universe-adapter'
IEDB_ANTIGEN_ADAPTER_VERSION = 'v0.1'
IEDB_ANTIGEN_ADAPTER_EXCLUSION_REASON_CODES = (
    'insufficient_cutoff_safe_fields',
    'missing_parent_antigen_iri',
    'source_metadata_record',
    'source_organism_out_of_scope',
    'table_out_of_scope',
)

_ARTIFACT_ID_PATTERN = r'^[a-z][a-z0-9._-]{0,109}$'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_OFFICIAL_ORIGIN = 'https://query-api.iedb.org'
IEDB_TIER_A_ANTIGEN_TABLES = ('bcell_search', 'mhc_search', 'tcell_search')
_IEDB_TIER_A_ID_FIELDS = {
    'bcell_search': 'bcell_id',
    'mhc_search': 'elution_id',
    'tcell_search': 'tcell_id',
}
_IEDB_TIER_A_REQUIRED_ROW_FIELDS = frozenset(
    {
        'parent_source_antigen_iri',
        'reference_iri',
        'source_organism_iri',
        'structure_iri',
    }
)
_REQUEST_HEADERS = frozenset({'accept', 'accept-encoding', 'host', 'range', 'range-unit', 'user-agent'})
_RESPONSE_HEADERS = frozenset(
    {
        'cache-control',
        'content-encoding',
        'content-length',
        'content-range',
        'content-type',
        'date',
        'etag',
        'expires',
        'last-modified',
        'vary',
    }
)


class IedbProductionSourceError(ValueError):
    """A captured IEDB source cannot support the committed production claim."""


class IedbProductionAdapterError(ValueError):
    """An IEDB inventory cannot support the precommitted antigen universe."""


class IedbCapturedPage(StrictModel):
    """Bind a precommitted collector artifact ID to one complete IQ-API page."""

    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    page: IedbApiPageSpec

    @model_validator(mode='after')
    def require_json(self) -> Self:
        if self.page.data_format != 'json':
            raise ValueError('the production IEDB verifier accepts only strict JSON IQ-API pages')
        return self


class IedbPromotionLayout(StrictModel):
    """Exact source layout committed before the first scheduled capture."""

    schema_version: Literal['vaxreplay.iedb-promotion-layout.v0.1'] = IEDB_PROMOTION_LAYOUT_SCHEMA_VERSION
    metrics_url: Literal['https://query-api.iedb.org/api_metrics?order=search_table_name'] = (
        'https://query-api.iedb.org/api_metrics?order=search_table_name'
    )
    metrics_before_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    metrics_after_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    expected_table_names: tuple[str, ...] = Field(min_length=1, max_length=64)
    pages: tuple[IedbCapturedPage, ...] = Field(min_length=1, max_length=4096)

    @field_validator('expected_table_names')
    @classmethod
    def validate_table_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('expected_table_names must be sorted and unique')
        return value

    @model_validator(mode='after')
    def validate_layout(self) -> Self:
        artifact_ids = (
            self.metrics_before_artifact_id,
            self.metrics_after_artifact_id,
            *(item.artifact_id for item in self.pages),
        )
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError('every IEDB collector artifact ID must be unique')
        page_artifact_ids = tuple(item.artifact_id for item in self.pages)
        if page_artifact_ids != tuple(sorted(page_artifact_ids)):
            raise ValueError('IEDB page artifact IDs must use canonical sorted order')
        if not all(
            self.metrics_before_artifact_id < artifact_id < self.metrics_after_artifact_id
            for artifact_id in page_artifact_ids
        ):
            raise ValueError('IEDB artifact IDs must make static collection order metrics-before, pages, metrics-after')
        table_names = {item.page.table_name for item in self.pages}
        if table_names != set(self.expected_table_names):
            raise ValueError('IEDB pages must exactly cover expected_table_names')
        paths = tuple(
            path for item in self.pages for path in (item.page.data_relative_path, item.page.receipt_relative_path)
        )
        if len(paths) != len(set(paths)):
            raise ValueError('IEDB page data and receipt paths must be globally unique')
        return self

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        return (
            self.metrics_before_artifact_id,
            self.metrics_after_artifact_id,
            *(item.artifact_id for item in self.pages),
        )


class IedbSourceVerifierPolicy(StrictModel):
    """Reviewed source-authentication and completeness policy."""

    schema_version: Literal['vaxreplay.iedb-source-verifier-policy.v0.2'] = IEDB_SOURCE_VERIFIER_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    capture_id_prefix: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]{2,80}$')
    layout: IedbPromotionLayout
    release_evidence_table: str
    accepted_tls_versions: tuple[Literal['TLSv1.2', 'TLSv1.3'], ...] = ('TLSv1.2', 'TLSv1.3')
    require_tls_peer_certificate_sha256: Literal[True] = True
    scope_profile: Literal[
        'reviewed_declared_table_scope',
        'tier_a_antigen_all_assay_tables_v1',
    ] = 'reviewed_declared_table_scope'
    source_authentication: Literal['collector_receipt_consistent_with_system_ca_tls_to_official_origin'] = (
        'collector_receipt_consistent_with_system_ca_tls_to_official_origin'
    )
    completeness_semantics: Literal['stable_api_metrics_and_complete_ordered_pages'] = (
        'stable_api_metrics_and_complete_ordered_pages'
    )
    source_release_semantics: Literal['api_metrics_creation_date'] = 'api_metrics_creation_date'
    captures_per_verification: Literal[1] = 1

    @field_validator('accepted_tls_versions')
    @classmethod
    def validate_tls_versions(
        cls,
        value: tuple[Literal['TLSv1.2', 'TLSv1.3'], ...],
    ) -> tuple[Literal['TLSv1.2', 'TLSv1.3'], ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('accepted_tls_versions must be sorted and unique')
        return value

    @model_validator(mode='after')
    def validate_release_table(self) -> Self:
        if self.release_evidence_table not in self.layout.expected_table_names:
            raise ValueError('release_evidence_table must be inside the complete table scope')
        if self.scope_profile == 'tier_a_antigen_all_assay_tables_v1':
            if self.layout.expected_table_names != IEDB_TIER_A_ANTIGEN_TABLES:
                raise ValueError('the Tier A antigen profile requires all B-cell, MHC-ligand, and T-cell assay tables')
            for captured_page in self.layout.pages:
                page = captured_page.page
                if page.id_field != _IEDB_TIER_A_ID_FIELDS.get(page.table_name):
                    raise ValueError('the Tier A antigen profile requires each canonical assay identifier')
                if 'select' in dict(
                    parse_qsl(
                        urlsplit(page.request_url).query,
                        keep_blank_values=True,
                        strict_parsing=True,
                    )
                ):
                    raise ValueError('the Tier A antigen profile forbids projected IEDB table captures')
        return self


class IedbAntigenTablePolicy(StrictModel):
    """One IQ-API assay table eligible to nominate parent antigens."""

    endpoint: IedbEndpoint
    id_field: str = Field(pattern=r'^[a-z][a-z0-9_]*$')

    @model_validator(mode='after')
    def require_canonical_identifier(self) -> Self:
        expected = {
            IedbEndpoint.TCELL: 'tcell_id',
            IedbEndpoint.BCELL: 'bcell_id',
            IedbEndpoint.MHC: 'elution_id',
        }[self.endpoint]
        if self.id_field != expected:
            raise ValueError(f'{self.endpoint.value} antigen adaptation requires {expected}')
        return self

    @property
    def table_name(self) -> str:
        return self.endpoint.value


class IedbAntigenAdapterPolicy(StrictModel):
    """Precommitted open-universe antigen nomination policy.

    The universe is every distinct ``parent_source_antigen_iri`` in every eligible
    captured assay row for one of the explicitly scoped source organisms.  There is
    deliberately no score, count threshold, rank cutoff, or maximum candidate count.
    """

    schema_version: Literal['vaxreplay.iedb-antigen-adapter-policy.v0.2'] = IEDB_ANTIGEN_ADAPTER_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    episode_id: str = Field(min_length=1, max_length=1024)
    decision_at: datetime
    target_source_organism_iris: tuple[str, ...] = Field(min_length=1, max_length=4096)
    eligible_tables: tuple[IedbAntigenTablePolicy, ...] = Field(min_length=1, max_length=3)
    scope_profile: Literal[
        'reviewed_declared_table_scope',
        'tier_a_antigen_all_assay_tables_v1',
    ] = 'reviewed_declared_table_scope'
    minimum_candidate_count: int = Field(default=2, ge=2, le=1_000_000)
    license_id: str = Field(default='CC-BY-4.0', min_length=1, max_length=256)
    candidate_universe_semantics: Literal['all_distinct_parent_source_antigen_iris_without_ranking_or_count_filter'] = (
        'all_distinct_parent_source_antigen_iris_without_ranking_or_count_filter'
    )
    timestamp_semantics: Literal['captured_and_source_released_not_after_decision_at'] = (
        'captured_and_source_released_not_after_decision_at'
    )
    evidence_field_semantics: Literal['captured_iedb_assay_fields_only'] = 'captured_iedb_assay_fields_only'

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'IEDB antigen decision_at')

    @field_validator('target_source_organism_iris')
    @classmethod
    def validate_organisms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('target_source_organism_iris must be sorted and unique')
        if any(not item or item != item.strip() or len(item) > 2048 for item in value):
            raise ValueError('target source organism IRIs must be nonempty bounded exact strings')
        return value

    @model_validator(mode='after')
    def validate_tables(self) -> Self:
        names = tuple(item.table_name for item in self.eligible_tables)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError('eligible_tables must use sorted unique endpoint names')
        if self.scope_profile == 'tier_a_antigen_all_assay_tables_v1':
            if names != IEDB_TIER_A_ANTIGEN_TABLES:
                raise ValueError('the Tier A antigen adapter requires all three IEDB assay tables')
            if any(item.id_field != _IEDB_TIER_A_ID_FIELDS[item.table_name] for item in self.eligible_tables):
                raise ValueError('the Tier A antigen adapter requires canonical assay identifiers')
        return self


class IedbAntigenCandidateMapEntry(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=1024)
    parent_source_antigen_iri: str = Field(min_length=1, max_length=2048)
    observed_names: tuple[str, ...] = ()

    @field_validator('observed_names')
    @classmethod
    def validate_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('observed antigen names must be sorted and unique')
        return value


class IedbAntigenCandidateMap(StrictModel):
    """Auditable identity map for the complete generated candidate universe."""

    schema_version: Literal['vaxreplay.iedb-antigen-candidate-map.v0.1'] = IEDB_ANTIGEN_CANDIDATE_MAP_SCHEMA_VERSION
    policy_id: str
    episode_id: str
    candidate_universe_semantics: Literal['all_distinct_parent_source_antigen_iris_without_ranking_or_count_filter'] = (
        'all_distinct_parent_source_antigen_iris_without_ranking_or_count_filter'
    )
    candidates: tuple[IedbAntigenCandidateMapEntry, ...] = Field(min_length=2)

    @field_validator('candidates')
    @classmethod
    def validate_candidates(
        cls,
        value: tuple[IedbAntigenCandidateMapEntry, ...],
    ) -> tuple[IedbAntigenCandidateMapEntry, ...]:
        keys = tuple(item.candidate_id for item in value)
        iris = tuple(item.parent_source_antigen_iri for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)) or len(iris) != len(set(iris)):
            raise ValueError('candidate map must use sorted unique candidate IDs and antigen IRIs')
        return value


def iedb_source_verifier_policy_bytes(policy: IedbSourceVerifierPolicy) -> bytes:
    if not isinstance(policy, IedbSourceVerifierPolicy):
        raise TypeError('policy must be an IedbSourceVerifierPolicy')
    return canonical_json_bytes(policy)


def iedb_antigen_adapter_policy_bytes(policy: IedbAntigenAdapterPolicy) -> bytes:
    if not isinstance(policy, IedbAntigenAdapterPolicy):
        raise TypeError('policy must be an IedbAntigenAdapterPolicy')
    return canonical_json_bytes(policy)


def verify_tier_a_iedb_source(
    verifier_input: SourceVerificationInput,
    policy_bytes: bytes,
    *,
    implementation_sha256: str,
    execution_environment_sha256: str,
) -> SourceVerifierRunResult:
    """Production worker entrypoint for the fixed all-assay Tier A profile."""

    policy = _canonical_policy(policy_bytes)
    if policy.scope_profile != 'tier_a_antigen_all_assay_tables_v1':
        raise IedbProductionSourceError('the production IEDB worker requires tier_a_antigen_all_assay_tables_v1')
    return verify_iedb_source(
        verifier_input,
        policy_bytes,
        implementation_sha256=implementation_sha256,
        execution_environment_sha256=execution_environment_sha256,
    )


def adapt_tier_a_iedb_antigen_targets(
    inputs: tuple[AdapterSourceInput, ...],
    policy_bytes: bytes,
) -> AdapterRunResult:
    """Production adapter entrypoint with source/profile and schema cross-checks."""

    policy = _canonical_antigen_policy(policy_bytes)
    if policy.scope_profile != 'tier_a_antigen_all_assay_tables_v1':
        raise IedbProductionAdapterError('the production IEDB adapter requires tier_a_antigen_all_assay_tables_v1')
    if len(inputs) != 1:
        raise IedbProductionAdapterError('the production IEDB adapter requires exactly one source')
    source_input = inputs[0]
    if 'tier_a_antigen_all_assay_tables_v1' not in source_input.verification_result.result_codes:
        raise IedbProductionAdapterError(
            'the production IEDB adapter requires a matching all-assay source verification'
        )
    _preflight_tier_a_antigen_rows(source_input, policy)
    return adapt_iedb_antigen_targets(inputs, policy_bytes)


@dataclass(frozen=True)
class _EligibleIedbRow:
    source: SourceRecordBinding
    assay: NormalizedIedbAssay
    candidate_id: str
    evidence: EvidenceRecord


def adapt_iedb_antigen_targets(
    inputs: tuple[AdapterSourceInput, ...],
    policy_bytes: bytes,
) -> AdapterRunResult:
    """Enumerate the complete precommitted IEDB parent-antigen universe.

    Every verifier-enumerated source row receives exactly one disposition.  Eligible
    assay rows generate an evidence row and an edge to their parent-antigen candidate;
    exclusions use only the fixed reason-code vocabulary above.  Candidate generation
    never scores, samples, truncates, or chooses a top-N subset.
    """

    policy = _canonical_antigen_policy(policy_bytes)
    if len(inputs) != 1 or inputs[0].source_id != policy.source_id:
        raise IedbProductionAdapterError('IEDB antigen adaptation requires exactly its committed source input')
    source_input = inputs[0]
    verification = source_input.verification_result
    if verification.source_id != policy.source_id:
        raise IedbProductionAdapterError('IEDB verification result belongs to a different source')
    if verification.source_release.source_release_at > policy.decision_at:
        raise IedbProductionAdapterError('IEDB source release is after the precommitted decision cutoff')
    if any(capture.binding.captured_at > policy.decision_at for capture in source_input.captures):
        raise IedbProductionAdapterError('IEDB selected capture is after the precommitted decision cutoff')

    verified_record_bytes = _jsonl(source_input.verified_records)
    if (
        len(source_input.verified_records) != verification.verified_source_record_count
        or hashlib.sha256(verified_record_bytes).hexdigest() != verification.verified_source_record_inventory_sha256
    ):
        raise IedbProductionAdapterError('IEDB adapter input differs from its verified source inventory')
    if any(record.source_id != policy.source_id for record in source_input.verified_records):
        raise IedbProductionAdapterError('IEDB adapter input contains a foreign source record')

    raw_rows = _rebind_source_rows(source_input)
    table_policy = {item.table_name: item for item in policy.eligible_tables}
    eligible: list[_EligibleIedbRow] = []
    excluded: dict[tuple[str, str], str] = {}
    names_by_antigen: dict[str, set[str]] = defaultdict(set)
    candidate_by_antigen: dict[str, str] = {}
    antigen_by_candidate: dict[str, str] = {}
    available_at = verification.source_release.source_release_at

    for source in source_input.verified_records:
        key = (source.source_id, source.source_record_id)
        if source.source_record_id.startswith('api_metrics:'):
            excluded[key] = 'source_metadata_record'
            continue
        table_name, separator, identifier = source.source_record_id.partition(':')
        selected_table = table_policy.get(table_name)
        if not separator or selected_table is None:
            excluded[key] = 'table_out_of_scope'
            continue
        raw = raw_rows[key]
        raw_identifier = raw.get(selected_table.id_field)
        if (
            isinstance(raw_identifier, bool)
            or not isinstance(raw_identifier, int)
            or raw_identifier < 0
            or identifier != str(raw_identifier)
        ):
            raise IedbProductionAdapterError(
                f'{source.source_record_id} does not bind its committed stable table identifier'
            )
        try:
            assay = normalize_assay(selected_table.endpoint, raw)
        except IedbAdapterError:
            excluded[key] = 'insufficient_cutoff_safe_fields'
            continue
        if assay.source_organism_iri is None:
            excluded[key] = 'insufficient_cutoff_safe_fields'
            continue
        if assay.source_organism_iri not in policy.target_source_organism_iris:
            excluded[key] = 'source_organism_out_of_scope'
            continue
        antigen_iri = assay.parent_source_antigen_iri
        if antigen_iri is None:
            excluded[key] = 'missing_parent_antigen_iri'
            continue

        candidate_id = _antigen_candidate_id(antigen_iri)
        previous_iri = antigen_by_candidate.setdefault(candidate_id, antigen_iri)
        if previous_iri != antigen_iri:
            raise IedbProductionAdapterError('IEDB antigen candidate digest collision')
        candidate_by_antigen[antigen_iri] = candidate_id
        if assay.parent_source_antigen_name is not None:
            names_by_antigen[antigen_iri].add(assay.parent_source_antigen_name)
        evidence = _antigen_evidence_record(
            policy=policy,
            source=source,
            assay=assay,
            candidate_id=candidate_id,
            available_at=available_at,
        )
        eligible.append(_EligibleIedbRow(source, assay, candidate_id, evidence))

    if len(candidate_by_antigen) < policy.minimum_candidate_count:
        raise IedbProductionAdapterError(
            'complete IEDB antigen universe contains fewer candidates than the precommitted minimum'
        )

    candidates = tuple(
        sorted(
            (
                CandidateRecord(
                    episode_id=policy.episode_id,
                    candidate_id=candidate_id,
                    eligible=True,
                )
                for candidate_id in candidate_by_antigen.values()
            ),
            key=lambda item: (item.episode_id, item.candidate_id),
        )
    )
    evidence = tuple(
        sorted(
            (item.evidence for item in eligible),
            key=lambda item: (item.episode_id, item.evidence_id),
        )
    )
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise IedbProductionAdapterError('IEDB evidence digest collision')

    candidate_ref = {
        item.candidate_id: _normalized_reference(item.episode_id, item.candidate_id, item) for item in candidates
    }
    evidence_ref = {
        item.evidence_id: _normalized_reference(item.episode_id, item.evidence_id, item) for item in evidence
    }
    normalized_by_source = {
        (item.source.source_id, item.source.source_record_id): SourceRecordDisposition(
            source_id=item.source.source_id,
            source_record_id=item.source.source_record_id,
            source_record_sha256=item.source.source_record_sha256,
            source_artifact_sha256=item.source.source_artifact_sha256,
            disposition='normalized',
            candidate_record_refs=(candidate_ref[item.candidate_id],),
            evidence_record_refs=(evidence_ref[item.evidence.evidence_id],),
        )
        for item in eligible
    }
    dispositions = tuple(
        normalized_by_source.get(key)
        or SourceRecordDisposition(
            source_id=source.source_id,
            source_record_id=source.source_record_id,
            source_record_sha256=source.source_record_sha256,
            source_artifact_sha256=source.source_artifact_sha256,
            disposition='excluded',
            reason_code=excluded[key],
        )
        for source in source_input.verified_records
        for key in ((source.source_id, source.source_record_id),)
    )
    dispositions = tuple(sorted(dispositions, key=lambda item: (item.source_id, item.source_record_id)))

    candidate_map = IedbAntigenCandidateMap(
        policy_id=policy.policy_id,
        episode_id=policy.episode_id,
        candidates=tuple(
            sorted(
                (
                    IedbAntigenCandidateMapEntry(
                        candidate_id=candidate_id,
                        parent_source_antigen_iri=antigen_iri,
                        observed_names=tuple(sorted(names_by_antigen[antigen_iri])),
                    )
                    for antigen_iri, candidate_id in candidate_by_antigen.items()
                ),
                key=lambda item: item.candidate_id,
            )
        ),
    )
    return AdapterRunResult(
        candidate_records=_jsonl(candidates),
        evidence_records=_jsonl(evidence),
        dispositions=_jsonl(dispositions),
        auxiliary_outputs={'iedb-antigen-candidate-map': canonical_json_bytes(candidate_map)},
    )


def verify_iedb_source(
    verifier_input: SourceVerificationInput,
    policy_bytes: bytes,
    *,
    implementation_sha256: str,
    execution_environment_sha256: str,
) -> SourceVerifierRunResult:
    """Verify one exact declared IEDB scope.

    This lower-level function supports reviewed research scopes. Official worker images
    use :func:`verify_tier_a_iedb_source`, which requires the fixed all-assay profile.
    """

    policy = _canonical_policy(policy_bytes)
    _require_sha256(implementation_sha256, 'implementation_sha256')
    _require_sha256(execution_environment_sha256, 'execution_environment_sha256')
    if verifier_input.source_id != policy.source_id:
        raise IedbProductionSourceError('IEDB verifier input belongs to a different source')
    if len(verifier_input.captures) != policy.captures_per_verification:
        raise IedbProductionSourceError('IEDB production verification requires exactly one scheduled capture')
    capture = verifier_input.captures[0]
    if capture.binding.source_id != policy.source_id:
        raise IedbProductionSourceError('IEDB capture binding belongs to a different source')

    artifacts = {item.role: item for item in capture.artifacts}
    if len(artifacts) != len(capture.artifacts):
        raise IedbProductionSourceError('IEDB capture contains duplicate artifact roles')
    expected_body_roles = {f'body.{artifact_id}' for artifact_id in policy.layout.artifact_ids}
    expected_receipt_roles = {f'receipt.{artifact_id}' for artifact_id in policy.layout.artifact_ids}
    actual_body_roles = {role for role in artifacts if role.startswith('body.')}
    actual_receipt_roles = {role for role in artifacts if role.startswith('receipt.')}
    if actual_body_roles != expected_body_roles or actual_receipt_roles != expected_receipt_roles:
        raise IedbProductionSourceError('IEDB body/receipt inventory differs from the precommitted layout')

    https_receipt_by_artifact: dict[str, HttpsCaptureReceipt] = {}
    receipt_by_artifact: dict[str, IedbApiExchangeReceipt] = {}
    for artifact_id in policy.layout.artifact_ids:
        expected_url = _artifact_url(policy.layout, artifact_id)
        body = artifacts[f'body.{artifact_id}']
        receipt_artifact = artifacts[f'receipt.{artifact_id}']
        receipt = _verify_https_receipt(
            receipt_artifact.payload,
            body_sha256=body.sha256,
            body_bytes=body.byte_count,
            expected_url=expected_url,
            captured_at=capture.binding.captured_at,
            accepted_tls_versions=policy.accepted_tls_versions,
        )
        https_receipt_by_artifact[artifact_id] = receipt
        receipt_by_artifact[artifact_id] = _iedb_receipt(receipt)

    metrics_before_receipt = https_receipt_by_artifact[policy.layout.metrics_before_artifact_id]
    metrics_after_receipt = https_receipt_by_artifact[policy.layout.metrics_after_artifact_id]
    page_receipts = tuple(https_receipt_by_artifact[item.artifact_id] for item in policy.layout.pages)
    if (
        metrics_before_receipt.completed_at > min(item.started_at for item in page_receipts)
        or max(item.completed_at for item in page_receipts) > metrics_after_receipt.started_at
    ):
        raise IedbProductionSourceError(
            'IEDB receipt times do not prove metrics-before, pages, metrics-after bracketing'
        )

    with tempfile.TemporaryDirectory(prefix='vaxreplay-iedb-verify-') as temporary:
        root = Path(temporary)
        metrics_before_path = 'metrics-before.json'
        metrics_after_path = 'metrics-after.json'
        (root / metrics_before_path).write_bytes(artifacts[f'body.{policy.layout.metrics_before_artifact_id}'].payload)
        (root / metrics_after_path).write_bytes(artifacts[f'body.{policy.layout.metrics_after_artifact_id}'].payload)
        page_specs: list[IedbApiPageSpec] = []
        for captured_page in policy.layout.pages:
            page = captured_page.page
            data_path = root / page.data_relative_path
            receipt_path = root / page.receipt_relative_path
            data_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_bytes(artifacts[f'body.{captured_page.artifact_id}'].payload)
            receipt_path.write_bytes(canonical_json_bytes(receipt_by_artifact[captured_page.artifact_id]))
            page_specs.append(page)
        capture_spec = IedbApiCaptureSpec(
            capture_id=f'{policy.capture_id_prefix}-{capture.binding.attempt_id.removeprefix("attempt-")}',
            retrieved_at=capture.binding.captured_at,
            metrics_url=policy.layout.metrics_url,
            metrics_before_relative_path=metrics_before_path,
            metrics_after_relative_path=metrics_after_path,
            expected_table_names=policy.layout.expected_table_names,
            pages=tuple(page_specs),
        )
        built = build_api_capture(root, capture_spec)

    records, record_payloads = _source_records(policy, artifacts)
    record_bytes = b''.join(canonical_json_bytes(record) + b'\n' for record in records)
    release_record_id = f'api_metrics:{policy.release_evidence_table}'
    release_record = next((record for record in records if record.source_record_id == release_record_id), None)
    if release_record is None:
        raise IedbProductionSourceError('release-evidence api_metrics record is absent')
    metric = IedbApiMetric.model_validate(record_payloads[release_record_id])
    source_release_at = _metric_timestamp(metric)
    if source_release_at != built.manifest.source_build_at:
        raise IedbProductionSourceError('release-evidence metric differs from reconstructed source build')

    result = SourceVerificationResult(
        source_id=policy.source_id,
        verifier=SourceVerifierIdentity(
            verifier_id=IEDB_SOURCE_VERIFIER_ID,
            verifier_version=IEDB_SOURCE_VERIFIER_VERSION,
            implementation_sha256=implementation_sha256,
            execution_environment_sha256=execution_environment_sha256,
        ),
        verifier_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        verified_attempt_ids=(capture.binding.attempt_id,),
        source_release=AuthoritativeSourceRelease(
            source_release_at=source_release_at,
            basis=AuthoritativeReleaseBasis.SOURCE_API_PUBLICATION_FIELD,
            authority_locator=(
                f'{policy.layout.metrics_url}#search_table_name={quote(policy.release_evidence_table, safe="")}'
            ),
            authority_field='creation_date',
            evidence_attempt_id=capture.binding.attempt_id,
            evidence_role=f'body.{policy.layout.metrics_before_artifact_id}',
            evidence_sha256=release_record.source_artifact_sha256,
            evidence_source_record_id=release_record.source_record_id,
            evidence_source_record_sha256=release_record.source_record_sha256,
        ),
        verified_capture_inventory_sha256=verifier_input.capture_inventory_sha256,
        verified_source_record_inventory_sha256=hashlib.sha256(record_bytes).hexdigest(),
        verified_source_record_count=len(records),
        result_codes=(
            'collector_receipt_consistent_with_system_ca_tls_to_official_origin',
            'complete_ordered_iq_api_scope',
            'stable_api_metrics_build',
            *(
                ('tier_a_antigen_all_assay_tables_v1',)
                if policy.scope_profile == 'tier_a_antigen_all_assay_tables_v1'
                else ()
            ),
        ),
    )
    return SourceVerifierRunResult(result=result, verified_records=record_bytes)


def _canonical_antigen_policy(payload: bytes) -> IedbAntigenAdapterPolicy:
    if not isinstance(payload, bytes) or not payload:
        raise IedbProductionAdapterError('IEDB antigen adapter policy must be nonempty exact bytes')
    try:
        policy = IedbAntigenAdapterPolicy.model_validate_json(payload)
    except ValueError as error:
        raise IedbProductionAdapterError(f'invalid IEDB antigen adapter policy: {error}') from error
    if payload != canonical_json_bytes(policy):
        raise IedbProductionAdapterError('IEDB antigen adapter policy must use canonical JSON encoding')
    return policy


def _rebind_source_rows(source_input: AdapterSourceInput) -> dict[tuple[str, str], dict[str, Any]]:
    """Resolve each verified row back to the exact captured body that committed it."""

    payload_by_sha256: dict[str, bytes] = {}
    for capture in source_input.captures:
        for artifact in capture.artifacts:
            if not artifact.role.startswith('body.'):
                continue
            payload = artifact.payload
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_sha256 != artifact.sha256:
                raise IedbProductionAdapterError('IEDB captured body differs from its promoted binding')
            prior = payload_by_sha256.setdefault(actual_sha256, payload)
            if prior != payload:
                raise IedbProductionAdapterError('IEDB captured bodies exhibit a SHA-256 collision')

    row_index_by_artifact: dict[str, dict[str, dict[str, Any]]] = {}
    for artifact_sha256, payload in payload_by_sha256.items():
        rows = _strict_json_array(payload, f'captured body {artifact_sha256}')
        row_index: dict[str, dict[str, Any]] = {}
        for raw in rows:
            row_sha256 = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
            if row_sha256 in row_index:
                raise IedbProductionAdapterError('IEDB captured body contains duplicate canonical source records')
            row_index[row_sha256] = raw
        row_index_by_artifact[artifact_sha256] = row_index

    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for source in source_input.verified_records:
        rows = row_index_by_artifact.get(source.source_artifact_sha256)
        raw = None if rows is None else rows.get(source.source_record_sha256)
        if raw is None:
            raise IedbProductionAdapterError(
                f'IEDB verified row {source.source_record_id!r} cannot be rebound to captured bytes'
            )
        key = (source.source_id, source.source_record_id)
        if key in resolved:
            raise IedbProductionAdapterError('IEDB verified source record identities are duplicated')
        resolved[key] = raw
    return resolved


def _preflight_tier_a_antigen_rows(
    source_input: AdapterSourceInput,
    policy: IedbAntigenAdapterPolicy,
) -> None:
    required_metadata = {f'api_metrics:{table_name}' for table_name in IEDB_TIER_A_ANTIGEN_TABLES}
    record_ids = {record.source_record_id for record in source_input.verified_records}
    if not required_metadata.issubset(record_ids):
        raise IedbProductionAdapterError('the Tier A IEDB inventory omits required all-assay api_metrics records')
    raw_rows = _rebind_source_rows(source_input)
    table_policy = {item.table_name: item for item in policy.eligible_tables}
    for source in source_input.verified_records:
        if source.source_record_id.startswith('api_metrics:'):
            continue
        table_name, separator, _identifier = source.source_record_id.partition(':')
        selected_table = table_policy.get(table_name)
        if not separator or selected_table is None:
            raise IedbProductionAdapterError(
                'the Tier A IEDB inventory contains a row outside the fixed all-assay profile'
            )
        raw = raw_rows[(source.source_id, source.source_record_id)]
        required_fields = _IEDB_TIER_A_REQUIRED_ROW_FIELDS | {selected_table.id_field}
        missing = sorted(required_fields - raw.keys())
        if missing:
            raise IedbProductionAdapterError(
                f'{source.source_record_id} omits required unprojected IEDB fields: {missing!r}'
            )
        if raw['structure_iri'] is None or raw['reference_iri'] is None:
            continue
        try:
            normalize_assay(selected_table.endpoint, raw)
        except IedbAdapterError as error:
            raise IedbProductionAdapterError(
                f'{source.source_record_id} contains malformed Tier A assay fields'
            ) from error


def _antigen_candidate_id(parent_source_antigen_iri: str) -> str:
    digest = hashlib.sha256(parent_source_antigen_iri.encode('utf-8')).hexdigest()
    return f'iedb-antigen-{digest}'


def _antigen_evidence_record(
    *,
    policy: IedbAntigenAdapterPolicy,
    source: SourceRecordBinding,
    assay: NormalizedIedbAssay,
    candidate_id: str,
    available_at: datetime,
) -> EvidenceRecord:
    evidence_seed = canonical_json_bytes(
        {
            'adapter_id': IEDB_ANTIGEN_ADAPTER_ID,
            'adapter_version': IEDB_ANTIGEN_ADAPTER_VERSION,
            'episode_id': policy.episode_id,
            'source_id': source.source_id,
            'source_record_id': source.source_record_id,
            'source_record_sha256': source.source_record_sha256,
        }
    )
    evidence_id = f'iedb-evidence-{hashlib.sha256(evidence_seed).hexdigest()}'
    body = _render_antigen_evidence(assay, candidate_id)
    return EvidenceRecord(
        episode_id=policy.episode_id,
        evidence_id=evidence_id,
        source_type=SourceType.EXPERIMENTAL,
        collected_at=None,
        available_at=available_at,
        title=f'IEDB {assay.endpoint.value} evidence for {candidate_id}',
        body=body,
        body_sha256=hashlib.sha256(body.encode('utf-8')).hexdigest(),
        related_candidate_ids=[candidate_id],
        provenance_url=source.source_locator,
        license_id=policy.license_id,
        derivation=(
            f'Deterministic {IEDB_ANTIGEN_ADAPTER_ID} {IEDB_ANTIGEN_ADAPTER_VERSION} '
            'normalization from one verifier-enumerated cutoff capture row. Availability '
            'uses the verified captured IEDB source-build release field; no later lookup is used.'
        ),
    )


def _render_antigen_evidence(assay: NormalizedIedbAssay, candidate_id: str) -> str:
    lines = [
        f'Candidate ID: {candidate_id}.',
        f'Parent source antigen IRI: {assay.parent_source_antigen_iri}.',
        f'Assay endpoint: {assay.endpoint.value}.',
        f'Assay IRI: {assay.assay_iri}.',
        f'Epitope or structure IRI: {assay.structure_iri}.',
        f'Reference IRI: {assay.reference_iri}.',
        f'Qualitative result: {assay.qualitative_measure or "not reported"}.',
    ]
    _append_evidence_value(lines, 'Parent source antigen name', assay.parent_source_antigen_name)
    _append_evidence_value(lines, 'Curated source antigen name', assay.curated_source_antigen_name)
    _append_evidence_value(lines, 'Source organism IRI', assay.source_organism_iri)
    _append_evidence_value(lines, 'Source organism name', assay.source_organism_name)
    _append_evidence_value(lines, 'Host organism IRI', assay.host_organism_iri)
    _append_evidence_value(lines, 'Host organism name', assay.host_organism_name)
    _append_evidence_value(lines, 'MHC allele IRI', assay.mhc_allele_iri)
    _append_evidence_value(lines, 'MHC allele name', assay.mhc_allele_name)
    if assay.region_start is not None and assay.region_end is not None:
        lines.append(f'Curated antigen region: {assay.region_start}-{assay.region_end}.')
    if assay.assay_names:
        lines.append(f'Assay names: {"; ".join(assay.assay_names)}.')
    if assay.reference_titles:
        lines.append(f'Reference titles: {"; ".join(assay.reference_titles)}.')
    return '\n'.join(lines)


def _append_evidence_value(lines: list[str], label: str, value: str | None) -> None:
    if value is not None:
        lines.append(f'{label}: {value}.')


def _normalized_reference(
    episode_id: str,
    record_id: str,
    record: StrictModel,
) -> NormalizedRecordReference:
    return NormalizedRecordReference(
        episode_id=episode_id,
        record_id=record_id,
        record_sha256=hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
    )


def _jsonl(records: tuple[StrictModel, ...]) -> bytes:
    if not records:
        raise IedbProductionAdapterError('IEDB normalized record inventory cannot be empty')
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


def _canonical_policy(payload: bytes) -> IedbSourceVerifierPolicy:
    if not isinstance(payload, bytes) or not payload:
        raise IedbProductionSourceError('IEDB verifier policy must be nonempty exact bytes')
    try:
        policy = IedbSourceVerifierPolicy.model_validate_json(payload)
    except ValueError as error:
        raise IedbProductionSourceError(f'invalid IEDB verifier policy: {error}') from error
    if payload != canonical_json_bytes(policy):
        raise IedbProductionSourceError('IEDB verifier policy must use canonical JSON encoding')
    return policy


def _artifact_url(layout: IedbPromotionLayout, artifact_id: str) -> str:
    if artifact_id in {layout.metrics_before_artifact_id, layout.metrics_after_artifact_id}:
        return layout.metrics_url
    page = next((item.page for item in layout.pages if item.artifact_id == artifact_id), None)
    if page is None:
        raise IedbProductionSourceError(f'unknown IEDB artifact ID: {artifact_id}')
    return page.request_url


def _verify_https_receipt(
    payload: bytes,
    *,
    body_sha256: str,
    body_bytes: int,
    expected_url: str,
    captured_at: datetime,
    accepted_tls_versions: tuple[str, ...],
) -> HttpsCaptureReceipt:
    try:
        receipt = HttpsCaptureReceipt.model_validate_json(payload)
    except ValueError as error:
        raise IedbProductionSourceError(f'invalid IEDB HTTPS receipt: {error}') from error
    if payload != canonical_json_bytes(receipt):
        raise IedbProductionSourceError('IEDB HTTPS receipt is not canonical JSON')
    if (
        receipt.requested_url != expected_url
        or receipt.final_url != expected_url
        or receipt.body_sha256 != body_sha256
        or receipt.body_byte_count != body_bytes
    ):
        raise IedbProductionSourceError('IEDB HTTPS receipt differs from its exact body or URL')
    if receipt.completed_at > captured_at:
        raise IedbProductionSourceError('IEDB HTTPS receipt completes after the selected capture')
    peer = receipt.tls_peer
    expected_host = urlsplit(expected_url).hostname
    if (
        peer is None
        or peer.server_name != expected_host
        or peer.certificate_der_sha256 is None
        or peer.tls_version not in accepted_tls_versions
    ):
        raise IedbProductionSourceError(
            'IEDB capture lacks required collector-reported official-origin TLS peer metadata'
        )
    return receipt


def _iedb_receipt(receipt: HttpsCaptureReceipt) -> IedbApiExchangeReceipt:
    if receipt.status_code not in (200, 206):
        raise IedbProductionSourceError('IEDB API receipts must use status 200 or 206')
    status_code: Literal[200, 206] = 200 if receipt.status_code == 200 else 206
    request_headers: list[IedbHttpHeader] = []
    for header in receipt.request_headers:
        if header.name not in _REQUEST_HEADERS:
            raise IedbProductionSourceError(f'IEDB request uses unsupported header {header.name!r}')
        request_headers.append(IedbHttpHeader(name=header.name, value=header.value))
    response_headers: list[IedbHttpHeader] = []
    for header in receipt.response_headers:
        if header.name not in _RESPONSE_HEADERS:
            continue
        if len(header.values) != 1:
            raise IedbProductionSourceError(f'IEDB response header {header.name!r} is ambiguous')
        response_headers.append(IedbHttpHeader(name=header.name, value=header.values[0]))
    return IedbApiExchangeReceipt(
        request_url=receipt.requested_url,
        request_headers=tuple(sorted(request_headers, key=lambda item: item.name)),
        status_code=status_code,
        response_headers=tuple(sorted(response_headers, key=lambda item: item.name)),
    )


def _source_records(
    policy: IedbSourceVerifierPolicy,
    artifacts: dict[str, Any],
) -> tuple[tuple[SourceRecordBinding, ...], dict[str, dict[str, Any]]]:
    records: list[SourceRecordBinding] = []
    payload_by_id: dict[str, dict[str, Any]] = {}
    metrics_artifact = artifacts[f'body.{policy.layout.metrics_before_artifact_id}']
    metrics = _strict_json_array(metrics_artifact.payload, 'api_metrics')
    for raw in metrics:
        metric = IedbApiMetric.model_validate(raw)
        record_id = f'api_metrics:{metric.search_table_name}'
        _append_source_record(
            records,
            payload_by_id,
            source_id=policy.source_id,
            record_id=record_id,
            raw=raw,
            artifact_sha256=metrics_artifact.sha256,
            locator=f'{policy.layout.metrics_url}#search_table_name={quote(metric.search_table_name, safe="")}',
        )
    for captured_page in policy.layout.pages:
        page = captured_page.page
        body = artifacts[f'body.{captured_page.artifact_id}']
        for raw in _strict_json_array(body.payload, page.data_relative_path):
            identifier = raw.get(page.id_field)
            if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 0:
                raise IedbProductionSourceError(f'{page.data_relative_path} row has invalid {page.id_field}')
            record_id = f'{page.table_name}:{identifier}'
            _append_source_record(
                records,
                payload_by_id,
                source_id=policy.source_id,
                record_id=record_id,
                raw=raw,
                artifact_sha256=body.sha256,
                locator=f'{page.request_url}#{page.id_field}={identifier}',
            )
    ordered = tuple(sorted(records, key=lambda item: (item.source_id, item.source_record_id)))
    if len(ordered) != len(payload_by_id):
        raise IedbProductionSourceError('IEDB source record identifiers are not globally unique')
    return ordered, payload_by_id


def _append_source_record(
    records: list[SourceRecordBinding],
    payload_by_id: dict[str, dict[str, Any]],
    *,
    source_id: str,
    record_id: str,
    raw: dict[str, Any],
    artifact_sha256: str,
    locator: str,
) -> None:
    if record_id in payload_by_id:
        raise IedbProductionSourceError(f'duplicate IEDB source record ID: {record_id}')
    payload_by_id[record_id] = raw
    records.append(
        SourceRecordBinding(
            source_id=source_id,
            source_record_id=record_id,
            source_record_sha256=hashlib.sha256(canonical_json_bytes(raw)).hexdigest(),
            source_artifact_sha256=artifact_sha256,
            source_locator=locator,
        )
    )


def _strict_json_array(payload: bytes, label: str) -> tuple[dict[str, Any], ...]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise IedbProductionSourceError(f'{label} contains duplicate JSON key {name!r}')
            result[name] = value
        return result

    def reject_constant(value: str) -> None:
        raise IedbProductionSourceError(f'{label} contains non-finite JSON number {value}')

    try:
        value = json.loads(
            payload.decode('utf-8'),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IedbProductionSourceError(f'{label} is not strict UTF-8 JSON: {error}') from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise IedbProductionSourceError(f'{label} must contain a JSON array of objects')
    return tuple(value)


def _metric_timestamp(metric: IedbApiMetric) -> datetime:
    value = datetime.fromisoformat(metric.creation_date.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        # IEDB currently emits naive ISO timestamps. Its existing capture contract
        # explicitly freezes the interpretation as UTC.
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise IedbProductionSourceError(f'{label} must be a lowercase SHA-256 digest')
