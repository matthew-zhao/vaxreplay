"""Production ClinicalTrials.gov API v2 verification and cutoff-safe study nomination.

The verifier consumes one preregistered collection containing API ``/version``
responses before and after one complete, protocol-only, single-page parent query.
The parent query may search only condition and intervention protocol fields; result
fields, arbitrary advanced query syntax, explicit NCT lists, and status/result filters
are forbidden. It is complete only when ``countTotal=true``, ``totalCount`` equals
the exact returned study count, and no ``nextPageToken`` exists. This avoids both
result-dependent candidate selection and pretending that a static collector can
preregister opaque continuation tokens.

The adapter deliberately reads only an allowlisted projection of ``protocolSection``.
It never reads ``resultsSection``, ``derivedSection``, document content, or the
top-level ``hasResults`` signal.  Every verified record receives an explicit
disposition and every eligible study is emitted; there is no score or top-N filter.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, quote, urlsplit
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord, SourceType, StrictModel
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

CTGOV_LAYOUT_SCHEMA_VERSION = 'vaxreplay.ctgov-api-v2-layout.v0.2'
CTGOV_SOURCE_POLICY_SCHEMA_VERSION = 'vaxreplay.ctgov-source-verifier-policy.v0.3'
CTGOV_STUDY_ADAPTER_POLICY_SCHEMA_VERSION = 'vaxreplay.ctgov-study-adapter-policy.v0.1'
CTGOV_STUDY_MAP_SCHEMA_VERSION = 'vaxreplay.ctgov-study-candidate-map.v0.1'
CTGOV_SOURCE_VERIFIER_ID = 'clinicaltrials-gov-api-v2-offline-verifier'
CTGOV_SOURCE_VERIFIER_VERSION = 'v0.3'
CTGOV_STUDY_ADAPTER_ID = 'clinicaltrials-gov-protocol-study-universe-adapter'
CTGOV_STUDY_ADAPTER_VERSION = 'v0.1'
CTGOV_STUDY_ADAPTER_EXCLUSION_REASON_CODES = (
    'insufficient_protocol_fields',
    'intervention_type_out_of_scope',
    'not_interventional',
    'phase_out_of_scope',
    'source_metadata_record',
)

_OFFICIAL_ORIGIN = 'https://clinicaltrials.gov'
_VERSION_URL = 'https://clinicaltrials.gov/api/v2/version'
_ARTIFACT_ID_PATTERN = r'^[a-z][a-z0-9._-]{0,109}$'
_NCT_ID_RE = re.compile(r'^NCT\d{8}$')
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_PAGE_QUERY_CONTROL_NAMES = {'countTotal', 'format', 'markupFormat', 'pageSize', 'sort'}
_PROTOCOL_PARENT_QUERY_NAMES = frozenset({'query.cond', 'query.intr'})
_FORBIDDEN_QUERY_SYNTAX = re.compile(
    r'(?:[\[\]{}]|\b(?:AREA|RANGE|HasResults|ResultsFirstSubmitDate|ResultsFirstPostDate|'
    r'ResultsFirstPosted|Outcome|ResultsSection)\b)',
    re.IGNORECASE,
)


class CtgovProductionSourceError(ValueError):
    """A captured API v2 inventory cannot support the committed source claim."""


class CtgovProductionAdapterError(ValueError):
    """A verified API v2 inventory cannot support the committed study universe."""


class CtgovCapturedQuery(StrictModel):
    """One preregistered protocol-only parent universe that fits one API page."""

    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    query_id: str = Field(pattern=r'^[a-z][a-z0-9._-]{0,99}$')
    request_url: str = Field(min_length=1, max_length=8192)

    @field_validator('request_url')
    @classmethod
    def validate_request_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != 'https'
            or parsed.hostname != 'clinicaltrials.gov'
            or parsed.netloc != 'clinicaltrials.gov'
            or parsed.path != '/api/v2/studies'
            or parsed.fragment
        ):
            raise ValueError('ClinicalTrials.gov study queries must use the exact official API v2 endpoint')
        try:
            pairs = tuple(parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True))
        except ValueError as error:
            raise ValueError('ClinicalTrials.gov query string is ambiguous') from error
        names = tuple(name for name, _item in pairs)
        if not pairs or len(names) != len(set(names)) or any(not name or not item for name, item in pairs):
            raise ValueError('ClinicalTrials.gov query parameters must be unique and nonempty')
        if names != tuple(sorted(names)):
            raise ValueError('ClinicalTrials.gov query parameters must use canonical name order')
        query = dict(pairs)
        if query.get('format') != 'json' or query.get('countTotal') != 'true':
            raise ValueError('ClinicalTrials.gov complete partitions require format=json and countTotal=true')
        if query.get('pageSize') != '1000':
            raise ValueError('ClinicalTrials.gov complete partitions require the maximum pageSize=1000')
        if query.get('markupFormat') != 'legacy':
            raise ValueError('ClinicalTrials.gov captures must freeze markupFormat=legacy')
        if 'pageToken' in query:
            raise ValueError('opaque continuation tokens cannot be preregistered as static query partitions')
        scope_names = set(query) - _PAGE_QUERY_CONTROL_NAMES
        if not scope_names or not scope_names.issubset(_PROTOCOL_PARENT_QUERY_NAMES):
            raise ValueError('Tier A ClinicalTrials.gov parent queries may use only query.cond and query.intr')
        for name in sorted(scope_names):
            scope_value = query[name]
            if (
                scope_value != scope_value.strip()
                or not scope_value.isascii()
                or any(ord(character) < 0x20 or ord(character) == 0x7F for character in scope_value)
                or _FORBIDDEN_QUERY_SYNTAX.search(scope_value) is not None
            ):
                raise ValueError('Tier A ClinicalTrials.gov parent query values must be plain protocol-only text')
        return value


class CtgovPromotionLayout(StrictModel):
    schema_version: Literal['vaxreplay.ctgov-api-v2-layout.v0.2'] = CTGOV_LAYOUT_SCHEMA_VERSION
    version_url: Literal['https://clinicaltrials.gov/api/v2/version'] = _VERSION_URL
    version_before_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    version_after_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    queries: tuple[CtgovCapturedQuery, ...] = Field(min_length=1, max_length=1)

    @field_validator('queries')
    @classmethod
    def validate_queries(cls, value: tuple[CtgovCapturedQuery, ...]) -> tuple[CtgovCapturedQuery, ...]:
        if len(value) != 1:
            raise ValueError('Tier A ClinicalTrials.gov uses exactly one complete parent-universe query')
        keys = tuple((item.query_id, item.artifact_id) for item in value)
        if keys != tuple(sorted(keys)):
            raise ValueError('ClinicalTrials.gov query partitions must use canonical query/artifact order')
        if len({item.query_id for item in value}) != len(value):
            raise ValueError('ClinicalTrials.gov query IDs must be unique')
        if len({item.artifact_id for item in value}) != len(value):
            raise ValueError('ClinicalTrials.gov query artifact IDs must be unique')
        return value

    @model_validator(mode='after')
    def validate_artifacts(self) -> Self:
        artifact_ids = (
            self.version_before_artifact_id,
            self.version_after_artifact_id,
            *(item.artifact_id for item in self.queries),
        )
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError('every ClinicalTrials.gov artifact ID must be unique')
        query_artifact_ids = tuple(item.artifact_id for item in self.queries)
        if query_artifact_ids != tuple(sorted(query_artifact_ids)):
            raise ValueError('ClinicalTrials.gov query artifact IDs must use static collection order')
        if not all(
            self.version_before_artifact_id < artifact_id < self.version_after_artifact_id
            for artifact_id in query_artifact_ids
        ):
            raise ValueError('ClinicalTrials.gov artifact IDs must collect version-before, queries, version-after')
        return self

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        return (
            self.version_before_artifact_id,
            self.version_after_artifact_id,
            *(item.artifact_id for item in self.queries),
        )


class CtgovSourceVerifierPolicy(StrictModel):
    schema_version: Literal['vaxreplay.ctgov-source-verifier-policy.v0.3'] = CTGOV_SOURCE_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    expected_api_version: str = Field(min_length=1, max_length=100)
    layout: CtgovPromotionLayout
    accepted_tls_versions: tuple[Literal['TLSv1.2', 'TLSv1.3'], ...] = ('TLSv1.2', 'TLSv1.3')
    source_authentication: Literal['collector_receipt_consistent_with_system_ca_tls_to_official_origin'] = (
        'collector_receipt_consistent_with_system_ca_tls_to_official_origin'
    )
    completeness_semantics: Literal['complete_protocol_only_single_page_parent_universe'] = (
        'complete_protocol_only_single_page_parent_universe'
    )
    source_release_semantics: Literal['api_v2_data_timestamp_offset_or_conservative_america_new_york_later_fold'] = (
        'api_v2_data_timestamp_offset_or_conservative_america_new_york_later_fold'
    )
    naive_data_timestamp_timezone: Literal['America/New_York'] = 'America/New_York'
    captures_per_verification: Literal[1] = 1

    @field_validator('accepted_tls_versions')
    @classmethod
    def validate_tls_versions(
        cls,
        value: tuple[Literal['TLSv1.2', 'TLSv1.3'], ...],
    ) -> tuple[Literal['TLSv1.2', 'TLSv1.3'], ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('accepted TLS versions must be sorted and unique')
        return value


class CtgovStudyAdapterPolicy(StrictModel):
    """Precommitted all-study nomination policy over protocol-only fields."""

    schema_version: Literal['vaxreplay.ctgov-study-adapter-policy.v0.1'] = CTGOV_STUDY_ADAPTER_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    episode_id: str = Field(min_length=1, max_length=1024)
    decision_at: datetime
    allowed_intervention_types: tuple[str, ...] = Field(min_length=1, max_length=64)
    allowed_phases: tuple[str, ...] = Field(min_length=1, max_length=32)
    minimum_candidate_count: int = Field(default=2, ge=2, le=1_000_000)
    license_id: str = Field(default='ClinicalTrials.gov Terms and Conditions', min_length=1)
    candidate_universe_semantics: Literal['all_protocol_eligible_studies_without_ranking_or_count_filter'] = (
        'all_protocol_eligible_studies_without_ranking_or_count_filter'
    )
    field_allowlist_semantics: Literal['protocol_section_projection_v1_results_fields_forbidden'] = (
        'protocol_section_projection_v1_results_fields_forbidden'
    )

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ClinicalTrials.gov study decision_at')

    @field_validator('allowed_intervention_types', 'allowed_phases')
    @classmethod
    def validate_vocabularies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('ClinicalTrials.gov adapter vocabularies must be sorted and unique')
        if any(not item or item != item.strip() or item != item.upper() for item in value):
            raise ValueError('ClinicalTrials.gov adapter vocabulary values must be uppercase exact strings')
        return value


class CtgovStudyCandidateMapEntry(StrictModel):
    candidate_id: str
    nct_id: str = Field(pattern=r'^NCT\d{8}$')


class CtgovStudyCandidateMap(StrictModel):
    schema_version: Literal['vaxreplay.ctgov-study-candidate-map.v0.1'] = CTGOV_STUDY_MAP_SCHEMA_VERSION
    policy_id: str
    episode_id: str
    field_allowlist_semantics: Literal['protocol_section_projection_v1_results_fields_forbidden'] = (
        'protocol_section_projection_v1_results_fields_forbidden'
    )
    candidates: tuple[CtgovStudyCandidateMapEntry, ...] = Field(min_length=2)

    @field_validator('candidates')
    @classmethod
    def validate_candidates(
        cls,
        value: tuple[CtgovStudyCandidateMapEntry, ...],
    ) -> tuple[CtgovStudyCandidateMapEntry, ...]:
        keys = tuple((item.candidate_id, item.nct_id) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('ClinicalTrials.gov candidate map must be canonically sorted and unique')
        return value


@dataclass(frozen=True)
class _EligibleStudy:
    source: SourceRecordBinding
    candidate: CandidateRecord
    evidence: EvidenceRecord
    nct_id: str


def ctgov_source_verifier_policy_bytes(policy: CtgovSourceVerifierPolicy) -> bytes:
    if not isinstance(policy, CtgovSourceVerifierPolicy):
        raise TypeError('policy must be a CtgovSourceVerifierPolicy')
    return canonical_json_bytes(policy)


def ctgov_study_adapter_policy_bytes(policy: CtgovStudyAdapterPolicy) -> bytes:
    if not isinstance(policy, CtgovStudyAdapterPolicy):
        raise TypeError('policy must be a CtgovStudyAdapterPolicy')
    return canonical_json_bytes(policy)


def verify_ctgov_source(
    verifier_input: SourceVerificationInput,
    policy_bytes: bytes,
    *,
    implementation_sha256: str,
    execution_environment_sha256: str,
) -> SourceVerifierRunResult:
    """Authenticate and enumerate one exact, stable API v2 query inventory."""

    policy = _canonical_source_policy(policy_bytes)
    _require_sha256(implementation_sha256, 'implementation_sha256')
    _require_sha256(execution_environment_sha256, 'execution_environment_sha256')
    if verifier_input.source_id != policy.source_id or len(verifier_input.captures) != 1:
        raise CtgovProductionSourceError('ClinicalTrials.gov verification requires its one committed capture')
    capture = verifier_input.captures[0]
    if capture.binding.source_id != policy.source_id:
        raise CtgovProductionSourceError('ClinicalTrials.gov capture belongs to a different source')

    artifacts = {item.role: item for item in capture.artifacts}
    if len(artifacts) != len(capture.artifacts):
        raise CtgovProductionSourceError('ClinicalTrials.gov capture contains duplicate artifact roles')
    expected_bodies = {f'body.{item}' for item in policy.layout.artifact_ids}
    expected_receipts = {f'receipt.{item}' for item in policy.layout.artifact_ids}
    if {role for role in artifacts if role.startswith('body.')} != expected_bodies or {
        role for role in artifacts if role.startswith('receipt.')
    } != expected_receipts:
        raise CtgovProductionSourceError('ClinicalTrials.gov artifact inventory differs from its layout')

    receipts: dict[str, HttpsCaptureReceipt] = {}
    for artifact_id in policy.layout.artifact_ids:
        body = artifacts[f'body.{artifact_id}']
        if hashlib.sha256(body.payload).hexdigest() != body.sha256 or len(body.payload) != body.byte_count:
            raise CtgovProductionSourceError('ClinicalTrials.gov body differs from its promoted binding')
        receipts[artifact_id] = _verify_https_receipt(
            artifacts[f'receipt.{artifact_id}'].payload,
            body_sha256=body.sha256,
            body_bytes=body.byte_count,
            expected_url=_artifact_url(policy.layout, artifact_id),
            captured_at=capture.binding.captured_at,
            accepted_tls_versions=policy.accepted_tls_versions,
        )
    version_before_receipt = receipts[policy.layout.version_before_artifact_id]
    version_after_receipt = receipts[policy.layout.version_after_artifact_id]
    query_receipts = tuple(receipts[item.artifact_id] for item in policy.layout.queries)
    if (
        version_before_receipt.completed_at > min(item.started_at for item in query_receipts)
        or max(item.completed_at for item in query_receipts) > version_after_receipt.started_at
    ):
        raise CtgovProductionSourceError(
            'ClinicalTrials.gov receipt times do not prove version-before, queries, version-after bracketing'
        )

    version_before_artifact = artifacts[f'body.{policy.layout.version_before_artifact_id}']
    version_after_artifact = artifacts[f'body.{policy.layout.version_after_artifact_id}']
    version_before = _strict_json_object(version_before_artifact.payload, 'API version before')
    version_after = _strict_json_object(version_after_artifact.payload, 'API version after')
    if version_before != version_after:
        raise CtgovProductionSourceError('ClinicalTrials.gov API version changed during capture')
    api_version, source_release_at = _version_identity(version_before)
    if api_version != policy.expected_api_version:
        raise CtgovProductionSourceError('ClinicalTrials.gov API version differs from the precommitment')
    if source_release_at > capture.binding.captured_at:
        raise CtgovProductionSourceError('ClinicalTrials.gov dataTimestamp is after the selected capture')

    records: list[SourceRecordBinding] = []
    version_record_id = f'api_version:{api_version}'
    _append_record(
        records,
        source_id=policy.source_id,
        record_id=version_record_id,
        raw=version_before,
        artifact_sha256=version_before_artifact.sha256,
        locator=f'{policy.layout.version_url}#dataTimestamp',
    )
    seen_nct_ids: set[str] = set()
    for query in policy.layout.queries:
        body = artifacts[f'body.{query.artifact_id}']
        page = _strict_json_object(body.payload, f'query partition {query.query_id}')
        studies = _complete_study_page(page, query.query_id)
        for study in studies:
            nct_id = _study_nct_id(study)
            if nct_id in seen_nct_ids:
                raise CtgovProductionSourceError('ClinicalTrials.gov query partitions are not disjoint by NCT ID')
            seen_nct_ids.add(nct_id)
            _append_record(
                records,
                source_id=policy.source_id,
                record_id=f'study:{nct_id}',
                raw=study,
                artifact_sha256=body.sha256,
                locator=f'{query.request_url}#nctId={quote(nct_id, safe="")}',
            )

    ordered = tuple(sorted(records, key=lambda item: (item.source_id, item.source_record_id)))
    records_bytes = _jsonl(ordered, CtgovProductionSourceError)
    version_binding = next(item for item in ordered if item.source_record_id == version_record_id)
    result = SourceVerificationResult(
        source_id=policy.source_id,
        verifier=SourceVerifierIdentity(
            verifier_id=CTGOV_SOURCE_VERIFIER_ID,
            verifier_version=CTGOV_SOURCE_VERIFIER_VERSION,
            implementation_sha256=implementation_sha256,
            execution_environment_sha256=execution_environment_sha256,
        ),
        verifier_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        verified_attempt_ids=(capture.binding.attempt_id,),
        source_release=AuthoritativeSourceRelease(
            source_release_at=source_release_at,
            basis=AuthoritativeReleaseBasis.SOURCE_API_PUBLICATION_FIELD,
            authority_locator=f'{policy.layout.version_url}#dataTimestamp',
            authority_field='dataTimestamp',
            evidence_attempt_id=capture.binding.attempt_id,
            evidence_role=f'body.{policy.layout.version_before_artifact_id}',
            evidence_sha256=version_binding.source_artifact_sha256,
            evidence_source_record_id=version_binding.source_record_id,
            evidence_source_record_sha256=version_binding.source_record_sha256,
        ),
        verified_capture_inventory_sha256=verifier_input.capture_inventory_sha256,
        verified_source_record_inventory_sha256=hashlib.sha256(records_bytes).hexdigest(),
        verified_source_record_count=len(ordered),
        result_codes=(
            'api_version_stable_during_capture',
            'collector_receipt_consistent_with_system_ca_tls_to_official_origin',
            'complete_protocol_only_single_page_parent_universe',
        ),
    )
    return SourceVerifierRunResult(result=result, verified_records=records_bytes)


def adapt_ctgov_study_candidates(
    inputs: tuple[AdapterSourceInput, ...],
    policy_bytes: bytes,
) -> AdapterRunResult:
    """Emit every protocol-eligible study and ignore all results-bearing fields."""

    policy = _canonical_adapter_policy(policy_bytes)
    if len(inputs) != 1 or inputs[0].source_id != policy.source_id:
        raise CtgovProductionAdapterError('ClinicalTrials.gov adaptation requires its one committed source')
    source_input = inputs[0]
    verification = source_input.verification_result
    if verification.source_id != policy.source_id:
        raise CtgovProductionAdapterError('ClinicalTrials.gov verification belongs to a different source')
    if verification.source_release.source_release_at > policy.decision_at:
        raise CtgovProductionAdapterError('ClinicalTrials.gov dataTimestamp is after the decision cutoff')
    if any(capture.binding.captured_at > policy.decision_at for capture in source_input.captures):
        raise CtgovProductionAdapterError('ClinicalTrials.gov selected capture is after the decision cutoff')
    inventory_bytes = _jsonl(source_input.verified_records, CtgovProductionAdapterError)
    if (
        len(source_input.verified_records) != verification.verified_source_record_count
        or hashlib.sha256(inventory_bytes).hexdigest() != verification.verified_source_record_inventory_sha256
    ):
        raise CtgovProductionAdapterError('ClinicalTrials.gov adapter input differs from verified records')
    if any(record.source_id != policy.source_id for record in source_input.verified_records):
        raise CtgovProductionAdapterError('ClinicalTrials.gov adapter input contains a foreign source row')

    raw_rows = _rebind_rows(source_input)
    eligible: list[_EligibleStudy] = []
    excluded: dict[tuple[str, str], str] = {}
    for source in source_input.verified_records:
        key = (source.source_id, source.source_record_id)
        if source.source_record_id.startswith('api_version:'):
            excluded[key] = 'source_metadata_record'
            continue
        raw = raw_rows[key]
        try:
            projection = _protocol_projection(raw)
        except CtgovProductionAdapterError:
            excluded[key] = 'insufficient_protocol_fields'
            continue
        if projection['study_type'] != 'INTERVENTIONAL':
            excluded[key] = 'not_interventional'
            continue
        phases = projection['phases']
        if not set(phases) & set(policy.allowed_phases):
            excluded[key] = 'phase_out_of_scope'
            continue
        intervention_types = {item['type'] for item in projection['interventions']}
        if not intervention_types & set(policy.allowed_intervention_types):
            excluded[key] = 'intervention_type_out_of_scope'
            continue
        nct_id = projection['nct_id']
        candidate_id = f'ctgov-study-{nct_id}'
        candidate = CandidateRecord(
            episode_id=policy.episode_id,
            candidate_id=candidate_id,
            eligible=True,
        )
        body = _render_protocol_projection(projection, candidate_id)
        evidence_seed = canonical_json_bytes(
            {
                'adapter_id': CTGOV_STUDY_ADAPTER_ID,
                'adapter_version': CTGOV_STUDY_ADAPTER_VERSION,
                'episode_id': policy.episode_id,
                'source_record_id': source.source_record_id,
                'source_record_sha256': source.source_record_sha256,
            }
        )
        evidence = EvidenceRecord(
            episode_id=policy.episode_id,
            evidence_id=f'ctgov-evidence-{hashlib.sha256(evidence_seed).hexdigest()}',
            source_type=SourceType.PUBLIC_HEALTH,
            collected_at=None,
            available_at=verification.source_release.source_release_at,
            title=f'ClinicalTrials.gov protocol evidence for {nct_id}',
            body=body,
            body_sha256=hashlib.sha256(body.encode('utf-8')).hexdigest(),
            related_candidate_ids=[candidate_id],
            provenance_url=source.source_locator,
            license_id=policy.license_id,
            derivation=(
                f'Deterministic {CTGOV_STUDY_ADAPTER_ID} {CTGOV_STUDY_ADAPTER_VERSION} '
                'allowlisted protocolSection projection. Results, derived, document, and '
                'hasResults fields are never read.'
            ),
        )
        eligible.append(_EligibleStudy(source, candidate, evidence, nct_id))

    if len(eligible) < policy.minimum_candidate_count:
        raise CtgovProductionAdapterError(
            'complete ClinicalTrials.gov universe has fewer candidates than the committed minimum'
        )
    candidates = tuple(sorted((item.candidate for item in eligible), key=lambda item: item.candidate_id))
    evidence = tuple(sorted((item.evidence for item in eligible), key=lambda item: item.evidence_id))
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise CtgovProductionAdapterError('ClinicalTrials.gov candidate identities are duplicated')

    candidate_refs = {
        item.candidate_id: _normalized_reference(item.episode_id, item.candidate_id, item) for item in candidates
    }
    evidence_refs = {
        item.evidence_id: _normalized_reference(item.episode_id, item.evidence_id, item) for item in evidence
    }
    normalized = {
        (item.source.source_id, item.source.source_record_id): SourceRecordDisposition(
            source_id=item.source.source_id,
            source_record_id=item.source.source_record_id,
            source_record_sha256=item.source.source_record_sha256,
            source_artifact_sha256=item.source.source_artifact_sha256,
            disposition='normalized',
            candidate_record_refs=(candidate_refs[item.candidate.candidate_id],),
            evidence_record_refs=(evidence_refs[item.evidence.evidence_id],),
        )
        for item in eligible
    }
    dispositions = tuple(
        sorted(
            (
                normalized[key]
                if key in normalized
                else SourceRecordDisposition(
                    source_id=source.source_id,
                    source_record_id=source.source_record_id,
                    source_record_sha256=source.source_record_sha256,
                    source_artifact_sha256=source.source_artifact_sha256,
                    disposition='excluded',
                    reason_code=excluded[key],
                )
                for source in source_input.verified_records
                for key in ((source.source_id, source.source_record_id),)
            ),
            key=lambda item: (item.source_id, item.source_record_id),
        )
    )
    candidate_map = CtgovStudyCandidateMap(
        policy_id=policy.policy_id,
        episode_id=policy.episode_id,
        candidates=tuple(
            sorted(
                (
                    CtgovStudyCandidateMapEntry(
                        candidate_id=item.candidate.candidate_id,
                        nct_id=item.nct_id,
                    )
                    for item in eligible
                ),
                key=lambda item: (item.candidate_id, item.nct_id),
            )
        ),
    )
    return AdapterRunResult(
        candidate_records=_jsonl(candidates, CtgovProductionAdapterError),
        evidence_records=_jsonl(evidence, CtgovProductionAdapterError),
        dispositions=_jsonl(dispositions, CtgovProductionAdapterError),
        auxiliary_outputs={'ctgov-study-candidate-map': canonical_json_bytes(candidate_map)},
    )


def _canonical_source_policy(payload: bytes) -> CtgovSourceVerifierPolicy:
    return _canonical_model(payload, CtgovSourceVerifierPolicy, CtgovProductionSourceError, 'source verifier')


def _canonical_adapter_policy(payload: bytes) -> CtgovStudyAdapterPolicy:
    return _canonical_model(payload, CtgovStudyAdapterPolicy, CtgovProductionAdapterError, 'study adapter')


def _canonical_model[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    error_type: type[ValueError],
    label: str,
) -> ModelT:
    if not isinstance(payload, bytes) or not payload:
        raise error_type(f'ClinicalTrials.gov {label} policy must be nonempty exact bytes')
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise error_type(f'invalid ClinicalTrials.gov {label} policy: {error}') from error
    if payload != canonical_json_bytes(value):
        raise error_type(f'ClinicalTrials.gov {label} policy must use canonical JSON')
    return value


def _artifact_url(layout: CtgovPromotionLayout, artifact_id: str) -> str:
    if artifact_id in {layout.version_before_artifact_id, layout.version_after_artifact_id}:
        return layout.version_url
    query = next((item for item in layout.queries if item.artifact_id == artifact_id), None)
    if query is None:
        raise CtgovProductionSourceError(f'unknown ClinicalTrials.gov artifact ID: {artifact_id}')
    return query.request_url


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
        raise CtgovProductionSourceError(f'invalid ClinicalTrials.gov HTTPS receipt: {error}') from error
    if payload != canonical_json_bytes(receipt):
        raise CtgovProductionSourceError('ClinicalTrials.gov HTTPS receipt is not canonical JSON')
    if (
        receipt.requested_url != expected_url
        or receipt.final_url != expected_url
        or receipt.status_code != 200
        or receipt.body_sha256 != body_sha256
        or receipt.body_byte_count != body_bytes
        or receipt.completed_at > captured_at
    ):
        raise CtgovProductionSourceError('ClinicalTrials.gov receipt differs from its body, URL, or capture')
    peer = receipt.tls_peer
    if (
        peer is None
        or peer.server_name != 'clinicaltrials.gov'
        or peer.certificate_der_sha256 is None
        or peer.tls_version not in accepted_tls_versions
    ):
        raise CtgovProductionSourceError(
            'ClinicalTrials.gov capture lacks collector-reported official-origin TLS peer metadata'
        )
    request_headers = {item.name: item.value for item in receipt.request_headers}
    if request_headers.get('host') != 'clinicaltrials.gov' or request_headers.get('accept-encoding') != 'identity':
        raise CtgovProductionSourceError('ClinicalTrials.gov receipt lacks canonical request headers')
    content_types = [
        value for header in receipt.response_headers if header.name == 'content-type' for value in header.values
    ]
    if len(content_types) != 1 or not content_types[0].lower().startswith('application/json'):
        raise CtgovProductionSourceError('ClinicalTrials.gov response does not bind one JSON content type')
    return receipt


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise CtgovProductionSourceError(f'{label} contains duplicate JSON key {name!r}')
            result[name] = value
        return result

    def reject_constant(value: str) -> None:
        raise CtgovProductionSourceError(f'{label} contains non-finite JSON number {value}')

    try:
        value = json.loads(
            payload.decode('utf-8'),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CtgovProductionSourceError(f'{label} is not strict UTF-8 JSON: {error}') from error
    if not isinstance(value, dict):
        raise CtgovProductionSourceError(f'{label} must contain one JSON object')
    return value


def _version_identity(raw: dict[str, Any]) -> tuple[str, datetime]:
    api_version = raw.get('apiVersion')
    data_timestamp = raw.get('dataTimestamp')
    if not isinstance(api_version, str) or not api_version or not isinstance(data_timestamp, str):
        raise CtgovProductionSourceError('API version response lacks apiVersion or dataTimestamp')
    try:
        timestamp = datetime.fromisoformat(data_timestamp.replace('Z', '+00:00'))
    except ValueError as error:
        raise CtgovProductionSourceError('API version dataTimestamp is not ISO-8601') from error
    if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
        return api_version, timestamp.astimezone(timezone.utc)

    # The production endpoint currently emits a local timestamp with no offset.
    # ClinicalTrials.gov documents its refresh schedule in Eastern Time.  Freeze
    # that interpretation in policy and choose the later valid fold during the
    # autumn DST transition so the derived cutoff is conservative.
    eastern = ZoneInfo('America/New_York')
    candidates: set[datetime] = set()
    for fold in (0, 1):
        localized = timestamp.replace(tzinfo=eastern, fold=fold)
        candidate = localized.astimezone(timezone.utc)
        if candidate.astimezone(eastern).replace(tzinfo=None) == timestamp:
            candidates.add(candidate)
    if not candidates:
        raise CtgovProductionSourceError('API version dataTimestamp is a nonexistent America/New_York local time')
    return api_version, max(candidates)


def _complete_study_page(page: dict[str, Any], query_id: str) -> tuple[dict[str, Any], ...]:
    if set(page) - {'studies', 'nextPageToken', 'totalCount'}:
        raise CtgovProductionSourceError(f'query partition {query_id} contains unknown top-level fields')
    studies = page.get('studies')
    total_count = page.get('totalCount')
    next_token = page.get('nextPageToken')
    if not isinstance(studies, list) or any(not isinstance(item, dict) for item in studies):
        raise CtgovProductionSourceError(f'query partition {query_id} studies must be an object array')
    if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count != len(studies):
        raise CtgovProductionSourceError(f'query partition {query_id} totalCount does not prove completeness')
    if next_token not in (None, ''):
        raise CtgovProductionSourceError(f'query partition {query_id} requires an uncommitted continuation page')
    return tuple(studies)


def _study_nct_id(study: dict[str, Any]) -> str:
    protocol = study.get('protocolSection')
    if not isinstance(protocol, dict):
        raise CtgovProductionSourceError('ClinicalTrials.gov protocolSection must be an object')
    identification = protocol.get('identificationModule')
    if not isinstance(identification, dict):
        raise CtgovProductionSourceError('ClinicalTrials.gov identificationModule must be an object')
    nct_id = identification.get('nctId')
    if not isinstance(nct_id, str) or _NCT_ID_RE.fullmatch(nct_id) is None:
        raise CtgovProductionSourceError('ClinicalTrials.gov study lacks a canonical NCT ID')
    return nct_id


def _append_record(
    records: list[SourceRecordBinding],
    *,
    source_id: str,
    record_id: str,
    raw: dict[str, Any],
    artifact_sha256: str,
    locator: str,
) -> None:
    if any(item.source_record_id == record_id for item in records):
        raise CtgovProductionSourceError(f'duplicate ClinicalTrials.gov source record ID: {record_id}')
    records.append(
        SourceRecordBinding(
            source_id=source_id,
            source_record_id=record_id,
            source_record_sha256=hashlib.sha256(canonical_json_bytes(raw)).hexdigest(),
            source_artifact_sha256=artifact_sha256,
            source_locator=locator,
        )
    )


def _rebind_rows(source_input: AdapterSourceInput) -> dict[tuple[str, str], dict[str, Any]]:
    row_index_by_artifact: dict[str, dict[str, dict[str, Any]]] = {}
    for capture in source_input.captures:
        for artifact in capture.artifacts:
            if not artifact.role.startswith('body.'):
                continue
            payload = artifact.payload
            digest = hashlib.sha256(payload).hexdigest()
            if digest != artifact.sha256:
                raise CtgovProductionAdapterError('ClinicalTrials.gov body differs from promoted bytes')
            if digest in row_index_by_artifact:
                continue
            try:
                root = _strict_json_object(payload, f'captured body {digest}')
            except CtgovProductionSourceError as error:
                raise CtgovProductionAdapterError(str(error)) from error
            rows = [root]
            studies = root.get('studies')
            if isinstance(studies, list):
                rows.extend(item for item in studies if isinstance(item, dict))
            row_index: dict[str, dict[str, Any]] = {}
            for raw in rows:
                row_sha256 = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
                if row_sha256 in row_index:
                    raise CtgovProductionAdapterError('captured body contains duplicate canonical rows')
                row_index[row_sha256] = raw
            row_index_by_artifact[digest] = row_index
    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for source in source_input.verified_records:
        raw = row_index_by_artifact.get(source.source_artifact_sha256, {}).get(source.source_record_sha256)
        if raw is None:
            raise CtgovProductionAdapterError(
                f'ClinicalTrials.gov row {source.source_record_id!r} cannot be rebound to captured bytes'
            )
        resolved[(source.source_id, source.source_record_id)] = raw
    return resolved


def _protocol_projection(study: dict[str, Any]) -> dict[str, Any]:
    """Return an explicit protocol allowlist without touching results-bearing keys."""

    protocol = _mapping(study.get('protocolSection'), 'protocolSection')
    identification = _mapping(protocol.get('identificationModule'), 'identificationModule')
    design = _mapping(protocol.get('designModule'), 'designModule')
    status = _mapping(protocol.get('statusModule'), 'statusModule')
    conditions = _mapping(protocol.get('conditionsModule'), 'conditionsModule')
    arms = _mapping(protocol.get('armsInterventionsModule'), 'armsInterventionsModule')
    outcomes = _mapping(protocol.get('outcomesModule'), 'outcomesModule')

    nct_id = identification.get('nctId')
    brief_title = identification.get('briefTitle')
    study_type = design.get('studyType')
    overall_status = status.get('overallStatus')
    if (
        not isinstance(nct_id, str)
        or _NCT_ID_RE.fullmatch(nct_id) is None
        or not isinstance(brief_title, str)
        or not brief_title.strip()
        or not isinstance(study_type, str)
        or not isinstance(overall_status, str)
    ):
        raise CtgovProductionAdapterError('study lacks required protocol identity/status fields')

    raw_phases = design.get('phases', [])
    if not isinstance(raw_phases, list) or any(not isinstance(item, str) for item in raw_phases):
        raise CtgovProductionAdapterError('study phases are malformed')
    phases = tuple(sorted(set(raw_phases))) or ('NA',)
    raw_interventions = arms.get('interventions')
    if not isinstance(raw_interventions, list) or not raw_interventions:
        raise CtgovProductionAdapterError('study lacks intervention definitions')
    interventions: list[dict[str, str]] = []
    for intervention in raw_interventions:
        item = _mapping(intervention, 'intervention')
        intervention_type = item.get('type')
        name = item.get('name')
        if not isinstance(intervention_type, str) or not isinstance(name, str) or not name.strip():
            raise CtgovProductionAdapterError('study intervention is missing type or name')
        interventions.append({'name': _clean_text(name), 'type': intervention_type})

    raw_conditions = conditions.get('conditions', [])
    if not isinstance(raw_conditions, list) or any(not isinstance(item, str) for item in raw_conditions):
        raise CtgovProductionAdapterError('study conditions are malformed')
    raw_primary_outcomes = outcomes.get('primaryOutcomes', [])
    if not isinstance(raw_primary_outcomes, list):
        raise CtgovProductionAdapterError('study primary outcomes are malformed')
    primary_outcomes: list[dict[str, str]] = []
    for outcome in raw_primary_outcomes:
        item = _mapping(outcome, 'primary outcome')
        measure = item.get('measure')
        time_frame = item.get('timeFrame')
        if not isinstance(measure, str) or not measure.strip():
            raise CtgovProductionAdapterError('primary outcome lacks a measure')
        primary_outcomes.append(
            {
                'measure': _clean_text(measure),
                'time_frame': _clean_text(time_frame) if isinstance(time_frame, str) else 'not reported',
            }
        )

    enrollment = design.get('enrollmentInfo')
    enrollment_text = 'not reported'
    if enrollment is not None:
        enrollment_info = _mapping(enrollment, 'enrollmentInfo')
        count = enrollment_info.get('count')
        kind = enrollment_info.get('type')
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0 and isinstance(kind, str):
            enrollment_text = f'{count} ({kind})'
    return {
        'brief_title': _clean_text(brief_title),
        'conditions': tuple(sorted({_clean_text(item) for item in raw_conditions})),
        'enrollment': enrollment_text,
        'interventions': tuple(sorted(interventions, key=lambda item: (item['type'], item['name']))),
        'nct_id': nct_id,
        'overall_status': overall_status,
        'phases': phases,
        'primary_outcomes': tuple(sorted(primary_outcomes, key=lambda item: (item['measure'], item['time_frame']))),
        'study_type': study_type,
    }


def _render_protocol_projection(projection: dict[str, Any], candidate_id: str) -> str:
    lines = [
        f'Candidate ID: {candidate_id}.',
        f'NCT ID: {projection["nct_id"]}.',
        f'Brief title: {projection["brief_title"]}.',
        f'Study type: {projection["study_type"]}.',
        f'Phases: {"; ".join(projection["phases"])}.',
        f'Overall status at cutoff: {projection["overall_status"]}.',
        f'Enrollment at cutoff: {projection["enrollment"]}.',
    ]
    if projection['conditions']:
        lines.append(f'Conditions: {"; ".join(projection["conditions"])}.')
    for intervention in projection['interventions']:
        lines.append(f'Intervention: {intervention["type"]} — {intervention["name"]}.')
    for outcome in projection['primary_outcomes']:
        lines.append(f'Protocol primary outcome: {outcome["measure"]}; time frame: {outcome["time_frame"]}.')
    return '\n'.join(lines)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CtgovProductionAdapterError(f'{label} must be an object')
    return value


def _clean_text(value: str) -> str:
    cleaned = ' '.join(value.split())
    if not cleaned:
        raise CtgovProductionAdapterError('protocol text field is empty after normalization')
    return cleaned


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


def _jsonl(
    records: tuple[StrictModel, ...],
    error_type: type[ValueError],
) -> bytes:
    if not records:
        raise error_type('ClinicalTrials.gov record inventory cannot be empty')
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(_SHA256_PATTERN, value) is None:
        raise CtgovProductionSourceError(f'{label} must be a lowercase SHA-256 digest')
