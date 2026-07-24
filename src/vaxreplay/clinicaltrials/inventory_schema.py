"""Strict contracts for a content-addressed AACT clinical-cohort inventory.

The inventory is deliberately separate from episode admission.  It records the complete screening
state for every discovered NCT record, including failed and not-yet-assessed gates, without implying
that a historical replay episode is scientifically valid or safe to distribute.
"""

from __future__ import annotations

import enum
import hashlib
from datetime import date, datetime, time, timedelta, timezone
from pathlib import PurePosixPath
from typing import Literal, Self
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

AACT_OFFICIAL_CATALOG_PAGE_SCHEMA_VERSION = 'vaxreplay.aact-official-catalog-page.v0.1'
AACT_OFFICIAL_CATALOG_ENTRY_SCHEMA_VERSION = 'vaxreplay.aact-official-catalog-entry.v0.1'
AACT_OFFICIAL_CATALOG_SCHEMA_VERSION = 'vaxreplay.aact-official-catalog.v0.1'
AACT_ACQUISITION_ITEM_SCHEMA_VERSION = 'vaxreplay.aact-acquisition-item.v0.1'
AACT_ACQUISITION_PLAN_SCHEMA_VERSION = 'vaxreplay.aact-acquisition-plan.v0.1'
AACT_CANDIDATE_RECORD_SCHEMA_VERSION = 'vaxreplay.aact-candidate-inventory-record.v0.1'
AACT_CANDIDATE_INVENTORY_SCHEMA_VERSION = 'vaxreplay.aact-candidate-inventory.v0.1'
AACT_CATALOG_PARSER_ID = 'aact-official-monthly-flatfile-html-v1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SAFE_ID_PATTERN = r'^[a-z0-9][a-z0-9._-]{0,127}$'
_SNAPSHOT_ID_PATTERN = r'^aact-flatfiles-[0-9]{4}-[0-9]{2}-[0-9]{2}$'
_ARCHIVE_FILE_PATTERN = r'^[0-9]{8}_(?:pipe-delimited-export|export|export_ctgov)\.zip$'
_NCT_ID_PATTERN = r'^NCT[0-9]{8}$'
_OFFICIAL_HOST = 'aact.ctti-clinicaltrials.org'
_OFFICIAL_LISTING_PATH = '/downloads/snapshots'


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _records_sha256(records: tuple[StrictModel, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json_bytes(record))
        digest.update(b'\n')
    return digest.hexdigest()


def aact_inventory_model_sha256(value: StrictModel) -> str:
    """Return the exact canonical SHA-256 used by all inventory cross-bindings."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def aact_inventory_records_sha256(records: tuple[StrictModel, ...]) -> str:
    """Return a newline-framed canonical record-list commitment."""

    return _records_sha256(records)


def _validate_official_listing_url(value: str, year: int) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != 'https'
        or parsed.hostname != _OFFICIAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.path != _OFFICIAL_LISTING_PATH
        or parsed.fragment
        or parse_qsl(parsed.query, keep_blank_values=True) != [('type', 'flatfiles'), ('year', str(year))]
    ):
        raise ValueError('source_url must be the canonical official AACT flat-file listing URL')
    return value


class AactOfficialCatalogPage(StrictModel):
    """Exact receipt for one frozen annual AACT archive-listing HTML page."""

    schema_version: Literal['vaxreplay.aact-official-catalog-page.v0.1'] = AACT_OFFICIAL_CATALOG_PAGE_SCHEMA_VERSION
    year: int = Field(ge=2017, le=2100)
    source_url: str = Field(min_length=1, max_length=2048)
    retrieved_at: datetime
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    payload_bytes: int = Field(gt=0)

    @field_validator('retrieved_at')
    @classmethod
    def validate_retrieved_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'retrieved_at')

    @model_validator(mode='after')
    def validate_source(self) -> Self:
        _validate_official_listing_url(self.source_url, self.year)
        return self


class AactOfficialArchiveEntry(StrictModel):
    """One permanent monthly flat-file archive parsed from an official listing page."""

    schema_version: Literal['vaxreplay.aact-official-catalog-entry.v0.1'] = AACT_OFFICIAL_CATALOG_ENTRY_SCHEMA_VERSION
    snapshot_id: str = Field(pattern=_SNAPSHOT_ID_PATTERN)
    archive_date: date
    source_cutoff_at: datetime
    listing_year: int = Field(ge=2017, le=2100)
    file_name: str = Field(pattern=_ARCHIVE_FILE_PATTERN)
    file_name_date: date
    file_name_date_matches_archive_date: bool
    displayed_size: str = Field(min_length=1, max_length=64)
    download_path: str = Field(min_length=1, max_length=2048)
    source_url: str = Field(min_length=1, max_length=2048)
    listing_page_sha256: str = Field(pattern=_SHA256_PATTERN)
    archive_kind: Literal['flatfiles'] = 'flatfiles'
    permanent_monthly_archive: Literal[True] = True

    @field_validator('source_cutoff_at')
    @classmethod
    def validate_source_cutoff_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'source_cutoff_at')

    @model_validator(mode='after')
    def validate_identity(self) -> Self:
        compact_file_date = self.file_name_date.strftime('%Y%m%d')
        iso_date = self.archive_date.isoformat()
        expected_path = f'/static/exported_files/daily/{iso_date}?source=web'
        if self.listing_year != self.archive_date.year:
            raise ValueError('listing_year must equal the archive year')
        if self.file_name_date.year != self.listing_year:
            raise ValueError('archive file-name date must fall within the selected listing year')
        if not self.file_name.startswith(f'{compact_file_date}_'):
            raise ValueError('file_name_date must equal the date encoded in the archive file name')
        if self.file_name_date_matches_archive_date != (self.file_name_date == self.archive_date):
            raise ValueError('file-name/displayed-date match flag is inconsistent')
        if self.snapshot_id != f'aact-flatfiles-{iso_date}':
            raise ValueError('snapshot_id must be derived from archive_date')
        if self.download_path != expected_path:
            raise ValueError('download_path must use the exact official dated AACT route')
        if self.source_url != f'https://{_OFFICIAL_HOST}{expected_path}':
            raise ValueError('source_url must be the canonical absolute official download URL')
        expected_cutoff = datetime.combine(self.archive_date, time.max, tzinfo=timezone.utc)
        if self.source_cutoff_at != expected_cutoff:
            raise ValueError('source_cutoff_at must be the archive-date end-of-day UTC upper bound')
        if self.displayed_size != self.displayed_size.strip():
            raise ValueError('displayed_size must be trimmed')
        return self


class AactOfficialArchiveCatalog(StrictModel):
    """Canonical union of one or more frozen annual official archive listings."""

    schema_version: Literal['vaxreplay.aact-official-catalog.v0.1'] = AACT_OFFICIAL_CATALOG_SCHEMA_VERSION
    catalog_id: str = Field(pattern=_SAFE_ID_PATTERN)
    generated_at: datetime
    parser_id: Literal['aact-official-monthly-flatfile-html-v1'] = AACT_CATALOG_PARSER_ID
    parser_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    pages: tuple[AactOfficialCatalogPage, ...] = Field(min_length=1)
    source_pages_sha256: str = Field(pattern=_SHA256_PATTERN)
    entries: tuple[AactOfficialArchiveEntry, ...] = Field(min_length=1)
    entries_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('generated_at')
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'generated_at')

    @model_validator(mode='after')
    def validate_catalog(self) -> Self:
        years = tuple(page.year for page in self.pages)
        if years != tuple(sorted(set(years))):
            raise ValueError('catalog pages must have unique ascending years')
        if self.generated_at < max(page.retrieved_at for page in self.pages):
            raise ValueError('generated_at cannot precede a listing retrieval')
        if self.source_pages_sha256 != _records_sha256(self.pages):
            raise ValueError('source_pages_sha256 does not match the exact page receipts')

        entry_keys = tuple((entry.archive_date, entry.snapshot_id) for entry in self.entries)
        if entry_keys != tuple(sorted(set(entry_keys))):
            raise ValueError('catalog entries must be unique and sorted by archive date and snapshot ID')
        page_hash_by_year = {page.year: page.payload_sha256 for page in self.pages}
        for entry in self.entries:
            if page_hash_by_year.get(entry.listing_year) != entry.listing_page_sha256:
                raise ValueError('catalog entry is not bound to its frozen annual listing page')
        if self.entries_sha256 != _records_sha256(self.entries):
            raise ValueError('entries_sha256 does not match the exact catalog entries')
        return self


class AactArchiveAcquisitionRole(str, enum.Enum):
    DISCOVERY = 'discovery'
    DECISION_CANDIDATE = 'decision_candidate'
    LABEL_CANDIDATE = 'label_candidate'
    CONFIRMATION_CANDIDATE = 'confirmation_candidate'


class AactArchiveAcquisitionItem(StrictModel):
    """One exact catalog entry requested by an organizer acquisition plan."""

    schema_version: Literal['vaxreplay.aact-acquisition-item.v0.1'] = AACT_ACQUISITION_ITEM_SCHEMA_VERSION
    snapshot_id: str = Field(pattern=_SNAPSHOT_ID_PATTERN)
    archive_date: date
    catalog_entry_sha256: str = Field(pattern=_SHA256_PATTERN)
    roles: tuple[AactArchiveAcquisitionRole, ...] = Field(min_length=1)
    target_relative_path: str = Field(min_length=1, max_length=512)

    @field_validator('target_relative_path')
    @classmethod
    def validate_target_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
            raise ValueError('target_relative_path must be a normalized relative path')
        return value

    @model_validator(mode='after')
    def validate_item(self) -> Self:
        if self.snapshot_id != f'aact-flatfiles-{self.archive_date.isoformat()}':
            raise ValueError('acquisition item snapshot ID must match archive_date')
        role_values = tuple(role.value for role in self.roles)
        if role_values != tuple(sorted(set(role_values))):
            raise ValueError('acquisition roles must be unique and sorted')
        return self


class AactArchiveAcquisitionPlan(StrictModel):
    """Exact, reviewable set of full AACT archives approved for acquisition."""

    schema_version: Literal['vaxreplay.aact-acquisition-plan.v0.1'] = AACT_ACQUISITION_PLAN_SCHEMA_VERSION
    plan_id: str = Field(pattern=_SAFE_ID_PATTERN)
    created_at: datetime
    catalog_sha256: str = Field(pattern=_SHA256_PATTERN)
    catalog_entries_sha256: str = Field(pattern=_SHA256_PATTERN)
    screening_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    items: tuple[AactArchiveAcquisitionItem, ...] = Field(min_length=1)
    items_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'created_at')

    @model_validator(mode='after')
    def validate_plan(self) -> Self:
        item_keys = tuple((item.archive_date, item.snapshot_id) for item in self.items)
        if item_keys != tuple(sorted(set(item_keys))):
            raise ValueError('acquisition items must be unique and sorted by archive date')
        if self.items_sha256 != _records_sha256(self.items):
            raise ValueError('items_sha256 does not match the exact acquisition items')
        discovery_count = sum(AactArchiveAcquisitionRole.DISCOVERY in item.roles for item in self.items)
        if discovery_count != 1:
            raise ValueError('an acquisition plan must contain exactly one discovery archive')
        return self


class AactInventoryGate(str, enum.Enum):
    SOURCE_ARCHIVE = 'source_archive'
    VACCINE_RELEVANCE = 'vaccine_relevance'
    INTERVENTIONAL = 'interventional'
    RANDOMIZED = 'randomized'
    PARALLEL_ASSIGNMENT = 'parallel_assignment'
    PREVENTION_PURPOSE = 'prevention_purpose'
    EARLY_PHASE = 'early_phase'
    MULTI_ARM_PANEL = 'multi_arm_panel'
    BIOLOGICAL_CANDIDATES = 'biological_candidates'
    COMPARATOR = 'comparator'
    PLANNED_IMMUNE_ENDPOINT = 'planned_immune_endpoint'
    RESULTS_POSTED_ACTUAL = 'results_posted_actual'
    REPORTED_IMMUNE_ENDPOINT = 'reported_immune_endpoint'
    NUMERIC_OUTCOME_COVERAGE = 'numeric_outcome_coverage'
    DECISION_SNAPSHOT = 'decision_snapshot'
    ARM_MAPPING = 'arm_mapping'
    ENDPOINT_MAPPING = 'endpoint_mapping'
    LABEL_STABILITY = 'label_stability'
    PRE_CUTOFF_EVIDENCE = 'pre_cutoff_evidence'
    LINEAGE_ADJUDICATION = 'lineage_adjudication'
    REDISTRIBUTION = 'redistribution'
    LEAKAGE_REVIEW = 'leakage_review'


class AactGateStatus(str, enum.Enum):
    PASS = 'pass'
    FAIL = 'fail'
    NOT_ASSESSED = 'not_assessed'


class AactInventoryReason(str, enum.Enum):
    GATE_NOT_ASSESSED = 'GATE_NOT_ASSESSED'
    SOURCE_ARCHIVE_UNVERIFIED = 'SOURCE_ARCHIVE_UNVERIFIED'
    VACCINE_RELEVANCE_NOT_ADJUDICATED = 'VACCINE_RELEVANCE_NOT_ADJUDICATED'
    NOT_VACCINE_RELEVANT = 'NOT_VACCINE_RELEVANT'
    NOT_INTERVENTIONAL = 'NOT_INTERVENTIONAL'
    NOT_RANDOMIZED = 'NOT_RANDOMIZED'
    NOT_PARALLEL_ASSIGNMENT = 'NOT_PARALLEL_ASSIGNMENT'
    NOT_PREVENTION_PURPOSE = 'NOT_PREVENTION_PURPOSE'
    NOT_EARLY_PHASE = 'NOT_EARLY_PHASE'
    DESIGN_GROUP_COUNT_LT_3 = 'DESIGN_GROUP_COUNT_LT_3'
    EXPERIMENTAL_GROUP_COUNT_LT_2 = 'EXPERIMENTAL_GROUP_COUNT_LT_2'
    BIOLOGICAL_LINKED_GROUP_COUNT_LT_2 = 'BIOLOGICAL_LINKED_GROUP_COUNT_LT_2'
    COMPARATOR_GROUP_MISSING = 'COMPARATOR_GROUP_MISSING'
    PLANNED_IMMUNE_ENDPOINT_MISSING = 'PLANNED_IMMUNE_ENDPOINT_MISSING'
    RESULTS_FIRST_POST_DATE_MISSING = 'RESULTS_FIRST_POST_DATE_MISSING'
    RESULTS_FIRST_POST_DATE_ESTIMATED = 'RESULTS_FIRST_POST_DATE_ESTIMATED'
    REPORTED_IMMUNE_ENDPOINT_MISSING = 'REPORTED_IMMUNE_ENDPOINT_MISSING'
    NUMERIC_OUTCOME_COVERAGE_INCOMPLETE = 'NUMERIC_OUTCOME_COVERAGE_INCOMPLETE'
    PRE_ENROLLMENT_SNAPSHOT_MISSING = 'PRE_ENROLLMENT_SNAPSHOT_MISSING'
    PRE_RESULTS_SNAPSHOT_MISSING = 'PRE_RESULTS_SNAPSHOT_MISSING'
    ARM_MAPPING_NOT_ASSESSED = 'ARM_MAPPING_NOT_ASSESSED'
    ARM_MAPPING_AMBIGUOUS = 'ARM_MAPPING_AMBIGUOUS'
    ENDPOINT_MAPPING_NOT_ASSESSED = 'ENDPOINT_MAPPING_NOT_ASSESSED'
    ENDPOINTS_INCOMPARABLE = 'ENDPOINTS_INCOMPARABLE'
    LABEL_STABILITY_NOT_ASSESSED = 'LABEL_STABILITY_NOT_ASSESSED'
    LABEL_STABILITY_FAILED = 'LABEL_STABILITY_FAILED'
    PRE_CUTOFF_EVIDENCE_NOT_ASSESSED = 'PRE_CUTOFF_EVIDENCE_NOT_ASSESSED'
    PRE_CUTOFF_EVIDENCE_NOT_DISCRIMINATIVE = 'PRE_CUTOFF_EVIDENCE_NOT_DISCRIMINATIVE'
    LINEAGE_NOT_ADJUDICATED = 'LINEAGE_NOT_ADJUDICATED'
    REDISTRIBUTION_NOT_CLEARED = 'REDISTRIBUTION_NOT_CLEARED'
    LEAKAGE_REVIEW_NOT_ASSESSED = 'LEAKAGE_REVIEW_NOT_ASSESSED'
    LEAKAGE_REVIEW_FAILED = 'LEAKAGE_REVIEW_FAILED'


_REASONS_BY_GATE: dict[AactInventoryGate, frozenset[AactInventoryReason]] = {
    AactInventoryGate.SOURCE_ARCHIVE: frozenset({AactInventoryReason.SOURCE_ARCHIVE_UNVERIFIED}),
    AactInventoryGate.VACCINE_RELEVANCE: frozenset(
        {
            AactInventoryReason.VACCINE_RELEVANCE_NOT_ADJUDICATED,
            AactInventoryReason.NOT_VACCINE_RELEVANT,
        }
    ),
    AactInventoryGate.INTERVENTIONAL: frozenset({AactInventoryReason.NOT_INTERVENTIONAL}),
    AactInventoryGate.RANDOMIZED: frozenset({AactInventoryReason.NOT_RANDOMIZED}),
    AactInventoryGate.PARALLEL_ASSIGNMENT: frozenset({AactInventoryReason.NOT_PARALLEL_ASSIGNMENT}),
    AactInventoryGate.PREVENTION_PURPOSE: frozenset({AactInventoryReason.NOT_PREVENTION_PURPOSE}),
    AactInventoryGate.EARLY_PHASE: frozenset({AactInventoryReason.NOT_EARLY_PHASE}),
    AactInventoryGate.MULTI_ARM_PANEL: frozenset(
        {
            AactInventoryReason.DESIGN_GROUP_COUNT_LT_3,
            AactInventoryReason.EXPERIMENTAL_GROUP_COUNT_LT_2,
        }
    ),
    AactInventoryGate.BIOLOGICAL_CANDIDATES: frozenset({AactInventoryReason.BIOLOGICAL_LINKED_GROUP_COUNT_LT_2}),
    AactInventoryGate.COMPARATOR: frozenset({AactInventoryReason.COMPARATOR_GROUP_MISSING}),
    AactInventoryGate.PLANNED_IMMUNE_ENDPOINT: frozenset({AactInventoryReason.PLANNED_IMMUNE_ENDPOINT_MISSING}),
    AactInventoryGate.RESULTS_POSTED_ACTUAL: frozenset(
        {
            AactInventoryReason.RESULTS_FIRST_POST_DATE_MISSING,
            AactInventoryReason.RESULTS_FIRST_POST_DATE_ESTIMATED,
        }
    ),
    AactInventoryGate.REPORTED_IMMUNE_ENDPOINT: frozenset({AactInventoryReason.REPORTED_IMMUNE_ENDPOINT_MISSING}),
    AactInventoryGate.NUMERIC_OUTCOME_COVERAGE: frozenset({AactInventoryReason.NUMERIC_OUTCOME_COVERAGE_INCOMPLETE}),
    AactInventoryGate.DECISION_SNAPSHOT: frozenset(
        {
            AactInventoryReason.PRE_ENROLLMENT_SNAPSHOT_MISSING,
            AactInventoryReason.PRE_RESULTS_SNAPSHOT_MISSING,
        }
    ),
    AactInventoryGate.ARM_MAPPING: frozenset(
        {AactInventoryReason.ARM_MAPPING_NOT_ASSESSED, AactInventoryReason.ARM_MAPPING_AMBIGUOUS}
    ),
    AactInventoryGate.ENDPOINT_MAPPING: frozenset(
        {AactInventoryReason.ENDPOINT_MAPPING_NOT_ASSESSED, AactInventoryReason.ENDPOINTS_INCOMPARABLE}
    ),
    AactInventoryGate.LABEL_STABILITY: frozenset(
        {AactInventoryReason.LABEL_STABILITY_NOT_ASSESSED, AactInventoryReason.LABEL_STABILITY_FAILED}
    ),
    AactInventoryGate.PRE_CUTOFF_EVIDENCE: frozenset(
        {
            AactInventoryReason.PRE_CUTOFF_EVIDENCE_NOT_ASSESSED,
            AactInventoryReason.PRE_CUTOFF_EVIDENCE_NOT_DISCRIMINATIVE,
        }
    ),
    AactInventoryGate.LINEAGE_ADJUDICATION: frozenset({AactInventoryReason.LINEAGE_NOT_ADJUDICATED}),
    AactInventoryGate.REDISTRIBUTION: frozenset({AactInventoryReason.REDISTRIBUTION_NOT_CLEARED}),
    AactInventoryGate.LEAKAGE_REVIEW: frozenset(
        {AactInventoryReason.LEAKAGE_REVIEW_NOT_ASSESSED, AactInventoryReason.LEAKAGE_REVIEW_FAILED}
    ),
}


class AactCandidateDisposition(str, enum.Enum):
    HOLD = 'hold'
    EXCLUDE = 'exclude'
    ADMIT_PRE_ENROLLMENT = 'admit_pre_enrollment'
    ADMIT_PRE_RESULTS_DIAGNOSTIC = 'admit_pre_results_diagnostic'


class AactDecisionClass(str, enum.Enum):
    NOT_SELECTED = 'not_selected'
    PRE_ENROLLMENT = 'pre_enrollment'
    PRE_RESULTS_DIAGNOSTIC = 'pre_results_diagnostic'


class AactPostedDateType(str, enum.Enum):
    ACTUAL = 'actual'
    ESTIMATED = 'estimated'
    MISSING = 'missing'


class AactGateResult(StrictModel):
    gate: AactInventoryGate
    status: AactGateStatus
    reason_codes: tuple[AactInventoryReason, ...] = ()
    evidence_sha256: tuple[str, ...] = ()

    @field_validator('evidence_sha256')
    @classmethod
    def validate_evidence_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))) or any(
            len(item) != 64 or any(character not in '0123456789abcdef' for character in item) for item in value
        ):
            raise ValueError('gate evidence hashes must be unique sorted lowercase SHA-256 values')
        return value

    @model_validator(mode='after')
    def validate_result(self) -> Self:
        reason_values = tuple(reason.value for reason in self.reason_codes)
        if reason_values != tuple(sorted(set(reason_values))):
            raise ValueError('gate reason codes must be unique and sorted')
        if self.status == AactGateStatus.PASS and self.reason_codes:
            raise ValueError('passing gates cannot declare failure or hold reasons')
        if self.status != AactGateStatus.PASS and not self.reason_codes:
            raise ValueError('failed or unassessed gates require an explicit reason')
        if self.status == AactGateStatus.FAIL and AactInventoryReason.GATE_NOT_ASSESSED in self.reason_codes:
            raise ValueError('failed gates cannot use the generic not-assessed reason')
        invalid = set(self.reason_codes) - (_REASONS_BY_GATE[self.gate] | {AactInventoryReason.GATE_NOT_ASSESSED})
        if invalid:
            invalid_values = sorted(item.value for item in invalid)
            raise ValueError(f'reason codes are not valid for gate {self.gate.value}: {invalid_values}')
        if self.status == AactGateStatus.NOT_ASSESSED:
            if self.evidence_sha256:
                raise ValueError('unassessed gates cannot claim reviewed evidence')
        elif not self.evidence_sha256:
            raise ValueError('passing or reviewed gates require exact evidence hashes')
        return self


class AactScreenCounts(StrictModel):
    design_group_count: int = Field(ge=0)
    experimental_group_count: int = Field(ge=0)
    comparator_like_group_count: int = Field(ge=0)
    biological_linked_group_count: int = Field(ge=0)
    planned_immune_endpoint_count: int = Field(ge=0)
    reported_immune_endpoint_count: int = Field(ge=0)
    outcome_result_group_count: int = Field(ge=0)
    numeric_outcome_group_count: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        for field_name in (
            'experimental_group_count',
            'comparator_like_group_count',
            'biological_linked_group_count',
        ):
            if getattr(self, field_name) > self.design_group_count:
                raise ValueError(f'{field_name} cannot exceed design_group_count')
        if self.numeric_outcome_group_count > self.outcome_result_group_count:
            raise ValueError('numeric_outcome_group_count cannot exceed outcome_result_group_count')
        return self


class AactCandidateChronology(StrictModel):
    study_first_posted_date: date
    start_date_lower_bound: date | None = None
    start_date_lower_bound_source_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    later_actual_start_date: date | None = None
    later_actual_start_date_source_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    results_first_posted_date: date | None = None
    results_first_posted_date_type: AactPostedDateType
    results_first_posted_date_source_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_source_bound_dates(self) -> Self:
        if (self.start_date_lower_bound is None) != (self.start_date_lower_bound_source_sha256 is None):
            raise ValueError('start-date lower bound and exact source hash must be present together')
        if (self.later_actual_start_date is None) != (self.later_actual_start_date_source_sha256 is None):
            raise ValueError('later actual-start date and exact source hash must be present together')
        if self.results_first_posted_date_type == AactPostedDateType.MISSING:
            if self.results_first_posted_date is not None or self.results_first_posted_date_source_sha256 is not None:
                raise ValueError('missing results date type requires a missing results date and source hash')
        elif self.results_first_posted_date is None or self.results_first_posted_date_source_sha256 is None:
            raise ValueError('actual or estimated results date requires an exact source hash')
        return self


def _normalized_aact_value(value: str) -> str:
    return ' '.join(value.strip().casefold().split())


def aact_mechanical_gate_expectations(
    *,
    study_type: str,
    allocation: str,
    intervention_model: str,
    primary_purpose: str,
    phase: str,
    counts: AactScreenCounts,
    chronology: AactCandidateChronology,
) -> dict[AactInventoryGate, tuple[AactGateStatus, tuple[AactInventoryReason, ...]]]:
    """Derive every objective screening gate from normalized decision-record facts."""

    def binary(
        passed: bool,
        reason: AactInventoryReason,
    ) -> tuple[AactGateStatus, tuple[AactInventoryReason, ...]]:
        return (AactGateStatus.PASS, ()) if passed else (AactGateStatus.FAIL, (reason,))

    multi_arm_reasons: list[AactInventoryReason] = []
    if counts.design_group_count < 3:
        multi_arm_reasons.append(AactInventoryReason.DESIGN_GROUP_COUNT_LT_3)
    if counts.experimental_group_count < 2:
        multi_arm_reasons.append(AactInventoryReason.EXPERIMENTAL_GROUP_COUNT_LT_2)
    multi_arm = (
        (AactGateStatus.FAIL, tuple(sorted(multi_arm_reasons, key=lambda item: item.value)))
        if multi_arm_reasons
        else (AactGateStatus.PASS, ())
    )

    if chronology.results_first_posted_date_type == AactPostedDateType.ACTUAL:
        results_posted = (AactGateStatus.PASS, ())
    elif chronology.results_first_posted_date_type == AactPostedDateType.ESTIMATED:
        results_posted = (AactGateStatus.FAIL, (AactInventoryReason.RESULTS_FIRST_POST_DATE_ESTIMATED,))
    else:
        results_posted = (AactGateStatus.FAIL, (AactInventoryReason.RESULTS_FIRST_POST_DATE_MISSING,))

    normalized_phase = _normalized_aact_value(phase)
    return {
        AactInventoryGate.INTERVENTIONAL: binary(
            _normalized_aact_value(study_type) == 'interventional',
            AactInventoryReason.NOT_INTERVENTIONAL,
        ),
        AactInventoryGate.RANDOMIZED: binary(
            _normalized_aact_value(allocation) == 'randomized',
            AactInventoryReason.NOT_RANDOMIZED,
        ),
        AactInventoryGate.PARALLEL_ASSIGNMENT: binary(
            _normalized_aact_value(intervention_model) == 'parallel assignment',
            AactInventoryReason.NOT_PARALLEL_ASSIGNMENT,
        ),
        AactInventoryGate.PREVENTION_PURPOSE: binary(
            _normalized_aact_value(primary_purpose) == 'prevention',
            AactInventoryReason.NOT_PREVENTION_PURPOSE,
        ),
        AactInventoryGate.EARLY_PHASE: binary(
            normalized_phase in {'early phase 1', 'phase 1', 'phase 1/phase 2'},
            AactInventoryReason.NOT_EARLY_PHASE,
        ),
        AactInventoryGate.MULTI_ARM_PANEL: multi_arm,
        AactInventoryGate.BIOLOGICAL_CANDIDATES: binary(
            counts.biological_linked_group_count >= 2,
            AactInventoryReason.BIOLOGICAL_LINKED_GROUP_COUNT_LT_2,
        ),
        AactInventoryGate.COMPARATOR: binary(
            counts.comparator_like_group_count >= 1,
            AactInventoryReason.COMPARATOR_GROUP_MISSING,
        ),
        AactInventoryGate.PLANNED_IMMUNE_ENDPOINT: binary(
            counts.planned_immune_endpoint_count >= 1,
            AactInventoryReason.PLANNED_IMMUNE_ENDPOINT_MISSING,
        ),
        AactInventoryGate.RESULTS_POSTED_ACTUAL: results_posted,
        AactInventoryGate.REPORTED_IMMUNE_ENDPOINT: binary(
            counts.reported_immune_endpoint_count >= 1,
            AactInventoryReason.REPORTED_IMMUNE_ENDPOINT_MISSING,
        ),
        AactInventoryGate.NUMERIC_OUTCOME_COVERAGE: binary(
            counts.outcome_result_group_count > 0
            and counts.numeric_outcome_group_count == counts.outcome_result_group_count,
            AactInventoryReason.NUMERIC_OUTCOME_COVERAGE_INCOMPLETE,
        ),
    }


class AactInventorySnapshotBinding(StrictModel):
    snapshot_id: str = Field(pattern=_SNAPSHOT_ID_PATTERN)
    archive_date: date
    catalog_entry_sha256: str = Field(pattern=_SHA256_PATTERN)
    slice_receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_snapshot(self) -> Self:
        if self.snapshot_id != f'aact-flatfiles-{self.archive_date.isoformat()}':
            raise ValueError('snapshot binding ID must match archive_date')
        return self


class AactCandidateInventoryRecord(StrictModel):
    """Organizer-private, exhaustive gate record for one NCT candidate."""

    schema_version: Literal['vaxreplay.aact-candidate-inventory-record.v0.1'] = AACT_CANDIDATE_RECORD_SCHEMA_VERSION
    inventory_id: str = Field(pattern=_SAFE_ID_PATTERN)
    nct_id: str = Field(pattern=_NCT_ID_PATTERN)
    synthetic: Literal[False] = False
    discovery_snapshot: AactInventorySnapshotBinding
    chronology: AactCandidateChronology
    study_type: str = Field(min_length=1, max_length=80)
    allocation: str = Field(min_length=1, max_length=80)
    intervention_model: str = Field(min_length=1, max_length=80)
    primary_purpose: str = Field(min_length=1, max_length=80)
    phase: str = Field(min_length=1, max_length=80)
    counts: AactScreenCounts
    decision_class: AactDecisionClass = AactDecisionClass.NOT_SELECTED
    decision_snapshot: AactInventorySnapshotBinding | None = None
    label_snapshot: AactInventorySnapshotBinding | None = None
    confirmation_snapshot: AactInventorySnapshotBinding | None = None
    value_hidden_mapping_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    label_stability_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    pre_cutoff_evidence_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    lineage_group_id: str | None = Field(default=None, pattern=_SAFE_ID_PATTERN)
    lineage_adjudication_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    gate_results: tuple[AactGateResult, ...] = Field(min_length=len(AactInventoryGate))
    reason_codes: tuple[AactInventoryReason, ...] = ()
    disposition: AactCandidateDisposition

    @model_validator(mode='after')
    def validate_record(self) -> Self:
        gate_values = tuple(result.gate.value for result in self.gate_results)
        expected_gates = tuple(sorted(gate.value for gate in AactInventoryGate))
        if gate_values != expected_gates:
            raise ValueError('gate_results must contain every inventory gate exactly once in sorted order')
        result_by_gate = {result.gate: result for result in self.gate_results}
        mechanical_expectations = aact_mechanical_gate_expectations(
            study_type=self.study_type,
            allocation=self.allocation,
            intervention_model=self.intervention_model,
            primary_purpose=self.primary_purpose,
            phase=self.phase,
            counts=self.counts,
            chronology=self.chronology,
        )
        for gate, (expected_status, expected_reason_codes) in mechanical_expectations.items():
            result = result_by_gate[gate]
            if result.status != expected_status or result.reason_codes != expected_reason_codes:
                raise ValueError(
                    f'{gate.value} gate result is inconsistent with the normalized study fields and counts'
                )
        expected_reasons = tuple(
            sorted(
                {reason for result in self.gate_results for reason in result.reason_codes}, key=lambda item: item.value
            )
        )
        if self.reason_codes != expected_reasons:
            raise ValueError('reason_codes must equal the sorted union of gate-result reasons')

        failed = any(result.status == AactGateStatus.FAIL for result in self.gate_results)
        unassessed = any(result.status == AactGateStatus.NOT_ASSESSED for result in self.gate_results)
        admitted = self.disposition in {
            AactCandidateDisposition.ADMIT_PRE_ENROLLMENT,
            AactCandidateDisposition.ADMIT_PRE_RESULTS_DIAGNOSTIC,
        }
        if self.disposition == AactCandidateDisposition.EXCLUDE:
            if not failed:
                raise ValueError('excluded candidates require at least one failed gate')
        elif self.disposition == AactCandidateDisposition.HOLD:
            if failed or not unassessed:
                raise ValueError('held candidates require unassessed gates and cannot contain failed gates')
        elif failed or unassessed:
            raise ValueError('admitted candidates require every gate to pass')

        bindings = (self.decision_snapshot, self.label_snapshot, self.confirmation_snapshot)
        if (self.decision_class == AactDecisionClass.NOT_SELECTED) != (self.decision_snapshot is None):
            raise ValueError('decision_class and decision_snapshot must be selected together')
        if self.label_snapshot is not None and self.decision_snapshot is None:
            raise ValueError('a label snapshot requires a decision snapshot')
        if self.confirmation_snapshot is not None and self.label_snapshot is None:
            raise ValueError('a confirmation snapshot requires a label snapshot')
        if self.decision_snapshot is not None:
            if self.decision_class == AactDecisionClass.PRE_ENROLLMENT:
                lower_bound = self.chronology.start_date_lower_bound
                later_actual = self.chronology.later_actual_start_date
                if lower_bound is None or later_actual is None:
                    raise ValueError(
                        'pre-enrollment decision requires source-bound decision-time and later actual-start dates'
                    )
                conservative_lower_bound = min(lower_bound, later_actual)
                if self.decision_snapshot.archive_date >= conservative_lower_bound:
                    raise ValueError(
                        'pre-enrollment decision snapshot must strictly predate the start lower bound after '
                        'conservative actual-start adjudication'
                    )
            elif self.decision_class == AactDecisionClass.PRE_RESULTS_DIAGNOSTIC:
                results_date = self.chronology.results_first_posted_date
                if (
                    self.chronology.results_first_posted_date_type != AactPostedDateType.ACTUAL
                    or results_date is None
                    or self.decision_snapshot.archive_date >= results_date
                ):
                    raise ValueError('pre-results decision snapshot must strictly predate an actual results post date')
        if admitted:
            if any(binding is None or binding.slice_receipt_sha256 is None for binding in bindings):
                raise ValueError('admitted candidates require exact decision, label, and confirmation slices')
            required_commitments = (
                self.value_hidden_mapping_sha256,
                self.label_stability_sha256,
                self.pre_cutoff_evidence_sha256,
                self.lineage_group_id,
                self.lineage_adjudication_sha256,
            )
            if any(value is None for value in required_commitments):
                raise ValueError('admitted candidates require mapping, evidence, stability, and lineage commitments')
        if self.disposition == AactCandidateDisposition.ADMIT_PRE_ENROLLMENT:
            if self.decision_class != AactDecisionClass.PRE_ENROLLMENT:
                raise ValueError('pre-enrollment admission requires a pre-enrollment decision class')
        elif self.disposition == AactCandidateDisposition.ADMIT_PRE_RESULTS_DIAGNOSTIC:
            if self.decision_class != AactDecisionClass.PRE_RESULTS_DIAGNOSTIC:
                raise ValueError('pre-results admission requires a diagnostic decision class')
        if self.decision_snapshot is not None and self.label_snapshot is not None:
            if self.decision_snapshot.archive_date >= self.label_snapshot.archive_date:
                raise ValueError('decision snapshot must strictly predate the label snapshot')
        if (
            self.label_snapshot is not None
            and self.chronology.results_first_posted_date_type == AactPostedDateType.ACTUAL
            and self.chronology.results_first_posted_date is not None
            and self.label_snapshot.archive_date < self.chronology.results_first_posted_date
        ):
            raise ValueError('label snapshot cannot predate the source-bound actual results posting')
        if self.label_snapshot is not None and self.confirmation_snapshot is not None:
            minimum_confirmation_date = self.label_snapshot.archive_date + timedelta(days=90)
            if self.confirmation_snapshot.archive_date < minimum_confirmation_date:
                raise ValueError('confirmation snapshot must be at least 90 days after the label snapshot')
        return self


class AactCandidateInventory(StrictModel):
    """Canonical organizer-private case universe before benchmark split assignment."""

    schema_version: Literal['vaxreplay.aact-candidate-inventory.v0.1'] = AACT_CANDIDATE_INVENTORY_SCHEMA_VERSION
    inventory_id: str = Field(pattern=_SAFE_ID_PATTERN)
    created_at: datetime
    catalog_sha256: str = Field(pattern=_SHA256_PATTERN)
    acquisition_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    screening_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    gate_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    masking_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    records: tuple[AactCandidateInventoryRecord, ...] = Field(min_length=1)
    record_count: int = Field(gt=0)
    records_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'created_at')

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        nct_ids = tuple(record.nct_id for record in self.records)
        if nct_ids != tuple(sorted(set(nct_ids))):
            raise ValueError('candidate inventory records must be unique and sorted by NCT ID')
        if any(record.inventory_id != self.inventory_id for record in self.records):
            raise ValueError('candidate records must reference the enclosing inventory ID')
        if self.record_count != len(self.records):
            raise ValueError('record_count does not match the candidate record inventory')
        if self.records_sha256 != _records_sha256(self.records):
            raise ValueError('records_sha256 does not match the exact candidate records')
        return self


def make_unassessed_gate_results() -> tuple[AactGateResult, ...]:
    """Create a complete conservative gate vector for a newly discovered candidate."""

    return tuple(
        AactGateResult(
            gate=gate,
            status=AactGateStatus.NOT_ASSESSED,
            reason_codes=(AactInventoryReason.GATE_NOT_ASSESSED,),
        )
        for gate in sorted(AactInventoryGate, key=lambda item: item.value)
    )
