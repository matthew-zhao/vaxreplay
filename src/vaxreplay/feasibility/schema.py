"""Strict schemas for the ImmPort and ClinicalTrials.gov feasibility inventory."""

from __future__ import annotations

import enum
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.case_schema import StrictModel

FEASIBILITY_RECEIPT_VERSION = 'vaxreplay.feasibility_receipt.v0.1'
FEASIBILITY_SPEC_VERSION = 'vaxreplay.feasibility_spec.v0.1'
FEASIBILITY_REPORT_VERSION = 'vaxreplay.feasibility_report.v0.1'
FEASIBILITY_RECORDS_VERSION = 'vaxreplay.feasibility_records.v0.1'
PUBLIC_FEASIBILITY_SUMMARY_VERSION = 'vaxreplay.public_feasibility_summary.v0.1'
FEASIBILITY_ADAPTER_ID = 'immport-ctgov-feasibility-v0.1'
PUBLIC_SMALL_CELL_THRESHOLD = 5
PUBLIC_MINIMUM_REAL_RECORD_COUNT = 20
PUBLIC_COMPLEMENTARY_GROUPS = (
    ('exact_link_study_count', 'missing_link_count', 'ambiguous_link_count'),
    ('exact_link_study_count', 'matched_study_count', 'missing_ctgov_record_count'),
    (
        'matched_current_results_unique_nct_count',
        'matched_current_results_post_date_actual_count',
        'matched_current_results_post_date_estimated_count',
    ),
    (
        'matched_unique_nct_count',
        'matched_historical_post_date_actual_count',
        'matched_historical_post_date_estimated_count',
    ),
)
PUBLIC_DERIVED_DIFFERENCE_PAIRS = (('matched_unique_nct_count', 'matched_interventional_count'),)
PUBLIC_DERIVED_RESIDUAL_TRIPLES = (
    (
        'matched_current_results_unique_nct_count',
        'matched_current_results_post_date_actual_count',
        'matched_current_results_post_date_estimated_count',
    ),
    (
        'matched_unique_nct_count',
        'matched_historical_post_date_actual_count',
        'matched_historical_post_date_estimated_count',
    ),
)


class SnapshotSource(str, enum.Enum):
    IMMPORT = 'immport'
    CLINICALTRIALS_GOV = 'clinicaltrials.gov'


class VaccineRelevance(str, enum.Enum):
    SOURCE_FILTER = 'source_filter'
    CURATOR_CONFIRMED = 'curator_confirmed'
    REJECTED = 'rejected'


class HistorySurface(str, enum.Enum):
    SUPPORTED_API = 'supported_api'
    PUBLIC_UI_INTERNAL = 'public_ui_internal'
    EXTERNAL_ARCHIVE = 'external_archive'


class PostedDateType(str, enum.Enum):
    ACTUAL = 'ACTUAL'
    ESTIMATED = 'ESTIMATED'


class StudyType(str, enum.Enum):
    INTERVENTIONAL = 'INTERVENTIONAL'
    OBSERVATIONAL = 'OBSERVATIONAL'
    EXPANDED_ACCESS = 'EXPANDED_ACCESS'
    OTHER = 'OTHER'


class LinkStatus(str, enum.Enum):
    EXACT = 'exact'
    MISSING = 'missing'
    AMBIGUOUS = 'ambiguous'
    RECORD_NOT_FOUND = 'record_not_found'


class InventoryDisposition(str, enum.Enum):
    HOLD = 'hold'
    EXCLUDE = 'exclude'


class GateStatus(str, enum.Enum):
    PASS = 'pass'
    FAIL = 'fail'
    NOT_ASSESSED = 'not_assessed'


class AdmissionTier(str, enum.Enum):
    SOURCE_VALID = 'source_valid'
    LINKABLE = 'linkable'
    TEMPORALLY_REPLAYABLE = 'temporally_replayable'
    LABEL_PILOT_READY = 'label_pilot_ready'
    BENCHMARK_READY = 'benchmark_ready'


class InventoryReasonCode(str, enum.Enum):
    NO_EXPLICIT_NCT_LINK = 'NO_EXPLICIT_NCT_LINK'
    MULTIPLE_NCT_LINKS = 'MULTIPLE_NCT_LINKS'
    NCT_RECORD_MISSING = 'NCT_RECORD_MISSING'
    NOT_CLINICAL_TRIAL = 'NOT_CLINICAL_TRIAL'
    NOT_HUMAN = 'NOT_HUMAN'
    NOT_INTERVENTIONAL = 'NOT_INTERVENTIONAL'
    VACCINE_RELEVANCE_NOT_CURATED = 'VACCINE_RELEVANCE_NOT_CURATED'
    VACCINE_RELEVANCE_REJECTED = 'VACCINE_RELEVANCE_REJECTED'
    NO_IMMUNE_ASSAY_METADATA = 'NO_IMMUNE_ASSAY_METADATA'
    IMM_PORT_ARM_COUNT_LT_2 = 'IMM_PORT_ARM_COUNT_LT_2'
    HISTORICAL_ARM_COUNT_LT_2 = 'HISTORICAL_ARM_COUNT_LT_2'
    ARM_COUNT_MISMATCH = 'ARM_COUNT_MISMATCH'
    HISTORY_SURFACE_UNSUPPORTED = 'HISTORY_SURFACE_UNSUPPORTED'
    HISTORICAL_POST_DATE_MISSING = 'HISTORICAL_POST_DATE_MISSING'
    HISTORICAL_POST_DATE_TYPE_MISSING = 'HISTORICAL_POST_DATE_TYPE_MISSING'
    HISTORICAL_POST_DATE_ESTIMATED = 'HISTORICAL_POST_DATE_ESTIMATED'
    HISTORY_CONTAINS_RESULTS = 'HISTORY_CONTAINS_RESULTS'
    HISTORY_NOT_BEFORE_RESULTS = 'HISTORY_NOT_BEFORE_RESULTS'
    RESULTS_FIRST_POST_DATE_TYPE_MISSING = 'RESULTS_FIRST_POST_DATE_TYPE_MISSING'
    RESULTS_FIRST_POST_DATE_ESTIMATED = 'RESULTS_FIRST_POST_DATE_ESTIMATED'
    DUPLICATE_NCT_MAPPING = 'DUPLICATE_NCT_MAPPING'
    ARM_MAPPING_NOT_ASSESSED = 'ARM_MAPPING_NOT_ASSESSED'
    OUTCOME_COMPARABILITY_NOT_ASSESSED = 'OUTCOME_COMPARABILITY_NOT_ASSESSED'
    REDISTRIBUTION_NOT_CLEARED = 'REDISTRIBUTION_NOT_CLEARED'
    BIOLOGICAL_CANDIDATE_COUNT_LT_2 = 'BIOLOGICAL_CANDIDATE_COUNT_LT_2'
    ASSAY_METADATA_FIRST_OBSERVED_DATE_MISSING = 'ASSAY_METADATA_FIRST_OBSERVED_DATE_MISSING'
    ASSAY_METADATA_NOT_AFTER_DECISION = 'ASSAY_METADATA_NOT_AFTER_DECISION'
    ASSAY_METADATA_AFTER_OUTCOME_AS_OF = 'ASSAY_METADATA_AFTER_OUTCOME_AS_OF'
    ASSAY_METADATA_NO_OBSERVATION_IN_WINDOW = 'ASSAY_METADATA_NO_OBSERVATION_IN_WINDOW'


class SnapshotReceipt(StrictModel):
    schema_version: Literal['vaxreplay.feasibility_receipt.v0.1'] = FEASIBILITY_RECEIPT_VERSION
    snapshot_id: str = Field(min_length=1)
    source: SnapshotSource
    synthetic: bool
    source_version: str = Field(min_length=1)
    retrieved_at: datetime
    source_url: str = Field(min_length=1)
    raw_capture_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    records_relative_path: str = Field(min_length=1)
    records_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(ge=0)
    record_count: int = Field(ge=1)
    terms_url: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    redistribution_cleared: bool

    @field_validator('retrieved_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('retrieved_at must include a UTC offset')
        return value

    @field_validator('records_relative_path')
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('records_relative_path must stay inside the snapshot directory')
        return value

    @model_validator(mode='after')
    def validate_source_urls(self) -> Self:
        source_url = urlsplit(self.source_url)
        terms_url = urlsplit(self.terms_url)
        if source_url.scheme != 'https' or terms_url.scheme != 'https':
            raise ValueError('source and terms URLs must use HTTPS')
        if self.synthetic:
            return self
        expected_source_host = {
            SnapshotSource.IMMPORT: 'www.immport.org',
            SnapshotSource.CLINICALTRIALS_GOV: 'clinicaltrials.gov',
        }[self.source]
        expected_terms_host = {
            SnapshotSource.IMMPORT: 'docs.immport.org',
            SnapshotSource.CLINICALTRIALS_GOV: 'clinicaltrials.gov',
        }[self.source]
        if source_url.hostname != expected_source_host:
            raise ValueError(f'real {self.source.value} snapshots must use {expected_source_host}')
        if terms_url.hostname != expected_terms_host:
            raise ValueError(f'real {self.source.value} terms must use {expected_terms_host}')
        return self


class AssayMethodFirstObservation(StrictModel):
    method: str = Field(min_length=1)
    first_observed_date: date


class ImmportDiscoveryRecord(StrictModel):
    study_accession: str = Field(pattern=r'^SDY[0-9]+$')
    source_release_id: str = Field(pattern=r'^DR[0-9]+(?:\.[0-9]+)?$')
    clinical_trial: bool
    human: bool
    vaccine_relevance: VaccineRelevance
    explicit_nct_ids: tuple[str, ...] = ()
    arm_count: int = Field(ge=0)
    assay_methods: tuple[str, ...] = ()
    initial_release_date: date | None = None
    latest_release_date: date | None = None
    assay_first_observations: tuple[AssayMethodFirstObservation, ...] = ()

    @field_validator('explicit_nct_ids')
    @classmethod
    def validate_nct_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError('explicit_nct_ids must be unique and sorted')
        if any(len(nct_id) != 11 or not nct_id.startswith('NCT') or not nct_id[3:].isdigit() for nct_id in value):
            raise ValueError('explicit_nct_ids must use NCT followed by eight digits')
        return value

    @field_validator('assay_methods')
    @classmethod
    def validate_assay_methods(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not method for method in value) or value != tuple(sorted(set(value))):
            raise ValueError('assay_methods must be nonempty, unique, and sorted')
        return value

    @model_validator(mode='after')
    def validate_release_dates(self) -> Self:
        if (
            self.initial_release_date is not None
            and self.latest_release_date is not None
            and self.latest_release_date < self.initial_release_date
        ):
            raise ValueError('latest_release_date cannot precede initial_release_date')
        observation_methods = tuple(observation.method for observation in self.assay_first_observations)
        if observation_methods != tuple(sorted(set(observation_methods))):
            raise ValueError('assay first observations must have unique sorted methods')
        if not set(observation_methods).issubset(self.assay_methods):
            raise ValueError('assay first observations must refer to declared assay methods')
        for observation in self.assay_first_observations:
            if self.initial_release_date is not None and observation.first_observed_date < self.initial_release_date:
                raise ValueError('assay metadata cannot be observed before the initial study release')
            if self.latest_release_date is not None and observation.first_observed_date > self.latest_release_date:
                raise ValueError('assay metadata cannot be observed after the latest study release')
        return self


class CtgovHistoricalRecord(StrictModel):
    nct_id: str = Field(pattern=r'^NCT[0-9]{8}$')
    current_record_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    history_index_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    historical_version_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    history_surface: HistorySurface
    historical_version: Literal[0] = 0
    historical_submitted_date: date
    historical_posted_date: date | None
    historical_posted_date_type: PostedDateType | None
    historical_has_results: bool
    historical_results_section_present: bool
    historical_study_type: StudyType
    historical_arm_count: int = Field(ge=0)
    historical_biological_intervention_count: int = Field(ge=0)
    historical_primary_outcome_count: int = Field(ge=0)
    current_has_results: bool
    current_results_first_post_date: date | None
    current_results_first_post_date_type: PostedDateType | None

    @model_validator(mode='after')
    def validate_dates_and_results(self) -> Self:
        if self.historical_posted_date is None and self.historical_posted_date_type is not None:
            raise ValueError('historical posted date type requires a posted date')
        if self.historical_posted_date is not None and self.historical_submitted_date > self.historical_posted_date:
            raise ValueError('historical submitted date cannot follow its posted date')
        if self.current_has_results != (self.current_results_first_post_date is not None):
            raise ValueError('current results status and first-post date must agree')
        if self.current_results_first_post_date is None and self.current_results_first_post_date_type is not None:
            raise ValueError('current results first-post date type requires a date')
        return self


class AdmissionThresholds(StrictModel):
    minimum_exact_unique_nct: int = Field(ge=1)
    minimum_supported_pre_results_multi_arm: int = Field(ge=1)
    minimum_assay_metadata_matches: int = Field(ge=1)
    minimum_outcome_delay_days: int = Field(ge=1)


class FeasibilityInventorySpec(StrictModel):
    schema_version: Literal['vaxreplay.feasibility_spec.v0.1'] = FEASIBILITY_SPEC_VERSION
    inventory_id: str = Field(min_length=1)
    synthetic: bool
    immport_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    ctgov_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_rule: Literal['first_posted_pre_results_version'] = 'first_posted_pre_results_version'
    outcome_as_of: date
    required_assay_methods: tuple[str, ...] = Field(min_length=1)
    thresholds: AdmissionThresholds

    @field_validator('required_assay_methods')
    @classmethod
    def validate_required_assay_methods(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not method for method in value) or value != tuple(sorted(set(value))):
            raise ValueError('required_assay_methods must be nonempty, unique, and sorted')
        return value


class LinkedStudyInventoryRecord(StrictModel):
    schema_version: Literal['vaxreplay.feasibility_records.v0.1'] = FEASIBILITY_RECORDS_VERSION
    study_accession: str = Field(pattern=r'^SDY[0-9]+$')
    nct_ids: tuple[str, ...]
    nct_id: str | None = Field(default=None, pattern=r'^NCT[0-9]{8}$')
    link_status: LinkStatus
    disposition: InventoryDisposition
    immport_arm_count: int = Field(ge=0)
    ctgov_historical_arm_count: int | None = Field(default=None, ge=0)
    arm_count_agreement: bool | None = None
    assay_metadata_present: bool
    supported_pre_results_history: bool | None = None
    reasons: tuple[InventoryReasonCode, ...] = ()

    @field_validator('nct_ids')
    @classmethod
    def validate_nct_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError('nct_ids must be unique and sorted')
        return value

    @field_validator('reasons')
    @classmethod
    def validate_reasons(cls, value: tuple[InventoryReasonCode, ...]) -> tuple[InventoryReasonCode, ...]:
        if value != tuple(sorted(set(value), key=lambda reason: reason.value)):
            raise ValueError('reason codes must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_link_fields(self) -> Self:
        if self.link_status == LinkStatus.EXACT and (len(self.nct_ids) != 1 or self.nct_id != self.nct_ids[0]):
            raise ValueError('exact links require one matching nct_id')
        if self.link_status != LinkStatus.EXACT and self.nct_id is not None:
            raise ValueError('non-exact links cannot declare nct_id')
        if (
            self.link_status in (LinkStatus.MISSING, LinkStatus.AMBIGUOUS)
            and self.ctgov_historical_arm_count is not None
        ):
            raise ValueError('unresolved links cannot contain ClinicalTrials.gov fields')
        return self


class InventoryMetrics(StrictModel):
    immport_study_count: int = Field(ge=0)
    immport_clinical_trial_count: int = Field(ge=0)
    immport_human_count: int = Field(ge=0)
    immport_multi_arm_count: int = Field(ge=0)
    immport_assay_metadata_count: int = Field(ge=0)
    exact_link_study_count: int = Field(ge=0)
    exact_link_unique_nct_count: int = Field(ge=0)
    missing_link_count: int = Field(ge=0)
    ambiguous_link_count: int = Field(ge=0)
    matched_study_count: int = Field(ge=0)
    matched_unique_nct_count: int = Field(ge=0)
    missing_ctgov_record_count: int = Field(ge=0)
    matched_interventional_count: int = Field(ge=0)
    matched_pre_results_history_count: int = Field(ge=0)
    matched_pre_results_multi_arm_count: int = Field(ge=0)
    matched_historical_post_date_actual_count: int = Field(ge=0)
    matched_historical_post_date_estimated_count: int = Field(ge=0)
    matched_supported_pre_results_history_count: int = Field(ge=0)
    matched_supported_pre_results_multi_arm_count: int = Field(ge=0)
    matched_assay_metadata_count: int = Field(ge=0)
    matched_assay_metadata_temporal_count: int = Field(ge=0)
    matched_arm_count_agreement_count: int = Field(ge=0)
    screened_study_count: int = Field(ge=0)
    screened_unique_nct_count: int = Field(ge=0)
    matched_current_results_unique_nct_count: int = Field(ge=0)
    matched_current_results_post_date_actual_count: int = Field(ge=0)
    matched_current_results_post_date_estimated_count: int = Field(ge=0)
    duplicate_nct_mapping_count: int = Field(ge=0)


class GateResult(StrictModel):
    gate_id: str = Field(min_length=1)
    status: GateStatus
    detail: str = Field(min_length=1)


class FeasibilityInventoryReport(StrictModel):
    schema_version: Literal['vaxreplay.feasibility_report.v0.1'] = FEASIBILITY_REPORT_VERSION
    inventory_id: str = Field(min_length=1)
    adapter_id: Literal['immport-ctgov-feasibility-v0.1'] = FEASIBILITY_ADAPTER_ID
    spec_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    immport_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    ctgov_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    records_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    record_count: int = Field(ge=1)
    metrics: InventoryMetrics
    gates: tuple[GateResult, ...] = Field(min_length=1)
    admission_tier: AdmissionTier

    @field_validator('gates')
    @classmethod
    def validate_gates(cls, value: tuple[GateResult, ...]) -> tuple[GateResult, ...]:
        gate_ids = tuple(gate.gate_id for gate in value)
        if gate_ids != tuple(sorted(set(gate_ids))):
            raise ValueError('gates must have unique sorted IDs')
        return value


class PublicGateResult(StrictModel):
    gate_id: str = Field(min_length=1, pattern=r'^[a-z][a-z0-9_]*$')
    status: GateStatus


PublicCount = int | Literal['<5', 'suppressed']


class PublicInventoryMetrics(StrictModel):
    immport_study_count: PublicCount
    immport_clinical_trial_count: PublicCount
    immport_human_count: PublicCount
    immport_multi_arm_count: PublicCount
    immport_assay_metadata_count: PublicCount
    exact_link_study_count: PublicCount
    exact_link_unique_nct_count: PublicCount
    missing_link_count: PublicCount
    ambiguous_link_count: PublicCount
    matched_study_count: PublicCount
    matched_unique_nct_count: PublicCount
    missing_ctgov_record_count: PublicCount
    matched_interventional_count: PublicCount
    matched_pre_results_history_count: PublicCount
    matched_pre_results_multi_arm_count: PublicCount
    matched_historical_post_date_actual_count: PublicCount
    matched_historical_post_date_estimated_count: PublicCount
    matched_supported_pre_results_history_count: PublicCount
    matched_supported_pre_results_multi_arm_count: PublicCount
    matched_assay_metadata_count: PublicCount
    matched_assay_metadata_temporal_count: PublicCount
    matched_arm_count_agreement_count: PublicCount
    screened_study_count: PublicCount
    screened_unique_nct_count: PublicCount
    matched_current_results_unique_nct_count: PublicCount
    matched_current_results_post_date_actual_count: PublicCount
    matched_current_results_post_date_estimated_count: PublicCount
    duplicate_nct_mapping_count: PublicCount

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        if any(isinstance(value, int) and value < 0 for value in self.__dict__.values()):
            raise ValueError('public counts cannot be negative')
        return self


class PublicFeasibilitySummary(StrictModel):
    schema_version: Literal['vaxreplay.public_feasibility_summary.v0.1'] = PUBLIC_FEASIBILITY_SUMMARY_VERSION
    adapter_id: Literal['immport-ctgov-feasibility-v0.1'] = FEASIBILITY_ADAPTER_ID
    synthetic: bool
    record_count: int = Field(ge=1)
    metrics: PublicInventoryMetrics
    gates: tuple[PublicGateResult, ...] = Field(min_length=1)
    admission_tier: AdmissionTier

    @model_validator(mode='after')
    def validate_real_aggregation_boundary(self) -> Self:
        if self.synthetic:
            return self
        if self.record_count < PUBLIC_MINIMUM_REAL_RECORD_COUNT:
            raise ValueError(
                f'real public summaries require at least {PUBLIC_MINIMUM_REAL_RECORD_COUNT} source records'
            )
        if any(
            isinstance(value, int) and 0 < value < PUBLIC_SMALL_CELL_THRESHOLD
            for value in self.metrics.__dict__.values()
        ):
            raise ValueError(f'real nonzero public counts below {PUBLIC_SMALL_CELL_THRESHOLD} must be suppressed')
        for group in PUBLIC_COMPLEMENTARY_GROUPS:
            values = tuple(getattr(self.metrics, name) for name in group)
            if '<5' in values and 'suppressed' not in values:
                raise ValueError('small partition cells require complementary suppression')
        for total_name, subset_name in PUBLIC_DERIVED_DIFFERENCE_PAIRS:
            total = getattr(self.metrics, total_name)
            subset = getattr(self.metrics, subset_name)
            if isinstance(total, int) and isinstance(subset, int) and 0 < total - subset < PUBLIC_SMALL_CELL_THRESHOLD:
                raise ValueError('small derived differences require complementary suppression')
        for total_name, first_name, second_name in PUBLIC_DERIVED_RESIDUAL_TRIPLES:
            total = getattr(self.metrics, total_name)
            first = getattr(self.metrics, first_name)
            second = getattr(self.metrics, second_name)
            if all(isinstance(value, int) for value in (total, first, second)):
                residual = total - first - second
                if 0 < residual < PUBLIC_SMALL_CELL_THRESHOLD:
                    raise ValueError('small derived residuals require complementary suppression')
        return self

    @field_validator('gates')
    @classmethod
    def validate_public_gates(cls, value: tuple[PublicGateResult, ...]) -> tuple[PublicGateResult, ...]:
        gate_ids = tuple(gate.gate_id for gate in value)
        if gate_ids != tuple(sorted(set(gate_ids))):
            raise ValueError('public gates must have unique sorted IDs')
        return value
