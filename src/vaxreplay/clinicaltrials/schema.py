"""Strict contracts for content-addressed AACT clinical-trial replay slices.

The contracts keep three things separate:

* an immutable full-archive receipt;
* an exact, NCT-scoped table slice derived from that archive; and
* reward-affecting arm and endpoint configuration.

Arm joins are deliberately expressed through normalized regimen titles.  Database row order and
numeric outcome values are never part of the join key.
"""

from __future__ import annotations

import enum
import math
import re
import unicodedata
from datetime import date, datetime, time, timezone
from pathlib import PurePosixPath
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.case_schema import RANKING_REWARD_VERSION, Split, StrictModel

AACT_ARCHIVE_RECEIPT_SCHEMA_VERSION = 'vaxreplay.aact-archive-receipt.v0.1'
AACT_SLICE_RECEIPT_SCHEMA_VERSION = 'vaxreplay.aact-slice-receipt.v0.1'
AACT_EPISODE_SPEC_SCHEMA_VERSION = 'vaxreplay.aact-early-clinical-spec.v0.1'
AACT_PRIVATE_AUDIT_SCHEMA_VERSION = 'vaxreplay.aact-early-clinical-audit.v0.1'
AACT_ADAPTER_ID = 'aact-early-clinical-arm-prioritization-v0.1'
# Public reference semantics only: this fixed rubric is contamination-exposed and cannot define a
# hidden commercial or headline evaluation.
AACT_RUBRIC_VERSION = 'aact-day-point-estimate-placebo-geomean-v1'
EARLY_CLINICAL_TASK = 'early_clinical_arm_prioritization'

REGIMEN_DEFINITION_DIMENSION = 'regimen_definition'
ENDPOINT_ALIGNMENT_DIMENSION = 'endpoint_alignment'
REQUIRED_DIMENSIONS = (ENDPOINT_ALIGNMENT_DIMENSION, REGIMEN_DEFINITION_DIMENSION)

# Arithmetic is recomputed from JSON-decoded decimal values.  This tolerance only absorbs ordinary
# IEEE-754 serialization/reconstruction roundoff; it is intentionally too small to hide a material
# change to a fold, composite, or utility.
AUDIT_NUMERIC_REL_TOL = 1e-12
AUDIT_NUMERIC_ABS_TOL = 1e-12


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _safe_relative_path(value: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError(f'{field_name} must be a normalized relative path')
    return value


def normalize_regimen_title(value: str) -> str:
    """Return the stable, value-blind regimen key used for cross-snapshot joins.

    AACT commonly moves the word ``Group`` from the beginning to the end of a title between the
    protocol and results modules.  Removing only that structural token, while retaining every
    regimen/cohort token, handles that representation change without fuzzy or row-order matching.
    """

    normalized = unicodedata.normalize('NFKC', value).casefold()
    tokens = [token for token in re.findall(r'[a-z0-9]+', normalized) if token != 'group']
    if not tokens:
        raise ValueError('regimen title must contain a non-structural token')
    return '-'.join(tokens)


class AactSourceTable(str, enum.Enum):
    STUDIES = 'studies'
    DESIGNS = 'designs'
    DESIGN_GROUPS = 'design_groups'
    DESIGN_GROUP_INTERVENTIONS = 'design_group_interventions'
    INTERVENTIONS = 'interventions'
    DESIGN_OUTCOMES = 'design_outcomes'
    RESULT_GROUPS = 'result_groups'
    OUTCOMES = 'outcomes'
    OUTCOME_COUNTS = 'outcome_counts'
    OUTCOME_MEASUREMENTS = 'outcome_measurements'


class ArmRole(str, enum.Enum):
    CANDIDATE = 'candidate'
    CONTROL = 'control'


class AactArchiveReceipt(StrictModel):
    """Receipt for one immutable AACT full monthly ZIP.

    AACT guarantees an archive calendar date, not a timezone-qualified build instant.  The cutoff
    is therefore a benchmark convention: end-of-day UTC for that archive date.  It is an explicit
    conservative upper-bound convention, not a claim about the server's actual build time.
    """

    schema_version: Literal['vaxreplay.aact-archive-receipt.v0.1'] = AACT_ARCHIVE_RECEIPT_SCHEMA_VERSION
    snapshot_id: str = Field(min_length=1)
    archive_date: date
    source_cutoff_at: datetime
    cutoff_semantics: Literal['archive-date-end-utc-upper-bound-v1'] = 'archive-date-end-utc-upper-bound-v1'
    retrieved_at: datetime
    source_url: str = Field(min_length=1)
    archive_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    archive_bytes: int = Field(gt=0)
    etag: str = Field(min_length=1)
    last_modified_at: datetime
    complete_archive: Literal[True] = True
    permanent_archive: Literal[True] = True
    license_id: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    citation: str = Field(min_length=1)

    @field_validator('source_cutoff_at', 'retrieved_at', 'last_modified_at')
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode='after')
    def validate_archive(self) -> Self:
        expected_cutoff = datetime.combine(self.archive_date, time.max, tzinfo=timezone.utc)
        if self.source_cutoff_at != expected_cutoff:
            raise ValueError('source_cutoff_at must be end-of-day UTC for archive_date')
        if self.source_cutoff_at > self.retrieved_at:
            raise ValueError('source_cutoff_at cannot be after retrieved_at')
        parsed = urlsplit(self.source_url)
        if parsed.scheme.lower() != 'https' or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError('source_url must be an HTTPS URL without user information')
        return self


class AactTableReceipt(StrictModel):
    """Exact-byte and shape binding for one extracted pipe-delimited table."""

    table: AactSourceTable
    source_member_path: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(gt=0)
    row_count: int = Field(ge=0)
    header_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @field_validator('source_member_path', 'relative_path')
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        return _safe_relative_path(value, info.field_name)

    @model_validator(mode='after')
    def validate_table_paths(self) -> Self:
        expected = f'{self.table.value}.txt'
        if self.source_member_path != expected or self.relative_path != expected:
            raise ValueError(f'{self.table.value} must bind the canonical archive member {expected}')
        return self


class AactSliceReceipt(StrictModel):
    """Complete inventory of an exact NCT-scoped extraction from one full archive."""

    schema_version: Literal['vaxreplay.aact-slice-receipt.v0.1'] = AACT_SLICE_RECEIPT_SCHEMA_VERSION
    slice_id: str = Field(min_length=1)
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    archive: AactArchiveReceipt
    extraction_method: Literal['aact-pipe-exact-nct-v1'] = 'aact-pipe-exact-nct-v1'
    source_archive_verified: Literal[True] = True
    created_at: datetime
    tables: tuple[AactTableReceipt, ...] = Field(min_length=1)

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'created_at')

    @model_validator(mode='after')
    def validate_slice(self) -> Self:
        if self.created_at < self.archive.source_cutoff_at:
            raise ValueError('slice cannot be created before its conservative source cutoff')
        table_names = tuple(table.table.value for table in self.tables)
        if table_names != tuple(sorted(table_names)) or len(table_names) != len(set(table_names)):
            raise ValueError('slice tables must be unique and sorted by table name')
        relative_paths = tuple(table.relative_path for table in self.tables)
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError('slice table paths must be unique')
        return self


class ArmMappingSpec(StrictModel):
    """Release-committed cross-snapshot arm identity, independent of outcome values."""

    role: ArmRole
    candidate_id: str | None = Field(default=None, pattern=r'^cand-[a-z0-9][a-z0-9-]*$')
    regimen_key: str = Field(pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    decision_title: str = Field(min_length=1)
    result_title: str = Field(min_length=1)
    public_regimen: str = Field(min_length=12, max_length=330)

    @model_validator(mode='after')
    def validate_mapping(self) -> Self:
        if self.role == ArmRole.CANDIDATE and self.candidate_id is None:
            raise ValueError('candidate arms require candidate_id')
        if self.role == ArmRole.CONTROL and self.candidate_id is not None:
            raise ValueError('the control arm cannot have a candidate_id')
        decision_key = normalize_regimen_title(self.decision_title)
        result_key = normalize_regimen_title(self.result_title)
        if decision_key != self.regimen_key or result_key != self.regimen_key:
            raise ValueError('decision and result titles must independently normalize to regimen_key')
        return self


class PanelSelector(StrictModel):
    """Value-blind cohort selector applied to normalized protocol arm titles."""

    normalized_regimen_key_suffix: str = Field(pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    allowed_group_types: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_selector(self) -> Self:
        normalized_types = tuple(value.casefold() for value in self.allowed_group_types)
        if normalized_types != tuple(sorted(normalized_types)) or len(normalized_types) != len(set(normalized_types)):
            raise ValueError('allowed_group_types must be case-insensitively unique and sorted')
        return self


class OutcomeEndpointSpec(StrictModel):
    """One predeclared endpoint and its later AACT outcome crosswalk."""

    endpoint_id: str = Field(pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    public_name: str = Field(min_length=3, max_length=220)
    decision_outcome_id: str = Field(min_length=1)
    decision_measure: str = Field(min_length=3)
    result_outcome_id: str = Field(min_length=1)
    result_title: str = Field(min_length=3)
    classification: str = Field(min_length=1)
    param_type: str = Field(min_length=1)
    category: str = ''
    higher_is_better: Literal[True] = True


class OutcomeRubric(StrictModel):
    """Fixed transparent v1 conversion from later measurements to ranking grades."""

    version: Literal['aact-day-point-estimate-placebo-geomean-v1'] = AACT_RUBRIC_VERSION
    target_id: str = Field(min_length=1)
    endpoints: tuple[OutcomeEndpointSpec, ...] = Field(min_length=2)
    grade_thresholds: tuple[float, float, float, float] = (1.0, 2.0, 4.0, 8.0)
    positive_threshold: float = 8.0

    @model_validator(mode='after')
    def validate_rubric(self) -> Self:
        endpoint_ids = tuple(endpoint.endpoint_id for endpoint in self.endpoints)
        if endpoint_ids != tuple(sorted(endpoint_ids)) or len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError('rubric endpoints must have unique endpoint IDs in sorted order')
        if self.grade_thresholds != (1.0, 2.0, 4.0, 8.0):
            raise ValueError('the v1 grade thresholds are fixed at 1, 2, 4, and 8')
        if self.positive_threshold != 8.0:
            raise ValueError('the v1 binary forecast threshold is fixed at 8')
        decision_ids = tuple(endpoint.decision_outcome_id for endpoint in self.endpoints)
        result_ids = tuple(endpoint.result_outcome_id for endpoint in self.endpoints)
        if len(decision_ids) != len(set(decision_ids)) or len(result_ids) != len(set(result_ids)):
            raise ValueError('rubric endpoint source IDs must be unique within each snapshot')
        return self


class EarlyClinicalEpisodeSpec(StrictModel):
    """Reward-affecting configuration committed before a replay episode is released."""

    schema_version: Literal['vaxreplay.aact-early-clinical-spec.v0.1'] = AACT_EPISODE_SPEC_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    lineage_group_id: str = Field(min_length=1)
    synthetic: Literal[False] = False
    split: Split = Split.TEST
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    decision_snapshot_id: str = Field(min_length=1)
    label_snapshot_id: str = Field(min_length=1)
    decision_at: datetime
    outcome_as_of: datetime
    portfolio_size: int = Field(gt=0)
    panel_selector: PanelSelector
    arm_mappings: tuple[ArmMappingSpec, ...] = Field(min_length=3)
    rubric: OutcomeRubric
    forbidden_public_tokens: tuple[str, ...] = Field(min_length=1)
    required_dimensions: tuple[str, str] = REQUIRED_DIMENSIONS
    adjudication_version: str = Field(min_length=1)
    reward_version: Literal['v1.0'] = RANKING_REWARD_VERSION

    @field_validator('decision_at', 'outcome_as_of')
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode='after')
    def validate_episode(self) -> Self:
        if self.split != Split.TEST:
            raise ValueError('early clinical replay episodes must use the test split')
        if self.decision_snapshot_id == self.label_snapshot_id:
            raise ValueError('decision and label snapshots must differ')
        if self.outcome_as_of <= self.decision_at:
            raise ValueError('outcome_as_of must be after decision_at')
        if (self.outcome_as_of - self.decision_at).total_seconds() % 86_400:
            raise ValueError('decision-to-outcome horizon must be a whole number of days')
        if self.required_dimensions != REQUIRED_DIMENSIONS:
            raise ValueError(f'v1 requires the honest protocol dimensions {REQUIRED_DIMENSIONS!r}')
        normalized_forbidden = tuple(token.casefold() for token in self.forbidden_public_tokens)
        if (
            normalized_forbidden != tuple(sorted(normalized_forbidden))
            or len(normalized_forbidden) != len(set(normalized_forbidden))
            or any(not token.strip() for token in self.forbidden_public_tokens)
        ):
            raise ValueError('forbidden_public_tokens must be non-empty, case-insensitively unique, and sorted')
        public_text = '\n'.join(
            [mapping.public_regimen for mapping in self.arm_mappings]
            + [endpoint.public_name for endpoint in self.rubric.endpoints]
        ).casefold()
        leaked = [token for token in normalized_forbidden if token in public_text]
        if leaked:
            raise ValueError(f'public regimen/endpoint descriptions contain forbidden identity tokens: {leaked}')
        regimen_keys = tuple(mapping.regimen_key for mapping in self.arm_mappings)
        if regimen_keys != tuple(sorted(regimen_keys)) or len(regimen_keys) != len(set(regimen_keys)):
            raise ValueError('arm mappings must have unique regimen keys in sorted order')
        controls = tuple(mapping for mapping in self.arm_mappings if mapping.role == ArmRole.CONTROL)
        candidates = tuple(mapping for mapping in self.arm_mappings if mapping.role == ArmRole.CANDIDATE)
        if len(controls) != 1:
            raise ValueError('exactly one placebo/control arm is required')
        if len(candidates) < 2:
            raise ValueError('at least two candidate arms are required')
        candidate_ids = tuple(mapping.candidate_id for mapping in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError('candidate IDs must be unique')
        if self.portfolio_size >= len(candidates):
            raise ValueError('V1 portfolio_size must be smaller than the candidate count')
        return self


class ResolvedArmMapping(StrictModel):
    role: ArmRole
    candidate_id: str | None
    regimen_key: str = Field(min_length=1)
    decision_group_id: str = Field(min_length=1)
    result_group_id: str = Field(min_length=1)
    decision_title: str = Field(min_length=1)
    result_title: str = Field(min_length=1)
    decision_description: str = Field(min_length=1)
    result_description: str = Field(min_length=1)
    decision_title_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    result_title_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class EndpointValueAudit(StrictModel):
    endpoint_id: str = Field(min_length=1)
    outcome_id: str = Field(min_length=1)
    result_group_id: str = Field(min_length=1)
    measurement_id: str = Field(min_length=1)
    control_measurement_id: str = Field(min_length=1)
    classification: str = Field(min_length=1)
    param_type: str = Field(min_length=1)
    candidate_param_value: str = Field(min_length=1)
    control_param_value: str = Field(min_length=1)
    candidate_value: float = Field(gt=0, allow_inf_nan=False)
    control_value: float = Field(gt=0, allow_inf_nan=False)
    fold_over_control: float = Field(gt=0, allow_inf_nan=False)


class ResolvedEndpointMapping(StrictModel):
    endpoint_id: str = Field(min_length=1)
    decision_outcome_id: str = Field(min_length=1)
    decision_measure: str = Field(min_length=1)
    decision_time_frame: str = Field(min_length=1)
    decision_description: str = ''
    result_outcome_id: str = Field(min_length=1)
    result_title: str = Field(min_length=1)
    result_time_frame: str = Field(min_length=1)
    result_description: str = ''
    result_param_type: str = Field(min_length=1)


class CandidateOutcomeAudit(StrictModel):
    candidate_id: str = Field(min_length=1)
    endpoint_values: tuple[EndpointValueAudit, ...] = Field(min_length=2)
    composite_fold_over_control: float = Field(gt=0, allow_inf_nan=False)
    relevance_grade: int = Field(ge=0, le=4)
    binary_outcome: Literal[0, 1]
    candidate_utility: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_arithmetic_and_fixed_rubric(self) -> Self:
        endpoint_ids = tuple(endpoint.endpoint_id for endpoint in self.endpoint_values)
        if endpoint_ids != tuple(sorted(endpoint_ids)) or len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError('candidate audit endpoints must be unique and sorted by endpoint_id')
        for endpoint in self.endpoint_values:
            expected_fold = endpoint.candidate_value / endpoint.control_value
            if not math.isclose(
                endpoint.fold_over_control,
                expected_fold,
                rel_tol=AUDIT_NUMERIC_REL_TOL,
                abs_tol=AUDIT_NUMERIC_ABS_TOL,
            ):
                raise ValueError(f'{endpoint.endpoint_id} fold_over_control must equal candidate_value / control_value')
        expected_composite = math.exp(
            math.fsum(math.log(endpoint.fold_over_control) for endpoint in self.endpoint_values)
            / len(self.endpoint_values)
        )
        if not math.isclose(
            self.composite_fold_over_control,
            expected_composite,
            rel_tol=AUDIT_NUMERIC_REL_TOL,
            abs_tol=AUDIT_NUMERIC_ABS_TOL,
        ):
            raise ValueError('composite_fold_over_control must equal the geometric mean of endpoint folds')
        expected_grade = 0
        for index, threshold in enumerate((1.0, 2.0, 4.0, 8.0), start=1):
            if self.composite_fold_over_control >= threshold:
                expected_grade = index
        expected_binary = int(self.composite_fold_over_control >= 8.0)
        expected_utility = min(self.composite_fold_over_control / 8.0, 1.0)
        if self.relevance_grade != expected_grade or self.binary_outcome != expected_binary:
            raise ValueError('audit labels must follow the fixed composite grade and binary thresholds')
        if not math.isclose(
            self.candidate_utility,
            expected_utility,
            rel_tol=AUDIT_NUMERIC_REL_TOL,
            abs_tol=AUDIT_NUMERIC_ABS_TOL,
        ):
            raise ValueError('candidate_utility must equal min(composite / 8, 1)')
        return self


class AactPrivateAudit(StrictModel):
    schema_version: Literal['vaxreplay.aact-early-clinical-audit.v0.1'] = AACT_PRIVATE_AUDIT_SCHEMA_VERSION
    adapter_id: Literal['aact-early-clinical-arm-prioritization-v0.1'] = AACT_ADAPTER_ID
    episode_id: str = Field(min_length=1)
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    decision_slice_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_slice_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    results_first_posted_date: date
    outcome_snapshot_at: datetime
    decision_design_group_count: int = Field(ge=3)
    selected_decision_group_count: int = Field(ge=3)
    excluded_decision_group_count: int = Field(ge=0)
    target_outcome_group_count: int = Field(ge=3)
    selected_result_group_count: int = Field(ge=3)
    excluded_target_result_group_count: int = Field(ge=0)
    non_target_outcome_result_group_count: int = Field(ge=0)
    arm_mappings: tuple[ResolvedArmMapping, ...] = Field(min_length=3)
    endpoint_mappings: tuple[ResolvedEndpointMapping, ...] = Field(min_length=2)
    outcomes: tuple[CandidateOutcomeAudit, ...] = Field(min_length=2)

    @field_validator('outcome_snapshot_at')
    @classmethod
    def validate_outcome_snapshot_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'outcome_snapshot_at')

    @model_validator(mode='after')
    def validate_panel_counts(self) -> Self:
        if self.selected_decision_group_count + self.excluded_decision_group_count != self.decision_design_group_count:
            raise ValueError('decision panel counts must partition every design group')
        if (
            self.selected_result_group_count + self.excluded_target_result_group_count
            != self.target_outcome_group_count
        ):
            raise ValueError('result panel counts must partition every target-outcome group')
        return self
