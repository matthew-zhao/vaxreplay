"""Strict contracts for registry-observed vaccine-trial execution replay.

This task measures what a later AACT snapshot records about trial execution.  It does not measure
biological activity, vaccine efficacy, safety, clinical utility, or whether a development decision
was scientifically correct.  Decision-time rows and later label rows deliberately use different
schemas so later outcomes cannot be smuggled into cohort selection.
"""

from __future__ import annotations

import enum
import math
from datetime import date
from typing import Literal, Self

from pydantic import Field, model_validator

from vaxreplay.case_schema import StrictModel

EXECUTION_TASK_ID = 'registry_observed_trial_execution'
EXECUTION_TASK_SEMANTICS = (
    'Registry-observed trial execution only; not biological activity, efficacy, safety, clinical '
    'utility, or scientific development success.'
)
EXECUTION_ELIGIBILITY_RULE_ID = 'aact-fixed-anchor-pre-results-phase1-vaccine-v0.1'
EXECUTION_LABEL_RULE_ID = 'aact-plus-48-month-observable-execution-v0.1'
EXECUTION_POLICY_SCHEMA_VERSION = 'vaxreplay.aact-execution-policy.v0.1'
EXECUTION_INVENTORY_SCHEMA_VERSION = 'vaxreplay.aact-execution-inventory.v0.1'
EXECUTION_LABEL_SET_SCHEMA_VERSION = 'vaxreplay.aact-execution-label-set.v0.1'
LABEL_HORIZON_MONTHS = 48
ENROLLMENT_RATIO_DECIMAL_PLACES = 12
type ExecutionTaskSemantics = Literal[
    'Registry-observed trial execution only; not biological activity, efficacy, safety, clinical '
    'utility, or scientific development success.'
]


class DiseaseStratum(str, enum.Enum):
    COVID_19 = 'covid_19'
    NON_COVID_INFECTIOUS = 'non_covid_infectious'


class NormalizedStudyType(str, enum.Enum):
    INTERVENTIONAL = 'interventional'
    OTHER = 'other'


class NormalizedPhase(str, enum.Enum):
    EARLY_PHASE_1 = 'early_phase_1'
    PHASE_1 = 'phase_1'
    PHASE_1_PHASE_2 = 'phase_1_phase_2'
    OTHER = 'other'


class RegistryValueType(str, enum.Enum):
    ANTICIPATED = 'anticipated'
    ACTUAL = 'actual'


class RegistryStatus(str, enum.Enum):
    NOT_YET_RECRUITING = 'not_yet_recruiting'
    RECRUITING = 'recruiting'
    ENROLLING_BY_INVITATION = 'enrolling_by_invitation'
    ACTIVE_NOT_RECRUITING = 'active_not_recruiting'
    COMPLETED = 'completed'
    TERMINATED = 'terminated'
    WITHDRAWN = 'withdrawn'
    SUSPENDED = 'suspended'
    UNKNOWN = 'unknown'
    OTHER = 'other'


class EligibilityReason(str, enum.Enum):
    NOT_INTERVENTIONAL = 'not_interventional'
    PHASE_NOT_EARLY_CLINICAL = 'phase_not_early_clinical'
    NOT_HUMAN = 'not_human'
    NOT_PROPHYLACTIC = 'not_prophylactic'
    NOT_INFECTIOUS_DISEASE_VACCINE = 'not_infectious_disease_vaccine'
    NO_BIOLOGICAL_INTERVENTION = 'no_biological_intervention'
    RESULTS_ALREADY_PRESENT = 'results_already_present'
    DECISION_STATUS_INELIGIBLE = 'decision_status_ineligible'
    PLANNED_ENROLLMENT_MISSING = 'planned_enrollment_missing'
    PLANNED_ENROLLMENT_NOT_ANTICIPATED = 'planned_enrollment_not_anticipated'
    PLANNED_PRIMARY_COMPLETION_MISSING = 'planned_primary_completion_missing'
    PLANNED_PRIMARY_COMPLETION_NOT_ANTICIPATED = 'planned_primary_completion_not_anticipated'
    PLANNED_PRIMARY_COMPLETION_NOT_AFTER_ANCHOR = 'planned_primary_completion_not_after_anchor'


class RegistryOutcomeClass(str, enum.Enum):
    COMPLETED = 'completed'
    TERMINATED = 'terminated'
    WITHDRAWN = 'withdrawn'
    SUSPENDED = 'suspended'
    NON_TERMINAL = 'non_terminal'
    STATUS_MISSING = 'status_missing'
    RECORD_MISSING = 'record_missing'


class ObservationState(str, enum.Enum):
    OBSERVED_ACTUAL = 'observed_actual'
    NOT_ACTUAL = 'not_actual'
    VALUE_MISSING = 'value_missing'
    RECORD_MISSING = 'record_missing'


ELIGIBLE_PHASES = frozenset({NormalizedPhase.EARLY_PHASE_1, NormalizedPhase.PHASE_1, NormalizedPhase.PHASE_1_PHASE_2})
ELIGIBLE_DECISION_STATUSES = frozenset(
    {
        RegistryStatus.NOT_YET_RECRUITING,
        RegistryStatus.RECRUITING,
        RegistryStatus.ENROLLING_BY_INVITATION,
        RegistryStatus.ACTIVE_NOT_RECRUITING,
    }
)
# Suspension is retained as its own observed state but is not called a terminal failure because a
# suspended trial can resume.
OBSERVED_FAILURE_STATUSES = frozenset({RegistryStatus.TERMINATED, RegistryStatus.WITHDRAWN})


class AactExecutionDecisionRow(StrictModel):
    """Outcome-free normalized projection of one historical AACT study row."""

    snapshot_id: str = Field(min_length=1)
    archive_date: date
    source_record_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    lineage_group_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    disease_stratum: DiseaseStratum
    study_first_posted_date: date
    study_type: NormalizedStudyType
    phase: NormalizedPhase
    human: bool
    prophylactic_intent: bool
    infectious_disease_vaccine: bool
    biological_intervention_count: int = Field(ge=0)
    overall_status: RegistryStatus
    results_first_posted_date: date | None = None
    results_section_present: bool
    enrollment: int | None = Field(default=None, ge=0)
    enrollment_type: RegistryValueType | None = None
    primary_completion_date: date | None = None
    primary_completion_date_type: RegistryValueType | None = None

    @model_validator(mode='after')
    def validate_decision_row(self) -> Self:
        if self.study_first_posted_date > self.archive_date:
            raise ValueError('study_first_posted_date cannot be after the containing archive')
        if self.results_first_posted_date is not None:
            if self.results_first_posted_date > self.archive_date:
                raise ValueError('results_first_posted_date cannot be after the containing archive')
            if not self.results_section_present:
                raise ValueError('a posted-results date requires a results section')
        if (self.enrollment is None) != (self.enrollment_type is None):
            raise ValueError('enrollment and enrollment_type must be present together')
        if (self.primary_completion_date is None) != (self.primary_completion_date_type is None):
            raise ValueError('primary completion date and type must be present together')
        if (
            self.primary_completion_date_type == RegistryValueType.ACTUAL
            and self.primary_completion_date is not None
            and self.primary_completion_date > self.archive_date
        ):
            raise ValueError('an actual primary-completion date cannot be after the containing archive')
        return self


class AactExecutionOutcomeRow(StrictModel):
    """Normalized +48-month AACT observation for one already-selected trial.

    ``record_present=False`` is an explicit observation, not permission to drop the trial.
    """

    snapshot_id: str = Field(min_length=1)
    archive_date: date
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    record_present: bool
    source_record_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    overall_status: RegistryStatus | None = None
    enrollment: int | None = Field(default=None, ge=0)
    enrollment_type: RegistryValueType | None = None
    primary_completion_date: date | None = None
    primary_completion_date_type: RegistryValueType | None = None

    @model_validator(mode='after')
    def validate_outcome_row(self) -> Self:
        observed_values = (
            self.source_record_sha256,
            self.overall_status,
            self.enrollment,
            self.enrollment_type,
            self.primary_completion_date,
            self.primary_completion_date_type,
        )
        if not self.record_present:
            if any(value is not None for value in observed_values):
                raise ValueError('an absent registry record cannot carry observed values')
            return self
        if self.source_record_sha256 is None:
            raise ValueError('a present registry record requires source_record_sha256')
        if (self.enrollment is None) != (self.enrollment_type is None):
            raise ValueError('enrollment and enrollment_type must be present together')
        if (self.primary_completion_date is None) != (self.primary_completion_date_type is None):
            raise ValueError('primary completion date and type must be present together')
        if (
            self.primary_completion_date_type == RegistryValueType.ACTUAL
            and self.primary_completion_date is not None
            and self.primary_completion_date > self.archive_date
        ):
            raise ValueError('an actual primary-completion date cannot be after the containing archive')
        return self


class AnchorSourceBinding(StrictModel):
    """Precommitted complete decision projection and deterministic later snapshot."""

    anchor_date: date
    decision_snapshot_id: str = Field(min_length=1)
    decision_archive_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_rows_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_row_count: int = Field(ge=0)
    complete_decision_projection: Literal[True] = True
    label_snapshot_id: str = Field(min_length=1)
    label_archive_date: date
    label_archive_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ExecutionCohortPolicy(StrictModel):
    schema_version: Literal['vaxreplay.aact-execution-policy.v0.1'] = EXECUTION_POLICY_SCHEMA_VERSION
    policy_id: str = Field(min_length=1)
    synthetic: bool
    selection_universe_rule_id: str = Field(min_length=1)
    selection_universe_rule_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    lineage_grouping_rule_id: str = Field(min_length=1)
    lineage_grouping_rule_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    task_id: Literal['registry_observed_trial_execution'] = EXECUTION_TASK_ID
    task_semantics: ExecutionTaskSemantics = EXECUTION_TASK_SEMANTICS
    eligibility_rule_id: Literal['aact-fixed-anchor-pre-results-phase1-vaccine-v0.1'] = EXECUTION_ELIGIBILITY_RULE_ID
    label_rule_id: Literal['aact-plus-48-month-observable-execution-v0.1'] = EXECUTION_LABEL_RULE_ID
    label_horizon_months: Literal[48] = LABEL_HORIZON_MONTHS
    anchors: tuple[AnchorSourceBinding, ...] = Field(min_length=1)
    eligibility_uses_decision_snapshot_only: Literal[True] = True
    outcome_conditioned_selection_prohibited: Literal[True] = True
    missing_outcomes_retained: Literal[True] = True
    failed_trials_retained: Literal[True] = True
    earliest_eligible_anchor_required: Literal[True] = True
    lineage_grouping_required: Literal[True] = True
    covid_stratum_required: Literal[True] = True
    vaccine_relevance_and_lineage_precommitted: Literal[True] = True

    @model_validator(mode='after')
    def validate_policy(self) -> Self:
        anchor_dates = tuple(binding.anchor_date for binding in self.anchors)
        if anchor_dates != tuple(sorted(anchor_dates)) or len(anchor_dates) != len(set(anchor_dates)):
            raise ValueError('anchor bindings must be unique and sorted by anchor_date')
        decision_ids = tuple(binding.decision_snapshot_id for binding in self.anchors)
        label_ids = tuple(binding.label_snapshot_id for binding in self.anchors)
        if len(decision_ids) != len(set(decision_ids)) or len(label_ids) != len(set(label_ids)):
            raise ValueError('decision and label snapshot IDs must each be unique')
        if set(decision_ids) & set(label_ids):
            raise ValueError('decision and label snapshot IDs must be disjoint')
        for binding in self.anchors:
            if add_calendar_months(binding.anchor_date, LABEL_HORIZON_MONTHS) != binding.label_archive_date:
                raise ValueError('each label archive must be exactly 48 calendar months after its anchor')
        return self


class AnchorEligibilityAssessment(StrictModel):
    snapshot_id: str = Field(min_length=1)
    anchor_date: date
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    eligible: bool
    reason_codes: tuple[EligibilityReason, ...]

    @model_validator(mode='after')
    def validate_assessment(self) -> Self:
        sorted_codes = tuple(sorted(self.reason_codes, key=lambda value: value.value))
        if self.reason_codes != sorted_codes or len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError('eligibility reason codes must be unique and sorted')
        if self.eligible == bool(self.reason_codes):
            raise ValueError('eligible assessments have no reasons; ineligible assessments require reasons')
        return self


class TrialAnchorAssignment(StrictModel):
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    lineage_group_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    disease_stratum: DiseaseStratum
    decision_snapshot_id: str = Field(min_length=1)
    anchor_date: date
    label_snapshot_id: str = Field(min_length=1)
    label_archive_date: date
    planned_enrollment: int = Field(gt=0)
    planned_primary_completion_date: date


class TrialLineageGroup(StrictModel):
    lineage_group_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    disease_stratum: DiseaseStratum
    nct_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_members(self) -> Self:
        if self.nct_ids != tuple(sorted(self.nct_ids)) or len(self.nct_ids) != len(set(self.nct_ids)):
            raise ValueError('lineage NCT IDs must be unique and sorted')
        return self


class ExecutionCohortInventory(StrictModel):
    schema_version: Literal['vaxreplay.aact-execution-inventory.v0.1'] = EXECUTION_INVENTORY_SCHEMA_VERSION
    policy: ExecutionCohortPolicy
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_rows_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_rows: tuple[AactExecutionDecisionRow, ...]
    assessments: tuple[AnchorEligibilityAssessment, ...]
    assignments: tuple[TrialAnchorAssignment, ...]
    lineage_groups: tuple[TrialLineageGroup, ...]
    outcome_fields_used_for_selection: Literal[False] = False
    excluded_rows_retained: Literal[True] = True


class ObservableExecutionLabel(StrictModel):
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    lineage_group_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    disease_stratum: DiseaseStratum
    anchor_date: date
    label_snapshot_id: str = Field(min_length=1)
    label_archive_date: date
    registry_record_present: bool
    observed_registry_status: RegistryStatus | None
    registry_outcome_class: RegistryOutcomeClass
    failed_status_observed: bool
    planned_enrollment: int = Field(gt=0)
    observed_enrollment_type: RegistryValueType | None
    observed_actual_enrollment: int | None = Field(default=None, ge=0)
    enrollment_ratio: float | None = Field(default=None, ge=0)
    enrollment_observation: ObservationState
    planned_primary_completion_date: date
    observed_primary_completion_date_type: RegistryValueType | None
    observed_actual_primary_completion_date: date | None
    primary_completion_slippage_days: int | None
    primary_completion_observation: ObservationState
    task_semantics: ExecutionTaskSemantics = EXECUTION_TASK_SEMANTICS

    @model_validator(mode='after')
    def validate_derived_label(self) -> Self:
        if not self.registry_record_present and any(
            value is not None
            for value in (
                self.observed_registry_status,
                self.observed_enrollment_type,
                self.observed_actual_enrollment,
                self.observed_primary_completion_date_type,
                self.observed_actual_primary_completion_date,
            )
        ):
            raise ValueError('a missing registry record cannot carry observed registry values')
        expected_class = registry_outcome_class(self.registry_record_present, self.observed_registry_status)
        if self.registry_outcome_class != expected_class:
            raise ValueError('registry_outcome_class does not match the observed registry status')
        expected_failure = bool(
            self.registry_record_present and self.observed_registry_status in OBSERVED_FAILURE_STATUSES
        )
        if self.failed_status_observed != expected_failure:
            raise ValueError('failed_status_observed does not match the observed registry status')
        expected_enrollment_state = observation_state(
            self.registry_record_present,
            self.observed_actual_enrollment,
            self.observed_enrollment_type,
        )
        if self.enrollment_observation != expected_enrollment_state:
            raise ValueError('enrollment observation state is inconsistent')
        if self.observed_actual_enrollment is None:
            if self.enrollment_ratio is not None:
                raise ValueError('enrollment ratio requires observed actual enrollment')
        else:
            expected_ratio = round(
                self.observed_actual_enrollment / self.planned_enrollment,
                ENROLLMENT_RATIO_DECIMAL_PLACES,
            )
            if self.enrollment_ratio is None or not math.isclose(
                self.enrollment_ratio, expected_ratio, rel_tol=0.0, abs_tol=10**-ENROLLMENT_RATIO_DECIMAL_PLACES
            ):
                raise ValueError('enrollment ratio is not the deterministic actual/planned ratio')
        expected_completion_state = observation_state(
            self.registry_record_present,
            self.observed_actual_primary_completion_date,
            self.observed_primary_completion_date_type,
        )
        if self.primary_completion_observation != expected_completion_state:
            raise ValueError('primary-completion observation state is inconsistent')
        if self.observed_actual_primary_completion_date is None:
            if self.primary_completion_slippage_days is not None:
                raise ValueError('primary-completion slippage requires an observed actual date')
        else:
            expected_days = (self.observed_actual_primary_completion_date - self.planned_primary_completion_date).days
            if self.primary_completion_slippage_days != expected_days:
                raise ValueError('primary-completion slippage is not the deterministic date difference')
        return self


class ExecutionLabelSet(StrictModel):
    schema_version: Literal['vaxreplay.aact-execution-label-set.v0.1'] = EXECUTION_LABEL_SET_SCHEMA_VERSION
    inventory_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_rule_id: Literal['aact-plus-48-month-observable-execution-v0.1'] = EXECUTION_LABEL_RULE_ID
    outcome_rows_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    outcome_rows: tuple[AactExecutionOutcomeRow, ...]
    labels: tuple[ObservableExecutionLabel, ...]
    assigned_trial_count: int = Field(ge=0)
    missing_record_count: int = Field(ge=0)
    failed_status_count: int = Field(ge=0)
    outcome_conditioned_selection_used: Literal[False] = False
    missing_outcomes_retained: Literal[True] = True
    failed_trials_retained: Literal[True] = True
    task_semantics: ExecutionTaskSemantics = EXECUTION_TASK_SEMANTICS

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        outcome_nct_ids = tuple(row.nct_id for row in self.outcome_rows)
        if outcome_nct_ids != tuple(sorted(outcome_nct_ids)) or len(outcome_nct_ids) != len(set(outcome_nct_ids)):
            raise ValueError('outcome rows must be unique and sorted by NCT ID')
        nct_ids = tuple(label.nct_id for label in self.labels)
        if nct_ids != tuple(sorted(nct_ids)) or len(nct_ids) != len(set(nct_ids)):
            raise ValueError('labels must be unique and sorted by NCT ID')
        if self.assigned_trial_count != len(self.labels):
            raise ValueError('assigned_trial_count must equal the label count')
        if self.missing_record_count != sum(not label.registry_record_present for label in self.labels):
            raise ValueError('missing_record_count is inconsistent')
        if self.failed_status_count != sum(label.failed_status_observed for label in self.labels):
            raise ValueError('failed_status_count is inconsistent')
        return self


def add_calendar_months(value: date, months: int) -> date:
    """Add whole calendar months, clamping only when the destination month is shorter."""

    if months < 0:
        raise ValueError('months must be non-negative')
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = (next_month - date.resolution).day
    return date(year, month, min(value.day, last_day))


def registry_outcome_class(
    record_present: bool,
    status: RegistryStatus | None,
) -> RegistryOutcomeClass:
    if not record_present:
        return RegistryOutcomeClass.RECORD_MISSING
    if status is None:
        return RegistryOutcomeClass.STATUS_MISSING
    mapping = {
        RegistryStatus.COMPLETED: RegistryOutcomeClass.COMPLETED,
        RegistryStatus.TERMINATED: RegistryOutcomeClass.TERMINATED,
        RegistryStatus.WITHDRAWN: RegistryOutcomeClass.WITHDRAWN,
        RegistryStatus.SUSPENDED: RegistryOutcomeClass.SUSPENDED,
    }
    return mapping.get(status, RegistryOutcomeClass.NON_TERMINAL)


def observation_state(
    record_present: bool,
    observed_value: object | None,
    observed_type: RegistryValueType | None,
) -> ObservationState:
    """Return the only state consistent with record/value presence and the registry value type."""

    if not record_present:
        if observed_type is not None or observed_value is not None:
            raise ValueError('an absent record cannot carry a registry value type or value')
        return ObservationState.RECORD_MISSING
    if observed_type == RegistryValueType.ACTUAL:
        if observed_value is None:
            raise ValueError('an ACTUAL registry value type requires a value')
        return ObservationState.OBSERVED_ACTUAL
    if observed_value is not None:
        raise ValueError('a non-ACTUAL registry value cannot populate an observed-actual field')
    if observed_type == RegistryValueType.ANTICIPATED:
        return ObservationState.NOT_ACTUAL
    return ObservationState.VALUE_MISSING
