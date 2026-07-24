"""Deterministic cohort selection and private labels for AACT execution replay.

The selection path accepts only historical decision-row projections.  Outcome rows enter through a
separate label derivation function after cohort membership and earliest-anchor assignment are fixed.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_schema import (
    ELIGIBLE_DECISION_STATUSES,
    ELIGIBLE_PHASES,
    ENROLLMENT_RATIO_DECIMAL_PLACES,
    EXECUTION_LABEL_RULE_ID,
    LABEL_HORIZON_MONTHS,
    OBSERVED_FAILURE_STATUSES,
    AactExecutionDecisionRow,
    AactExecutionOutcomeRow,
    AnchorEligibilityAssessment,
    AnchorSourceBinding,
    DiseaseStratum,
    EligibilityReason,
    ExecutionCohortInventory,
    ExecutionCohortPolicy,
    ExecutionLabelSet,
    NormalizedStudyType,
    ObservableExecutionLabel,
    RegistryValueType,
    TrialAnchorAssignment,
    TrialLineageGroup,
    add_calendar_months,
    observation_state,
    registry_outcome_class,
)


class ExecutionInventoryError(ValueError):
    """Raised when a source projection, inventory, or label set violates the fixed contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_sequence_bytes(values: Sequence[object]) -> bytes:
    dumped: list[object] = []
    for value in values:
        model_dump = getattr(value, 'model_dump', None)
        dumped.append(model_dump(mode='json') if model_dump is not None else value)
    return canonical_json_bytes(dumped)


def decision_rows_sha256(rows: Iterable[AactExecutionDecisionRow]) -> str:
    ordered = tuple(sorted(rows, key=lambda row: (row.archive_date, row.nct_id)))
    return _sha256_bytes(_model_sequence_bytes(ordered))


def outcome_rows_sha256(rows: Iterable[AactExecutionOutcomeRow]) -> str:
    ordered = tuple(sorted(rows, key=lambda row: row.nct_id))
    return _sha256_bytes(_model_sequence_bytes(ordered))


def bind_anchor_source(
    *,
    anchor_date: date,
    decision_snapshot_id: str,
    decision_archive_manifest_sha256: str,
    label_snapshot_id: str,
    label_archive_manifest_sha256: str,
    rows: Iterable[AactExecutionDecisionRow],
) -> AnchorSourceBinding:
    """Bind a complete normalized decision projection before cohort construction."""

    ordered = tuple(sorted(rows, key=lambda row: row.nct_id))
    seen: set[str] = set()
    for row in ordered:
        if row.archive_date != anchor_date or row.snapshot_id != decision_snapshot_id:
            raise ExecutionInventoryError('anchor projection contains a row from a different archive')
        if row.nct_id in seen:
            raise ExecutionInventoryError('anchor projection contains duplicate NCT IDs')
        seen.add(row.nct_id)
    return AnchorSourceBinding(
        anchor_date=anchor_date,
        decision_snapshot_id=decision_snapshot_id,
        decision_archive_manifest_sha256=decision_archive_manifest_sha256,
        decision_rows_sha256=_sha256_bytes(_model_sequence_bytes(ordered)),
        decision_row_count=len(ordered),
        label_snapshot_id=label_snapshot_id,
        label_archive_date=add_calendar_months(anchor_date, LABEL_HORIZON_MONTHS),
        label_archive_manifest_sha256=label_archive_manifest_sha256,
    )


def eligibility_reasons(row: AactExecutionDecisionRow) -> tuple[EligibilityReason, ...]:
    """Apply the fixed, outcome-free eligibility rule to one historical row."""

    reasons: set[EligibilityReason] = set()
    if row.study_type != NormalizedStudyType.INTERVENTIONAL:
        reasons.add(EligibilityReason.NOT_INTERVENTIONAL)
    if row.phase not in ELIGIBLE_PHASES:
        reasons.add(EligibilityReason.PHASE_NOT_EARLY_CLINICAL)
    if not row.human:
        reasons.add(EligibilityReason.NOT_HUMAN)
    if not row.prophylactic_intent:
        reasons.add(EligibilityReason.NOT_PROPHYLACTIC)
    if not row.infectious_disease_vaccine:
        reasons.add(EligibilityReason.NOT_INFECTIOUS_DISEASE_VACCINE)
    if row.biological_intervention_count < 1:
        reasons.add(EligibilityReason.NO_BIOLOGICAL_INTERVENTION)
    if row.results_section_present or row.results_first_posted_date is not None:
        reasons.add(EligibilityReason.RESULTS_ALREADY_PRESENT)
    if row.overall_status not in ELIGIBLE_DECISION_STATUSES:
        reasons.add(EligibilityReason.DECISION_STATUS_INELIGIBLE)
    if row.enrollment is None or row.enrollment <= 0:
        reasons.add(EligibilityReason.PLANNED_ENROLLMENT_MISSING)
    elif row.enrollment_type != RegistryValueType.ANTICIPATED:
        reasons.add(EligibilityReason.PLANNED_ENROLLMENT_NOT_ANTICIPATED)
    if row.primary_completion_date is None:
        reasons.add(EligibilityReason.PLANNED_PRIMARY_COMPLETION_MISSING)
    elif row.primary_completion_date_type != RegistryValueType.ANTICIPATED:
        reasons.add(EligibilityReason.PLANNED_PRIMARY_COMPLETION_NOT_ANTICIPATED)
    elif row.primary_completion_date <= row.archive_date:
        reasons.add(EligibilityReason.PLANNED_PRIMARY_COMPLETION_NOT_AFTER_ANCHOR)
    return tuple(sorted(reasons, key=lambda reason: reason.value))


def _validate_and_group_decision_rows(
    policy: ExecutionCohortPolicy,
    decision_rows: Iterable[AactExecutionDecisionRow],
) -> tuple[
    tuple[AactExecutionDecisionRow, ...],
    tuple[AnchorEligibilityAssessment, ...],
    tuple[TrialAnchorAssignment, ...],
    tuple[TrialLineageGroup, ...],
]:
    ordered_rows = tuple(sorted(decision_rows, key=lambda row: (row.archive_date, row.nct_id)))
    binding_by_anchor = {binding.anchor_date: binding for binding in policy.anchors}
    rows_by_anchor: dict[date, list[AactExecutionDecisionRow]] = defaultdict(list)
    seen_keys: set[tuple[date, str]] = set()
    identity_by_nct: dict[str, tuple[str, DiseaseStratum]] = {}
    classification_by_nct: dict[str, tuple[bool, bool, bool]] = {}
    stratum_by_lineage: dict[str, DiseaseStratum] = {}

    for row in ordered_rows:
        binding = binding_by_anchor.get(row.archive_date)
        if binding is None:
            raise ExecutionInventoryError('decision row references an anchor outside the fixed policy')
        if row.snapshot_id != binding.decision_snapshot_id:
            raise ExecutionInventoryError('decision row snapshot does not match its anchor binding')
        key = (row.archive_date, row.nct_id)
        if key in seen_keys:
            raise ExecutionInventoryError('decision rows contain a duplicate anchor/NCT pair')
        seen_keys.add(key)
        identity = (row.lineage_group_id, row.disease_stratum)
        previous_identity = identity_by_nct.setdefault(row.nct_id, identity)
        if previous_identity != identity:
            raise ExecutionInventoryError('NCT lineage and COVID stratum must remain stable across anchors')
        classification = (row.human, row.prophylactic_intent, row.infectious_disease_vaccine)
        previous_classification = classification_by_nct.setdefault(row.nct_id, classification)
        if previous_classification != classification:
            raise ExecutionInventoryError('precommitted human and vaccine classifications must remain stable')
        previous_stratum = stratum_by_lineage.setdefault(row.lineage_group_id, row.disease_stratum)
        if previous_stratum != row.disease_stratum:
            raise ExecutionInventoryError('one lineage cannot cross the COVID and non-COVID strata')
        rows_by_anchor[row.archive_date].append(row)

    for binding in policy.anchors:
        anchor_rows = tuple(sorted(rows_by_anchor.get(binding.anchor_date, ()), key=lambda row: row.nct_id))
        if len(anchor_rows) != binding.decision_row_count:
            raise ExecutionInventoryError('complete anchor projection row count does not match its binding')
        if _sha256_bytes(_model_sequence_bytes(anchor_rows)) != binding.decision_rows_sha256:
            raise ExecutionInventoryError('complete anchor projection hash does not match its binding')

    assessments = tuple(
        AnchorEligibilityAssessment(
            snapshot_id=row.snapshot_id,
            anchor_date=row.archive_date,
            nct_id=row.nct_id,
            eligible=not (reasons := eligibility_reasons(row)),
            reason_codes=reasons,
        )
        for row in ordered_rows
    )
    row_by_key = {(row.archive_date, row.nct_id): row for row in ordered_rows}
    eligible_anchors_by_nct: dict[str, list[date]] = defaultdict(list)
    for assessment in assessments:
        if assessment.eligible:
            eligible_anchors_by_nct[assessment.nct_id].append(assessment.anchor_date)

    assignments: list[TrialAnchorAssignment] = []
    for nct_id in sorted(eligible_anchors_by_nct):
        earliest_anchor = min(eligible_anchors_by_nct[nct_id])
        row = row_by_key[(earliest_anchor, nct_id)]
        binding = binding_by_anchor[earliest_anchor]
        if row.enrollment is None or row.primary_completion_date is None:
            raise ExecutionInventoryError('eligible row unexpectedly lacks its planned execution fields')
        assignments.append(
            TrialAnchorAssignment(
                nct_id=nct_id,
                lineage_group_id=row.lineage_group_id,
                disease_stratum=row.disease_stratum,
                decision_snapshot_id=row.snapshot_id,
                anchor_date=earliest_anchor,
                label_snapshot_id=binding.label_snapshot_id,
                label_archive_date=binding.label_archive_date,
                planned_enrollment=row.enrollment,
                planned_primary_completion_date=row.primary_completion_date,
            )
        )

    ncts_by_lineage: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        ncts_by_lineage[assignment.lineage_group_id].append(assignment.nct_id)
    lineage_groups = tuple(
        TrialLineageGroup(
            lineage_group_id=lineage_group_id,
            disease_stratum=stratum_by_lineage[lineage_group_id],
            nct_ids=tuple(sorted(nct_ids)),
        )
        for lineage_group_id, nct_ids in sorted(ncts_by_lineage.items())
    )
    return ordered_rows, assessments, tuple(assignments), lineage_groups


def build_execution_inventory(
    *,
    policy: ExecutionCohortPolicy,
    decision_rows: Iterable[AactExecutionDecisionRow],
) -> ExecutionCohortInventory:
    """Select the cohort using only bound historical decision projections."""

    validated_policy = ExecutionCohortPolicy.model_validate_json(canonical_json_bytes(policy))
    validated_rows = tuple(
        AactExecutionDecisionRow.model_validate_json(canonical_json_bytes(row)) for row in decision_rows
    )
    rows, assessments, assignments, lineage_groups = _validate_and_group_decision_rows(
        validated_policy,
        validated_rows,
    )
    return ExecutionCohortInventory(
        policy=validated_policy,
        policy_sha256=_sha256_bytes(canonical_json_bytes(validated_policy)),
        decision_rows_sha256=decision_rows_sha256(rows),
        decision_rows=rows,
        assessments=assessments,
        assignments=assignments,
        lineage_groups=lineage_groups,
    )


def audit_execution_inventory(inventory: ExecutionCohortInventory) -> None:
    """Independently rebuild all reward-relevant cohort-selection material."""

    rebuilt = build_execution_inventory(policy=inventory.policy, decision_rows=inventory.decision_rows)
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(inventory):
        raise ExecutionInventoryError('execution inventory does not match deterministic reconstruction')


def _derive_label(
    assignment: TrialAnchorAssignment,
    outcome: AactExecutionOutcomeRow,
) -> ObservableExecutionLabel:
    if outcome.snapshot_id != assignment.label_snapshot_id or outcome.archive_date != assignment.label_archive_date:
        raise ExecutionInventoryError('outcome row does not come from the assigned +48-month snapshot')

    actual_enrollment = (
        outcome.enrollment if outcome.record_present and outcome.enrollment_type == RegistryValueType.ACTUAL else None
    )
    enrollment_state = observation_state(outcome.record_present, actual_enrollment, outcome.enrollment_type)
    enrollment_ratio = (
        round(actual_enrollment / assignment.planned_enrollment, ENROLLMENT_RATIO_DECIMAL_PLACES)
        if actual_enrollment is not None
        else None
    )
    actual_primary_completion = (
        outcome.primary_completion_date
        if outcome.record_present and outcome.primary_completion_date_type == RegistryValueType.ACTUAL
        else None
    )
    primary_completion_state = observation_state(
        outcome.record_present,
        actual_primary_completion,
        outcome.primary_completion_date_type,
    )
    slippage_days = (
        (actual_primary_completion - assignment.planned_primary_completion_date).days
        if actual_primary_completion is not None
        else None
    )
    status = outcome.overall_status if outcome.record_present else None
    return ObservableExecutionLabel(
        nct_id=assignment.nct_id,
        lineage_group_id=assignment.lineage_group_id,
        disease_stratum=assignment.disease_stratum,
        anchor_date=assignment.anchor_date,
        label_snapshot_id=assignment.label_snapshot_id,
        label_archive_date=assignment.label_archive_date,
        registry_record_present=outcome.record_present,
        observed_registry_status=status,
        registry_outcome_class=registry_outcome_class(outcome.record_present, status),
        failed_status_observed=bool(outcome.record_present and status in OBSERVED_FAILURE_STATUSES),
        planned_enrollment=assignment.planned_enrollment,
        observed_enrollment_type=outcome.enrollment_type if outcome.record_present else None,
        observed_actual_enrollment=actual_enrollment,
        enrollment_ratio=enrollment_ratio,
        enrollment_observation=enrollment_state,
        planned_primary_completion_date=assignment.planned_primary_completion_date,
        observed_primary_completion_date_type=(
            outcome.primary_completion_date_type if outcome.record_present else None
        ),
        observed_actual_primary_completion_date=actual_primary_completion,
        primary_completion_slippage_days=slippage_days,
        primary_completion_observation=primary_completion_state,
    )


def derive_execution_labels(
    *,
    inventory: ExecutionCohortInventory,
    outcome_rows: Iterable[AactExecutionOutcomeRow],
) -> ExecutionLabelSet:
    """Derive labels without allowing later records to add or remove cohort members."""

    audit_execution_inventory(inventory)
    validated_outcomes = tuple(
        AactExecutionOutcomeRow.model_validate_json(canonical_json_bytes(row)) for row in outcome_rows
    )
    ordered_outcomes = tuple(sorted(validated_outcomes, key=lambda row: row.nct_id))
    expected_nct_ids = tuple(assignment.nct_id for assignment in inventory.assignments)
    observed_nct_ids = tuple(row.nct_id for row in ordered_outcomes)
    if observed_nct_ids != expected_nct_ids:
        raise ExecutionInventoryError(
            'outcome rows must contain exactly one explicit present-or-missing row for every assigned trial'
        )
    labels = tuple(
        _derive_label(assignment, outcome)
        for assignment, outcome in zip(inventory.assignments, ordered_outcomes, strict=True)
    )
    return ExecutionLabelSet(
        inventory_sha256=_sha256_bytes(canonical_json_bytes(inventory)),
        label_rule_id=EXECUTION_LABEL_RULE_ID,
        outcome_rows_sha256=outcome_rows_sha256(ordered_outcomes),
        outcome_rows=ordered_outcomes,
        labels=labels,
        assigned_trial_count=len(labels),
        missing_record_count=sum(not label.registry_record_present for label in labels),
        failed_status_count=sum(label.failed_status_observed for label in labels),
    )


def audit_execution_label_set(
    *,
    inventory: ExecutionCohortInventory,
    label_set: ExecutionLabelSet,
) -> None:
    """Rebuild the private gold from its bound outcome rows and selected inventory."""

    rebuilt = derive_execution_labels(inventory=inventory, outcome_rows=label_set.outcome_rows)
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(label_set):
        raise ExecutionInventoryError('execution labels do not match deterministic reconstruction')
