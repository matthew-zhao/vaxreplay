from __future__ import annotations

import unittest
from datetime import date

from pydantic import ValidationError

from vaxreplay.clinicaltrials.execution_inventory import (
    ExecutionInventoryError,
    audit_execution_inventory,
    audit_execution_label_set,
    bind_anchor_source,
    build_execution_inventory,
    derive_execution_labels,
    eligibility_reasons,
)
from vaxreplay.clinicaltrials.execution_schema import (
    EXECUTION_TASK_SEMANTICS,
    AactExecutionDecisionRow,
    AactExecutionOutcomeRow,
    DiseaseStratum,
    EligibilityReason,
    ExecutionCohortPolicy,
    NormalizedPhase,
    NormalizedStudyType,
    ObservationState,
    RegistryOutcomeClass,
    RegistryStatus,
    RegistryValueType,
    TrialAnchorAssignment,
    add_calendar_months,
)

_ANCHOR_2020 = date(2020, 1, 1)
_ANCHOR_2021 = date(2021, 1, 1)


def _decision_row(
    *,
    anchor: date,
    nct_id: str,
    lineage: str,
    stratum: DiseaseStratum = DiseaseStratum.NON_COVID_INFECTIOUS,
    planned_enrollment: int = 100,
    planned_completion: date | None = None,
    prophylactic: bool = True,
) -> AactExecutionDecisionRow:
    return AactExecutionDecisionRow(
        snapshot_id=f'aact-{anchor.isoformat()}',
        archive_date=anchor,
        source_record_sha256=(nct_id[-1].lower() * 64),
        nct_id=nct_id,
        lineage_group_id=lineage,
        disease_stratum=stratum,
        study_first_posted_date=date(anchor.year - 1, 6, 1),
        study_type=NormalizedStudyType.INTERVENTIONAL,
        phase=NormalizedPhase.PHASE_1,
        human=True,
        prophylactic_intent=prophylactic,
        infectious_disease_vaccine=True,
        biological_intervention_count=1,
        overall_status=RegistryStatus.RECRUITING,
        results_section_present=False,
        enrollment=planned_enrollment,
        enrollment_type=RegistryValueType.ANTICIPATED,
        primary_completion_date=planned_completion or date(anchor.year, 9, 1),
        primary_completion_date_type=RegistryValueType.ANTICIPATED,
    )


def _policy_and_rows() -> tuple[ExecutionCohortPolicy, tuple[AactExecutionDecisionRow, ...]]:
    first_anchor_rows = (
        _decision_row(
            anchor=_ANCHOR_2020,
            nct_id='NCT00000001',
            lineage='lineage-a',
            planned_enrollment=100,
            planned_completion=date(2020, 9, 1),
        ),
    )
    second_anchor_rows = (
        _decision_row(
            anchor=_ANCHOR_2021,
            nct_id='NCT00000001',
            lineage='lineage-a',
            planned_enrollment=120,
            planned_completion=date(2021, 10, 1),
        ),
        _decision_row(
            anchor=_ANCHOR_2021,
            nct_id='NCT00000002',
            lineage='lineage-b',
            stratum=DiseaseStratum.COVID_19,
            planned_enrollment=200,
            planned_completion=date(2021, 8, 1),
        ),
        _decision_row(
            anchor=_ANCHOR_2021,
            nct_id='NCT00000003',
            lineage='lineage-excluded',
            prophylactic=False,
        ),
        _decision_row(
            anchor=_ANCHOR_2021,
            nct_id='NCT00000004',
            lineage='lineage-b',
            stratum=DiseaseStratum.COVID_19,
            planned_enrollment=90,
            planned_completion=date(2021, 11, 1),
        ),
    )
    bindings = (
        bind_anchor_source(
            anchor_date=_ANCHOR_2020,
            decision_snapshot_id='aact-2020-01-01',
            decision_archive_manifest_sha256='a' * 64,
            label_snapshot_id='aact-2024-01-01',
            label_archive_manifest_sha256='c' * 64,
            rows=first_anchor_rows,
        ),
        bind_anchor_source(
            anchor_date=_ANCHOR_2021,
            decision_snapshot_id='aact-2021-01-01',
            decision_archive_manifest_sha256='b' * 64,
            label_snapshot_id='aact-2025-01-01',
            label_archive_manifest_sha256='d' * 64,
            rows=second_anchor_rows,
        ),
    )
    policy = ExecutionCohortPolicy(
        policy_id='fixed-anchor-execution-test',
        synthetic=True,
        selection_universe_rule_id='fictional-complete-anchor-universe-v1',
        selection_universe_rule_sha256='e' * 64,
        lineage_grouping_rule_id='fictional-lineage-rule-v1',
        lineage_grouping_rule_sha256='f' * 64,
        anchors=bindings,
    )
    return policy, first_anchor_rows + second_anchor_rows


def _outcome(
    *,
    assignment: TrialAnchorAssignment,
    status: RegistryStatus | None,
    enrollment: int | None,
    enrollment_type: RegistryValueType | None,
    primary_completion: date | None,
    primary_completion_type: RegistryValueType | None,
) -> AactExecutionOutcomeRow:
    return AactExecutionOutcomeRow(
        snapshot_id=assignment.label_snapshot_id,
        archive_date=assignment.label_archive_date,
        nct_id=assignment.nct_id,
        record_present=True,
        source_record_sha256=assignment.nct_id[-1] * 64,
        overall_status=status,
        enrollment=enrollment,
        enrollment_type=enrollment_type,
        primary_completion_date=primary_completion,
        primary_completion_date_type=primary_completion_type,
    )


class ClinicalTrialsExecutionInventoryTest(unittest.TestCase):
    def test_selects_only_from_decision_rows_and_assigns_earliest_anchor(self) -> None:
        policy, rows = _policy_and_rows()
        inventory = build_execution_inventory(policy=policy, decision_rows=reversed(rows))

        self.assertEqual(
            tuple(assignment.nct_id for assignment in inventory.assignments),
            ('NCT00000001', 'NCT00000002', 'NCT00000004'),
        )
        first = inventory.assignments[0]
        self.assertEqual(first.anchor_date, _ANCHOR_2020)
        self.assertEqual(first.planned_enrollment, 100)
        self.assertEqual(first.label_archive_date, _ANCHOR_2020.replace(year=2024))
        self.assertEqual(len(inventory.assessments), len(rows))
        excluded = next(item for item in inventory.assessments if item.nct_id == 'NCT00000003')
        self.assertFalse(excluded.eligible)
        self.assertEqual(excluded.reason_codes, (EligibilityReason.NOT_PROPHYLACTIC,))
        self.assertEqual(
            [(group.lineage_group_id, group.nct_ids) for group in inventory.lineage_groups],
            [
                ('lineage-a', ('NCT00000001',)),
                ('lineage-b', ('NCT00000002', 'NCT00000004')),
            ],
        )
        self.assertFalse(inventory.outcome_fields_used_for_selection)
        audit_execution_inventory(inventory)

    def test_labels_retain_failed_missing_and_nonactual_records(self) -> None:
        policy, rows = _policy_and_rows()
        inventory = build_execution_inventory(policy=policy, decision_rows=rows)
        by_nct = {assignment.nct_id: assignment for assignment in inventory.assignments}
        outcomes = (
            _outcome(
                assignment=by_nct['NCT00000001'],
                status=RegistryStatus.TERMINATED,
                enrollment=80,
                enrollment_type=RegistryValueType.ACTUAL,
                primary_completion=date(2020, 10, 1),
                primary_completion_type=RegistryValueType.ACTUAL,
            ),
            AactExecutionOutcomeRow(
                snapshot_id=by_nct['NCT00000002'].label_snapshot_id,
                archive_date=by_nct['NCT00000002'].label_archive_date,
                nct_id='NCT00000002',
                record_present=False,
            ),
            _outcome(
                assignment=by_nct['NCT00000004'],
                status=RegistryStatus.COMPLETED,
                enrollment=95,
                enrollment_type=RegistryValueType.ANTICIPATED,
                primary_completion=date(2022, 1, 1),
                primary_completion_type=RegistryValueType.ANTICIPATED,
            ),
        )

        label_set = derive_execution_labels(inventory=inventory, outcome_rows=reversed(outcomes))

        self.assertEqual(label_set.assigned_trial_count, 3)
        self.assertEqual(label_set.failed_status_count, 1)
        self.assertEqual(label_set.missing_record_count, 1)
        failed, missing, nonactual = label_set.labels
        self.assertEqual(failed.registry_outcome_class, RegistryOutcomeClass.TERMINATED)
        self.assertTrue(failed.failed_status_observed)
        self.assertEqual(failed.enrollment_ratio, 0.8)
        self.assertEqual(failed.primary_completion_slippage_days, 30)
        self.assertEqual(missing.registry_outcome_class, RegistryOutcomeClass.RECORD_MISSING)
        self.assertEqual(missing.enrollment_observation, ObservationState.RECORD_MISSING)
        self.assertIsNone(missing.enrollment_ratio)
        self.assertEqual(nonactual.registry_outcome_class, RegistryOutcomeClass.COMPLETED)
        self.assertEqual(nonactual.enrollment_observation, ObservationState.NOT_ACTUAL)
        self.assertEqual(nonactual.primary_completion_observation, ObservationState.NOT_ACTUAL)
        self.assertIsNone(nonactual.observed_actual_enrollment)
        self.assertIsNone(nonactual.primary_completion_slippage_days)
        self.assertEqual(nonactual.task_semantics, EXECUTION_TASK_SEMANTICS)
        audit_execution_label_set(inventory=inventory, label_set=label_set)

    def test_outcome_rows_cannot_change_or_prune_the_fixed_cohort(self) -> None:
        policy, rows = _policy_and_rows()
        inventory = build_execution_inventory(policy=policy, decision_rows=rows)
        assignment = inventory.assignments[0]
        one_outcome = _outcome(
            assignment=assignment,
            status=RegistryStatus.COMPLETED,
            enrollment=100,
            enrollment_type=RegistryValueType.ACTUAL,
            primary_completion=date(2020, 9, 1),
            primary_completion_type=RegistryValueType.ACTUAL,
        )
        with self.assertRaisesRegex(ExecutionInventoryError, 'exactly one explicit present-or-missing'):
            derive_execution_labels(inventory=inventory, outcome_rows=(one_outcome,))

    def test_bound_projection_rejects_post_binding_row_removal(self) -> None:
        policy, rows = _policy_and_rows()
        with self.assertRaisesRegex(ExecutionInventoryError, 'row count'):
            build_execution_inventory(policy=policy, decision_rows=rows[:-1])

    def test_audit_rejects_later_anchor_substitution_and_label_tampering(self) -> None:
        policy, rows = _policy_and_rows()
        inventory = build_execution_inventory(policy=policy, decision_rows=rows)
        first = inventory.assignments[0]
        later = first.model_copy(
            update={
                'anchor_date': _ANCHOR_2021,
                'decision_snapshot_id': 'aact-2021-01-01',
                'label_archive_date': date(2025, 1, 1),
                'label_snapshot_id': 'aact-2025-01-01',
            }
        )
        tampered_inventory = inventory.model_copy(update={'assignments': (later,) + inventory.assignments[1:]})
        with self.assertRaisesRegex(ExecutionInventoryError, 'deterministic reconstruction'):
            audit_execution_inventory(tampered_inventory)

        by_nct = {assignment.nct_id: assignment for assignment in inventory.assignments}
        outcomes = tuple(
            AactExecutionOutcomeRow(
                snapshot_id=assignment.label_snapshot_id,
                archive_date=assignment.label_archive_date,
                nct_id=assignment.nct_id,
                record_present=False,
            )
            for assignment in by_nct.values()
        )
        label_set = derive_execution_labels(inventory=inventory, outcome_rows=outcomes)
        changed_label = label_set.labels[0].model_copy(update={'registry_record_present': True})
        tampered_labels = label_set.model_copy(update={'labels': (changed_label,) + label_set.labels[1:]})
        with self.assertRaisesRegex(ExecutionInventoryError, 'deterministic reconstruction'):
            audit_execution_label_set(inventory=inventory, label_set=tampered_labels)

    def test_decision_schema_forbids_postcutoff_outcome_fields(self) -> None:
        row = _decision_row(
            anchor=_ANCHOR_2020,
            nct_id='NCT00000001',
            lineage='lineage-a',
        )
        payload = row.model_dump(mode='json')
        payload['later_actual_enrollment'] = 87
        with self.assertRaises(ValidationError):
            AactExecutionDecisionRow.model_validate(payload)

        policy, _ = _policy_and_rows()
        policy_payload = policy.model_dump(mode='json')
        policy_payload['outcome_conditioned_selection_prohibited'] = False
        with self.assertRaises(ValidationError):
            ExecutionCohortPolicy.model_validate(policy_payload)

    def test_ineligible_rows_are_retained_with_all_fixed_reasons(self) -> None:
        row = _decision_row(
            anchor=_ANCHOR_2020,
            nct_id='NCT00000001',
            lineage='lineage-a',
            prophylactic=False,
        ).model_copy(
            update={
                'study_type': NormalizedStudyType.OTHER,
                'phase': NormalizedPhase.OTHER,
                'human': False,
                'infectious_disease_vaccine': False,
                'biological_intervention_count': 0,
                'overall_status': RegistryStatus.COMPLETED,
                'results_section_present': True,
                'enrollment': 10,
                'enrollment_type': RegistryValueType.ACTUAL,
                'primary_completion_date': date(2019, 12, 1),
                'primary_completion_date_type': RegistryValueType.ANTICIPATED,
            }
        )
        reasons = eligibility_reasons(row)
        self.assertEqual(reasons, tuple(sorted(reasons, key=lambda value: value.value)))
        self.assertIn(EligibilityReason.NOT_INTERVENTIONAL, reasons)
        self.assertIn(EligibilityReason.PHASE_NOT_EARLY_CLINICAL, reasons)
        self.assertIn(EligibilityReason.RESULTS_ALREADY_PRESENT, reasons)
        self.assertIn(EligibilityReason.PLANNED_ENROLLMENT_NOT_ANTICIPATED, reasons)
        self.assertIn(EligibilityReason.PLANNED_PRIMARY_COMPLETION_NOT_AFTER_ANCHOR, reasons)

    def test_lineage_cannot_cross_covid_strata(self) -> None:
        policy, rows = _policy_and_rows()
        crossed = rows[-1].model_copy(update={'disease_stratum': DiseaseStratum.NON_COVID_INFECTIOUS})
        replacement_rows = rows[:-1] + (crossed,)
        second_anchor_rows = tuple(row for row in replacement_rows if row.archive_date == _ANCHOR_2021)
        replacement_binding = bind_anchor_source(
            anchor_date=_ANCHOR_2021,
            decision_snapshot_id='aact-2021-01-01',
            decision_archive_manifest_sha256='b' * 64,
            label_snapshot_id='aact-2025-01-01',
            label_archive_manifest_sha256='d' * 64,
            rows=second_anchor_rows,
        )
        replacement_policy = policy.model_copy(update={'anchors': (policy.anchors[0], replacement_binding)})
        with self.assertRaisesRegex(ExecutionInventoryError, 'cannot cross'):
            build_execution_inventory(policy=replacement_policy, decision_rows=replacement_rows)

    def test_calendar_horizon_is_exactly_48_months(self) -> None:
        self.assertEqual(add_calendar_months(date(2020, 2, 29), 48), date(2024, 2, 29))
        policy, _ = _policy_and_rows()
        binding = policy.anchors[0].model_copy(update={'label_archive_date': date(2023, 12, 1)})
        with self.assertRaisesRegex(ValidationError, '48 calendar months'):
            ExecutionCohortPolicy(
                policy_id='bad-horizon',
                synthetic=True,
                selection_universe_rule_id='fictional-complete-anchor-universe-v1',
                selection_universe_rule_sha256='e' * 64,
                lineage_grouping_rule_id='fictional-lineage-rule-v1',
                lineage_grouping_rule_sha256='f' * 64,
                anchors=(binding,),
            )


if __name__ == '__main__':
    unittest.main()
