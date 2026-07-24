from __future__ import annotations

import csv
import io
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

from vaxreplay.clinicaltrials.execution_adapter import (
    AactExecutionAdapterError,
    AactExecutionBuildReceipt,
    ScreenedDecisionRecordSet,
    build_aact_execution_cohort,
    main,
)
from vaxreplay.clinicaltrials.execution_schema import (
    DiseaseStratum,
    RegistryOutcomeClass,
    RegistryValueType,
)

_STUDY_FIELDS = (
    'nct_id',
    'study_first_posted_date',
    'results_first_posted_date',
    'study_type',
    'acronym',
    'brief_title',
    'official_title',
    'overall_status',
    'phase',
    'enrollment',
    'enrollment_type',
    'primary_completion_date',
    'primary_completion_date_type',
)


def _table_bytes(fields: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields, delimiter='|', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _study(
    nct_id: str,
    *,
    title: str,
    study_type: str = 'Interventional',
    phase: str = 'Phase 1',
    status: str = 'Recruiting',
    enrollment: str = '100',
    enrollment_type: str = 'Anticipated',
    completion: str = '2021-01-01',
    completion_type: str = 'Anticipated',
    first_posted: str = '2019-06-01',
    results_posted: str = '',
) -> dict[str, str]:
    return {
        'nct_id': nct_id,
        'study_first_posted_date': first_posted,
        'results_first_posted_date': results_posted,
        'study_type': study_type,
        'acronym': '',
        'brief_title': title,
        'official_title': title,
        'overall_status': status,
        'phase': phase,
        'enrollment': enrollment,
        'enrollment_type': enrollment_type,
        'primary_completion_date': completion,
        'primary_completion_date_type': completion_type,
    }


def _write_decision_zip(
    path: Path,
    *,
    decision_anchor: date = date(2020, 2, 1),
    include_second_study: bool = True,
    first_lead_sponsor: str = 'Moderna',
) -> None:
    studies = [
        _study('NCT00000001', title='SARS-CoV-2 vaccine safety and immunogenicity'),
        _study(
            'NCT00000002',
            title='Quadrivalent influenza vaccine study',
            phase='Phase 1/Phase 2',
            status='Active, not recruiting',
            enrollment='120',
            completion='2021-06-01',
        ),
        _study('NCT00000003', title='Therapeutic melanoma vaccine'),
        _study('NCT00000004', title='Nivolumab in cancer'),
        _study(
            'NCT00000005',
            title='Observational influenza vaccine registry',
            study_type='Observational',
            phase='N/A',
        ),
        _study('NCT00000006', title='Unrelated device feasibility study'),
    ]
    if decision_anchor != date(2020, 2, 1):
        for study in studies:
            study['study_first_posted_date'] = f'{decision_anchor.year - 1}-06-01'
            study['primary_completion_date'] = f'{decision_anchor.year + 1}-06-01'
    if not include_second_study:
        studies = [study for study in studies if study['nct_id'] != 'NCT00000002']
    interventions = [
        {
            'nct_id': 'NCT00000001',
            'intervention_type': 'Biological',
            'name': 'mRNA-1273',
            'description': 'A candidate vaccine with a quoted "sequence"\nand a second line.',
        },
        {
            'nct_id': 'NCT00000002',
            'intervention_type': 'Biological',
            'name': 'QIV-2020',
            'description': 'Quadrivalent influenza vaccine',
        },
        {
            'nct_id': 'NCT00000003',
            'intervention_type': 'Biological',
            'name': 'Melanoma vaccine',
            'description': 'Therapeutic cancer product',
        },
        {
            'nct_id': 'NCT00000004',
            'intervention_type': 'Biological',
            'name': 'Nivolumab',
            'description': 'Checkpoint inhibitor',
        },
        {
            'nct_id': 'NCT00000005',
            'intervention_type': 'Other',
            'name': 'Influenza vaccine exposure',
            'description': 'No assigned intervention',
        },
        {
            'nct_id': 'NCT00000006',
            'intervention_type': 'Device',
            'name': 'Device A',
            'description': '',
        },
    ]
    conditions = [
        {'nct_id': 'NCT00000001', 'name': 'COVID-19'},
        {'nct_id': 'NCT00000002', 'name': 'Influenza'},
        {'nct_id': 'NCT00000003', 'name': 'Melanoma'},
        {'nct_id': 'NCT00000004', 'name': 'Cancer'},
        {'nct_id': 'NCT00000005', 'name': 'Influenza'},
        {'nct_id': 'NCT00000006', 'name': 'Pain'},
    ]
    keywords = [
        {'nct_id': 'NCT00000001', 'name': 'vaccine'},
        {'nct_id': 'NCT00000002', 'name': 'immunization'},
        {'nct_id': 'NCT00000003', 'name': 'cancer vaccine'},
        {'nct_id': 'NCT00000005', 'name': 'vaccination'},
    ]
    sponsors = [
        {
            'nct_id': 'NCT00000001',
            'agency_class': 'Industry',
            'lead_or_collaborator': 'lead',
            'name': first_lead_sponsor,
        },
        {'nct_id': 'NCT00000002', 'agency_class': 'Industry', 'lead_or_collaborator': 'lead', 'name': 'Sanofi'},
        {'nct_id': 'NCT00000003', 'agency_class': 'Other', 'lead_or_collaborator': 'lead', 'name': 'Cancer Center'},
        {'nct_id': 'NCT00000004', 'agency_class': 'Industry', 'lead_or_collaborator': 'lead', 'name': 'BMS'},
        {'nct_id': 'NCT00000005', 'agency_class': 'Other', 'lead_or_collaborator': 'lead', 'name': 'Registry'},
        {'nct_id': 'NCT00000006', 'agency_class': 'Other', 'lead_or_collaborator': 'lead', 'name': 'Device Lab'},
    ]
    designs = [
        {'nct_id': 'NCT00000001', 'primary_purpose': 'Prevention'},
        {'nct_id': 'NCT00000002', 'primary_purpose': 'Prevention'},
        {'nct_id': 'NCT00000003', 'primary_purpose': 'Treatment'},
        {'nct_id': 'NCT00000004', 'primary_purpose': 'Treatment'},
        {'nct_id': 'NCT00000005', 'primary_purpose': 'Prevention'},
        {'nct_id': 'NCT00000006', 'primary_purpose': 'Other'},
    ]
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('studies.txt', _table_bytes(_STUDY_FIELDS, studies))
        archive.writestr(
            'interventions.txt',
            _table_bytes(('nct_id', 'intervention_type', 'name', 'description'), interventions),
        )
        archive.writestr('conditions.txt', _table_bytes(('nct_id', 'name'), conditions))
        archive.writestr('keywords.txt', _table_bytes(('nct_id', 'name'), keywords))
        archive.writestr(
            'sponsors.txt',
            _table_bytes(('nct_id', 'agency_class', 'lead_or_collaborator', 'name'), sponsors),
        )
        archive.writestr('designs.txt', _table_bytes(('nct_id', 'primary_purpose'), designs))


def _write_label_zip(path: Path, *, include_extra: bool = False, include_present_but_blank: bool = False) -> None:
    studies = [
        _study(
            'NCT00000001',
            title='Later record',
            status='Terminated',
            enrollment='80',
            enrollment_type='Actual',
            completion='2021-03-01',
            completion_type='Actual',
        ),
        _study(
            'NCT00000003',
            title='Unselected later record',
            status='Completed',
            enrollment='100',
            enrollment_type='Actual',
            completion='2021-01-01',
            completion_type='Actual',
        ),
    ]
    if include_extra:
        studies.append(
            _study(
                'NCT99999999',
                title='Extra outcome-only vaccine record',
                status='Completed',
                enrollment='500',
                enrollment_type='Actual',
                completion='2022-01-01',
                completion_type='Actual',
            )
        )
    if include_present_but_blank:
        studies.append(
            _study(
                'NCT00000002',
                title='Present record with execution fields missing',
                status='',
                enrollment='',
                enrollment_type='',
                completion='',
                completion_type='',
            )
        )
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('studies.txt', _table_bytes(_STUDY_FIELDS, studies))


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


class AactExecutionAdapterTests(unittest.TestCase):
    def test_synthetic_fixture_cannot_claim_real_and_retains_exclusions_and_missing_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_zip = root / 'decision.zip'
            label_zip = root / 'label.zip'
            output = root / 'cohort'
            _write_decision_zip(decision_zip)
            _write_label_zip(label_zip)

            build = build_aact_execution_cohort(
                decision_archive=decision_zip,
                decision_archive_date=date(2020, 2, 1),
                label_archive=label_zip,
                label_archive_date=date(2024, 2, 1),
                output_root=output,
                synthetic_test_only=True,
            )

            self.assertTrue(build.receipt.synthetic)
            self.assertEqual(build.receipt.release_status, 'synthetic_test_only')
            self.assertEqual(build.receipt.source_binding.mode, 'synthetic_test_only')
            self.assertFalse(build.receipt.tier_b_admitted)
            self.assertFalse(build.receipt.tier_a_official)
            self.assertFalse(build.receipt.biological_efficacy_claimed)
            self.assertFalse(build.receipt.active_vaccination_adjudication_bound)
            self.assertFalse(build.receipt.scored_cohort_eligible)
            self.assertTrue(build.receipt.manual_lineage_review_required)
            self.assertFalse(build.receipt.lineage_split_safe)
            self.assertEqual(build.receipt.screened_record_count, 5)
            self.assertEqual(build.receipt.normalized_record_count, 5)
            self.assertEqual(build.receipt.assigned_trial_count, 2)
            self.assertEqual(build.receipt.covid_assigned_trial_count, 1)
            self.assertEqual(build.receipt.non_covid_assigned_trial_count, 1)
            self.assertEqual(build.receipt.missing_label_record_count, 1)
            self.assertEqual(
                tuple(assignment.nct_id for assignment in build.inventory.assignments),
                ('NCT00000001', 'NCT00000002'),
            )
            self.assertEqual(build.inventory.assignments[0].disease_stratum, DiseaseStratum.COVID_19)

            excluded = next(
                assessment for assessment in build.inventory.assessments if assessment.nct_id == 'NCT00000004'
            )
            self.assertFalse(excluded.eligible)
            self.assertTrue(excluded.reason_codes)
            screened = ScreenedDecisionRecordSet.model_validate_json(
                (output / 'organizer' / 'screened-decision-records.json').read_bytes()
            ).records
            self.assertEqual(
                tuple(record.nct_id for record in screened), tuple(sorted(record.nct_id for record in screened))
            )
            source = screened[0].source_rows[0]
            self.assertGreater(source.byte_end, source.byte_start)
            self.assertEqual(len(source.raw_row_sha256), 64)
            self.assertEqual(source.fields_read, tuple(sorted(source.fields_read)))

            failed, missing = build.labels.labels
            self.assertEqual(failed.registry_outcome_class, RegistryOutcomeClass.TERMINATED)
            self.assertEqual(failed.observed_enrollment_type, RegistryValueType.ACTUAL)
            self.assertEqual(failed.enrollment_ratio, 0.8)
            self.assertEqual(failed.primary_completion_slippage_days, 59)
            self.assertEqual(missing.registry_outcome_class, RegistryOutcomeClass.RECORD_MISSING)
            self.assertFalse(missing.registry_record_present)
            receipt = AactExecutionBuildReceipt.model_validate_json((output / 'BUILD-RECEIPT.json').read_bytes())
            self.assertTrue(receipt.selection_frozen_before_label_studies_member_opened)
            self.assertTrue(all(artifact.organizer_private for artifact in receipt.artifacts))

    def test_outputs_are_byte_deterministic_and_outcome_only_records_cannot_join_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_zip = root / 'decision.zip'
            label_zip = root / 'label.zip'
            _write_decision_zip(decision_zip)
            _write_label_zip(label_zip, include_extra=True)

            first = build_aact_execution_cohort(
                decision_archive=decision_zip,
                decision_archive_date=date(2020, 2, 1),
                label_archive=label_zip,
                label_archive_date=date(2024, 2, 1),
                output_root=root / 'first',
                synthetic_test_only=True,
            )
            second = build_aact_execution_cohort(
                decision_archive=decision_zip,
                decision_archive_date=date(2020, 2, 1),
                label_archive=label_zip,
                label_archive_date=date(2024, 2, 1),
                output_root=root / 'second',
                synthetic_test_only=True,
            )

            self.assertEqual(_tree_bytes(first.root), _tree_bytes(second.root))
            self.assertEqual(tuple(label.nct_id for label in first.labels.labels), ('NCT00000001', 'NCT00000002'))
            self.assertNotIn('NCT99999999', {row.nct_id for row in first.labels.outcome_rows})

    def test_cli_defaults_target_2020_to_2024_and_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_zip = root / 'decision.zip'
            label_zip = root / 'label.zip'
            output = root / 'cohort'
            _write_decision_zip(decision_zip)
            _write_label_zip(label_zip)

            self.assertEqual(
                main(
                    [
                        '--decision-archive',
                        str(decision_zip),
                        '--label-archive',
                        str(label_zip),
                        '--synthetic-test-only',
                        '--output-root',
                        str(output),
                    ]
                ),
                0,
            )
            with self.assertRaises(FileExistsError):
                build_aact_execution_cohort(
                    decision_archive=decision_zip,
                    decision_archive_date=date(2020, 2, 1),
                    label_archive=label_zip,
                    label_archive_date=date(2024, 2, 1),
                    output_root=output,
                    synthetic_test_only=True,
                )

    def test_present_record_with_missing_fields_is_not_conflated_with_missing_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_zip = root / 'decision.zip'
            label_zip = root / 'label.zip'
            _write_decision_zip(decision_zip)
            _write_label_zip(label_zip, include_present_but_blank=True)
            build = build_aact_execution_cohort(
                decision_archive=decision_zip,
                decision_archive_date=date(2020, 2, 1),
                label_archive=label_zip,
                label_archive_date=date(2024, 2, 1),
                output_root=root / 'cohort',
                synthetic_test_only=True,
            )

            blank = next(label for label in build.labels.labels if label.nct_id == 'NCT00000002')
            self.assertTrue(blank.registry_record_present)
            self.assertEqual(blank.registry_outcome_class, RegistryOutcomeClass.STATUS_MISSING)
            self.assertIsNone(blank.observed_registry_status)
            self.assertEqual(build.labels.missing_record_count, 0)

    def test_requires_exact_48_month_label_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_zip = root / 'decision.zip'
            label_zip = root / 'label.zip'
            _write_decision_zip(decision_zip)
            _write_label_zip(label_zip)
            with self.assertRaisesRegex(AactExecutionAdapterError, r'exactly \+48 calendar months'):
                build_aact_execution_cohort(
                    decision_archive=decision_zip,
                    decision_archive_date=date(2020, 2, 1),
                    label_archive=label_zip,
                    label_archive_date=date(2024, 3, 1),
                    output_root=root / 'cohort',
                    synthetic_test_only=True,
                )

    def test_generic_archives_cannot_emit_real_data_without_trusted_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_zip = root / 'decision.zip'
            label_zip = root / 'label.zip'
            _write_decision_zip(decision_zip)
            _write_label_zip(label_zip)
            with self.assertRaisesRegex(AactExecutionAdapterError, 'provide exactly one source mode'):
                build_aact_execution_cohort(
                    decision_archive=decision_zip,
                    decision_archive_date=date(2020, 2, 1),
                    label_archive=label_zip,
                    label_archive_date=date(2024, 2, 1),
                    output_root=root / 'cohort',
                )


if __name__ == '__main__':
    unittest.main()
