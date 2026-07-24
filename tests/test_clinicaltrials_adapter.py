from __future__ import annotations

import csv
import hashlib
import hmac
import json
import tempfile
import unittest
from datetime import date, datetime, time, timezone
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.adapter import AactAdapterError, audit_episode, build_episode
from vaxreplay.clinicaltrials.extract import inspect_table
from vaxreplay.clinicaltrials.schema import (
    AactArchiveReceipt,
    AactPrivateAudit,
    AactSliceReceipt,
    AactSourceTable,
    ArmMappingSpec,
    ArmRole,
    EarlyClinicalEpisodeSpec,
    OutcomeEndpointSpec,
    OutcomeRubric,
    PanelSelector,
)

_UTC = timezone.utc
_NCT = 'NCT00000001'
_DECISION_AT = datetime.combine(date(2020, 2, 1), time.max, tzinfo=_UTC)
_OUTCOME_AT = datetime.combine(date(2022, 4, 1), time.max, tzinfo=_UTC)


def _write_table(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as destination:
        writer = csv.writer(destination, delimiter='|', quotechar='"', lineterminator='\n')
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(column, '') for column in header])


def _archive(snapshot_id: str, source_build_at: datetime, payload: bytes) -> AactArchiveReceipt:
    return AactArchiveReceipt(
        snapshot_id=snapshot_id,
        archive_date=source_build_at.date(),
        source_cutoff_at=source_build_at,
        retrieved_at=datetime(2026, 7, 13, tzinfo=_UTC),
        source_url=f'https://example.test/{snapshot_id}.zip',
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        archive_bytes=len(payload),
        etag=f'etag-{snapshot_id}',
        last_modified_at=datetime(2022, 6, 10, tzinfo=_UTC),
        license_id='AACT-TERMS',
        license_url='https://aact.ctti-clinicaltrials.org/terms',
        citation='AACT database',
    )


def _receipt(root: Path, archive: AactArchiveReceipt) -> AactSliceReceipt:
    tables = []
    for path in sorted(root.glob('*.txt')):
        table = AactSourceTable(path.stem)
        tables.append(inspect_table(path, table, _NCT))
    tables.sort(key=lambda item: item.table.value)
    return AactSliceReceipt(
        slice_id=f'{archive.snapshot_id}-nct-slice',
        nct_id=_NCT,
        archive=archive,
        created_at=datetime(2026, 7, 13, tzinfo=_UTC),
        tables=tuple(tables),
    )


def _arm(
    role: ArmRole,
    decision_title: str,
    result_title: str,
    public_regimen: str,
    candidate_id: str | None = None,
) -> ArmMappingSpec:
    from vaxreplay.clinicaltrials.schema import normalize_regimen_title

    return ArmMappingSpec(
        role=role,
        candidate_id=candidate_id,
        regimen_key=normalize_regimen_title(decision_title),
        decision_title=decision_title,
        result_title=result_title,
        public_regimen=public_regimen,
    )


def _spec() -> EarlyClinicalEpisodeSpec:
    mappings = (
        _arm(
            ArmRole.CANDIDATE,
            'Group High Dose_PLAIN_B',
            'High Dose_PLAIN_B Group',
            'older-adult cohort, high antigen dose, no adjuvant, two-dose schedule',
            'cand-high',
        ),
        _arm(
            ArmRole.CANDIDATE,
            'Group Low Dose_PLAIN_B',
            'Low Dose_PLAIN_B Group',
            'older-adult cohort, low antigen dose, no adjuvant, two-dose schedule',
            'cand-low',
        ),
        _arm(
            ArmRole.CANDIDATE,
            'Group Peak Dose_PLAIN_B',
            'Peak Dose_PLAIN_B Group',
            'older-adult cohort, peak antigen dose, no adjuvant, two-dose schedule',
            'cand-peak',
        ),
        _arm(
            ArmRole.CONTROL,
            'Group Placebo_B',
            'Placebo_B Group',
            'older-adult cohort, saline control, two-dose schedule',
        ),
    )
    endpoints = tuple(
        OutcomeEndpointSpec(
            endpoint_id=f'endpoint-{index}',
            public_name=public_name,
            decision_outcome_id=f'd{index}',
            decision_measure=f'RSV protocol response {index}',
            result_outcome_id=f'o{index}',
            result_title=f'RSV held-out result {index}',
            classification='Day 91',
            param_type='Geometric Mean',
        )
        for index, public_name in (
            (1, 'functional antibody response'),
            (2, 'antigen-specific binding antibody response'),
            (3, 'antigen-specific helper T-cell response'),
        )
    )
    return EarlyClinicalEpisodeSpec(
        episode_id='aact-real-pilot-001',
        lineage_group_id='aact-real-pilot-lineage',
        nct_id=_NCT,
        decision_snapshot_id='aact-2020-02-01',
        label_snapshot_id='aact-2022-04-01',
        decision_at=_DECISION_AT,
        outcome_as_of=_OUTCOME_AT,
        portfolio_size=1,
        panel_selector=PanelSelector(
            normalized_regimen_key_suffix='b',
            allowed_group_types=('Experimental', 'Placebo Comparator'),
        ),
        arm_mappings=mappings,
        rubric=OutcomeRubric(
            target_id='day-91-multiaxis-response',
            endpoints=endpoints,
        ),
        forbidden_public_tokens=('as01', 'fake sponsor', 'nct00000001', 'product-x', 'rsv'),
        adjudication_version='pilot-rubric-1',
    )


def _make_slices(
    root: Path,
    *,
    decision_has_results: bool = False,
) -> tuple[Path, AactSliceReceipt, Path, AactSliceReceipt]:
    decision = root / 'decision'
    label = root / 'label'
    decision.mkdir()
    label.mkdir()
    study_header = [
        'nct_id',
        'results_first_submitted_date',
        'results_first_posted_date',
        'last_update_posted_date',
        'brief_title',
        'official_title',
        'overall_status',
        'phase',
        'source',
        'number_of_arms',
    ]
    _write_table(
        decision / 'studies.txt',
        study_header,
        [
            {
                'nct_id': _NCT,
                'results_first_submitted_date': '2020-01-20' if decision_has_results else '',
                'results_first_posted_date': '2020-01-21' if decision_has_results else '',
                'last_update_posted_date': '2020-01-18',
                'brief_title': 'Fake Sponsor Product-X RSV study',
                'official_title': 'Fake Sponsor Product-X RSV official study',
                'overall_status': 'Active, not recruiting',
                'phase': 'Phase 1',
                'source': 'Fake Sponsor',
                'number_of_arms': '5',
            }
        ],
    )
    _write_table(
        decision / 'designs.txt',
        ['id', 'nct_id', 'allocation', 'intervention_model', 'primary_purpose', 'masking'],
        [
            {
                'id': 'design-1',
                'nct_id': _NCT,
                'allocation': 'Randomized',
                'intervention_model': 'Parallel Assignment',
                'primary_purpose': 'Prevention',
                'masking': 'Triple',
            }
        ],
    )
    group_rows = [
        ('dg-a', 'Experimental', 'Group Low Dose_PLAIN_A', 'Part A Product-X RSV low dose'),
        ('dg-high', 'Experimental', 'Group High Dose_PLAIN_B', 'Part B Product-X RSV high dose'),
        ('dg-low', 'Experimental', 'Group Low Dose_PLAIN_B', 'Part B Product-X RSV low dose'),
        ('dg-peak', 'Experimental', 'Group Peak Dose_PLAIN_B', 'Part B Product-X RSV peak dose'),
        ('dg-control', 'Placebo Comparator', 'Group Placebo_B', 'Part B saline control'),
    ]
    _write_table(
        decision / 'design_groups.txt',
        ['id', 'nct_id', 'group_type', 'title', 'description'],
        [
            {'id': row_id, 'nct_id': _NCT, 'group_type': group_type, 'title': title, 'description': description}
            for row_id, group_type, title, description in group_rows
        ],
    )
    _write_table(
        decision / 'interventions.txt',
        ['id', 'nct_id', 'intervention_type', 'name', 'description'],
        [
            {
                'id': f'int-{index}',
                'nct_id': _NCT,
                'intervention_type': 'Biological',
                'name': f'Fake Sponsor Product-X RSV intervention {index}',
                'description': 'Two doses',
            }
            for index in range(1, 6)
        ],
    )
    _write_table(
        decision / 'design_group_interventions.txt',
        ['id', 'nct_id', 'design_group_id', 'intervention_id'],
        [
            {
                'id': f'link-{index}',
                'nct_id': _NCT,
                'design_group_id': group[0],
                'intervention_id': f'int-{index}',
            }
            for index, group in enumerate(group_rows, start=1)
        ],
    )
    _write_table(
        decision / 'design_outcomes.txt',
        ['id', 'nct_id', 'outcome_type', 'measure', 'time_frame', 'population', 'description'],
        [
            {
                'id': f'd{index}',
                'nct_id': _NCT,
                'outcome_type': 'secondary',
                'measure': f'RSV protocol response {index}',
                'time_frame': 'At Day 1 and Day 91',
                'population': '',
                'description': f'Private predeclared Product-X assay {index}',
            }
            for index in range(1, 4)
        ],
    )

    _write_table(
        label / 'studies.txt',
        study_header,
        [
            {
                'nct_id': _NCT,
                'results_first_submitted_date': '2022-02-22',
                'results_first_posted_date': '2022-03-23',
                'last_update_posted_date': '2022-03-23',
                'brief_title': 'Fake Sponsor Product-X RSV study',
                'official_title': 'Fake Sponsor Product-X RSV official study',
                'overall_status': 'Completed',
                'phase': 'Phase 1',
                'source': 'Fake Sponsor',
                'number_of_arms': '5',
            }
        ],
    )
    result_rows = [
        ('rg-a', 'OG000', 'Low Dose_PLAIN_A Group', 'Part A Product-X RSV low dose'),
        ('rg-high', 'OG001', 'High Dose_PLAIN_B Group', 'Part B Product-X RSV high dose'),
        ('rg-low', 'OG002', 'Low Dose_PLAIN_B Group', 'Part B Product-X RSV low dose'),
        ('rg-peak', 'OG003', 'Peak Dose_PLAIN_B Group', 'Part B Product-X RSV peak dose'),
        ('rg-control', 'OG004', 'Placebo_B Group', 'Part B saline control'),
        ('rg-other', 'OG005', 'Low Dose_PLAIN_B Group', 'Separate non-target result-group family'),
    ]
    _write_table(
        label / 'result_groups.txt',
        ['id', 'nct_id', 'ctgov_group_code', 'result_type', 'title', 'description'],
        [
            {
                'id': row_id,
                'nct_id': _NCT,
                'ctgov_group_code': code,
                'result_type': 'Outcome',
                'title': title,
                'description': description,
            }
            for row_id, code, title, description in result_rows
        ],
    )
    _write_table(
        label / 'outcomes.txt',
        [
            'id',
            'nct_id',
            'outcome_type',
            'title',
            'description',
            'time_frame',
            'population',
            'units',
            'dispersion_type',
            'param_type',
        ],
        [
            {
                'id': f'o{index}',
                'nct_id': _NCT,
                'outcome_type': 'Secondary',
                'title': f'RSV held-out result {index}',
                'description': f'Private Product-X result {index}',
                'time_frame': 'Day 91',
                'population': 'Per protocol',
                'units': 'units',
                'dispersion_type': '95% Confidence Interval',
                'param_type': 'Geometric Mean',
            }
            for index in range(1, 4)
        ],
    )
    measurement_header = [
        'id',
        'nct_id',
        'outcome_id',
        'result_group_id',
        'ctgov_group_code',
        'classification',
        'category',
        'title',
        'description',
        'units',
        'param_type',
        'param_value',
        'param_value_num',
        'dispersion_type',
        'dispersion_value',
        'dispersion_value_num',
        'dispersion_lower_limit',
        'dispersion_upper_limit',
        'explanation_of_na',
    ]
    values = {'rg-a': 20.0, 'rg-high': 50.0, 'rg-low': 15.0, 'rg-peak': 100.0, 'rg-control': 10.0}
    measurements = []
    for endpoint_index in range(1, 4):
        for group_index, (group_id, value) in enumerate(reversed(tuple(values.items())), start=1):
            measurements.append(
                {
                    'id': f'm-{endpoint_index}-{group_index}',
                    'nct_id': _NCT,
                    'outcome_id': f'o{endpoint_index}',
                    'result_group_id': group_id,
                    'ctgov_group_code': f'OG{group_index:03d}',
                    'classification': 'Day 91',
                    'category': '',
                    'title': f'RSV held-out result {endpoint_index}',
                    'description': f'Private Product-X result {endpoint_index}',
                    'units': 'units',
                    'param_type': 'Geometric Mean',
                    'param_value': str(value),
                    'param_value_num': str(value),
                    'dispersion_type': '95% Confidence Interval',
                    'dispersion_value': '',
                    'dispersion_value_num': '',
                    'dispersion_lower_limit': '',
                    'dispersion_upper_limit': '',
                    'explanation_of_na': '',
                }
            )
    _write_table(label / 'outcome_measurements.txt', measurement_header, list(reversed(measurements)))

    decision_receipt = _receipt(decision, _archive('aact-2020-02-01', _DECISION_AT, b'decision archive'))
    label_receipt = _receipt(label, _archive('aact-2022-04-01', _OUTCOME_AT, b'label archive'))
    return decision, decision_receipt, label, label_receipt


class ClinicalTrialsAdapterTest(unittest.TestCase):
    def test_builds_identity_masked_complete_panel_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            decision, decision_receipt, label, label_receipt = _make_slices(root)
            output = root / 'episode'

            bundle = build_episode(
                spec=_spec(),
                decision_snapshot_root=decision,
                decision_receipt=decision_receipt,
                label_snapshot_root=label,
                label_receipt=label_receipt,
                output_root=output,
                label_commitment_key=b'k' * 32,
            )

            self.assertEqual(bundle.manifest.task_type, 'early_clinical_arm_prioritization')
            self.assertFalse(bundle.manifest.synthetic)
            self.assertEqual(bundle.manifest.candidate_ids, ['cand-high', 'cand-low', 'cand-peak'])
            public_text = json.dumps(bundle.public_view()).casefold()
            for forbidden in ('nct00000001', 'fake sponsor', 'product-x', 'rsv', 'as01'):
                self.assertNotIn(forbidden, public_text)
            self.assertNotIn('held-out', public_text)
            self.assertIn('equal-weight geometric mean', public_text)
            self.assertIn('benchmark-defined post-hoc target', public_text)
            protocol_records = [record for record in bundle.evidence if record.title.startswith('Pre-cutoff')]
            self.assertTrue(protocol_records)
            self.assertTrue(
                all(record.license_id == decision_receipt.archive.license_id for record in protocol_records)
            )
            self.assertTrue(
                all(record.provenance_url == decision_receipt.archive.source_url for record in protocol_records)
            )

            assert bundle.ranking_labels is not None
            self.assertEqual(
                {item.candidate_id: item.relevance_grade for item in bundle.ranking_labels},
                {'cand-high': 3, 'cand-low': 1, 'cand-peak': 4},
            )
            assert bundle.private_labels is not None
            self.assertEqual(
                {item.candidate_id: item.outcome for item in bundle.private_labels.outcomes},
                {'cand-high': 0, 'cand-low': 0, 'cand-peak': 1},
            )
            utilities = {item.candidate_id: item.candidate_utility for item in bundle.private_labels.outcomes}
            self.assertAlmostEqual(utilities['cand-high'], 0.625)
            self.assertAlmostEqual(utilities['cand-low'], 0.1875)
            self.assertAlmostEqual(utilities['cand-peak'], 1.0)
            audit = AactPrivateAudit.model_validate_json((output / 'private' / 'aact_audit.json').read_bytes())
            self.assertEqual(audit.decision_design_group_count, 5)
            self.assertEqual(audit.selected_decision_group_count, 4)
            self.assertEqual(audit.excluded_decision_group_count, 1)
            self.assertEqual(audit.target_outcome_group_count, 5)
            self.assertEqual(audit.selected_result_group_count, 4)
            self.assertEqual(audit.excluded_target_result_group_count, 1)
            self.assertEqual(audit.non_target_outcome_result_group_count, 1)
            self.assertEqual(
                {item.regimen_key: item.result_group_id for item in audit.arm_mappings},
                {
                    'high-dose-plain-b': 'rg-high',
                    'low-dose-plain-b': 'rg-low',
                    'peak-dose-plain-b': 'rg-peak',
                    'placebo-b': 'rg-control',
                },
            )
            self.assertTrue(all(len(item.endpoint_values) == 3 for item in audit.outcomes))
            self.assertEqual(audit_episode(output)['excluded_protocol_arm_count'], 1)

    def test_audit_recomputes_folds_and_composite_even_after_hmac_recommit(self) -> None:
        cases = (
            (
                'fold',
                lambda value: value['outcomes'][0]['endpoint_values'][0].__setitem__(
                    'fold_over_control',
                    value['outcomes'][0]['endpoint_values'][0]['fold_over_control'] * 1.01,
                ),
                'fold_over_control must equal candidate_value / control_value',
            ),
            (
                'composite',
                lambda value: value['outcomes'][0].__setitem__(
                    'composite_fold_over_control',
                    value['outcomes'][0]['composite_fold_over_control'] * 1.01,
                ),
                'composite_fold_over_control must equal the geometric mean',
            ),
        )
        for case_name, tamper, expected_error in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                decision, decision_receipt, label, label_receipt = _make_slices(root)
                output = root / 'episode'
                build_episode(
                    spec=_spec(),
                    decision_snapshot_root=decision,
                    decision_receipt=decision_receipt,
                    label_snapshot_root=label,
                    label_receipt=label_receipt,
                    output_root=output,
                    label_commitment_key=b'k' * 32,
                )
                audit_path = output / 'private' / 'aact_audit.json'
                raw_audit = json.loads(audit_path.read_text(encoding='utf-8'))
                tamper(raw_audit)
                audit_path.write_bytes(canonical_json_bytes(raw_audit) + b'\n')

                # Recommit the tampered audit with the real organizer key.  The arithmetic check,
                # rather than the HMAC mismatch, must still reject it.
                new_commitment = hmac.new(b'k' * 32, canonical_json_bytes(raw_audit), hashlib.sha256).hexdigest()
                manifest_path = output / 'manifest.json'
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                manifest['source_provenance']['private_audit_commitment'] = new_commitment
                manifest_path.write_bytes(canonical_json_bytes(manifest) + b'\n')

                with self.assertRaisesRegex(AactAdapterError, expected_error):
                    audit_episode(output)

    def test_explicit_empty_label_commitment_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            decision, decision_receipt, label, label_receipt = _make_slices(root)

            with self.assertRaisesRegex(AactAdapterError, 'at least 32 bytes'):
                build_episode(
                    spec=_spec(),
                    decision_snapshot_root=decision,
                    decision_receipt=decision_receipt,
                    label_snapshot_root=label,
                    label_receipt=label_receipt,
                    output_root=root / 'episode',
                    label_commitment_key=b'',
                )

    def test_rejects_organizer_omission_from_value_blind_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            decision, decision_receipt, label, label_receipt = _make_slices(root)
            spec = _spec()
            incomplete = spec.model_copy(
                update={
                    'arm_mappings': tuple(
                        mapping for mapping in spec.arm_mappings if mapping.candidate_id != 'cand-peak'
                    )
                }
            )

            with self.assertRaisesRegex(AactAdapterError, 'all and only protocol arms'):
                build_episode(
                    spec=incomplete,
                    decision_snapshot_root=decision,
                    decision_receipt=decision_receipt,
                    label_snapshot_root=label,
                    label_receipt=label_receipt,
                    output_root=root / 'episode',
                    label_commitment_key=b'k' * 32,
                )

    def test_rejects_results_present_at_decision_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            decision, decision_receipt, label, label_receipt = _make_slices(root, decision_has_results=True)

            with self.assertRaisesRegex(AactAdapterError, 'not pre-results'):
                build_episode(
                    spec=_spec(),
                    decision_snapshot_root=decision,
                    decision_receipt=decision_receipt,
                    label_snapshot_root=label,
                    label_receipt=label_receipt,
                    output_root=root / 'episode',
                    label_commitment_key=b'k' * 32,
                )

    def test_rejects_target_endpoint_with_incomplete_group_universe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            decision, decision_receipt, label, _ = _make_slices(root)
            measurement_path = label / 'outcome_measurements.txt'
            lines = measurement_path.read_text(encoding='utf-8').splitlines()
            measurement_path.write_text(
                '\n'.join(line for line in lines if '|o3|rg-a|' not in line) + '\n',
                encoding='utf-8',
            )
            label_receipt = _receipt(label, _archive('aact-2022-04-01', _OUTCOME_AT, b'label archive'))

            with self.assertRaisesRegex(AactAdapterError, 'same complete result-group set'):
                build_episode(
                    spec=_spec(),
                    decision_snapshot_root=decision,
                    decision_receipt=decision_receipt,
                    label_snapshot_root=label,
                    label_receipt=label_receipt,
                    output_root=root / 'episode',
                    label_commitment_key=b'k' * 32,
                )


if __name__ == '__main__':
    unittest.main()
