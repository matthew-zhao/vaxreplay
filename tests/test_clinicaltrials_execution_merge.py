from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from tests.test_clinicaltrials_execution_adapter import (
    _tree_bytes,
    _write_decision_zip,
    _write_label_zip,
)
from vaxreplay.clinicaltrials.execution_adapter import build_aact_execution_cohort
from vaxreplay.clinicaltrials.execution_merge import (
    AactExecutionMergeError,
    StableFieldRemapSet,
    main,
    merge_aact_execution_builds,
)
from vaxreplay.clinicaltrials.execution_schema import RegistryOutcomeClass


def _source_builds(root: Path) -> tuple[Path, Path]:
    decision_2018 = root / 'decision-2018.zip'
    label_2022 = root / 'label-2022.zip'
    decision_2020 = root / 'decision-2020.zip'
    label_2024 = root / 'label-2024.zip'
    _write_decision_zip(
        decision_2018,
        decision_anchor=date(2018, 4, 1),
        include_second_study=False,
        first_lead_sponsor='Moderna',
    )
    _write_label_zip(label_2022)
    _write_decision_zip(decision_2020, first_lead_sponsor='Moderna, Inc.')
    _write_label_zip(label_2024)
    first = root / 'source-2018'
    second = root / 'source-2020'
    build_aact_execution_cohort(
        decision_archive=decision_2018,
        decision_archive_date=date(2018, 4, 1),
        label_archive=label_2022,
        label_archive_date=date(2022, 4, 1),
        output_root=first,
        synthetic_test_only=True,
    )
    build_aact_execution_cohort(
        decision_archive=decision_2020,
        decision_archive_date=date(2020, 2, 1),
        label_archive=label_2024,
        label_archive_date=date(2024, 2, 1),
        output_root=second,
        synthetic_test_only=True,
    )
    return first, second


class AactExecutionMergeTests(unittest.TestCase):
    def test_merges_at_earliest_eligible_anchor_and_uses_that_anchors_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = _source_builds(root)
            build = merge_aact_execution_builds(
                source_roots=(second, first),
                output_root=root / 'merged',
            )

            assignments = {assignment.nct_id: assignment for assignment in build.inventory.assignments}
            self.assertEqual(assignments['NCT00000001'].anchor_date, date(2018, 4, 1))
            self.assertEqual(assignments['NCT00000001'].label_archive_date, date(2022, 4, 1))
            self.assertEqual(assignments['NCT00000002'].anchor_date, date(2020, 2, 1))
            self.assertEqual(assignments['NCT00000002'].label_archive_date, date(2024, 2, 1))
            self.assertEqual(len(assignments), 2)

            labels = {label.nct_id: label for label in build.labels.labels}
            self.assertEqual(labels['NCT00000001'].registry_outcome_class, RegistryOutcomeClass.TERMINATED)
            self.assertEqual(labels['NCT00000002'].registry_outcome_class, RegistryOutcomeClass.RECORD_MISSING)
            self.assertEqual(
                tuple(item.assignment_count for item in build.receipt.anchor_assignment_counts),
                (1, 1),
            )
            self.assertTrue(build.receipt.synthetic)
            self.assertEqual(build.receipt.release_status, 'synthetic_test_only')
            self.assertFalse(build.receipt.scored_cohort_eligible)
            self.assertFalse(build.receipt.active_vaccination_adjudication_bound)
            self.assertTrue(build.receipt.manual_lineage_review_required)
            self.assertFalse(build.receipt.lineage_split_safe)
            self.assertGreaterEqual(build.receipt.stability_remap_count, 1)
            remaps = StableFieldRemapSet.model_validate_json(
                (build.root / 'organizer' / 'stability-remaps.json').read_bytes()
            )
            nct_one_remap = next(item for item in remaps.remaps if item.nct_id == 'NCT00000001')
            self.assertIn('lineage_group_id', nct_one_remap.changed_fields)

    def test_merge_is_byte_deterministic_and_cli_orders_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = _source_builds(root)
            one = merge_aact_execution_builds(source_roots=(first, second), output_root=root / 'one')
            self.assertEqual(
                main(
                    [
                        '--source-root',
                        str(second),
                        '--source-root',
                        str(first),
                        '--output-root',
                        str(root / 'two'),
                    ]
                ),
                0,
            )
            self.assertEqual(_tree_bytes(one.root), _tree_bytes(root / 'two'))

    def test_tampered_source_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = _source_builds(root)
            inventory_path = first / 'organizer' / 'cohort-inventory.json'
            inventory_path.write_bytes(inventory_path.read_bytes() + b' ')
            with self.assertRaisesRegex(AactExecutionMergeError, 'does not match its build receipt'):
                merge_aact_execution_builds(source_roots=(first, second), output_root=root / 'merged')


if __name__ == '__main__':
    unittest.main()
