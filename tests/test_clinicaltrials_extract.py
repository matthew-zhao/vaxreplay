from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
import zipfile
from datetime import date, datetime, time, timezone
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.clinicaltrials.extract import (
    AactSliceError,
    InterventionRow,
    extract_nct_slice_from_archive,
    inspect_table,
    load_slice,
    verify_archive_file,
)
from vaxreplay.clinicaltrials.schema import (
    AactArchiveReceipt,
    AactSliceReceipt,
    AactSourceTable,
    normalize_regimen_title,
)

_UTC = timezone.utc


def _archive(payload: bytes = b'archive') -> AactArchiveReceipt:
    return AactArchiveReceipt(
        snapshot_id='aact-2020-02-01',
        archive_date=date(2020, 2, 1),
        source_cutoff_at=datetime.combine(date(2020, 2, 1), time.max, tzinfo=_UTC),
        retrieved_at=datetime(2026, 7, 13, tzinfo=_UTC),
        source_url='https://example.test/aact-2020-02-01.zip',
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        archive_bytes=len(payload),
        etag='opaque-etag',
        last_modified_at=datetime(2022, 6, 10, tzinfo=_UTC),
        license_id='AACT-TERMS',
        license_url='https://aact.ctti-clinicaltrials.org/terms',
        citation='AACT database',
    )


def _write_pipe(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as destination:
        writer = csv.writer(destination, delimiter='|', quotechar='"', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


class ClinicalTrialsExtractTest(unittest.TestCase):
    def test_archive_cutoff_is_explicit_end_of_archive_date_not_invented_midnight(self) -> None:
        values = _archive().model_dump()
        values['source_cutoff_at'] = datetime(2020, 2, 1, tzinfo=_UTC)

        with self.assertRaisesRegex(ValidationError, 'end-of-day UTC'):
            AactArchiveReceipt.model_validate(values)

    def test_normalizes_group_position_without_fuzzy_matching(self) -> None:
        self.assertEqual(normalize_regimen_title('Group Low Dose_AS01E_B'), 'low-dose-as01e-b')
        self.assertEqual(normalize_regimen_title('Low Dose_AS01E_B Group'), 'low-dose-as01e-b')
        self.assertNotEqual(
            normalize_regimen_title('Low Dose_AS01E_A Group'),
            normalize_regimen_title('Low Dose_AS01E_B Group'),
        )

    def test_parses_documented_quoted_embedded_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / 'interventions.txt'
            _write_pipe(
                path,
                ['id', 'nct_id', 'intervention_type', 'name', 'description'],
                [['1', 'NCT00000001', 'Biological', 'Product-X', 'contains | an embedded delimiter']],
            )
            table_receipt = inspect_table(path, AactSourceTable.INTERVENTIONS, 'NCT00000001')
            receipt = AactSliceReceipt(
                slice_id='slice-one',
                nct_id='NCT00000001',
                archive=_archive(),
                created_at=datetime(2026, 7, 13, tzinfo=_UTC),
                tables=(table_receipt,),
            )

            loaded = load_slice(root, receipt)

            rows = loaded.rows(AactSourceTable.INTERVENTIONS, InterventionRow)
            self.assertEqual(rows[0].description, 'contains | an embedded delimiter')

    def test_requires_exact_nct_in_every_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'interventions.txt'
            _write_pipe(
                path,
                ['id', 'nct_id', 'intervention_type', 'name', 'description'],
                [
                    ['1', 'NCT00000001', 'Biological', 'one', 'description'],
                    ['2', 'NCT99999999', 'Biological', 'two', 'description'],
                ],
            )

            with self.assertRaisesRegex(AactSliceError, 'expected exact NCT NCT00000001'):
                inspect_table(path, AactSourceTable.INTERVENTIONS, 'NCT00000001')

    def test_rejects_tampering_and_extra_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / 'interventions.txt'
            _write_pipe(
                path,
                ['id', 'nct_id', 'intervention_type', 'name', 'description'],
                [['1', 'NCT00000001', 'Biological', 'one', 'description']],
            )
            table_receipt = inspect_table(path, AactSourceTable.INTERVENTIONS, 'NCT00000001')
            receipt = AactSliceReceipt(
                slice_id='slice-one',
                nct_id='NCT00000001',
                archive=_archive(),
                created_at=datetime(2026, 7, 13, tzinfo=_UTC),
                tables=(table_receipt,),
            )
            (root / 'organizer-notes.txt').write_text('private', encoding='utf-8')

            with self.assertRaisesRegex(AactSliceError, 'inventory mismatch'):
                load_slice(root, receipt)

            (root / 'organizer-notes.txt').unlink()
            path.write_text(path.read_text(encoding='utf-8').replace('description', 'tampered', 1), encoding='utf-8')
            with self.assertRaisesRegex(AactSliceError, 'SHA-256'):
                load_slice(root, receipt)

    def test_verifies_full_archive_byte_count_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'archive.zip'
            path.write_bytes(b'archive')

            verify_archive_file(path, _archive())

            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(AactSliceError, 'byte count mismatch'):
                verify_archive_file(path, _archive())

    def test_regenerates_exact_nct_slice_from_verified_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_table = root / 'source-interventions.txt'
            _write_pipe(
                source_table,
                ['id', 'nct_id', 'intervention_type', 'name', 'description'],
                [
                    ['1', 'NCT00000001', 'Biological', 'target', 'quoted | pipe\nand newline'],
                    ['2', 'NCT99999999', 'Biological', 'other', 'must not be extracted'],
                ],
            )
            archive_path = root / 'archive.zip'
            with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(source_table, 'interventions.txt')
            payload = archive_path.read_bytes()
            archive_receipt = _archive(payload)
            output = root / 'slice'

            receipt = extract_nct_slice_from_archive(
                archive_path=archive_path,
                archive_receipt=archive_receipt,
                nct_id='NCT00000001',
                tables=(AactSourceTable.INTERVENTIONS,),
                output_root=output,
                slice_id='regenerated-slice',
                created_at=datetime(2026, 7, 13, tzinfo=_UTC),
            )

            loaded = load_slice(output, receipt)
            rows = loaded.rows(AactSourceTable.INTERVENTIONS, InterventionRow)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].name, 'target')
            self.assertEqual(rows[0].description, 'quoted | pipe\nand newline')
            self.assertNotIn('NCT99999999', (output / 'interventions.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
