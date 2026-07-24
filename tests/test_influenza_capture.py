from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from vaxreplay.influenza.capture import (
    InfluenzaCaptureError,
    build_offline_capture,
    load_offline_capture,
    verify_raw_files,
)

_CUTOFF = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
_RETRIEVED = datetime(2026, 7, 31, 23, 59, tzinfo=timezone.utc)


def _write_csv(path: Path, rows: list[tuple[str, str]], *, header: str = 'record_id,release_at') -> bytes:
    payload = header + '\n' + ''.join(f'{record_id},{release_at}\n' for record_id, release_at in rows)
    encoded = payload.encode('utf-8')
    path.write_bytes(encoded)
    return encoded


def _build(raw_files: list[Path], output: Path):
    return build_offline_capture(
        raw_files=raw_files,
        output_root=output,
        capture_id='who-sh-2026-predecision',
        source_id='user-supplied-public-influenza-metadata',
        cutoff_at=_CUTOFF,
        retrieved_at=_RETRIEVED,
    )


class InfluenzaCaptureTest(unittest.TestCase):
    def test_builds_complete_deterministic_content_addressed_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_a = root / 'a.csv'
            raw_z = root / 'z.csv'
            payload_a = _write_csv(
                raw_a,
                [
                    ('opaque-zeta', '2026-07-01T12:00:00-04:00'),
                    ('opaque-late', '2026-08-02T00:00:00Z'),
                ],
            )
            payload_z = _write_csv(
                raw_z,
                [
                    ('opaque-alpha', '2026-06-01T00:00:00Z'),
                    ('opaque-mu', '2026-07-02T00:00:00+00:00'),
                ],
            )

            first = _build([raw_z, raw_a], root / 'capture-first')
            second = _build([raw_a, raw_z], root / 'capture-second')

            self.assertEqual(
                [record.record_id for record in first.accepted_records],
                ['opaque-alpha', 'opaque-mu', 'opaque-zeta'],
            )
            self.assertEqual([record.record_id for record in first.rejected_records], ['opaque-late'])
            self.assertEqual(first.rejected_records[0].reason, 'release-after-cutoff')
            self.assertEqual(first.accepted_records[-1].release_at.isoformat(), '2026-07-01T16:00:00+00:00')
            self.assertEqual(first.manifest.raw_record_count, 4)
            self.assertEqual(first.manifest.accepted_record_count, 3)
            self.assertEqual(first.manifest.rejected_record_count, 1)
            self.assertEqual([receipt.file_name for receipt in first.manifest.raw_files], ['a.csv', 'z.csv'])
            self.assertEqual(first.manifest.raw_files[0].sha256, hashlib.sha256(payload_a).hexdigest())
            self.assertEqual(first.manifest.raw_files[0].byte_count, len(payload_a))
            self.assertEqual(first.manifest.raw_files[0].row_count, 2)
            self.assertEqual(first.manifest.raw_files[1].sha256, hashlib.sha256(payload_z).hexdigest())
            self.assertEqual(first.manifest.inclusion_rule.admission_predicate, 'release_at <= cutoff_at')
            self.assertEqual(first.manifest.inclusion_rule.identifier_semantics, 'opaque-verbatim-utf8')
            self.assertTrue(first.manifest.external_timestamp_required)
            self.assertEqual(
                first.seal_target.capture_manifest_file_sha256,
                first.manifest_file_sha256,
            )
            self.assertEqual(first.manifest, second.manifest)
            self.assertEqual(first.manifest_file_sha256, second.manifest_file_sha256)
            self.assertEqual(first.seal_target_file_sha256, second.seal_target_file_sha256)
            source_artifact = first.as_source_capture_artifact(witnessed_at=_CUTOFF)
            self.assertEqual(
                source_artifact.source_id,
                'user-supplied-public-influenza-metadata:who-sh-2026-predecision',
            )
            self.assertEqual(source_artifact.captured_at, first.manifest.retrieved_at)
            self.assertEqual(source_artifact.witnessed_at, _CUTOFF)
            self.assertEqual(source_artifact.manifest_bytes, first.manifest_bytes)

            verify_raw_files(first, [raw_a, raw_z])
            self.assertEqual(load_offline_capture(first.root), first)

    def test_rejects_duplicate_opaque_ids_across_all_raw_files_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / 'first.csv'
            second = root / 'second.csv'
            _write_csv(first, [('same-opaque-id', '2026-06-01T00:00:00Z')])
            _write_csv(second, [('same-opaque-id', '2026-06-02T00:00:00Z')])
            output = root / 'capture'

            with self.assertRaisesRegex(InfluenzaCaptureError, 'globally unique'):
                _build([first, second], output)

            self.assertFalse(output.exists())

    def test_fail_closed_timestamp_parsing_rejects_missing_naive_and_unknown_offsets(self) -> None:
        invalid_values = ('', '2026-07-01', '2026-07-01T00:00:00', '2026-07-01T00:00:00-00:00')
        for index, invalid in enumerate(invalid_values):
            with self.subTest(release_at=invalid), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                raw = root / 'raw.csv'
                _write_csv(raw, [(f'opaque-{index}', invalid)])
                output = root / 'capture'

                with self.assertRaisesRegex(InfluenzaCaptureError, 'release_at'):
                    _build([raw], output)

                self.assertFalse(output.exists())

    def test_rejects_post_cutoff_retrieval_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / 'raw.csv'
            _write_csv(raw, [('opaque-1', '2026-06-01T00:00:00Z')])

            with self.assertRaisesRegex(InfluenzaCaptureError, 'post-cutoff input'):
                build_offline_capture(
                    raw_files=[raw],
                    output_root=root / 'capture',
                    capture_id='capture-after-cutoff',
                    source_id='opaque-source',
                    cutoff_at=_CUTOFF,
                    retrieved_at=datetime(2026, 8, 1, 0, 0, 1, tzinfo=timezone.utc),
                )

    def test_rejects_extra_columns_including_sequence_or_collection_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / 'raw.csv'
            _write_csv(
                raw,
                [('opaque-1', '2026-06-01T00:00:00Z')],
                header='record_id,release_at,collection_at,sequence',
            )

            with self.assertRaisesRegex(InfluenzaCaptureError, 'exact header'):
                _build([raw], root / 'capture')

    def test_rejects_claimed_release_after_retrieval_even_when_before_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / 'raw.csv'
            _write_csv(raw, [('opaque-1', '2026-07-31T23:59:30Z')])

            with self.assertRaisesRegex(InfluenzaCaptureError, 'after the claimed retrieval time'):
                _build([raw], root / 'capture')

    def test_detects_normalized_artifact_and_raw_input_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / 'raw.csv'
            _write_csv(raw, [('opaque-1', '2026-06-01T00:00:00Z')])
            capture = _build([raw], root / 'capture')

            raw.write_text('record_id,release_at\nopaque-2,2026-06-01T00:00:00Z\n', encoding='utf-8')
            with self.assertRaisesRegex(InfluenzaCaptureError, 'receipts do not match'):
                verify_raw_files(capture, [raw])

            accepted_path = capture.root / 'accepted-records.jsonl'
            accepted_path.write_bytes(accepted_path.read_bytes() + b' ')
            with self.assertRaisesRegex(InfluenzaCaptureError, 'accepted record hash mismatch'):
                load_offline_capture(capture.root)


if __name__ == '__main__':
    unittest.main()
