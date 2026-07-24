from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.iedb.cli import main as iedb_main
from vaxreplay.iedb.live_capture import (
    IedbApiCaptureBinding,
    IedbApiCaptureSpec,
    IedbApiPageSpec,
    IedbFullExportCaptureBinding,
    IedbFullExportCaptureSpec,
    IedbFullExportIdentity,
    IedbHttpHeader,
    IedbLiveCaptureError,
    IedbLiveCaptureManifest,
    build_api_capture,
    build_full_export_capture,
    verify_capture_manifest,
    write_capture_manifest,
)

_BUILD_AT = datetime(2026, 7, 6, tzinfo=timezone.utc)
_RETRIEVED_AT = datetime(2026, 7, 7, tzinfo=timezone.utc)
_WITNESSED_AT = datetime(2026, 7, 7, 1, tzinfo=timezone.utc)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _receipt(url: str, start: int, requested_end: int, response_end: int, total: int) -> dict[str, object]:
    content_range = '*/0' if total == 0 else f'{start}-{response_end}/{total}'
    return {
        'schema_version': 'vaxreplay.iedb-api-exchange.v0.1',
        'request_url': url,
        'request_headers': [
            {'name': 'range', 'value': f'{start}-{requested_end}'},
            {'name': 'range-unit', 'value': 'items'},
        ],
        'status_code': 206,
        'response_headers': [
            {'name': 'content-range', 'value': content_range},
        ],
    }


def _make_api_capture(root: Path) -> IedbApiCaptureSpec:
    metrics = [
        {
            'search_table_name': 'bcell_search',
            'record_count': 1,
            'creation_date': '2026-07-06T00:00:00Z',
        },
        {
            'search_table_name': 'tcell_search',
            'record_count': 4,
            'creation_date': '2026-07-06T00:00:00Z',
        },
    ]
    _write_json(root / 'metrics/before.json', metrics)
    _write_json(root / 'metrics/after.json', metrics)

    page_specs = []
    tables = {
        'tcell_search': ('tcell_id', ((0, (1, 2)), (2, (5, 9)))),
        'bcell_search': ('bcell_id', ((0, (11,)),)),
    }
    for table_name, (id_field, pages) in tables.items():
        url = f'https://query-api.iedb.org/{table_name}?select={id_field},structure_iri&order={id_field}'
        total = sum(len(ids) for _start, ids in pages)
        for ordinal, (start, ids) in enumerate(pages):
            data_path = f'data/{table_name}-{ordinal}.json'
            receipt_path = f'receipts/{table_name}-{ordinal}.json'
            _write_json(
                root / data_path,
                [{id_field: identifier, 'structure_iri': f'IEDB_EPITOPE:{identifier}'} for identifier in ids],
            )
            _write_json(
                root / receipt_path,
                _receipt(url, start, start + 1, start + len(ids) - 1, total),
            )
            page_specs.append(
                IedbApiPageSpec(
                    table_name=table_name,
                    id_field=id_field,
                    request_url=url,
                    data_relative_path=data_path,
                    receipt_relative_path=receipt_path,
                    data_format='json',
                )
            )
    return IedbApiCaptureSpec(
        capture_id='iedb-api-2026-07-06',
        retrieved_at=_RETRIEVED_AT,
        metrics_url='https://query-api.iedb.org/api_metrics?order=search_table_name',
        metrics_before_relative_path='metrics/before.json',
        metrics_after_relative_path='metrics/after.json',
        expected_table_names=('bcell_search', 'tcell_search'),
        pages=tuple(reversed(page_specs)),
    )


def _rewrite_metrics(root: Path, transform) -> None:
    for name in ('before.json', 'after.json'):
        path = root / 'metrics' / name
        value = json.loads(path.read_text(encoding='utf-8'))
        transform(value, name)
        _write_json(path, value)


def _full_export_spec(root: Path, *, expected_member_count: int = 2) -> IedbFullExportCaptureSpec:
    artifact = root / 'exports/iedb-2026-07-06.zip'
    return IedbFullExportCaptureSpec(
        capture_id='iedb-full-2026-07-06',
        retrieved_at=_RETRIEVED_AT,
        identity=IedbFullExportIdentity(
            release_id='iedb-full-2026-07-06-csv',
            source_build_at=_BUILD_AT,
            source_url='https://www.iedb.org/downloader.php?file_name=iedb_csv.zip',
            artifact_relative_path='exports/iedb-2026-07-06.zip',
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            artifact_byte_count=artifact.stat().st_size,
            artifact_format='zip',
            expected_member_count=expected_member_count,
        ),
    )


def _run_iedb_cli(*args: str) -> tuple[dict[str, object], bytes]:
    stdout = io.StringIO()
    with patch.object(sys, 'argv', ['vaxreplay-iedb', *args]), redirect_stdout(stdout):
        iedb_main()
    stdout_bytes = stdout.getvalue().encode('utf-8')
    return json.loads(stdout_bytes), stdout_bytes


class IedbApiLiveCaptureTest(unittest.TestCase):
    def test_builds_canonical_complete_capture_for_external_sealing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root / 'raw')

            capture = build_api_capture(root / 'raw', spec)

            self.assertEqual(capture.manifest.source_build_at, _BUILD_AT)
            self.assertEqual(capture.manifest.retrieved_at, _RETRIEVED_AT)
            self.assertFalse(capture.manifest.source_authenticity_verified)
            self.assertFalse(capture.manifest.tier_a_eligible)
            self.assertTrue(capture.manifest.external_timestamp_required)
            self.assertEqual(len(capture.manifest.inventory), 8)
            binding = capture.manifest.source_binding
            self.assertIsInstance(binding, IedbApiCaptureBinding)
            assert isinstance(binding, IedbApiCaptureBinding)
            self.assertEqual(binding.completeness_scope, 'declared-api-table-set')
            self.assertTrue(binding.complete_enumeration_within_scope)
            self.assertEqual(binding.expected_table_names, ('bcell_search', 'tcell_search'))
            self.assertEqual(
                tuple(table.table_name for table in binding.tables),
                ('bcell_search', 'tcell_search'),
            )
            self.assertEqual(binding.tables[1].record_count, 4)
            self.assertEqual(binding.tables[1].page_count, 2)
            self.assertEqual(capture.manifest_bytes, canonical_json_bytes(capture.manifest))
            self.assertEqual(hashlib.sha256(capture.manifest_bytes).hexdigest(), capture.manifest_sha256)
            self.assertEqual(IedbLiveCaptureManifest.model_validate_json(capture.manifest_bytes), capture.manifest)
            self.assertEqual(
                verify_capture_manifest(root / 'raw', spec, capture.manifest_bytes),
                capture,
            )

            source_capture = capture.as_source_capture_artifact(witnessed_at=_WITNESSED_AT)
            self.assertEqual(source_capture.source_id, 'iedb:iedb-api-2026-07-06')
            self.assertEqual(source_capture.source_release_at, _BUILD_AT)
            self.assertEqual(source_capture.captured_at, _RETRIEVED_AT)
            self.assertEqual(source_capture.witnessed_at, _WITNESSED_AT)
            self.assertEqual(source_capture.manifest_bytes, capture.manifest_bytes)

            output = write_capture_manifest(capture, root / 'commitments/iedb.json')
            self.assertEqual(output.read_bytes(), capture.manifest_bytes)
            with self.assertRaisesRegex(IedbLiveCaptureError, 'already exists'):
                write_capture_manifest(capture, output)

    def test_manifest_publish_is_exclusive_under_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / 'raw'
            capture = build_api_capture(raw_root, _make_api_capture(raw_root))
            output = root / 'commitments/iedb.json'

            def publish() -> str:
                try:
                    write_capture_manifest(capture, output)
                    return 'published'
                except IedbLiveCaptureError as error:
                    self.assertIn('already exists', str(error))
                    return 'exists'

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = tuple(executor.map(lambda _ordinal: publish(), range(8)))

            self.assertEqual(results.count('published'), 1)
            self.assertEqual(results.count('exists'), 7)
            self.assertEqual(output.read_bytes(), capture.manifest_bytes)

    def test_manifest_is_independent_of_root_and_file_mtimes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_spec = _make_api_capture(root / 'first')
            shutil.copytree(root / 'first', root / 'second')
            for path in (root / 'second').rglob('*'):
                os.utime(path, ns=(1_900_000_000_000_000_000, 1_900_000_000_000_000_000))

            first = build_api_capture(root / 'first', first_spec)
            second = build_api_capture(root / 'second', first_spec)

            self.assertEqual(first.manifest_bytes, second.manifest_bytes)
            self.assertEqual(first.manifest_sha256, second.manifest_sha256)

    def test_rejects_torn_metrics_and_mixed_table_builds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            after = root / 'metrics/after.json'
            value = json.loads(after.read_text(encoding='utf-8'))
            value[0]['record_count'] = 2
            _write_json(after, value)
            with self.assertRaisesRegex(IedbLiveCaptureError, 'changed'):
                build_api_capture(root, spec)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)

            def mixed(value, _name):
                value[0]['creation_date'] = '2026-07-05T00:00:00Z'

            _rewrite_metrics(root, mixed)
            with self.assertRaisesRegex(IedbLiveCaptureError, 'mixed source build timestamps'):
                build_api_capture(root, spec)

    def test_rejects_duplicate_ids_and_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            page = root / 'data/tcell_search-1.json'
            value = json.loads(page.read_text(encoding='utf-8'))
            value[0]['tcell_id'] = 2
            _write_json(page, value)
            with self.assertRaisesRegex(IedbLiveCaptureError, 'duplicate IDs'):
                build_api_capture(root, spec)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)

            def wrong_count(value, _name):
                value[1]['record_count'] = 5

            _rewrite_metrics(root, wrong_count)
            with self.assertRaisesRegex(IedbLiveCaptureError, 'does not match api_metrics count'):
                build_api_capture(root, spec)

    def test_declared_table_scope_rejects_missing_or_unexpected_page_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            tcell_only = tuple(page for page in spec.pages if page.table_name == 'tcell_search')

            with self.assertRaisesRegex(ValidationError, 'cover exactly expected_table_names'):
                IedbApiCaptureSpec.model_validate(
                    {**spec.model_dump(), 'pages': tuple(page.model_dump() for page in tcell_only)}
                )

            with self.assertRaisesRegex(ValidationError, 'cover exactly expected_table_names'):
                IedbApiCaptureSpec.model_validate(
                    {
                        **spec.model_dump(),
                        'expected_table_names': ('tcell_search',),
                    }
                )

            bypassed_validation = spec.model_copy(update={'pages': tcell_only})
            with self.assertRaisesRegex(IedbLiveCaptureError, 'declared table set'):
                build_api_capture(root, bypassed_validation)

    def test_rejects_declared_table_missing_from_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)

            def drop_bcell(value, _name):
                value[:] = [metric for metric in value if metric['search_table_name'] != 'bcell_search']

            _rewrite_metrics(root, drop_bcell)
            with self.assertRaisesRegex(IedbLiveCaptureError, 'missing declared tables'):
                build_api_capture(root, spec)

    def test_rejects_missing_extra_and_symlinked_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            (root / 'data/tcell_search-0.json').unlink()
            with self.assertRaisesRegex(IedbLiveCaptureError, 'missing declared files'):
                build_api_capture(root, spec)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            (root / 'undeclared.txt').write_text('not part of capture', encoding='utf-8')
            with self.assertRaisesRegex(IedbLiveCaptureError, 'undeclared files'):
                build_api_capture(root, spec)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            target = root / 'data/tcell_search-0.json'
            outside = Path(temporary_directory).parent / 'vaxreplay-iedb-live-symlink-target.json'
            outside.write_bytes(target.read_bytes())
            target.unlink()
            target.symlink_to(outside)
            try:
                with self.assertRaisesRegex(IedbLiveCaptureError, 'symbolic link'):
                    build_api_capture(root, spec)
            finally:
                outside.unlink(missing_ok=True)

    def test_rejects_any_capture_directory_traversal_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            inaccessible = root / 'hidden'

            def denied_walk(_root, *, topdown, onerror, followlinks):
                self.assertTrue(topdown)
                self.assertFalse(followlinks)
                self.assertIsNotNone(onerror)
                onerror(PermissionError(13, 'Permission denied', str(inaccessible)))
                yield  # pragma: no cover - the callback must raise

            with (
                patch('vaxreplay.iedb.live_capture.os.walk', side_effect=denied_walk),
                self.assertRaisesRegex(IedbLiveCaptureError, 'cannot traverse raw capture'),
            ):
                build_api_capture(root, spec)

    def test_hash_rejects_same_size_file_metadata_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'artifact.bin'
            path.write_bytes(b'same-size-content')

            from vaxreplay.iedb import live_capture

            real_fstat = os.fstat
            for changed_field in ('st_mtime_ns', 'st_ctime_ns'):
                call_count = 0

                def changed_fstat(descriptor):
                    nonlocal call_count
                    call_count += 1
                    observed = real_fstat(descriptor)
                    if call_count == 1:
                        return observed
                    values = {
                        'st_dev': observed.st_dev,
                        'st_ino': observed.st_ino,
                        'st_size': observed.st_size,
                        'st_mtime_ns': observed.st_mtime_ns,
                        'st_ctime_ns': observed.st_ctime_ns,
                    }
                    values[changed_field] += 1
                    return SimpleNamespace(**values)

                with (
                    self.subTest(changed_field=changed_field),
                    patch('vaxreplay.iedb.live_capture.os.fstat', side_effect=changed_fstat),
                    self.assertRaisesRegex(IedbLiveCaptureError, 'changed while hashing'),
                ):
                    live_capture._hash_regular_file(path)

    def test_rejects_ambiguous_or_filtered_query_metadata(self) -> None:
        base = {
            'table_name': 'tcell_search',
            'id_field': 'tcell_id',
            'data_relative_path': 'data.json',
            'receipt_relative_path': 'receipt.json',
            'data_format': 'json',
        }
        with self.assertRaises(ValidationError):
            IedbApiPageSpec(
                **base,
                request_url='https://query-api.iedb.org/tcell_search?order=tcell_id&order=tcell_id',
            )
        with self.assertRaises(ValidationError):
            IedbApiPageSpec(
                **base,
                request_url='https://query-api.iedb.org/tcell_search?order=tcell_id&reference_id=eq.1',
            )

    def test_rejects_duplicate_json_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            receipt = root / 'receipts/tcell_search-0.json'
            text = receipt.read_text(encoding='utf-8')
            receipt.write_text(text.replace('{', '{"status_code":200,', 1), encoding='utf-8')
            with self.assertRaisesRegex(IedbLiveCaptureError, 'duplicate JSON key'):
                build_api_capture(root, spec)

    def test_rejects_secret_bearing_headers_in_exchange_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            receipt_path = root / 'receipts/tcell_search-0.json'
            receipt = json.loads(receipt_path.read_bytes())
            receipt['request_headers'].insert(0, {'name': 'authorization', 'value': 'Bearer redacted-but-still-secret'})
            _write_json(receipt_path, receipt)

            with self.assertRaisesRegex(IedbLiveCaptureError, 'secret-bearing headers'):
                build_api_capture(root, spec)

    def test_rejects_unapproved_or_non_ascii_receipt_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            receipt_path = root / 'receipts/tcell_search-0.json'
            receipt = json.loads(receipt_path.read_bytes())
            receipt['response_headers'].append({'name': 'server', 'value': 'example'})
            _write_json(receipt_path, receipt)

            with self.assertRaisesRegex(IedbLiveCaptureError, 'outside the allowlist'):
                build_api_capture(root, spec)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            receipt_path = root / 'receipts/tcell_search-0.json'
            receipt = json.loads(receipt_path.read_bytes())
            receipt['response_headers'][0]['value'] += '\u2603'
            _write_json(receipt_path, receipt)

            with self.assertRaisesRegex(IedbLiveCaptureError, 'printable ASCII'):
                build_api_capture(root, spec)

        for invalid_value in ('contains\ta-control', 'x' * 16_385):
            with self.subTest(invalid_value_length=len(invalid_value)), self.assertRaises(ValidationError):
                IedbHttpHeader(name='etag', value=invalid_value)

    def test_manifest_cross_binds_api_file_commitments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = build_api_capture(root, _make_api_capture(root))
            forged = capture.manifest.model_dump(mode='json')
            forged['source_binding']['tables'][0]['pages'][0]['data_sha256'] = '0' * 64

            with self.assertRaisesRegex(ValidationError, 'data commitment does not match'):
                IedbLiveCaptureManifest.model_validate_json(canonical_json_bytes(forged))

    def test_trusted_loader_rejects_structurally_valid_semantic_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = _make_api_capture(root)
            capture = build_api_capture(root, spec)
            forged = capture.manifest.model_dump(mode='json')
            forged['source_binding']['tables'][0]['ids_sha256'] = '0' * 64
            forged_manifest = IedbLiveCaptureManifest.model_validate_json(canonical_json_bytes(forged))

            with self.assertRaisesRegex(IedbLiveCaptureError, 'do not reproduce'):
                verify_capture_manifest(root, spec, canonical_json_bytes(forged_manifest))


class IedbFullExportLiveCaptureTest(unittest.TestCase):
    def test_inventories_exact_full_export_and_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / 'exports/iedb-2026-07-06.zip'
            artifact.parent.mkdir(parents=True)
            with zipfile.ZipFile(artifact, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('csv/epitope.csv', 'epitope_id,name\n1,alpha\n')
                archive.writestr('csv/reference.csv', 'reference_id,title\n2,example\n')
            spec = _full_export_spec(root)

            capture = build_full_export_capture(root, spec)

            self.assertEqual(len(capture.manifest.inventory), 3)
            file_entry, *members = capture.manifest.inventory
            self.assertEqual(file_entry.kind, 'file')
            self.assertEqual(
                tuple(member.relative_path for member in members),
                ('csv/epitope.csv', 'csv/reference.csv'),
            )
            self.assertTrue(all(member.container_relative_path == file_entry.relative_path for member in members))
            binding = capture.manifest.source_binding
            self.assertIsInstance(binding, IedbFullExportCaptureBinding)
            assert isinstance(binding, IedbFullExportCaptureBinding)
            self.assertEqual(binding.identity, spec.identity)
            self.assertEqual(binding.completeness_scope, 'exact-supplied-artifact')
            self.assertTrue(binding.complete_inventory_within_scope)
            self.assertEqual(binding.source_identity_verification, 'caller-asserted-unverified')
            self.assertFalse(capture.manifest.source_authenticity_verified)
            self.assertFalse(capture.manifest.tier_a_eligible)
            self.assertTrue(capture.manifest.external_timestamp_required)

    def test_rejects_digest_member_count_and_mutable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / 'exports/iedb-2026-07-06.zip'
            artifact.parent.mkdir(parents=True)
            with zipfile.ZipFile(artifact, 'w') as archive:
                archive.writestr('one.csv', 'id\n1\n')
            spec = _full_export_spec(root, expected_member_count=1)
            bad_identity = spec.identity.model_copy(update={'artifact_sha256': '0' * 64})
            with self.assertRaisesRegex(IedbLiveCaptureError, 'SHA-256'):
                build_full_export_capture(root, spec.model_copy(update={'identity': bad_identity}))
            with self.assertRaisesRegex(IedbLiveCaptureError, 'member count'):
                build_full_export_capture(
                    root,
                    spec.model_copy(update={'identity': spec.identity.model_copy(update={'expected_member_count': 2})}),
                )

            with self.assertRaises(ValidationError):
                IedbFullExportIdentity.model_validate(
                    {
                        **spec.identity.model_dump(),
                        'release_id': 'iedb-latest-2026-07-06',
                    }
                )

            with self.assertRaisesRegex(ValidationError, 'credential-like query parameters'):
                IedbFullExportIdentity.model_validate(
                    {
                        **spec.identity.model_dump(),
                        'source_url': 'https://www.iedb.org/downloader.php?access_token=secret',
                    }
                )

    def test_rejects_traversal_and_duplicate_archive_member_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / 'exports/iedb-2026-07-06.zip'
            artifact.parent.mkdir(parents=True)
            with zipfile.ZipFile(artifact, 'w') as archive:
                archive.writestr('../escape.csv', 'id\n1\n')
            with self.assertRaisesRegex(ValueError, 'remain inside'):
                build_full_export_capture(root, _full_export_spec(root, expected_member_count=1))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / 'exports/iedb-2026-07-06.zip'
            artifact.parent.mkdir(parents=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(artifact, 'w') as archive:
                    archive.writestr('duplicate.csv', 'id\n1\n')
                    archive.writestr('duplicate.csv', 'id\n2\n')
            with self.assertRaisesRegex(IedbLiveCaptureError, 'duplicate member path'):
                build_full_export_capture(root, _full_export_spec(root))

    def test_rejects_archive_over_aggregate_unpacked_byte_limit_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / 'exports/iedb-2026-07-06.zip'
            artifact.parent.mkdir(parents=True)
            with zipfile.ZipFile(artifact, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('large.csv', 'x' * 1024)
            spec = _full_export_spec(root, expected_member_count=1)

            with (
                patch('vaxreplay.iedb.live_capture._MAX_ARCHIVE_UNPACKED_BYTES', 512),
                self.assertRaisesRegex(IedbLiveCaptureError, 'aggregate unpacked-byte limit'),
            ):
                build_full_export_capture(root, spec)

    def test_rejects_zip_replaced_between_scan_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / 'exports/iedb-2026-07-06.zip'
            artifact.parent.mkdir(parents=True)
            with zipfile.ZipFile(artifact, 'w') as archive:
                archive.writestr('original.csv', 'id\n1\n')
            replacement = io.BytesIO()
            with zipfile.ZipFile(replacement, 'w') as archive:
                archive.writestr('replacement.csv', 'id\n2\n')
            spec = _full_export_spec(root, expected_member_count=1)

            from vaxreplay.iedb import live_capture

            original_inventory = live_capture._inventory_zip

            def replace_then_inventory(physical_file):
                artifact.unlink()
                artifact.write_bytes(replacement.getvalue())
                return original_inventory(physical_file)

            with (
                patch('vaxreplay.iedb.live_capture._inventory_zip', side_effect=replace_then_inventory),
                self.assertRaisesRegex(IedbLiveCaptureError, 'changed between capture scan and inventory'),
            ):
                build_full_export_capture(root, spec)

    def test_manifest_cross_binds_full_export_member_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / 'exports/iedb-2026-07-06.zip'
            artifact.parent.mkdir(parents=True)
            with zipfile.ZipFile(artifact, 'w') as archive:
                archive.writestr('one.csv', 'id\n1\n')
            capture = build_full_export_capture(root, _full_export_spec(root, expected_member_count=1))
            forged = capture.manifest.model_dump(mode='json')
            forged['source_binding']['member_inventory_sha256'] = '0' * 64

            with self.assertRaisesRegex(ValidationError, 'does not bind the ZIP member inventory'):
                IedbLiveCaptureManifest.model_validate_json(canonical_json_bytes(forged))


class IedbLiveCaptureCliTest(unittest.TestCase):
    def test_capture_api_writes_canonical_manifest_and_reports_unsealed_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture_root = root / 'raw-api-capture'
            spec = _make_api_capture(capture_root)
            spec_path = root / 'api-spec.json'
            manifest_path = root / 'commitments/iedb-api.json'
            spec_path.write_bytes(canonical_json_bytes(spec))
            expected = build_api_capture(capture_root, spec)

            result, stdout_bytes = _run_iedb_cli(
                'capture-api',
                '--capture-dir',
                str(capture_root),
                '--spec',
                str(spec_path),
                '--manifest-output',
                str(manifest_path),
            )

            self.assertEqual(manifest_path.read_bytes(), canonical_json_bytes(expected.manifest))
            self.assertEqual(result['capture_manifest_sha256'], expected.manifest_sha256)
            self.assertEqual(result['source_mode'], 'api')
            self.assertEqual(result['source_build_at'], _BUILD_AT.isoformat())
            self.assertEqual(result['retrieved_at'], _RETRIEVED_AT.isoformat())
            self.assertEqual(result['capture_manifest_path'], str(manifest_path))
            self.assertEqual(result['completeness_scope'], 'declared-api-table-set')
            self.assertTrue(result['complete_within_scope'])
            self.assertFalse(result['source_authenticity_verified'])
            self.assertFalse(result['tier_a_eligible'])
            self.assertTrue(result['external_timestamp_required'])
            self.assertEqual(stdout_bytes, canonical_json_bytes(result) + b'\n')

    def test_capture_full_export_writes_canonical_manifest_and_reports_unsealed_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture_root = root / 'raw-full-export'
            artifact = capture_root / 'exports/iedb-2026-07-06.zip'
            artifact.parent.mkdir(parents=True)
            with zipfile.ZipFile(artifact, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('csv/epitope.csv', 'epitope_id,name\n1,alpha\n')
                archive.writestr('csv/reference.csv', 'reference_id,title\n2,example\n')
            spec = _full_export_spec(capture_root)
            spec_path = root / 'full-export-spec.json'
            manifest_path = root / 'commitments/iedb-full-export.json'
            spec_path.write_bytes(canonical_json_bytes(spec))
            expected = build_full_export_capture(capture_root, spec)

            result, stdout_bytes = _run_iedb_cli(
                'capture-full-export',
                '--capture-dir',
                str(capture_root),
                '--spec',
                str(spec_path),
                '--manifest-output',
                str(manifest_path),
            )

            self.assertEqual(manifest_path.read_bytes(), canonical_json_bytes(expected.manifest))
            self.assertEqual(result['capture_manifest_sha256'], expected.manifest_sha256)
            self.assertEqual(result['source_mode'], 'full_export')
            self.assertEqual(result['source_build_at'], _BUILD_AT.isoformat())
            self.assertEqual(result['retrieved_at'], _RETRIEVED_AT.isoformat())
            self.assertEqual(result['capture_manifest_path'], str(manifest_path))
            self.assertEqual(result['completeness_scope'], 'exact-supplied-artifact')
            self.assertTrue(result['complete_within_scope'])
            self.assertFalse(result['source_authenticity_verified'])
            self.assertFalse(result['tier_a_eligible'])
            self.assertTrue(result['external_timestamp_required'])
            self.assertEqual(stdout_bytes, canonical_json_bytes(result) + b'\n')

    def test_capture_refuses_to_replace_an_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture_root = root / 'raw-api-capture'
            spec = _make_api_capture(capture_root)
            spec_path = root / 'api-spec.json'
            manifest_path = root / 'commitments/iedb-api.json'
            spec_path.write_bytes(canonical_json_bytes(spec))
            manifest_path.parent.mkdir(parents=True)
            original = b'already committed\n'
            manifest_path.write_bytes(original)

            with self.assertRaisesRegex(IedbLiveCaptureError, 'already exists'):
                _run_iedb_cli(
                    'capture-api',
                    '--capture-dir',
                    str(capture_root),
                    '--spec',
                    str(spec_path),
                    '--manifest-output',
                    str(manifest_path),
                )

            self.assertEqual(manifest_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
