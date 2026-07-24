"""Exact-byte verification and typed parsing for NCT-scoped AACT table slices."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.schema import (
    AactArchiveReceipt,
    AactSliceReceipt,
    AactSourceTable,
    AactTableReceipt,
)


class AactSliceError(ValueError):
    """Raised when an AACT archive receipt or extracted slice is not exact and self-consistent."""


class StudyRow(StrictModel):
    nct_id: str = Field(min_length=1)
    results_first_submitted_date: str = ''
    results_first_posted_date: str = ''
    last_update_posted_date: str = Field(min_length=1)
    brief_title: str = ''
    official_title: str = ''
    overall_status: str = ''
    phase: str = ''
    source: str = ''
    number_of_arms: str = ''


class DesignRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    allocation: str = ''
    intervention_model: str = ''
    primary_purpose: str = ''
    masking: str = ''


class DesignGroupRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    group_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DesignGroupInterventionRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    design_group_id: str = Field(min_length=1)
    intervention_id: str = Field(min_length=1)


class InterventionRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    intervention_type: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ''


class DesignOutcomeRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    outcome_type: str = Field(min_length=1)
    measure: str = Field(min_length=1)
    time_frame: str = Field(min_length=1)
    population: str = ''
    description: str = ''


class ResultGroupRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    ctgov_group_code: str = Field(min_length=1)
    result_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class OutcomeRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    outcome_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ''
    time_frame: str = Field(min_length=1)
    population: str = ''
    units: str = ''
    dispersion_type: str = ''
    param_type: str = Field(min_length=1)


class OutcomeCountRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    outcome_id: str = Field(min_length=1)
    result_group_id: str = Field(min_length=1)
    ctgov_group_code: str = Field(min_length=1)
    scope: str = ''
    units: str = ''
    count: str = ''


class OutcomeMeasurementRow(StrictModel):
    id: str = Field(min_length=1)
    nct_id: str = Field(min_length=1)
    outcome_id: str = Field(min_length=1)
    result_group_id: str = Field(min_length=1)
    ctgov_group_code: str = Field(min_length=1)
    classification: str = ''
    category: str = ''
    title: str = Field(min_length=1)
    description: str = ''
    units: str = ''
    param_type: str = Field(min_length=1)
    param_value: str = ''
    param_value_num: float | None = Field(default=None, allow_inf_nan=False)
    dispersion_type: str = ''
    dispersion_value: str = ''
    dispersion_value_num: float | None = Field(default=None, allow_inf_nan=False)
    dispersion_lower_limit: str = ''
    dispersion_upper_limit: str = ''
    explanation_of_na: str = ''


type AactSourceRow = (
    StudyRow
    | DesignRow
    | DesignGroupRow
    | DesignGroupInterventionRow
    | InterventionRow
    | DesignOutcomeRow
    | ResultGroupRow
    | OutcomeRow
    | OutcomeCountRow
    | OutcomeMeasurementRow
)


_ROW_MODEL: dict[AactSourceTable, type[AactSourceRow]] = {
    AactSourceTable.STUDIES: StudyRow,
    AactSourceTable.DESIGNS: DesignRow,
    AactSourceTable.DESIGN_GROUPS: DesignGroupRow,
    AactSourceTable.DESIGN_GROUP_INTERVENTIONS: DesignGroupInterventionRow,
    AactSourceTable.INTERVENTIONS: InterventionRow,
    AactSourceTable.DESIGN_OUTCOMES: DesignOutcomeRow,
    AactSourceTable.RESULT_GROUPS: ResultGroupRow,
    AactSourceTable.OUTCOMES: OutcomeRow,
    AactSourceTable.OUTCOME_COUNTS: OutcomeCountRow,
    AactSourceTable.OUTCOME_MEASUREMENTS: OutcomeMeasurementRow,
}


@dataclass(frozen=True)
class LoadedAactSlice:
    root: Path
    receipt: AactSliceReceipt
    receipt_sha256: str
    rows_by_table: dict[AactSourceTable, tuple[AactSourceRow, ...]]

    def rows[RowT: AactSourceRow](self, table: AactSourceTable, model: type[RowT]) -> tuple[RowT, ...]:
        raw_rows = self.rows_by_table.get(table)
        if raw_rows is None:
            raise AactSliceError(f'slice {self.receipt.slice_id} does not contain {table.value}')
        if _ROW_MODEL[table] is not model:
            raise AactSliceError(f'{table.value} is parsed as {_ROW_MODEL[table].__name__}, not {model.__name__}')
        return tuple(row for row in raw_rows if isinstance(row, model))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise AactSliceError(f'cannot read {path}: {error}') from error
    return digest.hexdigest()


def verify_archive_file(path: Path, receipt: AactArchiveReceipt) -> None:
    """Verify a retained full archive against the immutable receipt."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AactSliceError(f'AACT archive must be a regular, non-symlink file: {expanded}')
    path = expanded.resolve()
    if not path.is_file():
        raise AactSliceError(f'AACT archive must be a regular, non-symlink file: {path}')
    try:
        byte_count = path.stat().st_size
    except OSError as error:
        raise AactSliceError(f'cannot stat {path}: {error}') from error
    if byte_count != receipt.archive_bytes:
        raise AactSliceError(
            f'archive byte count mismatch for {receipt.snapshot_id}: {byte_count} != {receipt.archive_bytes}'
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != receipt.archive_sha256:
        raise AactSliceError(f'archive SHA-256 mismatch for {receipt.snapshot_id}')


def inspect_table(path: Path, table: AactSourceTable, nct_id: str) -> AactTableReceipt:
    """Derive a table receipt after validating its exact NCT scope and typed rows."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AactSliceError(f'AACT slice table must be a regular, non-symlink file: {expanded}')
    path = expanded.resolve()
    if not path.is_file():
        raise AactSliceError(f'AACT slice table must be a regular, non-symlink file: {path}')
    header, rows = _parse_table(path, table, nct_id)
    return AactTableReceipt(
        table=table,
        source_member_path=f'{table.value}.txt',
        relative_path=f'{table.value}.txt',
        sha256=file_sha256(path),
        byte_count=path.stat().st_size,
        row_count=len(rows),
        header_sha256=_header_sha256(header),
    )


def load_slice(root: Path, receipt: AactSliceReceipt) -> LoadedAactSlice:
    """Verify exact inventory/hashes and parse every table in a receipt."""

    expanded = root.expanduser()
    if expanded.is_symlink():
        raise AactSliceError(f'AACT slice root must be a directory, not a symlink: {expanded}')
    root = expanded.resolve()
    if not root.is_dir():
        raise AactSliceError(f'AACT slice root must be a directory, not a symlink: {root}')
    expected_files = {table.relative_path for table in receipt.tables}
    actual_files: set[str] = set()
    try:
        for path in root.rglob('*'):
            if path.is_symlink():
                raise AactSliceError(f'AACT slices cannot contain symbolic links: {path}')
            if path.is_file():
                actual_files.add(path.relative_to(root).as_posix())
    except OSError as error:
        raise AactSliceError(f'cannot enumerate AACT slice {root}: {error}') from error
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise AactSliceError(f'AACT slice inventory mismatch; missing={missing}, extra={extra}')

    rows_by_table: dict[AactSourceTable, tuple[AactSourceRow, ...]] = {}
    for table_receipt in receipt.tables:
        path = _safe_slice_path(root, table_receipt.relative_path)
        if file_sha256(path) != table_receipt.sha256:
            raise AactSliceError(f'{table_receipt.table.value} SHA-256 does not match its receipt')
        if path.stat().st_size != table_receipt.byte_count:
            raise AactSliceError(f'{table_receipt.table.value} byte count does not match its receipt')
        header, rows = _parse_table(path, table_receipt.table, receipt.nct_id)
        if _header_sha256(header) != table_receipt.header_sha256:
            raise AactSliceError(f'{table_receipt.table.value} header does not match its receipt')
        if len(rows) != table_receipt.row_count:
            raise AactSliceError(f'{table_receipt.table.value} row count does not match its receipt')
        rows_by_table[table_receipt.table] = rows

    receipt_sha256 = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    return LoadedAactSlice(
        root=root,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        rows_by_table=rows_by_table,
    )


def extract_nct_slice_from_archive(
    *,
    archive_path: Path,
    archive_receipt: AactArchiveReceipt,
    nct_id: str,
    tables: Iterable[AactSourceTable],
    output_root: Path,
    slice_id: str,
    created_at: datetime,
) -> AactSliceReceipt:
    """Regenerate a deterministic exact-NCT slice from a verified full AACT ZIP.

    The ZIP is content-verified before it is opened.  Each requested archive member must occur
    exactly once at the ZIP root.  Rows are parsed with AACT's documented pipe/double-quote CSV
    rules and reserialized canonically, so embedded delimiters and newlines cannot bypass the NCT
    filter.  No unrequested member is extracted.
    """

    ordered_tables = tuple(sorted(tables, key=lambda item: item.value))
    if not ordered_tables or len(ordered_tables) != len(set(ordered_tables)):
        raise AactSliceError('archive extraction tables must be a non-empty unique collection')
    verify_archive_file(archive_path, archive_receipt)
    archive_path = archive_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise AactSliceError(f'output directory already exists: {output_root}')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f'.{output_root.name}.', dir=output_root.parent)).resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                members_by_name.setdefault(info.filename, []).append(info)
            for table in ordered_tables:
                member_name = f'{table.value}.txt'
                matches = members_by_name.get(member_name, [])
                if len(matches) != 1:
                    raise AactSliceError(
                        f'archive must contain exactly one top-level {member_name}; found {len(matches)}'
                    )
                info = matches[0]
                if info.is_dir() or info.flag_bits & 0x1:
                    raise AactSliceError(f'archive member must be a readable regular file: {member_name}')
                _extract_exact_nct_member(
                    archive,
                    info,
                    temporary_root / member_name,
                    nct_id,
                )
        table_receipts = tuple(
            inspect_table(temporary_root / f'{table.value}.txt', table, nct_id) for table in ordered_tables
        )
        receipt = AactSliceReceipt(
            slice_id=slice_id,
            nct_id=nct_id,
            archive=archive_receipt,
            created_at=created_at,
            tables=table_receipts,
        )
        load_slice(temporary_root, receipt)
        temporary_root.rename(output_root)
        return receipt
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise AactSliceError(f'failed to extract verified AACT archive: {error}') from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)


def _extract_exact_nct_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination_path: Path,
    nct_id: str,
) -> None:
    try:
        with archive.open(info, 'r') as raw_source:
            with io.TextIOWrapper(raw_source, encoding='utf-8', newline='') as source:
                reader = csv.reader(
                    source,
                    delimiter='|',
                    quotechar='"',
                    doublequote=True,
                    quoting=csv.QUOTE_MINIMAL,
                    strict=True,
                )
                header = next(reader, None)
                if header is None or 'nct_id' not in header:
                    raise AactSliceError(f'{info.filename} is empty or lacks nct_id')
                if len(header) != len(set(header)) or any(not column for column in header):
                    raise AactSliceError(f'{info.filename} has an invalid header')
                nct_index = header.index('nct_id')
                with destination_path.open('w', encoding='utf-8', newline='') as destination:
                    writer = csv.writer(
                        destination,
                        delimiter='|',
                        quotechar='"',
                        doublequote=True,
                        quoting=csv.QUOTE_MINIMAL,
                        lineterminator='\n',
                    )
                    writer.writerow(header)
                    for row_number, row in enumerate(reader, start=2):
                        if len(row) != len(header):
                            raise AactSliceError(
                                f'{info.filename} row {row_number} has {len(row)} fields; expected {len(header)}'
                            )
                        if row[nct_index] == nct_id:
                            writer.writerow(row)
    except UnicodeDecodeError as error:
        raise AactSliceError(f'{info.filename} is not valid UTF-8: {error}') from error
    except csv.Error as error:
        raise AactSliceError(f'invalid AACT pipe data in {info.filename}: {error}') from error


def _safe_slice_path(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if path.is_symlink():
        raise AactSliceError(f'AACT slices cannot contain symbolic links: {relative_path}')
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise AactSliceError(f'AACT slice path escapes its root: {relative_path}')
    if not resolved.is_file():
        raise AactSliceError(f'AACT slice table is not a regular file: {relative_path}')
    return resolved


def _parse_table(
    path: Path,
    table: AactSourceTable,
    nct_id: str,
) -> tuple[tuple[str, ...], tuple[AactSourceRow, ...]]:
    try:
        with path.open(encoding='utf-8', newline='') as source:
            # AACT's flat-file contract uses double-quoted fields when a value contains ``|``.
            # The standard reader also handles doubled quotes and quoted embedded newlines.
            reader = csv.reader(
                source,
                delimiter='|',
                quotechar='"',
                doublequote=True,
                quoting=csv.QUOTE_MINIMAL,
                strict=True,
            )
            raw_header = next(reader, None)
            if raw_header is None:
                raise AactSliceError(f'{path} is empty')
            header = tuple(raw_header)
            if not header or any(not column for column in header) or len(header) != len(set(header)):
                raise AactSliceError(f'{path} must have a non-empty header with unique columns')
            required = set(_required_columns(table))
            missing_columns = sorted(required - set(header))
            if missing_columns:
                raise AactSliceError(f'{table.value} is missing required columns {missing_columns}')
            rows: list[AactSourceRow] = []
            row_keys: set[str] = set()
            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise AactSliceError(
                        f'{table.value} line {line_number} has {len(values)} fields, expected {len(header)}'
                    )
                raw = dict(zip(header, values, strict=True))
                if raw['nct_id'] != nct_id:
                    raise AactSliceError(
                        f'{table.value} line {line_number} contains {raw["nct_id"]!r}, expected exact NCT {nct_id}'
                    )
                row = _typed_row(table, raw, line_number)
                row_key = row.nct_id if isinstance(row, StudyRow) else row.id
                if row_key in row_keys:
                    raise AactSliceError(f'{table.value} contains duplicate source row ID {row_key}')
                row_keys.add(row_key)
                rows.append(row)
    except UnicodeDecodeError as error:
        raise AactSliceError(f'{path} is not valid UTF-8: {error}') from error
    except csv.Error as error:
        raise AactSliceError(f'invalid pipe-delimited data in {path}: {error}') from error
    except OSError as error:
        raise AactSliceError(f'cannot read {path}: {error}') from error
    return header, tuple(rows)


def _required_columns(table: AactSourceTable) -> tuple[str, ...]:
    model = _ROW_MODEL[table]
    return tuple(model.model_fields)


def _typed_row(table: AactSourceTable, raw: dict[str, str], line_number: int) -> AactSourceRow:
    model = _ROW_MODEL[table]
    values: dict[str, Any] = {field: raw[field] for field in model.model_fields}
    if model is OutcomeMeasurementRow:
        values['param_value_num'] = _optional_float(raw['param_value_num'], table, line_number, 'param_value_num')
        values['dispersion_value_num'] = _optional_float(
            raw['dispersion_value_num'], table, line_number, 'dispersion_value_num'
        )
    try:
        return model.model_validate(values)
    except ValueError as error:
        raise AactSliceError(f'invalid {table.value} row at line {line_number}: {error}') from error


def _optional_float(value: str, table: AactSourceTable, line_number: int, field_name: str) -> float | None:
    if not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise AactSliceError(f'{table.value} line {line_number} has invalid {field_name}') from error
    if not math.isfinite(parsed):
        raise AactSliceError(f'{table.value} line {line_number} has non-finite {field_name}')
    return parsed


def _header_sha256(header: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(header)).hexdigest()
