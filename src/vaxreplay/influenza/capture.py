"""Deterministic, offline capture of an opaque influenza record inventory.

This module is intentionally limited to software provenance.  It accepts CSV files
containing only an opaque record identifier and a source release timestamp.  It
does not accept, inspect, copy, transform, or score biological sequences.

The resulting manifest binds every input byte, every parsed source row, the exact
inclusion rule, and the complete accepted/rejected enumeration.  ``seal-target``
is designed to be submitted to an external timestamp authority before the
benchmark cutoff; timestamps supplied by the caller are metadata, not proof.
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

if TYPE_CHECKING:
    from vaxreplay.prospective import SourceCaptureArtifact

CAPTURE_SCHEMA_VERSION = 'vaxreplay.influenza-capture.v0.1'
INCLUSION_RULE_SCHEMA_VERSION = 'vaxreplay.influenza-inclusion-rule.v0.1'
OPAQUE_RECORD_SCHEMA_VERSION = 'vaxreplay.influenza-opaque-record.v0.1'
REJECTED_RECORD_SCHEMA_VERSION = 'vaxreplay.influenza-rejected-record.v0.1'
SEAL_TARGET_SCHEMA_VERSION = 'vaxreplay.influenza-capture-seal-target.v0.1'

_EXPECTED_HEADER = ('record_id', 'release_at')
_RFC3339 = re.compile(
    r'^(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}:\d{2}:\d{2})'
    r'(?P<fraction>\.\d{1,6})?(?P<offset>Z|[+-]\d{2}:\d{2})$'
)
_SHA256 = r'^[0-9a-f]{64}$'


class InfluenzaCaptureError(ValueError):
    """Raised when an opaque archive cannot be captured without guessing."""


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InfluenzaCaptureError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class CommittedInclusionRule(StrictModel):
    """Executable semantics committed into every capture manifest.

    All fields except ``rule_id`` are literals so a manifest cannot describe one
    rule while the adapter executes another.
    """

    schema_version: Literal['vaxreplay.influenza-inclusion-rule.v0.1'] = INCLUSION_RULE_SCHEMA_VERSION
    rule_id: str = Field(min_length=1, max_length=200)
    input_format: Literal['strict-rfc4180-single-line-csv'] = 'strict-rfc4180-single-line-csv'
    required_header: tuple[Literal['record_id'], Literal['release_at']] = _EXPECTED_HEADER
    identifier_semantics: Literal['opaque-verbatim-utf8'] = 'opaque-verbatim-utf8'
    availability_basis: Literal['source-release-timestamp-only'] = 'source-release-timestamp-only'
    timestamp_format: Literal['rfc3339-explicit-offset'] = 'rfc3339-explicit-offset'
    admission_predicate: Literal['release_at <= cutoff_at'] = 'release_at <= cutoff_at'
    duplicate_policy: Literal['fail-entire-capture'] = 'fail-entire-capture'
    malformed_record_policy: Literal['fail-entire-capture'] = 'fail-entire-capture'
    late_record_policy: Literal['exclude-and-audit'] = 'exclude-and-audit'
    output_order: Literal['record-id-utf8-byte-order'] = 'record-id-utf8-byte-order'


DEFAULT_INCLUSION_RULE = CommittedInclusionRule(rule_id='opaque-source-release-cutoff-v1')


class RawFileReceipt(StrictModel):
    file_name: str = Field(min_length=1, max_length=255)
    format: Literal['csv'] = 'csv'
    sha256: str = Field(pattern=_SHA256)
    byte_count: int = Field(ge=0)
    row_count: int = Field(ge=0)

    @field_validator('file_name')
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        if PurePath(value).name != value or value in {'.', '..'} or '\\' in value:
            raise ValueError('file_name must be a single portable path component')
        return value


class CapturedOpaqueRecord(StrictModel):
    schema_version: Literal['vaxreplay.influenza-opaque-record.v0.1'] = OPAQUE_RECORD_SCHEMA_VERSION
    record_id: str = Field(min_length=1, max_length=512)
    release_at: datetime
    source_file: str = Field(min_length=1, max_length=255)
    source_row: int = Field(ge=2)
    raw_row_sha256: str = Field(pattern=_SHA256)

    @field_validator('record_id')
    @classmethod
    def validate_record_id(cls, value: str) -> str:
        return _validate_opaque_id(value)

    @field_validator('release_at')
    @classmethod
    def validate_release_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('release_at must include a UTC offset')
        return value.astimezone(timezone.utc)


class RejectedOpaqueRecord(StrictModel):
    schema_version: Literal['vaxreplay.influenza-rejected-record.v0.1'] = REJECTED_RECORD_SCHEMA_VERSION
    record_id: str = Field(min_length=1, max_length=512)
    release_at: datetime
    source_file: str = Field(min_length=1, max_length=255)
    source_row: int = Field(ge=2)
    raw_row_sha256: str = Field(pattern=_SHA256)
    reason: Literal['release-after-cutoff'] = 'release-after-cutoff'

    @field_validator('record_id')
    @classmethod
    def validate_record_id(cls, value: str) -> str:
        return _validate_opaque_id(value)

    @field_validator('release_at')
    @classmethod
    def validate_release_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('release_at must include a UTC offset')
        return value.astimezone(timezone.utc)


class InfluenzaCaptureManifest(StrictModel):
    schema_version: Literal['vaxreplay.influenza-capture.v0.1'] = CAPTURE_SCHEMA_VERSION
    capture_id: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=1, max_length=200)
    cutoff_at: datetime
    retrieved_at: datetime
    inclusion_rule: CommittedInclusionRule
    inclusion_rule_sha256: str = Field(pattern=_SHA256)
    raw_files: tuple[RawFileReceipt, ...] = Field(min_length=1)
    raw_record_count: int = Field(ge=1)
    accepted_record_count: int = Field(ge=1)
    rejected_record_count: int = Field(ge=0)
    accepted_records_relative_path: Literal['accepted-records.jsonl'] = 'accepted-records.jsonl'
    accepted_records_sha256: str = Field(pattern=_SHA256)
    accepted_record_ids_sha256: str = Field(pattern=_SHA256)
    rejected_records_relative_path: Literal['rejected-records.jsonl'] = 'rejected-records.jsonl'
    rejected_records_sha256: str = Field(pattern=_SHA256)
    latest_admitted_release_at: datetime
    enumeration_basis: Literal['every-row-in-committed-raw-files'] = 'every-row-in-committed-raw-files'
    seal_target_relative_path: Literal['seal-target.json'] = 'seal-target.json'
    external_timestamp_required: Literal[True] = True

    @field_validator('cutoff_at', 'retrieved_at', 'latest_admitted_release_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('capture timestamps must include a UTC offset')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def validate_manifest(self) -> Self:
        if self.retrieved_at > self.cutoff_at:
            raise ValueError('retrieved_at cannot be after cutoff_at')
        if self.latest_admitted_release_at > self.cutoff_at:
            raise ValueError('latest admitted release cannot be after cutoff_at')
        if self.latest_admitted_release_at > self.retrieved_at:
            raise ValueError('latest admitted release cannot be after retrieved_at')
        if hashlib.sha256(canonical_json_bytes(self.inclusion_rule)).hexdigest() != self.inclusion_rule_sha256:
            raise ValueError('inclusion_rule_sha256 must bind the exact executable rule')
        names = [receipt.file_name for receipt in self.raw_files]
        if names != sorted(names, key=lambda item: item.encode('utf-8')):
            raise ValueError('raw_files must use deterministic UTF-8 filename order')
        if len(names) != len(set(names)):
            raise ValueError('raw file names must be unique')
        if sum(receipt.row_count for receipt in self.raw_files) != self.raw_record_count:
            raise ValueError('raw_record_count must equal the sum of raw file row counts')
        if self.accepted_record_count + self.rejected_record_count != self.raw_record_count:
            raise ValueError('every raw record must be either accepted or rejected')
        return self


class InfluenzaCaptureSealTarget(StrictModel):
    """Small artifact whose exact bytes are intended for external timestamping."""

    schema_version: Literal['vaxreplay.influenza-capture-seal-target.v0.1'] = SEAL_TARGET_SCHEMA_VERSION
    artifact_role: Literal['candidate-universe-or-panel'] = 'candidate-universe-or-panel'
    capture_id: str = Field(min_length=1, max_length=200)
    cutoff_at: datetime
    retrieved_at: datetime
    capture_manifest_file_sha256: str = Field(pattern=_SHA256)
    capture_manifest_byte_count: int = Field(gt=0)
    external_timestamp_required: Literal[True] = True

    @field_validator('cutoff_at', 'retrieved_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('seal target timestamps must include a UTC offset')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def validate_seal_target(self) -> Self:
        if self.retrieved_at > self.cutoff_at:
            raise ValueError('retrieved_at cannot be after cutoff_at')
        return self


@dataclass(frozen=True)
class LoadedInfluenzaCapture:
    root: Path
    manifest: InfluenzaCaptureManifest
    accepted_records: tuple[CapturedOpaqueRecord, ...]
    rejected_records: tuple[RejectedOpaqueRecord, ...]
    seal_target: InfluenzaCaptureSealTarget
    manifest_file_sha256: str
    seal_target_file_sha256: str

    @property
    def manifest_bytes(self) -> bytes:
        """Return the exact capture-manifest bytes committed by ``seal-target.json``."""

        return _read_bytes(self.root / 'manifest.json')

    def as_source_capture_artifact(self, *, witnessed_at: datetime) -> SourceCaptureArtifact:
        """Adapt this capture after its checkpoint proof has been externally verified."""

        from vaxreplay.prospective import SourceCaptureArtifact

        return SourceCaptureArtifact(
            source_id=f'{self.manifest.source_id}:{self.manifest.capture_id}',
            source_release_at=self.manifest.latest_admitted_release_at,
            captured_at=self.manifest.retrieved_at,
            witnessed_at=witnessed_at,
            manifest_bytes=self.manifest_bytes,
        )


@dataclass(frozen=True)
class _RawObservation:
    record_id: str
    release_at: datetime
    source_file: str
    source_row: int
    raw_row_sha256: str


def build_offline_capture(
    *,
    raw_files: Iterable[Path],
    output_root: Path,
    capture_id: str,
    source_id: str,
    cutoff_at: datetime,
    retrieved_at: datetime,
    inclusion_rule: CommittedInclusionRule = DEFAULT_INCLUSION_RULE,
) -> LoadedInfluenzaCapture:
    """Normalize already-fetched opaque metadata into a sealable capture.

    ``retrieved_at`` is a caller assertion and cannot establish temporal priority.
    A Tier A workflow must externally timestamp ``seal-target.json`` no later than
    ``cutoff_at``.
    """

    cutoff_at = _as_utc(cutoff_at, 'cutoff_at')
    retrieved_at = _as_utc(retrieved_at, 'retrieved_at')
    if retrieved_at > cutoff_at:
        raise InfluenzaCaptureError('post-cutoff input is forbidden: retrieved_at is after cutoff_at')

    paths = _validate_raw_paths(raw_files)
    receipts: list[RawFileReceipt] = []
    observations: list[_RawObservation] = []
    for path in paths:
        receipt, parsed = _parse_raw_csv(path)
        receipts.append(receipt)
        observations.extend(parsed)

    duplicate_ids = _duplicates(observation.record_id for observation in observations)
    if duplicate_ids:
        raise InfluenzaCaptureError(
            'opaque record IDs must be globally unique; duplicate IDs: '
            + ', '.join(repr(item) for item in duplicate_ids)
        )

    observations.sort(key=lambda record: record.record_id.encode('utf-8'))
    accepted: list[CapturedOpaqueRecord] = []
    rejected: list[RejectedOpaqueRecord] = []
    for observation in observations:
        if observation.release_at <= cutoff_at:
            if observation.release_at > retrieved_at:
                raise InfluenzaCaptureError(
                    f'{observation.source_file}:{observation.source_row} was released after the claimed retrieval time'
                )
            accepted.append(_accepted_record(observation))
        else:
            rejected.append(_rejected_record(observation))

    if not accepted:
        raise InfluenzaCaptureError('capture must contain at least one record released at or before the cutoff')

    accepted_bytes = _jsonl_bytes(accepted)
    rejected_bytes = _jsonl_bytes(rejected)
    rule_sha256 = hashlib.sha256(canonical_json_bytes(inclusion_rule)).hexdigest()
    manifest = InfluenzaCaptureManifest(
        capture_id=capture_id,
        source_id=source_id,
        cutoff_at=cutoff_at,
        retrieved_at=retrieved_at,
        inclusion_rule=inclusion_rule,
        inclusion_rule_sha256=rule_sha256,
        raw_files=tuple(receipts),
        raw_record_count=len(observations),
        accepted_record_count=len(accepted),
        rejected_record_count=len(rejected),
        accepted_records_sha256=hashlib.sha256(accepted_bytes).hexdigest(),
        accepted_record_ids_sha256=hashlib.sha256(
            canonical_json_bytes([record.record_id for record in accepted])
        ).hexdigest(),
        rejected_records_sha256=hashlib.sha256(rejected_bytes).hexdigest(),
        latest_admitted_release_at=max(record.release_at for record in accepted),
    )
    manifest_bytes = canonical_json_bytes(manifest) + b'\n'
    seal_target = InfluenzaCaptureSealTarget(
        capture_id=manifest.capture_id,
        cutoff_at=manifest.cutoff_at,
        retrieved_at=manifest.retrieved_at,
        capture_manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        capture_manifest_byte_count=len(manifest_bytes),
    )
    seal_target_bytes = canonical_json_bytes(seal_target) + b'\n'

    output_root = output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise InfluenzaCaptureError(f'output root already exists: {output_root}')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f'.{output_root.name}.', dir=output_root.parent))
    try:
        (temporary_root / 'accepted-records.jsonl').write_bytes(accepted_bytes)
        (temporary_root / 'rejected-records.jsonl').write_bytes(rejected_bytes)
        (temporary_root / 'manifest.json').write_bytes(manifest_bytes)
        (temporary_root / 'seal-target.json').write_bytes(seal_target_bytes)
        load_offline_capture(temporary_root)
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return load_offline_capture(output_root)


def load_offline_capture(root: Path) -> LoadedInfluenzaCapture:
    """Load and fully verify a normalized capture without consulting a network."""

    requested_root = root.expanduser()
    if requested_root.is_symlink():
        raise InfluenzaCaptureError('capture root cannot be a symbolic link')
    root = requested_root.resolve()
    expected_names = {
        'accepted-records.jsonl',
        'rejected-records.jsonl',
        'manifest.json',
        'seal-target.json',
    }
    try:
        children = tuple(root.iterdir())
    except OSError as error:
        raise InfluenzaCaptureError(f'cannot enumerate capture root {root}: {error}') from error
    actual_names = {path.name for path in children}
    if actual_names != expected_names:
        raise InfluenzaCaptureError(
            f'capture file inventory mismatch: expected {sorted(expected_names)}, found {sorted(actual_names)}'
        )
    for path in children:
        if path.is_symlink() or not path.is_file():
            raise InfluenzaCaptureError(f'capture artifact must be a regular non-symlink file: {path.name}')

    manifest_path = root / 'manifest.json'
    manifest_bytes = _read_bytes(manifest_path)
    try:
        manifest = InfluenzaCaptureManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise InfluenzaCaptureError(f'invalid capture manifest: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest) + b'\n':
        raise InfluenzaCaptureError('manifest.json must use exact canonical JSON encoding')

    accepted_path = root / manifest.accepted_records_relative_path
    rejected_path = root / manifest.rejected_records_relative_path
    accepted_bytes = _read_bytes(accepted_path)
    rejected_bytes = _read_bytes(rejected_path)
    if hashlib.sha256(accepted_bytes).hexdigest() != manifest.accepted_records_sha256:
        raise InfluenzaCaptureError('accepted record hash mismatch')
    if hashlib.sha256(rejected_bytes).hexdigest() != manifest.rejected_records_sha256:
        raise InfluenzaCaptureError('rejected record hash mismatch')

    accepted = _load_jsonl(accepted_bytes, CapturedOpaqueRecord, 'accepted-records.jsonl')
    rejected = _load_jsonl(rejected_bytes, RejectedOpaqueRecord, 'rejected-records.jsonl')
    _validate_enumeration(manifest, accepted, rejected)

    seal_target_path = root / manifest.seal_target_relative_path
    seal_target_bytes = _read_bytes(seal_target_path)
    try:
        seal_target = InfluenzaCaptureSealTarget.model_validate_json(seal_target_bytes)
    except ValueError as error:
        raise InfluenzaCaptureError(f'invalid seal target: {error}') from error
    if seal_target_bytes != canonical_json_bytes(seal_target) + b'\n':
        raise InfluenzaCaptureError('seal-target.json must use exact canonical JSON encoding')
    manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if seal_target.capture_id != manifest.capture_id:
        raise InfluenzaCaptureError('seal target capture_id does not match the manifest')
    if seal_target.cutoff_at != manifest.cutoff_at or seal_target.retrieved_at != manifest.retrieved_at:
        raise InfluenzaCaptureError('seal target timestamps do not match the manifest')
    if seal_target.capture_manifest_file_sha256 != manifest_file_sha256:
        raise InfluenzaCaptureError('seal target does not bind the exact manifest file')
    if seal_target.capture_manifest_byte_count != len(manifest_bytes):
        raise InfluenzaCaptureError('seal target manifest byte count mismatch')

    return LoadedInfluenzaCapture(
        root=root,
        manifest=manifest,
        accepted_records=accepted,
        rejected_records=rejected,
        seal_target=seal_target,
        manifest_file_sha256=manifest_file_sha256,
        seal_target_file_sha256=hashlib.sha256(seal_target_bytes).hexdigest(),
    )


def verify_raw_files(capture: LoadedInfluenzaCapture, raw_files: Iterable[Path]) -> None:
    """Re-parse source files and prove they exactly reproduce the committed capture."""

    paths = _validate_raw_paths(raw_files)
    receipts: list[RawFileReceipt] = []
    observations: list[_RawObservation] = []
    for path in paths:
        receipt, parsed = _parse_raw_csv(path)
        receipts.append(receipt)
        observations.extend(parsed)
    if tuple(receipts) != capture.manifest.raw_files:
        raise InfluenzaCaptureError('raw file receipts do not match the capture manifest')
    duplicate_ids = _duplicates(observation.record_id for observation in observations)
    if duplicate_ids:
        raise InfluenzaCaptureError('raw files contain duplicate opaque record IDs')
    observations.sort(key=lambda record: record.record_id.encode('utf-8'))
    expected_accepted: list[CapturedOpaqueRecord] = []
    expected_rejected: list[RejectedOpaqueRecord] = []
    for observation in observations:
        if observation.release_at <= capture.manifest.cutoff_at:
            expected_accepted.append(_accepted_record(observation))
        else:
            expected_rejected.append(_rejected_record(observation))
    if tuple(expected_accepted) != capture.accepted_records or tuple(expected_rejected) != capture.rejected_records:
        raise InfluenzaCaptureError('raw files do not reproduce the committed record enumeration')


def _validate_raw_paths(raw_files: Iterable[Path]) -> tuple[Path, ...]:
    provided = tuple(Path(path).expanduser() for path in raw_files)
    if not provided:
        raise InfluenzaCaptureError('at least one raw CSV file is required')
    paths: list[Path] = []
    for path in provided:
        if path.is_symlink():
            raise InfluenzaCaptureError(f'raw input cannot be a symbolic link: {path}')
        resolved = path.resolve()
        if not resolved.is_file():
            raise InfluenzaCaptureError(f'raw input must be a regular file: {path}')
        if resolved.suffix.lower() != '.csv':
            raise InfluenzaCaptureError(f'raw input must use the .csv extension: {path.name}')
        paths.append(resolved)
    if len(paths) != len(set(paths)):
        raise InfluenzaCaptureError('the same raw file cannot be supplied more than once')
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise InfluenzaCaptureError('raw CSV basenames must be unique')
    return tuple(sorted(paths, key=lambda path: path.name.encode('utf-8')))


def _accepted_record(observation: _RawObservation) -> CapturedOpaqueRecord:
    return CapturedOpaqueRecord(
        record_id=observation.record_id,
        release_at=observation.release_at,
        source_file=observation.source_file,
        source_row=observation.source_row,
        raw_row_sha256=observation.raw_row_sha256,
    )


def _rejected_record(observation: _RawObservation) -> RejectedOpaqueRecord:
    return RejectedOpaqueRecord(
        record_id=observation.record_id,
        release_at=observation.release_at,
        source_file=observation.source_file,
        source_row=observation.source_row,
        raw_row_sha256=observation.raw_row_sha256,
    )


def _parse_raw_csv(path: Path) -> tuple[RawFileReceipt, tuple[_RawObservation, ...]]:
    digest = hashlib.sha256()
    byte_count = 0
    observations: list[_RawObservation] = []
    header_seen = False
    try:
        with path.open('rb') as source:
            for physical_line, raw_line in enumerate(source, start=1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                if b'\x00' in raw_line:
                    raise InfluenzaCaptureError(f'{path.name}:{physical_line} contains a NUL byte')
                try:
                    text = raw_line.decode('utf-8')
                except UnicodeDecodeError as error:
                    raise InfluenzaCaptureError(f'{path.name}:{physical_line} is not strict UTF-8') from error
                try:
                    parsed_rows = list(csv.reader([text], strict=True))
                except csv.Error as error:
                    raise InfluenzaCaptureError(
                        f'{path.name}:{physical_line} is not a single well-formed CSV row: {error}'
                    ) from error
                if len(parsed_rows) != 1 or not parsed_rows[0]:
                    raise InfluenzaCaptureError(f'{path.name}:{physical_line} cannot be blank')
                row = parsed_rows[0]
                if not header_seen:
                    if tuple(row) != _EXPECTED_HEADER:
                        raise InfluenzaCaptureError(
                            f'{path.name}:1 must have the exact header record_id,release_at and no other columns'
                        )
                    header_seen = True
                    continue
                if len(row) != len(_EXPECTED_HEADER):
                    raise InfluenzaCaptureError(
                        f'{path.name}:{physical_line} must have exactly two fields: record_id and release_at'
                    )
                record_id = _validate_opaque_id(row[0], location=f'{path.name}:{physical_line}')
                release_at = _parse_release_timestamp(row[1], location=f'{path.name}:{physical_line}')
                observations.append(
                    _RawObservation(
                        record_id=record_id,
                        release_at=release_at,
                        source_file=path.name,
                        source_row=physical_line,
                        raw_row_sha256=hashlib.sha256(raw_line).hexdigest(),
                    )
                )
    except OSError as error:
        raise InfluenzaCaptureError(f'cannot read raw file {path}: {error}') from error
    if not header_seen:
        raise InfluenzaCaptureError(f'{path.name} is empty; the exact CSV header is required')
    return (
        RawFileReceipt(
            file_name=path.name,
            sha256=digest.hexdigest(),
            byte_count=byte_count,
            row_count=len(observations),
        ),
        tuple(observations),
    )


def _validate_opaque_id(value: str, *, location: str = 'record_id') -> str:
    if not value:
        raise InfluenzaCaptureError(f'{location} cannot be empty')
    if len(value) > 512:
        raise InfluenzaCaptureError(f'{location} exceeds 512 Unicode code points')
    if value != value.strip():
        raise InfluenzaCaptureError(f'{location} cannot have leading or trailing whitespace')
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise InfluenzaCaptureError(f'{location} cannot contain control characters')
    return value


def _parse_release_timestamp(value: str, *, location: str) -> datetime:
    match = _RFC3339.fullmatch(value)
    if match is None or value.endswith('-00:00'):
        raise InfluenzaCaptureError(
            f'{location} release_at must be unambiguous RFC3339 with an explicit known UTC offset'
        )
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise InfluenzaCaptureError(f'{location} has an invalid release_at timestamp') from error
    if parsed.utcoffset() is None:
        raise InfluenzaCaptureError(f'{location} release_at must include a UTC offset')
    return parsed.astimezone(timezone.utc)


def _jsonl_bytes(records: Iterable[StrictModel]) -> bytes:
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


def _load_jsonl[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    file_name: str,
) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b'\n') or line == b'\n':
            raise InfluenzaCaptureError(f'{file_name}:{line_number} must be one nonblank canonical JSON line')
        try:
            record = model.model_validate_json(line[:-1])
        except ValueError as error:
            raise InfluenzaCaptureError(f'invalid {file_name}:{line_number}: {error}') from error
        if line != canonical_json_bytes(record) + b'\n':
            raise InfluenzaCaptureError(f'{file_name}:{line_number} must use exact canonical JSON encoding')
        records.append(record)
    return tuple(records)


def _validate_enumeration(
    manifest: InfluenzaCaptureManifest,
    accepted: tuple[CapturedOpaqueRecord, ...],
    rejected: tuple[RejectedOpaqueRecord, ...],
) -> None:
    if len(accepted) != manifest.accepted_record_count:
        raise InfluenzaCaptureError('accepted record count mismatch')
    if len(rejected) != manifest.rejected_record_count:
        raise InfluenzaCaptureError('rejected record count mismatch')
    accepted_ids = [record.record_id for record in accepted]
    rejected_ids = [record.record_id for record in rejected]
    if accepted_ids != sorted(accepted_ids, key=lambda item: item.encode('utf-8')):
        raise InfluenzaCaptureError('accepted records are not in committed UTF-8 identifier order')
    if rejected_ids != sorted(rejected_ids, key=lambda item: item.encode('utf-8')):
        raise InfluenzaCaptureError('rejected records are not in committed UTF-8 identifier order')
    if len(accepted_ids + rejected_ids) != len(set(accepted_ids + rejected_ids)):
        raise InfluenzaCaptureError('opaque record IDs are not globally unique')
    if any(record.release_at > manifest.cutoff_at for record in accepted):
        raise InfluenzaCaptureError('accepted record was released after the cutoff')
    if any(record.release_at > manifest.retrieved_at for record in accepted):
        raise InfluenzaCaptureError('accepted record was released after the claimed retrieval time')
    if any(record.release_at <= manifest.cutoff_at for record in rejected):
        raise InfluenzaCaptureError('rejected record does not satisfy the committed late-record rule')
    if max(record.release_at for record in accepted) != manifest.latest_admitted_release_at:
        raise InfluenzaCaptureError('latest admitted release timestamp mismatch')
    if hashlib.sha256(canonical_json_bytes(accepted_ids)).hexdigest() != manifest.accepted_record_ids_sha256:
        raise InfluenzaCaptureError('accepted record identifier commitment mismatch')

    receipt_by_name = {receipt.file_name: receipt for receipt in manifest.raw_files}
    rows_by_file: dict[str, set[int]] = {name: set() for name in receipt_by_name}
    for record in (*accepted, *rejected):
        receipt = receipt_by_name.get(record.source_file)
        if receipt is None:
            raise InfluenzaCaptureError(f'record references unknown raw file {record.source_file}')
        if record.source_row > receipt.row_count + 1:
            raise InfluenzaCaptureError(f'record references impossible source row {record.source_row}')
        if record.source_row in rows_by_file[record.source_file]:
            raise InfluenzaCaptureError('multiple records reference the same raw source row')
        rows_by_file[record.source_file].add(record.source_row)
    for receipt in manifest.raw_files:
        expected_rows = set(range(2, receipt.row_count + 2))
        if rows_by_file[receipt.file_name] != expected_rows:
            raise InfluenzaCaptureError(f'capture does not enumerate every row in {receipt.file_name}')


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates, key=lambda item: item.encode('utf-8')))


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise InfluenzaCaptureError(f'cannot read {path}: {error}') from error
