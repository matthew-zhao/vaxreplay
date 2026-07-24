"""Fail-closed normalization of prospective IEDB source captures.

This module does not fetch data or authenticate the source.  A collector first downloads
either a declared set of complete IQ-API tables (including the request/response receipts
and API metrics bracketing the fetch) or a caller-identified full-export artifact.  The
functions below then validate those bytes offline and produce one deterministic JSON
manifest.  Those exact manifest bytes are suitable for an external timestamp and for use
as a :class:`vaxreplay.prospective.SourceCaptureArtifact`.

The distinction between a retrieval timestamp and an independently witnessed timestamp is
intentional: this module proves what the supplied capture contains and completeness only
within its explicitly declared scope.  It does not prove that a caller-supplied artifact
came from IEDB or that its caller-supplied timestamps are true.  The prospective
decision-package workflow proves when the resulting commitment existed only after an
independent witness has accepted it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.iedb.raw_schema import IedbApiMetric

IEDB_LIVE_CAPTURE_SCHEMA_VERSION = 'vaxreplay.iedb-live-capture.v0.2'
IEDB_API_EXCHANGE_SCHEMA_VERSION = 'vaxreplay.iedb-api-exchange.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_TABLE_PATTERN = r'^[a-z][a-z0-9_]*$'
_FIELD_PATTERN = r'^[a-z][a-z0-9_]*$'
_CAPTURE_ID_PATTERN = r'^[a-z0-9][a-z0-9._-]{2,127}$'
_OFFICIAL_API_ORIGIN = 'https://query-api.iedb.org'
_OFFICIAL_EXPORT_HOSTS = frozenset({'iedb.org', 'www.iedb.org', 'help.iedb.org'})
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_API_PAGE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_INVENTORY_FILES = 100_000
_MAX_ARCHIVE_MEMBERS = 100_000
_MAX_ARCHIVE_UNPACKED_BYTES = 64 * 1024 * 1024 * 1024
_SECRET_BEARING_HEADERS = frozenset(
    {
        'authorization',
        'cookie',
        'proxy-authorization',
        'set-cookie',
        'x-api-key',
    }
)
_API_REQUEST_HEADER_ALLOWLIST = frozenset(
    {
        'accept',
        'accept-encoding',
        'host',
        'range',
        'range-unit',
        'user-agent',
    }
)
_API_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        'cache-control',
        'content-encoding',
        'content-length',
        'content-range',
        'content-type',
        'date',
        'etag',
        'expires',
        'last-modified',
        'vary',
    }
)
_CREDENTIAL_QUERY_NAME = re.compile(
    r'(?:^|[_-])(?:access[_-]?token|api[_-]?key|auth(?:orization)?|code|credential|jwt|key|'
    r'password|secret|sig(?:nature)?|token)(?:$|[_-])',
    re.IGNORECASE,
)


class IedbLiveCaptureError(ValueError):
    """Raised when raw bytes cannot support an immutable, scoped capture manifest."""


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _validate_relative_path(value: str, field_name: str) -> str:
    if not value or '\\' in value or '\x00' in value or unicodedata.normalize('NFC', value) != value:
        raise ValueError(f'{field_name} must be a nonempty NFC POSIX path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '.' in path.parts or path.as_posix() != value:
        raise ValueError(f'{field_name} must be normalized and remain inside the capture root')
    return value


def _validate_official_api_url(value: str, *, expected_path: str) -> tuple[tuple[str, str], ...]:
    parsed = urlsplit(value)
    if (
        parsed.scheme != 'https'
        or parsed.hostname != 'query-api.iedb.org'
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected_path
        or parsed.fragment
    ):
        raise ValueError(f'IEDB API URL must use {_OFFICIAL_API_ORIGIN}{expected_path}')
    try:
        pairs = tuple(parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True))
    except ValueError as error:
        raise ValueError('IEDB API URL contains an ambiguous query string') from error
    names = tuple(name for name, _value in pairs)
    if any(not name or not item for name, item in pairs) or len(names) != len(set(names)):
        raise ValueError('IEDB API query parameters must be nonempty and unique')
    _reject_credential_query_parameters(pairs)
    return pairs


def _parse_strict_query(value: str, label: str) -> tuple[tuple[str, str], ...]:
    try:
        pairs = tuple(parse_qsl(value, keep_blank_values=True, strict_parsing=True))
    except ValueError as error:
        raise ValueError(f'{label} contains an ambiguous query string') from error
    names = tuple(name for name, _value in pairs)
    if any(not name or not item for name, item in pairs) or len(names) != len(set(names)):
        raise ValueError(f'{label} query parameters must be nonempty and unique')
    _reject_credential_query_parameters(pairs)
    return pairs


def _reject_credential_query_parameters(pairs: tuple[tuple[str, str], ...]) -> None:
    forbidden = sorted(name for name, _value in pairs if _CREDENTIAL_QUERY_NAME.search(name))
    if forbidden:
        raise ValueError(f'capture URLs cannot persist credential-like query parameters: {forbidden!r}')


def _validate_header_set(
    value: tuple[IedbHttpHeader, ...],
    *,
    allowlist: frozenset[str],
    direction: str,
) -> tuple[IedbHttpHeader, ...]:
    names = tuple(header.name for header in value)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError('HTTP headers must use unique lowercase names in sorted order')
    forbidden = sorted(set(names) & _SECRET_BEARING_HEADERS)
    if forbidden:
        raise ValueError(f'HTTP exchange receipts cannot persist secret-bearing headers: {forbidden!r}')
    unexpected = sorted(set(names) - allowlist)
    if unexpected:
        raise ValueError(f'HTTP {direction} receipt contains headers outside the allowlist: {unexpected!r}')
    return value


class IedbHttpHeader(StrictModel):
    """One already-normalized HTTP header from an offline exchange receipt."""

    name: str = Field(max_length=128, pattern=r'^[a-z0-9][a-z0-9-]*$')
    value: str = Field(min_length=1, max_length=16_384)

    @field_validator('value')
    @classmethod
    def validate_value(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError('HTTP header values cannot contain outer whitespace')
        if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
            raise ValueError('HTTP header values must contain printable ASCII without control characters')
        return value


class IedbApiExchangeReceipt(StrictModel):
    """Collector-produced record of the exact request and response metadata for one page."""

    schema_version: Literal['vaxreplay.iedb-api-exchange.v0.1'] = IEDB_API_EXCHANGE_SCHEMA_VERSION
    request_url: str = Field(min_length=1)
    request_headers: tuple[IedbHttpHeader, ...] = Field(min_length=1)
    status_code: Literal[200, 206]
    response_headers: tuple[IedbHttpHeader, ...] = Field(min_length=1)

    @field_validator('request_headers')
    @classmethod
    def validate_request_headers(cls, value: tuple[IedbHttpHeader, ...]) -> tuple[IedbHttpHeader, ...]:
        return _validate_header_set(value, allowlist=_API_REQUEST_HEADER_ALLOWLIST, direction='request')

    @field_validator('response_headers')
    @classmethod
    def validate_response_headers(cls, value: tuple[IedbHttpHeader, ...]) -> tuple[IedbHttpHeader, ...]:
        return _validate_header_set(value, allowlist=_API_RESPONSE_HEADER_ALLOWLIST, direction='response')

    def request_header(self, name: str) -> str | None:
        return next((header.value for header in self.request_headers if header.name == name), None)

    def response_header(self, name: str) -> str | None:
        return next((header.value for header in self.response_headers if header.name == name), None)


class IedbApiPageSpec(StrictModel):
    table_name: str = Field(pattern=_TABLE_PATTERN)
    id_field: str = Field(pattern=_FIELD_PATTERN)
    request_url: str = Field(min_length=1)
    data_relative_path: str = Field(min_length=1)
    receipt_relative_path: str = Field(min_length=1)
    data_format: Literal['json', 'jsonl', 'csv']

    @field_validator('data_relative_path', 'receipt_relative_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_path(value, 'API page path')

    @model_validator(mode='after')
    def validate_query(self) -> Self:
        if self.data_relative_path == self.receipt_relative_path:
            raise ValueError('API data and receipt paths must be different')
        pairs = _validate_official_api_url(self.request_url, expected_path=f'/{self.table_name}')
        query = dict(pairs)
        if set(query) - {'order', 'select'}:
            raise ValueError('complete API captures permit only order and optional select query parameters')
        if query.get('order') not in {self.id_field, f'{self.id_field}.asc'}:
            raise ValueError('API page query must use a unique ascending ID order')
        selected = query.get('select')
        if selected is not None:
            fields = tuple(part.strip() for part in selected.split(','))
            if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
                raise ValueError('select must contain a unique nonempty column list')
            if '*' not in fields and self.id_field not in fields:
                raise ValueError('select must retain the stable ID field')
        return self


class IedbApiCaptureSpec(StrictModel):
    capture_id: str = Field(pattern=_CAPTURE_ID_PATTERN)
    source_mode: Literal['api'] = 'api'
    retrieved_at: datetime
    metrics_url: str = Field(min_length=1)
    metrics_before_relative_path: str = Field(min_length=1)
    metrics_after_relative_path: str = Field(min_length=1)
    expected_table_names: tuple[str, ...] = Field(min_length=1)
    pages: tuple[IedbApiPageSpec, ...] = Field(min_length=1)

    @field_validator('retrieved_at')
    @classmethod
    def validate_retrieved_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'retrieved_at')

    @field_validator('metrics_before_relative_path', 'metrics_after_relative_path')
    @classmethod
    def validate_metrics_path(cls, value: str) -> str:
        return _validate_relative_path(value, 'API metrics path')

    @field_validator('metrics_url')
    @classmethod
    def validate_metrics_url(cls, value: str) -> str:
        pairs = _validate_official_api_url(value, expected_path='/api_metrics')
        if dict(pairs) != {'order': 'search_table_name'}:
            raise ValueError('api_metrics must use the stable order=search_table_name query')
        return value

    @field_validator('expected_table_names')
    @classmethod
    def validate_expected_table_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(re.fullmatch(_TABLE_PATTERN, name) is None for name in value):
            raise ValueError('expected_table_names contains an invalid table name')
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('expected_table_names must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_paths(self) -> Self:
        if self.metrics_before_relative_path == self.metrics_after_relative_path:
            raise ValueError('metrics-before and metrics-after must be distinct artifacts')
        paths = [self.metrics_before_relative_path, self.metrics_after_relative_path]
        paths.extend(page.data_relative_path for page in self.pages)
        paths.extend(page.receipt_relative_path for page in self.pages)
        if len(paths) != len(set(paths)):
            raise ValueError('each API artifact path must be unique')
        table_keys = {(page.table_name, page.id_field, page.request_url) for page in self.pages}
        table_names = {page.table_name for page in self.pages}
        if len(table_keys) != len(table_names):
            raise ValueError('all pages for a table must use the exact same ID field and query')
        if table_names != set(self.expected_table_names):
            missing = sorted(set(self.expected_table_names) - table_names)
            unexpected = sorted(table_names - set(self.expected_table_names))
            raise ValueError(
                f'API pages must cover exactly expected_table_names; missing={missing!r}, unexpected={unexpected!r}'
            )
        return self


class IedbFullExportIdentity(StrictModel):
    """Caller-asserted identity for one purported IEDB full-export artifact.

    A date-bearing release ID plus the exact artifact digest and size creates an immutable
    local identity.  It does not authenticate the artifact or prove the asserted build time.
    """

    release_id: str = Field(pattern=_CAPTURE_ID_PATTERN)
    source_build_at: datetime
    source_url: str = Field(min_length=1)
    artifact_relative_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_byte_count: int = Field(gt=0)
    artifact_format: Literal['raw', 'zip']
    expected_member_count: int = Field(ge=0)
    identity_basis: Literal['caller-asserted-release-date-and-content-sha256'] = (
        'caller-asserted-release-date-and-content-sha256'
    )

    @field_validator('source_build_at')
    @classmethod
    def validate_source_build_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'source_build_at')

    @field_validator('artifact_relative_path')
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        return _validate_relative_path(value, 'full-export artifact path')

    @field_validator('source_url')
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != 'https'
            or parsed.hostname not in _OFFICIAL_EXPORT_HOSTS
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.path
            or parsed.fragment
        ):
            raise ValueError('full exports must bind an exact HTTPS URL on an official IEDB host')
        _parse_strict_query(parsed.query, 'full-export source URL')
        return value

    @model_validator(mode='after')
    def validate_identity(self) -> Self:
        release_id = self.release_id.lower()
        if any(token in release_id for token in ('latest', 'current', 'rolling')):
            raise ValueError('full-export release_id cannot use a mutable alias')
        date = self.source_build_at.date()
        if date.isoformat() not in self.release_id and date.strftime('%Y%m%d') not in self.release_id:
            raise ValueError('full-export release_id must contain its source build date')
        if self.artifact_format == 'raw' and self.expected_member_count != 0:
            raise ValueError('raw full exports cannot declare archive members')
        if self.artifact_format == 'zip' and self.expected_member_count == 0:
            raise ValueError('ZIP full exports must declare at least one expected member')
        return self


class IedbFullExportCaptureSpec(StrictModel):
    capture_id: str = Field(pattern=_CAPTURE_ID_PATTERN)
    source_mode: Literal['full_export'] = 'full_export'
    retrieved_at: datetime
    identity: IedbFullExportIdentity

    @field_validator('retrieved_at')
    @classmethod
    def validate_retrieved_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'retrieved_at')

    @model_validator(mode='after')
    def validate_capture(self) -> Self:
        if self.retrieved_at < self.identity.source_build_at:
            raise ValueError('retrieved_at cannot precede the full-export source build')
        return self


class IedbRawInventoryEntry(StrictModel):
    kind: Literal['file', 'zip_member']
    relative_path: str = Field(min_length=1)
    container_relative_path: str | None = None
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)

    @field_validator('relative_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_path(value, 'inventory path')

    @field_validator('container_relative_path')
    @classmethod
    def validate_container_path(cls, value: str | None) -> str | None:
        if value is not None:
            return _validate_relative_path(value, 'inventory container path')
        return value

    @model_validator(mode='after')
    def validate_kind(self) -> Self:
        if self.kind == 'file' and self.container_relative_path is not None:
            raise ValueError('regular file inventory entries cannot declare a container')
        if self.kind == 'zip_member' and self.container_relative_path is None:
            raise ValueError('ZIP member inventory entries require a container')
        return self


class IedbApiPageCommitment(StrictModel):
    data_relative_path: str
    data_sha256: str = Field(pattern=_SHA256_PATTERN)
    data_byte_count: int = Field(ge=0)
    receipt_relative_path: str
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    receipt_byte_count: int = Field(gt=0)
    range_start: int = Field(ge=0)
    range_end: int = Field(ge=-1)
    total_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    first_id: int | None = Field(default=None, ge=0)
    last_id: int | None = Field(default=None, ge=0)
    ids_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        if self.row_count == 0:
            if self.range_start != 0 or self.range_end != -1 or self.total_count != 0:
                raise ValueError('empty API pages must use the canonical 0,-1,0 range')
            if self.first_id is not None or self.last_id is not None:
                raise ValueError('empty API pages cannot have boundary IDs')
        elif (
            self.range_end < self.range_start
            or self.range_end - self.range_start + 1 != self.row_count
            or self.first_id is None
            or self.last_id is None
        ):
            raise ValueError('API page range and ID boundaries must match its rows')
        return self


class IedbApiTableCommitment(StrictModel):
    table_name: str = Field(pattern=_TABLE_PATTERN)
    id_field: str = Field(pattern=_FIELD_PATTERN)
    request_url: str = Field(min_length=1)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_build_at: datetime
    record_count: int = Field(ge=0)
    page_count: int = Field(ge=1)
    ids_sha256: str = Field(pattern=_SHA256_PATTERN)
    pages: tuple[IedbApiPageCommitment, ...] = Field(min_length=1)

    @field_validator('source_build_at')
    @classmethod
    def validate_source_build_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'source_build_at')

    @model_validator(mode='after')
    def validate_page_count(self) -> Self:
        if self.page_count != len(self.pages):
            raise ValueError('page_count does not match page commitments')
        if sum(page.row_count for page in self.pages) != self.record_count:
            raise ValueError('table record_count does not match page commitments')
        return self


class IedbApiCaptureBinding(StrictModel):
    source_mode: Literal['api'] = 'api'
    source_base_url: Literal['https://query-api.iedb.org'] = _OFFICIAL_API_ORIGIN
    metrics_url: str
    metrics_before_relative_path: str
    metrics_before_sha256: str = Field(pattern=_SHA256_PATTERN)
    metrics_after_relative_path: str
    metrics_after_sha256: str = Field(pattern=_SHA256_PATTERN)
    metrics_normalized_sha256: str = Field(pattern=_SHA256_PATTERN)
    completeness_scope: Literal['declared-api-table-set'] = 'declared-api-table-set'
    expected_table_names: tuple[str, ...] = Field(min_length=1)
    complete_enumeration_within_scope: Literal[True] = True
    tables: tuple[IedbApiTableCommitment, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_scope(self) -> Self:
        table_names = tuple(table.table_name for table in self.tables)
        if self.expected_table_names != tuple(sorted(self.expected_table_names)):
            raise ValueError('expected_table_names must use canonical sorted order')
        if len(self.expected_table_names) != len(set(self.expected_table_names)):
            raise ValueError('expected_table_names must be unique')
        if table_names != self.expected_table_names:
            raise ValueError('API table commitments must exactly match the declared completeness scope')
        return self


class IedbFullExportCaptureBinding(StrictModel):
    source_mode: Literal['full_export'] = 'full_export'
    identity: IedbFullExportIdentity
    completeness_scope: Literal['exact-supplied-artifact'] = 'exact-supplied-artifact'
    complete_inventory_within_scope: Literal[True] = True
    source_identity_verification: Literal['caller-asserted-unverified'] = 'caller-asserted-unverified'
    member_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)


class IedbLiveCaptureManifest(StrictModel):
    schema_version: Literal['vaxreplay.iedb-live-capture.v0.2'] = IEDB_LIVE_CAPTURE_SCHEMA_VERSION
    capture_id: str = Field(pattern=_CAPTURE_ID_PATTERN)
    source_build_at: datetime
    retrieved_at: datetime
    source_authenticity_verified: Literal[False] = False
    tier_a_eligible: Literal[False] = False
    external_timestamp_required: Literal[True] = True
    inventory: tuple[IedbRawInventoryEntry, ...] = Field(min_length=1)
    inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_binding: IedbApiCaptureBinding | IedbFullExportCaptureBinding

    @field_validator('source_build_at', 'retrieved_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'capture timestamp')

    @model_validator(mode='after')
    def validate_manifest(self) -> Self:
        if self.retrieved_at < self.source_build_at:
            raise ValueError('retrieved_at cannot precede source_build_at')
        keys = tuple((entry.container_relative_path or '', entry.kind, entry.relative_path) for entry in self.inventory)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('inventory entries must have unique canonical ordering')
        expected = hashlib.sha256(
            canonical_json_bytes([entry.model_dump(mode='json') for entry in self.inventory])
        ).hexdigest()
        if self.inventory_sha256 != expected:
            raise ValueError('inventory_sha256 does not bind the canonical raw inventory')
        files = {entry.relative_path: entry for entry in self.inventory if entry.kind == 'file'}
        members = tuple(entry for entry in self.inventory if entry.kind == 'zip_member')
        binding = self.source_binding
        if isinstance(binding, IedbApiCaptureBinding):
            if members:
                raise ValueError('API captures cannot contain ZIP member inventory entries')
            if any(table.source_build_at != self.source_build_at for table in binding.tables):
                raise ValueError('API table build timestamps must match the manifest source_build_at')
            if any(
                table.request_sha256 != hashlib.sha256(table.request_url.encode('utf-8')).hexdigest()
                for table in binding.tables
            ):
                raise ValueError('API request_sha256 must bind the exact table request URL')
            bound_page_paths = tuple(
                path
                for table in binding.tables
                for page in table.pages
                for path in (page.data_relative_path, page.receipt_relative_path)
            )
            if len(bound_page_paths) != len(set(bound_page_paths)):
                raise ValueError('API page data and receipt paths must be globally unique')
            expected_paths = {
                binding.metrics_before_relative_path,
                binding.metrics_after_relative_path,
            }
            expected_paths.update(bound_page_paths)
            if set(files) != expected_paths:
                raise ValueError('API source binding paths must exactly match the raw file inventory')
            if files[binding.metrics_before_relative_path].sha256 != binding.metrics_before_sha256:
                raise ValueError('metrics-before commitment does not match the raw file inventory')
            if files[binding.metrics_after_relative_path].sha256 != binding.metrics_after_sha256:
                raise ValueError('metrics-after commitment does not match the raw file inventory')
            for table in binding.tables:
                for page in table.pages:
                    data = files[page.data_relative_path]
                    receipt = files[page.receipt_relative_path]
                    if (data.sha256, data.byte_count) != (page.data_sha256, page.data_byte_count):
                        raise ValueError('API page data commitment does not match the raw file inventory')
                    if (receipt.sha256, receipt.byte_count) != (
                        page.receipt_sha256,
                        page.receipt_byte_count,
                    ):
                        raise ValueError('API page receipt commitment does not match the raw file inventory')
        else:
            identity = binding.identity
            if identity.source_build_at != self.source_build_at:
                raise ValueError('full-export identity build time must match manifest source_build_at')
            if set(files) != {identity.artifact_relative_path}:
                raise ValueError('full-export binding must identify the only raw file in its inventory')
            artifact = files[identity.artifact_relative_path]
            if (artifact.sha256, artifact.byte_count) != (
                identity.artifact_sha256,
                identity.artifact_byte_count,
            ):
                raise ValueError('full-export identity does not match the raw file inventory')
            if any(member.container_relative_path != identity.artifact_relative_path for member in members):
                raise ValueError('full-export ZIP members must be bound to the identified artifact')
            if identity.artifact_format == 'raw' and members:
                raise ValueError('raw full exports cannot contain ZIP member inventory entries')
            if len(members) != identity.expected_member_count:
                raise ValueError('full-export member inventory does not match the asserted member count')
            member_digest = hashlib.sha256(
                canonical_json_bytes([entry.model_dump(mode='json') for entry in members])
            ).hexdigest()
            if member_digest != binding.member_inventory_sha256:
                raise ValueError('member_inventory_sha256 does not bind the ZIP member inventory')
        return self


@dataclass(frozen=True)
class BuiltIedbCapture:
    root: Path
    manifest: IedbLiveCaptureManifest
    manifest_bytes: bytes
    manifest_sha256: str

    def as_source_capture_artifact(self, *, witnessed_at: datetime):
        """Convert this raw-byte-derived commitment to a source-capture input.

        Do not construct :class:`BuiltIedbCapture` from an independently parsed manifest.
        Use :func:`verify_capture_manifest` to rebuild and compare untrusted manifest bytes.
        ``witnessed_at`` must come from a separately verified external checkpoint proof.
        """

        from vaxreplay.prospective import SourceCaptureArtifact

        return SourceCaptureArtifact(
            source_id=f'iedb:{self.manifest.capture_id}',
            source_release_at=self.manifest.source_build_at,
            captured_at=self.manifest.retrieved_at,
            witnessed_at=witnessed_at,
            manifest_bytes=self.manifest_bytes,
        )


@dataclass(frozen=True)
class _PhysicalFile:
    relative_path: str
    path: Path
    sha256: str
    byte_count: int
    device: int
    inode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _Page:
    spec: IedbApiPageSpec
    receipt: IedbApiExchangeReceipt
    rows: tuple[dict[str, Any], ...]
    ids: tuple[int, ...]
    start: int
    end: int
    total: int


def build_api_capture(root: Path, spec: IedbApiCaptureSpec) -> BuiltIedbCapture:
    """Validate stable, complete enumeration of the declared IQ-API table set."""

    page_table_names = {page.table_name for page in spec.pages}
    if page_table_names != set(spec.expected_table_names):
        raise IedbLiveCaptureError('API pages do not cover exactly the declared table set')
    resolved_root, files = _scan_capture_root(root)
    expected_paths = {
        spec.metrics_before_relative_path,
        spec.metrics_after_relative_path,
        *(page.data_relative_path for page in spec.pages),
        *(page.receipt_relative_path for page in spec.pages),
    }
    _require_exact_files(files, expected_paths)

    metrics_before_bytes = _read_bounded(files[spec.metrics_before_relative_path], _MAX_METADATA_BYTES)
    metrics_after_bytes = _read_bounded(files[spec.metrics_after_relative_path], _MAX_METADATA_BYTES)
    metrics_before = _load_metrics(metrics_before_bytes, spec.metrics_before_relative_path)
    metrics_after = _load_metrics(metrics_after_bytes, spec.metrics_after_relative_path)
    if metrics_before != metrics_after:
        raise IedbLiveCaptureError('api_metrics changed while the API tables were captured')
    metric_by_table = {metric.search_table_name: metric for metric in metrics_before}
    missing_metrics = sorted(set(spec.expected_table_names) - set(metric_by_table))
    if missing_metrics:
        raise IedbLiveCaptureError(f'api_metrics is missing declared tables: {missing_metrics!r}')

    pages_by_table: dict[str, list[_Page]] = defaultdict(list)
    for page_spec in spec.pages:
        data_file = files[page_spec.data_relative_path]
        data_bytes = _read_bounded(data_file, _MAX_API_PAGE_BYTES)
        receipt_bytes = _read_bounded(files[page_spec.receipt_relative_path], _MAX_METADATA_BYTES)
        receipt = _load_receipt(receipt_bytes, page_spec.receipt_relative_path)
        if receipt.request_url != page_spec.request_url:
            raise IedbLiveCaptureError(f'{page_spec.receipt_relative_path} request URL does not match its page spec')
        if receipt.response_header('content-encoding') not in (None, 'identity'):
            raise IedbLiveCaptureError('API page artifacts must record decoded identity bytes, not ambiguous encoding')
        start, end, total = _parse_content_range(receipt.response_header('content-range'))
        request_start, request_end = _parse_request_range(receipt.request_header('range'))
        if start == 0 and end == -1:
            if request_start != 0:
                raise IedbLiveCaptureError('an empty enumeration must begin with range zero')
        elif start != request_start or end > request_end:
            raise IedbLiveCaptureError('response Content-Range is inconsistent with the requested Range')
        if receipt.request_header('range-unit') not in (None, 'items'):
            raise IedbLiveCaptureError('IEDB API pagination must use the items range unit')
        rows = _load_rows(data_bytes, page_spec.data_format, page_spec.data_relative_path)
        expected_rows = 0 if end == -1 else end - start + 1
        if len(rows) != expected_rows:
            raise IedbLiveCaptureError(f'{page_spec.data_relative_path} row count does not match Content-Range')
        ids = _extract_ids(rows, page_spec.id_field, page_spec.data_relative_path)
        pages_by_table[page_spec.table_name].append(_Page(page_spec, receipt, rows, ids, start, end, total))

    table_commitments: list[IedbApiTableCommitment] = []
    build_times: set[datetime] = set()
    for table_name in sorted(pages_by_table):
        pages = sorted(pages_by_table[table_name], key=lambda page: (page.start, page.spec.data_relative_path))
        metric = metric_by_table.get(table_name)
        if metric is None:
            raise IedbLiveCaptureError(f'api_metrics does not contain captured table {table_name}')
        build_at = _metric_timestamp(metric)
        build_times.add(build_at)
        _validate_complete_pages(table_name, pages, metric.record_count)
        all_ids = tuple(identifier for page in pages for identifier in page.ids)
        if len(all_ids) != len(set(all_ids)):
            raise IedbLiveCaptureError(f'{table_name} contains duplicate IDs')
        if any(left >= right for left, right in zip(all_ids, all_ids[1:], strict=False)):
            raise IedbLiveCaptureError(f'{table_name} IDs are not in unique strictly ascending order')
        page_commitments = tuple(_page_commitment(page, files) for page in pages)
        first = pages[0].spec
        table_commitments.append(
            IedbApiTableCommitment(
                table_name=table_name,
                id_field=first.id_field,
                request_url=first.request_url,
                request_sha256=hashlib.sha256(first.request_url.encode('utf-8')).hexdigest(),
                source_build_at=build_at,
                record_count=len(all_ids),
                page_count=len(pages),
                ids_sha256=_ids_sha256(all_ids),
                pages=page_commitments,
            )
        )
    if len(build_times) != 1:
        raise IedbLiveCaptureError('captured API tables have mixed source build timestamps')
    source_build_at = next(iter(build_times))
    if spec.retrieved_at < source_build_at:
        raise IedbLiveCaptureError('retrieved_at cannot precede the API source build')

    inventory = tuple(_file_inventory(file) for file in sorted(files.values(), key=lambda item: item.relative_path))
    binding = IedbApiCaptureBinding(
        metrics_url=spec.metrics_url,
        metrics_before_relative_path=spec.metrics_before_relative_path,
        metrics_before_sha256=files[spec.metrics_before_relative_path].sha256,
        metrics_after_relative_path=spec.metrics_after_relative_path,
        metrics_after_sha256=files[spec.metrics_after_relative_path].sha256,
        metrics_normalized_sha256=hashlib.sha256(
            canonical_json_bytes([metric.model_dump(mode='json') for metric in metrics_before])
        ).hexdigest(),
        expected_table_names=spec.expected_table_names,
        tables=tuple(table_commitments),
    )
    return _built_capture(
        resolved_root,
        capture_id=spec.capture_id,
        source_build_at=source_build_at,
        retrieved_at=spec.retrieved_at,
        inventory=inventory,
        binding=binding,
    )


def build_full_export_capture(root: Path, spec: IedbFullExportCaptureSpec) -> BuiltIedbCapture:
    """Validate and inventory one content-addressed, caller-identified IEDB export."""

    resolved_root, files = _scan_capture_root(root)
    identity = spec.identity
    _require_exact_files(files, {identity.artifact_relative_path})
    artifact = files[identity.artifact_relative_path]
    if artifact.sha256 != identity.artifact_sha256:
        raise IedbLiveCaptureError('full-export artifact SHA-256 does not match its immutable identity')
    if artifact.byte_count != identity.artifact_byte_count:
        raise IedbLiveCaptureError('full-export artifact byte count does not match its immutable identity')

    member_entries: tuple[IedbRawInventoryEntry, ...] = ()
    if identity.artifact_format == 'zip':
        member_entries = _inventory_zip(artifact)
    if len(member_entries) != identity.expected_member_count:
        raise IedbLiveCaptureError('full-export archive member count does not match its immutable identity')

    file_entry = _file_inventory(artifact)
    inventory = tuple(
        sorted(
            (file_entry, *member_entries),
            key=lambda entry: (entry.container_relative_path or '', entry.kind, entry.relative_path),
        )
    )
    binding = IedbFullExportCaptureBinding(
        identity=identity,
        member_inventory_sha256=hashlib.sha256(
            canonical_json_bytes([entry.model_dump(mode='json') for entry in member_entries])
        ).hexdigest(),
    )
    return _built_capture(
        resolved_root,
        capture_id=spec.capture_id,
        source_build_at=identity.source_build_at,
        retrieved_at=spec.retrieved_at,
        inventory=inventory,
        binding=binding,
    )


def verify_capture_manifest(
    root: Path,
    spec: IedbApiCaptureSpec | IedbFullExportCaptureSpec,
    manifest_bytes: bytes,
) -> BuiltIedbCapture:
    """Rebuild from exact raw bytes and require equality with an untrusted manifest.

    Pydantic validation alone proves only structural consistency.  This function is the
    trusted loading boundary for a stored manifest: it rejects noncanonical bytes and any
    claim that cannot be reproduced from the supplied raw capture and committed spec.
    Source authenticity and time still require independent evidence.
    """

    try:
        manifest = IedbLiveCaptureManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise IedbLiveCaptureError(f'capture manifest is invalid: {error}') from error
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise IedbLiveCaptureError('capture manifest must use exact canonical JSON bytes')
    rebuilt = (
        build_api_capture(root, spec) if isinstance(spec, IedbApiCaptureSpec) else build_full_export_capture(root, spec)
    )
    if rebuilt.manifest_bytes != manifest_bytes:
        raise IedbLiveCaptureError('capture manifest claims do not reproduce from the exact raw capture and spec')
    return rebuilt


def write_capture_manifest(capture: BuiltIedbCapture, output_path: Path) -> Path:
    """Write exact canonical manifest bytes outside the raw capture root, without overwriting."""

    requested = output_path.expanduser().absolute()
    if os.path.lexists(requested):
        raise IedbLiveCaptureError(f'capture manifest output already exists: {requested}')
    resolved_parent = requested.parent.resolve()
    if resolved_parent == capture.root or capture.root in resolved_parent.parents:
        raise IedbLiveCaptureError('capture manifest must be written outside the immutable raw capture root')
    requested.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{requested.name}.', dir=requested.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as destination:
            destination.write(capture.manifest_bytes)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, requested)
        except FileExistsError as error:
            raise IedbLiveCaptureError(f'capture manifest output already exists: {requested}') from error
        temporary.unlink()
        directory_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
        directory_descriptor = os.open(requested.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return requested


def _built_capture(
    root: Path,
    *,
    capture_id: str,
    source_build_at: datetime,
    retrieved_at: datetime,
    inventory: tuple[IedbRawInventoryEntry, ...],
    binding: IedbApiCaptureBinding | IedbFullExportCaptureBinding,
) -> BuiltIedbCapture:
    inventory_sha256 = hashlib.sha256(
        canonical_json_bytes([entry.model_dump(mode='json') for entry in inventory])
    ).hexdigest()
    manifest = IedbLiveCaptureManifest(
        capture_id=capture_id,
        source_build_at=source_build_at,
        retrieved_at=retrieved_at,
        inventory=inventory,
        inventory_sha256=inventory_sha256,
        source_binding=binding,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    return BuiltIedbCapture(
        root=root,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _scan_capture_root(root: Path) -> tuple[Path, dict[str, _PhysicalFile]]:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise IedbLiveCaptureError('raw capture root cannot be a symbolic link')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise IedbLiveCaptureError(f'raw capture root does not exist: {resolved}')
    files: dict[str, _PhysicalFile] = {}
    portable_paths: dict[str, str] = {}

    def fail_walk(error: OSError) -> None:
        location = error.filename or str(resolved)
        raise IedbLiveCaptureError(f'cannot traverse raw capture at {location}: {error.strerror or error}') from error

    try:
        for current, directory_names, file_names in os.walk(
            resolved,
            topdown=True,
            onerror=fail_walk,
            followlinks=False,
        ):
            current_path = Path(current)
            for name in sorted(directory_names):
                path = current_path / name
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise IedbLiveCaptureError(f'raw capture contains a symbolic link: {path.relative_to(resolved)}')
                if not stat.S_ISDIR(mode):
                    raise IedbLiveCaptureError(
                        f'raw capture contains a non-directory node: {path.relative_to(resolved)}'
                    )
                _portable_path(path.relative_to(resolved).as_posix(), portable_paths)
            for name in sorted(file_names):
                path = current_path / name
                relative_path = path.relative_to(resolved).as_posix()
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise IedbLiveCaptureError(f'raw capture contains a symbolic link: {relative_path}')
                if not stat.S_ISREG(mode):
                    raise IedbLiveCaptureError(f'raw capture contains a non-regular file: {relative_path}')
                _validate_relative_path(relative_path, 'raw capture path')
                _portable_path(relative_path, portable_paths)
                sha256, byte_count, device, inode, modified_ns, changed_ns = _hash_regular_file(path)
                files[relative_path] = _PhysicalFile(
                    relative_path,
                    path,
                    sha256,
                    byte_count,
                    device,
                    inode,
                    modified_ns,
                    changed_ns,
                )
                if len(files) > _MAX_INVENTORY_FILES:
                    raise IedbLiveCaptureError('raw capture contains too many files')
    except IedbLiveCaptureError:
        raise
    except OSError as error:
        location = error.filename or str(resolved)
        raise IedbLiveCaptureError(f'cannot inspect raw capture at {location}: {error.strerror or error}') from error
    if not files:
        raise IedbLiveCaptureError('raw capture cannot be empty')
    return resolved, files


def _portable_path(value: str, seen: dict[str, str]) -> None:
    normalized = unicodedata.normalize('NFC', value).casefold()
    previous = seen.setdefault(normalized, value)
    if previous != value:
        raise IedbLiveCaptureError(f'capture paths are ambiguous across filesystems: {previous!r} and {value!r}')


def _hash_regular_file(path: Path) -> tuple[str, int, int, int, int, int]:
    digest = hashlib.sha256()
    byte_count = 0
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, 'rb') as source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise IedbLiveCaptureError(f'capture artifact is not a regular file: {path}')
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            finished = os.fstat(source.fileno())
            if _stat_identity(opened) != _stat_identity(finished):
                raise IedbLiveCaptureError(f'capture artifact changed while hashing: {path}')
    except OSError as error:
        raise IedbLiveCaptureError(f'cannot read capture artifact {path}: {error}') from error
    if byte_count != finished.st_size:
        raise IedbLiveCaptureError(f'capture artifact changed while hashing: {path}')
    return (
        digest.hexdigest(),
        byte_count,
        finished.st_dev,
        finished.st_ino,
        finished.st_mtime_ns,
        finished.st_ctime_ns,
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_bounded(file: _PhysicalFile, maximum_bytes: int) -> bytes:
    if file.byte_count > maximum_bytes:
        raise IedbLiveCaptureError(f'capture artifact exceeds safe parse limit: {file.relative_path}')
    try:
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(file.path, flags)
        with os.fdopen(descriptor, 'rb') as source:
            payload = source.read(maximum_bytes + 1)
    except OSError as error:
        raise IedbLiveCaptureError(f'cannot read capture artifact {file.relative_path}: {error}') from error
    if len(payload) != file.byte_count or hashlib.sha256(payload).hexdigest() != file.sha256:
        raise IedbLiveCaptureError(f'capture artifact changed during normalization: {file.relative_path}')
    return payload


def _require_exact_files(files: dict[str, _PhysicalFile], expected_paths: set[str]) -> None:
    actual = set(files)
    missing = expected_paths - actual
    unexpected = actual - expected_paths
    if missing:
        raise IedbLiveCaptureError(f'raw capture is missing declared files: {sorted(missing)}')
    if unexpected:
        raise IedbLiveCaptureError(f'raw capture contains undeclared files: {sorted(unexpected)}')


def _load_unique_json(payload: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise IedbLiveCaptureError(f'{label} contains duplicate JSON key {key!r}')
            result[key] = value
        return result

    try:
        text = payload.decode('utf-8')
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IedbLiveCaptureError(f'{label} is not valid strict UTF-8 JSON: {error}') from error


def _reject_json_constant(value: str) -> None:
    raise IedbLiveCaptureError(f'non-finite JSON number {value} is not allowed')


def _load_metrics(payload: bytes, label: str) -> tuple[IedbApiMetric, ...]:
    value = _load_unique_json(payload, label)
    if not isinstance(value, list):
        raise IedbLiveCaptureError(f'{label} must contain a JSON array')
    metrics: list[IedbApiMetric] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise IedbLiveCaptureError(f'{label} metric {index} must be an object')
        try:
            metrics.append(IedbApiMetric.model_validate(raw))
        except ValueError as error:
            raise IedbLiveCaptureError(f'{label} metric {index} is invalid: {error}') from error
    names = tuple(metric.search_table_name for metric in metrics)
    if len(names) != len(set(names)):
        raise IedbLiveCaptureError(f'{label} contains duplicate table IDs')
    return tuple(sorted(metrics, key=lambda metric: metric.search_table_name))


def parse_iedb_api_metrics(
    payload: bytes,
    *,
    label: str = 'IEDB api_metrics',
) -> tuple[IedbApiMetric, ...]:
    """Strictly parse one exact IQ-API metrics body for pre-capture planning.

    Planning callers may use the counts to compile a finite Range-request plan.  The
    returned metrics are not source evidence: the official capture must fetch metrics
    again before and after every planned page and pass :func:`build_api_capture`.
    """

    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_METADATA_BYTES:
        raise IedbLiveCaptureError('IEDB api_metrics must be nonempty bounded bytes')
    return _load_metrics(payload, label)


def _metric_timestamp(metric: IedbApiMetric) -> datetime:
    value = datetime.fromisoformat(metric.creation_date.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _load_receipt(payload: bytes, label: str) -> IedbApiExchangeReceipt:
    value = _load_unique_json(payload, label)
    try:
        # Strict models intentionally reject a Python list for a tuple field.  Validate the
        # duplicate-key-free value through JSON again so JSON arrays receive the documented
        # tuple interpretation without weakening strict scalar validation.
        return IedbApiExchangeReceipt.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise IedbLiveCaptureError(f'{label} is not a valid API exchange receipt: {error}') from error


def _load_rows(payload: bytes, data_format: str, label: str) -> tuple[dict[str, Any], ...]:
    if data_format == 'json':
        value = _load_unique_json(payload, label)
        if not isinstance(value, list):
            raise IedbLiveCaptureError(f'{label} must contain a JSON array')
        raw_rows = value
    elif data_format == 'jsonl':
        if payload and not payload.endswith(b'\n'):
            raise IedbLiveCaptureError(f'{label} JSONL must end with a newline')
        raw_rows = [
            _load_unique_json(line, f'{label} line {index}')
            for index, line in enumerate(payload.splitlines(), start=1)
            if line
        ]
    else:
        try:
            text = payload.decode('utf-8-sig')
            source = io.StringIO(text, newline='')
            reader = csv.DictReader(source, strict=True)
            if reader.fieldnames is None or any(not field for field in reader.fieldnames):
                raise IedbLiveCaptureError(f'{label} requires a nonempty CSV header')
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise IedbLiveCaptureError(f'{label} contains duplicate CSV columns')
            raw_rows = list(reader)
        except (UnicodeDecodeError, csv.Error) as error:
            raise IedbLiveCaptureError(f'{label} is not valid UTF-8 CSV: {error}') from error
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict) or not all(isinstance(key, str) for key in row):
            raise IedbLiveCaptureError(f'{label} row {index} must be an object with string keys')
        rows.append(row)
    return tuple(rows)


def _extract_ids(rows: tuple[dict[str, Any], ...], id_field: str, label: str) -> tuple[int, ...]:
    identifiers: list[int] = []
    for index, row in enumerate(rows):
        value = row.get(id_field)
        if isinstance(value, bool):
            raise IedbLiveCaptureError(f'{label} row {index} has a boolean ID')
        if isinstance(value, int):
            identifier = value
        elif isinstance(value, str) and re.fullmatch(r'(?:0|[1-9][0-9]*)', value):
            identifier = int(value)
        else:
            raise IedbLiveCaptureError(f'{label} row {index} lacks an unambiguous nonnegative integer ID')
        if identifier < 0:
            raise IedbLiveCaptureError(f'{label} row {index} has a negative ID')
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise IedbLiveCaptureError(f'{label} contains duplicate IDs')
    if any(left >= right for left, right in zip(identifiers, identifiers[1:], strict=False)):
        raise IedbLiveCaptureError(f'{label} is not in unique strictly ascending ID order')
    return tuple(identifiers)


def _parse_content_range(value: str | None) -> tuple[int, int, int]:
    if value is None:
        raise IedbLiveCaptureError('API response receipt is missing Content-Range')
    if re.fullmatch(r'(?:items )?\*/0', value):
        return 0, -1, 0
    match = re.fullmatch(r'(?:items )?([0-9]+)-([0-9]+)/([0-9]+)', value)
    if match is None:
        raise IedbLiveCaptureError('API Content-Range must contain an exact finite total')
    start, end, total = (int(part) for part in match.groups())
    if end < start or total <= end:
        raise IedbLiveCaptureError('API Content-Range bounds are invalid')
    return start, end, total


def _parse_request_range(value: str | None) -> tuple[int, int]:
    if value is None:
        raise IedbLiveCaptureError('API request receipt is missing Range')
    match = re.fullmatch(r'(?:items=)?([0-9]+)-([0-9]+)', value)
    if match is None:
        raise IedbLiveCaptureError('API request Range must use exact finite bounds')
    start, end = (int(part) for part in match.groups())
    if end < start:
        raise IedbLiveCaptureError('API request Range bounds are invalid')
    return start, end


def _validate_complete_pages(table_name: str, pages: list[_Page], metric_count: int) -> None:
    totals = {page.total for page in pages}
    if len(totals) != 1:
        raise IedbLiveCaptureError(f'{table_name} pages disagree on total count')
    total = next(iter(totals))
    if total != metric_count:
        raise IedbLiveCaptureError(f'{table_name} Content-Range total does not match api_metrics count')
    if total == 0:
        if len(pages) != 1 or pages[0].start != 0 or pages[0].end != -1:
            raise IedbLiveCaptureError(f'{table_name} empty enumeration must use one canonical page')
        return
    expected_start = 0
    for page in pages:
        if page.start != expected_start:
            raise IedbLiveCaptureError(f'{table_name} pages have a gap, overlap, or duplicate range')
        expected_start = page.end + 1
    if expected_start != total:
        raise IedbLiveCaptureError(f'{table_name} pages do not enumerate the exact complete result')


def _page_commitment(page: _Page, files: dict[str, _PhysicalFile]) -> IedbApiPageCommitment:
    ids = page.ids
    data_file = files[page.spec.data_relative_path]
    receipt_file = files[page.spec.receipt_relative_path]
    return IedbApiPageCommitment(
        data_relative_path=page.spec.data_relative_path,
        data_sha256=data_file.sha256,
        data_byte_count=data_file.byte_count,
        receipt_relative_path=page.spec.receipt_relative_path,
        receipt_sha256=receipt_file.sha256,
        receipt_byte_count=receipt_file.byte_count,
        range_start=page.start,
        range_end=page.end,
        total_count=page.total,
        row_count=len(ids),
        first_id=ids[0] if ids else None,
        last_id=ids[-1] if ids else None,
        ids_sha256=_ids_sha256(ids),
    )


def _ids_sha256(identifiers: tuple[int, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(identifiers))).hexdigest()


def _file_inventory(file: _PhysicalFile) -> IedbRawInventoryEntry:
    return IedbRawInventoryEntry(
        kind='file',
        relative_path=file.relative_path,
        sha256=file.sha256,
        byte_count=file.byte_count,
    )


def _inventory_zip(artifact: _PhysicalFile) -> tuple[IedbRawInventoryEntry, ...]:
    entries: list[IedbRawInventoryEntry] = []
    seen_names: set[str] = set()
    portable_paths: dict[str, str] = {}
    total_unpacked_bytes = 0
    try:
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(artifact.path, flags)
        with os.fdopen(descriptor, 'rb') as raw_archive, tempfile.TemporaryFile(mode='w+b') as staged_archive:
            opened = os.fstat(raw_archive.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise IedbLiveCaptureError('full-export ZIP is not a regular file')
            if _stat_identity(opened) != (
                artifact.device,
                artifact.inode,
                artifact.byte_count,
                artifact.modified_ns,
                artifact.changed_ns,
            ):
                raise IedbLiveCaptureError('full-export ZIP changed between capture scan and inventory')
            archive_digest = hashlib.sha256()
            archive_byte_count = 0
            while chunk := raw_archive.read(1024 * 1024):
                archive_digest.update(chunk)
                archive_byte_count += len(chunk)
                staged_archive.write(chunk)
            finished = os.fstat(raw_archive.fileno())
            if _stat_identity(finished) != _stat_identity(opened):
                raise IedbLiveCaptureError('full-export ZIP changed while staging exact bytes')
            if (archive_digest.hexdigest(), archive_byte_count) != (artifact.sha256, artifact.byte_count):
                raise IedbLiveCaptureError('full-export ZIP bytes changed between capture scan and inventory')

            staged_archive.seek(0)
            with zipfile.ZipFile(staged_archive) as archive:
                infos = archive.infolist()
                if len(infos) > _MAX_ARCHIVE_MEMBERS:
                    raise IedbLiveCaptureError('full-export ZIP contains too many members')
                declared_unpacked_bytes = sum(info.file_size for info in infos if not info.is_dir())
                if declared_unpacked_bytes > _MAX_ARCHIVE_UNPACKED_BYTES:
                    raise IedbLiveCaptureError('full-export ZIP exceeds the aggregate unpacked-byte limit')
                for info in infos:
                    name = info.filename
                    _validate_relative_path(name.rstrip('/'), 'ZIP member path')
                    _portable_path(name.rstrip('/'), portable_paths)
                    if name in seen_names:
                        raise IedbLiveCaptureError(f'full-export ZIP contains duplicate member path {name!r}')
                    seen_names.add(name)
                    unix_mode = info.external_attr >> 16
                    if stat.S_ISLNK(unix_mode):
                        raise IedbLiveCaptureError(f'full-export ZIP contains symbolic link {name!r}')
                    if info.flag_bits & 0x1:
                        raise IedbLiveCaptureError(f'full-export ZIP contains encrypted member {name!r}')
                    if info.is_dir():
                        continue
                    file_type = stat.S_IFMT(unix_mode)
                    if file_type not in (0, stat.S_IFREG):
                        raise IedbLiveCaptureError(f'full-export ZIP contains non-regular member {name!r}')
                    digest = hashlib.sha256()
                    byte_count = 0
                    with archive.open(info, 'r') as source:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            byte_count += len(chunk)
                            total_unpacked_bytes += len(chunk)
                            if total_unpacked_bytes > _MAX_ARCHIVE_UNPACKED_BYTES:
                                raise IedbLiveCaptureError('full-export ZIP exceeds the aggregate unpacked-byte limit')
                    if byte_count != info.file_size:
                        raise IedbLiveCaptureError(f'full-export ZIP member size mismatch for {name!r}')
                    entries.append(
                        IedbRawInventoryEntry(
                            kind='zip_member',
                            relative_path=name,
                            container_relative_path=artifact.relative_path,
                            sha256=digest.hexdigest(),
                            byte_count=byte_count,
                        )
                    )
    except IedbLiveCaptureError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise IedbLiveCaptureError(f'invalid full-export ZIP {artifact.relative_path}: {error}') from error
    if not entries:
        raise IedbLiveCaptureError('full-export ZIP must contain at least one regular member')
    return tuple(sorted(entries, key=lambda entry: entry.relative_path))
