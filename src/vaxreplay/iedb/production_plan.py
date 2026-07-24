"""Deterministic pre-selection planning for complete production IEDB captures.

The IQ-API exposes table counts through ``api_metrics`` and paginates table bodies
with HTTP ``Range`` headers.  This module turns one exact discovery response into a
finite static collection plan and the matching production verifier policy.  The
discovery response is sizing input only: it is never admitted as source evidence.
The resulting plan must be selected and witnessed before execution, and the official
capture fetches ``api_metrics`` again before the first page and after the last page.  Any growth,
shrinkage, mixed build timestamp, missing page, or range discontinuity therefore fails
the existing production verifier closed.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self, TypeAlias

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.iedb.live_capture import IedbApiPageSpec, parse_iedb_api_metrics
from vaxreplay.operations.collector import (
    StaticHttpsArtifactSpec,
    StaticHttpsCollectionPlan,
    static_plan_sha256,
)
from vaxreplay.operations.http_capture import (
    HttpRequestHeader,
    HttpsCaptureReceipt,
    HttpsCaptureRequest,
    HttpsTransport,
    capture_https_to_tempfile,
    prepared_request_headers,
)
from vaxreplay.sources.iedb import (
    IEDB_TIER_A_ANTIGEN_TABLES,
    IedbCapturedPage,
    IedbPromotionLayout,
    IedbSourceVerifierPolicy,
)

IEDB_PRODUCTION_PLAN_COMPILER_POLICY_SCHEMA_VERSION = 'vaxreplay.iedb-production-plan-compiler-policy.v0.1'
IEDB_PRODUCTION_PLAN_COMPILATION_SCHEMA_VERSION = 'vaxreplay.iedb-production-plan-compilation.v0.1'

_METRICS_URL = 'https://query-api.iedb.org/api_metrics?order=search_table_name'
_OFFICIAL_HOST = 'query-api.iedb.org'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_POLICY_BYTES = 8 * 1024 * 1024
_MAX_METRICS_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_LAYOUT_PAGES = 4094
_OUTPUT_FILES = (
    'discovery-api-metrics.json',
    'discovery-api-metrics-receipt.json',
    'plan-compilation.json',
    'source-verifier-policy.json',
    'static-collection-plan.json',
)
IedbTableName: TypeAlias = Literal['bcell_search', 'mhc_search', 'tcell_search']
IedbIdField: TypeAlias = Literal['bcell_id', 'elution_id', 'tcell_id']

_ID_FIELDS: dict[IedbTableName, IedbIdField] = {
    'bcell_search': 'bcell_id',
    'mhc_search': 'elution_id',
    'tcell_search': 'tcell_id',
}


class IedbProductionPlanError(ValueError):
    """Discovery bytes cannot safely compile the committed Tier-A IEDB plan."""


class IedbProductionPlanCompilerPolicy(StrictModel):
    """Exact reviewed inputs to deterministic IEDB Range-plan compilation."""

    schema_version: Literal['vaxreplay.iedb-production-plan-compiler-policy.v0.1'] = (
        IEDB_PRODUCTION_PLAN_COMPILER_POLICY_SCHEMA_VERSION
    )
    compiler_policy_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    plan_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_verifier_policy_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    capture_id_prefix: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]{2,80}$')
    release_evidence_table: Literal['tcell_search'] = 'tcell_search'
    expected_table_names: tuple[Literal['bcell_search', 'mhc_search', 'tcell_search'], ...] = IEDB_TIER_A_ANTIGEN_TABLES
    range_page_size: int = Field(default=10_000, ge=1, le=10_000)
    maximum_total_pages: int = Field(default=_MAX_LAYOUT_PAGES, ge=3, le=_MAX_LAYOUT_PAGES)
    metrics_max_body_bytes: int = Field(
        default=8 * 1024 * 1024,
        ge=1,
        le=_MAX_METRICS_BYTES,
    )
    page_max_body_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1,
        le=2 * 1024 * 1024 * 1024,
    )
    request_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    accepted_tls_versions: tuple[Literal['TLSv1.2', 'TLSv1.3'], ...] = (
        'TLSv1.2',
        'TLSv1.3',
    )
    scope_profile: Literal['tier_a_antigen_all_assay_tables_v1'] = 'tier_a_antigen_all_assay_tables_v1'
    discovery_is_planning_input_only: Literal[True] = True
    official_capture_must_refetch_bracketing_metrics: Literal[True] = True

    @field_validator('expected_table_names', 'accepted_tls_versions')
    @classmethod
    def validate_sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('IEDB compiler tuple fields must be sorted and unique')
        return value

    @model_validator(mode='after')
    def validate_tier_a_scope(self) -> Self:
        if self.expected_table_names != IEDB_TIER_A_ANTIGEN_TABLES:
            raise ValueError('the production compiler requires all three IEDB assay tables')
        return self


class IedbPlannedTable(StrictModel):
    table_name: Literal['bcell_search', 'mhc_search', 'tcell_search']
    id_field: Literal['bcell_id', 'elution_id', 'tcell_id']
    discovery_record_count: int = Field(ge=0)
    page_count: int = Field(ge=1, le=_MAX_LAYOUT_PAGES)

    @model_validator(mode='after')
    def validate_identifier(self) -> Self:
        if self.id_field != _ID_FIELDS[self.table_name]:
            raise ValueError('IEDB planned table uses the wrong canonical identifier')
        return self


class IedbProductionPlanCompilation(StrictModel):
    """Public commitment explaining how discovery counts sized the frozen plan."""

    schema_version: Literal['vaxreplay.iedb-production-plan-compilation.v0.1'] = (
        IEDB_PRODUCTION_PLAN_COMPILATION_SCHEMA_VERSION
    )
    compiler_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    discovery_metrics_sha256: str = Field(pattern=_SHA256_PATTERN)
    discovery_metrics_bytes: int = Field(gt=0, le=_MAX_METRICS_BYTES)
    discovery_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    discovery_receipt_bytes: int = Field(gt=0, le=_MAX_RECEIPT_BYTES)
    discovery_completed_at: datetime
    discovery_source_build_at: datetime
    normalized_selected_metrics_sha256: str = Field(pattern=_SHA256_PATTERN)
    tables: tuple[IedbPlannedTable, ...] = Field(min_length=3, max_length=3)
    total_page_count: int = Field(ge=3, le=_MAX_LAYOUT_PAGES)
    static_collection_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_verifier_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    discovery_source_bytes_admitted_as_official_capture: Literal[False] = False
    plan_requires_pre_capture_selection_registry_commitment: Literal[True] = True
    official_capture_requires_metrics_before_and_after: Literal[True] = True
    official_capture_requires_complete_contiguous_ranges: Literal[True] = True
    tier_a_release_ready: Literal[False] = False

    @field_validator('discovery_completed_at', 'discovery_source_build_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('IEDB planning times must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        names = tuple(item.table_name for item in self.tables)
        if names != IEDB_TIER_A_ANTIGEN_TABLES:
            raise ValueError('IEDB planned tables must use canonical all-assay order')
        if self.total_page_count != sum(item.page_count for item in self.tables):
            raise ValueError('IEDB total page count differs from its table inventory')
        return self


@dataclass(frozen=True, slots=True)
class CompiledIedbProductionPlan:
    policy: IedbProductionPlanCompilerPolicy
    compilation: IedbProductionPlanCompilation
    static_plan: StaticHttpsCollectionPlan
    source_verifier_policy: IedbSourceVerifierPolicy
    discovery_metrics_bytes: bytes
    discovery_receipt_bytes: bytes


def metrics_discovery_request(
    policy: IedbProductionPlanCompilerPolicy,
) -> HttpsCaptureRequest:
    """Return the one exact unauthenticated request used only to size a future plan."""

    return HttpsCaptureRequest(
        url=_METRICS_URL,
        allowed_host=_OFFICIAL_HOST,
        allowed_query_names=('order',),
        request_headers=(HttpRequestHeader(name='accept', value='application/json'),),
        allowed_status_codes=(200,),
        max_body_bytes=policy.metrics_max_body_bytes,
        timeout_seconds=policy.request_timeout_seconds,
    )


def compile_iedb_production_plan(
    *,
    policy_bytes: bytes,
    expected_policy_sha256: str,
    discovery_metrics_bytes: bytes,
    discovery_receipt_bytes: bytes,
) -> CompiledIedbProductionPlan:
    """Compile exact plan/policy bytes from one authenticated-transport discovery receipt."""

    policy = _canonical_policy(policy_bytes, expected_policy_sha256)
    receipt = _canonical_receipt(discovery_receipt_bytes)
    _verify_discovery_receipt(
        receipt,
        body=discovery_metrics_bytes,
        policy=policy,
    )
    try:
        metrics = parse_iedb_api_metrics(discovery_metrics_bytes)
    except ValueError as error:
        raise IedbProductionPlanError('IEDB discovery metrics are invalid') from error
    by_name = {item.search_table_name: item for item in metrics}
    if not set(policy.expected_table_names).issubset(by_name):
        raise IedbProductionPlanError('IEDB discovery metrics omit a required assay table')

    selected = tuple(by_name[name] for name in policy.expected_table_names)
    build_times = {_metric_time(item.creation_date) for item in selected}
    if len(build_times) != 1:
        raise IedbProductionPlanError('IEDB discovery metrics use mixed assay-table build times')
    source_build_at = next(iter(build_times))
    if source_build_at > receipt.completed_at:
        raise IedbProductionPlanError('IEDB discovery source build time postdates its receipt')

    planned_tables: list[IedbPlannedTable] = []
    artifacts: list[StaticHttpsArtifactSpec] = [
        StaticHttpsArtifactSpec(
            artifact_id='a-metrics-before',
            request=metrics_discovery_request(policy),
        )
    ]
    captured_pages: list[IedbCapturedPage] = []
    for table_ordinal, (table_name, metric) in enumerate(zip(policy.expected_table_names, selected, strict=True)):
        id_field = _ID_FIELDS[table_name]
        page_count = max(
            1,
            (metric.record_count + policy.range_page_size - 1) // policy.range_page_size,
        )
        planned_tables.append(
            IedbPlannedTable(
                table_name=table_name,
                id_field=id_field,
                discovery_record_count=metric.record_count,
                page_count=page_count,
            )
        )
        for page_index in range(page_count):
            start = page_index * policy.range_page_size
            end = start + policy.range_page_size - 1
            artifact_id = f'm-{table_ordinal:02d}-{page_index:04d}-{table_name}'
            request_url = f'https://{_OFFICIAL_HOST}/{table_name}?order={id_field}.asc'
            request = HttpsCaptureRequest(
                url=request_url,
                allowed_host=_OFFICIAL_HOST,
                allowed_query_names=('order',),
                request_headers=(
                    HttpRequestHeader(name='accept', value='application/json'),
                    HttpRequestHeader(name='range', value=f'{start}-{end}'),
                    HttpRequestHeader(name='range-unit', value='items'),
                ),
                allowed_status_codes=(200, 206),
                max_body_bytes=policy.page_max_body_bytes,
                timeout_seconds=policy.request_timeout_seconds,
            )
            artifacts.append(StaticHttpsArtifactSpec(artifact_id=artifact_id, request=request))
            captured_pages.append(
                IedbCapturedPage(
                    artifact_id=artifact_id,
                    page=IedbApiPageSpec(
                        table_name=table_name,
                        id_field=id_field,
                        request_url=request_url,
                        data_relative_path=f'pages/{table_name}-{page_index:04d}.json',
                        receipt_relative_path=(f'receipts/{table_name}-{page_index:04d}.json'),
                        data_format='json',
                    ),
                )
            )

    total_pages = sum(item.page_count for item in planned_tables)
    if total_pages > policy.maximum_total_pages:
        raise IedbProductionPlanError('IEDB discovery counts exceed the precommitted maximum page inventory')
    artifacts.append(
        StaticHttpsArtifactSpec(
            artifact_id='z-metrics-after',
            request=metrics_discovery_request(policy),
        )
    )
    static_plan = StaticHttpsCollectionPlan(
        plan_id=policy.plan_id,
        source_id=policy.source_id,
        artifacts=tuple(artifacts),
    )
    verifier_policy = IedbSourceVerifierPolicy(
        policy_id=policy.source_verifier_policy_id,
        source_id=policy.source_id,
        capture_id_prefix=policy.capture_id_prefix,
        layout=IedbPromotionLayout(
            metrics_before_artifact_id='a-metrics-before',
            metrics_after_artifact_id='z-metrics-after',
            expected_table_names=policy.expected_table_names,
            pages=tuple(captured_pages),
        ),
        release_evidence_table=policy.release_evidence_table,
        accepted_tls_versions=policy.accepted_tls_versions,
        scope_profile=policy.scope_profile,
    )
    verifier_policy_bytes = canonical_json_bytes(verifier_policy)
    selected_metrics = tuple(item.model_dump(mode='json') for item in selected)
    compilation = IedbProductionPlanCompilation(
        compiler_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        discovery_metrics_sha256=hashlib.sha256(discovery_metrics_bytes).hexdigest(),
        discovery_metrics_bytes=len(discovery_metrics_bytes),
        discovery_receipt_sha256=hashlib.sha256(discovery_receipt_bytes).hexdigest(),
        discovery_receipt_bytes=len(discovery_receipt_bytes),
        discovery_completed_at=receipt.completed_at,
        discovery_source_build_at=source_build_at,
        normalized_selected_metrics_sha256=hashlib.sha256(canonical_json_bytes(selected_metrics)).hexdigest(),
        tables=tuple(planned_tables),
        total_page_count=total_pages,
        static_collection_plan_sha256=static_plan_sha256(static_plan),
        source_verifier_policy_sha256=hashlib.sha256(verifier_policy_bytes).hexdigest(),
    )
    return CompiledIedbProductionPlan(
        policy=policy,
        compilation=compilation,
        static_plan=static_plan,
        source_verifier_policy=verifier_policy,
        discovery_metrics_bytes=discovery_metrics_bytes,
        discovery_receipt_bytes=discovery_receipt_bytes,
    )


def discover_and_compile_iedb_production_plan(
    *,
    policy_bytes: bytes,
    expected_policy_sha256: str,
    transport: HttpsTransport | None = None,
) -> CompiledIedbProductionPlan:
    """Perform the one live planning request, then compile the future official plan."""

    policy = _canonical_policy(policy_bytes, expected_policy_sha256)
    temporary = capture_https_to_tempfile(
        metrics_discovery_request(policy),
        transport=transport,
    )
    try:
        body = _read_regular(temporary.path, maximum=policy.metrics_max_body_bytes)
        receipt_bytes = canonical_json_bytes(temporary.receipt)
    finally:
        temporary.delete()
    return compile_iedb_production_plan(
        policy_bytes=policy_bytes,
        expected_policy_sha256=expected_policy_sha256,
        discovery_metrics_bytes=body,
        discovery_receipt_bytes=receipt_bytes,
    )


def write_compiled_iedb_production_plan(
    compiled: CompiledIedbProductionPlan,
    output_root: Path,
) -> Path:
    """Atomically publish a create-once planning bundle for registry selection."""

    compiled = _validated_compiled_bundle(compiled)
    requested_target = output_root.expanduser().absolute()
    if not requested_target.name or requested_target.name in {'.', '..'}:
        raise IedbProductionPlanError('IEDB planning output requires a named child directory')
    parent = requested_target.parent.resolve(strict=True)
    target = parent / requested_target.name
    parent_identity = _directory_identity(parent)
    stage = Path(tempfile.mkdtemp(prefix=f'.{target.name}.staging-', dir=parent))
    try:
        if stage.parent.resolve(strict=True) != parent:
            raise IedbProductionPlanError('IEDB planning staging directory escaped its resolved parent')
        os.chmod(stage, 0o700)
        payloads = {
            'discovery-api-metrics.json': compiled.discovery_metrics_bytes,
            'discovery-api-metrics-receipt.json': compiled.discovery_receipt_bytes,
            'plan-compilation.json': canonical_json_bytes(compiled.compilation),
            'source-verifier-policy.json': canonical_json_bytes(compiled.source_verifier_policy),
            'static-collection-plan.json': canonical_json_bytes(compiled.static_plan),
        }
        for name in _OUTPUT_FILES:
            _write_new(stage / name, payloads[name])
        fsync_directory(stage)
        if _directory_identity(parent) != parent_identity:
            raise IedbProductionPlanError('IEDB planning output parent changed before publication')
        rename_directory_noreplace(stage, target)
        if _directory_identity(parent) != parent_identity:
            raise IedbProductionPlanError('IEDB planning output parent changed during publication')
        fsync_directory(parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def read_iedb_production_plan_input(
    path: Path,
    *,
    kind: Literal['compiler_policy', 'discovery_metrics', 'discovery_receipt'],
) -> bytes:
    """Read one bounded planning input without following a final-component symlink."""

    maximum_by_kind = {
        'compiler_policy': _MAX_POLICY_BYTES,
        'discovery_metrics': _MAX_METRICS_BYTES,
        'discovery_receipt': _MAX_RECEIPT_BYTES,
    }
    return _read_regular(
        path.expanduser().absolute(),
        maximum=maximum_by_kind[kind],
    )


def _canonical_policy(
    payload: bytes,
    expected_sha256: str,
) -> IedbProductionPlanCompilerPolicy:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_POLICY_BYTES:
        raise IedbProductionPlanError('IEDB compiler policy must be nonempty bounded bytes')
    digest = hashlib.sha256(payload).hexdigest()
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in '0123456789abcdef' for character in expected_sha256)
        or not hmac.compare_digest(digest, expected_sha256)
    ):
        raise IedbProductionPlanError('IEDB compiler policy differs from its out-of-band expected digest')
    try:
        policy = IedbProductionPlanCompilerPolicy.model_validate_json(payload)
    except ValueError as error:
        raise IedbProductionPlanError('IEDB compiler policy is invalid') from error
    if payload != canonical_json_bytes(policy):
        raise IedbProductionPlanError('IEDB compiler policy must use canonical JSON')
    return policy


def _validated_compiled_bundle(
    compiled: CompiledIedbProductionPlan,
) -> CompiledIedbProductionPlan:
    if not isinstance(compiled, CompiledIedbProductionPlan):
        raise TypeError('compiled must be a CompiledIedbProductionPlan')
    policy_bytes = canonical_json_bytes(compiled.policy)
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    rebuilt = compile_iedb_production_plan(
        policy_bytes=policy_bytes,
        expected_policy_sha256=policy_sha256,
        discovery_metrics_bytes=compiled.discovery_metrics_bytes,
        discovery_receipt_bytes=compiled.discovery_receipt_bytes,
    )
    if compiled != rebuilt:
        raise IedbProductionPlanError('IEDB planning bundle fields differ from their exact policy and discovery inputs')
    return rebuilt


def _canonical_receipt(payload: bytes) -> HttpsCaptureReceipt:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise IedbProductionPlanError('IEDB discovery receipt must be nonempty bounded bytes')
    try:
        receipt = HttpsCaptureReceipt.model_validate_json(payload)
    except ValueError as error:
        raise IedbProductionPlanError('IEDB discovery receipt is invalid') from error
    if payload != canonical_json_bytes(receipt):
        raise IedbProductionPlanError('IEDB discovery receipt must use canonical JSON')
    return receipt


def _verify_discovery_receipt(
    receipt: HttpsCaptureReceipt,
    *,
    body: bytes,
    policy: IedbProductionPlanCompilerPolicy,
) -> None:
    if not isinstance(body, bytes) or not body or len(body) > policy.metrics_max_body_bytes:
        raise IedbProductionPlanError('IEDB discovery metrics exceed the compiler policy')
    header_names = {item.name for item in receipt.request_headers}
    response_names = {item.name for item in receipt.response_headers}
    peer = receipt.tls_peer
    if (
        receipt.requested_url != _METRICS_URL
        or receipt.final_url != _METRICS_URL
        or receipt.request_headers != prepared_request_headers(metrics_discovery_request(policy))
        or receipt.status_code != 200
        or receipt.body_sha256 != hashlib.sha256(body).hexdigest()
        or receipt.body_byte_count != len(body)
        or {'range', 'range-unit'} & header_names
        or 'content-range' in response_names
        or peer is None
        or peer.server_name != _OFFICIAL_HOST
        or peer.certificate_der_sha256 is None
        or peer.tls_version not in policy.accepted_tls_versions
    ):
        raise IedbProductionPlanError(
            'IEDB discovery receipt differs from exact full metrics bytes or official TLS policy'
        )


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise IedbProductionPlanError('IEDB planning output parent is unavailable') from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise IedbProductionPlanError('IEDB planning output parent is not a directory')
    return metadata.st_dev, metadata.st_ino


def _metric_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise IedbProductionPlanError('IEDB metric creation time is invalid') from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short IEDB planning artifact write')
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular(path: Path, *, maximum: int) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum:
            raise IedbProductionPlanError('IEDB planning input is not a bounded regular file')
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise IedbProductionPlanError('IEDB planning input changed while reading')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise IedbProductionPlanError('IEDB planning input changed while reading')
        after = os.fstat(descriptor)

        def identities(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if identities(before) != identities(after):
            raise IedbProductionPlanError('IEDB planning input changed while reading')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


__all__ = [
    'CompiledIedbProductionPlan',
    'IEDB_PRODUCTION_PLAN_COMPILATION_SCHEMA_VERSION',
    'IEDB_PRODUCTION_PLAN_COMPILER_POLICY_SCHEMA_VERSION',
    'IedbPlannedTable',
    'IedbProductionPlanCompilation',
    'IedbProductionPlanCompilerPolicy',
    'IedbProductionPlanError',
    'compile_iedb_production_plan',
    'discover_and_compile_iedb_production_plan',
    'metrics_discovery_request',
    'read_iedb_production_plan_input',
    'write_compiled_iedb_production_plan',
]
