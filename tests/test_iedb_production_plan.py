from __future__ import annotations

import hashlib
import io
import json
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.iedb.cli import main as iedb_main
from vaxreplay.iedb.production_plan import (
    CompiledIedbProductionPlan,
    IedbProductionPlanCompilerPolicy,
    IedbProductionPlanError,
    compile_iedb_production_plan,
    discover_and_compile_iedb_production_plan,
    metrics_discovery_request,
    read_iedb_production_plan_input,
    write_compiled_iedb_production_plan,
)
from vaxreplay.operations.collector import static_plan_sha256
from vaxreplay.operations.http_capture import (
    HttpRequestHeader,
    HttpsCaptureReceipt,
    NormalizedResponseHeader,
    PreparedHttpsRequest,
    TlsPeerMetadata,
    prepared_request_headers,
)

_METRICS_URL = 'https://query-api.iedb.org/api_metrics?order=search_table_name'
_OFFICIAL_HOST = 'query-api.iedb.org'
_BUILD_AT = datetime(2026, 7, 10, tzinfo=UTC)
_STARTED_AT = datetime(2026, 7, 11, 11, 59, tzinfo=UTC)
_COMPLETED_AT = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
_CERTIFICATE_SHA256 = 'a' * 64


def _policy(**updates: object) -> IedbProductionPlanCompilerPolicy:
    values: dict[str, object] = {
        'compiler_policy_id': 'iedb-production-compiler-v1',
        'plan_id': 'iedb-production-plan-v1',
        'source_id': 'iedb:iq-api',
        'source_verifier_policy_id': 'iedb-production-verifier-v1',
        'capture_id_prefix': 'iedb-prod',
    }
    values.update(updates)
    return IedbProductionPlanCompilerPolicy.model_validate(values)


def _policy_material(
    policy: IedbProductionPlanCompilerPolicy,
) -> tuple[bytes, str]:
    policy_bytes = canonical_json_bytes(policy)
    return policy_bytes, hashlib.sha256(policy_bytes).hexdigest()


def _metric_rows(
    counts: dict[str, int] | None = None,
    *,
    build_times: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    active_counts = counts or {
        'bcell_search': 1,
        'mhc_search': 1,
        'tcell_search': 1,
    }
    active_times = build_times or {}
    return [
        {
            'search_table_name': table_name,
            'record_count': record_count,
            'creation_date': active_times.get(
                table_name,
                _BUILD_AT.isoformat().replace('+00:00', 'Z'),
            ),
        }
        for table_name, record_count in active_counts.items()
    ]


def _metrics(
    counts: dict[str, int] | None = None,
    *,
    build_times: dict[str, str] | None = None,
) -> bytes:
    return canonical_json_bytes(_metric_rows(counts, build_times=build_times))


def _tls_peer(
    *,
    server_name: str = _OFFICIAL_HOST,
    tls_version: str = 'TLSv1.3',
    certificate_der_sha256: str | None = _CERTIFICATE_SHA256,
) -> TlsPeerMetadata:
    return TlsPeerMetadata(
        server_name=server_name,
        peer_address='203.0.113.10',
        peer_port=443,
        tls_version=tls_version,
        cipher_suite='TLS_AES_256_GCM_SHA384',
        cipher_bits=256,
        certificate_der_sha256=certificate_der_sha256,
    )


def _receipt(
    body: bytes,
    policy: IedbProductionPlanCompilerPolicy,
    *,
    requested_url: str = _METRICS_URL,
    request_headers: tuple[HttpRequestHeader, ...] | None = None,
    status_code: int = 200,
    response_headers: tuple[NormalizedResponseHeader, ...] | None = None,
    completed_at: datetime = _COMPLETED_AT,
    tls_peer: TlsPeerMetadata | None = None,
    omit_tls_peer: bool = False,
) -> HttpsCaptureReceipt:
    peer = None if omit_tls_peer else (tls_peer or _tls_peer())
    return HttpsCaptureReceipt(
        requested_url=requested_url,
        final_url=requested_url,
        request_headers=(
            request_headers
            if request_headers is not None
            else prepared_request_headers(metrics_discovery_request(policy))
        ),
        status_code=status_code,
        response_headers=(
            response_headers
            if response_headers is not None
            else (
                NormalizedResponseHeader(
                    name='content-length',
                    values=(str(len(body)),),
                ),
            )
        ),
        body_sha256=hashlib.sha256(body).hexdigest(),
        body_byte_count=len(body),
        started_at=_STARTED_AT,
        completed_at=completed_at,
        tls_peer=peer,
    )


def _compile(
    policy: IedbProductionPlanCompilerPolicy,
    metrics_bytes: bytes,
    *,
    receipt: HttpsCaptureReceipt | None = None,
) -> CompiledIedbProductionPlan:
    policy_bytes, policy_sha256 = _policy_material(policy)
    active_receipt = receipt or _receipt(metrics_bytes, policy)
    return compile_iedb_production_plan(
        policy_bytes=policy_bytes,
        expected_policy_sha256=policy_sha256,
        discovery_metrics_bytes=metrics_bytes,
        discovery_receipt_bytes=canonical_json_bytes(active_receipt),
    )


def _header_value(request_headers: tuple[HttpRequestHeader, ...], name: str) -> str | None:
    return next((header.value for header in request_headers if header.name == name), None)


def test_compiles_exact_deterministic_page_and_range_plan() -> None:
    policy = _policy(
        range_page_size=3,
        page_max_body_bytes=123_456,
        request_timeout_seconds=17.5,
    )
    rows = _metric_rows(
        {
            'tcell_search': 7,
            'unrelated_search': 99,
            'mhc_search': 3,
            'bcell_search': 0,
        }
    )
    rows[1]['creation_date'] = '1999-01-01T00:00:00Z'
    metrics_bytes = canonical_json_bytes(rows)

    compiled = _compile(policy, metrics_bytes)
    compiled_again = _compile(policy, metrics_bytes)

    assert compiled == compiled_again
    assert tuple(table.model_dump(mode='python') for table in compiled.compilation.tables) == (
        {
            'table_name': 'bcell_search',
            'id_field': 'bcell_id',
            'discovery_record_count': 0,
            'page_count': 1,
        },
        {
            'table_name': 'mhc_search',
            'id_field': 'elution_id',
            'discovery_record_count': 3,
            'page_count': 1,
        },
        {
            'table_name': 'tcell_search',
            'id_field': 'tcell_id',
            'discovery_record_count': 7,
            'page_count': 3,
        },
    )
    assert compiled.compilation.total_page_count == 5
    assert compiled.compilation.discovery_source_build_at == _BUILD_AT
    assert not compiled.compilation.discovery_source_bytes_admitted_as_official_capture
    assert compiled.compilation.plan_requires_pre_capture_selection_registry_commitment
    assert compiled.compilation.official_capture_requires_metrics_before_and_after
    assert compiled.compilation.official_capture_requires_complete_contiguous_ranges
    assert not compiled.compilation.tier_a_release_ready

    expected_artifact_ids = (
        'a-metrics-before',
        'm-00-0000-bcell_search',
        'm-01-0000-mhc_search',
        'm-02-0000-tcell_search',
        'm-02-0001-tcell_search',
        'm-02-0002-tcell_search',
        'z-metrics-after',
    )
    artifacts = compiled.static_plan.artifacts
    assert tuple(artifact.artifact_id for artifact in artifacts) == expected_artifact_ids
    assert artifacts[0].request == metrics_discovery_request(policy)
    assert artifacts[-1].request == metrics_discovery_request(policy)

    expected_pages = (
        ('bcell_search', 'bcell_id', '0-2'),
        ('mhc_search', 'elution_id', '0-2'),
        ('tcell_search', 'tcell_id', '0-2'),
        ('tcell_search', 'tcell_id', '3-5'),
        ('tcell_search', 'tcell_id', '6-8'),
    )
    for artifact, (table_name, id_field, expected_range) in zip(
        artifacts[1:-1],
        expected_pages,
        strict=True,
    ):
        request = artifact.request
        assert request.url == f'https://{_OFFICIAL_HOST}/{table_name}?order={id_field}.asc'
        assert request.allowed_host == _OFFICIAL_HOST
        assert request.allowed_query_names == ('order',)
        assert request.allowed_status_codes == (200, 206)
        assert request.max_body_bytes == 123_456
        assert request.timeout_seconds == 17.5
        assert _header_value(request.request_headers, 'accept') == 'application/json'
        assert _header_value(request.request_headers, 'range') == expected_range
        assert _header_value(request.request_headers, 'range-unit') == 'items'

    layout = compiled.source_verifier_policy.layout
    assert layout.metrics_before_artifact_id == 'a-metrics-before'
    assert layout.metrics_after_artifact_id == 'z-metrics-after'
    assert layout.expected_table_names == ('bcell_search', 'mhc_search', 'tcell_search')
    assert tuple(page.artifact_id for page in layout.pages) == expected_artifact_ids[1:-1]
    assert tuple(page.page.data_relative_path for page in layout.pages) == (
        'pages/bcell_search-0000.json',
        'pages/mhc_search-0000.json',
        'pages/tcell_search-0000.json',
        'pages/tcell_search-0001.json',
        'pages/tcell_search-0002.json',
    )
    assert tuple(page.page.receipt_relative_path for page in layout.pages) == (
        'receipts/bcell_search-0000.json',
        'receipts/mhc_search-0000.json',
        'receipts/tcell_search-0000.json',
        'receipts/tcell_search-0001.json',
        'receipts/tcell_search-0002.json',
    )


def test_metric_order_does_not_change_normalized_plan() -> None:
    policy = _policy(range_page_size=2)
    rows = _metric_rows(
        {
            'bcell_search': 2,
            'mhc_search': 3,
            'tcell_search': 4,
        }
    )
    forward = _compile(policy, canonical_json_bytes(rows))
    reverse = _compile(policy, canonical_json_bytes(tuple(reversed(rows))))

    assert forward.static_plan == reverse.static_plan
    assert forward.source_verifier_policy == reverse.source_verifier_policy
    assert (
        forward.compilation.normalized_selected_metrics_sha256 == reverse.compilation.normalized_selected_metrics_sha256
    )
    assert forward.compilation.static_collection_plan_sha256 == reverse.compilation.static_collection_plan_sha256
    assert forward.compilation.discovery_metrics_sha256 != reverse.compilation.discovery_metrics_sha256


@pytest.mark.parametrize('missing_table', ('bcell_search', 'mhc_search', 'tcell_search'))
def test_requires_every_tier_a_assay_table(missing_table: str) -> None:
    policy = _policy()
    counts = {
        'bcell_search': 1,
        'mhc_search': 1,
        'tcell_search': 1,
    }
    del counts[missing_table]
    metrics_bytes = _metrics(counts)

    with pytest.raises(IedbProductionPlanError, match='omit a required assay table'):
        _compile(policy, metrics_bytes)


def test_policy_cannot_narrow_the_tier_a_assay_scope() -> None:
    values = _policy().model_dump(mode='python')
    values['expected_table_names'] = ('bcell_search', 'tcell_search')

    with pytest.raises(ValidationError, match='requires all three IEDB assay tables'):
        IedbProductionPlanCompilerPolicy.model_validate(values)


def test_build_times_must_match_and_not_postdate_discovery() -> None:
    policy = _policy()
    mixed_metrics = _metrics(
        build_times={
            'bcell_search': '2026-07-10T00:00:00Z',
            'mhc_search': '2026-07-10T00:00:01Z',
            'tcell_search': '2026-07-10T00:00:00Z',
        }
    )
    with pytest.raises(IedbProductionPlanError, match='mixed assay-table build times'):
        _compile(policy, mixed_metrics)

    future_metrics = _metrics(
        build_times={
            table_name: '2026-07-12T00:00:00Z' for table_name in ('bcell_search', 'mhc_search', 'tcell_search')
        }
    )
    with pytest.raises(IedbProductionPlanError, match='postdates its receipt'):
        _compile(policy, future_metrics)


def test_equivalent_offset_build_times_normalize_to_utc() -> None:
    policy = _policy()
    metrics_bytes = _metrics(
        build_times={
            'bcell_search': '2026-07-10T00:00:00Z',
            'mhc_search': '2026-07-09T17:00:00-07:00',
            'tcell_search': '2026-07-10T02:00:00+02:00',
        }
    )

    compiled = _compile(policy, metrics_bytes)

    assert compiled.compilation.discovery_source_build_at == _BUILD_AT


def test_page_inventory_cap_is_enforced_and_zero_counts_still_get_one_page() -> None:
    policy = _policy(range_page_size=1, maximum_total_pages=3)
    allowed = _compile(
        policy,
        _metrics({'bcell_search': 0, 'mhc_search': 0, 'tcell_search': 0}),
    )
    assert tuple(table.page_count for table in allowed.compilation.tables) == (1, 1, 1)

    over_cap = _metrics({'bcell_search': 2, 'mhc_search': 1, 'tcell_search': 1})
    with pytest.raises(IedbProductionPlanError, match='maximum page inventory'):
        _compile(policy, over_cap)


def test_policy_and_receipt_bytes_are_canonical_and_digest_bound() -> None:
    policy = _policy()
    metrics_bytes = _metrics()
    policy_bytes, policy_sha256 = _policy_material(policy)
    receipt_bytes = canonical_json_bytes(_receipt(metrics_bytes, policy))

    with pytest.raises(IedbProductionPlanError, match='out-of-band expected digest'):
        compile_iedb_production_plan(
            policy_bytes=policy_bytes,
            expected_policy_sha256='0' * 64,
            discovery_metrics_bytes=metrics_bytes,
            discovery_receipt_bytes=receipt_bytes,
        )

    noncanonical_policy = b' ' + policy_bytes
    with pytest.raises(IedbProductionPlanError, match='policy must use canonical JSON'):
        compile_iedb_production_plan(
            policy_bytes=noncanonical_policy,
            expected_policy_sha256=hashlib.sha256(noncanonical_policy).hexdigest(),
            discovery_metrics_bytes=metrics_bytes,
            discovery_receipt_bytes=receipt_bytes,
        )

    with pytest.raises(IedbProductionPlanError, match='receipt must use canonical JSON'):
        compile_iedb_production_plan(
            policy_bytes=policy_bytes,
            expected_policy_sha256=policy_sha256,
            discovery_metrics_bytes=metrics_bytes,
            discovery_receipt_bytes=b' ' + receipt_bytes,
        )

    other_metrics = _metrics({'bcell_search': 2, 'mhc_search': 1, 'tcell_search': 1})
    with pytest.raises(IedbProductionPlanError, match='exact full metrics bytes'):
        compile_iedb_production_plan(
            policy_bytes=policy_bytes,
            expected_policy_sha256=policy_sha256,
            discovery_metrics_bytes=other_metrics,
            discovery_receipt_bytes=receipt_bytes,
        )


def test_compilation_binds_every_input_and_output_digest() -> None:
    policy = _policy(range_page_size=2)
    metrics_bytes = _metrics({'bcell_search': 2, 'mhc_search': 3, 'tcell_search': 4})
    receipt_bytes = canonical_json_bytes(_receipt(metrics_bytes, policy))
    policy_bytes, policy_sha256 = _policy_material(policy)

    compiled = compile_iedb_production_plan(
        policy_bytes=policy_bytes,
        expected_policy_sha256=policy_sha256,
        discovery_metrics_bytes=metrics_bytes,
        discovery_receipt_bytes=receipt_bytes,
    )
    compilation = compiled.compilation

    assert compilation.compiler_policy_sha256 == hashlib.sha256(policy_bytes).hexdigest()
    assert compilation.discovery_metrics_sha256 == hashlib.sha256(metrics_bytes).hexdigest()
    assert compilation.discovery_metrics_bytes == len(metrics_bytes)
    assert compilation.discovery_receipt_sha256 == hashlib.sha256(receipt_bytes).hexdigest()
    assert compilation.discovery_receipt_bytes == len(receipt_bytes)
    assert compilation.discovery_completed_at == _COMPLETED_AT
    assert compilation.static_collection_plan_sha256 == static_plan_sha256(compiled.static_plan)
    assert (
        compilation.source_verifier_policy_sha256
        == hashlib.sha256(canonical_json_bytes(compiled.source_verifier_policy)).hexdigest()
    )


def test_rejects_partial_or_untrusted_discovery_receipts() -> None:
    policy = _policy()
    metrics_bytes = _metrics()
    exact_headers = prepared_request_headers(metrics_discovery_request(policy))
    range_headers = tuple(
        sorted(
            (*exact_headers, HttpRequestHeader(name='range', value='0-99')),
            key=lambda header: header.name,
        )
    )
    content_range_headers = (
        NormalizedResponseHeader(name='content-length', values=(str(len(metrics_bytes)),)),
        NormalizedResponseHeader(name='content-range', values=('0-2/3',)),
    )
    cases = {
        'wrong URL': _receipt(
            metrics_bytes,
            policy,
            requested_url='https://query-api.iedb.org/api_metrics?order=creation_date',
        ),
        'partial request': _receipt(
            metrics_bytes,
            policy,
            request_headers=range_headers,
        ),
        'partial response': _receipt(
            metrics_bytes,
            policy,
            response_headers=content_range_headers,
        ),
        'wrong status': _receipt(metrics_bytes, policy, status_code=206),
        'missing TLS': _receipt(metrics_bytes, policy, omit_tls_peer=True),
        'wrong SNI': _receipt(
            metrics_bytes,
            policy,
            tls_peer=_tls_peer(server_name='other.example.org'),
        ),
        'missing certificate': _receipt(
            metrics_bytes,
            policy,
            tls_peer=_tls_peer(certificate_der_sha256=None),
        ),
        'unaccepted TLS': _receipt(
            metrics_bytes,
            policy,
            tls_peer=_tls_peer(tls_version='TLSv1.1'),
        ),
    }

    for label, receipt in cases.items():
        with pytest.raises(IedbProductionPlanError, match='exact full metrics bytes') as raised:
            _compile(policy, metrics_bytes, receipt=receipt)
        assert raised.value.args, label


def test_discovery_receipt_rejects_any_exact_request_header_drift() -> None:
    policy = _policy()
    metrics_bytes = _metrics()
    drifted_headers = tuple(
        HttpRequestHeader(name=header.name, value='application/problem+json') if header.name == 'accept' else header
        for header in prepared_request_headers(metrics_discovery_request(policy))
    )

    with pytest.raises(IedbProductionPlanError, match='exact full metrics bytes'):
        _compile(
            policy,
            metrics_bytes,
            receipt=_receipt(
                metrics_bytes,
                policy,
                request_headers=drifted_headers,
            ),
        )


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.status_code = 200
        self.final_url = _METRICS_URL
        self.response_headers = (('Content-Length', str(len(body))),)
        self._body = body
        self._position = 0
        self.closed = False

    def read(self, size: int) -> bytes:
        chunk = self._body[self._position : self._position + size]
        self._position += len(chunk)
        return chunk

    def tls_peer_metadata(self) -> TlsPeerMetadata:
        return _tls_peer()

    def close(self) -> None:
        self.closed = True


class _FakeTransport:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests: list[PreparedHttpsRequest] = []

    def open(self, request: PreparedHttpsRequest) -> _FakeResponse:
        self.requests.append(request)
        return self.response


def test_live_discovery_uses_one_exact_fake_transport_request() -> None:
    policy = _policy(metrics_max_body_bytes=4096, request_timeout_seconds=9.25)
    policy_bytes, policy_sha256 = _policy_material(policy)
    metrics_bytes = _metrics({'bcell_search': 2, 'mhc_search': 3, 'tcell_search': 4})
    response = _FakeResponse(metrics_bytes)
    transport = _FakeTransport(response)

    compiled = discover_and_compile_iedb_production_plan(
        policy_bytes=policy_bytes,
        expected_policy_sha256=policy_sha256,
        transport=transport,
    )

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.method == 'GET'
    assert request.url == _METRICS_URL
    assert request.headers == prepared_request_headers(metrics_discovery_request(policy))
    assert request.timeout_seconds == 9.25
    assert response.closed
    assert compiled.discovery_metrics_bytes == metrics_bytes
    assert compiled.discovery_receipt_bytes == canonical_json_bytes(
        HttpsCaptureReceipt.model_validate_json(compiled.discovery_receipt_bytes)
    )
    assert compiled.compilation.discovery_metrics_sha256 == hashlib.sha256(metrics_bytes).hexdigest()


def test_create_once_output_is_exact_and_concurrency_safe(tmp_path: Path) -> None:
    compiled = _compile(_policy(range_page_size=2), _metrics())
    output = tmp_path / 'planning-bundle'

    def publish() -> str:
        try:
            write_compiled_iedb_production_plan(compiled, output)
        except FileExistsError:
            return 'exists'
        return 'published'

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = tuple(executor.map(lambda _ordinal: publish(), range(6)))

    assert results.count('published') == 1
    assert results.count('exists') == 5
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert tuple(sorted(path.name for path in output.iterdir())) == (
        'discovery-api-metrics-receipt.json',
        'discovery-api-metrics.json',
        'plan-compilation.json',
        'source-verifier-policy.json',
        'static-collection-plan.json',
    )
    assert (output / 'discovery-api-metrics.json').read_bytes() == compiled.discovery_metrics_bytes
    assert (output / 'discovery-api-metrics-receipt.json').read_bytes() == compiled.discovery_receipt_bytes
    assert (output / 'plan-compilation.json').read_bytes() == canonical_json_bytes(compiled.compilation)
    assert (output / 'source-verifier-policy.json').read_bytes() == canonical_json_bytes(
        compiled.source_verifier_policy
    )
    assert (output / 'static-collection-plan.json').read_bytes() == canonical_json_bytes(compiled.static_plan)
    assert not tuple(tmp_path.glob('.planning-bundle.staging-*'))


def test_offline_compiler_cli_writes_selection_ready_planning_bundle(
    tmp_path: Path,
) -> None:
    policy = _policy(range_page_size=2)
    policy_bytes, policy_sha256 = _policy_material(policy)
    metrics_bytes = _metrics({'bcell_search': 2, 'mhc_search': 3, 'tcell_search': 4})
    receipt_bytes = canonical_json_bytes(_receipt(metrics_bytes, policy))
    policy_path = tmp_path / 'policy.json'
    metrics_path = tmp_path / 'metrics.json'
    receipt_path = tmp_path / 'receipt.json'
    output_path = tmp_path / 'compiled'
    policy_path.write_bytes(policy_bytes)
    metrics_path.write_bytes(metrics_bytes)
    receipt_path.write_bytes(receipt_bytes)
    stdout = io.StringIO()

    with (
        patch.object(
            sys,
            'argv',
            [
                'vaxreplay-iedb',
                'compile-production-plan',
                '--policy',
                str(policy_path),
                '--expected-policy-sha256',
                policy_sha256,
                '--metrics-body',
                str(metrics_path),
                '--metrics-receipt',
                str(receipt_path),
                '--output-dir',
                str(output_path),
            ],
        ),
        redirect_stdout(stdout),
    ):
        iedb_main()

    result = json.loads(stdout.getvalue())
    assert result['status'] == 'planned_not_tier_a_release_ready'
    assert not result['tier_a_release_ready']
    assert result['discovery_is_planning_input_only']
    assert result['plan_requires_pre_capture_selection_registry_commitment']
    assert result['compiler_policy_sha256'] == policy_sha256
    assert result['output_dir'] == str(output_path.resolve())
    assert stdout.getvalue().encode() == canonical_json_bytes(result) + b'\n'
    assert (output_path / 'plan-compilation.json').is_file()
    assert (output_path / 'static-collection-plan.json').is_file()
    assert (output_path / 'source-verifier-policy.json').is_file()


def test_writer_revalidates_dataclass_replacement_before_creating_output(
    tmp_path: Path,
) -> None:
    compiled = _compile(_policy(), _metrics())
    inconsistent_plan = compiled.static_plan.model_copy(update={'plan_id': 'attacker-substituted-plan'})
    inconsistent = replace(compiled, static_plan=inconsistent_plan)
    output = tmp_path / 'must-not-exist'

    with pytest.raises(IedbProductionPlanError, match='fields differ'):
        write_compiled_iedb_production_plan(inconsistent, output)

    assert not output.exists()


def test_input_reader_rejects_empty_nonregular_oversized_and_symlink_inputs(
    tmp_path: Path,
) -> None:
    regular = tmp_path / 'policy.json'
    regular.write_bytes(b'{}')
    assert read_iedb_production_plan_input(regular, kind='compiler_policy') == b'{}'

    empty = tmp_path / 'empty.json'
    empty.touch()
    with pytest.raises(IedbProductionPlanError, match='bounded regular file'):
        read_iedb_production_plan_input(empty, kind='compiler_policy')

    directory = tmp_path / 'directory'
    directory.mkdir()
    with pytest.raises(IedbProductionPlanError, match='bounded regular file'):
        read_iedb_production_plan_input(directory, kind='compiler_policy')

    oversized = tmp_path / 'oversized-receipt.json'
    with oversized.open('wb') as stream:
        stream.truncate(1024 * 1024 + 1)
    with pytest.raises(IedbProductionPlanError, match='bounded regular file'):
        read_iedb_production_plan_input(oversized, kind='discovery_receipt')

    symlink = tmp_path / 'policy-link.json'
    symlink.symlink_to(regular)
    with pytest.raises(OSError):
        read_iedb_production_plan_input(symlink, kind='compiler_policy')


def test_metrics_body_cap_is_checked_before_parsing() -> None:
    policy = _policy(metrics_max_body_bytes=16)
    metrics_bytes = _metrics()
    assert len(metrics_bytes) > policy.metrics_max_body_bytes

    with pytest.raises(IedbProductionPlanError, match='exceed the compiler policy'):
        _compile(policy, metrics_bytes)


def test_duplicate_metric_table_ids_fail_closed() -> None:
    policy = _policy()
    rows = _metric_rows()
    rows.append(dict(rows[0]))
    metrics_bytes = canonical_json_bytes(rows)

    with pytest.raises(IedbProductionPlanError, match='discovery metrics are invalid'):
        _compile(policy, metrics_bytes)
