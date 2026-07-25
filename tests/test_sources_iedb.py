from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord
from vaxreplay.iedb.live_capture import IedbApiPageSpec
from vaxreplay.iedb.raw_schema import IedbEndpoint
from vaxreplay.operations.http_capture import (
    HttpRequestHeader,
    HttpsCaptureReceipt,
    NormalizedResponseHeader,
    TlsPeerMetadata,
)
from vaxreplay.operations.promotion import (
    AdapterSourceInput,
    ExactPromotedCapture,
    SourceVerificationInput,
    _normalize_adapter_result,
)
from vaxreplay.operations.promotion_schema import SourceRecordBinding, SourceRecordDisposition
from vaxreplay.sources.iedb import (
    IEDB_ANTIGEN_ADAPTER_EXCLUSION_REASON_CODES,
    IEDB_TIER_A_ANTIGEN_TABLES,
    IedbAntigenAdapterPolicy,
    IedbAntigenCandidateMap,
    IedbAntigenTablePolicy,
    IedbCapturedPage,
    IedbProductionAdapterError,
    IedbProductionSourceError,
    IedbPromotionLayout,
    IedbSourceVerifierPolicy,
    adapt_iedb_antigen_targets,
    adapt_tier_a_iedb_antigen_targets,
    iedb_antigen_adapter_policy_bytes,
    iedb_source_verifier_policy_bytes,
    verify_iedb_source,
    verify_tier_a_iedb_source,
)

_SOURCE_ID = 'iedb:prospective-iq-api'
_CAPTURED_AT = datetime(2026, 7, 14, 0, 5, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _Artifact:
    role: str
    payload: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def byte_count(self) -> int:
        return len(self.payload)


def _policy() -> IedbSourceVerifierPolicy:
    page = IedbApiPageSpec(
        table_name='tcell_search',
        id_field='tcell_id',
        request_url='https://query-api.iedb.org/tcell_search?order=tcell_id',
        data_relative_path='pages/tcell.json',
        receipt_relative_path='receipts/tcell.json',
        data_format='json',
    )
    return IedbSourceVerifierPolicy(
        policy_id='iedb-prospective-verifier-policy-v1',
        source_id=_SOURCE_ID,
        capture_id_prefix='iedb-prospective',
        layout=IedbPromotionLayout(
            metrics_before_artifact_id='a-metrics-before',
            metrics_after_artifact_id='z-metrics-after',
            expected_table_names=('tcell_search',),
            pages=(IedbCapturedPage(artifact_id='m-tcell-page-0', page=page),),
        ),
        release_evidence_table='tcell_search',
    )


def test_layout_requires_static_collector_artifact_order_to_bracket_pages() -> None:
    page = _policy().layout.pages[0]
    with pytest.raises(ValueError, match='static collection order'):
        IedbPromotionLayout(
            metrics_before_artifact_id='z-metrics-before',
            metrics_after_artifact_id='a-metrics-after',
            expected_table_names=('tcell_search',),
            pages=(page,),
        )


def _tier_a_source_policy(*, encoded_select: bool = False) -> IedbSourceVerifierPolicy:
    pages = []
    for ordinal, (table_name, id_field) in enumerate(
        (
            ('bcell_search', 'bcell_id'),
            ('mhc_search', 'elution_id'),
            ('tcell_search', 'tcell_id'),
        )
    ):
        query = f'order={id_field}'
        if encoded_select and table_name == 'bcell_search':
            query += f'&%73elect={id_field}'
        page = IedbApiPageSpec(
            table_name=table_name,
            id_field=id_field,
            request_url=f'https://query-api.iedb.org/{table_name}?{query}',
            data_relative_path=f'pages/{table_name}.json',
            receipt_relative_path=f'receipts/{table_name}.json',
            data_format='json',
        )
        pages.append(IedbCapturedPage(artifact_id=f'm-{ordinal}-{table_name}', page=page))
    return IedbSourceVerifierPolicy(
        policy_id='iedb-tier-a-source-v1',
        source_id=_SOURCE_ID,
        capture_id_prefix='iedb-tier-a',
        scope_profile='tier_a_antigen_all_assay_tables_v1',
        layout=IedbPromotionLayout(
            metrics_before_artifact_id='a-metrics-before',
            metrics_after_artifact_id='z-metrics-after',
            expected_table_names=IEDB_TIER_A_ANTIGEN_TABLES,
            pages=tuple(pages),
        ),
        release_evidence_table='tcell_search',
    )


def test_tier_a_source_profile_requires_all_assays_and_unprojected_rows() -> None:
    policy = _tier_a_source_policy()
    assert policy.layout.expected_table_names == ('bcell_search', 'mhc_search', 'tcell_search')

    with pytest.raises(ValueError, match='all B-cell'):
        IedbSourceVerifierPolicy.model_validate(
            {**_policy().model_dump(mode='python'), 'scope_profile': 'tier_a_antigen_all_assay_tables_v1'}
        )
    with pytest.raises(ValueError, match='forbids projected'):
        _tier_a_source_policy(encoded_select=True)


def test_production_iedb_workers_reject_research_scoped_policies() -> None:
    policy, verifier_input = _input()
    with pytest.raises(IedbProductionSourceError, match='production IEDB worker'):
        verify_tier_a_iedb_source(
            verifier_input,
            iedb_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )

    adapter_input, adapter_policy = _antigen_adapter_fixture()
    with pytest.raises(IedbProductionAdapterError, match='production IEDB adapter'):
        adapt_tier_a_iedb_antigen_targets(
            (adapter_input,),
            iedb_antigen_adapter_policy_bytes(adapter_policy),
        )


def _receipt(
    payload: bytes,
    url: str,
    *,
    content_range: str | None = None,
    started_seconds_before_capture: int = 4,
    completed_seconds_before_capture: int = 3,
) -> bytes:
    request_headers = [
        HttpRequestHeader(name='accept', value='application/json'),
        HttpRequestHeader(name='accept-encoding', value='identity'),
        HttpRequestHeader(name='host', value='query-api.iedb.org'),
    ]
    response_headers = [
        NormalizedResponseHeader(name='content-length', values=(str(len(payload)),)),
        NormalizedResponseHeader(name='content-type', values=('application/json',)),
    ]
    status = 200
    if content_range is not None:
        request_headers.extend(
            (
                HttpRequestHeader(name='range', value='0-9999'),
                HttpRequestHeader(name='range-unit', value='items'),
            )
        )
        response_headers.append(NormalizedResponseHeader(name='content-range', values=(content_range,)))
        status = 206
    receipt = HttpsCaptureReceipt(
        requested_url=url,
        final_url=url,
        request_headers=tuple(sorted(request_headers, key=lambda item: item.name)),
        status_code=status,
        response_headers=tuple(sorted(response_headers, key=lambda item: item.name)),
        body_sha256=hashlib.sha256(payload).hexdigest(),
        body_byte_count=len(payload),
        started_at=_CAPTURED_AT - timedelta(seconds=started_seconds_before_capture),
        completed_at=_CAPTURED_AT - timedelta(seconds=completed_seconds_before_capture),
        tls_peer=TlsPeerMetadata(
            server_name='query-api.iedb.org',
            tls_version='TLSv1.3',
            certificate_der_sha256='a' * 64,
        ),
    )
    return canonical_json_bytes(receipt)


def _input(*, tls_receipt_override: bytes | None = None, metrics_after: bytes | None = None):
    policy = _policy()
    metrics = canonical_json_bytes(
        [
            {
                'creation_date': '2026-07-14T00:00:00Z',
                'record_count': 2,
                'search_table_name': 'tcell_search',
            }
        ]
    )
    page = canonical_json_bytes(
        [
            {
                'reference_iri': 'VAXREPLAY_FIXTURE_REFERENCE:ROW-1',
                'structure_iri': 'VAXREPLAY_FIXTURE_EPITOPE:ROW-1',
                'tcell_id': 1,
            },
            {
                'reference_iri': 'VAXREPLAY_FIXTURE_REFERENCE:ROW-2',
                'structure_iri': 'VAXREPLAY_FIXTURE_EPITOPE:ROW-2',
                'tcell_id': 2,
            },
        ]
    )
    artifacts = (
        _Artifact('body.a-metrics-before', metrics),
        _Artifact('body.m-tcell-page-0', page),
        _Artifact('body.z-metrics-after', metrics if metrics_after is None else metrics_after),
        _Artifact(
            'receipt.z-metrics-after',
            _receipt(
                metrics if metrics_after is None else metrics_after,
                policy.layout.metrics_url,
                started_seconds_before_capture=2,
                completed_seconds_before_capture=1,
            ),
        ),
        _Artifact(
            'receipt.a-metrics-before',
            _receipt(
                metrics,
                policy.layout.metrics_url,
                started_seconds_before_capture=6,
                completed_seconds_before_capture=5,
            ),
        ),
        _Artifact(
            'receipt.m-tcell-page-0',
            tls_receipt_override
            or _receipt(
                page,
                policy.layout.pages[0].page.request_url,
                content_range='0-1/2',
            ),
        ),
    )
    capture = cast(
        ExactPromotedCapture,
        SimpleNamespace(
            binding=SimpleNamespace(
                source_id=_SOURCE_ID,
                attempt_id=f'attempt-{"1" * 32}',
                captured_at=_CAPTURED_AT,
            ),
            artifacts=artifacts,
        ),
    )
    verifier_input = SourceVerificationInput(
        source_id=_SOURCE_ID,
        captures=(capture,),
        capture_inventory_sha256='2' * 64,
    )
    return policy, verifier_input


def test_verifies_collector_tls_metadata_stable_metrics_and_complete_ordered_pages() -> None:
    policy, verifier_input = _input()
    policy_bytes = iedb_source_verifier_policy_bytes(policy)

    run = verify_iedb_source(
        verifier_input,
        policy_bytes,
        implementation_sha256='3' * 64,
        execution_environment_sha256='4' * 64,
    )

    assert run.result.source_release.source_release_at == datetime(2026, 7, 14, tzinfo=timezone.utc)
    assert run.result.source_release.authority_field == 'creation_date'
    assert run.result.source_release.evidence_source_record_id == 'api_metrics:tcell_search'
    assert run.result.verified_source_record_count == 3
    assert [
        line['source_record_id']
        for line in (json.loads(raw) for raw in run.verified_records.decode('utf-8').splitlines())
    ] == ['api_metrics:tcell_search', 'tcell_search:1', 'tcell_search:2']


def test_rejects_noncanonical_policy_bytes() -> None:
    policy, verifier_input = _input()
    with pytest.raises(IedbProductionSourceError, match='canonical'):
        verify_iedb_source(
            verifier_input,
            iedb_source_verifier_policy_bytes(policy) + b'\n',
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_rejects_missing_tls_peer_evidence() -> None:
    policy, verifier_input = _input()
    page = next(item for item in verifier_input.captures[0].artifacts if item.role == 'body.m-tcell-page-0')
    url = policy.layout.pages[0].page.request_url
    insecure = HttpsCaptureReceipt(
        requested_url=url,
        final_url=url,
        request_headers=(
            HttpRequestHeader(name='accept-encoding', value='identity'),
            HttpRequestHeader(name='host', value='query-api.iedb.org'),
        ),
        status_code=206,
        response_headers=(
            NormalizedResponseHeader(name='content-length', values=(str(page.byte_count),)),
            NormalizedResponseHeader(name='content-range', values=('0-1/2',)),
        ),
        body_sha256=page.sha256,
        body_byte_count=page.byte_count,
        started_at=_CAPTURED_AT - timedelta(seconds=2),
        completed_at=_CAPTURED_AT - timedelta(seconds=1),
        tls_peer=None,
    )
    _policy_value, insecure_input = _input(tls_receipt_override=canonical_json_bytes(insecure))
    with pytest.raises(IedbProductionSourceError, match='TLS peer metadata'):
        verify_iedb_source(
            insecure_input,
            iedb_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_rejects_source_build_change_during_capture() -> None:
    changed = canonical_json_bytes(
        [
            {
                'creation_date': '2026-07-14T00:01:00Z',
                'record_count': 2,
                'search_table_name': 'tcell_search',
            }
        ]
    )
    policy, verifier_input = _input(metrics_after=changed)
    with pytest.raises(ValueError, match='api_metrics changed'):
        verify_iedb_source(
            verifier_input,
            iedb_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def _antigen_adapter_fixture():
    source_policy = _policy()
    metrics = canonical_json_bytes(
        [
            {
                'creation_date': '2026-07-14T00:00:00Z',
                'record_count': 5,
                'search_table_name': 'tcell_search',
            }
        ]
    )
    rows = [
        {
            'parent_source_antigen_iri': 'VAXREPLAY_FIXTURE_PROTEIN:ALPHA',
            'parent_source_antigen_name': 'Surface protein one',
            'qualitative_measure': 'Positive',
            'reference_iri': 'VAXREPLAY_FIXTURE_REFERENCE:ROW-1',
            'source_organism_iri': 'VAXREPLAY_FIXTURE_TAXON:TARGET',
            'source_organism_name': 'Prospective pathogen',
            'structure_iri': 'VAXREPLAY_FIXTURE_EPITOPE:ROW-1',
            'tcell_id': 1,
        },
        {
            'parent_source_antigen_iri': 'VAXREPLAY_FIXTURE_PROTEIN:BETA',
            'parent_source_antigen_name': 'Surface protein two',
            'qualitative_measure': 'Negative',
            'reference_iri': 'VAXREPLAY_FIXTURE_REFERENCE:ROW-2',
            'source_organism_iri': 'VAXREPLAY_FIXTURE_TAXON:TARGET',
            'source_organism_name': 'Prospective pathogen',
            'structure_iri': 'VAXREPLAY_FIXTURE_EPITOPE:ROW-2',
            'tcell_id': 2,
        },
        {
            'parent_source_antigen_iri': 'VAXREPLAY_FIXTURE_PROTEIN:ALPHA',
            'parent_source_antigen_name': 'Surface protein one',
            'qualitative_measure': 'Positive-Low',
            'reference_iri': 'VAXREPLAY_FIXTURE_REFERENCE:ROW-3',
            'source_organism_iri': 'VAXREPLAY_FIXTURE_TAXON:TARGET',
            'source_organism_name': 'Prospective pathogen',
            'structure_iri': 'VAXREPLAY_FIXTURE_EPITOPE:ROW-3',
            'tcell_id': 3,
        },
        {
            'parent_source_antigen_iri': 'VAXREPLAY_FIXTURE_PROTEIN:OUT-OF-SCOPE',
            'qualitative_measure': 'Positive',
            'reference_iri': 'VAXREPLAY_FIXTURE_REFERENCE:ROW-4',
            'source_organism_iri': 'VAXREPLAY_FIXTURE_TAXON:OUT-OF-SCOPE',
            'structure_iri': 'VAXREPLAY_FIXTURE_EPITOPE:ROW-4',
            'tcell_id': 4,
        },
        {
            'qualitative_measure': 'Positive',
            'reference_iri': 'VAXREPLAY_FIXTURE_REFERENCE:ROW-5',
            'source_organism_iri': 'VAXREPLAY_FIXTURE_TAXON:TARGET',
            'structure_iri': 'VAXREPLAY_FIXTURE_EPITOPE:ROW-5',
            'tcell_id': 5,
        },
    ]
    page = canonical_json_bytes(rows)
    artifacts = (
        _Artifact('body.a-metrics-before', metrics),
        _Artifact('body.m-tcell-page-0', page),
        _Artifact('body.z-metrics-after', metrics),
        _Artifact(
            'receipt.a-metrics-before',
            _receipt(
                metrics,
                source_policy.layout.metrics_url,
                started_seconds_before_capture=6,
                completed_seconds_before_capture=5,
            ),
        ),
        _Artifact(
            'receipt.m-tcell-page-0',
            _receipt(
                page,
                source_policy.layout.pages[0].page.request_url,
                content_range='0-4/5',
            ),
        ),
        _Artifact(
            'receipt.z-metrics-after',
            _receipt(
                metrics,
                source_policy.layout.metrics_url,
                started_seconds_before_capture=2,
                completed_seconds_before_capture=1,
            ),
        ),
    )
    capture = cast(
        ExactPromotedCapture,
        SimpleNamespace(
            binding=SimpleNamespace(
                source_id=_SOURCE_ID,
                attempt_id=f'attempt-{"6" * 32}',
                captured_at=_CAPTURED_AT,
            ),
            artifacts=artifacts,
        ),
    )
    verifier_input = SourceVerificationInput(
        source_id=_SOURCE_ID,
        captures=(capture,),
        capture_inventory_sha256='7' * 64,
    )
    verified = verify_iedb_source(
        verifier_input,
        iedb_source_verifier_policy_bytes(source_policy),
        implementation_sha256='8' * 64,
        execution_environment_sha256='9' * 64,
    )
    records = tuple(SourceRecordBinding.model_validate_json(line) for line in verified.verified_records.splitlines())
    adapter_input = AdapterSourceInput(
        source_id=_SOURCE_ID,
        captures=(capture,),
        verification_result=verified.result,
        verified_records=records,
    )
    adapter_policy = IedbAntigenAdapterPolicy(
        policy_id='iedb-antigen-open-universe-v1',
        source_id=_SOURCE_ID,
        episode_id='prospective-antigen-001',
        decision_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        target_source_organism_iris=('VAXREPLAY_FIXTURE_TAXON:TARGET',),
        eligible_tables=(IedbAntigenTablePolicy(endpoint=IedbEndpoint.TCELL, id_field='tcell_id'),),
    )
    return adapter_input, adapter_policy


def test_antigen_adapter_enumerates_all_distinct_targets_without_top_n() -> None:
    adapter_input, policy = _antigen_adapter_fixture()
    result = adapt_iedb_antigen_targets(
        (adapter_input,),
        iedb_antigen_adapter_policy_bytes(policy),
    )

    candidates = tuple(CandidateRecord.model_validate_json(line) for line in result.candidate_records.splitlines())
    evidence = tuple(EvidenceRecord.model_validate_json(line) for line in result.evidence_records.splitlines())
    dispositions = tuple(SourceRecordDisposition.model_validate_json(line) for line in result.dispositions.splitlines())
    assert len(candidates) == 2
    assert len(evidence) == 3
    assert sum(item.disposition == 'normalized' for item in dispositions) == 3
    assert {item.reason_code for item in dispositions if item.disposition == 'excluded'} == {
        'missing_parent_antigen_iri',
        'source_metadata_record',
        'source_organism_out_of_scope',
    }
    assert all(item.eligible for item in candidates)
    assert all(item.available_at == datetime(2026, 7, 14, tzinfo=timezone.utc) for item in evidence)

    assert result.auxiliary_outputs is not None
    candidate_map = IedbAntigenCandidateMap.model_validate_json(result.auxiliary_outputs['iedb-antigen-candidate-map'])
    assert {item.parent_source_antigen_iri for item in candidate_map.candidates} == {
        'VAXREPLAY_FIXTURE_PROTEIN:ALPHA',
        'VAXREPLAY_FIXTURE_PROTEIN:BETA',
    }
    _normalize_adapter_result(
        result,
        (adapter_input,),
        IEDB_ANTIGEN_ADAPTER_EXCLUSION_REASON_CODES,
    )


def test_antigen_adapter_rejects_post_cutoff_capture() -> None:
    adapter_input, policy = _antigen_adapter_fixture()
    too_early = policy.model_copy(update={'decision_at': _CAPTURED_AT - timedelta(seconds=1)})
    with pytest.raises(IedbProductionAdapterError, match='source release|selected capture'):
        adapt_iedb_antigen_targets(
            (adapter_input,),
            iedb_antigen_adapter_policy_bytes(too_early),
        )


def test_antigen_adapter_rejects_noncanonical_policy_bytes() -> None:
    adapter_input, policy = _antigen_adapter_fixture()
    with pytest.raises(IedbProductionAdapterError, match='canonical'):
        adapt_iedb_antigen_targets(
            (adapter_input,),
            iedb_antigen_adapter_policy_bytes(policy) + b'\n',
        )
