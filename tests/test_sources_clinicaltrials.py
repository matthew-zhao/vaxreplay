from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord
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
from vaxreplay.sources.clinicaltrials import (
    CTGOV_STUDY_ADAPTER_EXCLUSION_REASON_CODES,
    CtgovCapturedQuery,
    CtgovProductionAdapterError,
    CtgovProductionSourceError,
    CtgovPromotionLayout,
    CtgovSourceVerifierPolicy,
    CtgovStudyAdapterPolicy,
    CtgovStudyCandidateMap,
    adapt_ctgov_study_candidates,
    ctgov_source_verifier_policy_bytes,
    ctgov_study_adapter_policy_bytes,
    verify_ctgov_source,
)

_SOURCE_ID = 'clinicaltrials-gov:prospective-api-v2'
_CAPTURED_AT = datetime(2026, 7, 14, 14, 5, tzinfo=timezone.utc)
_VERSION_URL = 'https://clinicaltrials.gov/api/v2/version'
_QUERY_URL = (
    'https://clinicaltrials.gov/api/v2/studies?countTotal=true&format=json&'
    'markupFormat=legacy&pageSize=1000&query.intr=vaccine'
)


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


def _receipt(
    payload: bytes,
    url: str,
    *,
    started_seconds_before_capture: int = 4,
    completed_seconds_before_capture: int = 3,
) -> bytes:
    receipt = HttpsCaptureReceipt(
        requested_url=url,
        final_url=url,
        request_headers=(
            HttpRequestHeader(name='accept', value='application/json'),
            HttpRequestHeader(name='accept-encoding', value='identity'),
            HttpRequestHeader(name='host', value='clinicaltrials.gov'),
        ),
        status_code=200,
        response_headers=(
            NormalizedResponseHeader(name='content-length', values=(str(len(payload)),)),
            NormalizedResponseHeader(name='content-type', values=('application/json',)),
        ),
        body_sha256=hashlib.sha256(payload).hexdigest(),
        body_byte_count=len(payload),
        started_at=_CAPTURED_AT - timedelta(seconds=started_seconds_before_capture),
        completed_at=_CAPTURED_AT - timedelta(seconds=completed_seconds_before_capture),
        tls_peer=TlsPeerMetadata(
            server_name='clinicaltrials.gov',
            tls_version='TLSv1.3',
            certificate_der_sha256='a' * 64,
        ),
    )
    return canonical_json_bytes(receipt)


def _study(
    ordinal: int,
    *,
    study_type: str = 'INTERVENTIONAL',
    intervention_type: str = 'BIOLOGICAL',
    results_canary: str | None = None,
) -> dict[str, object]:
    study: dict[str, object] = {
        'protocolSection': {
            'armsInterventionsModule': {
                'interventions': [
                    {
                        'name': f'Vaccine candidate {ordinal}',
                        'type': intervention_type,
                    }
                ]
            },
            'conditionsModule': {'conditions': ['Prospective pathogen infection']},
            'designModule': {
                'enrollmentInfo': {'count': 100 + ordinal, 'type': 'ESTIMATED'},
                'phases': ['PHASE1'],
                'studyType': study_type,
            },
            'identificationModule': {
                'briefTitle': f'Prospective vaccine protocol {ordinal}',
                'nctId': f'NCT{ordinal:08d}',
            },
            'outcomesModule': {
                'primaryOutcomes': [
                    {
                        'measure': 'Protocol immunogenicity endpoint',
                        'timeFrame': 'Day 28',
                    }
                ]
            },
            'statusModule': {'overallStatus': 'RECRUITING'},
        }
    }
    if results_canary is not None:
        study['hasResults'] = True
        study['resultsSection'] = {'outcomeMeasuresModule': {'outcomeMeasures': [{'title': results_canary}]}}
        study['derivedSection'] = {'miscInfoModule': {'versionHolder': results_canary}}
        study['documentSection'] = {'largeDocumentModule': {'largeDocs': [results_canary]}}
    return study


def _policy() -> CtgovSourceVerifierPolicy:
    return CtgovSourceVerifierPolicy(
        policy_id='ctgov-prospective-source-v1',
        source_id=_SOURCE_ID,
        expected_api_version='2.0.3',
        layout=CtgovPromotionLayout(
            version_before_artifact_id='a-version-before',
            version_after_artifact_id='z-version-after',
            queries=(
                CtgovCapturedQuery(
                    artifact_id='m-studies-vaccine',
                    query_id='vaccine',
                    request_url=_QUERY_URL,
                ),
            ),
        ),
    )


def test_layout_requires_static_collector_artifact_order_to_bracket_queries() -> None:
    query = _policy().layout.queries[0]
    with pytest.raises(ValueError, match='version-before, queries, version-after'):
        CtgovPromotionLayout(
            version_before_artifact_id='z-version-before',
            version_after_artifact_id='a-version-after',
            queries=(query,),
        )


@pytest.mark.parametrize(
    'scope',
    (
        'query.term=AREA%5BHasResults%5Dtrue',
        'query.term=AREA%5BResultsFirstSubmitDate%5DRANGE%5BMIN%2C+2026-01-01%5D',
        'filter.ids=NCT00000001',
        'query.intr=AREA%5BHasResults%5Dtrue',
    ),
)
def test_tier_a_parent_query_rejects_result_fields_and_manual_id_lists(scope: str) -> None:
    parameters = tuple(
        sorted(
            ('countTotal=true', 'format=json', 'markupFormat=legacy', 'pageSize=1000', scope),
            key=lambda item: item.partition('=')[0],
        )
    )
    url = f'https://clinicaltrials.gov/api/v2/studies?{"&".join(parameters)}'
    with pytest.raises(ValueError, match='protocol-only|query.cond and query.intr'):
        CtgovCapturedQuery(
            artifact_id='m-forbidden-query',
            query_id='forbidden-query',
            request_url=url,
        )


def test_tier_a_layout_rejects_multiple_cherry_pickable_partitions() -> None:
    first = _policy().layout.queries[0]
    second = first.model_copy(
        update={
            'artifact_id': 'n-studies-second',
            'query_id': 'second',
            'request_url': _QUERY_URL.replace('vaccine', 'influenza'),
        }
    )
    with pytest.raises(ValueError, match='at most 1|exactly one'):
        CtgovPromotionLayout(
            version_before_artifact_id='a-version-before',
            version_after_artifact_id='z-version-after',
            queries=(first, second),
        )


def _fixture(
    *,
    next_page_token: str | None = None,
    version_after_override: bytes | None = None,
):
    policy = _policy()
    version = canonical_json_bytes(
        {
            'apiVersion': '2.0.3',
            'dataTimestamp': '2026-07-14T09:00:00',
        }
    )
    studies = [
        _study(1, results_canary='PRIVATE FUTURE RESULT CANARY'),
        _study(2),
        _study(3, study_type='OBSERVATIONAL'),
        _study(4, intervention_type='DRUG'),
    ]
    page_value: dict[str, object] = {'studies': studies, 'totalCount': len(studies)}
    if next_page_token is not None:
        page_value['nextPageToken'] = next_page_token
    page = canonical_json_bytes(page_value)
    version_after = version if version_after_override is None else version_after_override
    artifacts = (
        _Artifact('body.a-version-before', version),
        _Artifact('body.m-studies-vaccine', page),
        _Artifact('body.z-version-after', version_after),
        _Artifact(
            'receipt.a-version-before',
            _receipt(
                version,
                _VERSION_URL,
                started_seconds_before_capture=6,
                completed_seconds_before_capture=5,
            ),
        ),
        _Artifact('receipt.m-studies-vaccine', _receipt(page, _QUERY_URL)),
        _Artifact(
            'receipt.z-version-after',
            _receipt(
                version_after,
                _VERSION_URL,
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
    return policy, verifier_input, capture


def _verified_adapter_fixture():
    source_policy, verifier_input, capture = _fixture()
    verified = verify_ctgov_source(
        verifier_input,
        ctgov_source_verifier_policy_bytes(source_policy),
        implementation_sha256='3' * 64,
        execution_environment_sha256='4' * 64,
    )
    records = tuple(SourceRecordBinding.model_validate_json(line) for line in verified.verified_records.splitlines())
    adapter_input = AdapterSourceInput(
        source_id=_SOURCE_ID,
        captures=(capture,),
        verification_result=verified.result,
        verified_records=records,
    )
    adapter_policy = CtgovStudyAdapterPolicy(
        policy_id='ctgov-study-universe-v1',
        source_id=_SOURCE_ID,
        episode_id='prospective-clinical-development-001',
        decision_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        allowed_intervention_types=('BIOLOGICAL',),
        allowed_phases=('PHASE1',),
    )
    return adapter_input, adapter_policy


def test_source_verifier_binds_stable_version_and_complete_disjoint_queries() -> None:
    policy, verifier_input, _capture = _fixture()
    run = verify_ctgov_source(
        verifier_input,
        ctgov_source_verifier_policy_bytes(policy),
        implementation_sha256='3' * 64,
        execution_environment_sha256='4' * 64,
    )

    assert run.result.source_release.source_release_at == datetime(
        2026,
        7,
        14,
        13,
        tzinfo=timezone.utc,
    )
    assert run.result.source_release.authority_field == 'dataTimestamp'
    assert run.result.verified_source_record_count == 5
    records = tuple(SourceRecordBinding.model_validate_json(line) for line in run.verified_records.splitlines())
    assert [item.source_record_id for item in records] == [
        'api_version:2.0.3',
        'study:NCT00000001',
        'study:NCT00000002',
        'study:NCT00000003',
        'study:NCT00000004',
    ]


def test_source_verifier_rejects_uncommitted_continuation_page() -> None:
    policy, verifier_input, _capture = _fixture(next_page_token='opaque-token')
    with pytest.raises(CtgovProductionSourceError, match='continuation'):
        verify_ctgov_source(
            verifier_input,
            ctgov_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_version_change_during_capture() -> None:
    changed = canonical_json_bytes(
        {
            'apiVersion': '2.0.3',
            'dataTimestamp': '2026-07-14T09:01:00',
        }
    )
    policy, verifier_input, _capture = _fixture(version_after_override=changed)
    with pytest.raises(CtgovProductionSourceError, match='changed during capture'):
        verify_ctgov_source(
            verifier_input,
            ctgov_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_study_adapter_is_exhaustive_and_never_reads_results_fields() -> None:
    adapter_input, policy = _verified_adapter_fixture()
    result = adapt_ctgov_study_candidates(
        (adapter_input,),
        ctgov_study_adapter_policy_bytes(policy),
    )

    candidates = tuple(CandidateRecord.model_validate_json(line) for line in result.candidate_records.splitlines())
    evidence = tuple(EvidenceRecord.model_validate_json(line) for line in result.evidence_records.splitlines())
    dispositions = tuple(SourceRecordDisposition.model_validate_json(line) for line in result.dispositions.splitlines())
    assert [item.candidate_id for item in candidates] == [
        'ctgov-study-NCT00000001',
        'ctgov-study-NCT00000002',
    ]
    assert len(evidence) == 2
    assert 'PRIVATE FUTURE RESULT CANARY' not in result.evidence_records.decode('utf-8')
    assert sum(item.disposition == 'normalized' for item in dispositions) == 2
    assert {item.reason_code for item in dispositions if item.disposition == 'excluded'} == {
        'intervention_type_out_of_scope',
        'not_interventional',
        'source_metadata_record',
    }
    assert result.auxiliary_outputs is not None
    candidate_map = CtgovStudyCandidateMap.model_validate_json(result.auxiliary_outputs['ctgov-study-candidate-map'])
    assert [item.nct_id for item in candidate_map.candidates] == ['NCT00000001', 'NCT00000002']
    _normalize_adapter_result(
        result,
        (adapter_input,),
        CTGOV_STUDY_ADAPTER_EXCLUSION_REASON_CODES,
    )


def test_study_adapter_rejects_post_cutoff_source() -> None:
    adapter_input, policy = _verified_adapter_fixture()
    too_early = policy.model_copy(update={'decision_at': datetime(2026, 7, 13, tzinfo=timezone.utc)})
    with pytest.raises(CtgovProductionAdapterError, match='after the decision cutoff'):
        adapt_ctgov_study_candidates(
            (adapter_input,),
            ctgov_study_adapter_policy_bytes(too_early),
        )


def test_query_policy_rejects_opaque_page_tokens() -> None:
    with pytest.raises(ValueError, match='continuation tokens'):
        CtgovCapturedQuery(
            artifact_id='bad-page',
            query_id='bad',
            request_url=(
                'https://clinicaltrials.gov/api/v2/studies?countTotal=true&format=json&'
                'markupFormat=legacy&pageSize=1000&pageToken=opaque&query.term=vaccine'
            ),
        )
