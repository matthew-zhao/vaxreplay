from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord, SourceType
from vaxreplay.operations.hermetic_callback_protocol import (
    HermeticAdapterInput,
    HermeticAdapterSourceInput,
    HermeticArtifactPayload,
    HermeticCallbackProtocolError,
    HermeticCapturePayload,
    HermeticSourceVerifierInput,
    decode_callback_bytes,
    encode_callback_bytes,
    parse_adapter_output,
    parse_source_output,
)
from vaxreplay.operations.hermetic_callback_worker import (
    run_adapter_worker,
    run_source_verifier_worker,
)
from vaxreplay.operations.hermetic_execution import (
    HermeticExecutionRequest,
    HermeticExecutionResponse,
    HermeticMaterialBinding,
)
from vaxreplay.operations.promotion import AdapterRunResult, SourceVerifierRunResult
from vaxreplay.operations.promotion_schema import (
    AuthoritativeReleaseBasis,
    AuthoritativeSourceRelease,
    NormalizedRecordReference,
    PromotedCaptureBinding,
    PromotedRawArtifactBinding,
    PromotionFileBinding,
    SourceRecordBinding,
    SourceRecordDisposition,
    SourceVerificationResult,
    SourceVerifierIdentity,
)
from vaxreplay.operations.schema import CaptureJobSpec, job_spec_sha256, scheduled_logical_run_id

_T0 = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
_SOURCE_ID = 'publisher:worker-test'
_POLICY = b'{"policy":"fixture"}'
_IMPLEMENTATION = b'fixture combined worker implementation'
_ENVIRONMENT = b'fixture execution environment'


def _binding(payload: bytes) -> HermeticMaterialBinding:
    return HermeticMaterialBinding(sha256=hashlib.sha256(payload).hexdigest(), byte_count=len(payload))


def _capture() -> tuple[HermeticCapturePayload, bytes]:
    payload = b'{"released_at":"2026-07-14T12:00:00Z"}'
    job = CaptureJobSpec(
        job_id='worker-test-job',
        collector_id='static-https-v0.1',
        schedule_anchor_at=_T0,
        schedule_interval_seconds=86400,
    )
    job_sha256 = job_spec_sha256(job)
    artifact = PromotedRawArtifactBinding(
        role='body.document',
        file=PromotionFileBinding(
            path='raw/000000/body.document.bin',
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
        ),
        first_recorded_at=_T0 + timedelta(seconds=3),
        stored_event_sequence=4,
        stored_event_sha256='4' * 64,
        attached_event_sequence=5,
        attached_event_sha256='5' * 64,
    )
    binding = PromotedCaptureBinding(
        source_id=_SOURCE_ID,
        attempt_id=f'attempt-{"1" * 32}',
        logical_run_id=scheduled_logical_run_id(job_sha256, _T0),
        job_id=job.job_id,
        collector_id=job.collector_id,
        job_spec=job,
        job_spec_sha256=job_sha256,
        scheduled_for=_T0,
        attempt_started_at=_T0 + timedelta(seconds=2),
        captured_at=_T0 + timedelta(seconds=6),
        job_registered_event_sequence=1,
        job_registered_event_sha256='1' * 64,
        run_registered_event_sequence=2,
        run_registered_event_sha256='2' * 64,
        started_event_sequence=3,
        started_event_sha256='3' * 64,
        succeeded_event_sequence=6,
        succeeded_event_sha256='6' * 64,
        artifacts=(artifact,),
    )
    return (
        HermeticCapturePayload(
            binding=binding,
            artifacts=(HermeticArtifactPayload(binding=artifact, payload_base64=encode_callback_bytes(payload)),),
        ),
        payload,
    )


def _request(*, purpose: str, input_bytes: bytes, invocation_index: int = 0) -> bytes:
    request = HermeticExecutionRequest(
        invocation_id='fixture-worker-invocation',
        invocation_index=invocation_index,
        purpose=purpose,
        implementation=_binding(_IMPLEMENTATION),
        execution_environment=_binding(_ENVIRONMENT),
        callback_policy=_binding(_POLICY),
        sandbox_policy_sha256='a' * 64,
        input=_binding(input_bytes),
        callback_policy_base64=base64.b64encode(_POLICY).decode('ascii'),
        input_base64=base64.b64encode(input_bytes).decode('ascii'),
    )
    return canonical_json_bytes(request)


def _source_output_fixture(capture: HermeticCapturePayload, raw: bytes):
    source_record = SourceRecordBinding(
        source_id=_SOURCE_ID,
        source_record_id='release-record',
        source_record_sha256=hashlib.sha256(raw).hexdigest(),
        source_artifact_sha256=hashlib.sha256(raw).hexdigest(),
        source_locator='https://publisher.example/release.json#/record',
    )
    records = canonical_json_bytes(source_record) + b'\n'
    result = SourceVerificationResult(
        source_id=_SOURCE_ID,
        verifier=SourceVerifierIdentity(
            verifier_id='fixture-hermetic-source-verifier',
            verifier_version='v1',
            implementation_sha256=hashlib.sha256(_IMPLEMENTATION).hexdigest(),
            execution_environment_sha256=hashlib.sha256(_ENVIRONMENT).hexdigest(),
        ),
        verifier_policy_sha256=hashlib.sha256(_POLICY).hexdigest(),
        verified_attempt_ids=(capture.binding.attempt_id,),
        source_release=AuthoritativeSourceRelease(
            source_release_at=_T0,
            basis=AuthoritativeReleaseBasis.SOURCE_API_PUBLICATION_FIELD,
            authority_locator='https://publisher.example/release.json#/released_at',
            authority_field='publication_date',
            evidence_attempt_id=capture.binding.attempt_id,
            evidence_role='body.document',
            evidence_sha256=hashlib.sha256(raw).hexdigest(),
            evidence_source_record_id=source_record.source_record_id,
            evidence_source_record_sha256=source_record.source_record_sha256,
        ),
        verified_capture_inventory_sha256='c' * 64,
        verified_source_record_inventory_sha256=hashlib.sha256(records).hexdigest(),
        verified_source_record_count=1,
        result_codes=('complete_fixture_inventory',),
    )
    return source_record, records, result


def test_source_worker_round_trip_binds_standard_output() -> None:
    capture, raw = _capture()
    inner = HermeticSourceVerifierInput(
        source_id=_SOURCE_ID,
        capture_inventory_sha256='c' * 64,
        captures=(capture,),
    )
    source_record, records, result = _source_output_fixture(capture, raw)

    def worker(source_input, policy_bytes, *, implementation_sha256, execution_environment_sha256):
        assert source_input.captures[0].artifacts[0].payload == raw
        assert policy_bytes == _POLICY
        assert implementation_sha256 == hashlib.sha256(_IMPLEMENTATION).hexdigest()
        assert execution_environment_sha256 == hashlib.sha256(_ENVIRONMENT).hexdigest()
        return SourceVerifierRunResult(result=result, verified_records=records)

    request_bytes = _request(purpose='source_verifier', input_bytes=canonical_json_bytes(inner))
    response_bytes = run_source_verifier_worker(request_bytes, worker)
    response = HermeticExecutionResponse.model_validate_json(response_bytes)
    output = parse_source_output(decode_callback_bytes(response.output_base64))

    assert output.result == result
    assert decode_callback_bytes(output.verified_records_base64) == records
    assert source_record.source_record_id in records.decode('utf-8')


def test_adapter_worker_round_trip_and_artifact_tamper_rejection() -> None:
    capture, raw = _capture()
    source_record, records, verification = _source_output_fixture(capture, raw)
    adapter_input = HermeticAdapterInput(
        sources=(
            HermeticAdapterSourceInput(
                source_id=_SOURCE_ID,
                captures=(capture,),
                verification_result=verification,
                verified_records_base64=encode_callback_bytes(records),
            ),
        )
    )
    candidate = CandidateRecord(episode_id='episode-hermetic', candidate_id='candidate-a')
    body = 'Cutoff-safe fixture evidence.'
    evidence = EvidenceRecord(
        episode_id='episode-hermetic',
        evidence_id='evidence-a',
        source_type=SourceType.PUBLIC_HEALTH,
        available_at=_T0,
        title='Fixture evidence',
        body=body,
        body_sha256=hashlib.sha256(body.encode()).hexdigest(),
        related_candidate_ids=['candidate-a'],
        provenance_url=source_record.source_locator,
        license_id='public-domain',
        derivation='hermetic fixture adapter',
    )
    disposition = SourceRecordDisposition(
        source_id=_SOURCE_ID,
        source_record_id=source_record.source_record_id,
        source_record_sha256=source_record.source_record_sha256,
        source_artifact_sha256=source_record.source_artifact_sha256,
        disposition='normalized',
        candidate_record_refs=(
            NormalizedRecordReference(
                episode_id=candidate.episode_id,
                record_id=candidate.candidate_id,
                record_sha256=hashlib.sha256(canonical_json_bytes(candidate)).hexdigest(),
            ),
        ),
        evidence_record_refs=(
            NormalizedRecordReference(
                episode_id=evidence.episode_id,
                record_id=evidence.evidence_id,
                record_sha256=hashlib.sha256(canonical_json_bytes(evidence)).hexdigest(),
            ),
        ),
    )

    def adapter(inputs, policy_bytes):
        assert inputs[0].captures[0].artifacts[0].payload == raw
        assert inputs[0].verified_records == (source_record,)
        assert policy_bytes == _POLICY
        return AdapterRunResult(
            candidate_records=canonical_json_bytes(candidate) + b'\n',
            evidence_records=canonical_json_bytes(evidence) + b'\n',
            dispositions=canonical_json_bytes(disposition) + b'\n',
            auxiliary_outputs={'mapping': b'{"candidate-a":"release-record"}'},
        )

    request_bytes = _request(purpose='adapter', input_bytes=canonical_json_bytes(adapter_input), invocation_index=1)
    response_bytes = run_adapter_worker(request_bytes, adapter)
    response = HermeticExecutionResponse.model_validate_json(response_bytes)
    output = parse_adapter_output(decode_callback_bytes(response.output_base64))
    assert decode_callback_bytes(output.candidate_records_base64) == canonical_json_bytes(candidate) + b'\n'
    assert output.auxiliary_outputs[0].name == 'mapping'

    with pytest.raises(ValidationError, match='embedded artifact bytes'):
        HermeticArtifactPayload(
            binding=capture.artifacts[0].binding,
            payload_base64=encode_callback_bytes(b'tampered'),
        )
    with pytest.raises(HermeticCallbackProtocolError, match='canonical'):
        parse_adapter_output(decode_callback_bytes(response.output_base64) + b'\n')
