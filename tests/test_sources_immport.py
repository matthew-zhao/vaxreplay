from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord
from vaxreplay.operations.hermetic_callback_protocol import (
    HermeticAdapterInput,
    HermeticAdapterSourceInput,
    HermeticArtifactPayload,
    HermeticCapturePayload,
    HermeticSourceVerifierInput,
    decode_callback_bytes,
    encode_callback_bytes,
    parse_adapter_output,
    parse_source_output,
)
from vaxreplay.operations.hermetic_execution import (
    HermeticExecutionRequest,
    HermeticExecutionResponse,
    HermeticMaterialBinding,
)
from vaxreplay.operations.promotion import (
    AdapterRunResult,
    AdapterSourceInput,
    ExactPromotedArtifact,
    ExactPromotedCapture,
    SourceVerificationInput,
    _normalize_adapter_result,
)
from vaxreplay.operations.promotion_schema import (
    PromotedCaptureBinding,
    PromotedRawArtifactBinding,
    PromotionFileBinding,
    SourceRecordBinding,
    SourceRecordDisposition,
)
from vaxreplay.operations.schema import (
    CaptureJobSpec,
    job_spec_sha256,
    scheduled_logical_run_id,
)
from vaxreplay.sources import worker_cli
from vaxreplay.sources.immport import (
    IMMPORT_ARM_ADAPTER_EXCLUSION_REASON_CODES,
    ImmportArmAdapterPolicy,
    ImmportArmCandidateMap,
    ImmportProductionAdapterError,
    ImmportProductionSourceError,
    ImmportPromotionLayout,
    ImmportSanitizedCaptureReceipt,
    ImmportSourceVerifierPolicy,
    ImmportTlsPeerBinding,
    _verify_receipt,
    adapt_tier_a_immport_arms,
    immport_arm_adapter_policy_bytes,
    immport_source_verifier_policy_bytes,
    verify_tier_a_immport_source,
)

_SOURCE_ID = 'immport:prospective-shared-data'
_STUDY = 'SDY00000000'
_CAPTURED_AT = datetime(2026, 6, 27, 12, tzinfo=timezone.utc)
_OPENAPI_URL = 'https://www.immport.org/data/query/v3/api-docs'


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


def _endpoint_url(kind: str) -> str:
    suffix = {
        'study': f'study/{_STUDY}?format=json',
        'arm': f'study/arm/{_STUDY}?format=json',
        'experiment': f'study/experiment/{_STUDY}?format=json',
        'link': f'study/link/{_STUDY}?format=json',
        'manifest': f'study/manifest/{_STUDY}?fileType=release_file&format=json',
    }[kind]
    return f'https://www.immport.org/data/query/api/{suffix}'


def _openapi() -> dict[str, Any]:
    schema_names = {
        '/api/study/{studyAccession}': 'StudyApi',
        '/api/study/arm/{studyAccession}': 'StudyArmApi',
        '/api/study/experiment/{studyAccession}': 'StudyExperimentApi',
        '/api/study/link/{studyAccession}': 'StudyLinkApi',
        '/api/study/manifest/{studyAccession}': 'FileDetails',
    }
    properties = {
        'StudyApi': {
            'studyAccession': {},
            'clinicalTrial': {},
            'sharedStudy': {},
            'latestDataReleaseVersion': {},
            'latestDataReleaseDate': {},
            'briefDescription': {},
            'briefTitle': {},
            'doi': {},
            'endpoints': {},
            'hypothesis': {},
            'interventionAgent': {},
            'objectives': {},
        },
        'StudyArmApi': {
            'armAccession': {},
            'studyAccession': {},
            'description': {},
            'name': {},
            'typeReported': {},
            'typePreferred': {},
        },
        'StudyExperimentApi': {
            'experimentAccession': {},
            'studyAccession': {},
            'measurementTechnique': {},
            'description': {},
            'name': {},
        },
        'StudyLinkApi': {
            'studyLinkId': {},
            'studyAccession': {},
            'name': {},
            'type': {},
            'value': {},
        },
        'FileDetails': {
            'fileDetailsId': {},
            'workspaceId': {},
            'reportedMD5': {},
            'generatedMD5': {},
            'studyAccession': {},
            'filesizeBytes': {},
            'fileType': {},
            'fileAccession': {},
            'fileName': {},
            'path': {},
            'dateFileUpdated': {},
            'fileUUID': {},
            'drsObjectCreated': {},
            'aigeneratedKeywords': {},
            'aigeneratedSummary': {},
        },
    }
    return {
        'openapi': '3.0.1',
        'info': {'title': 'Shared Data API', 'version': 'v1'},
        'paths': {
            path: {
                'get': {
                    'security': [{'immport-security': []}],
                    'responses': {
                        '200': {
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'array',
                                        'items': {'$ref': f'#/components/schemas/{schema_name}'},
                                    }
                                }
                            }
                        }
                    },
                }
            }
            for path, schema_name in schema_names.items()
        },
        'components': {
            'schemas': {name: {'type': 'object', 'properties': values} for name, values in properties.items()}
        },
    }


def _layout() -> ImmportPromotionLayout:
    return ImmportPromotionLayout(
        study_accession=_STUDY,
        openapi_before_artifact_id='a-openapi-before',
        study_before_artifact_id='b-study-before',
        manifest_before_artifact_id='c-manifest-before',
        arm_artifact_id='d-arm',
        experiment_artifact_id='e-experiment',
        link_artifact_id='f-link',
        manifest_after_artifact_id='x-manifest-after',
        study_after_artifact_id='y-study-after',
        openapi_after_artifact_id='z-openapi-after',
    )


def _receipt(payload: bytes, url: str, *, start: int, end: int) -> bytes:
    authenticated = url != _OPENAPI_URL
    receipt = ImmportSanitizedCaptureReceipt(
        requested_url=url,
        final_url=url,
        authentication=('immport_scoped_api_key_bearer_redacted' if authenticated else 'none'),
        authorization_applied=authenticated,
        credential_source='runtime_secret_broker' if authenticated else 'not_applicable',
        response_content_type='application/json;charset=UTF-8',
        body_sha256=hashlib.sha256(payload).hexdigest(),
        body_byte_count=len(payload),
        started_at=_CAPTURED_AT - timedelta(seconds=start),
        completed_at=_CAPTURED_AT - timedelta(seconds=end),
        tls_peer=ImmportTlsPeerBinding(
            tls_version='TLSv1.3',
            certificate_der_sha256='a' * 64,
        ),
        collector_id='immport-secret-broker-collector',
        collector_implementation_sha256='b' * 64,
        collector_execution_environment_sha256='c' * 64,
    )
    return canonical_json_bytes(receipt)


def _fixture(
    *,
    openapi_after: dict[str, Any] | None = None,
    shared_study: str = 'Y',
    release_version: str = 'DR65',
    release_date: str = '2026-06-25',
    arms: list[dict[str, Any]] | None = None,
    manifest_override: list[dict[str, Any]] | None = None,
    study_after_override: list[dict[str, Any]] | None = None,
    manifest_after_override: list[dict[str, Any]] | None = None,
):
    layout = _layout()
    contract = canonical_json_bytes(_openapi())
    study_rows = [
        {
            'studyAccession': _STUDY,
            'clinicalTrial': 'Y',
            'sharedStudy': shared_study,
            'latestDataReleaseVersion': release_version,
            'latestDataReleaseDate': release_date,
            'briefDescription': 'FUTURE STUDY RESULT CANARY',
            'briefTitle': 'IDENTITY TITLE CANARY',
            'doi': '10.0000/IDENTITY-CANARY',
            'endpoints': 'RESULT ENDPOINT CANARY',
            'hypothesis': 'HYPOTHESIS CANARY',
            'interventionAgent': 'Vaccine regimens',
            'objectives': 'OBJECTIVE CANARY',
        }
    ]
    study = canonical_json_bytes(study_rows)
    study_after = canonical_json_bytes(study_after_override) if study_after_override is not None else study
    arm_rows = arms or [
        {
            'armAccession': 'ARM1',
            'studyAccession': _STUDY,
            'description': (
                'Low-dose vaccine regimen from SDY00000000 / NCT12345678; '
                'see https://example.org/result and 10.1234/result-canary'
            ),
            'name': 'Low dose',
            'typeReported': 'Experimental',
            'typePreferred': 'Experimental Arm',
        },
        {
            'armAccession': 'ARM2',
            'studyAccession': _STUDY,
            'description': 'High-dose vaccine regimen',
            'name': 'High dose',
            'typeReported': 'Experimental',
            'typePreferred': 'Experimental Arm',
        },
        {
            'armAccession': 'ARM3',
            'studyAccession': _STUDY,
            'description': 'Saline comparator regimen',
            'name': 'Comparator',
            'typeReported': 'Placebo Comparator',
            'typePreferred': 'Placebo Comparator Arm',
        },
    ]
    arm = canonical_json_bytes(arm_rows)
    experiment = canonical_json_bytes(
        [
            {
                'experimentAccession': 'EXP1',
                'studyAccession': _STUDY,
                'measurementTechnique': 'ELISA',
                'description': 'FUTURE ASSAY RESULT CANARY',
                'name': 'EXPERIMENT IDENTITY CANARY',
            },
            {
                'experimentAccession': 'EXP2',
                'studyAccession': _STUDY,
                'measurementTechnique': 'HAI',
                'description': 'SECOND RESULT CANARY',
                'name': 'SECOND EXPERIMENT CANARY',
            },
        ]
    )
    link = canonical_json_bytes(
        [
            {
                'studyLinkId': 1,
                'studyAccession': _STUDY,
                'name': 'clinicaltrials.gov',
                'type': 'website',
                'value': 'https://clinicaltrials.gov/study/NCT12345678',
            },
            {
                'studyLinkId': 2,
                'studyAccession': _STUDY,
                'name': 'publication',
                'type': 'website',
                'value': 'https://example.org/current-result-canary',
            },
            {
                'studyLinkId': 3,
                'studyAccession': _STUDY,
                'name': 'clinicaltrials.gov',
                'type': 'website',
                'value': 'https://evil.example/results/NCT87654321',
            },
        ]
    )
    manifest_rows = (
        manifest_override
        if manifest_override is not None
        else [
            {
                'fileDetailsId': 1,
                'workspaceId': 2,
                'reportedMD5': None,
                'generatedMD5': '1' * 32,
                'studyAccession': _STUDY,
                'filesizeBytes': 100,
                'fileType': 'release_file',
                'fileAccession': 'SFL1',
                'fileName': f'{_STUDY}_{release_version}_Tab.zip',
                'path': f'{_STUDY}/{release_version}/{_STUDY}_{release_version}_Tab.zip',
                'dateFileUpdated': '2026-06-25T00:00:00Z',
                'fileUUID': '11111111-1111-4111-8111-111111111111',
                'drsObjectCreated': 'Y',
                'aigeneratedKeywords': 'MANIFEST KEYWORD CANARY',
                'aigeneratedSummary': 'MANIFEST RESULT CANARY',
            },
            {
                'fileDetailsId': 2,
                'workspaceId': 2,
                'reportedMD5': None,
                'generatedMD5': '2' * 32,
                'studyAccession': _STUDY,
                'filesizeBytes': 25,
                'fileType': 'release_file',
                'fileAccession': 'SFL2',
                'fileName': f'{_STUDY}_{release_version}_table_count.txt',
                'path': (f'{_STUDY}/{release_version}/{_STUDY}_{release_version}_table_count.txt'),
                'dateFileUpdated': '2026-06-25T00:00:00Z',
                'fileUUID': '22222222-2222-4222-8222-222222222222',
                'drsObjectCreated': 'Y',
                'aigeneratedKeywords': None,
                'aigeneratedSummary': None,
            },
        ]
    )
    manifest = canonical_json_bytes(manifest_rows)
    manifest_after = canonical_json_bytes(manifest_after_override) if manifest_after_override is not None else manifest
    openapi_after_bytes = canonical_json_bytes(openapi_after) if openapi_after is not None else contract
    bodies = {
        layout.openapi_before_artifact_id: (contract, _OPENAPI_URL),
        layout.study_before_artifact_id: (study, _endpoint_url('study')),
        layout.manifest_before_artifact_id: (manifest, _endpoint_url('manifest')),
        layout.arm_artifact_id: (arm, _endpoint_url('arm')),
        layout.experiment_artifact_id: (experiment, _endpoint_url('experiment')),
        layout.link_artifact_id: (link, _endpoint_url('link')),
        layout.manifest_after_artifact_id: (manifest_after, _endpoint_url('manifest')),
        layout.study_after_artifact_id: (study_after, _endpoint_url('study')),
        layout.openapi_after_artifact_id: (openapi_after_bytes, _OPENAPI_URL),
    }
    order = {
        layout.openapi_before_artifact_id: (18, 17),
        layout.study_before_artifact_id: (16, 15),
        layout.manifest_before_artifact_id: (14, 13),
        layout.arm_artifact_id: (12, 11),
        layout.experiment_artifact_id: (10, 9),
        layout.link_artifact_id: (8, 7),
        layout.manifest_after_artifact_id: (6, 5),
        layout.study_after_artifact_id: (4, 3),
        layout.openapi_after_artifact_id: (2, 1),
    }
    artifacts: list[_Artifact] = []
    for artifact_id, (payload, url) in bodies.items():
        artifacts.append(_Artifact(f'body.{artifact_id}', payload))
        start, end = order[artifact_id]
        artifacts.append(_Artifact(f'receipt.{artifact_id}', _receipt(payload, url, start=start, end=end)))
    capture = cast(
        ExactPromotedCapture,
        SimpleNamespace(
            binding=SimpleNamespace(
                source_id=_SOURCE_ID,
                attempt_id=f'attempt-{"1" * 32}',
                captured_at=_CAPTURED_AT,
            ),
            artifacts=tuple(artifacts),
        ),
    )
    verifier_input = SourceVerificationInput(
        source_id=_SOURCE_ID,
        captures=(capture,),
        capture_inventory_sha256='2' * 64,
    )
    policy = ImmportSourceVerifierPolicy(
        policy_id='immport-study-panel-v1',
        source_id=_SOURCE_ID,
        study_universe_registry_sha256='5' * 64,
        layout=layout,
        expected_openapi_sha256=hashlib.sha256(contract).hexdigest(),
        expected_openapi_info_version='v1',
        expected_latest_release_version=release_version,
        expected_latest_release_date=date(2026, 6, 25),
        expected_collector_id='immport-secret-broker-collector',
        expected_collector_implementation_sha256='b' * 64,
        expected_collector_execution_environment_sha256='c' * 64,
    )
    return policy, verifier_input, capture


def _adapter_fixture():
    source_policy, verifier_input, capture = _fixture()
    verified = verify_tier_a_immport_source(
        verifier_input,
        immport_source_verifier_policy_bytes(source_policy),
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
    adapter_policy = ImmportArmAdapterPolicy(
        policy_id='immport-complete-arm-panel-v1',
        source_id=_SOURCE_ID,
        episode_id='prospective-early-clinical-001',
        study_accession=_STUDY,
        study_universe_registry_sha256='5' * 64,
        outcome_adjudication_spec_sha256='6' * 64,
        decision_at=datetime(2026, 6, 28, tzinfo=timezone.utc),
    )
    return adapter_input, adapter_policy


def _material_binding(payload: bytes) -> HermeticMaterialBinding:
    return HermeticMaterialBinding(
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
    )


def _real_promoted_capture(
    fixture_capture: ExactPromotedCapture,
) -> tuple[ExactPromotedCapture, HermeticCapturePayload]:
    scheduled_for = _CAPTURED_AT - timedelta(days=1)
    job = CaptureJobSpec(
        job_id='immport-canonical-wire-smoke',
        collector_id='immport-secret-broker-collector',
        schedule_anchor_at=scheduled_for,
        schedule_interval_seconds=24 * 60 * 60,
    )
    spec_sha256 = job_spec_sha256(job)
    promoted_artifacts: list[ExactPromotedArtifact] = []
    for ordinal, artifact in enumerate(
        sorted(fixture_capture.artifacts, key=lambda item: item.role),
        start=1,
    ):
        binding = PromotedRawArtifactBinding(
            role=artifact.role,
            file=PromotionFileBinding(
                path=f'raw/000000/{artifact.role}.bin',
                sha256=artifact.sha256,
                byte_count=artifact.byte_count,
            ),
            first_recorded_at=_CAPTURED_AT - timedelta(seconds=1),
            stored_event_sequence=10 + ordinal * 2,
            stored_event_sha256=hashlib.sha256(f'stored:{artifact.role}'.encode('utf-8')).hexdigest(),
            attached_event_sequence=11 + ordinal * 2,
            attached_event_sha256=hashlib.sha256(f'attached:{artifact.role}'.encode('utf-8')).hexdigest(),
        )
        promoted_artifacts.append(ExactPromotedArtifact(binding=binding, payload=artifact.payload))
    artifact_bindings = tuple(item.binding for item in promoted_artifacts)
    capture_binding = PromotedCaptureBinding(
        source_id=_SOURCE_ID,
        attempt_id=f'attempt-{"1" * 32}',
        logical_run_id=scheduled_logical_run_id(spec_sha256, scheduled_for),
        job_id=job.job_id,
        collector_id=job.collector_id,
        job_spec=job,
        job_spec_sha256=spec_sha256,
        scheduled_for=scheduled_for,
        attempt_started_at=_CAPTURED_AT - timedelta(seconds=30),
        captured_at=_CAPTURED_AT,
        job_registered_event_sequence=1,
        job_registered_event_sha256='1' * 64,
        run_registered_event_sequence=2,
        run_registered_event_sha256='2' * 64,
        started_event_sequence=3,
        started_event_sha256='3' * 64,
        succeeded_event_sequence=100,
        succeeded_event_sha256='4' * 64,
        artifacts=artifact_bindings,
    )
    exact_capture = ExactPromotedCapture(
        binding=capture_binding,
        artifacts=tuple(promoted_artifacts),
    )
    hermetic_capture = HermeticCapturePayload(
        binding=capture_binding,
        artifacts=tuple(
            HermeticArtifactPayload(
                binding=artifact.binding,
                payload_base64=encode_callback_bytes(artifact.payload),
            )
            for artifact in promoted_artifacts
        ),
    )
    return exact_capture, hermetic_capture


def _capture_inventory_sha256(capture: ExactPromotedCapture) -> str:
    inventory = (
        {
            'artifacts': tuple(
                {
                    'byte_count': artifact.byte_count,
                    'role': artifact.role,
                    'sha256': artifact.sha256,
                }
                for artifact in capture.artifacts
            ),
            'attempt_id': capture.binding.attempt_id,
            'logical_run_id': capture.binding.logical_run_id,
            'scheduled_for': capture.binding.scheduled_for.isoformat().replace('+00:00', 'Z'),
            'source_id': capture.binding.source_id,
        },
    )
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def test_canonical_wire_round_trip_normalizes_every_immport_source_record() -> None:
    source_policy, _verifier_input, fixture_capture = _fixture()
    capture, hermetic_capture = _real_promoted_capture(fixture_capture)
    source_policy_bytes = immport_source_verifier_policy_bytes(source_policy)
    source_implementation = b'reviewed ImmPort source verifier implementation'
    execution_environment = b'python runtime lock for ImmPort canonical-wire smoke'
    source_inner_bytes = canonical_json_bytes(
        HermeticSourceVerifierInput(
            source_id=_SOURCE_ID,
            capture_inventory_sha256=_capture_inventory_sha256(capture),
            captures=(hermetic_capture,),
        )
    )
    source_request = HermeticExecutionRequest(
        invocation_id='immport-source-verifier-smoke',
        invocation_index=0,
        purpose='source_verifier',
        implementation=_material_binding(source_implementation),
        execution_environment=_material_binding(execution_environment),
        callback_policy=_material_binding(source_policy_bytes),
        sandbox_policy_sha256='5' * 64,
        input=_material_binding(source_inner_bytes),
        callback_policy_base64=encode_callback_bytes(source_policy_bytes),
        input_base64=encode_callback_bytes(source_inner_bytes),
    )
    source_request_bytes = canonical_json_bytes(source_request)
    source_response_bytes = worker_cli.dispatch(
        'immport-source-verifier',
        source_request_bytes,
    )
    source_response = HermeticExecutionResponse.model_validate_json(source_response_bytes)
    assert canonical_json_bytes(source_response) == source_response_bytes
    assert source_response.request_sha256 == hashlib.sha256(source_request_bytes).hexdigest()
    source_output = parse_source_output(decode_callback_bytes(source_response.output_base64))
    verified_records_bytes = decode_callback_bytes(source_output.verified_records_base64)
    verified_records = tuple(
        SourceRecordBinding.model_validate_json(line) for line in verified_records_bytes.splitlines()
    )
    assert len(verified_records) == source_output.result.verified_source_record_count == 12
    assert source_output.result.verifier.implementation_sha256 == hashlib.sha256(source_implementation).hexdigest()
    assert (
        source_output.result.verifier.execution_environment_sha256 == hashlib.sha256(execution_environment).hexdigest()
    )

    adapter_policy = ImmportArmAdapterPolicy(
        policy_id='immport-complete-arm-panel-v1',
        source_id=_SOURCE_ID,
        episode_id='prospective-early-clinical-001',
        study_accession=_STUDY,
        study_universe_registry_sha256='5' * 64,
        outcome_adjudication_spec_sha256='6' * 64,
        decision_at=datetime(2026, 6, 28, tzinfo=timezone.utc),
    )
    adapter_policy_bytes = immport_arm_adapter_policy_bytes(adapter_policy)
    adapter_inner_bytes = canonical_json_bytes(
        HermeticAdapterInput(
            sources=(
                HermeticAdapterSourceInput(
                    source_id=_SOURCE_ID,
                    captures=(hermetic_capture,),
                    verification_result=source_output.result,
                    verified_records_base64=encode_callback_bytes(verified_records_bytes),
                ),
            )
        )
    )
    adapter_request = HermeticExecutionRequest(
        invocation_id='immport-arm-adapter-smoke',
        invocation_index=1,
        purpose='adapter',
        implementation=_material_binding(b'reviewed ImmPort arm adapter implementation'),
        execution_environment=_material_binding(execution_environment),
        callback_policy=_material_binding(adapter_policy_bytes),
        sandbox_policy_sha256='5' * 64,
        input=_material_binding(adapter_inner_bytes),
        callback_policy_base64=encode_callback_bytes(adapter_policy_bytes),
        input_base64=encode_callback_bytes(adapter_inner_bytes),
    )
    adapter_request_bytes = canonical_json_bytes(adapter_request)
    adapter_response_bytes = worker_cli.dispatch(
        'immport-arm-adapter',
        adapter_request_bytes,
    )
    adapter_response = HermeticExecutionResponse.model_validate_json(adapter_response_bytes)
    assert canonical_json_bytes(adapter_response) == adapter_response_bytes
    assert adapter_response.request_sha256 == hashlib.sha256(adapter_request_bytes).hexdigest()
    adapter_output = parse_adapter_output(decode_callback_bytes(adapter_response.output_base64))
    adapter_result = AdapterRunResult(
        candidate_records=decode_callback_bytes(adapter_output.candidate_records_base64),
        evidence_records=decode_callback_bytes(adapter_output.evidence_records_base64),
        dispositions=decode_callback_bytes(adapter_output.dispositions_base64),
        auxiliary_outputs={
            item.name: decode_callback_bytes(item.payload_base64) for item in adapter_output.auxiliary_outputs
        },
    )
    adapter_source_input = AdapterSourceInput(
        source_id=_SOURCE_ID,
        captures=(capture,),
        verification_result=source_output.result,
        verified_records=verified_records,
    )
    _outputs, _disposition_bytes, dispositions = _normalize_adapter_result(
        adapter_result,
        (adapter_source_input,),
        IMMPORT_ARM_ADAPTER_EXCLUSION_REASON_CODES,
    )
    assert len(adapter_result.candidate_records.splitlines()) == 3
    assert len(dispositions) == len(verified_records) == 12
    assert {(item.source_id, item.source_record_id) for item in dispositions} == {
        (item.source_id, item.source_record_id) for item in verified_records
    }
    assert tuple(item.name for item in adapter_output.auxiliary_outputs) == (
        'immport-arm-candidate-map',
        'immport-candidate-set-definition',
    )


def test_source_verifier_binds_stable_contract_release_manifest_and_every_row() -> None:
    policy, verifier_input, _capture = _fixture()
    result = verify_tier_a_immport_source(
        verifier_input,
        immport_source_verifier_policy_bytes(policy),
        implementation_sha256='3' * 64,
        execution_environment_sha256='4' * 64,
    )

    assert result.result.verified_source_record_count == 12
    assert result.result.source_release.source_release_at == datetime(
        2026,
        6,
        26,
        3,
        59,
        59,
        999999,
        tzinfo=timezone.utc,
    )
    assert any('not_scientific_change' in code for code in result.result.result_codes)
    records = tuple(SourceRecordBinding.model_validate_json(line) for line in result.verified_records.splitlines())
    assert sum(item.source_record_id.startswith('arm:') for item in records) == 3
    assert sum(item.source_record_id.startswith('manifest:') for item in records) == 2


def test_adapter_emits_every_arm_and_blinds_direct_id_and_result_canaries() -> None:
    adapter_input, policy = _adapter_fixture()
    result = adapt_tier_a_immport_arms(
        (adapter_input,),
        immport_arm_adapter_policy_bytes(policy),
    )

    candidates = tuple(CandidateRecord.model_validate_json(line) for line in result.candidate_records.splitlines())
    evidence = tuple(EvidenceRecord.model_validate_json(line) for line in result.evidence_records.splitlines())
    dispositions = tuple(SourceRecordDisposition.model_validate_json(line) for line in result.dispositions.splitlines())
    assert len(candidates) == 3
    assert len(evidence) == 3
    assert all(item.candidate_id.startswith('cand-immport-') for item in candidates)
    public_bytes = result.candidate_records + result.evidence_records
    for canary in (
        _STUDY,
        'ARM1',
        'NCT12345678',
        'IDENTITY TITLE CANARY',
        'FUTURE STUDY RESULT CANARY',
        'FUTURE ASSAY RESULT CANARY',
        'MANIFEST RESULT CANARY',
        '10.0000/IDENTITY-CANARY',
        '10.1234/result-canary',
        'https://example.org/result',
    ):
        assert canary.encode('utf-8') not in public_bytes
    assert b'ELISA' in result.evidence_records
    assert b'HAI' in result.evidence_records
    assert b'[source identifier redacted]' in result.evidence_records
    assert sum(item.disposition == 'normalized' for item in dispositions) == 7
    assert {item.reason_code for item in dispositions if item.disposition == 'excluded'} == {
        'non_exact_clinical_trial_registry_link',
        'source_metadata_record',
    }
    assert result.auxiliary_outputs is not None
    candidate_map = ImmportArmCandidateMap.model_validate_json(result.auxiliary_outputs['immport-arm-candidate-map'])
    assert candidate_map.organizer_only is True
    assert {item.arm_accession for item in candidate_map.candidates} == {'ARM1', 'ARM2', 'ARM3'}
    assert {item.nct_ids for item in candidate_map.candidates} == {('NCT12345678',)}
    assert {item.arm_role for item in candidate_map.candidates} == {'intervention', 'control'}
    assert sum(item.eligible for item in candidates) == 2
    _normalize_adapter_result(
        result,
        (adapter_input,),
        IMMPORT_ARM_ADAPTER_EXCLUSION_REASON_CODES,
    )


def test_adapter_candidate_universe_is_order_independent_and_includes_control_arms() -> None:
    rows = [
        {
            'armAccession': accession,
            'studyAccession': _STUDY,
            'description': description,
            'name': name,
            'typeReported': kind,
            'typePreferred': f'{kind} Arm',
        }
        for accession, description, name, kind in (
            ('ARM9', 'Comparator', 'Control', 'Placebo Comparator'),
            ('ARM8', 'Candidate B', 'B', 'Experimental'),
            ('ARM7', 'Candidate A', 'A', 'Experimental'),
        )
    ]
    first_policy, first_input, first_capture = _fixture(arms=rows)
    second_policy, second_input, second_capture = _fixture(arms=list(reversed(rows)))

    def adapt(source_policy, verifier_input, capture):
        verified = verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(source_policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )
        source = AdapterSourceInput(
            source_id=_SOURCE_ID,
            captures=(capture,),
            verification_result=verified.result,
            verified_records=tuple(
                SourceRecordBinding.model_validate_json(line) for line in verified.verified_records.splitlines()
            ),
        )
        policy = ImmportArmAdapterPolicy(
            policy_id='immport-complete-arm-panel-v1',
            source_id=_SOURCE_ID,
            episode_id='prospective-early-clinical-001',
            study_accession=_STUDY,
            study_universe_registry_sha256='5' * 64,
            outcome_adjudication_spec_sha256='6' * 64,
            decision_at=datetime(2026, 6, 28, tzinfo=timezone.utc),
        )
        return adapt_tier_a_immport_arms((source,), immport_arm_adapter_policy_bytes(policy))

    first = adapt(first_policy, first_input, first_capture)
    second = adapt(second_policy, second_input, second_capture)
    assert first.candidate_records == second.candidate_records
    assert first.evidence_records == second.evidence_records
    # Dispositions intentionally bind the raw artifact hash, so publisher row order
    # changes that provenance output even though normalized public rows stay stable.
    assert first.auxiliary_outputs == second.auxiliary_outputs


def test_source_verifier_rejects_openapi_drift_during_capture() -> None:
    changed = _openapi()
    changed['info'] = {'title': 'Shared Data API', 'version': 'v2'}
    policy, verifier_input, _capture = _fixture(openapi_after=changed)
    with pytest.raises(ImmportProductionSourceError, match='changed during capture'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_release_identity_drift_during_capture() -> None:
    policy, verifier_input, _capture = _fixture(
        study_after_override=[
            {
                'studyAccession': _STUDY,
                'clinicalTrial': 'Y',
                'sharedStudy': 'Y',
                'latestDataReleaseVersion': 'DR66',
                'latestDataReleaseDate': '2026-07-30',
            }
        ]
    )
    with pytest.raises(ImmportProductionSourceError, match='study bytes changed during capture'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_manifest_drift_during_capture() -> None:
    policy, verifier_input, _capture = _fixture(manifest_after_override=[])
    with pytest.raises(ImmportProductionSourceError, match='manifest bytes changed during capture'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_release_identifier_prefix_match() -> None:
    near_collision_manifest = [
        {
            'fileDetailsId': 1,
            'workspaceId': 2,
            'reportedMD5': None,
            'generatedMD5': '1' * 32,
            'studyAccession': _STUDY,
            'filesizeBytes': 100,
            'fileType': 'release_file',
            'fileAccession': 'SFL1',
            'fileName': f'{_STUDY}_DR65_Tab.zip',
            'path': f'{_STUDY}/DR65/{_STUDY}_DR65_Tab.zip',
            'dateFileUpdated': '2026-06-25T00:00:00Z',
            'fileUUID': '11111111-1111-4111-8111-111111111111',
            'drsObjectCreated': 'Y',
            'aigeneratedKeywords': None,
            'aigeneratedSummary': None,
        }
    ]
    policy, verifier_input, _capture = _fixture(
        release_version='DR6',
        manifest_override=near_collision_manifest,
    )
    with pytest.raises(ImmportProductionSourceError, match='does not bind'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_any_uncommitted_artifact_role() -> None:
    policy, verifier_input, capture = _fixture()
    injected = cast(
        ExactPromotedCapture,
        SimpleNamespace(
            binding=capture.binding,
            artifacts=(*capture.artifacts, _Artifact('participant-results-secret', b'Bearer SECRET')),
        ),
    )
    with pytest.raises(ImmportProductionSourceError, match='exact source and structural role set'):
        verify_tier_a_immport_source(
            SourceVerificationInput(
                source_id=_SOURCE_ID,
                captures=(injected,),
                capture_inventory_sha256=verifier_input.capture_inventory_sha256,
            ),
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_restricted_study() -> None:
    policy, verifier_input, _capture = _fixture(shared_study='N')
    with pytest.raises(ImmportProductionSourceError, match='only shared studies'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_release_upper_bound_after_capture() -> None:
    policy, verifier_input, _capture = _fixture(release_date='2026-06-28')
    policy = policy.model_copy(update={'expected_latest_release_date': date(2026, 6, 28)})
    with pytest.raises(ImmportProductionSourceError, match='upper bound is after'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_self_reported_uncommitted_collector() -> None:
    policy, verifier_input, _capture = _fixture()
    policy = policy.model_copy(update={'expected_collector_id': 'different-reviewed-collector'})
    with pytest.raises(ImmportProductionSourceError, match='committed collector'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_duplicate_arm_identity() -> None:
    duplicate = {
        'armAccession': 'ARM1',
        'studyAccession': _STUDY,
        'description': 'Duplicate regimen',
        'name': 'Duplicate',
        'typeReported': 'Experimental',
        'typePreferred': 'Experimental Arm',
    }
    policy, verifier_input, _capture = _fixture(
        arms=[
            duplicate,
            duplicate | {'description': 'Different bytes, same accession'},
        ]
    )
    with pytest.raises(ImmportProductionSourceError, match='duplicate armAccession'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_source_verifier_rejects_unsafe_or_unchecksummed_manifest() -> None:
    bad_manifest = [
        {
            'fileDetailsId': 1,
            'workspaceId': 2,
            'reportedMD5': None,
            'generatedMD5': None,
            'studyAccession': _STUDY,
            'filesizeBytes': 100,
            'fileType': 'release_file',
            'fileAccession': 'SFL1',
            'fileName': 'payload.zip',
            'path': '../payload.zip',
            'dateFileUpdated': '2026-06-25T00:00:00Z',
            'fileUUID': '11111111-1111-4111-8111-111111111111',
            'drsObjectCreated': 'Y',
            'aigeneratedKeywords': None,
            'aigeneratedSummary': None,
        }
    ]
    policy, verifier_input, _capture = _fixture(manifest_override=bad_manifest)
    with pytest.raises(ImmportProductionSourceError, match='generated MD5'):
        verify_tier_a_immport_source(
            verifier_input,
            immport_source_verifier_policy_bytes(policy),
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )


def test_receipt_schema_makes_bearer_value_and_presigned_url_unrepresentable() -> None:
    payload = canonical_json_bytes([])
    raw = {
        **ImmportSanitizedCaptureReceipt(
            requested_url=_endpoint_url('arm'),
            final_url=_endpoint_url('arm'),
            authentication='immport_scoped_api_key_bearer_redacted',
            authorization_applied=True,
            credential_source='runtime_secret_broker',
            response_content_type='application/json',
            body_sha256=hashlib.sha256(payload).hexdigest(),
            body_byte_count=len(payload),
            started_at=_CAPTURED_AT - timedelta(seconds=2),
            completed_at=_CAPTURED_AT - timedelta(seconds=1),
            tls_peer=ImmportTlsPeerBinding(
                tls_version='TLSv1.3',
                certificate_der_sha256='a' * 64,
            ),
            collector_id='collector',
            collector_implementation_sha256='b' * 64,
            collector_execution_environment_sha256='c' * 64,
        ).model_dump(),
        'authorization': 'Bearer SECRET-CANARY',
    }
    with pytest.raises(ValueError, match='Extra inputs'):
        ImmportSanitizedCaptureReceipt.model_validate(raw)
    tls_peer = cast(dict[str, object], raw['tls_peer'])
    tls_covert_channel = {
        **{key: value for key, value in raw.items() if key != 'authorization'},
        'tls_peer': {
            **tls_peer,
            'cipher_suite': 'Bearer-SECRET-TLS-CANARY',
        },
    }
    with pytest.raises(ValueError, match='Extra inputs'):
        ImmportSanitizedCaptureReceipt.model_validate(tls_covert_channel)
    content_type_covert_channel = {
        **{key: value for key, value in raw.items() if key != 'authorization'},
        'response_content_type': 'application/json; bearer=SECRET-CONTENT-TYPE-CANARY',
    }
    with pytest.raises(ValueError, match='Input should be'):
        ImmportSanitizedCaptureReceipt.model_validate(content_type_covert_channel)
    with pytest.raises(ValueError, match='unapproved query parameter'):
        ImmportSanitizedCaptureReceipt.model_validate(
            {
                key: (f'{value}&X-Amz-Signature=SECRET' if key in {'requested_url', 'final_url'} else value)
                for key, value in raw.items()
                if key != 'authorization'
            }
        )


def test_receipt_verifier_does_not_echo_rejected_secret_input() -> None:
    body = canonical_json_bytes([])
    url = _endpoint_url('arm')
    valid = ImmportSanitizedCaptureReceipt(
        requested_url=url,
        final_url=url,
        authentication='immport_scoped_api_key_bearer_redacted',
        authorization_applied=True,
        credential_source='runtime_secret_broker',
        response_content_type='application/json',
        body_sha256=hashlib.sha256(body).hexdigest(),
        body_byte_count=len(body),
        started_at=_CAPTURED_AT - timedelta(seconds=2),
        completed_at=_CAPTURED_AT - timedelta(seconds=1),
        tls_peer=ImmportTlsPeerBinding(
            tls_version='TLSv1.3',
            certificate_der_sha256='a' * 64,
        ),
        collector_id='collector',
        collector_implementation_sha256='b' * 64,
        collector_execution_environment_sha256='c' * 64,
    )
    invalid = canonical_json_bytes(
        {
            **valid.model_dump(mode='json'),
            'authorization': 'Bearer SECRET-ERROR-CANARY',
        }
    )
    with pytest.raises(ImmportProductionSourceError) as caught:
        _verify_receipt(
            invalid,
            body_sha256=hashlib.sha256(body).hexdigest(),
            body_bytes=len(body),
            expected_url=url,
            captured_at=_CAPTURED_AT,
            accepted_tls_versions=('TLSv1.2', 'TLSv1.3'),
            expected_collector_id='collector',
            expected_collector_implementation_sha256='b' * 64,
            expected_collector_execution_environment_sha256='c' * 64,
        )
    assert 'SECRET-ERROR-CANARY' not in str(caught.value)
    assert 'SECRET-ERROR-CANARY' not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_adapter_rejects_post_cutoff_capture() -> None:
    adapter_input, policy = _adapter_fixture()
    too_early = policy.model_copy(update={'decision_at': datetime(2026, 6, 26, tzinfo=timezone.utc)})
    with pytest.raises(ImmportProductionAdapterError, match='after the decision cutoff'):
        adapt_tier_a_immport_arms(
            (adapter_input,),
            immport_arm_adapter_policy_bytes(too_early),
        )


@pytest.mark.parametrize(
    'episode_id',
    (
        'episode-SDY00000000',
        'episode-NCT12345678',
        'episodeSDY00000000suffix',
        'prefixNCT12345678suffix',
        'episode-10.1234/result',
        'episode-https://example.org/result',
    ),
)
def test_adapter_policy_rejects_direct_identifiers_in_public_episode_id(
    episode_id: str,
) -> None:
    with pytest.raises(ValueError, match='public episode_id cannot contain'):
        ImmportArmAdapterPolicy(
            policy_id='immport-complete-arm-panel-v1',
            source_id=_SOURCE_ID,
            episode_id=episode_id,
            study_accession=_STUDY,
            study_universe_registry_sha256='5' * 64,
            outcome_adjudication_spec_sha256='6' * 64,
            decision_at=datetime(2026, 6, 28, tzinfo=timezone.utc),
        )


def test_adapter_policy_pins_reviewed_license_claim() -> None:
    with pytest.raises(ValueError, match='ImmPort User Agreement'):
        ImmportArmAdapterPolicy.model_validate(
            {
                'policy_id': 'immport-complete-arm-panel-v1',
                'source_id': _SOURCE_ID,
                'episode_id': 'prospective-early-clinical-001',
                'study_accession': _STUDY,
                'decision_at': datetime(2026, 6, 28, tzinfo=timezone.utc),
                'license_id': 'CC0',
            }
        )


def test_policy_bytes_must_be_canonical() -> None:
    policy, verifier_input, _capture = _fixture()
    noncanonical = immport_source_verifier_policy_bytes(policy) + b'\n'
    with pytest.raises(ImmportProductionSourceError, match='canonical JSON'):
        verify_tier_a_immport_source(
            verifier_input,
            noncanonical,
            implementation_sha256='3' * 64,
            execution_environment_sha256='4' * 64,
        )
