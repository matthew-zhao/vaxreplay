from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import vaxreplay.operations.hermetic_execution as hermetic_execution_module
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.collector import (
    StaticCollectionError,
    StaticHttpsArtifactSpec,
    StaticHttpsCollectionPlan,
    static_plan_sha256,
)
from vaxreplay.operations.collector_semantics import verify_all_supported_run_manifests
from vaxreplay.operations.hermetic_execution import (
    IMPLEMENTATION_LABEL,
    Ed25519ReceiptSigner,
    HermeticOciEnvironment,
    HermeticSandboxPolicy,
    OciHermeticCallbackExecutor,
)
from vaxreplay.operations.http_capture import HttpsCaptureRequest
from vaxreplay.operations.immport_capture import (
    MAX_IMMPORT_CAPTURE_BODY_BYTES,
    MAX_IMMPORT_RECEIPT_BYTES,
    ImmportAuthenticatedArtifactSpec,
    ImmportAuthenticatedCaptureError,
    ImmportAuthenticatedCollectionPlan,
    ImmportCapturedExchange,
    immport_authenticated_plan_sha256,
    load_immport_authenticated_run_manifest,
    record_immport_authenticated_capture,
)
from vaxreplay.operations.plan_selection import (
    AuthenticatedPlanSelectionFacts,
    PlanSelectionClaim,
    PlanSelectionCommitment,
    PlanSelectionMaterialSpec,
    PlanSelectionPolicyBinding,
    broker_plan_selection,
)
from vaxreplay.operations.policy import (
    IMMPORT_AUTHENTICATED_COLLECTOR_ID,
    STATIC_HTTPS_COLLECTOR_ID,
    parse_immport_authenticated_job_configuration,
)
from vaxreplay.operations.promotion import (
    AdapterSpec,
    HermeticExecutionSpec,
    SourceVerifierSpec,
    WitnessMaterialSpec,
    _snapshot_one_capture,
    _verify_portable_immport_capture,
    build_capture_promotion,
    load_capture_promotion,
)
from vaxreplay.operations.promotion_schema import (
    PromotionIntegrityError,
    PromotionScopePolicy,
    PromotionSourceScope,
)
from vaxreplay.operations.schema import CaptureJobSpec
from vaxreplay.operations.scope_precommit import (
    build_scope_precommit,
    derive_plan_selection_commitment,
    derive_pre_capture_plan,
)
from vaxreplay.operations.store import OperationalStore
from vaxreplay.operations.witness import (
    AuthenticatedExternalWitnessFacts,
    ExternalWitnessClaim,
    ExternalWitnessMethod,
    WitnessPolicyBinding,
    broker_witness_checkpoint,
)
from vaxreplay.runner._process import BoundedProcessResult
from vaxreplay.sources import worker_cli
from vaxreplay.sources.immport import (
    IMMPORT_ARM_ADAPTER_EXCLUSION_REASON_CODES,
    IMMPORT_ARM_ADAPTER_ID,
    IMMPORT_ARM_ADAPTER_VERSION,
    IMMPORT_SOURCE_VERIFIER_ID,
    IMMPORT_SOURCE_VERIFIER_VERSION,
    ImmportArmAdapterPolicy,
    ImmportPromotionLayout,
    ImmportSanitizedCaptureReceipt,
    ImmportSourceVerifierPolicy,
    ImmportTlsPeerBinding,
    immport_arm_adapter_policy_bytes,
    immport_source_verifier_policy_bytes,
)

_T0 = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
_SOURCE_ID = 'immport:operational-test'
_STUDY = 'SDY00000000'
_IMPLEMENTATION_SHA256 = 'b' * 64
_ENVIRONMENT_SHA256 = 'c' * 64


class _Clock:
    def __init__(self, values: tuple[datetime, ...] | None = None) -> None:
        self._values = iter(values or tuple(_T0 + timedelta(seconds=offset) for offset in range(1, 100)))

    def __call__(self) -> datetime:
        return next(self._values)


def _urls() -> tuple[str, ...]:
    origin = 'https://www.immport.org'
    study = f'{origin}/data/query/api/study/{_STUDY}?format=json'
    manifest = f'{origin}/data/query/api/study/manifest/{_STUDY}?fileType=release_file&format=json'
    return (
        f'{origin}/data/query/v3/api-docs',
        study,
        manifest,
        f'{origin}/data/query/api/study/arm/{_STUDY}?format=json',
        f'{origin}/data/query/api/study/experiment/{_STUDY}?format=json',
        f'{origin}/data/query/api/study/link/{_STUDY}?format=json',
        manifest,
        study,
        f'{origin}/data/query/v3/api-docs',
    )


def _plan(
    *,
    max_body_bytes: int = 1024,
    timeout_seconds: int = 10,
    panel_deadline_seconds: int = 60,
) -> ImmportAuthenticatedCollectionPlan:
    return ImmportAuthenticatedCollectionPlan(
        plan_id='immport-authenticated-test-v1',
        source_id=_SOURCE_ID,
        study_accession=_STUDY,
        panel_deadline_seconds=panel_deadline_seconds,
        artifacts=tuple(
            ImmportAuthenticatedArtifactSpec(
                artifact_id=f'a{ordinal:02d}-{name}',
                requested_url=url,
                authentication=('none' if url.endswith('/v3/api-docs') else 'immport_scoped_api_key_bearer_redacted'),
                max_body_bytes=max_body_bytes,
                timeout_seconds=timeout_seconds,
            )
            for ordinal, (name, url) in enumerate(
                zip(
                    (
                        'openapi-before',
                        'study-before',
                        'manifest-before',
                        'arm',
                        'experiment',
                        'link',
                        'manifest-after',
                        'study-after',
                        'openapi-after',
                    ),
                    _urls(),
                    strict=True,
                ),
                start=1,
            )
        ),
    )


def _store(tmp_path: Path, plan: ImmportAuthenticatedCollectionPlan):
    store = OperationalStore.initialize(
        tmp_path / 'operations',
        created_at=_T0,
        store_id='a' * 32,
        trusted_lease_clock=None,
    )
    job = store.register_job(
        CaptureJobSpec(
            job_id='immport-authenticated-test',
            collector_id=IMMPORT_AUTHENTICATED_COLLECTOR_ID,
            schedule_anchor_at=_T0,
            schedule_interval_seconds=86400,
            configuration={
                'collection_plan_sha256': immport_authenticated_plan_sha256(plan),
                'source_id': plan.source_id,
                'lease_seconds': 600,
                'max_attempts_per_slot': 1,
                'collector_implementation_sha256': _IMPLEMENTATION_SHA256,
                'collector_execution_environment_sha256': _ENVIRONMENT_SHA256,
            },
        ),
        registered_at=_T0,
    )
    run = store.register_logical_run(job.spec_sha256, _T0, registered_at=_T0)
    return store, job, run


def _receipt(body: bytes, url: str, started_at: datetime, completed_at: datetime) -> bytes:
    authenticated = not url.endswith('/v3/api-docs')
    return canonical_json_bytes(
        ImmportSanitizedCaptureReceipt(
            requested_url=url,
            final_url=url,
            authentication=('immport_scoped_api_key_bearer_redacted' if authenticated else 'none'),
            authorization_applied=authenticated,
            credential_source='runtime_secret_broker' if authenticated else 'not_applicable',
            response_content_type='application/json;charset=UTF-8',
            body_sha256=hashlib.sha256(body).hexdigest(),
            body_byte_count=len(body),
            started_at=started_at,
            completed_at=completed_at,
            tls_peer=ImmportTlsPeerBinding(
                tls_version='TLSv1.3',
                certificate_der_sha256='d' * 64,
            ),
            collector_id=IMMPORT_AUTHENTICATED_COLLECTOR_ID,
            collector_implementation_sha256=_IMPLEMENTATION_SHA256,
            collector_execution_environment_sha256=_ENVIRONMENT_SHA256,
        )
    )


def _producer(
    *,
    start_offset: timedelta = timedelta(microseconds=100),
    body_override: bytes | None = None,
):
    def produce(plan, attempt):
        exchanges = []
        cursor = attempt.started_at + start_offset
        for ordinal, spec in enumerate(plan.artifacts):
            body = body_override if body_override is not None and ordinal == 0 else b'{}'
            started_at = cursor
            completed_at = cursor + timedelta(microseconds=50)
            exchanges.append(
                ImmportCapturedExchange(
                    artifact_id=spec.artifact_id,
                    body=body,
                    receipt=_receipt(body, spec.requested_url, started_at, completed_at),
                )
            )
            cursor = completed_at + timedelta(microseconds=50)
        return tuple(exchanges)

    return produce


def _production_openapi() -> dict[str, object]:
    schema_names = {
        '/api/study/{studyAccession}': 'StudyApi',
        '/api/study/arm/{studyAccession}': 'StudyArmApi',
        '/api/study/experiment/{studyAccession}': 'StudyExperimentApi',
        '/api/study/link/{studyAccession}': 'StudyLinkApi',
        '/api/study/manifest/{studyAccession}': 'FileDetails',
    }
    properties = {
        'StudyApi': (
            'studyAccession',
            'clinicalTrial',
            'sharedStudy',
            'latestDataReleaseVersion',
            'latestDataReleaseDate',
        ),
        'StudyArmApi': (
            'armAccession',
            'studyAccession',
            'description',
            'name',
            'typeReported',
            'typePreferred',
        ),
        'StudyExperimentApi': (
            'experimentAccession',
            'studyAccession',
            'measurementTechnique',
        ),
        'StudyLinkApi': ('studyLinkId', 'studyAccession', 'name', 'type', 'value'),
        'FileDetails': (
            'generatedMD5',
            'studyAccession',
            'filesizeBytes',
            'fileType',
            'fileName',
            'path',
            'fileUUID',
            'drsObjectCreated',
        ),
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
            'schemas': {
                name: {'type': 'object', 'properties': {field: {} for field in fields}}
                for name, fields in properties.items()
            }
        },
    }


def _production_bodies() -> tuple[bytes, ...]:
    contract = canonical_json_bytes(_production_openapi())
    study = canonical_json_bytes(
        [
            {
                'studyAccession': _STUDY,
                'clinicalTrial': 'Y',
                'sharedStudy': 'Y',
                'latestDataReleaseVersion': 'DR65',
                'latestDataReleaseDate': '2026-06-25',
            }
        ]
    )
    manifest = canonical_json_bytes(
        [
            {
                'generatedMD5': '1' * 32,
                'studyAccession': _STUDY,
                'filesizeBytes': 100,
                'fileType': 'release_file',
                'fileName': f'{_STUDY}_DR65_Tab.zip',
                'path': f'{_STUDY}/DR65/{_STUDY}_DR65_Tab.zip',
                'fileUUID': '11111111-1111-4111-8111-111111111111',
                'drsObjectCreated': 'Y',
            }
        ]
    )
    arms = canonical_json_bytes(
        [
            {
                'armAccession': 'ARM1',
                'studyAccession': _STUDY,
                'description': 'Low dose',
                'name': 'Low dose',
                'typeReported': 'Experimental',
                'typePreferred': 'Experimental Arm',
            },
            {
                'armAccession': 'ARM2',
                'studyAccession': _STUDY,
                'description': 'Comparator',
                'name': 'Comparator',
                'typeReported': 'Placebo Comparator',
                'typePreferred': 'Placebo Comparator Arm',
            },
        ]
    )
    experiments = canonical_json_bytes(
        [
            {
                'experimentAccession': 'EXP1',
                'studyAccession': _STUDY,
                'measurementTechnique': 'ELISA',
            }
        ]
    )
    links = canonical_json_bytes(
        [
            {
                'studyLinkId': 1,
                'studyAccession': _STUDY,
                'name': 'clinicaltrials.gov',
                'type': 'website',
                'value': 'https://clinicaltrials.gov/study/NCT12345678',
            }
        ]
    )
    return (contract, study, manifest, arms, experiments, links, manifest, study, contract)


def _production_producer(plan, attempt):
    exchanges = []
    cursor = attempt.started_at + timedelta(microseconds=100)
    for spec, body in zip(plan.artifacts, _production_bodies(), strict=True):
        completed_at = cursor + timedelta(microseconds=50)
        exchanges.append(
            ImmportCapturedExchange(
                artifact_id=spec.artifact_id,
                body=body,
                receipt=_receipt(body, spec.requested_url, cursor, completed_at),
            )
        )
        cursor = completed_at + timedelta(microseconds=50)
    return tuple(exchanges)


_SECCOMP = b'{"defaultAction":"SCMP_ACT_ERRNO","syscalls":[]}'
_RECEIPT_PRIVATE_KEY = bytes(range(32))


def _hermetic_environment() -> HermeticOciEnvironment:
    return HermeticOciEnvironment(
        environment_id='immport-production-worker-test',
        image_ref='registry.example/vaxreplay/immport@sha256:' + 'a' * 64,
        expected_image_id='sha256:' + 'e' * 64,
        platform='linux/amd64',
        entrypoint=('/opt/vaxreplay/immport-worker',),
    )


def _hermetic_sandbox(signer: Ed25519ReceiptSigner) -> HermeticSandboxPolicy:
    return HermeticSandboxPolicy(
        policy_id='immport-hermetic-test-v1',
        authority_id='independent-hermetic-test-runner',
        signing_key_id=signer.key_id,
        signing_public_key_sha256=hashlib.sha256(signer.public_key_bytes).hexdigest(),
        seccomp_profile_sha256=hashlib.sha256(_SECCOMP).hexdigest(),
        wall_seconds=30,
        memory_mib=256,
        milli_cpus=500,
        pids=32,
        scratch_mib=32,
        open_files=64,
        max_input_bytes=8 * 1024 * 1024,
        max_callback_policy_bytes=1024 * 1024,
        max_output_bytes=8 * 1024 * 1024,
        max_worker_response_bytes=12 * 1024 * 1024,
        max_log_bytes=64 * 1024,
    )


class _ProductionWorkerExecutor:
    """Signed OCI-boundary simulator that dispatches the real reviewed worker wire path."""

    def __init__(self) -> None:
        signer = Ed25519ReceiptSigner(
            key_id='immport-hermetic-test-key',
            private_key_bytes=_RECEIPT_PRIVATE_KEY,
        )
        self.signer = signer
        self.sandbox = _hermetic_sandbox(signer)
        with patch(
            'vaxreplay.operations.hermetic_execution._resolve_docker',
            return_value='/usr/local/bin/docker',
        ):
            self.inner = OciHermeticCallbackExecutor(
                sandbox_policy=self.sandbox,
                seccomp_profile_bytes=_SECCOMP,
                signer=signer,
                clock=lambda: _T0 + timedelta(seconds=101),
            )

    def execute(self, *, purpose, invocation_id, invocation_index, input_bytes, materials):
        def preflight(environment, implementation_sha256):
            inspection = {
                'Id': environment.expected_image_id,
                'Os': 'linux',
                'Architecture': 'amd64',
                'RepoDigests': [environment.image_ref],
                'Config': {
                    'Volumes': None,
                    'Cmd': None,
                    'Env': None,
                    'Labels': {IMPLEMENTATION_LABEL: implementation_sha256},
                },
            }
            return 'test-runtime-1', canonical_json_bytes(inspection)

        def run_worker(_argv, *, input_bytes, **_kwargs):
            worker_name = 'immport-source-verifier' if purpose == 'source_verifier' else 'immport-arm-adapter'
            return BoundedProcessResult(
                exit_code=0,
                duration_ms=5,
                stdout=worker_cli.dispatch(worker_name, input_bytes),
                stderr=b'',
                termination='exited',
                stdout_truncated=False,
                stderr_truncated=False,
            )

        with (
            patch.object(self.inner, '_preflight', side_effect=preflight),
            patch.object(self.inner, '_query', return_value=b'f' * 64),
            patch.object(self.inner, '_cleanup', return_value=True),
            patch.object(self.inner, '_request_remove'),
            patch.object(
                hermetic_execution_module,
                'run_bounded_process',
                side_effect=run_worker,
            ),
        ):
            return self.inner.execute(
                purpose=purpose,
                invocation_id=invocation_id,
                invocation_index=invocation_index,
                input_bytes=input_bytes,
                materials=materials,
            )


def _selection_policy() -> tuple[PlanSelectionPolicyBinding, bytes, bytes, bytes]:
    policy_bytes = b'immport first-write-wins policy'
    trust_bytes = b'immport selection registry trust'
    verifier_bytes = b'immport selection verifier'
    return (
        PlanSelectionPolicyBinding(
            campaign_id='immport-production-campaign',
            selection_key='immport-study-arm-panel',
            registry_id='independent-selection-registry',
            authority_id='benchmark-authority',
            policy_id='first-write-wins-v1',
            policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
            trust_policy_id='selection-trust-v1',
            trust_policy_sha256=hashlib.sha256(trust_bytes).hexdigest(),
            verifier_id='selection-verifier-v1',
            verifier_implementation_sha256=hashlib.sha256(verifier_bytes).hexdigest(),
        ),
        policy_bytes,
        trust_bytes,
        verifier_bytes,
    )


def _broker_selection(root: Path, commitment: PlanSelectionCommitment):
    policy, policy_bytes, trust_bytes, verifier_bytes = _selection_policy()
    proof = b'immport independent selection proof'

    def verifier(
        commitment_bytes,
        proof_bytes,
        expected_policy,
        exact_policy_bytes,
        exact_trust_policy_bytes,
    ):
        assert proof_bytes == proof
        assert expected_policy == policy
        assert exact_policy_bytes == policy_bytes
        assert exact_trust_policy_bytes == trust_bytes
        selected = PlanSelectionCommitment.model_validate_json(commitment_bytes)
        return AuthenticatedPlanSelectionFacts(
            receipt_id='immport-selection-receipt',
            registry_id=policy.registry_id,
            authority_id=policy.authority_id,
            campaign_id=policy.campaign_id,
            selection_key=policy.selection_key,
            commitment_sha256=hashlib.sha256(commitment_bytes).hexdigest(),
            store_id=selected.store_id,
            checkpoint_sha256=selected.checkpoint_sha256,
            scope_policy_sha256=selected.scope_policy_sha256,
            pre_capture_plan_sha256=selected.pre_capture_plan_sha256,
            selected_at_upper_bound=_T0 + timedelta(seconds=1),
            registry_entry_id='immport-selection-entry',
            registry_sequence=0,
            signed_checkpoint_sha256='9' * 64,
            signed_checkpoint_size=1,
        )

    materials = PlanSelectionMaterialSpec(
        policy=policy,
        policy_bytes=policy_bytes,
        trust_policy_bytes=trust_bytes,
        verifier_implementation_bytes=verifier_bytes,
        verifier=verifier,
    )
    selected = broker_plan_selection(
        root,
        commitment=commitment,
        materials=materials,
        provider=lambda _request: (
            PlanSelectionClaim(verification_uri='https://registry.example/immport-entry'),
            proof,
        ),
        verified_at=_T0 + timedelta(seconds=1, microseconds=1),
    )
    return selected, materials


def test_authenticated_immport_store_snapshot_and_portable_replay(tmp_path: Path) -> None:
    plan = _plan()
    store, job, run = _store(tmp_path, plan)
    result = record_immport_authenticated_capture(
        store,
        run.logical_run_id,
        plan,
        owner_id='immport-worker',
        producer=_producer(),
        clock=_Clock(),
    )

    assert load_immport_authenticated_run_manifest(store, result.attempt.attempt_id) == result.manifest
    assert verify_all_supported_run_manifests(store) == (result.manifest,)
    checkpoint = store.checkpoint(
        created_at=_T0 + timedelta(seconds=50),
        semantic_verifier=lambda: verify_all_supported_run_manifests(store),
    )
    assert checkpoint.through_sequence == len(store.events())

    capture = _snapshot_one_capture(
        store,
        plan.source_id,
        result.attempt.attempt_id,
        store.events(),
    )
    configuration = parse_immport_authenticated_job_configuration(job.spec.configuration)
    _verify_portable_immport_capture(capture, configuration)
    assert capture.binding.collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID
    assert {item.role for item in capture.artifacts} == {
        'collection-plan',
        'run-manifest',
        *(f'body.{item.artifact_id}' for item in plan.artifacts),
        *(f'receipt.{item.artifact_id}' for item in plan.artifacts),
    }


def test_unknown_successful_collector_fails_closed(tmp_path: Path) -> None:
    store = OperationalStore.initialize(
        tmp_path / 'operations',
        created_at=_T0,
        trusted_lease_clock=None,
    )
    job = store.register_job(
        CaptureJobSpec(
            job_id='unknown',
            collector_id='unknown-collector-v1',
            schedule_anchor_at=_T0,
            schedule_interval_seconds=60,
        ),
        registered_at=_T0,
    )
    run = store.register_logical_run(job.spec_sha256, _T0, registered_at=_T0)
    attempt = store.begin_attempt(run.logical_run_id, owner_id='worker', now=_T0)
    manifest = store.put_bytes(b'opaque', recorded_at=_T0 + timedelta(seconds=1))
    store.succeed_attempt(
        attempt.attempt_id,
        owner_id='worker',
        run_manifest_sha256=manifest.sha256,
        now=_T0 + timedelta(seconds=2),
    )
    with pytest.raises(StaticCollectionError, match='no semantic verifier is registered'):
        verify_all_supported_run_manifests(store)


@pytest.mark.parametrize(
    ('collector_id', 'terminal_state'),
    (
        (STATIC_HTTPS_COLLECTOR_ID, 'failed'),
        (STATIC_HTTPS_COLLECTOR_ID, 'abandoned'),
        (IMMPORT_AUTHENTICATED_COLLECTOR_ID, 'failed'),
        (IMMPORT_AUTHENTICATED_COLLECTOR_ID, 'abandoned'),
    ),
)
def test_semantic_checkpoint_replays_policy_for_unsuccessful_attempts(
    tmp_path: Path,
    collector_id: str,
    terminal_state: str,
) -> None:
    store = OperationalStore.initialize(
        tmp_path / 'operations',
        created_at=_T0,
        trusted_lease_clock=None,
    )
    if collector_id == STATIC_HTTPS_COLLECTOR_ID:
        static_plan = StaticHttpsCollectionPlan(
            plan_id='failed-static-test',
            source_id='static:unsuccessful-test',
            artifacts=(
                StaticHttpsArtifactSpec(
                    artifact_id='document',
                    request=HttpsCaptureRequest(
                        url='https://public.example.org/document.json',
                        allowed_host='public.example.org',
                        max_body_bytes=1024,
                    ),
                ),
            ),
        )
        plan_bytes = canonical_json_bytes(static_plan)
        configuration = {
            'collection_plan_sha256': static_plan_sha256(static_plan),
            'dns_resolution_attempts': 1,
            'dns_resolution_timeout_seconds': 10,
            'source_id': static_plan.source_id,
            'lease_seconds': 60,
            'max_dns_addresses': 16,
            'max_attempts_per_slot': 1,
            'max_total_body_bytes': 1024 * 1024,
            'plan_deadline_seconds': 30,
            'request_deadline_seconds': 20,
        }
        wrapper_name = 'verify_static_attempt_policy'
    else:
        immport_plan = _plan()
        plan_bytes = canonical_json_bytes(immport_plan)
        configuration = {
            'collection_plan_sha256': immport_authenticated_plan_sha256(immport_plan),
            'source_id': immport_plan.source_id,
            'lease_seconds': 60,
            'max_attempts_per_slot': 1,
            'collector_implementation_sha256': _IMPLEMENTATION_SHA256,
            'collector_execution_environment_sha256': _ENVIRONMENT_SHA256,
        }
        wrapper_name = 'verify_immport_attempt_policy'
    plan_artifact = store.put_bytes(plan_bytes, recorded_at=_T0)
    job = store.register_job(
        CaptureJobSpec(
            job_id=f'{terminal_state}-{collector_id}',
            collector_id=collector_id,
            schedule_anchor_at=_T0,
            schedule_interval_seconds=60,
            configuration=configuration,
        ),
        registered_at=_T0,
    )
    run = store.register_logical_run(job.spec_sha256, _T0, registered_at=_T0)
    attempt = store.begin_attempt(
        run.logical_run_id,
        owner_id='unsuccessful-worker',
        now=_T0 + timedelta(seconds=1),
        initial_artifacts={'collection-plan': plan_artifact.sha256},
    )
    if terminal_state == 'failed':
        store.fail_attempt(
            attempt.attempt_id,
            owner_id='unsuccessful-worker',
            terminal_code='source_unavailable',
            now=_T0 + timedelta(seconds=2),
        )
    else:
        assert store.abandon_expired_attempts(now=_T0 + timedelta(seconds=61))

    patch_target = f'vaxreplay.operations.collector_semantics.{wrapper_name}'
    with patch(patch_target, side_effect=StaticCollectionError('attempt-policy-canary')) as replay:
        with pytest.raises(StaticCollectionError, match='attempt-policy-canary'):
            verify_all_supported_run_manifests(store)
        replay.assert_called_once_with(store, run)
    with patch(patch_target, side_effect=StaticCollectionError('attempt-policy-canary')):
        with pytest.raises(StaticCollectionError, match='attempt-policy-canary'):
            store.checkpoint(
                created_at=_T0 + timedelta(seconds=100),
                semantic_verifier=lambda: verify_all_supported_run_manifests(store),
            )


def test_producer_exception_cannot_surface_or_persist_secret(tmp_path: Path) -> None:
    secret = 'BEARER-CANARY-DO-NOT-PERSIST'
    plan = _plan()
    store, _job, run = _store(tmp_path, plan)

    def failing_producer(_plan, _attempt):
        raise RuntimeError(f'Authorization: Bearer {secret}')

    with pytest.raises(ImmportAuthenticatedCaptureError) as caught:
        record_immport_authenticated_capture(
            store,
            run.logical_run_id,
            plan,
            owner_id='immport-worker',
            producer=failing_producer,
            clock=_Clock(),
        )
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert all(secret.encode() not in path.read_bytes() for path in store.root.rglob('*') if path.is_file())


def test_invalid_receipt_cannot_surface_or_persist_secret(tmp_path: Path) -> None:
    secret = 'RECEIPT-CANARY-DO-NOT-PERSIST'
    plan = _plan()
    store, _job, run = _store(tmp_path, plan)

    def invalid_receipt_producer(plan, attempt):
        exchanges = list(_producer()(plan, attempt))
        invalid = (
            b'{"authorization":"Bearer '
            + secret.encode()
            + b'","schema_version":"vaxreplay.immport-sanitized-capture-receipt.v0.1"}'
        )
        exchanges[0] = ImmportCapturedExchange(
            artifact_id=exchanges[0].artifact_id,
            body=exchanges[0].body,
            receipt=invalid,
        )
        return tuple(exchanges)

    with pytest.raises(ImmportAuthenticatedCaptureError) as caught:
        record_immport_authenticated_capture(
            store,
            run.logical_run_id,
            plan,
            owner_id='immport-worker',
            producer=invalid_receipt_producer,
            clock=_Clock(),
        )
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert all(secret.encode() not in path.read_bytes() for path in store.root.rglob('*') if path.is_file())


def test_malformed_producer_output_cannot_surface_or_persist_secret(tmp_path: Path) -> None:
    secret = 'MALFORMED-OUTPUT-CANARY-DO-NOT-PERSIST'
    plan = _plan()
    store, _job, run = _store(tmp_path, plan)

    class HostileOutput:
        def __repr__(self) -> str:
            return secret

    def malformed_producer(_plan, _attempt):
        return (HostileOutput(),)

    with pytest.raises(ImmportAuthenticatedCaptureError) as caught:
        record_immport_authenticated_capture(
            store,
            run.logical_run_id,
            plan,
            owner_id='immport-worker',
            producer=malformed_producer,
            clock=_Clock(),
        )
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert all(secret.encode() not in path.read_bytes() for path in store.root.rglob('*') if path.is_file())


def test_oversized_body_rejected_before_body_or_receipt_persistence(tmp_path: Path) -> None:
    plan = _plan(max_body_bytes=1)
    store, _job, run = _store(tmp_path, plan)
    with pytest.raises(ImmportAuthenticatedCaptureError, match='byte bound'):
        record_immport_authenticated_capture(
            store,
            run.logical_run_id,
            plan,
            owner_id='immport-worker',
            producer=_producer(body_override=b'xx'),
            clock=_Clock(),
        )
    attempt = store.list_attempts(logical_run_id=run.logical_run_id)[0]
    assert set(store.list_attempt_artifacts(attempt.attempt_id)) == {'collection-plan'}


def test_plan_body_budget_leaves_headroom_for_hermetic_base64_wire() -> None:
    assert MAX_IMMPORT_CAPTURE_BODY_BYTES == 64 * 1024 * 1024
    # Raw artifacts are base64-encoded inside the callback input and that input is
    # base64-encoded again inside the signed worker request.  Fourfold headroom is a
    # deliberately stronger invariant than the exact ~16/9 expansion.
    assert MAX_IMMPORT_CAPTURE_BODY_BYTES * 4 < 512 * 1024 * 1024
    with pytest.raises(ValueError, match='aggregate body bound'):
        _plan(max_body_bytes=8 * 1024 * 1024)


def test_impossible_lease_deadline_rejected_before_attempt_or_producer(tmp_path: Path) -> None:
    plan = _plan(panel_deadline_seconds=601)
    store, _job, run = _store(tmp_path, plan)
    producer_called = False

    def producer(_plan, _attempt):
        nonlocal producer_called
        producer_called = True
        return ()

    with pytest.raises(ImmportAuthenticatedCaptureError, match='lease is shorter'):
        record_immport_authenticated_capture(
            store,
            run.logical_run_id,
            plan,
            owner_id='immport-worker',
            producer=producer,
            clock=_Clock(),
        )
    assert not producer_called
    assert store.list_attempts(logical_run_id=run.logical_run_id) == ()
    assert store.verify().object_count == 0


@pytest.mark.parametrize(
    'start_offset',
    (
        timedelta(seconds=-1),
        timedelta(seconds=5),
        timedelta(seconds=601),
    ),
)
def test_receipts_outside_attempt_validation_or_lease_are_rejected(
    tmp_path: Path,
    start_offset: timedelta,
) -> None:
    plan = _plan()
    store, _job, run = _store(tmp_path, plan)
    with pytest.raises(ImmportAuthenticatedCaptureError):
        record_immport_authenticated_capture(
            store,
            run.logical_run_id,
            plan,
            owner_id='immport-worker',
            producer=_producer(start_offset=start_offset),
            clock=_Clock(),
        )


def test_portable_oversized_receipt_rejected_before_parsing(tmp_path: Path) -> None:
    plan = _plan()
    store, job, run = _store(tmp_path, plan)
    result = record_immport_authenticated_capture(
        store,
        run.logical_run_id,
        plan,
        owner_id='immport-worker',
        producer=_producer(),
        clock=_Clock(),
    )
    capture = _snapshot_one_capture(
        store,
        plan.source_id,
        result.attempt.attempt_id,
        store.events(),
    )
    target_index = next(index for index, item in enumerate(capture.artifacts) if item.role.startswith('receipt.'))
    target = capture.artifacts[target_index]
    oversized = b'{' + b' ' * MAX_IMMPORT_RECEIPT_BYTES + b'}'
    tampered_binding = target.binding.model_copy(
        update={
            'file': target.binding.file.model_copy(
                update={
                    'sha256': hashlib.sha256(oversized).hexdigest(),
                    'byte_count': len(oversized),
                }
            )
        }
    )
    tampered = target.__class__(binding=tampered_binding, payload=oversized)
    capture = capture.__class__(
        binding=capture.binding,
        artifacts=tuple(tampered if index == target_index else item for index, item in enumerate(capture.artifacts)),
    )
    configuration = parse_immport_authenticated_job_configuration(job.spec.configuration)
    with pytest.raises(PromotionIntegrityError, match='receipt exceeds its byte bound'):
        _verify_portable_immport_capture(capture, configuration)


def test_authenticated_immport_witnessed_hermetic_promotion_round_trip(
    tmp_path: Path,
) -> None:
    plan = _plan(max_body_bytes=1024 * 1024)
    store = OperationalStore.initialize(
        tmp_path / 'operations',
        created_at=_T0,
        store_id='a' * 32,
        trusted_lease_clock=None,
    )
    scheduled_for = _T0 + timedelta(seconds=10)
    job = store.register_job(
        CaptureJobSpec(
            job_id='immport-production-round-trip',
            collector_id=IMMPORT_AUTHENTICATED_COLLECTOR_ID,
            schedule_anchor_at=scheduled_for,
            schedule_interval_seconds=86400,
            configuration={
                'collection_plan_sha256': immport_authenticated_plan_sha256(plan),
                'source_id': plan.source_id,
                'lease_seconds': 600,
                'max_attempts_per_slot': 1,
                'collector_implementation_sha256': _IMPLEMENTATION_SHA256,
                'collector_execution_environment_sha256': _ENVIRONMENT_SHA256,
            },
        ),
        registered_at=_T0,
    )

    layout = ImmportPromotionLayout(
        study_accession=_STUDY,
        openapi_before_artifact_id=plan.artifacts[0].artifact_id,
        study_before_artifact_id=plan.artifacts[1].artifact_id,
        manifest_before_artifact_id=plan.artifacts[2].artifact_id,
        arm_artifact_id=plan.artifacts[3].artifact_id,
        experiment_artifact_id=plan.artifacts[4].artifact_id,
        link_artifact_id=plan.artifacts[5].artifact_id,
        manifest_after_artifact_id=plan.artifacts[6].artifact_id,
        study_after_artifact_id=plan.artifacts[7].artifact_id,
        openapi_after_artifact_id=plan.artifacts[8].artifact_id,
    )
    source_policy = ImmportSourceVerifierPolicy(
        policy_id='immport-production-source-policy-v1',
        source_id=plan.source_id,
        study_universe_registry_sha256='5' * 64,
        layout=layout,
        expected_openapi_sha256=hashlib.sha256(_production_bodies()[0]).hexdigest(),
        expected_openapi_info_version='v1',
        expected_latest_release_version='DR65',
        expected_latest_release_date=date(2026, 6, 25),
        expected_collector_id=IMMPORT_AUTHENTICATED_COLLECTOR_ID,
        expected_collector_implementation_sha256=_IMPLEMENTATION_SHA256,
        expected_collector_execution_environment_sha256=_ENVIRONMENT_SHA256,
    )
    adapter_policy = ImmportArmAdapterPolicy(
        policy_id='immport-production-arm-policy-v1',
        source_id=plan.source_id,
        episode_id='immport-production-episode',
        study_accession=_STUDY,
        study_universe_registry_sha256='5' * 64,
        outcome_adjudication_spec_sha256='6' * 64,
        decision_at=_T0 + timedelta(seconds=90),
        minimum_candidate_count=2,
    )
    executor = _ProductionWorkerExecutor()
    environment_bytes = canonical_json_bytes(_hermetic_environment())
    hermetic = HermeticExecutionSpec(
        sandbox_policy_bytes=canonical_json_bytes(executor.sandbox),
        seccomp_profile_bytes=_SECCOMP,
        trusted_public_key_bytes=executor.signer.public_key_bytes,
        executor=executor,
    )
    source_spec = SourceVerifierSpec(
        verifier_id=IMMPORT_SOURCE_VERIFIER_ID,
        verifier_version=IMMPORT_SOURCE_VERIFIER_VERSION,
        implementation_bytes=b'reviewed ImmPort source verifier implementation',
        policy_bytes=immport_source_verifier_policy_bytes(source_policy),
        execution_environment_bytes=environment_bytes,
        hermetic_execution=hermetic,
    )
    adapter_spec = AdapterSpec(
        adapter_id=IMMPORT_ARM_ADAPTER_ID,
        adapter_version=IMMPORT_ARM_ADAPTER_VERSION,
        implementation_bytes=b'reviewed ImmPort arm adapter implementation',
        policy_bytes=immport_arm_adapter_policy_bytes(adapter_policy),
        execution_environment_bytes=environment_bytes,
        hermetic_execution=hermetic,
        allowed_exclusion_reason_codes=IMMPORT_ARM_ADAPTER_EXCLUSION_REASON_CODES,
    )
    source_specs = {plan.source_id: source_spec}

    capture_policy_bytes = b'immport capture witness policy'
    capture_trust_bytes = b'immport capture witness trust'
    capture_verifier_bytes = b'immport capture witness verifier'
    capture_policy = WitnessPolicyBinding(
        authority_id='independent-capture-log',
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id='immport-capture-witness-v1',
        policy_sha256=hashlib.sha256(capture_policy_bytes).hexdigest(),
        trust_policy_id='immport-capture-trust-v1',
        trust_policy_sha256=hashlib.sha256(capture_trust_bytes).hexdigest(),
        verifier_id='immport-capture-verifier-v1',
        verifier_implementation_sha256=hashlib.sha256(capture_verifier_bytes).hexdigest(),
    )
    scope_policy_bytes = b'immport scope witness policy'
    scope_trust_bytes = b'immport scope witness trust'
    scope_verifier_bytes = b'immport scope witness verifier'
    scope_witness_policy = WitnessPolicyBinding(
        authority_id='independent-scope-log',
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id='immport-scope-witness-v1',
        policy_sha256=hashlib.sha256(scope_policy_bytes).hexdigest(),
        trust_policy_id='immport-scope-trust-v1',
        trust_policy_sha256=hashlib.sha256(scope_trust_bytes).hexdigest(),
        verifier_id='immport-scope-verifier-v1',
        verifier_implementation_sha256=hashlib.sha256(scope_verifier_bytes).hexdigest(),
    )
    promotion_scope = PromotionScopePolicy(
        policy_id='immport-promotion-scope-v1',
        store_id=store.store_id,
        checkpoint_created_at_not_before=_T0 + timedelta(seconds=99),
        checkpoint_created_at_not_after=_T0 + timedelta(seconds=100),
        sources=(
            PromotionSourceScope(
                source_id=plan.source_id,
                job_spec_sha256s=(job.spec_sha256,),
                scheduled_from=scheduled_for,
                scheduled_through=scheduled_for,
            ),
        ),
    )
    selection_policy, _policy_bytes, _trust_bytes, _verifier_bytes = _selection_policy()
    pre_capture_plan = derive_pre_capture_plan(
        scope_policy=promotion_scope,
        selection_policy=selection_policy,
        capture_witness_policy=capture_policy,
        source_verifiers=source_specs,
        adapter=adapter_spec,
    )
    store.put_bytes(canonical_json_bytes(promotion_scope), recorded_at=_T0 + timedelta(seconds=1))
    store.put_bytes(canonical_json_bytes(pre_capture_plan), recorded_at=_T0 + timedelta(seconds=1))
    precommit_checkpoint = store.checkpoint(created_at=_T0 + timedelta(seconds=1))
    selection_commitment = derive_plan_selection_commitment(
        promotion_scope,
        pre_capture_plan,
        precommit_checkpoint,
    )
    plan_selection, selection_materials = _broker_selection(
        tmp_path / 'selection',
        selection_commitment,
    )

    def scope_witness_verifier(target, proof, expected_policy):
        assert target == canonical_json_bytes(precommit_checkpoint)
        assert proof == b'immport scope witness proof'
        assert expected_policy == scope_witness_policy
        return AuthenticatedExternalWitnessFacts(
            receipt_id='immport-scope-witness-receipt',
            authority_id=scope_witness_policy.authority_id,
            witness_id='independent-scope-log-key',
            method=scope_witness_policy.method,
            policy_id=scope_witness_policy.policy_id,
            checkpoint_sha256=hashlib.sha256(canonical_json_bytes(precommit_checkpoint)).hexdigest(),
            witnessed_at=_T0 + timedelta(seconds=2),
        )

    scope_witness = broker_witness_checkpoint(
        tmp_path / 'scope-witness',
        checkpoint=precommit_checkpoint,
        policy=scope_witness_policy,
        provider=lambda _request: (
            ExternalWitnessClaim(verification_uri='https://log.example/immport-scope'),
            b'immport scope witness proof',
        ),
        verifier=scope_witness_verifier,
        verified_at=_T0 + timedelta(seconds=2, microseconds=1),
    )
    scope_witness_materials = WitnessMaterialSpec(
        policy=scope_witness_policy,
        policy_bytes=scope_policy_bytes,
        trust_policy_bytes=scope_trust_bytes,
        verifier_implementation_bytes=scope_verifier_bytes,
        verifier=scope_witness_verifier,
    )
    scope_precommit = build_scope_precommit(
        tmp_path / 'scope-precommit',
        store=store,
        scope_policy=promotion_scope,
        pre_capture_plan=pre_capture_plan,
        witness_root=scope_witness.root,
        witness_materials=scope_witness_materials,
        selection_root=plan_selection.root,
        expected_selection_manifest_sha256=plan_selection.manifest_sha256,
        selection_materials=selection_materials,
        created_at=_T0 + timedelta(seconds=3),
        verified_at=_T0 + timedelta(seconds=4),
    )

    run = store.register_logical_run(
        job.spec_sha256,
        scheduled_for,
        registered_at=scheduled_for,
    )
    record_immport_authenticated_capture(
        store,
        run.logical_run_id,
        plan,
        owner_id='immport-production-worker',
        producer=_production_producer,
        clock=_Clock(tuple(_T0 + timedelta(seconds=offset) for offset in range(11, 90))),
    )
    checkpoint = store.checkpoint(
        created_at=_T0 + timedelta(seconds=100),
        semantic_verifier=lambda: verify_all_supported_run_manifests(store),
    )

    def capture_witness_verifier(target, proof, expected_policy):
        assert target == canonical_json_bytes(checkpoint)
        assert proof == b'immport capture witness proof'
        assert expected_policy == capture_policy
        return AuthenticatedExternalWitnessFacts(
            receipt_id='immport-capture-witness-receipt',
            authority_id=capture_policy.authority_id,
            witness_id='independent-capture-log-key',
            method=capture_policy.method,
            policy_id=capture_policy.policy_id,
            checkpoint_sha256=hashlib.sha256(canonical_json_bytes(checkpoint)).hexdigest(),
            witnessed_at=_T0 + timedelta(seconds=101),
        )

    capture_witness = broker_witness_checkpoint(
        tmp_path / 'capture-witness',
        checkpoint=checkpoint,
        policy=capture_policy,
        provider=lambda _request: (
            ExternalWitnessClaim(verification_uri='https://log.example/immport-capture'),
            b'immport capture witness proof',
        ),
        verifier=capture_witness_verifier,
        verified_at=_T0 + timedelta(seconds=102),
    )
    capture_witness_materials = WitnessMaterialSpec(
        policy=capture_policy,
        policy_bytes=capture_policy_bytes,
        trust_policy_bytes=capture_trust_bytes,
        verifier_implementation_bytes=capture_verifier_bytes,
        verifier=capture_witness_verifier,
    )
    built = build_capture_promotion(
        tmp_path / 'promotion',
        promotion_id='immport-production-promotion-v1',
        store=store,
        witness_root=capture_witness.root,
        witness_materials=capture_witness_materials,
        scope_policy=promotion_scope,
        scope_precommit_root=scope_precommit.root,
        scope_precommit_witness_materials=scope_witness_materials,
        expected_scope_precommit_sha256=scope_precommit.archive_sha256,
        selection_materials=selection_materials,
        expected_selection_manifest_sha256=plan_selection.manifest_sha256,
        source_verifiers=source_specs,
        adapter=adapter_spec,
        created_at=_T0 + timedelta(seconds=103),
        verified_at=_T0 + timedelta(seconds=104),
    )
    loaded = load_capture_promotion(
        built.root,
        expected_scope_policy=promotion_scope,
        scope_precommit_witness_materials=scope_witness_materials,
        witness_materials=capture_witness_materials,
        source_verifiers=source_specs,
        adapter=adapter_spec,
        verified_at=_T0 + timedelta(seconds=105),
        expected_scope_precommit_sha256=scope_precommit.archive_sha256,
        expected_promotion_sha256=built.manifest_sha256,
        selection_materials=selection_materials,
        expected_selection_manifest_sha256=plan_selection.manifest_sha256,
    )
    assert loaded.index == built.index
    assert len(loaded.candidates) == 2
    assert len(loaded.evidence) == 2
    assert {item.collector_id for item in loaded.index.captures} == {IMMPORT_AUTHENTICATED_COLLECTOR_ID}
    assert len(loaded.index.hermetic_executions) == 3
    assert sorted(item.purpose for item in loaded.index.hermetic_executions) == [
        'adapter',
        'adapter',
        'source_verifier',
    ]
    assert {item.subject_id for item in loaded.index.hermetic_executions} == {
        IMMPORT_ARM_ADAPTER_ID,
        plan.source_id,
    }
