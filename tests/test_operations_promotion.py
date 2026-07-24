from __future__ import annotations

import hashlib
import inspect
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import vaxreplay.operations.promotion as promotion_module
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord, ForecastTarget, SourceType, Split
from vaxreplay.operations.collector import (
    STATIC_HTTPS_COLLECTOR_ID,
    StaticHttpsArtifactSpec,
    StaticHttpsCollectionPlan,
    run_static_https_collection,
    static_plan_sha256,
    verify_all_static_run_manifests,
)
from vaxreplay.operations.http_capture import HttpsCaptureRequest
from vaxreplay.operations.plan_selection import (
    AuthenticatedPlanSelectionFacts,
    PlanSelectionClaim,
    PlanSelectionCommitment,
    PlanSelectionMaterialSpec,
    PlanSelectionPolicyBinding,
    broker_plan_selection,
)
from vaxreplay.operations.policy import parse_static_job_configuration
from vaxreplay.operations.promotion import (
    AdapterRunResult,
    AdapterSourceInput,
    AdapterSpec,
    SourceVerifierRunResult,
    SourceVerifierSpec,
    WitnessMaterialSpec,
    _load_scoped_jobs,
    _normalize_adapter_result,
    _parse_registered_jobs,
    _run_source_verifiers,
    _snapshot_one_capture,
    _verify_static_attempt_history,
    build_capture_promotion,
    build_prospective_decision_package_from_promotion,
    load_capture_promotion,
    make_promotion_source_capture_verifier,
)
from vaxreplay.operations.promotion_schema import (
    AuthoritativeReleaseBasis,
    AuthoritativeSourceRelease,
    CaptureIndex,
    CapturePromotionManifest,
    HermeticExecutionPromotionBinding,
    NormalizedRecordReference,
    PromotionFileBinding,
    PromotionIntegrityError,
    PromotionScopePolicy,
    PromotionSourceScope,
    SourceRecordBinding,
    SourceRecordDisposition,
    SourceVerificationResult,
    SourceVerifierIdentity,
    source_verification_result_sha256,
)
from vaxreplay.operations.schema import CaptureJobSpec, LedgerEventType, checkpoint_sha256
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
from vaxreplay.temporal_schema import DecisionTimeConfig

_T0 = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


class _Clock:
    def __init__(self) -> None:
        self._values = iter(_T0 + timedelta(seconds=offset) for offset in range(1, 30))

    def __call__(self) -> datetime:
        return next(self._values)


class _OffsetClock:
    def __init__(self, first_offset: int, *, base: datetime = _T0) -> None:
        self._values = iter(base + timedelta(seconds=offset) for offset in range(first_offset, 60))

    def __call__(self) -> datetime:
        return next(self._values)


class _StepClock:
    def __init__(self, step_seconds: int) -> None:
        self._values = iter(_T0 + timedelta(seconds=step_seconds * ordinal) for ordinal in range(1, 60))

    def __call__(self) -> datetime:
        return next(self._values)


def _hermetic_binding_for_path_test(
    subject_id: str,
    invocation_index: int,
    *,
    seccomp_path: str,
) -> HermeticExecutionPromotionBinding:
    def file(name: str) -> PromotionFileBinding:
        return PromotionFileBinding(
            path=f'hermetic/{subject_id}/{name}',
            sha256='a' * 64,
            byte_count=1,
        )

    return HermeticExecutionPromotionBinding(
        subject_id=subject_id,
        purpose='adapter',
        invocation_id=f'invocation-{subject_id}',
        invocation_index=invocation_index,
        request=file('request.json'),
        response=file('response.json'),
        receipt=file('receipt.json'),
        image_inspection=file('image-inspection.json'),
        sandbox_policy=file('sandbox-policy.json'),
        seccomp_profile=PromotionFileBinding(
            path=seccomp_path,
            sha256='b' * 64,
            byte_count=1,
        ),
        trusted_public_key=file('trusted-public-key.bin'),
        output_sha256='c' * 64,
        output_byte_count=1,
        authority_id='runner-authority',
        signing_key_id='runner-key',
        issued_at=_T0,
    )


def test_capture_index_rejects_cross_execution_seccomp_path_reuse() -> None:
    shared_path = 'hermetic/shared/seccomp-profile.json'
    bindings = (
        _hermetic_binding_for_path_test('adapter-a', 0, seccomp_path=shared_path),
        _hermetic_binding_for_path_test('adapter-b', 1, seccomp_path=shared_path),
    )
    with pytest.raises(ValueError, match='cannot share archive paths'):
        CaptureIndex.validate_hermetic_executions(bindings)


class _Response:
    status_code = 200
    response_headers: tuple[tuple[str, str], ...]

    def __init__(self, body: bytes, url: str) -> None:
        self.body = body
        self.final_url = url
        self.response_headers = (('Content-Length', str(len(body))),)
        self.position = 0

    def read(self, size: int) -> bytes:
        value = self.body[self.position : self.position + size]
        self.position += len(value)
        return value

    def tls_peer_metadata(self):
        return None

    def close(self) -> None:
        pass


class _Transport:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def open(self, request):
        del request
        return self.response


def _plan_selection_policy(
    campaign_id: str = 'fixture-pandemic-campaign-2027',
    selection_key: str = 'antigen-prioritization-plan',
):
    policy_bytes = f'first-write-wins policy for {campaign_id}/{selection_key}'.encode()
    trust_policy_bytes = f'pinned registry trust for {campaign_id}'.encode()
    verifier_bytes = f'offline registry verifier for {campaign_id}'.encode()
    policy = PlanSelectionPolicyBinding(
        campaign_id=campaign_id,
        selection_key=selection_key,
        registry_id='fixture-independent-plan-registry',
        authority_id='fixture-benchmark-authority',
        policy_id='fixture-first-write-wins-policy-v1',
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        trust_policy_id='fixture-plan-registry-trust-v1',
        trust_policy_sha256=hashlib.sha256(trust_policy_bytes).hexdigest(),
        verifier_id='fixture-plan-registry-verifier-v1',
        verifier_implementation_sha256=hashlib.sha256(verifier_bytes).hexdigest(),
    )
    return policy, policy_bytes, trust_policy_bytes, verifier_bytes


def _plan_selection_fixture(
    root: Path,
    *,
    commitment: PlanSelectionCommitment,
    campaign_id: str = 'fixture-pandemic-campaign-2027',
    selection_key: str = 'antigen-prioritization-plan',
    selected_at: datetime = _T0,
):
    policy, policy_bytes, trust_policy_bytes, verifier_bytes = _plan_selection_policy(
        campaign_id,
        selection_key,
    )
    assert commitment.policy == policy
    proof = canonical_json_bytes(
        {
            'campaign_id': campaign_id,
            'selection_key': selection_key,
            'commitment_sha256': hashlib.sha256(canonical_json_bytes(commitment)).hexdigest(),
            'registry_entry_id': f'entry-{campaign_id}',
            'registry_sequence': 0,
            'signed_checkpoint_size': 1,
        }
    )

    def verifier(
        commitment_bytes: bytes,
        proof_bytes: bytes,
        expected_policy: PlanSelectionPolicyBinding,
        exact_policy_bytes: bytes,
        exact_trust_policy_bytes: bytes,
    ) -> AuthenticatedPlanSelectionFacts:
        assert expected_policy == policy
        assert exact_policy_bytes == policy_bytes
        assert exact_trust_policy_bytes == trust_policy_bytes
        assert proof_bytes == proof
        selected_commitment = PlanSelectionCommitment.model_validate_json(commitment_bytes)
        return AuthenticatedPlanSelectionFacts(
            receipt_id=f'receipt-{campaign_id}',
            registry_id=policy.registry_id,
            authority_id=policy.authority_id,
            campaign_id=policy.campaign_id,
            selection_key=policy.selection_key,
            commitment_sha256=hashlib.sha256(commitment_bytes).hexdigest(),
            store_id=selected_commitment.store_id,
            checkpoint_sha256=selected_commitment.checkpoint_sha256,
            scope_policy_sha256=selected_commitment.scope_policy_sha256,
            pre_capture_plan_sha256=selected_commitment.pre_capture_plan_sha256,
            selected_at_upper_bound=selected_at,
            registry_entry_id=f'entry-{campaign_id}',
            registry_sequence=0,
            signed_checkpoint_sha256=hashlib.sha256(f'signed checkpoint for {campaign_id}'.encode()).hexdigest(),
            signed_checkpoint_size=1,
        )

    materials = PlanSelectionMaterialSpec(
        policy=policy,
        policy_bytes=policy_bytes,
        trust_policy_bytes=trust_policy_bytes,
        verifier_implementation_bytes=verifier_bytes,
        verifier=verifier,
    )
    selection = broker_plan_selection(
        root,
        commitment=commitment,
        materials=materials,
        provider=lambda _request: (
            PlanSelectionClaim(verification_uri='https://registry.example/fixture-entry'),
            proof,
        ),
        verified_at=selected_at + timedelta(microseconds=1),
    )
    return selection, materials


def _jsonl(*records) -> bytes:
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


@pytest.mark.parametrize(
    'field_name',
    ('http_header_last_modified', 'http-header-date', 'x-origin-date', 'response Date'),
)
def test_authoritative_release_rejects_transport_time_aliases(field_name: str) -> None:
    with pytest.raises(ValueError, match='generic HTTP'):
        AuthoritativeSourceRelease(
            source_release_at=_T0,
            basis=AuthoritativeReleaseBasis.SOURCE_API_PUBLICATION_FIELD,
            authority_locator='publisher record',
            authority_field=field_name,
            evidence_attempt_id=f'attempt-{"a" * 32}',
            evidence_role='body.release',
            evidence_sha256='b' * 64,
            evidence_source_record_id='release-record',
            evidence_source_record_sha256='c' * 64,
        )


@pytest.mark.parametrize('field_name', ('publication_date', 'release_date'))
def test_authoritative_release_allows_source_semantic_dates(field_name: str) -> None:
    release = AuthoritativeSourceRelease(
        source_release_at=_T0,
        basis=AuthoritativeReleaseBasis.SOURCE_API_PUBLICATION_FIELD,
        authority_locator='publisher record',
        authority_field=field_name,
        evidence_attempt_id=f'attempt-{"a" * 32}',
        evidence_role='body.release',
        evidence_sha256='b' * 64,
        evidence_source_record_id='release-record',
        evidence_source_record_sha256='c' * 64,
    )
    assert release.authority_field == field_name


def _normalization_unit_fixture():
    source_id = 'publisher:unit'
    source_records = (
        SourceRecordBinding(
            source_id=source_id,
            source_record_id='record-1',
            source_record_sha256='1' * 64,
            source_artifact_sha256='a' * 64,
            source_locator='snapshot.json#/records/1',
        ),
        SourceRecordBinding(
            source_id=source_id,
            source_record_id='record-2',
            source_record_sha256='2' * 64,
            source_artifact_sha256='b' * 64,
            source_locator='snapshot.json#/records/2',
        ),
    )
    source_record_bytes = _jsonl(*source_records)
    verification = SourceVerificationResult(
        source_id=source_id,
        verifier=SourceVerifierIdentity(
            verifier_id='unit-verifier',
            verifier_version='v1',
            implementation_sha256='3' * 64,
            execution_environment_sha256='7' * 64,
        ),
        verifier_policy_sha256='4' * 64,
        verified_attempt_ids=(f'attempt-{"5" * 32}',),
        source_release=AuthoritativeSourceRelease(
            source_release_at=_T0,
            basis=AuthoritativeReleaseBasis.SOURCE_VERSION_MANIFEST,
            authority_locator='snapshot.json#/released_at',
            authority_field='released_at',
            evidence_attempt_id=f'attempt-{"5" * 32}',
            evidence_role='body.snapshot',
            evidence_sha256=source_records[0].source_artifact_sha256,
            evidence_source_record_id=source_records[0].source_record_id,
            evidence_source_record_sha256=source_records[0].source_record_sha256,
        ),
        verified_capture_inventory_sha256='6' * 64,
        verified_source_record_inventory_sha256=hashlib.sha256(source_record_bytes).hexdigest(),
        verified_source_record_count=len(source_records),
        result_codes=('complete_inventory',),
    )
    inputs = (
        AdapterSourceInput(
            source_id=source_id,
            captures=(),
            verification_result=verification,
            verified_records=source_records,
        ),
    )
    candidates = (
        CandidateRecord(episode_id='unit-episode', candidate_id='candidate-a'),
        CandidateRecord(episode_id='unit-episode', candidate_id='candidate-b'),
    )
    evidence = EvidenceRecord(
        episode_id='unit-episode',
        evidence_id='evidence-a',
        source_type=SourceType.PUBLIC_HEALTH,
        collected_at=_T0,
        available_at=_T0,
        title='Unit evidence',
        body='Exact unit evidence.',
        body_sha256=hashlib.sha256(b'Exact unit evidence.').hexdigest(),
        related_candidate_ids=['candidate-a'],
        provenance_url='https://public.example.org/snapshot.json',
        license_id='public-domain',
        derivation='unit normalization fixture',
    )
    candidate_refs = tuple(
        NormalizedRecordReference(
            episode_id=record.episode_id,
            record_id=record.candidate_id,
            record_sha256=hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
        )
        for record in candidates
    )
    evidence_ref = NormalizedRecordReference(
        episode_id=evidence.episode_id,
        record_id=evidence.evidence_id,
        record_sha256=hashlib.sha256(canonical_json_bytes(evidence)).hexdigest(),
    )
    normalized = SourceRecordDisposition(
        source_id=source_id,
        source_record_id=source_records[0].source_record_id,
        source_record_sha256=source_records[0].source_record_sha256,
        source_artifact_sha256=source_records[0].source_artifact_sha256,
        disposition='normalized',
        candidate_record_refs=candidate_refs,
        evidence_record_refs=(evidence_ref,),
    )
    excluded = SourceRecordDisposition(
        source_id=source_id,
        source_record_id=source_records[1].source_record_id,
        source_record_sha256=source_records[1].source_record_sha256,
        source_artifact_sha256=source_records[1].source_artifact_sha256,
        disposition='excluded',
        reason_code='not_relevant',
    )
    return inputs, candidates, evidence, normalized, excluded


@pytest.mark.parametrize('case', ('omitted', 'invented', 'duplicate'))
def test_adapter_rejects_nonexhaustive_source_record_dispositions(case: str) -> None:
    inputs, candidates, evidence, normalized, excluded = _normalization_unit_fixture()
    if case == 'omitted':
        dispositions = (normalized,)
        message = 'cover every verified source record'
    elif case == 'invented':
        invented = excluded.model_copy(
            update={
                'source_record_id': 'record-3',
                'source_record_sha256': '7' * 64,
                'source_artifact_sha256': '8' * 64,
            }
        )
        dispositions = (normalized, excluded, invented)
        message = 'cover every verified source record'
    else:
        dispositions = (normalized, normalized, excluded)
        message = 'duplicate source-record dispositions'
    with pytest.raises(PromotionIntegrityError, match=message):
        _normalize_adapter_result(
            AdapterRunResult(
                candidate_records=_jsonl(*candidates),
                evidence_records=_jsonl(evidence),
                dispositions=_jsonl(*dispositions),
            ),
            inputs,
            ('not_relevant',),
        )


def test_adapter_rejects_normalized_output_without_incoming_source_edge() -> None:
    inputs, candidates, evidence, normalized, excluded = _normalization_unit_fixture()
    unreferenced = normalized.model_copy(update={'candidate_record_refs': normalized.candidate_record_refs[:1]})
    with pytest.raises(PromotionIntegrityError, match='every normalized candidate and evidence row'):
        _normalize_adapter_result(
            AdapterRunResult(
                candidate_records=_jsonl(*candidates),
                evidence_records=_jsonl(evidence),
                dispositions=_jsonl(unreferenced, excluded),
            ),
            inputs,
            ('not_relevant',),
        )


@pytest.mark.parametrize(
    ('allowed_codes', 'disposition_update', 'message'),
    (
        ((), {}, 'independently allowlisted'),
        (('not_relevant',), {'reason_code': 'posthoc_choice'}, 'independently allowlisted'),
        (('not_relevant',), {'source_artifact_sha256': '9' * 64}, 'exact verified source record'),
    ),
)
def test_adapter_rejects_unapproved_or_misbound_exclusion(
    allowed_codes: tuple[str, ...],
    disposition_update: dict[str, object],
    message: str,
) -> None:
    inputs, candidates, evidence, normalized, excluded = _normalization_unit_fixture()
    excluded = excluded.model_copy(update=disposition_update)
    with pytest.raises(PromotionIntegrityError, match=message):
        _normalize_adapter_result(
            AdapterRunResult(
                candidate_records=_jsonl(*candidates),
                evidence_records=_jsonl(evidence),
                dispositions=_jsonl(normalized, excluded),
            ),
            inputs,
            allowed_codes,
        )


@pytest.mark.parametrize(
    'path',
    (
        'C:/drive.bin',
        'unsafe\\path.bin',
        'unsafe\x00path.bin',
        'unsafe\npath.bin',
        'unsafe path.bin',
        'scope/CON/file.bin',
        'scope/trailing./file.bin',
    ),
)
def test_promotion_file_binding_rejects_nonportable_paths(path: str) -> None:
    with pytest.raises(ValueError, match='safe normalized relative POSIX'):
        PromotionFileBinding(path=path, sha256='a' * 64, byte_count=1)


def test_promotion_manifest_rejects_casefold_colliding_paths() -> None:
    capture_index = PromotionFileBinding(path='capture-index.json', sha256='a' * 64, byte_count=1)
    scope_precommit = PromotionFileBinding(
        path='scope/precommit/scope-precommit.json',
        sha256='b' * 64,
        byte_count=1,
    )
    files = tuple(
        sorted(
            (
                capture_index,
                scope_precommit,
                PromotionFileBinding(path='extra/Foo.bin', sha256='c' * 64, byte_count=1),
                PromotionFileBinding(path='extra/foo.bin', sha256='d' * 64, byte_count=1),
            ),
            key=lambda item: item.path,
        )
    )
    with pytest.raises(ValueError, match='case folding'):
        CapturePromotionManifest(
            promotion_id='casefold-test',
            campaign_id='casefold-campaign',
            selection_key='casefold-selection',
            selection_policy_sha256='e' * 64,
            selection_policy_artifact_sha256='f' * 64,
            plan_selection_commitment_sha256='1' * 64,
            selection_manifest_sha256='2' * 64,
            selected_at_upper_bound=_T0,
            created_at=_T0,
            capture_index=capture_index,
            scope_precommit=scope_precommit,
            files=files,
        )


def test_promotion_publication_preserves_target_created_at_install_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / 'promotion'
    original_rename = promotion_module.rename_directory_noreplace

    def race_at_install(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / 'owner.txt').write_bytes(b'created by competing publisher')
        original_rename(source, destination)

    monkeypatch.setattr(promotion_module, 'rename_directory_noreplace', race_at_install)
    with pytest.raises(PromotionIntegrityError, match='output already exists'):
        promotion_module._durable_publish(target, {'payload.bin': b'payload'}, b'{}')

    assert (target / 'owner.txt').read_bytes() == b'created by competing publisher'
    assert not (tmp_path / '.promotion.publish.lock').exists()
    assert not any(path.name.startswith('.promotion.') for path in tmp_path.iterdir())


def _static_fixture(store: OperationalStore, *, max_attempts_per_slot: int):
    request = HttpsCaptureRequest(
        url='https://public.example.org/release.json',
        allowed_host='public.example.org',
        max_body_bytes=4096,
    )
    plan = StaticHttpsCollectionPlan(
        plan_id='release-lifecycle-v1',
        source_id='publisher:lifecycle',
        artifacts=(StaticHttpsArtifactSpec(artifact_id='document', request=request),),
    )
    job = store.register_job(
        CaptureJobSpec(
            job_id='lifecycle-release',
            collector_id=STATIC_HTTPS_COLLECTOR_ID,
            schedule_anchor_at=_T0,
            schedule_interval_seconds=86400,
            configuration={
                'collection_plan_sha256': static_plan_sha256(plan),
                'dns_resolution_attempts': 1,
                'dns_resolution_timeout_seconds': 10,
                'lease_seconds': 3600,
                'max_dns_addresses': 16,
                'max_attempts_per_slot': max_attempts_per_slot,
                'max_total_body_bytes': 1024 * 1024,
                'plan_deadline_seconds': 300,
                'request_deadline_seconds': 60,
                'source_id': plan.source_id,
            },
        ),
        registered_at=_T0,
    )
    run = store.register_logical_run(job.spec_sha256, _T0, registered_at=_T0)
    scope = PromotionScopePolicy(
        policy_id='lifecycle-scope-v1',
        store_id=store.store_id,
        checkpoint_created_at_not_before=_T0,
        checkpoint_created_at_not_after=_T0 + timedelta(days=1),
        sources=(
            PromotionSourceScope(
                source_id=plan.source_id,
                job_spec_sha256s=(job.spec_sha256,),
                scheduled_from=_T0,
                scheduled_through=_T0,
            ),
        ),
    )
    return request, plan, job, run, scope


def test_source_release_rejects_mismatched_verified_record(tmp_path: Path) -> None:
    store = OperationalStore.initialize(
        tmp_path / 'operations',
        created_at=_T0,
        store_id='f' * 32,
        trusted_lease_clock=None,
    )
    request, plan, _job, run, _scope = _static_fixture(store, max_attempts_per_slot=1)
    transport = _Transport(_Response(b'{"released_at":"2026-07-13T12:00:00Z"}', request.url))
    with patch('vaxreplay.operations.collector.UrllibHttpsTransport', return_value=transport):
        result = run_static_https_collection(
            store,
            run.logical_run_id,
            plan,
            owner_id='worker-a',
            clock=_Clock(),
        )
    capture = _snapshot_one_capture(store, plan.source_id, result.attempt.attempt_id, store.events())
    body = next(artifact for artifact in capture.artifacts if artifact.role == 'body.document')
    source_record = SourceRecordBinding(
        source_id=plan.source_id,
        source_record_id='release-record',
        source_record_sha256=hashlib.sha256(body.payload).hexdigest(),
        source_artifact_sha256=body.sha256,
        source_locator='release.json#/records/0',
    )
    verified_records = _jsonl(source_record)
    implementation = b'release mismatch verifier'
    policy = b'release mismatch policy'

    def verifier(source_input, policy_bytes):
        assert policy_bytes == policy
        return SourceVerifierRunResult(
            result=SourceVerificationResult(
                source_id=source_input.source_id,
                verifier=SourceVerifierIdentity(
                    verifier_id='release-mismatch-verifier',
                    verifier_version='v1',
                    implementation_sha256=hashlib.sha256(implementation).hexdigest(),
                    execution_environment_sha256=hashlib.sha256(b'release mismatch verifier environment').hexdigest(),
                ),
                verifier_policy_sha256=hashlib.sha256(policy).hexdigest(),
                verified_attempt_ids=(capture.binding.attempt_id,),
                source_release=AuthoritativeSourceRelease(
                    source_release_at=_T0,
                    basis=AuthoritativeReleaseBasis.SOURCE_API_PUBLICATION_FIELD,
                    authority_locator='release.json#/released_at',
                    authority_field='released_at',
                    evidence_attempt_id=capture.binding.attempt_id,
                    evidence_role=body.role,
                    evidence_sha256=body.sha256,
                    evidence_source_record_id='different-record',
                    evidence_source_record_sha256=source_record.source_record_sha256,
                ),
                verified_capture_inventory_sha256=source_input.capture_inventory_sha256,
                verified_source_record_inventory_sha256=hashlib.sha256(verified_records).hexdigest(),
                verified_source_record_count=1,
                result_codes=('complete_release_verified',),
            ),
            verified_records=verified_records,
        )

    with pytest.raises(PromotionIntegrityError, match='exact verified source record and artifact'):
        _run_source_verifiers(
            (capture,),
            {
                plan.source_id: SourceVerifierSpec(
                    verifier_id='release-mismatch-verifier',
                    verifier_version='v1',
                    implementation_bytes=implementation,
                    policy_bytes=policy,
                    execution_environment_bytes=b'release mismatch verifier environment',
                    verifier=verifier,
                )
            },
            {},
        )


def test_v0_promotion_rejects_retry_selective_omission_on_build_and_offline_load(
    tmp_path: Path,
) -> None:
    store = OperationalStore.initialize(
        tmp_path / 'operations',
        created_at=_T0,
        store_id='c' * 32,
        trusted_lease_clock=None,
    )
    request, plan, job, run, scope = _static_fixture(store, max_attempts_per_slot=2)
    plan_artifact = store.put_bytes(canonical_json_bytes(plan), recorded_at=_T0 + timedelta(seconds=1))
    first = store.begin_attempt(
        run.logical_run_id,
        owner_id='worker-failed',
        now=_T0 + timedelta(seconds=2),
        initial_artifacts={'collection-plan': plan_artifact.sha256},
    )
    retained = store.put_bytes(b'retained failure diagnostic', recorded_at=_T0 + timedelta(seconds=3))
    store.attach_artifact(
        first.attempt_id,
        owner_id='worker-failed',
        role='failure-diagnostic',
        artifact_sha256=retained.sha256,
        now=_T0 + timedelta(seconds=4),
    )
    store.fail_attempt(
        first.attempt_id,
        owner_id='worker-failed',
        terminal_code='source_unavailable',
        now=_T0 + timedelta(seconds=5),
    )
    transport = _Transport(_Response(b'{"released_at":"2026-07-13T12:00:00Z"}', request.url))
    with patch('vaxreplay.operations.collector.UrllibHttpsTransport', return_value=transport):
        second = run_static_https_collection(
            store,
            run.logical_run_id,
            plan,
            owner_id='worker-success',
            clock=_OffsetClock(6),
        )
    assert second.attempt.attempt_number == 2
    assert 'failure-diagnostic' in store.list_attempt_artifacts(first.attempt_id)
    events = store.events()
    store.verify(verified_at=_T0 + timedelta(minutes=2))
    verify_all_static_run_manifests(store)

    with pytest.raises(PromotionIntegrityError, match='max_attempts_per_slot == 1'):
        _load_scoped_jobs(store, scope, events)
    with pytest.raises(PromotionIntegrityError, match='max_attempts_per_slot == 1'):
        _parse_registered_jobs(_jsonl(job), events, scope)


@pytest.mark.parametrize('time_case', ('before-start', 'after-success'))
def test_offline_static_lifecycle_rejects_tampered_non_plan_attachment_time(
    tmp_path: Path,
    time_case: str,
) -> None:
    store = OperationalStore.initialize(
        tmp_path / 'operations',
        created_at=_T0,
        store_id='d' * 32,
        trusted_lease_clock=None,
    )
    request, plan, job, run, _scope = _static_fixture(store, max_attempts_per_slot=1)
    transport = _Transport(_Response(b'{"released_at":"2026-07-13T12:00:00Z"}', request.url))
    with patch('vaxreplay.operations.collector.UrllibHttpsTransport', return_value=transport):
        result = run_static_https_collection(
            store,
            run.logical_run_id,
            plan,
            owner_id='worker-a',
            clock=_Clock(),
        )
    events = store.events()
    capture = _snapshot_one_capture(store, plan.source_id, result.attempt.attempt_id, events)
    configuration = parse_static_job_configuration(job.spec.configuration)
    _verify_static_attempt_history(events, capture.binding, configuration)

    body_attachment = next(
        event
        for event in events
        if event.event_type is LedgerEventType.ATTEMPT_ARTIFACT_ATTACHED
        and event.payload.get('attempt_id') == result.attempt.attempt_id
        and event.payload.get('role') == 'body.document'
    )
    tampered_at = (
        capture.binding.attempt_started_at - timedelta(microseconds=1)
        if time_case == 'before-start'
        else capture.binding.captured_at + timedelta(microseconds=1)
    )
    tampered = body_attachment.model_copy(update={'occurred_at': tampered_at})
    tampered_events = tuple(tampered if event is body_attachment else event for event in events)
    with pytest.raises(PromotionIntegrityError, match='attachment disagrees'):
        _verify_static_attempt_history(tampered_events, capture.binding, configuration)


@pytest.mark.parametrize(
    ('event_type', 'payload_update', 'message'),
    (
        (LedgerEventType.ATTEMPT_STARTED, {'attempt_number': 2}, 'attempt start'),
        (LedgerEventType.ATTEMPT_SUCCEEDED, {'terminal_code': 'not-success'}, 'exact V0 terminal'),
    ),
)
def test_offline_static_lifecycle_rejects_tampered_start_and_terminal_fields(
    tmp_path: Path,
    event_type: LedgerEventType,
    payload_update: dict[str, object],
    message: str,
) -> None:
    store = OperationalStore.initialize(
        tmp_path / 'operations',
        created_at=_T0,
        store_id='e' * 32,
        trusted_lease_clock=None,
    )
    request, plan, job, run, _scope = _static_fixture(store, max_attempts_per_slot=1)
    transport = _Transport(_Response(b'{"released_at":"2026-07-13T12:00:00Z"}', request.url))
    with patch('vaxreplay.operations.collector.UrllibHttpsTransport', return_value=transport):
        result = run_static_https_collection(
            store,
            run.logical_run_id,
            plan,
            owner_id='worker-a',
            clock=_Clock(),
        )
    events = store.events()
    capture = _snapshot_one_capture(store, plan.source_id, result.attempt.attempt_id, events)
    configuration = parse_static_job_configuration(job.spec.configuration)
    target = next(
        event
        for event in events
        if event.event_type is event_type and event.payload.get('attempt_id') == result.attempt.attempt_id
    )
    tampered = target.model_copy(update={'payload': {**target.payload, **payload_update}})
    tampered_events = tuple(tampered if event is target else event for event in events)
    with pytest.raises(PromotionIntegrityError, match=message):
        _verify_static_attempt_history(tampered_events, capture.binding, configuration)


@pytest.mark.parametrize(
    ('clock', 'configuration_update', 'message'),
    (
        (_Clock(), {'max_total_body_bytes': 1}, 'aggregate body-byte budget'),
        (
            _Clock(),
            {
                'dns_resolution_timeout_seconds': 1,
                'plan_deadline_seconds': 1,
                'request_deadline_seconds': 1,
            },
            'plan deadline',
        ),
        (
            _StepClock(2),
            {
                'dns_resolution_timeout_seconds': 1,
                'request_deadline_seconds': 1,
            },
            'body/receipt replay',
        ),
    ),
)
def test_portable_static_replay_reenforces_resource_policy(
    tmp_path: Path,
    clock,
    configuration_update: dict[str, int],
    message: str,
) -> None:
    store = OperationalStore.initialize(
        tmp_path / message.replace('/', '-').replace(' ', '-'),
        created_at=_T0,
        store_id=hashlib.sha256(message.encode()).hexdigest()[:32],
        trusted_lease_clock=None,
    )
    request, plan, job, run, _scope = _static_fixture(store, max_attempts_per_slot=1)
    transport = _Transport(_Response(b'{"released_at":"2026-07-13T12:00:00Z"}', request.url))
    with patch('vaxreplay.operations.collector.UrllibHttpsTransport', return_value=transport):
        result = run_static_https_collection(
            store,
            run.logical_run_id,
            plan,
            owner_id='worker-a',
            clock=clock,
        )
    capture = _snapshot_one_capture(store, plan.source_id, result.attempt.attempt_id, store.events())
    configuration = parse_static_job_configuration({**job.spec.configuration, **configuration_update})

    with pytest.raises(PromotionIntegrityError, match=message):
        promotion_module._verify_portable_static_capture(capture, configuration)


def test_witnessed_capture_promotion_round_trip() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        store = OperationalStore.initialize(
            root / 'operations',
            created_at=_T0,
            store_id='a' * 32,
            trusted_lease_clock=None,
        )
        request = HttpsCaptureRequest(
            url='https://public.example.org/release.json',
            allowed_host='public.example.org',
            max_body_bytes=4096,
        )
        plan = StaticHttpsCollectionPlan(
            plan_id='release-v1',
            source_id='publisher:example',
            artifacts=(StaticHttpsArtifactSpec(artifact_id='document', request=request),),
        )
        job = store.register_job(
            CaptureJobSpec(
                job_id='daily-release',
                collector_id=STATIC_HTTPS_COLLECTOR_ID,
                schedule_anchor_at=_T0 + timedelta(seconds=1),
                schedule_interval_seconds=86400,
                configuration={
                    'collection_plan_sha256': static_plan_sha256(plan),
                    'dns_resolution_attempts': 1,
                    'dns_resolution_timeout_seconds': 10,
                    'lease_seconds': 3600,
                    'max_dns_addresses': 16,
                    'max_attempts_per_slot': 1,
                    'max_total_body_bytes': 1024 * 1024,
                    'plan_deadline_seconds': 300,
                    'request_deadline_seconds': 60,
                    'source_id': plan.source_id,
                },
            ),
            registered_at=_T0,
        )
        witness_policy_bytes = b'public witness submission policy v1'
        trust_policy_bytes = b'pinned independent log root v1'
        witness_verifier_bytes = b'offline witness verifier implementation v1'
        witness_policy = WitnessPolicyBinding(
            authority_id='independent-log',
            method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
            policy_id='witness-policy-v1',
            policy_sha256=hashlib.sha256(witness_policy_bytes).hexdigest(),
            trust_policy_id='trust-policy-v1',
            trust_policy_sha256=hashlib.sha256(trust_policy_bytes).hexdigest(),
            verifier_id='witness-verifier-v1',
            verifier_implementation_sha256=hashlib.sha256(witness_verifier_bytes).hexdigest(),
        )
        scope_witness_policy_bytes = b'scope precommit witness policy v1'
        scope_trust_policy_bytes = b'scope precommit trust root v1'
        scope_witness_verifier_bytes = b'scope precommit verifier implementation v1'
        scope_witness_policy = WitnessPolicyBinding(
            authority_id='scope-independent-log',
            method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
            policy_id='scope-witness-policy-v1',
            policy_sha256=hashlib.sha256(scope_witness_policy_bytes).hexdigest(),
            trust_policy_id='scope-trust-policy-v1',
            trust_policy_sha256=hashlib.sha256(scope_trust_policy_bytes).hexdigest(),
            verifier_id='scope-witness-verifier-v1',
            verifier_implementation_sha256=hashlib.sha256(scope_witness_verifier_bytes).hexdigest(),
        )

        def witness_verifier(target, proof, expected_policy):
            assert target == canonical_json_bytes(checkpoint)
            assert proof == b'external inclusion proof'
            assert expected_policy == witness_policy
            return AuthenticatedExternalWitnessFacts(
                receipt_id='receipt-1',
                authority_id=witness_policy.authority_id,
                witness_id='independent-log-key',
                method=witness_policy.method,
                policy_id=witness_policy.policy_id,
                checkpoint_sha256=checkpoint_sha256(checkpoint),
                witnessed_at=_T0 + timedelta(seconds=41),
            )

        source_policy = b'publisher release-field verifier policy'
        source_implementation = b'publisher verifier implementation'

        def source_verifier(source_input, policy_bytes):
            assert policy_bytes == source_policy
            body = next(artifact for artifact in source_input.captures[0].artifacts if artifact.role == 'body.document')
            source_record = SourceRecordBinding(
                source_id=source_input.source_id,
                source_record_id='release-record-1',
                source_record_sha256=hashlib.sha256(body.payload).hexdigest(),
                source_artifact_sha256=body.sha256,
                source_locator='release.json#/records/0',
            )
            verified_records = _jsonl(source_record)
            return SourceVerifierRunResult(
                result=SourceVerificationResult(
                    source_id=source_input.source_id,
                    verifier=SourceVerifierIdentity(
                        verifier_id='publisher-verifier',
                        verifier_version='v1',
                        implementation_sha256=hashlib.sha256(source_implementation).hexdigest(),
                        execution_environment_sha256=hashlib.sha256(b'publisher verifier environment lock').hexdigest(),
                    ),
                    verifier_policy_sha256=hashlib.sha256(source_policy).hexdigest(),
                    verified_attempt_ids=tuple(sorted(capture.binding.attempt_id for capture in source_input.captures)),
                    source_release=AuthoritativeSourceRelease(
                        source_release_at=_T0,
                        basis=AuthoritativeReleaseBasis.SOURCE_API_PUBLICATION_FIELD,
                        authority_locator='release.json#/released_at',
                        authority_field='released_at',
                        evidence_attempt_id=source_input.captures[0].binding.attempt_id,
                        evidence_role=body.role,
                        evidence_sha256=body.sha256,
                        evidence_source_record_id=source_record.source_record_id,
                        evidence_source_record_sha256=source_record.source_record_sha256,
                    ),
                    verified_capture_inventory_sha256=source_input.capture_inventory_sha256,
                    verified_source_record_inventory_sha256=hashlib.sha256(verified_records).hexdigest(),
                    verified_source_record_count=1,
                    result_codes=('complete_release_verified',),
                ),
                verified_records=verified_records,
            )

        candidate = CandidateRecord(episode_id='future-episode', candidate_id='candidate-a')
        candidate_b = CandidateRecord(episode_id='future-episode', candidate_id='candidate-b')
        evidence_body = 'Publisher release captured before the decision cutoff.'
        evidence = EvidenceRecord(
            episode_id='future-episode',
            evidence_id='evidence-a',
            source_type=SourceType.PUBLIC_HEALTH,
            collected_at=_T0,
            available_at=_T0,
            title='Publisher release',
            body=evidence_body,
            body_sha256=hashlib.sha256(evidence_body.encode()).hexdigest(),
            related_candidate_ids=['candidate-a'],
            provenance_url='https://public.example.org/release.json',
            license_id='public-domain',
            derivation='deterministic fixture adapter',
        )
        adapter_policy = b'deterministic adapter policy'
        disposition = SourceRecordDisposition(
            source_id=plan.source_id,
            source_record_id='release-record-1',
            source_record_sha256=hashlib.sha256(b'{"released_at":"2026-07-13T12:00:00Z"}').hexdigest(),
            source_artifact_sha256=hashlib.sha256(b'{"released_at":"2026-07-13T12:00:00Z"}').hexdigest(),
            disposition='normalized',
            candidate_record_refs=tuple(
                sorted(
                    (
                        NormalizedRecordReference(
                            episode_id=record.episode_id,
                            record_id=record.candidate_id,
                            record_sha256=hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
                        )
                        for record in (candidate, candidate_b)
                    ),
                    key=lambda item: (item.episode_id, item.record_id),
                )
            ),
            evidence_record_refs=(
                NormalizedRecordReference(
                    episode_id=evidence.episode_id,
                    record_id=evidence.evidence_id,
                    record_sha256=hashlib.sha256(canonical_json_bytes(evidence)).hexdigest(),
                ),
            ),
        )
        adapter_spec = AdapterSpec(
            adapter_id='fixture-adapter',
            adapter_version='v1',
            implementation_bytes=b'fixture adapter implementation',
            policy_bytes=adapter_policy,
            execution_environment_bytes=b'python fixture lock',
            adapter=lambda _inputs, policy: (
                AdapterRunResult(
                    candidate_records=_jsonl(candidate, candidate_b),
                    evidence_records=_jsonl(evidence),
                    dispositions=_jsonl(disposition),
                )
                if policy == adapter_policy
                else (_ for _ in ()).throw(AssertionError('wrong adapter policy'))
            ),
        )
        source_specs = {
            plan.source_id: SourceVerifierSpec(
                verifier_id='publisher-verifier',
                verifier_version='v1',
                implementation_bytes=source_implementation,
                policy_bytes=source_policy,
                execution_environment_bytes=b'publisher verifier environment lock',
                verifier=source_verifier,
            )
        }
        witness_materials = WitnessMaterialSpec(
            policy=witness_policy,
            policy_bytes=witness_policy_bytes,
            trust_policy_bytes=trust_policy_bytes,
            verifier_implementation_bytes=witness_verifier_bytes,
            verifier=witness_verifier,
        )
        scope = PromotionScopePolicy(
            policy_id='promotion-scope-v1',
            store_id='a' * 32,
            checkpoint_created_at_not_before=_T0 + timedelta(seconds=39),
            checkpoint_created_at_not_after=_T0 + timedelta(seconds=40),
            sources=(
                PromotionSourceScope(
                    source_id=plan.source_id,
                    job_spec_sha256s=(job.spec_sha256,),
                    scheduled_from=_T0 + timedelta(seconds=1),
                    scheduled_through=_T0 + timedelta(seconds=1),
                ),
            ),
        )
        selection_policy, _selection_policy_bytes, _selection_trust_bytes, _selection_verifier_bytes = (
            _plan_selection_policy()
        )
        pre_capture_plan = derive_pre_capture_plan(
            scope_policy=scope,
            selection_policy=selection_policy,
            capture_witness_policy=witness_policy,
            source_verifiers=source_specs,
            adapter=adapter_spec,
        )
        assert (
            pre_capture_plan.source_verifiers[0].execution_environment_sha256
            == hashlib.sha256(source_specs[plan.source_id].execution_environment_bytes).hexdigest()
        )
        store.put_bytes(canonical_json_bytes(scope), recorded_at=_T0)
        store.put_bytes(canonical_json_bytes(pre_capture_plan), recorded_at=_T0)
        precommit_checkpoint = store.checkpoint(created_at=_T0)
        selection_commitment = derive_plan_selection_commitment(
            scope,
            pre_capture_plan,
            precommit_checkpoint,
        )
        plan_selection, selection_materials = _plan_selection_fixture(
            root / 'plan-selection',
            commitment=selection_commitment,
        )

        def scope_witness_verifier(target, proof, expected_policy):
            assert target == canonical_json_bytes(precommit_checkpoint)
            assert proof == b'scope precommit inclusion proof'
            assert expected_policy == scope_witness_policy
            return AuthenticatedExternalWitnessFacts(
                receipt_id='scope-precommit-receipt',
                authority_id=scope_witness_policy.authority_id,
                witness_id='independent-log-key',
                method=scope_witness_policy.method,
                policy_id=scope_witness_policy.policy_id,
                checkpoint_sha256=checkpoint_sha256(precommit_checkpoint),
                witnessed_at=_T0,
            )

        scope_witness = broker_witness_checkpoint(
            root / 'scope-witness',
            checkpoint=precommit_checkpoint,
            policy=scope_witness_policy,
            provider=lambda _request: (
                ExternalWitnessClaim(verification_uri='https://log.example/scope-precommit'),
                b'scope precommit inclusion proof',
            ),
            verifier=scope_witness_verifier,
            verified_at=_T0 + timedelta(microseconds=1),
        )
        scope_witness_materials = WitnessMaterialSpec(
            policy=scope_witness_policy,
            policy_bytes=scope_witness_policy_bytes,
            trust_policy_bytes=scope_trust_policy_bytes,
            verifier_implementation_bytes=scope_witness_verifier_bytes,
            verifier=scope_witness_verifier,
        )
        scope_precommit = build_scope_precommit(
            root / 'scope-precommit',
            store=store,
            scope_policy=scope,
            pre_capture_plan=pre_capture_plan,
            witness_root=scope_witness.root,
            witness_materials=scope_witness_materials,
            selection_root=plan_selection.root,
            expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            selection_materials=selection_materials,
            created_at=_T0 + timedelta(microseconds=2),
            verified_at=_T0 + timedelta(microseconds=3),
        )

        run = store.register_logical_run(
            job.spec_sha256,
            _T0 + timedelta(seconds=1),
            registered_at=_T0 + timedelta(seconds=1),
        )
        transport = _Transport(_Response(b'{"released_at":"2026-07-13T12:00:00Z"}', request.url))
        with patch('vaxreplay.operations.collector.UrllibHttpsTransport', return_value=transport):
            run_static_https_collection(
                store,
                run.logical_run_id,
                plan,
                owner_id='worker-a',
                clock=_Clock(),
            )
        checkpoint = store.checkpoint(
            created_at=_T0 + timedelta(seconds=40),
            semantic_verifier=lambda: verify_all_static_run_manifests(store),
        )
        capture_witness = broker_witness_checkpoint(
            root / 'witness',
            checkpoint=checkpoint,
            policy=witness_policy,
            provider=lambda _request: (
                ExternalWitnessClaim(verification_uri='https://log.example/receipt-1'),
                b'external inclusion proof',
            ),
            verifier=witness_verifier,
            verified_at=_T0 + timedelta(seconds=42),
        )
        built = build_capture_promotion(
            root / 'promotion',
            promotion_id='promotion-1',
            store=store,
            witness_root=capture_witness.root,
            witness_materials=witness_materials,
            scope_policy=scope,
            scope_precommit_root=scope_precommit.root,
            scope_precommit_witness_materials=scope_witness_materials,
            expected_scope_precommit_sha256=scope_precommit.archive_sha256,
            selection_materials=selection_materials,
            expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            source_verifiers=source_specs,
            adapter=adapter_spec,
            created_at=_T0 + timedelta(seconds=43),
            verified_at=_T0 + timedelta(seconds=44),
        )
        loaded = load_capture_promotion(
            built.root,
            expected_scope_policy=scope,
            scope_precommit_witness_materials=scope_witness_materials,
            witness_materials=witness_materials,
            source_verifiers=source_specs,
            adapter=adapter_spec,
            verified_at=_T0 + timedelta(seconds=45),
            expected_scope_precommit_sha256=scope_precommit.archive_sha256,
            expected_promotion_sha256=built.manifest_sha256,
            selection_materials=selection_materials,
            expected_selection_manifest_sha256=plan_selection.manifest_sha256,
        )
        assert loaded.index == built.index
        assert loaded.candidates == (candidate, candidate_b)
        assert loaded.evidence == (evidence,)
        assert loaded.source_captures[0].source_id == 'promotion:promotion-1'
        assert loaded.source_captures[0].witnessed_at == _T0 + timedelta(seconds=41)
        assert loaded.handoff_descriptor.promotion_manifest_sha256 == built.manifest_sha256
        assert loaded.handoff_descriptor.promotion_created_at == built.manifest.created_at
        assert loaded.index.campaign_id == selection_policy.campaign_id
        assert loaded.index.selection_key == selection_policy.selection_key
        assert loaded.index.selection_manifest_sha256 == plan_selection.manifest_sha256
        assert loaded.index.selected_at_upper_bound == _T0
        assert loaded.manifest.selection_manifest_sha256 == plan_selection.manifest_sha256
        assert loaded.handoff_descriptor.selection_manifest_sha256 == plan_selection.manifest_sha256
        stale_binding = loaded.index.scope_precommit.model_dump()
        stale_binding['schema_version'] = 'vaxreplay.scope-precommit-promotion-binding.v0.1'
        with pytest.raises(ValueError, match='schema_version'):
            type(loaded.index.scope_precommit).model_validate(stale_binding)
        for model, value, stale_schema in (
            (CaptureIndex, loaded.index, 'vaxreplay.capture-index.v0.3'),
            (CapturePromotionManifest, loaded.manifest, 'vaxreplay.capture-promotion.v0.3'),
            (
                type(loaded.handoff_descriptor),
                loaded.handoff_descriptor,
                'vaxreplay.promotion-handoff.v0.3',
            ),
        ):
            stale = value.model_dump()
            stale['schema_version'] = stale_schema
            with pytest.raises(ValueError, match='schema_version'):
                model.model_validate(stale)
        assert loaded.source_captures[0].manifest_bytes == loaded.handoff_descriptor_bytes
        assert loaded.source_captures[0].manifest_bytes != loaded.index_bytes
        source_verification = loaded.index.source_verifications[0]
        assert (
            source_verification.result.verifier.execution_environment_sha256
            == hashlib.sha256(source_specs[plan.source_id].execution_environment_bytes).hexdigest()
        )
        assert (
            built.root.joinpath(source_verification.verifier_execution_environment.path).read_bytes()
            == source_specs[plan.source_id].execution_environment_bytes
        )

        promotion_parent_link = root / 'promotion-parent-link'
        promotion_parent_link.symlink_to(built.root.parent, target_is_directory=True)
        with pytest.raises(PromotionIntegrityError, match='unsafe capture promotion tree'):
            load_capture_promotion(
                promotion_parent_link / built.root.name,
                expected_scope_policy=scope,
                scope_precommit_witness_materials=scope_witness_materials,
                witness_materials=witness_materials,
                source_verifiers=source_specs,
                adapter=adapter_spec,
                verified_at=_T0 + timedelta(seconds=45),
                expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                expected_promotion_sha256=built.manifest_sha256,
                selection_materials=selection_materials,
                expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            )

        changed_allowlist_adapter = AdapterSpec(
            adapter_id=adapter_spec.adapter_id,
            adapter_version=adapter_spec.adapter_version,
            implementation_bytes=adapter_spec.implementation_bytes,
            policy_bytes=adapter_spec.policy_bytes,
            execution_environment_bytes=adapter_spec.execution_environment_bytes,
            adapter=adapter_spec.adapter,
            allowed_exclusion_reason_codes=('unsupported_record',),
        )
        with pytest.raises(PromotionIntegrityError, match='pre-capture plan|out-of-band'):
            load_capture_promotion(
                built.root,
                expected_scope_policy=scope,
                scope_precommit_witness_materials=scope_witness_materials,
                witness_materials=witness_materials,
                source_verifiers=source_specs,
                adapter=changed_allowlist_adapter,
                verified_at=_T0 + timedelta(seconds=45),
                expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                expected_promotion_sha256=built.manifest_sha256,
                selection_materials=selection_materials,
                expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            )

        changed_environment_source_specs = {
            plan.source_id: SourceVerifierSpec(
                verifier_id=source_specs[plan.source_id].verifier_id,
                verifier_version=source_specs[plan.source_id].verifier_version,
                implementation_bytes=source_specs[plan.source_id].implementation_bytes,
                policy_bytes=source_specs[plan.source_id].policy_bytes,
                execution_environment_bytes=b'different publisher verifier environment lock',
                verifier=source_specs[plan.source_id].verifier,
            )
        }
        with pytest.raises(PromotionIntegrityError, match='pre-capture plan|out-of-band'):
            load_capture_promotion(
                built.root,
                expected_scope_policy=scope,
                scope_precommit_witness_materials=scope_witness_materials,
                witness_materials=witness_materials,
                source_verifiers=changed_environment_source_specs,
                adapter=adapter_spec,
                verified_at=_T0 + timedelta(seconds=45),
                expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                expected_promotion_sha256=built.manifest_sha256,
                selection_materials=selection_materials,
                expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            )

        branch_store = OperationalStore.initialize(
            root / 'branch-operations',
            created_at=_T0,
            store_id='a' * 32,
            trusted_lease_clock=None,
        )
        branch_job = branch_store.register_job(job.spec, registered_at=_T0)
        assert branch_job.spec_sha256 == job.spec_sha256
        branch_selection_policy, *_branch_selection_material_bytes = _plan_selection_policy(
            'fixture-branch-campaign-2027',
            'branch-antigen-plan',
        )
        branch_pre_capture_plan = derive_pre_capture_plan(
            scope_policy=scope,
            selection_policy=branch_selection_policy,
            capture_witness_policy=witness_policy,
            source_verifiers=source_specs,
            adapter=adapter_spec,
        )
        branch_store.put_bytes(b'branch-only-ledger-object', recorded_at=_T0)
        branch_store.put_bytes(canonical_json_bytes(scope), recorded_at=_T0)
        branch_store.put_bytes(canonical_json_bytes(branch_pre_capture_plan), recorded_at=_T0)
        branch_checkpoint = branch_store.checkpoint(created_at=_T0)
        branch_plan_selection, branch_selection_materials = _plan_selection_fixture(
            root / 'branch-plan-selection',
            commitment=derive_plan_selection_commitment(
                scope,
                branch_pre_capture_plan,
                branch_checkpoint,
            ),
            campaign_id='fixture-branch-campaign-2027',
            selection_key='branch-antigen-plan',
        )

        def branch_scope_witness_verifier(target, proof, expected_policy):
            assert target == canonical_json_bytes(branch_checkpoint)
            assert proof == b'branch scope inclusion proof'
            assert expected_policy == scope_witness_policy
            return AuthenticatedExternalWitnessFacts(
                receipt_id='branch-scope-precommit-receipt',
                authority_id=scope_witness_policy.authority_id,
                witness_id='independent-log-key',
                method=scope_witness_policy.method,
                policy_id=scope_witness_policy.policy_id,
                checkpoint_sha256=checkpoint_sha256(branch_checkpoint),
                witnessed_at=_T0,
            )

        branch_scope_witness = broker_witness_checkpoint(
            root / 'branch-scope-witness',
            checkpoint=branch_checkpoint,
            policy=scope_witness_policy,
            provider=lambda _request: (
                ExternalWitnessClaim(verification_uri='https://log.example/branch-scope'),
                b'branch scope inclusion proof',
            ),
            verifier=branch_scope_witness_verifier,
            verified_at=_T0 + timedelta(microseconds=1),
        )
        branch_scope_materials = WitnessMaterialSpec(
            policy=scope_witness_policy,
            policy_bytes=scope_witness_policy_bytes,
            trust_policy_bytes=scope_trust_policy_bytes,
            verifier_implementation_bytes=scope_witness_verifier_bytes,
            verifier=branch_scope_witness_verifier,
        )
        branch_precommit = build_scope_precommit(
            root / 'branch-scope-precommit',
            store=branch_store,
            scope_policy=scope,
            pre_capture_plan=branch_pre_capture_plan,
            witness_root=branch_scope_witness.root,
            witness_materials=branch_scope_materials,
            selection_root=branch_plan_selection.root,
            expected_selection_manifest_sha256=branch_plan_selection.manifest_sha256,
            selection_materials=branch_selection_materials,
            created_at=_T0 + timedelta(microseconds=2),
            verified_at=_T0 + timedelta(microseconds=3),
        )
        with pytest.raises(PromotionIntegrityError, match='capture-ledger prefix|checkpoint head'):
            build_capture_promotion(
                root / 'branch-prefix-promotion',
                promotion_id='branch-prefix-promotion',
                store=store,
                witness_root=capture_witness.root,
                witness_materials=witness_materials,
                scope_policy=scope,
                scope_precommit_root=branch_precommit.root,
                scope_precommit_witness_materials=branch_scope_materials,
                expected_scope_precommit_sha256=branch_precommit.archive_sha256,
                selection_materials=branch_selection_materials,
                expected_selection_manifest_sha256=branch_plan_selection.manifest_sha256,
                source_verifiers=source_specs,
                adapter=adapter_spec,
                created_at=_T0 + timedelta(seconds=43),
                verified_at=_T0 + timedelta(seconds=44),
            )

        # A release attached to a later evidence capture must not be compared to
        # an unrelated earlier slot from the same source.
        first_capture = built.index.captures[0]
        second_attempt_id = f'attempt-{"0" * 32}'
        if second_attempt_id == first_capture.attempt_id:
            second_attempt_id = f'attempt-{"f" * 32}'
        second_capture = first_capture.model_copy(
            update={
                'attempt_id': second_attempt_id,
                'logical_run_id': f'run-{"f" * 64}',
                'captured_at': first_capture.captured_at + timedelta(seconds=2),
            }
        )
        captures = tuple(
            sorted(
                (first_capture, second_capture),
                key=lambda capture: (capture.source_id, capture.succeeded_event_sequence, capture.attempt_id),
            )
        )
        verification = built.index.source_verifications[0]
        source_release = verification.result.source_release.model_copy(
            update={
                'source_release_at': first_capture.captured_at + timedelta(seconds=1),
                'evidence_attempt_id': second_attempt_id,
            }
        )
        inventory_sha256 = 'c' * 64
        result = verification.result.model_copy(
            update={
                'verified_attempt_ids': tuple(sorted((first_capture.attempt_id, second_attempt_id))),
                'source_release': source_release,
                'verified_capture_inventory_sha256': inventory_sha256,
            }
        )
        second_disposition = verification.run_dispositions[0].model_copy(
            update={
                'attempt_id': second_attempt_id,
                'logical_run_id': second_capture.logical_run_id,
            }
        )
        dispositions = tuple(
            sorted(
                (*verification.run_dispositions, second_disposition),
                key=lambda disposition: (disposition.succeeded_event_sequence, disposition.attempt_id),
            )
        )
        result_sha256 = source_verification_result_sha256(result)
        source_verifications = (
            verification.model_copy(
                update={
                    'result': result,
                    'result_sha256': result_sha256,
                    'run_dispositions': dispositions,
                }
            ),
        )
        adapter_inputs = (
            built.index.adapter.input_inventories[0].model_copy(
                update={
                    'capture_inventory_sha256': inventory_sha256,
                    'source_verification_result_sha256': result_sha256,
                }
            ),
        )
        CaptureIndex.model_validate(
            built.index.model_copy(
                update={
                    'captures': captures,
                    'source_verifications': source_verifications,
                    'adapter': built.index.adapter.model_copy(update={'input_inventories': adapter_inputs}),
                }
            ).model_dump()
        )

        prospective = build_prospective_decision_package_from_promotion(
            root / 'prospective',
            promotion_root=built.root,
            expected_scope_policy=scope,
            scope_precommit_witness_materials=scope_witness_materials,
            witness_materials=witness_materials,
            source_verifiers=source_specs,
            adapter=adapter_spec,
            verified_at=_T0 + timedelta(seconds=45),
            config=DecisionTimeConfig(
                episode_id='future-episode',
                lineage_group_id='future-lineage',
                synthetic=False,
                task_type='antigen_target_prioritization',
                split=Split.TEST,
                decision_at=_T0 + timedelta(days=1),
                portfolio_size=1,
                candidate_ids=('candidate-a', 'candidate-b'),
                forecast_targets=(ForecastTarget(target_id='advancement', horizon_days=30),),
                required_dimensions=('immunogenicity',),
                adjudication_version='v1',
                reward_version='v0.1',
            ),
            protocol_artifacts={
                'candidate_set_definition': b'complete deterministic candidate rule',
                'evidence_acquisition_spec': b'exact witnessed source capture rule',
                'outcome_adjudication_spec': b'fixed future outcome adjudication rule',
            },
            expected_scope_precommit_sha256=scope_precommit.archive_sha256,
            expected_promotion_sha256=built.manifest_sha256,
            selection_materials=selection_materials,
            expected_selection_manifest_sha256=plan_selection.manifest_sha256,
        )
        assert prospective.candidates == loaded.candidates
        assert prospective.evidence == loaded.evidence
        assert prospective.manifest.source_captures[0].source_id == 'promotion:promotion-1'
        assert (
            prospective.manifest.episode.decision_snapshot.protocol_commitments.candidate_set_available_at
            == built.manifest.created_at
        )
        assert set(inspect.signature(build_prospective_decision_package_from_promotion).parameters).isdisjoint(
            {'candidates', 'evidence', 'loaded_promotion'}
        )
        assert 'loaded_promotion' not in inspect.signature(make_promotion_source_capture_verifier).parameters
        for required_selection_parameter in (
            'selection_materials',
            'expected_selection_manifest_sha256',
        ):
            assert (
                inspect.signature(load_capture_promotion).parameters[required_selection_parameter].default
                is inspect.Parameter.empty
            )

        source_capture_policy = b'promotion admission policy v1'
        admission_verifier = make_promotion_source_capture_verifier(
            promotion_root=built.root,
            expected_scope_policy=scope,
            scope_precommit_witness_materials=scope_witness_materials,
            witness_materials=witness_materials,
            source_verifiers=source_specs,
            adapter=adapter_spec,
            verified_at=_T0 + timedelta(seconds=45),
            expected_scope_precommit_sha256=scope_precommit.archive_sha256,
            expected_promotion_sha256=built.manifest_sha256,
            selection_materials=selection_materials,
            expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            expected_source_capture_policy=source_capture_policy,
        )
        prospective_binding = prospective.manifest.source_captures[0]
        assert admission_verifier(
            prospective_binding,
            prospective.source_capture_artifacts[prospective_binding.source_id],
            source_capture_policy,
        )
        fake_descriptor = loaded.handoff_descriptor.model_copy(update={'promotion_manifest_sha256': 'f' * 64})
        fake_bytes = canonical_json_bytes(fake_descriptor)
        fake_binding = prospective_binding.model_copy(
            update={
                'file': prospective_binding.file.model_copy(
                    update={
                        'sha256': hashlib.sha256(fake_bytes).hexdigest(),
                        'byte_count': len(fake_bytes),
                    }
                )
            }
        )
        assert not admission_verifier(fake_binding, fake_bytes, source_capture_policy)
        assert not admission_verifier(
            prospective_binding,
            prospective.source_capture_artifacts[prospective_binding.source_id],
            b'wrong policy',
        )

        with pytest.raises(PromotionIntegrityError, match='expected digest'):
            load_capture_promotion(
                built.root,
                expected_scope_policy=scope,
                scope_precommit_witness_materials=scope_witness_materials,
                witness_materials=witness_materials,
                source_verifiers=source_specs,
                adapter=adapter_spec,
                verified_at=_T0 + timedelta(seconds=45),
                expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                expected_promotion_sha256='0' * 64,
                selection_materials=selection_materials,
                expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            )

        with pytest.raises(PromotionIntegrityError, match='plan-selection|plan selection'):
            load_capture_promotion(
                built.root,
                expected_scope_policy=scope,
                scope_precommit_witness_materials=scope_witness_materials,
                witness_materials=witness_materials,
                source_verifiers=source_specs,
                adapter=adapter_spec,
                verified_at=_T0 + timedelta(seconds=45),
                expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                expected_promotion_sha256=built.manifest_sha256,
                selection_materials=selection_materials,
                expected_selection_manifest_sha256='f' * 64,
            )

        changed_selection_policy_bytes = b'different first-write-wins policy bytes'
        changed_selection_policy = selection_policy.model_copy(
            update={
                'selection_key': 'different-antigen-plan',
                'policy_sha256': hashlib.sha256(changed_selection_policy_bytes).hexdigest(),
            }
        )
        changed_selection_materials = PlanSelectionMaterialSpec(
            policy=changed_selection_policy,
            policy_bytes=changed_selection_policy_bytes,
            trust_policy_bytes=selection_materials.trust_policy_bytes,
            verifier_implementation_bytes=selection_materials.verifier_implementation_bytes,
            verifier=selection_materials.verifier,
        )
        with pytest.raises(PromotionIntegrityError, match='pre-capture plan|plan selection|out-of-band'):
            load_capture_promotion(
                built.root,
                expected_scope_policy=scope,
                scope_precommit_witness_materials=scope_witness_materials,
                witness_materials=witness_materials,
                source_verifiers=source_specs,
                adapter=adapter_spec,
                verified_at=_T0 + timedelta(seconds=45),
                expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                expected_promotion_sha256=built.manifest_sha256,
                selection_materials=changed_selection_materials,
                expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            )

        wrong_store_scope = scope.model_copy(update={'store_id': 'b' * 32})
        with pytest.raises(PromotionIntegrityError, match='store_id|scope policy'):
            load_capture_promotion(
                built.root,
                expected_scope_policy=wrong_store_scope,
                scope_precommit_witness_materials=scope_witness_materials,
                witness_materials=witness_materials,
                source_verifiers=source_specs,
                adapter=adapter_spec,
                verified_at=_T0 + timedelta(seconds=45),
                expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                expected_promotion_sha256=built.manifest_sha256,
                selection_materials=selection_materials,
                expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            )

        adapter_calls = 0

        def nondeterministic_adapter(_inputs, policy):
            nonlocal adapter_calls
            assert policy == adapter_policy
            adapter_calls += 1
            selected_candidate = (
                candidate
                if adapter_calls % 2
                else CandidateRecord(episode_id='future-episode', candidate_id='candidate-c')
            )
            dynamic_refs = tuple(
                sorted(
                    (
                        NormalizedRecordReference(
                            episode_id=record.episode_id,
                            record_id=record.candidate_id,
                            record_sha256=hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
                        )
                        for record in (selected_candidate, candidate_b)
                    ),
                    key=lambda item: (item.episode_id, item.record_id),
                )
            )
            return AdapterRunResult(
                candidate_records=_jsonl(selected_candidate, candidate_b),
                evidence_records=_jsonl(evidence),
                dispositions=_jsonl(disposition.model_copy(update={'candidate_record_refs': dynamic_refs})),
            )

        nondeterministic_spec = AdapterSpec(
            adapter_id=adapter_spec.adapter_id,
            adapter_version=adapter_spec.adapter_version,
            implementation_bytes=adapter_spec.implementation_bytes,
            policy_bytes=adapter_spec.policy_bytes,
            execution_environment_bytes=adapter_spec.execution_environment_bytes,
            adapter=nondeterministic_adapter,
        )
        with pytest.raises(PromotionIntegrityError, match='identical outputs'):
            load_capture_promotion(
                built.root,
                expected_scope_policy=scope,
                scope_precommit_witness_materials=scope_witness_materials,
                witness_materials=witness_materials,
                source_verifiers=source_specs,
                adapter=nondeterministic_spec,
                verified_at=_T0 + timedelta(seconds=45),
                expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                expected_promotion_sha256=built.manifest_sha256,
                selection_materials=selection_materials,
                expected_selection_manifest_sha256=plan_selection.manifest_sha256,
            )

        for relative_path in (
            'normalized/candidates.jsonl',
            loaded.index.source_verifications[0].verifier_execution_environment.path,
            next(
                artifact.file.path
                for capture in built.index.captures
                for artifact in capture.artifacts
                if artifact.role == 'body.document'
            ),
        ):
            artifact_path = built.root / relative_path
            original = artifact_path.read_bytes()
            artifact_path.chmod(0o644)
            artifact_path.write_bytes(original + b'\n')
            with pytest.raises(PromotionIntegrityError, match='binding'):
                load_capture_promotion(
                    built.root,
                    expected_scope_policy=scope,
                    scope_precommit_witness_materials=scope_witness_materials,
                    witness_materials=witness_materials,
                    source_verifiers=source_specs,
                    adapter=adapter_spec,
                    verified_at=_T0 + timedelta(seconds=45),
                    expected_scope_precommit_sha256=scope_precommit.archive_sha256,
                    expected_promotion_sha256=built.manifest_sha256,
                    selection_materials=selection_materials,
                    expected_selection_manifest_sha256=plan_selection.manifest_sha256,
                )
            artifact_path.write_bytes(original)
            artifact_path.chmod(0o444)
