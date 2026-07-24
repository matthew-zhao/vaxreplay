from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.test_agentic_clinical_execution_bridge import _workspace
from tests.test_agentic_firecracker import _make_spec
from tests.test_agentic_guest_rpc import _execution_submission
from vaxreplay.agentic.clinical_execution_bridge import (
    LoadedClinicalAgenticWorkspace,
    clinical_workspace_receipt_key_id,
)
from vaxreplay.agentic.clinical_production_run import (
    ClinicalProductionRunError,
    LoadedClinicalProductionRun,
    clinical_production_run_key_id,
    finalize_clinical_production_run,
    load_clinical_production_run,
)
from vaxreplay.agentic.firecracker import (
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerWorkerAttestation,
    FirecrackerWorkerSpec,
    firecracker_attestation_key_id,
    firecracker_model_sha256,
    firecracker_static_config_bytes,
)
from vaxreplay.agentic.gateway import AgenticModelMessage
from vaxreplay.agentic.gateway_auth import InMemoryGatewaySecretStore
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcHostSession,
    GuestRpcMethod,
    GuestRpcPolicy,
    GuestRpcRequest,
    ModelGenerateRequest,
    SubmitRequest,
    encode_guest_rpc_frame,
    guest_rpc_policy_sha256,
    guest_rpc_session_key_id,
)
from vaxreplay.agentic.protocol import AgenticExecutionPolicy, AgenticRunLimits, agentic_policy_sha256
from vaxreplay.agentic.provider_adapter import ScriptedProviderAdapter, ScriptedProviderTurn
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewayPolicy,
    AuthenticatedGatewaySession,
    AuthenticatedProviderGateway,
    GatewayModelRoute,
    GatewayTerminalReason,
    SqliteGatewayLedger,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    gateway_session_key_id,
    issue_gateway_capability,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.run_artifact import AgenticHarnessIdentity
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission
from vaxreplay.runner.schema import IsolationTier

RUN_ID = 'a' * 32
ATTEMPT = 'b' * 64
SESSION_ID = 'c' * 32
WORKER_KEY = b'clinical-worker-attestation-key-0001'
GATEWAY_KEY = b'clinical-gateway-receipt-key-00001'
GUEST_KEY = b'clinical-guest-rpc-receipt-key-001'
PRODUCTION_KEY = b'clinical-production-receipt-key-0001'
CAPABILITY_SECRET = b'C' * 32
WORKSPACE_KEY = b'clinical-public-workspace-test-key-01'
SHA_A = '1' * 64
SHA_B = '2' * 64
SHA_C = '3' * 64


@dataclass(frozen=True)
class Materials:
    workspace: LoadedClinicalAgenticWorkspace
    policy: AgenticExecutionPolicy
    spec: FirecrackerWorkerSpec
    worker: AuthenticatedFirecrackerWorkerAttestation
    gateway: AuthenticatedGatewaySession
    guest: AuthenticatedGuestRpcSession
    harness: AgenticHarnessIdentity
    submission: ExecutionSubmission


def _limits() -> AgenticRunLimits:
    return AgenticRunLimits(
        max_model_calls=3,
        max_input_tokens=100,
        max_output_tokens=40,
        max_reasoning_tokens=50,
        wall_seconds=30,
        cpus=1,
        memory_mib=128,
        scratch_mib=1,
        pids=32,
    )


def _worker(spec, *, started_at: datetime, finished_at: datetime) -> AuthenticatedFirecrackerWorkerAttestation:
    duration_ms = round((finished_at - started_at).total_seconds() * 1000)
    launched_monotonic_ns = 1_000_000_000
    cgroup_empty_monotonic_ns = launched_monotonic_ns + duration_ms * 1_000_000
    attestation = FirecrackerWorkerAttestation(
        run_id=RUN_ID,
        attempt_reservation_sha256=ATTEMPT,
        worker_spec_sha256=firecracker_model_sha256(spec),
        host_preflight_sha256=SHA_A,
        prepared_worker_sha256=SHA_B,
        cleanup_receipt_sha256=SHA_C,
        jailer_argv_sha256='4' * 64,
        runtime_release=spec.runtime.release,
        firecracker_sha256=spec.runtime.firecracker.sha256,
        jailer_sha256=spec.runtime.jailer.sha256,
        kernel_sha256=spec.images.kernel.sha256,
        rootfs_sha256=spec.images.rootfs.sha256,
        harness_sha256=spec.images.harness.sha256,
        initial_scratch_sha256=spec.images.scratch_template.sha256,
        config_sha256=hashlib.sha256(firecracker_static_config_bytes(spec)).hexdigest(),
        guest_cid=spec.guest_cid,
        guest_rpc_port=spec.guest_rpc_port,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        launched_monotonic_ns=launched_monotonic_ns,
        wall_deadline_monotonic_ns=launched_monotonic_ns + spec.limits.wall_seconds * 1_000_000_000,
        watchdog_triggered_at=None,
        watchdog_triggered_monotonic_ns=None,
        jailer_reaped_at=started_at,
        jailer_reaped_monotonic_ns=launched_monotonic_ns,
        cgroup_empty_at=finished_at,
        cgroup_empty_monotonic_ns=cgroup_empty_monotonic_ns,
        cleanup_finished_at=finished_at + timedelta(milliseconds=1),
        cleanup_finished_monotonic_ns=cgroup_empty_monotonic_ns + 1_000_000,
        jailer_exit_code=0,
        wall_seconds=spec.limits.wall_seconds,
        wall_timeout_triggered=False,
    )
    signature = hmac.new(
        WORKER_KEY,
        b'vaxreplay.firecracker-worker-attestation.v0.2\x00' + canonical_json_bytes(attestation),
        hashlib.sha256,
    ).hexdigest()
    return AuthenticatedFirecrackerWorkerAttestation(
        attestation=attestation,
        attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
        attestation_hmac_sha256=signature,
    )


def _materials(tmp_path: Path, *, workspace_index: int = 1) -> Materials:
    tmp_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp_path.chmod(0o700)
    _, workspace = _workspace(tmp_path / 'clinical', workspace_index)
    firecracker_root = tmp_path / 'firecracker'
    firecracker_root.mkdir()
    spec = _make_spec(firecracker_root)
    policy = AgenticExecutionPolicy(
        required_isolation=IsolationTier.DEVELOPMENT,
        response_protocol=AgenticResponseProtocol.CLINICAL_EXECUTION,
        limits=_limits(),
        required_workspace_broker_id='clinical-logical-broker',
        required_workspace_broker_version='1',
        required_workspace_broker_executable_sha256='5' * 64,
    )
    started_at = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    gateway_policy = AuthenticatedGatewayPolicy(
        gateway_id='clinical-test-gateway',
        gateway_version='1',
        gateway_executable_sha256=SHA_A,
        gateway_config_sha256=SHA_B,
        model_registry_sha256=SHA_C,
        receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
    )
    route = GatewayModelRoute(
        route_id='clinical-model-route',
        logical_model_id='clinical-model',
        provider='fixture-provider',
        provider_model_id='fixture-model-v1',
        resolved_model_id='fixture-model-snapshot',
        accepted_provider_model_ids=('fixture-model-snapshot', 'fixture-model-v1'),
        adapter_id='scripted',
        adapter_version='1',
        adapter_executable_sha256=SHA_A,
        adapter_config_sha256=SHA_B,
        endpoint_origin='https://api.fixture.invalid',
        endpoint_path='/v1/responses',
        fixed_parameters_sha256=SHA_C,
        max_context_tokens=128,
        max_output_tokens=32,
        input_preflight='conservative_upper_bound',
        reasoning_accounting='reported',
        provider_data_control='default',
    )
    adapter = ScriptedProviderAdapter(
        provider=route.provider,
        adapter_id=route.adapter_id,
        adapter_version=route.adapter_version,
        executable_sha256=route.adapter_executable_sha256,
        config_sha256=route.adapter_config_sha256,
        turns=(ScriptedProviderTurn(content='clinical forecast complete', input_tokens=7, output_tokens=3),),
        clock=lambda: started_at + timedelta(seconds=1),
    )
    store = InMemoryGatewaySecretStore()
    store.register(CAPABILITY_SECRET)
    grant = issue_gateway_capability(
        secret=CAPABILITY_SECRET,
        run_id=RUN_ID,
        attempt_reservation_sha256=ATTEMPT,
        execution_policy_sha256=agentic_policy_sha256(policy),
        workspace_manifest_sha256=workspace.manifest_sha256,
        policy=gateway_policy,
        route=route,
        issued_at=started_at,
        expires_at=started_at + timedelta(seconds=20),
        expected_peer_cid=spec.guest_cid,
        limits=_limits(),
    )
    gateway_runtime = AuthenticatedProviderGateway(
        policy=gateway_policy,
        ledger=SqliteGatewayLedger(tmp_path / 'gateway.sqlite3'),
        secret_resolver=store,
        adapters=(adapter,),
        receipt_key=GATEWAY_KEY,
    )
    gateway_runtime.register_session(grant=grant, route=route, secret=CAPABILITY_SECRET)
    rpc_policy = GuestRpcPolicy(
        rpc_server_id='clinical-test-rpc',
        rpc_server_version='1',
        rpc_server_executable_sha256='6' * 64,
    )
    rpc = GuestRpcHostSession(
        session_id=SESSION_ID,
        run_id=RUN_ID,
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        task_invocation=workspace.invocation,
        expected_response_protocol=AgenticResponseProtocol.CLINICAL_EXECUTION,
        worker_spec_sha256=firecracker_model_sha256(spec),
        execution_policy_sha256=agentic_policy_sha256(policy),
        broker=workspace.brokered_surface(),
        gateway=gateway_runtime,
        gateway_grant=grant,
        gateway_secret=CAPABILITY_SECRET,
        observed_peer_cid=spec.guest_cid,
        rpc_port=spec.guest_rpc_port,
        policy=rpc_policy,
        receipt_key=GUEST_KEY,
        expected_receipt_key_id=guest_rpc_session_key_id(GUEST_KEY),
        clock=lambda: started_at + timedelta(seconds=2),
    )
    model_request = GuestRpcRequest(
        session_id=SESSION_ID,
        sequence=0,
        method=GuestRpcMethod.MODEL_GENERATE.value,
        body=ModelGenerateRequest(
            messages=(AgenticModelMessage(role='system', content='Use only the clinical workspace.'),),
            max_output_tokens=10,
        ).model_dump(mode='json'),
    )
    rpc.handle_frame(encode_guest_rpc_frame(model_request, maximum_body_bytes=rpc_policy.maximum_frame_body_bytes))
    submission = _execution_submission(workspace.task)
    submit_request = GuestRpcRequest(
        session_id=SESSION_ID,
        sequence=1,
        method=GuestRpcMethod.SUBMIT.value,
        body=SubmitRequest(submission=submission).model_dump(mode='json'),
    )
    rpc.handle_frame(encode_guest_rpc_frame(submit_request, maximum_body_bytes=rpc_policy.maximum_frame_body_bytes))
    guest = rpc.seal(sealed_at=started_at + timedelta(seconds=3))
    gateway = gateway_runtime.seal_session(
        grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=started_at + timedelta(seconds=4),
    )
    worker = _worker(spec, started_at=started_at, finished_at=started_at + timedelta(seconds=5))
    harness = AgenticHarnessIdentity(
        harness_id='clinical-fixture-agent',
        harness_version='1',
        harness_image_or_commitment=f'sha256:{spec.images.harness.sha256}',
        harness_manifest_sha256='3' * 64,
        harness_behavior_sha256='4' * 64,
        harness_execution_mode='fixed_model_loop',
        requested_model_id=route.logical_model_id,
        adapter_id=route.adapter_id,
    )
    return Materials(workspace, policy, spec, worker, gateway, guest, harness, submission)


def _finalize(tmp_path: Path, materials: Materials) -> LoadedClinicalProductionRun:
    workspace = materials.workspace
    return finalize_clinical_production_run(
        output_root=tmp_path / 'clinical-production-run',
        run_id=RUN_ID,
        workspace=workspace,
        expected_authenticated_workspace_receipt_sha256=workspace.authenticated_receipt_sha256,
        workspace_receipt_key=WORKSPACE_KEY,
        expected_workspace_receipt_key_id=clinical_workspace_receipt_key_id(WORKSPACE_KEY),
        attempt_reservation_sha256=ATTEMPT,
        policy=materials.policy,
        harness=materials.harness,
        worker_spec=materials.spec,
        worker_attestation=materials.worker,
        worker_attestation_key=WORKER_KEY,
        expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
        gateway_session=materials.gateway,
        gateway_receipt_key=GATEWAY_KEY,
        expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
        expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(materials.gateway.policy),
        expected_gateway_route_sha256=gateway_model_route_sha256(materials.gateway.route),
        guest_rpc_session=materials.guest,
        guest_rpc_receipt_key=GUEST_KEY,
        expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_KEY),
        expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(materials.guest.policy),
        submission=materials.submission,
        receipt_key=PRODUCTION_KEY,
        expected_receipt_key_id=clinical_production_run_key_id(PRODUCTION_KEY),
        sealed_at=datetime(2025, 1, 2, 3, 4, 11, tzinfo=UTC),
    )


def _load(root: Path, materials: Materials, **updates) -> LoadedClinicalProductionRun:
    workspace = updates.pop('workspace', materials.workspace)
    values: dict[str, Any] = {
        'workspace': workspace,
        'expected_authenticated_workspace_receipt_sha256': workspace.authenticated_receipt_sha256,
        'workspace_receipt_key': WORKSPACE_KEY,
        'expected_workspace_receipt_key_id': clinical_workspace_receipt_key_id(WORKSPACE_KEY),
        'expected_run_id': RUN_ID,
        'expected_attempt_reservation_sha256': ATTEMPT,
        'policy': materials.policy,
        'harness': materials.harness,
        'worker_spec': materials.spec,
        'worker_attestation_key': WORKER_KEY,
        'expected_worker_attestation_key_id': firecracker_attestation_key_id(WORKER_KEY),
        'gateway_receipt_key': GATEWAY_KEY,
        'expected_gateway_receipt_key_id': gateway_session_key_id(GATEWAY_KEY),
        'expected_gateway_policy_sha256': authenticated_gateway_policy_sha256(materials.gateway.policy),
        'expected_gateway_route_sha256': gateway_model_route_sha256(materials.gateway.route),
        'guest_rpc_receipt_key': GUEST_KEY,
        'expected_guest_rpc_receipt_key_id': guest_rpc_session_key_id(GUEST_KEY),
        'expected_guest_rpc_policy_sha256': guest_rpc_policy_sha256(materials.guest.policy),
        'receipt_key': PRODUCTION_KEY,
        'expected_receipt_key_id': clinical_production_run_key_id(PRODUCTION_KEY),
    }
    values.update(updates)
    return load_clinical_production_run(root, **values)


def test_clinical_finalizer_cross_binds_complete_outer_evidence(tmp_path: Path) -> None:
    materials = _materials(tmp_path)
    loaded = _finalize(tmp_path, materials)

    assert loaded.submission == materials.submission
    assert loaded.workspace.task == materials.workspace.task
    assert loaded.receipt.usage.model_calls == 1
    assert loaded.receipt.resolved_model_id == materials.gateway.route.resolved_model_id
    assert loaded.receipt.harness_image_bound_to_worker
    assert loaded.receipt.launch_hash_bound_across_worker_gateway_and_guest
    assert not loaded.receipt.complete_guest_tool_trace_claimed
    assert not loaded.receipt.independently_attested_immutable_model_weights
    assert not loaded.receipt.linux_kvm_runtime_qualified
    assert not loaded.receipt.official_execution_qualified
    assert {path.name for path in loaded.root.iterdir()} == {
        'clinical-run.hmac',
        'clinical-run.json',
        'gateway-session.json',
        'guest-rpc-session.json',
        'submission.json',
        'worker-attestation.json',
        'workspace-receipt.json',
    }
    assert _load(loaded.root, materials).authenticated_receipt_sha256 == loaded.authenticated_receipt_sha256


@pytest.mark.parametrize(
    ('field_name', 'value', 'match'),
    [
        ('expected_run_id', '0' * 32, 'unexpected run ID'),
        ('expected_attempt_reservation_sha256', '0' * 64, 'authentication failed'),
        ('expected_gateway_route_sha256', '0' * 64, 'model route differs'),
        ('expected_guest_rpc_policy_sha256', '0' * 64, 'guest-RPC policy differs'),
    ],
)
def test_loader_rejects_wrong_launch_model_or_rpc_pin(
    tmp_path: Path,
    field_name: str,
    value: str,
    match: str,
) -> None:
    materials = _materials(tmp_path)
    loaded = _finalize(tmp_path, materials)

    with pytest.raises(ClinicalProductionRunError, match=match):
        _load(loaded.root, materials, **{field_name: value})


def test_finalizer_rejects_wrong_harness_and_submission(tmp_path: Path) -> None:
    materials = _materials(tmp_path)
    wrong_harness = materials.harness.model_copy(update={'harness_version': 'other'})
    loaded = _finalize(tmp_path, materials)

    with pytest.raises(ClinicalProductionRunError, match='mismatched harness'):
        _load(loaded.root, materials, harness=wrong_harness)
    with pytest.raises(ClinicalProductionRunError, match='retained terminal submission'):
        finalize_clinical_production_run(
            output_root=tmp_path / 'wrong-submission',
            run_id=RUN_ID,
            workspace=materials.workspace,
            expected_authenticated_workspace_receipt_sha256=(materials.workspace.authenticated_receipt_sha256),
            workspace_receipt_key=WORKSPACE_KEY,
            expected_workspace_receipt_key_id=clinical_workspace_receipt_key_id(WORKSPACE_KEY),
            attempt_reservation_sha256=ATTEMPT,
            policy=materials.policy,
            harness=materials.harness,
            worker_spec=materials.spec,
            worker_attestation=materials.worker,
            worker_attestation_key=WORKER_KEY,
            expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
            gateway_session=materials.gateway,
            gateway_receipt_key=GATEWAY_KEY,
            expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
            expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(materials.gateway.policy),
            expected_gateway_route_sha256=gateway_model_route_sha256(materials.gateway.route),
            guest_rpc_session=materials.guest,
            guest_rpc_receipt_key=GUEST_KEY,
            expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_KEY),
            expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(materials.guest.policy),
            submission=materials.submission.model_copy(update={'target_trial_id': 'trial-other'}),
            receipt_key=PRODUCTION_KEY,
            expected_receipt_key_id=clinical_production_run_key_id(PRODUCTION_KEY),
        )


def test_loader_rejects_wrong_workspace_tampering_and_extra_inventory(tmp_path: Path) -> None:
    materials = _materials(tmp_path)
    loaded = _finalize(tmp_path, materials)
    other = _materials(tmp_path / 'other', workspace_index=2)

    with pytest.raises(ClinicalProductionRunError, match='authentication failed|workspace'):
        _load(loaded.root, materials, workspace=other.workspace)

    receipt_path = loaded.root / 'clinical-run.hmac'
    original = receipt_path.read_bytes()
    receipt_path.write_bytes(b'0' * 64 + b'\n')
    receipt_path.chmod(0o600)
    with pytest.raises(ClinicalProductionRunError, match='authentication failed'):
        _load(loaded.root, materials)
    receipt_path.write_bytes(original)
    receipt_path.chmod(0o600)

    extra = loaded.root / 'unlisted.json'
    extra.write_bytes(b'{}')
    extra.chmod(0o600)
    with pytest.raises(ClinicalProductionRunError, match='inventory'):
        _load(loaded.root, materials)
