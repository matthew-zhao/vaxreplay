from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.test_agentic_firecracker import _make_spec
from tests.test_agentic_run_artifact import KEY, _broker_attestation, _policy
from tests.test_agentic_scoring import _build_case
from vaxreplay.agentic.firecracker import (
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerWorkerAttestation,
    firecracker_attestation_key_id,
    firecracker_model_sha256,
    firecracker_static_config_bytes,
)
from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest
from vaxreplay.agentic.gateway_auth import InMemoryGatewaySecretStore
from vaxreplay.agentic.guest_rpc import (
    GuestRpcHostSession,
    GuestRpcMethod,
    GuestRpcPolicy,
    GuestRpcRequest,
    ModelGenerateRequest,
    SubmitRequest,
    encode_guest_rpc_frame,
    guest_rpc_policy_sha256,
    guest_rpc_session_key_id,
    guest_rpc_session_seal_hmac,
)
from vaxreplay.agentic.production_run import (
    ProductionAgenticRunError,
    finalize_production_agentic_run,
    load_production_agentic_run,
)
from vaxreplay.agentic.protocol import (
    AgenticRunFailureCode,
    AgenticRunLimits,
    AgenticTool,
    agentic_policy_sha256,
    agentic_receipt_key_id,
)
from vaxreplay.agentic.provider_adapter import ScriptedProviderAdapter, ScriptedProviderTurn
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewayPolicy,
    AuthenticatedProviderGateway,
    GatewayModelRoute,
    GatewayTerminalReason,
    SqliteGatewayLedger,
    authenticated_gateway_policy_sha256,
    build_gateway_request_frame,
    gateway_model_route_sha256,
    gateway_session_key_id,
    issue_gateway_capability,
)
from vaxreplay.agentic.run_artifact import AgenticHarnessIdentity
from vaxreplay.agentic.task_protocol import AgenticTaskInvocation
from vaxreplay.bundle import canonical_json_bytes

RUN_ID = '4' * 32
ATTEMPT = '5' * 64
WORKER_KEY = b'worker-attestation-test-key-material-01'
GATEWAY_KEY = b'gateway-attestation-test-key-material-1'
GUEST_RPC_KEY = b'guest-rpc-attestation-test-key-material'
SECRET = b'production-test-capability-key-1'
RPC_SESSION_ID = '6' * 32
SHA_A = 'a' * 64
SHA_B = 'b' * 64
SHA_C = 'c' * 64


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


def _gateway_runtime(
    tmp_path: Path,
    *,
    workspace_sha256: str,
    policy_sha256: str,
    started_at: datetime,
    content: str = 'ranked candidates',
    capability_seconds: int = 120,
):
    receipt_key_id = gateway_session_key_id(GATEWAY_KEY)
    gateway_policy = AuthenticatedGatewayPolicy(
        gateway_id='production-test-gateway',
        gateway_version='1',
        gateway_executable_sha256=SHA_A,
        gateway_config_sha256=SHA_B,
        model_registry_sha256=SHA_C,
        receipt_key_id=receipt_key_id,
    )
    route = GatewayModelRoute(
        route_id='organizer-model-route',
        logical_model_id='organizer-model',
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
        turns=(ScriptedProviderTurn(content=content, input_tokens=7, output_tokens=3, reasoning_tokens=2),),
    )
    store = InMemoryGatewaySecretStore()
    store.register(SECRET)
    grant = issue_gateway_capability(
        secret=SECRET,
        run_id=RUN_ID,
        attempt_reservation_sha256=ATTEMPT,
        execution_policy_sha256=policy_sha256,
        workspace_manifest_sha256=workspace_sha256,
        policy=gateway_policy,
        route=route,
        issued_at=started_at,
        expires_at=started_at + timedelta(seconds=capability_seconds),
        expected_peer_cid=42,
        limits=_limits(),
    )
    gateway = AuthenticatedProviderGateway(
        policy=gateway_policy,
        ledger=SqliteGatewayLedger(tmp_path / 'gateway.sqlite3'),
        secret_resolver=store,
        adapters=(adapter,),
        receipt_key=GATEWAY_KEY,
    )
    gateway.register_session(grant=grant, route=route, secret=SECRET)
    return gateway, grant


def _worker_attestation(spec, *, started_at: datetime, finished_at: datetime):
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
        jailer_argv_sha256='d' * 64,
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
    authentication = hmac.new(
        WORKER_KEY,
        b'vaxreplay.firecracker-worker-attestation.v0.2\x00' + canonical_json_bytes(attestation),
        hashlib.sha256,
    ).hexdigest()
    return AuthenticatedFirecrackerWorkerAttestation(
        attestation=attestation,
        attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
        attestation_hmac_sha256=authentication,
    )


def _materials(tmp_path: Path):
    case = _build_case(tmp_path / 'case')
    policy = _policy().model_copy(update={'limits': _limits()})
    firecracker_root = tmp_path / 'firecracker'
    firecracker_root.mkdir()
    spec = _make_spec(firecracker_root)
    started_at = datetime.now(UTC) - timedelta(seconds=3)
    gateway_runtime, gateway_grant = _gateway_runtime(
        tmp_path,
        workspace_sha256=case.workspace.manifest_sha256,
        policy_sha256=agentic_policy_sha256(policy),
        started_at=started_at,
    )
    rpc_policy = GuestRpcPolicy(
        rpc_server_id='production-test-rpc',
        rpc_server_version='1',
        rpc_server_executable_sha256='d' * 64,
    )
    rpc_session = GuestRpcHostSession(
        session_id=RPC_SESSION_ID,
        run_id=RUN_ID,
        workspace_manifest_sha256=case.workspace.manifest_sha256,
        workspace_tree_sha256=case.workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=case.workspace.manifest.model_visible_surface_sha256,
        task_invocation=AgenticTaskInvocation.from_task(
            case.workspace.task,
            workspace_manifest_sha256=case.workspace.manifest_sha256,
        ),
        expected_response_protocol=policy.response_protocol,
        worker_spec_sha256=firecracker_model_sha256(spec),
        execution_policy_sha256=agentic_policy_sha256(policy),
        broker=case.workspace.brokered_surface(),
        gateway=gateway_runtime,
        gateway_grant=gateway_grant,
        gateway_secret=SECRET,
        observed_peer_cid=spec.guest_cid,
        rpc_port=spec.guest_rpc_port,
        policy=rpc_policy,
        receipt_key=GUEST_RPC_KEY,
        expected_receipt_key_id=guest_rpc_session_key_id(GUEST_RPC_KEY),
    )
    requests = (
        GuestRpcRequest(
            session_id=RPC_SESSION_ID,
            sequence=0,
            method=GuestRpcMethod.LIST.value,
            body={'cursor': 0, 'limit': 100},
        ),
        GuestRpcRequest(
            session_id=RPC_SESSION_ID,
            sequence=1,
            method=GuestRpcMethod.MODEL_GENERATE.value,
            body=ModelGenerateRequest(
                messages=(AgenticModelMessage(role='system', content='Use only admitted evidence.'),),
                max_output_tokens=10,
            ).model_dump(mode='json'),
        ),
        GuestRpcRequest(
            session_id=RPC_SESSION_ID,
            sequence=2,
            method=GuestRpcMethod.SUBMIT.value,
            body=SubmitRequest(submission=case.oracle).model_dump(mode='json'),
        ),
    )
    for request in requests:
        rpc_session.handle_frame(
            encode_guest_rpc_frame(request, maximum_body_bytes=rpc_policy.maximum_frame_body_bytes)
        )
    guest_rpc = rpc_session.seal(sealed_at=datetime.now(UTC))
    gateway = gateway_runtime.seal_session(
        gateway_grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=datetime.now(UTC),
    )
    finished_at = datetime.now(UTC)
    worker = _worker_attestation(spec, started_at=started_at, finished_at=finished_at)
    harness = AgenticHarnessIdentity(
        harness_id='fixture-agent',
        harness_version='1',
        harness_image_or_commitment=f'sha256:{spec.images.harness.sha256}',
        harness_manifest_sha256='3' * 64,
        harness_behavior_sha256='4' * 64,
        harness_execution_mode='fixed_model_loop',
        requested_model_id=gateway.route.logical_model_id,
        adapter_id=gateway.route.adapter_id,
    )
    return case, policy, spec, worker, gateway, guest_rpc, harness


def test_production_wrapper_cross_binds_worker_gateway_and_inner_run(tmp_path: Path) -> None:
    case, policy, spec, worker, gateway, guest_rpc, harness = _materials(tmp_path)
    output = tmp_path / 'production-run'
    loaded = finalize_production_agentic_run(
        output_root=output,
        run_id=RUN_ID,
        workspace=case.workspace,
        admission=case.admission,
        expected_admission_sha256=case.admission_sha256,
        attempt_reservation_sha256=ATTEMPT,
        policy=policy,
        harness=harness,
        worker_spec=spec,
        worker_attestation=worker,
        worker_attestation_key=WORKER_KEY,
        expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
        gateway_session=gateway,
        gateway_receipt_key=GATEWAY_KEY,
        expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
        expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(gateway.policy),
        expected_gateway_route_sha256=gateway_model_route_sha256(gateway.route),
        guest_rpc_session=guest_rpc,
        guest_rpc_receipt_key=GUEST_RPC_KEY,
        expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_RPC_KEY),
        expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(guest_rpc.policy),
        workspace_broker_attestation=_broker_attestation(case.workspace),
        scratch_files={},
        receipt_key=KEY,
        expected_receipt_key_id=agentic_receipt_key_id(KEY),
    )

    assert loaded.run.receipt.accepted
    assert loaded.run.receipt.transcript_sha256 == gateway.seal.transcript_sha256
    assert loaded.authenticated_seal.seal.authenticated_worker_lifecycle
    assert loaded.authenticated_seal.seal.authenticated_provider_gateway
    assert loaded.authenticated_seal.seal.authenticated_guest_rpc_trace_present
    assert not loaded.authenticated_seal.seal.direct_guest_local_compute_trace_present
    assert not loaded.authenticated_seal.seal.direct_guest_scratch_write_trace_present
    assert not loaded.authenticated_seal.seal.complete_guest_tool_trace_claimed
    assert not loaded.authenticated_seal.seal.official_release_qualified
    assert loaded.run.tool_events == guest_rpc.projected_tool_events
    assert tuple(event.tool for event in loaded.run.tool_events) == (
        AgenticTool.LIST_WORKSPACE,
        AgenticTool.MODEL_GENERATE,
    )
    assert loaded.run.submission == guest_rpc.submission
    assert loaded.run.receipt.tool_tracing_authoritative is False

    def reload_with_harness(candidate: AgenticHarnessIdentity):
        return load_production_agentic_run(
            output,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            expected_attempt_reservation_sha256=ATTEMPT,
            policy=policy,
            harness=candidate,
            worker_spec=spec,
            worker_attestation_key=WORKER_KEY,
            expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
            gateway_receipt_key=GATEWAY_KEY,
            expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
            expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(gateway.policy),
            expected_gateway_route_sha256=gateway_model_route_sha256(gateway.route),
            guest_rpc_receipt_key=GUEST_RPC_KEY,
            expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_RPC_KEY),
            expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(guest_rpc.policy),
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
        )

    changed_behavior = harness.model_copy(update={'harness_behavior_sha256': 'f' * 64})
    with pytest.raises(ProductionAgenticRunError, match='inner run harness identity'):
        reload_with_harness(changed_behavior)

    seal_path = output / 'production-seal.hmac'
    seal_path.write_bytes(b'0' * 64 + b'\n')
    with pytest.raises(ProductionAgenticRunError, match='seal authentication'):
        reload_with_harness(harness)


def test_production_wrapper_rejects_harness_not_bound_to_worker_image(tmp_path: Path) -> None:
    case, policy, spec, worker, gateway, guest_rpc, harness = _materials(tmp_path)
    forged = harness.model_copy(update={'harness_image_or_commitment': 'sha256:' + 'f' * 64})

    with pytest.raises(ProductionAgenticRunError, match='harness identity'):
        finalize_production_agentic_run(
            output_root=tmp_path / 'production-run',
            run_id=RUN_ID,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            attempt_reservation_sha256=ATTEMPT,
            policy=policy,
            harness=forged,
            worker_spec=spec,
            worker_attestation=worker,
            worker_attestation_key=WORKER_KEY,
            expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
            gateway_session=gateway,
            gateway_receipt_key=GATEWAY_KEY,
            expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
            expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(gateway.policy),
            expected_gateway_route_sha256=gateway_model_route_sha256(gateway.route),
            guest_rpc_session=guest_rpc,
            guest_rpc_receipt_key=GUEST_RPC_KEY,
            expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_RPC_KEY),
            expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(guest_rpc.policy),
            workspace_broker_attestation=_broker_attestation(case.workspace),
            scratch_files={},
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
        )


def test_production_wrapper_rejects_authenticated_rpc_bound_to_other_worker_spec(tmp_path: Path) -> None:
    case, policy, spec, worker, gateway, guest_rpc, harness = _materials(tmp_path)
    forged_seal = guest_rpc.seal.model_copy(update={'worker_spec_sha256': 'f' * 64})
    forged_guest_rpc = guest_rpc.model_copy(
        update={
            'seal': forged_seal,
            'seal_hmac': guest_rpc_session_seal_hmac(forged_seal, GUEST_RPC_KEY),
        }
    )

    with pytest.raises(ProductionAgenticRunError, match='guest RPC session is bound to a different'):
        finalize_production_agentic_run(
            output_root=tmp_path / 'production-run',
            run_id=RUN_ID,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            attempt_reservation_sha256=ATTEMPT,
            policy=policy,
            harness=harness,
            worker_spec=spec,
            worker_attestation=worker,
            worker_attestation_key=WORKER_KEY,
            expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
            gateway_session=gateway,
            gateway_receipt_key=GATEWAY_KEY,
            expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
            expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(gateway.policy),
            expected_gateway_route_sha256=gateway_model_route_sha256(gateway.route),
            guest_rpc_session=forged_guest_rpc,
            guest_rpc_receipt_key=GUEST_RPC_KEY,
            expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_RPC_KEY),
            expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(guest_rpc.policy),
            workspace_broker_attestation=_broker_attestation(case.workspace),
            scratch_files={},
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
        )


def test_production_wrapper_rejects_independently_valid_but_different_gateway_exchange(tmp_path: Path) -> None:
    case, policy, spec, worker, _gateway, guest_rpc, harness = _materials(tmp_path)
    alternate_gateway, alternate_grant = _gateway_runtime(
        tmp_path / 'alternate-gateway',
        workspace_sha256=case.workspace.manifest_sha256,
        policy_sha256=agentic_policy_sha256(policy),
        started_at=worker.attestation.started_at,
        content='a different provider response',
    )
    assert alternate_grant == guest_rpc.gateway_grant
    request = AgenticModelRequest(
        run_id=RUN_ID,
        call_index=0,
        messages=(AgenticModelMessage(role='system', content='Use only admitted evidence.'),),
        max_output_tokens=10,
    )
    alternate_gateway.handle_frame(
        build_gateway_request_frame(alternate_grant, request, secret=SECRET),
        peer_cid=spec.guest_cid,
        observed_at=worker.attestation.started_at + timedelta(seconds=1),
    )
    alternate_session = alternate_gateway.seal_session(
        alternate_grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=datetime.now(UTC),
    )
    later_worker = _worker_attestation(
        spec,
        started_at=worker.attestation.started_at,
        finished_at=datetime.now(UTC),
    )

    with pytest.raises(ProductionAgenticRunError, match='exchange bodies differ'):
        finalize_production_agentic_run(
            output_root=tmp_path / 'production-run-mismatch',
            run_id=RUN_ID,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            attempt_reservation_sha256=ATTEMPT,
            policy=policy,
            harness=harness,
            worker_spec=spec,
            worker_attestation=later_worker,
            worker_attestation_key=WORKER_KEY,
            expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
            gateway_session=alternate_session,
            gateway_receipt_key=GATEWAY_KEY,
            expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
            expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(alternate_session.policy),
            expected_gateway_route_sha256=gateway_model_route_sha256(alternate_session.route),
            guest_rpc_session=guest_rpc,
            guest_rpc_receipt_key=GUEST_RPC_KEY,
            expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_RPC_KEY),
            expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(guest_rpc.policy),
            workspace_broker_attestation=_broker_attestation(case.workspace),
            scratch_files={},
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
        )


def test_production_wrapper_preserves_authenticated_expired_gateway_failure(tmp_path: Path) -> None:
    case = _build_case(tmp_path / 'failure-case')
    policy = _policy().model_copy(update={'limits': _limits()})
    firecracker_root = tmp_path / 'failure-firecracker'
    firecracker_root.mkdir()
    spec = _make_spec(firecracker_root)
    started_at = datetime.now(UTC) - timedelta(seconds=10)
    gateway_runtime, gateway_grant = _gateway_runtime(
        tmp_path / 'failure-gateway',
        workspace_sha256=case.workspace.manifest_sha256,
        policy_sha256=agentic_policy_sha256(policy),
        started_at=started_at,
        capability_seconds=1,
    )
    rpc_policy = GuestRpcPolicy(
        rpc_server_id='production-test-rpc',
        rpc_server_version='1',
        rpc_server_executable_sha256='d' * 64,
    )
    rpc_time = started_at + timedelta(seconds=2)
    rpc_session = GuestRpcHostSession(
        session_id=RPC_SESSION_ID,
        run_id=RUN_ID,
        workspace_manifest_sha256=case.workspace.manifest_sha256,
        workspace_tree_sha256=case.workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=case.workspace.manifest.model_visible_surface_sha256,
        task_invocation=AgenticTaskInvocation.from_task(
            case.workspace.task,
            workspace_manifest_sha256=case.workspace.manifest_sha256,
        ),
        expected_response_protocol=policy.response_protocol,
        worker_spec_sha256=firecracker_model_sha256(spec),
        execution_policy_sha256=agentic_policy_sha256(policy),
        broker=case.workspace.brokered_surface(),
        gateway=gateway_runtime,
        gateway_grant=gateway_grant,
        gateway_secret=SECRET,
        observed_peer_cid=spec.guest_cid,
        rpc_port=spec.guest_rpc_port,
        policy=rpc_policy,
        receipt_key=GUEST_RPC_KEY,
        expected_receipt_key_id=guest_rpc_session_key_id(GUEST_RPC_KEY),
        clock=lambda: rpc_time,
    )
    rpc_session.handle_frame(
        encode_guest_rpc_frame(
            GuestRpcRequest(
                session_id=RPC_SESSION_ID,
                sequence=0,
                method=GuestRpcMethod.MODEL_GENERATE.value,
                body=ModelGenerateRequest(
                    messages=(AgenticModelMessage(role='system', content='Use only admitted evidence.'),),
                    max_output_tokens=10,
                ).model_dump(mode='json'),
            ),
            maximum_body_bytes=rpc_policy.maximum_frame_body_bytes,
        )
    )
    guest_rpc = rpc_session.seal(sealed_at=started_at + timedelta(seconds=3))
    gateway = gateway_runtime.seal_session(
        gateway_grant.capability_id,
        terminal_reason=GatewayTerminalReason.FAILED,
        sealed_at=started_at + timedelta(seconds=3),
    )
    worker = _worker_attestation(
        spec,
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=4),
    )
    harness = AgenticHarnessIdentity(
        harness_id='fixture-agent',
        harness_version='1',
        harness_image_or_commitment=f'sha256:{spec.images.harness.sha256}',
        harness_manifest_sha256='3' * 64,
        harness_behavior_sha256='4' * 64,
        harness_execution_mode='fixed_model_loop',
        requested_model_id=gateway.route.logical_model_id,
        adapter_id=gateway.route.adapter_id,
    )

    loaded = finalize_production_agentic_run(
        output_root=tmp_path / 'production-failed-run',
        run_id=RUN_ID,
        workspace=case.workspace,
        admission=case.admission,
        expected_admission_sha256=case.admission_sha256,
        attempt_reservation_sha256=ATTEMPT,
        policy=policy,
        harness=harness,
        worker_spec=spec,
        worker_attestation=worker,
        worker_attestation_key=WORKER_KEY,
        expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
        gateway_session=gateway,
        gateway_receipt_key=GATEWAY_KEY,
        expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
        expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(gateway.policy),
        expected_gateway_route_sha256=gateway_model_route_sha256(gateway.route),
        guest_rpc_session=guest_rpc,
        guest_rpc_receipt_key=GUEST_RPC_KEY,
        expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_RPC_KEY),
        expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(guest_rpc.policy),
        workspace_broker_attestation=_broker_attestation(case.workspace),
        scratch_files={},
        receipt_key=KEY,
        expected_receipt_key_id=agentic_receipt_key_id(KEY),
        failure_code=AgenticRunFailureCode.GATEWAY_FAILURE,
    )

    assert not loaded.run.receipt.accepted
    assert loaded.run.receipt.failure_code == AgenticRunFailureCode.GATEWAY_FAILURE
    assert gateway.seal.attempt_count == guest_rpc.seal.model_call_count == 1
