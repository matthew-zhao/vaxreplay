from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.test_agentic_scoring import _build_case
from vaxreplay.agentic.gateway import MeteredFakeGateway, ScriptedGatewayTurn
from vaxreplay.agentic.protocol import (
    AgenticExecutionPolicy,
    AgenticRunFailureCode,
    AgenticRunLimits,
    AgenticTool,
    agentic_receipt_key_id,
)
from vaxreplay.agentic.run_artifact import (
    AgenticHarnessIdentity,
    AgenticRunArtifactError,
    AgenticToolEvent,
    AgenticWorkspaceBrokerAttestation,
    finalize_agentic_run,
    load_agentic_run_artifact,
)
from vaxreplay.agentic.schema import agentic_model_sha256
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.runner.schema import BackendCapabilities, IsolationTier

RUN_ID = '1' * 32
KEY = bytes.fromhex('ab' * 32)
ATTEMPT = '9' * 64
STARTED = datetime(2026, 7, 13, 12, tzinfo=UTC)
BROKER_ID = 'fixture-exact-byte-broker'
BROKER_VERSION = '1'
BROKER_EXECUTABLE_SHA256 = 'b' * 64


def _capabilities(tier: IsolationTier = IsolationTier.DEVELOPMENT, *, complete: bool = False) -> BackendCapabilities:
    return BackendCapabilities(
        backend_id='fixture-backend',
        backend_version='1',
        isolation_tier=tier,
        network_isolation=complete,
        host_filesystem_isolation=complete,
        read_only_root=complete,
        non_root_user=complete,
        capability_drop=complete,
        no_new_privileges=complete,
        process_limit=complete,
        memory_limit=complete,
        cpu_limit=complete,
        scratch_limit=complete,
        fresh_worker_per_episode=complete,
    )


def _harness() -> AgenticHarnessIdentity:
    return AgenticHarnessIdentity(
        harness_id='fixture-agent',
        harness_version='1',
        harness_image_or_commitment='sha256:' + '7' * 64,
        harness_manifest_sha256='5' * 64,
        harness_behavior_sha256='6' * 64,
        harness_execution_mode='fixed_model_loop',
        requested_model_id='organizer-model',
        adapter_id='fixture-gateway',
    )


def _policy(tier: IsolationTier = IsolationTier.DEVELOPMENT) -> AgenticExecutionPolicy:
    return AgenticExecutionPolicy(
        required_isolation=tier,
        limits=AgenticRunLimits(),
        required_workspace_broker_id=BROKER_ID,
        required_workspace_broker_version=BROKER_VERSION,
        required_workspace_broker_executable_sha256=BROKER_EXECUTABLE_SHA256,
    )


def _broker_attestation(workspace) -> AgenticWorkspaceBrokerAttestation:
    surface = workspace.manifest.model_visible_surface_sha256
    return AgenticWorkspaceBrokerAttestation(
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=surface,
        broker_id=BROKER_ID,
        broker_version=BROKER_VERSION,
        broker_executable_sha256=BROKER_EXECUTABLE_SHA256,
        surface_sha256_before_run=surface,
        surface_sha256_after_run=surface,
    )


def _transcript():
    gateway = MeteredFakeGateway(
        run_id=RUN_ID,
        resolved_model_id='organizer-model-snapshot',
        limits=AgenticRunLimits(),
        scripted_turns=(ScriptedGatewayTurn(content='analysis complete', input_tokens=10, output_tokens=3),),
    )
    from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest

    gateway.generate(
        AgenticModelRequest(
            run_id=RUN_ID,
            call_index=0,
            messages=(AgenticModelMessage(role='system', content='Use only the admitted workspace.'),),
            max_output_tokens=10,
        )
    )
    return gateway.transcript


def _event() -> AgenticToolEvent:
    exchange = _transcript().exchanges[0]
    request_bytes = canonical_json_bytes(exchange.request)
    response_bytes = canonical_json_bytes(exchange.response)
    return AgenticToolEvent(
        event_index=0,
        tool=AgenticTool.MODEL_GENERATE,
        gateway_call_index=0,
        started_at=STARTED,
        finished_at=STARTED + timedelta(milliseconds=100),
        request_sha256=exchange.receipt.request_sha256,
        request_bytes=len(request_bytes),
        response_sha256=exchange.receipt.response_sha256,
        response_bytes=len(response_bytes),
        succeeded=True,
    )


def _finalize(tmp_path: Path):
    case = _build_case(tmp_path)
    policy = _policy()
    artifact = finalize_agentic_run(
        output_root=tmp_path / 'run',
        run_id=RUN_ID,
        workspace=case.workspace,
        admission=case.admission,
        expected_admission_sha256=case.admission_sha256,
        attempt_reservation_sha256=ATTEMPT,
        policy=policy,
        harness=_harness(),
        capabilities=_capabilities(),
        workspace_broker_attestation=_broker_attestation(case.workspace),
        gateway_transcript=_transcript(),
        tool_events=(_event(),),
        scratch_files={'notes/summary.txt': b'private scratch summary'},
        final_submission_bytes=canonical_json_bytes(case.oracle),
        started_at=STARTED,
        finished_at=STARTED + timedelta(seconds=1),
        receipt_key=KEY,
        expected_receipt_key_id=agentic_receipt_key_id(KEY),
        gateway_channel_isolation=False,
        tool_tracing_authoritative=False,
    )
    return case, policy, artifact


def test_finalizer_binds_admission_gateway_tools_scratch_and_submission(tmp_path: Path) -> None:
    case, _, artifact = _finalize(tmp_path)

    assert artifact.receipt.accepted
    assert artifact.submission == case.oracle
    assert artifact.receipt.workspace_admission_sha256 == case.admission_sha256
    assert artifact.receipt.build_policy_sha256 == case.workspace.manifest.build_policy_sha256
    assert artifact.receipt.usage.model_calls == 1
    assert artifact.receipt.resolved_model_id == 'organizer-model-snapshot'
    assert artifact.receipt.harness_manifest_sha256 == _harness().harness_manifest_sha256
    assert artifact.receipt.harness_behavior_sha256 == _harness().harness_behavior_sha256
    assert artifact.receipt.harness_execution_mode == _harness().harness_execution_mode
    assert artifact.workspace_broker_attestation.worker_workspace_filesystem_mounted is False
    assert artifact.scratch_manifest[0].path == 'notes/summary.txt'


def test_loader_rejects_tampered_submission_and_wrong_attempt_reservation(tmp_path: Path) -> None:
    case, policy, artifact = _finalize(tmp_path)
    submission_path = artifact.root / 'submission.json'
    submission_path.write_bytes(submission_path.read_bytes() + b' ')
    with pytest.raises(AgenticRunArtifactError, match='artifact hashes'):
        load_agentic_run_artifact(
            artifact.root,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            expected_attempt_reservation_sha256=ATTEMPT,
            policy=policy,
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
        )

    submission_path.write_bytes(canonical_json_bytes(case.oracle))
    with pytest.raises(AgenticRunArtifactError, match='different workspace, admission, or policy'):
        load_agentic_run_artifact(
            artifact.root,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            expected_attempt_reservation_sha256='8' * 64,
            policy=policy,
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
        )


def test_official_finalizer_requires_complete_backend_and_gateway_controls(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    policy = _policy(IsolationTier.OFFICIAL)
    with pytest.raises(AgenticRunArtifactError, match='official release admission'):
        finalize_agentic_run(
            output_root=tmp_path / 'run',
            run_id=RUN_ID,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=agentic_model_sha256(case.admission),
            attempt_reservation_sha256=ATTEMPT,
            policy=policy,
            harness=_harness(),
            capabilities=_capabilities(IsolationTier.OFFICIAL, complete=False),
            workspace_broker_attestation=_broker_attestation(case.workspace),
            gateway_transcript=_transcript(),
            tool_events=(_event(),),
            scratch_files={},
            final_submission_bytes=canonical_json_bytes(case.oracle),
            started_at=STARTED,
            finished_at=STARTED + timedelta(seconds=1),
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
            gateway_channel_isolation=False,
            tool_tracing_authoritative=False,
        )


def test_rejected_run_requires_empty_submission_and_failure_code(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    policy = _policy()
    artifact = finalize_agentic_run(
        output_root=tmp_path / 'run',
        run_id=RUN_ID,
        workspace=case.workspace,
        admission=case.admission,
        expected_admission_sha256=case.admission_sha256,
        attempt_reservation_sha256=ATTEMPT,
        policy=policy,
        harness=_harness(),
        capabilities=_capabilities(),
        workspace_broker_attestation=_broker_attestation(case.workspace),
        gateway_transcript=_transcript(),
        tool_events=(_event(),),
        scratch_files={},
        final_submission_bytes=b'',
        started_at=STARTED,
        finished_at=STARTED + timedelta(seconds=1),
        receipt_key=KEY,
        expected_receipt_key_id=agentic_receipt_key_id(KEY),
        failure_code=AgenticRunFailureCode.HARNESS_FAILURE,
        gateway_channel_isolation=False,
        tool_tracing_authoritative=False,
    )
    assert artifact.submission is None
    assert artifact.receipt.failure_code == AgenticRunFailureCode.HARNESS_FAILURE


def test_model_tool_event_must_bind_exact_gateway_exchange(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    with pytest.raises(AgenticRunArtifactError, match='exact gateway exchange'):
        finalize_agentic_run(
            output_root=tmp_path / 'run',
            run_id=RUN_ID,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            attempt_reservation_sha256=ATTEMPT,
            policy=_policy(),
            harness=_harness(),
            capabilities=_capabilities(),
            workspace_broker_attestation=_broker_attestation(case.workspace),
            gateway_transcript=_transcript(),
            tool_events=(_event().model_copy(update={'request_sha256': '0' * 64}),),
            scratch_files={},
            final_submission_bytes=canonical_json_bytes(case.oracle),
            started_at=STARTED,
            finished_at=STARTED + timedelta(seconds=1),
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
            gateway_channel_isolation=False,
            tool_tracing_authoritative=False,
        )


def test_finalizer_rejects_run_interval_beyond_wall_clock_limit(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    policy = _policy().model_copy(update={'limits': AgenticRunLimits(wall_seconds=1)})

    with pytest.raises(AgenticRunArtifactError, match='wall-clock limit'):
        finalize_agentic_run(
            output_root=tmp_path / 'run',
            run_id=RUN_ID,
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            attempt_reservation_sha256=ATTEMPT,
            policy=policy,
            harness=_harness(),
            capabilities=_capabilities(),
            workspace_broker_attestation=_broker_attestation(case.workspace),
            gateway_transcript=_transcript(),
            tool_events=(_event(),),
            scratch_files={},
            final_submission_bytes=canonical_json_bytes(case.oracle),
            started_at=STARTED,
            finished_at=STARTED + timedelta(hours=1),
            receipt_key=KEY,
            expected_receipt_key_id=agentic_receipt_key_id(KEY),
            gateway_channel_isolation=False,
            tool_tracing_authoritative=False,
        )
