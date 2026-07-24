from __future__ import annotations

import hashlib
import socket
import struct
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.test_clinicaltrials_execution_scoring import _case as _execution_case
from tests.test_clinicaltrials_execution_scoring import _submission as _execution_submission
from vaxreplay.agentic.gateway import AgenticModelMessage
from vaxreplay.agentic.gateway_auth import InMemoryGatewaySecretStore
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcClient,
    GuestRpcError,
    GuestRpcErrorCode,
    GuestRpcHostServer,
    GuestRpcHostSession,
    GuestRpcMethod,
    GuestRpcPolicy,
    GuestRpcRequest,
    GuestRpcResponse,
    GuestRpcTerminalStatus,
    ListWorkspaceRequest,
    ListWorkspaceResult,
    ModelGenerateRequest,
    ModelGenerateResult,
    SearchWorkspaceResult,
    SubmitRequest,
    decode_guest_rpc_frame,
    encode_guest_rpc_frame,
    guest_rpc_session_key_id,
    verify_authenticated_guest_rpc_session,
)
from vaxreplay.agentic.protocol import AgenticRunLimits, AgenticTool
from vaxreplay.agentic.provider_adapter import ScriptedProviderAdapter, ScriptedProviderTurn
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewayPolicy,
    AuthenticatedProviderGateway,
    GatewayCapabilityGrant,
    GatewayModelRoute,
    SqliteGatewayLedger,
    gateway_session_key_id,
    issue_gateway_capability,
)
from vaxreplay.agentic.schema import (
    AgenticFactQuery,
    AgenticMediaType,
    AgenticTaskEnvelope,
    AgenticValueType,
    AgenticWorkspaceEntry,
)
from vaxreplay.agentic.scoring import (
    AgenticDecision,
    AgenticSubmissionV1,
    CandidateProbability,
    DecisionStatus,
    FactAnswer,
    FactAnswerStatus,
)
from vaxreplay.agentic.task_protocol import AgenticTaskInvocation, agentic_task_invocation_sha256
from vaxreplay.agentic.workspace import AgenticLogicalWorkspaceBroker, model_visible_surface_bytes
from vaxreplay.bundle import canonical_json_bytes

_RUN_ID = '1' * 32
_SESSION_ID = '2' * 32
_WORKSPACE_SHA = '3' * 64
_WORKSPACE_TREE_SHA = '4' * 64
_SURFACE_SHA = '5' * 64
_EXECUTION_POLICY_SHA = '6' * 64
_WORKER_SPEC_SHA = '7' * 64
_PEER_CID = 37
_RPC_PORT = 52
_GATEWAY_SECRET = b'g' * 32
_GATEWAY_RECEIPT_KEY = b'G' * 32
_RPC_RECEIPT_KEY = b'R' * 32
_ISSUED_AT = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)


@dataclass(frozen=True)
class _Fixture:
    session: GuestRpcHostSession
    adapter: ScriptedProviderAdapter
    grant: GatewayCapabilityGrant


def _broker() -> AgenticLogicalWorkspaceBroker:
    files = {
        'TASK.md': b'Rank the candidates using only frozen evidence.\n',
        'sources/evidence.txt': b'candidate A dose 10\ncandidate B dose 20\n',
    }
    entries = tuple(
        AgenticWorkspaceEntry(
            path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
            media_type=AgenticMediaType.TEXT,
            provenance_node_id=f'fixture:{path}',
        )
        for path, content in sorted(files.items())
    )
    return AgenticLogicalWorkspaceBroker(entries=entries, surface=model_visible_surface_bytes(files))


def _submission() -> AgenticSubmissionV1:
    return AgenticSubmissionV1(
        task_id='fixture-task',
        workspace_manifest_sha256=_WORKSPACE_SHA,
        fact_answers=(FactAnswer(query_id='dose', status=FactAnswerStatus.NOT_FOUND),),
        decision=AgenticDecision(
            status=DecisionStatus.INSUFFICIENT_EVIDENCE,
            ranking=('candidate-a', 'candidate-b'),
            advancement_probabilities=(
                CandidateProbability(candidate_id='candidate-a', probability=0.5),
                CandidateProbability(candidate_id='candidate-b', probability=0.5),
            ),
        ),
    )


def _task_invocation() -> AgenticTaskInvocation:
    task = AgenticTaskEnvelope(
        task_id='fixture-task',
        episode_id='fixture-episode',
        episode_manifest_sha256='a' * 64,
        decision_at=_ISSUED_AT,
        task_type='candidate_ranking',
        candidate_ids=('candidate-a', 'candidate-b'),
        portfolio_size=1,
        instructions='Rank the candidates using only the frozen evidence.',
        fact_queries=(
            AgenticFactQuery(
                query_id='dose',
                description='Extract the candidate dose.',
                value_type=AgenticValueType.NUMBER,
            ),
        ),
        historically_preregistered=False,
    )
    return AgenticTaskInvocation.from_task(task, workspace_manifest_sha256=_WORKSPACE_SHA)


def _fixture(
    tmp_path: Path,
    *,
    maximum_frame_body_bytes: int = 1024 * 1024,
    maximum_session_wire_bytes: int | None = None,
    maximum_requests: int = 100,
    provider_content: str = 'Candidate B has stronger evidence.',
    task_invocation: AgenticTaskInvocation | None = None,
    workspace_tree_sha256: str = _WORKSPACE_TREE_SHA,
    model_visible_surface_sha256: str = _SURFACE_SHA,
    broker: AgenticLogicalWorkspaceBroker | None = None,
) -> _Fixture:
    selected_invocation = task_invocation or _task_invocation()
    gateway_receipt_key = _GATEWAY_RECEIPT_KEY
    gateway_policy = AuthenticatedGatewayPolicy(
        gateway_id='fixture-gateway',
        gateway_version='1',
        gateway_executable_sha256='a' * 64,
        gateway_config_sha256='b' * 64,
        model_registry_sha256='c' * 64,
        receipt_key_id=gateway_session_key_id(gateway_receipt_key),
        maximum_frame_body_bytes=1024 * 1024,
        maximum_session_wire_bytes=8 * 1024 * 1024,
        maximum_provider_call_seconds=30,
    )
    route = GatewayModelRoute(
        route_id='fixture-route',
        logical_model_id='fixture-model',
        provider='fixture-provider',
        provider_model_id='fixture-provider-model',
        resolved_model_id='fixture-model-2025-01-02',
        accepted_provider_model_ids=('fixture-model-2025-01-02', 'fixture-provider-model'),
        adapter_id='scripted',
        adapter_version='1',
        adapter_executable_sha256='d' * 64,
        adapter_config_sha256='e' * 64,
        endpoint_origin='https://provider.invalid',
        endpoint_path='/v1/responses',
        fixed_parameters_sha256='f' * 64,
        max_context_tokens=256,
        max_output_tokens=64,
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
        turns=(
            ScriptedProviderTurn(
                content=provider_content,
                input_tokens=7,
                output_tokens=5,
                reasoning_tokens=2,
            ),
        ),
        clock=lambda: _ISSUED_AT + timedelta(seconds=1),
    )
    store = InMemoryGatewaySecretStore()
    store.register(_GATEWAY_SECRET)
    grant = issue_gateway_capability(
        secret=_GATEWAY_SECRET,
        run_id=_RUN_ID,
        attempt_reservation_sha256='8' * 64,
        execution_policy_sha256=_EXECUTION_POLICY_SHA,
        workspace_manifest_sha256=selected_invocation.workspace_manifest_sha256,
        policy=gateway_policy,
        route=route,
        issued_at=_ISSUED_AT,
        expires_at=_ISSUED_AT + timedelta(minutes=10),
        expected_peer_cid=_PEER_CID,
        limits=AgenticRunLimits(
            max_model_calls=3,
            max_input_tokens=100,
            max_output_tokens=40,
            max_reasoning_tokens=50,
        ),
    )
    gateway = AuthenticatedProviderGateway(
        policy=gateway_policy,
        ledger=SqliteGatewayLedger(tmp_path / 'gateway.sqlite3'),
        secret_resolver=store,
        adapters=(adapter,),
        receipt_key=gateway_receipt_key,
    )
    gateway.register_session(grant=grant, route=route, secret=_GATEWAY_SECRET)
    policy = GuestRpcPolicy(
        rpc_server_id='fixture-rpc',
        rpc_server_version='1',
        rpc_server_executable_sha256='9' * 64,
        maximum_frame_body_bytes=maximum_frame_body_bytes,
        maximum_session_wire_bytes=(
            maximum_session_wire_bytes if maximum_session_wire_bytes is not None else 8 * maximum_frame_body_bytes
        ),
        maximum_requests=maximum_requests,
        maximum_list_entries=10,
        maximum_read_bytes=min(4096, maximum_frame_body_bytes // 2),
        maximum_search_results=10,
        maximum_submission_bytes=min(4096, maximum_frame_body_bytes // 2),
    )
    session = GuestRpcHostSession(
        session_id=_SESSION_ID,
        run_id=_RUN_ID,
        workspace_manifest_sha256=selected_invocation.workspace_manifest_sha256,
        workspace_tree_sha256=workspace_tree_sha256,
        model_visible_surface_sha256=model_visible_surface_sha256,
        task_invocation=selected_invocation,
        expected_response_protocol=selected_invocation.response_protocol,
        worker_spec_sha256=_WORKER_SPEC_SHA,
        execution_policy_sha256=_EXECUTION_POLICY_SHA,
        broker=broker or _broker(),
        gateway=gateway,
        gateway_grant=grant,
        gateway_secret=_GATEWAY_SECRET,
        observed_peer_cid=_PEER_CID,
        rpc_port=_RPC_PORT,
        policy=policy,
        receipt_key=_RPC_RECEIPT_KEY,
        expected_receipt_key_id=guest_rpc_session_key_id(_RPC_RECEIPT_KEY),
        clock=lambda: _ISSUED_AT + timedelta(seconds=1),
    )
    return _Fixture(session=session, adapter=adapter, grant=grant)


def _request(sequence: int, method: str, body: dict) -> GuestRpcRequest:
    return GuestRpcRequest(
        session_id=_SESSION_ID,
        sequence=sequence,
        method=method,
        body=body,
    )


def _send(session: GuestRpcHostSession, request: GuestRpcRequest) -> tuple[bytes, GuestRpcResponse]:
    frame = encode_guest_rpc_frame(request, maximum_body_bytes=session.policy.maximum_frame_body_bytes)
    response_frame = session.handle_frame(frame)
    response, _ = decode_guest_rpc_frame(
        response_frame,
        GuestRpcResponse,
        maximum_body_bytes=session.policy.maximum_frame_body_bytes,
    )
    return response_frame, response


def _verify(
    artifact: AuthenticatedGuestRpcSession,
    *,
    task_invocation: AgenticTaskInvocation | None = None,
) -> None:
    expected_invocation = task_invocation or _task_invocation()
    verify_authenticated_guest_rpc_session(
        artifact,
        receipt_key=_RPC_RECEIPT_KEY,
        expected_receipt_key_id=guest_rpc_session_key_id(_RPC_RECEIPT_KEY),
        expected_run_id=_RUN_ID,
        expected_workspace_manifest_sha256=expected_invocation.workspace_manifest_sha256,
        expected_execution_policy_sha256=_EXECUTION_POLICY_SHA,
        expected_task_invocation_sha256=agentic_task_invocation_sha256(expected_invocation),
        expected_response_protocol=expected_invocation.response_protocol,
        expected_peer_cid=_PEER_CID,
        expected_rpc_port=_RPC_PORT,
    )


def _fill_with_list_requests(session: GuestRpcHostSession, *, count: int) -> int:
    wire_bytes = 0
    for sequence in range(count):
        request = _request(sequence, GuestRpcMethod.LIST.value, {'cursor': 0, 'limit': 1})
        request_frame = encode_guest_rpc_frame(
            request,
            maximum_body_bytes=session.policy.maximum_frame_body_bytes,
        )
        response_frame, response = _send(session, request)
        assert response.succeeded
        wire_bytes += len(request_frame) + len(response_frame)
    return wire_bytes


def test_guest_rpc_full_flow_exact_replay_and_authenticated_terminal_seal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    session = fixture.session

    _, listed = _send(
        session,
        _request(0, GuestRpcMethod.LIST.value, ListWorkspaceRequest(cursor=0, limit=10).model_dump(mode='json')),
    )
    assert listed.succeeded
    assert listed.result is not None
    listed_result = ListWorkspaceResult.model_validate_json(canonical_json_bytes(listed.result))
    assert [item.path for item in listed_result.files] == ['TASK.md', 'sources/evidence.txt']

    _, read = _send(
        session,
        _request(
            1,
            GuestRpcMethod.READ.value,
            {'path': 'sources/evidence.txt', 'offset': 0, 'limit': 20},
        ),
    )
    assert read.succeeded

    _, searched = _send(
        session,
        _request(
            2,
            GuestRpcMethod.SEARCH.value,
            {'needle': 'candidate B', 'paths': None, 'max_results': 10},
        ),
    )
    assert searched.succeeded
    assert searched.result is not None
    searched_result = SearchWorkspaceResult.model_validate_json(canonical_json_bytes(searched.result))
    assert searched_result.hits[0].path == 'sources/evidence.txt'

    model_request = ModelGenerateRequest(
        messages=(
            AgenticModelMessage(role='system', content='Use only frozen evidence.'),
            AgenticModelMessage(role='user', content='Which candidate is stronger?'),
        ),
        max_output_tokens=10,
    )
    rpc_model_request = _request(
        3,
        GuestRpcMethod.MODEL_GENERATE.value,
        model_request.model_dump(mode='json'),
    )
    generated_frame, generated = _send(session, rpc_model_request)
    replayed_model_frame, replayed_model = _send(session, rpc_model_request)
    assert generated.succeeded
    assert replayed_model == generated
    assert replayed_model_frame == generated_frame
    assert generated.result is not None
    generated_result = ModelGenerateResult.model_validate_json(canonical_json_bytes(generated.result))
    assert generated_result.response.resolved_model_id == 'fixture-model-2025-01-02'
    assert fixture.adapter.call_count == 1

    submit_request = _request(
        4,
        GuestRpcMethod.SUBMIT.value,
        SubmitRequest(submission=_submission()).model_dump(mode='json'),
    )
    first_submit_frame, submitted = _send(session, submit_request)
    replay_submit_frame, replayed = _send(session, submit_request)
    assert submitted.succeeded
    assert replayed == submitted
    assert replay_submit_frame == first_submit_frame
    assert len(session.attempts) == 5

    artifact = session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.seal.terminal_status == GuestRpcTerminalStatus.COMPLETED
    assert artifact.seal.exact_replay_count == 2
    assert artifact.seal.accepted_sequence_count == 5
    assert artifact.seal.final_submission_sha256 == hashlib.sha256(canonical_json_bytes(_submission())).hexdigest()
    assert tuple(event.tool for event in artifact.projected_tool_events) == (
        AgenticTool.LIST_WORKSPACE,
        AgenticTool.READ_WORKSPACE,
        AgenticTool.SEARCH_WORKSPACE,
        AgenticTool.MODEL_GENERATE,
    )
    assert artifact.projected_tool_events[-1].gateway_call_index == 0
    assert _GATEWAY_SECRET not in canonical_json_bytes(artifact)


def test_guest_rpc_accepts_task_bound_clinical_execution_submission(tmp_path: Path) -> None:
    task, _ = _execution_case()
    submission = _execution_submission(task)
    invocation = AgenticTaskInvocation.from_task(task, workspace_manifest_sha256=_WORKSPACE_SHA)
    fixture = _fixture(tmp_path, task_invocation=invocation)

    _, response = _send(
        fixture.session,
        _request(
            0,
            GuestRpcMethod.SUBMIT.value,
            SubmitRequest(submission=submission).model_dump(mode='json'),
        ),
    )

    assert response.succeeded
    artifact = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact, task_invocation=invocation)
    assert artifact.task_invocation == invocation
    assert artifact.submission == submission
    assert artifact.seal.response_protocol == invocation.response_protocol
    assert artifact.seal.task_invocation_sha256 == agentic_task_invocation_sha256(invocation)


def test_guest_rpc_rejects_cross_family_and_wrong_task_clinical_submissions(tmp_path: Path) -> None:
    task, _ = _execution_case()
    invocation = AgenticTaskInvocation.from_task(task, workspace_manifest_sha256=_WORKSPACE_SHA)

    cross_family = _fixture(tmp_path / 'cross-family', task_invocation=invocation)
    _, cross_family_response = _send(
        cross_family.session,
        _request(
            0,
            GuestRpcMethod.SUBMIT.value,
            SubmitRequest(submission=_submission()).model_dump(mode='json'),
        ),
    )
    assert cross_family_response.error_code == GuestRpcErrorCode.SUBMISSION_REJECTED
    cross_family_artifact = cross_family.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(cross_family_artifact, task_invocation=invocation)
    assert cross_family_artifact.submission is None

    wrong_task = _fixture(tmp_path / 'wrong-task', task_invocation=invocation)
    mismatched = _execution_submission(task).model_copy(update={'target_trial_id': 'trial-other'})
    _, wrong_task_response = _send(
        wrong_task.session,
        _request(
            0,
            GuestRpcMethod.SUBMIT.value,
            SubmitRequest(submission=mismatched).model_dump(mode='json'),
        ),
    )
    assert wrong_task_response.error_code == GuestRpcErrorCode.SUBMISSION_REJECTED
    wrong_task_artifact = wrong_task.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(wrong_task_artifact, task_invocation=invocation)
    assert wrong_task_artifact.submission is None


def test_model_call_is_not_dispatched_when_success_cannot_fit_wire_budget(tmp_path: Path) -> None:
    maximum_frame_body_bytes = 4096
    model_body = ModelGenerateRequest(
        messages=(
            AgenticModelMessage(role='system', content='Use only frozen evidence.'),
            AgenticModelMessage(role='user', content='Which candidate is stronger?'),
        ),
        max_output_tokens=10,
    ).model_dump(mode='json')

    baseline = _fixture(tmp_path / 'baseline-model', maximum_frame_body_bytes=maximum_frame_body_bytes)
    prefix_wire_bytes = _fill_with_list_requests(baseline.session, count=15)
    request = _request(15, GuestRpcMethod.MODEL_GENERATE.value, model_body)
    request_frame = encode_guest_rpc_frame(request, maximum_body_bytes=maximum_frame_body_bytes)
    success_frame, success = _send(baseline.session, request)
    assert success.succeeded
    success_exchange_bytes = len(request_frame) + len(success_frame)

    limit_frame = encode_guest_rpc_frame(
        GuestRpcResponse(
            session_id=_SESSION_ID,
            sequence=15,
            succeeded=False,
            error_code=GuestRpcErrorCode.LIMIT_EXCEEDED,
            error_message='rpc request rejected',
        ),
        maximum_body_bytes=maximum_frame_body_bytes,
    )
    limit_exchange_bytes = len(request_frame) + len(limit_frame)
    maximum_session_wire_bytes = prefix_wire_bytes + success_exchange_bytes - 1
    assert prefix_wire_bytes + limit_exchange_bytes <= maximum_session_wire_bytes

    constrained = _fixture(
        tmp_path / 'constrained-model',
        maximum_frame_body_bytes=maximum_frame_body_bytes,
        maximum_session_wire_bytes=maximum_session_wire_bytes,
    )
    assert _fill_with_list_requests(constrained.session, count=15) == prefix_wire_bytes
    _, rejected = _send(constrained.session, request)

    assert rejected.error_code == GuestRpcErrorCode.LIMIT_EXCEEDED
    assert constrained.adapter.call_count == 0
    artifact = constrained.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.seal.terminal_status == GuestRpcTerminalStatus.FAILED
    assert artifact.seal.model_call_count == 0
    assert artifact.seal.wire_bytes == prefix_wire_bytes + limit_exchange_bytes
    assert artifact.attempts[-1].tool == AgenticTool.MODEL_GENERATE
    assert artifact.attempts[-1].gateway_call_index is None


def test_dispatched_model_with_oversize_guest_response_remains_sealable(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        maximum_frame_body_bytes=1024,
        provider_content='x' * 2000,
    )
    request = _request(
        0,
        GuestRpcMethod.MODEL_GENERATE.value,
        ModelGenerateRequest(
            messages=(AgenticModelMessage(role='system', content='Use frozen evidence.'),),
            max_output_tokens=10,
        ).model_dump(mode='json'),
    )

    _, rejected = _send(fixture.session, request)

    assert rejected.error_code == GuestRpcErrorCode.LIMIT_EXCEEDED
    assert fixture.adapter.call_count == 1
    artifact = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.seal.terminal_status == GuestRpcTerminalStatus.FAILED
    assert artifact.seal.model_call_count == 1
    assert artifact.attempts[-1].gateway_call_index == 0
    assert artifact.attempts[-1].projected_tool_event_index is None


def test_submission_is_not_accepted_when_success_cannot_fit_wire_budget(tmp_path: Path) -> None:
    maximum_frame_body_bytes = 4096
    submit_body = SubmitRequest(submission=_submission()).model_dump(mode='json')

    baseline = _fixture(tmp_path / 'baseline-submit', maximum_frame_body_bytes=maximum_frame_body_bytes)
    prefix_wire_bytes = _fill_with_list_requests(baseline.session, count=15)
    request = _request(15, GuestRpcMethod.SUBMIT.value, submit_body)
    request_frame = encode_guest_rpc_frame(request, maximum_body_bytes=maximum_frame_body_bytes)
    success_frame, success = _send(baseline.session, request)
    assert success.succeeded
    success_exchange_bytes = len(request_frame) + len(success_frame)

    limit_frame = encode_guest_rpc_frame(
        GuestRpcResponse(
            session_id=_SESSION_ID,
            sequence=15,
            succeeded=False,
            error_code=GuestRpcErrorCode.LIMIT_EXCEEDED,
            error_message='rpc request rejected',
        ),
        maximum_body_bytes=maximum_frame_body_bytes,
    )
    limit_exchange_bytes = len(request_frame) + len(limit_frame)
    maximum_session_wire_bytes = prefix_wire_bytes + success_exchange_bytes - 1
    assert prefix_wire_bytes + limit_exchange_bytes <= maximum_session_wire_bytes

    constrained = _fixture(
        tmp_path / 'constrained-submit',
        maximum_frame_body_bytes=maximum_frame_body_bytes,
        maximum_session_wire_bytes=maximum_session_wire_bytes,
    )
    assert _fill_with_list_requests(constrained.session, count=15) == prefix_wire_bytes
    _, rejected = _send(constrained.session, request)

    assert rejected.error_code == GuestRpcErrorCode.LIMIT_EXCEEDED
    artifact = constrained.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.seal.terminal_status == GuestRpcTerminalStatus.FAILED
    assert artifact.seal.wire_bytes == prefix_wire_bytes + limit_exchange_bytes
    assert artifact.seal.submit_attempted
    assert not artifact.seal.submit_accepted
    assert artifact.submission is None
    assert artifact.seal.final_submission_bytes == 0


@pytest.mark.parametrize(
    ('method', 'body', 'expected_tool'),
    [
        (
            GuestRpcMethod.MODEL_GENERATE,
            ModelGenerateRequest(
                messages=(AgenticModelMessage(role='system', content='Use frozen evidence.'),),
                max_output_tokens=1,
            ).model_dump(mode='json'),
            AgenticTool.MODEL_GENERATE,
        ),
        (
            GuestRpcMethod.SUBMIT,
            SubmitRequest(submission=_submission()).model_dump(mode='json'),
            None,
        ),
    ],
)
def test_side_effecting_request_limit_rejection_is_sealable(
    tmp_path: Path,
    method: GuestRpcMethod,
    body: dict,
    expected_tool: AgenticTool | None,
) -> None:
    fixture = _fixture(tmp_path, maximum_requests=1)
    _send(fixture.session, _request(0, GuestRpcMethod.LIST.value, {'cursor': 0, 'limit': 1}))

    _, rejected = _send(fixture.session, _request(1, method.value, body))

    assert rejected.error_code == GuestRpcErrorCode.LIMIT_EXCEEDED
    assert fixture.adapter.call_count == 0
    artifact = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.attempts[-1].tool == expected_tool
    assert artifact.submission is None


def test_conflicting_retry_cannot_rewrite_completed_submission(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    accepted_request = _request(
        0,
        GuestRpcMethod.SUBMIT.value,
        SubmitRequest(submission=_submission()).model_dump(mode='json'),
    )
    _, accepted = _send(fixture.session, accepted_request)
    assert accepted.succeeded
    conflicting_submission = _submission().model_copy(update={'task_id': 'different-task'})

    _, rejected = _send(
        fixture.session,
        _request(
            0,
            GuestRpcMethod.SUBMIT.value,
            SubmitRequest(submission=conflicting_submission).model_dump(mode='json'),
        ),
    )

    assert rejected.error_code == GuestRpcErrorCode.TERMINAL
    assert len(fixture.session.attempts) == 1
    artifact = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.seal.terminal_status == GuestRpcTerminalStatus.COMPLETED
    assert artifact.submission == _submission()


@pytest.mark.parametrize('kind', ['oversize', 'truncated', 'trailing', 'noncanonical'])
def test_guest_rpc_rejects_bad_frames_before_dispatch(tmp_path: Path, kind: str) -> None:
    fixture = _fixture(tmp_path, maximum_frame_body_bytes=4096)
    request = _request(0, GuestRpcMethod.LIST.value, {'cursor': 0, 'limit': 1})
    frame = encode_guest_rpc_frame(request, maximum_body_bytes=4096)
    if kind == 'oversize':
        bad = struct.pack('>I', 4097)
    elif kind == 'truncated':
        bad = frame[:-1]
    elif kind == 'trailing':
        bad = frame + b'x'
    else:
        body = canonical_json_bytes(request) + b' '
        bad = struct.pack('>I', len(body)) + body

    with pytest.raises(GuestRpcError):
        fixture.session.handle_frame(bad)

    assert fixture.adapter.call_count == 0
    artifact = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.seal.terminal_error_code == GuestRpcErrorCode.INVALID_BODY
    assert artifact.attempts == ()


def test_guest_rpc_rejects_skipped_and_conflicting_duplicate_sequences(tmp_path: Path) -> None:
    skipped_fixture = _fixture(tmp_path / 'skipped')
    _, skipped = _send(
        skipped_fixture.session,
        _request(1, GuestRpcMethod.LIST.value, {'cursor': 0, 'limit': 1}),
    )
    assert skipped.error_code == GuestRpcErrorCode.OUT_OF_ORDER
    assert skipped_fixture.adapter.call_count == 0
    skipped_artifact = skipped_fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(skipped_artifact)
    assert skipped_artifact.seal.accepted_sequence_count == 0

    conflict_fixture = _fixture(tmp_path / 'conflict')
    original = _request(0, GuestRpcMethod.LIST.value, {'cursor': 0, 'limit': 1})
    first_frame, _ = _send(conflict_fixture.session, original)
    replay_frame, _ = _send(conflict_fixture.session, original)
    assert replay_frame == first_frame
    assert len(conflict_fixture.session.attempts) == 1
    _, conflict = _send(
        conflict_fixture.session,
        _request(0, GuestRpcMethod.LIST.value, {'cursor': 0, 'limit': 2}),
    )
    assert conflict.error_code == GuestRpcErrorCode.REPLAY_CONFLICT
    artifact = conflict_fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.seal.terminal_status == GuestRpcTerminalStatus.FAILED
    assert artifact.seal.exact_replay_count == 1
    assert artifact.seal.accepted_sequence_count == 1


@pytest.mark.parametrize(
    ('method', 'body', 'expected'),
    [
        ('open_shell', {}, GuestRpcErrorCode.UNKNOWN_METHOD),
        (
            GuestRpcMethod.MODEL_GENERATE.value,
            {
                'messages': [{'role': 'system', 'content': 'hello'}],
                'max_output_tokens': 1,
                'response_schema_sha256': None,
                'provider': 'attacker-selected',
            },
            GuestRpcErrorCode.INVALID_BODY,
        ),
    ],
)
def test_guest_rpc_unknown_method_and_provider_route_injection_never_dispatch(
    tmp_path: Path,
    method: str,
    body: dict,
    expected: GuestRpcErrorCode,
) -> None:
    fixture = _fixture(tmp_path)

    response_frame, response = _send(fixture.session, _request(0, method, body))

    assert response.error_code == expected
    assert response.result is None
    assert response.error_message == 'rpc request rejected'
    assert fixture.adapter.call_count == 0
    assert _GATEWAY_SECRET not in response_frame
    artifact = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
    assert artifact.seal.terminal_status == GuestRpcTerminalStatus.FAILED


def test_guest_rpc_session_tampering_fails_verification(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _send(
        fixture.session,
        _request(
            0,
            GuestRpcMethod.SUBMIT.value,
            SubmitRequest(submission=_submission()).model_dump(mode='json'),
        ),
    )
    artifact = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    tampered_seal = artifact.model_copy(
        update={'seal': artifact.seal.model_copy(update={'rpc_port': artifact.seal.rpc_port + 1})}
    )
    attempt = artifact.attempts[0]
    tampered_response = attempt.response.model_copy(
        update={'result': {'submission_sha256': '0' * 64, 'submission_bytes': 1}}
    )
    tampered_attempts = artifact.model_copy(
        update={'attempts': (attempt.model_copy(update={'response': tampered_response}),)}
    )

    with pytest.raises(ValueError, match='authentication failed'):
        _verify(tampered_seal)
    with pytest.raises(ValueError, match='attempt log'):
        _verify(tampered_attempts)


def test_guest_rpc_client_and_host_server_over_existing_socket(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    host_socket, guest_socket = socket.socketpair()
    host_socket.settimeout(2)
    guest_socket.settimeout(2)
    server = GuestRpcHostServer(fixture.session)
    thread = threading.Thread(target=server.serve, args=(host_socket,))
    thread.start()
    try:
        client = GuestRpcClient(
            guest_socket,
            session_id=_SESSION_ID,
            task_invocation=_task_invocation(),
            maximum_body_bytes=fixture.session.policy.maximum_frame_body_bytes,
        )
        listed = client.list_workspace(limit=1)
        assert listed.files[0].path == 'TASK.md'
        read = client.read_workspace('sources/evidence.txt', limit=9)
        assert read.content == b'candidate'
        generated = client.model_generate(
            messages=(
                AgenticModelMessage(role='system', content='Use frozen evidence.'),
                AgenticModelMessage(role='user', content='Rank candidates.'),
            ),
            max_output_tokens=10,
        )
        assert generated.content.startswith('Candidate B')
        result = client.submit(_submission())
        assert result.submission_bytes == len(canonical_json_bytes(_submission()))
    finally:
        guest_socket.close()
        thread.join(timeout=3)
        host_socket.close()
    assert not thread.is_alive()
    artifact = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))
    _verify(artifact)
