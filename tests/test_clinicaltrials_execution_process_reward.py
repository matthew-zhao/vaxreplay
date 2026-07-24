from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pytest

from tests.test_agentic_guest_rpc import (
    _ISSUED_AT,
    _RPC_RECEIPT_KEY,
    _fixture,
    _request,
    _send,
)
from tests.test_clinicaltrials_execution_scoring import _case as _execution_case
from tests.test_clinicaltrials_execution_scoring import _submission as _execution_submission
from vaxreplay.agentic.clinical_execution_bridge import (
    build_clinical_agentic_workspace,
    clinical_workspace_receipt_key_id,
)
from vaxreplay.agentic.gateway import AgenticModelMessage
from vaxreplay.agentic.guest_rpc import (
    GuestRpcMethod,
    ModelGenerateRequest,
    SubmitRequest,
    guest_rpc_session_key_id,
)
from vaxreplay.clinicaltrials.execution_process_reward import (
    AuthenticatedLaneAProcessEvidence,
    LaneAProcessEvent,
    LaneAProcessEventKind,
    LaneAProcessEvidence,
    LaneAProcessRewardError,
    LaneAProcessSource,
    LaneAProcessTerminalStatus,
    authenticate_lane_a_process_evidence,
    lane_a_process_trace_key_id,
    project_and_authenticate_lane_a_process_evidence_from_guest_rpc,
    score_authenticated_lane_a_process_evidence,
)

_KEY = b'process-trace-key-material-32bytes!'
_KEY_ID = lane_a_process_trace_key_id(_KEY)
_WORKSPACE = 'a' * 64
_CONTEXT = 'b' * 64


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _event(index: int, kind: LaneAProcessEventKind, **updates) -> LaneAProcessEvent:
    succeeded = updates.pop('succeeded', True)
    return LaneAProcessEvent(
        event_index=index,
        kind=kind,
        succeeded=succeeded,
        request_sha256=_hash(f'request-{index}'),
        response_sha256=_hash(f'response-{index}'),
        **updates,
    )


def _evidence() -> LaneAProcessEvidence:
    first = 'sources/source-001.txt'
    second = 'sources/source-002.txt'
    events = (
        _event(0, LaneAProcessEventKind.READ, logical_path=first, offset=0, byte_count=100),
        _event(
            1,
            LaneAProcessEventKind.SEARCH,
            query_sha256=_hash('targeted-query-1'),
            restricted_paths=(second,),
            hit_paths=(second,),
        ),
        _event(2, LaneAProcessEventKind.READ, logical_path=second, offset=0, byte_count=100),
        _event(
            3,
            LaneAProcessEventKind.SEARCH,
            query_sha256=_hash('targeted-query-2'),
            restricted_paths=(first,),
            hit_paths=(first,),
        ),
        # Duplicate bytes do not increase coverage, but this later read proves the second search
        # led to retrieval without relying on generated prose.
        _event(4, LaneAProcessEventKind.READ, logical_path=first, offset=0, byte_count=10),
        _event(
            5,
            LaneAProcessEventKind.MODEL,
        ),
        _event(6, LaneAProcessEventKind.SUBMIT, submission_contract_valid=True),
    )
    return LaneAProcessEvidence(
        episode_id='fictional-process-task-001',
        task_context_sha256=_CONTEXT,
        workspace_manifest_sha256=_WORKSPACE,
        workspace_tree_sha256='c' * 64,
        model_visible_surface_sha256='d' * 64,
        broker_attestation_sha256='e' * 64,
        authenticated_source_trace_sha256='f' * 64,
        sources=(
            LaneAProcessSource(logical_path=first, sha256=_hash('source-1'), byte_count=100),
            LaneAProcessSource(logical_path=second, sha256=_hash('source-2'), byte_count=100),
        ),
        events=events,
        terminal_status=LaneAProcessTerminalStatus.COMPLETED,
        trace_event_count=len(events),
    )


def test_full_authenticated_process_trace_scores_without_outcome_or_reasoning() -> None:
    authenticated = authenticate_lane_a_process_evidence(_evidence(), key=_KEY, expected_key_id=_KEY_ID)
    score = score_authenticated_lane_a_process_evidence(
        authenticated,
        key=_KEY,
        expected_key_id=_KEY_ID,
        expected_workspace_manifest_sha256=_WORKSPACE,
        expected_task_context_sha256=_CONTEXT,
    )

    assert score.process_reward == pytest.approx(0.7)
    assert tuple(component.score for component in score.components) == pytest.approx((1.0, 1.0, 0.0, 1.0, 0.0))
    assert score.maximum_creditable_process_reward == pytest.approx(0.7)
    assert score.reward_role == 'auxiliary_diagnostic_only'
    assert not score.eligible_as_primary_outcome_reward
    assert not score.complete_guest_tool_trace_claimed
    assert not score.evidence_reference_credit_enabled
    assert not score.reproducible_computation_credit_enabled
    assert score.combined_reward is None
    assert score.successful_model_call_observed
    assert score.research_process_credit_eligible
    assert not score.outcome_reward_included
    assert not score.chain_of_thought_used
    assert not score.model_response_content_used
    assert not score.later_outcome_used
    assert not score.private_gold_used
    assert 'registry_outcome_class' not in _evidence().model_dump_json()


def test_duplicate_reads_and_search_spam_do_not_create_extra_credit() -> None:
    evidence = _evidence()
    spam = _event(
        6,
        LaneAProcessEventKind.SEARCH,
        succeeded=False,
        query_sha256=_hash('spam'),
        restricted_paths=('sources/source-001.txt',),
        hit_paths=(),
    )
    submit = evidence.events[-1].model_copy(update={'event_index': 7})
    changed = evidence.model_copy(update={'events': (*evidence.events[:-1], spam, submit), 'trace_event_count': 8})
    authenticated = authenticate_lane_a_process_evidence(changed, key=_KEY, expected_key_id=_KEY_ID)
    score = score_authenticated_lane_a_process_evidence(
        authenticated,
        key=_KEY,
        expected_key_id=_KEY_ID,
        expected_workspace_manifest_sha256=_WORKSPACE,
        expected_task_context_sha256=_CONTEXT,
    )

    source = next(item for item in score.components if item.component == 'source_coverage')
    retrieval = next(item for item in score.components if item.component == 'targeted_retrieval')
    assert source.credited_items == source.denominator_items == 200
    assert retrieval.credited_items == 2
    assert retrieval.denominator_items == 3
    assert retrieval.score == pytest.approx(2 / 3)
    assert score.process_reward < score.maximum_creditable_process_reward


def test_no_model_call_and_zero_byte_post_search_read_receive_no_process_credit() -> None:
    evidence = _evidence()
    no_model_events = tuple(
        event.model_copy(update={'event_index': index})
        for index, event in enumerate(item for item in evidence.events if item.kind != LaneAProcessEventKind.MODEL)
    )
    no_model = evidence.model_copy(update={'events': no_model_events, 'trace_event_count': len(no_model_events)})
    authenticated = authenticate_lane_a_process_evidence(no_model, key=_KEY, expected_key_id=_KEY_ID)
    score = score_authenticated_lane_a_process_evidence(
        authenticated,
        key=_KEY,
        expected_key_id=_KEY_ID,
        expected_workspace_manifest_sha256=_WORKSPACE,
        expected_task_context_sha256=_CONTEXT,
    )

    assert score.process_reward == 0.0
    assert not score.successful_model_call_observed
    assert not score.research_process_credit_eligible

    zero_byte_events = list(evidence.events)
    zero_byte_events[4] = zero_byte_events[4].model_copy(update={'byte_count': 0})
    zero_byte = evidence.model_copy(update={'events': tuple(zero_byte_events)})
    authenticated_zero = authenticate_lane_a_process_evidence(zero_byte, key=_KEY, expected_key_id=_KEY_ID)
    zero_score = score_authenticated_lane_a_process_evidence(
        authenticated_zero,
        key=_KEY,
        expected_key_id=_KEY_ID,
        expected_workspace_manifest_sha256=_WORKSPACE,
        expected_task_context_sha256=_CONTEXT,
    )
    retrieval = next(item for item in zero_score.components if item.component == 'targeted_retrieval')
    assert retrieval.credited_items == 1
    assert retrieval.score == pytest.approx(0.5)


def test_unbound_guest_local_compute_cannot_claim_reserved_credit() -> None:
    evidence = _evidence()
    compute = _event(
        6,
        LaneAProcessEventKind.COMPUTE,
        computation_recipe_sha256=_hash('trivial-identity-recipe'),
        computation_executable_sha256=_hash('participant-controlled-executable'),
        computation_input_sha256=_hash('unproven-input'),
        computation_output_sha256=_hash('unconsumed-output'),
        deterministic_computation=True,
        computation_network_used=False,
    )
    submit = evidence.events[-1].model_copy(update={'event_index': 7})
    changed = evidence.model_copy(update={'events': (*evidence.events[:-1], compute, submit), 'trace_event_count': 8})

    with pytest.raises(ValueError, match='does not authenticate guest-local computation'):
        LaneAProcessEvidence.model_validate_json(changed.model_dump_json())


def test_authenticated_guest_rpc_adapter_projects_only_authoritative_events(tmp_path: Path) -> None:
    task, _ = _execution_case()
    workspace_key = b'clinical-workspace-process-test-key'
    workspace = build_clinical_agentic_workspace(
        task=task,
        workspace_id='clinical-process-fixture',
        output_root=tmp_path / 'workspace',
        receipt_key=workspace_key,
        expected_receipt_key_id=clinical_workspace_receipt_key_id(workspace_key),
    )
    fixture = _fixture(
        tmp_path / 'rpc',
        task_invocation=workspace.invocation,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        broker=workspace.brokered_surface(),
    )
    source = next(item for item in workspace.manifest.entries if item.path.startswith('sources/'))
    _, searched = _send(
        fixture.session,
        _request(
            0,
            GuestRpcMethod.SEARCH.value,
            {'needle': 'Phase', 'paths': [source.path], 'max_results': 10},
        ),
    )
    assert searched.succeeded
    _, read = _send(
        fixture.session,
        _request(
            1,
            GuestRpcMethod.READ.value,
            {'path': source.path, 'offset': 0, 'limit': source.byte_count},
        ),
    )
    assert read.succeeded
    _, generated = _send(
        fixture.session,
        _request(
            2,
            GuestRpcMethod.MODEL_GENERATE.value,
            ModelGenerateRequest(
                messages=(
                    AgenticModelMessage(role='system', content='Use only the frozen workspace.'),
                    AgenticModelMessage(role='user', content='Assess the target trial.'),
                ),
                max_output_tokens=10,
            ).model_dump(mode='json'),
        ),
    )
    assert generated.succeeded
    _, submitted = _send(
        fixture.session,
        _request(
            3,
            GuestRpcMethod.SUBMIT.value,
            SubmitRequest(submission=_execution_submission(task)).model_dump(mode='json'),
        ),
    )
    assert submitted.succeeded
    session = fixture.session.seal(sealed_at=_ISSUED_AT + timedelta(seconds=2))

    authenticated = project_and_authenticate_lane_a_process_evidence_from_guest_rpc(
        session,
        workspace_manifest=workspace.manifest,
        guest_rpc_receipt_key=_RPC_RECEIPT_KEY,
        expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(_RPC_RECEIPT_KEY),
        expected_run_id=session.seal.run_id,
        expected_execution_policy_sha256=session.seal.execution_policy_sha256,
        expected_peer_cid=session.seal.expected_peer_cid,
        expected_rpc_port=session.seal.rpc_port,
        process_trace_key=_KEY,
        expected_process_trace_key_id=_KEY_ID,
    )
    score = score_authenticated_lane_a_process_evidence(
        authenticated,
        key=_KEY,
        expected_key_id=_KEY_ID,
        expected_workspace_manifest_sha256=workspace.manifest_sha256,
        expected_task_context_sha256=task.context_sha256,
    )

    assert tuple(item.kind for item in authenticated.evidence.events) == (
        LaneAProcessEventKind.SEARCH,
        LaneAProcessEventKind.READ,
        LaneAProcessEventKind.MODEL,
        LaneAProcessEventKind.SUBMIT,
    )
    assert authenticated.evidence.sources == (
        LaneAProcessSource(logical_path=source.path, sha256=source.sha256, byte_count=source.byte_count),
    )
    assert all(not item.referenced_source_paths for item in authenticated.evidence.events)
    assert all(item.kind != LaneAProcessEventKind.COMPUTE for item in authenticated.evidence.events)
    assert not authenticated.evidence.complete_guest_tool_trace_claimed
    assert not authenticated.evidence.guest_local_computation_authoritative
    assert score.process_reward == pytest.approx(0.7)


def test_tampered_or_cross_workspace_process_evidence_fails_closed() -> None:
    authenticated = authenticate_lane_a_process_evidence(_evidence(), key=_KEY, expected_key_id=_KEY_ID)
    forged = authenticated.model_copy(update={'evidence_hmac_sha256': '0' * 64})
    with pytest.raises(LaneAProcessRewardError, match='authentication'):
        score_authenticated_lane_a_process_evidence(
            forged,
            key=_KEY,
            expected_key_id=_KEY_ID,
            expected_workspace_manifest_sha256=_WORKSPACE,
            expected_task_context_sha256=_CONTEXT,
        )
    with pytest.raises(LaneAProcessRewardError, match='different task or workspace'):
        score_authenticated_lane_a_process_evidence(
            authenticated,
            key=_KEY,
            expected_key_id=_KEY_ID,
            expected_workspace_manifest_sha256='0' * 64,
            expected_task_context_sha256=_CONTEXT,
        )
    with pytest.raises(ValueError, match='exact trace'):
        AuthenticatedLaneAProcessEvidence(
            evidence=_evidence().model_copy(update={'episode_id': 'different'}),
            evidence_sha256=authenticated.evidence_sha256,
            trace_key_id=_KEY_ID,
            evidence_hmac_sha256=authenticated.evidence_hmac_sha256,
        )
