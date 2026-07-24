from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from typing import Literal

import pytest

from tests.test_clinicaltrials_execution_scoring import _case, _submission
from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_HARNESS_POLICY,
    LaneAGuestHarnessError,
    LaneAGuestHarnessFailureCode,
    run_lane_a_guest_harness,
)
from vaxreplay.agentic.gateway import AgenticGatewayUsage, AgenticModelMessage, AgenticModelResponse
from vaxreplay.agentic.guest_rpc import (
    ListWorkspaceResult,
    LogicalFileResult,
    LogicalSearchHitResult,
    ReadWorkspaceResult,
    SearchWorkspaceResult,
    SubmitResult,
)
from vaxreplay.agentic.task_protocol import AgenticTaskInvocation
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission, ExecutionTask

_WORKSPACE_MANIFEST_SHA256 = 'a' * 64
_RUN_ID = '1' * 32


class _FakeGuestRpcClient:
    """Scripted RPC boundary; the script, rather than this fake, chooses each tool action."""

    def __init__(
        self,
        model_outputs: Sequence[str],
        *,
        stop_reasons: Sequence[Literal['completed', 'max_output_tokens', 'refusal', 'provider_error']] | None = None,
    ) -> None:
        self.model_outputs = tuple(model_outputs)
        self.stop_reasons = tuple(stop_reasons or ('completed',) * len(self.model_outputs))
        assert len(self.stop_reasons) == len(self.model_outputs)
        self.files = {
            'TASK.md': b'Forecast the frozen clinical trial using only broker evidence.\n',
            'sources/source-001.txt': b'Alpha record contains TargetMarker and an early status.\n',
            'sources/source-002.txt': b'Beta record contains ReferenceMarker and a later status.\n',
            'sources/source-999.txt': b'Decoy record contains DecoyMarker and must not be selected.\n',
        }
        self.model_calls: list[tuple[tuple[AgenticModelMessage, ...], int, str | None]] = []
        self.list_calls: list[tuple[int, int]] = []
        self.read_calls: list[tuple[str, int, int]] = []
        self.search_calls: list[tuple[str, tuple[str, ...] | None, int]] = []
        self.submissions: list[ExecutionSubmission] = []

    def list_workspace(self, *, cursor: int = 0, limit: int = 100) -> ListWorkspaceResult:
        self.list_calls.append((cursor, limit))
        paths = sorted(self.files)
        page = paths[cursor : cursor + limit]
        next_cursor = cursor + len(page) if cursor + len(page) < len(paths) else None
        return ListWorkspaceResult(
            files=tuple(
                LogicalFileResult(
                    path=path,
                    media_type='text/plain',
                    sha256=hashlib.sha256(self.files[path]).hexdigest(),
                    byte_count=len(self.files[path]),
                )
                for path in page
            ),
            next_cursor=next_cursor,
        )

    def read_workspace(self, path: str, *, offset: int = 0, limit: int) -> ReadWorkspaceResult:
        self.read_calls.append((path, offset, limit))
        content = self.files[path]
        selected = content[offset : offset + limit]
        return ReadWorkspaceResult(
            content_base64=base64.b64encode(selected).decode('ascii'),
            offset=offset,
            byte_count=len(selected),
            eof=offset + len(selected) >= len(content),
        )

    def search_workspace(
        self,
        needle: str,
        *,
        paths: tuple[str, ...] | None = None,
        max_results: int = 100,
    ) -> SearchWorkspaceResult:
        self.search_calls.append((needle, paths, max_results))
        needle_bytes = needle.encode('utf-8')
        selected_paths = paths if paths is not None else tuple(sorted(self.files))
        hits: list[LogicalSearchHitResult] = []
        for path in selected_paths:
            start = self.files[path].find(needle_bytes)
            if start >= 0:
                hits.append(
                    LogicalSearchHitResult(
                        path=path,
                        start_byte=start,
                        end_byte=start + len(needle_bytes),
                    )
                )
            if len(hits) == max_results:
                break
        return SearchWorkspaceResult(hits=tuple(hits))

    def model_generate(
        self,
        *,
        messages: tuple[AgenticModelMessage, ...],
        max_output_tokens: int,
        response_schema_sha256: str | None = None,
    ) -> AgenticModelResponse:
        call_index = len(self.model_calls)
        self.model_calls.append((messages, max_output_tokens, response_schema_sha256))
        return AgenticModelResponse(
            run_id=_RUN_ID,
            call_index=call_index,
            resolved_model_id='fixture-model-2025-01-02',
            content=self.model_outputs[call_index],
            stop_reason=self.stop_reasons[call_index],
            usage=AgenticGatewayUsage(input_tokens=17, output_tokens=7, reasoning_tokens=None),
        )

    def submit(self, submission: ExecutionSubmission) -> SubmitResult:
        self.submissions.append(submission)
        payload = canonical_json_bytes(submission)
        return SubmitResult(
            submission_sha256=hashlib.sha256(payload).hexdigest(),
            submission_bytes=len(payload),
        )


def _invocation(*, with_fact: bool = False) -> AgenticTaskInvocation:
    task, _ = _case(with_fact=with_fact)
    return AgenticTaskInvocation.from_task(
        task,
        workspace_manifest_sha256=_WORKSPACE_MANIFEST_SHA256,
    )


def _json_action(value: object) -> str:
    return canonical_json_bytes(value).decode('utf-8')


def _submit_action(submission: ExecutionSubmission) -> str:
    return _json_action(
        {
            'action': 'submit',
            'submission': submission.model_dump(mode='json'),
        }
    )


@pytest.mark.parametrize(
    ('selected_path', 'needle'),
    (
        ('sources/source-001.txt', 'TargetMarker'),
        ('sources/source-002.txt', 'ReferenceMarker'),
    ),
)
def test_model_controls_selective_retrieval_and_submits_after_four_calls(
    selected_path: str,
    needle: str,
) -> None:
    invocation = _invocation()
    assert isinstance(invocation.task, ExecutionTask)
    submission = _submission(invocation.task)
    fake = _FakeGuestRpcClient(
        (
            _json_action({'action': 'list_workspace', 'cursor': 0, 'limit': 100}),
            _json_action(
                {
                    'action': 'search_workspace',
                    'needle': needle,
                    'paths': [selected_path],
                    'max_results': 5,
                }
            ),
            _json_action(
                {
                    'action': 'read_workspace',
                    'path': selected_path,
                    'offset': fake_hit_offset(selected_path, needle),
                    'limit': len(needle.encode('utf-8')),
                }
            ),
            _submit_action(submission),
        )
    )

    result = run_lane_a_guest_harness(fake, task_invocation=invocation)

    assert result.model_call_count == 4
    assert result.workspace_action_count == 3
    assert result.targeted_source_read_count == 1
    assert result.submission == submission
    assert fake.search_calls == [(needle, (selected_path,), 5)]
    assert [call[0] for call in fake.read_calls] == [selected_path]
    assert all(call[0] != 'sources/source-999.txt' for call in fake.read_calls)
    assert fake.submissions == [submission]
    assert len(fake.model_calls) == 4
    for messages, max_output_tokens, response_schema_sha256 in fake.model_calls:
        assert tuple(message.role for message in messages) == ('system', 'user')
        assert max_output_tokens == LANE_A_GUEST_HARNESS_POLICY.maximum_model_output_tokens_per_call
        assert response_schema_sha256 is None
        combined_prompt = '\n'.join(message.content for message in messages).lower()
        assert 'chain-of-thought' not in combined_prompt
        assert 'show your work' not in combined_prompt
        state = json.loads(messages[1].content)
        assert state['budgets']['automatic_repair_prompts'] is False
        assert state['budgets']['model_output_rationale_requested'] is False


def fake_hit_offset(path: str, needle: str) -> int:
    contents = {
        'sources/source-001.txt': b'Alpha record contains TargetMarker and an early status.\n',
        'sources/source-002.txt': b'Beta record contains ReferenceMarker and a later status.\n',
    }
    return contents[path].index(needle.encode('utf-8'))


@pytest.mark.parametrize(
    'malformed_output',
    (
        '```json\n{"action":"list_workspace","cursor":0,"limit":10}\n```',
        '{"action":"list_workspace","cursor":0,"limit":10} trailing prose',
        '{"action":"list_workspace","action":"submit","cursor":0,"limit":10}',
        '{"action":"list_workspace","cursor":0,"limit":10,"unexpected":true}',
    ),
)
def test_malformed_actions_fail_terminally_without_retry_or_broker_access(
    malformed_output: str,
) -> None:
    fake = _FakeGuestRpcClient((malformed_output,))

    with pytest.raises(LaneAGuestHarnessError) as raised:
        run_lane_a_guest_harness(fake, task_invocation=_invocation())

    assert raised.value.code is LaneAGuestHarnessFailureCode.MALFORMED_MODEL_ACTION
    assert len(fake.model_calls) == 1
    assert fake.list_calls == []
    assert fake.search_calls == []
    assert fake.read_calls == []
    assert fake.submissions == []


def test_nonterminal_tenth_action_exhausts_the_fixed_step_budget() -> None:
    list_action = _json_action({'action': 'list_workspace', 'cursor': 0, 'limit': 100})
    search_action = _json_action(
        {
            'action': 'search_workspace',
            'needle': 'ReferenceMarker',
            'paths': ['sources/source-002.txt'],
            'max_results': 1,
        }
    )
    fake = _FakeGuestRpcClient((list_action, *(search_action for _ in range(9))))

    with pytest.raises(LaneAGuestHarnessError) as raised:
        run_lane_a_guest_harness(fake, task_invocation=_invocation())

    assert raised.value.code is LaneAGuestHarnessFailureCode.STEP_BUDGET_EXHAUSTED
    assert len(fake.model_calls) == LANE_A_GUEST_HARNESS_POLICY.maximum_model_calls
    assert fake.list_calls == [(0, 100)]
    assert len(fake.search_calls) == 8
    assert fake.submissions == []


def test_task_binding_rejects_a_well_formed_submission_for_another_episode() -> None:
    invocation = _invocation()
    assert isinstance(invocation.task, ExecutionTask)
    valid_submission = _submission(invocation.task)
    wrong_submission = valid_submission.model_copy(update={'episode_id': 'execution-dev-other'})
    selected_path = 'sources/source-001.txt'
    needle = 'TargetMarker'
    fake = _FakeGuestRpcClient(
        (
            _json_action({'action': 'list_workspace', 'cursor': 0, 'limit': 100}),
            _json_action(
                {
                    'action': 'search_workspace',
                    'needle': needle,
                    'paths': [selected_path],
                    'max_results': 1,
                }
            ),
            _json_action(
                {
                    'action': 'read_workspace',
                    'path': selected_path,
                    'offset': fake_hit_offset(selected_path, needle),
                    'limit': len(needle.encode('utf-8')),
                }
            ),
            _submit_action(wrong_submission),
        )
    )

    with pytest.raises(LaneAGuestHarnessError) as raised:
        run_lane_a_guest_harness(fake, task_invocation=invocation)

    assert raised.value.code is LaneAGuestHarnessFailureCode.SUBMISSION_BINDING_INVALID
    assert len(fake.model_calls) == 4
    assert fake.submissions == []
