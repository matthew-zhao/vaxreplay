"""Benchmark-native multi-step guest harness for Lane A clinical execution.

The harness has no filesystem, network, or shell integration.  Its only effectful dependency is
the narrow :class:`GuestRpcClient` contract: the model chooses one strict JSON action per call,
the harness executes that action through guest RPC, and one task-bound ``ExecutionSubmission`` is
submitted terminally.  Invalid output ends the attempt; there is no repair prompt or hidden retry.

Only chosen actions and broker observations are carried between model calls.  The prompt neither
requests nor retains a rationale or private reasoning.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, TypeAdapter, ValidationError, field_validator, model_validator

from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelResponse
from vaxreplay.agentic.guest_rpc import (
    ListWorkspaceResult,
    LogicalFileResult,
    ReadWorkspaceResult,
    SearchWorkspaceResult,
    SubmitResult,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.task_protocol import (
    AgenticTaskInvocation,
    agentic_task_invocation_sha256,
    validate_submission_for_invocation,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission, ExecutionTask

LANE_A_GUEST_HARNESS_POLICY_ID = 'lane-a-benchmark-native-retrieval-agent-v0.1'
LANE_A_GUEST_HARNESS_RESULT_SCHEMA_VERSION = 'vaxreplay.lane-a-guest-harness-result.dev-v0.1'
_STATE_SCHEMA_VERSION = 'vaxreplay.lane-a-guest-harness-state.dev-v0.1'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class LaneAGuestHarnessFailureCode(str, enum.Enum):
    INVALID_TASK_INVOCATION = 'invalid_task_invocation'
    MODEL_RPC_REJECTED = 'model_rpc_rejected'
    MODEL_RESPONSE_INCOMPLETE = 'model_response_incomplete'
    MODEL_ACTION_OVERSIZED = 'model_action_oversized'
    MALFORMED_MODEL_ACTION = 'malformed_model_action'
    ACTION_ORDER_VIOLATION = 'action_order_violation'
    ACTION_BUDGET_EXCEEDED = 'action_budget_exceeded'
    BROKER_RPC_REJECTED = 'broker_rpc_rejected'
    BROKER_RESPONSE_INVALID = 'broker_response_invalid'
    OBSERVATION_BUDGET_EXCEEDED = 'observation_budget_exceeded'
    SUBMISSION_TOO_EARLY = 'submission_too_early'
    SUBMISSION_BINDING_INVALID = 'submission_binding_invalid'
    SUBMISSION_RPC_REJECTED = 'submission_rpc_rejected'
    SUBMISSION_RECEIPT_INVALID = 'submission_receipt_invalid'
    STEP_BUDGET_EXHAUSTED = 'step_budget_exhausted'


class LaneAGuestHarnessError(RuntimeError):
    """Stable, content-free terminal harness failure."""

    def __init__(self, code: LaneAGuestHarnessFailureCode):
        super().__init__(code.value)
        self.code = code


class LaneAGuestHarnessPolicy(StrictModel):
    """Fixed v0.1 action-loop and observation budgets."""

    policy_id: Literal['lane-a-benchmark-native-retrieval-agent-v0.1'] = LANE_A_GUEST_HARNESS_POLICY_ID
    maximum_model_calls: Literal[10] = 10
    maximum_steps: Literal[10] = 10
    minimum_model_calls_before_submission: Literal[4] = 4
    maximum_model_output_tokens_per_call: Literal[4096] = 4096
    maximum_action_response_bytes: Literal[65_536] = 65_536
    maximum_list_entries_per_action: Literal[100] = 100
    maximum_read_bytes_per_action: Literal[32_768] = 32_768
    maximum_total_read_bytes: Literal[131_072] = 131_072
    maximum_search_results_per_action: Literal[20] = 20
    maximum_search_paths_per_action: Literal[8] = 8
    maximum_search_needle_bytes: Literal[256] = 256
    maximum_observation_bytes: Literal[196_608] = 196_608
    list_before_other_workspace_actions: Literal[True] = True
    discovered_paths_only: Literal[True] = True
    source_scoped_search_required: Literal[True] = True
    targeted_source_read_required: Literal[True] = True
    one_terminal_submission: Literal[True] = True
    malformed_action_is_terminal: Literal[True] = True
    automatic_repair_prompts: Literal[False] = False
    model_output_rationale_requested: Literal[False] = False

    @model_validator(mode='after')
    def validate_fixed_budget_relationships(self) -> Self:
        if self.maximum_steps != self.maximum_model_calls:
            raise ValueError('each harness step must consume exactly one model call')
        if self.minimum_model_calls_before_submission > self.maximum_model_calls:
            raise ValueError('minimum call count cannot exceed the fixed model-call budget')
        if self.maximum_total_read_bytes > self.maximum_observation_bytes:
            raise ValueError('read budget cannot exceed the complete observation budget')
        return self


LANE_A_GUEST_HARNESS_POLICY = LaneAGuestHarnessPolicy()


class LaneAListWorkspaceAction(StrictModel):
    action: Literal['list_workspace']
    cursor: int = Field(ge=0)
    limit: int = Field(ge=1, le=LANE_A_GUEST_HARNESS_POLICY.maximum_list_entries_per_action)


class LaneAReadWorkspaceAction(StrictModel):
    action: Literal['read_workspace']
    path: str = Field(min_length=1, max_length=4096)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=LANE_A_GUEST_HARNESS_POLICY.maximum_read_bytes_per_action)


class LaneASearchWorkspaceAction(StrictModel):
    action: Literal['search_workspace']
    needle: str = Field(min_length=1)
    paths: tuple[str, ...] = Field(
        min_length=1,
        max_length=LANE_A_GUEST_HARNESS_POLICY.maximum_search_paths_per_action,
    )
    max_results: int = Field(ge=1, le=LANE_A_GUEST_HARNESS_POLICY.maximum_search_results_per_action)

    @field_validator('needle')
    @classmethod
    def validate_needle_bytes(cls, value: str) -> str:
        if len(value.encode('utf-8')) > LANE_A_GUEST_HARNESS_POLICY.maximum_search_needle_bytes:
            raise ValueError('search needle exceeds the fixed UTF-8 byte budget')
        return value

    @field_validator('paths')
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError('search paths must be unique and sorted')
        return value


class LaneASubmitAction(StrictModel):
    action: Literal['submit']
    submission: ExecutionSubmission


type LaneAGuestAction = Annotated[
    LaneAListWorkspaceAction | LaneAReadWorkspaceAction | LaneASearchWorkspaceAction | LaneASubmitAction,
    Field(discriminator='action'),
]

_ACTION_ADAPTER = TypeAdapter(LaneAGuestAction)
LANE_A_GUEST_ACTION_JSON_SCHEMA = _ACTION_ADAPTER.json_schema()
LANE_A_GUEST_ACTION_SCHEMA_SHA256 = hashlib.sha256(canonical_json_bytes(LANE_A_GUEST_ACTION_JSON_SCHEMA)).hexdigest()

LANE_A_GUEST_SYSTEM_PROMPT = (
    'You control a bounded historical clinical-trial research harness. '
    'Choose exactly one next action from the JSON schema below. Return only one JSON object; '
    'do not add prose, Markdown, explanations, or rationale. Use only broker observations in the '
    'current state. The final action must contain the complete task-bound forecast submission. '
    'Malformed output terminates the attempt and is never repaired or retried.\n'
    f'Policy: {LANE_A_GUEST_HARNESS_POLICY_ID}\n'
    f'Action schema SHA-256: {LANE_A_GUEST_ACTION_SCHEMA_SHA256}\n'
    'Action JSON Schema:\n' + canonical_json_bytes(LANE_A_GUEST_ACTION_JSON_SCHEMA).decode('utf-8')
)


class LaneAGuestRpcClient(Protocol):
    """The complete effectful surface available to this harness."""

    def list_workspace(self, *, cursor: int = 0, limit: int = 100) -> ListWorkspaceResult: ...

    def read_workspace(self, path: str, *, offset: int = 0, limit: int) -> ReadWorkspaceResult: ...

    def search_workspace(
        self,
        needle: str,
        *,
        paths: tuple[str, ...] | None = None,
        max_results: int = 100,
    ) -> SearchWorkspaceResult: ...

    def model_generate(
        self,
        *,
        messages: tuple[AgenticModelMessage, ...],
        max_output_tokens: int,
        response_schema_sha256: str | None = None,
    ) -> AgenticModelResponse: ...

    def submit(self, submission: ExecutionSubmission) -> SubmitResult: ...


class LaneAGuestHarnessResult(StrictModel):
    schema_version: Literal['vaxreplay.lane-a-guest-harness-result.dev-v0.1'] = (
        LANE_A_GUEST_HARNESS_RESULT_SCHEMA_VERSION
    )
    policy_id: Literal['lane-a-benchmark-native-retrieval-agent-v0.1'] = LANE_A_GUEST_HARNESS_POLICY_ID
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    action_trace_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_call_count: int = Field(ge=4, le=LANE_A_GUEST_HARNESS_POLICY.maximum_model_calls)
    workspace_action_count: int = Field(ge=3)
    list_action_count: int = Field(ge=1)
    read_action_count: int = Field(ge=1)
    search_action_count: int = Field(ge=1)
    total_read_bytes: int = Field(ge=1, le=LANE_A_GUEST_HARNESS_POLICY.maximum_total_read_bytes)
    targeted_source_read_count: int = Field(ge=1)
    terminal_submission_accepted: Literal[True] = True
    automatic_repair_attempt_count: Literal[0] = 0
    rationale_collected: Literal[False] = False
    submission: ExecutionSubmission

    @model_validator(mode='after')
    def validate_submission_hash(self) -> Self:
        if self.submission_sha256 != hashlib.sha256(canonical_json_bytes(self.submission)).hexdigest():
            raise ValueError('harness result submission hash does not bind its exact submission')
        if self.workspace_action_count != self.list_action_count + self.read_action_count + self.search_action_count:
            raise ValueError('workspace action count does not equal the typed action counts')
        return self


@dataclass
class _HarnessState:
    observations: list[dict[str, object]] = field(default_factory=list)
    action_trace: list[dict[str, object]] = field(default_factory=list)
    discovered: dict[str, LogicalFileResult] = field(default_factory=dict)
    search_hits: list[tuple[str, int, int]] = field(default_factory=list)
    observation_bytes: int = 0
    total_read_bytes: int = 0
    list_action_count: int = 0
    read_action_count: int = 0
    search_action_count: int = 0
    targeted_source_read_count: int = 0
    next_list_cursor: int | None = 0

    @property
    def listed(self) -> bool:
        return self.list_action_count > 0


def run_lane_a_guest_harness(
    client: LaneAGuestRpcClient,
    *,
    task_invocation: AgenticTaskInvocation,
) -> LaneAGuestHarnessResult:
    """Run one fixed-policy, model-controlled Lane A research loop.

    Every workspace operation, inference request, and terminal submission goes through ``client``.
    The supplied invocation is organizer-owned task-binding metadata, not an alternate evidence
    source; cutoff documents are deliberately omitted from the model prompt.
    """

    invocation = _validated_lane_a_invocation(task_invocation)
    state = _HarnessState()
    policy = LANE_A_GUEST_HARNESS_POLICY

    for step_number in range(1, policy.maximum_steps + 1):
        response = _model_action(client, invocation=invocation, state=state, step_number=step_number)
        action = _parse_action(response)
        state.action_trace.append(_action_trace_entry(step_number, action))

        if isinstance(action, LaneASubmitAction):
            return _submit(
                client,
                invocation=invocation,
                state=state,
                action=action,
                model_call_count=step_number,
            )
        if step_number == policy.maximum_steps:
            raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.STEP_BUDGET_EXHAUSTED)
        _execute_workspace_action(client, state=state, action=action, step_number=step_number)

    raise AssertionError('fixed Lane A harness loop became nonexhaustive')


def _validated_lane_a_invocation(invocation: AgenticTaskInvocation) -> AgenticTaskInvocation:
    try:
        validated = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(invocation))
    except (TypeError, ValueError):
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.INVALID_TASK_INVOCATION) from None
    if validated.response_protocol != AgenticResponseProtocol.CLINICAL_EXECUTION or not isinstance(
        validated.task, ExecutionTask
    ):
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.INVALID_TASK_INVOCATION)
    return validated


def _model_action(
    client: LaneAGuestRpcClient,
    *,
    invocation: AgenticTaskInvocation,
    state: _HarnessState,
    step_number: int,
) -> AgenticModelResponse:
    messages = (
        AgenticModelMessage(role='system', content=LANE_A_GUEST_SYSTEM_PROMPT),
        AgenticModelMessage(
            role='user',
            content=canonical_json_bytes(_model_state(invocation, state, step_number)).decode('utf-8'),
        ),
    )
    try:
        # The production OpenAI adapter currently rejects provider-native response schemas.  The
        # committed schema is embedded in the fixed prompt and enforced locally without repair.
        return client.model_generate(
            messages=messages,
            max_output_tokens=LANE_A_GUEST_HARNESS_POLICY.maximum_model_output_tokens_per_call,
            response_schema_sha256=None,
        )
    except Exception:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.MODEL_RPC_REJECTED) from None


def _model_state(
    invocation: AgenticTaskInvocation,
    state: _HarnessState,
    step_number: int,
) -> dict[str, object]:
    task = invocation.task
    assert isinstance(task, ExecutionTask)
    context = task.context
    return {
        'schema_version': _STATE_SCHEMA_VERSION,
        'policy_id': LANE_A_GUEST_HARNESS_POLICY_ID,
        'step_number': step_number,
        'remaining_model_calls_including_this_one': (LANE_A_GUEST_HARNESS_POLICY.maximum_model_calls - step_number + 1),
        'task_binding': {
            'episode_id': context.episode_id,
            'target_trial_id': context.target_trial_id,
            'task_context_sha256': task.context_sha256,
            'anchor_date': context.anchor_date.isoformat(),
            'planned_enrollment': context.planned_enrollment,
            'planned_primary_completion_date': context.planned_primary_completion_date.isoformat(),
            'enrollment_ratio_spec': context.enrollment_ratio_spec.model_dump(mode='json'),
            'primary_completion_slippage_days_spec': (
                context.primary_completion_slippage_days_spec.model_dump(mode='json')
            ),
            'fact_questions': tuple(question.model_dump(mode='json') for question in context.fact_questions),
        },
        'submission_gate': {
            'minimum_model_calls': LANE_A_GUEST_HARNESS_POLICY.minimum_model_calls_before_submission,
            'list_observed': state.listed,
            'source_search_with_hit_observed': bool(state.search_hits),
            'targeted_source_read_observed': state.targeted_source_read_count > 0,
        },
        'budgets': LANE_A_GUEST_HARNESS_POLICY.model_dump(mode='json'),
        'discovered_paths': tuple(sorted(state.discovered)),
        'observation_bytes_remaining': (
            LANE_A_GUEST_HARNESS_POLICY.maximum_observation_bytes - state.observation_bytes
        ),
        'read_bytes_remaining': LANE_A_GUEST_HARNESS_POLICY.maximum_total_read_bytes - state.total_read_bytes,
        'observations': tuple(state.observations),
    }


def _parse_action(response: AgenticModelResponse) -> LaneAGuestAction:
    if response.stop_reason != 'completed':
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.MODEL_RESPONSE_INCOMPLETE)
    raw = response.content.encode('utf-8')
    if len(raw) > LANE_A_GUEST_HARNESS_POLICY.maximum_action_response_bytes:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.MODEL_ACTION_OVERSIZED)
    try:
        payload = json.loads(response.content, object_pairs_hook=_unique_json_object, parse_constant=_reject_constant)
        if not isinstance(payload, dict):
            raise ValueError('action must be an object')
        # Parse once with the duplicate-key hook, then validate the exact JSON value rather than
        # coercing Python containers.  In particular, JSON arrays remain valid inputs for tuple
        # fields while strings/numbers cannot be coerced into stricter schema types.
        return _ACTION_ADAPTER.validate_json(canonical_json_bytes(payload), strict=True)
    except (TypeError, ValueError, ValidationError):
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.MALFORMED_MODEL_ACTION) from None


def _execute_workspace_action(
    client: LaneAGuestRpcClient,
    *,
    state: _HarnessState,
    action: LaneAListWorkspaceAction | LaneAReadWorkspaceAction | LaneASearchWorkspaceAction,
    step_number: int,
) -> None:
    if not state.listed and not isinstance(action, LaneAListWorkspaceAction):
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.ACTION_ORDER_VIOLATION)
    if isinstance(action, LaneAListWorkspaceAction):
        _list_workspace(client, state=state, action=action, step_number=step_number)
    elif isinstance(action, LaneAReadWorkspaceAction):
        _read_workspace(client, state=state, action=action, step_number=step_number)
    else:
        _search_workspace(client, state=state, action=action, step_number=step_number)


def _list_workspace(
    client: LaneAGuestRpcClient,
    *,
    state: _HarnessState,
    action: LaneAListWorkspaceAction,
    step_number: int,
) -> None:
    if state.next_list_cursor is None or action.cursor != state.next_list_cursor:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.ACTION_ORDER_VIOLATION)
    try:
        result = client.list_workspace(cursor=action.cursor, limit=action.limit)
    except Exception:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RPC_REJECTED) from None
    if len(result.files) > action.limit:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RESPONSE_INVALID)
    for item in result.files:
        previous = state.discovered.get(item.path)
        if previous is not None and previous != item:
            raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RESPONSE_INVALID)
        state.discovered[item.path] = item
    state.list_action_count += 1
    state.next_list_cursor = result.next_cursor
    _append_observation(
        state,
        {
            'step': step_number,
            'action': action.action,
            'cursor': action.cursor,
            'files': tuple(item.model_dump(mode='json') for item in result.files),
            'next_cursor': result.next_cursor,
        },
    )


def _read_workspace(
    client: LaneAGuestRpcClient,
    *,
    state: _HarnessState,
    action: LaneAReadWorkspaceAction,
    step_number: int,
) -> None:
    metadata = state.discovered.get(action.path)
    if metadata is None:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.ACTION_ORDER_VIOLATION)
    if state.total_read_bytes + action.limit > LANE_A_GUEST_HARNESS_POLICY.maximum_total_read_bytes:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.ACTION_BUDGET_EXCEEDED)
    try:
        result = client.read_workspace(action.path, offset=action.offset, limit=action.limit)
    except Exception:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RPC_REJECTED) from None
    if not _valid_read_result(result, action=action, metadata=metadata):
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RESPONSE_INVALID)
    try:
        text = result.content.decode('utf-8')
    except UnicodeDecodeError:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RESPONSE_INVALID) from None
    state.total_read_bytes += result.byte_count
    state.read_action_count += 1
    end = action.offset + result.byte_count
    targeted = (
        action.path.startswith('sources/')
        and result.byte_count > 0
        and any(
            path == action.path and action.offset < hit_end and end > hit_start
            for path, hit_start, hit_end in state.search_hits
        )
    )
    if targeted:
        state.targeted_source_read_count += 1
    _append_observation(
        state,
        {
            'step': step_number,
            'action': action.action,
            'path': action.path,
            'offset': result.offset,
            'byte_count': result.byte_count,
            'eof': result.eof,
            'targeted_search_hit_read': targeted,
            'content': text,
        },
    )


def _search_workspace(
    client: LaneAGuestRpcClient,
    *,
    state: _HarnessState,
    action: LaneASearchWorkspaceAction,
    step_number: int,
) -> None:
    if any(path not in state.discovered or not path.startswith('sources/') for path in action.paths):
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.ACTION_ORDER_VIOLATION)
    try:
        result = client.search_workspace(
            action.needle,
            paths=action.paths,
            max_results=action.max_results,
        )
    except Exception:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RPC_REJECTED) from None
    if len(result.hits) > action.max_results:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RESPONSE_INVALID)
    for hit in result.hits:
        metadata = state.discovered.get(hit.path)
        if (
            metadata is None
            or hit.path not in action.paths
            or hit.start_byte < 0
            or hit.end_byte <= hit.start_byte
            or hit.end_byte > metadata.byte_count
        ):
            raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.BROKER_RESPONSE_INVALID)
        state.search_hits.append((hit.path, hit.start_byte, hit.end_byte))
    state.search_action_count += 1
    _append_observation(
        state,
        {
            'step': step_number,
            'action': action.action,
            'needle': action.needle,
            'paths': action.paths,
            'hits': tuple(hit.model_dump(mode='json') for hit in result.hits),
        },
    )


def _submit(
    client: LaneAGuestRpcClient,
    *,
    invocation: AgenticTaskInvocation,
    state: _HarnessState,
    action: LaneASubmitAction,
    model_call_count: int,
) -> LaneAGuestHarnessResult:
    policy = LANE_A_GUEST_HARNESS_POLICY
    if (
        model_call_count < policy.minimum_model_calls_before_submission
        or not state.listed
        or not state.search_hits
        or state.targeted_source_read_count == 0
    ):
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.SUBMISSION_TOO_EARLY)
    try:
        validate_submission_for_invocation(invocation, action.submission)
    except ValueError:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.SUBMISSION_BINDING_INVALID) from None
    submission_bytes = canonical_json_bytes(action.submission)
    submission_sha256 = hashlib.sha256(submission_bytes).hexdigest()
    try:
        receipt = client.submit(action.submission)
    except Exception:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.SUBMISSION_RPC_REJECTED) from None
    if (receipt.submission_sha256, receipt.submission_bytes) != (submission_sha256, len(submission_bytes)):
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.SUBMISSION_RECEIPT_INVALID)
    return LaneAGuestHarnessResult(
        task_invocation_sha256=agentic_task_invocation_sha256(invocation),
        submission_sha256=submission_sha256,
        action_trace_sha256=hashlib.sha256(canonical_json_bytes(state.action_trace)).hexdigest(),
        model_call_count=model_call_count,
        workspace_action_count=state.list_action_count + state.read_action_count + state.search_action_count,
        list_action_count=state.list_action_count,
        read_action_count=state.read_action_count,
        search_action_count=state.search_action_count,
        total_read_bytes=state.total_read_bytes,
        targeted_source_read_count=state.targeted_source_read_count,
        submission=action.submission,
    )


def _valid_read_result(
    result: ReadWorkspaceResult,
    *,
    action: LaneAReadWorkspaceAction,
    metadata: LogicalFileResult,
) -> bool:
    end = action.offset + result.byte_count
    return (
        result.offset == action.offset
        and result.byte_count <= action.limit
        and end <= metadata.byte_count
        and result.eof == (end >= metadata.byte_count)
    )


def _append_observation(state: _HarnessState, observation: dict[str, object]) -> None:
    byte_count = len(canonical_json_bytes(observation))
    if state.observation_bytes + byte_count > LANE_A_GUEST_HARNESS_POLICY.maximum_observation_bytes:
        raise LaneAGuestHarnessError(LaneAGuestHarnessFailureCode.OBSERVATION_BUDGET_EXCEEDED)
    state.observations.append(observation)
    state.observation_bytes += byte_count


def _action_trace_entry(step_number: int, action: LaneAGuestAction) -> dict[str, object]:
    action_bytes = canonical_json_bytes(action)
    return {
        'step': step_number,
        'action': action.action,
        'action_sha256': hashlib.sha256(action_bytes).hexdigest(),
        'action_bytes': len(action_bytes),
    }


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON object key')
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f'non-finite JSON constant is forbidden: {value}')


__all__ = [
    'LANE_A_GUEST_ACTION_JSON_SCHEMA',
    'LANE_A_GUEST_ACTION_SCHEMA_SHA256',
    'LANE_A_GUEST_HARNESS_POLICY',
    'LANE_A_GUEST_HARNESS_POLICY_ID',
    'LANE_A_GUEST_SYSTEM_PROMPT',
    'LaneAGuestAction',
    'LaneAGuestHarnessError',
    'LaneAGuestHarnessFailureCode',
    'LaneAGuestHarnessPolicy',
    'LaneAGuestHarnessResult',
    'LaneAGuestRpcClient',
    'LaneAListWorkspaceAction',
    'LaneAReadWorkspaceAction',
    'LaneASearchWorkspaceAction',
    'LaneASubmitAction',
    'run_lane_a_guest_harness',
]
