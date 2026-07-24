"""Outcome-blind research-process reward for Lane A clinical-execution tasks.

The scorer consumes a host-authenticated projection of broker/tool events.  It never receives
private gold, later registry observations, model responses, hidden reasoning, or chain-of-thought.
The process score is diagnostic and remains separate from the proper outcome reward.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import math
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_execution_bridge import ClinicalAgenticWorkspaceManifest
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcMethod,
    ReadWorkspaceRequest,
    ReadWorkspaceResult,
    SearchWorkspaceRequest,
    SearchWorkspaceResult,
    verify_authenticated_guest_rpc_session,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.task_protocol import agentic_task_invocation_sha256
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_task import ExecutionTask

LANE_A_PROCESS_POLICY_SCHEMA_VERSION = 'vaxreplay.lane-a-process-policy.dev-v0.1'
LANE_A_PROCESS_EVIDENCE_SCHEMA_VERSION = 'vaxreplay.lane-a-process-evidence.dev-v0.1'
AUTHENTICATED_LANE_A_PROCESS_EVIDENCE_SCHEMA_VERSION = 'vaxreplay.authenticated-lane-a-process-evidence.dev-v0.1'
LANE_A_PROCESS_SCORE_SCHEMA_VERSION = 'vaxreplay.lane-a-process-score.dev-v0.1'
LANE_A_PROCESS_POLICY_ID = 'lane-a-authenticated-research-process-v0.1'

SOURCE_COVERAGE_WEIGHT = 0.30
TARGETED_RETRIEVAL_WEIGHT = 0.20
EVIDENCE_REFERENCE_WEIGHT = 0.20
FORECAST_BOOKKEEPING_WEIGHT = 0.20
REPRODUCIBLE_COMPUTATION_WEIGHT = 0.10
MAXIMUM_CREDITABLE_PROCESS_REWARD = SOURCE_COVERAGE_WEIGHT + TARGETED_RETRIEVAL_WEIGHT + FORECAST_BOOKKEEPING_WEIGHT

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_TRACE_KEY_ID_DOMAIN = b'vaxreplay.lane-a-process-trace-key-id.v0.1\x00'
_TRACE_HMAC_DOMAIN = b'vaxreplay.lane-a-process-trace.dev-v0.1\x00'


class LaneAProcessRewardError(ValueError):
    """Authenticated process evidence is incomplete, forged, or incompatible."""


class LaneAProcessEventKind(str, enum.Enum):
    LIST = 'list_workspace'
    READ = 'read_workspace'
    SEARCH = 'search_workspace'
    MODEL = 'model_generate'
    COMPUTE = 'local_compute'
    SUBMIT = 'submit'
    RPC_REJECTION = 'rpc_rejection'


class LaneAProcessTerminalStatus(str, enum.Enum):
    COMPLETED = 'completed'
    FAILED = 'failed'
    ABORTED = 'aborted'


class LaneAProcessRewardPolicy(StrictModel):
    schema_version: Literal['vaxreplay.lane-a-process-policy.dev-v0.1'] = LANE_A_PROCESS_POLICY_SCHEMA_VERSION
    policy_id: Literal['lane-a-authenticated-research-process-v0.1'] = LANE_A_PROCESS_POLICY_ID
    source_coverage_weight: float = SOURCE_COVERAGE_WEIGHT
    targeted_retrieval_weight: float = TARGETED_RETRIEVAL_WEIGHT
    evidence_reference_weight: float = EVIDENCE_REFERENCE_WEIGHT
    forecast_bookkeeping_weight: float = FORECAST_BOOKKEEPING_WEIGHT
    reproducible_computation_weight: float = REPRODUCIBLE_COMPUTATION_WEIGHT
    overlapping_reads_count_once: Literal[True] = True
    duplicate_searches_do_not_increase_credit: Literal[True] = True
    failed_or_spam_searches_remain_in_denominator: Literal[True] = True
    zero_byte_post_search_reads_do_not_increase_credit: Literal[True] = True
    successful_model_call_between_evidence_and_submit_required: Literal[True] = True
    evidence_references_require_prior_authenticated_read: Literal[True] = True
    computation_requires_deterministic_hashed_recipe: Literal[True] = True
    authenticated_guest_rpc_projection_required: Literal[True] = True
    evidence_reference_credit_enabled: Literal[False] = False
    reproducible_computation_credit_enabled: Literal[False] = False
    unbound_or_guest_local_computation_receives_credit: Literal[False] = False
    maximum_creditable_process_reward: float = Field(
        default=MAXIMUM_CREDITABLE_PROCESS_REWARD,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    reward_role: Literal['auxiliary_diagnostic_only'] = 'auxiliary_diagnostic_only'
    eligible_as_primary_outcome_reward: Literal[False] = False
    chain_of_thought_read: Literal[False] = False
    model_response_content_read: Literal[False] = False
    later_outcomes_read: Literal[False] = False
    private_gold_read: Literal[False] = False
    combined_with_outcome_reward: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_weights(self) -> Self:
        weights = (
            self.source_coverage_weight,
            self.targeted_retrieval_weight,
            self.evidence_reference_weight,
            self.forecast_bookkeeping_weight,
            self.reproducible_computation_weight,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in weights):
            raise ValueError('process reward weights must be finite and nonnegative')
        if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('process reward weights must sum to one')
        if not math.isclose(
            self.maximum_creditable_process_reward,
            MAXIMUM_CREDITABLE_PROCESS_REWARD,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('process reward ceiling must match the currently authenticated components')
        return self


LANE_A_PROCESS_REWARD_POLICY = LaneAProcessRewardPolicy()


def lane_a_process_reward_policy_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(LANE_A_PROCESS_REWARD_POLICY)).hexdigest()


def _normalized_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or not path.parts or '..' in path.parts:
        raise ValueError('process trace paths must be normalized relative POSIX paths')
    return value


class LaneAProcessSource(StrictModel):
    logical_path: str
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)

    @field_validator('logical_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        _normalized_path(value)
        if not value.startswith('sources/'):
            raise ValueError('process reward sources must remain inside sources/')
        return value


class LaneAProcessEvent(StrictModel):
    """Content-free projection of one exact host-observed tool event."""

    event_index: int = Field(ge=0)
    kind: LaneAProcessEventKind
    succeeded: bool
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_path: str | None = None
    offset: int | None = Field(default=None, ge=0)
    byte_count: int | None = Field(default=None, ge=0)
    query_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    restricted_paths: tuple[str, ...] = ()
    hit_paths: tuple[str, ...] = ()
    referenced_source_paths: tuple[str, ...] = ()
    computation_recipe_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    computation_executable_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    computation_input_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    computation_output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    deterministic_computation: bool | None = None
    computation_network_used: bool | None = None
    submission_contract_valid: bool | None = None

    @field_validator('logical_path')
    @classmethod
    def validate_optional_path(cls, value: str | None) -> str | None:
        if value is not None:
            _normalized_path(value)
        return value

    @field_validator('restricted_paths', 'hit_paths', 'referenced_source_paths')
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            _normalized_path(item)
        if value != tuple(sorted(set(value))):
            raise ValueError('event path collections must be unique and sorted')
        return value

    @model_validator(mode='after')
    def validate_kind_fields(self) -> Self:
        read_fields = (self.logical_path, self.offset, self.byte_count)
        search_fields = (self.query_sha256, self.restricted_paths, self.hit_paths)
        compute_fields = (
            self.computation_recipe_sha256,
            self.computation_executable_sha256,
            self.computation_input_sha256,
            self.computation_output_sha256,
            self.deterministic_computation,
            self.computation_network_used,
        )
        if self.kind == LaneAProcessEventKind.READ:
            if any(value is None for value in read_fields):
                raise ValueError('read events require path, offset, and byte count')
        elif any(value is not None for value in read_fields):
            raise ValueError('only read events may carry byte ranges')
        if self.kind == LaneAProcessEventKind.SEARCH:
            if self.query_sha256 is None:
                raise ValueError('search events require a hashed query')
        elif any(search_fields):
            raise ValueError('only search events may carry query and hit metadata')
        if self.kind == LaneAProcessEventKind.MODEL:
            pass
        elif self.referenced_source_paths:
            raise ValueError('only model events may carry evidence references')
        if self.kind == LaneAProcessEventKind.COMPUTE:
            if any(value is None for value in compute_fields):
                raise ValueError('compute events require exact deterministic computation bindings')
            if not self.deterministic_computation or self.computation_network_used:
                raise ValueError('creditable compute events must be deterministic and network-free')
        elif any(value is not None for value in compute_fields):
            raise ValueError('only compute events may carry computation bindings')
        if self.kind == LaneAProcessEventKind.SUBMIT:
            if self.submission_contract_valid is None:
                raise ValueError('submit events require their public-contract validation result')
        elif self.submission_contract_valid is not None:
            raise ValueError('only submit events may carry submission validation')
        return self


class LaneAProcessEvidence(StrictModel):
    schema_version: Literal['vaxreplay.lane-a-process-evidence.dev-v0.1'] = LANE_A_PROCESS_EVIDENCE_SCHEMA_VERSION
    projection_source: Literal['authenticated_guest_rpc_session.v0.2'] = 'authenticated_guest_rpc_session.v0.2'
    episode_id: str = Field(min_length=1)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    broker_attestation_sha256: str = Field(pattern=_SHA256_PATTERN)
    authenticated_source_trace_sha256: str = Field(pattern=_SHA256_PATTERN)
    sources: tuple[LaneAProcessSource, ...] = Field(min_length=1)
    events: tuple[LaneAProcessEvent, ...]
    terminal_status: LaneAProcessTerminalStatus
    trace_event_count: int = Field(ge=0)
    complete_authenticated_rpc_attempt_projection: Literal[True] = True
    complete_guest_tool_trace_claimed: Literal[False] = False
    broker_trace_authoritative: Literal[True] = True
    workspace_and_gateway_rpc_trace_authoritative: Literal[True] = True
    guest_local_tool_trace_authoritative: Literal[False] = False
    evidence_reference_metadata_authoritative: Literal[False] = False
    guest_local_computation_authoritative: Literal[False] = False
    source_inventory_fixed_before_run: Literal[True] = True
    hidden_reasoning_omitted: Literal[True] = True
    model_response_content_omitted: Literal[True] = True
    later_outcomes_omitted: Literal[True] = True
    private_gold_omitted: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_trace(self) -> Self:
        paths = tuple(item.logical_path for item in self.sources)
        if paths != tuple(sorted(set(paths))):
            raise ValueError('process sources must have unique canonical path order')
        if self.trace_event_count != len(self.events):
            raise ValueError('process trace event count is inconsistent')
        if tuple(item.event_index for item in self.events) != tuple(range(len(self.events))):
            raise ValueError('process events must have contiguous indexes')
        if any(item.referenced_source_paths for item in self.events):
            raise ValueError('guest RPC v0.2 does not authenticate evidence-reference metadata')
        if any(item.kind == LaneAProcessEventKind.COMPUTE for item in self.events):
            raise ValueError('guest RPC v0.2 does not authenticate guest-local computation')
        return self


class AuthenticatedLaneAProcessEvidence(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-lane-a-process-evidence.dev-v0.1'] = (
        AUTHENTICATED_LANE_A_PROCESS_EVIDENCE_SCHEMA_VERSION
    )
    evidence: LaneAProcessEvidence
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    authentication: Literal['hmac-sha256-domain-separated'] = 'hmac-sha256-domain-separated'
    trace_key_id: str = Field(pattern=_SHA256_PATTERN)
    evidence_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_hash(self) -> Self:
        if hashlib.sha256(canonical_json_bytes(self.evidence)).hexdigest() != self.evidence_sha256:
            raise ValueError('authenticated process evidence does not bind its exact trace')
        return self


class LaneAProcessComponentScore(StrictModel):
    component: Literal[
        'source_coverage',
        'targeted_retrieval',
        'evidence_references',
        'forecast_bookkeeping',
        'reproducible_computation',
    ]
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    credited_items: int = Field(ge=0)
    denominator_items: int = Field(ge=0)


class LaneAProcessScore(StrictModel):
    schema_version: Literal['vaxreplay.lane-a-process-score.dev-v0.1'] = LANE_A_PROCESS_SCORE_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    policy_id: Literal['lane-a-authenticated-research-process-v0.1'] = LANE_A_PROCESS_POLICY_ID
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    process_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    maximum_creditable_process_reward: float = Field(
        default=MAXIMUM_CREDITABLE_PROCESS_REWARD,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    components: tuple[LaneAProcessComponentScore, ...] = Field(min_length=5, max_length=5)
    authenticated_trace_verified: Literal[True] = True
    complete_authenticated_rpc_attempt_projection_used: Literal[True] = True
    complete_guest_tool_trace_claimed: Literal[False] = False
    successful_model_call_observed: bool
    research_process_credit_eligible: bool
    evidence_reference_credit_enabled: Literal[False] = False
    reproducible_computation_credit_enabled: Literal[False] = False
    reward_role: Literal['auxiliary_diagnostic_only'] = 'auxiliary_diagnostic_only'
    eligible_as_primary_outcome_reward: Literal[False] = False
    chain_of_thought_used: Literal[False] = False
    model_response_content_used: Literal[False] = False
    later_outcome_used: Literal[False] = False
    private_gold_used: Literal[False] = False
    outcome_reward_included: Literal[False] = False
    combined_reward: None = None
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_total(self) -> Self:
        expected_order = (
            'source_coverage',
            'targeted_retrieval',
            'evidence_references',
            'forecast_bookkeeping',
            'reproducible_computation',
        )
        if tuple(item.component for item in self.components) != expected_order:
            raise ValueError('process score components must use fixed canonical order')
        expected = math.fsum(item.score * item.weight for item in self.components)
        if not math.isclose(self.process_reward, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('process reward does not match its fixed weighted components')
        disabled = {item.component: item for item in self.components}
        if disabled['evidence_references'].score != 0.0 or disabled['reproducible_computation'].score != 0.0:
            raise ValueError('unsupported process components cannot receive credit')
        if not math.isclose(
            self.maximum_creditable_process_reward,
            MAXIMUM_CREDITABLE_PROCESS_REWARD,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('process score ceiling differs from the fixed policy')
        if self.process_reward > self.maximum_creditable_process_reward:
            raise ValueError('process reward exceeds the authenticated guest-RPC ceiling')
        if self.research_process_credit_eligible and not self.successful_model_call_observed:
            raise ValueError('research-process credit cannot be eligible without a successful model call')
        if not self.research_process_credit_eligible and self.process_reward != 0.0:
            raise ValueError('ineligible traces cannot receive research-process credit')
        return self


def lane_a_process_trace_key_id(key: bytes) -> str:
    if len(key) < 32:
        raise LaneAProcessRewardError('process trace HMAC key must contain at least 32 bytes')
    return hashlib.sha256(_TRACE_KEY_ID_DOMAIN + key).hexdigest()


def authenticate_lane_a_process_evidence(
    evidence: LaneAProcessEvidence,
    *,
    key: bytes,
    expected_key_id: str,
) -> AuthenticatedLaneAProcessEvidence:
    evidence = LaneAProcessEvidence.model_validate_json(canonical_json_bytes(evidence))
    if lane_a_process_trace_key_id(key) != expected_key_id:
        raise LaneAProcessRewardError('process trace key does not match its expected key ID')
    payload = canonical_json_bytes(evidence)
    return AuthenticatedLaneAProcessEvidence(
        evidence=evidence,
        evidence_sha256=hashlib.sha256(payload).hexdigest(),
        trace_key_id=expected_key_id,
        evidence_hmac_sha256=hmac.new(key, _TRACE_HMAC_DOMAIN + payload, hashlib.sha256).hexdigest(),
    )


def _event_from_authenticated_rpc_attempt(attempt) -> LaneAProcessEvent:
    common = {
        'event_index': attempt.attempt_index,
        'succeeded': attempt.response.succeeded,
        'request_sha256': attempt.request_sha256,
        'response_sha256': attempt.response_sha256,
    }
    if not attempt.sequence_accepted:
        return LaneAProcessEvent(kind=LaneAProcessEventKind.RPC_REJECTION, **common)
    try:
        method = GuestRpcMethod(attempt.request.method)
    except ValueError:
        return LaneAProcessEvent(kind=LaneAProcessEventKind.RPC_REJECTION, **common)
    if method == GuestRpcMethod.LIST:
        return LaneAProcessEvent(kind=LaneAProcessEventKind.LIST, **common)
    if method == GuestRpcMethod.READ:
        try:
            request = ReadWorkspaceRequest.model_validate_json(canonical_json_bytes(attempt.request.body))
            result = (
                ReadWorkspaceResult.model_validate_json(canonical_json_bytes(attempt.response.result))
                if attempt.response.succeeded
                else None
            )
        except ValueError:
            return LaneAProcessEvent(kind=LaneAProcessEventKind.RPC_REJECTION, **common)
        return LaneAProcessEvent(
            kind=LaneAProcessEventKind.READ,
            logical_path=request.path,
            offset=result.offset if result is not None else request.offset,
            byte_count=result.byte_count if result is not None else 0,
            **common,
        )
    if method == GuestRpcMethod.SEARCH:
        try:
            request = SearchWorkspaceRequest.model_validate_json(canonical_json_bytes(attempt.request.body))
            result = (
                SearchWorkspaceResult.model_validate_json(canonical_json_bytes(attempt.response.result))
                if attempt.response.succeeded
                else None
            )
        except ValueError:
            return LaneAProcessEvent(kind=LaneAProcessEventKind.RPC_REJECTION, **common)
        restricted = () if request.paths is None else tuple(sorted(set(request.paths)))
        hits = () if result is None else tuple(sorted({item.path for item in result.hits}))
        return LaneAProcessEvent(
            kind=LaneAProcessEventKind.SEARCH,
            query_sha256=hashlib.sha256(request.needle.encode('utf-8')).hexdigest(),
            restricted_paths=restricted,
            hit_paths=hits,
            **common,
        )
    if method == GuestRpcMethod.MODEL_GENERATE:
        return LaneAProcessEvent(kind=LaneAProcessEventKind.MODEL, **common)
    if method == GuestRpcMethod.SUBMIT:
        return LaneAProcessEvent(
            kind=LaneAProcessEventKind.SUBMIT,
            submission_contract_valid=attempt.response.succeeded,
            **common,
        )
    return LaneAProcessEvent(kind=LaneAProcessEventKind.RPC_REJECTION, **common)


def project_and_authenticate_lane_a_process_evidence_from_guest_rpc(
    session: AuthenticatedGuestRpcSession,
    *,
    workspace_manifest: ClinicalAgenticWorkspaceManifest,
    guest_rpc_receipt_key: bytes,
    expected_guest_rpc_receipt_key_id: str,
    expected_run_id: str,
    expected_execution_policy_sha256: str,
    expected_peer_cid: int,
    expected_rpc_port: int,
    process_trace_key: bytes,
    expected_process_trace_key_id: str,
) -> AuthenticatedLaneAProcessEvidence:
    """Project a verified clinical guest-RPC session into content-free process evidence.

    The current RPC authenticates list/read/search/model/submit attempts, but not guest-local
    compute, scratch writes, or content-free evidence-reference metadata.  Those components are
    therefore absent and receive zero credit rather than trusting harness-supplied claims.
    """

    session = AuthenticatedGuestRpcSession.model_validate_json(canonical_json_bytes(session))
    manifest = ClinicalAgenticWorkspaceManifest.model_validate_json(canonical_json_bytes(workspace_manifest))
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    if manifest_sha256 != session.seal.workspace_manifest_sha256:
        raise LaneAProcessRewardError('clinical workspace manifest differs from the authenticated RPC session')
    task = session.task_invocation.task
    if not isinstance(task, ExecutionTask):
        raise LaneAProcessRewardError('Lane A process reward requires a clinical-execution task invocation')
    task = ExecutionTask.model_validate_json(canonical_json_bytes(task))
    if (
        manifest.episode_id,
        manifest.target_trial_id,
        manifest.task_sha256,
        manifest.workspace_tree_sha256,
        manifest.model_visible_surface_sha256,
        manifest.response_protocol,
    ) != (
        task.context.episode_id,
        task.context.target_trial_id,
        hashlib.sha256(canonical_json_bytes(task)).hexdigest(),
        session.seal.workspace_tree_sha256,
        session.seal.model_visible_surface_sha256,
        AgenticResponseProtocol.CLINICAL_EXECUTION,
    ):
        raise LaneAProcessRewardError('clinical workspace, task, and authenticated RPC bindings disagree')
    try:
        verify_authenticated_guest_rpc_session(
            session,
            receipt_key=guest_rpc_receipt_key,
            expected_receipt_key_id=expected_guest_rpc_receipt_key_id,
            expected_run_id=expected_run_id,
            expected_workspace_manifest_sha256=manifest_sha256,
            expected_execution_policy_sha256=expected_execution_policy_sha256,
            expected_task_invocation_sha256=agentic_task_invocation_sha256(session.task_invocation),
            expected_response_protocol=AgenticResponseProtocol.CLINICAL_EXECUTION,
            expected_peer_cid=expected_peer_cid,
            expected_rpc_port=expected_rpc_port,
        )
    except ValueError as error:
        raise LaneAProcessRewardError('guest RPC process source failed authentication') from error

    sources = tuple(
        LaneAProcessSource(logical_path=item.path, sha256=item.sha256, byte_count=item.byte_count)
        for item in manifest.entries
        if item.path.startswith('sources/')
    )
    if not sources:
        raise LaneAProcessRewardError('Lane A process reward requires at least one fixed workspace source')
    events = tuple(_event_from_authenticated_rpc_attempt(item) for item in session.attempts)
    evidence = LaneAProcessEvidence(
        episode_id=task.context.episode_id,
        task_context_sha256=task.context_sha256,
        workspace_manifest_sha256=manifest_sha256,
        workspace_tree_sha256=manifest.workspace_tree_sha256,
        model_visible_surface_sha256=manifest.model_visible_surface_sha256,
        broker_attestation_sha256=hashlib.sha256(canonical_json_bytes(session.seal)).hexdigest(),
        authenticated_source_trace_sha256=session.seal.attempt_log_sha256,
        sources=sources,
        events=events,
        terminal_status=LaneAProcessTerminalStatus(session.seal.terminal_status.value),
        trace_event_count=len(events),
    )
    return authenticate_lane_a_process_evidence(
        evidence,
        key=process_trace_key,
        expected_key_id=expected_process_trace_key_id,
    )


def _covered_bytes(intervals: list[tuple[int, int]], maximum: int) -> int:
    clipped = sorted((max(0, start), min(maximum, end)) for start, end in intervals if start < maximum and end > 0)
    total = 0
    cursor = 0
    for start, end in clipped:
        if end <= start:
            continue
        if start > cursor:
            total += end - start
            cursor = end
        elif end > cursor:
            total += end - cursor
            cursor = end
    return total


def score_authenticated_lane_a_process_evidence(
    authenticated: AuthenticatedLaneAProcessEvidence,
    *,
    key: bytes,
    expected_key_id: str,
    expected_workspace_manifest_sha256: str,
    expected_task_context_sha256: str,
) -> LaneAProcessScore:
    """Score only authenticated event structure; no outcome or generated content is accepted."""

    authenticated = AuthenticatedLaneAProcessEvidence.model_validate_json(canonical_json_bytes(authenticated))
    if lane_a_process_trace_key_id(key) != expected_key_id or authenticated.trace_key_id != expected_key_id:
        raise LaneAProcessRewardError('process trace key identity mismatch')
    payload = canonical_json_bytes(authenticated.evidence)
    expected_hmac = hmac.new(key, _TRACE_HMAC_DOMAIN + payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_hmac, authenticated.evidence_hmac_sha256):
        raise LaneAProcessRewardError('process trace authentication failed')
    evidence = authenticated.evidence
    if (
        evidence.workspace_manifest_sha256 != expected_workspace_manifest_sha256
        or evidence.task_context_sha256 != expected_task_context_sha256
    ):
        raise LaneAProcessRewardError('process evidence is bound to a different task or workspace')

    sources = {item.logical_path: item for item in evidence.sources}
    source_paths = set(sources)
    reads_by_path: dict[str, list[tuple[int, int, int]]] = {path: [] for path in source_paths}
    for event in evidence.events:
        if event.kind == LaneAProcessEventKind.READ and event.succeeded and event.logical_path in source_paths:
            if event.offset is None or event.byte_count is None:
                raise LaneAProcessRewardError('authenticated read event is missing its byte range')
            reads_by_path[event.logical_path].append((event.event_index, event.offset, event.offset + event.byte_count))
    covered = sum(
        _covered_bytes([(start, end) for _, start, end in reads_by_path[path]], source.byte_count)
        for path, source in sources.items()
    )
    total_source_bytes = sum(item.byte_count for item in evidence.sources)
    source_score = covered / total_source_bytes

    search_events = [item for item in evidence.events if item.kind == LaneAProcessEventKind.SEARCH]
    productive_queries: set[str] = set()
    for event in search_events:
        if not event.succeeded or event.query_sha256 is None:
            continue
        if not event.restricted_paths or not set(event.restricted_paths).issubset(source_paths):
            continue
        known_hits = set(event.hit_paths) & source_paths
        if any(
            any(
                read_index > event.event_index and read_end > read_start
                for read_index, read_start, read_end in reads_by_path[path]
            )
            for path in known_hits
        ):
            productive_queries.add(event.query_sha256)
    targeted_denominator = max(len(search_events), min(2, len(source_paths)))
    targeted_score = min(len(productive_queries) / targeted_denominator, 1.0)

    # Guest RPC v0.2 authenticates exact model-call hashes but does not expose a separate,
    # content-free citation-reference field.  Self-declared references would be gameable, so this
    # reserved component remains zero until the RPC protocol gains a trusted projection.
    referenced: set[str] = set()
    evidence_score = 0.0

    submit_events = [item for item in evidence.events if item.kind == LaneAProcessEventKind.SUBMIT]
    valid_submits = [item for item in submit_events if item.succeeded and item.submission_contract_valid is True]
    bookkeeping_score = float(
        evidence.terminal_status == LaneAProcessTerminalStatus.COMPLETED
        and len(submit_events) == 1
        and len(valid_submits) == 1
    )

    # A blind script can list/read/search files and emit a schema-valid answer without performing
    # any model-mediated research.  Treat those actions as observable diagnostics, but grant no
    # process reward unless an authenticated successful model call occurs after a nonempty source
    # read and before the one valid terminal submission.  This does not prove that the model used
    # the evidence; it closes the trivial no-model maximum and remains an auxiliary proxy.
    positive_read_indexes = {
        event.event_index
        for event in evidence.events
        if event.kind == LaneAProcessEventKind.READ
        and event.succeeded
        and event.logical_path in source_paths
        and (event.byte_count or 0) > 0
    }
    successful_model_indexes = {
        event.event_index for event in evidence.events if event.kind == LaneAProcessEventKind.MODEL and event.succeeded
    }
    valid_submit_indexes = {event.event_index for event in valid_submits}
    successful_model_call_observed = bool(successful_model_indexes)
    research_process_credit_eligible = any(
        read_index < model_index < submit_index
        for read_index in positive_read_indexes
        for model_index in successful_model_indexes
        for submit_index in valid_submit_indexes
    )
    if not research_process_credit_eligible:
        source_score = 0.0
        targeted_score = 0.0
        bookkeeping_score = 0.0

    # Local compute and scratch writes do not traverse guest RPC v0.2.  A deterministic hash alone
    # does not establish input provenance or downstream use, so arbitrary compute receives no
    # credit.  The fixed 0.10 policy weight is intentionally unavailable in this version.
    successful_recipes: set[str] = set()
    compute_denominator = 1
    compute_score = 0.0

    components = (
        LaneAProcessComponentScore(
            component='source_coverage',
            score=source_score,
            weight=SOURCE_COVERAGE_WEIGHT,
            credited_items=covered,
            denominator_items=total_source_bytes,
        ),
        LaneAProcessComponentScore(
            component='targeted_retrieval',
            score=targeted_score,
            weight=TARGETED_RETRIEVAL_WEIGHT,
            credited_items=len(productive_queries),
            denominator_items=targeted_denominator,
        ),
        LaneAProcessComponentScore(
            component='evidence_references',
            score=evidence_score,
            weight=EVIDENCE_REFERENCE_WEIGHT,
            credited_items=len(referenced),
            denominator_items=len(source_paths),
        ),
        LaneAProcessComponentScore(
            component='forecast_bookkeeping',
            score=bookkeeping_score,
            weight=FORECAST_BOOKKEEPING_WEIGHT,
            credited_items=len(valid_submits),
            denominator_items=1,
        ),
        LaneAProcessComponentScore(
            component='reproducible_computation',
            score=compute_score,
            weight=REPRODUCIBLE_COMPUTATION_WEIGHT,
            credited_items=len(successful_recipes),
            denominator_items=compute_denominator,
        ),
    )
    return LaneAProcessScore(
        episode_id=evidence.episode_id,
        task_context_sha256=evidence.task_context_sha256,
        policy_sha256=lane_a_process_reward_policy_sha256(),
        process_reward=math.fsum(item.score * item.weight for item in components),
        components=components,
        successful_model_call_observed=successful_model_call_observed,
        research_process_credit_eligible=research_process_credit_eligible,
    )


__all__ = [
    'AUTHENTICATED_LANE_A_PROCESS_EVIDENCE_SCHEMA_VERSION',
    'LANE_A_PROCESS_REWARD_POLICY',
    'MAXIMUM_CREDITABLE_PROCESS_REWARD',
    'AuthenticatedLaneAProcessEvidence',
    'LaneAProcessComponentScore',
    'LaneAProcessEvent',
    'LaneAProcessEventKind',
    'LaneAProcessEvidence',
    'LaneAProcessRewardError',
    'LaneAProcessRewardPolicy',
    'LaneAProcessScore',
    'LaneAProcessSource',
    'LaneAProcessTerminalStatus',
    'authenticate_lane_a_process_evidence',
    'lane_a_process_reward_policy_sha256',
    'lane_a_process_trace_key_id',
    'project_and_authenticate_lane_a_process_evidence_from_guest_rpc',
    'score_authenticated_lane_a_process_evidence',
]
