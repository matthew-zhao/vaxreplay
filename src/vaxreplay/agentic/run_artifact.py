"""Trusted finalization and verification of one Agentic Replay run artifact.

This module does not launch hostile code.  It is the one-way handoff used by a trusted supervisor
after execution: it rechecks workspace admission, gateway usage, tool/scratch inventories, the
final submission, and backend capabilities before authenticating an immutable run receipt.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.admission import AgenticWorkspaceAdmission, require_workspace_admission
from vaxreplay.agentic.gateway import AgenticGatewayTranscript
from vaxreplay.agentic.protocol import (
    AgenticExecutionPolicy,
    AgenticModelUsage,
    AgenticRunFailureCode,
    AgenticRunReceipt,
    AgenticTool,
    agentic_policy_sha256,
    agentic_receipt_key_id,
    agentic_run_receipt_hmac,
)
from vaxreplay.agentic.schema import normalized_relative_path
from vaxreplay.agentic.scoring import AgenticSubmissionV1
from vaxreplay.agentic.workspace import LoadedAgenticWorkspace, load_agentic_workspace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.runner.schema import BackendCapabilities, IsolationTier

AGENTIC_TOOL_EVENT_SCHEMA_VERSION = 'vaxreplay.agentic-tool-event.v0.1'
AGENTIC_SCRATCH_ENTRY_SCHEMA_VERSION = 'vaxreplay.agentic-scratch-entry.v0.1'
AGENTIC_HARNESS_IDENTITY_SCHEMA_VERSION = 'vaxreplay.agentic-harness-identity.v0.2'
AGENTIC_WORKSPACE_BROKER_ATTESTATION_SCHEMA_VERSION = 'vaxreplay.agentic-workspace-broker-attestation.v0.1'

_RUN_FILES = {
    'run.json',
    'run.hmac',
    'transcript.json',
    'tool-events.json',
    'scratch-manifest.json',
    'submission.json',
    'workspace-broker-attestation.json',
}
_MAX_JSON_BYTES = 64 * 1024 * 1024


class AgenticRunArtifactError(ValueError):
    """Raised when a run handoff is incomplete, unbound, or unauthenticated."""


class AgenticHarnessIdentity(StrictModel):
    schema_version: Literal['vaxreplay.agentic-harness-identity.v0.2'] = AGENTIC_HARNESS_IDENTITY_SCHEMA_VERSION
    harness_id: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)
    harness_image_or_commitment: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    harness_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    harness_behavior_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    harness_execution_mode: Literal['fixed_model_loop', 'submitted_guest_agent']
    requested_model_id: str = Field(min_length=1)
    adapter_id: str = Field(min_length=1)


class AgenticToolEvent(StrictModel):
    schema_version: Literal['vaxreplay.agentic-tool-event.v0.1'] = AGENTIC_TOOL_EVENT_SCHEMA_VERSION
    event_index: int = Field(ge=0)
    tool: AgenticTool
    gateway_call_index: int | None = Field(default=None, ge=0)
    started_at: datetime
    finished_at: datetime
    request_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    request_bytes: int = Field(ge=0)
    response_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    response_bytes: int = Field(ge=0)
    succeeded: bool

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError('tool event cannot finish before it starts')
        if (self.tool == AgenticTool.MODEL_GENERATE) != (self.gateway_call_index is not None):
            raise ValueError('only model_generate events must carry a gateway call index')
        return self


class AgenticScratchEntry(StrictModel):
    schema_version: Literal['vaxreplay.agentic-scratch-entry.v0.1'] = AGENTIC_SCRATCH_ENTRY_SCHEMA_VERSION
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(ge=0)

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        normalized_relative_path(f'scratch/{value}', prefix='scratch')
        return value


class AgenticWorkspaceBrokerAttestation(StrictModel):
    """Trusted-supervisor observation of the worker's only workspace access channel."""

    schema_version: Literal['vaxreplay.agentic-workspace-broker-attestation.v0.1'] = (
        AGENTIC_WORKSPACE_BROKER_ATTESTATION_SCHEMA_VERSION
    )
    workspace_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    workspace_tree_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_visible_surface_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    broker_id: str = Field(min_length=1)
    broker_version: str = Field(min_length=1)
    broker_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    access_mode: Literal['brokered_exact_bytes_only'] = 'brokered_exact_bytes_only'
    worker_workspace_filesystem_mounted: Literal[False] = False
    filesystem_metadata_exposed: Literal[False] = False
    origin_metadata_exposed: Literal[False] = False
    canonical_logical_paths_only: Literal[True] = True
    exact_byte_reads_only: Literal[True] = True
    immutable_snapshot_for_run: Literal[True] = True
    list_read_search_events_authoritative: Literal[True] = True
    surface_sha256_before_run: str = Field(pattern=r'^[0-9a-f]{64}$')
    surface_sha256_after_run: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_surface_unchanged(self) -> Self:
        if (
            self.surface_sha256_before_run != self.model_visible_surface_sha256
            or self.surface_sha256_after_run != self.model_visible_surface_sha256
        ):
            raise ValueError('workspace broker surface must match before and after the run')
        return self


@dataclass(frozen=True)
class LoadedAgenticRunArtifact:
    root: Path
    receipt: AgenticRunReceipt
    receipt_sha256: str
    submission: AgenticSubmissionV1 | None
    transcript: AgenticGatewayTranscript
    tool_events: tuple[AgenticToolEvent, ...]
    scratch_manifest: tuple[AgenticScratchEntry, ...]
    workspace_broker_attestation: AgenticWorkspaceBrokerAttestation


def finalize_agentic_run(
    *,
    output_root: Path,
    run_id: str,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
    attempt_reservation_sha256: str,
    policy: AgenticExecutionPolicy,
    harness: AgenticHarnessIdentity,
    capabilities: BackendCapabilities,
    workspace_broker_attestation: AgenticWorkspaceBrokerAttestation,
    gateway_transcript: AgenticGatewayTranscript,
    tool_events: tuple[AgenticToolEvent, ...],
    scratch_files: Mapping[str, bytes],
    final_submission_bytes: bytes,
    started_at: datetime,
    finished_at: datetime,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    failure_code: AgenticRunFailureCode | None = None,
    gateway_channel_isolation: bool,
    tool_tracing_authoritative: bool,
    provider_cost_usd: float | None = None,
) -> LoadedAgenticRunArtifact:
    """Create an authenticated exact-inventory handoff from trusted supervisor observations."""

    workspace = load_agentic_workspace(workspace.root)
    admission = require_workspace_admission(
        workspace,
        admission,
        expected_admission_sha256=expected_admission_sha256,
    )
    if policy.required_isolation == IsolationTier.OFFICIAL and not admission.official_release_eligible:
        raise AgenticRunArtifactError('official execution requires an authenticated official release admission')
    _validate_run_id(run_id)
    started_at, finished_at, duration_ms = _validate_run_interval(
        started_at,
        finished_at,
        wall_seconds=policy.limits.wall_seconds,
    )
    if gateway_transcript.run_id != run_id:
        raise AgenticRunArtifactError('gateway transcript is bound to a different run')
    _validate_backend(capabilities, policy, gateway_channel_isolation, tool_tracing_authoritative)
    _validate_workspace_broker(workspace_broker_attestation, workspace=workspace, policy=policy)
    _validate_gateway(gateway_transcript, policy)
    _validate_tool_events(
        tool_events,
        policy,
        transcript=gateway_transcript,
        run_started_at=started_at,
        run_finished_at=finished_at,
    )
    scratch_manifest = _scratch_manifest(scratch_files, policy)

    accepted = failure_code is None
    _validate_submission(
        final_submission_bytes,
        workspace=workspace,
        accepted=accepted,
        maximum_bytes=policy.limits.max_final_bytes,
    )
    if not accepted and final_submission_bytes:
        raise AgenticRunArtifactError('rejected Agentic runs must hand off an empty final submission')

    transcript_bytes = canonical_json_bytes(gateway_transcript)
    tool_event_bytes = canonical_json_bytes([event.model_dump(mode='json') for event in tool_events])
    scratch_bytes = canonical_json_bytes([entry.model_dump(mode='json') for entry in scratch_manifest])
    broker_attestation_bytes = canonical_json_bytes(workspace_broker_attestation)
    usage = AgenticModelUsage(
        model_calls=len(gateway_transcript.exchanges),
        input_tokens=gateway_transcript.input_tokens,
        output_tokens=gateway_transcript.output_tokens,
        reasoning_tokens=gateway_transcript.reasoning_tokens,
        provider_cost_usd=provider_cost_usd,
        gateway_metering_authoritative=gateway_transcript.metering_authoritative,
    )
    key_id = agentic_receipt_key_id(receipt_key)
    if key_id != expected_receipt_key_id:
        raise AgenticRunArtifactError('receipt key does not match the release-pinned key ID')
    sealed = capabilities.isolation_tier == IsolationTier.OFFICIAL
    receipt = AgenticRunReceipt(
        run_id=run_id,
        task_id=workspace.task.task_id,
        episode_manifest_sha256=workspace.task.episode_manifest_sha256,
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        build_policy_sha256=workspace.manifest.build_policy_sha256,
        discovery_manifest_sha256=workspace.manifest.discovery_manifest_sha256,
        alias_seed_commitment_sha256=workspace.manifest.alias_seed_commitment_sha256,
        alias_permutation_receipt_sha256=workspace.manifest.alias_permutation_receipt_sha256,
        temporal_admission_sha256=admission.temporal_admission_sha256,
        contamination_admission_sha256=admission.contamination_binding_sha256,
        workspace_admission_sha256=expected_admission_sha256,
        attempt_reservation_sha256=attempt_reservation_sha256,
        policy_sha256=agentic_policy_sha256(policy),
        receipt_key_id=key_id,
        harness_id=harness.harness_id,
        harness_version=harness.harness_version,
        harness_image_or_commitment=harness.harness_image_or_commitment,
        harness_manifest_sha256=harness.harness_manifest_sha256,
        harness_behavior_sha256=harness.harness_behavior_sha256,
        harness_execution_mode=harness.harness_execution_mode,
        requested_model_id=harness.requested_model_id,
        resolved_model_id=gateway_transcript.resolved_model_id,
        adapter_id=harness.adapter_id,
        isolation_tier=capabilities.isolation_tier,
        sealed=sealed,
        network_isolation=capabilities.network_isolation,
        host_filesystem_isolation=capabilities.host_filesystem_isolation,
        gateway_channel_isolation=gateway_channel_isolation,
        tool_tracing_authoritative=tool_tracing_authoritative,
        development_only=not sealed,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        usage=usage,
        transcript_sha256=_sha256(transcript_bytes),
        tool_events_sha256=_sha256(tool_event_bytes),
        workspace_broker_attestation_sha256=_sha256(broker_attestation_bytes),
        scratch_tree_sha256=_sha256(scratch_bytes),
        final_submission_sha256=_sha256(final_submission_bytes),
        final_submission_bytes=len(final_submission_bytes),
        accepted=accepted,
        failure_code=failure_code,
        residual_retrospective_selection_contamination=(admission.residual_retrospective_selection_contamination),
    )

    target = output_root.expanduser().resolve()
    if target.exists():
        raise AgenticRunArtifactError(f'run artifact output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        files = {
            'run.json': canonical_json_bytes(receipt),
            'run.hmac': (agentic_run_receipt_hmac(receipt, receipt_key) + '\n').encode('ascii'),
            'transcript.json': transcript_bytes,
            'tool-events.json': tool_event_bytes,
            'scratch-manifest.json': scratch_bytes,
            'submission.json': final_submission_bytes,
            'workspace-broker-attestation.json': broker_attestation_bytes,
        }
        for name, content in files.items():
            destination = staging / name
            destination.write_bytes(content)
            destination.chmod(0o600)
        staging.chmod(0o700)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_agentic_run_artifact(
        target,
        workspace=workspace,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
        expected_attempt_reservation_sha256=attempt_reservation_sha256,
        policy=policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
    )


def load_agentic_run_artifact(
    root: Path,
    *,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
    expected_attempt_reservation_sha256: str,
    policy: AgenticExecutionPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> LoadedAgenticRunArtifact:
    """Verify an authenticated run handoff without executing participant code or loading gold."""

    workspace = load_agentic_workspace(workspace.root)
    admission = require_workspace_admission(
        workspace,
        admission,
        expected_admission_sha256=expected_admission_sha256,
    )
    if policy.required_isolation == IsolationTier.OFFICIAL and not admission.official_release_eligible:
        raise AgenticRunArtifactError('official execution requires an authenticated official release admission')
    resolved = root.expanduser()
    if resolved.is_symlink():
        raise AgenticRunArtifactError('run artifact root cannot be a symlink')
    resolved = resolved.resolve()
    if not resolved.is_dir() or stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise AgenticRunArtifactError('run artifact root must be a private mode-0700 directory')
    actual = {entry.name for entry in os.scandir(resolved)}
    if actual != _RUN_FILES:
        raise AgenticRunArtifactError('run artifact exact file inventory mismatch')

    receipt_bytes = _read_file(resolved / 'run.json', _MAX_JSON_BYTES)
    transcript_bytes = _read_file(resolved / 'transcript.json', _MAX_JSON_BYTES)
    tool_event_bytes = _read_file(resolved / 'tool-events.json', _MAX_JSON_BYTES)
    scratch_bytes = _read_file(resolved / 'scratch-manifest.json', _MAX_JSON_BYTES)
    broker_attestation_bytes = _read_file(
        resolved / 'workspace-broker-attestation.json',
        _MAX_JSON_BYTES,
    )
    final_bytes = _read_file(resolved / 'submission.json', policy.limits.max_final_bytes)
    hmac_bytes = _read_file(resolved / 'run.hmac', 65)
    try:
        receipt = AgenticRunReceipt.model_validate_json(receipt_bytes)
        transcript = AgenticGatewayTranscript.model_validate_json(transcript_bytes)
        raw_events = _canonical_json_list(tool_event_bytes)
        events = tuple(AgenticToolEvent.model_validate_json(canonical_json_bytes(value)) for value in raw_events)
        raw_scratch = _canonical_json_list(scratch_bytes)
        scratch = tuple(AgenticScratchEntry.model_validate_json(canonical_json_bytes(value)) for value in raw_scratch)
        broker_attestation = AgenticWorkspaceBrokerAttestation.model_validate_json(broker_attestation_bytes)
    except ValueError as error:
        raise AgenticRunArtifactError(f'invalid Agentic run artifact: {error}') from error
    if receipt_bytes != canonical_json_bytes(receipt):
        raise AgenticRunArtifactError('run receipt must use canonical JSON')
    if transcript_bytes != canonical_json_bytes(transcript):
        raise AgenticRunArtifactError('gateway transcript must use canonical JSON')
    if tool_event_bytes != canonical_json_bytes([event.model_dump(mode='json') for event in events]):
        raise AgenticRunArtifactError('tool events must use canonical JSON')
    if scratch_bytes != canonical_json_bytes([entry.model_dump(mode='json') for entry in scratch]):
        raise AgenticRunArtifactError('scratch manifest must use canonical JSON')
    if broker_attestation_bytes != canonical_json_bytes(broker_attestation):
        raise AgenticRunArtifactError('workspace broker attestation must use canonical JSON')

    key_id = agentic_receipt_key_id(receipt_key)
    if key_id != expected_receipt_key_id or receipt.receipt_key_id != key_id:
        raise AgenticRunArtifactError('run receipt uses a different authentication key')
    expected_hmac = (agentic_run_receipt_hmac(receipt, receipt_key) + '\n').encode('ascii')
    if not hmac.compare_digest(hmac_bytes, expected_hmac):
        raise AgenticRunArtifactError('run receipt HMAC authentication failed')
    _validate_loaded_bindings(
        receipt,
        workspace=workspace,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
        expected_attempt_reservation_sha256=expected_attempt_reservation_sha256,
        policy=policy,
        transcript=transcript,
        tool_events=events,
        transcript_bytes=transcript_bytes,
        tool_event_bytes=tool_event_bytes,
        scratch_bytes=scratch_bytes,
        broker_attestation=broker_attestation,
        broker_attestation_bytes=broker_attestation_bytes,
        final_bytes=final_bytes,
    )
    submission = _validate_submission(
        final_bytes,
        workspace=workspace,
        accepted=receipt.accepted,
        maximum_bytes=policy.limits.max_final_bytes,
    )
    return LoadedAgenticRunArtifact(
        root=resolved,
        receipt=receipt,
        receipt_sha256=_sha256(receipt_bytes),
        submission=submission,
        transcript=transcript,
        tool_events=events,
        scratch_manifest=scratch,
        workspace_broker_attestation=broker_attestation,
    )


def _validate_loaded_bindings(
    receipt: AgenticRunReceipt,
    *,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
    expected_attempt_reservation_sha256: str,
    policy: AgenticExecutionPolicy,
    transcript: AgenticGatewayTranscript,
    tool_events: tuple[AgenticToolEvent, ...],
    transcript_bytes: bytes,
    tool_event_bytes: bytes,
    scratch_bytes: bytes,
    broker_attestation: AgenticWorkspaceBrokerAttestation,
    broker_attestation_bytes: bytes,
    final_bytes: bytes,
) -> None:
    _, _, expected_duration_ms = _validate_run_interval(
        receipt.started_at,
        receipt.finished_at,
        wall_seconds=policy.limits.wall_seconds,
    )
    if receipt.duration_ms != expected_duration_ms:
        raise AgenticRunArtifactError('run receipt duration does not match its timestamps')
    expected_workspace = (
        workspace.task.task_id,
        workspace.task.episode_manifest_sha256,
        workspace.manifest_sha256,
        workspace.manifest.workspace_tree_sha256,
        workspace.manifest.model_visible_surface_sha256,
        workspace.manifest.build_policy_sha256,
        workspace.manifest.discovery_manifest_sha256,
        workspace.manifest.alias_seed_commitment_sha256,
        workspace.manifest.alias_permutation_receipt_sha256,
        admission.temporal_admission_sha256,
        admission.contamination_binding_sha256,
        expected_admission_sha256,
        expected_attempt_reservation_sha256,
        agentic_policy_sha256(policy),
    )
    actual_workspace = (
        receipt.task_id,
        receipt.episode_manifest_sha256,
        receipt.workspace_manifest_sha256,
        receipt.workspace_tree_sha256,
        receipt.model_visible_surface_sha256,
        receipt.build_policy_sha256,
        receipt.discovery_manifest_sha256,
        receipt.alias_seed_commitment_sha256,
        receipt.alias_permutation_receipt_sha256,
        receipt.temporal_admission_sha256,
        receipt.contamination_admission_sha256,
        receipt.workspace_admission_sha256,
        receipt.attempt_reservation_sha256,
        receipt.policy_sha256,
    )
    if actual_workspace != expected_workspace:
        raise AgenticRunArtifactError('run receipt is bound to a different workspace, admission, or policy')
    if (
        receipt.transcript_sha256,
        receipt.tool_events_sha256,
        receipt.workspace_broker_attestation_sha256,
        receipt.scratch_tree_sha256,
        receipt.final_submission_sha256,
        receipt.final_submission_bytes,
    ) != (
        _sha256(transcript_bytes),
        _sha256(tool_event_bytes),
        _sha256(broker_attestation_bytes),
        _sha256(scratch_bytes),
        _sha256(final_bytes),
        len(final_bytes),
    ):
        raise AgenticRunArtifactError('run receipt artifact hashes do not match the exact handoff files')
    if receipt.resolved_model_id != transcript.resolved_model_id:
        raise AgenticRunArtifactError('run receipt resolved model does not match the gateway transcript')
    if (
        receipt.usage.model_calls,
        receipt.usage.input_tokens,
        receipt.usage.output_tokens,
        receipt.usage.reasoning_tokens,
        receipt.usage.gateway_metering_authoritative,
    ) != (
        len(transcript.exchanges),
        transcript.input_tokens,
        transcript.output_tokens,
        transcript.reasoning_tokens,
        transcript.metering_authoritative,
    ):
        raise AgenticRunArtifactError('run receipt usage does not match the gateway transcript')
    _validate_gateway(transcript, policy)
    _validate_workspace_broker(broker_attestation, workspace=workspace, policy=policy)
    _validate_tool_events(
        tool_events,
        policy,
        transcript=transcript,
        run_started_at=receipt.started_at,
        run_finished_at=receipt.finished_at,
    )
    if receipt.accepted != (receipt.failure_code is None):
        raise AgenticRunArtifactError('run receipt terminal status is inconsistent')
    if not receipt.accepted and final_bytes:
        raise AgenticRunArtifactError('rejected run artifact contains a final submission')


def _validate_gateway(transcript: AgenticGatewayTranscript, policy: AgenticExecutionPolicy) -> None:
    limits = policy.limits
    if (
        len(transcript.exchanges) > limits.max_model_calls
        or transcript.input_tokens > limits.max_input_tokens
        or transcript.output_tokens > limits.max_output_tokens
        or (transcript.reasoning_tokens is not None and transcript.reasoning_tokens > limits.max_reasoning_tokens)
    ):
        raise AgenticRunArtifactError('gateway transcript exceeds the execution policy budget')


def _validate_workspace_broker(
    attestation: AgenticWorkspaceBrokerAttestation,
    *,
    workspace: LoadedAgenticWorkspace,
    policy: AgenticExecutionPolicy,
) -> None:
    expected = (
        workspace.manifest_sha256,
        workspace.manifest.workspace_tree_sha256,
        workspace.manifest.model_visible_surface_sha256,
        policy.required_workspace_broker_id,
        policy.required_workspace_broker_version,
        policy.required_workspace_broker_executable_sha256,
    )
    actual = (
        attestation.workspace_manifest_sha256,
        attestation.workspace_tree_sha256,
        attestation.model_visible_surface_sha256,
        attestation.broker_id,
        attestation.broker_version,
        attestation.broker_executable_sha256,
    )
    if actual != expected:
        raise AgenticRunArtifactError('workspace broker attestation does not match the workspace or policy')


def _validate_tool_events(
    events: tuple[AgenticToolEvent, ...],
    policy: AgenticExecutionPolicy,
    *,
    transcript: AgenticGatewayTranscript,
    run_started_at: datetime,
    run_finished_at: datetime,
) -> None:
    if tuple(event.event_index for event in events) != tuple(range(len(events))):
        raise AgenticRunArtifactError('tool events must be contiguous and start at zero')
    allowed = set(policy.tool_policy.allowed_tools)
    if any(event.tool not in allowed for event in events):
        raise AgenticRunArtifactError('tool trace contains a tool forbidden by policy')
    if any(event.started_at < run_started_at or event.finished_at > run_finished_at for event in events):
        raise AgenticRunArtifactError('tool trace contains an event outside the run interval')
    gateway_events = tuple(event for event in events if event.tool == AgenticTool.MODEL_GENERATE)
    if len(gateway_events) > policy.limits.max_model_calls:
        raise AgenticRunArtifactError('tool trace exceeds the model-call budget')
    if len(gateway_events) != len(transcript.exchanges):
        raise AgenticRunArtifactError('tool trace must contain exactly one event per gateway exchange')
    for call_index, (event, exchange) in enumerate(zip(gateway_events, transcript.exchanges, strict=True)):
        request_bytes = canonical_json_bytes(exchange.request)
        response_bytes = canonical_json_bytes(exchange.response)
        if (
            event.gateway_call_index,
            event.request_sha256,
            event.request_bytes,
            event.response_sha256,
            event.response_bytes,
            event.succeeded,
        ) != (
            call_index,
            exchange.receipt.request_sha256,
            len(request_bytes),
            exchange.receipt.response_sha256,
            len(response_bytes),
            True,
        ):
            raise AgenticRunArtifactError('model tool event does not bind its exact gateway exchange')


def _validate_backend(
    capabilities: BackendCapabilities,
    policy: AgenticExecutionPolicy,
    gateway_channel_isolation: bool,
    tool_tracing_authoritative: bool,
) -> None:
    if capabilities.isolation_tier != policy.required_isolation:
        raise AgenticRunArtifactError('backend isolation tier does not meet the execution policy')
    if capabilities.isolation_tier == IsolationTier.OFFICIAL and not all(
        (
            capabilities.network_isolation,
            capabilities.host_filesystem_isolation,
            capabilities.read_only_root,
            capabilities.non_root_user,
            capabilities.capability_drop,
            capabilities.no_new_privileges,
            capabilities.process_limit,
            capabilities.memory_limit,
            capabilities.cpu_limit,
            capabilities.scratch_limit,
            capabilities.fresh_worker_per_episode,
            gateway_channel_isolation,
            tool_tracing_authoritative,
        )
    ):
        raise AgenticRunArtifactError('official backend lacks a required isolation or tracing capability')


def _scratch_manifest(
    scratch_files: Mapping[str, bytes],
    policy: AgenticExecutionPolicy,
) -> tuple[AgenticScratchEntry, ...]:
    total = 0
    entries: list[AgenticScratchEntry] = []
    folded: set[str] = set()
    for path, content in sorted(scratch_files.items()):
        if not isinstance(content, bytes):
            raise AgenticRunArtifactError('scratch files must be exact bytes')
        normalized_relative_path(f'scratch/{path}', prefix='scratch')
        folded_path = path.casefold()
        if folded_path in folded:
            raise AgenticRunArtifactError('scratch paths cannot collide under case folding')
        folded.add(folded_path)
        total += len(content)
        if total > policy.limits.scratch_mib * 1024 * 1024:
            raise AgenticRunArtifactError('scratch tree exceeds the execution policy budget')
        entries.append(
            AgenticScratchEntry(
                path=path,
                sha256=_sha256(content),
                byte_count=len(content),
            )
        )
    return tuple(entries)


def _validate_submission(
    payload: bytes,
    *,
    workspace: LoadedAgenticWorkspace,
    accepted: bool,
    maximum_bytes: int,
) -> AgenticSubmissionV1 | None:
    if len(payload) > maximum_bytes:
        raise AgenticRunArtifactError('final submission exceeds the execution policy byte limit')
    if not accepted:
        return None
    if not payload:
        raise AgenticRunArtifactError('accepted Agentic run requires a final submission')
    try:
        submission = AgenticSubmissionV1.model_validate_json(payload)
    except ValueError as error:
        raise AgenticRunArtifactError(f'invalid final Agentic submission: {error}') from error
    if payload != canonical_json_bytes(submission):
        raise AgenticRunArtifactError('final Agentic submission must use canonical JSON')
    if (
        submission.task_id != workspace.task.task_id
        or submission.workspace_manifest_sha256 != workspace.manifest_sha256
    ):
        raise AgenticRunArtifactError('final submission is bound to a different task or workspace')
    return submission


def _canonical_json_list(payload: bytes) -> list[object]:
    import json

    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AgenticRunArtifactError('run artifact list is not valid UTF-8 JSON') from error
    if not isinstance(value, list):
        raise AgenticRunArtifactError('run artifact inventory must be a JSON list')
    return value


def _read_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AgenticRunArtifactError(f'cannot open run artifact file {path}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum_bytes
        ):
            raise AgenticRunArtifactError('run artifact must contain private bounded regular files')
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes - len(content) + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise AgenticRunArtifactError('run artifact file exceeds its byte limit')
    finally:
        os.close(descriptor)


def _validate_run_id(run_id: str) -> None:
    if len(run_id) != 32 or any(character not in '0123456789abcdef' for character in run_id):
        raise AgenticRunArtifactError('run_id must contain exactly 32 lowercase hexadecimal characters')


def _validate_run_interval(
    started_at: datetime,
    finished_at: datetime,
    *,
    wall_seconds: int,
) -> tuple[datetime, datetime, int]:
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise AgenticRunArtifactError('started_at must include a UTC offset')
    if finished_at.tzinfo is None or finished_at.utcoffset() is None:
        raise AgenticRunArtifactError('finished_at must include a UTC offset')
    normalized_start = started_at.astimezone(UTC)
    normalized_finish = finished_at.astimezone(UTC)
    if normalized_finish < normalized_start:
        raise AgenticRunArtifactError('finished_at cannot precede started_at')
    duration_ms = round((normalized_finish - normalized_start).total_seconds() * 1000)
    if duration_ms > wall_seconds * 1000:
        raise AgenticRunArtifactError('run interval exceeds the execution policy wall-clock limit')
    return normalized_start, normalized_finish, duration_ms


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
