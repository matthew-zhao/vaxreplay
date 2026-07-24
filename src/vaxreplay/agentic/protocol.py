"""Versioned execution contracts for the task-level Agentic Replay track.

These contracts intentionally do not reuse the one-shot runner receipt.  An agent run has a
read-only workspace, writable scratch, repeated metered inference calls, and a larger observable
trace.  The protocol records those facts without pretending that a local vendor CLI is sealed.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import math
from datetime import datetime, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.response_protocol import (
    AGENTIC_RANKING_RESPONSE_PROTOCOL,
    AgenticResponseProtocol,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.runner.schema import IsolationTier

AGENTIC_RESPONSE_PROTOCOL = AGENTIC_RANKING_RESPONSE_PROTOCOL
AGENTIC_TOOL_POLICY_SCHEMA_VERSION = 'vaxreplay.agentic-tool-policy.v0.1'
AGENTIC_RUN_LIMITS_SCHEMA_VERSION = 'vaxreplay.agentic-run-limits.v0.1'
AGENTIC_EXECUTION_POLICY_SCHEMA_VERSION = 'vaxreplay.agentic-execution-policy.v0.1'
AGENTIC_MODEL_USAGE_SCHEMA_VERSION = 'vaxreplay.agentic-model-usage.v0.1'
AGENTIC_RUN_RECEIPT_SCHEMA_VERSION = 'vaxreplay.agentic-run-receipt.v0.2'
AGENTIC_RECEIPT_AUTHENTICATION = 'hmac-sha256-domain-separated'
_AGENTIC_RECEIPT_HMAC_DOMAIN = b'vaxreplay.agentic-run-receipt.v0.2\x00'


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class AgenticTool(str, enum.Enum):
    LIST_WORKSPACE = 'list_workspace'
    READ_WORKSPACE = 'read_workspace'
    SEARCH_WORKSPACE = 'search_workspace'
    LOCAL_COMPUTE = 'local_compute'
    WRITE_SCRATCH = 'write_scratch'
    MODEL_GENERATE = 'model_generate'


_TOOL_ORDER = tuple(AgenticTool)


class AgenticToolPolicy(StrictModel):
    """Fail-closed tool surface inside the hostile harness worker."""

    schema_version: Literal['vaxreplay.agentic-tool-policy.v0.1'] = AGENTIC_TOOL_POLICY_SCHEMA_VERSION
    allowed_tools: tuple[AgenticTool, ...] = _TOOL_ORDER
    general_network_allowed: Literal[False] = False
    browser_allowed: Literal[False] = False
    remote_mcp_allowed: Literal[False] = False
    provider_storage_allowed: Literal[False] = False
    persistent_sessions_allowed: Literal[False] = False
    host_workspace_allowed: Literal[False] = False
    arbitrary_shell_network_allowed: Literal[False] = False

    @field_validator('allowed_tools')
    @classmethod
    def validate_tools(cls, value: tuple[AgenticTool, ...]) -> tuple[AgenticTool, ...]:
        expected = tuple(tool for tool in _TOOL_ORDER if tool in value)
        if value != expected or len(value) != len(set(value)):
            raise ValueError('allowed_tools must be unique and use canonical protocol order')
        required = {
            AgenticTool.LIST_WORKSPACE,
            AgenticTool.READ_WORKSPACE,
            AgenticTool.SEARCH_WORKSPACE,
            AgenticTool.WRITE_SCRATCH,
            AgenticTool.MODEL_GENERATE,
        }
        if not required.issubset(value):
            raise ValueError('Agentic Replay requires workspace retrieval, scratch, and model generation')
        return value


class AgenticRunLimits(StrictModel):
    schema_version: Literal['vaxreplay.agentic-run-limits.v0.1'] = AGENTIC_RUN_LIMITS_SCHEMA_VERSION
    max_model_calls: int = Field(default=20, ge=1, le=100)
    max_input_tokens: int = Field(default=262_144, ge=1, le=4_000_000)
    max_output_tokens: int = Field(default=32_768, ge=1, le=1_000_000)
    max_reasoning_tokens: int = Field(default=262_144, ge=1, le=4_000_000)
    wall_seconds: int = Field(default=1_200, ge=1, le=86_400)
    cpus: float = Field(default=4.0, gt=0, le=64, allow_inf_nan=False)
    memory_mib: int = Field(default=8_192, ge=128, le=262_144)
    scratch_mib: int = Field(default=1_024, ge=1, le=65_536)
    pids: int = Field(default=256, ge=16, le=4_096)
    max_log_bytes: int = Field(default=8 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    max_final_bytes: int = Field(default=1024 * 1024, ge=1, le=16 * 1024 * 1024)


class AgenticExecutionPolicy(StrictModel):
    schema_version: Literal['vaxreplay.agentic-execution-policy.v0.1'] = AGENTIC_EXECUTION_POLICY_SCHEMA_VERSION
    required_isolation: IsolationTier = IsolationTier.OFFICIAL
    tool_policy: AgenticToolPolicy = Field(default_factory=AgenticToolPolicy)
    limits: AgenticRunLimits = Field(default_factory=AgenticRunLimits)
    inference_gateway_required: Literal[True] = True
    provider_credentials_exposed_to_worker: Literal[False] = False
    one_attempt: Literal[True] = True
    intermediate_scoring_feedback: Literal[False] = False
    response_protocol: AgenticResponseProtocol = AgenticResponseProtocol.RANKING
    workspace_access_mode: Literal['brokered_exact_bytes_only'] = 'brokered_exact_bytes_only'
    workspace_filesystem_mounted_to_worker: Literal[False] = False
    workspace_metadata_exposed_to_worker: Literal[False] = False
    required_workspace_broker_id: str = Field(min_length=1)
    required_workspace_broker_version: str = Field(min_length=1)
    required_workspace_broker_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_development_protocol(self) -> Self:
        if (
            self.response_protocol == AgenticResponseProtocol.CLINICAL_EXECUTION
            and self.required_isolation != IsolationTier.DEVELOPMENT
        ):
            raise ValueError('clinical-execution response protocol is development-only')
        return self


class AgenticModelUsage(StrictModel):
    schema_version: Literal['vaxreplay.agentic-model-usage.v0.1'] = AGENTIC_MODEL_USAGE_SCHEMA_VERSION
    model_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    provider_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    gateway_metering_authoritative: bool


class AgenticRunFailureCode(str, enum.Enum):
    POLICY_REJECTED = 'policy_rejected'
    WORKSPACE_INTEGRITY = 'workspace_integrity'
    TEMPORAL_ADMISSION = 'temporal_admission'
    CONTAMINATION_ADMISSION = 'contamination_admission'
    GATEWAY_FAILURE = 'gateway_failure'
    BUDGET_EXCEEDED = 'budget_exceeded'
    TIMED_OUT = 'timed_out'
    HARNESS_FAILURE = 'harness_failure'
    INVALID_SUBMISSION = 'invalid_submission'


class AgenticRunReceipt(StrictModel):
    """Organizer-authenticated binding of one terminal task attempt."""

    schema_version: Literal['vaxreplay.agentic-run-receipt.v0.2'] = AGENTIC_RUN_RECEIPT_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    task_id: str = Field(min_length=1)
    episode_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    workspace_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    workspace_tree_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_visible_surface_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    build_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    discovery_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    alias_seed_commitment_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    alias_permutation_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    temporal_admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contamination_admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    workspace_admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    attempt_reservation_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    receipt_authentication: Literal['hmac-sha256-domain-separated'] = AGENTIC_RECEIPT_AUTHENTICATION
    receipt_key_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    harness_id: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)
    harness_image_or_commitment: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    harness_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    harness_behavior_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    harness_execution_mode: Literal['fixed_model_loop', 'submitted_guest_agent']
    requested_model_id: str = Field(min_length=1)
    resolved_model_id: str | None = None
    adapter_id: str = Field(min_length=1)
    isolation_tier: IsolationTier
    sealed: bool
    network_isolation: bool
    host_filesystem_isolation: bool
    gateway_channel_isolation: bool
    tool_tracing_authoritative: bool
    development_only: bool
    one_attempt: Literal[True] = True
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)
    usage: AgenticModelUsage
    transcript_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    tool_events_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    workspace_broker_attestation_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    scratch_tree_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    final_submission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    final_submission_bytes: int = Field(ge=0)
    accepted: bool
    failure_code: AgenticRunFailureCode | None = None
    residual_model_weight_contamination: Literal[True] = True
    residual_harness_embedded_knowledge: Literal[True] = True
    residual_retrospective_selection_contamination: bool
    proves_absence_of_contamination: Literal[False] = False

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _aware(value, info.field_name)

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError('finished_at cannot precede started_at')
        if not math.isclose(
            self.duration_ms / 1000,
            (self.finished_at - self.started_at).total_seconds(),
            rel_tol=0.1,
            abs_tol=2.0,
        ):
            raise ValueError('duration_ms is inconsistent with receipt timestamps')
        expected_sealed = self.isolation_tier == IsolationTier.OFFICIAL
        if self.sealed != expected_sealed:
            raise ValueError('sealed must reflect the declared isolation tier')
        if self.development_only == self.sealed:
            raise ValueError('development_only must be the inverse of sealed')
        if self.sealed and not all(
            (
                self.network_isolation,
                self.host_filesystem_isolation,
                self.gateway_channel_isolation,
                self.tool_tracing_authoritative,
                self.usage.gateway_metering_authoritative,
            )
        ):
            raise ValueError('official Agentic receipts require every isolation and metering control')
        if self.accepted == (self.failure_code is not None):
            raise ValueError('accepted receipts cannot have a failure; rejected receipts require one')
        if self.accepted and self.final_submission_bytes == 0:
            raise ValueError('accepted receipts require a non-empty final submission')
        return self


def agentic_policy_sha256(policy: AgenticExecutionPolicy) -> str:
    return hashlib.sha256(canonical_json_bytes(policy)).hexdigest()


def agentic_run_receipt_hmac(receipt: AgenticRunReceipt, key: bytes) -> str:
    if len(key) < 32:
        raise ValueError('Agentic run receipt HMAC key must contain at least 32 bytes')
    return hmac.new(key, _AGENTIC_RECEIPT_HMAC_DOMAIN + canonical_json_bytes(receipt), hashlib.sha256).hexdigest()


def agentic_receipt_key_id(key: bytes) -> str:
    if len(key) < 32:
        raise ValueError('Agentic run receipt HMAC key must contain at least 32 bytes')
    return hashlib.sha256(b'vaxreplay.agentic-receipt-key-id.v0.1\x00' + key).hexdigest()
