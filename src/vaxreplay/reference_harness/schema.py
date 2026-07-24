"""Contracts for development-only local CLI reference runs.

These records are deliberately separate from the sealed-runner receipts.  A local
Codex, Claude, or Cursor invocation is useful for diagnostics, but it does not establish a
network boundary or control knowledge already present in model weights.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.case_schema import StrictModel, Submission
from vaxreplay.prompt import PromptVariant

REFERENCE_HARNESS_RECEIPT_SCHEMA_VERSION = 'vaxreplay.reference-harness-receipt.v0.4'
LEGACY_REFERENCE_HARNESS_RECEIPT_SCHEMA_VERSION = 'vaxreplay.reference-harness-receipt.v0.3'
CURSOR_PARSE_FAILURE_INVENTORY_SCHEMA_VERSION = 'vaxreplay.cursor-parse-failure-inventory.v0.1'


class ReferenceHarnessName(str, enum.Enum):
    CODEX = 'codex-cli'
    CLAUDE = 'claude-code-cli'
    CURSOR = 'cursor-agent-cli'


class RawResponseSource(str, enum.Enum):
    CODEX_LAST_MESSAGE = 'codex_output_last_message'
    CLAUDE_STDOUT = 'claude_json_stdout'
    CURSOR_STDOUT = 'cursor_stream_json_stdout'
    NONE = 'none'


class ReferenceHarnessFailureCode(str, enum.Enum):
    VERSION_UNAVAILABLE = 'version_unavailable'
    LAUNCH_ERROR = 'launch_error'
    EXECUTION_ERROR = 'execution_error'
    TIMED_OUT = 'timed_out'
    CLI_STDOUT_LIMIT = 'cli_stdout_limit'
    CLI_STDERR_LIMIT = 'cli_stderr_limit'
    RESPONSE_LIMIT = 'response_limit'
    NONZERO_EXIT = 'nonzero_exit'
    MISSING_RESPONSE = 'missing_response'
    INVALID_UTF8 = 'invalid_utf8'
    INVALID_WRAPPER = 'invalid_wrapper'
    PROVIDER_ERROR = 'provider_error'
    UNEXPECTED_TOOL_CALL = 'unexpected_tool_call'
    INVALID_SUBMISSION = 'invalid_submission'


class ReferenceHarnessLimits(StrictModel):
    wall_seconds: int = Field(default=600, ge=1, le=86_400)
    max_response_bytes: int = Field(default=1_048_576, ge=1_024, le=67_108_864)
    max_cli_stdout_bytes: int = Field(default=1_048_576, ge=1_024, le=67_108_864)
    max_cli_stderr_bytes: int = Field(default=1_048_576, ge=1_024, le=67_108_864)


class ReferenceHarnessFailure(StrictModel):
    code: ReferenceHarnessFailureCode
    detail: str = Field(min_length=1)
    validation_error_count: int | None = Field(default=None, ge=1)


class CursorEventKindObservation(StrictModel):
    """A content-free count of one sanitized Cursor event kind."""

    event_type: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    event_subtype: str | None = Field(default=None, pattern=r'^[a-z][a-z0-9_]{0,63}$')
    count: int = Field(ge=1)
    first_line: int = Field(ge=1)
    last_line: int = Field(ge=1)

    @model_validator(mode='after')
    def validate_lines(self) -> Self:
        if self.last_line < self.first_line:
            raise ValueError('event inventory last_line cannot precede first_line')
        return self


class CursorParseConsistencyFlags(StrictModel):
    event_type_subtype_contract_valid: bool
    first_system_init_valid: bool
    single_system_init: bool
    user_event_position_valid: bool
    terminal_event_position_valid: bool
    session_metadata_consistent: bool
    model_metadata_consistent: bool
    workspace_metadata_consistent: bool
    user_transcript_consistent: bool


class CursorParseFailureInventory(StrictModel):
    """Bounded event metadata emitted only when a Cursor response fails closed."""

    schema_version: Literal['vaxreplay.cursor-parse-failure-inventory.v0.1'] = (
        CURSOR_PARSE_FAILURE_INVENTORY_SCHEMA_VERSION
    )
    total_lines: int = Field(ge=1)
    nonempty_lines: int = Field(ge=1)
    parsed_event_lines: int = Field(ge=0)
    first_unparseable_line: int | None = Field(default=None, ge=1)
    duplicate_json_key_observed: bool
    event_kinds: tuple[CursorEventKindObservation, ...]
    tool_event_count: int = Field(ge=0)
    terminal_event_count: int = Field(ge=0)
    consistency: CursorParseConsistencyFlags

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        if self.nonempty_lines > self.total_lines:
            raise ValueError('nonempty_lines cannot exceed total_lines')
        if self.parsed_event_lines > self.nonempty_lines:
            raise ValueError('parsed_event_lines cannot exceed nonempty_lines')
        if self.first_unparseable_line is not None and self.first_unparseable_line > self.total_lines:
            raise ValueError('first_unparseable_line exceeds total_lines')
        if self.first_unparseable_line is None and self.parsed_event_lines != self.nonempty_lines:
            raise ValueError('a partially parsed inventory must identify its first unparseable line')
        if self.duplicate_json_key_observed and self.first_unparseable_line is None:
            raise ValueError('a duplicate JSON key must identify an unparseable line')
        if sum(item.count for item in self.event_kinds) != self.parsed_event_lines:
            raise ValueError('event kind counts must cover every parsed event line')
        if len({(item.event_type, item.event_subtype) for item in self.event_kinds}) != len(self.event_kinds):
            raise ValueError('event kind observations must be unique')
        if any(item.last_line > self.total_lines for item in self.event_kinds):
            raise ValueError('event kind line exceeds total_lines')
        if self.tool_event_count > self.parsed_event_lines or self.terminal_event_count > self.parsed_event_lines:
            raise ValueError('special event counts cannot exceed parsed_event_lines')
        return self


class ReferenceHarnessRuntimeIdentity(StrictModel):
    """Best-effort local bytes identity, never a remote or hardware attestation."""

    scope: Literal['cursor_runtime_regular_file_tree_v1'] = 'cursor_runtime_regular_file_tree_v1'
    tree_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    regular_file_count: int = Field(ge=3)
    total_file_bytes: int = Field(ge=1)
    chunk_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    chunk_file_count: int = Field(ge=1)
    entrypoint_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    package_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    identity_attested: Literal[False] = False

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        if self.chunk_file_count > self.regular_file_count:
            raise ValueError('chunk_file_count cannot exceed regular_file_count')
        return self


class ReferenceHarnessReceipt(StrictModel):
    """Canonical, content-free audit record for one local reference invocation."""

    schema_version: Literal[
        'vaxreplay.reference-harness-receipt.v0.3',
        'vaxreplay.reference-harness-receipt.v0.4',
    ] = REFERENCE_HARNESS_RECEIPT_SCHEMA_VERSION
    development_only: Literal[True] = True
    sealed_execution: Literal[False] = False
    network_isolation: Literal[False] = False
    provider_route_attested: Literal[False] = False
    local_configuration_isolation_attested: Literal[False] = False
    model_weight_contamination_uncontrolled: Literal[True] = True

    challenge_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    suite_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    envelope_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    ordinal: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    episode_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    prompt_variant: PromptVariant

    harness_name: ReferenceHarnessName
    harness_version: str = Field(min_length=1, max_length=256)
    harness_executable_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    harness_executable_sha256_scope: Literal['invoked_file_bytes', 'unavailable'] = 'unavailable'
    harness_executable_identity_attested: Literal[False] = False
    harness_runtime_identity: ReferenceHarnessRuntimeIdentity | None = None
    requested_model: str | None = Field(default=None, min_length=1, max_length=512)
    resolved_model: str | None = Field(default=None, min_length=1, max_length=512)
    additional_reported_models: tuple[str, ...] = ()

    command_argv: tuple[str, ...] = Field(min_length=1)
    command_argv_redacted: Literal[True] = True
    prompt_passed_via_stdin: Literal[True] = True
    rendered_prompt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    rendered_prompt_bytes: int = Field(gt=0)
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)
    process_termination: Literal['not_started', 'exited', 'timed_out', 'response_limit', 'log_limit']
    exit_code: int | None = None

    raw_response_source: RawResponseSource
    raw_response_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    raw_response_bytes: int = Field(ge=0)
    raw_response_truncated: bool
    cli_stdout_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    cli_stdout_bytes: int = Field(ge=0)
    cli_stdout_truncated: bool
    cli_stderr_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    cli_stderr_bytes: int = Field(ge=0)
    cli_stderr_truncated: bool

    submission: Submission | None = None
    failure: ReferenceHarnessFailure | None = None
    cursor_parse_failure_inventory: CursorParseFailureInventory | None = None

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('reference-harness timestamps must include a UTC offset')
        return value

    @field_validator('additional_reported_models')
    @classmethod
    def validate_additional_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not model or len(model) > 512 for model in value):
            raise ValueError('reported model IDs must contain between 1 and 512 characters')
        if len(value) != len(set(value)):
            raise ValueError('additional reported model IDs must be unique')
        return value

    @field_validator('command_argv')
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument or '\x00' in argument for argument in value):
            raise ValueError('receipt argv must contain non-empty, NUL-free arguments')
        return value

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError('finished_at cannot precede started_at')
        if (self.submission is None) == (self.failure is None):
            raise ValueError('exactly one of submission or failure must be present')
        if self.raw_response_source == RawResponseSource.NONE and self.raw_response_bytes != 0:
            raise ValueError('a missing raw response must have zero bytes')
        if self.resolved_model is not None and self.resolved_model in self.additional_reported_models:
            raise ValueError('the primary resolved model cannot also be an additional model')
        if (self.harness_executable_sha256 is None) != (self.harness_executable_sha256_scope == 'unavailable'):
            raise ValueError('harness executable hash and scope must agree')
        if self.harness_runtime_identity is not None and self.harness_name != ReferenceHarnessName.CURSOR:
            raise ValueError('only Cursor receipts can contain a Cursor runtime identity')
        if self.cursor_parse_failure_inventory is not None:
            if self.harness_name != ReferenceHarnessName.CURSOR or self.failure is None:
                raise ValueError('Cursor parse inventory requires a failed Cursor receipt')
        return self
