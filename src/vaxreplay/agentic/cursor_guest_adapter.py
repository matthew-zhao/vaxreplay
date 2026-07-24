"""Fail-closed Cursor Agent *protocol fixture* for the Lane A guest boundary.

This module deliberately stops short of claiming a runnable Cursor harness.  The pinned standard
``cursor-agent`` distribution can emit the bounded stream-JSON grammar implemented here, but its
ordinary mode requires Cursor authentication and Cursor's backend.  Its credential-free custom
provider path belongs to a separate ``agent-cli-local`` runtime that is not present in the measured
payload.  VaxReplay therefore has no checked-in provider-gateway bridge for Cursor today.

The fixture is still useful: it measures the pinned wrapper, materializes a complete read-only
workspace through authenticated guest RPC, freezes the intended read-only argv/environment,
validates a caller-supplied deterministic stream-JSON transcript, and submits one task-bound answer.
There is intentionally no default process transport, so this API cannot accidentally make a live
Cursor or external-model call.  Passing this fixture is not development-adapter integration, Linux
payload evidence, KVM qualification, or leaderboard admission.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import json
import os
import platform
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self, cast

from pydantic import Field, JsonValue, ValidationError, field_validator, model_validator

from vaxreplay.agentic.guest_rpc import (
    ListWorkspaceResult,
    LogicalFileResult,
    ReadWorkspaceResult,
    SubmitResult,
)
from vaxreplay.agentic.headless_guest_adapter import (
    MODEL_SELECTOR_TOKEN,
    WORKSPACE_PATH_TOKEN,
    HeadlessGuestAdapterConfig,
    HeadlessInvocationProtocol,
    HeadlessResponseChannel,
    headless_guest_adapter_config_sha256,
    require_headless_guest_adapter_binding,
)
from vaxreplay.agentic.submitted_harness import (
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessManifest,
    submitted_harness_manifest_sha256,
)
from vaxreplay.agentic.task_protocol import (
    AgenticRuntimeSubmission,
    AgenticTaskInvocation,
    agentic_task_invocation_sha256,
    parse_submission_for_invocation,
    submission_json_schema_for_invocation,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.runner._process import BoundedProcessResult, run_bounded_process

CURSOR_PROTOCOL_FIXTURE_CONFIG_SCHEMA_VERSION = 'vaxreplay.cursor-protocol-fixture-config.dev-v0.1'
CURSOR_PROTOCOL_FIXTURE_RECEIPT_SCHEMA_VERSION = 'vaxreplay.cursor-protocol-fixture-receipt.dev-v0.1'
CURSOR_VENDOR_IDENTITY_EVIDENCE_SCHEMA_VERSION = 'vaxreplay.cursor-vendor-identity-evidence.dev-v0.1'
CURSOR_PROTOCOL_FIXTURE_ID = 'vaxreplay-cursor-protocol-fixture'
CURSOR_PROTOCOL_FIXTURE_VERSION = 'dev-v0.1'
CURSOR_SUPPORTED_VENDOR_VERSION = '2026.07.09-a3815c0'
CURSOR_VENDOR_EXECUTABLE_PATH = '/opt/vaxreplay/vendor/cursor/cursor-agent'
CURSOR_EMPTY_HOME_PATH = '/run/vaxreplay/scratch/home'
CURSOR_TMP_PATH = '/run/vaxreplay/scratch/tmp'
CURSOR_DATA_PATH = '/run/vaxreplay/scratch/cursor-data'
CURSOR_CONFIG_PATH = '/run/vaxreplay/scratch/cursor-config'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MODEL_SELECTOR_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,499}$')
_READ_ONLY_TOOL_KINDS = frozenset({'readToolCall', 'lsToolCall', 'grepToolCall', 'globToolCall'})
_TOOL_CALL_METADATA_KEYS = frozenset({'hookAdditionalContexts', 'toolCallId', 'startedAtMs', 'completedAtMs'})


class CursorProtocolFixtureFailureCode(str, enum.Enum):
    BINDING_REJECTED = 'binding_rejected'
    RUNTIME_LAYOUT_REJECTED = 'runtime_layout_rejected'
    VENDOR_EXECUTABLE_REJECTED = 'vendor_executable_rejected'
    WORKSPACE_REJECTED = 'workspace_rejected'
    TRANSPORT_REJECTED = 'transport_rejected'
    OUTPUT_REJECTED = 'output_rejected'
    SUBMISSION_REJECTED = 'submission_rejected'


class CursorProtocolFixtureError(RuntimeError):
    """Content-free fixture failure suitable for the guest appliance boundary."""

    def __init__(self, code: CursorProtocolFixtureFailureCode):
        super().__init__(code.value)
        self.code = code


class CursorProtocolFixtureLimits(StrictModel):
    maximum_workspace_files: int = Field(default=2_000, ge=1, le=100_000)
    maximum_workspace_bytes: int = Field(
        default=128 * 1024 * 1024,
        ge=1,
        le=2 * 1024 * 1024 * 1024,
    )
    workspace_list_page_size: int = Field(default=100, ge=1, le=1_000)
    workspace_read_chunk_bytes: int = Field(
        default=1024 * 1024,
        ge=1,
        le=16 * 1024 * 1024,
    )
    maximum_prompt_bytes: int = Field(default=4 * 1024 * 1024, ge=1_024, le=32 * 1024 * 1024)
    maximum_stream_bytes: int = Field(default=4 * 1024 * 1024, ge=1_024, le=64 * 1024 * 1024)
    maximum_stderr_bytes: int = Field(default=1024 * 1024, ge=1_024, le=16 * 1024 * 1024)
    maximum_stream_events: int = Field(default=2_000, ge=4, le=100_000)
    maximum_stream_line_bytes: int = Field(default=2 * 1024 * 1024, ge=1_024, le=16 * 1024 * 1024)
    maximum_tool_calls: int = Field(default=100, ge=0, le=10_000)
    vendor_wall_seconds: int = Field(default=900, ge=1, le=7_200)


class CursorProtocolFixtureConfig(StrictModel):
    """Hash-bound facts for the non-runnable Cursor protocol fixture."""

    schema_version: Literal['vaxreplay.cursor-protocol-fixture-config.dev-v0.1'] = (
        CURSOR_PROTOCOL_FIXTURE_CONFIG_SCHEMA_VERSION
    )
    fixture_id: Literal['vaxreplay-cursor-protocol-fixture'] = CURSOR_PROTOCOL_FIXTURE_ID
    fixture_version: Literal['dev-v0.1'] = CURSOR_PROTOCOL_FIXTURE_VERSION
    headless_adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    submitted_harness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_path: Literal['/opt/vaxreplay/vendor/cursor/cursor-agent'] = CURSOR_VENDOR_EXECUTABLE_PATH
    vendor_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_byte_count: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    supported_vendor_version: Literal['2026.07.09-a3815c0'] = CURSOR_SUPPORTED_VENDOR_VERSION
    limits: CursorProtocolFixtureLimits = CursorProtocolFixtureLimits()
    workspace_materialization_fixture_uses_only_authenticated_guest_rpc: Literal[True] = True
    final_submission_fixture_uses_authenticated_guest_rpc: Literal[True] = True
    stream_json_grammar_is_bounded_and_fail_closed: Literal[True] = True
    caller_supplied_transport_required: Literal[True] = True
    default_live_transport_exists: Literal[False] = False
    cursor_provider_gateway_bridge_implemented: Literal[False] = False
    standard_cursor_payload_supports_authless_local_provider: Literal[False] = False
    actual_cursor_binary_end_to_end_validated: Literal[False] = False
    actual_external_model_call_claimed: Literal[False] = False
    development_adapter_integrated: Literal[False] = False
    linux_kvm_qualified: Literal[False] = False
    protocol_fixture_only: Literal[True] = True
    development_only: Literal[True] = True


class CursorVendorIdentityEvidence(StrictModel):
    schema_version: Literal['vaxreplay.cursor-vendor-identity-evidence.dev-v0.1'] = (
        CURSOR_VENDOR_IDENTITY_EVIDENCE_SCHEMA_VERSION
    )
    executable_path: str = Field(min_length=2, max_length=4096)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    executable_byte_count: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    version_argv_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stdout_bytes: int = Field(ge=0, le=4096)
    version_stderr_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stderr_bytes: int = Field(ge=0, le=4096)
    reported_version: Literal['2026.07.09-a3815c0'] = CURSOR_SUPPORTED_VENDOR_VERSION
    observed_os: str = Field(min_length=1, max_length=100)
    observed_architecture: str = Field(min_length=1, max_length=100)
    executable_was_regular_nonsymlink_file: Literal[True] = True
    version_environment_was_exact_and_credential_free: Literal[True] = True
    evidence_is_only_outer_wrapper_identity: Literal[True] = True
    evidence_is_not_dependency_closure: Literal[True] = True
    evidence_is_not_provider_bridge_validation: Literal[True] = True
    evidence_is_not_linux_kvm_qualification: Literal[True] = True
    development_only: Literal[True] = True

    @field_validator('executable_path')
    @classmethod
    def validate_executable_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or path.as_posix() != value:
            raise ValueError('measured Cursor executable path must be absolute and normalized')
        return value


class CursorMaterializedWorkspaceFile(StrictModel):
    path: str = Field(min_length=1, max_length=4096)
    media_type: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)


class CursorProtocolEventReceipt(StrictModel):
    event_index: int = Field(ge=0)
    event_type: Literal['system', 'user', 'assistant', 'thinking', 'tool_call', 'result']
    event_subtype: str | None = Field(default=None, max_length=100)
    event_sha256: str = Field(pattern=_SHA256_PATTERN)
    event_bytes: int = Field(gt=0)
    call_id_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    tool_kind: Literal['readToolCall', 'lsToolCall', 'grepToolCall', 'globToolCall'] | None = None


class CursorProtocolFixtureReceipt(StrictModel):
    schema_version: Literal['vaxreplay.cursor-protocol-fixture-receipt.dev-v0.1'] = (
        CURSOR_PROTOCOL_FIXTURE_RECEIPT_SCHEMA_VERSION
    )
    fixture_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    headless_adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    submitted_harness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_model_selector_sha256: str = Field(pattern=_SHA256_PATTERN)
    reported_cursor_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_byte_count: int = Field(gt=0)
    vendor_argv_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_bytes: int = Field(gt=0)
    submission_schema_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_file_count: int = Field(ge=1)
    workspace_byte_count: int = Field(ge=0)
    protocol_events: tuple[CursorProtocolEventReceipt, ...] = Field(min_length=4)
    protocol_event_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    completed_read_only_tool_calls: int = Field(ge=0)
    transport_duration_ms: int = Field(ge=0)
    transport_stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    transport_stdout_bytes: int = Field(gt=0)
    transport_stderr_sha256: str = Field(pattern=_SHA256_PATTERN)
    transport_stderr_bytes: int = Field(ge=0)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    submit_result: SubmitResult
    complete_workspace_was_materialized_read_only: Literal[True] = True
    transport_was_caller_supplied: Literal[True] = True
    transport_provenance_is_not_attested: Literal[True] = True
    no_builtin_live_transport: Literal[True] = True
    adapter_itself_makes_no_model_generate_call: Literal[True] = True
    cursor_provider_gateway_bridge_implemented: Literal[False] = False
    actual_cursor_binary_end_to_end_validated: Literal[False] = False
    actual_external_model_call_claimed: Literal[False] = False
    development_adapter_integrated: Literal[False] = False
    linux_kvm_qualified: Literal[False] = False
    protocol_fixture_only: Literal[True] = True
    development_only: Literal[True] = True

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        event_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in self.protocol_events])
        if self.protocol_event_log_sha256 != _sha256(event_bytes):
            raise ValueError('protocol event log hash does not match its exact receipts')
        if self.submit_result.submission_sha256 != self.submission_sha256:
            raise ValueError('submit result does not match the fixture submission')
        return self


class _CursorTextBlock(StrictModel):
    type: Literal['text']
    text: str = Field(max_length=2_000_000)


class _CursorMessage(StrictModel):
    role: Literal['user', 'assistant']
    content: tuple[_CursorTextBlock, ...] = Field(min_length=1, max_length=10_000)


class _CursorSystemInitEvent(StrictModel):
    type: Literal['system']
    subtype: Literal['init']
    apiKeySource: str = Field(min_length=1, max_length=100)
    cwd: str = Field(min_length=1, max_length=4096)
    session_id: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=500)
    permissionMode: Literal['default']


class _CursorUserEvent(StrictModel):
    type: Literal['user']
    message: _CursorMessage
    session_id: str = Field(min_length=1, max_length=500)

    @model_validator(mode='after')
    def require_user_role(self) -> Self:
        if self.message.role != 'user':
            raise ValueError('user event has the wrong message role')
        return self


class _CursorAssistantEvent(StrictModel):
    type: Literal['assistant']
    message: _CursorMessage
    session_id: str = Field(min_length=1, max_length=500)
    model_call_id: str | None = Field(default=None, max_length=500)
    timestamp_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def require_assistant_role(self) -> Self:
        if self.message.role != 'assistant':
            raise ValueError('assistant event has the wrong message role')
        return self


class _CursorThinkingEvent(StrictModel):
    type: Literal['thinking']
    subtype: Literal['delta', 'completed']
    text: str | None = Field(default=None, max_length=2_000_000)
    session_id: str = Field(min_length=1, max_length=500)
    timestamp_ms: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_text_shape(self) -> Self:
        if (self.subtype == 'delta') != (self.text is not None):
            raise ValueError('thinking text shape does not match its subtype')
        return self


class _CursorToolCallEvent(StrictModel):
    type: Literal['tool_call']
    subtype: Literal['started', 'completed']
    call_id: str = Field(min_length=1, max_length=500)
    tool_call: dict[str, JsonValue]
    model_call_id: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=500)
    timestamp_ms: int = Field(ge=0)


class _CursorResultEvent(StrictModel):
    type: Literal['result']
    subtype: Literal['success']
    duration_ms: int = Field(ge=0)
    duration_api_ms: int = Field(ge=0)
    is_error: Literal[False]
    result: str = Field(min_length=1, max_length=2_000_000)
    session_id: str = Field(min_length=1, max_length=500)
    request_id: str = Field(min_length=1, max_length=500)
    usage: dict[str, JsonValue] | None = None


class CursorProtocolFixtureGuestRpcClient(Protocol):
    def list_workspace(self, *, cursor: int = 0, limit: int = 100) -> ListWorkspaceResult: ...

    def read_workspace(self, path: str, *, offset: int = 0, limit: int) -> ReadWorkspaceResult: ...

    def submit(self, submission: AgenticRuntimeSubmission) -> SubmitResult: ...


class CursorProtocolFixtureTransport(Protocol):
    """Explicit test-double transport; there is intentionally no production default."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        wall_seconds: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        on_abort: Callable[[], None],
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> BoundedProcessResult: ...


@dataclass(frozen=True)
class _RuntimePaths:
    root: Path

    def physical(self, logical: str) -> Path:
        path = PurePosixPath(logical)
        if not path.is_absolute() or '..' in path.parts or path.as_posix() != logical:
            raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.RUNTIME_LAYOUT_REJECTED)
        if self.root == Path('/'):
            return Path(logical)
        return self.root.joinpath(*path.parts[1:])


@dataclass(frozen=True)
class _ParsedCursorStream:
    submission: AgenticRuntimeSubmission
    reported_model: str
    receipts: tuple[CursorProtocolEventReceipt, ...]
    completed_read_only_tool_calls: int


def cursor_protocol_fixture_config_sha256(config: CursorProtocolFixtureConfig) -> str:
    canonical = CursorProtocolFixtureConfig.model_validate_json(canonical_json_bytes(config))
    return _sha256(canonical_json_bytes(canonical))


def cursor_vendor_argv_template() -> tuple[str, ...]:
    """Return the pinned read-only standard Cursor CLI shape used by protocol fixtures."""

    return (
        CURSOR_VENDOR_EXECUTABLE_PATH,
        '--print',
        '--output-format',
        'stream-json',
        '--mode',
        'ask',
        '--sandbox',
        'enabled',
        '--workspace',
        WORKSPACE_PATH_TOKEN,
        '--model',
        MODEL_SELECTOR_TOKEN,
        '--disable-auto-update',
        '--disable-project-configs',
        '--disable-indexing',
        '--disable-codebase-ref',
        '--exclude-workspace-context',
        '--single-turn',
        '--allowed-tools',
        'read_tool_call',
        '--allowed-tools',
        'ls_tool_call',
        '--allowed-tools',
        'grep_tool_call',
        '--allowed-tools',
        'glob_tool_call',
    )


def require_cursor_protocol_fixture_binding(
    *,
    config: CursorProtocolFixtureConfig,
    headless_config: HeadlessGuestAdapterConfig,
    submitted_manifest: SubmittedHarnessManifest,
) -> None:
    try:
        runtime = CursorProtocolFixtureConfig.model_validate_json(canonical_json_bytes(config))
        headless = HeadlessGuestAdapterConfig.model_validate_json(canonical_json_bytes(headless_config))
        submitted = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(submitted_manifest))
        require_headless_guest_adapter_binding(config=headless, manifest=submitted)
    except (TypeError, ValueError):
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.BINDING_REJECTED) from None
    expected = (
        HarnessFamily.CURSOR,
        HeadlessInvocationProtocol.CURSOR_PRINT,
        HeadlessResponseChannel.BOUNDED_JSONL_STDOUT,
        HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED,
        False,
        False,
        False,
        False,
        CURSOR_VENDOR_EXECUTABLE_PATH,
        runtime.vendor_executable_sha256,
        CURSOR_SUPPORTED_VENDOR_VERSION,
        cursor_vendor_argv_template(),
        headless_guest_adapter_config_sha256(headless),
        submitted_harness_manifest_sha256(submitted),
    )
    actual = (
        headless.family,
        headless.invocation_protocol,
        headless.response_channel,
        submitted.runtime_support,
        headless.adapter_implementation_checked_in,
        headless.provider_shim_implementation_checked_in,
        headless.workspace_materialization_bridge_implementation_checked_in,
        headless.local_shell_enabled,
        headless.vendor_executable_path,
        headless.vendor_executable_sha256,
        headless.vendor_reported_version,
        headless.vendor_argv_template,
        runtime.headless_adapter_config_sha256,
        runtime.submitted_harness_manifest_sha256,
    )
    if actual != expected:
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.BINDING_REJECTED)


def render_cursor_vendor_argv(
    *,
    organizer_model_selector: str,
    paths: _RuntimePaths | None = None,
) -> tuple[str, ...]:
    if not _MODEL_SELECTOR_PATTERN.fullmatch(organizer_model_selector):
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.BINDING_REJECTED)
    runtime_paths = paths or _RuntimePaths(Path('/'))
    substitutions = {
        CURSOR_VENDOR_EXECUTABLE_PATH: str(runtime_paths.physical(CURSOR_VENDOR_EXECUTABLE_PATH)),
        MODEL_SELECTOR_TOKEN: organizer_model_selector,
        WORKSPACE_PATH_TOKEN: str(runtime_paths.physical('/run/vaxreplay/workspace')),
    }
    return tuple(substitutions.get(item, item) for item in cursor_vendor_argv_template())


def capture_cursor_vendor_identity(
    executable_path: Path,
    *,
    process_runner: Callable[..., BoundedProcessResult] = run_bounded_process,
) -> CursorVendorIdentityEvidence:
    """Measure one regular Cursor wrapper and bounded ``--version`` output without model access."""

    path_text = executable_path.as_posix()
    pure_path = PurePosixPath(path_text)
    if (
        not executable_path.is_absolute()
        or '..' in pure_path.parts
        or pure_path.as_posix() != path_text
        or executable_path.is_symlink()
    ):
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.VENDOR_EXECUTABLE_REJECTED)
    try:
        executable_sha256, executable_bytes = _measure_regular_executable(executable_path)
        with tempfile.TemporaryDirectory(prefix='vaxreplay-cursor-identity-') as temporary:
            temporary_root = Path(temporary)
            home = temporary_root / 'home'
            scratch = temporary_root / 'tmp'
            data = temporary_root / 'data'
            config = temporary_root / 'config'
            for directory in (home, scratch, data, config):
                directory.mkdir(mode=0o700)
            environment = _identity_environment(
                home=home,
                scratch=scratch,
                data=data,
                config=config,
            )
            result = process_runner(
                (path_text, '--version'),
                input_bytes=b'',
                wall_seconds=10,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                on_abort=lambda: None,
                env=environment,
            )
        if (
            result.termination != 'exited'
            or result.exit_code != 0
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            raise ValueError('version process failed')
        version_bytes = result.stdout.strip() or result.stderr.strip()
        reported_version = version_bytes.decode('utf-8')
        if reported_version != CURSOR_SUPPORTED_VENDOR_VERSION:
            raise ValueError('unsupported version')
        return CursorVendorIdentityEvidence(
            executable_path=path_text,
            executable_sha256=executable_sha256,
            executable_byte_count=executable_bytes,
            version_argv_sha256=_sha256(canonical_json_bytes((path_text, '--version'))),
            version_stdout_sha256=_sha256(result.stdout),
            version_stdout_bytes=len(result.stdout),
            version_stderr_sha256=_sha256(result.stderr),
            version_stderr_bytes=len(result.stderr),
            reported_version=cast(Literal['2026.07.09-a3815c0'], reported_version),
            observed_os=platform.system(),
            observed_architecture=platform.machine(),
        )
    except CursorProtocolFixtureError:
        raise
    except (OSError, UnicodeDecodeError, ValueError):
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.VENDOR_EXECUTABLE_REJECTED) from None


def run_cursor_protocol_fixture(
    client: CursorProtocolFixtureGuestRpcClient,
    *,
    task_invocation: AgenticTaskInvocation,
    organizer_model_selector: str,
    config: CursorProtocolFixtureConfig,
    headless_config: HeadlessGuestAdapterConfig,
    submitted_manifest: SubmittedHarnessManifest,
    protocol_transport: CursorProtocolFixtureTransport,
    _guest_root: Path = Path('/'),
) -> CursorProtocolFixtureReceipt:
    """Validate a deterministic Cursor transcript; never launch Cursor by default."""

    try:
        invocation = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(task_invocation))
        runtime = CursorProtocolFixtureConfig.model_validate_json(canonical_json_bytes(config))
        require_cursor_protocol_fixture_binding(
            config=runtime,
            headless_config=headless_config,
            submitted_manifest=submitted_manifest,
        )
        if not _guest_root.is_absolute() or _guest_root.is_symlink():
            raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.RUNTIME_LAYOUT_REJECTED)
        paths = _RuntimePaths(_guest_root)
        vendor_path = paths.physical(runtime.vendor_executable_path)
        _verify_vendor_executable(vendor_path, runtime)
        workspace_files = _materialize_workspace(client, paths=paths, limits=runtime.limits)
        submission_schema_bytes = canonical_json_bytes(submission_json_schema_for_invocation(invocation))
        inventory_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in workspace_files])
        prompt = _render_cursor_prompt(
            invocation,
            workspace_files,
            submission_schema_bytes=submission_schema_bytes,
        )
        if len(prompt) > runtime.limits.maximum_prompt_bytes:
            raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.WORKSPACE_REJECTED)
        argv = render_cursor_vendor_argv(
            organizer_model_selector=organizer_model_selector,
            paths=paths,
        )
        environment = _sealed_vendor_environment(paths)
        try:
            transport_result = protocol_transport(
                argv,
                input_bytes=prompt,
                wall_seconds=runtime.limits.vendor_wall_seconds,
                max_stdout_bytes=runtime.limits.maximum_stream_bytes,
                max_stderr_bytes=runtime.limits.maximum_stderr_bytes,
                on_abort=lambda: None,
                env=environment,
                cwd=paths.physical('/run/vaxreplay/workspace'),
            )
        except Exception:
            raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.TRANSPORT_REJECTED) from None
        if (
            transport_result.termination != 'exited'
            or transport_result.exit_code != 0
            or transport_result.stdout_truncated
            or transport_result.stderr_truncated
            or len(transport_result.stdout) > runtime.limits.maximum_stream_bytes
            or len(transport_result.stderr) > runtime.limits.maximum_stderr_bytes
        ):
            raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.TRANSPORT_REJECTED)
        parsed = _parse_cursor_stream_json(
            invocation,
            transport_result.stdout,
            expected_prompt=prompt,
            expected_workspace_path=str(paths.physical('/run/vaxreplay/workspace')),
            limits=runtime.limits,
        )
        try:
            submit_result = client.submit(parsed.submission)
        except Exception:
            raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.SUBMISSION_REJECTED) from None
        submission_bytes = canonical_json_bytes(parsed.submission)
        protocol_event_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in parsed.receipts])
        return CursorProtocolFixtureReceipt(
            fixture_config_sha256=cursor_protocol_fixture_config_sha256(runtime),
            headless_adapter_config_sha256=runtime.headless_adapter_config_sha256,
            submitted_harness_manifest_sha256=runtime.submitted_harness_manifest_sha256,
            task_invocation_sha256=agentic_task_invocation_sha256(invocation),
            organizer_model_selector_sha256=_sha256(organizer_model_selector.encode('utf-8')),
            reported_cursor_model_sha256=_sha256(parsed.reported_model.encode('utf-8')),
            vendor_executable_sha256=runtime.vendor_executable_sha256,
            vendor_executable_byte_count=runtime.vendor_executable_byte_count,
            vendor_argv_sha256=_sha256(canonical_json_bytes(argv)),
            vendor_environment_sha256=_sha256(canonical_json_bytes(dict(sorted(environment.items())))),
            prompt_sha256=_sha256(prompt),
            prompt_bytes=len(prompt),
            submission_schema_sha256=_sha256(submission_schema_bytes),
            workspace_inventory_sha256=_sha256(inventory_bytes),
            workspace_file_count=len(workspace_files),
            workspace_byte_count=sum(item.byte_count for item in workspace_files),
            protocol_events=parsed.receipts,
            protocol_event_log_sha256=_sha256(protocol_event_bytes),
            completed_read_only_tool_calls=parsed.completed_read_only_tool_calls,
            transport_duration_ms=transport_result.duration_ms,
            transport_stdout_sha256=_sha256(transport_result.stdout),
            transport_stdout_bytes=len(transport_result.stdout),
            transport_stderr_sha256=_sha256(transport_result.stderr),
            transport_stderr_bytes=len(transport_result.stderr),
            submission_sha256=_sha256(submission_bytes),
            submit_result=submit_result,
        )
    except CursorProtocolFixtureError:
        raise
    except (TypeError, ValueError, ValidationError):
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.BINDING_REJECTED) from None


def _parse_cursor_stream_json(
    invocation: AgenticTaskInvocation,
    stdout: bytes,
    *,
    expected_prompt: bytes,
    expected_workspace_path: str,
    limits: CursorProtocolFixtureLimits,
) -> _ParsedCursorStream:
    try:
        if not stdout or len(stdout) > limits.maximum_stream_bytes or b'\r' in stdout:
            raise ValueError('invalid stream framing')
        lines = stdout.split(b'\n')
        if lines[-1] == b'':
            lines.pop()
        if (
            len(lines) < 4
            or len(lines) > limits.maximum_stream_events
            or any(not line or len(line) > limits.maximum_stream_line_bytes for line in lines)
        ):
            raise ValueError('invalid stream line count or size')
        raw_payloads = [_load_unique_json(line) for line in lines]
        if any(not isinstance(payload, dict) for payload in raw_payloads):
            raise ValueError('stream event is not an object')
        payloads = [cast(dict[str, JsonValue], payload) for payload in raw_payloads]
        system = _CursorSystemInitEvent.model_validate_json(canonical_json_bytes(payloads[0]))
        user = _CursorUserEvent.model_validate_json(canonical_json_bytes(payloads[1]))
        if system.cwd != expected_workspace_path or user.session_id != system.session_id:
            raise ValueError('initial event binding mismatch')
        prompt_text = expected_prompt.decode('utf-8')
        if tuple(block.text for block in user.message.content) != (prompt_text,):
            raise ValueError('user event does not echo the exact fixture prompt')

        receipts = [
            _event_receipt(0, payloads[0]),
            _event_receipt(1, payloads[1]),
        ]
        open_calls: dict[str, tuple[str, str]] = {}
        seen_call_ids: set[str] = set()
        started_calls = 0
        completed_calls = 0
        assistant_text: list[str] = []
        thinking_open = False
        terminal: _CursorResultEvent | None = None
        for index, payload in enumerate(payloads[2:], start=2):
            event_type = payload.get('type')
            if event_type == 'assistant':
                event = _CursorAssistantEvent.model_validate_json(canonical_json_bytes(payload))
                if event.session_id != system.session_id:
                    raise ValueError('assistant session mismatch')
                assistant_text.extend(block.text for block in event.message.content)
            elif event_type == 'thinking':
                event = _CursorThinkingEvent.model_validate_json(canonical_json_bytes(payload))
                if event.session_id != system.session_id:
                    raise ValueError('thinking session mismatch')
                if event.subtype == 'delta':
                    thinking_open = True
                elif not thinking_open:
                    raise ValueError('thinking completion without a delta')
                else:
                    thinking_open = False
            elif event_type == 'tool_call':
                event = _CursorToolCallEvent.model_validate_json(canonical_json_bytes(payload))
                if event.session_id != system.session_id:
                    raise ValueError('tool session mismatch')
                tool_kind = _tool_kind(event.tool_call, expected_call_id=event.call_id)
                if event.subtype == 'started':
                    if event.call_id in seen_call_ids or started_calls >= limits.maximum_tool_calls:
                        raise ValueError('duplicate or excess tool start')
                    seen_call_ids.add(event.call_id)
                    open_calls[event.call_id] = (tool_kind, event.model_call_id)
                    started_calls += 1
                else:
                    expected = open_calls.pop(event.call_id, None)
                    if expected != (tool_kind, event.model_call_id):
                        raise ValueError('tool completion does not match its start')
                    completed_calls += 1
            elif event_type == 'result':
                if index != len(payloads) - 1 or terminal is not None:
                    raise ValueError('result must be the single final event')
                terminal = _CursorResultEvent.model_validate_json(canonical_json_bytes(payload))
                if terminal.session_id != system.session_id:
                    raise ValueError('result session mismatch')
            else:
                raise ValueError('unknown or unadmitted Cursor stream event')
            receipts.append(_event_receipt(index, payload))
        if terminal is None or open_calls or thinking_open or not assistant_text:
            raise ValueError('stream did not reach one complete terminal state')
        if ''.join(assistant_text) != terminal.result:
            raise ValueError('assistant stream differs from terminal result')
        submission = parse_submission_for_invocation(
            invocation,
            terminal.result.encode('utf-8'),
        )
        return _ParsedCursorStream(
            submission=submission,
            reported_model=system.model,
            receipts=tuple(receipts),
            completed_read_only_tool_calls=completed_calls,
        )
    except CursorProtocolFixtureError:
        raise
    except (TypeError, ValueError, ValidationError, UnicodeDecodeError):
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.OUTPUT_REJECTED) from None


def _event_receipt(index: int, payload: dict[str, JsonValue]) -> CursorProtocolEventReceipt:
    event_type = payload.get('type')
    if event_type not in {'system', 'user', 'assistant', 'thinking', 'tool_call', 'result'}:
        raise ValueError('unknown event type')
    subtype = payload.get('subtype')
    if subtype is not None and not isinstance(subtype, str):
        raise ValueError('invalid event subtype')
    event_bytes = canonical_json_bytes(payload)
    call_id = payload.get('call_id')
    if call_id is not None and not isinstance(call_id, str):
        raise ValueError('invalid call ID')
    tool_kind: str | None = None
    if event_type == 'tool_call':
        tool_call = payload.get('tool_call')
        if not isinstance(tool_call, dict):
            raise ValueError('invalid tool call')
        tool_kind = _tool_kind(
            cast(dict[str, JsonValue], tool_call),
            expected_call_id=call_id,
        )
    return CursorProtocolEventReceipt(
        event_index=index,
        event_type=cast(
            Literal['system', 'user', 'assistant', 'thinking', 'tool_call', 'result'],
            event_type,
        ),
        event_subtype=subtype,
        event_sha256=_sha256(event_bytes),
        event_bytes=len(event_bytes),
        call_id_sha256=_sha256(call_id.encode('utf-8')) if call_id is not None else None,
        tool_kind=cast(
            Literal['readToolCall', 'lsToolCall', 'grepToolCall', 'globToolCall'] | None,
            tool_kind,
        ),
    )


def _tool_kind(
    tool_call: dict[str, JsonValue],
    *,
    expected_call_id: str | None = None,
) -> str:
    unknown = set(tool_call) - _READ_ONLY_TOOL_KINDS - _TOOL_CALL_METADATA_KEYS
    branches = set(tool_call) & _READ_ONLY_TOOL_KINDS
    if unknown or len(branches) != 1:
        raise ValueError('tool call must contain one exact admitted oneof branch')
    tool_kind = next(iter(branches))
    if not isinstance(tool_call[tool_kind], dict):
        raise ValueError('tool call is outside the read-only fixture surface')
    contexts = tool_call.get('hookAdditionalContexts')
    if contexts is not None and not isinstance(contexts, list):
        raise ValueError('tool call contexts have an invalid shape')
    inner_call_id = tool_call.get('toolCallId')
    if inner_call_id is not None and (
        not isinstance(inner_call_id, str)
        or not inner_call_id
        or (expected_call_id is not None and inner_call_id != expected_call_id)
    ):
        raise ValueError('inner and outer tool call IDs differ')
    for timestamp_name in ('startedAtMs', 'completedAtMs'):
        timestamp = tool_call.get(timestamp_name)
        if timestamp is not None and not (
            isinstance(timestamp, int)
            and not isinstance(timestamp, bool)
            and timestamp >= 0
            or isinstance(timestamp, str)
            and timestamp.isascii()
            and timestamp.isdecimal()
        ):
            raise ValueError('tool call timestamp has an invalid shape')
    return tool_kind


def _materialize_workspace(
    client: CursorProtocolFixtureGuestRpcClient,
    *,
    paths: _RuntimePaths,
    limits: CursorProtocolFixtureLimits,
) -> tuple[CursorMaterializedWorkspaceFile, ...]:
    workspace = paths.physical('/run/vaxreplay/workspace')
    try:
        _create_runtime_layout(paths)
        cursor = 0
        seen_cursors: set[int] = set()
        listed: list[LogicalFileResult] = []
        while True:
            if cursor in seen_cursors:
                raise ValueError('repeated cursor')
            seen_cursors.add(cursor)
            page = client.list_workspace(cursor=cursor, limit=limits.workspace_list_page_size)
            listed.extend(page.files)
            if len(listed) > limits.maximum_workspace_files:
                raise ValueError('too many files')
            if page.next_cursor is None:
                break
            if page.next_cursor <= cursor:
                raise ValueError('non-increasing cursor')
            cursor = page.next_cursor
        if not listed:
            raise ValueError('empty workspace')
        paths_seen: set[str] = set()
        total_bytes = 0
        materialized: list[CursorMaterializedWorkspaceFile] = []
        for item in listed:
            logical_path = _validate_workspace_path(item.path)
            if item.path in paths_seen:
                raise ValueError('duplicate path')
            paths_seen.add(item.path)
            total_bytes += item.byte_count
            if total_bytes > limits.maximum_workspace_bytes:
                raise ValueError('workspace too large')
            target = workspace.joinpath(*logical_path.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise ValueError('workspace target exists')
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            no_follow = getattr(os, 'O_NOFOLLOW', None)
            if not isinstance(no_follow, int):
                raise ValueError('O_NOFOLLOW unavailable')
            descriptor = os.open(target, flags | no_follow, 0o400)
            digest = hashlib.sha256()
            written = 0
            try:
                while written < item.byte_count:
                    limit = min(limits.workspace_read_chunk_bytes, item.byte_count - written)
                    chunk = client.read_workspace(item.path, offset=written, limit=limit)
                    if chunk.offset != written or chunk.byte_count == 0 or chunk.byte_count > limit:
                        raise ValueError('invalid read chunk')
                    if chunk.eof != (written + chunk.byte_count == item.byte_count):
                        raise ValueError('invalid eof')
                    _write_all(descriptor, chunk.content)
                    digest.update(chunk.content)
                    written += chunk.byte_count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if written != item.byte_count or not hmac.compare_digest(
                digest.hexdigest(),
                item.sha256,
            ):
                raise ValueError('workspace bytes differ from listing')
            os.chmod(target, 0o400, follow_symlinks=False)
            materialized.append(
                CursorMaterializedWorkspaceFile(
                    path=item.path,
                    media_type=item.media_type,
                    sha256=item.sha256,
                    byte_count=item.byte_count,
                )
            )
        for directory in sorted(
            (path for path in workspace.rglob('*') if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o500, follow_symlinks=False)
        os.chmod(workspace, 0o500, follow_symlinks=False)
        return tuple(sorted(materialized, key=lambda item: item.path.encode('utf-8')))
    except CursorProtocolFixtureError:
        raise
    except Exception:
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.WORKSPACE_REJECTED) from None


def _create_runtime_layout(paths: _RuntimePaths) -> None:
    base = paths.physical('/run/vaxreplay')
    if base.exists():
        if base.is_symlink() or not base.is_dir() or any(base.iterdir()):
            raise ValueError('runtime base is not fresh')
    else:
        base.mkdir(mode=0o700, parents=True)
    for logical in (
        '/run/vaxreplay/workspace',
        '/run/vaxreplay/scratch',
        CURSOR_EMPTY_HOME_PATH,
        CURSOR_TMP_PATH,
        CURSOR_DATA_PATH,
        CURSOR_CONFIG_PATH,
    ):
        path = paths.physical(logical)
        if path.exists() or path.is_symlink():
            raise ValueError('runtime path already exists')
        path.mkdir(mode=0o700)


def _validate_workspace_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or '..' in path.parts
        or '.' in path.parts
        or path.as_posix() != value
        or any(part in {'.git', '.cursor', '.envrc', 'mcp.json'} for part in path.parts)
    ):
        raise ValueError('unsafe workspace path')
    return path


def _render_cursor_prompt(
    invocation: AgenticTaskInvocation,
    workspace_files: tuple[CursorMaterializedWorkspaceFile, ...],
    *,
    submission_schema_bytes: bytes,
) -> bytes:
    return canonical_json_bytes(
        {
            'schema_version': 'vaxreplay.cursor-protocol-fixture-prompt.dev-v0.1',
            'instructions': (
                'Complete the bound VaxReplay task using only the complete read-only workspace. '
                'The intended Cursor surface is ask mode with only read, list, grep, and glob. Do '
                'not use writes, shell, internet, LAN, ambient configuration, credentials, '
                'persistence, plugins, MCP, skills, or files outside the snapshot. Emit no '
                'narration before tool calls. Return only one JSON object matching '
                'submission_schema. The protocol fixture validates and terminally submits it.'
            ),
            'task_invocation': invocation.model_dump(mode='json'),
            'workspace_inventory': [item.model_dump(mode='json') for item in workspace_files],
            'submission_schema': json.loads(submission_schema_bytes),
        }
    )


def _sealed_vendor_environment(paths: _RuntimePaths) -> dict[str, str]:
    return {
        'PATH': '/opt/vaxreplay/bin:/usr/bin:/bin',
        'HOME': str(paths.physical(CURSOR_EMPTY_HOME_PATH)),
        'TMPDIR': str(paths.physical(CURSOR_TMP_PATH)),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'NO_COLOR': '1',
        'TERM': 'dumb',
        'CURSOR_DATA_DIR': str(paths.physical(CURSOR_DATA_PATH)),
        'CURSOR_CONFIG_DIR': str(paths.physical(CURSOR_CONFIG_PATH)),
        'CURSOR_AGENT_CLI_AUTHLESS_MODE': 'true',
        'CURSOR_AGENT_DISABLE_DEBUG_LOG': '1',
        'DIRENV_DISABLE': '1',
        'DISABLE_AUTOUPDATER': '1',
        'AGENT_CLI_CREDENTIAL_STORE': 'memory',
    }


def _identity_environment(
    *,
    home: Path,
    scratch: Path,
    data: Path,
    config: Path,
) -> dict[str, str]:
    return {
        'PATH': '/usr/bin:/bin',
        'HOME': str(home),
        'TMPDIR': str(scratch),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'NO_COLOR': '1',
        'TERM': 'dumb',
        'CURSOR_DATA_DIR': str(data),
        'CURSOR_CONFIG_DIR': str(config),
        'CURSOR_AGENT_CLI_AUTHLESS_MODE': 'true',
        'CURSOR_AGENT_DISABLE_DEBUG_LOG': '1',
        'DIRENV_DISABLE': '1',
        'DISABLE_AUTOUPDATER': '1',
        'AGENT_CLI_CREDENTIAL_STORE': 'memory',
    }


def _verify_vendor_executable(
    path: Path,
    config: CursorProtocolFixtureConfig,
) -> None:
    try:
        digest, byte_count = _measure_regular_executable(path)
    except (OSError, ValueError):
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.VENDOR_EXECUTABLE_REJECTED) from None
    if byte_count != config.vendor_executable_byte_count or not hmac.compare_digest(
        digest,
        config.vendor_executable_sha256,
    ):
        raise CursorProtocolFixtureError(CursorProtocolFixtureFailureCode.VENDOR_EXECUTABLE_REJECTED)


def _measure_regular_executable(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise ValueError('O_NOFOLLOW is unavailable')
    descriptor = -1
    try:
        descriptor = os.open(path, flags | no_follow)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o022
            or not before.st_mode & 0o111
            or not 0 < before.st_size <= 2 * 1024 * 1024 * 1024
        ):
            raise ValueError('bad executable metadata')
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError('bad executable identity')
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), before.st_size


def _write_all(descriptor: int, body: bytes) -> None:
    offset = 0
    while offset < len(body):
        written = os.write(descriptor, body[offset:])
        if written <= 0:
            raise OSError('short write')
        offset += written


def _load_unique_json(body: bytes) -> JsonValue:
    def pairs_hook(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        output: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError('duplicate JSON object key')
            output[key] = value
        return output

    def reject_constant(_value: str) -> JsonValue:
        raise ValueError('non-finite JSON number')

    try:
        return cast(
            JsonValue,
            json.loads(
                body.decode('utf-8'),
                object_pairs_hook=pairs_hook,
                parse_constant=reject_constant,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError('invalid JSON') from None


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


__all__ = [
    'CURSOR_PROTOCOL_FIXTURE_ID',
    'CURSOR_PROTOCOL_FIXTURE_VERSION',
    'CURSOR_SUPPORTED_VENDOR_VERSION',
    'CURSOR_VENDOR_EXECUTABLE_PATH',
    'CursorProtocolEventReceipt',
    'CursorProtocolFixtureConfig',
    'CursorProtocolFixtureError',
    'CursorProtocolFixtureFailureCode',
    'CursorProtocolFixtureLimits',
    'CursorProtocolFixtureReceipt',
    'CursorVendorIdentityEvidence',
    'capture_cursor_vendor_identity',
    'cursor_protocol_fixture_config_sha256',
    'cursor_vendor_argv_template',
    'render_cursor_vendor_argv',
    'require_cursor_protocol_fixture_binding',
    'run_cursor_protocol_fixture',
]
