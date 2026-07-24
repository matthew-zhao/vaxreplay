"""Fail-closed Claude Code development adapter for a sealed Lane A guest.

The adapter reconstructs a read-only workspace through authenticated guest RPC, runs one exact
Claude Code executable in print mode with only the built-in ``Read`` tool, serves a loopback-only
Anthropic Messages compatibility endpoint backed by ``model_generate``, and terminally submits one
task-bound response.  It is development machinery: no Linux payload or KVM qualification is
claimed, and the pinned local macOS binary could not be exercised without ambient administrator
policy on the development host (see ``docs/claude_code_guest_adapter.md``).
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import http.server
import json
import os
import platform
import re
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, Self, cast

from pydantic import Field, JsonValue, TypeAdapter, ValidationError, field_validator, model_validator

from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelResponse
from vaxreplay.agentic.guest_rpc import (
    ListWorkspaceResult,
    LogicalFileResult,
    ReadWorkspaceResult,
    SubmitResult,
)
from vaxreplay.agentic.headless_guest_adapter import (
    CLAUDE_MESSAGES_SHIM_PORT,
    MODEL_SELECTOR_TOKEN,
    SUBMISSION_SCHEMA_JSON_TOKEN,
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

CLAUDE_CODE_GUEST_ADAPTER_CONFIG_SCHEMA_VERSION = 'vaxreplay.claude-code-guest-adapter-config.dev-v0.1'
CLAUDE_CODE_GUEST_ADAPTER_RECEIPT_SCHEMA_VERSION = 'vaxreplay.claude-code-guest-adapter-receipt.dev-v0.1'
CLAUDE_CODE_VENDOR_IDENTITY_EVIDENCE_SCHEMA_VERSION = 'vaxreplay.claude-code-vendor-identity-evidence.dev-v0.1'
CLAUDE_CODE_GUEST_ADAPTER_ID = 'vaxreplay-claude-code-guest-adapter'
CLAUDE_CODE_GUEST_ADAPTER_VERSION = 'dev-v0.1'
CLAUDE_CODE_SUPPORTED_VENDOR_VERSION = '2.1.195 (Claude Code)'
CLAUDE_CODE_VENDOR_EXECUTABLE_PATH = '/opt/vaxreplay/vendor/claude-code/claude'
CLAUDE_CODE_EMPTY_HOME_PATH = '/run/vaxreplay/scratch/home'
CLAUDE_CODE_TMP_PATH = '/run/vaxreplay/scratch/tmp'
CLAUDE_CODE_EMPTY_MANAGED_SETTINGS_PATH = '/run/vaxreplay/control/empty-managed-settings.json'
CLAUDE_CODE_EMPTY_MCP_CONFIG_JSON = '{"mcpServers":{}}'
CLAUDE_CODE_LOOPBACK_API_KEY_SENTINEL = 'vaxreplay-loopback-not-a-provider-credential-v1'
CLAUDE_CODE_ANTHROPIC_VERSION = '2023-06-01'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MODEL_SELECTOR_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,499}$')
_READ_TOOL_NAME = 'Read'
_EMPTY_MANAGED_SETTINGS_BYTES = b'{}\n'


class ClaudeCodeGuestAdapterFailureCode(str, enum.Enum):
    BINDING_REJECTED = 'binding_rejected'
    RUNTIME_LAYOUT_REJECTED = 'runtime_layout_rejected'
    VENDOR_EXECUTABLE_REJECTED = 'vendor_executable_rejected'
    WORKSPACE_REJECTED = 'workspace_rejected'
    SHIM_REJECTED = 'shim_rejected'
    LAUNCH_REJECTED = 'launch_rejected'
    SUBPROCESS_REJECTED = 'subprocess_rejected'
    OUTPUT_REJECTED = 'output_rejected'
    SUBMISSION_REJECTED = 'submission_rejected'


class ClaudeCodeGuestAdapterError(RuntimeError):
    """Content-free failure suitable for the guest appliance boundary."""

    def __init__(self, code: ClaudeCodeGuestAdapterFailureCode):
        super().__init__(code.value)
        self.code = code


class ClaudeCodeGuestAdapterLimits(StrictModel):
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
    maximum_vendor_stdout_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=1_024,
        le=64 * 1024 * 1024,
    )
    maximum_vendor_stderr_bytes: int = Field(
        default=1024 * 1024,
        ge=1_024,
        le=16 * 1024 * 1024,
    )
    maximum_shim_request_bytes: int = Field(
        default=8 * 1024 * 1024,
        ge=1_024,
        le=64 * 1024 * 1024,
    )
    maximum_shim_calls: int = Field(default=10, ge=1, le=100)
    maximum_shim_decision_tokens: int = Field(default=4_096, ge=128, le=32_768)
    maximum_anthropic_request_tokens: int = Field(default=32_768, ge=128, le=200_000)
    vendor_wall_seconds: int = Field(default=900, ge=1, le=7_200)


class ClaudeCodeGuestAdapterConfig(StrictModel):
    """Hash-bound runtime facts not already carried by the generic headless contract."""

    schema_version: Literal['vaxreplay.claude-code-guest-adapter-config.dev-v0.1'] = (
        CLAUDE_CODE_GUEST_ADAPTER_CONFIG_SCHEMA_VERSION
    )
    adapter_id: Literal['vaxreplay-claude-code-guest-adapter'] = CLAUDE_CODE_GUEST_ADAPTER_ID
    adapter_version: Literal['dev-v0.1'] = CLAUDE_CODE_GUEST_ADAPTER_VERSION
    headless_adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    submitted_harness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_path: Literal['/opt/vaxreplay/vendor/claude-code/claude'] = CLAUDE_CODE_VENDOR_EXECUTABLE_PATH
    vendor_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_byte_count: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    supported_vendor_version: Literal['2.1.195 (Claude Code)'] = CLAUDE_CODE_SUPPORTED_VENDOR_VERSION
    messages_shim_host: Literal['127.0.0.1'] = '127.0.0.1'
    messages_shim_port: Literal[43124] = CLAUDE_MESSAGES_SHIM_PORT
    loopback_api_key_sentinel: Literal['vaxreplay-loopback-not-a-provider-credential-v1'] = (
        CLAUDE_CODE_LOOPBACK_API_KEY_SENTINEL
    )
    empty_managed_settings_path: Literal['/run/vaxreplay/control/empty-managed-settings.json'] = (
        CLAUDE_CODE_EMPTY_MANAGED_SETTINGS_PATH
    )
    limits: ClaudeCodeGuestAdapterLimits = ClaudeCodeGuestAdapterLimits()
    workspace_materialization_uses_only_authenticated_guest_rpc: Literal[True] = True
    model_transport_is_loopback_only_anthropic_messages_shim: Literal[True] = True
    final_submission_uses_authenticated_guest_rpc: Literal[True] = True
    inherited_environment_allowed: Literal[False] = False
    provider_credentials_in_guest_allowed: Literal[False] = False
    loopback_api_key_is_fixed_public_protocol_sentinel: Literal[True] = True
    ambient_claude_home_allowed: Literal[False] = False
    shell_command_construction_allowed: Literal[False] = False
    local_tool_surface: tuple[Literal['Read'], ...] = ('Read',)
    session_persistence_allowed: Literal[False] = False
    mcp_plugins_hooks_and_skills_allowed: Literal[False] = False
    linux_kvm_qualified: Literal[False] = False
    actual_pinned_macos_claude_end_to_end_validated: Literal[False] = False
    actual_pinned_linux_claude_end_to_end_validated: Literal[False] = False
    development_only: Literal[True] = True


class ClaudeCodeVendorIdentityEvidence(StrictModel):
    schema_version: Literal['vaxreplay.claude-code-vendor-identity-evidence.dev-v0.1'] = (
        CLAUDE_CODE_VENDOR_IDENTITY_EVIDENCE_SCHEMA_VERSION
    )
    executable_path: str = Field(min_length=2, max_length=4096)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    executable_byte_count: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    version_argv_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stdout_bytes: int = Field(ge=0, le=4096)
    version_stderr_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stderr_bytes: int = Field(ge=0, le=4096)
    reported_version: Literal['2.1.195 (Claude Code)'] = CLAUDE_CODE_SUPPORTED_VENDOR_VERSION
    observed_os: str = Field(min_length=1, max_length=100)
    observed_architecture: str = Field(min_length=1, max_length=100)
    executable_was_regular_nonsymlink_file: Literal[True] = True
    version_environment_was_exact_and_credential_free: Literal[True] = True
    evidence_is_not_dependency_closure: Literal[True] = True
    evidence_is_not_linux_kvm_qualification: Literal[True] = True
    development_only: Literal[True] = True

    @field_validator('executable_path')
    @classmethod
    def validate_executable_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or path.as_posix() != value:
            raise ValueError('measured Claude executable path must be absolute and normalized')
        return value


class ClaudeCodeMaterializedWorkspaceFile(StrictModel):
    path: str = Field(min_length=1, max_length=4096)
    media_type: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)


class ClaudeCodeShimExchangeReceipt(StrictModel):
    call_index: int = Field(ge=0)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    request_bytes: int = Field(gt=0)
    request_headers_sha256: str = Field(pattern=_SHA256_PATTERN)
    forwarded_request_sha256: str = Field(pattern=_SHA256_PATTERN)
    forwarded_request_bytes: int = Field(gt=0)
    prior_tool_use_count: int = Field(ge=0, le=10_000)
    prior_tool_result_count: int = Field(ge=0, le=10_000)
    decision_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_bytes: int = Field(gt=0)
    sse_sha256: str = Field(pattern=_SHA256_PATTERN)
    sse_bytes: int = Field(gt=0)
    resolved_model_id_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_content_type: Literal['text', 'tool_use']


class ClaudeCodeGuestAdapterReceipt(StrictModel):
    schema_version: Literal['vaxreplay.claude-code-guest-adapter-receipt.dev-v0.1'] = (
        CLAUDE_CODE_GUEST_ADAPTER_RECEIPT_SCHEMA_VERSION
    )
    adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    headless_adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    submitted_harness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_model_selector_sha256: str = Field(pattern=_SHA256_PATTERN)
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
    shim_exchanges: tuple[ClaudeCodeShimExchangeReceipt, ...] = Field(min_length=1)
    shim_exchange_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_duration_ms: int = Field(ge=0)
    vendor_stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_stdout_bytes: int = Field(gt=0)
    vendor_stderr_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_stderr_bytes: int = Field(ge=0)
    resolved_model_id_sha256: str = Field(pattern=_SHA256_PATTERN)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    submit_result: SubmitResult
    workspace_was_complete_materialized_snapshot: Literal[True] = True
    local_vendor_tool_surface_was_exactly_read_only: Literal[True] = True
    no_provider_credentials_or_ambient_user_config_in_subprocess: Literal[True] = True
    outer_guest_rpc_events_remain_authoritative: Literal[True] = True
    actual_pinned_macos_claude_end_to_end_validated: Literal[False] = False
    actual_pinned_linux_claude_end_to_end_validated: Literal[False] = False
    linux_kvm_qualified: Literal[False] = False
    development_only: Literal[True] = True

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        exchange_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in self.shim_exchanges])
        if self.shim_exchange_log_sha256 != _sha256(exchange_bytes):
            raise ValueError('shim exchange log hash does not match its exact receipts')
        if self.submit_result.submission_sha256 != self.submission_sha256:
            raise ValueError('submit result does not match the adapter submission')
        return self


class _ClaudeAssistantDecision(StrictModel):
    kind: Literal['assistant_text']
    text: str = Field(min_length=1, max_length=2_000_000)


class _ClaudeToolDecision(StrictModel):
    kind: Literal['tool_call']
    tool_name: Literal['Read']
    payload: dict[str, JsonValue]


type ClaudeCodeShimDecision = Annotated[
    _ClaudeAssistantDecision | _ClaudeToolDecision,
    Field(discriminator='kind'),
]
_CLAUDE_DECISION_ADAPTER = TypeAdapter(ClaudeCodeShimDecision)
_CLAUDE_DECISION_SCHEMA = _CLAUDE_DECISION_ADAPTER.json_schema()
_CLAUDE_SHIM_SYSTEM_PROMPT = (
    'You are the model behind the pinned Claude Code Anthropic Messages compatibility shim. '
    'The user message is the complete canonical Messages request emitted by Claude Code. Return '
    'exactly one JSON object matching the decision schema. Choose assistant_text for ordinary '
    'assistant output. Choose tool_call only for the Read tool that is present in the request; its '
    'payload must be the exact JSON argument object. Never invent another tool, provider route, '
    'credential, web action, or text outside the JSON object.\nDecision JSON Schema:\n'
    + canonical_json_bytes(_CLAUDE_DECISION_SCHEMA).decode('utf-8')
)


class _ClaudeMessagesRequest(StrictModel):
    model: str = Field(min_length=1, max_length=500)
    max_tokens: int = Field(ge=1, le=200_000)
    messages: tuple[dict[str, JsonValue], ...] = Field(min_length=1, max_length=10_000)
    system: str | tuple[dict[str, JsonValue], ...] | None = None
    tools: tuple[dict[str, JsonValue], ...] = Field(min_length=1, max_length=10)
    tool_choice: dict[str, JsonValue] | None = None
    metadata: dict[str, JsonValue] | None = None
    stream: Literal[True]
    temperature: int | float | None = None
    top_k: int | None = None
    top_p: int | float | None = None
    stop_sequences: tuple[str, ...] | None = None
    thinking: dict[str, JsonValue] | None = None
    output_config: dict[str, JsonValue] | None = None
    service_tier: str | None = None
    context_management: dict[str, JsonValue] | None = None


class _ClaudePrintResult(StrictModel):
    type: Literal['result']
    subtype: str
    is_error: bool
    duration_ms: int = Field(ge=0)
    duration_api_ms: int = Field(ge=0)
    num_turns: int = Field(ge=1)
    result: str | None = None
    structured_output: JsonValue | None = None
    stop_reason: str | None = None
    session_id: str = Field(min_length=1, max_length=500)
    total_cost_usd: int | float = Field(ge=0)
    usage: dict[str, JsonValue]
    model_usage: dict[str, dict[str, JsonValue]] = Field(alias='modelUsage')
    permission_denials: tuple[dict[str, JsonValue], ...]
    terminal_reason: str | None = None
    fast_mode_state: str | None = None
    uuid: str | None = None
    errors: tuple[str, ...] | None = None
    model: str | None = None


class ClaudeCodeGuestRpcClient(Protocol):
    def list_workspace(self, *, cursor: int = 0, limit: int = 100) -> ListWorkspaceResult: ...

    def read_workspace(self, path: str, *, offset: int = 0, limit: int) -> ReadWorkspaceResult: ...

    def model_generate(
        self,
        *,
        messages: tuple[AgenticModelMessage, ...],
        max_output_tokens: int,
        response_schema_sha256: str | None = None,
    ) -> AgenticModelResponse: ...

    def submit(self, submission: AgenticRuntimeSubmission) -> SubmitResult: ...


class ProcessRunner(Protocol):
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
            raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED)
        if self.root == Path('/'):
            return Path(logical)
        return self.root.joinpath(*path.parts[1:])


def claude_code_guest_adapter_config_sha256(config: ClaudeCodeGuestAdapterConfig) -> str:
    canonical = ClaudeCodeGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
    return _sha256(canonical_json_bytes(canonical))


def claude_code_vendor_argv_template() -> tuple[str, ...]:
    """Return the only Claude Code argv shape implemented by this adapter."""

    return (
        CLAUDE_CODE_VENDOR_EXECUTABLE_PATH,
        '--bare',
        '--safe-mode',
        '-p',
        '--input-format',
        'text',
        '--output-format',
        'json',
        '--no-session-persistence',
        '--no-chrome',
        '--disable-slash-commands',
        '--strict-mcp-config',
        '--mcp-config',
        CLAUDE_CODE_EMPTY_MCP_CONFIG_JSON,
        '--tools',
        _READ_TOOL_NAME,
        '--allowedTools',
        _READ_TOOL_NAME,
        '--permission-mode',
        'dontAsk',
        '--prompt-suggestions',
        'false',
        '--add-dir',
        WORKSPACE_PATH_TOKEN,
        '--json-schema',
        SUBMISSION_SCHEMA_JSON_TOKEN,
        '--model',
        MODEL_SELECTOR_TOKEN,
    )


def require_claude_code_guest_adapter_binding(
    *,
    config: ClaudeCodeGuestAdapterConfig,
    headless_config: HeadlessGuestAdapterConfig,
    submitted_manifest: SubmittedHarnessManifest,
) -> None:
    try:
        runtime = ClaudeCodeGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
        headless = HeadlessGuestAdapterConfig.model_validate_json(canonical_json_bytes(headless_config))
        submitted = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(submitted_manifest))
        require_headless_guest_adapter_binding(config=headless, manifest=submitted)
    except (TypeError, ValueError):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.BINDING_REJECTED) from None
    expected = (
        HarnessFamily.CLAUDE_CODE,
        HeadlessInvocationProtocol.CLAUDE_PRINT,
        HeadlessResponseChannel.BOUNDED_JSON_STDOUT,
        HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED,
        True,
        True,
        True,
        False,
        CLAUDE_CODE_VENDOR_EXECUTABLE_PATH,
        runtime.vendor_executable_sha256,
        CLAUDE_CODE_SUPPORTED_VENDOR_VERSION,
        claude_code_vendor_argv_template(),
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
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.BINDING_REJECTED)


def render_claude_code_vendor_argv(
    *,
    organizer_model_selector: str,
    submission_schema_json: str,
    paths: _RuntimePaths | None = None,
) -> tuple[str, ...]:
    if not _MODEL_SELECTOR_PATTERN.fullmatch(organizer_model_selector):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.BINDING_REJECTED)
    if not submission_schema_json or '\x00' in submission_schema_json:
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.BINDING_REJECTED)
    runtime_paths = paths or _RuntimePaths(Path('/'))
    substitutions = {
        CLAUDE_CODE_VENDOR_EXECUTABLE_PATH: str(runtime_paths.physical(CLAUDE_CODE_VENDOR_EXECUTABLE_PATH)),
        MODEL_SELECTOR_TOKEN: organizer_model_selector,
        WORKSPACE_PATH_TOKEN: str(runtime_paths.physical('/run/vaxreplay/workspace')),
        SUBMISSION_SCHEMA_JSON_TOKEN: submission_schema_json,
    }
    return tuple(substitutions.get(item, item) for item in claude_code_vendor_argv_template())


def capture_claude_code_vendor_identity(
    executable_path: Path,
    *,
    process_runner: ProcessRunner = run_bounded_process,
) -> ClaudeCodeVendorIdentityEvidence:
    """Measure a regular local Claude Code file and bounded ``--version`` output."""

    path_text = executable_path.as_posix()
    pure_path = PurePosixPath(path_text)
    if (
        not executable_path.is_absolute()
        or '..' in pure_path.parts
        or pure_path.as_posix() != path_text
        or executable_path.is_symlink()
    ):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED)
    try:
        executable_sha256, executable_bytes = _measure_regular_executable(executable_path)
        with tempfile.TemporaryDirectory(prefix='vaxreplay-claude-identity-') as temporary:
            temporary_root = Path(temporary)
            home = temporary_root / 'home'
            scratch = temporary_root / 'tmp'
            managed = temporary_root / 'empty-managed-settings.json'
            home.mkdir(mode=0o700)
            scratch.mkdir(mode=0o700)
            managed.write_bytes(_EMPTY_MANAGED_SETTINGS_BYTES)
            managed.chmod(0o400)
            environment = _identity_environment(home=home, scratch=scratch, managed=managed)
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
        if reported_version != CLAUDE_CODE_SUPPORTED_VENDOR_VERSION:
            raise ValueError('unsupported version')
        return ClaudeCodeVendorIdentityEvidence(
            executable_path=path_text,
            executable_sha256=executable_sha256,
            executable_byte_count=executable_bytes,
            version_argv_sha256=_sha256(canonical_json_bytes((path_text, '--version'))),
            version_stdout_sha256=_sha256(result.stdout),
            version_stdout_bytes=len(result.stdout),
            version_stderr_sha256=_sha256(result.stderr),
            version_stderr_bytes=len(result.stderr),
            reported_version=cast(Literal['2.1.195 (Claude Code)'], reported_version),
            observed_os=platform.system(),
            observed_architecture=platform.machine(),
        )
    except ClaudeCodeGuestAdapterError:
        raise
    except (OSError, UnicodeDecodeError, ValueError):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED) from None


def run_claude_code_guest_adapter(
    client: ClaudeCodeGuestRpcClient,
    *,
    task_invocation: AgenticTaskInvocation,
    organizer_model_selector: str,
    config: ClaudeCodeGuestAdapterConfig,
    headless_config: HeadlessGuestAdapterConfig,
    submitted_manifest: SubmittedHarnessManifest,
    process_runner: ProcessRunner = run_bounded_process,
    _guest_root: Path = Path('/'),
) -> ClaudeCodeGuestAdapterReceipt:
    """Run one bounded Claude Code attempt after authenticated guest bootstrap."""

    try:
        invocation = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(task_invocation))
        runtime = ClaudeCodeGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
        require_claude_code_guest_adapter_binding(
            config=runtime,
            headless_config=headless_config,
            submitted_manifest=submitted_manifest,
        )
        if not _guest_root.is_absolute() or _guest_root.is_symlink():
            raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED)
        paths = _RuntimePaths(_guest_root)
        vendor_path = paths.physical(runtime.vendor_executable_path)
        _verify_vendor_executable(vendor_path, runtime)
        workspace_files = _materialize_workspace(client, paths=paths, limits=runtime.limits)
        managed_settings_path = paths.physical(runtime.empty_managed_settings_path)
        _write_control_file(managed_settings_path, _EMPTY_MANAGED_SETTINGS_BYTES)
        submission_schema = submission_json_schema_for_invocation(invocation)
        submission_schema_bytes = canonical_json_bytes(submission_schema)
        submission_schema_json = submission_schema_bytes.decode('utf-8')
        inventory_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in workspace_files])
        prompt = _render_claude_prompt(invocation, workspace_files)
        if len(prompt) > runtime.limits.maximum_prompt_bytes:
            raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.WORKSPACE_REJECTED)
        argv = render_claude_code_vendor_argv(
            organizer_model_selector=organizer_model_selector,
            submission_schema_json=submission_schema_json,
            paths=paths,
        )
        environment = _sealed_vendor_environment(paths)
        shim = _ClaudeMessagesShim(
            client,
            organizer_model_selector=organizer_model_selector,
            expected_submission_schema=submission_schema,
            limits=runtime.limits,
        )
        with _ClaudeMessagesServer(
            shim,
            port=runtime.messages_shim_port,
            api_key_sentinel=runtime.loopback_api_key_sentinel,
        ):
            try:
                process = process_runner(
                    argv,
                    input_bytes=prompt,
                    wall_seconds=runtime.limits.vendor_wall_seconds,
                    max_stdout_bytes=runtime.limits.maximum_vendor_stdout_bytes,
                    max_stderr_bytes=runtime.limits.maximum_vendor_stderr_bytes,
                    on_abort=shim.abort,
                    env=environment,
                    cwd=paths.physical('/run/vaxreplay/workspace'),
                )
            except (OSError, RuntimeError, ValueError):
                raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.LAUNCH_REJECTED) from None
        if shim.failure is not None:
            raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED)
        if (
            process.termination != 'exited'
            or process.exit_code != 0
            or process.stdout_truncated
            or process.stderr_truncated
        ):
            raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.SUBPROCESS_REJECTED)
        submission, resolved_model_id = _parse_claude_stdout(
            invocation,
            process.stdout,
            exchanges=shim.exchanges,
        )
        try:
            submit_result = client.submit(submission)
        except Exception:
            raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.SUBMISSION_REJECTED) from None
        submission_bytes = canonical_json_bytes(submission)
        exchanges = shim.exchanges
        exchange_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in exchanges])
        return ClaudeCodeGuestAdapterReceipt(
            adapter_config_sha256=claude_code_guest_adapter_config_sha256(runtime),
            headless_adapter_config_sha256=runtime.headless_adapter_config_sha256,
            submitted_harness_manifest_sha256=runtime.submitted_harness_manifest_sha256,
            task_invocation_sha256=agentic_task_invocation_sha256(invocation),
            organizer_model_selector_sha256=_sha256(organizer_model_selector.encode('utf-8')),
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
            shim_exchanges=exchanges,
            shim_exchange_log_sha256=_sha256(exchange_bytes),
            vendor_duration_ms=process.duration_ms,
            vendor_stdout_sha256=_sha256(process.stdout),
            vendor_stdout_bytes=len(process.stdout),
            vendor_stderr_sha256=_sha256(process.stderr),
            vendor_stderr_bytes=len(process.stderr),
            resolved_model_id_sha256=_sha256(resolved_model_id.encode('utf-8')),
            submission_sha256=_sha256(submission_bytes),
            submit_result=submit_result,
        )
    except ClaudeCodeGuestAdapterError:
        raise
    except (TypeError, ValueError, ValidationError):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.BINDING_REJECTED) from None


class _ClaudeMessagesShim:
    def __init__(
        self,
        client: ClaudeCodeGuestRpcClient,
        *,
        organizer_model_selector: str,
        expected_submission_schema: dict[str, object],
        limits: ClaudeCodeGuestAdapterLimits,
    ) -> None:
        self._client = client
        self._model_selector = organizer_model_selector
        self._expected_submission_schema = expected_submission_schema
        self._limits = limits
        self._lock = threading.Lock()
        self._exchanges: list[ClaudeCodeShimExchangeReceipt] = []
        self._request_hashes: set[str] = set()
        self._resolved_model_id: str | None = None
        self._failure: ClaudeCodeGuestAdapterFailureCode | None = None

    @property
    def failure(self) -> ClaudeCodeGuestAdapterFailureCode | None:
        with self._lock:
            return self._failure

    @property
    def exchanges(self) -> tuple[ClaudeCodeShimExchangeReceipt, ...]:
        with self._lock:
            return tuple(self._exchanges)

    @property
    def maximum_request_bytes(self) -> int:
        return self._limits.maximum_shim_request_bytes

    def abort(self) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = ClaudeCodeGuestAdapterFailureCode.SUBPROCESS_REJECTED

    def reject(self) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED

    def handle(self, body: bytes, *, request_headers_sha256: str) -> bytes:
        with self._lock:
            body_sha256 = _sha256(body)
            if (
                self._failure is not None
                or len(self._exchanges) >= self._limits.maximum_shim_calls
                or body_sha256 in self._request_hashes
            ):
                self._failure = ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED
                raise ValueError('shim rejected')
            call_index = len(self._exchanges)
            try:
                payload = _load_unique_json(body)
                if not isinstance(payload, dict):
                    raise ValueError('Messages request is not an object')
                request = _ClaudeMessagesRequest.model_validate_json(canonical_json_bytes(payload))
                prior_tool_uses, prior_tool_results = _validate_messages_request(
                    request,
                    model_selector=self._model_selector,
                    expected_submission_schema=self._expected_submission_schema,
                    maximum_tokens=self._limits.maximum_anthropic_request_tokens,
                )
                forwarded_request = canonical_json_bytes(request)
                response = self._client.model_generate(
                    messages=(
                        AgenticModelMessage(role='system', content=_CLAUDE_SHIM_SYSTEM_PROMPT),
                        AgenticModelMessage(
                            role='user',
                            content=forwarded_request.decode('utf-8'),
                        ),
                    ),
                    max_output_tokens=self._limits.maximum_shim_decision_tokens,
                    response_schema_sha256=None,
                )
                if response.stop_reason != 'completed':
                    raise ValueError('model response incomplete')
                if self._resolved_model_id is not None and response.resolved_model_id != self._resolved_model_id:
                    raise ValueError('resolved model changed within one attempt')
                decision_payload = _load_unique_json(response.content.encode('utf-8'))
                decision = _CLAUDE_DECISION_ADAPTER.validate_json(
                    canonical_json_bytes(decision_payload),
                    strict=True,
                )
                sse, content_type = _render_messages_sse(
                    decision=decision,
                    call_index=call_index,
                    response=response,
                )
                decision_bytes = canonical_json_bytes(decision)
                receipt = ClaudeCodeShimExchangeReceipt(
                    call_index=call_index,
                    request_sha256=body_sha256,
                    request_bytes=len(body),
                    request_headers_sha256=request_headers_sha256,
                    forwarded_request_sha256=_sha256(forwarded_request),
                    forwarded_request_bytes=len(forwarded_request),
                    prior_tool_use_count=prior_tool_uses,
                    prior_tool_result_count=prior_tool_results,
                    decision_sha256=_sha256(decision_bytes),
                    decision_bytes=len(decision_bytes),
                    sse_sha256=_sha256(sse),
                    sse_bytes=len(sse),
                    resolved_model_id_sha256=_sha256(response.resolved_model_id.encode('utf-8')),
                    response_content_type=content_type,
                )
            except Exception:
                self._failure = ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED
                raise ValueError('shim rejected') from None
            self._request_hashes.add(body_sha256)
            self._resolved_model_id = response.resolved_model_id
            self._exchanges.append(receipt)
            return sse


class _ClaudeMessagesServer:
    def __init__(
        self,
        shim: _ClaudeMessagesShim,
        *,
        port: int,
        api_key_sentinel: str,
    ) -> None:
        self._shim = shim
        self._port = port
        self._api_key_sentinel = api_key_sentinel
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _ClaudeMessagesServer:
        shim = self._shim
        port = self._port
        api_key_sentinel = self._api_key_sentinel

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                host_headers = self.headers.get_all('Host', failobj=[])
                content_type_headers = self.headers.get_all('Content-Type', failobj=[])
                length_headers = self.headers.get_all('Content-Length', failobj=[])
                api_key_headers = self.headers.get_all('x-api-key', failobj=[])
                version_headers = self.headers.get_all('anthropic-version', failobj=[])
                if (
                    self.path != '/v1/messages'
                    or host_headers != [f'127.0.0.1:{port}']
                    or len(content_type_headers) != 1
                    or content_type_headers[0].partition(';')[0].strip().lower() != 'application/json'
                    or len(length_headers) != 1
                    or api_key_headers != [api_key_sentinel]
                    or version_headers != [CLAUDE_CODE_ANTHROPIC_VERSION]
                    or self.headers.get('Transfer-Encoding') is not None
                    or self.headers.get('Content-Encoding') is not None
                    or self.headers.get('Authorization') is not None
                    or self.headers.get('Proxy-Authorization') is not None
                ):
                    shim.reject()
                    self._reject()
                    return
                try:
                    length = int(length_headers[0])
                except ValueError:
                    length = -1
                if not 0 < length <= shim.maximum_request_bytes:
                    shim.reject()
                    self._reject()
                    return
                try:
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError('short body')
                    header_digest = _request_headers_sha256(self.headers)
                    response = shim.handle(body, request_headers_sha256=header_digest)
                except (OSError, ValueError):
                    self._reject()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.send_header('request-id', f'req_vaxreplay_{len(shim.exchanges) - 1:08d}')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
                self.close_connection = True

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reject_unexpected_method()

            def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reject_unexpected_method()

            def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reject_unexpected_method()

            def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reject_unexpected_method()

            def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reject_unexpected_method()

            def do_CONNECT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._reject_unexpected_method()

            def _reject_unexpected_method(self) -> None:
                shim.reject()
                self._reject()

            def _reject(self) -> None:
                body = b'{"type":"error","error":{"type":"invalid_request_error","message":"request rejected"}}'
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass
                self.close_connection = True

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        class LoopbackHTTPServer(http.server.HTTPServer):
            allow_reuse_address = True

        try:
            server = LoopbackHTTPServer(('127.0.0.1', port), Handler, bind_and_activate=True)
        except OSError:
            raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED) from None
        if server.server_address != ('127.0.0.1', port):
            server.server_close()
            raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED)
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={'poll_interval': 0.01},
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _validate_messages_request(
    request: _ClaudeMessagesRequest,
    *,
    model_selector: str,
    expected_submission_schema: dict[str, object],
    maximum_tokens: int,
) -> tuple[int, int]:
    if request.model != model_selector or request.max_tokens > maximum_tokens:
        raise ValueError('Messages route or token bound differs')
    if len(request.tools) != 1:
        raise ValueError('Claude tool surface must contain exactly Read')
    tool = request.tools[0]
    allowed_tool_keys = {'name', 'description', 'input_schema', 'cache_control'}
    if (
        set(tool) - allowed_tool_keys
        or tool.get('name') != _READ_TOOL_NAME
        or not isinstance(tool.get('description'), str)
        or not isinstance(tool.get('input_schema'), dict)
    ):
        raise ValueError('Claude tool surface differs from the admitted Read tool')
    if request.tool_choice is not None:
        if set(request.tool_choice) - {'type', 'disable_parallel_tool_use'}:
            raise ValueError('unexpected tool choice field')
        if request.tool_choice.get('type') != 'auto':
            raise ValueError('only automatic Read selection is admitted')
        disable_parallel = request.tool_choice.get('disable_parallel_tool_use')
        if disable_parallel not in {None, True}:
            raise ValueError('parallel tools are not admitted')
    if request.output_config is not None:
        if set(request.output_config) != {'format'}:
            raise ValueError('unexpected structured-output config')
        output_format = request.output_config.get('format')
        if not isinstance(output_format, dict) or set(output_format) != {'type', 'schema'}:
            raise ValueError('unexpected structured-output format')
        if output_format.get('type') != 'json_schema':
            raise ValueError('unexpected structured-output type')
        if canonical_json_bytes(output_format.get('schema')) != canonical_json_bytes(expected_submission_schema):
            raise ValueError('Claude request changed the task-bound submission schema')
    if request.metadata is not None and set(request.metadata) - {'user_id'}:
        raise ValueError('unexpected Messages metadata')
    if request.context_management is not None:
        raise ValueError('server-side context management is not admitted')

    tool_uses = 0
    tool_results = 0
    seen_tool_ids: set[str] = set()
    for message in request.messages:
        if set(message) != {'role', 'content'} or message.get('role') not in {'user', 'assistant'}:
            raise ValueError('unexpected Messages conversation shape')
        content = message.get('content')
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise ValueError('unexpected Messages content')
        for block in content:
            if not isinstance(block, dict):
                raise ValueError('unexpected Messages content block')
            block_type = block.get('type')
            if block_type == 'text':
                if not isinstance(block.get('text'), str):
                    raise ValueError('invalid text block')
            elif block_type in {'thinking', 'redacted_thinking'}:
                if message.get('role') != 'assistant':
                    raise ValueError('thinking block in user message')
            elif block_type == 'tool_use':
                identifier = block.get('id')
                if (
                    message.get('role') != 'assistant'
                    or block.get('name') != _READ_TOOL_NAME
                    or not isinstance(identifier, str)
                    or not identifier
                    or identifier in seen_tool_ids
                    or not isinstance(block.get('input'), dict)
                ):
                    raise ValueError('unadmitted tool use')
                seen_tool_ids.add(identifier)
                tool_uses += 1
            elif block_type == 'tool_result':
                tool_use_id = block.get('tool_use_id')
                if (
                    message.get('role') != 'user'
                    or not isinstance(tool_use_id, str)
                    or tool_use_id not in seen_tool_ids
                ):
                    raise ValueError('unbound tool result')
                tool_results += 1
            else:
                raise ValueError('unadmitted Messages content block')
    if tool_results > tool_uses or tool_uses - tool_results > 1:
        raise ValueError('tool transcript is not sequential')
    return tool_uses, tool_results


def _render_messages_sse(
    *,
    decision: ClaudeCodeShimDecision,
    call_index: int,
    response: AgenticModelResponse,
) -> tuple[bytes, Literal['text', 'tool_use']]:
    message_id = f'msg_vaxreplay_{call_index:08d}'
    start = {
        'type': 'message_start',
        'message': {
            'id': message_id,
            'type': 'message',
            'role': 'assistant',
            'content': [],
            'model': response.resolved_model_id,
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'output_tokens': 0,
            },
        },
    }
    if isinstance(decision, _ClaudeAssistantDecision):
        content_type: Literal['text', 'tool_use'] = 'text'
        block_start = {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'text', 'text': ''},
        }
        block_delta = {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'text_delta', 'text': decision.text},
        }
        stop_reason = 'end_turn'
    else:
        content_type = 'tool_use'
        block_start = {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {
                'type': 'tool_use',
                'id': f'toolu_vaxreplay_{call_index:08d}',
                'name': _READ_TOOL_NAME,
                'input': {},
            },
        }
        block_delta = {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {
                'type': 'input_json_delta',
                'partial_json': canonical_json_bytes(decision.payload).decode('utf-8'),
            },
        }
        stop_reason = 'tool_use'
    events = (
        ('message_start', start),
        ('content_block_start', block_start),
        ('content_block_delta', block_delta),
        ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
        (
            'message_delta',
            {
                'type': 'message_delta',
                'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
                'usage': {'output_tokens': response.usage.output_tokens},
            },
        ),
        ('message_stop', {'type': 'message_stop'}),
    )
    return b''.join(_sse_event(name, event) for name, event in events), content_type


def _sse_event(name: str, event: Mapping[str, JsonValue]) -> bytes:
    return b'event: ' + name.encode('ascii') + b'\ndata: ' + canonical_json_bytes(event) + b'\n\n'


def _parse_claude_stdout(
    invocation: AgenticTaskInvocation,
    stdout: bytes,
    *,
    exchanges: tuple[ClaudeCodeShimExchangeReceipt, ...],
) -> tuple[AgenticRuntimeSubmission, str]:
    try:
        payload = _load_unique_json(stdout)
        if not isinstance(payload, dict) or 'structured_output' not in payload:
            raise ValueError('missing structured output')
        wrapper = _ClaudePrintResult.model_validate_json(canonical_json_bytes(payload))
        if (
            wrapper.subtype != 'success'
            or wrapper.is_error
            or wrapper.permission_denials
            or wrapper.errors not in {None, ()}
            or wrapper.num_turns != len(exchanges)
        ):
            raise ValueError('Claude result wrapper reports an unadmitted outcome')
        resolved_hashes = {item.resolved_model_id_sha256 for item in exchanges}
        if len(resolved_hashes) != 1 or len(wrapper.model_usage) != 1:
            raise ValueError('Claude result wrapper reports multiple model routes')
        resolved_model_id = next(iter(wrapper.model_usage))
        if _sha256(resolved_model_id.encode('utf-8')) not in resolved_hashes:
            raise ValueError('Claude result wrapper model differs from shim evidence')
        submission = parse_submission_for_invocation(
            invocation,
            canonical_json_bytes(wrapper.structured_output),
        )
        return submission, resolved_model_id
    except ClaudeCodeGuestAdapterError:
        raise
    except (TypeError, ValueError, ValidationError):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.OUTPUT_REJECTED) from None


def _materialize_workspace(
    client: ClaudeCodeGuestRpcClient,
    *,
    paths: _RuntimePaths,
    limits: ClaudeCodeGuestAdapterLimits,
) -> tuple[ClaudeCodeMaterializedWorkspaceFile, ...]:
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
        materialized: list[ClaudeCodeMaterializedWorkspaceFile] = []
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
                ClaudeCodeMaterializedWorkspaceFile(
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
    except ClaudeCodeGuestAdapterError:
        raise
    except Exception:
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.WORKSPACE_REJECTED) from None


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
        CLAUDE_CODE_EMPTY_HOME_PATH,
        CLAUDE_CODE_TMP_PATH,
        '/run/vaxreplay/output',
        '/run/vaxreplay/control',
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
        or any(part in {'', '.git', '.claude', '.mcp.json'} for part in path.parts)
    ):
        raise ValueError('unsafe workspace path')
    return path


def _write_control_file(path: Path, body: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED)
    try:
        descriptor = os.open(path, flags | no_follow, 0o400)
        try:
            _write_all(descriptor, body)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED) from None


def _render_claude_prompt(
    invocation: AgenticTaskInvocation,
    workspace_files: tuple[ClaudeCodeMaterializedWorkspaceFile, ...],
) -> bytes:
    payload = {
        'schema_version': 'vaxreplay.claude-code-guest-prompt.dev-v0.1',
        'instructions': (
            'Complete the bound VaxReplay task. The current directory is the complete read-only '
            'workspace snapshot obtained through the authenticated broker. Use only the built-in '
            'Read tool when inspecting files. Do not use the internet, LAN, ambient configuration, '
            'credentials, persistence, MCP, plugins, hooks, skills, or files outside this snapshot. '
            'Return only one object matching the task-bound JSON schema; the adapter submits it '
            'terminally.'
        ),
        'task_invocation': invocation.model_dump(mode='json'),
        'workspace_inventory': [item.model_dump(mode='json') for item in workspace_files],
    }
    return canonical_json_bytes(payload)


def _sealed_vendor_environment(paths: _RuntimePaths) -> dict[str, str]:
    return {
        'PATH': '/opt/vaxreplay/bin:/usr/bin:/bin',
        'HOME': str(paths.physical(CLAUDE_CODE_EMPTY_HOME_PATH)),
        'TMPDIR': str(paths.physical(CLAUDE_CODE_TMP_PATH)),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'NO_COLOR': '1',
        'TERM': 'dumb',
        'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{CLAUDE_MESSAGES_SHIM_PORT}',
        'ANTHROPIC_API_KEY': CLAUDE_CODE_LOOPBACK_API_KEY_SENTINEL,
        'CLAUDE_CODE_MANAGED_SETTINGS_PATH': str(paths.physical(CLAUDE_CODE_EMPTY_MANAGED_SETTINGS_PATH)),
        'CLAUDE_CODE_ENABLE_TELEMETRY': '0',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'CLAUDE_CODE_DISABLE_TERMINAL_TITLE': '1',
        'DISABLE_TELEMETRY': '1',
        'DISABLE_ERROR_REPORTING': '1',
        'DISABLE_AUTOUPDATER': '1',
        'ENABLE_TOOL_SEARCH': '0',
        'OTEL_METRICS_EXPORTER': 'none',
        'OTEL_LOGS_EXPORTER': 'none',
    }


def _identity_environment(*, home: Path, scratch: Path, managed: Path) -> dict[str, str]:
    return {
        'PATH': '/usr/bin:/bin',
        'HOME': str(home),
        'TMPDIR': str(scratch),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'NO_COLOR': '1',
        'TERM': 'dumb',
        'CLAUDE_CODE_MANAGED_SETTINGS_PATH': str(managed),
        'CLAUDE_CODE_ENABLE_TELEMETRY': '0',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'DISABLE_TELEMETRY': '1',
        'DISABLE_ERROR_REPORTING': '1',
        'DISABLE_AUTOUPDATER': '1',
        'OTEL_METRICS_EXPORTER': 'none',
        'OTEL_LOGS_EXPORTER': 'none',
    }


def _verify_vendor_executable(path: Path, config: ClaudeCodeGuestAdapterConfig) -> None:
    try:
        digest, byte_count = _measure_regular_executable(path)
    except (OSError, ValueError):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED) from None
    if byte_count != config.vendor_executable_byte_count or not hmac.compare_digest(
        digest,
        config.vendor_executable_sha256,
    ):
        raise ClaudeCodeGuestAdapterError(ClaudeCodeGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED)


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


def _request_headers_sha256(headers: object) -> str:
    get_all = getattr(headers, 'get_all', None)
    keys = getattr(headers, 'keys', None)
    if not callable(get_all) or not callable(keys):
        raise ValueError('invalid HTTP headers')
    pairs: list[tuple[str, str]] = []
    for name in keys():
        if not isinstance(name, str):
            raise ValueError('invalid HTTP header name')
        values = get_all(name, failobj=[])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError('invalid HTTP header value')
        pairs.extend((name.lower(), value) for value in values)
    return _sha256(canonical_json_bytes(sorted(pairs)))


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
    'CLAUDE_CODE_GUEST_ADAPTER_ID',
    'CLAUDE_CODE_GUEST_ADAPTER_VERSION',
    'CLAUDE_CODE_LOOPBACK_API_KEY_SENTINEL',
    'CLAUDE_CODE_SUPPORTED_VENDOR_VERSION',
    'CLAUDE_CODE_VENDOR_EXECUTABLE_PATH',
    'ClaudeCodeGuestAdapterConfig',
    'ClaudeCodeGuestAdapterError',
    'ClaudeCodeGuestAdapterFailureCode',
    'ClaudeCodeGuestAdapterLimits',
    'ClaudeCodeGuestAdapterReceipt',
    'ClaudeCodeVendorIdentityEvidence',
    'capture_claude_code_vendor_identity',
    'claude_code_guest_adapter_config_sha256',
    'claude_code_vendor_argv_template',
    'render_claude_code_vendor_argv',
    'require_claude_code_guest_adapter_binding',
    'run_claude_code_guest_adapter',
]
