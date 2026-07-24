"""Development-integrated Codex CLI adapter for a sealed Lane A guest.

The adapter materializes the logical workspace only through :class:`GuestRpcClient`, starts one
loopback-only Responses compatibility shim whose model turns are backed by ``model_generate``,
launches an exact pinned Codex executable without a shell or inherited environment, parses one
bounded final response, and submits it through the same authenticated guest RPC stream.

This is a runnable guest-side development boundary, not a production-support claim.  The pinned
Linux Codex binary and dependency closure still need a reproducible disk build plus KVM, egress,
sandbox, crash, and load qualification before official admission.
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
    CODEX_RESPONSES_SHIM_PORT,
    CODEX_SEALED_CONFIG_OVERRIDES,
    MODEL_SELECTOR_TOKEN,
    OUTPUT_PATH_TOKEN,
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

CODEX_GUEST_ADAPTER_CONFIG_SCHEMA_VERSION = 'vaxreplay.codex-guest-adapter-config.dev-v0.1'
CODEX_GUEST_ADAPTER_RECEIPT_SCHEMA_VERSION = 'vaxreplay.codex-guest-adapter-receipt.dev-v0.1'
CODEX_VENDOR_IDENTITY_EVIDENCE_SCHEMA_VERSION = 'vaxreplay.codex-vendor-identity-evidence.dev-v0.1'
CODEX_GUEST_ADAPTER_ID = 'vaxreplay-codex-guest-adapter'
CODEX_GUEST_ADAPTER_VERSION = 'dev-v0.1'
CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION = 'codex-cli 0.144.3'
CODEX_VENDOR_EXECUTABLE_PATH = '/opt/vaxreplay/vendor/codex/codex'
CODEX_SUBMISSION_SCHEMA_PATH = '/run/vaxreplay/control/submission.schema.json'
CODEX_WORKSPACE_INVENTORY_PATH = '/run/vaxreplay/control/workspace.inventory.json'
CODEX_EMPTY_HOME_PATH = '/run/vaxreplay/scratch/home'
CODEX_EMPTY_CODEX_HOME_PATH = '/run/vaxreplay/scratch/codex-home'
CODEX_TMP_PATH = '/run/vaxreplay/scratch/tmp'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MODEL_SELECTOR_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,499}$')
_TOOL_NAME_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$')
_ALLOWED_CODEX_LOCAL_TOOL_NAMES = frozenset({'exec_command', 'write_stdin'})
_DISCARDED_PINNED_CODEX_TOOL_NAMES = frozenset({'apply_patch', 'request_user_input', 'update_plan', 'view_image'})
_EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()
_FORBIDDEN_ENVIRONMENT_NAME_PARTS = (
    'API_KEY',
    'AUTH',
    'BEARER',
    'CREDENTIAL',
    'PASSWORD',
    'PROXY',
    'SECRET',
    'TOKEN',
)


class CodexGuestAdapterFailureCode(str, enum.Enum):
    BINDING_REJECTED = 'binding_rejected'
    RUNTIME_LAYOUT_REJECTED = 'runtime_layout_rejected'
    VENDOR_EXECUTABLE_REJECTED = 'vendor_executable_rejected'
    WORKSPACE_REJECTED = 'workspace_rejected'
    SHIM_REJECTED = 'shim_rejected'
    LAUNCH_REJECTED = 'launch_rejected'
    SUBPROCESS_REJECTED = 'subprocess_rejected'
    OUTPUT_REJECTED = 'output_rejected'
    SUBMISSION_REJECTED = 'submission_rejected'


class CodexGuestAdapterError(RuntimeError):
    """Content-free adapter failure suitable for the guest appliance boundary."""

    def __init__(self, code: CodexGuestAdapterFailureCode):
        super().__init__(code.value)
        self.code = code


class CodexGuestAdapterLimits(StrictModel):
    maximum_workspace_files: int = Field(default=2_000, ge=1, le=100_000)
    maximum_workspace_bytes: int = Field(default=128 * 1024 * 1024, ge=1, le=2 * 1024 * 1024 * 1024)
    workspace_list_page_size: int = Field(default=100, ge=1, le=1_000)
    workspace_read_chunk_bytes: int = Field(default=1024 * 1024, ge=1, le=16 * 1024 * 1024)
    maximum_prompt_bytes: int = Field(default=4 * 1024 * 1024, ge=1_024, le=32 * 1024 * 1024)
    maximum_vendor_stdout_bytes: int = Field(default=4 * 1024 * 1024, ge=1_024, le=64 * 1024 * 1024)
    maximum_vendor_stderr_bytes: int = Field(default=1024 * 1024, ge=1_024, le=16 * 1024 * 1024)
    maximum_final_output_bytes: int = Field(default=2 * 1024 * 1024, ge=1_024, le=16 * 1024 * 1024)
    maximum_shim_request_bytes: int = Field(default=8 * 1024 * 1024, ge=1_024, le=64 * 1024 * 1024)
    maximum_shim_calls: int = Field(default=10, ge=1, le=100)
    maximum_shim_decision_tokens: int = Field(default=4_096, ge=128, le=32_768)
    vendor_wall_seconds: int = Field(default=900, ge=1, le=7_200)


class CodexGuestAdapterConfig(StrictModel):
    """Hash-bound runtime facts not already carried by the generic headless contract."""

    schema_version: Literal['vaxreplay.codex-guest-adapter-config.dev-v0.1'] = CODEX_GUEST_ADAPTER_CONFIG_SCHEMA_VERSION
    adapter_id: Literal['vaxreplay-codex-guest-adapter'] = CODEX_GUEST_ADAPTER_ID
    adapter_version: Literal['dev-v0.1'] = CODEX_GUEST_ADAPTER_VERSION
    headless_adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    submitted_harness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_path: Literal['/opt/vaxreplay/vendor/codex/codex'] = CODEX_VENDOR_EXECUTABLE_PATH
    vendor_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_byte_count: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    supported_vendor_version: Literal['codex-cli 0.144.3'] = CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION
    responses_shim_host: Literal['127.0.0.1'] = '127.0.0.1'
    responses_shim_port: Literal[43123] = CODEX_RESPONSES_SHIM_PORT
    submission_schema_path: Literal['/run/vaxreplay/control/submission.schema.json'] = CODEX_SUBMISSION_SCHEMA_PATH
    workspace_inventory_path: Literal['/run/vaxreplay/control/workspace.inventory.json'] = (
        CODEX_WORKSPACE_INVENTORY_PATH
    )
    limits: CodexGuestAdapterLimits = CodexGuestAdapterLimits()
    workspace_materialization_uses_only_authenticated_guest_rpc: Literal[True] = True
    model_transport_is_loopback_only_responses_shim: Literal[True] = True
    final_submission_uses_authenticated_guest_rpc: Literal[True] = True
    inherited_environment_allowed: Literal[False] = False
    provider_credentials_in_guest_allowed: Literal[False] = False
    ambient_codex_home_allowed: Literal[False] = False
    shell_command_construction_allowed: Literal[False] = False
    automatic_vendor_retry_allowed: Literal[False] = False
    linux_kvm_qualified: Literal[False] = False
    actual_pinned_linux_codex_end_to_end_validated: Literal[False] = False
    development_only: Literal[True] = True


class CodexVendorIdentityEvidence(StrictModel):
    """A local, non-KVM measurement that can be pinned before building the harness disk."""

    schema_version: Literal['vaxreplay.codex-vendor-identity-evidence.dev-v0.1'] = (
        CODEX_VENDOR_IDENTITY_EVIDENCE_SCHEMA_VERSION
    )
    executable_path: str = Field(min_length=2, max_length=4096)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    executable_byte_count: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    version_argv_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stdout_bytes: int = Field(ge=0, le=4096)
    version_stderr_sha256: str = Field(pattern=_SHA256_PATTERN)
    version_stderr_bytes: int = Field(ge=0, le=4096)
    reported_version: Literal['codex-cli 0.144.3'] = CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION
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
            raise ValueError('measured Codex executable path must be absolute and normalized')
        return value


class CodexMaterializedWorkspaceFile(StrictModel):
    path: str = Field(min_length=1, max_length=4096)
    media_type: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)


class CodexShimExchangeReceipt(StrictModel):
    call_index: int = Field(ge=0)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    request_bytes: int = Field(gt=0)
    forwarded_request_sha256: str = Field(pattern=_SHA256_PATTERN)
    forwarded_request_bytes: int = Field(gt=0)
    discarded_vendor_tool_count: int = Field(ge=0, le=1_000)
    prior_vendor_tool_call_item_count: int = Field(ge=0, le=10_000)
    prior_vendor_tool_output_item_count: int = Field(ge=0, le=10_000)
    decision_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_bytes: int = Field(gt=0)
    sse_sha256: str = Field(pattern=_SHA256_PATTERN)
    sse_bytes: int = Field(gt=0)
    response_item_type: Literal['message', 'function_call', 'custom_tool_call']


class CodexGuestAdapterReceipt(StrictModel):
    schema_version: Literal['vaxreplay.codex-guest-adapter-receipt.dev-v0.1'] = (
        CODEX_GUEST_ADAPTER_RECEIPT_SCHEMA_VERSION
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
    workspace_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_file_count: int = Field(ge=1)
    workspace_byte_count: int = Field(ge=0)
    shim_exchanges: tuple[CodexShimExchangeReceipt, ...] = Field(min_length=1)
    shim_exchange_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_duration_ms: int = Field(ge=0)
    vendor_stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_stdout_bytes: int = Field(ge=0)
    vendor_stderr_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_stderr_bytes: int = Field(ge=0)
    final_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    final_output_bytes: int = Field(gt=0)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    submit_result: SubmitResult
    workspace_was_complete_materialized_snapshot: Literal[True] = True
    no_provider_credentials_or_ambient_config_in_subprocess: Literal[True] = True
    outer_guest_rpc_events_remain_authoritative: Literal[True] = True
    actual_pinned_linux_codex_end_to_end_validated: Literal[False] = False
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


class _CodexShimAssistantDecision(StrictModel):
    kind: Literal['assistant_text']
    text: str = Field(min_length=1, max_length=2_000_000)


class _CodexShimToolDecision(StrictModel):
    kind: Literal['tool_call']
    tool_name: str = Field(pattern=r'^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$')
    payload: JsonValue


type CodexShimDecision = Annotated[
    _CodexShimAssistantDecision | _CodexShimToolDecision,
    Field(discriminator='kind'),
]
_CODEX_SHIM_DECISION_ADAPTER = TypeAdapter(CodexShimDecision)
_CODEX_SHIM_DECISION_SCHEMA = _CODEX_SHIM_DECISION_ADAPTER.json_schema()
_CODEX_SHIM_SYSTEM_PROMPT = (
    'You are the model behind an exact Codex CLI Responses compatibility shim. The user message '
    'contains the canonical Responses request emitted by Codex after the sealed adapter has removed '
    'vendor tools outside the fixed local read-only execution surface. It includes the current '
    'conversation and remaining tool definitions. Return exactly one JSON object matching the '
    'decision schema. Choose assistant_text for ordinary assistant output. Choose tool_call only '
    'for a tool name present in this request. For a function tool, payload must be its JSON '
    'argument object. For a custom tool, payload must be its string input. Never invent a tool, '
    'provider route, credential, web search, or explanation outside the JSON object.\n'
    'Decision JSON Schema:\n' + canonical_json_bytes(_CODEX_SHIM_DECISION_SCHEMA).decode('utf-8')
)


class _CodexResponsesRequest(StrictModel):
    """The exact request fields emitted by pinned Codex Responses HTTP transport."""

    model: str = Field(min_length=1, max_length=500)
    instructions: str = Field(max_length=2_000_000)
    input: tuple[dict[str, JsonValue], ...] = Field(min_length=1, max_length=10_000)
    tools: tuple[dict[str, JsonValue], ...] | None = Field(default=None, max_length=1_000)
    tool_choice: JsonValue
    parallel_tool_calls: bool
    reasoning: dict[str, JsonValue] | None = None
    store: bool
    stream: Literal[True]
    stream_options: dict[str, JsonValue] | None = None
    include: tuple[str, ...] = Field(max_length=100)
    service_tier: str | None = Field(default=None, max_length=100)
    prompt_cache_key: str | None = Field(default=None, max_length=500)
    text: dict[str, JsonValue] | None = None
    client_metadata: dict[str, JsonValue] | None = None

    @model_validator(mode='after')
    def validate_sealed_request(self) -> Self:
        if self.store:
            raise ValueError('sealed Codex requests cannot ask the compatibility shim to store responses')
        return self


class CodexGuestRpcClient(Protocol):
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
    ) -> BoundedProcessResult: ...


@dataclass(frozen=True)
class _RuntimePaths:
    root: Path

    def physical(self, logical: str) -> Path:
        path = PurePosixPath(logical)
        if not path.is_absolute() or '..' in path.parts or path.as_posix() != logical:
            raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED)
        if self.root == Path('/'):
            return Path(logical)
        return self.root.joinpath(*path.parts[1:])


def codex_guest_adapter_config_sha256(config: CodexGuestAdapterConfig) -> str:
    canonical = CodexGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
    return _sha256(canonical_json_bytes(canonical))


def capture_codex_vendor_identity(
    executable_path: Path,
    *,
    process_runner: ProcessRunner = run_bounded_process,
) -> CodexVendorIdentityEvidence:
    """Measure an exact local Codex file and its bounded ``--version`` output.

    A symlink such as a package-manager convenience path is deliberately rejected; callers pin
    the resolved regular file that must later be copied into the guest dependency closure.
    """

    path_text = executable_path.as_posix()
    pure_path = PurePosixPath(path_text)
    if (
        not executable_path.is_absolute()
        or '..' in pure_path.parts
        or pure_path.as_posix() != path_text
        or executable_path.is_symlink()
    ):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED)
    try:
        executable_sha256, executable_bytes = _measure_regular_executable(executable_path)
        with tempfile.TemporaryDirectory(prefix='vaxreplay-codex-identity-') as temporary:
            temporary_root = Path(temporary)
            home = temporary_root / 'home'
            codex_home = temporary_root / 'codex-home'
            scratch = temporary_root / 'tmp'
            for directory in (home, codex_home, scratch):
                directory.mkdir(mode=0o700)
            environment = {
                'PATH': '/usr/bin:/bin',
                'HOME': str(home),
                'CODEX_HOME': str(codex_home),
                'TMPDIR': str(scratch),
                'LANG': 'C.UTF-8',
                'LC_ALL': 'C.UTF-8',
                'NO_COLOR': '1',
                'TERM': 'dumb',
            }
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
        if reported_version != CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION:
            raise ValueError('unsupported version')
        return CodexVendorIdentityEvidence(
            executable_path=path_text,
            executable_sha256=executable_sha256,
            executable_byte_count=executable_bytes,
            version_argv_sha256=_sha256(canonical_json_bytes((path_text, '--version'))),
            version_stdout_sha256=_sha256(result.stdout),
            version_stdout_bytes=len(result.stdout),
            version_stderr_sha256=_sha256(result.stderr),
            version_stderr_bytes=len(result.stderr),
            reported_version=reported_version,
            observed_os=platform.system(),
            observed_architecture=platform.machine(),
        )
    except CodexGuestAdapterError:
        raise
    except (OSError, UnicodeDecodeError, ValueError):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED) from None


def codex_vendor_argv_template() -> tuple[str, ...]:
    """Return the only Codex argv shape implemented by this adapter."""

    config_arguments = tuple(item for override in CODEX_SEALED_CONFIG_OVERRIDES for item in ('--config', override))
    return (
        CODEX_VENDOR_EXECUTABLE_PATH,
        'exec',
        '--ephemeral',
        '--ignore-user-config',
        '--ignore-rules',
        '--disable',
        'tool_suggest',
        '--disable',
        'tool_search',
        '--disable',
        'multi_agent',
        '--skip-git-repo-check',
        '--sandbox',
        'read-only',
        '--cd',
        WORKSPACE_PATH_TOKEN,
        *config_arguments,
        '--color',
        'never',
        '--output-schema',
        CODEX_SUBMISSION_SCHEMA_PATH,
        '--output-last-message',
        OUTPUT_PATH_TOKEN,
        '--model',
        MODEL_SELECTOR_TOKEN,
        '-',
    )


def require_codex_guest_adapter_binding(
    *,
    config: CodexGuestAdapterConfig,
    headless_config: HeadlessGuestAdapterConfig,
    submitted_manifest: SubmittedHarnessManifest,
) -> None:
    """Cross-bind runtime config, generic contract, and submitted harness identity."""

    try:
        runtime = CodexGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
        headless = HeadlessGuestAdapterConfig.model_validate_json(canonical_json_bytes(headless_config))
        submitted = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(submitted_manifest))
        require_headless_guest_adapter_binding(config=headless, manifest=submitted)
    except (TypeError, ValueError):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.BINDING_REJECTED) from None
    expected = (
        HarnessFamily.CODEX,
        HeadlessInvocationProtocol.CODEX_EXEC,
        HeadlessResponseChannel.BOUNDED_OUTPUT_FILE,
        HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED,
        True,
        True,
        True,
        CODEX_VENDOR_EXECUTABLE_PATH,
        runtime.vendor_executable_sha256,
        CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
        codex_vendor_argv_template(),
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
        headless.vendor_executable_path,
        headless.vendor_executable_sha256,
        headless.vendor_reported_version,
        headless.vendor_argv_template,
        runtime.headless_adapter_config_sha256,
        runtime.submitted_harness_manifest_sha256,
    )
    if actual != expected:
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.BINDING_REJECTED)


def render_codex_vendor_argv(
    *,
    organizer_model_selector: str,
    paths: _RuntimePaths | None = None,
) -> tuple[str, ...]:
    """Render trusted values as argv elements and never construct a command string."""

    if not _MODEL_SELECTOR_PATTERN.fullmatch(organizer_model_selector):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.BINDING_REJECTED)
    runtime_paths = paths or _RuntimePaths(Path('/'))
    substitutions = {
        CODEX_VENDOR_EXECUTABLE_PATH: str(runtime_paths.physical(CODEX_VENDOR_EXECUTABLE_PATH)),
        MODEL_SELECTOR_TOKEN: organizer_model_selector,
        WORKSPACE_PATH_TOKEN: str(runtime_paths.physical('/run/vaxreplay/workspace')),
        OUTPUT_PATH_TOKEN: str(runtime_paths.physical('/run/vaxreplay/output/final-response')),
        CODEX_SUBMISSION_SCHEMA_PATH: str(runtime_paths.physical(CODEX_SUBMISSION_SCHEMA_PATH)),
    }
    return tuple(substitutions.get(item, item) for item in codex_vendor_argv_template())


def run_codex_guest_adapter(
    client: CodexGuestRpcClient,
    *,
    task_invocation: AgenticTaskInvocation,
    organizer_model_selector: str,
    config: CodexGuestAdapterConfig,
    headless_config: HeadlessGuestAdapterConfig,
    submitted_manifest: SubmittedHarnessManifest,
    process_runner: ProcessRunner = run_bounded_process,
    _guest_root: Path = Path('/'),
) -> CodexGuestAdapterReceipt:
    """Run one bounded Codex attempt after authenticated guest bootstrap has created ``client``."""

    try:
        invocation = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(task_invocation))
        runtime = CodexGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
        require_codex_guest_adapter_binding(
            config=runtime,
            headless_config=headless_config,
            submitted_manifest=submitted_manifest,
        )
        if not _guest_root.is_absolute() or _guest_root.is_symlink():
            raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED)
        paths = _RuntimePaths(_guest_root)
        vendor_path = paths.physical(runtime.vendor_executable_path)
        _verify_vendor_executable(vendor_path, runtime)
        workspace_files = _materialize_workspace(client, paths=paths, limits=runtime.limits)
        schema_bytes = canonical_json_bytes(submission_json_schema_for_invocation(invocation))
        _write_control_file(paths.physical(runtime.submission_schema_path), schema_bytes)
        inventory_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in workspace_files])
        _write_control_file(paths.physical(runtime.workspace_inventory_path), inventory_bytes)
        prompt = _render_codex_prompt(invocation, workspace_files)
        if len(prompt) > runtime.limits.maximum_prompt_bytes:
            raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.WORKSPACE_REJECTED)
        argv = render_codex_vendor_argv(
            organizer_model_selector=organizer_model_selector,
            paths=paths,
        )
        environment = _sealed_vendor_environment(paths)
        shim = _CodexResponsesShim(
            client,
            organizer_model_selector=organizer_model_selector,
            limits=runtime.limits,
        )
        with _CodexResponsesServer(shim, port=runtime.responses_shim_port):
            try:
                process = process_runner(
                    argv,
                    input_bytes=prompt,
                    wall_seconds=runtime.limits.vendor_wall_seconds,
                    max_stdout_bytes=runtime.limits.maximum_vendor_stdout_bytes,
                    max_stderr_bytes=runtime.limits.maximum_vendor_stderr_bytes,
                    on_abort=shim.abort,
                    env=environment,
                )
            except (OSError, RuntimeError, ValueError):
                raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.LAUNCH_REJECTED) from None
        if shim.failure is not None:
            raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.SHIM_REJECTED)
        if (
            process.termination != 'exited'
            or process.exit_code != 0
            or process.stdout_truncated
            or process.stderr_truncated
        ):
            raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.SUBPROCESS_REJECTED)
        output = _read_output_file(
            paths.physical('/run/vaxreplay/output/final-response'),
            runtime.limits.maximum_final_output_bytes,
        )
        submission = _parse_final_submission(invocation, output)
        try:
            submit_result = client.submit(submission)
        except Exception:
            raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.SUBMISSION_REJECTED) from None
        submission_bytes = canonical_json_bytes(submission)
        exchanges = shim.exchanges
        exchange_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in exchanges])
        return CodexGuestAdapterReceipt(
            adapter_config_sha256=codex_guest_adapter_config_sha256(runtime),
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
            final_output_sha256=_sha256(output),
            final_output_bytes=len(output),
            submission_sha256=_sha256(submission_bytes),
            submit_result=submit_result,
        )
    except CodexGuestAdapterError:
        raise
    except (TypeError, ValueError, ValidationError):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.BINDING_REJECTED) from None


class _CodexResponsesShim:
    def __init__(
        self,
        client: CodexGuestRpcClient,
        *,
        organizer_model_selector: str,
        limits: CodexGuestAdapterLimits,
    ) -> None:
        self._client = client
        self._model_selector = organizer_model_selector
        self._limits = limits
        self._lock = threading.Lock()
        self._exchanges: list[CodexShimExchangeReceipt] = []
        self._failure: CodexGuestAdapterFailureCode | None = None

    @property
    def failure(self) -> CodexGuestAdapterFailureCode | None:
        with self._lock:
            return self._failure

    @property
    def exchanges(self) -> tuple[CodexShimExchangeReceipt, ...]:
        with self._lock:
            return tuple(self._exchanges)

    @property
    def maximum_request_bytes(self) -> int:
        return self._limits.maximum_shim_request_bytes

    def abort(self) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = CodexGuestAdapterFailureCode.SUBPROCESS_REJECTED

    def reject(self) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = CodexGuestAdapterFailureCode.SHIM_REJECTED

    def handle(self, body: bytes) -> bytes:
        with self._lock:
            if self._failure is not None or len(self._exchanges) >= self._limits.maximum_shim_calls:
                self._failure = CodexGuestAdapterFailureCode.SHIM_REJECTED
                raise ValueError('shim rejected')
            call_index = len(self._exchanges)
            try:
                payload = _load_unique_json(body)
                request = _CodexResponsesRequest.model_validate_json(canonical_json_bytes(payload))
                if request.model != self._model_selector:
                    raise ValueError('model selector mismatch')
                tools, forwarded_tools, discarded_tool_count = _sealed_request_tool_surface(request)
                prior_tool_calls, prior_tool_outputs = _count_prior_tool_items(request)
                forwarded_request = request.model_copy(update={'tools': forwarded_tools})
                forwarded_request_bytes = canonical_json_bytes(forwarded_request)
                response = self._client.model_generate(
                    messages=(
                        AgenticModelMessage(role='system', content=_CODEX_SHIM_SYSTEM_PROMPT),
                        AgenticModelMessage(
                            role='user',
                            content=forwarded_request_bytes.decode('utf-8'),
                        ),
                    ),
                    max_output_tokens=self._limits.maximum_shim_decision_tokens,
                    response_schema_sha256=None,
                )
                if response.stop_reason != 'completed':
                    raise ValueError('model response incomplete')
                decision_payload = _load_unique_json(response.content.encode('utf-8'))
                decision = _CODEX_SHIM_DECISION_ADAPTER.validate_json(
                    canonical_json_bytes(decision_payload),
                    strict=True,
                )
                item = _response_item(decision, tools=tools, call_index=call_index)
                sse = _render_responses_sse(
                    item=item,
                    call_index=call_index,
                    resolved_model_id=response.resolved_model_id,
                    usage=response,
                )
                decision_bytes = canonical_json_bytes(decision)
                receipt = CodexShimExchangeReceipt(
                    call_index=call_index,
                    request_sha256=_sha256(body),
                    request_bytes=len(body),
                    forwarded_request_sha256=_sha256(forwarded_request_bytes),
                    forwarded_request_bytes=len(forwarded_request_bytes),
                    discarded_vendor_tool_count=discarded_tool_count,
                    prior_vendor_tool_call_item_count=prior_tool_calls,
                    prior_vendor_tool_output_item_count=prior_tool_outputs,
                    decision_sha256=_sha256(decision_bytes),
                    decision_bytes=len(decision_bytes),
                    sse_sha256=_sha256(sse),
                    sse_bytes=len(sse),
                    response_item_type=cast(Literal['message', 'function_call', 'custom_tool_call'], item['type']),
                )
            except Exception:
                self._failure = CodexGuestAdapterFailureCode.SHIM_REJECTED
                raise ValueError('shim rejected') from None
            self._exchanges.append(receipt)
            return sse


class _CodexResponsesServer:
    def __init__(self, shim: _CodexResponsesShim, *, port: int) -> None:
        self._shim = shim
        self._port = port
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _CodexResponsesServer:
        shim = self._shim
        port = self._port

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                host_headers = self.headers.get_all('Host', failobj=[])
                content_type_headers = self.headers.get_all('Content-Type', failobj=[])
                length_headers = self.headers.get_all('Content-Length', failobj=[])
                if (
                    self.path != '/v1/responses'
                    or host_headers != [f'127.0.0.1:{port}']
                    or len(content_type_headers) != 1
                    or content_type_headers[0].partition(';')[0].strip().lower() != 'application/json'
                    or len(length_headers) != 1
                    or self.headers.get('Transfer-Encoding') is not None
                    or self.headers.get('Content-Encoding') is not None
                    or self.headers.get('Authorization') is not None
                ):
                    shim.reject()
                    self._reject()
                    return
                length_text = self.headers.get('Content-Length')
                try:
                    length = int(length_text or '')
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
                    response = shim.handle(body)
                except (OSError, ValueError):
                    self._reject()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
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
                body = b'{"error":{"message":"request rejected","type":"invalid_request_error"}}'
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
                return

        class LoopbackHTTPServer(http.server.HTTPServer):
            # Reuse only clears a prior closed listener's TIME_WAIT state.  SO_REUSEPORT is never
            # enabled, so a concurrently bound process still makes this exact bind fail closed.
            allow_reuse_address = True

        try:
            server = LoopbackHTTPServer(('127.0.0.1', self._port), Handler, bind_and_activate=True)
        except OSError:
            raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.SHIM_REJECTED) from None
        if server.server_address != ('127.0.0.1', self._port):
            server.server_close()
            raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.SHIM_REJECTED)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _sealed_request_tool_surface(
    request: _CodexResponsesRequest,
) -> tuple[
    dict[str, Literal['function', 'custom']],
    tuple[dict[str, JsonValue], ...],
    int,
]:
    tools: dict[str, Literal['function', 'custom']] = {}
    forwarded: list[dict[str, JsonValue]] = []
    discarded = 0
    for tool in request.tools or ():
        tool_type = tool.get('type')
        name = tool.get('name')
        if tool_type == 'tool_search' and name is None:
            discarded += 1
            continue
        if tool_type not in {'function', 'custom'} or not isinstance(name, str):
            raise ValueError('unsupported tool surface')
        if not _TOOL_NAME_PATTERN.fullmatch(name):
            raise ValueError('unsupported tool name')
        if name in _DISCARDED_PINNED_CODEX_TOOL_NAMES:
            discarded += 1
            continue
        if name not in _ALLOWED_CODEX_LOCAL_TOOL_NAMES:
            raise ValueError('unrecognized pinned Codex tool')
        if name in tools:
            raise ValueError('duplicate tool name')
        tools[name] = cast(Literal['function', 'custom'], tool_type)
        forwarded.append(tool)
    return tools, tuple(forwarded), discarded


def _count_prior_tool_items(request: _CodexResponsesRequest) -> tuple[int, int]:
    calls = 0
    outputs = 0
    for item in request.input:
        item_type = item.get('type')
        if item_type in {'function_call', 'custom_tool_call'}:
            calls += 1
        elif item_type in {'function_call_output', 'custom_tool_call_output'}:
            outputs += 1
    return calls, outputs


def _response_item(
    decision: CodexShimDecision,
    *,
    tools: Mapping[str, Literal['function', 'custom']],
    call_index: int,
) -> dict[str, JsonValue]:
    suffix = f'{call_index:08d}'
    if isinstance(decision, _CodexShimAssistantDecision):
        return {
            'type': 'message',
            'id': f'msg_vaxreplay_{suffix}',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': decision.text}],
            'phase': 'final_answer',
        }
    tool_type = tools.get(decision.tool_name)
    if tool_type is None:
        raise ValueError('decision names an unavailable tool')
    if tool_type == 'function':
        if not isinstance(decision.payload, dict):
            raise ValueError('function tool payload must be an object')
        return {
            'type': 'function_call',
            'id': f'fc_vaxreplay_{suffix}',
            'call_id': f'call_vaxreplay_{suffix}',
            'name': decision.tool_name,
            'arguments': canonical_json_bytes(decision.payload).decode('utf-8'),
        }
    if not isinstance(decision.payload, str) or not decision.payload:
        raise ValueError('custom tool payload must be a nonempty string')
    return {
        'type': 'custom_tool_call',
        'id': f'ctc_vaxreplay_{suffix}',
        'call_id': f'call_vaxreplay_{suffix}',
        'name': decision.tool_name,
        'input': decision.payload,
    }


def _render_responses_sse(
    *,
    item: Mapping[str, JsonValue],
    call_index: int,
    resolved_model_id: str,
    usage: AgenticModelResponse,
) -> bytes:
    response_id = f'resp_vaxreplay_{call_index:08d}'
    events: tuple[dict[str, JsonValue], ...] = (
        {
            'type': 'response.created',
            'response': {'id': response_id, 'headers': {'openai-model': resolved_model_id}},
        },
        {'type': 'response.output_item.added', 'output_index': 0, 'item': dict(item)},
        {'type': 'response.output_item.done', 'output_index': 0, 'item': dict(item)},
        {
            'type': 'response.completed',
            'response': {
                'id': response_id,
                'end_turn': item['type'] == 'message',
                'usage': {
                    'input_tokens': usage.usage.input_tokens,
                    'input_tokens_details': {'cached_tokens': 0},
                    'output_tokens': usage.usage.output_tokens,
                    'output_tokens_details': {'reasoning_tokens': usage.usage.reasoning_tokens or 0},
                    'total_tokens': usage.usage.input_tokens + usage.usage.output_tokens,
                },
            },
        },
    )
    return b''.join(b'data: ' + canonical_json_bytes(event) + b'\n\n' for event in events)


def _materialize_workspace(
    client: CodexGuestRpcClient,
    *,
    paths: _RuntimePaths,
    limits: CodexGuestAdapterLimits,
) -> tuple[CodexMaterializedWorkspaceFile, ...]:
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
        materialized: list[CodexMaterializedWorkspaceFile] = []
        for raw_item in listed:
            item = raw_item
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
                raise ValueError('no nofollow')
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
            if written != item.byte_count or not hmac.compare_digest(digest.hexdigest(), item.sha256):
                raise ValueError('workspace bytes differ from listing')
            os.chmod(target, 0o400, follow_symlinks=False)
            materialized.append(
                CodexMaterializedWorkspaceFile(
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
    except CodexGuestAdapterError:
        raise
    except Exception:
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.WORKSPACE_REJECTED) from None


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
        CODEX_EMPTY_HOME_PATH,
        CODEX_EMPTY_CODEX_HOME_PATH,
        CODEX_TMP_PATH,
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
        or any(part in {'', '.git', '.codex'} for part in path.parts)
    ):
        raise ValueError('unsafe workspace path')
    return path


def _write_control_file(path: Path, body: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED)
    try:
        descriptor = os.open(path, flags | no_follow, 0o400)
        try:
            _write_all(descriptor, body)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED) from None


def _render_codex_prompt(
    invocation: AgenticTaskInvocation,
    workspace_files: tuple[CodexMaterializedWorkspaceFile, ...],
) -> bytes:
    payload = {
        'schema_version': 'vaxreplay.codex-guest-prompt.dev-v0.1',
        'instructions': (
            'Complete the bound VaxReplay task. The current directory is a complete read-only '
            'snapshot obtained through the authenticated workspace broker. Inspect relevant files '
            'with local read-only tools. Do not use the internet, LAN, ambient configuration, '
            'credentials, persistence, or files outside this snapshot. Return only one JSON object '
            'matching the supplied output schema; the adapter will submit it terminally.'
        ),
        'task_invocation': invocation.model_dump(mode='json'),
        'workspace_inventory': [item.model_dump(mode='json') for item in workspace_files],
    }
    return canonical_json_bytes(payload)


def _sealed_vendor_environment(paths: _RuntimePaths) -> dict[str, str]:
    environment = {
        'PATH': '/opt/vaxreplay/bin:/usr/bin:/bin',
        'HOME': str(paths.physical(CODEX_EMPTY_HOME_PATH)),
        'CODEX_HOME': str(paths.physical(CODEX_EMPTY_CODEX_HOME_PATH)),
        'TMPDIR': str(paths.physical(CODEX_TMP_PATH)),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'NO_COLOR': '1',
        'TERM': 'dumb',
    }
    if any(part in name.upper() for name in environment for part in _FORBIDDEN_ENVIRONMENT_NAME_PARTS):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.RUNTIME_LAYOUT_REJECTED)
    return environment


def _verify_vendor_executable(path: Path, config: CodexGuestAdapterConfig) -> None:
    try:
        digest, byte_count = _measure_regular_executable(path)
    except (OSError, ValueError):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED) from None
    if byte_count != config.vendor_executable_byte_count or not hmac.compare_digest(
        digest,
        config.vendor_executable_sha256,
    ):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED)


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


def _read_output_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.OUTPUT_REJECTED)
    descriptor = -1
    try:
        descriptor = os.open(path, flags | no_follow)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o022
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise ValueError('bad output metadata')
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b''.join(chunks)
        after = os.fstat(descriptor)
        if (
            len(body) != before.st_size
            or len(body) > maximum_bytes
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError('unstable output')
        return body
    except (OSError, ValueError):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.OUTPUT_REJECTED) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parse_final_submission(
    invocation: AgenticTaskInvocation,
    output: bytes,
) -> AgenticRuntimeSubmission:
    try:
        payload = _load_unique_json(output)
        return parse_submission_for_invocation(invocation, canonical_json_bytes(payload))
    except (TypeError, ValueError, ValidationError):
        raise CodexGuestAdapterError(CodexGuestAdapterFailureCode.OUTPUT_REJECTED) from None


def _load_unique_json(body: bytes | str) -> JsonValue:
    def unique_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    try:
        payload = json.loads(body, object_pairs_hook=unique_pairs, parse_constant=lambda _value: _reject_json())
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError('invalid JSON') from None
    return cast(JsonValue, payload)


def _reject_json() -> None:
    raise ValueError('non-finite JSON number')


def _write_all(descriptor: int, body: bytes) -> None:
    view = memoryview(body)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError('short write')
        view = view[written:]


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


__all__ = [
    'CODEX_GUEST_ADAPTER_CONFIG_SCHEMA_VERSION',
    'CODEX_GUEST_ADAPTER_ID',
    'CODEX_GUEST_ADAPTER_RECEIPT_SCHEMA_VERSION',
    'CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION',
    'CODEX_GUEST_ADAPTER_VERSION',
    'CODEX_VENDOR_IDENTITY_EVIDENCE_SCHEMA_VERSION',
    'CODEX_VENDOR_EXECUTABLE_PATH',
    'CodexGuestAdapterConfig',
    'CodexGuestAdapterError',
    'CodexGuestAdapterFailureCode',
    'CodexGuestAdapterLimits',
    'CodexGuestAdapterReceipt',
    'CodexGuestRpcClient',
    'CodexMaterializedWorkspaceFile',
    'CodexShimExchangeReceipt',
    'CodexVendorIdentityEvidence',
    'capture_codex_vendor_identity',
    'codex_guest_adapter_config_sha256',
    'codex_vendor_argv_template',
    'render_codex_vendor_argv',
    'require_codex_guest_adapter_binding',
    'run_codex_guest_adapter',
]
