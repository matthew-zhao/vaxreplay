"""Fail-closed launch contract for a pinned vendor agent inside the Lane A guest.

This generic layer freezes the outer adapter, inner CLI, argv template, filesystem locations, and
required RPC bridges.  Codex- and Claude-specific development launchers live in their respective
guest-adapter modules; Cursor and custom families remain contract-only.  Neither state is
production qualification.
"""

from __future__ import annotations

import enum
import hashlib
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.guest_rpc import GuestRpcMethod
from vaxreplay.agentic.submitted_harness import (
    HarnessExecutionMode,
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessError,
    SubmittedHarnessManifest,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

HEADLESS_GUEST_ADAPTER_CONFIG_SCHEMA_VERSION = 'vaxreplay.headless-guest-adapter-config.dev-v0.1'
HEADLESS_GUEST_ADAPTER_CONTRACT_ID = 'vaxreplay-lane-a-headless-guest-adapter-v0.1'
HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH = '/opt/vaxreplay/bin/vaxreplay-headless-guest-adapter'
HEADLESS_GUEST_ADAPTER_CONFIG_DIGEST_FLAG = '--expected-config-sha256'

MODEL_SELECTOR_TOKEN = '__VAXREPLAY_ORGANIZER_MODEL_SELECTOR__'
WORKSPACE_PATH_TOKEN = '__VAXREPLAY_WORKSPACE_PATH__'
SCRATCH_PATH_TOKEN = '__VAXREPLAY_SCRATCH_PATH__'
OUTPUT_PATH_TOKEN = '__VAXREPLAY_OUTPUT_PATH__'
SUBMISSION_SCHEMA_JSON_TOKEN = '__VAXREPLAY_SUBMISSION_SCHEMA_JSON__'
_RUNTIME_TOKENS = (
    MODEL_SELECTOR_TOKEN,
    WORKSPACE_PATH_TOKEN,
    SCRATCH_PATH_TOKEN,
    OUTPUT_PATH_TOKEN,
    SUBMISSION_SCHEMA_JSON_TOKEN,
)
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_ALL_RPC_METHODS = tuple(method.value for method in GuestRpcMethod)
CODEX_RESPONSES_SHIM_PORT = 43123
CLAUDE_MESSAGES_SHIM_PORT = 43124
CODEX_SEALED_CONFIG_OVERRIDES = (
    'web_search="disabled"',
    'approval_policy="never"',
    'model_provider="vaxreplay"',
    (
        'model_providers.vaxreplay={name="VaxReplay sealed RPC",'
        f'base_url="http://127.0.0.1:{CODEX_RESPONSES_SHIM_PORT}/v1",'
        'wire_api="responses",requires_openai_auth=false,supports_websockets=false,'
        'request_max_retries=0,stream_max_retries=0,stream_idle_timeout_ms=60000}'
    ),
    'shell_environment_policy.inherit="none"',
    (
        'shell_environment_policy.set={PATH="/opt/vaxreplay/bin:/usr/bin:/bin",'
        'LANG="C.UTF-8",LC_ALL="C.UTF-8",HOME="/run/vaxreplay/scratch/home",'
        'TMPDIR="/run/vaxreplay/scratch/tmp"}'
    ),
)
_FORBIDDEN_VENDOR_ARGUMENTS = frozenset(
    {
        '-H',
        '-c',
        '-f',
        '-i',
        '-r',
        '-w',
        '--agent',
        '--agents',
        '--api-key',
        '--approve-mcps',
        '--betas',
        '--chrome',
        '--continue',
        '--dangerously-bypass-approvals-and-sandbox',
        '--dangerously-bypass-hook-trust',
        '--dangerously-skip-permissions',
        '--fallback-model',
        '--file',
        '--fork-session',
        '--force',
        '--header',
        '--ide',
        '--image',
        '--local-provider',
        '--oss',
        '--plugin-dir',
        '--plugin-url',
        '--remote',
        '--remote-control',
        '--resume',
        '--search',
        '--setting-sources',
        '--settings',
        '--worktree',
        '--yolo',
    }
)


class HeadlessInvocationProtocol(str, enum.Enum):
    CODEX_EXEC = 'codex_exec'
    CLAUDE_PRINT = 'claude_print'
    CURSOR_PRINT = 'cursor_print'
    CUSTOM_HEADLESS = 'custom_headless'


class HeadlessResponseChannel(str, enum.Enum):
    BOUNDED_OUTPUT_FILE = 'bounded_output_file'
    BOUNDED_JSON_STDOUT = 'bounded_json_stdout'
    BOUNDED_JSONL_STDOUT = 'bounded_jsonl_stdout'


class HeadlessGuestAdapterConfig(StrictModel):
    """Immutable inputs for one organizer-owned external-harness adapter state."""

    schema_version: Literal['vaxreplay.headless-guest-adapter-config.dev-v0.1'] = (
        HEADLESS_GUEST_ADAPTER_CONFIG_SCHEMA_VERSION
    )
    contract_id: Literal['vaxreplay-lane-a-headless-guest-adapter-v0.1'] = HEADLESS_GUEST_ADAPTER_CONTRACT_ID
    family: HarnessFamily
    invocation_protocol: HeadlessInvocationProtocol
    adapter_executable_path: Literal['/opt/vaxreplay/bin/vaxreplay-headless-guest-adapter'] = (
        HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH
    )
    adapter_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_executable_path: str = Field(min_length=2, max_length=4096)
    vendor_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    complete_dependency_closure_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_reported_version: str = Field(min_length=1, max_length=500)
    vendor_version_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_config_template_sha256: str = Field(pattern=_SHA256_PATTERN)
    vendor_argv_template: tuple[str, ...] = Field(min_length=2, max_length=128)
    response_channel: HeadlessResponseChannel
    fixed_workspace_path: Literal['/run/vaxreplay/workspace'] = '/run/vaxreplay/workspace'
    fixed_scratch_path: Literal['/run/vaxreplay/scratch'] = '/run/vaxreplay/scratch'
    fixed_output_path: Literal['/run/vaxreplay/output/final-response'] = '/run/vaxreplay/output/final-response'
    local_shell_enabled: bool
    rpc_methods: tuple[
        Literal['list_workspace', 'read_workspace', 'search_workspace', 'model_generate', 'submit'],
        ...,
    ] = (
        'list_workspace',
        'read_workspace',
        'search_workspace',
        'model_generate',
        'submit',
    )
    provider_transport: Literal['guest_loopback_api_shim_to_authenticated_model_generate_rpc'] = (
        'guest_loopback_api_shim_to_authenticated_model_generate_rpc'
    )
    tool_transport: Literal['guest_local_bridge_to_authenticated_workspace_and_submit_rpc'] = (
        'guest_local_bridge_to_authenticated_workspace_and_submit_rpc'
    )
    no_shell_command_construction: Literal[True] = True
    inherited_environment_allowed: Literal[False] = False
    ambient_user_or_project_config_allowed: Literal[False] = False
    provider_credentials_in_guest_allowed: Literal[False] = False
    generic_http_proxy_allowed: Literal[False] = False
    internet_or_lan_allowed: Literal[False] = False
    organizer_route_selected_outside_harness: Literal[True] = True
    outer_rpc_events_remain_authoritative: Literal[True] = True
    adapter_implementation_checked_in: bool = False
    provider_shim_implementation_checked_in: bool = False
    workspace_materialization_bridge_implementation_checked_in: bool = False
    linux_kvm_qualified: Literal[False] = False
    development_only: Literal[True] = True

    @field_validator('vendor_executable_path')
    @classmethod
    def validate_vendor_executable_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or '..' in path.parts
            or path.as_posix() != value
            or not value.startswith('/opt/vaxreplay/')
        ):
            raise ValueError('vendor executable must be a normalized path on the read-only harness disk')
        return value

    @field_validator('vendor_argv_template')
    @classmethod
    def validate_vendor_argv_template(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or '\x00' in item or len(item.encode('utf-8')) > 4096 for item in value):
            raise ValueError('vendor argv template contains an empty, NUL, or oversized value')
        if any(
            item in _FORBIDDEN_VENDOR_ARGUMENTS
            or any(item.startswith(f'{forbidden}=') for forbidden in _FORBIDDEN_VENDOR_ARGUMENTS)
            or item.startswith('--mcp-config=')
            for item in value
        ):
            raise ValueError('vendor argv template contains a credential, egress, persistence, or bypass flag')
        if any(item.startswith(('-c=', '--config=')) for item in value):
            raise ValueError('Codex config overrides must use separately pinned argv elements')
        if value.count(MODEL_SELECTOR_TOKEN) != 1:
            raise ValueError('vendor argv template must contain exactly one organizer model selector token')
        unknown_tokens = {item for item in value if item.startswith('__VAXREPLAY_')} - set(_RUNTIME_TOKENS)
        if unknown_tokens or any(value.count(token) > 1 for token in _RUNTIME_TOKENS):
            raise ValueError('vendor argv template contains an unknown or repeated runtime token')
        return value

    @model_validator(mode='after')
    def validate_family_protocol_and_surface(self) -> Self:
        expected = {
            HarnessFamily.CODEX: HeadlessInvocationProtocol.CODEX_EXEC,
            HarnessFamily.CLAUDE_CODE: HeadlessInvocationProtocol.CLAUDE_PRINT,
            HarnessFamily.CURSOR: HeadlessInvocationProtocol.CURSOR_PRINT,
            HarnessFamily.CUSTOM: HeadlessInvocationProtocol.CUSTOM_HEADLESS,
        }
        if self.family not in expected:
            raise ValueError('the benchmark-native harness does not use the external headless adapter')
        if self.invocation_protocol != expected[self.family]:
            raise ValueError('headless invocation protocol does not match the declared harness family')
        if self.vendor_argv_template[0] != self.vendor_executable_path:
            raise ValueError('vendor argv[0] must be the exact pinned vendor executable')
        if self.vendor_argv_template.count(WORKSPACE_PATH_TOKEN) != 1:
            raise ValueError('vendor argv template must contain exactly one sealed workspace path token')
        output_token_count = self.vendor_argv_template.count(OUTPUT_PATH_TOKEN)
        if self.response_channel == HeadlessResponseChannel.BOUNDED_OUTPUT_FILE and output_token_count != 1:
            raise ValueError('a bounded-output-file protocol requires exactly one fixed output path token')
        if (
            self.response_channel
            in {
                HeadlessResponseChannel.BOUNDED_JSON_STDOUT,
                HeadlessResponseChannel.BOUNDED_JSONL_STDOUT,
            }
            and output_token_count != 0
        ):
            raise ValueError('a stdout response protocol cannot also name an output file token')
        if tuple(self.rpc_methods) != _ALL_RPC_METHODS:
            raise ValueError('the adapter may bridge exactly the existing sealed RPC method set')
        response_by_protocol = {
            HeadlessInvocationProtocol.CODEX_EXEC: HeadlessResponseChannel.BOUNDED_OUTPUT_FILE,
            HeadlessInvocationProtocol.CLAUDE_PRINT: HeadlessResponseChannel.BOUNDED_JSON_STDOUT,
            HeadlessInvocationProtocol.CURSOR_PRINT: HeadlessResponseChannel.BOUNDED_JSONL_STDOUT,
        }
        expected_response = response_by_protocol.get(self.invocation_protocol)
        if expected_response is not None and self.response_channel != expected_response:
            raise ValueError('response channel does not match the pinned vendor headless protocol')
        config_overrides = tuple(
            self.vendor_argv_template[index + 1]
            for index, argument in enumerate(self.vendor_argv_template[:-1])
            if argument in {'-c', '--config'}
        )
        dangling_config = self.vendor_argv_template[-1] in {'-c', '--config'}
        if dangling_config:
            raise ValueError('Codex config override flag is missing its separately pinned value')
        if any(
            argument in {'-c', '--config'} and index > 0 and self.vendor_argv_template[index - 1] in {'-c', '--config'}
            for index, argument in enumerate(self.vendor_argv_template)
        ):
            raise ValueError('Codex config override values cannot themselves be config flags')
        if config_overrides and self.family != HarnessFamily.CODEX:
            raise ValueError('only the Codex adapter contract permits fixed config overrides')
        if config_overrides and config_overrides != CODEX_SEALED_CONFIG_OVERRIDES:
            raise ValueError('Codex config overrides must equal the complete sealed override set')
        if self.adapter_implementation_checked_in:
            if self.family not in {HarnessFamily.CODEX, HarnessFamily.CLAUDE_CODE}:
                raise ValueError('this harness family has no checked-in development adapter')
            if not (
                self.provider_shim_implementation_checked_in
                and self.workspace_materialization_bridge_implementation_checked_in
            ):
                raise ValueError('a checked-in adapter requires both checked-in bridges')
            if self.family == HarnessFamily.CODEX:
                if not self.local_shell_enabled:
                    raise ValueError('the Codex adapter requires its local shell')
                if config_overrides != CODEX_SEALED_CONFIG_OVERRIDES:
                    raise ValueError('the checked-in Codex adapter requires the exact sealed config override set')
            elif self.local_shell_enabled:
                raise ValueError('the checked-in Claude adapter exposes Read but no local shell')
            if self.family == HarnessFamily.CLAUDE_CODE:
                expected_mcp = ('--mcp-config', '{"mcpServers":{}}')
                observed_mcp = tuple(
                    self.vendor_argv_template[index : index + 2]
                    for index, argument in enumerate(self.vendor_argv_template[:-1])
                    if argument == '--mcp-config'
                )
                if observed_mcp != (expected_mcp,):
                    raise ValueError('the checked-in Claude adapter requires one exact empty MCP config')
                if self.vendor_argv_template.count(SUBMISSION_SCHEMA_JSON_TOKEN) != 1:
                    raise ValueError('the checked-in Claude adapter requires one task-bound submission schema token')
        elif self.provider_shim_implementation_checked_in or (
            self.workspace_materialization_bridge_implementation_checked_in
        ):
            raise ValueError('a checked-in bridge requires the outer adapter implementation')
        return self


def headless_guest_adapter_config_sha256(config: HeadlessGuestAdapterConfig) -> str:
    canonical = HeadlessGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def render_headless_vendor_argv(
    config: HeadlessGuestAdapterConfig,
    *,
    organizer_model_selector: str,
    submission_schema_json: str | None = None,
) -> tuple[str, ...]:
    """Substitute typed values as argv elements; never invoke a shell or accept a provider route."""

    canonical = HeadlessGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
    if not organizer_model_selector or '\x00' in organizer_model_selector or len(organizer_model_selector) > 500:
        raise SubmittedHarnessError('organizer model selector is invalid')
    substitutions = {
        MODEL_SELECTOR_TOKEN: organizer_model_selector,
        WORKSPACE_PATH_TOKEN: canonical.fixed_workspace_path,
        SCRATCH_PATH_TOKEN: canonical.fixed_scratch_path,
        OUTPUT_PATH_TOKEN: canonical.fixed_output_path,
    }
    if SUBMISSION_SCHEMA_JSON_TOKEN in canonical.vendor_argv_template:
        if not submission_schema_json or '\x00' in submission_schema_json:
            raise SubmittedHarnessError('task-bound submission schema JSON is invalid')
        substitutions[SUBMISSION_SCHEMA_JSON_TOKEN] = submission_schema_json
    return tuple(substitutions.get(argument, argument) for argument in canonical.vendor_argv_template)


def require_headless_guest_adapter_binding(
    *,
    config: HeadlessGuestAdapterConfig,
    manifest: SubmittedHarnessManifest,
) -> None:
    """Cross-bind the config to its contract-only or development-integrated manifest."""

    canonical = HeadlessGuestAdapterConfig.model_validate_json(canonical_json_bytes(config))
    submitted = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(manifest))
    config_sha256 = headless_guest_adapter_config_sha256(canonical)
    expected_argv = (
        HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
        HEADLESS_GUEST_ADAPTER_CONFIG_DIGEST_FLAG,
        config_sha256,
    )
    if submitted.family != canonical.family:
        raise SubmittedHarnessError('headless adapter family differs from the submitted harness manifest')
    expected_support = (
        HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED
        if canonical.adapter_implementation_checked_in
        else HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED
    )
    if (
        submitted.execution_mode != HarnessExecutionMode.SUBMITTED_GUEST_AGENT
        or submitted.runtime_support != expected_support
    ):
        raise SubmittedHarnessError(
            'external headless adapter support does not match its checked-in implementation state'
        )
    if (
        submitted.guest_executable_path != canonical.adapter_executable_path
        or submitted.guest_executable_sha256 != canonical.adapter_executable_sha256
        or submitted.guest_argv != expected_argv
        or submitted.baked_config_sha256 != config_sha256
        or submitted.dependency_closure_sha256 != canonical.complete_dependency_closure_sha256
    ):
        raise SubmittedHarnessError('submitted harness manifest does not bind the exact headless adapter config')
    if (
        not submitted.interface.guest_local_subprocesses_allowed
        or submitted.interface.guest_local_shell_allowed != canonical.local_shell_enabled
    ):
        raise SubmittedHarnessError('submitted harness interface differs from the headless adapter process surface')


__all__ = [
    'HEADLESS_GUEST_ADAPTER_CONFIG_DIGEST_FLAG',
    'HEADLESS_GUEST_ADAPTER_CONTRACT_ID',
    'HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH',
    'CODEX_RESPONSES_SHIM_PORT',
    'CODEX_SEALED_CONFIG_OVERRIDES',
    'MODEL_SELECTOR_TOKEN',
    'OUTPUT_PATH_TOKEN',
    'SCRATCH_PATH_TOKEN',
    'SUBMISSION_SCHEMA_JSON_TOKEN',
    'WORKSPACE_PATH_TOKEN',
    'HeadlessGuestAdapterConfig',
    'HeadlessInvocationProtocol',
    'HeadlessResponseChannel',
    'headless_guest_adapter_config_sha256',
    'render_headless_vendor_argv',
    'require_headless_guest_adapter_binding',
]
