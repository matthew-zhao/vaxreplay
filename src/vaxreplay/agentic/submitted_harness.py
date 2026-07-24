"""Versioned artifact contract for model-only and submitted agent-harness systems.

This module distinguishes a harness implementation from the model route it uses.  The existing
Lane A appliance is the only production-runtime-integrated harness today.  Codex and Claude Code
have checked-in development guest adapters that still fail official admission; Cursor and custom
agents remain contract-only.  Every external harness must use the same narrow RPC and submission
boundary and pass independent Linux/KVM qualification before production admission.
"""

from __future__ import annotations

import enum
import hashlib
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.guest_rpc import (
    GUEST_RPC_REQUEST_SCHEMA_VERSION,
    GUEST_RPC_RESPONSE_SCHEMA_VERSION,
    GuestRpcMethod,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.run_artifact import AgenticHarnessIdentity
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

SUBMITTED_HARNESS_INTERFACE_SCHEMA_VERSION = 'vaxreplay.submitted-harness-interface.dev-v0.1'
SUBMITTED_HARNESS_MANIFEST_SCHEMA_VERSION = 'vaxreplay.submitted-harness-manifest.dev-v0.1'
SUBMITTED_HARNESS_CONTRACT_ID = 'vaxreplay-lane-a-submitted-harness-v0.1'
NATIVE_LANE_A_GUEST_EXECUTABLE_PATH = '/opt/vaxreplay/bin/vaxreplay-lane-a-clinical-guest'
NATIVE_LANE_A_CONFIG_DIGEST_FLAG = '--expected-config-sha256'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_ID_PATTERN = r'^[a-z0-9][a-z0-9._-]*$'
_ALL_RPC_METHODS = tuple(method.value for method in GuestRpcMethod)


class SubmittedHarnessError(ValueError):
    """A harness artifact or evaluated-system binding failed closed."""


class HarnessExecutionMode(str, enum.Enum):
    FIXED_MODEL_LOOP = 'fixed_model_loop'
    SUBMITTED_GUEST_AGENT = 'submitted_guest_agent'


class HarnessFamily(str, enum.Enum):
    VAXREPLAY_NATIVE = 'vaxreplay_native'
    CODEX = 'codex'
    CLAUDE_CODE = 'claude_code'
    CURSOR = 'cursor'
    CUSTOM = 'custom'


class HarnessRuntimeSupport(str, enum.Enum):
    RUNTIME_INTEGRATED = 'runtime_integrated'
    DEVELOPMENT_ADAPTER_INTEGRATED = 'development_adapter_integrated'
    CONTRACT_ONLY_ADAPTER_REQUIRED = 'contract_only_adapter_required'


class SubmittedHarnessInterface(StrictModel):
    """The complete host-facing capability surface a guest harness may use."""

    schema_version: Literal['vaxreplay.submitted-harness-interface.dev-v0.1'] = (
        SUBMITTED_HARNESS_INTERFACE_SCHEMA_VERSION
    )
    contract_id: Literal['vaxreplay-lane-a-submitted-harness-v0.1'] = SUBMITTED_HARNESS_CONTRACT_ID
    request_schema_version: Literal['vaxreplay.guest-rpc-request.v0.1'] = GUEST_RPC_REQUEST_SCHEMA_VERSION
    response_schema_version: Literal['vaxreplay.guest-rpc-response.v0.1'] = GUEST_RPC_RESPONSE_SCHEMA_VERSION
    transport: Literal['one_authenticated_af_vsock_stream'] = 'one_authenticated_af_vsock_stream'
    rpc_methods: tuple[
        Literal['list_workspace'],
        Literal['read_workspace'],
        Literal['search_workspace'],
        Literal['model_generate'],
        Literal['submit'],
    ] = (
        'list_workspace',
        'read_workspace',
        'search_workspace',
        'model_generate',
        'submit',
    )
    workspace_access: Literal['brokered_exact_bytes_only'] = 'brokered_exact_bytes_only'
    model_access: Literal['organizer_gateway_only'] = 'organizer_gateway_only'
    final_answer_channel: Literal['one_validated_submit_rpc'] = 'one_validated_submit_rpc'
    provider_credentials_visible_to_guest: Literal[False] = False
    internet_or_lan_access: Literal[False] = False
    host_filesystem_mounted: Literal[False] = False
    hidden_gold_mounted: Literal[False] = False
    direct_registry_identifier_lookup: Literal[False] = False
    local_computation_and_scratch_allowed: Literal[True] = True
    guest_local_subprocesses_allowed: bool
    guest_local_shell_allowed: bool
    host_observed_rpc_events_authoritative: Literal[True] = True
    local_reasoning_or_computation_fully_observable: Literal[False] = False
    one_fresh_microvm_per_attempt: Literal[True] = True
    automatic_retry_after_terminal_failure: Literal[False] = False

    @model_validator(mode='after')
    def validate_surface(self) -> Self:
        if tuple(self.rpc_methods) != _ALL_RPC_METHODS:
            raise ValueError('submitted harness interface must expose exactly the fixed RPC method set')
        if self.guest_local_shell_allowed and not self.guest_local_subprocesses_allowed:
            raise ValueError('a local shell requires local subprocess support')
        return self


class SubmittedHarnessManifest(StrictModel):
    """Exact immutable identity for one guest harness artifact, separate from its model."""

    schema_version: Literal['vaxreplay.submitted-harness-manifest.dev-v0.1'] = SUBMITTED_HARNESS_MANIFEST_SCHEMA_VERSION
    harness_id: str = Field(pattern=_ID_PATTERN)
    harness_version: str = Field(min_length=1, max_length=200)
    family: HarnessFamily
    execution_mode: HarnessExecutionMode
    runtime_support: HarnessRuntimeSupport
    harness_image_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_image_byte_count: int = Field(gt=0, le=1024 * 1024 * 1024 * 1024)
    normalized_runtime_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_executable_path: str = Field(min_length=2, max_length=4096)
    guest_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_argv: tuple[str, ...] = Field(min_length=1, max_length=128)
    baked_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    dependency_closure_sha256: str = Field(pattern=_SHA256_PATTERN)
    reproducible_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    interface: SubmittedHarnessInterface
    response_protocol: Literal[AgenticResponseProtocol.CLINICAL_EXECUTION] = AgenticResponseProtocol.CLINICAL_EXECUTION
    display_name: str = Field(min_length=1, max_length=500)
    submitter: str = Field(min_length=1, max_length=500)
    exact_image_verified_before_launch_required: Literal[True] = True
    linux_kvm_qualification_required: Literal[True] = True
    harness_specific_adapter_qualification_required: Literal[True] = True
    provider_route_bound_separately: Literal[True] = True
    caller_claimed_vendor_name_is_not_qualification: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @field_validator('guest_executable_path')
    @classmethod
    def validate_executable_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or path.as_posix() != value:
            raise ValueError('guest executable path must be absolute and normalized')
        return value

    @field_validator('guest_argv')
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or '\x00' in item or len(item.encode('utf-8')) > 4096 for item in value):
            raise ValueError('guest argv contains an empty, NUL, or oversized value')
        return value

    @model_validator(mode='after')
    def validate_family_and_runtime_support(self) -> Self:
        if self.guest_argv[0] != self.guest_executable_path:
            raise ValueError('guest argv[0] must be the exact pinned executable path')
        native = self.family == HarnessFamily.VAXREPLAY_NATIVE
        fixed = self.execution_mode == HarnessExecutionMode.FIXED_MODEL_LOOP
        if native != fixed:
            raise ValueError('only the VaxReplay-native family may use the fixed model loop')
        if native and self.runtime_support != HarnessRuntimeSupport.RUNTIME_INTEGRATED:
            raise ValueError('the benchmark-native harness must use its runtime-integrated implementation')
        if not native and self.runtime_support == HarnessRuntimeSupport.RUNTIME_INTEGRATED:
            raise ValueError(
                'runtime support cannot claim production integration for an external harness before qualification'
            )
        if self.runtime_support == HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED and self.family not in {
            HarnessFamily.CODEX,
            HarnessFamily.CLAUDE_CODE,
        }:
            raise ValueError('only a family with a checked-in development adapter may claim development integration')
        if fixed and (self.interface.guest_local_shell_allowed or self.interface.guest_local_subprocesses_allowed):
            raise ValueError('the fixed model loop cannot claim a local shell or subprocess surface')
        if fixed and (
            self.guest_executable_path != NATIVE_LANE_A_GUEST_EXECUTABLE_PATH
            or self.guest_argv
            != (
                NATIVE_LANE_A_GUEST_EXECUTABLE_PATH,
                NATIVE_LANE_A_CONFIG_DIGEST_FLAG,
                self.baked_config_sha256,
            )
        ):
            raise ValueError('the fixed model loop must use the reproducible rootfs init argv')
        if not fixed and not self.interface.guest_local_subprocesses_allowed:
            raise ValueError('a submitted guest agent requires its pinned local executable process')
        return self


def submitted_harness_manifest_sha256(manifest: SubmittedHarnessManifest) -> str:
    validated = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(manifest))
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


def submitted_harness_behavior_sha256(manifest: SubmittedHarnessManifest) -> str:
    """Alias-resistant runtime identity; display/vendor labels and submitter are excluded."""

    validated = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(manifest))
    behavior = {
        'schema_version': 'vaxreplay.submitted-harness-behavior.dev-v0.1',
        'execution_mode': validated.execution_mode.value,
        'harness_image_sha256': validated.harness_image_sha256,
        'harness_image_byte_count': validated.harness_image_byte_count,
        'normalized_runtime_tree_sha256': validated.normalized_runtime_tree_sha256,
        'guest_executable_path': validated.guest_executable_path,
        'guest_executable_sha256': validated.guest_executable_sha256,
        'guest_argv': validated.guest_argv,
        'baked_config_sha256': validated.baked_config_sha256,
        'dependency_closure_sha256': validated.dependency_closure_sha256,
        'interface': validated.interface.model_dump(mode='json'),
        'response_protocol': validated.response_protocol.value,
    }
    return hashlib.sha256(canonical_json_bytes(behavior)).hexdigest()


def make_agentic_harness_identity(
    *,
    manifest: SubmittedHarnessManifest,
    requested_model_id: str,
    adapter_id: str,
) -> AgenticHarnessIdentity:
    validated = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(manifest))
    return AgenticHarnessIdentity(
        harness_id=validated.harness_id,
        harness_version=validated.harness_version,
        harness_image_or_commitment=f'sha256:{validated.harness_image_sha256}',
        harness_manifest_sha256=submitted_harness_manifest_sha256(validated),
        harness_behavior_sha256=submitted_harness_behavior_sha256(validated),
        harness_execution_mode=validated.execution_mode.value,
        requested_model_id=requested_model_id,
        adapter_id=adapter_id,
    )


def require_submitted_harness_binding(
    *,
    manifest: SubmittedHarnessManifest,
    identity: AgenticHarnessIdentity,
    worker_harness_sha256: str,
    worker_harness_byte_count: int,
    logical_model_id: str,
    adapter_id: str,
    require_runtime_integrated: bool = True,
) -> None:
    validated = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(manifest))
    expected = make_agentic_harness_identity(
        manifest=validated,
        requested_model_id=logical_model_id,
        adapter_id=adapter_id,
    )
    if identity != expected:
        raise SubmittedHarnessError('harness identity does not bind the exact submitted manifest and model route')
    if (
        validated.harness_image_sha256,
        validated.harness_image_byte_count,
    ) != (
        worker_harness_sha256,
        worker_harness_byte_count,
    ):
        raise SubmittedHarnessError('submitted harness manifest differs from the worker harness disk')
    if require_runtime_integrated and validated.runtime_support != HarnessRuntimeSupport.RUNTIME_INTEGRATED:
        raise SubmittedHarnessError(
            'submitted harness contract is valid, but its guest adapter is not runtime-integrated or qualified'
        )


def harness_family_support_matrix() -> tuple[tuple[HarnessFamily, HarnessRuntimeSupport], ...]:
    """Machine-readable, fail-closed implementation state for every harness family."""

    return tuple(
        (
            family,
            (
                HarnessRuntimeSupport.RUNTIME_INTEGRATED
                if family == HarnessFamily.VAXREPLAY_NATIVE
                else HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED
                if family in {HarnessFamily.CODEX, HarnessFamily.CLAUDE_CODE}
                else HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED
            ),
        )
        for family in HarnessFamily
    )


__all__ = [
    'SUBMITTED_HARNESS_CONTRACT_ID',
    'NATIVE_LANE_A_CONFIG_DIGEST_FLAG',
    'NATIVE_LANE_A_GUEST_EXECUTABLE_PATH',
    'HarnessExecutionMode',
    'HarnessFamily',
    'HarnessRuntimeSupport',
    'SubmittedHarnessError',
    'SubmittedHarnessInterface',
    'SubmittedHarnessManifest',
    'harness_family_support_matrix',
    'make_agentic_harness_identity',
    'require_submitted_harness_binding',
    'submitted_harness_behavior_sha256',
    'submitted_harness_manifest_sha256',
]
