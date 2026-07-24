from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from vaxreplay.agentic.claude_code_guest_adapter import (
    CLAUDE_CODE_SUPPORTED_VENDOR_VERSION,
    CLAUDE_CODE_VENDOR_EXECUTABLE_PATH,
    claude_code_vendor_argv_template,
)
from vaxreplay.agentic.codex_guest_adapter import (
    CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
    CODEX_VENDOR_EXECUTABLE_PATH,
    codex_vendor_argv_template,
)
from vaxreplay.agentic.guest_boot_dispatch import (
    HEADLESS_GUEST_CONFIG_PATH,
    HEADLESS_GUEST_EXECUTABLE_PATH,
    GuestBootConfigSchema,
    GuestBootDispatchAdmission,
    GuestBootDispatchError,
    GuestBootDispatchManifest,
    guest_boot_dispatch_manifest_sha256,
    load_pinned_guest_boot_dispatch_manifest,
    make_native_guest_boot_dispatch_manifest,
    require_guest_boot_dispatch_binding,
    validate_guest_boot_config_bytes,
)
from vaxreplay.agentic.headless_guest_adapter import (
    HeadlessGuestAdapterConfig,
    HeadlessInvocationProtocol,
    HeadlessResponseChannel,
    headless_guest_adapter_config_sha256,
)
from vaxreplay.agentic.submitted_harness import (
    HarnessExecutionMode,
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessInterface,
    SubmittedHarnessManifest,
)
from vaxreplay.bundle import canonical_json_bytes


def _headless_config(family: HarnessFamily) -> HeadlessGuestAdapterConfig:
    if family == HarnessFamily.CODEX:
        return HeadlessGuestAdapterConfig(
            family=family,
            invocation_protocol=HeadlessInvocationProtocol.CODEX_EXEC,
            adapter_executable_sha256='1' * 64,
            vendor_executable_path=CODEX_VENDOR_EXECUTABLE_PATH,
            vendor_executable_sha256='2' * 64,
            complete_dependency_closure_sha256='3' * 64,
            vendor_reported_version=CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
            vendor_version_output_sha256='4' * 64,
            vendor_config_template_sha256='5' * 64,
            vendor_argv_template=codex_vendor_argv_template(),
            response_channel=HeadlessResponseChannel.BOUNDED_OUTPUT_FILE,
            local_shell_enabled=True,
            adapter_implementation_checked_in=True,
            provider_shim_implementation_checked_in=True,
            workspace_materialization_bridge_implementation_checked_in=True,
        )
    if family == HarnessFamily.CLAUDE_CODE:
        return HeadlessGuestAdapterConfig(
            family=family,
            invocation_protocol=HeadlessInvocationProtocol.CLAUDE_PRINT,
            adapter_executable_sha256='1' * 64,
            vendor_executable_path=CLAUDE_CODE_VENDOR_EXECUTABLE_PATH,
            vendor_executable_sha256='2' * 64,
            complete_dependency_closure_sha256='3' * 64,
            vendor_reported_version=CLAUDE_CODE_SUPPORTED_VENDOR_VERSION,
            vendor_version_output_sha256='4' * 64,
            vendor_config_template_sha256='5' * 64,
            vendor_argv_template=claude_code_vendor_argv_template(),
            response_channel=HeadlessResponseChannel.BOUNDED_JSON_STDOUT,
            local_shell_enabled=False,
            adapter_implementation_checked_in=True,
            provider_shim_implementation_checked_in=True,
            workspace_materialization_bridge_implementation_checked_in=True,
        )
    raise AssertionError('test helper supports only checked-in development adapters')


def _development_dispatch(
    config: HeadlessGuestAdapterConfig,
) -> GuestBootDispatchManifest:
    digest = headless_guest_adapter_config_sha256(config)
    return GuestBootDispatchManifest(
        family=config.family,
        runtime_support=HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED,
        admission=GuestBootDispatchAdmission.DEVELOPMENT_PACKAGING_ONLY,
        config_schema=GuestBootConfigSchema.HEADLESS_ADAPTER,
        guest_executable_path=HEADLESS_GUEST_EXECUTABLE_PATH,
        guest_executable_sha256=config.adapter_executable_sha256,
        guest_config_path=HEADLESS_GUEST_CONFIG_PATH,
        guest_config_sha256=digest,
        guest_argv=(
            HEADLESS_GUEST_EXECUTABLE_PATH,
            '--expected-config-sha256',
            digest,
        ),
    )


def _submitted(
    config: HeadlessGuestAdapterConfig,
    dispatch: GuestBootDispatchManifest,
) -> SubmittedHarnessManifest:
    return SubmittedHarnessManifest(
        harness_id=f'{config.family.value}-development-dispatch',
        harness_version='dev-v0.1',
        family=config.family,
        execution_mode=HarnessExecutionMode.SUBMITTED_GUEST_AGENT,
        runtime_support=HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED,
        harness_image_sha256='6' * 64,
        harness_image_byte_count=4096,
        normalized_runtime_tree_sha256='7' * 64,
        guest_executable_path=dispatch.guest_executable_path,
        guest_executable_sha256=dispatch.guest_executable_sha256,
        guest_argv=dispatch.guest_argv,
        baked_config_sha256=dispatch.guest_config_sha256,
        dependency_closure_sha256=config.complete_dependency_closure_sha256,
        reproducible_build_receipt_sha256='8' * 64,
        interface=SubmittedHarnessInterface(
            guest_local_subprocesses_allowed=True,
            guest_local_shell_allowed=config.local_shell_enabled,
        ),
        display_name=f'{config.family.value} development dispatch',
        submitter='fixture',
    )


@pytest.mark.parametrize('family', (HarnessFamily.CODEX, HarnessFamily.CLAUDE_CODE))
def test_checked_in_external_adapters_have_packageable_but_unqualified_dispatch(
    family: HarnessFamily,
) -> None:
    config = _headless_config(family)
    dispatch = _development_dispatch(config)
    payload = canonical_json_bytes(config)

    validate_guest_boot_config_bytes(dispatch, payload)
    require_guest_boot_dispatch_binding(
        dispatch=dispatch,
        submitted_harness=_submitted(config, dispatch),
    )

    assert dispatch.admission == GuestBootDispatchAdmission.DEVELOPMENT_PACKAGING_ONLY
    assert dispatch.linux_kvm_qualification_claimed is False
    assert dispatch.guest_environment == ()
    assert dispatch.submitted_command_string_or_shell_construction_allowed is False


def test_native_dispatch_is_runtime_integrated_but_carries_no_qualification_claim() -> None:
    dispatch = make_native_guest_boot_dispatch_manifest(
        guest_executable_sha256='1' * 64,
        guest_config_sha256='2' * 64,
    )

    assert dispatch.admission == (GuestBootDispatchAdmission.RUNTIME_INTEGRATED_REQUIRES_EXTERNAL_QUALIFICATION)
    assert dispatch.linux_kvm_qualification_claimed is False
    assert dispatch.image_and_receipt_bindings_are_external is True


def test_cursor_and_command_string_dispatch_fail_closed() -> None:
    config = _headless_config(HarnessFamily.CODEX)
    payload = _development_dispatch(config).model_dump(mode='python')
    payload['family'] = HarnessFamily.CURSOR
    with pytest.raises(ValidationError, match='no bootable guest-dispatch'):
        GuestBootDispatchManifest.model_validate(payload)

    payload = _development_dispatch(config).model_dump(mode='python')
    payload['guest_argv'] = (
        HEADLESS_GUEST_EXECUTABLE_PATH,
        '--expected-config-sha256',
        '$(ambient-command)',
    )
    with pytest.raises(ValidationError, match='shell-inert'):
        GuestBootDispatchManifest.model_validate(payload)


def test_manifest_loader_requires_external_digest_and_exact_canonical_bytes(
    tmp_path: Path,
) -> None:
    dispatch = _development_dispatch(_headless_config(HarnessFamily.CODEX))
    payload = canonical_json_bytes(dispatch)
    path = tmp_path / 'dispatch.json'
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    assert load_pinned_guest_boot_dispatch_manifest(path, expected_sha256=digest) == dispatch
    with pytest.raises(GuestBootDispatchError, match='external pin'):
        load_pinned_guest_boot_dispatch_manifest(path, expected_sha256='f' * 64)

    path.write_bytes(payload + b'\n')
    with pytest.raises(GuestBootDispatchError, match='canonical'):
        load_pinned_guest_boot_dispatch_manifest(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_dispatch_digest_changes_for_executable_config_or_argv_identity() -> None:
    base = make_native_guest_boot_dispatch_manifest(
        guest_executable_sha256='1' * 64,
        guest_config_sha256='2' * 64,
    )
    changed_executable = base.model_copy(update={'guest_executable_sha256': '3' * 64})
    changed_config_payload = base.model_dump(mode='python')
    changed_config_payload['guest_config_sha256'] = '4' * 64
    changed_config_payload['guest_argv'] = (
        base.guest_executable_path,
        '--expected-config-sha256',
        '4' * 64,
    )
    changed_config = GuestBootDispatchManifest.model_validate(changed_config_payload)

    assert guest_boot_dispatch_manifest_sha256(base) != guest_boot_dispatch_manifest_sha256(changed_executable)
    assert guest_boot_dispatch_manifest_sha256(base) != guest_boot_dispatch_manifest_sha256(changed_config)
