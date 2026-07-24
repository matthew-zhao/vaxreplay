from __future__ import annotations

import pytest
from pydantic import ValidationError

from vaxreplay.agentic.headless_guest_adapter import (
    HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
    MODEL_SELECTOR_TOKEN,
    OUTPUT_PATH_TOKEN,
    WORKSPACE_PATH_TOKEN,
    HeadlessGuestAdapterConfig,
    HeadlessInvocationProtocol,
    HeadlessResponseChannel,
    headless_guest_adapter_config_sha256,
    render_headless_vendor_argv,
    require_headless_guest_adapter_binding,
)
from vaxreplay.agentic.submitted_harness import (
    HarnessExecutionMode,
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessError,
    SubmittedHarnessInterface,
    SubmittedHarnessManifest,
)


def _config(*, family: HarnessFamily = HarnessFamily.CODEX) -> HeadlessGuestAdapterConfig:
    protocol = {
        HarnessFamily.CODEX: HeadlessInvocationProtocol.CODEX_EXEC,
        HarnessFamily.CLAUDE_CODE: HeadlessInvocationProtocol.CLAUDE_PRINT,
        HarnessFamily.CURSOR: HeadlessInvocationProtocol.CURSOR_PRINT,
    }[family]
    response = {
        HarnessFamily.CODEX: HeadlessResponseChannel.BOUNDED_OUTPUT_FILE,
        HarnessFamily.CLAUDE_CODE: HeadlessResponseChannel.BOUNDED_JSON_STDOUT,
        HarnessFamily.CURSOR: HeadlessResponseChannel.BOUNDED_JSONL_STDOUT,
    }[family]
    executable = f'/opt/vaxreplay/vendor/{family.value}/agent'
    argv = [
        executable,
        '--model',
        MODEL_SELECTOR_TOKEN,
        '--workspace',
        WORKSPACE_PATH_TOKEN,
    ]
    if response == HeadlessResponseChannel.BOUNDED_OUTPUT_FILE:
        argv.extend(('--output', OUTPUT_PATH_TOKEN))
    return HeadlessGuestAdapterConfig(
        family=family,
        invocation_protocol=protocol,
        adapter_executable_sha256='1' * 64,
        vendor_executable_path=executable,
        vendor_executable_sha256='2' * 64,
        complete_dependency_closure_sha256='3' * 64,
        vendor_reported_version='pinned-local-version',
        vendor_version_output_sha256='4' * 64,
        vendor_config_template_sha256='5' * 64,
        vendor_argv_template=tuple(argv),
        response_channel=response,
        local_shell_enabled=True,
    )


def _manifest(config: HeadlessGuestAdapterConfig) -> SubmittedHarnessManifest:
    digest = headless_guest_adapter_config_sha256(config)
    return SubmittedHarnessManifest(
        harness_id=f'{config.family.value}-headless-adapter',
        harness_version='1',
        family=config.family,
        execution_mode=HarnessExecutionMode.SUBMITTED_GUEST_AGENT,
        runtime_support=HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED,
        harness_image_sha256='6' * 64,
        harness_image_byte_count=4096,
        normalized_runtime_tree_sha256='7' * 64,
        guest_executable_path=HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
        guest_executable_sha256=config.adapter_executable_sha256,
        guest_argv=(
            HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
            '--expected-config-sha256',
            digest,
        ),
        baked_config_sha256=digest,
        dependency_closure_sha256=config.complete_dependency_closure_sha256,
        reproducible_build_receipt_sha256='8' * 64,
        interface=SubmittedHarnessInterface(
            guest_local_subprocesses_allowed=True,
            guest_local_shell_allowed=config.local_shell_enabled,
        ),
        display_name='Contract-only vendor harness',
        submitter='fixture',
    )


@pytest.mark.parametrize('family', (HarnessFamily.CODEX, HarnessFamily.CLAUDE_CODE, HarnessFamily.CURSOR))
def test_external_family_config_cross_binds_but_remains_contract_only(family: HarnessFamily) -> None:
    config = _config(family=family)
    manifest = _manifest(config)

    require_headless_guest_adapter_binding(config=config, manifest=manifest)
    assert manifest.runtime_support == HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED
    assert config.adapter_implementation_checked_in is False
    assert config.provider_shim_implementation_checked_in is False
    assert config.linux_kvm_qualified is False


def test_argv_substitution_is_direct_and_does_not_construct_a_shell_command() -> None:
    config = _config()
    adversarial_but_organizer_owned_selector = 'model;$(touch /tmp/not-executed)'

    argv = render_headless_vendor_argv(
        config,
        organizer_model_selector=adversarial_but_organizer_owned_selector,
    )

    assert argv[2] == adversarial_but_organizer_owned_selector
    assert argv[4] == config.fixed_workspace_path
    assert argv[6] == config.fixed_output_path
    assert len(argv) == 7


@pytest.mark.parametrize(
    'forbidden',
    (
        '--api-key',
        '--api-key=secret',
        '--search',
        '--resume',
        '--force',
        '--approve-mcps',
        '--mcp-config=/tmp/untrusted.json',
        '-c',
    ),
)
def test_contract_rejects_known_credential_egress_persistence_and_bypass_flags(forbidden: str) -> None:
    payload = _config().model_dump(mode='python')
    payload['vendor_argv_template'] = (
        payload['vendor_executable_path'],
        '--model',
        MODEL_SELECTOR_TOKEN,
        forbidden,
    )
    with pytest.raises(ValidationError, match='credential, egress, persistence, or bypass'):
        HeadlessGuestAdapterConfig.model_validate(payload)


def test_family_label_cannot_select_another_vendor_protocol() -> None:
    payload = _config().model_dump(mode='python')
    payload['invocation_protocol'] = HeadlessInvocationProtocol.CLAUDE_PRINT
    with pytest.raises(ValidationError, match='does not match'):
        HeadlessGuestAdapterConfig.model_validate(payload)


def test_contract_requires_workspace_and_protocol_specific_output_tokens() -> None:
    codex = _config()
    payload = codex.model_dump(mode='python')
    payload['vendor_argv_template'] = tuple(
        item for item in payload['vendor_argv_template'] if item != WORKSPACE_PATH_TOKEN
    )
    with pytest.raises(ValidationError, match='sealed workspace path token'):
        HeadlessGuestAdapterConfig.model_validate(payload)

    payload = codex.model_dump(mode='python')
    payload['vendor_argv_template'] = tuple(
        item for item in payload['vendor_argv_template'] if item != OUTPUT_PATH_TOKEN
    )
    with pytest.raises(ValidationError, match='requires exactly one fixed output path token'):
        HeadlessGuestAdapterConfig.model_validate(payload)

    claude = _config(family=HarnessFamily.CLAUDE_CODE)
    payload = claude.model_dump(mode='python')
    payload['vendor_argv_template'] = (*payload['vendor_argv_template'], OUTPUT_PATH_TOKEN)
    with pytest.raises(ValidationError, match='cannot also name an output file token'):
        HeadlessGuestAdapterConfig.model_validate(payload)


def test_manifest_cannot_substitute_adapter_or_claim_runtime_integration() -> None:
    config = _config()
    manifest = _manifest(config)
    substituted = manifest.model_copy(update={'guest_executable_sha256': '9' * 64})
    with pytest.raises(SubmittedHarnessError, match='exact headless adapter config'):
        require_headless_guest_adapter_binding(config=config, manifest=substituted)

    # SubmittedHarnessManifest itself rejects this unearned support claim before binding.
    payload = manifest.model_dump(mode='python')
    payload['runtime_support'] = HarnessRuntimeSupport.RUNTIME_INTEGRATED
    with pytest.raises(ValidationError, match='runtime support'):
        SubmittedHarnessManifest.model_validate(payload)
