from __future__ import annotations

import pytest
from pydantic import ValidationError

from vaxreplay.agentic.submitted_harness import (
    HarnessExecutionMode,
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessError,
    SubmittedHarnessInterface,
    SubmittedHarnessManifest,
    harness_family_support_matrix,
    make_agentic_harness_identity,
    require_submitted_harness_binding,
    submitted_harness_behavior_sha256,
    submitted_harness_manifest_sha256,
)


def _manifest(
    *,
    family: HarnessFamily = HarnessFamily.VAXREPLAY_NATIVE,
) -> SubmittedHarnessManifest:
    native = family == HarnessFamily.VAXREPLAY_NATIVE
    return SubmittedHarnessManifest(
        harness_id='lane-a-native' if native else f'{family.value}-adapter',
        harness_version='1',
        family=family,
        execution_mode=(
            HarnessExecutionMode.FIXED_MODEL_LOOP if native else HarnessExecutionMode.SUBMITTED_GUEST_AGENT
        ),
        runtime_support=(
            HarnessRuntimeSupport.RUNTIME_INTEGRATED if native else HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED
        ),
        harness_image_sha256='1' * 64,
        harness_image_byte_count=4096,
        normalized_runtime_tree_sha256='2' * 64,
        guest_executable_path=(
            '/opt/vaxreplay/bin/vaxreplay-lane-a-clinical-guest' if native else '/opt/vaxreplay/bin/agent'
        ),
        guest_executable_sha256='3' * 64,
        guest_argv=(
            (
                '/opt/vaxreplay/bin/vaxreplay-lane-a-clinical-guest',
                '--expected-config-sha256',
                '4' * 64,
            )
            if native
            else ('/opt/vaxreplay/bin/agent', '--sealed')
        ),
        baked_config_sha256='4' * 64,
        dependency_closure_sha256='5' * 64,
        reproducible_build_receipt_sha256='6' * 64,
        interface=SubmittedHarnessInterface(
            guest_local_subprocesses_allowed=not native,
            guest_local_shell_allowed=not native,
        ),
        display_name='Lane A native' if native else family.value,
        submitter='fixture-submitter',
    )


def test_manifest_and_behavior_are_separate_exact_identities() -> None:
    manifest = _manifest()
    relabeled = manifest.model_copy(
        update={
            'harness_id': 'renamed-native',
            'harness_version': 'display-v2',
            'display_name': 'A different public label',
            'submitter': 'another-submitter',
        }
    )

    assert submitted_harness_manifest_sha256(manifest) != submitted_harness_manifest_sha256(relabeled)
    assert submitted_harness_behavior_sha256(manifest) == submitted_harness_behavior_sha256(relabeled)


def test_agentic_identity_binds_manifest_behavior_image_and_model_route() -> None:
    manifest = _manifest()
    identity = make_agentic_harness_identity(
        manifest=manifest,
        requested_model_id='logical-model',
        adapter_id='provider-adapter',
    )

    assert identity.harness_manifest_sha256 == submitted_harness_manifest_sha256(manifest)
    assert identity.harness_behavior_sha256 == submitted_harness_behavior_sha256(manifest)
    assert identity.harness_execution_mode == HarnessExecutionMode.FIXED_MODEL_LOOP.value
    require_submitted_harness_binding(
        manifest=manifest,
        identity=identity,
        worker_harness_sha256=manifest.harness_image_sha256,
        worker_harness_byte_count=manifest.harness_image_byte_count,
        logical_model_id='logical-model',
        adapter_id='provider-adapter',
    )

    with pytest.raises(SubmittedHarnessError, match='model route'):
        require_submitted_harness_binding(
            manifest=manifest,
            identity=identity,
            worker_harness_sha256=manifest.harness_image_sha256,
            worker_harness_byte_count=manifest.harness_image_byte_count,
            logical_model_id='different-model',
            adapter_id='provider-adapter',
        )

    with pytest.raises(SubmittedHarnessError, match='worker harness disk'):
        require_submitted_harness_binding(
            manifest=manifest,
            identity=identity,
            worker_harness_sha256=manifest.harness_image_sha256,
            worker_harness_byte_count=manifest.harness_image_byte_count + 1,
            logical_model_id='logical-model',
            adapter_id='provider-adapter',
        )


@pytest.mark.parametrize(
    'family',
    (HarnessFamily.CODEX, HarnessFamily.CLAUDE_CODE, HarnessFamily.CURSOR, HarnessFamily.CUSTOM),
)
def test_external_agent_families_have_a_contract_but_fail_closed_until_adapter_is_integrated(
    family: HarnessFamily,
) -> None:
    manifest = _manifest(family=family)
    identity = make_agentic_harness_identity(
        manifest=manifest,
        requested_model_id='logical-model',
        adapter_id='provider-adapter',
    )
    require_submitted_harness_binding(
        manifest=manifest,
        identity=identity,
        worker_harness_sha256=manifest.harness_image_sha256,
        worker_harness_byte_count=manifest.harness_image_byte_count,
        logical_model_id='logical-model',
        adapter_id='provider-adapter',
        require_runtime_integrated=False,
    )
    with pytest.raises(SubmittedHarnessError, match='not runtime-integrated'):
        require_submitted_harness_binding(
            manifest=manifest,
            identity=identity,
            worker_harness_sha256=manifest.harness_image_sha256,
            worker_harness_byte_count=manifest.harness_image_byte_count,
            logical_model_id='logical-model',
            adapter_id='provider-adapter',
        )


def test_vendor_label_cannot_falsely_claim_runtime_support() -> None:
    payload = _manifest(family=HarnessFamily.CODEX).model_dump(mode='python')
    payload['runtime_support'] = HarnessRuntimeSupport.RUNTIME_INTEGRATED
    with pytest.raises(ValidationError, match='runtime support'):
        SubmittedHarnessManifest.model_validate(payload)


def test_support_matrix_is_explicit_and_fail_closed() -> None:
    support = dict(harness_family_support_matrix())

    assert support[HarnessFamily.VAXREPLAY_NATIVE] == HarnessRuntimeSupport.RUNTIME_INTEGRATED
    assert support[HarnessFamily.CODEX] == HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED
    assert support[HarnessFamily.CLAUDE_CODE] == HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED
    assert support[HarnessFamily.CURSOR] == HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED
