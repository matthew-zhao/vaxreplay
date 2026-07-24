from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import vaxreplay.agentic.clinical_launcher as launcher_module
import vaxreplay.agentic.clinical_operator as operator_module
import vaxreplay.agentic.clinical_operator_cli as operator_cli_module
import vaxreplay.agentic.clinical_production_run_v02 as loader_module
import vaxreplay.agentic.firecracker_clinical_runtime as runtime_module
import vaxreplay.agentic.firecracker_qualification_collector as qualification_collector_module
import vaxreplay.agentic.provider_subprocess as provider_subprocess_module
from tests.test_agentic_clinical_production_run import (
    GATEWAY_KEY,
    GUEST_KEY,
    PRODUCTION_KEY,
    WORKER_KEY,
    WORKSPACE_KEY,
    Materials,
    _materials,
)
from tests.test_agentic_clinical_production_run_v02 import (
    BOOTSTRAP_AUTHORIZATION_KEY_ID,
    BOOTSTRAP_RECEIPT_KEY,
    _independent_trust_anchor,
)
from tests.test_agentic_firecracker_qualification_live_collector import (
    _COLLECTOR_KEY,
    _collect,
)
from tests.test_agentic_guest_boot_dispatch import (
    _development_dispatch,
    _headless_config,
    _submitted,
)
from vaxreplay.agentic.clinical_execution_bridge import clinical_workspace_receipt_key_id
from vaxreplay.agentic.clinical_guest_bootstrap import clinical_guest_bootstrap_receipt_key_id
from vaxreplay.agentic.clinical_guest_executable import (
    LaneAClinicalGuestConfig,
    lane_a_clinical_guest_config_sha256,
)
from vaxreplay.agentic.clinical_launcher import (
    CanonicalClinicalLauncherDeployment,
    clinical_launcher_failure_key_id,
)
from vaxreplay.agentic.clinical_operator import (
    CanonicalClinicalOperatorManifest,
    ClinicalOperatorError,
    ClinicalOperatorValidatedInputs,
    PinnedClinicalProductionRunV02Loader,
    execute_managed_operator_task,
    execute_operator_task,
    expected_system_identity,
    load_canonical_clinical_operator_manifest,
    load_operator_secret_directory,
    validate_operator_guest_disk_binding,
    validate_side_effect_free_runtime_parity,
    verify_operator_provider_data_control_attestation,
)
from vaxreplay.agentic.clinical_production_registry import (
    clinical_production_system_core_sha256,
    clinical_production_system_identity_sha256,
)
from vaxreplay.agentic.firecracker import firecracker_model_sha256
from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalRuntimeConfig,
    FirecrackerClinicalRuntimeKeys,
    firecracker_clinical_runtime_config_sha256,
)
from vaxreplay.agentic.firecracker_qualification import (
    firecracker_qualification_key_id,
    verify_and_retain_firecracker_live_qualification,
)
from vaxreplay.agentic.firecracker_qualification_probe import (
    FirecrackerQualificationBoundaryIdentity,
    FirecrackerQualificationBoundaryKind,
    FirecrackerQualificationCollectionMode,
    authenticate_firecracker_qualification_collection,
    firecracker_live_collector_key_id,
    firecracker_qualification_probe_manifest_sha256,
    firecracker_qualification_verifier_source_sha256,
)
from vaxreplay.agentic.guest_boot_dispatch import (
    guest_boot_dispatch_manifest_sha256,
    make_native_guest_boot_dispatch_manifest,
)
from vaxreplay.agentic.guest_disk_build import (
    GuestDiskExecutionBoundary,
    GuestDiskOutputIdentity,
    GuestDiskSourceIdentity,
    GuestDiskSourceKind,
    GuestDiskToolIdentity,
    GuestDiskToolLinkage,
    LaneAGuestDiskBuildReceipt,
    lane_a_guest_disk_build_receipt_sha256,
)
from vaxreplay.agentic.managed_clinical_ownership import (
    DurableManagedClinicalOwnershipLedger,
)
from vaxreplay.agentic.managed_clinical_registry import ManagedClinicalRegistryClient
from vaxreplay.agentic.managed_clinical_startup import ManagedClinicalStartupReconciler
from vaxreplay.agentic.provider_adapter import ProviderAdapterDescriptor
from vaxreplay.agentic.provider_subprocess import (
    ProviderSubprocessSpec,
    provider_subprocess_behavior_sha256,
    provider_subprocess_spec_sha256,
)
from vaxreplay.agentic.submitted_harness import (
    HarnessExecutionMode,
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessInterface,
    SubmittedHarnessManifest,
    make_agentic_harness_identity,
)
from vaxreplay.bundle import canonical_json_bytes

_FAILURE_KEY = b'clinical-operator-launcher-failure-key'
_QUALIFICATION_KEY = b'clinical-operator-qualification-key'


def _sha_module(module) -> str:
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> tuple[CanonicalClinicalOperatorManifest, Materials]:
    materials = _materials(tmp_path / 'materials')
    anchor = _independent_trust_anchor(materials)
    guest_config_sha256 = lane_a_clinical_guest_config_sha256(
        LaneAClinicalGuestConfig(
            trust_anchor=anchor,
            guest_rpc_port=materials.spec.guest_rpc_port,
        )
    )
    submitted_harness = SubmittedHarnessManifest(
        harness_id='clinical-fixture-agent',
        harness_version='1',
        family=HarnessFamily.VAXREPLAY_NATIVE,
        execution_mode=HarnessExecutionMode.FIXED_MODEL_LOOP,
        runtime_support=HarnessRuntimeSupport.RUNTIME_INTEGRATED,
        harness_image_sha256=materials.spec.images.harness.sha256,
        harness_image_byte_count=materials.spec.images.harness.byte_count,
        normalized_runtime_tree_sha256='7' * 64,
        guest_executable_path='/opt/vaxreplay/bin/vaxreplay-lane-a-clinical-guest',
        guest_executable_sha256='8' * 64,
        guest_argv=(
            '/opt/vaxreplay/bin/vaxreplay-lane-a-clinical-guest',
            '--expected-config-sha256',
            guest_config_sha256,
        ),
        baked_config_sha256=guest_config_sha256,
        dependency_closure_sha256='a' * 64,
        reproducible_build_receipt_sha256='b' * 64,
        interface=SubmittedHarnessInterface(
            guest_local_subprocesses_allowed=False,
            guest_local_shell_allowed=False,
        ),
        display_name='Clinical fixture agent',
        submitter='fixture-submitter',
    )
    guest_boot_dispatch = make_native_guest_boot_dispatch_manifest(
        guest_executable_sha256=submitted_harness.guest_executable_sha256,
        guest_config_sha256=guest_config_sha256,
    )
    materials = replace(
        materials,
        harness=make_agentic_harness_identity(
            manifest=submitted_harness,
            requested_model_id=materials.gateway.route.logical_model_id,
            adapter_id=materials.gateway.route.adapter_id,
        ),
    )
    runtime_config = FirecrackerClinicalRuntimeConfig(
        runtime_id='canonical-clinical-runtime',
        runtime_version='dev-v0.1',
        runtime_executable_sha256=_sha_module(runtime_module),
        bootstrap_authorization_key_id=BOOTSTRAP_AUTHORIZATION_KEY_ID,
        bootstrap_receipt_key_id=clinical_guest_bootstrap_receipt_key_id(BOOTSTRAP_RECEIPT_KEY),
        bootstrap_connection_timeout_seconds=3,
        bootstrap_validity_seconds=30,
        cleanup_grace_seconds=3,
    )
    deployment = CanonicalClinicalLauncherDeployment(
        registry_authority_id='clinical-operator-test-authority',
        canonical_launcher_id='clinical-operator-test-launcher',
        canonical_launcher_executable_sha256=_sha_module(launcher_module),
        expected_system_identity_sha256='0' * 64,
        runtime_id=runtime_config.runtime_id,
        runtime_version=runtime_config.runtime_version,
        runtime_executable_sha256=runtime_config.runtime_executable_sha256,
        runtime_config_sha256=firecracker_clinical_runtime_config_sha256(runtime_config),
        failure_receipt_key_id=clinical_launcher_failure_key_id(_FAILURE_KEY),
    )
    route = materials.gateway.route
    adapter = ProviderAdapterDescriptor(
        adapter_id=route.adapter_id,
        adapter_version=route.adapter_version,
        executable_sha256=route.adapter_executable_sha256,
        config_sha256=route.adapter_config_sha256,
        provider=route.provider,
    )
    values: dict[str, object] = {
        'operator_executable_sha256': _sha_module(operator_module),
        'strict_evidence_loader_executable_sha256': _sha_module(loader_module),
        'provider_subprocess_module_source_sha256': _sha_module(provider_subprocess_module),
        'deployment': deployment,
        'runtime_config': runtime_config,
        'worker_spec_path': str(tmp_path / 'worker-spec.json'),
        'expected_worker_spec_sha256': firecracker_model_sha256(materials.spec),
        'guest_disk_build_receipt_path': str(tmp_path / 'guest-disk-build.json'),
        'expected_guest_disk_build_receipt_sha256': 'b' * 64,
        'expected_guest_disk_builder_source_sha256': 'c' * 64,
        'expected_base_rootfs_source_sha256': 'd' * 64,
        'expected_harness_payload_source_sha256': 'e' * 64,
        'expected_mke2fs_sha256': 'f' * 64,
        'expected_e2fsck_sha256': '0' * 64,
        'expected_debugfs_sha256': '1' * 64,
        'expected_tool_runtime_closure_manifest_sha256': '2' * 64,
        'qualification_root': str(tmp_path / 'qualification'),
        'expected_qualification_artifact_sha256': '1' * 64,
        'expected_qualification_key_id': hashlib.sha256(
            b'vaxreplay.firecracker-qualification-key-id.v0.1\x00' + _QUALIFICATION_KEY
        ).hexdigest(),
        'expected_collector_evidence_sha256': '3' * 64,
        'expected_probe_manifest_sha256': '4' * 64,
        'expected_driver_runtime_closure_manifest_sha256': '8' * 64,
        'expected_driver_runtime_closure_receipt_sha256': '9' * 64,
        'expected_driver_runtime_closure_sha256': 'a' * 64,
        'expected_collector_public_key_hex': '5' * 64,
        'expected_collector_key_id': '6' * 64,
        'expected_qualification_verifier_source_sha256': '7' * 64,
        'registry_path': str(tmp_path / 'registry.sqlite3'),
        'gateway_ledger_path': str(tmp_path / 'gateway.sqlite3'),
        'evidence_root': str(tmp_path / 'evidence'),
        'reservation_sha256': '2' * 64,
        'episode_id': materials.workspace.task.context.episode_id,
        'workspace_root': str(materials.workspace.root),
        'expected_authenticated_workspace_receipt_sha256': materials.workspace.authenticated_receipt_sha256,
        'expected_workspace_receipt_key_id': clinical_workspace_receipt_key_id(WORKSPACE_KEY),
        'execution_policy': materials.policy,
        'gateway_policy': materials.gateway.policy,
        'gateway_route': route,
        'guest_rpc_policy': materials.guest.policy,
        'harness': materials.harness,
        'submitted_harness': submitted_harness,
        'guest_boot_dispatch': guest_boot_dispatch,
        'provider_adapter': adapter,
        'provider_subprocess': ProviderSubprocessSpec(
            executable_path='/usr/bin/python3',
            executable_sha256=route.adapter_executable_sha256,
            argv_suffix=('-m', 'vaxreplay.agentic.provider_subprocess'),
            maximum_call_seconds=10,
        ),
        'bootstrap_trust_anchor': anchor,
    }
    provisional = CanonicalClinicalOperatorManifest.model_validate(values)
    keys = FirecrackerClinicalRuntimeKeys(
        workspace_receipt_key=WORKSPACE_KEY,
        worker_attestation_key=WORKER_KEY,
        gateway_receipt_key=GATEWAY_KEY,
        guest_rpc_receipt_key=GUEST_KEY,
        clinical_guest_bootstrap_receipt_key=BOOTSTRAP_RECEIPT_KEY,
        production_receipt_key=PRODUCTION_KEY,
    )
    system_sha256 = clinical_production_system_identity_sha256(expected_system_identity(provisional, keys))
    deployment = deployment.model_copy(update={'expected_system_identity_sha256': system_sha256})
    return CanonicalClinicalOperatorManifest.model_validate({**values, 'deployment': deployment}), materials


def _secret_root(tmp_path: Path) -> Path:
    root = tmp_path / 'secrets'
    root.mkdir(mode=0o700)
    values = {
        'workspace-receipt.key': WORKSPACE_KEY,
        'worker-attestation.key': WORKER_KEY,
        'gateway-receipt.key': GATEWAY_KEY,
        'guest-rpc-receipt.key': GUEST_KEY,
        'bootstrap-receipt.key': BOOTSTRAP_RECEIPT_KEY,
        'production-receipt.key': PRODUCTION_KEY,
        'launcher-failure-receipt.key': _FAILURE_KEY,
        'qualification.key': _QUALIFICATION_KEY.hex().encode('ascii') + b'\n',
        'bootstrap-authorization.seed': b'\x19' * 32,
        'provider-credential': b'sk-test-provider-credential',
    }
    for name, content in values.items():
        path = root / name
        path.write_bytes(content)
        path.chmod(0o600)
    return root


def test_manifest_requires_canonical_bytes_external_digest_and_checked_in_loader(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    legacy = manifest.model_dump(mode='python')
    legacy['schema_version'] = 'vaxreplay.canonical-clinical-operator-manifest.dev-v0.7'
    with pytest.raises(ValueError):
        CanonicalClinicalOperatorManifest.model_validate(legacy)

    path = tmp_path / 'operator.json'
    content = canonical_json_bytes(manifest)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    loaded, observed = load_canonical_clinical_operator_manifest(path, expected_manifest_sha256=digest)

    assert loaded == manifest
    assert observed == digest
    with pytest.raises(ClinicalOperatorError, match='external SHA-256'):
        load_canonical_clinical_operator_manifest(path, expected_manifest_sha256='f' * 64)

    substituted = manifest.model_copy(update={'strict_evidence_loader_executable_sha256': 'e' * 64})
    path.write_bytes(canonical_json_bytes(substituted))
    with pytest.raises(ClinicalOperatorError, match='strict v0.2 evidence loader'):
        load_canonical_clinical_operator_manifest(
            path,
            expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    substituted = manifest.model_copy(update={'provider_subprocess_module_source_sha256': 'e' * 64})
    path.write_bytes(canonical_json_bytes(substituted))
    with pytest.raises(ClinicalOperatorError, match='provider child module source'):
        load_canonical_clinical_operator_manifest(
            path,
            expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_operator_accepts_positive_live_qualification_with_exact_runtime_closure_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, materials = _manifest(tmp_path)
    worker_spec_path = Path(manifest.worker_spec_path)
    worker_spec_path.write_bytes(canonical_json_bytes(materials.spec))

    spec, spec_sha256, probe_manifest, collector_public_key, development = _collect(
        tmp_path / 'collector-source',
        spec=materials.spec,
    )
    probe_manifest_sha256 = firecracker_qualification_probe_manifest_sha256(probe_manifest)
    collector_key_id = firecracker_live_collector_key_id(collector_public_key)
    verifier_source_sha256 = firecracker_qualification_verifier_source_sha256()
    boundary = FirecrackerQualificationBoundaryIdentity(
        boundary_id='pinned-linux-kvm-operator-regression',
        kind=FirecrackerQualificationBoundaryKind.PINNED_LINUX_KVM_DRIVER,
        executable_sha256='d' * 64,
        external_executable_pin_enforced=True,
        direct_linux_kvm_launch=True,
        injected_test_boundary=False,
        runtime_closure_manifest_sha256=(manifest.expected_driver_runtime_closure_manifest_sha256),
        runtime_closure_receipt_sha256=(manifest.expected_driver_runtime_closure_receipt_sha256),
        runtime_closure_sha256=manifest.expected_driver_runtime_closure_sha256,
        transitive_runtime_pin_enforced=True,
    )
    production_collection = development.authenticated.collection.model_copy(
        update={
            'mode': FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM,
            'boundary_identity': boundary,
            'driver_runtime_closure_sha256': (manifest.expected_driver_runtime_closure_sha256),
            'development_simulated': False,
            'production_qualification_eligible': True,
        }
    )
    authenticated = authenticate_firecracker_qualification_collection(
        production_collection,
        private_key=_COLLECTOR_KEY,
    )
    published = qualification_collector_module._publish_collector_evidence(
        output_root=tmp_path / 'production-collector',
        authenticated=authenticated,
        spec_bytes=canonical_json_bytes(spec),
        manifest_bytes=canonical_json_bytes(probe_manifest),
        expected_worker_spec_sha256=spec_sha256,
        expected_probe_manifest_sha256=probe_manifest_sha256,
        expected_collector_public_key_hex=collector_public_key.hex(),
        expected_collector_key_id=collector_key_id,
        expected_driver_runtime_closure_manifest_sha256=(manifest.expected_driver_runtime_closure_manifest_sha256),
        expected_driver_runtime_closure_receipt_sha256=(manifest.expected_driver_runtime_closure_receipt_sha256),
        expected_driver_runtime_closure_sha256=(manifest.expected_driver_runtime_closure_sha256),
    )
    qualification = verify_and_retain_firecracker_live_qualification(
        collector_evidence_root=Path(published.root),
        expected_collector_evidence_sha256=published.evidence_sha256,
        worker_spec_path=worker_spec_path,
        expected_worker_spec_sha256=spec_sha256,
        expected_probe_manifest_sha256=probe_manifest_sha256,
        expected_driver_runtime_closure_manifest_sha256=(manifest.expected_driver_runtime_closure_manifest_sha256),
        expected_driver_runtime_closure_receipt_sha256=(manifest.expected_driver_runtime_closure_receipt_sha256),
        expected_driver_runtime_closure_sha256=(manifest.expected_driver_runtime_closure_sha256),
        expected_host_preflight_sha256=production_collection.host_preflight_sha256,
        expected_collector_public_key_hex=collector_public_key.hex(),
        expected_collector_key_id=collector_key_id,
        expected_verifier_source_sha256=verifier_source_sha256,
        output_root=Path(manifest.qualification_root),
        qualification_key=_QUALIFICATION_KEY,
        expected_qualification_key_id=firecracker_qualification_key_id(_QUALIFICATION_KEY),
        qualification_id='a' * 32,
    )
    manifest = manifest.model_copy(
        update={
            'expected_qualification_artifact_sha256': qualification.artifact_sha256,
            'expected_collector_evidence_sha256': published.evidence_sha256,
            'expected_probe_manifest_sha256': probe_manifest_sha256,
            'expected_collector_public_key_hex': collector_public_key.hex(),
            'expected_collector_key_id': collector_key_id,
            'expected_qualification_verifier_source_sha256': verifier_source_sha256,
        }
    )
    guest_disks = SimpleNamespace(
        receipt_sha256=manifest.expected_guest_disk_build_receipt_sha256,
        receipt=SimpleNamespace(
            guest_boot_dispatch_manifest_sha256=guest_boot_dispatch_manifest_sha256(manifest.guest_boot_dispatch),
            guest_boot_dispatch=manifest.guest_boot_dispatch,
            rootfs=materials.spec.images.rootfs,
            harness=materials.spec.images.harness,
        ),
    )
    monkeypatch.setattr(
        operator_module,
        'load_and_verify_operator_guest_disks',
        lambda *_args, **_kwargs: guest_disks,
    )
    monkeypatch.setattr(
        operator_module.FirecrackerSupervisor,
        'preflight',
        lambda _self: production_collection.host_preflight,
    )

    inputs = operator_module.validate_operator_inputs(
        manifest,
        manifest_sha256='e' * 64,
        secret_root=_secret_root(tmp_path),
    )
    try:
        assert inputs.qualification == qualification
        assert inputs.qualification.authenticated.record.qualified is True
        assert inputs.qualification.authenticated.record.collector_verification is not None
        assert (
            inputs.qualification.authenticated.record.collector_verification.driver_runtime_closure_sha256
            == manifest.expected_driver_runtime_closure_sha256
        )
        report = operator_module.dry_run_report(inputs)
        assert report.qualification_driver_runtime_closure_manifest_sha256 == (
            manifest.expected_driver_runtime_closure_manifest_sha256
        )
        assert report.qualification_driver_runtime_closure_receipt_sha256 == (
            manifest.expected_driver_runtime_closure_receipt_sha256
        )
        assert report.qualification_driver_runtime_closure_sha256 == (manifest.expected_driver_runtime_closure_sha256)
    finally:
        inputs.secrets.close()


def test_secret_directory_is_exact_private_and_provider_credential_stays_an_fd(tmp_path: Path) -> None:
    root = _secret_root(tmp_path)
    secrets = load_operator_secret_directory(root)
    try:
        assert os.fstat(secrets.provider_credential_fd).st_size == len(b'sk-test-provider-credential')
        assert secrets.workspace_receipt_key == WORKSPACE_KEY
        assert secrets.qualification_key == _QUALIFICATION_KEY
    finally:
        secrets.close()

    extra = root / 'unexpected'
    extra.write_bytes(b'x' * 32)
    extra.chmod(0o600)
    with pytest.raises(ClinicalOperatorError, match='unexpected file inventory'):
        load_operator_secret_directory(root)


def test_operator_rejects_raw_or_malformed_qualification_key_file(tmp_path: Path) -> None:
    root = _secret_root(tmp_path)
    key_path = root / 'qualification.key'
    key_path.write_bytes(_QUALIFICATION_KEY)
    with pytest.raises(ClinicalOperatorError, match='trimmed ASCII-hex'):
        load_operator_secret_directory(root)

    key_path.write_bytes(_QUALIFICATION_KEY.hex().encode('ascii') + b' \n')
    with pytest.raises(ClinicalOperatorError, match='trimmed ASCII-hex'):
        load_operator_secret_directory(root)


def test_manifest_schema_cannot_contain_secret_material(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    payload = canonical_json_bytes(manifest)
    assert b'sk-test-provider-credential' not in payload
    assert b'clinical-operator-qualification-key' not in payload
    with pytest.raises(ValueError):
        CanonicalClinicalOperatorManifest.model_validate(
            {**manifest.model_dump(mode='json'), 'provider_credential': 'secret'}
        )


def test_pinned_loader_calls_only_strict_v02_with_captured_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, materials = _manifest(tmp_path)
    keys = FirecrackerClinicalRuntimeKeys(
        workspace_receipt_key=WORKSPACE_KEY,
        worker_attestation_key=WORKER_KEY,
        gateway_receipt_key=GATEWAY_KEY,
        guest_rpc_receipt_key=GUEST_KEY,
        clinical_guest_bootstrap_receipt_key=BOOTSTRAP_RECEIPT_KEY,
        production_receipt_key=PRODUCTION_KEY,
    )
    observed: dict[str, object] = {}
    sentinel = object()

    def strict_loader(root, **kwargs):
        observed['root'] = root
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(operator_module, 'load_clinical_production_run_v02', strict_loader)
    loader = PinnedClinicalProductionRunV02Loader(materials.workspace, manifest, materials.spec, keys)
    root = tmp_path / ('a' * 32)

    assert loader(root, 'b' * 64) is sentinel
    assert observed['expected_run_id'] == 'a' * 32
    assert observed['expected_attempt_reservation_sha256'] == 'b' * 64
    assert observed['clinical_guest_bootstrap_trust_anchor'] == manifest.bootstrap_trust_anchor
    with pytest.raises(ClinicalOperatorError, match='canonical run ID'):
        loader(tmp_path / 'caller-chosen-evidence', 'b' * 64)


def test_provider_and_retry_policy_cannot_be_ambient_or_enabled(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    values = dict(manifest)
    values['automatic_task_retry'] = True
    with pytest.raises(ValueError):
        CanonicalClinicalOperatorManifest.model_validate(values)
    values = dict(manifest)
    values['ambient_provider_route_allowed'] = True
    with pytest.raises(ValueError):
        CanonicalClinicalOperatorManifest.model_validate(values)


def test_nondefault_provider_data_control_requires_exact_loaded_evidence(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    evidence_path = tmp_path / 'provider-data-control-evidence.json'
    evidence_bytes = b'{"account_scope":"organizer-reviewed-test-fixture"}'
    evidence_path.write_bytes(evidence_bytes)
    evidence_path.chmod(0o400)
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    route = manifest.gateway_route.model_copy(
        update={
            'provider_data_control': 'zero_data_retention',
            'provider_data_control_attested': True,
            'provider_data_control_attestation_sha256': evidence_sha256,
        }
    )

    with pytest.raises(ValueError, match='operator-loaded attestation artifact'):
        CanonicalClinicalOperatorManifest.model_validate({**manifest.model_dump(mode='python'), 'gateway_route': route})

    bound = CanonicalClinicalOperatorManifest.model_validate(
        {
            **manifest.model_dump(mode='python'),
            'gateway_route': route,
            'provider_data_control_attestation_path': str(evidence_path),
        }
    )
    assert verify_operator_provider_data_control_attestation(bound) == evidence_sha256

    evidence_path.chmod(0o600)
    evidence_path.write_bytes(b'{"different":true}')
    evidence_path.chmod(0o400)
    with pytest.raises(ClinicalOperatorError, match='differ from the route commitment'):
        verify_operator_provider_data_control_attestation(bound)


def test_provider_data_control_evidence_must_be_trusted_owner_and_nonwritable(
    tmp_path: Path,
) -> None:
    manifest, _ = _manifest(tmp_path)
    evidence_path = tmp_path / 'provider-data-control-evidence.json'
    evidence_bytes = b'{"reviewed":true}'
    evidence_path.write_bytes(evidence_bytes)
    evidence_path.chmod(0o622)
    route = manifest.gateway_route.model_copy(
        update={
            'provider_data_control': 'modified_abuse_monitoring',
            'provider_data_control_attested': True,
            'provider_data_control_attestation_sha256': hashlib.sha256(evidence_bytes).hexdigest(),
        }
    )
    bound = CanonicalClinicalOperatorManifest.model_validate(
        {
            **manifest.model_dump(mode='python'),
            'gateway_route': route,
            'provider_data_control_attestation_path': str(evidence_path),
        }
    )

    with pytest.raises(ClinicalOperatorError, match='unsafe metadata'):
        verify_operator_provider_data_control_attestation(bound)


def test_dry_configuration_binds_the_full_guest_trust_anchor(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    values = dict(manifest)
    values['bootstrap_trust_anchor'] = manifest.bootstrap_trust_anchor.model_copy(
        update={'execution_policy_sha256': 'f' * 64}
    )
    with pytest.raises(ValueError, match='full static guest policy pins'):
        CanonicalClinicalOperatorManifest.model_validate(values)

    values = dict(manifest)
    limits = manifest.bootstrap_trust_anchor.rpc_limits.model_copy(
        update={'maximum_requests': manifest.bootstrap_trust_anchor.rpc_limits.maximum_requests + 1}
    )
    values['bootstrap_trust_anchor'] = manifest.bootstrap_trust_anchor.model_copy(update={'rpc_limits': limits})
    with pytest.raises(ValueError, match='full static guest policy pins'):
        CanonicalClinicalOperatorManifest.model_validate(values)


def test_manifest_is_honest_about_nontransitive_python_source_pins(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    assert manifest.provider_subprocess_module_source_sha256 == _sha_module(provider_subprocess_module)
    assert manifest.entry_module_source_pins_verified_only is True
    assert manifest.transitive_dependency_closure_attested is False
    assert manifest.executing_process_image_attested is False


def test_reserved_system_identity_binds_provider_child_spec_and_module_source(
    tmp_path: Path,
) -> None:
    manifest, _ = _manifest(tmp_path)
    keys = FirecrackerClinicalRuntimeKeys(
        workspace_receipt_key=WORKSPACE_KEY,
        worker_attestation_key=WORKER_KEY,
        gateway_receipt_key=GATEWAY_KEY,
        guest_rpc_receipt_key=GUEST_KEY,
        clinical_guest_bootstrap_receipt_key=BOOTSTRAP_RECEIPT_KEY,
        production_receipt_key=PRODUCTION_KEY,
    )
    identity = expected_system_identity(manifest, keys)

    assert identity.provider_subprocess_spec_sha256 == provider_subprocess_spec_sha256(manifest.provider_subprocess)
    assert identity.provider_subprocess_behavior_sha256 == provider_subprocess_behavior_sha256(
        manifest.provider_subprocess
    )
    assert identity.provider_subprocess_module_source_sha256 == (manifest.provider_subprocess_module_source_sha256)

    changed_specs = (
        manifest.provider_subprocess.model_copy(update={'argv_suffix': ('-c', 'raise SystemExit(0)')}),
        manifest.provider_subprocess.model_copy(
            update={'maximum_call_seconds': manifest.provider_subprocess.maximum_call_seconds + 1}
        ),
        manifest.provider_subprocess.model_copy(
            update={'maximum_response_bytes': manifest.provider_subprocess.maximum_response_bytes + 1}
        ),
    )
    changed_manifests = tuple(manifest.model_copy(update={'provider_subprocess': spec}) for spec in changed_specs) + (
        manifest.model_copy(update={'provider_subprocess_module_source_sha256': 'f' * 64}),
    )
    for changed_manifest in changed_manifests:
        changed_identity = expected_system_identity(changed_manifest, keys)
        assert clinical_production_system_identity_sha256(changed_identity) != (
            clinical_production_system_identity_sha256(identity)
        )
        assert clinical_production_system_core_sha256(changed_identity) != (
            clinical_production_system_core_sha256(identity)
        )

    path_only_spec = manifest.provider_subprocess.model_copy(
        update={'executable_path': '/opt/vaxreplay/bin/provider-child-renamed'}
    )
    path_only_identity = expected_system_identity(
        manifest.model_copy(update={'provider_subprocess': path_only_spec}),
        keys,
    )
    assert clinical_production_system_identity_sha256(path_only_identity) != (
        clinical_production_system_identity_sha256(identity)
    )
    assert clinical_production_system_core_sha256(path_only_identity) == (
        clinical_production_system_core_sha256(identity)
    )


def test_dry_configuration_mirrors_runtime_resource_and_harness_checks(tmp_path: Path) -> None:
    manifest, materials = _manifest(tmp_path)
    validate_side_effect_free_runtime_parity(manifest, materials.spec)

    wrong_anchor = manifest.bootstrap_trust_anchor.model_copy(update={'worker_bootstrap_profile_sha256': 'f' * 64})
    with pytest.raises(ClinicalOperatorError, match='worker bootstrap profile'):
        validate_side_effect_free_runtime_parity(
            manifest.model_copy(update={'bootstrap_trust_anchor': wrong_anchor}),
            materials.spec,
        )

    wrong_limits = materials.spec.limits.model_copy(update={'memory_mib': materials.spec.limits.memory_mib * 2})
    wrong_spec = materials.spec.model_copy(update={'limits': wrong_limits})
    with pytest.raises(ClinicalOperatorError, match='resources'):
        validate_side_effect_free_runtime_parity(manifest, wrong_spec)

    wrong_harness = manifest.harness.model_copy(update={'harness_image_or_commitment': f'sha256:{"f" * 64}'})
    wrong_manifest = manifest.model_copy(update={'harness': wrong_harness})
    with pytest.raises(ClinicalOperatorError, match='harness image'):
        validate_side_effect_free_runtime_parity(wrong_manifest, materials.spec)

    wrong_harness_artifact = materials.spec.images.harness.model_copy(
        update={'byte_count': materials.spec.images.harness.byte_count + 1}
    )
    wrong_images = materials.spec.images.model_copy(update={'harness': wrong_harness_artifact})
    wrong_spec = materials.spec.model_copy(update={'images': wrong_images})
    with pytest.raises(ClinicalOperatorError, match='harness image'):
        validate_side_effect_free_runtime_parity(manifest, wrong_spec)


def test_canonical_operator_rejects_packageable_development_adapter(tmp_path: Path) -> None:
    manifest, materials = _manifest(tmp_path)
    config = _headless_config(HarnessFamily.CODEX)
    dispatch = _development_dispatch(config)
    submitted = _submitted(config, dispatch).model_copy(
        update={
            'harness_image_sha256': materials.spec.images.harness.sha256,
            'harness_image_byte_count': materials.spec.images.harness.byte_count,
            'reproducible_build_receipt_sha256': (manifest.expected_guest_disk_build_receipt_sha256),
        }
    )
    harness = make_agentic_harness_identity(
        manifest=submitted,
        requested_model_id=manifest.gateway_route.logical_model_id,
        adapter_id=manifest.gateway_route.adapter_id,
    )

    with pytest.raises(ValueError, match='submitted harness binding is not executable'):
        CanonicalClinicalOperatorManifest.model_validate(
            {
                **manifest.model_dump(mode='python'),
                'submitted_harness': submitted,
                'guest_boot_dispatch': dispatch,
                'harness': harness,
            }
        )


def _receipt_bound_manifest(
    manifest: CanonicalClinicalOperatorManifest,
    materials: Materials,
) -> tuple[CanonicalClinicalOperatorManifest, LaneAGuestDiskBuildReceipt]:
    base_source = GuestDiskSourceIdentity(
        kind=GuestDiskSourceKind.NORMALIZED_TAR,
        sha256=manifest.expected_base_rootfs_source_sha256,
        byte_count=4096,
        normalized_tree_sha256='2' * 64,
    )
    harness_source = base_source.model_copy(update={'sha256': manifest.expected_harness_payload_source_sha256})
    receipt = LaneAGuestDiskBuildReceipt(
        execution_boundary=GuestDiskExecutionBoundary.PINNED_LINUX_PROCFS,
        production_eligible=True,
        source_date_epoch=1_700_000_000,
        base_rootfs_source=base_source,
        harness_payload_source=harness_source,
        mke2fs=GuestDiskToolIdentity(
            name='mke2fs',
            sha256=manifest.expected_mke2fs_sha256,
            version='fixture-e2fsprogs',
            executed_via_proc_self_fd=True,
            linkage=GuestDiskToolLinkage.STATIC_ELF,
        ),
        e2fsck=GuestDiskToolIdentity(
            name='e2fsck',
            sha256=manifest.expected_e2fsck_sha256,
            version='fixture-e2fsprogs',
            executed_via_proc_self_fd=True,
            linkage=GuestDiskToolLinkage.STATIC_ELF,
        ),
        debugfs=GuestDiskToolIdentity(
            name='debugfs',
            sha256=manifest.expected_debugfs_sha256,
            version='fixture-e2fsprogs',
            executed_via_proc_self_fd=True,
            linkage=GuestDiskToolLinkage.STATIC_ELF,
        ),
        tool_runtime_closure_manifest_sha256='2' * 64,
        tool_runtime_closure_manifest_byte_count=4096,
        tool_runtime_closure_external_pin_checked=True,
        tool_runtime_closure_bindings_checked=True,
        tool_runtime_closure_contains_dynamic_linkage=False,
        builder_source_sha256=manifest.expected_guest_disk_builder_source_sha256,
        build_contract_sha256='3' * 64,
        inspection_contract_sha256='4' * 64,
        mke2fs_argv_sha256='5' * 64,
        build_environment_sha256='6' * 64,
        inspection_argv_sha256='7' * 64,
        init_sha256='8' * 64,
        guest_boot_dispatch=manifest.guest_boot_dispatch,
        guest_boot_dispatch_manifest_sha256=guest_boot_dispatch_manifest_sha256(manifest.guest_boot_dispatch),
        guest_executable_path=manifest.submitted_harness.guest_executable_path,
        guest_executable_sha256=manifest.submitted_harness.guest_executable_sha256,
        guest_config_path=manifest.guest_boot_dispatch.guest_config_path,
        guest_config_sha256=manifest.submitted_harness.baked_config_sha256,
        dependency_closure_sha256=manifest.submitted_harness.dependency_closure_sha256,
        fixed_guest_argv=manifest.submitted_harness.guest_argv,
        canonical_operator_runtime_supported=True,
        rootfs=GuestDiskOutputIdentity(
            role='rootfs',
            sha256=materials.spec.images.rootfs.sha256,
            byte_count=materials.spec.images.rootfs.byte_count,
            uuid='00000000-0000-4000-8000-000000000201',
            label='vaxlanea-root',
            normalized_tree_sha256='9' * 64,
        ),
        harness=GuestDiskOutputIdentity(
            role='harness',
            sha256=materials.spec.images.harness.sha256,
            byte_count=materials.spec.images.harness.byte_count,
            uuid='00000000-0000-4000-8000-000000000202',
            label='vaxlanea-harness',
            normalized_tree_sha256=manifest.submitted_harness.normalized_runtime_tree_sha256,
        ),
    )
    receipt_sha256 = lane_a_guest_disk_build_receipt_sha256(receipt)
    submitted = manifest.submitted_harness.model_copy(update={'reproducible_build_receipt_sha256': receipt_sha256})
    bound = manifest.model_copy(
        update={
            'expected_guest_disk_build_receipt_sha256': receipt_sha256,
            'submitted_harness': submitted,
            'harness': make_agentic_harness_identity(
                manifest=submitted,
                requested_model_id=manifest.gateway_route.logical_model_id,
                adapter_id=manifest.gateway_route.adapter_id,
            ),
        }
    )
    return bound, receipt


def test_verified_guest_disk_receipt_cross_binds_worker_config_and_harness(tmp_path: Path) -> None:
    manifest, materials = _manifest(tmp_path)
    bound, receipt = _receipt_bound_manifest(manifest, materials)

    validate_operator_guest_disk_binding(bound, materials.spec, receipt)

    wrong = receipt.model_copy(
        update={'harness': receipt.harness.model_copy(update={'normalized_tree_sha256': 'f' * 64})}
    )
    with pytest.raises(ClinicalOperatorError, match='verified task disks differ'):
        validate_operator_guest_disk_binding(bound, materials.spec, wrong)


def test_regular_provider_credential_fd_is_repeatable_without_parent_read(tmp_path: Path) -> None:
    root = _secret_root(tmp_path)
    secrets = load_operator_secret_directory(root)
    try:
        # Deliberately move the shared file offset.  The Linux child reader uses pread with an
        # explicit offset for regular files, so sequential model calls do not consume the token.
        os.lseek(secrets.provider_credential_fd, 5, os.SEEK_SET)
        first = provider_subprocess_module._read_credential_descriptor(secrets.provider_credential_fd)
        os.lseek(secrets.provider_credential_fd, 11, os.SEEK_SET)
        second = provider_subprocess_module._read_credential_descriptor(secrets.provider_credential_fd)
        assert bytes(first) == bytes(second) == b'sk-test-provider-credential'
    finally:
        secrets.close()


def test_managed_manifest_cannot_fall_back_to_local_sqlite(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    managed = CanonicalClinicalOperatorManifest.model_validate(
        {
            **manifest.model_dump(mode='python'),
            'registry_execution_mode': 'managed-unix-authority',
            'managed_registry_config_sha256': 'c' * 64,
            'managed_startup_config_sha256': 'd' * 64,
            'managed_ownership_config_sha256': 'e' * 64,
        }
    )
    inputs = cast(
        ClinicalOperatorValidatedInputs,
        SimpleNamespace(manifest=managed),
    )

    with pytest.raises(ClinicalOperatorError, match='cannot fall back'):
        execute_operator_task(inputs)


def test_managed_manifest_requires_an_ownership_config_pin(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)

    with pytest.raises(ValueError, match='requires all deployment config pins'):
        CanonicalClinicalOperatorManifest.model_validate(
            {
                **manifest.model_dump(mode='python'),
                'registry_execution_mode': 'managed-unix-authority',
                'managed_registry_config_sha256': 'c' * 64,
                'managed_startup_config_sha256': 'd' * 64,
            }
        )


def test_managed_execution_rejects_an_injected_registry_protocol(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    managed = CanonicalClinicalOperatorManifest.model_validate(
        {
            **manifest.model_dump(mode='python'),
            'registry_execution_mode': 'managed-unix-authority',
            'managed_registry_config_sha256': 'c' * 64,
            'managed_startup_config_sha256': 'd' * 64,
            'managed_ownership_config_sha256': 'e' * 64,
        }
    )
    inputs = cast(
        ClinicalOperatorValidatedInputs,
        SimpleNamespace(manifest=managed),
    )

    with pytest.raises(ClinicalOperatorError, match='pinned service configuration'):
        execute_managed_operator_task(
            inputs,
            managed_registry=cast(
                ManagedClinicalRegistryClient,
                SimpleNamespace(config_sha256='c' * 64),
            ),
            startup_reconciler=cast(
                ManagedClinicalStartupReconciler,
                SimpleNamespace(),
            ),
            managed_ownership=cast(
                DurableManagedClinicalOwnershipLedger,
                SimpleNamespace(),
            ),
        )


def test_cli_dry_run_stops_before_launcher_vm_provider_and_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, _ = _manifest(tmp_path)
    closed: list[bool] = []

    class Secrets:
        def close(self) -> None:
            closed.append(True)

    class Inputs:
        secrets = Secrets()

    monkeypatch.setattr(
        operator_cli_module,
        'load_canonical_clinical_operator_manifest',
        lambda *_args, **_kwargs: (manifest, 'a' * 64),
    )
    monkeypatch.setattr(
        operator_cli_module,
        'validate_operator_inputs',
        lambda *_args, **_kwargs: Inputs(),
    )
    monkeypatch.setattr(
        operator_cli_module,
        'dry_run_report',
        lambda _inputs: {
            'provider_call_made': False,
            'worker_launched': False,
            'registry_mutated': False,
            'official_execution_qualified': False,
        },
    )
    monkeypatch.setattr(
        operator_cli_module,
        'execute_operator_task',
        lambda _inputs: pytest.fail('dry run must never construct or execute the launcher'),
    )
    monkeypatch.setattr(
        'sys.argv',
        [
            'vaxreplay-clinical-operator',
            'run-task',
            '--manifest',
            str(tmp_path / 'manifest.json'),
            '--expected-manifest-sha256',
            'a' * 64,
            '--secret-root',
            str(tmp_path / 'secrets'),
            '--allow-development-local-sqlite',
            '--dry-run',
        ],
    )

    operator_cli_module.main()

    assert '"worker_launched":false' in capsys.readouterr().out
    assert closed == [True]
