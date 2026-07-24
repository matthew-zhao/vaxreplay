"""Fail-closed operator composition for one development Lane A task.

This is the checked-in composition root.  It is intentionally narrower than the reusable
launcher/runtime classes: a canonical, externally hash-pinned manifest selects the five Python
entry modules and every modeled policy, while secret bytes are accepted only from a private
directory of fixed-name files.  These source-file pins are not an attestation of the transitive
Python dependency closure or executing process image.  No provider route, evidence loader, or
retry policy can be supplied by the run caller.

The composition remains development-only.  A positive Firecracker qualification is required to
run, but neither that qualification nor a successful task admits evidence to a leaderboard or
claims provider/model, image-bake, identity-contamination, or model-weight qualification.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Literal, Self

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import Field, field_validator, model_validator

import vaxreplay.agentic.clinical_launcher as clinical_launcher_module
import vaxreplay.agentic.clinical_operator as clinical_operator_module
import vaxreplay.agentic.clinical_production_run_v02 as clinical_production_run_v02_module
import vaxreplay.agentic.firecracker_clinical_runtime as firecracker_clinical_runtime_module
import vaxreplay.agentic.provider_subprocess as provider_subprocess_module
from vaxreplay.agentic.clinical_execution_bridge import (
    LoadedClinicalAgenticWorkspace,
    clinical_workspace_receipt_key_id,
    load_clinical_agentic_workspace,
)
from vaxreplay.agentic.clinical_guest_bootstrap import (
    CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
    ClinicalGuestBootstrapTrustAnchor,
    ClinicalGuestRpcLimits,
    clinical_guest_bootstrap_authorization_key_id,
    clinical_guest_bootstrap_receipt_key_id,
)
from vaxreplay.agentic.clinical_guest_executable import (
    LaneAClinicalGuestConfig,
    lane_a_clinical_guest_config_sha256,
)
from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_ACTION_SCHEMA_SHA256,
    LANE_A_GUEST_HARNESS_POLICY_ID,
)
from vaxreplay.agentic.clinical_launcher import (
    CanonicalClinicalLauncher,
    CanonicalClinicalLauncherDeployment,
    CanonicalClinicalRecoveryTerminalizer,
    ClinicalLauncherResult,
    ClinicalProductionRegistryBoundary,
    clinical_launcher_failure_key_id,
)
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionSystemIdentity,
    SqliteClinicalProductionRegistry,
    clinical_production_system_identity_sha256,
)
from vaxreplay.agentic.clinical_production_run import clinical_production_run_key_id
from vaxreplay.agentic.clinical_production_run_v02 import (
    LoadedClinicalProductionRunV02,
    load_clinical_production_run_v02,
)
from vaxreplay.agentic.firecracker import (
    FirecrackerHostPreflightReceipt,
    FirecrackerSupervisor,
    FirecrackerWorkerSpec,
    firecracker_attestation_key_id,
    firecracker_guest_bootstrap_profile_sha256,
)
from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalRuntime,
    FirecrackerClinicalRuntimeConfig,
    FirecrackerClinicalRuntimeKeys,
    firecracker_clinical_runtime_config_sha256,
)
from vaxreplay.agentic.firecracker_qualification import (
    LoadedFirecrackerQualification,
    decode_firecracker_qualification_key,
    firecracker_qualification_key_id,
    load_firecracker_qualification,
    load_pinned_firecracker_worker_spec,
)
from vaxreplay.agentic.gateway_auth import InMemoryGatewaySecretStore
from vaxreplay.agentic.guest_boot_dispatch import (
    GuestBootDispatchAdmission,
    GuestBootDispatchError,
    GuestBootDispatchManifest,
    guest_boot_dispatch_manifest_sha256,
    require_guest_boot_dispatch_binding,
)
from vaxreplay.agentic.guest_disk_build import (
    LaneAGuestDiskBuildError,
    LaneAGuestDiskBuildReceipt,
    VerifiedLaneAGuestDisks,
    lane_a_guest_disk_build_receipt_sha256,
    load_lane_a_guest_disk_build_receipt,
    verify_lane_a_guest_disk_build,
)
from vaxreplay.agentic.guest_rpc import GuestRpcPolicy, guest_rpc_policy_sha256, guest_rpc_session_key_id
from vaxreplay.agentic.managed_clinical_ownership import (
    DurableManagedClinicalOwnershipLedger,
    LinuxManagedClinicalHostAdapter,
    managed_clinical_ownership_config_sha256,
)
from vaxreplay.agentic.managed_clinical_registry import ManagedClinicalRegistryClient
from vaxreplay.agentic.managed_clinical_startup import (
    ManagedClinicalStartupReconciler,
    managed_clinical_startup_config_sha256,
    reconcile_canonical_managed_runtime_startup,
)
from vaxreplay.agentic.managed_gateway_capability import (
    RestartVisibleManagedGatewayCapabilityLedger,
)
from vaxreplay.agentic.protocol import AgenticExecutionPolicy, agentic_policy_sha256
from vaxreplay.agentic.provider_adapter import ProviderAdapterDescriptor
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewayPolicy,
    AuthenticatedProviderGateway,
    GatewayModelRoute,
    SqliteGatewayLedger,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    gateway_session_key_id,
)
from vaxreplay.agentic.provider_subprocess import (
    ProviderSubprocessSpec,
    SubprocessProviderAdapter,
    provider_subprocess_behavior_sha256,
    provider_subprocess_spec_sha256,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.run_artifact import AgenticHarnessIdentity
from vaxreplay.agentic.submitted_harness import (
    SubmittedHarnessError,
    SubmittedHarnessManifest,
    require_submitted_harness_binding,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.signing import LocalEd25519Signer
from vaxreplay.runner.schema import IsolationTier

CANONICAL_CLINICAL_OPERATOR_MANIFEST_SCHEMA_VERSION = 'vaxreplay.canonical-clinical-operator-manifest.dev-v0.8'
CANONICAL_CLINICAL_OPERATOR_ID = 'vaxreplay-lane-a-canonical-operator'
CANONICAL_CLINICAL_OPERATOR_VERSION = 'dev-v0.1'
STRICT_CLINICAL_V02_LOADER_ID = 'vaxreplay-strict-clinical-production-run-loader'
STRICT_CLINICAL_V02_LOADER_VERSION = 'dev-v0.2'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_SECRET_BYTES = 512
_SECRET_FILE_NAMES = frozenset(
    {
        'bootstrap-authorization.seed',
        'bootstrap-receipt.key',
        'gateway-receipt.key',
        'guest-rpc-receipt.key',
        'launcher-failure-receipt.key',
        'production-receipt.key',
        'provider-credential',
        'qualification.key',
        'worker-attestation.key',
        'workspace-receipt.key',
    }
)


class ClinicalOperatorError(RuntimeError):
    """A bounded operator configuration or host rejection."""


class CanonicalClinicalOperatorManifest(StrictModel):
    """Exact non-secret composition and one task selection, externally SHA-256 pinned."""

    schema_version: Literal['vaxreplay.canonical-clinical-operator-manifest.dev-v0.8'] = (
        CANONICAL_CLINICAL_OPERATOR_MANIFEST_SCHEMA_VERSION
    )
    operator_id: Literal['vaxreplay-lane-a-canonical-operator'] = CANONICAL_CLINICAL_OPERATOR_ID
    operator_version: Literal['dev-v0.1'] = CANONICAL_CLINICAL_OPERATOR_VERSION
    operator_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    strict_evidence_loader_id: Literal['vaxreplay-strict-clinical-production-run-loader'] = (
        STRICT_CLINICAL_V02_LOADER_ID
    )
    strict_evidence_loader_version: Literal['dev-v0.2'] = STRICT_CLINICAL_V02_LOADER_VERSION
    strict_evidence_loader_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider_subprocess_module_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    deployment: CanonicalClinicalLauncherDeployment
    runtime_config: FirecrackerClinicalRuntimeConfig
    worker_spec_path: str
    expected_worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_disk_build_receipt_path: str
    expected_guest_disk_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_guest_disk_builder_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_base_rootfs_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_harness_payload_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_mke2fs_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_e2fsck_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_debugfs_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_tool_runtime_closure_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_root: str
    expected_qualification_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_qualification_key_id: str = Field(pattern=_SHA256_PATTERN)
    expected_collector_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_probe_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_driver_runtime_closure_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_driver_runtime_closure_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_driver_runtime_closure_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_collector_public_key_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    expected_collector_key_id: str = Field(pattern=_SHA256_PATTERN)
    expected_qualification_verifier_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    registry_path: str
    registry_execution_mode: Literal[
        'development-local-sqlite',
        'managed-unix-authority',
    ] = 'development-local-sqlite'
    managed_registry_config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    managed_startup_config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    managed_ownership_config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    gateway_ledger_path: str
    evidence_root: str
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1, max_length=500)
    workspace_root: str
    expected_authenticated_workspace_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_workspace_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    execution_policy: AgenticExecutionPolicy
    gateway_policy: AuthenticatedGatewayPolicy
    gateway_route: GatewayModelRoute
    provider_data_control_attestation_path: str | None = None
    guest_rpc_policy: GuestRpcPolicy
    harness: AgenticHarnessIdentity
    submitted_harness: SubmittedHarnessManifest
    guest_boot_dispatch: GuestBootDispatchManifest
    provider_adapter: ProviderAdapterDescriptor
    provider_subprocess: ProviderSubprocessSpec
    bootstrap_trust_anchor: ClinicalGuestBootstrapTrustAnchor
    automatic_task_retry: Literal[False] = False
    automatic_provider_retry: Literal[False] = False
    ambient_provider_route_allowed: Literal[False] = False
    strict_v02_evidence_loader_only: Literal[True] = True
    single_authoritative_registry: Literal[True] = True
    secrets_in_manifest: Literal[False] = False
    entry_module_source_pins_verified_only: Literal[True] = True
    transitive_dependency_closure_attested: Literal[False] = False
    executing_process_image_attested: Literal[False] = False
    development_only: Literal[True] = True
    official_execution_qualified: Literal[False] = False
    leaderboard_admitted: Literal[False] = False

    @field_validator(
        'worker_spec_path',
        'guest_disk_build_receipt_path',
        'qualification_root',
        'registry_path',
        'gateway_ledger_path',
        'evidence_root',
        'workspace_root',
    )
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('operator paths must be absolute and normalized')
        return value

    @field_validator('provider_data_control_attestation_path')
    @classmethod
    def validate_optional_attestation_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('provider data-control attestation path must be absolute and normalized')
        return value

    @model_validator(mode='after')
    def validate_static_composition(self) -> Self:
        externally_controlled = self.gateway_route.provider_data_control != 'default'
        if externally_controlled != (self.provider_data_control_attestation_path is not None):
            raise ValueError('non-default provider data control requires one operator-loaded attestation artifact')
        if self.expected_guest_disk_build_receipt_sha256 != (self.submitted_harness.reproducible_build_receipt_sha256):
            raise ValueError('operator and submitted harness name different guest-disk build receipts')
        managed_pins = (
            self.managed_registry_config_sha256,
            self.managed_startup_config_sha256,
            self.managed_ownership_config_sha256,
        )
        if self.registry_execution_mode == 'development-local-sqlite':
            if any(value is not None for value in managed_pins):
                raise ValueError('local SQLite mode cannot carry managed-authority pins')
        elif any(value is None for value in managed_pins):
            raise ValueError('managed authority mode requires all deployment config pins')
        if self.execution_policy.required_isolation != IsolationTier.DEVELOPMENT or (
            self.execution_policy.response_protocol != AgenticResponseProtocol.CLINICAL_EXECUTION
        ):
            raise ValueError('canonical clinical operator requires the development clinical-execution policy')
        if self.deployment.runtime_config_sha256 != firecracker_clinical_runtime_config_sha256(self.runtime_config):
            raise ValueError('deployment does not bind the exact runtime config')
        if (
            self.deployment.runtime_id,
            self.deployment.runtime_version,
            self.deployment.runtime_executable_sha256,
        ) != (
            self.runtime_config.runtime_id,
            self.runtime_config.runtime_version,
            self.runtime_config.runtime_executable_sha256,
        ):
            raise ValueError('deployment and runtime config identities differ')
        if self.bootstrap_trust_anchor.authorization_key_id != self.runtime_config.bootstrap_authorization_key_id:
            raise ValueError('bootstrap trust anchor differs from the runtime authorization-key pin')
        expected_rpc_limits = ClinicalGuestRpcLimits(
            maximum_frame_body_bytes=self.guest_rpc_policy.maximum_frame_body_bytes,
            maximum_session_wire_bytes=self.guest_rpc_policy.maximum_session_wire_bytes,
            maximum_requests=self.guest_rpc_policy.maximum_requests,
            maximum_list_entries=self.guest_rpc_policy.maximum_list_entries,
            maximum_read_bytes=self.guest_rpc_policy.maximum_read_bytes,
            maximum_search_results=self.guest_rpc_policy.maximum_search_results,
            maximum_submission_bytes=self.guest_rpc_policy.maximum_submission_bytes,
        )
        if (
            self.bootstrap_trust_anchor.execution_policy_sha256,
            self.bootstrap_trust_anchor.harness_policy_id,
            self.bootstrap_trust_anchor.harness_policy_sha256,
            self.bootstrap_trust_anchor.action_schema_sha256,
            self.bootstrap_trust_anchor.rpc_limits,
        ) != (
            agentic_policy_sha256(self.execution_policy),
            LANE_A_GUEST_HARNESS_POLICY_ID,
            CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
            LANE_A_GUEST_ACTION_SCHEMA_SHA256,
            expected_rpc_limits,
        ):
            raise ValueError('bootstrap trust anchor differs from the full static guest policy pins')
        route_identity = (
            self.gateway_route.provider,
            self.gateway_route.adapter_id,
            self.gateway_route.adapter_version,
            self.gateway_route.adapter_executable_sha256,
            self.gateway_route.adapter_config_sha256,
        )
        adapter_identity = (
            self.provider_adapter.provider,
            self.provider_adapter.adapter_id,
            self.provider_adapter.adapter_version,
            self.provider_adapter.executable_sha256,
            self.provider_adapter.config_sha256,
        )
        if route_identity != adapter_identity:
            raise ValueError('provider adapter does not implement the one pinned gateway route')
        if self.provider_subprocess.executable_sha256 != self.provider_adapter.executable_sha256:
            raise ValueError('provider child executable differs from its adapter identity')
        if (
            self.gateway_route.logical_model_id,
            self.gateway_route.adapter_id,
        ) != (self.harness.requested_model_id, self.harness.adapter_id):
            raise ValueError('harness identity differs from the one pinned provider route')
        try:
            require_submitted_harness_binding(
                manifest=self.submitted_harness,
                identity=self.harness,
                worker_harness_sha256=self.submitted_harness.harness_image_sha256,
                worker_harness_byte_count=self.submitted_harness.harness_image_byte_count,
                logical_model_id=self.gateway_route.logical_model_id,
                adapter_id=self.gateway_route.adapter_id,
            )
            require_guest_boot_dispatch_binding(
                dispatch=self.guest_boot_dispatch,
                submitted_harness=self.submitted_harness,
            )
        except SubmittedHarnessError as error:
            raise ValueError('submitted harness binding is not executable by this operator') from error
        except GuestBootDispatchError as error:
            raise ValueError('guest boot dispatch differs from the submitted harness') from error
        if self.guest_boot_dispatch.admission != (
            GuestBootDispatchAdmission.RUNTIME_INTEGRATED_REQUIRES_EXTERNAL_QUALIFICATION
        ):
            raise ValueError('canonical operator cannot admit a development-only guest adapter dispatch')
        return self


@dataclass(frozen=True, slots=True)
class ClinicalOperatorSecretMaterial:
    workspace_receipt_key: bytes
    worker_attestation_key: bytes
    gateway_receipt_key: bytes
    guest_rpc_receipt_key: bytes
    bootstrap_receipt_key: bytes
    production_receipt_key: bytes
    launcher_failure_receipt_key: bytes
    qualification_key: bytes
    bootstrap_authorization_seed: bytes
    provider_credential_fd: int

    def close(self) -> None:
        try:
            os.close(self.provider_credential_fd)
        except OSError:
            pass


@dataclass(frozen=True, slots=True)
class ClinicalOperatorRecoverySecretMaterial:
    """Minimum secret surface needed by cleanup-only crash recovery.

    Recovery authenticates retained guest-bootstrap journals and signs permanent launcher-failure
    records. It does not need, open, or retain the provider credential (or any of the execution
    and scoring keys used only by the ordinary operator).
    """

    bootstrap_receipt_key: bytes
    launcher_failure_receipt_key: bytes


@dataclass(frozen=True, slots=True)
class ClinicalOperatorValidatedInputs:
    manifest: CanonicalClinicalOperatorManifest
    manifest_sha256: str
    worker_spec: FirecrackerWorkerSpec
    guest_disks: VerifiedLaneAGuestDisks
    qualification: LoadedFirecrackerQualification
    current_preflight: FirecrackerHostPreflightReceipt
    workspace: LoadedClinicalAgenticWorkspace
    secrets: ClinicalOperatorSecretMaterial


class ClinicalOperatorDryRunReport(StrictModel):
    manifest_sha256: str
    worker_spec_sha256: str
    guest_disk_build_receipt_sha256: str
    guest_boot_dispatch_manifest_sha256: str
    guest_boot_dispatch_admission: GuestBootDispatchAdmission
    rootfs_sha256: str
    harness_sha256: str
    qualification_artifact_sha256: str
    qualification_id: str
    qualification_driver_runtime_closure_manifest_sha256: str
    qualification_driver_runtime_closure_receipt_sha256: str
    qualification_driver_runtime_closure_sha256: str
    qualification_full_runtime_passed: bool
    current_linux_kvm_preflight_passed: bool
    reservation_sha256: str
    episode_id: str
    strict_evidence_loader_id: str = STRICT_CLINICAL_V02_LOADER_ID
    provider_call_made: Literal[False] = False
    worker_launched: Literal[False] = False
    registry_mutated: Literal[False] = False
    development_only: Literal[True] = True
    official_execution_qualified: Literal[False] = False
    leaderboard_admitted: Literal[False] = False


@dataclass(frozen=True, slots=True)
class PinnedClinicalProductionRunV02Loader:
    """Non-injectable strict loader with every verifier input captured by composition."""

    workspace: LoadedClinicalAgenticWorkspace
    manifest: CanonicalClinicalOperatorManifest
    worker_spec: FirecrackerWorkerSpec
    keys: FirecrackerClinicalRuntimeKeys

    def __call__(self, root: Path, expected_attempt_sha256: str) -> LoadedClinicalProductionRunV02:
        run_id = root.name
        if len(run_id) != 32 or any(character not in '0123456789abcdef' for character in run_id):
            raise ClinicalOperatorError('strict v0.2 evidence root does not end in a canonical run ID')
        system = self._expected_system()
        return load_clinical_production_run_v02(
            root,
            workspace=self.workspace,
            expected_authenticated_workspace_receipt_sha256=(
                self.manifest.expected_authenticated_workspace_receipt_sha256
            ),
            workspace_receipt_key=self.keys.workspace_receipt_key,
            expected_workspace_receipt_key_id=self.manifest.expected_workspace_receipt_key_id,
            expected_run_id=run_id,
            expected_attempt_reservation_sha256=expected_attempt_sha256,
            policy=self.manifest.execution_policy,
            harness=self.manifest.harness,
            worker_spec=self.worker_spec,
            worker_attestation_key=self.keys.worker_attestation_key,
            expected_worker_attestation_key_id=system.worker_attestation_key_id,
            gateway_receipt_key=self.keys.gateway_receipt_key,
            expected_gateway_receipt_key_id=system.gateway_receipt_key_id,
            expected_gateway_policy_sha256=system.gateway_policy_sha256,
            expected_gateway_route_sha256=system.gateway_route_sha256,
            guest_rpc_receipt_key=self.keys.guest_rpc_receipt_key,
            expected_guest_rpc_receipt_key_id=system.guest_rpc_receipt_key_id,
            expected_guest_rpc_policy_sha256=system.guest_rpc_policy_sha256,
            clinical_guest_bootstrap_receipt_key=self.keys.clinical_guest_bootstrap_receipt_key,
            expected_clinical_guest_bootstrap_receipt_key_id=system.guest_bootstrap_receipt_key_id,
            clinical_guest_bootstrap_trust_anchor=self.manifest.bootstrap_trust_anchor,
            receipt_key=self.keys.production_receipt_key,
            expected_receipt_key_id=system.production_receipt_key_id,
        )

    def _expected_system(self) -> ClinicalProductionSystemIdentity:
        return expected_system_identity(self.manifest, self.keys)


def load_canonical_clinical_operator_manifest(
    path: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[CanonicalClinicalOperatorManifest, str]:
    """Read exact canonical JSON and require an external, non-manifest digest."""

    _require_sha256(expected_manifest_sha256, 'operator manifest pin')
    content = _read_stable_regular_file(path, maximum_bytes=_MAX_MANIFEST_BYTES, private=False)
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(observed_sha256, expected_manifest_sha256):
        raise ClinicalOperatorError('operator manifest differs from its external SHA-256 pin')
    try:
        manifest = CanonicalClinicalOperatorManifest.model_validate_json(content)
    except ValueError:
        raise ClinicalOperatorError('operator manifest has an invalid strict schema') from None
    if canonical_json_bytes(manifest) != content:
        raise ClinicalOperatorError('operator manifest must use exact canonical JSON')
    validate_checked_in_executable_pins(manifest)
    return manifest, observed_sha256


def validate_checked_in_executable_pins(manifest: CanonicalClinicalOperatorManifest) -> None:
    """Bind five entry source modules; this explicitly is not transitive/process attestation."""

    expected = (
        (clinical_operator_module, manifest.operator_executable_sha256, 'operator'),
        (
            clinical_launcher_module,
            manifest.deployment.canonical_launcher_executable_sha256,
            'canonical launcher',
        ),
        (
            firecracker_clinical_runtime_module,
            manifest.runtime_config.runtime_executable_sha256,
            'clinical Firecracker runtime',
        ),
        (
            clinical_production_run_v02_module,
            manifest.strict_evidence_loader_executable_sha256,
            'strict v0.2 evidence loader',
        ),
        (
            provider_subprocess_module,
            manifest.provider_subprocess_module_source_sha256,
            'provider child module source',
        ),
    )
    for module, pinned, label in expected:
        if not hmac.compare_digest(_module_sha256(module), pinned):
            raise ClinicalOperatorError(f'{label} differs from its deployment pin')


def validate_operator_inputs(
    manifest: CanonicalClinicalOperatorManifest,
    *,
    manifest_sha256: str,
    secret_root: Path,
) -> ClinicalOperatorValidatedInputs:
    """Authenticate secrets, prior qualification, current host, and workspace without state writes."""

    secrets = load_operator_secret_directory(secret_root)
    try:
        _validate_secret_pins(manifest, secrets)
        verify_operator_provider_data_control_attestation(manifest)
        worker_spec, _ = load_pinned_firecracker_worker_spec(
            Path(manifest.worker_spec_path),
            expected_worker_spec_sha256=manifest.expected_worker_spec_sha256,
        )
        validate_side_effect_free_runtime_parity(manifest, worker_spec)
        guest_disks = load_and_verify_operator_guest_disks(manifest, worker_spec)
        qualification = load_firecracker_qualification(
            Path(manifest.qualification_root),
            qualification_key=secrets.qualification_key,
            expected_qualification_key_id=manifest.expected_qualification_key_id,
            expected_worker_spec_sha256=manifest.expected_worker_spec_sha256,
            expected_artifact_sha256=manifest.expected_qualification_artifact_sha256,
            expected_collector_evidence_sha256=manifest.expected_collector_evidence_sha256,
            expected_probe_manifest_sha256=manifest.expected_probe_manifest_sha256,
            expected_driver_runtime_closure_manifest_sha256=(manifest.expected_driver_runtime_closure_manifest_sha256),
            expected_driver_runtime_closure_receipt_sha256=(manifest.expected_driver_runtime_closure_receipt_sha256),
            expected_driver_runtime_closure_sha256=(manifest.expected_driver_runtime_closure_sha256),
            expected_collector_public_key_hex=manifest.expected_collector_public_key_hex,
            expected_collector_key_id=manifest.expected_collector_key_id,
            expected_verifier_source_sha256=(manifest.expected_qualification_verifier_source_sha256),
        )
        record = qualification.authenticated.record
        if (
            record.qualified is not True
            or record.preflight is None
            or record.full_suite_evidence is None
            or record.full_suite_evidence.all_required_drills_passed is not True
        ):
            raise ClinicalOperatorError('Firecracker configuration lacks authenticated full-runtime qualification')
        supervisor = FirecrackerSupervisor(worker_spec)
        try:
            current_preflight = supervisor.preflight()
        except Exception:
            raise ClinicalOperatorError('current host failed Linux/KVM Firecracker preflight') from None
        _validate_current_host_matches_qualification(current_preflight, qualification)
        workspace = load_clinical_agentic_workspace(
            Path(manifest.workspace_root),
            expected_authenticated_receipt_sha256=manifest.expected_authenticated_workspace_receipt_sha256,
            receipt_key=secrets.workspace_receipt_key,
            expected_receipt_key_id=manifest.expected_workspace_receipt_key_id,
        )
        if workspace.task.context.episode_id != manifest.episode_id:
            raise ClinicalOperatorError('workspace episode differs from the one pinned by the run manifest')
        return ClinicalOperatorValidatedInputs(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            worker_spec=worker_spec,
            guest_disks=guest_disks,
            qualification=qualification,
            current_preflight=current_preflight,
            workspace=workspace,
            secrets=secrets,
        )
    except BaseException:
        secrets.close()
        raise


def verify_operator_provider_data_control_attestation(
    manifest: CanonicalClinicalOperatorManifest,
) -> str | None:
    """Load the exact external evidence committed by a non-default provider route.

    The gateway route carries only a SHA-256 commitment.  This operator boundary proves that a
    separately provisioned, trusted-owner, non-writable artifact with those exact bytes exists;
    it does not independently prove the provider honored the underlying contract or account
    setting.
    """

    path_value = manifest.provider_data_control_attestation_path
    expected_sha256 = manifest.gateway_route.provider_data_control_attestation_sha256
    externally_controlled = manifest.gateway_route.provider_data_control != 'default'
    if not externally_controlled:
        if path_value is not None or expected_sha256 is not None:
            raise ClinicalOperatorError('default provider data control cannot carry an attestation artifact')
        return None
    if path_value is None or expected_sha256 is None:
        raise ClinicalOperatorError('non-default provider data control lacks its external attestation artifact')
    content = _read_stable_regular_file(
        Path(path_value),
        maximum_bytes=8 * 1024 * 1024,
        private=False,
        trusted_public=True,
    )
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(observed_sha256, expected_sha256):
        raise ClinicalOperatorError('provider data-control attestation bytes differ from the route commitment')
    return observed_sha256


def dry_run_report(inputs: ClinicalOperatorValidatedInputs) -> ClinicalOperatorDryRunReport:
    """Return a content-free readiness report; it neither opens SQLite nor starts a worker."""

    record = inputs.qualification.authenticated.record
    return ClinicalOperatorDryRunReport(
        manifest_sha256=inputs.manifest_sha256,
        worker_spec_sha256=inputs.manifest.expected_worker_spec_sha256,
        guest_disk_build_receipt_sha256=inputs.guest_disks.receipt_sha256,
        guest_boot_dispatch_manifest_sha256=(inputs.guest_disks.receipt.guest_boot_dispatch_manifest_sha256),
        guest_boot_dispatch_admission=(inputs.guest_disks.receipt.guest_boot_dispatch.admission),
        rootfs_sha256=inputs.guest_disks.receipt.rootfs.sha256,
        harness_sha256=inputs.guest_disks.receipt.harness.sha256,
        qualification_artifact_sha256=inputs.qualification.artifact_sha256,
        qualification_id=record.qualification_id,
        qualification_driver_runtime_closure_manifest_sha256=(
            inputs.manifest.expected_driver_runtime_closure_manifest_sha256
        ),
        qualification_driver_runtime_closure_receipt_sha256=(
            inputs.manifest.expected_driver_runtime_closure_receipt_sha256
        ),
        qualification_driver_runtime_closure_sha256=(inputs.manifest.expected_driver_runtime_closure_sha256),
        qualification_full_runtime_passed=record.qualified,
        current_linux_kvm_preflight_passed=True,
        reservation_sha256=inputs.manifest.reservation_sha256,
        episode_id=inputs.manifest.episode_id,
    )


def execute_operator_task(inputs: ClinicalOperatorValidatedInputs) -> ClinicalLauncherResult:
    """Construct the only supported deployment composition and consume exactly one task attempt."""

    if inputs.manifest.registry_execution_mode != 'development-local-sqlite':
        raise ClinicalOperatorError('managed-authority manifests cannot fall back to caller-selected local SQLite')
    return _execute_operator_task(
        inputs,
        managed_registry=None,
        startup_reconciler=None,
        managed_ownership=None,
    )


def execute_managed_operator_task(
    inputs: ClinicalOperatorValidatedInputs,
    *,
    managed_registry: ManagedClinicalRegistryClient,
    startup_reconciler: ManagedClinicalStartupReconciler,
    managed_ownership: DurableManagedClinicalOwnershipLedger,
) -> ClinicalLauncherResult:
    """Run through the managed authority and mandatory global startup cleanup gate.

    The registry service and reaper are deployment-owned objects, not manifest/run-caller plugins.
    The standalone managed entrypoint constructs them from its fixed root-owned configuration.
    """

    manifest = inputs.manifest
    if manifest.registry_execution_mode != 'managed-unix-authority':
        raise ClinicalOperatorError('local-SQLite manifests cannot be relabeled as managed authority executions')
    if (
        not isinstance(managed_registry, ManagedClinicalRegistryClient)
        or managed_registry.config_sha256 != manifest.managed_registry_config_sha256
    ):
        raise ClinicalOperatorError('managed registry client differs from the externally pinned service configuration')
    if managed_clinical_startup_config_sha256(startup_reconciler.config) != manifest.managed_startup_config_sha256:
        raise ClinicalOperatorError(
            'managed startup reconciler differs from the externally pinned deployment configuration'
        )
    if (
        not isinstance(managed_ownership, DurableManagedClinicalOwnershipLedger)
        or managed_clinical_ownership_config_sha256(managed_ownership.config)
        != manifest.managed_ownership_config_sha256
    ):
        raise ClinicalOperatorError(
            'managed ownership ledger differs from the externally pinned deployment configuration'
        )
    if (
        not isinstance(startup_reconciler.host, LinuxManagedClinicalHostAdapter)
        or startup_reconciler.host.ownership is not managed_ownership
    ):
        raise ClinicalOperatorError('startup cleanup is not bound to the exact managed ownership ledger')
    validate_managed_gateway_capability_binding(
        startup_reconciler.capabilities,
        managed_ownership=managed_ownership,
        expected_ledger_path=Path(manifest.gateway_ledger_path),
        expected_model_route_sha256=gateway_model_route_sha256(manifest.gateway_route),
    )
    return _execute_operator_task(
        inputs,
        managed_registry=managed_registry,
        startup_reconciler=startup_reconciler,
        managed_ownership=managed_ownership,
    )


def validate_managed_gateway_capability_binding(
    capabilities: object,
    *,
    managed_ownership: DurableManagedClinicalOwnershipLedger,
    expected_ledger_path: Path,
    expected_model_route_sha256: str,
) -> None:
    """Reject a type-correct reaper wired to another route or SQLite authority."""

    if (
        type(capabilities) is not RestartVisibleManagedGatewayCapabilityLedger
        or capabilities.ownership is not managed_ownership
        or type(capabilities.gateway_ledger) is not SqliteGatewayLedger
        or not hmac.compare_digest(
            capabilities.expected_model_route_sha256,
            expected_model_route_sha256,
        )
    ):
        raise ClinicalOperatorError(
            'startup capability cleanup differs from the exact ownership, route, or gateway ledger'
        )
    supplied = expected_ledger_path.expanduser()
    if supplied.is_symlink():
        raise ClinicalOperatorError('managed gateway ledger cannot be a symbolic link')
    try:
        parent = supplied.parent.resolve(strict=True)
        canonical = parent / supplied.name
        canonical_lock = canonical.with_name(f'{canonical.name}.admission.lock')
        expected_metadata = supplied.lstat()
        expected_lock_metadata = canonical_lock.lstat()
        bridge_identity = capabilities.gateway_ledger.identity
    except (OSError, ValueError):
        raise ClinicalOperatorError('managed gateway ledger or admission lock identity is unavailable') from None
    if (
        capabilities.gateway_ledger.path != canonical
        or bridge_identity.resolved_path != str(canonical)
        or bridge_identity.admission_lock_resolved_path != str(canonical_lock)
        or parent != supplied.parent
        or not stat.S_ISREG(expected_metadata.st_mode)
        or expected_metadata.st_uid != os.geteuid()
        or expected_metadata.st_nlink != 1
        or stat.S_IMODE(expected_metadata.st_mode) != 0o600
        or not stat.S_ISREG(expected_lock_metadata.st_mode)
        or expected_lock_metadata.st_uid != os.geteuid()
        or expected_lock_metadata.st_nlink != 1
        or stat.S_IMODE(expected_lock_metadata.st_mode) != 0o600
        or (expected_metadata.st_dev, expected_metadata.st_ino) != (bridge_identity.device_id, bridge_identity.inode)
        or (expected_lock_metadata.st_dev, expected_lock_metadata.st_ino)
        != (
            bridge_identity.admission_lock_device_id,
            bridge_identity.admission_lock_inode,
        )
    ):
        raise ClinicalOperatorError(
            'startup capability cleanup uses a different gateway-ledger database or admission lock'
        )


def _execute_operator_task(
    inputs: ClinicalOperatorValidatedInputs,
    *,
    managed_registry: ClinicalProductionRegistryBoundary | None,
    startup_reconciler: ManagedClinicalStartupReconciler | None,
    managed_ownership: DurableManagedClinicalOwnershipLedger | None,
) -> ClinicalLauncherResult:
    """Shared checked composition; managed mode cannot silently fall back to local SQLite."""

    manifest = inputs.manifest
    secrets = inputs.secrets
    runtime_keys = FirecrackerClinicalRuntimeKeys(
        workspace_receipt_key=secrets.workspace_receipt_key,
        worker_attestation_key=secrets.worker_attestation_key,
        gateway_receipt_key=secrets.gateway_receipt_key,
        guest_rpc_receipt_key=secrets.guest_rpc_receipt_key,
        clinical_guest_bootstrap_receipt_key=secrets.bootstrap_receipt_key,
        production_receipt_key=secrets.production_receipt_key,
    )
    managed_components = (
        managed_registry,
        startup_reconciler,
        managed_ownership,
    )
    if any(component is None for component in managed_components) and not all(
        component is None for component in managed_components
    ):
        raise ClinicalOperatorError(
            'managed registry, startup reconciler, and ownership ledger must be supplied together'
        )
    if managed_registry is None:
        registry_path = Path(manifest.registry_path)
        _require_existing_private_file(registry_path, 'authoritative clinical registry')
        registry: ClinicalProductionRegistryBoundary = SqliteClinicalProductionRegistry(
            registry_path,
            authority_id=manifest.deployment.registry_authority_id,
        )
    else:
        registry = managed_registry
        if registry.authority_id != manifest.deployment.registry_authority_id:
            raise ClinicalOperatorError('managed registry belongs to a different authority')
    context = registry.reservation_context(manifest.reservation_sha256)
    expected_system = expected_system_identity(manifest, runtime_keys)
    if context.reservation.system != expected_system or (
        context.reservation.system_identity_sha256 != clinical_production_system_identity_sha256(expected_system)
    ):
        raise ClinicalOperatorError('authoritative reservation differs from the operator composition')
    if manifest.deployment.expected_system_identity_sha256 != context.reservation.system_identity_sha256:
        raise ClinicalOperatorError('launcher deployment differs from the authoritative reservation')
    binding = next(
        (item for item in context.reservation.tasks if item.episode_id == manifest.episode_id),
        None,
    )
    if binding is None or binding.authenticated_workspace_receipt_sha256 != (
        manifest.expected_authenticated_workspace_receipt_sha256
    ):
        raise ClinicalOperatorError('operator workspace is not the exact reserved task binding')

    secret_store = InMemoryGatewaySecretStore()
    gateway = AuthenticatedProviderGateway(
        policy=manifest.gateway_policy,
        ledger=SqliteGatewayLedger(Path(manifest.gateway_ledger_path)),
        secret_resolver=secret_store,
        adapters=(
            SubprocessProviderAdapter(
                descriptor=manifest.provider_adapter,
                spec=manifest.provider_subprocess,
                credential_descriptor_supplier=lambda: secrets.provider_credential_fd,
            ),
        ),
        receipt_key=secrets.gateway_receipt_key,
    )
    signer = LocalEd25519Signer(Ed25519PrivateKey.from_private_bytes(secrets.bootstrap_authorization_seed))
    supervisor = FirecrackerSupervisor(inputs.worker_spec)
    runtime = FirecrackerClinicalRuntime(
        config=manifest.runtime_config,
        supervisor=supervisor,
        gateway=gateway,
        gateway_secret_store=secret_store,
        execution_policy=manifest.execution_policy,
        gateway_route=manifest.gateway_route,
        provider_subprocess_spec_sha256=provider_subprocess_spec_sha256(manifest.provider_subprocess),
        provider_subprocess_behavior_sha256=provider_subprocess_behavior_sha256(manifest.provider_subprocess),
        provider_subprocess_module_source_sha256=(manifest.provider_subprocess_module_source_sha256),
        guest_rpc_policy=manifest.guest_rpc_policy,
        harness=manifest.harness,
        keys=runtime_keys,
        bootstrap_authorization_signer=signer,
        bootstrap_trust_anchor=manifest.bootstrap_trust_anchor,
        evidence_root=Path(manifest.evidence_root),
        require_global_startup_reconciliation=startup_reconciler is not None,
        managed_ownership=managed_ownership,
    )
    managed_attempt_snapshot = None
    if startup_reconciler is not None:
        if not isinstance(registry, ManagedClinicalRegistryClient):
            raise ClinicalOperatorError('managed startup reconciliation requires the concrete registry client')
        managed_attempt_snapshot = registry.begin_reconciliation()
        reconcile_canonical_managed_runtime_startup(
            runtime,
            reconciler=startup_reconciler,
        )
        terminalizer = CanonicalClinicalRecoveryTerminalizer(
            registry=registry,
            deployment=manifest.deployment,
            failure_receipt_key=secrets.launcher_failure_receipt_key,
        )
        consumed_run_ids = {item.run_id for item in managed_attempt_snapshot if item.state == 'launched'}
        terminalized = tuple(
            failure
            for reservation_sha256 in sorted(
                {item.reservation_sha256 for item in managed_attempt_snapshot if item.state == 'launched'}
            )
            for failure in terminalizer.reconcile_consumed_tasks(reservation_sha256=reservation_sha256)
        )
        if {item.launch.run_id for item in terminalized} != consumed_run_ids:
            raise ClinicalOperatorError('managed startup did not terminalize the exact consumed attempt snapshot')
    loader = PinnedClinicalProductionRunV02Loader(
        workspace=inputs.workspace,
        manifest=manifest,
        worker_spec=inputs.worker_spec,
        keys=runtime_keys,
    )
    launcher = CanonicalClinicalLauncher(
        registry=registry,
        deployment=manifest.deployment,
        runtime=runtime,
        evidence_loader=loader,
        failure_receipt_key=secrets.launcher_failure_receipt_key,
    )
    return launcher.execute_reserved_task(
        reservation_sha256=manifest.reservation_sha256,
        episode_id=manifest.episode_id,
        workspace=inputs.workspace,
    )


def expected_system_identity(
    manifest: CanonicalClinicalOperatorManifest,
    keys: FirecrackerClinicalRuntimeKeys,
) -> ClinicalProductionSystemIdentity:
    return ClinicalProductionSystemIdentity(
        harness=manifest.harness,
        execution_policy_sha256=agentic_policy_sha256(manifest.execution_policy),
        worker_spec_sha256=manifest.expected_worker_spec_sha256,
        gateway_policy_sha256=authenticated_gateway_policy_sha256(manifest.gateway_policy),
        gateway_route=manifest.gateway_route,
        gateway_route_sha256=gateway_model_route_sha256(manifest.gateway_route),
        provider_subprocess_spec_sha256=provider_subprocess_spec_sha256(manifest.provider_subprocess),
        provider_subprocess_behavior_sha256=provider_subprocess_behavior_sha256(manifest.provider_subprocess),
        provider_subprocess_module_source_sha256=(manifest.provider_subprocess_module_source_sha256),
        guest_rpc_policy_sha256=guest_rpc_policy_sha256(manifest.guest_rpc_policy),
        guest_bootstrap_authorization_key_id=manifest.runtime_config.bootstrap_authorization_key_id,
        guest_bootstrap_receipt_key_id=manifest.runtime_config.bootstrap_receipt_key_id,
        worker_attestation_key_id=firecracker_attestation_key_id(keys.worker_attestation_key),
        gateway_receipt_key_id=gateway_session_key_id(keys.gateway_receipt_key),
        guest_rpc_receipt_key_id=guest_rpc_session_key_id(keys.guest_rpc_receipt_key),
        production_receipt_key_id=clinical_production_run_key_id(keys.production_receipt_key),
        canonical_launcher_id=manifest.deployment.canonical_launcher_id,
        canonical_launcher_executable_sha256=manifest.deployment.canonical_launcher_executable_sha256,
    )


def validate_side_effect_free_runtime_parity(
    manifest: CanonicalClinicalOperatorManifest,
    worker_spec: FirecrackerWorkerSpec,
) -> None:
    """Mirror constructor checks which need the externally pinned worker specification."""

    worker_limits = worker_spec.limits
    policy_limits = manifest.execution_policy.limits
    if (
        worker_limits.wall_seconds != policy_limits.wall_seconds
        or worker_limits.memory_mib != policy_limits.memory_mib
        or worker_limits.pids != policy_limits.pids
        or worker_limits.scratch_bytes != policy_limits.scratch_mib * 1024 * 1024
        or not math.isclose(
            worker_limits.cpu_quota_us / worker_limits.cpu_period_us,
            policy_limits.cpus,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or worker_limits.vcpu_count < math.ceil(policy_limits.cpus)
    ):
        raise ClinicalOperatorError('worker resources do not exactly implement the execution policy')
    try:
        require_submitted_harness_binding(
            manifest=manifest.submitted_harness,
            identity=manifest.harness,
            worker_harness_sha256=worker_spec.images.harness.sha256,
            worker_harness_byte_count=worker_spec.images.harness.byte_count,
            logical_model_id=manifest.gateway_route.logical_model_id,
            adapter_id=manifest.gateway_route.adapter_id,
        )
    except SubmittedHarnessError as error:
        raise ClinicalOperatorError(
            'submitted harness image or identity differs from the pinned worker, or lacks runtime support'
        ) from error

    if manifest.bootstrap_trust_anchor.worker_bootstrap_profile_sha256 != (
        firecracker_guest_bootstrap_profile_sha256(worker_spec)
    ):
        raise ClinicalOperatorError('bootstrap trust anchor differs from the pinned worker bootstrap profile')

    expected_guest_config = LaneAClinicalGuestConfig(
        trust_anchor=manifest.bootstrap_trust_anchor,
        guest_rpc_port=worker_spec.guest_rpc_port,
    )
    expected_guest_config_sha256 = lane_a_clinical_guest_config_sha256(expected_guest_config)
    dispatch = manifest.guest_boot_dispatch
    if not hmac.compare_digest(
        expected_guest_config_sha256,
        manifest.submitted_harness.baked_config_sha256,
    ) or not hmac.compare_digest(
        expected_guest_config_sha256,
        dispatch.guest_config_sha256,
    ):
        raise ClinicalOperatorError(
            'guest dispatch config does not bind the operator trust anchor and worker vsock port'
        )


def load_and_verify_operator_guest_disks(
    manifest: CanonicalClinicalOperatorManifest,
    worker_spec: FirecrackerWorkerSpec,
) -> VerifiedLaneAGuestDisks:
    """Reload the externally pinned build receipt and exact task-disk bytes."""

    try:
        receipt = load_lane_a_guest_disk_build_receipt(
            Path(manifest.guest_disk_build_receipt_path),
            expected_receipt_sha256=manifest.expected_guest_disk_build_receipt_sha256,
        )
        verified = verify_lane_a_guest_disk_build(
            receipt=receipt,
            rootfs_path=Path(worker_spec.images.rootfs.source_path),
            harness_path=Path(worker_spec.images.harness.source_path),
            expected_base_rootfs_source_sha256=(manifest.expected_base_rootfs_source_sha256),
            expected_harness_payload_source_sha256=(manifest.expected_harness_payload_source_sha256),
            expected_guest_executable_sha256=(manifest.submitted_harness.guest_executable_sha256),
            expected_guest_config_sha256=manifest.submitted_harness.baked_config_sha256,
            expected_guest_boot_dispatch=manifest.guest_boot_dispatch,
            expected_guest_boot_dispatch_manifest_sha256=(
                guest_boot_dispatch_manifest_sha256(manifest.guest_boot_dispatch)
            ),
            expected_mke2fs_sha256=manifest.expected_mke2fs_sha256,
            expected_e2fsck_sha256=manifest.expected_e2fsck_sha256,
            expected_debugfs_sha256=manifest.expected_debugfs_sha256,
            expected_tool_runtime_closure_manifest_sha256=(manifest.expected_tool_runtime_closure_manifest_sha256),
            expected_builder_source_sha256=(manifest.expected_guest_disk_builder_source_sha256),
            require_production=True,
        )
    except (LaneAGuestDiskBuildError, ValueError):
        raise ClinicalOperatorError(
            'Lane A task disks or reproducible build receipt failed independent verification'
        ) from None
    validate_operator_guest_disk_binding(manifest, worker_spec, verified.receipt)
    return verified


def validate_operator_guest_disk_binding(
    manifest: CanonicalClinicalOperatorManifest,
    worker_spec: FirecrackerWorkerSpec,
    receipt: LaneAGuestDiskBuildReceipt,
) -> None:
    """Cross-bind one verified disk receipt to the worker and submitted harness."""

    submitted = manifest.submitted_harness
    dispatch = manifest.guest_boot_dispatch
    expected_guest_config_sha256 = lane_a_clinical_guest_config_sha256(
        LaneAClinicalGuestConfig(
            trust_anchor=manifest.bootstrap_trust_anchor,
            guest_rpc_port=worker_spec.guest_rpc_port,
        )
    )
    expected = (
        worker_spec.images.rootfs.sha256,
        worker_spec.images.rootfs.byte_count,
        worker_spec.images.harness.sha256,
        worker_spec.images.harness.byte_count,
        submitted.harness_image_sha256,
        submitted.harness_image_byte_count,
        submitted.normalized_runtime_tree_sha256,
        submitted.guest_executable_path,
        submitted.guest_executable_sha256,
        submitted.guest_argv,
        dispatch,
        guest_boot_dispatch_manifest_sha256(dispatch),
        expected_guest_config_sha256,
        submitted.baked_config_sha256,
        submitted.dependency_closure_sha256,
        submitted.reproducible_build_receipt_sha256,
    )
    observed = (
        receipt.rootfs.sha256,
        receipt.rootfs.byte_count,
        receipt.harness.sha256,
        receipt.harness.byte_count,
        receipt.harness.sha256,
        receipt.harness.byte_count,
        receipt.harness.normalized_tree_sha256,
        receipt.guest_executable_path,
        receipt.guest_executable_sha256,
        receipt.fixed_guest_argv,
        receipt.guest_boot_dispatch,
        receipt.guest_boot_dispatch_manifest_sha256,
        receipt.guest_config_sha256,
        receipt.guest_config_sha256,
        receipt.dependency_closure_sha256,
        lane_a_guest_disk_build_receipt_sha256(receipt),
    )
    if expected != observed:
        raise ClinicalOperatorError('verified task disks differ from the worker, guest config, or submitted harness')


def load_operator_secret_directory(root: Path) -> ClinicalOperatorSecretMaterial:
    """Load fixed-name private files; no secret bytes or paths enter the manifest or result JSON."""

    directory_fd = _open_private_directory_descriptor(root, 'operator secret directory')
    provider_fd = -1
    try:
        if set(os.listdir(directory_fd)) != _SECRET_FILE_NAMES:
            raise ClinicalOperatorError('operator secret directory has an unexpected file inventory')
        provider_fd = _open_private_secret_file_at(
            directory_fd,
            'provider-credential',
            minimum_bytes=16,
            maximum_bytes=64 * 1024,
        )
        return ClinicalOperatorSecretMaterial(
            workspace_receipt_key=_read_private_secret_at(directory_fd, 'workspace-receipt.key'),
            worker_attestation_key=_read_private_secret_at(directory_fd, 'worker-attestation.key'),
            gateway_receipt_key=_read_private_secret_at(directory_fd, 'gateway-receipt.key'),
            guest_rpc_receipt_key=_read_private_secret_at(directory_fd, 'guest-rpc-receipt.key'),
            bootstrap_receipt_key=_read_private_secret_at(directory_fd, 'bootstrap-receipt.key'),
            production_receipt_key=_read_private_secret_at(directory_fd, 'production-receipt.key'),
            launcher_failure_receipt_key=_read_private_secret_at(
                directory_fd,
                'launcher-failure-receipt.key',
            ),
            qualification_key=_read_private_qualification_key_at(
                directory_fd,
                'qualification.key',
            ),
            bootstrap_authorization_seed=_read_private_secret_at(
                directory_fd,
                'bootstrap-authorization.seed',
                exact_bytes=32,
            ),
            provider_credential_fd=provider_fd,
        )
    except BaseException:
        if provider_fd >= 0:
            os.close(provider_fd)
        raise
    finally:
        os.close(directory_fd)


def load_operator_recovery_secret_directory(
    root: Path,
) -> ClinicalOperatorRecoverySecretMaterial:
    """Load only cleanup keys while still requiring the fixed secret inventory.

    Reusing :func:`load_operator_secret_directory` would open and retain the provider credential
    even though recovery has no provider/model execution surface. The directory-name inventory
    remains exact, but only the two files required by cleanup are opened.
    """

    directory_fd = _open_private_directory_descriptor(
        root,
        'operator recovery secret directory',
    )
    try:
        if set(os.listdir(directory_fd)) != _SECRET_FILE_NAMES:
            raise ClinicalOperatorError('operator recovery secret directory has an unexpected file inventory')
        return ClinicalOperatorRecoverySecretMaterial(
            bootstrap_receipt_key=_read_private_secret_at(
                directory_fd,
                'bootstrap-receipt.key',
            ),
            launcher_failure_receipt_key=_read_private_secret_at(
                directory_fd,
                'launcher-failure-receipt.key',
            ),
        )
    finally:
        os.close(directory_fd)


def validate_operator_secret_material(
    manifest: CanonicalClinicalOperatorManifest,
    secrets: ClinicalOperatorSecretMaterial,
) -> None:
    """Verify every fixed-name secret against the root-owned manifest's key commitments."""

    _validate_secret_pins(manifest, secrets)


def validate_operator_recovery_secret_material(
    manifest: CanonicalClinicalOperatorManifest,
    secrets: ClinicalOperatorRecoverySecretMaterial,
) -> None:
    """Bind the only two secrets cleanup uses to their non-secret manifest IDs."""

    if (
        clinical_guest_bootstrap_receipt_key_id(secrets.bootstrap_receipt_key),
        clinical_launcher_failure_key_id(secrets.launcher_failure_receipt_key),
    ) != (
        manifest.runtime_config.bootstrap_receipt_key_id,
        manifest.deployment.failure_receipt_key_id,
    ):
        raise ClinicalOperatorError('one or more recovery secrets differ from their non-secret key-ID pins')


def _validate_secret_pins(
    manifest: CanonicalClinicalOperatorManifest,
    secrets: ClinicalOperatorSecretMaterial,
) -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.from_private_bytes(secrets.bootstrap_authorization_seed))
    observed = (
        clinical_workspace_receipt_key_id(secrets.workspace_receipt_key),
        firecracker_attestation_key_id(secrets.worker_attestation_key),
        gateway_session_key_id(secrets.gateway_receipt_key),
        guest_rpc_session_key_id(secrets.guest_rpc_receipt_key),
        clinical_guest_bootstrap_receipt_key_id(secrets.bootstrap_receipt_key),
        clinical_production_run_key_id(secrets.production_receipt_key),
        clinical_launcher_failure_key_id(secrets.launcher_failure_receipt_key),
        firecracker_qualification_key_id(secrets.qualification_key),
        clinical_guest_bootstrap_authorization_key_id(signer.public_key_bytes()),
    )
    expected = (
        manifest.expected_workspace_receipt_key_id,
        # Worker/gateway/RPC/production IDs are reservation-pinned and rechecked before execution;
        # bootstrap, launcher, and qualification IDs are pinned directly by this manifest.
        firecracker_attestation_key_id(secrets.worker_attestation_key),
        manifest.gateway_policy.receipt_key_id,
        guest_rpc_session_key_id(secrets.guest_rpc_receipt_key),
        manifest.runtime_config.bootstrap_receipt_key_id,
        clinical_production_run_key_id(secrets.production_receipt_key),
        manifest.deployment.failure_receipt_key_id,
        manifest.expected_qualification_key_id,
        manifest.runtime_config.bootstrap_authorization_key_id,
    )
    if observed != expected:
        raise ClinicalOperatorError('one or more operator secrets differ from their non-secret key-ID pins')
    if signer.public_key_bytes().hex() != manifest.bootstrap_trust_anchor.ed25519_public_key_hex:
        raise ClinicalOperatorError('bootstrap signing seed differs from the independent guest trust anchor')
    runtime_keys = FirecrackerClinicalRuntimeKeys(
        workspace_receipt_key=secrets.workspace_receipt_key,
        worker_attestation_key=secrets.worker_attestation_key,
        gateway_receipt_key=secrets.gateway_receipt_key,
        guest_rpc_receipt_key=secrets.guest_rpc_receipt_key,
        clinical_guest_bootstrap_receipt_key=secrets.bootstrap_receipt_key,
        production_receipt_key=secrets.production_receipt_key,
    )
    system_identity_sha256 = clinical_production_system_identity_sha256(
        expected_system_identity(manifest, runtime_keys)
    )
    if not hmac.compare_digest(system_identity_sha256, manifest.deployment.expected_system_identity_sha256):
        raise ClinicalOperatorError('secret-backed system identity differs from the launcher deployment pin')


def _validate_current_host_matches_qualification(
    current: FirecrackerHostPreflightReceipt,
    qualification: LoadedFirecrackerQualification,
) -> None:
    previous = qualification.authenticated.record.preflight
    if previous is None:
        raise ClinicalOperatorError('qualified artifact lacks its host preflight')
    if (
        current.worker_spec_sha256,
        current.host_os,
        current.host_architecture,
        current.host_kernel_release,
        current.effective_uid,
        current.cgroup_controllers,
    ) != (
        previous.worker_spec_sha256,
        previous.host_os,
        previous.host_architecture,
        previous.host_kernel_release,
        previous.effective_uid,
        previous.cgroup_controllers,
    ):
        raise ClinicalOperatorError('current host differs from the host configuration that passed qualification')


def _module_sha256(module: ModuleType) -> str:
    location = getattr(module, '__file__', None)
    if not isinstance(location, str):
        raise ClinicalOperatorError('checked-in executable module has no stable file')
    content = _read_stable_regular_file(Path(location), maximum_bytes=16 * 1024 * 1024, private=False)
    return hashlib.sha256(content).hexdigest()


def _read_stable_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    private: bool,
    trusted_public: bool = False,
) -> bytes:
    try:
        before = path.lstat()
    except OSError:
        raise ClinicalOperatorError('pinned input is unavailable') from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > maximum_bytes
        or (private and (before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) != 0o600))
        or (trusted_public and (before.st_uid not in {0, os.geteuid()} or bool(stat.S_IMODE(before.st_mode) & 0o022)))
    ):
        raise ClinicalOperatorError('pinned input has unsafe metadata')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        content = bytearray()
        while len(content) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        after = os.fstat(descriptor)
    except OSError:
        raise ClinicalOperatorError('pinned input could not be read') from None
    finally:
        os.close(descriptor)
    if len(content) > maximum_bytes or (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
        raise ClinicalOperatorError('pinned input changed while it was read')
    return bytes(content)


def _require_private_directory(path: Path, label: str) -> Path:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ClinicalOperatorError(f'{label} cannot be a symbolic link')
    try:
        resolved = supplied.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError:
        raise ClinicalOperatorError(f'{label} is unavailable') from None
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ClinicalOperatorError(f'{label} must be an owned mode-0700 directory')
    return resolved


def _open_private_directory_descriptor(path: Path, label: str) -> int:
    """Open one exact private directory without following its final path component."""

    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ClinicalOperatorError(f'{label} is unavailable') from None
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        os.close(descriptor)
        raise ClinicalOperatorError(f'{label} must be an owned mode-0700 directory')
    return descriptor


def _open_private_secret_file_at(
    directory_fd: int,
    name: str,
    *,
    minimum_bytes: int,
    maximum_bytes: int,
) -> int:
    """Open one fixed-name secret relative to the already authenticated directory."""

    if '/' in name or name in {'', '.', '..'}:
        raise ClinicalOperatorError('required operator secret name is invalid')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError:
        raise ClinicalOperatorError('required operator secret file is unavailable') from None
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not minimum_bytes <= metadata.st_size <= maximum_bytes
    ):
        os.close(descriptor)
        raise ClinicalOperatorError('required operator secret file has unsafe metadata')
    return descriptor


def _read_private_secret_at(
    directory_fd: int,
    name: str,
    *,
    exact_bytes: int | None = None,
) -> bytes:
    maximum = exact_bytes if exact_bytes is not None else _MAX_SECRET_BYTES
    minimum = exact_bytes if exact_bytes is not None else 32
    descriptor = _open_private_secret_file_at(
        directory_fd,
        name,
        minimum_bytes=minimum,
        maximum_bytes=maximum,
    )
    try:
        content = bytearray()
        while len(content) <= maximum:
            block = os.read(descriptor, min(65_536, maximum + 1 - len(content)))
            if not block:
                break
            content.extend(block)
        after = os.fstat(descriptor)
    except OSError:
        raise ClinicalOperatorError('operator secret could not be read safely') from None
    finally:
        os.close(descriptor)
    if len(content) > maximum or after.st_size != len(content):
        raise ClinicalOperatorError('operator secret changed while it was read')
    return bytes(content)


def _read_private_qualification_key_at(directory_fd: int, name: str) -> bytes:
    descriptor = _open_private_secret_file_at(
        directory_fd,
        name,
        minimum_bytes=1,
        maximum_bytes=1025,
    )
    try:
        content = bytearray()
        while len(content) <= 1025:
            block = os.read(descriptor, min(65_536, 1026 - len(content)))
            if not block:
                break
            content.extend(block)
        after = os.fstat(descriptor)
    except OSError:
        raise ClinicalOperatorError('qualification key could not be read safely') from None
    finally:
        os.close(descriptor)
    if not content or len(content) > 1025 or after.st_size != len(content):
        raise ClinicalOperatorError('qualification key changed while it was read')
    try:
        return decode_firecracker_qualification_key(bytes(content))
    except ValueError:
        raise ClinicalOperatorError(
            'qualification key must contain one trimmed ASCII-hex value encoding 32 to 512 bytes'
        ) from None


def _open_private_secret_file(path: Path, *, minimum_bytes: int, maximum_bytes: int) -> int:
    try:
        before = path.lstat()
    except OSError:
        raise ClinicalOperatorError('required operator secret file is unavailable') from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or not minimum_bytes <= before.st_size <= maximum_bytes
    ):
        raise ClinicalOperatorError('required operator secret file has unsafe metadata')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    after = os.fstat(descriptor)
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
        os.close(descriptor)
        raise ClinicalOperatorError('required operator secret file changed while it was opened')
    return descriptor


def _read_private_secret(path: Path, *, exact_bytes: int | None = None) -> bytes:
    maximum = exact_bytes if exact_bytes is not None else _MAX_SECRET_BYTES
    content = _read_stable_regular_file(path, maximum_bytes=maximum, private=True)
    if exact_bytes is not None:
        if len(content) != exact_bytes:
            raise ClinicalOperatorError('fixed-size operator secret has an invalid length')
    elif not 32 <= len(content) <= _MAX_SECRET_BYTES:
        raise ClinicalOperatorError('operator authentication secret has an invalid length')
    return content


def _read_private_qualification_key(path: Path) -> bytes:
    content = _read_stable_regular_file(path, maximum_bytes=1025, private=True)
    try:
        return decode_firecracker_qualification_key(content)
    except ValueError:
        raise ClinicalOperatorError(
            'qualification key must contain one trimmed ASCII-hex value encoding 32 to 512 bytes'
        ) from None


def _require_existing_private_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        raise ClinicalOperatorError(f'{label} must already exist') from None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ClinicalOperatorError(f'{label} must be an owned private mode-0600 file')


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ClinicalOperatorError(f'{label} must be a lowercase SHA-256 digest')


def current_operator_source_sha256() -> str:
    """Helper for deployment-manifest generation; not an authorization source."""

    return _module_sha256(clinical_operator_module)


__all__ = [
    'CANONICAL_CLINICAL_OPERATOR_ID',
    'CANONICAL_CLINICAL_OPERATOR_MANIFEST_SCHEMA_VERSION',
    'CANONICAL_CLINICAL_OPERATOR_VERSION',
    'STRICT_CLINICAL_V02_LOADER_ID',
    'STRICT_CLINICAL_V02_LOADER_VERSION',
    'CanonicalClinicalOperatorManifest',
    'ClinicalOperatorDryRunReport',
    'ClinicalOperatorError',
    'ClinicalOperatorRecoverySecretMaterial',
    'ClinicalOperatorValidatedInputs',
    'PinnedClinicalProductionRunV02Loader',
    'current_operator_source_sha256',
    'dry_run_report',
    'execute_operator_task',
    'execute_managed_operator_task',
    'expected_system_identity',
    'load_canonical_clinical_operator_manifest',
    'load_and_verify_operator_guest_disks',
    'load_operator_recovery_secret_directory',
    'load_operator_secret_directory',
    'validate_checked_in_executable_pins',
    'validate_managed_gateway_capability_binding',
    'validate_operator_inputs',
    'validate_operator_guest_disk_binding',
    'validate_operator_recovery_secret_material',
    'validate_operator_secret_material',
    'verify_operator_provider_data_control_attestation',
    'validate_side_effect_free_runtime_parity',
]
