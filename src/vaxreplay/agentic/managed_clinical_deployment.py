"""Fixed root-owned Lane A registry, reaper, revoker, and operator composition.

This is deliberately a one-task, one-process deployment boundary.  The executable accepts no
caller-selected database, manifest, secret, verifier, or callback path.  It reads one fixed
root-owned deployment file, verifies every nested configuration by SHA-256, starts the sole
registry authority in its closed state, runs mandatory orphan reconciliation, executes exactly
the pinned task, and joins the registry service before exit.

The code supplies a deployable composition contract; it is not evidence that its systemd unit,
Linux/KVM host, or provider-side revoker has passed a live qualification drill.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_launcher import (
    CanonicalClinicalRecoveryTerminalizer,
    ClinicalLauncherFailureCode,
    ClinicalLauncherResult,
)
from vaxreplay.agentic.clinical_operator import (
    CanonicalClinicalOperatorManifest,
    ClinicalOperatorValidatedInputs,
    PinnedClinicalProductionRunV02Loader,
    execute_managed_operator_task,
    load_operator_recovery_secret_directory,
    validate_checked_in_executable_pins,
    validate_operator_inputs,
    validate_operator_recovery_secret_material,
)
from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalRuntimeKeys,
    firecracker_clinical_startup_reconciliation_request_sha256,
    reconcile_firecracker_clinical_startup_without_execution,
)
from vaxreplay.agentic.firecracker_qualification import (
    load_pinned_firecracker_worker_spec,
)
from vaxreplay.agentic.managed_clinical_ownership import (
    DurableManagedClinicalOwnershipLedger,
    LinuxManagedClinicalHostAdapter,
    ManagedClinicalOwnershipConfig,
    managed_clinical_ownership_config_sha256,
)
from vaxreplay.agentic.managed_clinical_registry import (
    ManagedClinicalRegistryClient,
    ManagedClinicalRegistryConfig,
    ManagedClinicalRegistryService,
    managed_clinical_registry_config_sha256,
)
from vaxreplay.agentic.managed_clinical_startup import (
    ManagedClinicalStartupConfig,
    ManagedClinicalStartupReconciler,
    load_authenticated_managed_cleanup,
    managed_clinical_cleanup_key_id,
    managed_clinical_startup_config_sha256,
    verify_authenticated_managed_cleanup,
)
from vaxreplay.agentic.managed_gateway_capability import (
    RestartVisibleManagedGatewayCapabilityLedger,
)
from vaxreplay.agentic.protocol import agentic_policy_sha256
from vaxreplay.agentic.provider_gateway import (
    SqliteGatewayLedger,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

MANAGED_CLINICAL_DEPLOYMENT_SCHEMA_VERSION = 'vaxreplay.managed-clinical-standalone-deployment.dev-v0.1'
PROVIDER_CAPABILITY_REVOKER_SCHEMA_VERSION = 'vaxreplay.provider-capability-revoker.dev-v0.1'
PROVIDER_CAPABILITY_REVOCATION_REQUEST_SCHEMA_VERSION = 'vaxreplay.provider-capability-revocation-request.dev-v0.1'
PROVIDER_CAPABILITY_REVOCATION_RESPONSE_SCHEMA_VERSION = 'vaxreplay.provider-capability-revocation-response.dev-v0.1'
FIXED_MANAGED_CLINICAL_DEPLOYMENT_PATH = Path('/etc/vaxreplay/lane-a-managed/deployment.json')

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
_MAX_CONFIG_BYTES = 8 * 1024 * 1024
_MAX_SECRET_BYTES = 512
_MAX_REVOKER_RESPONSE_BYTES = 64 * 1024
_MANAGED_SECRET_FILE_NAMES = frozenset({'ownership.key', 'startup-cleanup.key'})
_REVOCATION_IDEMPOTENCY_DOMAIN = b'vaxreplay.provider-capability-revocation.dev-v0.1\x00'
MANAGED_CLINICAL_RECOVERY_RESULT_SCHEMA_VERSION = 'vaxreplay.managed-clinical-recovery-result.dev-v0.1'


class ManagedClinicalDeploymentError(RuntimeError):
    """The fixed managed deployment was unsafe, incomplete, or failed closed."""


class ManagedClinicalRecoveryResult(StrictModel):
    """Content-free receipt for one cleanup-only service invocation."""

    schema_version: Literal['vaxreplay.managed-clinical-recovery-result.dev-v0.1'] = (
        MANAGED_CLINICAL_RECOVERY_RESULT_SCHEMA_VERSION
    )
    deployment_id: str = Field(pattern=_ID_PATTERN)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    reconciliation_request_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    consumed_attempt_snapshot_count: int = Field(ge=0, le=1_000_000)
    terminalized_attempt_count: int = Field(ge=0, le=1_000_000)
    terminalized_unredeemed_claim_count: int = Field(ge=0, le=1_000_000)
    terminalized_redeemed_start_count: int = Field(ge=0, le=1_000_000)
    orphan_cleanup_complete: Literal[True] = True
    consumed_attempts_terminalized: Literal[True] = True
    recovery_only_registry_mode: Literal[True] = True
    harness_or_model_execution_available: Literal[False] = False
    automatic_task_retry: Literal[False] = False
    idempotent_on_terminal_state: Literal[True] = True
    live_linux_kvm_recovery_qualified: Literal[False] = False

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        if (
            self.terminalized_attempt_count
            != (self.terminalized_unredeemed_claim_count + self.terminalized_redeemed_start_count)
            or self.terminalized_attempt_count != self.consumed_attempt_snapshot_count
        ):
            raise ValueError('recovery must terminalize every and only consumed snapshot attempt')
        return self


class ProviderCapabilityRevokerConfig(StrictModel):
    schema_version: Literal['vaxreplay.provider-capability-revoker.dev-v0.1'] = (
        PROVIDER_CAPABILITY_REVOKER_SCHEMA_VERSION
    )
    revoker_id: str = Field(pattern=_ID_PATTERN)
    revoker_version: str = Field(min_length=1, max_length=200)
    revocation_namespace_id: str = Field(pattern=_ID_PATTERN)
    executable_path: str
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    timeout_seconds: float = Field(gt=0, le=60)
    canonical_request_on_stdin: Literal[True] = True
    canonical_response_on_stdout: Literal[True] = True
    ambient_environment_forwarded: Literal[False] = False
    capability_secret_passed_to_child: Literal[False] = False
    deterministic_idempotency_key_required: Literal[True] = True
    revoked_or_absent_ack_required: Literal[True] = True
    local_gateway_tombstone_required_before_hook: Literal[True] = True
    provider_api_key_revocation_claimed: Literal[False] = False
    already_dispatched_request_cancellation_claimed: Literal[False] = False

    @field_validator('executable_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        _require_normalized_absolute_path(value, label='provider revoker executable')
        return value


class ProviderCapabilityRevocationRequest(StrictModel):
    schema_version: Literal['vaxreplay.provider-capability-revocation-request.dev-v0.1'] = (
        PROVIDER_CAPABILITY_REVOCATION_REQUEST_SCHEMA_VERSION
    )
    revoker_id: str = Field(pattern=_ID_PATTERN)
    revocation_namespace_id: str = Field(pattern=_ID_PATTERN)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    capability_id: str = Field(pattern=_SHA256_PATTERN)
    idempotency_key: str = Field(pattern=_SHA256_PATTERN)
    revoke_even_if_already_absent: Literal[True] = True
    no_provider_secret_in_request: Literal[True] = True
    local_gateway_tombstone_already_persisted: Literal[True] = True
    provider_api_key_revocation_claimed: Literal[False] = False
    already_dispatched_request_cancellation_claimed: Literal[False] = False


class ProviderCapabilityRevocationResponse(StrictModel):
    schema_version: Literal['vaxreplay.provider-capability-revocation-response.dev-v0.1'] = (
        PROVIDER_CAPABILITY_REVOCATION_RESPONSE_SCHEMA_VERSION
    )
    revoker_id: str = Field(pattern=_ID_PATTERN)
    revocation_namespace_id: str = Field(pattern=_ID_PATTERN)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    capability_id: str = Field(pattern=_SHA256_PATTERN)
    idempotency_key: str = Field(pattern=_SHA256_PATTERN)
    revoked_or_already_absent: Literal[True] = True
    durable_before_ack: Literal[True] = True
    provider_api_key_revocation_claimed: Literal[False] = False
    already_dispatched_request_cancellation_claimed: Literal[False] = False


class ManagedClinicalStandaloneDeployment(StrictModel):
    """The only non-secret inputs accepted by the standalone production composition."""

    schema_version: Literal['vaxreplay.managed-clinical-standalone-deployment.dev-v0.1'] = (
        MANAGED_CLINICAL_DEPLOYMENT_SCHEMA_VERSION
    )
    deployment_id: str = Field(pattern=_ID_PATTERN)
    deployment_version: str = Field(min_length=1, max_length=200)
    deployment_config_path: Literal['/etc/vaxreplay/lane-a-managed/deployment.json'] = (
        '/etc/vaxreplay/lane-a-managed/deployment.json'
    )
    registry_config_path: str
    registry_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    startup_config_path: str
    startup_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    ownership_config_path: str
    ownership_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    operator_manifest_path: str
    operator_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    operator_secret_root: str
    managed_secret_root: str
    provider_revoker: ProviderCapabilityRevokerConfig | None = None
    service_startup_timeout_seconds: float = Field(gt=0, le=60)
    service_shutdown_timeout_seconds: float = Field(gt=0, le=60)
    fixed_root_owned_configuration_only: Literal[True] = True
    root_owned_mode_0600_secrets_only: Literal[True] = True
    registry_starts_quiesced_on_every_process_start: Literal[True] = True
    reconciliation_receipt_required_before_mutation: Literal[True] = True
    durable_local_gateway_revocation_tombstone_required: Literal[True] = True
    exact_one_operator_task_per_process: Literal[True] = True
    automatic_task_retry: Literal[False] = False
    caller_selected_paths_allowed: Literal[False] = False
    one_host_authority: Literal[True] = True
    cross_host_consensus_claimed: Literal[False] = False
    live_linux_kvm_deployment_claimed: Literal[False] = False

    @field_validator(
        'registry_config_path',
        'startup_config_path',
        'ownership_config_path',
        'operator_manifest_path',
        'operator_secret_root',
        'managed_secret_root',
    )
    @classmethod
    def validate_path(cls, value: str) -> str:
        _require_normalized_absolute_path(value, label='managed deployment input')
        return value

    @model_validator(mode='after')
    def validate_distinct_paths(self) -> Self:
        files = {
            self.registry_config_path,
            self.startup_config_path,
            self.ownership_config_path,
            self.operator_manifest_path,
        }
        if len(files) != 4:
            raise ValueError('managed deployment configuration files must be distinct')
        if self.operator_secret_root == self.managed_secret_root:
            raise ValueError('operator and managed authority secrets require distinct directories')
        return self


@dataclass(frozen=True, slots=True)
class ManagedClinicalDeploymentSecrets:
    startup_cleanup_key: bytes
    ownership_key: bytes


@dataclass(frozen=True, slots=True)
class LoadedManagedClinicalDeployment:
    deployment: ManagedClinicalStandaloneDeployment
    registry_config: ManagedClinicalRegistryConfig
    startup_config: ManagedClinicalStartupConfig
    ownership_config: ManagedClinicalOwnershipConfig
    manifest: CanonicalClinicalOperatorManifest
    manifest_sha256: str
    secrets: ManagedClinicalDeploymentSecrets


class PinnedProviderCapabilityRevoker:
    """Invoke one optional hash-pinned, idempotent external cleanup program.

    The mandatory restart-visible security boundary is the SQLite gateway admission tombstone.
    This hook runs only after that tombstone.  It receives a public capability identifier and may
    clean a separate volatile/provider-side namespace, but its acknowledgement explicitly does
    not claim API-key revocation or cancellation of an already dispatched request.
    """

    def __init__(
        self,
        config: ProviderCapabilityRevokerConfig,
        *,
        registry_authority_id: str,
        required_uid: int = 0,
    ) -> None:
        self.config = config
        self.registry_authority_id = registry_authority_id
        self._required_uid = required_uid
        descriptor = _open_trusted_file(
            Path(config.executable_path),
            required_uid=required_uid,
            exact_mode=None,
            maximum_bytes=64 * 1024 * 1024,
            executable=True,
        )
        try:
            _require_descriptor_sha256(
                descriptor,
                expected_sha256=config.executable_sha256,
                label='provider revoker executable',
            )
        finally:
            os.close(descriptor)

    def __call__(self, capability_id: str) -> None:
        if len(capability_id) != 64 or any(character not in '0123456789abcdef' for character in capability_id):
            raise ManagedClinicalDeploymentError('provider revocation capability ID is invalid')
        idempotency_key = hashlib.sha256(
            _REVOCATION_IDEMPOTENCY_DOMAIN
            + self.config.revocation_namespace_id.encode('utf-8')
            + b'\x00'
            + capability_id.encode('ascii')
        ).hexdigest()
        request = ProviderCapabilityRevocationRequest(
            revoker_id=self.config.revoker_id,
            revocation_namespace_id=self.config.revocation_namespace_id,
            registry_authority_id=self.registry_authority_id,
            capability_id=capability_id,
            idempotency_key=idempotency_key,
        )
        descriptor = _open_trusted_file(
            Path(self.config.executable_path),
            required_uid=self._required_uid,
            exact_mode=None,
            maximum_bytes=64 * 1024 * 1024,
            executable=True,
        )
        try:
            _require_descriptor_sha256(
                descriptor,
                expected_sha256=self.config.executable_sha256,
                label='provider revoker executable',
            )
            descriptor_path, descriptor_backed = _descriptor_execution_path(
                descriptor,
                original_path=Path(self.config.executable_path),
            )
            try:
                completed = subprocess.run(
                    (descriptor_path,),
                    executable=descriptor_path,
                    input=canonical_json_bytes(request),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    cwd='/',
                    env={},
                    pass_fds=(descriptor,),
                    close_fds=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                raise ManagedClinicalDeploymentError('provider capability revoker did not complete safely') from None
            if not descriptor_backed:
                _require_path_matches_descriptor(
                    Path(self.config.executable_path),
                    descriptor=descriptor,
                )
        finally:
            os.close(descriptor)
        if completed.returncode != 0 or not completed.stdout or len(completed.stdout) > _MAX_REVOKER_RESPONSE_BYTES:
            raise ManagedClinicalDeploymentError('provider capability revoker rejected the request')
        try:
            response = ProviderCapabilityRevocationResponse.model_validate_json(completed.stdout)
        except ValueError:
            raise ManagedClinicalDeploymentError('provider capability revoker returned an invalid response') from None
        if canonical_json_bytes(response) != completed.stdout or (
            response.revoker_id,
            response.revocation_namespace_id,
            response.registry_authority_id,
            response.capability_id,
            response.idempotency_key,
        ) != (
            request.revoker_id,
            request.revocation_namespace_id,
            request.registry_authority_id,
            request.capability_id,
            request.idempotency_key,
        ):
            raise ManagedClinicalDeploymentError('provider capability revoker response differs from the exact request')


def load_fixed_managed_clinical_deployment(
    path: Path = FIXED_MANAGED_CLINICAL_DEPLOYMENT_PATH,
    *,
    required_uid: int = 0,
    require_fixed_path: bool = True,
) -> LoadedManagedClinicalDeployment:
    """Load every deployment-owned config and secret through no-follow descriptors."""

    if require_fixed_path and path != FIXED_MANAGED_CLINICAL_DEPLOYMENT_PATH:
        raise ManagedClinicalDeploymentError('managed deployment path is not the compiled-in path')
    deployment = _load_trusted_model(
        path,
        model=ManagedClinicalStandaloneDeployment,
        expected_sha256=None,
        required_uid=required_uid,
    )
    registry_config = _load_trusted_model(
        Path(deployment.registry_config_path),
        model=ManagedClinicalRegistryConfig,
        expected_sha256=deployment.registry_config_sha256,
        required_uid=required_uid,
    )
    startup_config = _load_trusted_model(
        Path(deployment.startup_config_path),
        model=ManagedClinicalStartupConfig,
        expected_sha256=deployment.startup_config_sha256,
        required_uid=required_uid,
    )
    ownership_config = _load_trusted_model(
        Path(deployment.ownership_config_path),
        model=ManagedClinicalOwnershipConfig,
        expected_sha256=deployment.ownership_config_sha256,
        required_uid=required_uid,
    )
    manifest_bytes = _read_trusted_file(
        Path(deployment.operator_manifest_path),
        required_uid=required_uid,
        exact_mode=0o600,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    if not hmac.compare_digest(
        hashlib.sha256(manifest_bytes).hexdigest(),
        deployment.operator_manifest_sha256,
    ):
        raise ManagedClinicalDeploymentError('operator manifest differs from its deployment pin')
    try:
        manifest = CanonicalClinicalOperatorManifest.model_validate_json(manifest_bytes)
    except ValueError:
        raise ManagedClinicalDeploymentError('operator manifest has an invalid strict schema') from None
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ManagedClinicalDeploymentError('operator manifest must use exact canonical JSON')
    validate_checked_in_executable_pins(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    secrets = _load_managed_secrets(
        Path(deployment.managed_secret_root),
        required_uid=required_uid,
    )
    loaded = LoadedManagedClinicalDeployment(
        deployment=deployment,
        registry_config=registry_config,
        startup_config=startup_config,
        ownership_config=ownership_config,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        secrets=secrets,
    )
    validate_managed_clinical_deployment_binding(loaded)
    _audit_trusted_directory(
        Path(deployment.operator_secret_root),
        required_uid=required_uid,
        exact_mode=0o700,
    )
    return loaded


def validate_managed_clinical_deployment_binding(
    loaded: LoadedManagedClinicalDeployment,
) -> None:
    """Cross-bind the standalone, registry, reaper, ownership, and operator configs."""

    deployment = loaded.deployment
    registry = loaded.registry_config
    startup = loaded.startup_config
    ownership = loaded.ownership_config
    manifest = loaded.manifest
    if (
        managed_clinical_registry_config_sha256(registry) != deployment.registry_config_sha256
        or managed_clinical_startup_config_sha256(startup) != deployment.startup_config_sha256
        or managed_clinical_ownership_config_sha256(ownership) != deployment.ownership_config_sha256
    ):
        raise ManagedClinicalDeploymentError('managed deployment config digest binding failed')
    if manifest.registry_execution_mode != 'managed-unix-authority' or (
        manifest.managed_registry_config_sha256,
        manifest.managed_startup_config_sha256,
        manifest.managed_ownership_config_sha256,
    ) != (
        deployment.registry_config_sha256,
        deployment.startup_config_sha256,
        deployment.ownership_config_sha256,
    ):
        raise ManagedClinicalDeploymentError('operator manifest is not bound to this managed authority composition')
    authority_ids = {
        registry.registry_authority_id,
        startup.registry_authority_id,
        ownership.registry_authority_id,
        manifest.deployment.registry_authority_id,
    }
    if len(authority_ids) != 1:
        raise ManagedClinicalDeploymentError('managed components name different authorities')
    if (
        registry.database_path != manifest.registry_path
        or registry.production_evidence_root != manifest.evidence_root
        or registry.canonical_launcher_id != manifest.deployment.canonical_launcher_id
        or registry.canonical_launcher_executable_sha256 != manifest.deployment.canonical_launcher_executable_sha256
        or registry.startup_config_sha256 != deployment.startup_config_sha256
        or startup.runtime_config_sha256 != manifest.deployment.runtime_config_sha256
        or startup.worker_spec_sha256 != manifest.expected_worker_spec_sha256
        or ownership.worker_spec_sha256 != manifest.expected_worker_spec_sha256
        or ownership.jail_namespace_root != startup.jail_root
        or ownership.cgroup_namespace_root != startup.cgroup_root
        or startup.vsock_root != startup.jail_root
    ):
        raise ManagedClinicalDeploymentError(
            'managed paths, worker, runtime, launcher, or evidence verifier pins differ'
        )
    cleanup_key = loaded.secrets.startup_cleanup_key
    ownership_key = loaded.secrets.ownership_key
    if (
        not hmac.compare_digest(cleanup_key, ownership_key)
        or (
            managed_clinical_cleanup_key_id(cleanup_key),
            managed_clinical_cleanup_key_id(ownership_key),
        )
        != (
            startup.cleanup_receipt_key_id,
            ownership.ownership_key_id,
        )
        or registry.startup_cleanup_receipt_key_id != startup.cleanup_receipt_key_id
    ):
        raise ManagedClinicalDeploymentError('managed cleanup and ownership keys differ from their deployment IDs')


def execute_fixed_managed_clinical_task(
    path: Path = FIXED_MANAGED_CLINICAL_DEPLOYMENT_PATH,
) -> ClinicalLauncherResult:
    """Run exactly one pinned task through the closed-registry/reaper/operator transaction."""

    if sys.platform != 'linux' or os.geteuid() != 0:
        raise ManagedClinicalDeploymentError('fixed managed clinical execution requires effective UID 0 on Linux')
    loaded = load_fixed_managed_clinical_deployment(path)
    inputs: ClinicalOperatorValidatedInputs | None = None
    stop_event = threading.Event()
    ready_event = threading.Event()
    service_done = threading.Event()
    service_errors: list[BaseException] = []
    service_thread: threading.Thread | None = None
    try:
        inputs = validate_operator_inputs(
            loaded.manifest,
            manifest_sha256=loaded.manifest_sha256,
            secret_root=Path(loaded.deployment.operator_secret_root),
        )
        runtime_keys = FirecrackerClinicalRuntimeKeys(
            workspace_receipt_key=inputs.secrets.workspace_receipt_key,
            worker_attestation_key=inputs.secrets.worker_attestation_key,
            gateway_receipt_key=inputs.secrets.gateway_receipt_key,
            guest_rpc_receipt_key=inputs.secrets.guest_rpc_receipt_key,
            clinical_guest_bootstrap_receipt_key=inputs.secrets.bootstrap_receipt_key,
            production_receipt_key=inputs.secrets.production_receipt_key,
        )
        evidence_verifier = PinnedClinicalProductionRunV02Loader(
            workspace=inputs.workspace,
            manifest=loaded.manifest,
            worker_spec=inputs.worker_spec,
            keys=runtime_keys,
        )
        registry_service = ManagedClinicalRegistryService(
            config=loaded.registry_config,
            workspace_receipt_keys_by_id={
                loaded.manifest.expected_workspace_receipt_key_id: (inputs.secrets.workspace_receipt_key)
            },
            evidence_reauthenticator=evidence_verifier,
            startup_config=loaded.startup_config,
            startup_cleanup_receipt_key=loaded.secrets.startup_cleanup_key,
        )

        def serve() -> None:
            try:
                registry_service.serve_until(
                    stop_event=stop_event,
                    ready_event=ready_event,
                )
            except BaseException as error:
                service_errors.append(error)
            finally:
                service_done.set()

        service_thread = threading.Thread(
            target=serve,
            name='vaxreplay-managed-registry',
            daemon=False,
        )
        service_thread.start()
        if not ready_event.wait(loaded.deployment.service_startup_timeout_seconds):
            if service_done.is_set() and service_errors:
                raise ManagedClinicalDeploymentError(
                    'managed registry failed before publishing readiness'
                ) from service_errors[0]
            raise ManagedClinicalDeploymentError('managed registry readiness timed out')
        if service_done.is_set() or service_errors:
            raise ManagedClinicalDeploymentError('managed registry terminated immediately after readiness')

        registry_client = ManagedClinicalRegistryClient(
            loaded.registry_config,
            expected_config_sha256=loaded.deployment.registry_config_sha256,
        )
        ownership = DurableManagedClinicalOwnershipLedger(
            config=loaded.ownership_config,
            ownership_key=loaded.secrets.ownership_key,
        )
        host = LinuxManagedClinicalHostAdapter(
            config=loaded.startup_config,
            ownership=ownership,
            ownership_key=loaded.secrets.ownership_key,
        )
        provider_revoker = (
            None
            if loaded.deployment.provider_revoker is None
            else PinnedProviderCapabilityRevoker(
                loaded.deployment.provider_revoker,
                registry_authority_id=loaded.registry_config.registry_authority_id,
            )
        )
        capabilities = RestartVisibleManagedGatewayCapabilityLedger(
            ownership=ownership,
            ownership_key=loaded.secrets.ownership_key,
            gateway_ledger=SqliteGatewayLedger(Path(loaded.manifest.gateway_ledger_path)),
            expected_model_route_sha256=gateway_model_route_sha256(loaded.manifest.gateway_route),
            after_local_revocation=provider_revoker,
        )
        reconciler = ManagedClinicalStartupReconciler(
            config=loaded.startup_config,
            host=host,
            capabilities=capabilities,
            attempts=registry_client,
            cleanup_receipt_key=loaded.secrets.startup_cleanup_key,
            reconciliation_complete=registry_client.finish_reconciliation,
        )
        return execute_managed_operator_task(
            inputs,
            managed_registry=registry_client,
            startup_reconciler=reconciler,
            managed_ownership=ownership,
        )
    finally:
        stop_event.set()
        shutdown_error: ManagedClinicalDeploymentError | None = None
        shutdown_cause: BaseException | None = None
        if service_thread is not None:
            service_thread.join(loaded.deployment.service_shutdown_timeout_seconds)
            if service_thread.is_alive():
                shutdown_error = ManagedClinicalDeploymentError(
                    'managed registry did not stop before the service deadline'
                )
            elif service_errors:
                shutdown_error = ManagedClinicalDeploymentError(
                    'managed registry service failed during the managed task'
                )
                shutdown_cause = service_errors[0]
        if inputs is not None:
            inputs.secrets.close()
        if shutdown_error is not None:
            raise shutdown_error from shutdown_cause


def execute_fixed_managed_clinical_recovery(
    path: Path = FIXED_MANAGED_CLINICAL_DEPLOYMENT_PATH,
) -> ManagedClinicalRecoveryResult:
    """Clean orphan state and terminalize consumed attempts without a launch surface.

    The recovery process reads the same fixed deployment, manifest, secrets, registry, ownership
    ledger, startup-receipt verifier, gateway ledger, and optional revoker as the operator.  It
    deliberately does not construct ``FirecrackerSupervisor``, a provider adapter/gateway, an
    executable harness runtime, or an evidence-success loader.  Its registry service rejects
    reserve, claim, redeem, and success-publication operations even after cleanup completes.
    """

    if sys.platform != 'linux' or os.geteuid() != 0:
        raise ManagedClinicalDeploymentError('fixed managed clinical recovery requires effective UID 0 on Linux')
    loaded = load_fixed_managed_clinical_deployment(path)
    operator_secrets = None
    stop_event = threading.Event()
    ready_event = threading.Event()
    service_done = threading.Event()
    service_errors: list[BaseException] = []
    service_thread: threading.Thread | None = None
    try:
        operator_secrets = load_operator_recovery_secret_directory(Path(loaded.deployment.operator_secret_root))
        validate_operator_recovery_secret_material(loaded.manifest, operator_secrets)
        registry_service = ManagedClinicalRegistryService(
            config=loaded.registry_config,
            workspace_receipt_keys_by_id={},
            evidence_reauthenticator=None,
            startup_config=loaded.startup_config,
            startup_cleanup_receipt_key=loaded.secrets.startup_cleanup_key,
            recovery_only=True,
        )

        def serve() -> None:
            try:
                registry_service.serve_until(
                    stop_event=stop_event,
                    ready_event=ready_event,
                )
            except BaseException as error:
                service_errors.append(error)
            finally:
                service_done.set()

        service_thread = threading.Thread(
            target=serve,
            name='vaxreplay-managed-registry-recovery',
            daemon=False,
        )
        service_thread.start()
        if not ready_event.wait(loaded.deployment.service_startup_timeout_seconds):
            if service_done.is_set() and service_errors:
                raise ManagedClinicalDeploymentError(
                    'managed recovery registry failed before publishing readiness'
                ) from service_errors[0]
            raise ManagedClinicalDeploymentError('managed recovery registry readiness timed out')
        if service_done.is_set() or service_errors:
            raise ManagedClinicalDeploymentError('managed recovery registry terminated immediately after readiness')

        registry_client = ManagedClinicalRegistryClient(
            loaded.registry_config,
            expected_config_sha256=loaded.deployment.registry_config_sha256,
        )
        attempt_snapshot = registry_client.begin_reconciliation()
        consumed_snapshot = tuple(item for item in attempt_snapshot if item.state == 'launched')
        ownership = DurableManagedClinicalOwnershipLedger(
            config=loaded.ownership_config,
            ownership_key=loaded.secrets.ownership_key,
        )
        host = LinuxManagedClinicalHostAdapter(
            config=loaded.startup_config,
            ownership=ownership,
            ownership_key=loaded.secrets.ownership_key,
        )
        provider_revoker = (
            None
            if loaded.deployment.provider_revoker is None
            else PinnedProviderCapabilityRevoker(
                loaded.deployment.provider_revoker,
                registry_authority_id=(loaded.registry_config.registry_authority_id),
            )
        )
        capabilities = RestartVisibleManagedGatewayCapabilityLedger(
            ownership=ownership,
            ownership_key=loaded.secrets.ownership_key,
            gateway_ledger=SqliteGatewayLedger(Path(loaded.manifest.gateway_ledger_path)),
            expected_model_route_sha256=gateway_model_route_sha256(loaded.manifest.gateway_route),
            after_local_revocation=provider_revoker,
        )
        reconciler = ManagedClinicalStartupReconciler(
            config=loaded.startup_config,
            host=host,
            capabilities=capabilities,
            attempts=registry_client,
            cleanup_receipt_key=loaded.secrets.startup_cleanup_key,
            reconciliation_complete=registry_client.finish_reconciliation,
        )
        worker_spec, _ = load_pinned_firecracker_worker_spec(
            Path(loaded.manifest.worker_spec_path),
            expected_worker_spec_sha256=(loaded.manifest.expected_worker_spec_sha256),
        )
        report = reconcile_firecracker_clinical_startup_without_execution(
            config=loaded.manifest.runtime_config,
            execution_policy_sha256=agentic_policy_sha256(loaded.manifest.execution_policy),
            worker_spec=worker_spec,
            gateway_policy_sha256=authenticated_gateway_policy_sha256(loaded.manifest.gateway_policy),
            gateway_route_sha256=gateway_model_route_sha256(loaded.manifest.gateway_route),
            guest_rpc_policy=loaded.manifest.guest_rpc_policy,
            bootstrap_receipt_key=operator_secrets.bootstrap_receipt_key,
            bootstrap_trust_anchor=loaded.manifest.bootstrap_trust_anchor,
            evidence_root=Path(loaded.manifest.evidence_root),
            reconciler=reconciler,
        )
        authenticated = reconciler.last_authenticated_receipt
        if authenticated is None:
            raise ManagedClinicalDeploymentError('managed recovery reaper returned no authenticated cleanup receipt')
        persisted = load_authenticated_managed_cleanup(
            Path(authenticated.persisted_path),
            expected_root=Path(loaded.startup_config.receipt_root),
        )
        request_sha256 = firecracker_clinical_startup_reconciliation_request_sha256(report.request)
        receipt = verify_authenticated_managed_cleanup(
            persisted,
            key=loaded.secrets.startup_cleanup_key,
            expected_key_id=loaded.startup_config.cleanup_receipt_key_id,
            expected_config_sha256=managed_clinical_startup_config_sha256(loaded.startup_config),
            expected_request_sha256=request_sha256,
        )
        if persisted != authenticated or receipt != report.cleanup_receipt:
            raise ManagedClinicalDeploymentError(
                'managed recovery cleanup differs from its persisted authenticated receipt'
            )

        terminalizer = CanonicalClinicalRecoveryTerminalizer(
            registry=registry_client,
            deployment=loaded.manifest.deployment,
            failure_receipt_key=operator_secrets.launcher_failure_receipt_key,
        )
        terminalized = tuple(
            failure
            for reservation_sha256 in sorted({item.reservation_sha256 for item in consumed_snapshot})
            for failure in terminalizer.reconcile_consumed_tasks(reservation_sha256=reservation_sha256)
        )
        if {item.run_id for item in consumed_snapshot} != {item.launch.run_id for item in terminalized}:
            raise ManagedClinicalDeploymentError(
                'managed recovery did not terminalize the exact consumed attempt snapshot'
            )
        unredeemed_count = sum(
            item.failure_code == ClinicalLauncherFailureCode.RECONCILED_UNREDEEMED_CLAIM for item in terminalized
        )
        redeemed_count = sum(
            item.failure_code == ClinicalLauncherFailureCode.RECONCILED_REDEEMED_START for item in terminalized
        )
        return ManagedClinicalRecoveryResult(
            deployment_id=loaded.deployment.deployment_id,
            registry_authority_id=loaded.registry_config.registry_authority_id,
            reconciliation_request_sha256=request_sha256,
            attempt_inventory_sha256=authenticated.attempt_inventory_sha256,
            consumed_attempt_snapshot_count=len(consumed_snapshot),
            terminalized_attempt_count=len(terminalized),
            terminalized_unredeemed_claim_count=unredeemed_count,
            terminalized_redeemed_start_count=redeemed_count,
        )
    finally:
        stop_event.set()
        shutdown_error: ManagedClinicalDeploymentError | None = None
        shutdown_cause: BaseException | None = None
        if service_thread is not None:
            service_thread.join(loaded.deployment.service_shutdown_timeout_seconds)
            if service_thread.is_alive():
                shutdown_error = ManagedClinicalDeploymentError(
                    'managed recovery registry did not stop before the service deadline'
                )
            elif service_errors:
                shutdown_error = ManagedClinicalDeploymentError(
                    'managed recovery registry service failed during cleanup'
                )
                shutdown_cause = service_errors[0]
        if shutdown_error is not None:
            raise shutdown_error from shutdown_cause


def _load_managed_secrets(
    root: Path,
    *,
    required_uid: int,
) -> ManagedClinicalDeploymentSecrets:
    directory_fd = _open_trusted_directory(
        root,
        required_uid=required_uid,
        exact_mode=0o700,
    )
    try:
        if set(os.listdir(directory_fd)) != _MANAGED_SECRET_FILE_NAMES:
            raise ManagedClinicalDeploymentError('managed secret directory has an unexpected inventory')
        cleanup = _read_secret_at(
            directory_fd,
            'startup-cleanup.key',
            required_uid=required_uid,
        )
        ownership = _read_secret_at(
            directory_fd,
            'ownership.key',
            required_uid=required_uid,
        )
    finally:
        os.close(directory_fd)
    return ManagedClinicalDeploymentSecrets(
        startup_cleanup_key=cleanup,
        ownership_key=ownership,
    )


def _load_trusted_model[ModelT: StrictModel](
    path: Path,
    *,
    model: type[ModelT],
    expected_sha256: str | None,
    required_uid: int,
) -> ModelT:
    content = _read_trusted_file(
        path,
        required_uid=required_uid,
        exact_mode=0o600,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    if expected_sha256 is not None and not hmac.compare_digest(
        hashlib.sha256(content).hexdigest(),
        expected_sha256,
    ):
        raise ManagedClinicalDeploymentError('managed configuration differs from its SHA-256 pin')
    try:
        value = model.model_validate_json(content)
    except ValueError:
        raise ManagedClinicalDeploymentError('managed configuration has an invalid strict schema') from None
    if canonical_json_bytes(value) != content:
        raise ManagedClinicalDeploymentError('managed configuration must use exact canonical JSON')
    return value


def _read_trusted_file(
    path: Path,
    *,
    required_uid: int,
    exact_mode: int | None,
    maximum_bytes: int,
) -> bytes:
    descriptor = _open_trusted_file(
        path,
        required_uid=required_uid,
        exact_mode=exact_mode,
        maximum_bytes=maximum_bytes,
        executable=False,
    )
    try:
        content = bytearray()
        while len(content) <= maximum_bytes:
            block = os.read(
                descriptor,
                min(65_536, maximum_bytes + 1 - len(content)),
            )
            if not block:
                break
            content.extend(block)
        after = os.fstat(descriptor)
    except OSError:
        raise ManagedClinicalDeploymentError('trusted deployment file could not be read') from None
    finally:
        os.close(descriptor)
    if not content or len(content) > maximum_bytes or after.st_size != len(content):
        raise ManagedClinicalDeploymentError('trusted deployment file changed while it was read')
    return bytes(content)


def _open_trusted_file(
    path: Path,
    *,
    required_uid: int,
    exact_mode: int | None,
    maximum_bytes: int,
    executable: bool,
) -> int:
    parent_fd, name = _open_trusted_parent(path, required_uid=required_uid)
    try:
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError:
            raise ManagedClinicalDeploymentError('trusted deployment file is unavailable') from None
    finally:
        os.close(parent_fd)
    metadata = os.fstat(descriptor)
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != required_uid
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > maximum_bytes
        or (exact_mode is not None and mode != exact_mode)
        or (exact_mode is None and bool(mode & 0o022))
        or (executable and not bool(mode & stat.S_IXUSR))
    ):
        os.close(descriptor)
        raise ManagedClinicalDeploymentError('trusted deployment file has unsafe metadata')
    return descriptor


def _open_trusted_directory(
    path: Path,
    *,
    required_uid: int,
    exact_mode: int,
) -> int:
    parent_fd, name = _open_trusted_parent(path, required_uid=required_uid)
    try:
        flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError:
            raise ManagedClinicalDeploymentError('trusted deployment directory is unavailable') from None
    finally:
        os.close(parent_fd)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != required_uid
        or stat.S_IMODE(metadata.st_mode) != exact_mode
    ):
        os.close(descriptor)
        raise ManagedClinicalDeploymentError('trusted deployment directory has unsafe metadata')
    return descriptor


def _audit_trusted_directory(path: Path, *, required_uid: int, exact_mode: int) -> None:
    descriptor = _open_trusted_directory(
        path,
        required_uid=required_uid,
        exact_mode=exact_mode,
    )
    os.close(descriptor)


def _open_trusted_parent(path: Path, *, required_uid: int) -> tuple[int, str]:
    value = _require_normalized_absolute_path(str(path), label='trusted deployment path')
    parts = PurePosixPath(value).parts
    if len(parts) < 2:
        raise ManagedClinicalDeploymentError('trusted deployment path cannot name the filesystem root')
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open('/', flags)
    try:
        _require_trusted_directory_metadata(os.fstat(descriptor), required_uid=required_uid)
        for component in parts[1:-1]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError:
                raise ManagedClinicalDeploymentError('trusted deployment path has an unsafe parent') from None
            os.close(descriptor)
            descriptor = child
            _require_trusted_directory_metadata(os.fstat(descriptor), required_uid=required_uid)
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _require_trusted_directory_metadata(metadata: os.stat_result, *, required_uid: int) -> None:
    trusted_uids = {0, required_uid}
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid not in trusted_uids
        or bool(stat.S_IMODE(metadata.st_mode) & 0o022)
    ):
        raise ManagedClinicalDeploymentError('trusted deployment path has an untrusted or writable parent')


def _read_secret_at(directory_fd: int, name: str, *, required_uid: int) -> bytes:
    if '/' in name or name in {'', '.', '..'}:
        raise ManagedClinicalDeploymentError('managed secret name is invalid')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError:
        raise ManagedClinicalDeploymentError('managed secret is unavailable') from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != required_uid
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not 32 <= metadata.st_size <= _MAX_SECRET_BYTES
        ):
            raise ManagedClinicalDeploymentError('managed secret has unsafe metadata')
        content = bytearray()
        while len(content) <= _MAX_SECRET_BYTES:
            block = os.read(descriptor, min(65_536, _MAX_SECRET_BYTES + 1 - len(content)))
            if not block:
                break
            content.extend(block)
        after = os.fstat(descriptor)
    except OSError:
        raise ManagedClinicalDeploymentError('managed secret could not be read safely') from None
    finally:
        os.close(descriptor)
    if not 32 <= len(content) <= _MAX_SECRET_BYTES or after.st_size != len(content):
        raise ManagedClinicalDeploymentError('managed secret changed while it was read')
    return bytes(content)


def _require_descriptor_sha256(descriptor: int, *, expected_sha256: str, label: str) -> None:
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError:
        raise ManagedClinicalDeploymentError(f'{label} could not be measured') from None
    if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
        raise ManagedClinicalDeploymentError(f'{label} differs from its deployment pin')


def _descriptor_execution_path(
    descriptor: int,
    *,
    original_path: Path,
) -> tuple[str, bool]:
    if sys.platform == 'linux' and Path('/proc/self/fd').is_dir():
        return f'/proc/self/fd/{descriptor}', True
    # The standalone executor itself rejects non-Linux hosts.  This stable-path fallback exists
    # only so the revoker contract can be unit-tested on macOS; it rechecks the exact inode after
    # execution and is never the production path.
    return str(original_path), False


def _require_path_matches_descriptor(path: Path, *, descriptor: int) -> None:
    opened = os.fstat(descriptor)
    try:
        current = path.lstat()
    except OSError:
        raise ManagedClinicalDeploymentError(
            'provider revoker path changed during non-Linux contract testing'
        ) from None
    if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
        raise ManagedClinicalDeploymentError('provider revoker path changed during non-Linux contract testing')


def _require_normalized_absolute_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or str(path) != value or value == '/':
        raise ValueError(f'{label} must be a normalized absolute non-root path')
    return value


__all__ = [
    'FIXED_MANAGED_CLINICAL_DEPLOYMENT_PATH',
    'MANAGED_CLINICAL_RECOVERY_RESULT_SCHEMA_VERSION',
    'LoadedManagedClinicalDeployment',
    'ManagedClinicalDeploymentError',
    'ManagedClinicalRecoveryResult',
    'ManagedClinicalStandaloneDeployment',
    'PinnedProviderCapabilityRevoker',
    'ProviderCapabilityRevocationRequest',
    'ProviderCapabilityRevocationResponse',
    'ProviderCapabilityRevokerConfig',
    'execute_fixed_managed_clinical_recovery',
    'execute_fixed_managed_clinical_task',
    'load_fixed_managed_clinical_deployment',
    'validate_managed_clinical_deployment_binding',
]
