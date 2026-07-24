"""Signed development evidence for the managed Lane A real-Firecracker seam.

This artifact is deliberately narrower than production qualification.  It proves that one exact
clinical task passed through the managed registry, startup reconciliation, durable ownership
ledger, real Firecracker worker, authenticated guest/provider RPC, normal cleanup, and terminal
registry reauthentication.  The provider is a public deterministic subprocess fixture; no learned
model or external provider is involved.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_execution_bridge import (
    load_clinical_agentic_workspace,
)
from vaxreplay.agentic.clinical_guest_bootstrap import (
    AuthenticatedClinicalGuestBootstrap,
)
from vaxreplay.agentic.clinical_operator import (
    CanonicalClinicalOperatorManifest,
    load_and_verify_operator_guest_disks,
    validate_checked_in_executable_pins,
    validate_side_effect_free_runtime_parity,
)
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionReservation,
    ClinicalProductionTaskRecord,
    ClinicalProductionTerminalCode,
    SqliteClinicalProductionRegistry,
    clinical_production_reservation_sha256,
    clinical_production_start_redemption_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.clinical_production_run_v02 import (
    AuthenticatedClinicalProductionRunV02,
    load_clinical_production_run_v02,
)
from vaxreplay.agentic.firecracker import (
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerWorkerSpec,
    firecracker_model_sha256,
)
from vaxreplay.agentic.firecracker_clinical_runtime import (
    firecracker_clinical_runtime_config_sha256,
    firecracker_clinical_startup_reconciliation_request_sha256,
)
from vaxreplay.agentic.firecracker_qualification import (
    load_firecracker_qualification,
    load_pinned_firecracker_worker_spec,
)
from vaxreplay.agentic.firecracker_qualification_probe import (
    firecracker_live_collector_key_id,
)
from vaxreplay.agentic.firecracker_qualification_runtime_closure import (
    LoadedQualificationDriverRuntimeClosure,
    verify_qualification_driver_runtime_closure,
)
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcTerminalStatus,
    guest_rpc_policy_sha256,
)
from vaxreplay.agentic.managed_clinical_deployment import (
    LoadedManagedClinicalDeployment,
    ManagedClinicalDeploymentSecrets,
    ManagedClinicalStandaloneDeployment,
    validate_managed_clinical_deployment_binding,
)
from vaxreplay.agentic.managed_clinical_ownership import (
    AuthenticatedManagedClinicalOwnership,
    DurableManagedClinicalOwnershipLedger,
    LinuxManagedClinicalHostAdapter,
    ManagedClinicalOwnershipConfig,
    authenticated_managed_clinical_ownership_sha256,
    managed_clinical_ownership_config_sha256,
)
from vaxreplay.agentic.managed_clinical_registry import (
    AuthenticatedManagedClinicalRegistryAudit,
    ManagedBeginReconciliationRequest,
    ManagedClaimRequest,
    ManagedClinicalRegistryConfig,
    ManagedFinishReconciliationRequest,
    ManagedRecordRunRequest,
    load_authenticated_managed_registry_audit_chain,
    managed_clinical_registry_config_sha256,
)
from vaxreplay.agentic.managed_clinical_startup import (
    AuthenticatedManagedClinicalStartupCleanup,
    ManagedClinicalAttemptInventoryRecord,
    ManagedClinicalStartupConfig,
    load_authenticated_managed_cleanup,
    managed_clinical_startup_config_sha256,
    verify_authenticated_managed_cleanup,
)
from vaxreplay.agentic.managed_gateway_capability import (
    RestartVisibleManagedGatewayCapabilityLedger,
)
from vaxreplay.agentic.protocol import agentic_policy_sha256
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewaySession,
    GatewayCapabilityRevocation,
    GatewayCapabilityRevocationReason,
    GatewayLedgerIdentity,
    GatewayTerminalReason,
    SqliteGatewayLedger,
    authenticated_gateway_policy_sha256,
    gateway_capability_binding,
    gateway_model_route_sha256,
)
from vaxreplay.agentic.provider_subprocess import (
    provider_subprocess_behavior_sha256,
    provider_subprocess_spec_sha256,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission

MANAGED_CLINICAL_REAL_KVM_DRILL_SCHEMA_VERSION = 'vaxreplay.managed-clinical-real-kvm-drill.dev-v0.3'
AUTHENTICATED_MANAGED_CLINICAL_REAL_KVM_DRILL_SCHEMA_VERSION = (
    'vaxreplay.authenticated-managed-clinical-real-kvm-drill.dev-v0.3'
)

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_SIGNATURE_PATTERN = r'^[0-9a-f]{128}$'
_SIGNATURE_DOMAIN = b'vaxreplay.managed-clinical-real-kvm-drill.dev-v0.3\x00'
_KEY_ID_DOMAIN = b'vaxreplay.managed-clinical-real-kvm-drill-collector-key-id.dev-v0.3\x00'
_RELEASE_PINS_DOMAIN = b'vaxreplay.managed-clinical-real-kvm-release-pins.dev-v0.3\x00'
_CHALLENGE_DOMAIN = b'vaxreplay.managed-clinical-real-kvm-challenge.dev-v0.3\x00'
_EXPECTED_OWNERSHIP_STATES = (
    'preparing',
    'prepared',
    'start_bound',
    'running',
    'capability_revoked',
    'cleaned',
)
_EXPECTED_STARTUP_PHASES = (
    'reservation_provisioning',
    'managed_operator_startup',
    'retry_denial_reopen',
)
_OBSERVATION_GATE_TIMEOUT_SECONDS = 10


def _model_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _proc_cgroup_path_for(cgroup_path: str) -> str:
    try:
        relative = PurePosixPath(cgroup_path).relative_to('/sys/fs/cgroup')
    except ValueError:
        raise ValueError('managed drill cgroup path is outside the cgroup-v2 mount') from None
    return '/' + relative.as_posix()


def managed_clinical_real_kvm_release_pins_sha256(
    *,
    worker_spec_sha256: str,
    execution_policy_sha256: str,
    guest_rpc_policy_sha256: str,
    guest_config_sha256: str,
    disk_build_receipt_sha256: str,
    qualification_key_id: str,
    qualification_artifact_sha256: str,
    qualification_collector_evidence_sha256: str,
    qualification_probe_manifest_sha256: str,
    qualification_runtime_closure_manifest_sha256: str,
    qualification_runtime_closure_receipt_sha256: str,
    qualification_runtime_closure_sha256: str,
    qualification_collector_public_key_hex: str,
    qualification_collector_key_id: str,
    qualification_verifier_source_sha256: str,
    task_sha256: str,
    provider_child_executable_sha256: str,
    provider_plan_sha256: str,
    collector_entrypoint_sha256: str,
    collector_interpreter_sha256: str,
    collector_runtime_closure_manifest_sha256: str,
    collector_runtime_closure_receipt_sha256: str,
    collector_runtime_closure_sha256: str,
    collector_public_key_hex: str,
    collector_key_id: str,
    launcher_process_executable_sha256: str,
    bootstrap_authorization_key_id: str,
) -> str:
    pins = {
        'worker_spec_sha256': worker_spec_sha256,
        'execution_policy_sha256': execution_policy_sha256,
        'guest_rpc_policy_sha256': guest_rpc_policy_sha256,
        'guest_config_sha256': guest_config_sha256,
        'disk_build_receipt_sha256': disk_build_receipt_sha256,
        'qualification_key_id': qualification_key_id,
        'qualification_artifact_sha256': qualification_artifact_sha256,
        'qualification_collector_evidence_sha256': qualification_collector_evidence_sha256,
        'qualification_probe_manifest_sha256': qualification_probe_manifest_sha256,
        'qualification_runtime_closure_manifest_sha256': (qualification_runtime_closure_manifest_sha256),
        'qualification_runtime_closure_receipt_sha256': (qualification_runtime_closure_receipt_sha256),
        'qualification_runtime_closure_sha256': qualification_runtime_closure_sha256,
        'qualification_collector_public_key_hex': qualification_collector_public_key_hex,
        'qualification_collector_key_id': qualification_collector_key_id,
        'qualification_verifier_source_sha256': qualification_verifier_source_sha256,
        'task_sha256': task_sha256,
        'provider_child_executable_sha256': provider_child_executable_sha256,
        'provider_plan_sha256': provider_plan_sha256,
        'collector_entrypoint_sha256': collector_entrypoint_sha256,
        'collector_interpreter_sha256': collector_interpreter_sha256,
        'collector_runtime_closure_manifest_sha256': (collector_runtime_closure_manifest_sha256),
        'collector_runtime_closure_receipt_sha256': (collector_runtime_closure_receipt_sha256),
        'collector_runtime_closure_sha256': collector_runtime_closure_sha256,
        'collector_public_key_hex': collector_public_key_hex,
        'collector_key_id': collector_key_id,
        'launcher_process_executable_sha256': launcher_process_executable_sha256,
        'bootstrap_authorization_key_id': bootstrap_authorization_key_id,
    }
    if any(
        len(value) != 64 or any(character not in '0123456789abcdef' for character in value) for value in pins.values()
    ):
        raise ValueError('managed drill release pins require lowercase 32-byte hexadecimal values')
    return hashlib.sha256(_RELEASE_PINS_DOMAIN + canonical_json_bytes(pins)).hexdigest()


def managed_clinical_real_kvm_challenge_sha256(
    *,
    drill_id: str,
    challenge_nonce_hex: str,
    challenge_issued_at: datetime,
    release_pins_sha256: str,
) -> str:
    """Bind an organizer challenge to its timestamp and pre-existing release inputs."""

    if len(drill_id) != 32 or any(character not in '0123456789abcdef' for character in drill_id):
        raise ValueError('managed drill authorization drill ID is invalid')
    if len(challenge_nonce_hex) != 64 or any(character not in '0123456789abcdef' for character in challenge_nonce_hex):
        raise ValueError('managed drill authorization challenge nonce is invalid')
    if challenge_issued_at.tzinfo is None or challenge_issued_at.utcoffset() is None:
        raise ValueError('managed drill authorization time requires a UTC offset')
    if len(release_pins_sha256) != 64 or any(character not in '0123456789abcdef' for character in release_pins_sha256):
        raise ValueError('managed drill authorization release pin is invalid')
    issued_at = challenge_issued_at.astimezone(UTC)
    issued_at_json = issued_at.isoformat().replace('+00:00', 'Z')
    material = {
        'challenge_issued_at': issued_at_json,
        'challenge_nonce_hex': challenge_nonce_hex,
        'drill_id': drill_id,
        'release_pins_sha256': release_pins_sha256,
    }
    return hashlib.sha256(_CHALLENGE_DOMAIN + canonical_json_bytes(material)).hexdigest()


def managed_clinical_real_kvm_authority_id(
    *,
    challenge_sha256: str,
) -> str:
    """Derive the sole registry authority from organizer-supplied challenge bytes."""

    _require_challenge_sha256(challenge_sha256)
    return f'vaxreplay-managed-real-kvm-{challenge_sha256[:32]}'


def managed_clinical_real_kvm_deployment_id(
    *,
    challenge_sha256: str,
) -> str:
    """Derive the fixed managed deployment identity from the external challenge."""

    _require_challenge_sha256(challenge_sha256)
    return f'managed-real-kvm-{challenge_sha256[:32]}'


def managed_clinical_real_kvm_registered_entry_id(
    *,
    challenge_sha256: str,
) -> str:
    """Derive the one reservation entry identity from the external challenge."""

    _require_challenge_sha256(challenge_sha256)
    return f'managed-real-kvm-challenge-{challenge_sha256}'


def _require_challenge_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ValueError('managed drill challenge digest is invalid')


class ManagedClinicalRegistryDrillObservation(StrictModel):
    """Selected exact exchanges reloaded from the authority's authenticated audit chain."""

    record_run_audit: AuthenticatedManagedClinicalRegistryAudit
    retry_claim_audit: AuthenticatedManagedClinicalRegistryAudit
    terminal_task_record_before_retry_sha256: str = Field(pattern=_SHA256_PATTERN)
    terminal_task_record_after_retry_sha256: str = Field(pattern=_SHA256_PATTERN)


class ManagedClinicalStartupCleanupDrillObservation(StrictModel):
    """One phase-labelled, create-once startup reconciliation artifact."""

    phase: Literal[
        'reservation_provisioning',
        'managed_operator_startup',
        'retry_denial_reopen',
    ]
    authenticated_cleanup: AuthenticatedManagedClinicalStartupCleanup


class ManagedClinicalRealKvmExternalPins(StrictModel):
    """Release-owned values which must never be selected from the collected evidence."""

    drill_id: str = Field(pattern=_RUN_ID_PATTERN)
    challenge_nonce_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    challenge_issued_at: datetime
    release_pins_sha256: str = Field(pattern=_SHA256_PATTERN)
    challenge_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    disk_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_key_id: str = Field(pattern=_SHA256_PATTERN)
    qualification_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_collector_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_probe_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_runtime_closure_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_runtime_closure_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_runtime_closure_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_collector_public_key_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    qualification_collector_key_id: str = Field(pattern=_SHA256_PATTERN)
    qualification_verifier_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider_child_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_entrypoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_interpreter_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_runtime_closure_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_runtime_closure_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_runtime_closure_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_public_key_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    collector_key_id: str = Field(pattern=_SHA256_PATTERN)
    launcher_process_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_authorization_key_id: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('challenge_issued_at')
    @classmethod
    def validate_challenge_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('managed drill challenge time requires a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_challenge(self) -> Self:
        if self.collector_key_id != managed_clinical_real_kvm_collector_key_id(
            bytes.fromhex(self.collector_public_key_hex)
        ):
            raise ValueError('managed drill release collector key ID differs from its public key')
        if self.qualification_collector_key_id != firecracker_live_collector_key_id(
            bytes.fromhex(self.qualification_collector_public_key_hex)
        ):
            raise ValueError('managed drill release qualification-collector key ID differs from its public key')
        release_pins_sha256 = managed_clinical_real_kvm_release_pins_sha256(
            worker_spec_sha256=self.worker_spec_sha256,
            execution_policy_sha256=self.execution_policy_sha256,
            guest_rpc_policy_sha256=self.guest_rpc_policy_sha256,
            guest_config_sha256=self.guest_config_sha256,
            disk_build_receipt_sha256=self.disk_build_receipt_sha256,
            qualification_key_id=self.qualification_key_id,
            qualification_artifact_sha256=self.qualification_artifact_sha256,
            qualification_collector_evidence_sha256=(self.qualification_collector_evidence_sha256),
            qualification_probe_manifest_sha256=self.qualification_probe_manifest_sha256,
            qualification_runtime_closure_manifest_sha256=(self.qualification_runtime_closure_manifest_sha256),
            qualification_runtime_closure_receipt_sha256=(self.qualification_runtime_closure_receipt_sha256),
            qualification_runtime_closure_sha256=(self.qualification_runtime_closure_sha256),
            qualification_collector_public_key_hex=(self.qualification_collector_public_key_hex),
            qualification_collector_key_id=self.qualification_collector_key_id,
            qualification_verifier_source_sha256=(self.qualification_verifier_source_sha256),
            task_sha256=self.task_sha256,
            provider_child_executable_sha256=(self.provider_child_executable_sha256),
            provider_plan_sha256=self.provider_plan_sha256,
            collector_entrypoint_sha256=self.collector_entrypoint_sha256,
            collector_interpreter_sha256=self.collector_interpreter_sha256,
            collector_runtime_closure_manifest_sha256=(self.collector_runtime_closure_manifest_sha256),
            collector_runtime_closure_receipt_sha256=(self.collector_runtime_closure_receipt_sha256),
            collector_runtime_closure_sha256=(self.collector_runtime_closure_sha256),
            collector_public_key_hex=self.collector_public_key_hex,
            collector_key_id=self.collector_key_id,
            launcher_process_executable_sha256=(self.launcher_process_executable_sha256),
            bootstrap_authorization_key_id=self.bootstrap_authorization_key_id,
        )
        challenge_sha256 = managed_clinical_real_kvm_challenge_sha256(
            drill_id=self.drill_id,
            challenge_nonce_hex=self.challenge_nonce_hex,
            challenge_issued_at=self.challenge_issued_at,
            release_pins_sha256=release_pins_sha256,
        )
        if self.release_pins_sha256 != release_pins_sha256 or self.challenge_sha256 != challenge_sha256:
            raise ValueError('managed drill challenge differs from its release inputs')
        return self


@dataclass(frozen=True, slots=True)
class ManagedClinicalRealKvmVerifierKeys:
    """Organizer-held keys used only to reauthenticate persisted component evidence."""

    workspace_receipt_key: bytes
    worker_attestation_key: bytes
    gateway_receipt_key: bytes
    guest_rpc_receipt_key: bytes
    bootstrap_receipt_key: bytes
    production_receipt_key: bytes
    qualification_key: bytes
    ownership_key: bytes
    startup_cleanup_key: bytes


class ManagedClinicalRealKvmProcessObservation(StrictModel):
    """Collector observation made while the exact Firecracker child is still live."""

    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    ownership_envelope_sha256: str = Field(pattern=_SHA256_PATTERN)
    firecracker_pid: int = Field(gt=1, le=2**31 - 1)
    firecracker_start_time_ticks: int = Field(gt=0, le=2**63 - 1)
    firecracker_process_group_id: int = Field(gt=1, le=2**31 - 1)
    firecracker_session_id: int = Field(gt=1, le=2**31 - 1)
    firecracker_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    kvm_device_path: Literal['/dev/kvm'] = '/dev/kvm'
    kvm_device_id: int = Field(ge=0, le=2**63 - 1)
    kvm_device_inode: int = Field(gt=0, le=2**63 - 1)
    kvm_device_rdev: int = Field(gt=0, le=2**63 - 1)
    firecracker_kvm_fd: int = Field(ge=0, le=2**31 - 1)
    firecracker_kvm_fd_rdev: int = Field(gt=0, le=2**63 - 1)
    proc_cgroup_path: str = Field(min_length=2, max_length=4096)
    cgroup_path: str = Field(min_length=2, max_length=4096)
    cgroup_device_id: int = Field(ge=0, le=2**63 - 1)
    cgroup_inode: int = Field(gt=0, le=2**63 - 1)
    firecracker_pid_file_path: str = Field(min_length=2, max_length=4096)
    firecracker_pid_file_device_id: int = Field(ge=0, le=2**63 - 1)
    firecracker_pid_file_inode: int = Field(gt=0, le=2**63 - 1)
    firecracker_pid_file_owner_uid: int = Field(ge=0, le=2**31 - 1)
    firecracker_pid_file_mode: int = Field(ge=0, le=0o7777)
    observed_at: datetime

    @field_validator('proc_cgroup_path', 'cgroup_path', 'firecracker_pid_file_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed process observation paths must be normalized and absolute')
        return value

    @field_validator('observed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('managed process observation time requires a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_kvm_device(self) -> Self:
        if self.firecracker_kvm_fd_rdev != self.kvm_device_rdev:
            raise ValueError('managed process observation Firecracker fd is not the pinned KVM device')
        return self


class ManagedClinicalRealKvmObservationGateRelease(StrictModel):
    """Durable release proving the real worker was observed before provider call zero returned."""

    schema_version: Literal['vaxreplay.managed-clinical-real-kvm-observation-gate.dev-v0.1'] = (
        'vaxreplay.managed-clinical-real-kvm-observation-gate.dev-v0.1'
    )
    drill_id: str = Field(pattern=_RUN_ID_PATTERN)
    challenge_nonce_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    challenge_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    provider_call_index: Literal[0] = 0
    ownership_envelope_sha256: str = Field(pattern=_SHA256_PATTERN)
    live_process_observation_sha256: str = Field(pattern=_SHA256_PATTERN)
    gate_binding_token_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    observed_at: datetime
    released_at: datetime
    persisted_path: str = Field(min_length=2, max_length=4096)
    create_once: Literal[True] = True
    root_owned: Literal[True] = True
    file_mode: Literal[384] = 0o600
    file_fsynced: Literal[True] = True
    parent_directory_fsynced: Literal[True] = True

    @field_validator('observed_at', 'released_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} requires a UTC offset')
        return value.astimezone(UTC)

    @field_validator('persisted_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed observation-gate path must be normalized and absolute')
        return value

    @model_validator(mode='after')
    def validate_chronology(self) -> Self:
        if self.released_at < self.observed_at:
            raise ValueError('managed observation gate cannot be released before observation')
        return self


class ManagedClinicalRealKvmDrillEvidence(StrictModel):
    """Facts carried in a portable signed envelope, not self-contained proof.

    Independent verification also requires the retained on-host registry database, ownership
    ledger, task workspace, qualification and provider state, plus the current KVM device.
    """

    schema_version: Literal['vaxreplay.managed-clinical-real-kvm-drill.dev-v0.3'] = (
        MANAGED_CLINICAL_REAL_KVM_DRILL_SCHEMA_VERSION
    )
    drill_id: str = Field(pattern=_RUN_ID_PATTERN)
    challenge_nonce_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    worker_spec: FirecrackerWorkerSpec
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    disk_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    qualification_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    deployment_sha256: str = Field(pattern=_SHA256_PATTERN)
    registry_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    startup_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    ownership_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    operator_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider_child_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_entrypoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_interpreter_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_runtime_closure: LoadedQualificationDriverRuntimeClosure
    launcher_process_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    deployment: ManagedClinicalStandaloneDeployment
    registry_config: ManagedClinicalRegistryConfig
    startup_config: ManagedClinicalStartupConfig
    ownership_config: ManagedClinicalOwnershipConfig
    operator_manifest: CanonicalClinicalOperatorManifest
    reservation: ClinicalProductionReservation
    task_record: ClinicalProductionTaskRecord
    ownership_chain: tuple[AuthenticatedManagedClinicalOwnership, ...] = Field(
        min_length=6,
        max_length=6,
    )
    live_process_observation: ManagedClinicalRealKvmProcessObservation
    observation_gate_release: ManagedClinicalRealKvmObservationGateRelease
    registry_observation: ManagedClinicalRegistryDrillObservation
    gateway_ledger_identity: GatewayLedgerIdentity
    gateway_revocation: GatewayCapabilityRevocation
    production_run_root: str
    production_run: AuthenticatedClinicalProductionRunV02
    bootstrap: AuthenticatedClinicalGuestBootstrap
    guest_rpc: AuthenticatedGuestRpcSession
    gateway_session: AuthenticatedGatewaySession
    worker_attestation: AuthenticatedFirecrackerWorkerAttestation
    submission: ExecutionSubmission
    startup_cleanups: tuple[ManagedClinicalStartupCleanupDrillObservation, ...] = Field(
        min_length=3,
        max_length=3,
    )
    managed_entrypoint_stdout_path: str = Field(min_length=2, max_length=4096)
    managed_entrypoint_stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    managed_entrypoint_exit_code: Literal[0] = 0
    provider_child_call_count: Literal[4] = 4
    retry_claim_denied_by_managed_authority: Literal[True] = True
    post_reconciliation_active_ownership_count: Literal[0] = 0
    post_reconciliation_unrevoked_capability_count: Literal[0] = 0
    post_reconciliation_process_group_count: Literal[0] = 0
    post_reconciliation_cgroup_count: Literal[0] = 0
    post_reconciliation_jail_count: Literal[0] = 0
    post_reconciliation_vsock_count: Literal[0] = 0
    collected_at: datetime
    host_os: Literal['Linux'] = 'Linux'
    kvm_character_device_used: Literal[True] = True
    managed_registry_exercised: Literal[True] = True
    startup_reconciliation_exercised: Literal[True] = True
    real_firecracker_ownership_recorded: Literal[True] = True
    normal_cleanup_recorded: Literal[True] = True
    production_run_reauthenticated_by_registry_service: Literal[True] = True
    external_provider_called: Literal[False] = False
    learned_model_weights_used: Literal[False] = False
    provider_credential_semantically_used: Literal[False] = False
    crash_recovery_of_real_firecracker_exercised: Literal[False] = False
    fixed_systemd_boot_or_power_loss_exercised: Literal[False] = False
    development_only: Literal[True] = True
    official_leaderboard_execution_qualified: Literal[False] = False

    @field_validator('production_run_root', 'managed_entrypoint_stdout_path')
    @classmethod
    def validate_production_run_root(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed retained path must be normalized and absolute')
        return value

    @field_validator('collected_at')
    @classmethod
    def validate_collected_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('managed real-KVM drill time must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_cross_bindings(self) -> Self:
        if (
            self.worker_spec_sha256 != firecracker_model_sha256(self.worker_spec)
            or self.deployment_sha256 != _model_sha256(self.deployment)
            or self.registry_config_sha256 != managed_clinical_registry_config_sha256(self.registry_config)
            or self.startup_config_sha256 != managed_clinical_startup_config_sha256(self.startup_config)
            or self.ownership_config_sha256 != managed_clinical_ownership_config_sha256(self.ownership_config)
            or self.operator_manifest_sha256 != _model_sha256(self.operator_manifest)
            or self.reservation_sha256 != clinical_production_reservation_sha256(self.reservation)
            or self.provider_child_executable_sha256 != self.operator_manifest.provider_subprocess.executable_sha256
            or self.provider_plan_sha256 != self.operator_manifest.provider_adapter.config_sha256
            or self.provider_plan_sha256 != self.operator_manifest.gateway_route.adapter_config_sha256
            or self.reservation.system.provider_subprocess_spec_sha256
            != provider_subprocess_spec_sha256(self.operator_manifest.provider_subprocess)
            or self.registry_config.launcher_process_executable_sha256 != self.launcher_process_executable_sha256
        ):
            raise ValueError('managed drill exact inputs differ from their retained digests')
        closure = self.collector_runtime_closure
        if (
            closure.manifest.driver_entrypoint_sha256 != self.collector_entrypoint_sha256
            or closure.manifest.interpreter_sha256 != self.collector_interpreter_sha256
            or closure.receipt.manifest_sha256 != closure.manifest_sha256
            or closure.receipt.driver_entrypoint_sha256 != self.collector_entrypoint_sha256
            or closure.receipt.interpreter_sha256 != self.collector_interpreter_sha256
        ):
            raise ValueError('managed drill collector differs from its retained runtime closure')
        if (
            self.deployment.registry_config_sha256 != self.registry_config_sha256
            or self.deployment.startup_config_sha256 != self.startup_config_sha256
            or self.deployment.ownership_config_sha256 != self.ownership_config_sha256
            or self.deployment.operator_manifest_sha256 != self.operator_manifest_sha256
            or self.operator_manifest.managed_registry_config_sha256 != self.registry_config_sha256
            or self.operator_manifest.managed_startup_config_sha256 != self.startup_config_sha256
            or self.operator_manifest.managed_ownership_config_sha256 != self.ownership_config_sha256
            or self.operator_manifest.expected_worker_spec_sha256 != self.worker_spec_sha256
            or self.operator_manifest.expected_guest_disk_build_receipt_sha256 != self.disk_build_receipt_sha256
            or self.operator_manifest.expected_qualification_artifact_sha256 != self.qualification_artifact_sha256
            or self.operator_manifest.reservation_sha256 != self.reservation_sha256
            or self.registry_config.registry_authority_id != self.reservation.registry_authority_id
            or self.startup_config.registry_authority_id != self.reservation.registry_authority_id
            or self.ownership_config.registry_authority_id != self.reservation.registry_authority_id
            or self.startup_config.worker_spec_sha256 != self.worker_spec_sha256
            or self.ownership_config.worker_spec_sha256 != self.worker_spec_sha256
            or self.ownership_config.firecracker_executable_sha256 != self.worker_spec.runtime.firecracker.sha256
            or self.reservation.system.harness != self.operator_manifest.harness
            or self.reservation.system.execution_policy_sha256
            != agentic_policy_sha256(self.operator_manifest.execution_policy)
            or self.reservation.system.worker_spec_sha256 != self.worker_spec_sha256
            or self.reservation.system.gateway_policy_sha256
            != authenticated_gateway_policy_sha256(self.operator_manifest.gateway_policy)
            or self.reservation.system.gateway_route != self.operator_manifest.gateway_route
            or self.reservation.system.gateway_route_sha256
            != gateway_model_route_sha256(self.operator_manifest.gateway_route)
            or self.reservation.system.provider_subprocess_behavior_sha256
            != provider_subprocess_behavior_sha256(self.operator_manifest.provider_subprocess)
            or self.reservation.system.provider_subprocess_module_source_sha256
            != self.operator_manifest.provider_subprocess_module_source_sha256
            or self.reservation.system.guest_rpc_policy_sha256
            != guest_rpc_policy_sha256(self.operator_manifest.guest_rpc_policy)
            or self.reservation.system.guest_bootstrap_authorization_key_id
            != self.operator_manifest.runtime_config.bootstrap_authorization_key_id
            or self.reservation.system.guest_bootstrap_receipt_key_id
            != self.operator_manifest.runtime_config.bootstrap_receipt_key_id
            or self.reservation.system.canonical_launcher_id != self.operator_manifest.deployment.canonical_launcher_id
            or self.reservation.system.canonical_launcher_executable_sha256
            != self.operator_manifest.deployment.canonical_launcher_executable_sha256
        ):
            raise ValueError('managed drill deployment/configuration inputs are cross-bound incorrectly')
        record = self.task_record
        redemption = record.start_redemption
        if (
            record.state != 'succeeded'
            or record.terminal_code != ClinicalProductionTerminalCode.SUCCESS
            or record.launch is None
            or redemption is None
            or record.start_redemption_sha256 is None
            or record.evidence_sha256 is None
            or record.submission_sha256 is None
            or record.launch.run_id != self.run_id
            or record.launch.reservation_sha256 != self.reservation_sha256
            or clinical_production_start_redemption_sha256(redemption) != record.start_redemption_sha256
        ):
            raise ValueError('managed drill lacks one successful, redeemed registry attempt')
        task_bindings = tuple(item for item in self.reservation.tasks if item.episode_id == record.episode_id)
        if len(task_bindings) != 1:
            raise ValueError('managed drill task is absent or duplicated in the reservation')
        binding = task_bindings[0]
        launch = record.launch
        if (
            binding.task_sha256 != self.task_sha256
            or self.operator_manifest.episode_id != record.episode_id
            or record.launch_sha256 != clinical_production_task_launch_sha256(launch)
            or (
                launch.registry_authority_id,
                launch.reservation_sha256,
                launch.cohort_manifest_sha256,
                launch.system_identity_sha256,
                launch.episode_id,
                launch.workspace_manifest_sha256,
            )
            != (
                self.reservation.registry_authority_id,
                self.reservation_sha256,
                self.reservation.cohort_manifest_sha256,
                self.reservation.system_identity_sha256,
                binding.episode_id,
                binding.workspace_manifest_sha256,
            )
            or (
                redemption.registry_authority_id,
                redemption.reservation_sha256,
                redemption.launch_sha256,
                redemption.system_identity_sha256,
                redemption.episode_id,
                redemption.run_id,
                redemption.canonical_launcher_id,
                redemption.canonical_launcher_executable_sha256,
            )
            != (
                self.reservation.registry_authority_id,
                self.reservation_sha256,
                record.launch_sha256,
                self.reservation.system_identity_sha256,
                binding.episode_id,
                self.run_id,
                self.registry_config.canonical_launcher_id,
                self.registry_config.canonical_launcher_executable_sha256,
            )
        ):
            raise ValueError('managed drill task, launch, and redemption are not one exact attempt')

        states = tuple(item.record.state for item in self.ownership_chain)
        sequences = tuple(item.record.sequence for item in self.ownership_chain)
        if states != _EXPECTED_OWNERSHIP_STATES or sequences != tuple(range(6)):
            raise ValueError('managed drill ownership chain is incomplete or out of order')
        if any(
            item.record.run_id != self.run_id
            or item.record.reservation_sha256 != self.reservation_sha256
            or item.record.worker_spec_sha256 != self.worker_spec_sha256
            for item in self.ownership_chain
        ):
            raise ValueError('managed drill ownership chain differs from the exact attempt')
        first = self.ownership_chain[0].record
        running = self.ownership_chain[3].record
        cleaned = self.ownership_chain[-1].record
        if (
            (
                first.ledger_id,
                first.registry_authority_id,
                first.launch_sha256,
                first.episode_id,
            )
            != (
                self.ownership_config.ledger_id,
                self.reservation.registry_authority_id,
                record.launch_sha256,
                record.episode_id,
            )
            or running.prepared_worker_sha256 != redemption.prepared_worker_sha256
            or running.start_redemption_sha256 != record.start_redemption_sha256
            or running.capability_id != redemption.gateway_capability_id
            or running.firecracker_pid is None
            or running.firecracker_start_time_ticks is None
            or running.process_group_id is None
            or running.cgroup_device_id is None
            or running.cgroup_inode is None
            or running.firecracker_executable_sha256 is None
            or cleaned.terminal_reason != 'runtime_cleanup'
            or cleaned.capability_revoked is not True
            or cleaned.cleanup_receipt_sha256 is None
            or cleaned.capability_id != redemption.gateway_capability_id
        ):
            raise ValueError('managed drill lacks a real running identity and normal cleanup')
        observation = self.live_process_observation
        if (
            observation.run_id,
            observation.ownership_envelope_sha256,
            observation.firecracker_pid,
            observation.firecracker_start_time_ticks,
            observation.firecracker_process_group_id,
            observation.firecracker_session_id,
            observation.firecracker_executable_sha256,
            observation.cgroup_path,
            observation.cgroup_device_id,
            observation.cgroup_inode,
            observation.firecracker_pid_file_path,
            observation.firecracker_pid_file_device_id,
            observation.firecracker_pid_file_inode,
            observation.firecracker_pid_file_owner_uid,
            observation.firecracker_pid_file_mode,
        ) != (
            self.run_id,
            hashlib.sha256(canonical_json_bytes(self.ownership_chain[3])).hexdigest(),
            running.firecracker_pid,
            running.firecracker_start_time_ticks,
            running.process_group_id,
            running.process_group_session_id,
            running.firecracker_executable_sha256,
            running.cgroup_path,
            running.cgroup_device_id,
            running.cgroup_inode,
            running.firecracker_pid_file_path,
            running.firecracker_pid_file_device_id,
            running.firecracker_pid_file_inode,
            running.firecracker_pid_file_owner_uid,
            running.firecracker_pid_file_mode,
        ):
            raise ValueError('live process observation differs from the authenticated running record')
        if observation.proc_cgroup_path != _proc_cgroup_path_for(running.cgroup_path):
            raise ValueError('live process observation differs from the exact cgroup-v2 membership')
        gate = self.observation_gate_release
        first_provider_attempts = tuple(item for item in self.gateway_session.attempts if item.call_index == 0)
        if len(first_provider_attempts) != 1 or (first_provider_attempts[0].provider_result is None):
            raise ValueError('managed observation gate lacks one completed provider call zero')
        first_provider_result = first_provider_attempts[0].provider_result
        assert first_provider_result is not None
        if (
            gate.drill_id,
            gate.challenge_nonce_hex,
            gate.run_id,
            gate.ownership_envelope_sha256,
            gate.live_process_observation_sha256,
            gate.observed_at,
        ) != (
            self.drill_id,
            self.challenge_nonce_hex,
            self.run_id,
            observation.ownership_envelope_sha256,
            _model_sha256(observation),
            observation.observed_at,
        ):
            raise ValueError('managed observation-gate release differs from the live worker')
        if not (
            running.recorded_at
            <= gate.observed_at
            <= gate.released_at
            <= first_provider_result.started_at
            <= self.worker_attestation.attestation.cgroup_empty_at
            <= cleaned.recorded_at
            <= self.collected_at
        ):
            raise ValueError('managed observation-gate or provider timestamps are out of order')

        revocation = self.gateway_revocation
        if (
            revocation.capability_id != redemption.gateway_capability_id
            or revocation.run_id != self.run_id
            or revocation.attempt_reservation_sha256 != record.start_redemption_sha256
            or revocation.reason != GatewayCapabilityRevocationReason.RUNTIME_CLEANUP
            or revocation.registered_binding is None
        ):
            raise ValueError('managed drill gateway tombstone differs from the redeemed start')

        outer = self.production_run.receipt
        bootstrap_hello = self.bootstrap.signed_hello.hello
        guest = self.guest_rpc.seal
        gateway = self.gateway_session.seal
        worker = self.worker_attestation.attestation
        if revocation.registered_binding != gateway_capability_binding(self.gateway_session.grant):
            raise ValueError('managed drill gateway tombstone differs from the exact gateway grant')
        if not gateway.sealed_at <= revocation.revoked_at <= cleaned.recorded_at:
            raise ValueError('managed drill gateway tombstone timestamp is outside normal cleanup')
        if (
            outer.run_id,
            outer.start_redemption_sha256,
            outer.worker_spec_sha256,
            bootstrap_hello.run_id,
            bootstrap_hello.start_redemption_sha256,
            guest.run_id,
            guest.attempt_reservation_sha256,
            gateway.run_id,
            gateway.capability_id,
            worker.run_id,
            worker.attempt_reservation_sha256,
            worker.worker_spec_sha256,
        ) != (
            self.run_id,
            record.start_redemption_sha256,
            self.worker_spec_sha256,
            self.run_id,
            record.start_redemption_sha256,
            self.run_id,
            record.start_redemption_sha256,
            self.run_id,
            redemption.gateway_capability_id,
            self.run_id,
            record.start_redemption_sha256,
            self.worker_spec_sha256,
        ):
            raise ValueError('managed drill production artifacts differ from the registry attempt')
        if record.evidence_sha256 != hashlib.sha256(canonical_json_bytes(self.production_run)).hexdigest():
            raise ValueError('registry evidence digest differs from the exact production receipt')
        if cleaned.cleanup_receipt_sha256 != worker.cleanup_receipt_sha256:
            raise ValueError('managed ownership cleanup differs from the worker attestation')
        if not (
            running.recorded_at
            <= observation.observed_at
            <= worker.cgroup_empty_at
            <= cleaned.recorded_at
            <= self.collected_at
        ):
            raise ValueError('managed drill live observation or cleanup timestamps are out of order')
        if (
            guest.terminal_status != GuestRpcTerminalStatus.COMPLETED
            or not guest.submit_accepted
            or guest.model_call_count != self.provider_child_call_count
            or gateway.terminal_reason != GatewayTerminalReason.COMPLETED
            or gateway.successful_call_count != self.provider_child_call_count
        ):
            raise ValueError('managed drill did not complete the four-turn guest/provider loop')
        if hashlib.sha256(canonical_json_bytes(self.submission)).hexdigest() != (record.submission_sha256):
            raise ValueError('managed drill submission differs from the registry result')
        phases = tuple(item.phase for item in self.startup_cleanups)
        cleanups = tuple(item.authenticated_cleanup for item in self.startup_cleanups)
        if phases != _EXPECTED_STARTUP_PHASES:
            raise ValueError('managed drill startup cleanup phases are incomplete or out of order')
        if len({item.request_sha256 for item in cleanups}) != len(cleanups) or len(
            {item.persisted_path for item in cleanups}
        ) != len(cleanups):
            raise ValueError('managed drill startup cleanup phases must retain distinct requests and files')
        if any(
            item.cleanup_receipt.discovered_worker_count != item.cleanup_receipt.terminated_worker_count
            or item.cleanup_receipt.discovered_ephemeral_run_artifact_count
            != item.cleanup_receipt.removed_ephemeral_run_artifact_count
            or item.cleanup_receipt.discovered_capability_count != item.cleanup_receipt.revoked_capability_count
            for item in cleanups
        ):
            raise ValueError('managed drill contains an incomplete startup cleanup receipt')
        return self


class AuthenticatedManagedClinicalRealKvmDrill(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-managed-clinical-real-kvm-drill.dev-v0.3'] = (
        AUTHENTICATED_MANAGED_CLINICAL_REAL_KVM_DRILL_SCHEMA_VERSION
    )
    evidence: ManagedClinicalRealKvmDrillEvidence
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_public_key_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    collector_key_id: str = Field(pattern=_SHA256_PATTERN)
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    signature_hex: str = Field(pattern=_SIGNATURE_PATTERN)

    @model_validator(mode='after')
    def validate_hashes(self) -> Self:
        if self.evidence_sha256 != managed_clinical_real_kvm_drill_sha256(self.evidence):
            raise ValueError('managed drill signature envelope has the wrong evidence hash')
        public_key = bytes.fromhex(self.collector_public_key_hex)
        if self.collector_key_id != managed_clinical_real_kvm_collector_key_id(public_key):
            raise ValueError('managed drill collector key ID differs from its public key')
        return self


def managed_clinical_real_kvm_drill_sha256(
    evidence: ManagedClinicalRealKvmDrillEvidence,
) -> str:
    canonical = ManagedClinicalRealKvmDrillEvidence.model_validate_json(canonical_json_bytes(evidence))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def managed_clinical_real_kvm_collector_key_id(public_key: bytes) -> str:
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError('managed drill collector public key must contain exactly 32 bytes')
    return hashlib.sha256(_KEY_ID_DOMAIN + public_key).hexdigest()


def authenticate_managed_clinical_real_kvm_drill(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    private_key: Ed25519PrivateKey,
) -> AuthenticatedManagedClinicalRealKvmDrill:
    canonical = ManagedClinicalRealKvmDrillEvidence.model_validate_json(canonical_json_bytes(evidence))
    body = canonical_json_bytes(canonical)
    public_key = private_key.public_key().public_bytes_raw()
    return AuthenticatedManagedClinicalRealKvmDrill(
        evidence=canonical,
        evidence_sha256=hashlib.sha256(body).hexdigest(),
        collector_public_key_hex=public_key.hex(),
        collector_key_id=managed_clinical_real_kvm_collector_key_id(public_key),
        signature_hex=private_key.sign(_SIGNATURE_DOMAIN + body).hex(),
    )


def verify_authenticated_managed_clinical_real_kvm_drill(
    authenticated: AuthenticatedManagedClinicalRealKvmDrill,
    *,
    expected_collector_public_key_hex: str,
    expected_worker_spec_sha256: str,
    expected_disk_build_receipt_sha256: str,
    expected_qualification_artifact_sha256: str,
    expected_task_sha256: str,
    expected_provider_child_executable_sha256: str,
    expected_collector_entrypoint_sha256: str,
) -> ManagedClinicalRealKvmDrillEvidence:
    """Verify the collector signature and legacy minimum pins only.

    This compatibility helper does not independently qualify a drill.  New acceptance code must
    call :func:`independently_verify_authenticated_managed_clinical_real_kvm_drill`, which also
    reloads and authenticates every durable component.
    """

    canonical = AuthenticatedManagedClinicalRealKvmDrill.model_validate_json(canonical_json_bytes(authenticated))
    if canonical.collector_public_key_hex != expected_collector_public_key_hex:
        raise ValueError('managed drill collector public key differs from its external pin')
    expected = (
        expected_worker_spec_sha256,
        expected_disk_build_receipt_sha256,
        expected_qualification_artifact_sha256,
        expected_task_sha256,
        expected_provider_child_executable_sha256,
        expected_collector_entrypoint_sha256,
    )
    observed = (
        canonical.evidence.worker_spec_sha256,
        canonical.evidence.disk_build_receipt_sha256,
        canonical.evidence.qualification_artifact_sha256,
        canonical.evidence.task_sha256,
        canonical.evidence.provider_child_executable_sha256,
        canonical.evidence.collector_entrypoint_sha256,
    )
    if observed != expected:
        raise ValueError('managed drill differs from an external execution pin')
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(expected_collector_public_key_hex)).verify(
            bytes.fromhex(canonical.signature_hex),
            _SIGNATURE_DOMAIN + canonical_json_bytes(canonical.evidence),
        )
    except (InvalidSignature, ValueError):
        raise ValueError('managed drill collector signature verification failed') from None
    return canonical.evidence


def verify_managed_clinical_real_kvm_drill_from_persisted_state(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    external_pins: ManagedClinicalRealKvmExternalPins,
    keys: ManagedClinicalRealKvmVerifierKeys,
) -> ManagedClinicalRealKvmDrillEvidence:
    """Reload and authenticate every durable component before accepting collector claims."""

    canonical = ManagedClinicalRealKvmDrillEvidence.model_validate_json(canonical_json_bytes(evidence))
    _verify_external_pins(canonical, external_pins=external_pins)
    _verify_pinned_operator_inputs(canonical, keys=keys)
    _verify_provider_plan(canonical)
    _verify_observation_gate(canonical, external_pins=external_pins)
    _verify_managed_entrypoint_stdout(canonical)
    _verify_collector_runtime_closure(canonical, external_pins=external_pins)
    workspace = _verify_production_evidence(canonical, keys=keys)
    _verify_ownership_and_host_cleanup(canonical, keys=keys)
    _verify_startup_cleanups(canonical, keys=keys)
    _verify_gateway_tombstone(canonical, keys=keys)
    _verify_registry_observation(
        canonical,
        keys=keys,
        external_pins=external_pins,
    )
    if _model_sha256(workspace.task) != canonical.task_sha256:
        raise ValueError('managed drill workspace task differs from its external task pin')
    return canonical


def independently_verify_authenticated_managed_clinical_real_kvm_drill(
    authenticated: AuthenticatedManagedClinicalRealKvmDrill,
    *,
    expected_evidence_sha256: str,
    expected_collector_public_key_hex: str,
    external_pins: ManagedClinicalRealKvmExternalPins,
    keys: ManagedClinicalRealKvmVerifierKeys,
) -> ManagedClinicalRealKvmDrillEvidence:
    """Authenticate the collector and rederive the result using retained on-host state."""

    if len(expected_evidence_sha256) != 64 or any(
        character not in '0123456789abcdef' for character in expected_evidence_sha256
    ):
        raise ValueError('managed drill expected evidence digest is invalid')
    canonical = AuthenticatedManagedClinicalRealKvmDrill.model_validate_json(canonical_json_bytes(authenticated))
    if not hmac.compare_digest(canonical.evidence_sha256, expected_evidence_sha256):
        raise ValueError('managed drill evidence differs from its create-once external digest pin')
    if not hmac.compare_digest(
        expected_collector_public_key_hex,
        external_pins.collector_public_key_hex,
    ) or not hmac.compare_digest(
        canonical.collector_key_id,
        external_pins.collector_key_id,
    ):
        raise ValueError('managed drill collector identity differs from its pre-execution release pins')
    evidence = verify_authenticated_managed_clinical_real_kvm_drill(
        canonical,
        expected_collector_public_key_hex=expected_collector_public_key_hex,
        expected_worker_spec_sha256=external_pins.worker_spec_sha256,
        expected_disk_build_receipt_sha256=(external_pins.disk_build_receipt_sha256),
        expected_qualification_artifact_sha256=(external_pins.qualification_artifact_sha256),
        expected_task_sha256=external_pins.task_sha256,
        expected_provider_child_executable_sha256=(external_pins.provider_child_executable_sha256),
        expected_collector_entrypoint_sha256=(external_pins.collector_entrypoint_sha256),
    )
    return verify_managed_clinical_real_kvm_drill_from_persisted_state(
        evidence,
        external_pins=external_pins,
        keys=keys,
    )


def _verify_external_pins(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    external_pins: ManagedClinicalRealKvmExternalPins,
) -> None:
    external_pins = ManagedClinicalRealKvmExternalPins.model_validate_json(canonical_json_bytes(external_pins))
    if external_pins.challenge_issued_at > datetime.now(UTC):
        raise ValueError('managed drill external challenge was issued in the future')
    expected = (
        external_pins.drill_id,
        external_pins.challenge_nonce_hex,
        external_pins.worker_spec_sha256,
        external_pins.execution_policy_sha256,
        external_pins.guest_rpc_policy_sha256,
        external_pins.guest_config_sha256,
        external_pins.disk_build_receipt_sha256,
        external_pins.qualification_key_id,
        external_pins.qualification_artifact_sha256,
        external_pins.qualification_collector_evidence_sha256,
        external_pins.qualification_probe_manifest_sha256,
        external_pins.qualification_runtime_closure_manifest_sha256,
        external_pins.qualification_runtime_closure_receipt_sha256,
        external_pins.qualification_runtime_closure_sha256,
        external_pins.qualification_collector_public_key_hex,
        external_pins.qualification_collector_key_id,
        external_pins.qualification_verifier_source_sha256,
        external_pins.task_sha256,
        external_pins.provider_child_executable_sha256,
        external_pins.provider_plan_sha256,
        external_pins.collector_entrypoint_sha256,
        external_pins.collector_interpreter_sha256,
        external_pins.collector_runtime_closure_manifest_sha256,
        external_pins.collector_runtime_closure_receipt_sha256,
        external_pins.collector_runtime_closure_sha256,
        external_pins.launcher_process_executable_sha256,
        external_pins.bootstrap_authorization_key_id,
    )
    closure = evidence.collector_runtime_closure
    manifest = evidence.operator_manifest
    observed = (
        evidence.drill_id,
        evidence.challenge_nonce_hex,
        evidence.worker_spec_sha256,
        agentic_policy_sha256(manifest.execution_policy),
        guest_rpc_policy_sha256(manifest.guest_rpc_policy),
        manifest.submitted_harness.baked_config_sha256,
        evidence.disk_build_receipt_sha256,
        manifest.expected_qualification_key_id,
        evidence.qualification_artifact_sha256,
        manifest.expected_collector_evidence_sha256,
        manifest.expected_probe_manifest_sha256,
        manifest.expected_driver_runtime_closure_manifest_sha256,
        manifest.expected_driver_runtime_closure_receipt_sha256,
        manifest.expected_driver_runtime_closure_sha256,
        manifest.expected_collector_public_key_hex,
        manifest.expected_collector_key_id,
        manifest.expected_qualification_verifier_source_sha256,
        evidence.task_sha256,
        evidence.provider_child_executable_sha256,
        evidence.provider_plan_sha256,
        evidence.collector_entrypoint_sha256,
        evidence.collector_interpreter_sha256,
        closure.manifest_sha256,
        closure.receipt_sha256,
        closure.closure_sha256,
        evidence.launcher_process_executable_sha256,
        manifest.runtime_config.bootstrap_authorization_key_id,
    )
    if observed != expected:
        raise ValueError('managed drill differs from its stable pre-execution release pins')
    if manifest.guest_boot_dispatch.guest_config_sha256 != external_pins.guest_config_sha256:
        raise ValueError('managed drill guest configuration surfaces differ from their release pin')
    expected_authority_id = managed_clinical_real_kvm_authority_id(challenge_sha256=external_pins.challenge_sha256)
    if (
        evidence.registry_config.registry_authority_id != expected_authority_id
        or evidence.deployment.deployment_id
        != managed_clinical_real_kvm_deployment_id(challenge_sha256=external_pins.challenge_sha256)
        or evidence.reservation.registered_entry_id
        != managed_clinical_real_kvm_registered_entry_id(challenge_sha256=external_pins.challenge_sha256)
    ):
        raise ValueError('managed drill authority namespace differs from its external challenge')
    launch = evidence.task_record.launch
    redemption = evidence.task_record.start_redemption
    if launch is None or redemption is None:
        raise ValueError('managed drill challenge cannot bind a task without a redeemed launch')
    event_times = (
        evidence.reservation.reserved_at,
        launch.claimed_at,
        redemption.redeemed_at,
        *(item.record.recorded_at for item in evidence.ownership_chain),
        *(item.authenticated_cleanup.reconciliation_request.requested_at for item in evidence.startup_cleanups),
        evidence.registry_observation.record_run_audit.audited_at,
        evidence.registry_observation.retry_claim_audit.audited_at,
        evidence.live_process_observation.observed_at,
        evidence.observation_gate_release.observed_at,
        evidence.observation_gate_release.released_at,
        evidence.collected_at,
    )
    if any(value < external_pins.challenge_issued_at for value in event_times):
        raise ValueError('managed drill retained an event from before its external challenge')


def _verify_collector_runtime_closure(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    external_pins: ManagedClinicalRealKvmExternalPins,
) -> None:
    loaded = verify_qualification_driver_runtime_closure(
        Path(evidence.collector_runtime_closure.root),
        expected_manifest_sha256=(external_pins.collector_runtime_closure_manifest_sha256),
        expected_receipt_sha256=(external_pins.collector_runtime_closure_receipt_sha256),
        expected_closure_sha256=external_pins.collector_runtime_closure_sha256,
    )
    if loaded != evidence.collector_runtime_closure:
        raise ValueError('managed drill collector closure differs from persisted bytes')
    if (
        loaded.manifest.driver_entrypoint_sha256 != external_pins.collector_entrypoint_sha256
        or loaded.manifest.interpreter_sha256 != external_pins.collector_interpreter_sha256
    ):
        raise ValueError('managed drill collector runtime differs from executable pins')


def _verify_pinned_operator_inputs(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    keys: ManagedClinicalRealKvmVerifierKeys,
) -> None:
    manifest = evidence.operator_manifest
    worker_spec, _worker_spec_bytes = load_pinned_firecracker_worker_spec(
        Path(manifest.worker_spec_path),
        expected_worker_spec_sha256=evidence.worker_spec_sha256,
    )
    if worker_spec != evidence.worker_spec:
        raise ValueError('managed drill worker specification differs from persisted pinned bytes')
    validate_checked_in_executable_pins(manifest)
    validate_side_effect_free_runtime_parity(manifest, worker_spec)
    guest_disks = load_and_verify_operator_guest_disks(manifest, worker_spec)
    if guest_disks.receipt_sha256 != evidence.disk_build_receipt_sha256:
        raise ValueError('managed drill guest disk receipt differs from its external pin')

    qualification = load_firecracker_qualification(
        Path(manifest.qualification_root),
        qualification_key=keys.qualification_key,
        expected_qualification_key_id=manifest.expected_qualification_key_id,
        expected_worker_spec_sha256=evidence.worker_spec_sha256,
        expected_artifact_sha256=evidence.qualification_artifact_sha256,
        expected_collector_evidence_sha256=manifest.expected_collector_evidence_sha256,
        expected_probe_manifest_sha256=manifest.expected_probe_manifest_sha256,
        expected_driver_runtime_closure_manifest_sha256=(manifest.expected_driver_runtime_closure_manifest_sha256),
        expected_driver_runtime_closure_receipt_sha256=(manifest.expected_driver_runtime_closure_receipt_sha256),
        expected_driver_runtime_closure_sha256=(manifest.expected_driver_runtime_closure_sha256),
        expected_collector_public_key_hex=manifest.expected_collector_public_key_hex,
        expected_collector_key_id=manifest.expected_collector_key_id,
        expected_verifier_source_sha256=(manifest.expected_qualification_verifier_source_sha256),
    )
    qualification_record = qualification.authenticated.record
    full_suite = qualification_record.full_suite_evidence
    if (
        qualification.artifact_sha256 != evidence.qualification_artifact_sha256
        or qualification_record.qualified is not True
        or qualification_record.preflight is None
        or full_suite is None
        or full_suite.all_required_drills_passed is not True
    ):
        raise ValueError('managed drill lacks authenticated full-runtime qualification')

    validate_managed_clinical_deployment_binding(
        LoadedManagedClinicalDeployment(
            deployment=evidence.deployment,
            registry_config=evidence.registry_config,
            startup_config=evidence.startup_config,
            ownership_config=evidence.ownership_config,
            manifest=manifest,
            manifest_sha256=evidence.operator_manifest_sha256,
            secrets=ManagedClinicalDeploymentSecrets(
                startup_cleanup_key=keys.startup_cleanup_key,
                ownership_key=keys.ownership_key,
            ),
        )
    )


def _verify_provider_plan(evidence: ManagedClinicalRealKvmDrillEvidence) -> None:
    subprocess_spec = evidence.operator_manifest.provider_subprocess
    executable = _read_safe_regular_file(
        Path(subprocess_spec.executable_path),
        maximum_bytes=256 * 1024 * 1024,
    )
    if not hmac.compare_digest(
        hashlib.sha256(executable).hexdigest(),
        evidence.provider_child_executable_sha256,
    ):
        raise ValueError('managed drill provider executable differs from its exact digest pin')
    interpreter_path = evidence.collector_runtime_closure.manifest.interpreter_path
    try:
        expected_shebang = f'#!{interpreter_path} -ISB\n'.encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError('managed drill provider interpreter path is not UTF-8') from None
    if not executable.startswith(expected_shebang):
        raise ValueError('managed drill provider child does not disable ambient Python startup hooks')
    argv = subprocess_spec.argv_suffix
    if (
        len(argv) != 4
        or argv[0] != '--plan'
        or argv[2] != '--expected-plan-sha256'
        or argv[3] != evidence.provider_plan_sha256
    ):
        raise ValueError('managed drill provider child does not receive the exact pinned plan')
    plan_path = PurePosixPath(argv[1])
    if not plan_path.is_absolute() or '..' in plan_path.parts or str(plan_path) != argv[1]:
        raise ValueError('managed drill provider plan path is not normalized and absolute')
    body = _read_safe_regular_file(Path(argv[1]), maximum_bytes=8 * 1024 * 1024)
    if not hmac.compare_digest(hashlib.sha256(body).hexdigest(), evidence.provider_plan_sha256):
        raise ValueError('managed drill provider plan differs from its exact digest pin')


def _verify_observation_gate(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    external_pins: ManagedClinicalRealKvmExternalPins,
) -> None:
    """Reload the create-once gate and bind its token to the precommitted provider plan."""

    release = evidence.observation_gate_release
    if not hmac.compare_digest(release.challenge_sha256, external_pins.challenge_sha256):
        raise ValueError('managed observation-gate release differs from the external challenge')
    release_path = Path(release.persisted_path)
    body = _read_safe_regular_file(release_path, maximum_bytes=64 * 1024)
    try:
        loaded = ManagedClinicalRealKvmObservationGateRelease.model_validate_json(body)
    except ValueError:
        raise ValueError('managed observation-gate release has invalid persisted bytes') from None
    if canonical_json_bytes(loaded) != body or loaded != release:
        raise ValueError('managed observation-gate release differs from create-once persisted bytes')
    metadata = release_path.lstat()
    parent = release_path.parent.lstat()
    if (
        metadata.st_uid != 0
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise ValueError('managed observation-gate release lacks its root-private file identity')

    argv = evidence.operator_manifest.provider_subprocess.argv_suffix
    if len(argv) != 4:
        raise ValueError('managed observation gate cannot locate the pinned provider plan')
    plan_body = _read_safe_regular_file(Path(argv[1]), maximum_bytes=8 * 1024 * 1024)
    try:
        plan = json.loads(plan_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError('managed observation-gate provider plan is invalid JSON') from None
    if not isinstance(plan, dict) or canonical_json_bytes(plan) != plan_body:
        raise ValueError('managed observation-gate provider plan is not canonical JSON')
    gate = plan.get('observation_gate')
    expected_gate_fields = {
        'binding_token_sha256',
        'challenge_nonce_hex',
        'drill_id',
        'path',
        'provider_call_index',
        'timeout_seconds',
    }
    if not isinstance(gate, dict) or set(gate) != expected_gate_fields:
        raise ValueError('managed observation-gate provider plan lacks its exact static contract')
    expected = (
        release.persisted_path,
        hashlib.sha256(bytes.fromhex(release.gate_binding_token_hex)).hexdigest(),
        _OBSERVATION_GATE_TIMEOUT_SECONDS,
        evidence.drill_id,
        evidence.challenge_nonce_hex,
        0,
    )
    observed = (
        gate.get('path'),
        gate.get('binding_token_sha256'),
        gate.get('timeout_seconds'),
        gate.get('drill_id'),
        gate.get('challenge_nonce_hex'),
        gate.get('provider_call_index'),
    )
    if observed != expected:
        raise ValueError('managed observation-gate release differs from its precommitted plan')


def _verify_managed_entrypoint_stdout(evidence: ManagedClinicalRealKvmDrillEvidence) -> None:
    """Reload the bounded fixed-launcher success receipt instead of trusting its bare digest."""

    body = _read_safe_regular_file(
        Path(evidence.managed_entrypoint_stdout_path),
        maximum_bytes=1024 * 1024,
    )
    if not hmac.compare_digest(
        hashlib.sha256(body).hexdigest(),
        evidence.managed_entrypoint_stdout_sha256,
    ):
        raise ValueError('managed entrypoint stdout differs from its retained digest')
    if not body.endswith(b'\n'):
        raise ValueError('managed entrypoint stdout lacks one canonical line')
    try:
        receipt = json.loads(body[:-1])
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError('managed entrypoint stdout is invalid JSON') from None
    if not isinstance(receipt, dict) or canonical_json_bytes(receipt) + b'\n' != body:
        raise ValueError('managed entrypoint stdout is not canonical JSON')
    if set(receipt) != {
        'attempt_consumed',
        'episode_id',
        'evidence_sha256',
        'leaderboard_admitted',
        'live_deployment_qualification_claimed',
        'managed_one_host_authority',
        'reservation_sha256',
        'retry_permitted',
        'run_id',
        'status',
    }:
        raise ValueError('managed entrypoint stdout has an unexpected contract')
    expected = (
        'succeeded',
        evidence.reservation_sha256,
        evidence.task_record.episode_id,
        evidence.run_id,
        evidence.task_record.evidence_sha256,
        True,
        False,
        True,
        False,
        False,
    )
    observed = (
        receipt.get('status'),
        receipt.get('reservation_sha256'),
        receipt.get('episode_id'),
        receipt.get('run_id'),
        receipt.get('evidence_sha256'),
        receipt.get('attempt_consumed'),
        receipt.get('retry_permitted'),
        receipt.get('managed_one_host_authority'),
        receipt.get('live_deployment_qualification_claimed'),
        receipt.get('leaderboard_admitted'),
    )
    if observed != expected:
        raise ValueError('managed entrypoint stdout differs from the authenticated attempt')


def _verify_production_evidence(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    keys: ManagedClinicalRealKvmVerifierKeys,
):
    manifest = evidence.operator_manifest
    system = evidence.reservation.system
    workspace = load_clinical_agentic_workspace(
        Path(manifest.workspace_root),
        expected_authenticated_receipt_sha256=(manifest.expected_authenticated_workspace_receipt_sha256),
        receipt_key=keys.workspace_receipt_key,
        expected_receipt_key_id=manifest.expected_workspace_receipt_key_id,
    )
    loaded = load_clinical_production_run_v02(
        Path(evidence.production_run_root),
        workspace=workspace,
        expected_authenticated_workspace_receipt_sha256=(manifest.expected_authenticated_workspace_receipt_sha256),
        workspace_receipt_key=keys.workspace_receipt_key,
        expected_workspace_receipt_key_id=manifest.expected_workspace_receipt_key_id,
        expected_run_id=evidence.run_id,
        expected_attempt_reservation_sha256=(evidence.task_record.start_redemption_sha256 or ''),
        policy=manifest.execution_policy,
        harness=manifest.harness,
        worker_spec=evidence.worker_spec,
        worker_attestation_key=keys.worker_attestation_key,
        expected_worker_attestation_key_id=system.worker_attestation_key_id,
        gateway_receipt_key=keys.gateway_receipt_key,
        expected_gateway_receipt_key_id=system.gateway_receipt_key_id,
        expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(manifest.gateway_policy),
        expected_gateway_route_sha256=gateway_model_route_sha256(manifest.gateway_route),
        guest_rpc_receipt_key=keys.guest_rpc_receipt_key,
        expected_guest_rpc_receipt_key_id=system.guest_rpc_receipt_key_id,
        expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(manifest.guest_rpc_policy),
        clinical_guest_bootstrap_receipt_key=keys.bootstrap_receipt_key,
        expected_clinical_guest_bootstrap_receipt_key_id=(system.guest_bootstrap_receipt_key_id),
        clinical_guest_bootstrap_trust_anchor=manifest.bootstrap_trust_anchor,
        receipt_key=keys.production_receipt_key,
        expected_receipt_key_id=system.production_receipt_key_id,
    )
    if (
        loaded.authenticated_outer_receipt != evidence.production_run
        or loaded.clinical_guest_bootstrap != evidence.bootstrap
        or loaded.guest_rpc_session != evidence.guest_rpc
        or loaded.gateway_session != evidence.gateway_session
        or loaded.worker_attestation != evidence.worker_attestation
        or loaded.submission != evidence.submission
        or loaded.authenticated_outer_receipt_sha256 != evidence.task_record.evidence_sha256
    ):
        raise ValueError('managed drill production components differ from persisted authenticated bytes')
    binding = next(item for item in evidence.reservation.tasks if item.episode_id == evidence.task_record.episode_id)
    if (
        workspace.authenticated_receipt_sha256 != binding.authenticated_workspace_receipt_sha256
        or workspace.manifest_sha256 != binding.workspace_manifest_sha256
        or workspace.manifest.workspace_tree_sha256 != binding.workspace_tree_sha256
        or workspace.manifest.model_visible_surface_sha256 != binding.model_visible_surface_sha256
    ):
        raise ValueError('managed drill persisted workspace differs from its reservation binding')
    return workspace


def _verify_ownership_and_host_cleanup(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    keys: ManagedClinicalRealKvmVerifierKeys,
) -> None:
    ledger_root = Path(evidence.ownership_config.ledger_root)
    _require_private_directory(ledger_root, label='managed ownership ledger')
    ledger = DurableManagedClinicalOwnershipLedger(
        config=evidence.ownership_config,
        ownership_key=keys.ownership_key,
    )
    loaded_chain = ledger.chain(evidence.run_id)
    if loaded_chain != evidence.ownership_chain:
        raise ValueError('managed drill ownership chain differs from persisted authenticated bytes')
    if ledger.run_ids() != (evidence.run_id,):
        raise ValueError('managed drill ownership namespace contains another run')
    if ledger.active():
        raise ValueError('managed drill ownership ledger still contains an active run')
    running = loaded_chain[3].record
    if (
        authenticated_managed_clinical_ownership_sha256(loaded_chain[3])
        != evidence.live_process_observation.ownership_envelope_sha256
        or running.firecracker_executable_sha256 != evidence.worker_spec.runtime.firecracker.sha256
    ):
        raise ValueError('managed drill live process observation differs from verified ownership')
    try:
        kvm = Path(evidence.live_process_observation.kvm_device_path).lstat()
    except OSError:
        raise ValueError('managed drill KVM character device is unavailable') from None
    if not stat.S_ISCHR(kvm.st_mode) or (
        kvm.st_dev,
        kvm.st_ino,
        kvm.st_rdev,
    ) != (
        evidence.live_process_observation.kvm_device_id,
        evidence.live_process_observation.kvm_device_inode,
        evidence.live_process_observation.kvm_device_rdev,
    ):
        raise ValueError('managed drill KVM character device differs from the live observation')

    host = LinuxManagedClinicalHostAdapter(
        config=evidence.startup_config,
        ownership=ledger,
        ownership_key=keys.ownership_key,
    )
    remaining = (
        len(host.scan_process_groups()),
        len(host.scan_cgroups()),
        len(host.scan_jail_roots()),
        len(host.scan_vsock_endpoints()),
    )
    if remaining != (0, 0, 0, 0):
        raise ValueError('managed drill host still contains an owned process or artifact')


def _verify_startup_cleanups(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    keys: ManagedClinicalRealKvmVerifierKeys,
) -> None:
    receipt_root = Path(evidence.startup_config.receipt_root)
    _require_private_directory(receipt_root, label='managed startup cleanup receipts')
    retained_paths = {Path(item.authenticated_cleanup.persisted_path) for item in evidence.startup_cleanups}
    persisted_paths: set[Path] = set()
    try:
        with os.scandir(receipt_root) as entries:
            for entry in entries:
                if len(persisted_paths) >= len(retained_paths):
                    raise ValueError('managed startup cleanup namespace exceeds its exact receipt bound')
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError('managed startup cleanup namespace contains a non-regular entry')
                persisted_paths.add(Path(entry.path))
    except OSError:
        raise ValueError('managed startup cleanup receipt inventory is unavailable') from None
    if retained_paths != persisted_paths:
        raise ValueError('managed startup cleanup namespace contains another receipt')
    expected_static = (
        firecracker_clinical_runtime_config_sha256(evidence.operator_manifest.runtime_config),
        agentic_policy_sha256(evidence.operator_manifest.execution_policy),
        evidence.worker_spec_sha256,
        authenticated_gateway_policy_sha256(evidence.operator_manifest.gateway_policy),
        gateway_model_route_sha256(evidence.operator_manifest.gateway_route),
        evidence.operator_manifest.runtime_config.bootstrap_authorization_key_id,
        evidence.reservation.system.guest_bootstrap_receipt_key_id,
    )
    times: list[tuple[datetime, datetime]] = []
    for phase in evidence.startup_cleanups:
        artifact = phase.authenticated_cleanup
        request = artifact.reconciliation_request
        if (
            artifact.config_sha256 != evidence.startup_config_sha256
            or artifact.request_sha256 != firecracker_clinical_startup_reconciliation_request_sha256(request)
            or (
                request.runtime_config_sha256,
                request.execution_policy_sha256,
                request.worker_spec_sha256,
                request.gateway_policy_sha256,
                request.gateway_route_sha256,
                request.bootstrap_authorization_key_id,
                request.bootstrap_receipt_key_id,
            )
            != expected_static
        ):
            raise ValueError('managed drill startup cleanup differs from static execution pins')
        loaded = load_authenticated_managed_cleanup(
            Path(artifact.persisted_path),
            expected_root=Path(evidence.startup_config.receipt_root),
        )
        if loaded != artifact:
            raise ValueError('managed drill startup cleanup differs from create-once persisted bytes')
        receipt = verify_authenticated_managed_cleanup(
            loaded,
            key=keys.startup_cleanup_key,
            expected_key_id=evidence.startup_config.cleanup_receipt_key_id,
            expected_config_sha256=evidence.startup_config_sha256,
            expected_request_sha256=artifact.request_sha256,
        )
        if (
            receipt != artifact.cleanup_receipt
            or receipt.reconciliation_request_sha256 != artifact.request_sha256
            or receipt.retained_journal_count != len(request.retained_journals)
            or receipt.worker_inventory_sha256 != artifact.process_group_inventory_sha256
            or receipt.capability_inventory_sha256 != artifact.capability_inventory_sha256
            or receipt.attempt_registry_inventory_sha256 != artifact.attempt_inventory_sha256
            or request.requested_at > receipt.reconciled_at
        ):
            raise ValueError('managed drill startup cleanup receipt is misbound or time-inconsistent')
        times.append((request.requested_at, receipt.reconciled_at))

    if not (
        times[0][0] <= times[0][1] < times[1][0] <= times[1][1] < times[2][0] <= times[2][1] <= evidence.collected_at
    ):
        raise ValueError('managed drill startup cleanup phases are not chronologically distinct')
    launch = evidence.task_record.launch
    terminal_at = evidence.task_record.terminal_at
    if launch is None or terminal_at is None:
        raise ValueError('managed drill task lacks launch or terminal time')
    retry_payload = ManagedClaimRequest.model_validate_json(
        canonical_json_bytes(evidence.registry_observation.retry_claim_audit.request.payload)
    )
    if not (
        times[0][1] <= launch.claimed_at
        and times[1][1] <= launch.claimed_at
        and terminal_at <= times[2][0] <= times[2][1] <= retry_payload.claimed_at
    ):
        raise ValueError('managed drill startup phases do not surround launch and retry correctly')


def _verify_gateway_tombstone(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    keys: ManagedClinicalRealKvmVerifierKeys,
) -> None:
    if evidence.gateway_ledger_identity.resolved_path != (evidence.operator_manifest.gateway_ledger_path):
        raise ValueError('managed drill gateway ledger path differs from the operator manifest')
    gateway = SqliteGatewayLedger(Path(evidence.gateway_ledger_identity.resolved_path))
    if gateway.identity != evidence.gateway_ledger_identity:
        raise ValueError('managed drill gateway ledger inode differs after reopening')
    capability_id = evidence.gateway_session.grant.capability_id
    if (
        gateway.capability_revocation(capability_id) != evidence.gateway_revocation
        or gateway.capability_binding(capability_id) != gateway_capability_binding(evidence.gateway_session.grant)
        or any(item.capability_id == capability_id for item in gateway.unrevoked_capability_bindings())
    ):
        raise ValueError('managed drill gateway tombstone is absent, stale, or misbound')
    ownership = DurableManagedClinicalOwnershipLedger(
        config=evidence.ownership_config,
        ownership_key=keys.ownership_key,
    )
    capabilities = RestartVisibleManagedGatewayCapabilityLedger(
        ownership=ownership,
        ownership_key=keys.ownership_key,
        gateway_ledger=gateway,
        expected_model_route_sha256=gateway_model_route_sha256(evidence.operator_manifest.gateway_route),
    )
    if capabilities.inventory():
        raise ValueError('managed drill still has an unrevoked managed capability')


def _verify_registry_observation(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    keys: ManagedClinicalRealKvmVerifierKeys,
    external_pins: ManagedClinicalRealKvmExternalPins,
) -> None:
    observation = evidence.registry_observation
    chain = load_authenticated_managed_registry_audit_chain(
        Path(evidence.registry_config.protocol_audit_root),
        key=keys.startup_cleanup_key,
        expected_key_id=evidence.registry_config.startup_cleanup_receipt_key_id,
        expected_config_sha256=evidence.registry_config_sha256,
    )
    by_request_id = {item.request.request_id: item for item in chain}
    if (
        not chain
        or any(
            item.registry_authority_id != evidence.registry_config.registry_authority_id
            or item.audited_at < external_pins.challenge_issued_at
            for item in chain
        )
        or tuple(item.audited_at for item in chain) != tuple(sorted(item.audited_at for item in chain))
    ):
        raise ValueError('managed drill registry audit predates or differs from its challenge authority')
    selected = (observation.record_run_audit, observation.retry_claim_audit)
    if len(by_request_id) != len(chain) or any(by_request_id.get(item.request.request_id) != item for item in selected):
        raise ValueError('managed drill selected registry audit is absent from the verified chain')
    record_audit, retry_audit = selected
    if (
        record_audit.request.operation != 'record_run'
        or not record_audit.response.ok
        or record_audit.response.result is None
        or record_audit.response.error_code is not None
        or retry_audit.request.operation != 'claim'
        or retry_audit.response.ok
        or retry_audit.response.result is not None
        or retry_audit.response.error_code != 'rejected'
        or record_audit.sequence >= retry_audit.sequence
        or record_audit.audited_at > retry_audit.audited_at
    ):
        raise ValueError('managed drill audit does not prove success followed by retry denial')

    record_request = ManagedRecordRunRequest.model_validate_json(canonical_json_bytes(record_audit.request.payload))
    retry_request = ManagedClaimRequest.model_validate_json(canonical_json_bytes(retry_audit.request.payload))
    record_result = ClinicalProductionTaskRecord.model_validate_json(
        canonical_json_bytes(record_audit.response.result['task_record'])
    )
    terminal_at = evidence.task_record.terminal_at
    if terminal_at is None:
        raise ValueError('managed drill successful registry task lacks terminal time')
    if (
        record_result != evidence.task_record
        or record_request.reservation_sha256 != evidence.reservation_sha256
        or record_request.episode_id != evidence.task_record.episode_id
        or record_request.production_run_root != evidence.production_run_root
        or record_request.terminal_at != terminal_at
        or retry_request.reservation_sha256 != evidence.reservation_sha256
        or retry_request.episode_id != evidence.task_record.episode_id
        or retry_request.run_id == evidence.run_id
        or retry_request.claimed_at < terminal_at
        or record_request.terminal_at > record_audit.audited_at
        or retry_request.claimed_at > retry_audit.audited_at
    ):
        raise ValueError('managed drill registry exchanges differ from the exact terminal attempt')

    expected_task_sha256 = _model_sha256(evidence.task_record)
    if (
        observation.terminal_task_record_before_retry_sha256 != expected_task_sha256
        or observation.terminal_task_record_after_retry_sha256 != expected_task_sha256
    ):
        raise ValueError('managed drill retry changed the authoritative terminal task record')
    _verify_registry_server_restart(evidence, record_audit, retry_audit)
    _verify_retry_reconciliation_audits(
        evidence,
        chain=chain,
        record_audit=record_audit,
        retry_audit=retry_audit,
    )

    database = Path(evidence.registry_config.database_path)
    try:
        metadata_before = database.lstat()
    except OSError:
        raise ValueError('managed drill authoritative registry database is unavailable') from None
    registry = SqliteClinicalProductionRegistry(
        database,
        authority_id=evidence.registry_config.registry_authority_id,
    )
    context = registry.reservation_context(evidence.reservation_sha256)
    reservation_hashes = registry.reservation_hashes()
    records = tuple(
        item
        for item in registry.task_records(evidence.reservation_sha256)
        if item.episode_id == evidence.task_record.episode_id
    )
    try:
        metadata_after = database.lstat()
    except OSError:
        raise ValueError('managed drill authoritative registry database disappeared') from None
    if (
        not stat.S_ISREG(metadata_before.st_mode)
        or not stat.S_ISREG(metadata_after.st_mode)
        or metadata_before.st_uid != 0
        or metadata_after.st_uid != 0
        or metadata_before.st_nlink != 1
        or metadata_after.st_nlink != 1
        or stat.S_IMODE(metadata_before.st_mode) != 0o600
        or stat.S_IMODE(metadata_after.st_mode) != 0o600
        or context.reservation != evidence.reservation
        or reservation_hashes != (evidence.reservation_sha256,)
        or len(records) != 1
        or records[0] != evidence.task_record
        or (metadata_before.st_dev, metadata_before.st_ino) != (metadata_after.st_dev, metadata_after.st_ino)
        or (metadata_after.st_dev, metadata_after.st_ino)
        != (
            retry_audit.server.database_device_id,
            retry_audit.server.database_inode,
        )
    ):
        raise ValueError('managed drill authoritative registry reload differs from retained evidence')


def _verify_registry_server_restart(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    record_audit: AuthenticatedManagedClinicalRegistryAudit,
    retry_audit: AuthenticatedManagedClinicalRegistryAudit,
) -> None:
    config = evidence.registry_config
    expected_peer = (
        config.allowed_launcher_uid,
        config.allowed_launcher_gid,
        config.canonical_launcher_id,
        config.canonical_launcher_executable_sha256,
    )
    if any(
        (
            item.launcher_peer.uid,
            item.launcher_peer.gid,
            item.launcher_peer.canonical_launcher_id,
            item.launcher_peer.canonical_launcher_executable_sha256,
        )
        != expected_peer
        for item in (record_audit, retry_audit)
    ):
        raise ValueError('managed drill registry audit has an unauthorized launcher peer')
    if any(item.launcher_peer.pid != item.server.service_pid for item in (record_audit, retry_audit)):
        raise ValueError('managed drill registry launcher and service are not one fixed process')
    expected_server = (
        0,
        0,
        config.service_process_executable_sha256,
        config.socket_path,
        config.database_path,
    )
    if any(
        (
            item.server.service_uid,
            item.server.service_gid,
            item.server.service_executable_sha256,
            item.server.socket_path,
            item.server.database_path,
        )
        != expected_server
        for item in (record_audit, retry_audit)
    ):
        raise ValueError('managed drill registry audit has a different server or authority path')
    if (
        (
            record_audit.server.database_device_id,
            record_audit.server.database_inode,
        )
        != (
            retry_audit.server.database_device_id,
            retry_audit.server.database_inode,
        )
        or (
            record_audit.server.service_pid,
            record_audit.server.service_start_time_ticks,
        )
        == (
            retry_audit.server.service_pid,
            retry_audit.server.service_start_time_ticks,
        )
        or (
            record_audit.server.socket_device_id,
            record_audit.server.socket_inode,
        )
        == (
            retry_audit.server.socket_device_id,
            retry_audit.server.socket_inode,
        )
    ):
        raise ValueError('managed drill registry retry was not observed after an authority restart')


def _verify_retry_reconciliation_audits(
    evidence: ManagedClinicalRealKvmDrillEvidence,
    *,
    chain: tuple[AuthenticatedManagedClinicalRegistryAudit, ...],
    record_audit: AuthenticatedManagedClinicalRegistryAudit,
    retry_audit: AuthenticatedManagedClinicalRegistryAudit,
) -> None:
    between = tuple(
        item
        for item in chain
        if record_audit.sequence < item.sequence < retry_audit.sequence and item.server == retry_audit.server
    )
    begins = tuple(
        item
        for item in between
        if item.request.operation == 'begin_reconciliation'
        and item.response.ok
        and item.response.result is not None
        and item.response.error_code is None
    )
    finishes = tuple(
        item
        for item in between
        if item.request.operation == 'finish_reconciliation'
        and item.response.ok
        and item.response.result is not None
        and item.response.error_code is None
    )
    if len(begins) != 1 or len(finishes) != 1:
        raise ValueError('managed drill retry authority lacks one authenticated successful reconciliation')
    begin = begins[0]
    finish = finishes[0]
    if (
        not record_audit.sequence < begin.sequence < finish.sequence < retry_audit.sequence
        or not record_audit.audited_at <= begin.audited_at <= finish.audited_at <= retry_audit.audited_at
        or begin.launcher_peer != retry_audit.launcher_peer
        or finish.launcher_peer != retry_audit.launcher_peer
    ):
        raise ValueError('managed drill retry reconciliation is not bound to the restarted authority')

    begin_request = ManagedBeginReconciliationRequest.model_validate_json(canonical_json_bytes(begin.request.payload))
    finish_request = ManagedFinishReconciliationRequest.model_validate_json(
        canonical_json_bytes(finish.request.payload)
    )
    third_cleanup = evidence.startup_cleanups[2].authenticated_cleanup
    begin_result = begin.response.result
    finish_result = finish.response.result
    assert begin_result is not None and finish_result is not None
    attempts = tuple(
        ManagedClinicalAttemptInventoryRecord.model_validate(item) for item in begin_result.get('attempts', ())
    )
    if attempts != tuple(sorted(attempts, key=lambda item: (item.run_id, item.reservation_sha256))):
        raise ValueError('managed drill retry reconciliation attempt inventory is not canonical')
    inventory_sha256 = hashlib.sha256(
        canonical_json_bytes([item.model_dump(mode='json') for item in attempts])
    ).hexdigest()
    record = evidence.task_record
    launch = record.launch
    if launch is None or record.start_redemption_sha256 is None:
        raise ValueError('managed drill retry reconciliation lacks a redeemed task')
    expected_attempt = ManagedClinicalAttemptInventoryRecord(
        registry_authority_id=evidence.reservation.registry_authority_id,
        reservation_sha256=evidence.reservation_sha256,
        launch_sha256=clinical_production_task_launch_sha256(launch),
        start_redemption_sha256=record.start_redemption_sha256,
        run_id=evidence.run_id,
        episode_id=record.episode_id,
        worker_spec_sha256=evidence.worker_spec_sha256,
        state='succeeded',
    )
    if (
        attempts != (expected_attempt,)
        or begin_request.startup_config_sha256 != evidence.startup_config_sha256
        or begin_request.cleanup_receipt_key_id != evidence.startup_config.cleanup_receipt_key_id
        or begin_request.requested_at > begin.audited_at
        or begin_result.get('mutations_quiesced') is not True
        or begin_result.get('service_restart_recloses_authority') is not True
        or begin_result.get('attempt_inventory_sha256') != inventory_sha256
        or finish_request.authenticated_cleanup != third_cleanup
        or finish_request.lease_token != begin_result.get('lease_token')
        or finish_result.get('startup_reconciliation_admitted') is not True
        or finish_result.get('mutations_quiesced') is not False
        or finish_result.get('attempt_inventory_sha256') != inventory_sha256
        or third_cleanup.attempt_inventory_sha256 != inventory_sha256
        or third_cleanup.cleanup_receipt.attempt_registry_inventory_sha256 != inventory_sha256
    ):
        raise ValueError('managed drill retry denial is not downstream of the exact successful cleanup')


def _require_private_directory(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f'{label} cannot be a symbolic link')
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError:
        raise ValueError(f'{label} is unavailable') from None
    if (
        os.uname().sysname == 'Linux'
        and resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError(f'{label} must be one private owned mode-0700 directory')


def _read_safe_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise ValueError('managed drill pinned file cannot be a symbolic link')
    try:
        resolved = path.resolve(strict=True)
        before = resolved.lstat()
    except OSError:
        raise ValueError('managed drill pinned file is unavailable') from None
    if resolved != path:
        raise ValueError('managed drill pinned file path contains a symbolic-link component')
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != 0
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o022
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise ValueError('managed drill pinned file has unsafe metadata')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError:
        raise ValueError('managed drill pinned file cannot be opened safely') from None
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('managed drill pinned file changed while opening')
        body = bytearray()
        while len(body) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(body) > maximum_bytes:
        raise ValueError('managed drill pinned file exceeds its byte limit')
    try:
        after = resolved.lstat()
    except OSError:
        raise ValueError('managed drill pinned file disappeared while reading') from None
    stable_fields = (
        'st_dev',
        'st_ino',
        'st_mode',
        'st_uid',
        'st_gid',
        'st_nlink',
        'st_size',
        'st_mtime_ns',
        'st_ctime_ns',
    )
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise ValueError('managed drill pinned file changed while reading')
    return bytes(body)


__all__ = [
    'AUTHENTICATED_MANAGED_CLINICAL_REAL_KVM_DRILL_SCHEMA_VERSION',
    'MANAGED_CLINICAL_REAL_KVM_DRILL_SCHEMA_VERSION',
    'AuthenticatedManagedClinicalRealKvmDrill',
    'ManagedClinicalRealKvmExternalPins',
    'ManagedClinicalRealKvmObservationGateRelease',
    'ManagedClinicalRealKvmProcessObservation',
    'ManagedClinicalRealKvmDrillEvidence',
    'ManagedClinicalRealKvmVerifierKeys',
    'ManagedClinicalRegistryDrillObservation',
    'ManagedClinicalStartupCleanupDrillObservation',
    'authenticate_managed_clinical_real_kvm_drill',
    'independently_verify_authenticated_managed_clinical_real_kvm_drill',
    'managed_clinical_real_kvm_authority_id',
    'managed_clinical_real_kvm_challenge_sha256',
    'managed_clinical_real_kvm_collector_key_id',
    'managed_clinical_real_kvm_deployment_id',
    'managed_clinical_real_kvm_drill_sha256',
    'managed_clinical_real_kvm_registered_entry_id',
    'managed_clinical_real_kvm_release_pins_sha256',
    'verify_authenticated_managed_clinical_real_kvm_drill',
    'verify_managed_clinical_real_kvm_drill_from_persisted_state',
]
