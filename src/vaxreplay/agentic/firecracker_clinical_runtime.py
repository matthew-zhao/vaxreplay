"""Concrete development runtime behind the canonical Lane A launcher.

This module owns the host-side ordering between a prepared Firecracker worker, the one-time
start-redemption, a short-lived provider capability, the signed clinical guest bootstrap, the
guest-RPC and gateway seals, worker teardown/attestation, and the versioned outer evidence package.

It is deliberately development-only.  The real supervisor rejects hosts which do not satisfy its
Linux/KVM preflight, but this composition has not itself passed the separately versioned runtime
qualification suite.  In particular, a signed bootstrap acknowledgement is not remote guest
attestation and does not prove that the pinned public-key trust anchor was baked into the launched
image.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import secrets
import socket
import stat
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_execution_bridge import (
    clinical_workspace_receipt_key_id,
)
from vaxreplay.agentic.clinical_guest_bootstrap import (
    CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
    CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
    AuthenticatedClinicalGuestBootstrap,
    ClinicalGuestBootstrapHello,
    ClinicalGuestBootstrapTrustAnchor,
    ClinicalGuestRpcLimits,
    clinical_guest_bootstrap_authorization_key_id,
    clinical_guest_bootstrap_receipt_key_id,
    perform_host_clinical_guest_bootstrap,
    verify_authenticated_clinical_guest_bootstrap,
)
from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_ACTION_SCHEMA_SHA256,
    LANE_A_GUEST_HARNESS_POLICY_ID,
)
from vaxreplay.agentic.clinical_launcher import (
    ClinicalPreparedRuntime,
    ClinicalRuntimeBoundary,
    ClinicalRuntimeCompleted,
    ClinicalRuntimeFailed,
    ClinicalRuntimeFailureCode,
    ClinicalRuntimeOutcome,
    ClinicalRuntimePrepareRequest,
    ClinicalRuntimeStart,
    canonical_clinical_launcher_deployment_sha256,
    clinical_prepared_runtime_sha256,
)
from vaxreplay.agentic.clinical_production_registry import (
    clinical_production_start_redemption_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.clinical_production_run import clinical_production_run_key_id
from vaxreplay.agentic.firecracker import (
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerCleanupReceipt,
    FirecrackerPreboundGuestListener,
    FirecrackerPreparedWorker,
    FirecrackerSupervisor,
    FirecrackerWorkerSpec,
    RunningFirecrackerWorker,
    capture_firecracker_prebound_guest_listener,
    finalize_firecracker_worker_attestation,
    firecracker_attestation_key_id,
    firecracker_guest_bootstrap_profile_sha256,
    firecracker_guest_initiated_uds_path,
    firecracker_model_sha256,
)
from vaxreplay.agentic.gateway_auth import (
    GATEWAY_CAPABILITY_SECRET_BYTES,
    GatewaySecretResolver,
    gateway_capability_id,
)
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcErrorCode,
    GuestRpcHostServer,
    GuestRpcHostSession,
    GuestRpcPolicy,
    GuestRpcTerminalStatus,
    guest_rpc_policy_sha256,
    guest_rpc_session_key_id,
)
from vaxreplay.agentic.protocol import AgenticExecutionPolicy, agentic_policy_sha256
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewaySession,
    AuthenticatedProviderGateway,
    GatewayCapabilityRevocationReason,
    GatewayModelRoute,
    GatewayTerminalReason,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    gateway_session_key_id,
    issue_gateway_capability,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.run_artifact import AgenticHarnessIdentity
from vaxreplay.agentic.task_protocol import agentic_task_invocation_sha256
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission
from vaxreplay.operations.signing import Ed25519Signer, checked_signer
from vaxreplay.runner.schema import IsolationTier

FIRECRACKER_CLINICAL_RUNTIME_CONFIG_SCHEMA_VERSION = 'vaxreplay.firecracker-clinical-runtime-config.dev-v0.1'
FIRECRACKER_CLINICAL_STARTUP_ORPHAN_SCHEMA_VERSION = 'vaxreplay.firecracker-clinical-startup-orphan.dev-v0.1'
FIRECRACKER_CLINICAL_STARTUP_RECONCILIATION_REQUEST_SCHEMA_VERSION = (
    'vaxreplay.firecracker-clinical-startup-reconciliation-request.dev-v0.1'
)
FIRECRACKER_CLINICAL_STARTUP_CLEANUP_RECEIPT_SCHEMA_VERSION = (
    'vaxreplay.firecracker-clinical-startup-cleanup-receipt.dev-v0.1'
)
FIRECRACKER_CLINICAL_STARTUP_RECONCILIATION_REPORT_SCHEMA_VERSION = (
    'vaxreplay.firecracker-clinical-startup-reconciliation-report.dev-v0.1'
)
FIRECRACKER_CLINICAL_RECOVERY_RECONCILIATION_REPORT_SCHEMA_VERSION = (
    'vaxreplay.firecracker-clinical-recovery-reconciliation-report.dev-v0.1'
)

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
_MAX_BOOTSTRAP_JOURNAL_BYTES = 2 * CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES


class FirecrackerClinicalRuntimeError(RuntimeError):
    """A stable host-composition error which never includes task or provider content."""


class FirecrackerClinicalRuntimeConfig(StrictModel):
    """Deployment-pinned identity and bounded host behavior for this development runtime."""

    schema_version: Literal['vaxreplay.firecracker-clinical-runtime-config.dev-v0.1'] = (
        FIRECRACKER_CLINICAL_RUNTIME_CONFIG_SCHEMA_VERSION
    )
    runtime_id: str = Field(pattern=_ID_PATTERN)
    runtime_version: str = Field(min_length=1, max_length=200)
    runtime_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_authorization_key_id: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_connection_timeout_seconds: float = Field(gt=0, le=30)
    bootstrap_validity_seconds: int = Field(gt=0, le=300)
    cleanup_grace_seconds: float = Field(gt=0, le=60)
    one_worker_launch: Literal[True] = True
    automatic_worker_retry: Literal[False] = False
    automatic_provider_retry: Literal[False] = False
    strict_signed_clinical_bootstrap_required: Literal[True] = True
    outer_v02_bootstrap_binding_required: Literal[True] = True
    development_only: Literal[True] = True
    linux_kvm_runtime_qualified: Literal[False] = False
    official_execution_qualified: Literal[False] = False


def firecracker_clinical_runtime_config_sha256(config: FirecrackerClinicalRuntimeConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


@dataclass(frozen=True, slots=True)
class FirecrackerClinicalRuntimeKeys:
    """Host-only authentication material; none of these bytes enter retained JSON."""

    workspace_receipt_key: bytes
    worker_attestation_key: bytes
    gateway_receipt_key: bytes
    guest_rpc_receipt_key: bytes
    clinical_guest_bootstrap_receipt_key: bytes
    production_receipt_key: bytes


class GatewayCapabilitySecretStore(GatewaySecretResolver, Protocol):
    def register(self, secret: bytes) -> str: ...

    def revoke(self, capability_id: str) -> None: ...


class FirecrackerClinicalStartupOrphan(StrictModel):
    """Authenticated journal projection supplied to a deployment orphan-discovery boundary.

    The bootstrap journal does not retain the provider capability ID or a live process handle.
    Discovery therefore has to use the signed run/start binding and deployment-owned host state.
    """

    schema_version: Literal['vaxreplay.firecracker-clinical-startup-orphan.dev-v0.1'] = (
        FIRECRACKER_CLINICAL_STARTUP_ORPHAN_SCHEMA_VERSION
    )
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    bootstrap_journal_file_name: str = Field(pattern=r'^[0-9a-f]{32}\.json$')
    bootstrap_journal_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_redemption_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_session_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_authorization_key_id: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_ack_received_at: datetime
    capability_id_available_in_bootstrap_journal: Literal[False] = False
    worker_discovery_required: Literal[True] = True
    capability_discovery_required: Literal[True] = True
    retained_journal_must_not_be_deleted: Literal[True] = True

    @field_validator('bootstrap_ack_received_at')
    @classmethod
    def validate_ack_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('bootstrap acknowledgement timestamp must include a UTC offset')
        return value.astimezone(UTC)


def firecracker_clinical_startup_orphan_sha256(orphan: FirecrackerClinicalStartupOrphan) -> str:
    return hashlib.sha256(canonical_json_bytes(orphan)).hexdigest()


class FirecrackerClinicalStartupReconciliationRequest(StrictModel):
    """Complete host/capability inventory request, including any authenticated journals."""

    schema_version: Literal['vaxreplay.firecracker-clinical-startup-reconciliation-request.dev-v0.1'] = (
        FIRECRACKER_CLINICAL_STARTUP_RECONCILIATION_REQUEST_SCHEMA_VERSION
    )
    runtime_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_route_sha256: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_authorization_key_id: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    retained_journals: tuple[FirecrackerClinicalStartupOrphan, ...]
    requested_at: datetime
    surviving_workers_expected: Literal[0] = 0
    surviving_capabilities_expected: Literal[0] = 0
    complete_host_worker_inventory_required: Literal[True] = True
    complete_host_ephemeral_run_artifact_inventory_required: Literal[True] = True
    complete_gateway_capability_inventory_required: Literal[True] = True
    authoritative_attempt_registry_inventory_required: Literal[True] = True
    unjournaled_orphan_discovery_required: Literal[True] = True
    retained_journal_deletion_forbidden: Literal[True] = True

    @field_validator('requested_at')
    @classmethod
    def validate_requested_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('startup reconciliation request timestamp must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_journal_order(self) -> Self:
        run_ids = tuple(orphan.run_id for orphan in self.retained_journals)
        if run_ids != tuple(sorted(set(run_ids))):
            raise ValueError('startup reconciliation journals require unique canonical run order')
        return self


def firecracker_clinical_startup_reconciliation_request_sha256(
    request: FirecrackerClinicalStartupReconciliationRequest,
) -> str:
    return hashlib.sha256(canonical_json_bytes(request)).hexdigest()


class FirecrackerClinicalStartupCleanupReceipt(StrictModel):
    """Deployment-adapter assertion about one complete host startup inventory.

    This is a typed trust boundary, not repository-generated live-process proof.  A production
    adapter must derive the inventory digests from its actual process/cgroup/jail and capability
    stores.  The runtime validates cross-bindings and completeness but cannot recreate that state.
    """

    schema_version: Literal['vaxreplay.firecracker-clinical-startup-cleanup-receipt.dev-v0.1'] = (
        FIRECRACKER_CLINICAL_STARTUP_CLEANUP_RECEIPT_SCHEMA_VERSION
    )
    reconciler_id: str = Field(pattern=_ID_PATTERN)
    reconciler_version: str = Field(min_length=1, max_length=200)
    reconciliation_request_sha256: str = Field(pattern=_SHA256_PATTERN)
    retained_journal_count: int = Field(ge=0, le=1_000_000)
    worker_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    ephemeral_run_artifact_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    capability_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_registry_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    cleanup_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    discovered_worker_count: int = Field(ge=0, le=1_000_000)
    terminated_worker_count: int = Field(ge=0, le=1_000_000)
    discovered_ephemeral_run_artifact_count: int = Field(ge=0, le=1_000_000)
    removed_ephemeral_run_artifact_count: int = Field(ge=0, le=1_000_000)
    discovered_capability_count: int = Field(ge=0, le=1_000_000)
    revoked_capability_count: int = Field(ge=0, le=1_000_000)
    reconciled_at: datetime
    worker_discovery_complete: Literal[True] = True
    ephemeral_run_artifact_discovery_complete: Literal[True] = True
    capability_discovery_complete: Literal[True] = True
    authoritative_attempt_registry_reconciliation_complete: Literal[True] = True
    unjournaled_orphan_discovery_complete: Literal[True] = True
    every_discovered_worker_absent_or_terminated: Literal[True] = True
    every_discovered_ephemeral_run_artifact_absent_or_removed: Literal[True] = True
    every_discovered_capability_absent_or_revoked: Literal[True] = True
    every_retained_journal_accounted_for_in_attempt_registry: Literal[True] = True
    run_scoped_cleanup_idempotent: Literal[True] = True
    retained_bootstrap_journal_preserved: Literal[True] = True
    live_state_assertions_supplied_by_deployment_adapter: Literal[True] = True
    repository_inferred_live_process_or_capability_state: Literal[False] = False
    development_only: Literal[True] = True
    linux_kvm_cleanup_qualified: Literal[False] = False
    official_execution_qualified: Literal[False] = False

    @field_validator('reconciled_at')
    @classmethod
    def validate_reconciled_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('startup reconciliation timestamp must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_complete_cleanup(self) -> Self:
        if self.terminated_worker_count != self.discovered_worker_count:
            raise ValueError('startup cleanup must terminate every discovered worker')
        if self.removed_ephemeral_run_artifact_count != self.discovered_ephemeral_run_artifact_count:
            raise ValueError('startup cleanup must remove every discovered ephemeral run artifact')
        if self.revoked_capability_count != self.discovered_capability_count:
            raise ValueError('startup cleanup must revoke every discovered capability')
        return self


class FirecrackerClinicalStartupReconciler(Protocol):
    """Deployment-specific live worker/capability discovery and cleanup boundary."""

    def reconcile(
        self,
        request: FirecrackerClinicalStartupReconciliationRequest,
    ) -> FirecrackerClinicalStartupCleanupReceipt: ...


class FirecrackerClinicalStartupReconciliationReport(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-clinical-startup-reconciliation-report.dev-v0.1'] = (
        FIRECRACKER_CLINICAL_STARTUP_RECONCILIATION_REPORT_SCHEMA_VERSION
    )
    request: FirecrackerClinicalStartupReconciliationRequest
    cleanup_receipt: FirecrackerClinicalStartupCleanupReceipt
    startup_admission_allowed: Literal[True] = True
    all_retained_journals_authenticated: Literal[True] = True
    retained_bootstrap_journals_preserved: Literal[True] = True
    deployment_cleanup_adapter_invoked: Literal[True] = True
    repository_inferred_live_process_or_capability_state: Literal[False] = False
    deployment_receipts_are_independent_host_attestation: Literal[False] = False
    cleanup_receipt_cryptographically_authenticated: Literal[False] = False
    cleanup_adapter_identity_pinned_by_runtime_config: Literal[False] = False
    pre_start_journal_deletion_excluded_by_repository: Literal[False] = False
    reconciliation_report_persisted_by_runtime: Literal[False] = False
    development_only: Literal[True] = True
    linux_kvm_cleanup_qualified: Literal[False] = False
    official_execution_qualified: Literal[False] = False

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        if (
            self.cleanup_receipt.retained_journal_count != len(self.request.retained_journals)
            or self.cleanup_receipt.reconciliation_request_sha256
            != firecracker_clinical_startup_reconciliation_request_sha256(self.request)
            or self.cleanup_receipt.reconciled_at < self.request.requested_at
        ):
            raise ValueError('startup report receipt differs from its retained journal inventory')
        return self


class FirecrackerClinicalRecoveryReconciliationReport(StrictModel):
    """Cleanup proof from the deliberately non-runnable recovery composition."""

    schema_version: Literal['vaxreplay.firecracker-clinical-recovery-reconciliation-report.dev-v0.1'] = (
        FIRECRACKER_CLINICAL_RECOVERY_RECONCILIATION_REPORT_SCHEMA_VERSION
    )
    request: FirecrackerClinicalStartupReconciliationRequest
    cleanup_receipt: FirecrackerClinicalStartupCleanupReceipt
    cleanup_only: Literal[True] = True
    all_retained_journals_authenticated: Literal[True] = True
    retained_bootstrap_journals_preserved: Literal[True] = True
    deployment_cleanup_adapter_invoked: Literal[True] = True
    runtime_preparation_admitted: Literal[False] = False
    supervisor_or_worker_launch_available: Literal[False] = False
    provider_or_model_call_available: Literal[False] = False
    recovery_receipt_is_live_host_qualification: Literal[False] = False

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        if (
            self.cleanup_receipt.retained_journal_count != len(self.request.retained_journals)
            or self.cleanup_receipt.reconciliation_request_sha256
            != firecracker_clinical_startup_reconciliation_request_sha256(self.request)
            or self.cleanup_receipt.reconciled_at < self.request.requested_at
        ):
            raise ValueError('recovery receipt differs from its retained journal inventory')
        return self


def reconcile_firecracker_clinical_startup_without_execution(
    *,
    config: FirecrackerClinicalRuntimeConfig,
    execution_policy_sha256: str,
    worker_spec: FirecrackerWorkerSpec,
    gateway_policy_sha256: str,
    gateway_route_sha256: str,
    guest_rpc_policy: GuestRpcPolicy,
    bootstrap_receipt_key: bytes,
    bootstrap_trust_anchor: ClinicalGuestBootstrapTrustAnchor,
    evidence_root: Path,
    reconciler: FirecrackerClinicalStartupReconciler,
    clock: Callable[[], datetime] | None = None,
) -> FirecrackerClinicalRecoveryReconciliationReport:
    """Authenticate retained journals and invoke cleanup without a runnable worker boundary.

    This recovery-only entrypoint intentionally accepts no supervisor, gateway, provider adapter,
    harness, signer, workspace, or model credential.  It can scan and authenticate the exact
    retained bootstrap-journal namespace and call the deployment reaper, but it has no method or
    dependency capable of preparing or launching a guest.
    """

    canonical_worker_spec = FirecrackerWorkerSpec.model_validate_json(canonical_json_bytes(worker_spec))
    worker_spec_sha256 = firecracker_model_sha256(canonical_worker_spec)
    worker_bootstrap_profile_sha256 = firecracker_guest_bootstrap_profile_sha256(canonical_worker_spec)
    for value, label in (
        (execution_policy_sha256, 'execution policy SHA-256'),
        (worker_spec_sha256, 'worker specification SHA-256'),
        (gateway_policy_sha256, 'gateway policy SHA-256'),
        (gateway_route_sha256, 'gateway route SHA-256'),
    ):
        _require_lower_hex(value, 64, label)
    if clinical_guest_bootstrap_receipt_key_id(bootstrap_receipt_key) != config.bootstrap_receipt_key_id:
        raise FirecrackerClinicalRuntimeError('recovery bootstrap receipt key differs from the runtime config')
    expected_limits = _clinical_guest_rpc_limits(guest_rpc_policy)
    if (
        bootstrap_trust_anchor.authorization_key_id,
        bootstrap_trust_anchor.execution_policy_sha256,
        bootstrap_trust_anchor.worker_bootstrap_profile_sha256,
        bootstrap_trust_anchor.harness_policy_id,
        bootstrap_trust_anchor.harness_policy_sha256,
        bootstrap_trust_anchor.action_schema_sha256,
        bootstrap_trust_anchor.rpc_limits,
    ) != (
        config.bootstrap_authorization_key_id,
        execution_policy_sha256,
        worker_bootstrap_profile_sha256,
        LANE_A_GUEST_HARNESS_POLICY_ID,
        CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
        LANE_A_GUEST_ACTION_SCHEMA_SHA256,
        expected_limits,
    ):
        raise FirecrackerClinicalRuntimeError('recovery trust anchor differs from the runtime static pins')
    now = clock or (lambda: datetime.now(UTC))
    journal_root = _prepare_private_root(_prepare_private_root(evidence_root) / 'bootstrap-journal')
    current = _scan_firecracker_clinical_bootstrap_journals(
        journal_root=journal_root,
        guest_rpc_policy=guest_rpc_policy,
        bootstrap_receipt_key=bootstrap_receipt_key,
        bootstrap_trust_anchor=bootstrap_trust_anchor,
        expected_authorization_key_id=config.bootstrap_authorization_key_id,
        expected_receipt_key_id=config.bootstrap_receipt_key_id,
        expected_execution_policy_sha256=execution_policy_sha256,
        expected_worker_spec_sha256=worker_spec_sha256,
    )
    requested_at = now()
    if requested_at.tzinfo is None or requested_at.utcoffset() is None:
        raise FirecrackerClinicalRuntimeError('recovery clock must return a timezone-aware timestamp')
    requested_at = requested_at.astimezone(UTC)
    if current:
        requested_at = max(
            requested_at,
            *(orphan.bootstrap_ack_received_at for orphan in current),
        )
    request = FirecrackerClinicalStartupReconciliationRequest(
        runtime_config_sha256=firecracker_clinical_runtime_config_sha256(config),
        execution_policy_sha256=execution_policy_sha256,
        worker_spec_sha256=worker_spec_sha256,
        gateway_policy_sha256=gateway_policy_sha256,
        gateway_route_sha256=gateway_route_sha256,
        bootstrap_authorization_key_id=config.bootstrap_authorization_key_id,
        bootstrap_receipt_key_id=config.bootstrap_receipt_key_id,
        retained_journals=current,
        requested_at=requested_at,
    )
    try:
        supplied = reconciler.reconcile(request)
        receipt = FirecrackerClinicalStartupCleanupReceipt.model_validate_json(canonical_json_bytes(supplied))
        FirecrackerClinicalRuntime._validate_startup_cleanup_receipt(
            request,
            receipt,
        )
        after = _scan_firecracker_clinical_bootstrap_journals(
            journal_root=journal_root,
            guest_rpc_policy=guest_rpc_policy,
            bootstrap_receipt_key=bootstrap_receipt_key,
            bootstrap_trust_anchor=bootstrap_trust_anchor,
            expected_authorization_key_id=config.bootstrap_authorization_key_id,
            expected_receipt_key_id=config.bootstrap_receipt_key_id,
            expected_execution_policy_sha256=execution_policy_sha256,
            expected_worker_spec_sha256=worker_spec_sha256,
        )
    except BaseException:
        raise FirecrackerClinicalRuntimeError('recovery orphan cleanup could not be completely reconciled') from None
    if after != current:
        raise FirecrackerClinicalRuntimeError('bootstrap journal inventory changed during recovery reconciliation')
    return FirecrackerClinicalRecoveryReconciliationReport(
        request=request,
        cleanup_receipt=receipt,
    )


class FirecrackerSupervisorBoundary(Protocol):
    @property
    def spec(self) -> FirecrackerWorkerSpec: ...

    def prepare(self, *, run_id: str) -> FirecrackerPreparedWorker: ...

    def launch(
        self,
        prepared: FirecrackerPreparedWorker,
        *,
        prebound_guest_listener: FirecrackerPreboundGuestListener | None = None,
    ) -> RunningFirecrackerWorker: ...

    def wait_for_exit(
        self,
        running: RunningFirecrackerWorker,
        *,
        timeout_seconds: float,
    ) -> bool: ...

    def terminate_and_cleanup(
        self,
        running: RunningFirecrackerWorker,
        *,
        grace_seconds: float = 5.0,
    ) -> FirecrackerCleanupReceipt: ...

    def discard_prepared(self, prepared: FirecrackerPreparedWorker) -> FirecrackerCleanupReceipt: ...


class ManagedClinicalRuntimeOwnershipLedger(Protocol):
    """Crash-durable host ownership transitions required by the managed composition."""

    def begin_preparing(
        self,
        request: ClinicalRuntimePrepareRequest,
        *,
        spec: FirecrackerWorkerSpec,
    ) -> object: ...

    def record_prepared(self, worker: FirecrackerPreparedWorker) -> object: ...

    def record_start_bound(
        self,
        *,
        run_id: str,
        start: ClinicalRuntimeStart,
        capability_id: str,
    ) -> object: ...

    def record_running(self, running: RunningFirecrackerWorker) -> object: ...

    def record_capability_revoked(
        self,
        *,
        run_id: str,
        capability_id: str,
    ) -> object: ...

    def record_cleaned(
        self,
        *,
        run_id: str,
        terminal_reason: Literal['runtime_cleanup', 'startup_reaper', 'preparation_failed'],
        cleanup_receipt: FirecrackerCleanupReceipt | None = None,
    ) -> object: ...


class ClinicalBootstrapSessionRunner(Protocol):
    """Socket behavior seam; the runtime still owns all security-relevant ordering and pins."""

    @property
    def authenticated_bootstrap(self) -> AuthenticatedClinicalGuestBootstrap | None: ...

    def open(self) -> None: ...

    def serve_one(
        self,
        *,
        hello: ClinicalGuestBootstrapHello,
        session: GuestRpcHostSession,
        deadline_monotonic: float,
        expected_peer_pid: int,
    ) -> AuthenticatedClinicalGuestBootstrap: ...

    def close(self) -> None: ...


type UnixPeerIdentityVerifier = Callable[[socket.socket, int, int, int], None]


def verify_linux_unix_peer_identity(
    connection: socket.socket,
    expected_pid: int,
    expected_uid: int,
    expected_gid: int,
) -> None:
    """Bind an accepted Linux UDS peer to the exact tracked Firecracker process and account.

    ``SO_PEERCRED`` is Linux-specific.  Non-Linux hosts retain the development socket seam so the
    framing tests can run, but they cannot use this helper as evidence of a qualified worker.  An
    exact local PID match is not remote guest/image attestation and still requires live KVM and
    process/cgroup-provenance qualification.
    """

    if isinstance(expected_pid, bool) or not isinstance(expected_pid, int) or expected_pid <= 0:
        raise FirecrackerClinicalRuntimeError('pinned worker process ID is invalid')
    if sys.platform != 'linux':
        return
    option = getattr(socket, 'SO_PEERCRED', None)
    if not isinstance(option, int):
        raise FirecrackerClinicalRuntimeError('Linux Unix peer credentials are unavailable')
    credential_struct = struct.Struct('3i')
    try:
        raw_credentials = connection.getsockopt(
            socket.SOL_SOCKET,
            option,
            credential_struct.size,
        )
        peer_pid, peer_uid, peer_gid = credential_struct.unpack(raw_credentials)
    except (OSError, struct.error):
        raise FirecrackerClinicalRuntimeError('Linux Unix peer credentials could not be verified') from None
    if (peer_pid, peer_uid, peer_gid) != (expected_pid, expected_uid, expected_gid):
        raise FirecrackerClinicalRuntimeError('Linux Unix peer differs from the pinned worker process identity')


class FirecrackerClinicalGuestBootstrapSession:
    """One signed bootstrap and one guest-RPC session on a run-specific Unix socket."""

    def __init__(
        self,
        *,
        prepared: FirecrackerPreparedWorker,
        rpc_port: int,
        worker_uid: int,
        worker_gid: int,
        authorization_signer: Ed25519Signer,
        expected_authorization_key_id: str,
        receipt_key: bytes,
        journal_authenticated_bootstrap: Callable[[AuthenticatedClinicalGuestBootstrap], str],
        clock: Callable[[], datetime],
        connection_timeout_seconds: float,
        monotonic_clock: Callable[[], float] = time.monotonic,
        peer_identity_verifier: UnixPeerIdentityVerifier = verify_linux_unix_peer_identity,
    ) -> None:
        socket_path = Path(
            firecracker_guest_initiated_uds_path(
                uds_path=prepared.vsock_uds_path,
                port=rpc_port,
            )
        )
        if socket_path.parent != Path(prepared.vsock_uds_path).parent:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap socket escaped the prepared run')
        if not 0 < connection_timeout_seconds <= 30:
            raise ValueError('clinical bootstrap timeout must be greater than zero and at most 30 seconds')
        if worker_uid < 1 or worker_gid < 1:
            raise ValueError('clinical bootstrap worker UID and GID must be non-root identities')
        self.prepared = prepared
        self.rpc_port = rpc_port
        self.socket_path = socket_path
        self.worker_uid = worker_uid
        self.worker_gid = worker_gid
        self._authorization_signer = authorization_signer
        self._authorization_key_id = expected_authorization_key_id
        self._receipt_key = bytes(receipt_key)
        self._journal_authenticated_bootstrap = journal_authenticated_bootstrap
        self._clock = clock
        self._timeout = float(connection_timeout_seconds)
        self._monotonic = monotonic_clock
        self._peer_identity_verifier = peer_identity_verifier
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._served = False
        self._authenticated_bootstrap: AuthenticatedClinicalGuestBootstrap | None = None

    @property
    def authenticated_bootstrap(self) -> AuthenticatedClinicalGuestBootstrap | None:
        """Return a completed handshake even if the subsequent RPC loop failed."""

        return self._authenticated_bootstrap

    def open(self) -> None:
        if self._listener is not None or self._served:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap listener is not fresh')
        self._verify_worker_socket_parent()
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise FirecrackerClinicalRuntimeError('clinical bootstrap refuses an existing socket')
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            bound = self.socket_path.lstat()
            if not stat.S_ISSOCK(bound.st_mode):
                raise FirecrackerClinicalRuntimeError('clinical bootstrap bind did not create a Unix socket')
            self._socket_identity = (bound.st_dev, bound.st_ino)
            os.chown(
                self.socket_path,
                self.worker_uid,
                self.worker_gid,
                follow_symlinks=False,
            )
            os.chmod(self.socket_path, 0o600, follow_symlinks=False)
            self._verify_owned_socket()
            listener.listen(1)
            listener.settimeout(self._timeout)
        except Exception:
            cleanup_error: BaseException | None = None
            try:
                self._remove_owned_socket()
            except BaseException as error:
                cleanup_error = error
            finally:
                listener.close()
            if cleanup_error is not None:
                raise cleanup_error
            raise FirecrackerClinicalRuntimeError('clinical bootstrap listener could not be opened') from None
        self._listener = listener

    def serve_one(
        self,
        *,
        hello: ClinicalGuestBootstrapHello,
        session: GuestRpcHostSession,
        deadline_monotonic: float,
        expected_peer_pid: int,
    ) -> AuthenticatedClinicalGuestBootstrap:
        listener = self._listener
        if listener is None or self._served:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap listener is not open and fresh')
        if not _bootstrap_matches_guest_session(
            hello=hello,
            session=session,
            prepared=self.prepared,
            rpc_port=self.rpc_port,
        ):
            raise FirecrackerClinicalRuntimeError('clinical bootstrap differs from the guest RPC session')
        remaining = deadline_monotonic - self._monotonic()
        if remaining <= 0:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap deadline elapsed')
        self._verify_owned_socket()
        listener.settimeout(min(remaining, self._timeout))
        try:
            connection, _ = listener.accept()
        except OSError:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap connection was not established') from None
        self._served = True
        try:
            self._peer_identity_verifier(
                connection,
                expected_peer_pid,
                self.worker_uid,
                self.worker_gid,
            )
            remaining = deadline_monotonic - self._monotonic()
            if remaining <= 0:
                raise FirecrackerClinicalRuntimeError('clinical bootstrap deadline elapsed')
            connection.settimeout(min(remaining, self._timeout))
            artifact = perform_host_clinical_guest_bootstrap(
                connection,
                hello=hello,
                authorization_signer=self._authorization_signer,
                expected_authorization_key_id=self._authorization_key_id,
                receipt_key=self._receipt_key,
                clock=self._clock,
                timeout_seconds=min(remaining, self._timeout),
            )
            self._authenticated_bootstrap = artifact
            expected_journal_sha256 = hashlib.sha256(canonical_json_bytes(artifact)).hexdigest()
            if self._journal_authenticated_bootstrap(artifact) != expected_journal_sha256:
                raise FirecrackerClinicalRuntimeError('bootstrap journal returned a different artifact hash')
            GuestRpcHostServer(session).serve(connection)
            return artifact
        finally:
            connection.close()

    def close(self) -> None:
        listener = self._listener
        self._listener = None
        cleanup_error: BaseException | None = None
        try:
            self._remove_owned_socket()
        except BaseException as error:
            cleanup_error = error
        finally:
            if listener is not None:
                listener.close()
        if cleanup_error is not None:
            raise cleanup_error

    def _verify_worker_socket_parent(self) -> None:
        try:
            metadata = self.socket_path.parent.lstat()
        except OSError:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap socket parent is unavailable') from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FirecrackerClinicalRuntimeError('clinical bootstrap socket parent is not a real directory')
        if (
            stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != self.worker_uid
            or metadata.st_gid != self.worker_gid
        ):
            raise FirecrackerClinicalRuntimeError(
                'clinical bootstrap socket parent differs from the pinned worker account'
            )

    def _verify_owned_socket(self) -> None:
        identity = self._socket_identity
        try:
            metadata = self.socket_path.lstat()
        except OSError:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap socket is unavailable') from None
        if not stat.S_ISSOCK(metadata.st_mode):
            raise FirecrackerClinicalRuntimeError('clinical bootstrap path changed type')
        if identity is None or (metadata.st_dev, metadata.st_ino) != identity:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap socket changed identity')
        if (
            stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != self.worker_uid
            or metadata.st_gid != self.worker_gid
        ):
            raise FirecrackerClinicalRuntimeError('clinical bootstrap socket differs from the pinned worker account')

    def _remove_owned_socket(self) -> None:
        try:
            metadata = self.socket_path.lstat()
        except FileNotFoundError:
            self._socket_identity = None
            return
        if not stat.S_ISSOCK(metadata.st_mode):
            raise FirecrackerClinicalRuntimeError('clinical bootstrap path changed type during cleanup')
        if (
            self._socket_identity is None
            or (
                metadata.st_dev,
                metadata.st_ino,
            )
            != self._socket_identity
        ):
            raise FirecrackerClinicalRuntimeError('clinical bootstrap socket changed identity during cleanup')
        self.socket_path.unlink()
        self._socket_identity = None


type ClinicalBootstrapRunnerFactory = Callable[..., ClinicalBootstrapSessionRunner]
type WorkerAttestationFinalizer = Callable[..., AuthenticatedFirecrackerWorkerAttestation]


@dataclass(slots=True)
class _PreparedState:
    request: ClinicalRuntimePrepareRequest
    worker: FirecrackerPreparedWorker
    public_receipt: ClinicalPreparedRuntime
    capability_secret: bytes
    bootstrap_journal_path: Path | None = None
    bootstrap_journal_sha256: str | None = None
    consumed: bool = False


class FirecrackerClinicalRuntime(ClinicalRuntimeBoundary):
    """Canonical development implementation of the launcher's typed runtime boundary."""

    def __init__(
        self,
        *,
        config: FirecrackerClinicalRuntimeConfig,
        supervisor: FirecrackerSupervisorBoundary | FirecrackerSupervisor,
        gateway: AuthenticatedProviderGateway,
        gateway_secret_store: GatewayCapabilitySecretStore,
        execution_policy: AgenticExecutionPolicy,
        gateway_route: GatewayModelRoute,
        provider_subprocess_spec_sha256: str,
        provider_subprocess_behavior_sha256: str,
        provider_subprocess_module_source_sha256: str,
        guest_rpc_policy: GuestRpcPolicy,
        harness: AgenticHarnessIdentity,
        keys: FirecrackerClinicalRuntimeKeys,
        bootstrap_authorization_signer: Ed25519Signer,
        bootstrap_trust_anchor: ClinicalGuestBootstrapTrustAnchor,
        evidence_root: Path,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        token_bytes: Callable[[int], bytes] = secrets.token_bytes,
        token_hex: Callable[[int], str] = secrets.token_hex,
        bootstrap_runner_factory: ClinicalBootstrapRunnerFactory = FirecrackerClinicalGuestBootstrapSession,
        guest_session_factory: Callable[..., GuestRpcHostSession] = GuestRpcHostSession,
        finalize_worker: WorkerAttestationFinalizer = finalize_firecracker_worker_attestation,
        require_global_startup_reconciliation: bool = False,
        managed_ownership: ManagedClinicalRuntimeOwnershipLedger | None = None,
    ) -> None:
        if gateway.secret_resolver is not gateway_secret_store:
            raise ValueError('runtime and provider gateway must share the exact capability secret store')
        if gateway.provider_calls_forcibly_cancellable is not True:
            raise ValueError('runtime requires child-isolated forcibly cancellable provider calls')
        if not isinstance(require_global_startup_reconciliation, bool):
            raise TypeError('global startup reconciliation requirement must be a boolean')
        _require_lower_hex(
            provider_subprocess_spec_sha256,
            64,
            'provider subprocess specification SHA-256',
        )
        _require_lower_hex(
            provider_subprocess_behavior_sha256,
            64,
            'provider subprocess behavior SHA-256',
        )
        _require_lower_hex(
            provider_subprocess_module_source_sha256,
            64,
            'provider subprocess module source SHA-256',
        )
        signer = checked_signer(bootstrap_authorization_signer)
        authorization_public_key = signer.public_key_bytes()
        if (
            clinical_guest_bootstrap_authorization_key_id(authorization_public_key)
            != config.bootstrap_authorization_key_id
        ):
            raise ValueError('bootstrap signer differs from the runtime authorization-key pin')
        if (
            clinical_guest_bootstrap_receipt_key_id(keys.clinical_guest_bootstrap_receipt_key)
            != config.bootstrap_receipt_key_id
        ):
            raise ValueError('clinical bootstrap receipt key differs from the runtime config')
        expected_anchor = (
            config.bootstrap_authorization_key_id,
            authorization_public_key.hex(),
            agentic_policy_sha256(execution_policy),
            firecracker_guest_bootstrap_profile_sha256(supervisor.spec),
            LANE_A_GUEST_HARNESS_POLICY_ID,
            CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
            LANE_A_GUEST_ACTION_SCHEMA_SHA256,
            ClinicalGuestRpcLimits(
                maximum_frame_body_bytes=guest_rpc_policy.maximum_frame_body_bytes,
                maximum_session_wire_bytes=guest_rpc_policy.maximum_session_wire_bytes,
                maximum_requests=guest_rpc_policy.maximum_requests,
                maximum_list_entries=guest_rpc_policy.maximum_list_entries,
                maximum_read_bytes=guest_rpc_policy.maximum_read_bytes,
                maximum_search_results=guest_rpc_policy.maximum_search_results,
                maximum_submission_bytes=guest_rpc_policy.maximum_submission_bytes,
            ),
        )
        observed_anchor = (
            bootstrap_trust_anchor.authorization_key_id,
            bootstrap_trust_anchor.ed25519_public_key_hex,
            bootstrap_trust_anchor.execution_policy_sha256,
            bootstrap_trust_anchor.worker_bootstrap_profile_sha256,
            bootstrap_trust_anchor.harness_policy_id,
            bootstrap_trust_anchor.harness_policy_sha256,
            bootstrap_trust_anchor.action_schema_sha256,
            bootstrap_trust_anchor.rpc_limits,
        )
        if observed_anchor != expected_anchor:
            raise ValueError('guest bootstrap trust anchor differs from the runtime static pins')
        if execution_policy.required_isolation != IsolationTier.DEVELOPMENT or (
            execution_policy.response_protocol != AgenticResponseProtocol.CLINICAL_EXECUTION
        ):
            raise ValueError('Firecracker clinical runtime only accepts development Lane A policy')
        worker_limits = supervisor.spec.limits
        policy_limits = execution_policy.limits
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
            raise ValueError('Firecracker resources do not exactly implement the execution policy')
        if gateway_route.logical_model_id != harness.requested_model_id or (
            gateway_route.adapter_id != harness.adapter_id
        ):
            raise ValueError('harness model identity differs from the provider route')
        if harness.harness_image_or_commitment != f'sha256:{supervisor.spec.images.harness.sha256}':
            raise ValueError('harness identity differs from the pinned worker image')
        self.config = config
        self.supervisor = supervisor
        self.gateway = gateway
        self.gateway_secret_store = gateway_secret_store
        self.execution_policy = execution_policy
        self.gateway_route = gateway_route
        self.provider_subprocess_spec_sha256 = provider_subprocess_spec_sha256
        self.provider_subprocess_behavior_sha256 = provider_subprocess_behavior_sha256
        self.provider_subprocess_module_source_sha256 = provider_subprocess_module_source_sha256
        self.guest_rpc_policy = guest_rpc_policy
        self.harness = harness
        self.keys = keys
        self._authorization_signer = signer
        self.bootstrap_trust_anchor = bootstrap_trust_anchor
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic_clock or time.monotonic
        self._token_bytes = token_bytes
        self._token_hex = token_hex
        self._bootstrap_runner_factory = bootstrap_runner_factory
        self._guest_session_factory = guest_session_factory
        self._finalize_worker = finalize_worker
        self._managed_ownership = managed_ownership
        self._states: dict[str, _PreparedState] = {}
        self._lock = threading.RLock()
        self.evidence_root = _prepare_private_root(evidence_root)
        self.bootstrap_journal_root = _prepare_private_root(self.evidence_root / 'bootstrap-journal')
        self._startup_reconciliation_in_progress = False
        self._startup_orphans = self._scan_startup_bootstrap_journals()
        self._startup_cleanup_receipt: FirecrackerClinicalStartupCleanupReceipt | None = None
        self._startup_request: FirecrackerClinicalStartupReconciliationRequest | None = None
        self._startup_admission_allowed = not self._startup_orphans and not require_global_startup_reconciliation

    @property
    def startup_reconciliation_required(self) -> bool:
        """Whether preparation is blocked pending a complete deployment startup scan."""

        with self._lock:
            return not self._startup_admission_allowed

    def reconcile_startup(
        self,
        *,
        reconciler: FirecrackerClinicalStartupReconciler,
    ) -> FirecrackerClinicalStartupReconciliationReport:
        """Authenticate journals and require deployment-owned orphan cleanup before admission.

        Journal authentication and cross-binding are repository-owned.  Live VM/process-group,
        cgroup/jail, and capability discovery are not derivable from the journal and are delegated
        through ``reconciler``.  Journals are deliberately retained, so every new runtime process
        repeats discovery instead of trusting an unauthenticated prior in-memory report.
        """

        try:
            current = self._scan_startup_bootstrap_journals()
        except BaseException:
            with self._lock:
                self._startup_admission_allowed = False
            raise
        with self._lock:
            if self._states:
                raise FirecrackerClinicalRuntimeError('startup reconciliation cannot run after worker preparation')
            if self._startup_reconciliation_in_progress:
                raise FirecrackerClinicalRuntimeError('startup reconciliation is already in progress')
            if current != self._startup_orphans:
                self._startup_orphans = current
                self._startup_request = None
                self._startup_cleanup_receipt = None
                self._startup_admission_allowed = False
            if (
                self._startup_admission_allowed
                and self._startup_request is not None
                and self._startup_cleanup_receipt is not None
            ):
                return self._startup_reconciliation_report()
            self._startup_reconciliation_in_progress = True

        request = self._startup_reconciliation_request(current)
        receipt: FirecrackerClinicalStartupCleanupReceipt | None = None
        failed = False
        try:
            supplied = reconciler.reconcile(request)
            receipt = FirecrackerClinicalStartupCleanupReceipt.model_validate_json(canonical_json_bytes(supplied))
            self._validate_startup_cleanup_receipt(request, receipt)
            if self._scan_startup_bootstrap_journals() != current:
                raise FirecrackerClinicalRuntimeError(
                    'bootstrap journal inventory changed during startup reconciliation'
                )
        except BaseException:
            failed = True
        finally:
            with self._lock:
                self._startup_reconciliation_in_progress = False
        if failed:
            with self._lock:
                self._startup_request = None
                self._startup_cleanup_receipt = None
                self._startup_admission_allowed = False
            raise FirecrackerClinicalRuntimeError(
                'deployment orphan cleanup could not be completely reconciled'
            ) from None

        if receipt is None:
            raise FirecrackerClinicalRuntimeError('deployment orphan cleanup returned no reconciliation receipt')
        with self._lock:
            self._startup_request = request
            self._startup_cleanup_receipt = receipt
            self._startup_admission_allowed = True
            return self._startup_reconciliation_report()

    def prepare(self, request: ClinicalRuntimePrepareRequest) -> ClinicalPreparedRuntime:
        """Allocate an unlaunched worker plus private session identities.

        Preparation deliberately does not register the capability secret, issue a grant, open the
        bootstrap listener, construct a guest session, or launch Firecracker.
        """

        self._require_startup_admission()
        self._validate_prepare_request(request)
        journal_path = self._bootstrap_journal_path(request.launch.run_id)
        if journal_path.exists() or journal_path.is_symlink():
            raise FirecrackerClinicalRuntimeError('runtime refuses a run ID with retained bootstrap evidence')
        capability_secret = self._token_bytes(GATEWAY_CAPABILITY_SECRET_BYTES)
        if not isinstance(capability_secret, bytes) or len(capability_secret) != GATEWAY_CAPABILITY_SECRET_BYTES:
            raise FirecrackerClinicalRuntimeError('capability secret generator returned an invalid value')
        session_id = self._token_hex(16)
        _require_lower_hex(session_id, 32, 'guest RPC session ID')
        capability_id = gateway_capability_id(capability_secret)
        worker: FirecrackerPreparedWorker | None = None
        cleanup: FirecrackerCleanupReceipt | None = None
        ownership_begun = False
        try:
            if self._managed_ownership is not None:
                self._managed_ownership.begin_preparing(
                    request,
                    spec=self.supervisor.spec,
                )
                ownership_begun = True
            worker = self.supervisor.prepare(run_id=request.launch.run_id)
            expected_worker_sha256 = firecracker_model_sha256(self.supervisor.spec)
            if (
                worker.run_id,
                worker.worker_spec_sha256,
                worker.harness_sha256,
            ) != (
                request.launch.run_id,
                expected_worker_sha256,
                self.supervisor.spec.images.harness.sha256,
            ):
                raise FirecrackerClinicalRuntimeError('prepared worker differs from its fixed runtime pins')
            if self._managed_ownership is not None:
                self._managed_ownership.record_prepared(worker)
            prepared_at = max(self._now(), request.launch.claimed_at, worker.created_at)
            public = ClinicalPreparedRuntime(
                runtime_id=self.config.runtime_id,
                runtime_version=self.config.runtime_version,
                runtime_executable_sha256=self.config.runtime_executable_sha256,
                runtime_config_sha256=firecracker_clinical_runtime_config_sha256(self.config),
                launcher_deployment_sha256=canonical_clinical_launcher_deployment_sha256(request.deployment),
                reservation_sha256=request.launch.reservation_sha256,
                launch_sha256=clinical_production_task_launch_sha256(request.launch),
                system_identity_sha256=request.reservation.system_identity_sha256,
                episode_id=request.binding.episode_id,
                run_id=request.launch.run_id,
                workspace_manifest_sha256=request.binding.workspace_manifest_sha256,
                workspace_tree_sha256=request.binding.workspace_tree_sha256,
                model_visible_surface_sha256=request.binding.model_visible_surface_sha256,
                worker_spec_sha256=expected_worker_sha256,
                harness_sha256=self.supervisor.spec.images.harness.sha256,
                prepared_worker_sha256=firecracker_model_sha256(worker),
                guest_rpc_session_id=session_id,
                gateway_capability_id=capability_id,
                prepared_at=prepared_at,
            )
            state = _PreparedState(
                request=request,
                worker=worker,
                public_receipt=public,
                capability_secret=bytes(capability_secret),
                bootstrap_journal_path=journal_path,
            )
            with self._lock:
                if request.launch.run_id in self._states:
                    raise FirecrackerClinicalRuntimeError('runtime refuses to reuse a prepared run ID')
                self._states[request.launch.run_id] = state
            return public
        except BaseException:
            if worker is not None:
                try:
                    cleanup = self.supervisor.discard_prepared(worker)
                    _validate_cleanup(
                        cleanup,
                        run_id=worker.run_id,
                        lifecycle='never_launched',
                    )
                except BaseException:
                    pass
            if ownership_begun and self._managed_ownership is not None:
                try:
                    self._managed_ownership.record_cleaned(
                        run_id=request.launch.run_id,
                        terminal_reason='preparation_failed',
                        cleanup_receipt=cleanup,
                    )
                except BaseException:
                    pass
            capability_secret = b''
            raise

    def discard_prepared(self, prepared: ClinicalPreparedRuntime) -> None:
        """Destroy one exact unlaunched state; it can never subsequently be run."""

        with self._lock:
            state = self._states.pop(prepared.run_id, None)
            if state is not None:
                state.consumed = True
        if state is None:
            raise FirecrackerClinicalRuntimeError('prepared runtime is unavailable or already consumed')
        receipt_matches = hmac.compare_digest(
            clinical_prepared_runtime_sha256(state.public_receipt),
            clinical_prepared_runtime_sha256(prepared),
        )
        cleanup_error: BaseException | None = None
        try:
            cleanup = self.supervisor.discard_prepared(state.worker)
            _validate_cleanup(cleanup, run_id=state.worker.run_id, lifecycle='never_launched')
        except BaseException as error:
            cleanup_error = error
        try:
            self.gateway_secret_store.revoke(state.public_receipt.gateway_capability_id)
        except BaseException as error:
            cleanup_error = cleanup_error or error
        if cleanup_error is None and self._managed_ownership is not None:
            try:
                self._managed_ownership.record_cleaned(
                    run_id=state.worker.run_id,
                    terminal_reason='runtime_cleanup',
                    cleanup_receipt=cleanup,
                )
            except BaseException as error:
                cleanup_error = error
        state.capability_secret = b''
        if cleanup_error is not None:
            raise FirecrackerClinicalRuntimeError('prepared runtime cleanup could not be proved') from None
        if not receipt_matches:
            raise FirecrackerClinicalRuntimeError('discarded prepared receipt differed from private runtime state')

    def run(self, prepared: ClinicalPreparedRuntime, start: ClinicalRuntimeStart) -> ClinicalRuntimeOutcome:
        """Redeem the prepared state exactly once and execute the concrete host composition."""

        with self._lock:
            state = self._states.pop(prepared.run_id, None)
            if state is not None:
                state.consumed = True
        if state is None:
            return ClinicalRuntimeFailed(ClinicalRuntimeFailureCode.LAUNCH_FAILED)
        try:
            self._validate_start(state, prepared, start)
            if self._managed_ownership is not None:
                self._managed_ownership.record_start_bound(
                    run_id=prepared.run_id,
                    start=start,
                    capability_id=prepared.gateway_capability_id,
                )
        except BaseException:
            cleaned = self._discard_consumed_state(state)
            return ClinicalRuntimeFailed(
                ClinicalRuntimeFailureCode.LAUNCH_FAILED if cleaned else ClinicalRuntimeFailureCode.CLEANUP_FAILED
            )
        return self._run_consumed_state(state, start)

    def _run_consumed_state(
        self,
        state: _PreparedState,
        start: ClinicalRuntimeStart,
    ) -> ClinicalRuntimeOutcome:
        prepared = state.public_receipt
        attempt_sha256 = start.start_redemption_sha256
        runner: ClinicalBootstrapSessionRunner | None = None
        running: RunningFirecrackerWorker | None = None
        guest_session: GuestRpcHostSession | None = None
        guest_artifact: AuthenticatedGuestRpcSession | None = None
        gateway_artifact: AuthenticatedGatewaySession | None = None
        bootstrap_artifact: AuthenticatedClinicalGuestBootstrap | None = None
        expected_hello: ClinicalGuestBootstrapHello | None = None
        worker_artifact: AuthenticatedFirecrackerWorkerAttestation | None = None
        cleanup: FirecrackerCleanupReceipt | None = None
        gateway_registered = False
        listener_opened = False
        failure: ClinicalRuntimeFailureCode | None = None
        phase = 'gateway'
        capability_secret = state.capability_secret
        capability_id = prepared.gateway_capability_id

        try:
            registered_id = self.gateway_secret_store.register(capability_secret)
            if registered_id != capability_id:
                raise FirecrackerClinicalRuntimeError('secret store returned a different capability ID')
            issued_at = max(self._now(), start.start_redemption.redeemed_at)
            grant = issue_gateway_capability(
                secret=capability_secret,
                run_id=prepared.run_id,
                attempt_reservation_sha256=attempt_sha256,
                execution_policy_sha256=agentic_policy_sha256(self.execution_policy),
                workspace_manifest_sha256=prepared.workspace_manifest_sha256,
                policy=self.gateway.policy,
                route=self.gateway_route,
                issued_at=issued_at,
                expires_at=issued_at + timedelta(seconds=self.supervisor.spec.limits.wall_seconds),
                expected_peer_cid=self.supervisor.spec.guest_cid,
                limits=self.execution_policy.limits,
            )
            if grant.capability_id != start.start_redemption.gateway_capability_id:
                raise FirecrackerClinicalRuntimeError('gateway grant differs from the redeemed capability')
            self.gateway.register_session(grant=grant, route=self.gateway_route, secret=capability_secret)
            gateway_registered = True

            phase = 'bootstrap_open'
            hello = self._bootstrap_hello(state=state, start=start)
            expected_hello = hello
            runner = self._bootstrap_runner_factory(
                prepared=state.worker,
                rpc_port=self.supervisor.spec.guest_rpc_port,
                worker_uid=self.supervisor.spec.worker_uid,
                worker_gid=self.supervisor.spec.worker_gid,
                authorization_signer=self._authorization_signer,
                expected_authorization_key_id=self.config.bootstrap_authorization_key_id,
                receipt_key=self.keys.clinical_guest_bootstrap_receipt_key,
                journal_authenticated_bootstrap=lambda artifact: self._journal_authenticated_bootstrap(
                    state=state,
                    start=start,
                    expected_hello=hello,
                    artifact=artifact,
                ),
                clock=self._clock,
                connection_timeout_seconds=self.config.bootstrap_connection_timeout_seconds,
                monotonic_clock=self._monotonic,
            )
            runner.open()
            listener_opened = True
            prebound_guest_listener = None
            if isinstance(runner, FirecrackerClinicalGuestBootstrapSession):
                prebound_guest_listener = capture_firecracker_prebound_guest_listener(
                    state.worker,
                    spec=self.supervisor.spec,
                )

            phase = 'launch'
            if prebound_guest_listener is None:
                running = self.supervisor.launch(state.worker)
            else:
                running = self.supervisor.launch(
                    state.worker,
                    prebound_guest_listener=prebound_guest_listener,
                )
            if running.prepared != state.worker:
                raise FirecrackerClinicalRuntimeError('running worker differs from the redeemed preparation')
            expected_peer_pid = running.firecracker_pid
            if isinstance(expected_peer_pid, bool) or not isinstance(expected_peer_pid, int) or expected_peer_pid <= 0:
                raise FirecrackerClinicalRuntimeError('launched worker process ID is invalid')
            if self._managed_ownership is not None:
                self._managed_ownership.record_running(running)

            phase = 'bootstrap'
            guest_session = self._guest_session_factory(
                session_id=prepared.guest_rpc_session_id,
                run_id=prepared.run_id,
                workspace_manifest_sha256=prepared.workspace_manifest_sha256,
                workspace_tree_sha256=prepared.workspace_tree_sha256,
                model_visible_surface_sha256=prepared.model_visible_surface_sha256,
                task_invocation=state.request.workspace.invocation,
                expected_response_protocol=AgenticResponseProtocol.CLINICAL_EXECUTION,
                worker_spec_sha256=prepared.worker_spec_sha256,
                execution_policy_sha256=agentic_policy_sha256(self.execution_policy),
                broker=state.request.workspace.brokered_surface(),
                gateway=self.gateway,
                gateway_grant=grant,
                gateway_secret=capability_secret,
                observed_peer_cid=self.supervisor.spec.guest_cid,
                rpc_port=self.supervisor.spec.guest_rpc_port,
                policy=self.guest_rpc_policy,
                receipt_key=self.keys.guest_rpc_receipt_key,
                expected_receipt_key_id=state.request.reservation.system.guest_rpc_receipt_key_id,
                clock=self._clock,
            )
            candidate_bootstrap = runner.serve_one(
                hello=hello,
                session=guest_session,
                deadline_monotonic=running.wall_deadline_monotonic,
                expected_peer_pid=expected_peer_pid,
            )
            verify_authenticated_clinical_guest_bootstrap(
                candidate_bootstrap,
                key=self.keys.clinical_guest_bootstrap_receipt_key,
                expected_key_id=self.config.bootstrap_receipt_key_id,
                expected_hello=hello,
                trust_anchor=self.bootstrap_trust_anchor,
            )
            self._require_authenticated_bootstrap_journal(state, candidate_bootstrap)
            bootstrap_artifact = candidate_bootstrap
            if bootstrap_artifact.signed_hello.authorization_key_id != (self.config.bootstrap_authorization_key_id):
                raise FirecrackerClinicalRuntimeError('bootstrap receipt differs from the launcher signing key')
            if not guest_session.terminal:
                guest_session.abort(GuestRpcErrorCode.CONNECTION_CLOSED)
                failure = ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE
            elif guest_session.final_submission_bytes == b'':
                failure = ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE

            phase = 'worker_exit'
            if failure is None:
                remaining = running.wall_deadline_monotonic - self._monotonic()
                if remaining <= 0:
                    failure = ClinicalRuntimeFailureCode.WORKER_LOST
                else:
                    if not self.supervisor.wait_for_exit(running, timeout_seconds=remaining):
                        failure = ClinicalRuntimeFailureCode.WORKER_LOST
        except Exception:
            if runner is not None and bootstrap_artifact is None and expected_hello is not None:
                candidate = runner.authenticated_bootstrap
                if candidate is not None:
                    try:
                        verify_authenticated_clinical_guest_bootstrap(
                            candidate,
                            key=self.keys.clinical_guest_bootstrap_receipt_key,
                            expected_key_id=self.config.bootstrap_receipt_key_id,
                            expected_hello=expected_hello,
                            trust_anchor=self.bootstrap_trust_anchor,
                        )
                        self._require_authenticated_bootstrap_journal(state, candidate)
                        bootstrap_artifact = candidate
                    except Exception:
                        pass
            if phase in {'gateway', 'launch'}:
                failure = ClinicalRuntimeFailureCode.LAUNCH_FAILED
            elif phase == 'bootstrap' and bootstrap_artifact is not None:
                failure = ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE
            elif phase in {'bootstrap_open', 'bootstrap'}:
                failure = ClinicalRuntimeFailureCode.BOOTSTRAP_FAILED
            else:
                failure = ClinicalRuntimeFailureCode.WORKER_LOST

        bootstrap_sha256 = (
            None if bootstrap_artifact is None else hashlib.sha256(canonical_json_bytes(bootstrap_artifact)).hexdigest()
        )
        boundary_cleanup_failed = False
        if runner is not None:
            try:
                runner.close()
                listener_opened = False
            except BaseException:
                boundary_cleanup_failed = True
        elif listener_opened:
            boundary_cleanup_failed = True

        if guest_session is not None:
            try:
                if not guest_session.terminal:
                    guest_session.abort(GuestRpcErrorCode.INTERNAL)
                guest_artifact = guest_session.seal(sealed_at=self._now())
                if guest_artifact.seal.terminal_status != GuestRpcTerminalStatus.COMPLETED:
                    failure = failure or ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE
            except BaseException:
                failure = failure or ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE
        if gateway_registered:
            try:
                terminal_reason = (
                    GatewayTerminalReason.COMPLETED
                    if failure is None
                    and guest_artifact is not None
                    and guest_artifact.seal.terminal_status == GuestRpcTerminalStatus.COMPLETED
                    else GatewayTerminalReason.FAILED
                )
                gateway_artifact = self.gateway.seal_session(
                    capability_id,
                    terminal_reason=terminal_reason,
                    sealed_at=self._now(),
                    revoke_secret=False,
                )
                if gateway_artifact.seal.terminal_reason != terminal_reason:
                    failure = failure or ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE
            except BaseException:
                failure = failure or ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE

        try:
            if running is None:
                cleanup = self.supervisor.discard_prepared(state.worker)
                _validate_cleanup(cleanup, run_id=prepared.run_id, lifecycle='never_launched')
            else:
                cleanup = self.supervisor.terminate_and_cleanup(
                    running,
                    grace_seconds=self.config.cleanup_grace_seconds,
                )
                _validate_cleanup(cleanup, run_id=prepared.run_id, lifecycle='terminated')
        except BaseException:
            boundary_cleanup_failed = True

        if running is not None and cleanup is not None and cleanup.lifecycle == 'terminated':
            try:
                worker_artifact = self._finalize_worker(
                    spec=self.supervisor.spec,
                    running=running,
                    cleanup=cleanup,
                    attempt_reservation_sha256=attempt_sha256,
                    attestation_key=self.keys.worker_attestation_key,
                    expected_attestation_key_id=state.request.reservation.system.worker_attestation_key_id,
                )
            except BaseException:
                if failure is None:
                    failure = ClinicalRuntimeFailureCode.EVIDENCE_FINALIZATION_FAILED

        capability_revoked = False
        try:
            if gateway_registered:
                self.gateway.revoke_capability(
                    capability_id,
                    reason=GatewayCapabilityRevocationReason.RUNTIME_CLEANUP,
                    revoked_at=self._now(),
                )
            else:
                self.gateway.revoke_unregistered_capability(
                    capability_id,
                    run_id=prepared.run_id,
                    attempt_reservation_sha256=attempt_sha256,
                    model_route_sha256=gateway_model_route_sha256(self.gateway_route),
                    reason=GatewayCapabilityRevocationReason.RUNTIME_CLEANUP,
                    revoked_at=self._now(),
                )
            capability_revoked = True
        except BaseException:
            boundary_cleanup_failed = True
        if capability_revoked and self._managed_ownership is not None:
            try:
                self._managed_ownership.record_capability_revoked(
                    run_id=prepared.run_id,
                    capability_id=capability_id,
                )
            except BaseException:
                boundary_cleanup_failed = True
        if (
            cleanup is not None
            and capability_revoked
            and self._managed_ownership is not None
            and not boundary_cleanup_failed
        ):
            try:
                self._managed_ownership.record_cleaned(
                    run_id=prepared.run_id,
                    terminal_reason='runtime_cleanup',
                    cleanup_receipt=cleanup,
                )
            except BaseException:
                boundary_cleanup_failed = True
        state.capability_secret = b''
        capability_secret = b''
        if boundary_cleanup_failed:
            return ClinicalRuntimeFailed(
                ClinicalRuntimeFailureCode.CLEANUP_FAILED,
                authenticated_bootstrap_sha256=bootstrap_sha256,
            )
        if failure is not None:
            return ClinicalRuntimeFailed(failure, authenticated_bootstrap_sha256=bootstrap_sha256)
        if (
            bootstrap_artifact is None
            or guest_artifact is None
            or gateway_artifact is None
            or worker_artifact is None
            or not isinstance(guest_artifact.submission, ExecutionSubmission)
        ):
            return ClinicalRuntimeFailed(
                ClinicalRuntimeFailureCode.EVIDENCE_FINALIZATION_FAILED,
                authenticated_bootstrap_sha256=bootstrap_sha256,
            )
        outcome = self._finalize_success(
            state=state,
            attempt_sha256=attempt_sha256,
            bootstrap_artifact=bootstrap_artifact,
            worker_artifact=worker_artifact,
            gateway_artifact=gateway_artifact,
            guest_artifact=guest_artifact,
            submission=guest_artifact.submission,
        )
        # Retain the fsynced ACK journal even after v0.2 reload.  The v0.2 materializer does not
        # yet expose a power-loss durability proof, so deleting the independent journal here could
        # lose both copies after a host crash.  A qualified deployment may garbage-collect it only
        # after durable package publication and registry terminalization are separately proved.
        return outcome

    def _validate_prepare_request(self, request: ClinicalRuntimePrepareRequest) -> None:
        deployment = request.deployment
        expected_deployment = (
            self.config.runtime_id,
            self.config.runtime_version,
            self.config.runtime_executable_sha256,
            firecracker_clinical_runtime_config_sha256(self.config),
            request.reservation.registry_authority_id,
            request.reservation.system_identity_sha256,
        )
        observed_deployment = (
            deployment.runtime_id,
            deployment.runtime_version,
            deployment.runtime_executable_sha256,
            deployment.runtime_config_sha256,
            deployment.registry_authority_id,
            deployment.expected_system_identity_sha256,
        )
        if observed_deployment != expected_deployment:
            raise FirecrackerClinicalRuntimeError('runtime deployment differs from the reservation or config')
        system = request.reservation.system
        expected_system = (
            agentic_policy_sha256(self.execution_policy),
            firecracker_model_sha256(self.supervisor.spec),
            authenticated_gateway_policy_sha256(self.gateway.policy),
            gateway_model_route_sha256(self.gateway_route),
            self.provider_subprocess_spec_sha256,
            self.provider_subprocess_behavior_sha256,
            self.provider_subprocess_module_source_sha256,
            guest_rpc_policy_sha256(self.guest_rpc_policy),
            self.config.bootstrap_authorization_key_id,
            self.config.bootstrap_receipt_key_id,
            firecracker_attestation_key_id(self.keys.worker_attestation_key),
            gateway_session_key_id(self.keys.gateway_receipt_key),
            guest_rpc_session_key_id(self.keys.guest_rpc_receipt_key),
            clinical_production_run_key_id(self.keys.production_receipt_key),
            self.harness,
            self.gateway_route,
            AgenticResponseProtocol.CLINICAL_EXECUTION,
        )
        observed_system = (
            system.execution_policy_sha256,
            system.worker_spec_sha256,
            system.gateway_policy_sha256,
            system.gateway_route_sha256,
            system.provider_subprocess_spec_sha256,
            system.provider_subprocess_behavior_sha256,
            system.provider_subprocess_module_source_sha256,
            system.guest_rpc_policy_sha256,
            system.guest_bootstrap_authorization_key_id,
            system.guest_bootstrap_receipt_key_id,
            system.worker_attestation_key_id,
            system.gateway_receipt_key_id,
            system.guest_rpc_receipt_key_id,
            system.production_receipt_key_id,
            system.harness,
            system.gateway_route,
            system.response_protocol,
        )
        if observed_system != expected_system:
            raise FirecrackerClinicalRuntimeError('runtime composition differs from the reserved system')
        if clinical_workspace_receipt_key_id(self.keys.workspace_receipt_key) != (
            request.workspace.authenticated_receipt.receipt.receipt_key_id
        ):
            raise FirecrackerClinicalRuntimeError('workspace receipt key differs from the loaded workspace')
        binding = request.binding
        workspace = request.workspace
        observed_workspace = (
            request.launch.episode_id,
            request.launch.workspace_manifest_sha256,
            workspace.task.context.episode_id,
            workspace.task.context.target_trial_id,
            hashlib.sha256(canonical_json_bytes(workspace.task)).hexdigest(),
            workspace.task.context_sha256,
            workspace.manifest_sha256,
            workspace.manifest.workspace_tree_sha256,
            workspace.manifest.model_visible_surface_sha256,
            workspace.authenticated_receipt_sha256,
        )
        expected_workspace = (
            binding.episode_id,
            binding.workspace_manifest_sha256,
            binding.episode_id,
            binding.target_trial_id,
            binding.task_sha256,
            binding.task_context_sha256,
            binding.workspace_manifest_sha256,
            binding.workspace_tree_sha256,
            binding.model_visible_surface_sha256,
            binding.authenticated_workspace_receipt_sha256,
        )
        if observed_workspace != expected_workspace:
            raise FirecrackerClinicalRuntimeError('runtime workspace differs from its launch task binding')

    def _validate_start(
        self,
        state: _PreparedState,
        prepared: ClinicalPreparedRuntime,
        start: ClinicalRuntimeStart,
    ) -> None:
        expected_prepared_sha256 = clinical_prepared_runtime_sha256(state.public_receipt)
        if not hmac.compare_digest(expected_prepared_sha256, clinical_prepared_runtime_sha256(prepared)):
            raise FirecrackerClinicalRuntimeError('runtime start uses a different prepared receipt')
        redemption = start.start_redemption
        expected = (
            canonical_clinical_launcher_deployment_sha256(state.request.deployment),
            expected_prepared_sha256,
            clinical_production_start_redemption_sha256(redemption),
            state.request.launch.reservation_sha256,
            clinical_production_task_launch_sha256(state.request.launch),
            state.request.reservation.system_identity_sha256,
            state.request.binding.episode_id,
            state.request.launch.run_id,
            state.request.deployment.canonical_launcher_id,
            state.request.deployment.canonical_launcher_executable_sha256,
            prepared.prepared_worker_sha256,
            prepared.guest_rpc_session_id,
            prepared.gateway_capability_id,
        )
        observed = (
            start.launcher_deployment_sha256,
            start.prepared_runtime_sha256,
            start.start_redemption_sha256,
            redemption.reservation_sha256,
            redemption.launch_sha256,
            redemption.system_identity_sha256,
            redemption.episode_id,
            redemption.run_id,
            redemption.canonical_launcher_id,
            redemption.canonical_launcher_executable_sha256,
            redemption.prepared_worker_sha256,
            redemption.guest_rpc_session_id,
            redemption.gateway_capability_id,
        )
        if observed != expected or redemption.redeemed_at < prepared.prepared_at:
            raise FirecrackerClinicalRuntimeError('runtime start differs from its one-time redemption')

    def _bootstrap_hello(
        self,
        *,
        state: _PreparedState,
        start: ClinicalRuntimeStart,
    ) -> ClinicalGuestBootstrapHello:
        nonce = self._token_hex(32)
        _require_lower_hex(nonce, 64, 'clinical bootstrap nonce')
        now = max(self._now(), start.start_redemption.redeemed_at)
        validity = min(
            self.config.bootstrap_validity_seconds,
            self.supervisor.spec.limits.wall_seconds,
        )
        limits = ClinicalGuestRpcLimits(
            maximum_frame_body_bytes=self.guest_rpc_policy.maximum_frame_body_bytes,
            maximum_session_wire_bytes=self.guest_rpc_policy.maximum_session_wire_bytes,
            maximum_requests=self.guest_rpc_policy.maximum_requests,
            maximum_list_entries=self.guest_rpc_policy.maximum_list_entries,
            maximum_read_bytes=self.guest_rpc_policy.maximum_read_bytes,
            maximum_search_results=self.guest_rpc_policy.maximum_search_results,
            maximum_submission_bytes=self.guest_rpc_policy.maximum_submission_bytes,
        )
        workspace = state.request.workspace
        return ClinicalGuestBootstrapHello(
            run_id=state.public_receipt.run_id,
            start_redemption_sha256=start.start_redemption_sha256,
            session_id=state.public_receipt.guest_rpc_session_id,
            task_invocation=workspace.invocation,
            task_invocation_sha256=agentic_task_invocation_sha256(workspace.invocation),
            workspace_manifest_sha256=workspace.manifest_sha256,
            workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
            model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
            execution_policy_sha256=agentic_policy_sha256(self.execution_policy),
            worker_bootstrap_profile_sha256=(firecracker_guest_bootstrap_profile_sha256(self.supervisor.spec)),
            worker_spec_sha256=firecracker_model_sha256(self.supervisor.spec),
            harness_policy_id=LANE_A_GUEST_HARNESS_POLICY_ID,
            harness_policy_sha256=CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
            action_schema_sha256=LANE_A_GUEST_ACTION_SCHEMA_SHA256,
            rpc_limits=limits,
            nonce=nonce,
            valid_from=now,
            expires_at=now + timedelta(seconds=validity),
        )

    def _scan_startup_bootstrap_journals(
        self,
    ) -> tuple[FirecrackerClinicalStartupOrphan, ...]:
        """Read the exact private inventory and authenticate every retained journal."""

        return _scan_firecracker_clinical_bootstrap_journals(
            journal_root=self.bootstrap_journal_root,
            guest_rpc_policy=self.guest_rpc_policy,
            bootstrap_receipt_key=self.keys.clinical_guest_bootstrap_receipt_key,
            bootstrap_trust_anchor=self.bootstrap_trust_anchor,
            expected_authorization_key_id=self.config.bootstrap_authorization_key_id,
            expected_receipt_key_id=self.config.bootstrap_receipt_key_id,
            expected_execution_policy_sha256=agentic_policy_sha256(self.execution_policy),
            expected_worker_spec_sha256=firecracker_model_sha256(self.supervisor.spec),
        )

    def _startup_reconciliation_request(
        self,
        orphans: tuple[FirecrackerClinicalStartupOrphan, ...],
    ) -> FirecrackerClinicalStartupReconciliationRequest:
        requested_at = self._now()
        if orphans:
            requested_at = max(
                requested_at,
                *(orphan.bootstrap_ack_received_at for orphan in orphans),
            )
        return FirecrackerClinicalStartupReconciliationRequest(
            runtime_config_sha256=firecracker_clinical_runtime_config_sha256(self.config),
            execution_policy_sha256=agentic_policy_sha256(self.execution_policy),
            worker_spec_sha256=firecracker_model_sha256(self.supervisor.spec),
            gateway_policy_sha256=authenticated_gateway_policy_sha256(self.gateway.policy),
            gateway_route_sha256=gateway_model_route_sha256(self.gateway_route),
            bootstrap_authorization_key_id=self.config.bootstrap_authorization_key_id,
            bootstrap_receipt_key_id=self.config.bootstrap_receipt_key_id,
            retained_journals=orphans,
            requested_at=requested_at,
        )

    @staticmethod
    def _validate_startup_cleanup_receipt(
        request: FirecrackerClinicalStartupReconciliationRequest,
        receipt: FirecrackerClinicalStartupCleanupReceipt,
    ) -> None:
        if (
            receipt.reconciliation_request_sha256,
            receipt.retained_journal_count,
        ) != (
            firecracker_clinical_startup_reconciliation_request_sha256(request),
            len(request.retained_journals),
        ) or receipt.reconciled_at < request.requested_at:
            raise FirecrackerClinicalRuntimeError(
                'deployment cleanup receipt differs from its complete startup inventory request'
            )

    def _require_startup_admission(self) -> None:
        try:
            current = self._scan_startup_bootstrap_journals()
        except BaseException:
            with self._lock:
                self._startup_admission_allowed = False
            raise
        with self._lock:
            if current != self._startup_orphans:
                self._startup_orphans = current
                self._startup_request = None
                self._startup_cleanup_receipt = None
                self._startup_admission_allowed = False
            if not self._startup_admission_allowed or self._startup_reconciliation_in_progress:
                raise FirecrackerClinicalRuntimeError('runtime startup orphan reconciliation is incomplete')

    def _startup_reconciliation_report(self) -> FirecrackerClinicalStartupReconciliationReport:
        request = self._startup_request
        receipt = self._startup_cleanup_receipt
        if request is None or receipt is None:
            raise FirecrackerClinicalRuntimeError('startup reconciliation evidence is unavailable')
        return FirecrackerClinicalStartupReconciliationReport(
            request=request,
            cleanup_receipt=receipt,
        )

    def _bootstrap_journal_path(self, run_id: str) -> Path:
        _require_lower_hex(run_id, 32, 'clinical bootstrap journal run ID')
        path = self.bootstrap_journal_root / f'{run_id}.json'
        if path.parent != self.bootstrap_journal_root or path.name != f'{run_id}.json':
            raise FirecrackerClinicalRuntimeError('clinical bootstrap journal path escaped its private root')
        return path

    def _journal_authenticated_bootstrap(
        self,
        *,
        state: _PreparedState,
        start: ClinicalRuntimeStart,
        expected_hello: ClinicalGuestBootstrapHello,
        artifact: AuthenticatedClinicalGuestBootstrap,
    ) -> str:
        """Atomically retain the authenticated handshake before any guest RPC is served."""

        canonical = AuthenticatedClinicalGuestBootstrap.model_validate_json(canonical_json_bytes(artifact))
        verify_authenticated_clinical_guest_bootstrap(
            canonical,
            key=self.keys.clinical_guest_bootstrap_receipt_key,
            expected_key_id=self.config.bootstrap_receipt_key_id,
            expected_hello=expected_hello,
            trust_anchor=self.bootstrap_trust_anchor,
        )
        if not self._bootstrap_matches_runtime_state(state=state, start=start, artifact=canonical):
            raise FirecrackerClinicalRuntimeError('clinical bootstrap journal differs from the redeemed runtime')
        payload = canonical_json_bytes(canonical)
        if len(payload) > _MAX_BOOTSTRAP_JOURNAL_BYTES:
            raise FirecrackerClinicalRuntimeError('clinical bootstrap journal exceeds its fixed byte limit')
        digest = hashlib.sha256(payload).hexdigest()
        expected_path = self._bootstrap_journal_path(state.public_receipt.run_id)
        if (
            state.bootstrap_journal_path != expected_path
            or state.bootstrap_journal_sha256 is not None
            or expected_path.exists()
            or expected_path.is_symlink()
        ):
            raise FirecrackerClinicalRuntimeError('clinical bootstrap journal is not fresh')
        _write_new_private_journal(expected_path, payload)
        state.bootstrap_journal_sha256 = digest
        return digest

    def _require_authenticated_bootstrap_journal(
        self,
        state: _PreparedState,
        artifact: AuthenticatedClinicalGuestBootstrap,
    ) -> str:
        payload = canonical_json_bytes(artifact)
        digest = hashlib.sha256(payload).hexdigest()
        path = state.bootstrap_journal_path
        if (
            path is None
            or path != self._bootstrap_journal_path(state.public_receipt.run_id)
            or state.bootstrap_journal_sha256 is None
            or not hmac.compare_digest(state.bootstrap_journal_sha256, digest)
        ):
            raise FirecrackerClinicalRuntimeError('authenticated bootstrap was not durably journaled')
        retained = _read_private_bootstrap_journal(path)
        if not hmac.compare_digest(retained, payload):
            raise FirecrackerClinicalRuntimeError('authenticated bootstrap journal changed after creation')
        return digest

    def _bootstrap_matches_runtime_state(
        self,
        *,
        state: _PreparedState,
        start: ClinicalRuntimeStart,
        artifact: AuthenticatedClinicalGuestBootstrap,
    ) -> bool:
        hello = artifact.signed_hello.hello
        workspace = state.request.workspace
        expected_limits = ClinicalGuestRpcLimits(
            maximum_frame_body_bytes=self.guest_rpc_policy.maximum_frame_body_bytes,
            maximum_session_wire_bytes=self.guest_rpc_policy.maximum_session_wire_bytes,
            maximum_requests=self.guest_rpc_policy.maximum_requests,
            maximum_list_entries=self.guest_rpc_policy.maximum_list_entries,
            maximum_read_bytes=self.guest_rpc_policy.maximum_read_bytes,
            maximum_search_results=self.guest_rpc_policy.maximum_search_results,
            maximum_submission_bytes=self.guest_rpc_policy.maximum_submission_bytes,
        )
        expected = (
            state.public_receipt.run_id,
            start.start_redemption_sha256,
            state.public_receipt.guest_rpc_session_id,
            workspace.invocation,
            agentic_task_invocation_sha256(workspace.invocation),
            workspace.manifest_sha256,
            workspace.manifest.workspace_tree_sha256,
            workspace.manifest.model_visible_surface_sha256,
            agentic_policy_sha256(self.execution_policy),
            firecracker_model_sha256(self.supervisor.spec),
            expected_limits,
        )
        observed = (
            hello.run_id,
            hello.start_redemption_sha256,
            hello.session_id,
            hello.task_invocation,
            hello.task_invocation_sha256,
            hello.workspace_manifest_sha256,
            hello.workspace_tree_sha256,
            hello.model_visible_surface_sha256,
            hello.execution_policy_sha256,
            hello.worker_spec_sha256,
            hello.rpc_limits,
        )
        return expected == observed

    def _discard_consumed_state(self, state: _PreparedState) -> bool:
        cleaned = True
        cleanup: FirecrackerCleanupReceipt | None = None
        try:
            cleanup = self.supervisor.discard_prepared(state.worker)
            _validate_cleanup(cleanup, run_id=state.worker.run_id, lifecycle='never_launched')
        except BaseException:
            cleaned = False
        try:
            self.gateway_secret_store.revoke(state.public_receipt.gateway_capability_id)
        except BaseException:
            cleaned = False
        if cleaned and self._managed_ownership is not None:
            try:
                self._managed_ownership.record_cleaned(
                    run_id=state.worker.run_id,
                    terminal_reason='runtime_cleanup',
                    cleanup_receipt=cleanup,
                )
            except BaseException:
                cleaned = False
        state.capability_secret = b''
        return cleaned

    def _finalize_success(
        self,
        *,
        state: _PreparedState,
        attempt_sha256: str,
        bootstrap_artifact: AuthenticatedClinicalGuestBootstrap,
        worker_artifact: AuthenticatedFirecrackerWorkerAttestation,
        gateway_artifact: AuthenticatedGatewaySession,
        guest_artifact: AuthenticatedGuestRpcSession,
        submission: ExecutionSubmission,
    ) -> ClinicalRuntimeOutcome:
        """Versioned v0.2 outer binding is imported only after its exact API is available."""

        try:
            journal_sha256 = self._require_authenticated_bootstrap_journal(
                state,
                bootstrap_artifact,
            )
            from vaxreplay.agentic.clinical_production_run_v02 import (
                CLINICAL_PRODUCTION_RUN_V02_SCHEMA_VERSION,
                clinical_guest_bootstrap_evidence_sha256,
                finalize_clinical_production_run_v02,
                load_clinical_production_run_v02,
            )

            system = state.request.reservation.system
            output_root = self.evidence_root / state.public_receipt.run_id
            loaded = finalize_clinical_production_run_v02(
                output_root=output_root,
                run_id=state.public_receipt.run_id,
                workspace=state.request.workspace,
                expected_authenticated_workspace_receipt_sha256=(state.request.workspace.authenticated_receipt_sha256),
                workspace_receipt_key=self.keys.workspace_receipt_key,
                expected_workspace_receipt_key_id=(
                    state.request.workspace.authenticated_receipt.receipt.receipt_key_id
                ),
                attempt_reservation_sha256=attempt_sha256,
                policy=self.execution_policy,
                harness=self.harness,
                worker_spec=self.supervisor.spec,
                worker_attestation=worker_artifact,
                worker_attestation_key=self.keys.worker_attestation_key,
                expected_worker_attestation_key_id=system.worker_attestation_key_id,
                gateway_session=gateway_artifact,
                gateway_receipt_key=self.keys.gateway_receipt_key,
                expected_gateway_receipt_key_id=system.gateway_receipt_key_id,
                expected_gateway_policy_sha256=system.gateway_policy_sha256,
                expected_gateway_route_sha256=system.gateway_route_sha256,
                guest_rpc_session=guest_artifact,
                guest_rpc_receipt_key=self.keys.guest_rpc_receipt_key,
                expected_guest_rpc_receipt_key_id=system.guest_rpc_receipt_key_id,
                expected_guest_rpc_policy_sha256=system.guest_rpc_policy_sha256,
                submission=submission,
                clinical_guest_bootstrap=bootstrap_artifact,
                clinical_guest_bootstrap_receipt_key=(self.keys.clinical_guest_bootstrap_receipt_key),
                expected_clinical_guest_bootstrap_receipt_key_id=(self.config.bootstrap_receipt_key_id),
                clinical_guest_bootstrap_trust_anchor=self.bootstrap_trust_anchor,
                receipt_key=self.keys.production_receipt_key,
                expected_receipt_key_id=system.production_receipt_key_id,
                sealed_at=self._now(),
            )
            reloaded = load_clinical_production_run_v02(
                loaded.root,
                workspace=state.request.workspace,
                expected_authenticated_workspace_receipt_sha256=(state.request.workspace.authenticated_receipt_sha256),
                workspace_receipt_key=self.keys.workspace_receipt_key,
                expected_workspace_receipt_key_id=(
                    state.request.workspace.authenticated_receipt.receipt.receipt_key_id
                ),
                expected_run_id=state.public_receipt.run_id,
                expected_attempt_reservation_sha256=attempt_sha256,
                policy=self.execution_policy,
                harness=self.harness,
                worker_spec=self.supervisor.spec,
                worker_attestation_key=self.keys.worker_attestation_key,
                expected_worker_attestation_key_id=system.worker_attestation_key_id,
                gateway_receipt_key=self.keys.gateway_receipt_key,
                expected_gateway_receipt_key_id=system.gateway_receipt_key_id,
                expected_gateway_policy_sha256=system.gateway_policy_sha256,
                expected_gateway_route_sha256=system.gateway_route_sha256,
                guest_rpc_receipt_key=self.keys.guest_rpc_receipt_key,
                expected_guest_rpc_receipt_key_id=system.guest_rpc_receipt_key_id,
                expected_guest_rpc_policy_sha256=system.guest_rpc_policy_sha256,
                clinical_guest_bootstrap_receipt_key=(self.keys.clinical_guest_bootstrap_receipt_key),
                expected_clinical_guest_bootstrap_receipt_key_id=(self.config.bootstrap_receipt_key_id),
                clinical_guest_bootstrap_trust_anchor=self.bootstrap_trust_anchor,
                receipt_key=self.keys.production_receipt_key,
                expected_receipt_key_id=system.production_receipt_key_id,
            )
            bootstrap_sha256 = clinical_guest_bootstrap_evidence_sha256(bootstrap_artifact)
            if (
                journal_sha256 != bootstrap_sha256
                or reloaded.root != loaded.root
                or reloaded.clinical_guest_bootstrap_evidence_sha256 != bootstrap_sha256
            ):
                raise FirecrackerClinicalRuntimeError('independent v0.2 evidence reload differs')
            self._require_authenticated_bootstrap_journal(state, bootstrap_artifact)
            return ClinicalRuntimeCompleted(
                production_run_root=reloaded.root,
                production_evidence_schema_version=CLINICAL_PRODUCTION_RUN_V02_SCHEMA_VERSION,
                authenticated_bootstrap_sha256=bootstrap_sha256,
            )
        except Exception:
            bootstrap_sha256 = hashlib.sha256(canonical_json_bytes(bootstrap_artifact)).hexdigest()
            return ClinicalRuntimeFailed(
                ClinicalRuntimeFailureCode.EVIDENCE_FINALIZATION_FAILED,
                authenticated_bootstrap_sha256=bootstrap_sha256,
            )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise FirecrackerClinicalRuntimeError('runtime clock returned a naive timestamp')
        return value.astimezone(UTC)


def _clinical_guest_rpc_limits(policy: GuestRpcPolicy) -> ClinicalGuestRpcLimits:
    return ClinicalGuestRpcLimits(
        maximum_frame_body_bytes=policy.maximum_frame_body_bytes,
        maximum_session_wire_bytes=policy.maximum_session_wire_bytes,
        maximum_requests=policy.maximum_requests,
        maximum_list_entries=policy.maximum_list_entries,
        maximum_read_bytes=policy.maximum_read_bytes,
        maximum_search_results=policy.maximum_search_results,
        maximum_submission_bytes=policy.maximum_submission_bytes,
    )


def _scan_firecracker_clinical_bootstrap_journals(
    *,
    journal_root: Path,
    guest_rpc_policy: GuestRpcPolicy,
    bootstrap_receipt_key: bytes,
    bootstrap_trust_anchor: ClinicalGuestBootstrapTrustAnchor,
    expected_authorization_key_id: str,
    expected_receipt_key_id: str,
    expected_execution_policy_sha256: str,
    expected_worker_spec_sha256: str,
) -> tuple[FirecrackerClinicalStartupOrphan, ...]:
    """Authenticate one stable, exact journal inventory for runtime or recovery use."""

    try:
        entries = tuple(sorted(os.scandir(journal_root), key=lambda item: item.name))
    except OSError:
        raise FirecrackerClinicalRuntimeError('startup bootstrap journal inventory is unavailable') from None
    orphans: list[FirecrackerClinicalStartupOrphan] = []
    expected_limits = _clinical_guest_rpc_limits(guest_rpc_policy)
    for entry in entries:
        name = entry.name
        if len(name) != 37 or not name.endswith('.json'):
            raise FirecrackerClinicalRuntimeError('startup bootstrap journal inventory contains an unexpected entry')
        run_id = name[:-5]
        _require_lower_hex(run_id, 32, 'startup bootstrap journal run ID')
        path = journal_root / f'{run_id}.json'
        if path.parent != journal_root or path.name != name or Path(entry.path) != path:
            raise FirecrackerClinicalRuntimeError('startup bootstrap journal path is not canonical')
        payload = _read_private_bootstrap_journal(path)
        try:
            artifact = AuthenticatedClinicalGuestBootstrap.model_validate_json(payload)
        except ValueError:
            raise FirecrackerClinicalRuntimeError('startup bootstrap journal has an invalid strict schema') from None
        if canonical_json_bytes(artifact) != payload:
            raise FirecrackerClinicalRuntimeError('startup bootstrap journal is not canonical JSON')
        hello = artifact.signed_hello.hello
        try:
            verify_authenticated_clinical_guest_bootstrap(
                artifact,
                key=bootstrap_receipt_key,
                expected_key_id=expected_receipt_key_id,
                expected_hello=hello,
                trust_anchor=bootstrap_trust_anchor,
            )
        except BaseException:
            raise FirecrackerClinicalRuntimeError('startup bootstrap journal authentication failed') from None
        if (
            hello.run_id,
            hello.execution_policy_sha256,
            hello.worker_spec_sha256,
            hello.rpc_limits,
            artifact.signed_hello.authorization_key_id,
            artifact.receipt.receipt_key_id,
        ) != (
            run_id,
            expected_execution_policy_sha256,
            expected_worker_spec_sha256,
            expected_limits,
            expected_authorization_key_id,
            expected_receipt_key_id,
        ):
            raise FirecrackerClinicalRuntimeError('startup bootstrap journal differs from the runtime static pins')
        orphans.append(
            FirecrackerClinicalStartupOrphan(
                run_id=run_id,
                bootstrap_journal_file_name=name,
                bootstrap_journal_sha256=hashlib.sha256(payload).hexdigest(),
                start_redemption_sha256=hello.start_redemption_sha256,
                guest_rpc_session_id=hello.session_id,
                task_invocation_sha256=hello.task_invocation_sha256,
                workspace_manifest_sha256=hello.workspace_manifest_sha256,
                execution_policy_sha256=hello.execution_policy_sha256,
                worker_spec_sha256=hello.worker_spec_sha256,
                bootstrap_authorization_key_id=(artifact.signed_hello.authorization_key_id),
                bootstrap_receipt_key_id=artifact.receipt.receipt_key_id,
                bootstrap_ack_received_at=artifact.receipt.ack_received_at,
            )
        )
    return tuple(orphans)


def _write_new_private_journal(path: Path, payload: bytes) -> None:
    if not payload or len(payload) > _MAX_BOOTSTRAP_JOURNAL_BYTES:
        raise FirecrackerClinicalRuntimeError('clinical bootstrap journal has an invalid byte count')
    staging = path.parent / f'.{path.name}.{secrets.token_hex(16)}.tmp'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(staging, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short bootstrap journal write')
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(staging, path, follow_symlinks=False)
        staging.unlink()
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(payload)
        ):
            raise OSError('bootstrap journal metadata mismatch')
        _fsync_directory(path.parent)
    except OSError:
        raise FirecrackerClinicalRuntimeError('authenticated bootstrap journal could not be committed') from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def _read_private_bootstrap_journal(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise FirecrackerClinicalRuntimeError('authenticated bootstrap journal is unavailable') from None
    try:
        metadata = os.fstat(descriptor)
        try:
            path_metadata = path.lstat()
        except OSError:
            raise FirecrackerClinicalRuntimeError('authenticated bootstrap journal is unavailable') from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_BOOTSTRAP_JOURNAL_BYTES
            or (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise FirecrackerClinicalRuntimeError('authenticated bootstrap journal is not an exact private file')
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, _MAX_BOOTSTRAP_JOURNAL_BYTES - len(payload) + 1))
            if not chunk:
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > _MAX_BOOTSTRAP_JOURNAL_BYTES:
                raise FirecrackerClinicalRuntimeError('authenticated bootstrap journal exceeds its byte limit')
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_private_root(root: Path) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ValueError('clinical runtime evidence root cannot be a symbolic link')
    supplied.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = supplied.resolve()
    metadata = resolved.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError('clinical runtime evidence root must be a current-user-owned private mode-0700 directory')
    return resolved


def _validate_cleanup(
    receipt: FirecrackerCleanupReceipt,
    *,
    run_id: str,
    lifecycle: Literal['terminated', 'never_launched'],
) -> None:
    if receipt.run_id != run_id or receipt.lifecycle != lifecycle:
        raise FirecrackerClinicalRuntimeError('cleanup receipt differs from the worker lifecycle')


def _bootstrap_matches_guest_session(
    *,
    hello: ClinicalGuestBootstrapHello,
    session: GuestRpcHostSession,
    prepared: FirecrackerPreparedWorker,
    rpc_port: int,
) -> bool:
    """Cross-bind the signed authorization before accepting any guest connection."""

    expected_limits = ClinicalGuestRpcLimits(
        maximum_frame_body_bytes=session.policy.maximum_frame_body_bytes,
        maximum_session_wire_bytes=session.policy.maximum_session_wire_bytes,
        maximum_requests=session.policy.maximum_requests,
        maximum_list_entries=session.policy.maximum_list_entries,
        maximum_read_bytes=session.policy.maximum_read_bytes,
        maximum_search_results=session.policy.maximum_search_results,
        maximum_submission_bytes=session.policy.maximum_submission_bytes,
    )
    expected = (
        session.run_id,
        session.gateway_grant.attempt_reservation_sha256,
        session.session_id,
        session.task_invocation,
        agentic_task_invocation_sha256(session.task_invocation),
        session.workspace_manifest_sha256,
        session.workspace_tree_sha256,
        session.model_visible_surface_sha256,
        session.execution_policy_sha256,
        session.worker_spec_sha256,
        expected_limits,
        prepared.run_id,
        prepared.worker_spec_sha256,
        rpc_port,
    )
    observed = (
        hello.run_id,
        hello.start_redemption_sha256,
        hello.session_id,
        hello.task_invocation,
        hello.task_invocation_sha256,
        hello.workspace_manifest_sha256,
        hello.workspace_tree_sha256,
        hello.model_visible_surface_sha256,
        hello.execution_policy_sha256,
        hello.worker_spec_sha256,
        hello.rpc_limits,
        session.run_id,
        session.worker_spec_sha256,
        session.rpc_port,
    )
    return expected == observed


def _require_lower_hex(value: str, length: int, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise FirecrackerClinicalRuntimeError(f'{label} has an invalid identifier')


__all__ = [
    'FIRECRACKER_CLINICAL_RUNTIME_CONFIG_SCHEMA_VERSION',
    'FIRECRACKER_CLINICAL_RECOVERY_RECONCILIATION_REPORT_SCHEMA_VERSION',
    'FIRECRACKER_CLINICAL_STARTUP_CLEANUP_RECEIPT_SCHEMA_VERSION',
    'FIRECRACKER_CLINICAL_STARTUP_ORPHAN_SCHEMA_VERSION',
    'FIRECRACKER_CLINICAL_STARTUP_RECONCILIATION_REQUEST_SCHEMA_VERSION',
    'FIRECRACKER_CLINICAL_STARTUP_RECONCILIATION_REPORT_SCHEMA_VERSION',
    'ClinicalBootstrapRunnerFactory',
    'ClinicalBootstrapSessionRunner',
    'FirecrackerClinicalGuestBootstrapSession',
    'FirecrackerClinicalRecoveryReconciliationReport',
    'FirecrackerClinicalRuntime',
    'FirecrackerClinicalRuntimeConfig',
    'FirecrackerClinicalRuntimeError',
    'FirecrackerClinicalRuntimeKeys',
    'FirecrackerClinicalStartupCleanupReceipt',
    'FirecrackerClinicalStartupOrphan',
    'FirecrackerClinicalStartupReconciler',
    'FirecrackerClinicalStartupReconciliationRequest',
    'FirecrackerClinicalStartupReconciliationReport',
    'FirecrackerSupervisorBoundary',
    'GatewayCapabilitySecretStore',
    'ManagedClinicalRuntimeOwnershipLedger',
    'firecracker_clinical_runtime_config_sha256',
    'firecracker_clinical_startup_orphan_sha256',
    'firecracker_clinical_startup_reconciliation_request_sha256',
    'reconcile_firecracker_clinical_startup_without_execution',
]
