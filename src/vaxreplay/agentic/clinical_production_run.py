"""Authenticated Lane A production handoff for one clinical-execution task.

This is deliberately separate from the ranking run artifact: a Lane A workspace has its own
authenticated public-task receipt and no ranking admission object.  The artifact binds the exact
workspace receipt, worker lifecycle, provider gateway route and usage, guest-RPC trace, harness
image, policy, launch hash, and final ``ExecutionSubmission`` under an organizer-held HMAC key.

The package is development-only.  It does not attest immutable provider weights, a qualified
Linux/KVM host, complete tracing of guest-local computation or scratch writes, or absence of
identity/model-weight contamination.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_execution_bridge import (
    AuthenticatedClinicalAgenticWorkspace,
    LoadedClinicalAgenticWorkspace,
    load_clinical_agentic_workspace,
)
from vaxreplay.agentic.firecracker import (
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerAttestationError,
    FirecrackerWorkerAttestation,
    FirecrackerWorkerSpec,
    firecracker_model_sha256,
    firecracker_static_config_bytes,
    verify_firecracker_worker_attestation,
)
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcTerminalStatus,
    guest_rpc_policy_sha256,
    verify_authenticated_guest_rpc_session,
)
from vaxreplay.agentic.production_run import (
    ProductionAgenticRunError,
    _validate_worker_resources,
    _verify_rpc_gateway_exchange_consistency,
)
from vaxreplay.agentic.protocol import AgenticExecutionPolicy, AgenticModelUsage, agentic_policy_sha256
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewaySession,
    GatewayTerminalReason,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    verify_authenticated_gateway_session,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.run_artifact import AgenticHarnessIdentity
from vaxreplay.agentic.task_protocol import agentic_task_invocation_sha256, validate_submission_for_invocation
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission, ExecutionTask
from vaxreplay.runner.schema import IsolationTier

CLINICAL_PRODUCTION_RUN_RECEIPT_SCHEMA_VERSION = 'vaxreplay.clinical-production-run-receipt.dev-v0.1'
AUTHENTICATED_CLINICAL_PRODUCTION_RUN_SCHEMA_VERSION = 'vaxreplay.authenticated-clinical-production-run.dev-v0.1'
CLINICAL_PRODUCTION_RUN_AUTHENTICATION = 'hmac-sha256-domain-separated'

_KEY_ID_DOMAIN = b'vaxreplay.clinical-production-run-key-id.dev-v0.1\x00'
_HMAC_DOMAIN = b'vaxreplay.clinical-production-run-receipt.dev-v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_FILES = {
    'clinical-run.hmac',
    'clinical-run.json',
    'gateway-session.json',
    'guest-rpc-session.json',
    'submission.json',
    'worker-attestation.json',
    'workspace-receipt.json',
}


class ClinicalProductionRunError(ValueError):
    """Clinical production evidence is unauthenticated, incomplete, or cross-bound incorrectly."""


class ClinicalProductionRunReceipt(StrictModel):
    """Organizer-authenticated receipt for one successful Lane A task attempt."""

    schema_version: Literal['vaxreplay.clinical-production-run-receipt.dev-v0.1'] = (
        CLINICAL_PRODUCTION_RUN_RECEIPT_SCHEMA_VERSION
    )
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    authenticated_workspace_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness: AgenticHarnessIdentity
    resolved_model_id: str = Field(min_length=1)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_attestation_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_attestation_key_id: str = Field(pattern=_SHA256_PATTERN)
    gateway_session_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_route_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    gateway_transcript_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_attempt_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_session_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_attempt_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_projected_tool_events_sha256: str = Field(pattern=_SHA256_PATTERN)
    submission_sha256: str = Field(pattern=_SHA256_PATTERN)
    submission_bytes: int = Field(gt=0)
    usage: AgenticModelUsage
    gateway_attempt_count: int = Field(gt=0)
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)
    receipt_authentication: Literal['hmac-sha256-domain-separated'] = CLINICAL_PRODUCTION_RUN_AUTHENTICATION
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    response_protocol: Literal[AgenticResponseProtocol.CLINICAL_EXECUTION] = AgenticResponseProtocol.CLINICAL_EXECUTION
    accepted: Literal[True] = True
    authenticated_clinical_workspace: Literal[True] = True
    authenticated_worker_lifecycle: Literal[True] = True
    authenticated_provider_gateway: Literal[True] = True
    authenticated_guest_rpc_trace_present: Literal[True] = True
    provider_route_reported_model_and_usage_bound: Literal[True] = True
    harness_image_bound_to_worker: Literal[True] = True
    launch_hash_bound_across_worker_gateway_and_guest: Literal[True] = True
    final_submission_bound: Literal[True] = True
    guest_rpc_trace_scope: Literal['brokered_workspace_and_provider_gateway_methods_only'] = (
        'brokered_workspace_and_provider_gateway_methods_only'
    )
    direct_guest_local_compute_trace_present: Literal[False] = False
    direct_guest_scratch_write_trace_present: Literal[False] = False
    complete_guest_tool_trace_claimed: Literal[False] = False
    independently_attested_immutable_model_weights: Literal[False] = False
    external_provider_data_control_attested: Literal[False] = False
    linux_kvm_runtime_qualified: Literal[False] = False
    ancestor_symlink_race_hardened: Literal[False] = False
    global_single_execution_proven_by_artifact: Literal[False] = False
    model_weight_contamination_controlled: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    development_only: Literal[True] = True
    official_execution_qualified: Literal[False] = False
    leaderboard_admitted: Literal[False] = False
    sealed_at: datetime

    @field_validator('started_at', 'finished_at', 'sealed_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_interval_and_identity(self) -> Self:
        if self.finished_at < self.started_at or self.sealed_at < self.finished_at:
            raise ValueError('clinical production receipt timestamps are inconsistent')
        expected_duration = round((self.finished_at - self.started_at).total_seconds() * 1000)
        if self.duration_ms != expected_duration:
            raise ValueError('clinical production duration does not match worker timestamps')
        if self.harness.requested_model_id == '' or self.harness.adapter_id == '':
            raise ValueError('clinical production harness identity is incomplete')
        if self.usage.model_calls == 0 or self.gateway_attempt_count < self.usage.model_calls:
            raise ValueError('accepted clinical production runs require authenticated model use')
        return self


class AuthenticatedClinicalProductionRun(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-clinical-production-run.dev-v0.1'] = (
        AUTHENTICATED_CLINICAL_PRODUCTION_RUN_SCHEMA_VERSION
    )
    receipt: ClinicalProductionRunReceipt
    receipt_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class LoadedClinicalProductionRun:
    root: Path
    workspace: LoadedClinicalAgenticWorkspace
    submission: ExecutionSubmission
    worker_attestation: AuthenticatedFirecrackerWorkerAttestation
    gateway_session: AuthenticatedGatewaySession
    guest_rpc_session: AuthenticatedGuestRpcSession
    authenticated_receipt: AuthenticatedClinicalProductionRun
    authenticated_receipt_sha256: str

    @property
    def receipt(self) -> ClinicalProductionRunReceipt:
        return self.authenticated_receipt.receipt


def clinical_production_run_key_id(key: bytes) -> str:
    _require_key(key, 'clinical production receipt key')
    return _sha256(_KEY_ID_DOMAIN + key)


def clinical_production_run_receipt_hmac(receipt: ClinicalProductionRunReceipt, key: bytes) -> str:
    _require_key(key, 'clinical production receipt key')
    return hmac.new(key, _HMAC_DOMAIN + canonical_json_bytes(receipt), hashlib.sha256).hexdigest()


def finalize_clinical_production_run(
    *,
    output_root: Path,
    run_id: str,
    workspace: LoadedClinicalAgenticWorkspace,
    expected_authenticated_workspace_receipt_sha256: str,
    workspace_receipt_key: bytes,
    expected_workspace_receipt_key_id: str,
    attempt_reservation_sha256: str,
    policy: AgenticExecutionPolicy,
    harness: AgenticHarnessIdentity,
    worker_spec: FirecrackerWorkerSpec,
    worker_attestation: AuthenticatedFirecrackerWorkerAttestation,
    worker_attestation_key: bytes,
    expected_worker_attestation_key_id: str,
    gateway_session: AuthenticatedGatewaySession,
    gateway_receipt_key: bytes,
    expected_gateway_receipt_key_id: str,
    expected_gateway_policy_sha256: str,
    expected_gateway_route_sha256: str,
    guest_rpc_session: AuthenticatedGuestRpcSession,
    guest_rpc_receipt_key: bytes,
    expected_guest_rpc_receipt_key_id: str,
    expected_guest_rpc_policy_sha256: str,
    submission: ExecutionSubmission,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    sealed_at: datetime | None = None,
) -> LoadedClinicalProductionRun:
    """Re-authenticate and atomically materialize one exact Lane A production package."""

    verified_workspace = load_clinical_agentic_workspace(
        workspace.root,
        expected_authenticated_receipt_sha256=expected_authenticated_workspace_receipt_sha256,
        receipt_key=workspace_receipt_key,
        expected_receipt_key_id=expected_workspace_receipt_key_id,
    )
    submission = ExecutionSubmission.model_validate_json(canonical_json_bytes(submission))
    worker = _verify_clinical_execution_evidence(
        run_id=run_id,
        workspace=verified_workspace,
        attempt_reservation_sha256=attempt_reservation_sha256,
        policy=policy,
        harness=harness,
        worker_spec=worker_spec,
        worker_attestation=worker_attestation,
        worker_attestation_key=worker_attestation_key,
        expected_worker_attestation_key_id=expected_worker_attestation_key_id,
        gateway_session=gateway_session,
        gateway_receipt_key=gateway_receipt_key,
        expected_gateway_receipt_key_id=expected_gateway_receipt_key_id,
        expected_gateway_policy_sha256=expected_gateway_policy_sha256,
        expected_gateway_route_sha256=expected_gateway_route_sha256,
        guest_rpc_session=guest_rpc_session,
        guest_rpc_receipt_key=guest_rpc_receipt_key,
        expected_guest_rpc_receipt_key_id=expected_guest_rpc_receipt_key_id,
        expected_guest_rpc_policy_sha256=expected_guest_rpc_policy_sha256,
        submission=submission,
    )
    if clinical_production_run_key_id(receipt_key) != expected_receipt_key_id:
        raise ClinicalProductionRunError('clinical production receipt key does not match its expected key ID')

    worker_bytes = canonical_json_bytes(worker_attestation)
    gateway_bytes = canonical_json_bytes(gateway_session)
    guest_bytes = canonical_json_bytes(guest_rpc_session)
    workspace_receipt_bytes = canonical_json_bytes(verified_workspace.authenticated_receipt)
    submission_bytes = canonical_json_bytes(submission)
    usage = _gateway_usage(gateway_session)
    sealed = sealed_at or datetime.now(UTC)
    if sealed.tzinfo is None or sealed.utcoffset() is None:
        raise ClinicalProductionRunError('clinical production seal timestamp must include a UTC offset')
    sealed = sealed.astimezone(UTC)
    receipt = ClinicalProductionRunReceipt(
        run_id=run_id,
        attempt_reservation_sha256=attempt_reservation_sha256,
        workspace_manifest_sha256=verified_workspace.manifest_sha256,
        workspace_tree_sha256=verified_workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=verified_workspace.manifest.model_visible_surface_sha256,
        authenticated_workspace_receipt_sha256=verified_workspace.authenticated_receipt_sha256,
        workspace_receipt_key_id=expected_workspace_receipt_key_id,
        task_sha256=_sha256(canonical_json_bytes(verified_workspace.task)),
        task_context_sha256=verified_workspace.task.context_sha256,
        task_invocation_sha256=agentic_task_invocation_sha256(verified_workspace.invocation),
        execution_policy_sha256=agentic_policy_sha256(policy),
        harness=harness,
        resolved_model_id=gateway_session.route.resolved_model_id,
        worker_spec_sha256=firecracker_model_sha256(worker_spec),
        worker_attestation_sha256=_sha256(worker_bytes),
        worker_attestation_key_id=expected_worker_attestation_key_id,
        gateway_session_sha256=_sha256(gateway_bytes),
        gateway_policy_sha256=expected_gateway_policy_sha256,
        gateway_route_sha256=expected_gateway_route_sha256,
        gateway_receipt_key_id=expected_gateway_receipt_key_id,
        gateway_transcript_sha256=gateway_session.seal.transcript_sha256,
        gateway_attempt_log_sha256=gateway_session.seal.attempt_log_sha256,
        guest_rpc_session_sha256=_sha256(guest_bytes),
        guest_rpc_policy_sha256=expected_guest_rpc_policy_sha256,
        guest_rpc_receipt_key_id=expected_guest_rpc_receipt_key_id,
        guest_rpc_attempt_log_sha256=guest_rpc_session.seal.attempt_log_sha256,
        guest_rpc_projected_tool_events_sha256=guest_rpc_session.seal.projected_tool_events_sha256,
        submission_sha256=_sha256(submission_bytes),
        submission_bytes=len(submission_bytes),
        usage=usage,
        gateway_attempt_count=gateway_session.seal.attempt_count,
        started_at=worker.started_at,
        finished_at=worker.finished_at,
        duration_ms=worker.duration_ms,
        receipt_key_id=expected_receipt_key_id,
        sealed_at=sealed,
    )
    authenticated = AuthenticatedClinicalProductionRun(
        receipt=receipt,
        receipt_hmac_sha256=clinical_production_run_receipt_hmac(receipt, receipt_key),
    )
    target = output_root.expanduser()
    if target.is_symlink():
        raise ClinicalProductionRunError('clinical production output cannot be a symbolic link')
    target = target.resolve()
    if target.exists():
        raise ClinicalProductionRunError(f'clinical production output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    staging.chmod(0o700)
    try:
        files = {
            'clinical-run.json': canonical_json_bytes(authenticated),
            'clinical-run.hmac': (authenticated.receipt_hmac_sha256 + '\n').encode('ascii'),
            'workspace-receipt.json': workspace_receipt_bytes,
            'worker-attestation.json': worker_bytes,
            'gateway-session.json': gateway_bytes,
            'guest-rpc-session.json': guest_bytes,
            'submission.json': submission_bytes,
        }
        for name, payload in files.items():
            path = staging / name
            path.write_bytes(payload)
            path.chmod(0o600)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_clinical_production_run(
        target,
        workspace=verified_workspace,
        expected_authenticated_workspace_receipt_sha256=expected_authenticated_workspace_receipt_sha256,
        workspace_receipt_key=workspace_receipt_key,
        expected_workspace_receipt_key_id=expected_workspace_receipt_key_id,
        expected_run_id=run_id,
        expected_attempt_reservation_sha256=attempt_reservation_sha256,
        policy=policy,
        harness=harness,
        worker_spec=worker_spec,
        worker_attestation_key=worker_attestation_key,
        expected_worker_attestation_key_id=expected_worker_attestation_key_id,
        gateway_receipt_key=gateway_receipt_key,
        expected_gateway_receipt_key_id=expected_gateway_receipt_key_id,
        expected_gateway_policy_sha256=expected_gateway_policy_sha256,
        expected_gateway_route_sha256=expected_gateway_route_sha256,
        guest_rpc_receipt_key=guest_rpc_receipt_key,
        expected_guest_rpc_receipt_key_id=expected_guest_rpc_receipt_key_id,
        expected_guest_rpc_policy_sha256=expected_guest_rpc_policy_sha256,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
    )


def load_clinical_production_run(
    root: Path,
    *,
    workspace: LoadedClinicalAgenticWorkspace,
    expected_authenticated_workspace_receipt_sha256: str,
    workspace_receipt_key: bytes,
    expected_workspace_receipt_key_id: str,
    expected_run_id: str,
    expected_attempt_reservation_sha256: str,
    policy: AgenticExecutionPolicy,
    harness: AgenticHarnessIdentity,
    worker_spec: FirecrackerWorkerSpec,
    worker_attestation_key: bytes,
    expected_worker_attestation_key_id: str,
    gateway_receipt_key: bytes,
    expected_gateway_receipt_key_id: str,
    expected_gateway_policy_sha256: str,
    expected_gateway_route_sha256: str,
    guest_rpc_receipt_key: bytes,
    expected_guest_rpc_receipt_key_id: str,
    expected_guest_rpc_policy_sha256: str,
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> LoadedClinicalProductionRun:
    """Reload and independently verify every byte in a clinical production package."""

    verified_workspace = load_clinical_agentic_workspace(
        workspace.root,
        expected_authenticated_receipt_sha256=expected_authenticated_workspace_receipt_sha256,
        receipt_key=workspace_receipt_key,
        expected_receipt_key_id=expected_workspace_receipt_key_id,
    )
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ClinicalProductionRunError('clinical production root cannot be a symbolic link')
    resolved = supplied.resolve()
    if not resolved.is_dir() or stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise ClinicalProductionRunError('clinical production root must be a private mode-0700 directory')
    if {entry.name for entry in os.scandir(resolved)} != _FILES:
        raise ClinicalProductionRunError('clinical production exact file inventory mismatch')
    receipt_bytes = _read_private_file(resolved / 'clinical-run.json', _MAX_EVIDENCE_BYTES)
    receipt_hmac_bytes = _read_private_file(resolved / 'clinical-run.hmac', 65)
    workspace_receipt_bytes = _read_private_file(resolved / 'workspace-receipt.json', _MAX_EVIDENCE_BYTES)
    worker_bytes = _read_private_file(resolved / 'worker-attestation.json', _MAX_EVIDENCE_BYTES)
    gateway_bytes = _read_private_file(resolved / 'gateway-session.json', _MAX_EVIDENCE_BYTES)
    guest_bytes = _read_private_file(resolved / 'guest-rpc-session.json', _MAX_EVIDENCE_BYTES)
    submission_bytes = _read_private_file(resolved / 'submission.json', policy.limits.max_final_bytes)
    try:
        authenticated = AuthenticatedClinicalProductionRun.model_validate_json(receipt_bytes)
        retained_workspace_receipt = AuthenticatedClinicalAgenticWorkspace.model_validate_json(workspace_receipt_bytes)
        worker_attestation = AuthenticatedFirecrackerWorkerAttestation.model_validate_json(worker_bytes)
        gateway_session = AuthenticatedGatewaySession.model_validate_json(gateway_bytes)
        guest_rpc_session = AuthenticatedGuestRpcSession.model_validate_json(guest_bytes)
        submission = ExecutionSubmission.model_validate_json(submission_bytes)
    except ValueError as error:
        raise ClinicalProductionRunError('clinical production evidence has an invalid strict schema') from error
    values_and_bytes = (
        (authenticated, receipt_bytes),
        (retained_workspace_receipt, workspace_receipt_bytes),
        (worker_attestation, worker_bytes),
        (gateway_session, gateway_bytes),
        (guest_rpc_session, guest_bytes),
        (submission, submission_bytes),
    )
    if any(canonical_json_bytes(value) != payload for value, payload in values_and_bytes):
        raise ClinicalProductionRunError('clinical production evidence must use canonical JSON')
    if clinical_production_run_key_id(receipt_key) != expected_receipt_key_id:
        raise ClinicalProductionRunError('clinical production receipt key does not match its expected key ID')
    expected_hmac = clinical_production_run_receipt_hmac(authenticated.receipt, receipt_key)
    if not hmac.compare_digest(authenticated.receipt_hmac_sha256, expected_hmac) or not hmac.compare_digest(
        receipt_hmac_bytes, (expected_hmac + '\n').encode('ascii')
    ):
        raise ClinicalProductionRunError('clinical production receipt authentication failed')
    if retained_workspace_receipt != verified_workspace.authenticated_receipt:
        raise ClinicalProductionRunError('retained workspace receipt differs from the authenticated workspace')
    if authenticated.receipt.run_id != expected_run_id:
        raise ClinicalProductionRunError('clinical production receipt uses an unexpected run ID')

    worker = _verify_clinical_execution_evidence(
        run_id=authenticated.receipt.run_id,
        workspace=verified_workspace,
        attempt_reservation_sha256=expected_attempt_reservation_sha256,
        policy=policy,
        harness=harness,
        worker_spec=worker_spec,
        worker_attestation=worker_attestation,
        worker_attestation_key=worker_attestation_key,
        expected_worker_attestation_key_id=expected_worker_attestation_key_id,
        gateway_session=gateway_session,
        gateway_receipt_key=gateway_receipt_key,
        expected_gateway_receipt_key_id=expected_gateway_receipt_key_id,
        expected_gateway_policy_sha256=expected_gateway_policy_sha256,
        expected_gateway_route_sha256=expected_gateway_route_sha256,
        guest_rpc_session=guest_rpc_session,
        guest_rpc_receipt_key=guest_rpc_receipt_key,
        expected_guest_rpc_receipt_key_id=expected_guest_rpc_receipt_key_id,
        expected_guest_rpc_policy_sha256=expected_guest_rpc_policy_sha256,
        submission=submission,
    )
    expected_receipt = _receipt_bindings(
        workspace=verified_workspace,
        policy=policy,
        harness=harness,
        worker_spec=worker_spec,
        worker=worker,
        worker_attestation_bytes=worker_bytes,
        worker_attestation_key_id=expected_worker_attestation_key_id,
        gateway_session=gateway_session,
        gateway_session_bytes=gateway_bytes,
        gateway_policy_sha256=expected_gateway_policy_sha256,
        gateway_route_sha256=expected_gateway_route_sha256,
        gateway_receipt_key_id=expected_gateway_receipt_key_id,
        guest_rpc_session=guest_rpc_session,
        guest_rpc_session_bytes=guest_bytes,
        guest_rpc_policy_sha256_value=expected_guest_rpc_policy_sha256,
        guest_rpc_receipt_key_id=expected_guest_rpc_receipt_key_id,
        submission_bytes=submission_bytes,
        attempt_reservation_sha256=expected_attempt_reservation_sha256,
        receipt_key_id=expected_receipt_key_id,
    )
    receipt = authenticated.receipt
    actual_receipt = receipt.model_dump(mode='python')
    for field_name, expected in expected_receipt.items():
        if actual_receipt[field_name] != expected:
            raise ClinicalProductionRunError(f'clinical production receipt has a mismatched {field_name}')
    if receipt.sealed_at < max(worker.finished_at, gateway_session.seal.sealed_at, guest_rpc_session.seal.sealed_at):
        raise ClinicalProductionRunError('clinical production receipt predates authenticated execution evidence')
    return LoadedClinicalProductionRun(
        root=resolved,
        workspace=verified_workspace,
        submission=submission,
        worker_attestation=worker_attestation,
        gateway_session=gateway_session,
        guest_rpc_session=guest_rpc_session,
        authenticated_receipt=authenticated,
        authenticated_receipt_sha256=_sha256(receipt_bytes),
    )


def _verify_clinical_execution_evidence(
    *,
    run_id: str,
    workspace: LoadedClinicalAgenticWorkspace,
    attempt_reservation_sha256: str,
    policy: AgenticExecutionPolicy,
    harness: AgenticHarnessIdentity,
    worker_spec: FirecrackerWorkerSpec,
    worker_attestation: AuthenticatedFirecrackerWorkerAttestation,
    worker_attestation_key: bytes,
    expected_worker_attestation_key_id: str,
    gateway_session: AuthenticatedGatewaySession,
    gateway_receipt_key: bytes,
    expected_gateway_receipt_key_id: str,
    expected_gateway_policy_sha256: str,
    expected_gateway_route_sha256: str,
    guest_rpc_session: AuthenticatedGuestRpcSession,
    guest_rpc_receipt_key: bytes,
    expected_guest_rpc_receipt_key_id: str,
    expected_guest_rpc_policy_sha256: str,
    submission: ExecutionSubmission,
) -> FirecrackerWorkerAttestation:
    if policy.required_isolation != IsolationTier.DEVELOPMENT or (
        policy.response_protocol != AgenticResponseProtocol.CLINICAL_EXECUTION
    ):
        raise ClinicalProductionRunError('Lane A production artifacts currently require development isolation')
    try:
        worker = verify_firecracker_worker_attestation(
            worker_attestation,
            attestation_key=worker_attestation_key,
            expected_attestation_key_id=expected_worker_attestation_key_id,
            expected_run_id=run_id,
            expected_attempt_reservation_sha256=attempt_reservation_sha256,
            expected_worker_spec_sha256=firecracker_model_sha256(worker_spec),
        )
        verify_authenticated_gateway_session(
            gateway_session,
            receipt_key=gateway_receipt_key,
            expected_receipt_key_id=expected_gateway_receipt_key_id,
        )
        verify_authenticated_guest_rpc_session(
            guest_rpc_session,
            receipt_key=guest_rpc_receipt_key,
            expected_receipt_key_id=expected_guest_rpc_receipt_key_id,
            expected_run_id=run_id,
            expected_workspace_manifest_sha256=workspace.manifest_sha256,
            expected_execution_policy_sha256=agentic_policy_sha256(policy),
            expected_task_invocation_sha256=agentic_task_invocation_sha256(workspace.invocation),
            expected_response_protocol=AgenticResponseProtocol.CLINICAL_EXECUTION,
            expected_peer_cid=worker.guest_cid,
            expected_rpc_port=worker.guest_rpc_port,
        )
    except (ValueError, FirecrackerAttestationError) as error:
        raise ClinicalProductionRunError('worker, gateway, or guest-RPC authentication failed') from error
    if authenticated_gateway_policy_sha256(gateway_session.policy) != expected_gateway_policy_sha256 or (
        gateway_model_route_sha256(gateway_session.route) != expected_gateway_route_sha256
    ):
        raise ClinicalProductionRunError('authenticated gateway policy or model route differs from its pin')
    if guest_rpc_policy_sha256(guest_rpc_session.policy) != expected_guest_rpc_policy_sha256:
        raise ClinicalProductionRunError('authenticated guest-RPC policy differs from its pin')
    if guest_rpc_session.task_invocation != workspace.invocation:
        raise ClinicalProductionRunError('authenticated guest task invocation differs from the clinical workspace')
    grant = gateway_session.grant
    expected_gateway = (
        run_id,
        attempt_reservation_sha256,
        agentic_policy_sha256(policy),
        workspace.manifest_sha256,
        worker.guest_cid,
        policy.limits,
    )
    actual_gateway = (
        grant.run_id,
        grant.attempt_reservation_sha256,
        grant.execution_policy_sha256,
        grant.workspace_manifest_sha256,
        grant.expected_peer_cid,
        grant.limits,
    )
    if actual_gateway != expected_gateway or guest_rpc_session.gateway_grant != grant:
        raise ClinicalProductionRunError('gateway grant differs from the run, launch, workspace, or policy')
    guest_seal = guest_rpc_session.seal
    broker = workspace.brokered_surface()
    expected_guest = (
        run_id,
        attempt_reservation_sha256,
        agentic_policy_sha256(policy),
        workspace.manifest_sha256,
        workspace.manifest.workspace_tree_sha256,
        workspace.manifest.model_visible_surface_sha256,
        firecracker_model_sha256(worker_spec),
        worker.guest_cid,
        worker.guest_rpc_port,
        broker.contract_version,
        broker.contract_sha256,
    )
    actual_guest = (
        guest_seal.run_id,
        guest_seal.attempt_reservation_sha256,
        guest_seal.execution_policy_sha256,
        guest_seal.workspace_manifest_sha256,
        guest_seal.workspace_tree_sha256,
        guest_seal.model_visible_surface_sha256,
        guest_seal.worker_spec_sha256,
        guest_seal.observed_peer_cid,
        guest_seal.rpc_port,
        guest_seal.workspace_broker_contract_version,
        guest_seal.workspace_broker_contract_sha256,
    )
    if actual_guest != expected_guest:
        raise ClinicalProductionRunError('guest-RPC evidence differs from the clinical run boundary')
    expected_worker = (
        worker_spec.runtime.release,
        worker_spec.runtime.firecracker.sha256,
        worker_spec.runtime.jailer.sha256,
        worker_spec.images.kernel.sha256,
        worker_spec.images.rootfs.sha256,
        worker_spec.images.harness.sha256,
        worker_spec.images.scratch_template.sha256,
        _sha256(firecracker_static_config_bytes(worker_spec)),
        worker_spec.guest_cid,
        worker_spec.guest_rpc_port,
        worker_spec.limits.wall_seconds,
    )
    actual_worker = (
        worker.runtime_release,
        worker.firecracker_sha256,
        worker.jailer_sha256,
        worker.kernel_sha256,
        worker.rootfs_sha256,
        worker.harness_sha256,
        worker.initial_scratch_sha256,
        worker.config_sha256,
        worker.guest_cid,
        worker.guest_rpc_port,
        worker.wall_seconds,
    )
    if actual_worker != expected_worker:
        raise ClinicalProductionRunError('authenticated worker artifacts differ from the worker specification')
    if (
        harness.requested_model_id != gateway_session.route.logical_model_id
        or harness.adapter_id != gateway_session.route.adapter_id
        or harness.harness_image_or_commitment != f'sha256:{worker.harness_sha256}'
    ):
        raise ClinicalProductionRunError('harness identity differs from the worker image or model route')
    try:
        _validate_worker_resources(worker_spec, policy)
        _verify_rpc_gateway_exchange_consistency(guest_rpc_session, gateway_session)
    except ProductionAgenticRunError as error:
        raise ClinicalProductionRunError('worker resources or gateway/RPC exchanges differ') from error
    if (
        guest_seal.terminal_status != GuestRpcTerminalStatus.COMPLETED
        or not guest_seal.submit_accepted
        or not isinstance(guest_rpc_session.task_invocation.task, ExecutionTask)
        or not isinstance(guest_rpc_session.submission, ExecutionSubmission)
        or guest_rpc_session.submission != submission
    ):
        raise ClinicalProductionRunError('clinical guest did not emit the retained terminal submission')
    try:
        validate_submission_for_invocation(workspace.invocation, submission)
    except ValueError as error:
        raise ClinicalProductionRunError('clinical submission violates the authenticated task') from error
    if (
        gateway_session.seal.terminal_reason != GatewayTerminalReason.COMPLETED
        or gateway_session.seal.successful_call_count == 0
        or gateway_session.seal.successful_call_count != gateway_session.seal.attempt_count
        or worker.jailer_exit_code != 0
        or worker.wall_timeout_triggered
    ):
        raise ClinicalProductionRunError('accepted clinical run requires clean worker and provider completion')
    if not (worker.started_at <= guest_seal.started_at <= guest_seal.sealed_at <= worker.finished_at) or any(
        attempt.started_at < worker.started_at or attempt.finished_at > worker.finished_at
        for attempt in guest_rpc_session.attempts
    ):
        raise ClinicalProductionRunError('guest-RPC evidence falls outside the authenticated worker lifecycle')
    for attempt in gateway_session.attempts:
        result = attempt.provider_result
        if result is not None and not (
            worker.started_at <= result.started_at <= result.finished_at <= worker.finished_at
        ):
            raise ClinicalProductionRunError('provider evidence falls outside the authenticated worker lifecycle')
    return worker


def _receipt_bindings(
    *,
    workspace: LoadedClinicalAgenticWorkspace,
    policy: AgenticExecutionPolicy,
    harness: AgenticHarnessIdentity,
    worker_spec: FirecrackerWorkerSpec,
    worker: FirecrackerWorkerAttestation,
    worker_attestation_bytes: bytes,
    worker_attestation_key_id: str,
    gateway_session: AuthenticatedGatewaySession,
    gateway_session_bytes: bytes,
    gateway_policy_sha256: str,
    gateway_route_sha256: str,
    gateway_receipt_key_id: str,
    guest_rpc_session: AuthenticatedGuestRpcSession,
    guest_rpc_session_bytes: bytes,
    guest_rpc_policy_sha256_value: str,
    guest_rpc_receipt_key_id: str,
    submission_bytes: bytes,
    attempt_reservation_sha256: str,
    receipt_key_id: str,
) -> dict[str, object]:
    return {
        'run_id': worker.run_id,
        'attempt_reservation_sha256': attempt_reservation_sha256,
        'workspace_manifest_sha256': workspace.manifest_sha256,
        'workspace_tree_sha256': workspace.manifest.workspace_tree_sha256,
        'model_visible_surface_sha256': workspace.manifest.model_visible_surface_sha256,
        'authenticated_workspace_receipt_sha256': workspace.authenticated_receipt_sha256,
        'workspace_receipt_key_id': workspace.authenticated_receipt.receipt.receipt_key_id,
        'task_sha256': _sha256(canonical_json_bytes(workspace.task)),
        'task_context_sha256': workspace.task.context_sha256,
        'task_invocation_sha256': agentic_task_invocation_sha256(workspace.invocation),
        'execution_policy_sha256': agentic_policy_sha256(policy),
        'harness': harness.model_dump(mode='python'),
        'resolved_model_id': gateway_session.route.resolved_model_id,
        'worker_spec_sha256': firecracker_model_sha256(worker_spec),
        'worker_attestation_sha256': _sha256(worker_attestation_bytes),
        'worker_attestation_key_id': worker_attestation_key_id,
        'gateway_session_sha256': _sha256(gateway_session_bytes),
        'gateway_policy_sha256': gateway_policy_sha256,
        'gateway_route_sha256': gateway_route_sha256,
        'gateway_receipt_key_id': gateway_receipt_key_id,
        'gateway_transcript_sha256': gateway_session.seal.transcript_sha256,
        'gateway_attempt_log_sha256': gateway_session.seal.attempt_log_sha256,
        'guest_rpc_session_sha256': _sha256(guest_rpc_session_bytes),
        'guest_rpc_policy_sha256': guest_rpc_policy_sha256_value,
        'guest_rpc_receipt_key_id': guest_rpc_receipt_key_id,
        'guest_rpc_attempt_log_sha256': guest_rpc_session.seal.attempt_log_sha256,
        'guest_rpc_projected_tool_events_sha256': guest_rpc_session.seal.projected_tool_events_sha256,
        'submission_sha256': _sha256(submission_bytes),
        'submission_bytes': len(submission_bytes),
        'usage': _gateway_usage(gateway_session).model_dump(mode='python'),
        'gateway_attempt_count': gateway_session.seal.attempt_count,
        'started_at': worker.started_at,
        'finished_at': worker.finished_at,
        'duration_ms': worker.duration_ms,
        'receipt_key_id': receipt_key_id,
    }


def _gateway_usage(session: AuthenticatedGatewaySession) -> AgenticModelUsage:
    return AgenticModelUsage(
        model_calls=session.seal.successful_call_count,
        input_tokens=session.seal.input_tokens,
        output_tokens=session.seal.output_tokens,
        reasoning_tokens=session.seal.reasoning_tokens,
        provider_cost_usd=session.seal.provider_cost_usd,
        gateway_metering_authoritative=session.transcript.metering_authoritative,
    )


def _read_private_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ClinicalProductionRunError('cannot open clinical production evidence file') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size <= 0
            or metadata.st_size > maximum_bytes
        ):
            raise ClinicalProductionRunError('clinical production evidence must be private and bounded')
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes - len(payload) + 1))
            if not chunk:
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise ClinicalProductionRunError('clinical production evidence exceeds its byte limit')
    finally:
        os.close(descriptor)


def _require_key(key: bytes, label: str) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ClinicalProductionRunError(f'{label} must contain at least 32 bytes')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    'AUTHENTICATED_CLINICAL_PRODUCTION_RUN_SCHEMA_VERSION',
    'CLINICAL_PRODUCTION_RUN_RECEIPT_SCHEMA_VERSION',
    'AuthenticatedClinicalProductionRun',
    'ClinicalProductionRunError',
    'ClinicalProductionRunReceipt',
    'LoadedClinicalProductionRun',
    'clinical_production_run_key_id',
    'clinical_production_run_receipt_hmac',
    'finalize_clinical_production_run',
    'load_clinical_production_run',
]
