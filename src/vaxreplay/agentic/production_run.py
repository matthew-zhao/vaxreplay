"""Cross-bound handoff for authenticated microVM, guest-RPC, and gateway evidence.

The authenticated guest-RPC artifact covers only the host-observed workspace and provider-gateway
methods exposed by that protocol.  Guest-local computation and direct scratch-drive writes do not
traverse the RPC broker, so this wrapper deliberately keeps the complete-tool-trace and official
release claims false.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from vaxreplay.agentic.admission import AgenticWorkspaceAdmission
from vaxreplay.agentic.firecracker import (
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerAttestationError,
    FirecrackerWorkerAttestation,
    FirecrackerWorkerSpec,
    firecracker_model_sha256,
    verify_firecracker_worker_attestation,
)
from vaxreplay.agentic.gateway import AgenticModelRequest
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcMethod,
    GuestRpcTerminalStatus,
    ModelGenerateRequest,
    ModelGenerateResult,
    guest_rpc_policy_sha256,
    verify_authenticated_guest_rpc_session,
)
from vaxreplay.agentic.protocol import (
    AgenticExecutionPolicy,
    AgenticRunFailureCode,
    agentic_policy_sha256,
    agentic_receipt_key_id,
)
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewaySession,
    GatewayTerminalReason,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    verify_authenticated_gateway_session,
)
from vaxreplay.agentic.run_artifact import (
    AgenticHarnessIdentity,
    AgenticWorkspaceBrokerAttestation,
    LoadedAgenticRunArtifact,
    finalize_agentic_run,
    load_agentic_run_artifact,
)
from vaxreplay.agentic.task_protocol import AgenticTaskInvocation, agentic_task_invocation_sha256
from vaxreplay.agentic.workspace import AgenticLogicalWorkspaceBroker, LoadedAgenticWorkspace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.runner.schema import BackendCapabilities, IsolationTier

PRODUCTION_RUN_SEAL_SCHEMA_VERSION = 'vaxreplay.production-agentic-run-seal.v0.2'
AUTHENTICATED_PRODUCTION_RUN_SCHEMA_VERSION = 'vaxreplay.authenticated-production-agentic-run.v0.2'
PRODUCTION_RUN_AUTHENTICATION = 'hmac-sha256-domain-separated'

_PRODUCTION_HMAC_DOMAIN = b'vaxreplay.production-agentic-run.v0.2\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_FILES = {
    'gateway-session.json',
    'guest-rpc-session.json',
    'production-seal.hmac',
    'production-seal.json',
    'run',
    'worker-attestation.json',
}


class ProductionAgenticRunError(ValueError):
    """Raised when independently authenticated execution evidence does not cross-bind."""


class ProductionAgenticRunSeal(StrictModel):
    schema_version: Literal['vaxreplay.production-agentic-run-seal.v0.2'] = PRODUCTION_RUN_SEAL_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_attestation_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_session_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_session_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_route_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_transcript_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_attempt_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_projected_tool_events_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_attestation_key_id: str = Field(pattern=_SHA256_PATTERN)
    gateway_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    receipt_authentication: Literal['hmac-sha256-domain-separated'] = PRODUCTION_RUN_AUTHENTICATION
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    authenticated_worker_lifecycle: Literal[True] = True
    authenticated_provider_gateway: Literal[True] = True
    authenticated_guest_rpc_trace_present: Literal[True] = True
    guest_rpc_trace_scope: Literal['brokered_workspace_and_provider_gateway_methods_only'] = (
        'brokered_workspace_and_provider_gateway_methods_only'
    )
    direct_guest_local_compute_trace_present: Literal[False] = False
    direct_guest_scratch_write_trace_present: Literal[False] = False
    complete_guest_tool_trace_claimed: Literal[False] = False
    official_release_qualified: Literal[False] = False
    sealed_at: datetime

    @field_validator('sealed_at')
    @classmethod
    def validate_sealed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('production seal timestamp must include a UTC offset')
        return value.astimezone(UTC)


class AuthenticatedProductionAgenticRun(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-production-agentic-run.v0.2'] = (
        AUTHENTICATED_PRODUCTION_RUN_SCHEMA_VERSION
    )
    seal: ProductionAgenticRunSeal
    seal_hmac: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True)
class LoadedProductionAgenticRun:
    root: Path
    run: LoadedAgenticRunArtifact
    gateway_session: AuthenticatedGatewaySession
    guest_rpc_session: AuthenticatedGuestRpcSession
    worker_attestation: AuthenticatedFirecrackerWorkerAttestation
    authenticated_seal: AuthenticatedProductionAgenticRun


def production_run_seal_hmac(seal: ProductionAgenticRunSeal, receipt_key: bytes) -> str:
    if len(receipt_key) < 32:
        raise ValueError('production run receipt key must contain at least 32 bytes')
    return hmac.new(receipt_key, _PRODUCTION_HMAC_DOMAIN + canonical_json_bytes(seal), hashlib.sha256).hexdigest()


def finalize_production_agentic_run(
    *,
    output_root: Path,
    run_id: str,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
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
    workspace_broker_attestation: AgenticWorkspaceBrokerAttestation,
    scratch_files: Mapping[str, bytes],
    receipt_key: bytes,
    expected_receipt_key_id: str,
    failure_code: AgenticRunFailureCode | None = None,
) -> LoadedProductionAgenticRun:
    """Finalize a run from independently authenticated worker, RPC, and gateway observations."""

    worker = _verify_execution_evidence(
        run_id=run_id,
        workspace=workspace,
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
        failure_code=failure_code,
    )
    if policy.required_isolation != IsolationTier.DEVELOPMENT:
        raise ProductionAgenticRunError(
            'official execution remains blocked pending full local-compute/scratch tracing and runtime qualification'
        )
    if agentic_receipt_key_id(receipt_key) != expected_receipt_key_id:
        raise ProductionAgenticRunError('production receipt key does not match the expected key ID')

    target = output_root.expanduser()
    if target.is_symlink():
        raise ProductionAgenticRunError('production run output cannot be a symlink')
    target = target.resolve()
    if target.exists():
        raise ProductionAgenticRunError(f'production run output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    staging.chmod(0o700)
    try:
        capabilities = _derived_backend_capabilities(worker)
        run = finalize_agentic_run(
            output_root=staging / 'run',
            run_id=run_id,
            workspace=workspace,
            admission=admission,
            expected_admission_sha256=expected_admission_sha256,
            attempt_reservation_sha256=attempt_reservation_sha256,
            policy=policy,
            harness=harness,
            capabilities=capabilities,
            workspace_broker_attestation=workspace_broker_attestation,
            gateway_transcript=gateway_session.transcript,
            tool_events=guest_rpc_session.projected_tool_events,
            scratch_files=scratch_files,
            final_submission_bytes=(
                b'' if guest_rpc_session.submission is None else canonical_json_bytes(guest_rpc_session.submission)
            ),
            started_at=worker.started_at,
            finished_at=worker.finished_at,
            receipt_key=receipt_key,
            expected_receipt_key_id=expected_receipt_key_id,
            failure_code=failure_code,
            gateway_channel_isolation=True,
            tool_tracing_authoritative=False,
            provider_cost_usd=gateway_session.seal.provider_cost_usd,
        )
        worker_bytes = canonical_json_bytes(worker_attestation)
        gateway_bytes = canonical_json_bytes(gateway_session)
        guest_rpc_bytes = canonical_json_bytes(guest_rpc_session)
        seal = ProductionAgenticRunSeal(
            run_id=run_id,
            attempt_reservation_sha256=attempt_reservation_sha256,
            workspace_manifest_sha256=workspace.manifest_sha256,
            execution_policy_sha256=agentic_policy_sha256(policy),
            run_receipt_sha256=run.receipt_sha256,
            worker_attestation_sha256=_sha256(worker_bytes),
            gateway_session_sha256=_sha256(gateway_bytes),
            guest_rpc_session_sha256=_sha256(guest_rpc_bytes),
            worker_spec_sha256=firecracker_model_sha256(worker_spec),
            gateway_policy_sha256=expected_gateway_policy_sha256,
            gateway_route_sha256=gateway_model_route_sha256(gateway_session.route),
            gateway_transcript_sha256=gateway_session.seal.transcript_sha256,
            guest_rpc_policy_sha256=guest_rpc_policy_sha256(guest_rpc_session.policy),
            guest_rpc_attempt_log_sha256=guest_rpc_session.seal.attempt_log_sha256,
            guest_rpc_projected_tool_events_sha256=(guest_rpc_session.seal.projected_tool_events_sha256),
            worker_attestation_key_id=expected_worker_attestation_key_id,
            gateway_receipt_key_id=expected_gateway_receipt_key_id,
            guest_rpc_receipt_key_id=expected_guest_rpc_receipt_key_id,
            receipt_key_id=expected_receipt_key_id,
            sealed_at=datetime.now(UTC),
        )
        authenticated = AuthenticatedProductionAgenticRun(
            seal=seal,
            seal_hmac=production_run_seal_hmac(seal, receipt_key),
        )
        files = {
            'worker-attestation.json': worker_bytes,
            'gateway-session.json': gateway_bytes,
            'guest-rpc-session.json': guest_rpc_bytes,
            'production-seal.json': canonical_json_bytes(authenticated),
            'production-seal.hmac': (authenticated.seal_hmac + '\n').encode('ascii'),
        }
        for name, payload in files.items():
            path = staging / name
            path.write_bytes(payload)
            path.chmod(0o600)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_production_agentic_run(
        target,
        workspace=workspace,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
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


def load_production_agentic_run(
    root: Path,
    *,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
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
) -> LoadedProductionAgenticRun:
    """Re-authenticate the complete outer evidence package and its inner run artifact."""

    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ProductionAgenticRunError('production run root cannot be a symlink')
    resolved = supplied.resolve()
    if not resolved.is_dir() or stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise ProductionAgenticRunError('production run root must be a private mode-0700 directory')
    if {entry.name for entry in os.scandir(resolved)} != _FILES:
        raise ProductionAgenticRunError('production run exact file inventory mismatch')
    worker_bytes = _read_private_file(resolved / 'worker-attestation.json', _MAX_EVIDENCE_BYTES)
    gateway_bytes = _read_private_file(resolved / 'gateway-session.json', _MAX_EVIDENCE_BYTES)
    guest_rpc_bytes = _read_private_file(resolved / 'guest-rpc-session.json', _MAX_EVIDENCE_BYTES)
    seal_bytes = _read_private_file(resolved / 'production-seal.json', _MAX_EVIDENCE_BYTES)
    seal_hmac_bytes = _read_private_file(resolved / 'production-seal.hmac', 65)
    try:
        worker_attestation = AuthenticatedFirecrackerWorkerAttestation.model_validate_json(worker_bytes)
        gateway_session = AuthenticatedGatewaySession.model_validate_json(gateway_bytes)
        guest_rpc_session = AuthenticatedGuestRpcSession.model_validate_json(guest_rpc_bytes)
        authenticated = AuthenticatedProductionAgenticRun.model_validate_json(seal_bytes)
    except ValueError as error:
        raise ProductionAgenticRunError('production execution evidence has an invalid strict schema') from error
    if (
        canonical_json_bytes(worker_attestation) != worker_bytes
        or canonical_json_bytes(gateway_session) != gateway_bytes
        or canonical_json_bytes(guest_rpc_session) != guest_rpc_bytes
        or canonical_json_bytes(authenticated) != seal_bytes
    ):
        raise ProductionAgenticRunError('production execution evidence must use canonical JSON')
    if agentic_receipt_key_id(receipt_key) != expected_receipt_key_id:
        raise ProductionAgenticRunError('production receipt key does not match the expected key ID')
    expected_hmac = production_run_seal_hmac(authenticated.seal, receipt_key)
    if not hmac.compare_digest(authenticated.seal_hmac, expected_hmac) or not hmac.compare_digest(
        seal_hmac_bytes, (expected_hmac + '\n').encode('ascii')
    ):
        raise ProductionAgenticRunError('production execution seal authentication failed')

    run = load_agentic_run_artifact(
        resolved / 'run',
        workspace=workspace,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
        expected_attempt_reservation_sha256=expected_attempt_reservation_sha256,
        policy=policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
    )
    _require_inner_run_harness_binding(run, harness)
    failure_code = run.receipt.failure_code
    _verify_execution_evidence(
        run_id=run.receipt.run_id,
        workspace=workspace,
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
        failure_code=failure_code,
    )
    seal = authenticated.seal
    expected_bindings = (
        run.receipt.run_id,
        expected_attempt_reservation_sha256,
        workspace.manifest_sha256,
        agentic_policy_sha256(policy),
        run.receipt_sha256,
        _sha256(worker_bytes),
        _sha256(gateway_bytes),
        _sha256(guest_rpc_bytes),
        firecracker_model_sha256(worker_spec),
        expected_gateway_policy_sha256,
        expected_gateway_route_sha256,
        gateway_session.seal.transcript_sha256,
        expected_guest_rpc_policy_sha256,
        guest_rpc_session.seal.attempt_log_sha256,
        guest_rpc_session.seal.projected_tool_events_sha256,
        expected_worker_attestation_key_id,
        expected_gateway_receipt_key_id,
        expected_guest_rpc_receipt_key_id,
        expected_receipt_key_id,
    )
    actual_bindings = (
        seal.run_id,
        seal.attempt_reservation_sha256,
        seal.workspace_manifest_sha256,
        seal.execution_policy_sha256,
        seal.run_receipt_sha256,
        seal.worker_attestation_sha256,
        seal.gateway_session_sha256,
        seal.guest_rpc_session_sha256,
        seal.worker_spec_sha256,
        seal.gateway_policy_sha256,
        seal.gateway_route_sha256,
        seal.gateway_transcript_sha256,
        seal.guest_rpc_policy_sha256,
        seal.guest_rpc_attempt_log_sha256,
        seal.guest_rpc_projected_tool_events_sha256,
        seal.worker_attestation_key_id,
        seal.gateway_receipt_key_id,
        seal.guest_rpc_receipt_key_id,
        seal.receipt_key_id,
    )
    if actual_bindings != expected_bindings:
        raise ProductionAgenticRunError('production seal does not bind the exact run, worker, gateway, and policy')
    if seal.sealed_at < max(
        worker_attestation.attestation.finished_at,
        gateway_session.seal.sealed_at,
        guest_rpc_session.seal.sealed_at,
    ):
        raise ProductionAgenticRunError('production seal predates authenticated execution evidence')
    if run.transcript != gateway_session.transcript:
        raise ProductionAgenticRunError('inner run transcript differs from the authenticated gateway transcript')
    if run.tool_events != guest_rpc_session.projected_tool_events:
        raise ProductionAgenticRunError('inner run tool events differ from the authenticated guest RPC projection')
    if (run.receipt.final_submission_sha256, run.receipt.final_submission_bytes) != (
        guest_rpc_session.seal.final_submission_sha256,
        guest_rpc_session.seal.final_submission_bytes,
    ):
        raise ProductionAgenticRunError('inner run submission differs from the authenticated guest RPC submission')
    if run.receipt.usage.provider_cost_usd != gateway_session.seal.provider_cost_usd:
        raise ProductionAgenticRunError('inner run cost differs from authenticated provider evidence')
    if (run.receipt.started_at, run.receipt.finished_at, run.receipt.duration_ms) != (
        worker_attestation.attestation.started_at,
        worker_attestation.attestation.finished_at,
        worker_attestation.attestation.duration_ms,
    ):
        raise ProductionAgenticRunError('inner run interval differs from the authenticated worker lifecycle')
    return LoadedProductionAgenticRun(
        root=resolved,
        run=run,
        gateway_session=gateway_session,
        guest_rpc_session=guest_rpc_session,
        worker_attestation=worker_attestation,
        authenticated_seal=authenticated,
    )


def _require_inner_run_harness_binding(
    run: LoadedAgenticRunArtifact,
    harness: AgenticHarnessIdentity,
) -> None:
    receipt = run.receipt
    observed = (
        receipt.harness_id,
        receipt.harness_version,
        receipt.harness_image_or_commitment,
        receipt.harness_manifest_sha256,
        receipt.harness_behavior_sha256,
        receipt.harness_execution_mode,
        receipt.requested_model_id,
        receipt.adapter_id,
    )
    expected = (
        harness.harness_id,
        harness.harness_version,
        harness.harness_image_or_commitment,
        harness.harness_manifest_sha256,
        harness.harness_behavior_sha256,
        harness.harness_execution_mode,
        harness.requested_model_id,
        harness.adapter_id,
    )
    if observed != expected:
        raise ProductionAgenticRunError(
            'inner run harness identity differs from the externally pinned production harness'
        )


def _verify_execution_evidence(
    *,
    run_id: str,
    workspace: LoadedAgenticWorkspace,
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
    failure_code: AgenticRunFailureCode | None,
) -> FirecrackerWorkerAttestation:
    task_invocation = AgenticTaskInvocation.from_task(
        workspace.task,
        workspace_manifest_sha256=workspace.manifest_sha256,
    )
    if task_invocation.response_protocol != policy.response_protocol:
        raise ProductionAgenticRunError('execution policy response protocol does not match the workspace task')
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
            expected_task_invocation_sha256=agentic_task_invocation_sha256(task_invocation),
            expected_response_protocol=task_invocation.response_protocol,
            expected_peer_cid=worker.guest_cid,
            expected_rpc_port=worker.guest_rpc_port,
        )
    except (ValueError, FirecrackerAttestationError) as error:
        raise ProductionAgenticRunError('worker, gateway, or guest RPC evidence authentication failed') from error
    grant = gateway_session.grant
    if (
        authenticated_gateway_policy_sha256(gateway_session.policy) != expected_gateway_policy_sha256
        or gateway_model_route_sha256(gateway_session.route) != expected_gateway_route_sha256
    ):
        raise ProductionAgenticRunError('authenticated gateway policy or model route differs from the release pins')
    if guest_rpc_policy_sha256(guest_rpc_session.policy) != expected_guest_rpc_policy_sha256:
        raise ProductionAgenticRunError('authenticated guest RPC policy differs from the release pin')
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
    if actual_gateway != expected_gateway:
        raise ProductionAgenticRunError('authenticated gateway grant is bound to a different run, worker, or policy')
    guest_seal = guest_rpc_session.seal
    expected_guest_boundary = (
        run_id,
        attempt_reservation_sha256,
        agentic_policy_sha256(policy),
        workspace.manifest_sha256,
        workspace.manifest.workspace_tree_sha256,
        workspace.manifest.model_visible_surface_sha256,
        firecracker_model_sha256(worker_spec),
        worker.guest_cid,
        worker.guest_rpc_port,
        AgenticLogicalWorkspaceBroker.contract_version,
        AgenticLogicalWorkspaceBroker.contract_sha256,
        gateway_session.grant,
    )
    actual_guest_boundary = (
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
        guest_rpc_session.gateway_grant,
    )
    if actual_guest_boundary != expected_guest_boundary:
        raise ProductionAgenticRunError(
            'authenticated guest RPC session is bound to a different run, worker, workspace, or gateway grant'
        )
    _verify_rpc_gateway_exchange_consistency(guest_rpc_session, gateway_session)
    if (
        harness.requested_model_id != gateway_session.route.logical_model_id
        or harness.adapter_id != gateway_session.route.adapter_id
        or harness.harness_image_or_commitment != f'sha256:{worker.harness_sha256}'
    ):
        raise ProductionAgenticRunError('harness identity does not match the authenticated worker and model route')
    _validate_worker_resources(worker_spec, policy)
    accepted = failure_code is None
    if accepted and (
        guest_seal.terminal_status != GuestRpcTerminalStatus.COMPLETED
        or not guest_seal.submit_accepted
        or guest_rpc_session.submission is None
    ):
        raise ProductionAgenticRunError('accepted runs require an authenticated guest RPC submission')
    if accepted and (
        gateway_session.seal.terminal_reason != GatewayTerminalReason.COMPLETED
        or gateway_session.seal.successful_call_count == 0
        or gateway_session.seal.successful_call_count != gateway_session.seal.attempt_count
        or worker.jailer_exit_code != 0
        or worker.wall_timeout_triggered
    ):
        raise ProductionAgenticRunError(
            'accepted runs require a clean worker exit and a completely successful gateway session'
        )
    if worker.wall_timeout_triggered != (failure_code == AgenticRunFailureCode.TIMED_OUT):
        raise ProductionAgenticRunError('worker wall-time outcome does not match the run failure code')
    for attempt in gateway_session.attempts:
        result = attempt.provider_result
        if result is not None and not (
            worker.started_at <= result.started_at <= result.finished_at <= worker.finished_at
            and grant.issued_at <= result.started_at < grant.expires_at
        ):
            raise ProductionAgenticRunError(
                'provider attempt falls outside the authenticated worker or capability interval'
            )
    if not (worker.started_at <= guest_seal.started_at <= guest_seal.sealed_at <= worker.finished_at) or any(
        attempt.started_at < worker.started_at or attempt.finished_at > worker.finished_at
        for attempt in guest_rpc_session.attempts
    ):
        raise ProductionAgenticRunError('guest RPC session falls outside the authenticated worker lifecycle')
    return worker


def _validate_worker_resources(spec: FirecrackerWorkerSpec, policy: AgenticExecutionPolicy) -> None:
    limits = spec.limits
    policy_limits = policy.limits
    cpu_capacity = limits.cpu_quota_us / limits.cpu_period_us
    if (
        limits.wall_seconds != policy_limits.wall_seconds
        or limits.memory_mib != policy_limits.memory_mib
        or limits.pids != policy_limits.pids
        or limits.scratch_bytes != policy_limits.scratch_mib * 1024 * 1024
        or not math.isclose(cpu_capacity, policy_limits.cpus, rel_tol=0, abs_tol=1e-9)
        or limits.vcpu_count < math.ceil(policy_limits.cpus)
    ):
        raise ProductionAgenticRunError('Firecracker resources do not exactly implement the execution policy')


def _verify_rpc_gateway_exchange_consistency(
    guest_rpc_session: AuthenticatedGuestRpcSession,
    gateway_session: AuthenticatedGatewaySession,
) -> None:
    """Require both independently authenticated observers to describe the same model calls."""

    rpc_model_attempts = tuple(
        attempt
        for attempt in guest_rpc_session.attempts
        if attempt.sequence_accepted
        and attempt.request.method == GuestRpcMethod.MODEL_GENERATE.value
        and attempt.gateway_call_index is not None
    )
    gateway_attempts = gateway_session.attempts
    if len(rpc_model_attempts) != len(gateway_attempts):
        raise ProductionAgenticRunError('guest RPC and provider gateway observed different model-call counts')

    exchanges_by_index = {exchange.request.call_index: exchange for exchange in gateway_session.transcript.exchanges}
    if len(exchanges_by_index) != len(gateway_session.transcript.exchanges):
        raise ProductionAgenticRunError('provider gateway transcript repeats a model-call index')

    successful_rpc_indexes: set[int] = set()
    for rpc_attempt, gateway_attempt in zip(rpc_model_attempts, gateway_attempts, strict=True):
        call_index = rpc_attempt.gateway_call_index
        if call_index is None or gateway_attempt.call_index != call_index:
            raise ProductionAgenticRunError('guest RPC and provider gateway model-call indexes differ')
        try:
            rpc_request = ModelGenerateRequest.model_validate_json(canonical_json_bytes(rpc_attempt.request.body))
        except ValueError as error:
            raise ProductionAgenticRunError('guest RPC model request is not valid under its closed schema') from error
        inner_request = AgenticModelRequest(
            run_id=guest_rpc_session.seal.run_id,
            call_index=call_index,
            messages=rpc_request.messages,
            max_output_tokens=rpc_request.max_output_tokens,
            response_schema_sha256=rpc_request.response_schema_sha256,
        )
        inner_request_bytes = canonical_json_bytes(inner_request)
        if (gateway_attempt.request_sha256, gateway_attempt.request_bytes) != (
            _sha256(inner_request_bytes),
            len(inner_request_bytes),
        ):
            raise ProductionAgenticRunError('guest RPC and provider gateway model requests differ')

        if rpc_attempt.response.succeeded != gateway_attempt.succeeded:
            raise ProductionAgenticRunError('guest RPC and provider gateway model-call outcomes differ')
        exchange = exchanges_by_index.get(call_index)
        if rpc_attempt.response.succeeded:
            if exchange is None or rpc_attempt.response.result is None:
                raise ProductionAgenticRunError('successful guest RPC model call is missing gateway exchange evidence')
            try:
                rpc_result = ModelGenerateResult.model_validate_json(canonical_json_bytes(rpc_attempt.response.result))
            except ValueError as error:
                raise ProductionAgenticRunError(
                    'guest RPC model response is not valid under its closed schema'
                ) from error
            if exchange.request != inner_request or exchange.response != rpc_result.response:
                raise ProductionAgenticRunError('guest RPC and provider gateway model exchange bodies differ')
            successful_rpc_indexes.add(call_index)
        elif exchange is not None:
            raise ProductionAgenticRunError(
                'failed guest RPC model call unexpectedly has a successful gateway exchange'
            )

    if successful_rpc_indexes != set(exchanges_by_index):
        raise ProductionAgenticRunError('guest RPC and provider gateway successful exchange sets differ')


def _derived_backend_capabilities(worker: FirecrackerWorkerAttestation) -> BackendCapabilities:
    return BackendCapabilities(
        backend_id='vaxreplay-firecracker',
        backend_version=worker.runtime_release,
        isolation_tier=IsolationTier.DEVELOPMENT,
        network_isolation=worker.network_interfaces_absent,
        host_filesystem_isolation=True,
        read_only_root=worker.rootfs_read_only,
        # The host-side jailer UID is attested, but the guest harness UID is not yet attested.
        non_root_user=False,
        capability_drop=False,
        no_new_privileges=False,
        process_limit=worker.cgroup_cpu_memory_pids_enforced,
        memory_limit=worker.cgroup_cpu_memory_pids_enforced,
        cpu_limit=worker.cgroup_cpu_memory_pids_enforced,
        scratch_limit=worker.fresh_bounded_writable_scratch,
        fresh_worker_per_episode=worker.fresh_bounded_writable_scratch,
    )


def _read_private_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProductionAgenticRunError('cannot open production evidence file') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum_bytes
        ):
            raise ProductionAgenticRunError('production evidence must be a private bounded regular file')
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes - len(payload) + 1))
            if not chunk:
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise ProductionAgenticRunError('production evidence file exceeds its byte limit')
    finally:
        os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
