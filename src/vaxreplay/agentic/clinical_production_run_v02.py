"""Versioned Lane A production evidence with a signed guest-bootstrap binding.

The legacy ``clinical_production_run`` v0.1 package intentionally remains unchanged.  This module
adds one exact evidence file and authenticates a v0.2 outer receipt.  The retained bootstrap file
is the complete canonical ``AuthenticatedClinicalGuestBootstrap``: it includes the exact
launcher-signed hello and the host-authenticated acknowledgement receipt.  Loading is fail closed:
the v0.1 evidence is independently re-authenticated, the launcher signature is checked against an
out-of-band guest trust anchor, and all bootstrap/run identities and timestamps are cross-checked.

This remains a development artifact.  In particular, binding a trust anchor does not prove that it
was baked into a measured guest image, and this module does not qualify a Linux/KVM deployment or
admit a run to an official leaderboard.
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

from vaxreplay.agentic.clinical_execution_bridge import LoadedClinicalAgenticWorkspace
from vaxreplay.agentic.clinical_guest_bootstrap import (
    AuthenticatedClinicalGuestBootstrap,
    ClinicalGuestBootstrapError,
    ClinicalGuestBootstrapTrustAnchor,
    ClinicalGuestRpcLimits,
    clinical_guest_bootstrap_signed_hello_sha256,
    verify_authenticated_clinical_guest_bootstrap,
)
from vaxreplay.agentic.clinical_production_run import (
    AuthenticatedClinicalProductionRun,
    ClinicalProductionRunError,
    LoadedClinicalProductionRun,
    clinical_production_run_key_id,
    finalize_clinical_production_run,
    load_clinical_production_run,
)
from vaxreplay.agentic.firecracker import (
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerWorkerSpec,
)
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
)
from vaxreplay.agentic.protocol import AgenticExecutionPolicy
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewaySession,
)
from vaxreplay.agentic.run_artifact import AgenticHarnessIdentity
from vaxreplay.agentic.task_protocol import agentic_task_invocation_sha256
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission

CLINICAL_PRODUCTION_RUN_V02_SCHEMA_VERSION = 'vaxreplay.authenticated-clinical-production-run.dev-v0.2'
CLINICAL_PRODUCTION_RUN_OUTER_RECEIPT_V02_SCHEMA_VERSION = 'vaxreplay.clinical-production-run-outer-receipt.dev-v0.2'
CLINICAL_PRODUCTION_RUN_V02_AUTHENTICATION = 'hmac-sha256-domain-separated'

_HMAC_DOMAIN_V02 = b'vaxreplay.clinical-production-run-outer-receipt.dev-v0.2\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_FILES_V02 = {
    'clinical-guest-bootstrap.json',
    'clinical-run.hmac',
    'clinical-run.json',
    'gateway-session.json',
    'guest-rpc-session.json',
    'submission.json',
    'worker-attestation.json',
    'workspace-receipt.json',
}
_LEGACY_COMPONENT_FILES = _FILES_V02 - {'clinical-guest-bootstrap.json', 'clinical-run.hmac', 'clinical-run.json'}


class ClinicalProductionRunOuterReceiptV02(StrictModel):
    """Organizer-authenticated binding from legacy evidence to the exact bootstrap exchange."""

    schema_version: Literal['vaxreplay.clinical-production-run-outer-receipt.dev-v0.2'] = (
        CLINICAL_PRODUCTION_RUN_OUTER_RECEIPT_V02_SCHEMA_VERSION
    )
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    start_redemption_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_session_id: str = Field(pattern=_RUN_ID_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_limits_sha256: str = Field(pattern=_SHA256_PATTERN)
    base_authenticated_run_sha256: str = Field(pattern=_SHA256_PATTERN)
    clinical_guest_bootstrap_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    clinical_guest_bootstrap_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    clinical_guest_bootstrap_authorization_key_id: str = Field(pattern=_SHA256_PATTERN)
    clinical_guest_bootstrap_signed_hello_sha256: str = Field(pattern=_SHA256_PATTERN)
    clinical_guest_bootstrap_hello_sha256: str = Field(pattern=_SHA256_PATTERN)
    bootstrap_valid_from: datetime
    bootstrap_expires_at: datetime
    bootstrap_hello_sent_at: datetime
    bootstrap_ack_received_at: datetime
    bootstrap_guest_accepted_at: datetime
    sealed_at: datetime
    receipt_authentication: Literal['hmac-sha256-domain-separated'] = CLINICAL_PRODUCTION_RUN_V02_AUTHENTICATION
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    authenticated_base_run_bound: Literal[True] = True
    exact_signed_bootstrap_hello_retained: Literal[True] = True
    authenticated_bootstrap_receipt_retained: Literal[True] = True
    bootstrap_bound_into_outer_run_receipt: Literal[True] = True
    bootstrap_precedes_first_guest_rpc_attempt: Literal[True] = True
    bootstrap_launcher_signature_verified: Literal[True] = True
    out_of_band_guest_trust_anchor_required: Literal[True] = True
    measured_guest_image_trust_anchor_attested: Literal[False] = False
    linux_kvm_runtime_qualified: Literal[False] = False
    model_weight_contamination_controlled: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    development_only: Literal[True] = True
    official_release_qualified: Literal[False] = False
    official_execution_qualified: Literal[False] = False
    leaderboard_admitted: Literal[False] = False

    @field_validator(
        'bootstrap_valid_from',
        'bootstrap_expires_at',
        'bootstrap_hello_sent_at',
        'bootstrap_ack_received_at',
        'bootstrap_guest_accepted_at',
        'sealed_at',
    )
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_bootstrap_interval(self) -> Self:
        if not (
            self.bootstrap_valid_from
            <= self.bootstrap_hello_sent_at
            <= self.bootstrap_ack_received_at
            <= self.bootstrap_expires_at
            and self.bootstrap_valid_from <= self.bootstrap_guest_accepted_at <= self.bootstrap_expires_at
            and self.sealed_at >= self.bootstrap_ack_received_at
        ):
            raise ValueError('v0.2 outer receipt has inconsistent bootstrap timestamps')
        return self


class AuthenticatedClinicalProductionRunV02(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-clinical-production-run.dev-v0.2'] = (
        CLINICAL_PRODUCTION_RUN_V02_SCHEMA_VERSION
    )
    receipt: ClinicalProductionRunOuterReceiptV02
    base_authenticated_run: AuthenticatedClinicalProductionRun
    receipt_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class LoadedClinicalProductionRunV02(LoadedClinicalProductionRun):
    """A legacy-compatible loaded run plus independently verified bootstrap evidence."""

    authenticated_outer_receipt: AuthenticatedClinicalProductionRunV02
    authenticated_outer_receipt_sha256: str
    clinical_guest_bootstrap: AuthenticatedClinicalGuestBootstrap
    clinical_guest_bootstrap_evidence_sha256: str


def clinical_guest_bootstrap_evidence_sha256(artifact: AuthenticatedClinicalGuestBootstrap) -> str:
    """Hash the exact strict bootstrap artifact retained by the v0.2 package."""

    canonical = AuthenticatedClinicalGuestBootstrap.model_validate_json(canonical_json_bytes(artifact))
    return _sha256(canonical_json_bytes(canonical))


def clinical_production_run_outer_receipt_hmac_v02(
    receipt: ClinicalProductionRunOuterReceiptV02,
    key: bytes,
) -> str:
    _require_key(key, 'clinical production receipt key')
    return hmac.new(key, _HMAC_DOMAIN_V02 + canonical_json_bytes(receipt), hashlib.sha256).hexdigest()


def finalize_clinical_production_run_v02(
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
    clinical_guest_bootstrap: AuthenticatedClinicalGuestBootstrap,
    clinical_guest_bootstrap_receipt_key: bytes,
    expected_clinical_guest_bootstrap_receipt_key_id: str,
    clinical_guest_bootstrap_trust_anchor: ClinicalGuestBootstrapTrustAnchor,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    sealed_at: datetime | None = None,
) -> LoadedClinicalProductionRunV02:
    """Create one v0.2 package, after independently verifying its complete v0.1 base."""

    target = _safe_new_target(output_root)
    work = Path(tempfile.mkdtemp(prefix='.clinical-production-v02-base.', dir=target.parent))
    work.chmod(0o700)
    legacy_root = work / 'base'
    try:
        base = finalize_clinical_production_run(
            output_root=legacy_root,
            run_id=run_id,
            workspace=workspace,
            expected_authenticated_workspace_receipt_sha256=expected_authenticated_workspace_receipt_sha256,
            workspace_receipt_key=workspace_receipt_key,
            expected_workspace_receipt_key_id=expected_workspace_receipt_key_id,
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
            receipt_key=receipt_key,
            expected_receipt_key_id=expected_receipt_key_id,
            sealed_at=sealed_at,
        )
        bootstrap_receipt = _verify_bootstrap(
            clinical_guest_bootstrap,
            receipt_key=clinical_guest_bootstrap_receipt_key,
            expected_receipt_key_id=expected_clinical_guest_bootstrap_receipt_key_id,
            trust_anchor=clinical_guest_bootstrap_trust_anchor,
        )
        _cross_check_bootstrap(base, clinical_guest_bootstrap, bootstrap_receipt)
        authenticated = _make_outer_receipt(
            base=base,
            bootstrap=clinical_guest_bootstrap,
            bootstrap_receipt=bootstrap_receipt,
            receipt_key=receipt_key,
            expected_receipt_key_id=expected_receipt_key_id,
        )
        _materialize_v02(
            target,
            legacy_root=legacy_root,
            authenticated=authenticated,
            bootstrap=clinical_guest_bootstrap,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return load_clinical_production_run_v02(
        target,
        workspace=workspace,
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
        clinical_guest_bootstrap_receipt_key=clinical_guest_bootstrap_receipt_key,
        expected_clinical_guest_bootstrap_receipt_key_id=(expected_clinical_guest_bootstrap_receipt_key_id),
        clinical_guest_bootstrap_trust_anchor=clinical_guest_bootstrap_trust_anchor,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
    )


def load_clinical_production_run_v02(
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
    clinical_guest_bootstrap_receipt_key: bytes,
    expected_clinical_guest_bootstrap_receipt_key_id: str,
    clinical_guest_bootstrap_trust_anchor: ClinicalGuestBootstrapTrustAnchor,
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> LoadedClinicalProductionRunV02:
    """Reload v0.2 bytes and independently verify the legacy run and strict bootstrap."""

    resolved = _safe_existing_root(root)
    receipt_bytes = _read_private_file(resolved / 'clinical-run.json', _MAX_EVIDENCE_BYTES)
    receipt_hmac_bytes = _read_private_file(resolved / 'clinical-run.hmac', 65)
    bootstrap_bytes = _read_private_file(resolved / 'clinical-guest-bootstrap.json', _MAX_EVIDENCE_BYTES)
    component_bytes = {
        name: _read_private_file(
            resolved / name,
            policy.limits.max_final_bytes if name == 'submission.json' else _MAX_EVIDENCE_BYTES,
        )
        for name in _LEGACY_COMPONENT_FILES
    }
    try:
        authenticated = AuthenticatedClinicalProductionRunV02.model_validate_json(receipt_bytes)
        bootstrap = AuthenticatedClinicalGuestBootstrap.model_validate_json(bootstrap_bytes)
    except ValueError as error:
        raise ClinicalProductionRunError('v0.2 clinical production evidence has an invalid strict schema') from error
    if canonical_json_bytes(authenticated) != receipt_bytes or canonical_json_bytes(bootstrap) != bootstrap_bytes:
        raise ClinicalProductionRunError('v0.2 clinical production evidence must use canonical JSON')
    if clinical_production_run_key_id(receipt_key) != expected_receipt_key_id:
        raise ClinicalProductionRunError('v0.2 clinical production receipt key differs from its expected key ID')
    expected_hmac = clinical_production_run_outer_receipt_hmac_v02(authenticated.receipt, receipt_key)
    if not hmac.compare_digest(authenticated.receipt_hmac_sha256, expected_hmac) or not hmac.compare_digest(
        receipt_hmac_bytes,
        (expected_hmac + '\n').encode('ascii'),
    ):
        raise ClinicalProductionRunError('v0.2 clinical production receipt authentication failed')
    if authenticated.receipt.base_authenticated_run_sha256 != _sha256(
        canonical_json_bytes(authenticated.base_authenticated_run)
    ):
        raise ClinicalProductionRunError('v0.2 receipt does not bind the exact base authenticated run')
    bootstrap_sha256 = clinical_guest_bootstrap_evidence_sha256(bootstrap)
    if authenticated.receipt.clinical_guest_bootstrap_evidence_sha256 != bootstrap_sha256:
        raise ClinicalProductionRunError('v0.2 receipt does not bind the exact clinical guest bootstrap')

    bootstrap_receipt = _verify_bootstrap(
        bootstrap,
        receipt_key=clinical_guest_bootstrap_receipt_key,
        expected_receipt_key_id=expected_clinical_guest_bootstrap_receipt_key_id,
        trust_anchor=clinical_guest_bootstrap_trust_anchor,
    )
    base = _reload_base_from_v02(
        authenticated=authenticated,
        component_bytes=component_bytes,
        workspace=workspace,
        expected_authenticated_workspace_receipt_sha256=expected_authenticated_workspace_receipt_sha256,
        workspace_receipt_key=workspace_receipt_key,
        expected_workspace_receipt_key_id=expected_workspace_receipt_key_id,
        expected_run_id=expected_run_id,
        expected_attempt_reservation_sha256=expected_attempt_reservation_sha256,
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
    _cross_check_bootstrap(base, bootstrap, bootstrap_receipt)
    expected_outer = _make_outer_receipt(
        base=base,
        bootstrap=bootstrap,
        bootstrap_receipt=bootstrap_receipt,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
    ).receipt
    if authenticated.receipt != expected_outer:
        raise ClinicalProductionRunError('v0.2 outer receipt differs from the independently verified evidence')
    return LoadedClinicalProductionRunV02(
        root=resolved,
        workspace=base.workspace,
        submission=base.submission,
        worker_attestation=base.worker_attestation,
        gateway_session=base.gateway_session,
        guest_rpc_session=base.guest_rpc_session,
        authenticated_receipt=base.authenticated_receipt,
        authenticated_receipt_sha256=base.authenticated_receipt_sha256,
        authenticated_outer_receipt=authenticated,
        authenticated_outer_receipt_sha256=_sha256(receipt_bytes),
        clinical_guest_bootstrap=bootstrap,
        clinical_guest_bootstrap_evidence_sha256=bootstrap_sha256,
    )


def _verify_bootstrap(
    artifact: AuthenticatedClinicalGuestBootstrap,
    *,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    trust_anchor: ClinicalGuestBootstrapTrustAnchor,
):
    try:
        return verify_authenticated_clinical_guest_bootstrap(
            artifact,
            key=receipt_key,
            expected_key_id=expected_receipt_key_id,
            expected_hello=artifact.signed_hello.hello,
            trust_anchor=trust_anchor,
        )
    except (ClinicalGuestBootstrapError, TypeError, ValueError) as error:
        raise ClinicalProductionRunError('clinical guest bootstrap authentication failed') from error


def _cross_check_bootstrap(base, artifact, bootstrap_receipt) -> None:
    hello = artifact.signed_hello.hello
    worker = base.worker_attestation.attestation
    guest = base.guest_rpc_session
    expected_limits = ClinicalGuestRpcLimits(
        maximum_frame_body_bytes=guest.policy.maximum_frame_body_bytes,
        maximum_session_wire_bytes=guest.policy.maximum_session_wire_bytes,
        maximum_requests=guest.policy.maximum_requests,
        maximum_list_entries=guest.policy.maximum_list_entries,
        maximum_read_bytes=guest.policy.maximum_read_bytes,
        maximum_search_results=guest.policy.maximum_search_results,
        maximum_submission_bytes=guest.policy.maximum_submission_bytes,
    )
    expected = (
        base.receipt.run_id,
        base.receipt.attempt_reservation_sha256,
        guest.seal.session_id,
        base.workspace.invocation,
        agentic_task_invocation_sha256(base.workspace.invocation),
        base.workspace.manifest_sha256,
        base.workspace.manifest.workspace_tree_sha256,
        base.workspace.manifest.model_visible_surface_sha256,
        base.receipt.execution_policy_sha256,
        base.receipt.worker_spec_sha256,
        expected_limits,
    )
    actual = (
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
    if actual != expected:
        raise ClinicalProductionRunError('clinical guest bootstrap differs from the authenticated run boundary')
    signed_sha = clinical_guest_bootstrap_signed_hello_sha256(artifact.signed_hello)
    if (
        bootstrap_receipt.signed_hello_sha256 != signed_sha
        or bootstrap_receipt.hello_sha256 != artifact.signed_hello.hello_sha256
    ):
        raise ClinicalProductionRunError('clinical guest bootstrap receipt differs from its signed hello')
    # ``ack_received_at`` is measured by the host and can therefore order host-side RPC
    # observations. ``guest_accepted_at`` is measured by the guest's independent clock; its only
    # safe time claim is membership in the signed hello validity interval, enforced by
    # ``ClinicalGuestBootstrapReceipt`` above.
    host_bootstrap_precedes_rpc = (
        worker.started_at <= bootstrap_receipt.hello_sent_at <= bootstrap_receipt.ack_received_at <= worker.finished_at
        and all(attempt.started_at >= bootstrap_receipt.ack_received_at for attempt in guest.attempts)
    )
    if not host_bootstrap_precedes_rpc:
        raise ClinicalProductionRunError('clinical guest bootstrap timestamps do not precede guest RPC activity')


def _make_outer_receipt(
    *,
    base: LoadedClinicalProductionRun,
    bootstrap: AuthenticatedClinicalGuestBootstrap,
    bootstrap_receipt,
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> AuthenticatedClinicalProductionRunV02:
    if clinical_production_run_key_id(receipt_key) != expected_receipt_key_id:
        raise ClinicalProductionRunError('v0.2 clinical production receipt key differs from its expected key ID')
    base_bytes = canonical_json_bytes(base.authenticated_receipt)
    bootstrap_bytes = canonical_json_bytes(bootstrap)
    hello = bootstrap.signed_hello.hello
    rpc_limits_sha256 = _sha256(canonical_json_bytes(hello.rpc_limits))
    receipt = ClinicalProductionRunOuterReceiptV02(
        run_id=base.receipt.run_id,
        start_redemption_sha256=base.receipt.attempt_reservation_sha256,
        guest_rpc_session_id=base.guest_rpc_session.seal.session_id,
        task_invocation_sha256=base.receipt.task_invocation_sha256,
        workspace_manifest_sha256=base.receipt.workspace_manifest_sha256,
        workspace_tree_sha256=base.receipt.workspace_tree_sha256,
        model_visible_surface_sha256=base.receipt.model_visible_surface_sha256,
        execution_policy_sha256=base.receipt.execution_policy_sha256,
        worker_spec_sha256=base.receipt.worker_spec_sha256,
        guest_rpc_policy_sha256=base.receipt.guest_rpc_policy_sha256,
        guest_rpc_limits_sha256=rpc_limits_sha256,
        base_authenticated_run_sha256=_sha256(base_bytes),
        clinical_guest_bootstrap_evidence_sha256=_sha256(bootstrap_bytes),
        clinical_guest_bootstrap_receipt_key_id=bootstrap_receipt.receipt_key_id,
        clinical_guest_bootstrap_authorization_key_id=bootstrap_receipt.authorization_key_id,
        clinical_guest_bootstrap_signed_hello_sha256=bootstrap_receipt.signed_hello_sha256,
        clinical_guest_bootstrap_hello_sha256=bootstrap_receipt.hello_sha256,
        bootstrap_valid_from=bootstrap_receipt.valid_from,
        bootstrap_expires_at=bootstrap_receipt.expires_at,
        bootstrap_hello_sent_at=bootstrap_receipt.hello_sent_at,
        bootstrap_ack_received_at=bootstrap_receipt.ack_received_at,
        bootstrap_guest_accepted_at=bootstrap_receipt.guest_accepted_at,
        sealed_at=base.receipt.sealed_at,
        receipt_key_id=expected_receipt_key_id,
    )
    return AuthenticatedClinicalProductionRunV02(
        receipt=receipt,
        base_authenticated_run=base.authenticated_receipt,
        receipt_hmac_sha256=clinical_production_run_outer_receipt_hmac_v02(receipt, receipt_key),
    )


def _materialize_v02(
    target: Path,
    *,
    legacy_root: Path,
    authenticated: AuthenticatedClinicalProductionRunV02,
    bootstrap: AuthenticatedClinicalGuestBootstrap,
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    staging.chmod(0o700)
    try:
        files = {
            'clinical-run.json': canonical_json_bytes(authenticated),
            'clinical-run.hmac': (authenticated.receipt_hmac_sha256 + '\n').encode('ascii'),
            'clinical-guest-bootstrap.json': canonical_json_bytes(bootstrap),
        }
        files.update(
            {name: _read_private_file(legacy_root / name, _MAX_EVIDENCE_BYTES) for name in _LEGACY_COMPONENT_FILES}
        )
        for name, payload in files.items():
            _write_durable_private_file(staging / name, payload)
        _fsync_directory(staging)
        if target.exists() or target.is_symlink():
            raise ClinicalProductionRunError('v0.2 clinical production output appeared during commit')
        os.replace(staging, target)
        _fsync_directory(target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _reload_base_from_v02(
    *,
    authenticated: AuthenticatedClinicalProductionRunV02,
    component_bytes: dict[str, bytes],
    **loader_kwargs,
) -> LoadedClinicalProductionRun:
    temporary = Path(tempfile.mkdtemp(prefix='.clinical-production-v02-verify.'))
    temporary.chmod(0o700)
    try:
        base_bytes = canonical_json_bytes(authenticated.base_authenticated_run)
        files = dict(component_bytes)
        files['clinical-run.json'] = base_bytes
        files['clinical-run.hmac'] = (authenticated.base_authenticated_run.receipt_hmac_sha256 + '\n').encode('ascii')
        for name, payload in files.items():
            path = temporary / name
            path.write_bytes(payload)
            path.chmod(0o600)
        return load_clinical_production_run(temporary, **loader_kwargs)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _safe_new_target(output_root: Path) -> Path:
    supplied = output_root.expanduser()
    if supplied.is_symlink():
        raise ClinicalProductionRunError('v0.2 clinical production output cannot be a symbolic link')
    target = supplied.resolve()
    if target.exists():
        raise ClinicalProductionRunError(f'v0.2 clinical production output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = target.parent.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ClinicalProductionRunError(
            'v0.2 clinical production parent must be a current-user-owned private mode-0700 directory'
        )
    return target


def _safe_existing_root(root: Path) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ClinicalProductionRunError('v0.2 clinical production root cannot be a symbolic link')
    resolved = supplied.resolve()
    metadata = resolved.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ClinicalProductionRunError(
            'v0.2 clinical production root must be a current-user-owned private mode-0700 directory'
        )
    if {entry.name for entry in os.scandir(resolved)} != _FILES_V02:
        raise ClinicalProductionRunError('v0.2 clinical production exact file inventory mismatch')
    return resolved


def _read_private_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ClinicalProductionRunError('cannot open v0.2 clinical production evidence file') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size <= 0
            or metadata.st_size > maximum_bytes
        ):
            raise ClinicalProductionRunError('v0.2 clinical production evidence must be private and bounded')
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes - len(payload) + 1))
            if not chunk:
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise ClinicalProductionRunError('v0.2 clinical production evidence exceeds its byte limit')
    finally:
        os.close(descriptor)


def _write_durable_private_file(path: Path, payload: bytes) -> None:
    if not payload or len(payload) > _MAX_EVIDENCE_BYTES:
        raise ClinicalProductionRunError('v0.2 clinical production evidence has an invalid byte count')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short v0.2 evidence write')
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(payload)
        ):
            raise OSError('v0.2 evidence metadata mismatch')
    except OSError as error:
        raise ClinicalProductionRunError('v0.2 clinical production evidence write failed') from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ClinicalProductionRunError('v0.2 clinical production directory cannot be opened') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ClinicalProductionRunError('v0.2 clinical production directory is not current-user-owned')
        os.fsync(descriptor)
    except OSError as error:
        raise ClinicalProductionRunError('v0.2 clinical production directory fsync failed') from error
    finally:
        os.close(descriptor)


def _require_key(key: bytes, label: str) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ClinicalProductionRunError(f'{label} must contain at least 32 bytes')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    'CLINICAL_PRODUCTION_RUN_OUTER_RECEIPT_V02_SCHEMA_VERSION',
    'CLINICAL_PRODUCTION_RUN_V02_SCHEMA_VERSION',
    'AuthenticatedClinicalProductionRunV02',
    'ClinicalProductionRunOuterReceiptV02',
    'LoadedClinicalProductionRunV02',
    'clinical_guest_bootstrap_evidence_sha256',
    'clinical_production_run_outer_receipt_hmac_v02',
    'finalize_clinical_production_run_v02',
    'load_clinical_production_run_v02',
]
