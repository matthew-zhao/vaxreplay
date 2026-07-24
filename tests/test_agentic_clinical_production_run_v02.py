from __future__ import annotations

import os
import socket
import stat
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import vaxreplay.agentic.clinical_production_run_v02 as production_v02_module
from tests.test_agentic_clinical_production_run import (
    ATTEMPT,
    GATEWAY_KEY,
    GUEST_KEY,
    PRODUCTION_KEY,
    RUN_ID,
    WORKER_KEY,
    WORKSPACE_KEY,
    Materials,
    _materials,
)
from vaxreplay.agentic.clinical_execution_bridge import clinical_workspace_receipt_key_id
from vaxreplay.agentic.clinical_guest_bootstrap import (
    CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
    AuthenticatedClinicalGuestBootstrap,
    ClinicalGuestBootstrapError,
    ClinicalGuestBootstrapFailureCode,
    ClinicalGuestBootstrapHello,
    ClinicalGuestBootstrapTrustAnchor,
    ClinicalGuestRpcLimits,
    InMemoryClinicalGuestBootstrapReplayGuard,
    clinical_guest_bootstrap_authorization_key_id,
    clinical_guest_bootstrap_receipt_key_id,
    perform_guest_clinical_bootstrap,
    perform_host_clinical_guest_bootstrap,
    sign_clinical_guest_bootstrap_hello,
    verify_signed_clinical_guest_bootstrap_hello,
)
from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_ACTION_SCHEMA_SHA256,
    LANE_A_GUEST_HARNESS_POLICY_ID,
)
from vaxreplay.agentic.clinical_production_run import (
    ClinicalProductionRunError,
    clinical_production_run_key_id,
)
from vaxreplay.agentic.clinical_production_run_v02 import (
    AuthenticatedClinicalProductionRunV02,
    LoadedClinicalProductionRunV02,
    clinical_guest_bootstrap_evidence_sha256,
    clinical_production_run_outer_receipt_hmac_v02,
    finalize_clinical_production_run_v02,
    load_clinical_production_run_v02,
)
from vaxreplay.agentic.firecracker import (
    firecracker_attestation_key_id,
    firecracker_guest_bootstrap_profile_sha256,
    firecracker_model_sha256,
)
from vaxreplay.agentic.guest_rpc import guest_rpc_policy_sha256, guest_rpc_session_key_id
from vaxreplay.agentic.protocol import agentic_policy_sha256
from vaxreplay.agentic.provider_gateway import (
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    gateway_session_key_id,
)
from vaxreplay.agentic.task_protocol import agentic_task_invocation_sha256
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.signing import LocalEd25519Signer

BOOTSTRAP_RECEIPT_KEY = b'clinical-bootstrap-receipt-key-0001'
BOOTSTRAP_SIGNER = LocalEd25519Signer(Ed25519PrivateKey.from_private_bytes(b'\x19' * 32))
BOOTSTRAP_AUTHORIZATION_KEY_ID = clinical_guest_bootstrap_authorization_key_id(BOOTSTRAP_SIGNER.public_key_bytes())


def _rpc_limits(materials: Materials) -> ClinicalGuestRpcLimits:
    policy = materials.guest.policy
    return ClinicalGuestRpcLimits(
        maximum_frame_body_bytes=policy.maximum_frame_body_bytes,
        maximum_session_wire_bytes=policy.maximum_session_wire_bytes,
        maximum_requests=policy.maximum_requests,
        maximum_list_entries=policy.maximum_list_entries,
        maximum_read_bytes=policy.maximum_read_bytes,
        maximum_search_results=policy.maximum_search_results,
        maximum_submission_bytes=policy.maximum_submission_bytes,
    )


def _independent_trust_anchor(materials: Materials) -> ClinicalGuestBootstrapTrustAnchor:
    """Build guest image pins from organizer-owned material, never from a received hello."""

    return ClinicalGuestBootstrapTrustAnchor(
        authorization_key_id=BOOTSTRAP_AUTHORIZATION_KEY_ID,
        ed25519_public_key_hex=BOOTSTRAP_SIGNER.public_key_bytes().hex(),
        execution_policy_sha256=agentic_policy_sha256(materials.policy),
        worker_bootstrap_profile_sha256=(firecracker_guest_bootstrap_profile_sha256(materials.spec)),
        harness_policy_id=LANE_A_GUEST_HARNESS_POLICY_ID,
        harness_policy_sha256=CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
        action_schema_sha256=LANE_A_GUEST_ACTION_SCHEMA_SHA256,
        rpc_limits=_rpc_limits(materials),
    )


def _bootstrap(
    materials: Materials,
    *,
    hello_updates: dict[str, Any] | None = None,
    host_observed_at: datetime | None = None,
    guest_accepted_at: datetime | None = None,
) -> tuple[AuthenticatedClinicalGuestBootstrap, ClinicalGuestBootstrapTrustAnchor]:
    default_observed_at = materials.worker.attestation.started_at + timedelta(seconds=1)
    host_observed_at = default_observed_at if host_observed_at is None else host_observed_at
    guest_accepted_at = default_observed_at if guest_accepted_at is None else guest_accepted_at
    invocation = materials.workspace.invocation
    hello = ClinicalGuestBootstrapHello(
        run_id=RUN_ID,
        start_redemption_sha256=ATTEMPT,
        session_id=materials.guest.seal.session_id,
        task_invocation=invocation,
        task_invocation_sha256=agentic_task_invocation_sha256(invocation),
        workspace_manifest_sha256=materials.workspace.manifest_sha256,
        workspace_tree_sha256=materials.workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=materials.workspace.manifest.model_visible_surface_sha256,
        execution_policy_sha256=agentic_policy_sha256(materials.policy),
        worker_bootstrap_profile_sha256=(firecracker_guest_bootstrap_profile_sha256(materials.spec)),
        worker_spec_sha256=firecracker_model_sha256(materials.spec),
        harness_policy_id=LANE_A_GUEST_HARNESS_POLICY_ID,
        harness_policy_sha256=CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
        action_schema_sha256=LANE_A_GUEST_ACTION_SCHEMA_SHA256,
        rpc_limits=_rpc_limits(materials),
        nonce='9' * 64,
        valid_from=materials.worker.attestation.started_at,
        expires_at=materials.worker.attestation.finished_at,
    )
    if hello_updates:
        hello = hello.model_copy(update=hello_updates)
    anchor = _independent_trust_anchor(materials)
    host, guest = socket.socketpair()
    outcome: dict[str, object] = {}

    def run_guest() -> None:
        try:
            outcome['context'] = perform_guest_clinical_bootstrap(
                guest,
                trust_anchor=anchor,
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: guest_accepted_at,
                timeout_seconds=1,
            )
        except BaseException as error:
            outcome['error'] = error

    thread = threading.Thread(target=run_guest)
    thread.start()
    try:
        artifact = perform_host_clinical_guest_bootstrap(
            host,
            hello=hello,
            authorization_signer=BOOTSTRAP_SIGNER,
            expected_authorization_key_id=BOOTSTRAP_AUTHORIZATION_KEY_ID,
            receipt_key=BOOTSTRAP_RECEIPT_KEY,
            clock=lambda: host_observed_at,
            timeout_seconds=1,
        )
        thread.join(timeout=2)
    finally:
        host.close()
        guest.close()
    assert not thread.is_alive()
    if 'error' in outcome:
        error = outcome['error']
        if not isinstance(error, BaseException):
            raise AssertionError('guest bootstrap fixture returned a non-exception error')
        raise AssertionError('guest bootstrap fixture failed') from error
    return artifact, anchor


def test_v02_orders_guest_rpc_after_host_ack_despite_guest_clock_skew(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    first_attempt_at = materials.guest.attempts[0].started_at
    bootstrap, anchor = _bootstrap(
        materials,
        host_observed_at=first_attempt_at - timedelta(milliseconds=100),
        guest_accepted_at=first_attempt_at + timedelta(milliseconds=250),
    )

    loaded = _finalize(tmp_path / 'evidence', materials, bootstrap, anchor)

    receipt = loaded.clinical_guest_bootstrap.receipt
    assert receipt.ack_received_at < first_attempt_at < receipt.guest_accepted_at
    assert _load(loaded.root, materials, anchor) == loaded


def test_v02_does_not_order_guest_clock_against_host_worker_or_seal(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    worker_finished_at = materials.worker.attestation.finished_at
    sealed_at = worker_finished_at + timedelta(seconds=6)
    first_attempt_at = materials.guest.attempts[0].started_at
    bootstrap, anchor = _bootstrap(
        materials,
        hello_updates={'expires_at': worker_finished_at + timedelta(seconds=10)},
        host_observed_at=first_attempt_at - timedelta(milliseconds=100),
        guest_accepted_at=sealed_at + timedelta(seconds=1),
    )

    loaded = _finalize(tmp_path / 'evidence', materials, bootstrap, anchor)

    receipt = loaded.authenticated_outer_receipt.receipt
    assert receipt.bootstrap_guest_accepted_at > worker_finished_at
    assert receipt.bootstrap_guest_accepted_at > receipt.sealed_at
    assert _load(loaded.root, materials, anchor) == loaded


def test_v02_still_bounds_guest_clock_to_signed_validity_interval(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    bootstrap, anchor = _bootstrap(materials)
    loaded = _finalize(tmp_path / 'evidence', materials, bootstrap, anchor)
    receipt = loaded.authenticated_outer_receipt.receipt
    payload = receipt.model_dump(mode='python')
    payload['bootstrap_guest_accepted_at'] = receipt.bootstrap_expires_at + timedelta(microseconds=1)

    with pytest.raises(ValueError, match='inconsistent bootstrap timestamps'):
        production_v02_module.ClinicalProductionRunOuterReceiptV02.model_validate(payload)


def test_v02_rejects_guest_rpc_observed_before_host_ack(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    first_attempt_at = materials.guest.attempts[0].started_at
    bootstrap, anchor = _bootstrap(
        materials,
        host_observed_at=first_attempt_at + timedelta(milliseconds=100),
        guest_accepted_at=first_attempt_at - timedelta(milliseconds=100),
    )

    with pytest.raises(ClinicalProductionRunError, match='timestamps do not precede guest RPC'):
        _finalize(tmp_path / 'evidence', materials, bootstrap, anchor)
    assert not (tmp_path / 'evidence').exists()


def _finalize(
    root: Path,
    materials: Materials,
    bootstrap: AuthenticatedClinicalGuestBootstrap,
    anchor: ClinicalGuestBootstrapTrustAnchor,
) -> LoadedClinicalProductionRunV02:
    workspace = materials.workspace
    return finalize_clinical_production_run_v02(
        output_root=root,
        run_id=RUN_ID,
        workspace=workspace,
        expected_authenticated_workspace_receipt_sha256=workspace.authenticated_receipt_sha256,
        workspace_receipt_key=WORKSPACE_KEY,
        expected_workspace_receipt_key_id=clinical_workspace_receipt_key_id(WORKSPACE_KEY),
        attempt_reservation_sha256=ATTEMPT,
        policy=materials.policy,
        harness=materials.harness,
        worker_spec=materials.spec,
        worker_attestation=materials.worker,
        worker_attestation_key=WORKER_KEY,
        expected_worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
        gateway_session=materials.gateway,
        gateway_receipt_key=GATEWAY_KEY,
        expected_gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
        expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(materials.gateway.policy),
        expected_gateway_route_sha256=gateway_model_route_sha256(materials.gateway.route),
        guest_rpc_session=materials.guest,
        guest_rpc_receipt_key=GUEST_KEY,
        expected_guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_KEY),
        expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(materials.guest.policy),
        submission=materials.submission,
        clinical_guest_bootstrap=bootstrap,
        clinical_guest_bootstrap_receipt_key=BOOTSTRAP_RECEIPT_KEY,
        expected_clinical_guest_bootstrap_receipt_key_id=(
            clinical_guest_bootstrap_receipt_key_id(BOOTSTRAP_RECEIPT_KEY)
        ),
        clinical_guest_bootstrap_trust_anchor=anchor,
        receipt_key=PRODUCTION_KEY,
        expected_receipt_key_id=clinical_production_run_key_id(PRODUCTION_KEY),
        sealed_at=materials.worker.attestation.finished_at + timedelta(seconds=6),
    )


def _load(
    root: Path,
    materials: Materials,
    anchor: ClinicalGuestBootstrapTrustAnchor,
    **updates: Any,
) -> LoadedClinicalProductionRunV02:
    workspace = materials.workspace
    arguments: dict[str, Any] = {
        'workspace': workspace,
        'expected_authenticated_workspace_receipt_sha256': workspace.authenticated_receipt_sha256,
        'workspace_receipt_key': WORKSPACE_KEY,
        'expected_workspace_receipt_key_id': clinical_workspace_receipt_key_id(WORKSPACE_KEY),
        'expected_run_id': RUN_ID,
        'expected_attempt_reservation_sha256': ATTEMPT,
        'policy': materials.policy,
        'harness': materials.harness,
        'worker_spec': materials.spec,
        'worker_attestation_key': WORKER_KEY,
        'expected_worker_attestation_key_id': firecracker_attestation_key_id(WORKER_KEY),
        'gateway_receipt_key': GATEWAY_KEY,
        'expected_gateway_receipt_key_id': gateway_session_key_id(GATEWAY_KEY),
        'expected_gateway_policy_sha256': authenticated_gateway_policy_sha256(materials.gateway.policy),
        'expected_gateway_route_sha256': gateway_model_route_sha256(materials.gateway.route),
        'guest_rpc_receipt_key': GUEST_KEY,
        'expected_guest_rpc_receipt_key_id': guest_rpc_session_key_id(GUEST_KEY),
        'expected_guest_rpc_policy_sha256': guest_rpc_policy_sha256(materials.guest.policy),
        'clinical_guest_bootstrap_receipt_key': BOOTSTRAP_RECEIPT_KEY,
        'expected_clinical_guest_bootstrap_receipt_key_id': (
            clinical_guest_bootstrap_receipt_key_id(BOOTSTRAP_RECEIPT_KEY)
        ),
        'clinical_guest_bootstrap_trust_anchor': anchor,
        'receipt_key': PRODUCTION_KEY,
        'expected_receipt_key_id': clinical_production_run_key_id(PRODUCTION_KEY),
    }
    arguments.update(updates)
    return load_clinical_production_run_v02(root, **arguments)


def test_v02_retains_and_authenticates_exact_signed_bootstrap(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    bootstrap, anchor = _bootstrap(materials)
    loaded = _finalize(tmp_path / 'evidence', materials, bootstrap, anchor)

    assert loaded.clinical_guest_bootstrap == bootstrap
    assert loaded.clinical_guest_bootstrap_evidence_sha256 == clinical_guest_bootstrap_evidence_sha256(bootstrap)
    assert loaded.authenticated_outer_receipt.receipt.clinical_guest_bootstrap_authorization_key_id == (
        BOOTSTRAP_AUTHORIZATION_KEY_ID
    )
    assert loaded.authenticated_outer_receipt.receipt.clinical_guest_bootstrap_receipt_key_id == (
        clinical_guest_bootstrap_receipt_key_id(BOOTSTRAP_RECEIPT_KEY)
    )
    assert loaded.authenticated_outer_receipt.receipt.bootstrap_precedes_first_guest_rpc_attempt
    assert not loaded.authenticated_outer_receipt.receipt.linux_kvm_runtime_qualified
    assert (
        loaded.authenticated_receipt_sha256 == loaded.authenticated_outer_receipt.receipt.base_authenticated_run_sha256
    )
    assert loaded.authenticated_outer_receipt_sha256 != loaded.authenticated_receipt_sha256
    assert {item.name for item in loaded.root.iterdir()} == {
        'clinical-guest-bootstrap.json',
        'clinical-run.hmac',
        'clinical-run.json',
        'gateway-session.json',
        'guest-rpc-session.json',
        'submission.json',
        'worker-attestation.json',
        'workspace-receipt.json',
    }
    assert _load(loaded.root, materials, anchor).clinical_guest_bootstrap == bootstrap


def test_v02_commit_fsyncs_files_final_directory_and_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materials = _materials(tmp_path / 'materials')
    bootstrap, anchor = _bootstrap(materials)
    directory_syncs: list[Path] = []
    regular_file_syncs = 0
    original_directory_sync = production_v02_module._fsync_directory
    original_fsync = production_v02_module.os.fsync

    def track_directory(path: Path) -> None:
        directory_syncs.append(path)
        original_directory_sync(path)

    def track_fsync(descriptor: int) -> None:
        nonlocal regular_file_syncs
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            regular_file_syncs += 1
        original_fsync(descriptor)

    monkeypatch.setattr(production_v02_module, '_fsync_directory', track_directory)
    monkeypatch.setattr(production_v02_module.os, 'fsync', track_fsync)
    loaded = _finalize(tmp_path / 'evidence', materials, bootstrap, anchor)

    assert regular_file_syncs >= 8
    assert directory_syncs[-2:] == [loaded.root, loaded.root.parent]
    assert directory_syncs[0].parent == loaded.root.parent
    assert directory_syncs[0].name.startswith(f'.{loaded.root.name}.')


def test_v02_guest_trust_anchor_rejects_bootstrap_profile_substitution(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    bootstrap, _ = _bootstrap(materials)
    wrong_hello = bootstrap.signed_hello.hello.model_copy(update={'worker_bootstrap_profile_sha256': 'f' * 64})
    signed = sign_clinical_guest_bootstrap_hello(wrong_hello, signer=BOOTSTRAP_SIGNER)

    with pytest.raises(ClinicalGuestBootstrapError) as raised:
        verify_signed_clinical_guest_bootstrap_hello(
            signed,
            trust_anchor=_independent_trust_anchor(materials),
        )
    assert raised.value.code is ClinicalGuestBootstrapFailureCode.TRUST_ANCHOR_MISMATCH


@pytest.mark.parametrize(
    ('field_name', 'value'),
    [
        ('session_id', 'd' * 32),
        ('start_redemption_sha256', 'e' * 64),
        ('workspace_tree_sha256', 'f' * 64),
    ],
)
def test_v02_finalizer_rejects_valid_signed_bootstrap_for_another_boundary(
    tmp_path: Path,
    field_name: str,
    value: str,
) -> None:
    materials = _materials(tmp_path / 'materials')
    wrong, anchor = _bootstrap(materials, hello_updates={field_name: value})

    with pytest.raises(ClinicalProductionRunError, match='authenticated run boundary'):
        _finalize(tmp_path / 'evidence', materials, wrong, anchor)
    assert not (tmp_path / 'evidence').exists()


def test_v02_loader_rejects_rehmaced_valid_bootstrap_substitution(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    bootstrap, anchor = _bootstrap(materials)
    loaded = _finalize(tmp_path / 'evidence', materials, bootstrap, anchor)
    wrong, _ = _bootstrap(materials, hello_updates={'session_id': 'd' * 32})

    outer_path = loaded.root / 'clinical-run.json'
    outer = AuthenticatedClinicalProductionRunV02.model_validate_json(outer_path.read_bytes())
    wrong_receipt = wrong.receipt
    replaced_receipt = outer.receipt.model_copy(
        update={
            'guest_rpc_session_id': wrong.signed_hello.hello.session_id,
            'clinical_guest_bootstrap_evidence_sha256': clinical_guest_bootstrap_evidence_sha256(wrong),
            'clinical_guest_bootstrap_signed_hello_sha256': wrong_receipt.signed_hello_sha256,
            'clinical_guest_bootstrap_hello_sha256': wrong_receipt.hello_sha256,
            'bootstrap_valid_from': wrong_receipt.valid_from,
            'bootstrap_expires_at': wrong_receipt.expires_at,
            'bootstrap_hello_sent_at': wrong_receipt.hello_sent_at,
            'bootstrap_ack_received_at': wrong_receipt.ack_received_at,
            'bootstrap_guest_accepted_at': wrong_receipt.guest_accepted_at,
        }
    )
    replaced = outer.model_copy(
        update={
            'receipt': replaced_receipt,
            'receipt_hmac_sha256': clinical_production_run_outer_receipt_hmac_v02(
                replaced_receipt,
                PRODUCTION_KEY,
            ),
        }
    )
    (loaded.root / 'clinical-guest-bootstrap.json').write_bytes(canonical_json_bytes(wrong))
    (loaded.root / 'clinical-guest-bootstrap.json').chmod(0o600)
    outer_path.write_bytes(canonical_json_bytes(replaced))
    outer_path.chmod(0o600)
    (loaded.root / 'clinical-run.hmac').write_bytes((replaced.receipt_hmac_sha256 + '\n').encode('ascii'))
    (loaded.root / 'clinical-run.hmac').chmod(0o600)

    with pytest.raises(ClinicalProductionRunError, match='authenticated run boundary'):
        _load(loaded.root, materials, anchor)


def test_v02_loader_requires_the_out_of_band_signing_anchor(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    bootstrap, anchor = _bootstrap(materials)
    loaded = _finalize(tmp_path / 'evidence', materials, bootstrap, anchor)
    other_signer = LocalEd25519Signer(Ed25519PrivateKey.from_private_bytes(b'\x20' * 32))
    wrong_anchor = anchor.model_copy(
        update={
            'authorization_key_id': clinical_guest_bootstrap_authorization_key_id(other_signer.public_key_bytes()),
            'ed25519_public_key_hex': other_signer.public_key_bytes().hex(),
        }
    )

    with pytest.raises(ClinicalProductionRunError, match='bootstrap authentication failed'):
        _load(loaded.root, materials, wrong_anchor)
