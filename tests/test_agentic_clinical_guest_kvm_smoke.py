from __future__ import annotations

import hashlib
import hmac
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.test_agentic_clinical_production_run import RUN_ID, WORKER_KEY, _materials
from tests.test_agentic_clinical_production_run_v02 import _bootstrap
from vaxreplay.agentic.clinical_guest_kvm_smoke import (
    ClinicalGuestKvmSmokeEvidence,
    authenticate_clinical_guest_kvm_smoke,
    verify_authenticated_clinical_guest_kvm_smoke,
)
from vaxreplay.agentic.firecracker import (
    FirecrackerCleanupReceipt,
    firecracker_guest_bootstrap_profile_sha256,
    firecracker_model_sha256,
)
from vaxreplay.agentic.guest_rpc import guest_rpc_policy_sha256
from vaxreplay.agentic.protocol import agentic_policy_sha256
from vaxreplay.bundle import canonical_json_bytes


def _evidence(tmp_path: Path) -> ClinicalGuestKvmSmokeEvidence:
    materials = _materials(tmp_path / 'materials')
    bootstrap, _ = _bootstrap(materials)
    guest = materials.guest.model_copy(update={'seal': materials.guest.seal.model_copy(update={'model_call_count': 4})})
    gateway = materials.gateway.model_copy(
        update={'seal': materials.gateway.seal.model_copy(update={'attempt_count': 4, 'successful_call_count': 4})}
    )
    worker = materials.worker.attestation
    cleanup = FirecrackerCleanupReceipt(
        run_id=RUN_ID,
        launched_monotonic_ns=worker.launched_monotonic_ns,
        wall_deadline_monotonic_ns=worker.wall_deadline_monotonic_ns,
        watchdog_triggered_at=None,
        watchdog_triggered_monotonic_ns=None,
        jailer_reaped_at=worker.jailer_reaped_at,
        jailer_reaped_monotonic_ns=worker.jailer_reaped_monotonic_ns,
        cgroup_empty_at=worker.cgroup_empty_at,
        cgroup_empty_monotonic_ns=worker.cgroup_empty_monotonic_ns,
        cleanup_finished_at=worker.cleanup_finished_at,
        cleanup_finished_monotonic_ns=worker.cleanup_finished_monotonic_ns,
        lifecycle='terminated',
        jailer_exit_code=worker.jailer_exit_code,
        wall_watchdog_armed=True,
        wall_timeout_triggered=False,
    )
    worker = worker.model_copy(update={'cleanup_receipt_sha256': firecracker_model_sha256(cleanup)})
    authenticated_worker = materials.worker.model_copy(
        update={
            'attestation': worker,
            'attestation_hmac_sha256': hmac.new(
                WORKER_KEY,
                b'vaxreplay.firecracker-worker-attestation.v0.2\x00' + canonical_json_bytes(worker),
                hashlib.sha256,
            ).hexdigest(),
        }
    )
    return ClinicalGuestKvmSmokeEvidence(
        worker_spec=materials.spec,
        worker_spec_sha256=firecracker_model_sha256(materials.spec),
        worker_bootstrap_profile_sha256=(firecracker_guest_bootstrap_profile_sha256(materials.spec)),
        disk_build_receipt_sha256='1' * 64,
        guest_config_sha256='2' * 64,
        execution_policy=materials.policy,
        execution_policy_sha256=agentic_policy_sha256(materials.policy),
        task_sha256=hashlib.sha256(canonical_json_bytes(materials.workspace.task)).hexdigest(),
        collector_entrypoint_sha256='3' * 64,
        authenticated_bootstrap=bootstrap,
        guest_rpc_session=guest,
        gateway_session=gateway,
        worker_attestation=authenticated_worker,
        cleanup_receipt=cleanup,
        collected_at=worker.cleanup_finished_at + timedelta(seconds=1),
    )


def test_public_signature_and_external_pins_authenticate_task_guest_smoke(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    private_key = Ed25519PrivateKey.from_private_bytes(b'K' * 32)
    authenticated = authenticate_clinical_guest_kvm_smoke(
        evidence,
        private_key=private_key,
    )

    verified = verify_authenticated_clinical_guest_kvm_smoke(
        authenticated,
        expected_collector_public_key_hex=authenticated.collector_public_key_hex,
        expected_worker_spec_sha256=evidence.worker_spec_sha256,
        expected_disk_build_receipt_sha256=evidence.disk_build_receipt_sha256,
        expected_guest_config_sha256=evidence.guest_config_sha256,
        expected_execution_policy_sha256=evidence.execution_policy_sha256,
        expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(evidence.guest_rpc_session.policy),
        expected_task_sha256=evidence.task_sha256,
        expected_collector_entrypoint_sha256=evidence.collector_entrypoint_sha256,
    )

    assert verified == evidence
    assert verified.guest_rpc_session.seal.model_call_count == 4
    assert verified.guest_rpc_session.seal.submit_accepted
    assert not verified.official_leaderboard_execution_qualified


def test_task_guest_smoke_rejects_signature_and_bootstrap_profile_tampering(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    authenticated = authenticate_clinical_guest_kvm_smoke(
        evidence,
        private_key=Ed25519PrivateKey.from_private_bytes(b'K' * 32),
    )
    tampered_signature = authenticated.model_copy(update={'signature_hex': '0' * 128})

    with pytest.raises(ValueError, match='authentication failed'):
        verify_authenticated_clinical_guest_kvm_smoke(
            tampered_signature,
            expected_collector_public_key_hex=authenticated.collector_public_key_hex,
            expected_worker_spec_sha256=evidence.worker_spec_sha256,
            expected_disk_build_receipt_sha256=evidence.disk_build_receipt_sha256,
            expected_guest_config_sha256=evidence.guest_config_sha256,
            expected_execution_policy_sha256=evidence.execution_policy_sha256,
            expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(evidence.guest_rpc_session.policy),
            expected_task_sha256=evidence.task_sha256,
            expected_collector_entrypoint_sha256=evidence.collector_entrypoint_sha256,
        )

    tampered_profile = evidence.model_copy(update={'worker_bootstrap_profile_sha256': 'f' * 64})
    with pytest.raises(ValueError, match='top-level hashes'):
        ClinicalGuestKvmSmokeEvidence.model_validate_json(canonical_json_bytes(tampered_profile))

    tampered_task = evidence.model_copy(update={'task_sha256': 'e' * 64})
    with pytest.raises(ValueError, match='task hash'):
        ClinicalGuestKvmSmokeEvidence.model_validate_json(canonical_json_bytes(tampered_task))
