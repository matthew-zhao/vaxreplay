"""Publicly verifiable evidence for a real Lane A task-guest KVM smoke.

The seven isolation drills use a purpose-built probe guest.  This narrower artifact proves a
different fact: the exact clinical rootfs and harness disks booted, accepted the launcher-signed
bootstrap, completed the ordinary authenticated guest-RPC/model/tool loop, submitted once, and
were fully cleaned up.  It remains development evidence, not leaderboard qualification.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import Literal, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_guest_bootstrap import (
    AuthenticatedClinicalGuestBootstrap,
    clinical_guest_bootstrap_hello_sha256,
    clinical_guest_bootstrap_signed_hello_sha256,
)
from vaxreplay.agentic.firecracker import (
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerCleanupReceipt,
    FirecrackerWorkerSpec,
    firecracker_guest_bootstrap_profile_sha256,
    firecracker_model_sha256,
)
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcTerminalStatus,
    guest_rpc_policy_sha256,
)
from vaxreplay.agentic.protocol import AgenticExecutionPolicy, agentic_policy_sha256
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewaySession,
    GatewayTerminalReason,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

CLINICAL_GUEST_KVM_SMOKE_SCHEMA_VERSION = 'vaxreplay.clinical-guest-kvm-smoke.dev-v0.1'
AUTHENTICATED_CLINICAL_GUEST_KVM_SMOKE_SCHEMA_VERSION = 'vaxreplay.authenticated-clinical-guest-kvm-smoke.dev-v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SIGNATURE_DOMAIN = b'vaxreplay.clinical-guest-kvm-smoke.dev-v0.1\x00'
_COLLECTOR_KEY_ID_DOMAIN = b'vaxreplay.clinical-guest-kvm-smoke-collector-key-id.dev-v0.1\x00'


class ClinicalGuestKvmSmokeEvidence(StrictModel):
    """One exact task-guest lifecycle, with all authenticated sub-artifacts retained."""

    schema_version: Literal['vaxreplay.clinical-guest-kvm-smoke.dev-v0.1'] = CLINICAL_GUEST_KVM_SMOKE_SCHEMA_VERSION
    worker_spec: FirecrackerWorkerSpec
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_bootstrap_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    disk_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy: AgenticExecutionPolicy
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_entrypoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    authenticated_bootstrap: AuthenticatedClinicalGuestBootstrap
    guest_rpc_session: AuthenticatedGuestRpcSession
    gateway_session: AuthenticatedGatewaySession
    worker_attestation: AuthenticatedFirecrackerWorkerAttestation
    cleanup_receipt: FirecrackerCleanupReceipt
    collected_at: datetime
    host_os: Literal['Linux'] = 'Linux'
    kvm_character_device_used: Literal[True] = True
    firecracker_task_guest_booted: Literal[True] = True
    guest_poweroff_observed: Literal[True] = True
    launcher_signed_bootstrap_accepted: Literal[True] = True
    bootstrap_receipt_hmac_verified_by_collector: Literal[True] = True
    authenticated_guest_rpc_completed: Literal[True] = True
    terminal_submission_accepted: Literal[True] = True
    provider_adapter_kind: Literal['public_deterministic_scripted_smoke'] = 'public_deterministic_scripted_smoke'
    external_provider_or_paid_model_called: Literal[False] = False
    learned_model_weights_used: Literal[False] = False
    authoritative_one_attempt_registry_exercised: Literal[False] = False
    collector_runtime_closure_attested: Literal[False] = False
    exact_cgroup_and_jail_cleanup_attested: Literal[True] = True
    development_only: Literal[True] = True
    official_leaderboard_execution_qualified: Literal[False] = False

    @field_validator('collected_at')
    @classmethod
    def validate_collected_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('KVM smoke collection time must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_cross_bindings(self) -> Self:
        spec_sha256 = firecracker_model_sha256(self.worker_spec)
        profile_sha256 = firecracker_guest_bootstrap_profile_sha256(self.worker_spec)
        policy_sha256 = agentic_policy_sha256(self.execution_policy)
        hello = self.authenticated_bootstrap.signed_hello.hello
        signed_hello = self.authenticated_bootstrap.signed_hello
        bootstrap_receipt = self.authenticated_bootstrap.receipt
        guest = self.guest_rpc_session.seal
        gateway = self.gateway_session.seal
        worker = self.worker_attestation.attestation
        cleanup = self.cleanup_receipt
        if (
            self.worker_spec_sha256,
            self.worker_bootstrap_profile_sha256,
            self.execution_policy_sha256,
        ) != (spec_sha256, profile_sha256, policy_sha256):
            raise ValueError('KVM smoke top-level hashes do not bind their exact models')
        if (
            hello.worker_spec_sha256,
            hello.worker_bootstrap_profile_sha256,
            hello.execution_policy_sha256,
            hello.start_redemption_sha256,
            hello.run_id,
            hello.session_id,
            hello.workspace_manifest_sha256,
        ) != (
            spec_sha256,
            profile_sha256,
            policy_sha256,
            guest.attempt_reservation_sha256,
            guest.run_id,
            guest.session_id,
            guest.workspace_manifest_sha256,
        ):
            raise ValueError('signed guest bootstrap differs from the completed RPC session')
        if self.task_sha256 != hashlib.sha256(canonical_json_bytes(hello.task_invocation.task)).hexdigest():
            raise ValueError('KVM smoke task hash does not bind the bootstrapped task')
        if (
            bootstrap_receipt.authorization_key_id,
            bootstrap_receipt.run_id,
            bootstrap_receipt.start_redemption_sha256,
            bootstrap_receipt.session_id,
            bootstrap_receipt.task_invocation_sha256,
            bootstrap_receipt.workspace_manifest_sha256,
            bootstrap_receipt.workspace_tree_sha256,
            bootstrap_receipt.model_visible_surface_sha256,
            bootstrap_receipt.execution_policy_sha256,
            bootstrap_receipt.worker_bootstrap_profile_sha256,
            bootstrap_receipt.worker_spec_sha256,
            bootstrap_receipt.harness_policy_sha256,
            bootstrap_receipt.action_schema_sha256,
            bootstrap_receipt.rpc_limits_sha256,
            bootstrap_receipt.hello_sha256,
            bootstrap_receipt.signed_hello_sha256,
        ) != (
            signed_hello.authorization_key_id,
            hello.run_id,
            guest.attempt_reservation_sha256,
            hello.session_id,
            hello.task_invocation_sha256,
            hello.workspace_manifest_sha256,
            hello.workspace_tree_sha256,
            hello.model_visible_surface_sha256,
            hello.execution_policy_sha256,
            hello.worker_bootstrap_profile_sha256,
            hello.worker_spec_sha256,
            hello.harness_policy_sha256,
            hello.action_schema_sha256,
            hashlib.sha256(canonical_json_bytes(hello.rpc_limits)).hexdigest(),
            clinical_guest_bootstrap_hello_sha256(hello),
            clinical_guest_bootstrap_signed_hello_sha256(signed_hello),
        ):
            raise ValueError('authenticated bootstrap receipt differs from the RPC attempt')
        if (
            guest.worker_spec_sha256,
            guest.execution_policy_sha256,
            guest.rpc_policy_sha256,
            guest.gateway_capability_id,
            guest.terminal_status,
            guest.submit_attempted,
            guest.submit_accepted,
        ) != (
            spec_sha256,
            policy_sha256,
            guest_rpc_policy_sha256(self.guest_rpc_session.policy),
            gateway.capability_id,
            GuestRpcTerminalStatus.COMPLETED,
            True,
            True,
        ):
            raise ValueError('KVM smoke does not contain one completed authenticated RPC session')
        if (
            gateway.run_id,
            gateway.execution_policy_sha256,
            gateway.workspace_manifest_sha256,
            gateway.terminal_reason,
            gateway.terminal_error_code,
            gateway.successful_call_count,
        ) != (
            guest.run_id,
            policy_sha256,
            guest.workspace_manifest_sha256,
            GatewayTerminalReason.COMPLETED,
            None,
            guest.model_call_count,
        ):
            raise ValueError('KVM smoke gateway session differs from the guest model calls')
        if guest.model_call_count < 4 or guest.final_submission_bytes <= 0:
            raise ValueError('KVM smoke did not exercise the multi-step native harness')
        if (
            worker.run_id,
            worker.worker_spec_sha256,
            worker.attempt_reservation_sha256,
            worker.rootfs_sha256,
            worker.harness_sha256,
            worker.cleanup_receipt_sha256,
            worker.jailer_exit_code,
            worker.wall_timeout_triggered,
            cleanup.run_id,
            cleanup.lifecycle,
            cleanup.wall_timeout_triggered,
        ) != (
            guest.run_id,
            spec_sha256,
            guest.attempt_reservation_sha256,
            self.worker_spec.images.rootfs.sha256,
            self.worker_spec.images.harness.sha256,
            firecracker_model_sha256(cleanup),
            0,
            False,
            guest.run_id,
            'terminated',
            False,
        ):
            raise ValueError('KVM smoke worker lifecycle differs from the task disks or RPC run')
        if self.guest_rpc_session.submission is None:
            raise ValueError('KVM smoke is missing its accepted terminal submission')
        return self


class AuthenticatedClinicalGuestKvmSmoke(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-clinical-guest-kvm-smoke.dev-v0.1'] = (
        AUTHENTICATED_CLINICAL_GUEST_KVM_SMOKE_SCHEMA_VERSION
    )
    evidence: ClinicalGuestKvmSmokeEvidence
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_public_key_hex: str = Field(pattern=r'^[0-9a-f]{64}$')
    collector_key_id: str = Field(pattern=_SHA256_PATTERN)
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    signature_hex: str = Field(pattern=r'^[0-9a-f]{128}$')

    @model_validator(mode='after')
    def validate_hash_and_key_id(self) -> Self:
        if self.evidence_sha256 != clinical_guest_kvm_smoke_sha256(self.evidence):
            raise ValueError('authenticated KVM smoke hash does not bind its evidence')
        public_key = bytes.fromhex(self.collector_public_key_hex)
        if self.collector_key_id != clinical_guest_kvm_smoke_collector_key_id(public_key):
            raise ValueError('authenticated KVM smoke key ID does not bind its public key')
        return self


def clinical_guest_kvm_smoke_sha256(
    evidence: ClinicalGuestKvmSmokeEvidence,
) -> str:
    canonical = ClinicalGuestKvmSmokeEvidence.model_validate_json(canonical_json_bytes(evidence))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def clinical_guest_kvm_smoke_collector_key_id(public_key: bytes) -> str:
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError('KVM smoke collector public key must contain exactly 32 bytes')
    return hashlib.sha256(_COLLECTOR_KEY_ID_DOMAIN + public_key).hexdigest()


def authenticate_clinical_guest_kvm_smoke(
    evidence: ClinicalGuestKvmSmokeEvidence,
    *,
    private_key: Ed25519PrivateKey,
) -> AuthenticatedClinicalGuestKvmSmoke:
    canonical = ClinicalGuestKvmSmokeEvidence.model_validate_json(canonical_json_bytes(evidence))
    public_key = private_key.public_key().public_bytes_raw()
    evidence_sha256 = clinical_guest_kvm_smoke_sha256(canonical)
    signature = private_key.sign(_SIGNATURE_DOMAIN + canonical_json_bytes(canonical))
    return AuthenticatedClinicalGuestKvmSmoke(
        evidence=canonical,
        evidence_sha256=evidence_sha256,
        collector_public_key_hex=public_key.hex(),
        collector_key_id=clinical_guest_kvm_smoke_collector_key_id(public_key),
        signature_hex=signature.hex(),
    )


def verify_authenticated_clinical_guest_kvm_smoke(
    authenticated: AuthenticatedClinicalGuestKvmSmoke,
    *,
    expected_collector_public_key_hex: str,
    expected_worker_spec_sha256: str,
    expected_disk_build_receipt_sha256: str,
    expected_guest_config_sha256: str,
    expected_execution_policy_sha256: str,
    expected_guest_rpc_policy_sha256: str,
    expected_task_sha256: str,
    expected_collector_entrypoint_sha256: str,
) -> ClinicalGuestKvmSmokeEvidence:
    try:
        canonical = AuthenticatedClinicalGuestKvmSmoke.model_validate_json(canonical_json_bytes(authenticated))
        public_key = bytes.fromhex(expected_collector_public_key_hex)
        if len(public_key) != 32:
            raise ValueError('invalid expected public key')
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(canonical.signature_hex),
            _SIGNATURE_DOMAIN + canonical_json_bytes(canonical.evidence),
        )
    except (InvalidSignature, TypeError, ValueError):
        raise ValueError('clinical guest KVM smoke authentication failed') from None
    evidence = canonical.evidence
    expected = (
        expected_collector_public_key_hex,
        expected_worker_spec_sha256,
        expected_disk_build_receipt_sha256,
        expected_guest_config_sha256,
        expected_execution_policy_sha256,
        expected_guest_rpc_policy_sha256,
        expected_task_sha256,
        expected_collector_entrypoint_sha256,
    )
    observed = (
        canonical.collector_public_key_hex,
        evidence.worker_spec_sha256,
        evidence.disk_build_receipt_sha256,
        evidence.guest_config_sha256,
        evidence.execution_policy_sha256,
        guest_rpc_policy_sha256(evidence.guest_rpc_session.policy),
        evidence.task_sha256,
        evidence.collector_entrypoint_sha256,
    )
    if not all(hmac.compare_digest(left, right) for left, right in zip(expected, observed, strict=True)):
        raise ValueError('clinical guest KVM smoke differs from its external pins')
    return evidence


__all__ = [
    'AUTHENTICATED_CLINICAL_GUEST_KVM_SMOKE_SCHEMA_VERSION',
    'CLINICAL_GUEST_KVM_SMOKE_SCHEMA_VERSION',
    'AuthenticatedClinicalGuestKvmSmoke',
    'ClinicalGuestKvmSmokeEvidence',
    'authenticate_clinical_guest_kvm_smoke',
    'clinical_guest_kvm_smoke_collector_key_id',
    'clinical_guest_kvm_smoke_sha256',
    'verify_authenticated_clinical_guest_kvm_smoke',
]
