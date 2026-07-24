from __future__ import annotations

import hashlib
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.test_agentic_clinical_execution_bridge import (
    _cohort_manifest,
    _terminal_session,
    _workspace,
    _workspace_keys,
)
from tests.test_agentic_production_run import _materials
from vaxreplay.agentic.clinical_guest_bootstrap import (
    CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
    AuthenticatedClinicalGuestBootstrap,
    ClinicalGuestBootstrapHello,
    ClinicalGuestBootstrapReceipt,
    ClinicalGuestRpcLimits,
    SignedClinicalGuestBootstrapHello,
    clinical_guest_bootstrap_hello_sha256,
    clinical_guest_bootstrap_signed_hello_sha256,
)
from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_ACTION_SCHEMA_SHA256,
    LANE_A_GUEST_HARNESS_POLICY_ID,
)
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionCohortStatus,
    ClinicalProductionRegistryError,
    ClinicalProductionSystemIdentity,
    ClinicalProductionTerminalCode,
    SqliteClinicalProductionRegistry,
    clinical_production_start_redemption_sha256,
    clinical_production_system_core_sha256,
    clinical_production_system_identity_sha256,
    clinical_production_task_launch_sha256,
    require_official_model_snapshot_attestation,
)
from vaxreplay.agentic.clinical_production_run import (
    AuthenticatedClinicalProductionRun,
    ClinicalProductionRunReceipt,
    LoadedClinicalProductionRun,
)
from vaxreplay.agentic.clinical_production_run_v02 import (
    AuthenticatedClinicalProductionRunV02,
    ClinicalProductionRunOuterReceiptV02,
    LoadedClinicalProductionRunV02,
)
from vaxreplay.agentic.guest_rpc import guest_rpc_policy_sha256
from vaxreplay.agentic.protocol import AgenticModelUsage, agentic_receipt_key_id
from vaxreplay.agentic.provider_gateway import (
    authenticated_gateway_policy_sha256,
    gateway_capability_grant_sha256,
    gateway_model_route_sha256,
)
from vaxreplay.agentic.task_protocol import agentic_task_invocation_sha256
from vaxreplay.bundle import canonical_json_bytes

_LAUNCHER_ID = 'lane-a-canonical-launcher'
_LAUNCHER_SHA256 = '6' * 64
_PROVIDER_SUBPROCESS_SPEC_SHA256 = '9' * 64
_PROVIDER_SUBPROCESS_BEHAVIOR_SHA256 = 'b' * 64
_PROVIDER_SUBPROCESS_MODULE_SOURCE_SHA256 = 'a' * 64


def _system(tmp_path: Path, workspace):
    materials_root = tmp_path / 'production-materials'
    materials_root.mkdir(mode=0o700)
    _, policy, spec, worker, gateway, _, harness = _materials(materials_root)
    clinical_session = _terminal_session(tmp_path / 'clinical-session', workspace)
    rpc_policy = clinical_session.policy.model_copy(
        update={
            'maximum_list_entries': 100,
            'maximum_read_bytes': 32_768,
            'maximum_search_results': 20,
            'maximum_submission_bytes': 65_536,
        }
    )
    rpc_policy = type(rpc_policy).model_validate_json(canonical_json_bytes(rpc_policy))
    clinical_session = clinical_session.model_copy(
        update={
            'policy': rpc_policy,
            'seal': clinical_session.seal.model_copy(update={'rpc_policy_sha256': guest_rpc_policy_sha256(rpc_policy)}),
        }
    )
    identity = ClinicalProductionSystemIdentity(
        harness=harness,
        execution_policy_sha256=clinical_session.seal.execution_policy_sha256,
        worker_spec_sha256=clinical_session.seal.worker_spec_sha256,
        gateway_policy_sha256=authenticated_gateway_policy_sha256(gateway.policy),
        gateway_route=gateway.route,
        gateway_route_sha256=gateway_model_route_sha256(gateway.route),
        provider_subprocess_spec_sha256=_PROVIDER_SUBPROCESS_SPEC_SHA256,
        provider_subprocess_behavior_sha256=_PROVIDER_SUBPROCESS_BEHAVIOR_SHA256,
        provider_subprocess_module_source_sha256=(_PROVIDER_SUBPROCESS_MODULE_SOURCE_SHA256),
        guest_rpc_policy_sha256=guest_rpc_policy_sha256(clinical_session.policy),
        guest_bootstrap_authorization_key_id='7' * 64,
        guest_bootstrap_receipt_key_id='8' * 64,
        worker_attestation_key_id=worker.attestation_key_id,
        gateway_receipt_key_id=gateway.policy.receipt_key_id,
        guest_rpc_receipt_key_id=clinical_session.seal.receipt_key_id,
        production_receipt_key_id=agentic_receipt_key_id(b'production-registry-test-receipt-key'),
        canonical_launcher_id=_LAUNCHER_ID,
        canonical_launcher_executable_sha256=_LAUNCHER_SHA256,
    )
    return identity, worker, gateway, clinical_session, policy, spec


def _loaded_evidence(
    tmp_path: Path,
    *,
    system: ClinicalProductionSystemIdentity,
    workspace,
    launch,
    worker,
    gateway,
    clinical_session,
    start_redemption,
) -> LoadedClinicalProductionRunV02:
    attempt_sha256 = clinical_production_start_redemption_sha256(start_redemption)
    grant = gateway.grant.model_copy(
        update={
            'run_id': launch.run_id,
            'attempt_reservation_sha256': attempt_sha256,
            'execution_policy_sha256': system.execution_policy_sha256,
            'workspace_manifest_sha256': workspace.manifest_sha256,
            'gateway_policy_sha256': system.gateway_policy_sha256,
            'model_route_sha256': system.gateway_route_sha256,
        }
    )
    gateway_seal = gateway.seal.model_copy(
        update={
            'run_id': launch.run_id,
            'attempt_reservation_sha256': attempt_sha256,
            'execution_policy_sha256': system.execution_policy_sha256,
            'workspace_manifest_sha256': workspace.manifest_sha256,
            'gateway_policy_sha256': system.gateway_policy_sha256,
            'model_route_sha256': system.gateway_route_sha256,
            'grant_sha256': gateway_capability_grant_sha256(grant),
        }
    )
    gateway = gateway.model_copy(update={'grant': grant, 'seal': gateway_seal})
    guest_seal = clinical_session.seal.model_copy(
        update={
            'run_id': launch.run_id,
            'attempt_reservation_sha256': attempt_sha256,
            'execution_policy_sha256': system.execution_policy_sha256,
            'worker_spec_sha256': system.worker_spec_sha256,
            'rpc_policy_sha256': system.guest_rpc_policy_sha256,
            'workspace_manifest_sha256': workspace.manifest_sha256,
            'gateway_capability_id': grant.capability_id,
            'gateway_grant_sha256': gateway_capability_grant_sha256(grant),
        }
    )
    clinical_session = clinical_session.model_copy(update={'gateway_grant': grant, 'seal': guest_seal})
    worker_attestation = worker.attestation.model_copy(
        update={
            'run_id': launch.run_id,
            'attempt_reservation_sha256': attempt_sha256,
            'worker_spec_sha256': system.worker_spec_sha256,
            'started_at': start_redemption.redeemed_at,
            'finished_at': start_redemption.redeemed_at,
            'duration_ms': 0,
        }
    )
    worker = worker.model_copy(update={'attestation': worker_attestation})
    submission = clinical_session.submission
    assert submission is not None
    submission_bytes = canonical_json_bytes(submission)
    receipt = ClinicalProductionRunReceipt(
        run_id=launch.run_id,
        attempt_reservation_sha256=attempt_sha256,
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        authenticated_workspace_receipt_sha256=workspace.authenticated_receipt_sha256,
        workspace_receipt_key_id=workspace.authenticated_receipt.receipt.receipt_key_id,
        task_sha256=__import__('hashlib').sha256(canonical_json_bytes(workspace.task)).hexdigest(),
        task_context_sha256=workspace.task.context_sha256,
        task_invocation_sha256=agentic_task_invocation_sha256(workspace.invocation),
        execution_policy_sha256=system.execution_policy_sha256,
        harness=system.harness,
        resolved_model_id=system.gateway_route.resolved_model_id,
        worker_spec_sha256=system.worker_spec_sha256,
        worker_attestation_sha256=hashlib.sha256(canonical_json_bytes(worker)).hexdigest(),
        gateway_policy_sha256=system.gateway_policy_sha256,
        gateway_route_sha256=system.gateway_route_sha256,
        gateway_session_sha256=hashlib.sha256(canonical_json_bytes(gateway)).hexdigest(),
        guest_rpc_policy_sha256=system.guest_rpc_policy_sha256,
        guest_rpc_session_sha256=hashlib.sha256(canonical_json_bytes(clinical_session)).hexdigest(),
        worker_attestation_key_id=system.worker_attestation_key_id,
        gateway_receipt_key_id=system.gateway_receipt_key_id,
        guest_rpc_receipt_key_id=system.guest_rpc_receipt_key_id,
        receipt_key_id=system.production_receipt_key_id,
        gateway_transcript_sha256=gateway.seal.transcript_sha256,
        gateway_attempt_log_sha256=gateway.seal.attempt_log_sha256,
        guest_rpc_attempt_log_sha256=clinical_session.seal.attempt_log_sha256,
        guest_rpc_projected_tool_events_sha256=clinical_session.seal.projected_tool_events_sha256,
        submission_sha256=__import__('hashlib').sha256(submission_bytes).hexdigest(),
        submission_bytes=len(submission_bytes),
        usage=AgenticModelUsage(
            model_calls=gateway.seal.successful_call_count,
            input_tokens=gateway.seal.input_tokens,
            output_tokens=gateway.seal.output_tokens,
            reasoning_tokens=gateway.seal.reasoning_tokens,
            provider_cost_usd=gateway.seal.provider_cost_usd,
            gateway_metering_authoritative=gateway.transcript.metering_authoritative,
        ),
        gateway_attempt_count=gateway.seal.attempt_count,
        started_at=worker.attestation.started_at,
        finished_at=worker.attestation.finished_at,
        duration_ms=worker.attestation.duration_ms,
        sealed_at=worker.attestation.finished_at,
    )
    authenticated = AuthenticatedClinicalProductionRun(
        receipt=receipt,
        receipt_hmac_sha256='4' * 64,
    )
    authenticated_sha256 = hashlib.sha256(canonical_json_bytes(authenticated)).hexdigest()
    rpc_limits = ClinicalGuestRpcLimits(
        maximum_frame_body_bytes=clinical_session.policy.maximum_frame_body_bytes,
        maximum_session_wire_bytes=clinical_session.policy.maximum_session_wire_bytes,
        maximum_requests=clinical_session.policy.maximum_requests,
        maximum_list_entries=clinical_session.policy.maximum_list_entries,
        maximum_read_bytes=clinical_session.policy.maximum_read_bytes,
        maximum_search_results=clinical_session.policy.maximum_search_results,
        maximum_submission_bytes=clinical_session.policy.maximum_submission_bytes,
    )
    bootstrap_time = worker.attestation.finished_at
    hello = ClinicalGuestBootstrapHello(
        run_id=launch.run_id,
        start_redemption_sha256=attempt_sha256,
        session_id=start_redemption.guest_rpc_session_id,
        task_invocation=workspace.invocation,
        task_invocation_sha256=agentic_task_invocation_sha256(workspace.invocation),
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        execution_policy_sha256=system.execution_policy_sha256,
        worker_bootstrap_profile_sha256='d' * 64,
        worker_spec_sha256=system.worker_spec_sha256,
        harness_policy_id=LANE_A_GUEST_HARNESS_POLICY_ID,
        harness_policy_sha256=CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
        action_schema_sha256=LANE_A_GUEST_ACTION_SCHEMA_SHA256,
        rpc_limits=rpc_limits,
        nonce='9' * 64,
        valid_from=bootstrap_time,
        expires_at=bootstrap_time,
    )
    signed_hello = SignedClinicalGuestBootstrapHello(
        authorization_key_id=system.guest_bootstrap_authorization_key_id,
        hello=hello,
        hello_sha256=clinical_guest_bootstrap_hello_sha256(hello),
        signature_hex='a' * 128,
    )
    signed_hello_sha256 = clinical_guest_bootstrap_signed_hello_sha256(signed_hello)
    bootstrap_receipt = ClinicalGuestBootstrapReceipt(
        receipt_key_id=system.guest_bootstrap_receipt_key_id,
        authorization_key_id=system.guest_bootstrap_authorization_key_id,
        run_id=launch.run_id,
        start_redemption_sha256=attempt_sha256,
        session_id=start_redemption.guest_rpc_session_id,
        task_invocation_sha256=agentic_task_invocation_sha256(workspace.invocation),
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        execution_policy_sha256=system.execution_policy_sha256,
        worker_bootstrap_profile_sha256=hello.worker_bootstrap_profile_sha256,
        worker_spec_sha256=system.worker_spec_sha256,
        harness_policy_sha256=hello.harness_policy_sha256,
        action_schema_sha256=hello.action_schema_sha256,
        rpc_limits_sha256=hashlib.sha256(canonical_json_bytes(rpc_limits)).hexdigest(),
        nonce_sha256=hashlib.sha256(hello.nonce.encode('ascii')).hexdigest(),
        hello_sha256=signed_hello.hello_sha256,
        hello_bytes=len(canonical_json_bytes(hello)),
        signed_hello_sha256=signed_hello_sha256,
        signed_hello_bytes=len(canonical_json_bytes(signed_hello)),
        ack_sha256='b' * 64,
        ack_bytes=1,
        valid_from=bootstrap_time,
        expires_at=bootstrap_time,
        hello_sent_at=bootstrap_time,
        ack_received_at=bootstrap_time,
        guest_accepted_at=bootstrap_time,
    )
    bootstrap = AuthenticatedClinicalGuestBootstrap(
        signed_hello=signed_hello,
        receipt=bootstrap_receipt,
        receipt_hmac_sha256='c' * 64,
    )
    bootstrap_sha256 = hashlib.sha256(canonical_json_bytes(bootstrap)).hexdigest()
    outer_receipt = ClinicalProductionRunOuterReceiptV02(
        run_id=launch.run_id,
        start_redemption_sha256=attempt_sha256,
        guest_rpc_session_id=start_redemption.guest_rpc_session_id,
        task_invocation_sha256=agentic_task_invocation_sha256(workspace.invocation),
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        execution_policy_sha256=system.execution_policy_sha256,
        worker_spec_sha256=system.worker_spec_sha256,
        guest_rpc_policy_sha256=system.guest_rpc_policy_sha256,
        guest_rpc_limits_sha256=bootstrap_receipt.rpc_limits_sha256,
        base_authenticated_run_sha256=authenticated_sha256,
        clinical_guest_bootstrap_evidence_sha256=bootstrap_sha256,
        clinical_guest_bootstrap_receipt_key_id=system.guest_bootstrap_receipt_key_id,
        clinical_guest_bootstrap_authorization_key_id=system.guest_bootstrap_authorization_key_id,
        clinical_guest_bootstrap_signed_hello_sha256=signed_hello_sha256,
        clinical_guest_bootstrap_hello_sha256=signed_hello.hello_sha256,
        bootstrap_valid_from=bootstrap_time,
        bootstrap_expires_at=bootstrap_time,
        bootstrap_hello_sent_at=bootstrap_time,
        bootstrap_ack_received_at=bootstrap_time,
        bootstrap_guest_accepted_at=bootstrap_time,
        sealed_at=receipt.sealed_at,
        receipt_key_id=system.production_receipt_key_id,
    )
    outer = AuthenticatedClinicalProductionRunV02(
        receipt=outer_receipt,
        base_authenticated_run=authenticated,
        receipt_hmac_sha256='d' * 64,
    )
    root = tmp_path / f'evidence-{launch.run_id}'
    root.mkdir(mode=0o700)
    return LoadedClinicalProductionRunV02(
        root=root,
        workspace=workspace,
        submission=submission,
        gateway_session=gateway,
        guest_rpc_session=clinical_session,
        worker_attestation=worker,
        authenticated_receipt=authenticated,
        authenticated_receipt_sha256=authenticated_sha256,
        authenticated_outer_receipt=outer,
        authenticated_outer_receipt_sha256=hashlib.sha256(canonical_json_bytes(outer)).hexdigest(),
        clinical_guest_bootstrap=bootstrap,
        clinical_guest_bootstrap_evidence_sha256=bootstrap_sha256,
    )


def _reservation(tmp_path: Path):
    case, workspace = _workspace(tmp_path, 1)
    manifest = _cohort_manifest((case,))
    system, worker, gateway, session, _, _ = _system(tmp_path, workspace)
    registry = SqliteClinicalProductionRegistry(tmp_path / 'registry.sqlite3', authority_id='lane-a-test')
    context = registry.reserve_cohort(
        manifest=manifest,
        workspaces=(workspace,),
        workspace_receipt_keys_by_id=_workspace_keys(),
        system=system,
        registered_entry_id='entry-one',
        reserved_at=datetime.now(UTC),
    )
    return registry, context, manifest, workspace, system, worker, gateway, session


def _replace_bootstrap(
    loaded: LoadedClinicalProductionRunV02,
    bootstrap: AuthenticatedClinicalGuestBootstrap,
    **outer_receipt_updates: object,
) -> LoadedClinicalProductionRunV02:
    bootstrap_sha256 = hashlib.sha256(canonical_json_bytes(bootstrap)).hexdigest()
    outer_receipt = loaded.authenticated_outer_receipt.receipt.model_copy(
        update={
            'clinical_guest_bootstrap_evidence_sha256': bootstrap_sha256,
            'clinical_guest_bootstrap_signed_hello_sha256': (
                clinical_guest_bootstrap_signed_hello_sha256(bootstrap.signed_hello)
            ),
            'clinical_guest_bootstrap_hello_sha256': bootstrap.signed_hello.hello_sha256,
            **outer_receipt_updates,
        }
    )
    outer = loaded.authenticated_outer_receipt.model_copy(update={'receipt': outer_receipt})
    return replace(
        loaded,
        authenticated_outer_receipt=outer,
        authenticated_outer_receipt_sha256=hashlib.sha256(canonical_json_bytes(outer)).hexdigest(),
        clinical_guest_bootstrap=bootstrap,
        clinical_guest_bootstrap_evidence_sha256=bootstrap_sha256,
    )


def _redeem_start(registry, context, workspace, system, worker, gateway, session, launch):
    return registry.redeem_task_start(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        launch_sha256=clinical_production_task_launch_sha256(launch),
        canonical_launcher_id=system.canonical_launcher_id,
        canonical_launcher_executable_sha256=system.canonical_launcher_executable_sha256,
        prepared_worker_sha256=worker.attestation.prepared_worker_sha256,
        guest_rpc_session_id=session.seal.session_id,
        gateway_capability_id=gateway.grant.capability_id,
        redeemed_at=launch.claimed_at,
    )


def test_full_production_evidence_yields_exact_cohort_submission(tmp_path: Path) -> None:
    registry, context, manifest, workspace, system, worker, gateway, session = _reservation(tmp_path)
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='a' * 32,
        claimed_at=datetime.now(UTC),
    )
    start_redemption = _redeem_start(
        registry,
        context,
        workspace,
        system,
        worker,
        gateway,
        session,
        launch,
    )
    loaded = _loaded_evidence(
        tmp_path,
        system=system,
        workspace=workspace,
        launch=launch,
        worker=worker,
        gateway=gateway,
        clinical_session=session,
        start_redemption=start_redemption,
    )
    observed_hashes: list[str] = []

    def reauthenticate(root: Path, expected_attempt_sha256: str) -> LoadedClinicalProductionRun:
        assert root == loaded.root
        observed_hashes.append(expected_attempt_sha256)
        return loaded

    record = registry.record_production_run(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        production_run_root=loaded.root,
        reauthenticate=reauthenticate,
        terminal_at=datetime.now(UTC),
    )
    result = registry.result(reservation_sha256=context.reservation_sha256, manifest=manifest)

    assert observed_hashes == [clinical_production_start_redemption_sha256(start_redemption)]
    assert record.state == 'succeeded'
    assert result.status == ClinicalProductionCohortStatus.COMPLETED
    assert result.cohort_submission is not None
    assert result.cohort_submission.submissions[0] == session.submission
    assert result.worker_gateway_route_model_and_harness_pins_checked
    assert result.one_attempt_enforced_within_registry_authority
    assert result.one_start_redemption_per_task_enforced
    assert result.start_redemption_bound_to_prepared_worker_and_sessions
    assert result.bootstrap_keys_bound_in_system_identity
    assert result.provider_subprocess_bound_in_system_identity
    assert result.provider_child_module_source_bound_in_system_identity
    assert result.strict_signed_guest_bootstrap_required
    assert result.bootstrap_bound_outer_v02_evidence_required
    assert not result.cross_registry_global_uniqueness_claimed
    assert result.clinical_production_run_finalizer_available
    assert not result.official_execution_qualified


def test_registry_v04_rejects_legacy_v01_evidence_downgrade(tmp_path: Path) -> None:
    registry, context, _, workspace, system, worker, gateway, session = _reservation(tmp_path)
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='1' * 32,
        claimed_at=datetime.now(UTC),
    )
    redemption = _redeem_start(
        registry,
        context,
        workspace,
        system,
        worker,
        gateway,
        session,
        launch,
    )
    loaded = _loaded_evidence(
        tmp_path,
        system=system,
        workspace=workspace,
        launch=launch,
        worker=worker,
        gateway=gateway,
        clinical_session=session,
        start_redemption=redemption,
    )
    legacy = LoadedClinicalProductionRun(
        root=loaded.root,
        workspace=loaded.workspace,
        submission=loaded.submission,
        worker_attestation=loaded.worker_attestation,
        gateway_session=loaded.gateway_session,
        guest_rpc_session=loaded.guest_rpc_session,
        authenticated_receipt=loaded.authenticated_receipt,
        authenticated_receipt_sha256=loaded.authenticated_receipt_sha256,
    )

    record = registry.record_production_run(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        production_run_root=legacy.root,
        reauthenticate=lambda *_: legacy,  # type: ignore[return-value]
        terminal_at=datetime.now(UTC),
    )

    assert record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED
    assert record.evidence_sha256 is None


def test_registry_rejects_coherently_rehashed_bootstrap_key_substitution(tmp_path: Path) -> None:
    registry, context, _, workspace, system, worker, gateway, session = _reservation(tmp_path)
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='2' * 32,
        claimed_at=datetime.now(UTC),
    )
    redemption = _redeem_start(
        registry,
        context,
        workspace,
        system,
        worker,
        gateway,
        session,
        launch,
    )
    loaded = _loaded_evidence(
        tmp_path,
        system=system,
        workspace=workspace,
        launch=launch,
        worker=worker,
        gateway=gateway,
        clinical_session=session,
        start_redemption=redemption,
    )
    wrong_key_id = '0' * 64
    signed = loaded.clinical_guest_bootstrap.signed_hello.model_copy(update={'authorization_key_id': wrong_key_id})
    receipt = loaded.clinical_guest_bootstrap.receipt.model_copy(
        update={
            'authorization_key_id': wrong_key_id,
            'signed_hello_sha256': clinical_guest_bootstrap_signed_hello_sha256(signed),
        }
    )
    bootstrap = loaded.clinical_guest_bootstrap.model_copy(update={'signed_hello': signed, 'receipt': receipt})
    wrong = _replace_bootstrap(
        loaded,
        bootstrap,
        clinical_guest_bootstrap_authorization_key_id=wrong_key_id,
    )

    record = registry.record_production_run(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        production_run_root=wrong.root,
        reauthenticate=lambda *_: wrong,
        terminal_at=datetime.now(UTC),
    )

    assert record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_BINDING_MISMATCH
    assert record.evidence_sha256 == wrong.authenticated_outer_receipt_sha256


def test_registry_rejects_coherently_rehashed_bootstrap_session_substitution(tmp_path: Path) -> None:
    registry, context, _, workspace, system, worker, gateway, session = _reservation(tmp_path)
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='3' * 32,
        claimed_at=datetime.now(UTC),
    )
    redemption = _redeem_start(
        registry,
        context,
        workspace,
        system,
        worker,
        gateway,
        session,
        launch,
    )
    loaded = _loaded_evidence(
        tmp_path,
        system=system,
        workspace=workspace,
        launch=launch,
        worker=worker,
        gateway=gateway,
        clinical_session=session,
        start_redemption=redemption,
    )
    wrong_session_id = 'f' * 32
    hello = loaded.clinical_guest_bootstrap.signed_hello.hello.model_copy(update={'session_id': wrong_session_id})
    signed = loaded.clinical_guest_bootstrap.signed_hello.model_copy(
        update={
            'hello': hello,
            'hello_sha256': clinical_guest_bootstrap_hello_sha256(hello),
        }
    )
    receipt = loaded.clinical_guest_bootstrap.receipt.model_copy(
        update={
            'session_id': wrong_session_id,
            'hello_sha256': signed.hello_sha256,
            'signed_hello_sha256': clinical_guest_bootstrap_signed_hello_sha256(signed),
        }
    )
    bootstrap = loaded.clinical_guest_bootstrap.model_copy(update={'signed_hello': signed, 'receipt': receipt})
    wrong = _replace_bootstrap(
        loaded,
        bootstrap,
        guest_rpc_session_id=wrong_session_id,
    )

    record = registry.record_production_run(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        production_run_root=wrong.root,
        reauthenticate=lambda *_: wrong,
        terminal_at=datetime.now(UTC),
    )

    assert record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_BINDING_MISMATCH


def test_registry_rejects_false_outer_receipt_hash_claim(tmp_path: Path) -> None:
    registry, context, _, workspace, system, worker, gateway, session = _reservation(tmp_path)
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='4' * 32,
        claimed_at=datetime.now(UTC),
    )
    redemption = _redeem_start(
        registry,
        context,
        workspace,
        system,
        worker,
        gateway,
        session,
        launch,
    )
    loaded = _loaded_evidence(
        tmp_path,
        system=system,
        workspace=workspace,
        launch=launch,
        worker=worker,
        gateway=gateway,
        clinical_session=session,
        start_redemption=redemption,
    )
    wrong = replace(loaded, authenticated_outer_receipt_sha256='f' * 64)

    record = registry.record_production_run(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        production_run_root=wrong.root,
        reauthenticate=lambda *_: wrong,
        terminal_at=datetime.now(UTC),
    )

    assert record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED
    assert (
        record.evidence_sha256 == hashlib.sha256(canonical_json_bytes(loaded.authenticated_outer_receipt)).hexdigest()
    )


def test_same_executable_cannot_reregister_by_renaming_harness(tmp_path: Path) -> None:
    registry, context, manifest, workspace, system, *_ = _reservation(tmp_path)
    renamed_route = system.gateway_route.model_copy(
        update={'route_id': 'renamed-route', 'logical_model_id': 'renamed-logical-model'}
    )
    renamed = system.model_copy(
        update={
            'harness': system.harness.model_copy(
                update={
                    'harness_id': 'renamed-harness',
                    'harness_version': 'marketing-v2',
                    'requested_model_id': renamed_route.logical_model_id,
                }
            ),
            'gateway_route': renamed_route,
            'gateway_route_sha256': gateway_model_route_sha256(renamed_route),
        }
    )

    with pytest.raises(ClinicalProductionRegistryError, match='renaming'):
        registry.reserve_cohort(
            manifest=manifest,
            workspaces=(workspace,),
            workspace_receipt_keys_by_id=_workspace_keys(),
            system=renamed,
            registered_entry_id='renamed-entry',
            reserved_at=context.reservation.reserved_at,
        )


@pytest.mark.parametrize(
    'field',
    ('provider_subprocess_behavior_sha256', 'provider_subprocess_module_source_sha256'),
)
def test_provider_child_change_alters_exact_and_alias_resistant_system_identity(
    tmp_path: Path,
    field: str,
) -> None:
    _, _, _, _, system, *_ = _reservation(tmp_path)
    changed = system.model_copy(update={field: 'f' * 64})

    assert system.schema_version == 'vaxreplay.clinical-production-system-identity.dev-v0.5'
    assert clinical_production_system_identity_sha256(changed) != (clinical_production_system_identity_sha256(system))
    assert clinical_production_system_core_sha256(changed) != (clinical_production_system_core_sha256(system))


def test_provider_child_path_only_change_alters_exact_identity_but_not_core(
    tmp_path: Path,
) -> None:
    _, _, _, _, system, *_ = _reservation(tmp_path)
    changed = system.model_copy(update={'provider_subprocess_spec_sha256': 'f' * 64})

    assert clinical_production_system_identity_sha256(changed) != (clinical_production_system_identity_sha256(system))
    assert clinical_production_system_core_sha256(changed) == (clinical_production_system_core_sha256(system))


def test_one_launch_and_first_terminal_failure_are_permanent(tmp_path: Path) -> None:
    registry, context, manifest, workspace, system, worker, gateway, session = _reservation(tmp_path)
    episode_id = workspace.task.context.episode_id
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=episode_id,
        run_id='b' * 32,
        claimed_at=datetime.now(UTC),
    )
    _redeem_start(registry, context, workspace, system, worker, gateway, session, launch)
    with pytest.raises(ClinicalProductionRegistryError, match='one launch'):
        registry.claim_task_launch(
            reservation_sha256=context.reservation_sha256,
            episode_id=episode_id,
            run_id='c' * 32,
            claimed_at=datetime.now(UTC),
        )
    first = registry.record_explicit_failure(
        reservation_sha256=context.reservation_sha256,
        episode_id=episode_id,
        terminal_code=ClinicalProductionTerminalCode.WORKER_LOST,
        failure_record=b'worker disappeared after launch',
        terminal_at=datetime.now(UTC),
    )
    with pytest.raises(ClinicalProductionRegistryError, match='not awaiting'):
        registry.record_production_run(
            reservation_sha256=context.reservation_sha256,
            episode_id=episode_id,
            production_run_root=tmp_path,
            reauthenticate=lambda *_: pytest.fail('terminal task must not be reauthenticated'),
            terminal_at=datetime.now(UTC),
        )
    result = registry.result(reservation_sha256=context.reservation_sha256, manifest=manifest)
    assert first.terminal_code == ClinicalProductionTerminalCode.WORKER_LOST
    assert result.status == ClinicalProductionCohortStatus.FAILED
    assert result.tasks[0].terminal_code == ClinicalProductionTerminalCode.WORKER_LOST


def test_two_registry_clients_racing_for_one_task_get_exactly_one_launch(tmp_path: Path) -> None:
    registry, context, _, workspace, *_ = _reservation(tmp_path)
    second_client = SqliteClinicalProductionRegistry(registry.path, authority_id=registry.authority_id)
    barrier = threading.Barrier(2)

    def claim(client: SqliteClinicalProductionRegistry, run_id: str) -> str:
        barrier.wait()
        try:
            client.claim_task_launch(
                reservation_sha256=context.reservation_sha256,
                episode_id=workspace.task.context.episode_id,
                run_id=run_id,
                claimed_at=datetime.now(UTC),
            )
        except ClinicalProductionRegistryError:
            return 'rejected'
        return 'launched'

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            future.result()
            for future in (
                executor.submit(claim, registry, 'e' * 32),
                executor.submit(claim, second_client, 'f' * 32),
            )
        )
    assert sorted(outcomes) == ['launched', 'rejected']


def test_two_workers_presenting_one_launch_ticket_get_one_start_redemption(tmp_path: Path) -> None:
    registry, context, manifest, workspace, system, *_ = _reservation(tmp_path)
    second_client = SqliteClinicalProductionRegistry(registry.path, authority_id=registry.authority_id)
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='7' * 32,
        claimed_at=datetime.now(UTC),
    )
    launch_sha256 = clinical_production_task_launch_sha256(launch)
    barrier = threading.Barrier(2)

    def redeem(
        client: SqliteClinicalProductionRegistry,
        prepared_worker_sha256: str,
        guest_rpc_session_id: str,
        gateway_capability_id: str,
    ) -> str:
        barrier.wait()
        try:
            client.redeem_task_start(
                reservation_sha256=context.reservation_sha256,
                episode_id=workspace.task.context.episode_id,
                launch_sha256=launch_sha256,
                canonical_launcher_id=system.canonical_launcher_id,
                canonical_launcher_executable_sha256=(system.canonical_launcher_executable_sha256),
                prepared_worker_sha256=prepared_worker_sha256,
                guest_rpc_session_id=guest_rpc_session_id,
                gateway_capability_id=gateway_capability_id,
                redeemed_at=launch.claimed_at,
            )
        except ClinicalProductionRegistryError:
            return 'rejected'
        return 'redeemed'

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            future.result()
            for future in (
                executor.submit(redeem, registry, '7' * 64, '1' * 32, '9' * 64),
                executor.submit(redeem, second_client, '8' * 64, '2' * 32, 'a' * 64),
            )
        )

    assert sorted(outcomes) == ['redeemed', 'rejected']
    task = registry.result(
        reservation_sha256=context.reservation_sha256,
        manifest=manifest,
    )
    assert task.tasks[0].start_redemption is not None
    assert task.tasks[0].start_redemption.worker_start_count == 1


def test_unredeemed_launch_cannot_enter_production_evidence_collector(tmp_path: Path) -> None:
    registry, context, _, workspace, *_ = _reservation(tmp_path)
    registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='8' * 32,
        claimed_at=datetime.now(UTC),
    )
    called = False

    def reauthenticate(*_args):
        nonlocal called
        called = True
        pytest.fail('unredeemed launch must fail before evidence authentication')

    with pytest.raises(ClinicalProductionRegistryError, match='not redeemed'):
        registry.record_production_run(
            reservation_sha256=context.reservation_sha256,
            episode_id=workspace.task.context.episode_id,
            production_run_root=tmp_path,
            reauthenticate=reauthenticate,
            terminal_at=datetime.now(UTC),
        )
    assert not called


def test_authentication_or_binding_failure_burns_attempt_without_cherry_pick(tmp_path: Path) -> None:
    registry, context, manifest, workspace, system, worker, gateway, session = _reservation(tmp_path)
    episode_id = workspace.task.context.episode_id
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=episode_id,
        run_id='d' * 32,
        claimed_at=datetime.now(UTC),
    )
    _redeem_start(registry, context, workspace, system, worker, gateway, session, launch)

    record = registry.record_production_run(
        reservation_sha256=context.reservation_sha256,
        episode_id=episode_id,
        production_run_root=tmp_path,
        reauthenticate=lambda *_: (_ for _ in ()).throw(ValueError('forged HMAC')),
        terminal_at=datetime.now(UTC),
    )
    assert record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED
    assert (
        registry.result(
            reservation_sha256=context.reservation_sha256,
            manifest=manifest,
        ).status
        == ClinicalProductionCohortStatus.FAILED
    )


def test_authenticated_package_from_wrong_resolved_model_is_terminal_failure(tmp_path: Path) -> None:
    registry, context, manifest, workspace, system, worker, gateway, session = _reservation(tmp_path)
    episode_id = workspace.task.context.episode_id
    launch = registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=episode_id,
        run_id='0' * 32,
        claimed_at=datetime.now(UTC),
    )
    start_redemption = _redeem_start(
        registry,
        context,
        workspace,
        system,
        worker,
        gateway,
        session,
        launch,
    )
    loaded = _loaded_evidence(
        tmp_path,
        system=system,
        workspace=workspace,
        launch=launch,
        worker=worker,
        gateway=gateway,
        clinical_session=session,
        start_redemption=start_redemption,
    )
    wrong_base_receipt = loaded.receipt.model_copy(update={'resolved_model_id': 'different-provider-model'})
    wrong_authenticated = loaded.authenticated_receipt.model_copy(update={'receipt': wrong_base_receipt})
    wrong_base_sha256 = hashlib.sha256(canonical_json_bytes(wrong_authenticated)).hexdigest()
    wrong_outer_receipt = loaded.authenticated_outer_receipt.receipt.model_copy(
        update={'base_authenticated_run_sha256': wrong_base_sha256}
    )
    wrong_outer = loaded.authenticated_outer_receipt.model_copy(
        update={
            'receipt': wrong_outer_receipt,
            'base_authenticated_run': wrong_authenticated,
        }
    )
    wrong = LoadedClinicalProductionRunV02(
        root=loaded.root,
        workspace=loaded.workspace,
        submission=loaded.submission,
        gateway_session=loaded.gateway_session,
        guest_rpc_session=loaded.guest_rpc_session,
        worker_attestation=loaded.worker_attestation,
        authenticated_receipt=wrong_authenticated,
        authenticated_receipt_sha256=wrong_base_sha256,
        authenticated_outer_receipt=wrong_outer,
        authenticated_outer_receipt_sha256=hashlib.sha256(canonical_json_bytes(wrong_outer)).hexdigest(),
        clinical_guest_bootstrap=loaded.clinical_guest_bootstrap,
        clinical_guest_bootstrap_evidence_sha256=loaded.clinical_guest_bootstrap_evidence_sha256,
    )

    record = registry.record_production_run(
        reservation_sha256=context.reservation_sha256,
        episode_id=episode_id,
        production_run_root=wrong.root,
        reauthenticate=lambda *_: wrong,
        terminal_at=datetime.now(UTC),
    )
    assert record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_BINDING_MISMATCH
    assert record.terminal_record_sha256 is not None
    assert (
        registry.result(
            reservation_sha256=context.reservation_sha256,
            manifest=manifest,
        ).status
        == ClinicalProductionCohortStatus.FAILED
    )


def test_reservation_requires_exact_cohort_workspace_coverage(tmp_path: Path) -> None:
    case_1, workspace_1 = _workspace(tmp_path, 1)
    case_2, _ = _workspace(tmp_path, 2)
    manifest = _cohort_manifest((case_1, case_2))
    system, *_ = _system(tmp_path, workspace_1)
    registry = SqliteClinicalProductionRegistry(tmp_path / 'registry.sqlite3', authority_id='lane-a-test')

    with pytest.raises(ClinicalProductionRegistryError, match='exactly cover'):
        registry.reserve_cohort(
            manifest=manifest,
            workspaces=(workspace_1,),
            workspace_receipt_keys_by_id=_workspace_keys(),
            system=system,
            registered_entry_id='partial-entry',
            reserved_at=datetime.now(UTC),
        )


def test_reported_or_alias_model_id_never_passes_official_snapshot_gate(tmp_path: Path) -> None:
    _, workspace = _workspace(tmp_path, 1)
    system, *_ = _system(tmp_path, workspace)

    with pytest.raises(ClinicalProductionRegistryError, match='externally attested immutable'):
        require_official_model_snapshot_attestation(
            system,
            external_attestation_sha256=None,
        )
    with pytest.raises(ClinicalProductionRegistryError, match='externally attested immutable'):
        require_official_model_snapshot_attestation(
            system,
            external_attestation_sha256='a' * 64,
        )


def test_registry_rejects_preexisting_public_or_hardlinked_database_path(tmp_path: Path) -> None:
    public = tmp_path / 'public.sqlite3'
    public.write_bytes(b'not a trusted database')
    public.chmod(0o644)
    with pytest.raises(ClinicalProductionRegistryError, match='private mode-0600'):
        SqliteClinicalProductionRegistry(public, authority_id='lane-a-test')

    private = tmp_path / 'private.sqlite3'
    private.write_bytes(b'not a trusted database')
    private.chmod(0o600)
    hardlink = tmp_path / 'hardlink.sqlite3'
    hardlink.hardlink_to(private)
    with pytest.raises(ClinicalProductionRegistryError, match='private mode-0600'):
        SqliteClinicalProductionRegistry(private, authority_id='lane-a-test')


def test_registry_safely_creates_private_parent_for_database_and_sidecars(tmp_path: Path) -> None:
    parent = tmp_path / 'organizer-state' / 'clinical-registry'

    registry = SqliteClinicalProductionRegistry(
        parent / 'registry.sqlite3',
        authority_id='lane-a-test',
    )

    metadata = parent.lstat()
    assert registry.path.parent == parent.resolve()
    assert stat.S_ISDIR(metadata.st_mode)
    assert metadata.st_uid == os.geteuid()
    assert stat.S_IMODE(metadata.st_mode) == 0o700


@pytest.mark.parametrize('mode', [0o750, 0o777])
def test_registry_rejects_nonprivate_parent_directory(tmp_path: Path, mode: int) -> None:
    parent = tmp_path / f'nonprivate-{mode:o}'
    parent.mkdir(mode=0o700)
    parent.chmod(mode)
    try:
        with pytest.raises(ClinicalProductionRegistryError, match='private mode-0700'):
            SqliteClinicalProductionRegistry(
                parent / 'registry.sqlite3',
                authority_id='lane-a-test',
            )
    finally:
        parent.chmod(0o700)


def test_registry_rejects_symlink_parent_even_when_target_is_private(tmp_path: Path) -> None:
    target = tmp_path / 'private-target'
    target.mkdir(mode=0o700)
    parent = tmp_path / 'parent-link'
    parent.symlink_to(target, target_is_directory=True)

    with pytest.raises(ClinicalProductionRegistryError, match='parent directory cannot be a symlink'):
        SqliteClinicalProductionRegistry(
            parent / 'registry.sqlite3',
            authority_id='lane-a-test',
        )


def test_registry_rejects_parent_not_owned_by_current_user(tmp_path: Path, monkeypatch) -> None:
    parent = tmp_path / 'foreign-owned'
    parent.mkdir(mode=0o700)
    actual_euid = os.geteuid()
    monkeypatch.setattr(os, 'geteuid', lambda: actual_euid + 1)

    with pytest.raises(ClinicalProductionRegistryError, match='owned by the current user'):
        SqliteClinicalProductionRegistry(
            parent / 'registry.sqlite3',
            authority_id='lane-a-test',
        )


def test_registry_rechecks_parent_privacy_before_each_connection(tmp_path: Path) -> None:
    parent = tmp_path / 'mutable-parent'
    parent.mkdir(mode=0o700)
    registry = SqliteClinicalProductionRegistry(
        parent / 'registry.sqlite3',
        authority_id='lane-a-test',
    )
    parent.chmod(0o777)
    try:
        with pytest.raises(ClinicalProductionRegistryError, match='private mode-0700'):
            registry.task_records('a' * 64)
    finally:
        parent.chmod(0o700)


def test_registry_rejects_parent_directory_replacement(tmp_path: Path) -> None:
    parent = tmp_path / 'replaceable-parent'
    parent.mkdir(mode=0o700)
    registry = SqliteClinicalProductionRegistry(
        parent / 'registry.sqlite3',
        authority_id='lane-a-test',
    )
    displaced = tmp_path / 'displaced-parent'
    parent.rename(displaced)
    parent.mkdir(mode=0o700)

    with pytest.raises(ClinicalProductionRegistryError, match='replaced after initialization'):
        registry.task_records('a' * 64)
