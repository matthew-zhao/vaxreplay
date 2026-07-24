from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import vaxreplay.agentic.clinical_production_run_v02 as production_v02
import vaxreplay.agentic.firecracker_clinical_runtime as runtime_module
from tests.test_agentic_clinical_production_run import (
    GATEWAY_KEY,
    GUEST_KEY,
    PRODUCTION_KEY,
    RUN_ID,
    WORKER_KEY,
    WORKSPACE_KEY,
    Materials,
    _materials,
)
from tests.test_agentic_clinical_production_run_v02 import (
    BOOTSTRAP_AUTHORIZATION_KEY_ID,
    BOOTSTRAP_RECEIPT_KEY,
    BOOTSTRAP_SIGNER,
    _bootstrap,
)
from tests.test_agentic_firecracker_clinical_runtime import (
    _PROVIDER_SUBPROCESS_BEHAVIOR_SHA256,
    _PROVIDER_SUBPROCESS_MODULE_SOURCE_SHA256,
    _PROVIDER_SUBPROCESS_SPEC_SHA256,
    _artifact_for_hello,
    _Gateway,
    _SecretStore,
    _Supervisor,
)
from vaxreplay.agentic.clinical_guest_bootstrap import (
    AuthenticatedClinicalGuestBootstrap,
    ClinicalGuestBootstrapHello,
    ClinicalGuestBootstrapTrustAnchor,
    clinical_guest_bootstrap_receipt_key_id,
)
from vaxreplay.agentic.clinical_launcher import (
    CanonicalClinicalLauncherDeployment,
    ClinicalPreparedRuntime,
    ClinicalRuntimeCompleted,
    ClinicalRuntimeFailed,
    ClinicalRuntimeFailureCode,
    ClinicalRuntimePrepareRequest,
    ClinicalRuntimeStart,
    canonical_clinical_launcher_deployment_sha256,
    clinical_prepared_runtime_sha256,
)
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionReservation,
    ClinicalProductionStartRedemption,
    ClinicalProductionSystemIdentity,
    ClinicalProductionTaskBinding,
    ClinicalProductionTaskLaunch,
    clinical_production_reservation_sha256,
    clinical_production_start_redemption_sha256,
    clinical_production_system_core_sha256,
    clinical_production_system_identity_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.clinical_production_run import clinical_production_run_key_id
from vaxreplay.agentic.firecracker import (
    FirecrackerCleanupReceipt,
    FirecrackerPreparedWorker,
    RunningFirecrackerWorker,
    firecracker_attestation_key_id,
    firecracker_model_sha256,
)
from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalRuntime,
    FirecrackerClinicalRuntimeConfig,
    FirecrackerClinicalRuntimeError,
    FirecrackerClinicalRuntimeKeys,
    FirecrackerClinicalStartupCleanupReceipt,
    FirecrackerClinicalStartupReconciliationRequest,
    firecracker_clinical_runtime_config_sha256,
    firecracker_clinical_startup_reconciliation_request_sha256,
    reconcile_firecracker_clinical_startup_without_execution,
)
from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcHostSession,
    GuestRpcTerminalStatus,
    guest_rpc_policy_sha256,
    guest_rpc_session_key_id,
)
from vaxreplay.agentic.protocol import agentic_policy_sha256
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewaySession,
    AuthenticatedProviderGateway,
    GatewayCapabilityGrant,
    GatewayTerminalReason,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    gateway_session_key_id,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split


class _Process:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.pid = 4242
        self.returncode = 0

    def wait(self, *, timeout: float) -> int:
        assert timeout > 0
        self.events.append('worker.wait')
        return self.returncode


class _BoundarySupervisor(_Supervisor):
    def __init__(
        self,
        materials: Materials,
        root: Path,
        events: list[str],
        now: datetime,
        *,
        cleanup_fails: bool,
    ) -> None:
        super().__init__(materials, root, events, now)
        self.cleanup_fails = cleanup_fails

    def launch(self, prepared: FirecrackerPreparedWorker) -> RunningFirecrackerWorker:
        self.events.append('worker.launch')
        self.launch_count += 1
        return cast(
            RunningFirecrackerWorker,
            SimpleNamespace(
                prepared=prepared,
                wall_deadline_monotonic=2.0,
                firecracker_pid=4343,
                process=_Process(self.events),
            ),
        )

    def wait_for_exit(
        self,
        running: RunningFirecrackerWorker,
        *,
        timeout_seconds: float,
    ) -> bool:
        del running
        assert timeout_seconds > 0
        self.events.append('worker.wait')
        return True

    def terminate_and_cleanup(
        self,
        running: RunningFirecrackerWorker,
        *,
        grace_seconds: float = 5.0,
    ) -> FirecrackerCleanupReceipt:
        if self.cleanup_fails:
            self.events.append('worker.cleanup-failed')
            raise RuntimeError('private cleanup failure')
        return super().terminate_and_cleanup(running, grace_seconds=grace_seconds)


class _BoundaryGateway(_Gateway):
    def __init__(self, materials: Materials, store: _SecretStore, events: list[str]) -> None:
        super().__init__(materials, store, events)
        self.materials = materials
        self.grants: list[GatewayCapabilityGrant] = []

    def register_session(self, **kwargs: Any) -> None:
        self.events.append('gateway.register')
        self.grants.append(kwargs['grant'])

    def seal_session(
        self,
        _capability_id: str,
        *,
        terminal_reason: GatewayTerminalReason,
        sealed_at: datetime,
        revoke_secret: bool,
    ) -> AuthenticatedGatewaySession:
        del sealed_at
        self.events.append(f'gateway.seal:{terminal_reason.value}')
        self.revoke_secret_arguments.append(revoke_secret)
        return self.materials.gateway


class _CompletedGuestSession:
    def __init__(self, materials: Materials, events: list[str]) -> None:
        self.materials = materials
        self.events = events
        self.terminal = True
        self.final_submission_bytes = canonical_json_bytes(materials.submission)

    def abort(self, _code: object) -> None:
        raise AssertionError('a completed guest session must not be aborted')

    def seal(self, *, sealed_at: datetime) -> AuthenticatedGuestRpcSession:
        del sealed_at
        self.events.append('guest.seal')
        assert self.materials.guest.seal.terminal_status == GuestRpcTerminalStatus.COMPLETED
        return self.materials.guest


class _BoundaryBootstrapRunner:
    authenticated_bootstrap: AuthenticatedClinicalGuestBootstrap | None = None

    def __init__(
        self,
        *,
        events: list[str],
        journal: Any,
        anchor: ClinicalGuestBootstrapTrustAnchor,
        now: datetime,
        fail_before_ack: bool,
        mutate_nonce: bool,
    ) -> None:
        self.events = events
        self.journal = journal
        self.anchor = anchor
        self.now = now
        self.fail_before_ack = fail_before_ack
        self.mutate_nonce = mutate_nonce

    def open(self) -> None:
        self.events.append('bootstrap.open')

    def serve_one(
        self,
        *,
        hello: ClinicalGuestBootstrapHello,
        **_kwargs: Any,
    ) -> AuthenticatedClinicalGuestBootstrap:
        self.events.append('bootstrap.serve')
        if self.fail_before_ack:
            raise FirecrackerClinicalRuntimeError('private pre-ACK bootstrap failure')
        if self.mutate_nonce:
            hello = hello.model_copy(update={'nonce': 'd' * 64})
        artifact = _artifact_for_hello(hello, anchor=self.anchor, now=self.now)
        self.authenticated_bootstrap = artifact
        self.journal(artifact)
        self.events.append('bootstrap.journal')
        return artifact

    def close(self) -> None:
        self.events.append('bootstrap.close')


class _StartupReconciler:
    def __init__(self, *, mismatch: bool = False, fails: bool = False) -> None:
        self.mismatch = mismatch
        self.fails = fails
        self.calls: list[FirecrackerClinicalStartupReconciliationRequest] = []

    def reconcile(
        self,
        request: FirecrackerClinicalStartupReconciliationRequest,
    ) -> FirecrackerClinicalStartupCleanupReceipt:
        self.calls.append(request)
        if self.fails:
            raise RuntimeError('private deployment discovery failure')
        return FirecrackerClinicalStartupCleanupReceipt(
            reconciler_id='test-host-orphan-reconciler',
            reconciler_version='1',
            reconciliation_request_sha256=(
                'f' * 64 if self.mismatch else firecracker_clinical_startup_reconciliation_request_sha256(request)
            ),
            retained_journal_count=len(request.retained_journals),
            worker_inventory_sha256='1' * 64,
            ephemeral_run_artifact_inventory_sha256='4' * 64,
            capability_inventory_sha256='2' * 64,
            attempt_registry_inventory_sha256='5' * 64,
            cleanup_evidence_sha256='3' * 64,
            discovered_worker_count=1,
            terminated_worker_count=1,
            discovered_ephemeral_run_artifact_count=1,
            removed_ephemeral_run_artifact_count=1,
            discovered_capability_count=1,
            revoked_capability_count=1,
            reconciled_at=request.requested_at,
        )


@dataclass(slots=True)
class _BoundaryCase:
    runtime: FirecrackerClinicalRuntime
    request: ClinicalRuntimePrepareRequest
    prepared: ClinicalPreparedRuntime
    start: ClinicalRuntimeStart
    supervisor: _BoundarySupervisor
    gateway: _BoundaryGateway
    store: _SecretStore
    events: list[str]
    journal_path: Path
    bootstrap_artifacts: list[AuthenticatedClinicalGuestBootstrap]


def _case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_before_ack: bool = False,
    cleanup_fails: bool = False,
    mutate_nonce: bool = False,
) -> _BoundaryCase:
    materials = _materials(tmp_path / 'materials')
    _, anchor = _bootstrap(materials)
    now = materials.worker.attestation.started_at
    events: list[str] = []
    store = _SecretStore(events)
    gateway = _BoundaryGateway(materials, store, events)
    supervisor = _BoundarySupervisor(
        materials,
        tmp_path / 'jails',
        events,
        now,
        cleanup_fails=cleanup_fails,
    )
    system = ClinicalProductionSystemIdentity(
        harness=materials.harness,
        execution_policy_sha256=agentic_policy_sha256(materials.policy),
        worker_spec_sha256=firecracker_model_sha256(materials.spec),
        gateway_policy_sha256=authenticated_gateway_policy_sha256(materials.gateway.policy),
        gateway_route=materials.gateway.route,
        gateway_route_sha256=gateway_model_route_sha256(materials.gateway.route),
        provider_subprocess_spec_sha256=_PROVIDER_SUBPROCESS_SPEC_SHA256,
        provider_subprocess_behavior_sha256=_PROVIDER_SUBPROCESS_BEHAVIOR_SHA256,
        provider_subprocess_module_source_sha256=(_PROVIDER_SUBPROCESS_MODULE_SOURCE_SHA256),
        guest_rpc_policy_sha256=guest_rpc_policy_sha256(materials.guest.policy),
        guest_bootstrap_authorization_key_id=BOOTSTRAP_AUTHORIZATION_KEY_ID,
        guest_bootstrap_receipt_key_id=clinical_guest_bootstrap_receipt_key_id(BOOTSTRAP_RECEIPT_KEY),
        worker_attestation_key_id=firecracker_attestation_key_id(WORKER_KEY),
        gateway_receipt_key_id=gateway_session_key_id(GATEWAY_KEY),
        guest_rpc_receipt_key_id=guest_rpc_session_key_id(GUEST_KEY),
        production_receipt_key_id=clinical_production_run_key_id(PRODUCTION_KEY),
        canonical_launcher_id='lane-a-launcher',
        canonical_launcher_executable_sha256='6' * 64,
    )
    system_sha256 = clinical_production_system_identity_sha256(system)
    binding = ClinicalProductionTaskBinding(
        episode_id=materials.workspace.task.context.episode_id,
        target_trial_id=materials.workspace.task.context.target_trial_id,
        task_sha256=sha256(canonical_json_bytes(materials.workspace.task)).hexdigest(),
        task_context_sha256=materials.workspace.task.context_sha256,
        workspace_manifest_sha256=materials.workspace.manifest_sha256,
        workspace_tree_sha256=materials.workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=materials.workspace.manifest.model_visible_surface_sha256,
        authenticated_workspace_receipt_sha256=materials.workspace.authenticated_receipt_sha256,
    )
    reservation = ClinicalProductionReservation(
        registry_authority_id='runtime-boundary-test-authority',
        registered_entry_id='runtime-boundary-test-entry',
        cohort_id='runtime-boundary-test-cohort',
        cohort_manifest_sha256='a' * 64,
        evaluation_split=Split.TEST,
        system=system,
        system_identity_sha256=system_sha256,
        system_core_sha256=clinical_production_system_core_sha256(system),
        tasks=(binding,),
        reserved_at=now,
    )
    reservation_sha256 = clinical_production_reservation_sha256(reservation)
    launch = ClinicalProductionTaskLaunch(
        registry_authority_id=reservation.registry_authority_id,
        reservation_sha256=reservation_sha256,
        cohort_manifest_sha256=reservation.cohort_manifest_sha256,
        system_identity_sha256=system_sha256,
        episode_id=binding.episode_id,
        workspace_manifest_sha256=binding.workspace_manifest_sha256,
        run_id=RUN_ID,
        claimed_at=now,
    )
    config = FirecrackerClinicalRuntimeConfig(
        runtime_id='firecracker-clinical-runtime',
        runtime_version='test-v1',
        runtime_executable_sha256='7' * 64,
        bootstrap_authorization_key_id=BOOTSTRAP_AUTHORIZATION_KEY_ID,
        bootstrap_receipt_key_id=clinical_guest_bootstrap_receipt_key_id(BOOTSTRAP_RECEIPT_KEY),
        bootstrap_connection_timeout_seconds=1,
        bootstrap_validity_seconds=5,
        cleanup_grace_seconds=1,
    )
    deployment = CanonicalClinicalLauncherDeployment(
        registry_authority_id=reservation.registry_authority_id,
        canonical_launcher_id=system.canonical_launcher_id,
        canonical_launcher_executable_sha256=system.canonical_launcher_executable_sha256,
        expected_system_identity_sha256=system_sha256,
        runtime_id=config.runtime_id,
        runtime_version=config.runtime_version,
        runtime_executable_sha256=config.runtime_executable_sha256,
        runtime_config_sha256=firecracker_clinical_runtime_config_sha256(config),
        failure_receipt_key_id='8' * 64,
    )
    runners: list[_BoundaryBootstrapRunner] = []

    def runner_factory(**kwargs: Any) -> _BoundaryBootstrapRunner:
        runner = _BoundaryBootstrapRunner(
            events=events,
            journal=kwargs['journal_authenticated_bootstrap'],
            anchor=anchor,
            now=now,
            fail_before_ack=fail_before_ack,
            mutate_nonce=mutate_nonce,
        )
        runners.append(runner)
        return runner

    runtime = FirecrackerClinicalRuntime(
        config=config,
        supervisor=supervisor,
        gateway=cast(AuthenticatedProviderGateway, gateway),
        gateway_secret_store=store,
        execution_policy=materials.policy,
        gateway_route=materials.gateway.route,
        provider_subprocess_spec_sha256=_PROVIDER_SUBPROCESS_SPEC_SHA256,
        provider_subprocess_behavior_sha256=_PROVIDER_SUBPROCESS_BEHAVIOR_SHA256,
        provider_subprocess_module_source_sha256=(_PROVIDER_SUBPROCESS_MODULE_SOURCE_SHA256),
        guest_rpc_policy=materials.guest.policy,
        harness=materials.harness,
        keys=FirecrackerClinicalRuntimeKeys(
            workspace_receipt_key=WORKSPACE_KEY,
            worker_attestation_key=WORKER_KEY,
            gateway_receipt_key=GATEWAY_KEY,
            guest_rpc_receipt_key=GUEST_KEY,
            clinical_guest_bootstrap_receipt_key=BOOTSTRAP_RECEIPT_KEY,
            production_receipt_key=PRODUCTION_KEY,
        ),
        bootstrap_authorization_signer=BOOTSTRAP_SIGNER,
        bootstrap_trust_anchor=anchor,
        evidence_root=tmp_path / 'evidence',
        clock=lambda: now,
        monotonic_clock=lambda: 1.0,
        token_bytes=lambda count: b'Z' * count,
        token_hex=lambda count: 'c' * (2 * count),
        bootstrap_runner_factory=runner_factory,
        guest_session_factory=cast(
            Callable[..., GuestRpcHostSession],
            lambda **_kwargs: events.append('guest.create') or _CompletedGuestSession(materials, events),
        ),
        finalize_worker=lambda **_kwargs: events.append('worker.attest') or materials.worker,
    )
    request = ClinicalRuntimePrepareRequest(
        deployment=deployment,
        reservation=reservation,
        binding=binding,
        launch=launch,
        workspace=materials.workspace,
    )
    prepared = runtime.prepare(request)
    redemption = ClinicalProductionStartRedemption(
        registry_authority_id=reservation.registry_authority_id,
        reservation_sha256=reservation_sha256,
        launch_sha256=clinical_production_task_launch_sha256(launch),
        system_identity_sha256=system_sha256,
        episode_id=binding.episode_id,
        run_id=RUN_ID,
        canonical_launcher_id=system.canonical_launcher_id,
        canonical_launcher_executable_sha256=system.canonical_launcher_executable_sha256,
        prepared_worker_sha256=prepared.prepared_worker_sha256,
        guest_rpc_session_id=prepared.guest_rpc_session_id,
        gateway_capability_id=prepared.gateway_capability_id,
        redeemed_at=prepared.prepared_at,
    )
    start = ClinicalRuntimeStart(
        launcher_deployment_sha256=canonical_clinical_launcher_deployment_sha256(deployment),
        prepared_runtime_sha256=clinical_prepared_runtime_sha256(prepared),
        start_redemption=redemption,
        start_redemption_sha256=clinical_production_start_redemption_sha256(redemption),
    )
    bootstrap_artifacts: list[AuthenticatedClinicalGuestBootstrap] = []

    def fake_finalize(**kwargs: Any):
        events.append('run.finalize-v02')
        artifact = kwargs['clinical_guest_bootstrap']
        bootstrap_artifacts.append(artifact)
        output_root = kwargs['output_root']
        output_root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(root=output_root)

    def fake_load(root: Path, **_kwargs: Any):
        events.append('run.load-v02')
        assert len(bootstrap_artifacts) == 1
        return SimpleNamespace(
            root=root,
            clinical_guest_bootstrap_evidence_sha256=sha256(canonical_json_bytes(bootstrap_artifacts[0])).hexdigest(),
        )

    monkeypatch.setattr(production_v02, 'finalize_clinical_production_run_v02', fake_finalize)
    monkeypatch.setattr(production_v02, 'load_clinical_production_run_v02', fake_load)
    return _BoundaryCase(
        runtime=runtime,
        request=request,
        prepared=prepared,
        start=start,
        supervisor=supervisor,
        gateway=gateway,
        store=store,
        events=events,
        journal_path=runtime.bootstrap_journal_root / f'{RUN_ID}.json',
        bootstrap_artifacts=bootstrap_artifacts,
    )


def _restart_runtime(
    case: _BoundaryCase,
    *,
    require_global_startup_reconciliation: bool = False,
) -> FirecrackerClinicalRuntime:
    runtime = case.runtime
    return FirecrackerClinicalRuntime(
        config=runtime.config,
        supervisor=case.supervisor,
        gateway=cast(AuthenticatedProviderGateway, case.gateway),
        gateway_secret_store=case.store,
        execution_policy=runtime.execution_policy,
        gateway_route=runtime.gateway_route,
        provider_subprocess_spec_sha256=runtime.provider_subprocess_spec_sha256,
        provider_subprocess_behavior_sha256=runtime.provider_subprocess_behavior_sha256,
        provider_subprocess_module_source_sha256=(runtime.provider_subprocess_module_source_sha256),
        guest_rpc_policy=runtime.guest_rpc_policy,
        harness=runtime.harness,
        keys=runtime.keys,
        bootstrap_authorization_signer=BOOTSTRAP_SIGNER,
        bootstrap_trust_anchor=runtime.bootstrap_trust_anchor,
        evidence_root=runtime.evidence_root,
        clock=runtime._clock,
        monotonic_clock=runtime._monotonic,
        token_bytes=lambda count: b'R' * count,
        token_hex=lambda count: 'e' * (2 * count),
        require_global_startup_reconciliation=require_global_startup_reconciliation,
    )


def _alternate_request(case: _BoundaryCase) -> ClinicalRuntimePrepareRequest:
    launch = case.request.launch.model_copy(update={'run_id': 'e' * 32})
    return replace(case.request, launch=launch)


def test_runtime_success_has_strict_ordering_and_cross_checks_v02_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    assert case.events == ['worker.prepare']

    outcome = case.runtime.run(case.prepared, case.start)

    assert isinstance(outcome, ClinicalRuntimeCompleted)
    assert outcome.production_evidence_schema_version == production_v02.CLINICAL_PRODUCTION_RUN_V02_SCHEMA_VERSION
    assert (
        outcome.authenticated_bootstrap_sha256 == sha256(canonical_json_bytes(case.bootstrap_artifacts[0])).hexdigest()
    )
    assert case.supervisor.launch_count == 1
    assert len(case.gateway.grants) == 1
    assert case.gateway.grants[0].attempt_reservation_sha256 == case.start.start_redemption_sha256
    assert case.store.secrets == {}
    assert case.journal_path.exists()
    assert sha256(case.journal_path.read_bytes()).hexdigest() == outcome.authenticated_bootstrap_sha256
    assert case.events == [
        'worker.prepare',
        'secret.register',
        'gateway.register',
        'bootstrap.open',
        'worker.launch',
        'guest.create',
        'bootstrap.serve',
        'bootstrap.journal',
        'worker.wait',
        'bootstrap.close',
        'guest.seal',
        'gateway.seal:completed',
        'worker.cleanup',
        'worker.attest',
        'secret.revoke',
        'run.finalize-v02',
        'run.load-v02',
    ]


def test_restart_authenticates_journal_and_blocks_until_deployment_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    assert isinstance(case.runtime.run(case.prepared, case.start), ClinicalRuntimeCompleted)
    restarted = _restart_runtime(case)
    request = _alternate_request(case)

    assert restarted.startup_reconciliation_required
    with pytest.raises(FirecrackerClinicalRuntimeError, match='reconciliation is incomplete'):
        restarted.prepare(request)

    reconciler = _StartupReconciler()
    report = restarted.reconcile_startup(reconciler=reconciler)

    assert len(reconciler.calls) == 1
    assert report.request == reconciler.calls[0]
    assert len(report.request.retained_journals) == 1
    assert report.cleanup_receipt.retained_journal_count == 1
    assert report.startup_admission_allowed
    assert report.retained_bootstrap_journals_preserved
    assert report.deployment_cleanup_adapter_invoked
    assert not report.repository_inferred_live_process_or_capability_state
    assert not report.deployment_receipts_are_independent_host_attestation
    assert not report.cleanup_receipt_cryptographically_authenticated
    assert not report.cleanup_adapter_identity_pinned_by_runtime_config
    assert not report.pre_start_journal_deletion_excluded_by_repository
    assert not report.reconciliation_report_persisted_by_runtime
    assert not report.linux_kvm_cleanup_qualified
    assert case.journal_path.exists()
    assert not restarted.startup_reconciliation_required
    assert restarted.prepare(request).run_id == request.launch.run_id


def test_recovery_only_startup_scan_authenticates_journal_without_launch_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    assert isinstance(
        case.runtime.run(case.prepared, case.start),
        ClinicalRuntimeCompleted,
    )
    events_before_recovery = tuple(case.events)
    launches_before_recovery = case.supervisor.launch_count
    reconciler = _StartupReconciler()

    report = reconcile_firecracker_clinical_startup_without_execution(
        config=case.runtime.config,
        execution_policy_sha256=agentic_policy_sha256(case.runtime.execution_policy),
        worker_spec=case.supervisor.spec,
        gateway_policy_sha256=authenticated_gateway_policy_sha256(case.runtime.gateway.policy),
        gateway_route_sha256=gateway_model_route_sha256(case.runtime.gateway_route),
        guest_rpc_policy=case.runtime.guest_rpc_policy,
        bootstrap_receipt_key=(case.runtime.keys.clinical_guest_bootstrap_receipt_key),
        bootstrap_trust_anchor=case.runtime.bootstrap_trust_anchor,
        evidence_root=case.runtime.evidence_root,
        reconciler=reconciler,
        clock=case.runtime._clock,
    )

    assert report.request == reconciler.calls[0]
    assert len(report.request.retained_journals) == 1
    assert case.journal_path.exists()
    assert case.supervisor.launch_count == launches_before_recovery
    assert tuple(case.events) == events_before_recovery


def test_recovery_only_startup_scan_fails_closed_on_wrong_static_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    reconciler = _StartupReconciler()

    with pytest.raises(FirecrackerClinicalRuntimeError, match='trust anchor'):
        reconcile_firecracker_clinical_startup_without_execution(
            config=case.runtime.config,
            execution_policy_sha256='f' * 64,
            worker_spec=case.supervisor.spec,
            gateway_policy_sha256=authenticated_gateway_policy_sha256(case.runtime.gateway.policy),
            gateway_route_sha256=gateway_model_route_sha256(case.runtime.gateway_route),
            guest_rpc_policy=case.runtime.guest_rpc_policy,
            bootstrap_receipt_key=(case.runtime.keys.clinical_guest_bootstrap_receipt_key),
            bootstrap_trust_anchor=case.runtime.bootstrap_trust_anchor,
            evidence_root=case.runtime.evidence_root,
            reconciler=reconciler,
        )

    assert reconciler.calls == []


def test_deployment_mode_requires_global_scan_without_a_retained_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch, fail_before_ack=True)
    assert case.runtime.run(case.prepared, case.start) == ClinicalRuntimeFailed(
        ClinicalRuntimeFailureCode.BOOTSTRAP_FAILED
    )
    assert not case.journal_path.exists()
    restarted = _restart_runtime(case, require_global_startup_reconciliation=True)
    request = _alternate_request(case)

    with pytest.raises(FirecrackerClinicalRuntimeError, match='reconciliation is incomplete'):
        restarted.prepare(request)

    reconciler = _StartupReconciler()
    report = restarted.reconcile_startup(reconciler=reconciler)

    assert len(report.request.retained_journals) == 0
    assert report.cleanup_receipt.unjournaled_orphan_discovery_complete
    assert report.cleanup_receipt.worker_discovery_complete
    assert report.cleanup_receipt.capability_discovery_complete
    assert restarted.prepare(request).run_id == request.launch.run_id


@pytest.mark.parametrize('mode', ('adapter_failure', 'receipt_mismatch'))
def test_incomplete_startup_cleanup_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    assert isinstance(case.runtime.run(case.prepared, case.start), ClinicalRuntimeCompleted)
    restarted = _restart_runtime(case)
    reconciler = _StartupReconciler(
        fails=mode == 'adapter_failure',
        mismatch=mode == 'receipt_mismatch',
    )

    with pytest.raises(FirecrackerClinicalRuntimeError, match='could not be completely reconciled'):
        restarted.reconcile_startup(reconciler=reconciler)

    assert restarted.startup_reconciliation_required
    assert case.journal_path.exists()
    with pytest.raises(FirecrackerClinicalRuntimeError, match='reconciliation is incomplete'):
        restarted.prepare(_alternate_request(case))


def test_restart_rejects_tampered_or_noncanonical_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    assert isinstance(case.runtime.run(case.prepared, case.start), ClinicalRuntimeCompleted)
    case.journal_path.write_bytes(case.journal_path.read_bytes() + b'\n')
    case.journal_path.chmod(0o600)

    with pytest.raises(FirecrackerClinicalRuntimeError, match='not canonical JSON'):
        _restart_runtime(case)


def test_restart_rejects_canonical_journal_with_invalid_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    assert isinstance(case.runtime.run(case.prepared, case.start), ClinicalRuntimeCompleted)
    artifact = AuthenticatedClinicalGuestBootstrap.model_validate_json(case.journal_path.read_bytes())
    forged = artifact.model_copy(update={'receipt_hmac_sha256': '0' * 64})
    case.journal_path.write_bytes(canonical_json_bytes(forged))
    case.journal_path.chmod(0o600)

    with pytest.raises(FirecrackerClinicalRuntimeError, match='authentication failed'):
        _restart_runtime(case)


def test_restart_rejects_unexpected_journal_inventory_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    assert isinstance(case.runtime.run(case.prepared, case.start), ClinicalRuntimeCompleted)
    unexpected = case.runtime.bootstrap_journal_root / 'leftover.tmp'
    unexpected.write_bytes(b'incomplete journal staging bytes')
    unexpected.chmod(0o600)

    with pytest.raises(FirecrackerClinicalRuntimeError, match='unexpected entry'):
        _restart_runtime(case)


def test_bootstrap_failure_is_terminal_and_same_preparation_cannot_launch_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch, fail_before_ack=True)

    first = case.runtime.run(case.prepared, case.start)
    second = case.runtime.run(case.prepared, case.start)

    assert first == ClinicalRuntimeFailed(ClinicalRuntimeFailureCode.BOOTSTRAP_FAILED)
    assert second == ClinicalRuntimeFailed(ClinicalRuntimeFailureCode.LAUNCH_FAILED)
    assert case.supervisor.launch_count == 1
    assert not case.journal_path.exists()
    assert 'run.finalize-v02' not in case.events
    assert case.store.secrets == {}


def test_bootstrap_journal_rejects_a_valid_signature_for_a_different_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch, mutate_nonce=True)

    outcome = case.runtime.run(case.prepared, case.start)

    assert outcome == ClinicalRuntimeFailed(ClinicalRuntimeFailureCode.BOOTSTRAP_FAILED)
    assert not case.journal_path.exists()
    assert case.supervisor.launch_count == 1


@pytest.mark.parametrize(
    'field',
    ('guest_bootstrap_authorization_key_id', 'guest_bootstrap_receipt_key_id'),
)
def test_prepare_rejects_reserved_bootstrap_key_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    bad_system = case.request.reservation.system.model_copy(update={field: '0' * 64})
    bad_reservation = case.request.reservation.model_copy(update={'system': bad_system})
    bad_request = replace(case.request, reservation=bad_reservation)

    with pytest.raises(FirecrackerClinicalRuntimeError, match='composition differs'):
        case.runtime.prepare(bad_request)

    assert case.events == ['worker.prepare']


@pytest.mark.parametrize(
    'field',
    (
        'provider_subprocess_spec_sha256',
        'provider_subprocess_behavior_sha256',
        'provider_subprocess_module_source_sha256',
    ),
)
def test_prepare_rejects_reserved_provider_child_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    bad_system = case.request.reservation.system.model_copy(update={field: '0' * 64})
    bad_reservation = case.request.reservation.model_copy(update={'system': bad_system})
    bad_request = replace(case.request, reservation=bad_reservation)

    with pytest.raises(FirecrackerClinicalRuntimeError, match='composition differs'):
        case.runtime.prepare(bad_request)

    assert case.events == ['worker.prepare']


def test_cleanup_failure_after_ack_retains_exact_private_bootstrap_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch, cleanup_fails=True)

    outcome = case.runtime.run(case.prepared, case.start)

    assert isinstance(outcome, ClinicalRuntimeFailed)
    assert outcome.code == ClinicalRuntimeFailureCode.CLEANUP_FAILED
    journal_bytes = case.journal_path.read_bytes()
    retained = AuthenticatedClinicalGuestBootstrap.model_validate_json(journal_bytes)
    assert canonical_json_bytes(retained) == journal_bytes
    assert sha256(journal_bytes).hexdigest() == outcome.authenticated_bootstrap_sha256
    assert case.journal_path.stat().st_mode & 0o777 == 0o600
    assert 'run.finalize-v02' not in case.events
    assert case.store.secrets == {}


def test_private_evidence_root_rejects_foreign_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'foreign-owner'
    root.mkdir(mode=0o700)
    actual_uid = runtime_module.os.geteuid()
    monkeypatch.setattr(runtime_module.os, 'geteuid', lambda: actual_uid + 1)

    with pytest.raises(ValueError, match='current-user-owned'):
        runtime_module._prepare_private_root(root)
