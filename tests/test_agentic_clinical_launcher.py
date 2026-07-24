from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_agentic_clinical_production_registry import (
    _loaded_evidence,
    _redeem_start,
    _reservation,
)
from vaxreplay.agentic.clinical_launcher import (
    AuthenticatedClinicalLauncherFailure,
    CanonicalClinicalLauncher,
    CanonicalClinicalLauncherDeployment,
    CanonicalClinicalRecoveryTerminalizer,
    ClinicalLauncherError,
    ClinicalLauncherFailure,
    ClinicalLauncherFailureCode,
    ClinicalLauncherSuccess,
    ClinicalPreparedRuntime,
    ClinicalRuntimeCompleted,
    ClinicalRuntimeFailed,
    ClinicalRuntimeFailureCode,
    ClinicalRuntimePrepareRequest,
    ClinicalRuntimeStart,
    canonical_clinical_launcher_deployment_sha256,
    clinical_launcher_failure_key_id,
    verify_authenticated_clinical_launcher_failure,
)
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionRegistryError,
    ClinicalProductionTerminalCode,
    clinical_production_start_redemption_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.managed_clinical_registry import (
    ManagedClinicalRegistryClient,
    ManagedClinicalRegistryConfig,
    managed_clinical_registry_config_sha256,
)

FAILURE_KEY = b'canonical-clinical-launcher-failure-key-01'


def _deployment(registry, context) -> CanonicalClinicalLauncherDeployment:
    system = context.reservation.system
    return CanonicalClinicalLauncherDeployment(
        registry_authority_id=registry.authority_id,
        canonical_launcher_id=system.canonical_launcher_id,
        canonical_launcher_executable_sha256=system.canonical_launcher_executable_sha256,
        expected_system_identity_sha256=context.reservation.system_identity_sha256,
        runtime_id='clinical-test-runtime',
        runtime_version='1',
        runtime_executable_sha256='a' * 64,
        runtime_config_sha256='b' * 64,
        failure_receipt_key_id=clinical_launcher_failure_key_id(FAILURE_KEY),
    )


class FakeClinicalRuntime:
    def __init__(
        self,
        *,
        tmp_path: Path,
        system,
        workspace,
        worker,
        gateway,
        session,
        outcome_code: ClinicalRuntimeFailureCode | None = None,
        raise_prepare: bool = False,
        raise_run: bool = False,
        prepared_updates: dict[str, object] | None = None,
        discard_raises: bool = False,
        prepare_barrier: threading.Barrier | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.system = system
        self.workspace = workspace
        self.worker = worker
        self.gateway = gateway
        self.session = session
        self.outcome_code = outcome_code
        self.raise_prepare = raise_prepare
        self.raise_run = raise_run
        self.prepared_updates = prepared_updates or {}
        self.discard_raises = discard_raises
        self.prepare_barrier = prepare_barrier
        self.events: list[str] = []
        self.loaded = None
        self.last_start: ClinicalRuntimeStart | None = None

    def prepare(self, request: ClinicalRuntimePrepareRequest) -> ClinicalPreparedRuntime:
        self.events.append('prepare')
        if self.prepare_barrier is not None:
            self.prepare_barrier.wait(timeout=5)
        if self.raise_prepare:
            raise RuntimeError('private preparation details must not escape')
        binding = request.binding
        deployment = request.deployment
        prepared = ClinicalPreparedRuntime(
            runtime_id=deployment.runtime_id,
            runtime_version=deployment.runtime_version,
            runtime_executable_sha256=deployment.runtime_executable_sha256,
            runtime_config_sha256=deployment.runtime_config_sha256,
            launcher_deployment_sha256=canonical_clinical_launcher_deployment_sha256(deployment),
            reservation_sha256=request.launch.reservation_sha256,
            launch_sha256=clinical_production_task_launch_sha256(request.launch),
            system_identity_sha256=request.reservation.system_identity_sha256,
            episode_id=binding.episode_id,
            run_id=request.launch.run_id,
            workspace_manifest_sha256=binding.workspace_manifest_sha256,
            workspace_tree_sha256=binding.workspace_tree_sha256,
            model_visible_surface_sha256=binding.model_visible_surface_sha256,
            worker_spec_sha256=request.reservation.system.worker_spec_sha256,
            harness_sha256=request.reservation.system.harness.harness_image_or_commitment.removeprefix('sha256:'),
            prepared_worker_sha256=self.worker.attestation.prepared_worker_sha256,
            guest_rpc_session_id=self.session.seal.session_id,
            gateway_capability_id=self.gateway.grant.capability_id,
            prepared_at=request.launch.claimed_at,
        )
        return prepared.model_copy(update=self.prepared_updates)

    def discard_prepared(self, prepared: ClinicalPreparedRuntime) -> None:
        del prepared
        self.events.append('discard')
        if self.discard_raises:
            raise RuntimeError('private discard failure')

    def run(self, prepared: ClinicalPreparedRuntime, start: ClinicalRuntimeStart):
        del prepared
        self.events.append('run')
        self.last_start = start
        if self.raise_run:
            raise RuntimeError('private runtime details must not escape')
        if self.outcome_code is not None:
            return ClinicalRuntimeFailed(self.outcome_code)
        launch = start.start_redemption
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.loaded = _loaded_evidence(
            self.tmp_path,
            system=self.system,
            workspace=self.workspace,
            launch=type('LaunchView', (), {'run_id': launch.run_id})(),
            worker=self.worker,
            gateway=self.gateway,
            clinical_session=self.session,
            start_redemption=start.start_redemption,
        )
        return ClinicalRuntimeCompleted(
            production_run_root=self.loaded.root,
            production_evidence_schema_version='vaxreplay.authenticated-clinical-production-run.dev-v0.2',
            authenticated_bootstrap_sha256=self.loaded.clinical_guest_bootstrap_evidence_sha256,
        )


def _launcher_materials(tmp_path: Path, **runtime_updates):
    registry, context, manifest, workspace, system, worker, gateway, session = _reservation(tmp_path)
    deployment = _deployment(registry, context)
    runtime = FakeClinicalRuntime(
        tmp_path=tmp_path,
        system=system,
        workspace=workspace,
        worker=worker,
        gateway=gateway,
        session=session,
        **runtime_updates,
    )
    loader_events: list[tuple[Path, str]] = []

    def loader(root: Path, attempt_sha256: str):
        loader_events.append((root, attempt_sha256))
        if runtime.loaded is None or runtime.loaded.root != root:
            raise ValueError('missing independently authenticated evidence')
        if runtime.loaded.receipt.attempt_reservation_sha256 != attempt_sha256:
            raise ValueError('evidence belongs to a different start redemption')
        return runtime.loaded

    launcher = CanonicalClinicalLauncher(
        registry=registry,
        deployment=deployment,
        runtime=runtime,
        evidence_loader=loader,
        failure_receipt_key=FAILURE_KEY,
        clock=lambda: datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: 'd' * 32,
    )
    return (
        launcher,
        runtime,
        loader_events,
        registry,
        context,
        manifest,
        workspace,
        system,
        worker,
        gateway,
        session,
    )


def _execute(launcher, context, workspace):
    return launcher.execute_reserved_task(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        workspace=workspace,
    )


def _verify_failure(result: ClinicalLauncherFailure, code: ClinicalLauncherFailureCode) -> None:
    assert result.failure_code == code
    artifact = result.authenticated_failure
    assert isinstance(artifact, AuthenticatedClinicalLauncherFailure)
    receipt = verify_authenticated_clinical_launcher_failure(
        artifact,
        key=FAILURE_KEY,
        expected_key_id=clinical_launcher_failure_key_id(FAILURE_KEY),
    )
    assert receipt.failure_code == code
    assert receipt.attempt_consumed
    assert not receipt.retry_permitted
    assert not receipt.linux_kvm_runtime_qualified
    assert not receipt.official_execution_qualified


def test_launcher_composes_exact_order_and_returns_independently_reloaded_success(tmp_path: Path) -> None:
    launcher, runtime, loader_events, registry, context, _, workspace, *_ = _launcher_materials(tmp_path)

    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherSuccess)
    assert result.record.state == 'succeeded'
    assert result.loaded_run is runtime.loaded
    assert runtime.events == ['prepare', 'run']
    assert runtime.last_start is not None
    assert runtime.last_start.start_redemption_sha256 == clinical_production_start_redemption_sha256(
        result.start_redemption
    )
    assert loader_events == [(result.loaded_run.root, runtime.last_start.start_redemption_sha256)]
    retained = registry.task_records(context.reservation_sha256)[0]
    assert retained.state == 'succeeded'
    assert retained.start_redemption == result.start_redemption


def test_launcher_success_through_managed_client_reloads_after_service_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, runtime, loader_events, registry, context, _, workspace, *_ = _launcher_materials(tmp_path)
    system = context.reservation.system
    config = ManagedClinicalRegistryConfig(
        service_id='managed-launcher-regression',
        service_version='test-v1',
        registry_authority_id=registry.authority_id,
        database_path=str(tmp_path / 'managed' / 'attempts.sqlite3'),
        socket_path='/tmp/vrk-managed-launcher-regression.sock',
        production_evidence_root=str(tmp_path / 'managed-evidence'),
        protocol_audit_root=str(tmp_path / 'managed-audit'),
        canonical_launcher_id=system.canonical_launcher_id,
        canonical_launcher_executable_sha256=system.canonical_launcher_executable_sha256,
        launcher_process_executable_sha256='1' * 64,
        service_process_executable_sha256='2' * 64,
        startup_config_sha256='3' * 64,
        startup_cleanup_receipt_key_id='4' * 64,
        connection_timeout_seconds=1,
    )
    managed = ManagedClinicalRegistryClient(
        config,
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
    )
    service_reloads: list[tuple[Path, str]] = []

    def service_reload(root: Path, attempt_sha256: str):  # noqa: ANN202
        service_reloads.append((root, attempt_sha256))
        assert runtime.loaded is not None
        assert runtime.loaded.root == root
        assert runtime.loaded.receipt.attempt_reservation_sha256 == attempt_sha256
        return runtime.loaded

    def dispatch(operation, payload):  # noqa: ANN001, ANN202
        if operation == 'status':
            reservation = registry.reservation_context(payload.reservation_sha256)
            return {
                'reservation': reservation.reservation.model_dump(mode='json'),
                'reservation_sha256': reservation.reservation_sha256,
                'task_records': [
                    item.model_dump(mode='json') for item in registry.task_records(payload.reservation_sha256)
                ],
            }
        if operation == 'claim':
            result = registry.claim_task_launch(**payload.model_dump(mode='python'))
            return {'launch': result.model_dump(mode='json')}
        if operation == 'redeem':
            result = registry.redeem_task_start(
                **payload.model_dump(mode='python'),
                canonical_launcher_id=config.canonical_launcher_id,
                canonical_launcher_executable_sha256=(config.canonical_launcher_executable_sha256),
            )
            return {'start_redemption': result.model_dump(mode='json')}
        if operation == 'record_run':
            result = registry.record_production_run(
                reservation_sha256=payload.reservation_sha256,
                episode_id=payload.episode_id,
                production_run_root=Path(payload.production_run_root),
                reauthenticate=service_reload,
                terminal_at=payload.terminal_at,
            )
            return {'task_record': result.model_dump(mode='json')}
        raise AssertionError(operation)

    monkeypatch.setattr(managed, '_call', dispatch)
    managed_launcher = CanonicalClinicalLauncher(
        registry=managed,
        deployment=launcher.deployment,
        runtime=runtime,
        evidence_loader=launcher.evidence_loader,
        failure_receipt_key=FAILURE_KEY,
        clock=lambda: datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: 'd' * 32,
    )

    result = _execute(managed_launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherSuccess)
    assert result.record.state == 'succeeded'
    assert runtime.last_start is not None
    expected_reload = (result.loaded_run.root, runtime.last_start.start_redemption_sha256)
    assert service_reloads == [expected_reload]
    assert loader_events == [expected_reload]


def test_prepare_failure_is_permanent_and_never_retried(tmp_path: Path) -> None:
    launcher, runtime, _, registry, context, _, workspace, *_ = _launcher_materials(tmp_path, raise_prepare=True)

    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    _verify_failure(result, ClinicalLauncherFailureCode.PREPARE_FAILED)
    assert result.record.terminal_code == ClinicalProductionTerminalCode.WORKER_LAUNCH_FAILURE
    assert result.record.start_redemption is None
    assert runtime.events == ['prepare']
    with pytest.raises(ClinicalProductionRegistryError, match='not open|one launch'):
        _execute(launcher, context, workspace)
    assert runtime.events == ['prepare']
    assert registry.task_records(context.reservation_sha256)[0].state == 'failed'


@pytest.mark.parametrize('discard_raises', [False, True])
def test_prepared_binding_mismatch_is_discarded_and_terminalized(
    tmp_path: Path,
    discard_raises: bool,
) -> None:
    launcher, runtime, _, _, context, _, workspace, *_ = _launcher_materials(
        tmp_path,
        prepared_updates={'workspace_manifest_sha256': '0' * 64},
        discard_raises=discard_raises,
    )

    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    _verify_failure(result, ClinicalLauncherFailureCode.PREPARED_BINDING_MISMATCH)
    assert result.authenticated_failure is not None
    assert result.authenticated_failure.receipt.prepared_runtime_discarded is (not discard_raises)
    assert result.record.start_redemption is None
    assert runtime.events == ['prepare', 'discard']


def test_redemption_failure_discards_and_permanently_records_scheduler_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, runtime, _, registry, context, _, workspace, *_ = _launcher_materials(tmp_path)

    def reject_redemption(**_kwargs):
        raise ClinicalProductionRegistryError('injected start gate failure')

    monkeypatch.setattr(registry, 'redeem_task_start', reject_redemption)
    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    _verify_failure(result, ClinicalLauncherFailureCode.START_REDEMPTION_FAILED)
    assert result.record.terminal_code == ClinicalProductionTerminalCode.SCHEDULER_FAILURE
    assert result.record.start_redemption is None
    assert runtime.events == ['prepare', 'discard']


@pytest.mark.parametrize(
    ('runtime_code', 'launcher_code', 'terminal_code'),
    [
        (
            ClinicalRuntimeFailureCode.LAUNCH_FAILED,
            ClinicalLauncherFailureCode.RUNTIME_LAUNCH_FAILED,
            ClinicalProductionTerminalCode.WORKER_LAUNCH_FAILURE,
        ),
        (
            ClinicalRuntimeFailureCode.BOOTSTRAP_FAILED,
            ClinicalLauncherFailureCode.RUNTIME_BOOTSTRAP_FAILED,
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
        ),
        (
            ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE,
            ClinicalLauncherFailureCode.RUNTIME_TERMINAL_FAILURE,
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
        ),
        (
            ClinicalRuntimeFailureCode.WORKER_LOST,
            ClinicalLauncherFailureCode.RUNTIME_WORKER_LOST,
            ClinicalProductionTerminalCode.WORKER_LOST,
        ),
        (
            ClinicalRuntimeFailureCode.CLEANUP_FAILED,
            ClinicalLauncherFailureCode.RUNTIME_CLEANUP_FAILED,
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
        ),
        (
            ClinicalRuntimeFailureCode.EVIDENCE_FINALIZATION_FAILED,
            ClinicalLauncherFailureCode.EVIDENCE_FINALIZATION_FAILED,
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
        ),
    ],
)
def test_every_typed_post_redemption_runtime_failure_is_permanent(
    tmp_path: Path,
    runtime_code: ClinicalRuntimeFailureCode,
    launcher_code: ClinicalLauncherFailureCode,
    terminal_code: ClinicalProductionTerminalCode,
) -> None:
    launcher, runtime, loader_events, registry, context, _, workspace, *_ = _launcher_materials(
        tmp_path, outcome_code=runtime_code
    )

    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    _verify_failure(result, launcher_code)
    assert result.record.terminal_code == terminal_code
    assert result.start_redemption is not None
    assert result.record.start_redemption == result.start_redemption
    assert runtime.events == ['prepare', 'run']
    assert loader_events == []
    assert registry.task_records(context.reservation_sha256)[0].state == 'failed'
    with pytest.raises(ClinicalProductionRegistryError, match='not awaiting|already redeemed'):
        registry.redeem_task_start(
            reservation_sha256=context.reservation_sha256,
            episode_id=workspace.task.context.episode_id,
            launch_sha256=clinical_production_task_launch_sha256(result.launch),
            canonical_launcher_id=context.reservation.system.canonical_launcher_id,
            canonical_launcher_executable_sha256=(context.reservation.system.canonical_launcher_executable_sha256),
            prepared_worker_sha256='1' * 64,
            guest_rpc_session_id='2' * 32,
            gateway_capability_id='3' * 64,
            redeemed_at=result.launch.claimed_at,
        )


def test_cleanup_failure_after_nominal_work_cannot_preserve_success(tmp_path: Path) -> None:
    launcher, runtime, loader_events, _, context, _, workspace, *_ = _launcher_materials(
        tmp_path, outcome_code=ClinicalRuntimeFailureCode.CLEANUP_FAILED
    )

    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    _verify_failure(result, ClinicalLauncherFailureCode.RUNTIME_CLEANUP_FAILED)
    assert result.record.terminal_code == ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE
    assert result.record.submission_sha256 is None
    assert loader_events == []
    assert runtime.events == ['prepare', 'run']


def test_unexpected_runtime_exception_after_redemption_is_worker_lost(tmp_path: Path) -> None:
    launcher, runtime, _, _, context, _, workspace, *_ = _launcher_materials(tmp_path, raise_run=True)

    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    _verify_failure(result, ClinicalLauncherFailureCode.RUNTIME_INTERNAL_FAILURE)
    assert result.record.terminal_code == ClinicalProductionTerminalCode.WORKER_LOST
    assert result.start_redemption is not None
    assert runtime.events == ['prepare', 'run']


def test_runtime_success_with_wrong_evidence_schema_is_evidence_authentication_failure(
    tmp_path: Path,
) -> None:
    launcher, runtime, _, _, context, _, workspace, *_ = _launcher_materials(tmp_path)
    original_run = runtime.run

    def wrong_schema(prepared: ClinicalPreparedRuntime, start: ClinicalRuntimeStart):
        outcome = original_run(prepared, start)
        assert isinstance(outcome, ClinicalRuntimeCompleted)
        return ClinicalRuntimeCompleted(
            production_run_root=outcome.production_run_root,
            production_evidence_schema_version='vaxreplay.wrong-evidence.dev-v9',  # ty: ignore[invalid-argument-type]
            authenticated_bootstrap_sha256=outcome.authenticated_bootstrap_sha256,
        )

    runtime.run = wrong_schema  # type: ignore[method-assign]
    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    _verify_failure(result, ClinicalLauncherFailureCode.EVIDENCE_AUTHENTICATION_FAILED)
    assert result.record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED


def test_evidence_loader_cannot_spoof_v02_with_matching_attribute_names(tmp_path: Path) -> None:
    launcher, runtime, _, registry, context, _, workspace, *_ = _launcher_materials(tmp_path)

    def spoofed_loader(_root: Path, _attempt_sha256: str):
        return SimpleNamespace(
            authenticated_outer_receipt=SimpleNamespace(
                schema_version='vaxreplay.authenticated-clinical-production-run.dev-v0.2'
            ),
            clinical_guest_bootstrap_evidence_sha256='e' * 64,
        )

    launcher = CanonicalClinicalLauncher(
        registry=registry,
        deployment=launcher.deployment,
        runtime=runtime,
        evidence_loader=spoofed_loader,
        failure_receipt_key=FAILURE_KEY,
        clock=lambda: datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: 'd' * 32,
    )
    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    assert result.failure_code == ClinicalLauncherFailureCode.EVIDENCE_AUTHENTICATION_FAILED
    assert result.record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED


def test_evidence_loader_failure_is_terminal_and_cannot_preserve_submission(tmp_path: Path) -> None:
    (
        launcher,
        runtime,
        _,
        registry,
        context,
        _,
        workspace,
        *_,
    ) = _launcher_materials(tmp_path)

    def reject_evidence(_root: Path, _attempt_sha256: str):
        raise ValueError('forged or unreadable authenticated package')

    launcher = CanonicalClinicalLauncher(
        registry=registry,
        deployment=launcher.deployment,
        runtime=runtime,
        evidence_loader=reject_evidence,
        failure_receipt_key=FAILURE_KEY,
        clock=lambda: datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: 'd' * 32,
    )
    result = _execute(launcher, context, workspace)

    assert isinstance(result, ClinicalLauncherFailure)
    assert result.failure_code == ClinicalLauncherFailureCode.EVIDENCE_AUTHENTICATION_FAILED
    assert result.authenticated_failure is None
    assert result.record.terminal_code == ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED
    assert result.record.submission_sha256 is None
    assert registry.task_records(context.reservation_sha256)[0].state == 'failed'


def test_deployment_mismatch_fails_before_claim_or_runtime(tmp_path: Path) -> None:
    launcher, runtime, _, registry, context, _, workspace, *_ = _launcher_materials(tmp_path)
    wrong = launcher.deployment.model_copy(update={'canonical_launcher_executable_sha256': '0' * 64})
    launcher = CanonicalClinicalLauncher(
        registry=registry,
        deployment=wrong,
        runtime=runtime,
        evidence_loader=launcher.evidence_loader,
        failure_receipt_key=FAILURE_KEY,
    )

    with pytest.raises(ClinicalLauncherError, match='does not match'):
        _execute(launcher, context, workspace)

    assert runtime.events == []
    assert registry.task_records(context.reservation_sha256)[0].state == 'reserved'


def test_concurrent_launchers_have_one_owner_and_loser_never_terminalizes_winner(tmp_path: Path) -> None:
    materials = _launcher_materials(tmp_path)
    first, first_runtime, _, registry, context, _, workspace, system, worker, gateway, session = materials
    second_runtime = FakeClinicalRuntime(
        tmp_path=tmp_path / 'second',
        system=system,
        workspace=workspace,
        worker=worker,
        gateway=gateway,
        session=session,
    )

    def second_loader(root: Path, attempt_sha256: str):
        if second_runtime.loaded is None or second_runtime.loaded.root != root:
            raise ValueError('missing second runtime evidence')
        assert second_runtime.loaded.receipt.attempt_reservation_sha256 == attempt_sha256
        return second_runtime.loaded

    second = CanonicalClinicalLauncher(
        registry=registry,
        deployment=first.deployment,
        runtime=second_runtime,
        evidence_loader=second_loader,
        failure_receipt_key=FAILURE_KEY,
        clock=lambda: datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: 'e' * 32,
    )
    barrier = threading.Barrier(2)

    def execute(candidate: CanonicalClinicalLauncher):
        barrier.wait(timeout=5)
        try:
            return _execute(candidate, context, workspace)
        except ClinicalProductionRegistryError:
            return 'claim-rejected'

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(execute, (first, second)))

    assert sum(isinstance(item, ClinicalLauncherSuccess) for item in outcomes) == 1
    assert outcomes.count('claim-rejected') == 1
    assert first_runtime.events.count('prepare') + second_runtime.events.count('prepare') == 1
    retained = registry.task_records(context.reservation_sha256)[0]
    assert retained.state == 'succeeded'
    assert retained.terminal_code == ClinicalProductionTerminalCode.SUCCESS


def test_reconcile_terminalizes_unredeemed_claim_without_running(tmp_path: Path) -> None:
    launcher, runtime, _, registry, context, _, workspace, *_ = _launcher_materials(tmp_path)
    registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='1' * 32,
        claimed_at=datetime.now(UTC),
    )

    failures = launcher.reconcile_consumed_tasks(reservation_sha256=context.reservation_sha256)

    assert len(failures) == 1
    _verify_failure(failures[0], ClinicalLauncherFailureCode.RECONCILED_UNREDEEMED_CLAIM)
    assert failures[0].record.terminal_code == ClinicalProductionTerminalCode.SCHEDULER_FAILURE
    assert runtime.events == []
    assert launcher.reconcile_consumed_tasks(reservation_sha256=context.reservation_sha256) == ()


def test_recovery_terminalizer_has_no_runtime_and_is_idempotent(tmp_path: Path) -> None:
    launcher, runtime, _, registry, context, _, workspace, *_ = _launcher_materials(tmp_path)
    registry.claim_task_launch(
        reservation_sha256=context.reservation_sha256,
        episode_id=workspace.task.context.episode_id,
        run_id='3' * 32,
        claimed_at=datetime.now(UTC),
    )
    terminalizer = CanonicalClinicalRecoveryTerminalizer(
        registry=registry,
        deployment=launcher.deployment,
        failure_receipt_key=FAILURE_KEY,
    )

    assert not hasattr(terminalizer, 'execute_reserved_task')
    assert not hasattr(terminalizer, 'runtime')
    failures = terminalizer.reconcile_consumed_tasks(reservation_sha256=context.reservation_sha256)

    assert len(failures) == 1
    _verify_failure(
        failures[0],
        ClinicalLauncherFailureCode.RECONCILED_UNREDEEMED_CLAIM,
    )
    assert runtime.events == []
    assert terminalizer.reconcile_consumed_tasks(reservation_sha256=context.reservation_sha256) == ()


def test_reconcile_terminalizes_redeemed_start_without_relaunching(tmp_path: Path) -> None:
    launcher, runtime, _, registry, context, _, workspace, system, worker, gateway, session = _launcher_materials(
        tmp_path
    )
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

    failures = launcher.reconcile_consumed_tasks(reservation_sha256=context.reservation_sha256)

    assert len(failures) == 1
    _verify_failure(failures[0], ClinicalLauncherFailureCode.RECONCILED_REDEEMED_START)
    assert failures[0].record.terminal_code == ClinicalProductionTerminalCode.WORKER_LOST
    assert failures[0].start_redemption == redemption
    assert runtime.events == []


def test_reconcile_ignores_reserved_unclaimed_tasks(tmp_path: Path) -> None:
    launcher, runtime, _, registry, context, _, _, *_ = _launcher_materials(tmp_path)

    failures = launcher.reconcile_consumed_tasks(reservation_sha256=context.reservation_sha256)

    assert failures == ()
    assert runtime.events == []
    assert registry.task_records(context.reservation_sha256)[0].state == 'reserved'
