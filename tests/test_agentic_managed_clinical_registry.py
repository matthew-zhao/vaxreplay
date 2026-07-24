from __future__ import annotations

import hashlib
import os
import shutil
import socket
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

import vaxreplay.agentic.managed_clinical_registry as managed_registry_module
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionStartRedemption,
    ClinicalProductionTaskLaunch,
    ClinicalProductionTaskRecord,
    ClinicalProductionTerminalCode,
    SqliteClinicalProductionRegistry,
    clinical_production_start_redemption_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalStartupReconciliationRequest,
)
from vaxreplay.agentic.managed_clinical_registry import (
    AuthenticatedManagedClinicalRegistryAudit,
    ManagedBeginReconciliationRequest,
    ManagedClinicalPeerIdentity,
    ManagedClinicalRegistryAuditServerIdentity,
    ManagedClinicalRegistryConfig,
    ManagedClinicalRegistryRequest,
    ManagedClinicalRegistryResponse,
    ManagedClinicalRegistryService,
    ManagedFinishReconciliationRequest,
    ManagedRecordRunRequest,
    ManagedRedeemRequest,
    ManagedStatusRequest,
    authenticate_managed_registry_peer,
    authenticate_managed_registry_server,
    authenticated_managed_clinical_registry_audit_sha256,
    load_authenticated_managed_clinical_registry_audit,
    load_authenticated_managed_registry_audit_chain,
    managed_clinical_registry_audit_hmac,
    managed_clinical_registry_config_sha256,
    verify_authenticated_managed_clinical_registry_audit,
)
from vaxreplay.agentic.managed_clinical_startup import (
    ManagedClinicalAttemptInventory,
    ManagedClinicalCapabilityLedger,
    ManagedClinicalHostAdapter,
    ManagedClinicalStartupConfig,
    ManagedClinicalStartupReconciler,
    managed_clinical_cleanup_key_id,
    managed_clinical_startup_config_sha256,
)
from vaxreplay.bundle import canonical_json_bytes

NOW = datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)
AUTHORITY = 'organizer.lane-a.example'
LAUNCHER_ID = 'vaxreplay-lane-a-canonical-operator'
LAUNCHER_SHA = '1' * 64
RESERVATION_SHA = '2' * 64
LAUNCH_SHA = '3' * 64
SYSTEM_SHA = '4' * 64
PREPARED_SHA = '5' * 64
CAPABILITY_ID = '6' * 64
SESSION_ID = '7' * 32
RUN_ID = '8' * 32
STARTUP_KEY = b'managed-registry-startup-cleanup-key-001'


def _config(tmp_path: Path) -> ManagedClinicalRegistryConfig:
    socket_namespace = hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:16]
    return ManagedClinicalRegistryConfig(
        service_id='vaxreplay-managed-registry',
        service_version='dev-v0.1',
        registry_authority_id=AUTHORITY,
        database_path=str(tmp_path / 'registry' / 'attempts.sqlite3'),
        socket_path=str(Path('/tmp') / f'vrk-registry-{socket_namespace}' / 'run' / 'attempts.sock'),
        production_evidence_root=str(tmp_path / 'evidence'),
        protocol_audit_root=str(tmp_path / 'registry-audit'),
        allowed_launcher_uid=0,
        allowed_launcher_gid=0,
        canonical_launcher_id=LAUNCHER_ID,
        canonical_launcher_executable_sha256=LAUNCHER_SHA,
        launcher_process_executable_sha256='9' * 64,
        service_process_executable_sha256='a' * 64,
        startup_config_sha256='b' * 64,
        startup_cleanup_receipt_key_id='c' * 64,
        connection_timeout_seconds=5,
    )


def _peer(*, uid: int = 0, pid: int = 4567) -> ManagedClinicalPeerIdentity:
    return ManagedClinicalPeerIdentity(
        pid=pid,
        uid=uid,
        gid=0,
        canonical_launcher_id=LAUNCHER_ID,
        canonical_launcher_executable_sha256=LAUNCHER_SHA,
    )


def _terminal_task_record(*, succeeded: bool) -> ClinicalProductionTaskRecord:
    launch = ClinicalProductionTaskLaunch(
        registry_authority_id=AUTHORITY,
        reservation_sha256=RESERVATION_SHA,
        cohort_manifest_sha256='d' * 64,
        system_identity_sha256=SYSTEM_SHA,
        episode_id='episode-001',
        workspace_manifest_sha256='e' * 64,
        run_id=RUN_ID,
        claimed_at=NOW,
    )
    redemption = ClinicalProductionStartRedemption(
        registry_authority_id=AUTHORITY,
        reservation_sha256=RESERVATION_SHA,
        launch_sha256=clinical_production_task_launch_sha256(launch),
        system_identity_sha256=SYSTEM_SHA,
        episode_id='episode-001',
        run_id=RUN_ID,
        canonical_launcher_id=LAUNCHER_ID,
        canonical_launcher_executable_sha256=LAUNCHER_SHA,
        prepared_worker_sha256=PREPARED_SHA,
        guest_rpc_session_id=SESSION_ID,
        gateway_capability_id=CAPABILITY_ID,
        redeemed_at=NOW,
    )
    return ClinicalProductionTaskRecord(
        episode_id='episode-001',
        state='succeeded' if succeeded else 'failed',
        launch=launch,
        launch_sha256=clinical_production_task_launch_sha256(launch),
        start_redemption=redemption,
        start_redemption_sha256=clinical_production_start_redemption_sha256(redemption),
        terminal_code=(
            ClinicalProductionTerminalCode.SUCCESS
            if succeeded
            else ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED
        ),
        evidence_sha256='f' * 64,
        terminal_record_sha256=None if succeeded else '0' * 64,
        submission_sha256='1' * 64 if succeeded else None,
        terminal_at=NOW,
    )


class _FakeRegistry:
    authority_id = AUTHORITY

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def reservation_hashes(self) -> tuple[str, ...]:
        return ()

    def reservation_context(self, reservation_sha256: str):
        assert reservation_sha256 == RESERVATION_SHA
        system = SimpleNamespace(
            canonical_launcher_id=LAUNCHER_ID,
            canonical_launcher_executable_sha256=LAUNCHER_SHA,
        )
        return SimpleNamespace(reservation=SimpleNamespace(system=system))

    def redeem_task_start(self, **kwargs):
        self.calls.append(kwargs)
        return ClinicalProductionStartRedemption(
            registry_authority_id=AUTHORITY,
            reservation_sha256=RESERVATION_SHA,
            launch_sha256=LAUNCH_SHA,
            system_identity_sha256=SYSTEM_SHA,
            episode_id='episode-001',
            run_id=RUN_ID,
            canonical_launcher_id=str(kwargs['canonical_launcher_id']),
            canonical_launcher_executable_sha256=str(kwargs['canonical_launcher_executable_sha256']),
            prepared_worker_sha256=PREPARED_SHA,
            guest_rpc_session_id=SESSION_ID,
            gateway_capability_id=CAPABILITY_ID,
            redeemed_at=NOW,
        )


class _EmptyHost:
    def owned_run_ids(self):
        return ()

    def scan_process_groups(self):
        return ()

    def scan_cgroups(self):
        return ()

    def scan_jail_roots(self):
        return ()

    def scan_vsock_endpoints(self):
        return ()

    def terminate_process_group(self, artifact, *, grace_seconds):  # noqa: ANN001
        raise AssertionError((artifact, grace_seconds))

    def reap_process_group(self, artifact):  # noqa: ANN001
        raise AssertionError(artifact)

    def remove_vsock_endpoint(self, artifact):  # noqa: ANN001
        raise AssertionError(artifact)

    def remove_cgroup(self, artifact):  # noqa: ANN001
        raise AssertionError(artifact)

    def remove_jail_root(self, artifact):  # noqa: ANN001
        raise AssertionError(artifact)

    def finalize_reconciled_run(self, run_id):  # noqa: ANN001
        raise AssertionError(run_id)


class _EmptyCapabilities:
    def inventory(self):
        return ()

    def revoke(self, capability):  # noqa: ANN001
        raise AssertionError(capability)


class _EmptyAttempts:
    authority_id = AUTHORITY

    def inventory(self):
        return ()


def _startup_config(tmp_path: Path) -> ManagedClinicalStartupConfig:
    return ManagedClinicalStartupConfig(
        reconciler_id='vaxreplay-managed-startup',
        reconciler_version='test-v1',
        registry_authority_id=AUTHORITY,
        runtime_config_sha256='d' * 64,
        worker_spec_sha256='e' * 64,
        cleanup_receipt_key_id=managed_clinical_cleanup_key_id(STARTUP_KEY),
        cgroup_root=str(tmp_path / 'cgroups'),
        jail_root=str(tmp_path / 'jails'),
        vsock_root=str(tmp_path / 'jails'),
        receipt_root=str(tmp_path / 'startup-receipts'),
        cleanup_grace_seconds=1,
    )


def _quiesced_service(
    tmp_path: Path,
) -> tuple[ManagedClinicalRegistryService, ManagedClinicalStartupConfig, _FakeRegistry]:
    startup = _startup_config(tmp_path)
    config = _config(tmp_path).model_copy(
        update={
            'startup_config_sha256': managed_clinical_startup_config_sha256(startup),
            'startup_cleanup_receipt_key_id': startup.cleanup_receipt_key_id,
        }
    )
    registry = _FakeRegistry()
    service = ManagedClinicalRegistryService(
        config=config,
        workspace_receipt_keys_by_id={},
        registry=cast(SqliteClinicalProductionRegistry, registry),
        startup_config=startup,
        startup_cleanup_receipt_key=STARTUP_KEY,
    )
    return service, startup, registry


def _startup_reconciliation_request() -> FirecrackerClinicalStartupReconciliationRequest:
    return FirecrackerClinicalStartupReconciliationRequest(
        runtime_config_sha256='d' * 64,
        execution_policy_sha256='1' * 64,
        worker_spec_sha256='e' * 64,
        gateway_policy_sha256='2' * 64,
        gateway_route_sha256='3' * 64,
        bootstrap_authorization_key_id='4' * 64,
        bootstrap_receipt_key_id='5' * 64,
        retained_journals=(),
        requested_at=NOW,
    )


def _redeem_registry_request() -> ManagedClinicalRegistryRequest:
    return ManagedClinicalRegistryRequest(
        request_id='a' * 32,
        operation='redeem',
        payload=ManagedRedeemRequest(
            reservation_sha256=RESERVATION_SHA,
            episode_id='episode-001',
            launch_sha256=LAUNCH_SHA,
            prepared_worker_sha256=PREPARED_SHA,
            guest_rpc_session_id=SESSION_ID,
            gateway_capability_id=CAPABILITY_ID,
            redeemed_at=NOW,
        ).model_dump(mode='json'),
    )


def _begin_registry_request(
    config: ManagedClinicalRegistryConfig,
    *,
    request_id: str = 'b' * 32,
) -> ManagedClinicalRegistryRequest:
    return ManagedClinicalRegistryRequest(
        request_id=request_id,
        operation='begin_reconciliation',
        payload=ManagedBeginReconciliationRequest(
            startup_config_sha256=config.startup_config_sha256,
            cleanup_receipt_key_id=config.startup_cleanup_receipt_key_id,
            requested_at=NOW,
        ).model_dump(mode='json'),
    )


def _service(
    tmp_path: Path,
    registry: _FakeRegistry,
) -> ManagedClinicalRegistryService:
    service = ManagedClinicalRegistryService(
        config=_config(tmp_path),
        workspace_receipt_keys_by_id={},
        registry=cast(SqliteClinicalProductionRegistry, registry),
    )
    # Identity-focused dispatcher tests predate the boot reconciliation protocol.  Production
    # construction has no equivalent bypass; dedicated tests below exercise the closed-at-boot
    # state and signed release transaction.
    service._reconciliation_required = False
    return service


def test_managed_registry_request_cannot_select_database_or_launcher_identity() -> None:
    with pytest.raises(ValidationError):
        ManagedClinicalRegistryRequest.model_validate(
            {
                'request_id': 'a' * 32,
                'operation': 'status',
                'payload': {'reservation_sha256': RESERVATION_SHA},
                'database_path': '/tmp/copied.sqlite3',
            }
        )
    with pytest.raises(ValidationError):
        ManagedClinicalRegistryRequest.model_validate(
            {
                'request_id': 'a' * 32,
                'operation': 'status',
                'payload': {'reservation_sha256': RESERVATION_SHA},
                'launcher_identity_supplied': True,
            }
        )


def test_managed_registry_redeem_derives_launcher_identity_from_service_config(
    tmp_path: Path,
) -> None:
    registry = _FakeRegistry()
    service = _service(tmp_path, registry)
    payload = ManagedRedeemRequest(
        reservation_sha256=RESERVATION_SHA,
        episode_id='episode-001',
        launch_sha256=LAUNCH_SHA,
        prepared_worker_sha256=PREPARED_SHA,
        guest_rpc_session_id=SESSION_ID,
        gateway_capability_id=CAPABILITY_ID,
        redeemed_at=NOW,
    )
    request = ManagedClinicalRegistryRequest(
        request_id='a' * 32,
        operation='redeem',
        payload=payload.model_dump(mode='json'),
    )

    response = service.handle_authenticated(request, peer=_peer())

    assert response.ok
    assert response.result is not None
    assert response.result['start_redemption']['canonical_launcher_id'] == LAUNCHER_ID
    assert len(registry.calls) == 1
    assert registry.calls[0]['canonical_launcher_id'] == LAUNCHER_ID
    assert registry.calls[0]['canonical_launcher_executable_sha256'] == LAUNCHER_SHA


@pytest.mark.parametrize('operation', ('reserve', 'claim', 'redeem', 'record_run'))
def test_recovery_only_registry_cannot_admit_or_publish_work(
    tmp_path: Path,
    operation: str,
) -> None:
    registry = _FakeRegistry()
    service = ManagedClinicalRegistryService(
        config=_config(tmp_path),
        workspace_receipt_keys_by_id={},
        registry=cast(SqliteClinicalProductionRegistry, registry),
        recovery_only=True,
    )
    service._reconciliation_required = False
    request = ManagedClinicalRegistryRequest(
        request_id='f' * 32,
        operation=cast(Any, operation),
        payload={},
    )

    response = service.handle_authenticated(request, peer=_peer())

    assert not response.ok
    assert response.error_code == 'rejected'
    assert registry.calls == []


def test_managed_registry_rejects_wrong_authenticated_peer_before_backend_access(
    tmp_path: Path,
) -> None:
    registry = _FakeRegistry()
    service = _service(tmp_path, registry)
    request = ManagedClinicalRegistryRequest(
        request_id='a' * 32,
        operation='redeem',
        payload=ManagedRedeemRequest(
            reservation_sha256=RESERVATION_SHA,
            episode_id='episode-001',
            launch_sha256=LAUNCH_SHA,
            prepared_worker_sha256=PREPARED_SHA,
            guest_rpc_session_id=SESSION_ID,
            gateway_capability_id=CAPABILITY_ID,
            redeemed_at=NOW,
        ).model_dump(mode='json'),
    )

    response = service.handle_authenticated(request, peer=_peer(uid=9999))

    assert not response.ok
    assert response.error_code == 'unauthorized'
    assert registry.calls == []


def test_status_rechecks_reservation_launcher_identity(tmp_path: Path) -> None:
    class _WrongLauncherRegistry(_FakeRegistry):
        def reservation_context(self, reservation_sha256: str):
            del reservation_sha256
            system = SimpleNamespace(
                canonical_launcher_id='different-launcher',
                canonical_launcher_executable_sha256='f' * 64,
            )
            return SimpleNamespace(reservation=SimpleNamespace(system=system))

        def task_records(self, _reservation_sha256: str):
            raise AssertionError('task records must not be disclosed for another launcher')

    service = _service(tmp_path, _WrongLauncherRegistry())
    request = ManagedClinicalRegistryRequest(
        request_id='a' * 32,
        operation='status',
        payload=ManagedStatusRequest(
            reservation_sha256=RESERVATION_SHA,
        ).model_dump(mode='json'),
    )

    response = service.handle_authenticated(request, peer=_peer())

    assert not response.ok
    assert response.error_code == 'rejected'


def test_production_result_cannot_select_an_arbitrary_absolute_root(tmp_path: Path) -> None:
    class _RunRegistry(_FakeRegistry):
        def task_records(self, _reservation_sha256: str):
            return (
                SimpleNamespace(
                    episode_id='episode-001',
                    launch=SimpleNamespace(run_id=RUN_ID),
                ),
            )

    service = _service(tmp_path, _RunRegistry())
    payload = ManagedRecordRunRequest(
        reservation_sha256=RESERVATION_SHA,
        episode_id='episode-001',
        production_run_root=str(tmp_path / 'caller-selected' / RUN_ID),
        terminal_at=NOW,
    )

    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='service-owned run namespace',
    ):
        service._require_production_run_root(payload)


def test_linux_peer_authentication_uses_kernel_credentials_not_frame_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(socket, 'SO_PEERCRED', 17, raising=False)
    verified: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        managed_registry_module,
        '_require_linux_process_image',
        lambda pid, *, expected_sha256, label: verified.append((pid, expected_sha256, label)),
    )

    class _Connection:
        def getsockopt(self, level: int, option: int, size: int) -> bytes:
            assert (level, option, size) == (socket.SOL_SOCKET, 17, struct.calcsize('3i'))
            return struct.pack('3i', 4567, 0, 0)

    identity = authenticate_managed_registry_peer(
        cast(socket.socket, _Connection()),
        config=config,
    )

    assert identity.pid == 4567
    assert identity.uid == 0
    assert identity.canonical_launcher_id == LAUNCHER_ID
    assert identity.derived_from_so_peercred_and_service_config
    assert verified == [(4567, config.launcher_process_executable_sha256, 'launcher')]


def test_client_authenticates_root_service_process_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(socket, 'SO_PEERCRED', 17, raising=False)
    verified: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        managed_registry_module,
        '_require_linux_process_image',
        lambda pid, *, expected_sha256, label: verified.append((pid, expected_sha256, label)),
    )

    class _Connection:
        def getsockopt(self, level: int, option: int, size: int) -> bytes:
            assert (level, option, size) == (socket.SOL_SOCKET, 17, struct.calcsize('3i'))
            return struct.pack('3i', 7654, 0, 0)

    identity = authenticate_managed_registry_server(
        cast(socket.socket, _Connection()),
        config=config,
    )

    assert identity.pid == 7654
    assert verified == [(7654, config.service_process_executable_sha256, 'service')]


def test_managed_registry_config_has_an_external_client_pin(tmp_path: Path) -> None:
    config = _config(tmp_path)
    expected = managed_clinical_registry_config_sha256(config)

    client = managed_registry_module.ManagedClinicalRegistryClient(
        config,
        expected_config_sha256=expected,
    )

    assert client.config_sha256 == expected
    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='external SHA-256 pin',
    ):
        managed_registry_module.ManagedClinicalRegistryClient(
            config,
            expected_config_sha256='f' * 64,
        )


def test_client_success_performs_one_launcher_reload_after_authoritative_service_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    client = managed_registry_module.ManagedClinicalRegistryClient(
        config,
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
    )
    expected = _terminal_task_record(succeeded=True)
    production_root = tmp_path / 'production-evidence'
    events: list[tuple[str, object]] = []

    def service_call(operation: object, payload: object) -> dict[str, object]:
        events.append(('service', (operation, payload)))
        return {'task_record': expected.model_dump(mode='json')}

    def launcher_reload(root: Path, attempt_sha256: str):  # noqa: ANN202
        events.append(('launcher-reload', (root, attempt_sha256)))
        return SimpleNamespace(authenticated_outer_receipt_sha256=expected.evidence_sha256)

    monkeypatch.setattr(client, '_call', service_call)
    observed = client.record_production_run(
        reservation_sha256=RESERVATION_SHA,
        episode_id='episode-001',
        production_run_root=production_root,
        reauthenticate=launcher_reload,  # type: ignore[arg-type]
        terminal_at=NOW,
    )

    assert observed == expected
    assert [name for name, _ in events] == ['service', 'launcher-reload']
    assert events[1][1] == (production_root, expected.start_redemption_sha256)


def test_client_rejects_launcher_reload_that_differs_from_authoritative_service_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    client = managed_registry_module.ManagedClinicalRegistryClient(
        config,
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
    )
    expected = _terminal_task_record(succeeded=True)
    monkeypatch.setattr(
        client,
        '_call',
        lambda _operation, _payload: {'task_record': expected.model_dump(mode='json')},
    )

    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='differs from the authoritative managed-registry digest',
    ):
        client.record_production_run(
            reservation_sha256=RESERVATION_SHA,
            episode_id='episode-001',
            production_run_root=tmp_path / 'changed-evidence',
            reauthenticate=lambda *_: SimpleNamespace(authenticated_outer_receipt_sha256='a' * 64),  # type: ignore[arg-type]
            terminal_at=NOW,
        )


def test_client_service_failure_never_invokes_launcher_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    client = managed_registry_module.ManagedClinicalRegistryClient(
        config,
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
    )
    expected = _terminal_task_record(succeeded=False)
    reload_calls: list[object] = []
    monkeypatch.setattr(
        client,
        '_call',
        lambda _operation, _payload: {'task_record': expected.model_dump(mode='json')},
    )

    observed = client.record_production_run(
        reservation_sha256=RESERVATION_SHA,
        episode_id='episode-001',
        production_run_root=tmp_path / 'rejected-evidence',
        reauthenticate=lambda *_: reload_calls.append(object()),  # type: ignore[arg-type,return-value]
        terminal_at=NOW,
    )

    assert observed == expected
    assert reload_calls == []


def test_client_decodes_strict_response_models_in_json_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    client = managed_registry_module.ManagedClinicalRegistryClient(
        config,
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
    )
    expected = ClinicalProductionStartRedemption(
        registry_authority_id=AUTHORITY,
        reservation_sha256=RESERVATION_SHA,
        launch_sha256=LAUNCH_SHA,
        system_identity_sha256=SYSTEM_SHA,
        episode_id='episode-001',
        run_id=RUN_ID,
        canonical_launcher_id=LAUNCHER_ID,
        canonical_launcher_executable_sha256=LAUNCHER_SHA,
        prepared_worker_sha256=PREPARED_SHA,
        guest_rpc_session_id=SESSION_ID,
        gateway_capability_id=CAPABILITY_ID,
        redeemed_at=NOW,
    )

    def response(_operation: object, _payload: object) -> dict[str, object]:
        return {'start_redemption': expected.model_dump(mode='json')}

    monkeypatch.setattr(client, '_call', response)
    observed = client.redeem_task_start(
        reservation_sha256=RESERVATION_SHA,
        episode_id='episode-001',
        launch_sha256=LAUNCH_SHA,
        canonical_launcher_id=LAUNCHER_ID,
        canonical_launcher_executable_sha256=LAUNCHER_SHA,
        prepared_worker_sha256=PREPARED_SHA,
        guest_rpc_session_id=SESSION_ID,
        gateway_capability_id=CAPABILITY_ID,
        redeemed_at=NOW,
    )

    assert observed == expected


def test_serving_rejects_an_injected_backend_without_the_configured_path(tmp_path: Path) -> None:
    service = _service(tmp_path, _FakeRegistry())

    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='must expose its configured database path',
    ):
        service._prepare_and_pin_database()


def test_managed_registry_paths_are_fixed_normalized_absolute_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.database_path.endswith('/registry/attempts.sqlite3')
    assert config.socket_path.endswith('/run/attempts.sock')
    assert config.production_evidence_root.endswith('/evidence')
    assert config.database_path_selected_by_service_only
    assert config.production_evidence_namespace_selected_by_service_only
    assert config.root_owned_service_required
    assert config.one_host_authority
    assert not config.cross_host_consensus_claimed

    with pytest.raises(ValidationError):
        ManagedClinicalRegistryConfig.model_validate(
            {**config.model_dump(mode='python'), 'database_path': '../copied.sqlite3'}
        )


def test_managed_registry_config_enforces_linux_af_unix_encoded_path_limit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    exact_limit = '/' + ('x' * (managed_registry_module.LINUX_AF_UNIX_PATHNAME_MAX_BYTES - 1))
    too_long = '/' + ('x' * managed_registry_module.LINUX_AF_UNIX_PATHNAME_MAX_BYTES)

    accepted = ManagedClinicalRegistryConfig.model_validate(
        {**config.model_dump(mode='python'), 'socket_path': exact_limit}
    )
    assert len(os.fsencode(accepted.socket_path)) == managed_registry_module.LINUX_AF_UNIX_PATHNAME_MAX_BYTES
    with pytest.raises(ValidationError, match='Linux 107-byte limit'):
        ManagedClinicalRegistryConfig.model_validate({**config.model_dump(mode='python'), 'socket_path': too_long})


def test_serve_rejects_unchecked_long_socket_before_database_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_long = '/' + ('x' * managed_registry_module.LINUX_AF_UNIX_PATHNAME_MAX_BYTES)
    config = _config(tmp_path).model_copy(update={'socket_path': too_long})
    service = ManagedClinicalRegistryService(
        config=config,
        workspace_receipt_keys_by_id={},
    )
    boundary_calls: list[str] = []

    def unexpected_boundary() -> None:
        boundary_calls.append('root-boundary')

    def unexpected_database_preparation() -> None:
        boundary_calls.append('database-preparation')

    monkeypatch.setattr(service, '_require_root_host_boundary', unexpected_boundary)
    monkeypatch.setattr(service, '_prepare_and_pin_database', unexpected_database_preparation)

    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='AF_UNIX pathname exceeds the Linux 107-byte limit',
    ):
        service.serve_until(stop_event=threading.Event())

    assert boundary_calls == []
    assert not Path(config.database_path).exists()


def test_registry_is_closed_at_boot_and_reopens_only_for_exact_signed_cleanup(
    tmp_path: Path,
) -> None:
    service, startup, registry = _quiesced_service(tmp_path)

    blocked = service.handle_authenticated(_redeem_registry_request(), peer=_peer())
    assert not blocked.ok
    assert blocked.error_code == 'rejected'
    assert registry.calls == []

    begun = service.handle_authenticated(
        _begin_registry_request(service.config),
        peer=_peer(),
    )
    assert begun.ok
    assert begun.result is not None
    assert begun.result['attempts'] == []
    lease_token = str(begun.result['lease_token'])

    reconciler = ManagedClinicalStartupReconciler(
        config=startup,
        host=cast(ManagedClinicalHostAdapter, _EmptyHost()),
        capabilities=cast(ManagedClinicalCapabilityLedger, _EmptyCapabilities()),
        attempts=cast(ManagedClinicalAttemptInventory, _EmptyAttempts()),
        cleanup_receipt_key=STARTUP_KEY,
        clock=lambda: NOW,
    )
    reconciler.reconcile(_startup_reconciliation_request())
    authenticated = reconciler.last_authenticated_receipt
    assert authenticated is not None
    finished = service.handle_authenticated(
        ManagedClinicalRegistryRequest(
            request_id='c' * 32,
            operation='finish_reconciliation',
            payload=ManagedFinishReconciliationRequest(
                lease_token=lease_token,
                authenticated_cleanup=authenticated,
            ).model_dump(mode='json'),
        ),
        peer=_peer(),
    )
    assert finished.ok
    assert finished.result is not None
    assert finished.result['startup_reconciliation_admitted'] is True

    admitted = service.handle_authenticated(_redeem_registry_request(), peer=_peer())
    assert admitted.ok
    assert len(registry.calls) == 1

    restarted = ManagedClinicalRegistryService(
        config=service.config,
        workspace_receipt_keys_by_id={},
        registry=cast(SqliteClinicalProductionRegistry, registry),
        startup_config=startup,
        startup_cleanup_receipt_key=STARTUP_KEY,
    )
    reclosed = restarted.handle_authenticated(_redeem_registry_request(), peer=_peer())
    assert not reclosed.ok
    assert reclosed.error_code == 'rejected'
    assert len(registry.calls) == 1


def test_reconciliation_lease_allows_only_one_launcher_process(
    tmp_path: Path,
) -> None:
    service, _startup, registry = _quiesced_service(tmp_path)

    def begin(pid: int, request_id: str):
        return service.handle_authenticated(
            _begin_registry_request(service.config, request_id=request_id),
            peer=_peer(pid=pid),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = tuple(
            future.result()
            for future in (
                executor.submit(begin, 4567, 'd' * 32),
                executor.submit(begin, 7654, 'e' * 32),
            )
        )

    assert sum(response.ok for response in responses) == 1
    assert sum(response.error_code == 'rejected' for response in responses) == 1
    still_blocked = service.handle_authenticated(_redeem_registry_request(), peer=_peer())
    assert not still_blocked.ok
    assert registry.calls == []


def _audit_artifact(
    tmp_path: Path,
    *,
    sequence: int = 0,
    predecessor: str = '0' * 64,
    config: ManagedClinicalRegistryConfig | None = None,
) -> AuthenticatedManagedClinicalRegistryAudit:
    config = config or _config(tmp_path)
    request = _redeem_registry_request()
    response = ManagedClinicalRegistryResponse(
        request_id=request.request_id,
        operation=request.operation,
        ok=False,
        error_code='rejected',
        registry_authority_id=AUTHORITY,
    )
    path = Path(config.protocol_audit_root) / f'{sequence:020d}-{request.request_id}.json'
    unsigned = AuthenticatedManagedClinicalRegistryAudit(
        registry_config_sha256=managed_clinical_registry_config_sha256(config),
        registry_authority_id=AUTHORITY,
        sequence=sequence,
        predecessor_audit_sha256=predecessor,
        request=request,
        request_sha256=hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
        response=response,
        response_sha256=hashlib.sha256(canonical_json_bytes(response)).hexdigest(),
        launcher_peer=_peer(),
        server=ManagedClinicalRegistryAuditServerIdentity(
            service_pid=123,
            service_start_time_ticks=456,
            service_uid=0,
            service_gid=0,
            service_executable_sha256=config.service_process_executable_sha256,
            socket_path=config.socket_path,
            socket_device_id=1,
            socket_inode=2,
            database_path=config.database_path,
            database_device_id=3,
            database_inode=4,
        ),
        audited_at=NOW,
        audit_key_id=managed_clinical_cleanup_key_id(STARTUP_KEY),
        audit_hmac_sha256='0' * 64,
        persisted_path=str(path),
    )
    return unsigned.model_copy(
        update={
            'audit_hmac_sha256': managed_clinical_registry_audit_hmac(
                unsigned,
                key=STARTUP_KEY,
            )
        }
    )


def test_authenticated_protocol_audit_chain_is_create_once_and_tamper_evident(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    root = Path(config.protocol_audit_root)
    root.mkdir(mode=0o700)
    first = _audit_artifact(tmp_path)
    path = Path(first.persisted_path)
    path.write_bytes(canonical_json_bytes(first))
    path.chmod(0o600)

    loaded = load_authenticated_managed_clinical_registry_audit(
        path,
        expected_root=root,
        required_uid=os.geteuid(),
    )
    verified = verify_authenticated_managed_clinical_registry_audit(
        loaded,
        key=STARTUP_KEY,
        expected_key_id=managed_clinical_cleanup_key_id(STARTUP_KEY),
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
        expected_sequence=0,
        expected_predecessor_sha256='0' * 64,
    )
    assert verified == first
    assert load_authenticated_managed_registry_audit_chain(
        root,
        key=STARTUP_KEY,
        expected_key_id=managed_clinical_cleanup_key_id(STARTUP_KEY),
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
        required_uid=os.geteuid(),
    ) == (first,)

    forged = first.model_copy(update={'response_sha256': 'f' * 64})
    with pytest.raises(ValidationError, match='inconsistent exact bindings'):
        AuthenticatedManagedClinicalRegistryAudit.model_validate_json(canonical_json_bytes(forged))


def _write_raw_protocol_audit_member(root: Path, *, sequence: int, size: int) -> Path:
    path = root / f'{sequence:020d}-{sequence:032x}.json'
    path.write_bytes(b'x' * size)
    path.chmod(0o600)
    return path


@pytest.mark.parametrize('limit_kind', ['entry-count', 'aggregate-bytes'])
def test_protocol_audit_inventory_rejects_bounds_before_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    config = _config(tmp_path)
    root = Path(config.protocol_audit_root)
    root.mkdir(mode=0o700)
    _write_raw_protocol_audit_member(root, sequence=0, size=3)
    _write_raw_protocol_audit_member(root, sequence=1, size=3)
    if limit_kind == 'entry-count':
        monkeypatch.setattr(
            managed_registry_module,
            'MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES',
            1,
        )
        expected = 'entry-count limit'
    else:
        monkeypatch.setattr(
            managed_registry_module,
            'MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES',
            5,
        )
        expected = 'aggregate-byte limit'

    def decoding_is_too_late(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError('bounded inventory must reject before decoding JSON')

    monkeypatch.setattr(
        managed_registry_module,
        'load_authenticated_managed_clinical_registry_audit',
        decoding_is_too_late,
    )
    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match=expected,
    ):
        load_authenticated_managed_registry_audit_chain(
            root,
            key=STARTUP_KEY,
            expected_key_id=managed_clinical_cleanup_key_id(STARTUP_KEY),
            expected_config_sha256=managed_clinical_registry_config_sha256(config),
            required_uid=os.geteuid(),
        )


@pytest.mark.parametrize('limit_kind', ['entry-count', 'aggregate-bytes'])
def test_protocol_audit_creation_fails_closed_at_history_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    root = tmp_path / 'registry-audit'
    root.mkdir(mode=0o700)
    existing = _write_raw_protocol_audit_member(root, sequence=0, size=3)
    target = root / f'{1:020d}-{1:032x}.json'
    if limit_kind == 'entry-count':
        monkeypatch.setattr(
            managed_registry_module,
            'MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES',
            1,
        )
        expected = 'entry-count limit'
    else:
        monkeypatch.setattr(
            managed_registry_module,
            'MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES',
            5,
        )
        expected = 'aggregate-byte limit'

    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match=expected,
    ):
        managed_registry_module._write_create_once_audit(target, b'abc')
    assert tuple(root.iterdir()) == (existing,)
    assert existing.read_bytes() == b'xxx'


def test_protocol_audit_service_startup_rejects_exhausted_bounded_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup = _startup_config(tmp_path)
    config = _config(tmp_path).model_copy(
        update={
            'startup_config_sha256': managed_clinical_startup_config_sha256(startup),
            'startup_cleanup_receipt_key_id': startup.cleanup_receipt_key_id,
        }
    )
    root = Path(config.protocol_audit_root)
    root.mkdir(mode=0o700)
    artifact = _audit_artifact(tmp_path, config=config)
    path = Path(artifact.persisted_path)
    path.write_bytes(canonical_json_bytes(artifact))
    path.chmod(0o600)
    service = ManagedClinicalRegistryService(
        config=config,
        workspace_receipt_keys_by_id={},
        registry=cast(SqliteClinicalProductionRegistry, _FakeRegistry()),
        startup_config=startup,
        startup_cleanup_receipt_key=STARTUP_KEY,
    )
    monkeypatch.setattr(
        managed_registry_module,
        'MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES',
        1,
    )

    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='no bounded capacity',
    ):
        service._initialize_protocol_audit()
    assert service._audit_sequence is None
    assert service._audit_predecessor_sha256 is None
    assert service._audit_aggregate_bytes is None


def test_protocol_audit_loader_rejects_metadata_change_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    root = Path(config.protocol_audit_root)
    root.mkdir(mode=0o700)
    artifact = _audit_artifact(tmp_path)
    path = Path(artifact.persisted_path)
    path.write_bytes(canonical_json_bytes(artifact))
    path.chmod(0o600)

    real_fstat = os.fstat
    calls = 0

    def changing_fstat(descriptor: int):  # noqa: ANN202
        nonlocal calls
        observed = real_fstat(descriptor)
        calls += 1
        if calls != 2:
            return observed
        values = list(observed)
        # ``stat_result`` tuple slot 8 is mtime.  Reconstructing through the public
        # type also updates ``st_mtime_ns`` on every supported Python platform.
        values[8] = observed.st_mtime + 1
        return os.stat_result(values)

    monkeypatch.setattr(os, 'fstat', changing_fstat)
    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='changed while reading',
    ):
        load_authenticated_managed_clinical_registry_audit(
            path,
            expected_root=root,
            required_uid=os.geteuid(),
        )


def test_protocol_audit_create_once_failure_preserves_preexisting_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / 'registry-audit'
    root.mkdir(mode=0o700)
    path = root / f'00000000000000000000-{"a" * 32}.json'
    original = b'preexisting-authoritative-bytes'
    path.write_bytes(original)
    path.chmod(0o600)

    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='persisted create-once',
    ):
        managed_registry_module._write_create_once_audit(path, b'replacement')
    assert path.read_bytes() == original


def test_protocol_audit_torn_staging_write_never_publishes_partial_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'registry-audit'
    root.mkdir(mode=0o700)
    path = root / f'00000000000000000000-{"a" * 32}.json'
    real_write = os.write
    writes = 0

    def torn_write(descriptor: int, body: bytes) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_write(descriptor, body[:3])
        raise OSError('simulated torn write')

    monkeypatch.setattr(os, 'write', torn_write)
    with pytest.raises(
        managed_registry_module.ManagedClinicalRegistryError,
        match='persisted create-once',
    ):
        managed_registry_module._write_create_once_audit(path, b'complete-audit-body')

    assert not path.exists()
    assert tuple(root.iterdir()) == ()


@pytest.mark.parametrize('crash_at', ['write', 'publish'])
def test_protocol_audit_restart_reaps_only_unpublished_atomic_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_at: str,
) -> None:
    root = tmp_path / 'registry-audit'
    root.mkdir(mode=0o700)
    path = root / f'00000000000000000000-{"a" * 32}.json'
    payload = b'complete-audit-body'

    if crash_at == 'write':
        real_write = os.write
        writes = 0

        def interrupted_write(descriptor: int, body: bytes) -> int:
            nonlocal writes
            writes += 1
            if writes == 1:
                return real_write(descriptor, body[:3])
            raise KeyboardInterrupt

        with monkeypatch.context() as context:
            context.setattr(os, 'write', interrupted_write)
            with pytest.raises(KeyboardInterrupt):
                managed_registry_module._write_create_once_audit(path, payload)
    else:
        with monkeypatch.context() as context:
            context.setattr(
                managed_registry_module,
                'rename_file_noreplace',
                lambda _source, _target: (_ for _ in ()).throw(KeyboardInterrupt()),
            )
            with pytest.raises(KeyboardInterrupt):
                managed_registry_module._write_create_once_audit(path, payload)

    assert not path.exists()
    stages = tuple(root.iterdir())
    assert len(stages) == 1 and stages[0].name.startswith('.audit-stage-')

    managed_registry_module._reap_incomplete_protocol_audit_staging(root)
    assert tuple(root.iterdir()) == ()
    managed_registry_module._write_create_once_audit(path, payload)
    assert path.read_bytes() == payload
    assert tuple(root.iterdir()) == (path,)


@pytest.mark.skipif(
    sys.platform != 'linux' or os.geteuid() != 0 or not hasattr(socket, 'SO_PEERCRED'),
    reason='wire audit integration requires root Linux SO_PEERCRED',
)
def test_unix_service_persists_authenticated_wire_audit_before_response_and_across_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    for directory in ('registry', 'run', 'evidence', 'registry-audit', 'startup-receipts'):
        (tmp_path / directory).mkdir(mode=0o700)
    startup = _startup_config(tmp_path)
    executable_sha256 = hashlib.sha256(Path('/proc/self/exe').read_bytes()).hexdigest()
    config = _config(tmp_path).model_copy(
        update={
            'canonical_launcher_executable_sha256': LAUNCHER_SHA,
            'launcher_process_executable_sha256': executable_sha256,
            'service_process_executable_sha256': executable_sha256,
            'startup_config_sha256': managed_clinical_startup_config_sha256(startup),
            'startup_cleanup_receipt_key_id': startup.cleanup_receipt_key_id,
        }
    )
    socket_parent = Path(config.socket_path).parent
    socket_parent.mkdir(mode=0o700, parents=True)
    socket_parent.chmod(0o700)
    request.addfinalizer(lambda: shutil.rmtree(socket_parent.parent, ignore_errors=True))
    stop = threading.Event()
    ready = threading.Event()
    errors: list[BaseException] = []

    original_send_frame = managed_registry_module._send_frame

    def assert_audit_precedes_response(connection, payload, maximum):  # noqa: ANN001
        assert tuple(Path(config.protocol_audit_root).iterdir())
        original_send_frame(connection, payload, maximum)

    monkeypatch.setattr(managed_registry_module, '_send_frame', assert_audit_precedes_response)

    def start_service() -> tuple[ManagedClinicalRegistryService, threading.Thread]:
        service = ManagedClinicalRegistryService(
            config=config,
            workspace_receipt_keys_by_id={},
            startup_config=startup,
            startup_cleanup_receipt_key=STARTUP_KEY,
        )

        def serve() -> None:
            try:
                service.serve_until(stop_event=stop, ready_event=ready)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=False)
        thread.start()
        assert ready.wait(5)
        return service, thread

    def exchange(request: ManagedClinicalRegistryRequest) -> ManagedClinicalRegistryResponse:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(5)
            connection.connect(config.socket_path)
            request_bytes = canonical_json_bytes(request)
            connection.sendall(struct.pack('!I', len(request_bytes)) + request_bytes)
            header = connection.recv(4)
            assert len(header) == 4
            (length,) = struct.unpack('!I', header)
            response_bytes = bytearray()
            while len(response_bytes) < length:
                response_bytes.extend(connection.recv(length - len(response_bytes)))
            response = ManagedClinicalRegistryResponse.model_validate_json(bytes(response_bytes))
            assert canonical_json_bytes(response) == bytes(response_bytes)
            return response
        finally:
            connection.close()

    _service_one, thread_one = start_service()
    first_request = _begin_registry_request(config)
    first_response = exchange(first_request)
    assert first_response.ok
    stop.set()
    thread_one.join(5)
    assert not thread_one.is_alive() and not errors

    stop.clear()
    ready.clear()
    _service_two, thread_two = start_service()
    second_request = _redeem_registry_request()
    second_response = exchange(second_request)
    assert not second_response.ok and second_response.error_code == 'rejected'
    stop.set()
    thread_two.join(5)
    assert not thread_two.is_alive() and not errors

    chain = load_authenticated_managed_registry_audit_chain(
        Path(config.protocol_audit_root),
        key=STARTUP_KEY,
        expected_key_id=startup.cleanup_receipt_key_id,
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
    )
    assert len(chain) == 2
    assert (chain[0].request, chain[0].response) == (first_request, first_response)
    assert (chain[1].request, chain[1].response) == (second_request, second_response)
    assert chain[1].predecessor_audit_sha256 == (authenticated_managed_clinical_registry_audit_sha256(chain[0]))
    database_metadata = Path(config.database_path).lstat()
    for artifact in chain:
        assert artifact.launcher_peer.pid == os.getpid()
        assert artifact.server.service_pid == os.getpid()
        assert artifact.server.database_device_id == database_metadata.st_dev
        assert artifact.server.database_inode == database_metadata.st_ino
        assert artifact.server.service_executable_sha256 == executable_sha256
    assert chain[0].server.socket_inode != chain[1].server.socket_inode


def test_forged_cleanup_cannot_release_quiescence(
    tmp_path: Path,
) -> None:
    service, startup, registry = _quiesced_service(tmp_path)
    begun = service.handle_authenticated(
        _begin_registry_request(service.config),
        peer=_peer(),
    )
    assert begun.ok and begun.result is not None
    reconciler = ManagedClinicalStartupReconciler(
        config=startup,
        host=cast(ManagedClinicalHostAdapter, _EmptyHost()),
        capabilities=cast(ManagedClinicalCapabilityLedger, _EmptyCapabilities()),
        attempts=cast(ManagedClinicalAttemptInventory, _EmptyAttempts()),
        cleanup_receipt_key=STARTUP_KEY,
        clock=lambda: NOW,
    )
    reconciler.reconcile(_startup_reconciliation_request())
    authenticated = reconciler.last_authenticated_receipt
    assert authenticated is not None
    forged = authenticated.model_copy(update={'cleanup_hmac_sha256': '0' * 64})
    rejected = service.handle_authenticated(
        ManagedClinicalRegistryRequest(
            request_id='f' * 32,
            operation='finish_reconciliation',
            payload=ManagedFinishReconciliationRequest(
                lease_token=str(begun.result['lease_token']),
                authenticated_cleanup=forged,
            ).model_dump(mode='json'),
        ),
        peer=_peer(),
    )
    assert not rejected.ok
    assert rejected.error_code == 'rejected'
    still_blocked = service.handle_authenticated(_redeem_registry_request(), peer=_peer())
    assert not still_blocked.ok
    assert registry.calls == []
