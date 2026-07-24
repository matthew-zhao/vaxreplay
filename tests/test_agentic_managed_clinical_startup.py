from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pytest

from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalRuntime,
    FirecrackerClinicalStartupReconciliationReport,
    FirecrackerClinicalStartupReconciliationRequest,
    firecracker_clinical_startup_reconciliation_request_sha256,
)
from vaxreplay.agentic.managed_clinical_startup import (
    ManagedClinicalAttemptInventoryRecord,
    ManagedClinicalCapability,
    ManagedClinicalHostArtifact,
    ManagedClinicalStartupConfig,
    ManagedClinicalStartupError,
    ManagedClinicalStartupReconciler,
    load_authenticated_managed_cleanup,
    managed_clinical_cleanup_key_id,
    managed_clinical_ownership_hmac,
    managed_clinical_startup_config_sha256,
    reconcile_canonical_managed_runtime_startup,
    verify_authenticated_managed_cleanup,
)

NOW = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
RUN_ID = 'a' * 32
RUNTIME_SHA = '1' * 64
WORKER_SHA = '2' * 64
RESERVATION_SHA = '3' * 64
LAUNCH_SHA = '4' * 64
START_SHA = '5' * 64
AUTHORITY = 'organizer.lane-a.example'
KEY = b'k' * 32


class _Attempts:
    authority_id = AUTHORITY

    def __init__(self, values: tuple[ManagedClinicalAttemptInventoryRecord, ...]):
        self.values = values

    def inventory(self) -> tuple[ManagedClinicalAttemptInventoryRecord, ...]:
        return self.values


class _Capabilities:
    def __init__(self, values: tuple[ManagedClinicalCapability, ...], events: list[str]):
        self.values = list(values)
        self.events = events

    def inventory(self) -> tuple[ManagedClinicalCapability, ...]:
        return tuple(self.values)

    def revoke(self, capability: ManagedClinicalCapability) -> None:
        self.events.append(f'revoke:{capability.run_id}')
        self.values.remove(capability)


class _Host:
    def __init__(
        self,
        artifacts: tuple[ManagedClinicalHostArtifact, ...],
        events: list[str],
        *,
        retain_jail: bool = False,
    ) -> None:
        self.values = list(artifacts)
        self.events = events
        self.retain_jail = retain_jail

    def owned_run_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.run_id for item in self.values}))

    def _kind(self, name: str) -> tuple[ManagedClinicalHostArtifact, ...]:
        return tuple(item for item in self.values if item.artifact_kind == name)

    def scan_process_groups(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        return self._kind('process_group')

    def scan_cgroups(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        return self._kind('cgroup')

    def scan_jail_roots(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        return self._kind('jail_root')

    def scan_vsock_endpoints(self) -> tuple[ManagedClinicalHostArtifact, ...]:
        return self._kind('vsock_endpoint')

    def terminate_process_group(self, artifact: ManagedClinicalHostArtifact, *, grace_seconds: float) -> None:
        assert grace_seconds == 5
        self.events.append(f'terminate:{artifact.run_id}')

    def reap_process_group(self, artifact: ManagedClinicalHostArtifact) -> None:
        self.events.append(f'reap:{artifact.run_id}')
        self.values.remove(artifact)

    def remove_vsock_endpoint(self, artifact: ManagedClinicalHostArtifact) -> None:
        self.events.append(f'vsock:{artifact.run_id}')
        self.values.remove(artifact)

    def remove_cgroup(self, artifact: ManagedClinicalHostArtifact) -> None:
        self.events.append(f'cgroup:{artifact.run_id}')
        self.values.remove(artifact)

    def remove_jail_root(self, artifact: ManagedClinicalHostArtifact) -> None:
        self.events.append(f'jail:{artifact.run_id}')
        if not self.retain_jail:
            self.values.remove(artifact)

    def finalize_reconciled_run(self, run_id: str) -> None:
        self.events.append(f'finalize:{run_id}')


def _request() -> FirecrackerClinicalStartupReconciliationRequest:
    return FirecrackerClinicalStartupReconciliationRequest(
        runtime_config_sha256=RUNTIME_SHA,
        execution_policy_sha256='6' * 64,
        worker_spec_sha256=WORKER_SHA,
        gateway_policy_sha256='7' * 64,
        gateway_route_sha256='8' * 64,
        bootstrap_authorization_key_id='9' * 64,
        bootstrap_receipt_key_id='a' * 64,
        retained_journals=(),
        requested_at=NOW,
    )


def _attempt() -> ManagedClinicalAttemptInventoryRecord:
    return ManagedClinicalAttemptInventoryRecord(
        registry_authority_id=AUTHORITY,
        reservation_sha256=RESERVATION_SHA,
        launch_sha256=LAUNCH_SHA,
        start_redemption_sha256=START_SHA,
        run_id=RUN_ID,
        episode_id='episode-001',
        worker_spec_sha256=WORKER_SHA,
        state='launched',
    )


def _artifacts(tmp_path: Path) -> tuple[ManagedClinicalHostArtifact, ...]:
    roots = {
        'cgroup': tmp_path / 'cgroups' / RUN_ID,
        'jail_root': tmp_path / 'jails' / RUN_ID / 'root',
        'vsock_endpoint': tmp_path / 'jails' / RUN_ID / 'root' / 'run' / 'vsock.sock',
    }
    values: list[tuple[Literal['process_group', 'cgroup', 'jail_root', 'vsock_endpoint'], str]] = [
        ('process_group', 'pgid:4242'),
        ('cgroup', str(roots['cgroup'])),
        ('jail_root', str(roots['jail_root'])),
        ('vsock_endpoint', str(roots['vsock_endpoint'])),
    ]
    unsigned = tuple(
        ManagedClinicalHostArtifact(
            artifact_kind=kind,
            artifact_id=str(identifier),
            run_id=RUN_ID,
            registry_authority_id=AUTHORITY,
            reservation_sha256=RESERVATION_SHA,
            launch_sha256=LAUNCH_SHA,
            start_redemption_sha256=START_SHA,
            worker_spec_sha256=WORKER_SHA,
            ownership_record_sha256='b' * 64,
            ownership_authentication_hmac_sha256='0' * 64,
            process_group_leader_start_time_ticks=(123456 if kind == 'process_group' else None),
            process_group_session_id=(4000 if kind == 'process_group' else None),
            process_identity_source=('durable-jailer-group' if kind == 'process_group' else None),
            process_witness_pid=(4243 if kind == 'process_group' else None),
            process_witness_start_time_ticks=(123466 if kind == 'process_group' else None),
            path_device_id=(None if kind == 'process_group' else 123),
            path_inode=(None if kind == 'process_group' else 456),
            process_cgroup_device_id=(123 if kind == 'process_group' else None),
            process_cgroup_inode=(456 if kind == 'process_group' else None),
        )
        for kind, identifier in values
    )
    return tuple(
        item.model_copy(
            update={
                'ownership_authentication_hmac_sha256': managed_clinical_ownership_hmac(
                    item,
                    key=KEY,
                )
            }
        )
        for item in unsigned
    )


def _capability() -> ManagedClinicalCapability:
    unsigned = ManagedClinicalCapability(
        capability_id='c' * 64,
        run_id=RUN_ID,
        registry_authority_id=AUTHORITY,
        reservation_sha256=RESERVATION_SHA,
        launch_sha256=LAUNCH_SHA,
        start_redemption_sha256=START_SHA,
        worker_spec_sha256=WORKER_SHA,
        ownership_record_sha256='b' * 64,
        ownership_authentication_hmac_sha256='0' * 64,
    )
    return unsigned.model_copy(
        update={
            'ownership_authentication_hmac_sha256': managed_clinical_ownership_hmac(
                unsigned,
                key=KEY,
            )
        }
    )


def _resign(item: ManagedClinicalHostArtifact) -> ManagedClinicalHostArtifact:
    unsigned = item.model_copy(update={'ownership_authentication_hmac_sha256': '0' * 64})
    return unsigned.model_copy(
        update={
            'ownership_authentication_hmac_sha256': managed_clinical_ownership_hmac(
                unsigned,
                key=KEY,
            )
        }
    )


def _config(tmp_path: Path) -> ManagedClinicalStartupConfig:
    return ManagedClinicalStartupConfig(
        reconciler_id='vaxreplay-managed-startup',
        reconciler_version='dev-v0.1',
        registry_authority_id=AUTHORITY,
        runtime_config_sha256=RUNTIME_SHA,
        worker_spec_sha256=WORKER_SHA,
        cleanup_receipt_key_id=managed_clinical_cleanup_key_id(KEY),
        cgroup_root=str(tmp_path / 'cgroups'),
        jail_root=str(tmp_path / 'jails'),
        vsock_root=str(tmp_path / 'jails'),
        receipt_root=str(tmp_path / 'receipts'),
        cleanup_grace_seconds=5,
    )


def _reconciler(
    tmp_path: Path,
    *,
    artifacts: tuple[ManagedClinicalHostArtifact, ...] | None = None,
    retain_jail: bool = False,
    include_capability: bool = True,
) -> tuple[ManagedClinicalStartupReconciler, list[str]]:
    events: list[str] = []
    reconciler = ManagedClinicalStartupReconciler(
        config=_config(tmp_path),
        host=_Host(
            _artifacts(tmp_path) if artifacts is None else artifacts,
            events,
            retain_jail=retain_jail,
        ),
        capabilities=_Capabilities((_capability(),) if include_capability else (), events),
        attempts=_Attempts((_attempt(),)),
        cleanup_receipt_key=KEY,
        clock=lambda: NOW,
    )
    return reconciler, events


def test_managed_startup_cleans_in_safe_order_and_persists_authenticated_receipt(
    tmp_path: Path,
) -> None:
    reconciler, events = _reconciler(tmp_path)
    request = _request()

    receipt = reconciler.reconcile(request)

    assert events == [
        f'terminate:{RUN_ID}',
        f'reap:{RUN_ID}',
        f'revoke:{RUN_ID}',
        f'vsock:{RUN_ID}',
        f'cgroup:{RUN_ID}',
        f'jail:{RUN_ID}',
        f'finalize:{RUN_ID}',
    ]
    assert receipt.discovered_worker_count == 1
    assert receipt.discovered_capability_count == 1
    assert receipt.discovered_ephemeral_run_artifact_count == 3
    artifact = reconciler.last_authenticated_receipt
    assert artifact is not None
    assert (
        verify_authenticated_managed_cleanup(
            artifact,
            key=KEY,
            expected_key_id=managed_clinical_cleanup_key_id(KEY),
            expected_config_sha256=managed_clinical_startup_config_sha256(reconciler.config),
            expected_request_sha256=firecracker_clinical_startup_reconciliation_request_sha256(request),
        )
        == receipt
    )
    path = Path(artifact.persisted_path)
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (
        load_authenticated_managed_cleanup(
            path,
            expected_root=Path(reconciler.config.receipt_root),
        )
        == artifact
    )


def test_managed_startup_fails_closed_on_unowned_or_escaped_artifact(tmp_path: Path) -> None:
    artifacts = list(_artifacts(tmp_path))
    artifacts[1] = artifacts[1].model_copy(update={'artifact_id': str(tmp_path / 'outside' / RUN_ID)})
    reconciler, events = _reconciler(tmp_path, artifacts=tuple(artifacts))

    with pytest.raises(ManagedClinicalStartupError, match='outside its owned root'):
        reconciler.reconcile(_request())

    assert events == []
    assert reconciler.last_authenticated_receipt is None


def test_managed_startup_fails_closed_on_conflicting_ownership_records(tmp_path: Path) -> None:
    artifacts = list(_artifacts(tmp_path))
    artifacts[-1] = _resign(artifacts[-1].model_copy(update={'ownership_record_sha256': 'd' * 64}))
    reconciler, events = _reconciler(tmp_path, artifacts=tuple(artifacts))

    with pytest.raises(ManagedClinicalStartupError, match='ambiguous per-run ownership'):
        reconciler.reconcile(_request())

    assert events == []


def test_managed_startup_rejects_forged_exact_artifact_ownership(tmp_path: Path) -> None:
    artifacts = list(_artifacts(tmp_path))
    artifacts[0] = artifacts[0].model_copy(update={'process_group_leader_start_time_ticks': 654321})
    reconciler, events = _reconciler(tmp_path, artifacts=tuple(artifacts))

    with pytest.raises(ManagedClinicalStartupError, match='unauthenticated ownership'):
        reconciler.reconcile(_request())

    assert events == []


def test_managed_startup_rejects_unsafe_process_group_even_when_authenticated(
    tmp_path: Path,
) -> None:
    artifacts = list(_artifacts(tmp_path))
    artifacts[0] = _resign(artifacts[0].model_copy(update={'artifact_id': 'pgid:0'}))
    reconciler, events = _reconciler(tmp_path, artifacts=tuple(artifacts))

    with pytest.raises(ManagedClinicalStartupError, match='unsafe exact ID'):
        reconciler.reconcile(_request())

    assert events == []


def test_managed_startup_fails_closed_when_post_cleanup_scan_is_not_empty(tmp_path: Path) -> None:
    reconciler, _ = _reconciler(tmp_path, retain_jail=True)

    with pytest.raises(ManagedClinicalStartupError, match='surviving artifacts'):
        reconciler.reconcile(_request())

    assert reconciler.last_authenticated_receipt is None
    assert not (tmp_path / 'receipts').exists()


def test_managed_startup_receipt_tampering_is_rejected(tmp_path: Path) -> None:
    reconciler, _ = _reconciler(tmp_path)
    request = _request()
    reconciler.reconcile(request)
    artifact = reconciler.last_authenticated_receipt
    assert artifact is not None

    forged = artifact.model_copy(update={'cleanup_hmac_sha256': '0' * 64})
    with pytest.raises(ManagedClinicalStartupError, match='authentication failed'):
        verify_authenticated_managed_cleanup(
            forged,
            key=KEY,
            expected_key_id=managed_clinical_cleanup_key_id(KEY),
            expected_config_sha256=managed_clinical_startup_config_sha256(reconciler.config),
            expected_request_sha256=firecracker_clinical_startup_reconciliation_request_sha256(request),
        )


def test_managed_startup_receipt_is_create_once_for_same_request(tmp_path: Path) -> None:
    first, _ = _reconciler(tmp_path)
    first.reconcile(_request())
    second, _ = _reconciler(tmp_path, artifacts=(), include_capability=False)

    with pytest.raises(ManagedClinicalStartupError, match='already used'):
        second.reconcile(_request())


def test_managed_startup_rejects_receipt_root_with_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / 'real-parent'
    real_parent.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real_parent, target_is_directory=True)
    config = _config(tmp_path).model_copy(update={'receipt_root': str(alias / 'receipts')})
    events: list[str] = []
    reconciler = ManagedClinicalStartupReconciler(
        config=config,
        host=_Host((), events),
        capabilities=_Capabilities((), events),
        attempts=_Attempts(()),
        cleanup_receipt_key=KEY,
        clock=lambda: NOW,
    )

    with pytest.raises(ManagedClinicalStartupError, match='symbolic-link components'):
        reconciler.reconcile(_request())

    assert events == []
    assert reconciler.last_authenticated_receipt is None
    assert not (real_parent / 'receipts').exists()


def test_managed_startup_receipt_recovers_from_torn_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    reconciler = ManagedClinicalStartupReconciler(
        config=_config(tmp_path),
        host=_Host((), events),
        capabilities=_Capabilities((), events),
        attempts=_Attempts(()),
        cleanup_receipt_key=KEY,
        clock=lambda: NOW,
    )
    real_write = os.write
    write_count = 0

    def torn_write(descriptor: int, content: bytes) -> int:
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            return real_write(descriptor, content[:5])
        raise OSError('injected torn cleanup-receipt write')

    monkeypatch.setattr(os, 'write', torn_write)
    with pytest.raises(ManagedClinicalStartupError, match='unavailable or already used'):
        reconciler.reconcile(_request())
    receipt_root = Path(reconciler.config.receipt_root)
    assert not tuple(receipt_root.glob('*.json'))

    stale = receipt_root / ('.cleanup-stage-' + 'a' * 64)
    stale.write_bytes(b'partial')
    stale.chmod(0o600)
    monkeypatch.setattr(os, 'write', real_write)
    receipt = reconciler.reconcile(_request())

    assert receipt.discovered_worker_count == 0
    assert events == []
    assert not stale.exists()
    artifact = reconciler.last_authenticated_receipt
    assert artifact is not None
    assert (
        load_authenticated_managed_cleanup(
            Path(artifact.persisted_path),
            expected_root=receipt_root,
        )
        == artifact
    )


def test_canonical_managed_admission_observes_required_to_admitted_transition(
    tmp_path: Path,
) -> None:
    reconciler, _ = _reconciler(tmp_path)

    class _Runtime:
        startup_reconciliation_required = True

        def reconcile_startup(self, *, reconciler):  # noqa: ANN001
            receipt = reconciler.reconcile(_request())
            self.startup_reconciliation_required = False
            return FirecrackerClinicalStartupReconciliationReport(
                request=_request(),
                cleanup_receipt=receipt,
            )

    runtime = _Runtime()
    admission = reconcile_canonical_managed_runtime_startup(
        cast(FirecrackerClinicalRuntime, runtime),
        reconciler=reconciler,
    )

    assert admission.runtime_preparation_admitted
    assert runtime.startup_reconciliation_required is False
