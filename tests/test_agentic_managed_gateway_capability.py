from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.test_agentic_managed_clinical_ownership import KEY, _ownership_stack
from tests.test_agentic_provider_gateway import _make_fixture
from vaxreplay.agentic.managed_clinical_ownership import (
    ManagedClinicalOwnershipError,
)
from vaxreplay.agentic.managed_gateway_capability import (
    RestartVisibleManagedGatewayCapabilityLedger,
)
from vaxreplay.agentic.provider_gateway import SqliteGatewayLedger


def _stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    register_gateway_session: bool = True,
    advance_running: bool = True,
    after_local_revocation=None,  # noqa: ANN001
):
    ownership, _host, _old_capabilities, _start, _artifacts, _old_revoked = _ownership_stack(
        tmp_path / 'ownership',
        monkeypatch,
        advance_running=advance_running,
    )
    record = ownership.active()[0].record
    provider = _make_fixture(tmp_path / 'provider-fixture')
    gateway_path = tmp_path / 'managed-gateway' / 'gateway.sqlite3'
    gateway = SqliteGatewayLedger(gateway_path)
    grant = provider.grant.model_copy(
        update={
            'capability_id': record.capability_id,
            'run_id': record.run_id,
            # Current runtime deliberately puts the redeemed-start hash in this field.
            'attempt_reservation_sha256': record.start_redemption_sha256,
        }
    )
    if register_gateway_session:
        gateway.register(grant, provider.route, provider.policy)
    bridge = RestartVisibleManagedGatewayCapabilityLedger(
        ownership=ownership,
        ownership_key=KEY,
        gateway_ledger=gateway,
        expected_model_route_sha256=grant.model_route_sha256,
        after_local_revocation=after_local_revocation,
        clock=lambda: datetime(2025, 1, 2, 3, 4, 6, tzinfo=UTC),
    )
    return ownership, gateway, bridge, grant, provider


def test_registered_cleanup_orders_tombstone_callback_and_ownership_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def external_cleanup(capability_id: str) -> None:
        assert gateway.capability_revocation(capability_id) is not None
        assert ownership.latest(grant.run_id).record.state == 'running'
        events.append('external')

    ownership, gateway, bridge, grant, _provider = _stack(
        tmp_path,
        monkeypatch,
        after_local_revocation=external_cleanup,
    )
    capability = bridge.inventory()[0]

    bridge.revoke(capability)

    assert events == ['external']
    assert gateway.capability_revocation(capability.capability_id) is not None
    assert ownership.latest(grant.run_id).record.state == 'capability_revoked'
    assert bridge.inventory() == ()


def test_callback_crash_after_local_tombstone_is_restart_replay_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crash_after_local_tombstone(capability_id: str) -> None:
        assert gateway.capability_revocation(capability_id) is not None
        raise RuntimeError('simulated external cleanup crash')

    ownership, gateway, bridge, grant, _provider = _stack(
        tmp_path,
        monkeypatch,
        after_local_revocation=crash_after_local_tombstone,
    )
    capability = bridge.inventory()[0]

    with pytest.raises(RuntimeError, match='simulated external cleanup crash'):
        bridge.revoke(capability)

    assert gateway.capability_revocation(capability.capability_id) is not None
    assert ownership.latest(grant.run_id).record.state == 'running'
    observed: list[str] = []
    restarted = RestartVisibleManagedGatewayCapabilityLedger(
        ownership=ownership,
        ownership_key=KEY,
        gateway_ledger=SqliteGatewayLedger(gateway.path),
        expected_model_route_sha256=grant.model_route_sha256,
        after_local_revocation=observed.append,
        clock=lambda: datetime(2025, 1, 2, 3, 4, 7, tzinfo=UTC),
    )
    retry = restarted.inventory()[0]
    restarted.revoke(retry)

    assert observed == [capability.capability_id]
    assert ownership.latest(grant.run_id).record.state == 'capability_revoked'


def test_pre_registration_tombstone_prevents_later_session_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real pre-registration crash leaves start_bound rather than running ownership. The gateway
    # API test covers that exact transition; here we exercise the bridge's absent-session revoke
    # path using an authenticated owned capability and then prove resurrection is impossible.
    ownership, gateway, bridge, grant, provider = _stack(
        tmp_path,
        monkeypatch,
        register_gateway_session=False,
        advance_running=False,
    )
    capability = bridge.inventory()[0]

    bridge.revoke(capability)

    assert ownership.latest(grant.run_id).record.state == 'capability_revoked'
    with pytest.raises(ValueError, match='durable revocation tombstone'):
        gateway.register(grant, provider.route, provider.policy)


def test_inventory_fails_closed_on_rogue_gateway_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ownership, gateway, bridge, grant, provider = _stack(tmp_path, monkeypatch)
    rogue = provider.grant.model_copy(
        update={
            'capability_id': 'e' * 64,
            'run_id': 'e' * 32,
            'attempt_reservation_sha256': 'e' * 64,
        }
    )
    gateway.register(rogue, provider.route, provider.policy)

    with pytest.raises(ManagedClinicalOwnershipError, match='unowned untombstoned'):
        bridge.inventory()
    assert gateway.capability_revocation(grant.capability_id) is None


def test_inventory_rejects_ownership_successor_without_gateway_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ownership, gateway, bridge, grant, _provider = _stack(tmp_path, monkeypatch)
    capability = bridge.inventory()[0]
    ownership.record_capability_revoked(
        run_id=grant.run_id,
        capability_id=capability.capability_id,
    )
    record = ownership.latest(grant.run_id).record
    shutil.rmtree(record.run_container_path)
    shutil.rmtree(record.cgroup_path)
    shutil.rmtree(Path(ownership.config.proc_root) / str(record.firecracker_pid))
    ownership.record_cleaned(
        run_id=grant.run_id,
        terminal_reason='runtime_cleanup',
    )
    assert ownership.active() == ()

    with pytest.raises(ManagedClinicalOwnershipError, match='unowned untombstoned'):
        bridge.inventory()
    assert gateway.capability_revocation(capability.capability_id) is None


def test_registered_binding_uses_redeemed_start_not_outer_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ownership, _gateway, bridge, grant, _provider = _stack(tmp_path, monkeypatch)
    record = ownership.latest(grant.run_id).record

    assert record.start_redemption_sha256 is not None
    assert grant.attempt_reservation_sha256 == record.start_redemption_sha256
    assert grant.attempt_reservation_sha256 != record.reservation_sha256
    assert bridge.inventory()[0].start_redemption_sha256 == grant.attempt_reservation_sha256
