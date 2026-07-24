from __future__ import annotations

import hashlib
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import vaxreplay.agentic.clinical_guest_executable as guest_module
from vaxreplay.agentic.clinical_guest_bootstrap import (
    CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
    ClinicalGuestBootstrapReplayGuard,
    ClinicalGuestBootstrapTrustAnchor,
    ClinicalGuestRpcLimits,
    clinical_guest_bootstrap_authorization_key_id,
)
from vaxreplay.agentic.clinical_guest_executable import (
    LANE_A_CLINICAL_GUEST_CONFIG_PATH,
    LINUX_REBOOT_CMD_POWER_OFF,
    LINUX_VMADDR_CID_HOST,
    LaneAClinicalGuestConfig,
    LaneAClinicalGuestError,
    LaneAClinicalGuestExitCode,
    connect_lane_a_clinical_guest_vsock,
    execute_baked_lane_a_clinical_guest,
    lane_a_clinical_guest_config_sha256,
    linux_guest_poweroff,
    load_lane_a_clinical_guest_config,
)
from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_ACTION_SCHEMA_SHA256,
    LANE_A_GUEST_HARNESS_POLICY_ID,
    LaneAGuestHarnessResult,
)
from vaxreplay.bundle import canonical_json_bytes

_NOW = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)


def _config() -> LaneAClinicalGuestConfig:
    private_key = Ed25519PrivateKey.from_private_bytes(b'\x11' * 32)
    public_key = private_key.public_key().public_bytes_raw()
    limits = ClinicalGuestRpcLimits(
        maximum_frame_body_bytes=1024 * 1024,
        maximum_session_wire_bytes=8 * 1024 * 1024,
        maximum_requests=100,
        maximum_list_entries=100,
        maximum_read_bytes=32_768,
        maximum_search_results=20,
        maximum_submission_bytes=65_536,
    )
    anchor = ClinicalGuestBootstrapTrustAnchor(
        authorization_key_id=clinical_guest_bootstrap_authorization_key_id(public_key),
        ed25519_public_key_hex=public_key.hex(),
        execution_policy_sha256='7' * 64,
        worker_bootstrap_profile_sha256='8' * 64,
        harness_policy_id=LANE_A_GUEST_HARNESS_POLICY_ID,
        harness_policy_sha256=CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
        action_schema_sha256=LANE_A_GUEST_ACTION_SCHEMA_SHA256,
        rpc_limits=limits,
    )
    return LaneAClinicalGuestConfig(trust_anchor=anchor, guest_rpc_port=7000)


def test_canonical_config_load_requires_external_exact_digest(tmp_path: Path) -> None:
    config = _config()
    body = canonical_json_bytes(config)
    path = tmp_path / 'guest.json'
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()

    loaded = load_lane_a_clinical_guest_config(path, expected_sha256=digest)

    assert loaded == config
    assert lane_a_clinical_guest_config_sha256(loaded) == digest


@pytest.mark.parametrize('mutation', ['wrong-hash', 'noncanonical', 'extra-field'])
def test_config_loader_rejects_unpinned_or_noncanonical_bytes(tmp_path: Path, mutation: str) -> None:
    config = _config()
    body = canonical_json_bytes(config)
    if mutation == 'noncanonical':
        body += b'\n'
    elif mutation == 'extra-field':
        body = body[:-1] + b',"unexpected":true}'
    path = tmp_path / 'guest.json'
    path.write_bytes(body)
    digest = 'f' * 64 if mutation == 'wrong-hash' else hashlib.sha256(body).hexdigest()

    with pytest.raises(LaneAClinicalGuestError, match='configuration rejected'):
        load_lane_a_clinical_guest_config(path, expected_sha256=digest)


def test_config_loader_rejects_symlink(tmp_path: Path) -> None:
    body = canonical_json_bytes(_config())
    target = tmp_path / 'real.json'
    target.write_bytes(body)
    link = tmp_path / 'guest.json'
    link.symlink_to(target)

    with pytest.raises(LaneAClinicalGuestError, match='configuration rejected'):
        load_lane_a_clinical_guest_config(link, expected_sha256=hashlib.sha256(body).hexdigest())


def test_config_has_fixed_host_transport_and_no_alternate_surfaces() -> None:
    wire = _config().model_dump(mode='json')

    assert wire['host_cid'] == LINUX_VMADDR_CID_HOST
    assert wire['address_family'] == 'AF_VSOCK'
    assert wire['socket_type'] == 'SOCK_STREAM'
    assert wire['one_connection'] is True
    assert wire['signed_bootstrap_and_rpc_share_socket'] is True
    assert wire['runtime_endpoint_negotiation_allowed'] is False
    assert wire['ip_network_allowed'] is False
    assert wire['shell_allowed'] is False
    assert wire['ambient_credentials_allowed'] is False
    assert wire['automatic_retry_allowed'] is False
    assert wire['task_content_logging_allowed'] is False
    assert wire['poweroff_after_terminal_result_required'] is True


class _FakeSocket:
    def __init__(self, *, fail_connect: bool = False) -> None:
        self.fail_connect = fail_connect
        self.inheritable: bool | None = None
        self.timeouts: list[float | None] = []
        self.addresses: list[tuple[int, int]] = []
        self.closed = False

    def set_inheritable(self, inheritable: bool) -> None:
        self.inheritable = inheritable

    def settimeout(self, timeout: float | None) -> None:
        self.timeouts.append(timeout)

    def connect(self, address: tuple[int, int]) -> None:
        self.addresses.append(address)
        if self.fail_connect:
            raise OSError('secret endpoint error')

    def close(self) -> None:
        self.closed = True


def test_vsock_connector_uses_one_fixed_linux_host_connection() -> None:
    created: list[tuple[int, int]] = []
    fake = _FakeSocket()

    def factory(family: int, kind: int) -> socket.socket:
        created.append((family, kind))
        return cast(socket.socket, fake)

    connection = connect_lane_a_clinical_guest_vsock(
        _config(),
        socket_factory=factory,
        platform='linux',
        af_vsock=40,
        vmaddr_cid_host=LINUX_VMADDR_CID_HOST,
    )

    assert connection is cast(socket.socket, fake)
    assert created == [(40, socket.SOCK_STREAM)]
    assert fake.addresses == [(LINUX_VMADDR_CID_HOST, 7000)]
    assert fake.timeouts == [5.0, None]
    assert fake.inheritable is False


def test_vsock_connector_fails_closed_without_linux_vsock_and_does_not_retry() -> None:
    calls = 0

    def factory(family: int, kind: int) -> socket.socket:
        nonlocal calls
        del family, kind
        calls += 1
        return cast(socket.socket, _FakeSocket())

    with pytest.raises(LaneAClinicalGuestError, match='connection rejected'):
        connect_lane_a_clinical_guest_vsock(
            _config(),
            socket_factory=factory,
            platform='darwin',
            af_vsock=40,
            vmaddr_cid_host=LINUX_VMADDR_CID_HOST,
        )
    assert calls == 0


def test_vsock_connector_closes_single_failed_connection() -> None:
    fake = _FakeSocket(fail_connect=True)

    with pytest.raises(LaneAClinicalGuestError, match='connection rejected'):
        connect_lane_a_clinical_guest_vsock(
            _config(),
            socket_factory=lambda family, kind: cast(socket.socket, fake),
            platform='linux',
            af_vsock=40,
            vmaddr_cid_host=LINUX_VMADDR_CID_HOST,
        )

    assert fake.addresses == [(LINUX_VMADDR_CID_HOST, 7000)]
    assert fake.closed is True


def test_appliance_runs_entry_on_same_socket_then_closes_and_powers_off() -> None:
    config = _config()
    guest, peer = socket.socketpair()
    observed: dict[str, object] = {}
    poweroffs: list[bool] = []
    guard = cast(ClinicalGuestBootstrapReplayGuard, object())

    def loader(path: Path, *, expected_sha256: str) -> LaneAClinicalGuestConfig:
        observed['path'] = path
        observed['digest'] = expected_sha256
        return config

    def entry(
        connection: socket.socket,
        *,
        trust_anchor: ClinicalGuestBootstrapTrustAnchor,
        replay_guard: ClinicalGuestBootstrapReplayGuard,
        clock,
        timeout_seconds: float,
    ) -> LaneAGuestHarnessResult:
        observed['connection'] = connection
        observed['anchor'] = trust_anchor
        observed['guard'] = replay_guard
        observed['time'] = clock()
        observed['timeout'] = timeout_seconds
        return cast(LaneAGuestHarnessResult, object())

    try:
        result = execute_baked_lane_a_clinical_guest(
            'a' * 64,
            config_loader=loader,
            connector=lambda config: guest,
            guest_entry=entry,
            replay_guard=guard,
            poweroff=lambda: poweroffs.append(True),
            clock=lambda: _NOW,
        )
    finally:
        peer.close()

    assert result == LaneAClinicalGuestExitCode.SUCCESS
    assert observed == {
        'path': LANE_A_CLINICAL_GUEST_CONFIG_PATH,
        'digest': 'a' * 64,
        'connection': guest,
        'anchor': config.trust_anchor,
        'guard': guard,
        'time': _NOW,
        'timeout': 5.0,
    }
    assert guest.fileno() == -1
    assert poweroffs == [True]


@pytest.mark.parametrize('phase', ['config', 'connect', 'entry'])
def test_appliance_powers_off_once_after_each_terminal_failure(phase: str) -> None:
    config = _config()
    guest, peer = socket.socketpair()
    poweroffs: list[bool] = []

    def loader(path: Path, *, expected_sha256: str) -> LaneAClinicalGuestConfig:
        del path, expected_sha256
        if phase == 'config':
            raise RuntimeError('task-content-secret')
        return config

    def connector(config: LaneAClinicalGuestConfig) -> socket.socket:
        del config
        if phase == 'connect':
            raise RuntimeError('socket-secret')
        return guest

    def entry(*args, **kwargs) -> LaneAGuestHarnessResult:
        del args, kwargs
        if phase == 'entry':
            raise RuntimeError('model-output-secret')
        return cast(LaneAGuestHarnessResult, object())

    try:
        result = execute_baked_lane_a_clinical_guest(
            'a' * 64,
            config_loader=loader,
            connector=connector,
            guest_entry=entry,
            poweroff=lambda: poweroffs.append(True),
        )
    finally:
        peer.close()
        guest.close()

    expected = (
        LaneAClinicalGuestExitCode.CONFIGURATION_REJECTED
        if phase == 'config'
        else LaneAClinicalGuestExitCode.EXECUTION_FAILED
    )
    assert result == expected
    assert poweroffs == [True]


def test_appliance_poweroff_failure_is_terminal() -> None:
    def fail_poweroff() -> None:
        raise RuntimeError('capability detail must not escape')

    result = execute_baked_lane_a_clinical_guest(
        'a' * 64,
        config_loader=lambda path, *, expected_sha256: (_ for _ in ()).throw(ValueError('secret')),
        poweroff=fail_poweroff,
    )

    assert result == LaneAClinicalGuestExitCode.POWEROFF_FAILED


def test_appliance_constructs_exactly_one_process_local_replay_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    guest, peer = socket.socketpair()
    created: list[ClinicalGuestBootstrapReplayGuard] = []
    observed: list[ClinicalGuestBootstrapReplayGuard] = []

    class Guard:
        def consume(self, *, nonce: str, hello_sha256: str) -> bool:
            del nonce, hello_sha256
            return True

    def guard_factory() -> ClinicalGuestBootstrapReplayGuard:
        guard = Guard()
        created.append(guard)
        return guard

    def entry(connection: socket.socket, **kwargs) -> LaneAGuestHarnessResult:
        del connection
        observed.append(kwargs['replay_guard'])
        return cast(LaneAGuestHarnessResult, object())

    monkeypatch.setattr(guest_module, 'InMemoryClinicalGuestBootstrapReplayGuard', guard_factory)
    try:
        result = execute_baked_lane_a_clinical_guest(
            'a' * 64,
            config_loader=lambda path, *, expected_sha256: config,
            connector=lambda config: guest,
            guest_entry=entry,
            poweroff=lambda: None,
        )
    finally:
        peer.close()

    assert result == LaneAClinicalGuestExitCode.SUCCESS
    assert len(created) == 1
    assert observed == created


def test_linux_poweroff_uses_direct_reboot_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    class FakeReboot:
        argtypes: object = None
        restype: object = None

        def __call__(self, command: int) -> int:
            calls.append(command)
            return -1

    class FakeLibc:
        reboot = FakeReboot()

    monkeypatch.setattr(guest_module.sys, 'platform', 'linux')
    monkeypatch.setattr(guest_module.ctypes, 'CDLL', lambda *args, **kwargs: FakeLibc())
    monkeypatch.setattr(guest_module.os, 'sync', lambda: calls.append('sync'))

    with pytest.raises(LaneAClinicalGuestError, match='poweroff rejected'):
        linux_guest_poweroff()

    assert calls == ['sync', LINUX_REBOOT_CMD_POWER_OFF]


def test_cli_emits_only_fixed_content_free_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    observed: list[str] = []

    def execute(digest: str) -> LaneAClinicalGuestExitCode:
        observed.append(digest)
        return LaneAClinicalGuestExitCode.EXECUTION_FAILED

    monkeypatch.setattr(guest_module, 'execute_baked_lane_a_clinical_guest', execute)

    with pytest.raises(SystemExit) as caught:
        guest_module.main(['--expected-config-sha256', 'a' * 64])

    assert caught.value.code == 70
    assert observed == ['a' * 64]
    assert capsys.readouterr().err == 'lane-a clinical guest terminated: bounded execution failed\n'


def test_cli_does_not_echo_malformed_arguments(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    observed: list[str] = []

    def execute(digest: str) -> LaneAClinicalGuestExitCode:
        observed.append(digest)
        return LaneAClinicalGuestExitCode.CONFIGURATION_REJECTED

    monkeypatch.setattr(guest_module, 'execute_baked_lane_a_clinical_guest', execute)

    with pytest.raises(SystemExit) as caught:
        guest_module.main(['--host', 'sensitive-task-content.example'])

    assert caught.value.code == 64
    assert observed == ['']
    output = capsys.readouterr()
    assert output.out == ''
    assert output.err == 'lane-a clinical guest rejected: baked configuration is invalid\n'
    assert 'sensitive' not in output.err
