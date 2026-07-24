from __future__ import annotations

import os
import shutil
import socket
import stat
import sys
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

import vaxreplay.agentic.firecracker_clinical_runtime as runtime_module
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
from tests.test_agentic_clinical_production_run_v02 import (
    BOOTSTRAP_AUTHORIZATION_KEY_ID,
    BOOTSTRAP_RECEIPT_KEY,
    BOOTSTRAP_SIGNER,
    _bootstrap,
)
from vaxreplay.agentic.clinical_guest_bootstrap import (
    AuthenticatedClinicalGuestBootstrap,
    ClinicalGuestBootstrapHello,
    ClinicalGuestBootstrapTrustAnchor,
    InMemoryClinicalGuestBootstrapReplayGuard,
    clinical_guest_bootstrap_receipt_key_id,
    perform_guest_clinical_bootstrap,
    perform_host_clinical_guest_bootstrap,
)
from vaxreplay.agentic.clinical_launcher import (
    CanonicalClinicalLauncherDeployment,
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
    AuthenticatedFirecrackerWorkerAttestation,
    FirecrackerCleanupReceipt,
    FirecrackerHostPreflightReceipt,
    FirecrackerPreparedWorker,
    FirecrackerWorkerSpec,
    RunningFirecrackerWorker,
    firecracker_attestation_key_id,
    firecracker_model_sha256,
    firecracker_static_config_bytes,
)
from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalGuestBootstrapSession,
    FirecrackerClinicalRuntime,
    FirecrackerClinicalRuntimeConfig,
    FirecrackerClinicalRuntimeError,
    FirecrackerClinicalRuntimeKeys,
    firecracker_clinical_runtime_config_sha256,
    verify_linux_unix_peer_identity,
)
from vaxreplay.agentic.gateway_auth import gateway_capability_id
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
    GatewayTerminalReason,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    gateway_session_key_id,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split

_PROVIDER_SUBPROCESS_SPEC_SHA256 = '9' * 64
_PROVIDER_SUBPROCESS_BEHAVIOR_SHA256 = 'b' * 64
_PROVIDER_SUBPROCESS_MODULE_SOURCE_SHA256 = 'a' * 64


def _short_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix='vr-', dir='/tmp'))
    (root / 'run').mkdir(mode=0o700)
    return root


def _prepared(root: Path, materials: Materials) -> FirecrackerPreparedWorker:
    return cast(
        FirecrackerPreparedWorker,
        SimpleNamespace(
            run_id=RUN_ID,
            worker_spec_sha256=firecracker_model_sha256(materials.spec),
            vsock_uds_path=str(root / 'run' / 'vsock.sock'),
        ),
    )


def _session(materials: Materials) -> GuestRpcHostSession:
    return cast(
        GuestRpcHostSession,
        SimpleNamespace(
            run_id=RUN_ID,
            worker_spec_sha256=firecracker_model_sha256(materials.spec),
            rpc_port=materials.spec.guest_rpc_port,
            gateway_grant=materials.gateway.grant,
            session_id=materials.guest.seal.session_id,
            task_invocation=materials.workspace.invocation,
            workspace_manifest_sha256=materials.workspace.manifest_sha256,
            workspace_tree_sha256=materials.workspace.manifest.workspace_tree_sha256,
            model_visible_surface_sha256=materials.workspace.manifest.model_visible_surface_sha256,
            execution_policy_sha256=agentic_policy_sha256(materials.policy),
            policy=materials.guest.policy,
        ),
    )


def _runner(root: Path, materials: Materials) -> FirecrackerClinicalGuestBootstrapSession:
    os.chown(root / 'run', materials.spec.worker_uid, materials.spec.worker_gid)
    peer_verifier = verify_linux_unix_peer_identity
    if (os.geteuid(), os.getegid()) != (
        materials.spec.worker_uid,
        materials.spec.worker_gid,
    ):
        # A root-run unit test cannot make its in-process client thread become the non-root jailer.
        # The production default remains independently covered by the Linux SO_PEERCRED test.
        def ignore_peer_identity(
            _connection: socket.socket,
            _pid: int,
            _uid: int,
            _gid: int,
        ) -> None:
            pass

        peer_verifier = ignore_peer_identity
    return FirecrackerClinicalGuestBootstrapSession(
        prepared=_prepared(root, materials),
        rpc_port=materials.spec.guest_rpc_port,
        worker_uid=materials.spec.worker_uid,
        worker_gid=materials.spec.worker_gid,
        authorization_signer=BOOTSTRAP_SIGNER,
        expected_authorization_key_id=BOOTSTRAP_AUTHORIZATION_KEY_ID,
        receipt_key=BOOTSTRAP_RECEIPT_KEY,
        journal_authenticated_bootstrap=lambda artifact: sha256(canonical_json_bytes(artifact)).hexdigest(),
        clock=lambda: (
            materials.worker.attestation.started_at.replace(microsecond=0)
            + (materials.worker.attestation.finished_at - materials.worker.attestation.started_at) / 5
        ),
        connection_timeout_seconds=1,
        monotonic_clock=lambda: 1.0,
        peer_identity_verifier=peer_verifier,
    )


def _artifact_for_hello(
    hello: ClinicalGuestBootstrapHello,
    *,
    anchor: ClinicalGuestBootstrapTrustAnchor,
    now: datetime,
) -> AuthenticatedClinicalGuestBootstrap:
    host, guest = socket.socketpair()
    outcome: dict[str, object] = {}

    def guest_bootstrap() -> None:
        try:
            outcome['context'] = perform_guest_clinical_bootstrap(
                guest,
                trust_anchor=anchor,
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: now,
                timeout_seconds=1,
            )
        except BaseException as error:
            outcome['error'] = error

    thread = threading.Thread(target=guest_bootstrap)
    thread.start()
    try:
        artifact = perform_host_clinical_guest_bootstrap(
            host,
            hello=hello,
            authorization_signer=BOOTSTRAP_SIGNER,
            expected_authorization_key_id=BOOTSTRAP_AUTHORIZATION_KEY_ID,
            receipt_key=BOOTSTRAP_RECEIPT_KEY,
            clock=lambda: now,
            timeout_seconds=1,
        )
        thread.join(timeout=2)
    finally:
        host.close()
        guest.close()
    assert not thread.is_alive()
    if 'error' in outcome:
        error = outcome['error']
        assert isinstance(error, BaseException)
        raise AssertionError('runtime bootstrap fixture failed') from error
    return artifact


def test_runtime_socket_uses_one_stream_for_signed_bootstrap_then_rpc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materials = _materials(tmp_path / 'materials')
    expected_artifact, anchor = _bootstrap(materials)
    hello = expected_artifact.signed_hello.hello
    root = _short_root()
    runner = _runner(root, materials)
    served: list[GuestRpcHostSession] = []

    class FakeRpcServer:
        def __init__(self, session: GuestRpcHostSession) -> None:
            served.append(session)

        def serve(self, connection: socket.socket) -> None:
            connection.sendall(b'R')

    monkeypatch.setattr(runtime_module, 'GuestRpcHostServer', FakeRpcServer)
    runner.open()
    socket_metadata = runner.socket_path.stat()
    assert stat.S_IMODE(socket_metadata.st_mode) == 0o600
    assert (socket_metadata.st_uid, socket_metadata.st_gid) == (
        materials.spec.worker_uid,
        materials.spec.worker_gid,
    )
    outcome: dict[str, object] = {}

    def guest() -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(runner.socket_path))
            outcome['context'] = perform_guest_clinical_bootstrap(
                connection,
                trust_anchor=anchor,
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: (
                    materials.worker.attestation.started_at
                    + (materials.worker.attestation.finished_at - materials.worker.attestation.started_at) / 5
                ),
                timeout_seconds=1,
            )
            outcome['rpc_marker'] = connection.recv(1)
        except BaseException as error:
            outcome['error'] = error
        finally:
            connection.close()

    thread = threading.Thread(target=guest)
    thread.start()
    try:
        artifact = runner.serve_one(
            hello=hello,
            session=_session(materials),
            deadline_monotonic=2.0,
            expected_peer_pid=os.getpid(),
        )
        thread.join(timeout=2)
    finally:
        runner.close()
        shutil.rmtree(root)

    assert not thread.is_alive()
    assert 'error' not in outcome
    assert outcome['rpc_marker'] == b'R'
    assert artifact == runner.authenticated_bootstrap
    assert artifact.signed_hello == expected_artifact.signed_hello
    assert len(served) == 1
    assert not runner.socket_path.exists()


def test_runtime_socket_chowns_exact_pinned_worker_when_host_identity_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materials = _materials(tmp_path / 'materials')
    root = _short_root()
    runner = _runner(root, materials)
    actual_chown = os.chown
    observed: list[tuple[Path, int, int, bool]] = []

    def recording_chown(
        path: str | os.PathLike[str],
        uid: int,
        gid: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        observed.append((Path(path), uid, gid, follow_symlinks))
        actual_chown(path, uid, gid, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(runtime_module.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(runtime_module.os, 'chown', recording_chown)
    try:
        runner.open()
        metadata = runner.socket_path.lstat()
        assert observed == [
            (
                runner.socket_path,
                materials.spec.worker_uid,
                materials.spec.worker_gid,
                False,
            )
        ]
        assert (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode)) == (
            materials.spec.worker_uid,
            materials.spec.worker_gid,
            0o600,
        )
    finally:
        runner.close()
        shutil.rmtree(root)


def test_runtime_socket_chown_failure_is_terminal_and_removes_bound_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materials = _materials(tmp_path / 'materials')
    root = _short_root()
    runner = _runner(root, materials)

    def fail_chown(*_args: object, **_kwargs: object) -> None:
        raise PermissionError('private ownership failure detail')

    monkeypatch.setattr(runtime_module.os, 'chown', fail_chown)
    try:
        with pytest.raises(FirecrackerClinicalRuntimeError, match='could not be opened'):
            runner.open()
        assert not runner.socket_path.exists()
    finally:
        runner.close()
        shutil.rmtree(root)


def test_runtime_socket_rejects_worker_writable_parent_with_wrong_owner_or_mode(
    tmp_path: Path,
) -> None:
    materials = _materials(tmp_path / 'materials')
    root = _short_root()
    runner = _runner(root, materials)
    (root / 'run').chmod(0o750)
    try:
        with pytest.raises(FirecrackerClinicalRuntimeError, match='pinned worker account'):
            runner.open()
        assert not runner.socket_path.exists()
    finally:
        shutil.rmtree(root)


@pytest.mark.skipif(sys.platform != 'linux', reason='SO_PEERCRED is Linux-specific')
def test_linux_unix_peer_identity_requires_exact_pid_uid_and_gid() -> None:
    host, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        verify_linux_unix_peer_identity(host, os.getpid(), os.geteuid(), os.getegid())
        with pytest.raises(FirecrackerClinicalRuntimeError, match='pinned worker process identity'):
            verify_linux_unix_peer_identity(host, os.getpid() + 1, os.geteuid(), os.getegid())
        with pytest.raises(FirecrackerClinicalRuntimeError, match='pinned worker process identity'):
            verify_linux_unix_peer_identity(host, os.getpid(), os.geteuid() + 1, os.getegid())
    finally:
        host.close()
        peer.close()


def test_unix_peer_identity_has_deterministic_exact_linux_credential_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (4242, 501, 20)
    encoded = runtime_module.struct.Struct('3i').pack(*expected)

    class CredentialSocket:
        def getsockopt(self, level: int, option: int, size: int) -> bytes:
            assert (level, option, size) == (
                socket.SOL_SOCKET,
                17,
                runtime_module.struct.Struct('3i').size,
            )
            return encoded

    monkeypatch.setattr(runtime_module.sys, 'platform', 'linux')
    monkeypatch.setattr(runtime_module.socket, 'SO_PEERCRED', 17, raising=False)
    connection = cast(socket.socket, CredentialSocket())
    verify_linux_unix_peer_identity(connection, *expected)
    for observed in ((4243, 501, 20), (4242, 502, 20), (4242, 501, 21)):
        with pytest.raises(FirecrackerClinicalRuntimeError, match='pinned worker process identity'):
            verify_linux_unix_peer_identity(connection, *observed)


def test_unix_peer_identity_rejects_invalid_expected_pid_before_platform_seam() -> None:
    host, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(FirecrackerClinicalRuntimeError, match='process ID is invalid'):
            verify_linux_unix_peer_identity(host, 0, os.geteuid(), os.getegid())
    finally:
        host.close()
        peer.close()


def test_runtime_socket_rejects_cross_session_hello_before_accepting_a_peer(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    artifact, _ = _bootstrap(materials)
    wrong = artifact.signed_hello.hello.model_copy(update={'session_id': 'd' * 32})
    root = _short_root()
    runner = _runner(root, materials)
    runner.open()
    try:
        with pytest.raises(FirecrackerClinicalRuntimeError, match='differs from the guest RPC session'):
            runner.serve_one(
                hello=wrong,
                session=_session(materials),
                deadline_monotonic=2.0,
                expected_peer_pid=os.getpid(),
            )
    finally:
        runner.close()
        shutil.rmtree(root)


def test_runtime_socket_preserves_completed_bootstrap_when_rpc_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materials = _materials(tmp_path / 'materials')
    expected_artifact, anchor = _bootstrap(materials)
    root = _short_root()
    runner = _runner(root, materials)

    class FailingRpcServer:
        def __init__(self, _session: GuestRpcHostSession) -> None:
            pass

        def serve(self, _connection: socket.socket) -> None:
            raise RuntimeError('sensitive RPC failure body')

    monkeypatch.setattr(runtime_module, 'GuestRpcHostServer', FailingRpcServer)
    runner.open()

    def guest() -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(runner.socket_path))
            perform_guest_clinical_bootstrap(
                connection,
                trust_anchor=anchor,
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: (
                    materials.worker.attestation.started_at
                    + (materials.worker.attestation.finished_at - materials.worker.attestation.started_at) / 5
                ),
                timeout_seconds=1,
            )
        finally:
            connection.close()

    thread = threading.Thread(target=guest)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match='sensitive RPC failure body'):
            runner.serve_one(
                hello=expected_artifact.signed_hello.hello,
                session=_session(materials),
                deadline_monotonic=2.0,
                expected_peer_pid=os.getpid(),
            )
        thread.join(timeout=2)
    finally:
        runner.close()
        shutil.rmtree(root)

    assert not thread.is_alive()
    assert runner.authenticated_bootstrap is not None
    assert runner.authenticated_bootstrap.signed_hello == expected_artifact.signed_hello


def test_runtime_socket_refuses_preexisting_path_and_detects_cleanup_substitution(
    tmp_path: Path,
) -> None:
    materials = _materials(tmp_path / 'materials')
    root = _short_root()
    runner = _runner(root, materials)
    runner.socket_path.write_bytes(b'host-controlled collision')
    try:
        with pytest.raises(FirecrackerClinicalRuntimeError, match='existing socket'):
            runner.open()
    finally:
        runner.socket_path.unlink()

    runner.open()
    runner.socket_path.unlink()
    runner.socket_path.write_bytes(b'substituted after bind')
    try:
        with pytest.raises(FirecrackerClinicalRuntimeError, match='changed type'):
            runner.close()
        assert runner.socket_path.read_bytes() == b'substituted after bind'
    finally:
        runner.socket_path.unlink()
        shutil.rmtree(root)


def test_runtime_socket_detects_same_type_inode_substitution_and_preserves_replacement(
    tmp_path: Path,
) -> None:
    materials = _materials(tmp_path / 'materials')
    root = _short_root()
    runner = _runner(root, materials)
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    runner.open()
    runner.socket_path.unlink()
    replacement.bind(str(runner.socket_path))
    try:
        with pytest.raises(FirecrackerClinicalRuntimeError, match='changed identity'):
            runner.close()
        assert stat.S_ISSOCK(runner.socket_path.lstat().st_mode)
    finally:
        replacement.close()
        runner.socket_path.unlink()
        shutil.rmtree(root)


def test_runtime_socket_deadline_and_single_use_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materials = _materials(tmp_path / 'materials')
    expected_artifact, anchor = _bootstrap(materials)
    root = _short_root()
    runner = _runner(root, materials)
    runner.open()
    with pytest.raises(FirecrackerClinicalRuntimeError, match='deadline elapsed'):
        runner.serve_one(
            hello=expected_artifact.signed_hello.hello,
            session=_session(materials),
            deadline_monotonic=1.0,
            expected_peer_pid=os.getpid(),
        )

    class EmptyRpcServer:
        def __init__(self, _session: GuestRpcHostSession) -> None:
            pass

        def serve(self, _connection: socket.socket) -> None:
            pass

    monkeypatch.setattr(runtime_module, 'GuestRpcHostServer', EmptyRpcServer)

    def guest() -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(runner.socket_path))
            perform_guest_clinical_bootstrap(
                connection,
                trust_anchor=anchor,
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: (
                    materials.worker.attestation.started_at
                    + (materials.worker.attestation.finished_at - materials.worker.attestation.started_at) / 5
                ),
                timeout_seconds=1,
            )
        finally:
            connection.close()

    thread = threading.Thread(target=guest)
    thread.start()
    try:
        runner.serve_one(
            hello=expected_artifact.signed_hello.hello,
            session=_session(materials),
            deadline_monotonic=2.0,
            expected_peer_pid=os.getpid(),
        )
        with pytest.raises(FirecrackerClinicalRuntimeError, match='not open and fresh'):
            runner.serve_one(
                hello=expected_artifact.signed_hello.hello,
                session=_session(materials),
                deadline_monotonic=2.0,
                expected_peer_pid=os.getpid(),
            )
        thread.join(timeout=2)
    finally:
        runner.close()
        shutil.rmtree(root)

    assert not thread.is_alive()
    assert ATTEMPT == expected_artifact.receipt.start_redemption_sha256


class _SecretStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.secrets: dict[str, bytes] = {}
        self.revocations = 0

    def register(self, secret: bytes) -> str:
        self.events.append('secret.register')
        capability_id = gateway_capability_id(secret)
        self.secrets[capability_id] = bytes(secret)
        return capability_id

    def resolve(self, capability_id: str) -> bytes:
        return self.secrets[capability_id]

    def revoke(self, capability_id: str) -> None:
        self.events.append('secret.revoke')
        self.revocations += 1
        self.secrets.pop(capability_id, None)


class _Ownership:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def begin_preparing(
        self,
        request: ClinicalRuntimePrepareRequest,
        *,
        spec: FirecrackerWorkerSpec,
    ) -> None:
        del request, spec
        self.events.append('ownership.begin')

    def record_prepared(self, worker: FirecrackerPreparedWorker) -> None:
        del worker
        self.events.append('ownership.prepared')

    def record_start_bound(
        self,
        *,
        run_id: str,
        start: ClinicalRuntimeStart,
        capability_id: str,
    ) -> None:
        del run_id, start, capability_id
        self.events.append('ownership.start-bound')

    def record_running(self, running: RunningFirecrackerWorker) -> None:
        del running
        self.events.append('ownership.running')

    def record_capability_revoked(self, *, run_id: str, capability_id: str) -> None:
        del run_id, capability_id
        self.events.append('ownership.capability-revoked')

    def record_cleaned(
        self,
        *,
        run_id: str,
        terminal_reason: Literal['runtime_cleanup', 'startup_reaper', 'preparation_failed'],
        cleanup_receipt: FirecrackerCleanupReceipt | None = None,
    ) -> None:
        del run_id, terminal_reason, cleanup_receipt
        self.events.append('ownership.cleaned')


class _Gateway:
    provider_calls_forcibly_cancellable = True

    def __init__(self, materials: Materials, store: _SecretStore, events: list[str]) -> None:
        self.policy = materials.gateway.policy
        self.secret_resolver = store
        self.events = events
        self.revoke_secret_arguments: list[bool] = []

    def register_session(self, **_kwargs: Any) -> None:
        self.events.append('gateway.register')

    def revoke_capability(self, capability_id: str, **_kwargs: Any) -> None:
        self.secret_resolver.revoke(capability_id)

    def revoke_unregistered_capability(
        self,
        capability_id: str,
        **_kwargs: Any,
    ) -> None:
        self.secret_resolver.revoke(capability_id)

    def seal_session(
        self,
        _capability_id: str,
        *,
        terminal_reason: GatewayTerminalReason,
        sealed_at: datetime,
        revoke_secret: bool,
    ) -> AuthenticatedGatewaySession:
        del sealed_at
        self.events.append('gateway.seal')
        self.revoke_secret_arguments.append(revoke_secret)
        return cast(
            AuthenticatedGatewaySession,
            SimpleNamespace(seal=SimpleNamespace(terminal_reason=terminal_reason)),
        )


class _Supervisor:
    def __init__(self, materials: Materials, root: Path, events: list[str], now: datetime) -> None:
        self.spec = materials.spec
        self.root = root
        self.events = events
        self.now = now
        self.launch_count = 0

    def prepare(self, *, run_id: str) -> FirecrackerPreparedWorker:
        self.events.append('worker.prepare')
        jail_root = self.root / run_id / 'root'
        preflight = FirecrackerHostPreflightReceipt(
            worker_spec_sha256=firecracker_model_sha256(self.spec),
            collected_at=self.now,
            host_architecture='aarch64',
            host_kernel_release='test-kernel',
            cgroup_controllers=('cpu', 'memory', 'pids'),
        )
        return FirecrackerPreparedWorker(
            run_id=run_id,
            worker_spec_sha256=firecracker_model_sha256(self.spec),
            host_preflight=preflight,
            host_preflight_sha256=firecracker_model_sha256(preflight),
            jail_root=str(jail_root),
            config_path=str(jail_root / 'firecracker-config.json'),
            config_sha256=sha256(firecracker_static_config_bytes(self.spec)).hexdigest(),
            kernel_sha256=self.spec.images.kernel.sha256,
            rootfs_sha256=self.spec.images.rootfs.sha256,
            harness_sha256=self.spec.images.harness.sha256,
            initial_scratch_sha256=self.spec.images.scratch_template.sha256,
            vsock_uds_path=str(jail_root / 'run' / 'vsock.sock'),
            created_at=self.now,
        )

    def launch(self, prepared: FirecrackerPreparedWorker) -> RunningFirecrackerWorker:
        self.events.append('worker.launch')
        self.launch_count += 1
        return cast(
            RunningFirecrackerWorker,
            SimpleNamespace(
                prepared=prepared,
                wall_deadline_monotonic=2.0,
                firecracker_pid=4343,
                process=SimpleNamespace(pid=4242),
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
        del running, grace_seconds
        self.events.append('worker.cleanup')
        return FirecrackerCleanupReceipt(
            run_id=RUN_ID,
            launched_monotonic_ns=1_000_000_000,
            wall_deadline_monotonic_ns=31_000_000_000,
            watchdog_triggered_at=None,
            watchdog_triggered_monotonic_ns=None,
            jailer_reaped_at=self.now + timedelta(milliseconds=100),
            jailer_reaped_monotonic_ns=1_100_000_000,
            cgroup_empty_at=self.now + timedelta(seconds=1),
            cgroup_empty_monotonic_ns=2_000_000_000,
            cleanup_finished_at=self.now + timedelta(seconds=2),
            cleanup_finished_monotonic_ns=3_000_000_000,
            lifecycle='terminated',
            jailer_exit_code=1,
            wall_watchdog_armed=True,
            wall_timeout_triggered=False,
        )

    def discard_prepared(self, prepared: FirecrackerPreparedWorker) -> FirecrackerCleanupReceipt:
        del prepared
        self.events.append('worker.discard')
        return FirecrackerCleanupReceipt(
            run_id=RUN_ID,
            launched_monotonic_ns=None,
            wall_deadline_monotonic_ns=None,
            watchdog_triggered_at=None,
            watchdog_triggered_monotonic_ns=None,
            jailer_reaped_at=None,
            jailer_reaped_monotonic_ns=None,
            cgroup_empty_at=None,
            cgroup_empty_monotonic_ns=None,
            cleanup_finished_at=self.now,
            cleanup_finished_monotonic_ns=1_000_000_000,
            lifecycle='never_launched',
            jailer_exit_code=None,
            wall_watchdog_armed=False,
            wall_timeout_triggered=False,
        )


class _GuestSession:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.terminal = False
        self.final_submission_bytes = b''

    def abort(self, _code: object) -> None:
        self.events.append('guest.abort')
        self.terminal = True

    def seal(self, *, sealed_at: datetime) -> AuthenticatedGuestRpcSession:
        del sealed_at
        self.events.append('guest.seal')
        return cast(
            AuthenticatedGuestRpcSession,
            SimpleNamespace(seal=SimpleNamespace(terminal_status=GuestRpcTerminalStatus.ABORTED)),
        )


class _JournalThenFailRunner:
    authenticated_bootstrap: AuthenticatedClinicalGuestBootstrap | None = None

    def __init__(
        self,
        events: list[str],
        *,
        journal: Callable[[AuthenticatedClinicalGuestBootstrap], str],
        anchor: ClinicalGuestBootstrapTrustAnchor,
        now: datetime,
    ) -> None:
        self.events = events
        self.journal = journal
        self.anchor = anchor
        self.now = now
        self.serve_count = 0
        self.expected_peer_pids: list[int] = []

    def open(self) -> None:
        self.events.append('bootstrap.open')

    def serve_one(
        self,
        *,
        hello: ClinicalGuestBootstrapHello,
        expected_peer_pid: int,
        **_kwargs: Any,
    ) -> AuthenticatedClinicalGuestBootstrap:
        self.events.append('bootstrap.serve')
        self.serve_count += 1
        self.expected_peer_pids.append(expected_peer_pid)
        artifact = _artifact_for_hello(hello, anchor=self.anchor, now=self.now)
        self.authenticated_bootstrap = artifact
        self.journal(artifact)
        self.events.append('bootstrap.journal')
        raise FirecrackerClinicalRuntimeError('private post-bootstrap RPC failure detail')

    def close(self) -> None:
        self.events.append('bootstrap.close')


def test_runtime_composition_has_one_launch_one_revoke_and_ordered_failure_cleanup(tmp_path: Path) -> None:
    materials = _materials(tmp_path / 'materials')
    _, anchor = _bootstrap(materials)
    now = materials.worker.attestation.started_at
    events: list[str] = []
    ownership = _Ownership(events)
    store = _SecretStore(events)
    gateway = _Gateway(materials, store, events)
    supervisor = _Supervisor(materials, tmp_path / 'jails', events, now)
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
        registry_authority_id='runtime-test-authority',
        registered_entry_id='runtime-test-entry',
        cohort_id='runtime-test-cohort',
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
    runners: list[_JournalThenFailRunner] = []

    def runner_factory(**kwargs: Any) -> _JournalThenFailRunner:
        assert kwargs['worker_uid'] == materials.spec.worker_uid
        assert kwargs['worker_gid'] == materials.spec.worker_gid
        runner = _JournalThenFailRunner(
            events,
            journal=kwargs['journal_authenticated_bootstrap'],
            anchor=anchor,
            now=now,
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
            lambda **_kwargs: _GuestSession(events),
        ),
        finalize_worker=lambda **_kwargs: (
            events.append('worker.finalize') or cast(AuthenticatedFirecrackerWorkerAttestation, SimpleNamespace())
        ),
        managed_ownership=ownership,
    )
    request = ClinicalRuntimePrepareRequest(
        deployment=deployment,
        reservation=reservation,
        binding=binding,
        launch=launch,
        workspace=materials.workspace,
    )
    prepared = runtime.prepare(request)
    private_state = runtime._states[RUN_ID]
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

    outcome = runtime.run(prepared, start)

    assert len(runners) == 1
    authenticated_bootstrap = runners[0].authenticated_bootstrap
    assert authenticated_bootstrap is not None
    bootstrap_sha256 = sha256(canonical_json_bytes(authenticated_bootstrap)).hexdigest()
    assert outcome == ClinicalRuntimeFailed(
        ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE,
        authenticated_bootstrap_sha256=bootstrap_sha256,
    )
    assert supervisor.launch_count == 1
    assert runners[0].serve_count == 1
    assert runners[0].expected_peer_pids == [4343]
    assert store.revocations == 1
    assert store.secrets == {}
    assert gateway.revoke_secret_arguments == [False]
    journal_path = runtime.bootstrap_journal_root / f'{RUN_ID}.json'
    assert journal_path.read_bytes() == canonical_json_bytes(authenticated_bootstrap)
    assert stat.S_IMODE(journal_path.stat().st_mode) == 0o600
    assert events == [
        'ownership.begin',
        'worker.prepare',
        'ownership.prepared',
        'ownership.start-bound',
        'secret.register',
        'gateway.register',
        'bootstrap.open',
        'worker.launch',
        'ownership.running',
        'bootstrap.serve',
        'bootstrap.journal',
        'bootstrap.close',
        'guest.abort',
        'guest.seal',
        'gateway.seal',
        'worker.cleanup',
        'worker.finalize',
        'secret.revoke',
        'ownership.capability-revoked',
        'ownership.cleaned',
    ]
    assert runtime.run(prepared, start) == ClinicalRuntimeFailed(ClinicalRuntimeFailureCode.LAUNCH_FAILED)
    assert supervisor.launch_count == 1
    assert journal_path.exists()

    assert private_state.bootstrap_journal_path == journal_path
    assert private_state.bootstrap_journal_sha256 == bootstrap_sha256


def test_runtime_constructor_rejects_non_cancellable_in_process_provider() -> None:
    store = SimpleNamespace()
    gateway = SimpleNamespace(
        secret_resolver=store,
        provider_calls_forcibly_cancellable=False,
    )
    with pytest.raises(ValueError, match='forcibly cancellable provider calls'):
        FirecrackerClinicalRuntime(
            config=cast(Any, None),
            supervisor=cast(Any, None),
            gateway=cast(AuthenticatedProviderGateway, gateway),
            gateway_secret_store=cast(Any, store),
            execution_policy=cast(Any, None),
            gateway_route=cast(Any, None),
            provider_subprocess_spec_sha256=_PROVIDER_SUBPROCESS_SPEC_SHA256,
            provider_subprocess_behavior_sha256=_PROVIDER_SUBPROCESS_BEHAVIOR_SHA256,
            provider_subprocess_module_source_sha256=(_PROVIDER_SUBPROCESS_MODULE_SOURCE_SHA256),
            guest_rpc_policy=cast(Any, None),
            harness=cast(Any, None),
            keys=cast(Any, None),
            bootstrap_authorization_signer=cast(Any, None),
            bootstrap_trust_anchor=cast(Any, None),
            evidence_root=Path('/not-reached'),
        )
