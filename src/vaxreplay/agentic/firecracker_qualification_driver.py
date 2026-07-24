"""Concrete root-Linux/KVM ``run-drill`` driver for Firecracker qualification.

Unlike the collector's subprocess boundary, this module performs the work: it derives the only
allowed qualification worker spec, prepares and directly boots Firecracker, authenticates the
guest-initiated vsock peer, reads procfs/cgroupfs/monotonic/lstat evidence, and tears every worker
down before emitting one canonical raw drill observation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import socket
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from vaxreplay.agentic.firecracker import (
    FirecrackerPreboundGuestListener,
    FirecrackerPreparedWorker,
    FirecrackerSupervisor,
    FirecrackerWorkerError,
    FirecrackerWorkerSpec,
    RunningFirecrackerWorker,
    capture_firecracker_prebound_guest_listener,
    firecracker_guest_initiated_uds_path,
    firecracker_model_sha256,
)
from vaxreplay.agentic.firecracker_qualification import (
    FirecrackerQualificationClaim,
    FirecrackerQualificationDrillId,
    required_firecracker_qualification_claims,
)
from vaxreplay.agentic.firecracker_qualification_cgroup_canary import (
    FirecrackerQualificationCgroupCanaryError,
    run_host_cgroup_controller_canary,
)
from vaxreplay.agentic.firecracker_qualification_guest import FirecrackerQualificationGuestRequest
from vaxreplay.agentic.firecracker_qualification_probe import (
    AuthenticatedFirecrackerQualificationGuestResponse,
    FirecrackerQualificationCgroupSnapshot,
    FirecrackerQualificationChallenge,
    FirecrackerQualificationClaimMeasurement,
    FirecrackerQualificationGuestCommand,
    FirecrackerQualificationHostCgroupCanary,
    FirecrackerQualificationObservationSource,
    FirecrackerQualificationProbeManifest,
    FirecrackerQualificationRawDrillObservation,
    FirecrackerQualificationTeardownMeasurement,
    FirecrackerQualificationWallTimeoutMeasurement,
    FirecrackerQualificationWorkerBinding,
    FirecrackerQualificationWorkerInterval,
    derive_firecracker_qualification_worker_spec,
    firecracker_qualification_probe_manifest_sha256,
    firecracker_qualification_static_config_sha256,
    validate_probe_manifest_for_worker,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

FIRECRACKER_QUALIFICATION_DRIVER_PROTOCOL = 'vaxreplay.firecracker-qualification-driver.v0.1'
FIRECRACKER_QUALIFICATION_DRIVER_REQUEST_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-driver-request.v0.1'
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_GUEST_FRAME_BYTES = 1024 * 1024
_FRAME_LENGTH = struct.Struct('!I')
_PEER_CREDENTIALS = struct.Struct('3i')

_COMMAND_BY_DRILL = {
    FirecrackerQualificationDrillId.LIVE_BOOT: FirecrackerQualificationGuestCommand.BOOT_READY_AND_EXIT,
    FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP: FirecrackerQualificationGuestCommand.VSOCK_NONCE_ECHO,
    FirecrackerQualificationDrillId.GUEST_ISOLATION: FirecrackerQualificationGuestCommand.ISOLATION_PROBES,
    FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT: FirecrackerQualificationGuestCommand.CGROUP_STRESS,
    FirecrackerQualificationDrillId.WALL_TIMEOUT: FirecrackerQualificationGuestCommand.INTENTIONAL_HANG,
    FirecrackerQualificationDrillId.LOAD_CANARY: FirecrackerQualificationGuestCommand.LOAD_CANARY,
}

_SOURCE_BY_CLAIM = {
    FirecrackerQualificationClaim.FIRECRACKER_PROCESS_STARTED: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.GUEST_READY_AUTHENTICATED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.CLEAN_GUEST_EXIT: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.HOST_VSOCK_HANDSHAKE: FirecrackerQualificationObservationSource.HOST_VSOCK,
    FirecrackerQualificationClaim.GUEST_RPC_ROUND_TRIP: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.PEER_CID_BOUND: FirecrackerQualificationObservationSource.HOST_VSOCK,
    FirecrackerQualificationClaim.ROOTFS_WRITE_DENIED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.HARNESS_WRITE_DENIED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.SCRATCH_WRITE_SUCCEEDED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.SCRATCH_FRESH: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.NETWORK_UNREACHABLE: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.MMDS_UNREACHABLE: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.CPU_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.MEMORY_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.SWAP_DISABLED_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.PIDS_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.WALL_WATCHDOG_TRIGGERED: FirecrackerQualificationObservationSource.HOST_MONOTONIC,
    FirecrackerQualificationClaim.PROCESS_GROUP_KILLED: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.DEADLINE_BOUND: FirecrackerQualificationObservationSource.HOST_MONOTONIC,
    FirecrackerQualificationClaim.CGROUP_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.JAIL_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.VSOCK_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.PARALLEL_WORKERS_DISTINCT: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.ALL_WORKERS_COMPLETED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.ALL_WORKERS_TORN_DOWN: FirecrackerQualificationObservationSource.HOST_LSTAT,
}


class FirecrackerQualificationDriverError(RuntimeError):
    """A live drill could not be measured and cleaned up exactly."""


class FirecrackerQualificationDriverRequest(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-driver-request.v0.1'] = (
        FIRECRACKER_QUALIFICATION_DRIVER_REQUEST_SCHEMA_VERSION
    )
    challenge: FirecrackerQualificationChallenge
    worker_spec: FirecrackerWorkerSpec
    probe_manifest: FirecrackerQualificationProbeManifest
    caller_assertions_are_evidence: Literal[False] = False

    @model_validator(mode='after')
    def validate_release(self) -> Self:
        validate_probe_manifest_for_worker(self.probe_manifest, self.worker_spec)
        if (
            firecracker_model_sha256(self.worker_spec) != self.challenge.worker_spec_sha256
            or firecracker_qualification_probe_manifest_sha256(self.probe_manifest)
            != self.challenge.probe_manifest_sha256
        ):
            raise ValueError('qualification driver request differs from its fresh challenge')
        return self


class LinuxQualificationEvidenceReader:
    """Narrow filesystem/process seam; production fixes all roots to real Linux kernel filesystems."""

    def __init__(
        self,
        *,
        proc_root: Path = Path('/proc'),
        cgroup_root: Path = Path('/sys/fs/cgroup'),
        production: bool = True,
        getpgid: Callable[[int], int] = os.getpgid,
    ) -> None:
        injected = proc_root != Path('/proc') or cgroup_root != Path('/sys/fs/cgroup') or getpgid is not os.getpgid
        if production and injected:
            raise FirecrackerQualificationDriverError('production evidence reader forbids injected kernel seams')
        self.proc_root = proc_root
        self.cgroup_root = cgroup_root
        self.getpgid = getpgid

    def cgroup_path(self, spec: FirecrackerWorkerSpec, run_id: str) -> Path:
        return self.cgroup_root.joinpath(*spec.cgroup_parent.split('/'), run_id)

    def process_binding(
        self,
        *,
        process_pid: int,
        expected_process_group_id: int,
        expected_session_id: int,
        cgroup_path: Path,
        expected_executable_sha256: str,
    ) -> tuple[tuple[int, ...], bool, bool, str, int]:
        members = _read_integer_lines(cgroup_path / 'cgroup.procs')
        try:
            proc_cgroup = _read_small_file(self.proc_root / str(process_pid) / 'cgroup', 64 * 1024).decode('ascii')
            parent_pid, process_group, session_id, start_time_ticks = _proc_process_identity(
                self.proc_root / str(process_pid) / 'stat'
            )
            del parent_pid
            syscall_process_group = self.getpgid(process_pid)
            executable_sha256 = _proc_executable_sha256(self.proc_root / str(process_pid) / 'exe')
        except (OSError, UnicodeDecodeError):
            return members, False, False, '0' * 64, 1
        expected_relative = '/' + cgroup_path.relative_to(self.cgroup_root).as_posix()
        cgroup_bound = f'0::{expected_relative}' in proc_cgroup.splitlines() and process_pid in members
        process_tree = (
            process_group == expected_process_group_id
            and syscall_process_group == expected_process_group_id
            and session_id == expected_session_id
            and process_pid in members
            and executable_sha256 == expected_executable_sha256
        )
        return members, process_tree, cgroup_bound, executable_sha256, start_time_ticks

    def pid_file_binding(
        self,
        *,
        path: Path,
        expected_pid: int,
        expected_device_id: int,
        expected_inode: int,
    ) -> bool:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError:
            return False
        try:
            metadata = os.fstat(descriptor)
            content = os.read(descriptor, 11)
        finally:
            os.close(descriptor)
        return (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == 0
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600
            and (metadata.st_dev, metadata.st_ino) == (expected_device_id, expected_inode)
            and content.isdigit()
            and int(content) == expected_pid
        )

    def cgroup_snapshot(
        self,
        *,
        spec: FirecrackerWorkerSpec,
        run_id: str,
        clock: Callable[[], datetime],
    ) -> FirecrackerQualificationCgroupSnapshot:
        path = self.cgroup_path(spec, run_id)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FirecrackerQualificationDriverError('worker cgroup is not a real cgroup-v2 directory')
        cpu_max = _read_tokens(path / 'cpu.max', expected=2)
        cpu_stat = _read_keyed_integers(path / 'cpu.stat')
        memory_events = _read_keyed_integers(path / 'memory.events')
        pids_events = _read_keyed_integers(path / 'pids.events')
        return FirecrackerQualificationCgroupSnapshot(
            run_id=run_id,
            cgroup_path=str(Path('/sys/fs/cgroup').joinpath(*spec.cgroup_parent.split('/'), run_id)),
            cgroup_inode=metadata.st_ino,
            observed_at=clock(),
            cpu_max_quota_us=_bounded_integer(cpu_max[0], label='cpu.max quota'),
            cpu_max_period_us=_bounded_integer(cpu_max[1], label='cpu.max period'),
            memory_max_bytes=_bounded_integer(_read_ascii(path / 'memory.max'), label='memory.max'),
            memory_swap_max_bytes=_bounded_integer(_read_ascii(path / 'memory.swap.max'), label='memory.swap.max'),
            pids_max=_bounded_integer(_read_ascii(path / 'pids.max'), label='pids.max'),
            cpu_nr_throttled=cpu_stat.get('nr_throttled', 0),
            cpu_throttled_usec=cpu_stat.get('throttled_usec', 0),
            memory_oom=memory_events.get('oom', 0),
            memory_oom_kill=memory_events.get('oom_kill', 0),
            pids_max_events=pids_events.get('max', 0),
            member_pids=_read_integer_lines(path / 'cgroup.procs'),
        )

    def path_absent(self, path: Path) -> bool:
        try:
            path.lstat()
        except FileNotFoundError:
            return True
        except OSError as error:
            raise FirecrackerQualificationDriverError('cannot lstat a teardown path') from error
        return False


class _GuestVsockListener:
    def __init__(self, *, prepared: FirecrackerPreparedWorker, spec: FirecrackerWorkerSpec) -> None:
        self.prepared = prepared
        self.spec = spec
        self.path = Path(
            firecracker_guest_initiated_uds_path(
                uds_path=prepared.vsock_uds_path,
                port=spec.guest_rpc_port,
            )
        )
        self.listener: socket.socket | None = None
        self.connection: socket.socket | None = None
        self.peer: tuple[int, int, int] | None = None
        self.socket_identity: tuple[int, int] | None = None

    def open(self) -> None:
        if self.path.exists() or self.path.is_symlink():
            raise FirecrackerQualificationDriverError('qualification vsock listener is not fresh')
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.path))
            self.listener = listener
            os.chown(self.path, self.spec.worker_uid, self.spec.worker_gid, follow_symlinks=False)
            os.chmod(self.path, 0o600, follow_symlinks=False)
            self._verify_owned_socket(record=True)
            listener.listen(1)
            listener.settimeout(float(self.spec.limits.wall_seconds))
        except BaseException:
            try:
                self.close()
            except BaseException:
                raise FirecrackerQualificationDriverError('qualification vsock setup and cleanup both failed') from None
            raise

    def accept(self, *, expected_pid: int) -> tuple[int, int, int]:
        if self.listener is None:
            raise FirecrackerQualificationDriverError('qualification vsock listener was not opened')
        self._verify_owned_socket()
        connection, _ = self.listener.accept()
        option = getattr(socket, 'SO_PEERCRED', None)
        if platform.system() != 'Linux' or not isinstance(option, int):
            connection.close()
            raise FirecrackerQualificationDriverError('Linux SO_PEERCRED is unavailable')
        peer = _PEER_CREDENTIALS.unpack(connection.getsockopt(socket.SOL_SOCKET, option, _PEER_CREDENTIALS.size))
        expected = (expected_pid, self.spec.worker_uid, self.spec.worker_gid)
        if peer != expected:
            connection.close()
            raise FirecrackerQualificationDriverError(
                'qualification vsock peer differs from the launched Firecracker process'
            )
        connection.settimeout(float(self.spec.limits.wall_seconds))
        self.connection = connection
        self.peer = peer
        return peer

    def prebound_identity(self) -> FirecrackerPreboundGuestListener:
        if self.listener is None or self.socket_identity is None:
            raise FirecrackerQualificationDriverError('qualification vsock listener was not opened')
        self._verify_owned_socket()
        identity = capture_firecracker_prebound_guest_listener(
            self.prepared,
            spec=self.spec,
        )
        if (identity.device_id, identity.inode) != self.socket_identity:
            raise FirecrackerQualificationDriverError('qualification vsock listener identity changed')
        return identity

    def exchange(
        self,
        request: FirecrackerQualificationGuestRequest,
    ) -> AuthenticatedFirecrackerQualificationGuestResponse:
        if self.connection is None:
            raise FirecrackerQualificationDriverError('qualification guest did not establish vsock')
        payload = canonical_json_bytes(request)
        self.connection.sendall(_FRAME_LENGTH.pack(len(payload)) + payload)
        (length,) = _FRAME_LENGTH.unpack(_recv_exact(self.connection, _FRAME_LENGTH.size))
        if length == 0 or length > _MAX_GUEST_FRAME_BYTES:
            raise FirecrackerQualificationDriverError('qualification guest response is oversized')
        response_bytes = _recv_exact(self.connection, length)
        try:
            response = AuthenticatedFirecrackerQualificationGuestResponse.model_validate_json(response_bytes)
        except ValueError as error:
            raise FirecrackerQualificationDriverError('qualification guest returned an invalid response') from error
        if canonical_json_bytes(response) != response_bytes:
            raise FirecrackerQualificationDriverError('qualification guest response is not canonical JSON')
        return response

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(metadata.st_mode):
            raise FirecrackerQualificationDriverError('qualification vsock listener changed type')
        self._verify_owned_socket()
        self.path.unlink()
        self.socket_identity = None

    def _verify_owned_socket(self, *, record: bool = False) -> None:
        try:
            metadata = self.path.lstat()
        except OSError as error:
            raise FirecrackerQualificationDriverError('qualification vsock listener is unavailable') from error
        observed_identity = (metadata.st_dev, metadata.st_ino)
        if record:
            self.socket_identity = observed_identity
        if (
            not stat.S_ISSOCK(metadata.st_mode)
            or self.socket_identity != observed_identity
            or metadata.st_uid != self.spec.worker_uid
            or metadata.st_gid != self.spec.worker_gid
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise FirecrackerQualificationDriverError('qualification vsock listener identity changed')


class LinuxKvmFirecrackerQualificationDriver:
    """One-shot concrete drill runner.  Public construction has no production injection points."""

    def __init__(self, request: FirecrackerQualificationDriverRequest) -> None:
        _require_root_linux_kvm()
        self.request = request
        self.task_spec = request.worker_spec
        self.manifest = request.probe_manifest
        self.spec = derive_firecracker_qualification_worker_spec(self.manifest, task_worker_spec=self.task_spec)
        self.supervisor = FirecrackerSupervisor(self.spec)
        self.reader = LinuxQualificationEvidenceReader()
        self.clock: Callable[[], datetime] = lambda: datetime.now(UTC)
        self.monotonic_ns: Callable[[], int] = time.monotonic_ns

    def run(self) -> FirecrackerQualificationRawDrillObservation:
        drill = self.request.challenge.drill_id
        if drill == FirecrackerQualificationDrillId.LOAD_CANARY:
            return self._run_load_canary()
        return self._run_one(drill)

    def _run_one(self, drill: FirecrackerQualificationDrillId) -> FirecrackerQualificationRawDrillObservation:
        challenge = self.request.challenge
        run_id = challenge.run_ids[0]
        started = self.clock()
        prepared: FirecrackerPreparedWorker | None = None
        running: RunningFirecrackerWorker | None = None
        listener: _GuestVsockListener | None = None
        response: AuthenticatedFirecrackerQualificationGuestResponse | None = None
        binding: FirecrackerQualificationWorkerBinding | None = None
        snapshots: tuple[FirecrackerQualificationCgroupSnapshot, ...] = ()
        host_cgroup_canary: FirecrackerQualificationHostCgroupCanary | None = None
        wall: FirecrackerQualificationWallTimeoutMeasurement | None = None
        teardown: tuple[FirecrackerQualificationTeardownMeasurement, ...] = ()
        cleanup_finished = False
        clean_exit = False
        try:
            prepared = self.supervisor.prepare(run_id=run_id)
            listener = _GuestVsockListener(prepared=prepared, spec=self.spec)
            listener.open()
            running = self.supervisor.launch(
                prepared,
                prebound_guest_listener=listener.prebound_identity(),
            )
            peer = listener.accept(expected_pid=running.firecracker_pid)
            binding = self._binding(prepared=prepared, running=running, peer=peer)
            if drill == FirecrackerQualificationDrillId.TEARDOWN:
                listener.close()
                listener = None
                self.supervisor.terminate_and_cleanup(running)
                cleanup_finished = True
                teardown = (self._teardown(binding),)
            else:
                guest_request = self._guest_request(binding)
                if drill == FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT:
                    before = self.reader.cgroup_snapshot(spec=self.spec, run_id=run_id, clock=self.clock)
                    response = listener.exchange(guest_request)
                    during = self.reader.cgroup_snapshot(spec=self.spec, run_id=run_id, clock=self.clock)
                    canary = run_host_cgroup_controller_canary(
                        spec=self.spec,
                        running=running,
                        binding=binding,
                        baseline=before,
                        guest_pressure=during,
                        snapshot=lambda: self.reader.cgroup_snapshot(
                            spec=self.spec,
                            run_id=run_id,
                            clock=self.clock,
                        ),
                        monotonic_ns=self.monotonic_ns,
                    )
                    snapshots = canary.snapshots
                    host_cgroup_canary = canary.measurement
                    if self._binding(prepared=prepared, running=running, peer=peer) != binding:
                        raise FirecrackerQualificationDriverError(
                            'Firecracker binding changed while host controller canaries ran'
                        )
                else:
                    response = listener.exchange(guest_request)
                listener.close()
                listener = None
                if drill == FirecrackerQualificationDrillId.WALL_TIMEOUT:
                    wall = self._await_wall_timeout(running=running, binding=binding)
                    cleanup_finished = True
                elif drill == FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT:
                    self.supervisor.terminate_and_cleanup(running)
                    cleanup_finished = True
                else:
                    clean_exit = self.supervisor.wait_for_exit(
                        running,
                        timeout_seconds=min(10.0, float(self.spec.limits.wall_seconds)),
                    )
                    self.supervisor.terminate_and_cleanup(running)
                    cleanup_finished = True
        except (
            OSError,
            FirecrackerQualificationCgroupCanaryError,
            FirecrackerWorkerError,
            ValueError,
            subprocess.SubprocessError,
        ) as error:
            raise FirecrackerQualificationDriverError('live qualification drill failed') from error
        finally:
            cleanup_errors: list[BaseException] = []
            if listener is not None:
                try:
                    listener.close()
                except BaseException as error:
                    cleanup_errors.append(error)
            if running is not None and not cleanup_finished:
                try:
                    self.supervisor.terminate_and_cleanup(running)
                except BaseException as error:
                    cleanup_errors.append(error)
            elif prepared is not None and running is None:
                try:
                    self.supervisor.discard_prepared(prepared)
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                raise FirecrackerQualificationDriverError(
                    'qualification drill cleanup could not be proved'
                ) from cleanup_errors[0]
        if binding is None:
            raise FirecrackerQualificationDriverError('qualification drill did not bind a worker')
        observed = self._observed_claims(
            drill=drill,
            response=response,
            binding=binding,
            clean_exit=clean_exit,
            snapshots=snapshots,
            wall=wall,
            teardown=teardown,
        )
        finished = self.clock()
        return FirecrackerQualificationRawDrillObservation(
            drill_id=drill,
            challenge=challenge,
            started_at=started,
            finished_at=finished,
            worker_bindings=(binding,),
            guest_responses=() if response is None else (response,),
            claim_measurements=self._measurements(drill=drill, observed=observed, observed_at=finished),
            cgroup_snapshots=snapshots,
            host_cgroup_canary=host_cgroup_canary,
            wall_timeout=wall,
            teardown_measurements=teardown,
        )

    def _run_load_canary(self) -> FirecrackerQualificationRawDrillObservation:
        challenge = self.request.challenge
        started = self.clock()
        prepared: list[FirecrackerPreparedWorker] = []
        running: list[RunningFirecrackerWorker] = []
        listeners: list[_GuestVsockListener] = []
        bindings: list[FirecrackerQualificationWorkerBinding] = []
        responses: list[AuthenticatedFirecrackerQualificationGuestResponse] = []
        intervals: list[FirecrackerQualificationWorkerInterval] = []
        cleaned: set[str] = set()
        try:
            for run_id in challenge.run_ids:
                item = self.supervisor.prepare(run_id=run_id)
                listener = _GuestVsockListener(prepared=item, spec=self.spec)
                listener.open()
                prepared.append(item)
                listeners.append(listener)
            for item, listener in zip(prepared, listeners, strict=True):
                running.append(
                    self.supervisor.launch(
                        item,
                        prebound_guest_listener=listener.prebound_identity(),
                    )
                )
            for item, worker, listener in zip(prepared, running, listeners, strict=True):
                peer = listener.accept(expected_pid=worker.firecracker_pid)
                binding = self._binding(prepared=item, running=worker, peer=peer)
                bindings.append(binding)
                response = listener.exchange(self._guest_request(binding))
                responses.append(response)
                intervals.append(
                    FirecrackerQualificationWorkerInterval(
                        run_id=binding.run_id,
                        started_monotonic_ns=worker.launched_at_monotonic_ns,
                        finished_monotonic_ns=self.monotonic_ns(),
                    )
                )
            for listener in listeners:
                listener.close()
            listeners.clear()
            for worker in running:
                self.supervisor.terminate_and_cleanup(worker)
                cleaned.add(worker.prepared.run_id)
        finally:
            cleanup_errors: list[BaseException] = []
            for listener in listeners:
                try:
                    listener.close()
                except BaseException as error:
                    cleanup_errors.append(error)
            for worker in running:
                if worker.prepared.run_id not in cleaned:
                    try:
                        self.supervisor.terminate_and_cleanup(worker)
                    except BaseException as error:
                        cleanup_errors.append(error)
            launched_ids = {item.prepared.run_id for item in running}
            for item in prepared:
                if item.run_id not in launched_ids:
                    try:
                        self.supervisor.discard_prepared(item)
                    except BaseException as error:
                        cleanup_errors.append(error)
            if cleanup_errors:
                raise FirecrackerQualificationDriverError(
                    'load-canary cleanup could not be proved for every worker'
                ) from cleanup_errors[0]
        if len(bindings) != 2 or len(responses) != 2:
            raise FirecrackerQualificationDriverError('load canary did not complete two workers')
        teardowns = tuple(self._teardown(binding) for binding in bindings)
        distinct = all(
            len(values) == 2
            for values in (
                {item.firecracker_pid for item in bindings},
                {item.process_group_id for item in bindings},
                {item.cgroup_path for item in bindings},
                {item.jail_root for item in bindings},
                {item.vsock_uds_path for item in bindings},
            )
        ) and max(item.started_monotonic_ns for item in intervals) < min(
            item.finished_monotonic_ns for item in intervals
        )
        observed = {
            FirecrackerQualificationClaim.PARALLEL_WORKERS_DISTINCT: distinct,
            FirecrackerQualificationClaim.ALL_WORKERS_COMPLETED: True,
            FirecrackerQualificationClaim.ALL_WORKERS_TORN_DOWN: True,
        }
        finished = self.clock()
        return FirecrackerQualificationRawDrillObservation(
            drill_id=FirecrackerQualificationDrillId.LOAD_CANARY,
            challenge=challenge,
            started_at=started,
            finished_at=finished,
            worker_bindings=tuple(bindings),
            guest_responses=tuple(responses),
            claim_measurements=self._measurements(
                drill=FirecrackerQualificationDrillId.LOAD_CANARY,
                observed=observed,
                observed_at=finished,
            ),
            teardown_measurements=teardowns,
            worker_intervals=tuple(intervals),
        )

    def _binding(
        self,
        *,
        prepared: FirecrackerPreparedWorker,
        running: RunningFirecrackerWorker,
        peer: tuple[int, int, int],
    ) -> FirecrackerQualificationWorkerBinding:
        cgroup = self.reader.cgroup_path(self.spec, prepared.run_id)
        metadata = cgroup.lstat()
        members, process_tree, cgroup_bound, executable_sha256, start_time_ticks = self.reader.process_binding(
            process_pid=running.firecracker_pid,
            expected_process_group_id=running.firecracker_process_group_id,
            expected_session_id=running.firecracker_session_id,
            cgroup_path=cgroup,
            expected_executable_sha256=self.spec.runtime.firecracker.sha256,
        )
        pid_file_verified = self.reader.pid_file_binding(
            path=Path(running.firecracker_pid_file_path),
            expected_pid=running.firecracker_pid,
            expected_device_id=running.firecracker_pid_file_device_id,
            expected_inode=running.firecracker_pid_file_inode,
        )
        if (
            not process_tree
            or not cgroup_bound
            or not pid_file_verified
            or executable_sha256 != self.spec.runtime.firecracker.sha256
            or start_time_ticks != running.firecracker_start_time_ticks
        ):
            raise FirecrackerQualificationDriverError(
                'launched PID is not the pinned Firecracker executable in the exact worker cgroup'
            )
        return FirecrackerQualificationWorkerBinding(
            run_id=prepared.run_id,
            worker_spec_sha256=self.request.challenge.worker_spec_sha256,
            qualification_worker_spec_sha256=firecracker_model_sha256(self.spec),
            qualification_static_config_sha256=firecracker_qualification_static_config_sha256(self.spec),
            prepared_worker_sha256=firecracker_model_sha256(prepared),
            probe_manifest_sha256=self.request.challenge.probe_manifest_sha256,
            firecracker_pid=running.firecracker_pid,
            firecracker_parent_pid_at_observation=running.firecracker_parent_pid_at_observation,
            firecracker_start_time_ticks=start_time_ticks,
            firecracker_session_id=running.firecracker_session_id,
            firecracker_executable_sha256=executable_sha256,
            firecracker_pid_file_path=running.firecracker_pid_file_path,
            firecracker_pid_file_device_id=running.firecracker_pid_file_device_id,
            firecracker_pid_file_inode=running.firecracker_pid_file_inode,
            firecracker_pid_from_jailer_file_verified=True,
            jailer_pid=running.process.pid,
            jailer_start_time_ticks=running.jailer_start_time_ticks,
            jailer_process_group_id=running.jailer_process_group_id,
            jailer_session_id=running.jailer_session_id,
            process_group_id=running.firecracker_process_group_id,
            worker_uid=self.spec.worker_uid,
            worker_gid=self.spec.worker_gid,
            cgroup_path=str(Path('/sys/fs/cgroup').joinpath(*self.spec.cgroup_parent.split('/'), prepared.run_id)),
            cgroup_inode=metadata.st_ino,
            cgroup_member_pids=members,
            jail_root=prepared.jail_root,
            vsock_uds_path=prepared.vsock_uds_path,
            guest_cid=self.spec.guest_cid,
            peer_pid=peer[0],
            peer_uid=peer[1],
            peer_gid=peer[2],
            process_tree_verified=process_tree,
            pid_cgroup_binding_verified=cgroup_bound,
        )

    def _guest_request(self, binding: FirecrackerQualificationWorkerBinding) -> FirecrackerQualificationGuestRequest:
        challenge = self.request.challenge
        return FirecrackerQualificationGuestRequest(
            challenge=challenge,
            worker_binding=binding,
            command=_COMMAND_BY_DRILL[challenge.drill_id],
            worker_spec_sha256=challenge.worker_spec_sha256,
            probe_manifest_sha256=challenge.probe_manifest_sha256,
            guest_probe_executable_sha256=self.manifest.guest_probe_executable_sha256,
        )

    def _await_wall_timeout(
        self,
        *,
        running: RunningFirecrackerWorker,
        binding: FirecrackerQualificationWorkerBinding,
    ) -> FirecrackerQualificationWallTimeoutMeasurement:
        armed = running.launched_at_monotonic_ns
        deadline = running.wall_deadline_monotonic_ns
        while not running.watchdog_timeout_triggered.wait(0.02):
            if self.monotonic_ns() > deadline + 5_000_000_000:
                raise FirecrackerQualificationDriverError('Firecracker wall watchdog did not trigger')
        members = binding.cgroup_member_pids
        cleanup = self.supervisor.terminate_and_cleanup(running, grace_seconds=5.0)
        triggered = cleanup.watchdog_triggered_monotonic_ns
        cgroup_empty = cleanup.cgroup_empty_monotonic_ns
        if (
            triggered is None
            or cgroup_empty is None
            or cleanup.launched_monotonic_ns != armed
            or cleanup.wall_deadline_monotonic_ns != deadline
        ):
            raise FirecrackerQualificationDriverError(
                'Firecracker timeout cleanup omitted its actual monotonic observations'
            )
        survivors = tuple(pid for pid in members if (self.reader.proc_root / str(pid)).exists())
        return FirecrackerQualificationWallTimeoutMeasurement(
            run_id=binding.run_id,
            process_group_id=binding.process_group_id,
            armed_monotonic_ns=armed,
            deadline_monotonic_ns=deadline,
            watchdog_triggered_monotonic_ns=triggered,
            process_group_reaped_monotonic_ns=cgroup_empty,
            allowed_teardown_grace_ns=10_000_000_000,
            member_pids_before_kill=members,
            surviving_pids_after_reap=survivors,
        )

    def _teardown(self, binding: FirecrackerQualificationWorkerBinding) -> FirecrackerQualificationTeardownMeasurement:
        paths = (Path(binding.cgroup_path), Path(binding.jail_root), Path(binding.vsock_uds_path))
        if not all(self.reader.path_absent(path) for path in paths):
            raise FirecrackerQualificationDriverError('worker teardown left cgroup, jail, or vsock state')
        return FirecrackerQualificationTeardownMeasurement(
            run_id=binding.run_id,
            cgroup_path=binding.cgroup_path,
            jail_root=binding.jail_root,
            vsock_uds_path=binding.vsock_uds_path,
            observed_at=self.clock(),
        )

    def _observed_claims(
        self,
        *,
        drill: FirecrackerQualificationDrillId,
        response: AuthenticatedFirecrackerQualificationGuestResponse | None,
        binding: FirecrackerQualificationWorkerBinding,
        clean_exit: bool,
        snapshots: tuple[FirecrackerQualificationCgroupSnapshot, ...],
        wall: FirecrackerQualificationWallTimeoutMeasurement | None,
        teardown: tuple[FirecrackerQualificationTeardownMeasurement, ...],
    ) -> dict[FirecrackerQualificationClaim, bool]:
        guest_claims = set() if response is None else set(response.response.verified_guest_claims)
        observed = {claim: claim in guest_claims for claim in required_firecracker_qualification_claims(drill)}
        if drill == FirecrackerQualificationDrillId.LIVE_BOOT:
            observed[FirecrackerQualificationClaim.FIRECRACKER_PROCESS_STARTED] = binding.process_tree_verified
            observed[FirecrackerQualificationClaim.CLEAN_GUEST_EXIT] = clean_exit
        elif drill == FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP:
            observed[FirecrackerQualificationClaim.HOST_VSOCK_HANDSHAKE] = response is not None
            observed[FirecrackerQualificationClaim.PEER_CID_BOUND] = (
                response is not None
                and binding.guest_cid == self.spec.guest_cid
                and binding.peer_pid == binding.firecracker_pid
            )
        elif drill == FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT:
            before, after = snapshots[0], snapshots[-1]
            observed[FirecrackerQualificationClaim.CPU_LIMIT_OBSERVED] = (
                after.cpu_nr_throttled > before.cpu_nr_throttled
                and after.cpu_throttled_usec > before.cpu_throttled_usec
            )
            observed[FirecrackerQualificationClaim.MEMORY_LIMIT_OBSERVED] = (
                after.memory_oom > before.memory_oom or after.memory_oom_kill > before.memory_oom_kill
            )
            observed[FirecrackerQualificationClaim.SWAP_DISABLED_OBSERVED] = (
                before.memory_swap_max_bytes == 0 and after.memory_swap_max_bytes == 0
            )
            observed[FirecrackerQualificationClaim.PIDS_LIMIT_OBSERVED] = after.pids_max_events > before.pids_max_events
        elif drill == FirecrackerQualificationDrillId.WALL_TIMEOUT:
            observed.update(
                {
                    FirecrackerQualificationClaim.WALL_WATCHDOG_TRIGGERED: wall is not None,
                    FirecrackerQualificationClaim.PROCESS_GROUP_KILLED: (
                        wall is not None and not wall.surviving_pids_after_reap
                    ),
                    FirecrackerQualificationClaim.DEADLINE_BOUND: wall is not None,
                }
            )
        elif drill == FirecrackerQualificationDrillId.TEARDOWN:
            success = len(teardown) == 1
            observed.update(
                {
                    FirecrackerQualificationClaim.CGROUP_ABSENT: success,
                    FirecrackerQualificationClaim.JAIL_ABSENT: success,
                    FirecrackerQualificationClaim.VSOCK_ABSENT: success,
                }
            )
        return observed

    def _measurements(
        self,
        *,
        drill: FirecrackerQualificationDrillId,
        observed: dict[FirecrackerQualificationClaim, bool],
        observed_at: datetime,
    ) -> tuple[FirecrackerQualificationClaimMeasurement, ...]:
        return tuple(
            FirecrackerQualificationClaimMeasurement(
                claim=claim,
                source=_SOURCE_BY_CLAIM[claim],
                observed=observed.get(claim, False),
                raw_observation_sha256=hashlib.sha256(
                    canonical_json_bytes(
                        {
                            'challenge_sha256': firecracker_model_sha256(self.request.challenge),
                            'claim': claim,
                            'observed': observed.get(claim, False),
                            'source': _SOURCE_BY_CLAIM[claim],
                        }
                    )
                ).hexdigest(),
                observed_at=observed_at,
            )
            for claim in required_firecracker_qualification_claims(drill)
        )


def run_firecracker_qualification_drill(
    request: FirecrackerQualificationDriverRequest,
) -> FirecrackerQualificationRawDrillObservation:
    return LinuxKvmFirecrackerQualificationDriver(request).run()


def main() -> None:
    parser = argparse.ArgumentParser(description='directly run one Firecracker Linux/KVM qualification drill')
    subparsers = parser.add_subparsers(dest='command', required=True)
    run = subparsers.add_parser('run-drill')
    run.add_argument('--protocol', required=True, choices=(FIRECRACKER_QUALIFICATION_DRIVER_PROTOCOL,))
    arguments = parser.parse_args()
    del arguments
    try:
        request_bytes = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
        if not request_bytes or len(request_bytes) > _MAX_REQUEST_BYTES:
            raise FirecrackerQualificationDriverError('qualification driver request size is invalid')
        request = FirecrackerQualificationDriverRequest.model_validate_json(request_bytes)
        if canonical_json_bytes(request) != request_bytes:
            raise FirecrackerQualificationDriverError('qualification driver request is not canonical JSON')
        observation = run_firecracker_qualification_drill(request)
        sys.stdout.buffer.write(canonical_json_bytes(observation))
        sys.stdout.buffer.flush()
    except (FirecrackerQualificationDriverError, ValueError) as error:
        sys.stderr.write(f'qualification driver rejected: {error}\n')
        raise SystemExit(64) from error


def _require_root_linux_kvm() -> None:
    if platform.system() != 'Linux' or os.geteuid() != 0:
        raise FirecrackerQualificationDriverError('qualification run-drill requires effective root on Linux')
    try:
        metadata = Path('/dev/kvm').lstat()
    except OSError as error:
        raise FirecrackerQualificationDriverError('/dev/kvm is unavailable') from error
    unsafe_kvm = stat.S_ISLNK(metadata.st_mode) or not stat.S_ISCHR(metadata.st_mode)
    if unsafe_kvm or not os.access('/dev/kvm', os.R_OK | os.W_OK):
        raise FirecrackerQualificationDriverError('/dev/kvm must be a readable non-symlink character device')


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    content = bytearray()
    while len(content) < count:
        chunk = connection.recv(count - len(content))
        if not chunk:
            raise FirecrackerQualificationDriverError('qualification guest connection closed mid-frame')
        content.extend(chunk)
    return bytes(content)


def _read_small_file(path: Path, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FirecrackerQualificationDriverError('kernel evidence path is not a regular pseudo-file')
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(4096, maximum + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > maximum:
                raise FirecrackerQualificationDriverError('kernel evidence file is oversized')
        return bytes(content)
    finally:
        os.close(descriptor)


def _proc_executable_sha256(path: Path) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise FirecrackerQualificationDriverError('cannot open the launched Firecracker executable') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FirecrackerQualificationDriverError('launched Firecracker executable is not a regular file')
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _proc_process_identity(path: Path) -> tuple[int, int, int, int]:
    content = _read_small_file(path, 64 * 1024)
    closing_parenthesis = content.rfind(b')')
    if closing_parenthesis < 2:
        raise FirecrackerQualificationDriverError('procfs process stat is malformed')
    fields_after_command = content[closing_parenthesis + 1 :].split()
    if len(fields_after_command) <= 19:
        raise FirecrackerQualificationDriverError('procfs process stat lacks process identity fields')
    try:
        parent_pid = int(fields_after_command[1])
        process_group_id = int(fields_after_command[2])
        session_id = int(fields_after_command[3])
        start_time = int(fields_after_command[19])
    except ValueError as error:
        raise FirecrackerQualificationDriverError('procfs process identity is malformed') from error
    if parent_pid < 0 or process_group_id <= 1 or session_id <= 0 or start_time <= 0:
        raise FirecrackerQualificationDriverError('procfs process identity is invalid')
    return parent_pid, process_group_id, session_id, start_time


def _read_ascii(path: Path) -> str:
    try:
        return _read_small_file(path, 64 * 1024).decode('ascii').strip()
    except (OSError, UnicodeDecodeError) as error:
        raise FirecrackerQualificationDriverError('kernel evidence is not bounded ASCII') from error


def _read_tokens(path: Path, *, expected: int) -> tuple[str, ...]:
    values = tuple(_read_ascii(path).split())
    if len(values) != expected:
        raise FirecrackerQualificationDriverError('kernel evidence token count is invalid')
    return values


def _read_keyed_integers(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in _read_ascii(path).splitlines():
        pieces = line.split()
        if len(pieces) != 2 or pieces[0] in values:
            raise FirecrackerQualificationDriverError('kernel counter evidence is malformed')
        values[pieces[0]] = _bounded_integer(pieces[1], label=pieces[0])
    return values


def _read_integer_lines(path: Path) -> tuple[int, ...]:
    values = tuple(_bounded_integer(value, label='PID') for value in _read_ascii(path).splitlines() if value)
    if any(value <= 1 for value in values):
        raise FirecrackerQualificationDriverError('kernel PID evidence contains a reserved PID')
    return tuple(sorted(set(values)))


def _bounded_integer(value: str, *, label: str) -> int:
    if not value or not value.isdigit() or len(value) > 20:
        raise FirecrackerQualificationDriverError(f'{label} is not a bounded integer')
    observed = int(value)
    if observed < 0 or observed > 2**63 - 1:
        raise FirecrackerQualificationDriverError(f'{label} is out of range')
    return observed


__all__ = [
    'FIRECRACKER_QUALIFICATION_DRIVER_PROTOCOL',
    'FirecrackerQualificationDriverError',
    'FirecrackerQualificationDriverRequest',
    'LinuxKvmFirecrackerQualificationDriver',
    'LinuxQualificationEvidenceReader',
    'main',
    'run_firecracker_qualification_drill',
]
