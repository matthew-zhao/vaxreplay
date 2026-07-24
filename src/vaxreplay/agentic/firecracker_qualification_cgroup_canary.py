"""Bounded host-side memory and PID controller canaries for qualification only.

Guest forks are virtual-machine PIDs, so they cannot exercise the host's ``pids.max``.  Likewise,
guest allocation normally consumes RAM already mapped by Firecracker and need not cross the host
``memory.max``.  This module creates two short-lived host helpers, moves each through the already
open exact worker-cgroup descriptor, and proves controller events plus complete helper cleanup.

This code is never used by a submitted model run.  A failure tears down the qualification worker;
it never retries a model or promotes a partial observation.
"""

from __future__ import annotations

import errno
import os
import resource
import select
import signal
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass

from vaxreplay.agentic.firecracker import FirecrackerWorkerSpec, RunningFirecrackerWorker, firecracker_model_sha256
from vaxreplay.agentic.firecracker_qualification_probe import (
    FirecrackerQualificationCgroupSnapshot,
    FirecrackerQualificationHostCgroupCanary,
    FirecrackerQualificationWorkerBinding,
)

_MIB = 1024 * 1024
_MAX_MEMORY_CANARY_BYTES = 1024 * _MIB
_MAX_PIDS_CANARY = 512
_MAX_CANARY_DURATION_NS = 8_000_000_000
_MIN_CANARY_DURATION_NS = 750_000_000
_WATCHDOG_RESERVE_NS = 250_000_000
_POLL_SECONDS = 0.01


class FirecrackerQualificationCgroupCanaryError(RuntimeError):
    """A host controller canary could not be bounded, measured, or fully reaped."""


@dataclass(frozen=True)
class FirecrackerQualificationCgroupCanaryResult:
    snapshots: tuple[FirecrackerQualificationCgroupSnapshot, ...]
    measurement: FirecrackerQualificationHostCgroupCanary


@dataclass
class _PausedHelper:
    pid: int
    pidfd: int
    command_fd: int
    status_fd: int
    wait_status: int | None = None


def run_host_cgroup_controller_canary(
    *,
    spec: FirecrackerWorkerSpec,
    running: RunningFirecrackerWorker,
    binding: FirecrackerQualificationWorkerBinding,
    baseline: FirecrackerQualificationCgroupSnapshot,
    guest_pressure: FirecrackerQualificationCgroupSnapshot,
    snapshot: Callable[[], FirecrackerQualificationCgroupSnapshot],
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> FirecrackerQualificationCgroupCanaryResult:
    """Exercise ``memory.max`` and ``pids.max`` in the exact bound worker cgroup.

    The caller already owns a dedicated qualification VM.  Helpers are direct children of this
    driver, start paused outside the cgroup, and cannot act until the root parent moves them through
    Firecracker's retained cgroup directory descriptor.  Linux pidfds prevent cleanup from ever
    signaling a reused PID.
    """

    _require_safe_limits(spec)
    _verify_exact_cgroup(running=running, binding=binding)
    _require_oom_group_disabled(running.cgroup_descriptor)
    _validate_snapshot_identity(baseline, binding=binding, expected_members=binding.cgroup_member_pids)
    _validate_snapshot_identity(guest_pressure, binding=binding, expected_members=binding.cgroup_member_pids)

    started_ns = monotonic_ns()
    available_ns = running.wall_deadline_monotonic_ns - started_ns - _WATCHDOG_RESERVE_NS
    allowed_duration_ns = min(_MAX_CANARY_DURATION_NS, available_ns)
    if allowed_duration_ns < _MIN_CANARY_DURATION_NS:
        raise FirecrackerQualificationCgroupCanaryError('worker wall watchdog leaves no bounded canary interval')
    deadline_ns = started_ns + allowed_duration_ns

    memory_helper: _PausedHelper | None = None
    pids_helper: _PausedHelper | None = None
    descendant_pidfds: dict[int, int] = {}
    try:
        virtual_size_bytes = _process_virtual_size_bytes()
        memory_helper = _spawn_paused_helper(
            action='memory',
            spec=spec,
            memory_address_space_bytes=virtual_size_bytes + baseline.memory_max_bytes + 64 * _MIB,
        )
        _move_helper_to_exact_cgroup(memory_helper, running=running, binding=binding, deadline_ns=deadline_ns)
        memory_armed = snapshot()
        _validate_snapshot_identity(
            memory_armed,
            binding=binding,
            expected_members=tuple(sorted((*binding.cgroup_member_pids, memory_helper.pid))),
        )
        _command(memory_helper, b'G')
        memory_status = _wait_for_memory_event_and_reap(
            memory_helper,
            before=memory_armed,
            snapshot=snapshot,
            deadline_ns=deadline_ns,
            monotonic_ns=monotonic_ns,
        )
        memory_helper.wait_status = memory_status
        if not os.WIFSIGNALED(memory_status) or os.WTERMSIG(memory_status) != signal.SIGKILL:
            raise FirecrackerQualificationCgroupCanaryError('memory helper was not killed by the cgroup OOM path')
        memory_reaped_ns = monotonic_ns()
        memory_triggered = snapshot()
        _validate_snapshot_identity(
            memory_triggered,
            binding=binding,
            expected_members=binding.cgroup_member_pids,
        )
        if (
            memory_triggered.memory_oom <= memory_armed.memory_oom
            or memory_triggered.memory_oom_kill <= memory_armed.memory_oom_kill
        ):
            raise FirecrackerQualificationCgroupCanaryError('memory helper did not increment exact cgroup OOM counters')

        pids_helper = _spawn_paused_helper(action='pids', spec=spec, memory_address_space_bytes=0)
        _move_helper_to_exact_cgroup(pids_helper, running=running, binding=binding, deadline_ns=deadline_ns)
        _command(pids_helper, b'G')
        if _read_status_byte(pids_helper, deadline_ns=deadline_ns, monotonic_ns=monotonic_ns) != b'L':
            raise FirecrackerQualificationCgroupCanaryError('PID helper did not report a controller rejection')
        pids_limit_observed_ns = monotonic_ns()
        pids_peak = snapshot()
        _validate_snapshot_identity(pids_peak, binding=binding)
        peak_members = set(pids_peak.member_pids)
        descendants = tuple(sorted(peak_members - set(binding.cgroup_member_pids) - {pids_helper.pid}))
        if (
            not descendants
            or len(descendants) > _MAX_PIDS_CANARY
            or peak_members != set(binding.cgroup_member_pids) | {pids_helper.pid, *descendants}
            or pids_peak.pids_max_events <= memory_triggered.pids_max_events
        ):
            raise FirecrackerQualificationCgroupCanaryError('PID helper evidence is incomplete or unbound')
        descendant_pidfds = {pid: _open_pidfd(pid) for pid in descendants}
        for pid in descendants:
            _require_process_in_exact_cgroup(pid, binding.cgroup_path)
        _command(pids_helper, b'R')
        if _read_status_byte(pids_helper, deadline_ns=deadline_ns, monotonic_ns=monotonic_ns) != b'C':
            raise FirecrackerQualificationCgroupCanaryError('PID helper did not confirm descendant cleanup')
        pids_status = _wait_direct_child(pids_helper, deadline_ns=deadline_ns, monotonic_ns=monotonic_ns)
        pids_helper.wait_status = pids_status
        if not os.WIFEXITED(pids_status) or os.WEXITSTATUS(pids_status) != 0:
            raise FirecrackerQualificationCgroupCanaryError('PID helper cleanup exited unsuccessfully')
        pids_reaped_ns = monotonic_ns()
        for descriptor in descendant_pidfds.values():
            _wait_pidfd_exit(descriptor, deadline_ns=deadline_ns, monotonic_ns=monotonic_ns)

        cleanup = snapshot()
        _validate_snapshot_identity(cleanup, binding=binding, expected_members=binding.cgroup_member_pids)
        finished_ns = monotonic_ns()
        measurement = FirecrackerQualificationHostCgroupCanary(
            run_id=binding.run_id,
            cgroup_path=binding.cgroup_path,
            cgroup_inode=binding.cgroup_inode,
            firecracker_pid=binding.firecracker_pid,
            memory_helper_pid=memory_helper.pid,
            pids_helper_pid=pids_helper.pid,
            pids_helper_descendant_pids=descendants,
            started_monotonic_ns=started_ns,
            memory_helper_reaped_monotonic_ns=memory_reaped_ns,
            pids_limit_observed_monotonic_ns=pids_limit_observed_ns,
            pids_helper_reaped_monotonic_ns=pids_reaped_ns,
            finished_monotonic_ns=finished_ns,
            allowed_duration_ns=allowed_duration_ns,
            baseline_snapshot_sha256=firecracker_model_sha256(baseline),
            guest_pressure_snapshot_sha256=firecracker_model_sha256(guest_pressure),
            memory_armed_snapshot_sha256=firecracker_model_sha256(memory_armed),
            memory_triggered_snapshot_sha256=firecracker_model_sha256(memory_triggered),
            pids_peak_snapshot_sha256=firecracker_model_sha256(pids_peak),
            cleanup_snapshot_sha256=firecracker_model_sha256(cleanup),
        )
        return FirecrackerQualificationCgroupCanaryResult(
            snapshots=(baseline, guest_pressure, memory_armed, memory_triggered, pids_peak, cleanup),
            measurement=measurement,
        )
    finally:
        cleanup_errors: list[BaseException] = []
        for helper in (memory_helper, pids_helper):
            if helper is not None:
                try:
                    _cleanup_helper(helper, monotonic_ns=monotonic_ns)
                except BaseException as error:
                    cleanup_errors.append(error)
        try:
            residual = snapshot()
            residual_helpers = tuple(sorted(set(residual.member_pids) - set(binding.cgroup_member_pids)))
            for pid in residual_helpers:
                descriptor = _open_pidfd(pid)
                try:
                    _signal_pidfd(descriptor, signal.SIGKILL, tolerate_gone=True)
                    _wait_pidfd_exit(
                        descriptor,
                        deadline_ns=monotonic_ns() + 1_000_000_000,
                        monotonic_ns=monotonic_ns,
                    )
                finally:
                    os.close(descriptor)
            residual = snapshot()
            _validate_snapshot_identity(
                residual,
                binding=binding,
                expected_members=binding.cgroup_member_pids,
            )
        except BaseException as error:
            cleanup_errors.append(error)
        for descriptor in descendant_pidfds.values():
            try:
                _signal_pidfd(descriptor, signal.SIGKILL, tolerate_gone=True)
                _wait_pidfd_exit(
                    descriptor,
                    deadline_ns=monotonic_ns() + 1_000_000_000,
                    monotonic_ns=monotonic_ns,
                    tolerate_timeout=True,
                )
                os.close(descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise FirecrackerQualificationCgroupCanaryError(
                'cannot prove cleanup of every canary helper'
            ) from cleanup_errors[0]


def _require_safe_limits(spec: FirecrackerWorkerSpec) -> None:
    memory_bytes = spec.limits.memory_mib * _MIB
    if memory_bytes > _MAX_MEMORY_CANARY_BYTES:
        raise FirecrackerQualificationCgroupCanaryError('memory.max exceeds the bounded host-canary admission limit')
    if spec.limits.pids > _MAX_PIDS_CANARY:
        raise FirecrackerQualificationCgroupCanaryError('pids.max exceeds the bounded host-canary admission limit')


def _verify_exact_cgroup(*, running: RunningFirecrackerWorker, binding: FirecrackerQualificationWorkerBinding) -> None:
    try:
        descriptor_metadata = os.fstat(running.cgroup_descriptor)
        path_metadata = os.lstat(binding.cgroup_path)
    except OSError as error:
        raise FirecrackerQualificationCgroupCanaryError('exact worker cgroup is unavailable') from error
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or not stat.S_ISDIR(path_metadata.st_mode)
        or (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != (running.cgroup_device_id, running.cgroup_inode)
        or (path_metadata.st_dev, path_metadata.st_ino) != (running.cgroup_device_id, running.cgroup_inode)
        or path_metadata.st_ino != binding.cgroup_inode
    ):
        raise FirecrackerQualificationCgroupCanaryError('worker cgroup descriptor, path, and binding differ')


def _require_oom_group_disabled(cgroup_descriptor: int) -> None:
    descriptor = _openat(cgroup_descriptor, 'memory.oom.group', os.O_RDONLY)
    try:
        if os.read(descriptor, 8).strip() != b'0':
            raise FirecrackerQualificationCgroupCanaryError('memory.oom.group must be disabled for helper-only OOM')
    finally:
        os.close(descriptor)


def _spawn_paused_helper(*, action: str, spec: FirecrackerWorkerSpec, memory_address_space_bytes: int) -> _PausedHelper:
    command_read, command_write = os.pipe()
    status_read, status_write = os.pipe()
    for descriptor in (command_read, command_write, status_read, status_write):
        os.set_inheritable(descriptor, False)
    try:
        pid = os.fork()
    except OSError:
        for descriptor in (command_read, command_write, status_read, status_write):
            os.close(descriptor)
        raise
    if pid == 0:
        os.close(command_write)
        os.close(status_read)
        try:
            _close_unneeded_fds({command_read, status_write})
            os.write(status_write, b'P')
            if os.read(command_read, 1) != b'G':
                os._exit(70)
            _run_released_helper_action(
                action=action,
                spec=spec,
                command_read=command_read,
                status_write=status_write,
                memory_address_space_bytes=memory_address_space_bytes,
            )
            os._exit(70)
        except BaseException:
            try:
                os.write(status_write, b'E')
            except OSError:
                pass
            os._exit(70)
    os.close(command_read)
    os.close(status_write)
    try:
        pidfd = _open_pidfd(pid)
    except BaseException:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(command_write)
        os.close(status_read)
        raise
    return _PausedHelper(pid=pid, pidfd=pidfd, command_fd=command_write, status_fd=status_read)


def _move_helper_to_exact_cgroup(
    helper: _PausedHelper,
    *,
    running: RunningFirecrackerWorker,
    binding: FirecrackerQualificationWorkerBinding,
    deadline_ns: int,
) -> None:
    if _read_status_byte(helper, deadline_ns=deadline_ns, monotonic_ns=time.monotonic_ns) != b'P':
        raise FirecrackerQualificationCgroupCanaryError('canary helper did not start paused')
    descriptor = _openat(running.cgroup_descriptor, 'cgroup.procs', os.O_WRONLY)
    try:
        content = f'{helper.pid}\n'.encode('ascii')
        if os.write(descriptor, content) != len(content):
            raise FirecrackerQualificationCgroupCanaryError('short write while moving helper into exact cgroup')
    finally:
        os.close(descriptor)
    _require_process_in_exact_cgroup(helper.pid, binding.cgroup_path)


def _run_released_helper_action(
    *,
    action: str,
    spec: FirecrackerWorkerSpec,
    command_read: int,
    status_write: int,
    memory_address_space_bytes: int,
) -> None:
    """Finish privileged setup only after the parent moved and released the paused helper."""

    if action == 'memory':
        _set_and_verify_oom_score_adj()
    _drop_privileges(spec)
    if action == 'memory':
        _memory_child(memory_address_space_bytes)
    elif action == 'pids':
        _pids_child(command_read, status_write, spec.limits.pids)
    else:
        raise FirecrackerQualificationCgroupCanaryError('unknown host canary helper action')


def _set_and_verify_oom_score_adj() -> None:
    """Make only the allocator the OOM target before the root-to-worker UID transition."""

    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open('/proc/self/oom_score_adj', flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FirecrackerQualificationCgroupCanaryError('OOM preference control changed type')
        value = b'1000'
        if os.write(descriptor, value) != len(value):
            raise FirecrackerQualificationCgroupCanaryError('OOM preference control write was incomplete')
        if os.lseek(descriptor, 0, os.SEEK_SET) != 0 or os.read(descriptor, 32).strip() != value:
            raise FirecrackerQualificationCgroupCanaryError('OOM preference control did not retain 1000')
    except OSError as error:
        raise FirecrackerQualificationCgroupCanaryError('cannot set and verify helper OOM preference') from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _memory_child(address_space_bytes: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
    allocations: list[bytearray] = []
    try:
        while True:
            block = bytearray(_MIB)
            block[::4096] = b'\x01' * (len(block) // 4096)
            allocations.append(block)
    except MemoryError:
        os._exit(71)


def _pids_child(command_read: int, status_write: int, pids_max: int) -> None:
    children: list[int] = []
    hit_limit = False
    try:
        for _ in range(pids_max + 1):
            try:
                pid = os.fork()
            except OSError as error:
                if error.errno == errno.EAGAIN:
                    hit_limit = True
                    break
                raise
            if pid == 0:
                os.close(command_read)
                os.close(status_write)
                while True:
                    signal.pause()
            children.append(pid)
        if not hit_limit:
            os.write(status_write, b'F')
            return
        os.write(status_write, b'L')
        if os.read(command_read, 1) != b'R':
            return
    finally:
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in children:
            while True:
                try:
                    waited, _ = os.waitpid(pid, 0)
                except InterruptedError:
                    continue
                except ChildProcessError:
                    break
                if waited == pid:
                    break
    if hit_limit:
        os.write(status_write, b'C')
        os._exit(0)


def _drop_privileges(spec: FirecrackerWorkerSpec) -> None:
    os.setgroups([])
    os.setgid(spec.worker_gid)
    os.setuid(spec.worker_uid)
    os.umask(0o077)


def _close_unneeded_fds(keep: set[int]) -> None:
    """Drop inherited VM/cgroup/listener capabilities before a helper can act."""

    try:
        descriptors = tuple(int(name) for name in os.listdir('/proc/self/fd') if name.isdigit())
    except OSError:
        os._exit(70)
    for descriptor in descriptors:
        if descriptor > 2 and descriptor not in keep:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _wait_for_memory_event_and_reap(
    helper: _PausedHelper,
    *,
    before: FirecrackerQualificationCgroupSnapshot,
    snapshot: Callable[[], FirecrackerQualificationCgroupSnapshot],
    deadline_ns: int,
    monotonic_ns: Callable[[], int],
) -> int:
    reaped_status: int | None = None
    latest = before
    while monotonic_ns() < deadline_ns:
        if reaped_status is None:
            reaped_status = _wait_direct_child_nohang(helper)
            if reaped_status is not None and (
                not os.WIFSIGNALED(reaped_status) or os.WTERMSIG(reaped_status) != signal.SIGKILL
            ):
                latest = snapshot()
                raise FirecrackerQualificationCgroupCanaryError(
                    'memory helper exited before cgroup OOM enforcement '
                    f'({_describe_wait_status(reaped_status)}; '
                    f'oom={latest.memory_oom}, oom_kill={latest.memory_oom_kill})'
                )
        if reaped_status is not None:
            latest = snapshot()
            if latest.memory_oom > before.memory_oom and latest.memory_oom_kill > before.memory_oom_kill:
                return reaped_status
        time.sleep(_POLL_SECONDS)
    if reaped_status is None:
        raise FirecrackerQualificationCgroupCanaryError('memory helper exceeded the bounded canary interval')
    raise FirecrackerQualificationCgroupCanaryError(
        'memory helper received SIGKILL but exact cgroup OOM counters did not advance '
        f'(before oom={before.memory_oom}, oom_kill={before.memory_oom_kill}; '
        f'latest oom={latest.memory_oom}, oom_kill={latest.memory_oom_kill})'
    )


def _describe_wait_status(status: int) -> str:
    if os.WIFEXITED(status):
        return f'exit={os.WEXITSTATUS(status)}'
    if os.WIFSIGNALED(status):
        return f'signal={os.WTERMSIG(status)}'
    if os.WIFSTOPPED(status):
        return f'stopped={os.WSTOPSIG(status)}'
    return f'raw_wait_status={status}'


def _wait_direct_child(helper: _PausedHelper, *, deadline_ns: int, monotonic_ns: Callable[[], int]) -> int:
    if helper.wait_status is not None:
        return helper.wait_status
    while monotonic_ns() < deadline_ns:
        status = _wait_direct_child_nohang(helper)
        if status is not None:
            return status
        time.sleep(_POLL_SECONDS)
    raise FirecrackerQualificationCgroupCanaryError('canary helper did not exit before its deadline')


def _wait_direct_child_nohang(helper: _PausedHelper) -> int | None:
    if helper.wait_status is not None:
        return helper.wait_status
    try:
        waited, status = os.waitpid(helper.pid, os.WNOHANG)
    except ChildProcessError:
        if helper.wait_status is None:
            raise FirecrackerQualificationCgroupCanaryError('canary helper lost its wait status') from None
        return helper.wait_status
    if waited == 0:
        return None
    helper.wait_status = status
    return status


def _cleanup_helper(helper: _PausedHelper, *, monotonic_ns: Callable[[], int]) -> None:
    try:
        if helper.wait_status is None:
            try:
                _command(helper, b'R')
            except (BrokenPipeError, FirecrackerQualificationCgroupCanaryError, OSError):
                pass
            gentle_deadline = monotonic_ns() + 100_000_000
            while helper.wait_status is None and monotonic_ns() < gentle_deadline:
                helper.wait_status = _wait_direct_child_nohang(helper)
                if helper.wait_status is None:
                    time.sleep(_POLL_SECONDS)
            if helper.wait_status is None:
                _signal_pidfd(helper.pidfd, signal.SIGKILL, tolerate_gone=True)
                helper.wait_status = _wait_direct_child(
                    helper,
                    deadline_ns=monotonic_ns() + 1_000_000_000,
                    monotonic_ns=monotonic_ns,
                )
    finally:
        for descriptor in (helper.command_fd, helper.status_fd, helper.pidfd):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_status_byte(helper: _PausedHelper, *, deadline_ns: int, monotonic_ns: Callable[[], int]) -> bytes:
    while monotonic_ns() < deadline_ns:
        timeout = min(0.05, max(0.0, (deadline_ns - monotonic_ns()) / 1_000_000_000))
        readable, _, _ = select.select((helper.status_fd,), (), (), timeout)
        if readable:
            value = os.read(helper.status_fd, 1)
            if not value:
                raise FirecrackerQualificationCgroupCanaryError('canary helper closed its status channel')
            return value
    raise FirecrackerQualificationCgroupCanaryError('canary helper status timed out')


def _command(helper: _PausedHelper, value: bytes) -> None:
    if len(value) != 1 or os.write(helper.command_fd, value) != 1:
        raise FirecrackerQualificationCgroupCanaryError('canary helper command write failed')


def _open_pidfd(pid: int) -> int:
    opener = getattr(os, 'pidfd_open', None)
    sender = getattr(signal, 'pidfd_send_signal', None)
    if opener is None or sender is None:
        raise FirecrackerQualificationCgroupCanaryError('Linux pidfd cleanup primitives are unavailable')
    try:
        return opener(pid, 0)
    except OSError as error:
        raise FirecrackerQualificationCgroupCanaryError('cannot bind canary helper pidfd') from error


def _signal_pidfd(descriptor: int, signal_number: int, *, tolerate_gone: bool) -> None:
    sender = getattr(signal, 'pidfd_send_signal', None)
    if sender is None:
        raise FirecrackerQualificationCgroupCanaryError('Linux pidfd signaling is unavailable')
    try:
        sender(descriptor, signal_number, None, 0)
    except ProcessLookupError:
        if not tolerate_gone:
            raise


def _wait_pidfd_exit(
    descriptor: int,
    *,
    deadline_ns: int,
    monotonic_ns: Callable[[], int],
    tolerate_timeout: bool = False,
) -> None:
    while monotonic_ns() < deadline_ns:
        readable, _, _ = select.select((descriptor,), (), (), _POLL_SECONDS)
        if readable:
            return
    if not tolerate_timeout:
        raise FirecrackerQualificationCgroupCanaryError('canary descendant did not exit before its deadline')


def _openat(directory: int, name: str, access: int) -> int:
    flags = access | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise FirecrackerQualificationCgroupCanaryError(f'cannot open exact cgroup {name}') from error
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise FirecrackerQualificationCgroupCanaryError(f'exact cgroup {name} changed type')
    return descriptor


def _require_process_in_exact_cgroup(pid: int, cgroup_path: str) -> None:
    relative = '/' + cgroup_path.removeprefix('/sys/fs/cgroup/').lstrip('/')
    try:
        descriptor = os.open(f'/proc/{pid}/cgroup', os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            content = os.read(descriptor, 64 * 1024).decode('ascii')
        finally:
            os.close(descriptor)
    except (OSError, UnicodeDecodeError) as error:
        raise FirecrackerQualificationCgroupCanaryError('cannot verify canary helper cgroup membership') from error
    if f'0::{relative}' not in content.splitlines():
        raise FirecrackerQualificationCgroupCanaryError('canary helper is not in the exact worker cgroup')


def _validate_snapshot_identity(
    snapshot: FirecrackerQualificationCgroupSnapshot,
    *,
    binding: FirecrackerQualificationWorkerBinding,
    expected_members: tuple[int, ...] | None = None,
) -> None:
    if (
        (snapshot.run_id, snapshot.cgroup_path, snapshot.cgroup_inode)
        != (binding.run_id, binding.cgroup_path, binding.cgroup_inode)
        or binding.firecracker_pid not in snapshot.member_pids
        or (expected_members is not None and snapshot.member_pids != expected_members)
    ):
        raise FirecrackerQualificationCgroupCanaryError('canary snapshot is not bound to the exact live worker')


def _process_virtual_size_bytes() -> int:
    try:
        fields = open('/proc/self/statm', encoding='ascii').read(256).split()  # noqa: SIM115 - procfs one-shot read
        pages = int(fields[0])
    except (OSError, ValueError, IndexError) as error:
        raise FirecrackerQualificationCgroupCanaryError('cannot bound memory helper address space') from error
    return pages * os.sysconf('SC_PAGE_SIZE')


__all__ = [
    'FirecrackerQualificationCgroupCanaryError',
    'FirecrackerQualificationCgroupCanaryResult',
    'run_host_cgroup_controller_canary',
]
