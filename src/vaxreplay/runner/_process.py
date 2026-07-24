"""Bounded host-process capture used to supervise an isolation runtime CLI."""

from __future__ import annotations

import math
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal


@dataclass(frozen=True)
class BoundedProcessResult:
    exit_code: int | None
    duration_ms: int
    stdout: bytes
    stderr: bytes
    termination: Literal['exited', 'timed_out', 'response_limit', 'log_limit']
    stdout_truncated: bool
    stderr_truncated: bool


def run_bounded_process(
    argv: Sequence[str],
    *,
    input_bytes: bytes,
    wall_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    on_abort: Callable[[], None],
    env: Mapping[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
    cwd: str | Path | None = None,
) -> BoundedProcessResult:
    """Run argv without a shell while bounding both output streams and wall time."""

    if not argv or any(not argument or '\x00' in argument for argument in argv):
        raise ValueError('argv must contain non-empty, NUL-free arguments')
    if (
        not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError('wall_seconds must be a finite positive number')
    if (
        not isinstance(pass_fds, tuple)
        or any(
            not isinstance(descriptor, int) or isinstance(descriptor, bool) or descriptor < 0 for descriptor in pass_fds
        )
        or len(pass_fds) != len(set(pass_fds))
    ):
        raise ValueError('pass_fds must be an exact tuple of unique nonnegative integer descriptors')
    started = time.monotonic()
    process = subprocess.Popen(  # noqa: S603 - argv is constructed by the trusted backend
        tuple(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        shell=False,
        close_fds=True,
        pass_fds=pass_fds,
        start_new_session=True,
        cwd=cwd,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    termination_events: queue.SimpleQueue[Literal['response_limit', 'log_limit']] = queue.SimpleQueue()
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    reader_errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

    stdout_thread = threading.Thread(
        target=_read_bounded,
        args=(process.stdout, stdout_buffer, max_stdout_bytes, 'response_limit', termination_events, reader_errors),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_bounded,
        args=(process.stderr, stderr_buffer, max_stderr_bytes, 'log_limit', termination_events, reader_errors),
        daemon=True,
    )
    stdin_thread = threading.Thread(
        target=_write_input,
        args=(process.stdin, input_bytes),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    stdin_thread.start()

    termination: Literal['exited', 'timed_out', 'response_limit', 'log_limit'] = 'exited'
    observed_limits: set[Literal['response_limit', 'log_limit']] = set()
    deadline = started + wall_seconds
    while process.poll() is None:
        try:
            termination = termination_events.get_nowait()
            observed_limits.add(termination)
            break
        except queue.Empty:
            pass
        if time.monotonic() >= deadline:
            termination = 'timed_out'
            break
        time.sleep(0.01)

    _kill_process_group(process)
    abort_error: BaseException | None = None
    if termination != 'exited':
        try:
            on_abort()
        except BaseException as error:
            abort_error = error
    try:
        exit_code = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        try:
            exit_code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            exit_code = None

    for stream in (process.stdin, process.stdout, process.stderr):
        try:
            stream.close()
        except OSError:
            pass
    for thread in (stdin_thread, stdout_thread, stderr_thread):
        thread.join(timeout=2)
    if not reader_errors.empty() and termination == 'exited':
        raise RuntimeError('failed while capturing isolation-runtime output') from reader_errors.get()
    if abort_error is not None:
        raise RuntimeError('isolation-runtime abort cleanup failed') from abort_error

    while True:
        try:
            late_event = termination_events.get_nowait()
        except queue.Empty:
            break
        observed_limits.add(late_event)
        if termination == 'exited':
            termination = late_event
    stdout_truncated = 'response_limit' in observed_limits
    stderr_truncated = 'log_limit' in observed_limits

    return BoundedProcessResult(
        exit_code=exit_code,
        duration_ms=max(0, round((time.monotonic() - started) * 1000)),
        stdout=bytes(stdout_buffer),
        stderr=bytes(stderr_buffer),
        termination=termination,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def _read_bounded(
    stream: BinaryIO,
    destination: bytearray,
    limit: int,
    limit_status: Literal['response_limit', 'log_limit'],
    termination_events: queue.SimpleQueue[Literal['response_limit', 'log_limit']],
    errors: queue.SimpleQueue[BaseException],
) -> None:
    try:
        while True:
            remaining = limit - len(destination)
            chunk = os.read(stream.fileno(), min(65_536, max(1, remaining + 1)))
            if not chunk:
                return
            if remaining > 0:
                destination.extend(chunk[:remaining])
            if len(chunk) > remaining:
                termination_events.put(limit_status)
                return
    except BaseException as error:
        errors.put(error)


def _write_input(stream: BinaryIO, input_bytes: bytes) -> None:
    try:
        stream.write(input_bytes)
        stream.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
