"""Runnable, fail-closed Linux guest appliance for the Lane A clinical harness.

The executable has one externally SHA-256-pinned canonical configuration at a fixed image path.
That configuration is the only source of the launcher trust anchor and vsock port.  The host CID is
the Linux ``VMADDR_CID_HOST`` constant; it is neither negotiated nor accepted from argv or the
environment.  One AF_VSOCK stream is used for both signed bootstrap and the complete guest-RPC
session.  The process never opens an IP socket, reads ambient credentials, invokes a shell, retries,
or logs task/model content.

This module is an image component, not evidence that an image was reproducibly built, measured, or
qualified on Linux/KVM.  The image build and qualification records must make those separate claims.
"""

from __future__ import annotations

import ctypes
import enum
import hashlib
import hmac
import os
import socket
import stat
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field

from vaxreplay.agentic.clinical_guest_bootstrap import (
    ClinicalGuestBootstrapReplayGuard,
    ClinicalGuestBootstrapTrustAnchor,
    InMemoryClinicalGuestBootstrapReplayGuard,
    run_lane_a_clinical_guest_entry,
)
from vaxreplay.agentic.clinical_guest_harness import LaneAGuestHarnessResult
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

LANE_A_CLINICAL_GUEST_CONFIG_SCHEMA_VERSION = 'vaxreplay.lane-a-clinical-guest-config.dev-v0.2'
LANE_A_CLINICAL_GUEST_ID = 'vaxreplay-lane-a-clinical-guest'
LANE_A_CLINICAL_GUEST_VERSION = 'dev-v0.2'
LANE_A_CLINICAL_GUEST_CONFIG_PATH = Path('/opt/vaxreplay/etc/lane-a-clinical-guest.json')
LANE_A_CLINICAL_GUEST_EXECUTABLE_PATH = Path('/opt/vaxreplay/bin/vaxreplay-lane-a-clinical-guest')
LANE_A_CLINICAL_GUEST_MAXIMUM_CONFIG_BYTES = 64 * 1024
LINUX_VMADDR_CID_HOST = 2
LINUX_REBOOT_CMD_POWER_OFF = 0x4321FEDC

_DIAGNOSTICS = {
    64: 'lane-a clinical guest rejected: baked configuration is invalid\n',
    70: 'lane-a clinical guest terminated: bounded execution failed\n',
    71: 'lane-a clinical guest terminated: poweroff failed\n',
}


class LaneAClinicalGuestError(RuntimeError):
    """Content-free guest appliance rejection."""


class LaneAClinicalGuestExitCode(enum.IntEnum):
    SUCCESS = 0
    CONFIGURATION_REJECTED = 64
    EXECUTION_FAILED = 70
    POWEROFF_FAILED = 71


class LaneAClinicalGuestConfig(StrictModel):
    """Static, image-baked inputs for exactly one Lane A guest session."""

    schema_version: Literal['vaxreplay.lane-a-clinical-guest-config.dev-v0.2'] = (
        LANE_A_CLINICAL_GUEST_CONFIG_SCHEMA_VERSION
    )
    guest_id: Literal['vaxreplay-lane-a-clinical-guest'] = LANE_A_CLINICAL_GUEST_ID
    guest_version: Literal['dev-v0.2'] = LANE_A_CLINICAL_GUEST_VERSION
    trust_anchor: ClinicalGuestBootstrapTrustAnchor
    host_cid: Literal[2] = LINUX_VMADDR_CID_HOST
    guest_rpc_port: int = Field(ge=1, le=2**32 - 1)
    connect_timeout_seconds: Literal[5] = 5
    bootstrap_timeout_seconds: Literal[5] = 5
    address_family: Literal['AF_VSOCK'] = 'AF_VSOCK'
    socket_type: Literal['SOCK_STREAM'] = 'SOCK_STREAM'
    one_connection: Literal[True] = True
    signed_bootstrap_and_rpc_share_socket: Literal[True] = True
    runtime_endpoint_negotiation_allowed: Literal[False] = False
    ip_network_allowed: Literal[False] = False
    shell_allowed: Literal[False] = False
    ambient_credentials_allowed: Literal[False] = False
    automatic_retry_allowed: Literal[False] = False
    task_content_logging_allowed: Literal[False] = False
    poweroff_after_terminal_result_required: Literal[True] = True
    development_only: Literal[True] = True
    measured_image_bake_attested: Literal[False] = False
    linux_kvm_qualified: Literal[False] = False


class LaneAClinicalGuestEntry(Protocol):
    def __call__(
        self,
        connection: socket.socket,
        *,
        trust_anchor: ClinicalGuestBootstrapTrustAnchor,
        replay_guard: ClinicalGuestBootstrapReplayGuard,
        clock: Callable[[], datetime],
        timeout_seconds: float,
    ) -> LaneAGuestHarnessResult: ...


class LaneAClinicalGuestConnector(Protocol):
    def __call__(self, config: LaneAClinicalGuestConfig) -> socket.socket: ...


class LaneAClinicalGuestConfigLoader(Protocol):
    def __call__(self, path: Path, *, expected_sha256: str) -> LaneAClinicalGuestConfig: ...


def lane_a_clinical_guest_config_sha256(config: LaneAClinicalGuestConfig) -> str:
    canonical = LaneAClinicalGuestConfig.model_validate_json(canonical_json_bytes(config))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def load_lane_a_clinical_guest_config(
    path: Path,
    *,
    expected_sha256: str,
) -> LaneAClinicalGuestConfig:
    """Load exact canonical JSON through a no-follow descriptor and verify its external digest."""

    if not _valid_sha256(expected_sha256):
        raise LaneAClinicalGuestError('configuration rejected')
    flags = os.O_RDONLY | os.O_CLOEXEC
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise LaneAClinicalGuestError('configuration rejected')
    flags |= no_follow
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= LANE_A_CLINICAL_GUEST_MAXIMUM_CONFIG_BYTES:
            raise LaneAClinicalGuestError('configuration rejected')
        body = _read_bounded(descriptor, LANE_A_CLINICAL_GUEST_MAXIMUM_CONFIG_BYTES)
        after = os.fstat(descriptor)
    except LaneAClinicalGuestError:
        raise
    except (OSError, ValueError):
        raise LaneAClinicalGuestError('configuration rejected') from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                raise LaneAClinicalGuestError('configuration rejected') from None
    if (
        len(body) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or not hmac.compare_digest(hashlib.sha256(body).hexdigest(), expected_sha256)
    ):
        raise LaneAClinicalGuestError('configuration rejected')
    try:
        config = LaneAClinicalGuestConfig.model_validate_json(body)
    except (TypeError, ValueError):
        raise LaneAClinicalGuestError('configuration rejected') from None
    if not hmac.compare_digest(body, canonical_json_bytes(config)):
        raise LaneAClinicalGuestError('configuration rejected')
    return config


def connect_lane_a_clinical_guest_vsock(
    config: LaneAClinicalGuestConfig,
    *,
    socket_factory: Callable[[int, int], socket.socket] = socket.socket,
    platform: str | None = None,
    af_vsock: int | None = None,
    vmaddr_cid_host: int | None = None,
) -> socket.socket:
    """Open the sole fixed host connection without DNS, IP, discovery, or retry."""

    canonical = LaneAClinicalGuestConfig.model_validate_json(canonical_json_bytes(config))
    observed_platform = sys.platform if platform is None else platform
    observed_af_vsock = getattr(socket, 'AF_VSOCK', None) if af_vsock is None else af_vsock
    observed_host_cid = getattr(socket, 'VMADDR_CID_HOST', None) if vmaddr_cid_host is None else vmaddr_cid_host
    if (
        observed_platform != 'linux'
        or not isinstance(observed_af_vsock, int)
        or not isinstance(observed_host_cid, int)
        or observed_host_cid != LINUX_VMADDR_CID_HOST
        or canonical.host_cid != observed_host_cid
    ):
        raise LaneAClinicalGuestError('connection rejected')
    connection: socket.socket | None = None
    try:
        connection = socket_factory(observed_af_vsock, socket.SOCK_STREAM)
        connection.set_inheritable(False)
        connection.settimeout(canonical.connect_timeout_seconds)
        connection.connect((canonical.host_cid, canonical.guest_rpc_port))
        connection.settimeout(None)
        return connection
    except (OSError, TypeError, ValueError):
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        raise LaneAClinicalGuestError('connection rejected') from None


def execute_baked_lane_a_clinical_guest(
    expected_config_sha256: str,
    *,
    config_loader: LaneAClinicalGuestConfigLoader = load_lane_a_clinical_guest_config,
    connector: LaneAClinicalGuestConnector = connect_lane_a_clinical_guest_vsock,
    guest_entry: LaneAClinicalGuestEntry = run_lane_a_clinical_guest_entry,
    replay_guard: ClinicalGuestBootstrapReplayGuard | None = None,
    poweroff: Callable[[], None] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> LaneAClinicalGuestExitCode:
    """Run one appliance attempt, close its only socket, then power off on every terminal path."""

    connection: socket.socket | None = None
    exit_code = LaneAClinicalGuestExitCode.CONFIGURATION_REJECTED
    try:
        config = config_loader(
            LANE_A_CLINICAL_GUEST_CONFIG_PATH,
            expected_sha256=expected_config_sha256,
        )
        guard = replay_guard or InMemoryClinicalGuestBootstrapReplayGuard()
        exit_code = LaneAClinicalGuestExitCode.EXECUTION_FAILED
        connection = connector(config)
        guest_entry(
            connection,
            trust_anchor=config.trust_anchor,
            replay_guard=guard,
            clock=clock,
            timeout_seconds=config.bootstrap_timeout_seconds,
        )
        exit_code = LaneAClinicalGuestExitCode.SUCCESS
    except BaseException:
        # An appliance has no interactive recovery path.  Collapse all task-, provider-, socket-,
        # and exception-bearing failures to the phase's fixed code, then power off below.
        pass
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException:
                exit_code = LaneAClinicalGuestExitCode.EXECUTION_FAILED
    shutdown = linux_guest_poweroff if poweroff is None else poweroff
    try:
        shutdown()
    except BaseException:
        return LaneAClinicalGuestExitCode.POWEROFF_FAILED
    # The real Linux reboot syscall does not return on success.  Returning is useful only for the
    # injected test seam; the CLI exits immediately with this bounded status.
    return exit_code


def linux_guest_poweroff() -> None:
    """Synchronize filesystems and invoke Linux ``reboot(POWER_OFF)`` without a shell.

    The guest process requires ``CAP_SYS_BOOT`` in the initial user namespace.  A successful call
    does not return.  Any unsupported platform, missing symbol, denied capability, or unexpected
    return is a terminal poweroff failure.
    """

    if sys.platform != 'linux':
        raise LaneAClinicalGuestError('poweroff rejected')
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        reboot = libc.reboot
        reboot.argtypes = [ctypes.c_int]
        reboot.restype = ctypes.c_int
        os.sync()
        result = reboot(LINUX_REBOOT_CMD_POWER_OFF)
    except (AttributeError, OSError, TypeError, ValueError):
        raise LaneAClinicalGuestError('poweroff rejected') from None
    if result != 0:
        raise LaneAClinicalGuestError('poweroff rejected')
    raise LaneAClinicalGuestError('poweroff returned unexpectedly')


def main(argv: Sequence[str] | None = None) -> None:
    """Console entrypoint with no caller-selectable path, CID, port, policy, or retry."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    expected_sha256 = arguments[1] if len(arguments) == 2 and arguments[0] == '--expected-config-sha256' else ''
    exit_code = execute_baked_lane_a_clinical_guest(expected_sha256)
    if exit_code != LaneAClinicalGuestExitCode.SUCCESS:
        sys.stderr.write(_DIAGNOSTICS[int(exit_code)])
    raise SystemExit(int(exit_code))


def _read_bounded(descriptor: int, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum_bytes + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(remaining, 16 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    body = b''.join(chunks)
    if len(body) > maximum_bytes:
        raise LaneAClinicalGuestError('configuration rejected')
    return body


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in '0123456789abcdef' for character in value)


__all__ = [
    'LANE_A_CLINICAL_GUEST_CONFIG_PATH',
    'LANE_A_CLINICAL_GUEST_CONFIG_SCHEMA_VERSION',
    'LANE_A_CLINICAL_GUEST_EXECUTABLE_PATH',
    'LANE_A_CLINICAL_GUEST_ID',
    'LANE_A_CLINICAL_GUEST_MAXIMUM_CONFIG_BYTES',
    'LANE_A_CLINICAL_GUEST_VERSION',
    'LINUX_REBOOT_CMD_POWER_OFF',
    'LINUX_VMADDR_CID_HOST',
    'LaneAClinicalGuestConfig',
    'LaneAClinicalGuestError',
    'LaneAClinicalGuestExitCode',
    'connect_lane_a_clinical_guest_vsock',
    'execute_baked_lane_a_clinical_guest',
    'lane_a_clinical_guest_config_sha256',
    'linux_guest_poweroff',
    'load_lane_a_clinical_guest_config',
    'main',
]


if __name__ == '__main__':
    main()
