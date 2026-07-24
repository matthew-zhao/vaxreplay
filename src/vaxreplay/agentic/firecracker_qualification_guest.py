"""Separate in-guest executable used only for Firecracker qualification drills.

The task harness is deliberately not imported here.  One qualification VM accepts exactly one
challenge-bound request over a guest-initiated vsock stream, measures the requested guest-local
property, returns an Ed25519-signed response, and exits.  The intentional-hang command returns a
signed acknowledgement and then remains alive until the host watchdog kills the VM.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import signal
import socket
import stat
import struct
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self

# ``-I`` prevents ambient paths from changing imports.  The qualification disk builder places the
# exact committed package tree here, and the whole harness ext4 digest binds these dependencies.
_PINNED_GUEST_LIBRARY = '/opt/vaxreplay/lib'
if _PINNED_GUEST_LIBRARY not in sys.path:
    sys.path.insert(0, _PINNED_GUEST_LIBRARY)

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from pydantic import Field, model_validator  # noqa: E402

from vaxreplay.agentic.firecracker_qualification import (  # noqa: E402
    FirecrackerQualificationClaim,
    FirecrackerQualificationDrillId,
)
from vaxreplay.agentic.firecracker_qualification_probe import (  # noqa: E402
    AuthenticatedFirecrackerQualificationGuestResponse,
    FirecrackerQualificationChallenge,
    FirecrackerQualificationGuestCommand,
    FirecrackerQualificationGuestResponse,
    FirecrackerQualificationWorkerBinding,
    firecracker_qualification_challenge_sha256,
    firecracker_qualification_guest_key_id,
    firecracker_qualification_worker_binding_sha256,
    sign_firecracker_qualification_guest_response,
)
from vaxreplay.bundle import canonical_json_bytes  # noqa: E402
from vaxreplay.case_schema import StrictModel  # noqa: E402

FIRECRACKER_QUALIFICATION_GUEST_CONFIG_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-guest-config.v0.1'
FIRECRACKER_QUALIFICATION_GUEST_REQUEST_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-guest-request.v0.1'
FIRECRACKER_QUALIFICATION_GUEST_RESULT_SCHEMA_VERSION = 'vaxreplay.firecracker-qualification-guest-result.v0.1'

QUALIFICATION_GUEST_EXECUTABLE_PATH = '/opt/vaxreplay/bin/vaxreplay-firecracker-qualification-probe'
QUALIFICATION_GUEST_CONFIG_PATH = '/opt/vaxreplay/etc/qualification-guest.json'
QUALIFICATION_HARNESS_MOUNT = '/opt/vaxreplay'
QUALIFICATION_SCRATCH_MOUNT = '/workspace'

_MAX_FRAME_BYTES = 1024 * 1024
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_HEX_64_PATTERN = r'^[0-9a-f]{64}$'
_FRAME_LENGTH = struct.Struct('!I')

_COMMAND_BY_DRILL = {
    FirecrackerQualificationDrillId.LIVE_BOOT: FirecrackerQualificationGuestCommand.BOOT_READY_AND_EXIT,
    FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP: FirecrackerQualificationGuestCommand.VSOCK_NONCE_ECHO,
    FirecrackerQualificationDrillId.GUEST_ISOLATION: FirecrackerQualificationGuestCommand.ISOLATION_PROBES,
    FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT: FirecrackerQualificationGuestCommand.CGROUP_STRESS,
    FirecrackerQualificationDrillId.WALL_TIMEOUT: FirecrackerQualificationGuestCommand.INTENTIONAL_HANG,
    FirecrackerQualificationDrillId.LOAD_CANARY: FirecrackerQualificationGuestCommand.LOAD_CANARY,
}


class FirecrackerQualificationGuestError(RuntimeError):
    """The qualification guest could not produce an authenticated measurement."""


class FirecrackerQualificationGuestConfig(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-guest-config.v0.1'] = (
        FIRECRACKER_QUALIFICATION_GUEST_CONFIG_SCHEMA_VERSION
    )
    host_cid: Literal[2] = 2
    rpc_port: int = Field(ge=1, le=2**32 - 1)
    guest_probe_private_key_hex: str = Field(pattern=_HEX_64_PATTERN)
    guest_probe_key_id: str = Field(pattern=_SHA256_PATTERN)
    guest_probe_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_mount: Literal['/opt/vaxreplay'] = QUALIFICATION_HARNESS_MOUNT
    scratch_mount: Literal['/workspace'] = QUALIFICATION_SCRATCH_MOUNT
    max_frame_bytes: Literal[1048576] = _MAX_FRAME_BYTES

    @model_validator(mode='after')
    def validate_key(self) -> Self:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(self.guest_probe_private_key_hex))
        public_key = private_key.public_key().public_bytes_raw()
        if firecracker_qualification_guest_key_id(public_key) != self.guest_probe_key_id:
            raise ValueError('qualification guest private key differs from the pinned key ID')
        return self


class FirecrackerQualificationGuestRequest(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-guest-request.v0.1'] = (
        FIRECRACKER_QUALIFICATION_GUEST_REQUEST_SCHEMA_VERSION
    )
    challenge: FirecrackerQualificationChallenge
    worker_binding: FirecrackerQualificationWorkerBinding
    command: FirecrackerQualificationGuestCommand
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_probe_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    caller_assertions_are_evidence: Literal[False] = False

    @model_validator(mode='after')
    def validate_bindings(self) -> Self:
        expected_command = _COMMAND_BY_DRILL.get(self.challenge.drill_id)
        if expected_command is None or self.command != expected_command:
            raise ValueError('qualification guest command differs from the challenged drill')
        binding = self.worker_binding
        if (
            binding.run_id not in self.challenge.run_ids
            or binding.worker_spec_sha256 != self.challenge.worker_spec_sha256
            or binding.probe_manifest_sha256 != self.challenge.probe_manifest_sha256
            or self.worker_spec_sha256 != self.challenge.worker_spec_sha256
            or self.probe_manifest_sha256 != self.challenge.probe_manifest_sha256
        ):
            raise ValueError('qualification guest request does not bind one challenge and worker')
        return self


class FirecrackerQualificationGuestResult(StrictModel):
    schema_version: Literal['vaxreplay.firecracker-qualification-guest-result.v0.1'] = (
        FIRECRACKER_QUALIFICATION_GUEST_RESULT_SCHEMA_VERSION
    )
    command: FirecrackerQualificationGuestCommand
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    nonce_sha256: str = Field(pattern=_SHA256_PATTERN)
    rootfs_write_denied: bool = False
    harness_write_denied: bool = False
    scratch_write_succeeded: bool = False
    scratch_fresh: bool = False
    network_unreachable: bool = False
    mmds_unreachable: bool = False
    stress_ready: bool = False
    hang_ready: bool = False


class GuestLocalProbe(Protocol):
    def isolation(self, *, harness_mount: Path, scratch_mount: Path) -> FirecrackerQualificationGuestResult: ...


class LinuxGuestLocalProbe:
    """Guest-local measurements; no host-provided success booleans are accepted."""

    def __init__(self, *, request: FirecrackerQualificationGuestRequest) -> None:
        self._request = request

    def isolation(self, *, harness_mount: Path, scratch_mount: Path) -> FirecrackerQualificationGuestResult:
        scratch_fresh = _scratch_is_fresh(scratch_mount)
        rootfs_write_denied = _write_is_read_only(Path('/.vaxreplay-qualification-write-probe'))
        harness_write_denied = _write_is_read_only(harness_mount / '.vaxreplay-qualification-write-probe')
        scratch_write_succeeded = _scratch_write_round_trip(scratch_mount, self._request.challenge.nonce_hex)
        return self._base(
            rootfs_write_denied=rootfs_write_denied,
            harness_write_denied=harness_write_denied,
            scratch_write_succeeded=scratch_write_succeeded,
            scratch_fresh=scratch_fresh,
            network_unreachable=_network_unreachable('1.1.1.1', 443),
            mmds_unreachable=_network_unreachable('169.254.169.254', 80),
        )

    def _base(self, **updates: bool) -> FirecrackerQualificationGuestResult:
        base = FirecrackerQualificationGuestResult(
            command=self._request.command,
            run_id=self._request.worker_binding.run_id,
            nonce_sha256=hashlib.sha256(bytes.fromhex(self._request.challenge.nonce_hex)).hexdigest(),
        )
        return base.model_copy(update=updates)


def _linux_local_probe_factory(request: FirecrackerQualificationGuestRequest) -> LinuxGuestLocalProbe:
    return LinuxGuestLocalProbe(request=request)


def execute_firecracker_qualification_guest_request(
    request: FirecrackerQualificationGuestRequest,
    *,
    config: FirecrackerQualificationGuestConfig,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    local_probe_factory: Callable[
        [FirecrackerQualificationGuestRequest], LinuxGuestLocalProbe
    ] = _linux_local_probe_factory,
    pressure_starter: Callable[[], None] | None = None,
) -> tuple[AuthenticatedFirecrackerQualificationGuestResponse, bool]:
    """Execute one fixed command and return its signed response plus whether to hang afterward."""

    if request.guest_probe_executable_sha256 != config.guest_probe_executable_sha256:
        raise FirecrackerQualificationGuestError('guest request differs from the executable pinned in guest config')
    result = FirecrackerQualificationGuestResult(
        command=request.command,
        run_id=request.worker_binding.run_id,
        nonce_sha256=hashlib.sha256(bytes.fromhex(request.challenge.nonce_hex)).hexdigest(),
    )
    if request.command == FirecrackerQualificationGuestCommand.ISOLATION_PROBES:
        result = local_probe_factory(request).isolation(
            harness_mount=Path(config.harness_mount),
            scratch_mount=Path(config.scratch_mount),
        )
    elif request.command == FirecrackerQualificationGuestCommand.CGROUP_STRESS:
        (pressure_starter or _start_bounded_guest_pressure)()
        result = result.model_copy(update={'stress_ready': True})
    elif request.command == FirecrackerQualificationGuestCommand.INTENTIONAL_HANG:
        result = result.model_copy(update={'hang_ready': True})

    claims = _claims_from_result(request.command, result)
    response = FirecrackerQualificationGuestResponse(
        challenge_sha256=firecracker_qualification_challenge_sha256(request.challenge),
        nonce_hex=request.challenge.nonce_hex,
        run_id=request.worker_binding.run_id,
        worker_binding_sha256=firecracker_qualification_worker_binding_sha256(request.worker_binding),
        worker_spec_sha256=request.worker_spec_sha256,
        probe_manifest_sha256=request.probe_manifest_sha256,
        guest_probe_executable_sha256=request.guest_probe_executable_sha256,
        command=request.command,
        verified_guest_claims=claims,
        result_bytes_sha256=hashlib.sha256(canonical_json_bytes(result)).hexdigest(),
        responded_at=clock(),
    )
    authenticated = sign_firecracker_qualification_guest_response(
        response,
        private_key=Ed25519PrivateKey.from_private_bytes(bytes.fromhex(config.guest_probe_private_key_hex)),
    )
    return authenticated, request.command in {
        FirecrackerQualificationGuestCommand.CGROUP_STRESS,
        FirecrackerQualificationGuestCommand.INTENTIONAL_HANG,
    }


def run_firecracker_qualification_guest(
    *,
    config_path: Path = Path(QUALIFICATION_GUEST_CONFIG_PATH),
    socket_factory: Callable[[int, int], socket.socket] = socket.socket,
) -> None:
    config = _load_config(config_path)
    executable_bytes = Path(__file__).read_bytes()
    if hashlib.sha256(executable_bytes).hexdigest() != config.guest_probe_executable_sha256:
        raise FirecrackerQualificationGuestError('running guest probe source differs from its image config pin')
    af_vsock = getattr(socket, 'AF_VSOCK', None)
    if not isinstance(af_vsock, int) or not isinstance(getattr(socket, 'VMADDR_CID_HOST', 2), int):
        raise FirecrackerQualificationGuestError('AF_VSOCK is unavailable in the qualification guest')
    connection = socket_factory(af_vsock, socket.SOCK_STREAM)
    try:
        connection.settimeout(30.0)
        connection.connect((config.host_cid, config.rpc_port))
        request_bytes = _recv_frame(connection, max_bytes=config.max_frame_bytes)
        try:
            request = FirecrackerQualificationGuestRequest.model_validate_json(request_bytes)
        except ValueError as error:
            raise FirecrackerQualificationGuestError('qualification guest request is invalid') from error
        if canonical_json_bytes(request) != request_bytes:
            raise FirecrackerQualificationGuestError('qualification guest request is not canonical JSON')
        response, hang_after_response = execute_firecracker_qualification_guest_request(request, config=config)
        _send_frame(connection, canonical_json_bytes(response), max_bytes=config.max_frame_bytes)
    finally:
        connection.close()
    if hang_after_response:
        while True:
            signal.pause()


def main() -> None:
    parser = argparse.ArgumentParser(description='run exactly one signed Firecracker qualification guest probe')
    parser.add_argument('--config', type=Path, default=Path(QUALIFICATION_GUEST_CONFIG_PATH))
    arguments = parser.parse_args()
    try:
        run_firecracker_qualification_guest(config_path=arguments.config)
    except FirecrackerQualificationGuestError as error:
        sys.stderr.write(f'qualification guest rejected: {error}\n')
        raise SystemExit(64) from error


def _load_config(path: Path) -> FirecrackerQualificationGuestConfig:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise FirecrackerQualificationGuestError('qualification guest config is unavailable') from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or stat.S_IMODE(before.st_mode) & 0o022:
            raise FirecrackerQualificationGuestError(
                'qualification guest config must be a root-owned non-writable regular file'
            )
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(64 * 1024 + 1 - len(content), 16 * 1024))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > 64 * 1024:
                raise FirecrackerQualificationGuestError('qualification guest config is oversized')
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise FirecrackerQualificationGuestError('qualification guest config changed while being read')
    finally:
        os.close(descriptor)
    try:
        config = FirecrackerQualificationGuestConfig.model_validate_json(bytes(content))
    except ValueError as error:
        raise FirecrackerQualificationGuestError('qualification guest config is invalid') from error
    if canonical_json_bytes(config) != bytes(content):
        raise FirecrackerQualificationGuestError('qualification guest config is not canonical JSON')
    return config


def _claims_from_result(
    command: FirecrackerQualificationGuestCommand,
    result: FirecrackerQualificationGuestResult,
) -> tuple[FirecrackerQualificationClaim, ...]:
    claims: set[FirecrackerQualificationClaim] = set()
    if command == FirecrackerQualificationGuestCommand.BOOT_READY_AND_EXIT:
        claims.add(FirecrackerQualificationClaim.GUEST_READY_AUTHENTICATED)
    elif command == FirecrackerQualificationGuestCommand.VSOCK_NONCE_ECHO:
        claims.add(FirecrackerQualificationClaim.GUEST_RPC_ROUND_TRIP)
    elif command == FirecrackerQualificationGuestCommand.ISOLATION_PROBES:
        mapping = {
            FirecrackerQualificationClaim.ROOTFS_WRITE_DENIED: result.rootfs_write_denied,
            FirecrackerQualificationClaim.HARNESS_WRITE_DENIED: result.harness_write_denied,
            FirecrackerQualificationClaim.SCRATCH_WRITE_SUCCEEDED: result.scratch_write_succeeded,
            FirecrackerQualificationClaim.SCRATCH_FRESH: result.scratch_fresh,
            FirecrackerQualificationClaim.NETWORK_UNREACHABLE: result.network_unreachable,
            FirecrackerQualificationClaim.MMDS_UNREACHABLE: result.mmds_unreachable,
        }
        claims.update(claim for claim, observed in mapping.items() if observed)
    return tuple(sorted(claims, key=lambda claim: claim.value))


def _write_is_read_only(path: Path) -> bool:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    except OSError as error:
        return error.errno == errno.EROFS
    else:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        return False


def _scratch_is_fresh(path: Path) -> bool:
    try:
        return {entry.name for entry in os.scandir(path)} <= {'lost+found'}
    except OSError:
        return False


def _scratch_write_round_trip(path: Path, nonce_hex: str) -> bool:
    target = path / f'.qualification-{nonce_hex}'
    payload = bytes.fromhex(nonce_hex)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        with os.fdopen(descriptor, 'wb', closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        observed = target.read_bytes() == payload
        target.unlink()
        return observed
    except OSError:
        return False


def _network_unreachable(address: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(0.2)
            return connection.connect_ex((address, port)) in {errno.ENETUNREACH, errno.EHOSTUNREACH}
    except OSError as error:
        return error.errno in {errno.ENETUNREACH, errno.EHOSTUNREACH}


def _start_bounded_guest_pressure() -> None:
    """Start pressure inside the VM; the host watchdog bounds its lifetime.

    CPU workers run hot, a high-oom-score worker allocates until the memory cgroup rejects it, and
    a fork worker creates sleeping descendants until ``pids.max`` rejects another fork.  The probe
    process returns its signed ready response only after these workers have been created, then PID 1
    waits for host teardown.  No pressure process survives the VM process group.
    """

    for _ in range(max(2, min(8, (os.cpu_count() or 1) + 1))):
        try:
            child = os.fork()
        except OSError:
            break
        if child == 0:
            value = 1
            while True:
                value = (value * 1103515245 + 12345) & 0x7FFFFFFF
    try:
        memory_child = os.fork()
    except OSError:
        memory_child = -1
    if memory_child == 0:
        try:
            Path('/proc/self/oom_score_adj').write_text('1000', encoding='ascii')
        except OSError:
            pass
        allocations: list[bytearray] = []
        try:
            while True:
                block = bytearray(4 * 1024 * 1024)
                block[::4096] = b'\x01' * (len(block) // 4096)
                allocations.append(block)
        except MemoryError:
            while True:
                signal.pause()
    try:
        pid_child = os.fork()
    except OSError:
        pid_child = -1
    if pid_child == 0:
        while True:
            try:
                descendant = os.fork()
            except OSError:
                break
            if descendant == 0:
                while True:
                    signal.pause()
        while True:
            signal.pause()
    # Give descendants a bounded head start so the first host snapshot can observe active pressure.
    time.sleep(0.05)


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = connection.recv(count - len(chunks))
        if not chunk:
            raise FirecrackerQualificationGuestError('qualification guest connection closed mid-frame')
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_frame(connection: socket.socket, *, max_bytes: int) -> bytes:
    (length,) = _FRAME_LENGTH.unpack(_recv_exact(connection, _FRAME_LENGTH.size))
    if length == 0 or length > max_bytes:
        raise FirecrackerQualificationGuestError('qualification guest frame length is invalid')
    return _recv_exact(connection, length)


def _send_frame(connection: socket.socket, payload: bytes, *, max_bytes: int) -> None:
    if not payload or len(payload) > max_bytes:
        raise FirecrackerQualificationGuestError('qualification guest response length is invalid')
    connection.sendall(_FRAME_LENGTH.pack(len(payload)) + payload)


__all__ = [
    'FIRECRACKER_QUALIFICATION_GUEST_CONFIG_SCHEMA_VERSION',
    'FIRECRACKER_QUALIFICATION_GUEST_REQUEST_SCHEMA_VERSION',
    'FirecrackerQualificationGuestConfig',
    'FirecrackerQualificationGuestError',
    'FirecrackerQualificationGuestRequest',
    'FirecrackerQualificationGuestResult',
    'QUALIFICATION_GUEST_CONFIG_PATH',
    'QUALIFICATION_GUEST_EXECUTABLE_PATH',
    'execute_firecracker_qualification_guest_request',
    'main',
    'run_firecracker_qualification_guest',
]


if __name__ == '__main__':
    main()
