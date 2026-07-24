"""Bounded operator adapters for isolated signers and clock-health collectors.

The adapters intentionally speak one canonical JSON request and one canonical
JSON response over stdio.  They never invoke a shell, inherit an environment,
or include child output, command arguments, or provider exceptions in errors.
Private-key custody and vendor-specific APIs remain behind the subprocess.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Literal, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import Field, field_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.clock_health import (
    ClockHealthGate,
    ClockHealthObservation,
    ClockHealthPolicy,
    ClockHealthProvider,
)
from vaxreplay.operations.signing import Ed25519Signer, IsolatedEd25519Signer, LocalEd25519Signer

ISOLATED_PROCESS_CONFIG_SCHEMA_VERSION = 'vaxreplay.isolated-process-config.v0.1'
EXTERNAL_SIGNER_REQUEST_SCHEMA_VERSION = 'vaxreplay.external-signer-request.v0.1'
EXTERNAL_SIGNER_RESPONSE_SCHEMA_VERSION = 'vaxreplay.external-signer-response.v0.1'
CLOCK_HEALTH_REQUEST_SCHEMA_VERSION = 'vaxreplay.clock-health-request.v0.1'

_MAX_CONFIG_BYTES = 64 * 1024
_MAX_COMMAND_ARGUMENTS = 64
_MAX_ARGUMENT_BYTES = 4096
_MAX_COMMAND_BYTES = 32 * 1024
_MAX_SIGNING_MESSAGE_BYTES = 65 * 1024 * 1024
_MAX_PROCESS_REQUEST_BYTES = 4 * ((_MAX_SIGNING_MESSAGE_BYTES + 2) // 3) + 4096
_MAX_CLOCK_POLICY_BYTES = 1024 * 1024
_MAX_PUBLIC_KEY_BYTES = 32
_MAX_PRIVATE_KEY_BYTES = 32


class OperatorTrustError(ValueError):
    """An operator trust adapter or its exact configuration failed closed."""


class IsolatedProcessConfig(StrictModel):
    """Pinned argv and resource limits for one no-shell one-shot process."""

    schema_version: Literal['vaxreplay.isolated-process-config.v0.1'] = ISOLATED_PROCESS_CONFIG_SCHEMA_VERSION
    process_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
    argv: tuple[str, ...] = Field(min_length=1, max_length=_MAX_COMMAND_ARGUMENTS)
    executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    executable_byte_count: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    max_stdout_bytes: int = Field(default=4096, ge=128, le=1024 * 1024)
    max_stderr_bytes: int = Field(default=64 * 1024, ge=0, le=1024 * 1024)

    @field_validator('argv')
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        total = 0
        for argument in value:
            if not argument or '\x00' in argument or '\r' in argument or '\n' in argument:
                raise ValueError('isolated-process argv contains an invalid argument')
            encoded = argument.encode('utf-8')
            if len(encoded) > _MAX_ARGUMENT_BYTES:
                raise ValueError('isolated-process argv argument is too large')
            total += len(encoded)
        if total > _MAX_COMMAND_BYTES:
            raise ValueError('isolated-process argv is too large')
        if not os.path.isabs(value[0]):
            raise ValueError('isolated-process executable must be an absolute path')
        return value


class ExternalSignerRequest(StrictModel):
    schema_version: Literal['vaxreplay.external-signer-request.v0.1'] = EXTERNAL_SIGNER_REQUEST_SCHEMA_VERSION
    operation: Literal['ed25519_sign'] = 'ed25519_sign'
    expected_public_key_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    message_base64: str

    @field_validator('message_base64')
    @classmethod
    def validate_message(cls, value: str) -> str:
        decoded = _canonical_base64(value, 'signing message')
        if not decoded or len(decoded) > _MAX_SIGNING_MESSAGE_BYTES:
            raise ValueError('signing message size is outside the supported range')
        return value


class ExternalSignerResponse(StrictModel):
    schema_version: Literal['vaxreplay.external-signer-response.v0.1'] = EXTERNAL_SIGNER_RESPONSE_SCHEMA_VERSION
    operation: Literal['ed25519_sign'] = 'ed25519_sign'
    signature_base64: str

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        signature = _canonical_base64(value, 'external signature')
        if len(signature) != 64:
            raise ValueError('external signature must contain exactly 64 bytes')
        return value


class ClockHealthRequest(StrictModel):
    schema_version: Literal['vaxreplay.clock-health-request.v0.1'] = CLOCK_HEALTH_REQUEST_SCHEMA_VERSION
    operation: Literal['observe_clock_health'] = 'observe_clock_health'
    expected_provider_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


class SubprocessEd25519Signer:
    """Ed25519 signer whose only private-key operation is a bounded child IPC."""

    def __init__(
        self,
        *,
        public_key: bytes,
        process: IsolatedProcessConfig,
    ) -> None:
        self._process = IsolatedProcessConfig.model_validate(process)
        self._public_key = bytes(public_key)
        self._delegate = IsolatedEd25519Signer(
            public_key=self._public_key,
            sign_operation=self._invoke,
        )

    def public_key_bytes(self) -> bytes:
        return self._delegate.public_key_bytes()

    def sign(self, message: bytes) -> bytes:
        return self._delegate.sign(message)

    def _invoke(self, message: bytes) -> bytes:
        if not isinstance(message, bytes) or not message or len(message) > _MAX_SIGNING_MESSAGE_BYTES:
            raise OperatorTrustError('external signer operation failed')
        request = ExternalSignerRequest(
            expected_public_key_sha256=hashlib.sha256(self._public_key).hexdigest(),
            message_base64=base64.b64encode(message).decode('ascii'),
        )
        payload = _run_bounded_process(self._process, canonical_json_bytes(request))
        try:
            response = ExternalSignerResponse.model_validate_json(payload)
            if payload != canonical_json_bytes(response):
                raise ValueError('noncanonical response')
            return base64.b64decode(response.signature_base64, validate=True)
        except (TypeError, ValueError):
            raise OperatorTrustError('external signer operation failed') from None


class SubprocessClockHealthProvider:
    """Clock-health provider backed by a bounded one-shot process."""

    def __init__(self, *, provider_id: str, process: IsolatedProcessConfig) -> None:
        self._request = ClockHealthRequest(expected_provider_id=provider_id)
        self._process = IsolatedProcessConfig.model_validate(process)

    def observe(self) -> ClockHealthObservation:
        payload = _run_bounded_process(self._process, canonical_json_bytes(self._request))
        try:
            observation = ClockHealthObservation.model_validate_json(payload)
            if payload != canonical_json_bytes(observation):
                raise ValueError('noncanonical observation')
            if observation.provider_id != self._request.expected_provider_id:
                raise ValueError('provider identity mismatch')
            return observation
        except (TypeError, ValueError):
            raise OperatorTrustError('clock-health provider failed') from None


def load_isolated_process_config(
    path: Path,
    *,
    expected_sha256: str,
) -> IsolatedProcessConfig:
    payload = _read_regular_nofollow(path, _MAX_CONFIG_BYTES, private=False)
    if not _is_sha256(expected_sha256) or not _constant_digest(payload, expected_sha256):
        raise OperatorTrustError('isolated-process configuration differs from its trusted digest')
    try:
        config = IsolatedProcessConfig.model_validate_json(payload)
    except ValueError:
        raise OperatorTrustError('isolated-process configuration is invalid') from None
    if payload != canonical_json_bytes(config):
        raise OperatorTrustError('isolated-process configuration must use canonical JSON')
    _validate_executable(
        config.argv[0],
        expected_sha256=config.executable_sha256,
        expected_byte_count=config.executable_byte_count,
    )
    return config


def load_external_signer(
    *,
    process_config: Path,
    process_config_sha256: str,
    public_key: Path,
) -> Ed25519Signer:
    key = _read_regular_nofollow(public_key, _MAX_PUBLIC_KEY_BYTES, private=False)
    if len(key) != _MAX_PUBLIC_KEY_BYTES:
        raise OperatorTrustError('external signer public key must contain exactly 32 bytes')
    try:
        return SubprocessEd25519Signer(
            public_key=key,
            process=load_isolated_process_config(
                process_config,
                expected_sha256=process_config_sha256,
            ),
        )
    except ValueError:
        raise OperatorTrustError('external signer configuration is invalid') from None


def load_dev_local_signer(path: Path) -> Ed25519Signer:
    """Load an explicit owner-only development key; never the production default."""

    key = _read_regular_nofollow(path, _MAX_PRIVATE_KEY_BYTES, private=True)
    if len(key) != _MAX_PRIVATE_KEY_BYTES:
        raise OperatorTrustError('development private key must contain exactly 32 bytes')
    try:
        return LocalEd25519Signer(Ed25519PrivateKey.from_private_bytes(key))
    except ValueError:
        raise OperatorTrustError('development private key is invalid') from None


def load_clock_health_gate(
    *,
    policy_path: Path,
    policy_sha256: str,
    process_config: Path,
    process_config_sha256: str,
) -> ClockHealthGate:
    policy_bytes = _read_regular_nofollow(policy_path, _MAX_CLOCK_POLICY_BYTES, private=False)
    if not _is_sha256(policy_sha256) or not _constant_digest(policy_bytes, policy_sha256):
        raise OperatorTrustError('clock-health policy differs from its trusted digest')
    try:
        policy = ClockHealthPolicy.model_validate_json(policy_bytes)
    except ValueError:
        raise OperatorTrustError('clock-health policy is invalid') from None
    if policy_bytes != canonical_json_bytes(policy):
        raise OperatorTrustError('clock-health policy must use canonical JSON')
    provider: ClockHealthProvider = SubprocessClockHealthProvider(
        provider_id=policy.provider_id,
        process=load_isolated_process_config(
            process_config,
            expected_sha256=process_config_sha256,
        ),
    )
    return ClockHealthGate(policy=policy, provider=provider)


def resolve_operator_trust(
    *,
    dev_private_key: Path | None,
    external_signer_process: Path | None,
    external_signer_process_sha256: str | None,
    external_signer_public_key: Path | None,
    clock_health_policy: Path | None,
    clock_health_policy_sha256: str | None,
    clock_health_process: Path | None,
    clock_health_process_sha256: str | None,
    require_clock_for_external_signer: bool = True,
) -> tuple[Ed25519Signer, ClockHealthGate | None]:
    """Resolve exactly one signer mode and an optional paired clock-health gate."""

    local_selected = dev_private_key is not None
    external_selected = any(
        value is not None
        for value in (
            external_signer_process,
            external_signer_process_sha256,
            external_signer_public_key,
        )
    )
    if local_selected == external_selected:
        raise OperatorTrustError('select exactly one development or external signer mode')
    clock_selected = any(
        value is not None
        for value in (
            clock_health_policy,
            clock_health_policy_sha256,
            clock_health_process,
            clock_health_process_sha256,
        )
    )
    if clock_selected and (
        clock_health_policy is None
        or clock_health_policy_sha256 is None
        or clock_health_process is None
        or clock_health_process_sha256 is None
    ):
        raise OperatorTrustError('clock-health policy and process configuration require both trusted digests')
    if external_selected:
        if (
            external_signer_process is None
            or external_signer_process_sha256 is None
            or external_signer_public_key is None
        ):
            raise OperatorTrustError(
                'external signer process, trusted digest, and public key must be supplied together'
            )
        if require_clock_for_external_signer and not clock_selected:
            raise OperatorTrustError('external signer mode requires a clock-health gate')
        signer = load_external_signer(
            process_config=external_signer_process,
            process_config_sha256=external_signer_process_sha256,
            public_key=external_signer_public_key,
        )
    else:
        assert dev_private_key is not None
        signer = load_dev_local_signer(dev_private_key)
    gate = None
    if clock_selected:
        assert (
            clock_health_policy is not None
            and clock_health_policy_sha256 is not None
            and clock_health_process is not None
            and clock_health_process_sha256 is not None
        )
        gate = load_clock_health_gate(
            policy_path=clock_health_policy,
            policy_sha256=clock_health_policy_sha256,
            process_config=clock_health_process,
            process_config_sha256=clock_health_process_sha256,
        )
    return signer, gate


def add_signer_arguments(parser: argparse.ArgumentParser, *, dev_required: bool = False) -> None:
    """Add consistent signer and clock options to an argparse parser."""

    group = parser.add_mutually_exclusive_group(required=dev_required)
    group.add_argument(
        '--dev-signing-private-key',
        help='explicit development-only owner-protected raw 32-byte Ed25519 key',
    )
    group.add_argument(
        '--external-signer-process',
        help='canonical isolated-process configuration for the production signer broker',
    )
    parser.add_argument('--external-signer-public-key')
    parser.add_argument('--external-signer-process-sha256')
    parser.add_argument('--clock-health-policy')
    parser.add_argument('--clock-health-policy-sha256')
    parser.add_argument('--clock-health-process')
    parser.add_argument('--clock-health-process-sha256')


def signer_and_clock_from_args(
    args: object,
    *,
    require_clock_for_external_signer: bool = True,
) -> tuple[Ed25519Signer, ClockHealthGate | None]:
    return resolve_operator_trust(
        dev_private_key=_optional_path(getattr(args, 'dev_signing_private_key', None)),
        external_signer_process=_optional_path(getattr(args, 'external_signer_process', None)),
        external_signer_process_sha256=getattr(args, 'external_signer_process_sha256', None),
        external_signer_public_key=_optional_path(getattr(args, 'external_signer_public_key', None)),
        clock_health_policy=_optional_path(getattr(args, 'clock_health_policy', None)),
        clock_health_policy_sha256=getattr(args, 'clock_health_policy_sha256', None),
        clock_health_process=_optional_path(getattr(args, 'clock_health_process', None)),
        clock_health_process_sha256=getattr(args, 'clock_health_process_sha256', None),
        require_clock_for_external_signer=require_clock_for_external_signer,
    )


def _optional_path(value: object) -> Path | None:
    return None if value is None else Path(str(value))


def _run_bounded_process(config: IsolatedProcessConfig, request: bytes) -> bytes:
    if not isinstance(request, bytes) or not request or len(request) > _MAX_PROCESS_REQUEST_BYTES:
        raise OperatorTrustError('isolated process failed')
    with _verified_executable_copy(
        config.argv[0],
        expected_sha256=config.executable_sha256,
        expected_byte_count=config.executable_byte_count,
    ) as executable:
        return _run_verified_process(config, request, executable=executable)


def _run_verified_process(
    config: IsolatedProcessConfig,
    request: bytes,
    *,
    executable: str,
) -> bytes:
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    try:
        process = subprocess.Popen(
            config.argv,
            executable=executable,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd='/',
            env={},
            close_fds=True,
            shell=False,
            start_new_session=True,
        )
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, ('stdin', 0))
        selector.register(process.stdout, selectors.EVENT_READ, ('stdout', 0))
        selector.register(process.stderr, selectors.EVENT_READ, ('stderr', 0))
        deadline = time.monotonic() + config.timeout_seconds
        input_offset = 0
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OperatorTrustError('isolated process failed')
            events = selector.select(min(remaining, 0.1))
            if not events and process.poll() is not None:
                # Continue once to drain any pipe bytes made visible at exit.
                events = selector.select(0)
                if not events:
                    for key in tuple(selector.get_map().values()):
                        if key.data[0] == 'stdin':
                            selector.unregister(key.fileobj)
                            cast(IO[bytes], key.fileobj).close()
                    if len(selector.get_map()) == 0:
                        break
            for key, _ in events:
                stream_name = key.data[0]
                stream = cast(IO[bytes], key.fileobj)
                if stream_name == 'stdin':
                    try:
                        count = os.write(stream.fileno(), request[input_offset : input_offset + 65536])
                    except BrokenPipeError:
                        count = 0
                    input_offset += count
                    if input_offset >= len(request) or count == 0:
                        selector.unregister(stream)
                        stream.close()
                    continue
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                destination = stdout if stream_name == 'stdout' else stderr
                maximum = config.max_stdout_bytes if stream_name == 'stdout' else config.max_stderr_bytes
                destination.extend(chunk)
                if len(destination) > maximum:
                    raise OperatorTrustError('isolated process failed')
        remaining = max(0.0, deadline - time.monotonic())
        return_code = process.wait(timeout=remaining)
        if return_code != 0:
            raise OperatorTrustError('isolated process failed')
        return bytes(stdout)
    except (OSError, subprocess.SubprocessError, OperatorTrustError):
        raise OperatorTrustError('isolated process failed') from None
    finally:
        selector.close()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.SubprocessError):
                pass
        for index in range(len(stderr)):
            stderr[index] = 0


def _read_regular_nofollow(path: Path, maximum: int, *, private: bool) -> bytes:
    requested = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError:
        raise OperatorTrustError('operator trust input could not be opened') from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OperatorTrustError('operator trust input is not a regular file')
        if private and metadata.st_mode & 0o077:
            raise OperatorTrustError('development private key must be owner-only')
        if metadata.st_size < 1 or metadata.st_size > maximum:
            raise OperatorTrustError('operator trust input size is invalid')
        payload = os.read(descriptor, maximum + 1)
        if len(payload) != metadata.st_size:
            raise OperatorTrustError('operator trust input changed while being read')
        return payload
    finally:
        os.close(descriptor)


def _validate_executable(
    value: str,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> None:
    _copy_and_validate_executable(
        value,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        destination_descriptor=None,
    )


@contextmanager
def _verified_executable_copy(
    value: str,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> Iterator[str]:
    """Yield a private execute-only copy of the exact bytes that passed verification."""

    try:
        with tempfile.TemporaryDirectory(prefix='vaxreplay-isolated-executable-') as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            root_metadata = root.stat()
            if (
                not stat.S_ISDIR(root_metadata.st_mode)
                or root_metadata.st_uid != os.geteuid()
                or root_metadata.st_mode & 0o077
            ):
                raise OperatorTrustError('isolated-process executable staging is unsafe')
            executable = root / 'executable'
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
            destination = os.open(executable, flags, 0o700)
            try:
                _copy_and_validate_executable(
                    value,
                    expected_sha256=expected_sha256,
                    expected_byte_count=expected_byte_count,
                    destination_descriptor=destination,
                )
                os.fsync(destination)
                os.fchmod(destination, 0o500)
                copied = os.fstat(destination)
                if (
                    not stat.S_ISREG(copied.st_mode)
                    or copied.st_uid != os.geteuid()
                    or copied.st_mode & 0o277
                    or copied.st_size != expected_byte_count
                ):
                    raise OperatorTrustError('isolated-process executable staging is unsafe')
            finally:
                os.close(destination)
            yield str(executable)
    except OperatorTrustError:
        raise
    except OSError:
        raise OperatorTrustError('isolated-process executable staging failed') from None


def _copy_and_validate_executable(
    value: str,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    destination_descriptor: int | None,
) -> None:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(value, flags)
    except OSError:
        raise OperatorTrustError('isolated-process executable is unavailable') from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111 == 0:
            raise OperatorTrustError('isolated-process executable is not an executable regular file')
        if metadata.st_uid not in {0, os.geteuid()} or metadata.st_mode & 0o022:
            raise OperatorTrustError('isolated-process executable ownership or permissions are unsafe')
        if metadata.st_size != expected_byte_count:
            raise OperatorTrustError('isolated-process executable differs from its trusted binding')
        digest = hashlib.sha256()
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise OperatorTrustError('isolated-process executable changed while being read')
            digest.update(chunk)
            if destination_descriptor is not None:
                _write_all(destination_descriptor, chunk)
            remaining -= len(chunk)
        final_metadata = os.fstat(descriptor)
        if (
            final_metadata.st_dev != metadata.st_dev
            or final_metadata.st_ino != metadata.st_ino
            or final_metadata.st_size != metadata.st_size
            or final_metadata.st_mtime_ns != metadata.st_mtime_ns
            or final_metadata.st_ctime_ns != metadata.st_ctime_ns
        ):
            raise OperatorTrustError('isolated-process executable changed while being read')
        if not _is_sha256(expected_sha256) or not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise OperatorTrustError('isolated-process executable differs from its trusted binding')
    except OSError:
        raise OperatorTrustError('isolated-process executable is unavailable') from None
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError('short executable-copy write')
        offset += written


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in '0123456789abcdef' for character in value)


def _constant_digest(payload: bytes, expected: str) -> bool:
    return hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected)


def _canonical_base64(value: str, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (TypeError, ValueError):
        raise ValueError(f'{label} is invalid base64') from None
    if base64.b64encode(decoded).decode('ascii') != value:
        raise ValueError(f'{label} is not canonical base64')
    return decoded


__all__ = [
    'CLOCK_HEALTH_REQUEST_SCHEMA_VERSION',
    'EXTERNAL_SIGNER_REQUEST_SCHEMA_VERSION',
    'EXTERNAL_SIGNER_RESPONSE_SCHEMA_VERSION',
    'ISOLATED_PROCESS_CONFIG_SCHEMA_VERSION',
    'ClockHealthRequest',
    'ExternalSignerRequest',
    'ExternalSignerResponse',
    'IsolatedProcessConfig',
    'OperatorTrustError',
    'SubprocessClockHealthProvider',
    'SubprocessEd25519Signer',
    'add_signer_arguments',
    'load_clock_health_gate',
    'load_dev_local_signer',
    'load_external_signer',
    'load_isolated_process_config',
    'resolve_operator_trust',
    'signer_and_clock_from_args',
]
