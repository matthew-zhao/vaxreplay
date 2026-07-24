"""Forcibly cancellable one-shot process boundary for provider calls.

The host gateway sends one canonical, bounded request to a fresh child process and accepts one
canonical, bounded response.  The provider credential is never included in argv, the environment,
or the IPC document: a runtime secret broker supplies a pre-opened descriptor which is inherited by
the child.  The parent deliberately never reads that descriptor.

This module is a process-containment and evidence-integrity primitive.  It does not attest a
provider model snapshot, qualify a Linux/KVM deployment, or turn a software-held descriptor into an
HSM-backed secret broker.  The packaged console wrapper and the operator's separate module-source
pin also do not attest the child interpreter, import resolution, transitive dependencies, or
executing process image.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.gateway import AgenticModelRequest
from vaxreplay.agentic.provider_adapter import (
    ProviderAdapterDescriptor,
    ProviderCallFailure,
    ProviderCallResult,
    ProviderFailureCode,
    ProviderModelRoute,
)
from vaxreplay.agentic.provider_gateway import GatewayModelRoute
from vaxreplay.agentic.providers.anthropic import (
    ANTHROPIC_MESSAGES_ADAPTER_ID,
    ANTHROPIC_MESSAGES_ENDPOINT_ORIGIN,
    ANTHROPIC_MESSAGES_ENDPOINT_PATH,
    ANTHROPIC_MESSAGES_FIXED_PARAMETERS_SHA256,
    AnthropicMessagesAdapter,
)
from vaxreplay.agentic.providers.openai import (
    OPENAI_RESPONSES_ADAPTER_ID,
    OPENAI_RESPONSES_ENDPOINT_ORIGIN,
    OPENAI_RESPONSES_ENDPOINT_PATH,
    OPENAI_RESPONSES_FIXED_PARAMETERS_SHA256,
    OpenAIResponsesAdapter,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.runner._process import BoundedProcessResult, run_bounded_process

PROVIDER_SUBPROCESS_SPEC_SCHEMA_VERSION = 'vaxreplay.provider-subprocess-spec.dev-v0.1'
PROVIDER_SUBPROCESS_BEHAVIOR_SCHEMA_VERSION = 'vaxreplay.provider-subprocess-behavior.dev-v0.1'
PROVIDER_SUBPROCESS_REQUEST_SCHEMA_VERSION = 'vaxreplay.provider-subprocess-request.dev-v0.1'
PROVIDER_SUBPROCESS_RESPONSE_SCHEMA_VERSION = 'vaxreplay.provider-subprocess-response.dev-v0.1'

_CREDENTIAL_FD_ENVIRONMENT = 'VAXREPLAY_PROVIDER_CREDENTIAL_FD'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAXIMUM_EXECUTABLE_BYTES = 256 * 1024 * 1024
_MAXIMUM_CREDENTIAL_BYTES = 16 * 1024
_MAXIMUM_CHILD_REQUEST_BYTES = 16 * 1024 * 1024
_MAXIMUM_CHILD_RESPONSE_BYTES = 32 * 1024 * 1024
_MAXIMUM_CHILD_LOG_BYTES = 64 * 1024


class ProviderCredentialDescriptorSupplier(Protocol):
    """Return a fresh or rewind-safe inherited descriptor without reading its contents."""

    def __call__(self) -> int: ...


class ProviderProcessRunner(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        input_bytes: bytes,
        wall_seconds: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        on_abort: Callable[[], None],
        env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> BoundedProcessResult: ...


class ProviderSubprocessSpec(StrictModel):
    """Pinned launch and IPC limits for the one-shot provider child."""

    schema_version: Literal['vaxreplay.provider-subprocess-spec.dev-v0.1'] = PROVIDER_SUBPROCESS_SPEC_SCHEMA_VERSION
    executable_path: str = Field(min_length=2, max_length=4096)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    argv_suffix: tuple[str, ...] = Field(default=(), max_length=32)
    maximum_call_seconds: int = Field(ge=1, le=3600)
    maximum_request_bytes: int = Field(default=_MAXIMUM_CHILD_REQUEST_BYTES, ge=1024, le=64 * 1024 * 1024)
    maximum_response_bytes: int = Field(default=_MAXIMUM_CHILD_RESPONSE_BYTES, ge=1024, le=64 * 1024 * 1024)
    maximum_log_bytes: int = Field(default=_MAXIMUM_CHILD_LOG_BYTES, ge=0, le=1024 * 1024)
    one_provider_call_per_process: Literal[True] = True
    canonical_bounded_ipc: Literal[True] = True
    automatic_retries: Literal[False] = False
    parent_reads_provider_credential: Literal[False] = False
    credential_material_in_argv_environment_or_ipc: Literal[False] = False
    process_group_kill_on_deadline_or_overflow: Literal[True] = True

    @field_validator('executable_path')
    @classmethod
    def validate_executable_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('provider child executable path must be absolute and normalized')
        return value

    @field_validator('argv_suffix')
    @classmethod
    def validate_argv_suffix(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or '\x00' in item or len(item) > 4096 for item in value):
            raise ValueError('provider child arguments must be bounded nonempty NUL-free strings')
        return value


def provider_subprocess_spec_sha256(spec: ProviderSubprocessSpec) -> str:
    """Bind the exact versioned child launch specification to one canonical digest."""

    canonical = ProviderSubprocessSpec.model_validate_json(canonical_json_bytes(spec))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def provider_subprocess_behavior_sha256(spec: ProviderSubprocessSpec) -> str:
    """Bind child behavior while excluding the renameable host executable path."""

    canonical = ProviderSubprocessSpec.model_validate_json(canonical_json_bytes(spec))
    behavior = {
        'schema_version': PROVIDER_SUBPROCESS_BEHAVIOR_SCHEMA_VERSION,
        'provider_subprocess_spec_schema_version': canonical.schema_version,
        'executable_sha256': canonical.executable_sha256,
        'argv_suffix': canonical.argv_suffix,
        'maximum_call_seconds': canonical.maximum_call_seconds,
        'maximum_request_bytes': canonical.maximum_request_bytes,
        'maximum_response_bytes': canonical.maximum_response_bytes,
        'maximum_log_bytes': canonical.maximum_log_bytes,
        'one_provider_call_per_process': canonical.one_provider_call_per_process,
        'canonical_bounded_ipc': canonical.canonical_bounded_ipc,
        'automatic_retries': canonical.automatic_retries,
        'parent_reads_provider_credential': canonical.parent_reads_provider_credential,
        'credential_material_in_argv_environment_or_ipc': (canonical.credential_material_in_argv_environment_or_ipc),
        'process_group_kill_on_deadline_or_overflow': (canonical.process_group_kill_on_deadline_or_overflow),
    }
    return hashlib.sha256(canonical_json_bytes(behavior)).hexdigest()


class ProviderSubprocessRequest(StrictModel):
    schema_version: Literal['vaxreplay.provider-subprocess-request.dev-v0.1'] = (
        PROVIDER_SUBPROCESS_REQUEST_SCHEMA_VERSION
    )
    request: AgenticModelRequest
    route: GatewayModelRoute
    adapter: ProviderAdapterDescriptor
    timeout_milliseconds: int = Field(ge=1, le=3_600_000)

    @model_validator(mode='after')
    def validate_adapter_route(self) -> Self:
        expected = (
            self.route.provider,
            self.route.adapter_id,
            self.route.adapter_version,
            self.route.adapter_executable_sha256,
            self.route.adapter_config_sha256,
        )
        actual = (
            self.adapter.provider,
            self.adapter.adapter_id,
            self.adapter.adapter_version,
            self.adapter.executable_sha256,
            self.adapter.config_sha256,
        )
        if actual != expected:
            raise ValueError('provider child request adapter differs from its fixed route')
        return self


class ProviderSubprocessResponse(StrictModel):
    schema_version: Literal['vaxreplay.provider-subprocess-response.dev-v0.1'] = (
        PROVIDER_SUBPROCESS_RESPONSE_SCHEMA_VERSION
    )
    succeeded: bool
    result: ProviderCallResult | None = None
    error_code: ProviderFailureCode | None = None

    @model_validator(mode='after')
    def validate_outcome(self) -> Self:
        if self.succeeded != (self.result is not None) or self.succeeded == (self.error_code is not None):
            raise ValueError('provider child response must contain exactly one result or stable error')
        return self


class SubprocessProviderAdapter:
    """Provider adapter whose irreversible ``generate`` call always occurs in a fresh child."""

    forcibly_cancellable_provider_calls: Literal[True] = True
    provider_credentials_child_side: Literal[True] = True

    def __init__(
        self,
        *,
        descriptor: ProviderAdapterDescriptor,
        spec: ProviderSubprocessSpec,
        credential_descriptor_supplier: ProviderCredentialDescriptorSupplier,
        process_runner: ProviderProcessRunner = run_bounded_process,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if descriptor.executable_sha256 != spec.executable_sha256:
            raise ValueError('provider subprocess executable differs from the adapter descriptor')
        if (
            not callable(credential_descriptor_supplier)
            or not callable(process_runner)
            or not callable(monotonic_clock)
        ):
            raise TypeError('provider subprocess dependencies must be callable')
        self._descriptor = ProviderAdapterDescriptor.model_validate_json(canonical_json_bytes(descriptor))
        self._spec = ProviderSubprocessSpec.model_validate_json(canonical_json_bytes(spec))
        self._credential_descriptor_supplier = credential_descriptor_supplier
        self._process_runner = process_runner
        self._monotonic = monotonic_clock
        _verify_executable(self._spec)

    @property
    def descriptor(self) -> ProviderAdapterDescriptor:
        return self._descriptor

    @property
    def spec(self) -> ProviderSubprocessSpec:
        return self._spec

    def estimate_input_tokens(self, request: AgenticModelRequest, route: ProviderModelRoute) -> int:
        """Return a credential-free conservative UTF-8 byte upper bound."""

        try:
            request_bytes = canonical_json_bytes(request)
            route_overhead = len(route.provider_model_id.encode('utf-8'))
        except (AttributeError, TypeError, ValueError):
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL) from None
        # Every tokenizer token consumes at least one request byte for the supported UTF-8 route;
        # 4096 bytes covers the fixed provider envelope with a deliberately conservative margin.
        return len(request_bytes) + route_overhead + 4096

    def generate(
        self,
        request: AgenticModelRequest,
        route: ProviderModelRoute,
        *,
        timeout_seconds: float,
    ) -> ProviderCallResult:
        started = self._monotonic()
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or not isinstance(route, GatewayModelRoute)
        ):
            raise ProviderCallFailure(ProviderFailureCode.TIMEOUT)
        total_seconds = min(float(timeout_seconds), float(self._spec.maximum_call_seconds))
        deadline = started + total_seconds
        executable_descriptor: int | None = None
        try:
            executable_descriptor = _open_verified_executable(self._spec)
            credential_descriptor = self._credential_descriptor_supplier()
            _validate_inherited_descriptor(credential_descriptor)
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise ProviderCallFailure(ProviderFailureCode.TIMEOUT)
            child_request = ProviderSubprocessRequest(
                request=request,
                route=route,
                adapter=self._descriptor,
                timeout_milliseconds=max(1, min(3_600_000, math.floor(remaining * 1000))),
            )
            request_bytes = canonical_json_bytes(child_request)
            if len(request_bytes) > self._spec.maximum_request_bytes:
                raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)
            result = self._process_runner(
                (f'/proc/self/fd/{executable_descriptor}', *self._spec.argv_suffix),
                input_bytes=request_bytes,
                wall_seconds=remaining,
                max_stdout_bytes=self._spec.maximum_response_bytes,
                max_stderr_bytes=self._spec.maximum_log_bytes,
                on_abort=lambda: None,
                env={
                    'LANG': 'C',
                    'LC_ALL': 'C',
                    'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
                    _CREDENTIAL_FD_ENVIRONMENT: str(credential_descriptor),
                },
                pass_fds=(executable_descriptor, credential_descriptor),
            )
        except ProviderCallFailure:
            raise
        except BaseException:
            raise ProviderCallFailure(ProviderFailureCode.INTERNAL) from None
        finally:
            if executable_descriptor is not None:
                os.close(executable_descriptor)

        if result.termination == 'timed_out' or self._monotonic() > deadline:
            raise ProviderCallFailure(ProviderFailureCode.TIMEOUT)
        if (
            result.termination != 'exited'
            or result.exit_code != 0
            or result.stderr
            or result.stdout_truncated
            or result.stderr_truncated
            or not result.stdout
        ):
            raise ProviderCallFailure(ProviderFailureCode.INTERNAL)
        try:
            response = ProviderSubprocessResponse.model_validate_json(result.stdout)
        except ValueError:
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL) from None
        if canonical_json_bytes(response) != result.stdout:
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)
        if not response.succeeded or response.result is None:
            raise ProviderCallFailure(response.error_code or ProviderFailureCode.INTERNAL)
        return response.result


def run_openai_provider_child(
    request_bytes: bytes,
    *,
    credential_descriptor: int,
    adapter_factory: Callable[..., OpenAIResponsesAdapter] = OpenAIResponsesAdapter,
) -> ProviderSubprocessResponse:
    """Execute the strict OpenAI child request; safe failures contain only an enum code."""

    if len(request_bytes) > _MAXIMUM_CHILD_REQUEST_BYTES:
        return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.PROTOCOL)
    try:
        request = ProviderSubprocessRequest.model_validate_json(request_bytes)
        if canonical_json_bytes(request) != request_bytes:
            raise ValueError('noncanonical child request')
        if _selected_provider(request) != 'openai':
            raise ValueError('child request does not select the pinned OpenAI adapter')
    except (TypeError, ValueError):
        return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.PROTOCOL)

    credential_buffer: bytearray | None = None
    credential_text: str | None = None
    try:
        credential_buffer = _read_credential_descriptor(credential_descriptor)
        credential_text = bytes(credential_buffer).decode('ascii')
        adapter = adapter_factory(
            credential_getter=lambda: credential_text,
            executable_sha256=request.adapter.executable_sha256,
            config_sha256=request.adapter.config_sha256,
            adapter_version=request.adapter.adapter_version,
        )
        result = adapter.generate(
            request.request,
            request.route,
            timeout_seconds=request.timeout_milliseconds / 1000,
        )
        return ProviderSubprocessResponse(succeeded=True, result=result)
    except ProviderCallFailure as error:
        return ProviderSubprocessResponse(succeeded=False, error_code=error.code)
    except BaseException:
        return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.INTERNAL)
    finally:
        credential_text = None
        if credential_buffer is not None:
            for index in range(len(credential_buffer)):
                credential_buffer[index] = 0


def run_anthropic_provider_child(
    request_bytes: bytes,
    *,
    credential_descriptor: int,
    adapter_factory: Callable[..., AnthropicMessagesAdapter] = AnthropicMessagesAdapter,
) -> ProviderSubprocessResponse:
    """Execute the strict Anthropic child request; safe failures contain only an enum code."""

    if len(request_bytes) > _MAXIMUM_CHILD_REQUEST_BYTES:
        return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.PROTOCOL)
    try:
        request = ProviderSubprocessRequest.model_validate_json(request_bytes)
        if canonical_json_bytes(request) != request_bytes:
            raise ValueError('noncanonical child request')
        if _selected_provider(request) != 'anthropic':
            raise ValueError('child request does not select the pinned Anthropic adapter')
    except (TypeError, ValueError):
        return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.PROTOCOL)

    credential_buffer: bytearray | None = None
    credential_text: str | None = None
    try:
        credential_buffer = _read_credential_descriptor(credential_descriptor)
        credential_text = bytes(credential_buffer).decode('ascii')
        adapter = adapter_factory(
            credential_getter=lambda: credential_text,
            executable_sha256=request.adapter.executable_sha256,
            config_sha256=request.adapter.config_sha256,
            adapter_version=request.adapter.adapter_version,
        )
        result = adapter.generate(
            request.request,
            request.route,
            timeout_seconds=request.timeout_milliseconds / 1000,
        )
        return ProviderSubprocessResponse(succeeded=True, result=result)
    except ProviderCallFailure as error:
        return ProviderSubprocessResponse(succeeded=False, error_code=error.code)
    except BaseException:
        return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.INTERNAL)
    finally:
        credential_text = None
        if credential_buffer is not None:
            for index in range(len(credential_buffer)):
                credential_buffer[index] = 0


def run_provider_child(
    request_bytes: bytes,
    *,
    credential_descriptor: int,
    openai_adapter_factory: Callable[..., OpenAIResponsesAdapter] = OpenAIResponsesAdapter,
    anthropic_adapter_factory: Callable[..., AnthropicMessagesAdapter] = AnthropicMessagesAdapter,
) -> ProviderSubprocessResponse:
    """Dispatch only from the canonical organizer-bound descriptor/route pair.

    The model request has no provider-selection field.  A provider is selected only when the
    canonical child envelope contains a mutually consistent adapter descriptor and route matching
    one compiled-in origin/path/policy tuple.  Unknown, cross-labelled, or noncanonical envelopes
    fail before the credential descriptor is read.
    """

    if len(request_bytes) > _MAXIMUM_CHILD_REQUEST_BYTES:
        return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.PROTOCOL)
    try:
        request = ProviderSubprocessRequest.model_validate_json(request_bytes)
        if canonical_json_bytes(request) != request_bytes:
            raise ValueError('noncanonical child request')
        provider = _selected_provider(request)
    except (TypeError, ValueError):
        return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.PROTOCOL)
    if provider == 'openai':
        return run_openai_provider_child(
            request_bytes,
            credential_descriptor=credential_descriptor,
            adapter_factory=openai_adapter_factory,
        )
    if provider == 'anthropic':
        return run_anthropic_provider_child(
            request_bytes,
            credential_descriptor=credential_descriptor,
            adapter_factory=anthropic_adapter_factory,
        )
    return ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.PROTOCOL)


def _selected_provider(request: ProviderSubprocessRequest) -> Literal['openai', 'anthropic']:
    """Return the one compiled-in adapter selected by both pinned descriptor and route."""

    selection = (
        request.adapter.provider,
        request.adapter.adapter_id,
        request.route.provider,
        request.route.adapter_id,
        request.route.endpoint_origin,
        request.route.endpoint_path,
        request.route.fixed_parameters_sha256,
        request.route.provider_storage_disabled,
    )
    if selection == (
        'openai',
        OPENAI_RESPONSES_ADAPTER_ID,
        'openai',
        OPENAI_RESPONSES_ADAPTER_ID,
        OPENAI_RESPONSES_ENDPOINT_ORIGIN,
        OPENAI_RESPONSES_ENDPOINT_PATH,
        OPENAI_RESPONSES_FIXED_PARAMETERS_SHA256,
        True,
    ):
        return 'openai'
    if selection == (
        'anthropic',
        ANTHROPIC_MESSAGES_ADAPTER_ID,
        'anthropic',
        ANTHROPIC_MESSAGES_ADAPTER_ID,
        ANTHROPIC_MESSAGES_ENDPOINT_ORIGIN,
        ANTHROPIC_MESSAGES_ENDPOINT_PATH,
        ANTHROPIC_MESSAGES_FIXED_PARAMETERS_SHA256,
        False,
    ):
        return 'anthropic'
    raise ValueError('provider child route does not select a compiled-in adapter')


def main() -> int:
    """One-request child entrypoint.  It never writes diagnostics or provider bodies."""

    response = ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.INTERNAL)
    try:
        credential_descriptor = _credential_descriptor_from_environment()
        request_bytes = _read_standard_input_bounded(_MAXIMUM_CHILD_REQUEST_BYTES)
        response = run_provider_child(
            request_bytes,
            credential_descriptor=credential_descriptor,
        )
    except BaseException:
        pass
    payload = canonical_json_bytes(response)
    if len(payload) > _MAXIMUM_CHILD_RESPONSE_BYTES:
        payload = canonical_json_bytes(
            ProviderSubprocessResponse(succeeded=False, error_code=ProviderFailureCode.INTERNAL)
        )
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


def _verify_executable(spec: ProviderSubprocessSpec) -> None:
    descriptor = _open_verified_executable(spec)
    os.close(descriptor)


def _open_verified_executable(spec: ProviderSubprocessSpec) -> int:
    """Open and hash the exact file later executed through its inherited Linux descriptor."""

    path = Path(spec.executable_path)
    try:
        path_metadata = path.lstat()
    except OSError as error:
        raise ValueError('provider child executable is unavailable') from error
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or not stat.S_ISREG(path_metadata.st_mode)
        or path_metadata.st_nlink != 1
        or path_metadata.st_size > _MAXIMUM_EXECUTABLE_BYTES
        or path_metadata.st_mode & 0o022
        or not path_metadata.st_mode & 0o100
    ):
        raise ValueError('provider child executable is not a pinned safe regular file')
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != path_metadata.st_dev
            or metadata.st_ino != path_metadata.st_ino
            or metadata.st_nlink != 1
            or metadata.st_size != path_metadata.st_size
            or metadata.st_mode & 0o022
            or not metadata.st_mode & 0o100
        ):
            raise ValueError('provider child executable changed while it was opened')
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != spec.executable_sha256:
            raise ValueError('provider child executable differs from its pinned digest')
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_inherited_descriptor(descriptor: int) -> None:
    if not isinstance(descriptor, int) or isinstance(descriptor, bool) or descriptor < 3:
        raise ValueError('provider credential descriptor must be an inherited descriptor at least 3')
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise ValueError('provider credential descriptor is unavailable') from error
    if stat.S_ISDIR(metadata.st_mode) or stat.S_ISSOCK(metadata.st_mode):
        raise ValueError('provider credential descriptor has an unsupported file type')


def _read_credential_descriptor(descriptor: int) -> bytearray:
    _validate_inherited_descriptor(descriptor)
    payload = bytearray()
    try:
        offset = 0
        while True:
            try:
                chunk = os.pread(descriptor, min(4096, _MAXIMUM_CREDENTIAL_BYTES + 2 - len(payload)), offset)
            except OSError:
                chunk = os.read(descriptor, min(4096, _MAXIMUM_CREDENTIAL_BYTES + 2 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            offset += len(chunk)
            if len(payload) > _MAXIMUM_CREDENTIAL_BYTES + 1:
                raise ValueError('provider credential exceeds its fixed byte limit')
        if payload.endswith(b'\n'):
            payload.pop()
        if not 16 <= len(payload) <= _MAXIMUM_CREDENTIAL_BYTES or any(not 0x21 <= byte <= 0x7E for byte in payload):
            raise ValueError('provider credential is malformed')
        return payload
    except BaseException:
        for index in range(len(payload)):
            payload[index] = 0
        raise


def _credential_descriptor_from_environment() -> int:
    value = os.environ.get(_CREDENTIAL_FD_ENVIRONMENT, '')
    if not value.isascii() or not value.isdigit() or len(value) > 10:
        raise ValueError('provider credential descriptor metadata is malformed')
    descriptor = int(value)
    _validate_inherited_descriptor(descriptor)
    return descriptor


def _read_standard_input_bounded(maximum_bytes: int) -> bytes:
    payload = bytearray()
    while True:
        chunk = os.read(sys.stdin.fileno(), min(65_536, maximum_bytes - len(payload) + 1))
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
        if len(payload) > maximum_bytes:
            raise ValueError('provider child request exceeds its fixed byte limit')


__all__ = [
    'PROVIDER_SUBPROCESS_BEHAVIOR_SCHEMA_VERSION',
    'PROVIDER_SUBPROCESS_REQUEST_SCHEMA_VERSION',
    'PROVIDER_SUBPROCESS_RESPONSE_SCHEMA_VERSION',
    'PROVIDER_SUBPROCESS_SPEC_SCHEMA_VERSION',
    'ProviderCredentialDescriptorSupplier',
    'ProviderProcessRunner',
    'ProviderSubprocessRequest',
    'ProviderSubprocessResponse',
    'ProviderSubprocessSpec',
    'SubprocessProviderAdapter',
    'main',
    'provider_subprocess_behavior_sha256',
    'provider_subprocess_spec_sha256',
    'run_anthropic_provider_child',
    'run_openai_provider_child',
    'run_provider_child',
]


if __name__ == '__main__':
    raise SystemExit(main())
