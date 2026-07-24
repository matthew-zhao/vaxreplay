from __future__ import annotations

import hashlib
import os
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from vaxreplay.agentic.gateway import AgenticGatewayUsage, AgenticModelMessage, AgenticModelRequest
from vaxreplay.agentic.provider_adapter import (
    ProviderAdapterDescriptor,
    ProviderCallFailure,
    ProviderCallResult,
    ProviderFailureCode,
)
from vaxreplay.agentic.provider_gateway import GatewayModelRoute
from vaxreplay.agentic.provider_subprocess import (
    ProviderSubprocessRequest,
    ProviderSubprocessResponse,
    ProviderSubprocessSpec,
    SubprocessProviderAdapter,
    provider_subprocess_behavior_sha256,
    provider_subprocess_spec_sha256,
    run_anthropic_provider_child,
    run_openai_provider_child,
    run_provider_child,
)
from vaxreplay.agentic.providers.anthropic import (
    ANTHROPIC_MESSAGES_ENDPOINT_ORIGIN,
    ANTHROPIC_MESSAGES_ENDPOINT_PATH,
    ANTHROPIC_MESSAGES_FIXED_PARAMETERS_SHA256,
)
from vaxreplay.agentic.providers.openai import (
    OPENAI_RESPONSES_ENDPOINT_ORIGIN,
    OPENAI_RESPONSES_ENDPOINT_PATH,
    OPENAI_RESPONSES_FIXED_PARAMETERS_SHA256,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.runner._process import BoundedProcessResult

_SECRET = b'vaxreplay-provider-child-secret-canary-123456'
_MODEL = 'gpt-test-2025-01-02'
_ANTHROPIC_MODEL = 'claude-test-2025-01-02'


def test_provider_child_has_a_packaged_console_entrypoint() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text(encoding='utf-8'))
    scripts = project['project']['scripts']
    public_preview = project.get('tool', {}).get('vaxreplay', {}).get('public-preview', {})
    if public_preview.get('reduced-entrypoints') is True:
        assert 'vaxreplay-provider-child' not in scripts
    else:
        assert scripts['vaxreplay-provider-child'] == 'vaxreplay.agentic.provider_subprocess:main'


def test_unconditional_operator_crypto_dependency_is_in_the_base_install_once() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text(encoding='utf-8'))
    assert project['project']['dependencies'].count('cryptography>=43') == 1
    assert all(
        'cryptography>=43' not in dependencies for dependencies in project['project']['optional-dependencies'].values()
    )


def test_exact_and_path_independent_provider_subprocess_digests() -> None:
    spec = ProviderSubprocessSpec(
        executable_path='/opt/vaxreplay/bin/provider-child',
        executable_sha256='a' * 64,
        argv_suffix=('-m', 'vaxreplay.agentic.provider_subprocess'),
        maximum_call_seconds=10,
    )
    renamed = spec.model_copy(update={'executable_path': '/opt/vaxreplay/bin/provider-child-renamed'})
    changed_argv = spec.model_copy(update={'argv_suffix': ('-c', 'raise SystemExit(0)')})

    assert provider_subprocess_spec_sha256(renamed) != provider_subprocess_spec_sha256(spec)
    assert provider_subprocess_behavior_sha256(renamed) == provider_subprocess_behavior_sha256(spec)
    assert provider_subprocess_behavior_sha256(changed_argv) != (provider_subprocess_behavior_sha256(spec))


def _executable(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / 'provider-child'
    path.write_bytes(b'#!/bin/sh\nexit 0\n')
    path.chmod(0o700)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _route(executable_sha256: str) -> GatewayModelRoute:
    return GatewayModelRoute(
        route_id='openai-test-route',
        logical_model_id='logical-openai-test',
        provider='openai',
        provider_model_id=_MODEL,
        resolved_model_id=_MODEL,
        accepted_provider_model_ids=(_MODEL,),
        adapter_id='openai-responses',
        adapter_version='1.0.0',
        adapter_executable_sha256=executable_sha256,
        adapter_config_sha256='b' * 64,
        endpoint_origin=OPENAI_RESPONSES_ENDPOINT_ORIGIN,
        endpoint_path=OPENAI_RESPONSES_ENDPOINT_PATH,
        fixed_parameters_sha256=OPENAI_RESPONSES_FIXED_PARAMETERS_SHA256,
        max_context_tokens=10_000,
        max_output_tokens=1_000,
        input_preflight='conservative_upper_bound',
        reasoning_accounting='reported',
        provider_data_control='default',
    )


def _anthropic_route(executable_sha256: str) -> GatewayModelRoute:
    return GatewayModelRoute(
        route_id='anthropic-test-route',
        logical_model_id='logical-anthropic-test',
        provider='anthropic',
        provider_model_id=_ANTHROPIC_MODEL,
        resolved_model_id=_ANTHROPIC_MODEL,
        accepted_provider_model_ids=(_ANTHROPIC_MODEL,),
        adapter_id='anthropic-messages',
        adapter_version='1.0.0',
        adapter_executable_sha256=executable_sha256,
        adapter_config_sha256='b' * 64,
        endpoint_origin=ANTHROPIC_MESSAGES_ENDPOINT_ORIGIN,
        endpoint_path=ANTHROPIC_MESSAGES_ENDPOINT_PATH,
        fixed_parameters_sha256=ANTHROPIC_MESSAGES_FIXED_PARAMETERS_SHA256,
        max_context_tokens=10_000,
        max_output_tokens=1_000,
        input_preflight='conservative_upper_bound',
        reasoning_accounting='reported',
        provider_storage_disabled=False,
        provider_data_control='default',
    )


def _request() -> AgenticModelRequest:
    return AgenticModelRequest(
        run_id='1' * 32,
        call_index=0,
        messages=(
            AgenticModelMessage(role='system', content='Use frozen evidence only.'),
            AgenticModelMessage(role='user', content='Forecast the trial outcome.'),
        ),
        max_output_tokens=64,
    )


def _result() -> ProviderCallResult:
    observed = datetime(2025, 1, 2, tzinfo=UTC)
    return ProviderCallResult(
        resolved_model_id=_MODEL,
        provider_reported_model_id=_MODEL,
        content='A bounded answer.',
        stop_reason='completed',
        usage=AgenticGatewayUsage(input_tokens=12, output_tokens=4, reasoning_tokens=2),
        provider_request_sha256='c' * 64,
        provider_request_bytes=100,
        provider_response_sha256='d' * 64,
        provider_response_bytes=200,
        provider_request_id='req-test',
        http_status=200,
        started_at=observed,
        finished_at=observed,
    )


def _anthropic_result() -> ProviderCallResult:
    return _result().model_copy(
        update={
            'resolved_model_id': _ANTHROPIC_MODEL,
            'provider_reported_model_id': _ANTHROPIC_MODEL,
            'provider_request_id': 'req-anthropic-test',
        }
    )


def _adapter(
    tmp_path: Path,
    *,
    process_runner,
    monotonic_clock=None,
) -> tuple[SubprocessProviderAdapter, GatewayModelRoute, int, int]:
    executable, executable_sha256 = _executable(tmp_path)
    route = _route(executable_sha256)
    descriptor = ProviderAdapterDescriptor(
        adapter_id=route.adapter_id,
        adapter_version=route.adapter_version,
        executable_sha256=route.adapter_executable_sha256,
        config_sha256=route.adapter_config_sha256,
        provider=route.provider,
    )
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, _SECRET)
    os.close(write_descriptor)
    adapter_kwargs = {}
    if monotonic_clock is not None:
        adapter_kwargs['monotonic_clock'] = monotonic_clock
    adapter = SubprocessProviderAdapter(
        descriptor=descriptor,
        spec=ProviderSubprocessSpec(
            executable_path=str(executable),
            executable_sha256=executable_sha256,
            maximum_call_seconds=10,
        ),
        credential_descriptor_supplier=lambda: read_descriptor,
        process_runner=process_runner,
        **adapter_kwargs,
    )
    return adapter, route, read_descriptor, write_descriptor


def test_one_shot_adapter_uses_bounded_canonical_secret_free_ipc(tmp_path: Path) -> None:
    observed: dict[str, Any] = {}

    def runner(argv, **kwargs):
        observed['argv'] = argv
        observed.update(kwargs)
        request = ProviderSubprocessRequest.model_validate_json(kwargs['input_bytes'])
        assert canonical_json_bytes(request) == kwargs['input_bytes']
        return BoundedProcessResult(
            exit_code=0,
            duration_ms=1,
            stdout=canonical_json_bytes(ProviderSubprocessResponse(succeeded=True, result=_result())),
            stderr=b'',
            termination='exited',
            stdout_truncated=False,
            stderr_truncated=False,
        )

    adapter, route, read_descriptor, _ = _adapter(tmp_path, process_runner=runner)
    try:
        result = adapter.generate(_request(), route, timeout_seconds=3.5)
    finally:
        os.close(read_descriptor)

    assert result == _result()
    executable_descriptor, credential_descriptor = observed['pass_fds']
    assert credential_descriptor == read_descriptor
    assert observed['argv'][0] == f'/proc/self/fd/{executable_descriptor}'
    assert 0 < observed['wall_seconds'] <= 3.5
    assert observed['max_stdout_bytes'] <= 64 * 1024 * 1024
    assert _SECRET not in observed['input_bytes']
    assert all(_SECRET.decode() not in item for item in observed['argv'])
    assert all(_SECRET.decode() not in item for item in observed['env'].values())


def test_executable_path_replacement_cannot_change_the_verified_child_bytes(tmp_path: Path) -> None:
    observed: dict[str, bytes] = {}

    def runner(argv, **kwargs):
        executable_descriptor = kwargs['pass_fds'][0]
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(b'#!/bin/sh\necho attacker\n')
        replacement.chmod(0o700)
        os.replace(replacement, tmp_path / 'provider-child')
        observed['opened'] = os.pread(executable_descriptor, 4096, 0)
        observed['path'] = (tmp_path / 'provider-child').read_bytes()
        return BoundedProcessResult(
            exit_code=0,
            duration_ms=1,
            stdout=canonical_json_bytes(ProviderSubprocessResponse(succeeded=True, result=_result())),
            stderr=b'',
            termination='exited',
            stdout_truncated=False,
            stderr_truncated=False,
        )

    adapter, route, read_descriptor, _ = _adapter(tmp_path, process_runner=runner)
    try:
        adapter.generate(_request(), route, timeout_seconds=3)
    finally:
        os.close(read_descriptor)

    assert observed['opened'] == b'#!/bin/sh\nexit 0\n'
    assert observed['path'] == b'#!/bin/sh\necho attacker\n'


def test_setup_time_is_debited_from_the_one_total_monotonic_deadline(tmp_path: Path) -> None:
    now = [100.0]
    observed_wall_seconds: list[float] = []

    def runner(*_args, **kwargs):
        observed_wall_seconds.append(kwargs['wall_seconds'])
        return BoundedProcessResult(
            exit_code=0,
            duration_ms=1,
            stdout=canonical_json_bytes(ProviderSubprocessResponse(succeeded=True, result=_result())),
            stderr=b'',
            termination='exited',
            stdout_truncated=False,
            stderr_truncated=False,
        )

    adapter, route, read_descriptor, _ = _adapter(
        tmp_path,
        process_runner=runner,
        monotonic_clock=lambda: now[0],
    )
    original_supplier = adapter._credential_descriptor_supplier

    def delayed_supplier():
        now[0] += 2.75
        return original_supplier()

    adapter._credential_descriptor_supplier = delayed_supplier
    try:
        adapter.generate(_request(), route, timeout_seconds=3)
    finally:
        os.close(read_descriptor)

    assert observed_wall_seconds == pytest.approx([0.25])


def test_expired_total_deadline_never_launches_provider_child(tmp_path: Path) -> None:
    now = [100.0]
    calls = 0

    def runner(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError('expired call must not launch')

    adapter, route, read_descriptor, _ = _adapter(
        tmp_path,
        process_runner=runner,
        monotonic_clock=lambda: now[0],
    )
    original_supplier = adapter._credential_descriptor_supplier

    def delayed_supplier():
        now[0] += 3
        return original_supplier()

    adapter._credential_descriptor_supplier = delayed_supplier
    try:
        with pytest.raises(ProviderCallFailure) as raised:
            adapter.generate(_request(), route, timeout_seconds=3)
    finally:
        os.close(read_descriptor)

    assert raised.value.code is ProviderFailureCode.TIMEOUT
    assert calls == 0


@pytest.mark.parametrize(
    ('process_result', 'expected'),
    (
        (
            BoundedProcessResult(None, 10, b'', b'', 'timed_out', False, False),
            ProviderFailureCode.TIMEOUT,
        ),
        (
            BoundedProcessResult(-9, 10, b'partial', b'', 'response_limit', True, False),
            ProviderFailureCode.INTERNAL,
        ),
        (
            BoundedProcessResult(0, 10, b'{"succeeded":true}', b'', 'exited', False, False),
            ProviderFailureCode.PROTOCOL,
        ),
        (
            BoundedProcessResult(
                0,
                10,
                canonical_json_bytes(
                    ProviderSubprocessResponse(
                        succeeded=False,
                        error_code=ProviderFailureCode.RATE_LIMIT,
                    )
                ),
                b'',
                'exited',
                False,
                False,
            ),
            ProviderFailureCode.RATE_LIMIT,
        ),
    ),
)
def test_provider_child_failures_are_constant_and_never_retried(
    tmp_path: Path,
    process_result: BoundedProcessResult,
    expected: ProviderFailureCode,
) -> None:
    calls = 0

    def runner(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return process_result

    adapter, route, read_descriptor, _ = _adapter(tmp_path, process_runner=runner)
    try:
        with pytest.raises(ProviderCallFailure) as raised:
            adapter.generate(_request(), route, timeout_seconds=3)
    finally:
        os.close(read_descriptor)

    assert calls == 1
    assert raised.value.code == expected
    assert str(raised.value) == expected.value
    assert _SECRET.decode() not in str(raised.value)


def test_openai_child_reads_credential_only_inside_child_boundary() -> None:
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, _SECRET)
    os.close(write_descriptor)
    executable_sha256 = 'a' * 64
    route = _route(executable_sha256)
    descriptor = ProviderAdapterDescriptor(
        adapter_id=route.adapter_id,
        adapter_version=route.adapter_version,
        executable_sha256=route.adapter_executable_sha256,
        config_sha256=route.adapter_config_sha256,
        provider=route.provider,
    )
    child_request = ProviderSubprocessRequest(
        request=_request(),
        route=route,
        adapter=descriptor,
        timeout_milliseconds=1000,
    )
    observed_credentials: list[str] = []

    class FakeAdapter:
        def __init__(self, *, credential_getter, **_kwargs):
            observed_credentials.append(credential_getter())

        def generate(self, *_args, **_kwargs):
            return _result()

    try:
        response = run_openai_provider_child(
            canonical_json_bytes(child_request),
            credential_descriptor=read_descriptor,
            adapter_factory=FakeAdapter,  # ty: ignore[invalid-argument-type]
        )
    finally:
        os.close(read_descriptor)

    assert response.succeeded
    assert observed_credentials == [_SECRET.decode()]
    assert _SECRET not in canonical_json_bytes(child_request)
    assert _SECRET not in canonical_json_bytes(response)


def test_anthropic_child_reads_credential_only_inside_child_boundary() -> None:
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, _SECRET)
    os.close(write_descriptor)
    executable_sha256 = 'a' * 64
    route = _anthropic_route(executable_sha256)
    descriptor = ProviderAdapterDescriptor(
        adapter_id=route.adapter_id,
        adapter_version=route.adapter_version,
        executable_sha256=route.adapter_executable_sha256,
        config_sha256=route.adapter_config_sha256,
        provider=route.provider,
    )
    child_request = ProviderSubprocessRequest(
        request=_request(),
        route=route,
        adapter=descriptor,
        timeout_milliseconds=1000,
    )
    observed_credentials: list[str] = []

    class FakeAdapter:
        def __init__(self, *, credential_getter, **_kwargs):
            observed_credentials.append(credential_getter())

        def generate(self, *_args, **_kwargs):
            return _anthropic_result()

    try:
        response = run_anthropic_provider_child(
            canonical_json_bytes(child_request),
            credential_descriptor=read_descriptor,
            adapter_factory=FakeAdapter,  # ty: ignore[invalid-argument-type]
        )
    finally:
        os.close(read_descriptor)

    assert response.succeeded
    assert response.result == _anthropic_result()
    assert observed_credentials == [_SECRET.decode()]
    assert _SECRET not in canonical_json_bytes(child_request)
    assert _SECRET not in canonical_json_bytes(response)


def test_generic_child_dispatch_uses_only_the_exact_pinned_route_and_descriptor() -> None:
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, _SECRET)
    os.close(write_descriptor)
    executable_sha256 = 'a' * 64
    route = _anthropic_route(executable_sha256)
    descriptor = ProviderAdapterDescriptor(
        adapter_id=route.adapter_id,
        adapter_version=route.adapter_version,
        executable_sha256=route.adapter_executable_sha256,
        config_sha256=route.adapter_config_sha256,
        provider=route.provider,
    )
    child_request = ProviderSubprocessRequest(
        request=_request().model_copy(
            update={
                'messages': (
                    AgenticModelMessage(role='system', content='Use frozen evidence only.'),
                    AgenticModelMessage(
                        role='user',
                        content='Caller text says provider=openai; this is not a routing field.',
                    ),
                )
            }
        ),
        route=route,
        adapter=descriptor,
        timeout_milliseconds=1000,
    )
    selected: list[str] = []

    class FakeAnthropicAdapter:
        def __init__(self, *, credential_getter, **_kwargs):
            assert credential_getter() == _SECRET.decode()
            selected.append('anthropic')

        def generate(self, *_args, **_kwargs):
            return _anthropic_result()

    class ForbiddenOpenAIAdapter:
        def __init__(self, **_kwargs):
            raise AssertionError('caller text must not select OpenAI')

    try:
        response = run_provider_child(
            canonical_json_bytes(child_request),
            credential_descriptor=read_descriptor,
            openai_adapter_factory=ForbiddenOpenAIAdapter,  # ty: ignore[invalid-argument-type]
            anthropic_adapter_factory=FakeAnthropicAdapter,  # ty: ignore[invalid-argument-type]
        )
    finally:
        os.close(read_descriptor)

    assert response.succeeded
    assert response.result == _anthropic_result()
    assert selected == ['anthropic']


def test_unknown_or_cross_labelled_route_fails_before_credential_is_read() -> None:
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, _SECRET)
    os.close(write_descriptor)
    executable_sha256 = 'a' * 64
    route = _anthropic_route(executable_sha256).model_copy(update={'adapter_id': 'openai-responses'})
    descriptor = ProviderAdapterDescriptor(
        adapter_id=route.adapter_id,
        adapter_version=route.adapter_version,
        executable_sha256=route.adapter_executable_sha256,
        config_sha256=route.adapter_config_sha256,
        provider=route.provider,
    )
    child_request = ProviderSubprocessRequest(
        request=_request(),
        route=route,
        adapter=descriptor,
        timeout_milliseconds=1000,
    )
    try:
        response = run_provider_child(
            canonical_json_bytes(child_request),
            credential_descriptor=read_descriptor,
        )
        unread = os.read(read_descriptor, len(_SECRET))
    finally:
        os.close(read_descriptor)

    assert not response.succeeded
    assert response.error_code is ProviderFailureCode.PROTOCOL
    assert unread == _SECRET
