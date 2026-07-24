from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import pytest

from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest
from vaxreplay.agentic.provider_adapter import ProviderCallFailure, ProviderFailureCode
from vaxreplay.agentic.providers.anthropic import (
    ANTHROPIC_API_VERSION,
    ANTHROPIC_MESSAGES_ADAPTER_ID,
    ANTHROPIC_MESSAGES_ENDPOINT_ORIGIN,
    ANTHROPIC_MESSAGES_ENDPOINT_PATH,
    ANTHROPIC_MESSAGES_FIXED_PARAMETERS_SHA256,
    ANTHROPIC_MESSAGES_URL,
    AnthropicHttpRequest,
    AnthropicHttpResponse,
    AnthropicMessagesAdapter,
    AnthropicTransportFailure,
    anthropic_messages_fixed_parameters,
)
from vaxreplay.bundle import canonical_json_bytes

_SECRET = 'vaxreplay-anthropic-secret-canary-THIS_MUST_NEVER_LEAK_1234567890'
_MODEL = 'claude-test-2025-01-02'
_SHA_A = 'a' * 64
_SHA_B = 'b' * 64


@dataclass(frozen=True)
class _Route:
    provider: str = 'anthropic'
    provider_model_id: str = _MODEL
    resolved_model_id: str = _MODEL
    accepted_provider_model_ids: tuple[str, ...] = (_MODEL,)
    endpoint_origin: str = ANTHROPIC_MESSAGES_ENDPOINT_ORIGIN
    endpoint_path: str = ANTHROPIC_MESSAGES_ENDPOINT_PATH
    fixed_parameters_sha256: str = ANTHROPIC_MESSAGES_FIXED_PARAMETERS_SHA256
    provider_storage_disabled: bool = False
    provider_data_control: str = 'default'
    provider_data_control_attested: bool = False
    provider_data_control_attestation_sha256: str | None = None


class _CredentialGetter:
    def __init__(self, value: str = _SECRET, *, failure: Exception | None = None) -> None:
        self.value = value
        self.failure = failure
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.value


class _FakeTransport:
    def __init__(
        self,
        response: AnthropicHttpResponse | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.response = response or _response()
        self.failure = failure
        self.requests: list[tuple[AnthropicHttpRequest, float]] = []

    def send(
        self,
        request: AnthropicHttpRequest,
        *,
        timeout_seconds: float,
    ) -> AnthropicHttpResponse:
        self.requests.append((request, timeout_seconds))
        if self.failure is not None:
            raise self.failure
        return self.response


def _request(
    *,
    messages: tuple[AgenticModelMessage, ...] | None = None,
    response_schema_sha256: str | None = None,
) -> AgenticModelRequest:
    return AgenticModelRequest(
        run_id='1' * 32,
        call_index=0,
        messages=messages
        or (
            AgenticModelMessage(role='system', content='Use only the frozen evidence.'),
            AgenticModelMessage(role='user', content='Rank candidates A and B.'),
        ),
        max_output_tokens=64,
        response_schema_sha256=response_schema_sha256,
    )


def _payload(
    *,
    text: str = 'Candidate B ranks first.',
    model: str = _MODEL,
    stop_reason: str = 'end_turn',
    stop_sequence: str | None = None,
    content_type: str = 'text',
    input_tokens: object = 19,
    output_tokens: object = 7,
    thinking_tokens: object | None = 3,
) -> dict[str, object]:
    usage: dict[str, object] = {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
    }
    if thinking_tokens is not None:
        usage['output_tokens_details'] = {'thinking_tokens': thinking_tokens}
    block: dict[str, object]
    if content_type == 'text':
        block = {'type': 'text', 'text': text}
    else:
        block = {'type': content_type, 'id': 'toolu_private', 'name': 'forbidden', 'input': {}}
    return {
        'id': 'msg_fixture_123',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': [block],
        'stop_reason': stop_reason,
        'stop_sequence': stop_sequence,
        'usage': usage,
    }


def _response(
    *,
    payload: object | None = None,
    body: bytes | None = None,
    status_code: int = 200,
    final_url: str = ANTHROPIC_MESSAGES_URL,
    request_id: str | None = 'req_fixture_123',
    content_type: str = 'application/json; charset=utf-8',
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> AnthropicHttpResponse:
    actual_body = body if body is not None else canonical_json_bytes(payload if payload is not None else _payload())
    headers: list[tuple[str, str]] = [
        ('Content-Type', content_type),
        ('Content-Length', str(len(actual_body))),
    ]
    if request_id is not None:
        headers.append(('request-id', request_id))
    headers.extend(extra_headers)
    return AnthropicHttpResponse(
        status_code=status_code,
        final_url=final_url,
        headers=tuple(headers),
        body=actual_body,
    )


def _adapter(
    transport: _FakeTransport,
    getter: _CredentialGetter | None = None,
    *,
    maximum_request_bytes: int = 4 * 1024 * 1024,
    maximum_response_bytes: int = 16 * 1024 * 1024,
    maximum_response_header_bytes: int = 64 * 1024,
) -> tuple[AnthropicMessagesAdapter, _CredentialGetter]:
    actual_getter = getter or _CredentialGetter()
    return (
        AnthropicMessagesAdapter(
            credential_getter=actual_getter,
            executable_sha256=_SHA_A,
            config_sha256=_SHA_B,
            adapter_version='1.0.0',
            transport=transport,
            maximum_request_bytes=maximum_request_bytes,
            maximum_response_bytes=maximum_response_bytes,
            maximum_response_header_bytes=maximum_response_header_bytes,
        ),
        actual_getter,
    )


def _assert_failure(
    adapter: AnthropicMessagesAdapter,
    code: ProviderFailureCode,
    *,
    request: AgenticModelRequest | None = None,
    route: _Route | None = None,
) -> ProviderCallFailure:
    with pytest.raises(ProviderCallFailure) as raised:
        adapter.generate(request or _request(), route or _Route(), timeout_seconds=5)
    assert raised.value.code == code
    assert str(raised.value) == code.value
    assert _SECRET not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    return raised.value


def test_success_uses_one_fixed_nonstreaming_messages_request_and_exact_evidence() -> None:
    transport = _FakeTransport()
    adapter, getter = _adapter(transport)
    request = _request()

    result = adapter.generate(request, _Route(), timeout_seconds=8.5)

    assert getter.calls == 1
    assert len(transport.requests) == 1
    prepared, timeout = transport.requests[0]
    assert prepared.url == ANTHROPIC_MESSAGES_URL
    assert timeout == 8.5
    assert _SECRET not in repr(prepared)
    assert _SECRET.encode() not in prepared.body
    assert b'Candidate B ranks first.' not in repr(transport.response).encode()
    assert json.loads(prepared.body) == {
        'max_tokens': 64,
        'messages': [{'content': 'Rank candidates A and B.', 'role': 'user'}],
        'model': _MODEL,
        'stream': False,
        'system': 'Use only the frozen evidence.',
    }
    assert adapter.estimate_input_tokens(request, _Route()) == len(prepared.body)
    assert result.resolved_model_id == _MODEL
    assert result.provider_reported_model_id == _MODEL
    assert result.content == 'Candidate B ranks first.'
    assert result.stop_reason == 'completed'
    assert result.usage.input_tokens == 19
    assert result.usage.output_tokens == 7
    assert result.usage.reasoning_tokens == 3
    assert result.provider_request_sha256 == hashlib.sha256(prepared.body).hexdigest()
    assert result.provider_response_sha256 == hashlib.sha256(transport.response.body).hexdigest()
    assert result.provider_request_id == 'req_fixture_123'
    assert result.http_status == 200
    assert _SECRET.encode() not in canonical_json_bytes(result)
    assert adapter.descriptor.adapter_id == ANTHROPIC_MESSAGES_ADAPTER_ID
    assert adapter.descriptor.provider == 'anthropic'
    assert adapter.descriptor.automatic_retries is False
    assert adapter.descriptor.credential_passed_in_request is False
    assert anthropic_messages_fixed_parameters() == {
        'anthropic_version_header': ANTHROPIC_API_VERSION,
        'stream': False,
    }


def test_multi_turn_dialogue_is_preserved_without_provider_role_coalescing() -> None:
    request = _request(
        messages=(
            AgenticModelMessage(role='system', content='Use frozen evidence.'),
            AgenticModelMessage(role='user', content='First question.'),
            AgenticModelMessage(role='assistant', content='First answer.'),
            AgenticModelMessage(role='user', content='Second question.'),
        )
    )
    transport = _FakeTransport()
    adapter, _ = _adapter(transport)

    adapter.generate(request, _Route(), timeout_seconds=5)

    assert json.loads(transport.requests[0][0].body)['messages'] == [
        {'content': 'First question.', 'role': 'user'},
        {'content': 'First answer.', 'role': 'assistant'},
        {'content': 'Second question.', 'role': 'user'},
    ]


@pytest.mark.parametrize(
    ('provider_reason', 'mapped_reason'),
    [
        ('max_tokens', 'max_output_tokens'),
        ('refusal', 'refusal'),
    ],
)
def test_supported_terminal_stop_reasons_are_mapped(
    provider_reason: str,
    mapped_reason: str,
) -> None:
    adapter, _ = _adapter(_FakeTransport(_response(payload=_payload(stop_reason=provider_reason))))

    result = adapter.generate(_request(), _Route(), timeout_seconds=5)

    assert result.stop_reason == mapped_reason


@pytest.mark.parametrize(
    ('status_code', 'code'),
    [
        (302, ProviderFailureCode.PROTOCOL),
        (400, ProviderFailureCode.REJECTED),
        (401, ProviderFailureCode.REJECTED),
        (408, ProviderFailureCode.TIMEOUT),
        (413, ProviderFailureCode.REJECTED),
        (429, ProviderFailureCode.RATE_LIMIT),
        (500, ProviderFailureCode.INTERNAL),
        (504, ProviderFailureCode.TIMEOUT),
        (529, ProviderFailureCode.INTERNAL),
    ],
)
def test_http_failures_are_stable_codes_without_body_leakage(
    status_code: int,
    code: ProviderFailureCode,
) -> None:
    private_diagnostics = 'upstream-private-diagnostics-must-not-cross-boundary'
    transport = _FakeTransport(_response(status_code=status_code, body=private_diagnostics.encode()))
    adapter, _ = _adapter(transport)

    failure = _assert_failure(adapter, code)

    assert len(transport.requests) == 1
    assert private_diagnostics not in str(failure)


@pytest.mark.parametrize('code', list(ProviderFailureCode))
def test_safe_typed_transport_failures_are_not_retried(code: ProviderFailureCode) -> None:
    transport = _FakeTransport(failure=AnthropicTransportFailure(code))
    adapter, getter = _adapter(transport)

    _assert_failure(adapter, code)

    assert getter.calls == 1
    assert len(transport.requests) == 1


def test_arbitrary_transport_and_credential_exceptions_cannot_cross_boundary() -> None:
    adapter, _ = _adapter(_FakeTransport(failure=RuntimeError(f'network failed with {_SECRET}')))
    _assert_failure(adapter, ProviderFailureCode.INTERNAL)

    getter = _CredentialGetter(failure=RuntimeError(f'vault failed with {_SECRET}'))
    transport = _FakeTransport()
    adapter, _ = _adapter(transport, getter)
    _assert_failure(adapter, ProviderFailureCode.INTERNAL)
    assert transport.requests == []


@pytest.mark.parametrize(
    'route',
    [
        replace(_Route(), provider='other'),
        replace(_Route(), endpoint_origin='https://evil.invalid'),
        replace(_Route(), endpoint_path='/v1/complete'),
        replace(_Route(), fixed_parameters_sha256='f' * 64),
        replace(_Route(), provider_storage_disabled=True),
        replace(
            _Route(),
            provider_data_control='zero_data_retention',
            provider_data_control_attested=False,
        ),
        replace(_Route(), accepted_provider_model_ids=('some-other-model',)),
        replace(_Route(), provider_model_id='bad\nmodel', accepted_provider_model_ids=('bad\nmodel', _MODEL)),
    ],
)
def test_route_cannot_turn_adapter_into_a_general_http_proxy(route: _Route) -> None:
    transport = _FakeTransport()
    adapter, getter = _adapter(transport)

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL, route=route)

    assert getter.calls == 0
    assert transport.requests == []


def test_requested_alias_must_resolve_to_the_exact_pinned_model() -> None:
    alias = 'claude-test-latest'
    route = replace(
        _Route(),
        provider_model_id=alias,
        accepted_provider_model_ids=tuple(sorted((alias, _MODEL))),
    )
    transport = _FakeTransport(_response(payload=_payload(model=alias)))
    adapter, _ = _adapter(transport)

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL, route=route)

    assert json.loads(transport.requests[0][0].body)['model'] == alias


@pytest.mark.parametrize(
    'response',
    [
        _response(final_url='https://api.anthropic.com/v1/other'),
        _response(request_id=None),
        _response(extra_headers=(('request-id', 'second-id'),)),
        _response(content_type='text/plain'),
        _response(body=b'{"type":"message","type":"duplicate"}'),
        _response(payload=_payload(model='uncommitted-model')),
        _response(payload=_payload(input_tokens=True)),
        _response(payload=_payload(thinking_tokens=8)),
        _response(payload=_payload(content_type='tool_use', stop_reason='tool_use')),
        _response(payload=_payload(stop_reason='pause_turn')),
        _response(payload=_payload(stop_reason='stop_sequence', stop_sequence='STOP')),
        _response(extra_headers=(('Content-Encoding', 'gzip'),)),
    ],
)
def test_malformed_or_uncommitted_success_responses_are_protocol_failures(
    response: AnthropicHttpResponse,
) -> None:
    adapter, _ = _adapter(_FakeTransport(response))

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)


def test_response_and_header_bounds_are_enforced_against_injected_transport() -> None:
    adapter, _ = _adapter(
        _FakeTransport(_response(body=b'{}' * 100)),
        maximum_response_bytes=100,
    )
    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)

    adapter, _ = _adapter(
        _FakeTransport(_response(extra_headers=(('X-Padding', 'x' * 100),))),
        maximum_response_header_bytes=64,
    )
    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)


def test_injected_credential_reflection_is_never_returned_or_committed() -> None:
    escaped_secret = _SECRET.replace('-', r'\u002d')
    body = canonical_json_bytes(_payload(text='placeholder')).replace(
        b'placeholder',
        escaped_secret.encode(),
    )
    adapter, _ = _adapter(_FakeTransport(_response(body=body)))
    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)

    adapter, _ = _adapter(_FakeTransport(_response(extra_headers=(('X-Debug', _SECRET),))))
    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)


def test_credential_in_worker_message_is_rejected_before_dispatch() -> None:
    request = _request(
        messages=(
            AgenticModelMessage(role='system', content='Use frozen evidence.'),
            AgenticModelMessage(role='user', content=f'Credential echoed: {_SECRET}'),
        )
    )
    transport = _FakeTransport()
    adapter, _ = _adapter(transport)

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL, request=request)

    assert transport.requests == []


@pytest.mark.parametrize(
    'model_request',
    [
        _request(
            messages=(
                AgenticModelMessage(role='system', content='Use frozen evidence.'),
                AgenticModelMessage(role='tool', content='Untrusted tool result.'),
            )
        ),
        _request(
            messages=(
                AgenticModelMessage(role='system', content='Use frozen evidence.'),
                AgenticModelMessage(role='user', content='Question.'),
                AgenticModelMessage(role='assistant', content='Unsupported prefill.'),
            )
        ),
        _request(response_schema_sha256='c' * 64),
    ],
)
def test_unsupported_histories_and_schema_requests_are_rejected(
    model_request: AgenticModelRequest,
) -> None:
    transport = _FakeTransport()
    adapter, getter = _adapter(transport)

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL, request=model_request)

    assert getter.calls == 0
    assert transport.requests == []


def test_request_bound_and_timeout_are_fail_closed() -> None:
    transport = _FakeTransport()
    adapter, getter = _adapter(transport, maximum_request_bytes=10)
    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)
    assert getter.calls == 0
    assert transport.requests == []

    adapter, getter = _adapter(transport)
    with pytest.raises(ProviderCallFailure) as raised:
        adapter.generate(_request(), _Route(), timeout_seconds=float('nan'))
    assert raised.value.code == ProviderFailureCode.TIMEOUT
    assert getter.calls == 0
    assert transport.requests == []
