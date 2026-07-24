from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest
from vaxreplay.agentic.provider_adapter import ProviderCallFailure, ProviderFailureCode
from vaxreplay.agentic.providers.openai import (
    OPENAI_RESPONSES_ADAPTER_ID,
    OPENAI_RESPONSES_ENDPOINT_ORIGIN,
    OPENAI_RESPONSES_ENDPOINT_PATH,
    OPENAI_RESPONSES_FIXED_PARAMETERS_SHA256,
    OPENAI_RESPONSES_URL,
    OpenAIHttpRequest,
    OpenAIHttpResponse,
    OpenAIResponsesAdapter,
    OpenAITransportFailure,
    openai_responses_fixed_parameters,
)
from vaxreplay.bundle import canonical_json_bytes

_SECRET = 'vaxreplay-secret-canary-THIS_MUST_NEVER_LEAK_1234567890'
_MODEL = 'gpt-test-2025-01-02'
_SHA_A = 'a' * 64
_SHA_B = 'b' * 64


@dataclass(frozen=True)
class _Route:
    provider: str = 'openai'
    provider_model_id: str = _MODEL
    resolved_model_id: str = _MODEL
    accepted_provider_model_ids: tuple[str, ...] = (_MODEL,)
    endpoint_origin: str = OPENAI_RESPONSES_ENDPOINT_ORIGIN
    endpoint_path: str = OPENAI_RESPONSES_ENDPOINT_PATH
    fixed_parameters_sha256: str = OPENAI_RESPONSES_FIXED_PARAMETERS_SHA256
    provider_storage_disabled: bool = True
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
        response: OpenAIHttpResponse | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.response = response or _response()
        self.failure = failure
        self.requests: list[tuple[OpenAIHttpRequest, float]] = []

    def send(self, request: OpenAIHttpRequest, *, timeout_seconds: float) -> OpenAIHttpResponse:
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
    status: str = 'completed',
    incomplete_reason: str | None = None,
    content_type: str = 'output_text',
    input_tokens: object = 19,
    output_tokens: object = 7,
    total_tokens: object = 26,
    reasoning_tokens: object = 3,
) -> dict[str, object]:
    part = {'type': content_type, 'refusal' if content_type == 'refusal' else 'text': text}
    return {
        'id': 'resp_fixture_123',
        'object': 'response',
        'status': status,
        'incomplete_details': None if incomplete_reason is None else {'reason': incomplete_reason},
        'model': model,
        'output': [
            {'type': 'reasoning', 'id': 'rs_fixture', 'summary': []},
            {
                'type': 'message',
                'id': 'msg_fixture',
                'status': 'completed',
                'role': 'assistant',
                'content': [part],
            },
        ],
        'usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'output_tokens_details': {'reasoning_tokens': reasoning_tokens},
        },
    }


def _response(
    *,
    payload: object | None = None,
    body: bytes | None = None,
    status_code: int = 200,
    final_url: str = OPENAI_RESPONSES_URL,
    request_id: str | None = 'req_fixture_123',
    content_type: str = 'application/json; charset=utf-8',
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> OpenAIHttpResponse:
    actual_body = body if body is not None else canonical_json_bytes(payload if payload is not None else _payload())
    headers: list[tuple[str, str]] = [
        ('Content-Type', content_type),
        ('Content-Length', str(len(actual_body))),
    ]
    if request_id is not None:
        headers.append(('x-request-id', request_id))
    headers.extend(extra_headers)
    return OpenAIHttpResponse(
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
) -> tuple[OpenAIResponsesAdapter, _CredentialGetter]:
    actual_getter = getter or _CredentialGetter()
    return (
        OpenAIResponsesAdapter(
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
    adapter: OpenAIResponsesAdapter,
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


def test_success_uses_one_fixed_nonstreaming_request_and_strict_usage() -> None:
    transport = _FakeTransport()
    adapter, getter = _adapter(transport)
    request = _request()

    result = adapter.generate(request, _Route(), timeout_seconds=8.5)

    assert getter.calls == 1
    assert len(transport.requests) == 1
    prepared, timeout = transport.requests[0]
    assert prepared.url == OPENAI_RESPONSES_URL
    assert timeout == 8.5
    assert _SECRET not in repr(prepared)
    assert _SECRET.encode() not in prepared.body
    assert b'Candidate B ranks first.' not in repr(transport.response).encode()
    request_payload = json.loads(prepared.body)
    assert request_payload == {
        'background': False,
        'input': [
            {'content': 'Use only the frozen evidence.', 'role': 'system'},
            {'content': 'Rank candidates A and B.', 'role': 'user'},
        ],
        'max_output_tokens': 64,
        'model': _MODEL,
        'store': False,
        'stream': False,
        'tools': [],
        'truncation': 'disabled',
    }
    assert adapter.estimate_input_tokens(request, _Route()) == len(prepared.body)
    assert result.resolved_model_id == _MODEL
    assert result.provider_reported_model_id == _MODEL
    assert result.content == 'Candidate B ranks first.'
    assert result.stop_reason == 'completed'
    assert result.usage.input_tokens == 19
    assert result.usage.output_tokens == 7
    assert result.usage.reasoning_tokens == 3
    assert result.provider_request_id == 'req_fixture_123'
    assert result.http_status == 200
    assert _SECRET.encode() not in canonical_json_bytes(result)
    assert adapter.descriptor.adapter_id == OPENAI_RESPONSES_ADAPTER_ID
    assert adapter.descriptor.automatic_retries is False
    assert adapter.descriptor.credential_passed_in_request is False
    assert adapter.descriptor.input_estimate_is_conservative_upper_bound is True
    assert openai_responses_fixed_parameters() == {
        'background': False,
        'store': False,
        'stream': False,
        'tools': [],
        'truncation': 'disabled',
    }


@pytest.mark.parametrize(
    ('payload', 'stop_reason', 'content'),
    [
        (_payload(text='I cannot help with that.', content_type='refusal'), 'refusal', 'I cannot help with that.'),
        (
            _payload(text='Candidate B ranks', status='incomplete', incomplete_reason='max_output_tokens'),
            'max_output_tokens',
            'Candidate B ranks',
        ),
    ],
)
def test_terminal_refusal_and_max_token_responses(
    payload: dict[str, object],
    stop_reason: str,
    content: str,
) -> None:
    adapter, _ = _adapter(_FakeTransport(_response(payload=payload)))

    result = adapter.generate(_request(), _Route(), timeout_seconds=5)

    assert result.stop_reason == stop_reason
    assert result.content == content


@pytest.mark.parametrize(
    ('status_code', 'code'),
    [
        (302, ProviderFailureCode.PROTOCOL),
        (400, ProviderFailureCode.REJECTED),
        (408, ProviderFailureCode.TIMEOUT),
        (429, ProviderFailureCode.RATE_LIMIT),
        (500, ProviderFailureCode.INTERNAL),
        (504, ProviderFailureCode.TIMEOUT),
    ],
)
def test_http_failures_are_stable_codes_without_body_leakage(
    status_code: int,
    code: ProviderFailureCode,
) -> None:
    private_diagnostics = 'upstream-private-diagnostics-must-not-cross-boundary'
    response = _response(status_code=status_code, body=private_diagnostics.encode())
    transport = _FakeTransport(response)
    adapter, _ = _adapter(transport)

    failure = _assert_failure(adapter, code)

    assert len(transport.requests) == 1
    assert private_diagnostics not in str(failure)


@pytest.mark.parametrize('code', list(ProviderFailureCode))
def test_safe_typed_transport_failures_are_not_retried(code: ProviderFailureCode) -> None:
    transport = _FakeTransport(failure=OpenAITransportFailure(code))
    adapter, getter = _adapter(transport)

    _assert_failure(adapter, code)

    assert getter.calls == 1
    assert len(transport.requests) == 1


def test_arbitrary_transport_and_credential_exceptions_cannot_cross_boundary() -> None:
    transport = _FakeTransport(failure=RuntimeError(f'network failed with {_SECRET}'))
    adapter, _ = _adapter(transport)
    _assert_failure(adapter, ProviderFailureCode.INTERNAL)

    getter = _CredentialGetter(failure=RuntimeError(f'vault failed with {_SECRET}'))
    unused_transport = _FakeTransport()
    adapter, _ = _adapter(unused_transport, getter)
    _assert_failure(adapter, ProviderFailureCode.INTERNAL)
    assert unused_transport.requests == []


@pytest.mark.parametrize(
    'route',
    [
        replace(_Route(), provider='other'),
        replace(_Route(), endpoint_origin='https://evil.invalid'),
        replace(_Route(), endpoint_path='/v1/chat/completions'),
        replace(_Route(), fixed_parameters_sha256='f' * 64),
        replace(_Route(), provider_storage_disabled=False),
        replace(_Route(), accepted_provider_model_ids=('some-other-model',)),
    ],
)
def test_route_cannot_turn_adapter_into_a_general_http_proxy(route: _Route) -> None:
    transport = _FakeTransport()
    adapter, getter = _adapter(transport)

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL, route=route)

    assert getter.calls == 0
    assert transport.requests == []


def test_requested_alias_cannot_be_mislabeled_as_the_pinned_resolved_model() -> None:
    alias = 'gpt-test-latest'
    route = replace(
        _Route(),
        provider_model_id=alias,
        accepted_provider_model_ids=tuple(sorted((alias, _MODEL))),
    )
    transport = _FakeTransport(_response(payload=_payload(model=alias)))
    adapter, _ = _adapter(transport)

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL, route=route)

    assert len(transport.requests) == 1
    assert json.loads(transport.requests[0][0].body)['model'] == alias


@pytest.mark.parametrize(
    'response',
    [
        _response(final_url='https://api.openai.com/v1/other'),
        _response(request_id=None),
        _response(extra_headers=(('x-request-id', 'second-id'),)),
        _response(content_type='text/plain'),
        _response(body=b'{"object":"response","object":"duplicate"}'),
        _response(payload=_payload(model='uncommitted-model')),
        _response(payload=_payload(input_tokens=True)),
        _response(payload=_payload(total_tokens=999)),
        _response(
            payload={
                **_payload(),
                'output': [{'type': 'function_call', 'name': 'exfiltrate', 'arguments': '{}'}],
            }
        ),
        _response(payload=_payload(status='incomplete', incomplete_reason='content_filter')),
        _response(extra_headers=(('Content-Encoding', 'gzip'),)),
    ],
)
def test_malformed_or_uncommitted_success_responses_are_protocol_failures(response: OpenAIHttpResponse) -> None:
    adapter, _ = _adapter(_FakeTransport(response))

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)


def test_response_and_header_bounds_are_enforced_against_injected_transport() -> None:
    response = _response(body=b'{}' * 100)
    adapter, _ = _adapter(_FakeTransport(response), maximum_response_bytes=100)
    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)

    response = _response(extra_headers=(('X-Padding', 'x' * 100),))
    adapter, _ = _adapter(_FakeTransport(response), maximum_response_header_bytes=64)
    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)


def test_injected_credential_reflection_is_never_returned_or_committed() -> None:
    escaped_secret = _SECRET.replace('-', r'\u002d')
    body = canonical_json_bytes(_payload(text='placeholder')).replace(b'placeholder', escaped_secret.encode())
    response = _response(body=body)
    adapter, _ = _adapter(_FakeTransport(response))

    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)

    response = _response(extra_headers=(('X-Debug', _SECRET),))
    adapter, _ = _adapter(_FakeTransport(response))
    _assert_failure(adapter, ProviderFailureCode.PROTOCOL)


def test_credential_in_worker_message_is_rejected_before_dispatch() -> None:
    request = _request(
        messages=(
            AgenticModelMessage(role='system', content='Use frozen evidence.'),
            AgenticModelMessage(role='user', content=f'Accidentally echoed organizer credential: {_SECRET}'),
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
        _request(response_schema_sha256='c' * 64),
    ],
)
def test_unsupported_tool_and_unavailable_schema_requests_are_rejected(model_request: AgenticModelRequest) -> None:
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
