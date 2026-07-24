"""Hardened synchronous OpenAI Responses API adapter.

The adapter is deliberately not a general HTTP client.  The trusted organizer route must commit
to the one OpenAI origin, path, and fixed request-parameter set below.  Credentials are injected
inside the trusted gateway process, are excluded from the provider request commitment, and have a
redacted representation.  Provider bodies and exception text never cross the adapter's safe error
boundary.

``estimate_input_tokens`` intentionally returns the byte length of the complete canonical request
body.  That is a coarse, deliberately high preflight estimate rather than provider-authoritative
metering; the gateway still enforces the usage reported in the response.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from vaxreplay.agentic.gateway import AgenticGatewayUsage, AgenticModelRequest
from vaxreplay.agentic.provider_adapter import (
    ProviderAdapterDescriptor,
    ProviderCallFailure,
    ProviderCallResult,
    ProviderFailureCode,
    ProviderModelRoute,
)
from vaxreplay.bundle import canonical_json_bytes

OPENAI_RESPONSES_ADAPTER_ID = 'openai-responses'
OPENAI_RESPONSES_ENDPOINT_ORIGIN = 'https://api.openai.com'
OPENAI_RESPONSES_ENDPOINT_PATH = '/v1/responses'
OPENAI_RESPONSES_URL = OPENAI_RESPONSES_ENDPOINT_ORIGIN + OPENAI_RESPONSES_ENDPOINT_PATH

_FIXED_PARAMETERS: dict[str, object] = {
    'background': False,
    'store': False,
    'stream': False,
    'tools': [],
    'truncation': 'disabled',
}
OPENAI_RESPONSES_FIXED_PARAMETERS_SHA256 = hashlib.sha256(canonical_json_bytes(_FIXED_PARAMETERS)).hexdigest()

_DEFAULT_MAXIMUM_REQUEST_BYTES = 4 * 1024 * 1024
_DEFAULT_MAXIMUM_RESPONSE_BYTES = 16 * 1024 * 1024
_DEFAULT_MAXIMUM_RESPONSE_HEADER_BYTES = 64 * 1024
_MAXIMUM_API_KEY_BYTES = 16 * 1024
_MAXIMUM_PROVIDER_ID_BYTES = 500
_MAXIMUM_OUTPUT_CHARS = 2_000_000
_USER_AGENT = 'vaxreplay-provider-gateway/0.2'


class _OpenAICredential:
    """A transient credential whose string and repr forms are always redacted."""

    __slots__ = ('__value',)

    def __init__(self, value: str) -> None:
        self.__value = value

    def __repr__(self) -> str:
        return '<OpenAI credential: redacted>'

    def __str__(self) -> str:
        return '<redacted>'

    def _bearer_header(self) -> str:
        return f'Bearer {self.__value}'


@dataclass(frozen=True, slots=True, repr=False)
class OpenAIHttpRequest:
    """One prepared request.  Its representation cannot expose the credential."""

    url: str
    body: bytes
    credential: _OpenAICredential = field(repr=False)

    def __repr__(self) -> str:
        return f'OpenAIHttpRequest(url={self.url!r}, body_bytes={len(self.body)}, credential=<redacted>)'


@dataclass(frozen=True, slots=True, repr=False)
class OpenAIHttpResponse:
    """Raw bounded response returned by a transport; validation belongs to the adapter."""

    status_code: int
    final_url: str
    headers: tuple[tuple[str, str], ...]
    body: bytes

    def __repr__(self) -> str:
        return (
            'OpenAIHttpResponse('
            f'status_code={self.status_code!r}, final_url={self.final_url!r}, '
            f'header_count={len(self.headers)}, body_bytes={len(self.body)})'
        )


class OpenAIHttpTransport(Protocol):
    """Single-attempt synchronous transport.  Implementations must not retry."""

    def send(self, request: OpenAIHttpRequest, *, timeout_seconds: float) -> OpenAIHttpResponse: ...


class OpenAITransportFailure(RuntimeError):
    """A provider-body-free transport failure safe to map at the adapter boundary."""

    def __init__(self, code: ProviderFailureCode):
        super().__init__(code.value)
        self.code = code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


class UrllibOpenAIHttpTransport:
    """One-shot HTTPS transport with redirects and ambient proxies disabled."""

    def __init__(self, *, maximum_response_bytes: int = _DEFAULT_MAXIMUM_RESPONSE_BYTES) -> None:
        if not _is_positive_int(maximum_response_bytes):
            raise ValueError('maximum_response_bytes must be a positive integer')
        self._maximum_response_bytes = maximum_response_bytes
        tls_context = ssl.create_default_context()
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        tls_context.check_hostname = True
        tls_context.verify_mode = ssl.CERT_REQUIRED
        if hasattr(tls_context, 'keylog_filename'):
            setattr(tls_context, 'keylog_filename', None)
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
            urllib.request.HTTPSHandler(context=tls_context),
        )

    def send(self, request: OpenAIHttpRequest, *, timeout_seconds: float) -> OpenAIHttpResponse:
        if request.url != OPENAI_RESPONSES_URL or not _valid_timeout(timeout_seconds):
            raise OpenAITransportFailure(ProviderFailureCode.PROTOCOL)
        http_request = urllib.request.Request(
            request.url,
            data=request.body,
            headers={
                'Accept': 'application/json',
                'Accept-Encoding': 'identity',
                'Authorization': request.credential._bearer_header(),
                'Content-Type': 'application/json',
                'User-Agent': _USER_AGENT,
            },
            method='POST',
        )
        response: Any
        failure: ProviderFailureCode | None = None
        try:
            response = self._opener.open(http_request, timeout=timeout_seconds)
        except urllib.error.HTTPError as error:
            response = error
        except (TimeoutError, socket.timeout):
            failure = ProviderFailureCode.TIMEOUT
            response = None
        except urllib.error.URLError as error:
            failure = (
                ProviderFailureCode.TIMEOUT
                if isinstance(error.reason, (TimeoutError, socket.timeout))
                else ProviderFailureCode.INTERNAL
            )
            response = None
        except (OSError, ssl.SSLError):
            failure = ProviderFailureCode.INTERNAL
            response = None
        except Exception:
            failure = ProviderFailureCode.INTERNAL
            response = None
        if failure is not None:
            raise OpenAITransportFailure(failure)
        assert response is not None

        try:
            body = response.read(self._maximum_response_bytes + 1)
            status_code = response.getcode()
            final_url = response.geturl()
            headers = tuple(response.headers.raw_items())
        except (TimeoutError, socket.timeout):
            failure = ProviderFailureCode.TIMEOUT
        except Exception:
            failure = ProviderFailureCode.INTERNAL
        finally:
            response.close()
        if failure is not None:
            raise OpenAITransportFailure(failure)
        if (
            not isinstance(body, bytes)
            or len(body) > self._maximum_response_bytes
            or not isinstance(status_code, int)
            or isinstance(status_code, bool)
            or not isinstance(final_url, str)
        ):
            raise OpenAITransportFailure(ProviderFailureCode.PROTOCOL)
        return OpenAIHttpResponse(
            status_code=status_code,
            final_url=final_url,
            headers=headers,
            body=body,
        )


class _ResponseProtocolViolation(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _ParsedResponse:
    model: str
    content: str
    stop_reason: Literal['completed', 'max_output_tokens', 'refusal']
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None
    request_id: str


class OpenAIResponsesAdapter:
    """Strict synchronous/non-streaming OpenAI Responses API provider adapter."""

    def __init__(
        self,
        *,
        credential_getter: Callable[[], str],
        executable_sha256: str,
        config_sha256: str,
        adapter_version: str,
        transport: OpenAIHttpTransport | None = None,
        maximum_request_bytes: int = _DEFAULT_MAXIMUM_REQUEST_BYTES,
        maximum_response_bytes: int = _DEFAULT_MAXIMUM_RESPONSE_BYTES,
        maximum_response_header_bytes: int = _DEFAULT_MAXIMUM_RESPONSE_HEADER_BYTES,
    ) -> None:
        if not callable(credential_getter):
            raise TypeError('credential_getter must be callable')
        for name, value in (
            ('maximum_request_bytes', maximum_request_bytes),
            ('maximum_response_bytes', maximum_response_bytes),
            ('maximum_response_header_bytes', maximum_response_header_bytes),
        ):
            if not _is_positive_int(value):
                raise ValueError(f'{name} must be a positive integer')
        self._credential_getter = credential_getter
        self._maximum_request_bytes = maximum_request_bytes
        self._maximum_response_bytes = maximum_response_bytes
        self._maximum_response_header_bytes = maximum_response_header_bytes
        self._transport = (
            transport
            if transport is not None
            else UrllibOpenAIHttpTransport(maximum_response_bytes=maximum_response_bytes)
        )
        self._descriptor = ProviderAdapterDescriptor(
            adapter_id=OPENAI_RESPONSES_ADAPTER_ID,
            adapter_version=adapter_version,
            executable_sha256=executable_sha256,
            config_sha256=config_sha256,
            provider='openai',
        )

    @property
    def descriptor(self) -> ProviderAdapterDescriptor:
        return self._descriptor

    def estimate_input_tokens(self, request: AgenticModelRequest, route: ProviderModelRoute) -> int:
        """Return a documented conservative byte-count preflight estimate, not billed usage."""

        body: bytes | None = None
        try:
            body = self._request_body(request, route)
        except (AttributeError, TypeError, ValueError):
            pass
        if body is None:
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)
        return len(body)

    def generate(
        self,
        request: AgenticModelRequest,
        route: ProviderModelRoute,
        *,
        timeout_seconds: float,
    ) -> ProviderCallResult:
        if not _valid_timeout(timeout_seconds):
            raise ProviderCallFailure(ProviderFailureCode.TIMEOUT)
        body: bytes | None = None
        try:
            body = self._request_body(request, route)
        except (AttributeError, TypeError, ValueError):
            pass
        if body is None:
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)
        if len(body) > self._maximum_request_bytes:
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)

        credential_value: str | None = None
        credential_failed = False
        try:
            credential_value = self._credential_getter()
        except Exception:
            credential_failed = True
        if credential_failed or not _valid_credential(credential_value):
            raise ProviderCallFailure(ProviderFailureCode.INTERNAL)
        assert credential_value is not None
        credential_bytes = credential_value.encode('ascii')
        if credential_bytes in body or any(credential_value in message.content for message in request.messages):
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)
        credential = _OpenAICredential(credential_value)

        started_at = datetime.now(UTC)
        transport_failure: ProviderFailureCode | None = None
        response: OpenAIHttpResponse | None = None
        try:
            response = self._transport.send(
                OpenAIHttpRequest(url=OPENAI_RESPONSES_URL, body=body, credential=credential),
                timeout_seconds=timeout_seconds,
            )
        except OpenAITransportFailure as error:
            transport_failure = error.code
        except Exception:
            transport_failure = ProviderFailureCode.INTERNAL
        finished_at = datetime.now(UTC)
        if transport_failure is not None or response is None:
            raise ProviderCallFailure(transport_failure or ProviderFailureCode.INTERNAL)

        envelope_failed = False
        try:
            _validate_response_envelope(
                response,
                credential=credential_value,
                maximum_body_bytes=self._maximum_response_bytes,
                maximum_header_bytes=self._maximum_response_header_bytes,
            )
        except Exception:
            envelope_failed = True
        if envelope_failed:
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)

        status_failure = _failure_for_status(response.status_code)
        if status_failure is not None:
            raise ProviderCallFailure(status_failure)

        parsed: _ParsedResponse | None = None
        parse_failed = False
        try:
            parsed = _parse_response(
                response,
                route=route,
                credential=credential_value,
                maximum_body_bytes=self._maximum_response_bytes,
                maximum_header_bytes=self._maximum_response_header_bytes,
            )
        except Exception:
            parse_failed = True
        if parse_failed or parsed is None:
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)
        if (
            credential_value in parsed.content
            or credential_value in parsed.model
            or credential_value in parsed.request_id
        ):
            raise ProviderCallFailure(ProviderFailureCode.PROTOCOL)

        return ProviderCallResult(
            resolved_model_id=parsed.model,
            provider_reported_model_id=parsed.model,
            content=parsed.content,
            stop_reason=parsed.stop_reason,
            usage=AgenticGatewayUsage(
                input_tokens=parsed.input_tokens,
                output_tokens=parsed.output_tokens,
                reasoning_tokens=parsed.reasoning_tokens,
            ),
            provider_request_sha256=hashlib.sha256(body).hexdigest(),
            provider_request_bytes=len(body),
            provider_response_sha256=hashlib.sha256(response.body).hexdigest(),
            provider_response_bytes=len(response.body),
            provider_request_id=parsed.request_id,
            http_status=response.status_code,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _request_body(self, request: AgenticModelRequest, route: ProviderModelRoute) -> bytes:
        _validate_route(route)
        if request.response_schema_sha256 is not None or any(message.role == 'tool' for message in request.messages):
            raise ValueError('unsupported request shape')
        return canonical_json_bytes(
            {
                **_FIXED_PARAMETERS,
                'input': [
                    {
                        'content': message.content,
                        'role': message.role,
                    }
                    for message in request.messages
                ],
                'max_output_tokens': request.max_output_tokens,
                'model': route.provider_model_id,
            }
        )


def _validate_route(route: ProviderModelRoute) -> None:
    if (
        route.provider != 'openai'
        or route.endpoint_origin != OPENAI_RESPONSES_ENDPOINT_ORIGIN
        or route.endpoint_path != OPENAI_RESPONSES_ENDPOINT_PATH
        or route.fixed_parameters_sha256 != OPENAI_RESPONSES_FIXED_PARAMETERS_SHA256
        or route.provider_storage_disabled is not True
        or route.provider_model_id not in route.accepted_provider_model_ids
        or route.resolved_model_id not in route.accepted_provider_model_ids
    ):
        raise ValueError('route does not match the fixed OpenAI Responses adapter')


def _valid_timeout(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_credential(value: object) -> bool:
    if not isinstance(value, str) or not value.isascii():
        return False
    encoded = value.encode('ascii')
    return 16 <= len(encoded) <= _MAXIMUM_API_KEY_BYTES and all(0x21 <= byte <= 0x7E for byte in encoded)


def _failure_for_status(value: object) -> ProviderFailureCode | None:
    if not isinstance(value, int) or isinstance(value, bool) or not 100 <= value <= 599:
        return ProviderFailureCode.PROTOCOL
    if value == 200:
        return None
    if value in {408, 504}:
        return ProviderFailureCode.TIMEOUT
    if value == 429:
        return ProviderFailureCode.RATE_LIMIT
    if 300 <= value <= 399:
        return ProviderFailureCode.PROTOCOL
    if 400 <= value <= 499:
        return ProviderFailureCode.REJECTED
    if 500 <= value <= 599:
        return ProviderFailureCode.INTERNAL
    return ProviderFailureCode.PROTOCOL


def _parse_response(
    response: OpenAIHttpResponse,
    *,
    route: ProviderModelRoute,
    credential: str,
    maximum_body_bytes: int,
    maximum_header_bytes: int,
) -> _ParsedResponse:
    _validate_response_envelope(
        response,
        credential=credential,
        maximum_body_bytes=maximum_body_bytes,
        maximum_header_bytes=maximum_header_bytes,
    )
    if not response.body:
        raise _ResponseProtocolViolation('successful provider body is empty')
    headers = _validated_headers(response.headers, maximum_header_bytes=maximum_header_bytes, credential=credential)
    content_type = _single_header(headers, 'content-type')
    if content_type is None or content_type.split(';', 1)[0].strip().lower() != 'application/json':
        raise _ResponseProtocolViolation('provider response content type is not JSON')
    content_length = _single_header(headers, 'content-length')
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdigit() or len(content_length) > 20:
            raise _ResponseProtocolViolation('provider content length is malformed')
        if int(content_length) != len(response.body):
            raise _ResponseProtocolViolation('provider content length does not match its body')
    content_encoding = _single_header(headers, 'content-encoding')
    if content_encoding is not None and content_encoding.lower() != 'identity':
        raise _ResponseProtocolViolation('encoded provider bodies are forbidden')
    request_id = _single_header(headers, 'x-request-id')
    if request_id is None:
        raise _ResponseProtocolViolation('provider request ID is missing')
    _validate_provider_identifier(request_id, 'provider request ID')

    try:
        decoded = response.body.decode('utf-8')
        payload = json.loads(decoded, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError):
        raise _ResponseProtocolViolation('provider response is not strict JSON') from None
    if not isinstance(payload, dict) or payload.get('object') != 'response':
        raise _ResponseProtocolViolation('provider response object is malformed')
    if _contains_secret(payload, credential):
        raise _ResponseProtocolViolation('provider JSON contained the injected credential')
    _validate_provider_identifier(payload.get('id'), 'provider response ID')
    model = payload.get('model')
    _validate_provider_identifier(model, 'provider model ID')
    assert isinstance(model, str)
    if model != route.resolved_model_id:
        raise _ResponseProtocolViolation('provider did not attest the pinned resolved model ID')

    input_tokens, output_tokens, reasoning_tokens = _parse_usage(payload.get('usage'))
    content, has_refusal = _parse_output(payload.get('output'))
    status = payload.get('status')
    incomplete_details = payload.get('incomplete_details')
    if status == 'completed':
        if incomplete_details is not None:
            raise _ResponseProtocolViolation('completed provider response declared incomplete details')
        stop_reason: Literal['completed', 'max_output_tokens', 'refusal'] = 'refusal' if has_refusal else 'completed'
    elif status == 'incomplete':
        if not isinstance(incomplete_details, dict) or incomplete_details.get('reason') != 'max_output_tokens':
            raise _ResponseProtocolViolation('unsupported incomplete provider response')
        if has_refusal:
            raise _ResponseProtocolViolation('incomplete provider response cannot also be a refusal')
        stop_reason = 'max_output_tokens'
    else:
        raise _ResponseProtocolViolation('provider response is not in a terminal supported state')

    return _ParsedResponse(
        model=model,
        content=content,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        request_id=request_id,
    )


def _validate_response_envelope(
    response: OpenAIHttpResponse,
    *,
    credential: str,
    maximum_body_bytes: int,
    maximum_header_bytes: int,
) -> None:
    if response.final_url != OPENAI_RESPONSES_URL:
        raise _ResponseProtocolViolation('transport changed the fixed provider URL')
    if not isinstance(response.body, bytes) or len(response.body) > maximum_body_bytes:
        raise _ResponseProtocolViolation('provider body is outside the configured bound')
    if credential.encode('ascii') in response.body:
        raise _ResponseProtocolViolation('provider response contained the injected credential')
    _validated_headers(response.headers, maximum_header_bytes=maximum_header_bytes, credential=credential)


def _validated_headers(
    raw_headers: object,
    *,
    maximum_header_bytes: int,
    credential: str,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw_headers, tuple):
        raise _ResponseProtocolViolation('provider response headers are malformed')
    total_bytes = 0
    collected: dict[str, list[str]] = {}
    for pair in raw_headers:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise _ResponseProtocolViolation('provider response header is malformed')
        name, value = pair
        if not isinstance(name, str) or not isinstance(value, str) or not name or name != name.strip():
            raise _ResponseProtocolViolation('provider response header is malformed')
        if not name.isascii() or not all(character.isalnum() or character in "!#$%&'*+-.^_`|~" for character in name):
            raise _ResponseProtocolViolation('provider response header name is malformed')
        if any((ord(character) < 0x20 and character != '\t') or ord(character) == 0x7F for character in value):
            raise _ResponseProtocolViolation('provider response header value is malformed')
        total_bytes += len(name.encode('ascii')) + len(value.encode('utf-8')) + 4
        if total_bytes > maximum_header_bytes or credential in value:
            raise _ResponseProtocolViolation('provider response headers exceed policy')
        collected.setdefault(name.lower(), []).append(value.strip(' \t'))
    return {name: tuple(values) for name, values in collected.items()}


def _single_header(headers: dict[str, tuple[str, ...]], name: str) -> str | None:
    values = headers.get(name)
    if values is None:
        return None
    if len(values) != 1:
        raise _ResponseProtocolViolation(f'provider response has ambiguous {name}')
    return values[0]


def _validate_provider_identifier(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or value != value.strip()
        or len(value.encode('utf-8')) > _MAXIMUM_PROVIDER_ID_BYTES
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise _ResponseProtocolViolation(f'{field_name} is malformed')


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _ResponseProtocolViolation('provider JSON contains a duplicate key')
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    del value
    raise _ResponseProtocolViolation('provider JSON contains a non-finite number')


def _contains_secret(value: object, secret: str) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, list):
        return any(_contains_secret(item, secret) for item in value)
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and secret in key) or _contains_secret(item, secret) for key, item in value.items()
        )
    return False


def _parse_usage(value: object) -> tuple[int, int, int | None]:
    if not isinstance(value, dict):
        raise _ResponseProtocolViolation('provider usage is missing')
    input_tokens = _nonnegative_int(value.get('input_tokens'), 'input_tokens')
    output_tokens = _nonnegative_int(value.get('output_tokens'), 'output_tokens')
    total_tokens = _nonnegative_int(value.get('total_tokens'), 'total_tokens')
    if total_tokens != input_tokens + output_tokens:
        raise _ResponseProtocolViolation('provider total usage is inconsistent')
    details = value.get('output_tokens_details')
    if details is None:
        reasoning_tokens = None
    elif isinstance(details, dict):
        reasoning_tokens = _nonnegative_int(details.get('reasoning_tokens'), 'reasoning_tokens')
    else:
        raise _ResponseProtocolViolation('provider output token details are malformed')
    return input_tokens, output_tokens, reasoning_tokens


def _nonnegative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _ResponseProtocolViolation(f'provider {field_name} must be a nonnegative integer')
    return value


def _parse_output(value: object) -> tuple[str, bool]:
    if not isinstance(value, list) or not value:
        raise _ResponseProtocolViolation('provider output is empty or malformed')
    text_fragments: list[str] = []
    refusal_fragments: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            raise _ResponseProtocolViolation('provider output item is malformed')
        item_type = item.get('type')
        if item_type == 'reasoning':
            continue
        if item_type != 'message' or item.get('role') != 'assistant':
            raise _ResponseProtocolViolation('provider returned a forbidden non-message output item')
        content = item.get('content')
        if not isinstance(content, list) or not content:
            raise _ResponseProtocolViolation('provider message content is empty or malformed')
        for part in content:
            if not isinstance(part, dict):
                raise _ResponseProtocolViolation('provider content item is malformed')
            part_type = part.get('type')
            if part_type == 'output_text':
                text = part.get('text')
                if not isinstance(text, str) or not text:
                    raise _ResponseProtocolViolation('provider output text is empty or malformed')
                text_fragments.append(text)
            elif part_type == 'refusal':
                refusal = part.get('refusal')
                if not isinstance(refusal, str) or not refusal:
                    raise _ResponseProtocolViolation('provider refusal text is empty or malformed')
                refusal_fragments.append(refusal)
            else:
                raise _ResponseProtocolViolation('provider returned a forbidden content item')
    if bool(text_fragments) == bool(refusal_fragments):
        raise _ResponseProtocolViolation('provider must return exactly one of output text or refusal text')
    fragments = refusal_fragments or text_fragments
    content = ''.join(fragments)
    if not content or len(content) > _MAXIMUM_OUTPUT_CHARS:
        raise _ResponseProtocolViolation('provider output text is outside the configured bound')
    return content, bool(refusal_fragments)


def openai_responses_fixed_parameters() -> dict[str, object]:
    """Return a defensive copy for route-registry construction and independent verification."""

    return {**_FIXED_PARAMETERS, 'tools': []}
