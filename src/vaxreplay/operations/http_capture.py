"""Security-conscious, exact-byte HTTPS capture primitives.

The collector in this module deliberately has a narrow job: perform one bounded
``GET`` and bind the bytes that arrived to a transport receipt.  It does not claim
that a source enumeration is complete, that a publisher timestamp is trustworthy,
or that the receipt has been independently witnessed.

The default transport uses the system TLS trust store, disables ambient proxy
configuration, and refuses redirects.  Tests and higher-level collectors can inject
the small :class:`HttpsTransport` protocol without opening a network connection.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import math
import multiprocessing
import os
import re
import socket
import ssl
import tempfile
import time
import urllib.error
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol, Self
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.case_schema import StrictModel

HTTPS_CAPTURE_REQUEST_SCHEMA_VERSION = 'vaxreplay.https-capture-request.v0.1'
HTTPS_CAPTURE_RECEIPT_SCHEMA_VERSION = 'vaxreplay.https-capture-receipt.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_HEADER_NAME_PATTERN = r'^[a-z0-9][a-z0-9-]*$'
_DNS_LABEL = re.compile(r'^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$')
_CHUNK_SIZE = 64 * 1024
_DEFAULT_USER_AGENT = 'VaxReplay-Archival-Capture/0.1'
_DEFAULT_DNS_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_DNS_ADDRESSES = 16
_MAX_RAW_RESPONSE_HEADER_FIELDS = 256
_MAX_RAW_RESPONSE_HEADER_CHARACTERS = 256 * 1024

# Request headers are intentionally an allowlist.  In particular, callers cannot
# smuggle credentials through uncommon ``x-*`` names or override wire encodings.
_SAFE_REQUEST_HEADERS = frozenset(
    {
        'accept',
        'accept-encoding',
        'host',
        'if-match',
        'if-modified-since',
        'if-none-match',
        'if-unmodified-since',
        'range',
        'range-unit',
        'user-agent',
    }
)

# Keep receipts useful while ensuring response cookies and authentication material
# never enter the operational ledger through this module.
_RECORDED_RESPONSE_HEADERS = frozenset(
    {
        'accept-ranges',
        'cache-control',
        'content-encoding',
        'content-language',
        'content-length',
        'content-range',
        'content-type',
        'date',
        'digest',
        'etag',
        'expires',
        'last-modified',
        'retry-after',
        'transfer-encoding',
        'vary',
    }
)


class HttpsCaptureError(ValueError):
    """Base class for an HTTPS capture that failed closed."""


class RequestPolicyError(HttpsCaptureError):
    """The requested or returned URL violates the capture policy."""


class DisallowedHostError(RequestPolicyError):
    """A transport returned a URL on a host other than the exact allowlisted host."""


class RedirectRejectedError(RequestPolicyError):
    """The server or transport attempted to redirect the archival request."""


class UnexpectedStatusError(HttpsCaptureError):
    """The response status was not one of the preregistered success statuses."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        final_url: str,
        response_headers: tuple[NormalizedResponseHeader, ...],
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.final_url = final_url
        self.response_headers = response_headers


class ResponseProtocolError(HttpsCaptureError):
    """Response metadata was malformed or ambiguous."""


class BodyTooLargeError(HttpsCaptureError):
    """The response exceeded the committed byte limit."""


class ContentLengthMismatchError(ResponseProtocolError):
    """The received entity length disagreed with ``Content-Length``."""


class TruncatedBodyError(ContentLengthMismatchError):
    """The transport ended before the advertised entity length arrived."""


class HttpsTransportError(HttpsCaptureError):
    """The HTTPS transport failed before a valid capture completed."""


class CaptureDeadlineExceededError(HttpsTransportError):
    """A committed monotonic request or plan deadline was exhausted."""


class DnsResolutionTimeoutError(HttpsTransportError):
    """DNS resolution did not complete within its independent bound."""


class DnsAddressLimitError(HttpsTransportError):
    """DNS returned more addresses than the committed collector bound."""


class SinkWriteError(HttpsCaptureError):
    """A caller-supplied sink did not accept the full byte stream."""


def _validate_header_value(value: str, *, field_name: str) -> str:
    if value != value.strip() or not value or len(value) > 16_384:
        raise ValueError(f'{field_name} must be nonempty, bounded, and have no outer whitespace')
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f'{field_name} cannot contain control characters')
    return value


class HttpRequestHeader(StrictModel):
    """One normalized, non-secret header supplied by the capture specification."""

    name: str = Field(pattern=_HEADER_NAME_PATTERN)
    value: str = Field(min_length=1, max_length=16_384)

    @field_validator('value')
    @classmethod
    def validate_value(cls, value: str) -> str:
        validated = _validate_header_value(value, field_name='request header value')
        try:
            validated.encode('ascii')
        except UnicodeEncodeError as error:
            raise ValueError('request header values must be ASCII') from error
        return validated

    @model_validator(mode='after')
    def validate_name(self) -> Self:
        if self.name not in _SAFE_REQUEST_HEADERS:
            raise ValueError(f'request header {self.name!r} is not allowed for unauthenticated archival capture')
        if self.name == 'accept-encoding' and self.value != 'identity':
            raise ValueError('accept-encoding, when implementation-created, must be identity')
        if self.name == 'host':
            _validate_dns_host(self.value)
        return self


class NormalizedResponseHeader(StrictModel):
    """Selected response header values after lowercase-name/OWS normalization."""

    name: str = Field(pattern=_HEADER_NAME_PATTERN)
    values: tuple[str, ...] = Field(min_length=1)

    @field_validator('values')
    @classmethod
    def validate_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _validate_header_value(value, field_name='response header value')
        return values

    @model_validator(mode='after')
    def validate_name(self) -> Self:
        if self.name not in _RECORDED_RESPONSE_HEADERS:
            raise ValueError(f'response header {self.name!r} is not part of the selected receipt schema')
        return self


def _validate_dns_host(value: str) -> str:
    if not value or len(value) > 253 or value != value.lower() or value.endswith('.'):
        raise ValueError('allowed_host must be a lowercase, unqualified-by-port DNS name')
    try:
        value.encode('ascii')
    except UnicodeEncodeError as error:
        raise ValueError('allowed_host must be ASCII; use its explicit IDNA form') from error
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise ValueError('allowed_host must be a DNS name, not an IP literal')
    labels = value.split('.')
    if len(labels) < 2 or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError('allowed_host must be a fully qualified DNS name without wildcards')
    return value


def _url_host(value: str) -> str:
    if len(value) > 8192 or '\\' in value or '#' in value:
        raise RequestPolicyError('capture URL is too long or contains a backslash/fragment')
    try:
        value.encode('ascii')
    except UnicodeEncodeError as error:
        raise RequestPolicyError('capture URL must use an explicit ASCII/IDNA representation') from error
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        raise RequestPolicyError('capture URL cannot contain whitespace or control characters')
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise RequestPolicyError('capture URL is malformed') from error
    if parsed.scheme != 'https':
        raise RequestPolicyError('capture URL must use https')
    if parsed.username is not None or parsed.password is not None or '@' in parsed.netloc:
        raise RequestPolicyError('capture URL cannot contain user information')
    if '?' in value and not parsed.query:
        raise RequestPolicyError('capture URL cannot contain an empty query delimiter')
    if port is not None:
        raise RequestPolicyError('capture URL cannot select a non-default or explicit port')
    if not parsed.hostname or parsed.netloc != parsed.hostname:
        raise RequestPolicyError('capture URL must contain one canonical lowercase DNS host')
    try:
        _validate_dns_host(parsed.hostname)
    except ValueError as error:
        raise RequestPolicyError('capture URL host must be a fully qualified DNS name') from error
    try:
        query_items = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=False)
    except ValueError as error:
        raise RequestPolicyError('capture URL query is malformed') from error
    credential_tokens = ('authorization', 'auth', 'bearer', 'credential', 'password', 'secret', 'token', 'apikey')
    for raw_name, _raw_value in query_items:
        normalized_name = ''.join(character for character in raw_name.lower() if character.isalnum())
        if any(token in normalized_name for token in credential_tokens):
            raise RequestPolicyError('capture URL cannot contain credential-like query parameters')
    return parsed.hostname


class HttpsCaptureRequest(StrictModel):
    """Preregistered policy for one bounded, unauthenticated HTTPS ``GET``."""

    schema_version: Literal['vaxreplay.https-capture-request.v0.1'] = HTTPS_CAPTURE_REQUEST_SCHEMA_VERSION
    method: Literal['GET'] = 'GET'
    url: str = Field(min_length=1, max_length=8192)
    allowed_host: str = Field(min_length=1, max_length=253)
    allowed_query_names: tuple[str, ...] = Field(default=(), max_length=64)
    request_headers: tuple[HttpRequestHeader, ...] = ()
    allowed_status_codes: tuple[int, ...] = (200,)
    max_body_bytes: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    redirect_policy: Literal['reject'] = 'reject'

    @field_validator('allowed_host')
    @classmethod
    def validate_allowed_host(cls, value: str) -> str:
        return _validate_dns_host(value)

    @field_validator('request_headers')
    @classmethod
    def validate_request_headers(cls, value: tuple[HttpRequestHeader, ...]) -> tuple[HttpRequestHeader, ...]:
        names = tuple(header.name for header in value)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError('request_headers must use unique lowercase names in sorted order')
        if {'accept-encoding', 'host'} & set(names):
            raise ValueError('callers cannot set accept-encoding or host; the collector controls them')
        return value

    @field_validator('allowed_query_names')
    @classmethod
    def validate_allowed_query_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('allowed_query_names must use sorted unique names')
        credential_names = {
            'accesskey',
            'authorization',
            'auth',
            'bearer',
            'code',
            'credential',
            'key',
            'password',
            'secret',
            'sig',
            'signature',
            'token',
            'xamzcredential',
            'xamzsignature',
        }
        for name in value:
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,127}', name):
                raise ValueError('allowed query names must be exact safe case-sensitive ASCII identifiers')
            normalized = ''.join(character for character in name.lower() if character.isalnum())
            if normalized in credential_names or any(
                token in normalized for token in ('apikey', 'password', 'secret', 'signature', 'token')
            ):
                raise ValueError('credential-like query names cannot be allowlisted')
        return value

    @field_validator('allowed_status_codes')
    @classmethod
    def validate_status_codes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('allowed_status_codes must be a nonempty sorted unique tuple')
        if any(isinstance(status, bool) or status < 200 or status > 299 for status in value):
            raise ValueError('allowed_status_codes may contain only HTTP success statuses')
        return value

    @model_validator(mode='after')
    def validate_url(self) -> Self:
        if _url_host(self.url) != self.allowed_host:
            raise ValueError('url host must exactly equal allowed_host')
        query_names = tuple(name for name, _value in parse_qsl(urlsplit(self.url).query, keep_blank_values=True))
        unexpected = sorted(set(query_names) - set(self.allowed_query_names))
        if unexpected:
            raise ValueError(f'URL query names are not explicitly allowlisted: {unexpected!r}')
        return self


class TlsPeerMetadata(StrictModel):
    """Best-effort metadata read from the authenticated TLS socket."""

    server_name: str | None = Field(default=None, min_length=1, max_length=253)
    peer_address: str | None = Field(default=None, min_length=1, max_length=128)
    peer_port: int | None = Field(default=None, ge=1, le=65_535)
    tls_version: str | None = Field(default=None, min_length=1, max_length=64)
    cipher_suite: str | None = Field(default=None, min_length=1, max_length=256)
    cipher_bits: int | None = Field(default=None, ge=0)
    certificate_der_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator('server_name', 'peer_address', 'tls_version', 'cipher_suite')
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_header_value(value, field_name='TLS metadata')
        return value

    @model_validator(mode='after')
    def require_metadata(self) -> Self:
        if all(
            value is None
            for value in (
                self.server_name,
                self.peer_address,
                self.peer_port,
                self.tls_version,
                self.cipher_suite,
                self.cipher_bits,
                self.certificate_der_sha256,
            )
        ):
            raise ValueError('TLS peer metadata cannot be empty')
        return self


class HttpsCaptureReceipt(StrictModel):
    """Receipt binding exact response bytes to selected transport metadata."""

    schema_version: Literal['vaxreplay.https-capture-receipt.v0.1'] = HTTPS_CAPTURE_RECEIPT_SCHEMA_VERSION
    method: Literal['GET'] = 'GET'
    requested_url: str = Field(min_length=1, max_length=8192)
    final_url: str = Field(min_length=1, max_length=8192)
    request_headers: tuple[HttpRequestHeader, ...] = Field(min_length=1)
    status_code: int = Field(ge=200, le=299)
    response_headers: tuple[NormalizedResponseHeader, ...]
    body_sha256: str = Field(pattern=_SHA256_PATTERN)
    body_byte_count: int = Field(ge=0)
    started_at: datetime
    completed_at: datetime
    tls_peer: TlsPeerMetadata | None = None

    @field_validator('started_at', 'completed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('capture timestamps must include a UTC offset')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        request_names = tuple(header.name for header in self.request_headers)
        if request_names != tuple(sorted(request_names)) or len(request_names) != len(set(request_names)):
            raise ValueError('receipt request headers must use sorted unique names')
        accept_encoding = next(
            (header.value for header in self.request_headers if header.name == 'accept-encoding'),
            None,
        )
        if accept_encoding != 'identity':
            raise ValueError('receipt must bind accept-encoding: identity')
        receipt_host = next((header.value for header in self.request_headers if header.name == 'host'), None)
        if receipt_host != _url_host(self.requested_url):
            raise ValueError('receipt Host header must bind the requested URL host')
        response_names = tuple(header.name for header in self.response_headers)
        if response_names != tuple(sorted(response_names)) or len(response_names) != len(set(response_names)):
            raise ValueError('receipt response headers must use sorted unique names')
        selected_field_count = sum(len(header.values) for header in self.response_headers)
        if selected_field_count > _MAX_RAW_RESPONSE_HEADER_FIELDS:
            raise ValueError('receipt response header field count exceeds the implementation bound')
        selected_character_count = sum(
            len(header.name) + len(value) for header in self.response_headers for value in header.values
        )
        if selected_character_count > _MAX_RAW_RESPONSE_HEADER_CHARACTERS:
            raise ValueError('receipt response header characters exceed the implementation bound')
        if self.final_url != self.requested_url:
            raise ValueError('redirect-rejecting receipts require final_url to equal requested_url')
        try:
            _url_host(self.requested_url)
        except RequestPolicyError as error:
            raise ValueError('receipt URLs must be canonical HTTPS URLs') from error
        _validate_entity_encoding(self.response_headers)
        expected_length = _content_length(self.response_headers)
        if expected_length is not None and expected_length != self.body_byte_count:
            raise ValueError('receipt body_byte_count must equal Content-Length')
        if self.completed_at < self.started_at:
            raise ValueError('completed_at cannot precede started_at')
        return self


@dataclass(frozen=True)
class PreparedHttpsRequest:
    """Transport-facing request with collector-controlled headers included."""

    method: Literal['GET']
    url: str
    headers: tuple[HttpRequestHeader, ...]
    timeout_seconds: float


class HttpsTransportResponse(Protocol):
    """Minimal streaming response required from an injected transport."""

    @property
    def status_code(self) -> int: ...

    @property
    def final_url(self) -> str: ...

    @property
    def response_headers(self) -> Iterable[tuple[str, str]]: ...

    def read(self, size: int) -> bytes: ...

    def tls_peer_metadata(self) -> TlsPeerMetadata | None: ...

    def close(self) -> None: ...


class HttpsTransport(Protocol):
    """Injectable HTTPS transport used by :func:`capture_https`."""

    def open(self, request: PreparedHttpsRequest) -> HttpsTransportResponse: ...


type SocketAddress = tuple[object, ...]
type AddressInfo = tuple[int, int, int, str, SocketAddress]
type DnsWorker = Callable[[Connection, str, int, int], None]


class HttpsDnsResolver(Protocol):
    """One bounded DNS operation used by the default HTTPS transport."""

    def resolve(
        self,
        host: str,
        port: int,
        *,
        timeout_seconds: float,
        max_addresses: int,
    ) -> tuple[AddressInfo, ...]: ...


def _system_getaddrinfo_worker(
    connection: Connection,
    host: str,
    port: int,
    max_addresses: int,
) -> None:
    """Resolve in a killable child because libc ``getaddrinfo`` has no deadline API."""

    try:
        answers = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        if len(answers) > max_addresses:
            connection.send(('too_many', len(answers)))
        else:
            connection.send(('ok', tuple(answers)))
    except BaseException as error:
        try:
            connection.send(('error', type(error).__name__))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class SubprocessHttpsDnsResolver:
    """System DNS with a parent-enforced wall deadline and kill/reap boundary."""

    def __init__(
        self,
        *,
        start_method: Literal['spawn', 'forkserver'] = 'spawn',
        worker: DnsWorker = _system_getaddrinfo_worker,
    ) -> None:
        self._context: Any = multiprocessing.get_context(start_method)
        self._worker = worker

    def resolve(
        self,
        host: str,
        port: int,
        *,
        timeout_seconds: float,
        max_addresses: int,
    ) -> tuple[AddressInfo, ...]:
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError('DNS timeout_seconds must be finite and positive')
        if not isinstance(max_addresses, int) or isinstance(max_addresses, bool) or not 1 <= max_addresses <= 64:
            raise ValueError('DNS max_addresses must be between 1 and 64')

        receiver, sender = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=self._worker,
            args=(sender, host, port, max_addresses),
            daemon=True,
        )
        payload: object | None = None
        timed_out = False
        cleanup_failed = False
        try:
            process.start()
        except BaseException as error:
            receiver.close()
            sender.close()
            process.close()
            raise HttpsTransportError(f'DNS resolver child could not start for {host!r}') from error
        sender.close()
        try:
            if receiver.poll(timeout_seconds):
                try:
                    payload = receiver.recv()
                except EOFError:
                    payload = None
            else:
                timed_out = True
        finally:
            receiver.close()
            if process.is_alive():
                process.kill()
            # ``join`` is bounded so a defective platform cannot turn cleanup into
            # another unbounded operation. The process has already been killed.
            process.join(timeout=1.0)
            if process.is_alive():
                cleanup_failed = True
            else:
                process.close()

        if cleanup_failed:
            raise HttpsTransportError(f'DNS resolver child could not be reaped for {host!r}')
        if timed_out:
            raise DnsResolutionTimeoutError(f'DNS resolution timed out for {host!r}')
        if not isinstance(payload, tuple) or len(payload) != 2 or not isinstance(payload[0], str):
            raise HttpsTransportError(f'DNS resolver failed for {host!r}')
        status, value = payload
        if status == 'too_many':
            raise DnsAddressLimitError(f'DNS for {host!r} exceeded max_addresses={max_addresses}')
        if status != 'ok' or not isinstance(value, tuple):
            raise HttpsTransportError(f'DNS resolution failed for {host!r}')
        if len(value) > max_addresses:
            raise DnsAddressLimitError(f'DNS for {host!r} exceeded max_addresses={max_addresses}')
        return value


def _tls_metadata_from_socket(socket_object: ssl.SSLSocket, *, server_name: str) -> TlsPeerMetadata | None:
    peer_address = tls_version = cipher_suite = certificate_sha256 = None
    peer_port = cipher_bits = None
    try:
        peer = socket_object.getpeername()
        if isinstance(peer, tuple) and len(peer) >= 2:
            peer_address = str(peer[0])
            if isinstance(peer[1], int):
                peer_port = peer[1]
    except (AttributeError, OSError, ValueError):
        pass
    try:
        version_value = socket_object.version()
        if isinstance(version_value, str) and version_value:
            tls_version = version_value
    except (AttributeError, OSError, ValueError):
        pass
    try:
        cipher = socket_object.cipher()
        if isinstance(cipher, tuple) and cipher:
            if isinstance(cipher[0], str) and cipher[0]:
                cipher_suite = cipher[0]
            if len(cipher) >= 3 and isinstance(cipher[2], int):
                cipher_bits = cipher[2]
    except (AttributeError, OSError, ValueError):
        pass
    try:
        certificate = socket_object.getpeercert(binary_form=True)
        if isinstance(certificate, bytes) and certificate:
            certificate_sha256 = hashlib.sha256(certificate).hexdigest()
    except (AttributeError, OSError, ValueError, ssl.SSLError):
        pass

    values = (server_name, peer_address, peer_port, tls_version, cipher_suite, cipher_bits, certificate_sha256)
    if all(value is None for value in values):
        return None
    try:
        return TlsPeerMetadata(
            server_name=server_name,
            peer_address=peer_address,
            peer_port=peer_port,
            tls_version=tls_version,
            cipher_suite=cipher_suite,
            cipher_bits=cipher_bits,
            certificate_der_sha256=certificate_sha256,
        )
    except ValueError:
        # TLS details are supplementary.  A platform-specific, unrepresentable
        # value must not make an otherwise exact byte capture platform-dependent.
        return None


class _HttpClientResponse:
    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPSConnection,
        *,
        requested_url: str,
        tls_peer: TlsPeerMetadata | None,
        operation_timeout: Callable[[], float] | None = None,
    ) -> None:
        self._response = response
        self._connection = connection
        self._status_code = response.status
        self._final_url = requested_url
        self._response_headers = tuple((str(name), str(value)) for name, value in response.getheaders())
        self._tls_peer = tls_peer
        self._operation_timeout = operation_timeout

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def final_url(self) -> str:
        return self._final_url

    @property
    def response_headers(self) -> tuple[tuple[str, str], ...]:
        return self._response_headers

    def read(self, size: int) -> bytes:
        if self._operation_timeout is not None:
            timeout_seconds = self._operation_timeout()
            if self._connection.sock is None:
                raise HttpsTransportError('HTTPS connection lost its authenticated socket')
            self._connection.sock.settimeout(timeout_seconds)
        result = self._response.read(size)
        if self._operation_timeout is not None:
            self._operation_timeout()
        return result

    def tls_peer_metadata(self) -> TlsPeerMetadata | None:
        return self._tls_peer

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()


def _is_public_unicast(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_reserved
        and not getattr(address, 'is_site_local', False)
    )


def _public_endpoints(
    host: str,
    *,
    resolver: HttpsDnsResolver,
    timeout_seconds: float,
    max_addresses: int,
) -> tuple[tuple[int, int, int, SocketAddress], ...]:
    """Resolve once and reject an oversized, malformed, or non-public answer set."""

    try:
        answers = resolver.resolve(
            host,
            443,
            timeout_seconds=timeout_seconds,
            max_addresses=max_addresses,
        )
    except HttpsCaptureError:
        raise
    except (OSError, ValueError) as error:
        raise HttpsTransportError(f'DNS resolution failed for {host!r}') from error
    if not isinstance(answers, tuple):
        raise RequestPolicyError('DNS resolver returned a non-tuple answer set')
    if len(answers) > max_addresses:
        raise DnsAddressLimitError(f'DNS for {host!r} exceeded max_addresses={max_addresses}')
    endpoints: list[tuple[int, int, int, tuple[object, ...]]] = []
    seen: set[tuple[int, tuple[object, ...]]] = set()
    for answer in answers:
        if not isinstance(answer, tuple) or len(answer) != 5:
            raise RequestPolicyError('DNS returned a malformed address-info record')
        family, socket_type, protocol, _canonical_name, socket_address = answer
        if (
            family not in {socket.AF_INET, socket.AF_INET6}
            or socket_type != socket.SOCK_STREAM
            or protocol != socket.IPPROTO_TCP
            or not isinstance(socket_address, tuple)
            or len(socket_address) < 2
            or socket_address[1] != 443
        ):
            raise RequestPolicyError('DNS returned an unsupported address family')
        address = socket_address[0]
        if not isinstance(address, str):
            raise RequestPolicyError('DNS returned a malformed address')
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError as error:
            raise RequestPolicyError('DNS returned a non-IP endpoint') from error
        if not _is_public_unicast(parsed_address):
            raise RequestPolicyError(f'DNS for {host!r} returned non-public endpoint {address!r}')
        key = (family, socket_address)
        if key not in seen:
            endpoints.append((family, socket_type, protocol, socket_address))
            seen.add(key)
    if not endpoints:
        raise HttpsTransportError(f'DNS resolution returned no usable endpoints for {host!r}')
    return tuple(endpoints)


class UrllibHttpsTransport:
    """Default transport: DNS-pinned public egress, system TLS, and no redirects.

    The legacy-looking name describes the original implementation boundary; the
    implementation uses :mod:`http.client` so it can bind the connection to an
    already-vetted IP address and eliminate a second DNS lookup.
    """

    def __init__(
        self,
        *,
        resolver: HttpsDnsResolver | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        deadline_monotonic: float | None = None,
        dns_timeout_seconds: float = _DEFAULT_DNS_TIMEOUT_SECONDS,
        dns_resolution_attempts: Literal[1] = 1,
        max_dns_addresses: int = _DEFAULT_MAX_DNS_ADDRESSES,
    ) -> None:
        if not callable(monotonic):
            raise TypeError('monotonic must be callable')
        if deadline_monotonic is not None and (
            not isinstance(deadline_monotonic, (int, float))
            or isinstance(deadline_monotonic, bool)
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError('deadline_monotonic must be finite when supplied')
        if (
            not isinstance(dns_timeout_seconds, (int, float))
            or isinstance(dns_timeout_seconds, bool)
            or not math.isfinite(dns_timeout_seconds)
            or dns_timeout_seconds <= 0
            or dns_timeout_seconds > 300
        ):
            raise ValueError('dns_timeout_seconds must be finite and between 0 and 300')
        if dns_resolution_attempts != 1:
            raise ValueError('dns_resolution_attempts must equal one')
        if (
            not isinstance(max_dns_addresses, int)
            or isinstance(max_dns_addresses, bool)
            or not 1 <= max_dns_addresses <= 64
        ):
            raise ValueError('max_dns_addresses must be between 1 and 64')
        self._resolver = resolver if resolver is not None else SubprocessHttpsDnsResolver()
        self._monotonic = monotonic
        self._deadline_monotonic = float(deadline_monotonic) if deadline_monotonic is not None else None
        self._dns_timeout_seconds = float(dns_timeout_seconds)
        self._dns_resolution_attempts = dns_resolution_attempts
        self._max_dns_addresses = max_dns_addresses
        self._last_monotonic: float | None = None

    def _remaining(self) -> float | None:
        if self._deadline_monotonic is None:
            return None
        value = self._monotonic()
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise HttpsTransportError('monotonic clock returned a non-finite numeric value')
        current = float(value)
        if self._last_monotonic is not None and current < self._last_monotonic:
            raise HttpsTransportError('monotonic clock moved backwards')
        self._last_monotonic = current
        remaining = self._deadline_monotonic - current
        if remaining <= 0:
            raise CaptureDeadlineExceededError('HTTPS request exhausted its monotonic deadline')
        return remaining

    def _operation_timeout(self, configured_timeout: float) -> float:
        remaining = self._remaining()
        return configured_timeout if remaining is None else min(configured_timeout, remaining)

    def open(self, request: PreparedHttpsRequest) -> HttpsTransportResponse:
        parsed = urlsplit(request.url)
        host = parsed.hostname
        if host is None:
            raise RequestPolicyError('prepared HTTPS request has no host')
        remaining = self._remaining()
        dns_timeout = self._dns_timeout_seconds if remaining is None else min(self._dns_timeout_seconds, remaining)
        # The v0.2 collector commits this as exactly one. Keeping the value in the
        # transport makes the no-retry boundary explicit and independently testable.
        if self._dns_resolution_attempts != 1:
            raise HttpsTransportError('DNS resolution attempt policy changed after construction')
        endpoints = _public_endpoints(
            host,
            resolver=self._resolver,
            timeout_seconds=dns_timeout,
            max_addresses=self._max_dns_addresses,
        )
        self._remaining()
        context = ssl.create_default_context()
        target = parsed.path or '/'
        if parsed.query:
            target = f'{target}?{parsed.query}'
        last_error: BaseException | None = None
        for family, socket_type, protocol, socket_address in endpoints:
            raw_socket: socket.socket | None = None
            tls_socket: ssl.SSLSocket | None = None
            connection: http.client.HTTPSConnection | None = None
            try:
                raw_socket = socket.socket(family, socket_type, protocol)
                raw_socket.settimeout(self._operation_timeout(request.timeout_seconds))
                raw_socket.connect(socket_address)
                raw_socket.settimeout(self._operation_timeout(request.timeout_seconds))
                tls_socket = context.wrap_socket(raw_socket, server_hostname=host)
                raw_socket = None  # ownership moved to tls_socket
                tls_socket.settimeout(self._operation_timeout(request.timeout_seconds))
                peer = tls_socket.getpeername()
                peer_address = ipaddress.ip_address(str(peer[0]))
                if not _is_public_unicast(peer_address) or str(peer_address) != str(
                    ipaddress.ip_address(str(socket_address[0]))
                ):
                    raise RequestPolicyError('connected TLS peer does not match the vetted public endpoint')
                tls_peer = _tls_metadata_from_socket(tls_socket, server_name=host)

                connection = http.client.HTTPSConnection(
                    host,
                    port=443,
                    timeout=self._operation_timeout(request.timeout_seconds),
                    context=context,
                )
                connection.sock = tls_socket
                tls_socket = None  # ownership moved to connection
                if connection.sock is None:
                    raise HttpsTransportError('HTTPS connection lost its authenticated socket')
                connection.sock.settimeout(self._operation_timeout(request.timeout_seconds))
                connection.request(
                    request.method,
                    target,
                    headers={header.name: header.value for header in request.headers},
                )
                if connection.sock is None:
                    raise HttpsTransportError('HTTPS connection lost its authenticated socket')
                connection.sock.settimeout(self._operation_timeout(request.timeout_seconds))
                response = connection.getresponse()
                self._remaining()
                captured_response = _HttpClientResponse(
                    response,
                    connection,
                    requested_url=request.url,
                    tls_peer=tls_peer,
                    operation_timeout=(
                        (lambda: self._operation_timeout(request.timeout_seconds))
                        if self._deadline_monotonic is not None
                        else None
                    ),
                )
                connection = None  # ownership moved to captured_response
                return captured_response
            except RequestPolicyError:
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException) as error:
                last_error = error
            finally:
                if connection is not None:
                    connection.close()
                if tls_socket is not None:
                    tls_socket.close()
                if raw_socket is not None:
                    raw_socket.close()
        raise HttpsTransportError(f'all vetted HTTPS endpoints failed for {host!r}') from last_error


def prepared_request_headers(request: HttpsCaptureRequest) -> tuple[HttpRequestHeader, ...]:
    """Return the exact canonical request headers committed by a capture request.

    This is public so trusted replay verification can independently reconstruct the
    wire-level request commitment instead of trusting a stored receipt.
    """

    by_name = {header.name: header for header in request.request_headers}
    by_name['accept-encoding'] = HttpRequestHeader(name='accept-encoding', value='identity')
    by_name['host'] = HttpRequestHeader(name='host', value=request.allowed_host)
    by_name.setdefault('user-agent', HttpRequestHeader(name='user-agent', value=_DEFAULT_USER_AGENT))
    return tuple(by_name[name] for name in sorted(by_name))


def _normalize_response_headers(
    pairs: Iterable[tuple[str, str]],
) -> tuple[NormalizedResponseHeader, ...]:
    collected: dict[str, list[str]] = {}
    raw_character_count = 0
    try:
        iterator = iter(pairs)
    except TypeError as error:
        raise ResponseProtocolError('response headers are not iterable name/value pairs') from error
    for ordinal, pair in enumerate(iterator, start=1):
        if ordinal > _MAX_RAW_RESPONSE_HEADER_FIELDS:
            raise ResponseProtocolError('response header field count exceeds the implementation bound')
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ResponseProtocolError('response headers must be two-item tuples')
        raw_name, raw_value = pair
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise ResponseProtocolError('response header names and values must be strings')
        raw_character_count += len(raw_name) + len(raw_value)
        if raw_character_count > _MAX_RAW_RESPONSE_HEADER_CHARACTERS:
            raise ResponseProtocolError('response header characters exceed the implementation bound')
        if raw_name != raw_name.strip():
            raise ResponseProtocolError('response header names cannot contain outer whitespace')
        name = raw_name.lower()
        if name not in _RECORDED_RESPONSE_HEADERS:
            continue
        if not re.fullmatch(_HEADER_NAME_PATTERN, name):
            raise ResponseProtocolError('selected response header name is malformed')
        value = raw_value.strip(' \t')
        try:
            _validate_header_value(value, field_name='response header value')
        except ValueError as error:
            raise ResponseProtocolError(str(error)) from error
        collected.setdefault(name, []).append(value)
    return tuple(NormalizedResponseHeader(name=name, values=tuple(collected[name])) for name in sorted(collected))


def _content_length(headers: tuple[NormalizedResponseHeader, ...]) -> int | None:
    header = next((item for item in headers if item.name == 'content-length'), None)
    if header is None:
        return None
    if len(header.values) != 1:
        raise ResponseProtocolError('multiple Content-Length fields are ambiguous')
    value = header.values[0]
    if not value.isascii() or not value.isdigit():
        raise ResponseProtocolError('Content-Length must be one nonnegative ASCII decimal integer')
    if len(value) > 20:
        raise ResponseProtocolError('Content-Length exceeds the supported integer range')
    return int(value)


def _validate_entity_encoding(headers: tuple[NormalizedResponseHeader, ...]) -> None:
    content_encoding = next((item for item in headers if item.name == 'content-encoding'), None)
    if content_encoding is not None and (
        len(content_encoding.values) != 1 or content_encoding.values[0].lower() != 'identity'
    ):
        raise ResponseProtocolError('server ignored Accept-Encoding: identity')
    content_length = next((item for item in headers if item.name == 'content-length'), None)
    transfer_encoding = next((item for item in headers if item.name == 'transfer-encoding'), None)
    if content_length is not None and transfer_encoding is not None:
        raise ResponseProtocolError('response cannot contain both Content-Length and Transfer-Encoding')
    if transfer_encoding is not None:
        if len(transfer_encoding.values) != 1 or transfer_encoding.values[0].lower() != 'chunked':
            raise ResponseProtocolError('Transfer-Encoding must be exactly one chunked field')


def _utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def _clock_value(clock: Callable[[], datetime], field_name: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise HttpsCaptureError(f'{field_name} clock value must be an offset-aware datetime')
    return value.astimezone(timezone.utc)


def _monotonic_clock_value(clock: Callable[[], float], field_name: str) -> float:
    try:
        value = clock()
    except Exception as error:
        raise HttpsTransportError(f'{field_name} monotonic clock failed') from error
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise HttpsTransportError(f'{field_name} monotonic clock must return a finite numeric value')
    return float(value)


def _write_all(sink: BinaryIO, chunk: bytes) -> None:
    view = memoryview(chunk)
    while view:
        try:
            written = sink.write(view)
        except (OSError, ValueError) as error:
            raise SinkWriteError('capture sink write failed') from error
        if not isinstance(written, int) or isinstance(written, bool) or written <= 0 or written > len(view):
            raise SinkWriteError('capture sink did not report a valid positive byte count')
        view = view[written:]


def _validate_returned_url(url: str, request: HttpsCaptureRequest) -> None:
    try:
        host = _url_host(url)
    except RequestPolicyError as error:
        raise RequestPolicyError('transport returned an invalid final URL') from error
    if host != request.allowed_host:
        raise DisallowedHostError(
            f'transport returned host {host!r}; expected exact allowed host {request.allowed_host!r}'
        )
    if url != request.url:
        raise RedirectRejectedError('transport changed the URL while redirects are forbidden')


def capture_https(
    request: HttpsCaptureRequest,
    sink: BinaryIO,
    *,
    transport: HttpsTransport | None = None,
    clock: Callable[[], datetime] = _utc_clock,
    monotonic: Callable[[], float] = time.monotonic,
) -> HttpsCaptureReceipt:
    """Stream one exact response entity into ``sink`` and return its receipt.

    ``sink`` remains owned by the caller and may contain a prefix if capture fails.
    Use :func:`capture_https_to_tempfile` when all-or-cleanup file semantics are
    preferable.  Network, status, protocol, size, and truncation failures produce no
    successful receipt.
    """

    if not hasattr(sink, 'write'):
        raise SinkWriteError('capture sink must provide a binary write method')
    headers = prepared_request_headers(request)
    prepared = PreparedHttpsRequest(
        method='GET',
        url=request.url,
        headers=headers,
        timeout_seconds=request.timeout_seconds,
    )
    started_at = _clock_value(clock, 'started_at')
    default_deadline = None
    if transport is None:
        monotonic_started = _monotonic_clock_value(monotonic, 'HTTPS request')
        default_deadline = monotonic_started + request.timeout_seconds
        active_transport: HttpsTransport = UrllibHttpsTransport(
            monotonic=monotonic,
            deadline_monotonic=default_deadline,
            dns_timeout_seconds=min(_DEFAULT_DNS_TIMEOUT_SECONDS, request.timeout_seconds),
        )
    else:
        active_transport = transport
    try:
        response = active_transport.open(prepared)
    except (OSError, urllib.error.URLError, ssl.SSLError, http.client.HTTPException) as error:
        raise HttpsTransportError('HTTPS request could not be opened') from error

    try:
        status_code = response.status_code
        if not isinstance(status_code, int) or isinstance(status_code, bool) or not 100 <= status_code <= 599:
            raise ResponseProtocolError('transport returned an invalid HTTP status code')
        final_url = response.final_url
        if not isinstance(final_url, str):
            raise ResponseProtocolError('transport returned a non-string final URL')
        _validate_returned_url(final_url, request)
        if 300 <= status_code <= 399:
            raise RedirectRejectedError(f'HTTP redirect status {status_code} is forbidden')

        response_headers = _normalize_response_headers(response.response_headers)
        if status_code not in request.allowed_status_codes:
            raise UnexpectedStatusError(
                f'HTTP status {status_code} is not in allowed_status_codes={request.allowed_status_codes!r}',
                status_code=status_code,
                final_url=final_url,
                response_headers=response_headers,
            )

        _validate_entity_encoding(response_headers)
        expected_length = _content_length(response_headers)
        if expected_length is not None and expected_length > request.max_body_bytes:
            raise BodyTooLargeError(f'Content-Length {expected_length} exceeds max_body_bytes={request.max_body_bytes}')

        digest = hashlib.sha256()
        byte_count = 0
        while True:
            try:
                chunk = response.read(_CHUNK_SIZE)
            except http.client.IncompleteRead as error:
                raise TruncatedBodyError('response ended before its advertised Content-Length') from error
            except (OSError, ssl.SSLError, TimeoutError) as error:
                if expected_length is not None and byte_count < expected_length:
                    raise TruncatedBodyError('transport failed before Content-Length bytes arrived') from error
                raise HttpsTransportError('HTTPS response body read failed') from error
            if not isinstance(chunk, bytes):
                raise ResponseProtocolError('transport read must return bytes')
            if not chunk:
                break
            if byte_count + len(chunk) > request.max_body_bytes:
                raise BodyTooLargeError(f'response exceeds max_body_bytes={request.max_body_bytes}')
            _write_all(sink, chunk)
            if default_deadline is not None and _monotonic_clock_value(monotonic, 'HTTPS request') >= default_deadline:
                raise CaptureDeadlineExceededError('HTTPS request exhausted its monotonic deadline')
            digest.update(chunk)
            byte_count += len(chunk)

        if expected_length is not None and byte_count < expected_length:
            raise TruncatedBodyError(f'received {byte_count} bytes but Content-Length advertised {expected_length}')
        if expected_length is not None and byte_count > expected_length:
            raise ContentLengthMismatchError(
                f'received {byte_count} bytes but Content-Length advertised {expected_length}'
            )
        tls_peer = response.tls_peer_metadata()
        if tls_peer is not None and not isinstance(tls_peer, TlsPeerMetadata):
            raise ResponseProtocolError('transport TLS metadata must be TlsPeerMetadata or None')
        completed_at = _clock_value(clock, 'completed_at')
        return HttpsCaptureReceipt(
            requested_url=request.url,
            final_url=final_url,
            request_headers=headers,
            status_code=status_code,
            response_headers=response_headers,
            body_sha256=digest.hexdigest(),
            body_byte_count=byte_count,
            started_at=started_at,
            completed_at=completed_at,
            tls_peer=tls_peer,
        )
    finally:
        try:
            response.close()
        except (OSError, ValueError):
            # Closing occurs after all security-relevant body reads.  A close error
            # must not mask a more specific capture failure or invalidate exact bytes.
            pass


@dataclass(frozen=True)
class TemporaryHttpsCapture:
    """Successful temporary body file and the receipt that binds its exact bytes."""

    path: Path
    receipt: HttpsCaptureReceipt

    def delete(self) -> None:
        """Remove the temporary body if it still exists."""

        self.path.unlink(missing_ok=True)


def capture_https_to_tempfile(
    request: HttpsCaptureRequest,
    *,
    directory: str | os.PathLike[str] | None = None,
    transport: HttpsTransport | None = None,
    clock: Callable[[], datetime] = _utc_clock,
    monotonic: Callable[[], float] = time.monotonic,
) -> TemporaryHttpsCapture:
    """Capture to a newly created file, deleting it on every failed attempt."""

    default_deadline = None
    if transport is None:
        default_deadline = _monotonic_clock_value(monotonic, 'HTTPS tempfile request') + request.timeout_seconds
    descriptor, raw_path = tempfile.mkstemp(prefix='vaxreplay-https-', suffix='.body', dir=directory)
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, 'wb') as sink:
            receipt = capture_https(
                request,
                sink,
                transport=transport,
                clock=clock,
                monotonic=monotonic,
            )
            sink.flush()
            os.fsync(sink.fileno())
            if (
                default_deadline is not None
                and _monotonic_clock_value(monotonic, 'HTTPS tempfile request') >= default_deadline
            ):
                raise CaptureDeadlineExceededError('HTTPS tempfile request exhausted its monotonic deadline')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return TemporaryHttpsCapture(path=path, receipt=receipt)
