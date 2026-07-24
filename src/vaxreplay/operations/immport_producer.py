"""One-shot, credential-bearing ImmPort HTTPS producer.

This module is the deliberately small secret-handling half of authenticated ImmPort
collection.  A supervisor passes a scoped API key on an already-open file descriptor;
the key is never accepted through argv, the environment, a JSON field, or a path.  The
producer performs the exact nine requests in a precommitted
:class:`ImmportAuthenticatedCollectionPlan`, emits only structurally sanitized receipts,
and overwrites its mutable credential buffer before returning.

The default transport disables proxies and redirects by speaking HTTPS directly to a
DNS-vetted public endpoint with the system trust store.  It sends the bearer value as a
separate socket segment so no complete Authorization header is constructed as a Python
``str``.  Python and the TLS implementation may still make internal copies, which is why
the command is intentionally one-shot and belongs in an isolated process with a hard
supervisor deadline and restricted egress.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import ipaddress
import os
import socket
import ssl
import stat
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol, Self, cast
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.immport_capture import (
    MAX_IMMPORT_ARTIFACT_BODY_BYTES,
    MAX_IMMPORT_CAPTURE_BODY_BYTES,
    ImmportAuthenticatedArtifactSpec,
    ImmportAuthenticatedCollectionPlan,
    ImmportCapturedExchange,
)
from vaxreplay.operations.policy import IMMPORT_AUTHENTICATED_COLLECTOR_ID
from vaxreplay.operations.schema import AttemptLease, AttemptState, aware_utc
from vaxreplay.runner._process import run_bounded_process
from vaxreplay.sources.immport import ImmportSanitizedCaptureReceipt, ImmportTlsPeerBinding

IMMPORT_PRODUCER_REQUEST_SCHEMA_VERSION = 'vaxreplay.immport-producer-request.v0.1'
IMMPORT_PRODUCER_RESPONSE_SCHEMA_VERSION = 'vaxreplay.immport-producer-response.v0.1'

IMMPORT_CREDENTIAL_FD = 3
MAX_IMMPORT_CREDENTIAL_BYTES = 4096
MAX_IMMPORT_PRODUCER_REQUEST_BYTES = 256 * 1024
MAX_IMMPORT_PRODUCER_RESPONSE_BYTES = 96 * 1024 * 1024

_HOST = 'www.immport.org'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_CHUNK_SIZE = 64 * 1024
_MAX_HEADER_BYTES = 64 * 1024
_MAX_BASE64_BODY_CHARS = ((MAX_IMMPORT_ARTIFACT_BODY_BYTES + 2) // 3) * 4
_CONTENT_TYPES = frozenset(
    {
        'application/json',
        'application/json;charset=UTF-8',
        'application/json;charset=utf-8',
        'application/json; charset=UTF-8',
        'application/json; charset=utf-8',
    }
)
ImmportContentType = Literal[
    'application/json',
    'application/json;charset=UTF-8',
    'application/json;charset=utf-8',
    'application/json; charset=UTF-8',
    'application/json; charset=utf-8',
]
ImmportTlsVersion = Literal['TLSv1.2', 'TLSv1.3']
_REQUEST_PREFIX = (
    b' HTTP/1.1\r\n'
    b'Host: www.immport.org\r\n'
    b'Accept: application/json\r\n'
    b'Accept-Encoding: identity\r\n'
    b'User-Agent: VaxReplay-ImmPort-Producer/0.1\r\n'
)
_AUTHORIZATION_PREFIX = b'Authorization: Bearer '
_REQUEST_SUFFIX = b'Connection: close\r\n\r\n'


class ImmportProducerError(RuntimeError):
    """The isolated producer failed without exposing secret-bearing diagnostics."""


class ImmportProducerRequest(StrictModel):
    """Canonical, public input passed to the credential-bearing process on stdin."""

    schema_version: Literal['vaxreplay.immport-producer-request.v0.1'] = IMMPORT_PRODUCER_REQUEST_SCHEMA_VERSION
    plan: ImmportAuthenticatedCollectionPlan
    attempt: AttemptLease
    collector_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_attempt(self) -> Self:
        if self.attempt.state is not AttemptState.STARTED:
            raise ValueError('ImmPort producer requires a live started attempt')
        if (self.attempt.lease_expires_at - self.attempt.started_at).total_seconds() < (
            self.plan.panel_deadline_seconds
        ):
            raise ValueError('ImmPort attempt lease cannot contain the panel deadline')
        return self


class ImmportProducerExchange(StrictModel):
    """One body and its credential-free receipt on the producer wire protocol."""

    artifact_id: str = Field(pattern=r'^[a-z][a-z0-9._-]{0,109}$')
    body_base64: str = Field(min_length=0, max_length=_MAX_BASE64_BODY_CHARS)
    receipt: ImmportSanitizedCaptureReceipt

    @field_validator('body_base64')
    @classmethod
    def validate_body_base64(cls, value: str) -> str:
        try:
            raw = value.encode('ascii')
            decoded = base64.b64decode(raw, validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as error:
            raise ValueError('ImmPort producer body must be canonical base64') from error
        if base64.b64encode(decoded) != raw:
            raise ValueError('ImmPort producer body must use canonical padded base64')
        if len(decoded) > MAX_IMMPORT_ARTIFACT_BODY_BYTES:
            raise ValueError('ImmPort producer body exceeds its absolute byte bound')
        return value

    @model_validator(mode='after')
    def bind_receipt(self) -> Self:
        body = self.body_bytes()
        if self.receipt.body_byte_count != len(body) or self.receipt.body_sha256 != hashlib.sha256(body).hexdigest():
            raise ValueError('ImmPort producer receipt does not bind its body')
        return self

    def body_bytes(self) -> bytes:
        return base64.b64decode(self.body_base64.encode('ascii'), validate=True)


class ImmportProducerResponse(StrictModel):
    """Canonical successful response; failures use no attacker-controlled detail."""

    schema_version: Literal['vaxreplay.immport-producer-response.v0.1'] = IMMPORT_PRODUCER_RESPONSE_SCHEMA_VERSION
    exchanges: tuple[ImmportProducerExchange, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        identifiers = tuple(item.artifact_id for item in self.exchanges)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError('ImmPort producer exchanges must use canonical unique order')
        if sum(item.receipt.body_byte_count for item in self.exchanges) > (MAX_IMMPORT_CAPTURE_BODY_BYTES):
            raise ValueError('ImmPort producer response exceeds its aggregate body bound')
        return self


@dataclass(frozen=True)
class PreparedImmportRequest:
    """Exact public request facts presented to the injectable transport."""

    url: str
    authorization_applied: bool
    timeout_seconds: float


class ImmportTransportResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def final_url(self) -> str: ...

    @property
    def response_headers(self) -> Iterable[tuple[str, str]]: ...

    def read(self, size: int, *, timeout_seconds: float) -> bytes: ...

    def tls_peer_binding(self) -> ImmportTlsPeerBinding | None: ...

    def close(self) -> None: ...


class ImmportCredentialedTransport(Protocol):
    """Transport TCB; implementations must never retain or log ``credential``."""

    def open(
        self,
        request: PreparedImmportRequest,
        credential: memoryview,
    ) -> ImmportTransportResponse: ...


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


def _public_endpoints() -> tuple[tuple[int, int, int, tuple[object, ...]], ...]:
    try:
        answers = socket.getaddrinfo(
            _HOST,
            443,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as error:
        raise ImmportProducerError('ImmPort producer transport failed') from error
    endpoints: list[tuple[int, int, int, tuple[object, ...]]] = []
    seen: set[tuple[int, tuple[object, ...]]] = set()
    for family, socket_type, protocol, _canonical_name, socket_address in answers:
        if family not in {socket.AF_INET, socket.AF_INET6} or not isinstance(socket_address, tuple):
            raise ImmportProducerError('ImmPort producer transport failed')
        raw_address = socket_address[0]
        if not isinstance(raw_address, str):
            raise ImmportProducerError('ImmPort producer transport failed')
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as error:
            raise ImmportProducerError('ImmPort producer transport failed') from error
        if not _is_public_unicast(address):
            raise ImmportProducerError('ImmPort producer transport failed')
        key = (family, socket_address)
        if key not in seen:
            endpoints.append((family, socket_type, protocol, socket_address))
            seen.add(key)
    if not endpoints:
        raise ImmportProducerError('ImmPort producer transport failed')
    return tuple(endpoints)


class _SocketImmportResponse:
    def __init__(
        self,
        response: http.client.HTTPResponse,
        tls_socket: ssl.SSLSocket,
        *,
        requested_url: str,
        tls_peer: ImmportTlsPeerBinding,
    ) -> None:
        self._response = response
        self._socket = tls_socket
        self._requested_url = requested_url
        self._headers = tuple((str(name), str(value)) for name, value in response.getheaders())
        self._tls_peer = tls_peer

    @property
    def status_code(self) -> int:
        return self._response.status

    @property
    def final_url(self) -> str:
        return self._requested_url

    @property
    def response_headers(self) -> tuple[tuple[str, str], ...]:
        return self._headers

    def read(self, size: int, *, timeout_seconds: float) -> bytes:
        self._socket.settimeout(timeout_seconds)
        return self._response.read(size)

    def tls_peer_binding(self) -> ImmportTlsPeerBinding:
        return self._tls_peer

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._socket.close()


class DirectImmportHttpsTransport:
    """System-CA HTTPS with one DNS answer set, IP pinning, and no proxy state."""

    def open(
        self,
        request: PreparedImmportRequest,
        credential: memoryview,
    ) -> ImmportTransportResponse:
        parsed = urlsplit(request.url)
        if (
            parsed.scheme != 'https'
            or parsed.hostname != _HOST
            or parsed.netloc != _HOST
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ImmportProducerError('ImmPort producer request policy failed')
        target = parsed.path or '/'
        if parsed.query:
            target = f'{target}?{parsed.query}'
        try:
            target_bytes = target.encode('ascii')
        except UnicodeEncodeError as error:
            raise ImmportProducerError('ImmPort producer request policy failed') from error
        if not target_bytes.startswith(b'/') or any(value <= 32 or value == 127 for value in target_bytes):
            raise ImmportProducerError('ImmPort producer request policy failed')

        context = ssl.create_default_context()
        last_failure = False
        for family, socket_type, protocol, socket_address in _public_endpoints():
            raw_socket: socket.socket | None = None
            tls_socket: ssl.SSLSocket | None = None
            try:
                raw_socket = socket.socket(family, socket_type, protocol)
                raw_socket.settimeout(request.timeout_seconds)
                raw_socket.connect(socket_address)
                tls_socket = context.wrap_socket(raw_socket, server_hostname=_HOST)
                raw_socket = None
                peer = tls_socket.getpeername()
                connected = ipaddress.ip_address(str(peer[0]))
                expected = ipaddress.ip_address(str(socket_address[0]))
                if not _is_public_unicast(connected) or connected != expected:
                    raise ImmportProducerError('ImmPort producer transport failed')
                tls_version = tls_socket.version()
                certificate = tls_socket.getpeercert(binary_form=True)
                if tls_version not in {'TLSv1.2', 'TLSv1.3'} or not certificate:
                    raise ImmportProducerError('ImmPort producer TLS binding failed')
                tls_peer = ImmportTlsPeerBinding(
                    tls_version=cast(ImmportTlsVersion, tls_version),
                    certificate_der_sha256=hashlib.sha256(certificate).hexdigest(),
                )

                tls_socket.sendall(b'GET ' + target_bytes + _REQUEST_PREFIX)
                if request.authorization_applied:
                    tls_socket.sendall(_AUTHORIZATION_PREFIX)
                    tls_socket.sendall(credential)
                    tls_socket.sendall(b'\r\n')
                tls_socket.sendall(_REQUEST_SUFFIX)
                response = http.client.HTTPResponse(tls_socket, method='GET')
                response.begin()
                captured = _SocketImmportResponse(
                    response,
                    tls_socket,
                    requested_url=request.url,
                    tls_peer=tls_peer,
                )
                tls_socket = None
                return captured
            except ImmportProducerError:
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException, ValueError):
                last_failure = True
            finally:
                if tls_socket is not None:
                    tls_socket.close()
                if raw_socket is not None:
                    raw_socket.close()
        if last_failure:
            raise ImmportProducerError('ImmPort producer transport failed')
        raise ImmportProducerError('ImmPort producer transport failed')


def read_runtime_credential(fd: int = IMMPORT_CREDENTIAL_FD) -> bytearray:
    """Consume one bounded visible-ASCII secret from a pre-opened descriptor.

    The descriptor is always closed.  ``readv`` writes directly into the mutable buffer,
    avoiding an application-level immutable copy while ingesting the credential.
    """

    if not isinstance(fd, int) or isinstance(fd, bool) or fd < 0:
        raise ImmportProducerError('ImmPort runtime credential is unavailable')
    buffer = bytearray(MAX_IMMPORT_CREDENTIAL_BYTES + 2)
    length = 0
    failed = False
    try:
        os.set_inheritable(fd, False)
        while length < len(buffer):
            view = memoryview(buffer)[length:]
            try:
                count = os.readv(fd, (view,))
            finally:
                view.release()
            if count == 0:
                break
            length += count
    except BaseException:
        failed = True
    finally:
        try:
            os.close(fd)
        except OSError:
            failed = True
    if failed or length == len(buffer):
        _zeroize(buffer)
        raise ImmportProducerError('ImmPort runtime credential is unavailable') from None
    del buffer[length:]
    if buffer.endswith(b'\n'):
        del buffer[-1:]
    if (
        len(buffer) < 16
        or len(buffer) > MAX_IMMPORT_CREDENTIAL_BYTES
        or any(value < 33 or value > 126 for value in buffer)
    ):
        _zeroize(buffer)
        raise ImmportProducerError('ImmPort runtime credential is unavailable') from None
    return buffer


def _zeroize(value: bytearray) -> None:
    value[:] = b'\x00' * len(value)


def _now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    try:
        return aware_utc(value, 'ImmPort producer timestamp')
    except (AttributeError, TypeError, ValueError) as error:
        raise ImmportProducerError('ImmPort producer clock failed') from error


def _remaining(monotonic: Callable[[], float], deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise ImmportProducerError('ImmPort producer deadline exceeded')
    return remaining


class _CredentialRepresentationScanner:
    """Bounded detector for common reversible encodings of one runtime credential.

    This is defense in depth, not a proof that arbitrary encodings cannot disclose a key.
    The deployment must still scope the credential, restrict egress, and destroy the
    one-shot producer after the panel.  Mutable representation buffers are overwritten
    with the original credential in the producer's terminal ``finally`` block.
    """

    def __init__(self, credential: bytearray) -> None:
        values = _credential_representations(credential)
        self._values = tuple(values)
        self.maximum_length = max(len(value) for value in values)

    def contains(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        start: int = 0,
    ) -> bool:
        temporary: bytearray | None = None
        if isinstance(payload, memoryview):
            temporary = bytearray(payload)
            candidate: bytes | bytearray = temporary
        else:
            candidate = payload
        try:
            return any(candidate.find(value, start) >= 0 for value in self._values)
        finally:
            if temporary is not None:
                _zeroize(temporary)

    def zeroize(self) -> None:
        for value in self._values:
            _zeroize(value)


def _credential_representations(credential: bytearray) -> list[bytearray]:
    values: list[bytearray] = []

    def add(value: bytearray) -> None:
        if value and all(value != existing for existing in values):
            values.append(value)

    add(bytearray(credential))
    json_escaped = bytearray()
    json_solidus_escaped = bytearray()
    percent_upper = bytearray()
    percent_lower = bytearray()
    percent_selective = bytearray()
    hex_lower = bytearray()
    hex_upper = bytearray()
    safe_url = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'
    lower_hex = b'0123456789abcdef'
    upper_hex = b'0123456789ABCDEF'
    for value in credential:
        if value in (0x22, 0x5C):
            json_escaped.extend((0x5C, value))
            json_solidus_escaped.extend((0x5C, value))
        elif value == 0x2F:
            json_escaped.append(value)
            json_solidus_escaped.extend(b'\\/')
        else:
            json_escaped.append(value)
            json_solidus_escaped.append(value)
        percent_upper.extend((0x25, upper_hex[value >> 4], upper_hex[value & 0x0F]))
        percent_lower.extend((0x25, lower_hex[value >> 4], lower_hex[value & 0x0F]))
        if value in safe_url:
            percent_selective.append(value)
        else:
            percent_selective.extend((0x25, upper_hex[value >> 4], upper_hex[value & 0x0F]))
        hex_lower.extend((lower_hex[value >> 4], lower_hex[value & 0x0F]))
        hex_upper.extend((upper_hex[value >> 4], upper_hex[value & 0x0F]))
    add(json_escaped)
    add(json_solidus_escaped)
    add(percent_upper)
    add(percent_lower)
    add(percent_selective)
    add(hex_lower)
    add(hex_upper)
    for alphabet in (
        b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/',
        b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_',
    ):
        encoded = _base64_mutable(credential, alphabet)
        add(encoded)
        if encoded.endswith(b'=='):
            add(encoded[:-2])
        elif encoded.endswith(b'='):
            add(encoded[:-1])
    return values


def _base64_mutable(value: bytearray, alphabet: bytes) -> bytearray:
    encoded = bytearray()
    for offset in range(0, len(value), 3):
        remaining = len(value) - offset
        first = value[offset]
        second = value[offset + 1] if remaining > 1 else 0
        third = value[offset + 2] if remaining > 2 else 0
        encoded.extend(
            (
                alphabet[first >> 2],
                alphabet[((first & 0x03) << 4) | (second >> 4)],
                alphabet[((second & 0x0F) << 2) | (third >> 6)] if remaining > 1 else 0x3D,
                alphabet[third & 0x3F] if remaining > 2 else 0x3D,
            )
        )
    return encoded


def _scan_header_material(
    pairs: Iterable[tuple[str, str]],
    scanner: _CredentialRepresentationScanner,
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    byte_count = 0
    try:
        iterator = iter(pairs)
    except TypeError as error:
        raise ImmportProducerError('ImmPort response protocol failed') from error
    for pair in iterator:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ImmportProducerError('ImmPort response protocol failed')
        name, value = pair
        if not isinstance(name, str) or not isinstance(value, str):
            raise ImmportProducerError('ImmPort response protocol failed')
        encoded = (name + ': ' + value).encode('latin-1', errors='replace')
        byte_count += len(encoded)
        if byte_count > _MAX_HEADER_BYTES or scanner.contains(encoded):
            raise ImmportProducerError('ImmPort response sanitation failed')
        result.append((name, value))
    return tuple(result)


def _header_values(headers: tuple[tuple[str, str], ...], name: str) -> tuple[str, ...]:
    return tuple(value.strip(' \t') for key, value in headers if key.lower() == name)


def _response_metadata(
    headers: tuple[tuple[str, str], ...],
    max_body_bytes: int,
) -> tuple[ImmportContentType, int | None]:
    content_types = _header_values(headers, 'content-type')
    if len(content_types) != 1 or content_types[0] not in _CONTENT_TYPES:
        raise ImmportProducerError('ImmPort response content type failed')
    content_encoding = _header_values(headers, 'content-encoding')
    if content_encoding and (len(content_encoding) != 1 or content_encoding[0].lower() != 'identity'):
        raise ImmportProducerError('ImmPort response encoding failed')
    content_lengths = _header_values(headers, 'content-length')
    transfer_encodings = _header_values(headers, 'transfer-encoding')
    if content_lengths and transfer_encodings:
        raise ImmportProducerError('ImmPort response framing failed')
    if transfer_encodings and (len(transfer_encodings) != 1 or transfer_encodings[0].lower() != 'chunked'):
        raise ImmportProducerError('ImmPort response framing failed')
    expected_length = None
    if content_lengths:
        if (
            len(content_lengths) != 1
            or not content_lengths[0].isascii()
            or not content_lengths[0].isdigit()
            or len(content_lengths[0]) > 20
        ):
            raise ImmportProducerError('ImmPort response framing failed')
        expected_length = int(content_lengths[0])
        if expected_length > max_body_bytes:
            raise ImmportProducerError('ImmPort response body limit exceeded')
    return cast(ImmportContentType, content_types[0]), expected_length


def _read_body_sanitized(
    response: ImmportTransportResponse,
    scanner: _CredentialRepresentationScanner,
    *,
    deadline: float,
    max_body_bytes: int,
    monotonic: Callable[[], float],
) -> bytearray:
    body = bytearray()
    try:
        while True:
            try:
                chunk = response.read(
                    _CHUNK_SIZE,
                    timeout_seconds=_remaining(monotonic, deadline),
                )
            except http.client.IncompleteRead as error:
                raise ImmportProducerError('ImmPort response framing failed') from error
            if not isinstance(chunk, bytes):
                raise ImmportProducerError('ImmPort response protocol failed')
            if not chunk:
                return body
            if len(body) + len(chunk) > max_body_bytes:
                raise ImmportProducerError('ImmPort response body limit exceeded')
            scan_start = max(0, len(body) - scanner.maximum_length + 1)
            body.extend(chunk)
            if scanner.contains(body, start=scan_start):
                raise ImmportProducerError('ImmPort response sanitation failed')
    except BaseException:
        if body:
            _zeroize(body)
        raise


def _capture_one(
    spec: ImmportAuthenticatedArtifactSpec,
    credential: bytearray,
    scanner: _CredentialRepresentationScanner,
    *,
    collector_implementation_sha256: str,
    collector_execution_environment_sha256: str,
    panel_deadline: float,
    transport: ImmportCredentialedTransport,
    clock: Callable[[], datetime],
    monotonic: Callable[[], float],
) -> ImmportProducerExchange:
    # ``spec`` is kept local to avoid widening the public producer API; the plan schema
    # has already validated all of these exact fields.
    requested_url = spec.requested_url
    authenticated = spec.authentication == 'immport_scoped_api_key_bearer_redacted'
    started_at = _now(clock)
    artifact_deadline = min(panel_deadline, monotonic() + float(spec.timeout_seconds))
    prepared = PreparedImmportRequest(
        url=requested_url,
        authorization_applied=authenticated,
        timeout_seconds=_remaining(monotonic, artifact_deadline),
    )
    wire_credential = memoryview(credential) if authenticated else memoryview(b'')
    try:
        response = transport.open(prepared, wire_credential)
    finally:
        wire_credential.release()
    body = bytearray()
    try:
        headers = _scan_header_material(response.response_headers, scanner)
        if response.status_code != 200 or response.final_url != requested_url:
            body = _read_body_sanitized(
                response,
                scanner,
                deadline=artifact_deadline,
                max_body_bytes=spec.max_body_bytes,
                monotonic=monotonic,
            )
            raise ImmportProducerError('ImmPort response status or redirect policy failed')
        content_type, expected_length = _response_metadata(headers, spec.max_body_bytes)
        body = _read_body_sanitized(
            response,
            scanner,
            deadline=artifact_deadline,
            max_body_bytes=spec.max_body_bytes,
            monotonic=monotonic,
        )
        if expected_length is not None and expected_length != len(body):
            raise ImmportProducerError('ImmPort response framing failed')
        tls_peer = response.tls_peer_binding()
        if not isinstance(tls_peer, ImmportTlsPeerBinding):
            raise ImmportProducerError('ImmPort producer TLS binding failed')
        completed_at = _now(clock)
        _remaining(monotonic, artifact_deadline)
        if completed_at < started_at:
            raise ImmportProducerError('ImmPort producer clock failed')
        immutable_body = bytes(body)
        receipt = ImmportSanitizedCaptureReceipt(
            requested_url=requested_url,
            final_url=requested_url,
            authentication=spec.authentication,
            authorization_applied=authenticated,
            credential_source='runtime_secret_broker' if authenticated else 'not_applicable',
            response_content_type=content_type,
            body_sha256=hashlib.sha256(immutable_body).hexdigest(),
            body_byte_count=len(immutable_body),
            started_at=started_at,
            completed_at=completed_at,
            tls_peer=tls_peer,
            collector_id=IMMPORT_AUTHENTICATED_COLLECTOR_ID,
            collector_implementation_sha256=collector_implementation_sha256,
            collector_execution_environment_sha256=collector_execution_environment_sha256,
        )
        return ImmportProducerExchange(
            artifact_id=spec.artifact_id,
            body_base64=base64.b64encode(immutable_body).decode('ascii'),
            receipt=receipt,
        )
    finally:
        try:
            response.close()
        except (OSError, ValueError):
            pass
        if body:
            _zeroize(body)


def produce_immport_response(
    request: ImmportProducerRequest,
    credential: bytearray,
    *,
    transport: ImmportCredentialedTransport | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic: Callable[[], float] = time.monotonic,
    public_request_bytes: bytes | None = None,
) -> ImmportProducerResponse:
    """Consume and zeroize ``credential`` while producing one exact serial panel."""

    failed = False
    response: ImmportProducerResponse | None = None
    scanner: _CredentialRepresentationScanner | None = None
    try:
        if type(credential) is not bytearray or not 16 <= len(credential) <= (MAX_IMMPORT_CREDENTIAL_BYTES):
            raise ImmportProducerError('ImmPort runtime credential is unavailable')
        if any(value < 33 or value > 126 for value in credential):
            raise ImmportProducerError('ImmPort runtime credential is unavailable')
        scanner = _CredentialRepresentationScanner(credential)
        if public_request_bytes is not None and scanner.contains(public_request_bytes):
            raise ImmportProducerError('ImmPort producer request sanitation failed')
        observed_at = _now(clock)
        if observed_at < request.attempt.started_at or observed_at >= request.attempt.lease_expires_at:
            raise ImmportProducerError('ImmPort attempt is not live')
        lease_remaining = (request.attempt.lease_expires_at - observed_at).total_seconds()
        if lease_remaining < request.plan.panel_deadline_seconds:
            raise ImmportProducerError('ImmPort attempt cannot contain the panel deadline')
        panel_deadline = monotonic() + request.plan.panel_deadline_seconds
        active_transport = transport or DirectImmportHttpsTransport()
        exchanges = tuple(
            _capture_one(
                spec,
                credential,
                scanner,
                collector_implementation_sha256=request.collector_implementation_sha256,
                collector_execution_environment_sha256=(request.collector_execution_environment_sha256),
                panel_deadline=panel_deadline,
                transport=active_transport,
                clock=clock,
                monotonic=monotonic,
            )
            for spec in request.plan.artifacts
        )
        response = ImmportProducerResponse(exchanges=exchanges)
        response_bytes = canonical_json_bytes(response)
        if len(response_bytes) > MAX_IMMPORT_PRODUCER_RESPONSE_BYTES or scanner.contains(response_bytes):
            raise ImmportProducerError('ImmPort producer response sanitation failed')
    except Exception:
        failed = True
    finally:
        if scanner is not None:
            scanner.zeroize()
        _zeroize(credential)
    if failed or response is None:
        raise ImmportProducerError('authenticated ImmPort producer failed') from None
    return response


def parse_immport_producer_response(
    payload: bytes,
    request: ImmportProducerRequest,
) -> tuple[ImmportCapturedExchange, ...]:
    """Validate a one-shot response against its exact request for the parent collector."""

    if not isinstance(payload, bytes) or len(payload) > MAX_IMMPORT_PRODUCER_RESPONSE_BYTES:
        raise ImmportProducerError('invalid ImmPort producer response')
    parsed = None
    try:
        parsed = ImmportProducerResponse.model_validate_json(payload)
    except ValueError:
        pass
    if parsed is None or canonical_json_bytes(parsed) != payload:
        raise ImmportProducerError('invalid ImmPort producer response') from None
    if tuple(item.artifact_id for item in parsed.exchanges) != tuple(
        item.artifact_id for item in request.plan.artifacts
    ):
        raise ImmportProducerError('ImmPort producer response inventory differs from its request')
    exchanges: list[ImmportCapturedExchange] = []
    for spec, item in zip(request.plan.artifacts, parsed.exchanges, strict=True):
        body = item.body_bytes()
        if (
            len(body) > spec.max_body_bytes
            or item.receipt.requested_url != spec.requested_url
            or item.receipt.authentication != spec.authentication
            or item.receipt.collector_implementation_sha256 != request.collector_implementation_sha256
            or item.receipt.collector_execution_environment_sha256 != request.collector_execution_environment_sha256
        ):
            raise ImmportProducerError('ImmPort producer response differs from its request')
        exchanges.append(
            ImmportCapturedExchange(
                artifact_id=item.artifact_id,
                body=body,
                receipt=canonical_json_bytes(item.receipt),
            )
        )
    return tuple(exchanges)


def producer_request_bytes(request: ImmportProducerRequest) -> bytes:
    payload = canonical_json_bytes(request)
    if len(payload) > MAX_IMMPORT_PRODUCER_REQUEST_BYTES:
        raise ImmportProducerError('ImmPort producer request exceeds its byte bound')
    return payload


def parse_immport_producer_request(payload: bytes) -> ImmportProducerRequest:
    if not isinstance(payload, bytes) or len(payload) > MAX_IMMPORT_PRODUCER_REQUEST_BYTES:
        raise ImmportProducerError('invalid ImmPort producer request')
    request = None
    try:
        request = ImmportProducerRequest.model_validate_json(payload)
    except ValueError:
        pass
    if request is None or canonical_json_bytes(request) != payload:
        raise ImmportProducerError('invalid ImmPort producer request') from None
    return request


@dataclass(frozen=True)
class IsolatedImmportProducerClient:
    """Secret-free parent callback for :func:`record_immport_authenticated_capture`.

    ``invoke`` is a deployment boundary (for example, a one-shot container job or a
    socket-activated local service).  It receives only canonical public request bytes;
    credential acquisition happens inside that boundary on descriptor 3.
    """

    collector_implementation_sha256: str
    collector_execution_environment_sha256: str
    invoke: Callable[[bytes], bytes]

    def __call__(
        self,
        plan: ImmportAuthenticatedCollectionPlan,
        attempt: AttemptLease,
    ) -> tuple[ImmportCapturedExchange, ...]:
        request = ImmportProducerRequest(
            plan=plan,
            attempt=attempt,
            collector_implementation_sha256=self.collector_implementation_sha256,
            collector_execution_environment_sha256=(self.collector_execution_environment_sha256),
        )
        failed = False
        output: bytes | object = b''
        try:
            output = self.invoke(producer_request_bytes(request))
        except Exception:
            failed = True
        if failed or type(output) is not bytes:
            raise ImmportProducerError('isolated ImmPort producer invocation failed') from None
        return parse_immport_producer_response(output, request)


@dataclass(frozen=True)
class InheritedFdImmportProducerInvoker:
    """Bounded one-shot supervisor for an already-brokered descriptor 3.

    The supervisor never reads the descriptor.  It passes the inherited descriptor to one
    exact isolated Python module, bounds both runtime and output, kills the complete child
    process group on failure, discards all stderr detail, and closes its own descriptor copy
    after the single invocation.  Obtaining descriptor 3 from Vault, a cloud secret manager,
    an HSM-backed broker, or another deployment-specific service remains outside this module.
    """

    argv: tuple[str, str, str, str]
    hard_deadline_margin_seconds: int = 5
    credential_fd: Literal[3] = IMMPORT_CREDENTIAL_FD

    def __post_init__(self) -> None:
        if (
            type(self.argv) is not tuple
            or self.argv[1:] != ('-I', '-m', 'vaxreplay.operations.immport_producer_cli')
            or not os.path.isabs(self.argv[0])
            or '\x00' in self.argv[0]
            or not 1 <= self.hard_deadline_margin_seconds <= 300
        ):
            raise ImmportProducerError('invalid isolated ImmPort producer command policy')

    def __call__(self, payload: bytes) -> bytes:
        request = parse_immport_producer_request(payload)
        failed = False
        output: bytes | None = None
        try:
            descriptor = os.fstat(self.credential_fd)
            if not (stat.S_ISFIFO(descriptor.st_mode) or stat.S_ISSOCK(descriptor.st_mode)):
                raise ImmportProducerError('runtime credential descriptor type is forbidden')
            result = run_bounded_process(
                self.argv,
                input_bytes=payload,
                wall_seconds=(request.plan.panel_deadline_seconds + self.hard_deadline_margin_seconds),
                max_stdout_bytes=MAX_IMMPORT_PRODUCER_RESPONSE_BYTES,
                max_stderr_bytes=0,
                on_abort=lambda: None,
                env={},
                pass_fds=(self.credential_fd,),
            )
            if (
                result.termination != 'exited'
                or result.exit_code != 0
                or result.stderr
                or result.stdout_truncated
                or result.stderr_truncated
            ):
                raise ImmportProducerError('isolated ImmPort producer process failed')
            output = result.stdout
        except BaseException:
            failed = True
        finally:
            try:
                os.close(self.credential_fd)
            except OSError:
                failed = True
        if failed or output is None:
            raise ImmportProducerError('isolated ImmPort producer process failed') from None
        return output
