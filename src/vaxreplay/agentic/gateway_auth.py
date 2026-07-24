"""Authenticated, bounded framing for the worker-to-provider-gateway channel.

The capability ID is deliberately visible in the fixed-size header so the gateway can resolve a
per-run secret before parsing attacker-controlled JSON.  The exact header and body bytes are
authenticated before Pydantic or ``json`` sees them.
"""

from __future__ import annotations

import hashlib
import hmac
import socket
import struct
import threading
from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from vaxreplay.bundle import canonical_json_bytes

GATEWAY_FRAME_MAGIC = b'VAXRGW01'
GATEWAY_FRAME_VERSION = 'vaxreplay.gateway-frame.v0.1'
GATEWAY_CAPABILITY_SECRET_BYTES = 32
GATEWAY_CAPABILITY_ID_BYTES = 64
GATEWAY_FRAME_MAC_BYTES = 32
DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES = 4 * 1024 * 1024

_HEADER = struct.Struct(f'>8s{GATEWAY_CAPABILITY_ID_BYTES}sI')
GATEWAY_FRAME_FIXED_BYTES = _HEADER.size + GATEWAY_FRAME_MAC_BYTES
_CAPABILITY_ID_DOMAIN = b'vaxreplay.gateway-capability-id.v0.1\x00'
_REQUEST_KEY_DOMAIN = b'vaxreplay.gateway-request-key.v0.1\x00'
_RESPONSE_KEY_DOMAIN = b'vaxreplay.gateway-response-key.v0.1\x00'
_REQUEST_MAC_DOMAIN = b'vaxreplay.gateway-request-frame.v0.1\x00'
_RESPONSE_MAC_DOMAIN = b'vaxreplay.gateway-response-frame.v0.1\x00'

GatewayFrameDirection = Literal['request', 'response']
ModelT = TypeVar('ModelT', bound=BaseModel)


class GatewayFrameError(ValueError):
    """Raised for malformed, oversized, noncanonical, or unauthenticated frames."""


class GatewaySecretUnavailable(GatewayFrameError):
    """Raised without revealing whether a capability once existed or was revoked."""


class GatewaySecretResolver(Protocol):
    def resolve(self, capability_id: str) -> bytes: ...


@runtime_checkable
class RevocableGatewaySecretResolver(GatewaySecretResolver, Protocol):
    def revoke(self, capability_id: str) -> None: ...


class InMemoryGatewaySecretStore:
    """Small test/reference secret store; production should inject a vault-backed resolver."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._secrets: dict[str, bytes] = {}

    def register(self, secret: bytes) -> str:
        capability_id = gateway_capability_id(secret)
        with self._lock:
            if capability_id in self._secrets:
                raise ValueError('gateway capability is already registered')
            self._secrets[capability_id] = bytes(secret)
        return capability_id

    def resolve(self, capability_id: str) -> bytes:
        _validate_capability_id(capability_id)
        with self._lock:
            secret = self._secrets.get(capability_id)
        if secret is None:
            raise GatewaySecretUnavailable('gateway capability is unavailable')
        return secret

    def revoke(self, capability_id: str) -> None:
        _validate_capability_id(capability_id)
        with self._lock:
            self._secrets.pop(capability_id, None)

    def contains(self, capability_id: str) -> bool:
        _validate_capability_id(capability_id)
        with self._lock:
            return capability_id in self._secrets


def maximum_gateway_frame_bytes(maximum_body_bytes: int) -> int:
    """Return the exact full-frame bound for a configured canonical-JSON body bound."""

    _validate_maximum(maximum_body_bytes)
    return GATEWAY_FRAME_FIXED_BYTES + maximum_body_bytes


def gateway_capability_id(secret: bytes) -> str:
    _validate_secret(secret)
    return hashlib.sha256(_CAPABILITY_ID_DOMAIN + secret).hexdigest()


def derive_gateway_frame_key(secret: bytes, direction: GatewayFrameDirection) -> bytes:
    _validate_secret(secret)
    domain = _REQUEST_KEY_DOMAIN if direction == 'request' else _RESPONSE_KEY_DOMAIN
    return hmac.new(secret, domain, hashlib.sha256).digest()


def encode_gateway_frame(
    payload: BaseModel,
    *,
    capability_id: str,
    secret: bytes,
    direction: GatewayFrameDirection,
    maximum_body_bytes: int = DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES,
) -> bytes:
    _validate_maximum(maximum_body_bytes)
    _validate_capability_id(capability_id)
    if gateway_capability_id(secret) != capability_id:
        raise GatewayFrameError('gateway capability secret does not match its identifier')
    body = canonical_json_bytes(payload)
    if not body or len(body) > maximum_body_bytes:
        raise GatewayFrameError('gateway frame body exceeds its byte limit')
    header = _HEADER.pack(GATEWAY_FRAME_MAGIC, capability_id.encode('ascii'), len(body))
    mac = hmac.new(
        derive_gateway_frame_key(secret, direction),
        _mac_domain(direction) + header + body,
        hashlib.sha256,
    ).digest()
    return header + body + mac


def peek_gateway_frame_capability_id(
    frame: bytes,
    *,
    maximum_body_bytes: int = DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES,
) -> str:
    """Validate only the public bounded header; do not parse the unauthenticated body."""

    _, capability_id, _ = _split_frame(frame, maximum_body_bytes=maximum_body_bytes)
    return capability_id


def decode_gateway_frame(
    frame: bytes,
    model_type: type[ModelT],
    *,
    secret: bytes,
    direction: GatewayFrameDirection,
    expected_capability_id: str | None = None,
    maximum_body_bytes: int = DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES,
) -> ModelT:
    header, capability_id, body = _split_frame(frame, maximum_body_bytes=maximum_body_bytes)
    if expected_capability_id is not None and capability_id != expected_capability_id:
        raise GatewayFrameError('gateway frame capability does not match the session')
    if gateway_capability_id(secret) != capability_id:
        raise GatewayFrameError('gateway frame authentication failed')
    supplied_mac = frame[-GATEWAY_FRAME_MAC_BYTES:]
    expected_mac = hmac.new(
        derive_gateway_frame_key(secret, direction),
        _mac_domain(direction) + header + body,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(supplied_mac, expected_mac):
        raise GatewayFrameError('gateway frame authentication failed')
    try:
        value = model_type.model_validate_json(body)
    except ValidationError as error:
        raise GatewayFrameError('gateway frame body does not match its strict schema') from error
    if canonical_json_bytes(value) != body:
        raise GatewayFrameError('gateway frame body must use canonical JSON')
    return value


def receive_gateway_frame(
    connection: socket.socket,
    model_type: type[ModelT],
    *,
    secret_resolver: GatewaySecretResolver,
    direction: GatewayFrameDirection,
    maximum_body_bytes: int = DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES,
) -> tuple[str, ModelT]:
    """Read one complete frame with a caller-supplied socket deadline."""

    _validate_maximum(maximum_body_bytes)
    header = _receive_exact(connection, _HEADER.size)
    magic, capability_bytes, body_length = _HEADER.unpack(header)
    if magic != GATEWAY_FRAME_MAGIC or body_length == 0 or body_length > maximum_body_bytes:
        raise GatewayFrameError('gateway frame header is invalid')
    try:
        capability_id = capability_bytes.decode('ascii')
    except UnicodeDecodeError as error:
        raise GatewayFrameError('gateway frame header is invalid') from error
    _validate_capability_id(capability_id)
    secret = secret_resolver.resolve(capability_id)
    remainder = _receive_exact(connection, body_length + GATEWAY_FRAME_MAC_BYTES)
    frame = header + remainder
    value = decode_gateway_frame(
        frame,
        model_type,
        secret=secret,
        direction=direction,
        expected_capability_id=capability_id,
        maximum_body_bytes=maximum_body_bytes,
    )
    return capability_id, value


def receive_gateway_frame_bytes(
    connection: socket.socket,
    *,
    secret_resolver: GatewaySecretResolver,
    direction: GatewayFrameDirection,
    maximum_body_bytes: int = DEFAULT_MAX_GATEWAY_FRAME_BODY_BYTES,
) -> tuple[str, bytes]:
    """Read and authenticate one exact frame while preserving the original wire bytes."""

    _validate_maximum(maximum_body_bytes)
    header = _receive_exact(connection, _HEADER.size)
    magic, capability_bytes, body_length = _HEADER.unpack(header)
    if magic != GATEWAY_FRAME_MAGIC or body_length == 0 or body_length > maximum_body_bytes:
        raise GatewayFrameError('gateway frame header is invalid')
    try:
        capability_id = capability_bytes.decode('ascii')
    except UnicodeDecodeError as error:
        raise GatewayFrameError('gateway frame header is invalid') from error
    _validate_capability_id(capability_id)
    secret = secret_resolver.resolve(capability_id)
    frame = header + _receive_exact(connection, body_length + GATEWAY_FRAME_MAC_BYTES)
    # Decode into a minimal BaseModel only after authenticating would lose strict schema checking.
    # Callers therefore authenticate the exact MAC here and let their endpoint decode its schema.
    supplied_mac = frame[-GATEWAY_FRAME_MAC_BYTES:]
    body = frame[_HEADER.size : -GATEWAY_FRAME_MAC_BYTES]
    expected_mac = hmac.new(
        derive_gateway_frame_key(secret, direction),
        _mac_domain(direction) + header + body,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(supplied_mac, expected_mac):
        raise GatewayFrameError('gateway frame authentication failed')
    return capability_id, frame


def send_gateway_frame(connection: socket.socket, frame: bytes) -> None:
    """Write every byte or fail; socket deadlines are configured by the caller."""

    view = memoryview(frame)
    while view:
        sent = connection.send(view)
        if sent <= 0:
            raise GatewayFrameError('gateway connection closed during frame write')
        view = view[sent:]


def _split_frame(frame: bytes, *, maximum_body_bytes: int) -> tuple[bytes, str, bytes]:
    _validate_maximum(maximum_body_bytes)
    minimum = _HEADER.size + 1 + GATEWAY_FRAME_MAC_BYTES
    if len(frame) < minimum:
        raise GatewayFrameError('gateway frame is truncated')
    header = frame[: _HEADER.size]
    magic, capability_bytes, body_length = _HEADER.unpack(header)
    if magic != GATEWAY_FRAME_MAGIC or body_length == 0 or body_length > maximum_body_bytes:
        raise GatewayFrameError('gateway frame header is invalid')
    expected_length = _HEADER.size + body_length + GATEWAY_FRAME_MAC_BYTES
    if len(frame) != expected_length:
        raise GatewayFrameError('gateway frame has truncated or trailing bytes')
    try:
        capability_id = capability_bytes.decode('ascii')
    except UnicodeDecodeError as error:
        raise GatewayFrameError('gateway frame header is invalid') from error
    _validate_capability_id(capability_id)
    return header, capability_id, frame[_HEADER.size : -GATEWAY_FRAME_MAC_BYTES]


def _receive_exact(connection: socket.socket, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = connection.recv(byte_count - len(chunks))
        if not chunk:
            raise GatewayFrameError('gateway connection closed during frame read')
        chunks.extend(chunk)
    return bytes(chunks)


def _mac_domain(direction: GatewayFrameDirection) -> bytes:
    if direction == 'request':
        return _REQUEST_MAC_DOMAIN
    if direction == 'response':
        return _RESPONSE_MAC_DOMAIN
    raise ValueError('unsupported gateway frame direction')


def _validate_secret(secret: bytes) -> None:
    if not isinstance(secret, bytes) or len(secret) != GATEWAY_CAPABILITY_SECRET_BYTES:
        raise ValueError(f'gateway capability secret must contain exactly {GATEWAY_CAPABILITY_SECRET_BYTES} bytes')


def _validate_capability_id(capability_id: str) -> None:
    if len(capability_id) != GATEWAY_CAPABILITY_ID_BYTES or any(
        character not in '0123456789abcdef' for character in capability_id
    ):
        raise GatewayFrameError('gateway capability ID is invalid')


def _validate_maximum(maximum_body_bytes: int) -> None:
    if maximum_body_bytes <= 0 or maximum_body_bytes > 64 * 1024 * 1024:
        raise ValueError('gateway maximum frame body must be between 1 byte and 64 MiB')
