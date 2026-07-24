from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
import threading
from typing import Literal

import pytest

from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest
from vaxreplay.agentic.gateway_auth import (
    GATEWAY_FRAME_MAGIC,
    GatewayFrameError,
    GatewaySecretUnavailable,
    InMemoryGatewaySecretStore,
    decode_gateway_frame,
    derive_gateway_frame_key,
    encode_gateway_frame,
    gateway_capability_id,
    peek_gateway_frame_capability_id,
    receive_gateway_frame,
    send_gateway_frame,
)

_HEADER = struct.Struct('>8s64sI')
_REQUEST_MAC_DOMAIN = b'vaxreplay.gateway-request-frame.v0.1\x00'
_SECRET = b'SECRET_DO_NOT_LEAK_0123456789ABC'


def _request() -> AgenticModelRequest:
    return AgenticModelRequest(
        run_id='a' * 32,
        call_index=0,
        messages=(
            AgenticModelMessage(role='system', content='Use only frozen evidence.'),
            AgenticModelMessage(role='user', content='Rank the candidates.'),
        ),
        max_output_tokens=16,
    )


def _frame(*, direction: Literal['request', 'response'] = 'request') -> bytes:
    capability_id = gateway_capability_id(_SECRET)
    return encode_gateway_frame(
        _request(),
        capability_id=capability_id,
        secret=_SECRET,
        direction=direction,
    )


def test_authenticated_frame_round_trip_uses_direction_separated_keys() -> None:
    capability_id = gateway_capability_id(_SECRET)
    request_frame = _frame()

    assert peek_gateway_frame_capability_id(request_frame) == capability_id
    assert (
        decode_gateway_frame(
            request_frame,
            AgenticModelRequest,
            secret=_SECRET,
            direction='request',
            expected_capability_id=capability_id,
        )
        == _request()
    )
    with pytest.raises(GatewayFrameError, match='authentication failed'):
        decode_gateway_frame(request_frame, AgenticModelRequest, secret=_SECRET, direction='response')

    response_frame = _frame(direction='response')
    with pytest.raises(GatewayFrameError, match='authentication failed'):
        decode_gateway_frame(response_frame, AgenticModelRequest, secret=_SECRET, direction='request')


@pytest.mark.parametrize('mutation', ['magic', 'capability', 'body', 'mac'])
def test_authenticated_frame_rejects_header_body_and_mac_mutations(mutation: str) -> None:
    frame = bytearray(_frame())
    if mutation == 'magic':
        frame[0] ^= 1
    elif mutation == 'capability':
        frame[8] = ord('0') if frame[8] != ord('0') else ord('1')
    elif mutation == 'body':
        frame[_HEADER.size] ^= 1
    else:
        frame[-1] ^= 1

    with pytest.raises(GatewayFrameError):
        decode_gateway_frame(bytes(frame), AgenticModelRequest, secret=_SECRET, direction='request')


def test_authenticated_frame_rejects_wrong_secret_and_capability_binding() -> None:
    frame = _frame()
    wrong_secret = b'x' * 32

    with pytest.raises(GatewayFrameError, match='authentication failed'):
        decode_gateway_frame(frame, AgenticModelRequest, secret=wrong_secret, direction='request')
    with pytest.raises(GatewayFrameError, match='does not match'):
        encode_gateway_frame(
            _request(),
            capability_id=gateway_capability_id(_SECRET),
            secret=wrong_secret,
            direction='request',
        )
    with pytest.raises(GatewayFrameError, match='does not match the session'):
        decode_gateway_frame(
            frame,
            AgenticModelRequest,
            secret=_SECRET,
            direction='request',
            expected_capability_id='0' * 64,
        )


def test_authenticated_frame_rejects_truncation_trailing_bytes_and_size_overrun() -> None:
    frame = _frame()
    _, _, body_length = _HEADER.unpack(frame[: _HEADER.size])

    for malformed in (frame[:-1], frame + b'\x00'):
        with pytest.raises(GatewayFrameError, match='truncated or trailing'):
            peek_gateway_frame_capability_id(malformed)
    with pytest.raises(GatewayFrameError, match='header is invalid'):
        peek_gateway_frame_capability_id(frame, maximum_body_bytes=body_length - 1)
    with pytest.raises(GatewayFrameError, match='byte limit'):
        encode_gateway_frame(
            _request(),
            capability_id=gateway_capability_id(_SECRET),
            secret=_SECRET,
            direction='request',
            maximum_body_bytes=body_length - 1,
        )


def test_authenticated_frame_rejects_authenticated_but_noncanonical_json() -> None:
    canonical_frame = _frame()
    _, capability_bytes, body_length = _HEADER.unpack(canonical_frame[: _HEADER.size])
    canonical_body = canonical_frame[_HEADER.size : _HEADER.size + body_length]
    noncanonical_body = json.dumps(json.loads(canonical_body), sort_keys=False).encode('utf-8')
    assert noncanonical_body != canonical_body
    header = _HEADER.pack(GATEWAY_FRAME_MAGIC, capability_bytes, len(noncanonical_body))
    mac = hmac.new(
        derive_gateway_frame_key(_SECRET, 'request'),
        _REQUEST_MAC_DOMAIN + header + noncanonical_body,
        hashlib.sha256,
    ).digest()

    with pytest.raises(GatewayFrameError, match='canonical JSON'):
        decode_gateway_frame(
            header + noncanonical_body + mac,
            AgenticModelRequest,
            secret=_SECRET,
            direction='request',
        )


def test_socket_helpers_handle_fragmented_frame_io() -> None:
    frame = _frame()
    store = InMemoryGatewaySecretStore()
    capability_id = store.register(_SECRET)
    reader, writer = socket.socketpair()

    def write_fragments() -> None:
        try:
            for offset in range(0, len(frame), 7):
                send_gateway_frame(writer, frame[offset : offset + 7])
        finally:
            writer.close()

    thread = threading.Thread(target=write_fragments)
    thread.start()
    try:
        received_capability, request = receive_gateway_frame(
            reader,
            AgenticModelRequest,
            secret_resolver=store,
            direction='request',
        )
    finally:
        reader.close()
        thread.join()

    assert received_capability == capability_id
    assert request == _request()


def test_secret_store_revocation_is_fail_closed() -> None:
    store = InMemoryGatewaySecretStore()
    capability_id = store.register(_SECRET)
    assert store.resolve(capability_id) == _SECRET

    store.revoke(capability_id)

    assert not store.contains(capability_id)
    with pytest.raises(GatewaySecretUnavailable, match='unavailable'):
        store.resolve(capability_id)
