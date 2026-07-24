from __future__ import annotations

import socket
import tempfile
import threading
from pathlib import Path

import pytest

from tests.test_agentic_provider_gateway import _ISSUED_AT, _PEER_CID, _SECRET, _make_fixture, _request
from vaxreplay.agentic.gateway_server import AuthenticatedGatewayUnixServer, GatewayServerError
from vaxreplay.agentic.provider_gateway import (
    build_gateway_request_frame,
    parse_gateway_response_frame,
)


def test_private_unix_gateway_serves_authenticated_frame(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    endpoint = Path(tempfile.mkdtemp(prefix='vr-gw-', dir='/tmp')).resolve()
    endpoint.chmod(0o700)
    socket_path = endpoint / 'v.sock_4100'
    server = AuthenticatedGatewayUnixServer(
        gateway=fixture.gateway,
        socket_path=socket_path,
        bound_peer_cid=_PEER_CID,
    )
    request_frame = build_gateway_request_frame(fixture.grant, _request(), secret=_SECRET)

    try:
        with server:
            worker = threading.Thread(target=server.serve_one, kwargs={'observed_at': _ISSUED_AT})
            worker.start()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                client.sendall(request_frame)
                chunks = bytearray()
                while chunk := client.recv(65_536):
                    chunks.extend(chunk)
            worker.join(timeout=5)
            assert not worker.is_alive()

        response = parse_gateway_response_frame(bytes(chunks), fixture.grant, secret=_SECRET)
        assert response.succeeded
        assert fixture.adapter.call_count == 1
        assert not socket_path.exists()
    finally:
        endpoint.rmdir()


def test_unix_gateway_refuses_public_parent_and_existing_path(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    public = tmp_path / 'public'
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match='group or other'):
        AuthenticatedGatewayUnixServer(
            gateway=fixture.gateway,
            socket_path=public / 'v.sock_4100',
            bound_peer_cid=_PEER_CID,
        )

    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    path = private / 'v.sock_4100'
    path.write_text('do not replace')
    server = AuthenticatedGatewayUnixServer(
        gateway=fixture.gateway,
        socket_path=path,
        bound_peer_cid=_PEER_CID,
    )
    with pytest.raises(GatewayServerError, match='refuses to replace'):
        server.open()
    assert path.read_text() == 'do not replace'


def test_unix_gateway_accept_has_a_deadline(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    endpoint = Path(tempfile.mkdtemp(prefix='vr-gw-', dir='/tmp')).resolve()
    endpoint.chmod(0o700)
    socket_path = endpoint / 'v.sock_4100'
    server = AuthenticatedGatewayUnixServer(
        gateway=fixture.gateway,
        socket_path=socket_path,
        bound_peer_cid=_PEER_CID,
        connection_timeout_seconds=0.01,
    )
    try:
        with server:
            with pytest.raises(GatewayServerError, match='connection failed'):
                server.serve_one()
    finally:
        endpoint.rmdir()
