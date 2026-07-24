"""Bound Unix-socket transport for the authenticated provider gateway.

For a guest-initiated Firecracker vsock connection, Firecracker maps guest port ``P`` to a host
Unix listener at ``<uds_path>_P``.  The trusted supervisor gives each VM a unique socket namespace;
that binding, rather than Unix peer credentials, identifies the configured guest CID.
"""

from __future__ import annotations

import os
import socket
import stat
from datetime import datetime
from pathlib import Path

from vaxreplay.agentic.gateway_auth import (
    GatewayFrameError,
    receive_gateway_frame_bytes,
    send_gateway_frame,
)
from vaxreplay.agentic.provider_gateway import AuthenticatedProviderGateway


class GatewayServerError(RuntimeError):
    """Raised when the private host endpoint cannot be served safely."""


class AuthenticatedGatewayUnixServer:
    """One-frame-per-connection server for a supervisor-bound Firecracker vsock endpoint."""

    def __init__(
        self,
        *,
        gateway: AuthenticatedProviderGateway,
        socket_path: Path,
        bound_peer_cid: int,
        connection_timeout_seconds: float = 30.0,
    ) -> None:
        if bound_peer_cid < 3 or bound_peer_cid > 2**32 - 1:
            raise ValueError('bound Firecracker guest CID is outside the permitted range')
        if connection_timeout_seconds <= 0 or connection_timeout_seconds > 3600:
            raise ValueError('gateway connection timeout must be between 0 and 3600 seconds')
        supplied = socket_path.expanduser()
        if supplied.is_symlink():
            raise ValueError('gateway socket path cannot be a symlink')
        parent = supplied.parent.resolve()
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError('gateway socket parent must be an existing real directory')
        parent_mode = stat.S_IMODE(parent.stat().st_mode)
        if parent_mode & 0o077:
            raise ValueError('gateway socket parent must not be accessible by group or other users')
        self.gateway = gateway
        self.socket_path = parent / supplied.name
        self.bound_peer_cid = bound_peer_cid
        self.connection_timeout_seconds = connection_timeout_seconds
        self._listener: socket.socket | None = None

    def open(self) -> None:
        if self._listener is not None:
            raise GatewayServerError('gateway listener is already open')
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise GatewayServerError('gateway refuses to replace an existing socket path')
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600, follow_symlinks=False)
            listener.listen(1)
            listener.settimeout(self.connection_timeout_seconds)
        except BaseException:
            listener.close()
            self._remove_owned_socket()
            raise
        self._listener = listener

    def serve_one(self, *, observed_at: datetime | None = None) -> None:
        """Accept, authenticate, execute, and respond to exactly one model-call frame."""

        listener = self._listener
        if listener is None:
            raise GatewayServerError('gateway listener is not open')
        try:
            connection, _ = listener.accept()
        except OSError as error:
            raise GatewayServerError('authenticated gateway connection failed') from error
        try:
            connection.settimeout(self.connection_timeout_seconds)
            _, frame = receive_gateway_frame_bytes(
                connection,
                secret_resolver=self.gateway.secret_resolver,
                direction='request',
                maximum_body_bytes=self.gateway.policy.maximum_frame_body_bytes,
            )
            response = self.gateway.handle_frame(
                frame,
                peer_cid=self.bound_peer_cid,
                observed_at=observed_at,
            )
            send_gateway_frame(connection, response)
            try:
                connection.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        except (OSError, GatewayFrameError) as error:
            raise GatewayServerError('authenticated gateway connection failed') from error
        finally:
            connection.close()

    def close(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        self._remove_owned_socket()

    def __enter__(self) -> AuthenticatedGatewayUnixServer:
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def _remove_owned_socket(self) -> None:
        try:
            metadata = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(metadata.st_mode):
            self.socket_path.unlink()
