"""Minimal external Ed25519 signer boundary for Tier-A services.

Production adapters may forward :meth:`sign` to an isolated process, HSM, or KMS.
The caller never needs private-key bytes.  Every returned signature is verified
locally against the public key before it can be persisted or returned.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class Ed25519SignerError(ValueError):
    """An external signer identity or response failed closed validation."""


@runtime_checkable
class Ed25519Signer(Protocol):
    """Private-key-free interface implemented by file, KMS, and HSM adapters."""

    def public_key_bytes(self) -> bytes:
        """Return the exact 32-byte raw Ed25519 public key."""

    def sign(self, message: bytes) -> bytes:
        """Return one deterministic 64-byte Ed25519 signature over ``message``."""


class LocalEd25519Signer:
    """Development adapter around an in-process Ed25519 private key."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        if not isinstance(private_key, Ed25519PrivateKey):
            raise Ed25519SignerError('local signer requires an Ed25519 private key')
        self._private_key = private_key
        self._public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def public_key_bytes(self) -> bytes:
        return self._public_key

    def sign(self, message: bytes) -> bytes:
        return self._private_key.sign(_bounded_message(message))


class IsolatedEd25519Signer:
    """Verified callback adapter for a separately isolated KMS/HSM broker.

    ``sign_operation`` is expected to perform authenticated IPC or a vendor KMS
    request.  Its exceptions are replaced by a fixed error, and its returned
    signature is verified locally.  A malicious or misconfigured broker therefore
    cannot substitute a key or make unverified bytes enter a transparency log.
    """

    def __init__(
        self,
        *,
        public_key: bytes,
        sign_operation: Callable[[bytes], bytes],
    ) -> None:
        if not isinstance(public_key, bytes) or len(public_key) != 32:
            raise Ed25519SignerError('isolated signer public key must contain exactly 32 bytes')
        if not callable(sign_operation):
            raise Ed25519SignerError('isolated signer operation must be callable')
        try:
            parsed = Ed25519PublicKey.from_public_bytes(public_key)
        except ValueError as error:
            raise Ed25519SignerError('isolated signer public key is invalid') from error
        self._public_key_bytes = public_key
        self._public_key = parsed
        self._sign_operation = sign_operation

    def public_key_bytes(self) -> bytes:
        return self._public_key_bytes

    def sign(self, message: bytes) -> bytes:
        payload = _bounded_message(message)
        failed = False
        try:
            signature = self._sign_operation(payload)
        except BaseException:
            failed = True
            signature = b''
        # Raise outside the handler so the secret-bearing provider exception is not
        # retained in ``__context__`` even when callers log the complete exception.
        if failed:
            raise Ed25519SignerError('isolated signer operation failed') from None
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise Ed25519SignerError('isolated signer returned an invalid signature length')
        try:
            self._public_key.verify(signature, payload)
        except InvalidSignature:
            raise Ed25519SignerError('isolated signer returned a signature from a different key') from None
        return signature


def checked_signer(signer: Ed25519Signer, *, expected_public_key: bytes | None = None) -> Ed25519Signer:
    """Validate a signer boundary and optionally bind it to an out-of-band key."""

    if not isinstance(signer, Ed25519Signer):
        raise Ed25519SignerError('signer does not implement the Ed25519 signer interface')
    lookup_failed = False
    try:
        public_key = signer.public_key_bytes()
    except BaseException:
        lookup_failed = True
        public_key = b''
    if lookup_failed:
        raise Ed25519SignerError('signer public-key lookup failed') from None
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise Ed25519SignerError('signer public key must contain exactly 32 bytes')
    try:
        Ed25519PublicKey.from_public_bytes(public_key)
    except ValueError as error:
        raise Ed25519SignerError('signer public key is invalid') from error
    if expected_public_key is not None and not hmac.compare_digest(public_key, expected_public_key):
        raise Ed25519SignerError('signer public key differs from the out-of-band trusted key')
    return signer


def signer_public_key_bytes(signer: Ed25519Signer) -> bytes:
    return checked_signer(signer).public_key_bytes()


def _bounded_message(message: bytes) -> bytes:
    if not isinstance(message, bytes):
        raise Ed25519SignerError('signer message must be bytes')
    if not message or len(message) > 1024 * 1024 * 1024:
        raise Ed25519SignerError('signer message size is outside the supported range')
    return message


__all__ = [
    'Ed25519Signer',
    'Ed25519SignerError',
    'IsolatedEd25519Signer',
    'LocalEd25519Signer',
    'checked_signer',
    'signer_public_key_bytes',
]
