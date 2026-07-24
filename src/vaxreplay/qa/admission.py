"""Ed25519 issuance and fail-closed verification of gradient admission."""

from __future__ import annotations

import base64
import hashlib
import hmac
import threading
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.signing import Ed25519Signer, checked_signer
from vaxreplay.qa.schema import (
    GradientAdmissionToken,
    TrainingRunAdmission,
    training_run_admission_sha256,
)

_SIGNING_DOMAIN = b'vaxreplay.gradient-admission-token.v0.1\x00'
_KEY_ID_DOMAIN = b'vaxreplay.gradient-admission-key-id.v0.1\x00'
_TOKEN_ID_DOMAIN = b'vaxreplay.gradient-admission-token-id.v0.1\x00'


class GradientAdmissionError(ValueError):
    """A signature, time, binding, or single-use check failed closed."""


class AdmissionTokenConsumer(Protocol):
    def __call__(self, token_id: str, admission_sha256: str, /) -> bool:
        """Atomically return true only for the token ID's first consumption."""


def _revalidate_admission(
    admission: TrainingRunAdmission,
    *,
    operation: str,
) -> TrainingRunAdmission:
    """Re-run every schema invariant across the signer/verifier trust boundary."""

    try:
        return TrainingRunAdmission.model_validate_json(canonical_json_bytes(admission))
    except Exception as error:
        raise GradientAdmissionError(
            f'gradient admission failed canonical schema revalidation during {operation}'
        ) from error


def _revalidate_token(token: GradientAdmissionToken) -> GradientAdmissionToken:
    """Reject model-copy or otherwise unvalidated token instances."""

    try:
        return GradientAdmissionToken.model_validate_json(canonical_json_bytes(token))
    except Exception as error:
        raise GradientAdmissionError(
            'gradient token failed canonical schema revalidation during verification'
        ) from error


def gradient_admission_signing_key_id(public_key_bytes: bytes) -> str:
    if not isinstance(public_key_bytes, bytes) or len(public_key_bytes) != 32:
        raise GradientAdmissionError('trusted Ed25519 public key must contain exactly 32 bytes')
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes)
    except ValueError as error:
        raise GradientAdmissionError('trusted Ed25519 public key is invalid') from error
    return hashlib.sha256(_KEY_ID_DOMAIN + public_key_bytes).hexdigest()


def gradient_admission_signature_payload(
    admission: TrainingRunAdmission,
    token_id: str,
) -> bytes:
    return _SIGNING_DOMAIN + canonical_json_bytes(
        {
            'token_id': token_id,
            'training_run_admission_sha256': training_run_admission_sha256(admission),
        }
    )


def issue_gradient_admission_token(
    admission: TrainingRunAdmission,
    signer: Ed25519Signer,
    *,
    token_id: str | None = None,
    expected_signer_public_key_bytes: bytes | None = None,
) -> GradientAdmissionToken:
    admission = _revalidate_admission(admission, operation='issuance')
    signer = checked_signer(signer, expected_public_key=expected_signer_public_key_bytes)
    if expected_signer_public_key_bytes is None:
        public_key = signer.public_key_bytes()
        signer = checked_signer(signer, expected_public_key=public_key)
    else:
        public_key = expected_signer_public_key_bytes
    signing_key_id = gradient_admission_signing_key_id(public_key)
    if not hmac.compare_digest(admission.signing_key_id, signing_key_id):
        raise GradientAdmissionError('admission signing_key_id differs from the signer')
    admission_sha256 = training_run_admission_sha256(admission)
    if token_id is None:
        token_id = hashlib.sha256(_TOKEN_ID_DOMAIN + bytes.fromhex(admission_sha256)).hexdigest()[:32]
    if len(token_id) != 32 or any(character not in '0123456789abcdef' for character in token_id):
        raise GradientAdmissionError('token_id must contain exactly 32 lowercase hexadecimal characters')
    payload = gradient_admission_signature_payload(admission, token_id)
    try:
        signature = signer.sign(payload)
    except BaseException:
        raise GradientAdmissionError('gradient admission signer operation failed') from None
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise GradientAdmissionError('gradient admission signer returned an invalid signature length')
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except InvalidSignature:
        raise GradientAdmissionError('gradient admission signer returned a signature from another key') from None
    return GradientAdmissionToken(
        token_id=token_id,
        training_run_admission_sha256=admission_sha256,
        signing_key_id=signing_key_id,
        signature_base64=base64.b64encode(signature).decode('ascii'),
    )


def verify_gradient_admission_token(
    token: GradientAdmissionToken,
    admission: TrainingRunAdmission,
    trusted_public_key_bytes: bytes,
    *,
    now: datetime,
    consume_token: AdmissionTokenConsumer,
    expected_run_id: str,
    expected_trajectory_batch_sha256: str,
    expected_reward_artifact_sha256: str,
    expected_model_sha256: str,
    expected_harness_sha256: str,
    expected_tool_policy_sha256: str,
    expected_environment_sha256: str,
    expected_dataset_sha256: str,
    expected_optimizer_config_sha256: str,
    expected_reward_contract_sha256: str,
    expected_episode_manifest_sha256s: Iterable[str],
) -> TrainingRunAdmission:
    """Verify every gradient-bearing binding and atomically consume the token."""

    token = _revalidate_token(token)
    admission = _revalidate_admission(admission, operation='verification')
    if now.tzinfo is None or now.utcoffset() is None:
        raise GradientAdmissionError('verification time must include a UTC offset')
    now = now.astimezone(timezone.utc)
    expected_key_id = gradient_admission_signing_key_id(trusted_public_key_bytes)
    if not (
        hmac.compare_digest(token.signing_key_id, expected_key_id)
        and hmac.compare_digest(admission.signing_key_id, expected_key_id)
    ):
        raise GradientAdmissionError('gradient admission is signed by an untrusted key')
    admission_sha256 = training_run_admission_sha256(admission)
    if not hmac.compare_digest(token.training_run_admission_sha256, admission_sha256):
        raise GradientAdmissionError('gradient token does not bind the supplied admission')
    try:
        signature = base64.b64decode(token.signature_base64, validate=True)
    except ValueError as error:
        raise GradientAdmissionError('gradient token signature is not canonical base64') from error
    if len(signature) != 64 or base64.b64encode(signature).decode('ascii') != token.signature_base64:
        raise GradientAdmissionError('gradient token signature must be canonical Ed25519 bytes')
    try:
        Ed25519PublicKey.from_public_bytes(trusted_public_key_bytes).verify(
            signature,
            gradient_admission_signature_payload(admission, token.token_id),
        )
    except (InvalidSignature, ValueError) as error:
        raise GradientAdmissionError('gradient token signature verification failed') from error
    if not admission.not_before <= now < admission.expires_at:
        raise GradientAdmissionError('gradient admission is not currently valid')

    expected_episode_manifests = tuple(sorted(expected_episode_manifest_sha256s))
    bindings: tuple[tuple[str, str | tuple[str, ...], str | tuple[str, ...]], ...] = (
        ('run_id', admission.run_id, expected_run_id),
        (
            'trajectory_batch_sha256',
            admission.trajectory_batch_sha256,
            expected_trajectory_batch_sha256,
        ),
        ('reward_artifact_sha256', admission.reward_artifact_sha256, expected_reward_artifact_sha256),
        ('model_sha256', admission.model_sha256, expected_model_sha256),
        ('harness_sha256', admission.harness_sha256, expected_harness_sha256),
        ('tool_policy_sha256', admission.tool_policy_sha256, expected_tool_policy_sha256),
        ('environment_sha256', admission.environment_sha256, expected_environment_sha256),
        ('dataset_sha256', admission.dataset_sha256, expected_dataset_sha256),
        ('optimizer_config_sha256', admission.optimizer_config_sha256, expected_optimizer_config_sha256),
        ('reward_contract_sha256', admission.reward_contract_sha256, expected_reward_contract_sha256),
        ('episode_manifest_sha256s', admission.episode_manifest_sha256s, expected_episode_manifests),
    )
    for field_name, observed, expected in bindings:
        if observed != expected:
            raise GradientAdmissionError(f'gradient admission {field_name} binding mismatch')
    if not callable(consume_token):
        raise GradientAdmissionError('single-use admission requires an atomic token consumer')
    try:
        consumed = consume_token(token.token_id, admission_sha256)
    except BaseException:
        raise GradientAdmissionError('atomic gradient-token consumption failed') from None
    if consumed is not True:
        raise GradientAdmissionError('gradient admission token was already consumed')
    return admission


class InMemoryAdmissionTokenConsumer:
    """Process-local single-use guard for tests and development only."""

    def __init__(self) -> None:
        self._consumed: set[str] = set()
        self._lock = threading.Lock()

    def __call__(self, token_id: str, admission_sha256: str) -> bool:
        del admission_sha256
        with self._lock:
            if token_id in self._consumed:
                return False
            self._consumed.add(token_id)
            return True


def callback_token_consumer(
    operation: Callable[[str, str], bool],
) -> AdmissionTokenConsumer:
    if not callable(operation):
        raise GradientAdmissionError('token-consumption operation must be callable')
    return operation
