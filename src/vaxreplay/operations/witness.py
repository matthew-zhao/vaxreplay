"""Externally witness an operations-ledger checkpoint without trusting the broker.

The broker in this module is only orchestration: it submits a checkpoint commitment to
an injected provider, binds the returned raw proof bytes, and immediately asks separate
trusted code to verify those bytes.  It is not a clock or timestamp authority.  Loading
the resulting artifact always reruns the injected verifier, so the artifact remains
offline-verifiable without trusting a boolean written by its creator.

An externally witnessed checkpoint is one prerequisite for prospective provenance.  It
does not establish source completeness, publication time, semantic validity, or Tier A
eligibility by itself.  In particular, a locally generated signature is not an external
witness and is intentionally not representable as a witness method here.
"""

from __future__ import annotations

import base64
import binascii
import enum
import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.schema import (
    LEDGER_CHECKPOINT_SCHEMA_VERSION,
    SAFE_ID_PATTERN,
    LedgerCheckpoint,
    checkpoint_bytes,
    checkpoint_sha256,
)

WITNESS_POLICY_SCHEMA_VERSION = 'vaxreplay.operations-witness-policy.v0.1'
CHECKPOINT_WITNESS_REQUEST_SCHEMA_VERSION = 'vaxreplay.operations-checkpoint-witness-request.v0.1'
REGISTRY_CHECKPOINT_WITNESS_REQUEST_SCHEMA_VERSION = 'vaxreplay.operations-registry-checkpoint-witness-request.v0.1'
SIGNED_PLAN_SELECTION_REGISTRY_CHECKPOINT_SCHEMA_VERSION = 'vaxreplay.signed-plan-selection-registry-checkpoint.v0.1'
EXTERNAL_WITNESS_CLAIM_SCHEMA_VERSION = 'vaxreplay.operations-external-witness-claim.v0.2'
AUTHENTICATED_EXTERNAL_WITNESS_FACTS_SCHEMA_VERSION = 'vaxreplay.operations-authenticated-external-witness-facts.v0.1'
EXTERNAL_CHECKPOINT_WITNESS_RECEIPT_SCHEMA_VERSION = 'vaxreplay.operations-external-checkpoint-witness-receipt.v0.1'
WITNESSED_CHECKPOINT_SCHEMA_VERSION = 'vaxreplay.operations-witnessed-checkpoint.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_MODEL_BYTES = 16 * 1024 * 1024
_MAX_PROOF_BYTES = 16 * 1024 * 1024
_CHECKPOINT_PATH = 'checkpoint.json'
_PROOF_PATH = 'external-proof.bin'
_MANIFEST_PATH = 'witness.json'


class WitnessVerificationError(ValueError):
    """An external checkpoint witness artifact or proof failed closed verification."""


class ExternalWitnessMethod(str, enum.Enum):
    """Accepted independent witness mechanisms.

    Organizer attestations and local signatures are deliberately absent.  Selecting one
    of these values is not itself proof of independence; the injected trusted verifier
    must enforce the pinned trust policy against the raw external proof.
    """

    RFC3161_TIMESTAMP = 'rfc3161_timestamp'
    PUBLIC_TRANSPARENCY_LOG = 'public_transparency_log'


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class WitnessPolicyBinding(StrictModel):
    """Pinned witness, trust, and verifier-policy identities for one submission.

    The digest fields bind policy artifacts managed outside this generic layer.  A
    trusted verifier is expected to load those exact policy bytes independently and to
    reject proofs that do not satisfy them.
    """

    schema_version: Literal['vaxreplay.operations-witness-policy.v0.1'] = WITNESS_POLICY_SCHEMA_VERSION
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    method: ExternalWitnessMethod
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    trust_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    trust_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    verifier_id: str = Field(pattern=SAFE_ID_PATTERN)
    verifier_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)


class CheckpointWitnessRequest(StrictModel):
    """Minimal commitment sent by the broker to an external witness provider."""

    schema_version: Literal['vaxreplay.operations-checkpoint-witness-request.v0.1'] = (
        CHECKPOINT_WITNESS_REQUEST_SCHEMA_VERSION
    )
    checkpoint_schema_version: Literal['vaxreplay.operations-ledger-checkpoint.v0.1'] = LEDGER_CHECKPOINT_SCHEMA_VERSION
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_bytes: int = Field(gt=0)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    method: ExternalWitnessMethod
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)


class RegistryCheckpointWitnessRequest(StrictModel):
    """Exact signed registry head plus consistency path submitted to a monitor."""

    schema_version: Literal['vaxreplay.operations-registry-checkpoint-witness-request.v0.1'] = (
        REGISTRY_CHECKPOINT_WITNESS_REQUEST_SCHEMA_VERSION
    )
    checkpoint_schema_version: Literal['vaxreplay.signed-plan-selection-registry-checkpoint.v0.1'] = (
        SIGNED_PLAN_SELECTION_REGISTRY_CHECKPOINT_SCHEMA_VERSION
    )
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_bytes: int = Field(gt=0)
    signed_checkpoint_base64: str = Field(min_length=1)
    consistency_from_tree_size: int = Field(ge=0)
    consistency_from_root_sha256: str = Field(pattern=_SHA256_PATTERN)
    consistency_proof_sha256: tuple[str, ...] = Field(max_length=256)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    method: ExternalWitnessMethod
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('signed_checkpoint_base64')
    @classmethod
    def validate_signed_checkpoint(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError('signed_checkpoint_base64 must use valid base64') from error
        if not decoded or len(decoded) > _MAX_MODEL_BYTES:
            raise ValueError('signed_checkpoint_base64 has an invalid decoded size')
        if base64.b64encode(decoded).decode('ascii') != value:
            raise ValueError('signed_checkpoint_base64 must use canonical base64')
        return value

    @field_validator('consistency_proof_sha256')
    @classmethod
    def validate_consistency_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(item) != 64 or any(character not in '0123456789abcdef' for character in item) for item in value):
            raise ValueError('consistency proof must contain lowercase SHA-256 digests')
        return value


class ExternalWitnessClaim(StrictModel):
    """Untrusted, non-security metadata returned with opaque proof bytes.

    No time, identity, imprint, or policy field is accepted from the broker.  Those
    values must be parsed and authenticated from ``proof_bytes`` by the trusted
    verifier and returned as :class:`AuthenticatedExternalWitnessFacts`.
    """

    schema_version: Literal['vaxreplay.operations-external-witness-claim.v0.2'] = EXTERNAL_WITNESS_CLAIM_SCHEMA_VERSION
    verification_uri: str = Field(min_length=1, max_length=4096)

    @field_validator('verification_uri')
    @classmethod
    def validate_verification_uri(cls, value: str) -> str:
        if value.strip() != value or any(character in value for character in '\x00\r\n'):
            raise ValueError('verification_uri must be trimmed and contain no control separators')
        return value


class AuthenticatedExternalWitnessFacts(StrictModel):
    """Security facts parsed and authenticated from the raw external proof.

    A trusted verifier must return this model only after validating the complete
    RFC 3161 token or transparency-log proof under independently pinned policy and
    trust-root bytes.  In particular, ``witnessed_at`` and ``checkpoint_sha256``
    must come from authenticated proof fields, never from ``witness.json`` or a
    provider response.
    """

    schema_version: Literal['vaxreplay.operations-authenticated-external-witness-facts.v0.1'] = (
        AUTHENTICATED_EXTERNAL_WITNESS_FACTS_SCHEMA_VERSION
    )
    receipt_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=SAFE_ID_PATTERN)
    method: ExternalWitnessMethod
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    witnessed_at: datetime

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'witnessed_at')


class ExternalCheckpointWitnessReceipt(StrictModel):
    """Complete metadata binding a raw proof to exact canonical checkpoint bytes."""

    schema_version: Literal['vaxreplay.operations-external-checkpoint-witness-receipt.v0.1'] = (
        EXTERNAL_CHECKPOINT_WITNESS_RECEIPT_SCHEMA_VERSION
    )
    receipt_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=SAFE_ID_PATTERN)
    method: ExternalWitnessMethod
    checkpoint_schema_version: Literal['vaxreplay.operations-ledger-checkpoint.v0.1'] = LEDGER_CHECKPOINT_SCHEMA_VERSION
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_bytes: int = Field(gt=0)
    witnessed_at: datetime
    proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1, max_length=4096)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    trust_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    trust_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    verifier_id: str = Field(pattern=SAFE_ID_PATTERN)
    verifier_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'witnessed_at')

    @field_validator('verification_uri')
    @classmethod
    def validate_verification_uri(cls, value: str) -> str:
        if value.strip() != value or any(character in value for character in '\x00\r\n'):
            raise ValueError('verification_uri must be trimmed and contain no control separators')
        return value


class WitnessedCheckpointManifest(StrictModel):
    """Exact three-file manifest for one externally witnessed checkpoint."""

    schema_version: Literal['vaxreplay.operations-witnessed-checkpoint.v0.1'] = WITNESSED_CHECKPOINT_SCHEMA_VERSION
    checkpoint_path: Literal['checkpoint.json'] = _CHECKPOINT_PATH
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_bytes: int = Field(gt=0)
    proof_path: Literal['external-proof.bin'] = _PROOF_PATH
    receipt: ExternalCheckpointWitnessReceipt
    external_proof_verification_required: Literal[True] = True
    tier_a_eligibility_established: Literal[False] = False

    @model_validator(mode='after')
    def validate_receipt_binding(self) -> Self:
        if (
            self.receipt.checkpoint_sha256 != self.checkpoint_sha256
            or self.receipt.checkpoint_bytes != self.checkpoint_bytes
        ):
            raise ValueError('external witness receipt does not bind the declared checkpoint')
        return self


type ExternalCheckpointWitnessProvider = Callable[
    [CheckpointWitnessRequest],
    tuple[ExternalWitnessClaim, bytes],
]
type RegistryCheckpointWitnessProvider = Callable[
    [RegistryCheckpointWitnessRequest],
    tuple[ExternalWitnessClaim, bytes],
]
type TrustedCheckpointWitnessVerifier = Callable[
    [bytes, bytes, WitnessPolicyBinding],
    AuthenticatedExternalWitnessFacts,
]


@dataclass(frozen=True)
class LoadedWitnessedCheckpoint:
    """Freshly verified view; callers must not treat instances as capabilities.

    The dataclass is intentionally transparent and therefore publicly constructible.
    Security-sensitive consumers must call :func:`load_witnessed_checkpoint` themselves
    with an out-of-band pinned policy immediately before use.
    """

    root: Path
    manifest: WitnessedCheckpointManifest
    checkpoint: LedgerCheckpoint
    checkpoint_bytes: bytes
    proof_bytes: bytes
    manifest_sha256: str
    proof_reverified: Literal[True] = True

    @property
    def witnessed_at(self) -> datetime:
        return self.manifest.receipt.witnessed_at


def broker_witness_checkpoint(
    output_dir: Path,
    *,
    checkpoint: LedgerCheckpoint,
    policy: WitnessPolicyBinding,
    provider: ExternalCheckpointWitnessProvider,
    verifier: TrustedCheckpointWitnessVerifier,
    verified_at: datetime | None = None,
) -> LoadedWitnessedCheckpoint:
    """Submit only the checkpoint commitment, verify the response, and persist it.

    ``provider`` is the transport/service adapter.  It is untrusted for verification and
    receives no captured objects, source bodies, or ledger contents beyond the compact
    checkpoint commitment.  ``verifier`` must be separately supplied trusted code that
    validates the raw proof under the exact pinned policy and trust roots.
    """

    if provider is None:  # type: ignore[comparison-overlap]
        raise ValueError('an external checkpoint witness provider is required')
    if verifier is None:  # type: ignore[comparison-overlap]
        raise ValueError('an independent trusted checkpoint witness verifier is required')
    validated_checkpoint, exact_checkpoint_bytes = _validated_checkpoint(checkpoint, error_type=ValueError)
    validated_policy = _validated_model(policy, WitnessPolicyBinding, 'witness policy', error_type=ValueError)
    request = CheckpointWitnessRequest(
        checkpoint_sha256=hashlib.sha256(exact_checkpoint_bytes).hexdigest(),
        checkpoint_bytes=len(exact_checkpoint_bytes),
        authority_id=validated_policy.authority_id,
        method=validated_policy.method,
        policy_id=validated_policy.policy_id,
        policy_sha256=validated_policy.policy_sha256,
    )
    try:
        response = provider(request)
    except Exception as error:
        raise ValueError(f'external checkpoint witness provider failed: {error}') from error
    if not isinstance(response, tuple) or len(response) != 2:
        raise ValueError('external checkpoint witness provider returned an invalid response')
    claim, proof_bytes = response
    try:
        validated_claim = _validated_model(
            claim,
            ExternalWitnessClaim,
            'external witness claim',
            error_type=ValueError,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f'external checkpoint witness provider returned an invalid claim: {error}') from error
    if not isinstance(proof_bytes, bytes):
        raise ValueError('external checkpoint witness provider must return exact proof bytes')
    return build_witnessed_checkpoint(
        output_dir,
        checkpoint=validated_checkpoint,
        policy=validated_policy,
        claim=validated_claim,
        proof_bytes=proof_bytes,
        verifier=verifier,
        verified_at=verified_at,
    )


def build_witnessed_checkpoint(
    output_dir: Path,
    *,
    checkpoint: LedgerCheckpoint,
    policy: WitnessPolicyBinding,
    claim: ExternalWitnessClaim,
    proof_bytes: bytes,
    verifier: TrustedCheckpointWitnessVerifier,
    verified_at: datetime | None = None,
) -> LoadedWitnessedCheckpoint:
    """Verify external proof material and durably persist its canonical sidecar.

    All security-sensitive receipt fields are constructed from the authenticated
    facts returned by ``verifier``.  The provider's claim contributes only a
    non-authoritative verification locator.
    """

    if verifier is None:  # type: ignore[comparison-overlap]
        raise ValueError('an independent trusted checkpoint witness verifier is required')
    if not isinstance(proof_bytes, bytes):
        raise ValueError('proof_bytes must preserve the exact external proof bytes')
    if not proof_bytes:
        raise ValueError('external checkpoint witness proof cannot be empty')
    if len(proof_bytes) > _MAX_PROOF_BYTES:
        raise ValueError('external checkpoint witness proof exceeds the configured byte limit')
    checked_at = _aware(verified_at or _now_utc(), 'verified_at')
    checkpoint, exact_checkpoint_bytes = _validated_checkpoint(checkpoint, error_type=ValueError)
    policy = _validated_model(policy, WitnessPolicyBinding, 'witness policy', error_type=ValueError)
    claim = _validated_model(claim, ExternalWitnessClaim, 'external witness claim', error_type=ValueError)
    facts = _authenticate_external_proof(
        checkpoint,
        checkpoint_bytes=exact_checkpoint_bytes,
        proof_bytes=proof_bytes,
        verifier=verifier,
        expected_policy=policy,
        verified_at=checked_at,
    )
    receipt = ExternalCheckpointWitnessReceipt(
        receipt_id=facts.receipt_id,
        authority_id=facts.authority_id,
        witness_id=facts.witness_id,
        method=facts.method,
        checkpoint_sha256=hashlib.sha256(exact_checkpoint_bytes).hexdigest(),
        checkpoint_bytes=len(exact_checkpoint_bytes),
        witnessed_at=facts.witnessed_at,
        proof_sha256=hashlib.sha256(proof_bytes).hexdigest(),
        proof_bytes=len(proof_bytes),
        verification_uri=claim.verification_uri,
        policy_id=facts.policy_id,
        policy_sha256=policy.policy_sha256,
        trust_policy_id=policy.trust_policy_id,
        trust_policy_sha256=policy.trust_policy_sha256,
        verifier_id=policy.verifier_id,
        verifier_implementation_sha256=policy.verifier_implementation_sha256,
    )
    manifest = WitnessedCheckpointManifest(
        checkpoint_sha256=receipt.checkpoint_sha256,
        checkpoint_bytes=receipt.checkpoint_bytes,
        receipt=receipt,
    )

    target_request = Path(output_dir).expanduser().absolute()
    target_request.parent.mkdir(parents=True, exist_ok=True)
    target_root = target_request.parent.resolve(strict=True) / target_request.name
    lock_path = target_root.parent / f'.{target_root.name}.publish.lock'
    lock_descriptor = _acquire_publication_lock(lock_path)
    staging: Path | None = None
    try:
        if os.path.lexists(target_root):
            raise ValueError(f'witnessed checkpoint output already exists: {target_root}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{target_root.name}.', dir=target_root.parent))
        _write_durable_file(staging / _CHECKPOINT_PATH, exact_checkpoint_bytes)
        _write_durable_file(staging / _PROOF_PATH, proof_bytes)
        _write_durable_file(staging / _MANIFEST_PATH, canonical_json_bytes(manifest))
        staging.chmod(0o755)
        _fsync_directory(staging)
        # The sibling lock serializes cooperating publishers; the kernel-level
        # exclusive rename also protects against a writer that ignores that lock.
        try:
            rename_directory_noreplace(staging, target_root)
        except FileExistsError as error:
            raise ValueError(f'witnessed checkpoint output already exists: {target_root}') from error
        staging = None
        _fsync_directory(target_root.parent)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            _release_publication_lock(lock_path, lock_descriptor)
        finally:
            _fsync_directory(target_root.parent)
    # Publication above created the exact no-replace directory below a parent
    # that was canonicalized before staging.  Do not resolve the installed root
    # again: an attacker replacing it with a symlink between rename and reload
    # must be rejected by the descriptor walker, not silently followed.
    return load_witnessed_checkpoint(
        target_root,
        verifier=verifier,
        expected_policy=policy,
        verified_at=checked_at,
        expected_checkpoint_sha256=receipt.checkpoint_sha256,
    )


def load_witnessed_checkpoint(
    root: Path,
    *,
    verifier: TrustedCheckpointWitnessVerifier,
    expected_policy: WitnessPolicyBinding,
    verified_at: datetime | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> LoadedWitnessedCheckpoint:
    """Load exact bytes and rerun verification under an out-of-band pinned policy."""

    if verifier is None:  # type: ignore[comparison-overlap]
        raise WitnessVerificationError('an independent trusted checkpoint witness verifier is required')
    try:
        expected_policy = _validated_model(
            expected_policy,
            WitnessPolicyBinding,
            'expected witness policy',
            error_type=WitnessVerificationError,
        )
    except TypeError as error:
        raise WitnessVerificationError(str(error)) from error
    checked_at = _aware(verified_at or _now_utc(), 'verified_at')
    resolved, root_descriptor = _open_artifact_root(root)
    try:
        _require_exact_inventory(root_descriptor)
        manifest_bytes = _read_regular_file_at(root_descriptor, _MANIFEST_PATH, _MAX_MODEL_BYTES)
        try:
            manifest = WitnessedCheckpointManifest.model_validate_json(manifest_bytes)
        except ValueError as error:
            raise WitnessVerificationError(f'invalid witnessed checkpoint manifest: {error}') from error
        if manifest_bytes != canonical_json_bytes(manifest):
            raise WitnessVerificationError('witnessed checkpoint manifest must use canonical JSON encoding')

        exact_checkpoint_bytes = _read_regular_file_at(
            root_descriptor,
            manifest.checkpoint_path,
            _MAX_MODEL_BYTES,
        )
        try:
            checkpoint = LedgerCheckpoint.model_validate_json(exact_checkpoint_bytes)
        except ValueError as error:
            raise WitnessVerificationError(f'invalid witnessed ledger checkpoint: {error}') from error
        if exact_checkpoint_bytes != checkpoint_bytes(checkpoint):
            raise WitnessVerificationError('witnessed ledger checkpoint must use canonical JSON encoding')
        if (
            hashlib.sha256(exact_checkpoint_bytes).hexdigest() != manifest.checkpoint_sha256
            or len(exact_checkpoint_bytes) != manifest.checkpoint_bytes
        ):
            raise WitnessVerificationError('witnessed checkpoint bytes do not match their manifest binding')
        if expected_checkpoint_sha256 is not None and manifest.checkpoint_sha256 != expected_checkpoint_sha256:
            raise WitnessVerificationError('witnessed checkpoint does not match the expected checkpoint digest')

        proof_bytes = _read_regular_file_at(root_descriptor, manifest.proof_path, _MAX_PROOF_BYTES)
        # Detect a concurrent add/remove and ensure the path still names the exact
        # directory descriptor whose files were verified.
        _require_exact_inventory(root_descriptor)
        _require_root_path_identity(resolved, root_descriptor)
    finally:
        os.close(root_descriptor)
    verify_witnessed_checkpoint(
        checkpoint,
        checkpoint_bytes=exact_checkpoint_bytes,
        manifest=manifest,
        proof_bytes=proof_bytes,
        verifier=verifier,
        expected_policy=expected_policy,
        verified_at=checked_at,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    return LoadedWitnessedCheckpoint(
        root=resolved,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_bytes=exact_checkpoint_bytes,
        proof_bytes=proof_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def verify_witnessed_checkpoint(
    checkpoint: LedgerCheckpoint,
    *,
    checkpoint_bytes: bytes,
    manifest: WitnessedCheckpointManifest,
    proof_bytes: bytes,
    verifier: TrustedCheckpointWitnessVerifier,
    expected_policy: WitnessPolicyBinding,
    verified_at: datetime | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> None:
    """Fail closed on bindings, temporal sanity, or trusted proof rejection."""

    if verifier is None:  # type: ignore[comparison-overlap]
        raise WitnessVerificationError('an independent trusted checkpoint witness verifier is required')
    try:
        expected_policy = _validated_model(
            expected_policy,
            WitnessPolicyBinding,
            'expected witness policy',
            error_type=WitnessVerificationError,
        )
    except TypeError as error:
        raise WitnessVerificationError(str(error)) from error
    checked_at = _aware(verified_at or _now_utc(), 'verified_at')
    checkpoint, canonical_checkpoint = _validated_checkpoint(
        checkpoint,
        error_type=WitnessVerificationError,
    )
    if not isinstance(checkpoint_bytes, bytes) or checkpoint_bytes != canonical_checkpoint:
        raise WitnessVerificationError('checkpoint_bytes must be the exact canonical checkpoint bytes')
    try:
        manifest = _validated_model(
            manifest,
            WitnessedCheckpointManifest,
            'witnessed checkpoint manifest',
            error_type=WitnessVerificationError,
        )
    except TypeError as error:
        raise WitnessVerificationError(str(error)) from error
    if not isinstance(proof_bytes, bytes) or not proof_bytes:
        raise WitnessVerificationError('external checkpoint witness proof must be nonempty exact bytes')
    if len(proof_bytes) > _MAX_PROOF_BYTES:
        raise WitnessVerificationError('external checkpoint witness proof exceeds the configured byte limit')
    digest = checkpoint_sha256(checkpoint)
    if manifest.checkpoint_sha256 != digest or manifest.checkpoint_bytes != len(canonical_checkpoint):
        raise WitnessVerificationError('external witness manifest does not bind the exact checkpoint bytes')
    if expected_checkpoint_sha256 is not None and digest != expected_checkpoint_sha256:
        raise WitnessVerificationError('witnessed checkpoint does not match the expected checkpoint digest')
    receipt = manifest.receipt
    if (
        receipt.authority_id != expected_policy.authority_id
        or receipt.method is not expected_policy.method
        or receipt.policy_id != expected_policy.policy_id
        or receipt.policy_sha256 != expected_policy.policy_sha256
        or receipt.trust_policy_id != expected_policy.trust_policy_id
        or receipt.trust_policy_sha256 != expected_policy.trust_policy_sha256
        or receipt.verifier_id != expected_policy.verifier_id
        or receipt.verifier_implementation_sha256 != expected_policy.verifier_implementation_sha256
    ):
        raise WitnessVerificationError('external witness receipt does not match the out-of-band pinned policy')
    if receipt.checkpoint_sha256 != digest or receipt.checkpoint_bytes != len(canonical_checkpoint):
        raise WitnessVerificationError('external witness receipt does not bind the exact checkpoint bytes')
    if receipt.proof_sha256 != hashlib.sha256(proof_bytes).hexdigest() or receipt.proof_bytes != len(proof_bytes):
        raise WitnessVerificationError('external witness proof bytes do not match their hash and size binding')
    facts = _authenticate_external_proof(
        checkpoint,
        checkpoint_bytes=canonical_checkpoint,
        proof_bytes=proof_bytes,
        verifier=verifier,
        expected_policy=expected_policy,
        verified_at=checked_at,
    )
    if (
        receipt.receipt_id != facts.receipt_id
        or receipt.authority_id != facts.authority_id
        or receipt.witness_id != facts.witness_id
        or receipt.method is not facts.method
        or receipt.policy_id != facts.policy_id
        or receipt.checkpoint_sha256 != facts.checkpoint_sha256
        or receipt.witnessed_at != facts.witnessed_at
    ):
        raise WitnessVerificationError(
            'external witness receipt does not match the authenticated facts parsed from the proof'
        )


def _authenticate_external_proof(
    checkpoint: LedgerCheckpoint,
    *,
    checkpoint_bytes: bytes,
    proof_bytes: bytes,
    verifier: TrustedCheckpointWitnessVerifier,
    expected_policy: WitnessPolicyBinding,
    verified_at: datetime,
) -> AuthenticatedExternalWitnessFacts:
    """Invoke trusted proof parsing and validate every authenticated security fact."""

    try:
        returned = verifier(checkpoint_bytes, proof_bytes, expected_policy)
    except Exception as error:
        raise WitnessVerificationError(f'trusted external checkpoint witness verifier failed: {error}') from error
    if returned is False or returned is None:  # type: ignore[comparison-overlap]
        raise WitnessVerificationError('trusted external checkpoint witness verifier rejected the proof')
    try:
        facts = _validated_model(
            returned,
            AuthenticatedExternalWitnessFacts,
            'authenticated external witness facts',
            error_type=WitnessVerificationError,
        )
    except TypeError as error:
        raise WitnessVerificationError(str(error)) from error
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    if facts.checkpoint_sha256 != digest:
        raise WitnessVerificationError('authenticated external witness proof binds a different checkpoint imprint')
    if (
        facts.authority_id != expected_policy.authority_id
        or facts.method is not expected_policy.method
        or facts.policy_id != expected_policy.policy_id
    ):
        raise WitnessVerificationError('authenticated external witness facts do not satisfy the pinned policy')
    if facts.witnessed_at < checkpoint.created_at:
        raise WitnessVerificationError('authenticated external witness time predates creation of the exact checkpoint')
    if facts.witnessed_at > verified_at:
        raise WitnessVerificationError('verification time predates the authenticated external witness')
    return facts


def _validated_checkpoint(
    checkpoint: LedgerCheckpoint,
    *,
    error_type: type[ValueError],
) -> tuple[LedgerCheckpoint, bytes]:
    try:
        payload = checkpoint_bytes(checkpoint)
        validated = LedgerCheckpoint.model_validate_json(payload)
    except (AttributeError, TypeError, ValueError) as error:
        raise error_type(f'invalid ledger checkpoint: {error}') from error
    exact = checkpoint_bytes(validated)
    if exact != payload:
        raise error_type('ledger checkpoint is not canonically reproducible')
    return validated, exact


def _validated_model[ModelT: StrictModel](
    value: object,
    model: type[ModelT],
    label: str,
    *,
    error_type: type[ValueError],
) -> ModelT:
    if not isinstance(value, model):
        raise TypeError(f'{label} must be a {model.__name__}')
    try:
        return model.model_validate_json(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise error_type(f'invalid {label}: {error}') from error


def _open_artifact_root(root: Path) -> tuple[Path, int]:
    # Walk the caller's normalized path itself. Resolving first would silently
    # erase a symlink in an intermediate parent and defeat ``O_NOFOLLOW`` on the
    # final component.
    requested = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(requested.anchor, flags)
        for component in requested.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise WitnessVerificationError(f'cannot open witnessed checkpoint root: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise WitnessVerificationError('witnessed checkpoint root must be a directory')
    except BaseException:
        os.close(descriptor)
        raise
    return requested, descriptor


def _require_exact_inventory(root_descriptor: int) -> None:
    expected = {_CHECKPOINT_PATH, _PROOF_PATH, _MANIFEST_PATH}
    try:
        names = tuple(os.listdir(root_descriptor))
    except OSError as error:
        raise WitnessVerificationError(f'cannot enumerate witnessed checkpoint artifact: {error}') from error
    if set(names) != expected or len(names) != len(expected):
        raise WitnessVerificationError('witnessed checkpoint artifact must contain exactly three declared files')
    for name in names:
        try:
            metadata = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        except OSError as error:
            raise WitnessVerificationError(f'cannot inspect witnessed checkpoint artifact file: {error}') from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise WitnessVerificationError('witnessed checkpoint artifact entries must be regular files')


def _require_root_path_identity(resolved: Path, root_descriptor: int) -> None:
    """Require the resolved path to still name the opened artifact directory."""

    opened = os.fstat(root_descriptor)
    try:
        _current_path, current_descriptor = _open_artifact_root(resolved)
    except WitnessVerificationError as error:
        raise WitnessVerificationError('witnessed checkpoint root changed while being read') from error
    try:
        current = os.fstat(current_descriptor)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise WitnessVerificationError('witnessed checkpoint root changed while being read')
    finally:
        os.close(current_descriptor)


def _read_regular_file_at(root_descriptor: int, name: str, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_descriptor)
    except OSError as error:
        raise WitnessVerificationError(f'cannot open witnessed checkpoint artifact file: {name}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise WitnessVerificationError('witnessed checkpoint artifact entries must be regular files')
        if metadata.st_size < 1 or metadata.st_size > max_bytes:
            raise WitnessVerificationError(f'witnessed checkpoint artifact file has invalid size: {name}')
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise WitnessVerificationError(f'witnessed checkpoint artifact file changed while read: {name}')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise WitnessVerificationError(f'witnessed checkpoint artifact file changed while read: {name}')
        after = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if after_identity != identity:
            raise WitnessVerificationError(f'witnessed checkpoint artifact file changed while read: {name}')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


def _write_durable_file(path: Path, payload: bytes) -> None:
    """Exclusively create, flush, and fsync one staging artifact."""

    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o644)
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _acquire_publication_lock(lock_path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(f'witnessed checkpoint publication is already locked: {lock_path}') from error
    try:
        os.fsync(descriptor)
        _fsync_directory(lock_path.parent)
    except BaseException:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        raise
    return descriptor


def _release_publication_lock(lock_path: Path, descriptor: int) -> None:
    """Release only the exact lock inode acquired by this process."""

    try:
        acquired = os.fstat(descriptor)
        try:
            current = lock_path.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (acquired.st_dev, acquired.st_ino):
            lock_path.unlink()
    finally:
        os.close(descriptor)
