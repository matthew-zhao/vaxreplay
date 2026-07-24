"""Durable signed transparency registry for pre-capture plan selection.

The registry is deliberately separate from :mod:`plan_selection`: that module
defines the evidence boundary, while this module is one production-shaped
implementation of it.  Assignments are serialized with ``BEGIN IMMEDIATE`` and
a SQLite ``UNIQUE(campaign_id, selection_key)`` constraint.  The immutable FWW
reservation is durably committed *before* the checkpoint clock is sampled, so
the later independently witnessed tree head is a real upper bound rather than a
pre-commit approximation.  The reservation is emitted as an RFC 6962 SHA-256
Merkle-tree leaf, its tree head is signed with an externally stored Ed25519
private key, and the exact signed envelope is submitted to a stateful independent
witness.  A crash between phases leaves the winning reservation intact and
retryable, never open for reassignment.

Only public keys, requests, log entries, checkpoints, and signatures enter the
database or proof.  Private signing keys and service bearer tokens are supplied
from separate files and are never persisted by this module.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import secrets
import sqlite3
import stat
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Self
from urllib.parse import quote

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.clock_health import ClockHealthGate, require_clock_health
from vaxreplay.operations.plan_selection import (
    AuthenticatedPlanSelectionFacts,
    PlanSelectionClaim,
    PlanSelectionCommitment,
    PlanSelectionIntegrityError,
    PlanSelectionMaterialSpec,
    PlanSelectionPolicyBinding,
    PlanSelectionRequest,
)
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc
from vaxreplay.operations.signing import Ed25519Signer, checked_signer
from vaxreplay.operations.witness import (
    SIGNED_PLAN_SELECTION_REGISTRY_CHECKPOINT_SCHEMA_VERSION,
    ExternalWitnessMethod,
    RegistryCheckpointWitnessProvider,
    RegistryCheckpointWitnessRequest,
)
from vaxreplay.operations.witness_service import verify_witness_service_artifact
from vaxreplay.operations.witness_service_schema import (
    WitnessServicePolicy,
    WitnessServiceProof,
    WitnessServiceTrustPolicy,
)

REGISTRY_POLICY_SCHEMA_VERSION = 'vaxreplay.plan-selection-registry-policy.v0.3'
REGISTRY_TRUST_POLICY_SCHEMA_VERSION = 'vaxreplay.plan-selection-registry-trust.v0.3'
REGISTRY_ENTRY_SCHEMA_VERSION = 'vaxreplay.plan-selection-registry-entry.v0.1'
REGISTRY_CHECKPOINT_SCHEMA_VERSION = 'vaxreplay.plan-selection-registry-checkpoint.v0.1'
SIGNED_REGISTRY_CHECKPOINT_SCHEMA_VERSION = SIGNED_PLAN_SELECTION_REGISTRY_CHECKPOINT_SCHEMA_VERSION
REGISTRY_PROOF_SCHEMA_VERSION = 'vaxreplay.plan-selection-registry-proof.v0.2'
REGISTRY_RESPONSE_SCHEMA_VERSION = 'vaxreplay.plan-selection-registry-response.v0.1'

_SHA256_PATTERN: Final = r'^[0-9a-f]{64}$'
_APPLICATION_ID: Final = 0x56585253  # "VXRS"
_DATABASE_VERSION: Final = 2
_EMPTY_TREE_HASH: Final = hashlib.sha256(b'').digest()
_MAX_PROOF_BYTES: Final = 64 * 1024 * 1024


class SelectionRegistryError(ValueError):
    """The registry, proof, signer, or trust material failed closed."""


class RegistryConflictError(SelectionRegistryError):
    """A stable campaign selection key was already assigned differently."""


class RegistryWitnessUnavailableError(SelectionRegistryError):
    """A durable assignment exists but its independent witness anchor is unavailable."""


class SelectionRegistryPolicy(StrictModel):
    """Public, hash-pinned service and log semantics."""

    schema_version: Literal['vaxreplay.plan-selection-registry-policy.v0.3'] = REGISTRY_POLICY_SCHEMA_VERSION
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    hash_algorithm: Literal['sha256'] = 'sha256'
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    merkle_tree_algorithm: Literal['rfc6962_sha256'] = 'rfc6962_sha256'
    uniqueness_semantics: Literal['sqlite_unique_atomic_first_write_wins'] = 'sqlite_unique_atomic_first_write_wins'
    finality_semantics: Literal['append_only_no_update_or_delete'] = 'append_only_no_update_or_delete'
    checkpoint_witness_required: Literal[True] = True
    checkpoint_witness_artifact_schema: Literal['vaxreplay.signed-plan-selection-registry-checkpoint.v0.1'] = (
        SIGNED_REGISTRY_CHECKPOINT_SCHEMA_VERSION
    )
    write_authentication: Literal['sha256_bearer_token'] = 'sha256_bearer_token'
    max_request_bytes: int = Field(default=64 * 1024, ge=1024, le=1024 * 1024)
    max_proof_bytes: int = Field(default=16 * 1024 * 1024, ge=4096, le=_MAX_PROOF_BYTES)
    clock_health_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    clock_health_process_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    external_signer_process_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_runtime_trust_bindings(self) -> Self:
        bindings = (
            self.clock_health_policy_sha256,
            self.clock_health_process_sha256,
            self.external_signer_process_sha256,
        )
        if any(value is None for value in bindings) and any(value is not None for value in bindings):
            raise ValueError('registry runtime-trust digests must be all present or all null')
        return self


class RegistryTrustedSigningKey(StrictModel):
    """One Ed25519 public key authorized for a bounded checkpoint interval."""

    key_id: str = Field(pattern=SAFE_ID_PATTERN)
    public_key_base64: str = Field(min_length=43, max_length=44)
    valid_from: datetime
    valid_until: datetime | None = None

    @field_validator('valid_from', 'valid_until')
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else aware_utc(value, 'signing-key validity timestamp')

    @field_validator('public_key_base64')
    @classmethod
    def validate_public_key(cls, value: str) -> str:
        if len(_decode_base64(value, 'Ed25519 public key')) != 32:
            raise ValueError('Ed25519 public key must contain exactly 32 bytes')
        return value

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError('signing-key valid_until must follow valid_from')
        return self


class RegistryPinnedCheckpoint(StrictModel):
    tree_size: int = Field(ge=0)
    root_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_checkpoint_base64: str = Field(min_length=1)
    witness_proof_base64: str = Field(min_length=1)

    @field_validator('signed_checkpoint_base64', 'witness_proof_base64')
    @classmethod
    def validate_anchor_base64(cls, value: str) -> str:
        decoded = _decode_base64(value, 'pinned checkpoint anchor')
        if not decoded or len(decoded) > _MAX_PROOF_BYTES:
            raise ValueError('pinned checkpoint anchor has an invalid decoded size')
        if _encode_base64(decoded) != value:
            raise ValueError('pinned checkpoint anchor must use canonical base64')
        return value

    @model_validator(mode='after')
    def validate_empty_root(self) -> Self:
        if self.tree_size == 0 and self.root_sha256 != _EMPTY_TREE_HASH.hex():
            raise ValueError('the size-zero pinned checkpoint must use the RFC 6962 empty-tree root')
        return self


class SelectionRegistryTrustPolicy(StrictModel):
    """Out-of-band trust anchor and accepted checkpoint key ring."""

    schema_version: Literal['vaxreplay.plan-selection-registry-trust.v0.3'] = REGISTRY_TRUST_POLICY_SCHEMA_VERSION
    trust_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    pinned_checkpoint: RegistryPinnedCheckpoint
    signing_keys: tuple[RegistryTrustedSigningKey, ...] = Field(min_length=1, max_length=64)
    checkpoint_witness_policy: WitnessServicePolicy
    checkpoint_witness_trust_policy: WitnessServiceTrustPolicy

    @model_validator(mode='after')
    def validate_unique_keys(self) -> Self:
        key_ids = [key.key_id for key in self.signing_keys]
        if len(set(key_ids)) != len(key_ids):
            raise ValueError('trust policy contains duplicate signing key IDs')
        witness_policy_bytes = canonical_json_bytes(self.checkpoint_witness_policy)
        witness_trust = self.checkpoint_witness_trust_policy
        if (
            witness_trust.authority_id != self.checkpoint_witness_policy.authority_id
            or witness_trust.witness_id != self.checkpoint_witness_policy.witness_id
            or witness_trust.service_policy_sha256 != hashlib.sha256(witness_policy_bytes).hexdigest()
        ):
            raise ValueError('checkpoint witness trust does not bind the exact witness policy')
        if self.checkpoint_witness_policy.authority_id == self.authority_id:
            raise ValueError('checkpoint witness must declare an authority independent of the registry')
        monitors = [
            monitor
            for monitor in self.checkpoint_witness_policy.registry_monitors
            if monitor.registry_id == self.registry_id and monitor.authority_id == self.authority_id
        ]
        if len(monitors) != 1:
            raise ValueError('checkpoint witness policy must uniquely monitor this registry identity')
        registry_keys = {
            (key.key_id, key.public_key_base64, key.valid_from, key.valid_until) for key in self.signing_keys
        }
        monitored_keys = {
            (key.key_id, key.public_key_base64, key.valid_from, key.valid_until) for key in monitors[0].signing_keys
        }
        if not registry_keys.issubset(monitored_keys):
            raise ValueError('registry trust contains a key absent from the checkpoint witness monitor')
        return self


class RegistryLogEntry(StrictModel):
    """Canonical append-only leaf for a first-write assignment."""

    schema_version: Literal['vaxreplay.plan-selection-registry-entry.v0.1'] = REGISTRY_ENTRY_SCHEMA_VERSION
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    registry_sequence: int = Field(ge=0)
    registry_entry_id: str = Field(pattern=SAFE_ID_PATTERN)
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=SAFE_ID_PATTERN)
    commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    commitment_bytes: int = Field(gt=0)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected_at_upper_bound: datetime = Field(
        description=(
            'Registry-local post-reservation clock reading; production verification derives the '
            'authoritative selection upper bound from the independent witness proof.'
        )
    )
    operation: Literal['assign_if_absent'] = 'assign_if_absent'
    key_previously_unassigned: Literal[True] = True
    selection_key_history_count: Literal[1] = 1

    @field_validator('selected_at_upper_bound')
    @classmethod
    def validate_selected_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'selected_at_upper_bound')


class RegistryCheckpoint(StrictModel):
    """Canonical tree head whose exact bytes are Ed25519-signed."""

    schema_version: Literal['vaxreplay.plan-selection-registry-checkpoint.v0.1'] = REGISTRY_CHECKPOINT_SCHEMA_VERSION
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    tree_size: int = Field(ge=0)
    root_sha256: str = Field(pattern=_SHA256_PATTERN)
    issued_at_upper_bound: datetime = Field(
        description=('Registry-declared checkpoint issuance time; not an independent Tier-A timestamp.')
    )
    signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    previous_checkpoint_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator('issued_at_upper_bound')
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'issued_at_upper_bound')

    @model_validator(mode='after')
    def validate_predecessor(self) -> Self:
        if self.tree_size == 0:
            if self.root_sha256 != _EMPTY_TREE_HASH.hex() or self.previous_checkpoint_sha256 is not None:
                raise ValueError('registry genesis must be the canonical empty-tree head')
        elif self.previous_checkpoint_sha256 is None:
            raise ValueError('non-genesis registry checkpoint must bind its exact predecessor')
        return self


class SignedRegistryCheckpoint(StrictModel):
    """Portable signed tree head suitable for an external witness monitor."""

    schema_version: Literal['vaxreplay.signed-plan-selection-registry-checkpoint.v0.1'] = (
        SIGNED_REGISTRY_CHECKPOINT_SCHEMA_VERSION
    )
    checkpoint: RegistryCheckpoint
    signature_base64: str = Field(min_length=86, max_length=88)

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        if len(_decode_base64(value, 'checkpoint signature')) != 64:
            raise ValueError('Ed25519 checkpoint signature must contain exactly 64 bytes')
        return value

    @property
    def signed_checkpoint_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.checkpoint) + _decode_base64(self.signature_base64, 'checkpoint signature')
        ).hexdigest()


class RegistrySelectionProof(StrictModel):
    """Raw portable proof returned by the registry service."""

    schema_version: Literal['vaxreplay.plan-selection-registry-proof.v0.2'] = REGISTRY_PROOF_SCHEMA_VERSION
    entry: RegistryLogEntry
    checkpoint: RegistryCheckpoint
    checkpoint_signature_base64: str = Field(min_length=86, max_length=88)
    checkpoint_witness_proof_base64: str = Field(min_length=1)
    inclusion_proof_sha256: tuple[str, ...] = Field(max_length=256)
    consistency_proof_sha256: tuple[str, ...] = Field(max_length=256)
    consistency_from_tree_size: int = Field(ge=0)
    consistency_from_root_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('checkpoint_signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        if len(_decode_base64(value, 'checkpoint signature')) != 64:
            raise ValueError('Ed25519 checkpoint signature must contain exactly 64 bytes')
        return value

    @field_validator('checkpoint_witness_proof_base64')
    @classmethod
    def validate_witness_proof(cls, value: str) -> str:
        decoded = _decode_base64(value, 'checkpoint witness proof')
        if not decoded or len(decoded) > _MAX_PROOF_BYTES:
            raise ValueError('checkpoint witness proof has an invalid decoded size')
        if _encode_base64(decoded) != value:
            raise ValueError('checkpoint witness proof must use canonical base64')
        return value

    @field_validator('inclusion_proof_sha256', 'consistency_proof_sha256')
    @classmethod
    def validate_hash_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(item) != 64 or any(character not in '0123456789abcdef' for character in item) for item in value):
            raise ValueError('Merkle proof paths must contain lowercase SHA-256 hex digests')
        return value

    @model_validator(mode='after')
    def validate_positions(self) -> Self:
        if self.entry.registry_sequence >= self.checkpoint.tree_size:
            raise ValueError('proof entry lies outside its checkpoint tree')
        if self.entry.selected_at_upper_bound != self.checkpoint.issued_at_upper_bound:
            raise ValueError('entry upper bound must be its first checkpoint issuance time')
        return self


class RegistrySelectionResponse(StrictModel):
    """Bounded JSON wire response; the decoded proof is the persisted raw proof."""

    schema_version: Literal['vaxreplay.plan-selection-registry-response.v0.1'] = REGISTRY_RESPONSE_SCHEMA_VERSION
    claim: PlanSelectionClaim
    proof_base64: str = Field(min_length=1)


def _decode_base64(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f'{label} must use canonical base64') from error


def _encode_base64(value: bytes) -> str:
    return base64.b64encode(value).decode('ascii')


def _leaf_hash(entry_bytes: bytes) -> bytes:
    return hashlib.sha256(b'\x00' + entry_bytes).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b'\x01' + left + right).digest()


def _largest_power_of_two_less_than(value: int) -> int:
    if value <= 1:
        raise ValueError('value must exceed one')
    return 1 << ((value - 1).bit_length() - 1)


def merkle_root(leaf_hashes: Sequence[bytes]) -> bytes:
    """Return the RFC 6962 Merkle Tree Hash for already-domain-separated leaves."""

    count = len(leaf_hashes)
    if count == 0:
        return _EMPTY_TREE_HASH
    if count == 1:
        leaf = leaf_hashes[0]
        if len(leaf) != 32:
            raise SelectionRegistryError('Merkle leaves must be SHA-256 digests')
        return leaf
    split = _largest_power_of_two_less_than(count)
    return _node_hash(merkle_root(leaf_hashes[:split]), merkle_root(leaf_hashes[split:]))


def inclusion_proof(leaf_hashes: Sequence[bytes], index: int) -> tuple[bytes, ...]:
    """Build the RFC 6962 audit path for ``index``."""

    if index < 0 or index >= len(leaf_hashes):
        raise SelectionRegistryError('Merkle inclusion index is outside the tree')

    def walk(leaves: Sequence[bytes], position: int) -> tuple[bytes, ...]:
        if len(leaves) == 1:
            return ()
        split = _largest_power_of_two_less_than(len(leaves))
        if position < split:
            return (*walk(leaves[:split], position), merkle_root(leaves[split:]))
        return (*walk(leaves[split:], position - split), merkle_root(leaves[:split]))

    return walk(leaf_hashes, index)


def verify_inclusion_proof(
    leaf: bytes,
    *,
    index: int,
    tree_size: int,
    proof: Sequence[bytes],
    expected_root: bytes,
) -> bool:
    """Verify an RFC 6962 inclusion proof without trusting provider metadata."""

    if len(leaf) != 32 or len(expected_root) != 32 or index < 0 or index >= tree_size:
        return False
    node = leaf
    leaf_index = index
    last_node = tree_size - 1
    for sibling in proof:
        if len(sibling) != 32 or last_node == 0:
            return False
        if leaf_index & 1 or leaf_index == last_node:
            node = _node_hash(sibling, node)
            while leaf_index != 0 and not (leaf_index & 1):
                leaf_index >>= 1
                last_node >>= 1
        else:
            node = _node_hash(node, sibling)
        leaf_index >>= 1
        last_node >>= 1
    return last_node == 0 and hmac.compare_digest(node, expected_root)


def consistency_proof(leaf_hashes: Sequence[bytes], old_size: int) -> tuple[bytes, ...]:
    """Build an RFC 6962 consistency proof from ``old_size`` to the full tree."""

    new_size = len(leaf_hashes)
    if old_size < 0 or old_size > new_size:
        raise SelectionRegistryError('pinned checkpoint size is outside the registry tree')
    if old_size == 0 or old_size == new_size:
        return ()

    def subproof(leaves: Sequence[bytes], prefix_size: int, complete: bool) -> tuple[bytes, ...]:
        if prefix_size == len(leaves):
            return () if complete else (merkle_root(leaves),)
        split = _largest_power_of_two_less_than(len(leaves))
        if prefix_size <= split:
            return (*subproof(leaves[:split], prefix_size, complete), merkle_root(leaves[split:]))
        return (*subproof(leaves[split:], prefix_size - split, False), merkle_root(leaves[:split]))

    return subproof(leaf_hashes, old_size, True)


def verify_consistency_proof(
    *,
    old_size: int,
    new_size: int,
    old_root: bytes,
    new_root: bytes,
    proof: Sequence[bytes],
) -> bool:
    """Verify RFC 6962 append-only consistency between two tree heads."""

    if old_size < 0 or new_size < old_size or len(old_root) != 32 or len(new_root) != 32:
        return False
    if old_size == 0:
        return not proof and hmac.compare_digest(old_root, _EMPTY_TREE_HASH)
    if old_size == new_size:
        return not proof and hmac.compare_digest(old_root, new_root)

    first = old_size - 1
    second = new_size - 1
    while first & 1:
        first >>= 1
        second >>= 1
    position = 0
    if first == 0:
        old_hash = old_root
        new_hash = old_root
    else:
        if not proof or len(proof[0]) != 32:
            return False
        old_hash = proof[0]
        new_hash = proof[0]
        position = 1
    for sibling in proof[position:]:
        if len(sibling) != 32 or second == 0:
            return False
        if first & 1 or first == second:
            old_hash = _node_hash(sibling, old_hash)
            new_hash = _node_hash(sibling, new_hash)
            while first != 0 and not (first & 1):
                first >>= 1
                second >>= 1
        else:
            new_hash = _node_hash(new_hash, sibling)
        first >>= 1
        second >>= 1
    return second == 0 and hmac.compare_digest(old_hash, old_root) and hmac.compare_digest(new_hash, new_root)


def generate_ed25519_private_key(path: Path) -> Path:
    """Generate one raw Ed25519 seed in an exclusive owner-only file."""

    target = _exclusive_secret_target(path)
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
        0o600,
    )
    try:
        view = memoryview(private_bytes)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short write while persisting registry private key')
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(target.parent)
    return target


def load_ed25519_private_key(path: Path) -> Ed25519PrivateKey:
    """Load an owner-only regular raw seed or unencrypted PKCS8 PEM file."""

    request = Path(path).expanduser().absolute()
    parent = request.parent.resolve(strict=True)
    _require_protected_parent(parent)
    source = parent / request.name
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise SelectionRegistryError('registry private key must be a regular file, not a link or device')
    if metadata.st_mode & 0o077:
        raise SelectionRegistryError('registry private key must not be accessible by group or other users')
    if metadata.st_size > 16 * 1024:
        raise SelectionRegistryError('registry private key file is unexpectedly large')
    payload = source.read_bytes()
    try:
        if len(payload) == 32:
            return Ed25519PrivateKey.from_private_bytes(payload)
        loaded = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as error:
        raise SelectionRegistryError('registry private key is not a valid raw or unencrypted PKCS8 key') from error
    if not isinstance(loaded, Ed25519PrivateKey):
        raise SelectionRegistryError('registry private key is not Ed25519')
    return loaded


def _signer_public_key_bytes(signing_key: Ed25519PrivateKey | Ed25519Signer) -> bytes:
    if isinstance(signing_key, Ed25519PrivateKey):
        return signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    return checked_signer(signing_key).public_key_bytes()


def ed25519_public_key_base64(private_key: Ed25519PrivateKey | Ed25519Signer) -> str:
    public_bytes = _signer_public_key_bytes(private_key)
    return _encode_base64(public_bytes)


def build_signed_registry_genesis_checkpoint(
    *,
    registry_policy: SelectionRegistryPolicy,
    signing_key: Ed25519PrivateKey | Ed25519Signer,
    signing_key_id: str,
    issued_at: datetime,
) -> SignedRegistryCheckpoint:
    """Create the exact signed empty-tree head that must be witnessed out of band.

    A new registry is initialized only from this independently witnessed genesis.
    Operators should subsequently refresh the out-of-band pin to a witnessed
    nonempty checkpoint before using the registry for a Tier-A release.
    """

    policy = _canonical_model(registry_policy, SelectionRegistryPolicy, 'registry policy')
    when = aware_utc(issued_at, 'genesis checkpoint issuance time')
    checkpoint = RegistryCheckpoint(
        registry_id=policy.registry_id,
        authority_id=policy.authority_id,
        tree_size=0,
        root_sha256=_EMPTY_TREE_HASH.hex(),
        issued_at_upper_bound=when,
        signing_key_id=signing_key_id,
        previous_checkpoint_sha256=None,
    )
    signature = signing_key.sign(canonical_json_bytes(checkpoint))
    return SignedRegistryCheckpoint(
        checkpoint=checkpoint,
        signature_base64=_encode_base64(signature),
    )


def checkpoint_witness_request(
    envelope_bytes: bytes,
    witness_policy: WitnessServicePolicy,
    *,
    consistency_from_tree_size: int,
    consistency_from_root_sha256: str,
    consistency_proof_sha256: tuple[str, ...],
) -> RegistryCheckpointWitnessRequest:
    """Build the exact independent-witness request for a signed registry head."""

    _canonical_bytes_model(envelope_bytes, SignedRegistryCheckpoint, 'signed registry checkpoint')
    witness_policy_bytes = canonical_json_bytes(witness_policy)
    return RegistryCheckpointWitnessRequest(
        checkpoint_sha256=hashlib.sha256(envelope_bytes).hexdigest(),
        checkpoint_bytes=len(envelope_bytes),
        signed_checkpoint_base64=_encode_base64(envelope_bytes),
        consistency_from_tree_size=consistency_from_tree_size,
        consistency_from_root_sha256=consistency_from_root_sha256,
        consistency_proof_sha256=consistency_proof_sha256,
        authority_id=witness_policy.authority_id,
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id=witness_policy.policy_id,
        policy_sha256=hashlib.sha256(witness_policy_bytes).hexdigest(),
    )


def selection_registry_verifier_implementation_bytes() -> bytes:
    """Bind every local source module used by the default offline verifier."""

    directory = Path(__file__).resolve(strict=True).parent
    sources = {}
    for name in ('selection_registry.py', 'witness.py', 'witness_service.py', 'witness_service_schema.py'):
        payload = (directory / name).read_bytes()
        sources[name] = hashlib.sha256(payload).hexdigest()
    return canonical_json_bytes({'algorithm': 'sha256', 'sources': sources})


def build_plan_selection_policy_binding(
    *,
    campaign_id: str,
    selection_key: str,
    registry_policy_bytes: bytes,
    trust_policy_bytes: bytes,
    verifier_implementation_bytes: bytes | None = None,
    verifier_id: str = 'vaxreplay-selection-registry-verifier-v1',
) -> PlanSelectionPolicyBinding:
    """Build a generic bridge policy from exact production registry materials."""

    policy = _canonical_bytes_model(registry_policy_bytes, SelectionRegistryPolicy, 'registry policy')
    trust = _canonical_bytes_model(trust_policy_bytes, SelectionRegistryTrustPolicy, 'registry trust policy')
    if policy.registry_id != trust.registry_id or policy.authority_id != trust.authority_id:
        raise SelectionRegistryError('registry policy and trust policy identify different authorities')
    _verify_pinned_checkpoint(trust)
    if trust.pinned_checkpoint.tree_size == 0:
        raise SelectionRegistryError(
            'production plan selection requires a witnessed nonempty checkpoint pinned out of band'
        )
    verifier_bytes = verifier_implementation_bytes or selection_registry_verifier_implementation_bytes()
    if not verifier_bytes:
        raise SelectionRegistryError('verifier implementation bytes must be nonempty')
    return PlanSelectionPolicyBinding(
        campaign_id=campaign_id,
        selection_key=selection_key,
        registry_id=policy.registry_id,
        authority_id=policy.authority_id,
        policy_id=policy.policy_id,
        policy_sha256=hashlib.sha256(registry_policy_bytes).hexdigest(),
        trust_policy_id=trust.trust_policy_id,
        trust_policy_sha256=hashlib.sha256(trust_policy_bytes).hexdigest(),
        verifier_id=verifier_id,
        verifier_implementation_sha256=hashlib.sha256(verifier_bytes).hexdigest(),
    )


def production_plan_selection_materials(
    *,
    binding: PlanSelectionPolicyBinding,
    registry_policy_bytes: bytes,
    trust_policy_bytes: bytes,
    verifier_implementation_bytes: bytes | None = None,
) -> PlanSelectionMaterialSpec:
    """Create bridge materials using this module's cryptographic verifier."""

    implementation = verifier_implementation_bytes or selection_registry_verifier_implementation_bytes()
    return PlanSelectionMaterialSpec(
        policy=binding,
        policy_bytes=registry_policy_bytes,
        trust_policy_bytes=trust_policy_bytes,
        verifier_implementation_bytes=implementation,
        verifier=verify_registry_selection,
    )


def _verify_witnessed_checkpoint_envelope(
    envelope_bytes: bytes,
    witness_proof_bytes: bytes,
    trust: SelectionRegistryTrustPolicy,
) -> tuple[SignedRegistryCheckpoint, datetime, int]:
    envelope = _canonical_bytes_model(
        envelope_bytes,
        SignedRegistryCheckpoint,
        'signed registry checkpoint',
    )
    checkpoint = envelope.checkpoint
    if checkpoint.registry_id != trust.registry_id or checkpoint.authority_id != trust.authority_id:
        raise SelectionRegistryError('signed checkpoint differs from the registry trust identity')
    trusted_key = _trusted_key(trust, checkpoint.signing_key_id)
    signature = _decode_base64(envelope.signature_base64, 'checkpoint signature')
    try:
        Ed25519PublicKey.from_public_bytes(_decode_base64(trusted_key.public_key_base64, 'Ed25519 public key')).verify(
            signature, canonical_json_bytes(checkpoint)
        )
    except (InvalidSignature, ValueError) as error:
        raise SelectionRegistryError('registry checkpoint has an invalid Ed25519 signature') from error
    witness_policy_bytes = canonical_json_bytes(trust.checkpoint_witness_policy)
    witness_trust_bytes = canonical_json_bytes(trust.checkpoint_witness_trust_policy)
    try:
        facts = verify_witness_service_artifact(
            envelope_bytes,
            witness_proof_bytes,
            policy_bytes=witness_policy_bytes,
            trust_policy_bytes=witness_trust_bytes,
            checkpoint_schema_version=SIGNED_PLAN_SELECTION_REGISTRY_CHECKPOINT_SCHEMA_VERSION,
        )
        witness_proof = _canonical_bytes_model(
            witness_proof_bytes,
            WitnessServiceProof,
            'checkpoint witness proof',
        )
    except (TypeError, ValueError) as error:
        raise SelectionRegistryError(f'independent checkpoint witness proof is invalid: {error}') from error
    if checkpoint.issued_at_upper_bound > facts.witnessed_at:
        raise SelectionRegistryError('registry checkpoint claims issuance after its independent witness time')
    _require_key_valid_at(trusted_key, facts.witnessed_at)
    return envelope, facts.witnessed_at, witness_proof.statement.entry.sequence


def _verify_pinned_checkpoint(
    trust: SelectionRegistryTrustPolicy,
) -> tuple[SignedRegistryCheckpoint, datetime, int]:
    pinned = trust.pinned_checkpoint
    envelope_bytes = _decode_base64(
        pinned.signed_checkpoint_base64,
        'pinned signed registry checkpoint',
    )
    witness_proof_bytes = _decode_base64(
        pinned.witness_proof_base64,
        'pinned checkpoint witness proof',
    )
    envelope, witnessed_at, witness_sequence = _verify_witnessed_checkpoint_envelope(
        envelope_bytes,
        witness_proof_bytes,
        trust,
    )
    checkpoint = envelope.checkpoint
    if checkpoint.tree_size != pinned.tree_size or checkpoint.root_sha256 != pinned.root_sha256:
        raise SelectionRegistryError('pinned checkpoint metadata differs from its signed witnessed envelope')
    if checkpoint.tree_size == 0 and (
        checkpoint.root_sha256 != _EMPTY_TREE_HASH.hex() or checkpoint.previous_checkpoint_sha256 is not None
    ):
        raise SelectionRegistryError('pinned genesis checkpoint is not the canonical empty-tree head')
    return envelope, witnessed_at, witness_sequence


class SQLitePlanSelectionRegistry:
    """Durable atomic first-write-wins selection registry.

    One instance uses exactly one active external signer.  Key rotation is
    performed by adding the next public key to the append-only key table and
    restarting the service with that key's external private-key file.  Both
    public keys must also be authorized in the independently distributed trust
    policy for their non-overlapping validity intervals.
    """

    def __init__(
        self,
        database_path: Path,
        *,
        signing_key: Ed25519PrivateKey | Ed25519Signer,
        signing_key_id: str,
        registry_policy_bytes: bytes,
        trust_policy_bytes: bytes,
        public_base_url: str,
        checkpoint_witness_provider: RegistryCheckpointWitnessProvider | None = None,
        clock: Callable[[], datetime] | None = None,
        clock_health_gate: ClockHealthGate | None = None,
    ) -> None:
        if not isinstance(signing_key, Ed25519PrivateKey) and clock_health_gate is None:
            raise SelectionRegistryError('isolated registry signers require a fail-closed clock-health gate')
        self.database_path = _existing_database_path(database_path)
        self.policy_bytes = registry_policy_bytes
        self.trust_policy_bytes = trust_policy_bytes
        self.policy = _canonical_bytes_model(registry_policy_bytes, SelectionRegistryPolicy, 'registry policy')
        self.trust_policy = _canonical_bytes_model(
            trust_policy_bytes,
            SelectionRegistryTrustPolicy,
            'registry trust policy',
        )
        if (
            self.policy.registry_id != self.trust_policy.registry_id
            or self.policy.authority_id != self.trust_policy.authority_id
        ):
            raise SelectionRegistryError('registry policy and trust policy identify different authorities')
        _verify_pinned_checkpoint(self.trust_policy)
        if not public_base_url.startswith('https://') or public_base_url.endswith('/'):
            raise SelectionRegistryError('public_base_url must be an HTTPS origin or base path without trailing slash')
        self.public_base_url = public_base_url
        self.signing_key = signing_key
        self.signing_key_id = signing_key_id
        self.checkpoint_witness_provider = checkpoint_witness_provider
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.clock_health_gate = clock_health_gate
        self._validate_database_identity_and_signer()

    @classmethod
    def initialize(
        cls,
        database_path: Path,
        *,
        signing_key: Ed25519PrivateKey | Ed25519Signer,
        signing_key_id: str,
        registry_policy_bytes: bytes,
        trust_policy_bytes: bytes,
        public_base_url: str,
        checkpoint_witness_provider: RegistryCheckpointWitnessProvider | None = None,
        clock: Callable[[], datetime] | None = None,
        clock_health_gate: ClockHealthGate | None = None,
    ) -> SQLitePlanSelectionRegistry:
        """Create an exclusive registry database and append its first public key."""

        if not isinstance(signing_key, Ed25519PrivateKey) and clock_health_gate is None:
            raise SelectionRegistryError('isolated registry signers require a fail-closed clock-health gate')
        policy = _canonical_bytes_model(registry_policy_bytes, SelectionRegistryPolicy, 'registry policy')
        trust = _canonical_bytes_model(trust_policy_bytes, SelectionRegistryTrustPolicy, 'registry trust policy')
        if policy.registry_id != trust.registry_id or policy.authority_id != trust.authority_id:
            raise SelectionRegistryError('registry policy and trust policy identify different authorities')
        pinned_envelope, _pinned_witnessed_at, _pinned_witness_sequence = _verify_pinned_checkpoint(trust)
        if pinned_envelope.checkpoint.tree_size != 0:
            raise SelectionRegistryError('a new registry must initialize from a witnessed empty-tree genesis')
        if pinned_envelope.checkpoint.signing_key_id != signing_key_id:
            raise SelectionRegistryError('genesis checkpoint must use the active initialization signing key')
        target = _new_database_path(database_path)
        connection = sqlite3.connect(target, isolation_level=None)
        try:
            _configure_connection(connection, initializing=True)
            connection.executescript(_DATABASE_SCHEMA)
            connection.execute(f'PRAGMA application_id = {_APPLICATION_ID}')
            connection.execute(f'PRAGMA user_version = {_DATABASE_VERSION}')
            connection.execute(
                'INSERT INTO registry_metadata(singleton, registry_id, authority_id, policy_sha256) '
                'VALUES (1, ?, ?, ?)',
                (policy.registry_id, policy.authority_id, hashlib.sha256(registry_policy_bytes).hexdigest()),
            )
            public_key = ed25519_public_key_base64(signing_key)
            trusted = _trusted_key(trust, signing_key_id)
            if not hmac.compare_digest(public_key, trusted.public_key_base64):
                raise SelectionRegistryError('initial signer does not match its trust-policy public key')
            connection.execute(
                'INSERT INTO registry_signing_keys(key_id, public_key_base64, registered_at) VALUES (?, ?, ?)',
                (signing_key_id, public_key, _utc_text(datetime.now(timezone.utc))),
            )
            genesis_checkpoint_bytes = canonical_json_bytes(pinned_envelope.checkpoint)
            genesis_signature = _decode_base64(
                pinned_envelope.signature_base64,
                'genesis checkpoint signature',
            )
            connection.execute(
                'INSERT INTO registry_checkpoints('
                'tree_size, root_sha256, issued_at, signing_key_id, checkpoint_bytes, signature'
                ') VALUES (?, ?, ?, ?, ?, ?)',
                (
                    0,
                    pinned_envelope.checkpoint.root_sha256,
                    _utc_text(pinned_envelope.checkpoint.issued_at_upper_bound),
                    pinned_envelope.checkpoint.signing_key_id,
                    genesis_checkpoint_bytes,
                    genesis_signature,
                ),
            )
            connection.execute(
                'INSERT INTO registry_checkpoint_witnesses('
                'tree_size, envelope_sha256, witness_receipt_id, witnessed_at, witness_sequence, proof_bytes'
                ') VALUES (?, ?, ?, ?, ?, ?)',
                (
                    0,
                    hashlib.sha256(canonical_json_bytes(pinned_envelope)).hexdigest(),
                    WitnessServiceProof.model_validate_json(
                        _decode_base64(
                            trust.pinned_checkpoint.witness_proof_base64,
                            'genesis checkpoint witness proof',
                        )
                    ).receipt_id,
                    _utc_text(_pinned_witnessed_at),
                    _pinned_witness_sequence,
                    _decode_base64(
                        trust.pinned_checkpoint.witness_proof_base64,
                        'genesis checkpoint witness proof',
                    ),
                ),
            )
            connection.commit()
            connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        except Exception:
            connection.close()
            for suffix in ('', '-wal', '-shm'):
                try:
                    Path(f'{target}{suffix}').unlink()
                except FileNotFoundError:
                    pass
            raise
        finally:
            try:
                connection.close()
            except sqlite3.Error:
                pass
        os.chmod(target, 0o600)
        fsync_directory(target.parent)
        return cls(
            target,
            signing_key=signing_key,
            signing_key_id=signing_key_id,
            registry_policy_bytes=registry_policy_bytes,
            trust_policy_bytes=trust_policy_bytes,
            public_base_url=public_base_url,
            checkpoint_witness_provider=checkpoint_witness_provider,
            clock=clock,
            clock_health_gate=clock_health_gate,
        )

    def register_signing_key(self, *, key_id: str, public_key_base64: str, registered_at: datetime) -> None:
        """Append a rotation public key; private key activation requires service restart."""

        trusted = _trusted_key(self.trust_policy, key_id)
        raw = _decode_base64(public_key_base64, 'Ed25519 public key')
        if len(raw) != 32 or not hmac.compare_digest(public_key_base64, trusted.public_key_base64):
            raise SelectionRegistryError('rotation public key is not the trusted key-ring value')
        when = aware_utc(registered_at, 'registered_at')
        with self._connection() as connection:
            try:
                connection.execute('BEGIN IMMEDIATE')
                connection.execute(
                    'INSERT INTO registry_signing_keys(key_id, public_key_base64, registered_at) VALUES (?, ?, ?)',
                    (key_id, public_key_base64, _utc_text(when)),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def provider(self, request: PlanSelectionRequest) -> tuple[PlanSelectionClaim, bytes]:
        """``PlanSelectionProvider``-compatible atomic assignment operation."""

        request = _canonical_model(request, PlanSelectionRequest, 'plan-selection request')
        self._validate_request(request)
        proof = self.assign(request)
        uri = self._verification_uri(request.campaign_id, request.selection_key)
        return PlanSelectionClaim(verification_uri=uri), proof

    def assign(self, request: PlanSelectionRequest) -> bytes:
        """Atomically assign a key or replay the immutable identical assignment."""

        request = _canonical_model(request, PlanSelectionRequest, 'plan-selection request')
        self._validate_request(request)
        request_bytes = canonical_json_bytes(request)
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        with self._connection() as connection:
            try:
                connection.execute('BEGIN IMMEDIATE')
                assignment = connection.execute(
                    'SELECT assignment_sequence, commitment_sha256, commitment_bytes, request_sha256 '
                    'FROM registry_assignments '
                    'WHERE campaign_id = ? AND selection_key = ?',
                    (request.campaign_id, request.selection_key),
                ).fetchone()
                if assignment is not None:
                    if (
                        str(assignment[1]) != request.commitment_sha256
                        or int(assignment[2]) != request.commitment_bytes
                        or str(assignment[3]) != request_sha256
                    ):
                        raise RegistryConflictError('selection key is already immutably assigned to another commitment')
                    assignment_sequence = int(assignment[0])
                else:
                    assignment_sequence = int(
                        connection.execute('SELECT COUNT(*) FROM registry_assignments').fetchone()[0]
                    )
                    connection.execute(
                        'INSERT INTO registry_assignments('
                        'assignment_sequence, campaign_id, selection_key, commitment_sha256, '
                        'commitment_bytes, request_sha256, policy_id, policy_sha256, reserved_at'
                        ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (
                            assignment_sequence,
                            request.campaign_id,
                            request.selection_key,
                            request.commitment_sha256,
                            request.commitment_bytes,
                            request_sha256,
                            request.policy_id,
                            request.policy_sha256,
                            _utc_text(datetime.now(timezone.utc)),
                        ),
                    )
                # This commit is the first-write-wins selection event.  It is
                # deliberately durable before checkpoint signing and independent
                # witnessing, so either later failure is retryable without
                # reopening the key for assignment.
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise RegistryConflictError('selection key was concurrently assigned') from error
            except Exception:
                connection.rollback()
                raise
            try:
                connection.execute('BEGIN IMMEDIATE')
                existing_entry = connection.execute(
                    'SELECT first_checkpoint_size FROM registry_entries WHERE assignment_sequence = ?',
                    (assignment_sequence,),
                ).fetchone()
                if existing_entry is None:
                    checkpoint_size = self._append_assignment(
                        connection,
                        request,
                        request_sha256,
                        assignment_sequence=assignment_sequence,
                    )
                else:
                    checkpoint_size = int(existing_entry[0])
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        # Network I/O is never performed inside the selection or append
        # transaction.  If this fails, the reservation and signed tree head stay
        # durable and an identical retry resumes at this step.
        self.anchor_checkpoint(checkpoint_size)
        return self.proof_for(request.campaign_id, request.selection_key)

    def anchor_checkpoint(self, checkpoint_size: int) -> bytes:
        """Ensure one signed tree head has a durable independent witness proof."""

        with self._connection() as connection:
            stored = connection.execute(
                'SELECT proof_bytes FROM registry_checkpoint_witnesses WHERE tree_size = ?',
                (checkpoint_size,),
            ).fetchone()
            envelope_bytes = self._signed_checkpoint_bytes(connection, checkpoint_size)
        if stored is not None:
            proof_bytes = bytes(stored[0])
            self._validate_checkpoint_anchor(envelope_bytes, proof_bytes)
            return proof_bytes
        if self.checkpoint_witness_provider is None:
            raise RegistryWitnessUnavailableError(
                'signed registry checkpoint is durable but no independent witness provider is configured'
            )
        previous_size = checkpoint_size - 1
        if previous_size < 0:
            raise RegistryWitnessUnavailableError('registry genesis must be anchored during initialization')
        with self._connection() as connection:
            previous_anchor = connection.execute(
                'SELECT 1 FROM registry_checkpoint_witnesses WHERE tree_size = ?',
                (previous_size,),
            ).fetchone()
        if previous_anchor is None:
            self.anchor_checkpoint(previous_size)
        with self._connection() as connection:
            leaves = self._leaf_hashes(connection, through_size=checkpoint_size)
        previous_root = merkle_root(leaves[:previous_size]).hex()
        request = checkpoint_witness_request(
            envelope_bytes,
            self.trust_policy.checkpoint_witness_policy,
            consistency_from_tree_size=previous_size,
            consistency_from_root_sha256=previous_root,
            consistency_proof_sha256=tuple(digest.hex() for digest in consistency_proof(leaves, previous_size)),
        )
        try:
            response = self.checkpoint_witness_provider(request)
        except Exception as error:
            raise RegistryWitnessUnavailableError(
                f'independent checkpoint witness failed after durable selection: {error}'
            ) from error
        if not isinstance(response, tuple) or len(response) != 2 or not isinstance(response[1], bytes):
            raise RegistryWitnessUnavailableError('independent checkpoint witness returned an invalid response')
        witness_proof_bytes = response[1]
        witnessed_at, witness_sequence, receipt_id = self._validate_checkpoint_anchor(
            envelope_bytes,
            witness_proof_bytes,
        )
        with self._connection() as connection:
            try:
                connection.execute('BEGIN IMMEDIATE')
                existing = connection.execute(
                    'SELECT proof_bytes FROM registry_checkpoint_witnesses WHERE tree_size = ?',
                    (checkpoint_size,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        'INSERT INTO registry_checkpoint_witnesses('
                        'tree_size, envelope_sha256, witness_receipt_id, witnessed_at, '
                        'witness_sequence, proof_bytes) VALUES (?, ?, ?, ?, ?, ?)',
                        (
                            checkpoint_size,
                            hashlib.sha256(envelope_bytes).hexdigest(),
                            receipt_id,
                            _utc_text(witnessed_at),
                            witness_sequence,
                            witness_proof_bytes,
                        ),
                    )
                    selected_proof = witness_proof_bytes
                else:
                    selected_proof = bytes(existing[0])
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self._validate_checkpoint_anchor(envelope_bytes, selected_proof)
        return selected_proof

    def _validate_checkpoint_anchor(
        self,
        envelope_bytes: bytes,
        proof_bytes: bytes,
    ) -> tuple[datetime, int, str]:
        _envelope, witnessed_at, witness_sequence = _verify_witnessed_checkpoint_envelope(
            envelope_bytes,
            proof_bytes,
            self.trust_policy,
        )
        _pinned_envelope, pinned_witnessed_at, pinned_witness_sequence = _verify_pinned_checkpoint(self.trust_policy)
        if witnessed_at < pinned_witnessed_at or witness_sequence < pinned_witness_sequence:
            raise RegistryWitnessUnavailableError(
                'checkpoint witness proof predates the out-of-band pinned witnessed checkpoint'
            )
        parsed = _canonical_bytes_model(proof_bytes, WitnessServiceProof, 'checkpoint witness proof')
        return witnessed_at, witness_sequence, parsed.receipt_id

    def proof_for(self, campaign_id: str, selection_key: str) -> bytes:
        """Return public proof bytes for an existing immutable assignment."""

        with self._connection() as connection:
            row = connection.execute(
                'SELECT first_checkpoint_size FROM registry_entries WHERE campaign_id = ? AND selection_key = ?',
                (campaign_id, selection_key),
            ).fetchone()
            if row is None:
                raise KeyError((campaign_id, selection_key))
            proof = self._proof_for(connection, campaign_id, selection_key, int(row[0]))
            if len(proof) > self.policy.max_proof_bytes:
                raise SelectionRegistryError('registry proof exceeds the configured response byte limit')
            return proof

    def registry_entry_bytes(self, sequence: int) -> bytes:
        """Return one exact canonical immutable leaf for public enumeration."""

        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise SelectionRegistryError('registry entry sequence must be a nonnegative integer')
        with self._connection() as connection:
            row = connection.execute(
                'SELECT entry_bytes FROM registry_entries WHERE sequence = ?',
                (sequence,),
            ).fetchone()
        if row is None:
            raise KeyError(sequence)
        entry_bytes = bytes(row[0])
        entry = _canonical_bytes_model(entry_bytes, RegistryLogEntry, 'registry entry')
        if entry.registry_sequence != sequence:
            raise SelectionRegistryError('registry entry sequence differs from its public log position')
        return entry_bytes

    def tree_head(self) -> RegistryCheckpoint | None:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT checkpoint_bytes FROM registry_checkpoints ORDER BY tree_size DESC LIMIT 1'
            ).fetchone()
        return None if row is None else _canonical_bytes_model(bytes(row[0]), RegistryCheckpoint, 'checkpoint')

    def signed_tree_head(self) -> SignedRegistryCheckpoint | None:
        """Return the current signed envelope for independent witnessing/gossip."""

        with self._connection() as connection:
            row = connection.execute(
                'SELECT checkpoint_bytes, signature FROM registry_checkpoints ORDER BY tree_size DESC LIMIT 1'
            ).fetchone()
        if row is None:
            return None
        return SignedRegistryCheckpoint(
            checkpoint=_canonical_bytes_model(bytes(row[0]), RegistryCheckpoint, 'checkpoint'),
            signature_base64=_encode_base64(bytes(row[1])),
        )

    def signed_checkpoint_and_witness(self, tree_size: int) -> tuple[bytes, bytes]:
        """Export one exact signed envelope and its already durable witness proof."""

        with self._connection() as connection:
            envelope_bytes = self._signed_checkpoint_bytes(connection, tree_size)
            row = connection.execute(
                'SELECT proof_bytes FROM registry_checkpoint_witnesses WHERE tree_size = ?',
                (tree_size,),
            ).fetchone()
        if row is None:
            raise RegistryWitnessUnavailableError('registry checkpoint has not been independently witnessed')
        proof_bytes = bytes(row[0])
        _verify_witnessed_checkpoint_envelope(envelope_bytes, proof_bytes, self.trust_policy)
        return envelope_bytes, proof_bytes

    def _signed_checkpoint_bytes(
        self,
        connection: sqlite3.Connection,
        tree_size: int,
    ) -> bytes:
        row = connection.execute(
            'SELECT checkpoint_bytes, signature FROM registry_checkpoints WHERE tree_size = ?',
            (tree_size,),
        ).fetchone()
        if row is None:
            raise SelectionRegistryError('registry checkpoint does not exist')
        envelope = SignedRegistryCheckpoint(
            checkpoint=_canonical_bytes_model(bytes(row[0]), RegistryCheckpoint, 'checkpoint'),
            signature_base64=_encode_base64(bytes(row[1])),
        )
        return canonical_json_bytes(envelope)

    def _append_assignment(
        self,
        connection: sqlite3.Connection,
        request: PlanSelectionRequest,
        request_sha256: str,
        *,
        assignment_sequence: int,
    ) -> int:
        sequence = int(connection.execute('SELECT COUNT(*) FROM registry_entries').fetchone()[0])
        issued_at = aware_utc(self.clock(), 'registry clock')
        require_clock_health(self.clock_health_gate, security_time=issued_at)
        trusted_key = _trusted_key(self.trust_policy, self.signing_key_id)
        _require_key_valid_at(trusted_key, issued_at)
        entry = RegistryLogEntry(
            registry_id=self.policy.registry_id,
            authority_id=self.policy.authority_id,
            registry_sequence=sequence,
            registry_entry_id=f'entry-{sequence:020d}-{secrets.token_hex(8)}',
            campaign_id=request.campaign_id,
            selection_key=request.selection_key,
            commitment_sha256=request.commitment_sha256,
            commitment_bytes=request.commitment_bytes,
            request_sha256=request_sha256,
            policy_id=request.policy_id,
            policy_sha256=request.policy_sha256,
            selected_at_upper_bound=issued_at,
        )
        entry_bytes = canonical_json_bytes(entry)
        leaf_sha256 = _leaf_hash(entry_bytes).hex()
        first_checkpoint_size = sequence + 1
        try:
            connection.execute(
                'INSERT INTO registry_entries('
                'sequence, assignment_sequence, registry_entry_id, campaign_id, selection_key, commitment_sha256, '
                'request_sha256, selected_at, leaf_sha256, entry_bytes, first_checkpoint_size'
                ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    sequence,
                    assignment_sequence,
                    entry.registry_entry_id,
                    entry.campaign_id,
                    entry.selection_key,
                    entry.commitment_sha256,
                    entry.request_sha256,
                    _utc_text(issued_at),
                    leaf_sha256,
                    entry_bytes,
                    first_checkpoint_size,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise RegistryConflictError('selection key was concurrently assigned') from error
        leaves = self._leaf_hashes(connection, through_size=first_checkpoint_size)
        root = merkle_root(leaves).hex()
        previous = connection.execute(
            'SELECT checkpoint_bytes, signature FROM registry_checkpoints ORDER BY tree_size DESC LIMIT 1'
        ).fetchone()
        previous_sha256 = None
        if previous is not None:
            previous_checkpoint = _canonical_bytes_model(
                bytes(previous[0]),
                RegistryCheckpoint,
                'previous registry checkpoint',
            )
            if issued_at < previous_checkpoint.issued_at_upper_bound:
                raise SelectionRegistryError('registry clock regressed behind the previous signed checkpoint')
            previous_sha256 = hashlib.sha256(bytes(previous[0]) + bytes(previous[1])).hexdigest()
        checkpoint = RegistryCheckpoint(
            registry_id=self.policy.registry_id,
            authority_id=self.policy.authority_id,
            tree_size=first_checkpoint_size,
            root_sha256=root,
            issued_at_upper_bound=issued_at,
            signing_key_id=self.signing_key_id,
            previous_checkpoint_sha256=previous_sha256,
        )
        checkpoint_bytes = canonical_json_bytes(checkpoint)
        signature = self.signing_key.sign(checkpoint_bytes)
        connection.execute(
            'INSERT INTO registry_checkpoints('
            'tree_size, root_sha256, issued_at, signing_key_id, checkpoint_bytes, signature'
            ') VALUES (?, ?, ?, ?, ?, ?)',
            (
                first_checkpoint_size,
                root,
                _utc_text(issued_at),
                self.signing_key_id,
                checkpoint_bytes,
                signature,
            ),
        )
        return first_checkpoint_size

    def _proof_for(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
        selection_key: str,
        checkpoint_size: int,
    ) -> bytes:
        row = connection.execute(
            'SELECT entry_bytes FROM registry_entries WHERE campaign_id = ? AND selection_key = ?',
            (campaign_id, selection_key),
        ).fetchone()
        if row is None:
            raise KeyError((campaign_id, selection_key))
        entry = _canonical_bytes_model(bytes(row[0]), RegistryLogEntry, 'registry entry')
        checkpoint_row = connection.execute(
            'SELECT checkpoint_bytes, signature FROM registry_checkpoints WHERE tree_size = ?',
            (checkpoint_size,),
        ).fetchone()
        if checkpoint_row is None:
            raise SelectionRegistryError('registry entry is missing its first signed checkpoint')
        witness_row = connection.execute(
            'SELECT proof_bytes FROM registry_checkpoint_witnesses WHERE tree_size = ?',
            (checkpoint_size,),
        ).fetchone()
        if witness_row is None:
            raise RegistryWitnessUnavailableError('registry entry checkpoint has not been independently witnessed')
        checkpoint = _canonical_bytes_model(bytes(checkpoint_row[0]), RegistryCheckpoint, 'registry checkpoint')
        envelope_bytes = canonical_json_bytes(
            SignedRegistryCheckpoint(
                checkpoint=checkpoint,
                signature_base64=_encode_base64(bytes(checkpoint_row[1])),
            )
        )
        self._validate_checkpoint_anchor(envelope_bytes, bytes(witness_row[0]))
        leaves = self._leaf_hashes(connection, through_size=checkpoint.tree_size)
        pinned = self.trust_policy.pinned_checkpoint
        if pinned.tree_size > checkpoint.tree_size:
            raise SelectionRegistryError('trust-policy checkpoint is newer than the assignment checkpoint')
        if merkle_root(leaves[: pinned.tree_size]).hex() != pinned.root_sha256:
            raise SelectionRegistryError('local log is inconsistent with the pinned trust checkpoint')
        proof = RegistrySelectionProof(
            entry=entry,
            checkpoint=checkpoint,
            checkpoint_signature_base64=_encode_base64(bytes(checkpoint_row[1])),
            checkpoint_witness_proof_base64=_encode_base64(bytes(witness_row[0])),
            inclusion_proof_sha256=tuple(digest.hex() for digest in inclusion_proof(leaves, entry.registry_sequence)),
            consistency_proof_sha256=tuple(digest.hex() for digest in consistency_proof(leaves, pinned.tree_size)),
            consistency_from_tree_size=pinned.tree_size,
            consistency_from_root_sha256=pinned.root_sha256,
        )
        return canonical_json_bytes(proof)

    def _leaf_hashes(self, connection: sqlite3.Connection, *, through_size: int) -> tuple[bytes, ...]:
        rows = connection.execute(
            'SELECT sequence, leaf_sha256, entry_bytes FROM registry_entries WHERE sequence < ? ORDER BY sequence',
            (through_size,),
        ).fetchall()
        if len(rows) != through_size:
            raise SelectionRegistryError('registry append-only sequence has a gap')
        leaves: list[bytes] = []
        for expected, row in enumerate(rows):
            sequence = int(row[0])
            entry_bytes = bytes(row[2])
            if sequence != expected:
                raise SelectionRegistryError('registry append-only sequence is non-contiguous')
            leaf = _leaf_hash(entry_bytes)
            if leaf.hex() != str(row[1]):
                raise SelectionRegistryError('registry entry differs from its stored Merkle leaf')
            entry = _canonical_bytes_model(entry_bytes, RegistryLogEntry, 'registry entry')
            if entry.registry_sequence != sequence:
                raise SelectionRegistryError('registry entry sequence differs from its database position')
            leaves.append(leaf)
        return tuple(leaves)

    def _verification_uri(self, campaign_id: str, selection_key: str) -> str:
        return f'{self.public_base_url}/v1/proofs/{quote(campaign_id, safe="")}/{quote(selection_key, safe="")}'

    def _validate_request(self, request: PlanSelectionRequest) -> None:
        if len(canonical_json_bytes(request)) > self.policy.max_request_bytes:
            raise SelectionRegistryError('plan-selection request exceeds registry policy byte limit')
        if (
            request.registry_id != self.policy.registry_id
            or request.authority_id != self.policy.authority_id
            or request.policy_id != self.policy.policy_id
            or request.policy_sha256 != hashlib.sha256(self.policy_bytes).hexdigest()
        ):
            raise SelectionRegistryError('request differs from the registry service policy')

    def _validate_database_identity_and_signer(self) -> None:
        with self._connection() as connection:
            application_id = int(connection.execute('PRAGMA application_id').fetchone()[0])
            version = int(connection.execute('PRAGMA user_version').fetchone()[0])
            if application_id != _APPLICATION_ID or version != _DATABASE_VERSION:
                raise SelectionRegistryError('database is not the expected registry schema version')
            row = connection.execute(
                'SELECT registry_id, authority_id, policy_sha256 FROM registry_metadata WHERE singleton = 1'
            ).fetchone()
            if row is None or tuple(map(str, row)) != (
                self.policy.registry_id,
                self.policy.authority_id,
                hashlib.sha256(self.policy_bytes).hexdigest(),
            ):
                raise SelectionRegistryError('database identity differs from the registry policy')
            key_row = connection.execute(
                'SELECT public_key_base64 FROM registry_signing_keys WHERE key_id = ?',
                (self.signing_key_id,),
            ).fetchone()
            if key_row is None:
                raise SelectionRegistryError('active signer public key is not registered')
            expected_public = ed25519_public_key_base64(self.signing_key)
            if not hmac.compare_digest(str(key_row[0]), expected_public):
                raise SelectionRegistryError('external private key differs from registered public key')
            trusted = _trusted_key(self.trust_policy, self.signing_key_id)
            if not hmac.compare_digest(trusted.public_key_base64, expected_public):
                raise SelectionRegistryError('external private key is absent from the trusted key ring')
            quick_check = str(connection.execute('PRAGMA quick_check').fetchone()[0])
            if quick_check != 'ok':
                raise SelectionRegistryError(f'SQLite quick_check failed: {quick_check}')
            pinned = self.trust_policy.pinned_checkpoint
            leaves = self._leaf_hashes(connection, through_size=pinned.tree_size)
            if merkle_root(leaves).hex() != pinned.root_sha256:
                raise SelectionRegistryError('database log differs from the pinned trust checkpoint')
            pinned_checkpoint_row = connection.execute(
                'SELECT checkpoint_bytes, signature FROM registry_checkpoints WHERE tree_size = ?',
                (pinned.tree_size,),
            ).fetchone()
            pinned_witness_row = connection.execute(
                'SELECT proof_bytes FROM registry_checkpoint_witnesses WHERE tree_size = ?',
                (pinned.tree_size,),
            ).fetchone()
            if pinned_checkpoint_row is None or pinned_witness_row is None:
                raise SelectionRegistryError('database is missing the pinned witnessed checkpoint')
            stored_envelope = canonical_json_bytes(
                SignedRegistryCheckpoint(
                    checkpoint=_canonical_bytes_model(
                        bytes(pinned_checkpoint_row[0]),
                        RegistryCheckpoint,
                        'pinned database checkpoint',
                    ),
                    signature_base64=_encode_base64(bytes(pinned_checkpoint_row[1])),
                )
            )
            if stored_envelope != _decode_base64(
                pinned.signed_checkpoint_base64, 'pinned checkpoint envelope'
            ) or bytes(pinned_witness_row[0]) != _decode_base64(
                pinned.witness_proof_base64, 'pinned checkpoint witness proof'
            ):
                raise SelectionRegistryError('database pinned checkpoint differs from the out-of-band witnessed anchor')

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        try:
            _configure_connection(connection, initializing=False)
            yield connection
        finally:
            connection.close()


def verify_registry_selection(
    commitment_bytes: bytes,
    proof_bytes: bytes,
    binding: PlanSelectionPolicyBinding,
    policy_bytes: bytes,
    trust_policy_bytes: bytes,
) -> AuthenticatedPlanSelectionFacts:
    """Cryptographically verify raw registry proof using only pinned materials.

    This function has the exact :class:`TrustedPlanSelectionVerifier` ABI.  It
    does not access the registry database or network and therefore remains
    replayable after the service disappears.
    """

    if not isinstance(proof_bytes, bytes) or not proof_bytes or len(proof_bytes) > _MAX_PROOF_BYTES:
        raise PlanSelectionIntegrityError('registry proof bytes are empty or exceed the hard limit')
    policy = _canonical_bytes_model(policy_bytes, SelectionRegistryPolicy, 'registry policy')
    trust = _canonical_bytes_model(trust_policy_bytes, SelectionRegistryTrustPolicy, 'registry trust policy')
    commitment = _canonical_bytes_model(
        commitment_bytes,
        PlanSelectionCommitment,
        'plan-selection commitment',
    )
    proof = _canonical_bytes_model(proof_bytes, RegistrySelectionProof, 'registry selection proof')
    if (
        hashlib.sha256(policy_bytes).hexdigest() != binding.policy_sha256
        or hashlib.sha256(trust_policy_bytes).hexdigest() != binding.trust_policy_sha256
        or policy.policy_id != binding.policy_id
        or trust.trust_policy_id != binding.trust_policy_id
    ):
        raise PlanSelectionIntegrityError('registry materials differ from the generic bridge policy binding')
    if (
        policy.registry_id != binding.registry_id
        or policy.authority_id != binding.authority_id
        or trust.registry_id != binding.registry_id
        or trust.authority_id != binding.authority_id
        or commitment.policy != binding
    ):
        raise PlanSelectionIntegrityError('registry authority identity differs across pinned materials')
    entry = proof.entry
    checkpoint = proof.checkpoint
    commitment_sha256 = hashlib.sha256(commitment_bytes).hexdigest()
    request = PlanSelectionRequest(
        commitment_sha256=commitment_sha256,
        commitment_bytes=len(commitment_bytes),
        campaign_id=binding.campaign_id,
        selection_key=binding.selection_key,
        registry_id=binding.registry_id,
        authority_id=binding.authority_id,
        policy_id=binding.policy_id,
        policy_sha256=binding.policy_sha256,
    )
    expected_entry = {
        'registry_id': binding.registry_id,
        'authority_id': binding.authority_id,
        'campaign_id': binding.campaign_id,
        'selection_key': binding.selection_key,
        'commitment_sha256': commitment_sha256,
        'commitment_bytes': len(commitment_bytes),
        'request_sha256': hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
        'policy_id': binding.policy_id,
        'policy_sha256': binding.policy_sha256,
    }
    if any(getattr(entry, name) != value for name, value in expected_entry.items()):
        raise PlanSelectionIntegrityError('registry leaf binds a different request or commitment')
    if (
        checkpoint.registry_id != binding.registry_id
        or checkpoint.authority_id != binding.authority_id
        or checkpoint.tree_size <= entry.registry_sequence
    ):
        raise PlanSelectionIntegrityError('signed checkpoint differs from the registry policy or entry position')
    trusted_key = _trusted_key(trust, checkpoint.signing_key_id)
    public_key_bytes = _decode_base64(trusted_key.public_key_base64, 'Ed25519 public key')
    signature = _decode_base64(proof.checkpoint_signature_base64, 'checkpoint signature')
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature,
            canonical_json_bytes(checkpoint),
        )
    except (InvalidSignature, ValueError) as error:
        raise PlanSelectionIntegrityError('registry checkpoint has an invalid Ed25519 signature') from error
    envelope_bytes = canonical_json_bytes(
        SignedRegistryCheckpoint(
            checkpoint=checkpoint,
            signature_base64=proof.checkpoint_signature_base64,
        )
    )
    checkpoint_witness_proof = _decode_base64(
        proof.checkpoint_witness_proof_base64,
        'checkpoint witness proof',
    )
    try:
        _envelope, witnessed_at, witness_sequence = _verify_witnessed_checkpoint_envelope(
            envelope_bytes,
            checkpoint_witness_proof,
            trust,
        )
        _pinned_envelope, pinned_witnessed_at, pinned_witness_sequence = _verify_pinned_checkpoint(trust)
    except SelectionRegistryError as error:
        raise PlanSelectionIntegrityError(str(error)) from error
    if trust.pinned_checkpoint.tree_size == 0:
        raise PlanSelectionIntegrityError('production selection trust must pin a witnessed nonempty prior checkpoint')
    if witness_sequence <= pinned_witness_sequence or witnessed_at < pinned_witnessed_at:
        raise PlanSelectionIntegrityError('selection checkpoint was not witnessed after the pinned prior checkpoint')
    checkpoint_root = bytes.fromhex(checkpoint.root_sha256)
    inclusion = tuple(bytes.fromhex(item) for item in proof.inclusion_proof_sha256)
    if not verify_inclusion_proof(
        _leaf_hash(canonical_json_bytes(entry)),
        index=entry.registry_sequence,
        tree_size=checkpoint.tree_size,
        proof=inclusion,
        expected_root=checkpoint_root,
    ):
        raise PlanSelectionIntegrityError('registry entry has an invalid Merkle inclusion proof')
    pinned = trust.pinned_checkpoint
    if proof.consistency_from_tree_size != pinned.tree_size or proof.consistency_from_root_sha256 != pinned.root_sha256:
        raise PlanSelectionIntegrityError('proof does not start at the pinned trust checkpoint')
    consistency = tuple(bytes.fromhex(item) for item in proof.consistency_proof_sha256)
    if not verify_consistency_proof(
        old_size=pinned.tree_size,
        new_size=checkpoint.tree_size,
        old_root=bytes.fromhex(pinned.root_sha256),
        new_root=checkpoint_root,
        proof=consistency,
    ):
        raise PlanSelectionIntegrityError('registry checkpoint has an invalid append-only consistency proof')
    if entry.selected_at_upper_bound != checkpoint.issued_at_upper_bound:
        raise PlanSelectionIntegrityError('registry leaf timestamp differs from its first signed checkpoint')
    checkpoint_sha256 = hashlib.sha256(canonical_json_bytes(checkpoint) + signature).hexdigest()
    witness_receipt = _canonical_bytes_model(
        checkpoint_witness_proof,
        WitnessServiceProof,
        'checkpoint witness proof',
    )
    return AuthenticatedPlanSelectionFacts(
        receipt_id=witness_receipt.receipt_id,
        registry_id=binding.registry_id,
        authority_id=binding.authority_id,
        campaign_id=binding.campaign_id,
        selection_key=binding.selection_key,
        commitment_sha256=commitment_sha256,
        store_id=commitment.store_id,
        checkpoint_sha256=commitment.checkpoint_sha256,
        scope_policy_sha256=commitment.scope_policy_sha256,
        pre_capture_plan_sha256=commitment.pre_capture_plan_sha256,
        selected_at_upper_bound=witnessed_at,
        registry_entry_id=entry.registry_entry_id,
        registry_sequence=entry.registry_sequence,
        signed_checkpoint_sha256=checkpoint_sha256,
        signed_checkpoint_size=checkpoint.tree_size,
    )


def verify_signed_registry_checkpoint(
    envelope_bytes: bytes,
    trust_policy_bytes: bytes,
) -> RegistryCheckpoint:
    """Authenticate a public tree-head envelope before external witnessing."""

    envelope = _canonical_bytes_model(
        envelope_bytes,
        SignedRegistryCheckpoint,
        'signed registry checkpoint',
    )
    trust = _canonical_bytes_model(
        trust_policy_bytes,
        SelectionRegistryTrustPolicy,
        'registry trust policy',
    )
    checkpoint = envelope.checkpoint
    if checkpoint.registry_id != trust.registry_id or checkpoint.authority_id != trust.authority_id:
        raise SelectionRegistryError('signed checkpoint differs from the trust-policy authority')
    trusted_key = _trusted_key(trust, checkpoint.signing_key_id)
    _require_key_valid_at(trusted_key, checkpoint.issued_at_upper_bound)
    try:
        Ed25519PublicKey.from_public_bytes(_decode_base64(trusted_key.public_key_base64, 'Ed25519 public key')).verify(
            _decode_base64(envelope.signature_base64, 'checkpoint signature'),
            canonical_json_bytes(checkpoint),
        )
    except (InvalidSignature, ValueError) as error:
        raise SelectionRegistryError('registry checkpoint has an invalid Ed25519 signature') from error
    return checkpoint


def verify_service_bearer_token(expected_sha256: str, authorization_header: str | None) -> bool:
    """Constant-time bearer verification for the network write endpoint."""

    if len(expected_sha256) != 64 or any(character not in '0123456789abcdef' for character in expected_sha256):
        raise SelectionRegistryError('service bearer-token digest must be lowercase SHA-256')
    prefix = 'Bearer '
    if authorization_header is None or not authorization_header.startswith(prefix):
        return False
    token = authorization_header[len(prefix) :]
    if len(token) < 32 or len(token) > 4096 or token != token.strip():
        return False
    return hmac.compare_digest(hashlib.sha256(token.encode('utf-8')).hexdigest(), expected_sha256)


def _trusted_key(
    trust_policy: SelectionRegistryTrustPolicy,
    key_id: str,
) -> RegistryTrustedSigningKey:
    matches = [key for key in trust_policy.signing_keys if key.key_id == key_id]
    if len(matches) != 1:
        raise SelectionRegistryError(f'signing key {key_id!r} is not uniquely authorized by trust policy')
    return matches[0]


def _require_key_valid_at(key: RegistryTrustedSigningKey, when: datetime) -> None:
    when = aware_utc(when, 'checkpoint issuance time')
    if when < key.valid_from or (key.valid_until is not None and when >= key.valid_until):
        raise SelectionRegistryError('checkpoint signing key was not valid at issuance time')


def _utc_text(value: datetime) -> str:
    return aware_utc(value, 'registry timestamp').isoformat().replace('+00:00', 'Z')


def _canonical_model[ModelT: StrictModel](value: object, model: type[ModelT], label: str) -> ModelT:
    if not isinstance(value, model):
        raise SelectionRegistryError(f'{label} must be a {model.__name__}')
    try:
        return model.model_validate_json(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise SelectionRegistryError(f'invalid {label}: {error}') from error


def _canonical_bytes_model[ModelT: StrictModel](payload: bytes, model: type[ModelT], label: str) -> ModelT:
    if not isinstance(payload, bytes) or not payload:
        raise SelectionRegistryError(f'{label} must be nonempty exact bytes')
    try:
        result = model.model_validate_json(payload)
    except ValueError as error:
        raise SelectionRegistryError(f'invalid {label}: {error}') from error
    if payload != canonical_json_bytes(result):
        raise SelectionRegistryError(f'{label} must use canonical JSON encoding')
    return result


def _exclusive_secret_target(path: Path) -> Path:
    request = Path(path).expanduser().absolute()
    request.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = request.parent.resolve(strict=True)
    _require_protected_parent(parent)
    target = parent / request.name
    if os.path.lexists(target):
        raise FileExistsError(target)
    return target


def _new_database_path(path: Path) -> Path:
    request = Path(path).expanduser().absolute()
    request.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = request.parent.resolve(strict=True)
    _require_protected_parent(parent)
    target = parent / request.name
    if os.path.lexists(target):
        raise FileExistsError(target)
    descriptor = os.open(
        target,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
        0o600,
    )
    os.close(descriptor)
    return target


def _existing_database_path(path: Path) -> Path:
    request = Path(path).expanduser().absolute()
    parent = request.parent.resolve(strict=True)
    _require_protected_parent(parent)
    target = parent / request.name
    metadata = target.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise SelectionRegistryError('registry database must be a regular file, not a link or device')
    if metadata.st_mode & 0o077:
        raise SelectionRegistryError('registry database must not be accessible by group or other users')
    return target


def _require_protected_parent(parent: Path) -> None:
    metadata = parent.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise SelectionRegistryError('registry state parent must be a directory')
    if hasattr(os, 'getuid') and metadata.st_uid != os.getuid():
        raise SelectionRegistryError('registry state parent must be owned by the service user')
    if metadata.st_mode & 0o022:
        raise SelectionRegistryError('registry state parent must not be writable by group or other users')


def _configure_connection(connection: sqlite3.Connection, *, initializing: bool) -> None:
    connection.execute('PRAGMA foreign_keys = ON')
    connection.execute('PRAGMA trusted_schema = OFF')
    connection.execute('PRAGMA synchronous = FULL')
    connection.execute('PRAGMA busy_timeout = 30000')
    if initializing:
        connection.execute('PRAGMA journal_mode = WAL')
    else:
        mode = str(connection.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        if mode != 'wal':
            raise SelectionRegistryError('registry database must remain in WAL journal mode')


_DATABASE_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE registry_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    registry_id TEXT NOT NULL,
    authority_id TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE registry_signing_keys (
    key_id TEXT PRIMARY KEY,
    public_key_base64 TEXT NOT NULL UNIQUE,
    registered_at TEXT NOT NULL
) STRICT;
CREATE TABLE registry_assignments (
    assignment_sequence INTEGER PRIMARY KEY CHECK (assignment_sequence >= 0),
    campaign_id TEXT NOT NULL,
    selection_key TEXT NOT NULL,
    commitment_sha256 TEXT NOT NULL,
    commitment_bytes INTEGER NOT NULL CHECK (commitment_bytes > 0),
    request_sha256 TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    reserved_at TEXT NOT NULL,
    UNIQUE(campaign_id, selection_key)
) STRICT;
CREATE TABLE registry_entries (
    sequence INTEGER PRIMARY KEY CHECK (sequence >= 0),
    assignment_sequence INTEGER NOT NULL UNIQUE REFERENCES registry_assignments(assignment_sequence),
    registry_entry_id TEXT NOT NULL UNIQUE,
    campaign_id TEXT NOT NULL,
    selection_key TEXT NOT NULL,
    commitment_sha256 TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    selected_at TEXT NOT NULL,
    leaf_sha256 TEXT NOT NULL UNIQUE,
    entry_bytes BLOB NOT NULL,
    first_checkpoint_size INTEGER NOT NULL UNIQUE CHECK (first_checkpoint_size = sequence + 1),
    UNIQUE(campaign_id, selection_key)
) STRICT;
CREATE TABLE registry_checkpoints (
    tree_size INTEGER PRIMARY KEY CHECK (tree_size >= 0),
    root_sha256 TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    signing_key_id TEXT NOT NULL REFERENCES registry_signing_keys(key_id),
    checkpoint_bytes BLOB NOT NULL,
    signature BLOB NOT NULL
) STRICT;
CREATE TABLE registry_checkpoint_witnesses (
    tree_size INTEGER PRIMARY KEY REFERENCES registry_checkpoints(tree_size),
    envelope_sha256 TEXT NOT NULL UNIQUE,
    witness_receipt_id TEXT NOT NULL UNIQUE,
    witnessed_at TEXT NOT NULL,
    witness_sequence INTEGER NOT NULL UNIQUE CHECK (witness_sequence > 0),
    proof_bytes BLOB NOT NULL
) STRICT;
CREATE TRIGGER registry_metadata_no_update BEFORE UPDATE ON registry_metadata
BEGIN SELECT RAISE(ABORT, 'registry metadata is immutable'); END;
CREATE TRIGGER registry_metadata_no_delete BEFORE DELETE ON registry_metadata
BEGIN SELECT RAISE(ABORT, 'registry metadata is immutable'); END;
CREATE TRIGGER registry_signing_keys_no_update BEFORE UPDATE ON registry_signing_keys
BEGIN SELECT RAISE(ABORT, 'registry signing keys are append-only'); END;
CREATE TRIGGER registry_signing_keys_no_delete BEFORE DELETE ON registry_signing_keys
BEGIN SELECT RAISE(ABORT, 'registry signing keys are append-only'); END;
CREATE TRIGGER registry_assignments_no_update BEFORE UPDATE ON registry_assignments
BEGIN SELECT RAISE(ABORT, 'registry assignments are append-only'); END;
CREATE TRIGGER registry_assignments_no_delete BEFORE DELETE ON registry_assignments
BEGIN SELECT RAISE(ABORT, 'registry assignments are append-only'); END;
CREATE TRIGGER registry_entries_no_update BEFORE UPDATE ON registry_entries
BEGIN SELECT RAISE(ABORT, 'registry entries are append-only'); END;
CREATE TRIGGER registry_entries_no_delete BEFORE DELETE ON registry_entries
BEGIN SELECT RAISE(ABORT, 'registry entries are append-only'); END;
CREATE TRIGGER registry_checkpoints_no_update BEFORE UPDATE ON registry_checkpoints
BEGIN SELECT RAISE(ABORT, 'registry checkpoints are append-only'); END;
CREATE TRIGGER registry_checkpoints_no_delete BEFORE DELETE ON registry_checkpoints
BEGIN SELECT RAISE(ABORT, 'registry checkpoints are append-only'); END;
CREATE TRIGGER registry_checkpoint_witnesses_no_update BEFORE UPDATE ON registry_checkpoint_witnesses
BEGIN SELECT RAISE(ABORT, 'registry checkpoint witnesses are append-only'); END;
CREATE TRIGGER registry_checkpoint_witnesses_no_delete BEFORE DELETE ON registry_checkpoint_witnesses
BEGIN SELECT RAISE(ABORT, 'registry checkpoint witnesses are append-only'); END;
COMMIT;
"""


__all__ = [
    'REGISTRY_CHECKPOINT_SCHEMA_VERSION',
    'REGISTRY_ENTRY_SCHEMA_VERSION',
    'REGISTRY_POLICY_SCHEMA_VERSION',
    'REGISTRY_PROOF_SCHEMA_VERSION',
    'REGISTRY_RESPONSE_SCHEMA_VERSION',
    'REGISTRY_TRUST_POLICY_SCHEMA_VERSION',
    'SIGNED_REGISTRY_CHECKPOINT_SCHEMA_VERSION',
    'RegistryCheckpoint',
    'RegistryConflictError',
    'RegistryLogEntry',
    'RegistryPinnedCheckpoint',
    'RegistrySelectionProof',
    'RegistrySelectionResponse',
    'RegistryTrustedSigningKey',
    'RegistryWitnessUnavailableError',
    'SQLitePlanSelectionRegistry',
    'SelectionRegistryError',
    'SelectionRegistryPolicy',
    'SelectionRegistryTrustPolicy',
    'SignedRegistryCheckpoint',
    'build_plan_selection_policy_binding',
    'build_signed_registry_genesis_checkpoint',
    'checkpoint_witness_request',
    'consistency_proof',
    'ed25519_public_key_base64',
    'generate_ed25519_private_key',
    'inclusion_proof',
    'load_ed25519_private_key',
    'merkle_root',
    'production_plan_selection_materials',
    'selection_registry_verifier_implementation_bytes',
    'verify_consistency_proof',
    'verify_inclusion_proof',
    'verify_registry_selection',
    'verify_signed_registry_checkpoint',
    'verify_service_bearer_token',
]
