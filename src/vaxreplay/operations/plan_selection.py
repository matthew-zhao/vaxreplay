"""Externally authenticated, first-write-wins pre-capture plan selection.

A timestamp proves that a plan existed; it does not prove that the organizer
did not timestamp several plans and choose one after seeing captures.  This
module represents the stronger external fact needed to close that gap: an
independent registry atomically assigned one stable campaign selection key to
one exact pre-capture commitment and made that assignment immutable.

The injected provider is transport only.  Security facts are accepted solely
from a separately pinned verifier over the exact commitment and raw proof
bytes.  Loading always reruns that verifier under out-of-band policy, trust,
and verifier-implementation material.

The verifier is an external trusted-computing-base boundary, not an implementation
of a registry.  Its facts are evidence only when it authenticates a real signed
append-only-log inclusion proof and consistency proof from the pinned trust
checkpoint; process-local test registries do not provide that property.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations._immutable_tree import ImmutableTreeError, snapshot_immutable_tree
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc

PLAN_SELECTION_POLICY_SCHEMA_VERSION = 'vaxreplay.plan-selection-policy.v0.1'
PLAN_SELECTION_COMMITMENT_SCHEMA_VERSION = 'vaxreplay.plan-selection-commitment.v0.1'
PLAN_SELECTION_REQUEST_SCHEMA_VERSION = 'vaxreplay.plan-selection-request.v0.1'
PLAN_SELECTION_CLAIM_SCHEMA_VERSION = 'vaxreplay.plan-selection-claim.v0.1'
AUTHENTICATED_PLAN_SELECTION_FACTS_SCHEMA_VERSION = 'vaxreplay.authenticated-plan-selection-facts.v0.1'
PLAN_SELECTION_RECEIPT_SCHEMA_VERSION = 'vaxreplay.plan-selection-receipt.v0.1'
PLAN_SELECTION_MANIFEST_SCHEMA_VERSION = 'vaxreplay.plan-selection.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_COMMITMENT_PATH = 'commitment.json'
_PROOF_PATH = 'registry-proof.bin'
_MANIFEST_PATH = 'selection.json'
_MAX_MODEL_BYTES = 4 * 1024 * 1024
_MAX_PROOF_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_BYTES = _MAX_PROOF_BYTES + (2 * _MAX_MODEL_BYTES)


class PlanSelectionIntegrityError(ValueError):
    """A plan-selection proof, policy, or immutable sidecar failed closed."""


class PlanSelectionPolicyBinding(StrictModel):
    """Out-of-band identity of one authorized first-write-wins campaign key."""

    schema_version: Literal['vaxreplay.plan-selection-policy.v0.1'] = PLAN_SELECTION_POLICY_SCHEMA_VERSION
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=SAFE_ID_PATTERN)
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    trust_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    trust_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    verifier_id: str = Field(pattern=SAFE_ID_PATTERN)
    verifier_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    registry_protocol: Literal['signed_append_only_log_with_inclusion_and_consistency_proofs'] = (
        'signed_append_only_log_with_inclusion_and_consistency_proofs'
    )
    uniqueness_semantics: Literal['atomic_first_write_wins_per_selection_key'] = (
        'atomic_first_write_wins_per_selection_key'
    )
    finality_semantics: Literal['immutable_no_reassignment'] = 'immutable_no_reassignment'


class PlanSelectionCommitment(StrictModel):
    """Exact plan identity selected before the campaign's first capture slot."""

    schema_version: Literal['vaxreplay.plan-selection-commitment.v0.1'] = PLAN_SELECTION_COMMITMENT_SCHEMA_VERSION
    policy: PlanSelectionPolicyBinding
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_created_at: datetime
    scope_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    pre_capture_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    earliest_scheduled_slot: datetime

    @field_validator('checkpoint_created_at', 'earliest_scheduled_slot')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'plan selection timestamp')

    @model_validator(mode='after')
    def validate_chronology(self) -> Self:
        if self.checkpoint_created_at >= self.earliest_scheduled_slot:
            raise ValueError('plan-selection checkpoint must predate the first scheduled slot')
        return self


class PlanSelectionRequest(StrictModel):
    """Minimal commitment sent to the independent selection registry."""

    schema_version: Literal['vaxreplay.plan-selection-request.v0.1'] = PLAN_SELECTION_REQUEST_SCHEMA_VERSION
    commitment_schema_version: Literal['vaxreplay.plan-selection-commitment.v0.1'] = (
        PLAN_SELECTION_COMMITMENT_SCHEMA_VERSION
    )
    commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    commitment_bytes: int = Field(gt=0)
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=SAFE_ID_PATTERN)
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)


class PlanSelectionClaim(StrictModel):
    """Untrusted provider metadata; no security fact is accepted from it."""

    schema_version: Literal['vaxreplay.plan-selection-claim.v0.1'] = PLAN_SELECTION_CLAIM_SCHEMA_VERSION
    verification_uri: str = Field(min_length=1, max_length=4096)

    @field_validator('verification_uri')
    @classmethod
    def validate_verification_uri(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise ValueError('verification_uri must be trimmed and contain no control characters')
        return value


class AuthenticatedPlanSelectionFacts(StrictModel):
    """Facts parsed and authenticated by the independently pinned verifier."""

    schema_version: Literal['vaxreplay.authenticated-plan-selection-facts.v0.1'] = (
        AUTHENTICATED_PLAN_SELECTION_FACTS_SCHEMA_VERSION
    )
    receipt_id: str = Field(pattern=SAFE_ID_PATTERN)
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=SAFE_ID_PATTERN)
    commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    pre_capture_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected_at_upper_bound: datetime
    registry_entry_id: str = Field(pattern=SAFE_ID_PATTERN)
    registry_sequence: int = Field(ge=0)
    signed_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_checkpoint_size: int = Field(gt=0)
    valid_inclusion_proof: Literal[True] = True
    consistent_from_pinned_trust_checkpoint: Literal[True] = True
    selection_key_history_count: Literal[1] = 1
    key_previously_unassigned: Literal[True] = True
    atomic_first_write_wins_enforced: Literal[True] = True
    selection_final_and_immutable: Literal[True] = True

    @field_validator('selected_at_upper_bound')
    @classmethod
    def validate_selected_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'selected_at_upper_bound')

    @model_validator(mode='after')
    def validate_registry_position(self) -> Self:
        if self.registry_sequence >= self.signed_checkpoint_size:
            raise ValueError('registry sequence must be inside the signed checkpoint tree')
        return self


class PlanSelectionReceipt(StrictModel):
    schema_version: Literal['vaxreplay.plan-selection-receipt.v0.1'] = PLAN_SELECTION_RECEIPT_SCHEMA_VERSION
    facts: AuthenticatedPlanSelectionFacts
    policy: PlanSelectionPolicyBinding
    proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1, max_length=4096)

    @field_validator('verification_uri')
    @classmethod
    def validate_verification_uri(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise ValueError('verification_uri must be trimmed and contain no control characters')
        return value

    @model_validator(mode='after')
    def validate_policy_identity(self) -> Self:
        facts = self.facts
        policy = self.policy
        if (
            facts.registry_id != policy.registry_id
            or facts.authority_id != policy.authority_id
            or facts.campaign_id != policy.campaign_id
            or facts.selection_key != policy.selection_key
        ):
            raise ValueError('authenticated selection facts differ from the pinned registry policy')
        return self


class PlanSelectionManifest(StrictModel):
    schema_version: Literal['vaxreplay.plan-selection.v0.1'] = PLAN_SELECTION_MANIFEST_SCHEMA_VERSION
    commitment_path: Literal['commitment.json'] = _COMMITMENT_PATH
    commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    commitment_bytes: int = Field(gt=0)
    proof_path: Literal['registry-proof.bin'] = _PROOF_PATH
    receipt: PlanSelectionReceipt

    @model_validator(mode='after')
    def validate_commitment_receipt(self) -> Self:
        if self.receipt.facts.commitment_sha256 != self.commitment_sha256:
            raise ValueError('plan-selection receipt binds a different commitment')
        return self


type PlanSelectionProvider = Callable[[PlanSelectionRequest], tuple[PlanSelectionClaim, bytes]]
type TrustedPlanSelectionVerifier = Callable[
    [bytes, bytes, PlanSelectionPolicyBinding, bytes, bytes],
    AuthenticatedPlanSelectionFacts,
]


class PlanSelectionMaterialSpecProtocol(Protocol):
    @property
    def policy(self) -> PlanSelectionPolicyBinding: ...

    @property
    def policy_bytes(self) -> bytes: ...

    @property
    def trust_policy_bytes(self) -> bytes: ...

    @property
    def verifier_implementation_bytes(self) -> bytes: ...

    @property
    def verifier(self) -> TrustedPlanSelectionVerifier: ...


@dataclass(frozen=True)
class PlanSelectionMaterialSpec:
    """Exact trusted material supplied outside the selected artifact."""

    policy: PlanSelectionPolicyBinding
    policy_bytes: bytes
    trust_policy_bytes: bytes
    verifier_implementation_bytes: bytes
    verifier: TrustedPlanSelectionVerifier


@dataclass(frozen=True)
class LoadedPlanSelection:
    root: Path
    manifest: PlanSelectionManifest
    commitment: PlanSelectionCommitment
    commitment_bytes: bytes
    proof_bytes: bytes
    manifest_bytes: bytes
    manifest_sha256: str

    @property
    def selected_at_upper_bound(self) -> datetime:
        return self.manifest.receipt.facts.selected_at_upper_bound


@dataclass(frozen=True)
class VerifiedPlanSelectionBytes:
    """An exact three-file sidecar verified without reopening filesystem paths."""

    manifest: PlanSelectionManifest
    commitment: PlanSelectionCommitment
    commitment_bytes: bytes
    proof_bytes: bytes
    manifest_bytes: bytes
    manifest_sha256: str

    @property
    def selected_at_upper_bound(self) -> datetime:
        return self.manifest.receipt.facts.selected_at_upper_bound


def plan_selection_commitment_sha256(commitment: PlanSelectionCommitment) -> str:
    return hashlib.sha256(canonical_json_bytes(commitment)).hexdigest()


def plan_selection_manifest_sha256(manifest: PlanSelectionManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def plan_selection_request(commitment: PlanSelectionCommitment) -> PlanSelectionRequest:
    commitment = _canonical_model(commitment, PlanSelectionCommitment, 'plan-selection commitment')
    payload = canonical_json_bytes(commitment)
    policy = commitment.policy
    return PlanSelectionRequest(
        commitment_sha256=hashlib.sha256(payload).hexdigest(),
        commitment_bytes=len(payload),
        campaign_id=policy.campaign_id,
        selection_key=policy.selection_key,
        registry_id=policy.registry_id,
        authority_id=policy.authority_id,
        policy_id=policy.policy_id,
        policy_sha256=policy.policy_sha256,
    )


def broker_plan_selection(
    output_dir: Path,
    *,
    commitment: PlanSelectionCommitment,
    materials: PlanSelectionMaterialSpecProtocol,
    provider: PlanSelectionProvider,
    verified_at: datetime | None = None,
) -> LoadedPlanSelection:
    """Register one exact plan commitment and publish its verified sidecar."""

    if provider is None:  # type: ignore[comparison-overlap]
        raise PlanSelectionIntegrityError('an independent plan-selection provider is required')
    commitment = _canonical_model(commitment, PlanSelectionCommitment, 'plan-selection commitment')
    policy, verifier, policy_bytes, trust_policy_bytes = _validate_materials(materials)
    if commitment.policy != policy:
        raise PlanSelectionIntegrityError('plan-selection commitment uses a different out-of-band policy')
    checked_at = aware_utc(verified_at or datetime.now(timezone.utc), 'verified_at')
    request = plan_selection_request(commitment)
    try:
        returned = provider(request)
    except Exception as error:
        raise PlanSelectionIntegrityError(f'plan-selection provider failed: {error}') from error
    if not isinstance(returned, tuple) or len(returned) != 2:
        raise PlanSelectionIntegrityError('plan-selection provider returned an invalid response')
    claim = _canonical_model(returned[0], PlanSelectionClaim, 'plan-selection claim')
    proof_bytes = returned[1]
    facts = _authenticate(
        commitment,
        proof_bytes=proof_bytes,
        policy=policy,
        verifier=verifier,
        policy_bytes=policy_bytes,
        trust_policy_bytes=trust_policy_bytes,
        verified_at=checked_at,
    )
    commitment_bytes = canonical_json_bytes(commitment)
    receipt = PlanSelectionReceipt(
        facts=facts,
        policy=policy,
        proof_sha256=hashlib.sha256(proof_bytes).hexdigest(),
        proof_bytes=len(proof_bytes),
        verification_uri=claim.verification_uri,
    )
    manifest = PlanSelectionManifest(
        commitment_sha256=hashlib.sha256(commitment_bytes).hexdigest(),
        commitment_bytes=len(commitment_bytes),
        receipt=receipt,
    )
    target = _durable_publish(
        output_dir,
        {
            _COMMITMENT_PATH: commitment_bytes,
            _PROOF_PATH: proof_bytes,
            _MANIFEST_PATH: canonical_json_bytes(manifest),
        },
    )
    return load_plan_selection(
        target,
        expected_commitment=commitment,
        expected_manifest_sha256=plan_selection_manifest_sha256(manifest),
        materials=materials,
        verified_at=checked_at,
    )


def load_plan_selection(
    root: Path,
    *,
    expected_commitment: PlanSelectionCommitment,
    expected_manifest_sha256: str,
    materials: PlanSelectionMaterialSpecProtocol,
    verified_at: datetime | None = None,
) -> LoadedPlanSelection:
    """Load exact sidecar bytes and rerun the trusted registry verifier."""
    try:
        snapshot = snapshot_immutable_tree(
            root,
            max_files=3,
            max_directories=1,
            max_file_bytes=_MAX_PROOF_BYTES,
            max_total_bytes=_MAX_TOTAL_BYTES,
            max_path_characters=256,
            per_path_byte_limits={
                _COMMITMENT_PATH: _MAX_MODEL_BYTES,
                _MANIFEST_PATH: _MAX_MODEL_BYTES,
                _PROOF_PATH: _MAX_PROOF_BYTES,
            },
        )
        snapshot.require_exact_files({_COMMITMENT_PATH, _PROOF_PATH, _MANIFEST_PATH})
    except ImmutableTreeError as error:
        raise PlanSelectionIntegrityError(f'unsafe plan-selection artifact: {error}') from error
    verified = verify_plan_selection_bytes(
        commitment_bytes=snapshot.files[_COMMITMENT_PATH],
        proof_bytes=snapshot.files[_PROOF_PATH],
        manifest_bytes=snapshot.files[_MANIFEST_PATH],
        expected_commitment=expected_commitment,
        expected_manifest_sha256=expected_manifest_sha256,
        materials=materials,
        verified_at=verified_at,
    )
    return LoadedPlanSelection(
        root=snapshot.root,
        manifest=verified.manifest,
        commitment=verified.commitment,
        commitment_bytes=verified.commitment_bytes,
        proof_bytes=verified.proof_bytes,
        manifest_bytes=verified.manifest_bytes,
        manifest_sha256=verified.manifest_sha256,
    )


def verify_plan_selection_bytes(
    *,
    commitment_bytes: bytes,
    proof_bytes: bytes,
    manifest_bytes: bytes,
    expected_commitment: PlanSelectionCommitment,
    expected_manifest_sha256: str,
    materials: PlanSelectionMaterialSpecProtocol,
    verified_at: datetime | None = None,
) -> VerifiedPlanSelectionBytes:
    """Verify snapshotted exact sidecar bytes under independently pinned materials."""

    expected_commitment = _canonical_model(
        expected_commitment,
        PlanSelectionCommitment,
        'expected plan-selection commitment',
    )
    if not isinstance(expected_manifest_sha256, str) or not _is_sha256(expected_manifest_sha256):
        raise PlanSelectionIntegrityError('expected plan-selection manifest SHA-256 is invalid')
    policy, verifier, policy_bytes, trust_policy_bytes = _validate_materials(materials)
    if expected_commitment.policy != policy:
        raise PlanSelectionIntegrityError('expected plan selection uses a different out-of-band policy')
    checked_at = aware_utc(verified_at or datetime.now(timezone.utc), 'verified_at')
    exact_files = (
        ('plan-selection commitment', commitment_bytes, _MAX_MODEL_BYTES),
        ('plan-selection proof', proof_bytes, _MAX_PROOF_BYTES),
        ('plan-selection manifest', manifest_bytes, _MAX_MODEL_BYTES),
    )
    for label, payload, limit in exact_files:
        if not isinstance(payload, bytes) or not payload:
            raise PlanSelectionIntegrityError(f'{label} must be nonempty exact bytes')
        if len(payload) > limit:
            raise PlanSelectionIntegrityError(f'{label} exceeds its byte limit')
    if sum(len(payload) for _label, payload, _limit in exact_files) > _MAX_TOTAL_BYTES:
        raise PlanSelectionIntegrityError('plan-selection sidecar exceeds its aggregate byte limit')

    manifest = _canonical_bytes_model(manifest_bytes, PlanSelectionManifest, 'plan-selection manifest')
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise PlanSelectionIntegrityError('plan-selection manifest differs from the expected digest')
    commitment = _canonical_bytes_model(
        commitment_bytes,
        PlanSelectionCommitment,
        'plan-selection commitment',
    )
    if commitment != expected_commitment:
        raise PlanSelectionIntegrityError('plan-selection artifact binds a different expected commitment')
    if manifest.commitment_sha256 != hashlib.sha256(commitment_bytes).hexdigest() or manifest.commitment_bytes != len(
        commitment_bytes
    ):
        raise PlanSelectionIntegrityError('plan-selection commitment differs from its manifest binding')
    receipt = manifest.receipt
    if receipt.proof_sha256 != hashlib.sha256(proof_bytes).hexdigest() or receipt.proof_bytes != len(proof_bytes):
        raise PlanSelectionIntegrityError('plan-selection proof differs from its receipt binding')
    facts = _authenticate(
        commitment,
        proof_bytes=proof_bytes,
        policy=policy,
        verifier=verifier,
        policy_bytes=policy_bytes,
        trust_policy_bytes=trust_policy_bytes,
        verified_at=checked_at,
    )
    if facts != receipt.facts or receipt.policy != policy:
        raise PlanSelectionIntegrityError('persisted selection receipt differs from authenticated proof facts')
    return VerifiedPlanSelectionBytes(
        manifest=manifest,
        commitment=commitment,
        commitment_bytes=commitment_bytes,
        proof_bytes=proof_bytes,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _authenticate(
    commitment: PlanSelectionCommitment,
    *,
    proof_bytes: bytes,
    policy: PlanSelectionPolicyBinding,
    verifier: TrustedPlanSelectionVerifier,
    policy_bytes: bytes,
    trust_policy_bytes: bytes,
    verified_at: datetime,
) -> AuthenticatedPlanSelectionFacts:
    if not isinstance(proof_bytes, bytes) or not proof_bytes:
        raise PlanSelectionIntegrityError('plan-selection proof must be nonempty exact bytes')
    if len(proof_bytes) > _MAX_PROOF_BYTES:
        raise PlanSelectionIntegrityError('plan-selection proof exceeds its byte limit')
    commitment_bytes = canonical_json_bytes(commitment)
    try:
        returned = verifier(
            commitment_bytes,
            proof_bytes,
            policy,
            policy_bytes,
            trust_policy_bytes,
        )
    except Exception as error:
        raise PlanSelectionIntegrityError(f'trusted plan-selection verifier failed: {error}') from error
    facts = _canonical_model(returned, AuthenticatedPlanSelectionFacts, 'authenticated plan-selection facts')
    expected = {
        'registry_id': policy.registry_id,
        'authority_id': policy.authority_id,
        'campaign_id': policy.campaign_id,
        'selection_key': policy.selection_key,
        'commitment_sha256': hashlib.sha256(commitment_bytes).hexdigest(),
        'store_id': commitment.store_id,
        'checkpoint_sha256': commitment.checkpoint_sha256,
        'scope_policy_sha256': commitment.scope_policy_sha256,
        'pre_capture_plan_sha256': commitment.pre_capture_plan_sha256,
    }
    if any(getattr(facts, name) != value for name, value in expected.items()):
        raise PlanSelectionIntegrityError('authenticated plan-selection facts bind a different commitment or policy')
    selected_at = facts.selected_at_upper_bound
    if selected_at < commitment.checkpoint_created_at:
        raise PlanSelectionIntegrityError('plan selection predates its committed checkpoint')
    if selected_at >= commitment.earliest_scheduled_slot:
        raise PlanSelectionIntegrityError('plan selection is not strictly before the first scheduled slot')
    if selected_at > verified_at:
        raise PlanSelectionIntegrityError('plan selection postdates the caller verification time')
    return facts


def _validate_materials(
    materials: PlanSelectionMaterialSpecProtocol,
) -> tuple[PlanSelectionPolicyBinding, TrustedPlanSelectionVerifier, bytes, bytes]:
    try:
        policy = _canonical_model(materials.policy, PlanSelectionPolicyBinding, 'plan-selection policy')
        exact = (
            ('plan-selection policy', materials.policy_bytes, policy.policy_sha256),
            ('plan-selection trust policy', materials.trust_policy_bytes, policy.trust_policy_sha256),
            (
                'plan-selection verifier implementation',
                materials.verifier_implementation_bytes,
                policy.verifier_implementation_sha256,
            ),
        )
        verifier = materials.verifier
    except AttributeError as error:
        raise TypeError('plan-selection materials do not satisfy the required interface') from error
    for label, payload, expected_sha256 in exact:
        if not isinstance(payload, bytes) or not payload:
            raise PlanSelectionIntegrityError(f'{label} bytes must be nonempty')
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise PlanSelectionIntegrityError(f'{label} differs from its independently pinned digest')
    if verifier is None:  # type: ignore[comparison-overlap]
        raise PlanSelectionIntegrityError('an independent trusted plan-selection verifier is required')
    return policy, verifier, materials.policy_bytes, materials.trust_policy_bytes


def _canonical_model[ModelT: StrictModel](value: object, model: type[ModelT], label: str) -> ModelT:
    if not isinstance(value, model):
        raise PlanSelectionIntegrityError(f'{label} must be a {model.__name__}')
    try:
        return model.model_validate_json(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise PlanSelectionIntegrityError(f'invalid {label}: {error}') from error


def _canonical_bytes_model[ModelT: StrictModel](payload: bytes, model: type[ModelT], label: str) -> ModelT:
    try:
        result = model.model_validate_json(payload)
    except ValueError as error:
        raise PlanSelectionIntegrityError(f'invalid {label}: {error}') from error
    if payload != canonical_json_bytes(result):
        raise PlanSelectionIntegrityError(f'{label} must use canonical JSON encoding')
    return result


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in '0123456789abcdef' for character in value)


def _durable_publish(output_dir: Path, files: dict[str, bytes]) -> Path:
    target_request = Path(output_dir).expanduser().absolute()
    target_request.parent.mkdir(parents=True, exist_ok=True)
    parent = target_request.parent.resolve(strict=True)
    target = parent / target_request.name
    lock = parent / f'.{target.name}.publish.lock'
    try:
        lock_descriptor = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
            0o600,
        )
    except FileExistsError as error:
        raise PlanSelectionIntegrityError(f'plan-selection publication is already locked: {target}') from error
    staging: Path | None = None
    installed = False
    try:
        os.fsync(lock_descriptor)
        fsync_directory(parent)
        if os.path.lexists(target):
            raise PlanSelectionIntegrityError(f'plan-selection output already exists: {target}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=parent))
        for name, payload in sorted(files.items()):
            with (staging / name).open('xb') as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
        staging.chmod(0o555)
        fsync_directory(staging)
        try:
            rename_directory_noreplace(staging, target)
        except FileExistsError as error:
            raise PlanSelectionIntegrityError(f'plan-selection output already exists: {target}') from error
        installed = True
        fsync_directory(parent)
        return target
    finally:
        if staging is not None and not installed:
            try:
                staging.chmod(0o755)
                shutil.rmtree(staging, ignore_errors=True)
            except OSError:
                pass
        try:
            acquired = os.fstat(lock_descriptor)
            try:
                current = lock.lstat()
            except FileNotFoundError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (acquired.st_dev, acquired.st_ino):
                lock.unlink()
        finally:
            os.close(lock_descriptor)
            fsync_directory(parent)
