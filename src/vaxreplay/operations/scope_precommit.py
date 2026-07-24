"""Portable, externally witnessed commitment to a capture scope and protocol.

The scope precommit closes a specific prospective-evaluation loophole: choosing a
source/job/time universe or normalization protocol after inspecting captured data.
It preserves the exact scope policy, complete pre-capture plan, registered job
specifications, and operations-ledger prefix that an independent witness timestamped
before the first scheduled slot.

The archive also carries an externally verified first-write-wins plan-selection
sidecar.  Its generic verifier boundary only becomes anti-equivocation evidence when
backed by a real independently operated signed append-only registry and an expected
selection-manifest digest fixed outside the archive.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations._immutable_tree import ImmutableTreeError, snapshot_immutable_tree
from vaxreplay.operations.plan_selection import (
    PlanSelectionCommitment,
    PlanSelectionIntegrityError,
    PlanSelectionManifest,
    PlanSelectionMaterialSpecProtocol,
    PlanSelectionPolicyBinding,
    load_plan_selection,
    verify_plan_selection_bytes,
)
from vaxreplay.operations.portable_ledger import export_ledger_prefix, parse_ledger_prefix
from vaxreplay.operations.promotion_schema import (
    ExternalWitnessPromotionBinding,
    PreCapturePlan,
    PrecommittedAdapter,
    PrecommittedHermeticExecution,
    PrecommittedSourceVerifier,
    PromotionFileBinding,
    PromotionIntegrityError,
    PromotionScopePolicy,
)
from vaxreplay.operations.schema import (
    SAFE_ID_PATTERN,
    LedgerCheckpoint,
    LedgerEvent,
    LedgerEventType,
    RegisteredJob,
    aware_utc,
    checkpoint_sha256,
)
from vaxreplay.operations.store import OperationalStore
from vaxreplay.operations.witness import (
    TrustedCheckpointWitnessVerifier,
    WitnessedCheckpointManifest,
    WitnessPolicyBinding,
    load_witnessed_checkpoint,
    verify_witnessed_checkpoint,
)

SCOPE_PRECOMMIT_SCHEMA_VERSION = 'vaxreplay.scope-precommit.v0.3'

_MANIFEST_PATH = 'scope-precommit.json'
_SCOPE_POLICY_PATH = 'scope/policy.json'
_PRE_CAPTURE_PLAN_PATH = 'scope/pre-capture-plan.json'
_LEDGER_PREFIX_PATH = 'ledger/events.jsonl'
_REGISTERED_JOBS_PATH = 'ledger/jobs.jsonl'
_WITNESS_MANIFEST_PATH = 'witness/sidecar/witness.json'
_WITNESS_CHECKPOINT_PATH = 'witness/sidecar/checkpoint.json'
_WITNESS_PROOF_PATH = 'witness/sidecar/external-proof.bin'
_WITNESS_POLICY_PATH = 'witness/materials/policy.bin'
_WITNESS_TRUST_POLICY_PATH = 'witness/materials/trust-policy.bin'
_WITNESS_VERIFIER_PATH = 'witness/materials/verifier-implementation.bin'
_SELECTION_MANIFEST_PATH = 'selection/sidecar/selection.json'
_SELECTION_COMMITMENT_PATH = 'selection/sidecar/commitment.json'
_SELECTION_PROOF_PATH = 'selection/sidecar/registry-proof.bin'
_SELECTION_POLICY_PATH = 'selection/materials/policy.bin'
_SELECTION_TRUST_POLICY_PATH = 'selection/materials/trust-policy.bin'
_SELECTION_VERIFIER_PATH = 'selection/materials/verifier-implementation.bin'
_EXPECTED_PAYLOAD_PATHS = frozenset(
    {
        _SCOPE_POLICY_PATH,
        _PRE_CAPTURE_PLAN_PATH,
        _LEDGER_PREFIX_PATH,
        _REGISTERED_JOBS_PATH,
        _WITNESS_MANIFEST_PATH,
        _WITNESS_CHECKPOINT_PATH,
        _WITNESS_PROOF_PATH,
        _WITNESS_POLICY_PATH,
        _WITNESS_TRUST_POLICY_PATH,
        _WITNESS_VERIFIER_PATH,
        _SELECTION_MANIFEST_PATH,
        _SELECTION_COMMITMENT_PATH,
        _SELECTION_PROOF_PATH,
        _SELECTION_POLICY_PATH,
        _SELECTION_TRUST_POLICY_PATH,
        _SELECTION_VERIFIER_PATH,
    }
)
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_FILES = 64
_MAX_SCHEDULED_SLOTS = 100_000


class ScopePrecommitIntegrityError(PromotionIntegrityError):
    """The pre-capture commitment is incomplete, mutable, or inconsistent."""


class WitnessMaterialSpecProtocol(Protocol):
    """Structural witness-material interface, avoiding a promotion-module cycle."""

    @property
    def policy(self) -> WitnessPolicyBinding: ...

    @property
    def policy_bytes(self) -> bytes: ...

    @property
    def trust_policy_bytes(self) -> bytes: ...

    @property
    def verifier_implementation_bytes(self) -> bytes: ...

    @property
    def verifier(self) -> TrustedCheckpointWitnessVerifier: ...


class PlanSelectionArchiveBinding(StrictModel):
    """Exact selected-plan sidecar and independently pinned verification material."""

    selection_manifest: PromotionFileBinding
    commitment: PromotionFileBinding
    proof: PromotionFileBinding
    policy: PromotionFileBinding
    trust_policy: PromotionFileBinding
    verifier_implementation: PromotionFileBinding
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    commitment_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    selected_at_upper_bound: datetime
    registry_entry_id: str = Field(pattern=SAFE_ID_PATTERN)
    registry_sequence: int = Field(ge=0)
    signed_checkpoint_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    signed_checkpoint_size: int = Field(gt=0)
    valid_inclusion_proof: Literal[True] = True
    consistent_from_pinned_trust_checkpoint: Literal[True] = True
    selection_key_history_count: Literal[1] = 1
    key_previously_unassigned: Literal[True] = True
    atomic_first_write_wins_enforced: Literal[True] = True
    selection_final_and_immutable: Literal[True] = True
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    trust_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    trust_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    verifier_id: str = Field(pattern=SAFE_ID_PATTERN)
    verifier_implementation_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    proof_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    proof_byte_count: int = Field(gt=0)
    uniqueness_semantics: Literal['atomic_first_write_wins_per_selection_key'] = (
        'atomic_first_write_wins_per_selection_key'
    )
    finality_semantics: Literal['immutable_no_reassignment'] = 'immutable_no_reassignment'

    @field_validator('selected_at_upper_bound')
    @classmethod
    def validate_selected_at_upper_bound(cls, value: datetime) -> datetime:
        return aware_utc(value, 'selected_at_upper_bound')

    @model_validator(mode='after')
    def validate_file_digests(self) -> Self:
        if self.selection_manifest.sha256 != self.manifest_sha256:
            raise ValueError('plan-selection manifest binding differs from its declared digest')
        if self.commitment.sha256 != self.commitment_sha256:
            raise ValueError('plan-selection commitment binding differs from its declared digest')
        if self.proof.sha256 != self.proof_sha256 or self.proof.byte_count != self.proof_byte_count:
            raise ValueError('plan-selection proof binding differs from its declared receipt')
        if (
            self.policy.sha256 != self.policy_sha256
            or self.trust_policy.sha256 != self.trust_policy_sha256
            or self.verifier_implementation.sha256 != self.verifier_implementation_sha256
        ):
            raise ValueError('plan-selection metadata does not bind its copied policy and verifier materials')
        if self.registry_sequence >= self.signed_checkpoint_size:
            raise ValueError('plan-selection registry sequence is outside its signed checkpoint tree')
        return self


class ScopePrecommitManifest(StrictModel):
    """Canonical manifest recursively binding every portable archive payload."""

    schema_version: Literal['vaxreplay.scope-precommit.v0.3'] = SCOPE_PRECOMMIT_SCHEMA_VERSION
    created_at: datetime
    checkpoint_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    checkpoint: LedgerCheckpoint
    scope_policy: PromotionFileBinding
    pre_capture_plan: PromotionFileBinding
    ledger_prefix: PromotionFileBinding
    registered_jobs: PromotionFileBinding
    witness: ExternalWitnessPromotionBinding
    plan_selection: PlanSelectionArchiveBinding
    files: tuple[PromotionFileBinding, ...] = Field(min_length=len(_EXPECTED_PAYLOAD_PATHS))
    atomic_install: Literal[True] = True
    exact_file_inventory: Literal[True] = True
    external_proof_reverification_required: Literal[True] = True
    tier_a_benchmark_release_established: Literal[False] = False

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'scope precommit created_at')

    @field_validator('files')
    @classmethod
    def validate_files(cls, value: tuple[PromotionFileBinding, ...]) -> tuple[PromotionFileBinding, ...]:
        paths = tuple(item.path for item in value)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('scope precommit files must be sorted and unique')
        if len({path.casefold() for path in paths}) != len(paths):
            raise ValueError('scope precommit paths cannot collide under case folding')
        if set(paths) != _EXPECTED_PAYLOAD_PATHS:
            raise ValueError('scope precommit manifest must bind the exact V0 payload inventory')
        return value

    @model_validator(mode='after')
    def validate_cross_bindings(self) -> Self:
        if checkpoint_sha256(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError('scope precommit checkpoint digest does not bind its canonical checkpoint')
        required = {
            self.scope_policy.path: (_SCOPE_POLICY_PATH, self.scope_policy),
            self.pre_capture_plan.path: (_PRE_CAPTURE_PLAN_PATH, self.pre_capture_plan),
            self.ledger_prefix.path: (_LEDGER_PREFIX_PATH, self.ledger_prefix),
            self.registered_jobs.path: (_REGISTERED_JOBS_PATH, self.registered_jobs),
            self.witness.witness_manifest.path: (_WITNESS_MANIFEST_PATH, self.witness.witness_manifest),
            self.witness.checkpoint_file.path: (_WITNESS_CHECKPOINT_PATH, self.witness.checkpoint_file),
            self.witness.proof_file.path: (_WITNESS_PROOF_PATH, self.witness.proof_file),
            self.witness.policy.path: (_WITNESS_POLICY_PATH, self.witness.policy),
            self.witness.trust_policy.path: (_WITNESS_TRUST_POLICY_PATH, self.witness.trust_policy),
            self.witness.verifier_implementation.path: (
                _WITNESS_VERIFIER_PATH,
                self.witness.verifier_implementation,
            ),
            self.plan_selection.selection_manifest.path: (
                _SELECTION_MANIFEST_PATH,
                self.plan_selection.selection_manifest,
            ),
            self.plan_selection.commitment.path: (
                _SELECTION_COMMITMENT_PATH,
                self.plan_selection.commitment,
            ),
            self.plan_selection.proof.path: (_SELECTION_PROOF_PATH, self.plan_selection.proof),
            self.plan_selection.policy.path: (_SELECTION_POLICY_PATH, self.plan_selection.policy),
            self.plan_selection.trust_policy.path: (
                _SELECTION_TRUST_POLICY_PATH,
                self.plan_selection.trust_policy,
            ),
            self.plan_selection.verifier_implementation.path: (
                _SELECTION_VERIFIER_PATH,
                self.plan_selection.verifier_implementation,
            ),
        }
        if any(actual != expected for actual, (expected, _binding) in required.items()):
            raise ValueError('scope precommit binding uses a noncanonical path')
        by_path = {item.path: item for item in self.files}
        if any(by_path.get(path) != binding for path, (_expected, binding) in required.items()):
            raise ValueError('scope precommit field binding differs from its file inventory')
        if self.witness.checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError('scope precommit witness binds a different checkpoint')
        if self.witness.witnessed_at < self.checkpoint.created_at:
            raise ValueError('scope precommit witness predates its checkpoint')
        if self.created_at < self.witness.witnessed_at:
            raise ValueError('scope precommit archive creation predates its authenticated witness')
        if self.created_at < self.plan_selection.selected_at_upper_bound:
            raise ValueError('scope precommit archive creation predates its authenticated plan selection')
        return self


def scope_precommit_sha256(manifest: ScopePrecommitManifest) -> str:
    """Return the archive identity: SHA-256 of the exact canonical manifest."""

    if not isinstance(manifest, ScopePrecommitManifest):
        raise TypeError('manifest must be a ScopePrecommitManifest')
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def derive_pre_capture_plan(
    *,
    scope_policy: PromotionScopePolicy,
    selection_policy: PlanSelectionPolicyBinding,
    capture_witness_policy: WitnessPolicyBinding,
    source_verifiers: Mapping[str, object],
    adapter: object,
) -> PreCapturePlan:
    """Derive the exact plan bytes that must be stored before checkpointing."""

    committed_verifiers: list[PrecommittedSourceVerifier] = []
    for source_id in sorted(source_verifiers):
        spec = source_verifiers[source_id]
        try:
            implementation = _required_bytes(spec, 'implementation_bytes')
            policy = _required_bytes(spec, 'policy_bytes')
            execution_environment = _required_bytes(spec, 'execution_environment_bytes')
            committed_verifiers.append(
                PrecommittedSourceVerifier(
                    source_id=source_id,
                    verifier_id=_required_string(spec, 'verifier_id'),
                    verifier_version=_required_string(spec, 'verifier_version'),
                    implementation_sha256=hashlib.sha256(implementation).hexdigest(),
                    policy_sha256=hashlib.sha256(policy).hexdigest(),
                    execution_environment_sha256=hashlib.sha256(execution_environment).hexdigest(),
                    hermetic_execution=_precommitted_hermetic_execution(spec),
                )
            )
        except (TypeError, ValueError) as error:
            raise ScopePrecommitIntegrityError(f'invalid source verifier in pre-capture plan: {source_id}') from error
    try:
        allowed_codes = getattr(adapter, 'allowed_exclusion_reason_codes')
        if not isinstance(allowed_codes, tuple) or any(not isinstance(code, str) for code in allowed_codes):
            raise TypeError('allowed_exclusion_reason_codes must be a tuple of strings')
        committed_adapter = PrecommittedAdapter(
            adapter_id=_required_string(adapter, 'adapter_id'),
            adapter_version=_required_string(adapter, 'adapter_version'),
            implementation_sha256=hashlib.sha256(_required_bytes(adapter, 'implementation_bytes')).hexdigest(),
            policy_sha256=hashlib.sha256(_required_bytes(adapter, 'policy_bytes')).hexdigest(),
            execution_environment_sha256=hashlib.sha256(
                _required_bytes(adapter, 'execution_environment_bytes')
            ).hexdigest(),
            hermetic_execution=_precommitted_hermetic_execution(adapter),
            allowed_exclusion_reason_codes=allowed_codes,
        )
        return PreCapturePlan(
            scope_policy=scope_policy,
            selection_policy=selection_policy,
            capture_witness_policy=capture_witness_policy,
            source_verifiers=tuple(committed_verifiers),
            adapter=committed_adapter,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ScopePrecommitIntegrityError(f'invalid pre-capture plan inputs: {error}') from error


def _required_bytes(value: object, name: str) -> bytes:
    attribute = getattr(value, name, None)
    if not isinstance(attribute, bytes) or not attribute:
        raise TypeError(f'{name} must be nonempty exact bytes')
    return attribute


def _required_string(value: object, name: str) -> str:
    attribute = getattr(value, name, None)
    if not isinstance(attribute, str):
        raise TypeError(f'{name} must be a string')
    return attribute


def _precommitted_hermetic_execution(value: object) -> PrecommittedHermeticExecution | None:
    execution = getattr(value, 'hermetic_execution', None)
    if execution is None:
        return None
    try:
        from vaxreplay.operations.hermetic_execution import HermeticSandboxPolicy

        sandbox_policy_bytes = _required_bytes(execution, 'sandbox_policy_bytes')
        seccomp_profile_bytes = _required_bytes(execution, 'seccomp_profile_bytes')
        trusted_public_key_bytes = _required_bytes(execution, 'trusted_public_key_bytes')
        sandbox_policy = HermeticSandboxPolicy.model_validate_json(sandbox_policy_bytes)
    except (TypeError, ValueError) as error:
        raise TypeError('hermetic execution uses invalid sandbox or trust material') from error
    if canonical_json_bytes(sandbox_policy) != sandbox_policy_bytes:
        raise TypeError('hermetic sandbox policy must use exact canonical JSON bytes')
    trusted_public_key_sha256 = hashlib.sha256(trusted_public_key_bytes).hexdigest()
    if trusted_public_key_sha256 != sandbox_policy.signing_public_key_sha256:
        raise TypeError('hermetic receipt public key differs from the sandbox policy')
    if hashlib.sha256(seccomp_profile_bytes).hexdigest() != sandbox_policy.seccomp_profile_sha256:
        raise TypeError('hermetic seccomp profile differs from the sandbox policy')
    return PrecommittedHermeticExecution(
        sandbox_policy_sha256=hashlib.sha256(sandbox_policy_bytes).hexdigest(),
        trusted_public_key_sha256=trusted_public_key_sha256,
        seccomp_profile_sha256=sandbox_policy.seccomp_profile_sha256,
        authority_id=sandbox_policy.authority_id,
        signing_key_id=sandbox_policy.signing_key_id,
    )


@dataclass(frozen=True)
class LoadedScopePrecommit:
    """Freshly verified portable scope commitment; not an admission capability."""

    root: Path
    manifest: ScopePrecommitManifest
    manifest_bytes: bytes
    archive_sha256: str
    scope_policy: PromotionScopePolicy
    pre_capture_plan: PreCapturePlan
    plan_selection_manifest: PlanSelectionManifest
    plan_selection_commitment: PlanSelectionCommitment
    checkpoint: LedgerCheckpoint
    ledger_events: tuple[LedgerEvent, ...]
    registered_jobs: tuple[RegisteredJob, ...]
    file_payloads: tuple[tuple[str, bytes], ...]

    @property
    def manifest_sha256(self) -> str:
        """Compatibility name for the archive's canonical-manifest digest."""

        return self.archive_sha256


def build_scope_precommit(
    output_dir: Path,
    *,
    store: OperationalStore,
    scope_policy: PromotionScopePolicy,
    pre_capture_plan: PreCapturePlan,
    witness_root: Path,
    witness_materials: WitnessMaterialSpecProtocol,
    selection_root: Path,
    expected_selection_manifest_sha256: str,
    selection_materials: PlanSelectionMaterialSpecProtocol,
    created_at: datetime,
    verified_at: datetime,
) -> LoadedScopePrecommit:
    """Build, durably publish, and fully reload one witnessed scope commitment."""

    if not isinstance(store, OperationalStore):
        raise TypeError('store must be an OperationalStore')
    scope_policy = _validated_model(scope_policy, PromotionScopePolicy, 'scope policy')
    pre_capture_plan = _validated_model(pre_capture_plan, PreCapturePlan, 'pre-capture plan')
    if pre_capture_plan.scope_policy != scope_policy:
        raise ScopePrecommitIntegrityError('pre-capture plan does not embed the exact scope policy')
    created_at = aware_utc(created_at, 'created_at')
    verified_at = aware_utc(verified_at, 'verified_at')
    if created_at > verified_at:
        raise ScopePrecommitIntegrityError('scope precommit creation postdates verification')
    _validate_witness_materials(witness_materials)

    witnessed = load_witnessed_checkpoint(
        witness_root,
        verifier=witness_materials.verifier,
        expected_policy=witness_materials.policy,
        verified_at=verified_at,
    )
    _verify_store_and_time(
        store_id=store.store_id,
        scope_policy=scope_policy,
        checkpoint=witnessed.checkpoint,
        witnessed_at=witnessed.witnessed_at,
        created_at=created_at,
        verified_at=verified_at,
    )

    selection_commitment = derive_plan_selection_commitment(
        scope_policy,
        pre_capture_plan,
        witnessed.checkpoint,
    )
    try:
        selected_plan = load_plan_selection(
            selection_root,
            expected_commitment=selection_commitment,
            expected_manifest_sha256=expected_selection_manifest_sha256,
            materials=selection_materials,
            verified_at=verified_at,
        )
    except (PlanSelectionIntegrityError, TypeError) as error:
        raise ScopePrecommitIntegrityError(f'invalid externally selected pre-capture plan: {error}') from error

    scope_bytes = canonical_json_bytes(scope_policy)
    plan_bytes = canonical_json_bytes(pre_capture_plan)
    with store.verification_window():
        store.verify(checkpoint=witnessed.checkpoint, verified_at=verified_at)
        ledger_bytes = export_ledger_prefix(store, witnessed.checkpoint)
        events = parse_ledger_prefix(ledger_bytes, witnessed.checkpoint)
        jobs = _load_committed_jobs(store, scope_policy, events)
        _require_committed_artifact(store, events, scope_bytes, 'scope policy')
        _require_committed_artifact(store, events, plan_bytes, 'pre-capture plan')
    jobs_bytes = _records_bytes(jobs)

    payloads: dict[str, bytes] = {
        _SCOPE_POLICY_PATH: scope_bytes,
        _PRE_CAPTURE_PLAN_PATH: plan_bytes,
        _LEDGER_PREFIX_PATH: ledger_bytes,
        _REGISTERED_JOBS_PATH: jobs_bytes,
        _WITNESS_MANIFEST_PATH: canonical_json_bytes(witnessed.manifest),
        _WITNESS_CHECKPOINT_PATH: witnessed.checkpoint_bytes,
        _WITNESS_PROOF_PATH: witnessed.proof_bytes,
        _WITNESS_POLICY_PATH: witness_materials.policy_bytes,
        _WITNESS_TRUST_POLICY_PATH: witness_materials.trust_policy_bytes,
        _WITNESS_VERIFIER_PATH: witness_materials.verifier_implementation_bytes,
        _SELECTION_MANIFEST_PATH: selected_plan.manifest_bytes,
        _SELECTION_COMMITMENT_PATH: selected_plan.commitment_bytes,
        _SELECTION_PROOF_PATH: selected_plan.proof_bytes,
        _SELECTION_POLICY_PATH: selection_materials.policy_bytes,
        _SELECTION_TRUST_POLICY_PATH: selection_materials.trust_policy_bytes,
        _SELECTION_VERIFIER_PATH: selection_materials.verifier_implementation_bytes,
    }
    _validate_payloads(payloads)
    manifest = ScopePrecommitManifest(
        created_at=created_at,
        checkpoint_sha256=checkpoint_sha256(witnessed.checkpoint),
        checkpoint=witnessed.checkpoint,
        scope_policy=_binding(_SCOPE_POLICY_PATH, scope_bytes),
        pre_capture_plan=_binding(_PRE_CAPTURE_PLAN_PATH, plan_bytes),
        ledger_prefix=_binding(_LEDGER_PREFIX_PATH, ledger_bytes),
        registered_jobs=_binding(_REGISTERED_JOBS_PATH, jobs_bytes),
        witness=_witness_binding(payloads, witnessed.manifest),
        plan_selection=_plan_selection_binding(
            payloads,
            selected_plan.manifest,
            selected_plan.commitment,
        ),
        files=tuple(_binding(path, payloads[path]) for path in sorted(payloads)),
    )
    manifest_bytes = canonical_json_bytes(manifest)
    target = _durable_publish(output_dir, payloads, manifest_bytes)
    return load_scope_precommit(
        target,
        expected_archive_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        expected_scope_policy=scope_policy,
        expected_pre_capture_plan=pre_capture_plan,
        witness_materials=witness_materials,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
        selection_materials=selection_materials,
        verified_at=verified_at,
    )


def load_scope_precommit(
    root: Path,
    *,
    expected_archive_sha256: str,
    expected_scope_policy: PromotionScopePolicy,
    expected_pre_capture_plan: PreCapturePlan,
    witness_materials: WitnessMaterialSpecProtocol,
    expected_selection_manifest_sha256: str,
    selection_materials: PlanSelectionMaterialSpecProtocol,
    verified_at: datetime,
) -> LoadedScopePrecommit:
    """Load only after exact-inventory, ledger, and external-proof verification."""

    if not isinstance(expected_archive_sha256, str) or _SHA256_RE.fullmatch(expected_archive_sha256) is None:
        raise ScopePrecommitIntegrityError('expected archive SHA-256 must be a lowercase digest')
    expected_scope_policy = _validated_model(
        expected_scope_policy,
        PromotionScopePolicy,
        'expected scope policy',
    )
    expected_pre_capture_plan = _validated_model(
        expected_pre_capture_plan,
        PreCapturePlan,
        'expected pre-capture plan',
    )
    if expected_pre_capture_plan.scope_policy != expected_scope_policy:
        raise ScopePrecommitIntegrityError('expected pre-capture plan embeds a different scope policy')
    verified_at = aware_utc(verified_at, 'verified_at')
    _validate_witness_materials(witness_materials)

    try:
        snapshot = snapshot_immutable_tree(
            root,
            max_files=_MAX_FILES,
            max_directories=16,
            max_file_bytes=_MAX_FILE_BYTES,
            max_total_bytes=_MAX_TOTAL_BYTES,
            max_path_characters=1024,
            per_path_byte_limits={_MANIFEST_PATH: _MAX_MANIFEST_BYTES},
        )
        snapshot.require_exact_files({_MANIFEST_PATH, *_EXPECTED_PAYLOAD_PATHS})
    except ImmutableTreeError as error:
        raise ScopePrecommitIntegrityError(f'unsafe scope precommit archive: {error}') from error
    resolved = snapshot.root
    manifest_bytes = snapshot.files[_MANIFEST_PATH]
    try:
        manifest = ScopePrecommitManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise ScopePrecommitIntegrityError(f'invalid scope precommit manifest: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ScopePrecommitIntegrityError('scope precommit manifest is not canonical JSON')
    archive_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if archive_sha256 != expected_archive_sha256:
        raise ScopePrecommitIntegrityError('scope precommit archive differs from the expected digest')
    payloads = {path: snapshot.files[path] for path in _EXPECTED_PAYLOAD_PATHS}
    for binding in manifest.files:
        payload = payloads[binding.path]
        if _binding(binding.path, payload) != binding:
            raise ScopePrecommitIntegrityError(f'scope precommit payload differs from its binding: {binding.path}')

    scope_bytes = payloads[_SCOPE_POLICY_PATH]
    plan_bytes = payloads[_PRE_CAPTURE_PLAN_PATH]
    if scope_bytes != canonical_json_bytes(expected_scope_policy):
        raise ScopePrecommitIntegrityError('scope policy differs from the independently supplied policy')
    if plan_bytes != canonical_json_bytes(expected_pre_capture_plan):
        raise ScopePrecommitIntegrityError('pre-capture plan differs from the independently supplied plan')
    if _binding(_SCOPE_POLICY_PATH, scope_bytes) != manifest.scope_policy:
        raise ScopePrecommitIntegrityError('scope policy differs from its manifest binding')
    if _binding(_PRE_CAPTURE_PLAN_PATH, plan_bytes) != manifest.pre_capture_plan:
        raise ScopePrecommitIntegrityError('pre-capture plan differs from its manifest binding')

    _require_exact_material(payloads, _WITNESS_POLICY_PATH, witness_materials.policy_bytes)
    _require_exact_material(payloads, _WITNESS_TRUST_POLICY_PATH, witness_materials.trust_policy_bytes)
    _require_exact_material(
        payloads,
        _WITNESS_VERIFIER_PATH,
        witness_materials.verifier_implementation_bytes,
    )
    witness_manifest_bytes = payloads[_WITNESS_MANIFEST_PATH]
    try:
        witness_manifest = WitnessedCheckpointManifest.model_validate_json(witness_manifest_bytes)
        checkpoint = LedgerCheckpoint.model_validate_json(payloads[_WITNESS_CHECKPOINT_PATH])
    except ValueError as error:
        raise ScopePrecommitIntegrityError(f'invalid copied witness sidecar: {error}') from error
    if witness_manifest_bytes != canonical_json_bytes(witness_manifest):
        raise ScopePrecommitIntegrityError('copied witness manifest is not canonical JSON')
    if payloads[_WITNESS_CHECKPOINT_PATH] != canonical_json_bytes(checkpoint):
        raise ScopePrecommitIntegrityError('copied witness checkpoint is not canonical JSON')
    verify_witnessed_checkpoint(
        checkpoint,
        checkpoint_bytes=payloads[_WITNESS_CHECKPOINT_PATH],
        manifest=witness_manifest,
        proof_bytes=payloads[_WITNESS_PROOF_PATH],
        verifier=witness_materials.verifier,
        expected_policy=witness_materials.policy,
        verified_at=verified_at,
        expected_checkpoint_sha256=manifest.checkpoint_sha256,
    )
    if checkpoint != manifest.checkpoint:
        raise ScopePrecommitIntegrityError('copied witness checkpoint differs from the archive manifest')
    if _witness_binding(payloads, witness_manifest) != manifest.witness:
        raise ScopePrecommitIntegrityError('copied witness sidecar differs from the archive witness binding')

    selection_commitment = derive_plan_selection_commitment(
        expected_scope_policy,
        expected_pre_capture_plan,
        checkpoint,
    )
    try:
        selected_plan = verify_plan_selection_bytes(
            commitment_bytes=payloads[_SELECTION_COMMITMENT_PATH],
            proof_bytes=payloads[_SELECTION_PROOF_PATH],
            manifest_bytes=payloads[_SELECTION_MANIFEST_PATH],
            expected_commitment=selection_commitment,
            expected_manifest_sha256=expected_selection_manifest_sha256,
            materials=selection_materials,
            verified_at=verified_at,
        )
    except (PlanSelectionIntegrityError, TypeError) as error:
        raise ScopePrecommitIntegrityError(f'invalid copied plan-selection sidecar: {error}') from error
    _require_exact_material(
        payloads,
        _SELECTION_POLICY_PATH,
        selection_materials.policy_bytes,
        'plan-selection policy',
    )
    _require_exact_material(
        payloads,
        _SELECTION_TRUST_POLICY_PATH,
        selection_materials.trust_policy_bytes,
        'plan-selection trust policy',
    )
    _require_exact_material(
        payloads,
        _SELECTION_VERIFIER_PATH,
        selection_materials.verifier_implementation_bytes,
        'plan-selection verifier implementation',
    )
    if _plan_selection_binding(payloads, selected_plan.manifest, selected_plan.commitment) != manifest.plan_selection:
        raise ScopePrecommitIntegrityError('copied plan-selection sidecar differs from the archive selection binding')

    events = parse_ledger_prefix(payloads[_LEDGER_PREFIX_PATH], checkpoint)
    jobs = _parse_committed_jobs(payloads[_REGISTERED_JOBS_PATH], expected_scope_policy, events)
    _require_committed_artifact_from_events(events, scope_bytes, 'scope policy')
    _require_committed_artifact_from_events(events, plan_bytes, 'pre-capture plan')
    _verify_store_and_time(
        store_id=checkpoint.store_id,
        scope_policy=expected_scope_policy,
        checkpoint=checkpoint,
        witnessed_at=witness_manifest.receipt.witnessed_at,
        created_at=manifest.created_at,
        verified_at=verified_at,
    )
    return LoadedScopePrecommit(
        root=resolved,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        archive_sha256=archive_sha256,
        scope_policy=expected_scope_policy,
        pre_capture_plan=expected_pre_capture_plan,
        plan_selection_manifest=selected_plan.manifest,
        plan_selection_commitment=selected_plan.commitment,
        checkpoint=checkpoint,
        ledger_events=events,
        registered_jobs=jobs,
        file_payloads=tuple((path, payloads[path]) for path in sorted(payloads)),
    )


def _validated_model[ModelT: StrictModel](value: object, model: type[ModelT], label: str) -> ModelT:
    if not isinstance(value, model):
        raise TypeError(f'{label} must be a {model.__name__}')
    try:
        return model.model_validate_json(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ScopePrecommitIntegrityError(f'invalid {label}: {error}') from error


def _validate_witness_materials(materials: WitnessMaterialSpecProtocol) -> None:
    try:
        policy = materials.policy
        policy_bytes = materials.policy_bytes
        trust_policy_bytes = materials.trust_policy_bytes
        verifier_bytes = materials.verifier_implementation_bytes
        verifier = materials.verifier
    except AttributeError as error:
        raise TypeError('witness_materials does not satisfy the required structural interface') from error
    if not isinstance(policy, WitnessPolicyBinding):
        raise TypeError('witness material policy must be a WitnessPolicyBinding')
    pinned = (
        ('witness policy', policy_bytes, policy.policy_sha256),
        ('witness trust policy', trust_policy_bytes, policy.trust_policy_sha256),
        ('witness verifier implementation', verifier_bytes, policy.verifier_implementation_sha256),
    )
    for label, payload, expected_sha256 in pinned:
        if not isinstance(payload, bytes) or not payload or hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ScopePrecommitIntegrityError(f'{label} differs from its independently pinned digest')
    if verifier is None:  # type: ignore[comparison-overlap]
        raise ScopePrecommitIntegrityError('trusted external witness verifier is required')


def _verify_store_and_time(
    *,
    store_id: str,
    scope_policy: PromotionScopePolicy,
    checkpoint: LedgerCheckpoint,
    witnessed_at: datetime,
    created_at: datetime,
    verified_at: datetime,
) -> None:
    if store_id != scope_policy.store_id or checkpoint.store_id != scope_policy.store_id:
        raise ScopePrecommitIntegrityError('scope, operations store, and witnessed checkpoint use different stores')
    earliest_slot = min(source.scheduled_from for source in scope_policy.sources)
    if witnessed_at >= earliest_slot:
        raise ScopePrecommitIntegrityError(
            'scope precommit witness is not strictly before the first scheduled capture slot'
        )
    if created_at < witnessed_at or created_at > verified_at:
        raise ScopePrecommitIntegrityError('scope precommit creation is outside its witness/verification interval')


def _load_committed_jobs(
    store: OperationalStore,
    scope_policy: PromotionScopePolicy,
    events: tuple[LedgerEvent, ...],
) -> tuple[RegisteredJob, ...]:
    expected = _scoped_job_digests(scope_policy)
    event_by_digest = _job_registration_events(events)
    if expected - set(event_by_digest):
        raise ScopePrecommitIntegrityError('a scoped job registration is outside the witnessed ledger prefix')
    jobs: list[RegisteredJob] = []
    for digest in sorted(expected):
        job = store.get_job(digest)
        event = event_by_digest[digest]
        if (
            job.spec_sha256 != digest
            or event.payload.get('job_id') != job.spec.job_id
            or event.occurred_at != job.registered_at
        ):
            raise ScopePrecommitIntegrityError('scoped job does not reproduce its witnessed registration event')
        jobs.append(job)
    result = tuple(jobs)
    _validate_scoped_job_slots(result, scope_policy)
    return result


def _parse_committed_jobs(
    payload: bytes,
    scope_policy: PromotionScopePolicy,
    events: tuple[LedgerEvent, ...],
) -> tuple[RegisteredJob, ...]:
    jobs = _parse_model_jsonl(payload, RegisteredJob, 'registered job')
    digests = tuple(job.spec_sha256 for job in jobs)
    expected = tuple(sorted(_scoped_job_digests(scope_policy)))
    if digests != expected or len(digests) != len(set(digests)):
        raise ScopePrecommitIntegrityError('portable job inventory does not exactly match the committed scope')
    event_by_digest = _job_registration_events(events)
    if set(expected) - set(event_by_digest):
        raise ScopePrecommitIntegrityError('scoped job registration is absent from the witnessed prefix')
    for job in jobs:
        event = event_by_digest[job.spec_sha256]
        if event.payload.get('job_id') != job.spec.job_id or event.occurred_at != job.registered_at:
            raise ScopePrecommitIntegrityError('portable job does not reproduce its witnessed registration event')
    _validate_scoped_job_slots(jobs, scope_policy)
    return jobs


def _scoped_job_digests(scope_policy: PromotionScopePolicy) -> set[str]:
    return {digest for source in scope_policy.sources for digest in source.job_spec_sha256s}


def _job_registration_events(events: tuple[LedgerEvent, ...]) -> dict[str, LedgerEvent]:
    result: dict[str, LedgerEvent] = {}
    for event in events:
        if event.event_type is not LedgerEventType.JOB_REGISTERED:
            continue
        digest = event.payload.get('job_spec_sha256')
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None or digest in result:
            raise ScopePrecommitIntegrityError('witnessed ledger has a malformed or duplicate job registration')
        result[digest] = event
    return result


def _validate_scoped_job_slots(
    jobs: tuple[RegisteredJob, ...],
    scope_policy: PromotionScopePolicy,
) -> None:
    by_digest = {job.spec_sha256: job for job in jobs}
    slot_count = 0
    for source_scope in scope_policy.sources:
        for digest in source_scope.job_spec_sha256s:
            job = by_digest.get(digest)
            if job is None:
                raise ScopePrecommitIntegrityError('scope references an absent registered job')
            if job.spec.configuration.get('source_id') != source_scope.source_id:
                raise ScopePrecommitIntegrityError('scoped job source differs from its pre-capture scope')
            interval_microseconds = job.spec.schedule_interval_seconds * 1_000_000
            start_delta = source_scope.scheduled_from - job.spec.schedule_anchor_at
            end_delta = source_scope.scheduled_through - job.spec.schedule_anchor_at
            start_microseconds = _timedelta_microseconds(start_delta)
            end_microseconds = _timedelta_microseconds(end_delta)
            if (
                start_microseconds < 0
                or end_microseconds < 0
                or start_microseconds % interval_microseconds
                or end_microseconds % interval_microseconds
            ):
                raise ScopePrecommitIntegrityError('scope boundaries must be exact registered-job schedule slots')
            slot_count += (end_microseconds - start_microseconds) // interval_microseconds + 1
            if slot_count > _MAX_SCHEDULED_SLOTS:
                raise ScopePrecommitIntegrityError('scope exceeds the pre-capture scheduled-slot limit')


def _timedelta_microseconds(value: timedelta) -> int:
    return ((value.days * 86_400) + value.seconds) * 1_000_000 + value.microseconds


def _require_committed_artifact(
    store: OperationalStore,
    events: tuple[LedgerEvent, ...],
    payload: bytes,
    label: str,
) -> None:
    _require_committed_artifact_from_events(events, payload, label)
    digest = hashlib.sha256(payload).hexdigest()
    if store.read_artifact(digest, max_bytes=len(payload)) != payload:
        raise ScopePrecommitIntegrityError(f'{label} CAS object differs from its committed exact bytes')


def _require_committed_artifact_from_events(
    events: tuple[LedgerEvent, ...],
    payload: bytes,
    label: str,
) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    matches = tuple(
        event
        for event in events
        if event.event_type is LedgerEventType.ARTIFACT_STORED and event.payload.get('artifact_sha256') == digest
    )
    if len(matches) != 1 or matches[0].payload.get('byte_count') != len(payload):
        raise ScopePrecommitIntegrityError(f'{label} exact bytes were not stored inside the witnessed ledger prefix')


def _parse_model_jsonl[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    label: str,
) -> tuple[ModelT, ...]:
    if not payload or not payload.endswith(b'\n'):
        raise ScopePrecommitIntegrityError(f'portable {label} inventory must be nonempty canonical JSONL')
    raw_lines = payload.splitlines()
    if not raw_lines or any(not line for line in raw_lines):
        raise ScopePrecommitIntegrityError(f'portable {label} inventory contains blank records')
    records: list[ModelT] = []
    for line in raw_lines:
        try:
            record = model.model_validate_json(line)
        except ValueError as error:
            raise ScopePrecommitIntegrityError(f'invalid portable {label}: {error}') from error
        if line != canonical_json_bytes(record):
            raise ScopePrecommitIntegrityError(f'portable {label} record is not canonical JSON')
        records.append(record)
    return tuple(records)


def _records_bytes(records: Sequence[StrictModel]) -> bytes:
    if not records:
        raise ScopePrecommitIntegrityError('portable registered-job inventory cannot be empty')
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


def derive_plan_selection_commitment(
    scope_policy: PromotionScopePolicy,
    pre_capture_plan: PreCapturePlan,
    checkpoint: LedgerCheckpoint,
) -> PlanSelectionCommitment:
    """Derive selection identity only from the exact witnessed scope archive inputs."""

    if pre_capture_plan.scope_policy != scope_policy:
        raise ScopePrecommitIntegrityError('pre-capture plan embeds a different selection scope')
    try:
        return PlanSelectionCommitment(
            policy=pre_capture_plan.selection_policy,
            store_id=checkpoint.store_id,
            checkpoint_sha256=checkpoint_sha256(checkpoint),
            checkpoint_created_at=checkpoint.created_at,
            scope_policy_sha256=hashlib.sha256(canonical_json_bytes(scope_policy)).hexdigest(),
            pre_capture_plan_sha256=hashlib.sha256(canonical_json_bytes(pre_capture_plan)).hexdigest(),
            earliest_scheduled_slot=min(source.scheduled_from for source in scope_policy.sources),
        )
    except ValueError as error:
        raise ScopePrecommitIntegrityError(f'invalid derived plan-selection commitment: {error}') from error


def _plan_selection_binding(
    payloads: Mapping[str, bytes],
    manifest: PlanSelectionManifest,
    commitment: PlanSelectionCommitment,
) -> PlanSelectionArchiveBinding:
    if canonical_json_bytes(commitment) != payloads[_SELECTION_COMMITMENT_PATH]:
        raise ScopePrecommitIntegrityError('plan-selection commitment model differs from its copied exact bytes')
    receipt = manifest.receipt
    facts = receipt.facts
    policy = receipt.policy
    return PlanSelectionArchiveBinding(
        selection_manifest=_binding(_SELECTION_MANIFEST_PATH, payloads[_SELECTION_MANIFEST_PATH]),
        commitment=_binding(_SELECTION_COMMITMENT_PATH, payloads[_SELECTION_COMMITMENT_PATH]),
        proof=_binding(_SELECTION_PROOF_PATH, payloads[_SELECTION_PROOF_PATH]),
        policy=_binding(_SELECTION_POLICY_PATH, payloads[_SELECTION_POLICY_PATH]),
        trust_policy=_binding(_SELECTION_TRUST_POLICY_PATH, payloads[_SELECTION_TRUST_POLICY_PATH]),
        verifier_implementation=_binding(_SELECTION_VERIFIER_PATH, payloads[_SELECTION_VERIFIER_PATH]),
        manifest_sha256=hashlib.sha256(payloads[_SELECTION_MANIFEST_PATH]).hexdigest(),
        commitment_sha256=hashlib.sha256(payloads[_SELECTION_COMMITMENT_PATH]).hexdigest(),
        selected_at_upper_bound=facts.selected_at_upper_bound,
        registry_entry_id=facts.registry_entry_id,
        registry_sequence=facts.registry_sequence,
        signed_checkpoint_sha256=facts.signed_checkpoint_sha256,
        signed_checkpoint_size=facts.signed_checkpoint_size,
        valid_inclusion_proof=facts.valid_inclusion_proof,
        consistent_from_pinned_trust_checkpoint=facts.consistent_from_pinned_trust_checkpoint,
        selection_key_history_count=facts.selection_key_history_count,
        key_previously_unassigned=facts.key_previously_unassigned,
        atomic_first_write_wins_enforced=facts.atomic_first_write_wins_enforced,
        selection_final_and_immutable=facts.selection_final_and_immutable,
        registry_id=policy.registry_id,
        authority_id=policy.authority_id,
        campaign_id=policy.campaign_id,
        selection_key=policy.selection_key,
        policy_id=policy.policy_id,
        policy_sha256=policy.policy_sha256,
        trust_policy_id=policy.trust_policy_id,
        trust_policy_sha256=policy.trust_policy_sha256,
        verifier_id=policy.verifier_id,
        verifier_implementation_sha256=policy.verifier_implementation_sha256,
        proof_sha256=receipt.proof_sha256,
        proof_byte_count=receipt.proof_bytes,
        uniqueness_semantics=policy.uniqueness_semantics,
        finality_semantics=policy.finality_semantics,
    )


def _witness_binding(
    payloads: Mapping[str, bytes],
    manifest: WitnessedCheckpointManifest,
) -> ExternalWitnessPromotionBinding:
    receipt = manifest.receipt
    return ExternalWitnessPromotionBinding(
        witness_manifest=_binding(_WITNESS_MANIFEST_PATH, payloads[_WITNESS_MANIFEST_PATH]),
        checkpoint_file=_binding(_WITNESS_CHECKPOINT_PATH, payloads[_WITNESS_CHECKPOINT_PATH]),
        proof_file=_binding(_WITNESS_PROOF_PATH, payloads[_WITNESS_PROOF_PATH]),
        policy=_binding(_WITNESS_POLICY_PATH, payloads[_WITNESS_POLICY_PATH]),
        trust_policy=_binding(_WITNESS_TRUST_POLICY_PATH, payloads[_WITNESS_TRUST_POLICY_PATH]),
        verifier_implementation=_binding(_WITNESS_VERIFIER_PATH, payloads[_WITNESS_VERIFIER_PATH]),
        checkpoint_sha256=receipt.checkpoint_sha256,
        witnessed_at=receipt.witnessed_at,
        authority_id=receipt.authority_id,
        witness_id=receipt.witness_id,
        method=receipt.method,
        policy_id=receipt.policy_id,
        policy_sha256=receipt.policy_sha256,
        trust_policy_id=receipt.trust_policy_id,
        trust_policy_sha256=receipt.trust_policy_sha256,
        verifier_id=receipt.verifier_id,
        verifier_implementation_sha256=receipt.verifier_implementation_sha256,
        proof_sha256=receipt.proof_sha256,
        proof_byte_count=receipt.proof_bytes,
    )


def _binding(path: str, payload: bytes) -> PromotionFileBinding:
    return PromotionFileBinding(path=path, sha256=hashlib.sha256(payload).hexdigest(), byte_count=len(payload))


def _require_exact_material(
    payloads: Mapping[str, bytes],
    path: str,
    expected: bytes,
    label: str = 'witness material',
) -> None:
    if payloads.get(path) != expected:
        raise ScopePrecommitIntegrityError(f'copied trusted {label} differs from independently supplied bytes: {path}')


def _validate_payloads(payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != _EXPECTED_PAYLOAD_PATHS:
        raise ScopePrecommitIntegrityError('scope precommit builder has a non-exact V0 payload inventory')
    total = 0
    for path, payload in payloads.items():
        normalized = PurePosixPath(path)
        if normalized.is_absolute() or '..' in normalized.parts or normalized.as_posix() != path:
            raise ScopePrecommitIntegrityError('scope precommit contains a non-portable payload path')
        if not isinstance(payload, bytes) or len(payload) > _MAX_FILE_BYTES:
            raise ScopePrecommitIntegrityError(f'scope precommit file exceeds its size limit: {path}')
        total += len(payload)
    if total > _MAX_TOTAL_BYTES:
        raise ScopePrecommitIntegrityError('scope precommit exceeds the aggregate byte limit')


def _durable_publish(output_dir: Path, payloads: Mapping[str, bytes], manifest_bytes: bytes) -> Path:
    _validate_payloads(payloads)
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
        raise ScopePrecommitIntegrityError(f'scope precommit publication is already locked: {target}') from error
    staging: Path | None = None
    installed = False
    try:
        os.fsync(lock_descriptor)
        fsync_directory(parent)
        if os.path.lexists(target):
            raise ScopePrecommitIntegrityError(f'scope precommit output already exists: {target}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=parent))
        for relative, payload in sorted(payloads.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
        manifest_path = staging / _MANIFEST_PATH
        with manifest_path.open('xb') as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        for directory in sorted((entry for entry in staging.rglob('*') if entry.is_dir()), reverse=True):
            directory.chmod(0o555)
            fsync_directory(directory)
        staging.chmod(0o555)
        fsync_directory(staging)
        try:
            rename_directory_noreplace(staging, target)
        except FileExistsError as error:
            raise ScopePrecommitIntegrityError(f'scope precommit output already exists: {target}') from error
        installed = True
        fsync_directory(parent)
        return target
    finally:
        if staging is not None and not installed:
            try:
                staging.chmod(0o755)
                for directory in (entry for entry in staging.rglob('*') if entry.is_dir()):
                    directory.chmod(0o755)
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
