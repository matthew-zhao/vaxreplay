"""Promote witnessed operational captures into a portable, replayable artifact.

The live operations store is a coordination database.  This module creates the
portable handoff: it copies the witnessed ledger prefix, every raw attachment for
every in-scope successful slot, independently pinned verification materials, and
deterministic normalized records.  Loading repeats every trusted verification from
the copied bytes and independently supplied policies/callbacks.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from vaxreplay._atomic import rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord, StrictModel
from vaxreplay.operations._immutable_tree import (
    ImmutableTreeError,
    ImmutableTreeSnapshot,
    immutable_root_identity,
    snapshot_immutable_tree,
)
from vaxreplay.operations.collector import (
    StaticHttpsCollectionPlan,
    StaticHttpsRunManifest,
    load_static_run_manifest,
)
from vaxreplay.operations.hermetic_callback_protocol import (
    HermeticAdapterInput,
    HermeticAdapterOutput,
    HermeticAdapterSourceInput,
    HermeticArtifactPayload,
    HermeticCapturePayload,
    HermeticNamedOutput,
    HermeticSourceVerifierInput,
    HermeticSourceVerifierOutput,
    decode_callback_bytes,
    encode_callback_bytes,
    parse_adapter_output,
    parse_source_output,
)
from vaxreplay.operations.hermetic_execution import (
    ExecutionPurpose,
    HermeticCallbackExecutor,
    HermeticCallbackMaterials,
    HermeticExecutionBundle,
    HermeticOciEnvironment,
    HermeticSandboxPolicy,
    verify_hermetic_execution_bundle,
)
from vaxreplay.operations.http_capture import HttpsCaptureReceipt, prepared_request_headers
from vaxreplay.operations.immport_capture import (
    MAX_IMMPORT_CAPTURE_BODY_BYTES,
    MAX_IMMPORT_PLAN_BYTES,
    MAX_IMMPORT_RECEIPT_BYTES,
    MAX_IMMPORT_RUN_MANIFEST_BYTES,
    ImmportAuthenticatedCollectionPlan,
    ImmportAuthenticatedRunManifest,
    immport_authenticated_plan_sha256,
)
from vaxreplay.operations.plan_selection import (
    PlanSelectionMaterialSpecProtocol,
    plan_selection_commitment_sha256,
)
from vaxreplay.operations.policy import (
    IMMPORT_AUTHENTICATED_COLLECTOR_ID,
    STATIC_HTTPS_COLLECTOR_ID,
    ImmportAuthenticatedJobConfiguration,
    StaticHttpsJobConfiguration,
    SupportedCollectorJobConfiguration,
    parse_supported_collector_job_configuration,
)
from vaxreplay.operations.portable_ledger import export_ledger_prefix, parse_ledger_prefix
from vaxreplay.operations.promotion_schema import (
    AdapterBinding,
    AdapterInputInventoryBinding,
    CaptureIndex,
    CapturePromotionManifest,
    ExternalWitnessPromotionBinding,
    HermeticExecutionPromotionBinding,
    NormalizedOutputBinding,
    NormalizedOutputRole,
    NormalizedRecordReference,
    PreCapturePlan,
    PromotedCaptureBinding,
    PromotedRawArtifactBinding,
    PromotionFileBinding,
    PromotionHandoffDescriptor,
    PromotionIntegrityError,
    PromotionScopePolicy,
    ScopePrecommitPromotionBinding,
    SourceRecordBinding,
    SourceRecordDisposition,
    SourceVerificationBinding,
    SourceVerificationResult,
    SuccessfulRunDisposition,
    capture_index_sha256,
    capture_promotion_sha256,
    source_verification_result_sha256,
)
from vaxreplay.operations.schema import (
    AttemptLease,
    AttemptState,
    CaptureJobSpec,
    LedgerCheckpoint,
    LedgerEvent,
    LedgerEventType,
    RegisteredJob,
    aware_utc,
    checkpoint_bytes,
    checkpoint_sha256,
    scheduled_logical_run_id,
)
from vaxreplay.operations.scope_precommit import (
    LoadedScopePrecommit,
    derive_plan_selection_commitment,
    derive_pre_capture_plan,
    load_scope_precommit,
)
from vaxreplay.operations.store import OperationalStore
from vaxreplay.operations.witness import (
    LoadedWitnessedCheckpoint,
    TrustedCheckpointWitnessVerifier,
    WitnessedCheckpointManifest,
    WitnessPolicyBinding,
    load_witnessed_checkpoint,
    verify_witnessed_checkpoint,
)
from vaxreplay.prospective import (
    LoadedProspectiveDecisionPackage,
    ProspectiveSourceCaptureBinding,
    SourceCaptureArtifact,
    build_prospective_decision_package,
)
from vaxreplay.temporal_schema import DecisionTimeConfig

_PROMOTION_MANIFEST = 'promotion.json'
_CAPTURE_INDEX = 'capture-index.json'
_LEDGER_PREFIX = 'ledger/events.jsonl'
_REGISTERED_JOBS = 'ledger/jobs.jsonl'
_SCOPE_POLICY = 'scope/policy.json'
_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_FILES = 100_000
_MAX_DIRECTORIES = 100_000
_MAX_PATH_CHARACTERS = 1024
_MAX_SCHEDULED_SLOTS = 100_000


@dataclass(frozen=True)
class ExactPromotedArtifact:
    binding: PromotedRawArtifactBinding
    payload: bytes

    @property
    def role(self) -> str:
        return self.binding.role

    @property
    def sha256(self) -> str:
        return self.binding.file.sha256

    @property
    def byte_count(self) -> int:
        return self.binding.file.byte_count


@dataclass(frozen=True)
class ExactPromotedCapture:
    binding: PromotedCaptureBinding
    artifacts: tuple[ExactPromotedArtifact, ...]


@dataclass(frozen=True)
class SourceVerificationInput:
    source_id: str
    captures: tuple[ExactPromotedCapture, ...]
    capture_inventory_sha256: str


@dataclass(frozen=True)
class SourceVerifierRunResult:
    """Verifier result plus its exact canonical, sorted source-record JSONL."""

    result: SourceVerificationResult
    verified_records: bytes


type TrustedSourceVerifier = Callable[[SourceVerificationInput, bytes], SourceVerifierRunResult]


@dataclass(frozen=True)
class HermeticExecutionSpec:
    """Out-of-band executor and trust material for one precommitted OCI callback."""

    sandbox_policy_bytes: bytes
    seccomp_profile_bytes: bytes
    trusted_public_key_bytes: bytes
    executor: HermeticCallbackExecutor


@dataclass(frozen=True)
class SourceVerifierSpec:
    """Independently supplied verifier code, policy, environment, and callback."""

    verifier_id: str
    verifier_version: str
    implementation_bytes: bytes
    policy_bytes: bytes
    execution_environment_bytes: bytes
    verifier: TrustedSourceVerifier | None = None
    hermetic_execution: HermeticExecutionSpec | None = None


@dataclass(frozen=True)
class AdapterSourceInput:
    source_id: str
    captures: tuple[ExactPromotedCapture, ...]
    verification_result: SourceVerificationResult
    verified_records: tuple[SourceRecordBinding, ...]


@dataclass(frozen=True)
class AdapterRunResult:
    candidate_records: bytes
    evidence_records: bytes
    dispositions: bytes
    auxiliary_outputs: Mapping[str, bytes] | None = None


type TrustedPromotionAdapter = Callable[[tuple[AdapterSourceInput, ...], bytes], AdapterRunResult]


@dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    adapter_version: str
    implementation_bytes: bytes
    policy_bytes: bytes
    execution_environment_bytes: bytes
    adapter: TrustedPromotionAdapter | None = None
    hermetic_execution: HermeticExecutionSpec | None = None
    allowed_exclusion_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WitnessMaterialSpec:
    policy: WitnessPolicyBinding
    policy_bytes: bytes
    trust_policy_bytes: bytes
    verifier_implementation_bytes: bytes
    verifier: TrustedCheckpointWitnessVerifier


@dataclass(frozen=True)
class LoadedCapturePromotion:
    """Verified portable promotion; publicly constructible and not a capability."""

    root: Path
    manifest: CapturePromotionManifest
    manifest_sha256: str
    index: CaptureIndex
    index_bytes: bytes
    handoff_descriptor: PromotionHandoffDescriptor
    handoff_descriptor_bytes: bytes
    ledger_events: tuple[LedgerEvent, ...]
    candidates: tuple[CandidateRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    dispositions: tuple[SourceRecordDisposition, ...]
    auxiliary_outputs: Mapping[str, bytes]
    source_captures: tuple[SourceCaptureArtifact, ...]


def build_capture_promotion(
    output_dir: Path,
    *,
    promotion_id: str,
    store: OperationalStore,
    witness_root: Path,
    witness_materials: WitnessMaterialSpec,
    scope_policy: PromotionScopePolicy,
    scope_precommit_root: Path,
    scope_precommit_witness_materials: WitnessMaterialSpec,
    expected_scope_precommit_sha256: str,
    selection_materials: PlanSelectionMaterialSpecProtocol,
    expected_selection_manifest_sha256: str,
    source_verifiers: Mapping[str, SourceVerifierSpec],
    adapter: AdapterSpec,
    created_at: datetime,
    verified_at: datetime,
) -> LoadedCapturePromotion:
    """Build and then fully reload one immutable portable promotion."""

    created_at = aware_utc(created_at, 'created_at')
    verified_at = aware_utc(verified_at, 'verified_at')
    _validate_trusted_materials(witness_materials, source_verifiers, adapter)
    pre_capture_plan = derive_pre_capture_plan(
        scope_policy=scope_policy,
        selection_policy=selection_materials.policy,
        capture_witness_policy=witness_materials.policy,
        source_verifiers=source_verifiers,
        adapter=adapter,
    )
    scope_precommit = load_scope_precommit(
        scope_precommit_root,
        expected_archive_sha256=expected_scope_precommit_sha256,
        expected_scope_policy=scope_policy,
        expected_pre_capture_plan=pre_capture_plan,
        witness_materials=scope_precommit_witness_materials,
        selection_materials=selection_materials,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
        verified_at=verified_at,
    )
    _require_exact_plan_selection(
        scope_precommit,
        scope_policy=scope_policy,
        pre_capture_plan=pre_capture_plan,
        selection_materials=selection_materials,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
    )
    exact_scope_bytes = canonical_json_bytes(scope_policy)
    witnessed = load_witnessed_checkpoint(
        witness_root,
        verifier=witness_materials.verifier,
        expected_policy=witness_materials.policy,
        verified_at=verified_at,
    )
    if witnessed.checkpoint.store_id != scope_policy.store_id or store.store_id != scope_policy.store_id:
        raise PromotionIntegrityError(
            'witnessed checkpoint/store differ from the independently pinned scope policy store_id'
        )
    if not (
        scope_policy.checkpoint_created_at_not_before
        <= witnessed.checkpoint.created_at
        <= scope_policy.checkpoint_created_at_not_after
    ):
        raise PromotionIntegrityError('witnessed checkpoint lies outside the independently pinned scope window')
    if created_at < witnessed.witnessed_at or created_at > verified_at:
        raise PromotionIntegrityError('promotion creation must follow witness time and not postdate verification')

    with store.verification_window():
        store.verify(checkpoint=witnessed.checkpoint, verified_at=verified_at)
        ledger_bytes = export_ledger_prefix(store, witnessed.checkpoint)
        events = parse_ledger_prefix(ledger_bytes, witnessed.checkpoint)
        jobs = _load_scoped_jobs(store, scope_policy, events)
        jobs_bytes = _records_bytes(tuple(jobs[key] for key in sorted(jobs)))
        captures = _snapshot_scoped_captures(store, scope_policy, jobs, events)
    _require_scope_precommit_prefix(scope_precommit, events, witnessed.checkpoint.through_sequence)

    payloads: dict[str, bytes] = {
        _LEDGER_PREFIX: ledger_bytes,
        _REGISTERED_JOBS: jobs_bytes,
        _SCOPE_POLICY: exact_scope_bytes,
        'witness/sidecar/checkpoint.json': witnessed.checkpoint_bytes,
        'witness/sidecar/external-proof.bin': witnessed.proof_bytes,
        'witness/sidecar/witness.json': canonical_json_bytes(witnessed.manifest),
        'witness/materials/policy.bin': witness_materials.policy_bytes,
        'witness/materials/trust-policy.bin': witness_materials.trust_policy_bytes,
        'witness/materials/verifier-implementation.bin': witness_materials.verifier_implementation_bytes,
        'adapter/implementation.bin': adapter.implementation_bytes,
        'adapter/policy.bin': adapter.policy_bytes,
        'adapter/execution-environment.bin': adapter.execution_environment_bytes,
    }
    for relative, payload in scope_precommit.file_payloads:
        payloads[f'scope/precommit/{relative}'] = payload
    payloads['scope/precommit/scope-precommit.json'] = scope_precommit.manifest_bytes
    for capture in captures:
        for artifact in capture.artifacts:
            payloads[artifact.binding.file.path] = artifact.payload

    hermetic_bindings: list[HermeticExecutionPromotionBinding] = []
    source_bindings, adapter_inputs = _run_source_verifiers(
        captures,
        source_verifiers,
        payloads,
        hermetic_bindings=hermetic_bindings,
    )
    normalized_bindings, disposition_binding, _dispositions = _run_adapter_twice(
        adapter_inputs,
        adapter,
        payloads,
        hermetic_bindings=hermetic_bindings,
    )
    witness_binding = _witness_binding(payloads, witnessed, witness_materials)
    scope_precommit_binding = _scope_precommit_binding(scope_precommit)
    index = CaptureIndex(
        promotion_id=promotion_id,
        campaign_id=scope_precommit_binding.campaign_id,
        selection_key=scope_precommit_binding.selection_key,
        selection_policy_sha256=scope_precommit_binding.selection_policy_sha256,
        selection_policy_artifact_sha256=(scope_precommit_binding.selection_policy_artifact_sha256),
        plan_selection_commitment_sha256=(scope_precommit_binding.plan_selection_commitment_sha256),
        selection_manifest_sha256=scope_precommit_binding.selection_manifest_sha256,
        selected_at_upper_bound=scope_precommit_binding.selected_at_upper_bound,
        checkpoint_sha256=checkpoint_sha256(witnessed.checkpoint),
        checkpoint=witnessed.checkpoint,
        ledger_prefix=_binding(_LEDGER_PREFIX, ledger_bytes),
        scope_policy=_binding(_SCOPE_POLICY, exact_scope_bytes),
        scope_policy_id=scope_policy.policy_id,
        scope_precommit=scope_precommit_binding,
        witness=witness_binding,
        registered_jobs=_binding(_REGISTERED_JOBS, jobs_bytes),
        captures=tuple(capture.binding for capture in captures),
        source_verifications=source_bindings,
        adapter=_adapter_binding(adapter, source_bindings),
        hermetic_executions=tuple(
            sorted(
                hermetic_bindings,
                key=lambda item: (item.purpose, item.subject_id, item.invocation_index),
            )
        ),
        normalization_dispositions=disposition_binding,
        normalized_outputs=normalized_bindings,
    )
    index_bytes = canonical_json_bytes(index)
    payloads[_CAPTURE_INDEX] = index_bytes
    _validate_payload_inventory(payloads)
    manifest = CapturePromotionManifest(
        promotion_id=promotion_id,
        campaign_id=index.campaign_id,
        selection_key=index.selection_key,
        selection_policy_sha256=index.selection_policy_sha256,
        selection_policy_artifact_sha256=index.selection_policy_artifact_sha256,
        plan_selection_commitment_sha256=index.plan_selection_commitment_sha256,
        selection_manifest_sha256=index.selection_manifest_sha256,
        selected_at_upper_bound=index.selected_at_upper_bound,
        created_at=created_at,
        capture_index=_binding(_CAPTURE_INDEX, index_bytes),
        scope_precommit=scope_precommit_binding.archive_manifest,
        files=tuple(_binding(path, payloads[path]) for path in sorted(payloads)),
    )
    target = _durable_publish(output_dir, payloads, canonical_json_bytes(manifest))
    return load_capture_promotion(
        target,
        expected_scope_policy=scope_policy,
        scope_precommit_witness_materials=scope_precommit_witness_materials,
        witness_materials=witness_materials,
        source_verifiers=source_verifiers,
        adapter=adapter,
        verified_at=verified_at,
        expected_scope_precommit_sha256=expected_scope_precommit_sha256,
        expected_promotion_sha256=capture_promotion_sha256(manifest),
        selection_materials=selection_materials,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
    )


def load_capture_promotion(
    root: Path,
    *,
    expected_scope_policy: PromotionScopePolicy,
    scope_precommit_witness_materials: WitnessMaterialSpec,
    witness_materials: WitnessMaterialSpec,
    source_verifiers: Mapping[str, SourceVerifierSpec],
    adapter: AdapterSpec,
    verified_at: datetime,
    expected_scope_precommit_sha256: str,
    expected_promotion_sha256: str,
    selection_materials: PlanSelectionMaterialSpecProtocol,
    expected_selection_manifest_sha256: str,
) -> LoadedCapturePromotion:
    """Load only after offline ledger, witness, source, and adapter re-verification."""

    verified_at = aware_utc(verified_at, 'verified_at')
    _validate_trusted_materials(witness_materials, source_verifiers, adapter)
    pre_capture_plan = derive_pre_capture_plan(
        scope_policy=expected_scope_policy,
        selection_policy=selection_materials.policy,
        capture_witness_policy=witness_materials.policy,
        source_verifiers=source_verifiers,
        adapter=adapter,
    )
    try:
        snapshot = snapshot_immutable_tree(
            root,
            max_files=_MAX_FILES + 1,
            max_directories=_MAX_DIRECTORIES,
            max_file_bytes=_MAX_FILE_BYTES,
            max_total_bytes=_MAX_TOTAL_BYTES,
            max_path_characters=_MAX_PATH_CHARACTERS,
            per_path_byte_limits={_PROMOTION_MANIFEST: _MAX_MANIFEST_BYTES},
            aggregate_exempt_paths=frozenset({_PROMOTION_MANIFEST}),
        )
    except ImmutableTreeError as error:
        raise PromotionIntegrityError(f'unsafe capture promotion tree: {error}') from error
    resolved = snapshot.root
    manifest_bytes = snapshot.files.get(_PROMOTION_MANIFEST)
    if manifest_bytes is None:
        raise PromotionIntegrityError('capture promotion omits promotion.json')
    try:
        manifest = CapturePromotionManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise PromotionIntegrityError(f'invalid capture promotion manifest: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise PromotionIntegrityError('capture promotion manifest is not canonical JSON')
    manifest_sha256 = capture_promotion_sha256(manifest)
    if manifest_sha256 != expected_promotion_sha256:
        raise PromotionIntegrityError('capture promotion does not match the expected digest')
    try:
        snapshot.require_exact_files({_PROMOTION_MANIFEST, *(binding.path for binding in manifest.files)})
    except ImmutableTreeError as error:
        raise PromotionIntegrityError(f'unsafe capture promotion tree: {error}') from error
    payloads = _load_bound_payloads(snapshot, manifest)
    index_bytes = payloads.get(_CAPTURE_INDEX)
    if index_bytes is None:
        raise PromotionIntegrityError('capture promotion omits capture-index.json')
    try:
        index = CaptureIndex.model_validate_json(index_bytes)
    except ValueError as error:
        raise PromotionIntegrityError(f'invalid capture index: {error}') from error
    if index_bytes != canonical_json_bytes(index) or _binding(_CAPTURE_INDEX, index_bytes) != manifest.capture_index:
        raise PromotionIntegrityError('capture index is noncanonical or differs from its manifest binding')
    if index.promotion_id != manifest.promotion_id:
        raise PromotionIntegrityError('capture index belongs to a different promotion')
    if _plan_selection_identity(index) != _plan_selection_identity(manifest):
        raise PromotionIntegrityError('promotion manifest plan selection differs from its capture index')
    if index.checkpoint.store_id != expected_scope_policy.store_id:
        raise PromotionIntegrityError('promoted checkpoint differs from the independently pinned scope policy store_id')
    if manifest.created_at > verified_at:
        raise PromotionIntegrityError('promotion verification time predates its creation')

    scope_precommit = load_scope_precommit(
        resolved / 'scope' / 'precommit',
        expected_archive_sha256=expected_scope_precommit_sha256,
        expected_scope_policy=expected_scope_policy,
        expected_pre_capture_plan=pre_capture_plan,
        witness_materials=scope_precommit_witness_materials,
        selection_materials=selection_materials,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
        verified_at=verified_at,
    )
    _require_exact_plan_selection(
        scope_precommit,
        scope_policy=expected_scope_policy,
        pre_capture_plan=pre_capture_plan,
        selection_materials=selection_materials,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
    )
    expected_precommit_binding = _scope_precommit_binding(scope_precommit)
    if (
        manifest.scope_precommit != expected_precommit_binding.archive_manifest
        or index.scope_precommit != expected_precommit_binding
    ):
        raise PromotionIntegrityError('copied scope precommit differs from promotion/index bindings')
    if payloads.get(expected_precommit_binding.archive_manifest.path) != scope_precommit.manifest_bytes:
        raise PromotionIntegrityError('copied scope precommit manifest differs from the reverified archive')
    for relative, payload in scope_precommit.file_payloads:
        if payloads.get(f'scope/precommit/{relative}') != payload:
            raise PromotionIntegrityError('copied scope precommit payload differs from the reverified archive')

    scope_bytes = payloads.get(index.scope_policy.path)
    if (
        scope_bytes is None
        or scope_bytes != canonical_json_bytes(expected_scope_policy)
        or index.scope_policy_id != expected_scope_policy.policy_id
    ):
        raise PromotionIntegrityError('promotion scope policy differs from the independently supplied policy')
    if _binding(index.scope_policy.path, scope_bytes) != index.scope_policy:
        raise PromotionIntegrityError('promotion scope policy bytes differ from their capture-index binding')
    if not (
        expected_scope_policy.checkpoint_created_at_not_before
        <= index.checkpoint.created_at
        <= expected_scope_policy.checkpoint_created_at_not_after
    ):
        raise PromotionIntegrityError('promoted checkpoint lies outside the independently pinned scope window')
    if expected_scope_policy.store_id != index.checkpoint.store_id:
        raise PromotionIntegrityError('promoted checkpoint does not match the independently pinned store ID')
    ledger_payload = payloads.get(index.ledger_prefix.path)
    if ledger_payload is None or _binding(index.ledger_prefix.path, ledger_payload) != index.ledger_prefix:
        raise PromotionIntegrityError('portable ledger bytes differ from their capture-index binding')
    events = parse_ledger_prefix(ledger_payload, index.checkpoint)
    _require_scope_precommit_prefix(scope_precommit, events, index.checkpoint.through_sequence)
    jobs_payload = payloads.get(index.registered_jobs.path)
    if jobs_payload is None or _binding(index.registered_jobs.path, jobs_payload) != index.registered_jobs:
        raise PromotionIntegrityError('portable job inventory differs from its capture-index binding')
    jobs = _parse_registered_jobs(jobs_payload, events, expected_scope_policy)

    _require_exact_material(payloads, 'witness/materials/policy.bin', witness_materials.policy_bytes)
    _require_exact_material(payloads, 'witness/materials/trust-policy.bin', witness_materials.trust_policy_bytes)
    _require_exact_material(
        payloads,
        'witness/materials/verifier-implementation.bin',
        witness_materials.verifier_implementation_bytes,
    )
    witnessed = _verify_snapshotted_witness(
        resolved,
        payloads,
        materials=witness_materials,
        expected_checkpoint_sha256=index.checkpoint_sha256,
        verified_at=verified_at,
    )
    if (
        witnessed.checkpoint != index.checkpoint
        or _witness_binding(payloads, witnessed, witness_materials) != index.witness
    ):
        raise PromotionIntegrityError('copied external witness does not exactly reproduce the capture-index witness')
    if manifest.created_at < witnessed.witnessed_at or manifest.created_at > verified_at:
        raise PromotionIntegrityError('promotion creation time is outside its verified witness/load interval')

    captures = _load_exact_captures(index, payloads, events, jobs, expected_scope_policy)
    replay_payloads: dict[str, bytes] = {}
    source_bindings, adapter_inputs = _run_source_verifiers(
        captures,
        source_verifiers,
        replay_payloads,
    )
    if source_bindings != index.source_verifications:
        raise PromotionIntegrityError('portable source verification does not reproduce the capture index')
    for binding in source_bindings:
        _require_exact_material(
            payloads,
            binding.verifier_policy.path,
            replay_payloads[binding.verifier_policy.path],
        )
        _require_exact_material(
            payloads,
            binding.verifier_implementation.path,
            replay_payloads[binding.verifier_implementation.path],
        )
        _require_exact_material(
            payloads,
            binding.verifier_execution_environment.path,
            replay_payloads[binding.verifier_execution_environment.path],
        )
        _require_exact_material(
            payloads,
            binding.verified_records.path,
            replay_payloads[binding.verified_records.path],
        )
    if _adapter_binding(adapter, source_bindings) != index.adapter:
        raise PromotionIntegrityError('portable adapter identity or input binding differs from pinned code')
    _require_exact_material(payloads, index.adapter.implementation.path, adapter.implementation_bytes)
    _require_exact_material(payloads, index.adapter.policy.path, adapter.policy_bytes)
    _require_exact_material(
        payloads,
        index.adapter.execution_environment.path,
        adapter.execution_environment_bytes,
    )
    outputs, disposition_binding, dispositions = _run_adapter_twice(
        adapter_inputs,
        adapter,
        replay_payloads,
    )
    if outputs != index.normalized_outputs or disposition_binding != index.normalization_dispositions:
        raise PromotionIntegrityError(
            'portable adapter replay does not reproduce normalized outputs and exhaustive dispositions'
        )
    _require_exact_material(
        payloads,
        index.normalization_dispositions.path,
        replay_payloads[index.normalization_dispositions.path],
    )
    for output in outputs:
        if replay_payloads[output.file.path] != payloads.get(output.file.path):
            raise PromotionIntegrityError(f'normalized output does not match deterministic replay: {output.file.path}')
    _verify_archived_hermetic_executions(
        index=index,
        payloads=payloads,
        captures=captures,
        adapter_inputs=adapter_inputs,
        source_verifiers=source_verifiers,
        adapter=adapter,
    )

    candidate_binding = _one_output(index, NormalizedOutputRole.CANDIDATE_RECORDS)
    evidence_binding = _one_output(index, NormalizedOutputRole.EVIDENCE_RECORDS)
    candidates = _parse_record_jsonl(payloads[candidate_binding.file.path], CandidateRecord, 'candidate')
    evidence = _parse_record_jsonl(payloads[evidence_binding.file.path], EvidenceRecord, 'evidence')
    auxiliary_outputs = MappingProxyType(
        {
            PurePosixPath(output.file.path).name.removesuffix('.bin'): payloads[output.file.path]
            for output in index.normalized_outputs
            if output.role is NormalizedOutputRole.AUXILIARY
        }
    )
    if manifest.created_at < max(capture.captured_at for capture in index.captures):
        raise PromotionIntegrityError('promotion creation time predates a selected capture completion')
    handoff_descriptor = _build_handoff_descriptor(manifest, index)
    handoff_descriptor_bytes = canonical_json_bytes(handoff_descriptor)
    source_captures = (
        SourceCaptureArtifact(
            source_id=f'promotion:{index.promotion_id}',
            source_release_at=handoff_descriptor.maximum_source_release_at,
            captured_at=handoff_descriptor.maximum_captured_at,
            witnessed_at=handoff_descriptor.witnessed_at,
            manifest_bytes=handoff_descriptor_bytes,
        ),
    )
    return LoadedCapturePromotion(
        root=resolved,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        index=index,
        index_bytes=index_bytes,
        handoff_descriptor=handoff_descriptor,
        handoff_descriptor_bytes=handoff_descriptor_bytes,
        ledger_events=events,
        candidates=candidates,
        evidence=evidence,
        dispositions=dispositions,
        auxiliary_outputs=auxiliary_outputs,
        source_captures=source_captures,
    )


def build_prospective_decision_package_from_promotion(
    output_dir: Path,
    *,
    promotion_root: Path,
    expected_scope_policy: PromotionScopePolicy,
    scope_precommit_witness_materials: WitnessMaterialSpec,
    witness_materials: WitnessMaterialSpec,
    source_verifiers: Mapping[str, SourceVerifierSpec],
    adapter: AdapterSpec,
    verified_at: datetime,
    config: DecisionTimeConfig,
    protocol_artifacts: Mapping[str, bytes],
    expected_scope_precommit_sha256: str,
    expected_promotion_sha256: str,
    selection_materials: PlanSelectionMaterialSpecProtocol,
    expected_selection_manifest_sha256: str,
) -> LoadedProspectiveDecisionPackage:
    """Lineage-safe package builder with no caller-supplied candidates/evidence."""

    loaded = load_capture_promotion(
        promotion_root,
        expected_scope_policy=expected_scope_policy,
        scope_precommit_witness_materials=scope_precommit_witness_materials,
        witness_materials=witness_materials,
        source_verifiers=source_verifiers,
        adapter=adapter,
        verified_at=verified_at,
        expected_scope_precommit_sha256=expected_scope_precommit_sha256,
        expected_promotion_sha256=expected_promotion_sha256,
        selection_materials=selection_materials,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
    )
    return build_prospective_decision_package(
        output_dir,
        config=config,
        candidates=loaded.candidates,
        evidence=loaded.evidence,
        protocol_artifacts=protocol_artifacts,
        candidate_set_available_at=loaded.manifest.created_at,
        source_captures=loaded.source_captures,
    )


def make_promotion_source_capture_verifier(
    *,
    promotion_root: Path,
    expected_scope_policy: PromotionScopePolicy,
    scope_precommit_witness_materials: WitnessMaterialSpec,
    witness_materials: WitnessMaterialSpec,
    source_verifiers: Mapping[str, SourceVerifierSpec],
    adapter: AdapterSpec,
    verified_at: datetime,
    expected_scope_precommit_sha256: str,
    expected_promotion_sha256: str,
    selection_materials: PlanSelectionMaterialSpecProtocol,
    expected_selection_manifest_sha256: str,
    expected_source_capture_policy: bytes,
) -> Callable[[ProspectiveSourceCaptureBinding, bytes, bytes], bool]:
    """Create the trusted admission callback for one promotion-backed source.

    The small handoff descriptor in a decision package is structurally useful but
    can be fabricated self-consistently.  This callback resolves the separately
    retained, content-addressed promotion root and reruns ``load_capture_promotion``
    on every invocation.  Only an exact descriptor, prospective binding, and
    independently supplied source-capture policy are accepted.
    """

    if not expected_source_capture_policy:
        raise PromotionIntegrityError('promotion source-capture admission policy cannot be empty')
    try:
        resolved_root, _root_identity = immutable_root_identity(promotion_root)
    except ImmutableTreeError as error:
        raise PromotionIntegrityError(f'unsafe capture promotion root: {error}') from error

    def verify(
        binding: ProspectiveSourceCaptureBinding,
        descriptor_bytes: bytes,
        source_capture_policy: bytes,
    ) -> bool:
        if source_capture_policy != expected_source_capture_policy:
            return False
        loaded = load_capture_promotion(
            resolved_root,
            expected_scope_policy=expected_scope_policy,
            scope_precommit_witness_materials=scope_precommit_witness_materials,
            witness_materials=witness_materials,
            source_verifiers=source_verifiers,
            adapter=adapter,
            verified_at=verified_at,
            expected_scope_precommit_sha256=expected_scope_precommit_sha256,
            expected_promotion_sha256=expected_promotion_sha256,
            selection_materials=selection_materials,
            expected_selection_manifest_sha256=expected_selection_manifest_sha256,
        )
        try:
            supplied_descriptor = PromotionHandoffDescriptor.model_validate_json(descriptor_bytes)
        except ValueError:
            return False
        if descriptor_bytes != canonical_json_bytes(supplied_descriptor):
            return False
        expected_source_id = f'promotion:{loaded.handoff_descriptor.promotion_id}'
        if (
            supplied_descriptor != loaded.handoff_descriptor
            or descriptor_bytes != loaded.handoff_descriptor_bytes
            or binding.source_id != expected_source_id
            or binding.source_release_at != loaded.handoff_descriptor.maximum_source_release_at
            or binding.captured_at != loaded.handoff_descriptor.maximum_captured_at
            or binding.witnessed_at != loaded.handoff_descriptor.witnessed_at
            or binding.file.path != 'source-captures/000000.json'
            or binding.file.sha256 != hashlib.sha256(descriptor_bytes).hexdigest()
            or binding.file.byte_count != len(descriptor_bytes)
        ):
            return False
        return True

    return verify


def _build_handoff_descriptor(
    manifest: CapturePromotionManifest,
    index: CaptureIndex,
) -> PromotionHandoffDescriptor:
    return PromotionHandoffDescriptor(
        promotion_id=index.promotion_id,
        campaign_id=index.campaign_id,
        selection_key=index.selection_key,
        selection_policy_sha256=index.selection_policy_sha256,
        selection_policy_artifact_sha256=index.selection_policy_artifact_sha256,
        plan_selection_commitment_sha256=index.plan_selection_commitment_sha256,
        selection_manifest_sha256=index.selection_manifest_sha256,
        selected_at_upper_bound=index.selected_at_upper_bound,
        promotion_manifest_sha256=capture_promotion_sha256(manifest),
        promotion_created_at=manifest.created_at,
        capture_index_sha256=capture_index_sha256(index),
        capture_index=index,
        candidate_output=_one_output(index, NormalizedOutputRole.CANDIDATE_RECORDS),
        evidence_output=_one_output(index, NormalizedOutputRole.EVIDENCE_RECORDS),
        maximum_source_release_at=max(
            binding.result.source_release.source_release_at for binding in index.source_verifications
        ),
        maximum_captured_at=max(capture.captured_at for capture in index.captures),
        witnessed_at=index.witness.witnessed_at,
    )


def _scope_precommit_binding(precommit: LoadedScopePrecommit) -> ScopePrecommitPromotionBinding:
    manifest_path = 'scope/precommit/scope-precommit.json'
    selection_policy = precommit.plan_selection_commitment.policy
    return ScopePrecommitPromotionBinding(
        archive_manifest=_binding(manifest_path, precommit.manifest_bytes),
        archive_sha256=precommit.archive_sha256,
        scope_policy_sha256=precommit.manifest.scope_policy.sha256,
        pre_capture_plan_sha256=precommit.manifest.pre_capture_plan.sha256,
        campaign_id=selection_policy.campaign_id,
        selection_key=selection_policy.selection_key,
        selection_policy_sha256=hashlib.sha256(canonical_json_bytes(selection_policy)).hexdigest(),
        selection_policy_artifact_sha256=selection_policy.policy_sha256,
        plan_selection_commitment_sha256=plan_selection_commitment_sha256(precommit.plan_selection_commitment),
        selection_manifest_sha256=hashlib.sha256(canonical_json_bytes(precommit.plan_selection_manifest)).hexdigest(),
        selected_at_upper_bound=(precommit.plan_selection_manifest.receipt.facts.selected_at_upper_bound),
        store_id=precommit.checkpoint.store_id,
        checkpoint_sha256=checkpoint_sha256(precommit.checkpoint),
        checkpoint_through_sequence=precommit.checkpoint.through_sequence,
        checkpoint_through_event_sha256=precommit.checkpoint.through_event_sha256,
        witnessed_at=precommit.manifest.witness.witnessed_at,
    )


def _require_exact_plan_selection(
    precommit: LoadedScopePrecommit,
    *,
    scope_policy: PromotionScopePolicy,
    pre_capture_plan: PreCapturePlan,
    selection_materials: PlanSelectionMaterialSpecProtocol,
    expected_selection_manifest_sha256: str,
) -> None:
    """Independently reconstruct and exact-match the selected plan commitment."""

    commitment = derive_plan_selection_commitment(
        scope_policy,
        pre_capture_plan,
        precommit.checkpoint,
    )
    if commitment.policy != selection_materials.policy:
        raise PromotionIntegrityError('scope precommit plan selection uses a different independently supplied policy')
    actual_manifest_sha256 = hashlib.sha256(canonical_json_bytes(precommit.plan_selection_manifest)).hexdigest()
    if precommit.plan_selection_commitment != commitment:
        raise PromotionIntegrityError(
            'scope precommit plan selection differs from the independently reconstructed commitment'
        )
    if actual_manifest_sha256 != expected_selection_manifest_sha256:
        raise PromotionIntegrityError('scope precommit plan selection differs from the expected manifest digest')


def _plan_selection_identity(value: object) -> tuple[object, ...]:
    return (
        getattr(value, 'campaign_id'),
        getattr(value, 'selection_key'),
        getattr(value, 'selection_policy_sha256'),
        getattr(value, 'selection_policy_artifact_sha256'),
        getattr(value, 'plan_selection_commitment_sha256'),
        getattr(value, 'selection_manifest_sha256'),
        getattr(value, 'selected_at_upper_bound'),
    )


def _require_scope_precommit_prefix(
    precommit: LoadedScopePrecommit,
    capture_events: tuple[LedgerEvent, ...],
    capture_through_sequence: int,
) -> None:
    through_sequence = precommit.checkpoint.through_sequence
    if through_sequence >= capture_through_sequence or capture_through_sequence != len(capture_events):
        raise PromotionIntegrityError('scope precommit checkpoint must be a strict capture-ledger prefix')
    if len(precommit.ledger_events) != through_sequence:
        raise PromotionIntegrityError('scope precommit ledger does not exactly reach its checkpoint')
    if capture_events[through_sequence - 1].event_sha256 != precommit.checkpoint.through_event_sha256:
        raise PromotionIntegrityError('capture ledger does not contain the scope precommit checkpoint head')
    if tuple(capture_events[:through_sequence]) != precommit.ledger_events:
        raise PromotionIntegrityError('scope precommit ledger is not the exact capture-ledger prefix')


def _parse_registered_jobs(
    payload: bytes,
    events: tuple[LedgerEvent, ...],
    scope: PromotionScopePolicy,
) -> dict[str, RegisteredJob]:
    records = _parse_model_jsonl(payload, RegisteredJob, 'registered job')
    jobs = {record.spec_sha256: record for record in records}
    if len(jobs) != len(records):
        raise PromotionIntegrityError('portable registered-job inventory contains duplicate revisions')
    expected = {digest for source in scope.sources for digest in source.job_spec_sha256s}
    if set(jobs) != expected:
        raise PromotionIntegrityError('portable registered-job inventory does not exactly match pinned scope')
    event_maps = _event_maps(events)
    for source_scope in scope.sources:
        for digest in source_scope.job_spec_sha256s:
            job = jobs[digest]
            event = _require_event(event_maps['jobs'], digest, 'portable job registration')
            if event.occurred_at != job.registered_at or event.payload.get('job_id') != job.spec.job_id:
                raise PromotionIntegrityError('portable job does not reproduce its witnessed registration')
            try:
                configuration = parse_supported_collector_job_configuration(
                    job.spec.collector_id,
                    job.spec.configuration,
                )
            except ValueError as error:
                raise PromotionIntegrityError(
                    'portable job uses an unsupported or invalid collector configuration'
                ) from error
            _require_v0_single_attempt_job(configuration.max_attempts_per_slot)
            if configuration.source_id != source_scope.source_id:
                raise PromotionIntegrityError('portable job source differs from independently pinned scope')
    return jobs


def _parse_model_jsonl(payload: bytes, model, label: str):
    if not payload or not payload.endswith(b'\n'):
        raise PromotionIntegrityError(f'{label} JSONL must be nonempty and end in a newline')
    lines = payload.split(b'\n')
    if lines[-1] != b'' or any(not line for line in lines[:-1]):
        raise PromotionIntegrityError(f'{label} JSONL contains an empty or invalid LF-delimited record')
    records = []
    for ordinal, line in enumerate(lines[:-1], start=1):
        try:
            record = model.model_validate_json(line)
        except ValueError as error:
            raise PromotionIntegrityError(f'invalid {label} JSONL record {ordinal}') from error
        if line != canonical_json_bytes(record):
            raise PromotionIntegrityError(f'{label} JSONL record {ordinal} is not canonical')
        records.append(record)
    if b''.join(canonical_json_bytes(record) + b'\n' for record in records) != payload:
        raise PromotionIntegrityError(f'{label} JSONL does not use the exact canonical LF encoding')
    return tuple(records)


def _load_exact_captures(
    index: CaptureIndex,
    payloads: Mapping[str, bytes],
    events: tuple[LedgerEvent, ...],
    jobs: Mapping[str, RegisteredJob],
    scope: PromotionScopePolicy,
) -> tuple[ExactPromotedCapture, ...]:
    by_run = {capture.logical_run_id: capture for capture in index.captures}
    if len(by_run) != len(index.captures):
        raise PromotionIntegrityError('portable captures contain a duplicate logical_run_id')
    expected_run_ids: set[str] = set()
    for source_scope in scope.sources:
        for digest in source_scope.job_spec_sha256s:
            job = jobs[digest]
            for slot in _expected_slots(job.spec, source_scope.scheduled_from, source_scope.scheduled_through):
                expected_run_ids.add(scheduled_logical_run_id(digest, slot))
    if set(by_run) != expected_run_ids:
        raise PromotionIntegrityError('portable captures do not exactly cover every pinned schedule slot')

    event_maps = _event_maps(events)
    if set(event_maps['success_runs']).intersection(expected_run_ids) != expected_run_ids:
        raise PromotionIntegrityError('witnessed prefix lacks exactly one success for every pinned schedule slot')
    captures: list[ExactPromotedCapture] = []
    for binding in index.captures:
        if binding.logical_run_id != scheduled_logical_run_id(binding.job_spec_sha256, binding.scheduled_for):
            raise PromotionIntegrityError('portable capture logical_run_id does not bind its declared schedule slot')
        job = jobs.get(binding.job_spec_sha256)
        if job is None or job.spec != binding.job_spec or job.spec_sha256 != binding.job_spec_sha256:
            raise PromotionIntegrityError('portable capture does not bind a scoped registered job')
        job_event = _require_event(event_maps['jobs'], binding.job_spec_sha256, 'job registration')
        run_event = _require_event(event_maps['runs'], binding.logical_run_id, 'logical-run registration')
        start_event = _require_event(event_maps['starts'], binding.attempt_id, 'attempt start')
        success_event = _require_event(event_maps['successes'], binding.attempt_id, 'attempt success')
        if (
            (binding.job_registered_event_sequence, binding.job_registered_event_sha256)
            != (job_event.sequence, job_event.event_sha256)
            or (binding.run_registered_event_sequence, binding.run_registered_event_sha256)
            != (run_event.sequence, run_event.event_sha256)
            or (binding.started_event_sequence, binding.started_event_sha256)
            != (start_event.sequence, start_event.event_sha256)
            or (binding.succeeded_event_sequence, binding.succeeded_event_sha256)
            != (success_event.sequence, success_event.event_sha256)
            or job_event.payload.get('job_id') != binding.job_id
            or run_event.payload.get('logical_run_id') != binding.logical_run_id
            or run_event.payload.get('job_spec_sha256') != binding.job_spec_sha256
            or _event_timestamp(run_event.payload.get('scheduled_for')) != binding.scheduled_for
            or start_event.payload.get('attempt_id') != binding.attempt_id
            or start_event.payload.get('logical_run_id') != binding.logical_run_id
            or success_event.payload.get('attempt_id') != binding.attempt_id
            or success_event.payload.get('logical_run_id') != binding.logical_run_id
            or start_event.occurred_at != binding.attempt_started_at
            or success_event.occurred_at != binding.captured_at
        ):
            raise PromotionIntegrityError('portable capture lifecycle fields do not match the witnessed event prefix')
        try:
            configuration = parse_supported_collector_job_configuration(
                job.spec.collector_id,
                job.spec.configuration,
            )
        except ValueError as error:
            raise PromotionIntegrityError(
                'portable capture uses an unsupported or invalid collector configuration'
            ) from error
        if configuration.source_id != binding.source_id:
            raise PromotionIntegrityError('portable capture source does not match its immutable job')
        _verify_static_attempt_history(events, binding, configuration)
        attached_events = {
            key[1]: event
            for key, event in event_maps['attachments'].items()
            if isinstance(key, tuple) and key[0] == binding.attempt_id
        }
        if set(attached_events) != {artifact.role for artifact in binding.artifacts}:
            raise PromotionIntegrityError('portable capture does not bind its exact witnessed attachment set')
        exact_artifacts: list[ExactPromotedArtifact] = []
        for artifact in binding.artifacts:
            stored_event = _require_event(event_maps['artifacts'], artifact.file.sha256, 'artifact first record')
            attached_event = _require_event(
                event_maps['attachments'],
                (binding.attempt_id, artifact.role),
                'artifact attachment',
            )
            if (
                (artifact.stored_event_sequence, artifact.stored_event_sha256)
                != (stored_event.sequence, stored_event.event_sha256)
                or (artifact.attached_event_sequence, artifact.attached_event_sha256)
                != (attached_event.sequence, attached_event.event_sha256)
                or stored_event.occurred_at != artifact.first_recorded_at
                or stored_event.payload.get('byte_count') != artifact.file.byte_count
                or attached_event.payload.get('artifact_sha256') != artifact.file.sha256
            ):
                raise PromotionIntegrityError('raw artifact does not reproduce its witnessed store/attachment events')
            raw = payloads.get(artifact.file.path)
            if raw is None or _binding(artifact.file.path, raw) != artifact.file:
                raise PromotionIntegrityError(f'portable raw artifact differs from its binding: {artifact.file.path}')
            exact_artifacts.append(ExactPromotedArtifact(binding=artifact, payload=raw))
        capture = ExactPromotedCapture(binding=binding, artifacts=tuple(exact_artifacts))
        _verify_portable_supported_capture(capture, configuration)
        captures.append(capture)
    return tuple(captures)


def _event_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise PromotionIntegrityError('witnessed ledger event timestamp payload is malformed')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise PromotionIntegrityError('witnessed ledger event timestamp payload is invalid') from error
    return aware_utc(parsed, 'witnessed ledger event timestamp')


def _verify_static_attempt_history(events, binding, configuration) -> None:
    _require_v0_single_attempt_job(configuration.max_attempts_per_slot)
    starts = tuple(
        event
        for event in events
        if event.event_type is LedgerEventType.ATTEMPT_STARTED
        and event.payload.get('logical_run_id') == binding.logical_run_id
    )
    if len(starts) != 1:
        raise PromotionIntegrityError('V0 promotion requires exactly one attempt for every scoped static run')
    start = starts[0]
    if set(start.payload) != {
        'attempt_id',
        'attempt_number',
        'lease_expires_at',
        'logical_run_id',
        'owner_id',
    }:
        raise PromotionIntegrityError('portable static attempt-start payload is not the exact V0 store payload')
    attempt_id = start.payload.get('attempt_id')
    owner_id = start.payload.get('owner_id')
    attempt_number = start.payload.get('attempt_number')
    expiry_value = start.payload.get('lease_expires_at')
    expiry = _event_timestamp(expiry_value)
    if (
        not isinstance(attempt_id, str)
        or not isinstance(owner_id, str)
        or not isinstance(attempt_number, int)
        or isinstance(attempt_number, bool)
        or attempt_number != 1
        or attempt_id != binding.attempt_id
        or start.occurred_at != binding.attempt_started_at
        or start.occurred_at < binding.scheduled_for
        or expiry_value != _ledger_timestamp(expiry)
        or expiry - start.occurred_at != timedelta(seconds=configuration.lease_seconds)
    ):
        raise PromotionIntegrityError('portable static attempt start disagrees with immutable V0 lease policy')
    try:
        AttemptLease(
            attempt_id=attempt_id,
            logical_run_id=binding.logical_run_id,
            attempt_number=attempt_number,
            owner_id=owner_id,
            state=AttemptState.STARTED,
            started_at=start.occurred_at,
            lease_expires_at=expiry,
        )
    except ValueError as error:
        raise PromotionIntegrityError('portable static attempt-start identity is invalid') from error
    attempt_preimage = {
        'attempt_number': attempt_number,
        'logical_run_id': binding.logical_run_id,
        'owner_id': owner_id,
        'started_at': _ledger_timestamp(start.occurred_at),
    }
    expected_attempt_id = f'attempt-{hashlib.sha256(canonical_json_bytes(attempt_preimage)).hexdigest()[:32]}'
    if attempt_id != expected_attempt_id:
        raise PromotionIntegrityError('portable static attempt ID does not reproduce the V0 store preimage')

    renewals = tuple(
        event
        for event in events
        if event.event_type is LedgerEventType.ATTEMPT_LEASE_RENEWED and event.payload.get('attempt_id') == attempt_id
    )
    terminals = tuple(
        event
        for event in events
        if event.event_type
        in {
            LedgerEventType.ATTEMPT_FAILED,
            LedgerEventType.ATTEMPT_ABANDONED,
            LedgerEventType.ATTEMPT_SUCCEEDED,
        }
        and (
            event.payload.get('attempt_id') == attempt_id
            or event.payload.get('logical_run_id') == binding.logical_run_id
        )
    )
    if renewals or len(terminals) != 1:
        raise PromotionIntegrityError('portable static attempt lacks its exact single-attempt terminal lifecycle')
    terminal = terminals[0]
    if (
        terminal.event_type is not LedgerEventType.ATTEMPT_SUCCEEDED
        or set(terminal.payload) != {'attempt_id', 'logical_run_id', 'terminal_code'}
        or terminal.payload.get('attempt_id') != attempt_id
        or terminal.payload.get('logical_run_id') != binding.logical_run_id
        or terminal.payload.get('terminal_code') != 'success'
        or terminal.sequence != binding.succeeded_event_sequence
        or terminal.occurred_at != binding.captured_at
        or not (start.sequence < terminal.sequence)
        or not (start.occurred_at <= terminal.occurred_at < expiry)
    ):
        raise PromotionIntegrityError('portable static success is not the exact V0 terminal event')

    expected_attachments = {artifact.role: artifact.file.sha256 for artifact in binding.artifacts}
    attachments = tuple(
        event
        for event in events
        if event.event_type is LedgerEventType.ATTEMPT_ARTIFACT_ATTACHED
        and event.payload.get('attempt_id') == attempt_id
    )
    if len(attachments) != len(expected_attachments):
        raise PromotionIntegrityError('portable static attempt does not contain its exact attachment lifecycle')
    seen_roles: set[str] = set()
    for attachment in attachments:
        role = attachment.payload.get('role')
        if (
            set(attachment.payload) != {'artifact_sha256', 'attempt_id', 'role'}
            or not isinstance(role, str)
            or role in seen_roles
            or attachment.payload.get('attempt_id') != attempt_id
            or attachment.payload.get('artifact_sha256') != expected_attachments.get(role)
            or not (start.sequence < attachment.sequence < terminal.sequence)
            or not (start.occurred_at <= attachment.occurred_at < expiry)
            or attachment.occurred_at > terminal.occurred_at
        ):
            raise PromotionIntegrityError('portable static attachment disagrees with its exact attempt lifecycle')
        seen_roles.add(role)
    if seen_roles != set(expected_attachments):
        raise PromotionIntegrityError('portable static attachment roles differ from the promoted raw inventory')
    if expected_attachments.get('collection-plan') != configuration.collection_plan_sha256:
        raise PromotionIntegrityError('portable static attempt does not bind its immutable collection plan')


def _ledger_timestamp(value: datetime) -> str:
    return aware_utc(value, 'ledger timestamp').isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _require_v0_single_attempt_job(max_attempts_per_slot: int) -> None:
    if max_attempts_per_slot != 1:
        raise PromotionIntegrityError(
            'V0 promotion requires max_attempts_per_slot == 1; retry-capable promotion requires '
            'complete all-attempt artifact semantics'
        )


def _verify_portable_supported_capture(
    capture: ExactPromotedCapture,
    configuration: SupportedCollectorJobConfiguration,
) -> None:
    collector_id = capture.binding.collector_id
    if collector_id == STATIC_HTTPS_COLLECTOR_ID:
        if not isinstance(configuration, StaticHttpsJobConfiguration):
            raise PromotionIntegrityError('portable static capture has the wrong typed configuration')
        _verify_portable_static_capture(capture, configuration)
        return
    if collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID:
        if not isinstance(configuration, ImmportAuthenticatedJobConfiguration):
            raise PromotionIntegrityError('portable ImmPort capture has the wrong typed configuration')
        _verify_portable_immport_capture(capture, configuration)
        return
    raise PromotionIntegrityError(f'no portable semantic verifier is registered for collector {collector_id!r}')


def _verify_portable_immport_capture(
    capture: ExactPromotedCapture,
    configuration: ImmportAuthenticatedJobConfiguration,
) -> None:
    """Replay the credential-free ImmPort manifest from only promoted bytes."""

    from vaxreplay.sources.immport import ImmportSanitizedCaptureReceipt

    binding = capture.binding
    role_map = {artifact.role: artifact for artifact in capture.artifacts}
    plan_artifact = role_map.get('collection-plan')
    manifest_artifact = role_map.get('run-manifest')
    if plan_artifact is None or manifest_artifact is None:
        raise PromotionIntegrityError('portable ImmPort capture omits its plan or run manifest')
    if (
        plan_artifact.byte_count > MAX_IMMPORT_PLAN_BYTES
        or manifest_artifact.byte_count > MAX_IMMPORT_RUN_MANIFEST_BYTES
    ):
        raise PromotionIntegrityError('portable ImmPort plan or manifest exceeds its byte bound')
    try:
        plan = ImmportAuthenticatedCollectionPlan.model_validate_json(plan_artifact.payload)
        manifest = ImmportAuthenticatedRunManifest.model_validate_json(manifest_artifact.payload)
    except ValueError as error:
        raise PromotionIntegrityError('portable ImmPort plan or run manifest is invalid') from error
    if (
        canonical_json_bytes(plan) != plan_artifact.payload
        or canonical_json_bytes(manifest) != manifest_artifact.payload
    ):
        raise PromotionIntegrityError('portable ImmPort plan and manifest must be canonical JSON')
    plan_sha256 = immport_authenticated_plan_sha256(plan)
    if (
        binding.collector_id != IMMPORT_AUTHENTICATED_COLLECTOR_ID
        or configuration.collection_plan_sha256 != plan_sha256
        or configuration.source_id != binding.source_id
        or configuration.lease_seconds < plan.panel_deadline_seconds
        or plan.source_id != binding.source_id
        or manifest.plan_sha256 != plan_sha256
        or manifest.plan_id != plan.plan_id
        or manifest.source_id != binding.source_id
        or manifest.attempt_id != binding.attempt_id
        or manifest.logical_run_id != binding.logical_run_id
        or manifest.job_spec_sha256 != binding.job_spec_sha256
        or manifest.scheduled_for != binding.scheduled_for
        or manifest.attempt_started_at != binding.attempt_started_at
        or manifest.completed_at > binding.captured_at
        or manifest.collector_id != binding.collector_id
        or manifest.collector_implementation_sha256 != configuration.collector_implementation_sha256
        or manifest.collector_execution_environment_sha256 != configuration.collector_execution_environment_sha256
    ):
        raise PromotionIntegrityError(
            'portable ImmPort manifest does not bind its lifecycle, plan, and reviewed collector'
        )
    expected_roles = {'collection-plan', 'run-manifest'}
    expected_roles.update(f'body.{item.artifact_id}' for item in plan.artifacts)
    expected_roles.update(f'receipt.{item.artifact_id}' for item in plan.artifacts)
    if set(role_map) != expected_roles:
        raise PromotionIntegrityError('portable ImmPort attachment set differs from its exact plan')
    if tuple(item.artifact_id for item in manifest.artifacts) != tuple(item.artifact_id for item in plan.artifacts):
        raise PromotionIntegrityError('portable ImmPort manifest artifact order differs from its plan')

    receipts = []
    total_body_bytes = 0
    for spec, item in zip(plan.artifacts, manifest.artifacts, strict=True):
        body = role_map[f'body.{spec.artifact_id}']
        receipt_artifact = role_map[f'receipt.{spec.artifact_id}']
        if receipt_artifact.byte_count > MAX_IMMPORT_RECEIPT_BYTES:
            raise PromotionIntegrityError('portable sanitized ImmPort receipt exceeds its byte bound')
        receipt = None
        try:
            receipt = ImmportSanitizedCaptureReceipt.model_validate_json(receipt_artifact.payload)
        except ValueError:
            pass
        if receipt is None:
            raise PromotionIntegrityError('portable sanitized ImmPort receipt is invalid')
        if canonical_json_bytes(receipt) != receipt_artifact.payload:
            raise PromotionIntegrityError('portable sanitized ImmPort receipt is not canonical JSON')
        total_body_bytes += body.byte_count
        if (
            (body.sha256, body.byte_count) != (item.body_sha256, item.body_byte_count)
            or (receipt_artifact.sha256, receipt_artifact.byte_count) != (item.receipt_sha256, item.receipt_byte_count)
            or body.byte_count > spec.max_body_bytes
            or receipt.requested_url != spec.requested_url
            or receipt.authentication != spec.authentication
            or receipt.body_sha256 != body.sha256
            or receipt.body_byte_count != body.byte_count
            or receipt.started_at != item.started_at
            or receipt.completed_at != item.completed_at
            or receipt.started_at < binding.attempt_started_at
            or receipt.completed_at > binding.captured_at
            or (receipt.completed_at - receipt.started_at).total_seconds() > spec.timeout_seconds
            or receipt.collector_id != binding.collector_id
            or receipt.collector_implementation_sha256 != configuration.collector_implementation_sha256
            or receipt.collector_execution_environment_sha256 != configuration.collector_execution_environment_sha256
        ):
            raise PromotionIntegrityError('portable ImmPort body/receipt replay failed')
        receipts.append(receipt)
    if total_body_bytes > MAX_IMMPORT_CAPTURE_BODY_BYTES:
        raise PromotionIntegrityError('portable ImmPort bodies exceed the aggregate promotion budget')
    if any(left.completed_at > right.started_at for left, right in zip(receipts[:-1], receipts[1:], strict=True)):
        raise PromotionIntegrityError('portable ImmPort receipts are not a serial release bracket')
    if (receipts[-1].completed_at - receipts[0].started_at).total_seconds() > (plan.panel_deadline_seconds):
        raise PromotionIntegrityError('portable ImmPort receipts exceed the panel deadline')


def _verify_portable_static_capture(
    capture: ExactPromotedCapture,
    configuration: StaticHttpsJobConfiguration,
) -> None:
    from vaxreplay.operations.collector import static_plan_sha256

    binding = capture.binding
    role_map = {artifact.role: artifact for artifact in capture.artifacts}
    plan_artifact = role_map.get('collection-plan')
    manifest_artifact = role_map.get('run-manifest')
    if plan_artifact is None or manifest_artifact is None:
        raise PromotionIntegrityError('portable static capture omits plan or run manifest')
    try:
        plan = StaticHttpsCollectionPlan.model_validate_json(plan_artifact.payload)
        manifest = StaticHttpsRunManifest.model_validate_json(manifest_artifact.payload)
    except ValueError as error:
        raise PromotionIntegrityError('portable static plan or run manifest is invalid') from error
    if (
        canonical_json_bytes(plan) != plan_artifact.payload
        or canonical_json_bytes(manifest) != manifest_artifact.payload
    ):
        raise PromotionIntegrityError('portable static plan and run manifest must be canonical JSON')
    plan_sha256 = static_plan_sha256(plan)
    if (
        configuration.collection_plan_sha256 != plan_sha256
        or configuration.source_id != binding.source_id
        or plan.source_id != binding.source_id
        or manifest.plan_sha256 != plan_sha256
        or manifest.plan_id != plan.plan_id
        or manifest.source_id != binding.source_id
        or manifest.attempt_id != binding.attempt_id
        or manifest.logical_run_id != binding.logical_run_id
        or manifest.job_spec_sha256 != binding.job_spec_sha256
        or manifest.scheduled_for != binding.scheduled_for
        or manifest.attempt_started_at != binding.attempt_started_at
        or manifest.completed_at > binding.captured_at
    ):
        raise PromotionIntegrityError('portable static run manifest does not bind its capture lifecycle and plan')
    if (binding.captured_at - binding.attempt_started_at).total_seconds() > configuration.plan_deadline_seconds:
        raise PromotionIntegrityError('portable static success exceeds the immutable plan deadline')
    expected_roles = {'collection-plan', 'run-manifest'}
    expected_roles.update(f'body.{item.artifact_id}' for item in plan.artifacts)
    expected_roles.update(f'receipt.{item.artifact_id}' for item in plan.artifacts)
    if set(role_map) != expected_roles:
        raise PromotionIntegrityError('portable static capture attachment set does not exactly match its plan')
    if tuple(item.artifact_id for item in plan.artifacts) != tuple(item.artifact_id for item in manifest.artifacts):
        raise PromotionIntegrityError('portable static manifest artifact order differs from its plan')
    total_body_bytes = 0
    for artifact_spec, item in zip(plan.artifacts, manifest.artifacts, strict=True):
        body = role_map[f'body.{item.artifact_id}']
        receipt_artifact = role_map[f'receipt.{item.artifact_id}']
        try:
            receipt = HttpsCaptureReceipt.model_validate_json(receipt_artifact.payload)
        except ValueError as error:
            raise PromotionIntegrityError('portable HTTPS receipt is invalid') from error
        if canonical_json_bytes(receipt) != receipt_artifact.payload:
            raise PromotionIntegrityError('portable HTTPS receipt is not canonical JSON')
        request_sha256 = hashlib.sha256(canonical_json_bytes(artifact_spec.request)).hexdigest()
        if (
            (body.sha256, body.byte_count) != (item.body_sha256, item.body_byte_count)
            or (receipt_artifact.sha256, receipt_artifact.byte_count) != (item.receipt_sha256, item.receipt_byte_count)
            or item.request_sha256 != request_sha256
            or body.byte_count > artifact_spec.request.max_body_bytes
            or receipt.requested_url != artifact_spec.request.url
            or receipt.final_url != artifact_spec.request.url
            or receipt.request_headers != prepared_request_headers(artifact_spec.request)
            or receipt.status_code not in artifact_spec.request.allowed_status_codes
            or receipt.body_sha256 != body.sha256
            or receipt.body_byte_count != body.byte_count
            or receipt.started_at != item.started_at
            or receipt.completed_at != item.completed_at
            or receipt.started_at < binding.attempt_started_at
            or (receipt.completed_at - receipt.started_at).total_seconds() > configuration.request_deadline_seconds
        ):
            raise PromotionIntegrityError('portable static body/receipt replay failed')
        total_body_bytes += body.byte_count
        if total_body_bytes > configuration.max_total_body_bytes:
            raise PromotionIntegrityError('portable static bodies exceed the immutable aggregate body-byte budget')


def _load_bound_payloads(
    snapshot: ImmutableTreeSnapshot,
    manifest: CapturePromotionManifest,
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for binding in manifest.files:
        payload = snapshot.files.get(binding.path)
        if payload is None:
            raise PromotionIntegrityError(f'capture promotion omits a bound file: {binding.path}')
        if _binding(binding.path, payload) != binding:
            raise PromotionIntegrityError(f'capture promotion file differs from its binding: {binding.path}')
        payloads[binding.path] = payload
    return payloads


def _verify_snapshotted_witness(
    promotion_root: Path,
    payloads: Mapping[str, bytes],
    *,
    materials: WitnessMaterialSpec,
    expected_checkpoint_sha256: str,
    verified_at: datetime,
) -> LoadedWitnessedCheckpoint:
    manifest_path = 'witness/sidecar/witness.json'
    checkpoint_path = 'witness/sidecar/checkpoint.json'
    proof_path = 'witness/sidecar/external-proof.bin'
    manifest_bytes = payloads.get(manifest_path)
    exact_checkpoint_bytes = payloads.get(checkpoint_path)
    proof_bytes = payloads.get(proof_path)
    if manifest_bytes is None or exact_checkpoint_bytes is None or proof_bytes is None:
        raise PromotionIntegrityError('capture promotion omits the exact external witness sidecar')
    try:
        manifest = WitnessedCheckpointManifest.model_validate_json(manifest_bytes)
        checkpoint = LedgerCheckpoint.model_validate_json(exact_checkpoint_bytes)
    except ValueError as error:
        raise PromotionIntegrityError(f'invalid copied external witness sidecar: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest) or exact_checkpoint_bytes != checkpoint_bytes(checkpoint):
        raise PromotionIntegrityError('copied external witness sidecar is not canonical JSON')
    try:
        verify_witnessed_checkpoint(
            checkpoint,
            checkpoint_bytes=exact_checkpoint_bytes,
            manifest=manifest,
            proof_bytes=proof_bytes,
            verifier=materials.verifier,
            expected_policy=materials.policy,
            verified_at=verified_at,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
    except (TypeError, ValueError) as error:
        raise PromotionIntegrityError(f'copied external witness verification failed: {error}') from error
    return LoadedWitnessedCheckpoint(
        root=promotion_root / 'witness' / 'sidecar',
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_bytes=exact_checkpoint_bytes,
        proof_bytes=proof_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _require_exact_material(payloads: Mapping[str, bytes], path: str, expected: bytes) -> None:
    if payloads.get(path) != expected:
        raise PromotionIntegrityError(f'copied trusted material differs from independently supplied bytes: {path}')


def _one_output(index: CaptureIndex, role: NormalizedOutputRole) -> NormalizedOutputBinding:
    matches = tuple(output for output in index.normalized_outputs if output.role is role)
    if len(matches) != 1 or matches[0].file.byte_count < 1:
        raise PromotionIntegrityError(f'promotion requires exactly one nonempty {role.value} output')
    return matches[0]


def _parse_record_jsonl(payload: bytes, model, label: str):
    records = _parse_model_jsonl(payload, model, label)
    if not records:
        raise PromotionIntegrityError(f'normalized {label} records cannot be empty')
    return records


def _validate_trusted_materials(
    witness: WitnessMaterialSpec,
    source_verifiers: Mapping[str, SourceVerifierSpec],
    adapter: AdapterSpec,
) -> None:
    pinned = (
        ('witness policy', witness.policy_bytes, witness.policy.policy_sha256),
        ('witness trust policy', witness.trust_policy_bytes, witness.policy.trust_policy_sha256),
        (
            'witness verifier implementation',
            witness.verifier_implementation_bytes,
            witness.policy.verifier_implementation_sha256,
        ),
    )
    for label, payload, expected_sha256 in pinned:
        if not isinstance(payload, bytes) or not payload or hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise PromotionIntegrityError(f'{label} does not match its independently pinned digest')
    if witness.verifier is None:  # type: ignore[comparison-overlap]
        raise PromotionIntegrityError('trusted external witness verifier is required')
    if not source_verifiers:
        raise PromotionIntegrityError('at least one trusted source verifier is required')
    for source_id, spec in source_verifiers.items():
        if (
            not source_id
            or not isinstance(spec.implementation_bytes, bytes)
            or not spec.implementation_bytes
            or not isinstance(spec.policy_bytes, bytes)
            or not spec.policy_bytes
            or not isinstance(spec.execution_environment_bytes, bytes)
            or not spec.execution_environment_bytes
        ):
            raise PromotionIntegrityError(f'incomplete source verifier registration: {source_id!r}')
        if (spec.verifier is None) == (spec.hermetic_execution is None):
            raise PromotionIntegrityError(
                f'source verifier must select exactly one in-process or hermetic execution path: {source_id!r}'
            )
        if spec.hermetic_execution is not None:
            _validate_hermetic_materials(
                spec.hermetic_execution,
                execution_environment_bytes=spec.execution_environment_bytes,
                label=f'source verifier {source_id!r}',
            )
    for label, payload in (
        ('adapter implementation', adapter.implementation_bytes),
        ('adapter policy', adapter.policy_bytes),
        ('adapter execution environment', adapter.execution_environment_bytes),
    ):
        if not isinstance(payload, bytes) or not payload:
            raise PromotionIntegrityError(f'{label} must be nonempty exact bytes')
    reason_codes = adapter.allowed_exclusion_reason_codes
    if (
        not isinstance(reason_codes, tuple)
        or reason_codes != tuple(sorted(reason_codes))
        or len(reason_codes) != len(set(reason_codes))
        or any(
            not isinstance(code, str)
            or not code
            or len(code) > 100
            or not code[0].islower()
            or not code.replace('_', '').isalnum()
            for code in reason_codes
        )
    ):
        raise PromotionIntegrityError('adapter exclusion reason allowlist must be a canonical sorted tuple')
    if (adapter.adapter is None) == (adapter.hermetic_execution is None):
        raise PromotionIntegrityError('adapter must select exactly one in-process or hermetic execution path')
    if adapter.hermetic_execution is not None:
        _validate_hermetic_materials(
            adapter.hermetic_execution,
            execution_environment_bytes=adapter.execution_environment_bytes,
            label='adapter',
        )
    hermetic_modes = tuple(spec.hermetic_execution is not None for spec in source_verifiers.values()) + (
        adapter.hermetic_execution is not None,
    )
    if any(hermetic_modes) and not all(hermetic_modes):
        raise PromotionIntegrityError(
            'Tier A hermetic mode requires every source verifier and the adapter to cross the OCI boundary'
        )


def _validate_hermetic_materials(
    execution: HermeticExecutionSpec,
    *,
    execution_environment_bytes: bytes,
    label: str,
) -> None:
    if not isinstance(execution, HermeticExecutionSpec):
        raise PromotionIntegrityError(f'{label} hermetic execution spec has the wrong type')
    if not isinstance(execution.sandbox_policy_bytes, bytes) or not execution.sandbox_policy_bytes:
        raise PromotionIntegrityError(f'{label} hermetic sandbox policy must be nonempty exact bytes')
    if not isinstance(execution.seccomp_profile_bytes, bytes) or not execution.seccomp_profile_bytes:
        raise PromotionIntegrityError(f'{label} hermetic seccomp profile must be nonempty exact bytes')
    if not isinstance(execution.trusted_public_key_bytes, bytes) or len(execution.trusted_public_key_bytes) != 32:
        raise PromotionIntegrityError(f'{label} hermetic receipt public key must contain 32 bytes')
    try:
        environment = HermeticOciEnvironment.model_validate_json(execution_environment_bytes)
        sandbox = HermeticSandboxPolicy.model_validate_json(execution.sandbox_policy_bytes)
    except ValueError as error:
        raise PromotionIntegrityError(f'{label} hermetic environment or sandbox policy is invalid') from error
    if (
        canonical_json_bytes(environment) != execution_environment_bytes
        or canonical_json_bytes(sandbox) != execution.sandbox_policy_bytes
    ):
        raise PromotionIntegrityError(f'{label} hermetic materials must use exact canonical JSON bytes')
    if hashlib.sha256(execution.trusted_public_key_bytes).hexdigest() != sandbox.signing_public_key_sha256:
        raise PromotionIntegrityError(f'{label} hermetic receipt key differs from the sandbox policy')
    if hashlib.sha256(execution.seccomp_profile_bytes).hexdigest() != sandbox.seccomp_profile_sha256:
        raise PromotionIntegrityError(f'{label} hermetic seccomp profile differs from the sandbox policy')
    if execution.executor is None or not callable(getattr(execution.executor, 'execute', None)):
        raise PromotionIntegrityError(f'{label} requires a hermetic callback executor')


def _load_scoped_jobs(
    store: OperationalStore,
    scope: PromotionScopePolicy,
    events: tuple[LedgerEvent, ...],
) -> dict[str, RegisteredJob]:
    job_events = _event_maps(events)['jobs']
    jobs: dict[str, RegisteredJob] = {}
    for source_scope in scope.sources:
        for digest in source_scope.job_spec_sha256s:
            event = _require_event(job_events, digest, 'scoped job registration')
            try:
                job = store.get_job(digest)
            except Exception as error:
                raise PromotionIntegrityError(f'scope references an unknown job revision: {digest}') from error
            if (
                job.spec_sha256 != digest
                or event.occurred_at != job.registered_at
                or event.payload.get('job_id') != job.spec.job_id
            ):
                raise PromotionIntegrityError('scoped job does not reproduce its witnessed registration')
            try:
                configuration = parse_supported_collector_job_configuration(
                    job.spec.collector_id,
                    job.spec.configuration,
                )
            except ValueError as error:
                raise PromotionIntegrityError(
                    'scoped job uses an unsupported or invalid collector configuration'
                ) from error
            _require_v0_single_attempt_job(configuration.max_attempts_per_slot)
            if configuration.source_id != source_scope.source_id:
                raise PromotionIntegrityError('scoped job source differs from independently pinned scope')
            _expected_slots(job.spec, source_scope.scheduled_from, source_scope.scheduled_through)
            jobs[digest] = job
    expected = {digest for source in scope.sources for digest in source.job_spec_sha256s}
    if set(jobs) != expected:
        raise PromotionIntegrityError('scoped job inventory is not exhaustive')
    return jobs


def _expected_slots(spec: CaptureJobSpec, start: datetime, through: datetime) -> tuple[datetime, ...]:
    start = aware_utc(start, 'scheduled_from')
    through = aware_utc(through, 'scheduled_through')
    interval = timedelta(seconds=spec.schedule_interval_seconds)
    if start < spec.schedule_anchor_at:
        raise PromotionIntegrityError('scope schedule starts before the immutable job anchor')
    interval_microseconds = spec.schedule_interval_seconds * 1_000_000
    start_offset = start - spec.schedule_anchor_at
    start_microseconds = (start_offset.days * 86_400 + start_offset.seconds) * 1_000_000 + start_offset.microseconds
    if start_microseconds % interval_microseconds != 0:
        raise PromotionIntegrityError('scope scheduled_from is not an immutable job slot')
    through_offset = through - spec.schedule_anchor_at
    through_microseconds = (
        through_offset.days * 86_400 + through_offset.seconds
    ) * 1_000_000 + through_offset.microseconds
    if through_microseconds % interval_microseconds != 0:
        raise PromotionIntegrityError('scope scheduled_through is not an immutable job slot')
    count = int((through - start) // interval) + 1
    if count < 1 or count > _MAX_SCHEDULED_SLOTS:
        raise PromotionIntegrityError('scope scheduled-slot count is outside the V0 bound')
    return tuple(start + ordinal * interval for ordinal in range(count))


def _snapshot_scoped_captures(
    store: OperationalStore,
    scope: PromotionScopePolicy,
    jobs: Mapping[str, RegisteredJob],
    events: tuple[LedgerEvent, ...],
) -> tuple[ExactPromotedCapture, ...]:
    maps = _event_maps(events)
    drafts: list[tuple[str, str]] = []
    for source_scope in scope.sources:
        for digest in source_scope.job_spec_sha256s:
            job = jobs[digest]
            for scheduled_for in _expected_slots(job.spec, source_scope.scheduled_from, source_scope.scheduled_through):
                run_id = scheduled_logical_run_id(digest, scheduled_for)
                run_event = _require_event(maps['runs'], run_id, 'scoped logical run')
                success = _require_event(maps['success_runs'], run_id, 'scoped successful run')
                attempt_id = success.payload.get('attempt_id')
                if (
                    not isinstance(attempt_id, str)
                    or run_event.payload.get('job_spec_sha256') != digest
                    or _event_timestamp(run_event.payload.get('scheduled_for')) != scheduled_for
                ):
                    raise PromotionIntegrityError('scoped run does not bind its immutable schedule slot')
                drafts.append((source_scope.source_id, attempt_id))
    captures = tuple(
        sorted(
            (_snapshot_one_capture(store, source_id, attempt_id, events) for source_id, attempt_id in drafts),
            key=lambda item: (item.binding.source_id, item.binding.succeeded_event_sequence),
        )
    )
    if not captures or len({item.binding.attempt_id for item in captures}) != len(captures):
        raise PromotionIntegrityError('promotion scope produced no unique successful captures')
    return captures


def _snapshot_one_capture(
    store: OperationalStore,
    source_id: str,
    attempt_id: str,
    events: tuple[LedgerEvent, ...],
) -> ExactPromotedCapture:
    attempt = store.get_attempt(attempt_id)
    run = store.get_logical_run(attempt.logical_run_id)
    job = store.get_job(run.job_spec_sha256)
    try:
        if job.spec.collector_id == STATIC_HTTPS_COLLECTOR_ID:
            load_static_run_manifest(store, attempt_id)
        elif job.spec.collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID:
            from vaxreplay.operations.immport_capture import (
                load_immport_authenticated_run_manifest,
            )

            load_immport_authenticated_run_manifest(store, attempt_id)
        else:
            raise PromotionIntegrityError(f'no semantic verifier is registered for collector {job.spec.collector_id!r}')
    except PromotionIntegrityError:
        raise
    except Exception as error:
        raise PromotionIntegrityError('collector semantic replay rejected a scoped run') from error
    if attempt.state is not AttemptState.SUCCEEDED or attempt.finished_at is None:
        raise PromotionIntegrityError('scoped attempt is not a successful terminal capture')
    maps = _event_maps(events)
    job_event = _require_event(maps['jobs'], job.spec_sha256, 'job registration')
    run_event = _require_event(maps['runs'], run.logical_run_id, 'logical run')
    start_event = _require_event(maps['starts'], attempt_id, 'attempt start')
    success_event = _require_event(maps['successes'], attempt_id, 'attempt success')
    artifacts: list[ExactPromotedArtifact] = []
    raw_bindings: list[PromotedRawArtifactBinding] = []
    for ordinal, (role, artifact) in enumerate(sorted(store.list_attempt_artifacts(attempt_id).items())):
        stored_event = _require_event(maps['artifacts'], artifact.sha256, 'artifact first record')
        attached_event = _require_event(maps['attachments'], (attempt_id, role), 'artifact attachment')
        if not (stored_event.sequence < attached_event.sequence < success_event.sequence):
            raise PromotionIntegrityError('selected attachment is not transitively inside attempt success')
        payload = store.read_artifact(artifact.sha256, max_bytes=_MAX_FILE_BYTES)
        raw = PromotedRawArtifactBinding(
            role=role,
            file=_binding(f'raw/{success_event.sequence:012d}/{ordinal:06d}.bin', payload),
            first_recorded_at=artifact.first_recorded_at,
            stored_event_sequence=stored_event.sequence,
            stored_event_sha256=stored_event.event_sha256,
            attached_event_sequence=attached_event.sequence,
            attached_event_sha256=attached_event.event_sha256,
        )
        raw_bindings.append(raw)
        artifacts.append(ExactPromotedArtifact(binding=raw, payload=payload))
    binding = PromotedCaptureBinding(
        source_id=source_id,
        attempt_id=attempt_id,
        logical_run_id=run.logical_run_id,
        job_id=job.spec.job_id,
        collector_id=job.spec.collector_id,
        job_spec=job.spec,
        job_spec_sha256=job.spec_sha256,
        scheduled_for=run.scheduled_for,
        attempt_started_at=attempt.started_at,
        # Use the terminal success event as the conservative capture bound.  The
        # embedded static run manifest separately proves scientific-body completion
        # occurred no later than this value.
        captured_at=attempt.finished_at,
        job_registered_event_sequence=job_event.sequence,
        job_registered_event_sha256=job_event.event_sha256,
        run_registered_event_sequence=run_event.sequence,
        run_registered_event_sha256=run_event.event_sha256,
        started_event_sequence=start_event.sequence,
        started_event_sha256=start_event.event_sha256,
        succeeded_event_sequence=success_event.sequence,
        succeeded_event_sha256=success_event.event_sha256,
        artifacts=tuple(raw_bindings),
    )
    return ExactPromotedCapture(binding=binding, artifacts=tuple(artifacts))


def _event_maps(events: tuple[LedgerEvent, ...]) -> dict[str, dict[object, LedgerEvent]]:
    result: dict[str, dict[object, LedgerEvent]] = {
        'jobs': {},
        'runs': {},
        'starts': {},
        'successes': {},
        'success_runs': {},
        'artifacts': {},
        'attachments': {},
    }
    for event in events:
        payload = event.payload
        pairs: tuple[tuple[str, object], ...] = ()
        if event.event_type is LedgerEventType.JOB_REGISTERED:
            pairs = (('jobs', payload.get('job_spec_sha256')),)
        elif event.event_type is LedgerEventType.LOGICAL_RUN_REGISTERED:
            pairs = (('runs', payload.get('logical_run_id')),)
        elif event.event_type is LedgerEventType.ATTEMPT_STARTED:
            pairs = (('starts', payload.get('attempt_id')),)
        elif event.event_type is LedgerEventType.ATTEMPT_SUCCEEDED:
            pairs = (
                ('successes', payload.get('attempt_id')),
                ('success_runs', payload.get('logical_run_id')),
            )
        elif event.event_type is LedgerEventType.ARTIFACT_STORED:
            pairs = (('artifacts', payload.get('artifact_sha256')),)
        elif event.event_type is LedgerEventType.ATTEMPT_ARTIFACT_ATTACHED:
            attempt_id, role = payload.get('attempt_id'), payload.get('role')
            if not isinstance(attempt_id, str) or not isinstance(role, str):
                raise PromotionIntegrityError('witnessed ledger has a malformed attachment identity')
            pairs = (('attachments', (attempt_id, role)),)
        for name, key in pairs:
            if key is None or key in result[name]:
                raise PromotionIntegrityError(f'witnessed ledger has a malformed or duplicate {name} identity')
            result[name][key] = event
    return result


def _require_event(mapping: Mapping[object, LedgerEvent], key: object, label: str) -> LedgerEvent:
    event = mapping.get(key)
    if event is None:
        raise PromotionIntegrityError(f'{label} is outside the witnessed ledger prefix')
    return event


def _capture_inventory_sha256(captures: tuple[ExactPromotedCapture, ...]) -> str:
    inventory = tuple(
        {
            'artifacts': tuple(
                {'byte_count': item.byte_count, 'role': item.role, 'sha256': item.sha256} for item in capture.artifacts
            ),
            'attempt_id': capture.binding.attempt_id,
            'logical_run_id': capture.binding.logical_run_id,
            'scheduled_for': capture.binding.scheduled_for.isoformat().replace('+00:00', 'Z'),
            'source_id': capture.binding.source_id,
        }
        for capture in captures
    )
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def _hermetic_capture_payload(capture: ExactPromotedCapture) -> HermeticCapturePayload:
    return HermeticCapturePayload(
        binding=capture.binding,
        artifacts=tuple(
            HermeticArtifactPayload(
                binding=artifact.binding,
                payload_base64=encode_callback_bytes(artifact.payload),
            )
            for artifact in capture.artifacts
        ),
    )


def _source_verifier_input_bytes(value: SourceVerificationInput) -> bytes:
    return canonical_json_bytes(
        HermeticSourceVerifierInput(
            source_id=value.source_id,
            capture_inventory_sha256=value.capture_inventory_sha256,
            captures=tuple(
                _hermetic_capture_payload(capture)
                for capture in sorted(value.captures, key=lambda item: item.binding.attempt_id)
            ),
        )
    )


def _source_verifier_output_bytes(result: SourceVerificationResult, verified_records: bytes) -> bytes:
    return canonical_json_bytes(
        HermeticSourceVerifierOutput(
            result=result,
            verified_records_base64=encode_callback_bytes(verified_records),
        )
    )


def _adapter_input_bytes(values: tuple[AdapterSourceInput, ...]) -> bytes:
    return canonical_json_bytes(
        HermeticAdapterInput(
            sources=tuple(
                HermeticAdapterSourceInput(
                    source_id=value.source_id,
                    captures=tuple(
                        _hermetic_capture_payload(capture)
                        for capture in sorted(value.captures, key=lambda item: item.binding.attempt_id)
                    ),
                    verification_result=value.verification_result,
                    verified_records_base64=encode_callback_bytes(
                        b''.join(canonical_json_bytes(record) + b'\n' for record in value.verified_records)
                    ),
                )
                for value in values
            ),
        )
    )


def _adapter_output_bytes(value: AdapterRunResult) -> bytes:
    auxiliary = value.auxiliary_outputs or {}
    return canonical_json_bytes(
        HermeticAdapterOutput(
            candidate_records_base64=encode_callback_bytes(value.candidate_records),
            evidence_records_base64=encode_callback_bytes(value.evidence_records),
            dispositions_base64=encode_callback_bytes(value.dispositions),
            auxiliary_outputs=tuple(
                HermeticNamedOutput(name=name, payload_base64=encode_callback_bytes(auxiliary[name]))
                for name in sorted(auxiliary)
            ),
        )
    )


def _execute_hermetic_callback(
    *,
    execution: HermeticExecutionSpec,
    purpose: ExecutionPurpose,
    invocation_id: str,
    invocation_index: int,
    input_bytes: bytes,
    implementation_bytes: bytes,
    execution_environment_bytes: bytes,
    callback_policy_bytes: bytes,
) -> HermeticExecutionBundle:
    materials = HermeticCallbackMaterials(
        implementation_bytes=implementation_bytes,
        execution_environment_bytes=execution_environment_bytes,
        callback_policy_bytes=callback_policy_bytes,
    )
    try:
        bundle = execution.executor.execute(
            purpose=purpose,
            invocation_id=invocation_id,
            invocation_index=invocation_index,
            input_bytes=input_bytes,
            materials=materials,
        )
        verified = verify_hermetic_execution_bundle(
            request_bytes=bundle.request_bytes,
            response_bytes=bundle.response_bytes,
            receipt_bytes=bundle.receipt_bytes,
            expected_materials=materials,
            expected_sandbox_policy_bytes=execution.sandbox_policy_bytes,
            expected_seccomp_profile_bytes=execution.seccomp_profile_bytes,
            trusted_public_key_bytes=execution.trusted_public_key_bytes,
            image_inspection_bytes=bundle.image_inspection_bytes,
        )
    except Exception as error:
        raise PromotionIntegrityError(f'hermetic {purpose} execution failed closed: {error}') from error
    if (
        verified.request.purpose != purpose
        or verified.request.invocation_id != invocation_id
        or verified.request.invocation_index != invocation_index
        or decode_callback_bytes(verified.request.input_base64) != input_bytes
    ):
        raise PromotionIntegrityError(f'hermetic {purpose} receipt differs from its exact invocation or input')
    return verified


def _archive_hermetic_bundle(
    payloads: dict[str, bytes],
    *,
    bundle: HermeticExecutionBundle,
    execution: HermeticExecutionSpec,
    subject_id: str,
    prefix: str,
) -> HermeticExecutionPromotionBinding:
    artifacts = {
        f'{prefix}/image-inspection.json': bundle.image_inspection_bytes,
        f'{prefix}/receipt.json': bundle.receipt_bytes,
        f'{prefix}/request.json': bundle.request_bytes,
        f'{prefix}/response.json': bundle.response_bytes,
        f'{prefix}/sandbox-policy.json': execution.sandbox_policy_bytes,
        f'{prefix}/seccomp-profile.json': execution.seccomp_profile_bytes,
        f'{prefix}/trusted-public-key.bin': execution.trusted_public_key_bytes,
    }
    if any(path in payloads for path in artifacts):
        raise PromotionIntegrityError('hermetic execution archive path collides with an existing artifact')
    payloads.update(artifacts)
    attestation = bundle.receipt.attestation
    return HermeticExecutionPromotionBinding(
        subject_id=subject_id,
        purpose=attestation.purpose,
        invocation_id=attestation.invocation_id,
        invocation_index=attestation.invocation_index,
        request=_binding(f'{prefix}/request.json', bundle.request_bytes),
        response=_binding(f'{prefix}/response.json', bundle.response_bytes),
        receipt=_binding(f'{prefix}/receipt.json', bundle.receipt_bytes),
        image_inspection=_binding(f'{prefix}/image-inspection.json', bundle.image_inspection_bytes),
        sandbox_policy=_binding(f'{prefix}/sandbox-policy.json', execution.sandbox_policy_bytes),
        seccomp_profile=_binding(f'{prefix}/seccomp-profile.json', execution.seccomp_profile_bytes),
        trusted_public_key=_binding(f'{prefix}/trusted-public-key.bin', execution.trusted_public_key_bytes),
        output_sha256=hashlib.sha256(bundle.output_bytes).hexdigest(),
        output_byte_count=len(bundle.output_bytes),
        authority_id=attestation.authority_id,
        signing_key_id=attestation.signing_key_id,
        issued_at=attestation.issued_at,
    )


def _run_source_verifiers(
    captures: tuple[ExactPromotedCapture, ...],
    source_verifiers: Mapping[str, SourceVerifierSpec],
    payloads: dict[str, bytes],
    *,
    hermetic_bindings: list[HermeticExecutionPromotionBinding] | None = None,
) -> tuple[tuple[SourceVerificationBinding, ...], tuple[AdapterSourceInput, ...]]:
    from vaxreplay.operations.promotion_schema import SourceVerifierIdentity

    source_ids = tuple(sorted({capture.binding.source_id for capture in captures}))
    if set(source_verifiers) != set(source_ids):
        raise PromotionIntegrityError('trusted source verifiers must exactly cover promoted sources')
    bindings: list[SourceVerificationBinding] = []
    adapter_inputs: list[AdapterSourceInput] = []
    for ordinal, source_id in enumerate(source_ids):
        spec = source_verifiers[source_id]
        selected = tuple(capture for capture in captures if capture.binding.source_id == source_id)
        capture_inventory_sha256 = _capture_inventory_sha256(selected)
        verifier_input = SourceVerificationInput(source_id, selected, capture_inventory_sha256)
        if spec.hermetic_execution is not None:
            input_bytes = _source_verifier_input_bytes(verifier_input)
            bundle = _execute_hermetic_callback(
                execution=spec.hermetic_execution,
                purpose='source_verifier',
                invocation_id=f'source-verifier-{ordinal:06d}',
                invocation_index=ordinal,
                input_bytes=input_bytes,
                implementation_bytes=spec.implementation_bytes,
                execution_environment_bytes=spec.execution_environment_bytes,
                callback_policy_bytes=spec.policy_bytes,
            )
            output = parse_source_output(bundle.output_bytes)
            run_result = SourceVerifierRunResult(
                result=output.result,
                verified_records=decode_callback_bytes(output.verified_records_base64),
            )
            if hermetic_bindings is not None:
                hermetic_bindings.append(
                    _archive_hermetic_bundle(
                        payloads,
                        bundle=bundle,
                        execution=spec.hermetic_execution,
                        subject_id=source_id,
                        prefix=f'hermetic/source-verifier/{ordinal:06d}',
                    )
                )
        else:
            if spec.verifier is None:
                raise PromotionIntegrityError(f'trusted source verifier is absent for {source_id}')
            try:
                run_result = spec.verifier(verifier_input, spec.policy_bytes)
            except Exception as error:
                raise PromotionIntegrityError(f'trusted source verifier failed for {source_id}: {error}') from error
        if not isinstance(run_result, SourceVerifierRunResult):
            raise PromotionIntegrityError('trusted source verifier must return SourceVerifierRunResult')
        if not isinstance(run_result.result, SourceVerificationResult):
            raise PromotionIntegrityError('source verifier run result must contain SourceVerificationResult')
        if not isinstance(run_result.verified_records, bytes):
            raise PromotionIntegrityError('source verifier run result must contain exact verified-record bytes')
        result = SourceVerificationResult.model_validate_json(canonical_json_bytes(run_result.result))
        verified_records_bytes = run_result.verified_records
        verified_records = _parse_model_jsonl(
            verified_records_bytes,
            SourceRecordBinding,
            'verified source record',
        )
        if not verified_records:
            raise PromotionIntegrityError('source verifier record inventory cannot be empty')
        record_keys = tuple((record.source_id, record.source_record_id) for record in verified_records)
        if record_keys != tuple(sorted(record_keys)) or len(record_keys) != len(set(record_keys)):
            raise PromotionIntegrityError('verified source records must be canonically sorted and uniquely identified')
        if any(record.source_id != source_id for record in verified_records):
            raise PromotionIntegrityError('verified source record belongs to a different source')
        expected_identity = SourceVerifierIdentity(
            verifier_id=spec.verifier_id,
            verifier_version=spec.verifier_version,
            implementation_sha256=hashlib.sha256(spec.implementation_bytes).hexdigest(),
            execution_environment_sha256=hashlib.sha256(spec.execution_environment_bytes).hexdigest(),
        )
        if (
            result.source_id != source_id
            or result.verifier != expected_identity
            or result.verifier_policy_sha256 != hashlib.sha256(spec.policy_bytes).hexdigest()
            or result.verified_attempt_ids != tuple(sorted(capture.binding.attempt_id for capture in selected))
            or result.verified_capture_inventory_sha256 != capture_inventory_sha256
            or result.verified_source_record_inventory_sha256 != hashlib.sha256(verified_records_bytes).hexdigest()
            or result.verified_source_record_count != len(verified_records)
        ):
            raise PromotionIntegrityError(
                'source verifier result does not bind exact captures, records, policy, and code'
            )
        selected_body_artifacts = {
            artifact.sha256
            for capture in selected
            for artifact in capture.artifacts
            if artifact.role.startswith('body.')
        }
        if any(record.source_artifact_sha256 not in selected_body_artifacts for record in verified_records):
            raise PromotionIntegrityError('verified source record is not backed by a selected body artifact')
        release = result.source_release
        release_record = next(
            (record for record in verified_records if record.source_record_id == release.evidence_source_record_id),
            None,
        )
        if release_record is None or (
            release_record.source_record_sha256,
            release_record.source_artifact_sha256,
        ) != (
            release.evidence_source_record_sha256,
            release.evidence_sha256,
        ):
            raise PromotionIntegrityError(
                'authoritative source release does not bind its exact verified source record and artifact'
            )
        implementation_path = f'verifiers/{ordinal:06d}/implementation.bin'
        policy_path = f'verifiers/{ordinal:06d}/policy.bin'
        execution_environment_path = f'verifiers/{ordinal:06d}/execution-environment.bin'
        verified_records_path = f'sources/{ordinal:06d}/verified-records.jsonl'
        payloads[implementation_path] = spec.implementation_bytes
        payloads[policy_path] = spec.policy_bytes
        payloads[execution_environment_path] = spec.execution_environment_bytes
        payloads[verified_records_path] = verified_records_bytes
        binding = SourceVerificationBinding(
            source_id=source_id,
            verifier_policy=_binding(policy_path, spec.policy_bytes),
            verifier_implementation=_binding(implementation_path, spec.implementation_bytes),
            verifier_execution_environment=_binding(
                execution_environment_path,
                spec.execution_environment_bytes,
            ),
            verified_records=_binding(verified_records_path, verified_records_bytes),
            result=result,
            result_sha256=source_verification_result_sha256(result),
            run_dispositions=tuple(
                SuccessfulRunDisposition(
                    source_id=source_id,
                    attempt_id=capture.binding.attempt_id,
                    logical_run_id=capture.binding.logical_run_id,
                    succeeded_event_sequence=capture.binding.succeeded_event_sequence,
                    succeeded_event_sha256=capture.binding.succeeded_event_sha256,
                    disposition='selected',
                )
                for capture in selected
            ),
        )
        bindings.append(binding)
        adapter_inputs.append(AdapterSourceInput(source_id, selected, result, verified_records))
    return tuple(bindings), tuple(adapter_inputs)


def _normalize_adapter_result(
    value: object,
    inputs: tuple[AdapterSourceInput, ...],
    allowed_exclusion_reason_codes: tuple[str, ...],
) -> tuple[dict[str, bytes], bytes, tuple[SourceRecordDisposition, ...]]:
    if not isinstance(value, AdapterRunResult):
        raise PromotionIntegrityError('normalization adapter must return AdapterRunResult')
    outputs = {
        'normalized/candidates.jsonl': value.candidate_records,
        'normalized/evidence.jsonl': value.evidence_records,
    }
    for path, payload in outputs.items():
        if not isinstance(payload, bytes) or not payload:
            raise PromotionIntegrityError(f'normalized output must be nonempty exact bytes: {path}')
    candidates = _parse_model_jsonl(outputs['normalized/candidates.jsonl'], CandidateRecord, 'candidate records')
    evidence = _parse_model_jsonl(outputs['normalized/evidence.jsonl'], EvidenceRecord, 'evidence records')
    if not candidates or not evidence:
        raise PromotionIntegrityError('normalized candidate and evidence inventories must both be nonempty')
    candidate_keys = tuple((record.episode_id, record.candidate_id) for record in candidates)
    evidence_keys = tuple((record.episode_id, record.evidence_id) for record in evidence)
    if len(candidate_keys) != len(set(candidate_keys)):
        raise PromotionIntegrityError('normalized candidate output contains duplicate record identities')
    if len(evidence_keys) != len(set(evidence_keys)):
        raise PromotionIntegrityError('normalized evidence output contains duplicate record identities')
    auxiliary = value.auxiliary_outputs or {}
    for name, payload in sorted(auxiliary.items()):
        if not isinstance(name, str) or not name or '/' in name or name in {'.', '..'}:
            raise PromotionIntegrityError('adapter auxiliary output names must be portable single components')
        if not isinstance(payload, bytes) or not payload:
            raise PromotionIntegrityError('adapter auxiliary outputs must be nonempty exact bytes')
        outputs[f'normalized/auxiliary/{name}.bin'] = payload
    if not isinstance(value.dispositions, bytes):
        raise PromotionIntegrityError('adapter dispositions must be exact canonical JSONL bytes')
    dispositions = _parse_model_jsonl(
        value.dispositions,
        SourceRecordDisposition,
        'source record disposition',
    )
    if not dispositions:
        raise PromotionIntegrityError('adapter disposition inventory cannot be empty')
    disposition_keys = tuple((item.source_id, item.source_record_id) for item in dispositions)
    if disposition_keys != tuple(sorted(disposition_keys)):
        raise PromotionIntegrityError('adapter dispositions must be canonically sorted')
    if len(disposition_keys) != len(set(disposition_keys)):
        raise PromotionIntegrityError('adapter returned duplicate source-record dispositions')

    source_records = {
        (record.source_id, record.source_record_id): record
        for adapter_input in inputs
        for record in adapter_input.verified_records
    }
    if set(disposition_keys) != set(source_records):
        raise PromotionIntegrityError(
            'adapter dispositions must cover every verified source record exactly once without invention'
        )
    for disposition in dispositions:
        source_record = source_records[(disposition.source_id, disposition.source_record_id)]
        if (
            disposition.source_record_sha256 != source_record.source_record_sha256
            or disposition.source_artifact_sha256 != source_record.source_artifact_sha256
        ):
            raise PromotionIntegrityError('adapter disposition does not bind its exact verified source record')
        if disposition.disposition == 'excluded' and disposition.reason_code not in allowed_exclusion_reason_codes:
            raise PromotionIntegrityError('adapter exclusion reason is not independently allowlisted')

    candidate_refs = {
        key: NormalizedRecordReference(
            episode_id=record.episode_id,
            record_id=record.candidate_id,
            record_sha256=hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
        )
        for key, record in zip(candidate_keys, candidates, strict=True)
    }
    evidence_refs = {
        key: NormalizedRecordReference(
            episode_id=record.episode_id,
            record_id=record.evidence_id,
            record_sha256=hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
        )
        for key, record in zip(evidence_keys, evidence, strict=True)
    }
    candidate_edges: set[tuple[str, str]] = set()
    evidence_edges: set[tuple[str, str]] = set()
    for disposition in dispositions:
        for reference in disposition.candidate_record_refs:
            key = (reference.episode_id, reference.record_id)
            if candidate_refs.get(key) != reference:
                raise PromotionIntegrityError('source disposition references an unknown or changed candidate row')
            candidate_edges.add(key)
        for reference in disposition.evidence_record_refs:
            key = (reference.episode_id, reference.record_id)
            if evidence_refs.get(key) != reference:
                raise PromotionIntegrityError('source disposition references an unknown or changed evidence row')
            evidence_edges.add(key)
    if candidate_edges != set(candidate_refs) or evidence_edges != set(evidence_refs):
        raise PromotionIntegrityError('every normalized candidate and evidence row requires a source-record edge')
    return outputs, value.dispositions, dispositions


def _run_adapter_twice(
    inputs: tuple[AdapterSourceInput, ...],
    adapter: AdapterSpec,
    payloads: dict[str, bytes],
    *,
    hermetic_bindings: list[HermeticExecutionPromotionBinding] | None = None,
) -> tuple[
    tuple[NormalizedOutputBinding, ...],
    PromotionFileBinding,
    tuple[SourceRecordDisposition, ...],
]:
    source_ids = tuple(item.source_id for item in inputs)
    runs = []
    hermetic_input_bytes = _adapter_input_bytes(inputs) if adapter.hermetic_execution is not None else None
    for run_index in range(2):
        try:
            if adapter.hermetic_execution is not None:
                if hermetic_input_bytes is None:  # pragma: no cover - narrowed above
                    raise PromotionIntegrityError('hermetic adapter input is absent')
                bundle = _execute_hermetic_callback(
                    execution=adapter.hermetic_execution,
                    purpose='adapter',
                    invocation_id='normalization-adapter',
                    invocation_index=run_index,
                    input_bytes=hermetic_input_bytes,
                    implementation_bytes=adapter.implementation_bytes,
                    execution_environment_bytes=adapter.execution_environment_bytes,
                    callback_policy_bytes=adapter.policy_bytes,
                )
                output = parse_adapter_output(bundle.output_bytes)
                adapter_result = AdapterRunResult(
                    candidate_records=decode_callback_bytes(output.candidate_records_base64),
                    evidence_records=decode_callback_bytes(output.evidence_records_base64),
                    dispositions=decode_callback_bytes(output.dispositions_base64),
                    auxiliary_outputs={
                        item.name: decode_callback_bytes(item.payload_base64) for item in output.auxiliary_outputs
                    }
                    or None,
                )
                if hermetic_bindings is not None:
                    hermetic_bindings.append(
                        _archive_hermetic_bundle(
                            payloads,
                            bundle=bundle,
                            execution=adapter.hermetic_execution,
                            subject_id=adapter.adapter_id,
                            prefix=f'hermetic/adapter/{run_index:06d}',
                        )
                    )
            else:
                if adapter.adapter is None:
                    raise PromotionIntegrityError('trusted deterministic adapter is absent')
                adapter_result = adapter.adapter(inputs, adapter.policy_bytes)
            runs.append(
                _normalize_adapter_result(
                    adapter_result,
                    inputs,
                    adapter.allowed_exclusion_reason_codes,
                )
            )
        except Exception as error:
            if isinstance(error, PromotionIntegrityError):
                raise
            raise PromotionIntegrityError(f'normalization adapter failed: {error}') from error
    if runs[0] != runs[1]:
        raise PromotionIntegrityError('normalization adapter did not reproduce identical outputs')
    outputs, dispositions_bytes, dispositions = runs[0]
    payloads.update(outputs)
    disposition_path = 'normalized/dispositions.jsonl'
    payloads[disposition_path] = dispositions_bytes
    candidate_source_ids = tuple(sorted({item.source_id for item in dispositions if item.candidate_record_refs}))
    evidence_source_ids = tuple(sorted({item.source_id for item in dispositions if item.evidence_record_refs}))
    bindings = []
    for path in sorted(outputs):
        role = (
            NormalizedOutputRole.CANDIDATE_RECORDS
            if path == 'normalized/candidates.jsonl'
            else NormalizedOutputRole.EVIDENCE_RECORDS
            if path == 'normalized/evidence.jsonl'
            else NormalizedOutputRole.AUXILIARY
        )
        output_source_ids = (
            candidate_source_ids
            if role is NormalizedOutputRole.CANDIDATE_RECORDS
            else evidence_source_ids
            if role is NormalizedOutputRole.EVIDENCE_RECORDS
            else source_ids
        )
        bindings.append(
            NormalizedOutputBinding(role=role, source_ids=output_source_ids, file=_binding(path, outputs[path]))
        )
    return (
        tuple(sorted(bindings, key=lambda item: (item.role.value, item.file.path))),
        _binding(disposition_path, dispositions_bytes),
        dispositions,
    )


def _verify_archived_hermetic_executions(
    *,
    index: CaptureIndex,
    payloads: Mapping[str, bytes],
    captures: tuple[ExactPromotedCapture, ...],
    adapter_inputs: tuple[AdapterSourceInput, ...],
    source_verifiers: Mapping[str, SourceVerifierSpec],
    adapter: AdapterSpec,
) -> None:
    modes = tuple(spec.hermetic_execution is not None for spec in source_verifiers.values()) + (
        adapter.hermetic_execution is not None,
    )
    if not all(modes):
        if any(modes) or index.hermetic_executions:
            raise PromotionIntegrityError(
                'portable hermetic evidence requires every source verifier and the adapter to use hermetic execution'
            )
        return
    if len(index.hermetic_executions) != len(source_verifiers) + 2:
        raise PromotionIntegrityError('portable promotion omits required hermetic execution evidence')

    execution_by_key = {
        (binding.purpose, binding.subject_id, binding.invocation_index): binding
        for binding in index.hermetic_executions
    }
    for ordinal, source_id in enumerate(sorted(source_verifiers)):
        spec = source_verifiers[source_id]
        execution = spec.hermetic_execution
        if execution is None:  # pragma: no cover - all(modes) narrows this
            raise PromotionIntegrityError('source hermetic execution spec disappeared')
        selected = tuple(capture for capture in captures if capture.binding.source_id == source_id)
        source_input = SourceVerificationInput(source_id, selected, _capture_inventory_sha256(selected))
        verification = next(item for item in index.source_verifications if item.source_id == source_id)
        records = payloads[verification.verified_records.path]
        binding = execution_by_key.get(('source_verifier', source_id, ordinal))
        if binding is None:
            raise PromotionIntegrityError(f'portable promotion omits hermetic source receipt: {source_id}')
        _verify_archived_hermetic_bundle(
            binding=binding,
            payloads=payloads,
            execution=execution,
            materials=HermeticCallbackMaterials(
                implementation_bytes=spec.implementation_bytes,
                execution_environment_bytes=spec.execution_environment_bytes,
                callback_policy_bytes=spec.policy_bytes,
            ),
            expected_input_bytes=_source_verifier_input_bytes(source_input),
            expected_output_bytes=_source_verifier_output_bytes(verification.result, records),
        )

    adapter_execution = adapter.hermetic_execution
    if adapter_execution is None:  # pragma: no cover - all(modes) narrows this
        raise PromotionIntegrityError('adapter hermetic execution spec disappeared')
    candidate = _one_output(index, NormalizedOutputRole.CANDIDATE_RECORDS)
    evidence = _one_output(index, NormalizedOutputRole.EVIDENCE_RECORDS)
    auxiliary = {
        PurePosixPath(output.file.path).name.removesuffix('.bin'): payloads[output.file.path]
        for output in index.normalized_outputs
        if output.role is NormalizedOutputRole.AUXILIARY
    }
    expected_adapter_output = _adapter_output_bytes(
        AdapterRunResult(
            candidate_records=payloads[candidate.file.path],
            evidence_records=payloads[evidence.file.path],
            dispositions=payloads[index.normalization_dispositions.path],
            auxiliary_outputs=auxiliary or None,
        )
    )
    expected_adapter_input = _adapter_input_bytes(adapter_inputs)
    adapter_materials = HermeticCallbackMaterials(
        implementation_bytes=adapter.implementation_bytes,
        execution_environment_bytes=adapter.execution_environment_bytes,
        callback_policy_bytes=adapter.policy_bytes,
    )
    for run_index in range(2):
        binding = execution_by_key.get(('adapter', adapter.adapter_id, run_index))
        if binding is None:
            raise PromotionIntegrityError(f'portable promotion omits hermetic adapter receipt {run_index}')
        _verify_archived_hermetic_bundle(
            binding=binding,
            payloads=payloads,
            execution=adapter_execution,
            materials=adapter_materials,
            expected_input_bytes=expected_adapter_input,
            expected_output_bytes=expected_adapter_output,
        )


def _verify_archived_hermetic_bundle(
    *,
    binding: HermeticExecutionPromotionBinding,
    payloads: Mapping[str, bytes],
    execution: HermeticExecutionSpec,
    materials: HermeticCallbackMaterials,
    expected_input_bytes: bytes,
    expected_output_bytes: bytes,
) -> None:
    file_bindings = (
        binding.request,
        binding.response,
        binding.receipt,
        binding.image_inspection,
        binding.sandbox_policy,
        binding.seccomp_profile,
        binding.trusted_public_key,
    )
    for file_binding in file_bindings:
        payload = payloads.get(file_binding.path)
        if payload is None or _binding(file_binding.path, payload) != file_binding:
            raise PromotionIntegrityError(
                f'hermetic execution artifact differs from its capture-index binding: {file_binding.path}'
            )
    if (
        payloads[binding.sandbox_policy.path] != execution.sandbox_policy_bytes
        or payloads[binding.seccomp_profile.path] != execution.seccomp_profile_bytes
        or payloads[binding.trusted_public_key.path] != execution.trusted_public_key_bytes
    ):
        raise PromotionIntegrityError('archived hermetic trust material differs from its out-of-band specification')
    try:
        bundle = verify_hermetic_execution_bundle(
            request_bytes=payloads[binding.request.path],
            response_bytes=payloads[binding.response.path],
            receipt_bytes=payloads[binding.receipt.path],
            expected_materials=materials,
            expected_sandbox_policy_bytes=execution.sandbox_policy_bytes,
            expected_seccomp_profile_bytes=execution.seccomp_profile_bytes,
            trusted_public_key_bytes=execution.trusted_public_key_bytes,
            image_inspection_bytes=payloads[binding.image_inspection.path],
        )
    except Exception as error:
        raise PromotionIntegrityError(
            f'archived hermetic execution receipt failed offline verification: {error}'
        ) from error
    attestation = bundle.receipt.attestation
    if (
        bundle.request.purpose != binding.purpose
        or bundle.request.invocation_id != binding.invocation_id
        or bundle.request.invocation_index != binding.invocation_index
        or decode_callback_bytes(bundle.request.input_base64) != expected_input_bytes
        or bundle.output_bytes != expected_output_bytes
        or hashlib.sha256(bundle.output_bytes).hexdigest() != binding.output_sha256
        or len(bundle.output_bytes) != binding.output_byte_count
        or attestation.authority_id != binding.authority_id
        or attestation.signing_key_id != binding.signing_key_id
        or attestation.issued_at != binding.issued_at
    ):
        raise PromotionIntegrityError('archived hermetic execution differs from its exact promotion inputs or outputs')


def _adapter_binding(
    adapter: AdapterSpec,
    sources: tuple[SourceVerificationBinding, ...],
) -> AdapterBinding:
    return AdapterBinding(
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        implementation=_binding('adapter/implementation.bin', adapter.implementation_bytes),
        policy=_binding('adapter/policy.bin', adapter.policy_bytes),
        execution_environment=_binding('adapter/execution-environment.bin', adapter.execution_environment_bytes),
        input_inventories=tuple(
            AdapterInputInventoryBinding(
                source_id=binding.source_id,
                capture_inventory_sha256=binding.result.verified_capture_inventory_sha256,
                source_record_inventory_sha256=binding.result.verified_source_record_inventory_sha256,
                source_record_count=binding.result.verified_source_record_count,
                source_verification_result_sha256=binding.result_sha256,
            )
            for binding in sources
        ),
        allowed_exclusion_reason_codes=adapter.allowed_exclusion_reason_codes,
        disposition_count=sum(binding.result.verified_source_record_count for binding in sources),
    )


def _witness_binding(payloads, witnessed, materials: WitnessMaterialSpec) -> ExternalWitnessPromotionBinding:
    receipt = witnessed.manifest.receipt
    return ExternalWitnessPromotionBinding(
        witness_manifest=_binding('witness/sidecar/witness.json', payloads['witness/sidecar/witness.json']),
        checkpoint_file=_binding('witness/sidecar/checkpoint.json', payloads['witness/sidecar/checkpoint.json']),
        proof_file=_binding('witness/sidecar/external-proof.bin', payloads['witness/sidecar/external-proof.bin']),
        policy=_binding('witness/materials/policy.bin', materials.policy_bytes),
        trust_policy=_binding('witness/materials/trust-policy.bin', materials.trust_policy_bytes),
        verifier_implementation=_binding(
            'witness/materials/verifier-implementation.bin', materials.verifier_implementation_bytes
        ),
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


def _records_bytes(records: Sequence[StrictModel]) -> bytes:
    if not records:
        raise PromotionIntegrityError('portable record inventory cannot be empty')
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


def _validate_payload_inventory(payloads: Mapping[str, bytes]) -> None:
    if not payloads or len(payloads) > _MAX_FILES:
        raise PromotionIntegrityError('promotion file count is outside the V0 bound')
    total = 0
    for path, payload in payloads.items():
        normalized = PurePosixPath(path)
        if (
            not path
            or normalized.is_absolute()
            or '..' in normalized.parts
            or normalized.as_posix() != path
            or any(part in {'', '.'} for part in normalized.parts)
        ):
            raise PromotionIntegrityError('promotion contains a non-portable file path')
        if not isinstance(payload, bytes) or len(payload) > _MAX_FILE_BYTES:
            raise PromotionIntegrityError(f'promotion file exceeds the per-file byte limit: {path}')
        total += len(payload)
        if total > _MAX_TOTAL_BYTES:
            raise PromotionIntegrityError('promotion exceeds the V0 aggregate byte limit')


def _durable_publish(output_dir: Path, payloads: Mapping[str, bytes], manifest_bytes: bytes) -> Path:
    target_request = Path(output_dir).expanduser().absolute()
    target_request.parent.mkdir(parents=True, exist_ok=True)
    target = target_request.parent.resolve(strict=True) / target_request.name
    lock = target.parent / f'.{target.name}.publish.lock'
    try:
        lock_descriptor = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
            0o600,
        )
    except FileExistsError as error:
        raise PromotionIntegrityError(f'capture promotion publication is already locked: {target}') from error
    staging: Path | None = None
    installed = False
    try:
        os.fsync(lock_descriptor)
        _fsync_directory(target.parent)
        if os.path.lexists(target):
            raise PromotionIntegrityError(f'capture promotion output already exists: {target}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
        for relative, payload in sorted(payloads.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
        manifest_path = staging / _PROMOTION_MANIFEST
        with manifest_path.open('xb') as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        for directory in sorted((item for item in staging.rglob('*') if item.is_dir()), reverse=True):
            directory.chmod(0o555)
            _fsync_directory(directory)
        staging.chmod(0o555)
        _fsync_directory(staging)
        try:
            rename_directory_noreplace(staging, target)
        except FileExistsError as error:
            raise PromotionIntegrityError(f'capture promotion output already exists: {target}') from error
        installed = True
        _fsync_directory(target.parent)
        return target
    finally:
        if staging is not None and not installed:
            # Publication makes the immutable tree read-only before the final
            # rename. Restore directory write permission so an exclusive-rename
            # collision cannot strand the private staging artifact.
            staging.chmod(0o755)
            for directory in (item for item in staging.rglob('*') if item.is_dir()):
                directory.chmod(0o755)
            shutil.rmtree(staging, ignore_errors=True)
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
            _fsync_directory(target.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
