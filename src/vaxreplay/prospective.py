"""Build and verify immutable pre-outcome decision packages.

The prospective package is the first half of a Tier A episode.  It freezes only information that
can legitimately exist at decision time: the task configuration, candidate panel, visible
evidence, prespecified protocols, and exact source-capture manifests.  Labels and final episode
manifests are intentionally absent.

Creating and sealing a package does not, by itself, establish Tier A eligibility.  Official
admission also requires independent case-universe and source-capture verification, followed by the
remaining prospective release, execution, and outcome gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import rename_directory_noreplace as _rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord, Split, StrictModel
from vaxreplay.prospective_schema import ProspectiveEpisodeBinding
from vaxreplay.temporal_schema import (
    CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    DECISION_SNAPSHOT_SCHEMA_VERSION,
    EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION,
    PROTOCOL_ARTIFACT_NAMES,
    DecisionProtocolCommitments,
    DecisionTimeConfig,
    TemporalArtifactReceipt,
    TemporalArtifactRole,
    TemporalReceiptAuthority,
    TemporalReceiptVerifier,
    build_decision_snapshot_commitment,
)

PROSPECTIVE_PACKAGE_SCHEMA_VERSION = 'vaxreplay.prospective-decision-package.v0.3'
PROSPECTIVE_SEAL_SCHEMA_VERSION = 'vaxreplay.prospective-decision-seal.v0.2'
PROSPECTIVE_RECEIPT_REQUEST_SCHEMA_VERSION = 'vaxreplay.prospective-receipt-request.v0.1'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_ARTIFACT_BYTES = 1024 * 1024 * 1024
_DECISION_ROLES = (
    TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL,
    TemporalArtifactRole.EVIDENCE_SNAPSHOT,
    TemporalArtifactRole.DECISION_SNAPSHOT,
)
_PROSPECTIVE_AUTHORITIES = {
    TemporalReceiptAuthority.RFC3161_TIMESTAMP,
    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
}
# Top-level artifacts owned by the witnessed capture-to-release bridge.  These
# families are reserved even when the version is unknown: otherwise a future,
# stale, or deliberately misspelled bridge artifact could be presented as an
# opaque legacy source manifest and escape promotion-archive reverification.
_PROMOTION_BRIDGE_SOURCE_SCHEMA_PREFIXES = (
    'vaxreplay.promotion-handoff.',
    'vaxreplay.capture-index.',
    'vaxreplay.capture-promotion.',
    'vaxreplay.scope-precommit.',
)


class ProspectiveIntegrityError(ValueError):
    """Raised when prospective bytes or their external receipts fail closed verification."""


def is_promotion_bridge_source_schema_version(value: object) -> bool:
    """Return whether a source envelope claims a bridge-owned schema family.

    Matching by family, rather than only today's exact versions, makes unknown
    and superseded versions fail closed instead of silently downgrading to the
    generic legacy-source path.
    """

    return isinstance(value, str) and value.startswith(_PROMOTION_BRIDGE_SOURCE_SCHEMA_PREFIXES)


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class ProspectiveFileBinding(StrictModel):
    path: str = Field(
        pattern=r'^(?:candidates\.jsonl|evidence\.jsonl|protocols/[a-z_]+\.bin|source-captures/[0-9]{6}\.json)$'
    )
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
            raise ValueError('prospective artifact paths must be normalized and remain inside the package')
        return value


class ProspectiveProtocolBinding(StrictModel):
    name: Literal[
        'candidate_set_definition',
        'evidence_acquisition_spec',
        'outcome_adjudication_spec',
    ]
    file: ProspectiveFileBinding


class ProspectiveSourceCaptureBinding(StrictModel):
    source_id: str = Field(min_length=1)
    source_release_at: datetime
    captured_at: datetime
    witnessed_at: datetime
    file: ProspectiveFileBinding

    @field_validator('source_release_at', 'captured_at', 'witnessed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value, 'source capture timestamp')

    @model_validator(mode='after')
    def validate_capture_order(self) -> Self:
        if self.captured_at < self.source_release_at:
            raise ValueError('captured_at cannot predate the source release')
        if self.witnessed_at < self.captured_at:
            raise ValueError('witnessed_at cannot predate capture completion')
        return self


class ProspectiveDecisionContextCommitment(StrictModel):
    """The exact decision state covered by the third prospective receipt.

    The decision-snapshot identity alone does not include source-capture lineage.
    This commitment deliberately binds both that identity and every exact source
    descriptor binding, so a decision receipt cannot be reused with another
    capture history that happens to normalize to the same records.
    """

    schema_version: Literal['vaxreplay.prospective-decision-context.v0.1'] = PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    decision_snapshot_schema_version: Literal['vaxreplay.decision-snapshot.v0.1'] = DECISION_SNAPSHOT_SCHEMA_VERSION
    decision_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_snapshot_bytes: int = Field(gt=0)
    source_captures: tuple[ProspectiveSourceCaptureBinding, ...] = Field(min_length=1)


class ProspectiveReceiptRequest(StrictModel):
    """Exact digest request to submit to an external timestamp authority."""

    schema_version: Literal['vaxreplay.prospective-receipt-request.v0.1'] = PROSPECTIVE_RECEIPT_REQUEST_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    decision_at: datetime
    role: TemporalArtifactRole
    artifact_schema_version: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_bytes: int = Field(gt=0)

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return _aware(value, 'decision_at')

    @model_validator(mode='after')
    def validate_role(self) -> Self:
        if self.role not in _DECISION_ROLES:
            raise ValueError('prospective receipt requests can cover only decision-side artifacts')
        return self


class ProspectiveDecisionPackageManifest(StrictModel):
    schema_version: Literal['vaxreplay.prospective-decision-package.v0.3'] = PROSPECTIVE_PACKAGE_SCHEMA_VERSION
    episode: ProspectiveEpisodeBinding
    candidates: ProspectiveFileBinding
    evidence: ProspectiveFileBinding
    protocols: tuple[ProspectiveProtocolBinding, ...]
    source_captures: tuple[ProspectiveSourceCaptureBinding, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_bindings(self) -> Self:
        snapshot = self.episode.decision_snapshot
        if self.candidates.path != 'candidates.jsonl':
            raise ValueError('candidate artifact must use candidates.jsonl')
        if (
            self.candidates.sha256 != snapshot.candidate_universe_or_panel_sha256
            or self.candidates.byte_count != snapshot.candidate_universe_or_panel_bytes
        ):
            raise ValueError('candidate file binding does not match the decision snapshot')
        if self.evidence.path != 'evidence.jsonl':
            raise ValueError('evidence artifact must use evidence.jsonl')
        if (
            self.evidence.sha256 != snapshot.visible_evidence_sha256
            or self.evidence.byte_count != snapshot.visible_evidence_bytes
        ):
            raise ValueError('evidence file binding does not match the decision snapshot')

        protocol_names = tuple(binding.name for binding in self.protocols)
        if protocol_names != PROTOCOL_ARTIFACT_NAMES:
            raise ValueError('protocol bindings must use canonical name order')
        commitments = snapshot.protocol_commitments
        expected_protocol_hashes = {
            'candidate_set_definition': commitments.candidate_set_definition_sha256,
            'evidence_acquisition_spec': commitments.evidence_acquisition_spec_sha256,
            'outcome_adjudication_spec': commitments.outcome_adjudication_spec_sha256,
        }
        for binding in self.protocols:
            if binding.file.path != f'protocols/{binding.name}.bin':
                raise ValueError(f'{binding.name} uses the wrong package path')
            if binding.file.sha256 != expected_protocol_hashes[binding.name]:
                raise ValueError(f'{binding.name} does not match the decision-time commitment')

        source_ids = tuple(binding.source_id for binding in self.source_captures)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
            raise ValueError('source captures must have unique source IDs in sorted order')
        expected_paths = tuple(f'source-captures/{ordinal:06d}.json' for ordinal in range(len(source_ids)))
        actual_paths = tuple(binding.file.path for binding in self.source_captures)
        if actual_paths != expected_paths:
            raise ValueError('source capture paths must use contiguous canonical ordinals')
        if any(binding.witnessed_at > self.episode.decision_at for binding in self.source_captures):
            raise ValueError('source captures must be externally witnessed at or before decision_at')
        decision_context = ProspectiveDecisionContextCommitment(
            episode_id=self.episode.episode_id,
            decision_snapshot_sha256=self.episode.decision_snapshot_sha256,
            decision_snapshot_bytes=len(canonical_json_bytes(self.episode.decision_snapshot)),
            source_captures=self.source_captures,
        )
        decision_context_bytes = canonical_json_bytes(decision_context)
        if self.episode.decision_context_sha256 != _sha256(
            decision_context_bytes
        ) or self.episode.decision_context_bytes != len(decision_context_bytes):
            raise ValueError('prospective episode binding does not bind its exact source-capture decision context')
        return self


class ProspectiveProofBinding(StrictModel):
    receipt_id: str = Field(min_length=1)
    path: str = Field(pattern=r'^proofs/[0-9]{6}\.bin$')
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)


class ProspectiveDecisionSealManifest(StrictModel):
    schema_version: Literal['vaxreplay.prospective-decision-seal.v0.2'] = PROSPECTIVE_SEAL_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    decision_at: datetime
    decision_package_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    receipts: tuple[TemporalArtifactReceipt, ...] = Field(min_length=3, max_length=3)
    proofs: tuple[ProspectiveProofBinding, ...] = Field(min_length=3, max_length=3)
    verified_at: datetime

    @field_validator('decision_at', 'verified_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value, 'prospective seal timestamp')

    @model_validator(mode='after')
    def validate_seal(self) -> Self:
        roles = tuple(receipt.role for receipt in self.receipts)
        if roles != _DECISION_ROLES:
            raise ValueError('prospective seal receipts must use canonical decision-side role order')
        receipt_ids = tuple(receipt.receipt_id for receipt in self.receipts)
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError('prospective seal receipt IDs must be unique')
        if tuple(proof.receipt_id for proof in self.proofs) != receipt_ids:
            raise ValueError('prospective proof bindings must follow receipt order and IDs')
        expected_paths = tuple(f'proofs/{ordinal:06d}.bin' for ordinal in range(3))
        if tuple(proof.path for proof in self.proofs) != expected_paths:
            raise ValueError('prospective proof paths must use canonical ordinals')
        if any(receipt.authority_type not in _PROSPECTIVE_AUTHORITIES for receipt in self.receipts):
            raise ValueError('prospective seals require RFC 3161 or public transparency-log authorities')
        if any(receipt.witnessed_at > self.decision_at for receipt in self.receipts):
            raise ValueError('prospective decision-side receipts must be witnessed by decision_at')
        if any(receipt.witnessed_at > self.verified_at for receipt in self.receipts):
            raise ValueError('verified_at cannot predate an external receipt')
        decision_receipt = self.receipts[2]
        if (
            decision_receipt.artifact_schema_version != PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION
            or decision_receipt.artifact_sha256 != self.decision_context_sha256
        ):
            raise ValueError('decision receipt must bind the prospective decision-context commitment')
        return self


@dataclass(frozen=True)
class SourceCaptureArtifact:
    source_id: str
    source_release_at: datetime
    captured_at: datetime
    witnessed_at: datetime
    manifest_bytes: bytes

    def __post_init__(self) -> None:
        source_release_at = _aware(self.source_release_at, 'source_release_at')
        captured_at = _aware(self.captured_at, 'captured_at')
        witnessed_at = _aware(self.witnessed_at, 'witnessed_at')
        if captured_at < source_release_at:
            raise ValueError('captured_at cannot predate the source release')
        if witnessed_at < captured_at:
            raise ValueError('witnessed_at cannot predate capture completion')


@dataclass(frozen=True)
class LoadedProspectiveDecisionPackage:
    root: Path
    manifest: ProspectiveDecisionPackageManifest
    manifest_sha256: str
    candidates: tuple[CandidateRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    protocol_artifacts: Mapping[str, bytes]
    source_capture_artifacts: Mapping[str, bytes]

    @property
    def receipt_requests(self) -> tuple[ProspectiveReceiptRequest, ...]:
        return prospective_receipt_requests(self.manifest)


@dataclass(frozen=True)
class LoadedProspectiveDecisionSeal:
    root: Path
    manifest: ProspectiveDecisionSealManifest
    manifest_sha256: str
    proof_artifacts: Mapping[str, bytes]


def prospective_decision_package_sha256(manifest: ProspectiveDecisionPackageManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def prospective_decision_seal_sha256(manifest: ProspectiveDecisionSealManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def prospective_receipt_requests(
    manifest: ProspectiveDecisionPackageManifest,
) -> tuple[ProspectiveReceiptRequest, ...]:
    snapshot = manifest.episode.decision_snapshot
    decision_context = prospective_decision_context_commitment(manifest)
    decision_context_bytes = canonical_json_bytes(decision_context)
    artifacts = {
        TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: (
            CANDIDATE_ARTIFACT_SCHEMA_VERSION,
            snapshot.candidate_universe_or_panel_sha256,
            snapshot.candidate_universe_or_panel_bytes,
        ),
        TemporalArtifactRole.EVIDENCE_SNAPSHOT: (
            EVIDENCE_ARTIFACT_SCHEMA_VERSION,
            snapshot.visible_evidence_sha256,
            snapshot.visible_evidence_bytes,
        ),
        TemporalArtifactRole.DECISION_SNAPSHOT: (
            PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION,
            _sha256(decision_context_bytes),
            len(decision_context_bytes),
        ),
    }
    return tuple(
        ProspectiveReceiptRequest(
            episode_id=manifest.episode.episode_id,
            decision_at=manifest.episode.decision_at,
            role=role,
            artifact_schema_version=artifacts[role][0],
            artifact_sha256=artifacts[role][1],
            artifact_bytes=artifacts[role][2],
        )
        for role in _DECISION_ROLES
    )


def prospective_decision_context_commitment(
    manifest: ProspectiveDecisionPackageManifest,
) -> ProspectiveDecisionContextCommitment:
    """Derive the canonical lineage-bearing third-receipt artifact."""

    snapshot = manifest.episode.decision_snapshot
    return ProspectiveDecisionContextCommitment(
        episode_id=manifest.episode.episode_id,
        decision_snapshot_sha256=manifest.episode.decision_snapshot_sha256,
        decision_snapshot_bytes=len(canonical_json_bytes(snapshot)),
        source_captures=manifest.source_captures,
    )


def prospective_decision_context_sha256(manifest: ProspectiveDecisionPackageManifest) -> str:
    return _sha256(canonical_json_bytes(prospective_decision_context_commitment(manifest)))


def build_prospective_decision_package(
    output_dir: Path,
    *,
    config: DecisionTimeConfig,
    candidates: Sequence[CandidateRecord],
    evidence: Sequence[EvidenceRecord],
    protocol_artifacts: Mapping[str, bytes],
    candidate_set_available_at: datetime,
    source_captures: Sequence[SourceCaptureArtifact],
) -> LoadedProspectiveDecisionPackage:
    """Atomically freeze one exact, label-free decision package.

    The caller must provide evidence that is already cutoff-clean.  Future rows are rejected rather
    than silently filtered so the package cannot hide a contaminated input behind a clean digest.
    """

    if config.synthetic or config.split != Split.TEST:
        raise ValueError('prospective Tier A packages require a non-synthetic test decision config')
    candidate_records = tuple(candidates)
    evidence_records = tuple(evidence)
    if any(record.available_at > config.decision_at for record in evidence_records):
        raise ValueError('prospective packages reject evidence released after decision_at')
    if tuple(sorted(record.evidence_id for record in evidence_records)) != tuple(
        record.evidence_id for record in evidence_records
    ):
        raise ValueError('prospective evidence records must be sorted by evidence_id')
    if set(protocol_artifacts) != set(PROTOCOL_ARTIFACT_NAMES):
        raise ValueError('prospective packages require exactly the three protocol artifacts')
    if any(not protocol_artifacts[name] for name in PROTOCOL_ARTIFACT_NAMES):
        raise ValueError('prospective protocol artifacts cannot be empty')
    captures = tuple(sorted(source_captures, key=lambda capture: capture.source_id))
    if not captures:
        raise ValueError('prospective packages require at least one exact source capture manifest')
    source_ids = tuple(capture.source_id for capture in captures)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError('prospective source capture IDs must be unique')
    for capture in captures:
        _validate_canonical_json(capture.manifest_bytes, f'source capture {capture.source_id}')

    commitments = DecisionProtocolCommitments(
        candidate_set_available_at=candidate_set_available_at,
        candidate_set_definition_sha256=_sha256(protocol_artifacts['candidate_set_definition']),
        evidence_acquisition_spec_sha256=_sha256(protocol_artifacts['evidence_acquisition_spec']),
        outcome_adjudication_spec_sha256=_sha256(protocol_artifacts['outcome_adjudication_spec']),
    )
    snapshot = build_decision_snapshot_commitment(config, candidate_records, evidence_records, commitments)
    candidate_bytes = _records_bytes(candidate_records)
    evidence_bytes = _records_bytes(evidence_records)

    protocol_bindings = tuple(
        ProspectiveProtocolBinding(
            name=name,
            file=_file_binding(f'protocols/{name}.bin', protocol_artifacts[name]),
        )
        for name in PROTOCOL_ARTIFACT_NAMES
    )
    source_bindings = tuple(
        ProspectiveSourceCaptureBinding(
            source_id=capture.source_id,
            source_release_at=capture.source_release_at,
            captured_at=capture.captured_at,
            witnessed_at=capture.witnessed_at,
            file=_file_binding(f'source-captures/{ordinal:06d}.json', capture.manifest_bytes),
        )
        for ordinal, capture in enumerate(captures)
    )
    decision_context = ProspectiveDecisionContextCommitment(
        episode_id=config.episode_id,
        decision_snapshot_sha256=_sha256(canonical_json_bytes(snapshot)),
        decision_snapshot_bytes=len(canonical_json_bytes(snapshot)),
        source_captures=source_bindings,
    )
    decision_context_bytes = canonical_json_bytes(decision_context)
    episode = ProspectiveEpisodeBinding.from_decision_snapshot(
        snapshot,
        decision_context_sha256=_sha256(decision_context_bytes),
        decision_context_bytes=len(decision_context_bytes),
    )
    manifest = ProspectiveDecisionPackageManifest(
        episode=episode,
        candidates=_file_binding('candidates.jsonl', candidate_bytes),
        evidence=_file_binding('evidence.jsonl', evidence_bytes),
        protocols=protocol_bindings,
        source_captures=source_bindings,
    )

    target, staging, lock_path, lock_descriptor = _make_staging(output_dir)
    installed = False
    try:
        (staging / 'protocols').mkdir()
        (staging / 'source-captures').mkdir()
        _write_durable_file(staging / 'candidates.jsonl', candidate_bytes)
        _write_durable_file(staging / 'evidence.jsonl', evidence_bytes)
        for binding in protocol_bindings:
            _write_durable_file(staging / binding.file.path, protocol_artifacts[binding.name])
        for binding, capture in zip(source_bindings, captures, strict=True):
            _write_durable_file(staging / binding.file.path, capture.manifest_bytes)
        _write_durable_file(staging / 'decision.json', canonical_json_bytes(manifest))
        _sync_staging_tree(staging)
        if os.path.lexists(target):
            raise ValueError(f'prospective output already exists: {target}')
        _rename_directory_noreplace(staging, target)
        installed = True
        _fsync_directory(target.parent)
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)
        _release_publication_lock(lock_path, lock_descriptor)
    return load_prospective_decision_package(target)


def load_prospective_decision_package(root: Path) -> LoadedProspectiveDecisionPackage:
    resolved, root_descriptor, root_identity = _open_artifact_root(root, 'prospective decision package')
    try:
        manifest_bytes, manifest_identity = _read_regular_file_snapshot_at(
            root_descriptor,
            'decision.json',
            _MAX_MANIFEST_BYTES,
        )
        try:
            manifest = ProspectiveDecisionPackageManifest.model_validate_json(manifest_bytes)
        except ValueError as error:
            raise ProspectiveIntegrityError(f'invalid prospective decision manifest: {error}') from error
        if manifest_bytes != canonical_json_bytes(manifest):
            raise ProspectiveIntegrityError('prospective decision manifest must use canonical JSON encoding')
        file_bindings = (
            manifest.candidates,
            manifest.evidence,
            *(binding.file for binding in manifest.protocols),
            *(binding.file for binding in manifest.source_captures),
        )
        expected_files = {'decision.json', *(binding.path for binding in file_bindings)}
        allowed_directories = {'protocols', 'source-captures'}
        _validate_inventory_at(root_descriptor, expected_files, allowed_directories=allowed_directories)
        total_bytes = 0
        artifacts: dict[str, bytes] = {}
        file_identities = {'decision.json': manifest_identity}
        for binding in file_bindings:
            payload, identity = _read_regular_file_snapshot_at(
                root_descriptor,
                binding.path,
                _MAX_ARTIFACT_BYTES,
            )
            total_bytes += len(payload)
            if total_bytes > _MAX_TOTAL_ARTIFACT_BYTES:
                raise ProspectiveIntegrityError('prospective decision package exceeds the aggregate size limit')
            if len(payload) != binding.byte_count or _sha256(payload) != binding.sha256:
                raise ProspectiveIntegrityError(f'prospective artifact does not match its binding: {binding.path}')
            artifacts[binding.path] = payload
            file_identities[binding.path] = identity
        # Re-enumerate after reading so a concurrent add, remove, or directory
        # substitution cannot pass only the pre-read allowlist check.
        _validate_inventory_at(root_descriptor, expected_files, allowed_directories=allowed_directories)
        _require_file_identities_at(root_descriptor, file_identities)
        _require_root_path_identity(resolved, root_identity, 'prospective decision package')
    finally:
        os.close(root_descriptor)
    try:
        candidates = _parse_jsonl(artifacts['candidates.jsonl'], CandidateRecord)
        evidence = _parse_jsonl(artifacts['evidence.jsonl'], EvidenceRecord)
    except ValueError as error:
        raise ProspectiveIntegrityError(f'invalid prospective record artifact: {error}') from error
    expected_snapshot = build_decision_snapshot_commitment(
        manifest.episode.decision_snapshot.config,
        candidates,
        evidence,
        manifest.episode.decision_snapshot.protocol_commitments,
    )
    if expected_snapshot != manifest.episode.decision_snapshot:
        raise ProspectiveIntegrityError('prospective records do not reproduce the committed decision snapshot')
    for binding in manifest.source_captures:
        _validate_canonical_json(artifacts[binding.file.path], f'source capture {binding.source_id}')
    _validate_promoted_capture_lineage(manifest, artifacts)
    return LoadedProspectiveDecisionPackage(
        root=resolved,
        manifest=manifest,
        manifest_sha256=prospective_decision_package_sha256(manifest),
        candidates=candidates,
        evidence=evidence,
        protocol_artifacts={binding.name: artifacts[binding.file.path] for binding in manifest.protocols},
        source_capture_artifacts={
            binding.source_id: artifacts[binding.file.path] for binding in manifest.source_captures
        },
    )


def _validate_promoted_capture_lineage(
    manifest: ProspectiveDecisionPackageManifest,
    artifacts: Mapping[str, bytes],
) -> None:
    """Cross-bind recognized promotion handoffs to exact records and source times.

    Legacy/manual source manifests remain supported for research packages.  A
    ``promotion:`` source ID is a reserved namespace and always requires the
    one known handoff schema.  The descriptor remains structural evidence only;
    official admission must resolve and fully reverify its out-of-band promotion.
    """

    from vaxreplay.operations.promotion_schema import (  # local import avoids a module cycle
        PROMOTION_HANDOFF_SCHEMA_VERSION,
        PromotionHandoffDescriptor,
    )

    promotion_bindings = tuple(
        binding for binding in manifest.source_captures if binding.source_id.startswith('promotion:')
    )
    if promotion_bindings and len(manifest.source_captures) != 1:
        raise ProspectiveIntegrityError('promotion-backed decision packages cannot mix promotion and legacy captures')

    for binding in manifest.source_captures:
        payload = artifacts[binding.file.path]
        try:
            envelope = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        schema_version = envelope.get('schema_version') if isinstance(envelope, dict) else None
        declares_promotion_schema = is_promotion_bridge_source_schema_version(schema_version)
        if declares_promotion_schema and schema_version != PROMOTION_HANDOFF_SCHEMA_VERSION:
            raise ProspectiveIntegrityError('promotion source uses an unknown or invalid handoff schema')
        if declares_promotion_schema and not binding.source_id.startswith('promotion:'):
            raise ProspectiveIntegrityError(
                'promotion handoff schemas require the reserved promotion: source namespace'
            )
    if not promotion_bindings:
        return

    binding = promotion_bindings[0]
    payload = artifacts[binding.file.path]
    try:
        envelope = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProspectiveIntegrityError('promotion source uses an invalid handoff descriptor') from error
    if not isinstance(envelope, dict) or envelope.get('schema_version') != PROMOTION_HANDOFF_SCHEMA_VERSION:
        raise ProspectiveIntegrityError('promotion source uses an unknown or invalid handoff schema')
    try:
        descriptor = PromotionHandoffDescriptor.model_validate_json(payload)
    except ValueError as error:
        raise ProspectiveIntegrityError(f'invalid promotion handoff descriptor: {error}') from error
    if payload != canonical_json_bytes(descriptor):
        raise ProspectiveIntegrityError('promotion handoff descriptor must use canonical JSON encoding')
    expected_source_id = f'promotion:{descriptor.promotion_id}'
    if binding.source_id != expected_source_id:
        raise ProspectiveIntegrityError('promoted source capture ID does not bind its promotion identity')
    candidate_output = descriptor.candidate_output.file
    evidence_output = descriptor.evidence_output.file
    if (
        manifest.candidates.sha256 != candidate_output.sha256
        or manifest.candidates.byte_count != candidate_output.byte_count
    ):
        raise ProspectiveIntegrityError('prospective candidates do not match the promoted candidate output')
    if manifest.evidence.sha256 != evidence_output.sha256 or manifest.evidence.byte_count != evidence_output.byte_count:
        raise ProspectiveIntegrityError('prospective evidence does not match the promoted evidence output')

    if (
        binding.source_release_at != descriptor.maximum_source_release_at
        or binding.captured_at != descriptor.maximum_captured_at
        or binding.witnessed_at != descriptor.witnessed_at
    ):
        raise ProspectiveIntegrityError('prospective source times do not match the promotion handoff')
    if (
        manifest.episode.decision_snapshot.protocol_commitments.candidate_set_available_at
        != descriptor.promotion_created_at
    ):
        raise ProspectiveIntegrityError('candidate-set availability must equal the verified promotion creation time')


def build_prospective_decision_seal(
    output_dir: Path,
    *,
    package: LoadedProspectiveDecisionPackage,
    receipts: Sequence[TemporalArtifactReceipt],
    proof_artifacts: Mapping[str, bytes],
    receipt_verifier: TemporalReceiptVerifier,
    verified_at: datetime,
) -> LoadedProspectiveDecisionSeal:
    """Verify external timestamp proofs and atomically write a portable seal sidecar."""

    ordered_receipts = tuple(receipts)
    _verify_receipts(package, ordered_receipts, proof_artifacts, receipt_verifier)
    proof_bindings = tuple(
        ProspectiveProofBinding(
            receipt_id=receipt.receipt_id,
            path=f'proofs/{ordinal:06d}.bin',
            sha256=_sha256(proof_artifacts[receipt.receipt_id]),
            byte_count=len(proof_artifacts[receipt.receipt_id]),
        )
        for ordinal, receipt in enumerate(ordered_receipts)
    )
    manifest = ProspectiveDecisionSealManifest(
        episode_id=package.manifest.episode.episode_id,
        decision_at=package.manifest.episode.decision_at,
        decision_package_sha256=package.manifest_sha256,
        decision_snapshot_sha256=package.manifest.episode.decision_snapshot_sha256,
        decision_context_sha256=prospective_decision_context_sha256(package.manifest),
        receipts=ordered_receipts,
        proofs=proof_bindings,
        verified_at=verified_at,
    )
    target, staging, lock_path, lock_descriptor = _make_staging(output_dir)
    installed = False
    try:
        (staging / 'proofs').mkdir()
        for binding in proof_bindings:
            _write_durable_file(staging / binding.path, proof_artifacts[binding.receipt_id])
        _write_durable_file(staging / 'seal.json', canonical_json_bytes(manifest))
        _sync_staging_tree(staging)
        if os.path.lexists(target):
            raise ValueError(f'prospective output already exists: {target}')
        _rename_directory_noreplace(staging, target)
        installed = True
        _fsync_directory(target.parent)
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)
        _release_publication_lock(lock_path, lock_descriptor)
    return load_prospective_decision_seal(
        target,
        package=package,
        receipt_verifier=receipt_verifier,
    )


def load_prospective_decision_seal(
    root: Path,
    *,
    package: LoadedProspectiveDecisionPackage,
    receipt_verifier: TemporalReceiptVerifier,
) -> LoadedProspectiveDecisionSeal:
    """Load a seal only after re-verifying each independent timestamp proof."""

    resolved, root_descriptor, root_identity = _open_artifact_root(root, 'prospective decision seal')
    try:
        manifest_bytes, manifest_identity = _read_regular_file_snapshot_at(
            root_descriptor,
            'seal.json',
            _MAX_MANIFEST_BYTES,
        )
        try:
            manifest = ProspectiveDecisionSealManifest.model_validate_json(manifest_bytes)
        except ValueError as error:
            raise ProspectiveIntegrityError(f'invalid prospective decision seal: {error}') from error
        if manifest_bytes != canonical_json_bytes(manifest):
            raise ProspectiveIntegrityError('prospective decision seal must use canonical JSON encoding')
        expected_files = {'seal.json', *(proof.path for proof in manifest.proofs)}
        _validate_inventory_at(root_descriptor, expected_files, allowed_directories={'proofs'})
        proof_artifacts: dict[str, bytes] = {}
        file_identities = {'seal.json': manifest_identity}
        for proof in manifest.proofs:
            payload, identity = _read_regular_file_snapshot_at(
                root_descriptor,
                proof.path,
                _MAX_ARTIFACT_BYTES,
            )
            if len(payload) != proof.byte_count or _sha256(payload) != proof.sha256:
                raise ProspectiveIntegrityError(f'prospective proof does not match its binding: {proof.receipt_id}')
            proof_artifacts[proof.receipt_id] = payload
            file_identities[proof.path] = identity
        _validate_inventory_at(root_descriptor, expected_files, allowed_directories={'proofs'})
        _require_file_identities_at(root_descriptor, file_identities)
        _require_root_path_identity(resolved, root_identity, 'prospective decision seal')
    finally:
        os.close(root_descriptor)
    if (
        manifest.episode_id != package.manifest.episode.episode_id
        or manifest.decision_package_sha256 != package.manifest_sha256
        or manifest.decision_snapshot_sha256 != package.manifest.episode.decision_snapshot_sha256
        or manifest.decision_context_sha256 != prospective_decision_context_sha256(package.manifest)
    ):
        raise ProspectiveIntegrityError('prospective seal does not bind the supplied decision package')
    _verify_receipts(package, manifest.receipts, proof_artifacts, receipt_verifier)
    return LoadedProspectiveDecisionSeal(
        root=resolved,
        manifest=manifest,
        manifest_sha256=prospective_decision_seal_sha256(manifest),
        proof_artifacts=proof_artifacts,
    )


def _verify_receipts(
    package: LoadedProspectiveDecisionPackage,
    receipts: Sequence[TemporalArtifactReceipt],
    proof_artifacts: Mapping[str, bytes],
    receipt_verifier: TemporalReceiptVerifier,
) -> None:
    ordered = tuple(receipts)
    requests = package.receipt_requests
    if tuple(receipt.role for receipt in ordered) != _DECISION_ROLES:
        raise ProspectiveIntegrityError('prospective seal requires exactly the three decision-side receipts')
    if len({receipt.receipt_id for receipt in ordered}) != len(ordered):
        raise ProspectiveIntegrityError('prospective receipt IDs must be unique')
    if set(proof_artifacts) != {receipt.receipt_id for receipt in ordered}:
        raise ProspectiveIntegrityError('prospective seal requires exactly one proof artifact per receipt')
    for request, receipt in zip(requests, ordered, strict=True):
        if (
            receipt.role != request.role
            or receipt.artifact_schema_version != request.artifact_schema_version
            or receipt.artifact_sha256 != request.artifact_sha256
            or receipt.artifact_bytes != request.artifact_bytes
        ):
            raise ProspectiveIntegrityError(f'{receipt.role.value} receipt does not bind the requested artifact')
        if receipt.authority_type not in _PROSPECTIVE_AUTHORITIES:
            raise ProspectiveIntegrityError('prospective receipts require an external timestamp authority')
        if receipt.witnessed_at > package.manifest.episode.decision_at:
            raise ProspectiveIntegrityError('prospective receipts must be witnessed by decision_at')
        proof = proof_artifacts[receipt.receipt_id]
        if len(proof) != receipt.receipt_bytes or _sha256(proof) != receipt.receipt_sha256:
            raise ProspectiveIntegrityError(f'proof bytes do not match receipt {receipt.receipt_id}')
        try:
            verified = receipt_verifier(receipt, proof)
        except Exception as error:
            raise ProspectiveIntegrityError(f'receipt verifier failed for {receipt.receipt_id}: {error}') from error
        if not verified:
            raise ProspectiveIntegrityError(f'receipt verifier rejected {receipt.receipt_id}')

    snapshot = package.manifest.episode.decision_snapshot
    candidate_receipt, evidence_receipt, decision_receipt = ordered
    if candidate_receipt.witnessed_at < snapshot.protocol_commitments.candidate_set_available_at:
        raise ProspectiveIntegrityError('candidate receipt cannot predate candidate-set availability')
    if evidence_receipt.witnessed_at < snapshot.latest_visible_evidence_at:
        raise ProspectiveIntegrityError('evidence receipt cannot predate included evidence availability')
    if decision_receipt.witnessed_at < max(candidate_receipt.witnessed_at, evidence_receipt.witnessed_at):
        raise ProspectiveIntegrityError('decision receipt cannot predate its candidate or evidence components')


def _file_binding(path: str, payload: bytes) -> ProspectiveFileBinding:
    return ProspectiveFileBinding(path=path, sha256=_sha256(payload), byte_count=len(payload))


def _records_bytes(records: Sequence[StrictModel]) -> bytes:
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


def _parse_jsonl(payload: bytes, model: type[CandidateRecord] | type[EvidenceRecord]):
    if not payload.endswith(b'\n'):
        raise ValueError('record JSONL must end with a newline')
    records = []
    for ordinal, line in enumerate(payload.splitlines(), start=1):
        try:
            record = model.model_validate_json(line)
        except ValueError as error:
            raise ValueError(f'invalid JSONL record {ordinal}: {error}') from error
        if line != canonical_json_bytes(record):
            raise ValueError(f'JSONL record {ordinal} must use canonical JSON encoding')
        records.append(record)
    if not records:
        raise ValueError('record JSONL cannot be empty')
    return tuple(records)


def _validate_canonical_json(payload: bytes, label: str) -> None:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'{label} is not valid UTF-8 JSON: {error}') from error
    canonical = canonical_json_bytes(value)
    if payload not in {canonical, canonical + b'\n'}:
        raise ValueError(f'{label} must use canonical JSON encoding with at most one trailing newline')


def _make_staging(output_dir: Path) -> tuple[Path, Path, Path, int]:
    """Acquire an exclusive sibling lock and create a private staging directory."""

    requested = output_dir.expanduser().absolute()
    if not requested.name:
        raise ValueError('prospective output must name a child directory')
    requested.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise ValueError(f'cannot resolve prospective output parent: {error}') from error
    if not parent.is_dir():
        raise ValueError(f'prospective output parent is not a directory: {parent}')
    target = parent / requested.name
    lock_path = parent / f'.{target.name}.publish.lock'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0)
    try:
        lock_descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(f'prospective output publication is already locked: {target}') from error
    staging: Path | None = None
    try:
        os.fsync(lock_descriptor)
        _fsync_directory(parent)
        if os.path.lexists(target):
            raise ValueError(f'prospective output already exists: {target}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=parent))
        _fsync_directory(parent)
        return target, staging, lock_path, lock_descriptor
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        _release_publication_lock(lock_path, lock_descriptor)
        raise


def _write_durable_file(path: Path, payload: bytes) -> None:
    """Create one staging file exclusively and durably flush its exact bytes."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError('short write while publishing prospective artifact')
            written += count
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_staging_tree(root: Path) -> None:
    """Normalize modes and fsync every staging directory from leaves upward."""

    for directory, subdirectories, files in os.walk(root, topdown=False, followlinks=False):
        directory_path = Path(directory)
        for name in files:
            file_path = directory_path / name
            metadata = file_path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f'prospective staging artifact is not a regular file: {file_path}')
        for name in subdirectories:
            child = directory_path / name
            metadata = child.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f'prospective staging artifact is not a directory: {child}')
        directory_path.chmod(0o755)
        _fsync_directory(directory_path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        _fsync_directory(lock_path.parent)


def _open_artifact_root(root: Path, label: str) -> tuple[Path, int, tuple[int, int]]:
    # Preserve and walk the normalized caller path. Pre-resolving would erase an
    # intermediate symlink before the descriptor-relative ``O_NOFOLLOW`` checks.
    supplied = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(supplied.anchor, flags)
        for component in supplied.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ProspectiveIntegrityError(f'cannot open {label} root: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProspectiveIntegrityError(f'{label} root must be a directory')
        return supplied, descriptor, (metadata.st_dev, metadata.st_ino)
    except BaseException:
        os.close(descriptor)
        raise


def _require_root_path_identity(resolved: Path, expected: tuple[int, int], label: str) -> None:
    try:
        _current_path, descriptor, current_identity = _open_artifact_root(resolved, label)
    except ProspectiveIntegrityError as error:
        raise ProspectiveIntegrityError(f'{label} root changed while being read') from error
    try:
        if current_identity != expected:
            raise ProspectiveIntegrityError(f'{label} root changed while being read')
    finally:
        os.close(descriptor)


def _open_directory_at(parent_descriptor: int, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ProspectiveIntegrityError(f'cannot open prospective directory {name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise ProspectiveIntegrityError(f'cannot inspect prospective directory {name}: {error}') from error
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ProspectiveIntegrityError(f'prospective artifact entry is not a directory: {name}')
    return descriptor


def _read_regular_file_at(root_descriptor: int, relative_path: str, maximum_bytes: int) -> bytes:
    payload, _identity = _read_regular_file_snapshot_at(root_descriptor, relative_path, maximum_bytes)
    return payload


def _read_regular_file_snapshot_at(
    root_descriptor: int,
    relative_path: str,
    maximum_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    parts = PurePosixPath(relative_path).parts
    if not parts or PurePosixPath(relative_path).is_absolute() or '..' in parts:
        raise ProspectiveIntegrityError(f'unsafe prospective artifact path: {relative_path}')
    parent_descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            child_descriptor = _open_directory_at(parent_descriptor, component)
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        try:
            descriptor = os.open(parts[-1], flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise ProspectiveIntegrityError(f'cannot open prospective file {relative_path}: {error}') from error
    finally:
        os.close(parent_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProspectiveIntegrityError(f'prospective artifact is not a regular file: {relative_path}')
        if metadata.st_size > maximum_bytes:
            raise ProspectiveIntegrityError(f'prospective artifact exceeds its size limit: {relative_path}')
        identity = _file_identity(metadata)
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ProspectiveIntegrityError(f'prospective artifact changed while read: {relative_path}')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProspectiveIntegrityError(f'prospective artifact changed while read: {relative_path}')
        after = os.fstat(descriptor)
        after_identity = _file_identity(after)
        if after_identity != identity:
            raise ProspectiveIntegrityError(f'prospective artifact changed while read: {relative_path}')
        return b''.join(chunks), identity
    except OSError as error:
        raise ProspectiveIntegrityError(f'cannot read prospective file {relative_path}: {error}') from error
    finally:
        os.close(descriptor)


def _require_file_identities_at(
    root_descriptor: int,
    expected: Mapping[str, tuple[int, int, int, int, int, int]],
) -> None:
    for relative_path, identity in expected.items():
        parts = PurePosixPath(relative_path).parts
        parent_descriptor = os.dup(root_descriptor)
        try:
            for component in parts[:-1]:
                child_descriptor = _open_directory_at(parent_descriptor, component)
                os.close(parent_descriptor)
                parent_descriptor = child_descriptor
            try:
                metadata = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
            except OSError as error:
                raise ProspectiveIntegrityError(f'prospective artifact changed after read: {relative_path}') from error
        finally:
            os.close(parent_descriptor)
        if not stat.S_ISREG(metadata.st_mode) or _file_identity(metadata) != identity:
            raise ProspectiveIntegrityError(f'prospective artifact changed after read: {relative_path}')


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_inventory_at(
    root_descriptor: int,
    expected_files: set[str],
    *,
    allowed_directories: set[str],
) -> None:
    expected_by_directory: dict[str, set[str]] = {'': set()}
    for relative_path in expected_files:
        parts = PurePosixPath(relative_path).parts
        if len(parts) == 1:
            expected_by_directory[''].add(parts[0])
        elif len(parts) == 2 and parts[0] in allowed_directories:
            expected_by_directory.setdefault(parts[0], set()).add(parts[1])
        else:
            raise ProspectiveIntegrityError(f'prospective manifest declares an unsafe path: {relative_path}')
    if set(expected_by_directory) - {''} != allowed_directories:
        raise ProspectiveIntegrityError('prospective directory allowlist mismatch')
    try:
        root_names = tuple(os.listdir(root_descriptor))
    except OSError as error:
        raise ProspectiveIntegrityError(f'cannot enumerate prospective artifact root: {error}') from error
    root_metadata: dict[str, os.stat_result] = {}
    for name in root_names:
        try:
            metadata = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        except OSError as error:
            raise ProspectiveIntegrityError(f'cannot inspect prospective artifact entry: {name}') from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ProspectiveIntegrityError(f'prospective artifact cannot contain symlinks: {name}')
        root_metadata[name] = metadata
    expected_root_names = expected_by_directory[''] | allowed_directories
    if set(root_names) != expected_root_names or len(root_names) != len(expected_root_names):
        missing = sorted(expected_root_names - set(root_names))
        extra = sorted(set(root_names) - expected_root_names)
        raise ProspectiveIntegrityError(f'prospective file allowlist mismatch; missing={missing}, extra={extra}')
    for name in root_names:
        metadata = root_metadata[name]
        if name in allowed_directories:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ProspectiveIntegrityError(f'prospective artifact entry is not a directory: {name}')
            directory_descriptor = _open_directory_at(root_descriptor, name)
            try:
                child_names = tuple(os.listdir(directory_descriptor))
                expected_children = expected_by_directory[name]
                child_metadata_by_name: dict[str, os.stat_result] = {}
                for child in child_names:
                    child_metadata = os.stat(child, dir_fd=directory_descriptor, follow_symlinks=False)
                    if stat.S_ISLNK(child_metadata.st_mode):
                        raise ProspectiveIntegrityError(f'prospective artifact cannot contain symlinks: {name}/{child}')
                    child_metadata_by_name[child] = child_metadata
                if set(child_names) != expected_children or len(child_names) != len(expected_children):
                    missing = sorted(expected_children - set(child_names))
                    extra = sorted(set(child_names) - expected_children)
                    raise ProspectiveIntegrityError(
                        f'prospective file allowlist mismatch; missing={missing}, extra={extra}'
                    )
                for child in child_names:
                    child_metadata = child_metadata_by_name[child]
                    if not stat.S_ISREG(child_metadata.st_mode):
                        raise ProspectiveIntegrityError(
                            f'prospective artifact contains a non-regular file: {name}/{child}'
                        )
            except OSError as error:
                raise ProspectiveIntegrityError(f'cannot enumerate prospective directory {name}: {error}') from error
            finally:
                os.close(directory_descriptor)
        elif not stat.S_ISREG(metadata.st_mode):
            raise ProspectiveIntegrityError(f'prospective artifact contains a non-regular file: {name}')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
