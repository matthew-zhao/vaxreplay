"""Externally witness a complete prospective cohort release before submissions open.

This seal proves that the exact, fully reverified release tree existed strictly before the
declared submissions-open boundary and before the run deadline.  It deliberately does not
claim that the timestamp authority has seen the only release, that the release is unique, or
that later run-attempt selection is fair; those are separate registry and attempt controls.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import AtomicDirectoryPublication
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.prospective_release_approval_identity import (
    TierAProspectiveReleaseApprovalIdentity,
    TierAProspectiveReleaseApprovalIdentityError,
    TierAProspectiveReleaseApprovalReplay,
    reverify_tier_a_prospective_release_approval_identity,
)
from vaxreplay.prompt import PromptVariant
from vaxreplay.prospective_admission import CaseUniverseSealVerifier, SourceCaptureVerifier
from vaxreplay.prospective_release import (
    LoadedProspectiveCohortRelease,
    ProspectiveReleaseIntegrityError,
    load_prospective_cohort_release,
)
from vaxreplay.prospective_schema import (
    prospective_challenge_admission_sha256,
    prospective_suite_manifest_sha256,
)
from vaxreplay.temporal_schema import TemporalReceiptAuthority, TemporalReceiptVerifier

PROSPECTIVE_RELEASE_SEAL_TARGET_SCHEMA_VERSION = 'vaxreplay.prospective-release-seal-target.v0.2'
PROSPECTIVE_RELEASE_TIMESTAMP_PROOF_SCHEMA_VERSION = 'vaxreplay.prospective-release-timestamp-proof.v0.2'
PROSPECTIVE_RELEASE_SEAL_SCHEMA_VERSION = 'vaxreplay.prospective-release-seal.v0.2'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_PROOF_BYTES = 512 * 1024 * 1024
_MAX_RELEASE_FILE_BYTES = 512 * 1024 * 1024
_MAX_RELEASE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_RELEASE_FILES = 100_000
_MAX_RELEASE_DIRECTORIES = 20_000
_EXTERNAL_AUTHORITIES = {
    TemporalReceiptAuthority.RFC3161_TIMESTAMP,
    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
}


class ProspectiveReleaseSealIntegrityError(ValueError):
    """Raised when a pre-run release seal or one of its bound inputs changed."""


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class ProspectiveReleaseSealTarget(StrictModel):
    """Canonical complete-release identity sent to an independent timestamp authority."""

    schema_version: Literal['vaxreplay.prospective-release-seal-target.v0.2'] = (
        PROSPECTIVE_RELEASE_SEAL_TARGET_SCHEMA_VERSION
    )
    tier_a_release_approval: TierAProspectiveReleaseApprovalIdentity
    release_id: str = Field(min_length=1)
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_release_manifest_bytes: int = Field(gt=0)
    release_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_tree_bytes: int = Field(gt=0)
    release_tree_file_count: int = Field(gt=0)
    release_tree_directory_count: int = Field(ge=0)
    challenge_id: str = Field(min_length=1)
    challenge_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    challenge_manifest_bytes: int = Field(gt=0)
    prompt_variant: PromptVariant
    challenge_sample_index: int = Field(ge=0)
    prospective_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_admission_bytes: int = Field(gt=0)
    suite_id: str = Field(min_length=1)
    prospective_suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_suite_bytes: int = Field(gt=0)
    episode_count: int = Field(gt=0)
    latest_prerequisite_witnessed_at: datetime
    submissions_open_at: datetime
    run_deadline_at: datetime

    @field_validator(
        'latest_prerequisite_witnessed_at',
        'submissions_open_at',
        'run_deadline_at',
    )
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value, 'prospective release-seal target timestamp')

    @model_validator(mode='after')
    def validate_window(self) -> Self:
        if self.submissions_open_at <= self.latest_prerequisite_witnessed_at:
            raise ValueError('submissions must open after every source and case-universe witness')
        if self.submissions_open_at >= self.run_deadline_at:
            raise ValueError('submissions must open before the preregistered run deadline')
        approval = self.tier_a_release_approval
        if (
            approval.release_id != self.release_id
            or approval.prospective_release_sha256 != self.prospective_release_sha256
            or approval.release_tree_sha256 != self.release_tree_sha256
            or approval.challenge_bundle_sha256 != self.challenge_bundle_sha256
            or approval.episode_count != self.episode_count
        ):
            raise ValueError('release-seal target direct fields differ from its Tier A approval identity')
        if self.latest_prerequisite_witnessed_at < approval.verified_at:
            raise ValueError('release-seal prerequisites omit the Tier A approval verification time')
        return self


class ProspectiveReleaseTimestampProof(StrictModel):
    """Metadata for exact external proof bytes over one canonical release-seal target."""

    schema_version: Literal['vaxreplay.prospective-release-timestamp-proof.v0.2'] = (
        PROSPECTIVE_RELEASE_TIMESTAMP_PROOF_SCHEMA_VERSION
    )
    receipt_id: str = Field(min_length=1)
    authority_type: TemporalReceiptAuthority
    authority_id: str = Field(min_length=1)
    target_schema_version: Literal['vaxreplay.prospective-release-seal-target.v0.2'] = (
        PROSPECTIVE_RELEASE_SEAL_TARGET_SCHEMA_VERSION
    )
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    witnessed_at: datetime
    proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'prospective release witness timestamp')

    @field_validator('authority_type')
    @classmethod
    def validate_authority(cls, value: TemporalReceiptAuthority) -> TemporalReceiptAuthority:
        if value not in _EXTERNAL_AUTHORITIES:
            raise ValueError('release seals require RFC 3161 or a public transparency log')
        return value


class ProspectiveReleaseSealManifest(StrictModel):
    """Exact three-file allowlist for a verified pre-run release-seal artifact."""

    schema_version: Literal['vaxreplay.prospective-release-seal.v0.2'] = PROSPECTIVE_RELEASE_SEAL_SCHEMA_VERSION
    target_path: Literal['target.json'] = 'target.json'
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    proof_path: Literal['timestamp-proof.bin'] = 'timestamp-proof.bin'
    timestamp_proof: ProspectiveReleaseTimestampProof

    @model_validator(mode='after')
    def validate_proof_target(self) -> Self:
        if (
            self.timestamp_proof.target_sha256 != self.target_sha256
            or self.timestamp_proof.target_bytes != self.target_bytes
        ):
            raise ValueError('external timestamp proof does not bind the declared release target')
        return self


type ProspectiveReleaseTimestampVerifier = Callable[[ProspectiveReleaseTimestampProof, bytes], bool]


@dataclass(frozen=True)
class LoadedProspectiveReleaseSeal:
    root: Path
    manifest: ProspectiveReleaseSealManifest
    target: ProspectiveReleaseSealTarget
    proof_bytes: bytes
    manifest_sha256: str


@dataclass(frozen=True)
class _ReleaseTreeInventory:
    sha256: str
    byte_count: int
    file_count: int
    directory_count: int


def prospective_release_seal_target_sha256(target: ProspectiveReleaseSealTarget) -> str:
    return _sha256(canonical_json_bytes(target))


def prospective_release_seal_manifest_sha256(manifest: ProspectiveReleaseSealManifest) -> str:
    return _sha256(canonical_json_bytes(manifest))


def build_prospective_release_seal_target(
    release: LoadedProspectiveCohortRelease,
    *,
    submissions_open_at: datetime,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
) -> ProspectiveReleaseSealTarget:
    """Freshly reverify a complete release and construct the only bytes to timestamp."""

    opening = _aware(submissions_open_at, 'submissions_open_at')
    approval = _reverify_approval_identity(
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
        error_type=ValueError,
    )
    fresh = _require_fully_reverified_release(
        release,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        error_type=ValueError,
    )
    inventory = _inventory_release_tree(fresh.root, error_type=ValueError)
    _require_approval_matches_release(approval, fresh, inventory=inventory, error_type=ValueError)
    # Detect changes that raced the inventory walk before returning a target.
    confirmed = _require_fully_reverified_release(
        release,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        error_type=ValueError,
    )
    confirmed_inventory = _inventory_release_tree(confirmed.root, error_type=ValueError)
    confirmed_approval = _reverify_approval_identity(
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
        error_type=ValueError,
    )
    _require_approval_matches_release(
        confirmed_approval,
        confirmed,
        inventory=confirmed_inventory,
        error_type=ValueError,
    )
    if confirmed != fresh or confirmed_inventory != inventory or confirmed_approval != approval:
        raise ValueError('prospective cohort release changed while its seal target was constructed')

    release_manifest_bytes = canonical_json_bytes(fresh.manifest)
    challenge_manifest_bytes = canonical_json_bytes(fresh.challenge.manifest)
    admission_bytes = canonical_json_bytes(fresh.verified_admission.admission)
    suite_bytes = canonical_json_bytes(fresh.verified_admission.suite)
    sample_indices = {envelope.sample_index for envelope in fresh.challenge.envelopes}
    if len(sample_indices) != 1:
        raise ValueError('prospective challenge envelopes do not share one challenge sample index')
    latest_prerequisite = max(
        _latest_prerequisite_witnessed_at(fresh),
        _aware(approval.verified_at, 'Tier A approval verified_at'),
    )
    return ProspectiveReleaseSealTarget(
        tier_a_release_approval=approval,
        release_id=fresh.manifest.release_id,
        prospective_release_sha256=fresh.release_sha256,
        prospective_release_manifest_bytes=len(release_manifest_bytes),
        release_tree_sha256=inventory.sha256,
        release_tree_bytes=inventory.byte_count,
        release_tree_file_count=inventory.file_count,
        release_tree_directory_count=inventory.directory_count,
        challenge_id=fresh.challenge.manifest.challenge_id,
        challenge_bundle_sha256=fresh.challenge.manifest_sha256,
        challenge_manifest_bytes=len(challenge_manifest_bytes),
        prompt_variant=fresh.challenge.manifest.prompt_variant,
        challenge_sample_index=next(iter(sample_indices)),
        prospective_admission_sha256=prospective_challenge_admission_sha256(fresh.verified_admission.admission),
        prospective_admission_bytes=len(admission_bytes),
        suite_id=fresh.verified_admission.suite.suite_id,
        prospective_suite_sha256=prospective_suite_manifest_sha256(fresh.verified_admission.suite),
        prospective_suite_bytes=len(suite_bytes),
        episode_count=len(fresh.verified_admission.suite.episodes),
        latest_prerequisite_witnessed_at=latest_prerequisite,
        submissions_open_at=opening,
        run_deadline_at=fresh.verified_admission.admission.run_deadline_at,
    )


def build_prospective_release_seal(
    output_dir: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    submissions_open_at: datetime,
    timestamp_proof: ProspectiveReleaseTimestampProof,
    proof_bytes: bytes,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    timestamp_verifier: ProspectiveReleaseTimestampVerifier,
) -> LoadedProspectiveReleaseSeal:
    """Verify an independent pre-opening witness and atomically write its exact proof."""

    if timestamp_verifier is None:  # type: ignore[comparison-overlap]
        raise ValueError('an independent trusted release timestamp verifier is required')
    target = build_prospective_release_seal_target(
        release,
        submissions_open_at=submissions_open_at,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
    )
    target_bytes = canonical_json_bytes(target)
    _require_valid_timestamp_proof(
        target=target,
        target_bytes=target_bytes,
        timestamp_proof=timestamp_proof,
        proof_bytes=proof_bytes,
        timestamp_verifier=timestamp_verifier,
        error_type=ValueError,
    )
    manifest = ProspectiveReleaseSealManifest(
        target_sha256=_sha256(target_bytes),
        target_bytes=len(target_bytes),
        timestamp_proof=timestamp_proof,
    )

    with AtomicDirectoryPublication.create(output_dir) as publication:
        publication.write_bytes(manifest.target_path, target_bytes)
        publication.write_bytes(manifest.proof_path, proof_bytes)
        publication.write_bytes('seal.json', canonical_json_bytes(manifest))
        target_root = publication.publish()
        loaded = load_prospective_release_seal(
            target_root,
            release=release,
            submissions_open_at=submissions_open_at,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            timestamp_verifier=timestamp_verifier,
        )
        publication.commit()
        return loaded


def load_prospective_release_seal(
    root: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    submissions_open_at: datetime,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    timestamp_verifier: ProspectiveReleaseTimestampVerifier,
) -> LoadedProspectiveReleaseSeal:
    """Fail closed, freshly reverify the release, and rerun the trusted timestamp verifier."""

    if timestamp_verifier is None:  # type: ignore[comparison-overlap]
        raise ProspectiveReleaseSealIntegrityError('an independent trusted release timestamp verifier is required')
    resolved = _resolve_root(root)
    _require_exact_seal_inventory(resolved)
    manifest_bytes = _read_regular_file(resolved / 'seal.json', _MAX_MODEL_BYTES)
    try:
        manifest = ProspectiveReleaseSealManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise ProspectiveReleaseSealIntegrityError(f'invalid prospective release-seal manifest: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ProspectiveReleaseSealIntegrityError('prospective release-seal manifest must use canonical JSON encoding')

    target_bytes = _read_regular_file(resolved / manifest.target_path, _MAX_MODEL_BYTES)
    try:
        target = ProspectiveReleaseSealTarget.model_validate_json(target_bytes)
    except ValueError as error:
        raise ProspectiveReleaseSealIntegrityError(f'invalid prospective release-seal target: {error}') from error
    if target_bytes != canonical_json_bytes(target):
        raise ProspectiveReleaseSealIntegrityError('prospective release-seal target must use canonical JSON encoding')
    if _sha256(target_bytes) != manifest.target_sha256 or len(target_bytes) != manifest.target_bytes:
        raise ProspectiveReleaseSealIntegrityError(
            'prospective release-seal target does not match its manifest binding'
        )

    try:
        expected_target = build_prospective_release_seal_target(
            release,
            submissions_open_at=submissions_open_at,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
        )
    except ValueError as error:
        raise ProspectiveReleaseSealIntegrityError(
            f'prospective cohort release reverification failed: {error}'
        ) from error
    if target != expected_target:
        raise ProspectiveReleaseSealIntegrityError('prospective release seal is bound to different release inputs')

    proof_bytes = _read_regular_file(resolved / manifest.proof_path, _MAX_PROOF_BYTES)
    _require_valid_timestamp_proof(
        target=target,
        target_bytes=target_bytes,
        timestamp_proof=manifest.timestamp_proof,
        proof_bytes=proof_bytes,
        timestamp_verifier=timestamp_verifier,
        error_type=ProspectiveReleaseSealIntegrityError,
    )
    return LoadedProspectiveReleaseSeal(
        root=resolved,
        manifest=manifest,
        target=target,
        proof_bytes=proof_bytes,
        manifest_sha256=_sha256(manifest_bytes),
    )


def _require_fully_reverified_release(
    release: LoadedProspectiveCohortRelease,
    *,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    error_type: type[ValueError],
) -> LoadedProspectiveCohortRelease:
    if not isinstance(release, LoadedProspectiveCohortRelease):
        raise error_type('release seal requires a fully loaded prospective cohort release')
    if not release.challenge.authority_proofs_reverified:
        raise error_type('release decision proofs were not reverified by a trusted authority callback')
    try:
        reloaded = load_prospective_cohort_release(
            release.root,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_release_sha256=release.release_sha256,
        )
    except (ValueError, ProspectiveReleaseIntegrityError) as error:
        raise error_type(f'prospective cohort release reverification failed: {error}') from error
    if reloaded != release or not reloaded.challenge.authority_proofs_reverified:
        raise error_type('prospective cohort release changed after trusted loading')
    return reloaded


def _reverify_approval_identity(
    *,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    error_type: type[ValueError],
) -> TierAProspectiveReleaseApprovalIdentity:
    try:
        return reverify_tier_a_prospective_release_approval_identity(
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
        )
    except (TierAProspectiveReleaseApprovalIdentityError, TypeError, ValueError) as error:
        raise error_type(f'Tier A prospective-release approval reverification failed: {error}') from error


def _require_approval_matches_release(
    approval: TierAProspectiveReleaseApprovalIdentity,
    release: LoadedProspectiveCohortRelease,
    *,
    inventory: _ReleaseTreeInventory,
    error_type: type[ValueError],
) -> None:
    if (
        approval.release_id != release.manifest.release_id
        or approval.prospective_release_sha256 != release.release_sha256
        or approval.release_tree_sha256 != inventory.sha256
        or approval.challenge_bundle_sha256 != release.challenge.manifest_sha256
        or approval.episode_count != release.manifest.episode_count
        or approval.release_scope.tasks != (release.verified_admission.suite.task_type,)
    ):
        raise error_type('Tier A approval is bound to different prospective release inputs')


def _latest_prerequisite_witnessed_at(release: LoadedProspectiveCohortRelease) -> datetime:
    witness_times = [release.verified_admission.case_universe.seal.witnessed_at]
    for package in release.verified_admission.packages:
        witness_times.extend(capture.witnessed_at for capture in package.manifest.source_captures)
    for seal in release.verified_admission.seals:
        witness_times.extend(receipt.witnessed_at for receipt in seal.manifest.receipts)
    return max(_aware(value, 'release prerequisite witness') for value in witness_times)


def _require_valid_timestamp_proof(
    *,
    target: ProspectiveReleaseSealTarget,
    target_bytes: bytes,
    timestamp_proof: ProspectiveReleaseTimestampProof,
    proof_bytes: bytes,
    timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    error_type: type[ValueError],
) -> None:
    if timestamp_proof.authority_type not in _EXTERNAL_AUTHORITIES:
        raise error_type('release seal lacks an independent RFC 3161 or transparency-log witness')
    if (
        timestamp_proof.target_sha256 != _sha256(target_bytes)
        or timestamp_proof.target_bytes != len(target_bytes)
        or timestamp_proof.prospective_release_sha256 != target.prospective_release_sha256
    ):
        raise error_type('external timestamp proof does not bind the exact prospective release target')
    if timestamp_proof.proof_sha256 != _sha256(proof_bytes) or timestamp_proof.proof_bytes != len(proof_bytes):
        raise error_type('external timestamp proof bytes do not match their hash and size binding')
    if timestamp_proof.witnessed_at <= target.latest_prerequisite_witnessed_at:
        raise error_type('external release witness must follow every source and case-universe witness')
    if timestamp_proof.witnessed_at >= target.submissions_open_at:
        raise error_type('external release witness must arrive before submissions open')
    if timestamp_proof.witnessed_at >= target.run_deadline_at:
        raise error_type('external release witness did not precede the preregistered run deadline')
    try:
        accepted = timestamp_verifier(timestamp_proof, proof_bytes)
    except Exception as error:
        raise error_type(f'trusted external timestamp verifier failed: {error}') from error
    if not accepted:
        raise error_type('trusted external timestamp verifier rejected the release proof')


def _inventory_release_tree(
    root: Path,
    *,
    error_type: type[ValueError],
) -> _ReleaseTreeInventory:
    files: list[dict[str, object]] = []
    directories: set[str] = set()
    total_bytes = 0
    try:
        for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            parent = Path(directory)
            directory_names.sort()
            file_names.sort()
            for name in tuple(directory_names):
                path = parent / name
                relative = path.relative_to(root).as_posix()
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise error_type(f'prospective release tree contains an invalid directory: {relative}')
                directories.add(relative)
                if len(directories) > _MAX_RELEASE_DIRECTORIES:
                    raise error_type('prospective release tree contains too many directories')
            for name in file_names:
                path = parent / name
                relative = path.relative_to(root).as_posix()
                payload = _read_regular_file(path, _MAX_RELEASE_FILE_BYTES, error_type=error_type)
                total_bytes += len(payload)
                if total_bytes > _MAX_RELEASE_BYTES:
                    raise error_type('prospective release tree exceeds the aggregate size limit')
                files.append({'path': relative, 'sha256': _sha256(payload), 'byte_count': len(payload)})
                if len(files) > _MAX_RELEASE_FILES:
                    raise error_type('prospective release tree contains too many files')
    except ValueError:
        raise
    except OSError as error:
        raise error_type(f'cannot inventory prospective release tree: {error}') from error
    files.sort(key=lambda binding: str(binding['path']))
    digest_payload = {'directories': sorted(directories), 'files': files}
    return _ReleaseTreeInventory(
        sha256=_sha256(canonical_json_bytes(digest_payload)),
        byte_count=total_bytes,
        file_count=len(files),
        directory_count=len(directories),
    )


def _resolve_root(root: Path) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ProspectiveReleaseSealIntegrityError('prospective release-seal root cannot be a symlink')
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise ProspectiveReleaseSealIntegrityError(f'cannot resolve prospective release-seal root: {error}') from error
    if not resolved.is_dir():
        raise ProspectiveReleaseSealIntegrityError('prospective release-seal root must be a directory')
    return resolved


def _require_exact_seal_inventory(root: Path) -> None:
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ProspectiveReleaseSealIntegrityError('prospective release seal cannot contain symlinks')
                if entry.is_file(follow_symlinks=False):
                    actual_files.add(entry.name)
                elif entry.is_dir(follow_symlinks=False):
                    actual_directories.add(entry.name)
                else:
                    raise ProspectiveReleaseSealIntegrityError(
                        'prospective release seal can contain only regular files'
                    )
    except ProspectiveReleaseSealIntegrityError:
        raise
    except OSError as error:
        raise ProspectiveReleaseSealIntegrityError(f'cannot inventory prospective release seal: {error}') from error
    if actual_files != {'seal.json', 'target.json', 'timestamp-proof.bin'} or actual_directories:
        raise ProspectiveReleaseSealIntegrityError('prospective release seal exact file allowlist mismatch')


def _read_regular_file(
    path: Path,
    maximum_bytes: int,
    *,
    error_type: type[ValueError] = ProspectiveReleaseSealIntegrityError,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise error_type(f'cannot open {path.name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise error_type(f'{path.name} is not a regular file')
        if metadata.st_size > maximum_bytes:
            raise error_type(f'{path.name} exceeds its size limit')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise error_type(f'{path.name} exceeds its size limit')
    except OSError as error:
        raise error_type(f'cannot read {path.name}: {error}') from error
    finally:
        os.close(descriptor)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
