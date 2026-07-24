"""Deterministic semantic archives for prospective campaign publication.

The campaign publication layer intentionally treats its archive artifacts as opaque
bytes.  This module supplies the separate semantic bridge: it snapshots one complete
prospective cohort release, concatenates its files in sorted path order, and binds
every byte range in a canonical index.  Verification starts from out-of-band archive
and index digests, reconstructs the exact tree, and reruns the prospective release
loader with caller-supplied trust verifiers.

Successful verification establishes archive and prospective-release integrity only.
It does not establish campaign publication, readiness authorization, organizational
independence, or any other Tier A deployment fact.  In particular, this module never
accepts a serialized readiness or release-decision report as a trust capability.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import (
    AtomicDirectoryPublication,
    AtomicDirectoryPublicationError,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations._immutable_tree import ImmutableTreeError, snapshot_immutable_tree
from vaxreplay.operations.promotion_schema import PromotionHandoffDescriptor
from vaxreplay.operations.release_readiness import TierAReleaseScope
from vaxreplay.operations.schema import SAFE_ID_PATTERN
from vaxreplay.prospective_admission import CaseUniverseSealVerifier, SourceCaptureVerifier
from vaxreplay.prospective_release import (
    LoadedProspectiveCohortRelease,
    ProspectiveCohortReleaseManifest,
    load_prospective_cohort_release,
    prospective_cohort_release_sha256,
)
from vaxreplay.sources.clinicaltrials import CTGOV_SOURCE_VERIFIER_ID
from vaxreplay.sources.iedb import IEDB_SOURCE_VERIFIER_ID
from vaxreplay.sources.immport import IMMPORT_SOURCE_VERIFIER_ID
from vaxreplay.temporal_schema import TemporalReceiptVerifier

PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_SCHEMA_VERSION = 'vaxreplay.prospective-campaign-archive-index.v0.2'
PROSPECTIVE_CAMPAIGN_ARCHIVE_FORMAT = 'raw-concatenated-v0.1'
CAMPAIGN_ARCHIVE_FILE_NAME = 'release.archive'
CAMPAIGN_ARCHIVE_INDEX_FILE_NAME = 'release-index.json'

# A campaign publication artifact is currently bounded to 256 MiB.  Keep the
# semantic archive at or below that exact limit so it can be handed to the existing
# publication verifier without a second packaging layer.
MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_BYTES = 64 * 1024 * 1024
MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILE_BYTES = 256 * 1024 * 1024
MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILES = 100_000
MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_DIRECTORIES = 20_000
MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_PATH_CHARACTERS = 1_024

_SHA256_PATTERN = r'^[0-9a-f]{64}$'

type ProspectiveCampaignReleasePurpose = Literal['official_benchmark', 'prospective_research']


class ProspectiveCampaignArchiveError(ValueError):
    """The semantic archive is unsafe, malformed, changed, or unauthorized by its expected digests."""


class ProspectiveCampaignArchiveFileBinding(StrictModel):
    """One exact regular file range in the raw concatenated archive."""

    path: str = Field(min_length=1, max_length=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_PATH_CHARACTERS)
    offset: int = Field(ge=0, le=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES)
    byte_count: int = Field(ge=0, le=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILE_BYTES)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        _safe_archive_path(value)
        return value


class ProspectiveCampaignArchiveIndex(StrictModel):
    """Canonical complete-file inventory for one raw prospective release archive."""

    schema_version: Literal['vaxreplay.prospective-campaign-archive-index.v0.2'] = (
        PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_SCHEMA_VERSION
    )
    archive_format: Literal['raw-concatenated-v0.1'] = PROSPECTIVE_CAMPAIGN_ARCHIVE_FORMAT
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_purpose: ProspectiveCampaignReleasePurpose
    release_scope: TierAReleaseScope
    release_scope_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_json_path: Literal['release.json'] = 'release.json'
    release_json_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_json_bytes: int = Field(gt=0, le=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILE_BYTES)
    prospective_release_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    archive_byte_count: int = Field(gt=0, le=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES)
    tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    file_count: int = Field(gt=0, le=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILES)
    directory_count: int = Field(ge=0, le=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_DIRECTORIES)
    files: tuple[ProspectiveCampaignArchiveFileBinding, ...] = Field(
        min_length=1,
        max_length=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILES,
    )

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        if self.release_scope_sha256 != _sha256(canonical_json_bytes(self.release_scope)):
            raise ValueError('campaign archive release-scope digest differs from its canonical scope')
        paths = tuple(binding.path for binding in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('campaign archive file bindings must be sorted and unique by path')
        if self.file_count != len(self.files):
            raise ValueError('campaign archive file_count differs from its complete inventory')
        directories = _required_directories(paths)
        if self.directory_count != len(directories):
            raise ValueError('campaign archive directory_count differs from its inferred exact tree')
        _reject_file_directory_collisions(paths)

        cursor = 0
        for binding in self.files:
            if binding.offset != cursor:
                raise ValueError('campaign archive ranges must be contiguous, ordered, and non-overlapping')
            cursor += binding.byte_count
            if cursor > MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES:
                raise ValueError('campaign archive ranges exceed the aggregate byte limit')
        if cursor != self.archive_byte_count:
            raise ValueError('campaign archive ranges must cover the archive exactly without trailing bytes')

        release_binding = next(
            (binding for binding in self.files if binding.path == self.release_json_path),
            None,
        )
        if release_binding is None:
            raise ValueError('campaign archive inventory must contain release.json')
        if release_binding.sha256 != self.release_json_sha256 or release_binding.byte_count != self.release_json_bytes:
            raise ValueError('release.json binding differs from the archive index identity')
        if self.release_json_sha256 != self.prospective_release_manifest_sha256:
            raise ValueError('canonical release.json and typed prospective release identities must agree')
        if self.tree_sha256 != _tree_sha256(self.files):
            raise ValueError('campaign archive tree digest differs from its complete file inventory')
        return self


@dataclass(frozen=True)
class BuiltProspectiveCampaignArchive:
    """Deterministic in-memory archive and its exact canonical index bytes."""

    archive_bytes: bytes
    index: ProspectiveCampaignArchiveIndex
    index_bytes: bytes
    index_sha256: str


@dataclass(frozen=True)
class WrittenProspectiveCampaignArchive:
    """Atomically published create-once archive/index pair."""

    root: Path
    archive_path: Path
    index_path: Path
    index: ProspectiveCampaignArchiveIndex
    index_sha256: str


@dataclass(frozen=True)
class VerifiedProspectiveCampaignArchive:
    """Fresh semantic verification result with the fully reloaded prospective release.

    This is an in-process convenience value, not a serializable authorization token.
    Consumers requiring campaign approval must separately verify and cross-bind the
    publication/readiness decision to ``archive_sha256`` and ``index_sha256``.
    """

    archive_sha256: str
    index_sha256: str
    index: ProspectiveCampaignArchiveIndex
    release: LoadedProspectiveCohortRelease


def build_prospective_campaign_archive(
    release_root: Path,
    *,
    release_scope: TierAReleaseScope,
) -> BuiltProspectiveCampaignArchive:
    """Structurally snapshot a prospective release tree and deterministically encode it.

    The directory is traversed relative to one no-follow root descriptor.  Every
    regular file is read stably, and empty/unexpected directories, symlinks, and
    special files are rejected before any archive bytes are returned.  This low-level
    codec validates the typed canonical ``release.json`` but does not authenticate its
    external proofs.  Production publication should use
    :func:`build_verified_prospective_campaign_archive` or
    :func:`write_prospective_campaign_archive`.
    """

    _require_release_scope(release_scope)
    try:
        snapshot = snapshot_immutable_tree(
            release_root,
            max_files=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILES,
            max_directories=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_DIRECTORIES,
            max_file_bytes=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILE_BYTES,
            max_total_bytes=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES,
            max_path_characters=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_PATH_CHARACTERS,
        )
        snapshot.require_exact_files(frozenset(snapshot.files))
    except (ImmutableTreeError, OSError, ValueError) as error:
        raise ProspectiveCampaignArchiveError(f'cannot snapshot prospective release tree: {error}') from error

    release_payload = snapshot.files.get('release.json')
    if release_payload is None:
        raise ProspectiveCampaignArchiveError('prospective release tree is missing release.json')
    release_manifest = _canonical_release_manifest(release_payload)
    release_sha256 = prospective_cohort_release_sha256(release_manifest)

    archive_parts: list[bytes] = []
    bindings: list[ProspectiveCampaignArchiveFileBinding] = []
    offset = 0
    for path in sorted(snapshot.files):
        payload = snapshot.files[path]
        binding = ProspectiveCampaignArchiveFileBinding(
            path=path,
            offset=offset,
            byte_count=len(payload),
            sha256=_sha256(payload),
        )
        bindings.append(binding)
        archive_parts.append(payload)
        offset += len(payload)
    archive_bytes = b''.join(archive_parts)
    if offset != len(archive_bytes) or offset > MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES:
        raise ProspectiveCampaignArchiveError('prospective campaign archive exceeds its aggregate byte limit')

    file_bindings = tuple(bindings)
    index = ProspectiveCampaignArchiveIndex(
        release_id=release_manifest.release_id,
        release_purpose=release_manifest.purpose,
        release_scope=release_scope,
        release_scope_sha256=_sha256(canonical_json_bytes(release_scope)),
        release_json_sha256=_sha256(release_payload),
        release_json_bytes=len(release_payload),
        prospective_release_manifest_sha256=release_sha256,
        archive_sha256=_sha256(archive_bytes),
        archive_byte_count=len(archive_bytes),
        tree_sha256=_tree_sha256(file_bindings),
        file_count=len(file_bindings),
        directory_count=len(snapshot.directories),
        files=file_bindings,
    )
    index_bytes = canonical_json_bytes(index)
    if len(index_bytes) > MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_BYTES:
        raise ProspectiveCampaignArchiveError('prospective campaign archive index exceeds its byte limit')
    return BuiltProspectiveCampaignArchive(
        archive_bytes=archive_bytes,
        index=index,
        index_bytes=index_bytes,
        index_sha256=_sha256(index_bytes),
    )


def build_verified_prospective_campaign_archive(
    release_root: Path,
    *,
    expected_release_sha256: str,
    expected_release_id: str,
    expected_purpose: ProspectiveCampaignReleasePurpose,
    expected_release_scope: TierAReleaseScope,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> BuiltProspectiveCampaignArchive:
    """Freshly authenticate a release before and after deterministic archive encoding."""

    _require_expected_sha256(expected_release_sha256, 'prospective release')
    if not isinstance(expected_release_id, str) or not expected_release_id:
        raise ProspectiveCampaignArchiveError('expected release ID must be a nonempty string')
    _require_expected_purpose(expected_purpose)
    _require_release_scope(expected_release_scope)
    first = _load_source_release(
        release_root,
        expected_release_sha256=expected_release_sha256,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
    )
    if first.manifest.release_id != expected_release_id or first.manifest.purpose != expected_purpose:
        raise ProspectiveCampaignArchiveError(
            'source prospective release identifies a different expected release or purpose'
        )
    _verify_release_scope(first, expected_release_scope)
    built = build_prospective_campaign_archive(
        first.root,
        release_scope=expected_release_scope,
    )
    if (
        built.index.release_id != expected_release_id
        or built.index.release_purpose != expected_purpose
        or built.index.release_scope != expected_release_scope
        or built.index.prospective_release_manifest_sha256 != expected_release_sha256
    ):
        raise ProspectiveCampaignArchiveError('encoded archive differs from the authenticated source release')
    second = _load_source_release(
        first.root,
        expected_release_sha256=expected_release_sha256,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
    )
    _verify_release_scope(second, expected_release_scope)
    if second != first or second.manifest.purpose != expected_purpose:
        raise ProspectiveCampaignArchiveError('source prospective release changed while its archive was encoded')
    return built


def write_prospective_campaign_archive(
    release_root: Path,
    output_dir: Path,
    *,
    expected_release_sha256: str,
    expected_release_id: str,
    expected_purpose: ProspectiveCampaignReleasePurpose,
    expected_release_scope: TierAReleaseScope,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> WrittenProspectiveCampaignArchive:
    """Authenticate, then atomically publish a create-once archive/index pair."""

    built = build_verified_prospective_campaign_archive(
        release_root,
        expected_release_sha256=expected_release_sha256,
        expected_release_id=expected_release_id,
        expected_purpose=expected_purpose,
        expected_release_scope=expected_release_scope,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
    )
    try:
        with AtomicDirectoryPublication.create(output_dir) as publication:
            publication.write_bytes(CAMPAIGN_ARCHIVE_FILE_NAME, built.archive_bytes)
            publication.write_bytes(CAMPAIGN_ARCHIVE_INDEX_FILE_NAME, built.index_bytes)
            target = publication.publish()
            if (
                _read_stable_regular_file(
                    target / CAMPAIGN_ARCHIVE_FILE_NAME,
                    maximum=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES,
                    label='published prospective campaign archive',
                )
                != built.archive_bytes
                or _read_stable_regular_file(
                    target / CAMPAIGN_ARCHIVE_INDEX_FILE_NAME,
                    maximum=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_BYTES,
                    label='published prospective campaign archive index',
                )
                != built.index_bytes
            ):
                raise ProspectiveCampaignArchiveError('published prospective campaign archive changed before commit')
            publication.commit()
    except AtomicDirectoryPublicationError as error:
        raise ProspectiveCampaignArchiveError(f'prospective campaign archive publication failed: {error}') from error
    return WrittenProspectiveCampaignArchive(
        root=target,
        archive_path=target / CAMPAIGN_ARCHIVE_FILE_NAME,
        index_path=target / CAMPAIGN_ARCHIVE_INDEX_FILE_NAME,
        index=built.index,
        index_sha256=built.index_sha256,
    )


def verify_and_materialize_prospective_campaign_archive(
    *,
    archive_bytes: bytes,
    index_bytes: bytes,
    output_dir: Path,
    expected_archive_sha256: str,
    expected_index_sha256: str,
    expected_release_id: str,
    expected_purpose: ProspectiveCampaignReleasePurpose,
    expected_release_scope: TierAReleaseScope,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> VerifiedProspectiveCampaignArchive:
    """Verify exact archive bytes, materialize safely, and reload release semantics.

    The expected archive/index digests, release identity, purpose, and canonical
    release scope must come from independently authenticated inputs. Passing those
    primitive values does not cause this function to trust a serialized decision
    report or its claims.
    """

    _require_expected_sha256(expected_archive_sha256, 'archive')
    _require_expected_sha256(expected_index_sha256, 'archive index')
    if not isinstance(expected_release_id, str) or not expected_release_id:
        raise ProspectiveCampaignArchiveError('expected release ID must be a nonempty string')
    _require_expected_purpose(expected_purpose)
    _require_release_scope(expected_release_scope)
    _require_bounded_bytes(
        index_bytes,
        maximum=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_BYTES,
        label='prospective campaign archive index',
    )
    actual_index_sha256 = _sha256(index_bytes)
    if not hmac.compare_digest(actual_index_sha256, expected_index_sha256):
        raise ProspectiveCampaignArchiveError('archive index differs from its out-of-band expected digest')
    index = _canonical_index(index_bytes)
    if index.release_id != expected_release_id:
        raise ProspectiveCampaignArchiveError('archive index identifies a different expected release')
    if index.release_purpose != expected_purpose:
        raise ProspectiveCampaignArchiveError('archive index identifies a different expected release purpose')
    if index.release_scope != expected_release_scope:
        raise ProspectiveCampaignArchiveError('archive index identifies a different expected release scope')

    _require_bounded_bytes(
        archive_bytes,
        maximum=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES,
        label='prospective campaign archive',
    )
    actual_archive_sha256 = _sha256(archive_bytes)
    if not hmac.compare_digest(actual_archive_sha256, expected_archive_sha256):
        raise ProspectiveCampaignArchiveError('archive differs from its out-of-band expected digest')
    if len(archive_bytes) != index.archive_byte_count or not hmac.compare_digest(
        actual_archive_sha256, index.archive_sha256
    ):
        raise ProspectiveCampaignArchiveError('archive differs from its canonical index binding')

    release_manifest = _verify_archive_payloads(index, archive_bytes)
    if release_manifest.release_id != expected_release_id:
        raise ProspectiveCampaignArchiveError('release.json identifies a different expected release')
    if release_manifest.purpose != expected_purpose:
        raise ProspectiveCampaignArchiveError('release.json identifies a different expected release purpose')

    try:
        with AtomicDirectoryPublication.create(output_dir) as publication:
            _materialize_archive(publication, index, archive_bytes)
            _load_materialized_release(
                publication.private_tree_path(),
                index=index,
                decision_receipt_verifier=decision_receipt_verifier,
                case_universe_seal_verifier=case_universe_seal_verifier,
                source_capture_verifier=source_capture_verifier,
                expected_purpose=expected_purpose,
                expected_release_scope=expected_release_scope,
            )
            publication.require_private_tree_unchanged()
            target = publication.publish()
            release = _load_materialized_release(
                target,
                index=index,
                decision_receipt_verifier=decision_receipt_verifier,
                case_universe_seal_verifier=case_universe_seal_verifier,
                source_capture_verifier=source_capture_verifier,
                expected_purpose=expected_purpose,
                expected_release_scope=expected_release_scope,
            )
            publication.commit()
    except AtomicDirectoryPublicationError as error:
        raise ProspectiveCampaignArchiveError(
            f'materialized prospective release publication failed: {error}'
        ) from error

    return VerifiedProspectiveCampaignArchive(
        archive_sha256=actual_archive_sha256,
        index_sha256=actual_index_sha256,
        index=index,
        release=release,
    )


def verify_and_materialize_prospective_campaign_archive_files(
    *,
    archive_path: Path,
    index_path: Path,
    output_dir: Path,
    expected_archive_sha256: str,
    expected_index_sha256: str,
    expected_release_id: str,
    expected_purpose: ProspectiveCampaignReleasePurpose,
    expected_release_scope: TierAReleaseScope,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> VerifiedProspectiveCampaignArchive:
    """Stably read no-follow files, then invoke the exact-byte semantic verifier."""

    index_bytes = _read_stable_regular_file(
        index_path,
        maximum=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_BYTES,
        label='prospective campaign archive index',
    )
    archive_bytes = _read_stable_regular_file(
        archive_path,
        maximum=MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES,
        label='prospective campaign archive',
    )
    return verify_and_materialize_prospective_campaign_archive(
        archive_bytes=archive_bytes,
        index_bytes=index_bytes,
        output_dir=output_dir,
        expected_archive_sha256=expected_archive_sha256,
        expected_index_sha256=expected_index_sha256,
        expected_release_id=expected_release_id,
        expected_purpose=expected_purpose,
        expected_release_scope=expected_release_scope,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
    )


def _canonical_index(payload: bytes) -> ProspectiveCampaignArchiveIndex:
    try:
        index = ProspectiveCampaignArchiveIndex.model_validate_json(payload)
    except ValueError as error:
        raise ProspectiveCampaignArchiveError(f'prospective campaign archive index is invalid: {error}') from error
    if payload != canonical_json_bytes(index):
        raise ProspectiveCampaignArchiveError('prospective campaign archive index must use canonical JSON')
    return index


def _canonical_release_manifest(payload: bytes) -> ProspectiveCohortReleaseManifest:
    try:
        manifest = ProspectiveCohortReleaseManifest.model_validate_json(payload)
    except ValueError as error:
        raise ProspectiveCampaignArchiveError(f'archive release.json is invalid: {error}') from error
    if payload != canonical_json_bytes(manifest):
        raise ProspectiveCampaignArchiveError('archive release.json must use canonical JSON')
    return manifest


def _verify_archive_payloads(
    index: ProspectiveCampaignArchiveIndex,
    archive_bytes: bytes,
) -> ProspectiveCohortReleaseManifest:
    view = memoryview(archive_bytes)
    release_payload: bytes | None = None
    for binding in index.files:
        end = binding.offset + binding.byte_count
        payload = view[binding.offset : end]
        if len(payload) != binding.byte_count or not hmac.compare_digest(
            hashlib.sha256(payload).hexdigest(),
            binding.sha256,
        ):
            raise ProspectiveCampaignArchiveError(
                f'archive file range differs from its exact digest binding: {binding.path}'
            )
        if binding.path == index.release_json_path:
            release_payload = bytes(payload)
    if release_payload is None:
        raise ProspectiveCampaignArchiveError('archive does not contain its bound release.json')
    if len(release_payload) != index.release_json_bytes or not hmac.compare_digest(
        _sha256(release_payload), index.release_json_sha256
    ):
        raise ProspectiveCampaignArchiveError('archive release.json differs from its index binding')
    manifest = _canonical_release_manifest(release_payload)
    typed_sha256 = prospective_cohort_release_sha256(manifest)
    if not hmac.compare_digest(typed_sha256, index.prospective_release_manifest_sha256):
        raise ProspectiveCampaignArchiveError('typed prospective release manifest differs from its index binding')
    if manifest.release_id != index.release_id:
        raise ProspectiveCampaignArchiveError('archive index and release.json identify different releases')
    if manifest.purpose != index.release_purpose:
        raise ProspectiveCampaignArchiveError('archive index and release.json identify different release purposes')
    return manifest


def _load_materialized_release(
    root: Path,
    *,
    index: ProspectiveCampaignArchiveIndex,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_purpose: ProspectiveCampaignReleasePurpose,
    expected_release_scope: TierAReleaseScope,
) -> LoadedProspectiveCohortRelease:
    try:
        loaded = load_prospective_cohort_release(
            root,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_release_sha256=index.prospective_release_manifest_sha256,
        )
    except ValueError as error:
        raise ProspectiveCampaignArchiveError(
            f'materialized archive is not a valid prospective cohort release: {error}'
        ) from error
    if loaded.manifest.release_id != index.release_id or loaded.manifest.purpose != expected_purpose:
        raise ProspectiveCampaignArchiveError(
            'loaded prospective release identifies a different archive release or purpose'
        )
    _verify_release_scope(loaded, expected_release_scope)
    return loaded


def _load_source_release(
    root: Path,
    *,
    expected_release_sha256: str,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> LoadedProspectiveCohortRelease:
    try:
        return load_prospective_cohort_release(
            root,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_release_sha256=expected_release_sha256,
        )
    except ValueError as error:
        raise ProspectiveCampaignArchiveError(
            f'source tree is not an authenticated prospective cohort release: {error}'
        ) from error


def _materialize_archive(
    publication: AtomicDirectoryPublication,
    index: ProspectiveCampaignArchiveIndex,
    archive_bytes: bytes,
) -> None:
    directories = _required_directories(tuple(binding.path for binding in index.files))
    for relative in sorted(directories, key=lambda value: (len(PurePosixPath(value).parts), value)):
        try:
            publication.make_directory(relative)
        except AtomicDirectoryPublicationError as error:
            raise ProspectiveCampaignArchiveError(f'cannot create archive directory safely: {relative}') from error

    view = memoryview(archive_bytes)
    for binding in index.files:
        publication.write_bytes(
            binding.path,
            view[binding.offset : binding.offset + binding.byte_count],
        )


def _read_stable_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as error:
        raise ProspectiveCampaignArchiveError(f'cannot open {label} without following links') from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum:
            raise ProspectiveCampaignArchiveError(f'{label} must be a nonempty bounded regular file')
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
            except OSError as error:
                raise ProspectiveCampaignArchiveError(f'cannot read {label}') from error
            if not chunk:
                raise ProspectiveCampaignArchiveError(f'{label} changed while being read')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProspectiveCampaignArchiveError(f'{label} changed while being read')
        after = os.fstat(descriptor)
        try:
            current = os.stat(requested, follow_symlinks=False)
        except OSError as error:
            raise ProspectiveCampaignArchiveError(f'{label} path changed while being read') from error
        if _stat_identity(before) != _stat_identity(after) or _stat_identity(before) != _stat_identity(current):
            raise ProspectiveCampaignArchiveError(f'{label} changed while being read')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


def _safe_archive_path(value: str) -> PurePosixPath:
    if '\x00' in value or '\\' in value or len(value) > MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_PATH_CHARACTERS:
        raise ValueError('campaign archive file path is unsafe')
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in {'', '.', '..'} for part in path.parts)
    ):
        raise ValueError('campaign archive file path is unsafe')
    return path


def _required_directories(paths: tuple[str, ...]) -> tuple[str, ...]:
    directories: set[str] = set()
    for value in paths:
        path = _safe_archive_path(value)
        for parent in path.parents:
            if parent != PurePosixPath('.'):
                directories.add(parent.as_posix())
    if len(directories) > MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_DIRECTORIES:
        raise ValueError('campaign archive directory count exceeds its configured bound')
    return tuple(sorted(directories))


def _reject_file_directory_collisions(paths: tuple[str, ...]) -> None:
    file_paths = set(paths)
    for value in paths:
        for parent in PurePosixPath(value).parents:
            if parent.as_posix() in file_paths:
                raise ValueError('campaign archive path cannot be both a file and a directory')


def _tree_sha256(files: tuple[ProspectiveCampaignArchiveFileBinding, ...]) -> str:
    paths = tuple(binding.path for binding in files)
    payload = {
        'directories': list(_required_directories(paths)),
        'files': [
            {
                'path': binding.path,
                'sha256': binding.sha256,
                'byte_count': binding.byte_count,
            }
            for binding in files
        ],
    }
    return _sha256(canonical_json_bytes(payload))


def _require_expected_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise ProspectiveCampaignArchiveError(f'expected {label} digest must be lowercase SHA-256')


def _require_expected_purpose(value: object) -> None:
    if value not in {'official_benchmark', 'prospective_research'}:
        raise ProspectiveCampaignArchiveError('expected release purpose is invalid')


def _require_release_scope(value: object) -> None:
    if not isinstance(value, TierAReleaseScope):
        raise ProspectiveCampaignArchiveError('expected release scope must use the strict canonical scope model')


def _verify_release_scope(
    release: LoadedProspectiveCohortRelease,
    expected: TierAReleaseScope,
) -> None:
    """Bind scope to the suite task and exact promotion source-verifier union."""

    task = release.verified_admission.suite.task_type
    if expected.tasks != (task,):
        raise ProspectiveCampaignArchiveError(
            'campaign archive release scope task differs from the authenticated prospective suite'
        )
    if release.manifest.purpose == 'official_benchmark':
        actual_sources = _derive_official_release_sources(release)
        if expected.sources != actual_sources:
            raise ProspectiveCampaignArchiveError(
                'campaign archive release scope sources differ from the authenticated promotion source union'
            )


def _derive_official_release_sources(
    release: LoadedProspectiveCohortRelease,
) -> tuple[str, ...]:
    """Derive readiness source categories from canonical promotion handoffs.

    ``promotion:*`` is only a container identity.  The source-specific readiness
    gates are instead selected from the exact production verifier identity and its
    matching source namespace inside each fully reverified capture index. Unknown
    verifiers or namespace/identity mismatches fail closed.
    """

    if (
        release.manifest.purpose != 'official_benchmark'
        or release.verified_admission.admission.purpose != 'official_benchmark'
    ):
        raise ProspectiveCampaignArchiveError('official release source derivation requires official admission')
    categories: set[str] = set()
    promotion_ids: set[str] = set()
    for package in release.verified_admission.packages:
        for binding in package.manifest.source_captures:
            payload = package.source_capture_artifacts.get(binding.source_id)
            if not binding.source_id.startswith('promotion:') or type(payload) is not bytes:
                raise ProspectiveCampaignArchiveError(
                    'official release source scope requires canonical promotion handoff artifacts'
                )
            try:
                descriptor = PromotionHandoffDescriptor.model_validate_json(payload)
            except ValueError as error:
                raise ProspectiveCampaignArchiveError(
                    'official release source scope contains an invalid promotion handoff'
                ) from error
            if payload != canonical_json_bytes(descriptor):
                raise ProspectiveCampaignArchiveError(
                    'official release source scope requires canonical promotion handoff JSON'
                )
            if binding.source_id != f'promotion:{descriptor.promotion_id}' or descriptor.promotion_id in promotion_ids:
                raise ProspectiveCampaignArchiveError(
                    'official release source scope contains duplicate or mismatched promotion identity'
                )
            promotion_ids.add(descriptor.promotion_id)
            for verification in descriptor.capture_index.source_verifications:
                categories.add(
                    _readiness_source_for_verifier(
                        source_id=verification.source_id,
                        verifier_id=verification.result.verifier.verifier_id,
                    )
                )
    if not categories:
        raise ProspectiveCampaignArchiveError('official release source scope has no verified production sources')
    return tuple(sorted(categories))


def _readiness_source_for_verifier(*, source_id: str, verifier_id: str) -> str:
    profiles = {
        IEDB_SOURCE_VERIFIER_ID: ('iedb', 'iedb:'),
        IMMPORT_SOURCE_VERIFIER_ID: ('immport', 'immport:'),
        CTGOV_SOURCE_VERIFIER_ID: ('clinicaltrials.gov', 'clinicaltrials-gov:'),
    }
    profile = profiles.get(verifier_id)
    if profile is None:
        raise ProspectiveCampaignArchiveError(
            f'official release uses an unknown production source verifier: {verifier_id!r}'
        )
    readiness_source, namespace = profile
    if not source_id.startswith(namespace) or len(source_id) <= len(namespace):
        raise ProspectiveCampaignArchiveError(
            'official release source verifier identity differs from its source namespace'
        )
    return readiness_source


def _require_bounded_bytes(value: bytes, *, maximum: int, label: str) -> None:
    if not isinstance(value, bytes) or not value or len(value) > maximum:
        raise ProspectiveCampaignArchiveError(f'{label} must be nonempty bounded exact bytes')


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    'CAMPAIGN_ARCHIVE_FILE_NAME',
    'CAMPAIGN_ARCHIVE_INDEX_FILE_NAME',
    'MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES',
    'MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_DIRECTORIES',
    'MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILES',
    'MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_FILE_BYTES',
    'MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_BYTES',
    'MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_PATH_CHARACTERS',
    'PROSPECTIVE_CAMPAIGN_ARCHIVE_FORMAT',
    'PROSPECTIVE_CAMPAIGN_ARCHIVE_INDEX_SCHEMA_VERSION',
    'BuiltProspectiveCampaignArchive',
    'ProspectiveCampaignArchiveError',
    'ProspectiveCampaignArchiveFileBinding',
    'ProspectiveCampaignArchiveIndex',
    'ProspectiveCampaignReleasePurpose',
    'VerifiedProspectiveCampaignArchive',
    'WrittenProspectiveCampaignArchive',
    'build_prospective_campaign_archive',
    'build_verified_prospective_campaign_archive',
    'verify_and_materialize_prospective_campaign_archive',
    'verify_and_materialize_prospective_campaign_archive_files',
    'write_prospective_campaign_archive',
]
