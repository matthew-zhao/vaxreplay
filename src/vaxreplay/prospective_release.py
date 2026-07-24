"""Atomic, outcome-free packaging for an admitted prospective challenge cohort.

This release is the artifact that can be published and executed before any benchmark endpoint
matures.  Its identity is made entirely from decision-time packages, independent timestamp
proofs, the complete split and case inventories, and the policies that produced those
inventories.  Later scoring material belongs in a separate finalization artifact.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_inventory import CaseUniverseManifest, case_universe_sha256
from vaxreplay.case_schema import StrictModel
from vaxreplay.prompt import PromptVariant
from vaxreplay.prospective_admission import (
    CaseUniverseSealVerifier,
    SourceCaptureVerifier,
    VerifiedProspectiveAdmission,
    build_verified_prospective_admission,
)
from vaxreplay.prospective_schema import (
    ProspectiveAttemptPolicy,
    ProspectiveSplitInventory,
    prospective_challenge_admission_sha256,
    prospective_split_inventory_sha256,
    prospective_suite_manifest_sha256,
)
from vaxreplay.runner.prospective_challenge import (
    LoadedProspectiveChallengeBundle,
    build_prospective_challenge_bundle,
    load_prospective_challenge_bundle,
)
from vaxreplay.temporal_schema import TemporalReceiptVerifier

PROSPECTIVE_RELEASE_SCHEMA_VERSION = 'vaxreplay.prospective-cohort-release.v0.2'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_RELEASE_FILE_BYTES = 512 * 1024 * 1024
_MAX_RELEASE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_RELEASE_FILES = 100_000
_MAX_RELEASE_DIRECTORIES = 20_000

type ProspectiveReleaseArtifactPath = Literal[
    'split-inventory.json',
    'case-universe.json',
    'case-universe-proof.bin',
    'eligibility-protocol.bin',
    'verifier-policy.bin',
    'source-capture-policy.bin',
    'attempt-policy.bin',
]


class ProspectiveReleaseIntegrityError(ValueError):
    """Raised when a pre-outcome cohort release is incomplete or has changed."""


class ProspectiveReleaseFileBinding(StrictModel):
    """Exact bytes for one non-challenge release artifact."""

    path: ProspectiveReleaseArtifactPath
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)


class ProspectiveReleaseChallengeBinding(StrictModel):
    """Transitive commitment to the complete, independently verified challenge tree."""

    path: Literal['challenge'] = 'challenge'
    challenge_id: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)
    file_count: int = Field(gt=0)
    directory_count: int = Field(ge=0)


class ProspectiveCohortReleaseManifest(StrictModel):
    """Public pre-outcome release identity, intentionally unable to bind scoring data."""

    schema_version: Literal['vaxreplay.prospective-cohort-release.v0.2'] = PROSPECTIVE_RELEASE_SCHEMA_VERSION
    release_id: str = Field(min_length=1)
    purpose: Literal['official_benchmark', 'prospective_research']
    challenge: ProspectiveReleaseChallengeBinding
    suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_count: int = Field(gt=0)
    files: tuple[ProspectiveReleaseFileBinding, ...] = Field(min_length=7, max_length=7)

    @field_validator('files')
    @classmethod
    def validate_files(
        cls,
        value: tuple[ProspectiveReleaseFileBinding, ...],
    ) -> tuple[ProspectiveReleaseFileBinding, ...]:
        paths = tuple(binding.path for binding in value)
        if paths != tuple(sorted(paths)):
            raise ValueError('prospective release file bindings must be sorted by path')
        expected = {
            'split-inventory.json',
            'case-universe.json',
            'case-universe-proof.bin',
            'eligibility-protocol.bin',
            'verifier-policy.bin',
            'source-capture-policy.bin',
            'attempt-policy.bin',
        }
        if set(paths) != expected or len(paths) != len(set(paths)):
            raise ValueError('prospective release must bind exactly the seven cohort artifacts')
        return value

    @model_validator(mode='after')
    def validate_identity(self) -> Self:
        if self.episode_count > self.challenge.file_count:
            raise ValueError('episode count cannot exceed challenge file count')
        return self


@dataclass(frozen=True)
class LoadedProspectiveCohortRelease:
    root: Path
    manifest: ProspectiveCohortReleaseManifest
    release_sha256: str
    challenge: LoadedProspectiveChallengeBundle
    verified_admission: VerifiedProspectiveAdmission
    eligibility_protocol: bytes
    verifier_policy: bytes
    source_capture_policy: bytes
    attempt_policy: bytes
    case_universe_proof: bytes


@dataclass(frozen=True)
class _TreeInventory:
    files: frozenset[str]
    directories: frozenset[str]
    sha256: str
    byte_count: int


def prospective_cohort_release_sha256(manifest: ProspectiveCohortReleaseManifest) -> str:
    """Return the canonical identity of one pre-outcome cohort release."""

    return _sha256(canonical_json_bytes(manifest))


def build_prospective_cohort_release(
    output_dir: Path,
    *,
    challenge_id: str,
    verified_admission: VerifiedProspectiveAdmission,
    case_universe_proof: bytes,
    eligibility_protocol: bytes,
    verifier_policy: bytes,
    source_capture_policy: bytes,
    attempt_policy: bytes,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    sample_index: int = 0,
    prompt_variant: PromptVariant = PromptVariant.FULL,
) -> LoadedProspectiveCohortRelease:
    """Atomically package an already admitted cohort and reauthenticate every supplied byte.

    The trusted verifiers are required even though ``verified_admission`` was previously built by
    the organizer.  The dataclass is deliberately not treated as an unforgeable capability.
    """

    policies = _normalize_materials(
        case_universe_proof=case_universe_proof,
        eligibility_protocol=eligibility_protocol,
        verifier_policy=verifier_policy,
        source_capture_policy=source_capture_policy,
        attempt_policy=attempt_policy,
    )
    reverified = _rebuild_admission(
        verified_admission,
        case_universe_proof=policies['case-universe-proof.bin'],
        eligibility_protocol=policies['eligibility-protocol.bin'],
        verifier_policy=policies['verifier-policy.bin'],
        source_capture_policy=policies['source-capture-policy.bin'],
        attempt_policy=policies['attempt-policy.bin'],
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
    )
    if reverified != verified_admission:
        raise ValueError('verified prospective admission does not match its reauthenticated inputs')

    target = output_dir.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise ValueError(f'prospective release output already exists: {target}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    installed = False
    try:
        challenge = build_prospective_challenge_bundle(
            staging / 'challenge',
            challenge_id=challenge_id,
            suite_id=reverified.suite.suite_id,
            packages=reverified.packages,
            seals=reverified.seals,
            admission=reverified.admission,
            sample_index=sample_index,
            prompt_variant=prompt_variant,
        )
        split_bytes = canonical_json_bytes(reverified.split_inventory)
        universe_bytes = canonical_json_bytes(reverified.case_universe)
        release_files: dict[ProspectiveReleaseArtifactPath, bytes] = {
            'split-inventory.json': split_bytes,
            'case-universe.json': universe_bytes,
            **policies,
        }
        for path, payload in release_files.items():
            (staging / path).write_bytes(payload)

        challenge_inventory = _inventory_tree(staging / 'challenge')
        file_bindings = tuple(
            ProspectiveReleaseFileBinding(
                path=path,
                sha256=_sha256(payload),
                byte_count=len(payload),
            )
            for path, payload in sorted(release_files.items())
        )
        manifest = ProspectiveCohortReleaseManifest(
            release_id=reverified.admission.release_id,
            purpose=reverified.admission.purpose,
            challenge=ProspectiveReleaseChallengeBinding(
                challenge_id=challenge.manifest.challenge_id,
                bundle_sha256=challenge.manifest_sha256,
                tree_sha256=challenge_inventory.sha256,
                byte_count=challenge_inventory.byte_count,
                file_count=len(challenge_inventory.files),
                directory_count=len(challenge_inventory.directories),
            ),
            suite_sha256=prospective_suite_manifest_sha256(reverified.suite),
            admission_sha256=prospective_challenge_admission_sha256(reverified.admission),
            episode_count=len(reverified.suite.episodes),
            files=file_bindings,
        )
        (staging / 'release.json').write_bytes(canonical_json_bytes(manifest))
        _normalize_permissions(staging)
        # Verify the complete staging inventory before making the directory visible.
        _load_prospective_cohort_release(
            staging,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
        )
        os.replace(staging, target)
        installed = True
        return _load_prospective_cohort_release(
            target,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if installed:
            shutil.rmtree(target, ignore_errors=True)
        raise


def load_prospective_cohort_release(
    root: Path,
    *,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_release_sha256: str | None = None,
) -> LoadedProspectiveCohortRelease:
    """Load and fully reverify a packaged cohort, including both external proof families."""

    loaded = _load_prospective_cohort_release(
        root,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
    )
    if expected_release_sha256 is not None and loaded.release_sha256 != expected_release_sha256:
        raise ProspectiveReleaseIntegrityError('prospective release does not match the expected release identity')
    return loaded


def _load_prospective_cohort_release(
    root: Path,
    *,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> LoadedProspectiveCohortRelease:
    resolved = _resolve_root(root)
    release_inventory = _inventory_tree(resolved)
    manifest_bytes = _read_regular_file(resolved / 'release.json', _MAX_MANIFEST_BYTES)
    try:
        manifest = ProspectiveCohortReleaseManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise ProspectiveReleaseIntegrityError(f'invalid prospective release manifest: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ProspectiveReleaseIntegrityError('prospective release manifest must use canonical JSON encoding')

    challenge_root = resolved / manifest.challenge.path
    try:
        challenge = load_prospective_challenge_bundle(
            challenge_root,
            receipt_verifier=decision_receipt_verifier,
        )
    except ValueError as error:
        raise ProspectiveReleaseIntegrityError(f'invalid prospective challenge: {error}') from error
    challenge_inventory = _inventory_tree(challenge_root)
    if (
        manifest.challenge.challenge_id != challenge.manifest.challenge_id
        or manifest.challenge.bundle_sha256 != challenge.manifest_sha256
        or manifest.challenge.tree_sha256 != challenge_inventory.sha256
        or manifest.challenge.byte_count != challenge_inventory.byte_count
        or manifest.challenge.file_count != len(challenge_inventory.files)
        or manifest.challenge.directory_count != len(challenge_inventory.directories)
    ):
        raise ProspectiveReleaseIntegrityError('prospective challenge tree does not match its release binding')
    if not challenge.authority_proofs_reverified:
        raise ProspectiveReleaseIntegrityError('prospective challenge decision proofs were not reverified')

    binding_by_path = {binding.path: binding for binding in manifest.files}
    loaded_bytes: dict[str, bytes] = {}
    for path, binding in binding_by_path.items():
        payload = _read_regular_file(resolved / path, _MAX_RELEASE_FILE_BYTES)
        if len(payload) != binding.byte_count or _sha256(payload) != binding.sha256:
            raise ProspectiveReleaseIntegrityError(f'prospective release artifact changed: {path}')
        loaded_bytes[path] = payload

    split_inventory = _canonical_model(
        loaded_bytes['split-inventory.json'],
        ProspectiveSplitInventory,
        'prospective split inventory',
    )
    case_universe = _canonical_model(
        loaded_bytes['case-universe.json'],
        CaseUniverseManifest,
        'case universe',
    )
    if prospective_split_inventory_sha256(split_inventory) != challenge.admission.split_inventory_sha256:
        raise ProspectiveReleaseIntegrityError('split inventory does not match the prospective admission')
    if case_universe_sha256(case_universe) != challenge.admission.case_universe_sha256:
        raise ProspectiveReleaseIntegrityError('case universe does not match the prospective admission')
    if manifest.release_id != challenge.admission.release_id:
        raise ProspectiveReleaseIntegrityError('release ID does not match the prospective admission')
    if manifest.purpose != challenge.admission.purpose:
        raise ProspectiveReleaseIntegrityError('release purpose does not match the prospective admission')
    if manifest.suite_sha256 != prospective_suite_manifest_sha256(challenge.suite):
        raise ProspectiveReleaseIntegrityError('suite identity does not match the prospective release')
    if manifest.admission_sha256 != prospective_challenge_admission_sha256(challenge.admission):
        raise ProspectiveReleaseIntegrityError('admission identity does not match the prospective release')
    if manifest.episode_count != len(challenge.suite.episodes):
        raise ProspectiveReleaseIntegrityError('release episode count does not match the challenge')

    expected_files = {'release.json', *binding_by_path, *(f'challenge/{path}' for path in challenge_inventory.files)}
    expected_directories = {'challenge', *(f'challenge/{path}' for path in challenge_inventory.directories)}
    if release_inventory.files != expected_files:
        missing = sorted(expected_files - release_inventory.files)
        extra = sorted(release_inventory.files - expected_files)
        raise ProspectiveReleaseIntegrityError(
            f'prospective release file allowlist mismatch; missing={missing}, extra={extra}'
        )
    if release_inventory.directories != expected_directories:
        missing = sorted(expected_directories - release_inventory.directories)
        extra = sorted(release_inventory.directories - expected_directories)
        raise ProspectiveReleaseIntegrityError(
            f'prospective release directory allowlist mismatch; missing={missing}, extra={extra}'
        )

    try:
        rebuilt = build_verified_prospective_admission(
            release_id=manifest.release_id,
            suite_id=challenge.suite.suite_id,
            packages=challenge.packages,
            seals=challenge.seals,
            split_inventory=split_inventory,
            case_universe=case_universe,
            case_universe_proof=loaded_bytes['case-universe-proof.bin'],
            eligibility_protocol=loaded_bytes['eligibility-protocol.bin'],
            verifier_policy=loaded_bytes['verifier-policy.bin'],
            source_capture_policy=loaded_bytes['source-capture-policy.bin'],
            attempt_policy=loaded_bytes['attempt-policy.bin'],
            run_deadline_at=challenge.admission.run_deadline_at,
            receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
        )
    except ValueError as error:
        raise ProspectiveReleaseIntegrityError(f'prospective admission reconstruction failed: {error}') from error
    if (
        rebuilt.admission != challenge.admission
        or rebuilt.suite != challenge.suite
        or rebuilt.split_inventory != split_inventory
        or rebuilt.case_universe != case_universe
        or rebuilt.packages != challenge.packages
        or rebuilt.seals != challenge.seals
    ):
        raise ProspectiveReleaseIntegrityError('reconstructed prospective admission changed a release binding')

    return LoadedProspectiveCohortRelease(
        root=resolved,
        manifest=manifest,
        release_sha256=prospective_cohort_release_sha256(manifest),
        challenge=challenge,
        verified_admission=rebuilt,
        eligibility_protocol=loaded_bytes['eligibility-protocol.bin'],
        verifier_policy=loaded_bytes['verifier-policy.bin'],
        source_capture_policy=loaded_bytes['source-capture-policy.bin'],
        attempt_policy=loaded_bytes['attempt-policy.bin'],
        case_universe_proof=loaded_bytes['case-universe-proof.bin'],
    )


def _rebuild_admission(
    supplied: VerifiedProspectiveAdmission,
    *,
    case_universe_proof: bytes,
    eligibility_protocol: bytes,
    verifier_policy: bytes,
    source_capture_policy: bytes,
    attempt_policy: bytes,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> VerifiedProspectiveAdmission:
    try:
        return build_verified_prospective_admission(
            release_id=supplied.admission.release_id,
            suite_id=supplied.suite.suite_id,
            packages=supplied.packages,
            seals=supplied.seals,
            split_inventory=supplied.split_inventory,
            case_universe=supplied.case_universe,
            case_universe_proof=case_universe_proof,
            eligibility_protocol=eligibility_protocol,
            verifier_policy=verifier_policy,
            source_capture_policy=source_capture_policy,
            attempt_policy=attempt_policy,
            run_deadline_at=supplied.admission.run_deadline_at,
            receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
        )
    except ValueError as error:
        raise ValueError(f'prospective admission reauthentication failed: {error}') from error


def _normalize_materials(
    *,
    case_universe_proof: bytes,
    eligibility_protocol: bytes,
    verifier_policy: bytes,
    source_capture_policy: bytes,
    attempt_policy: bytes,
) -> dict[ProspectiveReleaseArtifactPath, bytes]:
    values: dict[ProspectiveReleaseArtifactPath, bytes] = {
        'case-universe-proof.bin': case_universe_proof,
        'eligibility-protocol.bin': eligibility_protocol,
        'verifier-policy.bin': verifier_policy,
        'source-capture-policy.bin': source_capture_policy,
        'attempt-policy.bin': attempt_policy,
    }
    for name, value in values.items():
        if not isinstance(value, bytes) or not value:
            raise TypeError(f'{name} must be non-empty bytes')
        if len(value) > _MAX_RELEASE_FILE_BYTES:
            raise ValueError(f'{name} exceeds the release artifact size limit')
    try:
        parsed_attempt_policy = ProspectiveAttemptPolicy.model_validate_json(values['attempt-policy.bin'])
    except ValueError as error:
        raise ValueError(f'attempt-policy.bin is not the registered Tier A attempt policy: {error}') from error
    if values['attempt-policy.bin'] != canonical_json_bytes(parsed_attempt_policy):
        raise ValueError('attempt-policy.bin must use canonical JSON encoding')
    return values


def _canonical_model(payload: bytes, model: type[StrictModel], label: str):
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise ProspectiveReleaseIntegrityError(f'invalid {label}: {error}') from error
    if payload != canonical_json_bytes(value):
        raise ProspectiveReleaseIntegrityError(f'{label} must use canonical JSON encoding')
    return value


def _resolve_root(root: Path) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ProspectiveReleaseIntegrityError('prospective release root cannot be a symlink')
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise ProspectiveReleaseIntegrityError(f'cannot resolve prospective release root: {error}') from error
    if not resolved.is_dir():
        raise ProspectiveReleaseIntegrityError('prospective release root must be a directory')
    return resolved


def _inventory_tree(root: Path) -> _TreeInventory:
    files: set[str] = set()
    directories: set[str] = set()
    file_entries: list[dict[str, object]] = []
    total_bytes = 0
    try:
        for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            parent = Path(directory)
            directory_names.sort()
            file_names.sort()
            for name in tuple(directory_names):
                path = parent / name
                relative = path.relative_to(root).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise ProspectiveReleaseIntegrityError(
                        f'prospective release cannot contain symlink directories: {relative}'
                    )
                if not stat.S_ISDIR(info.st_mode):
                    raise ProspectiveReleaseIntegrityError(
                        f'prospective release contains a non-directory tree entry: {relative}'
                    )
                directories.add(relative)
            for name in file_names:
                path = parent / name
                relative = path.relative_to(root).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise ProspectiveReleaseIntegrityError(
                        f'prospective release cannot contain symlink files: {relative}'
                    )
                if not stat.S_ISREG(info.st_mode):
                    raise ProspectiveReleaseIntegrityError(
                        f'prospective release contains a non-regular artifact: {relative}'
                    )
                if info.st_size > _MAX_RELEASE_FILE_BYTES:
                    raise ProspectiveReleaseIntegrityError(f'prospective release artifact is too large: {relative}')
                payload = _read_regular_file(path, _MAX_RELEASE_FILE_BYTES)
                total_bytes += len(payload)
                if total_bytes > _MAX_RELEASE_BYTES:
                    raise ProspectiveReleaseIntegrityError('prospective release exceeds the aggregate size limit')
                files.add(relative)
                file_entries.append({'path': relative, 'sha256': _sha256(payload), 'byte_count': len(payload)})
                if len(files) > _MAX_RELEASE_FILES:
                    raise ProspectiveReleaseIntegrityError('prospective release contains too many files')
            if len(directories) > _MAX_RELEASE_DIRECTORIES:
                raise ProspectiveReleaseIntegrityError('prospective release contains too many directories')
    except ProspectiveReleaseIntegrityError:
        raise
    except OSError as error:
        raise ProspectiveReleaseIntegrityError(f'cannot inventory prospective release: {error}') from error
    digest_payload = {
        'directories': sorted(directories),
        'files': sorted(file_entries, key=lambda entry: str(entry['path'])),
    }
    return _TreeInventory(
        files=frozenset(files),
        directories=frozenset(directories),
        sha256=_sha256(canonical_json_bytes(digest_payload)),
        byte_count=total_bytes,
    )


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProspectiveReleaseIntegrityError(
            f'cannot open prospective release artifact {path.name}: {error}'
        ) from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ProspectiveReleaseIntegrityError(f'prospective release artifact must be a regular file: {path.name}')
        if info.st_size > maximum_bytes:
            raise ProspectiveReleaseIntegrityError(f'prospective release artifact is too large: {path.name}')
        payload = bytearray()
        while True:
            remaining = maximum_bytes - len(payload)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise ProspectiveReleaseIntegrityError(f'prospective release artifact is too large: {path.name}')
    except OSError as error:
        raise ProspectiveReleaseIntegrityError(
            f'cannot read prospective release artifact {path.name}: {error}'
        ) from error
    finally:
        os.close(descriptor)


def _normalize_permissions(root: Path) -> None:
    for path in root.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
