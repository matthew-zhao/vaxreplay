"""Build and verify public-only, hash-bound challenge bundles."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from vaxreplay.aggregation import SuiteManifest, make_suite_manifest, suite_manifest_sha256
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes, resolve_episode_root
from vaxreplay.case_schema import LabelCommitmentScheme, Split
from vaxreplay.prompt import PromptVariant, build_episode_prompt, build_system_prompt
from vaxreplay.release_schema import ChallengeAdmissionCommitment
from vaxreplay.runner.schema import (
    ChallengeBundleManifest,
    ChallengeEnvelope,
    ChallengeEnvelopeFile,
    ChatMessage,
)

_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_ENVELOPE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_ENVELOPE_BYTES = 512 * 1024 * 1024


class ChallengeIntegrityError(ValueError):
    """Raised when a public challenge artifact is incomplete, noncanonical, or tampered with."""


@dataclass(frozen=True)
class LoadedChallengeBundle:
    root: Path
    manifest: ChallengeBundleManifest
    suite: SuiteManifest
    admission: ChallengeAdmissionCommitment | None
    envelopes: tuple[ChallengeEnvelope, ...]
    manifest_sha256: str


def challenge_envelope_sha256(envelope: ChallengeEnvelope) -> str:
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def challenge_bundle_sha256(manifest: ChallengeBundleManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def challenge_admission_sha256(admission: ChallengeAdmissionCommitment) -> str:
    return hashlib.sha256(canonical_json_bytes(admission)).hexdigest()


def build_challenge_bundle(
    output_dir: Path,
    *,
    challenge_id: str,
    suite_id: str,
    episode_dirs: Iterable[Path],
    sample_index: int = 0,
    prompt_variant: PromptVariant = PromptVariant.FULL,
    admission: ChallengeAdmissionCommitment | None = None,
) -> LoadedChallengeBundle:
    """Create a deterministic artifact containing only rendered public messages.

    Source episode directories are never copied or mounted. This matters because their complete
    evidence files may contain post-cutoff rows and their neighboring ``private/`` directories may
    contain outcomes and HMAC keys.
    """

    roots = tuple(resolve_episode_root(path) for path in episode_dirs)
    bundles = tuple(EpisodeBundle.load(root) for root in roots)
    if not bundles:
        raise ValueError('cannot create a challenge from zero episodes')
    for bundle in bundles:
        if (
            bundle.manifest.split == Split.TEST
            and bundle.manifest.label_commitment_scheme != LabelCommitmentScheme.HMAC_SHA256
        ):
            raise ValueError('every sealed test episode requires an HMAC-SHA256 label commitment')

    suite = make_suite_manifest(suite_id, bundles)
    suite_sha256 = suite_manifest_sha256(suite)
    if admission is not None:
        expected_admission_bindings = tuple((binding.episode_id, binding.manifest_sha256) for binding in suite.episodes)
        actual_admission_bindings = tuple(
            (binding.episode_id, binding.manifest_sha256) for binding in admission.episodes
        )
        if actual_admission_bindings != expected_admission_bindings:
            raise ValueError('challenge admission does not match the selected suite episodes')
    bundle_by_id = {bundle.manifest.episode_id: bundle for bundle in bundles}
    envelopes = tuple(
        ChallengeEnvelope(
            challenge_id=challenge_id,
            suite_id=suite.suite_id,
            suite_manifest_sha256=suite_sha256,
            ordinal=ordinal,
            sample_index=sample_index,
            prompt_variant=prompt_variant,
            binding=binding,
            messages=(
                ChatMessage(role='system', content=build_system_prompt(bundle_by_id[binding.episode_id])),
                ChatMessage(
                    role='user',
                    content=build_episode_prompt(bundle_by_id[binding.episode_id], variant=prompt_variant),
                ),
            ),
        )
        for ordinal, binding in enumerate(suite.episodes)
    )
    if sum(len(canonical_json_bytes(envelope)) for envelope in envelopes) > _MAX_TOTAL_ENVELOPE_BYTES:
        raise ValueError('challenge envelopes exceed the aggregate size limit')
    envelope_bindings = tuple(
        ChallengeEnvelopeFile(
            ordinal=envelope.ordinal,
            episode_id=envelope.binding.episode_id,
            path=f'episodes/{envelope.ordinal:06d}.json',
            envelope_sha256=challenge_envelope_sha256(envelope),
        )
        for envelope in envelopes
    )
    manifest = ChallengeBundleManifest(
        challenge_id=challenge_id,
        suite_id=suite.suite_id,
        suite_manifest_sha256=suite_sha256,
        prompt_variant=prompt_variant,
        admission_path='admission.json' if admission is not None else None,
        admission_sha256=challenge_admission_sha256(admission) if admission is not None else None,
        envelopes=envelope_bindings,
    )

    target = output_dir.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f'challenge output already exists: {target}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        (staging / 'episodes').mkdir()
        (staging / 'suite.json').write_bytes(canonical_json_bytes(suite))
        if admission is not None:
            (staging / 'admission.json').write_bytes(canonical_json_bytes(admission))
        for binding, envelope in zip(envelope_bindings, envelopes, strict=True):
            (staging / binding.path).write_bytes(canonical_json_bytes(envelope))
        (staging / 'challenge.json').write_bytes(canonical_json_bytes(manifest))
        for path in staging.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o644)
        staging.chmod(0o755)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_challenge_bundle(target)


def load_challenge_bundle(root: Path) -> LoadedChallengeBundle:
    """Verify the exact file allowlist, canonical encodings, and every cross-file binding."""

    supplied_root = root.expanduser()
    if supplied_root.is_symlink():
        raise ChallengeIntegrityError('challenge root cannot be a symlink')
    resolved_root = supplied_root.resolve()
    if not resolved_root.is_dir():
        raise ChallengeIntegrityError(f'challenge root does not exist: {resolved_root}')

    manifest_bytes = _read_regular_file(resolved_root / 'challenge.json', _MAX_MANIFEST_BYTES)
    try:
        manifest = ChallengeBundleManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise ChallengeIntegrityError(f'invalid challenge manifest: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ChallengeIntegrityError('challenge manifest must use canonical JSON encoding')

    expected_files = {
        'challenge.json',
        manifest.suite_path,
        *(binding.path for binding in manifest.envelopes),
    }
    if manifest.admission_path is not None:
        expected_files.add(manifest.admission_path)
    _validate_file_inventory(resolved_root, expected_files)

    suite_bytes = _read_regular_file(resolved_root / manifest.suite_path, _MAX_MANIFEST_BYTES)
    try:
        suite = SuiteManifest.model_validate_json(suite_bytes)
    except ValueError as error:
        raise ChallengeIntegrityError(f'invalid suite manifest: {error}') from error
    if suite_bytes != canonical_json_bytes(suite):
        raise ChallengeIntegrityError('suite manifest must use canonical JSON encoding')
    if suite_manifest_sha256(suite) != manifest.suite_manifest_sha256:
        raise ChallengeIntegrityError('suite manifest hash does not match the challenge manifest')
    if suite.suite_id != manifest.suite_id:
        raise ChallengeIntegrityError('suite ID does not match the challenge manifest')
    if len(suite.episodes) != len(manifest.envelopes):
        raise ChallengeIntegrityError('suite and challenge envelope counts differ')

    admission = None
    if manifest.admission_path is not None:
        admission_bytes = _read_regular_file(resolved_root / manifest.admission_path, _MAX_MANIFEST_BYTES)
        try:
            admission = ChallengeAdmissionCommitment.model_validate_json(admission_bytes)
        except ValueError as error:
            raise ChallengeIntegrityError(f'invalid challenge admission: {error}') from error
        if admission_bytes != canonical_json_bytes(admission):
            raise ChallengeIntegrityError('challenge admission must use canonical JSON encoding')
        if challenge_admission_sha256(admission) != manifest.admission_sha256:
            raise ChallengeIntegrityError('challenge admission hash does not match the challenge manifest')
        expected_bindings = tuple((binding.episode_id, binding.manifest_sha256) for binding in suite.episodes)
        actual_bindings = tuple((binding.episode_id, binding.manifest_sha256) for binding in admission.episodes)
        if actual_bindings != expected_bindings:
            raise ChallengeIntegrityError('challenge admission does not match the suite bindings')

    envelopes: list[ChallengeEnvelope] = []
    total_envelope_bytes = 0
    for file_binding, suite_binding in zip(manifest.envelopes, suite.episodes, strict=True):
        envelope_bytes = _read_regular_file(resolved_root / file_binding.path, _MAX_ENVELOPE_BYTES)
        total_envelope_bytes += len(envelope_bytes)
        if total_envelope_bytes > _MAX_TOTAL_ENVELOPE_BYTES:
            raise ChallengeIntegrityError('challenge envelopes exceed the aggregate size limit')
        try:
            envelope = ChallengeEnvelope.model_validate_json(envelope_bytes)
        except ValueError as error:
            raise ChallengeIntegrityError(f'invalid challenge envelope {file_binding.path}: {error}') from error
        if envelope_bytes != canonical_json_bytes(envelope):
            raise ChallengeIntegrityError(f'challenge envelope {file_binding.path} must use canonical JSON encoding')
        if challenge_envelope_sha256(envelope) != file_binding.envelope_sha256:
            raise ChallengeIntegrityError(f'challenge envelope hash mismatch for {file_binding.path}')
        if envelope.ordinal != file_binding.ordinal or envelope.binding.episode_id != file_binding.episode_id:
            raise ChallengeIntegrityError(f'challenge envelope path binding mismatch for {file_binding.path}')
        if envelope.binding != suite_binding:
            raise ChallengeIntegrityError(f'challenge envelope suite binding mismatch for {file_binding.path}')
        if (
            envelope.challenge_id != manifest.challenge_id
            or envelope.suite_id != suite.suite_id
            or envelope.suite_manifest_sha256 != manifest.suite_manifest_sha256
            or envelope.prompt_variant != manifest.prompt_variant
        ):
            raise ChallengeIntegrityError(f'challenge envelope metadata mismatch for {file_binding.path}')
        envelopes.append(envelope)

    return LoadedChallengeBundle(
        root=resolved_root,
        manifest=manifest,
        suite=suite,
        admission=admission,
        envelopes=tuple(envelopes),
        manifest_sha256=challenge_bundle_sha256(manifest),
    )


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ChallengeIntegrityError(f'cannot open challenge file {path.name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ChallengeIntegrityError(f'challenge artifact is not a regular file: {path.name}')
        if metadata.st_size > maximum_bytes:
            raise ChallengeIntegrityError(f'challenge file exceeds its size limit: {path.name}')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ChallengeIntegrityError(f'challenge file exceeds its size limit: {path.name}')
    except OSError as error:
        raise ChallengeIntegrityError(f'cannot read challenge file {path.name}: {error}') from error
    finally:
        os.close(descriptor)


def _validate_file_inventory(root: Path, expected_files: set[str]) -> None:
    expected_root_files = {path for path in expected_files if '/' not in path}
    expected_episode_files = {Path(path).name for path in expected_files if path.startswith('episodes/')}
    root_files: set[str] = set()
    root_directories: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ChallengeIntegrityError(f'challenge artifact cannot contain symlinks: {entry.name}')
                if entry.is_file(follow_symlinks=False):
                    root_files.add(entry.name)
                elif entry.is_dir(follow_symlinks=False):
                    root_directories.add(entry.name)
                else:
                    raise ChallengeIntegrityError(f'challenge artifact contains a non-regular file: {entry.name}')
    except OSError as error:
        raise ChallengeIntegrityError(f'cannot inventory challenge artifact: {error}') from error
    if root_directories != {'episodes'}:
        raise ChallengeIntegrityError('challenge artifact contains unexpected directories')
    if root_files != expected_root_files:
        missing = sorted(expected_root_files - root_files)
        extra = sorted(root_files - expected_root_files)
        raise ChallengeIntegrityError(f'challenge file allowlist mismatch; missing={missing}, extra={extra}')
    episode_files: set[str] = set()
    try:
        with os.scandir(root / 'episodes') as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ChallengeIntegrityError('challenge episodes can contain only regular files')
                episode_files.add(entry.name)
    except OSError as error:
        raise ChallengeIntegrityError(f'cannot inventory challenge episodes: {error}') from error
    if episode_files != expected_episode_files:
        actual_files = {f'episodes/{name}' for name in episode_files} | root_files
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ChallengeIntegrityError(f'challenge file allowlist mismatch; missing={missing}, extra={extra}')
