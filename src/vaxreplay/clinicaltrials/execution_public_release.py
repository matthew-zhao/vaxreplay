"""Standalone, authenticated public package for Lane A execution tasks.

The model-facing tree in an execution workspace intentionally lives beside organizer mappings and
private gold.  That is useful during construction, but it is the wrong filesystem root to mount in
a worker.  This module verifies the complete source workspace against an out-of-band receipt pin,
copies only receipt-bound ``public/`` files into new regular files under a separate root, and emits
an independently pinned release receipt.

This is a packaging boundary, not an admission decision.  Redistribution approval, benchmark
admission, sealed execution, and identity-contamination control all remain explicitly false.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_task import ExecutionTask, execution_task_context_sha256
from vaxreplay.clinicaltrials.execution_workspace import (
    ExecutionWorkspaceArtifactRole,
    ExecutionWorkspaceCount,
    LoadedExecutionWorkspaceBuild,
    verify_execution_workspace_build,
)

EXECUTION_PUBLIC_RELEASE_SCHEMA_VERSION = 'vaxreplay.clinical-execution-public-release.dev-v0.1'
EXECUTION_PUBLIC_RELEASE_BUILDER_ID = 'aact-lane-a-standalone-public-release-v0.1'
EXECUTION_PUBLIC_RELEASE_RECEIPT = 'RELEASE-RECEIPT.json'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_NCT_PATTERN = re.compile(rb'NCT\d{8}', re.IGNORECASE)
_EPISODE_ID_PATTERN = r'execution-dev-[0-9a-f]{24}'
_TASK_PATH_PATTERN = re.compile(
    rf'^tasks/(?P<episode>{_EPISODE_ID_PATTERN})/'
    r'(?P<leaf>TASK\.json|TASK\.md|task-manifest\.json|sources/target-profile\.json|'
    r'sources/reference-trials\.jsonl)$'
)
_EXPECTED_TASK_LEAVES = frozenset(
    {
        'TASK.json',
        'TASK.md',
        'task-manifest.json',
        'sources/target-profile.json',
        'sources/reference-trials.jsonl',
    }
)
_FORBIDDEN_PATH_PARTS = frozenset({'organizer', 'private'})
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_RELEASE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_RELEASE_FILES = 100_000


class ExecutionPublicReleaseError(ValueError):
    """A standalone Lane A public release failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ExecutionPublicReleaseArtifact(StrictModel):
    """Exact public bytes copied from one receipt-bound source workspace artifact."""

    relative_path: str = Field(min_length=1, max_length=4_096)
    source_workspace_relative_path: str = Field(min_length=1, max_length=4_103)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=_MAX_ARTIFACT_BYTES)
    mode: Literal['0444'] = '0444'

    @model_validator(mode='after')
    def validate_paths(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != self.relative_path:
            raise ValueError('public release artifact paths must be normalized and relative')
        if _FORBIDDEN_PATH_PARTS.intersection(part.casefold() for part in path.parts):
            raise ValueError('public release cannot contain organizer or private path components')
        match = _TASK_PATH_PATTERN.fullmatch(self.relative_path)
        if match is None:
            raise ValueError('public release artifact path is outside the fixed Lane A task surface')
        if self.source_workspace_relative_path != f'public/{self.relative_path}':
            raise ValueError('public release artifact must map exactly to its source public path')
        return self


def _release_tree_sha256(artifacts: tuple[ExecutionPublicReleaseArtifact, ...]) -> str:
    return _sha256(canonical_json_bytes([item.model_dump(mode='json') for item in artifacts]))


class ExecutionPublicReleaseReceipt(StrictModel):
    """Externally pinnable identity of a standalone, public-only development package."""

    schema_version: Literal['vaxreplay.clinical-execution-public-release.dev-v0.1'] = (
        EXECUTION_PUBLIC_RELEASE_SCHEMA_VERSION
    )
    builder_id: Literal['aact-lane-a-standalone-public-release-v0.1'] = EXECUTION_PUBLIC_RELEASE_BUILDER_ID
    release_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    source_workspace_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_workspace_context_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_workspace_public_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_count: int = Field(gt=0)
    split_counts: tuple[ExecutionWorkspaceCount, ...]
    artifacts: tuple[ExecutionPublicReleaseArtifact, ...] = Field(min_length=1, max_length=_MAX_RELEASE_FILES)
    source_workspace_external_receipt_pin_verified: Literal[True] = True
    source_workspace_build_integrity_verified: Literal[True] = True
    source_data_real: Literal[True] = True
    standalone_public_tree: Literal[True] = True
    public_files_copied_to_new_inodes: Literal[True] = True
    literal_registry_identifiers_removed: Literal[True] = True
    product_and_sponsor_names_removed: Literal[False] = False
    raw_intervention_and_sponsor_name_fields_omitted: Literal[True] = True
    free_text_removed: Literal[True] = True
    source_span_mapping_complete: Literal[False] = False
    organizer_mappings_included: Literal[False] = False
    private_gold_included: Literal[False] = False
    secret_key_material_included: Literal[False] = False
    development_only: Literal[True] = True
    redistribution_approved: Literal[False] = False
    distribution_ready: Literal[False] = False
    distribution_admitted: Literal[False] = False
    leaderboard_admitted: Literal[False] = False
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    residual_model_weight_reidentification_risk: Literal[True] = True

    @field_validator('split_counts')
    @classmethod
    def validate_split_counts(cls, value: tuple[ExecutionWorkspaceCount, ...]) -> tuple[ExecutionWorkspaceCount, ...]:
        names = tuple(item.name for item in value)
        if names != tuple(sorted(set(names))):
            raise ValueError('public release split counts must use unique canonical order')
        valid_names = {item.value for item in Split}
        if not names or any(name not in valid_names for name in names):
            raise ValueError('public release split counts contain an unsupported split')
        return value

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        paths = tuple(item.relative_path for item in self.artifacts)
        if paths != tuple(sorted(set(paths))):
            raise ValueError('public release artifacts must use unique canonical order')
        leaves_by_episode: dict[str, set[str]] = {}
        for artifact in self.artifacts:
            match = _TASK_PATH_PATTERN.fullmatch(artifact.relative_path)
            if match is None:  # Defensive: the artifact model already enforces this.
                raise ValueError('invalid public release artifact path')
            leaves_by_episode.setdefault(match.group('episode'), set()).add(match.group('leaf'))
        if any(leaves != _EXPECTED_TASK_LEAVES for leaves in leaves_by_episode.values()):
            raise ValueError('each public release task must contain exactly the fixed five-file surface')
        if len(leaves_by_episode) != self.task_count:
            raise ValueError('public release task count does not match its artifact inventory')
        if sum(item.count for item in self.split_counts) != self.task_count:
            raise ValueError('public release split counts do not sum to task count')
        if _release_tree_sha256(self.artifacts) != self.release_tree_sha256:
            raise ValueError('public release tree commitment does not reconstruct')
        return self


@dataclass(frozen=True)
class LoadedExecutionPublicRelease:
    root: Path
    receipt: ExecutionPublicReleaseReceipt
    receipt_sha256: str
    tasks: tuple[ExecutionTask, ...]


def _read_regular_file(path: Path, *, maximum_bytes: int, expected_mode: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ExecutionPublicReleaseError(f'cannot open public release file {path.name}: {error}') from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExecutionPublicReleaseError(f'public release artifact is not a regular file: {path.name}')
        if stat.S_IMODE(before.st_mode) != expected_mode:
            raise ExecutionPublicReleaseError(
                f'public release artifact must have mode {expected_mode:04o}: {path.name}'
            )
        if before.st_nlink != 1:
            raise ExecutionPublicReleaseError(f'public release artifacts cannot be hard linked: {path.name}')
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ExecutionPublicReleaseError(f'public release file has an invalid size: {path.name}')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ExecutionPublicReleaseError(f'public release file exceeds its size limit: {path.name}')
        after = os.fstat(descriptor)
        stable_fields = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ExecutionPublicReleaseError(f'public release file changed while being read: {path.name}')
        return bytes(content), after
    except OSError as error:
        raise ExecutionPublicReleaseError(f'cannot read public release file {path.name}: {error}') from error
    finally:
        os.close(descriptor)


def _write_public_file(root: Path, relative_path: str, payload: bytes) -> os.stat_result:
    if not payload or len(payload) > _MAX_ARTIFACT_BYTES:
        raise ExecutionPublicReleaseError(f'public release artifact has an invalid size: {relative_path}')
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return metadata


def _expected_directories(expected_files: set[str]) -> set[str]:
    directories: set[str] = set()
    for value in expected_files:
        parent = PurePosixPath(value).parent
        while parent != PurePosixPath('.'):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _inventory_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path, relative_directory: PurePosixPath) -> None:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = relative_directory / entry.name
                    relative_text = relative.as_posix()
                    if entry.is_symlink():
                        raise ExecutionPublicReleaseError(
                            f'public release cannot contain symbolic links: {relative_text}'
                        )
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(metadata.st_mode):
                        if stat.S_IMODE(metadata.st_mode) != 0o555:
                            raise ExecutionPublicReleaseError(
                                f'public release directories must have mode 0555: {relative_text}'
                            )
                        directories.add(relative_text)
                        visit(Path(entry.path), relative)
                    elif stat.S_ISREG(metadata.st_mode):
                        files.add(relative_text)
                    else:
                        raise ExecutionPublicReleaseError(
                            f'public release contains a non-regular filesystem object: {relative_text}'
                        )
                    if len(files) + len(directories) > _MAX_RELEASE_FILES * 3:
                        raise ExecutionPublicReleaseError('public release filesystem inventory exceeds its limit')
        except OSError as error:
            raise ExecutionPublicReleaseError(f'cannot inventory public release: {error}') from error

    visit(root, PurePosixPath('.'))
    return files, directories


def _assert_public_payload(payload: bytes, *, label: str) -> None:
    if _NCT_PATTERN.search(payload):
        raise ExecutionPublicReleaseError(f'public release artifact contains a registry identifier: {label}')


def _assert_disjoint_roots(source: Path, output: Path) -> None:
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ExecutionPublicReleaseError(
            'public release root must be separate from, not above or below, the source workspace'
        )


def _source_public_payloads(
    verified: LoadedExecutionWorkspaceBuild,
) -> tuple[tuple[ExecutionPublicReleaseArtifact, bytes, os.stat_result], ...]:
    copied: list[tuple[ExecutionPublicReleaseArtifact, bytes, os.stat_result]] = []
    total_bytes = 0
    for source_binding in verified.receipt.artifacts:
        if source_binding.role != ExecutionWorkspaceArtifactRole.PUBLIC:
            continue
        if not source_binding.relative_path.startswith('public/'):
            raise ExecutionPublicReleaseError('source workspace public artifact has an invalid path')
        relative_path = source_binding.relative_path.removeprefix('public/')
        payload, metadata = _read_regular_file(
            verified.root / source_binding.relative_path,
            maximum_bytes=_MAX_ARTIFACT_BYTES,
            expected_mode=0o444,
        )
        if len(payload) != source_binding.byte_count or not hmac.compare_digest(
            _sha256(payload), source_binding.sha256
        ):
            raise ExecutionPublicReleaseError(
                f'source public artifact changed after workspace verification: {source_binding.relative_path}'
            )
        _assert_public_payload(payload, label=source_binding.relative_path)
        artifact = ExecutionPublicReleaseArtifact(
            relative_path=relative_path,
            source_workspace_relative_path=source_binding.relative_path,
            sha256=source_binding.sha256,
            byte_count=source_binding.byte_count,
        )
        copied.append((artifact, payload, metadata))
        total_bytes += len(payload)
        if len(copied) > _MAX_RELEASE_FILES or total_bytes > _MAX_RELEASE_BYTES:
            raise ExecutionPublicReleaseError('source public tree exceeds release limits')
    if not copied:
        raise ExecutionPublicReleaseError('source workspace contains no public artifacts')
    return tuple(sorted(copied, key=lambda item: item[0].relative_path))


def build_execution_public_release(
    *,
    source_workspace_root: Path,
    expected_source_workspace_receipt_sha256: str,
    output_root: Path,
    release_id: str,
    expected_task_count: int | None = None,
) -> LoadedExecutionPublicRelease:
    """Build an atomic standalone copy of only the verified workspace's public task tree."""

    verified = verify_execution_workspace_build(
        source_workspace_root,
        expected_receipt_sha256=expected_source_workspace_receipt_sha256,
    )
    if expected_task_count is not None and verified.receipt.task_count != expected_task_count:
        raise ExecutionPublicReleaseError(
            f'source workspace has {verified.receipt.task_count} tasks, expected {expected_task_count}'
        )
    source = verified.root.resolve()
    requested_target = output_root.expanduser().absolute()
    if os.path.lexists(requested_target):
        raise FileExistsError(f'public release output already exists: {requested_target}')
    target = requested_target.resolve(strict=False)
    _assert_disjoint_roots(source, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = target.parent.resolve() / target.name
    _assert_disjoint_roots(source, target)
    if os.path.lexists(target):
        raise FileExistsError(f'public release output already exists: {target}')

    source_payloads = _source_public_payloads(verified)
    source_artifacts = tuple(item[0] for item in source_payloads)
    receipt = ExecutionPublicReleaseReceipt(
        release_id=release_id,
        source_workspace_receipt_sha256=expected_source_workspace_receipt_sha256,
        source_workspace_context_plan_sha256=verified.receipt.context_plan_sha256,
        source_workspace_public_tree_sha256=verified.receipt.public_tree_sha256,
        release_tree_sha256=_release_tree_sha256(source_artifacts),
        task_count=verified.receipt.task_count,
        split_counts=verified.receipt.split_counts,
        artifacts=source_artifacts,
    )

    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.staging-', dir=target.parent))
    try:
        for artifact, payload, source_metadata in source_payloads:
            copied_metadata = _write_public_file(staging, artifact.relative_path, payload)
            if (source_metadata.st_dev, source_metadata.st_ino) == (copied_metadata.st_dev, copied_metadata.st_ino):
                raise ExecutionPublicReleaseError('public release artifact was not copied to a new inode')
        receipt_payload = canonical_json_bytes(receipt)
        _write_public_file(staging, EXECUTION_PUBLIC_RELEASE_RECEIPT, receipt_payload)

        directories = sorted(
            (path for path in staging.rglob('*') if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            directory.chmod(0o555)
        staging.chmod(0o555)
        for directory in directories:
            fsync_directory(directory)
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    receipt_sha256 = _sha256(canonical_json_bytes(receipt))
    return verify_execution_public_release(
        target,
        expected_receipt_sha256=receipt_sha256,
        expected_source_workspace_receipt_sha256=expected_source_workspace_receipt_sha256,
        expected_task_count=expected_task_count,
    )


def _parse_task_manifest(payload: bytes, *, episode_id: str, task: ExecutionTask) -> str:
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutionPublicReleaseError(f'invalid public task manifest for {episode_id}: {error}') from error
    if canonical_json_bytes(parsed) != payload:
        raise ExecutionPublicReleaseError(f'public task manifest must use canonical JSON: {episode_id}')
    expected_keys = {
        'schema_version',
        'episode_id',
        'target_trial_id',
        'public_lineage_id',
        'split',
        'task_sha256',
        'task_context_sha256',
        'response_schema_version',
        'sources',
        'development_only',
        'leaderboard_admitted',
        'identity_contamination_controlled',
    }
    if not isinstance(parsed, dict) or set(parsed) != expected_keys:
        raise ExecutionPublicReleaseError(f'public task manifest has an unexpected schema: {episode_id}')
    if (
        parsed['schema_version'] != 'vaxreplay.clinical-execution-public-task-manifest.dev-v0.1'
        or parsed['episode_id'] != episode_id
        or parsed['target_trial_id'] != task.context.target_trial_id
        or parsed['task_sha256'] != _sha256(canonical_json_bytes(task))
        or parsed['task_context_sha256'] != task.context_sha256
        or parsed['response_schema_version'] != 'vaxreplay.clinical-execution-submission.dev-v0.1'
        or parsed['development_only'] is not True
        or parsed['leaderboard_admitted'] is not False
        or parsed['identity_contamination_controlled'] is not False
    ):
        raise ExecutionPublicReleaseError(f'public task manifest binding mismatch: {episode_id}')
    try:
        return Split(parsed['split']).value
    except (TypeError, ValueError) as error:
        raise ExecutionPublicReleaseError(f'public task manifest has an invalid split: {episode_id}') from error


def _verify_task_surface(root: Path, receipt: ExecutionPublicReleaseReceipt) -> tuple[ExecutionTask, ...]:
    artifact_by_path = {item.relative_path: item for item in receipt.artifacts}
    episode_ids = sorted(
        {
            match.group('episode')
            for item in receipt.artifacts
            if (match := _TASK_PATH_PATTERN.fullmatch(item.relative_path)) is not None
        }
    )
    tasks: list[ExecutionTask] = []
    split_counts: Counter[str] = Counter()
    for episode_id in episode_ids:
        task_path = f'tasks/{episode_id}/TASK.json'
        task_payload, _ = _read_regular_file(
            root / task_path,
            maximum_bytes=artifact_by_path[task_path].byte_count,
            expected_mode=0o444,
        )
        try:
            task = ExecutionTask.model_validate_json(task_payload)
        except ValueError as error:
            raise ExecutionPublicReleaseError(f'invalid public task for {episode_id}: {error}') from error
        if task_payload != canonical_json_bytes(task):
            raise ExecutionPublicReleaseError(f'public task must use canonical JSON: {episode_id}')
        if (
            task.context.episode_id != episode_id
            or task.context_sha256 != execution_task_context_sha256(task.context)
            or task.development_only is not True
            or task.leaderboard_admitted is not False
            or task.sealed_execution_supported is not False
            or task.identity_contamination_controlled is not False
            or task.context.development_only is not True
            or task.context.leaderboard_admitted is not False
            or task.context.sealed_execution_supported is not False
            or task.context.identity_contamination_controlled is not False
        ):
            raise ExecutionPublicReleaseError(f'public task carries invalid admission flags or bindings: {episode_id}')

        manifest_path = f'tasks/{episode_id}/task-manifest.json'
        manifest_payload, _ = _read_regular_file(
            root / manifest_path,
            maximum_bytes=artifact_by_path[manifest_path].byte_count,
            expected_mode=0o444,
        )
        split_counts[_parse_task_manifest(manifest_payload, episode_id=episode_id, task=task)] += 1

        expected_sources: dict[str, dict[str, object]] = {}
        for document in task.context.cutoff_documents:
            if document.document_id == 'target-profile':
                source_path = f'tasks/{episode_id}/sources/target-profile.json'
            elif document.document_id == 'reference-trials':
                source_path = f'tasks/{episode_id}/sources/reference-trials.jsonl'
            else:
                raise ExecutionPublicReleaseError(f'public task contains an unsupported source: {episode_id}')
            source_payload, _ = _read_regular_file(
                root / source_path,
                maximum_bytes=artifact_by_path[source_path].byte_count,
                expected_mode=0o444,
            )
            if source_payload != document.body.encode('utf-8') or _sha256(source_payload) != document.body_sha256:
                raise ExecutionPublicReleaseError(f'public source does not match task context: {source_path}')
            expected_sources[source_path.removeprefix(f'tasks/{episode_id}/')] = {
                'sha256': document.body_sha256,
                'byte_count': len(source_payload),
            }
        parsed_manifest = json.loads(manifest_payload)
        if parsed_manifest['sources'] != expected_sources:
            raise ExecutionPublicReleaseError(f'public task source manifest mismatch: {episode_id}')
        tasks.append(task)
    expected_counts = {item.name: item.count for item in receipt.split_counts}
    if dict(sorted(split_counts.items())) != expected_counts:
        raise ExecutionPublicReleaseError('public release split counts do not match task manifests')
    return tuple(tasks)


def verify_execution_public_release(
    root: Path,
    *,
    expected_receipt_sha256: str,
    expected_source_workspace_receipt_sha256: str | None = None,
    expected_task_count: int | None = None,
) -> LoadedExecutionPublicRelease:
    """Verify the external receipt pin, exact public inventory, modes, hashes, and task bindings."""

    if re.fullmatch(_SHA256_PATTERN, expected_receipt_sha256) is None:
        raise ExecutionPublicReleaseError('expected public release receipt SHA-256 is invalid')
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ExecutionPublicReleaseError('public release root cannot be a symbolic link')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ExecutionPublicReleaseError('public release root must be a directory')
    root_metadata = os.stat(resolved, follow_symlinks=False)
    if stat.S_IMODE(root_metadata.st_mode) != 0o555:
        raise ExecutionPublicReleaseError('public release root must have mode 0555')

    receipt_payload, _ = _read_regular_file(
        resolved / EXECUTION_PUBLIC_RELEASE_RECEIPT,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        expected_mode=0o444,
    )
    if not hmac.compare_digest(_sha256(receipt_payload), expected_receipt_sha256):
        raise ExecutionPublicReleaseError('public release receipt does not match its external pin')
    try:
        receipt = ExecutionPublicReleaseReceipt.model_validate_json(receipt_payload)
    except ValueError as error:
        raise ExecutionPublicReleaseError(f'invalid public release receipt: {error}') from error
    if receipt_payload != canonical_json_bytes(receipt):
        raise ExecutionPublicReleaseError('public release receipt must use canonical JSON encoding')
    if expected_source_workspace_receipt_sha256 is not None and not hmac.compare_digest(
        receipt.source_workspace_receipt_sha256, expected_source_workspace_receipt_sha256
    ):
        raise ExecutionPublicReleaseError('public release does not bind the expected source workspace receipt')
    if expected_task_count is not None and receipt.task_count != expected_task_count:
        raise ExecutionPublicReleaseError(
            f'public release has {receipt.task_count} tasks, expected {expected_task_count}'
        )

    expected_files = {item.relative_path for item in receipt.artifacts} | {EXECUTION_PUBLIC_RELEASE_RECEIPT}
    observed_files, observed_directories = _inventory_tree(resolved)
    if observed_files != expected_files or observed_directories != _expected_directories(expected_files):
        raise ExecutionPublicReleaseError('public release contains missing or uncommitted files or directories')

    total_bytes = 0
    for artifact in receipt.artifacts:
        payload, _ = _read_regular_file(
            resolved / artifact.relative_path,
            maximum_bytes=_MAX_ARTIFACT_BYTES,
            expected_mode=0o444,
        )
        total_bytes += len(payload)
        if (
            len(payload) != artifact.byte_count
            or not hmac.compare_digest(_sha256(payload), artifact.sha256)
            or total_bytes > _MAX_RELEASE_BYTES
        ):
            raise ExecutionPublicReleaseError(
                f'public release artifact does not match receipt: {artifact.relative_path}'
            )
        _assert_public_payload(payload, label=artifact.relative_path)

    tasks = _verify_task_surface(resolved, receipt)
    if len(tasks) != receipt.task_count:
        raise ExecutionPublicReleaseError('verified public task count does not match release receipt')
    return LoadedExecutionPublicRelease(
        root=resolved,
        receipt=receipt,
        receipt_sha256=expected_receipt_sha256,
        tasks=tasks,
    )


__all__ = [
    'EXECUTION_PUBLIC_RELEASE_BUILDER_ID',
    'EXECUTION_PUBLIC_RELEASE_RECEIPT',
    'EXECUTION_PUBLIC_RELEASE_SCHEMA_VERSION',
    'ExecutionPublicReleaseArtifact',
    'ExecutionPublicReleaseError',
    'ExecutionPublicReleaseReceipt',
    'LoadedExecutionPublicRelease',
    'build_execution_public_release',
    'verify_execution_public_release',
]
