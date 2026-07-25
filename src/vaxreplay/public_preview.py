"""Build and audit a sanitized VaxReplay technical-preview export."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PublicPreviewError(RuntimeError):
    """Raised when a public-preview export would cross its release boundary."""


class PreviewFileMap(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    source: str
    destination: str


class PreviewTreeMap(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    source: str
    destination: str
    exclude: tuple[str, ...] = ()


class ForbiddenTextPattern(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    name: str
    pattern: str


class PublicPreviewPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    schema_version: Literal[2]
    release_name: str
    approved_static_path_count: int = Field(gt=0)
    approved_static_paths_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    mapped_files: tuple[PreviewFileMap, ...]
    include_files: tuple[str, ...]
    optional_files: tuple[str, ...] = ()
    include_trees: tuple[PreviewTreeMap, ...]
    excluded_components: tuple[str, ...] = ()
    excluded_globs: tuple[str, ...] = ()
    forbidden_prefixes: tuple[str, ...] = ()
    forbidden_suffixes: tuple[str, ...] = ()
    forbidden_text_patterns: tuple[ForbiddenTextPattern, ...] = ()
    max_file_bytes: int = Field(gt=0)
    final_required_files: tuple[str, ...] = ()


class PublicPreviewBuild(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    release_name: str
    draft: bool
    source_revision: str
    source_dirty: bool
    file_count: int
    manifest_sha256: str
    private_export_policy_canonical_sha256: str
    static_export_path_count: int
    static_export_paths_sha256: str


_DRAFT_MARKER = 'DRAFT-NOT-FOR-DISTRIBUTION.md'
_BUILD_INFO = 'BUILD-INFO.json'
_MANIFEST = 'MANIFEST.sha256'
_UNRESOLVED_PLACEHOLDER = re.compile(r'\{\{[A-Z][A-Z0-9_]*\}\}')


def load_public_preview_policy(path: Path) -> PublicPreviewPolicy:
    """Load and validate a public-preview policy."""

    try:
        raw = path.read_text(encoding='utf-8')
    except OSError as error:
        raise PublicPreviewError(f'could not read policy {path}: {error}') from error
    return PublicPreviewPolicy.model_validate_json(raw)


def _safe_relative_path(raw: str, *, field: str) -> Path:
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or not candidate.parts or any(part in {'', '.', '..'} for part in candidate.parts):
        raise PublicPreviewError(f'{field} must be a safe relative POSIX path: {raw!r}')
    return Path(*candidate.parts)


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _tree_path_is_excluded(path: Path, *, tree: PreviewTreeMap, policy: PublicPreviewPolicy) -> bool:
    posix = path.as_posix()
    return (
        any(component in policy.excluded_components for component in path.parts)
        or _matches_any(posix, policy.excluded_globs)
        or _matches_any(posix, tree.exclude)
    )


def _copy_regular_file(
    source: Path,
    destination: Path,
    *,
    destination_relative: Path,
    copied: set[Path],
) -> None:
    if source.is_symlink():
        raise PublicPreviewError(f'symlinks are not allowed in the public preview: {source}')
    if not source.is_file():
        raise PublicPreviewError(f'public-preview source is not a regular file: {source}')
    if destination_relative in copied:
        raise PublicPreviewError(f'duplicate public-preview destination: {destination_relative.as_posix()}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied.add(destination_relative)


def _reject_symlinked_source_path(source: Path, *, source_root: Path) -> None:
    try:
        relative = source.relative_to(source_root)
    except ValueError as error:
        raise PublicPreviewError(f'public-preview source escapes the source root: {source}') from error

    current = source_root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise PublicPreviewError(f'symlinks are not allowed in public-preview source paths: {current}')

    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise PublicPreviewError(f'could not resolve public-preview source {source}: {error}') from error
    if not resolved.is_relative_to(source_root):
        raise PublicPreviewError(f'public-preview source escapes the source root: {source}')


def _copy_policy_files(
    *,
    source_root: Path,
    output_root: Path,
    policy: PublicPreviewPolicy,
) -> set[Path]:
    source_root = source_root.resolve()
    copied: set[Path] = set()

    for mapping in policy.mapped_files:
        source_relative = _safe_relative_path(mapping.source, field='mapped file source')
        destination_relative = _safe_relative_path(mapping.destination, field='mapped file destination')
        _reject_symlinked_source_path(source_root / source_relative, source_root=source_root)
        _copy_regular_file(
            source_root / source_relative,
            output_root / destination_relative,
            destination_relative=destination_relative,
            copied=copied,
        )

    for raw in policy.include_files:
        relative = _safe_relative_path(raw, field='included file')
        _reject_symlinked_source_path(source_root / relative, source_root=source_root)
        _copy_regular_file(
            source_root / relative,
            output_root / relative,
            destination_relative=relative,
            copied=copied,
        )

    for raw in policy.optional_files:
        relative = _safe_relative_path(raw, field='optional file')
        source = source_root / relative
        if source.exists() or source.is_symlink():
            _reject_symlinked_source_path(source, source_root=source_root)
            _copy_regular_file(
                source,
                output_root / relative,
                destination_relative=relative,
                copied=copied,
            )

    for tree in policy.include_trees:
        source_relative = _safe_relative_path(tree.source, field='included tree source')
        destination_relative = _safe_relative_path(tree.destination, field='included tree destination')
        tree_root = source_root / source_relative
        _reject_symlinked_source_path(tree_root, source_root=source_root)
        if not tree_root.is_dir():
            raise PublicPreviewError(f'public-preview tree does not exist: {tree_root}')
        for source in sorted(tree_root.rglob('*')):
            relative_in_tree = source.relative_to(tree_root)
            _reject_symlinked_source_path(source, source_root=source_root)
            if source.is_dir() or _tree_path_is_excluded(relative_in_tree, tree=tree, policy=policy):
                continue
            destination = destination_relative / relative_in_tree
            _copy_regular_file(
                source,
                output_root / destination,
                destination_relative=destination,
                copied=copied,
            )

    return copied


def _static_path_binding(paths: set[Path]) -> tuple[int, str]:
    canonical = ''.join(f'{path.as_posix()}\n' for path in sorted(paths, key=lambda item: item.as_posix()))
    return len(paths), hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _verify_static_path_binding(
    paths: set[Path],
    *,
    policy: PublicPreviewPolicy,
) -> tuple[int, str]:
    count, digest = _static_path_binding(paths)
    if count != policy.approved_static_path_count or digest != policy.approved_static_paths_sha256:
        raise PublicPreviewError(
            'public-preview static path inventory differs from the reviewed policy: '
            f'actual count={count}, sha256={digest}; '
            f'approved count={policy.approved_static_path_count}, '
            f'sha256={policy.approved_static_paths_sha256}'
        )
    return count, digest


def _canonical_policy_sha256(policy: PublicPreviewPolicy) -> str:
    canonical = json.dumps(
        policy.model_dump(mode='json'),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _audit_export(output_root: Path, *, policy: PublicPreviewPolicy, draft: bool) -> list[Path]:
    files: list[Path] = []
    compiled_patterns = tuple((item.name, re.compile(item.pattern)) for item in policy.forbidden_text_patterns)

    for path in sorted(output_root.rglob('*')):
        relative = path.relative_to(output_root)
        relative_posix = relative.as_posix()
        if path.is_symlink():
            raise PublicPreviewError(f'symlink found in public preview: {relative_posix}')
        if path.is_dir():
            continue
        if any(relative_posix.startswith(prefix) for prefix in policy.forbidden_prefixes):
            raise PublicPreviewError(f'forbidden path found in public preview: {relative_posix}')
        if any(path.name.endswith(suffix) for suffix in policy.forbidden_suffixes):
            raise PublicPreviewError(f'forbidden file type found in public preview: {relative_posix}')
        if path.stat().st_size > policy.max_file_bytes:
            raise PublicPreviewError(
                f'oversized file found in public preview: {relative_posix} '
                f'({path.stat().st_size} > {policy.max_file_bytes})'
            )
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = ''
        for name, pattern in compiled_patterns:
            if pattern.search(text):
                raise PublicPreviewError(f'{name} found in public preview file {relative_posix}')
        if not draft and _UNRESOLVED_PLACEHOLDER.search(text):
            raise PublicPreviewError(f'unresolved release placeholder found in {relative_posix}')
        files.append(relative)

    if not draft:
        missing = tuple(raw for raw in policy.final_required_files if not (output_root / raw).is_file())
        if missing:
            raise PublicPreviewError(f'final public preview is missing required files: {", ".join(missing)}')

    return files


def _write_manifest(output_root: Path, files: list[Path]) -> str:
    lines: list[str] = []
    for relative in sorted(files, key=lambda item: item.as_posix()):
        digest = hashlib.sha256((output_root / relative).read_bytes()).hexdigest()
        lines.append(f'{digest}  {relative.as_posix()}')
    manifest_text = '\n'.join(lines) + '\n'
    (output_root / _MANIFEST).write_text(manifest_text, encoding='utf-8')
    return hashlib.sha256(manifest_text.encode('utf-8')).hexdigest()


def build_public_preview(
    *,
    source_root: Path,
    output_root: Path,
    policy: PublicPreviewPolicy,
    draft: bool,
    source_revision: str,
    source_dirty: bool,
) -> PublicPreviewBuild:
    """Build one fresh, allowlisted public-preview directory."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise PublicPreviewError(f'output already exists: {output_root}')
    if not draft and source_dirty:
        raise PublicPreviewError('a final public preview cannot be built from a dirty source tree')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f'.{output_root.name}.staging-',
            dir=output_root.parent,
        )
    )
    try:
        copied = _copy_policy_files(source_root=source_root, output_root=staging_root, policy=policy)
        static_path_count, static_paths_sha256 = _verify_static_path_binding(copied, policy=policy)
        policy_sha256 = _canonical_policy_sha256(policy)

        if draft:
            (staging_root / _DRAFT_MARKER).write_text(
                '# Draft—not for distribution\n\n'
                'This export was built before all ownership, license, security-contact, and release '
                'gates were satisfied. It must not be published or distributed.\n',
                encoding='utf-8',
            )

        initial_files = _audit_export(staging_root, policy=policy, draft=draft)
        build_info = {
            'schema_version': 2,
            'release_name': policy.release_name,
            'draft': draft,
            'source_revision': source_revision,
            'source_dirty': source_dirty,
            'file_count_before_generated_metadata': len(initial_files),
            'private_export_policy_canonical_sha256': policy_sha256,
            'static_export_path_count': static_path_count,
            'static_export_paths_sha256': static_paths_sha256,
        }
        (staging_root / _BUILD_INFO).write_text(
            json.dumps(build_info, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        files = _audit_export(staging_root, policy=policy, draft=draft)
        manifest_sha256 = _write_manifest(staging_root, files)
        final_file_count = len(files) + 1
        staging_root.rename(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return PublicPreviewBuild(
        release_name=policy.release_name,
        draft=draft,
        source_revision=source_revision,
        source_dirty=source_dirty,
        file_count=final_file_count,
        manifest_sha256=manifest_sha256,
        private_export_policy_canonical_sha256=policy_sha256,
        static_export_path_count=static_path_count,
        static_export_paths_sha256=static_paths_sha256,
    )
