"""Race-resistant, bounded snapshots of immutable artifact directories.

The artifact formats in :mod:`vaxreplay.operations` are content addressed, but a
plain ``Path.read_bytes`` after an ``os.walk`` still leaves a check/use gap.  This
module keeps an opened root directory descriptor for the whole snapshot, walks
and opens every descendant relative to that descriptor with ``O_NOFOLLOW``, and
requires the complete inode/metadata inventory to remain identical before and
after all reads.

The returned object contains bytes, not live paths.  Callers must perform their
format-specific digest and manifest validation against those exact bytes.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


class ImmutableTreeError(ValueError):
    """The artifact tree is unsafe, changed during capture, or exceeds a bound."""


@dataclass(frozen=True)
class _EntryIdentity:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _TreeInventory:
    root: _EntryIdentity
    directories: tuple[tuple[str, _EntryIdentity], ...]
    files: tuple[tuple[str, _EntryIdentity], ...]

    @property
    def directory_map(self) -> dict[str, _EntryIdentity]:
        return dict(self.directories)

    @property
    def file_map(self) -> dict[str, _EntryIdentity]:
        return dict(self.files)


@dataclass(frozen=True)
class ImmutableTreeSnapshot:
    """Exact in-memory bytes from one stable directory-descriptor snapshot."""

    root: Path
    files: Mapping[str, bytes]
    directories: tuple[str, ...]

    def require_exact_files(self, expected_files: set[str] | frozenset[str]) -> None:
        """Reject undeclared files and empty or otherwise unexpected directories."""

        expected = set(expected_files)
        if not expected or len(expected) != len(expected_files):
            raise ImmutableTreeError('expected artifact file inventory is empty or duplicated')
        if set(self.files) != expected:
            raise ImmutableTreeError('artifact file inventory differs from its exact manifest')
        required_directories: set[str] = set()
        for relative in expected:
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or path.as_posix() != relative:
                raise ImmutableTreeError('expected artifact inventory contains an unsafe path')
            for parent in path.parents:
                if parent != PurePosixPath('.'):
                    required_directories.add(parent.as_posix())
        if set(self.directories) != required_directories:
            raise ImmutableTreeError('artifact directory inventory contains an empty, missing, or unexpected directory')


def immutable_root_identity(root: Path) -> tuple[Path, tuple[int, int]]:
    """Validate an exact no-symlink root path and return its kernel identity."""

    requested, descriptor = _open_root_directory(root)
    try:
        metadata = os.fstat(descriptor)
        return requested, (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)


def snapshot_immutable_tree(
    root: Path,
    *,
    max_files: int,
    max_directories: int,
    max_file_bytes: int,
    max_total_bytes: int,
    max_path_characters: int,
    per_path_byte_limits: Mapping[str, int] | None = None,
    aggregate_exempt_paths: frozenset[str] = frozenset(),
) -> ImmutableTreeSnapshot:
    """Read a complete, stable tree through one anchored root descriptor.

    ``per_path_byte_limits`` may tighten (or separately enlarge) the generic
    per-file limit for known files such as a manifest.  Exempt files are still
    bounded by their per-path/per-file limit but do not count toward the aggregate
    payload limit.
    """

    limits = (max_files, max_directories, max_file_bytes, max_total_bytes, max_path_characters)
    if any(not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 for limit in limits):
        raise ValueError('immutable-tree limits must be positive integers')
    path_limits = dict(per_path_byte_limits or {})
    if any(
        not isinstance(path, str) or not path or not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        for path, limit in path_limits.items()
    ):
        raise ValueError('per-path immutable-tree byte limits are invalid')
    if not aggregate_exempt_paths.issubset(path_limits):
        raise ValueError('aggregate-exempt paths require explicit per-path byte limits')

    resolved, root_descriptor = _open_root_directory(root)
    try:
        before = _inventory_tree(
            root_descriptor,
            max_files=max_files,
            max_directories=max_directories,
            max_path_characters=max_path_characters,
        )
        files: dict[str, bytes] = {}
        total = 0
        for relative, _expected_identity in before.files:
            byte_limit = path_limits.get(relative, max_file_bytes)
            payload = _read_regular_file_at(
                root_descriptor,
                relative,
                byte_limit,
                before,
            )
            if relative not in aggregate_exempt_paths:
                total += len(payload)
                if total > max_total_bytes:
                    raise ImmutableTreeError('artifact tree exceeds its aggregate byte limit')
            files[relative] = payload

        after = _inventory_tree(
            root_descriptor,
            max_files=max_files,
            max_directories=max_directories,
            max_path_characters=max_path_characters,
        )
        if after != before:
            raise ImmutableTreeError('artifact tree changed while being read')
        if _identity(os.fstat(root_descriptor)) != before.root:
            raise ImmutableTreeError('artifact root changed while being read')
        _require_root_path_identity(resolved, before.root)
        return ImmutableTreeSnapshot(
            root=resolved,
            files=MappingProxyType(files),
            directories=tuple(path for path, _identity_value in before.directories),
        )
    finally:
        os.close(root_descriptor)


def _open_root_directory(root: Path) -> tuple[Path, int]:
    # Normalize ``.``/``..`` lexically, then traverse the caller's exact path.
    # Resolving first would silently erase a symlink in an intermediate parent.
    requested = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    if os.open not in os.supports_dir_fd:
        raise ImmutableTreeError('this platform lacks descriptor-relative artifact traversal')
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    anchor = requested.anchor
    try:
        descriptor = os.open(anchor, flags)
        for component in requested.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            finally:
                os.close(descriptor)
            descriptor = child
    except OSError as error:
        raise ImmutableTreeError(f'cannot open artifact root without following links: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ImmutableTreeError('artifact root must be a directory')
    except BaseException:
        os.close(descriptor)
        raise
    return requested, descriptor


def _inventory_tree(
    root_descriptor: int,
    *,
    max_files: int,
    max_directories: int,
    max_path_characters: int,
) -> _TreeInventory:
    directories: dict[str, _EntryIdentity] = {}
    files: dict[str, _EntryIdentity] = {}
    root_identity = _identity(os.fstat(root_descriptor))
    if not stat.S_ISDIR(root_identity.mode):
        raise ImmutableTreeError('artifact root must remain a directory')

    def visit(directory_descriptor: int, prefix: str) -> None:
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as error:
            raise ImmutableTreeError(f'cannot enumerate artifact directory: {prefix or "."}') from error
        for name in names:
            relative = f'{prefix}/{name}' if prefix else name
            if len(relative) > max_path_characters:
                raise ImmutableTreeError('artifact tree contains a path longer than the configured bound')
            try:
                metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            except OSError as error:
                raise ImmutableTreeError(f'cannot inspect artifact entry: {relative}') from error
            entry_identity = _identity(metadata)
            if stat.S_ISLNK(metadata.st_mode):
                raise ImmutableTreeError('artifact tree cannot contain symbolic links')
            if stat.S_ISREG(metadata.st_mode):
                files[relative] = entry_identity
                if len(files) > max_files:
                    raise ImmutableTreeError('artifact file count exceeds the configured bound')
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                raise ImmutableTreeError('artifact tree entries must be regular files or directories')
            directories[relative] = entry_identity
            if len(directories) > max_directories:
                raise ImmutableTreeError('artifact directory count exceeds the configured bound')
            child_flags = (
                os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
            )
            try:
                child_descriptor = os.open(name, child_flags, dir_fd=directory_descriptor)
            except OSError as error:
                raise ImmutableTreeError(f'cannot open artifact directory: {relative}') from error
            try:
                if _identity(os.fstat(child_descriptor)) != entry_identity:
                    raise ImmutableTreeError(f'artifact directory changed while opened: {relative}')
                visit(child_descriptor, relative)
                if _identity(os.fstat(child_descriptor)) != entry_identity:
                    raise ImmutableTreeError(f'artifact directory changed while enumerated: {relative}')
            finally:
                os.close(child_descriptor)

    visit(root_descriptor, '')
    return _TreeInventory(
        root=root_identity,
        directories=tuple(sorted(directories.items())),
        files=tuple(sorted(files.items())),
    )


def _read_regular_file_at(
    root_descriptor: int,
    relative: str,
    max_bytes: int,
    inventory: _TreeInventory,
) -> bytes:
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or '..' in parts:
        raise ImmutableTreeError('cannot read an unsafe artifact path')
    directories = inventory.directory_map
    files = inventory.file_map
    expected_file = files.get(relative)
    if expected_file is None:
        raise ImmutableTreeError(f'artifact file was not present in the initial inventory: {relative}')

    current = os.dup(root_descriptor)
    try:
        if _identity(os.fstat(current)) != inventory.root:
            raise ImmutableTreeError('artifact root changed before a file could be read')
        prefix_parts: list[str] = []
        directory_flags = (
            os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        )
        for component in parts[:-1]:
            prefix_parts.append(component)
            prefix = '/'.join(prefix_parts)
            try:
                child = os.open(component, directory_flags, dir_fd=current)
            except OSError as error:
                raise ImmutableTreeError(f'cannot open artifact directory while reading: {prefix}') from error
            os.close(current)
            current = child
            if _identity(os.fstat(current)) != directories.get(prefix):
                raise ImmutableTreeError(f'artifact directory changed before file read: {prefix}')

        file_flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        try:
            descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        except OSError as error:
            raise ImmutableTreeError(f'cannot open artifact file: {relative}') from error
        try:
            before = _identity(os.fstat(descriptor))
            if before != expected_file or not stat.S_ISREG(before.mode) or before.size > max_bytes:
                raise ImmutableTreeError(f'artifact file changed or exceeds its byte limit: {relative}')
            chunks: list[bytes] = []
            remaining = before.size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ImmutableTreeError(f'artifact file changed while read: {relative}')
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ImmutableTreeError(f'artifact file grew while read: {relative}')
            if _identity(os.fstat(descriptor)) != before:
                raise ImmutableTreeError(f'artifact file changed while read: {relative}')
            try:
                current_entry = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
            except OSError as error:
                raise ImmutableTreeError(f'artifact file changed after read: {relative}') from error
            if _identity(current_entry) != before:
                raise ImmutableTreeError(f'artifact file path changed while read: {relative}')
            return b''.join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _require_root_path_identity(resolved: Path, expected: _EntryIdentity) -> None:
    try:
        _current_path, descriptor = _open_root_directory(resolved)
    except ImmutableTreeError as error:
        raise ImmutableTreeError('artifact root path changed while being read') from error
    try:
        if _identity(os.fstat(descriptor)) != expected:
            raise ImmutableTreeError('artifact root path changed while being read')
    finally:
        os.close(descriptor)


def _identity(metadata: os.stat_result) -> _EntryIdentity:
    return _EntryIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        links=metadata.st_nlink,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )
