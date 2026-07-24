"""Small filesystem primitives shared by immutable VaxReplay artifacts."""

from __future__ import annotations

import errno
import os
import secrets
import shutil
import stat
import sys
from ctypes import CDLL, c_char_p, c_int, c_uint, get_errno, set_errno
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Self


class AtomicDirectoryPublicationError(ValueError):
    """An immutable directory could not be built, published, or cleaned safely."""


@dataclass
class AtomicDirectoryPublication:
    """Build one immutable directory through held, descriptor-relative roots.

    The public staging name is only a handle used to create a private ``0700``
    container. All content lives in ``owned-tree`` beneath the already-opened
    container descriptor, so renaming or replacing the public staging name cannot
    redirect writes. Publication uses a descriptor-relative no-replace rename into
    the pinned output parent. Until :meth:`commit` is called, context-manager exit
    removes only the exact operation-owned tree (or fails closed and retains an
    unexpected replacement).
    """

    target: Path
    parent: Path
    _parent_descriptor: int
    _parent_identity: tuple[int, int]
    _container_name: str
    _container_descriptor: int
    _container_identity: tuple[int, int]
    _tree_descriptor: int
    _tree_identity: tuple[int, int]
    _directories: dict[str, int] = field(default_factory=dict)
    _published: bool = False
    _committed: bool = False
    _closed: bool = False

    @classmethod
    def create(cls, output_dir: Path) -> Self:
        """Create a private descriptor-anchored builder for a named child path."""

        _require_descriptor_primitives()
        requested = Path(os.path.abspath(os.fspath(Path(output_dir).expanduser())))
        if not requested.name or requested == Path(requested.anchor):
            raise AtomicDirectoryPublicationError('immutable output must be a named child directory')
        try:
            canonical_parent = requested.parent.resolve(strict=True)
        except OSError as error:
            raise AtomicDirectoryPublicationError(
                'immutable output parent must already exist and be resolvable'
            ) from error
        target = canonical_parent / requested.name
        parent, parent_descriptor = _open_directory_path(canonical_parent)
        parent_metadata = os.fstat(parent_descriptor)
        parent_identity = _directory_identity(parent_metadata, 'immutable output parent')
        if parent_metadata.st_uid != os.geteuid() or stat.S_IMODE(parent_metadata.st_mode) & 0o022:
            os.close(parent_descriptor)
            raise AtomicDirectoryPublicationError(
                'immutable output parent must be owned by this authority and not group/world writable'
            )
        container_name: str | None = None
        container_identity: tuple[int, int] | None = None
        container_descriptor: int | None = None
        tree_identity: tuple[int, int] | None = None
        tree_descriptor: int | None = None
        try:
            _require_missing_name(parent_descriptor, target.name, 'immutable output')
            container_name = _create_private_directory_at(parent_descriptor, target.name)
            container_metadata = os.stat(container_name, dir_fd=parent_descriptor, follow_symlinks=False)
            container_identity = _directory_identity(container_metadata, 'private publication container')
            try:
                container_descriptor = os.open(
                    container_name,
                    _directory_open_flags(),
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise AtomicDirectoryPublicationError(
                    'cannot open private publication container without following links'
                ) from error
            if (
                _directory_identity(
                    os.fstat(container_descriptor),
                    'opened private publication container',
                )
                != container_identity
            ):
                raise AtomicDirectoryPublicationError('private publication container changed while being opened')
            os.mkdir('owned-tree', mode=0o700, dir_fd=container_descriptor)
            tree_metadata = os.stat(
                'owned-tree',
                dir_fd=container_descriptor,
                follow_symlinks=False,
            )
            tree_identity = _directory_identity(tree_metadata, 'private publication tree')
            tree_descriptor = os.open(
                'owned-tree',
                _directory_open_flags(),
                dir_fd=container_descriptor,
            )
            if (
                _directory_identity(
                    os.fstat(tree_descriptor),
                    'opened private publication tree',
                )
                != tree_identity
            ):
                raise AtomicDirectoryPublicationError('private publication tree changed while being opened')
        except BaseException as error:
            if tree_descriptor is not None:
                os.close(tree_descriptor)
            cleanup_error: BaseException | None = None
            if container_name is not None and container_identity is not None:
                try:
                    _cleanup_partial_publication(
                        parent_descriptor=parent_descriptor,
                        container_name=container_name,
                        container_identity=container_identity,
                        container_descriptor=container_descriptor,
                        tree_identity=tree_identity,
                    )
                except BaseException as partial_cleanup_error:
                    cleanup_error = partial_cleanup_error
            if container_descriptor is not None:
                os.close(container_descriptor)
            os.close(parent_descriptor)
            if cleanup_error is not None:
                error.add_note(f'partial publication cleanup failed closed: {cleanup_error}')
            raise
        assert container_name is not None
        assert container_identity is not None
        assert container_descriptor is not None
        assert tree_identity is not None
        assert tree_descriptor is not None
        return cls(
            target=parent / target.name,
            parent=parent,
            _parent_descriptor=parent_descriptor,
            _parent_identity=parent_identity,
            _container_name=container_name,
            _container_descriptor=container_descriptor,
            _container_identity=container_identity,
            _tree_descriptor=tree_descriptor,
            _tree_identity=tree_identity,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_type, traceback
        cleanup_error: BaseException | None = None
        try:
            if not self._committed:
                self._remove_owned_tree()
            self._remove_private_container()
        except BaseException as error:
            cleanup_error = error
        finally:
            self.close()
        if cleanup_error is not None:
            failure = AtomicDirectoryPublicationError(f'immutable directory cleanup failed closed: {cleanup_error}')
            if exc_value is not None:
                failure.add_note(f'cleanup followed {type(exc_value).__name__}: {exc_value}')
                raise failure from exc_value
            raise failure from cleanup_error
        return False

    def make_directory(self, relative: str | PurePosixPath, *, mode: int = 0o755) -> None:
        """Create one relative directory and any missing parents through ``openat``."""

        self._require_building()
        path = _safe_relative_path(relative)
        descriptor = os.dup(self._tree_descriptor)
        prefix: list[str] = []
        try:
            for component in path.parts:
                prefix.append(component)
                key = '/'.join(prefix)
                try:
                    metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except OSError as error:
                        raise AtomicDirectoryPublicationError(
                            f'cannot create immutable artifact directory: {key}'
                        ) from error
                    metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                expected = _directory_identity(metadata, f'immutable artifact directory {key}')
                try:
                    child = os.open(component, _directory_open_flags(), dir_fd=descriptor)
                except OSError as error:
                    raise AtomicDirectoryPublicationError(f'cannot open immutable artifact directory: {key}') from error
                os.close(descriptor)
                descriptor = child
                if (
                    _directory_identity(
                        os.fstat(descriptor),
                        f'opened immutable artifact directory {key}',
                    )
                    != expected
                ):
                    raise AtomicDirectoryPublicationError(
                        f'immutable artifact directory changed while being opened: {key}'
                    )
                prior_mode = self._directories.setdefault(key, mode)
                if prior_mode != mode:
                    raise AtomicDirectoryPublicationError(
                        f'immutable artifact directory requested with conflicting modes: {key}'
                    )
        finally:
            os.close(descriptor)

    def write_bytes(
        self,
        relative: str | PurePosixPath,
        payload: bytes | memoryview,
        *,
        mode: int = 0o644,
    ) -> None:
        """Create, fully write, and fsync one new regular file beneath the held root."""

        self._require_building()
        path = _safe_relative_path(relative)
        if not isinstance(payload, (bytes, memoryview)):
            raise TypeError('immutable artifact payload must be exact bytes or a memoryview')
        parent_descriptor = _open_relative_directory(
            self._tree_descriptor,
            path.parent,
        )
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
            try:
                descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
            except OSError as error:
                raise AtomicDirectoryPublicationError(
                    f'cannot create immutable artifact file safely: {path.as_posix()}'
                ) from error
            try:
                view = memoryview(payload)
                offset = 0
                while offset < len(view):
                    try:
                        written = os.write(descriptor, view[offset:])
                    except InterruptedError:
                        continue
                    if written <= 0:
                        raise AtomicDirectoryPublicationError(
                            f'cannot completely write immutable artifact file: {path.as_posix()}'
                        )
                    offset += written
                os.fchmod(descriptor, mode)
                os.fsync(descriptor)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(view):
                    raise AtomicDirectoryPublicationError(
                        f'immutable artifact file changed while being written: {path.as_posix()}'
                    )
            finally:
                os.close(descriptor)
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    def private_tree_path(self) -> Path:
        """Return the checked private path for a read-only semantic preflight.

        Content creation must always use the descriptor-relative methods above.
        Some legacy semantic loaders accept only a path; they may use this path for
        a read-only preflight, then call :meth:`require_private_tree_unchanged`
        before publication. The final published tree must still be reloaded.
        """

        self._require_building()
        path = self.parent / self._container_name / 'owned-tree'
        self._require_private_tree_path_identity(path)
        return path

    def require_private_tree_unchanged(self) -> None:
        """Confirm that a path-only semantic preflight read the owned tree."""

        self._require_building()
        self._require_private_tree_path_identity(self.parent / self._container_name / 'owned-tree')

    def publish(self, *, root_mode: int = 0o755) -> Path:
        """Durably install the complete tree without replacing an existing target."""

        self._require_building()
        self._finalize_directory_modes(root_mode=root_mode)
        _require_open_path_identity(
            self.parent,
            self._parent_identity,
            'immutable output parent changed before publication',
        )
        _require_name_identity(
            self._container_descriptor,
            'owned-tree',
            self._tree_identity,
            'private publication tree changed before publication',
        )
        _require_missing_name(self._parent_descriptor, self.target.name, 'immutable output')
        try:
            _rename_directory_noreplace_at(
                self._container_descriptor,
                'owned-tree',
                self._parent_descriptor,
                self.target.name,
            )
        except FileExistsError as error:
            raise AtomicDirectoryPublicationError(f'immutable output already exists: {self.target}') from error
        except OSError as error:
            raise AtomicDirectoryPublicationError('cannot atomically publish immutable artifact directory') from error
        self._published = True
        try:
            _require_name_identity(
                self._parent_descriptor,
                self.target.name,
                self._tree_identity,
                'published immutable artifact has the wrong kernel identity',
            )
        except AtomicDirectoryPublicationError as error:
            self._restore_unexpected_move(
                source_descriptor=self._parent_descriptor,
                source_name=self.target.name,
                destination_descriptor=self._container_descriptor,
                destination_name='owned-tree',
                label='wrong-identity publication',
            )
            self._published = False
            raise AtomicDirectoryPublicationError(
                'publication source changed during atomic install; unexpected object was restored '
                'and the artifact was not installed'
            ) from error
        os.fsync(self._parent_descriptor)
        os.fsync(self._container_descriptor)
        _require_open_path_identity(
            self.parent,
            self._parent_identity,
            'immutable output parent changed during publication',
        )
        return self.target

    def commit(self) -> None:
        """Retain a successfully published tree after caller-side semantic reload."""

        if self._closed or not self._published or self._committed:
            raise AtomicDirectoryPublicationError(
                'immutable publication can be committed exactly once after publication'
            )
        _require_name_identity(
            self._parent_descriptor,
            self.target.name,
            self._tree_identity,
            'published immutable artifact changed before commit',
        )
        _require_open_path_identity(
            self.parent,
            self._parent_identity,
            'immutable output parent changed before commit',
        )
        self._committed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in (
            self._tree_descriptor,
            self._container_descriptor,
            self._parent_descriptor,
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _require_building(self) -> None:
        if self._closed or self._published or self._committed:
            raise AtomicDirectoryPublicationError('immutable artifact content can only be written before publication')

    def _finalize_directory_modes(self, *, root_mode: int) -> None:
        for relative in sorted(
            self._directories,
            key=lambda value: (-len(PurePosixPath(value).parts), value),
        ):
            descriptor = _open_relative_directory(
                self._tree_descriptor,
                PurePosixPath(relative),
            )
            try:
                os.fchmod(descriptor, self._directories[relative])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fchmod(self._tree_descriptor, root_mode)
        os.fsync(self._tree_descriptor)

    def _require_private_tree_path_identity(self, path: Path) -> None:
        _require_open_path_identity(
            self.parent,
            self._parent_identity,
            'immutable output parent changed during private semantic preflight',
        )
        _require_name_identity(
            self._parent_descriptor,
            self._container_name,
            self._container_identity,
            'private publication container changed during semantic preflight',
        )
        _require_name_identity(
            self._container_descriptor,
            'owned-tree',
            self._tree_identity,
            'private publication tree changed during semantic preflight',
        )
        _require_open_path_identity(
            path,
            self._tree_identity,
            'private publication path changed during semantic preflight',
        )

    def _remove_owned_tree(self) -> None:
        if self._published:
            source_descriptor = self._parent_descriptor
            source_name = self.target.name
            quarantine_name = 'failed-tree'
        else:
            source_descriptor = self._container_descriptor
            source_name = 'owned-tree'
            quarantine_name = 'failed-tree'
        _require_name_identity(
            source_descriptor,
            source_name,
            self._tree_identity,
            'operation-owned immutable tree was replaced; replacement left untouched',
        )
        try:
            os.rename(
                source_name,
                quarantine_name,
                src_dir_fd=source_descriptor,
                dst_dir_fd=self._container_descriptor,
            )
        except OSError as error:
            raise AtomicDirectoryPublicationError('cannot detach operation-owned immutable tree for cleanup') from error
        self._published = False
        try:
            _require_name_identity(
                self._container_descriptor,
                quarantine_name,
                self._tree_identity,
                'operation-owned immutable tree changed during cleanup',
            )
        except AtomicDirectoryPublicationError as error:
            self._restore_unexpected_move(
                source_descriptor=self._container_descriptor,
                source_name=quarantine_name,
                destination_descriptor=source_descriptor,
                destination_name=source_name,
                label='wrong-identity cleanup quarantine',
            )
            raise AtomicDirectoryPublicationError(
                'operation-owned immutable tree changed during cleanup; unrelated replacement was '
                'restored and left untouched'
            ) from error
        try:
            shutil.rmtree(quarantine_name, dir_fd=self._container_descriptor)
        except OSError as error:
            raise AtomicDirectoryPublicationError(
                'operation-owned immutable tree could not be removed and remains quarantined'
            ) from error
        try:
            os.stat(
                quarantine_name,
                dir_fd=self._container_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise AtomicDirectoryPublicationError('operation-owned immutable tree still exists after checked cleanup')
        os.fsync(self._container_descriptor)
        os.fsync(self._parent_descriptor)

    def _restore_unexpected_move(
        self,
        *,
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
        label: str,
    ) -> None:
        """Put a raced replacement back without overwriting another namespace entry."""

        _restore_directory_move_at(
            source_descriptor=source_descriptor,
            source_name=source_name,
            destination_descriptor=destination_descriptor,
            destination_name=destination_name,
            label=label,
        )

    def _remove_private_container(self) -> None:
        try:
            metadata = os.stat(
                self._container_name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise AtomicDirectoryPublicationError(
                'private publication container was renamed; detached empty directory retained'
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != self._container_identity:
            raise AtomicDirectoryPublicationError(
                'private publication container was replaced; replacement left untouched'
            )
        cleanup_name = f'{self._container_name}.cleanup-{secrets.token_hex(16)}'
        try:
            _rename_directory_noreplace_at(
                self._parent_descriptor,
                self._container_name,
                self._parent_descriptor,
                cleanup_name,
            )
        except OSError as error:
            raise AtomicDirectoryPublicationError(
                'private publication container could not be detached safely'
            ) from error
        try:
            _require_name_identity(
                self._parent_descriptor,
                cleanup_name,
                self._container_identity,
                'private publication container changed during cleanup',
            )
        except AtomicDirectoryPublicationError as error:
            _restore_directory_move_at(
                source_descriptor=self._parent_descriptor,
                source_name=cleanup_name,
                destination_descriptor=self._parent_descriptor,
                destination_name=self._container_name,
                label='private publication container',
            )
            raise AtomicDirectoryPublicationError(
                'private publication container changed during cleanup; replacement was restored'
            ) from error
        try:
            os.rmdir(cleanup_name, dir_fd=self._parent_descriptor)
        except OSError as error:
            raise AtomicDirectoryPublicationError(
                'private publication container could not be removed safely'
            ) from error
        os.fsync(self._parent_descriptor)


def _require_descriptor_primitives() -> None:
    required_dir_fd = (os.open, os.mkdir, os.rename, os.rmdir, os.stat)
    if any(function not in os.supports_dir_fd for function in required_dir_fd):
        raise AtomicDirectoryPublicationError(
            'this platform lacks descriptor-relative immutable publication primitives'
        )
    if os.chmod not in os.supports_fd or not shutil.rmtree.avoids_symlink_attacks:
        raise AtomicDirectoryPublicationError('this platform lacks descriptor-safe immutable publication or cleanup')


def _cleanup_partial_publication(
    *,
    parent_descriptor: int,
    container_name: str,
    container_identity: tuple[int, int],
    container_descriptor: int | None,
    tree_identity: tuple[int, int] | None,
) -> None:
    """Best-effort exact cleanup when construction fails before an object can own the descriptors."""

    if container_descriptor is not None:
        try:
            tree_metadata = os.stat(
                'owned-tree',
                dir_fd=container_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            tree_metadata = None
        if tree_metadata is not None:
            if (
                tree_identity is None
                or not stat.S_ISDIR(tree_metadata.st_mode)
                or (
                    tree_metadata.st_dev,
                    tree_metadata.st_ino,
                )
                != tree_identity
            ):
                raise AtomicDirectoryPublicationError(
                    'partial publication tree identity is unavailable or changed; private residue retained'
                )
            quarantine_name = f'failed-initialization-tree-{secrets.token_hex(16)}'
            _rename_directory_noreplace_at(
                container_descriptor,
                'owned-tree',
                container_descriptor,
                quarantine_name,
            )
            try:
                _require_name_identity(
                    container_descriptor,
                    quarantine_name,
                    tree_identity,
                    'partial publication tree changed during cleanup',
                )
            except AtomicDirectoryPublicationError as error:
                _restore_directory_move_at(
                    source_descriptor=container_descriptor,
                    source_name=quarantine_name,
                    destination_descriptor=container_descriptor,
                    destination_name='owned-tree',
                    label='partial publication tree',
                )
                raise AtomicDirectoryPublicationError(
                    'partial publication tree changed during cleanup; replacement was restored'
                ) from error
            shutil.rmtree(quarantine_name, dir_fd=container_descriptor)
            os.fsync(container_descriptor)

    _require_name_identity(
        parent_descriptor,
        container_name,
        container_identity,
        'partial publication container changed before cleanup',
    )
    cleanup_name = f'{container_name}.failed-initialization-{secrets.token_hex(16)}'
    _rename_directory_noreplace_at(
        parent_descriptor,
        container_name,
        parent_descriptor,
        cleanup_name,
    )
    try:
        _require_name_identity(
            parent_descriptor,
            cleanup_name,
            container_identity,
            'partial publication container changed during cleanup',
        )
    except AtomicDirectoryPublicationError as error:
        _restore_directory_move_at(
            source_descriptor=parent_descriptor,
            source_name=cleanup_name,
            destination_descriptor=parent_descriptor,
            destination_name=container_name,
            label='partial publication container',
        )
        raise AtomicDirectoryPublicationError(
            'partial publication container changed during cleanup; replacement was restored'
        ) from error
    try:
        os.rmdir(cleanup_name, dir_fd=parent_descriptor)
    except OSError as error:
        raise AtomicDirectoryPublicationError('partial publication container remains retained') from error
    os.fsync(parent_descriptor)


def _restore_directory_move_at(
    *,
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
    label: str,
) -> None:
    try:
        _rename_directory_noreplace_at(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
    except OSError as error:
        raise AtomicDirectoryPublicationError(
            f'{label} could not be restored safely; unexpected object remains retained'
        ) from error
    os.fsync(source_descriptor)
    if destination_descriptor != source_descriptor:
        os.fsync(destination_descriptor)


def _open_directory_path(path: Path) -> tuple[Path, int]:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = _directory_open_flags()
    descriptor = os.open(requested.anchor, flags)
    try:
        for component in requested.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        _directory_identity(os.fstat(descriptor), 'immutable output parent')
        return requested, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_directory(root_descriptor: int, relative: PurePosixPath) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        if relative == PurePosixPath('.'):
            return descriptor
        for component in relative.parts:
            expected = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            expected_identity = _directory_identity(
                expected,
                f'immutable artifact directory {relative.as_posix()}',
            )
            child = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            if (
                _directory_identity(
                    os.fstat(descriptor),
                    f'opened immutable artifact directory {relative.as_posix()}',
                )
                != expected_identity
            ):
                raise AtomicDirectoryPublicationError(
                    f'immutable artifact directory changed while being opened: {relative.as_posix()}'
                )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _safe_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path in {PurePosixPath('.'), PurePosixPath('')}
        or any(component in {'', '.', '..'} for component in path.parts)
        or path.as_posix() != str(value)
    ):
        raise AtomicDirectoryPublicationError('immutable artifact path must be canonical and relative')
    return path


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)


def _directory_identity(metadata: os.stat_result, label: str) -> tuple[int, int]:
    if not stat.S_ISDIR(metadata.st_mode):
        raise AtomicDirectoryPublicationError(f'{label} is not a directory')
    return metadata.st_dev, metadata.st_ino


def _require_open_path_identity(path: Path, expected: tuple[int, int], message: str) -> None:
    try:
        _resolved, descriptor = _open_directory_path(path)
    except (OSError, AtomicDirectoryPublicationError) as error:
        raise AtomicDirectoryPublicationError(message) from error
    try:
        if _directory_identity(os.fstat(descriptor), message) != expected:
            raise AtomicDirectoryPublicationError(message)
    finally:
        os.close(descriptor)


def _require_name_identity(
    directory_descriptor: int,
    name: str,
    expected: tuple[int, int],
    message: str,
) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as error:
        raise AtomicDirectoryPublicationError(message) from error
    if _directory_identity(metadata, message) != expected:
        raise AtomicDirectoryPublicationError(message)


def _require_missing_name(directory_descriptor: int, name: str, label: str) -> None:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise AtomicDirectoryPublicationError(f'cannot inspect {label} path safely') from error
    raise AtomicDirectoryPublicationError(f'{label} already exists: {name}')


def _create_private_directory_at(parent_descriptor: int, target_name: str) -> str:
    for _attempt in range(128):
        name = f'.{target_name}.vaxreplay-private-{secrets.token_hex(16)}'
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        except OSError as error:
            raise AtomicDirectoryPublicationError('cannot create private immutable publication container') from error
        return name
    raise AtomicDirectoryPublicationError('cannot allocate a unique private immutable publication container')


def _rename_directory_noreplace_at(
    source_descriptor: int,
    source_name: str,
    target_descriptor: int,
    target_name: str,
) -> None:
    libc = CDLL(None, use_errno=True)
    old_path = os.fsencode(source_name)
    new_path = os.fsencode(target_name)
    if sys.platform == 'darwin':
        try:
            rename_exclusive = libc.renameatx_np
        except AttributeError as error:
            raise OSError('this platform lacks descriptor-relative exclusive rename') from error
        rename_exclusive.argtypes = (c_int, c_char_p, c_int, c_char_p, c_uint)
        rename_exclusive.restype = c_int
        set_errno(0)
        result = rename_exclusive(
            source_descriptor,
            old_path,
            target_descriptor,
            new_path,
            0x00000004,
        )
    elif sys.platform.startswith('linux'):
        try:
            rename_exclusive = libc.renameat2
        except AttributeError as error:
            raise OSError('this platform lacks descriptor-relative exclusive rename') from error
        rename_exclusive.argtypes = (c_int, c_char_p, c_int, c_char_p, c_uint)
        rename_exclusive.restype = c_int
        set_errno(0)
        result = rename_exclusive(
            source_descriptor,
            old_path,
            target_descriptor,
            new_path,
            1,
        )
    else:
        raise OSError('this platform lacks descriptor-relative exclusive rename')
    if result != 0:
        error_number = get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number,
                'immutable artifact output already exists',
                target_name,
            )
        raise OSError(error_number, os.strerror(error_number), target_name)


def rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically install ``source`` without ever replacing ``target``.

    A check followed by :func:`os.rename` is not sufficient on POSIX: another
    process can create an empty target directory in the gap and ``rename`` will
    replace it. Official artifact publication therefore requires the native
    exclusive-rename primitive and fails closed on platforms without one.
    """

    libc = CDLL(None, use_errno=True)
    old_path = os.fsencode(source)
    new_path = os.fsencode(target)
    if sys.platform == 'darwin':
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = (c_char_p, c_char_p, c_uint)
        rename_exclusive.restype = c_int
        set_errno(0)
        result = rename_exclusive(old_path, new_path, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith('linux'):
        try:
            rename_exclusive = libc.renameat2
        except AttributeError as error:
            raise OSError('this platform lacks atomic no-replace directory publication') from error
        rename_exclusive.argtypes = (c_int, c_char_p, c_int, c_char_p, c_uint)
        rename_exclusive.restype = c_int
        set_errno(0)
        result = rename_exclusive(-100, old_path, -100, new_path, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif os.name == 'nt':
        # Windows directory rename already fails when the destination exists.
        os.rename(source, target)
        return
    else:
        raise OSError('this platform lacks atomic no-replace directory publication')
    if result != 0:
        error_number = get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, 'immutable artifact output already exists', target)
        raise OSError(error_number, os.strerror(error_number), target)


def rename_file_noreplace(source: Path, target: Path) -> None:
    """Atomically publish one regular file without replacing an existing name.

    The source and destination must be named children of the same already-existing
    directory.  Keeping the rename descriptor-relative closes ancestor/pathname
    substitution races and makes this suitable for create-once journals: a crash
    can leave only the unpublished staging name or the complete final file, never
    a partially written final file.
    """

    requested_source = Path(os.path.abspath(os.fspath(Path(source).expanduser())))
    requested_target = Path(os.path.abspath(os.fspath(Path(target).expanduser())))
    if not requested_source.name or not requested_target.name or requested_source.parent != requested_target.parent:
        raise AtomicDirectoryPublicationError('atomic file publication requires two named children of one directory')
    parent, descriptor = _open_directory_path(requested_source.parent)
    try:
        parent_metadata = os.fstat(descriptor)
        parent_identity = _directory_identity(parent_metadata, 'atomic file publication parent')
        if parent_metadata.st_uid != os.geteuid() or stat.S_IMODE(parent_metadata.st_mode) & 0o022:
            raise AtomicDirectoryPublicationError(
                'atomic file publication parent must be owned and not group/world writable'
            )
        try:
            source_metadata = os.stat(
                requested_source.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise AtomicDirectoryPublicationError('atomic file publication source is unavailable') from error
        if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_nlink != 1:
            raise AtomicDirectoryPublicationError('atomic file publication source must be one regular file')
        source_identity = (source_metadata.st_dev, source_metadata.st_ino)
        _require_missing_name(descriptor, requested_target.name, 'atomic file output')
        try:
            _rename_directory_noreplace_at(
                descriptor,
                requested_source.name,
                descriptor,
                requested_target.name,
            )
        except FileExistsError as error:
            raise AtomicDirectoryPublicationError(f'atomic file output already exists: {requested_target}') from error
        except OSError as error:
            raise AtomicDirectoryPublicationError('cannot atomically publish create-once file') from error
        installed = os.stat(
            requested_target.name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(installed.st_mode)
            or installed.st_nlink != 1
            or (installed.st_dev, installed.st_ino) != source_identity
        ):
            raise AtomicDirectoryPublicationError('atomic file publication installed a different object')
        os.fsync(descriptor)
        _require_open_path_identity(
            parent,
            parent_identity,
            'atomic file publication parent changed during install',
        )
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    """Durably flush directory metadata."""

    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
