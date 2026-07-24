"""Create-once, offline-verifiable runtime closure for the Linux/KVM qualification driver.

The qualification driver is a Python console script.  Hashing that small script does not bind the
interpreter, VaxReplay package, or third-party modules which actually perform the drills.  This
module inventories the complete, explicitly declared Python import roots and records their exact
files, directories, modes, owners, and digests.  Verification performs no imports and executes no
runtime content.

This is an installed-runtime integrity receipt, not a reproducible-build or self-contained-binary
claim.  Native operating-system libraries remain part of the separately qualified host boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

DRIVER_RUNTIME_CLOSURE_MANIFEST_FILE = 'driver-runtime-closure.json'
DRIVER_RUNTIME_CLOSURE_RECEIPT_FILE = 'driver-runtime-closure-receipt.json'
DRIVER_RUNTIME_CLOSURE_SHA256_FILE = 'DRIVER-RUNTIME-CLOSURE.sha256'

DRIVER_RUNTIME_CLOSURE_MANIFEST_SCHEMA_VERSION = 'vaxreplay.qualification-driver-runtime-closure.v0.1'
DRIVER_RUNTIME_CLOSURE_RECEIPT_SCHEMA_VERSION = 'vaxreplay.qualification-driver-runtime-closure-receipt.v0.1'

_CLOSURE_DIGEST_DOMAIN = b'vaxreplay.qualification-driver-runtime-closure.v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_BUNDLE_FILES = frozenset(
    {
        DRIVER_RUNTIME_CLOSURE_MANIFEST_FILE,
        DRIVER_RUNTIME_CLOSURE_RECEIPT_FILE,
        DRIVER_RUNTIME_CLOSURE_SHA256_FILE,
    }
)
_OBSERVATION_SCRIPT = (
    'import json,sys;'
    "print(json.dumps({'implementation':sys.implementation.name,'version':'.'.join(map(str,sys.version_info[:3])),"
    "'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'path':sys.path},"
    "sort_keys=True,separators=(',',':')))"
)


class QualificationDriverRuntimeClosureError(ValueError):
    """A runtime closure was incomplete, mutable, changed, or not externally pinned."""


class QualificationDriverRuntimeClosureEntry(StrictModel):
    path: str = Field(min_length=1, max_length=4096)
    kind: Literal['directory', 'regular_file']
    mode: int = Field(ge=0, le=0o7777)
    uid: int = Field(ge=0)
    gid: int = Field(ge=0)
    link_count: int = Field(ge=1)
    byte_count: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_kind_metadata(self) -> Self:
        is_file = self.kind == 'regular_file'
        if (self.byte_count is not None, self.sha256 is not None) != (is_file, is_file):
            raise ValueError('only regular runtime-closure files carry content metadata')
        if is_file and self.link_count != 1:
            raise ValueError('runtime-closure regular files cannot have hard links')
        return self


class QualificationDriverPythonImportPath(StrictModel):
    path: str = Field(min_length=1, max_length=4096)
    state: Literal['directory', 'regular_file', 'absent']


class QualificationDriverRuntimeClosureManifest(StrictModel):
    schema_version: Literal['vaxreplay.qualification-driver-runtime-closure.v0.1'] = (
        DRIVER_RUNTIME_CLOSURE_MANIFEST_SCHEMA_VERSION
    )
    closure_id: str = Field(min_length=1, max_length=200, pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
    driver_entrypoint_path: str = Field(min_length=1, max_length=4096)
    driver_entrypoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    interpreter_path: str = Field(min_length=1, max_length=4096)
    interpreter_sha256: str = Field(pattern=_SHA256_PATTERN)
    interpreter_implementation: Literal['cpython']
    interpreter_version: str = Field(pattern=r'^3\.[0-9]+\.[0-9]+$')
    interpreter_prefix: str = Field(min_length=1, max_length=4096)
    interpreter_base_prefix: str = Field(min_length=1, max_length=4096)
    interpreter_argv_prefix: tuple[Literal['-I'], Literal['-B']] = ('-I', '-B')
    runtime_roots: tuple[str, ...] = Field(min_length=1, max_length=64)
    python_import_paths: tuple[QualificationDriverPythonImportPath, ...] = Field(min_length=1, max_length=256)
    entries: tuple[QualificationDriverRuntimeClosureEntry, ...] = Field(min_length=2)
    entry_count: int = Field(gt=1)
    expected_uid: int = Field(ge=0)
    expected_gid: int = Field(ge=0)
    tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    complete_declared_import_roots_inventoried: Literal[True] = True
    exact_modes_and_ownership_recorded: Literal[True] = True
    symlinks_allowed: Literal[False] = False
    hardlinked_regular_files_allowed: Literal[False] = False
    special_files_allowed: Literal[False] = False
    native_operating_system_libraries_pinned: Literal[False] = False
    self_contained_executable_claimed: Literal[False] = False
    reproducible_build_claimed: Literal[False] = False

    @field_validator('runtime_roots')
    @classmethod
    def validate_runtime_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError('runtime roots must be unique and sorted')
        return value

    @field_validator('python_import_paths')
    @classmethod
    def validate_import_paths(
        cls, value: tuple[QualificationDriverPythonImportPath, ...]
    ) -> tuple[QualificationDriverPythonImportPath, ...]:
        paths = tuple(item.path for item in value)
        if paths != tuple(sorted(set(paths))):
            raise ValueError('Python import paths must be unique and sorted')
        return value

    @field_validator('entries')
    @classmethod
    def validate_entries(
        cls, value: tuple[QualificationDriverRuntimeClosureEntry, ...]
    ) -> tuple[QualificationDriverRuntimeClosureEntry, ...]:
        paths = tuple(item.path for item in value)
        if paths != tuple(sorted(set(paths))):
            raise ValueError('runtime-closure entries must be unique and sorted by path')
        return value

    @model_validator(mode='after')
    def validate_manifest_bindings(self) -> Self:
        if self.entry_count != len(self.entries):
            raise ValueError('runtime-closure entry count is not derived from its entries')
        if self.tree_sha256 != _entries_sha256(self.entries):
            raise ValueError('runtime-closure tree digest is not derived from its entries')
        by_path = {entry.path: entry for entry in self.entries}
        driver = by_path.get(self.driver_entrypoint_path)
        interpreter = by_path.get(self.interpreter_path)
        if (
            driver is None
            or driver.kind != 'regular_file'
            or driver.sha256 != self.driver_entrypoint_sha256
            or interpreter is None
            or interpreter.kind != 'regular_file'
            or interpreter.sha256 != self.interpreter_sha256
        ):
            raise ValueError('runtime closure does not bind its driver and interpreter entries')
        if not stat.S_IXUSR & driver.mode or not stat.S_IXUSR & interpreter.mode:
            raise ValueError('runtime-closure driver and interpreter must be owner-executable')
        roots = tuple(Path(item) for item in self.runtime_roots)
        for path in (self.driver_entrypoint_path, self.interpreter_path):
            if not _covered_by_roots(Path(path), roots):
                raise ValueError('driver and interpreter must be inside an inventoried runtime root')
        for item in self.python_import_paths:
            if item.state != 'absent' and not _covered_by_roots(Path(item.path), roots):
                raise ValueError('every existing Python import path must be inside an inventoried runtime root')
        return self


class QualificationDriverRuntimeClosureReceipt(StrictModel):
    schema_version: Literal['vaxreplay.qualification-driver-runtime-closure-receipt.v0.1'] = (
        DRIVER_RUNTIME_CLOSURE_RECEIPT_SCHEMA_VERSION
    )
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    builder_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_date_epoch: int = Field(ge=1, le=2**31 - 1)
    closure_id: str = Field(min_length=1, max_length=200)
    entry_count: int = Field(gt=1)
    runtime_root_count: int = Field(gt=0, le=64)
    driver_entrypoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    interpreter_sha256: str = Field(pattern=_SHA256_PATTERN)
    offline_verification_passed_at_publication: Literal[True] = True
    create_once_publication: Literal[True] = True
    installed_runtime_integrity_receipt_only: Literal[True] = True
    self_contained_executable_claimed: Literal[False] = False
    reproducible_build_claimed: Literal[False] = False


class LoadedQualificationDriverRuntimeClosure(StrictModel):
    root: str
    manifest: QualificationDriverRuntimeClosureManifest
    receipt: QualificationDriverRuntimeClosureReceipt
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    closure_sha256: str = Field(pattern=_SHA256_PATTERN)


class _PythonRuntimeObservation(StrictModel):
    implementation: str
    version: str
    executable: str
    prefix: str
    base_prefix: str
    path: tuple[str, ...]


def build_and_retain_qualification_driver_runtime_closure(
    *,
    closure_id: str,
    driver_entrypoint_path: Path,
    interpreter_path: Path,
    runtime_roots: Sequence[Path],
    output_root: Path,
    source_date_epoch: int,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> LoadedQualificationDriverRuntimeClosure:
    """Inventory a fixed installed runtime and publish its manifest/receipt exactly once."""

    entrypoint = _absolute_without_symlink(driver_entrypoint_path, require_exists=True)
    interpreter = _absolute_without_symlink(interpreter_path, require_exists=True)
    roots = tuple(sorted({_absolute_without_symlink(path, require_exists=True) for path in runtime_roots}))
    overlap = any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    )
    if not roots or overlap:
        raise QualificationDriverRuntimeClosureError('runtime roots must be non-overlapping directories')
    if not _covered_by_roots(entrypoint, roots) or not _covered_by_roots(interpreter, roots):
        raise QualificationDriverRuntimeClosureError('driver and interpreter must be inside a runtime root')
    entries_before_observation = _inventory_roots(
        roots,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    observation = _observe_python_runtime(interpreter)
    if observation.implementation != 'cpython' or not observation.version.startswith('3.'):
        raise QualificationDriverRuntimeClosureError('qualification runtime must use an observed CPython 3 interpreter')
    try:
        observed_executable = _absolute_without_symlink(Path(observation.executable), require_exists=True)
    except QualificationDriverRuntimeClosureError as error:
        raise QualificationDriverRuntimeClosureError('observed Python executable is unsafe') from error
    if observed_executable != interpreter:
        raise QualificationDriverRuntimeClosureError('observed Python executable differs from the selected interpreter')
    import_paths = _classify_import_paths(observation.path, roots=roots)
    entries = _inventory_roots(roots, expected_uid=expected_uid, expected_gid=expected_gid)
    if entries != entries_before_observation:
        raise QualificationDriverRuntimeClosureError(
            'selected Python runtime mutated its installed tree during isolated observation'
        )
    if not _entry_has_exact_shebang(entrypoint, interpreter):
        raise QualificationDriverRuntimeClosureError('driver entrypoint shebang differs from the selected interpreter')
    manifest = QualificationDriverRuntimeClosureManifest(
        closure_id=closure_id,
        driver_entrypoint_path=str(entrypoint),
        driver_entrypoint_sha256=_entry_sha256(entries, entrypoint),
        interpreter_path=str(interpreter),
        interpreter_sha256=_entry_sha256(entries, interpreter),
        interpreter_implementation='cpython',
        interpreter_version=observation.version,
        interpreter_prefix=observation.prefix,
        interpreter_base_prefix=observation.base_prefix,
        runtime_roots=tuple(str(root) for root in roots),
        python_import_paths=import_paths,
        entries=entries,
        entry_count=len(entries),
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        tree_sha256=_entries_sha256(entries),
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    receipt = QualificationDriverRuntimeClosureReceipt(
        manifest_sha256=manifest_sha256,
        builder_source_sha256=qualification_driver_runtime_closure_builder_source_sha256(),
        source_date_epoch=source_date_epoch,
        closure_id=manifest.closure_id,
        entry_count=manifest.entry_count,
        runtime_root_count=len(manifest.runtime_roots),
        driver_entrypoint_sha256=manifest.driver_entrypoint_sha256,
        interpreter_sha256=manifest.interpreter_sha256,
    )
    receipt_bytes = canonical_json_bytes(receipt)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    closure_sha256 = qualification_driver_runtime_closure_sha256(manifest_bytes, receipt_bytes)
    _publish_bundle(
        output_root,
        manifest_bytes=manifest_bytes,
        receipt_bytes=receipt_bytes,
        closure_sha256=closure_sha256,
    )
    return verify_qualification_driver_runtime_closure(
        output_root,
        expected_manifest_sha256=manifest_sha256,
        expected_receipt_sha256=receipt_sha256,
        expected_closure_sha256=closure_sha256,
        require_root_owned=expected_uid == 0 and expected_gid == 0,
    )


def verify_qualification_driver_runtime_closure(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
    expected_closure_sha256: str,
    require_root_owned: bool = True,
) -> LoadedQualificationDriverRuntimeClosure:
    """Verify the bundle and every installed runtime byte without executing the runtime."""

    for value, label in (
        (expected_manifest_sha256, 'manifest'),
        (expected_receipt_sha256, 'receipt'),
        (expected_closure_sha256, 'closure'),
    ):
        _require_sha256(value, label=label)
    resolved = _validate_bundle_root(root, require_root_owned=require_root_owned)
    manifest_bytes = _read_regular_file(
        resolved / DRIVER_RUNTIME_CLOSURE_MANIFEST_FILE,
        _MAX_MANIFEST_BYTES,
        require_root_owned=require_root_owned,
    )
    receipt_bytes = _read_regular_file(
        resolved / DRIVER_RUNTIME_CLOSURE_RECEIPT_FILE,
        _MAX_RECEIPT_BYTES,
        require_root_owned=require_root_owned,
    )
    digest_bytes = _read_regular_file(
        resolved / DRIVER_RUNTIME_CLOSURE_SHA256_FILE,
        65,
        require_root_owned=require_root_owned,
    )
    try:
        manifest = QualificationDriverRuntimeClosureManifest.model_validate_json(manifest_bytes)
        receipt = QualificationDriverRuntimeClosureReceipt.model_validate_json(receipt_bytes)
    except ValueError as error:
        raise QualificationDriverRuntimeClosureError('runtime-closure bundle contains invalid data') from error
    if canonical_json_bytes(manifest) != manifest_bytes or canonical_json_bytes(receipt) != receipt_bytes:
        raise QualificationDriverRuntimeClosureError('runtime-closure bundle must contain canonical JSON')
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    closure_sha256 = qualification_driver_runtime_closure_sha256(manifest_bytes, receipt_bytes)
    if not (
        hmac.compare_digest(manifest_sha256, expected_manifest_sha256)
        and hmac.compare_digest(receipt_sha256, expected_receipt_sha256)
        and hmac.compare_digest(closure_sha256, expected_closure_sha256)
        and hmac.compare_digest(digest_bytes, (closure_sha256 + '\n').encode('ascii'))
    ):
        raise QualificationDriverRuntimeClosureError('runtime-closure bundle differs from an external digest pin')
    if (
        receipt.manifest_sha256 != manifest_sha256
        or receipt.closure_id != manifest.closure_id
        or receipt.entry_count != manifest.entry_count
        or receipt.runtime_root_count != len(manifest.runtime_roots)
        or receipt.driver_entrypoint_sha256 != manifest.driver_entrypoint_sha256
        or receipt.interpreter_sha256 != manifest.interpreter_sha256
    ):
        raise QualificationDriverRuntimeClosureError('runtime-closure receipt differs from its manifest')
    if require_root_owned and (manifest.expected_uid, manifest.expected_gid) != (0, 0):
        raise QualificationDriverRuntimeClosureError('production runtime closure must require root ownership')
    _verify_installed_tree(manifest)
    return LoadedQualificationDriverRuntimeClosure(
        root=str(resolved),
        manifest=manifest,
        receipt=receipt,
        manifest_sha256=manifest_sha256,
        receipt_sha256=receipt_sha256,
        closure_sha256=closure_sha256,
    )


def qualification_driver_runtime_closure_sha256(manifest_bytes: bytes, receipt_bytes: bytes) -> str:
    return hashlib.sha256(_CLOSURE_DIGEST_DOMAIN + manifest_bytes + receipt_bytes).hexdigest()


def qualification_driver_runtime_closure_builder_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _observe_python_runtime(interpreter: Path) -> _PythonRuntimeObservation:
    try:
        completed = subprocess.run(
            [str(interpreter), '-I', '-B', '-c', _OBSERVATION_SCRIPT],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd='/',
            env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'},
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        raise QualificationDriverRuntimeClosureError('cannot observe the selected Python runtime') from None
    if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
        raise QualificationDriverRuntimeClosureError('selected Python runtime observation failed')
    try:
        observation = _PythonRuntimeObservation.model_validate_json(completed.stdout)
    except ValueError as error:
        raise QualificationDriverRuntimeClosureError('selected Python runtime returned invalid metadata') from error
    return observation


def _classify_import_paths(
    values: Sequence[str], *, roots: tuple[Path, ...]
) -> tuple[QualificationDriverPythonImportPath, ...]:
    result: list[QualificationDriverPythonImportPath] = []
    for raw in sorted(set(values)):
        path = Path(raw)
        if not path.is_absolute():
            raise QualificationDriverRuntimeClosureError('Python import paths must be absolute under isolated mode')
        _assert_no_symlink_components(path, allow_missing_final=True)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            result.append(QualificationDriverPythonImportPath(path=str(path), state='absent'))
            continue
        state: Literal['directory', 'regular_file', 'absent']
        if stat.S_ISDIR(metadata.st_mode):
            state = 'directory'
        elif stat.S_ISREG(metadata.st_mode):
            state = 'regular_file'
        else:
            raise QualificationDriverRuntimeClosureError('Python import path is not a regular file or directory')
        if not _covered_by_roots(path, roots):
            raise QualificationDriverRuntimeClosureError(
                'every existing isolated Python import path must be covered by a runtime root'
            )
        result.append(QualificationDriverPythonImportPath(path=str(path), state=state))
    if not result:
        raise QualificationDriverRuntimeClosureError('isolated Python runtime returned no import paths')
    return tuple(result)


def _inventory_roots(
    roots: tuple[Path, ...], *, expected_uid: int, expected_gid: int
) -> tuple[QualificationDriverRuntimeClosureEntry, ...]:
    result: list[QualificationDriverRuntimeClosureEntry] = []
    inodes: set[tuple[int, int]] = set()
    seen_paths: set[str] = set()
    for root in roots:
        for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            directory_names.sort()
            file_names.sort()
            paths = [Path(directory), *(Path(directory) / name for name in directory_names + file_names)]
            for path in paths:
                path_string = str(path)
                if path_string in seen_paths:
                    continue
                seen_paths.add(path_string)
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise QualificationDriverRuntimeClosureError('runtime closure cannot contain symbolic links')
                if (metadata.st_uid, metadata.st_gid) != (expected_uid, expected_gid):
                    raise QualificationDriverRuntimeClosureError('runtime closure has unexpected ownership')
                mode = stat.S_IMODE(metadata.st_mode)
                if mode & 0o022:
                    raise QualificationDriverRuntimeClosureError('runtime closure cannot be group/world writable')
                if stat.S_ISDIR(metadata.st_mode):
                    result.append(
                        QualificationDriverRuntimeClosureEntry(
                            path=path_string,
                            kind='directory',
                            mode=mode,
                            uid=metadata.st_uid,
                            gid=metadata.st_gid,
                            link_count=metadata.st_nlink,
                        )
                    )
                elif stat.S_ISREG(metadata.st_mode):
                    inode = (metadata.st_dev, metadata.st_ino)
                    if metadata.st_nlink != 1 or inode in inodes:
                        raise QualificationDriverRuntimeClosureError(
                            'runtime closure cannot contain hardlinked regular files'
                        )
                    inodes.add(inode)
                    result.append(
                        QualificationDriverRuntimeClosureEntry(
                            path=path_string,
                            kind='regular_file',
                            mode=mode,
                            uid=metadata.st_uid,
                            gid=metadata.st_gid,
                            link_count=metadata.st_nlink,
                            byte_count=metadata.st_size,
                            sha256=_sha256_file(path),
                        )
                    )
                else:
                    raise QualificationDriverRuntimeClosureError('runtime closure cannot contain special files')
    return tuple(sorted(result, key=lambda item: item.path))


def _verify_installed_tree(manifest: QualificationDriverRuntimeClosureManifest) -> None:
    roots = tuple(Path(item) for item in manifest.runtime_roots)
    observed = _inventory_roots(roots, expected_uid=manifest.expected_uid, expected_gid=manifest.expected_gid)
    if observed != manifest.entries:
        raise QualificationDriverRuntimeClosureError('installed runtime tree differs from its closure manifest')
    for item in manifest.python_import_paths:
        path = Path(item.path)
        _assert_no_symlink_components(path, allow_missing_final=item.state == 'absent')
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            observed_state = 'absent'
        else:
            observed_state = (
                'directory'
                if stat.S_ISDIR(metadata.st_mode)
                else 'regular_file'
                if stat.S_ISREG(metadata.st_mode)
                else 'invalid'
            )
        if observed_state != item.state:
            raise QualificationDriverRuntimeClosureError('Python import search path state changed after publication')
    entrypoint = Path(manifest.driver_entrypoint_path)
    interpreter = Path(manifest.interpreter_path)
    if not _entry_has_exact_shebang(entrypoint, interpreter):
        raise QualificationDriverRuntimeClosureError('driver entrypoint shebang differs from the pinned interpreter')


def _entry_has_exact_shebang(entrypoint: Path, interpreter: Path) -> bool:
    try:
        with entrypoint.open('rb') as handle:
            first_line = handle.readline(4097)
    except OSError:
        return False
    bare = f'#!{interpreter}\n'.encode()
    isolated = f'#!{interpreter} -IB\n'.encode()
    return first_line in {bare, isolated}


def _absolute_without_symlink(path: Path, *, require_exists: bool) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = expanded.absolute()
    _assert_no_symlink_components(expanded, allow_missing_final=not require_exists)
    if require_exists:
        try:
            expanded.lstat()
        except OSError as error:
            raise QualificationDriverRuntimeClosureError('runtime-closure path is unavailable') from error
    return expanded


def _assert_no_symlink_components(path: Path, *, allow_missing_final: bool) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if allow_missing_final and index == len(parts) - 1:
                return
            raise QualificationDriverRuntimeClosureError('runtime-closure path component is missing') from None
        if stat.S_ISLNK(metadata.st_mode):
            raise QualificationDriverRuntimeClosureError('runtime-closure paths cannot traverse symbolic links')


def _covered_by_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _entry_sha256(entries: tuple[QualificationDriverRuntimeClosureEntry, ...], path: Path) -> str:
    for entry in entries:
        if entry.path == str(path) and entry.sha256 is not None:
            return entry.sha256
    raise QualificationDriverRuntimeClosureError('runtime file is absent from the closure inventory')


def _entries_sha256(entries: tuple[QualificationDriverRuntimeClosureEntry, ...]) -> str:
    serializable = [entry.model_dump(mode='json') for entry in entries]
    return hashlib.sha256(canonical_json_bytes(serializable)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags)
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(descriptor)


def _publish_bundle(root: Path, *, manifest_bytes: bytes, receipt_bytes: bytes, closure_sha256: str) -> None:
    target = root.expanduser().absolute()
    if target.is_symlink() or target.exists():
        raise QualificationDriverRuntimeClosureError('runtime-closure output already exists or is a symlink')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    staging.chmod(0o700)
    try:
        for name, content in (
            (DRIVER_RUNTIME_CLOSURE_MANIFEST_FILE, manifest_bytes),
            (DRIVER_RUNTIME_CLOSURE_RECEIPT_FILE, receipt_bytes),
            (DRIVER_RUNTIME_CLOSURE_SHA256_FILE, (closure_sha256 + '\n').encode('ascii')),
        ):
            path = staging / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, content)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except FileExistsError as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise QualificationDriverRuntimeClosureError('runtime-closure output already exists') from error
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_bundle_root(root: Path, *, require_root_owned: bool) -> Path:
    expanded = root.expanduser()
    if expanded.is_symlink():
        raise QualificationDriverRuntimeClosureError('runtime-closure bundle root cannot be a symbolic link')
    try:
        resolved = expanded.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError as error:
        raise QualificationDriverRuntimeClosureError('runtime-closure bundle root is unavailable') from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise QualificationDriverRuntimeClosureError('runtime-closure bundle root must be a private directory')
    if require_root_owned and (metadata.st_uid, metadata.st_gid) != (0, 0):
        raise QualificationDriverRuntimeClosureError('production runtime-closure bundle must be root-owned')
    if {entry.name for entry in os.scandir(resolved)} != _BUNDLE_FILES:
        raise QualificationDriverRuntimeClosureError('runtime-closure bundle has an unexpected file inventory')
    return resolved


def _read_regular_file(path: Path, maximum_bytes: int, *, require_root_owned: bool) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum_bytes
            or (require_root_owned and (metadata.st_uid, metadata.st_gid) != (0, 0))
        ):
            raise QualificationDriverRuntimeClosureError('runtime-closure bundle file is unsafe')
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b''.join(chunks)
        if len(content) > maximum_bytes:
            raise QualificationDriverRuntimeClosureError('runtime-closure bundle file is oversized')
        return content
    finally:
        os.close(descriptor)


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise QualificationDriverRuntimeClosureError(f'expected {label} digest is not lowercase SHA-256')


__all__ = [
    'DRIVER_RUNTIME_CLOSURE_MANIFEST_FILE',
    'DRIVER_RUNTIME_CLOSURE_RECEIPT_FILE',
    'DRIVER_RUNTIME_CLOSURE_SHA256_FILE',
    'LoadedQualificationDriverRuntimeClosure',
    'QualificationDriverRuntimeClosureError',
    'QualificationDriverRuntimeClosureManifest',
    'QualificationDriverRuntimeClosureReceipt',
    'build_and_retain_qualification_driver_runtime_closure',
    'qualification_driver_runtime_closure_builder_source_sha256',
    'qualification_driver_runtime_closure_sha256',
    'verify_qualification_driver_runtime_closure',
]
