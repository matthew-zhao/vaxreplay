"""Single-host durable store for prospective source-capture operations.

The SQLite database coordinates immutable job revisions, deterministic schedule slots,
leased attempts, and an append-only event hash chain.  Exact bytes live in a filesystem
SHA-256 content-addressed store.  Local durability and hashes are intentionally not
presented as independent timestamp evidence; :meth:`checkpoint` emits the small target
that a separately operated witness must timestamp.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import secrets
import sqlite3
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Self

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.policy import (
    IMMPORT_AUTHENTICATED_COLLECTOR_ID,
    STATIC_HTTPS_COLLECTOR_ID,
    parse_immport_authenticated_job_configuration,
    parse_static_job_configuration,
)
from vaxreplay.operations.schema import (
    ARTIFACT_ROLE_PATTERN,
    SAFE_ID_PATTERN,
    AttemptLease,
    AttemptState,
    CaptureJobSpec,
    LedgerCheckpoint,
    LedgerEvent,
    LedgerEventType,
    LogicalRunRecord,
    LogicalRunState,
    RegisteredJob,
    StoredArtifact,
    StoreVerificationReport,
    aware_utc,
    job_spec_sha256,
    ledger_event_sha256,
    scheduled_logical_run_id,
)

_CHUNK_SIZE = 1024 * 1024
_DEFAULT_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
_DB_NAME = 'operations.sqlite3'
_SCHEMA_VERSION = 'vaxreplay.operations-store.v0.1'
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
)


class OperationsStoreError(RuntimeError):
    """Base class for operational-store failures."""


class OperationsIntegrityError(OperationsStoreError):
    """Durable state does not match its committed hashes or invariants."""


class LeaseConflictError(OperationsStoreError):
    """A different live lease owns the logical run or attempt."""


class AttemptStateError(OperationsStoreError):
    """The requested transition is invalid for the attempt's current state."""


class RunAlreadySucceededError(OperationsStoreError):
    """A logical run already has its unique successful terminal attempt."""


class AttemptBudgetExhaustedError(OperationsStoreError):
    """The immutable retry budget denies another attempt for this logical run."""


def _timestamp(value: datetime, field_name: str) -> str:
    return aware_utc(value, field_name).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _json_timestamp(value: datetime, field_name: str) -> str:
    return aware_utc(value, field_name).isoformat().replace('+00:00', 'Z')


def _parse_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (TypeError, ValueError) as error:
        raise OperationsIntegrityError(f'invalid persisted {field_name}') from error
    return aware_utc(parsed, field_name)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _validate_role(role: str) -> str:
    if not re.fullmatch(ARTIFACT_ROLE_PATTERN, role):
        raise ValueError('artifact role must use lowercase portable identifier syntax')
    return role


def _validate_safe_id(value: str, field_name: str) -> str:
    if not re.fullmatch(SAFE_ID_PATTERN, value):
        raise ValueError(f'{field_name} must use portable operational identifier syntax')
    return value


def _open_directory(path: Path, label: str) -> int:
    try:
        descriptor = os.open(path, _DIRECTORY_FLAGS)
    except OSError as error:
        raise OperationsIntegrityError(f'{label} is not a safe directory') from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OperationsIntegrityError(f'{label} is not a directory')
    return descriptor


def _open_directory_at(parent_descriptor: int, name: str, label: str) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
    except OSError as error:
        raise OperationsIntegrityError(f'{label} is not a safe directory') from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OperationsIntegrityError(f'{label} is not a directory')
    return descriptor


def _fsync_directory(path: Path, label: str) -> None:
    descriptor = _open_directory(path, label)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_objects_root(root: Path) -> int:
    root_descriptor = _open_directory(root, 'operations root')
    objects_descriptor: int | None = None
    try:
        objects_descriptor = _open_directory_at(root_descriptor, 'objects', 'CAS objects directory')
        return _open_directory_at(objects_descriptor, 'sha256', 'CAS SHA-256 directory')
    finally:
        if objects_descriptor is not None:
            os.close(objects_descriptor)
        os.close(root_descriptor)


class OperationalStore:
    """Durable single-host operations store.

    Instances are cheap handles; each operation opens its own SQLite connection so the
    same instance may be used by independent worker threads or processes.
    """

    def __init__(
        self,
        root: Path,
        *,
        trusted_lease_clock: Callable[[], datetime] | None = _now_utc,
    ) -> None:
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise OperationsIntegrityError('operations root cannot be a symbolic link')
        self.root = requested.resolve()
        self.database_path = self.root / _DB_NAME
        self.objects_root = self.root / 'objects' / 'sha256'
        self._trusted_lease_clock = trusted_lease_clock
        if not self.database_path.is_file():
            raise OperationsStoreError(f'operations store is not initialized: {self.database_path}')
        if self.database_path.is_symlink():
            raise OperationsIntegrityError('operations database must be a regular local path')
        objects_descriptor = _open_objects_root(self.root)
        os.close(objects_descriptor)
        with self._connect() as connection:
            self.store_id = self._metadata(connection, 'store_id')
            if self._metadata(connection, 'schema_version') != _SCHEMA_VERSION:
                raise OperationsIntegrityError('unsupported operations-store schema version')

    @classmethod
    def initialize(
        cls,
        root: Path,
        *,
        created_at: datetime | None = None,
        store_id: str | None = None,
        trusted_lease_clock: Callable[[], datetime] | None = _now_utc,
    ) -> Self:
        """Create a new empty store without replacing any existing database."""

        created_at = aware_utc(created_at or _now_utc(), 'created_at')
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise OperationsStoreError('operations root cannot be a symbolic link')
        requested.mkdir(parents=True, exist_ok=True)
        requested = requested.resolve()
        database_path = requested / _DB_NAME
        root_descriptor = _open_directory(requested, 'operations root')
        objects_descriptor: int | None = None
        sha256_descriptor: int | None = None
        try:
            try:
                os.mkdir('objects', mode=0o750, dir_fd=root_descriptor)
            except FileExistsError:
                pass
            objects_descriptor = _open_directory_at(root_descriptor, 'objects', 'CAS objects directory')
            try:
                os.mkdir('sha256', mode=0o750, dir_fd=objects_descriptor)
            except FileExistsError:
                pass
            sha256_descriptor = _open_directory_at(objects_descriptor, 'sha256', 'CAS SHA-256 directory')
            os.fsync(sha256_descriptor)
            os.fsync(objects_descriptor)
            os.fsync(root_descriptor)
        finally:
            if sha256_descriptor is not None:
                os.close(sha256_descriptor)
            if objects_descriptor is not None:
                os.close(objects_descriptor)
            os.close(root_descriptor)

        selected_store_id = store_id or secrets.token_hex(16)
        if len(selected_store_id) != 32 or any(character not in '0123456789abcdef' for character in selected_store_id):
            raise ValueError('store_id must be 32 lowercase hexadecimal characters')
        try:
            claim_descriptor = os.open(database_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise OperationsStoreError(f'operations database already exists: {database_path}') from error
        os.close(claim_descriptor)
        try:
            connection = sqlite3.connect(database_path, isolation_level=None)
            try:
                journal_mode = connection.execute('PRAGMA journal_mode=WAL').fetchone()
                if journal_mode is None or str(journal_mode[0]).lower() != 'wal':
                    raise OperationsIntegrityError('operations store requires SQLite WAL journal mode')
                connection.execute('PRAGMA synchronous=FULL')
                connection.execute('PRAGMA foreign_keys=ON')
                connection.executescript(_SCHEMA_SQL)
                connection.execute('BEGIN IMMEDIATE')
                connection.executemany(
                    'INSERT INTO metadata(key, value) VALUES (?, ?)',
                    (('schema_version', _SCHEMA_VERSION), ('store_id', selected_store_id)),
                )
                cls._append_event_static(
                    connection,
                    LedgerEventType.STORE_INITIALIZED,
                    created_at,
                    {'schema_version': _SCHEMA_VERSION, 'store_id': selected_store_id},
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
            _fsync_directory(requested, 'operations root')
        except BaseException:
            database_path.unlink(missing_ok=True)
            (requested / f'{_DB_NAME}-wal').unlink(missing_ok=True)
            (requested / f'{_DB_NAME}-shm').unlink(missing_ok=True)
            _fsync_directory(requested, 'operations root')
            raise
        return cls(requested, trusted_lease_clock=trusted_lease_clock)

    def _lease_operation_time(self, supplied: datetime | None, field_name: str = 'now') -> datetime:
        """Use the host clock for lease authority; ``None`` is test-fixture-only."""

        if self._trusted_lease_clock is None:
            return aware_utc(supplied or _now_utc(), field_name)
        try:
            observed = self._trusted_lease_clock()
        except Exception as error:
            raise OperationsStoreError('trusted lease clock failed') from error
        try:
            return aware_utc(observed, field_name)
        except (AttributeError, TypeError, ValueError) as error:
            raise OperationsStoreError('trusted lease clock returned an invalid timestamp') from error

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        journal_mode = connection.execute('PRAGMA journal_mode').fetchone()
        if journal_mode is None or str(journal_mode[0]).lower() != 'wal':
            connection.close()
            raise OperationsIntegrityError('operations store is not using required SQLite WAL journal mode')
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('PRAGMA busy_timeout=30000')
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _metadata(connection: sqlite3.Connection, key: str) -> str:
        row = connection.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
        if row is None:
            raise OperationsIntegrityError(f'missing operations metadata: {key}')
        return str(row['value'])

    @staticmethod
    def _append_event_static(
        connection: sqlite3.Connection,
        event_type: LedgerEventType,
        occurred_at: datetime,
        payload: dict[str, str | int | bool | None],
    ) -> LedgerEvent:
        previous = connection.execute(
            'SELECT sequence, event_sha256 FROM events ORDER BY sequence DESC LIMIT 1'
        ).fetchone()
        sequence = 1 if previous is None else int(previous['sequence']) + 1
        previous_sha256 = None if previous is None else str(previous['event_sha256'])
        occurred_at = aware_utc(occurred_at, 'occurred_at')
        hash_preimage = {
            'event_type': event_type.value,
            'occurred_at': _json_timestamp(occurred_at, 'occurred_at'),
            'payload': payload,
            'previous_event_sha256': previous_sha256,
            'schema_version': 'vaxreplay.operations-ledger-event.v0.1',
            'sequence': sequence,
        }
        event_sha256 = hashlib.sha256(canonical_json_bytes(hash_preimage)).hexdigest()
        event = LedgerEvent(
            sequence=sequence,
            event_type=event_type,
            occurred_at=occurred_at,
            previous_event_sha256=previous_sha256,
            payload=payload,
            event_sha256=event_sha256,
        )
        event_bytes = canonical_json_bytes(event)
        connection.execute(
            'INSERT INTO events(sequence, event_type, occurred_at, previous_event_sha256, event_sha256, event_json) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (
                sequence,
                event_type.value,
                _timestamp(event.occurred_at, 'occurred_at'),
                previous_sha256,
                event_sha256,
                event_bytes,
            ),
        )
        return event

    def _append_event(
        self,
        connection: sqlite3.Connection,
        event_type: LedgerEventType,
        occurred_at: datetime,
        payload: dict[str, str | int | bool | None],
    ) -> LedgerEvent:
        return self._append_event_static(connection, event_type, occurred_at, payload)

    def put_bytes(
        self,
        payload: bytes,
        *,
        recorded_at: datetime | None = None,
        max_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
    ) -> StoredArtifact:
        """Durably install exact bytes into CAS and append their first-record event."""

        return self.put_stream(io.BytesIO(payload), recorded_at=recorded_at, max_bytes=max_bytes)

    def put_file(
        self,
        path: Path,
        *,
        recorded_at: datetime | None = None,
        max_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
    ) -> StoredArtifact:
        """Snapshot one regular non-symlink file into CAS while detecting mutation."""

        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise OperationsStoreError('artifact input cannot be a symbolic link')
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
        try:
            descriptor = os.open(requested, flags)
        except OSError as error:
            raise OperationsStoreError(f'cannot open artifact input: {requested}') from error
        if max_bytes < 0:
            os.close(descriptor)
            raise ValueError('max_bytes must be nonnegative')
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise OperationsStoreError('artifact input must be a regular file')
            copied_bytes = 0
            with os.fdopen(descriptor, 'rb', closefd=False) as source, tempfile.TemporaryFile(mode='w+b') as staging:
                while chunk := source.read(_CHUNK_SIZE):
                    copied_bytes += len(chunk)
                    if copied_bytes > max_bytes:
                        raise OperationsStoreError(f'artifact exceeds max_bytes={max_bytes}')
                    staging.write(chunk)
                staging.flush()
                after = os.fstat(descriptor)
                identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                if identity_before != identity_after or copied_bytes != before.st_size:
                    raise OperationsStoreError('artifact input changed while it was being snapshotted')
                staging.seek(0)
                return self.put_stream(staging, recorded_at=recorded_at, max_bytes=max_bytes)
        finally:
            os.close(descriptor)

    def put_stream(
        self,
        source: BinaryIO,
        *,
        recorded_at: datetime | None = None,
        max_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
    ) -> StoredArtifact:
        """Stream into same-filesystem temporary storage, hash, then publish exclusively."""

        if max_bytes < 0:
            raise ValueError('max_bytes must be nonnegative')
        recorded_at = aware_utc(recorded_at or _now_utc(), 'recorded_at')
        objects_descriptor = _open_objects_root(self.root)
        temporary_directory_descriptor: int | None = None
        shard_descriptor: int | None = None
        temporary_name: str | None = None
        digest = hashlib.sha256()
        byte_count = 0
        try:
            try:
                os.mkdir('.tmp', mode=0o700, dir_fd=objects_descriptor)
            except FileExistsError:
                pass
            os.fsync(objects_descriptor)
            temporary_directory_descriptor = _open_directory_at(
                objects_descriptor,
                '.tmp',
                'CAS temporary directory',
            )
            temporary_name = f'object-{secrets.token_hex(16)}'
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
                0o600,
                dir_fd=temporary_directory_descriptor,
            )
            with os.fdopen(descriptor, 'wb') as destination:
                while True:
                    chunk = source.read(_CHUNK_SIZE)
                    if not isinstance(chunk, bytes):
                        raise OperationsStoreError('artifact stream must return bytes')
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > max_bytes:
                        raise OperationsStoreError(f'artifact exceeds max_bytes={max_bytes}')
                    destination.write(chunk)
                    digest.update(chunk)
                destination.flush()
                os.fsync(destination.fileno())
                os.fchmod(destination.fileno(), 0o440)
            sha256 = digest.hexdigest()
            shard_name = sha256[:2]
            try:
                os.mkdir(shard_name, mode=0o750, dir_fd=objects_descriptor)
            except FileExistsError:
                pass
            os.fsync(objects_descriptor)
            shard_descriptor = _open_directory_at(
                objects_descriptor,
                shard_name,
                'CAS digest shard',
            )
            try:
                os.link(
                    temporary_name,
                    sha256,
                    src_dir_fd=temporary_directory_descriptor,
                    dst_dir_fd=shard_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                try:
                    existing_descriptor = os.open(
                        sha256,
                        os.O_RDONLY
                        | getattr(os, 'O_NOFOLLOW', 0)
                        | getattr(os, 'O_NONBLOCK', 0)
                        | getattr(os, 'O_CLOEXEC', 0),
                        dir_fd=shard_descriptor,
                    )
                except OSError as error:
                    raise OperationsIntegrityError('cannot safely open existing CAS object') from error
                self._verify_object_descriptor(
                    existing_descriptor,
                    sha256,
                    byte_count,
                    label=f'objects/sha256/{shard_name}/{sha256}',
                )
            # Every writer crossing the CAS-to-ledger boundary fsyncs the directory.
            # This closes the race where a deduplicating writer observes a newly linked
            # object and commits its event before the installer has made the link durable.
            os.fsync(shard_descriptor)
            os.unlink(temporary_name, dir_fd=temporary_directory_descriptor)
            temporary_name = None
            relative_path = f'objects/sha256/{shard_name}/{sha256}'
            with self._connect() as connection:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    existing = connection.execute(
                        'SELECT sha256, byte_count, relative_path, first_recorded_at FROM artifacts WHERE sha256 = ?',
                        (sha256,),
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            'INSERT INTO artifacts(sha256, byte_count, relative_path, first_recorded_at) '
                            'VALUES (?, ?, ?, ?)',
                            (sha256, byte_count, relative_path, _timestamp(recorded_at, 'recorded_at')),
                        )
                        self._append_event(
                            connection,
                            LedgerEventType.ARTIFACT_STORED,
                            recorded_at,
                            {'artifact_sha256': sha256, 'byte_count': byte_count},
                        )
                        artifact = StoredArtifact(
                            sha256=sha256,
                            byte_count=byte_count,
                            relative_path=relative_path,
                            first_recorded_at=recorded_at,
                        )
                    else:
                        artifact = self._artifact_from_row(existing)
                        if artifact.byte_count != byte_count or artifact.relative_path != relative_path:
                            raise OperationsIntegrityError('artifact metadata conflicts with its content address')
                    connection.commit()
                    return artifact
                except BaseException:
                    connection.rollback()
                    raise
        finally:
            if temporary_name is not None and temporary_directory_descriptor is not None:
                try:
                    os.unlink(temporary_name, dir_fd=temporary_directory_descriptor)
                except FileNotFoundError:
                    pass
            if shard_descriptor is not None:
                os.close(shard_descriptor)
            if temporary_directory_descriptor is not None:
                os.close(temporary_directory_descriptor)
            os.close(objects_descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _open_cas_object(self, sha256: str) -> int:
        objects_descriptor = _open_objects_root(self.root)
        shard_descriptor: int | None = None
        try:
            shard_descriptor = _open_directory_at(
                objects_descriptor,
                sha256[:2],
                'CAS digest shard',
            )
            flags = (
                os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0)
            )
            try:
                return os.open(sha256, flags, dir_fd=shard_descriptor)
            except OSError as error:
                raise OperationsIntegrityError(f'cannot safely open CAS object: {sha256}') from error
        finally:
            if shard_descriptor is not None:
                os.close(shard_descriptor)
            os.close(objects_descriptor)

    def _verify_artifact_object(self, artifact: StoredArtifact) -> None:
        try:
            descriptor = self._open_cas_object(artifact.sha256)
        except OSError as error:
            raise OperationsIntegrityError(f'cannot read CAS object: {artifact.relative_path}') from error
        self._verify_object_descriptor(
            descriptor,
            artifact.sha256,
            artifact.byte_count,
            label=artifact.relative_path,
        )

    @staticmethod
    def _verify_object_descriptor(
        descriptor: int,
        expected_sha256: str,
        expected_bytes: int,
        *,
        label: str,
    ) -> None:
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with os.fdopen(descriptor, 'rb') as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    raise OperationsIntegrityError(f'CAS object is not a regular file: {label}')
                while chunk := source.read(_CHUNK_SIZE):
                    digest.update(chunk)
                    byte_count += len(chunk)
        except OSError as error:
            raise OperationsIntegrityError(f'cannot read CAS object: {label}') from error
        if byte_count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise OperationsIntegrityError(f'CAS object digest or size mismatch: {label}')

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> StoredArtifact:
        return StoredArtifact(
            sha256=str(row['sha256']),
            byte_count=int(row['byte_count']),
            relative_path=str(row['relative_path']),
            first_recorded_at=_parse_timestamp(str(row['first_recorded_at']), 'first_recorded_at'),
        )

    def artifact_path(self, sha256: str, *, verify: bool = True) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT sha256, byte_count, relative_path, first_recorded_at FROM artifacts WHERE sha256 = ?',
                (sha256,),
            ).fetchone()
        if row is None:
            raise OperationsStoreError(f'unknown artifact: {sha256}')
        artifact = self._artifact_from_row(row)
        path = self.root / artifact.relative_path
        if verify:
            self._verify_artifact_object(artifact)
        return path

    def read_artifact(self, sha256: str, *, max_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES) -> bytes:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT sha256, byte_count, relative_path, first_recorded_at FROM artifacts WHERE sha256 = ?',
                (sha256,),
            ).fetchone()
        if row is None:
            raise OperationsStoreError(f'unknown artifact: {sha256}')
        artifact = self._artifact_from_row(row)
        if artifact.byte_count > max_bytes:
            raise OperationsStoreError(f'artifact exceeds read max_bytes={max_bytes}')
        path = self.root / artifact.relative_path
        try:
            descriptor = self._open_cas_object(artifact.sha256)
            with os.fdopen(descriptor, 'rb') as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    raise OperationsIntegrityError(f'CAS object is not a regular file: {path}')
                payload = source.read(max_bytes + 1)
        except OSError as error:
            raise OperationsIntegrityError(f'cannot read CAS object: {path}') from error
        if len(payload) != artifact.byte_count or hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise OperationsIntegrityError(f'CAS object digest or size mismatch: {path}')
        return payload

    def register_job(self, spec: CaptureJobSpec, *, registered_at: datetime | None = None) -> RegisteredJob:
        registered_at = aware_utc(registered_at or _now_utc(), 'registered_at')
        configuration_valid = True
        try:
            if spec.collector_id == STATIC_HTTPS_COLLECTOR_ID:
                parse_static_job_configuration(spec.configuration)
            elif spec.collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID:
                parse_immport_authenticated_job_configuration(spec.configuration)
        except ValueError:
            configuration_valid = False
        if not configuration_valid:
            # Supported collector schemas are a persistence boundary.  Pydantic validation errors
            # can render rejected values, so discard them before raising the constant public error.
            raise OperationsStoreError('supported collector job configuration is invalid')
        spec_sha256 = job_spec_sha256(spec)
        spec_bytes = canonical_json_bytes(spec)
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                existing = connection.execute(
                    'SELECT job_id, spec_sha256, spec_json, registered_at FROM jobs WHERE spec_sha256 = ?',
                    (spec_sha256,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        'INSERT INTO jobs(job_id, spec_sha256, spec_json, registered_at) VALUES (?, ?, ?, ?)',
                        (spec.job_id, spec_sha256, spec_bytes, _timestamp(registered_at, 'registered_at')),
                    )
                    self._append_event(
                        connection,
                        LedgerEventType.JOB_REGISTERED,
                        registered_at,
                        {'job_id': spec.job_id, 'job_spec_sha256': spec_sha256},
                    )
                    result = RegisteredJob(spec=spec, spec_sha256=spec_sha256, registered_at=registered_at)
                else:
                    result = self._job_from_row(existing)
                    if result.spec != spec:
                        raise OperationsIntegrityError('job spec hash collision or noncanonical persisted spec')
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> RegisteredJob:
        try:
            spec = CaptureJobSpec.model_validate_json(bytes(row['spec_json']))
        except ValueError as error:
            raise OperationsIntegrityError('persisted job spec is invalid') from error
        if canonical_json_bytes(spec) != bytes(row['spec_json']):
            raise OperationsIntegrityError('persisted job spec is not canonical JSON')
        return RegisteredJob(
            spec=spec,
            spec_sha256=str(row['spec_sha256']),
            registered_at=_parse_timestamp(str(row['registered_at']), 'registered_at'),
        )

    def get_job(self, spec_sha256: str) -> RegisteredJob:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT job_id, spec_sha256, spec_json, registered_at FROM jobs WHERE spec_sha256 = ?',
                (spec_sha256,),
            ).fetchone()
        if row is None:
            raise OperationsStoreError(f'unknown job revision: {spec_sha256}')
        return self._job_from_row(row)

    def list_jobs(self, *, job_id: str | None = None) -> tuple[RegisteredJob, ...]:
        with self._connect() as connection:
            if job_id is None:
                rows = connection.execute(
                    'SELECT job_id, spec_sha256, spec_json, registered_at FROM jobs '
                    'ORDER BY job_id, registered_at, spec_sha256'
                ).fetchall()
            else:
                rows = connection.execute(
                    'SELECT job_id, spec_sha256, spec_json, registered_at FROM jobs '
                    'WHERE job_id = ? ORDER BY registered_at, spec_sha256',
                    (job_id,),
                ).fetchall()
        return tuple(self._job_from_row(row) for row in rows)

    def register_logical_run(
        self,
        job_spec_sha256: str,
        scheduled_for: datetime,
        *,
        registered_at: datetime | None = None,
    ) -> LogicalRunRecord:
        scheduled_for = aware_utc(scheduled_for, 'scheduled_for')
        registered_at = aware_utc(registered_at or _now_utc(), 'registered_at')
        logical_run_id = scheduled_logical_run_id(job_spec_sha256, scheduled_for)
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                job_row = connection.execute(
                    'SELECT job_id, spec_sha256, spec_json, registered_at FROM jobs WHERE spec_sha256 = ?',
                    (job_spec_sha256,),
                ).fetchone()
                if job_row is None:
                    raise OperationsStoreError(f'unknown job revision: {job_spec_sha256}')
                job = self._job_from_row(job_row)
                self._require_scheduled_slot(job.spec, scheduled_for)
                existing = connection.execute(
                    'SELECT * FROM logical_runs WHERE logical_run_id = ?', (logical_run_id,)
                ).fetchone()
                if existing is None:
                    connection.execute(
                        'INSERT INTO logical_runs('
                        'logical_run_id, job_id, job_spec_sha256, scheduled_for, state, successful_attempt_id'
                        ") VALUES (?, ?, ?, ?, 'pending', NULL)",
                        (logical_run_id, job.spec.job_id, job_spec_sha256, _timestamp(scheduled_for, 'scheduled_for')),
                    )
                    self._append_event(
                        connection,
                        LedgerEventType.LOGICAL_RUN_REGISTERED,
                        registered_at,
                        {
                            'job_spec_sha256': job_spec_sha256,
                            'logical_run_id': logical_run_id,
                            'scheduled_for': _timestamp(scheduled_for, 'scheduled_for'),
                        },
                    )
                    result = LogicalRunRecord(
                        logical_run_id=logical_run_id,
                        job_id=job.spec.job_id,
                        job_spec_sha256=job_spec_sha256,
                        scheduled_for=scheduled_for,
                        state=LogicalRunState.PENDING,
                    )
                else:
                    result = self._run_from_row(existing)
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _require_scheduled_slot(spec: CaptureJobSpec, scheduled_for: datetime) -> None:
        if scheduled_for < spec.schedule_anchor_at:
            raise ValueError('scheduled_for cannot precede the job schedule anchor')
        delta = scheduled_for - spec.schedule_anchor_at
        microseconds = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
        if microseconds % (spec.schedule_interval_seconds * 1_000_000) != 0:
            raise ValueError('scheduled_for must be exactly on the immutable job schedule')

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> LogicalRunRecord:
        return LogicalRunRecord(
            logical_run_id=str(row['logical_run_id']),
            job_id=str(row['job_id']),
            job_spec_sha256=str(row['job_spec_sha256']),
            scheduled_for=_parse_timestamp(str(row['scheduled_for']), 'scheduled_for'),
            state=LogicalRunState(str(row['state'])),
            successful_attempt_id=row['successful_attempt_id'],
        )

    def get_logical_run(self, logical_run_id: str) -> LogicalRunRecord:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM logical_runs WHERE logical_run_id = ?', (logical_run_id,)
            ).fetchone()
        if row is None:
            raise OperationsStoreError(f'unknown logical run: {logical_run_id}')
        return self._run_from_row(row)

    def list_logical_runs(
        self,
        *,
        job_spec_sha256: str | None = None,
    ) -> tuple[LogicalRunRecord, ...]:
        with self._connect() as connection:
            if job_spec_sha256 is None:
                rows = connection.execute(
                    'SELECT * FROM logical_runs ORDER BY scheduled_for, logical_run_id'
                ).fetchall()
            else:
                rows = connection.execute(
                    'SELECT * FROM logical_runs WHERE job_spec_sha256 = ? ORDER BY scheduled_for, logical_run_id',
                    (job_spec_sha256,),
                ).fetchall()
        return tuple(self._run_from_row(row) for row in rows)

    def begin_attempt(
        self,
        logical_run_id: str,
        *,
        owner_id: str,
        now: datetime | None = None,
        lease_seconds: int | None = None,
        max_attempts: int | None = None,
        initial_artifacts: Mapping[str, str] | None = None,
    ) -> AttemptLease:
        """Atomically claim a run and bind artifacts required to interpret the claim.

        ``initial_artifacts`` is intended for immutable execution intent such as a
        collection plan.  Every referenced CAS object is verified before the attempt,
        its attachments, and their ledger events are committed in one transaction.
        """

        owner_id = _validate_safe_id(owner_id, 'owner_id')
        initial_bindings = tuple(
            sorted(
                (
                    _validate_role(role),
                    artifact_sha256,
                )
                for role, artifact_sha256 in (initial_artifacts or {}).items()
            )
        )
        for _role, artifact_sha256 in initial_bindings:
            if not re.fullmatch(r'[0-9a-f]{64}', artifact_sha256):
                raise ValueError('initial artifact digest must be a lowercase SHA-256 hex string')
        requested_now = now
        if lease_seconds is not None and (lease_seconds < 1 or lease_seconds > 24 * 60 * 60):
            raise ValueError('lease_seconds must be between 1 and 86400 when provided')
        if max_attempts is not None and (max_attempts < 1 or max_attempts > 100):
            raise ValueError('max_attempts must be between 1 and 100 when provided')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                run_row = connection.execute(
                    'SELECT * FROM logical_runs WHERE logical_run_id = ?', (logical_run_id,)
                ).fetchone()
                if run_row is None:
                    raise OperationsStoreError(f'unknown logical run: {logical_run_id}')
                run = self._run_from_row(run_row)
                if run.state is LogicalRunState.SUCCEEDED:
                    raise RunAlreadySucceededError(f'logical run already succeeded: {logical_run_id}')
                job_row = connection.execute(
                    'SELECT * FROM jobs WHERE spec_sha256 = ?',
                    (run.job_spec_sha256,),
                ).fetchone()
                if job_row is None:
                    raise OperationsIntegrityError('logical run references an unknown job revision')
                job = self._job_from_row(job_row)
                selected_lease_seconds = lease_seconds if lease_seconds is not None else 900
                selected_max_attempts = max_attempts
                if job.spec.collector_id == STATIC_HTTPS_COLLECTOR_ID:
                    try:
                        static_policy = parse_static_job_configuration(job.spec.configuration)
                    except ValueError as error:
                        raise OperationsStoreError('static HTTPS job has invalid immutable policy') from error
                    if lease_seconds is not None and lease_seconds != static_policy.lease_seconds:
                        raise OperationsStoreError('lease_seconds differs from the immutable static job policy')
                    if max_attempts is not None and max_attempts != static_policy.max_attempts_per_slot:
                        raise OperationsStoreError('max_attempts differs from the immutable static job policy')
                    selected_lease_seconds = static_policy.lease_seconds
                    selected_max_attempts = static_policy.max_attempts_per_slot
                    if dict(initial_bindings) != {
                        'collection-plan': static_policy.collection_plan_sha256,
                    }:
                        raise OperationsStoreError(
                            'static HTTPS claims must atomically bind the exact immutable collection plan'
                        )
                elif job.spec.collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID:
                    try:
                        immport_policy = parse_immport_authenticated_job_configuration(job.spec.configuration)
                    except ValueError as error:
                        raise OperationsStoreError('authenticated ImmPort job has invalid immutable policy') from error
                    if lease_seconds is not None and lease_seconds != immport_policy.lease_seconds:
                        raise OperationsStoreError('lease_seconds differs from the immutable ImmPort job policy')
                    if max_attempts is not None and max_attempts != immport_policy.max_attempts_per_slot:
                        raise OperationsStoreError('max_attempts differs from the immutable ImmPort job policy')
                    selected_lease_seconds = immport_policy.lease_seconds
                    selected_max_attempts = immport_policy.max_attempts_per_slot
                    if dict(initial_bindings) != {
                        'collection-plan': immport_policy.collection_plan_sha256,
                    }:
                        raise OperationsStoreError(
                            'authenticated ImmPort claims must atomically bind the exact collection plan'
                        )
                active = connection.execute(
                    "SELECT attempt_id, lease_expires_at FROM attempts WHERE logical_run_id = ? AND state = 'started'",
                    (logical_run_id,),
                ).fetchone()
                if active is not None:
                    raise LeaseConflictError(
                        f'logical run has active or unreconciled attempt {active["attempt_id"]}; '
                        f'lease expires at {active["lease_expires_at"]}'
                    )
                number_row = connection.execute(
                    'SELECT COALESCE(MAX(attempt_number), 0) AS number FROM attempts WHERE logical_run_id = ?',
                    (logical_run_id,),
                ).fetchone()
                attempt_number = int(number_row['number']) + 1
                if selected_max_attempts is not None and attempt_number > selected_max_attempts:
                    raise AttemptBudgetExhaustedError(
                        f'logical run exhausted its immutable max_attempts={selected_max_attempts}'
                    )
                verified_initial_artifacts: list[tuple[str, str]] = []
                for role, artifact_sha256 in initial_bindings:
                    artifact_row = connection.execute(
                        'SELECT sha256, byte_count, relative_path, first_recorded_at FROM artifacts WHERE sha256 = ?',
                        (artifact_sha256,),
                    ).fetchone()
                    if artifact_row is None:
                        raise OperationsStoreError(f'unknown artifact: {artifact_sha256}')
                    self._verify_artifact_object(self._artifact_from_row(artifact_row))
                    verified_initial_artifacts.append((role, artifact_sha256))
                now = self._lease_operation_time(requested_now)
                if now < run.scheduled_for:
                    raise OperationsStoreError('logical run cannot be claimed before its scheduled time')
                lease_expires_at = now + timedelta(seconds=selected_lease_seconds)
                attempt_preimage = {
                    'attempt_number': attempt_number,
                    'logical_run_id': logical_run_id,
                    'owner_id': owner_id,
                    'started_at': _timestamp(now, 'started_at'),
                }
                attempt_id = f'attempt-{hashlib.sha256(canonical_json_bytes(attempt_preimage)).hexdigest()[:32]}'
                connection.execute(
                    'INSERT INTO attempts('
                    'attempt_id, logical_run_id, attempt_number, owner_id, state, started_at, '
                    'lease_expires_at, finished_at, terminal_code'
                    ") VALUES (?, ?, ?, ?, 'started', ?, ?, NULL, NULL)",
                    (
                        attempt_id,
                        logical_run_id,
                        attempt_number,
                        owner_id,
                        _timestamp(now, 'started_at'),
                        _timestamp(lease_expires_at, 'lease_expires_at'),
                    ),
                )
                connection.execute(
                    "UPDATE logical_runs SET state = 'running' WHERE logical_run_id = ? AND state = 'pending'",
                    (logical_run_id,),
                )
                self._append_event(
                    connection,
                    LedgerEventType.ATTEMPT_STARTED,
                    now,
                    {
                        'attempt_id': attempt_id,
                        'attempt_number': attempt_number,
                        'lease_expires_at': _timestamp(lease_expires_at, 'lease_expires_at'),
                        'logical_run_id': logical_run_id,
                        'owner_id': owner_id,
                    },
                )
                for role, artifact_sha256 in verified_initial_artifacts:
                    connection.execute(
                        'INSERT INTO attempt_artifacts('
                        'attempt_id, role, artifact_sha256, attached_at'
                        ') VALUES (?, ?, ?, ?)',
                        (attempt_id, role, artifact_sha256, _timestamp(now, 'attached_at')),
                    )
                    self._append_event(
                        connection,
                        LedgerEventType.ATTEMPT_ARTIFACT_ATTACHED,
                        now,
                        {
                            'artifact_sha256': artifact_sha256,
                            'attempt_id': attempt_id,
                            'role': role,
                        },
                    )
                connection.commit()
                return AttemptLease(
                    attempt_id=attempt_id,
                    logical_run_id=logical_run_id,
                    attempt_number=attempt_number,
                    owner_id=owner_id,
                    state=AttemptState.STARTED,
                    started_at=now,
                    lease_expires_at=lease_expires_at,
                )
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> AttemptLease:
        return AttemptLease(
            attempt_id=str(row['attempt_id']),
            logical_run_id=str(row['logical_run_id']),
            attempt_number=int(row['attempt_number']),
            owner_id=str(row['owner_id']),
            state=AttemptState(str(row['state'])),
            started_at=_parse_timestamp(str(row['started_at']), 'started_at'),
            lease_expires_at=_parse_timestamp(str(row['lease_expires_at']), 'lease_expires_at'),
            finished_at=(
                None if row['finished_at'] is None else _parse_timestamp(str(row['finished_at']), 'finished_at')
            ),
            terminal_code=row['terminal_code'],
        )

    def get_attempt(self, attempt_id: str) -> AttemptLease:
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM attempts WHERE attempt_id = ?', (attempt_id,)).fetchone()
        if row is None:
            raise OperationsStoreError(f'unknown attempt: {attempt_id}')
        return self._attempt_from_row(row)

    def list_attempts(self, *, logical_run_id: str | None = None) -> tuple[AttemptLease, ...]:
        with self._connect() as connection:
            if logical_run_id is None:
                rows = connection.execute('SELECT * FROM attempts ORDER BY logical_run_id, attempt_number').fetchall()
            else:
                rows = connection.execute(
                    'SELECT * FROM attempts WHERE logical_run_id = ? ORDER BY attempt_number',
                    (logical_run_id,),
                ).fetchall()
        return tuple(self._attempt_from_row(row) for row in rows)

    @staticmethod
    def _require_owned_started(row: sqlite3.Row, owner_id: str, now: datetime) -> AttemptLease:
        attempt = OperationalStore._attempt_from_row(row)
        if attempt.state is not AttemptState.STARTED:
            raise AttemptStateError(f'attempt is already terminal: {attempt.state.value}')
        if attempt.owner_id != owner_id:
            raise LeaseConflictError('attempt lease is owned by a different worker')
        if now < attempt.started_at:
            raise LeaseConflictError('operation time cannot predate the attempt start')
        if attempt.lease_expires_at <= now:
            raise LeaseConflictError('attempt lease has expired and must be reconciled')
        return attempt

    def renew_lease(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        now: datetime | None = None,
        lease_seconds: int = 900,
    ) -> AttemptLease:
        requested_now = now
        if lease_seconds < 1 or lease_seconds > 24 * 60 * 60:
            raise ValueError('lease_seconds must be between 1 and 86400')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                now = self._lease_operation_time(requested_now)
                new_expiry = now + timedelta(seconds=lease_seconds)
                row = connection.execute('SELECT * FROM attempts WHERE attempt_id = ?', (attempt_id,)).fetchone()
                if row is None:
                    raise OperationsStoreError(f'unknown attempt: {attempt_id}')
                attempt = self._require_owned_started(row, owner_id, now)
                collector_row = connection.execute(
                    'SELECT j.* FROM logical_runs r '
                    'JOIN jobs j ON j.spec_sha256 = r.job_spec_sha256 '
                    'WHERE r.logical_run_id = ?',
                    (attempt.logical_run_id,),
                ).fetchone()
                if collector_row is None:
                    raise OperationsIntegrityError('attempt references an unknown collector policy')
                if self._job_from_row(collector_row).spec.collector_id in {
                    STATIC_HTTPS_COLLECTOR_ID,
                    IMMPORT_AUTHENTICATED_COLLECTOR_ID,
                }:
                    raise AttemptStateError(
                        'precommitted collector attempt leases cannot be renewed; the immutable initial lease is final'
                    )
                if new_expiry <= attempt.lease_expires_at:
                    raise ValueError('renewed lease must extend the current lease expiry')
                connection.execute(
                    'UPDATE attempts SET lease_expires_at = ? WHERE attempt_id = ?',
                    (_timestamp(new_expiry, 'lease_expires_at'), attempt_id),
                )
                self._append_event(
                    connection,
                    LedgerEventType.ATTEMPT_LEASE_RENEWED,
                    now,
                    {'attempt_id': attempt_id, 'lease_expires_at': _timestamp(new_expiry, 'lease_expires_at')},
                )
                connection.commit()
                return attempt.model_copy(update={'lease_expires_at': new_expiry})
            except BaseException:
                connection.rollback()
                raise

    def attach_artifact(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        role: str,
        artifact_sha256: str,
        now: datetime | None = None,
    ) -> StoredArtifact:
        role = _validate_role(role)
        now = self._lease_operation_time(now)
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                attempt_row = connection.execute(
                    'SELECT * FROM attempts WHERE attempt_id = ?', (attempt_id,)
                ).fetchone()
                if attempt_row is None:
                    raise OperationsStoreError(f'unknown attempt: {attempt_id}')
                self._require_owned_started(attempt_row, owner_id, now)
                artifact_row = connection.execute(
                    'SELECT sha256, byte_count, relative_path, first_recorded_at FROM artifacts WHERE sha256 = ?',
                    (artifact_sha256,),
                ).fetchone()
                if artifact_row is None:
                    raise OperationsStoreError(f'unknown artifact: {artifact_sha256}')
                artifact = self._artifact_from_row(artifact_row)
                self._verify_artifact_object(artifact)
                commit_now = self._lease_operation_time(now)
                self._require_owned_started(attempt_row, owner_id, commit_now)
                existing = connection.execute(
                    'SELECT artifact_sha256 FROM attempt_artifacts WHERE attempt_id = ? AND role = ?',
                    (attempt_id, role),
                ).fetchone()
                if existing is not None:
                    if str(existing['artifact_sha256']) != artifact_sha256:
                        raise AttemptStateError(f'artifact role {role!r} is already bound to different bytes')
                    connection.commit()
                    return artifact
                connection.execute(
                    'INSERT INTO attempt_artifacts(attempt_id, role, artifact_sha256, attached_at) VALUES (?, ?, ?, ?)',
                    (attempt_id, role, artifact_sha256, _timestamp(commit_now, 'attached_at')),
                )
                self._append_event(
                    connection,
                    LedgerEventType.ATTEMPT_ARTIFACT_ATTACHED,
                    commit_now,
                    {'artifact_sha256': artifact_sha256, 'attempt_id': attempt_id, 'role': role},
                )
                connection.commit()
                return artifact
            except BaseException:
                connection.rollback()
                raise

    def list_attempt_artifacts(self, attempt_id: str) -> dict[str, StoredArtifact]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT aa.role, a.sha256, a.byte_count, a.relative_path, a.first_recorded_at '
                'FROM attempt_artifacts aa JOIN artifacts a ON a.sha256 = aa.artifact_sha256 '
                'WHERE aa.attempt_id = ? ORDER BY aa.role',
                (attempt_id,),
            ).fetchall()
        return {str(row['role']): self._artifact_from_row(row) for row in rows}

    def fail_attempt(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        terminal_code: str,
        now: datetime | None = None,
    ) -> AttemptLease:
        return self._terminalize_attempt(
            attempt_id,
            owner_id=owner_id,
            state=AttemptState.FAILED,
            terminal_code=terminal_code,
            now=now,
        )

    def succeed_attempt(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        run_manifest_sha256: str | None = None,
        now: datetime | None = None,
    ) -> AttemptLease:
        """Atomically bind a terminal manifest and mark an attempt successful.

        Existing callers may pre-attach ``run-manifest`` and omit its digest.  A
        collector should pass ``run_manifest_sha256`` so lease validation, immutable
        attachment, and success are committed in one transaction.
        """

        return self._terminalize_attempt(
            attempt_id,
            owner_id=owner_id,
            state=AttemptState.SUCCEEDED,
            terminal_code='success',
            run_manifest_sha256=run_manifest_sha256,
            now=now,
        )

    def _terminalize_attempt(
        self,
        attempt_id: str,
        *,
        owner_id: str,
        state: AttemptState,
        terminal_code: str,
        run_manifest_sha256: str | None = None,
        now: datetime | None = None,
    ) -> AttemptLease:
        now = self._lease_operation_time(now)
        if not terminal_code or len(terminal_code) > 200 or any(character in terminal_code for character in '\x00\r\n'):
            raise ValueError('terminal_code must be nonempty, bounded, and single-line')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                row = connection.execute('SELECT * FROM attempts WHERE attempt_id = ?', (attempt_id,)).fetchone()
                if row is None:
                    raise OperationsStoreError(f'unknown attempt: {attempt_id}')
                attempt = self._require_owned_started(row, owner_id, now)
                pending_manifest_sha256: str | None = None
                if state is AttemptState.SUCCEEDED:
                    existing_manifest = connection.execute(
                        "SELECT artifact_sha256 FROM attempt_artifacts WHERE attempt_id = ? AND role = 'run-manifest'",
                        (attempt_id,),
                    ).fetchone()
                    if run_manifest_sha256 is not None:
                        if existing_manifest is not None:
                            if str(existing_manifest['artifact_sha256']) != run_manifest_sha256:
                                raise AttemptStateError(
                                    "artifact role 'run-manifest' is already bound to different bytes"
                                )
                        else:
                            pending_manifest_sha256 = run_manifest_sha256
                    elif existing_manifest is None:
                        raise AttemptStateError('a successful attempt must bind a run-manifest artifact')

                    selected_manifest_sha256 = run_manifest_sha256 or str(existing_manifest['artifact_sha256'])
                    artifact_row = connection.execute(
                        'SELECT sha256, byte_count, relative_path, first_recorded_at FROM artifacts WHERE sha256 = ?',
                        (selected_manifest_sha256,),
                    ).fetchone()
                    if artifact_row is None:
                        raise OperationsStoreError(f'unknown artifact: {selected_manifest_sha256}')
                    self._verify_artifact_object(self._artifact_from_row(artifact_row))
                    run = connection.execute(
                        'SELECT state FROM logical_runs WHERE logical_run_id = ?', (attempt.logical_run_id,)
                    ).fetchone()
                    if run is None or str(run['state']) == LogicalRunState.SUCCEEDED.value:
                        raise RunAlreadySucceededError(f'logical run already succeeded: {attempt.logical_run_id}')
                terminal_now = self._lease_operation_time(now)
                attempt = self._require_owned_started(row, owner_id, terminal_now)
                if pending_manifest_sha256 is not None:
                    connection.execute(
                        'INSERT INTO attempt_artifacts('
                        'attempt_id, role, artifact_sha256, attached_at'
                        ') VALUES (?, ?, ?, ?)',
                        (attempt_id, 'run-manifest', pending_manifest_sha256, _timestamp(terminal_now, 'attached_at')),
                    )
                    self._append_event(
                        connection,
                        LedgerEventType.ATTEMPT_ARTIFACT_ATTACHED,
                        terminal_now,
                        {
                            'artifact_sha256': pending_manifest_sha256,
                            'attempt_id': attempt_id,
                            'role': 'run-manifest',
                        },
                    )
                connection.execute(
                    'UPDATE attempts SET state = ?, finished_at = ?, terminal_code = ? WHERE attempt_id = ?',
                    (state.value, _timestamp(terminal_now, 'finished_at'), terminal_code, attempt_id),
                )
                if state is AttemptState.SUCCEEDED:
                    updated = connection.execute(
                        "UPDATE logical_runs SET state = 'succeeded', successful_attempt_id = ? "
                        "WHERE logical_run_id = ? AND state != 'succeeded'",
                        (attempt_id, attempt.logical_run_id),
                    )
                    if updated.rowcount != 1:
                        raise RunAlreadySucceededError(f'logical run already succeeded: {attempt.logical_run_id}')
                    event_type = LedgerEventType.ATTEMPT_SUCCEEDED
                else:
                    connection.execute(
                        "UPDATE logical_runs SET state = 'pending' WHERE logical_run_id = ? AND state = 'running'",
                        (attempt.logical_run_id,),
                    )
                    event_type = LedgerEventType.ATTEMPT_FAILED
                self._append_event(
                    connection,
                    event_type,
                    terminal_now,
                    {
                        'attempt_id': attempt_id,
                        'logical_run_id': attempt.logical_run_id,
                        'terminal_code': terminal_code,
                    },
                )
                connection.commit()
                return attempt.model_copy(
                    update={'state': state, 'finished_at': terminal_now, 'terminal_code': terminal_code}
                )
            except BaseException:
                connection.rollback()
                raise

    def abandon_expired_attempts(self, *, now: datetime | None = None) -> tuple[AttemptLease, ...]:
        """Atomically retain and close every expired lease, making its slot retryable."""

        now = self._lease_operation_time(now)
        abandoned: list[AttemptLease] = []
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                rows = connection.execute(
                    "SELECT * FROM attempts WHERE state = 'started' AND lease_expires_at <= ? "
                    'ORDER BY logical_run_id, attempt_number',
                    (_timestamp(now, 'now'),),
                ).fetchall()
                for row in rows:
                    attempt = self._attempt_from_row(row)
                    connection.execute(
                        "UPDATE attempts SET state = 'abandoned', finished_at = ?, terminal_code = 'lease_expired' "
                        'WHERE attempt_id = ?',
                        (_timestamp(now, 'finished_at'), attempt.attempt_id),
                    )
                    connection.execute(
                        "UPDATE logical_runs SET state = 'pending' WHERE logical_run_id = ? AND state = 'running'",
                        (attempt.logical_run_id,),
                    )
                    self._append_event(
                        connection,
                        LedgerEventType.ATTEMPT_ABANDONED,
                        now,
                        {
                            'attempt_id': attempt.attempt_id,
                            'logical_run_id': attempt.logical_run_id,
                            'terminal_code': 'lease_expired',
                        },
                    )
                    abandoned.append(
                        attempt.model_copy(
                            update={
                                'state': AttemptState.ABANDONED,
                                'finished_at': now,
                                'terminal_code': 'lease_expired',
                            }
                        )
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return tuple(abandoned)

    def events(self) -> tuple[LedgerEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute('SELECT event_json FROM events ORDER BY sequence').fetchall()
        return tuple(self._load_event(bytes(row['event_json'])) for row in rows)

    @contextmanager
    def verification_window(self) -> Iterator[None]:
        """Block writers while generic and collector-specific verification run."""

        with self._connect() as reservation:
            reservation.execute('BEGIN IMMEDIATE')
            try:
                yield
                reservation.commit()
            except BaseException:
                reservation.rollback()
                raise

    @staticmethod
    def _load_event(payload: bytes) -> LedgerEvent:
        try:
            event = LedgerEvent.model_validate_json(payload)
        except ValueError as error:
            raise OperationsIntegrityError('persisted ledger event is invalid') from error
        if payload != canonical_json_bytes(event):
            raise OperationsIntegrityError('persisted ledger event is not canonical JSON')
        return event

    @staticmethod
    def _inventory_from_events(events: tuple[LedgerEvent, ...]) -> tuple[tuple[str, int], ...]:
        objects: dict[str, int] = {}
        for event in events:
            if event.event_type is not LedgerEventType.ARTIFACT_STORED:
                continue
            sha256 = event.payload.get('artifact_sha256')
            byte_count = event.payload.get('byte_count')
            if not isinstance(sha256, str) or not isinstance(byte_count, int) or isinstance(byte_count, bool):
                raise OperationsIntegrityError('artifact ledger event has malformed inventory fields')
            if sha256 in objects:
                raise OperationsIntegrityError('artifact ledger contains duplicate first-record events')
            objects[sha256] = byte_count
        return tuple(sorted(objects.items()))

    def checkpoint(
        self,
        *,
        created_at: datetime | None = None,
        semantic_verifier: Callable[[], object] | None = None,
    ) -> LedgerCheckpoint:
        """Create a verified canonical witness target from one writer-blocking snapshot."""

        created_at = aware_utc(created_at or _now_utc(), 'created_at')
        with self._connect() as reservation:
            reservation.execute('BEGIN IMMEDIATE')
            try:
                # The reserved write transaction prevents any writer from committing
                # between verification and selecting the checkpoint head.
                self.verify(verified_at=created_at)
                if semantic_verifier is not None:
                    semantic_verifier()
                rows = reservation.execute('SELECT event_json FROM events ORDER BY sequence').fetchall()
                events = tuple(self._load_event(bytes(row['event_json'])) for row in rows)
                if not events:
                    raise OperationsIntegrityError('operations ledger cannot be empty')
                if max(event.occurred_at for event in events) > created_at:
                    raise OperationsIntegrityError('checkpoint cannot predate any event in its ledger prefix')
                inventory = self._inventory_from_events(events)
                checkpoint = LedgerCheckpoint(
                    store_id=self.store_id,
                    created_at=created_at,
                    through_sequence=events[-1].sequence,
                    through_event_sha256=events[-1].event_sha256,
                    object_count=len(inventory),
                    object_inventory_sha256=hashlib.sha256(canonical_json_bytes(inventory)).hexdigest(),
                )
                reservation.commit()
                return checkpoint
            except BaseException:
                reservation.rollback()
                raise

    def verify_checkpoint(self, checkpoint: LedgerCheckpoint) -> None:
        if checkpoint.store_id != self.store_id:
            raise OperationsIntegrityError('checkpoint belongs to a different operations store')
        events = self.events()
        if checkpoint.through_sequence > len(events):
            raise OperationsIntegrityError('checkpoint extends beyond the local ledger')
        prefix = events[: checkpoint.through_sequence]
        if not prefix or prefix[-1].event_sha256 != checkpoint.through_event_sha256:
            raise OperationsIntegrityError('checkpoint ledger prefix does not match local history')
        if max(event.occurred_at for event in prefix) > checkpoint.created_at:
            raise OperationsIntegrityError('checkpoint predates an event in its ledger prefix')
        inventory = self._inventory_from_events(prefix)
        expected_sha256 = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
        if checkpoint.object_count != len(inventory) or checkpoint.object_inventory_sha256 != expected_sha256:
            raise OperationsIntegrityError('checkpoint object inventory does not match its ledger prefix')

    def verify(
        self,
        *,
        checkpoint: LedgerCheckpoint | None = None,
        verified_at: datetime | None = None,
    ) -> StoreVerificationReport:
        """Fail closed on broken ledger, metadata, state relations, or registered CAS bytes."""

        verified_at = aware_utc(verified_at or _now_utc(), 'verified_at')
        with self._connect() as connection:
            connection.execute('BEGIN')
            integrity = connection.execute('PRAGMA integrity_check').fetchone()
            if integrity is None or str(integrity[0]) != 'ok':
                raise OperationsIntegrityError('SQLite integrity_check failed')
            if self._metadata(connection, 'store_id') != self.store_id:
                raise OperationsIntegrityError('store_id changed after opening the store')
            event_rows = connection.execute('SELECT * FROM events ORDER BY sequence').fetchall()
            if not event_rows:
                raise OperationsIntegrityError('operations ledger is empty')
            events: list[LedgerEvent] = []
            previous_sha256: str | None = None
            for expected_sequence, row in enumerate(event_rows, start=1):
                event = self._load_event(bytes(row['event_json']))
                if event.sequence != expected_sequence or int(row['sequence']) != expected_sequence:
                    raise OperationsIntegrityError('ledger event sequence is not contiguous')
                if event.previous_event_sha256 != previous_sha256:
                    raise OperationsIntegrityError('ledger previous-event hash chain is broken')
                if (
                    str(row['event_type']) != event.event_type.value
                    or str(row['event_sha256']) != event.event_sha256
                    or row['previous_event_sha256'] != event.previous_event_sha256
                    or _parse_timestamp(str(row['occurred_at']), 'occurred_at') != event.occurred_at
                    or ledger_event_sha256(event) != event.event_sha256
                ):
                    raise OperationsIntegrityError('ledger indexed columns do not match canonical event bytes')
                events.append(event)
                previous_sha256 = event.event_sha256
            if any(event.occurred_at > verified_at for event in events):
                raise OperationsIntegrityError('verification time predates an event in the ledger')

            job_rows = connection.execute('SELECT * FROM jobs ORDER BY spec_sha256').fetchall()
            jobs = {str(row['spec_sha256']): self._job_from_row(row) for row in job_rows}
            run_rows = connection.execute('SELECT * FROM logical_runs ORDER BY logical_run_id').fetchall()
            runs = {str(row['logical_run_id']): self._run_from_row(row) for row in run_rows}
            attempt_rows = connection.execute(
                'SELECT * FROM attempts ORDER BY logical_run_id, attempt_number'
            ).fetchall()
            attempts = {str(row['attempt_id']): self._attempt_from_row(row) for row in attempt_rows}
            artifact_rows = connection.execute('SELECT * FROM artifacts ORDER BY sha256').fetchall()
            artifacts = {str(row['sha256']): self._artifact_from_row(row) for row in artifact_rows}
            attachment_rows = connection.execute(
                'SELECT attempt_id, role, artifact_sha256, attached_at FROM attempt_artifacts ORDER BY attempt_id, role'
            ).fetchall()
            connection.commit()

        for run in runs.values():
            job = jobs.get(run.job_spec_sha256)
            if job is None or run.job_id != job.spec.job_id:
                raise OperationsIntegrityError('logical run references the wrong job revision')
            if scheduled_logical_run_id(run.job_spec_sha256, run.scheduled_for) != run.logical_run_id:
                raise OperationsIntegrityError('logical run ID does not bind its revision and slot')
            self._require_scheduled_slot(job.spec, run.scheduled_for)
            if run.state is LogicalRunState.SUCCEEDED:
                attempt = attempts.get(run.successful_attempt_id or '')
                if attempt is None or attempt.state is not AttemptState.SUCCEEDED:
                    raise OperationsIntegrityError('successful logical run lacks its successful attempt')
        seen_attempt_numbers: set[tuple[str, int]] = set()
        for attempt in attempts.values():
            if attempt.logical_run_id not in runs:
                raise OperationsIntegrityError('attempt references an unknown logical run')
            key = (attempt.logical_run_id, attempt.attempt_number)
            if key in seen_attempt_numbers:
                raise OperationsIntegrityError('attempt number is duplicated within a logical run')
            seen_attempt_numbers.add(key)
        for row in attachment_rows:
            if str(row['attempt_id']) not in attempts or str(row['artifact_sha256']) not in artifacts:
                raise OperationsIntegrityError('attempt artifact references unknown durable state')
        attachment_keys = {(str(row['attempt_id']), str(row['role'])) for row in attachment_rows}
        if any(
            attempt.state is AttemptState.SUCCEEDED and (attempt.attempt_id, 'run-manifest') not in attachment_keys
            for attempt in attempts.values()
        ):
            raise OperationsIntegrityError('successful attempt lacks its immutable run-manifest attachment')
        for artifact in artifacts.values():
            expected_path = f'objects/sha256/{artifact.sha256[:2]}/{artifact.sha256}'
            if artifact.relative_path != expected_path:
                raise OperationsIntegrityError('artifact path is not derived from its digest')
            self._verify_artifact_object(artifact)

        inventory = self._inventory_from_events(tuple(events))
        persisted_inventory = tuple(sorted((artifact.sha256, artifact.byte_count) for artifact in artifacts.values()))
        if inventory != persisted_inventory:
            raise OperationsIntegrityError('artifact table does not match first-record ledger events')
        event_types = [event.event_type for event in events]
        if event_types[0] is not LedgerEventType.STORE_INITIALIZED:
            raise OperationsIntegrityError('first ledger event must initialize the store')
        if event_types.count(LedgerEventType.STORE_INITIALIZED) != 1:
            raise OperationsIntegrityError('store initialization event must be unique')
        self._verify_materialized_state(
            tuple(events),
            jobs=jobs,
            runs=runs,
            attempts=attempts,
            artifacts=artifacts,
            attachment_rows=attachment_rows,
        )

        checkpoint_verified = False
        if checkpoint is not None:
            self.verify_checkpoint(checkpoint)
            checkpoint_verified = True
        return StoreVerificationReport(
            store_id=self.store_id,
            verified_at=verified_at,
            event_count=len(events),
            ledger_head_sha256=events[-1].event_sha256,
            object_count=len(artifacts),
            job_count=len(jobs),
            logical_run_count=len(runs),
            attempt_count=len(attempts),
            checkpoint_verified=checkpoint_verified,
        )

    @staticmethod
    def _verify_materialized_state(
        events: tuple[LedgerEvent, ...],
        *,
        jobs: dict[str, RegisteredJob],
        runs: dict[str, LogicalRunRecord],
        attempts: dict[str, AttemptLease],
        artifacts: dict[str, StoredArtifact],
        attachment_rows: list[sqlite3.Row],
    ) -> None:
        registered_jobs: dict[str, tuple[str, datetime]] = {}
        registered_runs: dict[str, str] = {}
        reconstructed_attempts: dict[str, dict[str, object]] = {}
        reconstructed_attachments: dict[tuple[str, str], tuple[str, datetime]] = {}
        recorded_artifacts: dict[str, tuple[int, datetime]] = {}
        next_attempt_number_by_run: dict[str, int] = {}
        for event in events:
            payload = event.payload
            if event.event_type is LedgerEventType.ARTIFACT_STORED:
                sha256 = payload.get('artifact_sha256')
                byte_count = payload.get('byte_count')
                if (
                    not isinstance(sha256, str)
                    or not isinstance(byte_count, int)
                    or isinstance(byte_count, bool)
                    or sha256 in recorded_artifacts
                ):
                    raise OperationsIntegrityError('artifact first-record event is malformed or duplicated')
                recorded_artifacts[sha256] = (byte_count, event.occurred_at)
            elif event.event_type is LedgerEventType.JOB_REGISTERED:
                spec_sha256 = payload.get('job_spec_sha256')
                job_id = payload.get('job_id')
                if not isinstance(spec_sha256, str) or not isinstance(job_id, str) or spec_sha256 in registered_jobs:
                    raise OperationsIntegrityError('job registration event is malformed or duplicated')
                registered_jobs[spec_sha256] = (job_id, event.occurred_at)
            elif event.event_type is LedgerEventType.LOGICAL_RUN_REGISTERED:
                run_id = payload.get('logical_run_id')
                spec_sha256 = payload.get('job_spec_sha256')
                scheduled_for = payload.get('scheduled_for')
                if (
                    not isinstance(run_id, str)
                    or not isinstance(spec_sha256, str)
                    or not isinstance(scheduled_for, str)
                    or run_id in registered_runs
                    or spec_sha256 not in registered_jobs
                    or run_id not in runs
                ):
                    raise OperationsIntegrityError('logical-run registration event is malformed or duplicated')
                if runs[run_id].job_spec_sha256 != spec_sha256 or runs[run_id].scheduled_for != _parse_timestamp(
                    scheduled_for, 'scheduled_for'
                ):
                    raise OperationsIntegrityError('logical-run registration event does not bind its persisted run')
                registered_runs[run_id] = spec_sha256
            elif event.event_type is LedgerEventType.ATTEMPT_STARTED:
                attempt_id = payload.get('attempt_id')
                if not isinstance(attempt_id, str) or attempt_id in reconstructed_attempts:
                    raise OperationsIntegrityError('attempt-start event is malformed or duplicated')
                logical_run_id = payload.get('logical_run_id')
                attempt_number = payload.get('attempt_number')
                lease_expires_at = payload.get('lease_expires_at')
                owner_id = payload.get('owner_id')
                if (
                    not isinstance(logical_run_id, str)
                    or logical_run_id not in registered_runs
                    or not isinstance(attempt_number, int)
                    or isinstance(attempt_number, bool)
                    or not isinstance(lease_expires_at, str)
                    or not isinstance(owner_id, str)
                ):
                    raise OperationsIntegrityError('attempt-start event has malformed lifecycle fields')
                expected_attempt_number = next_attempt_number_by_run.get(logical_run_id, 1)
                if attempt_number != expected_attempt_number:
                    raise OperationsIntegrityError('attempt-start events are not contiguous within their logical run')
                if any(
                    state['logical_run_id'] == logical_run_id and state['state'] is AttemptState.STARTED
                    for state in reconstructed_attempts.values()
                ):
                    raise OperationsIntegrityError('attempt-start event overlaps an active attempt for its logical run')
                parsed_lease_expires_at = _parse_timestamp(lease_expires_at, 'lease_expires_at')
                if event.occurred_at < runs[logical_run_id].scheduled_for:
                    raise OperationsIntegrityError('attempt-start event predates its scheduled logical run')
                if parsed_lease_expires_at <= event.occurred_at:
                    raise OperationsIntegrityError('attempt-start event has a nonpositive lease interval')
                next_attempt_number_by_run[logical_run_id] = expected_attempt_number + 1
                reconstructed_attempts[attempt_id] = {
                    'attempt_number': attempt_number,
                    'lease_expires_at': lease_expires_at,
                    'logical_run_id': logical_run_id,
                    'owner_id': owner_id,
                    'started_at': event.occurred_at,
                    'state': AttemptState.STARTED,
                    'finished_at': None,
                    'terminal_code': None,
                }
            elif event.event_type is LedgerEventType.ATTEMPT_LEASE_RENEWED:
                attempt_id = payload.get('attempt_id')
                state = reconstructed_attempts.get(str(attempt_id))
                if state is None or state['state'] is not AttemptState.STARTED:
                    raise OperationsIntegrityError('lease-renewal event references a non-started attempt')
                current_expiry = state.get('lease_expires_at')
                started_at = state.get('started_at')
                renewed_expiry = payload.get('lease_expires_at')
                if (
                    not isinstance(current_expiry, str)
                    or not isinstance(started_at, datetime)
                    or not isinstance(renewed_expiry, str)
                ):
                    raise OperationsIntegrityError('lease-renewal event has malformed expiry fields')
                current_expiry_at = _parse_timestamp(current_expiry, 'lease_expires_at')
                renewed_expiry_at = _parse_timestamp(renewed_expiry, 'lease_expires_at')
                if (
                    event.occurred_at < started_at
                    or event.occurred_at >= current_expiry_at
                    or renewed_expiry_at <= current_expiry_at
                ):
                    raise OperationsIntegrityError('lease-renewal event does not extend a live lease')
                state['lease_expires_at'] = renewed_expiry
            elif event.event_type is LedgerEventType.ATTEMPT_ARTIFACT_ATTACHED:
                attempt_id = payload.get('attempt_id')
                role = payload.get('role')
                sha256 = payload.get('artifact_sha256')
                if not isinstance(attempt_id, str) or not isinstance(role, str) or not isinstance(sha256, str):
                    raise OperationsIntegrityError('attempt-artifact event is malformed')
                state = reconstructed_attempts.get(attempt_id)
                if state is None or state['state'] is not AttemptState.STARTED:
                    raise OperationsIntegrityError('attempt-artifact event references a non-started attempt')
                lease_expires_at = state.get('lease_expires_at')
                started_at = state.get('started_at')
                if not isinstance(lease_expires_at, str) or not isinstance(started_at, datetime):
                    raise OperationsIntegrityError('attempt-artifact event has no valid lease context')
                if event.occurred_at < started_at or event.occurred_at >= _parse_timestamp(
                    lease_expires_at, 'lease_expires_at'
                ):
                    raise OperationsIntegrityError('attempt-artifact event occurred after lease expiry')
                if sha256 not in recorded_artifacts:
                    raise OperationsIntegrityError('attempt-artifact event precedes the artifact first-record event')
                key = (attempt_id, role)
                if key in reconstructed_attachments:
                    raise OperationsIntegrityError('attempt artifact role is attached more than once in the ledger')
                reconstructed_attachments[key] = (sha256, event.occurred_at)
            elif event.event_type in {
                LedgerEventType.ATTEMPT_FAILED,
                LedgerEventType.ATTEMPT_ABANDONED,
                LedgerEventType.ATTEMPT_SUCCEEDED,
            }:
                attempt_id = payload.get('attempt_id')
                state = reconstructed_attempts.get(str(attempt_id))
                if state is None or state['state'] is not AttemptState.STARTED:
                    raise OperationsIntegrityError('attempt terminal event references a non-started attempt')
                logical_run_id = payload.get('logical_run_id')
                terminal_code = payload.get('terminal_code')
                lease_expires_at = state.get('lease_expires_at')
                started_at = state.get('started_at')
                if (
                    logical_run_id != state['logical_run_id']
                    or not isinstance(terminal_code, str)
                    or not terminal_code
                    or not isinstance(lease_expires_at, str)
                    or not isinstance(started_at, datetime)
                ):
                    raise OperationsIntegrityError('attempt terminal event has malformed lifecycle fields')
                expiry_at = _parse_timestamp(lease_expires_at, 'lease_expires_at')
                if event.occurred_at < started_at:
                    raise OperationsIntegrityError('attempt terminal event predates its start')
                if event.event_type is LedgerEventType.ATTEMPT_ABANDONED:
                    if event.occurred_at < expiry_at:
                        raise OperationsIntegrityError('attempt was abandoned before its lease expired')
                elif event.occurred_at >= expiry_at:
                    raise OperationsIntegrityError('attempt was terminalized after its lease expired')
                if (
                    event.event_type is LedgerEventType.ATTEMPT_SUCCEEDED
                    and (str(attempt_id), 'run-manifest') not in reconstructed_attachments
                ):
                    raise OperationsIntegrityError('attempt succeeded before its run manifest was attached')
                state['state'] = {
                    LedgerEventType.ATTEMPT_FAILED: AttemptState.FAILED,
                    LedgerEventType.ATTEMPT_ABANDONED: AttemptState.ABANDONED,
                    LedgerEventType.ATTEMPT_SUCCEEDED: AttemptState.SUCCEEDED,
                }[event.event_type]
                state['finished_at'] = event.occurred_at
                state['terminal_code'] = terminal_code

        if registered_jobs != {sha256: (job.spec.job_id, job.registered_at) for sha256, job in jobs.items()}:
            raise OperationsIntegrityError('job table does not match job-registration ledger events')
        if recorded_artifacts != {
            sha256: (artifact.byte_count, artifact.first_recorded_at) for sha256, artifact in artifacts.items()
        }:
            raise OperationsIntegrityError('artifact table does not match artifact first-record ledger events')
        if registered_runs != {run_id: run.job_spec_sha256 for run_id, run in runs.items()}:
            raise OperationsIntegrityError('logical-run table does not match registration ledger events')
        if set(reconstructed_attempts) != set(attempts):
            raise OperationsIntegrityError('attempt table does not match attempt-start ledger events')
        for attempt_id, attempt in attempts.items():
            expected = reconstructed_attempts[attempt_id]
            lease_value = expected['lease_expires_at']
            if not isinstance(lease_value, str):
                raise OperationsIntegrityError('attempt ledger has malformed lease expiry')
            if (
                expected['attempt_number'] != attempt.attempt_number
                or expected['logical_run_id'] != attempt.logical_run_id
                or expected['owner_id'] != attempt.owner_id
                or expected['started_at'] != attempt.started_at
                or _parse_timestamp(lease_value, 'lease_expires_at') != attempt.lease_expires_at
                or expected['state'] is not attempt.state
                or expected['finished_at'] != attempt.finished_at
                or expected['terminal_code'] != attempt.terminal_code
            ):
                raise OperationsIntegrityError('attempt table does not match its lifecycle ledger events')

        persisted_attachments = {
            (str(row['attempt_id']), str(row['role'])): (
                str(row['artifact_sha256']),
                _parse_timestamp(str(row['attached_at']), 'attached_at'),
            )
            for row in attachment_rows
        }
        if reconstructed_attachments != persisted_attachments:
            raise OperationsIntegrityError('attempt-artifact table does not match attachment ledger events')

        successful_by_run = {
            attempt.logical_run_id: attempt.attempt_id
            for attempt in attempts.values()
            if attempt.state is AttemptState.SUCCEEDED
        }
        active_runs = {attempt.logical_run_id for attempt in attempts.values() if attempt.state is AttemptState.STARTED}
        for run in runs.values():
            expected_success = successful_by_run.get(run.logical_run_id)
            expected_state = (
                LogicalRunState.SUCCEEDED
                if expected_success is not None
                else LogicalRunState.RUNNING
                if run.logical_run_id in active_runs
                else LogicalRunState.PENDING
            )
            if run.state is not expected_state or run.successful_attempt_id != expected_success:
                raise OperationsIntegrityError('logical-run table does not match attempt lifecycle events')


_SCHEMA_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE events (
    sequence INTEGER PRIMARY KEY CHECK (sequence >= 1),
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    previous_event_sha256 TEXT,
    event_sha256 TEXT NOT NULL UNIQUE CHECK (length(event_sha256) = 64),
    event_json BLOB NOT NULL
) STRICT;

CREATE TABLE jobs (
    job_id TEXT NOT NULL,
    spec_sha256 TEXT PRIMARY KEY CHECK (length(spec_sha256) = 64),
    spec_json BLOB NOT NULL,
    registered_at TEXT NOT NULL
) STRICT;
CREATE INDEX jobs_by_id ON jobs(job_id, registered_at);

CREATE TABLE logical_runs (
    logical_run_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    job_spec_sha256 TEXT NOT NULL REFERENCES jobs(spec_sha256),
    scheduled_for TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'succeeded')),
    successful_attempt_id TEXT,
    UNIQUE(job_spec_sha256, scheduled_for)
) STRICT;

CREATE TABLE attempts (
    attempt_id TEXT PRIMARY KEY,
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    owner_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started', 'failed', 'abandoned', 'succeeded')),
    started_at TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    finished_at TEXT,
    terminal_code TEXT,
    UNIQUE(logical_run_id, attempt_number)
) STRICT;
CREATE UNIQUE INDEX one_started_attempt_per_run ON attempts(logical_run_id) WHERE state = 'started';
CREATE UNIQUE INDEX one_succeeded_attempt_per_run ON attempts(logical_run_id) WHERE state = 'succeeded';

CREATE TABLE artifacts (
    sha256 TEXT PRIMARY KEY CHECK (length(sha256) = 64),
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    relative_path TEXT NOT NULL UNIQUE,
    first_recorded_at TEXT NOT NULL
) STRICT;

CREATE TABLE attempt_artifacts (
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    role TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
    attached_at TEXT NOT NULL,
    PRIMARY KEY(attempt_id, role)
) STRICT;

CREATE TRIGGER metadata_no_update BEFORE UPDATE ON metadata BEGIN SELECT RAISE(ABORT, 'metadata is immutable'); END;
CREATE TRIGGER metadata_no_delete BEFORE DELETE ON metadata BEGIN SELECT RAISE(ABORT, 'metadata is immutable'); END;
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
CREATE TRIGGER jobs_no_update BEFORE UPDATE ON jobs BEGIN SELECT RAISE(ABORT, 'jobs are immutable'); END;
CREATE TRIGGER jobs_no_delete BEFORE DELETE ON jobs BEGIN SELECT RAISE(ABORT, 'jobs are immutable'); END;
CREATE TRIGGER artifacts_no_update BEFORE UPDATE ON artifacts BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END;
CREATE TRIGGER artifacts_no_delete BEFORE DELETE ON artifacts BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END;
CREATE TRIGGER attachments_no_update BEFORE UPDATE ON attempt_artifacts
BEGIN SELECT RAISE(ABORT, 'attachments are immutable'); END;
CREATE TRIGGER attachments_no_delete BEFORE DELETE ON attempt_artifacts
BEGIN SELECT RAISE(ABORT, 'attachments are immutable'); END;
"""
