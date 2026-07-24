"""Consistent OperationalStore backup, restore, and read-only orphan audit.

Backups are ordinary directories so an operator can encrypt and archive them with a
separate, reviewed system.  This module never claims off-host retention, immutability,
encryption, or an independently witnessed timestamp.  It does make the SQLite/CAS
snapshot internally complete and exercises every verified backup in a fresh root.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.collector_semantics import verify_all_supported_run_manifests
from vaxreplay.operations.schema import (
    SAFE_ID_PATTERN,
    LedgerCheckpoint,
    LedgerEventType,
    StoredArtifact,
    aware_utc,
    checkpoint_sha256,
)
from vaxreplay.operations.store import OperationalStore

STORE_BACKUP_SCHEMA_VERSION = 'vaxreplay.operations-store-backup.v0.1'
STORE_BACKUP_REPORT_SCHEMA_VERSION = 'vaxreplay.operations-store-backup-report.v0.1'
STORE_RESTORE_REPORT_SCHEMA_VERSION = 'vaxreplay.operations-store-restore-report.v0.1'
ORPHAN_AUDIT_REPORT_SCHEMA_VERSION = 'vaxreplay.operations-store-orphan-audit.v0.1'

_DATABASE_NAME = 'operations.sqlite3'
_MANIFEST_NAME = 'manifest.json'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_OBJECT_RELATIVE_PATTERN = r'^objects/sha256/[0-9a-f]{2}/[0-9a-f]{64}$'
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_DATABASE_BYTES = 64 * 1024 * 1024 * 1024
_MAX_OBJECTS = 10_000_000
_COPY_CHUNK = 1024 * 1024


class StoreBackupError(RuntimeError):
    """Backup, restore, or offline inventory verification failed closed."""


class BackupMaterial(StrictModel):
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)


class BackupObject(StrictModel):
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)
    relative_path: str = Field(pattern=_OBJECT_RELATIVE_PATTERN)

    @model_validator(mode='after')
    def validate_derived_path(self) -> Self:
        if self.relative_path != f'objects/sha256/{self.sha256[:2]}/{self.sha256}':
            raise ValueError('backup object relative_path must be derived from sha256')
        return self


class StoreBackupManifest(StrictModel):
    schema_version: Literal['vaxreplay.operations-store-backup.v0.1'] = STORE_BACKUP_SCHEMA_VERSION
    backup_id: str = Field(pattern=SAFE_ID_PATTERN)
    created_at: datetime
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    database: BackupMaterial
    event_count: int = Field(ge=1)
    through_sequence: int = Field(ge=1)
    through_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    object_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    objects: tuple[BackupObject, ...] = Field(max_length=_MAX_OBJECTS)
    successful_run_semantics_verified: Literal[True] = True
    semantic_verifier_scope: Literal['all_successful_supported_runs'] = 'all_successful_supported_runs'
    source_checkpoint: LedgerCheckpoint | None = None
    source_checkpoint_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    independent_timestamp_proof_included: Literal[False] = False
    encrypted_at_rest_claimed: Literal[False] = False
    off_host_copy_claimed: Literal[False] = False

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'backup created_at')

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        if self.through_sequence != self.event_count:
            raise ValueError('backup must cover the complete local event prefix')
        keys = tuple((item.sha256, item.byte_count) for item in self.objects)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('backup objects must use unique sha256 sort order')
        expected_inventory_sha256 = hashlib.sha256(canonical_json_bytes(keys)).hexdigest()
        if self.object_inventory_sha256 != expected_inventory_sha256:
            raise ValueError('object_inventory_sha256 does not bind backup objects')
        if (self.source_checkpoint is None) != (self.source_checkpoint_sha256 is None):
            raise ValueError('source checkpoint and digest must be supplied together')
        if self.source_checkpoint is not None:
            if checkpoint_sha256(self.source_checkpoint) != self.source_checkpoint_sha256:
                raise ValueError('source_checkpoint_sha256 does not bind source_checkpoint')
            if self.source_checkpoint.store_id != self.store_id:
                raise ValueError('source checkpoint belongs to a different store')
            if self.source_checkpoint.through_sequence > self.through_sequence:
                raise ValueError('source checkpoint extends beyond the backup ledger')
        return self


class StoreBackupVerificationReport(StrictModel):
    schema_version: Literal['vaxreplay.operations-store-backup-report.v0.1'] = STORE_BACKUP_REPORT_SCHEMA_VERSION
    backup_id: str = Field(pattern=SAFE_ID_PATTERN)
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    database_sha256: str = Field(pattern=_SHA256_PATTERN)
    event_count: int = Field(ge=1)
    through_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    object_count: int = Field(ge=0)
    exact_file_inventory_verified: Literal[True] = True
    every_object_verified: Literal[True] = True
    sqlite_integrity_verified: Literal[True] = True
    sqlite_and_ledger_verified: bool
    successful_run_semantics_verified: bool
    clean_root_restore_verified: bool
    independent_timestamp_proof_verified: Literal[False] = False
    off_host_retention_verified: Literal[False] = False


class StoreRestoreVerificationReport(StrictModel):
    schema_version: Literal['vaxreplay.operations-store-restore-report.v0.1'] = STORE_RESTORE_REPORT_SCHEMA_VERSION
    backup_id: str = Field(pattern=SAFE_ID_PATTERN)
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    restored_root: str = Field(min_length=1, max_length=4096)
    event_count: int = Field(ge=1)
    through_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    object_count: int = Field(ge=0)
    sqlite_and_ledger_verified: Literal[True] = True
    successful_run_semantics_verified: Literal[True] = True
    orphan_free: Literal[True] = True
    source_checkpoint_prefix_verified: bool


class StoreOrphanAuditReport(StrictModel):
    schema_version: Literal['vaxreplay.operations-store-orphan-audit.v0.1'] = ORPHAN_AUDIT_REPORT_SCHEMA_VERSION
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    audited_at: datetime
    registered_object_count: int = Field(ge=0)
    present_registered_object_count: int = Field(ge=0)
    orphan_paths: tuple[str, ...]
    temporary_paths: tuple[str, ...]
    missing_registered_paths: tuple[str, ...]
    corrupt_registered_paths: tuple[str, ...]
    unsafe_paths: tuple[str, ...]
    clean: bool
    destructive_action_performed: Literal[False] = False
    ledger_structural_verification_performed: Literal[False] = False

    @field_validator('audited_at')
    @classmethod
    def validate_audited_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'orphan audit audited_at')

    @field_validator(
        'orphan_paths',
        'temporary_paths',
        'missing_registered_paths',
        'corrupt_registered_paths',
        'unsafe_paths',
    )
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('orphan-audit paths must be sorted and unique')
        if any(not item or len(item) > 4096 or item.startswith('/') or '\x00' in item for item in value):
            raise ValueError('orphan-audit paths must be bounded safe relative paths')
        return value

    @model_validator(mode='after')
    def validate_clean_flag(self) -> Self:
        expected_clean = not any(
            (
                self.orphan_paths,
                self.temporary_paths,
                self.missing_registered_paths,
                self.corrupt_registered_paths,
                self.unsafe_paths,
            )
        )
        if self.clean != expected_clean:
            raise ValueError('clean must reflect every orphan-audit finding category')
        if self.present_registered_object_count > self.registered_object_count:
            raise ValueError('present registered count cannot exceed registered count')
        return self


def create_store_backup(
    store: OperationalStore,
    destination: Path,
    *,
    backup_id: str,
    created_at: datetime | None = None,
    checkpoint: LedgerCheckpoint | None = None,
) -> StoreBackupManifest:
    """Create one atomically published SQLite/CAS snapshot under a writer lock."""

    created_at = aware_utc(created_at or datetime.now(timezone.utc), 'backup created_at')
    destination = _require_new_destination(destination)
    parent = destination.parent
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.tmp-', dir=parent))
    os.chmod(temporary, 0o700)
    try:
        (temporary / 'objects' / 'sha256').mkdir(parents=True, mode=0o700)
        with store.verification_window():
            report = store.verify(checkpoint=checkpoint, verified_at=created_at)
            verify_all_supported_run_manifests(store)
            database_path = temporary / _DATABASE_NAME
            _sqlite_online_backup(store.database_path, database_path)
            database_inventory = _read_database_inventory(database_path)
            if (
                database_inventory.store_id != report.store_id
                or database_inventory.event_count != report.event_count
                or database_inventory.through_event_sha256 != report.ledger_head_sha256
            ):
                raise StoreBackupError('SQLite backup differs from the writer-locked verified source')
            for item in database_inventory.objects:
                _copy_registered_object(store.root / item.relative_path, temporary / item.relative_path, item)

        database = _hash_regular_file(temporary / _DATABASE_NAME, maximum=_MAX_DATABASE_BYTES)
        manifest = StoreBackupManifest(
            backup_id=backup_id,
            created_at=created_at,
            store_id=report.store_id,
            database=database,
            event_count=report.event_count,
            through_sequence=report.event_count,
            through_event_sha256=report.ledger_head_sha256,
            object_inventory_sha256=database_inventory.object_inventory_sha256,
            objects=database_inventory.objects,
            source_checkpoint=checkpoint,
            source_checkpoint_sha256=checkpoint_sha256(checkpoint) if checkpoint is not None else None,
        )
        _write_exclusive(temporary / _MANIFEST_NAME, canonical_json_bytes(manifest), mode=0o400)
        _fsync_tree(temporary)
        if destination.exists() or destination.is_symlink():
            raise StoreBackupError('backup destination appeared during creation')
        os.rename(temporary, destination)
        _fsync_directory(parent)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_store_backup(
    backup_root: Path,
    *,
    clean_restore_test: bool = True,
) -> StoreBackupVerificationReport:
    """Verify exact backup bytes and, by default, restore into a fresh temporary root."""

    root = _require_existing_directory(backup_root, 'backup root')
    manifest_bytes = _read_regular_file(root / _MANIFEST_NAME, maximum=_MAX_MANIFEST_BYTES)
    manifest = _parse_manifest(manifest_bytes)
    _verify_exact_backup_file_inventory(root, manifest)
    database = _hash_regular_file(root / _DATABASE_NAME, maximum=_MAX_DATABASE_BYTES)
    if database != manifest.database:
        raise StoreBackupError('backup database differs from its manifest binding')
    for item in manifest.objects:
        if _hash_regular_file(root / item.relative_path) != BackupMaterial(
            sha256=item.sha256,
            byte_count=item.byte_count,
        ):
            raise StoreBackupError(f'backup object differs from its manifest binding: {item.relative_path}')
    inventory = _read_database_inventory(root / _DATABASE_NAME)
    _require_database_matches_manifest(inventory, manifest)

    clean_verified = False
    if clean_restore_test:
        with tempfile.TemporaryDirectory(prefix='vaxreplay-backup-restore-') as temp_dir:
            target = Path(temp_dir) / 'store'
            restored = _restore_verified_payload(root, manifest, target)
            clean_verified = restored.orphan_free
    return StoreBackupVerificationReport(
        backup_id=manifest.backup_id,
        store_id=manifest.store_id,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        database_sha256=manifest.database.sha256,
        event_count=manifest.event_count,
        through_event_sha256=manifest.through_event_sha256,
        object_count=len(manifest.objects),
        sqlite_and_ledger_verified=clean_verified,
        successful_run_semantics_verified=clean_verified,
        clean_root_restore_verified=clean_verified,
    )


def restore_store_backup(backup_root: Path, destination: Path) -> StoreRestoreVerificationReport:
    """Restore only into a nonexistent root, then run full structural and semantic replay."""

    root = _require_existing_directory(backup_root, 'backup root')
    verify_store_backup(root, clean_restore_test=False)
    manifest = _parse_manifest(_read_regular_file(root / _MANIFEST_NAME, maximum=_MAX_MANIFEST_BYTES))
    destination = _require_new_destination(destination)
    return _restore_verified_payload(root, manifest, destination)


def audit_store_orphans(
    store: OperationalStore,
    *,
    audited_at: datetime | None = None,
    max_entries: int = _MAX_OBJECTS,
) -> StoreOrphanAuditReport:
    """Inventory CAS drift read-only while preventing a concurrent writer false positive."""

    if max_entries < 1 or max_entries > _MAX_OBJECTS:
        raise ValueError(f'max_entries must be between 1 and {_MAX_OBJECTS}')
    audited_at = aware_utc(audited_at or datetime.now(timezone.utc), 'orphan audit audited_at')
    with store.verification_window():
        return _audit_store_orphans_in_verification_window(
            store,
            audited_at=audited_at,
            max_entries=max_entries,
        )


def _audit_store_orphans_in_verification_window(
    store: OperationalStore,
    *,
    audited_at: datetime,
    max_entries: int,
) -> StoreOrphanAuditReport:
    expected: dict[str, int] = {}
    for event in store.events():
        if event.event_type is not LedgerEventType.ARTIFACT_STORED:
            continue
        digest = event.payload.get('artifact_sha256')
        byte_count = event.payload.get('byte_count')
        if (
            not isinstance(digest, str)
            or re.fullmatch(_SHA256_PATTERN, digest) is None
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
            or digest in expected
        ):
            raise StoreBackupError('artifact ledger inventory is malformed or duplicated')
        expected[digest] = byte_count

    orphans: set[str] = set()
    temporary: set[str] = set()
    corrupt: set[str] = set()
    unsafe: set[str] = set()
    present: set[str] = set()
    count = 0
    objects_root = store.objects_root
    try:
        root_entries = tuple(os.scandir(objects_root))
    except OSError as error:
        raise StoreBackupError('cannot scan CAS SHA-256 root') from error
    for entry in root_entries:
        count += 1
        if count > max_entries:
            raise StoreBackupError('orphan audit exceeded max_entries')
        root_relative = f'objects/sha256/{entry.name}'
        if entry.name == '.tmp':
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                unsafe.add(root_relative)
                continue
            for child in os.scandir(entry.path):
                count += 1
                if count > max_entries:
                    raise StoreBackupError('orphan audit exceeded max_entries')
                child_relative = f'{root_relative}/{child.name}'
                if child.is_symlink() or not child.is_file(follow_symlinks=False):
                    unsafe.add(child_relative)
                else:
                    temporary.add(child_relative)
            continue
        if (
            re.fullmatch(r'[0-9a-f]{2}', entry.name) is None
            or entry.is_symlink()
            or not entry.is_dir(follow_symlinks=False)
        ):
            unsafe.add(root_relative)
            continue
        for child in os.scandir(entry.path):
            count += 1
            if count > max_entries:
                raise StoreBackupError('orphan audit exceeded max_entries')
            relative = f'{root_relative}/{child.name}'
            if (
                child.is_symlink()
                or not child.is_file(follow_symlinks=False)
                or re.fullmatch(_SHA256_PATTERN, child.name) is None
                or not child.name.startswith(entry.name)
            ):
                unsafe.add(relative)
                continue
            if child.name not in expected:
                orphans.add(relative)
                continue
            binding = _hash_regular_file(Path(child.path))
            if binding.sha256 != child.name or binding.byte_count != expected[child.name]:
                corrupt.add(relative)
            else:
                present.add(child.name)

    missing = {
        f'objects/sha256/{digest[:2]}/{digest}'
        for digest in expected
        if digest not in present and f'objects/sha256/{digest[:2]}/{digest}' not in corrupt
    }
    report = StoreOrphanAuditReport(
        store_id=store.store_id,
        audited_at=audited_at,
        registered_object_count=len(expected),
        present_registered_object_count=len(present),
        orphan_paths=tuple(sorted(orphans)),
        temporary_paths=tuple(sorted(temporary)),
        missing_registered_paths=tuple(sorted(missing)),
        corrupt_registered_paths=tuple(sorted(corrupt)),
        unsafe_paths=tuple(sorted(unsafe)),
        clean=not any((orphans, temporary, missing, corrupt, unsafe)),
    )
    return report


class _DatabaseInventory(StrictModel):
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    event_count: int = Field(ge=1)
    through_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    object_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    objects: tuple[BackupObject, ...]


def _read_database_inventory(database_path: Path) -> _DatabaseInventory:
    path = database_path.resolve()
    uri = f'file:{path.as_posix()}?mode=ro&immutable=1'
    try:
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            integrity = connection.execute('PRAGMA integrity_check').fetchone()
            if integrity is None or str(integrity[0]) != 'ok':
                raise StoreBackupError('backup SQLite integrity_check failed')
            metadata_rows = connection.execute('SELECT key, value FROM metadata ORDER BY key').fetchall()
            metadata = {str(row['key']): str(row['value']) for row in metadata_rows}
            if set(metadata) != {'schema_version', 'store_id'} or metadata['schema_version'] != (
                'vaxreplay.operations-store.v0.1'
            ):
                raise StoreBackupError('backup database has unsupported metadata')
            event_rows = connection.execute('SELECT sequence, event_sha256 FROM events ORDER BY sequence').fetchall()
            if not event_rows or any(int(row['sequence']) != index for index, row in enumerate(event_rows, 1)):
                raise StoreBackupError('backup database event sequence is not contiguous')
            artifact_rows = connection.execute(
                'SELECT sha256, byte_count, relative_path, first_recorded_at FROM artifacts ORDER BY sha256'
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise StoreBackupError('cannot read backup SQLite inventory') from error
    try:
        artifacts = tuple(
            StoredArtifact(
                sha256=str(row['sha256']),
                byte_count=int(row['byte_count']),
                relative_path=str(row['relative_path']),
                first_recorded_at=datetime.fromisoformat(str(row['first_recorded_at']).replace('Z', '+00:00')),
            )
            for row in artifact_rows
        )
    except (TypeError, ValueError) as error:
        raise StoreBackupError('backup artifact table contains invalid rows') from error
    objects = tuple(
        BackupObject(sha256=item.sha256, byte_count=item.byte_count, relative_path=item.relative_path)
        for item in artifacts
    )
    inventory = tuple((item.sha256, item.byte_count) for item in objects)
    return _DatabaseInventory(
        store_id=metadata['store_id'],
        event_count=len(event_rows),
        through_event_sha256=str(event_rows[-1]['event_sha256']),
        object_inventory_sha256=hashlib.sha256(canonical_json_bytes(inventory)).hexdigest(),
        objects=objects,
    )


def _require_database_matches_manifest(inventory: _DatabaseInventory, manifest: StoreBackupManifest) -> None:
    if (
        inventory.store_id != manifest.store_id
        or inventory.event_count != manifest.event_count
        or inventory.through_event_sha256 != manifest.through_event_sha256
        or inventory.object_inventory_sha256 != manifest.object_inventory_sha256
        or inventory.objects != manifest.objects
    ):
        raise StoreBackupError('backup SQLite inventory differs from manifest')


def _restore_verified_payload(
    backup_root: Path,
    manifest: StoreBackupManifest,
    destination: Path,
) -> StoreRestoreVerificationReport:
    destination = _require_new_destination(destination)
    parent = destination.parent
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.restore-', dir=parent))
    os.chmod(staging, 0o700)
    try:
        (staging / 'objects' / 'sha256').mkdir(parents=True, mode=0o700)
        _copy_bound_file(
            backup_root / _DATABASE_NAME,
            staging / _DATABASE_NAME,
            manifest.database,
            mode=0o600,
        )
        for item in manifest.objects:
            _copy_bound_file(
                backup_root / item.relative_path,
                staging / item.relative_path,
                BackupMaterial(sha256=item.sha256, byte_count=item.byte_count),
                mode=0o440,
            )
        _fsync_tree(staging)
        if destination.exists() or destination.is_symlink():
            raise StoreBackupError('restore destination appeared during restoration')
        os.rename(staging, destination)
        _fsync_directory(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        store = OperationalStore(destination)
        with store.verification_window():
            verification = store.verify(
                checkpoint=manifest.source_checkpoint,
                verified_at=manifest.created_at,
            )
            verify_all_supported_run_manifests(store)
        if (
            verification.store_id != manifest.store_id
            or verification.event_count != manifest.event_count
            or verification.ledger_head_sha256 != manifest.through_event_sha256
            or verification.object_count != len(manifest.objects)
        ):
            raise StoreBackupError('restored store differs from backup manifest')
        audit = audit_store_orphans(store, audited_at=manifest.created_at)
        if not audit.clean:
            raise StoreBackupError('freshly restored store contains CAS inventory drift')
        return StoreRestoreVerificationReport(
            backup_id=manifest.backup_id,
            store_id=manifest.store_id,
            restored_root=str(destination),
            event_count=verification.event_count,
            through_event_sha256=verification.ledger_head_sha256,
            object_count=verification.object_count,
            source_checkpoint_prefix_verified=manifest.source_checkpoint is not None,
        )
    except BaseException:
        # A failed restore is retained for forensic inspection; callers must never
        # mistake the absence of an exception for merely copying bytes.
        raise


def _sqlite_online_backup(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = sqlite3.connect(source_path)
        destination = sqlite3.connect(destination_path)
        try:
            destination.execute('PRAGMA journal_mode=DELETE')
            destination.execute('PRAGMA synchronous=FULL')
            source.backup(destination)
            destination.commit()
        finally:
            destination.close()
            source.close()
    except sqlite3.Error as error:
        raise StoreBackupError('SQLite online backup failed') from error
    os.chmod(destination_path, 0o400)
    _fsync_file(destination_path)
    _fsync_directory(destination_path.parent)


def _copy_registered_object(source: Path, destination: Path, item: BackupObject) -> None:
    _copy_bound_file(
        source,
        destination,
        BackupMaterial(sha256=item.sha256, byte_count=item.byte_count),
    )


def _copy_bound_file(
    source: Path,
    destination: Path,
    binding: BackupMaterial,
    *,
    mode: int = 0o400,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_descriptor = _open_regular_readonly(source)
    try:
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0),
            0o600,
        )
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with (
                os.fdopen(source_descriptor, 'rb', closefd=False) as source_file,
                os.fdopen(destination_descriptor, 'wb', closefd=False) as destination_file,
            ):
                while chunk := source_file.read(_COPY_CHUNK):
                    byte_count += len(chunk)
                    if byte_count > binding.byte_count:
                        raise StoreBackupError('copied file exceeds its committed byte count')
                    digest.update(chunk)
                    destination_file.write(chunk)
                destination_file.flush()
                os.fsync(destination_file.fileno())
                os.fchmod(destination_file.fileno(), mode)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)
    if byte_count != binding.byte_count or digest.hexdigest() != binding.sha256:
        destination.unlink(missing_ok=True)
        raise StoreBackupError('copied file differs from its committed digest or byte count')
    _fsync_directory(destination.parent)


def _verify_exact_backup_file_inventory(root: Path, manifest: StoreBackupManifest) -> None:
    expected = {_MANIFEST_NAME, _DATABASE_NAME, *(item.relative_path for item in manifest.objects)}
    expected_directories = {'objects', 'objects/sha256'} | {
        f'objects/sha256/{item.sha256[:2]}' for item in manifest.objects
    }
    observed: set[str] = set()
    observed_directories: set[str] = set()
    for current_root, directories, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_root)
        if current.is_symlink():
            raise StoreBackupError('backup tree cannot contain symbolic-link directories')
        for directory in tuple(directories):
            path = current / directory
            if path.is_symlink():
                raise StoreBackupError('backup tree cannot contain symbolic links')
            observed_directories.add(path.relative_to(root).as_posix())
        for filename in files:
            path = current / filename
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise StoreBackupError('backup tree entries must be regular non-symlink files')
            observed.add(relative)
    if observed != expected:
        raise StoreBackupError('backup file inventory has missing or extra entries')
    if observed_directories != expected_directories:
        raise StoreBackupError('backup directory inventory has missing or extra entries')


def _parse_manifest(payload: bytes) -> StoreBackupManifest:
    try:
        manifest = StoreBackupManifest.model_validate_json(payload)
    except ValueError as error:
        raise StoreBackupError('backup manifest does not match its strict schema') from error
    if payload != canonical_json_bytes(manifest):
        raise StoreBackupError('backup manifest must use exact canonical JSON bytes')
    return manifest


def _hash_regular_file(path: Path, *, maximum: int | None = None) -> BackupMaterial:
    descriptor = _open_regular_readonly(path)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(descriptor, 'rb', closefd=False) as source:
            while chunk := source.read(_COPY_CHUNK):
                byte_count += len(chunk)
                if maximum is not None and byte_count > maximum:
                    raise StoreBackupError(f'file exceeds maximum byte count: {path}')
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return BackupMaterial(sha256=digest.hexdigest(), byte_count=byte_count)


def _read_regular_file(path: Path, *, maximum: int) -> bytes:
    descriptor = _open_regular_readonly(path)
    try:
        size = os.fstat(descriptor).st_size
        if size > maximum:
            raise StoreBackupError(f'file exceeds maximum byte count: {path}')
        with os.fdopen(descriptor, 'rb', closefd=False) as source:
            payload = source.read(maximum + 1)
        if len(payload) > maximum:
            raise StoreBackupError(f'file exceeds maximum byte count: {path}')
        return payload
    finally:
        os.close(descriptor)


def _open_regular_readonly(path: Path) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0))
    except OSError as error:
        raise StoreBackupError(f'cannot safely open regular file: {path}') from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise StoreBackupError(f'path is not a regular file: {path}')
    return descriptor


def _write_exclusive(path: Path, payload: bytes, *, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StoreBackupError('short write while committing backup manifest')
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _require_new_destination(path: Path) -> Path:
    requested = Path(path).expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise StoreBackupError(f'destination must not exist: {requested}')
    parent = _require_existing_directory(requested.parent, 'destination parent')
    return parent / requested.name


def _require_existing_directory(path: Path, label: str) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise StoreBackupError(f'{label} must be a non-symlink directory')
    return requested.resolve()


def _fsync_file(path: Path) -> None:
    descriptor = _open_regular_readonly(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for current_root, _children, files in os.walk(root, topdown=False, followlinks=False):
        current = Path(current_root)
        for filename in files:
            _fsync_file(current / filename)
        directories.append(current)
    for directory in directories:
        _fsync_directory(directory)
