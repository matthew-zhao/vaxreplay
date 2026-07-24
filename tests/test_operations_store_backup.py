from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vaxreplay.operations.store import OperationalStore
from vaxreplay.operations.store_backup import (
    StoreBackupError,
    audit_store_orphans,
    create_store_backup,
    restore_store_backup,
    verify_store_backup,
)

_NOW = datetime(2026, 7, 14, 20, tzinfo=timezone.utc)


def _store(root: Path) -> tuple[OperationalStore, bytes]:
    store = OperationalStore.initialize(root, created_at=_NOW)
    payload = b'retained capture bytes'
    store.put_bytes(payload, recorded_at=_NOW + timedelta(seconds=1))
    return store, payload


def test_backup_create_verify_and_clean_restore_roundtrip(tmp_path: Path) -> None:
    store, _payload = _store(tmp_path / 'source')
    checkpoint = store.checkpoint(created_at=_NOW + timedelta(seconds=2))
    manifest = create_store_backup(
        store,
        tmp_path / 'backup',
        backup_id='nightly-20260714',
        created_at=_NOW + timedelta(seconds=3),
        checkpoint=checkpoint,
    )
    assert len(manifest.objects) == 1
    assert manifest.source_checkpoint == checkpoint
    assert not manifest.independent_timestamp_proof_included

    verification = verify_store_backup(tmp_path / 'backup')
    assert verification.every_object_verified
    assert verification.sqlite_and_ledger_verified
    assert verification.successful_run_semantics_verified
    assert verification.clean_root_restore_verified
    assert not verification.off_host_retention_verified

    restored = restore_store_backup(tmp_path / 'backup', tmp_path / 'restored')
    assert restored.store_id == store.store_id
    assert restored.orphan_free
    assert restored.source_checkpoint_prefix_verified
    recovered = OperationalStore(tmp_path / 'restored')
    assert recovered.verify(checkpoint=checkpoint, verified_at=_NOW + timedelta(seconds=3)).object_count == 1
    # A restore is an operational store, not a read-only archival copy.
    recovered.put_bytes(b'post-restore object', recorded_at=_NOW + timedelta(seconds=4))
    assert recovered.verify(verified_at=_NOW + timedelta(seconds=4)).object_count == 2


def test_backup_verification_rejects_tamper_and_extra_empty_directories(tmp_path: Path) -> None:
    store, payload = _store(tmp_path / 'source')
    create_store_backup(
        store,
        tmp_path / 'backup',
        backup_id='tamper-fixture',
        created_at=_NOW + timedelta(seconds=2),
    )
    digest = hashlib.sha256(payload).hexdigest()
    object_path = tmp_path / 'backup' / 'objects' / 'sha256' / digest[:2] / digest
    object_path.chmod(0o600)
    object_path.write_bytes(b'forged')
    with pytest.raises(StoreBackupError, match='object differs'):
        verify_store_backup(tmp_path / 'backup')

    store, _payload = _store(tmp_path / 'source-two')
    create_store_backup(
        store,
        tmp_path / 'backup-two',
        backup_id='extra-directory-fixture',
        created_at=_NOW + timedelta(seconds=2),
    )
    (tmp_path / 'backup-two' / 'uncommitted-empty-directory').mkdir()
    with pytest.raises(StoreBackupError, match='directory inventory'):
        verify_store_backup(tmp_path / 'backup-two')


def test_backup_and_restore_refuse_existing_destinations(tmp_path: Path) -> None:
    store, _payload = _store(tmp_path / 'source')
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(StoreBackupError, match='must not exist'):
        create_store_backup(store, existing, backup_id='no-overwrite')

    create_store_backup(
        store,
        tmp_path / 'backup',
        backup_id='restore-no-overwrite',
        created_at=_NOW + timedelta(seconds=2),
    )
    with pytest.raises(StoreBackupError, match='must not exist'):
        restore_store_backup(tmp_path / 'backup', existing)


def test_orphan_audit_is_read_only_and_reports_every_drift_class(tmp_path: Path) -> None:
    store, payload = _store(tmp_path / 'source')
    clean = audit_store_orphans(store, audited_at=_NOW + timedelta(seconds=2))
    assert clean.clean
    assert clean.registered_object_count == 1

    orphan_payload = b'orphan bytes'
    orphan_digest = hashlib.sha256(orphan_payload).hexdigest()
    orphan_path = store.objects_root / orphan_digest[:2] / orphan_digest
    orphan_path.parent.mkdir(exist_ok=True)
    orphan_path.write_bytes(orphan_payload)
    temporary_path = store.objects_root / '.tmp' / 'interrupted-copy'
    temporary_path.write_bytes(b'partial')
    unsafe_path = store.objects_root / 'unexpected-entry'
    unsafe_path.write_bytes(b'unsafe')
    registered_digest = hashlib.sha256(payload).hexdigest()
    registered_path = store.objects_root / registered_digest[:2] / registered_digest
    registered_path.unlink()

    before = {
        path.relative_to(store.root).as_posix(): path.read_bytes() for path in store.root.rglob('*') if path.is_file()
    }
    report = audit_store_orphans(store, audited_at=_NOW + timedelta(seconds=3))
    after = {
        path.relative_to(store.root).as_posix(): path.read_bytes() for path in store.root.rglob('*') if path.is_file()
    }
    assert before == after
    assert not report.clean
    assert report.orphan_paths == (f'objects/sha256/{orphan_digest[:2]}/{orphan_digest}',)
    assert report.temporary_paths == ('objects/sha256/.tmp/interrupted-copy',)
    assert report.missing_registered_paths == (f'objects/sha256/{registered_digest[:2]}/{registered_digest}',)
    assert report.unsafe_paths == ('objects/sha256/unexpected-entry',)
    assert not report.destructive_action_performed


def test_backup_cli_creates_and_exercises_clean_restore(tmp_path: Path) -> None:
    store, _payload = _store(tmp_path / 'source')
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')}
    create = subprocess.run(
        [
            sys.executable,
            '-m',
            'vaxreplay.operations.store_backup_cli',
            'create',
            '--root',
            str(store.root),
            '--output',
            str(tmp_path / 'backup'),
            '--backup-id',
            'cli-fixture',
        ],
        check=False,
        capture_output=True,
        env=environment,
    )
    assert create.returncode == 0, create.stderr.decode()
    assert json.loads(create.stdout)['backup_id'] == 'cli-fixture'
    verify = subprocess.run(
        [
            sys.executable,
            '-m',
            'vaxreplay.operations.store_backup_cli',
            'verify',
            '--backup',
            str(tmp_path / 'backup'),
        ],
        check=False,
        capture_output=True,
        env=environment,
    )
    assert verify.returncode == 0, verify.stderr.decode()
    assert json.loads(verify.stdout)['clean_root_restore_verified'] is True
