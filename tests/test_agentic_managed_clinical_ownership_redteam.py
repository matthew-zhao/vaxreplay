from __future__ import annotations

import hashlib
import os
import shutil
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

import vaxreplay.agentic.managed_clinical_ownership as ownership_module
from vaxreplay.agentic.managed_clinical_ownership import (
    AuthenticatedManagedClinicalOwnership,
    LinuxManagedClinicalHostAdapter,
    ManagedClinicalOwnershipError,
    ManagedClinicalOwnershipRecord,
)
from vaxreplay.agentic.managed_clinical_startup import (
    ManagedClinicalStartupConfig,
    managed_clinical_cleanup_key_id,
)

KEY = b'vaxreplay-red-team-owned-artifact-key'
RUN_ID = 'a' * 32
AUTHORITY = 'organizer.lane-a.example'
PGID = 4242
SESSION_ID = 4000
FIRECRACKER_PID = 4243
FIRECRACKER_START_TICKS = 123_456
NOW = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)


def _record(root: Path, cgroup_root: Path) -> ManagedClinicalOwnershipRecord:
    run_container = root / RUN_ID
    jail_root = run_container / 'root'
    return ManagedClinicalOwnershipRecord(
        ledger_id='lane-a-host-ownership',
        registry_authority_id=AUTHORITY,
        sequence=2,
        previous_envelope_sha256='1' * 64,
        state='start_bound',
        run_id=RUN_ID,
        reservation_sha256='2' * 64,
        launch_sha256='3' * 64,
        start_redemption_sha256='4' * 64,
        episode_id='episode-001',
        worker_spec_sha256='5' * 64,
        prepared_worker_sha256='6' * 64,
        run_container_path=str(run_container),
        jail_root_path=str(jail_root),
        vsock_path=str(jail_root / 'run' / 'vsock.sock'),
        cgroup_path=str(cgroup_root / RUN_ID),
        capability_id='7' * 64,
        recorded_at=NOW,
    )


def _envelope(record: ManagedClinicalOwnershipRecord) -> AuthenticatedManagedClinicalOwnership:
    return AuthenticatedManagedClinicalOwnership(
        record=record,
        ownership_key_id=managed_clinical_cleanup_key_id(KEY),
        ownership_hmac_sha256='8' * 64,
    )


def _startup_config(root: Path, cgroup_root: Path) -> ManagedClinicalStartupConfig:
    return ManagedClinicalStartupConfig(
        reconciler_id='lane-a-startup-reaper',
        reconciler_version='red-team-test-v1',
        registry_authority_id=AUTHORITY,
        runtime_config_sha256='9' * 64,
        worker_spec_sha256='5' * 64,
        cleanup_receipt_key_id=managed_clinical_cleanup_key_id(KEY),
        cgroup_root=str(cgroup_root),
        jail_root=str(root),
        vsock_root=str(root),
        receipt_root=str(root.parent / 'receipts'),
        cleanup_grace_seconds=1,
    )


def _adapter(
    *,
    root: Path,
    cgroup_root: Path,
    record: ManagedClinicalOwnershipRecord,
    executable_sha256: str = 'b' * 64,
) -> LinuxManagedClinicalHostAdapter:
    adapter = object.__new__(LinuxManagedClinicalHostAdapter)
    adapter.config = _startup_config(root, cgroup_root)
    adapter.ownership = cast(
        object,
        SimpleNamespace(
            active=lambda: (_envelope(record),),
            config=SimpleNamespace(
                proc_root=str(root.parent / 'proc'),
                firecracker_executable_name='firecracker',
                firecracker_executable_sha256=executable_sha256,
            ),
        ),
    )
    adapter._key = KEY  # noqa: SLF001
    return adapter


def _bind_unix_socket(path: Path) -> None:
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        endpoint.bind(str(path))
    finally:
        endpoint.close()


def _write_proc_stat(path: Path) -> None:
    path.parent.mkdir(parents=True)
    fields = ['S', '1', str(PGID), str(SESSION_ID)]
    fields.extend(['0'] * 15)
    fields.append(str(FIRECRACKER_START_TICKS))
    path.write_text(
        f'{FIRECRACKER_PID} (firecracker child) {" ".join(fields)}\n',
        encoding='ascii',
    )


def _install_trusted_pid_file_view(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pid_file: Path,
) -> None:
    """Model jailer's root:0600 file on an unprivileged unit-test host."""

    real_require_path_type = ownership_module._require_path_type

    def require_path_type(path: Path, *, directory: bool):
        metadata = real_require_path_type(path, directory=directory)
        if path != pid_file:
            return metadata
        return SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mode=(metadata.st_mode & ~0o7777) | 0o600,
            st_nlink=1,
            st_uid=0,
        )

    def read_pid_file(
        path: Path,
        *,
        expected_device: int,
        expected_inode: int,
        expected_owner_uid: int,
        expected_mode: int,
    ) -> int:
        metadata = path.lstat()
        assert (expected_device, expected_inode) == (metadata.st_dev, metadata.st_ino)
        assert expected_owner_uid == 0
        assert expected_mode == 0o600
        return int(path.read_text(encoding='ascii').strip())

    monkeypatch.setattr(ownership_module, '_require_path_type', require_path_type)
    monkeypatch.setattr(ownership_module, '_read_pid_file', read_pid_file)


def _recovery_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trusted_pid_file: bool = True,
) -> tuple[LinuxManagedClinicalHostAdapter, ManagedClinicalOwnershipRecord, os.stat_result, str]:
    root = tmp_path / 'jails'
    cgroup_root = tmp_path / 'cgroups'
    record = _record(root, cgroup_root)
    jail_root = Path(record.jail_root_path)
    jail_root.mkdir(parents=True)
    cgroup = Path(record.cgroup_path)
    cgroup.mkdir(parents=True)
    (cgroup / 'cgroup.procs').write_text(f'{FIRECRACKER_PID}\n', encoding='ascii')
    pid_file = jail_root / 'firecracker.pid'
    pid_file.write_text(f'{FIRECRACKER_PID}\n', encoding='ascii')
    pid_file.chmod(0o600 if trusted_pid_file else 0o644)
    proc_root = tmp_path / 'proc'
    _write_proc_stat(proc_root / str(FIRECRACKER_PID) / 'stat')
    executable = tmp_path / 'firecracker.bin'
    executable.write_bytes(b'pinned-firecracker-executable')
    (proc_root / str(FIRECRACKER_PID) / 'exe').hardlink_to(executable)
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    monkeypatch.setattr(ownership_module, '_linux_fd_mount_id', lambda _fd: 1)
    if trusted_pid_file:
        _install_trusted_pid_file_view(monkeypatch, pid_file=pid_file)
    adapter = _adapter(
        root=root,
        cgroup_root=cgroup_root,
        record=record,
        executable_sha256=executable_sha256,
    )
    return adapter, record, cgroup.stat(), executable_sha256


def test_vsock_scan_enumerates_namespace_and_rejects_an_extra_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = Path(tempfile.mkdtemp(prefix='vr-vsock-', dir='/tmp'))
    try:
        root = sandbox / 'jails'
        cgroup_root = sandbox / 'cgroups'
        record = _record(root, cgroup_root)
        expected = Path(record.vsock_path)
        expected.parent.mkdir(parents=True)
        cgroup_root.mkdir()
        _bind_unix_socket(expected)
        monkeypatch.setattr(ownership_module, '_linux_fd_mount_id', lambda _fd: 1)
        adapter = _adapter(root=root, cgroup_root=cgroup_root, record=record)

        artifacts = adapter.scan_vsock_endpoints()

        assert tuple(item.artifact_id for item in artifacts) == (str(expected),)
        rogue = expected.with_name('rogue.sock')
        _bind_unix_socket(rogue)
        with pytest.raises(ManagedClinicalOwnershipError, match='unrepresentable socket'):
            adapter.scan_vsock_endpoints()
        rogue.unlink()
        expected.unlink()
        expected.write_text('not a socket', encoding='ascii')
        with pytest.raises(ManagedClinicalOwnershipError, match='changed type or identity'):
            adapter.scan_vsock_endpoints()
    finally:
        shutil.rmtree(sandbox)


def test_vsock_scan_rejects_unknown_top_level_and_nested_symlink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = Path(tempfile.mkdtemp(prefix='vr-vsock-', dir='/tmp'))
    try:
        root = sandbox / 'jails'
        cgroup_root = sandbox / 'cgroups'
        record = _record(root, cgroup_root)
        expected = Path(record.vsock_path)
        expected.parent.mkdir(parents=True)
        cgroup_root.mkdir()
        _bind_unix_socket(expected)
        monkeypatch.setattr(ownership_module, '_linux_fd_mount_id', lambda _fd: 1)
        adapter = _adapter(root=root, cgroup_root=cgroup_root, record=record)
        unknown = root / ('f' * 32)
        unknown.mkdir()

        with pytest.raises(ManagedClinicalOwnershipError, match='unowned top-level'):
            adapter.scan_vsock_endpoints()

        unknown.rmdir()
        (Path(record.jail_root_path) / 'alias').symlink_to(sandbox)
        with pytest.raises(ManagedClinicalOwnershipError, match='unexpected symlink'):
            adapter.scan_vsock_endpoints()
    finally:
        shutil.rmtree(sandbox)


def test_vsock_scan_rejects_a_cross_mount_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = Path(tempfile.mkdtemp(prefix='vr-vsock-', dir='/tmp'))
    try:
        root = sandbox / 'jails'
        cgroup_root = sandbox / 'cgroups'
        record = _record(root, cgroup_root)
        Path(record.vsock_path).parent.mkdir(parents=True)
        cgroup_root.mkdir()
        mount_ids = iter((1, 2))
        monkeypatch.setattr(
            ownership_module,
            '_linux_fd_mount_id',
            lambda _fd: next(mount_ids),
        )
        adapter = _adapter(root=root, cgroup_root=cgroup_root, record=record)

        with pytest.raises(ManagedClinicalOwnershipError, match='mount boundary'):
            adapter.scan_vsock_endpoints()
    finally:
        shutil.rmtree(sandbox)


def test_crash_before_running_record_uses_child_witness_not_false_group_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, record, metadata, _executable_sha256 = _recovery_fixture(
        tmp_path,
        monkeypatch,
    )

    recovered = adapter._live_process_identity(  # noqa: SLF001
        record,
        expected_cgroup_device=metadata.st_dev,
        expected_cgroup_inode=metadata.st_ino,
    )

    assert recovered is not None
    assert recovered.process_group_id == PGID
    assert recovered.session_id == SESSION_ID
    assert recovered.identity_source == 'recovered-firecracker-child'
    assert recovered.process_group_leader_start_time_ticks is None
    assert recovered.witness_pid == FIRECRACKER_PID
    assert recovered.witness_start_time_ticks == FIRECRACKER_START_TICKS
    artifacts = adapter.scan_process_groups()
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_id == f'pgid:{PGID}'
    assert artifact.process_identity_source == 'recovered-firecracker-child'
    assert artifact.process_group_leader_start_time_ticks is None
    assert artifact.process_witness_pid == FIRECRACKER_PID
    assert artifact.process_witness_start_time_ticks == FIRECRACKER_START_TICKS


def test_crash_recovery_rejects_a_non_root_or_non_0600_pid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, record, metadata, _executable_sha256 = _recovery_fixture(
        tmp_path,
        monkeypatch,
        trusted_pid_file=False,
    )

    with pytest.raises(ManagedClinicalOwnershipError, match='root:0600 identity'):
        adapter._live_process_identity(  # noqa: SLF001
            record,
            expected_cgroup_device=metadata.st_dev,
            expected_cgroup_inode=metadata.st_ino,
        )


def test_crash_recovery_rejects_cgroup_membership_changing_during_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, record, metadata, _executable_sha256 = _recovery_fixture(
        tmp_path,
        monkeypatch,
    )
    real_read = ownership_module._read_cgroup_process_ids
    calls = 0

    def changing_read(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        return real_read(*args, **kwargs) if calls == 1 else ()

    monkeypatch.setattr(ownership_module, '_read_cgroup_process_ids', changing_read)

    with pytest.raises(ManagedClinicalOwnershipError, match='changed while its identity was pinned'):
        adapter._live_process_identity(  # noqa: SLF001
            record,
            expected_cgroup_device=metadata.st_dev,
            expected_cgroup_inode=metadata.st_ino,
        )


def test_crash_recovery_rejects_executable_changing_during_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, record, metadata, executable_sha256 = _recovery_fixture(
        tmp_path,
        monkeypatch,
    )
    calls = 0

    def changing_executable_sha256(*_args, **_kwargs) -> str:
        nonlocal calls
        calls += 1
        return executable_sha256 if calls == 1 else 'f' * 64

    monkeypatch.setattr(
        ownership_module,
        '_proc_executable_sha256',
        changing_executable_sha256,
    )

    with pytest.raises(ManagedClinicalOwnershipError, match='changed while its identity was pinned'):
        adapter._live_process_identity(  # noqa: SLF001
            record,
            expected_cgroup_device=metadata.st_dev,
            expected_cgroup_inode=metadata.st_ino,
        )


def test_startup_config_rejects_a_fictional_separate_vsock_root(tmp_path: Path) -> None:
    config = _startup_config(tmp_path / 'jails', tmp_path / 'cgroups')

    with pytest.raises(ValidationError, match='vsock root must be the jail namespace root'):
        ManagedClinicalStartupConfig.model_validate(
            {**config.model_dump(mode='python'), 'vsock_root': str(tmp_path / 'vsock-index')}
        )
