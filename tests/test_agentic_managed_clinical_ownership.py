from __future__ import annotations

import os
import shutil
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import vaxreplay.agentic.managed_clinical_ownership as ownership_module
from tests.test_agentic_clinical_production_run import RUN_ID
from tests.test_agentic_firecracker_clinical_runtime_boundary import _case
from vaxreplay.agentic.clinical_launcher import ClinicalRuntimeStart
from vaxreplay.agentic.clinical_production_registry import (
    clinical_production_start_redemption_sha256,
)
from vaxreplay.agentic.firecracker import (
    RunningFirecrackerWorker,
    firecracker_model_sha256,
)
from vaxreplay.agentic.managed_clinical_ownership import (
    DurableManagedClinicalCapabilityLedger,
    DurableManagedClinicalOwnershipLedger,
    LinuxManagedClinicalHostAdapter,
    ManagedClinicalOwnershipConfig,
    ManagedClinicalOwnershipError,
)
from vaxreplay.agentic.managed_clinical_startup import (
    ManagedClinicalHostArtifact,
    ManagedClinicalStartupConfig,
    managed_clinical_cleanup_key_id,
)

KEY = b'managed-clinical-ownership-key-001'
PID = 4242
SESSION_ID = 4000
FIRECRACKER_PID = 4243
START_TICKS = 123_456


def _write_proc_stat(
    proc_root: Path,
    *,
    pid: int,
    process_group_id: int,
    session_id: int,
    start_ticks: int,
) -> None:
    root = proc_root / str(pid)
    root.mkdir(parents=True)
    fields = ['S', '1', str(process_group_id), str(session_id)]
    fields.extend(['0'] * 15)
    fields.append(str(start_ticks))
    (root / 'stat').write_text(
        f'{pid} (managed jailer) {" ".join(fields)}\n',
        encoding='ascii',
    )


def _ownership_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    advance_running: bool = True,
) -> tuple[
    DurableManagedClinicalOwnershipLedger,
    LinuxManagedClinicalHostAdapter,
    DurableManagedClinicalCapabilityLedger,
    ClinicalRuntimeStart,
    tuple[ManagedClinicalHostArtifact, ...],
    list[str],
]:
    # The production adapter is Linux-only.  macOS unit tests supply a stable fake mount ID while
    # retaining the actual fd-relative no-follow traversal behavior.
    monkeypatch.setattr(ownership_module, '_linux_fd_mount_id', lambda _fd: 1)
    case = _case(tmp_path / 'case', monkeypatch)
    spec = case.runtime.supervisor.spec
    jail_namespace = Path(spec.chroot_base_dir) / Path(spec.runtime.firecracker.source_path).name
    cgroup_namespace = tmp_path / 'cgroups'
    proc_root = tmp_path / 'proc'
    jail_namespace.mkdir(parents=True, exist_ok=True)
    cgroup_namespace.mkdir(parents=True)
    proc_root.mkdir()
    config = ManagedClinicalOwnershipConfig(
        ledger_id='lane-a-host-ownership',
        ledger_version='test-v1',
        registry_authority_id=case.request.reservation.registry_authority_id,
        worker_spec_sha256=firecracker_model_sha256(spec),
        firecracker_executable_sha256=spec.runtime.firecracker.sha256,
        firecracker_executable_name=Path(spec.runtime.firecracker.source_path).name,
        ownership_key_id=managed_clinical_cleanup_key_id(KEY),
        ledger_root=str(tmp_path / 'ownership-ledger'),
        jail_namespace_root=str(jail_namespace),
        cgroup_namespace_root=str(cgroup_namespace),
        proc_root=str(proc_root),
    )
    ledger = DurableManagedClinicalOwnershipLedger(
        config=config,
        ownership_key=KEY,
        clock=lambda: case.request.launch.claimed_at,
    )
    ledger.begin_preparing(case.request, spec=spec)

    run_container = jail_namespace / RUN_ID
    jail_root = run_container / 'root'
    (jail_root / 'run').mkdir(parents=True)
    worker = case.runtime._states[RUN_ID].worker.model_copy(
        update={
            'jail_root': str(jail_root),
            'config_path': str(jail_root / 'firecracker-config.json'),
            'vsock_uds_path': str(jail_root / 'run' / 'vsock.sock'),
        }
    )
    ledger.record_prepared(worker)
    redemption = case.start.start_redemption.model_copy(
        update={'prepared_worker_sha256': firecracker_model_sha256(worker)}
    )
    start = ClinicalRuntimeStart(
        launcher_deployment_sha256=case.start.launcher_deployment_sha256,
        prepared_runtime_sha256=case.start.prepared_runtime_sha256,
        start_redemption=redemption,
        start_redemption_sha256=clinical_production_start_redemption_sha256(redemption),
    )
    ledger.record_start_bound(
        run_id=RUN_ID,
        start=start,
        capability_id=redemption.gateway_capability_id,
    )

    if not advance_running:
        startup = ManagedClinicalStartupConfig(
            reconciler_id='lane-a-startup-reaper',
            reconciler_version='test-v1',
            registry_authority_id=config.registry_authority_id,
            runtime_config_sha256='1' * 64,
            worker_spec_sha256=config.worker_spec_sha256,
            cleanup_receipt_key_id=config.ownership_key_id,
            cgroup_root=config.cgroup_namespace_root,
            jail_root=config.jail_namespace_root,
            vsock_root=config.jail_namespace_root,
            receipt_root=str(tmp_path / 'startup-receipts'),
            cleanup_grace_seconds=1,
        )
        adapter = LinuxManagedClinicalHostAdapter(
            config=startup,
            ownership=ledger,
            ownership_key=KEY,
        )
        revoked: list[str] = []
        capabilities = DurableManagedClinicalCapabilityLedger(
            ownership=ledger,
            ownership_key=KEY,
            capability_revoke=revoked.append,
        )
        artifacts = (
            *adapter.scan_process_groups(),
            *adapter.scan_vsock_endpoints(),
            *adapter.scan_cgroups(),
            *adapter.scan_jail_roots(),
        )
        return ledger, adapter, capabilities, start, artifacts, revoked

    cgroup = cgroup_namespace / RUN_ID
    cgroup.mkdir()
    (cgroup / 'cgroup.procs').write_text(f'{FIRECRACKER_PID}\n', encoding='ascii')
    pid_file = Path(worker.jail_root) / f'{config.firecracker_executable_name}.pid'
    pid_file.write_text(f'{FIRECRACKER_PID}\n', encoding='ascii')
    _write_proc_stat(
        proc_root,
        pid=FIRECRACKER_PID,
        process_group_id=PID,
        session_id=SESSION_ID,
        start_ticks=START_TICKS,
    )
    proc_executable = proc_root / str(FIRECRACKER_PID) / 'exe'
    proc_executable.hardlink_to(Path(spec.runtime.firecracker.source_path))
    pid_metadata = pid_file.stat()
    cgroup_metadata = cgroup.stat()
    cgroup_descriptor = os.open(cgroup, os.O_RDONLY)
    try:
        running = cast(
            RunningFirecrackerWorker,
            SimpleNamespace(
                prepared=worker,
                process=SimpleNamespace(pid=PID),
                jailer_start_time_ticks=START_TICKS - 10,
                jailer_process_group_id=PID,
                jailer_session_id=SESSION_ID,
                firecracker_pid=FIRECRACKER_PID,
                firecracker_parent_pid_at_observation=PID,
                firecracker_process_group_id=PID,
                firecracker_session_id=SESSION_ID,
                firecracker_start_time_ticks=START_TICKS,
                firecracker_executable_sha256=config.firecracker_executable_sha256,
                firecracker_pid_file_path=str(pid_file),
                firecracker_pid_file_device_id=pid_metadata.st_dev,
                firecracker_pid_file_inode=pid_metadata.st_ino,
                cgroup_descriptor=cgroup_descriptor,
                cgroup_device_id=cgroup_metadata.st_dev,
                cgroup_inode=cgroup_metadata.st_ino,
            ),
        )
        ledger.record_running(running)
    finally:
        os.close(cgroup_descriptor)

    short_socket_root = Path(tempfile.mkdtemp(prefix='vro-', dir='/tmp'))
    short_socket_path = short_socket_root / 'v.sock'
    vsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    vsock.bind(str(short_socket_path))
    vsock.close()
    Path(worker.vsock_uds_path).hardlink_to(short_socket_path)
    short_socket_path.unlink()
    short_socket_root.rmdir()
    startup = ManagedClinicalStartupConfig(
        reconciler_id='lane-a-startup-reaper',
        reconciler_version='test-v1',
        registry_authority_id=config.registry_authority_id,
        runtime_config_sha256='1' * 64,
        worker_spec_sha256=config.worker_spec_sha256,
        cleanup_receipt_key_id=config.ownership_key_id,
        cgroup_root=config.cgroup_namespace_root,
        jail_root=config.jail_namespace_root,
        vsock_root=config.jail_namespace_root,
        receipt_root=str(tmp_path / 'startup-receipts'),
        cleanup_grace_seconds=1,
    )
    adapter = LinuxManagedClinicalHostAdapter(
        config=startup,
        ownership=ledger,
        ownership_key=KEY,
    )
    revoked: list[str] = []
    capabilities = DurableManagedClinicalCapabilityLedger(
        ownership=ledger,
        ownership_key=KEY,
        capability_revoke=revoked.append,
    )
    artifacts = (
        *adapter.scan_process_groups(),
        *adapter.scan_vsock_endpoints(),
        *adapter.scan_cgroups(),
        *adapter.scan_jail_roots(),
    )
    return ledger, adapter, capabilities, start, artifacts, revoked


def test_exact_revocation_successor_can_remove_scanned_paths_and_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, adapter, capabilities, _start, artifacts, revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    by_kind = {item.artifact_kind: item for item in artifacts}

    shutil.rmtree(Path(ledger.config.proc_root) / str(FIRECRACKER_PID))
    cgroup_procs = Path(ledger.latest(RUN_ID).record.cgroup_path) / 'cgroup.procs'
    cgroup_procs.write_text('', encoding='ascii')
    adapter.reap_process_group(by_kind['process_group'])
    capability = capabilities.inventory()[0]
    capabilities.revoke(capability)

    adapter.remove_vsock_endpoint(by_kind['vsock_endpoint'])
    cgroup_procs.unlink()
    adapter.remove_cgroup(by_kind['cgroup'])
    adapter.remove_jail_root(by_kind['jail_root'])
    adapter.finalize_reconciled_run(RUN_ID)

    assert revoked == [capability.capability_id]
    assert ledger.latest(RUN_ID).record.state == 'cleaned'
    assert ledger.active() == ()


def test_reopened_chain_is_bound_to_config_and_requested_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _adapter, _capabilities, _start, _artifacts, _revoked = _ownership_stack(
        tmp_path, monkeypatch, advance_running=False
    )
    assert ledger.chain(RUN_ID)
    config = ledger.config
    altered_configs = (
        config.model_copy(update={'ledger_id': 'different-ledger'}),
        config.model_copy(update={'registry_authority_id': 'different-authority'}),
        config.model_copy(update={'worker_spec_sha256': 'f' * 64}),
        config.model_copy(update={'jail_namespace_root': str(tmp_path / 'different-jail')}),
        config.model_copy(update={'cgroup_namespace_root': str(tmp_path / 'different-cgroup')}),
    )
    for altered in altered_configs:
        reopened = DurableManagedClinicalOwnershipLedger(
            config=altered,
            ownership_key=KEY,
        )
        with pytest.raises(
            ManagedClinicalOwnershipError,
            match='requested run or configured namespace',
        ):
            reopened.chain(RUN_ID)

    renamed_run_id = 'f' * 32
    (ledger.root / RUN_ID).rename(ledger.root / renamed_run_id)
    with pytest.raises(
        ManagedClinicalOwnershipError,
        match='requested run or configured namespace',
    ):
        ledger.chain(renamed_run_id)


def test_create_once_record_recovers_torn_staging_without_partial_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'atomic-record'
    root.mkdir(mode=0o700)
    target = root / '0000.json'
    payload = b'{"complete":"ownership-envelope"}'
    real_write = os.write
    write_count = 0

    def torn_write(descriptor: int, content: bytes) -> int:
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            return real_write(descriptor, content[:5])
        raise OSError('injected torn ownership write')

    monkeypatch.setattr(os, 'write', torn_write)
    with pytest.raises(OSError, match='injected torn ownership write'):
        ownership_module._write_create_once(target, payload)
    assert not target.exists()
    assert (root / '.0000.json.pending').exists()

    monkeypatch.setattr(os, 'write', real_write)
    ownership_module._write_create_once(target, payload)
    assert target.read_bytes() == payload
    assert not (root / '.0000.json.pending').exists()


def test_empty_sequence_zero_staging_is_reaped_before_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    ledger_root = root / 'ownership-ledger'
    jail_root = root / 'jail'
    cgroup_root = root / 'cgroup'
    config = ManagedClinicalOwnershipConfig(
        ledger_id='lane-a-host-ownership',
        ledger_version='test-v1',
        registry_authority_id='organizer.lane-a.example',
        worker_spec_sha256='1' * 64,
        firecracker_executable_sha256='2' * 64,
        firecracker_executable_name='firecracker',
        ownership_key_id=managed_clinical_cleanup_key_id(KEY),
        ledger_root=str(ledger_root),
        jail_namespace_root=str(jail_root),
        cgroup_namespace_root=str(cgroup_root),
        proc_root=str(root / 'proc'),
    )
    ledger = DurableManagedClinicalOwnershipLedger(
        config=config,
        ownership_key=KEY,
    )
    interrupted = ledger.root / ('e' * 32)
    interrupted.mkdir(mode=0o700)
    pending = interrupted / '.0000.json.pending'
    pending.write_bytes(b'partial')
    pending.chmod(0o600)

    assert ledger.active() == ()
    assert not interrupted.exists()


def test_reap_requires_every_owned_cgroup_member_to_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, adapter, _capabilities, _start, artifacts, _revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    process = next(item for item in artifacts if item.artifact_kind == 'process_group')
    proc_root = Path(ledger.config.proc_root)
    shutil.rmtree(proc_root / str(FIRECRACKER_PID))
    child_pid = FIRECRACKER_PID + 1
    _write_proc_stat(
        proc_root,
        pid=child_pid,
        process_group_id=PID,
        session_id=SESSION_ID,
        start_ticks=START_TICKS + 1,
    )
    (Path(ledger.latest(RUN_ID).record.cgroup_path) / 'cgroup.procs').write_text(
        f'{child_pid}\n',
        encoding='ascii',
    )

    rescanned = adapter.scan_process_groups()
    assert len(rescanned) == 1
    assert rescanned[0].artifact_id == f'pgid:{PID}'
    with pytest.raises(ManagedClinicalOwnershipError, match='remains live'):
        adapter.reap_process_group(process)


def test_complete_cgroup_scan_rejects_unknown_namespace_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, adapter, _capabilities, _start, _artifacts, _revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    (Path(ledger.config.cgroup_namespace_root) / 'caller-file').write_text(
        'not kernel metadata',
        encoding='utf-8',
    )

    with pytest.raises(ManagedClinicalOwnershipError, match='unexpected entry'):
        adapter.scan_cgroups()


def test_process_scan_rejects_live_leader_missing_from_cgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, adapter, _capabilities, _start, _artifacts, _revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    (Path(ledger.latest(RUN_ID).record.cgroup_path) / 'cgroup.procs').write_text(
        '',
        encoding='ascii',
    )

    with pytest.raises(ManagedClinicalOwnershipError, match='missing from its cgroup'):
        adapter.scan_process_groups()


def test_tampered_append_only_record_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _adapter, _capabilities, _start, _artifacts, _revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    path = Path(ledger.config.ledger_root) / RUN_ID / '0000.json'
    content = path.read_bytes()
    marker = b'"ownership_hmac_sha256":"'
    offset = content.index(marker) + len(marker)
    original = content[offset : offset + 64]
    replacement = (b'0' if original[:1] != b'0' else b'1') + original[1:]
    path.write_bytes(content.replace(original, replacement, 1))

    with pytest.raises(ManagedClinicalOwnershipError, match='authentication failed'):
        ledger.latest(RUN_ID)


def test_recursive_jail_removal_refuses_a_mount_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'parent'
    target = root / 'owned'
    target.mkdir(parents=True)
    metadata = target.stat()
    mount_ids = iter((10, 11))
    monkeypatch.setattr(ownership_module, '_linux_fd_mount_id', lambda _fd: next(mount_ids))

    with pytest.raises(ManagedClinicalOwnershipError, match='mount boundary'):
        ownership_module._remove_tree_exact_fd(
            target,
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
        )

    assert target.is_dir()


def test_termination_uses_exact_cgroup_kill_and_never_a_reusable_pgid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, adapter, _capabilities, _start, artifacts, _revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    process = next(item for item in artifacts if item.artifact_kind == 'process_group')
    cgroup = Path(ledger.latest(RUN_ID).record.cgroup_path)
    cgroup_kill = cgroup / 'cgroup.kill'
    cgroup_kill.write_bytes(b'')
    times = iter((0.0, 2.0, 3.0, 3.5))
    adapter._monotonic = lambda: next(times)  # noqa: SLF001

    def emulate_kernel_cgroup_kill(_seconds: float) -> None:
        assert cgroup_kill.read_bytes() == b'1'
        shutil.rmtree(Path(ledger.config.proc_root) / str(FIRECRACKER_PID))
        (cgroup / 'cgroup.procs').write_bytes(b'')

    adapter._sleep = emulate_kernel_cgroup_kill  # noqa: SLF001
    monkeypatch.setattr(
        os,
        'killpg',
        lambda *_args: pytest.fail('managed cleanup must never signal a bare process group'),
    )

    adapter.terminate_process_group(process, grace_seconds=1)

    assert cgroup_kill.read_bytes() == b'1'


def test_process_cleanup_tolerates_exact_zombie_after_pidfd_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, adapter, _capabilities, _start, artifacts, _revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    process = next(item for item in artifacts if item.artifact_kind == 'process_group')
    proc_root = Path(ledger.config.proc_root) / str(FIRECRACKER_PID)
    cgroup_procs = Path(ledger.latest(RUN_ID).record.cgroup_path) / 'cgroup.procs'

    def become_zombie(*_args: object, **_kwargs: object) -> bool:
        (proc_root / 'exe').unlink()
        stat_path = proc_root / 'stat'
        stat_path.write_text(
            stat_path.read_text(encoding='ascii').replace(') S ', ') Z ', 1),
            encoding='ascii',
        )
        return True

    def reap_zombie(_seconds: float) -> None:
        shutil.rmtree(proc_root)
        cgroup_procs.write_text('', encoding='ascii')

    monkeypatch.setattr(adapter, '_signal_exact_firecracker_child', become_zombie)
    adapter._sleep = reap_zombie  # noqa: SLF001

    adapter.terminate_process_group(process, grace_seconds=1)

    assert cgroup_procs.read_text(encoding='ascii') == ''


def test_process_cleanup_rejects_replaced_exact_cgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, adapter, _capabilities, _start, artifacts, _revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    process = next(item for item in artifacts if item.artifact_kind == 'process_group')
    cgroup = Path(ledger.latest(RUN_ID).record.cgroup_path)
    original = cgroup.with_name(f'{cgroup.name}.replaced')
    cgroup.rename(original)
    cgroup.mkdir()
    (cgroup / 'cgroup.procs').write_text(f'{FIRECRACKER_PID}\n', encoding='ascii')
    (cgroup / 'cgroup.kill').write_bytes(b'')
    monkeypatch.setattr(
        os,
        'killpg',
        lambda *_args: pytest.fail('managed cleanup must never signal a bare process group'),
    )

    with pytest.raises(ManagedClinicalOwnershipError, match='changed identity'):
        adapter.terminate_process_group(process, grace_seconds=1)

    assert (cgroup / 'cgroup.kill').read_bytes() == b''


def test_process_cleanup_rejects_replaced_pid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, adapter, _capabilities, _start, artifacts, _revoked = _ownership_stack(
        tmp_path,
        monkeypatch,
    )
    process = next(item for item in artifacts if item.artifact_kind == 'process_group')
    pid_file = Path(ledger.latest(RUN_ID).record.firecracker_pid_file_path or '')
    pid_file.unlink()
    pid_file.write_text(f'{FIRECRACKER_PID}\n', encoding='ascii')

    with pytest.raises(ManagedClinicalOwnershipError, match='pid file changed identity'):
        adapter.reap_process_group(process)
