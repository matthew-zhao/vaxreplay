from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import vaxreplay.agentic.firecracker_qualification_guest as qualification_guest_module
from tests.test_agentic_firecracker import _make_spec
from vaxreplay.agentic.firecracker import firecracker_model_sha256
from vaxreplay.agentic.firecracker_qualification import FirecrackerQualificationClaim, FirecrackerQualificationDrillId
from vaxreplay.agentic.firecracker_qualification_driver import (
    FirecrackerQualificationDriverError,
    FirecrackerQualificationDriverRequest,
    LinuxKvmFirecrackerQualificationDriver,
    LinuxQualificationEvidenceReader,
)
from vaxreplay.agentic.firecracker_qualification_guest import (
    FirecrackerQualificationGuestConfig,
    FirecrackerQualificationGuestRequest,
    FirecrackerQualificationGuestResult,
    execute_firecracker_qualification_guest_request,
)
from vaxreplay.agentic.firecracker_qualification_guest_disk_build import (
    _INIT_BYTES,
    _install_rootfs_init,
    _resolve_guest_tree_path,
)
from vaxreplay.agentic.firecracker_qualification_probe import (
    FirecrackerQualificationChallenge,
    FirecrackerQualificationGuestCommand,
    FirecrackerQualificationGuestDiskBuildReceipt,
    FirecrackerQualificationProbeManifest,
    FirecrackerQualificationWorkerBinding,
    derive_firecracker_qualification_worker_spec,
    ed25519_public_key_bytes,
    firecracker_qualification_guest_key_id,
    firecracker_qualification_probe_manifest_sha256,
    firecracker_qualification_static_config_sha256,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_KEY = Ed25519PrivateKey.from_private_bytes(bytes.fromhex('31' * 32))


def _release(tmp_path: Path):
    worker = tmp_path / 'worker'
    worker.mkdir()
    spec = _make_spec(worker)
    public_key = ed25519_public_key_bytes(_KEY)
    receipt = FirecrackerQualificationGuestDiskBuildReceipt(
        source_date_epoch=1_700_000_000,
        base_rootfs_tree_sha256='1' * 64,
        package_tree_sha256='2' * 64,
        normalized_rootfs_tree_sha256='3' * 64,
        normalized_harness_tree_sha256='4' * 64,
        build_recipe_sha256='5' * 64,
        mke2fs_sha256='6' * 64,
        mke2fs_version='mke2fs test',
        e2fsck_sha256='7' * 64,
        e2fsck_version='e2fsck test',
        debugfs_sha256='8' * 64,
        debugfs_version='debugfs test',
        build_argv_and_env_sha256='9' * 64,
        init_sha256='a' * 64,
        guest_probe_executable_sha256='b' * 64,
        guest_config_sha256='c' * 64,
        rootfs_sha256='d' * 64,
        rootfs_byte_count=4096,
        harness_sha256='e' * 64,
        harness_byte_count=8192,
    )
    manifest = FirecrackerQualificationProbeManifest(
        manifest_id='qualification-test',
        task_worker_spec_sha256=firecracker_model_sha256(spec),
        task_rootfs_sha256=spec.images.rootfs.sha256,
        task_harness_sha256=spec.images.harness.sha256,
        qualification_kernel_sha256=spec.images.kernel.sha256,
        qualification_rootfs_path='/opt/vaxreplay/qualification/rootfs.ext4',
        qualification_rootfs_sha256=receipt.rootfs_sha256,
        qualification_rootfs_byte_count=receipt.rootfs_byte_count,
        qualification_harness_path='/opt/vaxreplay/qualification/harness.ext4',
        qualification_harness_sha256=receipt.harness_sha256,
        qualification_harness_byte_count=receipt.harness_byte_count,
        qualification_disk_build_receipt=receipt,
        qualification_disk_build_receipt_sha256=firecracker_model_sha256(receipt),
        guest_probe_executable_sha256=receipt.guest_probe_executable_sha256,
        guest_probe_public_key_hex=public_key.hex(),
        guest_probe_key_id=firecracker_qualification_guest_key_id(public_key),
    )
    return spec, manifest


def _challenge_and_binding(spec, manifest, drill=FirecrackerQualificationDrillId.GUEST_ISOLATION):
    challenge = FirecrackerQualificationChallenge(
        collection_id='1' * 32,
        challenge_id='2' * 32,
        nonce_hex='3' * 64,
        drill_id=drill,
        run_ids=('4' * 32,),
        worker_spec_sha256=firecracker_model_sha256(spec),
        probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
        issued_at=_NOW,
    )
    qualification_spec = derive_firecracker_qualification_worker_spec(manifest, task_worker_spec=spec)
    executable_name = Path(qualification_spec.runtime.firecracker.source_path).name
    jail_root = Path(qualification_spec.chroot_base_dir) / executable_name / challenge.run_ids[0] / 'root'
    binding = FirecrackerQualificationWorkerBinding(
        run_id=challenge.run_ids[0],
        worker_spec_sha256=challenge.worker_spec_sha256,
        qualification_worker_spec_sha256=firecracker_model_sha256(qualification_spec),
        qualification_static_config_sha256=firecracker_qualification_static_config_sha256(qualification_spec),
        prepared_worker_sha256='5' * 64,
        probe_manifest_sha256=challenge.probe_manifest_sha256,
        firecracker_pid=100,
        firecracker_parent_pid_at_observation=99,
        firecracker_start_time_ticks=10_000,
        firecracker_session_id=50,
        firecracker_executable_sha256=spec.runtime.firecracker.sha256,
        firecracker_pid_file_path=str(jail_root / f'{executable_name}.pid'),
        firecracker_pid_file_device_id=10,
        firecracker_pid_file_inode=11,
        jailer_pid=99,
        jailer_start_time_ticks=9_999,
        jailer_process_group_id=99,
        jailer_session_id=50,
        process_group_id=99,
        worker_uid=spec.worker_uid,
        worker_gid=spec.worker_gid,
        cgroup_path=f'/sys/fs/cgroup/{spec.cgroup_parent}/{challenge.run_ids[0]}',
        cgroup_inode=1000,
        cgroup_member_pids=(100,),
        jail_root=str(jail_root),
        vsock_uds_path=str(jail_root / 'run' / 'vsock.sock'),
        guest_cid=spec.guest_cid,
        peer_pid=100,
        peer_uid=spec.worker_uid,
        peer_gid=spec.worker_gid,
        process_tree_verified=True,
        pid_cgroup_binding_verified=True,
    )
    return challenge, binding


def test_qualification_spec_replaces_only_separate_probe_disks(tmp_path: Path) -> None:
    spec, manifest = _release(tmp_path)
    qualification = derive_firecracker_qualification_worker_spec(manifest, task_worker_spec=spec)
    assert qualification.images.rootfs.sha256 == manifest.qualification_rootfs_sha256
    assert qualification.images.harness.sha256 == manifest.qualification_harness_sha256
    assert qualification.images.kernel == spec.images.kernel
    assert qualification.images.scratch_template == spec.images.scratch_template
    assert qualification.limits == spec.limits
    assert qualification != spec


def test_guest_signs_only_locally_observed_isolation_claims(tmp_path: Path) -> None:
    spec, manifest = _release(tmp_path)
    challenge, binding = _challenge_and_binding(spec, manifest)
    request = FirecrackerQualificationGuestRequest(
        challenge=challenge,
        worker_binding=binding,
        command=FirecrackerQualificationGuestCommand.ISOLATION_PROBES,
        worker_spec_sha256=challenge.worker_spec_sha256,
        probe_manifest_sha256=challenge.probe_manifest_sha256,
        guest_probe_executable_sha256=manifest.guest_probe_executable_sha256,
    )
    config = FirecrackerQualificationGuestConfig(
        rpc_port=spec.guest_rpc_port,
        guest_probe_private_key_hex=_KEY.private_bytes_raw().hex(),
        guest_probe_key_id=manifest.guest_probe_key_id,
        guest_probe_executable_sha256=manifest.guest_probe_executable_sha256,
    )

    class LocalProbe:
        def __init__(self, _request) -> None:
            pass

        def isolation(self, *, harness_mount: Path, scratch_mount: Path):
            del harness_mount, scratch_mount
            return FirecrackerQualificationGuestResult(
                command=request.command,
                run_id=binding.run_id,
                nonce_sha256=hashlib.sha256(bytes.fromhex(challenge.nonce_hex)).hexdigest(),
                rootfs_write_denied=True,
                harness_write_denied=True,
                scratch_write_succeeded=True,
                scratch_fresh=True,
                network_unreachable=False,
                mmds_unreachable=True,
            )

    authenticated, hold = execute_firecracker_qualification_guest_request(
        request,
        config=config,
        clock=lambda: _NOW,
        local_probe_factory=LocalProbe,
    )
    assert hold is False
    assert set(authenticated.response.verified_guest_claims) == {
        FirecrackerQualificationClaim.ROOTFS_WRITE_DENIED,
        FirecrackerQualificationClaim.HARNESS_WRITE_DENIED,
        FirecrackerQualificationClaim.SCRATCH_WRITE_SUCCEEDED,
        FirecrackerQualificationClaim.SCRATCH_FRESH,
        FirecrackerQualificationClaim.MMDS_UNREACHABLE,
    }


def test_cgroup_stress_starts_pressure_and_holds_for_host_snapshots(tmp_path: Path) -> None:
    spec, manifest = _release(tmp_path)
    challenge, binding = _challenge_and_binding(spec, manifest, FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT)
    request = FirecrackerQualificationGuestRequest(
        challenge=challenge,
        worker_binding=binding,
        command=FirecrackerQualificationGuestCommand.CGROUP_STRESS,
        worker_spec_sha256=challenge.worker_spec_sha256,
        probe_manifest_sha256=challenge.probe_manifest_sha256,
        guest_probe_executable_sha256=manifest.guest_probe_executable_sha256,
    )
    config = FirecrackerQualificationGuestConfig(
        rpc_port=spec.guest_rpc_port,
        guest_probe_private_key_hex=_KEY.private_bytes_raw().hex(),
        guest_probe_key_id=manifest.guest_probe_key_id,
        guest_probe_executable_sha256=manifest.guest_probe_executable_sha256,
    )
    calls: list[str] = []
    response, hold = execute_firecracker_qualification_guest_request(
        request,
        config=config,
        clock=lambda: _NOW,
        pressure_starter=lambda: calls.append('started'),
    )
    assert calls == ['started']
    assert response.response.verified_guest_claims == ()
    assert hold is True


def test_injected_cgroupfs_reader_parses_real_counter_shape(tmp_path: Path) -> None:
    spec, _ = _release(tmp_path)
    cgroup_root = tmp_path / 'cgroup'
    run_id = '4' * 32
    cgroup = cgroup_root.joinpath(*spec.cgroup_parent.split('/'), run_id)
    cgroup.mkdir(parents=True)
    values = {
        'cpu.max': f'{spec.limits.cpu_quota_us} {spec.limits.cpu_period_us}\n',
        'cpu.stat': 'nr_throttled 3\nthrottled_usec 400\n',
        'memory.max': f'{spec.limits.memory_mib * 1024 * 1024}\n',
        'memory.swap.max': '0\n',
        'memory.events': 'oom 2\noom_kill 1\n',
        'pids.max': f'{spec.limits.pids}\n',
        'pids.events': 'max 5\n',
        'cgroup.procs': '100\n101\n',
    }
    for name, content in values.items():
        (cgroup / name).write_text(content, encoding='ascii')
    snapshot = LinuxQualificationEvidenceReader(
        proc_root=tmp_path / 'proc',
        cgroup_root=cgroup_root,
        production=False,
        getpgid=lambda pid: pid,
    ).cgroup_snapshot(spec=spec, run_id=run_id, clock=lambda: _NOW)
    assert snapshot.cpu_nr_throttled == 3
    assert snapshot.memory_oom_kill == 1
    assert snapshot.pids_max_events == 5
    assert snapshot.member_pids == (100, 101)


def test_guest_rootfs_absolute_symlink_resolves_inside_guest_tree(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    (root / 'bin').mkdir(parents=True)
    busybox = root / 'bin' / 'busybox'
    busybox.write_bytes(b'busybox')
    busybox.chmod(0o755)
    (root / 'bin' / 'sh').symlink_to('/bin/busybox')
    assert _resolve_guest_tree_path(root, PurePosixPath('/bin/sh')) == busybox


def test_qualification_rootfs_installs_empty_read_only_mountpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'root'
    existing = root / 'opt' / 'vaxreplay'
    existing.mkdir(parents=True)
    (existing / 'must-not-survive').write_text('future information', encoding='utf-8')
    monkeypatch.setattr(os, 'chown', lambda *_args: None)

    _install_rootfs_init(root, source_date_epoch=1_700_000_000)

    for mountpoint in (root / 'opt' / 'vaxreplay', root / 'workspace'):
        assert mountpoint.is_dir()
        assert tuple(mountpoint.iterdir()) == ()
        assert (mountpoint.stat().st_mode & 0o777) == 0o755
        assert int(mountpoint.stat().st_mtime) == 1_700_000_000
    assert (root / 'sbin' / 'init').read_bytes() == _INIT_BYTES


def test_packaged_qualification_probe_is_an_executable_script() -> None:
    completed = subprocess.run(
        (sys.executable, str(Path(qualification_guest_module.__file__)), '--help'),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0
    assert b'run exactly one signed Firecracker qualification guest probe' in completed.stdout


def test_driver_request_is_bound_but_live_constructor_requires_linux_kvm(tmp_path: Path) -> None:
    spec, manifest = _release(tmp_path)
    challenge, _ = _challenge_and_binding(spec, manifest)
    request = FirecrackerQualificationDriverRequest(
        challenge=challenge,
        worker_spec=spec,
        probe_manifest=manifest,
    )
    if platform.system() != 'Linux' or os.geteuid() != 0 or not Path('/dev/kvm').exists():
        with pytest.raises(FirecrackerQualificationDriverError):
            LinuxKvmFirecrackerQualificationDriver(request)
