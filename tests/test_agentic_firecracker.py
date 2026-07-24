from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from pydantic import ValidationError

import vaxreplay.agentic.firecracker as firecracker_module
from vaxreplay.agentic.firecracker import (
    FirecrackerArtifactIdentity,
    FirecrackerArtifactRole,
    FirecrackerAttestationError,
    FirecrackerGuestImages,
    FirecrackerHostPreflightReceipt,
    FirecrackerPreflightError,
    FirecrackerPreparationError,
    FirecrackerResourceLimits,
    FirecrackerRuntimeIdentity,
    FirecrackerStaticConfig,
    FirecrackerSupervisor,
    FirecrackerVsockError,
    FirecrackerWorkerSpec,
    RunningFirecrackerWorker,
    authenticated_firecracker_worker_attestation_sha256,
    build_firecracker_static_config,
    build_jailer_argv,
    capture_firecracker_prebound_guest_listener,
    connect_firecracker_vsock,
    finalize_firecracker_worker_attestation,
    firecracker_attestation_key_id,
    firecracker_guest_bootstrap_profile,
    firecracker_guest_bootstrap_profile_sha256,
    firecracker_guest_initiated_uds_path,
    firecracker_model_sha256,
    firecracker_static_config_bytes,
    preflight_firecracker_host,
    verify_firecracker_worker_attestation,
)

_RUN_ID = '1' * 32
_OTHER_RUN_ID = '2' * 32
_MIB = 1024 * 1024


def _identity(path: Path, role: FirecrackerArtifactRole) -> FirecrackerArtifactIdentity:
    content = path.read_bytes()
    return FirecrackerArtifactIdentity(
        artifact_id=f'{role.value}-test',
        role=role,
        source_path=str(path),
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
    )


def _make_spec(
    root: Path,
    *,
    scratch_source: Path | None = None,
    wall_seconds: int = 30,
) -> FirecrackerWorkerSpec:
    artifact_root = root / 'artifacts'
    artifact_root.mkdir()
    content_by_name = {
        'firecracker': b'firecracker-runtime',
        'jailer': b'jailer-runtime',
        'kernel': b'kernel-image',
        'rootfs': b'root-filesystem',
        'harness': b'harness-filesystem',
    }
    paths: dict[str, Path] = {}
    for name, content in content_by_name.items():
        path = artifact_root / name
        path.write_bytes(content)
        if name in {'firecracker', 'jailer'}:
            path.chmod(0o500)
        paths[name] = path
    if scratch_source is None:
        scratch_source = artifact_root / 'scratch-template'
        scratch_source.write_bytes(b'\x00' * _MIB)
    chroot_base = root / 'jails'
    chroot_base.mkdir(mode=0o700)
    worker_uid = max(os.geteuid(), 1)
    worker_gid = max(os.getegid(), 1)
    return FirecrackerWorkerSpec(
        worker_id='firecracker-worker-test',
        runtime=FirecrackerRuntimeIdentity(
            release='v1.12.0-test-pin',
            architecture='aarch64',
            firecracker=_identity(paths['firecracker'], FirecrackerArtifactRole.FIRECRACKER),
            jailer=_identity(paths['jailer'], FirecrackerArtifactRole.JAILER),
        ),
        images=FirecrackerGuestImages(
            kernel=_identity(paths['kernel'], FirecrackerArtifactRole.KERNEL),
            rootfs=_identity(paths['rootfs'], FirecrackerArtifactRole.ROOTFS),
            harness=_identity(paths['harness'], FirecrackerArtifactRole.HARNESS),
            scratch_template=_identity(scratch_source, FirecrackerArtifactRole.SCRATCH_TEMPLATE),
        ),
        limits=FirecrackerResourceLimits(
            wall_seconds=wall_seconds,
            vcpu_count=1,
            cpu_period_us=100_000,
            cpu_quota_us=100_000,
            memory_mib=128,
            pids=32,
            open_files=64,
            scratch_bytes=_MIB,
        ),
        chroot_base_dir=str(chroot_base),
        cgroup_parent='vaxreplay/official',
        worker_uid=worker_uid,
        worker_gid=worker_gid,
        guest_cid=42,
        guest_rpc_port=7000,
    )


def _preflight(spec: FirecrackerWorkerSpec) -> FirecrackerHostPreflightReceipt:
    return FirecrackerHostPreflightReceipt(
        worker_spec_sha256=firecracker_model_sha256(spec),
        collected_at=datetime.now(UTC),
        host_architecture='aarch64',
        host_kernel_release='test-kernel',
        cgroup_controllers=('cpu', 'memory', 'pids'),
    )


class FirecrackerSchemaAndConfigTest(unittest.TestCase):
    def test_guest_bootstrap_profile_breaks_only_the_two_disk_hash_cycles(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            changed_images = spec.images.model_copy(
                update={
                    'rootfs': spec.images.rootfs.model_copy(update={'sha256': 'a' * 64}),
                    'harness': spec.images.harness.model_copy(update={'sha256': 'b' * 64}),
                }
            )
            changed_disks = spec.model_copy(update={'images': changed_images})
            changed_cid = spec.model_copy(update={'guest_cid': spec.guest_cid + 1})

        self.assertNotEqual(firecracker_model_sha256(spec), firecracker_model_sha256(changed_disks))
        self.assertEqual(
            firecracker_guest_bootstrap_profile_sha256(spec),
            firecracker_guest_bootstrap_profile_sha256(changed_disks),
        )
        self.assertNotEqual(
            firecracker_guest_bootstrap_profile_sha256(spec),
            firecracker_guest_bootstrap_profile_sha256(changed_cid),
        )
        profile = firecracker_guest_bootstrap_profile(spec)
        self.assertEqual(profile.projected_worker_spec.images.rootfs.sha256, '0' * 64)
        self.assertEqual(profile.projected_worker_spec.images.harness.sha256, '0' * 64)
        self.assertEqual(profile.projected_worker_spec.images.kernel, spec.images.kernel)
        self.assertTrue(profile.full_worker_spec_sha256_in_signed_hello_required)

    def test_config_has_only_three_pinned_drives_and_vsock(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            config = build_firecracker_static_config(spec)
            wire = json.loads(firecracker_static_config_bytes(spec))

        self.assertEqual(set(wire), {'boot-source', 'drives', 'machine-config', 'vsock'})
        self.assertNotIn('network-interfaces', wire)
        self.assertNotIn('mmds-config', wire)
        self.assertEqual(
            [(drive.drive_id, drive.is_root_device, drive.is_read_only) for drive in config.drives],
            [('rootfs', True, True), ('harness', False, True), ('scratch', False, False)],
        )
        self.assertEqual(config.vsock.uds_path, '/run/vsock.sock')
        self.assertFalse(config.machine_config.smt)
        self.assertFalse(config.machine_config.track_dirty_pages)

    def test_complete_config_rejects_an_added_network_device(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            wire = json.loads(firecracker_static_config_bytes(spec))
        wire['network-interfaces'] = []
        with self.assertRaises(ValidationError):
            FirecrackerStaticConfig.model_validate(wire)

    def test_spec_rejects_scratch_size_not_equal_to_pinned_limit(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            values = spec.model_dump()
            values['limits']['scratch_bytes'] = _MIB + 1
            with self.assertRaisesRegex(ValidationError, 'scratch template byte count'):
                FirecrackerWorkerSpec.model_validate(values)

    def test_spec_rejects_unsafe_or_noncanonical_host_paths(self) -> None:
        with self.assertRaisesRegex(ValidationError, 'absolute normalized'):
            FirecrackerArtifactIdentity(
                artifact_id='unsafe',
                role=FirecrackerArtifactRole.KERNEL,
                source_path='../kernel',
                sha256='a' * 64,
                byte_count=1,
            )

    def test_spec_rejects_cgroup_parent_traversal(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            values = spec.model_dump()
            values['cgroup_parent'] = 'vaxreplay/../escape'
            with self.assertRaisesRegex(ValidationError, 'dot path'):
                FirecrackerWorkerSpec.model_validate(values)

    def test_jailer_argv_enforces_cgroup_v2_limits_pid_namespace_and_no_api(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            argv = build_jailer_argv(spec=spec, run_id=_RUN_ID)

        expected_pairs = (
            ('--cgroup-version', '2'),
            ('--parent-cgroup', 'vaxreplay/official'),
            ('--uid', str(spec.worker_uid)),
            ('--gid', str(spec.worker_gid)),
            ('--chroot-base-dir', spec.chroot_base_dir),
            ('--config-file', '/firecracker-config.json'),
        )
        for flag, value in expected_pairs:
            self.assertEqual(argv[argv.index(flag) + 1], value)
        cgroups = tuple(argv[index + 1] for index, value in enumerate(argv) if value == '--cgroup')
        self.assertEqual(
            cgroups,
            ('cpu.max=100000 100000', f'memory.max={128 * _MIB}', 'memory.swap.max=0', 'pids.max=32'),
        )
        resources = tuple(argv[index + 1] for index, value in enumerate(argv) if value == '--resource-limit')
        self.assertEqual(resources, ('no-file=64', f'fsize={_MIB}'))
        self.assertIn('--new-pid-ns', argv)
        self.assertIn('--no-api', argv)
        self.assertNotIn('--daemonize', argv)
        self.assertNotIn('--api-sock', argv)
        self.assertEqual(argv[0], spec.runtime.jailer.source_path)

    def test_invalid_run_id_never_reaches_jailer(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
        with self.assertRaisesRegex(ValueError, '32 lowercase hexadecimal'):
            build_jailer_argv(spec=spec, run_id='../../escape')


class FirecrackerPreflightTest(unittest.TestCase):
    def test_non_linux_host_fails_before_artifact_access(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            with patch('vaxreplay.agentic.firecracker.platform.system', return_value='Darwin'):
                with self.assertRaisesRegex(FirecrackerPreflightError, 'Linux'):
                    preflight_firecracker_host(spec)

    def test_runtime_architecture_must_match_host(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            with (
                patch('vaxreplay.agentic.firecracker.platform.system', return_value='Linux'),
                patch('vaxreplay.agentic.firecracker.platform.machine', return_value='x86_64'),
            ):
                with self.assertRaisesRegex(FirecrackerPreflightError, 'architecture'):
                    preflight_firecracker_host(spec)

    def test_root_is_required_before_kvm_or_runtime_checks(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            with (
                patch('vaxreplay.agentic.firecracker.platform.system', return_value='Linux'),
                patch('vaxreplay.agentic.firecracker.platform.machine', return_value='aarch64'),
                patch('vaxreplay.agentic.firecracker.os.geteuid', return_value=501),
            ):
                with self.assertRaisesRegex(FirecrackerPreflightError, 'UID 0'):
                    preflight_firecracker_host(spec)


class FirecrackerPreparationTest(unittest.TestCase):
    def test_preparation_copies_exact_images_with_pinned_modes_and_fresh_scratch(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                first = supervisor.prepare(run_id=_RUN_ID)
                second = supervisor.prepare(run_id=_OTHER_RUN_ID)

            first_root = Path(first.jail_root)
            second_root = Path(second.jail_root)
            self.assertEqual(
                {path.name for path in first_root.iterdir()},
                {'firecracker-config.json', 'harness.ext4', 'kernel.bin', 'rootfs.ext4', 'run', 'scratch.ext4'},
            )
            self.assertEqual(stat.S_IMODE((first_root / 'rootfs.ext4').stat().st_mode), 0o400)
            self.assertEqual(stat.S_IMODE((first_root / 'harness.ext4').stat().st_mode), 0o400)
            self.assertEqual(stat.S_IMODE((first_root / 'scratch.ext4').stat().st_mode), 0o600)
            self.assertEqual((first_root / 'scratch.ext4').read_bytes(), b'\x00' * _MIB)
            self.assertNotEqual(
                (first_root / 'scratch.ext4').stat().st_ino, (second_root / 'scratch.ext4').stat().st_ino
            )
            self.assertEqual(first.initial_scratch_sha256, second.initial_scratch_sha256)
            self.assertEqual(
                json.loads(Path(first.config_path).read_bytes()), json.loads(firecracker_static_config_bytes(spec))
            )

            first_cleanup = supervisor.discard_prepared(first)
            second_cleanup = supervisor.discard_prepared(second)
            self.assertTrue(first_cleanup.jail_root_removed)
            self.assertTrue(second_cleanup.jail_root_removed)
            self.assertFalse(first_root.parent.exists())
            self.assertFalse(second_root.parent.exists())

    def test_run_id_cannot_be_reused(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                prepared = supervisor.prepare(run_id=_RUN_ID)
                with self.assertRaisesRegex(FirecrackerPreparationError, 'never reused'):
                    supervisor.prepare(run_id=_RUN_ID)
            supervisor.discard_prepared(prepared)

    def test_preexisting_run_symlink_is_rejected_and_target_is_untouched(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _make_spec(root)
            executable_dir = Path(spec.chroot_base_dir) / Path(spec.runtime.firecracker.source_path).name
            executable_dir.mkdir()
            outside = root / 'outside'
            outside.mkdir()
            (executable_dir / _RUN_ID).symlink_to(outside, target_is_directory=True)
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                with self.assertRaisesRegex(FirecrackerPreparationError, 'never reused'):
                    supervisor.prepare(run_id=_RUN_ID)
            self.assertTrue(outside.is_dir())
            self.assertTrue((executable_dir / _RUN_ID).is_symlink())

    def test_source_symlink_is_rejected_even_after_mocked_preflight(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'scratch-target'
            target.write_bytes(b'\x00' * _MIB)
            link = root / 'scratch-link'
            link.symlink_to(target)
            spec = _make_spec(root, scratch_source=link)
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                with self.assertRaisesRegex(FirecrackerPreparationError, 'cannot open pinned source'):
                    supervisor.prepare(run_id=_RUN_ID)
            expected_run = Path(spec.chroot_base_dir) / 'firecracker' / _RUN_ID
            self.assertFalse(expected_run.exists())

    def test_source_digest_is_rechecked_during_copy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _make_spec(root)
            Path(spec.images.kernel.source_path).write_bytes(b'mutated-after-preflight')
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                with self.assertRaisesRegex(FirecrackerPreparationError, 'digest or size mismatch'):
                    supervisor.prepare(run_id=_RUN_ID)
            expected_run = Path(spec.chroot_base_dir) / 'firecracker' / _RUN_ID
            self.assertFalse(expected_run.exists())

    def test_mutated_config_is_rejected_before_launch(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                prepared = supervisor.prepare(run_id=_RUN_ID)
            config = Path(prepared.config_path)
            config.chmod(0o600)
            config.write_bytes(b'{}')
            with self.assertRaisesRegex(FirecrackerPreparationError, 'mode|digest'):
                supervisor.verify_prepared(prepared)

    def test_forged_prepared_artifact_commitment_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                prepared = supervisor.prepare(run_id=_RUN_ID)
            forged = prepared.model_copy(update={'kernel_sha256': 'f' * 64})
            with self.assertRaisesRegex(FirecrackerPreparationError, 'artifact commitment'):
                supervisor.verify_prepared(forged)
            supervisor.discard_prepared(prepared)

    def test_only_inode_bound_prelaunch_listener_is_accepted(self) -> None:
        with TemporaryDirectory(dir='/tmp') as temporary:
            spec = _make_spec(Path(temporary))
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                prepared = supervisor.prepare(run_id=_RUN_ID)
            socket_path = Path(
                firecracker_guest_initiated_uds_path(
                    uds_path=prepared.vsock_uds_path,
                    port=spec.guest_rpc_port,
                )
            )
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            replacement: socket.socket | None = None
            try:
                listener.bind(str(socket_path))
                os.chown(socket_path, spec.worker_uid, spec.worker_gid)
                socket_path.chmod(0o600)
                identity = capture_firecracker_prebound_guest_listener(prepared, spec=spec)
                supervisor.verify_prepared(prepared, prebound_guest_listener=identity)
                with self.assertRaisesRegex(FirecrackerPreparationError, 'empty before listener'):
                    supervisor.verify_prepared(prepared)

                socket_path.unlink()
                replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                replacement.bind(str(socket_path))
                os.chown(socket_path, spec.worker_uid, spec.worker_gid)
                socket_path.chmod(0o600)
                with self.assertRaisesRegex(FirecrackerPreparationError, 'identity changed'):
                    supervisor.verify_prepared(prepared, prebound_guest_listener=identity)
            finally:
                listener.close()
                if replacement is not None:
                    replacement.close()
                socket_path.unlink(missing_ok=True)
            supervisor.discard_prepared(prepared)

    def test_unexpected_file_prevents_unlaunched_cleanup(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                prepared = supervisor.prepare(run_id=_RUN_ID)
            unexpected = Path(prepared.jail_root) / 'unbound-secret'
            unexpected.write_bytes(b'do not delete without review')
            with self.assertRaisesRegex(FirecrackerPreparationError, 'unexpected'):
                supervisor.discard_prepared(prepared)
            self.assertTrue(unexpected.exists())
            unexpected.unlink()
            supervisor.discard_prepared(prepared)


class FirecrackerLifecycleAttestationTest(unittest.TestCase):
    def test_watchdog_retains_actual_trigger_before_async_cgroup_kill(self) -> None:
        stop = threading.Event()
        timeout_triggered = threading.Event()
        failure = threading.Event()
        timing = firecracker_module._WatchdogTiming()
        deadline_monotonic_ns = 1_000
        with (
            patch('vaxreplay.agentic.firecracker._pidfd_process_alive', return_value=True),
            patch(
                'vaxreplay.agentic.firecracker._read_proc_process_identity',
                return_value=SimpleNamespace(state='R', start_time_ticks=10, process_group_id=20),
            ),
            patch(
                'vaxreplay.agentic.firecracker.time.monotonic_ns',
                side_effect=(deadline_monotonic_ns + 7, deadline_monotonic_ns + 11),
            ),
            patch('vaxreplay.agentic.firecracker._kill_pinned_cgroup') as kill_cgroup,
        ):
            firecracker_module._watchdog_process_group(
                30,
                10,
                20,
                40,
                50,
                60,
                70,
                stop,
                timeout_triggered,
                failure,
                deadline_monotonic_ns,
                timing,
            )

        trigger = timing.triggered()
        self.assertIsNotNone(trigger)
        assert trigger is not None
        self.assertEqual(trigger.monotonic_ns, deadline_monotonic_ns + 11)
        self.assertTrue(timeout_triggered.is_set())
        self.assertFalse(failure.is_set())
        kill_cgroup.assert_called_once_with(50, expected_device_id=60, expected_inode=70)

    def test_pid_reuse_never_causes_a_bare_process_group_signal(self) -> None:
        running = SimpleNamespace(
            firecracker_process_group_id=44_444,
            cgroup_descriptor=10,
            cgroup_device_id=20,
            cgroup_inode=30,
        )
        with (
            patch('vaxreplay.agentic.firecracker._bound_firecracker_process_alive', return_value=False),
            patch(
                'vaxreplay.agentic.firecracker._pinned_cgroup_member_pids',
                side_effect=((55_555,), (55_555,), (), ()),
            ),
            patch('vaxreplay.agentic.firecracker._kill_pinned_cgroup') as kill_cgroup,
            patch('vaxreplay.agentic.firecracker.os.killpg') as killpg,
        ):
            cgroup_empty = firecracker_module._terminate_process_group(
                cast(RunningFirecrackerWorker, running),
                grace_seconds=0.0,
            )
        self.assertGreater(cgroup_empty.monotonic_ns, 0)
        kill_cgroup.assert_called_once_with(
            10,
            expected_device_id=20,
            expected_inode=30,
        )
        killpg.assert_not_called()

    def test_cleanup_and_authenticated_attestation_bind_complete_worker_lifecycle(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary))
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                prepared = supervisor.prepare(run_id=_RUN_ID)

            process = subprocess.Popen(  # noqa: S603 - fixed test interpreter argv; never a shell
                (sys.executable, '-c', 'pass'),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            try:

                def observe_child(**kwargs):
                    identity = kwargs['jailer_identity']
                    process.wait(timeout=2)
                    return SimpleNamespace(
                        pid=process.pid + 1,
                        parent_pid_at_observation=process.pid,
                        process_group_id=process.pid,
                        session_id=identity.session_id,
                        start_time_ticks=identity.start_time_ticks + 1,
                        executable_sha256=spec.runtime.firecracker.sha256,
                        pid_file_path=str(Path(prepared.jail_root) / 'firecracker.pid'),
                        pid_file_device_id=1,
                        pid_file_inode=2,
                        pidfd=os.open('/dev/null', os.O_RDONLY),
                        cgroup_descriptor=os.open('/dev/null', os.O_RDONLY),
                        cgroup_device_id=3,
                        cgroup_inode=4,
                        jailer_reaped_at=datetime.now(UTC),
                        jailer_reaped_monotonic_ns=time.monotonic_ns(),
                    )

                def wait_watchdog(*args) -> None:
                    args[7].wait()

                with (
                    patch.object(supervisor, 'preflight', return_value=_preflight(spec)),
                    patch('vaxreplay.agentic.firecracker.subprocess.Popen', return_value=process),
                    patch(
                        'vaxreplay.agentic.firecracker._open_pinned_jail_root',
                        side_effect=lambda path: os.open(path, os.O_RDONLY),
                    ),
                    patch(
                        'vaxreplay.agentic.firecracker._read_proc_process_identity',
                        return_value=SimpleNamespace(
                            process_group_id=process.pid,
                            session_id=42,
                            start_time_ticks=100,
                        ),
                    ),
                    patch('vaxreplay.agentic.firecracker._observe_launched_firecracker_child', observe_child),
                    patch('vaxreplay.agentic.firecracker._watchdog_process_group', wait_watchdog),
                ):
                    running = supervisor.launch(prepared)

                def terminate_test_process(_running, *, grace_seconds: float):
                    self.assertEqual(_running, running)
                    self.assertEqual(grace_seconds, 0.5)
                    return firecracker_module._observe_lifecycle_time()

                with patch(
                    'vaxreplay.agentic.firecracker._terminate_process_group', side_effect=terminate_test_process
                ):
                    cleanup = supervisor.terminate_and_cleanup(running, grace_seconds=0.5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)

            self.assertEqual(cleanup.lifecycle, 'terminated')
            self.assertIsNotNone(cleanup.jailer_exit_code)
            self.assertFalse(cleanup.wall_timeout_triggered)
            self.assertFalse(Path(prepared.jail_root).parent.exists())

            key = b'worker-attestation-key-material!' * 2
            key_id = firecracker_attestation_key_id(key)
            attempt = 'a' * 64
            authenticated = finalize_firecracker_worker_attestation(
                spec=spec,
                running=running,
                cleanup=cleanup,
                attempt_reservation_sha256=attempt,
                attestation_key=key,
                expected_attestation_key_id=key_id,
            )
            verified = verify_firecracker_worker_attestation(
                authenticated,
                attestation_key=key,
                expected_attestation_key_id=key_id,
                expected_run_id=_RUN_ID,
                expected_attempt_reservation_sha256=attempt,
                expected_worker_spec_sha256=firecracker_model_sha256(spec),
            )
            self.assertEqual(verified.config_sha256, prepared.config_sha256)
            self.assertTrue(verified.process_group_exit_verified)
            self.assertEqual(verified.finished_at, cleanup.cgroup_empty_at)
            self.assertLessEqual(
                verified.cgroup_empty_monotonic_ns,
                cleanup.cleanup_finished_monotonic_ns,
            )
            self.assertEqual(len(authenticated_firecracker_worker_attestation_sha256(authenticated)), 64)

            with self.assertRaises(ValidationError):
                type(cleanup).model_validate(
                    cleanup.model_dump() | {'schema_version': 'vaxreplay.firecracker-cleanup.v0.2'}
                )
            with self.assertRaises(ValidationError):
                type(authenticated.attestation).model_validate(
                    authenticated.attestation.model_dump()
                    | {'schema_version': 'vaxreplay.firecracker-worker-attestation.v0.1'}
                )
            with self.assertRaises(ValidationError):
                type(authenticated).model_validate(
                    authenticated.model_dump()
                    | {'schema_version': 'vaxreplay.authenticated-firecracker-worker-attestation.v0.1'}
                )

            tampered_attestation = authenticated.attestation.model_copy(update={'config_sha256': 'f' * 64})
            tampered = authenticated.model_copy(update={'attestation': tampered_attestation})
            with self.assertRaisesRegex(FirecrackerAttestationError, 'HMAC'):
                verify_firecracker_worker_attestation(
                    tampered,
                    attestation_key=key,
                    expected_attestation_key_id=key_id,
                    expected_run_id=_RUN_ID,
                    expected_attempt_reservation_sha256=attempt,
                    expected_worker_spec_sha256=firecracker_model_sha256(spec),
                )

            finalize_kwargs = {
                'spec': spec,
                'running': running,
                'cleanup': cleanup,
                'attempt_reservation_sha256': attempt,
                'attestation_key': key,
                'expected_attestation_key_id': key_id,
            }
            running.identity_descriptors_closed.clear()
            with self.assertRaisesRegex(FirecrackerAttestationError, 'descriptor remain open'):
                finalize_firecracker_worker_attestation(**finalize_kwargs)
            running.identity_descriptors_closed.set()

            running.watchdog_stop.clear()
            with self.assertRaisesRegex(FirecrackerAttestationError, 'watchdog has not been stopped'):
                finalize_firecracker_worker_attestation(**finalize_kwargs)
            running.watchdog_stop.set()

            with (
                patch.object(running.watchdog_thread, 'is_alive', return_value=True),
                self.assertRaisesRegex(FirecrackerAttestationError, 'watchdog has not been stopped'),
            ):
                finalize_firecracker_worker_attestation(**finalize_kwargs)

            running.watchdog_failure.set()
            with self.assertRaisesRegex(FirecrackerAttestationError, 'signaling failure'):
                finalize_firecracker_worker_attestation(**finalize_kwargs)
            running.watchdog_failure.clear()

            running.watchdog_timeout_triggered.set()
            with self.assertRaisesRegex(FirecrackerAttestationError, 'runtime watchdog event'):
                finalize_firecracker_worker_attestation(**finalize_kwargs)
            running.watchdog_timeout_triggered.clear()

            fake_cgroup = Path(temporary) / 'claimed-removed-cgroup'
            fake_cgroup.mkdir()
            with (
                patch('vaxreplay.agentic.firecracker._expected_cgroup_path', return_value=fake_cgroup),
                self.assertRaisesRegex(FirecrackerAttestationError, 'cgroup remains'),
            ):
                finalize_firecracker_worker_attestation(**finalize_kwargs)

            claimed_removed_vsock = Path(prepared.vsock_uds_path)
            claimed_removed_vsock.parent.mkdir(parents=True)
            claimed_removed_vsock.write_bytes(b'not actually removed')
            with self.assertRaisesRegex(FirecrackerAttestationError, 'run container remains'):
                finalize_firecracker_worker_attestation(**finalize_kwargs)

    def test_timeout_attests_actual_trigger_cgroup_empty_and_later_cleanup(self) -> None:
        with TemporaryDirectory() as temporary:
            spec = _make_spec(Path(temporary), wall_seconds=1)
            supervisor = FirecrackerSupervisor(spec)
            with patch.object(supervisor, 'preflight', return_value=_preflight(spec)):
                prepared = supervisor.prepare(run_id=_RUN_ID)

            process = subprocess.Popen(  # noqa: S603 - fixed test interpreter argv; never a shell
                (sys.executable, '-c', 'pass'),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            try:

                def observe_child(**kwargs):
                    identity = kwargs['jailer_identity']
                    process.wait(timeout=2)
                    return SimpleNamespace(
                        pid=process.pid + 1,
                        parent_pid_at_observation=process.pid,
                        process_group_id=process.pid,
                        session_id=identity.session_id,
                        start_time_ticks=identity.start_time_ticks + 1,
                        executable_sha256=spec.runtime.firecracker.sha256,
                        pid_file_path=str(Path(prepared.jail_root) / 'firecracker.pid'),
                        pid_file_device_id=1,
                        pid_file_inode=2,
                        pidfd=os.open('/dev/null', os.O_RDONLY),
                        cgroup_descriptor=os.open('/dev/null', os.O_RDONLY),
                        cgroup_device_id=3,
                        cgroup_inode=4,
                        jailer_reaped_at=datetime.now(UTC),
                        jailer_reaped_monotonic_ns=time.monotonic_ns(),
                    )

                def timeout_watchdog(*args) -> None:
                    stop = args[7]
                    timeout_triggered = args[8]
                    timing = args[11]
                    if not stop.wait(1.0):
                        timing.record_triggered()
                        timeout_triggered.set()

                with (
                    patch.object(supervisor, 'preflight', return_value=_preflight(spec)),
                    patch('vaxreplay.agentic.firecracker.subprocess.Popen', return_value=process),
                    patch(
                        'vaxreplay.agentic.firecracker._open_pinned_jail_root',
                        side_effect=lambda path: os.open(path, os.O_RDONLY),
                    ),
                    patch(
                        'vaxreplay.agentic.firecracker._read_proc_process_identity',
                        return_value=SimpleNamespace(
                            process_group_id=process.pid,
                            session_id=42,
                            start_time_ticks=100,
                        ),
                    ),
                    patch('vaxreplay.agentic.firecracker._observe_launched_firecracker_child', observe_child),
                    patch('vaxreplay.agentic.firecracker._watchdog_process_group', timeout_watchdog),
                ):
                    running = supervisor.launch(prepared)
                self.assertTrue(running.watchdog_timeout_triggered.wait(timeout=3))
                running.watchdog_thread.join(timeout=2)
                self.assertFalse(running.watchdog_thread.is_alive())
                with patch(
                    'vaxreplay.agentic.firecracker._terminate_process_group',
                    side_effect=lambda *_args, **_kwargs: firecracker_module._observe_lifecycle_time(),
                ):
                    cleanup = supervisor.terminate_and_cleanup(running, grace_seconds=0.5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)

            self.assertTrue(cleanup.wall_timeout_triggered)
            self.assertIsNotNone(cleanup.watchdog_triggered_at)
            watchdog_triggered_monotonic_ns = cleanup.watchdog_triggered_monotonic_ns
            cgroup_empty_monotonic_ns = cleanup.cgroup_empty_monotonic_ns
            assert watchdog_triggered_monotonic_ns is not None
            assert cgroup_empty_monotonic_ns is not None
            self.assertGreaterEqual(
                watchdog_triggered_monotonic_ns,
                running.wall_deadline_monotonic_ns,
            )
            self.assertGreaterEqual(
                cgroup_empty_monotonic_ns,
                watchdog_triggered_monotonic_ns,
            )
            delayed_cleanup = type(cleanup).model_validate(
                cleanup.model_dump()
                | {
                    'cleanup_finished_at': cleanup.cleanup_finished_at + timedelta(seconds=10),
                    'cleanup_finished_monotonic_ns': cleanup.cleanup_finished_monotonic_ns + 10_000_000_000,
                }
            )

            key = b'worker-attestation-key-material!' * 2
            authenticated = finalize_firecracker_worker_attestation(
                spec=spec,
                running=running,
                cleanup=delayed_cleanup,
                attempt_reservation_sha256='a' * 64,
                attestation_key=key,
                expected_attestation_key_id=firecracker_attestation_key_id(key),
            )
            worker = authenticated.attestation
            self.assertEqual(worker.finished_at, delayed_cleanup.cgroup_empty_at)
            self.assertEqual(worker.cgroup_empty_monotonic_ns, delayed_cleanup.cgroup_empty_monotonic_ns)
            self.assertGreaterEqual(worker.duration_ms, 1000)
            self.assertLess(worker.cgroup_empty_monotonic_ns, delayed_cleanup.cleanup_finished_monotonic_ns)
            self.assertEqual(worker.cleanup_receipt_sha256, firecracker_model_sha256(delayed_cleanup))


class FirecrackerVsockTest(unittest.TestCase):
    def _serve_once(self, server: socket.socket, response: bytes, observed: list[bytes]) -> threading.Thread:
        def serve() -> None:
            connection, _ = server.accept()
            with connection:
                request = bytearray()
                while not request.endswith(b'\n'):
                    chunk = connection.recv(1)
                    if not chunk:
                        break
                    request.extend(chunk)
                observed.append(bytes(request))
                connection.sendall(response)

        thread = threading.Thread(target=serve)
        thread.start()
        return thread

    def test_exact_connect_handshake_preserves_first_guest_payload_byte(self) -> None:
        with TemporaryDirectory() as temporary:
            uds_path = str(Path(temporary) / 'vsock.sock')
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(uds_path)
            server.listen(1)
            observed: list[bytes] = []
            thread = self._serve_once(server, b'OK 1073741824\nPAYLOAD', observed)
            connection = connect_firecracker_vsock(uds_path=uds_path, port=7000, timeout_seconds=2)
            try:
                self.assertEqual(connection.recv(7), b'PAYLOAD')
            finally:
                connection.close()
                thread.join(timeout=2)
                server.close()
            self.assertEqual(observed, [b'CONNECT 7000\n'])

    def test_guest_initiated_listener_uses_firecracker_port_suffix(self) -> None:
        self.assertEqual(
            firecracker_guest_initiated_uds_path(uds_path='/run/vsock.sock', port=7000),
            '/run/vsock.sock_7000',
        )

    def test_out_of_range_assigned_hostside_port_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            uds_path = str(Path(temporary) / 'vsock.sock')
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(uds_path)
            server.listen(1)
            observed: list[bytes] = []
            thread = self._serve_once(server, b'OK 4294967296\n', observed)
            with self.assertRaisesRegex(FirecrackerVsockError, 'out-of-range'):
                connect_firecracker_vsock(uds_path=uds_path, port=7000, timeout_seconds=2)
            thread.join(timeout=2)
            server.close()

    def test_bad_handshake_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            uds_path = str(Path(temporary) / 'vsock.sock')
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(uds_path)
            server.listen(1)
            observed: list[bytes] = []
            thread = self._serve_once(server, b'ERR 1\n', observed)
            with self.assertRaisesRegex(FirecrackerVsockError, 'invalid handshake'):
                connect_firecracker_vsock(uds_path=uds_path, port=7000, timeout_seconds=2)
            thread.join(timeout=2)
            server.close()

    def test_symlink_to_socket_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            socket_path = root / 'real.sock'
            link_path = root / 'linked.sock'
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(socket_path))
            link_path.symlink_to(socket_path)
            try:
                with self.assertRaisesRegex(FirecrackerVsockError, 'non-symlink'):
                    connect_firecracker_vsock(uds_path=str(link_path), port=7000, timeout_seconds=2)
            finally:
                server.close()


if __name__ == '__main__':
    unittest.main()
