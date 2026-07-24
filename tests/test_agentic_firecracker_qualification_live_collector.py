from __future__ import annotations

import hashlib
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.test_agentic_firecracker import _make_spec, _preflight
from vaxreplay.agentic import firecracker_qualification_collector as collector_module
from vaxreplay.agentic.firecracker import FirecrackerWorkerSpec, firecracker_model_sha256
from vaxreplay.agentic.firecracker_qualification import (
    FirecrackerHostObservation,
    FirecrackerQualificationClaim,
    FirecrackerQualificationDrillId,
    required_firecracker_qualification_claims,
)
from vaxreplay.agentic.firecracker_qualification_collector import (
    COLLECTOR_EVIDENCE_FILE,
    COLLECTOR_EVIDENCE_SHA256_FILE,
    PROBE_MANIFEST_FILE,
    WORKER_SPEC_FILE,
    FirecrackerQualificationCollectorError,
    collect_and_retain_firecracker_qualification_evidence,
    independently_verify_firecracker_qualification_collector_evidence,
    load_firecracker_qualification_collector_evidence,
)
from vaxreplay.agentic.firecracker_qualification_driver import FirecrackerQualificationDriverRequest
from vaxreplay.agentic.firecracker_qualification_probe import (
    FirecrackerQualificationBoundaryIdentity,
    FirecrackerQualificationBoundaryKind,
    FirecrackerQualificationCgroupSnapshot,
    FirecrackerQualificationChallenge,
    FirecrackerQualificationClaimMeasurement,
    FirecrackerQualificationCollectionMode,
    FirecrackerQualificationGuestCommand,
    FirecrackerQualificationGuestDiskBuildReceipt,
    FirecrackerQualificationGuestResponse,
    FirecrackerQualificationHostCgroupCanary,
    FirecrackerQualificationObservationSource,
    FirecrackerQualificationProbeError,
    FirecrackerQualificationProbeManifest,
    FirecrackerQualificationRawDrillObservation,
    FirecrackerQualificationTeardownMeasurement,
    FirecrackerQualificationWallTimeoutMeasurement,
    FirecrackerQualificationWorkerBinding,
    FirecrackerQualificationWorkerInterval,
    authenticate_firecracker_qualification_collection,
    derive_firecracker_full_suite_evidence,
    derive_firecracker_qualification_worker_spec,
    ed25519_public_key_bytes,
    firecracker_live_collector_key_id,
    firecracker_qualification_challenge_sha256,
    firecracker_qualification_guest_key_id,
    firecracker_qualification_probe_manifest_sha256,
    firecracker_qualification_static_config_sha256,
    firecracker_qualification_verifier_source_sha256,
    firecracker_qualification_worker_binding_sha256,
    sign_firecracker_qualification_guest_response,
    verify_authenticated_firecracker_qualification_collection,
)
from vaxreplay.bundle import canonical_json_bytes

_NOW = datetime(2025, 1, 1, tzinfo=UTC)
_GUEST_KEY = Ed25519PrivateKey.from_private_bytes(bytes.fromhex('11' * 32))
_COLLECTOR_KEY = Ed25519PrivateKey.from_private_bytes(bytes.fromhex('22' * 32))

_SOURCE_BY_CLAIM = {
    FirecrackerQualificationClaim.FIRECRACKER_PROCESS_STARTED: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.GUEST_READY_AUTHENTICATED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.CLEAN_GUEST_EXIT: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.HOST_VSOCK_HANDSHAKE: FirecrackerQualificationObservationSource.HOST_VSOCK,
    FirecrackerQualificationClaim.GUEST_RPC_ROUND_TRIP: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.PEER_CID_BOUND: FirecrackerQualificationObservationSource.HOST_VSOCK,
    FirecrackerQualificationClaim.ROOTFS_WRITE_DENIED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.HARNESS_WRITE_DENIED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.SCRATCH_WRITE_SUCCEEDED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.SCRATCH_FRESH: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.NETWORK_UNREACHABLE: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.MMDS_UNREACHABLE: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.CPU_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.MEMORY_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.SWAP_DISABLED_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.PIDS_LIMIT_OBSERVED: FirecrackerQualificationObservationSource.HOST_CGROUPFS,
    FirecrackerQualificationClaim.WALL_WATCHDOG_TRIGGERED: FirecrackerQualificationObservationSource.HOST_MONOTONIC,
    FirecrackerQualificationClaim.PROCESS_GROUP_KILLED: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.DEADLINE_BOUND: FirecrackerQualificationObservationSource.HOST_MONOTONIC,
    FirecrackerQualificationClaim.CGROUP_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.JAIL_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.VSOCK_ABSENT: FirecrackerQualificationObservationSource.HOST_LSTAT,
    FirecrackerQualificationClaim.PARALLEL_WORKERS_DISTINCT: FirecrackerQualificationObservationSource.HOST_PROCFS,
    FirecrackerQualificationClaim.ALL_WORKERS_COMPLETED: FirecrackerQualificationObservationSource.GUEST_SIGNED,
    FirecrackerQualificationClaim.ALL_WORKERS_TORN_DOWN: FirecrackerQualificationObservationSource.HOST_LSTAT,
}

_COMMAND_BY_DRILL = {
    FirecrackerQualificationDrillId.LIVE_BOOT: FirecrackerQualificationGuestCommand.BOOT_READY_AND_EXIT,
    FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP: FirecrackerQualificationGuestCommand.VSOCK_NONCE_ECHO,
    FirecrackerQualificationDrillId.GUEST_ISOLATION: FirecrackerQualificationGuestCommand.ISOLATION_PROBES,
    FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT: FirecrackerQualificationGuestCommand.CGROUP_STRESS,
    FirecrackerQualificationDrillId.WALL_TIMEOUT: FirecrackerQualificationGuestCommand.INTENTIONAL_HANG,
    FirecrackerQualificationDrillId.LOAD_CANARY: FirecrackerQualificationGuestCommand.LOAD_CANARY,
}


def _write_release(
    root: Path,
    *,
    spec: FirecrackerWorkerSpec | None = None,
):
    root.mkdir(parents=True, exist_ok=True)
    if spec is None:
        worker_root = root / 'worker'
        worker_root.mkdir()
        spec = _make_spec(worker_root)
    spec_path = root / 'worker-spec.json'
    spec_path.write_bytes(canonical_json_bytes(spec))
    spec_sha256 = firecracker_model_sha256(spec)
    guest_public_key = ed25519_public_key_bytes(_GUEST_KEY)
    receipt = FirecrackerQualificationGuestDiskBuildReceipt(
        source_date_epoch=1_700_000_000,
        base_rootfs_tree_sha256='d' * 64,
        package_tree_sha256='3' * 64,
        normalized_rootfs_tree_sha256='4' * 64,
        normalized_harness_tree_sha256='5' * 64,
        build_recipe_sha256='e' * 64,
        mke2fs_sha256='6' * 64,
        mke2fs_version='mke2fs test',
        e2fsck_sha256='7' * 64,
        e2fsck_version='e2fsck test',
        debugfs_sha256='8' * 64,
        debugfs_version='debugfs test',
        build_argv_and_env_sha256='9' * 64,
        init_sha256='f' * 64,
        guest_probe_executable_sha256='c' * 64,
        guest_config_sha256='1' * 64,
        rootfs_sha256='a' * 64,
        rootfs_byte_count=4096,
        harness_sha256='b' * 64,
        harness_byte_count=8192,
    )
    manifest = FirecrackerQualificationProbeManifest(
        manifest_id='qualification-guest-test-v1',
        task_worker_spec_sha256=spec_sha256,
        task_rootfs_sha256=spec.images.rootfs.sha256,
        task_harness_sha256=spec.images.harness.sha256,
        qualification_kernel_sha256=spec.images.kernel.sha256,
        qualification_rootfs_path='/opt/vaxreplay/qualification/rootfs.ext4',
        qualification_rootfs_sha256='a' * 64,
        qualification_rootfs_byte_count=4096,
        qualification_harness_path='/opt/vaxreplay/qualification/harness.ext4',
        qualification_harness_sha256='b' * 64,
        qualification_harness_byte_count=8192,
        qualification_disk_build_receipt=receipt,
        qualification_disk_build_receipt_sha256=firecracker_model_sha256(receipt),
        guest_probe_executable_sha256='c' * 64,
        guest_probe_public_key_hex=guest_public_key.hex(),
        guest_probe_key_id=firecracker_qualification_guest_key_id(guest_public_key),
    )
    manifest_path = root / 'probe-manifest.json'
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return spec, spec_path, spec_sha256, manifest, manifest_path


def _linux_observation() -> FirecrackerHostObservation:
    return FirecrackerHostObservation(
        collected_at=_NOW,
        host_os='Linux',
        host_architecture='aarch64',
        host_kernel_release='deterministic-development-kernel',
        effective_uid=0,
        kvm_path_present=True,
        kvm_non_symlink_character_device=True,
        kvm_read_write_access=True,
        cgroup_v2_controller_file_present=True,
        cgroup_controllers=('cpu', 'memory', 'pids'),
    )


def test_pinned_driver_request_model_is_canonical_json_serializable(tmp_path: Path) -> None:
    spec, _, spec_sha256, manifest, _ = _write_release(tmp_path)
    challenge = FirecrackerQualificationChallenge(
        collection_id='1' * 32,
        challenge_id='2' * 32,
        nonce_hex='3' * 64,
        drill_id=FirecrackerQualificationDrillId.LIVE_BOOT,
        run_ids=('4' * 32,),
        worker_spec_sha256=spec_sha256,
        probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
        issued_at=_NOW,
    )
    encoded = canonical_json_bytes(
        FirecrackerQualificationDriverRequest(
            challenge=challenge,
            worker_spec=spec,
            probe_manifest=manifest,
        )
    )

    decoded = FirecrackerQualificationDriverRequest.model_validate_json(encoded)
    assert decoded.challenge == challenge
    assert canonical_json_bytes(decoded) == encoded


def test_failed_driver_diagnostic_is_bounded_hash_only_metadata(tmp_path: Path) -> None:
    _, _, spec_sha256, manifest, _ = _write_release(tmp_path)
    challenge = FirecrackerQualificationChallenge(
        collection_id='1' * 32,
        challenge_id='2' * 32,
        nonce_hex='3' * 64,
        drill_id=FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT,
        run_ids=('4' * 32,),
        worker_spec_sha256=spec_sha256,
        probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
        issued_at=_NOW,
    )
    stdout = b'model-or-task-content-on-stdout'
    stderr = b'sensitive-driver-detail-on-stderr'
    diagnostic = collector_module._driver_failure_diagnostic(
        challenge=challenge,
        failure_kind=collector_module.FirecrackerDriverFailureKind.PROCESS_EXIT,
        exit_status=70,
        stdout_byte_count=len(stdout),
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_byte_count=len(stderr),
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
    )

    message = collector_module._driver_failure_message('driver failed', diagnostic)
    encoded = message.partition('diagnostic=')[2]
    decoded = collector_module.FirecrackerDriverFailureDiagnostic.model_validate_json(encoded)

    assert decoded == diagnostic
    assert diagnostic.drill_id == FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT
    assert diagnostic.exit_status == 70
    assert diagnostic.stdout_byte_count == len(stdout)
    assert diagnostic.stdout_sha256 == hashlib.sha256(stdout).hexdigest()
    assert diagnostic.stderr_byte_count == len(stderr)
    assert diagnostic.stderr_sha256 == hashlib.sha256(stderr).hexdigest()
    assert diagnostic.stderr_content_retained is False
    assert stdout.decode() not in message
    assert stderr.decode() not in message
    assert challenge.challenge_id not in message
    assert challenge.nonce_hex not in message
    assert challenge.worker_spec_sha256 not in message
    assert len(message) < 1_024


def test_driver_exit_status_is_bounded_before_reporting() -> None:
    assert collector_module._bounded_driver_exit_status(-9) == -9
    assert collector_module._bounded_driver_exit_status(70) == 70
    assert collector_module._bounded_driver_exit_status(256) == 'unknown'


def test_driver_capture_retains_small_stdout_but_only_hashes_stderr(tmp_path: Path) -> None:
    stdout = b'{"small":true}'
    stderr = b'sensitive-stderr-is-not-retained'
    result = collector_module._run_bounded_driver_process(
        argv=(
            sys.executable,
            '-c',
            f'import os; os.write(1, {stdout!r}); os.write(2, {stderr!r})',
        ),
        request=b'',
        cwd=tmp_path,
        env={},
        pass_fds=(),
        timeout_seconds=5,
        stdout_byte_limit=64 * 1024,
        stderr_byte_limit=64 * 1024,
    )

    assert result.failure_kind is None
    assert result.exit_status == 0
    assert result.stdout == stdout
    assert result.stdout_byte_count == len(stdout)
    assert result.stdout_sha256 == hashlib.sha256(stdout).hexdigest()
    assert result.stderr_byte_count == len(stderr)
    assert result.stderr_sha256 == hashlib.sha256(stderr).hexdigest()
    assert not hasattr(result, 'stderr')


@pytest.mark.parametrize(
    ('file_descriptor', 'byte_value', 'expected_kind'),
    (
        (1, b'x', collector_module.FirecrackerDriverFailureKind.STDOUT_LIMIT_EXCEEDED),
        (2, b'y', collector_module.FirecrackerDriverFailureKind.STDERR_LIMIT_EXCEEDED),
    ),
)
def test_driver_capture_kills_oversized_output_without_unbounded_retention(
    tmp_path: Path,
    file_descriptor: int,
    byte_value: bytes,
    expected_kind: collector_module.FirecrackerDriverFailureKind,
) -> None:
    byte_limit = 64 * 1024
    command = (
        sys.executable,
        '-c',
        (
            'import os\n'
            f'chunk={byte_value!r} * {collector_module._DRIVER_STREAM_CHUNK_BYTES}\n'
            f'fd={file_descriptor}\n'
            'while True:\n'
            '    os.write(fd, chunk)\n'
        ),
    )

    result = collector_module._run_bounded_driver_process(
        argv=command,
        request=b'',
        cwd=tmp_path,
        env={},
        pass_fds=(),
        timeout_seconds=5,
        stdout_byte_limit=byte_limit,
        stderr_byte_limit=byte_limit,
    )

    assert result.failure_kind == expected_kind
    observed_count = result.stdout_byte_count if file_descriptor == 1 else result.stderr_byte_count
    observed_sha256 = result.stdout_sha256 if file_descriptor == 1 else result.stderr_sha256
    assert byte_limit < observed_count <= byte_limit + 1024 * 1024
    assert observed_sha256 == hashlib.sha256(byte_value * observed_count).hexdigest()
    if file_descriptor == 1:
        assert len(result.stdout) == byte_limit
    else:
        assert result.stdout == b''
        assert result.stdout_byte_count == 0
        assert result.stdout_sha256 == hashlib.sha256(b'').hexdigest()


class _DeterministicBoundary:
    def __init__(self, spec, manifest: FirecrackerQualificationProbeManifest, *, failed_claim=None) -> None:
        self.spec = spec
        self.manifest = manifest
        self.failed_claim = failed_claim
        self.counter = 0

    @property
    def identity(self) -> FirecrackerQualificationBoundaryIdentity:
        return FirecrackerQualificationBoundaryIdentity(
            boundary_id='deterministic-development-boundary',
            kind=FirecrackerQualificationBoundaryKind.DETERMINISTIC_DEVELOPMENT,
            executable_sha256='d' * 64,
            external_executable_pin_enforced=False,
            direct_linux_kvm_launch=False,
            injected_test_boundary=True,
        )

    def live_boot(self, challenge):
        return self._raw(challenge)

    def vsock_round_trip(self, challenge):
        return self._raw(challenge)

    def guest_isolation(self, challenge):
        return self._raw(challenge)

    def cgroup_enforcement(self, challenge):
        return self._raw(challenge)

    def wall_timeout(self, challenge):
        return self._raw(challenge)

    def teardown(self, challenge):
        return self._raw(challenge)

    def load_canary(self, challenge):
        return self._raw(challenge)

    def _raw(self, challenge: FirecrackerQualificationChallenge) -> FirecrackerQualificationRawDrillObservation:
        started = challenge.issued_at + timedelta(milliseconds=1)
        finished = started + timedelta(milliseconds=10)
        bindings = tuple(self._binding(run_id) for run_id in challenge.run_ids)
        required = required_firecracker_qualification_claims(challenge.drill_id)
        measurements = tuple(
            FirecrackerQualificationClaimMeasurement(
                claim=claim,
                source=_SOURCE_BY_CLAIM[claim],
                observed=claim != self.failed_claim,
                raw_observation_sha256=hashlib.sha256(f'{challenge.challenge_id}:{claim.value}'.encode()).hexdigest(),
                observed_at=finished,
            )
            for claim in required
        )
        guest_claims = tuple(claim for claim in required if _SOURCE_BY_CLAIM[claim].value == 'guest_signed')
        responses = ()
        if challenge.drill_id != FirecrackerQualificationDrillId.TEARDOWN:
            responses = tuple(self._response(challenge, binding, guest_claims, finished) for binding in bindings)
        cgroups = ()
        host_cgroup_canary = None
        if challenge.drill_id == FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT:
            binding = bindings[0]
            memory_helper = binding.firecracker_pid + 10
            pids_helper = binding.firecracker_pid + 20
            descendants = (binding.firecracker_pid + 21, binding.firecracker_pid + 22)
            cgroups = (
                self._cgroup(binding, 0, started, memory=0, pids=0),
                self._cgroup(binding, 1, started + timedelta(milliseconds=1), memory=0, pids=0),
                self._cgroup(
                    binding,
                    2,
                    started + timedelta(milliseconds=2),
                    memory=0,
                    pids=0,
                    members=(binding.firecracker_pid, memory_helper),
                ),
                self._cgroup(binding, 3, started + timedelta(milliseconds=3), memory=1, pids=0),
                self._cgroup(
                    binding,
                    4,
                    started + timedelta(milliseconds=4),
                    memory=1,
                    pids=1,
                    members=(binding.firecracker_pid, pids_helper, *descendants),
                ),
                self._cgroup(binding, 5, started + timedelta(milliseconds=5), memory=1, pids=1),
            )
            host_cgroup_canary = FirecrackerQualificationHostCgroupCanary(
                run_id=binding.run_id,
                cgroup_path=binding.cgroup_path,
                cgroup_inode=binding.cgroup_inode,
                firecracker_pid=binding.firecracker_pid,
                memory_helper_pid=memory_helper,
                pids_helper_pid=pids_helper,
                pids_helper_descendant_pids=descendants,
                started_monotonic_ns=1,
                memory_helper_reaped_monotonic_ns=2,
                pids_limit_observed_monotonic_ns=3,
                pids_helper_reaped_monotonic_ns=4,
                finished_monotonic_ns=5,
                allowed_duration_ns=10,
                baseline_snapshot_sha256=firecracker_model_sha256(cgroups[0]),
                guest_pressure_snapshot_sha256=firecracker_model_sha256(cgroups[1]),
                memory_armed_snapshot_sha256=firecracker_model_sha256(cgroups[2]),
                memory_triggered_snapshot_sha256=firecracker_model_sha256(cgroups[3]),
                pids_peak_snapshot_sha256=firecracker_model_sha256(cgroups[4]),
                cleanup_snapshot_sha256=firecracker_model_sha256(cgroups[5]),
            )
        wall = None
        if challenge.drill_id == FirecrackerQualificationDrillId.WALL_TIMEOUT:
            binding = bindings[0]
            wall = FirecrackerQualificationWallTimeoutMeasurement(
                run_id=binding.run_id,
                process_group_id=binding.process_group_id,
                armed_monotonic_ns=1,
                deadline_monotonic_ns=100,
                watchdog_triggered_monotonic_ns=100,
                process_group_reaped_monotonic_ns=110,
                allowed_teardown_grace_ns=20,
                member_pids_before_kill=(binding.firecracker_pid,),
            )
        teardowns = ()
        if challenge.drill_id in {
            FirecrackerQualificationDrillId.TEARDOWN,
            FirecrackerQualificationDrillId.LOAD_CANARY,
        }:
            teardowns = tuple(
                FirecrackerQualificationTeardownMeasurement(
                    run_id=binding.run_id,
                    cgroup_path=binding.cgroup_path,
                    jail_root=binding.jail_root,
                    vsock_uds_path=binding.vsock_uds_path,
                    observed_at=finished,
                )
                for binding in bindings
            )
        intervals = ()
        if challenge.drill_id == FirecrackerQualificationDrillId.LOAD_CANARY:
            intervals = tuple(
                FirecrackerQualificationWorkerInterval(
                    run_id=binding.run_id,
                    started_monotonic_ns=10 + index,
                    finished_monotonic_ns=50 + index,
                )
                for index, binding in enumerate(bindings)
            )
        return FirecrackerQualificationRawDrillObservation(
            drill_id=challenge.drill_id,
            challenge=challenge,
            started_at=started,
            finished_at=finished,
            worker_bindings=bindings,
            guest_responses=responses,
            claim_measurements=measurements,
            cgroup_snapshots=cgroups,
            host_cgroup_canary=host_cgroup_canary,
            wall_timeout=wall,
            teardown_measurements=teardowns,
            worker_intervals=intervals,
        )

    def _binding(self, run_id: str) -> FirecrackerQualificationWorkerBinding:
        self.counter += 1
        pid = 1000 + self.counter
        qualification_spec = derive_firecracker_qualification_worker_spec(
            self.manifest,
            task_worker_spec=self.spec,
        )
        executable_name = Path(qualification_spec.runtime.firecracker.source_path).name
        jailer_pid = pid + 1000
        jail = str(Path(self.spec.chroot_base_dir) / executable_name / run_id / 'root')
        return FirecrackerQualificationWorkerBinding(
            run_id=run_id,
            worker_spec_sha256=firecracker_model_sha256(self.spec),
            qualification_worker_spec_sha256=firecracker_model_sha256(qualification_spec),
            qualification_static_config_sha256=firecracker_qualification_static_config_sha256(qualification_spec),
            prepared_worker_sha256='2' * 64,
            probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(self.manifest),
            firecracker_pid=pid,
            firecracker_parent_pid_at_observation=jailer_pid,
            firecracker_start_time_ticks=100_000 + self.counter,
            firecracker_session_id=500,
            firecracker_executable_sha256=self.spec.runtime.firecracker.sha256,
            firecracker_pid_file_path=f'{jail}/{executable_name}.pid',
            firecracker_pid_file_device_id=20_000 + self.counter,
            firecracker_pid_file_inode=30_000 + self.counter,
            jailer_pid=jailer_pid,
            jailer_start_time_ticks=99_000 + self.counter,
            jailer_process_group_id=jailer_pid,
            jailer_session_id=500,
            process_group_id=jailer_pid,
            worker_uid=self.spec.worker_uid,
            worker_gid=self.spec.worker_gid,
            cgroup_path=f'/sys/fs/cgroup/{self.spec.cgroup_parent}/{run_id}',
            cgroup_inode=10_000 + self.counter,
            cgroup_member_pids=(pid,),
            jail_root=jail,
            vsock_uds_path=f'{jail}/run/vsock.sock',
            guest_cid=self.spec.guest_cid,
            peer_pid=pid,
            peer_uid=self.spec.worker_uid,
            peer_gid=self.spec.worker_gid,
            process_tree_verified=True,
            pid_cgroup_binding_verified=True,
        )

    def _response(self, challenge, binding, guest_claims, responded_at):
        response = FirecrackerQualificationGuestResponse(
            challenge_sha256=firecracker_qualification_challenge_sha256(challenge),
            nonce_hex=challenge.nonce_hex,
            run_id=binding.run_id,
            worker_binding_sha256=firecracker_qualification_worker_binding_sha256(binding),
            worker_spec_sha256=firecracker_model_sha256(self.spec),
            probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(self.manifest),
            guest_probe_executable_sha256=self.manifest.guest_probe_executable_sha256,
            command=_COMMAND_BY_DRILL[challenge.drill_id],
            verified_guest_claims=guest_claims,
            result_bytes_sha256=hashlib.sha256(binding.run_id.encode()).hexdigest(),
            responded_at=responded_at,
        )
        return sign_firecracker_qualification_guest_response(response, private_key=_GUEST_KEY)

    def _cgroup(self, binding, index, observed_at, *, memory, pids, members=None):
        return FirecrackerQualificationCgroupSnapshot(
            run_id=binding.run_id,
            cgroup_path=binding.cgroup_path,
            cgroup_inode=binding.cgroup_inode,
            observed_at=observed_at,
            cpu_max_quota_us=self.spec.limits.cpu_quota_us,
            cpu_max_period_us=self.spec.limits.cpu_period_us,
            memory_max_bytes=self.spec.limits.memory_mib * 1024 * 1024,
            memory_swap_max_bytes=0,
            pids_max=self.spec.limits.pids,
            cpu_nr_throttled=index,
            cpu_throttled_usec=index * 100,
            memory_oom=memory,
            memory_oom_kill=memory,
            pids_max_events=pids,
            member_pids=members or (binding.firecracker_pid,),
        )


def _collect(
    tmp_path: Path,
    *,
    failed_claim=None,
    spec: FirecrackerWorkerSpec | None = None,
):
    spec, spec_path, spec_sha256, manifest, manifest_path = _write_release(
        tmp_path,
        spec=spec,
    )
    public_key = ed25519_public_key_bytes(_COLLECTOR_KEY)
    output = tmp_path / 'evidence'
    clock_counter = 0

    def clock() -> datetime:
        nonlocal clock_counter
        clock_counter += 1
        return _NOW + timedelta(seconds=clock_counter)

    loaded = collect_and_retain_firecracker_qualification_evidence(
        worker_spec_path=spec_path,
        expected_worker_spec_sha256=spec_sha256,
        probe_manifest_path=manifest_path,
        expected_probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
        boundary=_DeterministicBoundary(spec, manifest, failed_claim=failed_claim),
        mode=FirecrackerQualificationCollectionMode.DEVELOPMENT_SIMULATED,
        collector_private_key=_COLLECTOR_KEY,
        expected_collector_key_id=firecracker_live_collector_key_id(public_key),
        expected_collector_source_sha256=hashlib.sha256(
            Path('src/vaxreplay/agentic/firecracker_qualification_collector.py').read_bytes()
        ).hexdigest(),
        output_root=output,
        development_host_observation=_linux_observation(),
        development_host_preflight=_preflight(spec),
        development_collection_id='7' * 32,
        development_clock=clock,
    )
    return spec, spec_sha256, manifest, public_key, loaded


def test_development_collector_retains_all_raw_drills_but_can_never_qualify(tmp_path: Path) -> None:
    spec, spec_sha256, manifest, public_key, loaded = _collect(tmp_path)
    collection = loaded.authenticated.collection
    assert collection.development_simulated is True
    assert collection.production_qualification_eligible is False
    assert tuple(drill.drill_id for drill in collection.drills) == tuple(FirecrackerQualificationDrillId)
    assert len({drill.challenge.nonce_hex for drill in collection.drills}) == 7
    assert len(collection.drills[3].cgroup_snapshots) == 6
    assert collection.drills[3].host_cgroup_canary is not None
    assert collection.drills[4].wall_timeout is not None
    assert len(collection.drills[6].worker_bindings) == 2
    assert len(collection.drills[6].teardown_measurements) == 2
    assert {path.name for path in Path(loaded.root).iterdir()} == {
        COLLECTOR_EVIDENCE_FILE,
        WORKER_SPEC_FILE,
        PROBE_MANIFEST_FILE,
        COLLECTOR_EVIDENCE_SHA256_FILE,
    }
    assert stat.S_IMODE(Path(loaded.root).stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in Path(loaded.root).iterdir())
    assert derive_firecracker_full_suite_evidence(collection, worker_spec=spec).all_required_drills_passed is False
    with pytest.raises(FirecrackerQualificationCollectorError, match='development/simulated'):
        independently_verify_firecracker_qualification_collector_evidence(
            Path(loaded.root),
            expected_evidence_sha256=loaded.evidence_sha256,
            expected_worker_spec_sha256=spec_sha256,
            expected_probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
            expected_host_preflight_sha256=collection.host_preflight_sha256,
            expected_collector_public_key_hex=public_key.hex(),
            expected_collector_key_id=firecracker_live_collector_key_id(public_key),
            expected_verifier_source_sha256=firecracker_qualification_verifier_source_sha256(),
        )


def test_raw_failure_is_retained_and_independently_derived_as_failed(tmp_path: Path) -> None:
    spec, _, _, _, loaded = _collect(
        tmp_path,
        failed_claim=FirecrackerQualificationClaim.ROOTFS_WRITE_DENIED,
    )
    suite = derive_firecracker_full_suite_evidence(loaded.authenticated.collection, worker_spec=spec)
    assert suite.guest_isolation.passed is False
    assert suite.guest_isolation.failed_claims == (FirecrackerQualificationClaim.ROOTFS_WRITE_DENIED,)


def test_load_canary_signed_responses_are_completion_evidence_without_self_asserted_claims(
    tmp_path: Path,
) -> None:
    spec, _, _, _, loaded = _collect(tmp_path)
    collection = loaded.authenticated.collection
    load_canary = collection.drills[-1]
    responses = tuple(
        sign_firecracker_qualification_guest_response(
            authenticated.response.model_copy(update={'verified_guest_claims': ()}),
            private_key=_GUEST_KEY,
        )
        for authenticated in load_canary.guest_responses
    )
    updated_drill = load_canary.model_copy(update={'guest_responses': responses})
    updated_collection = collection.model_copy(update={'drills': (*collection.drills[:-1], updated_drill)})

    suite = derive_firecracker_full_suite_evidence(updated_collection, worker_spec=spec)
    assert suite.load_canary.passed is True
    assert FirecrackerQualificationClaim.ALL_WORKERS_COMPLETED in suite.load_canary.verified_claims


def test_independent_verifier_rejects_runtime_closure_substitution(tmp_path: Path) -> None:
    spec, spec_sha256, manifest, public_key, loaded = _collect(tmp_path)
    manifest_pin = '1' * 64
    receipt_pin = '2' * 64
    closure_pin = '3' * 64
    boundary = FirecrackerQualificationBoundaryIdentity(
        boundary_id='pinned-linux-kvm-test-boundary',
        kind=FirecrackerQualificationBoundaryKind.PINNED_LINUX_KVM_DRIVER,
        executable_sha256='d' * 64,
        external_executable_pin_enforced=True,
        direct_linux_kvm_launch=True,
        injected_test_boundary=False,
        runtime_closure_manifest_sha256=manifest_pin,
        runtime_closure_receipt_sha256=receipt_pin,
        runtime_closure_sha256=closure_pin,
        transitive_runtime_pin_enforced=True,
    )
    production = loaded.authenticated.collection.model_copy(
        update={
            'mode': FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM,
            'boundary_identity': boundary,
            'driver_runtime_closure_sha256': closure_pin,
            'development_simulated': False,
            'production_qualification_eligible': True,
        }
    )
    authenticated = authenticate_firecracker_qualification_collection(production, private_key=_COLLECTOR_KEY)
    published = collector_module._publish_collector_evidence(
        output_root=tmp_path / 'production-publish-regression',
        authenticated=authenticated,
        spec_bytes=canonical_json_bytes(spec),
        manifest_bytes=canonical_json_bytes(manifest),
        expected_worker_spec_sha256=spec_sha256,
        expected_probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
        expected_collector_public_key_hex=public_key.hex(),
        expected_collector_key_id=firecracker_live_collector_key_id(public_key),
        expected_driver_runtime_closure_manifest_sha256=manifest_pin,
        expected_driver_runtime_closure_receipt_sha256=receipt_pin,
        expected_driver_runtime_closure_sha256=closure_pin,
    )
    assert published.authenticated == authenticated
    common = {
        'worker_spec': spec,
        'expected_collector_public_key_hex': public_key.hex(),
        'expected_collector_key_id': firecracker_live_collector_key_id(public_key),
        'expected_worker_spec_sha256': spec_sha256,
        'expected_probe_manifest_sha256': firecracker_qualification_probe_manifest_sha256(manifest),
        'expected_driver_runtime_closure_manifest_sha256': manifest_pin,
        'expected_driver_runtime_closure_receipt_sha256': receipt_pin,
        'expected_host_preflight_sha256': production.host_preflight_sha256,
        'verifier_source_sha256': firecracker_qualification_verifier_source_sha256(),
    }
    verified = verify_authenticated_firecracker_qualification_collection(
        authenticated,
        expected_driver_runtime_closure_sha256=closure_pin,
        **common,
    )
    assert verified.production_qualification_eligible is True
    for substituted_pin in (
        'expected_driver_runtime_closure_manifest_sha256',
        'expected_driver_runtime_closure_receipt_sha256',
        'expected_driver_runtime_closure_sha256',
    ):
        arguments: dict[str, Any] = dict(common)
        arguments['expected_driver_runtime_closure_sha256'] = closure_pin
        arguments[substituted_pin] = '4' * 64
        with pytest.raises(FirecrackerQualificationProbeError, match='runtime closure'):
            verify_authenticated_firecracker_qualification_collection(authenticated, **arguments)


def test_runtime_closure_identity_is_strictly_production_only() -> None:
    with pytest.raises(ValueError, match='transitive runtime pin'):
        FirecrackerQualificationBoundaryIdentity(
            boundary_id='missing-production-closure',
            kind=FirecrackerQualificationBoundaryKind.PINNED_LINUX_KVM_DRIVER,
            executable_sha256='d' * 64,
            external_executable_pin_enforced=True,
            direct_linux_kvm_launch=True,
            injected_test_boundary=False,
        )
    with pytest.raises(ValueError, match='production boundary'):
        FirecrackerQualificationBoundaryIdentity(
            boundary_id='development-cannot-claim-closure',
            kind=FirecrackerQualificationBoundaryKind.DETERMINISTIC_DEVELOPMENT,
            executable_sha256='d' * 64,
            external_executable_pin_enforced=False,
            direct_linux_kvm_launch=False,
            injected_test_boundary=True,
            runtime_closure_manifest_sha256='1' * 64,
            runtime_closure_receipt_sha256='2' * 64,
            runtime_closure_sha256='3' * 64,
            transitive_runtime_pin_enforced=True,
        )


def test_evidence_is_create_once_externally_pinned_and_signature_authenticated(tmp_path: Path) -> None:
    _, spec_sha256, manifest, public_key, loaded = _collect(tmp_path)
    reloaded = load_firecracker_qualification_collector_evidence(
        Path(loaded.root),
        expected_evidence_sha256=loaded.evidence_sha256,
        expected_worker_spec_sha256=spec_sha256,
        expected_probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
        expected_collector_public_key_hex=public_key.hex(),
        expected_collector_key_id=firecracker_live_collector_key_id(public_key),
    )
    assert reloaded == loaded
    with pytest.raises(FirecrackerQualificationCollectorError, match='external pin'):
        load_firecracker_qualification_collector_evidence(
            Path(loaded.root),
            expected_evidence_sha256='f' * 64,
            expected_worker_spec_sha256=spec_sha256,
            expected_probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
            expected_collector_public_key_hex=public_key.hex(),
            expected_collector_key_id=firecracker_live_collector_key_id(public_key),
        )


def test_production_collection_rejects_simulated_boundary_before_host_access(tmp_path: Path) -> None:
    spec, spec_path, spec_sha256, manifest, manifest_path = _write_release(tmp_path)
    public_key = ed25519_public_key_bytes(_COLLECTOR_KEY)
    with pytest.raises(FirecrackerQualificationCollectorError, match='pinned Linux/KVM driver'):
        collect_and_retain_firecracker_qualification_evidence(
            worker_spec_path=spec_path,
            expected_worker_spec_sha256=spec_sha256,
            probe_manifest_path=manifest_path,
            expected_probe_manifest_sha256=firecracker_qualification_probe_manifest_sha256(manifest),
            boundary=_DeterministicBoundary(spec, manifest),
            mode=FirecrackerQualificationCollectionMode.PRODUCTION_LINUX_KVM,
            collector_private_key=_COLLECTOR_KEY,
            expected_collector_key_id=firecracker_live_collector_key_id(public_key),
            expected_collector_source_sha256=hashlib.sha256(
                Path('src/vaxreplay/agentic/firecracker_qualification_collector.py').read_bytes()
            ).hexdigest(),
            output_root=tmp_path / 'must-not-exist',
        )
    assert not (tmp_path / 'must-not-exist').exists()
