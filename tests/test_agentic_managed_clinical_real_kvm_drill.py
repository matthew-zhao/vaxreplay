from __future__ import annotations

import hashlib
import inspect
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

import vaxreplay.agentic.managed_clinical_real_kvm_drill as drill_module
from vaxreplay.agentic.managed_clinical_real_kvm_drill import (
    ManagedClinicalRealKvmExternalPins,
    ManagedClinicalRealKvmObservationGateRelease,
    ManagedClinicalRealKvmProcessObservation,
)
from vaxreplay.agentic.managed_clinical_registry import (
    ManagedClinicalPeerIdentity,
    ManagedClinicalRegistryAuditServerIdentity,
    ManagedClinicalRegistryConfig,
)

NOW = datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)
LAUNCHER_ID = 'vaxreplay-lane-a-canonical-operator'
LAUNCHER_SHA = '1' * 64
SERVICE_SHA = '2' * 64


def _registry_config(tmp_path) -> ManagedClinicalRegistryConfig:  # noqa: ANN001
    # Darwin's pytest base path is long enough to cross Linux's 107-byte
    # AF_UNIX pathname ceiling as the suite counter gains digits.  Keep the
    # database/audit fixtures isolated under ``tmp_path``, but give this
    # metadata-only socket identity a deterministic short pathname.
    socket_name = hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:24]
    return ManagedClinicalRegistryConfig(
        service_id='vaxreplay-managed-registry',
        service_version='test-v1',
        registry_authority_id='organizer.lane-a.example',
        database_path=str(tmp_path / 'registry' / 'attempts.sqlite3'),
        socket_path=f'/tmp/vrk-test-{socket_name}.sock',
        production_evidence_root=str(tmp_path / 'evidence'),
        protocol_audit_root=str(tmp_path / 'audit'),
        canonical_launcher_id=LAUNCHER_ID,
        canonical_launcher_executable_sha256=LAUNCHER_SHA,
        launcher_process_executable_sha256='3' * 64,
        service_process_executable_sha256=SERVICE_SHA,
        startup_config_sha256='4' * 64,
        startup_cleanup_receipt_key_id='5' * 64,
        connection_timeout_seconds=5,
    )


def _peer(*, pid: int = 501) -> ManagedClinicalPeerIdentity:
    return ManagedClinicalPeerIdentity(
        pid=pid,
        uid=0,
        gid=0,
        canonical_launcher_id=LAUNCHER_ID,
        canonical_launcher_executable_sha256=LAUNCHER_SHA,
    )


def _server(
    config: ManagedClinicalRegistryConfig,
    *,
    pid: int,
    start_time_ticks: int,
    socket_inode: int,
    database_inode: int = 22,
) -> ManagedClinicalRegistryAuditServerIdentity:
    return ManagedClinicalRegistryAuditServerIdentity(
        service_pid=pid,
        service_start_time_ticks=start_time_ticks,
        service_uid=0,
        service_gid=0,
        service_executable_sha256=SERVICE_SHA,
        socket_path=config.socket_path,
        socket_device_id=11,
        socket_inode=socket_inode,
        database_path=config.database_path,
        database_device_id=21,
        database_inode=database_inode,
    )


def _audit(server: ManagedClinicalRegistryAuditServerIdentity) -> SimpleNamespace:
    return SimpleNamespace(launcher_peer=_peer(pid=server.service_pid), server=server)


def test_registry_restart_proof_requires_new_service_and_socket_but_same_database(
    tmp_path,
) -> None:  # noqa: ANN001
    config = _registry_config(tmp_path)
    evidence = cast(Any, SimpleNamespace(registry_config=config))
    first = _audit(_server(config, pid=1001, start_time_ticks=50, socket_inode=12))
    second = _audit(_server(config, pid=2002, start_time_ticks=75, socket_inode=13))

    drill_module._verify_registry_server_restart(evidence, cast(Any, first), cast(Any, second))

    same_process = SimpleNamespace(
        launcher_peer=first.launcher_peer,
        server=second.server.model_copy(
            update={
                'service_pid': first.server.service_pid,
                'service_start_time_ticks': first.server.service_start_time_ticks,
            }
        ),
    )
    with pytest.raises(ValueError, match='not observed after an authority restart'):
        drill_module._verify_registry_server_restart(
            evidence,
            cast(Any, first),
            cast(Any, same_process),
        )

    same_socket = SimpleNamespace(
        launcher_peer=second.launcher_peer,
        server=second.server.model_copy(update={'socket_inode': first.server.socket_inode}),
    )
    with pytest.raises(ValueError, match='not observed after an authority restart'):
        drill_module._verify_registry_server_restart(
            evidence,
            cast(Any, first),
            cast(Any, same_socket),
        )

    replaced_database = SimpleNamespace(
        launcher_peer=second.launcher_peer,
        server=second.server.model_copy(update={'database_inode': 23}),
    )
    with pytest.raises(ValueError, match='not observed after an authority restart'):
        drill_module._verify_registry_server_restart(
            evidence,
            cast(Any, first),
            cast(Any, replaced_database),
        )


def test_registry_restart_proof_rejects_forged_launcher_identity(tmp_path) -> None:  # noqa: ANN001
    config = _registry_config(tmp_path)
    evidence = cast(Any, SimpleNamespace(registry_config=config))
    first = _audit(_server(config, pid=1001, start_time_ticks=50, socket_inode=12))
    forged = SimpleNamespace(
        launcher_peer=_peer().model_copy(update={'canonical_launcher_executable_sha256': 'f' * 64}),
        server=_server(config, pid=2002, start_time_ticks=75, socket_inode=13),
    )

    with pytest.raises(ValueError, match='unauthorized launcher peer'):
        drill_module._verify_registry_server_restart(
            evidence,
            cast(Any, first),
            cast(Any, forged),
        )


def _external_pins() -> ManagedClinicalRealKvmExternalPins:
    execution_policy = {'policy': 'fixed-test-policy'}
    guest_rpc_policy = {'policy': 'fixed-test-rpc-policy'}
    qualification_collector_public_key_hex = '1' * 64
    collector_public_key_hex = '2' * 64
    release = {
        'worker_spec_sha256': 'c' * 64,
        'execution_policy_sha256': drill_module.agentic_policy_sha256(cast(Any, execution_policy)),
        'guest_rpc_policy_sha256': drill_module.guest_rpc_policy_sha256(cast(Any, guest_rpc_policy)),
        'guest_config_sha256': '3' * 64,
        'disk_build_receipt_sha256': 'd' * 64,
        'qualification_key_id': '4' * 64,
        'qualification_artifact_sha256': 'e' * 64,
        'qualification_collector_evidence_sha256': '5' * 64,
        'qualification_probe_manifest_sha256': '6' * 64,
        'qualification_runtime_closure_manifest_sha256': '7' * 64,
        'qualification_runtime_closure_receipt_sha256': '8' * 64,
        'qualification_runtime_closure_sha256': '9' * 64,
        'qualification_collector_public_key_hex': qualification_collector_public_key_hex,
        'qualification_collector_key_id': drill_module.firecracker_live_collector_key_id(
            bytes.fromhex(qualification_collector_public_key_hex)
        ),
        'qualification_verifier_source_sha256': 'a' * 64,
        'task_sha256': 'f' * 64,
        'provider_child_executable_sha256': 'b' * 64,
        'provider_plan_sha256': 'c' * 64,
        'collector_entrypoint_sha256': 'd' * 64,
        'collector_interpreter_sha256': 'e' * 64,
        'collector_runtime_closure_manifest_sha256': 'f' * 64,
        'collector_runtime_closure_receipt_sha256': '0' * 64,
        'collector_runtime_closure_sha256': '1' * 64,
        'collector_public_key_hex': collector_public_key_hex,
        'collector_key_id': drill_module.managed_clinical_real_kvm_collector_key_id(
            bytes.fromhex(collector_public_key_hex)
        ),
        'launcher_process_executable_sha256': 'c' * 64,
        'bootstrap_authorization_key_id': 'd' * 64,
    }
    release_sha256 = drill_module.managed_clinical_real_kvm_release_pins_sha256(**release)
    challenge_sha256 = drill_module.managed_clinical_real_kvm_challenge_sha256(
        drill_id='a' * 32,
        challenge_nonce_hex='b' * 64,
        challenge_issued_at=NOW,
        release_pins_sha256=release_sha256,
    )
    return ManagedClinicalRealKvmExternalPins(
        drill_id='a' * 32,
        challenge_nonce_hex='b' * 64,
        challenge_issued_at=NOW,
        release_pins_sha256=release_sha256,
        challenge_sha256=challenge_sha256,
        **release,
    )


def _pin_bound_evidence(pins: ManagedClinicalRealKvmExternalPins) -> SimpleNamespace:
    authority_id = drill_module.managed_clinical_real_kvm_authority_id(challenge_sha256=pins.challenge_sha256)
    return SimpleNamespace(
        drill_id=pins.drill_id,
        challenge_nonce_hex=pins.challenge_nonce_hex,
        worker_spec_sha256=pins.worker_spec_sha256,
        disk_build_receipt_sha256=pins.disk_build_receipt_sha256,
        qualification_artifact_sha256=pins.qualification_artifact_sha256,
        task_sha256=pins.task_sha256,
        provider_child_executable_sha256=pins.provider_child_executable_sha256,
        provider_plan_sha256=pins.provider_plan_sha256,
        collector_entrypoint_sha256=pins.collector_entrypoint_sha256,
        collector_interpreter_sha256=pins.collector_interpreter_sha256,
        collector_runtime_closure=SimpleNamespace(
            manifest_sha256=pins.collector_runtime_closure_manifest_sha256,
            receipt_sha256=pins.collector_runtime_closure_receipt_sha256,
            closure_sha256=pins.collector_runtime_closure_sha256,
        ),
        launcher_process_executable_sha256=pins.launcher_process_executable_sha256,
        operator_manifest=SimpleNamespace(
            execution_policy={'policy': 'fixed-test-policy'},
            guest_rpc_policy={'policy': 'fixed-test-rpc-policy'},
            submitted_harness=SimpleNamespace(baked_config_sha256=pins.guest_config_sha256),
            guest_boot_dispatch=SimpleNamespace(guest_config_sha256=pins.guest_config_sha256),
            expected_qualification_key_id=pins.qualification_key_id,
            expected_collector_evidence_sha256=(pins.qualification_collector_evidence_sha256),
            expected_probe_manifest_sha256=pins.qualification_probe_manifest_sha256,
            expected_driver_runtime_closure_manifest_sha256=(pins.qualification_runtime_closure_manifest_sha256),
            expected_driver_runtime_closure_receipt_sha256=(pins.qualification_runtime_closure_receipt_sha256),
            expected_driver_runtime_closure_sha256=(pins.qualification_runtime_closure_sha256),
            expected_collector_public_key_hex=(pins.qualification_collector_public_key_hex),
            expected_collector_key_id=pins.qualification_collector_key_id,
            expected_qualification_verifier_source_sha256=(pins.qualification_verifier_source_sha256),
            runtime_config=SimpleNamespace(bootstrap_authorization_key_id=pins.bootstrap_authorization_key_id),
        ),
        registry_config=SimpleNamespace(registry_authority_id=authority_id),
        deployment=SimpleNamespace(
            deployment_id=drill_module.managed_clinical_real_kvm_deployment_id(challenge_sha256=pins.challenge_sha256)
        ),
        reservation=SimpleNamespace(
            registered_entry_id=(
                drill_module.managed_clinical_real_kvm_registered_entry_id(challenge_sha256=pins.challenge_sha256)
            ),
            reserved_at=NOW,
        ),
        task_record=SimpleNamespace(
            launch=SimpleNamespace(claimed_at=NOW),
            start_redemption=SimpleNamespace(redeemed_at=NOW),
        ),
        ownership_chain=(SimpleNamespace(record=SimpleNamespace(recorded_at=NOW)),),
        startup_cleanups=(
            SimpleNamespace(
                authenticated_cleanup=SimpleNamespace(reconciliation_request=SimpleNamespace(requested_at=NOW))
            ),
        ),
        registry_observation=SimpleNamespace(
            record_run_audit=SimpleNamespace(audited_at=NOW),
            retry_claim_audit=SimpleNamespace(audited_at=NOW),
        ),
        live_process_observation=SimpleNamespace(observed_at=NOW),
        observation_gate_release=SimpleNamespace(
            observed_at=NOW,
            released_at=NOW,
            challenge_sha256=pins.challenge_sha256,
        ),
        collected_at=NOW,
    )


def test_complete_external_pin_set_rejects_one_mutated_input() -> None:
    pins = _external_pins()
    evidence = _pin_bound_evidence(pins)
    drill_module._verify_external_pins(cast(Any, evidence), external_pins=pins)

    evidence.provider_plan_sha256 = 'e' * 64
    with pytest.raises(ValueError, match='stable pre-execution release pins'):
        drill_module._verify_external_pins(cast(Any, evidence), external_pins=pins)


def test_release_digest_commits_every_declared_stable_input() -> None:
    pins = _external_pins()
    parameter_names = tuple(inspect.signature(drill_module.managed_clinical_real_kvm_release_pins_sha256).parameters)
    release = {name: getattr(pins, name) for name in parameter_names}
    expected = drill_module.managed_clinical_real_kvm_release_pins_sha256(**release)
    expected_challenge = drill_module.managed_clinical_real_kvm_challenge_sha256(
        drill_id=pins.drill_id,
        challenge_nonce_hex=pins.challenge_nonce_hex,
        challenge_issued_at=pins.challenge_issued_at,
        release_pins_sha256=expected,
    )

    for name in parameter_names:
        mutated = dict(release)
        mutated[name] = '0' * 64 if release[name] != '0' * 64 else 'f' * 64
        mutated_release = drill_module.managed_clinical_real_kvm_release_pins_sha256(**mutated)
        assert mutated_release != expected, name
        assert (
            drill_module.managed_clinical_real_kvm_challenge_sha256(
                drill_id=pins.drill_id,
                challenge_nonce_hex=pins.challenge_nonce_hex,
                challenge_issued_at=pins.challenge_issued_at,
                release_pins_sha256=mutated_release,
            )
            != expected_challenge
        ), name


def test_external_challenge_rejects_forged_digest_namespace_and_old_event() -> None:
    pins = _external_pins()
    forged = pins.model_dump(mode='json')
    forged['challenge_sha256'] = '0' * 64
    with pytest.raises(ValidationError, match='challenge differs from its release inputs'):
        ManagedClinicalRealKvmExternalPins.model_validate_json(drill_module.canonical_json_bytes(forged))

    evidence = _pin_bound_evidence(pins)
    evidence.registry_config.registry_authority_id = 'forged-authority'
    with pytest.raises(ValueError, match='authority namespace'):
        drill_module._verify_external_pins(cast(Any, evidence), external_pins=pins)

    evidence = _pin_bound_evidence(pins)
    evidence.reservation.reserved_at = NOW - timedelta(microseconds=1)
    with pytest.raises(ValueError, match='event from before its external challenge'):
        drill_module._verify_external_pins(cast(Any, evidence), external_pins=pins)


def test_provider_plan_requires_exact_child_argv_and_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b'{"fixture":"deterministic-provider-plan-v1"}'
    digest = hashlib.sha256(body).hexdigest()
    executable = b'#!/usr/bin/python3 -ISB\n'
    executable_digest = hashlib.sha256(executable).hexdigest()
    provider_subprocess = SimpleNamespace(
        executable_path='/opt/vaxreplay/provider-child',
        argv_suffix=(
            '--plan',
            '/opt/vaxreplay/provider-plan.json',
            '--expected-plan-sha256',
            digest,
        ),
    )
    evidence = SimpleNamespace(
        provider_plan_sha256=digest,
        provider_child_executable_sha256=executable_digest,
        operator_manifest=SimpleNamespace(provider_subprocess=provider_subprocess),
        collector_runtime_closure=SimpleNamespace(manifest=SimpleNamespace(interpreter_path='/usr/bin/python3')),
    )

    def pinned_file(path, **_kwargs):  # noqa: ANN001
        return executable if str(path) == provider_subprocess.executable_path else body

    monkeypatch.setattr(drill_module, '_read_safe_regular_file', pinned_file)

    drill_module._verify_provider_plan(cast(Any, evidence))

    provider_subprocess.argv_suffix = (
        '--plan',
        '/opt/vaxreplay/other.json',
        '--expected-plan-sha256',
        '0' * 64,
    )
    with pytest.raises(ValueError, match='exact pinned plan'):
        drill_module._verify_provider_plan(cast(Any, evidence))

    provider_subprocess.argv_suffix = (
        '--plan',
        '/opt/vaxreplay/provider-plan.json',
        '--expected-plan-sha256',
        digest,
    )
    monkeypatch.setattr(
        drill_module,
        '_read_safe_regular_file',
        lambda path, **_kwargs: executable if str(path) == provider_subprocess.executable_path else b'forged-plan',
    )
    with pytest.raises(ValueError, match='exact digest pin'):
        drill_module._verify_provider_plan(cast(Any, evidence))

    unsafe_executable = b'#!/usr/bin/python3\n'
    evidence.provider_child_executable_sha256 = hashlib.sha256(unsafe_executable).hexdigest()
    monkeypatch.setattr(
        drill_module,
        '_read_safe_regular_file',
        lambda path, **_kwargs: unsafe_executable if str(path) == provider_subprocess.executable_path else body,
    )
    with pytest.raises(ValueError, match='ambient Python startup hooks'):
        drill_module._verify_provider_plan(cast(Any, evidence))


def _process_observation(**updates: object) -> ManagedClinicalRealKvmProcessObservation:
    values: dict[str, object] = {
        'run_id': 'a' * 32,
        'ownership_envelope_sha256': 'b' * 64,
        'firecracker_pid': 1234,
        'firecracker_start_time_ticks': 5678,
        'firecracker_process_group_id': 1234,
        'firecracker_session_id': 1234,
        'firecracker_executable_sha256': 'c' * 64,
        'kvm_device_id': 10,
        'kvm_device_inode': 11,
        'kvm_device_rdev': 12,
        'firecracker_kvm_fd': 7,
        'firecracker_kvm_fd_rdev': 12,
        'proc_cgroup_path': '/vaxreplay/a',
        'cgroup_path': '/sys/fs/cgroup/vaxreplay/a',
        'cgroup_device_id': 13,
        'cgroup_inode': 14,
        'firecracker_pid_file_path': '/var/lib/vaxreplay/a/firecracker.pid',
        'firecracker_pid_file_device_id': 15,
        'firecracker_pid_file_inode': 16,
        'firecracker_pid_file_owner_uid': 0,
        'firecracker_pid_file_mode': 0o600,
        'observed_at': NOW,
    }
    values.update(updates)
    return ManagedClinicalRealKvmProcessObservation.model_validate(values)


def test_live_process_observation_binds_firecracker_fd_to_kvm_and_cgroup_mount() -> None:
    assert _process_observation().firecracker_kvm_fd == 7
    assert drill_module._proc_cgroup_path_for('/sys/fs/cgroup/vaxreplay/a') == '/vaxreplay/a'

    with pytest.raises(ValidationError, match='not the pinned KVM device'):
        _process_observation(firecracker_kvm_fd_rdev=99)
    with pytest.raises(ValueError, match='outside the cgroup-v2 mount'):
        drill_module._proc_cgroup_path_for('/var/lib/vaxreplay/a')


def test_observation_gate_release_requires_observation_before_release() -> None:
    values: dict[str, object] = {
        'drill_id': 'a' * 32,
        'challenge_nonce_hex': 'b' * 64,
        'challenge_sha256': 'c' * 64,
        'run_id': 'd' * 32,
        'ownership_envelope_sha256': 'e' * 64,
        'live_process_observation_sha256': 'f' * 64,
        'gate_binding_token_hex': '0' * 64,
        'observed_at': NOW,
        'released_at': NOW,
        'persisted_path': '/var/lib/vaxreplay/observation-gate/release.json',
    }
    assert ManagedClinicalRealKvmObservationGateRelease.model_validate(values).file_mode == 0o600

    values['released_at'] = NOW - timedelta(microseconds=1)
    with pytest.raises(ValidationError, match='released before observation'):
        ManagedClinicalRealKvmObservationGateRelease.model_validate(values)


def test_observation_gate_verifier_reloads_receipt_and_precommitted_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = _external_pins()
    release = ManagedClinicalRealKvmObservationGateRelease(
        drill_id=pins.drill_id,
        challenge_nonce_hex=pins.challenge_nonce_hex,
        challenge_sha256=pins.challenge_sha256,
        run_id='d' * 32,
        ownership_envelope_sha256='e' * 64,
        live_process_observation_sha256='f' * 64,
        gate_binding_token_hex='0' * 64,
        observed_at=NOW,
        released_at=NOW,
        persisted_path='/var/lib/vaxreplay/observation-gate/release.json',
    )
    plan_path = '/var/lib/vaxreplay/observation-gate/provider-plan.json'
    gate = {
        'binding_token_sha256': hashlib.sha256(bytes.fromhex(release.gate_binding_token_hex)).hexdigest(),
        'challenge_nonce_hex': pins.challenge_nonce_hex,
        'drill_id': pins.drill_id,
        'path': release.persisted_path,
        'provider_call_index': 0,
        'timeout_seconds': 10,
    }
    plan = {'observation_gate': gate}
    evidence = SimpleNamespace(
        drill_id=pins.drill_id,
        challenge_nonce_hex=pins.challenge_nonce_hex,
        observation_gate_release=release,
        operator_manifest=SimpleNamespace(
            provider_subprocess=SimpleNamespace(
                argv_suffix=(
                    '--plan',
                    plan_path,
                    '--expected-plan-sha256',
                    pins.provider_plan_sha256,
                )
            )
        ),
    )
    release_body = drill_module.canonical_json_bytes(release)
    plan_bodies = [drill_module.canonical_json_bytes(plan)]

    def read_safe(path: Path, **_kwargs: object) -> bytes:
        return release_body if str(path) == release.persisted_path else plan_bodies[0]

    def fake_lstat(path: Path):  # noqa: ANN202
        is_release = str(path) == release.persisted_path
        return SimpleNamespace(
            st_uid=0,
            st_nlink=1,
            st_mode=(stat.S_IFREG | 0o600) if is_release else (stat.S_IFDIR | 0o700),
        )

    monkeypatch.setattr(drill_module, '_read_safe_regular_file', read_safe)
    monkeypatch.setattr(Path, 'lstat', fake_lstat)
    drill_module._verify_observation_gate(cast(Any, evidence), external_pins=pins)

    forged = {'observation_gate': {**gate, 'binding_token_sha256': '1' * 64}}
    plan_bodies[0] = drill_module.canonical_json_bytes(forged)
    with pytest.raises(ValueError, match='differs from its precommitted plan'):
        drill_module._verify_observation_gate(cast(Any, evidence), external_pins=pins)


def test_safe_pinned_reader_rejects_in_place_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path('/var/lib/vaxreplay/pinned.json')

    def metadata(*, size: int, ctime: int) -> SimpleNamespace:
        return SimpleNamespace(
            st_dev=1,
            st_ino=2,
            st_mode=stat.S_IFREG | 0o600,
            st_uid=0,
            st_gid=0,
            st_nlink=1,
            st_size=size,
            st_mtime_ns=3,
            st_ctime_ns=ctime,
        )

    before = metadata(size=2, ctime=4)
    mutated = metadata(size=2, ctime=5)
    lstat_values = [before, mutated]
    fstat_values = [before, mutated]
    read_values = [b'{}', b'']
    monkeypatch.setattr(Path, 'is_symlink', lambda _path: False)
    monkeypatch.setattr(Path, 'resolve', lambda _path, *, strict: path)
    monkeypatch.setattr(Path, 'lstat', lambda _path: lstat_values.pop(0))
    monkeypatch.setattr(os, 'open', lambda *_args, **_kwargs: 42)
    monkeypatch.setattr(os, 'fstat', lambda _descriptor: fstat_values.pop(0))
    monkeypatch.setattr(os, 'read', lambda _descriptor, _size: read_values.pop(0))
    monkeypatch.setattr(os, 'close', lambda _descriptor: None)

    with pytest.raises(ValueError, match='changed while reading'):
        drill_module._read_safe_regular_file(path, maximum_bytes=64)


def test_managed_entrypoint_stdout_is_rederived_from_retained_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        'attempt_consumed': True,
        'episode_id': 'episode-1',
        'evidence_sha256': 'a' * 64,
        'leaderboard_admitted': False,
        'live_deployment_qualification_claimed': False,
        'managed_one_host_authority': True,
        'reservation_sha256': 'b' * 64,
        'retry_permitted': False,
        'run_id': 'c' * 32,
        'status': 'succeeded',
    }
    bodies = [drill_module.canonical_json_bytes(receipt) + b'\n']
    evidence = SimpleNamespace(
        managed_entrypoint_stdout_path='/var/lib/vaxreplay/first.stdout',
        managed_entrypoint_stdout_sha256=hashlib.sha256(bodies[0]).hexdigest(),
        reservation_sha256='b' * 64,
        run_id='c' * 32,
        task_record=SimpleNamespace(episode_id='episode-1', evidence_sha256='a' * 64),
    )
    monkeypatch.setattr(
        drill_module,
        '_read_safe_regular_file',
        lambda *_args, **_kwargs: bodies[0],
    )
    drill_module._verify_managed_entrypoint_stdout(cast(Any, evidence))

    forged = {**receipt, 'run_id': 'd' * 32}
    bodies[0] = drill_module.canonical_json_bytes(forged) + b'\n'
    evidence.managed_entrypoint_stdout_sha256 = hashlib.sha256(bodies[0]).hexdigest()
    with pytest.raises(ValueError, match='authenticated attempt'):
        drill_module._verify_managed_entrypoint_stdout(cast(Any, evidence))


def test_independent_verifier_api_is_public() -> None:
    assert {
        'ManagedClinicalRealKvmExternalPins',
        'ManagedClinicalRealKvmObservationGateRelease',
        'ManagedClinicalRealKvmVerifierKeys',
        'independently_verify_authenticated_managed_clinical_real_kvm_drill',
        'managed_clinical_real_kvm_challenge_sha256',
        'managed_clinical_real_kvm_release_pins_sha256',
        'verify_managed_clinical_real_kvm_drill_from_persisted_state',
    }.issubset(drill_module.__all__)
