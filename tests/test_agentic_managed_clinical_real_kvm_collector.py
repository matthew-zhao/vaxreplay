from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import vaxreplay.agentic.managed_clinical_real_kvm_collector as collector
from vaxreplay.agentic.firecracker_qualification_probe import firecracker_live_collector_key_id
from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest
from vaxreplay.agentic.provider_adapter import ProviderAdapterDescriptor
from vaxreplay.agentic.provider_gateway import GatewayModelRoute
from vaxreplay.agentic.provider_subprocess import (
    ProviderSubprocessRequest,
    ProviderSubprocessResponse,
)


def _start_isolated_python(code: str) -> collector.RunningManagedInvocation:
    process = subprocess.Popen(
        (sys.executable, '-I', '-B', '-c', code),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd='/',
        env={
            'LANG': 'C',
            'LC_ALL': 'C',
            'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
        },
        close_fds=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    process_group_id = os.getpgid(process.pid)
    assert process_group_id == process.pid
    return collector.RunningManagedInvocation(
        process=process,
        process_group_id=process_group_id,
        stdout_drain=collector._start_bounded_pipe_drain(process.stdout, label='test stdout'),
        stderr_drain=collector._start_bounded_pipe_drain(process.stderr, label='test stderr'),
    )


def _force_cleanup(running: collector.RunningManagedInvocation) -> None:
    if collector._managed_process_group_exists(running.process_group_id):
        collector._kill_managed_process_group(
            running.process,
            process_group_id=running.process_group_id,
        )
    else:
        running.process.wait(timeout=3)
    running.stdout_drain.thread.join(5)
    running.stderr_drain.thread.join(5)


@pytest.mark.parametrize(('file_descriptor', 'label'), ((1, 'stdout'), (2, 'stderr')))
def test_bounded_managed_output_overflow_is_rejected_and_kills_process_group(
    monkeypatch: pytest.MonkeyPatch,
    file_descriptor: int,
    label: str,
) -> None:
    monkeypatch.setattr(collector, 'MAX_MANAGED_OUTPUT_BYTES', 4096)
    running = _start_isolated_python(
        'import os, time\n'
        f'file_descriptor = {file_descriptor}\n'
        'payload = b"x" * 65536\n'
        'while True:\n'
        '    try:\n'
        '        os.write(file_descriptor, payload)\n'
        '    except BrokenPipeError:\n'
        '        time.sleep(60)\n'
    )
    try:
        with pytest.raises(
            collector.ManagedClinicalRealKvmDrillError,
            match=rf'test {label} exceeded its in-memory output bound',
        ):
            collector._finish_managed_invocation(running, timeout_seconds=5)

        assert running.process.poll() is not None
        assert not collector._managed_process_group_exists(running.process_group_id)
    finally:
        _force_cleanup(running)


def test_exited_managed_leader_with_live_descendant_is_rejected_and_cleaned(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / 'child.pid'
    child_code = 'import time; time.sleep(60)'
    leader_code = (
        'import os, subprocess, sys\n'
        f'child = subprocess.Popen((sys.executable, "-I", "-B", "-c", {child_code!r}), '
        'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)\n'
        f'path = {str(child_pid_path)!r}\n'
        'descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)\n'
        'try:\n'
        '    os.write(descriptor, str(child.pid).encode("ascii"))\n'
        '    os.fsync(descriptor)\n'
        'finally:\n'
        '    os.close(descriptor)\n'
    )
    running = _start_isolated_python(leader_code)
    try:
        with pytest.raises(
            collector.ManagedClinicalRealKvmDrillError,
            match='left a descendant in its isolated process group',
        ):
            collector._finish_managed_invocation(running, timeout_seconds=5)

        child_pid = int(child_pid_path.read_text(encoding='ascii'))
        assert running.process.poll() is not None
        assert not collector._managed_process_group_exists(running.process_group_id)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail('managed descendant remained after process-group cleanup')
    finally:
        _force_cleanup(running)


def test_clean_managed_terminal_failure_reports_only_allowlisted_codes() -> None:
    reservation_sha256 = '1' * 64
    episode_id = 'execution-dev-terminal-failure'
    final = SimpleNamespace(
        reservation_context=SimpleNamespace(reservation_sha256=reservation_sha256),
        manifest=SimpleNamespace(episode_id=episode_id),
    )
    output = {
        'attempt_consumed': True,
        'episode_id': episode_id,
        'failure_code': 'runtime_bootstrap_failed',
        'leaderboard_admitted': False,
        'live_deployment_qualification_claimed': False,
        'managed_one_host_authority': True,
        'reservation_sha256': reservation_sha256,
        'retry_permitted': False,
        'run_id': '2' * 32,
        'status': 'failed',
        'terminal_code': 'worker_terminal_failure',
    }
    completed = collector.CompletedManagedInvocation(
        return_code=1,
        stdout=collector.canonical_json_bytes(output) + b'\n',
        stderr=b'',
    )

    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match=('reported a terminal failure: runtime_bootstrap_failed / worker_terminal_failure'),
    ):
        collector._parse_successful_managed_output(completed, final=final)


def test_malformed_managed_failure_output_is_not_reflected() -> None:
    completed = collector.CompletedManagedInvocation(
        return_code=1,
        stdout=(
            b'{"failure_code":"attacker-selected-detail","status":"failed","terminal_code":"worker_terminal_failure"}\n'
        ),
        stderr=b'',
    )
    final = SimpleNamespace(
        reservation_context=SimpleNamespace(reservation_sha256='1' * 64),
        manifest=SimpleNamespace(episode_id='execution-dev-terminal-failure'),
    )

    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='did not report one clean success',
    ) as raised:
        collector._parse_successful_managed_output(completed, final=final)

    assert 'attacker-selected-detail' not in str(raised.value)


def test_unhashable_managed_failure_code_is_rejected_as_malformed() -> None:
    reservation_sha256 = '1' * 64
    episode_id = 'execution-dev-terminal-failure'
    completed = collector.CompletedManagedInvocation(
        return_code=1,
        stdout=(
            collector.canonical_json_bytes(
                {
                    'attempt_consumed': True,
                    'episode_id': episode_id,
                    'failure_code': ['runtime_bootstrap_failed'],
                    'leaderboard_admitted': False,
                    'live_deployment_qualification_claimed': False,
                    'managed_one_host_authority': True,
                    'reservation_sha256': reservation_sha256,
                    'retry_permitted': False,
                    'run_id': '2' * 32,
                    'status': 'failed',
                    'terminal_code': 'worker_terminal_failure',
                }
            )
            + b'\n'
        ),
        stderr=b'',
    )
    final = SimpleNamespace(
        reservation_context=SimpleNamespace(reservation_sha256=reservation_sha256),
        manifest=SimpleNamespace(episode_id=episode_id),
    )

    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='did not report one clean success',
    ):
        collector._parse_successful_managed_output(completed, final=final)


def test_registry_audit_selection_parses_json_encoded_strict_datetimes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation_sha256 = '1' * 64
    episode_id = 'execution-dev-registry-audit'
    run_id = '2' * 32
    terminal_at = datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)
    record_payload = collector.ManagedRecordRunRequest(
        reservation_sha256=reservation_sha256,
        episode_id=episode_id,
        production_run_root=f'/var/lib/vaxreplay/evidence/{run_id}',
        terminal_at=terminal_at,
    ).model_dump(mode='json')
    retry_payload = collector.ManagedClaimRequest(
        reservation_sha256=reservation_sha256,
        episode_id=episode_id,
        run_id='3' * 32,
        claimed_at=terminal_at,
    ).model_dump(mode='json')
    assert isinstance(record_payload['terminal_at'], str)
    assert isinstance(retry_payload['claimed_at'], str)

    record_audit = SimpleNamespace(
        sequence=10,
        request=SimpleNamespace(operation='record_run', payload=record_payload),
        response=SimpleNamespace(ok=True, error_code=None),
    )
    retry_audit = SimpleNamespace(
        sequence=11,
        request=SimpleNamespace(operation='claim', payload=retry_payload),
        response=SimpleNamespace(ok=False, error_code='rejected'),
    )
    registry_config = SimpleNamespace(
        protocol_audit_root=str(tmp_path / 'registry-audit'),
        startup_cleanup_receipt_key_id='4' * 64,
    )
    final = SimpleNamespace(
        base=SimpleNamespace(registry_config=registry_config),
        reservation_context=SimpleNamespace(reservation_sha256=reservation_sha256),
    )
    task_record = SimpleNamespace(
        episode_id=episode_id,
        launch=SimpleNamespace(run_id=run_id),
    )

    monkeypatch.setattr(
        collector,
        'managed_clinical_registry_config_sha256',
        lambda config: '5' * 64,
    )
    monkeypatch.setattr(
        collector,
        'load_authenticated_managed_registry_audit_chain',
        lambda *args, **kwargs: (record_audit, retry_audit),
    )

    selected = collector._select_registry_audits(
        final,
        managed_key=b'managed-key',
        task_record=task_record,
    )

    assert selected == (record_audit, retry_audit)


def test_first_and_retry_raw_outputs_are_private_and_retained_before_contract_rejection(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    paths = SimpleNamespace(root=tmp_path)
    first = collector.CompletedManagedInvocation(
        return_code=1,
        stdout=b'not canonical managed output\n',
        stderr=b'private first diagnostic\n',
    )
    first_stdout, first_stderr = collector._persist_private_managed_invocation_output(
        paths,
        label='first',
        completed=first,
    )
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='did not report one clean success',
    ):
        collector._parse_successful_managed_output(
            first,
            final=SimpleNamespace(
                reservation_context=SimpleNamespace(reservation_sha256='1' * 64),
                manifest=SimpleNamespace(episode_id='episode-001'),
            ),
        )

    retry = collector.CompletedManagedInvocation(
        return_code=70,
        stdout=b'',
        stderr=b'private noncanonical retry diagnostic\n',
    )
    retry_stdout, retry_stderr = collector._persist_private_managed_invocation_output(
        paths,
        label='retry',
        completed=retry,
    )
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='retry process did not fail through the fixed managed denial surface',
    ):
        collector._require_retry_denial_output(retry)

    assert first_stdout.read_bytes() == first.stdout
    assert first_stderr.read_bytes() == first.stderr
    assert retry_stdout.read_bytes() == retry.stdout
    assert retry_stderr.read_bytes() == retry.stderr
    for path in (first_stdout, first_stderr, retry_stdout, retry_stderr):
        assert path.parent == tmp_path
        assert path.lstat().st_uid == os.geteuid()
        assert path.stat().st_mode & 0o777 == 0o600


def test_private_managed_output_is_create_once(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    paths = SimpleNamespace(root=tmp_path)
    completed = collector.CompletedManagedInvocation(
        return_code=1,
        stdout=b'first bytes',
        stderr=b'',
    )
    collector._persist_private_managed_invocation_output(
        paths,
        label='first',
        completed=completed,
    )

    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='create-once drill file failed and foreign or changed state was preserved',
    ):
        collector._persist_private_managed_invocation_output(
            paths,
            label='first',
            completed=collector.CompletedManagedInvocation(
                return_code=completed.return_code,
                stdout=b'replaced',
                stderr=completed.stderr,
            ),
        )

    assert (tmp_path / 'managed-entrypoint-first.stdout').read_bytes() == completed.stdout


@pytest.mark.skipif(
    sys.platform != 'linux',
    reason='macOS injects __CF_USER_TEXT_ENCODING into an otherwise exact exec environment',
)
def test_rendered_provider_child_blocks_until_canonical_gate_model_is_persisted(
    tmp_path: Path,
) -> None:
    interpreter = Path(sys.executable).resolve(strict=True)
    child_path = tmp_path / 'provider-child'
    child_body = collector.render_deterministic_provider_child(interpreter)
    child_path.write_bytes(child_body)
    child_path.chmod(0o500)
    gate_path = tmp_path / 'observation-gate.json'
    binding_token = bytes.fromhex('ab' * 32)
    drill_id = '1' * 32
    nonce = '2' * 64
    run_id = '3' * 32
    content = '{"action":"submit","submission":{"scores":[]}}'
    plan = {
        'adapter_id': collector._ADAPTER_ID,
        'adapter_version': collector._ADAPTER_VERSION,
        'logical_model_id': collector._PUBLIC_MODEL,
        'provider': collector._PUBLIC_PROVIDER,
        'provider_model_id': collector._PUBLIC_MODEL,
        'schema_version': 'vaxreplay.managed-real-kvm-provider-plan.dev-v0.1',
        'observation_gate': {
            'binding_token_sha256': hashlib.sha256(binding_token).hexdigest(),
            'challenge_nonce_hex': nonce,
            'drill_id': drill_id,
            'path': str(gate_path),
            'provider_call_index': 0,
            'timeout_seconds': collector.OBSERVATION_GATE_TIMEOUT_SECONDS,
        },
        'turns': [
            {
                'call_index': 0,
                'content': content,
                'input_tokens': 1,
                'output_tokens': 1,
                'reasoning_tokens': 0,
            }
        ],
    }
    plan_body = collector.canonical_json_bytes(plan)
    plan_sha256 = hashlib.sha256(plan_body).hexdigest()
    plan_path = tmp_path / 'provider-plan.json'
    plan_path.write_bytes(plan_body)
    plan_path.chmod(0o600)
    child_sha256 = hashlib.sha256(child_body).hexdigest()
    route = GatewayModelRoute(
        route_id='deterministic-test-route',
        logical_model_id=collector._PUBLIC_MODEL,
        provider=collector._PUBLIC_PROVIDER,
        provider_model_id=collector._PUBLIC_MODEL,
        resolved_model_id=collector._PUBLIC_MODEL,
        accepted_provider_model_ids=(collector._PUBLIC_MODEL,),
        adapter_id=collector._ADAPTER_ID,
        adapter_version=collector._ADAPTER_VERSION,
        adapter_executable_sha256=child_sha256,
        adapter_config_sha256=plan_sha256,
        endpoint_origin='https://fixture.invalid',
        endpoint_path='/never-called',
        fixed_parameters_sha256='4' * 64,
        max_context_tokens=4096,
        max_output_tokens=1024,
        input_preflight='conservative_upper_bound',
        reasoning_accounting='reported',
        provider_data_control='default',
    )
    adapter = ProviderAdapterDescriptor(
        adapter_id=collector._ADAPTER_ID,
        adapter_version=collector._ADAPTER_VERSION,
        executable_sha256=child_sha256,
        config_sha256=plan_sha256,
        provider=collector._PUBLIC_PROVIDER,
    )
    request = ProviderSubprocessRequest(
        request=AgenticModelRequest(
            run_id=run_id,
            call_index=0,
            messages=(AgenticModelMessage(role='system', content='Use only the sealed workspace.'),),
            max_output_tokens=32,
        ),
        route=route,
        adapter=adapter,
        timeout_milliseconds=10_000,
    )
    request_body = collector.canonical_json_bytes(request)
    credential_read, credential_write = os.pipe()
    process = subprocess.Popen(
        (
            str(child_path),
            '--plan',
            str(plan_path),
            '--expected-plan-sha256',
            plan_sha256,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd='/',
        env={
            'LANG': 'C',
            'LC_ALL': 'C',
            'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
            'VAXREPLAY_PROVIDER_CREDENTIAL_FD': str(credential_read),
        },
        pass_fds=(credential_read,),
        close_fds=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        process.stdin.write(request_body)
        process.stdin.close()
        time.sleep(0.05)
        assert process.poll() is None
        observed_at = datetime.now(UTC)
        release = collector.ManagedClinicalRealKvmObservationGateRelease(
            drill_id=drill_id,
            challenge_nonce_hex=nonce,
            challenge_sha256='5' * 64,
            run_id=run_id,
            ownership_envelope_sha256='6' * 64,
            live_process_observation_sha256='7' * 64,
            gate_binding_token_hex=binding_token.hex(),
            observed_at=observed_at,
            released_at=datetime.now(UTC),
            persisted_path=str(gate_path),
        )
        collector._write_create_once(
            gate_path,
            collector.canonical_json_bytes(release),
        )
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        assert process.wait(timeout=3) == 0, stderr
    finally:
        os.close(credential_read)
        os.close(credential_write)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
    assert stderr == b''
    response = ProviderSubprocessResponse.model_validate_json(stdout)
    assert response.succeeded is True
    assert response.result is not None
    assert response.result.content == content


def test_rendered_provider_child_pins_the_exact_minimal_environment() -> None:
    child = collector.render_deterministic_provider_child(Path(sys.executable).resolve(strict=True)).decode('utf-8')
    assert 'dict(os.environ) != expected_environment' in child
    assert "'LANG': 'C'" in child
    assert "'LC_ALL': 'C'" in child
    assert "'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'" in child
    assert 'CREDENTIAL_FD_NAME: credential_descriptor' in child
    assert '__CF_USER_TEXT_ENCODING' not in child


def test_partial_fixed_deployment_staging_is_scoped_and_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_parent = tmp_path / 'vaxreplay-config'
    paths = SimpleNamespace(
        config_root=config_parent / 'lane-a-managed',
        operator_secret_root=config_parent / 'lane-a-managed' / 'operator-secrets',
        managed_secret_root=config_parent / 'lane-a-managed' / 'managed-secrets',
        drill_id='8' * 32,
    )
    final = SimpleNamespace(
        base=SimpleNamespace(
            registry_config={'registry': 'pinned'},
            startup_config={'startup': 'pinned'},
            ownership_config={'ownership': 'pinned'},
        ),
        manifest={'operator': 'pinned'},
        deployment={'deployment': 'pinned'},
    )
    keys = collector.DrillKeys.generate()
    inputs = SimpleNamespace(qualification_key=b'q' * 32)
    monkeypatch.setattr(
        collector,
        '_require_root_directory_path',
        lambda path, **_kwargs: path,
    )
    original_write = collector._write_create_once
    write_count = 0

    def fail_mid_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 4:
            raise collector.ManagedClinicalRealKvmDrillError('injected staging failure')
        original_write(path, payload, mode=mode)

    monkeypatch.setattr(collector, '_write_create_once', fail_mid_write)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='injected staging failure',
    ):
        collector._write_fixed_deployment(
            final,
            paths=paths,
            keys=keys,
            inputs=inputs,
            bootstrap_seed=b'b' * 32,
        )

    staging_root = config_parent / f'.lane-a-managed.staging-{paths.drill_id}'
    assert not paths.config_root.exists()
    assert not staging_root.exists()
    assert tuple(config_parent.iterdir()) == ()


def _fixed_deployment_subject(tmp_path: Path) -> tuple[SimpleNamespace, SimpleNamespace, object, SimpleNamespace]:
    config_parent = tmp_path / 'vaxreplay-config'
    paths = SimpleNamespace(
        config_root=config_parent / 'lane-a-managed',
        operator_secret_root=config_parent / 'lane-a-managed' / 'operator-secrets',
        managed_secret_root=config_parent / 'lane-a-managed' / 'managed-secrets',
        drill_id='8' * 32,
    )
    final = SimpleNamespace(
        base=SimpleNamespace(
            registry_config={'registry': 'pinned'},
            startup_config={'startup': 'pinned'},
            ownership_config={'ownership': 'pinned'},
        ),
        manifest={'operator': 'pinned'},
        deployment={'deployment': 'pinned'},
    )
    return paths, final, collector.DrillKeys.generate(), SimpleNamespace(qualification_key=b'q' * 32)


def _write_fixed_test_subject(
    paths: SimpleNamespace,
    final: SimpleNamespace,
    keys: object,
    inputs: SimpleNamespace,
) -> tuple[int, int]:
    return collector._write_fixed_deployment(
        final,
        paths=paths,
        keys=keys,
        inputs=inputs,
        bootstrap_seed=b'b' * 32,
    )


def test_fixed_deployment_fsync_failure_removes_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, final, keys, inputs = _fixed_deployment_subject(tmp_path)
    monkeypatch.setattr(collector, '_require_root_directory_path', lambda path, **_kwargs: path)
    original_fsync = collector._fsync_directory
    staging_root = paths.config_root.parent / f'.lane-a-managed.staging-{paths.drill_id}'
    staged_root_fsync_count = 0
    injected = False

    def fail_final_staging_fsync(path: Path) -> None:
        nonlocal injected, staged_root_fsync_count
        if path == staging_root:
            staged_root_fsync_count += 1
            if staged_root_fsync_count == 6:
                injected = True
                raise OSError('injected staged directory fsync failure')
        original_fsync(path)

    monkeypatch.setattr(collector, '_fsync_directory', fail_final_staging_fsync)
    with pytest.raises(collector.ManagedClinicalRealKvmDrillError, match='could not be persisted'):
        _write_fixed_test_subject(paths, final, keys, inputs)

    assert injected is True
    assert not staging_root.exists()
    assert not paths.config_root.exists()
    assert tuple(paths.config_root.parent.iterdir()) == ()


def test_fixed_deployment_no_replace_race_preserves_target_and_removes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, final, keys, inputs = _fixed_deployment_subject(tmp_path)
    monkeypatch.setattr(collector, '_require_root_directory_path', lambda path, **_kwargs: path)
    staging_root = paths.config_root.parent / f'.lane-a-managed.staging-{paths.drill_id}'
    raced_target_identity: tuple[int, int] | None = None

    def lose_no_replace_race(source: Path, target: Path) -> None:
        nonlocal raced_target_identity
        assert source == staging_root
        assert target == paths.config_root
        target.mkdir(mode=0o700)
        (target / 'preexisting').write_bytes(b'do-not-replace-or-delete')
        metadata = target.lstat()
        raced_target_identity = metadata.st_dev, metadata.st_ino
        raise FileExistsError('injected atomic no-replace race')

    monkeypatch.setattr(collector, 'rename_directory_noreplace', lose_no_replace_race)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='could not be atomically published without replacement',
    ):
        _write_fixed_test_subject(paths, final, keys, inputs)

    assert not staging_root.exists()
    assert paths.config_root.is_dir()
    metadata = paths.config_root.lstat()
    assert (metadata.st_dev, metadata.st_ino) == raced_target_identity
    assert (paths.config_root / 'preexisting').read_bytes() == b'do-not-replace-or-delete'


def test_fixed_deployment_post_rename_validation_failure_removes_owned_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, final, keys, inputs = _fixed_deployment_subject(tmp_path)
    monkeypatch.setattr(collector, '_require_root_directory_path', lambda path, **_kwargs: path)
    original_lstat = Path.lstat
    validation_injected = False

    def invalid_once_after_publication(path: Path):  # noqa: ANN202
        nonlocal validation_injected
        metadata = original_lstat(path)
        if path == paths.config_root and not validation_injected:
            validation_injected = True
            return SimpleNamespace(
                st_mode=(metadata.st_mode & ~0o777) | 0o755,
                st_uid=0,
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
            )
        return metadata

    monkeypatch.setattr(Path, 'lstat', invalid_once_after_publication)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='atomically published fixed deployment changed identity',
    ):
        _write_fixed_test_subject(paths, final, keys, inputs)

    staging_root = paths.config_root.parent / f'.lane-a-managed.staging-{paths.drill_id}'
    assert validation_injected is True
    assert not staging_root.exists()
    assert not paths.config_root.exists()
    assert tuple(paths.config_root.parent.iterdir()) == ()


def test_fixed_deployment_staging_identity_swap_is_detected_without_deleting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, final, keys, inputs = _fixed_deployment_subject(tmp_path)
    monkeypatch.setattr(collector, '_require_root_directory_path', lambda path, **_kwargs: path)
    original_write = collector._write_create_once
    staging_root = paths.config_root.parent / f'.lane-a-managed.staging-{paths.drill_id}'
    write_count = 0

    def swap_staging_identity(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 4:
            shutil.rmtree(staging_root)
            staging_root.mkdir(mode=0o700)
            (staging_root / 'foreign-replacement').write_bytes(b'preserve-me')
            raise collector.ManagedClinicalRealKvmDrillError('injected identity swap')
        original_write(path, payload, mode=mode)

    monkeypatch.setattr(collector, '_write_create_once', swap_staging_identity)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='failed fixed-deployment staging changed identity before cleanup',
    ):
        _write_fixed_test_subject(paths, final, keys, inputs)

    assert not paths.config_root.exists()
    assert staging_root.is_dir()
    assert (staging_root / 'foreign-replacement').read_bytes() == b'preserve-me'


def _live_prepare_arguments(output_root: Path, *, challenge_issued_at: str) -> SimpleNamespace:
    qualification_public_key = b'q' * 32
    collector_private_key = Ed25519PrivateKey.from_private_bytes(b'c' * 32)
    collector_public_key = collector_private_key.public_key().public_bytes_raw()
    return SimpleNamespace(
        worker_spec=output_root.parent / 'worker.json',
        expected_worker_spec_sha256='1' * 64,
        execution_policy=output_root.parent / 'execution-policy.json',
        expected_execution_policy_sha256='2' * 64,
        guest_rpc_policy=output_root.parent / 'guest-rpc-policy.json',
        expected_guest_rpc_policy_sha256='3' * 64,
        guest_config=output_root.parent / 'guest-config.json',
        expected_guest_config_sha256='4' * 64,
        disk_build_receipt=output_root.parent / 'disk-build-receipt.json',
        expected_disk_build_receipt_sha256='5' * 64,
        task=output_root.parent / 'task.json',
        expected_task_sha256='6' * 64,
        qualification_root=output_root.parent / 'qualification',
        qualification_key_file=output_root.parent / 'qualification.key',
        expected_qualification_key_id='7' * 64,
        expected_qualification_artifact_sha256='8' * 64,
        expected_qualification_collector_evidence_sha256='9' * 64,
        expected_qualification_probe_manifest_sha256='a' * 64,
        expected_qualification_runtime_closure_manifest_sha256='b' * 64,
        expected_qualification_runtime_closure_receipt_sha256='c' * 64,
        expected_qualification_runtime_closure_sha256='d' * 64,
        expected_qualification_collector_public_key_hex=qualification_public_key.hex(),
        expected_qualification_collector_key_id=firecracker_live_collector_key_id(qualification_public_key),
        expected_qualification_verifier_source_sha256='e' * 64,
        collector_runtime_closure_root=output_root.parent / 'collector-runtime-closure',
        expected_collector_runtime_closure_manifest_sha256='f' * 64,
        expected_collector_runtime_closure_receipt_sha256='0' * 64,
        expected_collector_runtime_closure_sha256='1' * 64,
        expected_collector_entrypoint_sha256='2' * 64,
        expected_collector_interpreter_sha256='3' * 64,
        drill_id='4' * 32,
        challenge_nonce_hex='5' * 64,
        challenge_issued_at=challenge_issued_at,
        bootstrap_authorization_seed_file=output_root.parent / 'bootstrap.seed',
        collector_key_file=output_root.parent / 'collector.key',
        expected_collector_public_key_hex=collector_public_key.hex(),
        expected_collector_key_id=collector.managed_clinical_real_kvm_collector_key_id(collector_public_key),
        expected_bootstrap_authorization_key_id='6' * 64,
        output_root=output_root,
    )


def test_prepare_to_collect_is_repeatable_parseable_and_utc_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / 'prepared'
    arguments = _live_prepare_arguments(
        output_root,
        challenge_issued_at='2025-02-03T04:05:06Z',
    )
    private_key = Ed25519PrivateKey.from_private_bytes(b'c' * 32)
    closure = SimpleNamespace(
        manifest=SimpleNamespace(
            driver_entrypoint_path='/opt/vaxreplay/collector.py',
            interpreter_path='/opt/vaxreplay/python',
        )
    )
    inputs = SimpleNamespace(
        collector_runtime_closure=closure,
        guest_config=SimpleNamespace(
            trust_anchor=SimpleNamespace(
                authorization_key_id=arguments.expected_bootstrap_authorization_key_id,
            )
        ),
        task=object(),
    )

    def deterministic_plan(
        _task: object,
        *,
        observation_gate: tuple[Path, bytes, str, str] | None = None,
    ) -> bytes:
        assert observation_gate is not None
        gate_path, binding_token, drill_id, nonce = observation_gate
        return collector.canonical_json_bytes(
            {
                'binding_token_sha256': hashlib.sha256(binding_token).hexdigest(),
                'challenge_nonce_hex': nonce,
                'drill_id': drill_id,
                'gate_path': str(gate_path),
            }
        )

    monkeypatch.setattr(collector, '_require_linux_root_kvm', lambda: None)
    monkeypatch.setattr(collector, '_load_public_inputs', lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(collector, '_verify_loaded_module_runtime_binding', lambda _closure: None)
    monkeypatch.setattr(collector, '_load_collector_private_key', lambda _arguments: private_key)
    monkeypatch.setattr(collector, '_load_bootstrap_seed', lambda *_args, **_kwargs: b's' * 32)
    monkeypatch.setattr(collector, '_canonical_provider_plan', deterministic_plan)
    monkeypatch.setattr(collector, 'render_deterministic_provider_child', lambda _interpreter: b'pinned-child')
    monkeypatch.setattr(collector, '_require_root_directory_path', lambda path, **_kwargs: path)
    validated_cohort_challenges: list[str] = []

    def validate_execution_cohort(task: object, *, challenge_sha256: str) -> object:
        assert task is inputs.task
        assert not output_root.exists()
        validated_cohort_challenges.append(challenge_sha256)
        return object()

    monkeypatch.setattr(collector, '_execution_cohort', validate_execution_cohort)

    def prepare() -> tuple[dict[str, object], dict[str, bytes]]:
        receipt = collector._prepare_live(
            arguments,
            preverified_collector_runtime_closure=closure,
        )
        files = {path.name: path.read_bytes() for path in output_root.iterdir()}
        return receipt, files

    first_receipt, first_files = prepare()
    shutil.rmtree(output_root)
    arguments.challenge_issued_at = '2025-02-02T20:05:06-08:00'
    second_receipt, second_files = prepare()

    assert second_receipt == first_receipt
    assert second_files == first_files
    assert second_receipt['challenge_issued_at'] == '2025-02-03T04:05:06Z'
    for file_name, digest_field in (
        ('provider-plan.json', 'provider_plan_sha256'),
        ('provider-child', 'provider_child_sha256'),
        ('external-pins.json', 'external_pins_sha256'),
        ('collect-invocation.json', 'collect_invocation_sha256'),
        ('PREPARE-LIVE-RECEIPT.json', 'receipt_sha256'),
    ):
        assert hashlib.sha256(second_files[file_name]).hexdigest() == second_receipt[digest_field]
    invocation_bytes = second_files['collect-invocation.json']
    invocation = json.loads(invocation_bytes)
    argv = tuple(invocation['argv'])
    assert argv[:5] == (
        '/opt/vaxreplay/python',
        '-I',
        '-B',
        '/opt/vaxreplay/collector.py',
        'collect',
    )
    option_tokens = argv[5::2]
    assert all(option.startswith('--') for option in option_tokens)
    assert len(option_tokens) == len(set(option_tokens))

    parser = collector._parser()
    parsed = parser.parse_args(argv[4:])
    assert parsed.command == 'collect'
    assert parsed.challenge_issued_at == '2025-02-03T04:05:06Z'
    assert parsed.expected_release_pins_sha256 == second_receipt['release_pins_sha256']
    assert parsed.expected_challenge_sha256 == second_receipt['challenge_sha256']
    subparsers = next(
        action for action in parser._actions if getattr(action, 'choices', None) and 'collect' in action.choices
    )
    required_options = {
        action.option_strings[0] for action in subparsers.choices['collect']._actions if action.required
    }
    assert set(option_tokens) == required_options

    gate_path, binding_token = collector._observation_gate_inputs(arguments)
    provider_plan = deterministic_plan(
        inputs.task,
        observation_gate=(
            gate_path,
            binding_token,
            arguments.drill_id,
            arguments.challenge_nonce_hex,
        ),
    )
    authorization = collector._challenge_authorization(
        arguments,
        inputs=inputs,
        provider_plan=provider_plan,
        provider_child=b'pinned-child',
        collector_private_key=private_key,
        observation_gate_path=gate_path,
        observation_gate_binding_token=binding_token,
    )
    assert authorization.challenge_issued_at == datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)
    assert authorization.release_pins_sha256 == second_receipt['release_pins_sha256']
    assert authorization.challenge_sha256 == second_receipt['challenge_sha256']
    assert collector._collect_exec_argv(arguments, inputs=inputs, authorization=authorization) == argv
    assert validated_cohort_challenges == [authorization.challenge_sha256] * 2


@pytest.mark.parametrize('failure_call', (1, 2))
def test_output_drain_setup_failure_reaps_process_group_and_started_threads(
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    original_popen = collector.subprocess.Popen
    observed_processes: list[subprocess.Popen[bytes]] = []

    def start_sleeping_process(_argv, **kwargs):  # noqa: ANN001, ANN202
        process = original_popen(
            (sys.executable, '-I', '-B', '-c', 'import time; time.sleep(60)'),
            **kwargs,
        )
        observed_processes.append(process)
        return process

    original_start_drain = collector._start_bounded_pipe_drain
    observed_drains: list[collector.BoundedPipeDrain] = []
    drain_call_count = 0

    def fail_selected_drain(pipe, *, label: str):  # noqa: ANN001, ANN202
        nonlocal drain_call_count
        drain_call_count += 1
        if drain_call_count == failure_call:
            raise RuntimeError('injected drain failure')
        drain = original_start_drain(pipe, label=label)
        observed_drains.append(drain)
        return drain

    monkeypatch.setattr(collector.subprocess, 'Popen', start_sleeping_process)
    monkeypatch.setattr(collector, '_start_bounded_pipe_drain', fail_selected_drain)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='output drains could not be started',
    ):
        collector._start_managed_invocation(SimpleNamespace(), label='first')

    assert len(observed_processes) == 1
    process = observed_processes[0]
    assert process.poll() is not None
    assert not collector._managed_process_group_exists(process.pid)
    assert len(observed_drains) == failure_call - 1
    for drain in observed_drains:
        assert not drain.thread.is_alive()
        assert drain.done.is_set()


def test_startup_receipt_inventory_stops_before_materializing_a_fourth_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'startup-receipts'
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    for index in range(4):
        (root / f'{index:064x}.json').write_text('{}', encoding='utf-8')
    loader_called = False

    def unexpected_loader(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal loader_called
        loader_called = True
        raise AssertionError('receipt loader must not run after the inventory cap fails')

    monkeypatch.setattr(
        collector,
        'load_authenticated_managed_cleanup',
        unexpected_loader,
    )
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='exceeds three entries',
    ):
        collector._load_startup_receipt_inventory(SimpleNamespace(startup_receipt_root=root))
    assert loader_called is False


def test_evidence_snapshot_uses_one_global_discovered_entry_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'evidence'
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    (root / 'one').write_bytes(b'1')
    (root / 'two').write_bytes(b'2')
    monkeypatch.setattr(collector, 'MAX_EVIDENCE_SNAPSHOT_ENTRIES', 2)

    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='entry bound',
    ):
        collector._snapshot_evidence_tree(root)


def _live_path_transaction_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    preexisting_parents: bool = False,
) -> tuple[collector.DrillPaths, collector.DrillAuthorization]:
    state_parent = tmp_path / 'state'
    runtime_parent = tmp_path / 'runtime'
    gate_parent = tmp_path / 'gates'
    fixed_root = tmp_path / 'fixed-config'
    if preexisting_parents:
        for parent in (state_parent, runtime_parent, gate_parent):
            parent.mkdir(mode=0o700)
            parent.chmod(0o700)
    monkeypatch.setattr(collector, 'STATE_PARENT', state_parent)
    monkeypatch.setattr(collector, 'RUNTIME_SOCKET_PARENT', runtime_parent)
    monkeypatch.setattr(collector, 'OBSERVATION_GATE_PARENT', gate_parent)
    monkeypatch.setattr(collector, 'FIXED_CONFIG_ROOT', fixed_root)
    monkeypatch.setattr(collector, '_require_root_directory_path', lambda path, **_kwargs: path)
    # Pytest's macOS temporary root is itself longer than Linux's entire
    # sockaddr_un pathname budget.  These tests exercise transactional path
    # ownership, while the real encoded-length guard has dedicated tests below.
    monkeypatch.setattr(collector, '_require_linux_pathname_socket_path', lambda path: path)

    def local_pinned_read(
        path: Path,
        *,
        expected_sha256: str,
        maximum_bytes: int = collector.MAX_INPUT_BYTES,
        require_root_owner: bool,
    ) -> bytes:
        del require_root_owner
        body = path.read_bytes()
        assert len(body) <= maximum_bytes
        assert hashlib.sha256(body).hexdigest() == expected_sha256
        return body

    monkeypatch.setattr(collector, '_read_pinned_file', local_pinned_read)
    drill_id = '1' * 32
    challenge_sha256 = '2' * 64
    paths = collector.DrillPaths.live(drill_id, challenge_sha256)
    authorization = collector.DrillAuthorization(
        drill_id=drill_id,
        challenge_nonce_hex='3' * 64,
        challenge_issued_at=datetime.fromisoformat('2025-02-02T20:05:06-08:00'),
        release_pins_sha256='4' * 64,
        challenge_sha256=challenge_sha256,
        registry_authority_id='test-authority',
        deployment_id='test-deployment',
        registered_entry_id='test-entry',
        external_pins=SimpleNamespace(),  # type: ignore[arg-type]
        observation_gate_path=gate_parent / (f'{drill_id}-' + '3' * 64 + '.json'),
        observation_gate_binding_token=bytes.fromhex('5' * 64),
    )
    return paths, authorization


def test_exact_v4_ids_keep_full_state_namespace_and_fit_linux_af_unix_path() -> None:
    drill_id = '934cadac074a85514ec75a1fe61a26d1'
    challenge_sha256 = 'b0d3055e0019d213d1dce77b45142e685c2aaf6a2209c57b1a6af84024ee29a7'
    namespace = f'{drill_id}-{challenge_sha256[:32]}'

    paths = collector.DrillPaths.live(drill_id, challenge_sha256)

    assert paths.root == collector.STATE_PARENT / namespace
    assert paths.registry_socket == collector.RUNTIME_SOCKET_PARENT / f'{namespace}.sock'
    assert paths.registry_socket.parent == collector.RUNTIME_SOCKET_PARENT
    assert len(os.fsencode(paths.registry_socket)) <= collector.LINUX_AF_UNIX_PATHNAME_MAX_BYTES
    old_socket = Path('/run/vaxreplay-managed-real-kvm') / namespace / 'attempts.sock'
    assert len(os.fsencode(old_socket)) > collector.LINUX_AF_UNIX_PATHNAME_MAX_BYTES


def test_linux_af_unix_path_guard_counts_encoded_bytes() -> None:
    exact_limit = Path('/') / ('x' * (collector.LINUX_AF_UNIX_PATHNAME_MAX_BYTES - 1))
    too_long = Path('/') / ('x' * collector.LINUX_AF_UNIX_PATHNAME_MAX_BYTES)

    assert len(os.fsencode(exact_limit)) == collector.LINUX_AF_UNIX_PATHNAME_MAX_BYTES
    assert collector._require_linux_pathname_socket_path(exact_limit) == exact_limit
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='AF_UNIX pathname exceeds the Linux 107-byte limit',
    ):
        collector._require_linux_pathname_socket_path(too_long)


def test_pre_reservation_composition_rejects_long_socket_before_workspace_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_started = False

    def unexpected_workspace_build(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal workspace_started
        workspace_started = True
        raise AssertionError('workspace mutation must not start for an unbindable AF_UNIX path')

    monkeypatch.setattr(
        collector,
        'build_clinical_agentic_workspace',
        unexpected_workspace_build,
    )
    paths = SimpleNamespace(
        registry_socket=Path('/') / ('x' * collector.LINUX_AF_UNIX_PATHNAME_MAX_BYTES),
        workspace_root=tmp_path / 'workspace',
    )

    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='AF_UNIX pathname exceeds the Linux 107-byte limit',
    ):
        collector._build_pre_reservation_composition(
            SimpleNamespace(),
            inputs=SimpleNamespace(),
            paths=paths,  # type: ignore[arg-type]
            keys=collector.DrillKeys.generate(),
            authorization=SimpleNamespace(),  # type: ignore[arg-type]
            provider_plan=b'{}',
            provider_child=b'#!/bin/false\n',
        )

    assert workspace_started is False
    assert not paths.workspace_root.exists()


def _assert_no_owned_prelaunch_artifacts(
    paths: collector.DrillPaths,
    authorization: collector.DrillAuthorization,
) -> None:
    assert not collector.STATE_PARENT.exists()
    assert not collector.RUNTIME_SOCKET_PARENT.exists()
    assert not collector.OBSERVATION_GATE_PARENT.exists()
    assert not paths.root.exists()
    assert not paths.registry_socket.exists()
    assert not authorization.observation_gate_path.exists()
    assert not collector.FIXED_CONFIG_ROOT.exists()


def test_live_path_authorization_receipt_is_canonical_utc_before_any_live_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)

    collector._initialize_live_paths(paths, authorization=authorization)

    body = paths.authorization_receipt.read_bytes()
    value = json.loads(body)
    loaded = collector.ManagedClinicalRealKvmChallengeAuthorizationReceipt.model_validate_json(body)
    assert collector.canonical_json_bytes(loaded) == body
    assert value['challenge_issued_at'] == '2025-02-03T04:05:06Z'
    assert value['authenticated'] is False
    assert value['vm_started'] is False
    assert value['registry_started'] is False
    assert value['provider_started'] is False


def test_live_path_initialization_accepts_tmpfiles_recreated_static_runtime_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authorization = _live_path_transaction_subject(
        tmp_path,
        monkeypatch,
        preexisting_parents=True,
    )
    runtime_parent_identity = (
        collector.RUNTIME_SOCKET_PARENT.lstat().st_dev,
        collector.RUNTIME_SOCKET_PARENT.lstat().st_ino,
    )

    collector._initialize_live_paths(paths, authorization=authorization)

    observed = collector.RUNTIME_SOCKET_PARENT.lstat()
    assert (observed.st_dev, observed.st_ino) == runtime_parent_identity
    assert paths.registry_socket.parent == collector.RUNTIME_SOCKET_PARENT
    assert not paths.registry_socket.exists()
    assert paths.authorization_receipt.is_file()


@pytest.mark.parametrize('failure_call', range(1, 13))
def test_live_path_initialization_rolls_back_after_every_directory_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)
    original_create = collector._create_root_directory
    call_count = 0

    def fail_selected_create(path: Path, *, mode: int = 0o700):
        nonlocal call_count
        call_count += 1
        if call_count == failure_call:
            raise OSError('injected directory-stage failure')
        return original_create(path, mode=mode)

    monkeypatch.setattr(collector, '_create_root_directory', fail_selected_create)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match=r'unauthenticated prelaunch failure; stage=.*; cleanup=complete',
    ) as caught:
        collector._initialize_live_paths(paths, authorization=authorization)

    assert call_count == failure_call
    assert authorization.challenge_nonce_hex not in str(caught.value)
    assert authorization.challenge_sha256 not in str(caught.value)
    _assert_no_owned_prelaunch_artifacts(paths, authorization)


@pytest.mark.parametrize('failure_stage', ('write', 'reload'))
def test_live_path_initialization_rolls_back_receipt_stage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)
    if failure_stage == 'write':
        monkeypatch.setattr(
            collector,
            '_write_create_once',
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('injected receipt write failure')),
        )
    else:
        monkeypatch.setattr(
            collector,
            '_read_pinned_file',
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('injected receipt reload failure')),
        )

    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match=r'unauthenticated prelaunch failure; stage=authorization-receipt-.*; cleanup=complete',
    ):
        collector._initialize_live_paths(paths, authorization=authorization)

    _assert_no_owned_prelaunch_artifacts(paths, authorization)


@pytest.mark.parametrize('boundary', ('write', 'file_fsync', 'parent_fsync'))
def test_create_once_failure_removes_only_its_exact_file_and_fsyncs_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    path = tmp_path / 'receipt.json'
    if boundary == 'write':
        original_write = collector.os.write
        failed = False

        def fail_after_write(descriptor: int, body: bytes) -> int:
            nonlocal failed
            count = original_write(descriptor, body)
            if not failed:
                failed = True
                raise OSError('injected write boundary failure')
            return count

        monkeypatch.setattr(collector.os, 'write', fail_after_write)
    elif boundary == 'file_fsync':
        original_fsync = collector.os.fsync
        failed = False

        def fail_after_file_fsync(descriptor: int) -> None:
            nonlocal failed
            original_fsync(descriptor)
            if not failed:
                failed = True
                raise OSError('injected file fsync boundary failure')

        monkeypatch.setattr(collector.os, 'fsync', fail_after_file_fsync)
    else:
        original_directory_fsync = collector._fsync_directory
        failed = False

        def fail_after_parent_fsync(parent: Path) -> None:
            nonlocal failed
            original_directory_fsync(parent)
            if not failed:
                failed = True
                raise OSError('injected parent fsync boundary failure')

        monkeypatch.setattr(collector, '_fsync_directory', fail_after_parent_fsync)

    with pytest.raises(collector.ManagedClinicalRealKvmDrillError, match='could not be persisted'):
        collector._write_create_once(path, b'{}')

    assert not path.exists()


@pytest.mark.parametrize(
    'boundary',
    (
        'after_mkdir',
        'after_open_fstat',
        'after_fchmod',
        'after_directory_fsync',
        'before_parent_fsync',
        'after_parent_fsync',
        'after_postvalidation',
    ),
)
def test_create_root_directory_rolls_back_every_internal_mutation_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    path = tmp_path / 'new-directory'
    monkeypatch.setattr(collector, '_require_root_directory_path', lambda value, **_kwargs: value)
    original_open = collector.os.open
    original_fchmod = collector.os.fchmod
    original_fsync = collector.os.fsync
    original_directory_fsync = collector._fsync_directory
    original_postvalidate = collector._postvalidate_created_directory
    open_injected = False
    fchmod_injected = False
    fsync_injected = False
    postvalidate_injected = False
    directory_fsync_calls = 0

    def maybe_fail_open(value, flags, *args):  # noqa: ANN001, ANN202
        nonlocal open_injected
        if boundary == 'after_mkdir' and Path(value) == path and not open_injected:
            open_injected = True
            raise OSError('injected after mkdir')
        return original_open(value, flags, *args)

    def maybe_fail_fchmod(descriptor: int, mode: int) -> None:
        nonlocal fchmod_injected
        if boundary == 'after_open_fstat' and not fchmod_injected:
            fchmod_injected = True
            raise OSError('injected after open/fstat')
        original_fchmod(descriptor, mode)

    def maybe_fail_fsync(descriptor: int) -> None:
        nonlocal fsync_injected
        if boundary == 'after_fchmod' and not fsync_injected:
            fsync_injected = True
            raise OSError('injected after fchmod')
        original_fsync(descriptor)
        if boundary == 'after_directory_fsync' and not fsync_injected:
            fsync_injected = True
            raise OSError('injected after directory fsync')

    def maybe_fail_directory_fsync(parent: Path) -> None:
        nonlocal directory_fsync_calls
        directory_fsync_calls += 1
        if boundary == 'before_parent_fsync' and directory_fsync_calls == 1:
            raise OSError('injected before parent fsync')
        original_directory_fsync(parent)
        if boundary == 'after_parent_fsync' and directory_fsync_calls == 1:
            raise OSError('injected after parent fsync')

    def maybe_fail_postvalidate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal postvalidate_injected
        result = original_postvalidate(*args, **kwargs)
        if boundary == 'after_postvalidation' and not postvalidate_injected:
            postvalidate_injected = True
            raise OSError('injected after postvalidation')
        return result

    monkeypatch.setattr(collector.os, 'open', maybe_fail_open)
    monkeypatch.setattr(collector.os, 'fchmod', maybe_fail_fchmod)
    monkeypatch.setattr(collector.os, 'fsync', maybe_fail_fsync)
    monkeypatch.setattr(collector, '_fsync_directory', maybe_fail_directory_fsync)
    monkeypatch.setattr(collector, '_postvalidate_created_directory', maybe_fail_postvalidate)

    with pytest.raises(collector.ManagedClinicalRealKvmDrillError):
        collector._create_root_directory(path)

    assert not path.exists()
    assert directory_fsync_calls >= 1


def test_first_shared_parent_swap_between_mkdir_and_open_is_preserved_and_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)
    original_open = collector.os.open
    displaced = tmp_path / 'displaced-first-parent'
    sentinel = collector.STATE_PARENT / 'foreign-sentinel'
    swapped = False

    def swap_on_first_parent_open(value, flags, *args):  # noqa: ANN001, ANN202
        nonlocal swapped
        if Path(value) == collector.STATE_PARENT and not swapped:
            swapped = True
            collector.STATE_PARENT.rename(displaced)
            collector.STATE_PARENT.mkdir(mode=0o700)
            collector.STATE_PARENT.chmod(0o700)
            sentinel.write_text('foreign', encoding='utf-8')
        return original_open(value, flags, *args)

    monkeypatch.setattr(collector.os, 'open', swap_on_first_parent_open)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='cleanup=incomplete-foreign-state-preserved',
    ):
        collector._initialize_live_paths(paths, authorization=authorization)

    assert swapped is True
    assert sentinel.read_text(encoding='utf-8') == 'foreign'
    assert displaced.is_dir()
    assert not collector.RUNTIME_SOCKET_PARENT.exists()
    assert not collector.OBSERVATION_GATE_PARENT.exists()


def test_first_shared_parent_mkdir_then_raise_is_preserved_and_never_reported_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)
    original_mkdir = collector.os.mkdir
    injected = False

    def mkdir_then_raise(value, mode=0o777, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        nonlocal injected
        result = original_mkdir(value, mode, *args, **kwargs)
        if Path(value) == collector.STATE_PARENT and not injected:
            injected = True
            raise OSError('injected after successful mkdir')
        return result

    monkeypatch.setattr(collector.os, 'mkdir', mkdir_then_raise)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='cleanup=incomplete-foreign-state-preserved',
    ):
        collector._initialize_live_paths(paths, authorization=authorization)

    assert injected is True
    assert collector.STATE_PARENT.is_dir()
    assert not collector.RUNTIME_SOCKET_PARENT.exists()
    assert not collector.OBSERVATION_GATE_PARENT.exists()


def test_first_shared_parent_cleanup_fsync_failure_is_reported_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)
    original_directory_fsync = collector._fsync_directory

    def fail_parent_fsync(parent: Path) -> None:
        if parent == tmp_path:
            raise OSError('injected parent durability failure')
        original_directory_fsync(parent)

    monkeypatch.setattr(collector, '_fsync_directory', fail_parent_fsync)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='cleanup=incomplete-foreign-state-preserved',
    ):
        collector._initialize_live_paths(paths, authorization=authorization)

    assert not collector.STATE_PARENT.exists()
    assert not collector.RUNTIME_SOCKET_PARENT.exists()
    assert not collector.OBSERVATION_GATE_PARENT.exists()


@pytest.mark.parametrize('occupied_kind', ('fixed', 'gate', 'state', 'socket'))
def test_live_path_preflight_preserves_every_preexisting_fixed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied_kind: str,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)
    occupied = {
        'fixed': collector.FIXED_CONFIG_ROOT,
        'gate': authorization.observation_gate_path,
        'state': paths.root,
        'socket': paths.registry_socket,
    }[occupied_kind]
    occupied.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    occupied.parent.chmod(0o700)
    occupied.mkdir(mode=0o700)
    occupied.chmod(0o700)
    sentinel = occupied / 'foreign-sentinel'
    sentinel.write_text('foreign', encoding='utf-8')
    identity = (occupied.lstat().st_dev, occupied.lstat().st_ino)

    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match=r'stage=occupancy-preflight; cleanup=complete',
    ):
        collector._initialize_live_paths(paths, authorization=authorization)

    observed = occupied.lstat()
    assert (observed.st_dev, observed.st_ino) == identity
    assert sentinel.read_text(encoding='utf-8') == 'foreign'


def test_create_once_parent_fsync_swap_preserves_foreign_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'receipt.json'
    displaced = tmp_path / 'receipt.original'
    original_directory_fsync = collector._fsync_directory
    swapped = False

    def swap_after_parent_fsync(parent: Path) -> None:
        nonlocal swapped
        original_directory_fsync(parent)
        if not swapped:
            swapped = True
            path.rename(displaced)
            path.write_bytes(b'foreign')
            path.chmod(0o600)
            raise OSError('injected receipt replacement')

    monkeypatch.setattr(collector, '_fsync_directory', swap_after_parent_fsync)
    with pytest.raises(
        collector._ManagedClinicalCleanupIncompleteError,
        match='foreign or changed state was preserved',
    ):
        collector._write_create_once(path, b'canonical')

    assert path.read_bytes() == b'foreign'
    assert displaced.read_bytes() == b'canonical'


def test_create_once_open_creates_then_raises_preserves_untracked_file_and_reports_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'receipt.json'
    original_open = collector.os.open
    original_close = collector.os.close

    def create_then_raise(value, flags, mode=0o777, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        descriptor = original_open(value, flags, mode, *args, **kwargs)
        if Path(value) == path:
            original_close(descriptor)
            raise OSError('injected after successful O_EXCL create')
        return descriptor

    monkeypatch.setattr(collector.os, 'open', create_then_raise)
    with pytest.raises(collector._ManagedClinicalCleanupIncompleteError):
        collector._write_create_once(path, b'canonical')

    assert path.read_bytes() == b''


def test_live_path_receipt_open_creates_then_raises_is_never_reported_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)
    original_open = collector.os.open
    original_close = collector.os.close
    injected = False

    def create_receipt_then_raise(value, flags, mode=0o777, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        nonlocal injected
        descriptor = original_open(value, flags, mode, *args, **kwargs)
        if Path(value) == paths.authorization_receipt and not injected:
            injected = True
            original_close(descriptor)
            raise OSError('injected after successful receipt create')
        return descriptor

    monkeypatch.setattr(collector.os, 'open', create_receipt_then_raise)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='cleanup=incomplete-foreign-state-preserved',
    ):
        collector._initialize_live_paths(paths, authorization=authorization)

    assert injected is True
    assert paths.authorization_receipt.read_bytes() == b''
    assert paths.root.is_dir()
    assert not collector.RUNTIME_SOCKET_PARENT.exists()
    assert not collector.OBSERVATION_GATE_PARENT.exists()


@pytest.mark.parametrize('kind', ('file', 'directory'))
def test_descriptor_close_failure_continues_exact_identity_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    path = tmp_path / kind
    monkeypatch.setattr(collector, '_require_root_directory_path', lambda value, **_kwargs: value)
    original_close = collector._close_descriptor_best_effort
    injected = False

    def close_then_report_failure(descriptor: int) -> bool:
        nonlocal injected
        closed = original_close(descriptor)
        if not injected:
            injected = True
            return False
        return closed

    monkeypatch.setattr(collector, '_close_descriptor_best_effort', close_then_report_failure)
    with pytest.raises(collector._ManagedClinicalCleanupIncompleteError):
        if kind == 'file':
            collector._write_create_once(path, b'canonical')
        else:
            collector._create_root_directory(path)

    assert injected is True
    assert not path.exists()


def test_live_path_rollback_never_removes_preexisting_shared_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authorization = _live_path_transaction_subject(
        tmp_path,
        monkeypatch,
        preexisting_parents=True,
    )
    identities = {
        parent: (parent.lstat().st_dev, parent.lstat().st_ino)
        for parent in (collector.STATE_PARENT, collector.RUNTIME_SOCKET_PARENT, collector.OBSERVATION_GATE_PARENT)
    }
    original_create = collector._create_root_directory

    def fail_state_root(path: Path, *, mode: int = 0o700):
        if path == paths.root:
            raise OSError('injected state-root failure')
        return original_create(path, mode=mode)

    monkeypatch.setattr(collector, '_create_root_directory', fail_state_root)
    with pytest.raises(collector.ManagedClinicalRealKvmDrillError, match='cleanup=complete'):
        collector._initialize_live_paths(paths, authorization=authorization)

    for parent, identity in identities.items():
        observed = parent.lstat()
        assert (observed.st_dev, observed.st_ino) == identity


def test_live_path_rollback_preserves_foreign_replacement_and_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, authorization = _live_path_transaction_subject(tmp_path, monkeypatch)
    original_create = collector._create_root_directory
    displaced = tmp_path / 'displaced-original-state'
    sentinel = paths.root / 'foreign-sentinel'

    def swap_before_first_state_child(path: Path, *, mode: int = 0o700):
        if path == paths.private_root:
            paths.root.rename(displaced)
            paths.root.mkdir(mode=0o700)
            paths.root.chmod(0o700)
            sentinel.write_text('foreign', encoding='utf-8')
            raise OSError('injected foreign replacement')
        return original_create(path, mode=mode)

    monkeypatch.setattr(collector, '_create_root_directory', swap_before_first_state_child)
    with pytest.raises(
        collector.ManagedClinicalRealKvmDrillError,
        match='cleanup=incomplete-foreign-state-preserved',
    ):
        collector._initialize_live_paths(paths, authorization=authorization)

    assert sentinel.read_text(encoding='utf-8') == 'foreign'
    assert displaced.is_dir()
    assert not collector.RUNTIME_SOCKET_PARENT.exists()
    assert not collector.OBSERVATION_GATE_PARENT.exists()
