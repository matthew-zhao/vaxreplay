from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.test_operations_campaign_publication import _publication
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations import campaign_publication_cli, release_readiness_cli
from vaxreplay.operations.campaign_publication_cli import (
    ArtifactPathEntry,
    PublicationArtifactPathMap,
)
from vaxreplay.operations.campaign_publication_cli import (
    main as campaign_publication_main,
)
from vaxreplay.operations.checkpoint_gossip_cli import main as gossip_main
from vaxreplay.operations.clock_health import ClockHealthObservation, ClockHealthPolicy
from vaxreplay.operations.operator_trust import (
    IsolatedProcessConfig,
    OperatorTrustError,
    load_clock_health_gate,
    load_external_signer,
    load_isolated_process_config,
)
from vaxreplay.operations.witness_service_cli import main as witness_main

UTC = timezone.utc
T0 = datetime(2026, 7, 14, 20, 0, tzinfo=UTC)


def _write(path: Path, payload: bytes, *, mode: int = 0o644) -> Path:
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def _program(tmp_path: Path, name: str, body: str) -> Path:
    payload = f'#!{sys.executable}\n{body}\n'.encode()
    return _write(tmp_path / name, payload, mode=0o700)


def _config(tmp_path: Path, executable: Path, *, timeout: float = 2.0, stdout: int = 4096) -> tuple[Path, str]:
    executable_bytes = executable.read_bytes()
    config = IsolatedProcessConfig(
        process_id=f'{executable.stem}-fixture',
        argv=(str(executable),),
        executable_sha256=hashlib.sha256(executable_bytes).hexdigest(),
        executable_byte_count=len(executable_bytes),
        timeout_seconds=timeout,
        max_stdout_bytes=stdout,
        max_stderr_bytes=1024,
    )
    payload = canonical_json_bytes(config)
    path = _write(tmp_path / f'{executable.stem}-process.json', payload)
    return path, hashlib.sha256(payload).hexdigest()


def _signer_program(
    tmp_path: Path,
    key: Ed25519PrivateKey,
    *,
    behavior: str = 'valid',
    name: str | None = None,
) -> Path:
    private_bytes = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    if behavior == 'stderr-secret':
        action = "sys.stderr.write('CHILD-SECRET-MUST-NOT-ESCAPE'); raise SystemExit(7)"
    elif behavior == 'oversize':
        action = "sys.stdout.buffer.write(b'x' * 4096)"
    elif behavior == 'sleep':
        action = 'time.sleep(5)'
    else:
        action = f"""
request = json.loads(raw)
message = base64.b64decode(request['message_base64'], validate=True)
key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex('{private_bytes.hex()}'))
signature = key.sign(message)
response = {{
    'operation':'ed25519_sign',
    'schema_version':'vaxreplay.external-signer-response.v0.1',
    'signature_base64':base64.b64encode(signature).decode('ascii'),
}}
sys.stdout.buffer.write(json.dumps(response, sort_keys=True, separators=(',', ':')).encode('ascii'))
"""
    return _program(
        tmp_path,
        name or f'signer-{behavior}.py',
        """import base64
import json
import sys
import time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
raw = sys.stdin.buffer.read()
"""
        + action,
    )


def _clock_program(
    tmp_path: Path,
    observation: ClockHealthObservation,
    *,
    name: str = 'clock.py',
) -> Path:
    response = canonical_json_bytes(observation)
    return _program(
        tmp_path,
        name,
        f"""import json
import sys
request = json.loads(sys.stdin.buffer.read())
assert request['operation'] == 'observe_clock_health'
sys.stdout.buffer.write({response!r})""",
    )


def _clock_policy(provider_id: str = 'clock-provider-a') -> ClockHealthPolicy:
    return ClockHealthPolicy(
        policy_id='tier-a-clock-policy',
        provider_id=provider_id,
        max_observation_age_seconds=5,
        max_absolute_offset_milliseconds=5,
        max_root_distance_milliseconds=20,
        max_sample_age_milliseconds=1000,
        minimum_source_count=2,
    )


def _observation(provider_id: str = 'clock-provider-a') -> ClockHealthObservation:
    return ClockHealthObservation(
        provider_id=provider_id,
        checked_at=T0,
        synchronized=True,
        leap_status='normal',
        source_count=3,
        absolute_offset_milliseconds=0.2,
        root_distance_milliseconds=2.0,
        sample_age_milliseconds=50,
    )


def test_external_signer_uses_exact_canonical_ipc_and_revalidates_executable(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    executable = _signer_program(tmp_path, key)
    config_path, config_sha256 = _config(tmp_path, executable)
    public_key = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_path = _write(tmp_path / 'signer.pub', public_key)
    signer = load_external_signer(
        process_config=config_path,
        process_config_sha256=config_sha256,
        public_key=public_path,
    )
    message = b'canonical isolated signer request'
    assert signer.sign(message) == key.sign(message)

    executable.write_bytes(executable.read_bytes() + b'\n# swapped after validation\n')
    with pytest.raises(ValueError, match='isolated signer operation failed'):
        signer.sign(message)


def test_external_signer_executes_verified_copy_when_source_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_key = Ed25519PrivateKey.generate()
    replacement_key = Ed25519PrivateKey.generate()
    executable = _signer_program(tmp_path, trusted_key)
    replacement = _signer_program(tmp_path, replacement_key, name='replacement-signer.py')
    config_path, config_sha256 = _config(tmp_path, executable)
    public_path = _write(
        tmp_path / 'signer.pub',
        trusted_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
    )
    signer = load_external_signer(
        process_config=config_path,
        process_config_sha256=config_sha256,
        public_key=public_path,
    )
    real_popen = subprocess.Popen

    def replace_then_spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        executable.write_bytes(replacement.read_bytes())
        executable.chmod(0o700)
        return real_popen(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        'vaxreplay.operations.operator_trust.subprocess.Popen',
        replace_then_spawn,
    )
    message = b'execute the exact bytes that passed verification'
    assert signer.sign(message) == trusted_key.sign(message)


@pytest.mark.parametrize('mode', (0o720, 0o702))
def test_isolated_process_rejects_group_or_world_writable_executable(
    tmp_path: Path,
    mode: int,
) -> None:
    executable = _signer_program(tmp_path, Ed25519PrivateKey.generate())
    executable.chmod(mode)
    config_path, config_sha256 = _config(tmp_path, executable)

    with pytest.raises(OperatorTrustError, match='permissions are unsafe'):
        load_isolated_process_config(config_path, expected_sha256=config_sha256)


def test_process_config_digest_output_bound_deadline_and_child_logs_fail_closed(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    public_path = _write(
        tmp_path / 'signer.pub',
        key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
    )
    valid = _signer_program(tmp_path, key)
    config_path, config_sha256 = _config(tmp_path, valid)
    config_path.write_bytes(config_path.read_bytes() + b' ')
    with pytest.raises(OperatorTrustError, match='trusted digest'):
        load_isolated_process_config(config_path, expected_sha256=config_sha256)

    for behavior, timeout, stdout in (
        ('stderr-secret', 1.0, 4096),
        ('oversize', 1.0, 128),
        ('sleep', 0.1, 4096),
    ):
        executable = _signer_program(tmp_path, key, behavior=behavior)
        process_path, process_sha256 = _config(
            tmp_path,
            executable,
            timeout=timeout,
            stdout=stdout,
        )
        signer = load_external_signer(
            process_config=process_path,
            process_config_sha256=process_sha256,
            public_key=public_path,
        )
        with pytest.raises(ValueError) as caught:
            signer.sign(b'payload')
        assert str(caught.value) == 'isolated signer operation failed'
        assert 'CHILD-SECRET-MUST-NOT-ESCAPE' not in str(caught.value)


def test_clock_provider_and_policy_are_both_digest_pinned(tmp_path: Path) -> None:
    policy = _clock_policy()
    policy_bytes = canonical_json_bytes(policy)
    policy_path = _write(tmp_path / 'clock-policy.json', policy_bytes)
    executable = _clock_program(tmp_path, _observation())
    config_path, config_sha256 = _config(tmp_path, executable)
    gate = load_clock_health_gate(
        policy_path=policy_path,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        process_config=config_path,
        process_config_sha256=config_sha256,
    )
    assert gate.require_synchronized(security_time=T0).provider_id == policy.provider_id

    policy_path.write_bytes(canonical_json_bytes(policy.model_copy(update={'max_absolute_offset_milliseconds': 5000})))
    with pytest.raises(OperatorTrustError, match='trusted digest'):
        load_clock_health_gate(
            policy_path=policy_path,
            policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
            process_config=config_path,
            process_config_sha256=config_sha256,
        )


def test_witness_and_gossip_clis_never_silently_select_local_keys(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as witness_exit:
        witness_main(
            [
                'init',
                '--root',
                str(tmp_path / 'witness'),
                '--authority-id',
                'authority-a',
                '--witness-id',
                'witness-a',
                '--policy-id',
                'policy-a',
                '--trust-policy-id',
                'trust-a',
                '--endpoint-uri',
                'https://witness.invalid/v1/witness',
            ]
        )
    assert witness_exit.value.code == 2

    with pytest.raises(SystemExit) as gossip_exit:
        gossip_main(
            [
                'init',
                '--root',
                str(tmp_path / 'gossip'),
                '--policy',
                str(tmp_path / 'missing-policy.json'),
            ]
        )
    assert gossip_exit.value.code == 2


def _write_publication_fixture(tmp_path: Path) -> tuple[object, Path, Path, list[Path], Path]:
    fixture = _publication(tmp_path / 'fixture')
    signed_path = _write(tmp_path / 'signed-manifest.json', fixture.signed_manifest_bytes)
    trust_path = _write(tmp_path / 'trust-policy.json', fixture.trust_policy_bytes)
    artifact_directory = tmp_path / 'artifacts'
    artifact_directory.mkdir()
    entries = []
    for artifact_id, payload in sorted(fixture.artifacts.items()):
        filename = f'{artifact_id}.bin'
        _write(artifact_directory / filename, payload)
        entries.append(ArtifactPathEntry(artifact_id=artifact_id, path=f'artifacts/{filename}'))
    map_path = _write(
        tmp_path / 'artifact-map.json',
        canonical_json_bytes(PublicationArtifactPathMap(artifacts=tuple(entries))),
    )
    receipts = [
        _write(tmp_path / f'receipt-{index}.json', payload) for index, payload in enumerate(fixture.receipts, start=1)
    ]
    return fixture, signed_path, trust_path, receipts, map_path


def test_campaign_cli_signs_locally_and_offline_verifies_an_exact_artifact_map(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    fixture, signed_path, trust_path, receipts, map_path = _write_publication_fixture(tmp_path)
    private_key = fixture.release_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_path = _write(tmp_path / 'release.key', private_key, mode=0o600)
    manifest_path = _write(tmp_path / 'manifest.json', canonical_json_bytes(fixture.manifest))
    locally_signed = tmp_path / 'locally-signed.json'
    campaign_publication_main(
        [
            'sign-manifest',
            '--input',
            str(manifest_path),
            '--output',
            str(locally_signed),
            '--dev-signing-private-key',
            str(private_path),
        ]
    )
    capfd.readouterr()
    assert locally_signed.read_bytes() == fixture.signed_manifest_bytes

    arguments = [
        'verify',
        '--signed-manifest',
        str(signed_path),
        '--trust-policy',
        str(trust_path),
        '--expected-trust-policy-sha256',
        hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
        '--artifact-map',
        str(map_path),
        '--verified-at',
        fixture.verified_at.isoformat(),
    ]
    for receipt in receipts:
        arguments.extend(('--receipt', str(receipt)))
    campaign_publication_main(arguments)
    report = json.loads(capfd.readouterr().out)
    assert report['exact_artifact_bill_of_materials_verified'] is True
    assert report['policy_distinct_publication_organization_quorum_verified'] is True
    assert report['external_organizational_independence_cryptographically_proven'] is False


def test_campaign_cli_rejects_release_local_trust_without_expected_out_of_band_digest(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    fixture, signed_path, trust_path, receipts, map_path = _write_publication_fixture(tmp_path)
    arguments = [
        'verify',
        '--signed-manifest',
        str(signed_path),
        '--trust-policy',
        str(trust_path),
        '--expected-trust-policy-sha256',
        '0' * 64,
        '--artifact-map',
        str(map_path),
        '--verified-at',
        fixture.verified_at.isoformat(),
    ]
    for receipt in receipts:
        arguments.extend(('--receipt', str(receipt)))
    with pytest.raises(SystemExit) as caught:
        campaign_publication_main(arguments)
    assert caught.value.code == 2
    captured = capfd.readouterr()
    assert captured.out == ''
    assert json.loads(captured.err) == {
        'error': 'campaign_publication_operation_failed',
        'status': 'failed',
    }


def test_campaign_cli_reader_rejects_input_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = _write(tmp_path / 'material.json', b'{"stable":true}')
    real_fstat = os.fstat
    call_count = 0

    def unstable_fstat(descriptor: int) -> object:
        nonlocal call_count
        result = real_fstat(descriptor)
        call_count += 1
        if call_count == 2:
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_dev=result.st_dev,
                st_ino=result.st_ino,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns,
                st_ctime_ns=result.st_ctime_ns + 1,
            )
        return result

    monkeypatch.setattr(campaign_publication_cli.os, 'fstat', unstable_fstat)
    with pytest.raises(ValueError, match='changed while being read'):
        campaign_publication_cli._read_regular(material, 1024)


def test_readiness_cli_reader_rejects_input_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = _write(tmp_path / 'readiness.json', b'{"stable":true}')
    real_fstat = os.fstat
    call_count = 0

    def unstable_fstat(descriptor: int) -> object:
        nonlocal call_count
        result = real_fstat(descriptor)
        call_count += 1
        if call_count == 2:
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_dev=result.st_dev,
                st_ino=result.st_ino,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns + 1,
                st_ctime_ns=result.st_ctime_ns,
            )
        return result

    monkeypatch.setattr(release_readiness_cli.os, 'fstat', unstable_fstat)
    with pytest.raises(ValueError, match='changed while being read'):
        release_readiness_cli._read_regular(material, maximum=1024)


def test_campaign_cli_external_signing_requires_pinned_clock_and_processes(tmp_path: Path) -> None:
    fixture = _publication(tmp_path / 'fixture')
    manifest_path = _write(tmp_path / 'manifest.json', canonical_json_bytes(fixture.manifest))
    signer_executable = _signer_program(tmp_path, fixture.release_key)
    signer_config, signer_config_sha256 = _config(tmp_path, signer_executable)
    public_path = _write(
        tmp_path / 'release.pub',
        fixture.release_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
    )
    observation = _observation().model_copy(update={'checked_at': fixture.manifest.created_at})
    clock_executable = _clock_program(tmp_path, observation)
    clock_config, clock_config_sha256 = _config(tmp_path, clock_executable)
    clock_policy = _clock_policy()
    clock_policy_bytes = canonical_json_bytes(clock_policy)
    clock_policy_path = _write(tmp_path / 'clock-policy.json', clock_policy_bytes)
    output = tmp_path / 'external-signed.json'
    campaign_publication_main(
        [
            'sign-manifest',
            '--input',
            str(manifest_path),
            '--output',
            str(output),
            '--external-signer-process',
            str(signer_config),
            '--external-signer-process-sha256',
            signer_config_sha256,
            '--external-signer-public-key',
            str(public_path),
            '--clock-health-policy',
            str(clock_policy_path),
            '--clock-health-policy-sha256',
            hashlib.sha256(clock_policy_bytes).hexdigest(),
            '--clock-health-process',
            str(clock_config),
            '--clock-health-process-sha256',
            clock_config_sha256,
        ]
    )
    assert output.read_bytes() == fixture.signed_manifest_bytes


def test_private_key_fixture_is_owner_only(tmp_path: Path) -> None:
    path = _write(tmp_path / 'key', os.urandom(32), mode=0o600)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
