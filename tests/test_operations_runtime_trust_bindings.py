from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.test_operations_selection_registry import _registry
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations import (
    checkpoint_gossip_cli,
    selection_registry_service,
    witness_service_cli,
)
from vaxreplay.operations.checkpoint_gossip import (
    CheckpointGossipError,
    CheckpointGossipMonitorPolicy,
    RegistryGossipStreamPolicy,
)
from vaxreplay.operations.checkpoint_gossip_cli import (
    _require_runtime_trust_binding as require_gossip_runtime_trust,
)
from vaxreplay.operations.clock_health import (
    CallbackClockHealthProvider,
    ClockHealthGate,
    ClockHealthObservation,
    ClockHealthPolicy,
)
from vaxreplay.operations.operator_trust import IsolatedProcessConfig
from vaxreplay.operations.selection_registry import (
    RegistryPinnedCheckpoint,
    SelectionRegistryError,
    SelectionRegistryPolicy,
    SelectionRegistryTrustPolicy,
    SignedRegistryCheckpoint,
)
from vaxreplay.operations.selection_registry_service import (
    _require_runtime_trust_binding as require_registry_runtime_trust,
)
from vaxreplay.operations.signing import LocalEd25519Signer
from vaxreplay.operations.witness_service import WitnessServiceError
from vaxreplay.operations.witness_service_cli import (
    _require_runtime_trust_binding as require_witness_runtime_trust,
)
from vaxreplay.operations.witness_service_schema import (
    WitnessRegistryMonitor,
    WitnessRegistrySigningKey,
    WitnessServicePolicy,
)

UTC = timezone.utc
RUNTIME_DIGESTS = ('1' * 64, '2' * 64, '3' * 64)
RUNTIME_FIELDS = (
    'clock_health_policy_sha256',
    'clock_health_process_sha256',
    'external_signer_process_sha256',
)


@dataclass(frozen=True)
class RuntimeTrustFiles:
    clock_policy: Path
    clock_policy_sha256: str
    clock_process: Path
    clock_process_sha256: str
    signer_process: Path
    signer_process_sha256: str
    signer_public_key: Path

    @property
    def digests(self) -> tuple[str, str, str]:
        return (
            self.clock_policy_sha256,
            self.clock_process_sha256,
            self.signer_process_sha256,
        )

    def cli_arguments(self) -> list[str]:
        return [
            '--external-signer-process',
            str(self.signer_process),
            '--external-signer-public-key',
            str(self.signer_public_key),
            '--external-signer-process-sha256',
            self.signer_process_sha256,
            '--clock-health-policy',
            str(self.clock_policy),
            '--clock-health-policy-sha256',
            self.clock_policy_sha256,
            '--clock-health-process',
            str(self.clock_process),
            '--clock-health-process-sha256',
            self.clock_process_sha256,
        ]


def _write(path: Path, payload: bytes, *, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def _program(path: Path, body: str) -> Path:
    return _write(
        path,
        f'#!{sys.executable}\n{body}\n'.encode(),
        mode=0o700,
    )


def _process_config(
    path: Path,
    executable: Path,
    *,
    process_id: str,
) -> tuple[Path, str]:
    executable_bytes = executable.read_bytes()
    config = IsolatedProcessConfig(
        process_id=process_id,
        argv=(str(executable),),
        executable_sha256=hashlib.sha256(executable_bytes).hexdigest(),
        executable_byte_count=len(executable_bytes),
        timeout_seconds=2,
        max_stdout_bytes=4096,
        max_stderr_bytes=1024,
    )
    payload = canonical_json_bytes(config)
    return _write(path, payload), hashlib.sha256(payload).hexdigest()


def _runtime_files(root: Path, key: Ed25519PrivateKey) -> RuntimeTrustFiles:
    root.mkdir(parents=True, mode=0o700)
    private_key = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    signer_executable = _program(
        root / 'signer.py',
        f"""import base64
import json
import sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

request = json.loads(sys.stdin.buffer.read())
message = base64.b64decode(request['message_base64'], validate=True)
key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex('{private_key.hex()}'))
response = {{
    'operation': 'ed25519_sign',
    'schema_version': 'vaxreplay.external-signer-response.v0.1',
    'signature_base64': base64.b64encode(key.sign(message)).decode('ascii'),
}}
sys.stdout.buffer.write(json.dumps(response, sort_keys=True, separators=(',', ':')).encode())""",
    )
    signer_process, signer_process_sha256 = _process_config(
        root / 'signer-process.json',
        signer_executable,
        process_id='runtime-binding-signer',
    )
    signer_public_key = _write(
        root / 'signer-public-key.bin',
        key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
    )

    observation = ClockHealthObservation(
        provider_id='runtime-binding-clock',
        checked_at=datetime.now(UTC),
        synchronized=True,
        leap_status='normal',
        source_count=3,
        absolute_offset_milliseconds=0.1,
        root_distance_milliseconds=0.2,
        sample_age_milliseconds=1,
    )
    clock_executable = _program(
        root / 'clock.py',
        f"""import json
import sys

request = json.loads(sys.stdin.buffer.read())
assert request['operation'] == 'observe_clock_health'
sys.stdout.buffer.write({canonical_json_bytes(observation)!r})""",
    )
    clock_process, clock_process_sha256 = _process_config(
        root / 'clock-process.json',
        clock_executable,
        process_id='runtime-binding-clock',
    )
    clock_policy = ClockHealthPolicy(
        policy_id='runtime-binding-clock-policy',
        provider_id='runtime-binding-clock',
        max_observation_age_seconds=300,
        max_absolute_offset_milliseconds=1,
        max_root_distance_milliseconds=2,
        max_sample_age_milliseconds=100,
        minimum_source_count=2,
    )
    clock_policy_bytes = canonical_json_bytes(clock_policy)
    clock_policy_path = _write(root / 'clock-policy.json', clock_policy_bytes)
    return RuntimeTrustFiles(
        clock_policy=clock_policy_path,
        clock_policy_sha256=hashlib.sha256(clock_policy_bytes).hexdigest(),
        clock_process=clock_process,
        clock_process_sha256=clock_process_sha256,
        signer_process=signer_process,
        signer_process_sha256=signer_process_sha256,
        signer_public_key=signer_public_key,
    )


def _runtime_drift(
    runtime: RuntimeTrustFiles,
    root: Path,
    field: str,
) -> RuntimeTrustFiles:
    if field == 'clock_health_policy_sha256':
        policy = ClockHealthPolicy.model_validate_json(runtime.clock_policy.read_bytes())
        payload = canonical_json_bytes(policy.model_copy(update={'policy_id': 'runtime-binding-clock-policy-drift'}))
        path = _write(root / 'clock-policy-drift.json', payload)
        return replace(
            runtime,
            clock_policy=path,
            clock_policy_sha256=hashlib.sha256(payload).hexdigest(),
        )
    process_path = runtime.clock_process if field == 'clock_health_process_sha256' else runtime.signer_process
    process = IsolatedProcessConfig.model_validate_json(process_path.read_bytes())
    payload = canonical_json_bytes(process.model_copy(update={'process_id': f'{process.process_id}-drift'}))
    path = _write(root / f'{field}-drift.json', payload)
    if field == 'clock_health_process_sha256':
        return replace(
            runtime,
            clock_process=path,
            clock_process_sha256=hashlib.sha256(payload).hexdigest(),
        )
    return replace(
        runtime,
        signer_process=path,
        signer_process_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _private_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _public_key_base64(key: Ed25519PrivateKey) -> str:
    public_key = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(public_key).decode('ascii')


def _policies() -> tuple[
    SelectionRegistryPolicy,
    WitnessServicePolicy,
    CheckpointGossipMonitorPolicy,
]:
    now = datetime.now(UTC).replace(microsecond=0)
    registry_key = Ed25519PrivateKey.generate()
    registry_monitor = WitnessRegistryMonitor(
        registry_id='runtime-binding-registry',
        authority_id='runtime-binding-registry-authority',
        signing_keys=(
            WitnessRegistrySigningKey(
                key_id='runtime-binding-registry-key',
                public_key_base64=_public_key_base64(registry_key),
                valid_from=now - timedelta(days=1),
            ),
        ),
    )
    report_key = Ed25519PrivateKey.generate()
    return (
        SelectionRegistryPolicy(
            registry_id=registry_monitor.registry_id,
            authority_id=registry_monitor.authority_id,
            policy_id='runtime-binding-registry-policy',
        ),
        WitnessServicePolicy(
            authority_id='runtime-binding-witness-authority',
            witness_id='runtime-binding-witness',
            policy_id='runtime-binding-witness-policy',
            endpoint_uri='https://runtime-binding-witness.invalid/v1/witness',
            max_submission_bytes=4096,
            max_proof_bytes=4096,
            client_timeout_seconds=5.0,
            registry_monitors=(registry_monitor,),
        ),
        CheckpointGossipMonitorPolicy(
            monitor_id='runtime-binding-monitor',
            policy_id='runtime-binding-monitor-policy',
            streams=(
                RegistryGossipStreamPolicy(
                    stream_id='runtime-binding-registry-heads',
                    registry_monitor=registry_monitor,
                    bootstrap_tree_size=0,
                    bootstrap_signed_checkpoint_sha256='4' * 64,
                ),
            ),
            max_observation_age_seconds=60,
            report_signing_key_id='runtime-binding-report-key',
            report_signing_public_key_base64=_public_key_base64(report_key),
            report_signing_key_valid_from=now - timedelta(days=1),
        ),
    )


def _with_runtime_trust(policy):
    return type(policy).model_validate(
        {
            **policy.model_dump(mode='python'),
            **dict(zip(RUNTIME_FIELDS, RUNTIME_DIGESTS, strict=True)),
        }
    )


def _args(digests: tuple[str | None, str | None, str | None]) -> argparse.Namespace:
    return argparse.Namespace(
        clock_health_policy_sha256=digests[0],
        clock_health_process_sha256=digests[1],
        external_signer_process_sha256=digests[2],
    )


def test_runtime_trust_policy_digests_are_all_present_or_all_null() -> None:
    for policy in _policies():
        assert all(getattr(policy, field) is None for field in RUNTIME_FIELDS)
        _with_runtime_trust(policy)
        for field in RUNTIME_FIELDS:
            with pytest.raises(ValueError, match='all present or all null'):
                type(policy).model_validate(
                    {
                        **policy.model_dump(mode='python'),
                        field: 'a' * 64,
                    }
                )


def test_reopen_rejects_weaker_or_mismatched_runtime_configuration() -> None:
    helpers = (
        (require_registry_runtime_trust, SelectionRegistryError),
        (require_witness_runtime_trust, WitnessServiceError),
        (require_gossip_runtime_trust, CheckpointGossipError),
    )
    for policy, (require_binding, error_type) in zip(_policies(), helpers, strict=True):
        production_policy = _with_runtime_trust(policy)
        require_binding(production_policy, _args(RUNTIME_DIGESTS))
        with pytest.raises(error_type, match='differ from the persisted'):
            require_binding(production_policy, _args((None, None, None)))
        with pytest.raises(error_type, match='differ from the persisted'):
            require_binding(
                production_policy,
                _args((RUNTIME_DIGESTS[0], 'f' * 64, RUNTIME_DIGESTS[2])),
            )


def test_external_witness_cli_persists_bindings_and_rejects_restart_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())

    def observation() -> ClockHealthObservation:
        return ClockHealthObservation(
            provider_id='runtime-binding-clock',
            checked_at=datetime.now(UTC),
            synchronized=True,
            leap_status='normal',
            source_count=2,
            absolute_offset_milliseconds=0.1,
            root_distance_milliseconds=0.2,
            sample_age_milliseconds=1,
        )

    gate = ClockHealthGate(
        policy=ClockHealthPolicy(
            policy_id='runtime-binding-clock-policy',
            provider_id='runtime-binding-clock',
            max_observation_age_seconds=5,
            max_absolute_offset_milliseconds=1,
            max_root_distance_milliseconds=2,
            max_sample_age_milliseconds=100,
            minimum_source_count=2,
        ),
        provider=CallbackClockHealthProvider(observation),
    )
    monkeypatch.setattr(witness_service_cli, '_runtime_trust', lambda _args: (signer, gate))
    root = tmp_path / 'external-witness'
    base_arguments = [
        '--external-signer-process',
        str(tmp_path / 'signer-process.json'),
        '--external-signer-public-key',
        str(tmp_path / 'signer-public-key.bin'),
        '--external-signer-process-sha256',
        RUNTIME_DIGESTS[2],
        '--clock-health-policy',
        str(tmp_path / 'clock-policy.json'),
        '--clock-health-policy-sha256',
        RUNTIME_DIGESTS[0],
        '--clock-health-process',
        str(tmp_path / 'clock-process.json'),
        '--clock-health-process-sha256',
        RUNTIME_DIGESTS[1],
    ]
    witness_service_cli.main(
        [
            'init',
            '--root',
            str(root),
            '--authority-id',
            'runtime-binding-cli-authority',
            '--witness-id',
            'runtime-binding-cli-witness',
            '--policy-id',
            'runtime-binding-cli-policy',
            '--trust-policy-id',
            'runtime-binding-cli-trust',
            '--endpoint-uri',
            'https://runtime-binding-cli.invalid/v1/witness',
            *base_arguments,
        ]
    )
    persisted = WitnessServicePolicy.model_validate_json((root / 'policy.json').read_bytes())
    assert canonical_json_bytes(persisted) == (root / 'policy.json').read_bytes()
    assert tuple(getattr(persisted, field) for field in RUNTIME_FIELDS) == RUNTIME_DIGESTS

    mismatched = list(base_arguments)
    process_digest_index = mismatched.index('--clock-health-process-sha256') + 1
    mismatched[process_digest_index] = 'f' * 64
    with pytest.raises(SystemExit) as caught:
        witness_service_cli.main(['verify', '--root', str(root), *mismatched])
    assert caught.value.code == 2


def test_selection_registry_cli_reopen_enforces_persisted_runtime_trust(
    tmp_path: Path,
) -> None:
    key = Ed25519PrivateKey.generate()
    runtime = _runtime_files(tmp_path / 'registry-runtime', key)
    fixture_root = tmp_path / 'registry-fixture'
    fixture_root.mkdir(mode=0o700)
    source_registry, _key, policy_bytes, trust_bytes = _registry(
        fixture_root,
        key=key,
        runtime_trust_digests=runtime.digests,
    )
    genesis_envelope, genesis_witness_proof = source_registry.signed_checkpoint_and_witness(0)
    genesis_checkpoint = SignedRegistryCheckpoint.model_validate_json(genesis_envelope).checkpoint
    trust = SelectionRegistryTrustPolicy.model_validate_json(trust_bytes)
    genesis_trust = trust.model_copy(
        update={
            'pinned_checkpoint': RegistryPinnedCheckpoint(
                tree_size=0,
                root_sha256=genesis_checkpoint.root_sha256,
                signed_checkpoint_base64=base64.b64encode(genesis_envelope).decode('ascii'),
                witness_proof_base64=base64.b64encode(genesis_witness_proof).decode('ascii'),
            )
        }
    )
    policy_path = _write(tmp_path / 'registry-policy.json', policy_bytes)
    trust_path = _write(
        tmp_path / 'registry-trust.json',
        canonical_json_bytes(genesis_trust),
    )
    database_path = tmp_path / 'cli-selection-registry.sqlite'
    common_arguments = [
        '--database',
        str(database_path),
        '--signing-key-id',
        'registry-key-2026',
        '--registry-policy',
        str(policy_path),
        '--trust-policy',
        str(trust_path),
        '--public-base-url',
        'https://registry.example',
    ]

    selection_registry_service.main(['init', *common_arguments, *runtime.cli_arguments()])
    selection_registry_service.main(['status', *common_arguments, *runtime.cli_arguments()])

    for field in RUNTIME_FIELDS:
        drifted = _runtime_drift(runtime, tmp_path / 'registry-runtime', field)
        with pytest.raises(
            SelectionRegistryError,
            match='differ from the persisted registry policy',
        ):
            selection_registry_service.main(['status', *common_arguments, *drifted.cli_arguments()])

    wrong_public_key = _write(
        tmp_path / 'wrong-registry-signer-public-key.bin',
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
    )
    with pytest.raises(
        SelectionRegistryError,
        match='differs from registered public key',
    ):
        selection_registry_service.main(
            [
                'status',
                *common_arguments,
                *replace(runtime, signer_public_key=wrong_public_key).cli_arguments(),
            ]
        )

    local_key_path = _write(
        tmp_path / 'registry-development-key.bin',
        _private_key_bytes(key),
        mode=0o600,
    )
    with pytest.raises(
        SelectionRegistryError,
        match='differ from the persisted registry policy',
    ):
        selection_registry_service.main(
            [
                'status',
                *common_arguments,
                '--dev-signing-private-key',
                str(local_key_path),
            ]
        )


def test_gossip_cli_reopen_enforces_persisted_runtime_trust(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key = Ed25519PrivateKey.generate()
    runtime = _runtime_files(tmp_path / 'gossip-runtime', key)
    base_policy = _policies()[2]
    policy = CheckpointGossipMonitorPolicy.model_validate(
        {
            **base_policy.model_dump(mode='python'),
            'report_signing_public_key_base64': _public_key_base64(key),
            **dict(zip(RUNTIME_FIELDS, runtime.digests, strict=True)),
        }
    )
    policy_path = _write(
        tmp_path / 'gossip-policy.json',
        canonical_json_bytes(policy),
    )
    root = tmp_path / 'gossip-monitor'
    checkpoint_gossip_cli.main(
        [
            'init',
            '--root',
            str(root),
            '--policy',
            str(policy_path),
            *runtime.cli_arguments(),
        ]
    )
    checkpoint_gossip_cli.main(['verify', '--root', str(root), *runtime.cli_arguments()])
    capsys.readouterr()

    for field in RUNTIME_FIELDS:
        drifted = _runtime_drift(runtime, tmp_path / 'gossip-runtime', field)
        with pytest.raises(SystemExit) as caught:
            checkpoint_gossip_cli.main(['verify', '--root', str(root), *drifted.cli_arguments()])
        assert caught.value.code == 2
        assert 'differ from the persisted gossip policy' in capsys.readouterr().err

    wrong_public_key = _write(
        tmp_path / 'wrong-gossip-signer-public-key.bin',
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
    )
    with pytest.raises(SystemExit) as caught:
        checkpoint_gossip_cli.main(
            [
                'verify',
                '--root',
                str(root),
                *replace(runtime, signer_public_key=wrong_public_key).cli_arguments(),
            ]
        )
    assert caught.value.code == 2
    assert 'gossip report signer is invalid' in capsys.readouterr().err

    _write(
        root / 'report-ed25519-private-key.bin',
        _private_key_bytes(key),
        mode=0o600,
    )
    with pytest.raises(SystemExit) as caught:
        checkpoint_gossip_cli.main(['verify', '--root', str(root), '--dev-local-root-key'])
    assert caught.value.code == 2
    assert 'differ from the persisted gossip policy' in capsys.readouterr().err
