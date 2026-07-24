from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.test_operations_checkpoint_gossip import (
    _monitor_policy,
    _observe_bootstraps,
)
from tests.test_operations_checkpoint_gossip import (
    sources as make_sources,
)
from tests.test_operations_selection_registry import _commitment, _registry, _request
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.checkpoint_gossip import CheckpointGossipMonitorStore
from vaxreplay.operations.clock_health import (
    CallbackClockHealthProvider,
    ClockHealthError,
    ClockHealthGate,
    ClockHealthObservation,
    ClockHealthPolicy,
)
from vaxreplay.operations.signing import (
    Ed25519SignerError,
    IsolatedEd25519Signer,
)
from vaxreplay.operations.witness import ExternalWitnessMethod
from vaxreplay.operations.witness_service import WitnessServiceError, WitnessServiceStore
from vaxreplay.operations.witness_service_schema import WitnessServiceSubmission

UTC = timezone.utc
T0 = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


@dataclass
class MutableHealth:
    observation: ClockHealthObservation

    def __call__(self) -> ClockHealthObservation:
        return self.observation


def _health(*, checked_at: datetime = T0, synchronized: bool = True) -> ClockHealthObservation:
    return ClockHealthObservation(
        provider_id='chrony-sidecar-a',
        checked_at=checked_at,
        synchronized=synchronized,
        leap_status='normal' if synchronized else 'unsynchronized',
        source_count=3,
        absolute_offset_milliseconds=0.2,
        root_distance_milliseconds=1.2,
        sample_age_milliseconds=100,
    )


def _gate(provider: MutableHealth) -> ClockHealthGate:
    return ClockHealthGate(
        policy=ClockHealthPolicy(
            policy_id='tier-a-clock-health-v1',
            provider_id='chrony-sidecar-a',
            max_observation_age_seconds=5,
            max_absolute_offset_milliseconds=5,
            max_root_distance_milliseconds=20,
            max_sample_age_milliseconds=1000,
            minimum_source_count=2,
        ),
        provider=CallbackClockHealthProvider(provider),
    )


def _raw_public_key(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def test_isolated_signer_verifies_every_broker_response_and_scrubs_failures() -> None:
    trusted_key = Ed25519PrivateKey.generate()
    wrong_key = Ed25519PrivateKey.generate()
    payload = b'domain-separated security statement'
    signer = IsolatedEd25519Signer(
        public_key=_raw_public_key(trusted_key),
        sign_operation=trusted_key.sign,
    )
    assert signer.sign(payload) == trusted_key.sign(payload)

    substituted = IsolatedEd25519Signer(
        public_key=_raw_public_key(trusted_key),
        sign_operation=wrong_key.sign,
    )
    with pytest.raises(Ed25519SignerError, match='different key'):
        substituted.sign(payload)

    secret = 'kms-bearer-secret-must-not-escape'

    def failing(_payload: bytes) -> bytes:
        raise RuntimeError(secret)

    failed = IsolatedEd25519Signer(
        public_key=_raw_public_key(trusted_key),
        sign_operation=failing,
    )
    with pytest.raises(Ed25519SignerError, match='operation failed') as caught:
        failed.sign(payload)
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ('update', 'message'),
    [
        ({'checked_at': T0 - timedelta(seconds=6)}, 'stale'),
        ({'synchronized': False, 'leap_status': 'unsynchronized'}, 'not in an allowed'),
        ({'source_count': 1}, 'too few'),
        ({'absolute_offset_milliseconds': 5.1}, 'offset'),
        ({'root_distance_milliseconds': 20.1}, 'root distance'),
        ({'sample_age_milliseconds': 1001}, 'sample is too old'),
    ],
)
def test_clock_health_gate_rejects_each_degraded_dimension(
    update: dict[str, object],
    message: str,
) -> None:
    provider = MutableHealth(_health().model_copy(update=update))
    with pytest.raises(ClockHealthError, match=message):
        _gate(provider).require_synchronized(security_time=T0)


def test_clock_health_provider_failure_does_not_retain_secret_exception() -> None:
    secret = 'chrony-rpc-credential-must-not-escape'

    def failing() -> ClockHealthObservation:
        raise RuntimeError(secret)

    gate = ClockHealthGate(
        policy=_gate(MutableHealth(_health())).policy,
        provider=CallbackClockHealthProvider(failing),
    )
    with pytest.raises(ClockHealthError, match='provider failed') as caught:
        gate.require_synchronized(security_time=T0)
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_registry_clock_health_failure_rolls_back_new_log_signature(tmp_path: Path) -> None:
    registry, _key, policy_bytes, trust_bytes = _registry(tmp_path)
    existing = registry.tree_head()
    assert existing is not None
    registry_time = existing.issued_at_upper_bound
    provider = MutableHealth(_health(checked_at=registry_time))
    registry.clock_health_gate = _gate(provider)
    first = _commitment(policy_bytes, trust_bytes, campaign_id='clock-healthy-campaign')
    registry.assign(_request(first, policy_bytes))
    head = registry.tree_head()
    assert head is not None
    provider.observation = _health(checked_at=registry_time, synchronized=False)
    second = _commitment(policy_bytes, trust_bytes, campaign_id='clock-unhealthy-campaign')
    with pytest.raises(ClockHealthError, match='not in an allowed'):
        registry.assign(_request(second, policy_bytes))
    assert registry.tree_head() == head


def test_witness_can_use_isolated_signer_without_persisting_private_key(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    isolated = IsolatedEd25519Signer(public_key=_raw_public_key(key), sign_operation=key.sign)
    provider = MutableHealth(_health())
    service = WitnessServiceStore.initialize(
        tmp_path / 'isolated-witness',
        authority_id='independent-witness-operator',
        witness_id='isolated-witness-key',
        policy_id='isolated-witness-policy',
        trust_policy_id='isolated-witness-trust',
        endpoint_uri='https://witness.invalid/v1/witness',
        signer=isolated,
        clock=lambda: T0,
        clock_health_gate=_gate(provider),
    )
    assert not (service.root / 'ed25519-private-key.bin').exists()
    request = {
        'checkpoint_sha256': 'a' * 64,
        'checkpoint_bytes': 10,
        'authority_id': service.policy.authority_id,
        'method': ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        'policy_id': service.policy.policy_id,
        'policy_sha256': hashlib.sha256(service.policy_bytes).hexdigest(),
        'checkpoint_schema_version': 'vaxreplay.operations-ledger-checkpoint.v0.1',
    }
    submission = WitnessServiceSubmission(
        witness_request=request,
        client_nonce='1' * 64,
    )
    assert service.issue(canonical_json_bytes(submission)).created
    provider.observation = _health(synchronized=False)
    submission = submission.model_copy(update={'client_nonce': '2' * 64})
    with pytest.raises(ClockHealthError):
        service.issue(canonical_json_bytes(submission))


def test_isolated_witness_signer_cannot_start_without_clock_health_gate(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    isolated = IsolatedEd25519Signer(public_key=_raw_public_key(key), sign_operation=key.sign)
    root = tmp_path / 'must-not-be-created'
    with pytest.raises(WitnessServiceError, match='require.*clock-health'):
        WitnessServiceStore.initialize(
            root,
            authority_id='independent-witness-operator',
            witness_id='isolated-witness-key',
            policy_id='isolated-witness-policy',
            trust_policy_id='isolated-witness-trust',
            endpoint_uri='https://witness.invalid/v1/witness',
            signer=isolated,
        )
    assert not root.exists()


def test_gossip_monitor_uses_isolated_signer_and_gates_observation_and_report(
    tmp_path: Path,
) -> None:
    sources = make_sources.__wrapped__()
    report_key = Ed25519PrivateKey.generate()
    isolated = IsolatedEd25519Signer(
        public_key=_raw_public_key(report_key),
        sign_operation=report_key.sign,
    )
    provider = MutableHealth(_health(checked_at=T0 + timedelta(seconds=2)))
    monitor = CheckpointGossipMonitorStore.initialize(
        tmp_path / 'isolated-gossip-monitor',
        policy=_monitor_policy('isolated-monitor', report_key, sources),
        signer=isolated,
        clock=lambda: T0 + timedelta(seconds=2),
        clock_health_gate=_gate(provider),
    )
    assert not (monitor.root / 'report-ed25519-private-key.bin').exists()
    _observe_bootstraps(monitor, sources)
    assert monitor.signed_report().report.monitor_id == 'isolated-monitor'
    provider.observation = _health(
        checked_at=T0 + timedelta(seconds=2),
        synchronized=False,
    )
    with pytest.raises(ClockHealthError):
        monitor.signed_report()
