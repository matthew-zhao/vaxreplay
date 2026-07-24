from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.checkpoint_gossip import (
    CheckpointGossipError,
    CheckpointGossipMonitorPolicy,
    CheckpointGossipMonitorStore,
    GossipComparisonPolicy,
    GossipMonitorPolicyPin,
    RegistryGossipStreamPolicy,
    SignedGossipMonitorReport,
    WitnessGossipStreamPolicy,
    verify_gossip_agreement,
)
from vaxreplay.operations.checkpoint_gossip_cli import main as gossip_cli_main
from vaxreplay.operations.witness_service_schema import (
    ZERO_SHA256,
    WitnessedRegistryCheckpoint,
    WitnessedSignedRegistryCheckpoint,
    WitnessRegistryMonitor,
    WitnessRegistrySigningKey,
    WitnessServiceLogCheckpoint,
    WitnessServicePolicy,
    WitnessServiceSignedCheckpoint,
    WitnessServiceTrustPolicy,
)

UTC = timezone.utc
T0 = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
_WITNESS_CHECKPOINT_SIGNATURE_DOMAIN = b'VaxReplay witness log checkpoint v0.1\x00'


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value


def _private_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _signed_registry_checkpoint(
    private_key: Ed25519PrivateKey,
    *,
    tree_size: int,
    root_sha256: str,
    previous_checkpoint_sha256: str | None,
    issued_at: datetime = T0,
    signing_key_id: str = 'registry-key-1',
) -> bytes:
    checkpoint = WitnessedRegistryCheckpoint(
        schema_version='vaxreplay.plan-selection-registry-checkpoint.v0.1',
        registry_id='selection-registry',
        authority_id='registry-authority',
        tree_size=tree_size,
        root_sha256=root_sha256,
        issued_at_upper_bound=issued_at,
        signing_key_id=signing_key_id,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
    )
    signed = WitnessedSignedRegistryCheckpoint(
        schema_version='vaxreplay.signed-plan-selection-registry-checkpoint.v0.1',
        checkpoint=checkpoint,
        signature_base64=base64.b64encode(private_key.sign(canonical_json_bytes(checkpoint))).decode('ascii'),
    )
    return canonical_json_bytes(signed)


def _registry_predecessor_sha256(payload: bytes) -> str:
    signed = WitnessedSignedRegistryCheckpoint.model_validate_json(payload)
    return hashlib.sha256(
        canonical_json_bytes(signed.checkpoint) + base64.b64decode(signed.signature_base64, validate=True)
    ).hexdigest()


def _signed_witness_checkpoint(
    private_key: Ed25519PrivateKey,
    policy: WitnessServicePolicy,
    *,
    tree_size: int,
    through_entry_sha256: str,
    previous_checkpoint_sha256: str,
    issued_at: datetime = T0,
) -> bytes:
    checkpoint = WitnessServiceLogCheckpoint(
        authority_id=policy.authority_id,
        witness_id=policy.witness_id,
        policy_id=policy.policy_id,
        policy_sha256=hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
        tree_size=tree_size,
        through_entry_sha256=through_entry_sha256,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
        issued_at=issued_at,
    )
    signed = WitnessServiceSignedCheckpoint(
        checkpoint=checkpoint,
        signature_base64=base64.b64encode(
            private_key.sign(_WITNESS_CHECKPOINT_SIGNATURE_DOMAIN + canonical_json_bytes(checkpoint))
        ).decode('ascii'),
    )
    return canonical_json_bytes(signed)


def _witness_predecessor_sha256(payload: bytes) -> str:
    signed = WitnessServiceSignedCheckpoint.model_validate_json(payload)
    return hashlib.sha256(canonical_json_bytes(signed.checkpoint)).hexdigest()


@dataclass(frozen=True)
class SourceFixture:
    registry_key: Ed25519PrivateKey
    registry_genesis: bytes
    registry_one_a: bytes
    registry_one_b: bytes
    witness_key: Ed25519PrivateKey
    witness_one: bytes
    registry_stream: RegistryGossipStreamPolicy
    witness_stream: WitnessGossipStreamPolicy


@pytest.fixture
def sources() -> SourceFixture:
    registry_key = Ed25519PrivateKey.generate()
    registry_public = _public_bytes(registry_key)
    registry_genesis = _signed_registry_checkpoint(
        registry_key,
        tree_size=0,
        root_sha256=hashlib.sha256(b'').hexdigest(),
        previous_checkpoint_sha256=None,
    )
    registry_predecessor = _registry_predecessor_sha256(registry_genesis)
    registry_one_a = _signed_registry_checkpoint(
        registry_key,
        tree_size=1,
        root_sha256=hashlib.sha256(b'registry-root-a').hexdigest(),
        previous_checkpoint_sha256=registry_predecessor,
        issued_at=T0 + timedelta(seconds=1),
    )
    registry_one_b = _signed_registry_checkpoint(
        registry_key,
        tree_size=1,
        root_sha256=hashlib.sha256(b'registry-root-b').hexdigest(),
        previous_checkpoint_sha256=registry_predecessor,
        issued_at=T0 + timedelta(seconds=1),
    )
    registry_stream = RegistryGossipStreamPolicy(
        stream_id='registry-heads',
        registry_monitor=WitnessRegistryMonitor(
            registry_id='selection-registry',
            authority_id='registry-authority',
            signing_keys=(
                WitnessRegistrySigningKey(
                    key_id='registry-key-1',
                    public_key_base64=base64.b64encode(registry_public).decode('ascii'),
                    valid_from=T0 - timedelta(days=1),
                ),
            ),
        ),
        bootstrap_tree_size=0,
        bootstrap_signed_checkpoint_sha256=hashlib.sha256(registry_genesis).hexdigest(),
    )

    witness_key = Ed25519PrivateKey.generate()
    witness_public = _public_bytes(witness_key)
    witness_policy = WitnessServicePolicy(
        authority_id='witness-authority',
        witness_id='external-witness',
        policy_id='witness-policy',
        endpoint_uri='https://witness.example/v1/witness',
        max_submission_bytes=64 * 1024,
        max_proof_bytes=1024 * 1024,
        client_timeout_seconds=15,
    )
    witness_trust = WitnessServiceTrustPolicy(
        authority_id=witness_policy.authority_id,
        witness_id=witness_policy.witness_id,
        trust_policy_id='witness-trust',
        service_policy_sha256=hashlib.sha256(canonical_json_bytes(witness_policy)).hexdigest(),
        public_key_base64=base64.b64encode(witness_public).decode('ascii'),
        public_key_sha256=hashlib.sha256(witness_public).hexdigest(),
        key_valid_from=T0 - timedelta(days=1),
    )
    witness_one = _signed_witness_checkpoint(
        witness_key,
        witness_policy,
        tree_size=1,
        through_entry_sha256=hashlib.sha256(b'witness-entry-one').hexdigest(),
        previous_checkpoint_sha256=ZERO_SHA256,
    )
    witness_stream = WitnessGossipStreamPolicy(
        stream_id='witness-heads',
        service_policy=witness_policy,
        service_trust_policy=witness_trust,
        bootstrap_tree_size=1,
        bootstrap_signed_checkpoint_sha256=hashlib.sha256(witness_one).hexdigest(),
    )
    return SourceFixture(
        registry_key=registry_key,
        registry_genesis=registry_genesis,
        registry_one_a=registry_one_a,
        registry_one_b=registry_one_b,
        witness_key=witness_key,
        witness_one=witness_one,
        registry_stream=registry_stream,
        witness_stream=witness_stream,
    )


def _monitor_policy(
    monitor_id: str,
    report_key: Ed25519PrivateKey,
    sources: SourceFixture,
) -> CheckpointGossipMonitorPolicy:
    return CheckpointGossipMonitorPolicy(
        monitor_id=monitor_id,
        policy_id=f'{monitor_id}-policy',
        streams=(sources.registry_stream, sources.witness_stream),
        max_observation_age_seconds=60,
        max_future_clock_skew_seconds=5,
        report_signing_key_id=f'{monitor_id}-report-key',
        report_signing_public_key_base64=base64.b64encode(_public_bytes(report_key)).decode('ascii'),
        report_signing_key_valid_from=T0 - timedelta(days=1),
    )


def _new_monitor(
    root: Path,
    monitor_id: str,
    report_key: Ed25519PrivateKey,
    sources: SourceFixture,
    clock: MutableClock,
) -> CheckpointGossipMonitorStore:
    return CheckpointGossipMonitorStore.initialize(
        root,
        policy=_monitor_policy(monitor_id, report_key, sources),
        report_signing_private_key=_private_bytes(report_key),
        clock=clock,
    )


def _observe_bootstraps(
    monitor: CheckpointGossipMonitorStore,
    sources: SourceFixture,
) -> None:
    monitor.observe('registry-heads', sources.registry_genesis)
    monitor.observe('witness-heads', sources.witness_one)


def test_monitor_accepts_adjacent_heads_heartbeats_and_replays_journal(
    tmp_path: Path,
    sources: SourceFixture,
) -> None:
    clock = MutableClock(T0 + timedelta(seconds=2))
    report_key = Ed25519PrivateKey.generate()
    monitor = _new_monitor(tmp_path / 'monitor', 'monitor-a', report_key, sources, clock)

    _observe_bootstraps(monitor, sources)
    successor = monitor.observe('registry-heads', sources.registry_one_a)
    sibling = hashlib.sha256(b'registry-leaf-two').digest()
    registry_two = _signed_registry_checkpoint(
        sources.registry_key,
        tree_size=2,
        root_sha256=hashlib.sha256(b'\x01' + bytes.fromhex(successor.source_root_sha256) + sibling).hexdigest(),
        previous_checkpoint_sha256=_registry_predecessor_sha256(sources.registry_one_a),
        issued_at=T0 + timedelta(seconds=2),
    )
    registry_successor = monitor.observe(
        'registry-heads',
        registry_two,
        registry_consistency_proof_sha256=(sibling.hex(),),
    )
    witness_two = _signed_witness_checkpoint(
        sources.witness_key,
        sources.witness_stream.service_policy,
        tree_size=2,
        through_entry_sha256=hashlib.sha256(b'witness-entry-two').hexdigest(),
        previous_checkpoint_sha256=_witness_predecessor_sha256(sources.witness_one),
        issued_at=T0 + timedelta(seconds=1),
    )
    witness_successor = monitor.observe('witness-heads', witness_two)
    heartbeat = monitor.observe('witness-heads', witness_two)

    assert successor.transition == 'successor'
    assert registry_successor.transition == 'successor'
    assert witness_successor.transition == 'successor'
    assert heartbeat.transition == 'heartbeat'
    verification = monitor.verify()
    assert verification.observation_count == 6
    assert verification.stream_count == 2
    report = monitor.signed_report()
    assert report.report.monitor_id == 'monitor-a'
    assert [head.stream_id for head in report.report.stream_heads] == [
        'registry-heads',
        'witness-heads',
    ]

    reopened = CheckpointGossipMonitorStore(tmp_path / 'monitor', clock=clock)
    assert reopened.verify() == verification


def test_monitor_rejects_rollback_conflict_gap_broken_link_and_signer_mismatch(
    tmp_path: Path,
    sources: SourceFixture,
) -> None:
    clock = MutableClock(T0 + timedelta(seconds=5))
    monitor = _new_monitor(
        tmp_path / 'monitor',
        'monitor-a',
        Ed25519PrivateKey.generate(),
        sources,
        clock,
    )
    _observe_bootstraps(monitor, sources)
    monitor.observe('registry-heads', sources.registry_one_a)

    with pytest.raises(CheckpointGossipError, match='rollback'):
        monitor.observe('registry-heads', sources.registry_genesis)
    with pytest.raises(CheckpointGossipError, match='same-sequence conflicting'):
        monitor.observe('registry-heads', sources.registry_one_b)

    predecessor = _registry_predecessor_sha256(sources.registry_one_a)
    gap = _signed_registry_checkpoint(
        sources.registry_key,
        tree_size=3,
        root_sha256=hashlib.sha256(b'gap').hexdigest(),
        previous_checkpoint_sha256=predecessor,
        issued_at=T0 + timedelta(seconds=2),
    )
    with pytest.raises(CheckpointGossipError, match='tree-size gap'):
        monitor.observe('registry-heads', gap)

    broken = _signed_registry_checkpoint(
        sources.registry_key,
        tree_size=2,
        root_sha256=hashlib.sha256(b'broken').hexdigest(),
        previous_checkpoint_sha256='1' * 64,
        issued_at=T0 + timedelta(seconds=2),
    )
    with pytest.raises(CheckpointGossipError, match='exact predecessor'):
        monitor.observe('registry-heads', broken)

    inconsistent = _signed_registry_checkpoint(
        sources.registry_key,
        tree_size=2,
        root_sha256=hashlib.sha256(b'inconsistent').hexdigest(),
        previous_checkpoint_sha256=predecessor,
        issued_at=T0 + timedelta(seconds=2),
    )
    with pytest.raises(CheckpointGossipError, match='RFC6962 consistency proof'):
        monitor.observe('registry-heads', inconsistent)

    wrong_key = Ed25519PrivateKey.generate()
    mismatched = _signed_registry_checkpoint(
        wrong_key,
        tree_size=2,
        root_sha256=hashlib.sha256(b'mismatched').hexdigest(),
        previous_checkpoint_sha256=predecessor,
        issued_at=T0 + timedelta(seconds=2),
    )
    with pytest.raises(CheckpointGossipError, match='signature verification failed'):
        monitor.observe('registry-heads', mismatched)

    assert monitor.verify().observation_count == 3


def test_witness_predecessor_break_and_database_updates_fail_closed(
    tmp_path: Path,
    sources: SourceFixture,
) -> None:
    clock = MutableClock(T0 + timedelta(seconds=5))
    monitor = _new_monitor(
        tmp_path / 'monitor',
        'monitor-a',
        Ed25519PrivateKey.generate(),
        sources,
        clock,
    )
    _observe_bootstraps(monitor, sources)
    broken = _signed_witness_checkpoint(
        sources.witness_key,
        sources.witness_stream.service_policy,
        tree_size=2,
        through_entry_sha256=hashlib.sha256(b'witness-entry-two').hexdigest(),
        previous_checkpoint_sha256='2' * 64,
        issued_at=T0 + timedelta(seconds=1),
    )
    with pytest.raises(CheckpointGossipError, match='exact predecessor'):
        monitor.observe('witness-heads', broken)

    database = tmp_path / 'monitor' / 'gossip.sqlite3'
    connection = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            connection.execute('UPDATE observations SET stream_id = ?', ('other',))
    finally:
        connection.close()


def test_stale_observation_blocks_report_until_fresh_heartbeat(
    tmp_path: Path,
    sources: SourceFixture,
) -> None:
    clock = MutableClock(T0 + timedelta(seconds=2))
    monitor = _new_monitor(
        tmp_path / 'monitor',
        'monitor-a',
        Ed25519PrivateKey.generate(),
        sources,
        clock,
    )
    _observe_bootstraps(monitor, sources)
    clock.value += timedelta(seconds=61)
    with pytest.raises(CheckpointGossipError, match='stale'):
        monitor.signed_report()

    monitor.observe('registry-heads', sources.registry_genesis)
    monitor.observe('witness-heads', sources.witness_one)
    assert monitor.signed_report().report.generated_at == clock.value


def _comparison_policy(
    first: CheckpointGossipMonitorPolicy,
    second: CheckpointGossipMonitorPolicy,
) -> GossipComparisonPolicy:
    return GossipComparisonPolicy(
        comparison_policy_id='two-monitor-quorum',
        monitors=tuple(
            GossipMonitorPolicyPin(
                monitor_id=policy.monitor_id,
                monitor_policy_sha256=hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
                monitor_policy=policy,
            )
            for policy in (first, second)
        ),
        required_stream_ids=('registry-heads', 'witness-heads'),
        max_report_age_seconds=60,
        max_observation_age_seconds=60,
        max_future_clock_skew_seconds=5,
    )


def test_composite_gossip_schema_versions_are_explicit(sources: SourceFixture) -> None:
    witness_stream = sources.witness_stream
    witness_stream_bytes = canonical_json_bytes(witness_stream)
    assert witness_stream.schema_version == 'vaxreplay.witness-gossip-stream-policy.v0.2'
    assert WitnessGossipStreamPolicy.model_validate_json(witness_stream_bytes) == witness_stream

    legacy_stream = witness_stream.model_dump(mode='json')
    legacy_stream['schema_version'] = 'vaxreplay.witness-gossip-stream-policy.v0.1'
    with pytest.raises(ValueError, match='schema_version'):
        WitnessGossipStreamPolicy.model_validate(legacy_stream)

    first = _monitor_policy('monitor-a', Ed25519PrivateKey.generate(), sources)
    second = _monitor_policy('monitor-b', Ed25519PrivateKey.generate(), sources)
    first_bytes = canonical_json_bytes(first)
    assert first.schema_version == 'vaxreplay.checkpoint-gossip-monitor-policy.v0.3'
    assert CheckpointGossipMonitorPolicy.model_validate_json(first_bytes) == first

    legacy_monitor = first.model_dump(mode='json')
    legacy_monitor['schema_version'] = 'vaxreplay.checkpoint-gossip-monitor-policy.v0.2'
    with pytest.raises(ValueError, match='schema_version'):
        CheckpointGossipMonitorPolicy.model_validate(legacy_monitor)

    comparison = _comparison_policy(first, second)
    comparison_bytes = canonical_json_bytes(comparison)
    assert comparison.schema_version == 'vaxreplay.checkpoint-gossip-comparison-policy.v0.2'
    assert GossipComparisonPolicy.model_validate_json(comparison_bytes) == comparison

    legacy_comparison = comparison.model_dump(mode='json')
    legacy_comparison['schema_version'] = 'vaxreplay.checkpoint-gossip-comparison-policy.v0.1'
    with pytest.raises(ValueError, match='schema_version'):
        GossipComparisonPolicy.model_validate(legacy_comparison)


def test_two_monitor_agreement_and_disagreement_are_fail_closed(
    tmp_path: Path,
    sources: SourceFixture,
) -> None:
    clock = MutableClock(T0 + timedelta(seconds=2))
    key_a = Ed25519PrivateKey.generate()
    key_b = Ed25519PrivateKey.generate()
    monitor_a = _new_monitor(tmp_path / 'a', 'monitor-a', key_a, sources, clock)
    monitor_b = _new_monitor(tmp_path / 'b', 'monitor-b', key_b, sources, clock)
    for monitor in (monitor_a, monitor_b):
        _observe_bootstraps(monitor, sources)
        monitor.observe('registry-heads', sources.registry_one_a)
    comparison = _comparison_policy(monitor_a.policy, monitor_b.policy)
    comparison_bytes = canonical_json_bytes(comparison)
    reports = tuple(canonical_json_bytes(monitor.signed_report()) for monitor in (monitor_a, monitor_b))

    agreement = verify_gossip_agreement(reports, comparison_bytes, now=clock.value)
    assert agreement.monitor_ids == ('monitor-a', 'monitor-b')
    assert agreement.exact_latest_heads_agree is True

    monitor_c = _new_monitor(tmp_path / 'c', 'monitor-b', key_b, sources, clock)
    _observe_bootstraps(monitor_c, sources)
    monitor_c.observe('registry-heads', sources.registry_one_b)
    conflicting_reports = (
        canonical_json_bytes(monitor_a.signed_report()),
        canonical_json_bytes(monitor_c.signed_report()),
    )
    with pytest.raises(CheckpointGossipError, match='disagree'):
        verify_gossip_agreement(conflicting_reports, comparison_bytes, now=clock.value)


def test_report_signature_and_freshness_are_verified_during_comparison(
    tmp_path: Path,
    sources: SourceFixture,
) -> None:
    clock = MutableClock(T0 + timedelta(seconds=2))
    key_a = Ed25519PrivateKey.generate()
    key_b = Ed25519PrivateKey.generate()
    monitor_a = _new_monitor(tmp_path / 'a', 'monitor-a', key_a, sources, clock)
    monitor_b = _new_monitor(tmp_path / 'b', 'monitor-b', key_b, sources, clock)
    for monitor in (monitor_a, monitor_b):
        _observe_bootstraps(monitor, sources)
    comparison_bytes = canonical_json_bytes(_comparison_policy(monitor_a.policy, monitor_b.policy))
    report_a = monitor_a.signed_report()
    report_b = monitor_b.signed_report()
    bad_signature = SignedGossipMonitorReport(
        report=report_a.report,
        signing_key_id=report_a.signing_key_id,
        signature_base64=base64.b64encode(b'\x00' * 64).decode('ascii'),
    )
    with pytest.raises(CheckpointGossipError, match='signature verification failed'):
        verify_gossip_agreement(
            (canonical_json_bytes(bad_signature), canonical_json_bytes(report_b)),
            comparison_bytes,
            now=clock.value,
        )

    with pytest.raises(CheckpointGossipError, match='stale'):
        verify_gossip_agreement(
            (canonical_json_bytes(report_a), canonical_json_bytes(report_b)),
            comparison_bytes,
            now=clock.value + timedelta(seconds=61),
        )


def test_cli_generates_an_owner_only_report_key(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / 'report-key.bin'
    gossip_cli_main(['generate-report-key', '--output', str(output)])

    result = json.loads(capfd.readouterr().out)
    assert result['output'] == str(output)
    assert len(base64.b64decode(result['public_key_base64'], validate=True)) == 32
    assert len(output.read_bytes()) == 32
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
