from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.release_readiness import (
    NamedReadinessSubject,
    ReadinessEvidenceAuthority,
    ReadinessEvidenceStatement,
    ReadinessGateRequirement,
    ReadinessMaterial,
    ReadinessVerificationTimeAuthority,
    ReleaseReadinessError,
    ReleaseVerificationTimeStatement,
    SignedReadinessEvidence,
    SignedReleaseVerificationTime,
    TierAReleaseReadinessManifest,
    TierAReleaseReadinessPolicy,
    TierAReleaseScope,
    applicable_gate_ids,
    readiness_evidence_signature_payload,
    release_verification_time_signature_payload,
    verify_tier_a_release_readiness,
)

_NOW = datetime(2026, 7, 14, 20, tzinfo=timezone.utc)
_NEW_DEPLOYMENT_REGISTER_GATES = (
    'frozen_case_universe_and_decision_package',
    'legal_rights_approved',
    'monitoring_and_incident_response_qualified',
    'outcome_label_isolation_verified',
    'promotion_and_official_admission_replay_verified',
    'real_capture_schedule_complete',
    'registry_witness_organizational_independence',
)


class _ChameleonBytesMapping(Mapping[str, bytes]):
    def __init__(self, snapshot: Mapping[str, bytes], later: Mapping[str, bytes]) -> None:
        self._snapshot = dict(snapshot)
        self._later = dict(later)
        self.indexed_reads = 0

    def __getitem__(self, key: str) -> bytes:
        self.indexed_reads += 1
        return self._later[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._snapshot)

    def __len__(self) -> int:
        return len(self._snapshot)

    def items(self):  # noqa: ANN201
        return self._snapshot.items()


def _raw_public_key(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _fixture(
    *,
    scope: TierAReleaseScope | None = None,
    omitted_gate: str | None = None,
    two_authority_gate: str | None = None,
    maximum_evidence_age_seconds: int | None = None,
    verification_offset_seconds: int = 3,
) -> dict[str, object]:
    scope = scope or TierAReleaseScope(
        sources=('immport',),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    private_a = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    private_b = Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64)))
    private_time = Ed25519PrivateKey.from_private_bytes(bytes(range(64, 96)))
    public_a = _raw_public_key(private_a)
    public_b = _raw_public_key(private_b)
    public_time = _raw_public_key(private_time)
    authorities = (
        ReadinessEvidenceAuthority(
            authority_id='external-auditor-a',
            organization_id='audit-org-a',
            failure_domain_id='audit-domain-a',
            signing_public_key_sha256=hashlib.sha256(public_a).hexdigest(),
        ),
        ReadinessEvidenceAuthority(
            authority_id='external-auditor-b',
            organization_id='audit-org-b',
            failure_domain_id='audit-domain-b',
            signing_public_key_sha256=hashlib.sha256(public_b).hexdigest(),
        ),
    )
    gates = tuple(
        ReadinessGateRequirement(
            gate_id=gate_id,
            allowed_authority_ids=('external-auditor-a', 'external-auditor-b'),
            minimum_distinct_authorities=2 if gate_id == two_authority_gate else 1,
            allowed_media_types=('application/vnd.vaxreplay.audit+json',),
            maximum_evidence_age_seconds=maximum_evidence_age_seconds,
        )
        for gate_id in applicable_gate_ids(scope)
    )
    policy = TierAReleaseReadinessPolicy(
        policy_id='tier-a-policy-fixture',
        issued_at=_NOW,
        organizer_organization_id='vaxreplay-organizer',
        organizer_failure_domain_id='vaxreplay-organizer-domain',
        scope=scope,
        authorities=authorities,
        verification_time_authority=ReadinessVerificationTimeAuthority(
            authority_id='external-verification-time-authority',
            organization_id='verification-time-org',
            failure_domain_id='verification-time-domain',
            signing_public_key_sha256=hashlib.sha256(public_time).hexdigest(),
        ),
        gates=gates,
    )
    policy_bytes = canonical_json_bytes(policy)
    release_subject = b'exact Tier A release archive bytes'
    artifact = b'{"external_audit":"fixture-only"}'
    artifact_binding = ReadinessMaterial(
        sha256=hashlib.sha256(artifact).hexdigest(),
        byte_count=len(artifact),
    )
    subjects = (
        NamedReadinessSubject(
            role='release-archive',
            material=ReadinessMaterial(
                sha256=hashlib.sha256(release_subject).hexdigest(),
                byte_count=len(release_subject),
            ),
        ),
    )
    covered_gates = tuple(gate for gate in applicable_gate_ids(scope) if gate != omitted_gate)
    statement_a = ReadinessEvidenceStatement(
        statement_id='external-audit-statement-a',
        authority_id='external-auditor-a',
        issued_at=_NOW + timedelta(seconds=1),
        release_id='release-fixture',
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        scope_sha256=hashlib.sha256(canonical_json_bytes(scope)).hexdigest(),
        release_subjects=subjects,
        gate_ids=covered_gates,
        evidence_artifact=artifact_binding,
        evidence_media_type='application/vnd.vaxreplay.audit+json',
        immutable_locator=f'urn:sha256:{artifact_binding.sha256}',
    )
    envelopes = [
        SignedReadinessEvidence(
            statement=statement_a,
            signature_base64=base64.b64encode(
                private_a.sign(readiness_evidence_signature_payload(statement_a))
            ).decode(),
        )
    ]
    if two_authority_gate is not None:
        statement_b = ReadinessEvidenceStatement(
            statement_id='external-audit-statement-b',
            authority_id='external-auditor-b',
            issued_at=_NOW + timedelta(seconds=1),
            release_id='release-fixture',
            policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
            scope_sha256=hashlib.sha256(canonical_json_bytes(scope)).hexdigest(),
            release_subjects=subjects,
            gate_ids=(two_authority_gate,),
            evidence_artifact=artifact_binding,
            evidence_media_type='application/vnd.vaxreplay.audit+json',
            immutable_locator=f'urn:sha256:{artifact_binding.sha256}',
        )
        envelopes.append(
            SignedReadinessEvidence(
                statement=statement_b,
                signature_base64=base64.b64encode(
                    private_b.sign(readiness_evidence_signature_payload(statement_b))
                ).decode(),
            )
        )
    manifest = TierAReleaseReadinessManifest(
        release_id='release-fixture',
        created_at=_NOW + timedelta(seconds=2),
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        scope=scope,
        subjects=subjects,
        evidence=tuple(envelopes),
    )
    manifest_bytes = canonical_json_bytes(manifest)
    verified_at = _NOW + timedelta(seconds=verification_offset_seconds)
    time_statement = ReleaseVerificationTimeStatement(
        statement_id='release-verification-time-fixture',
        authority_id=policy.verification_time_authority.authority_id,
        release_id=manifest.release_id,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        readiness_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        verified_at=verified_at,
    )
    time_evidence = SignedReleaseVerificationTime(
        statement=time_statement,
        signature_base64=base64.b64encode(
            private_time.sign(release_verification_time_signature_payload(time_statement))
        ).decode(),
    )
    return {
        'policy_bytes': policy_bytes,
        'expected_policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
        'manifest_bytes': manifest_bytes,
        'release_subject_bytes': {'release-archive': release_subject},
        'evidence_artifact_bytes': {artifact_binding.sha256: artifact},
        'authority_public_key_bytes': {
            'external-auditor-a': public_a,
            'external-auditor-b': public_b,
        },
        'verification_time_evidence_bytes': canonical_json_bytes(time_evidence),
        'verification_time_public_key_bytes': public_time,
        'verified_at': verified_at,
    }


def test_readiness_verifier_satisfies_every_scope_derived_gate() -> None:
    fixture = _fixture()
    report = verify_tier_a_release_readiness(**fixture)
    expected_scope = TierAReleaseScope(
        sources=('immport',),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    assert report.applicable_gate_count == len(applicable_gate_ids(expected_scope))
    assert report.every_applicable_gate_authority_claim_verified
    assert report.every_signature_verified
    assert report.every_artifact_verified
    assert report.verification_time_attestation_verified
    assert report.machine_readiness_evidence_verified
    assert report.tier_a_release_ready_claimed_by_policy_authorities
    assert not report.deployment_tier_a_status_independently_determined
    assert not report.external_organizational_independence_cryptographically_proven


def test_readiness_verifier_uses_one_immutable_subject_snapshot() -> None:
    fixture = _fixture()
    honest = fixture['release_subject_bytes']
    assert isinstance(honest, dict)
    subjects = _ChameleonBytesMapping(
        honest,
        {'release-archive': b'unsigned replacement subject'},
    )
    fixture['release_subject_bytes'] = subjects

    verify_tier_a_release_readiness(**fixture)

    assert subjects.indexed_reads == 0


def test_scope_deterministically_adds_immport_open_set_and_model_gates() -> None:
    minimal = TierAReleaseScope(
        sources=('iedb',),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    expanded = TierAReleaseScope(
        sources=('iedb', 'immport'),
        tasks=('antigen_target_prioritization', 'open_set_nomination'),
        includes_model_leaderboard=True,
    )
    additions = set(applicable_gate_ids(expanded)) - set(applicable_gate_ids(minimal))
    assert additions == {
        'immport_arm_construct_mapping_review',
        'immport_egress_tls_enforced',
        'immport_fd_broker_handoff_qualified',
        'immport_host_memory_controls_qualified',
        'immport_producer_runtime_image_enforced',
        'immport_publisher_time_semantics_accepted',
        'immport_secret_broker_qualified',
        'immport_secret_scanning_zeroization',
        'immport_source_profile_qualified',
        'model_harness_identity_policy_frozen',
        'open_set_nomination_protocol_frozen',
        'provider_cancellation_qualified',
        'sealed_model_execution_qualified',
    }


def test_source_specific_and_deployment_register_gates_are_scope_derived() -> None:
    iedb = TierAReleaseScope(
        sources=('iedb',),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    clinicaltrials = TierAReleaseScope(
        sources=('clinicaltrials.gov',),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    assert 'iedb_source_profile_qualified' in applicable_gate_ids(iedb)
    assert 'clinicaltrials_parent_query_complete' not in applicable_gate_ids(iedb)
    assert {
        'clinicaltrials_parent_query_complete',
        'clinicaltrials_source_profile_qualified',
    }.issubset(applicable_gate_ids(clinicaltrials))
    assert 'iedb_source_profile_qualified' not in applicable_gate_ids(clinicaltrials)
    assert set(_NEW_DEPLOYMENT_REGISTER_GATES).issubset(applicable_gate_ids(iedb))


@pytest.mark.parametrize('omitted_gate', _NEW_DEPLOYMENT_REGISTER_GATES)
def test_policy_cannot_omit_new_deployment_register_gate(omitted_gate: str) -> None:
    policy = TierAReleaseReadinessPolicy.model_validate_json(_fixture()['policy_bytes'])
    incomplete = policy.model_copy(
        update={'gates': tuple(requirement for requirement in policy.gates if requirement.gate_id != omitted_gate)}
    )
    with pytest.raises(ValueError, match='exactly equal scope-derived applicable gates'):
        TierAReleaseReadinessPolicy.model_validate_json(canonical_json_bytes(incomplete))


@pytest.mark.parametrize(
    ('scope', 'omitted_gate'),
    (
        (
            TierAReleaseScope(
                sources=('iedb',),
                tasks=('antigen_target_prioritization',),
                includes_model_leaderboard=False,
            ),
            'iedb_source_profile_qualified',
        ),
        (
            TierAReleaseScope(
                sources=('clinicaltrials.gov',),
                tasks=('antigen_target_prioritization',),
                includes_model_leaderboard=False,
            ),
            'clinicaltrials_parent_query_complete',
        ),
        (
            TierAReleaseScope(
                sources=('immport',),
                tasks=('antigen_target_prioritization',),
                includes_model_leaderboard=False,
            ),
            'immport_publisher_time_semantics_accepted',
        ),
    ),
)
def test_policy_cannot_omit_scope_specific_source_gate(
    scope: TierAReleaseScope,
    omitted_gate: str,
) -> None:
    policy = TierAReleaseReadinessPolicy.model_validate_json(_fixture(scope=scope)['policy_bytes'])
    incomplete = policy.model_copy(
        update={'gates': tuple(requirement for requirement in policy.gates if requirement.gate_id != omitted_gate)}
    )
    with pytest.raises(ValueError, match='exactly equal scope-derived applicable gates'):
        TierAReleaseReadinessPolicy.model_validate_json(canonical_json_bytes(incomplete))


def test_missing_gate_tampered_artifact_and_extra_key_fail_closed() -> None:
    scope = TierAReleaseScope(
        sources=('iedb',),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    omitted = applicable_gate_ids(scope)[0]
    with pytest.raises(ReleaseReadinessError, match='lacks its authority quorum'):
        verify_tier_a_release_readiness(**_fixture(scope=scope, omitted_gate=omitted))

    fixture = _fixture(scope=scope)
    digest = next(iter(fixture['evidence_artifact_bytes']))
    fixture['evidence_artifact_bytes'] = {digest: b'tampered'}
    with pytest.raises(ReleaseReadinessError, match='differs from its signed binding'):
        verify_tier_a_release_readiness(**fixture)

    fixture = _fixture(scope=scope)
    fixture['release_subject_bytes'] = {'release-archive': b'tampered release'}
    with pytest.raises(ReleaseReadinessError, match='release subject differs'):
        verify_tier_a_release_readiness(**fixture)

    fixture = _fixture(scope=scope)
    fixture['expected_policy_sha256'] = 'f' * 64
    with pytest.raises(ReleaseReadinessError, match='out-of-band expected digest'):
        verify_tier_a_release_readiness(**fixture)

    fixture = _fixture(scope=scope)
    keys = dict(fixture['authority_public_key_bytes'])
    keys['untrusted-extra-key'] = b'x' * 32
    fixture['authority_public_key_bytes'] = keys
    with pytest.raises(ReleaseReadinessError, match='key inventory'):
        verify_tier_a_release_readiness(**fixture)


def test_gate_can_require_two_distinct_external_authorities() -> None:
    scope = TierAReleaseScope(
        sources=('iedb',),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    quorum_gate = 'timestamp_witnesses_independent'
    fixture = _fixture(scope=scope, two_authority_gate=quorum_gate)
    report = verify_tier_a_release_readiness(**fixture)
    assert report.evidence_statement_count == 2

    manifest = TierAReleaseReadinessManifest.model_validate_json(fixture['manifest_bytes'])
    fixture['manifest_bytes'] = canonical_json_bytes(manifest.model_copy(update={'evidence': manifest.evidence[:1]}))
    time_evidence = SignedReleaseVerificationTime.model_validate_json(fixture['verification_time_evidence_bytes'])
    rebound_statement = time_evidence.statement.model_copy(
        update={'readiness_manifest_sha256': hashlib.sha256(fixture['manifest_bytes']).hexdigest()}
    )
    time_private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(64, 96)))
    fixture['verification_time_evidence_bytes'] = canonical_json_bytes(
        SignedReleaseVerificationTime(
            statement=rebound_statement,
            signature_base64=base64.b64encode(
                time_private_key.sign(release_verification_time_signature_payload(rebound_statement))
            ).decode(),
        )
    )
    with pytest.raises(ReleaseReadinessError, match='lacks its authority quorum'):
        verify_tier_a_release_readiness(**fixture)


def test_evidence_freshness_uses_verification_time_not_unsigned_manifest_time() -> None:
    fixture = _fixture(
        maximum_evidence_age_seconds=5,
        verification_offset_seconds=10,
    )
    with pytest.raises(ReleaseReadinessError, match='older than its gate permits'):
        verify_tier_a_release_readiness(**fixture)


def test_verification_time_cannot_be_backdated_without_a_matching_external_signature() -> None:
    fixture = _fixture(maximum_evidence_age_seconds=5, verification_offset_seconds=10)
    fixture['verified_at'] = _NOW + timedelta(seconds=3)
    with pytest.raises(ReleaseReadinessError, match='verification-time evidence differs'):
        verify_tier_a_release_readiness(**fixture)


def test_readiness_authority_quorum_cannot_reuse_one_external_organization() -> None:
    policy = TierAReleaseReadinessPolicy.model_validate_json(_fixture()['policy_bytes'])
    duplicate_organization = policy.authorities[1].model_copy(
        update={'organization_id': policy.authorities[0].organization_id}
    )
    invalid = policy.model_copy(update={'authorities': (policy.authorities[0], duplicate_organization)})
    with pytest.raises(ValueError, match='distinct external organizations'):
        TierAReleaseReadinessPolicy.model_validate_json(canonical_json_bytes(invalid))

    duplicate_domain = policy.authorities[1].model_copy(
        update={'failure_domain_id': policy.authorities[0].failure_domain_id}
    )
    invalid = policy.model_copy(update={'authorities': (policy.authorities[0], duplicate_domain)})
    with pytest.raises(ValueError, match='distinct external failure domains'):
        TierAReleaseReadinessPolicy.model_validate_json(canonical_json_bytes(invalid))


def test_readiness_cli_verifies_flat_digest_inventory(tmp_path: Path) -> None:
    fixture = _fixture()
    policy = tmp_path / 'policy.json'
    manifest = tmp_path / 'manifest.json'
    evidence_root = tmp_path / 'evidence'
    evidence_root.mkdir()
    policy.write_bytes(fixture['policy_bytes'])
    manifest.write_bytes(fixture['manifest_bytes'])
    verification_time_evidence = tmp_path / 'verification-time.json'
    verification_time_key = tmp_path / 'verification-time.pub'
    verification_time_evidence.write_bytes(fixture['verification_time_evidence_bytes'])
    verification_time_key.write_bytes(fixture['verification_time_public_key_bytes'])
    (tmp_path / 'release.tar').write_bytes(fixture['release_subject_bytes']['release-archive'])
    for digest, payload in fixture['evidence_artifact_bytes'].items():
        (evidence_root / digest).write_bytes(payload)
    key_arguments: list[str] = []
    for authority_id, payload in fixture['authority_public_key_bytes'].items():
        path = tmp_path / f'{authority_id}.pub'
        path.write_bytes(payload)
        key_arguments.extend(('--authority-key', f'{authority_id}={path}'))
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')}
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'vaxreplay.operations.release_readiness_cli',
            'verify',
            '--policy',
            str(policy),
            '--expected-policy-sha256',
            fixture['expected_policy_sha256'],
            '--manifest',
            str(manifest),
            '--verified-at',
            fixture['verified_at'].isoformat(),
            '--verification-time-evidence',
            str(verification_time_evidence),
            '--verification-time-public-key',
            str(verification_time_key),
            '--subject',
            f'release-archive={tmp_path / "release.tar"}',
            '--evidence-root',
            str(evidence_root),
            *key_arguments,
        ],
        check=False,
        capture_output=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr.decode()
    output = json.loads(result.stdout)
    assert output['machine_readiness_evidence_verified'] is True
    assert output['deployment_tier_a_status_independently_determined'] is False


def test_readiness_cli_lists_exact_scope_derived_gates_without_decisions(tmp_path: Path) -> None:
    scope = TierAReleaseScope(
        sources=('clinicaltrials.gov', 'iedb'),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    scope_path = tmp_path / 'scope.json'
    scope_path.write_bytes(canonical_json_bytes(scope))
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')}
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'vaxreplay.operations.release_readiness_cli',
            'list-gates',
            '--scope',
            str(scope_path),
        ],
        check=False,
        capture_output=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr.decode()
    output = json.loads(result.stdout)
    assert output['gate_ids'] == list(applicable_gate_ids(scope))
    assert output['applicable_gate_count'] == len(applicable_gate_ids(scope))
    assert 'status' not in output
