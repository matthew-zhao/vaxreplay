from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterator, Mapping
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.test_operations_campaign_publication import _publication
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.campaign_publication import (
    CampaignPublicationTrustPolicy,
    SignedCampaignPublicationManifest,
)
from vaxreplay.operations.campaign_publication_cli import (
    ArtifactPathEntry,
    PublicationArtifactPathMap,
)
from vaxreplay.operations.release_decision import (
    TierAReleaseDecisionError,
    TierAReleaseDecisionReport,
    verify_tier_a_release_decision,
)
from vaxreplay.operations.release_decision_cli import main as release_decision_main
from vaxreplay.operations.release_readiness import (
    NamedReadinessSubject,
    ReadinessEvidenceAuthority,
    ReadinessEvidenceStatement,
    ReadinessGateRequirement,
    ReadinessMaterial,
    ReadinessVerificationTimeAuthority,
    ReleaseVerificationTimeStatement,
    SignedReadinessEvidence,
    SignedReleaseVerificationTime,
    TierAReleaseReadinessManifest,
    TierAReleaseReadinessPolicy,
    TierAReleaseScope,
    applicable_gate_ids,
    readiness_evidence_signature_payload,
    release_verification_time_signature_payload,
)


class _ChameleonBytesMapping(Mapping[str, bytes]):
    """Expose one items snapshot but attacker-controlled later indexed reads."""

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


def _decision(
    tmp_path: Path,
    *,
    wrong_trust_subject: bool = False,
    readiness_release_id: str | None = None,
    evidence_predates_release: bool = False,
    reuse_release_key_for_readiness: bool = False,
    campaign_release_id: str = 'future-immport-release-001',
    release_archive_bytes: bytes = b'fictional immutable release archive bytes',
    release_archive_index_bytes: bytes = b'{"archive":"fictional"}\n',
    readiness_scope: TierAReleaseScope | None = None,
) -> dict[str, object]:
    campaign = _publication(
        tmp_path,
        release_id=campaign_release_id,
        release_archive_bytes=release_archive_bytes,
        release_archive_index_bytes=release_archive_index_bytes,
    )
    campaign_trust = CampaignPublicationTrustPolicy.model_validate_json(campaign.trust_policy_bytes)
    created_at = campaign.manifest.created_at
    archive = campaign.artifacts[campaign.manifest.archive_artifact_id]
    archive_index = campaign.artifacts[campaign.manifest.archive_index_artifact_id]
    trust_subject = (
        b'an internally signed but wrong campaign trust subject' if wrong_trust_subject else campaign.trust_policy_bytes
    )
    subject_bytes = {
        'campaign-signed-manifest': campaign.signed_manifest_bytes,
        'campaign-trust-policy': trust_subject,
        'release-archive': archive,
        'release-archive-index': archive_index,
    }

    subjects = tuple(
        NamedReadinessSubject(
            role=role,
            material=ReadinessMaterial(
                sha256=hashlib.sha256(payload).hexdigest(),
                byte_count=len(payload),
            ),
        )
        for role, payload in sorted(subject_bytes.items())
    )
    scope = readiness_scope or TierAReleaseScope(
        sources=('immport',),
        tasks=('antigen_target_prioritization',),
        includes_model_leaderboard=False,
    )
    authority_key = (
        campaign.release_key
        if reuse_release_key_for_readiness
        else Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    )
    authority_public_key = authority_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    time_key = Ed25519PrivateKey.from_private_bytes(bytes(range(96, 128)))
    time_public_key = time_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    policy = TierAReleaseReadinessPolicy(
        policy_id='composed-release-readiness-policy',
        issued_at=created_at - timedelta(seconds=2),
        organizer_organization_id=campaign_trust.release_organization_id,
        organizer_failure_domain_id=campaign_trust.release_failure_domain_id,
        scope=scope,
        authorities=(
            ReadinessEvidenceAuthority(
                authority_id='external-release-auditor',
                organization_id='external-release-auditor-organization',
                failure_domain_id='external-release-auditor-domain',
                signing_public_key_sha256=hashlib.sha256(authority_public_key).hexdigest(),
            ),
        ),
        verification_time_authority=ReadinessVerificationTimeAuthority(
            authority_id='external-release-time-authority',
            organization_id='external-release-time-organization',
            failure_domain_id='external-release-time-domain',
            signing_public_key_sha256=hashlib.sha256(time_public_key).hexdigest(),
        ),
        gates=tuple(
            ReadinessGateRequirement(
                gate_id=gate_id,
                allowed_authority_ids=('external-release-auditor',),
                minimum_distinct_authorities=1,
                allowed_media_types=('application/vnd.vaxreplay.release-audit+json',),
                maximum_evidence_age_seconds=60,
            )
            for gate_id in applicable_gate_ids(scope)
        ),
    )
    policy_bytes = canonical_json_bytes(policy)
    evidence_artifact = b'{"external_release_audit":"fixture"}'
    evidence_binding = ReadinessMaterial(
        sha256=hashlib.sha256(evidence_artifact).hexdigest(),
        byte_count=len(evidence_artifact),
    )
    statement = ReadinessEvidenceStatement(
        statement_id='external-release-approval',
        authority_id='external-release-auditor',
        issued_at=created_at + timedelta(seconds=-1 if evidence_predates_release else 4),
        release_id=readiness_release_id or campaign.manifest.release_id,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        scope_sha256=hashlib.sha256(canonical_json_bytes(scope)).hexdigest(),
        release_subjects=subjects,
        gate_ids=applicable_gate_ids(scope),
        evidence_artifact=evidence_binding,
        evidence_media_type='application/vnd.vaxreplay.release-audit+json',
        immutable_locator=f'urn:sha256:{evidence_binding.sha256}',
    )
    signed_evidence = SignedReadinessEvidence(
        statement=statement,
        signature_base64=base64.b64encode(authority_key.sign(readiness_evidence_signature_payload(statement))).decode(
            'ascii'
        ),
    )
    readiness_manifest = TierAReleaseReadinessManifest(
        release_id=statement.release_id,
        created_at=created_at + timedelta(seconds=5),
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        scope=scope,
        subjects=subjects,
        evidence=(signed_evidence,),
    )
    readiness_manifest_bytes = canonical_json_bytes(readiness_manifest)
    verified_at = created_at + timedelta(seconds=6)
    time_statement = ReleaseVerificationTimeStatement(
        statement_id='external-release-verification-time',
        authority_id=policy.verification_time_authority.authority_id,
        release_id=readiness_manifest.release_id,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        readiness_manifest_sha256=hashlib.sha256(readiness_manifest_bytes).hexdigest(),
        verified_at=verified_at,
    )
    time_evidence = SignedReleaseVerificationTime(
        statement=time_statement,
        signature_base64=base64.b64encode(
            time_key.sign(release_verification_time_signature_payload(time_statement))
        ).decode('ascii'),
    )
    return {
        'signed_campaign_manifest_bytes': campaign.signed_manifest_bytes,
        'campaign_trust_policy_bytes': campaign.trust_policy_bytes,
        'expected_campaign_trust_policy_sha256': hashlib.sha256(campaign.trust_policy_bytes).hexdigest(),
        'campaign_artifacts': campaign.artifacts,
        'publication_receipt_bytes': campaign.receipts,
        'readiness_policy_bytes': policy_bytes,
        'expected_readiness_policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
        'readiness_manifest_bytes': readiness_manifest_bytes,
        'readiness_release_subject_bytes': subject_bytes,
        'readiness_evidence_artifact_bytes': {evidence_binding.sha256: evidence_artifact},
        'readiness_authority_public_key_bytes': {'external-release-auditor': authority_public_key},
        'verification_time_evidence_bytes': canonical_json_bytes(time_evidence),
        'verification_time_public_key_bytes': time_public_key,
        'verified_at': verified_at,
    }


def test_final_release_decision_composes_publication_and_readiness(tmp_path: Path) -> None:
    arguments = _decision(tmp_path)
    report = verify_tier_a_release_decision(**arguments)
    assert report.campaign_publication_verified
    assert report.readiness_authority_claims_verified
    assert report.exact_cross_component_subjects_verified
    assert report.cross_policy_signing_key_separation_verified
    assert report.cross_policy_declared_identity_separation_verified
    assert report.policy_authority_archive_byte_approval_composed
    assert not report.release_archive_semantics_verified_by_this_verifier
    assert not report.official_release_purpose_verified_by_this_verifier
    assert not report.deployment_facts_independently_observed_by_this_verifier
    assert not report.organizational_independence_cryptographically_proven

    with pytest.raises(ValueError, match='schema_version'):
        TierAReleaseDecisionReport.model_validate(
            {
                **report.model_dump(mode='json'),
                'schema_version': 'vaxreplay.tier-a-release-decision-report.v0.2',
            }
        )


def test_release_decision_uses_one_immutable_mapping_snapshot(tmp_path: Path) -> None:
    arguments = _decision(tmp_path)
    honest_artifacts = arguments['campaign_artifacts']
    honest_subjects = arguments['readiness_release_subject_bytes']
    assert isinstance(honest_artifacts, dict)
    assert isinstance(honest_subjects, dict)
    attacker_archive = b'attacker archive bytes never signed by the campaign'
    attacker_index = b'{"attacker":true}'
    later_artifacts = {
        **honest_artifacts,
        'archive': attacker_archive,
        'archive-index': attacker_index,
    }
    later_subjects = {
        **honest_subjects,
        'release-archive': attacker_archive,
        'release-archive-index': attacker_index,
    }
    artifact_mapping = _ChameleonBytesMapping(honest_artifacts, later_artifacts)
    subject_mapping = _ChameleonBytesMapping(honest_subjects, later_subjects)
    arguments['campaign_artifacts'] = artifact_mapping
    arguments['readiness_release_subject_bytes'] = subject_mapping

    report = verify_tier_a_release_decision(**arguments)

    assert report.release_archive_sha256 == hashlib.sha256(honest_artifacts['archive']).hexdigest()
    assert report.release_archive_index_sha256 == hashlib.sha256(honest_artifacts['archive-index']).hexdigest()
    assert artifact_mapping.indexed_reads == 0
    assert subject_mapping.indexed_reads == 0


def test_final_decision_rejects_wrong_out_of_band_campaign_trust_digest(
    tmp_path: Path,
) -> None:
    arguments = _decision(tmp_path)
    arguments['expected_campaign_trust_policy_sha256'] = 'f' * 64
    with pytest.raises(TierAReleaseDecisionError, match='out-of-band digest'):
        verify_tier_a_release_decision(**arguments)


def test_internally_valid_readiness_cannot_bind_a_different_campaign_subject(
    tmp_path: Path,
) -> None:
    arguments = _decision(tmp_path, wrong_trust_subject=True)
    with pytest.raises(TierAReleaseDecisionError, match='campaign-trust-policy'):
        verify_tier_a_release_decision(**arguments)


def test_readiness_release_id_and_signed_claim_time_must_match_publication(
    tmp_path: Path,
) -> None:
    with pytest.raises(TierAReleaseDecisionError, match='different releases'):
        verify_tier_a_release_decision(**_decision(tmp_path / 'wrong-id', readiness_release_id='different-release'))
    with pytest.raises(TierAReleaseDecisionError, match='predates'):
        verify_tier_a_release_decision(**_decision(tmp_path / 'old-claim', evidence_predates_release=True))


def test_release_signing_key_cannot_be_relabelled_as_external_readiness_authority(
    tmp_path: Path,
) -> None:
    with pytest.raises(TierAReleaseDecisionError, match='overlap campaign authority'):
        verify_tier_a_release_decision(**_decision(tmp_path, reuse_release_key_for_readiness=True))


def test_release_decision_cli_runs_the_complete_offline_composition(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    arguments = _decision(tmp_path / 'fixture')
    signed_path = tmp_path / 'campaign.json'
    campaign_trust_path = tmp_path / 'campaign-trust.json'
    readiness_policy_path = tmp_path / 'readiness-policy.json'
    readiness_manifest_path = tmp_path / 'readiness-manifest.json'
    signed_path.write_bytes(arguments['signed_campaign_manifest_bytes'])
    campaign_trust_path.write_bytes(arguments['campaign_trust_policy_bytes'])
    readiness_policy_path.write_bytes(arguments['readiness_policy_bytes'])
    readiness_manifest_path.write_bytes(arguments['readiness_manifest_bytes'])
    verification_time_path = tmp_path / 'verification-time.json'
    verification_time_key_path = tmp_path / 'verification-time.pub'
    verification_time_path.write_bytes(arguments['verification_time_evidence_bytes'])
    verification_time_key_path.write_bytes(arguments['verification_time_public_key_bytes'])

    signed = SignedCampaignPublicationManifest.model_validate_json(arguments['signed_campaign_manifest_bytes'])
    artifact_root = tmp_path / 'campaign-artifacts'
    artifact_root.mkdir()
    entries = []
    for artifact_id, payload in sorted(arguments['campaign_artifacts'].items()):
        path = artifact_root / f'{artifact_id}.bin'
        path.write_bytes(payload)
        entries.append(
            ArtifactPathEntry(
                artifact_id=artifact_id,
                path=f'campaign-artifacts/{path.name}',
            )
        )
    assert {item.artifact_id for item in signed.manifest.artifacts} == {item.artifact_id for item in entries}
    artifact_map_path = tmp_path / 'artifact-map.json'
    artifact_map_path.write_bytes(canonical_json_bytes(PublicationArtifactPathMap(artifacts=tuple(entries))))
    receipt_arguments: list[str] = []
    for index, payload in enumerate(arguments['publication_receipt_bytes'], start=1):
        path = tmp_path / f'receipt-{index}.json'
        path.write_bytes(payload)
        receipt_arguments.extend(('--publication-receipt', str(path)))
    subject_arguments: list[str] = []
    for role, payload in sorted(arguments['readiness_release_subject_bytes'].items()):
        path = tmp_path / f'subject-{role}.bin'
        path.write_bytes(payload)
        subject_arguments.extend(('--readiness-subject', f'{role}={path}'))
    evidence_root = tmp_path / 'readiness-evidence'
    evidence_root.mkdir()
    for digest, payload in arguments['readiness_evidence_artifact_bytes'].items():
        (evidence_root / digest).write_bytes(payload)
    key_arguments: list[str] = []
    for authority_id, payload in arguments['readiness_authority_public_key_bytes'].items():
        path = tmp_path / f'{authority_id}.pub'
        path.write_bytes(payload)
        key_arguments.extend(('--readiness-authority-key', f'{authority_id}={path}'))

    result = release_decision_main(
        [
            '--signed-campaign-manifest',
            str(signed_path),
            '--campaign-trust-policy',
            str(campaign_trust_path),
            '--expected-campaign-trust-policy-sha256',
            arguments['expected_campaign_trust_policy_sha256'],
            '--campaign-artifact-map',
            str(artifact_map_path),
            *receipt_arguments,
            '--readiness-policy',
            str(readiness_policy_path),
            '--expected-readiness-policy-sha256',
            arguments['expected_readiness_policy_sha256'],
            '--readiness-manifest',
            str(readiness_manifest_path),
            *subject_arguments,
            '--readiness-evidence-root',
            str(evidence_root),
            *key_arguments,
            '--verification-time-evidence',
            str(verification_time_path),
            '--verification-time-public-key',
            str(verification_time_key_path),
            '--verified-at',
            arguments['verified_at'].isoformat(),
        ]
    )
    assert result == 0
    report = json.loads(capfd.readouterr().out)
    assert report['release_id'] == signed.manifest.release_id
    assert report['exact_cross_component_subjects_verified'] is True
