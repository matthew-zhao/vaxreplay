"""Artifact-level composition of campaign publication and Tier-A readiness evidence.

Neither subsystem is sufficient alone: publication authenticates the release bill of
materials, while readiness authenticates external authority claims about a release.
This verifier requires those claims to name the exact signed campaign manifest, its
out-of-band trust policy, and its exact archive and index.  It does not independently
observe the claimed deployment or prove real-world organizational independence.
It also treats the authenticated release archive and index as exact opaque bytes. Use
``prospective_release_approval`` to parse, materialize, and reauthenticate those bytes
as the exact official prospective cohort release consumed downstream.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal

from pydantic import Field

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.campaign_publication import (
    CampaignPublicationError,
    CampaignPublicationManifest,
    CampaignPublicationTrustPolicy,
    SignedCampaignPublicationManifest,
    verify_campaign_publication,
)
from vaxreplay.operations.hermetic_execution import HermeticSandboxPolicy
from vaxreplay.operations.release_readiness import (
    ReleaseReadinessError,
    TierAReleaseReadinessManifest,
    TierAReleaseReadinessPolicy,
    verify_tier_a_release_readiness,
)
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc

TIER_A_RELEASE_DECISION_REPORT_SCHEMA_VERSION = 'vaxreplay.tier-a-release-decision-report.v0.3'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_REQUIRED_SUBJECT_ROLES = frozenset(
    {
        'campaign-signed-manifest',
        'campaign-trust-policy',
        'release-archive',
        'release-archive-index',
    }
)


class TierAReleaseDecisionError(ValueError):
    """The publication/readiness composition or one of its components failed closed."""


class TierAReleaseDecisionReport(StrictModel):
    schema_version: Literal['vaxreplay.tier-a-release-decision-report.v0.3'] = (
        TIER_A_RELEASE_DECISION_REPORT_SCHEMA_VERSION
    )
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    signed_campaign_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    campaign_trust_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_archive_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_at: datetime
    campaign_publication_verified: Literal[True] = True
    readiness_authority_claims_verified: Literal[True] = True
    exact_cross_component_subjects_verified: Literal[True] = True
    cross_policy_signing_key_separation_verified: Literal[True] = True
    cross_policy_declared_identity_separation_verified: Literal[True] = True
    policy_authority_archive_byte_approval_composed: Literal[True] = True
    release_archive_semantics_verified_by_this_verifier: Literal[False] = False
    official_release_purpose_verified_by_this_verifier: Literal[False] = False
    deployment_facts_independently_observed_by_this_verifier: Literal[False] = False
    organizational_independence_cryptographically_proven: Literal[False] = False


def verify_tier_a_release_decision(
    *,
    signed_campaign_manifest_bytes: bytes,
    campaign_trust_policy_bytes: bytes,
    expected_campaign_trust_policy_sha256: str,
    campaign_artifacts: Mapping[str, bytes],
    publication_receipt_bytes: Sequence[bytes],
    readiness_policy_bytes: bytes,
    expected_readiness_policy_sha256: str,
    readiness_manifest_bytes: bytes,
    readiness_release_subject_bytes: Mapping[str, bytes],
    readiness_evidence_artifact_bytes: Mapping[str, bytes],
    readiness_authority_public_key_bytes: Mapping[str, bytes],
    verification_time_evidence_bytes: bytes,
    verification_time_public_key_bytes: bytes,
    verified_at: datetime,
) -> TierAReleaseDecisionReport:
    """Verify both subsystems and exact authority-signed byte cross-bindings.

    This function does not parse the release archive. Its canonical report is an audit
    input, not a semantic-release capability. The composed prospective-release approval
    API reruns this verifier from the original inputs before materializing a release.
    """

    campaign_artifact_snapshot = _snapshot_bytes_mapping(campaign_artifacts, 'campaign artifacts')
    readiness_subject_snapshot = _snapshot_bytes_mapping(
        readiness_release_subject_bytes,
        'readiness release subjects',
    )
    readiness_evidence_snapshot = _snapshot_bytes_mapping(
        readiness_evidence_artifact_bytes,
        'readiness evidence artifacts',
    )
    readiness_key_snapshot = _snapshot_bytes_mapping(
        readiness_authority_public_key_bytes,
        'readiness authority public keys',
    )
    publication_receipt_snapshot = _snapshot_bytes_sequence(
        publication_receipt_bytes,
        'publication receipts',
    )
    _require_exact_bytes(signed_campaign_manifest_bytes, 'signed campaign manifest')
    _require_exact_bytes(campaign_trust_policy_bytes, 'campaign trust policy')
    _require_exact_bytes(readiness_policy_bytes, 'readiness policy')
    _require_exact_bytes(readiness_manifest_bytes, 'readiness manifest')
    _require_exact_bytes(verification_time_evidence_bytes, 'verification-time evidence')
    _require_exact_bytes(verification_time_public_key_bytes, 'verification-time public key')
    verification_time = aware_utc(verified_at, 'Tier A release decision verification time')
    campaign_trust_digest = hashlib.sha256(campaign_trust_policy_bytes).hexdigest()
    if not _matches_sha256(
        campaign_trust_digest,
        expected_campaign_trust_policy_sha256,
    ):
        raise TierAReleaseDecisionError('campaign trust policy differs from its out-of-band digest')
    try:
        signed_campaign = _canonical_model(
            signed_campaign_manifest_bytes,
            SignedCampaignPublicationManifest,
            'signed campaign manifest',
        )
        campaign_trust = _canonical_model(
            campaign_trust_policy_bytes,
            CampaignPublicationTrustPolicy,
            'campaign trust policy',
        )
        readiness_manifest = _canonical_model(
            readiness_manifest_bytes,
            TierAReleaseReadinessManifest,
            'readiness manifest',
        )
        readiness_policy = _canonical_model(
            readiness_policy_bytes,
            TierAReleaseReadinessPolicy,
            'readiness policy',
        )
        publication_report = verify_campaign_publication(
            signed_campaign_manifest_bytes,
            trust_policy_bytes=campaign_trust_policy_bytes,
            expected_trust_policy_sha256=expected_campaign_trust_policy_sha256,
            artifacts=campaign_artifact_snapshot,
            publication_receipt_bytes=publication_receipt_snapshot,
            verified_at=verification_time,
        )
        readiness_report = verify_tier_a_release_readiness(
            policy_bytes=readiness_policy_bytes,
            expected_policy_sha256=expected_readiness_policy_sha256,
            manifest_bytes=readiness_manifest_bytes,
            release_subject_bytes=readiness_subject_snapshot,
            evidence_artifact_bytes=readiness_evidence_snapshot,
            authority_public_key_bytes=readiness_key_snapshot,
            verification_time_evidence_bytes=verification_time_evidence_bytes,
            verification_time_public_key_bytes=verification_time_public_key_bytes,
            verified_at=verification_time,
        )
    except (CampaignPublicationError, ReleaseReadinessError, TypeError, ValueError):
        raise TierAReleaseDecisionError('Tier A release component verification failed') from None

    campaign_manifest = signed_campaign.manifest
    if (
        campaign_trust.campaign_id != campaign_manifest.campaign_id
        or readiness_manifest.release_id != campaign_manifest.release_id
        or readiness_report.release_id != campaign_manifest.release_id
        or readiness_manifest.created_at < campaign_manifest.created_at
    ):
        raise TierAReleaseDecisionError('publication and readiness identify different releases')
    _verify_cross_policy_authority_separation(
        campaign_trust=campaign_trust,
        campaign_manifest=campaign_manifest,
        campaign_artifacts=campaign_artifact_snapshot,
        readiness_policy=readiness_policy,
    )
    subjects = readiness_subject_snapshot
    if not _REQUIRED_SUBJECT_ROLES.issubset(subjects):
        raise TierAReleaseDecisionError('readiness omits required publication subjects')
    archive = campaign_artifact_snapshot[campaign_manifest.archive_artifact_id]
    archive_index = campaign_artifact_snapshot[campaign_manifest.archive_index_artifact_id]
    expected_subjects = {
        'campaign-signed-manifest': signed_campaign_manifest_bytes,
        'campaign-trust-policy': campaign_trust_policy_bytes,
        'release-archive': archive,
        'release-archive-index': archive_index,
    }
    for role, expected in expected_subjects.items():
        supplied = subjects[role]
        if not isinstance(supplied, bytes) or not hmac.compare_digest(supplied, expected):
            raise TierAReleaseDecisionError(f'readiness subject differs from exact campaign publication: {role}')
    if any(envelope.statement.issued_at < campaign_manifest.created_at for envelope in readiness_manifest.evidence):
        raise TierAReleaseDecisionError('readiness authority claim predates the signed campaign release')
    return TierAReleaseDecisionReport(
        campaign_id=campaign_manifest.campaign_id,
        release_id=campaign_manifest.release_id,
        signed_campaign_manifest_sha256=publication_report.signed_manifest_sha256,
        campaign_trust_policy_sha256=campaign_trust_digest,
        readiness_policy_sha256=readiness_report.policy_sha256,
        readiness_manifest_sha256=readiness_report.manifest_sha256,
        release_archive_sha256=hashlib.sha256(archive).hexdigest(),
        release_archive_index_sha256=hashlib.sha256(archive_index).hexdigest(),
        verified_at=verification_time,
    )


def _canonical_model[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        parsed = model.model_validate_json(payload)
    except (TypeError, ValueError):
        raise TierAReleaseDecisionError(f'{label} is invalid') from None
    if payload != canonical_json_bytes(parsed):
        raise TierAReleaseDecisionError(f'{label} is not canonical JSON')
    return parsed


def _matches_sha256(actual: str, expected: object) -> bool:
    return (
        isinstance(expected, str)
        and len(expected) == 64
        and all(character in '0123456789abcdef' for character in expected)
        and hmac.compare_digest(actual, expected)
    )


def _snapshot_bytes_mapping(value: Mapping[str, bytes], label: str) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise TierAReleaseDecisionError(f'{label} must be a mapping')
    try:
        items = tuple(value.items())
    except Exception as error:
        raise TierAReleaseDecisionError(f'{label} could not be snapshotted') from error
    result: dict[str, bytes] = {}
    for key, payload in items:
        if type(key) is not str or type(payload) is not bytes or key in result:
            raise TierAReleaseDecisionError(f'{label} must use unique exact-string keys and exact-bytes values')
        result[key] = payload
    return result


def _snapshot_bytes_sequence(value: Sequence[bytes], label: str) -> tuple[bytes, ...]:
    if isinstance(value, (bytes, bytearray, str)) or not isinstance(value, Sequence):
        raise TierAReleaseDecisionError(f'{label} must be a sequence of exact bytes')
    try:
        result = tuple(value)
    except Exception as error:
        raise TierAReleaseDecisionError(f'{label} could not be snapshotted') from error
    if not result or any(type(payload) is not bytes for payload in result):
        raise TierAReleaseDecisionError(f'{label} must be a nonempty sequence of exact bytes')
    return result


def _require_exact_bytes(value: object, label: str) -> None:
    if type(value) is not bytes or not value:
        raise TierAReleaseDecisionError(f'{label} must be nonempty exact bytes')


def _verify_cross_policy_authority_separation(
    *,
    campaign_trust: CampaignPublicationTrustPolicy,
    campaign_manifest: CampaignPublicationManifest,
    campaign_artifacts: Mapping[str, bytes],
    readiness_policy: TierAReleaseReadinessPolicy,
) -> None:
    if (
        readiness_policy.organizer_organization_id != campaign_trust.release_organization_id
        or readiness_policy.organizer_failure_domain_id != campaign_trust.release_failure_domain_id
    ):
        raise TierAReleaseDecisionError('readiness organizer identity differs from the campaign release authority')

    campaign_key_digests = {
        hashlib.sha256(base64.b64decode(key.public_key_base64, validate=True)).hexdigest()
        for key in campaign_trust.release_keys
    }
    campaign_authority_ids = {campaign_trust.release_authority_id}
    campaign_organization_ids = {campaign_trust.release_organization_id}
    campaign_failure_domain_ids = {campaign_trust.release_failure_domain_id}
    for authority in (
        *campaign_trust.worker_build_authorities,
        *campaign_trust.publication_authorities,
    ):
        campaign_authority_ids.add(authority.authority_id)
        campaign_organization_ids.add(authority.organization_id)
        campaign_failure_domain_ids.add(authority.failure_domain_id)
        campaign_key_digests.update(
            hashlib.sha256(base64.b64decode(key.public_key_base64, validate=True)).hexdigest() for key in authority.keys
        )
    campaign_authority_ids.update(
        {
            campaign_manifest.registry.authority_id,
            campaign_manifest.registry.registry_id,
            *(item.authority_id for item in campaign_manifest.witnesses),
            *(item.witness_id for item in campaign_manifest.witnesses),
            *(item.monitor_id for item in campaign_manifest.gossip.monitors),
            *(item.worker_id for item in campaign_manifest.workers),
        }
    )
    campaign_key_digests.update(campaign_manifest.registry.signing_public_key_sha256s)
    campaign_key_digests.update(item.signing_public_key_sha256 for item in campaign_manifest.witnesses)
    campaign_key_digests.update(item.report_signing_public_key_sha256 for item in campaign_manifest.gossip.monitors)
    for worker in campaign_manifest.workers:
        try:
            sandbox_bytes = campaign_artifacts[worker.sandbox_policy_artifact_id]
            sandbox = HermeticSandboxPolicy.model_validate_json(sandbox_bytes)
        except (KeyError, TypeError, ValueError):
            raise TierAReleaseDecisionError('worker sandbox authority cannot be cross-checked') from None
        if sandbox_bytes != canonical_json_bytes(sandbox):
            raise TierAReleaseDecisionError('worker sandbox authority is not canonical')
        campaign_authority_ids.add(sandbox.authority_id)
        campaign_key_digests.add(sandbox.signing_public_key_sha256)

    readiness_authorities = (
        *readiness_policy.authorities,
        readiness_policy.verification_time_authority,
    )
    if any(
        authority.authority_id in campaign_authority_ids
        or authority.organization_id in campaign_organization_ids
        or authority.failure_domain_id in campaign_failure_domain_ids
        or authority.signing_public_key_sha256 in campaign_key_digests
        for authority in readiness_authorities
    ):
        raise TierAReleaseDecisionError('readiness authorities overlap campaign authority identities or signing keys')


__all__ = [
    'TIER_A_RELEASE_DECISION_REPORT_SCHEMA_VERSION',
    'TierAReleaseDecisionError',
    'TierAReleaseDecisionReport',
    'verify_tier_a_release_decision',
]
