"""Compose authenticated campaign approval with prospective-release semantics.

The lower-level release-decision verifier authenticates the exact archive and index
bytes approved by campaign publication and readiness authorities.  The campaign
format deliberately treats those bytes as opaque.  This module closes that boundary
without treating a serialized decision report as a capability: it runs the decision
verifier and immediately passes the exact authenticated bytes and primitive digests
to the prospective campaign-archive verifier, which reconstructs and reloads the
release with caller-supplied trust verifiers.

This is a repository-level integrity and authorization composition.  It does not
create deployment facts, organizational independence, a prospective dataset, or an
official execution backend.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.campaign_publication import SignedCampaignPublicationManifest
from vaxreplay.operations.prospective_campaign_archive import (
    ProspectiveCampaignArchiveError,
    VerifiedProspectiveCampaignArchive,
    verify_and_materialize_prospective_campaign_archive,
)
from vaxreplay.operations.release_decision import (
    TierAReleaseDecisionError,
    TierAReleaseDecisionReport,
    verify_tier_a_release_decision,
)
from vaxreplay.operations.release_readiness import (
    TierAReleaseReadinessPolicy,
    TierAReleaseScope,
)
from vaxreplay.operations.schema import SAFE_ID_PATTERN
from vaxreplay.prospective_admission import CaseUniverseSealVerifier, SourceCaptureVerifier
from vaxreplay.temporal_schema import TemporalReceiptVerifier

TIER_A_PROSPECTIVE_RELEASE_APPROVAL_REPORT_SCHEMA_VERSION = 'vaxreplay.tier-a-prospective-release-approval-report.v0.2'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class TierAProspectiveReleaseApprovalError(ValueError):
    """The authenticated campaign decision did not yield one exact official release."""


class TierAProspectiveReleaseApprovalReport(StrictModel):
    """Serializable audit report; deliberately not an authorization capability."""

    schema_version: Literal['vaxreplay.tier-a-prospective-release-approval-report.v0.2'] = (
        TIER_A_PROSPECTIVE_RELEASE_APPROVAL_REPORT_SCHEMA_VERSION
    )
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_purpose: Literal['official_benchmark'] = 'official_benchmark'
    release_decision_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_campaign_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    campaign_trust_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_scope: TierAReleaseScope
    release_scope_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_archive_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    challenge_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_count: int = Field(gt=0)
    verified_at: datetime
    campaign_publication_and_readiness_verified: Literal[True] = True
    exact_archive_and_index_cross_binding_verified: Literal[True] = True
    archive_container_integrity_verified: Literal[True] = True
    prospective_release_semantics_reverified: Literal[True] = True
    official_benchmark_purpose_verified: Literal[True] = True
    deployment_facts_independently_observed_by_this_verifier: Literal[False] = False
    organizational_independence_cryptographically_proven: Literal[False] = False

    @model_validator(mode='after')
    def validate_release_scope_digest(self) -> TierAProspectiveReleaseApprovalReport:
        if self.release_scope_sha256 != _sha256(canonical_json_bytes(self.release_scope)):
            raise ValueError('release_scope_sha256 does not bind the canonical release scope')
        return self


@dataclass(frozen=True)
class VerifiedTierAProspectiveReleaseApproval:
    """Fresh in-process composition result, not a caller-transferable capability."""

    decision: TierAReleaseDecisionReport
    report: TierAProspectiveReleaseApprovalReport
    archive: VerifiedProspectiveCampaignArchive


def verify_and_materialize_tier_a_prospective_release(
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
    materialized_release_dir: Path,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> VerifiedTierAProspectiveReleaseApproval:
    """Verify approval and semantics in one fail-closed call over one byte snapshot.

    The expected campaign/readiness policy digests and verifier callbacks remain
    out-of-band trust inputs.  No previously serialized decision or readiness report
    is accepted.  The materialized release is installed only after the archive has
    identified an ``official_benchmark`` release and every prospective proof reloads.
    """

    artifacts = _freeze_bytes_mapping(campaign_artifacts, 'campaign artifacts')
    readiness_subjects = _freeze_bytes_mapping(
        readiness_release_subject_bytes,
        'readiness subjects',
    )
    evidence_artifacts = _freeze_bytes_mapping(
        readiness_evidence_artifact_bytes,
        'readiness evidence artifacts',
    )
    readiness_keys = _freeze_bytes_mapping(
        readiness_authority_public_key_bytes,
        'readiness authority keys',
    )
    receipts = _freeze_bytes_sequence(publication_receipt_bytes, 'publication receipts')
    try:
        decision = verify_tier_a_release_decision(
            signed_campaign_manifest_bytes=signed_campaign_manifest_bytes,
            campaign_trust_policy_bytes=campaign_trust_policy_bytes,
            expected_campaign_trust_policy_sha256=expected_campaign_trust_policy_sha256,
            campaign_artifacts=artifacts,
            publication_receipt_bytes=receipts,
            readiness_policy_bytes=readiness_policy_bytes,
            expected_readiness_policy_sha256=expected_readiness_policy_sha256,
            readiness_manifest_bytes=readiness_manifest_bytes,
            readiness_release_subject_bytes=readiness_subjects,
            readiness_evidence_artifact_bytes=evidence_artifacts,
            readiness_authority_public_key_bytes=readiness_keys,
            verification_time_evidence_bytes=verification_time_evidence_bytes,
            verification_time_public_key_bytes=verification_time_public_key_bytes,
            verified_at=verified_at,
        )
        signed_manifest = SignedCampaignPublicationManifest.model_validate_json(signed_campaign_manifest_bytes)
        readiness_policy = TierAReleaseReadinessPolicy.model_validate_json(readiness_policy_bytes)
    except (TierAReleaseDecisionError, TypeError, ValueError) as error:
        raise TierAProspectiveReleaseApprovalError(
            'campaign publication/readiness decision verification failed'
        ) from error
    if canonical_json_bytes(signed_manifest) != signed_campaign_manifest_bytes:
        raise TierAProspectiveReleaseApprovalError('signed campaign manifest is not canonical JSON')
    if canonical_json_bytes(readiness_policy) != readiness_policy_bytes:
        raise TierAProspectiveReleaseApprovalError('readiness policy is not canonical JSON')
    manifest = signed_manifest.manifest
    if manifest.release_id != decision.release_id or manifest.campaign_id != decision.campaign_id:
        raise TierAProspectiveReleaseApprovalError('campaign decision identity changed during composition')
    archive_bytes = artifacts[manifest.archive_artifact_id]
    index_bytes = artifacts[manifest.archive_index_artifact_id]
    if not hmac.compare_digest(_sha256(archive_bytes), decision.release_archive_sha256):
        raise TierAProspectiveReleaseApprovalError('campaign archive changed during decision composition')
    if not hmac.compare_digest(_sha256(index_bytes), decision.release_archive_index_sha256):
        raise TierAProspectiveReleaseApprovalError('campaign archive index changed during decision composition')

    try:
        archive = verify_and_materialize_prospective_campaign_archive(
            archive_bytes=archive_bytes,
            index_bytes=index_bytes,
            output_dir=materialized_release_dir,
            expected_archive_sha256=decision.release_archive_sha256,
            expected_index_sha256=decision.release_archive_index_sha256,
            expected_release_id=decision.release_id,
            expected_purpose='official_benchmark',
            expected_release_scope=readiness_policy.scope,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
        )
    except (ProspectiveCampaignArchiveError, TypeError, ValueError) as error:
        raise TierAProspectiveReleaseApprovalError(
            'authenticated campaign bytes are not one valid official prospective release'
        ) from error

    release = archive.release
    if (
        archive.archive_sha256 != decision.release_archive_sha256
        or archive.index_sha256 != decision.release_archive_index_sha256
        or archive.index.release_id != decision.release_id
        or release.manifest.release_id != decision.release_id
        or release.manifest.purpose != 'official_benchmark'
        or release.verified_admission.admission.purpose != 'official_benchmark'
        or release.release_sha256 != archive.index.prospective_release_manifest_sha256
        or archive.index.release_scope != readiness_policy.scope
    ):
        raise TierAProspectiveReleaseApprovalError(
            'materialized prospective release differs from its authenticated campaign decision'
        )

    report = TierAProspectiveReleaseApprovalReport(
        campaign_id=decision.campaign_id,
        release_id=decision.release_id,
        release_decision_report_sha256=_sha256(canonical_json_bytes(decision)),
        signed_campaign_manifest_sha256=decision.signed_campaign_manifest_sha256,
        campaign_trust_policy_sha256=decision.campaign_trust_policy_sha256,
        readiness_policy_sha256=decision.readiness_policy_sha256,
        readiness_manifest_sha256=decision.readiness_manifest_sha256,
        release_scope=readiness_policy.scope,
        release_scope_sha256=_sha256(canonical_json_bytes(readiness_policy.scope)),
        release_archive_sha256=archive.archive_sha256,
        release_archive_index_sha256=archive.index_sha256,
        prospective_release_sha256=release.release_sha256,
        release_tree_sha256=archive.index.tree_sha256,
        challenge_bundle_sha256=release.manifest.challenge.bundle_sha256,
        episode_count=release.manifest.episode_count,
        verified_at=decision.verified_at,
    )
    return VerifiedTierAProspectiveReleaseApproval(
        decision=decision,
        report=report,
        archive=archive,
    )


def _freeze_bytes_mapping(value: Mapping[str, bytes], label: str) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise TierAProspectiveReleaseApprovalError(f'{label} must be a mapping')
    try:
        items = tuple(value.items())
    except Exception as error:
        raise TierAProspectiveReleaseApprovalError(f'{label} could not be snapshotted') from error
    result: dict[str, bytes] = {}
    for key, payload in items:
        if not isinstance(key, str) or not isinstance(payload, bytes) or key in result:
            raise TierAProspectiveReleaseApprovalError(f'{label} must use unique string keys and exact bytes values')
        result[key] = payload
    return result


def _freeze_bytes_sequence(value: Sequence[bytes], label: str) -> tuple[bytes, ...]:
    if isinstance(value, (bytes, bytearray, str)) or not isinstance(value, Sequence):
        raise TierAProspectiveReleaseApprovalError(f'{label} must be a sequence of exact bytes')
    result = tuple(value)
    if not result or any(not isinstance(payload, bytes) for payload in result):
        raise TierAProspectiveReleaseApprovalError(f'{label} must be a nonempty sequence of exact bytes')
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    'TIER_A_PROSPECTIVE_RELEASE_APPROVAL_REPORT_SCHEMA_VERSION',
    'TierAProspectiveReleaseApprovalError',
    'TierAProspectiveReleaseApprovalReport',
    'VerifiedTierAProspectiveReleaseApproval',
    'verify_and_materialize_tier_a_prospective_release',
]
