"""Freshly derive the exact Tier A approval identity used by execution seals.

An approval report is an audit artifact, not a capability.  Consumers therefore
provide a typed, immutable snapshot of the original campaign/readiness inputs and
the existing proof-verifier trust roots.  This module invokes the concrete semantic
composition itself in a fresh temporary directory, reduces its result to a canonical
identity, and compares that identity with an out-of-band approval-report digest.
"""

from __future__ import annotations

import hashlib
import hmac
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.prospective_release_approval import (
    VerifiedTierAProspectiveReleaseApproval,
    verify_and_materialize_tier_a_prospective_release,
)
from vaxreplay.operations.release_readiness import TierAReleaseScope
from vaxreplay.operations.schema import SAFE_ID_PATTERN
from vaxreplay.prospective_admission import CaseUniverseSealVerifier, SourceCaptureVerifier
from vaxreplay.prospective_release import prospective_cohort_release_sha256
from vaxreplay.runner.prospective_challenge import prospective_challenge_bundle_sha256
from vaxreplay.temporal_schema import TemporalReceiptVerifier

TIER_A_PROSPECTIVE_RELEASE_APPROVAL_IDENTITY_SCHEMA_VERSION = (
    'vaxreplay.tier-a-prospective-release-approval-identity.v0.1'
)
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class TierAProspectiveReleaseApprovalIdentityError(ValueError):
    """The freshly reverified approval differs from its pinned release identity."""


class TierAProspectiveReleaseApprovalIdentity(StrictModel):
    """Canonical authorization identity bound into the pre-run release seal."""

    schema_version: Literal['vaxreplay.tier-a-prospective-release-approval-identity.v0.1'] = (
        TIER_A_PROSPECTIVE_RELEASE_APPROVAL_IDENTITY_SCHEMA_VERSION
    )
    approval_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_scope: TierAReleaseScope
    release_scope_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_decision_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_campaign_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    campaign_trust_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_archive_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    challenge_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_count: int = Field(gt=0)
    verified_at: datetime

    @field_validator('verified_at')
    @classmethod
    def validate_verified_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('approval verified_at must include a UTC offset')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def validate_scope(self) -> TierAProspectiveReleaseApprovalIdentity:
        if self.release_scope_sha256 != _sha256(canonical_json_bytes(self.release_scope)):
            raise ValueError('release_scope_sha256 does not bind the canonical release scope')
        if not self.release_scope.includes_model_leaderboard:
            raise ValueError('official model execution requires model-leaderboard readiness gates')
        return self


@dataclass(frozen=True)
class TierAProspectiveReleaseApprovalReplay:
    """Original semantic-approval inputs, never a serialized approval capability.

    Byte collections are snapshotted on construction.  Reverification invokes the
    concrete composite verifier directly and materializes into a fresh temporary
    directory.  The three callback fields are the existing cryptographic/proof
    trust roots, not callbacks that can manufacture an approval result.
    """

    signed_campaign_manifest_bytes: bytes
    campaign_trust_policy_bytes: bytes
    expected_campaign_trust_policy_sha256: str
    campaign_artifacts: Mapping[str, bytes]
    publication_receipt_bytes: Sequence[bytes]
    readiness_policy_bytes: bytes
    expected_readiness_policy_sha256: str
    readiness_manifest_bytes: bytes
    readiness_release_subject_bytes: Mapping[str, bytes]
    readiness_evidence_artifact_bytes: Mapping[str, bytes]
    readiness_authority_public_key_bytes: Mapping[str, bytes]
    verification_time_evidence_bytes: bytes
    verification_time_public_key_bytes: bytes
    verified_at: datetime
    decision_receipt_verifier: TemporalReceiptVerifier
    case_universe_seal_verifier: CaseUniverseSealVerifier
    source_capture_verifier: SourceCaptureVerifier
    materialization_parent: Path | None = None

    def __post_init__(self) -> None:
        byte_fields = (
            'signed_campaign_manifest_bytes',
            'campaign_trust_policy_bytes',
            'readiness_policy_bytes',
            'readiness_manifest_bytes',
            'verification_time_evidence_bytes',
            'verification_time_public_key_bytes',
        )
        for field_name in byte_fields:
            value = getattr(self, field_name)
            if not isinstance(value, bytes) or not value:
                raise TypeError(f'{field_name} must be nonempty exact bytes')
        for field_name in (
            'campaign_artifacts',
            'readiness_release_subject_bytes',
            'readiness_evidence_artifact_bytes',
            'readiness_authority_public_key_bytes',
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(_snapshot_bytes_mapping(getattr(self, field_name), field_name)),
            )
        receipts = tuple(self.publication_receipt_bytes)
        if not receipts or any(not isinstance(payload, bytes) or not payload for payload in receipts):
            raise TypeError('publication_receipt_bytes must contain nonempty exact bytes')
        object.__setattr__(self, 'publication_receipt_bytes', receipts)
        _require_sha256(self.expected_campaign_trust_policy_sha256, 'expected campaign trust policy')
        _require_sha256(self.expected_readiness_policy_sha256, 'expected readiness policy')
        if self.materialization_parent is not None:
            object.__setattr__(self, 'materialization_parent', Path(self.materialization_parent))


def reverify_tier_a_prospective_release_approval_identity(
    *,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
) -> TierAProspectiveReleaseApprovalIdentity:
    """Rerun a trusted verifier and reduce its internally cross-checked result.

    ``expected_approval_report_sha256`` is an out-of-band authorization pin.  The
    concrete semantic verifier is rerun here from ``approval_replay``'s original
    authority inputs on every invocation.  Stored report bytes and caller-provided
    approval dataclasses are not accepted.
    """

    _require_sha256(expected_approval_report_sha256, 'expected approval report')
    if not isinstance(approval_replay, TierAProspectiveReleaseApprovalReplay):
        raise TierAProspectiveReleaseApprovalIdentityError(
            'the original Tier A prospective-release approval replay is required'
        )
    try:
        parent = approval_replay.materialization_parent
        if parent is not None:
            parent = parent.expanduser().resolve(strict=True)
            if not parent.is_dir():
                raise ValueError('approval replay materialization parent is not a directory')
        with tempfile.TemporaryDirectory(prefix='vaxreplay-approval-replay-', dir=parent) as temporary_directory:
            approval = verify_and_materialize_tier_a_prospective_release(
                signed_campaign_manifest_bytes=approval_replay.signed_campaign_manifest_bytes,
                campaign_trust_policy_bytes=approval_replay.campaign_trust_policy_bytes,
                expected_campaign_trust_policy_sha256=approval_replay.expected_campaign_trust_policy_sha256,
                campaign_artifacts=approval_replay.campaign_artifacts,
                publication_receipt_bytes=approval_replay.publication_receipt_bytes,
                readiness_policy_bytes=approval_replay.readiness_policy_bytes,
                expected_readiness_policy_sha256=approval_replay.expected_readiness_policy_sha256,
                readiness_manifest_bytes=approval_replay.readiness_manifest_bytes,
                readiness_release_subject_bytes=approval_replay.readiness_release_subject_bytes,
                readiness_evidence_artifact_bytes=approval_replay.readiness_evidence_artifact_bytes,
                readiness_authority_public_key_bytes=approval_replay.readiness_authority_public_key_bytes,
                verification_time_evidence_bytes=approval_replay.verification_time_evidence_bytes,
                verification_time_public_key_bytes=approval_replay.verification_time_public_key_bytes,
                verified_at=approval_replay.verified_at,
                materialized_release_dir=Path(temporary_directory) / 'release',
                decision_receipt_verifier=approval_replay.decision_receipt_verifier,
                case_universe_seal_verifier=approval_replay.case_universe_seal_verifier,
                source_capture_verifier=approval_replay.source_capture_verifier,
            )
            return _approval_identity(
                approval,
                expected_approval_report_sha256=expected_approval_report_sha256,
            )
    except Exception as error:
        if isinstance(error, TierAProspectiveReleaseApprovalIdentityError):
            raise
        raise TierAProspectiveReleaseApprovalIdentityError(
            f'Tier A prospective-release approval replay failed: {error}'
        ) from error


def _approval_identity(
    approval: VerifiedTierAProspectiveReleaseApproval,
    *,
    expected_approval_report_sha256: str,
) -> TierAProspectiveReleaseApprovalIdentity:
    decision = approval.decision
    report = approval.report
    archive = approval.archive
    release = archive.release
    index = archive.index
    report_digest = _sha256(canonical_json_bytes(report))
    if not hmac.compare_digest(report_digest, expected_approval_report_sha256):
        raise TierAProspectiveReleaseApprovalIdentityError('fresh approval report differs from its out-of-band digest')

    # Defense in depth against an implementation regression inside the concrete
    # composite verifier.
    decision_digest = _sha256(canonical_json_bytes(decision))
    release_scope = report.release_scope
    scope_digest = _sha256(canonical_json_bytes(release_scope))
    if (
        report.release_decision_report_sha256 != decision_digest
        or report.campaign_id != decision.campaign_id
        or report.release_id != decision.release_id
        or report.signed_campaign_manifest_sha256 != decision.signed_campaign_manifest_sha256
        or report.campaign_trust_policy_sha256 != decision.campaign_trust_policy_sha256
        or report.readiness_policy_sha256 != decision.readiness_policy_sha256
        or report.readiness_manifest_sha256 != decision.readiness_manifest_sha256
        or report.release_archive_sha256 != decision.release_archive_sha256
        or report.release_archive_index_sha256 != decision.release_archive_index_sha256
        or report.verified_at != decision.verified_at
        or report.release_scope_sha256 != scope_digest
        or archive.archive_sha256 != report.release_archive_sha256
        or archive.index_sha256 != report.release_archive_index_sha256
        or archive.index_sha256 != _sha256(canonical_json_bytes(index))
        or index.archive_sha256 != archive.archive_sha256
        or index.release_id != report.release_id
        or index.release_purpose != 'official_benchmark'
        or index.release_scope != release_scope
        or index.release_scope_sha256 != scope_digest
        or index.prospective_release_manifest_sha256 != report.prospective_release_sha256
        or index.release_json_sha256 != _sha256(canonical_json_bytes(release.manifest))
        or index.tree_sha256 != report.release_tree_sha256
        or release.manifest.release_id != report.release_id
        or release.manifest.purpose != 'official_benchmark'
        or release.verified_admission.admission.purpose != 'official_benchmark'
        or release.release_sha256 != report.prospective_release_sha256
        or release.release_sha256 != prospective_cohort_release_sha256(release.manifest)
        or release.manifest.challenge.bundle_sha256 != report.challenge_bundle_sha256
        or release.challenge.manifest_sha256 != prospective_challenge_bundle_sha256(release.challenge.manifest)
        or release.challenge.manifest_sha256 != report.challenge_bundle_sha256
        or release.manifest.episode_count != report.episode_count
        or not release.challenge.authority_proofs_reverified
    ):
        raise TierAProspectiveReleaseApprovalIdentityError(
            'concrete approval replay returned internally inconsistent release authorization'
        )
    try:
        return TierAProspectiveReleaseApprovalIdentity(
            approval_report_sha256=report_digest,
            campaign_id=report.campaign_id,
            release_id=report.release_id,
            release_scope=release_scope,
            release_scope_sha256=scope_digest,
            release_decision_report_sha256=decision_digest,
            signed_campaign_manifest_sha256=report.signed_campaign_manifest_sha256,
            campaign_trust_policy_sha256=report.campaign_trust_policy_sha256,
            readiness_policy_sha256=report.readiness_policy_sha256,
            readiness_manifest_sha256=report.readiness_manifest_sha256,
            release_archive_sha256=report.release_archive_sha256,
            release_archive_index_sha256=report.release_archive_index_sha256,
            prospective_release_sha256=report.prospective_release_sha256,
            release_tree_sha256=report.release_tree_sha256,
            challenge_bundle_sha256=report.challenge_bundle_sha256,
            episode_count=report.episode_count,
            verified_at=report.verified_at,
        )
    except ValueError as error:
        raise TierAProspectiveReleaseApprovalIdentityError(
            f'fresh approval cannot authorize official model execution: {error}'
        ) from error


def tier_a_prospective_release_approval_identity_sha256(
    identity: TierAProspectiveReleaseApprovalIdentity,
) -> str:
    return _sha256(canonical_json_bytes(identity))


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise TierAProspectiveReleaseApprovalIdentityError(f'{label} digest must be exact lowercase SHA-256')


def _snapshot_bytes_mapping(value: object, label: str) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping')
    try:
        items = tuple(value.items())
    except Exception as error:
        raise TypeError(f'{label} could not be snapshotted') from error
    result: dict[str, bytes] = {}
    for key, payload in items:
        if not isinstance(key, str) or not isinstance(payload, bytes) or not payload or key in result:
            raise TypeError(f'{label} must use unique string keys and nonempty exact bytes values')
        result[key] = payload
    if not result:
        raise TypeError(f'{label} must not be empty')
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    'TIER_A_PROSPECTIVE_RELEASE_APPROVAL_IDENTITY_SCHEMA_VERSION',
    'TierAProspectiveReleaseApprovalIdentity',
    'TierAProspectiveReleaseApprovalIdentityError',
    'TierAProspectiveReleaseApprovalReplay',
    'reverify_tier_a_prospective_release_approval_identity',
    'tier_a_prospective_release_approval_identity_sha256',
]
