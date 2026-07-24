"""Versioned contracts that bind data admission to sealed execution releases."""

from __future__ import annotations

import enum
import hashlib
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.temporal_schema import TemporalSourceTier

CHALLENGE_ADMISSION_SCHEMA_VERSION = 'vaxreplay.challenge-admission.v0.1'
PUBLIC_RELEASE_SCHEMA_VERSION = 'vaxreplay.public-release.v0.1'
PRIVATE_RELEASE_SCHEMA_VERSION = 'vaxreplay.private-release.v0.1'


class ReleasePurpose(str, enum.Enum):
    """Scientific claim permitted for a release artifact."""

    SYNTHETIC_INTEGRATION = 'synthetic_integration'
    RETROSPECTIVE_RESEARCH = 'retrospective_research'
    OFFICIAL_BENCHMARK = 'official_benchmark'


class ChallengeTemporalAdmissionBinding(StrictModel):
    episode_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_tier: TemporalSourceTier
    temporal_admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ChallengeAdmissionCommitment(StrictModel):
    """Public, non-secret commitment to private split and temporal admission records."""

    schema_version: Literal['vaxreplay.challenge-admission.v0.1'] = CHALLENGE_ADMISSION_SCHEMA_VERSION
    release_id: str = Field(min_length=1)
    purpose: ReleasePurpose
    split_admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    split_inventory_complete: bool
    case_universe_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    case_selection_audit_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    case_inventory_complete: bool = False
    verifier_policy_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    contamination_policy_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    contamination_audit_manifest_sha256: str | None = Field(
        default=None,
        pattern=r'^[0-9a-f]{64}$',
    )
    contamination_inventory_complete: bool = False
    episodes: tuple[ChallengeTemporalAdmissionBinding, ...] = Field(min_length=1)

    @field_validator('episodes')
    @classmethod
    def validate_episodes(
        cls,
        value: tuple[ChallengeTemporalAdmissionBinding, ...],
    ) -> tuple[ChallengeTemporalAdmissionBinding, ...]:
        episode_ids = tuple(binding.episode_id for binding in value)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('challenge admission episode IDs must be unique')
        if episode_ids != tuple(sorted(episode_ids)):
            raise ValueError('challenge admission episodes must be sorted by episode_id')
        return value

    @model_validator(mode='after')
    def validate_claim(self) -> Self:
        tiers = {binding.source_tier for binding in self.episodes}
        if self.purpose == ReleasePurpose.SYNTHETIC_INTEGRATION:
            if tiers != {TemporalSourceTier.TIER_C}:
                raise ValueError('synthetic integration releases require Tier C temporal admissions')
            if (
                any(
                    value is not None
                    for value in (
                        self.case_universe_sha256,
                        self.case_selection_audit_sha256,
                        self.verifier_policy_sha256,
                        self.contamination_policy_sha256,
                        self.contamination_audit_manifest_sha256,
                    )
                )
                or self.case_inventory_complete
                or self.contamination_inventory_complete
            ):
                raise ValueError('synthetic integration releases cannot claim sealed case or contamination inventories')
        elif self.purpose == ReleasePurpose.RETROSPECTIVE_RESEARCH:
            if tiers != {TemporalSourceTier.TIER_B} or not self.split_inventory_complete:
                raise ValueError('retrospective releases require Tier B and a complete split inventory')
            if (
                self.case_universe_sha256 is None
                or self.case_selection_audit_sha256 is None
                or self.verifier_policy_sha256 is None
                or not self.case_inventory_complete
                or self.contamination_policy_sha256 is None
                or self.contamination_audit_manifest_sha256 is None
                or not self.contamination_inventory_complete
            ):
                raise ValueError('retrospective releases require complete sealed case and contamination inventories')
        else:
            if tiers != {TemporalSourceTier.TIER_A} or not self.split_inventory_complete:
                raise ValueError('official releases require Tier A and a complete split inventory')
            if (
                self.case_universe_sha256 is None
                or self.case_selection_audit_sha256 is None
                or self.verifier_policy_sha256 is None
                or not self.case_inventory_complete
            ):
                raise ValueError('official releases require a complete sealed case inventory')
        return self


class ReleaseEpisodeBinding(StrictModel):
    ordinal: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    private_path: str = Field(pattern=r'^episodes/[0-9]{6}$')
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    labels_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_commitment_key_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    temporal_admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_tier: TemporalSourceTier
    source_audit_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class PrivateFileBinding(StrictModel):
    path: str = Field(
        pattern=r'^(?:episodes|temporal|protocols|proofs|source-audits|source-materials)/[A-Za-z0-9._/-]+$'
    )
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(ge=0)

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
            raise ValueError('private release file paths must be normalized and remain inside the package')
        return value


class PrivateReleaseManifest(StrictModel):
    """Organizer-only exact inventory used to reconstruct private scoring."""

    schema_version: Literal['vaxreplay.private-release.v0.1'] = PRIVATE_RELEASE_SCHEMA_VERSION
    release_id: str = Field(min_length=1)
    purpose: ReleasePurpose
    challenge_id: str = Field(min_length=1)
    challenge_bundle_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    suite_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    receipt_key_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    split_admission_path: Literal['split-admission.json'] = 'split-admission.json'
    split_admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    split_inventory_complete: bool
    case_universe_path: Literal['case-universe.json'] | None = None
    case_universe_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    case_universe_proof_path: Literal['case-universe-proof.bin'] | None = None
    case_selection_audit_path: Literal['case-selection-audit.json'] | None = None
    case_selection_audit_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    verifier_policy_path: Literal['verifier-policy.json'] | None = None
    verifier_policy_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    case_inventory_complete: bool = False
    contamination_audit_manifest_path: Literal['contamination-audit.json'] | None = None
    contamination_audit_manifest_sha256: str | None = Field(
        default=None,
        pattern=r'^[0-9a-f]{64}$',
    )
    contamination_inventory_complete: bool = False
    episodes: tuple[ReleaseEpisodeBinding, ...] = Field(min_length=1)
    files: tuple[PrivateFileBinding, ...] = Field(min_length=1)

    @field_validator('episodes')
    @classmethod
    def validate_episodes(cls, value: tuple[ReleaseEpisodeBinding, ...]) -> tuple[ReleaseEpisodeBinding, ...]:
        ordinals = tuple(binding.ordinal for binding in value)
        if ordinals != tuple(range(len(value))):
            raise ValueError('private release episode ordinals must be contiguous and start at zero')
        episode_ids = tuple(binding.episode_id for binding in value)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('private release episode IDs must be unique')
        return value

    @field_validator('files')
    @classmethod
    def validate_files(cls, value: tuple[PrivateFileBinding, ...]) -> tuple[PrivateFileBinding, ...]:
        paths = tuple(binding.path for binding in value)
        if paths != tuple(sorted(paths)):
            raise ValueError('private release file bindings must be sorted by path')
        if len(paths) != len(set(paths)):
            raise ValueError('private release file paths must be unique')
        return value

    @model_validator(mode='after')
    def validate_claim(self) -> Self:
        tiers = {binding.source_tier for binding in self.episodes}
        if self.purpose == ReleasePurpose.SYNTHETIC_INTEGRATION:
            if tiers != {TemporalSourceTier.TIER_C} or self.split_inventory_complete:
                raise ValueError('synthetic private releases require Tier C and an incomplete split inventory')
            if (
                any(
                    value is not None
                    for value in (
                        self.case_universe_path,
                        self.case_universe_sha256,
                        self.case_universe_proof_path,
                        self.case_selection_audit_path,
                        self.case_selection_audit_sha256,
                        self.verifier_policy_path,
                        self.verifier_policy_sha256,
                        self.contamination_audit_manifest_path,
                        self.contamination_audit_manifest_sha256,
                    )
                )
                or self.case_inventory_complete
                or self.contamination_inventory_complete
            ):
                raise ValueError('synthetic private releases cannot claim sealed case or contamination inventories')
        elif self.purpose == ReleasePurpose.RETROSPECTIVE_RESEARCH:
            if tiers != {TemporalSourceTier.TIER_B} or not self.split_inventory_complete:
                raise ValueError('retrospective private releases require Tier B and a complete split inventory')
            required_case_values = (
                self.case_universe_path,
                self.case_universe_sha256,
                self.case_universe_proof_path,
                self.case_selection_audit_path,
                self.case_selection_audit_sha256,
                self.verifier_policy_path,
                self.verifier_policy_sha256,
            )
            if any(value is None for value in required_case_values) or not self.case_inventory_complete:
                raise ValueError('retrospective private releases require a complete sealed case inventory')
            if (
                self.contamination_audit_manifest_path is None
                or self.contamination_audit_manifest_sha256 is None
                or not self.contamination_inventory_complete
            ):
                raise ValueError('retrospective private releases require a complete contamination audit inventory')
        else:
            if tiers != {TemporalSourceTier.TIER_A} or not self.split_inventory_complete:
                raise ValueError('official private releases require Tier A and a complete split inventory')
            required_case_values = (
                self.case_universe_path,
                self.case_universe_sha256,
                self.case_universe_proof_path,
                self.case_selection_audit_path,
                self.case_selection_audit_sha256,
                self.verifier_policy_path,
                self.verifier_policy_sha256,
            )
            if any(value is None for value in required_case_values) or not self.case_inventory_complete:
                raise ValueError('official private releases require a complete sealed case inventory')
        return self


class PublicReleaseManifest(StrictModel):
    """Preregisterable public identity for a challenge and its private scorer package."""

    schema_version: Literal['vaxreplay.public-release.v0.1'] = PUBLIC_RELEASE_SCHEMA_VERSION
    release_id: str = Field(min_length=1)
    purpose: ReleasePurpose
    sealed_eligible: bool
    challenge_path: Literal['challenge'] = 'challenge'
    challenge_id: str = Field(min_length=1)
    challenge_bundle_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    suite_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    admission_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    policy_path: Literal['policy.json'] = 'policy.json'
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contamination_policy_path: Literal['contamination-policy.json'] | None = None
    contamination_policy_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    receipt_key_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    private_package_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    episode_count: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_claim(self) -> Self:
        expected_sealed = self.purpose == ReleasePurpose.OFFICIAL_BENCHMARK
        if self.sealed_eligible != expected_sealed:
            raise ValueError('sealed_eligible must be true only for an official benchmark release')
        has_contamination_policy = self.contamination_policy_path is not None
        if has_contamination_policy != (self.contamination_policy_sha256 is not None):
            raise ValueError('contamination policy path and hash must be declared together')
        if self.purpose == ReleasePurpose.RETROSPECTIVE_RESEARCH and not has_contamination_policy:
            raise ValueError('retrospective public releases require a committed contamination policy')
        if self.purpose == ReleasePurpose.SYNTHETIC_INTEGRATION and has_contamination_policy:
            raise ValueError('synthetic integration releases cannot claim a contamination policy')
        return self


def release_model_sha256(value: StrictModel) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
