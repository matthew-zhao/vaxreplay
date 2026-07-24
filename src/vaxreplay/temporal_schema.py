"""Fail-closed temporal provenance and benchmark-admission contracts.

The admission envelope is deliberately separate from :class:`EpisodeManifest`.  A Tier A
decision receipt must exist before outcome values (and therefore before the final manifest) can
exist, while the completed envelope binds that receipt to the assembled episode manifest.
"""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import EpisodeBundle, body_sha256, canonical_json_bytes
from vaxreplay.case_schema import (
    EARLY_CLINICAL_ARM_PRIORITIZATION_TASK,
    PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK,
    RANKING_REWARD_VERSION,
    CandidateRecord,
    EpisodeManifest,
    EvidenceRecord,
    ForecastTarget,
    LabelCommitmentScheme,
    RewardVersion,
    Split,
    StrictModel,
    TaskType,
)

DECISION_CONFIG_SCHEMA_VERSION = 'vaxreplay.decision-config.v0.1'
DECISION_SNAPSHOT_SCHEMA_VERSION = 'vaxreplay.decision-snapshot.v0.1'
OUTCOME_SNAPSHOT_SCHEMA_VERSION = 'vaxreplay.outcome-snapshot.v0.1'
TEMPORAL_RECEIPT_SCHEMA_VERSION = 'vaxreplay.temporal-receipt.v0.1'
TEMPORAL_ADMISSION_SCHEMA_VERSION = 'vaxreplay.temporal-admission.v0.2'
PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION = 'vaxreplay.prospective-decision-context.v0.1'
CANDIDATE_ARTIFACT_SCHEMA_VERSION = 'vaxreplay.candidate-records-jsonl.v0.1'
EVIDENCE_ARTIFACT_SCHEMA_VERSION = 'vaxreplay.evidence-records-jsonl.v0.1'

PROTOCOL_ARTIFACT_NAMES = (
    'candidate_set_definition',
    'evidence_acquisition_spec',
    'outcome_adjudication_spec',
)


class TemporalSourceTier(str, enum.Enum):
    TIER_A = 'tier_a'
    TIER_B = 'tier_b'
    TIER_C = 'tier_c'


class TemporalAdmissionUse(str, enum.Enum):
    OFFICIAL_BENCHMARK = 'official_benchmark'
    RETROSPECTIVE_RESEARCH = 'retrospective_research'
    TRAIN_DEBUG = 'train_debug'


class TemporalProvenanceBasis(str, enum.Enum):
    PROSPECTIVE_SEAL = 'prospective_seal'
    INDEPENDENT_ARCHIVE = 'independent_archive'
    RETROSPECTIVE_RECONSTRUCTION = 'retrospective_reconstruction'


class TemporalArtifactRole(str, enum.Enum):
    CANDIDATE_UNIVERSE_OR_PANEL = 'candidate_universe_or_panel'
    EVIDENCE_SNAPSHOT = 'evidence_snapshot'
    DECISION_SNAPSHOT = 'decision_snapshot'
    OUTCOME_SNAPSHOT = 'outcome_snapshot'


class TemporalReceiptAuthority(str, enum.Enum):
    RFC3161_TIMESTAMP = 'rfc3161_timestamp'
    PUBLIC_TRANSPARENCY_LOG = 'public_transparency_log'
    SOURCE_SIGNED_VERSION = 'source_signed_version'
    INDEPENDENT_ARCHIVE = 'independent_archive'
    ORGANIZER_ATTESTATION = 'organizer_attestation'


_RECEIPT_ROLE_ORDER = (
    TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL,
    TemporalArtifactRole.EVIDENCE_SNAPSHOT,
    TemporalArtifactRole.DECISION_SNAPSHOT,
    TemporalArtifactRole.OUTCOME_SNAPSHOT,
)

_PROSPECTIVE_RECEIPT_AUTHORITIES = {
    TemporalReceiptAuthority.RFC3161_TIMESTAMP,
    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
}
_INDEPENDENT_RECEIPT_AUTHORITIES = {
    *_PROSPECTIVE_RECEIPT_AUTHORITIES,
    TemporalReceiptAuthority.SOURCE_SIGNED_VERSION,
    TemporalReceiptAuthority.INDEPENDENT_ARCHIVE,
}


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class DecisionTimeConfig(StrictModel):
    """Preregistered decision semantics, excluding future values and commitments."""

    schema_version: Literal['vaxreplay.decision-config.v0.1'] = DECISION_CONFIG_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    lineage_group_id: str = Field(min_length=1)
    synthetic: bool
    task_type: TaskType
    split: Split
    decision_at: datetime
    closed_book: Literal[True] = True
    network_allowed: Literal[False] = False
    portfolio_size: int = Field(gt=0)
    candidate_ids: tuple[str, ...] = Field(min_length=2)
    forecast_targets: tuple[ForecastTarget, ...] = Field(min_length=1)
    required_dimensions: tuple[str, ...] = Field(min_length=1)
    adjudication_version: str = Field(min_length=1)
    reward_version: RewardVersion

    @classmethod
    def from_manifest(cls, manifest: EpisodeManifest) -> DecisionTimeConfig:
        return cls(
            episode_id=manifest.episode_id,
            lineage_group_id=manifest.lineage_group_id,
            synthetic=manifest.synthetic,
            task_type=manifest.task_type,
            split=manifest.split,
            decision_at=manifest.decision_at,
            closed_book=manifest.closed_book,
            network_allowed=manifest.network_allowed,
            portfolio_size=manifest.portfolio_size,
            candidate_ids=tuple(manifest.candidate_ids),
            forecast_targets=tuple(manifest.forecast_targets),
            required_dimensions=tuple(manifest.required_dimensions),
            adjudication_version=manifest.adjudication_version,
            reward_version=manifest.reward_version,
        )

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return _require_aware(value, 'decision_at')

    @model_validator(mode='after')
    def validate_config(self) -> Self:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError('candidate_ids must be unique')
        if self.portfolio_size > len(self.candidate_ids):
            raise ValueError('portfolio_size cannot exceed the candidate count')
        if len(self.required_dimensions) != len(set(self.required_dimensions)):
            raise ValueError('required_dimensions must be unique')
        target_keys = tuple((target.target_id, target.horizon_days) for target in self.forecast_targets)
        if len(target_keys) != len(set(target_keys)):
            raise ValueError('forecast target and horizon pairs must be unique')
        if self.task_type == PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK and self.reward_version != RANKING_REWARD_VERSION:
            raise ValueError('preclinical candidate advancement requires the V1 ranking reward')
        if self.task_type == EARLY_CLINICAL_ARM_PRIORITIZATION_TASK and self.reward_version != RANKING_REWARD_VERSION:
            raise ValueError('early clinical arm prioritization requires the V1 ranking reward')
        return self


class DecisionProtocolCommitments(StrictModel):
    """Hashes of the full prespecified rules stored beside the compact episode bundle."""

    candidate_set_available_at: datetime
    candidate_set_definition_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    evidence_acquisition_spec_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    outcome_adjudication_spec_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @field_validator('candidate_set_available_at')
    @classmethod
    def validate_candidate_set_available_at(cls, value: datetime) -> datetime:
        return _require_aware(value, 'candidate_set_available_at')


class DecisionSnapshotCommitment(StrictModel):
    """Canonical pre-outcome commitment to the complete model-visible decision state."""

    schema_version: Literal['vaxreplay.decision-snapshot.v0.1'] = DECISION_SNAPSHOT_SCHEMA_VERSION
    config: DecisionTimeConfig
    protocol_commitments: DecisionProtocolCommitments
    candidate_universe_or_panel_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    candidate_universe_or_panel_bytes: int = Field(gt=0)
    visible_evidence_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    visible_evidence_bytes: int = Field(gt=0)
    latest_visible_evidence_at: datetime

    @field_validator('latest_visible_evidence_at')
    @classmethod
    def validate_latest_visible_evidence_at(cls, value: datetime) -> datetime:
        return _require_aware(value, 'latest_visible_evidence_at')

    @model_validator(mode='after')
    def validate_availability(self) -> Self:
        if self.protocol_commitments.candidate_set_available_at > self.config.decision_at:
            raise ValueError('candidate set cannot become available after decision_at')
        if self.latest_visible_evidence_at > self.config.decision_at:
            raise ValueError('visible evidence cannot become available after decision_at')
        return self


class OutcomeTargetAvailability(StrictModel):
    target_id: str = Field(min_length=1)
    horizon_days: int = Field(gt=0)
    first_label_available_at: datetime

    @field_validator('first_label_available_at')
    @classmethod
    def validate_first_label_available_at(cls, value: datetime) -> datetime:
        return _require_aware(value, 'first_label_available_at')


class OutcomeSnapshotCommitment(StrictModel):
    schema_version: Literal['vaxreplay.outcome-snapshot.v0.1'] = OUTCOME_SNAPSHOT_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    labels_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_commitment_scheme: LabelCommitmentScheme
    outcome_adjudication_spec_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    raw_outcome_source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    raw_outcome_source_bytes: int = Field(gt=0)
    label_derivation_audit_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_derivation_audit_bytes: int = Field(gt=0)
    target_availability: tuple[OutcomeTargetAvailability, ...] = Field(min_length=1)

    @field_validator('target_availability')
    @classmethod
    def validate_target_availability(
        cls,
        value: tuple[OutcomeTargetAvailability, ...],
    ) -> tuple[OutcomeTargetAvailability, ...]:
        keys = tuple((target.target_id, target.horizon_days) for target in value)
        if len(keys) != len(set(keys)):
            raise ValueError('outcome target availability keys must be unique')
        if keys != tuple(sorted(keys)):
            raise ValueError('outcome target availability must be sorted by target and horizon')
        return value

    @property
    def first_label_available_at(self) -> datetime:
        return min(target.first_label_available_at for target in self.target_availability)


class TemporalArtifactReceipt(StrictModel):
    schema_version: Literal['vaxreplay.temporal-receipt.v0.1'] = TEMPORAL_RECEIPT_SCHEMA_VERSION
    receipt_id: str = Field(min_length=1)
    role: TemporalArtifactRole
    artifact_schema_version: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    artifact_bytes: int = Field(gt=0)
    witnessed_at: datetime
    authority_type: TemporalReceiptAuthority
    authority_id: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    receipt_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _require_aware(value, 'witnessed_at')


class TemporalAdmissionEnvelope(StrictModel):
    """Post-outcome admission decision bound to pre-outcome receipts and the final episode."""

    schema_version: Literal['vaxreplay.temporal-admission.v0.2'] = TEMPORAL_ADMISSION_SCHEMA_VERSION
    admission_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_tier: TemporalSourceTier
    admitted_use: TemporalAdmissionUse
    provenance_basis: TemporalProvenanceBasis
    decision_snapshot: DecisionSnapshotCommitment
    decision_context_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    decision_context_bytes: int | None = Field(default=None, gt=0)
    outcome_snapshot: OutcomeSnapshotCommitment
    receipts: tuple[TemporalArtifactReceipt, ...] = Field(min_length=1, max_length=4)
    admitted_at: datetime

    @field_validator('admitted_at')
    @classmethod
    def validate_admitted_at(cls, value: datetime) -> datetime:
        return _require_aware(value, 'admitted_at')

    @model_validator(mode='after')
    def validate_admission(self) -> Self:
        config = self.decision_snapshot.config
        if config.episode_id != self.episode_id or self.outcome_snapshot.episode_id != self.episode_id:
            raise ValueError('decision and outcome snapshots must match the admitted episode_id')
        if self.outcome_snapshot.first_label_available_at <= config.decision_at:
            raise ValueError('outcome labels must first become available after decision_at')
        if (
            self.outcome_snapshot.outcome_adjudication_spec_sha256
            != self.decision_snapshot.protocol_commitments.outcome_adjudication_spec_sha256
        ):
            raise ValueError('outcome snapshot must use the prespecified adjudication contract')

        expected_targets = {
            (target.target_id, target.horizon_days): config.decision_at + timedelta(days=target.horizon_days)
            for target in config.forecast_targets
        }
        actual_targets = {
            (target.target_id, target.horizon_days): target.first_label_available_at
            for target in self.outcome_snapshot.target_availability
        }
        if actual_targets.keys() != expected_targets.keys():
            raise ValueError('outcome target availability must match every forecast target')
        if any(actual_targets[key] < maturity_at for key, maturity_at in expected_targets.items()):
            raise ValueError('outcome labels cannot become available before their forecast horizon matures')

        roles = tuple(receipt.role for receipt in self.receipts)
        canonical_subset = tuple(role for role in _RECEIPT_ROLE_ORDER if role in roles)
        if roles != canonical_subset:
            raise ValueError('receipts must use canonical artifact-role order without duplicates')
        if self.source_tier in {TemporalSourceTier.TIER_A, TemporalSourceTier.TIER_B} and roles != _RECEIPT_ROLE_ORDER:
            raise ValueError('Tier A and Tier B require all four artifact receipts')
        receipt_ids = tuple(receipt.receipt_id for receipt in self.receipts)
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError('receipt IDs must be unique')

        if (self.decision_context_sha256 is None) != (self.decision_context_bytes is None):
            raise ValueError('decision-context hash and byte count must be supplied together')
        if self.source_tier == TemporalSourceTier.TIER_A:
            decision_receipt = self.receipts[2]
            if decision_receipt.artifact_schema_version != PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION:
                raise ValueError('Tier A cannot use a legacy bare decision-snapshot receipt')
            if self.decision_context_sha256 is None or self.decision_context_bytes is None:
                raise ValueError('Tier A requires a lineage-bearing prospective decision context')

        expected_artifact_hashes = {
            TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: (
                self.decision_snapshot.candidate_universe_or_panel_sha256
            ),
            TemporalArtifactRole.EVIDENCE_SNAPSHOT: self.decision_snapshot.visible_evidence_sha256,
            TemporalArtifactRole.OUTCOME_SNAPSHOT: model_sha256(self.outcome_snapshot),
        }
        expected_artifact_sizes = {
            TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: (
                self.decision_snapshot.candidate_universe_or_panel_bytes
            ),
            TemporalArtifactRole.EVIDENCE_SNAPSHOT: self.decision_snapshot.visible_evidence_bytes,
            TemporalArtifactRole.OUTCOME_SNAPSHOT: len(canonical_json_bytes(self.outcome_snapshot)),
        }
        expected_artifact_schemas = {
            TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: CANDIDATE_ARTIFACT_SCHEMA_VERSION,
            TemporalArtifactRole.EVIDENCE_SNAPSHOT: EVIDENCE_ARTIFACT_SCHEMA_VERSION,
            TemporalArtifactRole.OUTCOME_SNAPSHOT: OUTCOME_SNAPSHOT_SCHEMA_VERSION,
        }
        for receipt in self.receipts:
            if receipt.role == TemporalArtifactRole.DECISION_SNAPSHOT:
                if receipt.artifact_schema_version == DECISION_SNAPSHOT_SCHEMA_VERSION:
                    if self.decision_context_sha256 is not None:
                        raise ValueError('bare decision-snapshot receipt cannot bind a declared decision context')
                    expected_hash = model_sha256(self.decision_snapshot)
                    expected_size = len(canonical_json_bytes(self.decision_snapshot))
                elif receipt.artifact_schema_version == PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION:
                    if self.decision_context_sha256 is None or self.decision_context_bytes is None:
                        raise ValueError('prospective decision-context receipt requires its declared commitment')
                    expected_hash = self.decision_context_sha256
                    expected_size = self.decision_context_bytes
                else:
                    raise ValueError(f'{receipt.role.value} receipt has the wrong artifact schema')
                if receipt.artifact_sha256 != expected_hash:
                    raise ValueError(f'{receipt.role.value} receipt does not bind the declared artifact')
                if receipt.artifact_bytes != expected_size:
                    raise ValueError(f'{receipt.role.value} receipt has the wrong artifact byte count')
                continue
            if receipt.artifact_sha256 != expected_artifact_hashes[receipt.role]:
                raise ValueError(f'{receipt.role.value} receipt does not bind the declared artifact')
            if receipt.artifact_bytes != expected_artifact_sizes[receipt.role]:
                raise ValueError(f'{receipt.role.value} receipt has the wrong artifact byte count')
            if receipt.artifact_schema_version != expected_artifact_schemas[receipt.role]:
                raise ValueError(f'{receipt.role.value} receipt has the wrong artifact schema')

        receipt_by_role = {receipt.role: receipt for receipt in self.receipts}
        outcome_receipt = receipt_by_role.get(TemporalArtifactRole.OUTCOME_SNAPSHOT)
        if (
            outcome_receipt is not None
            and outcome_receipt.witnessed_at < self.outcome_snapshot.first_label_available_at
        ):
            raise ValueError('outcome snapshot cannot be witnessed before labels are first available')
        if any(receipt.witnessed_at > self.admitted_at for receipt in self.receipts):
            raise ValueError('admitted_at cannot precede an artifact receipt')

        expected_profile = {
            TemporalSourceTier.TIER_A: (
                TemporalAdmissionUse.OFFICIAL_BENCHMARK,
                TemporalProvenanceBasis.PROSPECTIVE_SEAL,
            ),
            TemporalSourceTier.TIER_B: (
                TemporalAdmissionUse.RETROSPECTIVE_RESEARCH,
                TemporalProvenanceBasis.INDEPENDENT_ARCHIVE,
            ),
            TemporalSourceTier.TIER_C: (
                TemporalAdmissionUse.TRAIN_DEBUG,
                TemporalProvenanceBasis.RETROSPECTIVE_RECONSTRUCTION,
            ),
        }[self.source_tier]
        if (self.admitted_use, self.provenance_basis) != expected_profile:
            raise ValueError(f'{self.source_tier.value} has a fixed provenance basis and admitted use')

        if self.source_tier == TemporalSourceTier.TIER_A:
            assert outcome_receipt is not None
            for role in _RECEIPT_ROLE_ORDER[:3]:
                receipt = receipt_by_role[role]
                if receipt.witnessed_at > config.decision_at:
                    raise ValueError('Tier A decision-side receipts must be witnessed at or before decision_at')
                if receipt.witnessed_at >= self.outcome_snapshot.first_label_available_at:
                    raise ValueError('Tier A decision-side receipts must predate label availability')
                if receipt.authority_type not in _PROSPECTIVE_RECEIPT_AUTHORITIES:
                    raise ValueError('Tier A decision-side receipts require a prospective timestamp authority')
            candidate_receipt = receipt_by_role[TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL]
            evidence_receipt = receipt_by_role[TemporalArtifactRole.EVIDENCE_SNAPSHOT]
            decision_receipt = receipt_by_role[TemporalArtifactRole.DECISION_SNAPSHOT]
            if candidate_receipt.witnessed_at < self.decision_snapshot.protocol_commitments.candidate_set_available_at:
                raise ValueError('candidate receipt cannot predate candidate-set availability')
            if evidence_receipt.witnessed_at < self.decision_snapshot.latest_visible_evidence_at:
                raise ValueError('evidence receipt cannot predate included evidence availability')
            if decision_receipt.witnessed_at < max(
                candidate_receipt.witnessed_at,
                evidence_receipt.witnessed_at,
            ):
                raise ValueError('decision receipt cannot predate its candidate or evidence components')
            if outcome_receipt.authority_type not in _INDEPENDENT_RECEIPT_AUTHORITIES:
                raise ValueError('Tier A outcome receipt requires an independent authority')
        elif self.source_tier == TemporalSourceTier.TIER_B:
            assert outcome_receipt is not None
            if any(receipt.authority_type not in _INDEPENDENT_RECEIPT_AUTHORITIES for receipt in self.receipts):
                raise ValueError('Tier B receipts require an independent archive or authority')
        return self


class TemporalAdmissionError(ValueError):
    """Raised when an episode is not eligible for official benchmark use."""


type TemporalReceiptVerifier = Callable[[TemporalArtifactReceipt, bytes], bool]


def model_sha256(value: StrictModel) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_decision_snapshot_commitment(
    config: DecisionTimeConfig,
    candidates: Sequence[CandidateRecord],
    evidence: Sequence[EvidenceRecord],
    protocol_commitments: DecisionProtocolCommitments,
) -> DecisionSnapshotCommitment:
    """Commit only the state available for the decision, never future labels/evidence."""

    candidate_records = tuple(candidates)
    evidence_records = tuple(evidence)
    if not candidate_records:
        raise ValueError('candidate universe or panel cannot be empty')
    candidate_ids = tuple(record.candidate_id for record in candidate_records)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError('candidate records must have unique candidate IDs')
    eligible_ids = tuple(record.candidate_id for record in candidate_records if record.eligible)
    if eligible_ids != config.candidate_ids:
        raise ValueError('eligible candidate order must exactly match decision config candidate_ids')
    if any(record.episode_id != config.episode_id for record in candidate_records):
        raise ValueError('candidate records must match the decision config episode_id')

    evidence_ids = tuple(record.evidence_id for record in evidence_records)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError('evidence records must have unique evidence IDs')
    candidate_id_set = set(candidate_ids)
    for record in evidence_records:
        if record.episode_id != config.episode_id:
            raise ValueError('evidence records must match the decision config episode_id')
        if body_sha256(record.body) != record.body_sha256:
            raise ValueError(f'evidence body hash mismatch for {record.evidence_id}')
        if not set(record.related_candidate_ids).issubset(candidate_id_set):
            raise ValueError(f'evidence {record.evidence_id} references an unknown candidate')
    visible_evidence = tuple(record for record in evidence_records if record.available_at <= config.decision_at)
    if not visible_evidence:
        raise ValueError('decision snapshot requires at least one visible evidence record')

    candidate_bytes = _records_bytes(candidate_records)
    visible_evidence_bytes = _records_bytes(visible_evidence)

    return DecisionSnapshotCommitment(
        config=config,
        protocol_commitments=protocol_commitments,
        candidate_universe_or_panel_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        candidate_universe_or_panel_bytes=len(candidate_bytes),
        visible_evidence_sha256=hashlib.sha256(visible_evidence_bytes).hexdigest(),
        visible_evidence_bytes=len(visible_evidence_bytes),
        latest_visible_evidence_at=max(record.available_at for record in visible_evidence),
    )


def require_official_temporal_admission(
    admission: TemporalAdmissionEnvelope | None,
    bundle: EpisodeBundle,
    *,
    receipt_artifacts: Mapping[str, bytes],
    receipt_verifier: TemporalReceiptVerifier,
    protocol_artifacts: Mapping[str, bytes],
    raw_outcome_source: bytes,
    label_derivation_audit: bytes,
    expected_decision_context_sha256: str,
    expected_decision_context_bytes: int,
) -> TemporalAdmissionEnvelope:
    """Require Tier A provenance inside the private evaluator and verify every supplied artifact.

    ``receipt_verifier`` is organizer-controlled trusted code that must cryptographically verify the
    RFC 3161 token or transparency-log proof represented by each receipt artifact.  Requiring the
    callback and exact proof bytes keeps the generic schema independent of any one timestamp service
    while preventing callers from silently skipping proof verification.
    """

    return _require_temporal_admission(
        admission,
        bundle,
        expected_tier=TemporalSourceTier.TIER_A,
        admission_label='official benchmark',
        receipt_artifacts=receipt_artifacts,
        receipt_verifier=receipt_verifier,
        protocol_artifacts=protocol_artifacts,
        raw_outcome_source=raw_outcome_source,
        label_derivation_audit=label_derivation_audit,
        expected_decision_context_sha256=expected_decision_context_sha256,
        expected_decision_context_bytes=expected_decision_context_bytes,
    )


def require_retrospective_temporal_admission(
    admission: TemporalAdmissionEnvelope | None,
    bundle: EpisodeBundle,
    *,
    receipt_artifacts: Mapping[str, bytes],
    receipt_verifier: TemporalReceiptVerifier,
    protocol_artifacts: Mapping[str, bytes],
    raw_outcome_source: bytes,
    label_derivation_audit: bytes,
) -> TemporalAdmissionEnvelope:
    """Require a fully bound Tier B envelope for the retrospective research track.

    This verifies the same derived decision/outcome artifacts as the Tier A gate.  It does *not*
    by itself prove that archived literature existed at the historical cutoff; the literature
    adapter additionally verifies exact source bytes, independent archive proofs, a complete
    panel, and the label-blind decision-package seal before calling this gate.
    """

    return _require_temporal_admission(
        admission,
        bundle,
        expected_tier=TemporalSourceTier.TIER_B,
        admission_label='retrospective research',
        receipt_artifacts=receipt_artifacts,
        receipt_verifier=receipt_verifier,
        protocol_artifacts=protocol_artifacts,
        raw_outcome_source=raw_outcome_source,
        label_derivation_audit=label_derivation_audit,
        expected_decision_context_sha256=None,
        expected_decision_context_bytes=None,
    )


def _require_temporal_admission(
    admission: TemporalAdmissionEnvelope | None,
    bundle: EpisodeBundle,
    *,
    expected_tier: TemporalSourceTier,
    admission_label: str,
    receipt_artifacts: Mapping[str, bytes],
    receipt_verifier: TemporalReceiptVerifier,
    protocol_artifacts: Mapping[str, bytes],
    raw_outcome_source: bytes,
    label_derivation_audit: bytes,
    expected_decision_context_sha256: str | None,
    expected_decision_context_bytes: int | None,
) -> TemporalAdmissionEnvelope:
    if admission is None:
        raise TemporalAdmissionError(f'{admission_label} admission requires a temporal admission envelope')
    try:
        admission = TemporalAdmissionEnvelope.model_validate_json(canonical_json_bytes(admission))
    except ValueError as error:
        raise TemporalAdmissionError(f'invalid temporal admission envelope: {error}') from error
    if bundle.manifest.split != Split.TEST:
        raise TemporalAdmissionError(f'{admission_label} admission requires a test episode')
    if bundle.manifest.synthetic:
        raise TemporalAdmissionError(f'synthetic episodes cannot enter {admission_label}')
    if bundle.manifest.label_commitment_scheme != LabelCommitmentScheme.HMAC_SHA256:
        raise TemporalAdmissionError(f'{admission_label} admission requires an HMAC-SHA256 label commitment')
    if bundle.private_labels is None:
        raise TemporalAdmissionError(f'{admission_label} admission must run inside the private evaluator')
    try:
        bundle.validate_integrity()
    except ValueError as error:
        raise TemporalAdmissionError(f'episode bundle integrity failed: {error}') from error
    if admission.source_tier != expected_tier:
        raise TemporalAdmissionError(
            f'{admission_label} admission requires {expected_tier.value.replace("_", " ").title()} temporal provenance'
        )
    if expected_tier == TemporalSourceTier.TIER_A:
        if (
            expected_decision_context_sha256 is None
            or len(expected_decision_context_sha256) != 64
            or any(character not in '0123456789abcdef' for character in expected_decision_context_sha256)
            or expected_decision_context_bytes is None
            or expected_decision_context_bytes <= 0
        ):
            raise TemporalAdmissionError('official benchmark admission requires an exact prospective decision context')
        if (
            admission.decision_context_sha256 != expected_decision_context_sha256
            or admission.decision_context_bytes != expected_decision_context_bytes
        ):
            raise TemporalAdmissionError(
                'temporal admission decision context does not match the prospectively admitted source lineage'
            )
        decision_receipt = admission.receipts[2]
        if (
            decision_receipt.artifact_schema_version != PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION
            or decision_receipt.artifact_sha256 != expected_decision_context_sha256
            or decision_receipt.artifact_bytes != expected_decision_context_bytes
        ):
            raise TemporalAdmissionError(
                'official benchmark decision receipt does not bind the exact prospective decision context'
            )
    if admission.episode_id != bundle.manifest.episode_id or admission.manifest_sha256 != bundle.manifest_sha256:
        raise TemporalAdmissionError('temporal admission is not bound to the assembled episode manifest')
    if admission.outcome_snapshot.labels_sha256 != bundle.manifest.labels_sha256:
        raise TemporalAdmissionError('outcome snapshot does not match the episode label commitment')
    if admission.outcome_snapshot.label_commitment_scheme != bundle.manifest.label_commitment_scheme:
        raise TemporalAdmissionError('outcome snapshot has the wrong label commitment scheme')

    expected_decision_snapshot = build_decision_snapshot_commitment(
        DecisionTimeConfig.from_manifest(bundle.manifest),
        bundle.candidates,
        bundle.evidence,
        admission.decision_snapshot.protocol_commitments,
    )
    if admission.decision_snapshot != expected_decision_snapshot:
        raise TemporalAdmissionError('decision snapshot does not match the assembled episode decision state')

    expected_protocol_hashes = {
        'candidate_set_definition': (admission.decision_snapshot.protocol_commitments.candidate_set_definition_sha256),
        'evidence_acquisition_spec': (
            admission.decision_snapshot.protocol_commitments.evidence_acquisition_spec_sha256
        ),
        'outcome_adjudication_spec': (
            admission.decision_snapshot.protocol_commitments.outcome_adjudication_spec_sha256
        ),
    }
    if set(protocol_artifacts) != set(PROTOCOL_ARTIFACT_NAMES):
        raise TemporalAdmissionError(f'{admission_label} admission requires exactly the three protocol artifacts')
    for artifact_name, expected_sha256 in expected_protocol_hashes.items():
        if hashlib.sha256(protocol_artifacts[artifact_name]).hexdigest() != expected_sha256:
            raise TemporalAdmissionError(f'{artifact_name} does not match its decision-time commitment')

    outcome_snapshot = admission.outcome_snapshot
    if (
        len(raw_outcome_source) != outcome_snapshot.raw_outcome_source_bytes
        or hashlib.sha256(raw_outcome_source).hexdigest() != outcome_snapshot.raw_outcome_source_sha256
    ):
        raise TemporalAdmissionError('raw outcome source does not match the outcome snapshot')
    if (
        len(label_derivation_audit) != outcome_snapshot.label_derivation_audit_bytes
        or hashlib.sha256(label_derivation_audit).hexdigest() != outcome_snapshot.label_derivation_audit_sha256
    ):
        raise TemporalAdmissionError('label derivation audit does not match the outcome snapshot')

    availability_by_target: dict[tuple[str, int], datetime] = {}
    for outcome in bundle.private_labels.outcomes:
        key = (outcome.target_id, outcome.horizon_days)
        previous = availability_by_target.get(key)
        if previous is None or outcome.revealed_at < previous:
            availability_by_target[key] = outcome.revealed_at
    expected_availability = tuple(
        OutcomeTargetAvailability(
            target_id=target_id,
            horizon_days=horizon_days,
            first_label_available_at=availability_by_target[(target_id, horizon_days)],
        )
        for target_id, horizon_days in sorted(availability_by_target)
    )
    if outcome_snapshot.target_availability != expected_availability:
        raise TemporalAdmissionError('outcome target availability does not match private outcomes')

    receipt_by_id = {receipt.receipt_id: receipt for receipt in admission.receipts}
    if set(receipt_artifacts) != set(receipt_by_id):
        raise TemporalAdmissionError(f'{admission_label} admission requires exactly one proof artifact per receipt')
    for receipt_id, receipt in receipt_by_id.items():
        payload = receipt_artifacts[receipt_id]
        if len(payload) != receipt.receipt_bytes or hashlib.sha256(payload).hexdigest() != receipt.receipt_sha256:
            raise TemporalAdmissionError(f'receipt proof bytes do not match {receipt_id}')
        try:
            verified = receipt_verifier(receipt, payload)
        except Exception as error:
            raise TemporalAdmissionError(f'receipt verifier failed for {receipt_id}: {error}') from error
        if not verified:
            raise TemporalAdmissionError(f'receipt verifier rejected {receipt_id}')
    return admission


def _records_bytes(records: Sequence[StrictModel]) -> bytes:
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)
