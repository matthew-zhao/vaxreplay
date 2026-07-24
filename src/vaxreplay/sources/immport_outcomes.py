"""Prospective, label-free ImmPort arm-outcome contract and future exhaustive join.

Nothing in this module fetches an outcome, participant, subject, or result endpoint at decision
time.  The adjudication specification is frozen before capture and only commits definitions and
join rules.  A later, independently witnessed organizer-only manifest can be converted to labels
only after every target horizon has matured and every rankable intervention arm has exactly one
observed-or-censored disposition.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import OutcomeRecord, StrictModel
from vaxreplay.operations.schema import aware_utc
from vaxreplay.sources.immport import ImmportArmCandidateMap

IMMPORT_OUTCOME_SPEC_SCHEMA_VERSION = 'vaxreplay.immport-outcome-adjudication-spec.v0.1'
IMMPORT_OUTCOME_CAPTURE_SCHEMA_VERSION = 'vaxreplay.immport-outcome-capture.v0.1'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SAFE_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$'


class ImmportOutcomeContractError(ValueError):
    """A future outcome inventory does not satisfy its prospective contract."""


class ImmportOutcomeTargetDefinition(StrictModel):
    """Hash-only binding to a separately reviewed binary clinical endpoint definition."""

    target_id: str = Field(pattern=_SAFE_ID_PATTERN)
    horizon_days: int = Field(gt=0, le=3650)
    binary_success_definition_sha256: str = Field(pattern=_SHA256_PATTERN)
    utility_if_failure: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    utility_if_success: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_utility_order(self) -> Self:
        if self.utility_if_success <= self.utility_if_failure:
            raise ValueError('success utility must be strictly greater than failure utility')
        return self


class ImmportProspectiveOutcomeAdjudicationSpec(StrictModel):
    """Decision-time rules containing no future values or outcome-source identifiers."""

    schema_version: Literal['vaxreplay.immport-outcome-adjudication-spec.v0.1'] = IMMPORT_OUTCOME_SPEC_SCHEMA_VERSION
    spec_id: str = Field(pattern=_SAFE_ID_PATTERN)
    episode_id: str = Field(min_length=1, max_length=1024)
    task_type: Literal['early_clinical_arm_prioritization'] = 'early_clinical_arm_prioritization'
    decision_at: datetime
    adapter_policy_id: str = Field(pattern=_SAFE_ID_PATTERN)
    study_universe_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    outcome_unit: Literal['immport_trial_intervention_arm'] = 'immport_trial_intervention_arm'
    vaccine_construct_mapping: Literal['unverified_not_claimed_and_not_usable_for_construct_level_labels'] = (
        'unverified_not_claimed_and_not_usable_for_construct_level_labels'
    )
    decision_time_endpoint_access: Literal['participant_result_subject_and_download_endpoints_forbidden'] = (
        'participant_result_subject_and_download_endpoints_forbidden'
    )
    future_capture_semantics: Literal['separate_post_horizon_independently_witnessed_organizer_only_capture'] = (
        'separate_post_horizon_independently_witnessed_organizer_only_capture'
    )
    join_semantics: Literal['exact_organizer_candidate_map_intervention_arms_cross_product_targets'] = (
        'exact_organizer_candidate_map_intervention_arms_cross_product_targets'
    )
    adjudication_semantics: Literal['two_distinct_reviewers_consensus_or_precommitted_censor_reason'] = (
        'two_distinct_reviewers_consensus_or_precommitted_censor_reason'
    )
    # Core VaxReplay bundles currently carry one candidate utility shared across
    # every forecast target. Distinct target outcomes could imply inconsistent
    # utilities, so v0.1 admits exactly one endpoint/horizon. A future multi-target
    # contract must first precommit an explicit utility aggregation rule.
    targets: tuple[ImmportOutcomeTargetDefinition, ...] = Field(min_length=1, max_length=1)
    allowed_censor_reasons: tuple[
        Literal[
            'source_not_released_by_deadline',
            'endpoint_not_reported_by_deadline',
            'arm_withdrawn_before_endpoint',
            'adjudicators_cannot_reach_consensus',
        ],
        ...,
    ]

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ImmPort outcome decision_at')

    @field_validator('targets')
    @classmethod
    def validate_targets(
        cls,
        value: tuple[ImmportOutcomeTargetDefinition, ...],
    ) -> tuple[ImmportOutcomeTargetDefinition, ...]:
        keys = tuple((item.target_id, item.horizon_days) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('ImmPort outcome targets must be sorted and unique')
        return value

    @field_validator('allowed_censor_reasons')
    @classmethod
    def validate_censor_reasons(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value or value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('ImmPort censor reasons must be nonempty, sorted, and unique')
        return value


class ImmportFutureOutcomeDisposition(StrictModel):
    """One organizer-only future disposition; no participant-level field is representable."""

    candidate_id: str = Field(pattern=r'^cand-immport-[0-9a-f]{32}$')
    target_id: str = Field(pattern=_SAFE_ID_PATTERN)
    horizon_days: int = Field(gt=0, le=3650)
    status: Literal['observed', 'censored']
    binary_outcome: Literal[0, 1] | None = None
    censor_reason: str | None = Field(default=None, pattern=r'^[a-z][a-z0-9_]{0,99}$')
    revealed_at: datetime
    source_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    adjudication_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    adjudicator_ids: tuple[str, str]
    adjudicator_consensus: Literal[True] = True

    @field_validator('revealed_at')
    @classmethod
    def validate_revealed_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ImmPort future outcome revealed_at')

    @field_validator('adjudicator_ids')
    @classmethod
    def validate_adjudicators(cls, value: tuple[str, str]) -> tuple[str, str]:
        if value != tuple(sorted(value)) or len(set(value)) != 2:
            raise ValueError('future outcomes require two sorted, distinct adjudicator IDs')
        if any(not item or len(item) > 200 for item in value):
            raise ValueError('future outcome adjudicator IDs are invalid')
        return value

    @model_validator(mode='after')
    def validate_status(self) -> Self:
        if self.status == 'observed':
            if self.binary_outcome is None or self.censor_reason is not None:
                raise ValueError('observed ImmPort outcomes require only a binary value')
        elif self.binary_outcome is not None or self.censor_reason is None:
            raise ValueError('censored ImmPort outcomes require only a censor reason')
        return self


class ImmportFutureOutcomeCapture(StrictModel):
    """Exact future manifest that must exhaust the prespecified join universe."""

    schema_version: Literal['vaxreplay.immport-outcome-capture.v0.1'] = IMMPORT_OUTCOME_CAPTURE_SCHEMA_VERSION
    capture_id: str = Field(pattern=_SAFE_ID_PATTERN)
    episode_id: str = Field(min_length=1, max_length=1024)
    outcome_adjudication_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_candidate_map_sha256: str = Field(pattern=_SHA256_PATTERN)
    captured_at: datetime
    witnessed_at: datetime
    organizer_only: Literal[True] = True
    participant_level_data_persisted: Literal[False] = False
    dispositions: tuple[ImmportFutureOutcomeDisposition, ...] = Field(min_length=1)

    @field_validator('captured_at', 'witnessed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ImmPort outcome capture timestamp')

    @field_validator('dispositions')
    @classmethod
    def validate_dispositions(
        cls,
        value: tuple[ImmportFutureOutcomeDisposition, ...],
    ) -> tuple[ImmportFutureOutcomeDisposition, ...]:
        keys = tuple((item.candidate_id, item.target_id, item.horizon_days) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('future outcome dispositions must be sorted and unique')
        return value

    @model_validator(mode='after')
    def validate_capture_order(self) -> Self:
        if self.witnessed_at < self.captured_at:
            raise ValueError('future outcome witness cannot predate capture completion')
        if any(item.revealed_at > self.captured_at for item in self.dispositions):
            raise ValueError('future outcome disposition cannot be revealed after capture')
        return self


def immport_outcome_adjudication_spec_bytes(
    spec: ImmportProspectiveOutcomeAdjudicationSpec,
) -> bytes:
    if not isinstance(spec, ImmportProspectiveOutcomeAdjudicationSpec):
        raise TypeError('spec must be an ImmportProspectiveOutcomeAdjudicationSpec')
    return canonical_json_bytes(spec)


def verify_and_join_immport_future_outcomes(
    *,
    spec_bytes: bytes,
    candidate_map_bytes: bytes,
    capture_bytes: bytes,
) -> tuple[OutcomeRecord, ...]:
    """Verify a mature exhaustive future join and create ordinary VaxReplay labels."""

    spec = _parse_canonical(
        spec_bytes,
        ImmportProspectiveOutcomeAdjudicationSpec,
        'outcome adjudication spec',
    )
    candidate_map = _parse_canonical(
        candidate_map_bytes,
        ImmportArmCandidateMap,
        'organizer candidate map',
    )
    capture = _parse_canonical(capture_bytes, ImmportFutureOutcomeCapture, 'future outcome capture')
    spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
    map_sha256 = hashlib.sha256(candidate_map_bytes).hexdigest()
    if (
        spec.episode_id != candidate_map.episode_id
        or spec.adapter_policy_id != candidate_map.policy_id
        or spec.study_universe_registry_sha256 != candidate_map.study_universe_registry_sha256
        or capture.episode_id != spec.episode_id
        or capture.organizer_candidate_map_sha256 != map_sha256
        or capture.outcome_adjudication_spec_sha256 != spec_sha256
        or candidate_map.outcome_adjudication_spec_sha256 != spec_sha256
    ):
        raise ImmportOutcomeContractError(
            'future outcome capture, candidate map, and adjudication spec are not cross-bound'
        )
    intervention_ids = tuple(
        sorted(
            item.candidate_id
            for item in candidate_map.candidates
            if item.decision_disposition == 'rankable_intervention_arm'
        )
    )
    forbidden_ids = {
        item.candidate_id
        for item in candidate_map.candidates
        if item.decision_disposition != 'rankable_intervention_arm'
    }
    expected = {
        (candidate_id, target.target_id, target.horizon_days)
        for candidate_id in intervention_ids
        for target in spec.targets
    }
    actual = {(item.candidate_id, item.target_id, item.horizon_days) for item in capture.dispositions}
    if actual != expected or any(item.candidate_id in forbidden_ids for item in capture.dispositions):
        raise ImmportOutcomeContractError(
            'future outcome capture must exactly cover intervention arms by target without controls'
        )
    targets = {(item.target_id, item.horizon_days): item for item in spec.targets}
    labels: list[OutcomeRecord] = []
    for item in capture.dispositions:
        maturity = spec.decision_at + timedelta(days=item.horizon_days)
        if item.revealed_at < maturity or capture.captured_at < maturity:
            raise ImmportOutcomeContractError('future outcome was captured before its target matured')
        if item.censor_reason is not None and item.censor_reason not in spec.allowed_censor_reasons:
            raise ImmportOutcomeContractError('future outcome uses an uncommitted censor reason')
        target = targets[(item.target_id, item.horizon_days)]
        utility = target.utility_if_success if item.binary_outcome == 1 else target.utility_if_failure
        labels.append(
            OutcomeRecord(
                episode_id=spec.episode_id,
                candidate_id=item.candidate_id,
                target_id=item.target_id,
                horizon_days=item.horizon_days,
                outcome=item.binary_outcome,
                candidate_utility=utility,
                revealed_at=item.revealed_at,
                censor_reason=item.censor_reason,
            )
        )
    return tuple(labels)


def _parse_canonical[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    label: str,
) -> ModelT:
    if not isinstance(payload, bytes) or not payload:
        raise ImmportOutcomeContractError(f'ImmPort {label} must be nonempty exact bytes')
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise ImmportOutcomeContractError(f'invalid ImmPort {label}: {error}') from error
    if payload != canonical_json_bytes(value):
        raise ImmportOutcomeContractError(f'ImmPort {label} must use canonical JSON')
    return value
