"""Organizer-only derivation of execution-task private gold from audited AACT material.

This adapter is intentionally downstream of cohort selection.  It independently rebuilds the
decision-only relevance queue from exact historical decision archives before it inspects the later
label material, and it refuses caller-authored outcome summaries.  The output remains a
development-only, unadmitted private artifact; this module does not build a public workspace or
claim identity masking, sealed execution, source-verified public tasks, or preregistration.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_adapter import ArtifactReceipt
from vaxreplay.clinicaltrials.execution_inventory import (
    ExecutionInventoryError,
    audit_execution_inventory,
    audit_execution_label_set,
)
from vaxreplay.clinicaltrials.execution_merge import (
    AactExecutionMultiAnchorMergeReceipt,
    SourceExecutionBuildReceipt,
)
from vaxreplay.clinicaltrials.execution_schema import (
    ENROLLMENT_RATIO_DECIMAL_PLACES,
    EXECUTION_TASK_ID,
    AactExecutionDecisionRow,
    AactExecutionOutcomeRow,
    ExecutionCohortInventory,
    ExecutionLabelSet,
    ObservationState,
    RegistryOutcomeClass,
    RegistryStatus,
    RegistryValueType,
    TrialAnchorAssignment,
    observation_state,
    registry_outcome_class,
)
from vaxreplay.clinicaltrials.execution_task import (
    EXECUTION_REWARD_VERSION,
    ContinuousForecastSpec,
    ExecutionPrivateGold,
    ExecutionTaskContext,
    execution_task_context_sha256,
)
from vaxreplay.clinicaltrials.lineage_split import (
    LINEAGE_SPLIT_POLICY,
    LineageCaseAssignment,
    LineageSplitAssignmentSet,
    LineageSplitReceipt,
)
from vaxreplay.clinicaltrials.relevance_adjudication import (
    ACTIVE_VACCINE_RELEVANCE_POLICY,
    RelevanceDisposition,
    RelevanceReviewError,
    VaccineRelevanceAdjudicationSet,
    VaccineRelevanceReviewQueue,
    VaccineRelevanceReviewReceipt,
    build_relevance_review_queue,
    finalize_relevance_adjudications,
)

GOLD_DERIVATION_SCHEMA_VERSION = 'vaxreplay.aact-execution-gold-derivation.v0.1'
GOLD_DERIVATION_ADAPTER_ID = 'aact-audited-label-to-private-gold-v0.1'
GOLD_COHORT_DERIVATION_SCHEMA_VERSION = 'vaxreplay.aact-execution-gold-cohort-derivation.v0.1'
GOLD_COHORT_DERIVATION_ADAPTER_ID = 'aact-audited-label-to-private-gold-cohort-v0.1'
GOLD_COHORT_TARGET_SET_SCHEMA_VERSION = 'vaxreplay.aact-execution-gold-cohort-targets.v0.1'
GOLD_COHORT_PRIVATE_SET_SCHEMA_VERSION = 'vaxreplay.aact-execution-private-gold-set.dev-v0.1'
FORECAST_SPEC_POLICY_SCHEMA_VERSION = 'vaxreplay.aact-execution-forecast-spec-policy.v0.1'
FORECAST_SPEC_POLICY_ID = 'aact-registry-execution-fixed-forecast-spec.dev-v0.1'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class ExecutionGoldDerivationError(ValueError):
    """Trusted source material cannot deterministically produce the requested private gold."""


class ExecutionForecastSpecPolicy(StrictModel):
    """Release-fixed scoring bounds that are independent of every realized outcome."""

    schema_version: Literal['vaxreplay.aact-execution-forecast-spec-policy.v0.1'] = FORECAST_SPEC_POLICY_SCHEMA_VERSION
    policy_id: Literal['aact-registry-execution-fixed-forecast-spec.dev-v0.1'] = FORECAST_SPEC_POLICY_ID
    task_id: Literal['registry_observed_trial_execution'] = EXECUTION_TASK_ID
    reward_version: Literal['vaxreplay.clinical-execution-reward.dev-v0.1'] = EXECUTION_REWARD_VERSION
    enrollment_ratio_spec: ContinuousForecastSpec
    primary_completion_slippage_days_spec: ContinuousForecastSpec
    fixed_before_private_outcomes_opened: Literal[True] = True
    outcome_values_used_to_choose_bounds: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False


def execution_forecast_spec_policy() -> ExecutionForecastSpecPolicy:
    """Return the only forecast-spec policy accepted by this adapter version."""

    return ExecutionForecastSpecPolicy(
        enrollment_ratio_spec=ContinuousForecastSpec(
            forecast_kind='point',
            lower_bound=0.0,
            upper_bound=2.0,
        ),
        primary_completion_slippage_days_spec=ContinuousForecastSpec(
            forecast_kind='quantiles',
            lower_bound=-365.0,
            upper_bound=730.0,
            quantile_levels=(0.1, 0.5, 0.9),
        ),
    )


def execution_forecast_spec_policy_sha256() -> str:
    return _model_sha256(execution_forecast_spec_policy())


class TrustedSourceBuildHashes(StrictModel):
    """External trust anchors for one source build named by the merge receipt."""

    anchor_date: date
    build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    label_archive_sha256: str = Field(pattern=_SHA256_PATTERN)


class TrustedExecutionGoldSourceHashes(StrictModel):
    """Hashes pinned outside the caller-controlled models passed to the derivation adapter.

    Merge artifacts are written by ``execution_merge`` as canonical JSON plus one newline.  The
    relevance models are hashed as canonical JSON, matching their internal queue/adjudication
    bindings.  Keeping these hashes as a separate input makes stale-hash model-copy attacks fail.
    """

    merge_receipt_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    inventory_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    label_set_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_queue_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_adjudication_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_builds: tuple[TrustedSourceBuildHashes, ...] = Field(min_length=2)

    @model_validator(mode='after')
    def validate_sources(self) -> Self:
        anchors = tuple(item.anchor_date for item in self.source_builds)
        if anchors != tuple(sorted(set(anchors))):
            raise ValueError('trusted source-build hashes must have unique ascending anchors')
        return self


class ExecutionGoldSourceObservation(StrictModel):
    """Raw normalized registry inputs retained beside all deterministic derived values."""

    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    assignment_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_snapshot_id: str = Field(min_length=1)
    anchor_date: date
    decision_archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_row_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_source_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    planned_enrollment: int = Field(gt=0)
    planned_primary_completion_date: date
    label_snapshot_id: str = Field(min_length=1)
    label_archive_date: date
    label_archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    outcome_row_sha256: str = Field(pattern=_SHA256_PATTERN)
    outcome_source_record_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    registry_record_present: bool
    raw_overall_status: RegistryStatus | None
    raw_enrollment: int | None = Field(default=None, ge=0)
    raw_enrollment_type: RegistryValueType | None
    raw_primary_completion_date: date | None
    raw_primary_completion_date_type: RegistryValueType | None
    registry_outcome_class: RegistryOutcomeClass
    enrollment_observation: ObservationState
    enrollment_ratio: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    primary_completion_observation: ObservationState
    primary_completion_slippage_days: int | None

    @model_validator(mode='after')
    def validate_derivations(self) -> Self:
        raw_values = (
            self.outcome_source_record_sha256,
            self.raw_overall_status,
            self.raw_enrollment,
            self.raw_enrollment_type,
            self.raw_primary_completion_date,
            self.raw_primary_completion_date_type,
        )
        if not self.registry_record_present and any(value is not None for value in raw_values):
            raise ValueError('a missing later registry record cannot carry raw observed fields')
        if self.registry_record_present != (self.outcome_source_record_sha256 is not None):
            raise ValueError('later record presence must match its source-row hash')
        if (self.raw_enrollment is None) != (self.raw_enrollment_type is None):
            raise ValueError('raw enrollment and type must be present together')
        if (self.raw_primary_completion_date is None) != (self.raw_primary_completion_date_type is None):
            raise ValueError('raw primary-completion date and type must be present together')
        expected_outcome = registry_outcome_class(self.registry_record_present, self.raw_overall_status)
        if self.registry_outcome_class != expected_outcome:
            raise ValueError('registry outcome class is not derived from the retained raw status')

        actual_enrollment = (
            self.raw_enrollment
            if self.registry_record_present and self.raw_enrollment_type == RegistryValueType.ACTUAL
            else None
        )
        expected_enrollment_state = observation_state(
            self.registry_record_present,
            actual_enrollment,
            self.raw_enrollment_type,
        )
        if self.enrollment_observation != expected_enrollment_state:
            raise ValueError('enrollment observation is not derived from the retained raw enrollment')
        expected_ratio = (
            round(actual_enrollment / self.planned_enrollment, ENROLLMENT_RATIO_DECIMAL_PLACES)
            if actual_enrollment is not None
            else None
        )
        if expected_ratio is None:
            if self.enrollment_ratio is not None:
                raise ValueError('enrollment ratio exists without an observed Actual enrollment')
        elif self.enrollment_ratio is None or not math.isclose(
            self.enrollment_ratio,
            expected_ratio,
            rel_tol=0.0,
            abs_tol=10**-ENROLLMENT_RATIO_DECIMAL_PLACES,
        ):
            raise ValueError('enrollment ratio is not recomputed from retained actual/planned enrollment')

        actual_completion = (
            self.raw_primary_completion_date
            if self.registry_record_present and self.raw_primary_completion_date_type == RegistryValueType.ACTUAL
            else None
        )
        expected_completion_state = observation_state(
            self.registry_record_present,
            actual_completion,
            self.raw_primary_completion_date_type,
        )
        if self.primary_completion_observation != expected_completion_state:
            raise ValueError('completion observation is not derived from the retained raw date')
        expected_slippage = (
            (actual_completion - self.planned_primary_completion_date).days if actual_completion is not None else None
        )
        if self.primary_completion_slippage_days != expected_slippage:
            raise ValueError('completion slippage is not recomputed from retained actual/planned dates')
        return self


class ExecutionGoldRelevanceBinding(StrictModel):
    policy_id: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    review_method_id: str = Field(min_length=1)
    review_queue_sha256: str = Field(pattern=_SHA256_PATTERN)
    adjudication_sha256: str = Field(pattern=_SHA256_PATTERN)
    authenticated_review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_sha256: str = Field(pattern=_SHA256_PATTERN)
    disposition: Literal['include'] = 'include'
    decision_only_queue_rebuilt_from_exact_archives: Literal[True] = True


class ExecutionGoldSourceDerivationReceipt(StrictModel):
    schema_version: Literal['vaxreplay.aact-execution-gold-derivation.v0.1'] = GOLD_DERIVATION_SCHEMA_VERSION
    adapter_id: Literal['aact-audited-label-to-private-gold-v0.1'] = GOLD_DERIVATION_ADAPTER_ID
    organizer_private_nct_id: str = Field(pattern=r'^NCT\d{8}$')
    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    trusted_source_hashes: TrustedExecutionGoldSourceHashes
    inventory_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    label_set_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    merge_receipt_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    forecast_spec_policy_id: Literal['aact-registry-execution-fixed-forecast-spec.dev-v0.1'] = FORECAST_SPEC_POLICY_ID
    forecast_spec_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_observation: ExecutionGoldSourceObservation
    relevance: ExecutionGoldRelevanceBinding
    private_gold_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_private: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    public_workspace_created: Literal[False] = False
    identity_masking_claimed: Literal[False] = False
    sealed_execution_claimed: Literal[False] = False
    later_data_used_for_selection: Literal[False] = False
    existing_dev_contract_release_flags_changed: Literal[False] = False
    derivation_recomputed_and_verified: Literal[True] = True

    @model_validator(mode='after')
    def validate_fixed_bindings(self) -> Self:
        if self.forecast_spec_policy_sha256 != execution_forecast_spec_policy_sha256():
            raise ValueError('source derivation does not bind the adapter-fixed forecast-spec policy')
        if self.organizer_private_nct_id != self.source_observation.nct_id:
            raise ValueError('source observation does not belong to the private NCT target')
        return self


@dataclass(frozen=True)
class ExecutionGoldDerivation:
    receipt: ExecutionGoldSourceDerivationReceipt
    private_gold: ExecutionPrivateGold


@dataclass(frozen=True)
class ExecutionGoldDerivationBuild:
    root: Path
    derivation: ExecutionGoldDerivation
    artifacts: tuple[ArtifactReceipt, ...]


class ExecutionGoldCohortTarget(StrictModel):
    """Organizer-private mapping from one real registry identity to its task context."""

    organizer_private_nct_id: str = Field(pattern=r'^NCT\d{8}$')
    context: ExecutionTaskContext


class ExecutionGoldCohortTargetSet(StrictModel):
    """Exact caller-provided contexts for every selected trial.

    Contexts are inputs rather than synthesized by this adapter because the final context hash must
    cover the outcome-blind workspace documents.  Until that workspace build exists, a target set
    may contain empty-document contexts, but the resulting artifact remains explicitly intermediate.
    """

    schema_version: Literal['vaxreplay.aact-execution-gold-cohort-targets.v0.1'] = GOLD_COHORT_TARGET_SET_SCHEMA_VERSION
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    targets: tuple[ExecutionGoldCohortTarget, ...] = Field(min_length=1)
    organizer_private: Literal[True] = True
    development_only: Literal[True] = True
    final_workspace_contexts_bound: bool = False
    public_workspace_created: Literal[False] = False
    identity_masking_claimed: Literal[False] = False
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_targets(self) -> Self:
        keys = tuple((item.context.anchor_date, item.organizer_private_nct_id) for item in self.targets)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('cohort targets must have unique ascending (anchor_date, nct_id) keys')
        episode_ids = tuple(item.context.episode_id for item in self.targets)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('cohort target episode IDs must be unique')
        documents_bound = all(bool(item.context.cutoff_documents) for item in self.targets)
        if self.final_workspace_contexts_bound != documents_bound:
            raise ValueError(
                'final_workspace_contexts_bound must be true exactly when every target context binds documents'
            )
        return self


class ExecutionPrivateGoldSet(StrictModel):
    """Organizer-private gold records for an exactly covered execution cohort."""

    schema_version: Literal['vaxreplay.aact-execution-private-gold-set.dev-v0.1'] = (
        GOLD_COHORT_PRIVATE_SET_SCHEMA_VERSION
    )
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    records: tuple[ExecutionPrivateGold, ...] = Field(min_length=1)
    organizer_private: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_records(self) -> Self:
        keys = tuple(item.organizer_private_nct_id for item in self.records)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('private gold records must have unique ascending NCT IDs')
        episode_ids = tuple(item.episode_id for item in self.records)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('private gold episode IDs must be unique')
        return self


class ExecutionGoldCohortSplitBinding(StrictModel):
    """Exact organizer-private lineage/split artifact used by this gold cohort."""

    split_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_assignments_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    id_key_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    assignment_count: int = Field(gt=0)
    lineage_count: int = Field(gt=0)
    include_coverage_complete: Literal[True] = True
    lineage_split_isolated: Literal[True] = True
    outcome_conditioned_grouping_or_split: Literal[False] = False
    public_ids_currently_safe_to_release: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False


class ExecutionGoldCohortCaseReceipt(StrictModel):
    """Compact per-case receipt; cohort-global trust anchors are stored only once."""

    organizer_private_nct_id: str = Field(pattern=r'^NCT\d{8}$')
    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_observation: ExecutionGoldSourceObservation
    relevance_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_decision_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_case_assignment_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    lineage_group_id: str | None = Field(default=None, min_length=1)
    split: Split | None = None
    private_gold_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_identity(self) -> Self:
        if self.organizer_private_nct_id != self.source_observation.nct_id:
            raise ValueError('cohort case receipt identity does not match its source observation')
        split_values = (self.lineage_case_assignment_sha256, self.lineage_group_id, self.split)
        if any(value is None for value in split_values) != all(value is None for value in split_values):
            raise ValueError('cohort case split fields must be all present or all absent')
        return self


class ExecutionGoldCohortSourceDerivationReceipt(StrictModel):
    """One authenticated receipt for the full organizer-private gold derivation."""

    schema_version: Literal['vaxreplay.aact-execution-gold-cohort-derivation.v0.1'] = (
        GOLD_COHORT_DERIVATION_SCHEMA_VERSION
    )
    adapter_id: Literal['aact-audited-label-to-private-gold-cohort-v0.1'] = GOLD_COHORT_DERIVATION_ADAPTER_ID
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    trusted_source_hashes: TrustedExecutionGoldSourceHashes
    inventory_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    label_set_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    merge_receipt_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    forecast_spec_policy_id: Literal['aact-registry-execution-fixed-forecast-spec.dev-v0.1'] = FORECAST_SPEC_POLICY_ID
    forecast_spec_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_policy_id: str = Field(min_length=1)
    relevance_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_review_method_id: str = Field(min_length=1)
    relevance_queue_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_adjudication_sha256: str = Field(pattern=_SHA256_PATTERN)
    authenticated_review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    eligible_include_count: int = Field(gt=0)
    derived_case_count: int = Field(gt=0)
    case_receipts: tuple[ExecutionGoldCohortCaseReceipt, ...] = Field(min_length=1)
    private_gold_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_binding: ExecutionGoldCohortSplitBinding | None = None
    source_models_verified_once: Literal[True] = True
    exact_include_coverage_verified: Literal[True] = True
    duplicate_missing_extra_cases_rejected: Literal[True] = True
    every_raw_outcome_recomputed: Literal[True] = True
    organizer_private: Literal[True] = True
    development_only: Literal[True] = True
    final_workspace_contexts_bound: bool = False
    split_inventory_bound: bool = False
    lineage_split_safe: bool = False
    leaderboard_admitted: Literal[False] = False
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    public_workspace_created: Literal[False] = False
    identity_masking_claimed: Literal[False] = False
    sealed_execution_claimed: Literal[False] = False
    later_data_used_for_selection: Literal[False] = False
    derivation_recomputed_and_verified: Literal[True] = True

    @model_validator(mode='after')
    def validate_cohort(self) -> Self:
        if self.forecast_spec_policy_sha256 != execution_forecast_spec_policy_sha256():
            raise ValueError('cohort derivation does not bind the adapter-fixed forecast-spec policy')
        if self.eligible_include_count != self.derived_case_count or self.derived_case_count != len(self.case_receipts):
            raise ValueError('cohort derivation must exactly cover every relevance INCLUDE case')
        keys = tuple(
            (item.source_observation.anchor_date, item.organizer_private_nct_id) for item in self.case_receipts
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('cohort case receipts must have unique ascending (anchor_date, nct_id) keys')
        cases_bind_split = all(item.split is not None for item in self.case_receipts)
        expected_bound = self.split_binding is not None
        if self.split_inventory_bound != expected_bound or self.lineage_split_safe != expected_bound:
            raise ValueError('split binding and split-safety flags must agree')
        if cases_bind_split != expected_bound:
            raise ValueError('every case must bind the finalized split exactly when the cohort does')
        if self.split_binding is not None:
            if self.split_binding.assignment_count != len(self.case_receipts):
                raise ValueError('split assignment count does not match cohort cases')
            if self.split_binding.lineage_count != len({item.lineage_group_id for item in self.case_receipts}):
                raise ValueError('split lineage count does not match cohort cases')
        return self


@dataclass(frozen=True)
class ExecutionGoldCohortDerivation:
    receipt: ExecutionGoldCohortSourceDerivationReceipt
    targets: ExecutionGoldCohortTargetSet
    private_gold: ExecutionPrivateGoldSet


@dataclass(frozen=True)
class ExecutionGoldCohortDerivationBuild:
    root: Path
    derivation: ExecutionGoldCohortDerivation
    artifacts: tuple[ArtifactReceipt, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_sha256(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _merge_artifact_sha256(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value) + b'\n')


def _artifact_by_path(receipt: AactExecutionMultiAnchorMergeReceipt, relative_path: str) -> ArtifactReceipt:
    matches = tuple(item for item in receipt.artifacts if item.relative_path == relative_path)
    if len(matches) != 1:
        raise ExecutionGoldDerivationError(f'merge receipt must bind exactly one {relative_path} artifact')
    return matches[0]


def _verify_merge_sources(
    *,
    inventory: ExecutionCohortInventory,
    merge_receipt: AactExecutionMultiAnchorMergeReceipt,
    trusted: TrustedExecutionGoldSourceHashes,
) -> dict[date, SourceExecutionBuildReceipt]:
    if _merge_artifact_sha256(merge_receipt) != trusted.merge_receipt_artifact_sha256:
        raise ExecutionGoldDerivationError('merge receipt does not match its external trusted hash')
    inventory_artifact = _artifact_by_path(merge_receipt, 'organizer/cohort-inventory.json')
    computed_inventory_artifact = _merge_artifact_sha256(inventory)
    if (
        inventory_artifact.sha256 != computed_inventory_artifact
        or trusted.inventory_artifact_sha256 != computed_inventory_artifact
    ):
        raise ExecutionGoldDerivationError('merged inventory does not match receipt and external trusted hash')
    if inventory_artifact.byte_count != len(canonical_json_bytes(inventory) + b'\n'):
        raise ExecutionGoldDerivationError('merged inventory byte count does not match receipt')

    receipt_sources = tuple(merge_receipt.source_builds)
    trusted_sources = tuple(trusted.source_builds)
    if len(receipt_sources) != len(inventory.policy.anchors) or len(receipt_sources) != len(trusted_sources):
        raise ExecutionGoldDerivationError('merge, policy, and externally trusted source counts disagree')
    by_anchor: dict[date, SourceExecutionBuildReceipt] = {}
    for source, trusted_source, anchor in zip(
        receipt_sources,
        trusted_sources,
        inventory.policy.anchors,
        strict=True,
    ):
        if source.anchor_date != trusted_source.anchor_date or source.anchor_date != anchor.anchor_date:
            raise ExecutionGoldDerivationError('source-build anchor order disagrees with policy or trust anchors')
        if (
            source.build_receipt_sha256 != trusted_source.build_receipt_sha256
            or source.decision_archive_sha256 != trusted_source.decision_archive_sha256
            or source.label_archive_sha256 != trusted_source.label_archive_sha256
        ):
            raise ExecutionGoldDerivationError('merge source build does not match its external trusted source hashes')
        if (
            source.decision_archive_sha256 != anchor.decision_archive_manifest_sha256
            or source.label_archive_sha256 != anchor.label_archive_manifest_sha256
            or source.label_archive_date != anchor.label_archive_date
        ):
            raise ExecutionGoldDerivationError('merge source archives do not match the exact policy anchor binding')
        by_anchor[source.anchor_date] = source
    if merge_receipt.mechanical_assignment_count != len(inventory.assignments):
        raise ExecutionGoldDerivationError('merge receipt assignment count does not match audited inventory')
    return by_anchor


def _verify_label_source(
    *,
    label_set: ExecutionLabelSet,
    merge_receipt: AactExecutionMultiAnchorMergeReceipt,
    trusted: TrustedExecutionGoldSourceHashes,
) -> None:
    label_artifact = _artifact_by_path(merge_receipt, 'private/execution-labels.json')
    computed_label_artifact = _merge_artifact_sha256(label_set)
    if label_artifact.sha256 != computed_label_artifact or trusted.label_set_artifact_sha256 != computed_label_artifact:
        raise ExecutionGoldDerivationError('merged label set does not match receipt and external trusted hash')
    if label_artifact.byte_count != len(canonical_json_bytes(label_set) + b'\n'):
        raise ExecutionGoldDerivationError('merged label-set byte count does not match receipt')
    if merge_receipt.missing_label_record_count != label_set.missing_record_count:
        raise ExecutionGoldDerivationError('merge receipt missing-label count does not match audited labels')
    if merge_receipt.failed_status_count != label_set.failed_status_count:
        raise ExecutionGoldDerivationError('merge receipt failed-status count does not match audited labels')


def _verify_relevance(
    *,
    inventory: ExecutionCohortInventory,
    decision_archives: Mapping[date, Path],
    queue: VaccineRelevanceReviewQueue,
    adjudications: VaccineRelevanceAdjudicationSet,
    review_receipt: VaccineRelevanceReviewReceipt,
    trusted: TrustedExecutionGoldSourceHashes,
) -> tuple[VaccineRelevanceReviewQueue, VaccineRelevanceAdjudicationSet]:
    if _model_sha256(queue) != trusted.relevance_queue_sha256:
        raise ExecutionGoldDerivationError('relevance queue does not match its external trusted hash')
    if _model_sha256(adjudications) != trusted.relevance_adjudication_sha256:
        raise ExecutionGoldDerivationError('relevance adjudications do not match their external trusted hash')
    if _model_sha256(review_receipt) != trusted.relevance_review_receipt_sha256:
        raise ExecutionGoldDerivationError('relevance review receipt does not match its external trusted hash')
    review_artifacts = {item.relative_path: item for item in review_receipt.artifacts}
    expected_artifact_payloads = {
        'organizer/relevance-adjudications.json': canonical_json_bytes(adjudications),
        'organizer/relevance-policy.json': canonical_json_bytes(ACTIVE_VACCINE_RELEVANCE_POLICY),
        'organizer/relevance-review-queue.json': canonical_json_bytes(queue),
    }
    if set(review_artifacts) != set(expected_artifact_payloads):
        raise ExecutionGoldDerivationError('relevance review receipt does not bind the exact required artifacts')
    for relative_path, payload in expected_artifact_payloads.items():
        artifact = review_artifacts[relative_path]
        if artifact.sha256 != _sha256_bytes(payload) or artifact.byte_count != len(payload):
            raise ExecutionGoldDerivationError(
                f'relevance review artifact does not match its authenticated receipt: {relative_path}'
            )
    if (
        review_receipt.include_count,
        review_receipt.exclude_count,
        review_receipt.hold_count,
    ) != (adjudications.include_count, adjudications.exclude_count, adjudications.hold_count):
        raise ExecutionGoldDerivationError('relevance review receipt counts do not match adjudications')
    permitted_inventory_hashes = {_model_sha256(inventory), _merge_artifact_sha256(inventory)}
    if queue.merged_inventory_sha256 not in permitted_inventory_hashes:
        raise ExecutionGoldDerivationError('relevance queue is not bound to the exact trusted inventory')
    try:
        rebuilt_queue = build_relevance_review_queue(
            inventory=inventory,
            merged_inventory_sha256=queue.merged_inventory_sha256,
            decision_archives=decision_archives,
        )
        rebuilt_adjudications = finalize_relevance_adjudications(
            queue=rebuilt_queue,
            reviews=adjudications.decisions,
        )
    except RelevanceReviewError as error:
        raise ExecutionGoldDerivationError(
            f'decision-only relevance material failed exact archive audit: {error}'
        ) from error
    if canonical_json_bytes(rebuilt_queue) != canonical_json_bytes(queue):
        raise ExecutionGoldDerivationError('relevance queue does not reconstruct from trusted inventory and archives')
    if canonical_json_bytes(rebuilt_adjudications) != canonical_json_bytes(adjudications):
        raise ExecutionGoldDerivationError('relevance adjudications do not reconstruct from the exact review queue')
    return rebuilt_queue, rebuilt_adjudications


def _find_exact[T](values: Sequence[T], *, name: str, predicate: Callable[[T], bool]) -> T:
    matches = tuple(value for value in values if predicate(value))
    if len(matches) != 1:
        raise ExecutionGoldDerivationError(f'expected exactly one {name}, found {len(matches)}')
    return matches[0]


def _verify_context(context: ExecutionTaskContext, assignment: TrialAnchorAssignment) -> str:
    context = ExecutionTaskContext.model_validate_json(canonical_json_bytes(context))
    expected_policy = execution_forecast_spec_policy()
    expected_source_fields = (
        assignment.decision_snapshot_id,
        assignment.anchor_date,
        assignment.label_snapshot_id,
        assignment.label_archive_date,
        assignment.planned_enrollment,
        assignment.planned_primary_completion_date,
    )
    observed_source_fields = (
        context.decision_snapshot_id,
        context.anchor_date,
        context.label_snapshot_id,
        context.label_archive_date,
        context.planned_enrollment,
        context.planned_primary_completion_date,
    )
    if observed_source_fields != expected_source_fields:
        raise ExecutionGoldDerivationError('task context source fields do not match the earliest-anchor assignment')
    if (
        context.enrollment_ratio_spec != expected_policy.enrollment_ratio_spec
        or context.primary_completion_slippage_days_spec != expected_policy.primary_completion_slippage_days_spec
    ):
        raise ExecutionGoldDerivationError('task context does not use the adapter-fixed forecast-spec policy')
    if context.fact_questions:
        raise ExecutionGoldDerivationError('this label-only adapter cannot derive caller-authored cutoff fact gold')
    if any(
        (
            not context.development_only,
            context.leaderboard_admitted,
            context.sealed_execution_supported,
            context.identity_contamination_controlled,
            context.span_mapped_identity_mask_receipt_present,
            context.per_episode_scalar_subset_selection_robust,
            context.source_derivation_verified,
            context.forecast_spec_preregistered,
        )
    ):
        raise ExecutionGoldDerivationError('task context weakens the explicit development-only false release flags')
    return execution_task_context_sha256(context)


def derive_execution_private_gold(
    *,
    nct_id: str,
    context: ExecutionTaskContext,
    inventory: ExecutionCohortInventory,
    label_set: ExecutionLabelSet,
    merge_receipt: AactExecutionMultiAnchorMergeReceipt,
    trusted_source_hashes: TrustedExecutionGoldSourceHashes,
    decision_archives: Mapping[date, Path],
    relevance_queue: VaccineRelevanceReviewQueue,
    relevance_adjudications: VaccineRelevanceAdjudicationSet,
    relevance_review_receipt: VaccineRelevanceReviewReceipt,
) -> ExecutionGoldDerivation:
    """Derive one private gold record from independently audited organizer-only sources.

    Cohort/relevance reconstruction occurs before the label set is audited or inspected.  The
    separate trusted hashes are expected to come from an organizer-controlled manifest, not from
    the same untrusted caller that supplies the models.
    """

    inventory = ExecutionCohortInventory.model_validate_json(canonical_json_bytes(inventory))
    merge_receipt = AactExecutionMultiAnchorMergeReceipt.model_validate_json(canonical_json_bytes(merge_receipt))
    trusted_source_hashes = TrustedExecutionGoldSourceHashes.model_validate_json(
        canonical_json_bytes(trusted_source_hashes)
    )
    try:
        relevance_queue = VaccineRelevanceReviewQueue.model_validate_json(canonical_json_bytes(relevance_queue))
        relevance_adjudications = VaccineRelevanceAdjudicationSet.model_validate_json(
            canonical_json_bytes(relevance_adjudications)
        )
        relevance_review_receipt = VaccineRelevanceReviewReceipt.model_validate_json(
            canonical_json_bytes(relevance_review_receipt)
        )
    except ValueError as error:
        raise ExecutionGoldDerivationError(f'relevance source material failed schema audit: {error}') from error
    try:
        audit_execution_inventory(inventory)
    except ExecutionInventoryError as error:
        raise ExecutionGoldDerivationError(f'execution inventory failed deterministic audit: {error}') from error
    source_by_anchor = _verify_merge_sources(
        inventory=inventory,
        merge_receipt=merge_receipt,
        trusted=trusted_source_hashes,
    )
    rebuilt_queue, rebuilt_adjudications = _verify_relevance(
        inventory=inventory,
        decision_archives=decision_archives,
        queue=relevance_queue,
        adjudications=relevance_adjudications,
        review_receipt=relevance_review_receipt,
        trusted=trusted_source_hashes,
    )

    assignment = _find_exact(
        inventory.assignments,
        name='earliest-anchor assignment',
        predicate=lambda item: item.nct_id == nct_id,
    )
    decision = _find_exact(
        inventory.decision_rows,
        name='assigned decision row',
        predicate=lambda item: item.nct_id == nct_id and item.archive_date == assignment.anchor_date,
    )
    relevance_record = _find_exact(
        rebuilt_queue.records,
        name='relevance evidence record',
        predicate=lambda item: item.nct_id == nct_id and item.anchor_date == assignment.anchor_date,
    )
    relevance_decision = _find_exact(
        rebuilt_adjudications.decisions,
        name='relevance decision',
        predicate=lambda item: item.nct_id == nct_id and item.anchor_date == assignment.anchor_date,
    )
    if relevance_decision.evidence_sha256 != relevance_record.evidence_sha256:
        raise ExecutionGoldDerivationError('relevance decision is not bound to the exact decision evidence')
    if relevance_decision.disposition != RelevanceDisposition.INCLUDE:
        raise ExecutionGoldDerivationError('private gold may be derived only for relevance INCLUDE assignments')

    context_sha256 = _verify_context(context, assignment)

    # Only after selection and decision-only relevance are fixed do we inspect later observations.
    try:
        label_set = ExecutionLabelSet.model_validate_json(canonical_json_bytes(label_set))
    except ValueError as error:
        raise ExecutionGoldDerivationError(f'execution label set failed schema audit: {error}') from error
    _verify_label_source(
        label_set=label_set,
        merge_receipt=merge_receipt,
        trusted=trusted_source_hashes,
    )
    try:
        audit_execution_label_set(inventory=inventory, label_set=label_set)
    except ExecutionInventoryError as error:
        raise ExecutionGoldDerivationError(f'execution label set failed deterministic audit: {error}') from error
    outcome = _find_exact(
        label_set.outcome_rows,
        name='later outcome row',
        predicate=lambda item: item.nct_id == nct_id,
    )
    label = _find_exact(
        label_set.labels,
        name='derived execution label',
        predicate=lambda item: item.nct_id == nct_id,
    )
    source_build = source_by_anchor[assignment.anchor_date]
    source_observation = ExecutionGoldSourceObservation(
        nct_id=nct_id,
        assignment_sha256=_model_sha256(assignment),
        decision_snapshot_id=assignment.decision_snapshot_id,
        anchor_date=assignment.anchor_date,
        decision_archive_sha256=source_build.decision_archive_sha256,
        decision_row_sha256=_model_sha256(decision),
        decision_source_record_sha256=decision.source_record_sha256,
        planned_enrollment=assignment.planned_enrollment,
        planned_primary_completion_date=assignment.planned_primary_completion_date,
        label_snapshot_id=assignment.label_snapshot_id,
        label_archive_date=assignment.label_archive_date,
        label_archive_sha256=source_build.label_archive_sha256,
        outcome_row_sha256=_model_sha256(outcome),
        outcome_source_record_sha256=outcome.source_record_sha256,
        registry_record_present=outcome.record_present,
        raw_overall_status=outcome.overall_status,
        raw_enrollment=outcome.enrollment,
        raw_enrollment_type=outcome.enrollment_type,
        raw_primary_completion_date=outcome.primary_completion_date,
        raw_primary_completion_date_type=outcome.primary_completion_date_type,
        registry_outcome_class=label.registry_outcome_class,
        enrollment_observation=label.enrollment_observation,
        enrollment_ratio=label.enrollment_ratio,
        primary_completion_observation=label.primary_completion_observation,
        primary_completion_slippage_days=label.primary_completion_slippage_days,
    )
    gold = ExecutionPrivateGold(
        episode_id=context.episode_id,
        target_trial_id=context.target_trial_id,
        organizer_private_nct_id=nct_id,
        organizer_private_decision_record_sha256=decision.source_record_sha256,
        task_context_sha256=context_sha256,
        registry_outcome_class=source_observation.registry_outcome_class,
        enrollment_observation=source_observation.enrollment_observation,
        enrollment_ratio=source_observation.enrollment_ratio,
        primary_completion_observation=source_observation.primary_completion_observation,
        primary_completion_slippage_days=source_observation.primary_completion_slippage_days,
    )
    relevance_binding = ExecutionGoldRelevanceBinding(
        policy_id=rebuilt_adjudications.policy_id,
        policy_sha256=rebuilt_adjudications.policy_sha256,
        review_method_id=rebuilt_adjudications.review_method_id,
        review_queue_sha256=_model_sha256(rebuilt_queue),
        adjudication_sha256=_model_sha256(rebuilt_adjudications),
        authenticated_review_receipt_sha256=_model_sha256(relevance_review_receipt),
        evidence_sha256=relevance_record.evidence_sha256,
        decision_sha256=_model_sha256(relevance_decision),
    )
    receipt = ExecutionGoldSourceDerivationReceipt(
        organizer_private_nct_id=nct_id,
        episode_id=context.episode_id,
        target_trial_id=context.target_trial_id,
        task_context_sha256=context_sha256,
        trusted_source_hashes=trusted_source_hashes,
        inventory_model_sha256=_model_sha256(inventory),
        label_set_model_sha256=_model_sha256(label_set),
        merge_receipt_model_sha256=_model_sha256(merge_receipt),
        forecast_spec_policy_sha256=execution_forecast_spec_policy_sha256(),
        source_observation=source_observation,
        relevance=relevance_binding,
        private_gold_sha256=_model_sha256(gold),
    )
    return ExecutionGoldDerivation(receipt=receipt, private_gold=gold)


def _unique_index[T, K](values: Sequence[T], *, name: str, key: Callable[[T], K]) -> dict[K, T]:
    result: dict[K, T] = {}
    for value in values:
        item_key = key(value)
        if item_key in result:
            raise ExecutionGoldDerivationError(f'duplicate {name}: {item_key!r}')
        result[item_key] = value
    return result


def _verify_lineage_split(
    *,
    assignments: LineageSplitAssignmentSet,
    receipt: LineageSplitReceipt,
    trusted_receipt_sha256: str,
    trusted_sources: TrustedExecutionGoldSourceHashes,
) -> tuple[dict[tuple[date, str], LineageCaseAssignment], ExecutionGoldCohortSplitBinding]:
    try:
        assignments = LineageSplitAssignmentSet.model_validate_json(canonical_json_bytes(assignments))
        receipt = LineageSplitReceipt.model_validate_json(canonical_json_bytes(receipt))
    except ValueError as error:
        raise ExecutionGoldDerivationError(f'lineage split failed schema audit: {error}') from error
    if _model_sha256(receipt) != trusted_receipt_sha256:
        raise ExecutionGoldDerivationError('lineage-split receipt does not match its external trusted hash')
    artifact_by_path = {item.relative_path: item for item in receipt.artifacts}
    assignments_artifact = artifact_by_path.get('organizer/lineage-split-assignments.json')
    policy_artifact = artifact_by_path.get('organizer/lineage-split-policy.json')
    if assignments_artifact is None or policy_artifact is None or len(artifact_by_path) != 2:
        raise ExecutionGoldDerivationError('lineage-split receipt does not bind the exact required artifacts')
    assignments_payload = canonical_json_bytes(assignments)
    policy_payload = canonical_json_bytes(LINEAGE_SPLIT_POLICY)
    if (
        assignments_artifact.sha256 != _sha256_bytes(assignments_payload)
        or assignments_artifact.byte_count != len(assignments_payload)
        or policy_artifact.sha256 != _sha256_bytes(policy_payload)
        or policy_artifact.byte_count != len(policy_payload)
    ):
        raise ExecutionGoldDerivationError('lineage-split artifacts do not match their authenticated receipt')
    expected_upstream = (
        trusted_sources.merge_receipt_artifact_sha256,
        trusted_sources.inventory_artifact_sha256,
        trusted_sources.relevance_review_receipt_sha256,
        trusted_sources.relevance_queue_sha256,
        trusted_sources.relevance_adjudication_sha256,
        _model_sha256(ACTIVE_VACCINE_RELEVANCE_POLICY),
    )
    observed_upstream = (
        assignments.merge_receipt_sha256,
        assignments.merged_inventory_artifact_sha256,
        assignments.relevance_review_receipt_sha256,
        assignments.relevance_queue_artifact_sha256,
        assignments.relevance_adjudication_artifact_sha256,
        assignments.relevance_policy_artifact_sha256,
    )
    if observed_upstream != expected_upstream:
        raise ExecutionGoldDerivationError('lineage split is not bound to the exact trusted cohort sources')
    receipt_fields = (
        receipt.policy_sha256,
        receipt.merge_receipt_sha256,
        receipt.merged_inventory_artifact_sha256,
        receipt.relevance_review_receipt_sha256,
        receipt.relevance_queue_artifact_sha256,
        receipt.relevance_adjudication_artifact_sha256,
        receipt.id_key_commitment_sha256,
        receipt.assignment_count,
        receipt.lineage_count,
        receipt.upstream_exclude_count,
        receipt.upstream_hold_count,
        receipt.split_counts,
    )
    assignment_fields = (
        assignments.policy_sha256,
        assignments.merge_receipt_sha256,
        assignments.merged_inventory_artifact_sha256,
        assignments.relevance_review_receipt_sha256,
        assignments.relevance_queue_artifact_sha256,
        assignments.relevance_adjudication_artifact_sha256,
        assignments.id_key_commitment_sha256,
        assignments.assignment_count,
        len(assignments.lineages),
        assignments.upstream_exclude_count,
        assignments.upstream_hold_count,
        assignments.split_counts,
    )
    if receipt_fields != assignment_fields:
        raise ExecutionGoldDerivationError('lineage-split receipt fields do not match its assignment artifact')
    if (
        not assignments.include_coverage_complete
        or not assignments.lineage_split_isolated
        or assignments.held_or_dropped_include_count != 0
        or assignments.outcome_conditioned_grouping_or_split
        or assignments.public_ids_currently_safe_to_release
        or assignments.leaderboard_admitted
    ):
        raise ExecutionGoldDerivationError('lineage split weakens required coverage, isolation, or release flags')
    cases = _unique_index(
        assignments.cases,
        name='lineage case assignment',
        key=lambda item: (item.anchor_date, item.nct_id),
    )
    return cases, ExecutionGoldCohortSplitBinding(
        split_receipt_sha256=trusted_receipt_sha256,
        split_assignments_sha256=_model_sha256(assignments),
        split_policy_sha256=assignments.policy_sha256,
        id_key_commitment_sha256=assignments.id_key_commitment_sha256,
        assignment_count=assignments.assignment_count,
        lineage_count=len(assignments.lineages),
    )


def _recomputed_source_observation(
    *,
    nct_id: str,
    assignment: TrialAnchorAssignment,
    decision: AactExecutionDecisionRow,
    outcome: AactExecutionOutcomeRow,
    source_build: SourceExecutionBuildReceipt,
) -> ExecutionGoldSourceObservation:
    """Construct derived values from retained raw rows, never from caller-authored label summaries."""

    # These concrete attributes are guaranteed by the audited execution schema.  Keeping this
    # helper below the schema audit avoids accepting a structurally similar caller object.
    record_present = outcome.record_present
    raw_status = outcome.overall_status
    raw_enrollment = outcome.enrollment
    raw_enrollment_type = outcome.enrollment_type
    raw_completion = outcome.primary_completion_date
    raw_completion_type = outcome.primary_completion_date_type
    actual_enrollment = raw_enrollment if record_present and raw_enrollment_type == RegistryValueType.ACTUAL else None
    actual_completion = raw_completion if record_present and raw_completion_type == RegistryValueType.ACTUAL else None
    return ExecutionGoldSourceObservation(
        nct_id=nct_id,
        assignment_sha256=_model_sha256(assignment),
        decision_snapshot_id=assignment.decision_snapshot_id,
        anchor_date=assignment.anchor_date,
        decision_archive_sha256=source_build.decision_archive_sha256,
        decision_row_sha256=_model_sha256(decision),
        decision_source_record_sha256=decision.source_record_sha256,
        planned_enrollment=assignment.planned_enrollment,
        planned_primary_completion_date=assignment.planned_primary_completion_date,
        label_snapshot_id=assignment.label_snapshot_id,
        label_archive_date=assignment.label_archive_date,
        label_archive_sha256=source_build.label_archive_sha256,
        outcome_row_sha256=_model_sha256(outcome),
        outcome_source_record_sha256=outcome.source_record_sha256,
        registry_record_present=record_present,
        raw_overall_status=raw_status,
        raw_enrollment=raw_enrollment,
        raw_enrollment_type=raw_enrollment_type,
        raw_primary_completion_date=raw_completion,
        raw_primary_completion_date_type=raw_completion_type,
        registry_outcome_class=registry_outcome_class(record_present, raw_status),
        enrollment_observation=observation_state(record_present, actual_enrollment, raw_enrollment_type),
        enrollment_ratio=(
            round(actual_enrollment / assignment.planned_enrollment, ENROLLMENT_RATIO_DECIMAL_PLACES)
            if actual_enrollment is not None
            else None
        ),
        primary_completion_observation=observation_state(record_present, actual_completion, raw_completion_type),
        primary_completion_slippage_days=(
            (actual_completion - assignment.planned_primary_completion_date).days
            if actual_completion is not None
            else None
        ),
    )


def derive_execution_private_gold_cohort(
    *,
    targets: ExecutionGoldCohortTargetSet,
    inventory: ExecutionCohortInventory,
    label_set: ExecutionLabelSet,
    merge_receipt: AactExecutionMultiAnchorMergeReceipt,
    trusted_source_hashes: TrustedExecutionGoldSourceHashes,
    decision_archives: Mapping[date, Path],
    relevance_queue: VaccineRelevanceReviewQueue,
    relevance_adjudications: VaccineRelevanceAdjudicationSet,
    relevance_review_receipt: VaccineRelevanceReviewReceipt,
    lineage_split_assignments: LineageSplitAssignmentSet | None = None,
    lineage_split_receipt: LineageSplitReceipt | None = None,
    trusted_lineage_split_receipt_sha256: str | None = None,
) -> ExecutionGoldCohortDerivation:
    """Verify the shared sources once, then derive gold for every and only INCLUDE assignment.

    ``targets`` must already contain the final outcome-blind context per NCT.  The exact-coverage
    check prevents a caller from selecting a favorable subset after seeing later labels.  This
    development adapter still does not claim that the supplied documents are identity-masked or
    outcome-blind; those properties belong to the later workspace and admission stages.
    """

    try:
        targets = ExecutionGoldCohortTargetSet.model_validate_json(canonical_json_bytes(targets))
        inventory = ExecutionCohortInventory.model_validate_json(canonical_json_bytes(inventory))
        merge_receipt = AactExecutionMultiAnchorMergeReceipt.model_validate_json(canonical_json_bytes(merge_receipt))
        trusted_source_hashes = TrustedExecutionGoldSourceHashes.model_validate_json(
            canonical_json_bytes(trusted_source_hashes)
        )
        relevance_queue = VaccineRelevanceReviewQueue.model_validate_json(canonical_json_bytes(relevance_queue))
        relevance_adjudications = VaccineRelevanceAdjudicationSet.model_validate_json(
            canonical_json_bytes(relevance_adjudications)
        )
        relevance_review_receipt = VaccineRelevanceReviewReceipt.model_validate_json(
            canonical_json_bytes(relevance_review_receipt)
        )
    except ValueError as error:
        raise ExecutionGoldDerivationError(f'cohort source material failed schema audit: {error}') from error
    try:
        audit_execution_inventory(inventory)
    except ExecutionInventoryError as error:
        raise ExecutionGoldDerivationError(f'execution inventory failed deterministic audit: {error}') from error

    # All decision-only sources and their external hashes are validated before the label model is
    # even schema-parsed.  These cohort-global checks run once, rather than once per selected case.
    source_by_anchor = _verify_merge_sources(
        inventory=inventory,
        merge_receipt=merge_receipt,
        trusted=trusted_source_hashes,
    )
    rebuilt_queue, rebuilt_adjudications = _verify_relevance(
        inventory=inventory,
        decision_archives=decision_archives,
        queue=relevance_queue,
        adjudications=relevance_adjudications,
        review_receipt=relevance_review_receipt,
        trusted=trusted_source_hashes,
    )

    assignment_by_key = _unique_index(
        inventory.assignments,
        name='earliest-anchor assignment',
        key=lambda item: (item.anchor_date, item.nct_id),
    )
    relevance_record_by_key = _unique_index(
        rebuilt_queue.records,
        name='relevance evidence record',
        key=lambda item: (item.anchor_date, item.nct_id),
    )
    relevance_decision_by_key = _unique_index(
        rebuilt_adjudications.decisions,
        name='relevance decision',
        key=lambda item: (item.anchor_date, item.nct_id),
    )
    include_keys = tuple(
        sorted(
            key
            for key, decision in relevance_decision_by_key.items()
            if decision.disposition == RelevanceDisposition.INCLUDE
        )
    )
    if len(include_keys) != rebuilt_adjudications.include_count:
        raise ExecutionGoldDerivationError('relevance INCLUDE count does not match unique selected assignments')
    if set(relevance_record_by_key) != set(assignment_by_key) or set(relevance_decision_by_key) != set(
        assignment_by_key
    ):
        raise ExecutionGoldDerivationError('relevance queue/adjudications do not exactly cover inventory assignments')
    target_by_key = _unique_index(
        targets.targets,
        name='cohort target',
        key=lambda item: (item.context.anchor_date, item.organizer_private_nct_id),
    )
    expected_keys = set(include_keys)
    observed_keys = set(target_by_key)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ExecutionGoldDerivationError(
            f'cohort targets must exactly cover relevance INCLUDE assignments; missing={missing}, extra={extra}'
        )
    split_inputs = (
        lineage_split_assignments,
        lineage_split_receipt,
        trusted_lineage_split_receipt_sha256,
    )
    if any(item is None for item in split_inputs) != all(item is None for item in split_inputs):
        raise ExecutionGoldDerivationError('lineage split assignments, receipt, and trusted hash are all-or-none')
    split_case_by_key: dict[tuple[date, str], LineageCaseAssignment] = {}
    split_binding: ExecutionGoldCohortSplitBinding | None = None
    if lineage_split_assignments is not None:
        if lineage_split_receipt is None or trusted_lineage_split_receipt_sha256 is None:
            raise ExecutionGoldDerivationError('incomplete lineage split binding')
        split_case_by_key, split_binding = _verify_lineage_split(
            assignments=lineage_split_assignments,
            receipt=lineage_split_receipt,
            trusted_receipt_sha256=trusted_lineage_split_receipt_sha256,
            trusted_sources=trusted_source_hashes,
        )
        if set(split_case_by_key) != expected_keys:
            raise ExecutionGoldDerivationError('finalized lineage split must exactly cover relevance INCLUDE cases')
    context_hash_by_key: dict[tuple[date, str], str] = {}
    for key in include_keys:
        assignment = assignment_by_key[key]
        relevance_record = relevance_record_by_key[key]
        relevance_decision = relevance_decision_by_key[key]
        if relevance_record.evidence_sha256 != relevance_decision.evidence_sha256:
            raise ExecutionGoldDerivationError(f'relevance evidence/decision mismatch for {key!r}')
        if split_case_by_key:
            split_case = split_case_by_key[key]
            if (
                split_case.source_assignment_sha256 != _model_sha256(assignment)
                or split_case.relevance_evidence_sha256 != relevance_record.evidence_sha256
                or split_case.relevance_decision_sha256 != _model_sha256(relevance_decision)
            ):
                raise ExecutionGoldDerivationError(f'lineage split case is not source-bound for {key!r}')
        context_hash_by_key[key] = _verify_context(target_by_key[key].context, assignment)

    # Selection and exact task coverage are now fixed.  Only now may later observations be opened.
    try:
        label_set = ExecutionLabelSet.model_validate_json(canonical_json_bytes(label_set))
    except ValueError as error:
        raise ExecutionGoldDerivationError(f'execution label set failed schema audit: {error}') from error
    _verify_label_source(
        label_set=label_set,
        merge_receipt=merge_receipt,
        trusted=trusted_source_hashes,
    )
    try:
        audit_execution_label_set(inventory=inventory, label_set=label_set)
    except ExecutionInventoryError as error:
        raise ExecutionGoldDerivationError(f'execution label set failed deterministic audit: {error}') from error

    decision_by_key = _unique_index(
        inventory.decision_rows,
        name='decision row',
        key=lambda item: (item.archive_date, item.nct_id),
    )
    outcome_by_nct = _unique_index(label_set.outcome_rows, name='later outcome row', key=lambda item: item.nct_id)
    label_by_nct = _unique_index(label_set.labels, name='derived execution label', key=lambda item: item.nct_id)
    inventory_nct_ids = {item.nct_id for item in inventory.assignments}
    if set(outcome_by_nct) != inventory_nct_ids or set(label_by_nct) != inventory_nct_ids:
        raise ExecutionGoldDerivationError('later outcome rows and labels must exactly cover inventory assignments')

    case_receipts: list[ExecutionGoldCohortCaseReceipt] = []
    gold_records: list[ExecutionPrivateGold] = []
    for key in include_keys:
        assignment = assignment_by_key[key]
        target = target_by_key[key]
        decision = decision_by_key.get(key)
        if decision is None:
            raise ExecutionGoldDerivationError(f'missing assigned decision row for {key!r}')
        outcome = outcome_by_nct[assignment.nct_id]
        audited_label = label_by_nct[assignment.nct_id]
        source = _recomputed_source_observation(
            nct_id=assignment.nct_id,
            assignment=assignment,
            decision=decision,
            outcome=outcome,
            source_build=source_by_anchor[assignment.anchor_date],
        )
        recomputed_label_fields = (
            source.registry_outcome_class,
            source.enrollment_observation,
            source.enrollment_ratio,
            source.primary_completion_observation,
            source.primary_completion_slippage_days,
        )
        audited_label_fields = (
            audited_label.registry_outcome_class,
            audited_label.enrollment_observation,
            audited_label.enrollment_ratio,
            audited_label.primary_completion_observation,
            audited_label.primary_completion_slippage_days,
        )
        if recomputed_label_fields != audited_label_fields:
            raise ExecutionGoldDerivationError(f'recomputed outcome does not match audited label for {key!r}')
        gold = ExecutionPrivateGold(
            episode_id=target.context.episode_id,
            target_trial_id=target.context.target_trial_id,
            organizer_private_nct_id=assignment.nct_id,
            organizer_private_decision_record_sha256=decision.source_record_sha256,
            task_context_sha256=context_hash_by_key[key],
            registry_outcome_class=source.registry_outcome_class,
            enrollment_observation=source.enrollment_observation,
            enrollment_ratio=source.enrollment_ratio,
            primary_completion_observation=source.primary_completion_observation,
            primary_completion_slippage_days=source.primary_completion_slippage_days,
        )
        relevance_decision = relevance_decision_by_key[key]
        split_case = split_case_by_key.get(key)
        case_receipts.append(
            ExecutionGoldCohortCaseReceipt(
                organizer_private_nct_id=assignment.nct_id,
                episode_id=target.context.episode_id,
                target_trial_id=target.context.target_trial_id,
                task_context_sha256=context_hash_by_key[key],
                source_observation=source,
                relevance_evidence_sha256=relevance_record_by_key[key].evidence_sha256,
                relevance_decision_sha256=_model_sha256(relevance_decision),
                lineage_case_assignment_sha256=(_model_sha256(split_case) if split_case is not None else None),
                lineage_group_id=split_case.lineage_group_id if split_case is not None else None,
                split=split_case.split if split_case is not None else None,
                private_gold_sha256=_model_sha256(gold),
            )
        )
        gold_records.append(gold)

    private_gold = ExecutionPrivateGoldSet(
        cohort_id=targets.cohort_id,
        records=tuple(sorted(gold_records, key=lambda item: item.organizer_private_nct_id)),
    )
    receipt = ExecutionGoldCohortSourceDerivationReceipt(
        cohort_id=targets.cohort_id,
        trusted_source_hashes=trusted_source_hashes,
        inventory_model_sha256=_model_sha256(inventory),
        label_set_model_sha256=_model_sha256(label_set),
        merge_receipt_model_sha256=_model_sha256(merge_receipt),
        forecast_spec_policy_sha256=execution_forecast_spec_policy_sha256(),
        relevance_policy_id=rebuilt_adjudications.policy_id,
        relevance_policy_sha256=rebuilt_adjudications.policy_sha256,
        relevance_review_method_id=rebuilt_adjudications.review_method_id,
        relevance_queue_sha256=_model_sha256(rebuilt_queue),
        relevance_adjudication_sha256=_model_sha256(rebuilt_adjudications),
        authenticated_review_receipt_sha256=_model_sha256(relevance_review_receipt),
        target_set_sha256=_model_sha256(targets),
        eligible_include_count=len(include_keys),
        derived_case_count=len(case_receipts),
        case_receipts=tuple(case_receipts),
        private_gold_set_sha256=_model_sha256(private_gold),
        split_binding=split_binding,
        final_workspace_contexts_bound=targets.final_workspace_contexts_bound,
        split_inventory_bound=split_binding is not None,
        lineage_split_safe=split_binding is not None,
    )
    return ExecutionGoldCohortDerivation(receipt=receipt, targets=targets, private_gold=private_gold)


def validate_execution_gold_cohort_derivation(
    derivation: ExecutionGoldCohortDerivation,
) -> ExecutionGoldCohortDerivation:
    try:
        receipt = ExecutionGoldCohortSourceDerivationReceipt.model_validate_json(
            canonical_json_bytes(derivation.receipt)
        )
        targets = ExecutionGoldCohortTargetSet.model_validate_json(canonical_json_bytes(derivation.targets))
        private_gold = ExecutionPrivateGoldSet.model_validate_json(canonical_json_bytes(derivation.private_gold))
    except ValueError as error:
        raise ExecutionGoldDerivationError(f'cohort derivation failed schema audit: {error}') from error
    if receipt.cohort_id != targets.cohort_id or receipt.cohort_id != private_gold.cohort_id:
        raise ExecutionGoldDerivationError('cohort IDs disagree across receipt, targets, and private gold')
    if receipt.final_workspace_contexts_bound != targets.final_workspace_contexts_bound:
        raise ExecutionGoldDerivationError('cohort receipt does not preserve the target-set workspace binding state')
    if receipt.target_set_sha256 != _model_sha256(targets):
        raise ExecutionGoldDerivationError('cohort targets do not match the source-derivation receipt')
    if receipt.private_gold_set_sha256 != _model_sha256(private_gold):
        raise ExecutionGoldDerivationError('cohort private gold does not match the source-derivation receipt')

    target_by_nct = _unique_index(
        targets.targets,
        name='cohort target',
        key=lambda item: item.organizer_private_nct_id,
    )
    gold_by_nct = _unique_index(
        private_gold.records,
        name='cohort private gold',
        key=lambda item: item.organizer_private_nct_id,
    )
    case_by_nct = _unique_index(
        receipt.case_receipts,
        name='cohort case receipt',
        key=lambda item: item.organizer_private_nct_id,
    )
    if set(target_by_nct) != set(gold_by_nct) or set(target_by_nct) != set(case_by_nct):
        raise ExecutionGoldDerivationError('cohort targets, case receipts, and private gold do not exactly align')
    for nct_id, target in target_by_nct.items():
        gold = gold_by_nct[nct_id]
        case = case_by_nct[nct_id]
        expected_context_hash = execution_task_context_sha256(target.context)
        if (
            case.episode_id,
            case.target_trial_id,
            case.task_context_sha256,
        ) != (
            target.context.episode_id,
            target.context.target_trial_id,
            expected_context_hash,
        ):
            raise ExecutionGoldDerivationError(f'case receipt does not bind its exact target context for {nct_id}')
        if (
            gold.episode_id,
            gold.target_trial_id,
            gold.task_context_sha256,
            gold.organizer_private_decision_record_sha256,
        ) != (
            case.episode_id,
            case.target_trial_id,
            case.task_context_sha256,
            case.source_observation.decision_source_record_sha256,
        ):
            raise ExecutionGoldDerivationError(f'private gold does not bind its exact case receipt for {nct_id}')
        if case.private_gold_sha256 != _model_sha256(gold):
            raise ExecutionGoldDerivationError(f'private gold hash does not match its case receipt for {nct_id}')
        source_outcomes = (
            case.source_observation.registry_outcome_class,
            case.source_observation.enrollment_observation,
            case.source_observation.enrollment_ratio,
            case.source_observation.primary_completion_observation,
            case.source_observation.primary_completion_slippage_days,
        )
        gold_outcomes = (
            gold.registry_outcome_class,
            gold.enrollment_observation,
            gold.enrollment_ratio,
            gold.primary_completion_observation,
            gold.primary_completion_slippage_days,
        )
        if source_outcomes != gold_outcomes:
            raise ExecutionGoldDerivationError(f'private gold outcomes do not match recomputed source for {nct_id}')
    return ExecutionGoldCohortDerivation(receipt=receipt, targets=targets, private_gold=private_gold)


def _write_exact(path: Path, value: object, relative_path: str) -> ArtifactReceipt:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return ArtifactReceipt(relative_path=relative_path, sha256=_sha256_bytes(payload), byte_count=len(payload))


def load_execution_gold_cohort_derivation(
    root: Path,
    *,
    expected_receipt_sha256: str,
) -> ExecutionGoldCohortDerivationBuild:
    """Load a complete immutable cohort build behind an external receipt-hash trust anchor."""

    if re.fullmatch(_SHA256_PATTERN, expected_receipt_sha256) is None:
        raise ExecutionGoldDerivationError('expected cohort receipt SHA-256 must be a 64-character digest')
    expanded = root.expanduser()
    if expanded.is_symlink():
        raise ExecutionGoldDerivationError('cohort derivation root cannot be a symbolic link')
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise ExecutionGoldDerivationError(f'cohort derivation root is not a directory: {resolved}')
    if stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise ExecutionGoldDerivationError('cohort derivation root must have mode 0700')
    expected_files = {
        'organizer/cohort-source-derivation-receipt.json',
        'organizer/cohort-targets.json',
        'organizer/forecast-spec-policy.json',
        'private/execution-private-gold-set.json',
    }
    expected_directories = {'organizer', 'private'}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for entry in resolved.rglob('*'):
        relative = entry.relative_to(resolved).as_posix()
        if entry.is_symlink():
            raise ExecutionGoldDerivationError(f'cohort derivation cannot contain symbolic links: {relative}')
        if entry.is_dir():
            if stat.S_IMODE(entry.stat().st_mode) != 0o700:
                raise ExecutionGoldDerivationError(f'cohort derivation directory must have mode 0700: {relative}')
            observed_directories.add(relative)
        elif entry.is_file():
            if stat.S_IMODE(entry.stat().st_mode) != 0o600:
                raise ExecutionGoldDerivationError(f'cohort derivation artifact must have mode 0600: {relative}')
            observed_files.add(relative)
        else:
            raise ExecutionGoldDerivationError(f'cohort derivation contains a non-regular artifact: {relative}')
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ExecutionGoldDerivationError(
            'cohort derivation root must contain exactly the receipt, targets, policy, and private-gold artifacts'
        )

    payloads = {relative: (resolved / relative).read_bytes() for relative in sorted(expected_files)}
    receipt_path = 'organizer/cohort-source-derivation-receipt.json'
    if _sha256_bytes(payloads[receipt_path]) != expected_receipt_sha256:
        raise ExecutionGoldDerivationError('cohort source-derivation receipt does not match its external hash')
    try:
        receipt = ExecutionGoldCohortSourceDerivationReceipt.model_validate_json(payloads[receipt_path])
        targets = ExecutionGoldCohortTargetSet.model_validate_json(payloads['organizer/cohort-targets.json'])
        policy = ExecutionForecastSpecPolicy.model_validate_json(payloads['organizer/forecast-spec-policy.json'])
        private_gold = ExecutionPrivateGoldSet.model_validate_json(payloads['private/execution-private-gold-set.json'])
    except ValueError as error:
        raise ExecutionGoldDerivationError(f'cohort derivation artifact failed schema audit: {error}') from error
    models_by_path = {
        receipt_path: receipt,
        'organizer/cohort-targets.json': targets,
        'organizer/forecast-spec-policy.json': policy,
        'private/execution-private-gold-set.json': private_gold,
    }
    for relative, model in models_by_path.items():
        if payloads[relative] != canonical_json_bytes(model):
            raise ExecutionGoldDerivationError(f'cohort derivation artifact is not canonical JSON: {relative}')
    if policy != execution_forecast_spec_policy() or receipt.forecast_spec_policy_sha256 != _model_sha256(policy):
        raise ExecutionGoldDerivationError('cohort derivation does not contain the adapter-fixed forecast policy')
    derivation = validate_execution_gold_cohort_derivation(
        ExecutionGoldCohortDerivation(receipt=receipt, targets=targets, private_gold=private_gold)
    )
    artifacts = tuple(
        ArtifactReceipt(
            relative_path=relative,
            sha256=_sha256_bytes(payload),
            byte_count=len(payload),
        )
        for relative, payload in sorted(payloads.items())
    )
    return ExecutionGoldCohortDerivationBuild(root=resolved, derivation=derivation, artifacts=artifacts)


def write_execution_gold_derivation(
    *,
    derivation: ExecutionGoldDerivation,
    output_root: Path,
) -> ExecutionGoldDerivationBuild:
    """Write only organizer/private artifacts; no public task or workspace is created."""

    receipt = ExecutionGoldSourceDerivationReceipt.model_validate_json(canonical_json_bytes(derivation.receipt))
    gold = ExecutionPrivateGold.model_validate_json(canonical_json_bytes(derivation.private_gold))
    if receipt.private_gold_sha256 != _model_sha256(gold):
        raise ExecutionGoldDerivationError('private gold does not match the source-derivation receipt')
    destination = output_root.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'immutable private-gold output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.staging-', dir=destination.parent))
    staging.chmod(0o700)
    try:
        artifacts = (
            _write_exact(
                staging / 'organizer' / 'forecast-spec-policy.json',
                execution_forecast_spec_policy(),
                'organizer/forecast-spec-policy.json',
            ),
            _write_exact(
                staging / 'organizer' / 'source-derivation-receipt.json',
                receipt,
                'organizer/source-derivation-receipt.json',
            ),
            _write_exact(
                staging / 'private' / 'execution-private-gold.json',
                gold,
                'private/execution-private-gold.json',
            ),
        )
        for directory in sorted(
            (path for path in staging.rglob('*') if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            fsync_directory(directory)
        fsync_directory(staging)
        rename_directory_noreplace(staging, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ExecutionGoldDerivationBuild(
        root=destination,
        derivation=ExecutionGoldDerivation(receipt=receipt, private_gold=gold),
        artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
    )


def write_execution_gold_cohort_derivation(
    *,
    derivation: ExecutionGoldCohortDerivation,
    output_root: Path,
) -> ExecutionGoldCohortDerivationBuild:
    """Write one immutable organizer/private cohort build and no public task material."""

    verified = validate_execution_gold_cohort_derivation(derivation)
    destination = output_root.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'immutable private-gold cohort output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.staging-', dir=destination.parent))
    staging.chmod(0o700)
    try:
        artifacts = (
            _write_exact(
                staging / 'organizer' / 'forecast-spec-policy.json',
                execution_forecast_spec_policy(),
                'organizer/forecast-spec-policy.json',
            ),
            _write_exact(
                staging / 'organizer' / 'cohort-targets.json',
                verified.targets,
                'organizer/cohort-targets.json',
            ),
            _write_exact(
                staging / 'organizer' / 'cohort-source-derivation-receipt.json',
                verified.receipt,
                'organizer/cohort-source-derivation-receipt.json',
            ),
            _write_exact(
                staging / 'private' / 'execution-private-gold-set.json',
                verified.private_gold,
                'private/execution-private-gold-set.json',
            ),
        )
        for directory in sorted(
            (path for path in staging.rglob('*') if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            fsync_directory(directory)
        fsync_directory(staging)
        rename_directory_noreplace(staging, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ExecutionGoldCohortDerivationBuild(
        root=destination,
        derivation=verified,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
    )


__all__ = [
    'FORECAST_SPEC_POLICY_ID',
    'GOLD_COHORT_DERIVATION_ADAPTER_ID',
    'GOLD_DERIVATION_ADAPTER_ID',
    'ExecutionForecastSpecPolicy',
    'ExecutionGoldCohortCaseReceipt',
    'ExecutionGoldCohortDerivation',
    'ExecutionGoldCohortDerivationBuild',
    'ExecutionGoldCohortSourceDerivationReceipt',
    'ExecutionGoldCohortSplitBinding',
    'ExecutionGoldCohortTarget',
    'ExecutionGoldCohortTargetSet',
    'ExecutionGoldDerivation',
    'ExecutionGoldDerivationBuild',
    'ExecutionGoldDerivationError',
    'ExecutionGoldRelevanceBinding',
    'ExecutionGoldSourceDerivationReceipt',
    'ExecutionGoldSourceObservation',
    'ExecutionPrivateGoldSet',
    'TrustedExecutionGoldSourceHashes',
    'TrustedSourceBuildHashes',
    'derive_execution_private_gold',
    'derive_execution_private_gold_cohort',
    'execution_forecast_spec_policy',
    'execution_forecast_spec_policy_sha256',
    'load_execution_gold_cohort_derivation',
    'validate_execution_gold_cohort_derivation',
    'write_execution_gold_derivation',
    'write_execution_gold_cohort_derivation',
]
