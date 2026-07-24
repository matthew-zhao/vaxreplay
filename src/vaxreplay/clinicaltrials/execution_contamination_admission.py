"""Fixed split admission and model-weight disclosure for Lane A historical replay.

The case denominator is chosen from organizer probes before any submitted system runs.  A target
system's memory probe is reported beside its score and cannot add or remove cases.  Known access to
organizer-private benchmark material can disqualify a *system* from the held-out leaderboard, but
ordinary public-data exposure and an ambiguous memory signal are labels rather than exclusions.
"""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_contamination import (
    ExecutionCaseContaminationRisk,
    ExecutionCaseRiskStratum,
    ExecutionContaminationStrataManifest,
    ExecutionSystemExposureStatus,
    ExecutionSystemProbeManifest,
    execution_contamination_strata_manifest_sha256,
)

EXECUTION_CONTAMINATION_ADMISSION_POLICY_SCHEMA_VERSION = (
    'vaxreplay.clinical-execution-contamination-admission-policy.dev-v0.1'
)
EXECUTION_CONTAMINATION_ADMISSION_MANIFEST_SCHEMA_VERSION = (
    'vaxreplay.clinical-execution-contamination-admission-manifest.dev-v0.1'
)
EXECUTION_MODEL_WEIGHT_DECLARATION_SCHEMA_VERSION = 'vaxreplay.clinical-execution-model-weight-declaration.dev-v0.1'
EXECUTION_SYSTEM_CONTAMINATION_REPORT_SCHEMA_VERSION = (
    'vaxreplay.clinical-execution-system-contamination-report.dev-v0.1'
)
EXECUTION_CONTAMINATION_ADMISSION_POLICY_ID = 'aact-split-and-weight-admission-v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class ExecutionContaminationAdmissionError(ValueError):
    """A case or system admission record violated the frozen policy."""


class ExposureDeclaration(str, enum.Enum):
    YES = 'yes'
    NO = 'no'
    UNKNOWN = 'unknown'


class ExecutionContaminationAdmissionPolicy(StrictModel):
    """The policy decision the benchmark applies to reidentification and model memory."""

    schema_version: Literal['vaxreplay.clinical-execution-contamination-admission-policy.dev-v0.1'] = (
        EXECUTION_CONTAMINATION_ADMISSION_POLICY_SCHEMA_VERSION
    )
    policy_id: Literal['aact-split-and-weight-admission-v0.1'] = EXECUTION_CONTAMINATION_ADMISSION_POLICY_ID
    train_allows_fingerprintable_and_unmeasured_cases: Literal[True] = True
    held_out_historical_all_retains_fingerprintable_and_unmeasured_cases: Literal[True] = True
    held_out_common_low_risk_requires_no_identity_signal: Literal[True] = True
    test_common_low_risk_is_primary_leaderboard_view: Literal[True] = True
    workspace_leak_or_incomplete_audit_excluded_from_every_benchmark_use: Literal[True] = True
    case_denominators_fixed_before_submitted_system_runs: Literal[True] = True
    submitted_system_probe_never_changes_case_denominator: Literal[True] = True
    public_source_training_exposure_is_reported_not_excluded: Literal[True] = True
    ambiguous_or_positive_memory_probe_is_reported_not_excluded: Literal[True] = True
    known_organizer_private_eval_exposure_disqualifies_held_out_leaderboard: Literal[True] = True
    known_benchmark_specific_test_tuning_disqualifies_held_out_leaderboard: Literal[True] = True
    unknown_model_training_corpus_is_labeled_not_excluded: Literal[True] = True
    machine_unlearning_required: Literal[False] = False
    no_signal_proves_clean_weights: Literal[False] = False
    self_declaration_proves_clean_weights: Literal[False] = False
    residual_model_weight_contamination_possible: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False


EXECUTION_CONTAMINATION_ADMISSION_POLICY = ExecutionContaminationAdmissionPolicy()


def execution_contamination_admission_policy_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(EXECUTION_CONTAMINATION_ADMISSION_POLICY)).hexdigest()


class ExecutionCaseSplitBinding(StrictModel):
    episode_id: str = Field(min_length=1)
    split: Split


class ExecutionCaseSplitAdmission(StrictModel):
    """One case's permissible uses, derived only from its frozen split and case stratum."""

    episode_id: str = Field(min_length=1)
    split: Split
    stratum: ExecutionCaseRiskStratum
    eligible_for_train_use: bool
    eligible_for_dev_historical_all: bool
    eligible_for_dev_common_low_risk: bool
    eligible_for_test_historical_all: bool
    eligible_for_test_common_low_risk: bool
    eligible_for_primary_leaderboard: bool
    exclusion_reason: str | None = None
    target_system_results_used: Literal[False] = False
    fixed_before_submitted_system_runs: Literal[True] = True

    @model_validator(mode='after')
    def validate_fixed_disposition(self) -> Self:
        expected = _split_disposition(split=self.split, stratum=self.stratum)
        observed = (
            self.eligible_for_train_use,
            self.eligible_for_dev_historical_all,
            self.eligible_for_dev_common_low_risk,
            self.eligible_for_test_historical_all,
            self.eligible_for_test_common_low_risk,
            self.eligible_for_primary_leaderboard,
            self.exclusion_reason,
        )
        if observed != expected:
            raise ValueError('case split admission does not follow the frozen policy')
        return self


def _split_disposition(
    *,
    split: Split,
    stratum: ExecutionCaseRiskStratum,
) -> tuple[bool, bool, bool, bool, bool, bool, str | None]:
    if stratum == ExecutionCaseRiskStratum.WORKSPACE_LEAK_EXCLUDED:
        return False, False, False, False, False, False, 'workspace_leak_detected'
    if stratum == ExecutionCaseRiskStratum.WORKSPACE_AUDIT_INCOMPLETE:
        return False, False, False, False, False, False, 'workspace_audit_incomplete'
    low_risk = stratum == ExecutionCaseRiskStratum.NO_IDENTITY_SIGNAL
    if split == Split.TRAIN:
        return True, False, False, False, False, False, None
    if split == Split.DEV:
        return False, True, low_risk, False, False, False, None
    if split == Split.TEST:
        return False, False, False, True, low_risk, low_risk, None
    raise AssertionError(f'unhandled split: {split}')


class ExecutionSplitAdmissionCount(StrictModel):
    split: Split
    case_count: int = Field(ge=0)


class ExecutionContaminationAdmissionManifest(StrictModel):
    """Complete case-use decision, fixed without observing any submitted system."""

    schema_version: Literal['vaxreplay.clinical-execution-contamination-admission-manifest.dev-v0.1'] = (
        EXECUTION_CONTAMINATION_ADMISSION_MANIFEST_SCHEMA_VERSION
    )
    manifest_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    contamination_strata_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_universe_sha256: str = Field(pattern=_SHA256_PATTERN)
    cases: tuple[ExecutionCaseSplitAdmission, ...] = Field(min_length=1)
    case_count: int = Field(gt=0)
    split_counts: tuple[ExecutionSplitAdmissionCount, ...]
    train_eligible_count: int = Field(ge=0)
    dev_historical_all_count: int = Field(ge=0)
    dev_common_low_risk_count: int = Field(ge=0)
    test_historical_all_count: int = Field(ge=0)
    test_common_low_risk_count: int = Field(ge=0)
    primary_leaderboard_count: int = Field(ge=0)
    globally_excluded_count: int = Field(ge=0)
    complete_case_universe_covered: Literal[True] = True
    fixed_before_submitted_system_runs: Literal[True] = True
    target_specific_case_selection_prohibited: Literal[True] = True
    target_specific_denominators_prohibited: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @field_validator('cases')
    @classmethod
    def validate_cases(
        cls,
        value: tuple[ExecutionCaseSplitAdmission, ...],
    ) -> tuple[ExecutionCaseSplitAdmission, ...]:
        episode_ids = tuple(item.episode_id for item in value)
        if episode_ids != tuple(sorted(set(episode_ids))):
            raise ValueError('case admissions must use unique ascending episode IDs')
        return value

    @model_validator(mode='after')
    def validate_summary(self) -> Self:
        if self.policy_sha256 != execution_contamination_admission_policy_sha256():
            raise ValueError('admission manifest does not bind the frozen policy')
        if self.case_count != len(self.cases):
            raise ValueError('case_count is inconsistent')
        expected_split_counts = tuple(
            ExecutionSplitAdmissionCount(
                split=split,
                case_count=sum(item.split == split for item in self.cases),
            )
            for split in (Split.TRAIN, Split.DEV, Split.TEST)
        )
        if self.split_counts != expected_split_counts:
            raise ValueError('split_counts are inconsistent')
        expected_counts = (
            sum(item.eligible_for_train_use for item in self.cases),
            sum(item.eligible_for_dev_historical_all for item in self.cases),
            sum(item.eligible_for_dev_common_low_risk for item in self.cases),
            sum(item.eligible_for_test_historical_all for item in self.cases),
            sum(item.eligible_for_test_common_low_risk for item in self.cases),
            sum(item.eligible_for_primary_leaderboard for item in self.cases),
            sum(item.exclusion_reason is not None for item in self.cases),
        )
        observed_counts = (
            self.train_eligible_count,
            self.dev_historical_all_count,
            self.dev_common_low_risk_count,
            self.test_historical_all_count,
            self.test_common_low_risk_count,
            self.primary_leaderboard_count,
            self.globally_excluded_count,
        )
        if observed_counts != expected_counts:
            raise ValueError('admission summary counts are inconsistent')
        return self


def _canonical_split_bindings(
    bindings: Iterable[ExecutionCaseSplitBinding],
) -> tuple[ExecutionCaseSplitBinding, ...]:
    validated = tuple(
        sorted(
            (ExecutionCaseSplitBinding.model_validate_json(canonical_json_bytes(item)) for item in bindings),
            key=lambda item: item.episode_id,
        )
    )
    episode_ids = tuple(item.episode_id for item in validated)
    if not validated or episode_ids != tuple(sorted(set(episode_ids))):
        raise ExecutionContaminationAdmissionError('split universe must contain unique episode IDs')
    return validated


def execution_split_universe_sha256(bindings: Iterable[ExecutionCaseSplitBinding]) -> str:
    validated = _canonical_split_bindings(bindings)
    return hashlib.sha256(canonical_json_bytes([item.model_dump(mode='json') for item in validated])).hexdigest()


def _build_case_split_admission(
    binding: ExecutionCaseSplitBinding,
    risk: ExecutionCaseContaminationRisk,
) -> ExecutionCaseSplitAdmission:
    disposition = _split_disposition(split=binding.split, stratum=risk.stratum)
    return ExecutionCaseSplitAdmission(
        episode_id=binding.episode_id,
        split=binding.split,
        stratum=risk.stratum,
        eligible_for_train_use=disposition[0],
        eligible_for_dev_historical_all=disposition[1],
        eligible_for_dev_common_low_risk=disposition[2],
        eligible_for_test_historical_all=disposition[3],
        eligible_for_test_common_low_risk=disposition[4],
        eligible_for_primary_leaderboard=disposition[5],
        exclusion_reason=disposition[6],
    )


def build_execution_contamination_admission_manifest(
    *,
    manifest_id: str,
    contamination_strata: ExecutionContaminationStrataManifest,
    split_bindings: Iterable[ExecutionCaseSplitBinding],
) -> ExecutionContaminationAdmissionManifest:
    strata = ExecutionContaminationStrataManifest.model_validate_json(canonical_json_bytes(contamination_strata))
    bindings = _canonical_split_bindings(split_bindings)
    if tuple(item.episode_id for item in bindings) != tuple(item.episode_id for item in strata.cases):
        raise ExecutionContaminationAdmissionError(
            'split bindings must exactly cover the frozen contamination case universe'
        )
    risks: Mapping[str, ExecutionCaseContaminationRisk] = {item.episode_id: item for item in strata.cases}
    cases = tuple(_build_case_split_admission(binding, risks[binding.episode_id]) for binding in bindings)
    split_counts = tuple(
        ExecutionSplitAdmissionCount(
            split=split,
            case_count=sum(item.split == split for item in cases),
        )
        for split in (Split.TRAIN, Split.DEV, Split.TEST)
    )
    return ExecutionContaminationAdmissionManifest(
        manifest_id=manifest_id,
        policy_sha256=execution_contamination_admission_policy_sha256(),
        contamination_strata_manifest_sha256=execution_contamination_strata_manifest_sha256(strata),
        split_universe_sha256=execution_split_universe_sha256(bindings),
        cases=cases,
        case_count=len(cases),
        split_counts=split_counts,
        train_eligible_count=sum(item.eligible_for_train_use for item in cases),
        dev_historical_all_count=sum(item.eligible_for_dev_historical_all for item in cases),
        dev_common_low_risk_count=sum(item.eligible_for_dev_common_low_risk for item in cases),
        test_historical_all_count=sum(item.eligible_for_test_historical_all for item in cases),
        test_common_low_risk_count=sum(item.eligible_for_test_common_low_risk for item in cases),
        primary_leaderboard_count=sum(item.eligible_for_primary_leaderboard for item in cases),
        globally_excluded_count=sum(item.exclusion_reason is not None for item in cases),
    )


def execution_contamination_admission_manifest_sha256(
    manifest: ExecutionContaminationAdmissionManifest,
) -> str:
    validated = ExecutionContaminationAdmissionManifest.model_validate_json(canonical_json_bytes(manifest))
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


class ExecutionModelWeightDeclaration(StrictModel):
    """Required submitter disclosure; useful evidence, but never proof of clean weights."""

    schema_version: Literal['vaxreplay.clinical-execution-model-weight-declaration.dev-v0.1'] = (
        EXECUTION_MODEL_WEIGHT_DECLARATION_SCHEMA_VERSION
    )
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    declared_model_id: str = Field(min_length=1, max_length=500)
    declared_training_data_cutoff: date | None = None
    public_aact_or_linked_publication_exposure: ExposureDeclaration
    organizer_private_eval_material_exposure: ExposureDeclaration
    benchmark_specific_test_tuning: ExposureDeclaration
    machine_unlearning_attempted: bool
    declaration_basis: str = Field(min_length=1, max_length=2_000)
    submitted_by: str = Field(min_length=1, max_length=500)
    submitter_attests_truthful_to_best_of_knowledge: Literal[True] = True
    proves_clean_weights: Literal[False] = False


class ExecutionSystemContaminationReport(StrictModel):
    """Model-weight disclosure and private probe summary reported beside one fixed score."""

    schema_version: Literal['vaxreplay.clinical-execution-system-contamination-report.dev-v0.1'] = (
        EXECUTION_SYSTEM_CONTAMINATION_REPORT_SCHEMA_VERSION
    )
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_admission_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    declaration_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_probe_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    probe_status_counts: tuple[tuple[ExecutionSystemExposureStatus, int], ...]
    probe_coverage_status: Literal['complete', 'not_run']
    public_source_exposure: ExposureDeclaration
    private_eval_material_exposure: ExposureDeclaration
    benchmark_specific_test_tuning: ExposureDeclaration
    eligible_for_train_and_dev_experimentation: Literal[True] = True
    eligible_for_held_out_leaderboard: bool
    held_out_ineligibility_reasons: tuple[str, ...]
    case_denominator_changed: Literal[False] = False
    machine_unlearning_required: Literal[False] = False
    no_signal_proves_clean_weights: Literal[False] = False
    declaration_proves_clean_weights: Literal[False] = False
    residual_model_weight_contamination_possible: Literal[True] = True

    @field_validator('probe_status_counts')
    @classmethod
    def validate_status_counts(
        cls,
        value: tuple[tuple[ExecutionSystemExposureStatus, int], ...],
    ) -> tuple[tuple[ExecutionSystemExposureStatus, int], ...]:
        expected_statuses = tuple(ExecutionSystemExposureStatus)
        if tuple(item[0] for item in value) != expected_statuses or any(item[1] < 0 for item in value):
            raise ValueError('probe status counts must cover every status in fixed enum order')
        return value

    @model_validator(mode='after')
    def validate_admission(self) -> Self:
        expected_reasons: list[str] = []
        if self.private_eval_material_exposure == ExposureDeclaration.YES:
            expected_reasons.append('known_organizer_private_eval_material_exposure')
        if self.benchmark_specific_test_tuning == ExposureDeclaration.YES:
            expected_reasons.append('known_benchmark_specific_test_tuning')
        if self.held_out_ineligibility_reasons != tuple(expected_reasons):
            raise ValueError('held-out ineligibility reasons do not follow the fixed policy')
        if self.eligible_for_held_out_leaderboard != (not expected_reasons):
            raise ValueError('held-out eligibility does not follow the fixed policy')
        if self.probe_coverage_status == 'not_run' and any(count for _, count in self.probe_status_counts):
            raise ValueError('a probe that was not run cannot report evaluated cases')
        if self.probe_coverage_status == 'complete' and not any(count for _, count in self.probe_status_counts):
            raise ValueError('a complete probe must report at least one evaluated case')
        return self


def execution_model_weight_declaration_sha256(
    declaration: ExecutionModelWeightDeclaration,
) -> str:
    validated = ExecutionModelWeightDeclaration.model_validate_json(canonical_json_bytes(declaration))
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


def build_execution_system_contamination_report(
    *,
    case_admission: ExecutionContaminationAdmissionManifest,
    declaration: ExecutionModelWeightDeclaration,
    system_probe: ExecutionSystemProbeManifest | None,
) -> ExecutionSystemContaminationReport:
    admission = ExecutionContaminationAdmissionManifest.model_validate_json(canonical_json_bytes(case_admission))
    declared = ExecutionModelWeightDeclaration.model_validate_json(canonical_json_bytes(declaration))
    if system_probe is None:
        probe_hash = None
        probe_status = 'not_run'
        counts = tuple((status, 0) for status in ExecutionSystemExposureStatus)
    else:
        probe = ExecutionSystemProbeManifest.model_validate_json(canonical_json_bytes(system_probe))
        if probe.system_manifest_sha256 != declared.system_manifest_sha256:
            raise ExecutionContaminationAdmissionError('system probe and declaration bind different systems')
        if probe.case_strata_manifest_sha256 != admission.contamination_strata_manifest_sha256:
            raise ExecutionContaminationAdmissionError(
                'system probe does not bind the case strata behind the admission manifest'
            )
        probe_hash = hashlib.sha256(canonical_json_bytes(probe)).hexdigest()
        probe_status = 'complete'
        counts = tuple((item.status, item.case_count) for item in probe.status_counts)
    reasons: list[str] = []
    if declared.organizer_private_eval_material_exposure == ExposureDeclaration.YES:
        reasons.append('known_organizer_private_eval_material_exposure')
    if declared.benchmark_specific_test_tuning == ExposureDeclaration.YES:
        reasons.append('known_benchmark_specific_test_tuning')
    return ExecutionSystemContaminationReport(
        system_manifest_sha256=declared.system_manifest_sha256,
        case_admission_manifest_sha256=execution_contamination_admission_manifest_sha256(admission),
        declaration_sha256=execution_model_weight_declaration_sha256(declared),
        system_probe_manifest_sha256=probe_hash,
        probe_status_counts=counts,
        probe_coverage_status=probe_status,
        public_source_exposure=declared.public_aact_or_linked_publication_exposure,
        private_eval_material_exposure=declared.organizer_private_eval_material_exposure,
        benchmark_specific_test_tuning=declared.benchmark_specific_test_tuning,
        eligible_for_held_out_leaderboard=not reasons,
        held_out_ineligibility_reasons=tuple(reasons),
    )


__all__ = [
    'EXECUTION_CONTAMINATION_ADMISSION_POLICY',
    'EXECUTION_CONTAMINATION_ADMISSION_POLICY_ID',
    'ExecutionCaseSplitAdmission',
    'ExecutionCaseSplitBinding',
    'ExecutionContaminationAdmissionError',
    'ExecutionContaminationAdmissionManifest',
    'ExecutionContaminationAdmissionPolicy',
    'ExecutionModelWeightDeclaration',
    'ExecutionSplitAdmissionCount',
    'ExecutionSystemContaminationReport',
    'ExposureDeclaration',
    'build_execution_contamination_admission_manifest',
    'build_execution_system_contamination_report',
    'execution_contamination_admission_manifest_sha256',
    'execution_contamination_admission_policy_sha256',
    'execution_model_weight_declaration_sha256',
    'execution_split_universe_sha256',
]
