"""Immutable contracts for VaxReplay reward QA and gradient admission."""

from __future__ import annotations

import enum
import hashlib
from datetime import datetime, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import RewardVersion, StrictModel

REWARD_CONTRACT_SCHEMA_VERSION = 'vaxreplay.reward-contract.v0.1'
REWARD_QA_REPORT_SCHEMA_VERSION = 'vaxreplay.reward-qa-report.v0.1'
TRAINING_RUN_ADMISSION_SCHEMA_VERSION = 'vaxreplay.training-run-admission.v0.1'
GRADIENT_ADMISSION_TOKEN_SCHEMA_VERSION = 'vaxreplay.gradient-admission-token.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _unique_sorted(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f'{field_name} must be unique')
    if values != tuple(sorted(values)):
        raise ValueError(f'{field_name} must use canonical sorted order')
    return values


class QADisposition(str, enum.Enum):
    ADMIT = 'admit'
    QUARANTINE = 'quarantine'
    REJECT = 'reject'


class QASeverity(str, enum.Enum):
    INFO = 'info'
    WARNING = 'warning'
    FATAL = 'fatal'


class QACategory(str, enum.Enum):
    PARSER_INTEGRITY = 'parser_integrity'
    SCORER_INTEGRITY = 'scorer_integrity'
    TEMPORAL_INTEGRITY = 'temporal_integrity'
    COMPONENT_INTEGRITY = 'component_integrity'
    COUNTERFACTUAL_CONSISTENCY = 'counterfactual_consistency'
    PARAMETRIC_MEMORY = 'parametric_memory'
    EVALUATOR_INTEGRITY = 'evaluator_integrity'
    RESOURCE_INTEGRITY = 'resource_integrity'


class ComponentFloorSpec(StrictModel):
    metric: str = Field(pattern=r'^[a-z][a-z0-9_]*$')
    minimum: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class RewardContract(StrictModel):
    """Precommitted scientific objective and non-collapsible reward constraints."""

    schema_version: Literal['vaxreplay.reward-contract.v0.1'] = REWARD_CONTRACT_SCHEMA_VERSION
    contract_id: str = Field(min_length=1)
    reward_version: RewardVersion
    scientific_objective: str = Field(min_length=1)
    scorer_sha256: str = Field(pattern=_SHA256_PATTERN)
    reference_scorer_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_compiler_sha256: str = Field(pattern=_SHA256_PATTERN)
    qa_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    attack_catalog_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_dimensions: tuple[str, ...] = Field(min_length=1)
    required_attack_ids: tuple[str, ...] = Field(min_length=1)
    component_floors: tuple[ComponentFloorSpec, ...] = Field(min_length=1)
    prohibited_strategies: tuple[str, ...] = Field(min_length=1)
    deterministic_scorer_required: Literal[True] = True
    independent_reference_scorer_required: Literal[True] = True
    aggregate_reward_cannot_override_veto: Literal[True] = True
    qa_signals_excluded_from_reward: Literal[True] = True
    invalid_or_quarantined_trajectories_excluded_from_training: Literal[True] = True
    item_level_private_feedback_to_actor: Literal[False] = False

    @field_validator('required_dimensions', 'required_attack_ids', 'prohibited_strategies')
    @classmethod
    def validate_sorted_strings(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError(f'{info.field_name} cannot contain blank values')
        return _unique_sorted(value, info.field_name)

    @field_validator('component_floors')
    @classmethod
    def validate_component_floors(
        cls,
        value: tuple[ComponentFloorSpec, ...],
    ) -> tuple[ComponentFloorSpec, ...]:
        names = tuple(item.metric for item in value)
        _unique_sorted(names, 'component_floors metrics')
        return value

    @model_validator(mode='after')
    def validate_independent_scorers(self) -> Self:
        if self.scorer_sha256 == self.reference_scorer_sha256:
            raise ValueError('primary and reference scorer hashes must differ')
        return self


class RewardQAFinding(StrictModel):
    finding_id: str = Field(pattern=r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')
    check_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    category: QACategory
    severity: QASeverity
    passed: bool
    disposition_on_failure: Literal[QADisposition.QUARANTINE, QADisposition.REJECT]
    detail: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    hidden_control: bool = True

    @model_validator(mode='after')
    def validate_severity(self) -> Self:
        if (
            not self.passed
            and self.severity == QASeverity.FATAL
            and self.disposition_on_failure != QADisposition.REJECT
        ):
            raise ValueError('failed fatal findings must reject')
        return self


class ComponentScore(StrictModel):
    metric: str = Field(pattern=r'^[a-z][a-z0-9_]*$')
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class RewardQAReport(StrictModel):
    """Vector QA result whose disposition is derived, not author-selected."""

    schema_version: Literal['vaxreplay.reward-qa-report.v0.1'] = REWARD_QA_REPORT_SCHEMA_VERSION
    report_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    trajectory_batch_sha256: str = Field(pattern=_SHA256_PATTERN)
    reward_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_manifest_sha256s: tuple[str, ...] = Field(min_length=1)
    reward_contract: RewardContract
    reward_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    component_scores: tuple[ComponentScore, ...] = Field(min_length=1)
    findings: tuple[RewardQAFinding, ...] = Field(min_length=1)
    disposition: QADisposition
    all_required_attacks_executed: bool
    independent_scorer_agreement: bool
    future_taint_reachability_zero: bool
    exact_replay: bool
    tamper_success_zero: bool
    item_level_private_feedback_withheld: bool

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _aware_utc(value, 'created_at')

    @field_validator('episode_manifest_sha256s')
    @classmethod
    def validate_episode_manifests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for digest in value:
            if len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest):
                raise ValueError('episode_manifest_sha256s must contain lowercase SHA-256 digests')
        return _unique_sorted(value, 'episode_manifest_sha256s')

    @field_validator('component_scores')
    @classmethod
    def validate_component_scores(cls, value: tuple[ComponentScore, ...]) -> tuple[ComponentScore, ...]:
        metrics = tuple(item.metric for item in value)
        _unique_sorted(metrics, 'component score metrics')
        return value

    @field_validator('findings')
    @classmethod
    def validate_findings(cls, value: tuple[RewardQAFinding, ...]) -> tuple[RewardQAFinding, ...]:
        finding_ids = tuple(item.finding_id for item in value)
        _unique_sorted(finding_ids, 'finding IDs')
        return value

    @model_validator(mode='after')
    def validate_report(self) -> Self:
        if self.reward_contract_sha256 != reward_contract_sha256(self.reward_contract):
            raise ValueError('reward_contract_sha256 does not bind the embedded contract')
        observed_checks = {finding.check_id for finding in self.findings}
        required_checks = set(self.reward_contract.required_attack_ids)
        if not required_checks.issubset(observed_checks):
            raise ValueError(f'QA report is missing required attack checks {sorted(required_checks - observed_checks)}')

        component_by_name = {item.metric: item.value for item in self.component_scores}
        component_failure = False
        for floor in self.reward_contract.component_floors:
            observed = component_by_name.get(floor.metric)
            if observed is None:
                raise ValueError(f'QA report omits required component score {floor.metric}')
            component_failure = component_failure or observed < floor.minimum

        hard_flags = (
            self.all_required_attacks_executed,
            self.independent_scorer_agreement,
            self.future_taint_reachability_zero,
            self.exact_replay,
            self.tamper_success_zero,
            self.item_level_private_feedback_withheld,
        )
        failed = tuple(finding for finding in self.findings if not finding.passed)
        if not all(hard_flags) or any(finding.disposition_on_failure == QADisposition.REJECT for finding in failed):
            expected = QADisposition.REJECT
        elif component_failure or failed:
            expected = QADisposition.QUARANTINE
        else:
            expected = QADisposition.ADMIT
        if self.disposition != expected:
            raise ValueError(f'QA disposition must be derived as {expected.value}')
        return self


class TrainingRunAdmission(StrictModel):
    """Exact batch/configuration grant that may be signed for gradient use."""

    schema_version: Literal['vaxreplay.training-run-admission.v0.1'] = TRAINING_RUN_ADMISSION_SCHEMA_VERSION
    admission_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    issued_at: datetime
    not_before: datetime
    expires_at: datetime
    trajectory_batch_sha256: str = Field(pattern=_SHA256_PATTERN)
    reward_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_sha256: str = Field(pattern=_SHA256_PATTERN)
    tool_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    optimizer_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_manifest_sha256s: tuple[str, ...] = Field(min_length=1)
    reward_qa_report: RewardQAReport
    reward_qa_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    reward_qa_attestation_sha256: str = Field(pattern=_SHA256_PATTERN)
    qa_signing_key_id: str = Field(pattern=_SHA256_PATTERN)
    reward_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    attack_catalog_sha256: str = Field(pattern=_SHA256_PATTERN)
    signing_key_id: str = Field(pattern=_SHA256_PATTERN)
    single_use: Literal[True] = True
    grants_gradient_access: Literal[True] = True
    item_level_feedback_to_actor: Literal[False] = False

    @field_validator('issued_at', 'not_before', 'expires_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @field_validator('episode_manifest_sha256s')
    @classmethod
    def validate_episode_manifests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for digest in value:
            if len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest):
                raise ValueError('episode_manifest_sha256s must contain lowercase SHA-256 digests')
        return _unique_sorted(value, 'episode_manifest_sha256s')

    @model_validator(mode='after')
    def validate_admission(self) -> Self:
        if not self.issued_at <= self.not_before < self.expires_at:
            raise ValueError('admission times must satisfy issued_at <= not_before < expires_at')
        if self.reward_qa_report.disposition != QADisposition.ADMIT:
            raise ValueError('only an admitted QA report can grant gradient access')
        if self.run_id != self.reward_qa_report.run_id:
            raise ValueError('admission run_id does not bind the QA report')
        if self.trajectory_batch_sha256 != self.reward_qa_report.trajectory_batch_sha256:
            raise ValueError('admission trajectory batch does not bind the QA report')
        if self.reward_artifact_sha256 != self.reward_qa_report.reward_artifact_sha256:
            raise ValueError('admission reward artifact does not bind the QA report')
        if self.episode_manifest_sha256s != self.reward_qa_report.episode_manifest_sha256s:
            raise ValueError('admission episode manifests do not bind the QA report')
        if self.reward_qa_report_sha256 != reward_qa_report_sha256(self.reward_qa_report):
            raise ValueError('reward_qa_report_sha256 does not bind the embedded report')
        if self.reward_contract_sha256 != self.reward_qa_report.reward_contract_sha256:
            raise ValueError('admission reward contract does not bind the QA report')
        if self.attack_catalog_sha256 != self.reward_qa_report.reward_contract.attack_catalog_sha256:
            raise ValueError('admission attack catalog does not bind the reward contract')
        if self.qa_signing_key_id == self.signing_key_id:
            raise ValueError('QA report and gradient admission signing identities must differ')
        return self


class GradientAdmissionToken(StrictModel):
    schema_version: Literal['vaxreplay.gradient-admission-token.v0.1'] = GRADIENT_ADMISSION_TOKEN_SCHEMA_VERSION
    token_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    training_run_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    signing_key_id: str = Field(pattern=_SHA256_PATTERN)
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    signature_base64: str = Field(min_length=88, max_length=88)


def reward_contract_sha256(value: RewardContract) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def reward_qa_report_sha256(value: RewardQAReport) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def training_run_admission_sha256(value: TrainingRunAdmission) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
