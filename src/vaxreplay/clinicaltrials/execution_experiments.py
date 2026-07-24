"""Provider-neutral Lane A harness/model experiment manifests and offline runner.

The standardized matrix crosses in-repository harnesses with organizer-gateway model routes.  Local
vendor CLIs are declared separately as combined systems because their internal routing and usage
cannot be factorized into a clean harness-versus-model comparison.  Every planned task produces a
terminal row; failures remain in reports and valid-only means are never emitted.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_process_reward import (
    MAXIMUM_CREDITABLE_PROCESS_REWARD,
    lane_a_process_reward_policy_sha256,
)

LANE_A_EXPERIMENT_MANIFEST_SCHEMA_VERSION = 'vaxreplay.lane-a-experiment-manifest.dev-v0.1'
LANE_A_EXPERIMENT_REPORT_SCHEMA_VERSION = 'vaxreplay.lane-a-experiment-report.dev-v0.1'
LANE_A_EXPERIMENT_POLICY_ID = 'lane-a-paired-complete-matrix-v0.1'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class LaneAExperimentError(ValueError):
    """An experiment would violate its preregistered complete paired design."""


class LaneAHarnessKind(str, enum.Enum):
    UNIFORM = 'uniform'
    SINGLE_CALL = 'single_call'
    RETRIEVAL_AGENT = 'retrieval_agent'
    COMPUTE_AGENT = 'compute_agent'


_HARNESS_ORDER = (
    LaneAHarnessKind.UNIFORM,
    LaneAHarnessKind.SINGLE_CALL,
    LaneAHarnessKind.RETRIEVAL_AGENT,
    LaneAHarnessKind.COMPUTE_AGENT,
)


class LaneAExperimentTerminalStatus(str, enum.Enum):
    COMPLETED = 'completed'
    FAILED = 'failed'
    TIMED_OUT = 'timed_out'
    INVALID_SUBMISSION = 'invalid_submission'


class LaneAExperimentRunMode(str, enum.Enum):
    OFFLINE_FAKE = 'offline_fake'
    LIVE_SYNTHETIC = 'live_synthetic'


class LaneAExperimentBudget(StrictModel):
    budget_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    max_model_calls: int = Field(ge=0, le=100)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_reasoning_tokens: int = Field(ge=0)
    wall_seconds: int = Field(gt=0)
    maximum_provider_cost_usd: float = Field(ge=0.0, allow_inf_nan=False)


def lane_a_experiment_budget_sha256(budget: LaneAExperimentBudget) -> str:
    return hashlib.sha256(canonical_json_bytes(budget)).hexdigest()


class LaneAHarnessSpec(StrictModel):
    kind: LaneAHarnessKind
    harness_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    harness_version: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_gateway_only: Literal[True] = True
    provider_credentials_available_to_harness: Literal[False] = False
    in_repository_standardized_harness: Literal[True] = True
    expected_model_calls: int = Field(ge=0, le=100)
    route_independent_uniform_baseline: bool = False

    @model_validator(mode='after')
    def validate_kind(self) -> Self:
        if (self.kind == LaneAHarnessKind.UNIFORM) != self.route_independent_uniform_baseline:
            raise ValueError('only the uniform harness may be route-independent')
        if self.kind == LaneAHarnessKind.UNIFORM and self.expected_model_calls != 0:
            raise ValueError('uniform harness cannot call a model')
        if self.kind != LaneAHarnessKind.UNIFORM and self.expected_model_calls == 0:
            raise ValueError('model-backed standardized harnesses require at least one call')
        return self


class LaneAModelRoute(StrictModel):
    route_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    provider: str = Field(min_length=1)
    requested_model_id: str = Field(min_length=1)
    expected_resolved_model_id: str = Field(min_length=1)
    route_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_gateway_route: Literal[True] = True
    provider_usage_authoritative: Literal[True] = True
    provider_credentials_available_to_harness: Literal[False] = False


class LaneAExternalCliSystem(StrictModel):
    system_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    cli_name: str = Field(min_length=1)
    cli_version: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    requested_model_id: str = Field(min_length=1)
    resolved_model_id: str | None = None
    combined_harness_model_system: Literal[True] = True
    provider_route_unattested: Literal[True] = True
    excluded_from_harness_model_factorization: Literal[True] = True
    real_aact_task_transmission_allowed: Literal[False] = False


class LaneAExperimentCell(StrictModel):
    cell_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    harness_kind: LaneAHarnessKind
    harness_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    budget_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_order: tuple[str, ...] = Field(min_length=1)
    paired_task_order: Literal[True] = True
    no_adaptive_task_selection: Literal[True] = True

    @model_validator(mode='after')
    def validate_tasks(self) -> Self:
        if len(self.task_order) != len(set(self.task_order)):
            raise ValueError('experiment cell task order must be unique')
        return self


class LaneAExperimentManifest(StrictModel):
    schema_version: Literal['vaxreplay.lane-a-experiment-manifest.dev-v0.1'] = LANE_A_EXPERIMENT_MANIFEST_SCHEMA_VERSION
    policy_id: Literal['lane-a-paired-complete-matrix-v0.1'] = LANE_A_EXPERIMENT_POLICY_ID
    experiment_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    task_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    process_reward_policy_sha256: str = Field(
        default_factory=lane_a_process_reward_policy_sha256, pattern=_SHA256_PATTERN
    )
    task_ids: tuple[str, ...] = Field(min_length=1)
    task_data_class: Literal['fictional_or_synthetic'] = 'fictional_or_synthetic'
    harnesses: tuple[LaneAHarnessSpec, ...] = Field(min_length=4, max_length=4)
    model_routes: tuple[LaneAModelRoute, ...] = Field(min_length=1)
    external_cli_systems: tuple[LaneAExternalCliSystem, ...] = ()
    budget: LaneAExperimentBudget
    cells: tuple[LaneAExperimentCell, ...] = Field(min_length=4)
    paired_tasks: Literal[True] = True
    paired_budget: Literal[True] = True
    paired_order: Literal[True] = True
    complete_cartesian_matrix_required: Literal[True] = True
    terminal_failures_retained: Literal[True] = True
    valid_only_means_forbidden: Literal[True] = True
    adaptive_cell_or_task_dropping_forbidden: Literal[True] = True
    outcome_and_process_rewards_reported_separately: Literal[True] = True
    process_reward_auxiliary_only: Literal[True] = True
    process_reward_maximum_creditable: float = Field(
        default=MAXIMUM_CREDITABLE_PROCESS_REWARD,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    outcome_reward_remains_primary: Literal[True] = True
    combined_scalar_emitted: Literal[False] = False
    real_aact_task_transmission_allowed: Literal[False] = False
    live_execution_requires_fictional_or_synthetic_tasks: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_design(self) -> Self:
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError('experiment task IDs must be unique')
        if self.process_reward_policy_sha256 != lane_a_process_reward_policy_sha256() or not math.isclose(
            self.process_reward_maximum_creditable,
            MAXIMUM_CREDITABLE_PROCESS_REWARD,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('experiment must pin the current auxiliary process-reward policy')
        if tuple(item.kind for item in self.harnesses) != _HARNESS_ORDER:
            raise ValueError('standardized harnesses must use the fixed four-harness order')
        harness_ids = tuple(item.harness_id for item in self.harnesses)
        route_ids = tuple(item.route_id for item in self.model_routes)
        external_ids = tuple(item.system_id for item in self.external_cli_systems)
        if len(harness_ids) != len(set(harness_ids)):
            raise ValueError('standardized harness IDs must be unique')
        if route_ids != tuple(sorted(set(route_ids))):
            raise ValueError('model routes must have unique canonical order')
        if external_ids != tuple(sorted(set(external_ids))):
            raise ValueError('external combined-system IDs must have unique canonical order')
        expected_pairs = tuple((harness.kind, route_id) for harness in self.harnesses for route_id in route_ids)
        observed_pairs = tuple((cell.harness_kind, cell.route_id) for cell in self.cells)
        if observed_pairs != expected_pairs:
            raise ValueError('experiment cells must contain the complete fixed harness/route matrix')
        budget_sha = lane_a_experiment_budget_sha256(self.budget)
        harness_by_kind = {item.kind: item for item in self.harnesses}
        for cell in self.cells:
            harness = harness_by_kind[cell.harness_kind]
            if (
                cell.harness_id != harness.harness_id
                or cell.budget_sha256 != budget_sha
                or cell.task_order != self.task_ids
            ):
                raise ValueError('every cell must use the same pinned harness, budget, tasks, and order')
        return self


def lane_a_experiment_manifest_sha256(manifest: LaneAExperimentManifest) -> str:
    manifest = LaneAExperimentManifest.model_validate_json(canonical_json_bytes(manifest))
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def make_lane_a_experiment_manifest(
    *,
    experiment_id: str,
    task_ids: Sequence[str],
    task_set_sha256: str,
    harnesses: Sequence[LaneAHarnessSpec],
    model_routes: Sequence[LaneAModelRoute],
    budget: LaneAExperimentBudget,
    external_cli_systems: Sequence[LaneAExternalCliSystem] = (),
) -> LaneAExperimentManifest:
    ordered_routes = tuple(sorted(model_routes, key=lambda item: item.route_id))
    task_order = tuple(task_ids)
    budget_sha = lane_a_experiment_budget_sha256(budget)
    cells = tuple(
        LaneAExperimentCell(
            cell_id=f'{harness.kind.value}--{route.route_id}',
            harness_kind=harness.kind,
            harness_id=harness.harness_id,
            route_id=route.route_id,
            budget_sha256=budget_sha,
            task_order=task_order,
        )
        for harness in harnesses
        for route in ordered_routes
    )
    return LaneAExperimentManifest(
        experiment_id=experiment_id,
        task_set_sha256=task_set_sha256,
        task_ids=task_order,
        harnesses=tuple(harnesses),
        model_routes=ordered_routes,
        external_cli_systems=tuple(sorted(external_cli_systems, key=lambda item: item.system_id)),
        budget=budget,
        cells=cells,
    )


class LaneAExperimentUsage(StrictModel):
    model_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    provider_cost_usd: float = Field(ge=0.0, allow_inf_nan=False)
    wall_milliseconds: int = Field(ge=0)
    provider_metering_authoritative: bool


class LaneAExperimentObservation(StrictModel):
    terminal_status: LaneAExperimentTerminalStatus
    resolved_model_id: str | None = None
    usage: LaneAExperimentUsage
    process_reward: float | None = Field(
        default=None,
        ge=0.0,
        le=MAXIMUM_CREDITABLE_PROCESS_REWARD,
        allow_inf_nan=False,
    )
    outcome_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    failure_code: str | None = None

    @model_validator(mode='after')
    def validate_terminal(self) -> Self:
        completed = self.terminal_status == LaneAExperimentTerminalStatus.COMPLETED
        if completed == (self.failure_code is not None):
            raise ValueError('completed observations cannot fail; terminal failures require a code')
        return self


class LaneAExperimentAttempt(StrictModel):
    attempt_index: int = Field(ge=0)
    cell_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    harness_kind: LaneAHarnessKind
    harness_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    requested_model_id: str = Field(min_length=1)
    resolved_model_id: str | None = None
    run_mode: LaneAExperimentRunMode
    terminal_status: LaneAExperimentTerminalStatus
    failure_code: str | None = None
    usage: LaneAExperimentUsage
    process_reward: float | None = Field(
        default=None,
        ge=0.0,
        le=MAXIMUM_CREDITABLE_PROCESS_REWARD,
        allow_inf_nan=False,
    )
    process_reward_auxiliary_only: Literal[True] = True
    outcome_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    combined_reward: None = None
    terminal_failure_retained: Literal[True] = True
    real_aact_task_transmitted: Literal[False] = False


class LaneAExperimentCellSummary(StrictModel):
    cell_id: str = Field(min_length=1)
    task_count: int = Field(gt=0)
    terminal_failure_count: int = Field(ge=0)
    terminal_failure_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_process_reward_all_tasks: float = Field(
        ge=0.0,
        le=MAXIMUM_CREDITABLE_PROCESS_REWARD,
        allow_inf_nan=False,
    )
    mean_outcome_reward_all_tasks: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    total_model_calls: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_reasoning_tokens: int = Field(ge=0)
    total_provider_cost_usd: float = Field(ge=0.0, allow_inf_nan=False)
    total_wall_milliseconds: int = Field(ge=0)
    valid_only_mean_emitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_failure_rate(self) -> Self:
        if self.terminal_failure_count > self.task_count or not math.isclose(
            self.terminal_failure_rate,
            self.terminal_failure_count / self.task_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('cell failure rate does not match retained terminal attempts')
        return self


class LaneAExternalCompatibilityResult(StrictModel):
    system_id: str = Field(min_length=1)
    status: Literal['not_run', 'offline_fake_pass', 'offline_fake_fail', 'live_synthetic_pass', 'live_synthetic_fail']
    requested_model_id: str = Field(min_length=1)
    resolved_model_id: str | None = None
    receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    diagnostic_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    grounding_reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    assessment_accuracy: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    one_shot_diagnostic: Literal[True] = True
    comparative_inference_allowed: Literal[False] = False
    provider_route_attested: Literal[False] = False
    real_aact_task_transmitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_status_evidence(self) -> Self:
        diagnostics = (
            self.diagnostic_reward,
            self.grounding_reward,
            self.assessment_accuracy,
        )
        if self.status == 'not_run':
            if self.receipt_sha256 is not None or any(value is not None for value in diagnostics):
                raise ValueError('an external diagnostic that was not run cannot carry evidence')
        elif self.status.endswith('_pass'):
            if self.receipt_sha256 is None or not all(value is not None for value in diagnostics):
                raise ValueError('passing external diagnostics require a receipt and complete metric fields')
        elif self.receipt_sha256 is None or any(value is not None for value in diagnostics):
            raise ValueError('failed external diagnostics retain a receipt but cannot invent score metrics')
        return self


class LaneAExperimentReport(StrictModel):
    schema_version: Literal['vaxreplay.lane-a-experiment-report.dev-v0.1'] = LANE_A_EXPERIMENT_REPORT_SCHEMA_VERSION
    experiment_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    process_reward_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_mode: LaneAExperimentRunMode
    attempts: tuple[LaneAExperimentAttempt, ...]
    cell_summaries: tuple[LaneAExperimentCellSummary, ...]
    external_compatibility: tuple[LaneAExternalCompatibilityResult, ...]
    expected_attempt_count: int = Field(gt=0)
    observed_attempt_count: int = Field(ge=0)
    complete_matrix_executed: Literal[True] = True
    all_terminal_attempts_retained: Literal[True] = True
    missing_attempts: Literal[0] = 0
    cherry_picked_attempts: Literal[0] = 0
    valid_only_means_emitted: Literal[False] = False
    outcome_and_process_rewards_separate: Literal[True] = True
    process_reward_auxiliary_only: Literal[True] = True
    outcome_reward_remains_primary: Literal[True] = True
    combined_scalar_emitted: Literal[False] = False
    real_aact_task_transmitted: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_coverage(self) -> Self:
        if self.process_reward_policy_sha256 != lane_a_process_reward_policy_sha256():
            raise ValueError('experiment report process policy differs from the current fixed policy')
        if (
            self.observed_attempt_count != len(self.attempts)
            or self.observed_attempt_count != self.expected_attempt_count
        ):
            raise ValueError('experiment report must retain every planned terminal attempt')
        indexes = tuple(item.attempt_index for item in self.attempts)
        if indexes != tuple(range(len(self.attempts))):
            raise ValueError('experiment attempt indexes must be contiguous')
        return self


class LaneAExperimentBackend(Protocol):
    @property
    def run_mode(self) -> LaneAExperimentRunMode: ...

    def run(
        self,
        *,
        manifest: LaneAExperimentManifest,
        cell: LaneAExperimentCell,
        task_id: str,
    ) -> LaneAExperimentObservation: ...


@dataclass(frozen=True)
class OfflineFakeLaneABackend:
    """Deterministic conformance backend; it never contacts a provider or opens a real task."""

    fail_attempts: frozenset[tuple[str, str]] = frozenset()
    run_mode: LaneAExperimentRunMode = LaneAExperimentRunMode.OFFLINE_FAKE

    def run(
        self,
        *,
        manifest: LaneAExperimentManifest,
        cell: LaneAExperimentCell,
        task_id: str,
    ) -> LaneAExperimentObservation:
        route = next(item for item in manifest.model_routes if item.route_id == cell.route_id)
        harness = next(item for item in manifest.harnesses if item.harness_id == cell.harness_id)
        calls = harness.expected_model_calls
        base = {
            LaneAHarnessKind.UNIFORM: 0.20,
            LaneAHarnessKind.SINGLE_CALL: 0.40,
            LaneAHarnessKind.RETRIEVAL_AGENT: 0.60,
            LaneAHarnessKind.COMPUTE_AGENT: MAXIMUM_CREDITABLE_PROCESS_REWARD,
        }[cell.harness_kind]
        usage = LaneAExperimentUsage(
            model_calls=calls,
            input_tokens=calls * 120,
            output_tokens=calls * 30,
            reasoning_tokens=calls * 20,
            provider_cost_usd=calls * 0.001,
            wall_milliseconds=25 + calls * 10,
            provider_metering_authoritative=False,
        )
        if (cell.cell_id, task_id) in self.fail_attempts:
            return LaneAExperimentObservation(
                terminal_status=LaneAExperimentTerminalStatus.FAILED,
                resolved_model_id=route.expected_resolved_model_id,
                usage=usage,
                process_reward=base / 2,
                failure_code='offline_scripted_failure',
            )
        return LaneAExperimentObservation(
            terminal_status=LaneAExperimentTerminalStatus.COMPLETED,
            resolved_model_id=route.expected_resolved_model_id,
            usage=usage,
            process_reward=base,
        )


def _cell_summary(cell: LaneAExperimentCell, attempts: Sequence[LaneAExperimentAttempt]) -> LaneAExperimentCellSummary:
    failures = sum(item.terminal_status != LaneAExperimentTerminalStatus.COMPLETED for item in attempts)
    outcome_values = tuple(item.outcome_reward for item in attempts)
    outcome_mean = None
    if all(value is not None for value in outcome_values):
        outcome_mean = math.fsum(value for value in outcome_values if value is not None) / len(attempts)
    return LaneAExperimentCellSummary(
        cell_id=cell.cell_id,
        task_count=len(attempts),
        terminal_failure_count=failures,
        terminal_failure_rate=failures / len(attempts),
        mean_process_reward_all_tasks=math.fsum(item.process_reward or 0.0 for item in attempts) / len(attempts),
        mean_outcome_reward_all_tasks=outcome_mean,
        total_model_calls=sum(item.usage.model_calls for item in attempts),
        total_input_tokens=sum(item.usage.input_tokens for item in attempts),
        total_output_tokens=sum(item.usage.output_tokens for item in attempts),
        total_reasoning_tokens=sum(item.usage.reasoning_tokens for item in attempts),
        total_provider_cost_usd=math.fsum(item.usage.provider_cost_usd for item in attempts),
        total_wall_milliseconds=sum(item.usage.wall_milliseconds for item in attempts),
    )


def _usage_exceeds_budget(usage: LaneAExperimentUsage, budget: LaneAExperimentBudget) -> bool:
    return (
        usage.model_calls > budget.max_model_calls
        or usage.input_tokens > budget.max_input_tokens
        or usage.output_tokens > budget.max_output_tokens
        or usage.reasoning_tokens > budget.max_reasoning_tokens
        or usage.provider_cost_usd > budget.maximum_provider_cost_usd
        or usage.wall_milliseconds > budget.wall_seconds * 1000
    )


def _observation_limit_failure(
    observation: LaneAExperimentObservation,
    *,
    route: LaneAModelRoute,
    budget: LaneAExperimentBudget,
) -> str | None:
    if _usage_exceeds_budget(observation.usage, budget):
        return 'budget_exceeded'
    if (
        observation.terminal_status == LaneAExperimentTerminalStatus.COMPLETED
        and observation.resolved_model_id != route.expected_resolved_model_id
    ):
        return 'resolved_model_mismatch'
    return None


def verify_lane_a_experiment_report(
    report: LaneAExperimentReport,
    *,
    manifest: LaneAExperimentManifest,
) -> LaneAExperimentReport:
    """Verify that a report exactly realizes its preregistered matrix and summaries."""

    report = LaneAExperimentReport.model_validate_json(canonical_json_bytes(report))
    manifest = LaneAExperimentManifest.model_validate_json(canonical_json_bytes(manifest))
    if report.manifest_sha256 != lane_a_experiment_manifest_sha256(manifest):
        raise LaneAExperimentError('experiment report is bound to a different manifest')
    expected: list[tuple[LaneAExperimentCell, str]] = [
        (cell, task_id) for cell in manifest.cells for task_id in cell.task_order
    ]
    if len(report.attempts) != len(expected):
        raise LaneAExperimentError('experiment report does not cover every manifest cell/task pair')
    harness_by_id = {item.harness_id: item for item in manifest.harnesses}
    route_by_id = {item.route_id: item for item in manifest.model_routes}
    for attempt, (cell, task_id) in zip(report.attempts, expected, strict=True):
        route = route_by_id[cell.route_id]
        if (
            attempt.cell_id,
            attempt.task_id,
            attempt.harness_kind,
            attempt.harness_id,
            attempt.route_id,
            attempt.requested_model_id,
            attempt.run_mode,
        ) != (
            cell.cell_id,
            task_id,
            cell.harness_kind,
            cell.harness_id,
            cell.route_id,
            route.requested_model_id,
            report.run_mode,
        ):
            raise LaneAExperimentError('experiment attempt order or cell/task binding differs from the manifest')
        harness = harness_by_id[cell.harness_id]
        completed = attempt.terminal_status == LaneAExperimentTerminalStatus.COMPLETED
        over_budget = _usage_exceeds_budget(attempt.usage, manifest.budget)
        if over_budget != (
            attempt.terminal_status == LaneAExperimentTerminalStatus.FAILED
            and attempt.failure_code == 'budget_exceeded'
        ):
            raise LaneAExperimentError('experiment budget status does not match retained authoritative usage')
        if completed and attempt.resolved_model_id != route.expected_resolved_model_id:
            raise LaneAExperimentError('completed experiment row has the wrong resolved model identity')
        if completed and attempt.usage.model_calls != harness.expected_model_calls:
            raise LaneAExperimentError('completed experiment row has the wrong standardized model-call count')
        if (
            completed
            and report.run_mode == LaneAExperimentRunMode.LIVE_SYNTHETIC
            and not attempt.usage.provider_metering_authoritative
        ):
            raise LaneAExperimentError('completed live experiment row lacks authoritative provider metering')
        if attempt.process_reward is not None and attempt.process_reward > manifest.process_reward_maximum_creditable:
            raise LaneAExperimentError('experiment attempt exceeds the pinned process-reward ceiling')

    cursor = 0
    expected_summaries: list[LaneAExperimentCellSummary] = []
    for cell in manifest.cells:
        cell_attempts = report.attempts[cursor : cursor + len(cell.task_order)]
        expected_summaries.append(_cell_summary(cell, cell_attempts))
        cursor += len(cell.task_order)
    observed_summary_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in report.cell_summaries])
    expected_summary_bytes = canonical_json_bytes([item.model_dump(mode='json') for item in expected_summaries])
    if observed_summary_bytes != expected_summary_bytes:
        raise LaneAExperimentError('experiment cell summaries do not rederive from retained attempts')

    external_by_id = {item.system_id: item for item in manifest.external_cli_systems}
    if tuple(item.system_id for item in report.external_compatibility) != tuple(external_by_id):
        raise LaneAExperimentError('experiment external rows differ from the declared combined systems')
    for result in report.external_compatibility:
        declared = external_by_id[result.system_id]
        if result.requested_model_id != declared.requested_model_id or (
            declared.resolved_model_id is not None and result.resolved_model_id != declared.resolved_model_id
        ):
            raise LaneAExperimentError('external diagnostic model identity differs from its declaration')
    return report


def run_lane_a_experiment(
    manifest: LaneAExperimentManifest,
    *,
    backend: LaneAExperimentBackend,
) -> LaneAExperimentReport:
    """Execute the preregistered complete matrix, retaining an attempt after every terminal path."""

    manifest = LaneAExperimentManifest.model_validate_json(canonical_json_bytes(manifest))
    route_by_id = {item.route_id: item for item in manifest.model_routes}
    attempts: list[LaneAExperimentAttempt] = []
    summaries: list[LaneAExperimentCellSummary] = []
    for cell in manifest.cells:
        cell_attempts: list[LaneAExperimentAttempt] = []
        for task_id in cell.task_order:
            try:
                observation = backend.run(manifest=manifest, cell=cell, task_id=task_id)
                observation = LaneAExperimentObservation.model_validate_json(canonical_json_bytes(observation))
            except Exception:
                observation = LaneAExperimentObservation(
                    terminal_status=LaneAExperimentTerminalStatus.FAILED,
                    usage=LaneAExperimentUsage(
                        model_calls=0,
                        input_tokens=0,
                        output_tokens=0,
                        reasoning_tokens=0,
                        provider_cost_usd=0.0,
                        wall_milliseconds=0,
                        provider_metering_authoritative=False,
                    ),
                    failure_code='backend_exception',
                )
            route = route_by_id[cell.route_id]
            if failure_code := _observation_limit_failure(observation, route=route, budget=manifest.budget):
                observation = LaneAExperimentObservation(
                    terminal_status=LaneAExperimentTerminalStatus.FAILED,
                    resolved_model_id=observation.resolved_model_id,
                    usage=observation.usage,
                    process_reward=observation.process_reward,
                    failure_code=failure_code,
                )
            attempt = LaneAExperimentAttempt(
                attempt_index=len(attempts),
                cell_id=cell.cell_id,
                task_id=task_id,
                harness_kind=cell.harness_kind,
                harness_id=cell.harness_id,
                route_id=cell.route_id,
                requested_model_id=route.requested_model_id,
                resolved_model_id=observation.resolved_model_id,
                run_mode=backend.run_mode,
                terminal_status=observation.terminal_status,
                failure_code=observation.failure_code,
                usage=observation.usage,
                process_reward=observation.process_reward,
                outcome_reward=observation.outcome_reward,
            )
            attempts.append(attempt)
            cell_attempts.append(attempt)
        summaries.append(_cell_summary(cell, cell_attempts))
    external = tuple(
        LaneAExternalCompatibilityResult(
            system_id=item.system_id,
            status='not_run',
            requested_model_id=item.requested_model_id,
            resolved_model_id=item.resolved_model_id,
        )
        for item in manifest.external_cli_systems
    )
    expected_count = len(manifest.cells) * len(manifest.task_ids)
    report = LaneAExperimentReport(
        experiment_id=manifest.experiment_id,
        manifest_sha256=lane_a_experiment_manifest_sha256(manifest),
        process_reward_policy_sha256=manifest.process_reward_policy_sha256,
        run_mode=backend.run_mode,
        attempts=tuple(attempts),
        cell_summaries=tuple(summaries),
        external_compatibility=external,
        expected_attempt_count=expected_count,
        observed_attempt_count=len(attempts),
    )
    return verify_lane_a_experiment_report(report, manifest=manifest)


def attach_external_compatibility_results(
    report: LaneAExperimentReport,
    *,
    manifest: LaneAExperimentManifest,
    results: Sequence[LaneAExternalCompatibilityResult],
) -> LaneAExperimentReport:
    """Attach non-comparative fictional diagnostics without changing standardized attempts."""

    report = verify_lane_a_experiment_report(report, manifest=manifest)
    manifest = LaneAExperimentManifest.model_validate_json(canonical_json_bytes(manifest))
    if report.manifest_sha256 != lane_a_experiment_manifest_sha256(manifest):
        raise LaneAExperimentError('external diagnostics are bound to a different experiment manifest')
    expected_ids = tuple(item.system_id for item in manifest.external_cli_systems)
    ordered = tuple(sorted(results, key=lambda item: item.system_id))
    if tuple(item.system_id for item in ordered) != expected_ids:
        raise LaneAExperimentError('external diagnostics must exactly cover declared combined systems')
    updated = LaneAExperimentReport.model_validate_json(
        canonical_json_bytes(report.model_copy(update={'external_compatibility': ordered}))
    )
    return verify_lane_a_experiment_report(updated, manifest=manifest)


__all__ = [
    'LANE_A_EXPERIMENT_POLICY_ID',
    'LaneAExperimentAttempt',
    'LaneAExperimentBackend',
    'LaneAExperimentBudget',
    'LaneAExperimentCell',
    'LaneAExperimentCellSummary',
    'LaneAExperimentError',
    'LaneAExperimentManifest',
    'LaneAExperimentObservation',
    'LaneAExperimentReport',
    'LaneAExperimentRunMode',
    'LaneAExperimentTerminalStatus',
    'LaneAExperimentUsage',
    'LaneAExternalCliSystem',
    'LaneAExternalCompatibilityResult',
    'LaneAHarnessKind',
    'LaneAHarnessSpec',
    'LaneAModelRoute',
    'OfflineFakeLaneABackend',
    'attach_external_compatibility_results',
    'lane_a_experiment_budget_sha256',
    'lane_a_experiment_manifest_sha256',
    'make_lane_a_experiment_manifest',
    'run_lane_a_experiment',
    'verify_lane_a_experiment_report',
]
