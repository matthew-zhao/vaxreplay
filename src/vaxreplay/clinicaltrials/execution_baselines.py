"""Public, label-blind reference submissions for clinical execution tasks.

These baselines are plumbing checks and calibration anchors, not model results.  They consume only
the public task contract.  In particular, they never inspect the private gold, organizer mapping,
or registry identity behind an opaque task.
"""

from __future__ import annotations

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_task import (
    ConditionalPointForecast,
    ConditionalQuantileForecast,
    ContinuousForecastSpec,
    ExecutionSubmission,
    ExecutionTask,
    ObservationStateProbabilities,
    QuantilePoint,
    RegistryOutcomeProbabilities,
)


def _midpoint_forecast(
    spec: ContinuousForecastSpec,
) -> ConditionalPointForecast | ConditionalQuantileForecast:
    midpoint = (spec.lower_bound + spec.upper_bound) / 2.0
    if spec.forecast_kind == 'point':
        return ConditionalPointForecast(value=midpoint)
    return ConditionalQuantileForecast(
        values=tuple(QuantilePoint(quantile=level, value=midpoint) for level in spec.quantile_levels)
    )


def uniform_execution_submission(task: ExecutionTask) -> ExecutionSubmission:
    """Return a valid public-only forecast with uniform categorical probabilities.

    Conditional continuous values use the midpoint of each preregistered interval.  Every quantile
    receives the same midpoint, which is monotone and deliberately uninformative.
    """

    task = ExecutionTask.model_validate_json(canonical_json_bytes(task))
    context = task.context
    return ExecutionSubmission(
        episode_id=context.episode_id,
        target_trial_id=context.target_trial_id,
        task_context_sha256=task.context_sha256,
        registry_outcome_probabilities=RegistryOutcomeProbabilities(
            completed=1.0 / 7.0,
            terminated=1.0 / 7.0,
            withdrawn=1.0 / 7.0,
            suspended=1.0 / 7.0,
            non_terminal=1.0 / 7.0,
            status_missing=1.0 / 7.0,
            record_missing=1.0 / 7.0,
        ),
        enrollment_observation_probabilities=ObservationStateProbabilities(
            observed_actual=0.25,
            not_actual=0.25,
            value_missing=0.25,
            record_missing=0.25,
        ),
        primary_completion_observation_probabilities=ObservationStateProbabilities(
            observed_actual=0.25,
            not_actual=0.25,
            value_missing=0.25,
            record_missing=0.25,
        ),
        enrollment_ratio_given_observed_actual=_midpoint_forecast(context.enrollment_ratio_spec),
        primary_completion_slippage_days_given_observed_actual=_midpoint_forecast(
            context.primary_completion_slippage_days_spec
        ),
    )


__all__ = ['uniform_execution_submission']
