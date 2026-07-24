from __future__ import annotations

import tempfile
from pathlib import Path

from tests.test_clinicaltrials_execution_gold_adapter import _fixture
from vaxreplay.clinicaltrials.execution_baselines import uniform_execution_submission
from vaxreplay.clinicaltrials.execution_scoring import validate_execution_submission
from vaxreplay.clinicaltrials.execution_task import (
    ExecutionPrivateGold,
    build_execution_task,
    execution_task_context_sha256,
)


def test_uniform_baseline_is_public_only_and_valid() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        merged, _archives, _queue, _adjudications, _review_receipt, _trusted, context = fixture
        assignment = next(item for item in merged.inventory.assignments if item.nct_id == 'NCT00000001')
        label = next(item for item in merged.labels.labels if item.nct_id == assignment.nct_id)
        decision = next(
            item
            for item in merged.inventory.decision_rows
            if item.nct_id == assignment.nct_id and item.archive_date == assignment.anchor_date
        )
        gold = ExecutionPrivateGold(
            episode_id=context.episode_id,
            target_trial_id=context.target_trial_id,
            organizer_private_nct_id=assignment.nct_id,
            organizer_private_decision_record_sha256=decision.source_record_sha256,
            task_context_sha256=execution_task_context_sha256(context),
            registry_outcome_class=label.registry_outcome_class,
            enrollment_observation=label.enrollment_observation,
            enrollment_ratio=label.enrollment_ratio,
            primary_completion_observation=label.primary_completion_observation,
            primary_completion_slippage_days=label.primary_completion_slippage_days,
        )
        task = build_execution_task(context=context, gold=gold, private_gold_key=b'k' * 32)

    first = uniform_execution_submission(task)
    second = uniform_execution_submission(task)
    assert first == second
    assert not validate_execution_submission(task, first)
    assert first.episode_id == task.context.episode_id
    assert 'organizer_private_nct_id' not in first.model_dump()
    submission = uniform_execution_submission(task)
    assert submission.enrollment_ratio_given_observed_actual.kind == context.enrollment_ratio_spec.forecast_kind
    assert (
        submission.primary_completion_slippage_days_given_observed_actual.kind
        == context.primary_completion_slippage_days_spec.forecast_kind
    )
