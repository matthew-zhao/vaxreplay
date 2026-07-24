"""Deterministic public-only smoke run for a complete clinical execution workspace.

This is not a model evaluation.  It proves that every public task can produce a valid submission
and that the exact private evaluator can score the complete frozen cohort without dropping cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split
from vaxreplay.clinicaltrials.execution_aggregation import (
    ExecutionCohortEvaluationCase,
    ExecutionCohortEvaluator,
    ExecutionCohortManifest,
    ExecutionCohortResult,
    ExecutionCohortSubmission,
    make_execution_cohort_manifest,
    make_execution_cohort_submission,
)
from vaxreplay.clinicaltrials.execution_baselines import uniform_execution_submission
from vaxreplay.clinicaltrials.execution_task import ExecutionPrivateGold, ExecutionTask
from vaxreplay.clinicaltrials.execution_workspace import (
    ExecutionWorkspaceContextPlan,
    ExecutionWorkspaceError,
    verify_execution_workspace_build,
)


@dataclass(frozen=True, slots=True)
class UniformExecutionReferenceRun:
    manifest: ExecutionCohortManifest
    submissions: ExecutionCohortSubmission
    result: ExecutionCohortResult


def run_uniform_execution_reference(
    *,
    workspace_root: Path,
    expected_workspace_receipt_sha256: str,
    lineage_split_manifest_sha256: str,
    gold_derivation_receipt_sha256: str,
    cohort_id: str,
    evaluation_split: Split,
) -> UniformExecutionReferenceRun:
    """Run the label-blind uniform baseline over every task in one verified split."""

    build = verify_execution_workspace_build(
        workspace_root,
        expected_receipt_sha256=expected_workspace_receipt_sha256,
    )
    try:
        plan = ExecutionWorkspaceContextPlan.model_validate_json(
            (build.root / 'organizer' / 'context-plan.json').read_bytes()
        )
    except ValueError as error:
        raise ExecutionWorkspaceError(f'invalid organizer context plan: {error}') from error
    if canonical_json_bytes(plan) != (build.root / 'organizer' / 'context-plan.json').read_bytes():
        raise ExecutionWorkspaceError('organizer context plan must use canonical JSON')
    if plan.split_manifest_sha256 != lineage_split_manifest_sha256:
        raise ExecutionWorkspaceError('reference run split manifest does not match the workspace plan')

    tasks_by_episode: dict[str, ExecutionTask] = {task.context.episode_id: task for task in build.tasks}
    gold_by_episode: dict[str, ExecutionPrivateGold] = {gold.episode_id: gold for gold in build.gold}
    entries_by_episode = {entry.context.episode_id: entry for entry in plan.entries}
    expected_episodes = set(entries_by_episode)
    if set(tasks_by_episode) != expected_episodes or set(gold_by_episode) != expected_episodes:
        raise ExecutionWorkspaceError('workspace tasks, gold, and context plan do not have exact coverage')
    selected_episodes = tuple(
        sorted(episode_id for episode_id, entry in entries_by_episode.items() if entry.split == evaluation_split)
    )
    if not selected_episodes:
        raise ExecutionWorkspaceError(f'workspace contains no tasks in the requested {evaluation_split.value} split')

    cases: list[ExecutionCohortEvaluationCase] = []
    public_only_submissions = []
    for episode_id in selected_episodes:
        task = tasks_by_episode[episode_id]
        entry = entries_by_episode[episode_id]
        key_path = build.root / 'private' / 'tasks' / episode_id / 'gold.key'
        key = key_path.read_bytes()
        cases.append(
            ExecutionCohortEvaluationCase(
                task=task,
                private_gold=gold_by_episode[episode_id],
                private_gold_key=key,
                split=entry.split,
                public_lineage_id=entry.public_lineage_id,
            )
        )
        # This function is deliberately called with the public task only.  Gold and organizer
        # state are used exclusively after the submission has been materialized.
        public_only_submissions.append(uniform_execution_submission(task))

    manifest = make_execution_cohort_manifest(
        cohort_id=cohort_id,
        cases=cases,
        lineage_split_manifest_sha256=lineage_split_manifest_sha256,
        workspace_build_receipt_sha256=expected_workspace_receipt_sha256,
        gold_derivation_receipt_sha256=gold_derivation_receipt_sha256,
    )
    submissions = make_execution_cohort_submission(
        manifest=manifest,
        submissions=public_only_submissions,
    )
    result = ExecutionCohortEvaluator(manifest=manifest, cases=cases).score(submissions)
    return UniformExecutionReferenceRun(manifest=manifest, submissions=submissions, result=result)


__all__ = ['UniformExecutionReferenceRun', 'run_uniform_execution_reference']
