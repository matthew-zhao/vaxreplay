from __future__ import annotations

import hashlib

import pytest

from vaxreplay.clinicaltrials.execution_experiments import (
    LaneAExperimentBudget,
    LaneAExperimentError,
    LaneAExperimentManifest,
    LaneAExperimentObservation,
    LaneAExperimentRunMode,
    LaneAExperimentTerminalStatus,
    LaneAExperimentUsage,
    LaneAExternalCliSystem,
    LaneAExternalCompatibilityResult,
    LaneAHarnessKind,
    LaneAHarnessSpec,
    LaneAModelRoute,
    OfflineFakeLaneABackend,
    attach_external_compatibility_results,
    make_lane_a_experiment_manifest,
    run_lane_a_experiment,
    verify_lane_a_experiment_report,
)
from vaxreplay.clinicaltrials.execution_process_reward import (
    MAXIMUM_CREDITABLE_PROCESS_REWARD,
    lane_a_process_reward_policy_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _manifest() -> LaneAExperimentManifest:
    harnesses = tuple(
        LaneAHarnessSpec(
            kind=kind,
            harness_id=f'lane-a-{kind.value}',
            harness_version='dev-v0.1',
            executable_sha256=_sha(f'harness:{kind.value}'),
            expected_model_calls={
                LaneAHarnessKind.UNIFORM: 0,
                LaneAHarnessKind.SINGLE_CALL: 1,
                LaneAHarnessKind.RETRIEVAL_AGENT: 2,
                LaneAHarnessKind.COMPUTE_AGENT: 2,
            }[kind],
            route_independent_uniform_baseline=kind == LaneAHarnessKind.UNIFORM,
        )
        for kind in LaneAHarnessKind
    )
    routes = (
        LaneAModelRoute(
            route_id='route-a',
            provider='fake-provider-a',
            requested_model_id='logical-model-a',
            expected_resolved_model_id='fake-model-a-2026-07-14',
            route_config_sha256=_sha('route-a'),
        ),
        LaneAModelRoute(
            route_id='route-b',
            provider='fake-provider-b',
            requested_model_id='logical-model-b',
            expected_resolved_model_id='fake-model-b-2026-07-14',
            route_config_sha256=_sha('route-b'),
        ),
    )
    external = (
        LaneAExternalCliSystem(
            system_id='claude-code-sonnet',
            cli_name='Claude Code',
            cli_version='2.1.195',
            executable_sha256=_sha('claude-code-cli'),
            requested_model_id='sonnet',
            resolved_model_id='claude-sonnet-4-6',
        ),
        LaneAExternalCliSystem(
            system_id='codex-cli-sol',
            cli_name='Codex CLI',
            cli_version='0.144.3',
            executable_sha256=_sha('codex-cli'),
            requested_model_id='gpt-5.6-sol',
        ),
    )
    return make_lane_a_experiment_manifest(
        experiment_id='fictional-lane-a-factorial-v0-1',
        task_ids=('fictional-task-002', 'fictional-task-001'),
        task_set_sha256=_sha('fictional-task-set'),
        harnesses=harnesses,
        model_routes=routes,
        budget=LaneAExperimentBudget(
            budget_id='paired-dev-budget-v0-1',
            max_model_calls=4,
            max_input_tokens=10_000,
            max_output_tokens=2_000,
            max_reasoning_tokens=4_000,
            wall_seconds=120,
            maximum_provider_cost_usd=1.0,
        ),
        external_cli_systems=external,
    )


def test_offline_complete_matrix_is_paired_and_retains_terminal_failures() -> None:
    manifest = _manifest()
    failure = ('retrieval_agent--route-a', 'fictional-task-001')
    report = run_lane_a_experiment(
        manifest,
        backend=OfflineFakeLaneABackend(fail_attempts=frozenset((failure,))),
    )

    assert report.run_mode == LaneAExperimentRunMode.OFFLINE_FAKE
    assert report.expected_attempt_count == report.observed_attempt_count == 16
    assert len(report.attempts) == 16
    assert sum(item.terminal_status.value != 'completed' for item in report.attempts) == 1
    assert all(cell.task_order == manifest.task_ids for cell in manifest.cells)
    assert len({cell.budget_sha256 for cell in manifest.cells}) == 1
    assert not report.valid_only_means_emitted
    assert not report.combined_scalar_emitted
    assert report.process_reward_auxiliary_only
    assert report.outcome_reward_remains_primary
    assert report.process_reward_policy_sha256 == lane_a_process_reward_policy_sha256()
    assert manifest.process_reward_maximum_creditable == pytest.approx(MAXIMUM_CREDITABLE_PROCESS_REWARD)
    assert not report.real_aact_task_transmitted
    assert all(item.combined_reward is None for item in report.attempts)
    assert all(item.outcome_reward is None for item in report.attempts)
    uniform = [item for item in report.attempts if item.harness_kind == LaneAHarnessKind.UNIFORM]
    assert uniform and all(item.usage.model_calls == 0 for item in uniform)
    failed_cell = next(item for item in report.cell_summaries if item.cell_id == failure[0])
    assert failed_cell.terminal_failure_count == 1
    assert failed_cell.task_count == 2
    assert failed_cell.mean_process_reward_all_tasks == pytest.approx((0.60 + 0.30) / 2)


def test_manifest_rejects_cell_or_task_cherry_picking() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match='complete fixed'):
        LaneAExperimentManifest.model_validate(manifest.model_copy(update={'cells': manifest.cells[:-1]}))
    first = manifest.cells[0].model_copy(update={'task_order': manifest.task_ids[:-1]})
    with pytest.raises(ValueError, match='same pinned'):
        LaneAExperimentManifest.model_validate(manifest.model_copy(update={'cells': (first, *manifest.cells[1:])}))


def test_report_verifier_rejects_same_count_task_substitution() -> None:
    manifest = _manifest()
    report = run_lane_a_experiment(manifest, backend=OfflineFakeLaneABackend())
    replaced = report.attempts[0].model_copy(update={'task_id': report.attempts[1].task_id})
    forged = report.model_copy(update={'attempts': (replaced, *report.attempts[1:])})

    with pytest.raises(LaneAExperimentError, match='binding differs'):
        verify_lane_a_experiment_report(forged, manifest=manifest)


def test_report_verifier_rejects_wrong_resolved_model_and_forged_usage() -> None:
    manifest = _manifest()
    report = run_lane_a_experiment(manifest, backend=OfflineFakeLaneABackend())
    wrong_model = report.attempts[4].model_copy(update={'resolved_model_id': 'different-model'})
    forged_model = report.model_copy(update={'attempts': (*report.attempts[:4], wrong_model, *report.attempts[5:])})

    with pytest.raises(LaneAExperimentError, match='resolved model identity'):
        verify_lane_a_experiment_report(forged_model, manifest=manifest)

    original = report.attempts[4]
    forged_calls = manifest.budget.max_model_calls + 95
    forged_usage = original.usage.model_copy(update={'model_calls': forged_calls})
    forged_attempt = original.model_copy(update={'usage': forged_usage})
    summary_index = next(
        index for index, summary in enumerate(report.cell_summaries) if summary.cell_id == original.cell_id
    )
    original_summary = report.cell_summaries[summary_index]
    forged_summary = original_summary.model_copy(
        update={'total_model_calls': original_summary.total_model_calls - original.usage.model_calls + forged_calls}
    )
    forged_summaries = list(report.cell_summaries)
    forged_summaries[summary_index] = forged_summary
    forged_report = report.model_copy(
        update={
            'attempts': (*report.attempts[:4], forged_attempt, *report.attempts[5:]),
            'cell_summaries': tuple(forged_summaries),
        }
    )

    with pytest.raises(LaneAExperimentError, match='budget status'):
        verify_lane_a_experiment_report(forged_report, manifest=manifest)


class _OverBudgetBackend:
    run_mode = LaneAExperimentRunMode.OFFLINE_FAKE

    def run(self, *, manifest, cell, task_id) -> LaneAExperimentObservation:
        route = next(item for item in manifest.model_routes if item.route_id == cell.route_id)
        return LaneAExperimentObservation(
            terminal_status=LaneAExperimentTerminalStatus.COMPLETED,
            resolved_model_id=route.expected_resolved_model_id,
            usage=LaneAExperimentUsage(
                model_calls=manifest.budget.max_model_calls + 1,
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                provider_cost_usd=0.0,
                wall_milliseconds=0,
                provider_metering_authoritative=False,
            ),
            process_reward=0.1,
        )


def test_budget_overrun_becomes_retained_terminal_failure() -> None:
    report = run_lane_a_experiment(_manifest(), backend=_OverBudgetBackend())

    assert len(report.attempts) == report.expected_attempt_count
    assert all(item.terminal_status == LaneAExperimentTerminalStatus.FAILED for item in report.attempts)
    assert all(item.failure_code == 'budget_exceeded' for item in report.attempts)
    assert all(item.usage.model_calls == 5 for item in report.attempts)
    assert all(item.terminal_failure_count == item.task_count for item in report.cell_summaries)


def test_external_cli_diagnostics_stay_separate_and_noncomparative() -> None:
    manifest = _manifest()
    report = run_lane_a_experiment(manifest, backend=OfflineFakeLaneABackend())
    diagnostics = (
        LaneAExternalCompatibilityResult(
            system_id='claude-code-sonnet',
            status='live_synthetic_pass',
            requested_model_id='sonnet',
            resolved_model_id='claude-sonnet-4-6',
            receipt_sha256='00dffbd7963cbb966bff3d66f2f0596d199c5f46ed54a5ad708ae2044935925f',
            diagnostic_reward=0.7883,
            grounding_reward=0.0,
            assessment_accuracy=1.0,
        ),
        LaneAExternalCompatibilityResult(
            system_id='codex-cli-sol',
            status='live_synthetic_pass',
            requested_model_id='gpt-5.6-sol',
            resolved_model_id='unattested',
            receipt_sha256='877e45cbb2865e9667dee9b6c21a7fceb088c9108561ec64a0c02dcefe16173d',
            diagnostic_reward=0.795,
            grounding_reward=0.0,
            assessment_accuracy=1.0,
        ),
    )
    updated = attach_external_compatibility_results(report, manifest=manifest, results=diagnostics)

    assert tuple(item.system_id for item in updated.external_compatibility) == (
        'claude-code-sonnet',
        'codex-cli-sol',
    )
    assert all(item.one_shot_diagnostic for item in updated.external_compatibility)
    assert all(not item.comparative_inference_allowed for item in updated.external_compatibility)
    assert all(not item.provider_route_attested for item in updated.external_compatibility)
    assert all(not item.real_aact_task_transmitted for item in updated.external_compatibility)
    # Attaching diagnostics cannot alter standardized paired attempts.
    assert updated.attempts == report.attempts


def test_external_results_require_exact_declared_system_coverage() -> None:
    manifest = _manifest()
    report = run_lane_a_experiment(manifest, backend=OfflineFakeLaneABackend())
    with pytest.raises(LaneAExperimentError, match='exactly cover'):
        attach_external_compatibility_results(report, manifest=manifest, results=())


def test_failed_external_diagnostic_retains_receipt_without_invented_metrics() -> None:
    result = LaneAExternalCompatibilityResult(
        system_id='cursor-agent-gemini',
        status='live_synthetic_fail',
        requested_model_id='gemini-3.1-pro',
        receipt_sha256='3' * 64,
    )

    assert result.receipt_sha256 == '3' * 64
    assert result.diagnostic_reward is None
    with pytest.raises(ValueError, match='cannot invent score metrics'):
        result.model_copy(update={'diagnostic_reward': 0.0}, deep=True).__class__.model_validate(
            {**result.model_dump(mode='json'), 'diagnostic_reward': 0.0}
        )
