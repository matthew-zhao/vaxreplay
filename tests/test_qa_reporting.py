from __future__ import annotations

from datetime import datetime, timezone

from vaxreplay.case_schema import Submission
from vaxreplay.qa.attack_catalog import attack_catalog_sha256, default_attack_catalog
from vaxreplay.qa.metamorphic import audit_nuisance_invariance
from vaxreplay.qa.reporting import finding_from_metamorphic, make_reward_qa_report
from vaxreplay.qa.schema import ComponentFloorSpec, QADisposition, RewardContract


def _contract() -> RewardContract:
    return RewardContract(
        contract_id='report-contract',
        reward_version='v0.1',
        scientific_objective='Grounded prioritization.',
        scorer_sha256='1' * 64,
        reference_scorer_sha256='2' * 64,
        prompt_compiler_sha256='3' * 64,
        qa_policy_sha256='4' * 64,
        attack_catalog_sha256=attack_catalog_sha256(default_attack_catalog()),
        required_dimensions=('immunogenicity',),
        required_attack_ids=('candidate.alias',),
        component_floors=(ComponentFloorSpec(metric='grounding_f1', minimum=0.5),),
        prohibited_strategies=('outcome lookup',),
    )


def _submission() -> Submission:
    return Submission(
        episode_id='episode',
        manifest_sha256='0' * 64,
        ranking=['candidate'],
        forecasts=[
            {
                'candidate_id': 'candidate',
                'target_id': 'target',
                'horizon_days': 1,
                'probability': 0.5,
            }
        ],
    )


def _report(*, grounding: float, variant: Submission | None = None):
    reference = _submission()
    comparisons = audit_nuisance_invariance(reference, variant or reference)
    comparison = next((item for item in comparisons if not item.passed), comparisons[0])
    finding = finding_from_metamorphic(comparison, check_id='candidate.alias')
    return make_reward_qa_report(
        report_id='report',
        run_id='run',
        created_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        trajectory_batch_sha256='5' * 64,
        reward_artifact_sha256='6' * 64,
        episode_manifest_sha256s=('7' * 64,),
        reward_contract=_contract(),
        component_scores={'grounding_f1': grounding},
        findings=(finding,),
        all_required_attacks_executed=True,
        independent_scorer_agreement=True,
        future_taint_reachability_zero=True,
        exact_replay=True,
        tamper_success_zero=True,
        item_level_private_feedback_withheld=True,
    )


def test_builder_derives_admit_for_passing_vector() -> None:
    assert _report(grounding=1.0).disposition == QADisposition.ADMIT


def test_builder_derives_quarantine_for_component_floor_or_failed_metamorphic_check() -> None:
    assert _report(grounding=0.0).disposition == QADisposition.QUARANTINE
    submission = _submission()
    changed = submission.model_copy(
        update={
            'forecasts': [
                submission.forecasts[0].model_copy(update={'probability': 0.9}),
            ]
        }
    )
    assert _report(grounding=1.0, variant=changed).disposition == QADisposition.QUARANTINE
