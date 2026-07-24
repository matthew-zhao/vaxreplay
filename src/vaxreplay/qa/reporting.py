"""Deterministic construction of vector QA reports from independent checks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import datetime

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.qa.metamorphic import MetamorphicFinding
from vaxreplay.qa.schema import (
    ComponentScore,
    QACategory,
    QADisposition,
    QASeverity,
    RewardContract,
    RewardQAFinding,
    RewardQAReport,
    reward_contract_sha256,
)


def finding_from_metamorphic(
    finding: MetamorphicFinding,
    *,
    check_id: str,
    severity: QASeverity = QASeverity.WARNING,
    disposition_on_failure: QADisposition = QADisposition.QUARANTINE,
    hidden_control: bool = True,
) -> RewardQAFinding:
    if disposition_on_failure == QADisposition.ADMIT:
        raise ValueError('failed QA findings cannot have an admit disposition')
    artifact = finding.as_dict()
    return RewardQAFinding(
        finding_id=f'metamorphic-{finding.finding_id}',
        check_id=check_id,
        category=QACategory.COUNTERFACTUAL_CONSISTENCY,
        severity=severity,
        passed=finding.passed,
        disposition_on_failure=disposition_on_failure,
        detail=f'{finding.subject}: expected {finding.expected}; observed {finding.observed}',
        artifact_sha256=hashlib.sha256(canonical_json_bytes(artifact)).hexdigest(),
        hidden_control=hidden_control,
    )


def make_reward_qa_report(
    *,
    report_id: str,
    run_id: str,
    created_at: datetime,
    trajectory_batch_sha256: str,
    reward_artifact_sha256: str,
    episode_manifest_sha256s: Iterable[str],
    reward_contract: RewardContract,
    component_scores: Mapping[str, float],
    findings: Iterable[RewardQAFinding],
    all_required_attacks_executed: bool,
    independent_scorer_agreement: bool,
    future_taint_reachability_zero: bool,
    exact_replay: bool,
    tamper_success_zero: bool,
    item_level_private_feedback_withheld: bool,
) -> RewardQAReport:
    ordered_components = tuple(
        ComponentScore(metric=metric, value=value) for metric, value in sorted(component_scores.items())
    )
    ordered_findings = tuple(sorted(findings, key=lambda finding: finding.finding_id))
    disposition = _derive_disposition(
        reward_contract,
        ordered_components,
        ordered_findings,
        hard_flags=(
            all_required_attacks_executed,
            independent_scorer_agreement,
            future_taint_reachability_zero,
            exact_replay,
            tamper_success_zero,
            item_level_private_feedback_withheld,
        ),
    )
    return RewardQAReport(
        report_id=report_id,
        run_id=run_id,
        created_at=created_at,
        trajectory_batch_sha256=trajectory_batch_sha256,
        reward_artifact_sha256=reward_artifact_sha256,
        episode_manifest_sha256s=tuple(sorted(episode_manifest_sha256s)),
        reward_contract=reward_contract,
        reward_contract_sha256=reward_contract_sha256(reward_contract),
        component_scores=ordered_components,
        findings=ordered_findings,
        disposition=disposition,
        all_required_attacks_executed=all_required_attacks_executed,
        independent_scorer_agreement=independent_scorer_agreement,
        future_taint_reachability_zero=future_taint_reachability_zero,
        exact_replay=exact_replay,
        tamper_success_zero=tamper_success_zero,
        item_level_private_feedback_withheld=item_level_private_feedback_withheld,
    )


def _derive_disposition(
    contract: RewardContract,
    component_scores: tuple[ComponentScore, ...],
    findings: tuple[RewardQAFinding, ...],
    *,
    hard_flags: tuple[bool, ...],
) -> QADisposition:
    failed = tuple(finding for finding in findings if not finding.passed)
    if not all(hard_flags) or any(finding.disposition_on_failure == QADisposition.REJECT for finding in failed):
        return QADisposition.REJECT
    values = {item.metric: item.value for item in component_scores}
    if any(values.get(floor.metric, -1.0) < floor.minimum for floor in contract.component_floors):
        return QADisposition.QUARANTINE
    if failed:
        return QADisposition.QUARANTINE
    return QADisposition.ADMIT
