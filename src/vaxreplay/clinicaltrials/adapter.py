"""Two-snapshot AACT adapter for early-clinical arm-prioritization replay."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from vaxreplay.bundle import (
    EpisodeBundle,
    body_sha256,
    canonical_json_bytes,
    jsonl_text,
    ranking_labels_commitment,
    records_sha256,
)
from vaxreplay.case_schema import (
    AdapterProvenance,
    AssessmentConclusion,
    CandidateRecord,
    EpisodeManifest,
    EvidenceRecord,
    EvidenceStance,
    ForecastTarget,
    GoldAssessmentRecord,
    GoldEvidenceRecord,
    LabelCommitmentScheme,
    OutcomeRecord,
    PrivateLabels,
    SourceSnapshotCommitment,
    SourceType,
)
from vaxreplay.clinicaltrials.extract import (
    DesignGroupInterventionRow,
    DesignGroupRow,
    DesignOutcomeRow,
    InterventionRow,
    LoadedAactSlice,
    OutcomeMeasurementRow,
    OutcomeRow,
    ResultGroupRow,
    StudyRow,
    load_slice,
)
from vaxreplay.clinicaltrials.schema import (
    AACT_ADAPTER_ID,
    EARLY_CLINICAL_TASK,
    ENDPOINT_ALIGNMENT_DIMENSION,
    REGIMEN_DEFINITION_DIMENSION,
    AactPrivateAudit,
    AactSliceReceipt,
    AactSourceTable,
    ArmRole,
    CandidateOutcomeAudit,
    EarlyClinicalEpisodeSpec,
    EndpointValueAudit,
    ResolvedArmMapping,
    ResolvedEndpointMapping,
    normalize_regimen_title,
)
from vaxreplay.ranking_schema import RankingLabelV1

_DECISION_REQUIRED_TABLES = {
    AactSourceTable.STUDIES,
    AactSourceTable.DESIGNS,
    AactSourceTable.DESIGN_GROUPS,
    AactSourceTable.DESIGN_GROUP_INTERVENTIONS,
    AactSourceTable.INTERVENTIONS,
    AactSourceTable.DESIGN_OUTCOMES,
}
_LABEL_REQUIRED_TABLES = {
    AactSourceTable.STUDIES,
    AactSourceTable.RESULT_GROUPS,
    AactSourceTable.OUTCOMES,
    AactSourceTable.OUTCOME_MEASUREMENTS,
}


class AactAdapterError(ValueError):
    """Raised when exact AACT slices cannot safely produce a replay episode."""


@dataclass(frozen=True)
class _PanelResolution:
    resolved: tuple[ResolvedArmMapping, ...]
    decision_group_by_regimen: dict[str, DesignGroupRow]
    result_group_by_regimen: dict[str, ResultGroupRow]
    decision_group_count: int
    excluded_decision_group_count: int
    target_outcome_group_count: int
    excluded_target_result_group_count: int
    non_target_outcome_result_group_count: int


def build_episode(
    *,
    spec: EarlyClinicalEpisodeSpec,
    decision_snapshot_root: Path,
    decision_receipt: AactSliceReceipt,
    label_snapshot_root: Path,
    label_receipt: AactSliceReceipt,
    output_root: Path,
    label_commitment_key: bytes | None = None,
) -> EpisodeBundle:
    """Build a sealed V1 episode from one pre-results and one post-results AACT slice."""

    key = secrets.token_bytes(32) if label_commitment_key is None else label_commitment_key
    if len(key) < 32:
        raise AactAdapterError('label commitment key must contain at least 32 bytes')
    decision = load_slice(decision_snapshot_root, decision_receipt)
    label = load_slice(label_snapshot_root, label_receipt)
    _validate_receipts(spec, decision, label)
    decision_study, label_study, results_first_posted = _validate_chronology(spec, decision, label)

    design_groups = decision.rows(AactSourceTable.DESIGN_GROUPS, DesignGroupRow)
    group_interventions = decision.rows(AactSourceTable.DESIGN_GROUP_INTERVENTIONS, DesignGroupInterventionRow)
    interventions = decision.rows(AactSourceTable.INTERVENTIONS, InterventionRow)
    design_outcomes = decision.rows(AactSourceTable.DESIGN_OUTCOMES, DesignOutcomeRow)
    result_groups = label.rows(AactSourceTable.RESULT_GROUPS, ResultGroupRow)
    outcomes = label.rows(AactSourceTable.OUTCOMES, OutcomeRow)
    measurements = label.rows(AactSourceTable.OUTCOME_MEASUREMENTS, OutcomeMeasurementRow)

    endpoint_audits, measurements_by_endpoint = _validate_endpoint_crosswalk(
        spec, design_outcomes, outcomes, measurements, result_groups
    )
    panel = _resolve_complete_panel(
        spec,
        design_groups,
        result_groups,
        measurements_by_endpoint,
    )
    _validate_intervention_links(panel, group_interventions, interventions)

    candidates = tuple(
        CandidateRecord(episode_id=spec.episode_id, candidate_id=candidate_id)
        for candidate_id in sorted(
            mapping.candidate_id
            for mapping in spec.arm_mappings
            if mapping.role == ArmRole.CANDIDATE and mapping.candidate_id is not None
        )
    )
    evidence, assessments_gold, evidence_gold = _build_protocol_evidence(
        spec,
        key,
        provenance_url=decision.receipt.archive.source_url,
        license_id=decision.receipt.archive.license_id,
    )
    private_outcomes, ranking_labels, outcome_audits = _derive_outcomes(
        spec,
        panel,
        measurements_by_endpoint,
    )
    if len({label.relevance_grade for label in ranking_labels}) < 2:
        raise AactAdapterError('the fixed rubric produced no strict ranking distinction in the selected panel')

    private_labels = PrivateLabels(
        outcomes=list(private_outcomes),
        assessments_gold=list(assessments_gold),
        evidence_gold=list(evidence_gold),
    )
    private_audit = AactPrivateAudit(
        episode_id=spec.episode_id,
        nct_id=spec.nct_id,
        decision_slice_sha256=decision.receipt_sha256,
        label_slice_sha256=label.receipt_sha256,
        results_first_posted_date=results_first_posted,
        outcome_snapshot_at=spec.outcome_as_of,
        decision_design_group_count=panel.decision_group_count,
        selected_decision_group_count=len(panel.resolved),
        excluded_decision_group_count=panel.excluded_decision_group_count,
        target_outcome_group_count=panel.target_outcome_group_count,
        selected_result_group_count=len(panel.resolved),
        excluded_target_result_group_count=panel.excluded_target_result_group_count,
        non_target_outcome_result_group_count=panel.non_target_outcome_result_group_count,
        arm_mappings=panel.resolved,
        endpoint_mappings=endpoint_audits,
        outcomes=outcome_audits,
    )

    spec_commitment = hmac.new(key, canonical_json_bytes(spec), hashlib.sha256).hexdigest()
    audit_commitment = hmac.new(key, canonical_json_bytes(private_audit), hashlib.sha256).hexdigest()
    source_provenance = AdapterProvenance(
        adapter_id=AACT_ADAPTER_ID,
        episode_spec_commitment=spec_commitment,
        decision_snapshot_id=decision.receipt.archive.snapshot_id,
        label_snapshot_id=label.receipt.archive.snapshot_id,
        snapshot_commitments=[
            _source_commitment(decision),
            _source_commitment(label),
        ],
        private_audit_commitment=audit_commitment,
    )
    labels_commitment = ranking_labels_commitment(
        private_labels,
        ranking_labels,
        LabelCommitmentScheme.HMAC_SHA256,
        key=key,
    )
    horizon_days = int((spec.outcome_as_of - spec.decision_at).total_seconds() // 86_400)
    manifest = EpisodeManifest(
        episode_id=spec.episode_id,
        lineage_group_id=spec.lineage_group_id,
        synthetic=False,
        task_type=EARLY_CLINICAL_TASK,
        split=spec.split,
        decision_at=spec.decision_at,
        portfolio_size=spec.portfolio_size,
        candidate_ids=[candidate.candidate_id for candidate in candidates],
        forecast_targets=[ForecastTarget(target_id=spec.rubric.target_id, horizon_days=horizon_days)],
        required_dimensions=list(spec.required_dimensions),
        evidence_sha256=records_sha256(evidence),
        candidates_sha256=records_sha256(candidates),
        labels_sha256=labels_commitment,
        label_commitment_scheme=LabelCommitmentScheme.HMAC_SHA256,
        label_commitment_key_id=hashlib.sha256(key).hexdigest(),
        adjudication_version=(
            f'{AACT_ADAPTER_ID}:{spec.rubric.version}:{spec.adjudication_version}:{spec_commitment[:12]}'
        ),
        source_provenance=source_provenance,
        reward_version=spec.reward_version,
    )
    _assert_public_identity_masked(spec, manifest, candidates, evidence)

    output_root = output_root.expanduser().resolve()
    _write_episode_atomically(
        output_root=output_root,
        manifest=manifest,
        candidates=candidates,
        evidence=evidence,
        private_labels=private_labels,
        ranking_labels=ranking_labels,
        private_audit=private_audit,
        spec=spec,
        decision_receipt=decision.receipt,
        label_receipt=label.receipt,
        label_commitment_key=key,
    )
    # Keep these variables deliberately live through construction: chronology was checked from the
    # exact study rows, and neither title nor sponsor is copied into public evidence.
    _ = (decision_study, label_study)
    return EpisodeBundle.load(output_root, include_private=True)


def audit_episode(root: Path) -> dict[str, Any]:
    """Recheck private commitments and the deterministic composite-to-label mapping."""

    bundle = EpisodeBundle.load(root.expanduser().resolve(), include_private=True)
    provenance = bundle.manifest.source_provenance
    if provenance is None or provenance.adapter_id != AACT_ADAPTER_ID:
        raise AactAdapterError('episode is not bound to the AACT early-clinical adapter')
    key = bundle.label_commitment_key
    if key is None:
        raise AactAdapterError('AACT private audit requires the label HMAC key')
    spec = _read_private_model(bundle.root / 'private' / 'aact_episode_spec.json', EarlyClinicalEpisodeSpec)
    audit = _read_private_model(bundle.root / 'private' / 'aact_audit.json', AactPrivateAudit)
    decision_receipt = _read_private_model(
        bundle.root / 'private' / 'aact_decision_slice_receipt.json', AactSliceReceipt
    )
    label_receipt = _read_private_model(bundle.root / 'private' / 'aact_label_slice_receipt.json', AactSliceReceipt)
    spec_commitment = hmac.new(key, canonical_json_bytes(spec), hashlib.sha256).hexdigest()
    audit_commitment = hmac.new(key, canonical_json_bytes(audit), hashlib.sha256).hexdigest()
    if spec_commitment != provenance.episode_spec_commitment:
        raise AactAdapterError('private AACT spec does not match its manifest commitment')
    if audit_commitment != provenance.private_audit_commitment:
        raise AactAdapterError('private AACT audit does not match its manifest commitment')
    decision_sha = hashlib.sha256(canonical_json_bytes(decision_receipt)).hexdigest()
    label_sha = hashlib.sha256(canonical_json_bytes(label_receipt)).hexdigest()
    if (decision_sha, label_sha) != (audit.decision_slice_sha256, audit.label_slice_sha256):
        raise AactAdapterError('private AACT receipts do not match the audit')
    commitment_by_snapshot = {item.snapshot_id: item.manifest_sha256 for item in provenance.snapshot_commitments}
    if commitment_by_snapshot != {
        decision_receipt.archive.snapshot_id: decision_sha,
        label_receipt.archive.snapshot_id: label_sha,
    }:
        raise AactAdapterError('private AACT receipts do not match source provenance')
    if (
        spec.episode_id != bundle.manifest.episode_id
        or spec.lineage_group_id != bundle.manifest.lineage_group_id
        or spec.decision_at != bundle.manifest.decision_at
        or spec.portfolio_size != bundle.manifest.portfolio_size
        or spec.reward_version != bundle.manifest.reward_version
        or bundle.manifest.task_type != EARLY_CLINICAL_TASK
    ):
        raise AactAdapterError('private AACT spec does not reconstruct the episode manifest')
    if audit.episode_id != spec.episode_id or audit.nct_id != spec.nct_id:
        raise AactAdapterError('private AACT audit identity mismatch')

    labels = bundle.private_labels
    ranking_labels = bundle.ranking_labels
    assert labels is not None and ranking_labels is not None
    private_outcomes = {
        outcome.candidate_id: (outcome.outcome, outcome.candidate_utility) for outcome in labels.outcomes
    }
    audit_outcomes = {
        outcome.candidate_id: (outcome.binary_outcome, outcome.candidate_utility) for outcome in audit.outcomes
    }
    if private_outcomes != audit_outcomes:
        raise AactAdapterError('private outcome labels do not match the AACT audit')
    grades = {label.candidate_id: label.relevance_grade for label in ranking_labels}
    audit_grades = {outcome.candidate_id: outcome.relevance_grade for outcome in audit.outcomes}
    if grades != audit_grades:
        raise AactAdapterError('private ranking labels do not match the fixed AACT rubric')
    return {
        'episode_id': bundle.manifest.episode_id,
        'manifest_sha256': bundle.manifest_sha256,
        'episode_spec_commitment': spec_commitment,
        'private_audit_commitment': audit_commitment,
        'candidate_count': len(bundle.manifest.candidate_ids),
        'selected_panel_count': audit.selected_decision_group_count,
        'excluded_protocol_arm_count': audit.excluded_decision_group_count,
        'endpoint_count': len(audit.endpoint_mappings),
    }


def _validate_receipts(
    spec: EarlyClinicalEpisodeSpec,
    decision: LoadedAactSlice,
    label: LoadedAactSlice,
) -> None:
    if decision.receipt.nct_id != spec.nct_id or label.receipt.nct_id != spec.nct_id:
        raise AactAdapterError('both slices must have the exact NCT ID committed by the episode spec')
    if decision.receipt.archive.snapshot_id != spec.decision_snapshot_id:
        raise AactAdapterError('decision receipt snapshot ID does not match the episode spec')
    if label.receipt.archive.snapshot_id != spec.label_snapshot_id:
        raise AactAdapterError('label receipt snapshot ID does not match the episode spec')
    if decision.receipt.archive.source_cutoff_at != spec.decision_at:
        raise AactAdapterError('decision_at must equal the decision archive conservative cutoff')
    if label.receipt.archive.source_cutoff_at != spec.outcome_as_of:
        raise AactAdapterError('outcome_as_of must equal the label archive conservative cutoff')
    decision_tables = set(decision.rows_by_table)
    label_tables = set(label.rows_by_table)
    if missing := _DECISION_REQUIRED_TABLES - decision_tables:
        raise AactAdapterError(f'decision slice is missing required tables {sorted(table.value for table in missing)}')
    if missing := _LABEL_REQUIRED_TABLES - label_tables:
        raise AactAdapterError(f'label slice is missing required tables {sorted(table.value for table in missing)}')


def _validate_chronology(
    spec: EarlyClinicalEpisodeSpec,
    decision: LoadedAactSlice,
    label: LoadedAactSlice,
) -> tuple[StudyRow, StudyRow, date]:
    decision_studies = decision.rows(AactSourceTable.STUDIES, StudyRow)
    label_studies = label.rows(AactSourceTable.STUDIES, StudyRow)
    if len(decision_studies) != 1 or len(label_studies) != 1:
        raise AactAdapterError('each exact NCT slice must contain exactly one studies row')
    decision_study, label_study = decision_studies[0], label_studies[0]
    if decision_study.results_first_submitted_date.strip() or decision_study.results_first_posted_date.strip():
        raise AactAdapterError('decision snapshot is not pre-results according to studies.txt')
    decision_last_update = _iso_date(decision_study.last_update_posted_date, 'decision last_update_posted_date')
    if decision_last_update > spec.decision_at.date():
        raise AactAdapterError('decision studies row contains a post-cutoff update')
    if not label_study.results_first_submitted_date.strip():
        raise AactAdapterError('label studies row is missing results_first_submitted_date')
    results_first_posted = _iso_date(label_study.results_first_posted_date, 'results_first_posted_date')
    if results_first_posted <= spec.decision_at.date():
        raise AactAdapterError('results must first become public after the decision cutoff')
    if results_first_posted > spec.outcome_as_of.date():
        raise AactAdapterError('label snapshot predates results_first_posted_date')
    label_last_update = _iso_date(label_study.last_update_posted_date, 'label last_update_posted_date')
    if label_last_update > spec.outcome_as_of.date():
        raise AactAdapterError('label studies row contains an update after outcome_as_of')
    return decision_study, label_study, results_first_posted


def _validate_endpoint_crosswalk(
    spec: EarlyClinicalEpisodeSpec,
    design_outcomes: tuple[DesignOutcomeRow, ...],
    outcomes: tuple[OutcomeRow, ...],
    measurements: tuple[OutcomeMeasurementRow, ...],
    result_groups: tuple[ResultGroupRow, ...],
) -> tuple[
    tuple[ResolvedEndpointMapping, ...],
    dict[str, dict[str, OutcomeMeasurementRow]],
]:
    design_by_id = {row.id: row for row in design_outcomes}
    outcome_by_id = {row.id: row for row in outcomes}
    result_group_by_id = {row.id: row for row in result_groups}
    endpoint_audits: list[ResolvedEndpointMapping] = []
    measurements_by_endpoint: dict[str, dict[str, OutcomeMeasurementRow]] = {}
    complete_group_set: frozenset[str] | None = None
    for endpoint in spec.rubric.endpoints:
        design = design_by_id.get(endpoint.decision_outcome_id)
        if design is None or design.measure != endpoint.decision_measure:
            raise AactAdapterError(f'pre-cutoff endpoint {endpoint.endpoint_id} does not match its exact design row')
        if endpoint.classification.casefold() not in design.time_frame.casefold():
            raise AactAdapterError(f'pre-cutoff endpoint {endpoint.endpoint_id} does not declare the scored timepoint')
        outcome = outcome_by_id.get(endpoint.result_outcome_id)
        if outcome is None or outcome.title != endpoint.result_title or outcome.param_type != endpoint.param_type:
            raise AactAdapterError(f'label endpoint {endpoint.endpoint_id} does not match its exact outcome row')
        selected = [
            row
            for row in measurements
            if row.outcome_id == endpoint.result_outcome_id
            and row.classification == endpoint.classification
            and row.category == endpoint.category
            and row.param_type == endpoint.param_type
        ]
        by_group: dict[str, OutcomeMeasurementRow] = {}
        for row in selected:
            if row.result_group_id in by_group:
                raise AactAdapterError(
                    f'endpoint {endpoint.endpoint_id} has duplicate measurements for group {row.result_group_id}'
                )
            if row.result_group_id not in result_group_by_id:
                raise AactAdapterError(f'endpoint {endpoint.endpoint_id} references an unknown result group')
            if result_group_by_id[row.result_group_id].result_type.casefold() != 'outcome':
                raise AactAdapterError(f'endpoint {endpoint.endpoint_id} references a non-outcome result group')
            if row.title != outcome.title:
                raise AactAdapterError(f'endpoint {endpoint.endpoint_id} measurement title disagrees with outcomes.txt')
            by_group[row.result_group_id] = row
        if not by_group:
            raise AactAdapterError(f'endpoint {endpoint.endpoint_id} has no measurement rows at the scored timepoint')
        group_set = frozenset(by_group)
        if complete_group_set is None:
            complete_group_set = group_set
        elif group_set != complete_group_set:
            raise AactAdapterError('every target endpoint must expose the exact same complete result-group set')
        measurements_by_endpoint[endpoint.endpoint_id] = by_group
        endpoint_audits.append(
            ResolvedEndpointMapping(
                endpoint_id=endpoint.endpoint_id,
                decision_outcome_id=design.id,
                decision_measure=design.measure,
                decision_time_frame=design.time_frame,
                decision_description=design.description,
                result_outcome_id=outcome.id,
                result_title=outcome.title,
                result_time_frame=outcome.time_frame,
                result_description=outcome.description,
                result_param_type=outcome.param_type,
            )
        )
    return tuple(endpoint_audits), measurements_by_endpoint


def _resolve_complete_panel(
    spec: EarlyClinicalEpisodeSpec,
    design_groups: tuple[DesignGroupRow, ...],
    result_groups: tuple[ResultGroupRow, ...],
    measurements_by_endpoint: dict[str, dict[str, OutcomeMeasurementRow]],
) -> _PanelResolution:
    selector = spec.panel_selector
    allowed_types = {value.casefold() for value in selector.allowed_group_types}
    selected_decision = [
        row
        for row in design_groups
        if row.group_type.casefold() in allowed_types
        and _has_regimen_suffix(normalize_regimen_title(row.title), selector.normalized_regimen_key_suffix)
    ]
    decision_by_key = _unique_groups_by_regimen(selected_decision, 'decision')
    configured_keys = {mapping.regimen_key for mapping in spec.arm_mappings}
    if set(decision_by_key) != configured_keys:
        raise AactAdapterError(
            'arm_mappings must equal all and only protocol arms selected by the value-blind panel selector'
        )

    target_group_ids = set(next(iter(measurements_by_endpoint.values())))
    result_by_id = {row.id: row for row in result_groups}
    target_groups = [result_by_id[group_id] for group_id in sorted(target_group_ids)]
    selected_result = [
        row
        for row in target_groups
        if _has_regimen_suffix(normalize_regimen_title(row.title), selector.normalized_regimen_key_suffix)
    ]
    result_by_key = _unique_groups_by_regimen(selected_result, 'result')
    if set(result_by_key) != configured_keys:
        raise AactAdapterError(
            'arm_mappings must equal all and only target-outcome groups selected by the value-blind panel selector'
        )

    resolved: list[ResolvedArmMapping] = []
    for mapping in spec.arm_mappings:
        decision = decision_by_key[mapping.regimen_key]
        result = result_by_key[mapping.regimen_key]
        if decision.title != mapping.decision_title or result.title != mapping.result_title:
            raise AactAdapterError(f'arm {mapping.regimen_key} title does not match its precommitted source title')
        resolved.append(
            ResolvedArmMapping(
                role=mapping.role,
                candidate_id=mapping.candidate_id,
                regimen_key=mapping.regimen_key,
                decision_group_id=decision.id,
                result_group_id=result.id,
                decision_title=decision.title,
                result_title=result.title,
                decision_description=decision.description,
                result_description=result.description,
                decision_title_sha256=hashlib.sha256(decision.title.encode()).hexdigest(),
                result_title_sha256=hashlib.sha256(result.title.encode()).hexdigest(),
            )
        )
    all_outcome_group_ids = {row.id for row in result_groups if row.result_type.casefold() == 'outcome'}
    return _PanelResolution(
        resolved=tuple(resolved),
        decision_group_by_regimen=decision_by_key,
        result_group_by_regimen=result_by_key,
        decision_group_count=len(design_groups),
        excluded_decision_group_count=len(design_groups) - len(selected_decision),
        target_outcome_group_count=len(target_groups),
        excluded_target_result_group_count=len(target_groups) - len(selected_result),
        non_target_outcome_result_group_count=len(all_outcome_group_ids - target_group_ids),
    )


def _validate_intervention_links(
    panel: _PanelResolution,
    links: tuple[DesignGroupInterventionRow, ...],
    interventions: tuple[InterventionRow, ...],
) -> None:
    intervention_ids = {row.id for row in interventions}
    links_by_group: dict[str, set[str]] = {}
    for link in links:
        if link.intervention_id not in intervention_ids:
            raise AactAdapterError(f'design group link references unknown intervention {link.intervention_id}')
        links_by_group.setdefault(link.design_group_id, set()).add(link.intervention_id)
    missing = [group.id for group in panel.decision_group_by_regimen.values() if not links_by_group.get(group.id)]
    if missing:
        raise AactAdapterError(f'every selected protocol arm must have an intervention link; missing {missing}')


def _derive_outcomes(
    spec: EarlyClinicalEpisodeSpec,
    panel: _PanelResolution,
    measurements_by_endpoint: dict[str, dict[str, OutcomeMeasurementRow]],
) -> tuple[tuple[OutcomeRecord, ...], tuple[RankingLabelV1, ...], tuple[CandidateOutcomeAudit, ...]]:
    control_spec = next(mapping for mapping in spec.arm_mappings if mapping.role == ArmRole.CONTROL)
    control_group_id = panel.result_group_by_regimen[control_spec.regimen_key].id
    horizon_days = int((spec.outcome_as_of - spec.decision_at).total_seconds() // 86_400)
    private_outcomes: list[OutcomeRecord] = []
    ranking_labels: list[RankingLabelV1] = []
    audits: list[CandidateOutcomeAudit] = []
    candidate_specs = sorted(
        (mapping for mapping in spec.arm_mappings if mapping.role == ArmRole.CANDIDATE),
        key=lambda mapping: mapping.candidate_id or '',
    )
    for mapping in candidate_specs:
        assert mapping.candidate_id is not None
        result_group_id = panel.result_group_by_regimen[mapping.regimen_key].id
        folds: list[float] = []
        endpoint_audits: list[EndpointValueAudit] = []
        for endpoint in spec.rubric.endpoints:
            candidate_row = measurements_by_endpoint[endpoint.endpoint_id][result_group_id]
            control_row = measurements_by_endpoint[endpoint.endpoint_id][control_group_id]
            candidate_value = _positive_measurement_value(candidate_row, endpoint.endpoint_id, 'candidate')
            control_value = _positive_measurement_value(control_row, endpoint.endpoint_id, 'control')
            fold = candidate_value / control_value
            folds.append(fold)
            endpoint_audits.append(
                EndpointValueAudit(
                    endpoint_id=endpoint.endpoint_id,
                    outcome_id=endpoint.result_outcome_id,
                    result_group_id=result_group_id,
                    measurement_id=candidate_row.id,
                    control_measurement_id=control_row.id,
                    classification=candidate_row.classification,
                    param_type=candidate_row.param_type,
                    candidate_param_value=candidate_row.param_value,
                    control_param_value=control_row.param_value,
                    candidate_value=candidate_value,
                    control_value=control_value,
                    fold_over_control=fold,
                )
            )
        composite = math.exp(math.fsum(math.log(value) for value in folds) / len(folds))
        grade = _grade(composite)
        binary_outcome: Literal[0, 1] = 1 if composite >= spec.rubric.positive_threshold else 0
        candidate_utility = min(composite / spec.rubric.positive_threshold, 1.0)
        private_outcomes.append(
            OutcomeRecord(
                episode_id=spec.episode_id,
                candidate_id=mapping.candidate_id,
                target_id=spec.rubric.target_id,
                horizon_days=horizon_days,
                outcome=binary_outcome,
                candidate_utility=candidate_utility,
                revealed_at=spec.outcome_as_of,
            )
        )
        ranking_labels.append(
            RankingLabelV1(
                episode_id=spec.episode_id,
                candidate_id=mapping.candidate_id,
                relevance_grade=grade,
            )
        )
        audits.append(
            CandidateOutcomeAudit(
                candidate_id=mapping.candidate_id,
                endpoint_values=tuple(endpoint_audits),
                composite_fold_over_control=composite,
                relevance_grade=grade,
                binary_outcome=binary_outcome,
                candidate_utility=candidate_utility,
            )
        )
    return tuple(private_outcomes), tuple(ranking_labels), tuple(audits)


def _build_protocol_evidence(
    spec: EarlyClinicalEpisodeSpec,
    id_key: bytes,
    *,
    provenance_url: str,
    license_id: str,
) -> tuple[
    tuple[EvidenceRecord, ...],
    tuple[GoldAssessmentRecord, ...],
    tuple[GoldEvidenceRecord, ...],
]:
    evidence: list[EvidenceRecord] = []
    assessments: list[GoldAssessmentRecord] = []
    gold_evidence: list[GoldEvidenceRecord] = []
    candidate_specs = sorted(
        (mapping for mapping in spec.arm_mappings if mapping.role == ArmRole.CANDIDATE),
        key=lambda mapping: mapping.candidate_id or '',
    )
    candidate_ids = [mapping.candidate_id for mapping in candidate_specs if mapping.candidate_id is not None]
    target_body = _target_definition_body(spec)
    target_seed = canonical_json_bytes(
        {
            'adapter_id': AACT_ADAPTER_ID,
            'episode_id': spec.episode_id,
            'target_id': spec.rubric.target_id,
            'rubric_version': spec.rubric.version,
        }
    )
    evidence.append(
        EvidenceRecord(
            episode_id=spec.episode_id,
            evidence_id=f'ev-{hmac.new(id_key, target_seed, hashlib.sha256).hexdigest()[:24]}',
            source_type=SourceType.OTHER,
            collected_at=None,
            available_at=spec.decision_at,
            title='Benchmark target definition',
            body=target_body,
            body_sha256=body_sha256(target_body),
            related_candidate_ids=candidate_ids,
            provenance_url=f'urn:vaxreplay:rubric:{spec.rubric.version}',
            license_id='VaxReplay-benchmark-definition',
            derivation=(
                'Outcome-free benchmark scoring definition committed in the public episode. '
                'This definition is post-hoc relative to the historical trial.'
            ),
        )
    )
    for mapping in candidate_specs:
        assert mapping.candidate_id is not None
        regimen_quote = f'Protocol regimen: {mapping.public_regimen.rstrip(" .")}.'
        endpoint_quotes = [
            (
                f'Predeclared endpoint: {endpoint.public_name.rstrip(" .")}; '
                f'evaluation timepoint: {endpoint.classification}.'
            )
            for endpoint in spec.rubric.endpoints
        ]
        body = '\n'.join(
            [
                f'Candidate: {mapping.candidate_id}.',
                regimen_quote,
                *endpoint_quotes,
                (
                    'Scope note: these statements describe the pre-cutoff registry protocol; '
                    'they do not report later performance or clinical efficacy.'
                ),
            ]
        )
        seed = canonical_json_bytes(
            {
                'adapter_id': AACT_ADAPTER_ID,
                'episode_id': spec.episode_id,
                'candidate_id': mapping.candidate_id,
                'regimen_key': mapping.regimen_key,
            }
        )
        evidence_id = f'ev-{hmac.new(id_key, seed, hashlib.sha256).hexdigest()[:24]}'
        record = EvidenceRecord(
            episode_id=spec.episode_id,
            evidence_id=evidence_id,
            source_type=SourceType.PUBLIC_HEALTH,
            collected_at=None,
            available_at=spec.decision_at,
            title=f'Pre-cutoff protocol evidence for {mapping.candidate_id}',
            body=body,
            body_sha256=body_sha256(body),
            related_candidate_ids=[mapping.candidate_id],
            provenance_url=provenance_url,
            license_id=license_id,
            derivation=(
                'Deterministic identity-masked normalization of pre-cutoff AACT protocol rows. '
                'Exact registry identities and row mappings are retained only in the private audit.'
            ),
        )
        evidence.append(record)
        assessments.extend(
            [
                GoldAssessmentRecord(
                    episode_id=spec.episode_id,
                    candidate_id=mapping.candidate_id,
                    dimension=ENDPOINT_ALIGNMENT_DIMENSION,
                    conclusion=AssessmentConclusion.FAVORABLE,
                ),
                GoldAssessmentRecord(
                    episode_id=spec.episode_id,
                    candidate_id=mapping.candidate_id,
                    dimension=REGIMEN_DEFINITION_DIMENSION,
                    conclusion=AssessmentConclusion.FAVORABLE,
                ),
            ]
        )
        gold_evidence.append(
            GoldEvidenceRecord(
                episode_id=spec.episode_id,
                candidate_id=mapping.candidate_id,
                dimension=REGIMEN_DEFINITION_DIMENSION,
                evidence_id=evidence_id,
                stance=EvidenceStance.SUPPORT,
                quote=regimen_quote,
            )
        )
        gold_evidence.extend(
            GoldEvidenceRecord(
                episode_id=spec.episode_id,
                candidate_id=mapping.candidate_id,
                dimension=ENDPOINT_ALIGNMENT_DIMENSION,
                evidence_id=evidence_id,
                stance=EvidenceStance.SUPPORT,
                quote=quote,
            )
            for quote in endpoint_quotes
        )
    return tuple(evidence), tuple(assessments), tuple(gold_evidence)


def _target_definition_body(spec: EarlyClinicalEpisodeSpec) -> str:
    control = next(mapping for mapping in spec.arm_mappings if mapping.role == ArmRole.CONTROL)
    endpoint_lines = [
        f'- {endpoint.public_name.rstrip(" .")} at {endpoint.classification}.' for endpoint in spec.rubric.endpoints
    ]
    return '\n'.join(
        [
            f'Target: {spec.rubric.target_id}.',
            f'Control denominator: {control.public_regimen.rstrip(" .")}.',
            'For each endpoint, divide the candidate point estimate by the matching panel control estimate.',
            'Combine all endpoint ratios using an equal-weight geometric mean.',
            'Binary outcome: positive when the composite is greater than or equal to 8; negative otherwise.',
            (
                'Ranking relevance grades: 0 below 1; 1 from 1 to below 2; 2 from 2 to below 4; '
                '3 from 4 to below 8; 4 at or above 8.'
            ),
            'Candidate utility: min(composite divided by 8, 1).',
            (
                'Proxy scope: point estimates only; confidence intervals, group denominators, safety, '
                'reactogenicity, and actual development advancement are not scored.'
            ),
            'Configured endpoints:',
            *endpoint_lines,
            (
                'Status: this is a benchmark-defined post-hoc target, not a historical trial endpoint, '
                'prospective registration, or validated clinical utility function.'
            ),
        ]
    )


def _source_commitment(snapshot: LoadedAactSlice) -> SourceSnapshotCommitment:
    archive = snapshot.receipt.archive
    return SourceSnapshotCommitment(
        snapshot_id=archive.snapshot_id,
        source_build_at=archive.source_cutoff_at,
        manifest_sha256=snapshot.receipt_sha256,
        source_url=archive.source_url,
        license_id=archive.license_id,
        license_url=archive.license_url,
        citation=archive.citation,
    )


def _assert_public_identity_masked(
    spec: EarlyClinicalEpisodeSpec,
    manifest: EpisodeManifest,
    candidates: tuple[CandidateRecord, ...],
    evidence: tuple[EvidenceRecord, ...],
) -> None:
    public_payload = (
        b'\n'.join(
            [canonical_json_bytes(manifest)]
            + [canonical_json_bytes(candidate) for candidate in candidates]
            + [canonical_json_bytes(record) for record in evidence]
        )
        .decode('utf-8')
        .casefold()
    )
    leaked = [token for token in spec.forbidden_public_tokens if token.casefold() in public_payload]
    if leaked or re.search(r'nct\d{8}', public_payload, flags=re.IGNORECASE):
        raise AactAdapterError(f'identity-masked public evidence contains forbidden source identity: {leaked}')


def _write_episode_atomically(
    *,
    output_root: Path,
    manifest: EpisodeManifest,
    candidates: tuple[CandidateRecord, ...],
    evidence: tuple[EvidenceRecord, ...],
    private_labels: PrivateLabels,
    ranking_labels: tuple[RankingLabelV1, ...],
    private_audit: AactPrivateAudit,
    spec: EarlyClinicalEpisodeSpec,
    decision_receipt: AactSliceReceipt,
    label_receipt: AactSliceReceipt,
    label_commitment_key: bytes,
) -> None:
    if output_root.exists():
        raise AactAdapterError(f'output directory already exists: {output_root}')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f'.{output_root.name}.', dir=output_root.parent)).resolve()
    try:
        private_root = temporary_root / 'private'
        private_root.mkdir()
        (temporary_root / 'manifest.json').write_bytes(canonical_json_bytes(manifest) + b'\n')
        (temporary_root / 'candidates.jsonl').write_text(jsonl_text(candidates), encoding='utf-8')
        (temporary_root / 'evidence.jsonl').write_text(jsonl_text(evidence), encoding='utf-8')
        (private_root / 'outcomes.jsonl').write_text(jsonl_text(private_labels.outcomes), encoding='utf-8')
        (private_root / 'assessments_gold.jsonl').write_text(
            jsonl_text(private_labels.assessments_gold), encoding='utf-8'
        )
        (private_root / 'evidence_gold.jsonl').write_text(jsonl_text(private_labels.evidence_gold), encoding='utf-8')
        (private_root / 'ranking_labels.jsonl').write_text(jsonl_text(ranking_labels), encoding='utf-8')
        (private_root / 'aact_audit.json').write_bytes(canonical_json_bytes(private_audit) + b'\n')
        (private_root / 'aact_episode_spec.json').write_bytes(canonical_json_bytes(spec) + b'\n')
        (private_root / 'aact_decision_slice_receipt.json').write_bytes(canonical_json_bytes(decision_receipt) + b'\n')
        (private_root / 'aact_label_slice_receipt.json').write_bytes(canonical_json_bytes(label_receipt) + b'\n')
        (private_root / 'label_commitment_key.hex').write_text(label_commitment_key.hex() + '\n', encoding='ascii')
        source_provenance = manifest.source_provenance
        assert source_provenance is not None
        report = {
            'adapter_id': AACT_ADAPTER_ID,
            'episode_id': spec.episode_id,
            'candidate_count': len(candidates),
            'evidence_count': len(evidence),
            'reward_version': spec.reward_version,
            'rubric_version': spec.rubric.version,
            'source_provenance': source_provenance.model_dump(mode='json'),
        }
        (temporary_root / 'ADAPTER_REPORT.json').write_bytes(canonical_json_bytes(report) + b'\n')
        (temporary_root / 'DATASET_CARD.md').write_text(_dataset_card(spec, len(candidates)), encoding='utf-8')
        EpisodeBundle.load(temporary_root, include_private=True)
        audit_episode(temporary_root)
        temporary_root.rename(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _dataset_card(spec: EarlyClinicalEpisodeSpec, candidate_count: int) -> str:
    return f"""# {spec.episode_id}

This real-data historical replay was derived from two content-addressed AACT monthly archives.
Public protocol evidence is identity-masked; exact registry identity, source rows, result values,
arm joins, and label commitments remain private.

- Adapter: `{AACT_ADAPTER_ID}`
- Task: `{EARLY_CLINICAL_TASK}`
- Reward: `{spec.reward_version}`
- Outcome rubric: `{spec.rubric.version}`
- Candidate arms: {candidate_count}
- Portfolio size: {spec.portfolio_size}
- Decision time: `{spec.decision_at.isoformat()}`

AACT guarantees the archive date rather than an exact build instant. Decision and outcome times use
the receipt's `archive-date-end-utc-upper-bound-v1` convention: end-of-day UTC on each archive date.
This is a conservative benchmark cutoff, not a claimed server build timestamp.

The label is an intentionally transparent diagnostic composite: each configured endpoint's point
estimate is divided by the contemporaneous control value, endpoint folds are combined by geometric
mean, grades use fixed boundaries `<1`, `1-<2`, `2-<4`, `4-<8`, and `>=8`, and the binary target is
positive at `>=8`. Candidate utility is `min(composite / 8, 1)`. This is not a validated clinical
utility function, vaccine-efficacy endpoint, or recommendation to conduct an experiment.
The V1 proxy uses point estimates only and omits confidence intervals, group denominators, safety,
reactogenicity, and the historical sponsor's actual advancement decision.

Grounding evaluates only two honest pre-cutoff protocol facts: the supplied regimen definition and
alignment to the predeclared endpoints. It does not treat protocol registration as evidence of later
immunogenicity, protection, safety, or efficacy.

The adapter commitments prove that the rubric and source mappings were fixed before this episode
was released. They do **not** prove that the rubric was fixed before the historical outcomes. This
pilot is a post-hoc historical replay, not a prospective or historically preregistered evaluation.
"""


def _read_private_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    if path.is_symlink() or not path.is_file():
        raise AactAdapterError(f'private AACT artifact must be a regular, non-symlink file: {path}')
    try:
        return model.model_validate_json(path.read_bytes())
    except OSError as error:
        raise AactAdapterError(f'cannot read {path}: {error}') from error
    except ValueError as error:
        raise AactAdapterError(f'invalid private AACT artifact {path.name}: {error}') from error


def _iso_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as error:
        raise AactAdapterError(f'{field_name} must be a non-empty ISO date') from error


def _has_regimen_suffix(regimen_key: str, suffix: str) -> bool:
    return regimen_key == suffix or regimen_key.endswith(f'-{suffix}')


def _unique_groups_by_regimen[RowT: DesignGroupRow | ResultGroupRow](
    rows: list[RowT],
    label: str,
) -> dict[str, RowT]:
    by_key: dict[str, RowT] = {}
    for row in rows:
        key = normalize_regimen_title(row.title)
        if key in by_key:
            raise AactAdapterError(f'{label} panel has duplicate normalized regimen key {key}')
        by_key[key] = row
    return by_key


def _positive_measurement_value(row: OutcomeMeasurementRow, endpoint_id: str, role: str) -> float:
    if row.param_value_num is None or not row.param_value.strip() or row.param_value_num <= 0:
        raise AactAdapterError(f'{endpoint_id} {role} measurement must contain a positive numeric point estimate')
    return row.param_value_num


def _grade(composite: float) -> int:
    grade = 0
    for index, threshold in enumerate((1.0, 2.0, 4.0, 8.0), start=1):
        if composite >= threshold:
            grade = index
    return grade
