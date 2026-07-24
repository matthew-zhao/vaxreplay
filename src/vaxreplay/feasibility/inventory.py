"""Deterministic offline matching and aggregate feasibility reporting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, date, timedelta
from pathlib import Path

from pydantic import BaseModel, ValidationError

from vaxreplay.bundle import canonical_json_bytes, jsonl_text, records_sha256
from vaxreplay.feasibility.schema import (
    PUBLIC_COMPLEMENTARY_GROUPS,
    PUBLIC_DERIVED_DIFFERENCE_PAIRS,
    PUBLIC_DERIVED_RESIDUAL_TRIPLES,
    PUBLIC_SMALL_CELL_THRESHOLD,
    AdmissionTier,
    CtgovHistoricalRecord,
    FeasibilityInventoryReport,
    FeasibilityInventorySpec,
    GateResult,
    GateStatus,
    HistorySurface,
    ImmportDiscoveryRecord,
    InventoryDisposition,
    InventoryMetrics,
    InventoryReasonCode,
    LinkedStudyInventoryRecord,
    LinkStatus,
    PostedDateType,
    PublicFeasibilitySummary,
    PublicGateResult,
    PublicInventoryMetrics,
    SnapshotReceipt,
    SnapshotSource,
    StudyType,
    VaccineRelevance,
)

RECEIPT_FILENAME = 'receipt.json'
RECORDS_FILENAME = 'records.jsonl'
SPEC_FILENAME = 'spec.json'
REPORT_FILENAME = 'report.json'
IMMPORT_SOURCE_RECORDS_FILENAME = 'immport_source_records.jsonl'
CTGOV_SOURCE_RECORDS_FILENAME = 'ctgov_source_records.jsonl'


class FeasibilityIntegrityError(ValueError):
    """Raised when an inventory snapshot or output breaks its commitment."""


def build_inventory(
    *,
    spec_path: Path,
    immport_root: Path,
    ctgov_root: Path,
    output_root: Path,
) -> FeasibilityInventoryReport:
    """Build a private record inventory and aggregate report from pinned normalized snapshots."""

    spec = _load_model(spec_path, FeasibilityInventorySpec)
    immport_receipt, immport_records = _load_snapshot(
        immport_root,
        expected_source=SnapshotSource.IMMPORT,
        model=ImmportDiscoveryRecord,
    )
    ctgov_receipt, ctgov_records = _load_snapshot(
        ctgov_root,
        expected_source=SnapshotSource.CLINICALTRIALS_GOV,
        model=CtgovHistoricalRecord,
    )
    _validate_input_bindings(spec, immport_receipt, ctgov_receipt, immport_records, ctgov_records)

    linked_records = _link_records(
        immport_records,
        ctgov_records,
        spec=spec,
        required_assay_methods=set(spec.required_assay_methods),
        redistribution_cleared=(immport_receipt.redistribution_cleared and ctgov_receipt.redistribution_cleared),
    )
    report = _make_report(
        spec,
        immport_receipt,
        ctgov_receipt,
        immport_records,
        ctgov_records,
        linked_records,
    )
    _write_inventory_output(
        output_root,
        spec,
        immport_receipt,
        ctgov_receipt,
        immport_records,
        ctgov_records,
        linked_records,
        report,
    )
    return report


def audit_inventory(root: Path) -> FeasibilityInventoryReport:
    """Recompute commitments for a previously built private inventory."""

    spec = _load_model(root / SPEC_FILENAME, FeasibilityInventorySpec)
    immport_receipt = _load_model(root / 'immport_receipt.json', SnapshotReceipt)
    ctgov_receipt = _load_model(root / 'ctgov_receipt.json', SnapshotReceipt)
    report = _load_model(root / REPORT_FILENAME, FeasibilityInventoryReport)
    records = _load_jsonl(root / RECORDS_FILENAME, LinkedStudyInventoryRecord)
    immport_records = _load_jsonl(root / IMMPORT_SOURCE_RECORDS_FILENAME, ImmportDiscoveryRecord)
    ctgov_records = _load_jsonl(root / CTGOV_SOURCE_RECORDS_FILENAME, CtgovHistoricalRecord)
    _validate_input_bindings(spec, immport_receipt, ctgov_receipt, immport_records, ctgov_records)
    if report.spec_sha256 != _model_sha256(spec):
        raise FeasibilityIntegrityError('inventory report has the wrong spec commitment')
    if report.immport_receipt_sha256 != _model_sha256(immport_receipt):
        raise FeasibilityIntegrityError('inventory report has the wrong ImmPort receipt commitment')
    if report.ctgov_receipt_sha256 != _model_sha256(ctgov_receipt):
        raise FeasibilityIntegrityError('inventory report has the wrong ClinicalTrials.gov receipt commitment')
    if report.records_sha256 != records_sha256(records):
        raise FeasibilityIntegrityError('inventory report has the wrong record commitment')
    if report.record_count != len(records):
        raise FeasibilityIntegrityError('inventory report record_count does not match records.jsonl')
    _verify_committed_records(immport_receipt, immport_records, SnapshotSource.IMMPORT)
    _verify_committed_records(ctgov_receipt, ctgov_records, SnapshotSource.CLINICALTRIALS_GOV)
    expected_records = _link_records(
        immport_records,
        ctgov_records,
        spec=spec,
        required_assay_methods=set(spec.required_assay_methods),
        redistribution_cleared=(immport_receipt.redistribution_cleared and ctgov_receipt.redistribution_cleared),
    )
    if records != expected_records:
        raise FeasibilityIntegrityError('inventory records do not match the committed source records')
    expected_report = _make_report(
        spec,
        immport_receipt,
        ctgov_receipt,
        immport_records,
        ctgov_records,
        expected_records,
    )
    if report != expected_report:
        raise FeasibilityIntegrityError('inventory report does not recompute from the committed records')
    return report


def export_public_summary(inventory_root: Path, output_path: Path) -> PublicFeasibilitySummary:
    """Export only the aggregate report; record-level source mappings remain private."""

    report = audit_inventory(inventory_root)
    spec = _load_model(inventory_root / SPEC_FILENAME, FeasibilityInventorySpec)
    try:
        public_summary = PublicFeasibilitySummary(
            synthetic=spec.synthetic,
            record_count=report.record_count,
            metrics=_make_public_metrics(report.metrics, synthetic=spec.synthetic),
            gates=tuple(PublicGateResult(gate_id=gate.gate_id, status=gate.status) for gate in report.gates),
            admission_tier=report.admission_tier,
        )
    except ValidationError:
        raise FeasibilityIntegrityError('private report contains metadata unsafe for public export') from None
    if output_path.exists():
        raise FeasibilityIntegrityError(f'public summary output already exists: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(public_summary) + b'\n')
    return public_summary


def _make_public_metrics(metrics: InventoryMetrics, *, synthetic: bool) -> PublicInventoryMetrics:
    private_values = metrics.model_dump(mode='python')
    values: dict[str, int | str] = {
        name: value if synthetic or value == 0 or value >= PUBLIC_SMALL_CELL_THRESHOLD else '<5'
        for name, value in private_values.items()
    }
    if not synthetic:
        for group in PUBLIC_COMPLEMENTARY_GROUPS:
            if any(0 < private_values[name] < PUBLIC_SMALL_CELL_THRESHOLD for name in group):
                complement = max(group, key=lambda name: private_values[name])
                values[complement] = 'suppressed'
        for total_name, subset_name in PUBLIC_DERIVED_DIFFERENCE_PAIRS:
            difference = private_values[total_name] - private_values[subset_name]
            if 0 < difference < PUBLIC_SMALL_CELL_THRESHOLD:
                values[subset_name] = 'suppressed'
        for total_name, first_name, second_name in PUBLIC_DERIVED_RESIDUAL_TRIPLES:
            residual = private_values[total_name] - private_values[first_name] - private_values[second_name]
            if 0 < residual < PUBLIC_SMALL_CELL_THRESHOLD:
                values[total_name] = 'suppressed'
    return PublicInventoryMetrics.model_validate(values)


def _validate_input_bindings(
    spec: FeasibilityInventorySpec,
    immport_receipt: SnapshotReceipt,
    ctgov_receipt: SnapshotReceipt,
    immport_records: tuple[ImmportDiscoveryRecord, ...],
    ctgov_records: tuple[CtgovHistoricalRecord, ...],
) -> None:
    if immport_receipt.source != SnapshotSource.IMMPORT:
        raise FeasibilityIntegrityError('ImmPort receipt declares the wrong source')
    if ctgov_receipt.source != SnapshotSource.CLINICALTRIALS_GOV:
        raise FeasibilityIntegrityError('ClinicalTrials.gov receipt declares the wrong source')
    if spec.immport_receipt_sha256 != _model_sha256(immport_receipt):
        raise FeasibilityIntegrityError('ImmPort receipt does not match the inventory spec')
    if spec.ctgov_receipt_sha256 != _model_sha256(ctgov_receipt):
        raise FeasibilityIntegrityError('ClinicalTrials.gov receipt does not match the inventory spec')
    if spec.synthetic != immport_receipt.synthetic or spec.synthetic != ctgov_receipt.synthetic:
        raise FeasibilityIntegrityError('spec and both source receipts must agree on synthetic status')
    if any(record.source_release_id != immport_receipt.source_version for record in immport_records):
        raise FeasibilityIntegrityError('ImmPort record release IDs must match the receipt source_version')
    immport_retrieval_date = immport_receipt.retrieved_at.astimezone(UTC).date()
    ctgov_retrieval_date = ctgov_receipt.retrieved_at.astimezone(UTC).date()
    if spec.outcome_as_of > min(immport_retrieval_date, ctgov_retrieval_date):
        raise FeasibilityIntegrityError('outcome_as_of cannot follow either source retrieval date')
    if any(
        observed_date is not None and observed_date > immport_retrieval_date
        for record in immport_records
        for observed_date in (
            record.initial_release_date,
            record.latest_release_date,
            *(observation.first_observed_date for observation in record.assay_first_observations),
        )
    ):
        raise FeasibilityIntegrityError('ImmPort availability dates cannot follow snapshot retrieval')
    _validate_ctgov_availability_dates(ctgov_receipt, ctgov_records)


def _validate_ctgov_availability_dates(
    receipt: SnapshotReceipt,
    records: tuple[CtgovHistoricalRecord, ...],
) -> None:
    retrieval_date = receipt.retrieved_at.astimezone(UTC).date()
    for record in records:
        dates = (
            record.historical_submitted_date,
            record.historical_posted_date,
            record.current_results_first_post_date,
        )
        if any(value is not None and value > retrieval_date for value in dates):
            raise FeasibilityIntegrityError('ClinicalTrials.gov availability dates cannot follow snapshot retrieval')


def _load_snapshot[
    RecordT: BaseModel,
](
    root: Path,
    *,
    expected_source: SnapshotSource,
    model: type[RecordT],
) -> tuple[SnapshotReceipt, tuple[RecordT, ...]]:
    receipt = _load_model(root / RECEIPT_FILENAME, SnapshotReceipt)
    if receipt.source != expected_source:
        raise FeasibilityIntegrityError(f'expected {expected_source.value} receipt, got {receipt.source.value}')
    records_path = _safe_child(root, receipt.records_relative_path)
    try:
        payload = records_path.read_bytes()
    except OSError as error:
        raise FeasibilityIntegrityError(f'cannot read {records_path}: {error}') from error
    if len(payload) != receipt.byte_count:
        raise FeasibilityIntegrityError(f'{expected_source.value} records byte_count mismatch')
    if hashlib.sha256(payload).hexdigest() != receipt.records_sha256:
        raise FeasibilityIntegrityError(f'{expected_source.value} records SHA-256 mismatch')
    records = _load_jsonl(records_path, model)
    if len(records) != receipt.record_count:
        raise FeasibilityIntegrityError(f'{expected_source.value} records count mismatch')
    if payload != jsonl_text(records).encode('utf-8'):
        raise FeasibilityIntegrityError(f'{expected_source.value} records must use canonical JSONL')
    return receipt, records


def _verify_committed_records(
    receipt: SnapshotReceipt,
    records: tuple[BaseModel, ...],
    source: SnapshotSource,
) -> None:
    payload = jsonl_text(records).encode('utf-8')
    if len(payload) != receipt.byte_count:
        raise FeasibilityIntegrityError(f'private {source.value} source records byte_count mismatch')
    if hashlib.sha256(payload).hexdigest() != receipt.records_sha256:
        raise FeasibilityIntegrityError(f'private {source.value} source records SHA-256 mismatch')
    if len(records) != receipt.record_count:
        raise FeasibilityIntegrityError(f'private {source.value} source records count mismatch')


def _load_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    try:
        with path.open(encoding='utf-8') as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise FeasibilityIntegrityError(f'{path} contains blank line {line_number}')
                _reject_duplicate_json_keys(line, path=path, line_number=line_number)
                records.append(model.model_validate_json(line))
    except OSError as error:
        raise FeasibilityIntegrityError(f'cannot read {path}: {error}') from error
    except (ValueError, ValidationError) as error:
        raise FeasibilityIntegrityError(f'invalid record in {path}: {error}') from error
    if not records:
        raise FeasibilityIntegrityError(f'{path} must contain at least one record')
    return tuple(records)


def _load_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        payload = path.read_bytes()
        _reject_duplicate_json_keys(payload, path=path)
        return model.model_validate_json(payload)
    except OSError as error:
        raise FeasibilityIntegrityError(f'cannot read {path}: {error}') from error
    except (ValueError, ValidationError) as error:
        raise FeasibilityIntegrityError(f'invalid {path}: {error}') from error


def _reject_duplicate_json_keys(
    payload: str | bytes,
    *,
    path: Path,
    line_number: int | None = None,
) -> None:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise FeasibilityIntegrityError(f'duplicate JSON key {key!r}')
            result[key] = value
        return result

    try:
        json.loads(payload, object_pairs_hook=reject_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError, FeasibilityIntegrityError) as error:
        location = f'{path}:{line_number}' if line_number is not None else str(path)
        raise FeasibilityIntegrityError(f'invalid unambiguous JSON in {location}: {error}') from error


def _safe_child(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise FeasibilityIntegrityError('snapshot record path escapes its root')
    return candidate


def _model_sha256(model: BaseModel) -> str:
    return hashlib.sha256(canonical_json_bytes(model)).hexdigest()


def _link_records(
    immport_records: tuple[ImmportDiscoveryRecord, ...],
    ctgov_records: tuple[CtgovHistoricalRecord, ...],
    *,
    spec: FeasibilityInventorySpec,
    required_assay_methods: set[str],
    redistribution_cleared: bool,
) -> tuple[LinkedStudyInventoryRecord, ...]:
    study_ids = [record.study_accession for record in immport_records]
    if len(study_ids) != len(set(study_ids)):
        raise FeasibilityIntegrityError('ImmPort study accessions must be unique')
    nct_ids = [record.nct_id for record in ctgov_records]
    if len(nct_ids) != len(set(nct_ids)):
        raise FeasibilityIntegrityError('ClinicalTrials.gov NCT IDs must be unique')
    ctgov_by_id = {record.nct_id: record for record in ctgov_records}
    exact_nct_counts = Counter(
        record.explicit_nct_ids[0] for record in immport_records if len(record.explicit_nct_ids) == 1
    )
    linked: list[LinkedStudyInventoryRecord] = []
    for immport in sorted(immport_records, key=lambda record: record.study_accession):
        linked.append(
            _link_one(
                immport,
                ctgov_by_id,
                spec=spec,
                duplicate_nct_ids={nct_id for nct_id, count in exact_nct_counts.items() if count > 1},
                required_assay_methods=required_assay_methods,
                redistribution_cleared=redistribution_cleared,
            )
        )
    return tuple(linked)


def _link_one(
    immport: ImmportDiscoveryRecord,
    ctgov_by_id: dict[str, CtgovHistoricalRecord],
    *,
    spec: FeasibilityInventorySpec,
    duplicate_nct_ids: set[str],
    required_assay_methods: set[str],
    redistribution_cleared: bool,
) -> LinkedStudyInventoryRecord:
    reasons: set[InventoryReasonCode] = set()
    if not immport.explicit_nct_ids:
        reasons.add(InventoryReasonCode.NO_EXPLICIT_NCT_LINK)
        return _unresolved_record(immport, LinkStatus.MISSING, reasons, required_assay_methods)
    if len(immport.explicit_nct_ids) > 1:
        reasons.add(InventoryReasonCode.MULTIPLE_NCT_LINKS)
        return _unresolved_record(immport, LinkStatus.AMBIGUOUS, reasons, required_assay_methods)

    nct_id = immport.explicit_nct_ids[0]
    ctgov = ctgov_by_id.get(nct_id)
    if ctgov is None:
        reasons.add(InventoryReasonCode.NCT_RECORD_MISSING)
        return LinkedStudyInventoryRecord(
            study_accession=immport.study_accession,
            nct_ids=immport.explicit_nct_ids,
            link_status=LinkStatus.RECORD_NOT_FOUND,
            disposition=InventoryDisposition.EXCLUDE,
            immport_arm_count=immport.arm_count,
            assay_metadata_present=bool(required_assay_methods & set(immport.assay_methods)),
            reasons=_sorted_reasons(reasons),
        )

    assay_metadata_present = bool(required_assay_methods & set(immport.assay_methods))
    if not immport.clinical_trial:
        reasons.add(InventoryReasonCode.NOT_CLINICAL_TRIAL)
    if not immport.human:
        reasons.add(InventoryReasonCode.NOT_HUMAN)
    if immport.vaccine_relevance == VaccineRelevance.SOURCE_FILTER:
        reasons.add(InventoryReasonCode.VACCINE_RELEVANCE_NOT_CURATED)
    elif immport.vaccine_relevance == VaccineRelevance.REJECTED:
        reasons.add(InventoryReasonCode.VACCINE_RELEVANCE_REJECTED)
    if not assay_metadata_present:
        reasons.add(InventoryReasonCode.NO_IMMUNE_ASSAY_METADATA)
    if immport.arm_count < 2:
        reasons.add(InventoryReasonCode.IMM_PORT_ARM_COUNT_LT_2)
    if ctgov.historical_study_type != StudyType.INTERVENTIONAL:
        reasons.add(InventoryReasonCode.NOT_INTERVENTIONAL)
    if ctgov.historical_arm_count < 2:
        reasons.add(InventoryReasonCode.HISTORICAL_ARM_COUNT_LT_2)
    if ctgov.historical_biological_intervention_count < 2:
        reasons.add(InventoryReasonCode.BIOLOGICAL_CANDIDATE_COUNT_LT_2)
    if immport.arm_count != ctgov.historical_arm_count:
        reasons.add(InventoryReasonCode.ARM_COUNT_MISMATCH)
    if ctgov.history_surface == HistorySurface.PUBLIC_UI_INTERNAL:
        reasons.add(InventoryReasonCode.HISTORY_SURFACE_UNSUPPORTED)
    if ctgov.historical_posted_date is None:
        reasons.add(InventoryReasonCode.HISTORICAL_POST_DATE_MISSING)
    elif ctgov.historical_posted_date_type is None:
        reasons.add(InventoryReasonCode.HISTORICAL_POST_DATE_TYPE_MISSING)
    elif ctgov.historical_posted_date_type == PostedDateType.ESTIMATED:
        reasons.add(InventoryReasonCode.HISTORICAL_POST_DATE_ESTIMATED)
    if ctgov.historical_has_results or ctgov.historical_results_section_present:
        reasons.add(InventoryReasonCode.HISTORY_CONTAINS_RESULTS)
    if (
        ctgov.current_results_first_post_date is not None
        and ctgov.historical_posted_date is not None
        and ctgov.historical_posted_date >= ctgov.current_results_first_post_date
    ):
        reasons.add(InventoryReasonCode.HISTORY_NOT_BEFORE_RESULTS)
    if ctgov.current_has_results and ctgov.current_results_first_post_date_type is None:
        reasons.add(InventoryReasonCode.RESULTS_FIRST_POST_DATE_TYPE_MISSING)
    elif ctgov.current_results_first_post_date_type == PostedDateType.ESTIMATED:
        reasons.add(InventoryReasonCode.RESULTS_FIRST_POST_DATE_ESTIMATED)
    assay_first_observed_dates = _required_assay_first_observed_dates(immport, required_assay_methods)
    if not assay_first_observed_dates:
        reasons.add(InventoryReasonCode.ASSAY_METADATA_FIRST_OBSERVED_DATE_MISSING)
    elif ctgov.historical_posted_date is not None:
        minimum_outcome_date = ctgov.historical_posted_date + timedelta(days=spec.thresholds.minimum_outcome_delay_days)
        if max(assay_first_observed_dates) < minimum_outcome_date:
            reasons.add(InventoryReasonCode.ASSAY_METADATA_NOT_AFTER_DECISION)
        elif min(assay_first_observed_dates) > spec.outcome_as_of:
            reasons.add(InventoryReasonCode.ASSAY_METADATA_AFTER_OUTCOME_AS_OF)
        elif not any(minimum_outcome_date <= observed <= spec.outcome_as_of for observed in assay_first_observed_dates):
            reasons.add(InventoryReasonCode.ASSAY_METADATA_NO_OBSERVATION_IN_WINDOW)
    if nct_id in duplicate_nct_ids:
        reasons.add(InventoryReasonCode.DUPLICATE_NCT_MAPPING)
    reasons.add(InventoryReasonCode.ARM_MAPPING_NOT_ASSESSED)
    reasons.add(InventoryReasonCode.OUTCOME_COMPARABILITY_NOT_ASSESSED)
    if not redistribution_cleared:
        reasons.add(InventoryReasonCode.REDISTRIBUTION_NOT_CLEARED)

    supported_pre_results = _supported_pre_results_history(ctgov)
    hard_exclusions = {
        InventoryReasonCode.NOT_CLINICAL_TRIAL,
        InventoryReasonCode.NOT_HUMAN,
        InventoryReasonCode.VACCINE_RELEVANCE_REJECTED,
        InventoryReasonCode.NOT_INTERVENTIONAL,
        InventoryReasonCode.HISTORY_CONTAINS_RESULTS,
        InventoryReasonCode.HISTORY_NOT_BEFORE_RESULTS,
    }
    return LinkedStudyInventoryRecord(
        study_accession=immport.study_accession,
        nct_ids=immport.explicit_nct_ids,
        nct_id=nct_id,
        link_status=LinkStatus.EXACT,
        disposition=(InventoryDisposition.EXCLUDE if reasons & hard_exclusions else InventoryDisposition.HOLD),
        immport_arm_count=immport.arm_count,
        ctgov_historical_arm_count=ctgov.historical_arm_count,
        arm_count_agreement=immport.arm_count == ctgov.historical_arm_count,
        assay_metadata_present=assay_metadata_present,
        supported_pre_results_history=supported_pre_results,
        reasons=_sorted_reasons(reasons),
    )


def _unresolved_record(
    immport: ImmportDiscoveryRecord,
    status: LinkStatus,
    reasons: set[InventoryReasonCode],
    required_assay_methods: set[str],
) -> LinkedStudyInventoryRecord:
    return LinkedStudyInventoryRecord(
        study_accession=immport.study_accession,
        nct_ids=immport.explicit_nct_ids,
        link_status=status,
        disposition=InventoryDisposition.EXCLUDE,
        immport_arm_count=immport.arm_count,
        assay_metadata_present=bool(required_assay_methods & set(immport.assay_methods)),
        reasons=_sorted_reasons(reasons),
    )


def _sorted_reasons(reasons: Iterable[InventoryReasonCode]) -> tuple[InventoryReasonCode, ...]:
    return tuple(sorted(set(reasons), key=lambda reason: reason.value))


def _supported_pre_results_history(record: CtgovHistoricalRecord) -> bool:
    return (
        record.history_surface != HistorySurface.PUBLIC_UI_INTERNAL
        and record.historical_posted_date is not None
        and record.historical_posted_date_type == PostedDateType.ACTUAL
        and (not record.current_has_results or record.current_results_first_post_date_type == PostedDateType.ACTUAL)
        and _pre_results_history(record)
    )


def _required_assay_first_observed_dates(
    record: ImmportDiscoveryRecord,
    required_assay_methods: set[str],
) -> tuple[date, ...]:
    observed_dates = tuple(
        observation.first_observed_date
        for observation in record.assay_first_observations
        if observation.method in required_assay_methods
    )
    return tuple(sorted(set(observed_dates)))


def _pre_results_history(record: CtgovHistoricalRecord) -> bool:
    if record.historical_has_results or record.historical_results_section_present:
        return False
    if record.historical_posted_date is None:
        return False
    results_first_post_date = record.current_results_first_post_date
    return results_first_post_date is None or record.historical_posted_date < results_first_post_date


def _eligible_supported_temporal_candidate(
    immport: ImmportDiscoveryRecord,
    ctgov: CtgovHistoricalRecord,
    linked: LinkedStudyInventoryRecord,
    *,
    duplicate_nct_ids: set[str],
    spec: FeasibilityInventorySpec,
) -> bool:
    if linked.nct_id is None or linked.nct_id in duplicate_nct_ids:
        return False
    if not immport.clinical_trial or not immport.human:
        return False
    if immport.vaccine_relevance != VaccineRelevance.CURATOR_CONFIRMED:
        return False
    if immport.arm_count < 2 or not linked.assay_metadata_present:
        return False
    if ctgov.historical_study_type != StudyType.INTERVENTIONAL:
        return False
    if ctgov.historical_arm_count < 2 or ctgov.historical_biological_intervention_count < 2:
        return False
    if not _supported_pre_results_history(ctgov):
        return False
    release_dates = _required_assay_first_observed_dates(immport, set(spec.required_assay_methods))
    decision_date = ctgov.historical_posted_date
    if not release_dates or decision_date is None:
        return False
    minimum_release_date = decision_date + timedelta(days=spec.thresholds.minimum_outcome_delay_days)
    return any(minimum_release_date <= release_date <= spec.outcome_as_of for release_date in release_dates)


def _make_report(
    spec: FeasibilityInventorySpec,
    immport_receipt: SnapshotReceipt,
    ctgov_receipt: SnapshotReceipt,
    immport_records: tuple[ImmportDiscoveryRecord, ...],
    ctgov_records: tuple[CtgovHistoricalRecord, ...],
    linked_records: tuple[LinkedStudyInventoryRecord, ...],
) -> FeasibilityInventoryReport:
    ctgov_by_id = {record.nct_id: record for record in ctgov_records}
    exact = tuple(record for record in immport_records if len(record.explicit_nct_ids) == 1)
    matched_source = tuple(record for record in exact if record.explicit_nct_ids[0] in ctgov_by_id)
    matched_nct_ids = {record.explicit_nct_ids[0] for record in matched_source}
    duplicate_mapping_count = sum(
        1 for count in Counter(record.explicit_nct_ids[0] for record in exact).values() if count > 1
    )
    linked_by_study = {record.study_accession: record for record in linked_records}
    exact_nct_counts = Counter(record.explicit_nct_ids[0] for record in exact)
    duplicate_nct_ids = {nct_id for nct_id, count in exact_nct_counts.items() if count > 1}
    technically_supported_nct_ids = {
        record.explicit_nct_ids[0]
        for record in matched_source
        if linked_by_study[record.study_accession].supported_pre_results_history
    }
    supported_candidate_source = tuple(
        record
        for record in matched_source
        if _eligible_supported_temporal_candidate(
            record,
            ctgov_by_id[record.explicit_nct_ids[0]],
            linked_by_study[record.study_accession],
            duplicate_nct_ids=duplicate_nct_ids,
            spec=spec,
        )
    )
    screened_source = tuple(
        record
        for record in matched_source
        if record.clinical_trial
        and record.human
        and record.arm_count >= 2
        and linked_by_study[record.study_accession].assay_metadata_present
        and ctgov_by_id[record.explicit_nct_ids[0]].historical_study_type == StudyType.INTERVENTIONAL
        and ctgov_by_id[record.explicit_nct_ids[0]].historical_arm_count >= 2
        and _pre_results_history(ctgov_by_id[record.explicit_nct_ids[0]])
    )
    metrics = InventoryMetrics(
        immport_study_count=len(immport_records),
        immport_clinical_trial_count=sum(record.clinical_trial for record in immport_records),
        immport_human_count=sum(record.human for record in immport_records),
        immport_multi_arm_count=sum(record.arm_count >= 2 for record in immport_records),
        immport_assay_metadata_count=sum(bool(record.assay_methods) for record in immport_records),
        exact_link_study_count=len(exact),
        exact_link_unique_nct_count=len({record.explicit_nct_ids[0] for record in exact}),
        missing_link_count=sum(not record.explicit_nct_ids for record in immport_records),
        ambiguous_link_count=sum(len(record.explicit_nct_ids) > 1 for record in immport_records),
        matched_study_count=len(matched_source),
        matched_unique_nct_count=len(matched_nct_ids),
        missing_ctgov_record_count=sum(record.link_status == LinkStatus.RECORD_NOT_FOUND for record in linked_records),
        matched_interventional_count=sum(
            ctgov_by_id[nct_id].historical_study_type == StudyType.INTERVENTIONAL for nct_id in matched_nct_ids
        ),
        matched_pre_results_history_count=sum(_pre_results_history(ctgov_by_id[nct_id]) for nct_id in matched_nct_ids),
        matched_pre_results_multi_arm_count=sum(
            _pre_results_history(ctgov_by_id[nct_id]) and ctgov_by_id[nct_id].historical_arm_count >= 2
            for nct_id in matched_nct_ids
        ),
        matched_historical_post_date_actual_count=sum(
            ctgov_by_id[nct_id].historical_posted_date_type == PostedDateType.ACTUAL for nct_id in matched_nct_ids
        ),
        matched_historical_post_date_estimated_count=sum(
            ctgov_by_id[nct_id].historical_posted_date_type == PostedDateType.ESTIMATED for nct_id in matched_nct_ids
        ),
        matched_supported_pre_results_history_count=len(technically_supported_nct_ids),
        matched_supported_pre_results_multi_arm_count=len(
            {record.explicit_nct_ids[0] for record in supported_candidate_source}
        ),
        matched_assay_metadata_count=sum(
            linked_by_study[record.study_accession].assay_metadata_present for record in matched_source
        ),
        matched_assay_metadata_temporal_count=sum(
            linked_by_study[record.study_accession].assay_metadata_present
            and bool(_required_assay_first_observed_dates(record, set(spec.required_assay_methods)))
            for record in matched_source
        ),
        matched_arm_count_agreement_count=sum(
            linked_by_study[record.study_accession].arm_count_agreement is True for record in matched_source
        ),
        screened_study_count=len(screened_source),
        screened_unique_nct_count=len({record.explicit_nct_ids[0] for record in screened_source}),
        matched_current_results_unique_nct_count=sum(
            ctgov_by_id[nct_id].current_has_results for nct_id in matched_nct_ids
        ),
        matched_current_results_post_date_actual_count=sum(
            ctgov_by_id[nct_id].current_results_first_post_date_type == PostedDateType.ACTUAL
            for nct_id in matched_nct_ids
        ),
        matched_current_results_post_date_estimated_count=sum(
            ctgov_by_id[nct_id].current_results_first_post_date_type == PostedDateType.ESTIMATED
            for nct_id in matched_nct_ids
        ),
        duplicate_nct_mapping_count=duplicate_mapping_count,
    )
    gates = _gates(spec, immport_receipt, ctgov_receipt, metrics)
    return FeasibilityInventoryReport(
        inventory_id=spec.inventory_id,
        spec_sha256=_model_sha256(spec),
        immport_receipt_sha256=_model_sha256(immport_receipt),
        ctgov_receipt_sha256=_model_sha256(ctgov_receipt),
        records_sha256=records_sha256(linked_records),
        record_count=len(linked_records),
        metrics=metrics,
        gates=gates,
        admission_tier=_admission_tier(gates),
    )


def _gates(
    spec: FeasibilityInventorySpec,
    immport_receipt: SnapshotReceipt,
    ctgov_receipt: SnapshotReceipt,
    metrics: InventoryMetrics,
) -> tuple[GateResult, ...]:
    thresholds = spec.thresholds
    results = (
        GateResult(
            gate_id='assay_temporal_provenance',
            status=(
                GateStatus.PASS
                if spec.synthetic
                and metrics.matched_assay_metadata_temporal_count >= thresholds.minimum_assay_metadata_matches
                else GateStatus.NOT_ASSESSED
            ),
            detail=(
                f'{metrics.matched_assay_metadata_temporal_count} matched ImmPort studies declare a '
                f'first-observed date for at least one accepted assay method; threshold is '
                f'{thresholds.minimum_assay_metadata_matches}. Real data cannot pass until the offline audit '
                f'verifies the underlying release-diff artifacts and archive coverage.'
            ),
        ),
        GateResult(
            gate_id='arm_mapping',
            status=GateStatus.NOT_ASSESSED,
            detail='Arm-count agreement is only a screen; authenticated candidate-to-arm mapping has not run.',
        ),
        GateResult(
            gate_id='exact_linkage',
            status=(
                GateStatus.PASS
                if metrics.matched_unique_nct_count >= thresholds.minimum_exact_unique_nct
                else GateStatus.FAIL
            ),
            detail=(
                f'{metrics.matched_unique_nct_count} unique NCT records matched; '
                f'threshold is {thresholds.minimum_exact_unique_nct}.'
            ),
        ),
        GateResult(
            gate_id='immune_assay_metadata',
            status=(
                GateStatus.PASS
                if metrics.matched_assay_metadata_count >= thresholds.minimum_assay_metadata_matches
                else GateStatus.FAIL
            ),
            detail=(
                f'{metrics.matched_assay_metadata_count} matched ImmPort studies expose a required assay method; '
                f'threshold is {thresholds.minimum_assay_metadata_matches}.'
            ),
        ),
        GateResult(
            gate_id='outcome_comparability',
            status=GateStatus.NOT_ASSESSED,
            detail='No subject-level assay values, units, antigens, timepoints, or per-arm coverage were read.',
        ),
        GateResult(
            gate_id='redistribution',
            status=(
                GateStatus.PASS
                if immport_receipt.redistribution_cleared and ctgov_receipt.redistribution_cleared
                else GateStatus.NOT_ASSESSED
            ),
            detail=(
                'Both receipts are cleared for redistribution.'
                if immport_receipt.redistribution_cleared and ctgov_receipt.redistribution_cleared
                else 'Raw or record-level redistribution remains subject to source-specific review.'
            ),
        ),
        GateResult(
            gate_id='normalized_snapshot_integrity',
            status=GateStatus.PASS,
            detail='Both offline record files match their committed receipt hashes, byte counts, and row counts.',
        ),
        GateResult(
            gate_id='raw_capture_verification',
            status=GateStatus.NOT_ASSESSED,
            detail=(
                'Receipts commit raw captures, but this inventory does not contain or re-verify those raw artifacts.'
            ),
        ),
        GateResult(
            gate_id='supported_historical_freeze',
            status=(
                GateStatus.FAIL
                if metrics.matched_supported_pre_results_history_count == 0
                else (
                    GateStatus.PASS
                    if spec.synthetic
                    and metrics.matched_supported_pre_results_multi_arm_count
                    >= thresholds.minimum_supported_pre_results_multi_arm
                    else GateStatus.NOT_ASSESSED
                )
            ),
            detail=(
                f'{metrics.matched_supported_pre_results_multi_arm_count} unique curator-confirmed, nonduplicate '
                f'human clinical trials have a supported, actual-date, pre-results multi-arm version and a later '
                f'first-observed required ImmPort assay release; threshold is '
                f'{thresholds.minimum_supported_pre_results_multi_arm}. Real data cannot pass this combined gate '
                f'until release-diff provenance is audited.'
            ),
        ),
        GateResult(
            gate_id='task_stage_fit',
            status=GateStatus.NOT_ASSESSED,
            detail=(
                'Clinical trial arms require a new early-clinical task contract; they cannot be '
                'silently admitted to the preclinical task.'
            ),
        ),
        GateResult(
            gate_id='vaccine_relevance',
            status=GateStatus.NOT_ASSESSED,
            detail='Source research-focus filtering is not a curator adjudication of vaccine-candidate relevance.',
        ),
    )
    return tuple(sorted(results, key=lambda gate: gate.gate_id))


def _admission_tier(gates: tuple[GateResult, ...]) -> AdmissionTier:
    by_id = {gate.gate_id: gate.status for gate in gates}
    if by_id['normalized_snapshot_integrity'] != GateStatus.PASS:
        raise FeasibilityIntegrityError('normalized snapshot integrity must pass before a report can be built')
    tier = AdmissionTier.SOURCE_VALID
    if by_id['exact_linkage'] == GateStatus.PASS:
        tier = AdmissionTier.LINKABLE
    if (
        tier == AdmissionTier.LINKABLE
        and by_id['immune_assay_metadata'] == GateStatus.PASS
        and by_id['assay_temporal_provenance'] == GateStatus.PASS
        and by_id['supported_historical_freeze'] == GateStatus.PASS
    ):
        tier = AdmissionTier.TEMPORALLY_REPLAYABLE
    if (
        tier == AdmissionTier.TEMPORALLY_REPLAYABLE
        and by_id['arm_mapping'] == GateStatus.PASS
        and by_id['outcome_comparability'] == GateStatus.PASS
        and by_id['task_stage_fit'] == GateStatus.PASS
    ):
        tier = AdmissionTier.LABEL_PILOT_READY
    if (
        tier == AdmissionTier.LABEL_PILOT_READY
        and by_id['redistribution'] == GateStatus.PASS
        and by_id['vaccine_relevance'] == GateStatus.PASS
    ):
        tier = AdmissionTier.BENCHMARK_READY
    return tier


def _write_inventory_output(
    root: Path,
    spec: FeasibilityInventorySpec,
    immport_receipt: SnapshotReceipt,
    ctgov_receipt: SnapshotReceipt,
    immport_records: tuple[ImmportDiscoveryRecord, ...],
    ctgov_records: tuple[CtgovHistoricalRecord, ...],
    records: tuple[LinkedStudyInventoryRecord, ...],
    report: FeasibilityInventoryReport,
) -> None:
    if root.exists():
        raise FeasibilityIntegrityError(f'inventory output already exists: {root}')
    root.mkdir(parents=True)
    (root / SPEC_FILENAME).write_bytes(canonical_json_bytes(spec) + b'\n')
    (root / 'immport_receipt.json').write_bytes(canonical_json_bytes(immport_receipt) + b'\n')
    (root / 'ctgov_receipt.json').write_bytes(canonical_json_bytes(ctgov_receipt) + b'\n')
    (root / IMMPORT_SOURCE_RECORDS_FILENAME).write_text(jsonl_text(immport_records), encoding='utf-8')
    (root / CTGOV_SOURCE_RECORDS_FILENAME).write_text(jsonl_text(ctgov_records), encoding='utf-8')
    (root / RECORDS_FILENAME).write_text(jsonl_text(records), encoding='utf-8')
    (root / REPORT_FILENAME).write_bytes(canonical_json_bytes(report) + b'\n')
