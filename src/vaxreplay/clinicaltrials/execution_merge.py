"""Deterministically merge provisional single-anchor AACT execution inventories.

The merger preserves every decision-time row, uses the earliest normalized observation to stabilize
identity/classification fields for an NCT across anchors, and delegates earliest-eligible assignment
to the audited execution-inventory contract.  It never promotes provisional inputs into a scored,
adjudicated, or split-safe cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_adapter import (
    AactExecutionBuildReceipt,
    ArtifactReceipt,
)
from vaxreplay.clinicaltrials.execution_inventory import (
    ExecutionInventoryError,
    audit_execution_inventory,
    audit_execution_label_set,
    bind_anchor_source,
    build_execution_inventory,
    derive_execution_labels,
)
from vaxreplay.clinicaltrials.execution_schema import (
    EXECUTION_TASK_ID,
    EXECUTION_TASK_SEMANTICS,
    AactExecutionDecisionRow,
    AactExecutionOutcomeRow,
    DiseaseStratum,
    ExecutionCohortInventory,
    ExecutionCohortPolicy,
    ExecutionLabelSet,
    ExecutionTaskSemantics,
)

MERGE_SCHEMA_VERSION = 'vaxreplay.aact-execution-multi-anchor-merge.v0.1'
MERGER_ID = 'aact-provisional-execution-earliest-anchor-merge-v0.1'
_STABLE_FIELDS = (
    'biological_intervention_count',
    'disease_stratum',
    'human',
    'infectious_disease_vaccine',
    'lineage_group_id',
    'prophylactic_intent',
)


class AactExecutionMergeError(ValueError):
    """A source build or deterministic multi-anchor reconstruction failed closed."""


class SourceExecutionBuildReceipt(StrictModel):
    anchor_date: date
    label_archive_date: date
    source_mode: Literal['trusted_official_real', 'synthetic_test_only']
    synthetic: bool
    build_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    inventory_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_set_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_archive_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    label_archive_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    screened_record_count: int = Field(ge=0)
    mechanical_assignment_count: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_mode(self) -> Self:
        if self.synthetic != (self.source_mode == 'synthetic_test_only'):
            raise ValueError('source synthetic flag and source mode disagree')
        return self


class StableFieldRemap(StrictModel):
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    authoritative_anchor_date: date
    remapped_anchor_date: date
    changed_fields: tuple[str, ...] = Field(min_length=1)
    input_row_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    output_row_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_remap(self) -> Self:
        if self.authoritative_anchor_date >= self.remapped_anchor_date:
            raise ValueError('a stable-field remap must flow from an earlier anchor')
        if self.changed_fields != tuple(sorted(set(self.changed_fields))):
            raise ValueError('changed fields must be unique and sorted')
        if set(self.changed_fields) - set(_STABLE_FIELDS):
            raise ValueError('a remap changed a field outside the stable pre-cutoff classification contract')
        if self.input_row_sha256 == self.output_row_sha256:
            raise ValueError('a remap must change the normalized row')
        return self


class StableFieldRemapSet(StrictModel):
    schema_version: Literal['vaxreplay.aact-execution-stability-remaps.v0.1'] = (
        'vaxreplay.aact-execution-stability-remaps.v0.1'
    )
    stable_fields: tuple[str, ...] = _STABLE_FIELDS
    earliest_observation_authoritative: Literal[True] = True
    remaps: tuple[StableFieldRemap, ...]

    @model_validator(mode='after')
    def validate_set(self) -> Self:
        if self.stable_fields != tuple(sorted(set(self.stable_fields))):
            raise ValueError('stable fields must be unique and sorted')
        keys = tuple((item.nct_id, item.remapped_anchor_date) for item in self.remaps)
        if keys != tuple(sorted(set(keys))):
            raise ValueError('stability remaps must be unique and sorted')
        return self


class AnchorAssignmentCount(StrictModel):
    anchor_date: date
    label_archive_date: date
    assignment_count: int = Field(ge=0)
    covid_assignment_count: int = Field(ge=0)
    non_covid_assignment_count: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        if self.covid_assignment_count + self.non_covid_assignment_count != self.assignment_count:
            raise ValueError('anchor disease-stratum counts do not add up')
        return self


class AactExecutionMultiAnchorMergeReceipt(StrictModel):
    schema_version: Literal['vaxreplay.aact-execution-multi-anchor-merge.v0.1'] = MERGE_SCHEMA_VERSION
    merger_id: Literal['aact-provisional-execution-earliest-anchor-merge-v0.1'] = MERGER_ID
    task_id: Literal['registry_observed_trial_execution'] = EXECUTION_TASK_ID
    task_semantics: ExecutionTaskSemantics = EXECUTION_TASK_SEMANTICS
    synthetic: bool
    release_status: Literal['provisional_high_recall_inventory', 'synthetic_test_only']
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    biological_efficacy_claimed: Literal[False] = False
    mechanical_eligibility_only: Literal[True] = True
    active_vaccination_adjudication_bound: Literal[False] = False
    scored_cohort_eligible: Literal[False] = False
    manual_lineage_review_required: Literal[True] = True
    lineage_split_safe: Literal[False] = False
    earliest_eligible_anchor_required: Literal[True] = True
    stable_identity_uses_earliest_observation: Literal[True] = True
    source_builds: tuple[SourceExecutionBuildReceipt, ...] = Field(min_length=2)
    stability_remap_count: int = Field(ge=0)
    stability_remaps_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_row_count: int = Field(ge=0)
    mechanical_assignment_count: int = Field(ge=0)
    unique_nct_assignment_count: int = Field(ge=0)
    anchor_assignment_counts: tuple[AnchorAssignmentCount, ...] = Field(min_length=2)
    missing_label_record_count: int = Field(ge=0)
    failed_status_count: int = Field(ge=0)
    artifacts: tuple[ArtifactReceipt, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        source_dates = tuple(item.anchor_date for item in self.source_builds)
        if source_dates != tuple(sorted(set(source_dates))):
            raise ValueError('source builds must have unique ascending anchors')
        modes = {item.source_mode for item in self.source_builds}
        if len(modes) != 1:
            raise ValueError('trusted and synthetic source builds cannot be mixed')
        expected_synthetic = modes == {'synthetic_test_only'}
        expected_status = 'synthetic_test_only' if expected_synthetic else 'provisional_high_recall_inventory'
        if self.synthetic != expected_synthetic or self.release_status != expected_status:
            raise ValueError('merged release labeling does not match its source mode')
        count_dates = tuple(item.anchor_date for item in self.anchor_assignment_counts)
        if count_dates != source_dates:
            raise ValueError('assignment counts must cover every source anchor in order')
        if sum(item.assignment_count for item in self.anchor_assignment_counts) != self.mechanical_assignment_count:
            raise ValueError('per-anchor assignment counts do not add up')
        if self.unique_nct_assignment_count != self.mechanical_assignment_count:
            raise ValueError('earliest-anchor output must assign each NCT exactly once')
        paths = tuple(item.relative_path for item in self.artifacts)
        if paths != tuple(sorted(set(paths))):
            raise ValueError('artifact receipts must be unique and sorted')
        return self


@dataclass(frozen=True)
class AactExecutionMergeBuild:
    root: Path
    receipt: AactExecutionMultiAnchorMergeReceipt
    inventory: ExecutionCohortInventory
    labels: ExecutionLabelSet


@dataclass(frozen=True)
class _LoadedSource:
    receipt: AactExecutionBuildReceipt
    inventory: ExecutionCohortInventory
    labels: ExecutionLabelSet
    source_receipt: SourceExecutionBuildReceipt


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise AactExecutionMergeError(f'expected a regular non-symlink source artifact: {path}')
    try:
        return path.read_bytes()
    except OSError as error:
        raise AactExecutionMergeError(f'cannot read source artifact {path}: {error}') from error


def _verify_artifact(root: Path, artifact: ArtifactReceipt) -> None:
    payload = _read_regular(root / artifact.relative_path)
    if len(payload) != artifact.byte_count or _sha256_bytes(payload) != artifact.sha256:
        raise AactExecutionMergeError(f'source artifact does not match its build receipt: {artifact.relative_path}')


def _load_source(root: Path) -> _LoadedSource:
    expanded = root.expanduser()
    if expanded.is_symlink():
        raise AactExecutionMergeError(f'source build cannot be a symbolic link: {expanded}')
    root = expanded.resolve()
    if not root.is_dir():
        raise AactExecutionMergeError(f'source build must be a directory: {root}')
    build_receipt_payload = _read_regular(root / 'BUILD-RECEIPT.json')
    try:
        receipt = AactExecutionBuildReceipt.model_validate_json(build_receipt_payload)
        inventory_payload = _read_regular(root / 'organizer' / 'cohort-inventory.json')
        label_payload = _read_regular(root / 'private' / 'execution-labels.json')
        inventory = ExecutionCohortInventory.model_validate_json(inventory_payload)
        labels = ExecutionLabelSet.model_validate_json(label_payload)
    except ValueError as error:
        raise AactExecutionMergeError(f'invalid source build contract in {root}: {error}') from error
    for artifact in receipt.artifacts:
        _verify_artifact(root, artifact)
    try:
        audit_execution_inventory(inventory)
        audit_execution_label_set(inventory=inventory, label_set=labels)
    except ExecutionInventoryError as error:
        raise AactExecutionMergeError(f'source build fails deterministic audit: {root}: {error}') from error
    if len(inventory.policy.anchors) != 1:
        raise AactExecutionMergeError('multi-anchor merger accepts only audited single-anchor source builds')
    binding = inventory.policy.anchors[0]
    if receipt.assigned_trial_count != len(inventory.assignments) or labels.assigned_trial_count != len(
        inventory.assignments
    ):
        raise AactExecutionMergeError('source build assignment counts are inconsistent')
    if (
        receipt.scored_cohort_eligible
        or receipt.active_vaccination_adjudication_bound
        or not receipt.manual_lineage_review_required
        or receipt.lineage_split_safe
    ):
        raise AactExecutionMergeError('source build weakens the provisional/manual-review safety state')
    source_receipt = SourceExecutionBuildReceipt(
        anchor_date=binding.anchor_date,
        label_archive_date=binding.label_archive_date,
        source_mode=receipt.source_binding.mode,
        synthetic=receipt.synthetic,
        build_receipt_sha256=_sha256_bytes(build_receipt_payload),
        inventory_sha256=_sha256_bytes(inventory_payload),
        label_set_sha256=_sha256_bytes(label_payload),
        decision_archive_sha256=receipt.decision_archive.archive_sha256,
        label_archive_sha256=receipt.label_archive.archive_sha256,
        screened_record_count=receipt.screened_record_count,
        mechanical_assignment_count=len(inventory.assignments),
    )
    return _LoadedSource(receipt=receipt, inventory=inventory, labels=labels, source_receipt=source_receipt)


def _stable_rows(
    sources: Sequence[_LoadedSource],
) -> tuple[tuple[AactExecutionDecisionRow, ...], StableFieldRemapSet]:
    rows = tuple(row for source in sources for row in source.inventory.decision_rows)
    keys = tuple((row.archive_date, row.nct_id) for row in rows)
    if len(keys) != len(set(keys)):
        raise AactExecutionMergeError('source builds contain a duplicate anchor/NCT decision row')
    by_nct: dict[str, list[AactExecutionDecisionRow]] = defaultdict(list)
    for row in rows:
        by_nct[row.nct_id].append(row)
    output: list[AactExecutionDecisionRow] = []
    remaps: list[StableFieldRemap] = []
    for nct_id in sorted(by_nct):
        nct_rows = sorted(by_nct[nct_id], key=lambda item: item.archive_date)
        authoritative = nct_rows[0]
        stable_values = {field: getattr(authoritative, field) for field in _STABLE_FIELDS}
        for row in nct_rows:
            changed_fields = tuple(
                sorted(field for field, value in stable_values.items() if getattr(row, field) != value)
            )
            normalized = row.model_copy(update=stable_values) if changed_fields else row
            output.append(normalized)
            if changed_fields:
                remaps.append(
                    StableFieldRemap(
                        nct_id=nct_id,
                        authoritative_anchor_date=authoritative.archive_date,
                        remapped_anchor_date=row.archive_date,
                        changed_fields=changed_fields,
                        input_row_sha256=_sha256_bytes(canonical_json_bytes(row)),
                        output_row_sha256=_sha256_bytes(canonical_json_bytes(normalized)),
                    )
                )
    ordered_rows = tuple(sorted(output, key=lambda item: (item.archive_date, item.nct_id)))
    return ordered_rows, StableFieldRemapSet(remaps=tuple(remaps))


def _combined_policy(
    sources: Sequence[_LoadedSource],
    rows: Sequence[AactExecutionDecisionRow],
) -> ExecutionCohortPolicy:
    first_policy = sources[0].inventory.policy
    for source in sources[1:]:
        policy = source.inventory.policy
        comparable = (
            policy.selection_universe_rule_id,
            policy.selection_universe_rule_sha256,
            policy.lineage_grouping_rule_id,
            policy.lineage_grouping_rule_sha256,
            policy.eligibility_rule_id,
            policy.label_rule_id,
        )
        expected = (
            first_policy.selection_universe_rule_id,
            first_policy.selection_universe_rule_sha256,
            first_policy.lineage_grouping_rule_id,
            first_policy.lineage_grouping_rule_sha256,
            first_policy.eligibility_rule_id,
            first_policy.label_rule_id,
        )
        if comparable != expected:
            raise AactExecutionMergeError('source builds use different selection, lineage, eligibility, or label rules')
    rows_by_anchor: dict[date, list[AactExecutionDecisionRow]] = defaultdict(list)
    for row in rows:
        rows_by_anchor[row.archive_date].append(row)
    bindings = []
    for source in sources:
        old = source.inventory.policy.anchors[0]
        bindings.append(
            bind_anchor_source(
                anchor_date=old.anchor_date,
                decision_snapshot_id=old.decision_snapshot_id,
                decision_archive_manifest_sha256=old.decision_archive_manifest_sha256,
                label_snapshot_id=old.label_snapshot_id,
                label_archive_manifest_sha256=old.label_archive_manifest_sha256,
                rows=rows_by_anchor[old.anchor_date],
            )
        )
    anchor_key = '-'.join(binding.anchor_date.isoformat() for binding in bindings)
    return ExecutionCohortPolicy(
        policy_id=f'aact-execution-multi-anchor-{anchor_key}-provisional-v0.1',
        synthetic=first_policy.synthetic,
        selection_universe_rule_id=first_policy.selection_universe_rule_id,
        selection_universe_rule_sha256=first_policy.selection_universe_rule_sha256,
        lineage_grouping_rule_id=first_policy.lineage_grouping_rule_id,
        lineage_grouping_rule_sha256=first_policy.lineage_grouping_rule_sha256,
        anchors=tuple(bindings),
    )


def _selected_outcomes(
    inventory: ExecutionCohortInventory,
    sources: Sequence[_LoadedSource],
) -> tuple[AactExecutionOutcomeRow, ...]:
    by_key: dict[tuple[str, str], AactExecutionOutcomeRow] = {}
    for source in sources:
        for outcome in source.labels.outcome_rows:
            key = (outcome.snapshot_id, outcome.nct_id)
            if key in by_key:
                raise AactExecutionMergeError('source builds contain duplicate label-snapshot/NCT outcome rows')
            by_key[key] = outcome
    selected: list[AactExecutionOutcomeRow] = []
    for assignment in inventory.assignments:
        key = (assignment.label_snapshot_id, assignment.nct_id)
        outcome = by_key.get(key)
        if outcome is None:
            raise AactExecutionMergeError(
                f'earliest-anchor assignment lacks its bound later observation: {assignment.nct_id}'
            )
        selected.append(outcome)
    return tuple(selected)


def _write_exact(path: Path, value: object, *, relative_path: str) -> ArtifactReceipt:
    payload = canonical_json_bytes(value) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return ArtifactReceipt(
        relative_path=relative_path,
        sha256=_sha256_bytes(payload),
        byte_count=len(payload),
    )


def merge_aact_execution_builds(
    *,
    source_roots: Sequence[Path],
    output_root: Path,
) -> AactExecutionMergeBuild:
    """Merge two or more immutable provisional single-anchor builds."""

    if len(source_roots) < 2:
        raise AactExecutionMergeError('at least two single-anchor source builds are required')
    destination = output_root.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'immutable multi-anchor output already exists: {destination}')
    sources = tuple(
        sorted((_load_source(root) for root in source_roots), key=lambda item: item.source_receipt.anchor_date)
    )
    source_dates = tuple(source.source_receipt.anchor_date for source in sources)
    if len(source_dates) != len(set(source_dates)):
        raise AactExecutionMergeError('source builds must have unique decision anchors')
    modes = {source.source_receipt.source_mode for source in sources}
    if len(modes) != 1:
        raise AactExecutionMergeError('trusted real and synthetic test builds cannot be merged')
    synthetic = modes == {'synthetic_test_only'}
    if any(source.inventory.policy.synthetic != synthetic for source in sources):
        raise AactExecutionMergeError('source policy synthetic flags disagree with source binding modes')

    rows, remap_set = _stable_rows(sources)
    policy = _combined_policy(sources, rows)
    inventory = build_execution_inventory(policy=policy, decision_rows=rows)
    audit_execution_inventory(inventory)
    labels = derive_execution_labels(inventory=inventory, outcome_rows=_selected_outcomes(inventory, sources))
    audit_execution_label_set(inventory=inventory, label_set=labels)

    assignment_counts = tuple(
        AnchorAssignmentCount(
            anchor_date=source.source_receipt.anchor_date,
            label_archive_date=source.source_receipt.label_archive_date,
            assignment_count=len(anchor_assignments),
            covid_assignment_count=sum(item.disease_stratum == DiseaseStratum.COVID_19 for item in anchor_assignments),
            non_covid_assignment_count=sum(
                item.disease_stratum == DiseaseStratum.NON_COVID_INFECTIOUS for item in anchor_assignments
            ),
        )
        for source in sources
        for anchor_assignments in (
            tuple(
                assignment
                for assignment in inventory.assignments
                if assignment.anchor_date == source.source_receipt.anchor_date
            ),
        )
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.staging-', dir=destination.parent))
    staging.chmod(0o700)
    try:
        artifacts = [
            _write_exact(
                staging / 'organizer' / 'cohort-inventory.json',
                inventory,
                relative_path='organizer/cohort-inventory.json',
            ),
            _write_exact(
                staging / 'organizer' / 'stability-remaps.json',
                remap_set,
                relative_path='organizer/stability-remaps.json',
            ),
            _write_exact(
                staging / 'private' / 'execution-labels.json',
                labels,
                relative_path='private/execution-labels.json',
            ),
        ]
        receipt = AactExecutionMultiAnchorMergeReceipt(
            synthetic=synthetic,
            release_status='synthetic_test_only' if synthetic else 'provisional_high_recall_inventory',
            source_builds=tuple(source.source_receipt for source in sources),
            stability_remap_count=len(remap_set.remaps),
            stability_remaps_sha256=_sha256_bytes(canonical_json_bytes(remap_set)),
            decision_row_count=len(inventory.decision_rows),
            mechanical_assignment_count=len(inventory.assignments),
            unique_nct_assignment_count=len({assignment.nct_id for assignment in inventory.assignments}),
            anchor_assignment_counts=assignment_counts,
            missing_label_record_count=labels.missing_record_count,
            failed_status_count=labels.failed_status_count,
            artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
        )
        _write_exact(staging / 'MERGE-RECEIPT.json', receipt, relative_path='MERGE-RECEIPT.json')
        for directory in sorted(
            (path for path in staging.rglob('*') if path.is_dir()), key=lambda path: len(path.parts), reverse=True
        ):
            fsync_directory(directory)
        fsync_directory(staging)
        rename_directory_noreplace(staging, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return AactExecutionMergeBuild(root=destination, receipt=receipt, inventory=inventory, labels=labels)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', action='append', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build = merge_aact_execution_builds(source_roots=args.source_root, output_root=args.output_root)
    print(canonical_json_bytes(build.receipt).decode('utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
