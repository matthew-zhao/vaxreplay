"""Reproducible two-phase construction of real Lane A model-facing workspaces.

``prepare`` opens only decision-time material.  It reconstructs the externally pinned lineage
split, creates the identity-scrubbed contexts, and freezes the exact target set that the private
gold adapter must later use.  ``finalize`` accepts only a complete cohort derivation bound to that
target set and split; it never accepts a loose collection of labels.

Every artifact produced here is organizer-private or development-only.  In particular, successful
construction does not claim leaderboard admission, sealed execution, or identity-contamination
control.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_gold_adapter import (
    ExecutionGoldCohortTarget,
    ExecutionGoldCohortTargetSet,
    ExecutionGoldDerivationError,
    load_execution_gold_cohort_derivation,
)
from vaxreplay.clinicaltrials.execution_inventory import ExecutionInventoryError, audit_execution_inventory
from vaxreplay.clinicaltrials.execution_merge import AactExecutionMultiAnchorMergeReceipt
from vaxreplay.clinicaltrials.execution_schema import ExecutionCohortInventory
from vaxreplay.clinicaltrials.execution_workspace import (
    ExecutionWorkspaceContextPlan,
    ExecutionWorkspaceCount,
    ExecutionWorkspaceError,
    LoadedExecutionWorkspaceBuild,
    _build_execution_workspace_context_plan,
    verify_execution_workspace_build,
    write_execution_workspace_build,
)
from vaxreplay.clinicaltrials.lineage_split import (
    LineageCaseAssignment,
    LineageSplitBuild,
    LineageSplitError,
    read_private_id_key,
    verify_lineage_split_build,
)
from vaxreplay.clinicaltrials.relevance_adjudication import (
    VaccineRelevanceAdjudicationSet,
    VaccineRelevanceReviewQueue,
    VaccineRelevanceReviewReceipt,
)

EXECUTION_WORKSPACE_PREPARATION_SCHEMA_VERSION = 'vaxreplay.clinical-execution-workspace-prepare.dev-v0.1'
EXECUTION_WORKSPACE_PREPARATION_BUILDER_ID = 'aact-lane-a-two-phase-workspace-pipeline-v0.1'
_PREPARE_RECEIPT_NAME = 'PREPARE-RECEIPT.json'
_CONTEXT_PLAN_PATH = 'organizer/context-plan.json'
_TARGET_SET_PATH = 'organizer/cohort-targets.json'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class ExecutionWorkspacePipelineError(ValueError):
    """A two-phase Lane A build failed closed."""


class ExecutionWorkspacePreparationArtifact(StrictModel):
    relative_path: str = Field(min_length=1, max_length=4_096)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)
    mode: Literal['0600'] = '0600'

    @model_validator(mode='after')
    def validate_path(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != self.relative_path:
            raise ValueError('preparation artifact path must be normalized and relative')
        return self


class ExecutionWorkspacePreparedCaseBinding(StrictModel):
    organizer_private_nct_id: str = Field(pattern=r'^NCT\d{8}$')
    episode_id: str = Field(min_length=1)
    context_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_case_assignment_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_group_id: str = Field(min_length=1)
    split: Split


class ExecutionWorkspacePreparationUpstream(StrictModel):
    merge_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    merged_inventory_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_queue_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_adjudication_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_split_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_split_assignments_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_split_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_id_key_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)


class ExecutionWorkspacePreparationReceipt(StrictModel):
    schema_version: Literal['vaxreplay.clinical-execution-workspace-prepare.dev-v0.1'] = (
        EXECUTION_WORKSPACE_PREPARATION_SCHEMA_VERSION
    )
    builder_id: Literal['aact-lane-a-two-phase-workspace-pipeline-v0.1'] = EXECUTION_WORKSPACE_PREPARATION_BUILDER_ID
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    merge_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    merged_inventory_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_queue_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_adjudication_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_split_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_split_assignments_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_split_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage_id_key_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_key_id: str = Field(pattern=_SHA256_PATTERN)
    inventory_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_queue_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_adjudications_model_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_count: int = Field(gt=0)
    lineage_count: int = Field(gt=0)
    split_counts: tuple[ExecutionWorkspaceCount, ...]
    case_bindings: tuple[ExecutionWorkspacePreparedCaseBinding, ...] = Field(min_length=1)
    artifacts: tuple[ExecutionWorkspacePreparationArtifact, ...] = Field(min_length=2, max_length=2)
    decision_only_sources_read: Literal[True] = True
    execution_labels_read: Literal[False] = False
    outcome_conditioned_selection_or_split: Literal[False] = False
    final_workspace_contexts_bound: Literal[True] = True
    organizer_private: Literal[True] = True
    development_only: Literal[True] = True
    public_workspace_created: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        paths = tuple(item.relative_path for item in self.artifacts)
        if paths != tuple(sorted((_CONTEXT_PLAN_PATH, _TARGET_SET_PATH))):
            raise ValueError('preparation receipt must bind exactly the context plan and target set')
        case_keys = tuple(item.organizer_private_nct_id for item in self.case_bindings)
        if case_keys != tuple(sorted(set(case_keys))) or len(case_keys) != self.task_count:
            raise ValueError('preparation case bindings must exactly cover unique ascending tasks')
        if self.lineage_count != len({item.lineage_group_id for item in self.case_bindings}):
            raise ValueError('preparation lineage count does not match case bindings')
        observed_counts: dict[str, int] = {}
        for binding in self.case_bindings:
            observed_counts[binding.split.value] = observed_counts.get(binding.split.value, 0) + 1
        if {item.name: item.count for item in self.split_counts} != dict(sorted(observed_counts.items())):
            raise ValueError('preparation split counts do not match case bindings')
        return self


@dataclass(frozen=True)
class ExecutionWorkspacePreparation:
    root: Path
    receipt: ExecutionWorkspacePreparationReceipt
    receipt_sha256: str
    plan: ExecutionWorkspaceContextPlan
    targets: ExecutionGoldCohortTargetSet


@dataclass(frozen=True)
class ExecutionWorkspaceFinalization:
    build: LoadedExecutionWorkspaceBuild
    receipt_sha256: str
    externally_pinned_receipt_verified: bool


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_sha256(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _validate_digest(value: str, label: str) -> None:
    if re.fullmatch(_SHA256_PATTERN, value) is None:
        raise ExecutionWorkspacePipelineError(f'{label} must be a 64-character SHA-256 digest')


def _stable_stat(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_regular(path: Path, *, label: str, expected_mode: int | None = None) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ExecutionWorkspacePipelineError(f'{label} cannot be a symbolic link')
    resolved = expanded.resolve()
    try:
        before = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise ExecutionWorkspacePipelineError(f'cannot stat {label}: {error}') from error
    if not stat.S_ISREG(before.st_mode):
        raise ExecutionWorkspacePipelineError(f'{label} must be a regular file')
    if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
        raise ExecutionWorkspacePipelineError(f'{label} must have mode {expected_mode:04o}')
    try:
        payload = resolved.read_bytes()
        after = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise ExecutionWorkspacePipelineError(f'cannot read {label}: {error}') from error
    if _stable_stat(before) != _stable_stat(after):
        raise ExecutionWorkspacePipelineError(f'{label} changed while it was being read')
    return payload


def read_workspace_secret_key(path: Path, *, purpose: str) -> bytes:
    """Read a distinct raw workspace key, requiring exact owner-only permissions."""

    key = _read_regular(path, label=f'{purpose} key', expected_mode=0o600)
    if len(key) < 32:
        raise ExecutionWorkspacePipelineError(f'{purpose} key must contain at least 32 bytes')
    return key


def _pinned_model[T: BaseModel](
    path: Path,
    *,
    expected_sha256: str,
    model: type[T],
    label: str,
) -> tuple[T, bytes]:
    _validate_digest(expected_sha256, f'expected {label} SHA-256')
    payload = _read_regular(path, label=label)
    if not hmac.compare_digest(_sha256(payload), expected_sha256):
        raise ExecutionWorkspacePipelineError(f'{label} does not match its external SHA-256 pin')
    try:
        return model.model_validate_json(payload), payload
    except ValueError as error:
        raise ExecutionWorkspacePipelineError(f'invalid {label}: {error}') from error


def _bound_artifact(
    *,
    root: Path,
    artifacts: Sequence[object],
    relative_path: str,
    label: str,
) -> tuple[bytes, str]:
    matches = tuple(item for item in artifacts if getattr(item, 'relative_path') == relative_path)
    if len(matches) != 1:
        raise ExecutionWorkspacePipelineError(f'{label} receipt does not bind exactly {relative_path}')
    artifact = matches[0]
    payload = _read_regular(root / relative_path, label=f'{label} artifact {relative_path}')
    if len(payload) != getattr(artifact, 'byte_count') or not hmac.compare_digest(
        _sha256(payload), getattr(artifact, 'sha256')
    ):
        raise ExecutionWorkspacePipelineError(f'{label} artifact does not match its receipt: {relative_path}')
    return payload, getattr(artifact, 'sha256')


def _load_decision_only_sources(
    *,
    merge_root: Path,
    expected_merge_receipt_sha256: str,
    relevance_root: Path,
    expected_relevance_receipt_sha256: str,
) -> tuple[
    ExecutionCohortInventory,
    VaccineRelevanceReviewQueue,
    VaccineRelevanceAdjudicationSet,
    AactExecutionMultiAnchorMergeReceipt,
    VaccineRelevanceReviewReceipt,
    str,
    str,
    str,
]:
    merge_root = merge_root.expanduser()
    relevance_root = relevance_root.expanduser()
    if merge_root.is_symlink() or relevance_root.is_symlink():
        raise ExecutionWorkspacePipelineError('upstream build roots cannot be symbolic links')
    merge_root = merge_root.resolve()
    relevance_root = relevance_root.resolve()
    merge_receipt, _ = _pinned_model(
        merge_root / 'MERGE-RECEIPT.json',
        expected_sha256=expected_merge_receipt_sha256,
        model=AactExecutionMultiAnchorMergeReceipt,
        label='merge receipt',
    )
    review_receipt, _ = _pinned_model(
        relevance_root / 'REVIEW-RECEIPT.json',
        expected_sha256=expected_relevance_receipt_sha256,
        model=VaccineRelevanceReviewReceipt,
        label='relevance-review receipt',
    )
    inventory_payload, inventory_artifact_sha256 = _bound_artifact(
        root=merge_root,
        artifacts=merge_receipt.artifacts,
        relative_path='organizer/cohort-inventory.json',
        label='merge',
    )
    queue_payload, queue_artifact_sha256 = _bound_artifact(
        root=relevance_root,
        artifacts=review_receipt.artifacts,
        relative_path='organizer/relevance-review-queue.json',
        label='relevance',
    )
    adjudication_payload, adjudication_artifact_sha256 = _bound_artifact(
        root=relevance_root,
        artifacts=review_receipt.artifacts,
        relative_path='organizer/relevance-adjudications.json',
        label='relevance',
    )
    try:
        inventory = ExecutionCohortInventory.model_validate_json(inventory_payload)
        queue = VaccineRelevanceReviewQueue.model_validate_json(queue_payload)
        adjudications = VaccineRelevanceAdjudicationSet.model_validate_json(adjudication_payload)
        audit_execution_inventory(inventory)
    except (ValueError, ExecutionInventoryError) as error:
        raise ExecutionWorkspacePipelineError(f'invalid decision-only source material: {error}') from error
    return (
        inventory,
        queue,
        adjudications,
        merge_receipt,
        review_receipt,
        inventory_artifact_sha256,
        queue_artifact_sha256,
        adjudication_artifact_sha256,
    )


def _write_private_json(root: Path, relative_path: str, value: object) -> ExecutionWorkspacePreparationArtifact:
    payload = canonical_json_bytes(value)
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return ExecutionWorkspacePreparationArtifact(
        relative_path=relative_path,
        sha256=_sha256(payload),
        byte_count=len(payload),
    )


def _write_preparation(
    *,
    plan: ExecutionWorkspaceContextPlan,
    targets: ExecutionGoldCohortTargetSet,
    case_bindings: Sequence[ExecutionWorkspacePreparedCaseBinding],
    upstream: ExecutionWorkspacePreparationUpstream,
    output_root: Path,
) -> ExecutionWorkspacePreparation:
    destination = output_root.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'workspace preparation already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.staging-', dir=destination.parent))
    staging.chmod(0o700)
    try:
        artifacts = tuple(
            sorted(
                (
                    _write_private_json(staging, _CONTEXT_PLAN_PATH, plan),
                    _write_private_json(staging, _TARGET_SET_PATH, targets),
                ),
                key=lambda item: item.relative_path,
            )
        )
        receipt = ExecutionWorkspacePreparationReceipt(
            cohort_id=targets.cohort_id,
            alias_key_id=plan.alias_key_id,
            inventory_model_sha256=plan.inventory_sha256,
            relevance_queue_model_sha256=plan.relevance_queue_sha256,
            relevance_adjudications_model_sha256=plan.relevance_adjudications_sha256,
            context_plan_sha256=_model_sha256(plan),
            target_set_sha256=_model_sha256(targets),
            task_count=plan.task_count,
            lineage_count=plan.lineage_count,
            split_counts=plan.split_counts,
            case_bindings=tuple(sorted(case_bindings, key=lambda item: item.organizer_private_nct_id)),
            artifacts=artifacts,
            merge_receipt_sha256=upstream.merge_receipt_sha256,
            merged_inventory_artifact_sha256=upstream.merged_inventory_artifact_sha256,
            relevance_review_receipt_sha256=upstream.relevance_review_receipt_sha256,
            relevance_queue_artifact_sha256=upstream.relevance_queue_artifact_sha256,
            relevance_adjudication_artifact_sha256=upstream.relevance_adjudication_artifact_sha256,
            lineage_split_receipt_sha256=upstream.lineage_split_receipt_sha256,
            lineage_split_assignments_sha256=upstream.lineage_split_assignments_sha256,
            lineage_split_policy_sha256=upstream.lineage_split_policy_sha256,
            lineage_id_key_commitment_sha256=upstream.lineage_id_key_commitment_sha256,
        )
        _write_private_json(staging, _PREPARE_RECEIPT_NAME, receipt)
        fsync_directory(staging / 'organizer')
        fsync_directory(staging)
        rename_directory_noreplace(staging, destination)
        fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt_sha256 = _sha256((destination / _PREPARE_RECEIPT_NAME).read_bytes())
    return verify_execution_workspace_preparation(
        destination,
        expected_receipt_sha256=receipt_sha256,
    )


def prepare_execution_workspace(
    *,
    merge_root: Path,
    expected_merge_receipt_sha256: str,
    relevance_root: Path,
    expected_relevance_receipt_sha256: str,
    lineage_root: Path,
    expected_lineage_receipt_sha256: str,
    lineage_id_key: bytes,
    alias_key: bytes,
    cohort_id: str,
    output_root: Path,
) -> ExecutionWorkspacePreparation:
    """Freeze final contexts and gold targets from externally pinned decision-only inputs."""

    if hmac.compare_digest(lineage_id_key, alias_key):
        raise ExecutionWorkspacePipelineError('lineage-ID and workspace-alias keys must be distinct')
    try:
        lineage: LineageSplitBuild = verify_lineage_split_build(
            lineage_root,
            expected_receipt_sha256=expected_lineage_receipt_sha256,
            merge_root=merge_root,
            expected_merge_receipt_sha256=expected_merge_receipt_sha256,
            relevance_root=relevance_root,
            expected_relevance_receipt_sha256=expected_relevance_receipt_sha256,
            id_key=lineage_id_key,
        )
    except LineageSplitError as error:
        raise ExecutionWorkspacePipelineError(f'lineage split verification failed: {error}') from error
    (
        inventory,
        queue,
        adjudications,
        merge_receipt,
        review_receipt,
        inventory_artifact_sha256,
        queue_artifact_sha256,
        adjudication_artifact_sha256,
    ) = _load_decision_only_sources(
        merge_root=merge_root,
        expected_merge_receipt_sha256=expected_merge_receipt_sha256,
        relevance_root=relevance_root,
        expected_relevance_receipt_sha256=expected_relevance_receipt_sha256,
    )
    assignment_artifact = next(
        (
            item
            for item in lineage.receipt.artifacts
            if item.relative_path == 'organizer/lineage-split-assignments.json'
        ),
        None,
    )
    if assignment_artifact is None:
        raise ExecutionWorkspacePipelineError('lineage receipt has no assignment artifact')
    upstream_fields = (
        lineage.receipt.merge_receipt_sha256,
        lineage.receipt.merged_inventory_artifact_sha256,
        lineage.receipt.relevance_review_receipt_sha256,
        lineage.receipt.relevance_queue_artifact_sha256,
        lineage.receipt.relevance_adjudication_artifact_sha256,
    )
    expected_upstream_fields = (
        expected_merge_receipt_sha256,
        inventory_artifact_sha256,
        expected_relevance_receipt_sha256,
        queue_artifact_sha256,
        adjudication_artifact_sha256,
    )
    if upstream_fields != expected_upstream_fields:
        raise ExecutionWorkspacePipelineError('lineage split is not bound to the exact pinned decision-only inputs')
    if merge_receipt.synthetic or review_receipt.execution_labels_read:
        raise ExecutionWorkspacePipelineError('workspace preparation requires real, outcome-blind source artifacts')

    cases_by_nct: dict[str, LineageCaseAssignment] = {}
    for case in lineage.assignments.cases:
        if case.nct_id in cases_by_nct:
            raise ExecutionWorkspacePipelineError(f'duplicate lineage case for {case.nct_id}')
        cases_by_nct[case.nct_id] = case
    plan = _build_execution_workspace_context_plan(
        inventory=inventory,
        relevance_queue=queue,
        relevance_adjudications=adjudications,
        trusted_relevance_review_receipt_sha256=expected_relevance_receipt_sha256,
        split_manifest_sha256=assignment_artifact.sha256,
        split_by_nct={nct_id: case.split for nct_id, case in cases_by_nct.items()},
        lineage_by_nct={nct_id: case.lineage_group_id for nct_id, case in cases_by_nct.items()},
        alias_key=alias_key,
    )
    targets = ExecutionGoldCohortTargetSet(
        cohort_id=cohort_id,
        targets=tuple(
            sorted(
                (
                    ExecutionGoldCohortTarget(
                        organizer_private_nct_id=entry.organizer_private_nct_id,
                        context=entry.context,
                    )
                    for entry in plan.entries
                ),
                key=lambda item: (item.context.anchor_date, item.organizer_private_nct_id),
            )
        ),
        final_workspace_contexts_bound=True,
    )
    entries_by_nct = {entry.organizer_private_nct_id: entry for entry in plan.entries}
    case_bindings = tuple(
        ExecutionWorkspacePreparedCaseBinding(
            organizer_private_nct_id=nct_id,
            episode_id=entries_by_nct[nct_id].context.episode_id,
            context_sha256=entries_by_nct[nct_id].context_sha256,
            lineage_case_assignment_sha256=_model_sha256(case),
            lineage_group_id=case.lineage_group_id,
            split=case.split,
        )
        for nct_id, case in sorted(cases_by_nct.items())
    )
    return _write_preparation(
        plan=plan,
        targets=targets,
        case_bindings=case_bindings,
        upstream=ExecutionWorkspacePreparationUpstream(
            merge_receipt_sha256=expected_merge_receipt_sha256,
            merged_inventory_artifact_sha256=inventory_artifact_sha256,
            relevance_review_receipt_sha256=expected_relevance_receipt_sha256,
            relevance_queue_artifact_sha256=queue_artifact_sha256,
            relevance_adjudication_artifact_sha256=adjudication_artifact_sha256,
            lineage_split_receipt_sha256=expected_lineage_receipt_sha256,
            lineage_split_assignments_sha256=assignment_artifact.sha256,
            lineage_split_policy_sha256=lineage.receipt.policy_sha256,
            lineage_id_key_commitment_sha256=lineage.receipt.id_key_commitment_sha256,
        ),
        output_root=output_root,
    )


def verify_execution_workspace_preparation(
    root: Path,
    *,
    expected_receipt_sha256: str,
) -> ExecutionWorkspacePreparation:
    """Verify an exact prepared tree against an externally stored receipt digest."""

    _validate_digest(expected_receipt_sha256, 'expected preparation receipt SHA-256')
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ExecutionWorkspacePipelineError('workspace preparation root cannot be a symbolic link')
    resolved = supplied.resolve()
    if not resolved.is_dir() or stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise ExecutionWorkspacePipelineError('workspace preparation root must be a mode-0700 directory')
    expected_files = {_PREPARE_RECEIPT_NAME, _CONTEXT_PLAN_PATH, _TARGET_SET_PATH}
    expected_directories = {'organizer'}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in resolved.rglob('*'):
        relative = path.relative_to(resolved).as_posix()
        if path.is_symlink():
            raise ExecutionWorkspacePipelineError(f'workspace preparation contains a symbolic link: {relative}')
        if path.is_dir():
            if stat.S_IMODE(path.stat().st_mode) != 0o700:
                raise ExecutionWorkspacePipelineError(f'workspace preparation directory is not mode 0700: {relative}')
            observed_directories.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise ExecutionWorkspacePipelineError(f'workspace preparation contains a non-regular entry: {relative}')
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ExecutionWorkspacePipelineError('workspace preparation contains missing or uncommitted paths')
    receipt_payload = _read_regular(
        resolved / _PREPARE_RECEIPT_NAME,
        label='workspace preparation receipt',
        expected_mode=0o600,
    )
    if not hmac.compare_digest(_sha256(receipt_payload), expected_receipt_sha256):
        raise ExecutionWorkspacePipelineError('workspace preparation receipt does not match its external pin')
    try:
        receipt = ExecutionWorkspacePreparationReceipt.model_validate_json(receipt_payload)
    except ValueError as error:
        raise ExecutionWorkspacePipelineError(f'invalid workspace preparation receipt: {error}') from error
    if receipt_payload != canonical_json_bytes(receipt):
        raise ExecutionWorkspacePipelineError('workspace preparation receipt is not canonical JSON')
    payloads: dict[str, bytes] = {}
    for artifact in receipt.artifacts:
        payload = _read_regular(
            resolved / artifact.relative_path,
            label=f'workspace preparation artifact {artifact.relative_path}',
            expected_mode=0o600,
        )
        if len(payload) != artifact.byte_count or not hmac.compare_digest(_sha256(payload), artifact.sha256):
            raise ExecutionWorkspacePipelineError(
                f'workspace preparation artifact does not match receipt: {artifact.relative_path}'
            )
        payloads[artifact.relative_path] = payload
    try:
        plan = ExecutionWorkspaceContextPlan.model_validate_json(payloads[_CONTEXT_PLAN_PATH])
        targets = ExecutionGoldCohortTargetSet.model_validate_json(payloads[_TARGET_SET_PATH])
    except ValueError as error:
        raise ExecutionWorkspacePipelineError(f'invalid prepared workspace model: {error}') from error
    if payloads[_CONTEXT_PLAN_PATH] != canonical_json_bytes(plan) or payloads[_TARGET_SET_PATH] != canonical_json_bytes(
        targets
    ):
        raise ExecutionWorkspacePipelineError('prepared workspace models must use canonical JSON')
    if (
        receipt.context_plan_sha256 != _model_sha256(plan)
        or receipt.target_set_sha256 != _model_sha256(targets)
        or receipt.cohort_id != targets.cohort_id
        or receipt.alias_key_id != plan.alias_key_id
        or receipt.inventory_model_sha256 != plan.inventory_sha256
        or receipt.relevance_queue_model_sha256 != plan.relevance_queue_sha256
        or receipt.relevance_adjudications_model_sha256 != plan.relevance_adjudications_sha256
        or receipt.relevance_review_receipt_sha256 != plan.trusted_relevance_review_receipt_sha256
        or receipt.lineage_split_assignments_sha256 != plan.split_manifest_sha256
        or receipt.task_count != plan.task_count
        or receipt.lineage_count != plan.lineage_count
        or receipt.split_counts != plan.split_counts
    ):
        raise ExecutionWorkspacePipelineError('preparation receipt does not bind its exact context plan and targets')
    entries_by_nct = {entry.organizer_private_nct_id: entry for entry in plan.entries}
    targets_by_nct = {target.organizer_private_nct_id: target for target in targets.targets}
    bindings_by_nct = {binding.organizer_private_nct_id: binding for binding in receipt.case_bindings}
    if set(entries_by_nct) != set(targets_by_nct) or set(entries_by_nct) != set(bindings_by_nct):
        raise ExecutionWorkspacePipelineError('prepared plan, target set, and case bindings do not exactly align')
    for nct_id, entry in entries_by_nct.items():
        target = targets_by_nct[nct_id]
        binding = bindings_by_nct[nct_id]
        if (
            target.context != entry.context
            or binding.episode_id != entry.context.episode_id
            or binding.context_sha256 != entry.context_sha256
            or binding.lineage_group_id != entry.lineage_group_id
            or binding.split != entry.split
        ):
            raise ExecutionWorkspacePipelineError(f'prepared case does not align for {nct_id}')
    if (
        not targets.final_workspace_contexts_bound
        or not all(target.context.cutoff_documents for target in targets.targets)
        or plan.outcome_or_label_data_read
        or plan.public_tasks_created
        or plan.leaderboard_admitted
    ):
        raise ExecutionWorkspacePipelineError('prepared workspace weakens final-context or release-safety flags')
    return ExecutionWorkspacePreparation(
        root=resolved,
        receipt=receipt,
        receipt_sha256=expected_receipt_sha256,
        plan=plan,
        targets=targets,
    )


def finalize_execution_workspace(
    *,
    preparation_root: Path,
    expected_preparation_receipt_sha256: str,
    gold_derivation_root: Path,
    expected_gold_derivation_receipt_sha256: str,
    private_gold_master_key: bytes,
    output_root: Path,
    expected_final_receipt_sha256: str | None = None,
) -> ExecutionWorkspaceFinalization:
    """Finalize only from a split-bound derivation of the exact prepared target set."""

    prepared = verify_execution_workspace_preparation(
        preparation_root,
        expected_receipt_sha256=expected_preparation_receipt_sha256,
    )
    try:
        gold_build = load_execution_gold_cohort_derivation(
            gold_derivation_root,
            expected_receipt_sha256=expected_gold_derivation_receipt_sha256,
        )
    except ExecutionGoldDerivationError as error:
        raise ExecutionWorkspacePipelineError(f'cohort-gold derivation verification failed: {error}') from error
    derivation = gold_build.derivation
    receipt = derivation.receipt
    if canonical_json_bytes(derivation.targets) != canonical_json_bytes(prepared.targets):
        raise ExecutionWorkspacePipelineError('cohort-gold derivation does not bind the exact prepared target set')
    split_binding = receipt.split_binding
    if (
        not receipt.final_workspace_contexts_bound
        or not receipt.split_inventory_bound
        or not receipt.lineage_split_safe
        or split_binding is None
        or split_binding.split_receipt_sha256 != prepared.receipt.lineage_split_receipt_sha256
        or split_binding.split_assignments_sha256 != prepared.receipt.lineage_split_assignments_sha256
        or split_binding.split_policy_sha256 != prepared.receipt.lineage_split_policy_sha256
        or split_binding.id_key_commitment_sha256 != prepared.receipt.lineage_id_key_commitment_sha256
    ):
        raise ExecutionWorkspacePipelineError('cohort-gold derivation is not bound to the prepared finalized split')
    prepared_cases = {item.organizer_private_nct_id: item for item in prepared.receipt.case_bindings}
    gold_cases = {item.organizer_private_nct_id: item for item in receipt.case_receipts}
    if set(prepared_cases) != set(gold_cases):
        raise ExecutionWorkspacePipelineError('cohort-gold cases do not exactly cover the prepared cohort')
    for nct_id, binding in prepared_cases.items():
        case = gold_cases[nct_id]
        if (
            case.episode_id != binding.episode_id
            or case.task_context_sha256 != binding.context_sha256
            or case.lineage_case_assignment_sha256 != binding.lineage_case_assignment_sha256
            or case.lineage_group_id != binding.lineage_group_id
            or case.split != binding.split
        ):
            raise ExecutionWorkspacePipelineError(f'cohort-gold split/context binding disagrees for {nct_id}')
    gold_by_nct = {item.organizer_private_nct_id: item for item in derivation.private_gold.records}
    try:
        build = write_execution_workspace_build(
            plan=prepared.plan,
            gold_by_nct=gold_by_nct,
            private_gold_master_key=private_gold_master_key,
            output_root=output_root,
        )
    except (ExecutionWorkspaceError, ValueError) as error:
        raise ExecutionWorkspacePipelineError(f'workspace finalization failed: {error}') from error
    observed_receipt_sha256 = _sha256((build.root / 'BUILD-RECEIPT.json').read_bytes())
    if expected_final_receipt_sha256 is not None:
        _validate_digest(expected_final_receipt_sha256, 'expected final receipt SHA-256')
        if not hmac.compare_digest(observed_receipt_sha256, expected_final_receipt_sha256):
            raise ExecutionWorkspacePipelineError('final workspace receipt does not match its external pin')
    verification_pin = expected_final_receipt_sha256 or observed_receipt_sha256
    verified = verify_execution_workspace_build(build.root, expected_receipt_sha256=verification_pin)
    return ExecutionWorkspaceFinalization(
        build=verified,
        receipt_sha256=observed_receipt_sha256,
        externally_pinned_receipt_verified=expected_final_receipt_sha256 is not None,
    )


def _summary(value: ExecutionWorkspacePreparation | ExecutionWorkspaceFinalization) -> dict[str, object]:
    if isinstance(value, ExecutionWorkspacePreparation):
        return {
            'phase': 'prepare',
            'root': str(value.root),
            'receipt_sha256': value.receipt_sha256,
            'task_count': value.receipt.task_count,
            'lineage_count': value.receipt.lineage_count,
            'split_counts': {item.name: item.count for item in value.receipt.split_counts},
            'final_workspace_contexts_bound': value.receipt.final_workspace_contexts_bound,
            'leaderboard_admitted': value.receipt.leaderboard_admitted,
        }
    return {
        'phase': 'finalize',
        'root': str(value.build.root),
        'receipt_sha256': value.receipt_sha256,
        'task_count': value.build.receipt.task_count,
        'externally_pinned_receipt_verified': value.externally_pinned_receipt_verified,
        'leaderboard_admitted': value.build.receipt.leaderboard_admitted,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--merge-root', type=Path, required=True)
    prepare.add_argument('--expected-merge-receipt-sha256', required=True)
    prepare.add_argument('--relevance-root', type=Path, required=True)
    prepare.add_argument('--expected-relevance-receipt-sha256', required=True)
    prepare.add_argument('--lineage-root', type=Path, required=True)
    prepare.add_argument('--expected-lineage-receipt-sha256', required=True)
    prepare.add_argument('--lineage-id-key-file', type=Path, required=True)
    prepare.add_argument('--alias-key-file', type=Path, required=True)
    prepare.add_argument('--cohort-id', required=True)
    prepare.add_argument('--output-root', type=Path, required=True)

    verify_prepare = commands.add_parser('verify-prepare')
    verify_prepare.add_argument('--root', type=Path, required=True)
    verify_prepare.add_argument('--expected-receipt-sha256', required=True)

    finalize = commands.add_parser('finalize')
    finalize.add_argument('--preparation-root', type=Path, required=True)
    finalize.add_argument('--expected-preparation-receipt-sha256', required=True)
    finalize.add_argument('--gold-derivation-root', type=Path, required=True)
    finalize.add_argument('--expected-gold-derivation-receipt-sha256', required=True)
    finalize.add_argument('--private-gold-master-key-file', type=Path, required=True)
    finalize.add_argument('--output-root', type=Path, required=True)
    finalize.add_argument('--expected-final-receipt-sha256')

    verify_final = commands.add_parser('verify-final')
    verify_final.add_argument('--root', type=Path, required=True)
    verify_final.add_argument('--expected-receipt-sha256', required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == 'prepare':
        value: ExecutionWorkspacePreparation | ExecutionWorkspaceFinalization = prepare_execution_workspace(
            merge_root=arguments.merge_root,
            expected_merge_receipt_sha256=arguments.expected_merge_receipt_sha256,
            relevance_root=arguments.relevance_root,
            expected_relevance_receipt_sha256=arguments.expected_relevance_receipt_sha256,
            lineage_root=arguments.lineage_root,
            expected_lineage_receipt_sha256=arguments.expected_lineage_receipt_sha256,
            lineage_id_key=read_private_id_key(arguments.lineage_id_key_file),
            alias_key=read_workspace_secret_key(arguments.alias_key_file, purpose='workspace alias'),
            cohort_id=arguments.cohort_id,
            output_root=arguments.output_root,
        )
    elif arguments.command == 'verify-prepare':
        value = verify_execution_workspace_preparation(
            arguments.root,
            expected_receipt_sha256=arguments.expected_receipt_sha256,
        )
    elif arguments.command == 'finalize':
        value = finalize_execution_workspace(
            preparation_root=arguments.preparation_root,
            expected_preparation_receipt_sha256=arguments.expected_preparation_receipt_sha256,
            gold_derivation_root=arguments.gold_derivation_root,
            expected_gold_derivation_receipt_sha256=arguments.expected_gold_derivation_receipt_sha256,
            private_gold_master_key=read_workspace_secret_key(
                arguments.private_gold_master_key_file,
                purpose='private-gold master',
            ),
            output_root=arguments.output_root,
            expected_final_receipt_sha256=arguments.expected_final_receipt_sha256,
        )
    else:
        build = verify_execution_workspace_build(
            arguments.root,
            expected_receipt_sha256=arguments.expected_receipt_sha256,
        )
        value = ExecutionWorkspaceFinalization(
            build=build,
            receipt_sha256=arguments.expected_receipt_sha256,
            externally_pinned_receipt_verified=True,
        )
    print(json.dumps(_summary(value), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = [
    'EXECUTION_WORKSPACE_PREPARATION_BUILDER_ID',
    'EXECUTION_WORKSPACE_PREPARATION_SCHEMA_VERSION',
    'ExecutionWorkspaceFinalization',
    'ExecutionWorkspacePipelineError',
    'ExecutionWorkspacePreparation',
    'ExecutionWorkspacePreparationReceipt',
    'finalize_execution_workspace',
    'main',
    'prepare_execution_workspace',
    'read_workspace_secret_key',
    'verify_execution_workspace_preparation',
]
