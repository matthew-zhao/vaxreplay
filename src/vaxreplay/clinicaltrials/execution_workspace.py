"""Identity-scrubbed model-facing workspaces for real Lane A execution tasks.

The builder deliberately exposes a conservative structured projection of decision-time AACT data.
It omits dedicated product-name, sponsor-name, title, acronym, summary, and description fields.
Categorical condition text is retained and can itself contain an identifying name, so the build
does not claim that all product/sponsor strings are removed or that a model cannot reidentify a
trial from public facts or memorized training data.

Public tasks, organizer mappings, private gold, and HMAC keys are written into separate trees.  All
release/admission flags remain false.  A future admitted release must add source-span proofs,
contamination admission, sealed runner qualification, and an externally authenticated build pin.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_gold_adapter import execution_forecast_spec_policy
from vaxreplay.clinicaltrials.execution_inventory import ExecutionInventoryError, audit_execution_inventory
from vaxreplay.clinicaltrials.execution_schema import (
    EXECUTION_TASK_ID,
    AactExecutionDecisionRow,
    ExecutionCohortInventory,
    NormalizedPhase,
    RegistryStatus,
    TrialAnchorAssignment,
)
from vaxreplay.clinicaltrials.execution_task import (
    CutoffDocument,
    ExecutionPrivateGold,
    ExecutionTask,
    ExecutionTaskContext,
    build_execution_task,
    execution_task_context_sha256,
    validate_execution_task_gold,
)
from vaxreplay.clinicaltrials.relevance_adjudication import (
    DecisionEvidenceRecord,
    RelevanceDisposition,
    VaccineRelevanceAdjudicationSet,
    VaccineRelevanceReviewQueue,
    finalize_relevance_adjudications,
)

EXECUTION_WORKSPACE_PLAN_SCHEMA_VERSION = 'vaxreplay.clinical-execution-workspace-plan.dev-v0.1'
EXECUTION_WORKSPACE_BUILD_SCHEMA_VERSION = 'vaxreplay.clinical-execution-workspace-build.dev-v0.1'
EXECUTION_WORKSPACE_TRIAL_VIEW_SCHEMA_VERSION = 'vaxreplay.clinical-execution-trial-view.dev-v0.1'
EXECUTION_WORKSPACE_ALIAS_SCHEME = 'hmac-sha256-secret-per-task-permutation-v0.1'
EXECUTION_WORKSPACE_BUILDER_ID = 'aact-structured-identity-scrubbed-workspace-v0.1'
_ALIAS_KEY_DOMAIN = b'vaxreplay.execution-workspace-alias-key-id.v0.1\x00'
_EPISODE_ALIAS_DOMAIN = b'vaxreplay.execution-workspace-episode.v0.1\x00'
_REFERENCE_ORDER_DOMAIN = b'vaxreplay.execution-workspace-reference-order.v0.1\x00'
_LINEAGE_ALIAS_DOMAIN = b'vaxreplay.execution-workspace-lineage.v0.1\x00'
_GOLD_KEY_DOMAIN = b'vaxreplay.execution-workspace-gold-key.v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_NCT_PATTERN = re.compile(r'NCT\d{8}', re.IGNORECASE)
_MAX_PUBLIC_ARTIFACT_BYTES = 16 * 1024 * 1024


class ExecutionWorkspaceError(ValueError):
    """A Lane A context plan or model-facing package failed closed."""


class ExecutionWorkspaceArtifactRole(str, enum.Enum):
    PUBLIC = 'public'
    ORGANIZER = 'organizer'
    PRIVATE = 'private'


class ExecutionWorkspaceCount(StrictModel):
    name: str = Field(min_length=1, max_length=256)
    count: int = Field(ge=0)


class ExecutionWorkspaceAliasBinding(StrictModel):
    """Organizer-private mapping for one trial on one task surface."""

    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    public_trial_id: str = Field(pattern=r'^trial-(?:target|reference-[0-9]{3})$')
    source_anchor_date: date
    decision_source_record_sha256: str = Field(pattern=_SHA256_PATTERN)


class ExecutionWorkspaceTrialView(StrictModel):
    """Conservative model-visible projection; no raw identity or free-text fields exist."""

    schema_version: Literal['vaxreplay.clinical-execution-trial-view.dev-v0.1'] = (
        EXECUTION_WORKSPACE_TRIAL_VIEW_SCHEMA_VERSION
    )
    public_trial_id: str = Field(pattern=r'^trial-(?:target|reference-[0-9]{3})$')
    source_anchor_date: date
    study_first_posted_date: date
    phase: NormalizedPhase
    decision_status: RegistryStatus
    planned_enrollment: int | None = Field(default=None, ge=0)
    planned_enrollment_type: str | None = None
    planned_primary_completion_date: date | None = None
    planned_primary_completion_date_type: str | None = None
    biological_intervention_count: int = Field(ge=0)
    intervention_type_counts: tuple[ExecutionWorkspaceCount, ...]
    sponsor_class_counts: tuple[ExecutionWorkspaceCount, ...]
    primary_purposes: tuple[str, ...]
    conditions: tuple[str, ...]
    results_section_present: bool
    identity_fields_removed: Literal[True] = True
    free_text_removed: Literal[True] = True

    @model_validator(mode='after')
    def validate_public_projection(self) -> Self:
        if (self.planned_enrollment is None) != (self.planned_enrollment_type is None):
            raise ValueError('planned enrollment and type must be present together')
        if (self.planned_primary_completion_date is None) != (self.planned_primary_completion_date_type is None):
            raise ValueError('planned primary-completion date and type must be present together')
        for values, label in (
            (self.intervention_type_counts, 'intervention types'),
            (self.sponsor_class_counts, 'sponsor classes'),
        ):
            names = tuple(item.name for item in values)
            if names != tuple(sorted(set(names))):
                raise ValueError(f'{label} must use unique canonical order')
        for values, label in ((self.primary_purposes, 'primary purposes'), (self.conditions, 'conditions')):
            if values != tuple(sorted(set(values))):
                raise ValueError(f'{label} must use unique canonical order')
        if _NCT_PATTERN.search(canonical_json_bytes(self).decode('utf-8')):
            raise ValueError('public trial projections cannot contain an NCT identifier')
        return self


class ExecutionWorkspacePlanEntry(StrictModel):
    """Organizer-private final context and exact alias map for one model-facing task."""

    organizer_private_nct_id: str = Field(pattern=r'^NCT\d{8}$')
    lineage_group_id: str = Field(min_length=1)
    public_lineage_id: str = Field(pattern=r'^lineage-[0-9a-f]{20}$')
    split: Split
    context: ExecutionTaskContext
    context_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_source_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_bindings: tuple[ExecutionWorkspaceAliasBinding, ...] = Field(min_length=1)
    reference_trial_count: int = Field(ge=0)
    public_source_bytes: int = Field(gt=0)
    literal_registry_identifiers_removed: Literal[True] = True
    product_and_sponsor_names_removed: Literal[False] = False
    raw_intervention_and_sponsor_name_fields_omitted: Literal[True] = True
    free_text_removed: Literal[True] = True
    identity_contamination_controlled: Literal[False] = False

    @model_validator(mode='after')
    def validate_bindings(self) -> Self:
        if execution_task_context_sha256(self.context) != self.context_sha256:
            raise ValueError('workspace context does not match context_sha256')
        if self.context.episode_id == self.organizer_private_nct_id or _NCT_PATTERN.search(self.context.episode_id):
            raise ValueError('workspace episode identity cannot expose the private registry ID')
        public_ids = tuple(item.public_trial_id for item in self.alias_bindings)
        private_ids = tuple(item.nct_id for item in self.alias_bindings)
        if len(public_ids) != len(set(public_ids)) or len(private_ids) != len(set(private_ids)):
            raise ValueError('workspace alias bindings must be bijective')
        if public_ids.count('trial-target') != 1 or self.reference_trial_count != len(public_ids) - 1:
            raise ValueError('workspace alias bindings require one target and the declared references')
        target = next(item for item in self.alias_bindings if item.public_trial_id == 'trial-target')
        if target.nct_id != self.organizer_private_nct_id:
            raise ValueError('workspace target alias does not map to its private target')
        if self.context.target_trial_id != 'trial-target':
            raise ValueError('workspace context must expose only the neutral target alias')
        if sum(len(item.body.encode('utf-8')) for item in self.context.cutoff_documents) != self.public_source_bytes:
            raise ValueError('workspace public source byte count is inconsistent')
        return self


class ExecutionWorkspaceContextPlan(StrictModel):
    schema_version: Literal['vaxreplay.clinical-execution-workspace-plan.dev-v0.1'] = (
        EXECUTION_WORKSPACE_PLAN_SCHEMA_VERSION
    )
    builder_id: Literal['aact-structured-identity-scrubbed-workspace-v0.1'] = EXECUTION_WORKSPACE_BUILDER_ID
    task_type: Literal['registry_observed_trial_execution'] = EXECUTION_TASK_ID
    inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_queue_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_adjudications_sha256: str = Field(pattern=_SHA256_PATTERN)
    trusted_relevance_review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_scheme: Literal['hmac-sha256-secret-per-task-permutation-v0.1'] = EXECUTION_WORKSPACE_ALIAS_SCHEME
    alias_key_id: str = Field(pattern=_SHA256_PATTERN)
    entries: tuple[ExecutionWorkspacePlanEntry, ...] = Field(min_length=1)
    task_count: int = Field(gt=0)
    split_counts: tuple[ExecutionWorkspaceCount, ...]
    lineage_count: int = Field(gt=0)
    outcome_or_label_data_read: Literal[False] = False
    external_receipt_pins_authenticated: Literal[False] = False
    lineage_split_source_authenticated: Literal[False] = False
    organizer_private: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    lineage_split_isolated: Literal[True] = True
    public_tasks_created: Literal[False] = False
    source_span_mapping_complete: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False

    @model_validator(mode='after')
    def validate_plan(self) -> Self:
        if self.task_count != len(self.entries):
            raise ValueError('workspace plan task_count does not match entries')
        nct_ids = tuple(item.organizer_private_nct_id for item in self.entries)
        episode_ids = tuple(item.context.episode_id for item in self.entries)
        if nct_ids != tuple(sorted(set(nct_ids))):
            raise ValueError('workspace plan entries must use unique ascending private IDs')
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('workspace plan episode IDs must be unique')
        expected_counts = Counter(item.split.value for item in self.entries)
        observed_counts = {item.name: item.count for item in self.split_counts}
        if observed_counts != dict(sorted(expected_counts.items())):
            raise ValueError('workspace plan split counts are inconsistent')
        lineage_splits: dict[str, set[Split]] = {}
        for entry in self.entries:
            lineage_splits.setdefault(entry.lineage_group_id, set()).add(entry.split)
        if any(len(splits) != 1 for splits in lineage_splits.values()):
            raise ValueError('one lineage cannot cross workspace splits')
        if self.lineage_count != len(lineage_splits):
            raise ValueError('workspace plan lineage_count is inconsistent')
        return self


class ExecutionWorkspaceArtifactBinding(StrictModel):
    relative_path: str = Field(min_length=1, max_length=4_096)
    role: ExecutionWorkspaceArtifactRole
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)
    mode: Literal['0444', '0600']

    @field_validator('relative_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
            raise ValueError('workspace artifact paths must be normalized and relative')
        return value


class ExecutionWorkspaceBuildReceipt(StrictModel):
    schema_version: Literal['vaxreplay.clinical-execution-workspace-build.dev-v0.1'] = (
        EXECUTION_WORKSPACE_BUILD_SCHEMA_VERSION
    )
    builder_id: Literal['aact-structured-identity-scrubbed-workspace-v0.1'] = EXECUTION_WORKSPACE_BUILDER_ID
    context_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_count: int = Field(gt=0)
    split_counts: tuple[ExecutionWorkspaceCount, ...]
    public_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifacts: tuple[ExecutionWorkspaceArtifactBinding, ...] = Field(min_length=1)
    source_data_real: Literal[True] = True
    literal_registry_identifiers_removed: Literal[True] = True
    product_and_sponsor_names_removed: Literal[False] = False
    raw_intervention_and_sponsor_name_fields_omitted: Literal[True] = True
    free_text_removed: Literal[True] = True
    source_span_mapping_complete: Literal[False] = False
    residual_model_weight_reidentification_risk: Literal[True] = True
    organizer_private_mapping_required: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    externally_authenticated_receipt_required_for_admission: Literal[True] = True

    @model_validator(mode='after')
    def validate_artifacts(self) -> Self:
        paths = tuple(item.relative_path for item in self.artifacts)
        if paths != tuple(sorted(set(paths))):
            raise ValueError('workspace artifact bindings must use unique canonical order')
        role_by_prefix = {
            'public': ExecutionWorkspaceArtifactRole.PUBLIC,
            'organizer': ExecutionWorkspaceArtifactRole.ORGANIZER,
            'private': ExecutionWorkspaceArtifactRole.PRIVATE,
        }
        for artifact in self.artifacts:
            prefix = PurePosixPath(artifact.relative_path).parts[0]
            if role_by_prefix.get(prefix) != artifact.role:
                raise ValueError('workspace artifact role does not match its tree prefix')
            expected_mode = '0444' if artifact.role == ExecutionWorkspaceArtifactRole.PUBLIC else '0600'
            if artifact.mode != expected_mode:
                raise ValueError('workspace artifact mode does not match its role')
        return self


@dataclass(frozen=True)
class LoadedExecutionWorkspaceBuild:
    root: Path
    receipt: ExecutionWorkspaceBuildReceipt
    tasks: tuple[ExecutionTask, ...]
    gold: tuple[ExecutionPrivateGold, ...]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_sha256(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _require_secret(key: bytes, name: str) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ExecutionWorkspaceError(f'{name} must contain at least 32 bytes')


def _opaque_digest(key: bytes, domain: bytes, value: str) -> str:
    return hmac.new(key, domain + value.encode('utf-8'), hashlib.sha256).hexdigest()


def _clean_public_term(value: str, *, fallback: str) -> str:
    normalized = ' '.join(value.split()).strip()
    if not normalized:
        return fallback
    if len(normalized.encode('utf-8')) > 512:
        raise ExecutionWorkspaceError('public categorical term exceeds 512 UTF-8 bytes')
    if _NCT_PATTERN.search(normalized):
        raise ExecutionWorkspaceError('public categorical term contains an NCT identifier')
    return normalized


def _counts(values: Sequence[str], *, fallback: str) -> tuple[ExecutionWorkspaceCount, ...]:
    counter = Counter(_clean_public_term(value, fallback=fallback) for value in values)
    return tuple(ExecutionWorkspaceCount(name=name, count=counter[name]) for name in sorted(counter))


def _exact_one[T](values: Sequence[T], *, label: str) -> T:
    if len(values) != 1:
        raise ExecutionWorkspaceError(f'expected exactly one {label}, found {len(values)}')
    return values[0]


def _trial_view(
    *,
    public_trial_id: str,
    assignment: TrialAnchorAssignment,
    decision: AactExecutionDecisionRow,
    evidence: DecisionEvidenceRecord,
) -> ExecutionWorkspaceTrialView:
    if (assignment.anchor_date, assignment.nct_id) != (evidence.anchor_date, evidence.nct_id):
        raise ExecutionWorkspaceError('decision evidence does not match its anchor assignment')
    if (decision.archive_date, decision.nct_id) != (assignment.anchor_date, assignment.nct_id):
        raise ExecutionWorkspaceError('decision row does not match its anchor assignment')
    purposes = tuple(
        sorted({_clean_public_term(value, fallback='not reported') for value in evidence.primary_purposes})
    )
    conditions = tuple(sorted({_clean_public_term(value, fallback='not reported') for value in evidence.conditions}))
    return ExecutionWorkspaceTrialView(
        public_trial_id=public_trial_id,
        source_anchor_date=assignment.anchor_date,
        study_first_posted_date=decision.study_first_posted_date,
        phase=decision.phase,
        decision_status=decision.overall_status,
        planned_enrollment=decision.enrollment,
        planned_enrollment_type=decision.enrollment_type.value if decision.enrollment_type is not None else None,
        planned_primary_completion_date=decision.primary_completion_date,
        planned_primary_completion_date_type=(
            decision.primary_completion_date_type.value if decision.primary_completion_date_type is not None else None
        ),
        biological_intervention_count=decision.biological_intervention_count,
        intervention_type_counts=_counts(
            tuple(item.intervention_type for item in evidence.interventions),
            fallback='not reported',
        ),
        sponsor_class_counts=_counts(
            tuple(item.agency_class for item in evidence.sponsors),
            fallback='not reported',
        ),
        primary_purposes=purposes,
        conditions=conditions,
        results_section_present=decision.results_section_present,
    )


def _document(document_id: str, available_on: date, body: str) -> CutoffDocument:
    return CutoffDocument(
        document_id=document_id,
        available_on=available_on,
        body=body,
        body_sha256=_sha256(body.encode('utf-8')),
    )


def _public_bytes_are_identity_scrubbed(payload: bytes, private_ids: Sequence[str]) -> None:
    try:
        text = payload.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ExecutionWorkspaceError('model-facing artifacts must be UTF-8') from error
    if _NCT_PATTERN.search(text) or any(private_id.casefold() in text.casefold() for private_id in private_ids):
        raise ExecutionWorkspaceError('model-facing artifacts expose a private registry identifier')


def _build_execution_workspace_context_plan(
    *,
    inventory: ExecutionCohortInventory,
    relevance_queue: VaccineRelevanceReviewQueue,
    relevance_adjudications: VaccineRelevanceAdjudicationSet,
    trusted_relevance_review_receipt_sha256: str,
    split_manifest_sha256: str,
    split_by_nct: Mapping[str, Split],
    lineage_by_nct: Mapping[str, str],
    alias_key: bytes,
) -> ExecutionWorkspaceContextPlan:
    """Low-level context projection; callers must authenticate every source separately."""

    _require_secret(alias_key, 'workspace alias key')
    try:
        inventory = ExecutionCohortInventory.model_validate_json(canonical_json_bytes(inventory))
        relevance_queue = VaccineRelevanceReviewQueue.model_validate_json(canonical_json_bytes(relevance_queue))
        relevance_adjudications = VaccineRelevanceAdjudicationSet.model_validate_json(
            canonical_json_bytes(relevance_adjudications)
        )
        audit_execution_inventory(inventory)
        rebuilt_adjudications = finalize_relevance_adjudications(
            queue=relevance_queue,
            reviews=relevance_adjudications.decisions,
        )
    except (ValueError, ExecutionInventoryError) as error:
        raise ExecutionWorkspaceError(f'invalid decision-only workspace source: {error}') from error
    if canonical_json_bytes(rebuilt_adjudications) != canonical_json_bytes(relevance_adjudications):
        raise ExecutionWorkspaceError('relevance adjudications do not reconstruct from their exact queue')
    for value, label in (
        (trusted_relevance_review_receipt_sha256, 'review receipt'),
        (split_manifest_sha256, 'split manifest'),
    ):
        if re.fullmatch(_SHA256_PATTERN, value) is None:
            raise ExecutionWorkspaceError(f'{label} SHA-256 must be a 64-character digest')

    included = {
        (item.anchor_date, item.nct_id): item
        for item in relevance_adjudications.decisions
        if item.disposition == RelevanceDisposition.INCLUDE
    }
    included_ids = {nct_id for _, nct_id in included}
    if set(split_by_nct) != included_ids or set(lineage_by_nct) != included_ids:
        raise ExecutionWorkspaceError('split and lineage mappings must cover every and only relevance INCLUDE case')
    assignment_by_id = {item.nct_id: item for item in inventory.assignments}
    if not included_ids.issubset(assignment_by_id):
        raise ExecutionWorkspaceError('relevance INCLUDE set is not contained in the audited inventory')
    evidence_by_key = {(item.anchor_date, item.nct_id): item for item in relevance_queue.records}
    decision_by_key: dict[tuple[date, str], AactExecutionDecisionRow] = {}
    for nct_id in included_ids | {item.nct_id for item in relevance_queue.records}:
        assignment = assignment_by_id.get(nct_id)
        if assignment is None:
            continue
        decision_by_key[(assignment.anchor_date, nct_id)] = _exact_one(
            tuple(
                item
                for item in inventory.decision_rows
                if item.nct_id == nct_id and item.archive_date == assignment.anchor_date
            ),
            label=f'decision row for {nct_id}',
        )

    policy = execution_forecast_spec_policy()
    all_private_ids = tuple(sorted(assignment_by_id))
    entries: list[ExecutionWorkspacePlanEntry] = []
    for nct_id in sorted(included_ids):
        assignment = assignment_by_id[nct_id]
        decision = decision_by_key[(assignment.anchor_date, nct_id)]
        evidence = evidence_by_key[(assignment.anchor_date, nct_id)]
        episode_digest = _opaque_digest(alias_key, _EPISODE_ALIAS_DOMAIN, nct_id)
        episode_id = f'execution-dev-{episode_digest[:24]}'
        reference_ids = tuple(
            sorted(
                (
                    item.nct_id
                    for item in relevance_queue.records
                    if item.anchor_date <= assignment.anchor_date and item.nct_id != nct_id
                ),
                key=lambda value: _opaque_digest(
                    alias_key,
                    _REFERENCE_ORDER_DOMAIN + episode_id.encode('ascii') + b'\x00',
                    value,
                ),
            )
        )
        public_by_nct = {nct_id: 'trial-target'}
        public_by_nct.update(
            {reference_nct: f'trial-reference-{index:03d}' for index, reference_nct in enumerate(reference_ids, 1)}
        )
        target_view = _trial_view(
            public_trial_id='trial-target',
            assignment=assignment,
            decision=decision,
            evidence=evidence,
        )
        reference_views: list[ExecutionWorkspaceTrialView] = []
        bindings: list[ExecutionWorkspaceAliasBinding] = []
        for private_id, public_id in sorted(public_by_nct.items(), key=lambda item: item[1]):
            reference_assignment = assignment_by_id[private_id]
            reference_decision = decision_by_key[(reference_assignment.anchor_date, private_id)]
            reference_evidence = evidence_by_key[(reference_assignment.anchor_date, private_id)]
            bindings.append(
                ExecutionWorkspaceAliasBinding(
                    nct_id=private_id,
                    public_trial_id=public_id,
                    source_anchor_date=reference_assignment.anchor_date,
                    decision_source_record_sha256=reference_decision.source_record_sha256,
                )
            )
            if private_id != nct_id:
                reference_views.append(
                    _trial_view(
                        public_trial_id=public_id,
                        assignment=reference_assignment,
                        decision=reference_decision,
                        evidence=reference_evidence,
                    )
                )
        target_body = canonical_json_bytes(target_view).decode('utf-8')
        reference_body = ''.join(canonical_json_bytes(item).decode('utf-8') + '\n' for item in reference_views) or '\n'
        _public_bytes_are_identity_scrubbed(
            target_body.encode('utf-8') + reference_body.encode('utf-8'),
            all_private_ids,
        )
        documents = (
            _document('target-profile', assignment.anchor_date, target_body),
            _document('reference-trials', assignment.anchor_date, reference_body),
        )
        context = ExecutionTaskContext(
            episode_id=episode_id,
            target_trial_id='trial-target',
            decision_snapshot_id=assignment.decision_snapshot_id,
            anchor_date=assignment.anchor_date,
            label_snapshot_id=assignment.label_snapshot_id,
            label_archive_date=assignment.label_archive_date,
            planned_enrollment=assignment.planned_enrollment,
            planned_primary_completion_date=assignment.planned_primary_completion_date,
            enrollment_ratio_spec=policy.enrollment_ratio_spec,
            primary_completion_slippage_days_spec=policy.primary_completion_slippage_days_spec,
            cutoff_documents=documents,
        )
        lineage_group_id = lineage_by_nct[nct_id]
        public_lineage_id = f'lineage-{_opaque_digest(alias_key, _LINEAGE_ALIAS_DOMAIN, lineage_group_id)[:20]}'
        entries.append(
            ExecutionWorkspacePlanEntry(
                organizer_private_nct_id=nct_id,
                lineage_group_id=lineage_group_id,
                public_lineage_id=public_lineage_id,
                split=split_by_nct[nct_id],
                context=context,
                context_sha256=execution_task_context_sha256(context),
                decision_source_record_sha256=decision.source_record_sha256,
                relevance_evidence_sha256=evidence.evidence_sha256,
                alias_bindings=tuple(bindings),
                reference_trial_count=len(reference_views),
                public_source_bytes=sum(len(item.body.encode('utf-8')) for item in documents),
            )
        )
    split_counter = Counter(item.split.value for item in entries)
    return ExecutionWorkspaceContextPlan(
        inventory_sha256=_model_sha256(inventory),
        relevance_queue_sha256=_model_sha256(relevance_queue),
        relevance_adjudications_sha256=_model_sha256(relevance_adjudications),
        trusted_relevance_review_receipt_sha256=trusted_relevance_review_receipt_sha256,
        split_manifest_sha256=split_manifest_sha256,
        alias_key_id=_sha256(_ALIAS_KEY_DOMAIN + alias_key),
        entries=tuple(entries),
        task_count=len(entries),
        split_counts=tuple(
            ExecutionWorkspaceCount(name=name, count=split_counter[name]) for name in sorted(split_counter)
        ),
        lineage_count=len({item.lineage_group_id for item in entries}),
    )


def _task_markdown(task: ExecutionTask) -> bytes:
    return (
        '# Registry-observed trial execution forecast\n\n'
        f'Episode: `{task.context.episode_id}`\n\n'
        'Use only the committed decision-time documents in this workspace. Forecast the registry state '
        'exactly 48 calendar months after the anchor. Return one JSON object conforming to '
        '`vaxreplay.clinical-execution-submission.dev-v0.1`. Provide probability distributions for '
        'registry outcome, enrollment observation, and completion observation, plus the conditional '
        'enrollment-ratio point forecast and completion-slippage quantiles declared in `TASK.json`.\n\n'
        'This task measures registry-observed execution, not vaccine efficacy, safety, or scientific merit.\n'
    ).encode('utf-8')


def _public_task_manifest(entry: ExecutionWorkspacePlanEntry, task: ExecutionTask) -> bytes:
    sources = {
        f'sources/{document.document_id}.json'
        if document.document_id == 'target-profile'
        else f'sources/{document.document_id}.jsonl': {
            'sha256': document.body_sha256,
            'byte_count': len(document.body.encode('utf-8')),
        }
        for document in task.context.cutoff_documents
    }
    return canonical_json_bytes(
        {
            'schema_version': 'vaxreplay.clinical-execution-public-task-manifest.dev-v0.1',
            'episode_id': task.context.episode_id,
            'target_trial_id': task.context.target_trial_id,
            'public_lineage_id': entry.public_lineage_id,
            'split': entry.split.value,
            'task_sha256': _model_sha256(task),
            'task_context_sha256': task.context_sha256,
            'response_schema_version': 'vaxreplay.clinical-execution-submission.dev-v0.1',
            'sources': sources,
            'development_only': True,
            'leaderboard_admitted': False,
            'identity_contamination_controlled': False,
        }
    )


def _write_file(
    root: Path, relative_path: str, payload: bytes, role: ExecutionWorkspaceArtifactRole
) -> ExecutionWorkspaceArtifactBinding:
    if not payload:
        raise ExecutionWorkspaceError(f'workspace artifact cannot be empty: {relative_path}')
    if len(payload) > _MAX_PUBLIC_ARTIFACT_BYTES and role == ExecutionWorkspaceArtifactRole.PUBLIC:
        raise ExecutionWorkspaceError(f'public workspace artifact exceeds byte limit: {relative_path}')
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = 0o444 if role == ExecutionWorkspaceArtifactRole.PUBLIC else 0o600
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return ExecutionWorkspaceArtifactBinding(
        relative_path=relative_path,
        role=role,
        sha256=_sha256(payload),
        byte_count=len(payload),
        mode='0444' if role == ExecutionWorkspaceArtifactRole.PUBLIC else '0600',
    )


def _tree_sha256(artifacts: Sequence[ExecutionWorkspaceArtifactBinding], role: ExecutionWorkspaceArtifactRole) -> str:
    return _sha256(canonical_json_bytes([item.model_dump(mode='json') for item in artifacts if item.role == role]))


def _expected_workspace_artifact_paths(plan: ExecutionWorkspaceContextPlan) -> set[str]:
    paths = {'organizer/context-plan.json', 'organizer/task-index.json'}
    for entry in plan.entries:
        episode_id = entry.context.episode_id
        base = f'public/tasks/{episode_id}'
        paths.update(
            {
                f'{base}/TASK.json',
                f'{base}/TASK.md',
                f'{base}/task-manifest.json',
                f'organizer/tasks/{episode_id}/alias-map.json',
                f'private/tasks/{episode_id}/gold.json',
                f'private/tasks/{episode_id}/gold.key',
            }
        )
        for document in entry.context.cutoff_documents:
            suffix = 'json' if document.document_id == 'target-profile' else 'jsonl'
            paths.add(f'{base}/sources/{document.document_id}.{suffix}')
    return paths


def write_execution_workspace_build(
    *,
    plan: ExecutionWorkspaceContextPlan,
    gold_by_nct: Mapping[str, ExecutionPrivateGold],
    private_gold_master_key: bytes,
    output_root: Path,
) -> LoadedExecutionWorkspaceBuild:
    """Materialize public tasks and separately protected organizer/private scoring material."""

    _require_secret(private_gold_master_key, 'private-gold master key')
    plan = ExecutionWorkspaceContextPlan.model_validate_json(canonical_json_bytes(plan))
    expected_nct_ids = {entry.organizer_private_nct_id for entry in plan.entries}
    if set(gold_by_nct) != expected_nct_ids:
        raise ExecutionWorkspaceError('private gold must cover every and only planned workspace task')
    target = output_root.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f'workspace build already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.staging-', dir=target.parent))
    tasks: list[ExecutionTask] = []
    validated_gold: list[ExecutionPrivateGold] = []
    artifacts: list[ExecutionWorkspaceArtifactBinding] = []
    index_rows: list[dict[str, object]] = []
    try:
        artifacts.append(
            _write_file(
                staging,
                'organizer/context-plan.json',
                canonical_json_bytes(plan),
                ExecutionWorkspaceArtifactRole.ORGANIZER,
            )
        )
        for entry in plan.entries:
            supplied_gold = ExecutionPrivateGold.model_validate_json(
                canonical_json_bytes(gold_by_nct[entry.organizer_private_nct_id])
            )
            if (
                supplied_gold.organizer_private_nct_id != entry.organizer_private_nct_id
                or supplied_gold.episode_id != entry.context.episode_id
                or supplied_gold.target_trial_id != entry.context.target_trial_id
                or supplied_gold.task_context_sha256 != entry.context_sha256
                or supplied_gold.organizer_private_decision_record_sha256 != entry.decision_source_record_sha256
            ):
                raise ExecutionWorkspaceError('private gold does not bind its exact final workspace context')
            task_key = hmac.new(
                private_gold_master_key,
                _GOLD_KEY_DOMAIN + entry.context.episode_id.encode('ascii'),
                hashlib.sha256,
            ).digest()
            task = build_execution_task(context=entry.context, gold=supplied_gold, private_gold_key=task_key)
            validate_execution_task_gold(task, supplied_gold, task_key)
            tasks.append(task)
            validated_gold.append(supplied_gold)
            base = f'public/tasks/{entry.context.episode_id}'
            task_payload = canonical_json_bytes(task)
            public_manifest = _public_task_manifest(entry, task)
            public_payloads = {
                f'{base}/TASK.json': task_payload,
                f'{base}/TASK.md': _task_markdown(task),
                f'{base}/task-manifest.json': public_manifest,
            }
            for document in task.context.cutoff_documents:
                suffix = 'json' if document.document_id == 'target-profile' else 'jsonl'
                public_payloads[f'{base}/sources/{document.document_id}.{suffix}'] = document.body.encode('utf-8')
            private_ids = tuple(item.nct_id for item in entry.alias_bindings)
            for path, payload in sorted(public_payloads.items()):
                _public_bytes_are_identity_scrubbed(payload, private_ids)
                artifacts.append(_write_file(staging, path, payload, ExecutionWorkspaceArtifactRole.PUBLIC))
            artifacts.extend(
                (
                    _write_file(
                        staging,
                        f'organizer/tasks/{entry.context.episode_id}/alias-map.json',
                        canonical_json_bytes([binding.model_dump(mode='json') for binding in entry.alias_bindings]),
                        ExecutionWorkspaceArtifactRole.ORGANIZER,
                    ),
                    _write_file(
                        staging,
                        f'private/tasks/{entry.context.episode_id}/gold.json',
                        canonical_json_bytes(supplied_gold),
                        ExecutionWorkspaceArtifactRole.PRIVATE,
                    ),
                    _write_file(
                        staging,
                        f'private/tasks/{entry.context.episode_id}/gold.key',
                        task_key,
                        ExecutionWorkspaceArtifactRole.PRIVATE,
                    ),
                )
            )
            index_rows.append(
                {
                    'episode_id': entry.context.episode_id,
                    'organizer_private_nct_id': entry.organizer_private_nct_id,
                    'lineage_group_id': entry.lineage_group_id,
                    'public_lineage_id': entry.public_lineage_id,
                    'split': entry.split.value,
                    'task_sha256': _model_sha256(task),
                    'gold_sha256': _model_sha256(supplied_gold),
                    'gold_key_id': hashlib.sha256(task_key).hexdigest(),
                }
            )
        artifacts.append(
            _write_file(
                staging,
                'organizer/task-index.json',
                canonical_json_bytes(
                    {
                        'schema_version': 'vaxreplay.clinical-execution-organizer-index.dev-v0.1',
                        'tasks': index_rows,
                    }
                ),
                ExecutionWorkspaceArtifactRole.ORGANIZER,
            )
        )
        artifacts_tuple = tuple(sorted(artifacts, key=lambda item: item.relative_path))
        receipt = ExecutionWorkspaceBuildReceipt(
            context_plan_sha256=_model_sha256(plan),
            task_count=len(tasks),
            split_counts=plan.split_counts,
            public_tree_sha256=_tree_sha256(artifacts_tuple, ExecutionWorkspaceArtifactRole.PUBLIC),
            organizer_tree_sha256=_tree_sha256(artifacts_tuple, ExecutionWorkspaceArtifactRole.ORGANIZER),
            private_tree_sha256=_tree_sha256(artifacts_tuple, ExecutionWorkspaceArtifactRole.PRIVATE),
            artifacts=artifacts_tuple,
        )
        receipt_path = staging / 'BUILD-RECEIPT.json'
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            payload = canonical_json_bytes(receipt)
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        for directory in sorted(
            (path for path in staging.rglob('*') if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            fsync_directory(directory)
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return LoadedExecutionWorkspaceBuild(
        root=target,
        receipt=receipt,
        tasks=tuple(tasks),
        gold=tuple(validated_gold),
    )


def verify_execution_workspace_build(
    root: Path,
    *,
    expected_receipt_sha256: str,
) -> LoadedExecutionWorkspaceBuild:
    """Verify exact package bytes; the expected receipt digest must be pinned out of band."""

    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ExecutionWorkspaceError('workspace build root cannot be a symbolic link')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ExecutionWorkspaceError('workspace build root must be a directory')
    if any(path.is_symlink() for path in resolved.rglob('*')):
        raise ExecutionWorkspaceError('workspace build cannot contain symbolic links')
    receipt_path = resolved / 'BUILD-RECEIPT.json'
    receipt_payload = receipt_path.read_bytes()
    if re.fullmatch(_SHA256_PATTERN, expected_receipt_sha256) is None or not hmac.compare_digest(
        _sha256(receipt_payload), expected_receipt_sha256
    ):
        raise ExecutionWorkspaceError('workspace build receipt does not match its external pin')
    try:
        receipt = ExecutionWorkspaceBuildReceipt.model_validate_json(receipt_payload)
    except ValueError as error:
        raise ExecutionWorkspaceError(f'invalid workspace build receipt: {error}') from error
    expected_paths = {item.relative_path for item in receipt.artifacts} | {'BUILD-RECEIPT.json'}
    observed_paths = {
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob('*')
        if path.is_file() and not path.is_symlink()
    }
    if observed_paths != expected_paths:
        raise ExecutionWorkspaceError('workspace build contains missing or uncommitted files')
    payload_by_path: dict[str, bytes] = {}
    for artifact in receipt.artifacts:
        path = resolved / artifact.relative_path
        if path.is_symlink() or not path.is_file():
            raise ExecutionWorkspaceError(f'workspace artifact is not a regular file: {artifact.relative_path}')
        payload = path.read_bytes()
        observed_mode = path.stat().st_mode & 0o777
        expected_mode = 0o444 if artifact.mode == '0444' else 0o600
        if (
            len(payload) != artifact.byte_count
            or not hmac.compare_digest(_sha256(payload), artifact.sha256)
            or observed_mode != expected_mode
        ):
            raise ExecutionWorkspaceError(f'workspace artifact does not match receipt: {artifact.relative_path}')
        payload_by_path[artifact.relative_path] = payload
    if (
        _tree_sha256(receipt.artifacts, ExecutionWorkspaceArtifactRole.PUBLIC) != receipt.public_tree_sha256
        or _tree_sha256(receipt.artifacts, ExecutionWorkspaceArtifactRole.ORGANIZER) != receipt.organizer_tree_sha256
        or _tree_sha256(receipt.artifacts, ExecutionWorkspaceArtifactRole.PRIVATE) != receipt.private_tree_sha256
    ):
        raise ExecutionWorkspaceError('workspace tree commitments do not reconstruct')
    try:
        plan = ExecutionWorkspaceContextPlan.model_validate_json(payload_by_path['organizer/context-plan.json'])
    except ValueError as error:
        raise ExecutionWorkspaceError(f'invalid workspace context plan: {error}') from error
    if _model_sha256(plan) != receipt.context_plan_sha256:
        raise ExecutionWorkspaceError('workspace context plan does not match the receipt')
    if set(payload_by_path) != _expected_workspace_artifact_paths(plan):
        raise ExecutionWorkspaceError('workspace receipt contains an unexpected semantic artifact inventory')
    all_private_ids = tuple(sorted({binding.nct_id for entry in plan.entries for binding in entry.alias_bindings}))
    for relative_path, payload in payload_by_path.items():
        if relative_path.startswith('public/'):
            _public_bytes_are_identity_scrubbed(payload, all_private_ids)
    task_by_episode: dict[str, ExecutionTask] = {}
    gold_by_episode: dict[str, ExecutionPrivateGold] = {}
    for entry in plan.entries:
        episode_id = entry.context.episode_id
        try:
            task = ExecutionTask.model_validate_json(payload_by_path[f'public/tasks/{episode_id}/TASK.json'])
            gold = ExecutionPrivateGold.model_validate_json(payload_by_path[f'private/tasks/{episode_id}/gold.json'])
        except ValueError as error:
            raise ExecutionWorkspaceError(f'invalid task/gold material for {episode_id}: {error}') from error
        key = payload_by_path[f'private/tasks/{episode_id}/gold.key']
        validate_execution_task_gold(task, gold, key)
        if task.context != entry.context or gold.organizer_private_nct_id != entry.organizer_private_nct_id:
            raise ExecutionWorkspaceError('workspace task/gold does not match its organizer plan')
        for document in task.context.cutoff_documents:
            suffix = 'json' if document.document_id == 'target-profile' else 'jsonl'
            source_payload = payload_by_path[f'public/tasks/{episode_id}/sources/{document.document_id}.{suffix}']
            if source_payload != document.body.encode('utf-8'):
                raise ExecutionWorkspaceError('workspace source bytes do not match their task context')
        task_by_episode[episode_id] = task
        gold_by_episode[episode_id] = gold
    if len(task_by_episode) != receipt.task_count:
        raise ExecutionWorkspaceError('workspace verified task count does not match receipt')
    return LoadedExecutionWorkspaceBuild(
        root=resolved,
        receipt=receipt,
        tasks=tuple(task_by_episode[key] for key in sorted(task_by_episode)),
        gold=tuple(gold_by_episode[key] for key in sorted(gold_by_episode)),
    )


__all__ = [
    'EXECUTION_WORKSPACE_ALIAS_SCHEME',
    'EXECUTION_WORKSPACE_BUILDER_ID',
    'ExecutionWorkspaceAliasBinding',
    'ExecutionWorkspaceArtifactBinding',
    'ExecutionWorkspaceBuildReceipt',
    'ExecutionWorkspaceContextPlan',
    'ExecutionWorkspaceError',
    'ExecutionWorkspacePlanEntry',
    'ExecutionWorkspaceTrialView',
    'LoadedExecutionWorkspaceBuild',
    'verify_execution_workspace_build',
    'write_execution_workspace_build',
]
