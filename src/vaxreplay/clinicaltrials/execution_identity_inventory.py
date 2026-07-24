"""Authenticated local-only identity-fingerprint inventory for real Lane A tasks.

The audit answers a deliberately narrow question: can the exact decision-time metadata shown for
an opaque target be joined back to organizer-known records in the frozen local AACT catalogs?  It
does not call a model or provider, inspect model weights, infer training membership, or establish
the independent-attacker evidence required by :mod:`execution_contamination` case strata.

Every input is opened only after a caller-supplied external receipt pin is checked.  The result is
HMAC authenticated, contains no NCT identifier, and binds the exact five-file public task surface.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.agentic.workspace import model_visible_surface_bytes
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_adapter import AactExecutionBuildReceipt
from vaxreplay.clinicaltrials.execution_contamination import ExecutionCaseSurfaceBinding
from vaxreplay.clinicaltrials.execution_inventory import audit_execution_inventory
from vaxreplay.clinicaltrials.execution_public_release import (
    LoadedExecutionPublicRelease,
    verify_execution_public_release,
)
from vaxreplay.clinicaltrials.execution_schema import (
    AactExecutionDecisionRow,
    ExecutionCohortInventory,
)
from vaxreplay.clinicaltrials.execution_task import ExecutionTask
from vaxreplay.clinicaltrials.execution_workspace import (
    ExecutionWorkspaceContextPlan,
    LoadedExecutionWorkspaceBuild,
    verify_execution_workspace_build,
)

EXECUTION_IDENTITY_FINGERPRINT_POLICY_SCHEMA_VERSION = (
    'vaxreplay.clinical-execution-identity-fingerprint-policy.dev-v0.1'
)
EXECUTION_IDENTITY_FINGERPRINT_INVENTORY_SCHEMA_VERSION = (
    'vaxreplay.clinical-execution-identity-fingerprint-inventory.dev-v0.1'
)
AUTHENTICATED_EXECUTION_IDENTITY_FINGERPRINT_INVENTORY_SCHEMA_VERSION = (
    'vaxreplay.authenticated-clinical-execution-identity-fingerprint-inventory.dev-v0.1'
)
EXECUTION_IDENTITY_FINGERPRINT_POLICY_ID = 'aact-local-exact-surface-fingerprint-inventory-v0.1'
EXECUTION_IDENTITY_FINGERPRINT_ARTIFACT = 'IDENTITY-FINGERPRINT-INVENTORY.json'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_NCT_PATTERN = re.compile(rb'NCT\d{8}', re.IGNORECASE)
_INVENTORY_HMAC_DOMAIN = b'vaxreplay.clinical-execution-identity-fingerprint-inventory.dev-v0.1\x00'
_INVENTORY_KEY_ID_DOMAIN = b'vaxreplay.clinical-execution-identity-fingerprint-key-id.dev-v0.1\x00'
_MAX_RECEIPT_BYTES = 32 * 1024 * 1024
_MAX_CATALOG_ARTIFACT_BYTES = 512 * 1024 * 1024

# These are an exact subset of fields present in ``ExecutionWorkspaceTrialView``.  A unique match
# on a subset is sufficient to show that the complete visible profile is fingerprinting-capable.
_REGISTRY_CORE_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ('source_anchor_date', 'archive_date'),
    ('study_first_posted_date', 'study_first_posted_date'),
    ('phase', 'phase'),
    ('decision_status', 'overall_status'),
    ('planned_enrollment', 'enrollment'),
    ('planned_enrollment_type', 'enrollment_type'),
    ('planned_primary_completion_date', 'primary_completion_date'),
    ('planned_primary_completion_date_type', 'primary_completion_date_type'),
    ('biological_intervention_count', 'biological_intervention_count'),
    ('results_section_present', 'results_section_present'),
)
_COARSENED_OMITTED_FIELDS = ('planned_enrollment', 'planned_primary_completion_date')
_MINIMAL_OMITTED_FIELDS = (
    'planned_enrollment',
    'planned_primary_completion_date',
    'study_first_posted_date',
)
_STATIC_VISIBLE_FIELDS = frozenset(
    {'schema_version', 'public_trial_id', 'identity_fields_removed', 'free_text_removed'}
)


class ExecutionIdentityFingerprintError(ValueError):
    """An identity inventory input or authenticated artifact failed closed."""


class ExecutionIdentityFingerprintPolicy(StrictModel):
    schema_version: Literal['vaxreplay.clinical-execution-identity-fingerprint-policy.dev-v0.1'] = (
        EXECUTION_IDENTITY_FINGERPRINT_POLICY_SCHEMA_VERSION
    )
    policy_id: Literal['aact-local-exact-surface-fingerprint-inventory-v0.1'] = EXECUTION_IDENTITY_FINGERPRINT_POLICY_ID
    public_surface_variant: Literal['released_task_five_file_surface'] = 'released_task_five_file_surface'
    registry_core_visible_fields: tuple[str, ...] = tuple(item[0] for item in _REGISTRY_CORE_FIELD_MAP)
    coarsened_omitted_fields: tuple[str, ...] = _COARSENED_OMITTED_FIELDS
    minimal_omitted_fields: tuple[str, ...] = _MINIMAL_OMITTED_FIELDS
    near_unique_maximum_candidates: Literal[3] = 3
    exact_same_anchor_catalog_required: Literal[True] = True
    true_target_membership_required: Literal[True] = True
    caller_supplied_external_receipt_pins_required: Literal[True] = True
    exact_release_and_workspace_cross_binding_required: Literal[True] = True
    exact_catalog_artifact_inventory_required: Literal[True] = True
    no_external_model_or_provider_calls: Literal[True] = True
    organizer_private_identifiers_omitted_from_output: Literal[True] = True
    formal_contamination_stratum_assigned: Literal[False] = False
    independent_organizer_attackers_run: Literal[False] = False
    proves_model_training_membership: Literal[False] = False
    proves_absence_of_model_weight_contamination: Literal[False] = False
    model_weight_cleanliness_claimed: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False


EXECUTION_IDENTITY_FINGERPRINT_POLICY = ExecutionIdentityFingerprintPolicy()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_sha256(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def execution_identity_fingerprint_policy_sha256() -> str:
    return _model_sha256(EXECUTION_IDENTITY_FINGERPRINT_POLICY)


def _require_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ExecutionIdentityFingerprintError('identity-inventory HMAC key must contain at least 32 bytes')


def execution_identity_fingerprint_key_id(key: bytes) -> str:
    _require_key(key)
    return _sha256(_INVENTORY_KEY_ID_DOMAIN + key)


class FrozenDecisionCatalogBinding(StrictModel):
    """Non-identifying binding to one verified, same-anchor decision catalog."""

    anchor_date: date
    snapshot_id: str = Field(min_length=1)
    source_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_rows_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_row_count: int = Field(gt=0)
    complete_decision_projection_verified: Literal[True] = True
    exact_artifact_inventory_verified: Literal[True] = True
    caller_supplied_external_pin_verified: Literal[True] = True
    source_data_real: bool


@dataclass(frozen=True, slots=True)
class LoadedFrozenDecisionCatalog:
    root: Path
    binding: FrozenDecisionCatalogBinding
    rows: tuple[AactExecutionDecisionRow, ...]


def _expected_directories(paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for value in paths:
        parent = PurePosixPath(value).parent
        while parent != PurePosixPath('.'):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _exact_tree_inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob('*'):
        relative = path.relative_to(root).as_posix()
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ExecutionIdentityFingerprintError('frozen decision catalog cannot contain symbolic links')
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ExecutionIdentityFingerprintError('frozen decision catalog files cannot be hard linked')
            files.add(relative)
        elif stat.S_ISDIR(metadata.st_mode):
            directories.add(relative)
        else:
            raise ExecutionIdentityFingerprintError('frozen decision catalog contains a non-regular descendant')
    return files, directories


def _read_exact_file(path: Path, *, maximum_bytes: int, expected_mode: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ExecutionIdentityFingerprintError(f'cannot open pinned identity-audit artifact: {path.name}') from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
            or (expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode)
        ):
            raise ExecutionIdentityFingerprintError('pinned identity-audit artifact is unsafe or oversized')
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes - len(content) + 1))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ExecutionIdentityFingerprintError('pinned identity-audit artifact exceeds its byte limit')
        after = os.fstat(descriptor)
        stable_fields = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ExecutionIdentityFingerprintError('pinned identity-audit artifact changed while being read')
        return bytes(content)
    finally:
        os.close(descriptor)


def load_frozen_decision_catalog(
    root: Path,
    *,
    expected_build_receipt_sha256: str,
    allow_synthetic_test_only: bool = False,
) -> LoadedFrozenDecisionCatalog:
    """Verify an exact AACT source build and return only its decision projection.

    ``expected_build_receipt_sha256`` must come from outside the catalog being opened.  In normal
    operation synthetic builds fail closed; the explicit escape hatch exists only for unit tests.
    """

    if re.fullmatch(_SHA256_PATTERN, expected_build_receipt_sha256) is None:
        raise ExecutionIdentityFingerprintError('expected frozen-catalog receipt SHA-256 is invalid')
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ExecutionIdentityFingerprintError('frozen decision catalog root cannot be a symbolic link')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ExecutionIdentityFingerprintError('frozen decision catalog root must be a directory')
    receipt_path = resolved / 'BUILD-RECEIPT.json'
    receipt_payload = _read_exact_file(receipt_path, maximum_bytes=_MAX_RECEIPT_BYTES)
    if not hmac.compare_digest(_sha256(receipt_payload), expected_build_receipt_sha256):
        raise ExecutionIdentityFingerprintError('frozen decision catalog does not match its external pin')
    try:
        receipt = AactExecutionBuildReceipt.model_validate_json(receipt_payload)
    except ValueError as error:
        raise ExecutionIdentityFingerprintError(f'invalid frozen decision catalog receipt: {error}') from error
    expected_receipt_payload = canonical_json_bytes(receipt) + b'\n'
    if receipt_payload != expected_receipt_payload:
        raise ExecutionIdentityFingerprintError('frozen decision catalog receipt must use canonical JSON plus LF')
    if receipt.synthetic and not allow_synthetic_test_only:
        raise ExecutionIdentityFingerprintError('real identity inventory cannot use a synthetic decision catalog')
    if not receipt.synthetic and receipt.source_binding.mode != 'trusted_official_real':
        raise ExecutionIdentityFingerprintError('real decision catalog lacks its trusted official source binding')

    expected_files = {item.relative_path for item in receipt.artifacts} | {'BUILD-RECEIPT.json'}
    observed_files, observed_directories = _exact_tree_inventory(resolved)
    if observed_files != expected_files or observed_directories != _expected_directories(expected_files):
        raise ExecutionIdentityFingerprintError('frozen decision catalog exact artifact inventory mismatch')
    payload_by_path: dict[str, bytes] = {}
    for artifact in receipt.artifacts:
        payload = _read_exact_file(resolved / artifact.relative_path, maximum_bytes=_MAX_CATALOG_ARTIFACT_BYTES)
        if (len(payload), _sha256(payload)) != (artifact.byte_count, artifact.sha256):
            raise ExecutionIdentityFingerprintError(
                f'frozen decision catalog artifact does not match receipt: {artifact.relative_path}'
            )
        payload_by_path[artifact.relative_path] = payload
    try:
        inventory = ExecutionCohortInventory.model_validate_json(payload_by_path['organizer/cohort-inventory.json'])
        audit_execution_inventory(inventory)
    except (KeyError, ValueError) as error:
        raise ExecutionIdentityFingerprintError(f'frozen decision inventory failed reconstruction: {error}') from error
    if payload_by_path['organizer/cohort-inventory.json'] != canonical_json_bytes(inventory) + b'\n':
        raise ExecutionIdentityFingerprintError('frozen decision inventory must use canonical JSON plus LF')
    if len(inventory.policy.anchors) != 1:
        raise ExecutionIdentityFingerprintError('each frozen decision catalog must contain exactly one anchor')
    anchor = inventory.policy.anchors[0]
    rows = tuple(inventory.decision_rows)
    if (
        anchor.anchor_date,
        anchor.decision_snapshot_id,
        anchor.decision_rows_sha256,
        anchor.decision_row_count,
    ) != (
        receipt.decision_archive.archive_date,
        receipt.decision_archive.snapshot_id,
        inventory.decision_rows_sha256,
        len(rows),
    ):
        raise ExecutionIdentityFingerprintError('frozen decision catalog receipt/inventory binding mismatch')
    if any((row.archive_date, row.snapshot_id) != (anchor.anchor_date, anchor.decision_snapshot_id) for row in rows):
        raise ExecutionIdentityFingerprintError('frozen decision catalog contains a row from another anchor')
    nct_ids = tuple(row.nct_id for row in rows)
    if len(nct_ids) != len(set(nct_ids)):
        raise ExecutionIdentityFingerprintError('frozen same-anchor decision catalog contains duplicate records')
    binding = FrozenDecisionCatalogBinding(
        anchor_date=anchor.anchor_date,
        snapshot_id=anchor.decision_snapshot_id,
        source_build_receipt_sha256=expected_build_receipt_sha256,
        decision_archive_sha256=receipt.decision_archive.archive_sha256,
        decision_rows_sha256=inventory.decision_rows_sha256,
        decision_row_count=len(rows),
        source_data_real=not receipt.synthetic,
    )
    return LoadedFrozenDecisionCatalog(root=resolved, binding=binding, rows=rows)


class ExecutionIdentityFingerprintCase(StrictModel):
    """One non-identifying local join result bound to an exact public task surface."""

    surface_binding: ExecutionCaseSurfaceBinding
    source_build_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    exact_visible_profile_candidate_count: int = Field(gt=0)
    exact_registry_core_candidate_count: int = Field(gt=0)
    coarsened_registry_core_candidate_count: int = Field(gt=0)
    minimal_registry_core_candidate_count: int = Field(gt=0)
    exact_visible_profile_unique: bool
    exact_registry_core_unique: bool
    coarsened_registry_core_unique: bool
    coarsened_registry_core_near_unique: bool
    minimal_registry_core_unique: bool
    minimal_registry_core_near_unique: bool
    true_target_present_in_every_candidate_set: Literal[True] = True
    organizer_private_identity_copied_to_output: Literal[False] = False

    @model_validator(mode='after')
    def validate_dispositions(self) -> Self:
        maximum = EXECUTION_IDENTITY_FINGERPRINT_POLICY.near_unique_maximum_candidates
        expected = (
            self.exact_visible_profile_candidate_count == 1,
            self.exact_registry_core_candidate_count == 1,
            self.coarsened_registry_core_candidate_count == 1,
            self.coarsened_registry_core_candidate_count <= maximum,
            self.minimal_registry_core_candidate_count == 1,
            self.minimal_registry_core_candidate_count <= maximum,
        )
        observed = (
            self.exact_visible_profile_unique,
            self.exact_registry_core_unique,
            self.coarsened_registry_core_unique,
            self.coarsened_registry_core_near_unique,
            self.minimal_registry_core_unique,
            self.minimal_registry_core_near_unique,
        )
        if observed != expected:
            raise ValueError('identity-fingerprint case flags do not follow the fixed policy')
        if _NCT_PATTERN.search(canonical_json_bytes(self)):
            raise ValueError('identity-fingerprint case output cannot expose an NCT identifier')
        return self


class ExecutionIdentityFingerprintAggregate(StrictModel):
    case_count: int = Field(gt=0)
    exact_visible_profile_unique_count: int = Field(ge=0)
    exact_registry_core_unique_count: int = Field(ge=0)
    coarsened_registry_core_unique_count: int = Field(ge=0)
    coarsened_registry_core_near_unique_count: int = Field(ge=0)
    minimal_registry_core_unique_count: int = Field(ge=0)
    minimal_registry_core_near_unique_count: int = Field(ge=0)
    maximum_exact_registry_core_candidate_count: int = Field(gt=0)
    maximum_coarsened_registry_core_candidate_count: int = Field(gt=0)
    maximum_minimal_registry_core_candidate_count: int = Field(gt=0)


class ExecutionIdentityFingerprintInventory(StrictModel):
    schema_version: Literal['vaxreplay.clinical-execution-identity-fingerprint-inventory.dev-v0.1'] = (
        EXECUTION_IDENTITY_FINGERPRINT_INVENTORY_SCHEMA_VERSION
    )
    inventory_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_release_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_release_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_workspace_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_workspace_public_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    frozen_catalogs: tuple[FrozenDecisionCatalogBinding, ...] = Field(min_length=1)
    cases: tuple[ExecutionIdentityFingerprintCase, ...] = Field(min_length=1)
    aggregate: ExecutionIdentityFingerprintAggregate
    exact_release_and_workspace_cross_binding_verified: Literal[True] = True
    exact_model_facing_surfaces_bound: Literal[True] = True
    complete_release_case_universe_covered: Literal[True] = True
    caller_supplied_external_input_pins_verified: Literal[True] = True
    local_frozen_catalog_join_only: Literal[True] = True
    external_model_or_provider_calls_made: Literal[False] = False
    organizer_private_identifiers_present: Literal[False] = False
    deterministic_join_is_not_independent_attacker_probe: Literal[True] = True
    formal_contamination_strata_assigned: Literal[False] = False
    workspace_future_leakage_audit_complete: Literal[False] = False
    model_weight_contamination_eliminated: Literal[False] = False
    proves_model_training_membership: Literal[False] = False
    proves_absence_of_model_weight_contamination: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @field_validator('frozen_catalogs')
    @classmethod
    def validate_catalogs(
        cls, value: tuple[FrozenDecisionCatalogBinding, ...]
    ) -> tuple[FrozenDecisionCatalogBinding, ...]:
        dates = tuple(item.anchor_date for item in value)
        if dates != tuple(sorted(set(dates))):
            raise ValueError('frozen decision catalogs must use unique ascending anchors')
        return value

    @field_validator('cases')
    @classmethod
    def validate_cases(
        cls, value: tuple[ExecutionIdentityFingerprintCase, ...]
    ) -> tuple[ExecutionIdentityFingerprintCase, ...]:
        episode_ids = tuple(item.surface_binding.episode_id for item in value)
        if episode_ids != tuple(sorted(set(episode_ids))):
            raise ValueError('identity-fingerprint cases must use unique ascending episode IDs')
        return value

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        if self.policy_sha256 != execution_identity_fingerprint_policy_sha256():
            raise ValueError('identity-fingerprint inventory does not bind the fixed policy')
        cases = self.cases
        expected = ExecutionIdentityFingerprintAggregate(
            case_count=len(cases),
            exact_visible_profile_unique_count=sum(item.exact_visible_profile_unique for item in cases),
            exact_registry_core_unique_count=sum(item.exact_registry_core_unique for item in cases),
            coarsened_registry_core_unique_count=sum(item.coarsened_registry_core_unique for item in cases),
            coarsened_registry_core_near_unique_count=sum(item.coarsened_registry_core_near_unique for item in cases),
            minimal_registry_core_unique_count=sum(item.minimal_registry_core_unique for item in cases),
            minimal_registry_core_near_unique_count=sum(item.minimal_registry_core_near_unique for item in cases),
            maximum_exact_registry_core_candidate_count=max(item.exact_registry_core_candidate_count for item in cases),
            maximum_coarsened_registry_core_candidate_count=max(
                item.coarsened_registry_core_candidate_count for item in cases
            ),
            maximum_minimal_registry_core_candidate_count=max(
                item.minimal_registry_core_candidate_count for item in cases
            ),
        )
        if self.aggregate != expected:
            raise ValueError('identity-fingerprint aggregate does not reconstruct from its cases')
        if _NCT_PATTERN.search(canonical_json_bytes(self)):
            raise ValueError('identity-fingerprint inventory cannot expose an NCT identifier')
        return self


class AuthenticatedExecutionIdentityFingerprintInventory(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-clinical-execution-identity-fingerprint-inventory.dev-v0.1'] = (
        AUTHENTICATED_EXECUTION_IDENTITY_FINGERPRINT_INVENTORY_SCHEMA_VERSION
    )
    inventory: ExecutionIdentityFingerprintInventory
    inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    hmac_key_id: str = Field(pattern=_SHA256_PATTERN)
    inventory_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_hmac_authenticated: Literal[True] = True
    key_included_in_artifact: Literal[False] = False

    @model_validator(mode='after')
    def validate_hash(self) -> Self:
        if self.inventory_sha256 != _model_sha256(self.inventory):
            raise ValueError('authenticated identity inventory carries the wrong inventory hash')
        if _NCT_PATTERN.search(canonical_json_bytes(self)):
            raise ValueError('authenticated identity inventory cannot expose an NCT identifier')
        return self


def _inventory_hmac(inventory: ExecutionIdentityFingerprintInventory, key: bytes) -> str:
    _require_key(key)
    return hmac.new(
        key,
        _INVENTORY_HMAC_DOMAIN + canonical_json_bytes(inventory),
        hashlib.sha256,
    ).hexdigest()


def _target_profile(task: ExecutionTask) -> dict[str, object]:
    matches = tuple(document for document in task.context.cutoff_documents if document.document_id == 'target-profile')
    if len(matches) != 1:
        raise ExecutionIdentityFingerprintError('each identity audit task requires one target-profile document')
    try:
        value = json.loads(matches[0].body)
    except (TypeError, ValueError) as error:
        raise ExecutionIdentityFingerprintError('target-profile document is not valid JSON') from error
    if not isinstance(value, dict) or value.get('public_trial_id') != 'trial-target':
        raise ExecutionIdentityFingerprintError('target-profile document has an invalid public target')
    missing = {field for field, _ in _REGISTRY_CORE_FIELD_MAP} - set(value)
    if missing:
        raise ExecutionIdentityFingerprintError(f'target profile is missing registry-core fields: {sorted(missing)}')
    return value


def _reference_profiles(task: ExecutionTask) -> tuple[dict[str, object], ...]:
    matches = tuple(
        document for document in task.context.cutoff_documents if document.document_id == 'reference-trials'
    )
    if len(matches) != 1:
        raise ExecutionIdentityFingerprintError('each identity audit task requires one reference-trials document')
    values: list[dict[str, object]] = []
    for line in matches[0].body.splitlines():
        if not line:
            continue
        try:
            value = json.loads(line)
        except (TypeError, ValueError) as error:
            raise ExecutionIdentityFingerprintError('reference-trials document is not valid JSONL') from error
        if not isinstance(value, dict):
            raise ExecutionIdentityFingerprintError('reference-trials JSONL rows must be objects')
        values.append(value)
    return tuple(values)


def _visible_profile_signature(profile: Mapping[str, object]) -> bytes:
    return canonical_json_bytes({key: profile[key] for key in sorted(set(profile) - _STATIC_VISIBLE_FIELDS)})


def _profile_core_signature(profile: Mapping[str, object], omitted: Sequence[str] = ()) -> bytes:
    omitted_set = set(omitted)
    return canonical_json_bytes(
        {field: profile[field] for field, _ in _REGISTRY_CORE_FIELD_MAP if field not in omitted_set}
    )


def _row_core_signature(row: AactExecutionDecisionRow, omitted: Sequence[str] = ()) -> bytes:
    omitted_set = set(omitted)
    dumped = row.model_dump(mode='json')
    return canonical_json_bytes(
        {
            visible_field: dumped[row_field]
            for visible_field, row_field in _REGISTRY_CORE_FIELD_MAP
            if visible_field not in omitted_set
        }
    )


def _released_task_surface(release: LoadedExecutionPublicRelease, episode_id: str) -> bytes:
    relative_paths = {
        'TASK.json': f'tasks/{episode_id}/TASK.json',
        'TASK.md': f'tasks/{episode_id}/TASK.md',
        'sources/reference-trials.jsonl': f'tasks/{episode_id}/sources/reference-trials.jsonl',
        'sources/target-profile.json': f'tasks/{episode_id}/sources/target-profile.json',
        'task-manifest.json': f'tasks/{episode_id}/task-manifest.json',
    }
    artifacts = {item.relative_path: item for item in release.receipt.artifacts}
    files: dict[str, bytes] = {}
    for visible_path, release_path in relative_paths.items():
        binding = artifacts.get(release_path)
        if binding is None:
            raise ExecutionIdentityFingerprintError('released task surface is absent from its verified receipt')
        payload = _read_exact_file(
            release.root / release_path,
            maximum_bytes=binding.byte_count,
            expected_mode=0o444,
        )
        if (len(payload), _sha256(payload)) != (binding.byte_count, binding.sha256):
            raise ExecutionIdentityFingerprintError('released task surface changed after verification')
        files[visible_path] = payload
    return model_visible_surface_bytes(files)


def _workspace_plan(workspace: LoadedExecutionWorkspaceBuild) -> ExecutionWorkspaceContextPlan:
    relative_path = 'organizer/context-plan.json'
    bindings = {item.relative_path: item for item in workspace.receipt.artifacts}
    binding = bindings.get(relative_path)
    if binding is None:
        raise ExecutionIdentityFingerprintError('verified workspace receipt omits its organizer context plan')
    try:
        payload = _read_exact_file(
            workspace.root / relative_path,
            maximum_bytes=binding.byte_count,
            expected_mode=0o600,
        )
        if (len(payload), _sha256(payload)) != (binding.byte_count, binding.sha256):
            raise ExecutionIdentityFingerprintError('workspace context plan changed after verification')
        return ExecutionWorkspaceContextPlan.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise ExecutionIdentityFingerprintError(f'cannot load verified workspace context plan: {error}') from error


def _cross_bind_release_and_workspace(
    release: LoadedExecutionPublicRelease,
    workspace: LoadedExecutionWorkspaceBuild,
) -> None:
    receipt = release.receipt
    if (
        receipt.source_workspace_receipt_sha256,
        receipt.source_workspace_context_plan_sha256,
        receipt.source_workspace_public_tree_sha256,
    ) != (
        _model_sha256(workspace.receipt),
        workspace.receipt.context_plan_sha256,
        workspace.receipt.public_tree_sha256,
    ):
        raise ExecutionIdentityFingerprintError('public release does not cross-bind the verified source workspace')
    release_tasks = {item.context.episode_id: canonical_json_bytes(item) for item in release.tasks}
    workspace_tasks = {item.context.episode_id: canonical_json_bytes(item) for item in workspace.tasks}
    if release_tasks != workspace_tasks:
        raise ExecutionIdentityFingerprintError('public release tasks differ from the verified source workspace')


def build_execution_identity_fingerprint_inventory(
    *,
    inventory_id: str,
    release: LoadedExecutionPublicRelease,
    workspace: LoadedExecutionWorkspaceBuild,
    catalogs: Sequence[LoadedFrozenDecisionCatalog],
) -> ExecutionIdentityFingerprintInventory:
    """Join exact verified task metadata to verified local catalogs without exporting identities."""

    _cross_bind_release_and_workspace(release, workspace)
    ordered_catalogs = tuple(sorted(catalogs, key=lambda item: item.binding.anchor_date))
    catalog_dates = tuple(item.binding.anchor_date for item in ordered_catalogs)
    if not ordered_catalogs or catalog_dates != tuple(sorted(set(catalog_dates))):
        raise ExecutionIdentityFingerprintError('identity inventory requires one catalog per unique anchor')
    by_anchor = {item.binding.anchor_date: item for item in ordered_catalogs}
    plan = _workspace_plan(workspace)
    plan_by_episode = {item.context.episode_id: item for item in plan.entries}
    task_by_episode = {item.context.episode_id: item for item in release.tasks}
    if set(plan_by_episode) != set(task_by_episode):
        raise ExecutionIdentityFingerprintError('workspace organizer plan and public release task universe differ')

    rows_by_anchor_nct: dict[date, dict[str, AactExecutionDecisionRow]] = {}
    core_indexes: dict[tuple[date, tuple[str, ...]], dict[bytes, set[str]]] = {}
    for catalog in ordered_catalogs:
        anchor = catalog.binding.anchor_date
        rows_by_anchor_nct[anchor] = {row.nct_id: row for row in catalog.rows}
        for omitted in ((), _COARSENED_OMITTED_FIELDS, _MINIMAL_OMITTED_FIELDS):
            index: dict[bytes, set[str]] = defaultdict(set)
            for row in catalog.rows:
                index[_row_core_signature(row, omitted)].add(row.nct_id)
            core_indexes[(anchor, tuple(omitted))] = index

    # The organizer-visible catalog covers every profile used by the exact task surfaces.  It is
    # reconstructed from verified task documents and the verified private alias maps, then used
    # only for counts; identities never enter the output model.
    visible_profile_index: dict[tuple[date, bytes], set[str]] = defaultdict(set)
    for episode_id, entry in plan_by_episode.items():
        task = task_by_episode[episode_id]
        profiles = (_target_profile(task), *_reference_profiles(task))
        binding_by_public_id = {item.public_trial_id: item for item in entry.alias_bindings}
        if {str(item['public_trial_id']) for item in profiles} != set(binding_by_public_id):
            raise ExecutionIdentityFingerprintError('visible profiles and organizer alias bindings differ')
        for profile in profiles:
            public_id = str(profile['public_trial_id'])
            private_binding = binding_by_public_id[public_id]
            catalog = by_anchor.get(private_binding.source_anchor_date)
            if catalog is None:
                raise ExecutionIdentityFingerprintError('visible profile has no verified same-anchor catalog')
            source_row = rows_by_anchor_nct[private_binding.source_anchor_date].get(private_binding.nct_id)
            if source_row is None or source_row.source_record_sha256 != private_binding.decision_source_record_sha256:
                raise ExecutionIdentityFingerprintError('visible profile alias does not bind a verified catalog row')
            if _profile_core_signature(profile) != _row_core_signature(source_row):
                raise ExecutionIdentityFingerprintError('visible profile registry core differs from its source row')
            visible_profile_index[(private_binding.source_anchor_date, _visible_profile_signature(profile))].add(
                private_binding.nct_id
            )

    cases: list[ExecutionIdentityFingerprintCase] = []
    for episode_id in sorted(task_by_episode):
        task = task_by_episode[episode_id]
        entry = plan_by_episode[episode_id]
        anchor = task.context.anchor_date
        catalog = by_anchor.get(anchor)
        if catalog is None:
            raise ExecutionIdentityFingerprintError('public task has no verified same-anchor decision catalog')
        target_profile = _target_profile(task)
        target_id = entry.organizer_private_nct_id
        target_row = rows_by_anchor_nct[anchor].get(target_id)
        if target_row is None or target_row.source_record_sha256 != entry.decision_source_record_sha256:
            raise ExecutionIdentityFingerprintError('public target does not bind its verified frozen catalog row')

        visible_matches = visible_profile_index[(anchor, _visible_profile_signature(target_profile))]
        exact_matches = core_indexes[(anchor, ())][_profile_core_signature(target_profile)]
        coarsened_matches = core_indexes[(anchor, _COARSENED_OMITTED_FIELDS)][
            _profile_core_signature(target_profile, _COARSENED_OMITTED_FIELDS)
        ]
        minimal_matches = core_indexes[(anchor, _MINIMAL_OMITTED_FIELDS)][
            _profile_core_signature(target_profile, _MINIMAL_OMITTED_FIELDS)
        ]
        if not all(
            target_id in values for values in (visible_matches, exact_matches, coarsened_matches, minimal_matches)
        ):
            raise ExecutionIdentityFingerprintError('true target is absent from a local fingerprint candidate set')

        surface = _released_task_surface(release, episode_id)
        binding = ExecutionCaseSurfaceBinding(
            episode_id=episode_id,
            task_context_sha256=task.context_sha256,
            public_surface_sha256=_sha256(surface),
        )
        maximum = EXECUTION_IDENTITY_FINGERPRINT_POLICY.near_unique_maximum_candidates
        cases.append(
            ExecutionIdentityFingerprintCase(
                surface_binding=binding,
                source_build_receipt_sha256=catalog.binding.source_build_receipt_sha256,
                exact_visible_profile_candidate_count=len(visible_matches),
                exact_registry_core_candidate_count=len(exact_matches),
                coarsened_registry_core_candidate_count=len(coarsened_matches),
                minimal_registry_core_candidate_count=len(minimal_matches),
                exact_visible_profile_unique=len(visible_matches) == 1,
                exact_registry_core_unique=len(exact_matches) == 1,
                coarsened_registry_core_unique=len(coarsened_matches) == 1,
                coarsened_registry_core_near_unique=len(coarsened_matches) <= maximum,
                minimal_registry_core_unique=len(minimal_matches) == 1,
                minimal_registry_core_near_unique=len(minimal_matches) <= maximum,
            )
        )
    cases_tuple = tuple(cases)
    aggregate = ExecutionIdentityFingerprintAggregate(
        case_count=len(cases_tuple),
        exact_visible_profile_unique_count=sum(item.exact_visible_profile_unique for item in cases_tuple),
        exact_registry_core_unique_count=sum(item.exact_registry_core_unique for item in cases_tuple),
        coarsened_registry_core_unique_count=sum(item.coarsened_registry_core_unique for item in cases_tuple),
        coarsened_registry_core_near_unique_count=sum(item.coarsened_registry_core_near_unique for item in cases_tuple),
        minimal_registry_core_unique_count=sum(item.minimal_registry_core_unique for item in cases_tuple),
        minimal_registry_core_near_unique_count=sum(item.minimal_registry_core_near_unique for item in cases_tuple),
        maximum_exact_registry_core_candidate_count=max(
            item.exact_registry_core_candidate_count for item in cases_tuple
        ),
        maximum_coarsened_registry_core_candidate_count=max(
            item.coarsened_registry_core_candidate_count for item in cases_tuple
        ),
        maximum_minimal_registry_core_candidate_count=max(
            item.minimal_registry_core_candidate_count for item in cases_tuple
        ),
    )
    return ExecutionIdentityFingerprintInventory(
        inventory_id=inventory_id,
        policy_sha256=execution_identity_fingerprint_policy_sha256(),
        public_release_receipt_sha256=release.receipt_sha256,
        public_release_tree_sha256=release.receipt.release_tree_sha256,
        source_workspace_receipt_sha256=release.receipt.source_workspace_receipt_sha256,
        source_workspace_public_tree_sha256=workspace.receipt.public_tree_sha256,
        frozen_catalogs=tuple(item.binding for item in ordered_catalogs),
        cases=cases_tuple,
        aggregate=aggregate,
    )


def build_execution_identity_fingerprint_inventory_from_roots(
    *,
    inventory_id: str,
    public_release_root: Path,
    expected_public_release_receipt_sha256: str,
    source_workspace_root: Path,
    expected_source_workspace_receipt_sha256: str,
    catalog_roots_and_pins: Sequence[tuple[Path, str]],
    expected_task_count: int,
) -> ExecutionIdentityFingerprintInventory:
    """Strict root-level entry point that requires every external pin before reading identities."""

    workspace = verify_execution_workspace_build(
        source_workspace_root,
        expected_receipt_sha256=expected_source_workspace_receipt_sha256,
    )
    release = verify_execution_public_release(
        public_release_root,
        expected_receipt_sha256=expected_public_release_receipt_sha256,
        expected_source_workspace_receipt_sha256=expected_source_workspace_receipt_sha256,
        expected_task_count=expected_task_count,
    )
    catalogs = tuple(
        load_frozen_decision_catalog(root, expected_build_receipt_sha256=pin) for root, pin in catalog_roots_and_pins
    )
    return build_execution_identity_fingerprint_inventory(
        inventory_id=inventory_id,
        release=release,
        workspace=workspace,
        catalogs=catalogs,
    )


def authenticate_execution_identity_fingerprint_inventory(
    inventory: ExecutionIdentityFingerprintInventory,
    *,
    key: bytes,
    expected_key_id: str,
) -> AuthenticatedExecutionIdentityFingerprintInventory:
    inventory = ExecutionIdentityFingerprintInventory.model_validate_json(canonical_json_bytes(inventory))
    key_id = execution_identity_fingerprint_key_id(key)
    if not hmac.compare_digest(key_id, expected_key_id):
        raise ExecutionIdentityFingerprintError('identity-inventory HMAC key does not match its expected key ID')
    return AuthenticatedExecutionIdentityFingerprintInventory(
        inventory=inventory,
        inventory_sha256=_model_sha256(inventory),
        hmac_key_id=key_id,
        inventory_hmac_sha256=_inventory_hmac(inventory, key),
    )


def write_authenticated_execution_identity_fingerprint_inventory(
    authenticated: AuthenticatedExecutionIdentityFingerprintInventory,
    *,
    output_root: Path,
) -> str:
    """Atomically write one read-only private/dev artifact and return its external SHA-256 pin."""

    authenticated = AuthenticatedExecutionIdentityFingerprintInventory.model_validate_json(
        canonical_json_bytes(authenticated)
    )
    payload = canonical_json_bytes(authenticated)
    target = output_root.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f'identity-fingerprint inventory already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.staging-', dir=target.parent))
    staging.chmod(0o700)
    try:
        artifact = staging / EXECUTION_IDENTITY_FINGERPRINT_ARTIFACT
        descriptor = os.open(artifact, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        fsync_directory(staging)
        staging.chmod(0o500)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _sha256(payload)


def load_authenticated_execution_identity_fingerprint_inventory(
    root: Path,
    *,
    expected_artifact_sha256: str,
    key: bytes,
    expected_key_id: str,
) -> AuthenticatedExecutionIdentityFingerprintInventory:
    """Verify external pin, exact private file inventory, schema, and private HMAC."""

    if re.fullmatch(_SHA256_PATTERN, expected_artifact_sha256) is None:
        raise ExecutionIdentityFingerprintError('expected identity-inventory artifact SHA-256 is invalid')
    if execution_identity_fingerprint_key_id(key) != expected_key_id:
        raise ExecutionIdentityFingerprintError('identity-inventory HMAC key does not match its expected key ID')
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ExecutionIdentityFingerprintError('identity-inventory root cannot be a symbolic link')
    resolved = supplied.resolve()
    if not resolved.is_dir() or stat.S_IMODE(os.stat(resolved, follow_symlinks=False).st_mode) != 0o500:
        raise ExecutionIdentityFingerprintError('identity-inventory root must be a private mode-0500 directory')
    files, directories = _exact_tree_inventory(resolved)
    if files != {EXECUTION_IDENTITY_FINGERPRINT_ARTIFACT} or directories:
        raise ExecutionIdentityFingerprintError('identity-inventory exact artifact inventory mismatch')
    artifact = resolved / EXECUTION_IDENTITY_FINGERPRINT_ARTIFACT
    if stat.S_IMODE(os.stat(artifact, follow_symlinks=False).st_mode) != 0o400:
        raise ExecutionIdentityFingerprintError('identity-inventory artifact must have mode 0400')
    payload = _read_exact_file(artifact, maximum_bytes=_MAX_RECEIPT_BYTES)
    if not hmac.compare_digest(_sha256(payload), expected_artifact_sha256):
        raise ExecutionIdentityFingerprintError('identity-inventory artifact does not match its external pin')
    try:
        authenticated = AuthenticatedExecutionIdentityFingerprintInventory.model_validate_json(payload)
    except ValueError as error:
        raise ExecutionIdentityFingerprintError(f'invalid authenticated identity inventory: {error}') from error
    if payload != canonical_json_bytes(authenticated):
        raise ExecutionIdentityFingerprintError('identity-inventory artifact must use canonical JSON')
    if authenticated.hmac_key_id != expected_key_id or not hmac.compare_digest(
        authenticated.inventory_hmac_sha256,
        _inventory_hmac(authenticated.inventory, key),
    ):
        raise ExecutionIdentityFingerprintError('identity-inventory private HMAC authentication failed')
    return authenticated


__all__ = [
    'AUTHENTICATED_EXECUTION_IDENTITY_FINGERPRINT_INVENTORY_SCHEMA_VERSION',
    'EXECUTION_IDENTITY_FINGERPRINT_ARTIFACT',
    'EXECUTION_IDENTITY_FINGERPRINT_INVENTORY_SCHEMA_VERSION',
    'EXECUTION_IDENTITY_FINGERPRINT_POLICY',
    'EXECUTION_IDENTITY_FINGERPRINT_POLICY_SCHEMA_VERSION',
    'AuthenticatedExecutionIdentityFingerprintInventory',
    'ExecutionIdentityFingerprintAggregate',
    'ExecutionIdentityFingerprintCase',
    'ExecutionIdentityFingerprintError',
    'ExecutionIdentityFingerprintInventory',
    'ExecutionIdentityFingerprintPolicy',
    'FrozenDecisionCatalogBinding',
    'LoadedFrozenDecisionCatalog',
    'authenticate_execution_identity_fingerprint_inventory',
    'build_execution_identity_fingerprint_inventory',
    'build_execution_identity_fingerprint_inventory_from_roots',
    'execution_identity_fingerprint_key_id',
    'execution_identity_fingerprint_policy_sha256',
    'load_authenticated_execution_identity_fingerprint_inventory',
    'load_frozen_decision_catalog',
    'write_authenticated_execution_identity_fingerprint_inventory',
]
