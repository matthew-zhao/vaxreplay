"""Development-only bridge from public Lane A tasks to Agentic Replay execution.

The workspace adapter materializes only model-visible task bytes and host authentication evidence;
it never accepts an organizer or private workspace root.  The collector retains every supplied
terminal guest-RPC artifact and emits a cohort submission only when the fixed evaluation manifest
has exactly one authenticated, successful clinical submission per task.

This module does not claim official isolation, admission, one-attempt enforcement, contamination
control, or absence of model-weight leakage.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.guest_rpc import (
    AuthenticatedGuestRpcSession,
    GuestRpcTerminalStatus,
    verify_authenticated_guest_rpc_session,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.schema import AgenticMediaType, AgenticWorkspaceEntry
from vaxreplay.agentic.task_protocol import (
    AgenticTaskInvocation,
    agentic_task_invocation_sha256,
    validate_submission_for_invocation,
)
from vaxreplay.agentic.workspace import AgenticLogicalWorkspaceBroker, model_visible_surface_bytes
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_aggregation import (
    ExecutionCohortManifest,
    ExecutionCohortSubmission,
    execution_cohort_manifest_sha256,
    make_execution_cohort_submission,
)
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission, ExecutionTask

CLINICAL_AGENTIC_WORKSPACE_MANIFEST_SCHEMA_VERSION = 'vaxreplay.clinical-agentic-workspace-manifest.dev-v0.1'
CLINICAL_AGENTIC_WORKSPACE_RECEIPT_SCHEMA_VERSION = 'vaxreplay.clinical-agentic-workspace-receipt.dev-v0.1'
AUTHENTICATED_CLINICAL_AGENTIC_WORKSPACE_SCHEMA_VERSION = 'vaxreplay.authenticated-clinical-agentic-workspace.dev-v0.1'
CLINICAL_EXECUTION_COLLECTION_SCHEMA_VERSION = 'vaxreplay.clinical-execution-collection.dev-v0.1'
CLINICAL_EXECUTION_RUN_EXPECTATION_SCHEMA_VERSION = 'vaxreplay.clinical-execution-run-expectation.dev-v0.1'
AUTHENTICATED_CLINICAL_EXECUTION_COLLECTION_SCHEMA_VERSION = (
    'vaxreplay.authenticated-clinical-execution-collection.dev-v0.1'
)

_WORKSPACE_HMAC_DOMAIN = b'vaxreplay.clinical-agentic-workspace-receipt.dev-v0.1\x00'
_WORKSPACE_KEY_ID_DOMAIN = b'vaxreplay.clinical-agentic-workspace-key-id.dev-v0.1\x00'
_COLLECTION_HMAC_DOMAIN = b'vaxreplay.clinical-execution-collection.dev-v0.1\x00'
_COLLECTION_KEY_ID_DOMAIN = b'vaxreplay.clinical-execution-collection-key-id.dev-v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_NCT_PATTERN = re.compile(rb'NCT\d{8}', re.IGNORECASE)
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
_HOST_FILES = {'workspace-manifest.json', 'workspace-receipt.json'}


class ClinicalExecutionBridgeError(ValueError):
    """The public workspace or terminal collection failed a trusted bridge check."""


class ClinicalAgenticWorkspaceManifest(StrictModel):
    schema_version: Literal['vaxreplay.clinical-agentic-workspace-manifest.dev-v0.1'] = (
        CLINICAL_AGENTIC_WORKSPACE_MANIFEST_SCHEMA_VERSION
    )
    workspace_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_protocol: Literal[AgenticResponseProtocol.CLINICAL_EXECUTION] = AgenticResponseProtocol.CLINICAL_EXECUTION
    entries: tuple[AgenticWorkspaceEntry, ...] = Field(min_length=3)
    public_only: Literal[True] = True
    labels_present: Literal[False] = False
    organizer_mapping_present: Literal[False] = False
    private_root_present: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False

    @field_validator('entries')
    @classmethod
    def validate_entries(cls, value: tuple[AgenticWorkspaceEntry, ...]) -> tuple[AgenticWorkspaceEntry, ...]:
        paths = tuple(item.path for item in value)
        if paths != tuple(sorted(set(paths))):
            raise ValueError('clinical workspace entries must use unique canonical path order')
        if not {'TASK.json', 'TASK.md', 'source-catalog.json'}.issubset(paths):
            raise ValueError('clinical workspace is missing a required public task file')
        folded = tuple(path.casefold() for path in paths)
        if len(folded) != len(set(folded)):
            raise ValueError('clinical workspace paths cannot collide under case folding')
        return value


class ClinicalAgenticWorkspaceReceipt(StrictModel):
    schema_version: Literal['vaxreplay.clinical-agentic-workspace-receipt.dev-v0.1'] = (
        CLINICAL_AGENTIC_WORKSPACE_RECEIPT_SCHEMA_VERSION
    )
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_protocol: Literal[AgenticResponseProtocol.CLINICAL_EXECUTION] = AgenticResponseProtocol.CLINICAL_EXECUTION
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    public_file_count: int = Field(ge=3)
    public_only: Literal[True] = True
    exact_inventory_verified: Literal[True] = True
    authenticated_guest_rpc_protocol_supported: Literal[True] = True
    organizer_or_private_root_opened: Literal[False] = False
    development_only: Literal[True] = True
    official_execution_qualified: Literal[False] = False
    leaderboard_admitted: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    upstream_public_release_receipt_verified: Literal[False] = False
    ancestor_symlink_race_hardened: Literal[False] = False


class AuthenticatedClinicalAgenticWorkspace(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-clinical-agentic-workspace.dev-v0.1'] = (
        AUTHENTICATED_CLINICAL_AGENTIC_WORKSPACE_SCHEMA_VERSION
    )
    receipt: ClinicalAgenticWorkspaceReceipt
    receipt_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class LoadedClinicalAgenticWorkspace:
    root: Path
    input_root: Path
    task: ExecutionTask
    manifest: ClinicalAgenticWorkspaceManifest
    manifest_sha256: str
    invocation: AgenticTaskInvocation
    authenticated_receipt: AuthenticatedClinicalAgenticWorkspace
    authenticated_receipt_sha256: str
    model_visible_surface: bytes

    def brokered_surface(self) -> AgenticLogicalWorkspaceBroker:
        return AgenticLogicalWorkspaceBroker(
            entries=self.manifest.entries,
            surface=self.model_visible_surface,
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_sha256(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _run_expectations_bytes(expectations: Iterable[ClinicalExecutionRunExpectation]) -> bytes:
    return canonical_json_bytes([item.model_dump(mode='json') for item in expectations])


def _require_key(key: bytes, label: str) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ClinicalExecutionBridgeError(f'{label} must contain at least 32 bytes')


def clinical_workspace_receipt_key_id(key: bytes) -> str:
    _require_key(key, 'clinical workspace receipt key')
    return _sha256(_WORKSPACE_KEY_ID_DOMAIN + key)


def clinical_collection_receipt_key_id(key: bytes) -> str:
    _require_key(key, 'clinical collection receipt key')
    return _sha256(_COLLECTION_KEY_ID_DOMAIN + key)


def _workspace_receipt_hmac(receipt: ClinicalAgenticWorkspaceReceipt, key: bytes) -> str:
    _require_key(key, 'clinical workspace receipt key')
    return hmac.new(key, _WORKSPACE_HMAC_DOMAIN + canonical_json_bytes(receipt), hashlib.sha256).hexdigest()


def _tree_sha256(files: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        path_bytes = path.encode('utf-8')
        content = files[path]
        digest.update(len(path_bytes).to_bytes(8, 'big'))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return digest.hexdigest()


def _task_markdown(task: ExecutionTask) -> bytes:
    context = task.context
    text = (
        '# Registry-observed trial-execution replay\n\n'
        f'Episode: `{context.episode_id}`\n\n'
        f'Decision anchor: `{context.anchor_date.isoformat()}`\n\n'
        'Use only the files exposed by the logical workspace. Forecast the registry-observed '
        'outcomes at the fixed 48-month horizon and submit exactly one '
        '`vaxreplay.clinical-execution-submission.dev-v0.1` response.\n'
    )
    return text.encode('utf-8')


def _visible_files(task: ExecutionTask) -> tuple[dict[str, bytes], dict[str, AgenticMediaType]]:
    files: dict[str, bytes] = {
        'TASK.json': canonical_json_bytes(task),
        'TASK.md': _task_markdown(task),
    }
    media_types = {
        'TASK.json': AgenticMediaType.JSON,
        'TASK.md': AgenticMediaType.MARKDOWN,
    }
    sources: list[dict[str, object]] = []
    for index, document in enumerate(task.context.cutoff_documents, start=1):
        path = f'sources/source-{index:03d}.txt'
        content = document.body.encode('utf-8')
        files[path] = content
        media_types[path] = AgenticMediaType.TEXT
        sources.append(
            {
                'source_id': f'source-{index:03d}',
                'document_id': document.document_id,
                'path': path,
                'available_on': document.available_on.isoformat(),
                'sha256': _sha256(content),
                'byte_count': len(content),
            }
        )
    catalog = {
        'schema_version': 'vaxreplay.clinical-agentic-source-catalog.dev-v0.1',
        'sources': sources,
    }
    files['source-catalog.json'] = canonical_json_bytes(catalog)
    media_types['source-catalog.json'] = AgenticMediaType.JSON
    return files, media_types


def _entries(
    files: Mapping[str, bytes],
    media_types: Mapping[str, AgenticMediaType],
) -> tuple[AgenticWorkspaceEntry, ...]:
    return tuple(
        AgenticWorkspaceEntry(
            path=path,
            sha256=_sha256(files[path]),
            byte_count=len(files[path]),
            media_type=media_types[path],
            provenance_node_id=f'clinical-public:{path}',
        )
        for path in sorted(files)
    )


def _reject_private_identity(files: Mapping[str, bytes]) -> None:
    aggregate_bytes = 0
    for path, payload in files.items():
        aggregate_bytes += len(payload)
        if not payload or len(payload) > _MAX_FILE_BYTES or aggregate_bytes > _MAX_WORKSPACE_BYTES:
            raise ClinicalExecutionBridgeError(f'public clinical workspace file is empty or oversized: {path}')
        if b'\x00' in payload or _NCT_PATTERN.search(payload):
            raise ClinicalExecutionBridgeError('public clinical workspace exposes a registry identifier or NUL')
        try:
            payload.decode('utf-8')
        except UnicodeDecodeError as error:
            raise ClinicalExecutionBridgeError('public clinical workspace must contain only UTF-8 files') from error


def _write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_clinical_agentic_workspace(
    *,
    task: ExecutionTask,
    workspace_id: str,
    output_root: Path,
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> LoadedClinicalAgenticWorkspace:
    """Authenticate an exact projection of one caller-supplied Lane A task.

    This does not authenticate the task's upstream public-release provenance or admit it for an
    official run.  Cohort binding is enforced later by the collection step.
    """

    task = ExecutionTask.model_validate_json(canonical_json_bytes(task))
    if clinical_workspace_receipt_key_id(receipt_key) != expected_receipt_key_id:
        raise ClinicalExecutionBridgeError('clinical workspace receipt key does not match its expected key ID')
    files, media_types = _visible_files(task)
    _reject_private_identity(files)
    entries = _entries(files, media_types)
    surface = model_visible_surface_bytes(dict(files))
    manifest = ClinicalAgenticWorkspaceManifest(
        workspace_id=workspace_id,
        episode_id=task.context.episode_id,
        target_trial_id=task.context.target_trial_id,
        task_sha256=_model_sha256(task),
        workspace_tree_sha256=_tree_sha256(files),
        model_visible_surface_sha256=_sha256(surface),
        entries=entries,
    )
    manifest_sha256 = _model_sha256(manifest)
    invocation = AgenticTaskInvocation.from_task(task, workspace_manifest_sha256=manifest_sha256)
    receipt = ClinicalAgenticWorkspaceReceipt(
        workspace_manifest_sha256=manifest_sha256,
        workspace_tree_sha256=manifest.workspace_tree_sha256,
        model_visible_surface_sha256=manifest.model_visible_surface_sha256,
        task_sha256=manifest.task_sha256,
        task_invocation_sha256=agentic_task_invocation_sha256(invocation),
        receipt_key_id=expected_receipt_key_id,
        public_file_count=len(files),
    )
    authenticated = AuthenticatedClinicalAgenticWorkspace(
        receipt=receipt,
        receipt_hmac_sha256=_workspace_receipt_hmac(receipt, receipt_key),
    )
    target = output_root.expanduser().resolve()
    if target.exists():
        raise ClinicalExecutionBridgeError(f'clinical Agentic workspace already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    staging.chmod(0o700)
    try:
        for path, payload in files.items():
            _write_file(staging / 'input' / path, payload)
        _write_file(staging / 'workspace-manifest.json', canonical_json_bytes(manifest))
        _write_file(staging / 'workspace-receipt.json', canonical_json_bytes(authenticated))
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_clinical_agentic_workspace(
        target,
        expected_authenticated_receipt_sha256=_model_sha256(authenticated),
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
    )


def _read_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ClinicalExecutionBridgeError(f'cannot open clinical workspace file: {path.name}') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size <= 0
            or metadata.st_size > maximum_bytes
        ):
            raise ClinicalExecutionBridgeError('clinical workspace contains an unsafe or oversized file')
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes - len(content) + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ClinicalExecutionBridgeError('clinical workspace file exceeds its byte limit')
    finally:
        os.close(descriptor)


def load_clinical_agentic_workspace(
    root: Path,
    *,
    expected_authenticated_receipt_sha256: str,
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> LoadedClinicalAgenticWorkspace:
    """Authenticate and reconstruct a public-only clinical logical workspace."""

    if clinical_workspace_receipt_key_id(receipt_key) != expected_receipt_key_id:
        raise ClinicalExecutionBridgeError('clinical workspace receipt key does not match its expected key ID')
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ClinicalExecutionBridgeError('clinical workspace root cannot be a symbolic link')
    resolved = supplied.resolve()
    if not resolved.is_dir() or stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise ClinicalExecutionBridgeError('clinical workspace root must be a private mode-0700 directory')
    descendants = tuple(resolved.rglob('*'))
    for path in descendants:
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            raise ClinicalExecutionBridgeError('clinical workspace cannot contain symbolic links')
        if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
            raise ClinicalExecutionBridgeError('clinical workspace contains a non-regular descendant')
    receipt_bytes = _read_file(resolved / 'workspace-receipt.json', _MAX_FILE_BYTES)
    if not hmac.compare_digest(_sha256(receipt_bytes), expected_authenticated_receipt_sha256):
        raise ClinicalExecutionBridgeError('clinical workspace receipt does not match its external pin')
    manifest_bytes = _read_file(resolved / 'workspace-manifest.json', _MAX_FILE_BYTES)
    try:
        authenticated = AuthenticatedClinicalAgenticWorkspace.model_validate_json(receipt_bytes)
        manifest = ClinicalAgenticWorkspaceManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise ClinicalExecutionBridgeError('clinical workspace host evidence has an invalid schema') from error
    if canonical_json_bytes(authenticated) != receipt_bytes or canonical_json_bytes(manifest) != manifest_bytes:
        raise ClinicalExecutionBridgeError('clinical workspace host evidence must use canonical JSON')
    if authenticated.receipt.receipt_key_id != expected_receipt_key_id or not hmac.compare_digest(
        authenticated.receipt_hmac_sha256,
        _workspace_receipt_hmac(authenticated.receipt, receipt_key),
    ):
        raise ClinicalExecutionBridgeError('clinical workspace receipt authentication failed')
    manifest_sha256 = _sha256(manifest_bytes)
    if authenticated.receipt.workspace_manifest_sha256 != manifest_sha256:
        raise ClinicalExecutionBridgeError('clinical workspace receipt does not bind its manifest')

    expected_paths = {f'input/{item.path}' for item in manifest.entries} | _HOST_FILES
    observed_paths = {path.relative_to(resolved).as_posix() for path in descendants if path.is_file()}
    if observed_paths != expected_paths:
        raise ClinicalExecutionBridgeError('clinical workspace exact file inventory mismatch')
    expected_directories = {'input'} | {
        str(Path(path).parent.as_posix()) for path in expected_paths if Path(path).parent.as_posix() not in {'', '.'}
    }
    observed_directories = {path.relative_to(resolved).as_posix() for path in descendants if path.is_dir()}
    if observed_directories != expected_directories or any(
        stat.S_IMODE((resolved / path).stat().st_mode) != 0o700 for path in observed_directories
    ):
        raise ClinicalExecutionBridgeError('clinical workspace exact directory inventory mismatch')
    files: dict[str, bytes] = {}
    for entry in manifest.entries:
        payload = _read_file(resolved / 'input' / entry.path, min(entry.byte_count, _MAX_FILE_BYTES))
        if (len(payload), _sha256(payload)) != (entry.byte_count, entry.sha256):
            raise ClinicalExecutionBridgeError('clinical workspace public file does not match its manifest')
        files[entry.path] = payload
    _reject_private_identity(files)
    surface = model_visible_surface_bytes(files)
    if (
        _tree_sha256(files),
        _sha256(surface),
        len(files),
    ) != (
        manifest.workspace_tree_sha256,
        manifest.model_visible_surface_sha256,
        authenticated.receipt.public_file_count,
    ):
        raise ClinicalExecutionBridgeError('clinical workspace hashes or counts do not match exact public bytes')
    try:
        task = ExecutionTask.model_validate_json(files['TASK.json'])
    except ValueError as error:
        raise ClinicalExecutionBridgeError('clinical workspace TASK.json is invalid') from error
    expected_files, expected_media_types = _visible_files(task)
    if files != expected_files or manifest.entries != _entries(expected_files, expected_media_types):
        raise ClinicalExecutionBridgeError('clinical workspace files are not the deterministic task projection')
    if (
        manifest.episode_id,
        manifest.target_trial_id,
        manifest.task_sha256,
    ) != (
        task.context.episode_id,
        task.context.target_trial_id,
        _model_sha256(task),
    ):
        raise ClinicalExecutionBridgeError('clinical workspace manifest is bound to a different task')
    invocation = AgenticTaskInvocation.from_task(task, workspace_manifest_sha256=manifest_sha256)
    receipt = authenticated.receipt
    if (
        receipt.workspace_tree_sha256,
        receipt.model_visible_surface_sha256,
        receipt.task_sha256,
        receipt.task_invocation_sha256,
    ) != (
        manifest.workspace_tree_sha256,
        manifest.model_visible_surface_sha256,
        manifest.task_sha256,
        agentic_task_invocation_sha256(invocation),
    ):
        raise ClinicalExecutionBridgeError('clinical workspace receipt does not bind its exact invocation')
    return LoadedClinicalAgenticWorkspace(
        root=resolved,
        input_root=resolved / 'input',
        task=task,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        invocation=invocation,
        authenticated_receipt=authenticated,
        authenticated_receipt_sha256=_sha256(receipt_bytes),
        model_visible_surface=surface,
    )


class ClinicalCollectionFailureCode(str, enum.Enum):
    MISSING_TASK_ATTEMPT = 'missing_task_attempt'
    DUPLICATE_TASK_ATTEMPT = 'duplicate_task_attempt'
    EXTRA_TASK_ATTEMPT = 'extra_task_attempt'
    UNAUTHENTICATED_ATTEMPT = 'unauthenticated_attempt'
    RUN_EXPECTATION_MISMATCH = 'run_expectation_mismatch'
    TASK_BINDING_MISMATCH = 'task_binding_mismatch'
    TASK_TERMINAL_FAILURE = 'task_terminal_failure'
    INVALID_TERMINAL_SUBMISSION = 'invalid_terminal_submission'


class ClinicalExecutionRunExpectation(StrictModel):
    """Externally supplied development-run pins for exactly one fixed cohort task.

    These values must come from the trusted run scheduler/configuration, not from the terminal
    session being collected.  Their presence still does not prove reservation admission,
    provider identity, harness identity, or model routing.
    """

    schema_version: Literal['vaxreplay.clinical-execution-run-expectation.dev-v0.1'] = (
        CLINICAL_EXECUTION_RUN_EXPECTATION_SCHEMA_VERSION
    )
    episode_id: str = Field(min_length=1)
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    rpc_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_invocation_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_broker_contract_version: str = Field(min_length=1)
    workspace_broker_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_capability_id: str = Field(pattern=_SHA256_PATTERN)
    gateway_grant_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_peer_cid: int = Field(ge=3, le=2**32 - 1)
    rpc_port: int = Field(ge=1, le=2**32 - 1)
    guest_rpc_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    response_protocol: Literal[AgenticResponseProtocol.CLINICAL_EXECUTION] = AgenticResponseProtocol.CLINICAL_EXECUTION
    development_only: Literal[True] = True
    external_reservation_verified: Literal[False] = False
    provider_identity_verified: Literal[False] = False
    harness_identity_verified: Literal[False] = False
    model_route_verified: Literal[False] = False


class ClinicalExecutionTerminalAttempt(StrictModel):
    record_index: int = Field(ge=0)
    claimed_episode_id: str = Field(min_length=1)
    session_sha256: str = Field(pattern=_SHA256_PATTERN)
    session: AuthenticatedGuestRpcSession
    authenticated: bool
    run_expectation_verified: bool
    task_binding_verified: bool
    successful_terminal_submission: bool
    issues: tuple[ClinicalCollectionFailureCode, ...] = ()

    @model_validator(mode='after')
    def validate_record(self) -> Self:
        if _model_sha256(self.session) != self.session_sha256:
            raise ValueError('terminal attempt does not bind its exact guest-RPC artifact')
        if self.issues != tuple(sorted(set(self.issues), key=lambda item: item.value)):
            raise ValueError('terminal attempt issues must be unique and sorted')
        if self.successful_terminal_submission and (
            not self.authenticated or not self.run_expectation_verified or not self.task_binding_verified or self.issues
        ):
            raise ValueError('successful terminal attempts require every trusted validation')
        return self


class ClinicalExecutionCollectionStatus(str, enum.Enum):
    COMPLETED = 'completed'
    FAILED = 'failed'


class ClinicalExecutionCollection(StrictModel):
    schema_version: Literal['vaxreplay.clinical-execution-collection.dev-v0.1'] = (
        CLINICAL_EXECUTION_COLLECTION_SCHEMA_VERSION
    )
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    cohort_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_split: Split
    expected_task_count: int = Field(gt=0)
    run_expectations_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_expectations: tuple[ClinicalExecutionRunExpectation, ...]
    supplied_attempt_count: int = Field(ge=0)
    attempts_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempts: tuple[ClinicalExecutionTerminalAttempt, ...]
    status: ClinicalExecutionCollectionStatus
    failure_codes: tuple[ClinicalCollectionFailureCode, ...] = ()
    cohort_submission: ExecutionCohortSubmission | None = None
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    terminal_attempts_retained: Literal[True] = True
    exact_one_submission_per_manifest_task_required: Literal[True] = True
    run_expectations_bound: Literal[True] = True
    partial_cohort_submission_emitted: Literal[False] = False
    one_attempt_registry_verified: Literal[False] = False
    external_run_reservations_verified: Literal[False] = False
    provider_identity_verified: Literal[False] = False
    harness_identity_verified: Literal[False] = False
    model_route_verified: Literal[False] = False
    production_run_collector: Literal[False] = False
    development_only: Literal[True] = True
    official_execution_qualified: Literal[False] = False
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_collection(self) -> Self:
        if tuple(item.episode_id for item in self.run_expectations) != tuple(
            sorted(item.episode_id for item in self.run_expectations)
        ) or len({item.episode_id for item in self.run_expectations}) != len(self.run_expectations):
            raise ValueError('run expectations must have unique canonical episode order')
        if self.expected_task_count != len(self.run_expectations):
            raise ValueError('run expectations must exactly cover the expected task count')
        if self.run_expectations_sha256 != _sha256(_run_expectations_bytes(self.run_expectations)):
            raise ValueError('collection does not bind its exact run expectations')
        if tuple(item.record_index for item in self.attempts) != tuple(range(len(self.attempts))):
            raise ValueError('terminal attempt record indexes must be contiguous')
        if self.supplied_attempt_count != len(self.attempts):
            raise ValueError('supplied attempt count does not match retained terminal attempts')
        if self.attempts_sha256 != _sha256(
            canonical_json_bytes([item.model_dump(mode='json') for item in self.attempts])
        ):
            raise ValueError('collection does not bind its retained terminal attempts')
        if self.failure_codes != tuple(sorted(set(self.failure_codes), key=lambda item: item.value)):
            raise ValueError('collection failure codes must be unique and sorted')
        if self.status == ClinicalExecutionCollectionStatus.COMPLETED:
            if (
                self.failure_codes
                or self.cohort_submission is None
                or len(self.attempts) != self.expected_task_count
                or not all(item.successful_terminal_submission for item in self.attempts)
            ):
                raise ValueError('completed collection requires exact successful task coverage')
        elif not self.failure_codes or self.cohort_submission is not None:
            raise ValueError('failed collection requires failures and cannot emit a cohort submission')
        return self


class AuthenticatedClinicalExecutionCollection(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-clinical-execution-collection.dev-v0.1'] = (
        AUTHENTICATED_CLINICAL_EXECUTION_COLLECTION_SCHEMA_VERSION
    )
    collection: ClinicalExecutionCollection
    collection_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


def _collection_hmac(collection: ClinicalExecutionCollection, key: bytes) -> str:
    _require_key(key, 'clinical collection receipt key')
    return hmac.new(key, _COLLECTION_HMAC_DOMAIN + canonical_json_bytes(collection), hashlib.sha256).hexdigest()


def _claimed_episode(session: AuthenticatedGuestRpcSession) -> str:
    task = session.task_invocation.task
    if isinstance(task, ExecutionTask):
        return task.context.episode_id
    return task.episode_id


def _validate_collection_workspaces(
    manifest: ExecutionCohortManifest,
    workspaces: Iterable[LoadedClinicalAgenticWorkspace],
    workspace_receipt_keys_by_id: Mapping[str, bytes],
) -> dict[str, LoadedClinicalAgenticWorkspace]:
    by_episode: dict[str, LoadedClinicalAgenticWorkspace] = {}
    for supplied_workspace in workspaces:
        key_id = supplied_workspace.authenticated_receipt.receipt.receipt_key_id
        key = workspace_receipt_keys_by_id.get(key_id)
        if key is None:
            raise ClinicalExecutionBridgeError('clinical collection lacks a workspace authentication key')
        workspace = load_clinical_agentic_workspace(
            supplied_workspace.root,
            expected_authenticated_receipt_sha256=supplied_workspace.authenticated_receipt_sha256,
            receipt_key=key,
            expected_receipt_key_id=key_id,
        )
        episode_id = workspace.task.context.episode_id
        if episode_id in by_episode:
            raise ClinicalExecutionBridgeError('clinical collection workspaces contain a duplicate task')
        by_episode[episode_id] = workspace
    expected = {item.episode_id: item for item in manifest.tasks}
    if set(by_episode) != set(expected):
        raise ClinicalExecutionBridgeError('clinical collection workspaces do not exactly cover the manifest')
    for episode_id, workspace in by_episode.items():
        binding = expected[episode_id]
        if (
            workspace.task.context.target_trial_id,
            workspace.task.context_sha256,
            _model_sha256(workspace.task),
            workspace.invocation.workspace_manifest_sha256,
        ) != (
            binding.target_trial_id,
            binding.task_context_sha256,
            binding.task_sha256,
            workspace.manifest_sha256,
        ):
            raise ClinicalExecutionBridgeError('clinical workspace does not match its fixed manifest task')
    return by_episode


def _validate_run_expectations(
    manifest: ExecutionCohortManifest,
    workspace_by_episode: Mapping[str, LoadedClinicalAgenticWorkspace],
    run_expectations: Iterable[ClinicalExecutionRunExpectation],
) -> tuple[ClinicalExecutionRunExpectation, ...]:
    materialized = tuple(
        ClinicalExecutionRunExpectation.model_validate_json(canonical_json_bytes(item)) for item in run_expectations
    )
    if len({item.episode_id for item in materialized}) != len(materialized):
        raise ClinicalExecutionBridgeError('clinical run expectations contain a duplicate task')
    by_episode = {item.episode_id: item for item in materialized}
    expected_ids = {item.episode_id for item in manifest.tasks}
    if set(by_episode) != expected_ids:
        raise ClinicalExecutionBridgeError('clinical run expectations do not exactly cover the fixed manifest')
    for episode_id, expectation in by_episode.items():
        workspace = workspace_by_episode[episode_id]
        broker = workspace.brokered_surface()
        if (
            expectation.workspace_manifest_sha256,
            expectation.workspace_tree_sha256,
            expectation.model_visible_surface_sha256,
            expectation.task_invocation_sha256,
            expectation.workspace_broker_contract_version,
            expectation.workspace_broker_contract_sha256,
        ) != (
            workspace.manifest_sha256,
            workspace.manifest.workspace_tree_sha256,
            workspace.manifest.model_visible_surface_sha256,
            agentic_task_invocation_sha256(workspace.invocation),
            broker.contract_version,
            broker.contract_sha256,
        ):
            raise ClinicalExecutionBridgeError(
                'clinical run expectation does not bind the fixed task workspace and broker'
            )
    return tuple(by_episode[episode_id] for episode_id in sorted(by_episode))


def _session_matches_run_expectation(
    session: AuthenticatedGuestRpcSession,
    expectation: ClinicalExecutionRunExpectation,
    workspace: LoadedClinicalAgenticWorkspace,
) -> bool:
    seal = session.seal
    broker = workspace.brokered_surface()
    return (
        seal.run_id,
        seal.attempt_reservation_sha256,
        seal.execution_policy_sha256,
        seal.worker_spec_sha256,
        seal.rpc_policy_sha256,
        seal.workspace_manifest_sha256,
        seal.workspace_tree_sha256,
        seal.model_visible_surface_sha256,
        seal.task_invocation_sha256,
        seal.response_protocol,
        seal.workspace_broker_contract_version,
        seal.workspace_broker_contract_sha256,
        seal.gateway_capability_id,
        seal.gateway_grant_sha256,
        seal.expected_peer_cid,
        seal.observed_peer_cid,
        seal.rpc_port,
        seal.receipt_key_id,
    ) == (
        expectation.run_id,
        expectation.attempt_reservation_sha256,
        expectation.execution_policy_sha256,
        expectation.worker_spec_sha256,
        expectation.rpc_policy_sha256,
        workspace.manifest_sha256,
        workspace.manifest.workspace_tree_sha256,
        workspace.manifest.model_visible_surface_sha256,
        agentic_task_invocation_sha256(workspace.invocation),
        AgenticResponseProtocol.CLINICAL_EXECUTION,
        broker.contract_version,
        broker.contract_sha256,
        expectation.gateway_capability_id,
        expectation.gateway_grant_sha256,
        expectation.expected_peer_cid,
        expectation.expected_peer_cid,
        expectation.rpc_port,
        expectation.guest_rpc_receipt_key_id,
    )


def collect_clinical_execution_sessions(
    *,
    manifest: ExecutionCohortManifest,
    workspaces: Iterable[LoadedClinicalAgenticWorkspace],
    run_expectations: Iterable[ClinicalExecutionRunExpectation],
    sessions: Iterable[AuthenticatedGuestRpcSession],
    workspace_receipt_keys_by_id: Mapping[str, bytes],
    guest_rpc_receipt_keys_by_id: Mapping[str, bytes],
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> AuthenticatedClinicalExecutionCollection:
    """Collect a development run against external pins; this is not production admission."""

    manifest = ExecutionCohortManifest.model_validate_json(canonical_json_bytes(manifest))
    workspace_by_episode = _validate_collection_workspaces(
        manifest,
        workspaces,
        workspace_receipt_keys_by_id,
    )
    expectations = _validate_run_expectations(manifest, workspace_by_episode, run_expectations)
    expectation_by_episode = {item.episode_id: item for item in expectations}
    if clinical_collection_receipt_key_id(receipt_key) != expected_receipt_key_id:
        raise ClinicalExecutionBridgeError('clinical collection key does not match its expected key ID')
    materialized = tuple(sessions)
    ordered_sessions = tuple(sorted(materialized, key=lambda item: (_claimed_episode(item), _model_sha256(item))))
    claimed_counts = Counter(_claimed_episode(item) for item in ordered_sessions)
    expected_ids = set(workspace_by_episode)
    claimed_ids = set(claimed_counts)
    global_failures: set[ClinicalCollectionFailureCode] = set()
    if expected_ids - claimed_ids:
        global_failures.add(ClinicalCollectionFailureCode.MISSING_TASK_ATTEMPT)
    if claimed_ids - expected_ids:
        global_failures.add(ClinicalCollectionFailureCode.EXTRA_TASK_ATTEMPT)
    if any(count > 1 for count in claimed_counts.values()):
        global_failures.add(ClinicalCollectionFailureCode.DUPLICATE_TASK_ATTEMPT)

    records: list[ClinicalExecutionTerminalAttempt] = []
    submissions: list[ExecutionSubmission] = []
    for record_index, session in enumerate(ordered_sessions):
        episode_id = _claimed_episode(session)
        workspace = workspace_by_episode.get(episode_id)
        issues: set[ClinicalCollectionFailureCode] = set()
        authenticated = False
        run_expectation_verified = False
        binding_verified = False
        expectation = expectation_by_episode.get(episode_id)
        key = guest_rpc_receipt_keys_by_id.get(session.seal.receipt_key_id)
        if key is None:
            issues.add(ClinicalCollectionFailureCode.UNAUTHENTICATED_ATTEMPT)
        elif workspace is None or expectation is None:
            issues.add(ClinicalCollectionFailureCode.EXTRA_TASK_ATTEMPT)
        else:
            try:
                # First authenticate and fully rederive the artifact on its own internally bound
                # values.  Trusted external run/workspace pins are checked separately below so a
                # valid artifact from the wrong run is distinguishable from forged evidence.
                verify_authenticated_guest_rpc_session(
                    session,
                    receipt_key=key,
                    expected_receipt_key_id=session.seal.receipt_key_id,
                    expected_run_id=session.seal.run_id,
                    expected_workspace_manifest_sha256=session.seal.workspace_manifest_sha256,
                    expected_execution_policy_sha256=session.seal.execution_policy_sha256,
                    expected_task_invocation_sha256=session.seal.task_invocation_sha256,
                    expected_response_protocol=session.seal.response_protocol,
                    expected_peer_cid=session.seal.expected_peer_cid,
                    expected_rpc_port=session.seal.rpc_port,
                )
                authenticated = True
            except ValueError:
                issues.add(ClinicalCollectionFailureCode.UNAUTHENTICATED_ATTEMPT)
            if authenticated and _session_matches_run_expectation(session, expectation, workspace):
                run_expectation_verified = True
            elif authenticated:
                issues.add(ClinicalCollectionFailureCode.RUN_EXPECTATION_MISMATCH)
            if authenticated and session.task_invocation == workspace.invocation:
                binding_verified = True
            elif authenticated:
                issues.add(ClinicalCollectionFailureCode.TASK_BINDING_MISMATCH)

        successful = False
        if authenticated and run_expectation_verified and binding_verified:
            if (
                session.seal.terminal_status != GuestRpcTerminalStatus.COMPLETED
                or not session.seal.submit_accepted
                or session.submission is None
            ):
                issues.add(ClinicalCollectionFailureCode.TASK_TERMINAL_FAILURE)
            elif not isinstance(session.submission, ExecutionSubmission):
                issues.add(ClinicalCollectionFailureCode.INVALID_TERMINAL_SUBMISSION)
            else:
                try:
                    validate_submission_for_invocation(session.task_invocation, session.submission)
                except ValueError:
                    issues.add(ClinicalCollectionFailureCode.INVALID_TERMINAL_SUBMISSION)
                else:
                    successful = True
                    submissions.append(session.submission)
        global_failures.update(issues)
        records.append(
            ClinicalExecutionTerminalAttempt(
                record_index=record_index,
                claimed_episode_id=episode_id,
                session_sha256=_model_sha256(session),
                session=session,
                authenticated=authenticated,
                run_expectation_verified=run_expectation_verified,
                task_binding_verified=binding_verified,
                successful_terminal_submission=successful,
                issues=tuple(sorted(issues, key=lambda item: item.value)),
            )
        )

    records_tuple = tuple(records)
    if not global_failures and len(submissions) == manifest.task_count:
        cohort_submission = make_execution_cohort_submission(manifest=manifest, submissions=submissions)
        status = ClinicalExecutionCollectionStatus.COMPLETED
    else:
        cohort_submission = None
        status = ClinicalExecutionCollectionStatus.FAILED
        if not global_failures:
            global_failures.add(ClinicalCollectionFailureCode.INVALID_TERMINAL_SUBMISSION)
    collection = ClinicalExecutionCollection(
        cohort_id=manifest.cohort_id,
        cohort_manifest_sha256=execution_cohort_manifest_sha256(manifest),
        evaluation_split=manifest.evaluation_split,
        expected_task_count=manifest.task_count,
        run_expectations_sha256=_sha256(_run_expectations_bytes(expectations)),
        run_expectations=expectations,
        supplied_attempt_count=len(records_tuple),
        attempts_sha256=_sha256(canonical_json_bytes([item.model_dump(mode='json') for item in records_tuple])),
        attempts=records_tuple,
        status=status,
        failure_codes=tuple(sorted(global_failures, key=lambda item: item.value)),
        cohort_submission=cohort_submission,
        receipt_key_id=expected_receipt_key_id,
    )
    return AuthenticatedClinicalExecutionCollection(
        collection=collection,
        collection_hmac_sha256=_collection_hmac(collection, receipt_key),
    )


def verify_authenticated_clinical_execution_collection(
    artifact: AuthenticatedClinicalExecutionCollection,
    *,
    manifest: ExecutionCohortManifest,
    workspaces: Iterable[LoadedClinicalAgenticWorkspace],
    run_expectations: Iterable[ClinicalExecutionRunExpectation],
    workspace_receipt_keys_by_id: Mapping[str, bytes],
    guest_rpc_receipt_keys_by_id: Mapping[str, bytes],
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> None:
    """Rebuild the deterministic collection from retained sessions and compare exact bytes."""

    if artifact.collection.receipt_key_id != expected_receipt_key_id or not hmac.compare_digest(
        artifact.collection_hmac_sha256,
        _collection_hmac(artifact.collection, receipt_key),
    ):
        raise ClinicalExecutionBridgeError('clinical collection authentication failed')
    rebuilt = collect_clinical_execution_sessions(
        manifest=manifest,
        workspaces=workspaces,
        run_expectations=run_expectations,
        sessions=tuple(item.session for item in artifact.collection.attempts),
        workspace_receipt_keys_by_id=workspace_receipt_keys_by_id,
        guest_rpc_receipt_keys_by_id=guest_rpc_receipt_keys_by_id,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise ClinicalExecutionBridgeError('clinical collection does not match its retained terminal sessions')


__all__ = [
    'AUTHENTICATED_CLINICAL_AGENTIC_WORKSPACE_SCHEMA_VERSION',
    'AUTHENTICATED_CLINICAL_EXECUTION_COLLECTION_SCHEMA_VERSION',
    'CLINICAL_AGENTIC_WORKSPACE_MANIFEST_SCHEMA_VERSION',
    'CLINICAL_AGENTIC_WORKSPACE_RECEIPT_SCHEMA_VERSION',
    'CLINICAL_EXECUTION_COLLECTION_SCHEMA_VERSION',
    'CLINICAL_EXECUTION_RUN_EXPECTATION_SCHEMA_VERSION',
    'AuthenticatedClinicalAgenticWorkspace',
    'AuthenticatedClinicalExecutionCollection',
    'ClinicalAgenticWorkspaceManifest',
    'ClinicalAgenticWorkspaceReceipt',
    'ClinicalCollectionFailureCode',
    'ClinicalExecutionBridgeError',
    'ClinicalExecutionCollection',
    'ClinicalExecutionCollectionStatus',
    'ClinicalExecutionRunExpectation',
    'ClinicalExecutionTerminalAttempt',
    'LoadedClinicalAgenticWorkspace',
    'build_clinical_agentic_workspace',
    'clinical_collection_receipt_key_id',
    'clinical_workspace_receipt_key_id',
    'collect_clinical_execution_sessions',
    'load_clinical_agentic_workspace',
    'verify_authenticated_clinical_execution_collection',
]
