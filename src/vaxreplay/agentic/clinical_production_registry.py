"""Durable one-attempt collection for Lane A production evidence.

The registry is the organizer-owned launch gate.  A cohort/system pair is registered once, every
task receives one run ID before a worker starts, and the first terminal result is immutable.  A
successful result is accepted only after an injected verifier has reloaded a complete
``LoadedClinicalProductionRunV02`` and this module has cross-checked its signed guest bootstrap,
outer receipt, worker, gateway/model route, harness, workspace, policy, and clinical submission
against the reservation.

SQLite provides global uniqueness only for users of one authoritative database.  Copying the
database creates a different authority; this module deliberately makes no cross-registry,
leaderboard-admission, official-isolation, or contamination-control claim.
"""

from __future__ import annotations

import enum
import hashlib
import os
import sqlite3
import stat
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_execution_bridge import (
    LoadedClinicalAgenticWorkspace,
    load_clinical_agentic_workspace,
)
from vaxreplay.agentic.clinical_guest_bootstrap import (
    AuthenticatedClinicalGuestBootstrap,
    ClinicalGuestRpcLimits,
    clinical_guest_bootstrap_hello_sha256,
    clinical_guest_bootstrap_signed_hello_sha256,
)
from vaxreplay.agentic.clinical_production_run_v02 import (
    CLINICAL_PRODUCTION_RUN_V02_SCHEMA_VERSION,
    AuthenticatedClinicalProductionRunV02,
    LoadedClinicalProductionRunV02,
)
from vaxreplay.agentic.guest_rpc import GuestRpcTerminalStatus, guest_rpc_policy_sha256
from vaxreplay.agentic.provider_gateway import (
    GatewayModelRoute,
    authenticated_gateway_policy_sha256,
    gateway_capability_grant_sha256,
    gateway_model_route_sha256,
)
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.run_artifact import AgenticHarnessIdentity
from vaxreplay.agentic.task_protocol import agentic_task_invocation_sha256, validate_submission_for_invocation
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_aggregation import (
    ExecutionCohortManifest,
    ExecutionCohortSubmission,
    execution_cohort_manifest_sha256,
    make_execution_cohort_submission,
)
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission, ExecutionTask

CLINICAL_PRODUCTION_SYSTEM_IDENTITY_SCHEMA_VERSION = 'vaxreplay.clinical-production-system-identity.dev-v0.5'
CLINICAL_PRODUCTION_RESERVATION_SCHEMA_VERSION = 'vaxreplay.clinical-production-reservation.dev-v0.5'
CLINICAL_PRODUCTION_LAUNCH_SCHEMA_VERSION = 'vaxreplay.clinical-production-launch.dev-v0.1'
CLINICAL_PRODUCTION_START_REDEMPTION_SCHEMA_VERSION = 'vaxreplay.clinical-production-start-redemption.dev-v0.1'
CLINICAL_PRODUCTION_RESULT_SCHEMA_VERSION = 'vaxreplay.clinical-production-result.dev-v0.5'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_MAX_MODEL_BYTES = 64 * 1024 * 1024


class ClinicalProductionRegistryError(ValueError):
    """The authoritative one-attempt registry rejected an operation."""


class ClinicalProductionTerminalCode(str, enum.Enum):
    SUCCESS = 'success'
    EVIDENCE_AUTHENTICATION_FAILED = 'evidence_authentication_failed'
    EVIDENCE_BINDING_MISMATCH = 'evidence_binding_mismatch'
    WORKER_TERMINAL_FAILURE = 'worker_terminal_failure'
    INVALID_CLINICAL_SUBMISSION = 'invalid_clinical_submission'
    SCHEDULER_FAILURE = 'scheduler_failure'
    WORKER_LAUNCH_FAILURE = 'worker_launch_failure'
    WORKER_LOST = 'worker_lost'


class ClinicalProductionCohortStatus(str, enum.Enum):
    OPEN = 'open'
    COMPLETED = 'completed'
    FAILED = 'failed'


type ClinicalProductionExplicitFailureCode = Literal[
    ClinicalProductionTerminalCode.SCHEDULER_FAILURE,
    ClinicalProductionTerminalCode.WORKER_LAUNCH_FAILURE,
    ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
    ClinicalProductionTerminalCode.WORKER_LOST,
    ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED,
]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_sha256(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _aware(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{label} must include a UTC offset')
    return value.astimezone(UTC)


class ClinicalProductionSystemIdentity(StrictModel):
    """Everything executable or routed that defines one evaluated system.

    Human-readable harness names are included in the exact identity.  The separate core identity
    intentionally excludes those names, so changing a label cannot buy another attempt.
    """

    schema_version: Literal['vaxreplay.clinical-production-system-identity.dev-v0.5'] = (
        CLINICAL_PRODUCTION_SYSTEM_IDENTITY_SCHEMA_VERSION
    )
    harness: AgenticHarnessIdentity
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    gateway_route: GatewayModelRoute
    gateway_route_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider_subprocess_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider_subprocess_behavior_sha256: str = Field(pattern=_SHA256_PATTERN)
    provider_subprocess_module_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_bootstrap_authorization_key_id: str = Field(pattern=_SHA256_PATTERN)
    guest_bootstrap_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    worker_attestation_key_id: str = Field(pattern=_SHA256_PATTERN)
    gateway_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    production_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    canonical_launcher_id: str = Field(pattern=_ID_PATTERN)
    canonical_launcher_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_protocol: Literal[AgenticResponseProtocol.CLINICAL_EXECUTION] = AgenticResponseProtocol.CLINICAL_EXECUTION

    @model_validator(mode='after')
    def validate_route_and_harness(self) -> Self:
        if self.gateway_route_sha256 != gateway_model_route_sha256(self.gateway_route):
            raise ValueError('gateway_route_sha256 does not bind the exact provider/model route')
        if (
            self.harness.requested_model_id != self.gateway_route.logical_model_id
            or self.harness.adapter_id != self.gateway_route.adapter_id
            or not self.harness.harness_image_or_commitment.startswith('sha256:')
            or len(self.harness.harness_image_or_commitment) != 71
        ):
            raise ValueError('harness identity must bind the pinned model route and a SHA-256 image')
        return self


def clinical_production_system_identity_sha256(identity: ClinicalProductionSystemIdentity) -> str:
    return _model_sha256(identity)


def clinical_production_system_core_sha256(identity: ClinicalProductionSystemIdentity) -> str:
    """Alias-resistant behavior identity, excluding organizer keys and display route labels."""

    route = identity.gateway_route
    return _model_sha256(
        {
            'schema_version': 'vaxreplay.clinical-production-system-core.dev-v0.3',
            'harness_image_or_commitment': identity.harness.harness_image_or_commitment,
            'harness_behavior_sha256': identity.harness.harness_behavior_sha256,
            'harness_execution_mode': identity.harness.harness_execution_mode,
            'provider': route.provider,
            'resolved_model_id': route.resolved_model_id,
            'adapter_executable_sha256': route.adapter_executable_sha256,
            'adapter_config_sha256': route.adapter_config_sha256,
            'endpoint_origin': route.endpoint_origin,
            'endpoint_path': route.endpoint_path,
            'fixed_parameters_sha256': route.fixed_parameters_sha256,
            'max_context_tokens': route.max_context_tokens,
            'max_output_tokens': route.max_output_tokens,
            'provider_subprocess_behavior_sha256': (identity.provider_subprocess_behavior_sha256),
            'provider_subprocess_module_source_sha256': (identity.provider_subprocess_module_source_sha256),
            'input_preflight': route.input_preflight,
            'reasoning_accounting': route.reasoning_accounting,
            'response_protocol': identity.response_protocol,
        }
    )


class ClinicalProductionTaskBinding(StrictModel):
    episode_id: str = Field(min_length=1)
    target_trial_id: str = Field(pattern=r'^trial-[a-z0-9][a-z0-9._-]*$')
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    authenticated_workspace_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)


class ClinicalProductionReservation(StrictModel):
    schema_version: Literal['vaxreplay.clinical-production-reservation.dev-v0.5'] = (
        CLINICAL_PRODUCTION_RESERVATION_SCHEMA_VERSION
    )
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    registered_entry_id: str = Field(pattern=_ID_PATTERN)
    cohort_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    cohort_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_split: Split
    system: ClinicalProductionSystemIdentity
    system_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_core_sha256: str = Field(pattern=_SHA256_PATTERN)
    tasks: tuple[ClinicalProductionTaskBinding, ...] = Field(min_length=1)
    reserved_at: datetime
    one_cohort_attempt: Literal[True] = True
    one_launch_per_task: Literal[True] = True
    one_start_redemption_per_task: Literal[True] = True
    first_terminal_result_retained: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @field_validator('reserved_at')
    @classmethod
    def validate_reserved_at(cls, value: datetime) -> datetime:
        return _aware(value, 'reservation timestamp')

    @model_validator(mode='after')
    def validate_bindings(self) -> Self:
        if self.system_identity_sha256 != clinical_production_system_identity_sha256(self.system):
            raise ValueError('reservation system hash does not bind its exact system identity')
        if self.system_core_sha256 != clinical_production_system_core_sha256(self.system):
            raise ValueError('reservation core hash does not bind its alias-resistant system identity')
        episode_ids = tuple(item.episode_id for item in self.tasks)
        if episode_ids != tuple(sorted(set(episode_ids))):
            raise ValueError('reservation tasks must use unique canonical episode order')
        return self


def clinical_production_reservation_sha256(reservation: ClinicalProductionReservation) -> str:
    return _model_sha256(reservation)


class ClinicalProductionTaskLaunch(StrictModel):
    schema_version: Literal['vaxreplay.clinical-production-launch.dev-v0.1'] = CLINICAL_PRODUCTION_LAUNCH_SCHEMA_VERSION
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    cohort_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    claimed_at: datetime
    launch_count: Literal[1] = 1

    @field_validator('claimed_at')
    @classmethod
    def validate_claimed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'launch claim timestamp')


def clinical_production_task_launch_sha256(launch: ClinicalProductionTaskLaunch) -> str:
    """The launch-ticket hash consumed by the canonical launcher start gate."""

    return _model_sha256(launch)


class ClinicalProductionStartRedemption(StrictModel):
    """Durable one-time authorization for one exact prepared worker and host sessions.

    The canonical organizer launcher must redeem a launch ticket after preparing the worker but
    before starting it.  The redemption hash, rather than the reusable launch-ticket hash, is then
    passed as ``attempt_reservation_sha256`` to the worker, gateway, guest RPC, and finalizer.
    """

    schema_version: Literal['vaxreplay.clinical-production-start-redemption.dev-v0.1'] = (
        CLINICAL_PRODUCTION_START_REDEMPTION_SCHEMA_VERSION
    )
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    launch_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    canonical_launcher_id: str = Field(pattern=_ID_PATTERN)
    canonical_launcher_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepared_worker_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_session_id: str = Field(pattern=_RUN_ID_PATTERN)
    gateway_capability_id: str = Field(pattern=_SHA256_PATTERN)
    redeemed_at: datetime
    worker_start_count: Literal[1] = 1

    @field_validator('redeemed_at')
    @classmethod
    def validate_redeemed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'worker start redemption timestamp')


def clinical_production_start_redemption_sha256(
    redemption: ClinicalProductionStartRedemption,
) -> str:
    """The unique start authorization bound into every execution-layer receipt."""

    return _model_sha256(redemption)


class ClinicalProductionTaskRecord(StrictModel):
    episode_id: str
    state: Literal['reserved', 'launched', 'succeeded', 'failed']
    launch: ClinicalProductionTaskLaunch | None = None
    launch_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    start_redemption: ClinicalProductionStartRedemption | None = None
    start_redemption_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    terminal_code: ClinicalProductionTerminalCode | None = None
    evidence_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    terminal_record_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    submission_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    terminal_at: datetime | None = None

    @field_validator('terminal_at')
    @classmethod
    def validate_terminal_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value, 'terminal timestamp')

    @model_validator(mode='after')
    def validate_state(self) -> Self:
        launched = self.launch is not None and self.launch_sha256 is not None
        start_redeemed = self.start_redemption is not None and self.start_redemption_sha256 is not None
        if (self.start_redemption is None) != (self.start_redemption_sha256 is None):
            raise ValueError('worker start redemption and its hash must be present together')
        if self.start_redemption is not None and (
            clinical_production_start_redemption_sha256(self.start_redemption) != self.start_redemption_sha256
        ):
            raise ValueError('worker start redemption hash does not bind its exact authorization')
        terminal = self.terminal_code is not None and self.terminal_at is not None
        retained_failure = self.terminal_record_sha256 is not None
        if self.state == 'reserved' and (launched or start_redeemed or terminal or retained_failure):
            raise ValueError('reserved task cannot contain launch or terminal data')
        if self.state == 'launched' and (not launched or terminal or retained_failure):
            raise ValueError('launched task requires only launch data')
        if self.state in {'succeeded', 'failed'} and (not launched or not terminal):
            raise ValueError('terminal task requires launch and terminal data')
        if self.state == 'succeeded' and (
            self.terminal_code != ClinicalProductionTerminalCode.SUCCESS
            or self.submission_sha256 is None
            or retained_failure
            or not start_redeemed
        ):
            raise ValueError('successful task requires one redeemed start, a submission, and no failure record')
        if self.state == 'failed' and (
            self.terminal_code == ClinicalProductionTerminalCode.SUCCESS
            or self.submission_sha256 is not None
            or self.terminal_record_sha256 is None
        ):
            raise ValueError('failed task requires a retained terminal record and no submission')
        return self


class ClinicalProductionCohortResult(StrictModel):
    schema_version: Literal['vaxreplay.clinical-production-result.dev-v0.5'] = CLINICAL_PRODUCTION_RESULT_SCHEMA_VERSION
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    cohort_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: ClinicalProductionCohortStatus
    tasks: tuple[ClinicalProductionTaskRecord, ...] = Field(min_length=1)
    cohort_submission: ExecutionCohortSubmission | None = None
    authoritative_registry_task_count: int = Field(gt=0)
    exact_split_and_cohort_coverage_required: Literal[True] = True
    full_outer_production_evidence_required: Literal[True] = True
    worker_gateway_route_model_and_harness_pins_checked: Literal[True] = True
    one_attempt_enforced_within_registry_authority: Literal[True] = True
    one_start_redemption_per_task_enforced: Literal[True] = True
    start_redemption_bound_to_prepared_worker_and_sessions: Literal[True] = True
    bootstrap_keys_bound_in_system_identity: Literal[True] = True
    provider_subprocess_bound_in_system_identity: Literal[True] = True
    provider_child_module_source_bound_in_system_identity: Literal[True] = True
    strict_signed_guest_bootstrap_required: Literal[True] = True
    bootstrap_bound_outer_v02_evidence_required: Literal[True] = True
    first_terminal_failure_retained: Literal[True] = True
    provider_credentials_stored: Literal[False] = False
    independent_provider_model_snapshot_attested: Literal[False] = False
    cross_registry_global_uniqueness_claimed: Literal[False] = False
    clinical_production_run_finalizer_available: Literal[True] = True
    development_only: Literal[True] = True
    official_execution_qualified: Literal[False] = False
    leaderboard_admitted: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False

    @model_validator(mode='after')
    def validate_result(self) -> Self:
        if self.authoritative_registry_task_count != len(self.tasks):
            raise ValueError('result task count does not match its retained registry records')
        states = {item.state for item in self.tasks}
        if self.status == ClinicalProductionCohortStatus.COMPLETED:
            if states != {'succeeded'} or self.cohort_submission is None:
                raise ValueError('completed production collection requires every task submission')
        elif self.status == ClinicalProductionCohortStatus.FAILED:
            if 'failed' not in states or self.cohort_submission is not None:
                raise ValueError('failed production collection requires a retained terminal failure')
        elif self.cohort_submission is not None:
            raise ValueError('open production collection cannot emit a cohort submission')
        return self


type ProductionRunReauthenticator = Callable[[Path, str], LoadedClinicalProductionRunV02]


def require_official_model_snapshot_attestation(
    identity: ClinicalProductionSystemIdentity,
    *,
    external_attestation_sha256: str | None,
) -> None:
    """Fail closed unless both route and external evidence attest an immutable snapshot.

    Current ``GatewayModelRoute`` records ``provider_model_snapshot_attested=false``, so current
    development routes always fail this official gate.  A reported resolved ID or moving alias is
    useful audit metadata but is not independently verified immutability.
    """

    identity = ClinicalProductionSystemIdentity.model_validate_json(canonical_json_bytes(identity))
    if (
        not identity.gateway_route.provider_model_snapshot_attested
        or external_attestation_sha256 is None
        or len(external_attestation_sha256) != 64
        or any(character not in '0123456789abcdef' for character in external_attestation_sha256)
    ):
        raise ClinicalProductionRegistryError(
            'official execution requires an externally attested immutable provider model snapshot'
        )


@dataclass(frozen=True, slots=True)
class ClinicalProductionReservationContext:
    reservation: ClinicalProductionReservation
    reservation_sha256: str


class SqliteClinicalProductionRegistry:
    """Single-authority transactional registry with no model/provider credentials.

    SQLite may create rollback journals and other sidecars beside the database.  The immediate
    parent is therefore part of the authority boundary: it is an owned mode-0700 directory whose
    device/inode identity is pinned for the lifetime of this registry client.
    """

    def __init__(self, path: Path, *, authority_id: str) -> None:
        if not authority_id or len(authority_id) > 256:
            raise ValueError('registry authority ID must contain 1 to 256 characters')
        supplied = path.expanduser()
        if supplied.is_symlink():
            raise ValueError('clinical production registry path cannot be a symlink')
        supplied_parent_metadata = self._prepare_safe_parent_directory(supplied.parent)
        try:
            resolved_parent = supplied.parent.resolve(strict=True)
        except OSError as error:
            raise ClinicalProductionRegistryError(
                'clinical production registry parent directory is unavailable'
            ) from error
        rechecked_parent_metadata = self._require_safe_parent_directory(supplied.parent)
        resolved_parent_metadata = self._require_safe_parent_directory(resolved_parent)
        supplied_parent_identity = (supplied_parent_metadata.st_dev, supplied_parent_metadata.st_ino)
        if (rechecked_parent_metadata.st_dev, rechecked_parent_metadata.st_ino) != supplied_parent_identity or (
            resolved_parent_metadata.st_dev,
            resolved_parent_metadata.st_ino,
        ) != supplied_parent_identity:
            raise ClinicalProductionRegistryError(
                'clinical production registry parent directory changed during validation'
            )
        self.path = resolved_parent / supplied.name
        if self.path.is_symlink():
            raise ValueError('clinical production registry path cannot be a symlink')
        if self.path.exists():
            self._require_safe_database(self.path)
        self._parent_device = resolved_parent_metadata.st_dev
        self._parent_inode = resolved_parent_metadata.st_ino
        self.authority_id = authority_id
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        self._require_current_parent_directory()
        if self.path.is_symlink():
            raise ClinicalProductionRegistryError('clinical production registry path became a symlink')
        if self.path.exists():
            self._require_safe_database(self.path)
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('PRAGMA synchronous = FULL')
        connection.execute('PRAGMA busy_timeout = 30000')
        return connection

    @classmethod
    def _prepare_safe_parent_directory(cls, path: Path) -> os.stat_result:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            pass
        except OSError as error:
            raise ClinicalProductionRegistryError(
                'clinical production registry parent directory could not be created safely'
            ) from error
        return cls._require_safe_parent_directory(path)

    @staticmethod
    def _require_safe_parent_directory(path: Path) -> os.stat_result:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ClinicalProductionRegistryError(
                'clinical production registry parent directory is unavailable'
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ClinicalProductionRegistryError('clinical production registry parent directory cannot be a symlink')
        if not stat.S_ISDIR(metadata.st_mode):
            raise ClinicalProductionRegistryError('clinical production registry parent must be a directory')
        if metadata.st_uid != os.geteuid():
            raise ClinicalProductionRegistryError(
                'clinical production registry parent directory must be owned by the current user'
            )
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise ClinicalProductionRegistryError(
                'clinical production registry parent directory must be private mode-0700'
            )
        return metadata

    def _require_current_parent_directory(self) -> None:
        metadata = self._require_safe_parent_directory(self.path.parent)
        if (metadata.st_dev, metadata.st_ino) != (self._parent_device, self._parent_inode):
            raise ClinicalProductionRegistryError(
                'clinical production registry parent directory was replaced after initialization'
            )

    @staticmethod
    def _require_safe_database(path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ClinicalProductionRegistryError('clinical production registry is unavailable') from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ClinicalProductionRegistryError('existing registry must be one owned private mode-0600 regular file')

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS registry_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    authority_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cohort_reservations (
                    reservation_sha256 TEXT PRIMARY KEY,
                    cohort_manifest_sha256 TEXT NOT NULL,
                    system_identity_sha256 TEXT NOT NULL,
                    system_core_sha256 TEXT NOT NULL,
                    target_bytes BLOB NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('open', 'completed', 'failed')),
                    UNIQUE (cohort_manifest_sha256, system_identity_sha256),
                    UNIQUE (cohort_manifest_sha256, system_core_sha256)
                );
                CREATE TABLE IF NOT EXISTS task_attempts (
                    reservation_sha256 TEXT NOT NULL REFERENCES cohort_reservations(reservation_sha256),
                    episode_id TEXT NOT NULL,
                    binding_bytes BLOB NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('reserved', 'launched', 'succeeded', 'failed')),
                    run_id TEXT UNIQUE,
                    launch_sha256 TEXT UNIQUE,
                    launch_bytes BLOB,
                    start_redemption_sha256 TEXT UNIQUE,
                    start_redemption_bytes BLOB,
                    terminal_code TEXT,
                    evidence_sha256 TEXT,
                    terminal_record_bytes BLOB,
                    submission_bytes BLOB,
                    terminal_at TEXT,
                    PRIMARY KEY (reservation_sha256, episode_id)
                );
                """
            )
            task_columns = {str(row[1]) for row in connection.execute('PRAGMA table_info(task_attempts)').fetchall()}
            redemption_columns = {'start_redemption_sha256', 'start_redemption_bytes'}
            missing_redemption_columns = redemption_columns - task_columns
            if missing_redemption_columns:
                if missing_redemption_columns != redemption_columns:
                    raise ClinicalProductionRegistryError('registry has a partial worker-start-redemption schema')
                connection.execute('ALTER TABLE task_attempts ADD COLUMN start_redemption_sha256 TEXT')
                connection.execute('ALTER TABLE task_attempts ADD COLUMN start_redemption_bytes BLOB')
            connection.execute(
                'CREATE UNIQUE INDEX IF NOT EXISTS task_attempts_start_redemption_unique '
                'ON task_attempts(start_redemption_sha256)'
            )
            connection.execute(
                'INSERT OR IGNORE INTO registry_meta(singleton, authority_id) VALUES (1, ?)',
                (self.authority_id,),
            )
            row = connection.execute('SELECT authority_id FROM registry_meta WHERE singleton = 1').fetchone()
            if row is None or row['authority_id'] != self.authority_id:
                raise ClinicalProductionRegistryError('registry belongs to a different authority')
        finally:
            connection.close()
        os.chmod(self.path, 0o600)
        metadata = self.path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ClinicalProductionRegistryError('registry must be one private mode-0600 regular file')

    def reserve_cohort(
        self,
        *,
        manifest: ExecutionCohortManifest,
        workspaces: Iterable[LoadedClinicalAgenticWorkspace],
        workspace_receipt_keys_by_id: Mapping[str, bytes],
        system: ClinicalProductionSystemIdentity,
        registered_entry_id: str,
        reserved_at: datetime,
    ) -> ClinicalProductionReservationContext:
        manifest = ExecutionCohortManifest.model_validate_json(canonical_json_bytes(manifest))
        system = ClinicalProductionSystemIdentity.model_validate_json(canonical_json_bytes(system))
        workspace_by_episode: dict[str, LoadedClinicalAgenticWorkspace] = {}
        for supplied in workspaces:
            key_id = supplied.authenticated_receipt.receipt.receipt_key_id
            key = workspace_receipt_keys_by_id.get(key_id)
            if key is None:
                raise ClinicalProductionRegistryError('reservation lacks a workspace authentication key')
            loaded = load_clinical_agentic_workspace(
                supplied.root,
                expected_authenticated_receipt_sha256=supplied.authenticated_receipt_sha256,
                receipt_key=key,
                expected_receipt_key_id=key_id,
            )
            episode_id = loaded.task.context.episode_id
            if episode_id in workspace_by_episode:
                raise ClinicalProductionRegistryError('reservation contains a duplicate task workspace')
            workspace_by_episode[episode_id] = loaded
        manifest_by_episode = {item.episode_id: item for item in manifest.tasks}
        if set(workspace_by_episode) != set(manifest_by_episode):
            raise ClinicalProductionRegistryError('reservation workspaces do not exactly cover the cohort')
        task_bindings: list[ClinicalProductionTaskBinding] = []
        for episode_id in sorted(manifest_by_episode):
            binding = manifest_by_episode[episode_id]
            workspace = workspace_by_episode[episode_id]
            task = workspace.task
            if (
                task.context.target_trial_id,
                task.context_sha256,
                _model_sha256(task),
            ) != (
                binding.target_trial_id,
                binding.task_context_sha256,
                binding.task_sha256,
            ):
                raise ClinicalProductionRegistryError('workspace is bound to a different manifest task')
            task_bindings.append(
                ClinicalProductionTaskBinding(
                    episode_id=episode_id,
                    target_trial_id=task.context.target_trial_id,
                    task_sha256=_model_sha256(task),
                    task_context_sha256=task.context_sha256,
                    workspace_manifest_sha256=workspace.manifest_sha256,
                    workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
                    model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
                    authenticated_workspace_receipt_sha256=workspace.authenticated_receipt_sha256,
                )
            )
        reservation = ClinicalProductionReservation(
            registry_authority_id=self.authority_id,
            registered_entry_id=registered_entry_id,
            cohort_id=manifest.cohort_id,
            cohort_manifest_sha256=execution_cohort_manifest_sha256(manifest),
            evaluation_split=manifest.evaluation_split,
            system=system,
            system_identity_sha256=clinical_production_system_identity_sha256(system),
            system_core_sha256=clinical_production_system_core_sha256(system),
            tasks=tuple(task_bindings),
            reserved_at=reserved_at,
        )
        reservation_bytes = canonical_json_bytes(reservation)
        reservation_sha256 = _sha256(reservation_bytes)
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(
                """INSERT INTO cohort_reservations(
                    reservation_sha256, cohort_manifest_sha256, system_identity_sha256,
                    system_core_sha256, target_bytes, status
                ) VALUES (?, ?, ?, ?, ?, 'open')""",
                (
                    reservation_sha256,
                    reservation.cohort_manifest_sha256,
                    reservation.system_identity_sha256,
                    reservation.system_core_sha256,
                    reservation_bytes,
                ),
            )
            connection.executemany(
                """INSERT INTO task_attempts(
                    reservation_sha256, episode_id, binding_bytes, state
                ) VALUES (?, ?, ?, 'reserved')""",
                (
                    (reservation_sha256, binding.episode_id, canonical_json_bytes(binding))
                    for binding in reservation.tasks
                ),
            )
            connection.commit()
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise ClinicalProductionRegistryError(
                'this cohort/system executable already has an attempt; renaming cannot create another'
            ) from error
        finally:
            connection.close()
        return ClinicalProductionReservationContext(reservation, reservation_sha256)

    def reservation_context(self, reservation_sha256: str) -> ClinicalProductionReservationContext:
        """Return one independently reloaded reservation for trusted launcher composition.

        This is intentionally a verified projection rather than access to a database row.  A
        canonical launcher uses it to check its deployment identity before consuming a task.
        """

        reservation = self._load_reservation(reservation_sha256)
        return ClinicalProductionReservationContext(
            reservation=reservation,
            reservation_sha256=clinical_production_reservation_sha256(reservation),
        )

    def reservation_hashes(self) -> tuple[str, ...]:
        """Return the complete verified inventory for this one SQLite authority.

        This is intentionally not a caller-selected subset.  The managed startup reaper uses it
        while the root-owned registry service is stopped, so unjournaled launched attempts cannot
        be omitted from orphan discovery.
        """

        connection = self._connect()
        try:
            rows = connection.execute(
                'SELECT reservation_sha256 FROM cohort_reservations ORDER BY reservation_sha256'
            ).fetchall()
        finally:
            connection.close()
        values = tuple(str(row['reservation_sha256']) for row in rows)
        for value in values:
            reservation = self._load_reservation(value)
            if clinical_production_reservation_sha256(reservation) != value:
                raise ClinicalProductionRegistryError('registry reservation inventory contains an invalid commitment')
        return values

    def task_records(self, reservation_sha256: str) -> tuple[ClinicalProductionTaskRecord, ...]:
        """Return verified task state without changing the cohort status.

        The canonical launcher restart reconciler uses this view only to terminalize already
        consumed ``launched`` tasks.  Reserved tasks are deliberately left untouched.
        """

        reservation = self._load_reservation(reservation_sha256)
        return tuple(self._record_from_row(row) for row in self._verified_task_rows(reservation))

    def claim_task_launch(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        run_id: str,
        claimed_at: datetime,
    ) -> ClinicalProductionTaskLaunch:
        reservation = self._load_reservation(reservation_sha256)
        binding = next((item for item in reservation.tasks if item.episode_id == episode_id), None)
        if binding is None:
            raise ClinicalProductionRegistryError('task is not in the fixed cohort reservation')
        launch = ClinicalProductionTaskLaunch(
            registry_authority_id=self.authority_id,
            reservation_sha256=reservation_sha256,
            cohort_manifest_sha256=reservation.cohort_manifest_sha256,
            system_identity_sha256=reservation.system_identity_sha256,
            episode_id=episode_id,
            workspace_manifest_sha256=binding.workspace_manifest_sha256,
            run_id=run_id,
            claimed_at=claimed_at,
        )
        launch_bytes = canonical_json_bytes(launch)
        launch_sha256 = _sha256(launch_bytes)
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            cohort = connection.execute(
                'SELECT status FROM cohort_reservations WHERE reservation_sha256 = ?',
                (reservation_sha256,),
            ).fetchone()
            if cohort is None or cohort['status'] != 'open':
                raise ClinicalProductionRegistryError('cohort reservation is not open for launches')
            cursor = connection.execute(
                """UPDATE task_attempts SET
                    state = 'launched', run_id = ?, launch_sha256 = ?, launch_bytes = ?
                WHERE reservation_sha256 = ? AND episode_id = ? AND state = 'reserved'""",
                (run_id, launch_sha256, launch_bytes, reservation_sha256, episode_id),
            )
            if cursor.rowcount != 1:
                raise ClinicalProductionRegistryError('task already consumed its one launch')
            connection.commit()
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise ClinicalProductionRegistryError('run ID or launch reservation was already used') from error
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return launch

    def redeem_task_start(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        launch_sha256: str,
        canonical_launcher_id: str,
        canonical_launcher_executable_sha256: str,
        prepared_worker_sha256: str,
        guest_rpc_session_id: str,
        gateway_capability_id: str,
        redeemed_at: datetime,
    ) -> ClinicalProductionStartRedemption:
        """Consume one launch ticket immediately before one exact worker/session start.

        This compare-and-set is the durable handoff to the trusted organizer launcher.  Preparing a
        microVM and allocating session identifiers may happen first, but no worker, guest RPC
        session, or provider gateway session may start until this method returns.  The returned
        redemption hash is the attempt reservation bound into every downstream receipt.
        """

        launch, reservation, _ = self._load_launch_context(reservation_sha256, episode_id)
        expected_launch_sha256 = clinical_production_task_launch_sha256(launch)
        if launch_sha256 != expected_launch_sha256:
            raise ClinicalProductionRegistryError('worker start presented a different launch ticket')
        system = reservation.system
        if (
            canonical_launcher_id,
            canonical_launcher_executable_sha256,
        ) != (
            system.canonical_launcher_id,
            system.canonical_launcher_executable_sha256,
        ):
            raise ClinicalProductionRegistryError('worker start came from a different canonical launcher')
        redeemed_at = _aware(redeemed_at, 'worker start redemption timestamp')
        if redeemed_at < launch.claimed_at:
            raise ClinicalProductionRegistryError('worker start redemption cannot predate its launch ticket')
        redemption = ClinicalProductionStartRedemption(
            registry_authority_id=self.authority_id,
            reservation_sha256=reservation_sha256,
            launch_sha256=launch_sha256,
            system_identity_sha256=reservation.system_identity_sha256,
            episode_id=episode_id,
            run_id=launch.run_id,
            canonical_launcher_id=canonical_launcher_id,
            canonical_launcher_executable_sha256=canonical_launcher_executable_sha256,
            prepared_worker_sha256=prepared_worker_sha256,
            guest_rpc_session_id=guest_rpc_session_id,
            gateway_capability_id=gateway_capability_id,
            redeemed_at=redeemed_at,
        )
        redemption_bytes = canonical_json_bytes(redemption)
        redemption_sha256 = _sha256(redemption_bytes)
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            cohort = connection.execute(
                'SELECT status FROM cohort_reservations WHERE reservation_sha256 = ?',
                (reservation_sha256,),
            ).fetchone()
            if cohort is None or cohort['status'] != 'open':
                raise ClinicalProductionRegistryError('cohort reservation is not open for worker starts')
            cursor = connection.execute(
                """UPDATE task_attempts SET
                    start_redemption_sha256 = ?, start_redemption_bytes = ?
                WHERE reservation_sha256 = ? AND episode_id = ? AND state = 'launched'
                    AND launch_sha256 = ? AND start_redemption_sha256 IS NULL
                    AND start_redemption_bytes IS NULL""",
                (
                    redemption_sha256,
                    redemption_bytes,
                    reservation_sha256,
                    episode_id,
                    launch_sha256,
                ),
            )
            if cursor.rowcount != 1:
                raise ClinicalProductionRegistryError(
                    'launch ticket already redeemed; a second worker/session start is forbidden'
                )
            connection.commit()
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise ClinicalProductionRegistryError('worker start redemption was already used by another task') from error
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return redemption

    def record_production_run(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        production_run_root: Path,
        reauthenticate: ProductionRunReauthenticator,
        terminal_at: datetime,
    ) -> ClinicalProductionTaskRecord:
        """Reload full evidence, cross-check release pins, then permanently terminalize once."""

        launch, reservation, binding = self._load_launch_context(reservation_sha256, episode_id)
        launch_sha256 = clinical_production_task_launch_sha256(launch)
        redemption = self._load_start_redemption(
            reservation_sha256,
            episode_id,
            launch=launch,
            reservation=reservation,
        )
        redemption_sha256 = clinical_production_start_redemption_sha256(redemption)
        evidence_sha256: str | None = None
        try:
            loaded = reauthenticate(production_run_root, redemption_sha256)
            if not isinstance(loaded, LoadedClinicalProductionRunV02):
                raise ClinicalProductionRegistryError(
                    'registry v0.3 requires strict-bootstrap-bound v0.2 clinical production evidence'
                )
            evidence_sha256 = _model_sha256(loaded.authenticated_outer_receipt)
            if loaded.authenticated_outer_receipt_sha256 != evidence_sha256:
                raise ClinicalProductionRegistryError('v0.2 evidence loader reported a false outer receipt hash')
        except Exception:
            failure_record = canonical_json_bytes(
                {
                    'schema_version': 'vaxreplay.clinical-production-terminal-record.dev-v0.1',
                    'terminal_code': ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED,
                    'launch_sha256': launch_sha256,
                    'start_redemption_sha256': redemption_sha256,
                    'details_disclosed': False,
                }
            )
            return self._terminalize(
                reservation_sha256=reservation_sha256,
                episode_id=episode_id,
                terminal_code=ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED,
                evidence_sha256=evidence_sha256,
                terminal_record=failure_record,
                submission=None,
                terminal_at=terminal_at,
            )
        try:
            submission = self._validate_loaded_run(
                loaded,
                reservation=reservation,
                binding=binding,
                launch=launch,
                launch_sha256=launch_sha256,
                redemption=redemption,
                redemption_sha256=redemption_sha256,
            )
        except ClinicalProductionRegistryError as error:
            code = (
                ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE
                if str(error) == 'authenticated run ended in terminal failure'
                else ClinicalProductionTerminalCode.EVIDENCE_BINDING_MISMATCH
            )
            failure_record = canonical_json_bytes(
                {
                    'schema_version': 'vaxreplay.clinical-production-terminal-record.dev-v0.1',
                    'terminal_code': code,
                    'launch_sha256': launch_sha256,
                    'start_redemption_sha256': redemption_sha256,
                    'details_disclosed': False,
                }
            )
            return self._terminalize(
                reservation_sha256=reservation_sha256,
                episode_id=episode_id,
                terminal_code=code,
                evidence_sha256=evidence_sha256,
                terminal_record=failure_record,
                submission=None,
                terminal_at=terminal_at,
            )
        except ValueError:
            failure_record = canonical_json_bytes(
                {
                    'schema_version': 'vaxreplay.clinical-production-terminal-record.dev-v0.1',
                    'terminal_code': ClinicalProductionTerminalCode.INVALID_CLINICAL_SUBMISSION,
                    'launch_sha256': launch_sha256,
                    'start_redemption_sha256': redemption_sha256,
                    'details_disclosed': False,
                }
            )
            return self._terminalize(
                reservation_sha256=reservation_sha256,
                episode_id=episode_id,
                terminal_code=ClinicalProductionTerminalCode.INVALID_CLINICAL_SUBMISSION,
                evidence_sha256=evidence_sha256,
                terminal_record=failure_record,
                submission=None,
                terminal_at=terminal_at,
            )
        return self._terminalize(
            reservation_sha256=reservation_sha256,
            episode_id=episode_id,
            terminal_code=ClinicalProductionTerminalCode.SUCCESS,
            evidence_sha256=evidence_sha256,
            terminal_record=None,
            submission=submission,
            terminal_at=terminal_at,
        )

    def record_explicit_failure(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        terminal_code: ClinicalProductionExplicitFailureCode,
        failure_record: bytes,
        terminal_at: datetime,
    ) -> ClinicalProductionTaskRecord:
        if not failure_record or len(failure_record) > _MAX_MODEL_BYTES:
            raise ClinicalProductionRegistryError('failure record must contain bounded retained bytes')
        if terminal_code in {
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
            ClinicalProductionTerminalCode.WORKER_LOST,
            ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED,
        }:
            launch, reservation, _ = self._load_launch_context(reservation_sha256, episode_id)
            self._load_start_redemption(
                reservation_sha256,
                episode_id,
                launch=launch,
                reservation=reservation,
            )
        return self._terminalize(
            reservation_sha256=reservation_sha256,
            episode_id=episode_id,
            terminal_code=terminal_code,
            evidence_sha256=_sha256(failure_record),
            terminal_record=failure_record,
            submission=None,
            terminal_at=terminal_at,
        )

    def result(
        self,
        *,
        reservation_sha256: str,
        manifest: ExecutionCohortManifest,
    ) -> ClinicalProductionCohortResult:
        reservation = self._load_reservation(reservation_sha256)
        manifest = ExecutionCohortManifest.model_validate_json(canonical_json_bytes(manifest))
        if (
            execution_cohort_manifest_sha256(manifest),
            manifest.evaluation_split,
            manifest.cohort_id,
        ) != (
            reservation.cohort_manifest_sha256,
            reservation.evaluation_split,
            reservation.cohort_id,
        ):
            raise ClinicalProductionRegistryError('result manifest differs from the fixed reservation')
        rows = self._verified_task_rows(reservation)
        records = tuple(self._record_from_row(row) for row in rows)
        states = {record.state for record in records}
        cohort_submission: ExecutionCohortSubmission | None = None
        if states == {'succeeded'}:
            submissions = tuple(ExecutionSubmission.model_validate_json(row['submission_bytes']) for row in rows)
            cohort_submission = make_execution_cohort_submission(manifest=manifest, submissions=submissions)
            status = ClinicalProductionCohortStatus.COMPLETED
            self._set_cohort_status(reservation_sha256, 'completed')
        elif 'failed' in states:
            status = ClinicalProductionCohortStatus.FAILED
            self._set_cohort_status(reservation_sha256, 'failed')
        else:
            status = ClinicalProductionCohortStatus.OPEN
        return ClinicalProductionCohortResult(
            registry_authority_id=self.authority_id,
            reservation_sha256=reservation_sha256,
            cohort_manifest_sha256=reservation.cohort_manifest_sha256,
            system_identity_sha256=reservation.system_identity_sha256,
            status=status,
            tasks=records,
            cohort_submission=cohort_submission,
            authoritative_registry_task_count=len(records),
        )

    def _validate_loaded_run(
        self,
        loaded: LoadedClinicalProductionRunV02,
        *,
        reservation: ClinicalProductionReservation,
        binding: ClinicalProductionTaskBinding,
        launch: ClinicalProductionTaskLaunch,
        launch_sha256: str,
        redemption: ClinicalProductionStartRedemption,
        redemption_sha256: str,
    ) -> ExecutionSubmission:
        receipt = loaded.receipt
        worker = loaded.worker_attestation.attestation
        gateway = loaded.gateway_session
        guest = loaded.guest_rpc_session
        system = reservation.system
        harness = system.harness
        self._validate_v02_bootstrap_binding(
            loaded,
            system=system,
            binding=binding,
            launch=launch,
            redemption=redemption,
            redemption_sha256=redemption_sha256,
        )
        if (
            redemption.registry_authority_id,
            redemption.reservation_sha256,
            redemption.launch_sha256,
            redemption.system_identity_sha256,
            redemption.episode_id,
            redemption.run_id,
            redemption.canonical_launcher_id,
            redemption.canonical_launcher_executable_sha256,
        ) != (
            self.authority_id,
            clinical_production_reservation_sha256(reservation),
            launch_sha256,
            reservation.system_identity_sha256,
            binding.episode_id,
            launch.run_id,
            system.canonical_launcher_id,
            system.canonical_launcher_executable_sha256,
        ):
            raise ClinicalProductionRegistryError('worker start redemption differs from the fixed launch')
        if clinical_production_start_redemption_sha256(redemption) != redemption_sha256:
            raise ClinicalProductionRegistryError('worker start redemption hash changed')
        if production_root := loaded.root:
            if production_root.is_symlink() or not production_root.is_dir():
                raise ClinicalProductionRegistryError('production evidence root is unsafe')
        workspace = loaded.workspace
        expected_workspace = (
            binding.workspace_manifest_sha256,
            binding.workspace_tree_sha256,
            binding.model_visible_surface_sha256,
            binding.authenticated_workspace_receipt_sha256,
            binding.task_sha256,
            binding.task_context_sha256,
        )
        actual_workspace = (
            workspace.manifest_sha256,
            workspace.manifest.workspace_tree_sha256,
            workspace.manifest.model_visible_surface_sha256,
            workspace.authenticated_receipt_sha256,
            _model_sha256(workspace.task),
            workspace.task.context_sha256,
        )
        if actual_workspace != expected_workspace:
            raise ClinicalProductionRegistryError('authenticated clinical workspace differs from the cohort binding')
        expected = (
            launch.run_id,
            redemption_sha256,
            binding.workspace_manifest_sha256,
            binding.workspace_tree_sha256,
            binding.model_visible_surface_sha256,
            binding.authenticated_workspace_receipt_sha256,
            binding.task_sha256,
            binding.task_context_sha256,
            agentic_task_invocation_sha256(workspace.invocation),
            system.execution_policy_sha256,
            system.harness,
            system.gateway_route.resolved_model_id,
            system.worker_spec_sha256,
            system.gateway_policy_sha256,
            system.gateway_route_sha256,
            system.guest_rpc_policy_sha256,
            system.worker_attestation_key_id,
            system.gateway_receipt_key_id,
            system.guest_rpc_receipt_key_id,
            system.production_receipt_key_id,
        )
        actual = (
            receipt.run_id,
            receipt.attempt_reservation_sha256,
            receipt.workspace_manifest_sha256,
            receipt.workspace_tree_sha256,
            receipt.model_visible_surface_sha256,
            receipt.authenticated_workspace_receipt_sha256,
            receipt.task_sha256,
            receipt.task_context_sha256,
            receipt.task_invocation_sha256,
            receipt.execution_policy_sha256,
            receipt.harness,
            receipt.resolved_model_id,
            receipt.worker_spec_sha256,
            receipt.gateway_policy_sha256,
            receipt.gateway_route_sha256,
            receipt.guest_rpc_policy_sha256,
            receipt.worker_attestation_key_id,
            receipt.gateway_receipt_key_id,
            receipt.guest_rpc_receipt_key_id,
            receipt.receipt_key_id,
        )
        if actual != expected:
            raise ClinicalProductionRegistryError(
                'clinical production receipt differs from the fixed launch/system pins'
            )
        if (
            gateway.route != system.gateway_route
            or gateway_model_route_sha256(gateway.route) != system.gateway_route_sha256
            or authenticated_gateway_policy_sha256(gateway.policy) != system.gateway_policy_sha256
            or guest_rpc_policy_sha256(guest.policy) != system.guest_rpc_policy_sha256
        ):
            raise ClinicalProductionRegistryError('authenticated gateway/model/RPC route differs from system pins')
        if (
            worker.run_id,
            worker.attempt_reservation_sha256,
            worker.worker_spec_sha256,
        ) != (launch.run_id, redemption_sha256, system.worker_spec_sha256):
            raise ClinicalProductionRegistryError('authenticated worker differs from the fixed launch')
        if (
            worker.prepared_worker_sha256 != redemption.prepared_worker_sha256
            or worker.started_at < redemption.redeemed_at
        ):
            raise ClinicalProductionRegistryError(
                'authenticated worker was not the one authorized after start redemption'
            )
        if (
            loaded.worker_attestation.attestation_key_id != system.worker_attestation_key_id
            or harness.harness_image_or_commitment != f'sha256:{worker.harness_sha256}'
            or gateway.policy.receipt_key_id != system.gateway_receipt_key_id
            or gateway.seal.receipt_key_id != system.gateway_receipt_key_id
            or guest.seal.receipt_key_id != system.guest_rpc_receipt_key_id
        ):
            raise ClinicalProductionRegistryError('authenticated worker/harness/evidence keys differ from pins')
        grant = gateway.grant
        if (
            grant.capability_id != redemption.gateway_capability_id
            or guest.seal.session_id != redemption.guest_rpc_session_id
        ):
            raise ClinicalProductionRegistryError(
                'authenticated gateway/guest sessions differ from the redeemed worker start'
            )
        if (
            grant.run_id,
            grant.attempt_reservation_sha256,
            grant.execution_policy_sha256,
            grant.workspace_manifest_sha256,
            grant.gateway_policy_sha256,
            grant.model_route_sha256,
        ) != (
            launch.run_id,
            redemption_sha256,
            system.execution_policy_sha256,
            binding.workspace_manifest_sha256,
            system.gateway_policy_sha256,
            system.gateway_route_sha256,
        ) or guest.gateway_grant != grant:
            raise ClinicalProductionRegistryError('authenticated gateway grant differs from the fixed launch')
        if (
            gateway.seal.run_id,
            gateway.seal.attempt_reservation_sha256,
            gateway.seal.execution_policy_sha256,
            gateway.seal.workspace_manifest_sha256,
            gateway.seal.gateway_policy_sha256,
            gateway.seal.model_route_sha256,
            gateway.seal.grant_sha256,
        ) != (
            launch.run_id,
            redemption_sha256,
            system.execution_policy_sha256,
            binding.workspace_manifest_sha256,
            system.gateway_policy_sha256,
            system.gateway_route_sha256,
            gateway_capability_grant_sha256(grant),
        ):
            raise ClinicalProductionRegistryError('authenticated gateway seal differs from the fixed launch')
        guest_seal = guest.seal
        if (
            guest_seal.run_id,
            guest_seal.attempt_reservation_sha256,
            guest_seal.execution_policy_sha256,
            guest_seal.worker_spec_sha256,
            guest_seal.rpc_policy_sha256,
            guest_seal.workspace_manifest_sha256,
            guest_seal.gateway_grant_sha256,
        ) != (
            launch.run_id,
            redemption_sha256,
            system.execution_policy_sha256,
            system.worker_spec_sha256,
            system.guest_rpc_policy_sha256,
            binding.workspace_manifest_sha256,
            gateway_capability_grant_sha256(grant),
        ):
            raise ClinicalProductionRegistryError('authenticated guest session differs from the fixed launch')
        invocation = workspace.invocation
        task = invocation.task
        submission = loaded.submission
        if (
            guest.seal.terminal_status != GuestRpcTerminalStatus.COMPLETED
            or not guest.seal.submit_accepted
            or invocation.response_protocol != AgenticResponseProtocol.CLINICAL_EXECUTION
            or not isinstance(task, ExecutionTask)
            or not isinstance(submission, ExecutionSubmission)
            or guest.task_invocation != invocation
            or guest.submission != submission
        ):
            raise ValueError('authenticated guest did not emit one terminal clinical submission')
        if (
            task.context.episode_id,
            task.context.target_trial_id,
            _model_sha256(task),
            task.context_sha256,
            invocation.workspace_manifest_sha256,
        ) != (
            binding.episode_id,
            binding.target_trial_id,
            binding.task_sha256,
            binding.task_context_sha256,
            binding.workspace_manifest_sha256,
        ):
            raise ClinicalProductionRegistryError('authenticated clinical task differs from its cohort binding')
        validate_submission_for_invocation(invocation, submission)
        submission_bytes = canonical_json_bytes(submission)
        if (receipt.submission_sha256, receipt.submission_bytes) != (
            _sha256(submission_bytes),
            len(submission_bytes),
        ):
            raise ClinicalProductionRegistryError('clinical production receipt differs from the submission')
        return submission

    @staticmethod
    def _validate_v02_bootstrap_binding(
        loaded: LoadedClinicalProductionRunV02,
        *,
        system: ClinicalProductionSystemIdentity,
        binding: ClinicalProductionTaskBinding,
        launch: ClinicalProductionTaskLaunch,
        redemption: ClinicalProductionStartRedemption,
        redemption_sha256: str,
    ) -> None:
        outer = loaded.authenticated_outer_receipt
        bootstrap = loaded.clinical_guest_bootstrap
        if not isinstance(outer, AuthenticatedClinicalProductionRunV02) or not isinstance(
            bootstrap, AuthenticatedClinicalGuestBootstrap
        ):
            raise ClinicalProductionRegistryError('v0.2 evidence contains a non-canonical nested artifact type')
        outer_receipt = outer.receipt
        bootstrap_receipt = bootstrap.receipt
        signed_hello = bootstrap.signed_hello
        hello = signed_hello.hello
        if (
            len(loaded.authenticated_outer_receipt_sha256) != 64
            or any(character not in '0123456789abcdef' for character in loaded.authenticated_outer_receipt_sha256)
            or len(loaded.clinical_guest_bootstrap_evidence_sha256) != 64
            or any(character not in '0123456789abcdef' for character in loaded.clinical_guest_bootstrap_evidence_sha256)
        ):
            raise ClinicalProductionRegistryError('v0.2 evidence hashes are not canonical SHA-256 digests')
        expected_rpc_limits = ClinicalGuestRpcLimits(
            maximum_frame_body_bytes=loaded.guest_rpc_session.policy.maximum_frame_body_bytes,
            maximum_session_wire_bytes=loaded.guest_rpc_session.policy.maximum_session_wire_bytes,
            maximum_requests=loaded.guest_rpc_session.policy.maximum_requests,
            maximum_list_entries=loaded.guest_rpc_session.policy.maximum_list_entries,
            maximum_read_bytes=loaded.guest_rpc_session.policy.maximum_read_bytes,
            maximum_search_results=loaded.guest_rpc_session.policy.maximum_search_results,
            maximum_submission_bytes=loaded.guest_rpc_session.policy.maximum_submission_bytes,
        )
        expected_base_sha256 = _model_sha256(loaded.authenticated_receipt)
        expected_bootstrap_sha256 = _model_sha256(bootstrap)
        expected_outer_sha256 = _model_sha256(outer)
        expected_hello_sha256 = clinical_guest_bootstrap_hello_sha256(hello)
        expected_signed_hello_sha256 = clinical_guest_bootstrap_signed_hello_sha256(signed_hello)
        expected_rpc_limits_sha256 = _model_sha256(expected_rpc_limits)
        expected = (
            launch.run_id,
            redemption_sha256,
            redemption.guest_rpc_session_id,
            agentic_task_invocation_sha256(loaded.workspace.invocation),
            binding.workspace_manifest_sha256,
            binding.workspace_tree_sha256,
            binding.model_visible_surface_sha256,
            system.execution_policy_sha256,
            system.worker_spec_sha256,
        )
        observed_outer = (
            outer_receipt.run_id,
            outer_receipt.start_redemption_sha256,
            outer_receipt.guest_rpc_session_id,
            outer_receipt.task_invocation_sha256,
            outer_receipt.workspace_manifest_sha256,
            outer_receipt.workspace_tree_sha256,
            outer_receipt.model_visible_surface_sha256,
            outer_receipt.execution_policy_sha256,
            outer_receipt.worker_spec_sha256,
        )
        observed_bootstrap_receipt = (
            bootstrap_receipt.run_id,
            bootstrap_receipt.start_redemption_sha256,
            bootstrap_receipt.session_id,
            bootstrap_receipt.task_invocation_sha256,
            bootstrap_receipt.workspace_manifest_sha256,
            bootstrap_receipt.workspace_tree_sha256,
            bootstrap_receipt.model_visible_surface_sha256,
            bootstrap_receipt.execution_policy_sha256,
            bootstrap_receipt.worker_spec_sha256,
        )
        observed_hello = (
            hello.run_id,
            hello.start_redemption_sha256,
            hello.session_id,
            hello.task_invocation_sha256,
            hello.workspace_manifest_sha256,
            hello.workspace_tree_sha256,
            hello.model_visible_surface_sha256,
            hello.execution_policy_sha256,
            hello.worker_spec_sha256,
        )
        if observed_outer != expected or observed_bootstrap_receipt != expected or observed_hello != expected:
            raise ClinicalProductionRegistryError(
                'strict guest bootstrap differs from the fixed launch, redemption, or system pins'
            )
        if (
            outer.schema_version != CLINICAL_PRODUCTION_RUN_V02_SCHEMA_VERSION
            or hello.task_invocation != loaded.workspace.invocation
            or hello.rpc_limits != expected_rpc_limits
            or bootstrap_receipt.rpc_limits_sha256 != expected_rpc_limits_sha256
            or outer_receipt.guest_rpc_limits_sha256 != expected_rpc_limits_sha256
            or outer_receipt.guest_rpc_policy_sha256 != system.guest_rpc_policy_sha256
            or signed_hello.authorization_key_id != system.guest_bootstrap_authorization_key_id
            or bootstrap_receipt.authorization_key_id != system.guest_bootstrap_authorization_key_id
            or outer_receipt.clinical_guest_bootstrap_authorization_key_id
            != system.guest_bootstrap_authorization_key_id
            or bootstrap_receipt.receipt_key_id != system.guest_bootstrap_receipt_key_id
            or outer_receipt.clinical_guest_bootstrap_receipt_key_id != system.guest_bootstrap_receipt_key_id
            or outer_receipt.receipt_key_id != system.production_receipt_key_id
            or outer.base_authenticated_run != loaded.authenticated_receipt
            or loaded.authenticated_receipt_sha256 != expected_base_sha256
            or outer_receipt.base_authenticated_run_sha256 != expected_base_sha256
            or loaded.clinical_guest_bootstrap_evidence_sha256 != expected_bootstrap_sha256
            or outer_receipt.clinical_guest_bootstrap_evidence_sha256 != expected_bootstrap_sha256
            or loaded.authenticated_outer_receipt_sha256 != expected_outer_sha256
            or signed_hello.hello_sha256 != expected_hello_sha256
            or bootstrap_receipt.hello_sha256 != expected_hello_sha256
            or outer_receipt.clinical_guest_bootstrap_hello_sha256 != expected_hello_sha256
            or bootstrap_receipt.signed_hello_sha256 != expected_signed_hello_sha256
            or outer_receipt.clinical_guest_bootstrap_signed_hello_sha256 != expected_signed_hello_sha256
            or (
                outer_receipt.bootstrap_valid_from,
                outer_receipt.bootstrap_expires_at,
                outer_receipt.bootstrap_hello_sent_at,
                outer_receipt.bootstrap_ack_received_at,
                outer_receipt.bootstrap_guest_accepted_at,
            )
            != (
                bootstrap_receipt.valid_from,
                bootstrap_receipt.expires_at,
                bootstrap_receipt.hello_sent_at,
                bootstrap_receipt.ack_received_at,
                bootstrap_receipt.guest_accepted_at,
            )
            or outer_receipt.sealed_at != loaded.receipt.sealed_at
        ):
            raise ClinicalProductionRegistryError(
                'v0.2 outer receipt does not bind the exact base run and signed guest authorization'
            )

    def _terminalize(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        terminal_code: ClinicalProductionTerminalCode,
        evidence_sha256: str | None,
        terminal_record: bytes | None,
        submission: ExecutionSubmission | None,
        terminal_at: datetime,
    ) -> ClinicalProductionTaskRecord:
        terminal_at = _aware(terminal_at, 'terminal timestamp')
        submission_bytes = None if submission is None else canonical_json_bytes(submission)
        state = 'succeeded' if terminal_code == ClinicalProductionTerminalCode.SUCCESS else 'failed'
        if (state == 'succeeded') != (submission_bytes is not None) or (state == 'failed') != (
            terminal_record is not None
        ):
            raise ClinicalProductionRegistryError('terminal status, submission, and retained failure record disagree')
        if terminal_record is not None and (not terminal_record or len(terminal_record) > _MAX_MODEL_BYTES):
            raise ClinicalProductionRegistryError('terminal failure record is empty or oversized')
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            cursor = connection.execute(
                """UPDATE task_attempts SET
                    state = ?, terminal_code = ?, evidence_sha256 = ?, terminal_record_bytes = ?,
                    submission_bytes = ?, terminal_at = ?
                WHERE reservation_sha256 = ? AND episode_id = ? AND state = 'launched'""",
                (
                    state,
                    terminal_code.value,
                    evidence_sha256,
                    terminal_record,
                    submission_bytes,
                    terminal_at.isoformat(),
                    reservation_sha256,
                    episode_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ClinicalProductionRegistryError(
                    'first terminal result is already retained or task was not launched'
                )
            if state == 'failed':
                connection.execute(
                    "UPDATE cohort_reservations SET status = 'failed' WHERE reservation_sha256 = ? AND status = 'open'",
                    (reservation_sha256,),
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self._record_for_episode(reservation_sha256, episode_id)

    def _load_reservation(self, reservation_sha256: str) -> ClinicalProductionReservation:
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT target_bytes, cohort_manifest_sha256, system_identity_sha256,
                system_core_sha256 FROM cohort_reservations WHERE reservation_sha256 = ?""",
                (reservation_sha256,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ClinicalProductionRegistryError('unknown cohort reservation')
        payload = bytes(row['target_bytes'])
        try:
            reservation = ClinicalProductionReservation.model_validate_json(payload)
        except ValueError as error:
            raise ClinicalProductionRegistryError('stored cohort reservation is invalid') from error
        if canonical_json_bytes(reservation) != payload or _sha256(payload) != reservation_sha256:
            raise ClinicalProductionRegistryError('stored cohort reservation changed')
        if (
            row['cohort_manifest_sha256'],
            row['system_identity_sha256'],
            row['system_core_sha256'],
        ) != (
            reservation.cohort_manifest_sha256,
            reservation.system_identity_sha256,
            reservation.system_core_sha256,
        ):
            raise ClinicalProductionRegistryError('stored reservation indexes differ from their target')
        if reservation.registry_authority_id != self.authority_id:
            raise ClinicalProductionRegistryError('reservation belongs to a different registry authority')
        return reservation

    def _load_launch_context(
        self, reservation_sha256: str, episode_id: str
    ) -> tuple[ClinicalProductionTaskLaunch, ClinicalProductionReservation, ClinicalProductionTaskBinding]:
        reservation = self._load_reservation(reservation_sha256)
        binding = next((item for item in reservation.tasks if item.episode_id == episode_id), None)
        if binding is None:
            raise ClinicalProductionRegistryError('task is not in the fixed cohort reservation')
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT state, binding_bytes, launch_sha256, launch_bytes FROM task_attempts
                WHERE reservation_sha256 = ? AND episode_id = ?""",
                (reservation_sha256, episode_id),
            ).fetchone()
        finally:
            connection.close()
        if row is None or row['state'] != 'launched' or row['launch_bytes'] is None:
            raise ClinicalProductionRegistryError('task is not awaiting its first terminal result')
        if bytes(row['binding_bytes']) != canonical_json_bytes(binding):
            raise ClinicalProductionRegistryError('stored task binding changed')
        payload = bytes(row['launch_bytes'])
        try:
            launch = ClinicalProductionTaskLaunch.model_validate_json(payload)
        except ValueError as error:
            raise ClinicalProductionRegistryError('stored launch record is invalid') from error
        if canonical_json_bytes(launch) != payload or _sha256(payload) != row['launch_sha256']:
            raise ClinicalProductionRegistryError('stored launch record changed')
        return launch, reservation, binding

    def _load_start_redemption(
        self,
        reservation_sha256: str,
        episode_id: str,
        *,
        launch: ClinicalProductionTaskLaunch,
        reservation: ClinicalProductionReservation,
    ) -> ClinicalProductionStartRedemption:
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT start_redemption_sha256, start_redemption_bytes FROM task_attempts
                WHERE reservation_sha256 = ? AND episode_id = ? AND state = 'launched'""",
                (reservation_sha256, episode_id),
            ).fetchone()
        finally:
            connection.close()
        if row is None or row['start_redemption_sha256'] is None or row['start_redemption_bytes'] is None:
            raise ClinicalProductionRegistryError('worker/session start was not redeemed by the canonical launcher')
        payload = bytes(row['start_redemption_bytes'])
        try:
            redemption = ClinicalProductionStartRedemption.model_validate_json(payload)
        except ValueError as error:
            raise ClinicalProductionRegistryError('stored worker start redemption is invalid') from error
        if canonical_json_bytes(redemption) != payload or _sha256(payload) != row['start_redemption_sha256']:
            raise ClinicalProductionRegistryError('stored worker start redemption changed')
        if (
            redemption.registry_authority_id,
            redemption.reservation_sha256,
            redemption.launch_sha256,
            redemption.system_identity_sha256,
            redemption.episode_id,
            redemption.run_id,
        ) != (
            self.authority_id,
            reservation_sha256,
            clinical_production_task_launch_sha256(launch),
            reservation.system_identity_sha256,
            episode_id,
            launch.run_id,
        ):
            raise ClinicalProductionRegistryError(
                'stored worker start redemption differs from its launch and reservation'
            )
        return redemption

    def _task_rows(self, reservation_sha256: str) -> tuple[sqlite3.Row, ...]:
        connection = self._connect()
        try:
            rows = tuple(
                connection.execute(
                    'SELECT * FROM task_attempts WHERE reservation_sha256 = ? ORDER BY episode_id',
                    (reservation_sha256,),
                )
            )
        finally:
            connection.close()
        if not rows:
            raise ClinicalProductionRegistryError('reservation has no retained task records')
        return rows

    def _verified_task_rows(self, reservation: ClinicalProductionReservation) -> tuple[sqlite3.Row, ...]:
        rows = self._task_rows(clinical_production_reservation_sha256(reservation))
        expected = {item.episode_id: canonical_json_bytes(item) for item in reservation.tasks}
        observed = {row['episode_id']: bytes(row['binding_bytes']) for row in rows}
        if observed != expected:
            raise ClinicalProductionRegistryError('registry task rows do not exactly match the fixed reservation')
        return rows

    def _record_from_row(self, row: sqlite3.Row) -> ClinicalProductionTaskRecord:
        launch = None
        if row['launch_bytes'] is not None:
            launch = ClinicalProductionTaskLaunch.model_validate_json(bytes(row['launch_bytes']))
        start_redemption = None
        if row['start_redemption_bytes'] is not None:
            start_redemption = ClinicalProductionStartRedemption.model_validate_json(
                bytes(row['start_redemption_bytes'])
            )
        submission_sha256 = None
        if row['submission_bytes'] is not None:
            submission_sha256 = _sha256(bytes(row['submission_bytes']))
        terminal_record_sha256 = None
        if row['terminal_record_bytes'] is not None:
            terminal_record_sha256 = _sha256(bytes(row['terminal_record_bytes']))
        return ClinicalProductionTaskRecord(
            episode_id=row['episode_id'],
            state=row['state'],
            launch=launch,
            launch_sha256=row['launch_sha256'],
            start_redemption=start_redemption,
            start_redemption_sha256=row['start_redemption_sha256'],
            terminal_code=(
                None if row['terminal_code'] is None else ClinicalProductionTerminalCode(row['terminal_code'])
            ),
            evidence_sha256=row['evidence_sha256'],
            terminal_record_sha256=terminal_record_sha256,
            submission_sha256=submission_sha256,
            terminal_at=None if row['terminal_at'] is None else datetime.fromisoformat(row['terminal_at']),
        )

    def _record_for_episode(self, reservation_sha256: str, episode_id: str) -> ClinicalProductionTaskRecord:
        rows = tuple(row for row in self._task_rows(reservation_sha256) if row['episode_id'] == episode_id)
        if len(rows) != 1:
            raise ClinicalProductionRegistryError('task record is missing or duplicated')
        return self._record_from_row(rows[0])

    def _set_cohort_status(self, reservation_sha256: str, status: Literal['completed', 'failed']) -> None:
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                'SELECT status FROM cohort_reservations WHERE reservation_sha256 = ?',
                (reservation_sha256,),
            ).fetchone()
            if row is None:
                raise ClinicalProductionRegistryError('unknown cohort reservation')
            if row['status'] not in {'open', status}:
                raise ClinicalProductionRegistryError('cohort terminal status cannot be rewritten')
            connection.execute(
                'UPDATE cohort_reservations SET status = ? WHERE reservation_sha256 = ?',
                (status, reservation_sha256),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


__all__ = [
    'CLINICAL_PRODUCTION_LAUNCH_SCHEMA_VERSION',
    'CLINICAL_PRODUCTION_RESERVATION_SCHEMA_VERSION',
    'CLINICAL_PRODUCTION_RESULT_SCHEMA_VERSION',
    'CLINICAL_PRODUCTION_START_REDEMPTION_SCHEMA_VERSION',
    'CLINICAL_PRODUCTION_SYSTEM_IDENTITY_SCHEMA_VERSION',
    'ClinicalProductionCohortResult',
    'ClinicalProductionCohortStatus',
    'ClinicalProductionExplicitFailureCode',
    'ClinicalProductionRegistryError',
    'ClinicalProductionReservation',
    'ClinicalProductionReservationContext',
    'ClinicalProductionStartRedemption',
    'ClinicalProductionSystemIdentity',
    'ClinicalProductionTaskBinding',
    'ClinicalProductionTaskLaunch',
    'ClinicalProductionTaskRecord',
    'ClinicalProductionTerminalCode',
    'ProductionRunReauthenticator',
    'SqliteClinicalProductionRegistry',
    'clinical_production_reservation_sha256',
    'clinical_production_system_core_sha256',
    'clinical_production_system_identity_sha256',
    'clinical_production_start_redemption_sha256',
    'clinical_production_task_launch_sha256',
    'require_official_model_snapshot_attestation',
]
