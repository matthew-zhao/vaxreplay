"""Canonical development launcher composition for Lane A clinical tasks.

The worker, provider gateway, guest RPC broker, clinical finalizer, and one-attempt registry have
separate security boundaries.  This module owns their ordering.  It deliberately delegates the
Linux/KVM-specific prepare and run operations to a typed runtime boundary so the state machine can
be tested on non-Linux hosts without claiming that a fake runtime is production execution.

The registry claim is the ownership boundary.  A caller which loses that atomic claim never writes
a failure for somebody else's attempt.  Once this launcher has claimed a task, every ordinary
Python failure is converted to one immutable terminal record.  A process or host crash is handled
by ``reconcile_consumed_tasks`` on restart; reconciliation never launches a consumed task.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.clinical_execution_bridge import LoadedClinicalAgenticWorkspace
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionExplicitFailureCode,
    ClinicalProductionReservation,
    ClinicalProductionReservationContext,
    ClinicalProductionStartRedemption,
    ClinicalProductionTaskBinding,
    ClinicalProductionTaskLaunch,
    ClinicalProductionTaskRecord,
    ClinicalProductionTerminalCode,
    ProductionRunReauthenticator,
    clinical_production_start_redemption_sha256,
    clinical_production_system_identity_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.clinical_production_run_v02 import LoadedClinicalProductionRunV02
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

CANONICAL_CLINICAL_LAUNCHER_DEPLOYMENT_SCHEMA_VERSION = 'vaxreplay.canonical-clinical-launcher-deployment.dev-v0.1'
CLINICAL_PREPARED_RUNTIME_SCHEMA_VERSION = 'vaxreplay.clinical-prepared-runtime.dev-v0.1'
CLINICAL_RUNTIME_START_SCHEMA_VERSION = 'vaxreplay.clinical-runtime-start.dev-v0.1'
CLINICAL_LAUNCHER_FAILURE_SCHEMA_VERSION = 'vaxreplay.clinical-launcher-failure.dev-v0.1'
AUTHENTICATED_CLINICAL_LAUNCHER_FAILURE_SCHEMA_VERSION = 'vaxreplay.authenticated-clinical-launcher-failure.dev-v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_FAILURE_KEY_ID_DOMAIN = b'vaxreplay.clinical-launcher-failure-key-id.dev-v0.1\x00'
_FAILURE_HMAC_DOMAIN = b'vaxreplay.clinical-launcher-failure.dev-v0.1\x00'


class ClinicalLauncherError(ValueError):
    """The pinned deployment, runtime boundary, or launcher result is inconsistent."""


class ClinicalRuntimeFailureCode(str, enum.Enum):
    """Stable failure categories returned by the injected prepare/run implementation."""

    LAUNCH_FAILED = 'launch_failed'
    BOOTSTRAP_FAILED = 'bootstrap_failed'
    WORKER_TERMINAL_FAILURE = 'worker_terminal_failure'
    WORKER_LOST = 'worker_lost'
    CLEANUP_FAILED = 'cleanup_failed'
    EVIDENCE_FINALIZATION_FAILED = 'evidence_finalization_failed'


class ClinicalLauncherFailureCode(str, enum.Enum):
    """Stable organizer-facing reason retained for each launcher-owned failure."""

    PREPARE_FAILED = 'prepare_failed'
    PREPARED_BINDING_MISMATCH = 'prepared_binding_mismatch'
    START_REDEMPTION_FAILED = 'start_redemption_failed'
    RUNTIME_LAUNCH_FAILED = 'runtime_launch_failed'
    RUNTIME_BOOTSTRAP_FAILED = 'runtime_bootstrap_failed'
    RUNTIME_TERMINAL_FAILURE = 'runtime_terminal_failure'
    RUNTIME_WORKER_LOST = 'runtime_worker_lost'
    RUNTIME_CLEANUP_FAILED = 'runtime_cleanup_failed'
    EVIDENCE_FINALIZATION_FAILED = 'evidence_finalization_failed'
    RUNTIME_INTERNAL_FAILURE = 'runtime_internal_failure'
    EVIDENCE_AUTHENTICATION_FAILED = 'evidence_authentication_failed'
    EVIDENCE_BINDING_MISMATCH = 'evidence_binding_mismatch'
    INVALID_CLINICAL_SUBMISSION = 'invalid_clinical_submission'
    RECONCILED_UNREDEEMED_CLAIM = 'reconciled_unredeemed_claim'
    RECONCILED_REDEEMED_START = 'reconciled_redeemed_start'


class ClinicalLauncherPhase(str, enum.Enum):
    PREPARE = 'prepare'
    START_REDEMPTION = 'start_redemption'
    RUN = 'run'
    EVIDENCE = 'evidence'
    RECONCILE = 'reconcile'


class CanonicalClinicalLauncherDeployment(StrictModel):
    """Pinned identity of the organizer deployment, never supplied by a run caller."""

    schema_version: Literal['vaxreplay.canonical-clinical-launcher-deployment.dev-v0.1'] = (
        CANONICAL_CLINICAL_LAUNCHER_DEPLOYMENT_SCHEMA_VERSION
    )
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    canonical_launcher_id: str = Field(pattern=_ID_PATTERN)
    canonical_launcher_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_system_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_id: str = Field(pattern=_ID_PATTERN)
    runtime_version: str = Field(min_length=1, max_length=200)
    runtime_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    failure_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    single_authoritative_registry_required: Literal[True] = True
    automatic_worker_retry: Literal[False] = False
    automatic_provider_retry: Literal[False] = False
    development_only: Literal[True] = True
    linux_kvm_runtime_qualified: Literal[False] = False
    official_execution_qualified: Literal[False] = False


def canonical_clinical_launcher_deployment_sha256(deployment: CanonicalClinicalLauncherDeployment) -> str:
    return hashlib.sha256(canonical_json_bytes(deployment)).hexdigest()


def clinical_launcher_failure_key_id(key: bytes) -> str:
    _require_failure_key(key)
    return hashlib.sha256(_FAILURE_KEY_ID_DOMAIN + key).hexdigest()


class ClinicalPreparedRuntime(StrictModel):
    """Content-free preparation receipt returned before a start can be redeemed."""

    schema_version: Literal['vaxreplay.clinical-prepared-runtime.dev-v0.1'] = CLINICAL_PREPARED_RUNTIME_SCHEMA_VERSION
    runtime_id: str = Field(pattern=_ID_PATTERN)
    runtime_version: str = Field(min_length=1, max_length=200)
    runtime_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    launcher_deployment_sha256: str = Field(pattern=_SHA256_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    launch_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_visible_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepared_worker_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_session_id: str = Field(pattern=_RUN_ID_PATTERN)
    gateway_capability_id: str = Field(pattern=_SHA256_PATTERN)
    prepared_at: datetime
    worker_started: Literal[False] = False
    development_only: Literal[True] = True
    official_execution_qualified: Literal[False] = False

    @field_validator('prepared_at')
    @classmethod
    def validate_prepared_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('runtime preparation timestamp must include a UTC offset')
        return value.astimezone(UTC)


def clinical_prepared_runtime_sha256(prepared: ClinicalPreparedRuntime) -> str:
    return hashlib.sha256(canonical_json_bytes(prepared)).hexdigest()


@dataclass(frozen=True, slots=True)
class ClinicalRuntimePrepareRequest:
    deployment: CanonicalClinicalLauncherDeployment
    reservation: ClinicalProductionReservation
    binding: ClinicalProductionTaskBinding
    launch: ClinicalProductionTaskLaunch
    workspace: LoadedClinicalAgenticWorkspace


class ClinicalRuntimeStart(StrictModel):
    """Exact authorization passed once from the launcher to the prepared runtime."""

    schema_version: Literal['vaxreplay.clinical-runtime-start.dev-v0.1'] = CLINICAL_RUNTIME_START_SCHEMA_VERSION
    launcher_deployment_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepared_runtime_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_redemption: ClinicalProductionStartRedemption
    start_redemption_sha256: str = Field(pattern=_SHA256_PATTERN)
    automatic_retry_permitted: Literal[False] = False
    development_only: Literal[True] = True

    @model_validator(mode='after')
    def validate_redemption(self) -> Self:
        if clinical_production_start_redemption_sha256(self.start_redemption) != self.start_redemption_sha256:
            raise ValueError('runtime start hash does not bind its exact redemption')
        return self


@dataclass(frozen=True, slots=True)
class ClinicalRuntimeCompleted:
    """The runtime produced strict-bootstrap-bound v0.2 evidence for independent reload."""

    production_run_root: Path
    production_evidence_schema_version: Literal['vaxreplay.authenticated-clinical-production-run.dev-v0.2']
    authenticated_bootstrap_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.production_run_root, Path):
            raise TypeError('production_run_root must be a pathlib.Path')
        _require_sha256(self.authenticated_bootstrap_sha256, 'authenticated bootstrap SHA-256')


@dataclass(frozen=True, slots=True)
class ClinicalRuntimeFailed:
    code: ClinicalRuntimeFailureCode
    authenticated_bootstrap_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, ClinicalRuntimeFailureCode):
            raise TypeError('clinical runtime failure code is invalid')
        if self.authenticated_bootstrap_sha256 is not None:
            _require_sha256(self.authenticated_bootstrap_sha256, 'authenticated bootstrap SHA-256')


type ClinicalRuntimeOutcome = ClinicalRuntimeCompleted | ClinicalRuntimeFailed


class ClinicalRuntimeBoundary(Protocol):
    """Injected owner of prepare, the only launch/run call, and authenticated package creation."""

    def prepare(self, request: ClinicalRuntimePrepareRequest) -> ClinicalPreparedRuntime: ...

    def discard_prepared(self, prepared: ClinicalPreparedRuntime) -> None: ...

    def run(self, prepared: ClinicalPreparedRuntime, start: ClinicalRuntimeStart) -> ClinicalRuntimeOutcome: ...


type ClinicalProductionRunLoader = Callable[[Path, str], LoadedClinicalProductionRunV02]


class ClinicalLauncherFailureReceipt(StrictModel):
    schema_version: Literal['vaxreplay.clinical-launcher-failure.dev-v0.1'] = CLINICAL_LAUNCHER_FAILURE_SCHEMA_VERSION
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    launch_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_redemption_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    authenticated_bootstrap_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    launcher_deployment_sha256: str = Field(pattern=_SHA256_PATTERN)
    launcher_failure_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    phase: ClinicalLauncherPhase
    failure_code: ClinicalLauncherFailureCode
    registry_terminal_code: ClinicalProductionTerminalCode
    prepared_runtime_discarded: bool | None
    failed_at: datetime
    details_disclosed: Literal[False] = False
    attempt_consumed: Literal[True] = True
    retry_permitted: Literal[False] = False
    development_only: Literal[True] = True
    linux_kvm_runtime_qualified: Literal[False] = False
    official_execution_qualified: Literal[False] = False

    @field_validator('failed_at')
    @classmethod
    def validate_failed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('launcher failure timestamp must include a UTC offset')
        return value.astimezone(UTC)


class AuthenticatedClinicalLauncherFailure(StrictModel):
    schema_version: Literal['vaxreplay.authenticated-clinical-launcher-failure.dev-v0.1'] = (
        AUTHENTICATED_CLINICAL_LAUNCHER_FAILURE_SCHEMA_VERSION
    )
    receipt: ClinicalLauncherFailureReceipt
    receipt_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)


def clinical_launcher_failure_hmac(receipt: ClinicalLauncherFailureReceipt, key: bytes) -> str:
    _require_failure_key(key)
    return hmac.new(key, _FAILURE_HMAC_DOMAIN + canonical_json_bytes(receipt), hashlib.sha256).hexdigest()


def verify_authenticated_clinical_launcher_failure(
    artifact: AuthenticatedClinicalLauncherFailure,
    *,
    key: bytes,
    expected_key_id: str,
) -> ClinicalLauncherFailureReceipt:
    if clinical_launcher_failure_key_id(key) != expected_key_id:
        raise ClinicalLauncherError('launcher failure key does not match the expected key ID')
    if artifact.receipt.launcher_failure_receipt_key_id != expected_key_id:
        raise ClinicalLauncherError('launcher failure receipt uses a different key ID')
    expected_hmac = clinical_launcher_failure_hmac(artifact.receipt, key)
    if not hmac.compare_digest(artifact.receipt_hmac_sha256, expected_hmac):
        raise ClinicalLauncherError('launcher failure receipt authentication failed')
    return artifact.receipt


@dataclass(frozen=True, slots=True)
class ClinicalLauncherSuccess:
    record: ClinicalProductionTaskRecord
    loaded_run: LoadedClinicalProductionRunV02
    launch: ClinicalProductionTaskLaunch
    start_redemption: ClinicalProductionStartRedemption


@dataclass(frozen=True, slots=True)
class ClinicalLauncherFailure:
    record: ClinicalProductionTaskRecord
    failure_code: ClinicalLauncherFailureCode
    launch: ClinicalProductionTaskLaunch
    start_redemption: ClinicalProductionStartRedemption | None
    authenticated_failure: AuthenticatedClinicalLauncherFailure | None


type ClinicalLauncherResult = ClinicalLauncherSuccess | ClinicalLauncherFailure


class ClinicalProductionRegistryBoundary(Protocol):
    """Launcher-facing boundary implemented by SQLite and the managed Unix-socket client."""

    @property
    def authority_id(self) -> str: ...

    def reservation_context(self, reservation_sha256: str) -> ClinicalProductionReservationContext: ...

    def task_records(self, reservation_sha256: str) -> tuple[ClinicalProductionTaskRecord, ...]: ...

    def claim_task_launch(
        self, *, reservation_sha256: str, episode_id: str, run_id: str, claimed_at: datetime
    ) -> ClinicalProductionTaskLaunch: ...

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
    ) -> ClinicalProductionStartRedemption: ...

    def record_production_run(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        production_run_root: Path,
        reauthenticate: ProductionRunReauthenticator,
        terminal_at: datetime,
    ) -> ClinicalProductionTaskRecord: ...

    def record_explicit_failure(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        terminal_code: ClinicalProductionExplicitFailureCode,
        failure_record: bytes,
        terminal_at: datetime,
    ) -> ClinicalProductionTaskRecord: ...


class CanonicalClinicalLauncher:
    """Compose one exact registry-owned attempt without retries or caller-provided identity."""

    def __init__(
        self,
        *,
        registry: ClinicalProductionRegistryBoundary,
        deployment: CanonicalClinicalLauncherDeployment,
        runtime: ClinicalRuntimeBoundary,
        evidence_loader: ClinicalProductionRunLoader,
        failure_receipt_key: bytes,
        clock: Callable[[], datetime] | None = None,
        run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if registry.authority_id != deployment.registry_authority_id:
            raise ClinicalLauncherError('launcher deployment belongs to a different registry authority')
        if clinical_launcher_failure_key_id(failure_receipt_key) != deployment.failure_receipt_key_id:
            raise ClinicalLauncherError('launcher failure key differs from its deployment pin')
        self.registry = registry
        self.deployment = deployment
        self.runtime = runtime
        self.evidence_loader = evidence_loader
        self._failure_receipt_key = bytes(failure_receipt_key)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._run_id_factory = run_id_factory or (lambda: secrets.token_hex(16))

    def execute_reserved_task(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        workspace: LoadedClinicalAgenticWorkspace,
    ) -> ClinicalLauncherResult:
        """Claim and execute one task.

        Validation before ``claim_task_launch`` does not consume an attempt.  Once claim succeeds,
        this method either records success or permanently records a failure.  A losing concurrent
        claimant receives the registry exception and never terminalizes the winning attempt.
        """

        context = self.registry.reservation_context(reservation_sha256)
        reservation = context.reservation
        self._validate_reservation_deployment(reservation)
        binding = next((item for item in reservation.tasks if item.episode_id == episode_id), None)
        if binding is None:
            raise ClinicalLauncherError('task is not in the pinned launcher reservation')
        self._validate_workspace(binding, workspace)
        run_id = self._run_id_factory()
        claimed_at = self._now()
        # This atomic call is the ownership boundary.  Do not catch it: the losing side of a
        # concurrent claim must never write a failure into the winning side's task record.
        launch = self.registry.claim_task_launch(
            reservation_sha256=reservation_sha256,
            episode_id=episode_id,
            run_id=run_id,
            claimed_at=claimed_at,
        )
        request = ClinicalRuntimePrepareRequest(
            deployment=self.deployment,
            reservation=reservation,
            binding=binding,
            launch=launch,
            workspace=workspace,
        )
        try:
            prepared = self.runtime.prepare(request)
        except Exception:
            return self._record_failure(
                reservation=reservation,
                launch=launch,
                start_redemption=None,
                phase=ClinicalLauncherPhase.PREPARE,
                code=ClinicalLauncherFailureCode.PREPARE_FAILED,
                terminal_code=ClinicalProductionTerminalCode.WORKER_LAUNCH_FAILURE,
                prepared_runtime_discarded=None,
            )

        try:
            self._validate_prepared(prepared, reservation=reservation, binding=binding, launch=launch)
        except Exception:
            discarded = self._discard_prepared(prepared)
            return self._record_failure(
                reservation=reservation,
                launch=launch,
                start_redemption=None,
                phase=ClinicalLauncherPhase.PREPARE,
                code=ClinicalLauncherFailureCode.PREPARED_BINDING_MISMATCH,
                terminal_code=ClinicalProductionTerminalCode.WORKER_LAUNCH_FAILURE,
                prepared_runtime_discarded=discarded,
            )

        try:
            start_redemption = self.registry.redeem_task_start(
                reservation_sha256=reservation_sha256,
                episode_id=episode_id,
                launch_sha256=clinical_production_task_launch_sha256(launch),
                canonical_launcher_id=self.deployment.canonical_launcher_id,
                canonical_launcher_executable_sha256=self.deployment.canonical_launcher_executable_sha256,
                prepared_worker_sha256=prepared.prepared_worker_sha256,
                guest_rpc_session_id=prepared.guest_rpc_session_id,
                gateway_capability_id=prepared.gateway_capability_id,
                redeemed_at=max(self._now(), launch.claimed_at),
            )
        except Exception:
            discarded = self._discard_prepared(prepared)
            return self._record_failure(
                reservation=reservation,
                launch=launch,
                start_redemption=None,
                phase=ClinicalLauncherPhase.START_REDEMPTION,
                code=ClinicalLauncherFailureCode.START_REDEMPTION_FAILED,
                terminal_code=ClinicalProductionTerminalCode.SCHEDULER_FAILURE,
                prepared_runtime_discarded=discarded,
            )

        try:
            start = ClinicalRuntimeStart(
                launcher_deployment_sha256=canonical_clinical_launcher_deployment_sha256(self.deployment),
                prepared_runtime_sha256=clinical_prepared_runtime_sha256(prepared),
                start_redemption=start_redemption,
                start_redemption_sha256=clinical_production_start_redemption_sha256(start_redemption),
            )
            outcome = self.runtime.run(prepared, start)
        except Exception:
            return self._record_failure(
                reservation=reservation,
                launch=launch,
                start_redemption=start_redemption,
                phase=ClinicalLauncherPhase.RUN,
                code=ClinicalLauncherFailureCode.RUNTIME_INTERNAL_FAILURE,
                terminal_code=ClinicalProductionTerminalCode.WORKER_LOST,
                prepared_runtime_discarded=None,
            )
        if isinstance(outcome, ClinicalRuntimeFailed):
            try:
                launcher_code, terminal_code = _runtime_failure_mapping(outcome.code)
            except (KeyError, TypeError):
                launcher_code = ClinicalLauncherFailureCode.RUNTIME_INTERNAL_FAILURE
                terminal_code = ClinicalProductionTerminalCode.WORKER_LOST
            return self._record_failure(
                reservation=reservation,
                launch=launch,
                start_redemption=start_redemption,
                phase=ClinicalLauncherPhase.RUN,
                code=launcher_code,
                terminal_code=terminal_code,
                prepared_runtime_discarded=None,
                authenticated_bootstrap_sha256=outcome.authenticated_bootstrap_sha256,
            )
        if not isinstance(outcome, ClinicalRuntimeCompleted):
            return self._record_failure(
                reservation=reservation,
                launch=launch,
                start_redemption=start_redemption,
                phase=ClinicalLauncherPhase.RUN,
                code=ClinicalLauncherFailureCode.RUNTIME_INTERNAL_FAILURE,
                terminal_code=ClinicalProductionTerminalCode.WORKER_LOST,
                prepared_runtime_discarded=None,
            )
        if outcome.production_evidence_schema_version != ('vaxreplay.authenticated-clinical-production-run.dev-v0.2'):
            return self._record_failure(
                reservation=reservation,
                launch=launch,
                start_redemption=start_redemption,
                phase=ClinicalLauncherPhase.EVIDENCE,
                code=ClinicalLauncherFailureCode.EVIDENCE_AUTHENTICATION_FAILED,
                terminal_code=ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED,
                prepared_runtime_discarded=None,
                authenticated_bootstrap_sha256=outcome.authenticated_bootstrap_sha256,
            )

        reloaded: list[LoadedClinicalProductionRunV02] = []

        def reauthenticate(root: Path, expected_attempt_sha256: str) -> LoadedClinicalProductionRunV02:
            loaded = self.evidence_loader(root, expected_attempt_sha256)
            if not isinstance(loaded, LoadedClinicalProductionRunV02):
                raise ClinicalLauncherError('independent evidence loader did not return the pinned v0.2 evidence type')
            outer = getattr(loaded, 'authenticated_outer_receipt', None)
            if (
                getattr(outer, 'schema_version', None) != 'vaxreplay.authenticated-clinical-production-run.dev-v0.2'
                or getattr(loaded, 'clinical_guest_bootstrap_evidence_sha256', None)
                != outcome.authenticated_bootstrap_sha256
            ):
                raise ClinicalLauncherError(
                    'independently loaded evidence lacks the exact strict-bootstrap v0.2 binding'
                )
            reloaded.append(loaded)
            return loaded

        record = self.registry.record_production_run(
            reservation_sha256=reservation_sha256,
            episode_id=episode_id,
            production_run_root=outcome.production_run_root,
            reauthenticate=reauthenticate,
            terminal_at=self._now(),
        )
        if record.state == 'succeeded':
            if record.terminal_code != ClinicalProductionTerminalCode.SUCCESS or len(reloaded) != 1:
                raise ClinicalLauncherError('registry reported success without one independent evidence reload')
            return ClinicalLauncherSuccess(
                record=record,
                loaded_run=reloaded[0],
                launch=launch,
                start_redemption=start_redemption,
            )
        return ClinicalLauncherFailure(
            record=record,
            failure_code=_registry_failure_mapping(record.terminal_code),
            launch=launch,
            start_redemption=start_redemption,
            authenticated_failure=None,
        )

    def reconcile_consumed_tasks(self, *, reservation_sha256: str) -> tuple[ClinicalLauncherFailure, ...]:
        """Fail closed after restart; never relaunch a claimed or redeemed task.

        Run this before admitting new launcher work and never concurrently with a live launcher for
        the same authority.  Reserved tasks have not consumed their attempt and are not changed.
        """

        terminalizer = CanonicalClinicalRecoveryTerminalizer(
            registry=self.registry,
            deployment=self.deployment,
            failure_receipt_key=self._failure_receipt_key,
            clock=self._clock,
        )
        return terminalizer.reconcile_consumed_tasks(
            reservation_sha256=reservation_sha256,
        )

    def _validate_reservation_deployment(self, reservation: ClinicalProductionReservation) -> None:
        _validate_recovery_reservation_deployment(
            reservation,
            deployment=self.deployment,
        )

    @staticmethod
    def _validate_workspace(
        binding: ClinicalProductionTaskBinding,
        workspace: LoadedClinicalAgenticWorkspace,
    ) -> None:
        observed = (
            workspace.task.context.episode_id,
            workspace.task.context.target_trial_id,
            hashlib.sha256(canonical_json_bytes(workspace.task)).hexdigest(),
            workspace.task.context_sha256,
            workspace.manifest_sha256,
            workspace.manifest.workspace_tree_sha256,
            workspace.manifest.model_visible_surface_sha256,
            workspace.authenticated_receipt_sha256,
        )
        expected = (
            binding.episode_id,
            binding.target_trial_id,
            binding.task_sha256,
            binding.task_context_sha256,
            binding.workspace_manifest_sha256,
            binding.workspace_tree_sha256,
            binding.model_visible_surface_sha256,
            binding.authenticated_workspace_receipt_sha256,
        )
        if observed != expected:
            raise ClinicalLauncherError('launcher workspace differs from the fixed registry task binding')

    def _validate_prepared(
        self,
        prepared: ClinicalPreparedRuntime,
        *,
        reservation: ClinicalProductionReservation,
        binding: ClinicalProductionTaskBinding,
        launch: ClinicalProductionTaskLaunch,
    ) -> None:
        expected = (
            self.deployment.runtime_id,
            self.deployment.runtime_version,
            self.deployment.runtime_executable_sha256,
            self.deployment.runtime_config_sha256,
            canonical_clinical_launcher_deployment_sha256(self.deployment),
            clinical_production_task_launch_sha256(launch),
            reservation.system_identity_sha256,
            binding.episode_id,
            launch.run_id,
            binding.workspace_manifest_sha256,
            binding.workspace_tree_sha256,
            binding.model_visible_surface_sha256,
            reservation.system.worker_spec_sha256,
            reservation.system.harness.harness_image_or_commitment.removeprefix('sha256:'),
        )
        observed = (
            prepared.runtime_id,
            prepared.runtime_version,
            prepared.runtime_executable_sha256,
            prepared.runtime_config_sha256,
            prepared.launcher_deployment_sha256,
            prepared.launch_sha256,
            prepared.system_identity_sha256,
            prepared.episode_id,
            prepared.run_id,
            prepared.workspace_manifest_sha256,
            prepared.workspace_tree_sha256,
            prepared.model_visible_surface_sha256,
            prepared.worker_spec_sha256,
            prepared.harness_sha256,
        )
        if prepared.reservation_sha256 != launch.reservation_sha256:
            raise ClinicalLauncherError('prepared runtime reservation differs from the launch')
        if observed != expected or prepared.prepared_at < launch.claimed_at:
            raise ClinicalLauncherError('prepared runtime differs from its deployment, launch, or task')

    def _discard_prepared(self, prepared: ClinicalPreparedRuntime) -> bool:
        try:
            self.runtime.discard_prepared(prepared)
        except Exception:
            return False
        return True

    def _record_failure(
        self,
        *,
        reservation: ClinicalProductionReservation,
        launch: ClinicalProductionTaskLaunch,
        start_redemption: ClinicalProductionStartRedemption | None,
        phase: ClinicalLauncherPhase,
        code: ClinicalLauncherFailureCode,
        terminal_code: ClinicalProductionExplicitFailureCode,
        prepared_runtime_discarded: bool | None,
        authenticated_bootstrap_sha256: str | None = None,
    ) -> ClinicalLauncherFailure:
        return _record_canonical_clinical_failure(
            registry=self.registry,
            deployment=self.deployment,
            failure_receipt_key=self._failure_receipt_key,
            reservation=reservation,
            launch=launch,
            start_redemption=start_redemption,
            phase=phase,
            code=code,
            terminal_code=terminal_code,
            prepared_runtime_discarded=prepared_runtime_discarded,
            authenticated_bootstrap_sha256=authenticated_bootstrap_sha256,
            now=self._now(),
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ClinicalLauncherError('launcher clock must return a timezone-aware timestamp')
        return value.astimezone(UTC)


class CanonicalClinicalRecoveryTerminalizer:
    """Terminalize already-consumed attempts without possessing a runtime or evidence loader.

    This deliberately exposes no prepare, run, workspace, provider, or model-call dependency.
    It can only inspect one authoritative reservation and convert records which are already in the
    consumed ``launched`` state into permanent failures.  Repeating it is therefore idempotent.
    """

    def __init__(
        self,
        *,
        registry: ClinicalProductionRegistryBoundary,
        deployment: CanonicalClinicalLauncherDeployment,
        failure_receipt_key: bytes,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if registry.authority_id != deployment.registry_authority_id:
            raise ClinicalLauncherError('recovery terminalizer belongs to a different registry authority')
        if clinical_launcher_failure_key_id(failure_receipt_key) != deployment.failure_receipt_key_id:
            raise ClinicalLauncherError('recovery failure key differs from its deployment pin')
        self.registry = registry
        self.deployment = deployment
        self._failure_receipt_key = bytes(failure_receipt_key)
        self._clock = clock or (lambda: datetime.now(UTC))

    def reconcile_consumed_tasks(
        self,
        *,
        reservation_sha256: str,
    ) -> tuple[ClinicalLauncherFailure, ...]:
        """Make every consumed, nonterminal attempt permanent; never claim or launch work."""

        context = self.registry.reservation_context(reservation_sha256)
        reservation = context.reservation
        _validate_recovery_reservation_deployment(
            reservation,
            deployment=self.deployment,
        )
        failures: list[ClinicalLauncherFailure] = []
        for record in self.registry.task_records(reservation_sha256):
            if record.state != 'launched':
                continue
            launch = record.launch
            if launch is None:
                raise ClinicalLauncherError('launched registry task is missing its launch receipt')
            redemption = record.start_redemption
            failures.append(
                _record_canonical_clinical_failure(
                    registry=self.registry,
                    deployment=self.deployment,
                    failure_receipt_key=self._failure_receipt_key,
                    reservation=reservation,
                    launch=launch,
                    start_redemption=redemption,
                    phase=ClinicalLauncherPhase.RECONCILE,
                    code=(
                        ClinicalLauncherFailureCode.RECONCILED_UNREDEEMED_CLAIM
                        if redemption is None
                        else ClinicalLauncherFailureCode.RECONCILED_REDEEMED_START
                    ),
                    terminal_code=(
                        ClinicalProductionTerminalCode.SCHEDULER_FAILURE
                        if redemption is None
                        else ClinicalProductionTerminalCode.WORKER_LOST
                    ),
                    prepared_runtime_discarded=None,
                    authenticated_bootstrap_sha256=None,
                    now=self._now(),
                )
            )
        return tuple(failures)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ClinicalLauncherError('recovery terminalizer clock must return a timezone-aware timestamp')
        return value.astimezone(UTC)


def _validate_recovery_reservation_deployment(
    reservation: ClinicalProductionReservation,
    *,
    deployment: CanonicalClinicalLauncherDeployment,
) -> None:
    system = reservation.system
    if (
        reservation.registry_authority_id,
        reservation.system_identity_sha256,
        clinical_production_system_identity_sha256(system),
        system.canonical_launcher_id,
        system.canonical_launcher_executable_sha256,
    ) != (
        deployment.registry_authority_id,
        deployment.expected_system_identity_sha256,
        deployment.expected_system_identity_sha256,
        deployment.canonical_launcher_id,
        deployment.canonical_launcher_executable_sha256,
    ):
        raise ClinicalLauncherError('launcher deployment does not match the reserved system and authority')


def _record_canonical_clinical_failure(
    *,
    registry: ClinicalProductionRegistryBoundary,
    deployment: CanonicalClinicalLauncherDeployment,
    failure_receipt_key: bytes,
    reservation: ClinicalProductionReservation,
    launch: ClinicalProductionTaskLaunch,
    start_redemption: ClinicalProductionStartRedemption | None,
    phase: ClinicalLauncherPhase,
    code: ClinicalLauncherFailureCode,
    terminal_code: ClinicalProductionExplicitFailureCode,
    prepared_runtime_discarded: bool | None,
    authenticated_bootstrap_sha256: str | None,
    now: datetime,
) -> ClinicalLauncherFailure:
    lower_bound = launch.claimed_at
    if start_redemption is not None:
        lower_bound = max(lower_bound, start_redemption.redeemed_at)
    failed_at = max(now, lower_bound)
    receipt = ClinicalLauncherFailureReceipt(
        registry_authority_id=deployment.registry_authority_id,
        reservation_sha256=launch.reservation_sha256,
        system_identity_sha256=reservation.system_identity_sha256,
        episode_id=launch.episode_id,
        run_id=launch.run_id,
        launch_sha256=clinical_production_task_launch_sha256(launch),
        start_redemption_sha256=(
            None if start_redemption is None else clinical_production_start_redemption_sha256(start_redemption)
        ),
        authenticated_bootstrap_sha256=authenticated_bootstrap_sha256,
        launcher_deployment_sha256=canonical_clinical_launcher_deployment_sha256(deployment),
        launcher_failure_receipt_key_id=deployment.failure_receipt_key_id,
        phase=phase,
        failure_code=code,
        registry_terminal_code=terminal_code,
        prepared_runtime_discarded=prepared_runtime_discarded,
        failed_at=failed_at,
    )
    artifact = AuthenticatedClinicalLauncherFailure(
        receipt=receipt,
        receipt_hmac_sha256=clinical_launcher_failure_hmac(
            receipt,
            failure_receipt_key,
        ),
    )
    verify_authenticated_clinical_launcher_failure(
        artifact,
        key=failure_receipt_key,
        expected_key_id=deployment.failure_receipt_key_id,
    )
    record = registry.record_explicit_failure(
        reservation_sha256=launch.reservation_sha256,
        episode_id=launch.episode_id,
        terminal_code=terminal_code,
        failure_record=canonical_json_bytes(artifact),
        terminal_at=failed_at,
    )
    return ClinicalLauncherFailure(
        record=record,
        failure_code=code,
        launch=launch,
        start_redemption=start_redemption,
        authenticated_failure=artifact,
    )


def _runtime_failure_mapping(
    code: ClinicalRuntimeFailureCode,
) -> tuple[ClinicalLauncherFailureCode, ClinicalProductionExplicitFailureCode]:
    mapping: dict[
        ClinicalRuntimeFailureCode,
        tuple[ClinicalLauncherFailureCode, ClinicalProductionExplicitFailureCode],
    ] = {
        ClinicalRuntimeFailureCode.LAUNCH_FAILED: (
            ClinicalLauncherFailureCode.RUNTIME_LAUNCH_FAILED,
            ClinicalProductionTerminalCode.WORKER_LAUNCH_FAILURE,
        ),
        ClinicalRuntimeFailureCode.BOOTSTRAP_FAILED: (
            ClinicalLauncherFailureCode.RUNTIME_BOOTSTRAP_FAILED,
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
        ),
        ClinicalRuntimeFailureCode.WORKER_TERMINAL_FAILURE: (
            ClinicalLauncherFailureCode.RUNTIME_TERMINAL_FAILURE,
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
        ),
        ClinicalRuntimeFailureCode.WORKER_LOST: (
            ClinicalLauncherFailureCode.RUNTIME_WORKER_LOST,
            ClinicalProductionTerminalCode.WORKER_LOST,
        ),
        ClinicalRuntimeFailureCode.CLEANUP_FAILED: (
            ClinicalLauncherFailureCode.RUNTIME_CLEANUP_FAILED,
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
        ),
        ClinicalRuntimeFailureCode.EVIDENCE_FINALIZATION_FAILED: (
            ClinicalLauncherFailureCode.EVIDENCE_FINALIZATION_FAILED,
            ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE,
        ),
    }
    return mapping[code]


def _registry_failure_mapping(code: ClinicalProductionTerminalCode | None) -> ClinicalLauncherFailureCode:
    mapping = {
        ClinicalProductionTerminalCode.EVIDENCE_AUTHENTICATION_FAILED: (
            ClinicalLauncherFailureCode.EVIDENCE_AUTHENTICATION_FAILED
        ),
        ClinicalProductionTerminalCode.EVIDENCE_BINDING_MISMATCH: (
            ClinicalLauncherFailureCode.EVIDENCE_BINDING_MISMATCH
        ),
        ClinicalProductionTerminalCode.INVALID_CLINICAL_SUBMISSION: (
            ClinicalLauncherFailureCode.INVALID_CLINICAL_SUBMISSION
        ),
        ClinicalProductionTerminalCode.WORKER_TERMINAL_FAILURE: (ClinicalLauncherFailureCode.RUNTIME_TERMINAL_FAILURE),
    }
    if code not in mapping:
        raise ClinicalLauncherError('registry returned an unexpected terminal code after evidence collection')
    return mapping[code]


def _require_failure_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ClinicalLauncherError('launcher failure receipt key must contain at least 32 bytes')


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ValueError(f'{label} must be a lowercase SHA-256 digest')


__all__ = [
    'AUTHENTICATED_CLINICAL_LAUNCHER_FAILURE_SCHEMA_VERSION',
    'CANONICAL_CLINICAL_LAUNCHER_DEPLOYMENT_SCHEMA_VERSION',
    'CLINICAL_LAUNCHER_FAILURE_SCHEMA_VERSION',
    'CLINICAL_PREPARED_RUNTIME_SCHEMA_VERSION',
    'CLINICAL_RUNTIME_START_SCHEMA_VERSION',
    'AuthenticatedClinicalLauncherFailure',
    'CanonicalClinicalLauncher',
    'CanonicalClinicalRecoveryTerminalizer',
    'CanonicalClinicalLauncherDeployment',
    'ClinicalLauncherError',
    'ClinicalLauncherFailure',
    'ClinicalLauncherFailureCode',
    'ClinicalLauncherFailureReceipt',
    'ClinicalLauncherPhase',
    'ClinicalLauncherResult',
    'ClinicalLauncherSuccess',
    'ClinicalPreparedRuntime',
    'ClinicalProductionRunLoader',
    'ClinicalRuntimeBoundary',
    'ClinicalRuntimeCompleted',
    'ClinicalRuntimeFailed',
    'ClinicalRuntimeFailureCode',
    'ClinicalRuntimeOutcome',
    'ClinicalRuntimePrepareRequest',
    'ClinicalRuntimeStart',
    'canonical_clinical_launcher_deployment_sha256',
    'clinical_launcher_failure_hmac',
    'clinical_launcher_failure_key_id',
    'clinical_prepared_runtime_sha256',
    'verify_authenticated_clinical_launcher_failure',
]
