"""Retry-safe execution of immutable exact-HTTPS collection plans."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.http_capture import (
    BodyTooLargeError,
    CaptureDeadlineExceededError,
    DnsAddressLimitError,
    DnsResolutionTimeoutError,
    HttpsCaptureError,
    HttpsCaptureReceipt,
    HttpsCaptureRequest,
    RedirectRejectedError,
    ResponseProtocolError,
    UnexpectedStatusError,
    UrllibHttpsTransport,
    capture_https_to_tempfile,
    prepared_request_headers,
)
from vaxreplay.operations.policy import (
    STATIC_HTTPS_COLLECTOR_ID,
    StaticHttpsJobConfiguration,
    parse_static_job_configuration,
)
from vaxreplay.operations.schema import (
    ARTIFACT_ROLE_PATTERN,
    AttemptLease,
    AttemptState,
    LedgerEventType,
    LogicalRunRecord,
    StoredArtifact,
    aware_utc,
)
from vaxreplay.operations.store import OperationalStore

STATIC_HTTPS_PLAN_SCHEMA_VERSION = 'vaxreplay.static-https-plan.v0.1'
STATIC_HTTPS_RUN_SCHEMA_VERSION = 'vaxreplay.static-https-run.v0.1'
STATIC_HTTPS_FAILURE_SCHEMA_VERSION = 'vaxreplay.static-https-failure.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class StaticCollectionError(RuntimeError):
    """An immutable plan cannot be executed against the registered job."""


class StaticCollectionAttemptError(StaticCollectionError):
    """An already-claimed attempt failed and was terminalized or abandoned."""

    def __init__(self, attempt_id: str, cause: Exception) -> None:
        super().__init__(f'static collection attempt {attempt_id} failed: {type(cause).__name__}')
        self.attempt_id = attempt_id
        self.cause = cause


class StaticHttpsArtifactSpec(StrictModel):
    """One exact response entity required by a static collection plan."""

    artifact_id: str = Field(pattern=ARTIFACT_ROLE_PATTERN, max_length=110)
    request: HttpsCaptureRequest


class StaticHttpsCollectionPlan(StrictModel):
    """Immutable list of official publication artifacts required for one slot."""

    schema_version: Literal['vaxreplay.static-https-plan.v0.1'] = STATIC_HTTPS_PLAN_SCHEMA_VERSION
    plan_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    # A complete IEDB IQ-API capture may require thousands of fixed Range pages.
    # The plan remains bounded and every response retains its own stricter byte cap.
    artifacts: tuple[StaticHttpsArtifactSpec, ...] = Field(min_length=1, max_length=8192)

    @field_validator('artifacts')
    @classmethod
    def validate_artifacts(cls, value: tuple[StaticHttpsArtifactSpec, ...]) -> tuple[StaticHttpsArtifactSpec, ...]:
        identifiers = tuple(item.artifact_id for item in value)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError('plan artifacts must use unique artifact_id values in sorted order')
        return value


def static_plan_sha256(plan: StaticHttpsCollectionPlan) -> str:
    return hashlib.sha256(canonical_json_bytes(plan)).hexdigest()


class CollectedHttpsArtifact(StrictModel):
    artifact_id: str = Field(pattern=ARTIFACT_ROLE_PATTERN, max_length=110)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    body_sha256: str = Field(pattern=_SHA256_PATTERN)
    body_byte_count: int = Field(ge=0)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    receipt_byte_count: int = Field(gt=0)
    started_at: datetime
    completed_at: datetime

    @field_validator('started_at', 'completed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'collection timestamp')

    @model_validator(mode='after')
    def validate_times(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError('artifact completion cannot precede its start')
        return self


class StaticHttpsRunManifest(StrictModel):
    """Successful terminal manifest; completeness is limited to the frozen plan."""

    schema_version: Literal['vaxreplay.static-https-run.v0.1'] = STATIC_HTTPS_RUN_SCHEMA_VERSION
    logical_run_id: str = Field(pattern=r'^run-[0-9a-f]{64}$')
    attempt_id: str = Field(pattern=r'^attempt-[0-9a-f]{32}$')
    job_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    scheduled_for: datetime
    plan_id: str = Field(min_length=1, max_length=200)
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_id: str = Field(min_length=1, max_length=200)
    attempt_started_at: datetime
    completed_at: datetime
    artifacts: tuple[CollectedHttpsArtifact, ...] = Field(min_length=1)
    plan_complete: Literal[True] = True
    source_enumeration_complete: Literal[False] = False
    external_timestamp_required: Literal[True] = True
    tier_a_eligible: Literal[False] = False

    @field_validator('scheduled_for', 'attempt_started_at', 'completed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'run timestamp')

    @model_validator(mode='after')
    def validate_manifest(self) -> Self:
        identifiers = tuple(item.artifact_id for item in self.artifacts)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError('collected artifacts must use canonical unique artifact order')
        if self.completed_at < self.attempt_started_at:
            raise ValueError('run completion cannot precede attempt start')
        if self.completed_at != max(item.completed_at for item in self.artifacts):
            raise ValueError('completed_at must equal the latest artifact completion')
        return self


class StaticHttpsFailureRecord(StrictModel):
    """Sanitized failure observation retained before terminalizing an attempt."""

    schema_version: Literal['vaxreplay.static-https-failure.v0.1'] = STATIC_HTTPS_FAILURE_SCHEMA_VERSION
    logical_run_id: str = Field(pattern=r'^run-[0-9a-f]{64}$')
    attempt_id: str = Field(pattern=r'^attempt-[0-9a-f]{32}$')
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_id: str = Field(pattern=ARTIFACT_ROLE_PATTERN, max_length=110)
    failure_code: str = Field(pattern=r'^[a-z][a-z0-9_]{0,99}$')
    observed_at: datetime
    http_status: int | None = Field(default=None, ge=100, le=599)
    final_url: str | None = Field(default=None, max_length=8192)

    @field_validator('observed_at')
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'observed_at')


@dataclass(frozen=True)
class StaticHttpsCollectionResult:
    attempt: AttemptLease
    manifest: StaticHttpsRunManifest
    manifest_artifact: StoredArtifact


@dataclass
class _MonotonicBudget:
    """One non-resettable plan budget shared with request transports."""

    clock: Callable[[], float]
    deadline: float
    last_value: float

    @classmethod
    def start(cls, clock: Callable[[], float], duration_seconds: int) -> _MonotonicBudget:
        if not callable(clock):
            raise StaticCollectionError('monotonic clock must be callable')
        try:
            value = clock()
        except Exception as error:
            raise StaticCollectionError('monotonic clock failed before collection') from error
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise StaticCollectionError('monotonic clock must return a finite numeric value')
        initial = float(value)
        return cls(clock=clock, deadline=initial + duration_seconds, last_value=initial)

    def now(self) -> float:
        try:
            value = self.clock()
        except Exception as error:
            raise StaticCollectionError('monotonic clock failed during collection') from error
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise StaticCollectionError('monotonic clock must return a finite numeric value')
        current = float(value)
        if current < self.last_value:
            raise StaticCollectionError('monotonic clock moved backwards during collection')
        self.last_value = current
        return current

    def remaining(self, *, deadline: float | None = None) -> float:
        active_deadline = self.deadline if deadline is None else min(self.deadline, deadline)
        remaining = active_deadline - self.now()
        if remaining <= 0:
            raise CaptureDeadlineExceededError('static HTTPS collection exhausted its monotonic deadline')
        return remaining


def _clock_utc() -> datetime:
    return datetime.now(timezone.utc)


def _operation_time(clock: Callable[[], datetime], field_name: str) -> datetime:
    try:
        return aware_utc(clock(), field_name)
    except (AttributeError, TypeError, ValueError) as error:
        raise StaticCollectionError(f'{field_name} clock value must be an offset-aware datetime') from error


def validate_static_job_configuration(
    configuration: dict[str, str | int | bool],
) -> StaticHttpsJobConfiguration:
    """Parse the exact static-collector policy, including committed lease and retry limits."""

    try:
        return parse_static_job_configuration(configuration)
    except ValueError as error:
        raise StaticCollectionError('static HTTPS job configuration is not on the collector allowlist') from error


def run_static_https_collection(
    store: OperationalStore,
    logical_run_id: str,
    plan: StaticHttpsCollectionPlan,
    *,
    owner_id: str,
    lease_seconds: int | None = None,
    max_attempts: int | None = None,
    clock: Callable[[], datetime] = _clock_utc,
    monotonic: Callable[[], float] = time.monotonic,
) -> StaticHttpsCollectionResult:
    """Claim and execute one slot, retaining every completed artifact and failure.

    The registered job must bind ``collection_plan_sha256`` and use
    :data:`STATIC_HTTPS_COLLECTOR_ID`.  This worker establishes that every URL in the
    immutable plan succeeded; it deliberately does not claim complete enumeration of a
    changing source, an independently proven time, or Tier A eligibility.

    The official entrypoint deliberately has no caller-supplied transport or resolver:
    arbitrary in-process I/O code cannot be forcibly interrupted or authenticated as the
    deadline-qualified default transport. Tests exercise dependency injection at the lower
    :class:`~vaxreplay.operations.http_capture.UrllibHttpsTransport` boundary.
    """

    run = store.get_logical_run(logical_run_id)
    job = store.get_job(run.job_spec_sha256)
    plan_sha256 = static_plan_sha256(plan)
    if job.spec.collector_id != STATIC_HTTPS_COLLECTOR_ID:
        raise StaticCollectionError(
            f'job collector_id must be {STATIC_HTTPS_COLLECTOR_ID!r}, found {job.spec.collector_id!r}'
        )
    configuration = validate_static_job_configuration(job.spec.configuration)
    if configuration.collection_plan_sha256 != plan_sha256:
        raise StaticCollectionError('registered job does not bind the supplied collection plan')
    if configuration.source_id != plan.source_id:
        raise StaticCollectionError('registered job source_id does not match the supplied collection plan')
    if lease_seconds is not None and lease_seconds != configuration.lease_seconds:
        raise StaticCollectionError('requested lease_seconds differs from the immutable job configuration')
    selected_lease_seconds = configuration.lease_seconds
    if max_attempts is not None and max_attempts != configuration.max_attempts_per_slot:
        raise StaticCollectionError('requested max_attempts differs from the immutable job configuration')
    selected_max_attempts = configuration.max_attempts_per_slot
    budget = _MonotonicBudget.start(monotonic, configuration.plan_deadline_seconds)

    plan_artifact = store.put_bytes(
        canonical_json_bytes(plan),
        recorded_at=_operation_time(clock, 'collection_plan_recorded_at'),
    )
    budget.remaining()
    requested_attempt_started_at = _operation_time(clock, 'attempt_started_at')
    attempt = store.begin_attempt(
        logical_run_id,
        owner_id=owner_id,
        now=requested_attempt_started_at,
        lease_seconds=selected_lease_seconds,
        max_attempts=selected_max_attempts,
        initial_artifacts={'collection-plan': plan_artifact.sha256},
    )
    try:
        return _execute_static_attempt(
            store,
            run=run,
            plan=plan,
            plan_sha256=plan_sha256,
            attempt=attempt,
            owner_id=owner_id,
            attempt_started_at=attempt.started_at,
            clock=clock,
            budget=budget,
            configuration=configuration,
        )
    except Exception as error:
        _terminalize_unfinished_attempt(
            store,
            attempt,
            plan_sha256=plan_sha256,
            owner_id=owner_id,
            error=error,
            clock=clock,
        )
        raise StaticCollectionAttemptError(attempt.attempt_id, error) from error


def _execute_static_attempt(
    store: OperationalStore,
    *,
    run: LogicalRunRecord,
    plan: StaticHttpsCollectionPlan,
    plan_sha256: str,
    attempt: AttemptLease,
    owner_id: str,
    attempt_started_at: datetime,
    clock: Callable[[], datetime],
    budget: _MonotonicBudget,
    configuration: StaticHttpsJobConfiguration,
) -> StaticHttpsCollectionResult:
    collected: list[CollectedHttpsArtifact] = []
    total_body_bytes = 0
    for artifact_spec in plan.artifacts:
        try:
            remaining_body_bytes = configuration.max_total_body_bytes - total_body_bytes
            if remaining_body_bytes <= 0:
                raise BodyTooLargeError('static HTTPS plan exhausted its aggregate body-byte budget')
            request_started = budget.now()
            request_deadline = min(
                budget.deadline,
                request_started + configuration.request_deadline_seconds,
            )
            request_remaining = budget.remaining(deadline=request_deadline)
            effective_request = artifact_spec.request.model_copy(
                update={
                    'max_body_bytes': min(
                        artifact_spec.request.max_body_bytes,
                        remaining_body_bytes,
                    ),
                    'timeout_seconds': min(
                        artifact_spec.request.timeout_seconds,
                        request_remaining,
                    ),
                }
            )
            active_transport = UrllibHttpsTransport(
                monotonic=budget.now,
                deadline_monotonic=request_deadline,
                dns_timeout_seconds=configuration.dns_resolution_timeout_seconds,
                dns_resolution_attempts=configuration.dns_resolution_attempts,
                max_dns_addresses=configuration.max_dns_addresses,
            )
            temporary = capture_https_to_tempfile(
                effective_request,
                directory=store.root,
                transport=active_transport,
                clock=clock,
            )
            try:
                budget.remaining(deadline=request_deadline)
                if temporary.receipt.body_byte_count > remaining_body_bytes:
                    raise BodyTooLargeError('HTTPS response exceeds the remaining aggregate body-byte budget')
                artifact_recorded_at = _operation_time(clock, 'artifact_recorded_at')
                if artifact_recorded_at >= attempt.lease_expires_at:
                    raise StaticCollectionError('artifact processing reached or passed the attempt lease expiry')
                body_artifact = store.put_file(
                    temporary.path,
                    recorded_at=artifact_recorded_at,
                    max_bytes=artifact_spec.request.max_body_bytes,
                )
                budget.remaining()
            finally:
                temporary.delete()
            receipt_bytes = canonical_json_bytes(temporary.receipt)
            receipt_artifact = store.put_bytes(receipt_bytes, recorded_at=artifact_recorded_at)
            budget.remaining()
            store.attach_artifact(
                attempt.attempt_id,
                owner_id=owner_id,
                role=f'body.{artifact_spec.artifact_id}',
                artifact_sha256=body_artifact.sha256,
                now=_operation_time(clock, 'body_attached_at'),
            )
            budget.remaining()
            store.attach_artifact(
                attempt.attempt_id,
                owner_id=owner_id,
                role=f'receipt.{artifact_spec.artifact_id}',
                artifact_sha256=receipt_artifact.sha256,
                now=_operation_time(clock, 'receipt_attached_at'),
            )
            budget.remaining()
            collected.append(
                _collected_artifact(
                    artifact_spec,
                    temporary.receipt,
                    body_artifact=body_artifact,
                    receipt_artifact=receipt_artifact,
                )
            )
            total_body_bytes += temporary.receipt.body_byte_count
        except HttpsCaptureError as error:
            _retain_failure(
                store,
                attempt,
                plan_sha256=plan_sha256,
                artifact_id=artifact_spec.artifact_id,
                error=error,
                owner_id=owner_id,
                clock=clock,
            )
            raise

    budget.remaining()
    completed_at = max(item.completed_at for item in collected)
    manifest = StaticHttpsRunManifest(
        logical_run_id=run.logical_run_id,
        attempt_id=attempt.attempt_id,
        job_spec_sha256=run.job_spec_sha256,
        scheduled_for=run.scheduled_for,
        plan_id=plan.plan_id,
        plan_sha256=plan_sha256,
        source_id=plan.source_id,
        attempt_started_at=attempt_started_at,
        completed_at=completed_at,
        artifacts=tuple(collected),
    )
    manifest_artifact = store.put_bytes(
        canonical_json_bytes(manifest),
        recorded_at=_operation_time(clock, 'run_manifest_recorded_at'),
    )
    budget.remaining()
    attempt_finished_at = _operation_time(clock, 'attempt_finished_at')
    budget.remaining()
    terminal = store.succeed_attempt(
        attempt.attempt_id,
        owner_id=owner_id,
        run_manifest_sha256=manifest_artifact.sha256,
        now=attempt_finished_at,
    )
    return StaticHttpsCollectionResult(attempt=terminal, manifest=manifest, manifest_artifact=manifest_artifact)


def _collected_artifact(
    artifact_spec: StaticHttpsArtifactSpec,
    receipt: HttpsCaptureReceipt,
    *,
    body_artifact: StoredArtifact,
    receipt_artifact: StoredArtifact,
) -> CollectedHttpsArtifact:
    return CollectedHttpsArtifact(
        artifact_id=artifact_spec.artifact_id,
        request_sha256=hashlib.sha256(canonical_json_bytes(artifact_spec.request)).hexdigest(),
        body_sha256=body_artifact.sha256,
        body_byte_count=body_artifact.byte_count,
        receipt_sha256=receipt_artifact.sha256,
        receipt_byte_count=receipt_artifact.byte_count,
        started_at=receipt.started_at,
        completed_at=receipt.completed_at,
    )


def _retain_failure(
    store: OperationalStore,
    attempt: AttemptLease,
    *,
    plan_sha256: str,
    artifact_id: str,
    error: HttpsCaptureError,
    owner_id: str,
    clock: Callable[[], datetime],
) -> None:
    observed_at = aware_utc(clock(), 'failure observed_at')
    code = _failure_code(error)
    status = error.status_code if isinstance(error, UnexpectedStatusError) else None
    final_url = error.final_url if isinstance(error, UnexpectedStatusError) else None
    _retain_failure_record(
        store,
        attempt,
        plan_sha256=plan_sha256,
        artifact_id=artifact_id,
        failure_code=code,
        observed_at=observed_at,
        owner_id=owner_id,
        http_status=status,
        final_url=final_url,
    )


def _retain_failure_record(
    store: OperationalStore,
    attempt: AttemptLease,
    *,
    plan_sha256: str,
    artifact_id: str,
    failure_code: str,
    observed_at: datetime,
    owner_id: str,
    http_status: int | None = None,
    final_url: str | None = None,
) -> None:
    failure = StaticHttpsFailureRecord(
        logical_run_id=attempt.logical_run_id,
        attempt_id=attempt.attempt_id,
        plan_sha256=plan_sha256,
        artifact_id=artifact_id,
        failure_code=failure_code,
        observed_at=observed_at,
        http_status=http_status,
        final_url=final_url,
    )
    failure_artifact = store.put_bytes(canonical_json_bytes(failure), recorded_at=observed_at)
    store.attach_artifact(
        attempt.attempt_id,
        owner_id=owner_id,
        role='failure-record',
        artifact_sha256=failure_artifact.sha256,
        now=observed_at,
    )
    store.fail_attempt(
        attempt.attempt_id,
        owner_id=owner_id,
        terminal_code=failure_code,
        now=observed_at,
    )


def _terminalize_unfinished_attempt(
    store: OperationalStore,
    attempt: AttemptLease,
    *,
    plan_sha256: str,
    owner_id: str,
    error: Exception,
    clock: Callable[[], datetime],
) -> None:
    """Best-effort terminalization that never masks the collector's original error."""

    try:
        current = store.get_attempt(attempt.attempt_id)
    except Exception:
        return
    if current.state is not AttemptState.STARTED:
        return
    try:
        observed_at = aware_utc(clock(), 'failure observed_at')
    except Exception:
        observed_at = datetime.now(timezone.utc)
    if observed_at >= current.lease_expires_at:
        try:
            store.abandon_expired_attempts(now=observed_at)
        except Exception:
            pass
        return
    failure_code = _failure_code(error) if isinstance(error, HttpsCaptureError) else 'collector_internal_error'
    try:
        _retain_failure_record(
            store,
            current,
            plan_sha256=plan_sha256,
            artifact_id='worker',
            failure_code=failure_code,
            observed_at=observed_at,
            owner_id=owner_id,
        )
    except Exception:
        try:
            store.fail_attempt(
                current.attempt_id,
                owner_id=owner_id,
                terminal_code=failure_code,
                now=observed_at,
            )
        except Exception:
            pass


def _failure_code(error: HttpsCaptureError) -> str:
    if isinstance(error, CaptureDeadlineExceededError):
        return 'capture_deadline_exceeded'
    if isinstance(error, DnsResolutionTimeoutError):
        return 'dns_resolution_timeout'
    if isinstance(error, DnsAddressLimitError):
        return 'dns_address_limit_exceeded'
    if isinstance(error, UnexpectedStatusError):
        return f'http_status_{error.status_code}'
    if isinstance(error, RedirectRejectedError):
        return 'redirect_rejected'
    if isinstance(error, BodyTooLargeError):
        return 'body_too_large'
    if isinstance(error, ResponseProtocolError):
        return 'response_protocol_error'
    return 'https_capture_error'


def _verify_static_attempt_policy(
    store: OperationalStore,
    run: LogicalRunRecord,
    configuration: StaticHttpsJobConfiguration,
) -> None:
    """Replay immutable plan, due-time, lease, and retry policy for every attempt."""

    attempt_history = store.list_attempts(logical_run_id=run.logical_run_id)
    if tuple(item.attempt_number for item in attempt_history) != tuple(range(1, len(attempt_history) + 1)):
        raise StaticCollectionError('static attempt history is not contiguous')
    if len(attempt_history) > configuration.max_attempts_per_slot:
        raise StaticCollectionError('static attempt history exceeds the immutable retry budget')
    expected_lease_duration = timedelta(seconds=configuration.lease_seconds)
    for historical_attempt in attempt_history:
        if historical_attempt.started_at < run.scheduled_for:
            raise StaticCollectionError('static attempt predates its scheduled slot')
        if historical_attempt.lease_expires_at - historical_attempt.started_at != expected_lease_duration:
            raise StaticCollectionError('static attempt lease differs from the immutable job policy')
        if (
            historical_attempt.state is AttemptState.SUCCEEDED
            and historical_attempt.finished_at is not None
            and (historical_attempt.finished_at - historical_attempt.started_at).total_seconds()
            > configuration.plan_deadline_seconds
        ):
            raise StaticCollectionError('successful static attempt exceeds the immutable plan deadline')
        plan_artifact = store.list_attempt_artifacts(historical_attempt.attempt_id).get('collection-plan')
        if plan_artifact is None or plan_artifact.sha256 != configuration.collection_plan_sha256:
            raise StaticCollectionError('static attempt does not bind its immutable collection plan')
    history_ids = {item.attempt_id for item in attempt_history}
    if any(
        event.event_type is LedgerEventType.ATTEMPT_LEASE_RENEWED and event.payload.get('attempt_id') in history_ids
        for event in store.events()
    ):
        raise StaticCollectionError('static attempt history contains a forbidden lease renewal')


def load_static_run_manifest(store: OperationalStore, attempt_id: str) -> StaticHttpsRunManifest:
    """Replay and cross-check a successful static run from every durable object.

    This is the trusted read path for this collector.  The generic operations store
    proves structural integrity; this function additionally reconstructs the committed
    plan and receipts and rejects self-consistent but semantically forged manifests.
    """

    attempt = store.get_attempt(attempt_id)
    if attempt.state is not AttemptState.SUCCEEDED:
        raise StaticCollectionError('static run manifest can be loaded only from a successful attempt')
    run = store.get_logical_run(attempt.logical_run_id)
    if run.successful_attempt_id != attempt_id:
        raise StaticCollectionError('logical run does not identify this successful attempt')
    job = store.get_job(run.job_spec_sha256)
    if job.spec.collector_id != STATIC_HTTPS_COLLECTOR_ID:
        raise StaticCollectionError('successful attempt is not owned by the static HTTPS collector')
    configuration = validate_static_job_configuration(job.spec.configuration)

    _verify_static_attempt_policy(store, run, configuration)

    attachments = store.list_attempt_artifacts(attempt_id)
    plan_artifact = attachments.get('collection-plan')
    manifest_artifact = attachments.get('run-manifest')
    if plan_artifact is None or manifest_artifact is None:
        raise StaticCollectionError('successful attempt is missing its plan or run manifest attachment')

    plan_bytes = store.read_artifact(plan_artifact.sha256)
    try:
        plan = StaticHttpsCollectionPlan.model_validate_json(plan_bytes)
    except ValueError as error:
        raise StaticCollectionError('collection-plan attachment is invalid') from error
    if canonical_json_bytes(plan) != plan_bytes:
        raise StaticCollectionError('collection-plan attachment is not canonical JSON')
    plan_sha256 = static_plan_sha256(plan)
    if plan_artifact.sha256 != plan_sha256 or configuration.collection_plan_sha256 != plan_sha256:
        raise StaticCollectionError('collection plan is not bound by its attachment and registered job')
    if configuration.source_id != plan.source_id:
        raise StaticCollectionError('collection plan source is not bound by the registered job')

    manifest_bytes = store.read_artifact(manifest_artifact.sha256)
    try:
        manifest = StaticHttpsRunManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise StaticCollectionError('run-manifest attachment is invalid') from error
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise StaticCollectionError('run-manifest attachment is not canonical JSON')
    if (
        manifest.attempt_id != attempt_id
        or manifest.logical_run_id != run.logical_run_id
        or manifest.job_spec_sha256 != run.job_spec_sha256
        or manifest.scheduled_for != run.scheduled_for
        or manifest.attempt_started_at != attempt.started_at
    ):
        raise StaticCollectionError('run manifest does not bind its attempt, job revision, and schedule slot')
    if manifest.plan_id != plan.plan_id or manifest.plan_sha256 != plan_sha256 or manifest.source_id != plan.source_id:
        raise StaticCollectionError('run manifest does not bind the committed collection plan')
    if attempt.finished_at is None or manifest.completed_at > attempt.finished_at:
        raise StaticCollectionError('run manifest completion is later than its durable terminal event')

    expected_roles = {'collection-plan', 'run-manifest'}
    expected_roles.update(f'body.{item.artifact_id}' for item in plan.artifacts)
    expected_roles.update(f'receipt.{item.artifact_id}' for item in plan.artifacts)
    if set(attachments) != expected_roles:
        raise StaticCollectionError('successful attempt attachment set does not exactly match its plan')
    if tuple(item.artifact_id for item in manifest.artifacts) != tuple(item.artifact_id for item in plan.artifacts):
        raise StaticCollectionError('run manifest artifact IDs do not exactly match its plan')

    total_body_bytes = 0
    for artifact_spec, item in zip(plan.artifacts, manifest.artifacts, strict=True):
        body = attachments.get(f'body.{item.artifact_id}')
        receipt = attachments.get(f'receipt.{item.artifact_id}')
        if (
            body is None
            or receipt is None
            or (body.sha256, body.byte_count) != (item.body_sha256, item.body_byte_count)
            or (receipt.sha256, receipt.byte_count) != (item.receipt_sha256, item.receipt_byte_count)
        ):
            raise StaticCollectionError(f'run manifest artifact binding failed for {item.artifact_id}')
        expected_request_sha256 = hashlib.sha256(canonical_json_bytes(artifact_spec.request)).hexdigest()
        if item.request_sha256 != expected_request_sha256:
            raise StaticCollectionError(f'request binding failed for {item.artifact_id}')

        body_bytes = store.read_artifact(body.sha256)
        if hashlib.sha256(body_bytes).hexdigest() != item.body_sha256 or len(body_bytes) != item.body_byte_count:
            raise StaticCollectionError(f'body replay failed for {item.artifact_id}')
        receipt_bytes = store.read_artifact(receipt.sha256)
        try:
            parsed_receipt = HttpsCaptureReceipt.model_validate_json(receipt_bytes)
        except ValueError as error:
            raise StaticCollectionError(f'receipt is invalid for {item.artifact_id}') from error
        if canonical_json_bytes(parsed_receipt) != receipt_bytes:
            raise StaticCollectionError(f'receipt is not canonical JSON for {item.artifact_id}')
        if (
            parsed_receipt.requested_url != artifact_spec.request.url
            or parsed_receipt.final_url != artifact_spec.request.url
            or parsed_receipt.request_headers != prepared_request_headers(artifact_spec.request)
            or parsed_receipt.status_code not in artifact_spec.request.allowed_status_codes
            or parsed_receipt.body_sha256 != item.body_sha256
            or parsed_receipt.body_byte_count != item.body_byte_count
            or parsed_receipt.body_byte_count > artifact_spec.request.max_body_bytes
            or parsed_receipt.started_at != item.started_at
            or parsed_receipt.completed_at != item.completed_at
            or parsed_receipt.started_at < attempt.started_at
            or (parsed_receipt.completed_at - parsed_receipt.started_at).total_seconds()
            > configuration.request_deadline_seconds
        ):
            raise StaticCollectionError(f'receipt replay failed for {item.artifact_id}')
        total_body_bytes += parsed_receipt.body_byte_count
        if total_body_bytes > configuration.max_total_body_bytes:
            raise StaticCollectionError('successful static run exceeds the immutable aggregate body-byte budget')
    return manifest


def verify_static_attempt_policy(
    store: OperationalStore,
    run: LogicalRunRecord,
) -> StaticHttpsJobConfiguration:
    """Replay lease, retry, plan-binding, and no-renewal policy for any static run state."""

    job = store.get_job(run.job_spec_sha256)
    if job.spec.collector_id != STATIC_HTTPS_COLLECTOR_ID:
        raise StaticCollectionError('logical run is not owned by the static HTTPS collector')
    configuration = validate_static_job_configuration(job.spec.configuration)
    _verify_static_attempt_policy(store, run, configuration)
    return configuration


def verify_all_static_run_manifests(store: OperationalStore) -> tuple[StaticHttpsRunManifest, ...]:
    """Replay every successful run, failing if its collector has no verifier.

    Static job configurations are validated even before they succeed so an operational
    checkpoint cannot silently include a static revision with uncommitted lease/retry policy.
    Failed and abandoned attempt payloads remain within structural-store verification only.
    """

    static_configurations: dict[str, StaticHttpsJobConfiguration] = {}
    for job in store.list_jobs():
        if job.spec.collector_id == STATIC_HTTPS_COLLECTOR_ID:
            static_configurations[job.spec_sha256] = validate_static_job_configuration(job.spec.configuration)
    manifests: list[StaticHttpsRunManifest] = []
    for run in store.list_logical_runs():
        configuration = static_configurations.get(run.job_spec_sha256)
        if configuration is not None:
            verify_static_attempt_policy(store, run)
        if run.successful_attempt_id is None:
            continue
        job = store.get_job(run.job_spec_sha256)
        if job.spec.collector_id != STATIC_HTTPS_COLLECTOR_ID:
            raise StaticCollectionError(
                f'no semantic verifier is registered for successful collector {job.spec.collector_id!r}'
            )
        manifests.append(load_static_run_manifest(store, run.successful_attempt_id))
    return tuple(manifests)
