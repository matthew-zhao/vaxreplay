"""Fail-closed composition for one reserved prospective benchmark execution.

This module joins the already implemented artifact contracts.  It does not provide
an official isolation backend, registry, timestamp authority, or deployment.  The
caller must supply those production trust boundaries explicitly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.prospective_release_approval_identity import (
    TierAProspectiveReleaseApprovalReplay,
)
from vaxreplay.prospective_admission import CaseUniverseSealVerifier, SourceCaptureVerifier
from vaxreplay.prospective_release import (
    LoadedProspectiveCohortRelease,
    load_prospective_cohort_release,
)
from vaxreplay.prospective_schema import ProspectiveAttemptPolicy
from vaxreplay.runner.backend import IsolationBackend, PreparedBackend, RawExecutionResult
from vaxreplay.runner.orchestrator import (
    LoadedRunArtifact,
    receipt_key_id,
    run_challenge_bundle,
)
from vaxreplay.runner.prospective_attempt_reservation import (
    LoadedProspectiveAttemptReservation,
    LoadedProspectiveAttemptStartAuthorization,
    ProspectiveAttemptCompletionTarget,
    ProspectiveAttemptRegistryVerifier,
    ProspectiveAttemptStartProof,
    ProspectiveAttemptStartTarget,
    ProspectiveAttemptStartVerifier,
    ProspectiveExplicitFailure,
    build_prospective_attempt_completion_target,
    load_prospective_attempt_reservation,
    load_prospective_attempt_start_authorization,
    prospective_attempt_completion_target_sha256,
    prospective_attempt_start_target_sha256,
)
from vaxreplay.runner.prospective_release_seal import (
    LoadedProspectiveReleaseSeal,
    ProspectiveReleaseTimestampVerifier,
    load_prospective_release_seal,
)
from vaxreplay.runner.schema import RunnerPolicy, SystemSubmissionManifest
from vaxreplay.temporal_schema import TemporalReceiptVerifier


class ProspectiveAttemptExecutionError(ValueError):
    """Raised before a run is admitted or before its terminal target is returned."""


class ProspectiveAttemptStartConsumer(Protocol):
    """Trusted stateful first-write-wins start-consumption boundary.

    Unlike ``ProspectiveAttemptStartVerifier``, which must be safe to call repeatedly,
    this operation mutates durable authority state.  Implementations must atomically
    return ``True`` only for the first consumption of ``target.attempt_key_sha256``
    and return ``False`` for every replay, including a replay using a different start
    authorization for the same attempt.
    """

    def consume_start(
        self,
        *,
        start_authorization_manifest_sha256: str,
        target: ProspectiveAttemptStartTarget,
        target_bytes: bytes,
        proof: ProspectiveAttemptStartProof,
        proof_bytes: bytes,
    ) -> bool: ...


class ProspectiveAttemptExecutionFailureRecord(StrictModel):
    """Canonical retained evidence for an exception after start consumption."""

    schema_version: Literal['vaxreplay.prospective-execution-failure.v0.1'] = (
        'vaxreplay.prospective-execution-failure.v0.1'
    )
    retry_allowed: Literal[False] = False
    prospective_release_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    canonical_cohort_id: str = Field(min_length=1)
    registered_entry_id: str = Field(min_length=1)
    attempt_key_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reservation_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    start_authorization_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    start_authorization_target_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    start_authorization_proof_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    failure_code: Literal['runner_or_backend_exception'] = 'runner_or_backend_exception'
    backend_id: str = Field(min_length=1, max_length=1024)
    started_at: datetime
    failed_at: datetime
    exception_type: str = Field(min_length=1, max_length=1024)
    exception_message: str = Field(max_length=16_384)

    @field_validator('started_at', 'failed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('execution failure timestamps must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_window(self) -> ProspectiveAttemptExecutionFailureRecord:
        if self.failed_at < self.started_at:
            raise ValueError('execution failure cannot predate its consumed start')
        return self


class ProspectiveConsumedStartFatalError(ProspectiveAttemptExecutionError):
    """Non-retryable consumed/indeterminate start requiring orphan reconciliation."""

    retry_allowed: Literal[False] = False

    def __init__(
        self,
        message: str,
        *,
        reservation: LoadedProspectiveAttemptReservation,
        start_authorization: LoadedProspectiveAttemptStartAuthorization,
        failure_record_bytes: bytes | None = None,
        run_receipt_sha256: str | None = None,
    ) -> None:
        super().__init__(message)
        target = reservation.target
        self.prospective_release_sha256 = target.prospective_release_sha256
        self.canonical_cohort_id = target.canonical_cohort_id
        self.registered_entry_id = target.registered_entry_id
        self.attempt_key_sha256 = target.attempt_key_sha256
        self.alias_key_sha256 = target.alias_key_sha256
        self.reservation_manifest_sha256 = reservation.manifest_sha256
        self.start_authorization_manifest_sha256 = start_authorization.manifest_sha256
        self.start_authorization_target_sha256 = prospective_attempt_start_target_sha256(start_authorization.target)
        self.start_authorization_proof_sha256 = start_authorization.manifest.start_proof.proof_sha256
        self.failure_record_bytes = failure_record_bytes
        self.run_receipt_sha256 = run_receipt_sha256


@dataclass(frozen=True)
class ProspectiveAttemptExecutionHandoff:
    """Authenticated success material ready for an external completion registry.

    ``completion_target_bytes`` are the exact bytes the registry must witness.  Once
    a trusted registry returns its proof, the caller passes this handoff's loaded
    inputs and ``run`` to ``build_prospective_attempt_completion``.  This dataclass
    is a convenience value, not an authority assertion or a deployed Tier A claim.
    """

    release: LoadedProspectiveCohortRelease
    release_seal: LoadedProspectiveReleaseSeal
    reservation: LoadedProspectiveAttemptReservation
    start_authorization: LoadedProspectiveAttemptStartAuthorization
    run: LoadedRunArtifact
    completion_target: ProspectiveAttemptCompletionTarget
    completion_target_bytes: bytes
    completion_target_sha256: str


@dataclass(frozen=True)
class ProspectiveAttemptFailureHandoff:
    """Registry-ready retained failure for a start that cannot be retried."""

    release: LoadedProspectiveCohortRelease
    release_seal: LoadedProspectiveReleaseSeal
    reservation: LoadedProspectiveAttemptReservation
    start_authorization: LoadedProspectiveAttemptStartAuthorization
    failure: ProspectiveExplicitFailure
    failure_record: ProspectiveAttemptExecutionFailureRecord
    failure_record_bytes: bytes
    completion_target: ProspectiveAttemptCompletionTarget
    completion_target_bytes: bytes
    completion_target_sha256: str
    retry_allowed: Literal[False] = False


@dataclass
class _BackendRecorder:
    backend: IsolationBackend
    prepared_backend_id: str | None = None

    @property
    def backend_id(self) -> str:
        if self.prepared_backend_id is not None:
            return self.prepared_backend_id
        backend_type = type(self.backend)
        return f'{backend_type.__module__}.{backend_type.__qualname__}'

    def prepare(self, system: SystemSubmissionManifest, policy: RunnerPolicy) -> PreparedBackend:
        prepared = self.backend.prepare(system, policy)
        self.prepared_backend_id = prepared.capabilities.backend_id
        return prepared

    def run(
        self,
        *,
        input_bytes: bytes,
        system: SystemSubmissionManifest,
        policy: RunnerPolicy,
        prepared: PreparedBackend,
    ) -> RawExecutionResult:
        return self.backend.run(
            input_bytes=input_bytes,
            system=system,
            policy=policy,
            prepared=prepared,
        )


def run_reserved_prospective_attempt(
    *,
    release_root: Path,
    expected_release_sha256: str,
    release_seal_root: Path,
    expected_release_seal_manifest_sha256: str,
    expected_approval_report_sha256: str,
    submissions_open_at: datetime,
    reservation_root: Path,
    expected_reservation_manifest_sha256: str,
    start_authorization_root: Path,
    expected_start_authorization_manifest_sha256: str,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    output_dir: Path,
    backend: IsolationBackend,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    start_consumer: ProspectiveAttemptStartConsumer,
) -> ProspectiveAttemptExecutionHandoff | ProspectiveAttemptFailureHandoff:
    """Reauthenticate one reservation, execute it once, and prepare its success target.

    All five expected digests are out-of-band pins.  All authority decisions remain
    caller-supplied verifier callbacks; this function never substitutes an in-band
    ``verified`` flag.  ``backend`` is an explicit isolation implementation and its
    reported capabilities are checked by ``run_challenge_bundle`` against the exact
    reserved policy.

    ``start_verifier`` is an idempotent proof check.  ``start_consumer`` is the
    separate trusted stateful boundary that atomically redeems the exact attempt only
    after every static identity check and immediately before backend work.  Once it
    succeeds, this function never reports an exception as retryable: ordinary runner
    failures return a registry-ready ``ProspectiveAttemptFailureHandoff`` and failures
    that cannot be represented by a valid completion target raise the typed
    ``ProspectiveConsumedStartFatalError`` for external orphan reconciliation.
    """

    _require_sha256(expected_release_sha256, 'expected_release_sha256')
    _require_sha256(
        expected_release_seal_manifest_sha256,
        'expected_release_seal_manifest_sha256',
    )
    _require_sha256(expected_approval_report_sha256, 'expected_approval_report_sha256')
    _require_sha256(
        expected_reservation_manifest_sha256,
        'expected_reservation_manifest_sha256',
    )
    _require_sha256(
        expected_start_authorization_manifest_sha256,
        'expected_start_authorization_manifest_sha256',
    )
    if backend is None:  # type: ignore[comparison-overlap]
        raise ProspectiveAttemptExecutionError('an explicit isolation backend is required')
    if start_consumer is None:  # type: ignore[comparison-overlap]
        raise ProspectiveAttemptExecutionError('a trusted stateful attempt-start consumer is required')

    try:
        release = load_prospective_cohort_release(
            release_root,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_release_sha256=expected_release_sha256,
        )
        release_seal = load_prospective_release_seal(
            release_seal_root,
            release=release,
            submissions_open_at=submissions_open_at,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            timestamp_verifier=release_timestamp_verifier,
        )
    except ValueError as error:
        raise ProspectiveAttemptExecutionError(f'prospective release context verification failed: {error}') from error
    if release_seal.manifest_sha256 != expected_release_seal_manifest_sha256:
        raise ProspectiveAttemptExecutionError(
            'prospective release seal does not match the expected out-of-band manifest identity'
        )

    try:
        reservation = load_prospective_attempt_reservation(
            reservation_root,
            release=release,
            release_seal=release_seal,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            release_timestamp_verifier=release_timestamp_verifier,
            registry_verifier=registry_verifier,
        )
    except ValueError as error:
        raise ProspectiveAttemptExecutionError(
            f'prospective attempt reservation verification failed: {error}'
        ) from error
    if reservation.manifest_sha256 != expected_reservation_manifest_sha256:
        raise ProspectiveAttemptExecutionError(
            'prospective attempt reservation does not match the expected out-of-band manifest identity'
        )

    try:
        start_authorization = load_prospective_attempt_start_authorization(
            start_authorization_root,
            release=release,
            release_seal=release_seal,
            reservation=reservation,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            release_timestamp_verifier=release_timestamp_verifier,
            registry_verifier=registry_verifier,
            start_verifier=start_verifier,
            expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
        )
    except ValueError as error:
        raise ProspectiveAttemptExecutionError(
            f'prospective attempt start-authorization verification failed: {error}'
        ) from error
    if start_authorization.manifest_sha256 != expected_start_authorization_manifest_sha256:
        raise ProspectiveAttemptExecutionError(
            'attempt start authorization does not match its expected out-of-band manifest identity'
        )

    _preflight_runner_identities(
        release=release,
        release_seal=release_seal,
        system=system,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
    )
    recorded_backend = _BackendRecorder(backend)
    target_bytes = canonical_json_bytes(start_authorization.target)
    try:
        consumed = start_consumer.consume_start(
            start_authorization_manifest_sha256=start_authorization.manifest_sha256,
            target=start_authorization.target,
            target_bytes=target_bytes,
            proof=start_authorization.manifest.start_proof,
            proof_bytes=start_authorization.proof_bytes,
        )
    except Exception as error:
        raise ProspectiveConsumedStartFatalError(
            f'attempt-start consumption outcome is indeterminate; do not retry: {error}',
            reservation=reservation,
            start_authorization=start_authorization,
        ) from error
    if consumed is not True:
        raise ProspectiveConsumedStartFatalError(
            'attempt start was already consumed or the stateful authority rejected it; do not retry',
            reservation=reservation,
            start_authorization=start_authorization,
        )
    try:
        execution_started_at = max(
            datetime.now(UTC),
            start_authorization.manifest.start_proof.witnessed_at,
        )
    except Exception as error:
        raise ProspectiveConsumedStartFatalError(
            f'attempt start was consumed but its launch time could not be recorded; do not retry: {error}',
            reservation=reservation,
            start_authorization=start_authorization,
        ) from error

    # Only non-retryable consumed-start timestamp recording occurs between the
    # durable transition above and the first backend operation below. No identity
    # verification or local preparation may enter this interval.
    try:
        run = run_challenge_bundle(
            release.challenge,
            expected_challenge_sha256=release_seal.target.challenge_bundle_sha256,
            system=system,
            policy=runner_policy,
            receipt_key=receipt_key,
            expected_receipt_key_id=expected_receipt_key_id,
            output_dir=output_dir,
            backend=recorded_backend,
        )
    except Exception as error:
        try:
            return _failure_handoff(
                error=error,
                execution_started_at=execution_started_at,
                backend_id=recorded_backend.backend_id,
                release=release,
                release_seal=release_seal,
                reservation=reservation,
                start_authorization=start_authorization,
                system=system,
                runner_policy=runner_policy,
                attempt_policy=attempt_policy,
                receipt_key=receipt_key,
                expected_receipt_key_id=expected_receipt_key_id,
                decision_receipt_verifier=decision_receipt_verifier,
                case_universe_seal_verifier=case_universe_seal_verifier,
                source_capture_verifier=source_capture_verifier,
                expected_approval_report_sha256=expected_approval_report_sha256,
                approval_replay=approval_replay,
                release_timestamp_verifier=release_timestamp_verifier,
                registry_verifier=registry_verifier,
                start_verifier=start_verifier,
                expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
            )
        except ProspectiveConsumedStartFatalError:
            raise
        except Exception as handoff_error:
            raise ProspectiveConsumedStartFatalError(
                f'consumed attempt failed and failure handoff construction also failed; do not retry: {handoff_error}',
                reservation=reservation,
                start_authorization=start_authorization,
            ) from handoff_error

    try:
        completion_target = build_prospective_attempt_completion_target(
            release=release,
            release_seal=release_seal,
            reservation=reservation,
            start_authorization=start_authorization,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            receipt_key=receipt_key,
            expected_receipt_key_id=expected_receipt_key_id,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            release_timestamp_verifier=release_timestamp_verifier,
            registry_verifier=registry_verifier,
            start_verifier=start_verifier,
            expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
            run=run,
        )
    except Exception as error:
        raise ProspectiveConsumedStartFatalError(
            f'reserved run failed terminal-target reverification; retain it and do not retry: {error}',
            reservation=reservation,
            start_authorization=start_authorization,
            run_receipt_sha256=run.receipt_sha256,
        ) from error
    completion_target_bytes = canonical_json_bytes(completion_target)
    return ProspectiveAttemptExecutionHandoff(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        start_authorization=start_authorization,
        run=run,
        completion_target=completion_target,
        completion_target_bytes=completion_target_bytes,
        completion_target_sha256=prospective_attempt_completion_target_sha256(completion_target),
    )


def _preflight_runner_identities(
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    system: SystemSubmissionManifest,
    receipt_key: bytes,
    expected_receipt_key_id: str,
) -> None:
    if release.challenge.manifest_sha256 != release_seal.target.challenge_bundle_sha256:
        raise ProspectiveAttemptExecutionError('release challenge does not match its sealed out-of-band identity')
    if {envelope.response_protocol for envelope in release.challenge.envelopes} != {system.response_protocol}:
        raise ProspectiveAttemptExecutionError(
            'system response protocol does not match every sealed challenge envelope'
        )
    if receipt_key_id(receipt_key) != expected_receipt_key_id:
        raise ProspectiveAttemptExecutionError('receipt key does not match the preregistered organizer key ID')


def _failure_handoff(
    *,
    error: Exception,
    execution_started_at: datetime,
    backend_id: str,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    expected_start_authorization_manifest_sha256: str,
) -> ProspectiveAttemptFailureHandoff:
    failed_at = max(datetime.now(UTC), execution_started_at)
    exception_type = f'{type(error).__module__}.{type(error).__qualname__}'
    failure_record = ProspectiveAttemptExecutionFailureRecord(
        prospective_release_sha256=reservation.target.prospective_release_sha256,
        canonical_cohort_id=reservation.target.canonical_cohort_id,
        registered_entry_id=reservation.target.registered_entry_id,
        attempt_key_sha256=reservation.target.attempt_key_sha256,
        reservation_manifest_sha256=reservation.manifest_sha256,
        start_authorization_manifest_sha256=start_authorization.manifest_sha256,
        start_authorization_target_sha256=prospective_attempt_start_target_sha256(start_authorization.target),
        start_authorization_proof_sha256=(start_authorization.manifest.start_proof.proof_sha256),
        backend_id=backend_id[:1024],
        started_at=execution_started_at,
        failed_at=failed_at,
        exception_type=exception_type[:1024],
        exception_message=str(error)[:16_384],
    )
    failure_record_bytes = canonical_json_bytes(failure_record)
    failure = ProspectiveExplicitFailure(
        failure_code=failure_record.failure_code,
        backend_id=failure_record.backend_id,
        started_at=failure_record.started_at,
        failed_at=failure_record.failed_at,
        failure_record_sha256=hashlib.sha256(failure_record_bytes).hexdigest(),
        failure_record_bytes=len(failure_record_bytes),
    )
    try:
        completion_target = build_prospective_attempt_completion_target(
            release=release,
            release_seal=release_seal,
            reservation=reservation,
            start_authorization=start_authorization,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            receipt_key=receipt_key,
            expected_receipt_key_id=expected_receipt_key_id,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            release_timestamp_verifier=release_timestamp_verifier,
            registry_verifier=registry_verifier,
            start_verifier=start_verifier,
            expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
            failure=failure,
            failure_record=failure_record_bytes,
        )
    except Exception as target_error:
        raise ProspectiveConsumedStartFatalError(
            'consumed attempt failed and its terminal target could not be built; '
            f'retain the failure record and do not retry: {target_error}',
            reservation=reservation,
            start_authorization=start_authorization,
            failure_record_bytes=failure_record_bytes,
        ) from target_error
    completion_target_bytes = canonical_json_bytes(completion_target)
    return ProspectiveAttemptFailureHandoff(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        start_authorization=start_authorization,
        failure=failure,
        failure_record=failure_record,
        failure_record_bytes=failure_record_bytes,
        completion_target=completion_target,
        completion_target_bytes=completion_target_bytes,
        completion_target_sha256=prospective_attempt_completion_target_sha256(completion_target),
    )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ProspectiveAttemptExecutionError(f'{name} must be an exact lowercase SHA-256 digest')
