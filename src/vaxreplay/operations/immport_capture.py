"""Secret-free operational contract for authenticated ImmPort captures.

The network-facing producer is intentionally a separate trusted deployment: it obtains a
scoped API key from a runtime secret broker, sends ``Authorization: Bearer ...`` only on the
wire, scans arbitrary response/error bytes for the live credential before zeroization, and
hands this module response bodies plus canonical :class:`ImmportSanitizedCaptureReceipt`
bytes.  This module validates the complete capture before putting any receipt into the
operational CAS.  Its receipt/plan schemas have no field capable of representing an
Authorization header, cookie, credential identifier, or presigned URL.  Arbitrary body bytes
cannot provide that same structural guarantee, so body/error sanitation remains in the
network producer's trusted computing base.

This is not a network client and never accepts a credential.  The producer deployment remains
responsible for secret-broker access control, egress restriction, and zeroization.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.policy import (
    IMMPORT_AUTHENTICATED_COLLECTOR_ID,
    ImmportAuthenticatedJobConfiguration,
    parse_immport_authenticated_job_configuration,
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

IMMPORT_AUTHENTICATED_PLAN_SCHEMA_VERSION = 'vaxreplay.immport-authenticated-plan.v0.1'
IMMPORT_AUTHENTICATED_RUN_SCHEMA_VERSION = 'vaxreplay.immport-authenticated-run.v0.1'

_OPENAPI_URL = 'https://www.immport.org/data/query/v3/api-docs'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
# These are intentionally much smaller than promotion's per-file/aggregate limits.  A
# hermetic promotion additionally archives base64 callback requests/responses (including two
# deterministic adapter passes), so admitting captures near the outer 512 MiB/2 GiB limits
# would create successful operational runs that cannot be promoted.
MAX_IMMPORT_ARTIFACT_BODY_BYTES = 32 * 1024 * 1024
MAX_IMMPORT_CAPTURE_BODY_BYTES = 64 * 1024 * 1024
MAX_IMMPORT_PLAN_BYTES = 128 * 1024
MAX_IMMPORT_RUN_MANIFEST_BYTES = 128 * 1024
MAX_IMMPORT_RECEIPT_BYTES = 64 * 1024


class ImmportAuthenticatedCaptureError(RuntimeError):
    """An authenticated ImmPort capture violates its immutable public contract."""


class ImmportAuthenticatedArtifactSpec(StrictModel):
    """One exact credential-free request committed before collection."""

    artifact_id: str = Field(pattern=ARTIFACT_ROLE_PATTERN, max_length=110)
    requested_url: str = Field(min_length=1, max_length=8192)
    authentication: Literal['none', 'immport_scoped_api_key_bearer_redacted']
    max_body_bytes: int = Field(ge=1, le=MAX_IMMPORT_ARTIFACT_BODY_BYTES)
    timeout_seconds: int = Field(ge=1, le=15 * 60)

    @model_validator(mode='after')
    def validate_request(self) -> Self:
        _validate_plan_url(self.requested_url)
        expected = 'none' if self.requested_url == _OPENAPI_URL else 'immport_scoped_api_key_bearer_redacted'
        if self.authentication != expected:
            raise ValueError('ImmPort plan authentication differs from its endpoint')
        return self


class ImmportAuthenticatedCollectionPlan(StrictModel):
    """Exact serial, release-bracketed request inventory for one ImmPort study."""

    schema_version: Literal['vaxreplay.immport-authenticated-plan.v0.1'] = IMMPORT_AUTHENTICATED_PLAN_SCHEMA_VERSION
    plan_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    study_accession: str = Field(pattern=r'^SDY[0-9]{1,12}$')
    panel_deadline_seconds: int = Field(ge=9, le=60 * 60)
    artifacts: tuple[ImmportAuthenticatedArtifactSpec, ...] = Field(min_length=9, max_length=9)

    @field_validator('artifacts')
    @classmethod
    def validate_artifacts(
        cls,
        value: tuple[ImmportAuthenticatedArtifactSpec, ...],
        info: Any,
    ) -> tuple[ImmportAuthenticatedArtifactSpec, ...]:
        identifiers = tuple(item.artifact_id for item in value)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError('ImmPort plan artifact IDs must be unique and encode serial order')
        accession = info.data.get('study_accession')
        if not isinstance(accession, str):
            raise ValueError('ImmPort plan requires its study accession before artifacts')
        expected_urls = _expected_study_urls(accession)
        if tuple(item.requested_url for item in value) != expected_urls:
            raise ValueError('ImmPort plan must use the exact nine-request release-bracketed profile')
        if sum(item.max_body_bytes for item in value) > MAX_IMMPORT_CAPTURE_BODY_BYTES:
            raise ValueError('ImmPort plan aggregate body bound exceeds the portable promotion budget')
        return value


def immport_authenticated_plan_sha256(plan: ImmportAuthenticatedCollectionPlan) -> str:
    return hashlib.sha256(canonical_json_bytes(plan)).hexdigest()


class ImmportCollectedArtifact(StrictModel):
    artifact_id: str = Field(pattern=ARTIFACT_ROLE_PATTERN, max_length=110)
    body_sha256: str = Field(pattern=_SHA256_PATTERN)
    body_byte_count: int = Field(ge=0)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    receipt_byte_count: int = Field(gt=0)
    started_at: datetime
    completed_at: datetime

    @field_validator('started_at', 'completed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ImmPort collection timestamp')

    @model_validator(mode='after')
    def validate_times(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError('ImmPort artifact completion cannot precede its start')
        return self


class ImmportAuthenticatedRunManifest(StrictModel):
    """Successful secret-free terminal manifest for one precommitted plan."""

    schema_version: Literal['vaxreplay.immport-authenticated-run.v0.1'] = IMMPORT_AUTHENTICATED_RUN_SCHEMA_VERSION
    logical_run_id: str = Field(pattern=r'^run-[0-9a-f]{64}$')
    attempt_id: str = Field(pattern=r'^attempt-[0-9a-f]{32}$')
    job_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    scheduled_for: datetime
    plan_id: str = Field(min_length=1, max_length=200)
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_id: str = Field(min_length=1, max_length=200)
    attempt_started_at: datetime
    completed_at: datetime
    collector_id: Literal['immport-secret-broker-collector'] = IMMPORT_AUTHENTICATED_COLLECTOR_ID
    collector_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifacts: tuple[ImmportCollectedArtifact, ...] = Field(min_length=9, max_length=9)
    plan_complete: Literal[True] = True
    source_enumeration_complete: Literal[False] = False
    external_timestamp_required: Literal[True] = True
    tier_a_eligible: Literal[False] = False

    @field_validator('scheduled_for', 'attempt_started_at', 'completed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ImmPort run timestamp')

    @model_validator(mode='after')
    def validate_manifest(self) -> Self:
        identifiers = tuple(item.artifact_id for item in self.artifacts)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError('ImmPort manifest artifacts must use canonical unique order')
        if self.completed_at != max(item.completed_at for item in self.artifacts):
            raise ValueError('ImmPort run completion must equal the latest receipt completion')
        if self.completed_at < self.attempt_started_at:
            raise ValueError('ImmPort run completion cannot precede its attempt')
        return self


@dataclass(frozen=True)
class ImmportCapturedExchange:
    """Unpersisted producer output crossing the secret-free trust boundary."""

    artifact_id: str
    body: bytes
    receipt: bytes


@dataclass(frozen=True)
class ImmportAuthenticatedCaptureResult:
    attempt: AttemptLease
    manifest: ImmportAuthenticatedRunManifest
    manifest_artifact: StoredArtifact


def record_immport_authenticated_capture(
    store: OperationalStore,
    logical_run_id: str,
    plan: ImmportAuthenticatedCollectionPlan,
    *,
    owner_id: str,
    producer: Callable[
        [ImmportAuthenticatedCollectionPlan, AttemptLease],
        tuple[ImmportCapturedExchange, ...],
    ],
    clock: Callable[[], datetime] | None = None,
) -> ImmportAuthenticatedCaptureResult:
    """Lease, produce, validate, and durably bind one credential-free capture.

    The producer callback runs only after the immutable attempt lease and collection plan are
    durable.  It returns only after dropping the bearer value.  All exchanges are then parsed
    and exact-matched before the first receipt/body is persisted, closing the path by which an
    Authorization header, cookie, or presigned URL could enter a promotion archive.
    """

    now = clock or (lambda: datetime.now(timezone.utc))
    run = store.get_logical_run(logical_run_id)
    job = store.get_job(run.job_spec_sha256)
    if job.spec.collector_id != IMMPORT_AUTHENTICATED_COLLECTOR_ID:
        raise ImmportAuthenticatedCaptureError('job is not owned by the authenticated ImmPort collector')
    configuration = _configuration(job.spec.configuration)
    plan_sha256 = immport_authenticated_plan_sha256(plan)
    if configuration.collection_plan_sha256 != plan_sha256:
        raise ImmportAuthenticatedCaptureError('registered job does not bind the supplied ImmPort plan')
    if configuration.source_id != plan.source_id:
        raise ImmportAuthenticatedCaptureError('registered job source differs from the ImmPort plan')
    if configuration.lease_seconds < plan.panel_deadline_seconds:
        raise ImmportAuthenticatedCaptureError('ImmPort lease is shorter than the precommitted panel deadline')

    plan_bytes = canonical_json_bytes(plan)
    if len(plan_bytes) > MAX_IMMPORT_PLAN_BYTES:
        raise ImmportAuthenticatedCaptureError('ImmPort collection plan exceeds its byte bound')
    plan_artifact = store.put_bytes(
        plan_bytes,
        recorded_at=_operation_time(now, 'plan recorded_at'),
        max_bytes=MAX_IMMPORT_PLAN_BYTES,
    )
    attempt = store.begin_attempt(
        logical_run_id,
        owner_id=owner_id,
        now=_operation_time(now, 'attempt started_at'),
        lease_seconds=configuration.lease_seconds,
        max_attempts=configuration.max_attempts_per_slot,
        initial_artifacts={'collection-plan': plan_artifact.sha256},
    )
    try:
        producer_failed = False
        exchanges: tuple[ImmportCapturedExchange, ...] | object = ()
        try:
            exchanges = producer(plan, attempt)
        except Exception:
            producer_failed = True
        if producer_failed:
            _fail_unfinished_attempt(store, attempt, owner_id=owner_id, clock=now)
            raise ImmportAuthenticatedCaptureError(
                'authenticated ImmPort producer failed before returning a sanitized capture'
            )
        if type(exchanges) is not tuple:
            raise ImmportAuthenticatedCaptureError(
                'ImmPort producer must return an exact tuple of secret-free exchanges'
            )
        parsed = _validate_exchange_set(plan, exchanges, configuration)
        validated_at = _operation_time(now, 'ImmPort capture validated_at')
        if any(
            receipt.started_at < attempt.started_at
            or receipt.completed_at > validated_at
            or receipt.completed_at >= attempt.lease_expires_at
            for receipt in parsed
        ):
            raise ImmportAuthenticatedCaptureError(
                'ImmPort exchanges lie outside the honest attempt/validation/lease interval'
            )
        collected: list[ImmportCollectedArtifact] = []
        for spec, exchange, receipt in zip(plan.artifacts, exchanges, parsed, strict=True):
            recorded_at = _operation_time(now, 'ImmPort artifact recorded_at')
            if recorded_at >= attempt.lease_expires_at:
                raise ImmportAuthenticatedCaptureError('ImmPort exchange lies outside the immutable attempt lease')
            body = store.put_bytes(
                exchange.body,
                recorded_at=recorded_at,
                max_bytes=spec.max_body_bytes,
            )
            receipt_artifact = store.put_bytes(
                exchange.receipt,
                recorded_at=recorded_at,
                max_bytes=MAX_IMMPORT_RECEIPT_BYTES,
            )
            store.attach_artifact(
                attempt.attempt_id,
                owner_id=owner_id,
                role=f'body.{spec.artifact_id}',
                artifact_sha256=body.sha256,
                now=_operation_time(now, 'ImmPort body attached_at'),
            )
            store.attach_artifact(
                attempt.attempt_id,
                owner_id=owner_id,
                role=f'receipt.{spec.artifact_id}',
                artifact_sha256=receipt_artifact.sha256,
                now=_operation_time(now, 'ImmPort receipt attached_at'),
            )
            collected.append(
                ImmportCollectedArtifact(
                    artifact_id=spec.artifact_id,
                    body_sha256=body.sha256,
                    body_byte_count=body.byte_count,
                    receipt_sha256=receipt_artifact.sha256,
                    receipt_byte_count=receipt_artifact.byte_count,
                    started_at=receipt.started_at,
                    completed_at=receipt.completed_at,
                )
            )
        manifest = ImmportAuthenticatedRunManifest(
            logical_run_id=run.logical_run_id,
            attempt_id=attempt.attempt_id,
            job_spec_sha256=run.job_spec_sha256,
            scheduled_for=run.scheduled_for,
            plan_id=plan.plan_id,
            plan_sha256=plan_sha256,
            source_id=plan.source_id,
            attempt_started_at=attempt.started_at,
            completed_at=max(item.completed_at for item in collected),
            collector_implementation_sha256=configuration.collector_implementation_sha256,
            collector_execution_environment_sha256=(configuration.collector_execution_environment_sha256),
            artifacts=tuple(collected),
        )
        manifest_bytes = canonical_json_bytes(manifest)
        if len(manifest_bytes) > MAX_IMMPORT_RUN_MANIFEST_BYTES:
            raise ImmportAuthenticatedCaptureError('ImmPort run manifest exceeds its byte bound')
        manifest_artifact = store.put_bytes(
            manifest_bytes,
            recorded_at=_operation_time(now, 'ImmPort manifest recorded_at'),
            max_bytes=MAX_IMMPORT_RUN_MANIFEST_BYTES,
        )
        terminal = store.succeed_attempt(
            attempt.attempt_id,
            owner_id=owner_id,
            run_manifest_sha256=manifest_artifact.sha256,
            now=_operation_time(now, 'ImmPort attempt finished_at'),
        )
        return ImmportAuthenticatedCaptureResult(terminal, manifest, manifest_artifact)
    except Exception:
        _fail_unfinished_attempt(store, attempt, owner_id=owner_id, clock=now)
        raise


def load_immport_authenticated_run_manifest(
    store: OperationalStore,
    attempt_id: str,
) -> ImmportAuthenticatedRunManifest:
    """Replay one successful authenticated capture from every durable object."""

    attempt = store.get_attempt(attempt_id)
    if attempt.state is not AttemptState.SUCCEEDED or attempt.finished_at is None:
        raise ImmportAuthenticatedCaptureError('ImmPort run must be a successful terminal attempt')
    run = store.get_logical_run(attempt.logical_run_id)
    if run.successful_attempt_id != attempt_id:
        raise ImmportAuthenticatedCaptureError('ImmPort logical run does not identify this attempt')
    job = store.get_job(run.job_spec_sha256)
    if job.spec.collector_id != IMMPORT_AUTHENTICATED_COLLECTOR_ID:
        raise ImmportAuthenticatedCaptureError('attempt is not owned by the authenticated ImmPort collector')
    configuration = _configuration(job.spec.configuration)
    _verify_attempt_policy(store, run, configuration)

    attachments = store.list_attempt_artifacts(attempt_id)
    plan_artifact = attachments.get('collection-plan')
    manifest_artifact = attachments.get('run-manifest')
    if plan_artifact is None or manifest_artifact is None:
        raise ImmportAuthenticatedCaptureError('ImmPort attempt omits its plan or run manifest')
    if (
        plan_artifact.byte_count > MAX_IMMPORT_PLAN_BYTES
        or manifest_artifact.byte_count > MAX_IMMPORT_RUN_MANIFEST_BYTES
    ):
        raise ImmportAuthenticatedCaptureError('ImmPort plan or manifest exceeds its byte bound')
    plan_bytes = store.read_artifact(
        plan_artifact.sha256,
        max_bytes=MAX_IMMPORT_PLAN_BYTES,
    )
    manifest_bytes = store.read_artifact(
        manifest_artifact.sha256,
        max_bytes=MAX_IMMPORT_RUN_MANIFEST_BYTES,
    )
    plan = _canonical_model(plan_bytes, ImmportAuthenticatedCollectionPlan, 'collection plan')
    manifest = _canonical_model(manifest_bytes, ImmportAuthenticatedRunManifest, 'run manifest')
    plan_sha256 = immport_authenticated_plan_sha256(plan)
    if (
        plan_artifact.sha256 != plan_sha256
        or configuration.collection_plan_sha256 != plan_sha256
        or configuration.source_id != plan.source_id
        or manifest.plan_sha256 != plan_sha256
        or manifest.plan_id != plan.plan_id
        or manifest.source_id != plan.source_id
        or manifest.logical_run_id != run.logical_run_id
        or manifest.attempt_id != attempt_id
        or manifest.job_spec_sha256 != run.job_spec_sha256
        or manifest.scheduled_for != run.scheduled_for
        or manifest.attempt_started_at != attempt.started_at
        or manifest.completed_at > attempt.finished_at
        or manifest.collector_implementation_sha256 != configuration.collector_implementation_sha256
        or manifest.collector_execution_environment_sha256 != configuration.collector_execution_environment_sha256
        or configuration.lease_seconds < plan.panel_deadline_seconds
    ):
        raise ImmportAuthenticatedCaptureError(
            'ImmPort manifest does not bind its job, attempt, plan, or reviewed collector'
        )

    expected_roles = {'collection-plan', 'run-manifest'}
    expected_roles.update(f'body.{item.artifact_id}' for item in plan.artifacts)
    expected_roles.update(f'receipt.{item.artifact_id}' for item in plan.artifacts)
    if set(attachments) != expected_roles:
        raise ImmportAuthenticatedCaptureError('ImmPort attachment set differs from its exact plan')
    if tuple(item.artifact_id for item in manifest.artifacts) != tuple(item.artifact_id for item in plan.artifacts):
        raise ImmportAuthenticatedCaptureError('ImmPort manifest artifact order differs from its plan')

    receipts = []
    for spec, item in zip(plan.artifacts, manifest.artifacts, strict=True):
        body = attachments[f'body.{spec.artifact_id}']
        receipt_artifact = attachments[f'receipt.{spec.artifact_id}']
        if receipt_artifact.byte_count > MAX_IMMPORT_RECEIPT_BYTES:
            raise ImmportAuthenticatedCaptureError('sanitized ImmPort receipt exceeds its byte bound')
        body_bytes = store.read_artifact(body.sha256, max_bytes=spec.max_body_bytes)
        receipt_bytes = store.read_artifact(
            receipt_artifact.sha256,
            max_bytes=MAX_IMMPORT_RECEIPT_BYTES,
        )
        receipt = _parse_receipt(receipt_bytes)
        if (
            (body.sha256, body.byte_count) != (item.body_sha256, item.body_byte_count)
            or hashlib.sha256(body_bytes).hexdigest() != item.body_sha256
            or len(body_bytes) != item.body_byte_count
            or (receipt_artifact.sha256, receipt_artifact.byte_count) != (item.receipt_sha256, item.receipt_byte_count)
            or receipt.requested_url != spec.requested_url
            or receipt.authentication != spec.authentication
            or receipt.body_sha256 != body.sha256
            or receipt.body_byte_count != body.byte_count
            or receipt.started_at != item.started_at
            or receipt.completed_at != item.completed_at
            or receipt.started_at < attempt.started_at
            or receipt.completed_at > attempt.finished_at
            or receipt.completed_at >= attempt.lease_expires_at
            or receipt.body_byte_count > spec.max_body_bytes
            or (receipt.completed_at - receipt.started_at).total_seconds() > spec.timeout_seconds
            or receipt.collector_id != IMMPORT_AUTHENTICATED_COLLECTOR_ID
            or receipt.collector_implementation_sha256 != configuration.collector_implementation_sha256
            or receipt.collector_execution_environment_sha256 != configuration.collector_execution_environment_sha256
        ):
            raise ImmportAuthenticatedCaptureError('ImmPort body/receipt replay failed')
        receipts.append(receipt)
    _require_serial_receipts(tuple(receipts))
    _require_panel_deadline(plan, tuple(receipts))
    return manifest


def verify_immport_attempt_policy(
    store: OperationalStore,
    run: LogicalRunRecord,
) -> ImmportAuthenticatedJobConfiguration:
    """Replay lease, retry, plan-binding, and no-renewal policy for any run state."""

    job = store.get_job(run.job_spec_sha256)
    if job.spec.collector_id != IMMPORT_AUTHENTICATED_COLLECTOR_ID:
        raise ImmportAuthenticatedCaptureError('logical run is not owned by the authenticated ImmPort collector')
    configuration = _configuration(job.spec.configuration)
    _verify_attempt_policy(store, run, configuration)
    return configuration


def _validate_exchange_set(
    plan: ImmportAuthenticatedCollectionPlan,
    exchanges: tuple[ImmportCapturedExchange, ...],
    configuration: ImmportAuthenticatedJobConfiguration,
) -> tuple[Any, ...]:
    if type(exchanges) is not tuple or any(type(item) is not ImmportCapturedExchange for item in exchanges):
        raise ImmportAuthenticatedCaptureError('ImmPort producer returned a malformed secret-free exchange inventory')
    if any(
        type(item.artifact_id) is not str or type(item.body) is not bytes or type(item.receipt) is not bytes
        for item in exchanges
    ):
        raise ImmportAuthenticatedCaptureError('ImmPort producer returned malformed secret-free exchange fields')
    if tuple(item.artifact_id for item in exchanges) != tuple(item.artifact_id for item in plan.artifacts):
        raise ImmportAuthenticatedCaptureError('producer exchange inventory differs from the exact ImmPort plan')
    if configuration.lease_seconds < plan.panel_deadline_seconds:
        raise ImmportAuthenticatedCaptureError('ImmPort lease is shorter than the precommitted panel deadline')
    if sum(len(item.body) for item in exchanges) > MAX_IMMPORT_CAPTURE_BODY_BYTES:
        raise ImmportAuthenticatedCaptureError('ImmPort capture exceeds the portable aggregate body budget')
    receipts = []
    for spec, exchange in zip(plan.artifacts, exchanges, strict=True):
        if len(exchange.body) > spec.max_body_bytes:
            raise ImmportAuthenticatedCaptureError('ImmPort producer body exceeds its precommitted byte bound')
        receipt = _parse_receipt(exchange.receipt)
        if (
            receipt.requested_url != spec.requested_url
            or receipt.authentication != spec.authentication
            or receipt.body_sha256 != hashlib.sha256(exchange.body).hexdigest()
            or receipt.body_byte_count != len(exchange.body)
            or receipt.collector_id != IMMPORT_AUTHENTICATED_COLLECTOR_ID
            or receipt.collector_implementation_sha256 != configuration.collector_implementation_sha256
            or receipt.collector_execution_environment_sha256 != configuration.collector_execution_environment_sha256
            or (receipt.completed_at - receipt.started_at).total_seconds() > spec.timeout_seconds
        ):
            raise ImmportAuthenticatedCaptureError(
                'producer receipt differs from its body, request, or reviewed collector'
            )
        receipts.append(receipt)
    _require_serial_receipts(tuple(receipts))
    _require_panel_deadline(plan, tuple(receipts))
    return tuple(receipts)


def _parse_receipt(payload: bytes) -> Any:
    # Lazy import avoids a module cycle: the scientific verifier imports promotion types.
    from vaxreplay.sources.immport import ImmportSanitizedCaptureReceipt

    if not isinstance(payload, bytes) or len(payload) > MAX_IMMPORT_RECEIPT_BYTES:
        raise ImmportAuthenticatedCaptureError('invalid sanitized ImmPort receipt') from None
    receipt = None
    try:
        receipt = ImmportSanitizedCaptureReceipt.model_validate_json(payload)
    except ValueError:
        pass
    if receipt is None:
        raise ImmportAuthenticatedCaptureError('invalid sanitized ImmPort receipt')
    if canonical_json_bytes(receipt) != payload:
        raise ImmportAuthenticatedCaptureError('sanitized ImmPort receipt is not canonical JSON')
    return receipt


def _require_serial_receipts(receipts: tuple[Any, ...]) -> None:
    if len(receipts) != 9 or any(
        left.completed_at > right.started_at for left, right in zip(receipts[:-1], receipts[1:], strict=True)
    ):
        raise ImmportAuthenticatedCaptureError('ImmPort receipts do not prove the exact serial release bracket')


def _require_panel_deadline(
    plan: ImmportAuthenticatedCollectionPlan,
    receipts: tuple[Any, ...],
) -> None:
    if (receipts[-1].completed_at - receipts[0].started_at).total_seconds() > (plan.panel_deadline_seconds):
        raise ImmportAuthenticatedCaptureError('ImmPort receipts exceed the precommitted panel deadline')


def _verify_attempt_policy(
    store: OperationalStore,
    run: LogicalRunRecord,
    configuration: ImmportAuthenticatedJobConfiguration,
) -> None:
    history = store.list_attempts(logical_run_id=run.logical_run_id)
    if tuple(item.attempt_number for item in history) != tuple(range(1, len(history) + 1)):
        raise ImmportAuthenticatedCaptureError('ImmPort attempt history is not contiguous')
    if len(history) > configuration.max_attempts_per_slot:
        raise ImmportAuthenticatedCaptureError('ImmPort attempt history exceeds its retry budget')
    for item in history:
        if item.started_at < run.scheduled_for:
            raise ImmportAuthenticatedCaptureError('ImmPort attempt predates its scheduled slot')
        if (item.lease_expires_at - item.started_at).total_seconds() != configuration.lease_seconds:
            raise ImmportAuthenticatedCaptureError('ImmPort attempt lease differs from immutable policy')
        plan = store.list_attempt_artifacts(item.attempt_id).get('collection-plan')
        if plan is None or plan.sha256 != configuration.collection_plan_sha256:
            raise ImmportAuthenticatedCaptureError('ImmPort attempt does not bind its collection plan')
    history_ids = {item.attempt_id for item in history}
    if any(
        event.event_type is LedgerEventType.ATTEMPT_LEASE_RENEWED and event.payload.get('attempt_id') in history_ids
        for event in store.events()
    ):
        raise ImmportAuthenticatedCaptureError('ImmPort attempt history contains a lease renewal')


def _configuration(values: Mapping[str, str | int | bool]) -> ImmportAuthenticatedJobConfiguration:
    configuration: ImmportAuthenticatedJobConfiguration | None = None
    try:
        configuration = parse_immport_authenticated_job_configuration(dict(values))
    except ValueError:
        pass
    if configuration is None:
        # Pydantic includes rejected input values in validation details.  Raise only after the
        # handler has exited so those details cannot survive as an exception cause or context.
        raise ImmportAuthenticatedCaptureError('authenticated ImmPort job configuration is not on the exact allowlist')
    return configuration


def _canonical_model(payload: bytes, model: Any, label: str) -> Any:
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise ImmportAuthenticatedCaptureError(f'invalid ImmPort {label}') from error
    if canonical_json_bytes(value) != payload:
        raise ImmportAuthenticatedCaptureError(f'ImmPort {label} is not canonical JSON')
    return value


def _operation_time(clock: Callable[[], datetime], label: str) -> datetime:
    try:
        return aware_utc(clock(), label)
    except (AttributeError, TypeError, ValueError) as error:
        raise ImmportAuthenticatedCaptureError(f'{label} must be an aware datetime') from error


def _fail_unfinished_attempt(
    store: OperationalStore,
    attempt: AttemptLease,
    *,
    owner_id: str,
    clock: Callable[[], datetime],
) -> None:
    try:
        current = store.get_attempt(attempt.attempt_id)
        now = _operation_time(clock, 'ImmPort failure observed_at')
        if current.state is AttemptState.STARTED and now < current.lease_expires_at:
            store.fail_attempt(
                current.attempt_id,
                owner_id=owner_id,
                terminal_code='collector_contract_rejected',
                now=now,
            )
    except Exception:
        pass


def _expected_study_urls(accession: str) -> tuple[str, ...]:
    study = f'https://www.immport.org/data/query/api/study/{accession}?format=json'
    manifest = f'https://www.immport.org/data/query/api/study/manifest/{accession}?fileType=release_file&format=json'
    return (
        _OPENAPI_URL,
        study,
        manifest,
        f'https://www.immport.org/data/query/api/study/arm/{accession}?format=json',
        f'https://www.immport.org/data/query/api/study/experiment/{accession}?format=json',
        f'https://www.immport.org/data/query/api/study/link/{accession}?format=json',
        manifest,
        study,
        _OPENAPI_URL,
    )


def _validate_plan_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != 'https'
        or parsed.netloc != 'www.immport.org'
        or parsed.hostname != 'www.immport.org'
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError('ImmPort plan URL must use the exact official HTTPS origin')
    try:
        pairs = tuple(parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True))
    except ValueError as error:
        raise ValueError('ImmPort plan URL has an invalid query') from error
    if len({key for key, _value in pairs}) != len(pairs):
        raise ValueError('ImmPort plan URL query names must be unique')
    if any(key not in {'fileType', 'format'} for key, _value in pairs):
        raise ValueError('ImmPort plan URL contains an unapproved query parameter')
    query = dict(pairs)
    if ('format' in query and query['format'] != 'json') or (
        'fileType' in query and query['fileType'] != 'release_file'
    ):
        raise ValueError('ImmPort plan URL contains an unapproved query value')
