"""Operator CLI for the single-host prospective capture layer."""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.collector import (
    STATIC_HTTPS_COLLECTOR_ID,
    StaticCollectionAttemptError,
    StaticHttpsCollectionPlan,
    run_static_https_collection,
    validate_static_job_configuration,
)
from vaxreplay.operations.collector_semantics import verify_all_supported_run_manifests
from vaxreplay.operations.immport_capture import (
    MAX_IMMPORT_PLAN_BYTES,
    ImmportAuthenticatedCollectionPlan,
    ImmportAuthenticatedRunManifest,
    record_immport_authenticated_capture,
)
from vaxreplay.operations.immport_producer import (
    InheritedFdImmportProducerInvoker,
    IsolatedImmportProducerClient,
)
from vaxreplay.operations.immport_producer_deployment import (
    ImmportProducerDeploymentError,
    ImmportProducerWorkloadPolicy,
    verify_immport_producer_execution_environment,
)
from vaxreplay.operations.policy import (
    IMMPORT_AUTHENTICATED_COLLECTOR_ID,
    parse_immport_authenticated_job_configuration,
)
from vaxreplay.operations.scheduler import enumerate_scheduled_slots, require_complete_registered_history
from vaxreplay.operations.schema import (
    AttemptState,
    CaptureJobSpec,
    LedgerCheckpoint,
    LogicalRunState,
    aware_utc,
    checkpoint_sha256,
)
from vaxreplay.operations.store import (
    AttemptBudgetExhaustedError,
    LeaseConflictError,
    OperationalStore,
    RunAlreadySucceededError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Operate append-only prospective source capture')
    subparsers = parser.add_subparsers(dest='command', required=True)

    initialize = subparsers.add_parser('init', help='initialize an empty local operations store')
    initialize.add_argument('--root', required=True)

    register = subparsers.add_parser('register-job', help='register one immutable job revision')
    register.add_argument('--root', required=True)
    register.add_argument('--spec', required=True)

    register_due = subparsers.add_parser('register-due', help='materialize every slot in a bounded inclusive window')
    register_due.add_argument('--root', required=True)
    register_due.add_argument('--job-spec-sha256', required=True)
    register_due.add_argument('--window-start', required=True)
    register_due.add_argument('--window-end', required=True)
    register_due.add_argument('--max-slots', type=int, default=1000)

    run_slot = subparsers.add_parser('run-static-slot', help='execute one registered exact-HTTPS logical run')
    run_slot.add_argument('--root', required=True)
    run_slot.add_argument('--logical-run-id', required=True)
    run_slot.add_argument('--plan', required=True)
    run_slot.add_argument('--owner-id', required=True)

    run_due = subparsers.add_parser('run-static-due', help='register and process the committed bounded catch-up window')
    run_due.add_argument('--root', required=True)
    run_due.add_argument('--job-spec-sha256', required=True)
    run_due.add_argument('--plan', required=True)
    run_due.add_argument('--owner-id', required=True)
    run_due.add_argument('--through', required=True)

    run_immport = subparsers.add_parser(
        'run-immport-slot',
        help='execute one claimed ImmPort slot through the isolated fd-3 producer',
    )
    run_immport.add_argument('--root', required=True)
    run_immport.add_argument('--logical-run-id', required=True)
    run_immport.add_argument('--plan', required=True)
    run_immport.add_argument('--owner-id', required=True)
    run_immport.add_argument('--execution-environment', required=True)
    run_immport.add_argument('--workload-policy', required=True)
    run_immport.add_argument('--workload-policy-sha256', required=True)

    reconcile = subparsers.add_parser('reconcile', help='retain and abandon expired leases')
    reconcile.add_argument('--root', required=True)

    verify = subparsers.add_parser('verify', help='verify the ledger, materialized state, and every registered object')
    verify.add_argument('--root', required=True)
    verify.add_argument('--checkpoint')

    checkpoint = subparsers.add_parser('checkpoint', help='write a canonical external-witness target')
    checkpoint.add_argument('--root', required=True)
    checkpoint.add_argument('--output', required=True)

    status = subparsers.add_parser('status', help='summarize durable local state without changing it')
    status.add_argument('--root', required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == 'init':
        store = OperationalStore.initialize(Path(args.root))
        report = store.verify()
        _write_json(
            {
                'event_count': report.event_count,
                'root': str(store.root),
                'schema_version': 'vaxreplay.operations-store.v0.1',
                'store_id': store.store_id,
                'tier_a_eligible': False,
            }
        )
        return

    store = OperationalStore(Path(args.root))
    if args.command == 'register-job':
        spec = _load_registration_spec(Path(args.spec))
        job = store.register_job(spec)
        _write_json(
            {
                'job_id': job.spec.job_id,
                'job_spec_sha256': job.spec_sha256,
                'registered_at': job.registered_at.isoformat(),
                'tier_a_eligible': False,
            }
        )
    elif args.command == 'register-due':
        observed_at = _clock_utc()
        window_start = _parse_datetime(args.window_start, 'window_start', now=observed_at)
        window_end = _parse_datetime(args.window_end, 'window_end', now=observed_at)
        _require_not_future(window_end, field_name='window_end', observed_at=observed_at)
        job = store.get_job(args.job_spec_sha256)
        slots = enumerate_scheduled_slots(
            job.spec,
            window_start=window_start,
            window_end=window_end,
            max_slots=args.max_slots,
        )
        runs = tuple(store.register_logical_run(job.spec_sha256, slot, registered_at=observed_at) for slot in slots)
        _write_json(
            {
                'job_spec_sha256': job.spec_sha256,
                'run_count': len(runs),
                'runs': [run.model_dump(mode='json') for run in runs],
            }
        )
    elif args.command == 'run-static-slot':
        observed_at = _clock_utc()
        run = store.get_logical_run(args.logical_run_id)
        _require_not_future(
            run.scheduled_for,
            field_name='logical run scheduled_for',
            observed_at=observed_at,
        )
        plan = StaticHttpsCollectionPlan.model_validate_json(Path(args.plan).read_bytes())
        result = run_static_https_collection(
            store,
            args.logical_run_id,
            plan,
            owner_id=args.owner_id,
        )
        _write_json(_run_result(result.manifest, result.manifest_artifact.sha256))
    elif args.command == 'run-static-due':
        observed_at = _clock_utc()
        through = _parse_datetime(args.through, 'through', now=observed_at)
        _require_not_future(through, field_name='through', observed_at=observed_at)
        plan = StaticHttpsCollectionPlan.model_validate_json(Path(args.plan).read_bytes())
        _write_json(
            _run_due(
                store,
                job_spec_sha256=args.job_spec_sha256,
                plan=plan,
                owner_id=args.owner_id,
                through=through,
                observed_at=observed_at,
            )
        )
    elif args.command == 'run-immport-slot':
        observed_at = _clock_utc()
        run = store.get_logical_run(args.logical_run_id)
        _require_not_future(
            run.scheduled_for,
            field_name='logical run scheduled_for',
            observed_at=observed_at,
        )
        job = store.get_job(run.job_spec_sha256)
        configuration = parse_immport_authenticated_job_configuration(job.spec.configuration)
        plan = _load_immport_plan(Path(args.plan))
        workload = _load_immport_execution_materials(
            execution_environment_path=Path(args.execution_environment),
            workload_policy_path=Path(args.workload_policy),
            expected_environment_sha256=(configuration.collector_execution_environment_sha256),
            expected_workload_policy_sha256=args.workload_policy_sha256,
            expected_collector_implementation_sha256=(configuration.collector_implementation_sha256),
        )
        if workload.plan_panel_deadline_seconds != plan.panel_deadline_seconds:
            raise ValueError('ImmPort workload policy differs from the plan deadline')
        invoker = InheritedFdImmportProducerInvoker(
            argv=workload.entrypoint,
            hard_deadline_margin_seconds=(
                workload.supervisor_hard_deadline_seconds - workload.plan_panel_deadline_seconds
            ),
        )
        producer = IsolatedImmportProducerClient(
            collector_implementation_sha256=configuration.collector_implementation_sha256,
            collector_execution_environment_sha256=(configuration.collector_execution_environment_sha256),
            invoke=invoker,
        )
        result = record_immport_authenticated_capture(
            store,
            run.logical_run_id,
            plan,
            owner_id=args.owner_id,
            producer=producer,
        )
        _write_json(_run_result(result.manifest, result.manifest_artifact.sha256))
    elif args.command == 'reconcile':
        observed_at = _clock_utc()
        abandoned = store.abandon_expired_attempts(now=observed_at)
        _write_json(
            {
                'abandoned_attempt_count': len(abandoned),
                'attempts': [attempt.model_dump(mode='json') for attempt in abandoned],
                'reconciled_at': observed_at.isoformat(),
            }
        )
    elif args.command == 'verify':
        checkpoint = None
        if args.checkpoint:
            checkpoint = LedgerCheckpoint.model_validate_json(Path(args.checkpoint).read_bytes())
        with store.verification_window():
            report = store.verify(checkpoint=checkpoint)
            manifests = verify_all_supported_run_manifests(store)
            static_manifests = tuple(
                item for item in manifests if not isinstance(item, ImmportAuthenticatedRunManifest)
            )
            immport_manifests = tuple(item for item in manifests if isinstance(item, ImmportAuthenticatedRunManifest))
        _write_json(
            {
                **report.model_dump(mode='json'),
                'static_run_manifest_count': len(static_manifests),
                'immport_authenticated_run_manifest_count': len(immport_manifests),
                'successful_run_semantics_verified': True,
                'semantic_verifier_scope': 'all_successful_runs_only',
                'local_integrity_verified': True,
                'tier_a_eligible': False,
            }
        )
    elif args.command == 'checkpoint':
        verified_manifests = []

        def verify_collector_semantics() -> None:
            verified_manifests.extend(verify_all_supported_run_manifests(store))

        checkpoint = store.checkpoint(semantic_verifier=verify_collector_semantics)
        output = _write_exclusive(Path(args.output), canonical_json_bytes(checkpoint))
        _write_json(
            {
                'checkpoint_path': str(output),
                'checkpoint_sha256': checkpoint_sha256(checkpoint),
                'successful_run_semantics_verified': True,
                'semantic_verifier_scope': 'all_successful_runs_only',
                'external_timestamp_required': True,
                'static_run_manifest_count': sum(
                    not isinstance(item, ImmportAuthenticatedRunManifest) for item in verified_manifests
                ),
                'immport_authenticated_run_manifest_count': sum(
                    isinstance(item, ImmportAuthenticatedRunManifest) for item in verified_manifests
                ),
                'through_event_sha256': checkpoint.through_event_sha256,
                'through_sequence': checkpoint.through_sequence,
                'tier_a_eligible': False,
            }
        )
    elif args.command == 'status':
        _write_json(_status(store))


def _run_due(
    store: OperationalStore,
    *,
    job_spec_sha256: str,
    plan: StaticHttpsCollectionPlan,
    owner_id: str,
    through: datetime,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    through = aware_utc(through, 'through')
    observed_at = aware_utc(observed_at or _clock_utc(), 'observed_at')
    _require_not_future(through, field_name='through', observed_at=observed_at)
    job = store.get_job(job_spec_sha256)
    catch_up_seconds = _configured_int(job.spec, 'catch_up_seconds', minimum=0, maximum=366 * 24 * 60 * 60)
    max_slots = _configured_int(job.spec, 'max_slots_per_wake', minimum=1, maximum=10_000)
    max_attempts = _configured_int(job.spec, 'max_attempts_per_slot', minimum=1, maximum=100)
    lease_seconds = _configured_int(job.spec, 'lease_seconds', minimum=1, maximum=24 * 60 * 60)
    catch_up_boundary = through - timedelta(seconds=catch_up_seconds)
    registered_runs = store.list_logical_runs(job_spec_sha256=job.spec_sha256)
    require_complete_registered_history(
        job.spec,
        (run.scheduled_for for run in registered_runs),
        before=catch_up_boundary,
    )
    store.abandon_expired_attempts(now=observed_at)
    slots = enumerate_scheduled_slots(
        job.spec,
        window_start=catch_up_boundary,
        window_end=through,
        max_slots=max_slots,
    )
    results: list[dict[str, object]] = []
    for slot in slots:
        run = store.register_logical_run(job.spec_sha256, slot, registered_at=observed_at)
        if run.state is LogicalRunState.SUCCEEDED:
            results.append({'logical_run_id': run.logical_run_id, 'status': 'already_succeeded'})
            continue
        attempts = store.list_attempts(logical_run_id=run.logical_run_id)
        if any(attempt.state is AttemptState.STARTED for attempt in attempts):
            results.append({'logical_run_id': run.logical_run_id, 'status': 'active_lease'})
            continue
        if len(attempts) >= max_attempts:
            results.append(
                {
                    'attempt_count': len(attempts),
                    'logical_run_id': run.logical_run_id,
                    'status': 'retry_budget_exhausted_nonterminal',
                }
            )
            continue
        try:
            result = run_static_https_collection(
                store,
                run.logical_run_id,
                plan,
                owner_id=owner_id,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
            )
            results.append(
                {
                    'attempt_id': result.attempt.attempt_id,
                    'logical_run_id': run.logical_run_id,
                    'manifest_sha256': result.manifest_artifact.sha256,
                    'status': 'succeeded',
                }
            )
        except StaticCollectionAttemptError as error:
            terminal = store.get_attempt(error.attempt_id)
            if terminal.state is AttemptState.STARTED:
                raise
            results.append(
                {
                    'attempt_id': terminal.attempt_id,
                    'logical_run_id': run.logical_run_id,
                    'status': terminal.state.value,
                    'terminal_code': terminal.terminal_code,
                }
            )
        except LeaseConflictError:
            results.append({'logical_run_id': run.logical_run_id, 'status': 'claim_conflict'})
        except AttemptBudgetExhaustedError:
            attempt_count = len(store.list_attempts(logical_run_id=run.logical_run_id))
            results.append(
                {
                    'attempt_count': attempt_count,
                    'logical_run_id': run.logical_run_id,
                    'status': 'retry_budget_exhausted_nonterminal',
                }
            )
        except RunAlreadySucceededError:
            results.append({'logical_run_id': run.logical_run_id, 'status': 'already_succeeded'})
    return {
        'job_spec_sha256': job_spec_sha256,
        'results': results,
        'slot_count': len(slots),
        'through': through.isoformat(),
        'tier_a_eligible': False,
    }


def _configured_int(
    spec: CaptureJobSpec,
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = spec.configuration.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f'job configuration {key!r} must be an integer from {minimum} through {maximum}')
    return value


def _run_result(manifest, manifest_sha256: str) -> dict[str, object]:
    return {
        'attempt_id': manifest.attempt_id,
        'logical_run_id': manifest.logical_run_id,
        'manifest_sha256': manifest_sha256,
        'plan_complete': manifest.plan_complete,
        'source_enumeration_complete': manifest.source_enumeration_complete,
        'status': 'succeeded',
        'tier_a_eligible': False,
    }


def _status(store: OperationalStore) -> dict[str, object]:
    with store.verification_window():
        report = store.verify()
        manifests = verify_all_supported_run_manifests(store)
        runs = store.list_logical_runs()
        attempts = store.list_attempts()
    return {
        'attempt_states': dict(sorted(Counter(attempt.state.value for attempt in attempts).items())),
        'successful_run_semantics_verified': True,
        'semantic_verifier_scope': 'all_successful_runs_only',
        'external_checkpoint_verified': False,
        'job_count': report.job_count,
        'ledger_event_count': report.event_count,
        'ledger_head_sha256': report.ledger_head_sha256,
        'local_integrity_verified': True,
        'logical_run_states': dict(sorted(Counter(run.state.value for run in runs).items())),
        'object_count': report.object_count,
        'static_run_manifest_count': sum(not isinstance(item, ImmportAuthenticatedRunManifest) for item in manifests),
        'immport_authenticated_run_manifest_count': sum(
            isinstance(item, ImmportAuthenticatedRunManifest) for item in manifests
        ),
        'store_id': store.store_id,
        'tier_a_eligible': False,
    }


def _clock_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_registration_spec(path: Path) -> CaptureJobSpec:
    """Parse one public job spec without reflecting rejected values in an exception."""

    spec: CaptureJobSpec | None = None
    try:
        candidate = CaptureJobSpec.model_validate_json(path.read_bytes())
        if candidate.collector_id == STATIC_HTTPS_COLLECTOR_ID:
            validate_static_job_configuration(candidate.configuration)
        elif candidate.collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID:
            parse_immport_authenticated_job_configuration(candidate.configuration)
        spec = candidate
    except (OSError, ValueError):
        pass
    if spec is None:
        # Raise outside the handler: the rejected parser exception can contain input values in
        # its rendered details, so it must not survive as either cause or implicit context.
        raise ValueError('job specification is invalid')
    return spec


def _load_immport_plan(path: Path) -> ImmportAuthenticatedCollectionPlan:
    plan: ImmportAuthenticatedCollectionPlan | None = None
    try:
        payload = path.read_bytes()
        candidate = ImmportAuthenticatedCollectionPlan.model_validate_json(payload)
        if len(payload) <= MAX_IMMPORT_PLAN_BYTES and canonical_json_bytes(candidate) == payload:
            plan = candidate
    except (OSError, ValueError):
        pass
    if plan is None:
        raise ValueError('ImmPort collection plan is invalid')
    return plan


def _load_immport_execution_materials(
    *,
    execution_environment_path: Path,
    workload_policy_path: Path,
    expected_environment_sha256: str,
    expected_workload_policy_sha256: str,
    expected_collector_implementation_sha256: str,
) -> ImmportProducerWorkloadPolicy:
    """Cross-bind once-read public runtime materials to independent registered pins."""

    policy: ImmportProducerWorkloadPolicy | None = None
    try:
        environment_bytes = _read_bounded_file(
            execution_environment_path,
            maximum_bytes=8 * 1024 * 1024,
        )
        workload_bytes = _read_bounded_file(
            workload_policy_path,
            maximum_bytes=1024 * 1024,
        )
        verify_immport_producer_execution_environment(
            environment_bytes,
            expected_environment_sha256=expected_environment_sha256,
            workload_policy_bytes=workload_bytes,
            expected_workload_policy_sha256=expected_workload_policy_sha256,
            expected_collector_implementation_sha256=(expected_collector_implementation_sha256),
        )
        candidate = ImmportProducerWorkloadPolicy.model_validate_json(workload_bytes)
        if canonical_json_bytes(candidate) == workload_bytes:
            policy = candidate
    except (ImmportProducerDeploymentError, OSError, ValueError):
        pass
    if policy is None:
        raise ValueError('ImmPort execution environment or workload policy differs from its trusted pins')
    return policy


def _read_bounded_file(path: Path, *, maximum_bytes: int) -> bytes:
    if maximum_bytes < 1:
        raise ValueError('maximum file size must be positive')
    requested = path.expanduser().absolute()
    descriptor = os.open(
        requested,
        os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum_bytes:
            raise ValueError('artifact must be a bounded nonempty regular file')
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError('artifact changed while being read')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError('artifact changed while being read')
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before:
            raise ValueError('artifact changed while being read')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


def _parse_datetime(value: str, field_name: str, *, now: datetime | None = None) -> datetime:
    if value == 'now':
        return aware_utc(now or _clock_utc(), 'now')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError(f'{field_name} must be an RFC 3339 timestamp') from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return parsed.astimezone(timezone.utc)


def _require_not_future(value: datetime, *, field_name: str, observed_at: datetime) -> None:
    value = aware_utc(value, field_name)
    observed_at = aware_utc(observed_at, 'observed_at')
    if value > observed_at:
        raise ValueError(
            f'{field_name} cannot be in the future relative to the local worker clock '
            f'({value.isoformat()} > {observed_at.isoformat()})'
        )


def _write_exclusive(path: Path, payload: bytes) -> Path:
    requested = path.expanduser().absolute()
    if os.path.lexists(requested):
        raise FileExistsError(f'output already exists: {requested}')
    requested.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{requested.name}.', dir=requested.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, requested)
        temporary.unlink()
        directory = os.open(requested.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return requested


def _write_json(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
