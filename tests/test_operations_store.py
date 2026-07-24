from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from unittest.mock import patch

from pydantic import ValidationError

from vaxreplay.operations import (
    STATIC_HTTPS_COLLECTOR_ID,
    AttemptBudgetExhaustedError,
    AttemptState,
    AttemptStateError,
    CaptureJobSpec,
    LeaseConflictError,
    LogicalRunState,
    OperationalStore,
    OperationsIntegrityError,
    OperationsStoreError,
    RunAlreadySucceededError,
    job_spec_sha256,
    scheduled_logical_run_id,
)

_T0 = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
_STORE_ID = '0123456789abcdef0123456789abcdef'


def _job(*, job_id: str = 'official-page-hourly') -> CaptureJobSpec:
    return CaptureJobSpec(
        job_id=job_id,
        collector_id='exact-https-v0.1',
        schedule_anchor_at=_T0,
        schedule_interval_seconds=3600,
        configuration={'allowed_host': 'example.org', 'max_body_bytes': 1024},
    )


class OperationalStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / 'operations'
        self.store = OperationalStore.initialize(
            self.root,
            created_at=_T0,
            store_id=_STORE_ID,
            trusted_lease_clock=None,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _registered_run(self, *, slot: int = 0):
        job = self.store.register_job(_job(), registered_at=_T0 + timedelta(seconds=1))
        run = self.store.register_logical_run(
            job.spec_sha256,
            _T0 + timedelta(hours=slot),
            registered_at=_T0 + timedelta(seconds=2),
        )
        return job, run

    def test_initialization_job_and_slot_registration_are_idempotent(self) -> None:
        report = self.store.verify(verified_at=_T0)
        self.assertEqual(report.event_count, 1)
        self.assertEqual(report.store_id, _STORE_ID)

        spec = _job()
        first_job = self.store.register_job(spec, registered_at=_T0 + timedelta(seconds=1))
        second_job = self.store.register_job(spec, registered_at=_T0 + timedelta(days=1))
        self.assertEqual(first_job, second_job)
        self.assertEqual(first_job.spec_sha256, job_spec_sha256(spec))

        slot = _T0 + timedelta(hours=4)
        first_run = self.store.register_logical_run(
            first_job.spec_sha256,
            slot,
            registered_at=_T0 + timedelta(seconds=2),
        )
        second_run = self.store.register_logical_run(
            first_job.spec_sha256,
            slot,
            registered_at=_T0 + timedelta(days=1),
        )
        self.assertEqual(first_run, second_run)
        self.assertEqual(first_run.logical_run_id, scheduled_logical_run_id(first_job.spec_sha256, slot))
        self.assertEqual(self.store.verify().event_count, 3)

        with self.assertRaisesRegex(ValueError, 'exactly on'):
            self.store.register_logical_run(first_job.spec_sha256, slot + timedelta(microseconds=1))
        with self.assertRaisesRegex(ValueError, 'precede'):
            self.store.register_logical_run(first_job.spec_sha256, _T0 - timedelta(hours=1))

    def test_job_revisions_are_immutable_and_secret_keys_are_rejected(self) -> None:
        first = self.store.register_job(_job(), registered_at=_T0)
        changed = _job().model_copy(update={'configuration': {'allowed_host': 'example.org', 'max_body_bytes': 2048}})
        second = self.store.register_job(changed, registered_at=_T0 + timedelta(seconds=1))
        self.assertNotEqual(first.spec_sha256, second.spec_sha256)
        self.assertEqual(first.spec.job_id, second.spec.job_id)

        with self.assertRaises(ValidationError):
            CaptureJobSpec(
                job_id='bad',
                collector_id='collector',
                schedule_anchor_at=_T0,
                schedule_interval_seconds=60,
                configuration={'api_token': 'must-not-persist'},
            )

        connection = sqlite3.connect(self.store.database_path)
        try:
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'jobs are immutable'):
                connection.execute("UPDATE jobs SET job_id = 'changed'")
        finally:
            connection.close()

    def test_cas_deduplicates_concurrent_writers_and_binds_exact_bytes(self) -> None:
        payload = b'exact source bytes\x00\xff'

        def put(_ordinal: int):
            return self.store.put_bytes(payload, recorded_at=_T0 + timedelta(seconds=1))

        with ThreadPoolExecutor(max_workers=8) as executor:
            artifacts = tuple(executor.map(put, range(16)))
        self.assertEqual({artifact.sha256 for artifact in artifacts}, {hashlib.sha256(payload).hexdigest()})
        self.assertEqual({artifact.first_recorded_at for artifact in artifacts}, {_T0 + timedelta(seconds=1)})
        self.assertEqual(self.store.read_artifact(artifacts[0].sha256), payload)
        self.assertEqual(
            sum(event.event_type.value == 'artifact_stored' for event in self.store.events()),
            1,
        )
        self.assertEqual(self.store.verify().object_count, 1)

    def test_put_file_rejects_symlink_and_byte_limit(self) -> None:
        source = self.root.parent / 'source.bin'
        source.write_bytes(b'1234')
        link = self.root.parent / 'source-link.bin'
        link.symlink_to(source)
        with self.assertRaisesRegex(OperationsStoreError, 'symbolic link'):
            self.store.put_file(link)
        with self.assertRaisesRegex(OperationsStoreError, 'max_bytes'):
            self.store.put_file(source, max_bytes=3)
        self.assertEqual(self.store.verify().object_count, 0)

    def test_cas_rejects_symlinked_digest_shard_without_writing_outside_root(self) -> None:
        payload = b'shard symlink attack'
        shard = hashlib.sha256(payload).hexdigest()[:2]
        outside = self.root.parent / 'outside'
        outside.mkdir()
        (self.store.objects_root / shard).symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(OperationsIntegrityError, 'safe directory'):
            self.store.put_bytes(payload, recorded_at=_T0 + timedelta(seconds=1))
        self.assertEqual(tuple(outside.iterdir()), ())

    def test_verification_observes_one_snapshot_during_concurrent_writes(self) -> None:
        def write(ordinal: int) -> None:
            self.store.put_bytes(
                f'concurrent-{ordinal}'.encode(),
                recorded_at=_T0 + timedelta(seconds=ordinal + 1),
            )

        def verify(_ordinal: int) -> None:
            self.store.verify()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(write, ordinal) for ordinal in range(32)]
            futures.extend(executor.submit(verify, ordinal) for ordinal in range(32))
            for future in futures:
                future.result()
        self.assertEqual(self.store.verify().object_count, 32)

    def test_failed_attempt_is_retained_before_successful_retry(self) -> None:
        _job_record, run = self._registered_run()
        first = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker-a',
            now=_T0 + timedelta(minutes=1),
            lease_seconds=60,
        )
        failed = self.store.fail_attempt(
            first.attempt_id,
            owner_id='worker-a',
            terminal_code='http_503',
            now=_T0 + timedelta(minutes=1, seconds=1),
        )
        self.assertEqual(failed.state, AttemptState.FAILED)
        self.assertEqual(self.store.get_logical_run(run.logical_run_id).state, LogicalRunState.PENDING)

        artifact = self.store.put_bytes(b'ok', recorded_at=_T0 + timedelta(minutes=2))
        second = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker-b',
            now=_T0 + timedelta(minutes=2),
            lease_seconds=60,
        )
        self.assertEqual(second.attempt_number, 2)
        self.store.attach_artifact(
            second.attempt_id,
            owner_id='worker-b',
            role='run-manifest',
            artifact_sha256=artifact.sha256,
            now=_T0 + timedelta(minutes=2, seconds=1),
        )
        succeeded = self.store.succeed_attempt(
            second.attempt_id,
            owner_id='worker-b',
            now=_T0 + timedelta(minutes=2, seconds=2),
        )
        self.assertEqual(succeeded.state, AttemptState.SUCCEEDED)
        completed_run = self.store.get_logical_run(run.logical_run_id)
        self.assertEqual(completed_run.state, LogicalRunState.SUCCEEDED)
        self.assertEqual(completed_run.successful_attempt_id, second.attempt_id)
        self.assertEqual(self.store.get_attempt(first.attempt_id).terminal_code, 'http_503')
        with self.assertRaises(RunAlreadySucceededError):
            self.store.begin_attempt(
                run.logical_run_id,
                owner_id='worker-c',
                now=_T0 + timedelta(minutes=3),
            )
        self.store.verify()

    def test_attempt_operations_reject_a_clock_value_before_attempt_start(self) -> None:
        _job_record, run = self._registered_run()
        artifact = self.store.put_bytes(b'preexisting', recorded_at=_T0 + timedelta(seconds=3))
        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker-a',
            now=_T0 + timedelta(minutes=1),
            lease_seconds=60,
        )

        with self.assertRaisesRegex(LeaseConflictError, 'predate'):
            self.store.attach_artifact(
                attempt.attempt_id,
                owner_id='worker-a',
                role='body',
                artifact_sha256=artifact.sha256,
                now=_T0 + timedelta(seconds=30),
            )

        self.assertEqual(self.store.list_attempt_artifacts(attempt.attempt_id), {})
        self.store.verify(verified_at=_T0 + timedelta(minutes=1))

    def test_attempt_claim_atomically_binds_initial_artifacts_or_rolls_back(self) -> None:
        _job_record, run = self._registered_run()
        plan = self.store.put_bytes(b'canonical collection plan', recorded_at=_T0 + timedelta(seconds=3))

        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker-a',
            now=_T0 + timedelta(minutes=1),
            initial_artifacts={'collection-plan': plan.sha256},
        )

        attachments = self.store.list_attempt_artifacts(attempt.attempt_id)
        self.assertEqual(tuple(attachments), ('collection-plan',))
        self.assertEqual(attachments['collection-plan'], plan)
        self.assertEqual(
            tuple(event.event_type.value for event in self.store.events())[-2:],
            ('attempt_started', 'attempt_artifact_attached'),
        )

        _job_record, other_run = self._registered_run(slot=1)
        with self.assertRaisesRegex(OperationsStoreError, 'unknown artifact'):
            self.store.begin_attempt(
                other_run.logical_run_id,
                owner_id='worker-b',
                now=_T0 + timedelta(hours=1, minutes=1),
                initial_artifacts={'collection-plan': 'f' * 64},
            )
        self.assertEqual(self.store.list_attempts(logical_run_id=other_run.logical_run_id), ())
        self.assertEqual(self.store.get_logical_run(other_run.logical_run_id).state, LogicalRunState.PENDING)
        self.store.verify()

    def test_static_claim_derives_plan_lease_retry_budget_and_due_time_from_job(self) -> None:
        plan = self.store.put_bytes(b'immutable static plan', recorded_at=_T0)
        job = self.store.register_job(
            CaptureJobSpec(
                job_id='static-policy-test',
                collector_id=STATIC_HTTPS_COLLECTOR_ID,
                schedule_anchor_at=_T0,
                schedule_interval_seconds=3600,
                configuration={
                    'collection_plan_sha256': plan.sha256,
                    'dns_resolution_attempts': 1,
                    'dns_resolution_timeout_seconds': 5,
                    'lease_seconds': 5,
                    'max_dns_addresses': 16,
                    'max_attempts_per_slot': 1,
                    'max_total_body_bytes': 1024,
                    'plan_deadline_seconds': 5,
                    'request_deadline_seconds': 5,
                    'source_id': 'publisher:example',
                },
            ),
            registered_at=_T0,
        )
        run = self.store.register_logical_run(job.spec_sha256, _T0, registered_at=_T0)
        intent = {'collection-plan': plan.sha256}

        with self.assertRaisesRegex(OperationsStoreError, 'differs from the immutable static job policy'):
            self.store.begin_attempt(
                run.logical_run_id,
                owner_id='worker',
                now=_T0 + timedelta(seconds=1),
                lease_seconds=10,
                initial_artifacts=intent,
            )
        with self.assertRaisesRegex(OperationsStoreError, 'atomically bind'):
            self.store.begin_attempt(
                run.logical_run_id,
                owner_id='worker',
                now=_T0 + timedelta(seconds=1),
            )

        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker',
            now=_T0 + timedelta(seconds=1),
            initial_artifacts=intent,
        )
        self.assertEqual(attempt.lease_expires_at - attempt.started_at, timedelta(seconds=5))
        with self.assertRaisesRegex(AttemptStateError, 'cannot be renewed'):
            self.store.renew_lease(
                attempt.attempt_id,
                owner_id='worker',
                now=_T0 + timedelta(seconds=2),
                lease_seconds=10,
            )
        self.store.fail_attempt(
            attempt.attempt_id,
            owner_id='worker',
            terminal_code='retryable',
            now=_T0 + timedelta(seconds=2),
        )
        with self.assertRaises(AttemptBudgetExhaustedError):
            self.store.begin_attempt(
                run.logical_run_id,
                owner_id='worker',
                now=_T0 + timedelta(seconds=3),
                initial_artifacts=intent,
            )

        future_run = self.store.register_logical_run(
            job.spec_sha256,
            _T0 + timedelta(hours=1),
            registered_at=_T0,
        )
        with self.assertRaisesRegex(OperationsStoreError, 'before its scheduled time'):
            self.store.begin_attempt(
                future_run.logical_run_id,
                owner_id='worker',
                now=_T0 + timedelta(minutes=59),
                initial_artifacts=intent,
            )
        self.assertEqual(self.store.list_attempts(logical_run_id=future_run.logical_run_id), ())
        self.store.verify(verified_at=_T0 + timedelta(hours=1))

    def test_begin_attempt_samples_trusted_clock_after_initial_object_verification(self) -> None:
        _job_record, run = self._registered_run()
        intent = self.store.put_bytes(b'intent', recorded_at=_T0)
        times = iter((_T0, _T0 + timedelta(seconds=2)))
        trusted = OperationalStore(self.root, trusted_lease_clock=lambda: next(times))
        original_verify = trusted._verify_artifact_object

        def verify_then_advance(artifact):
            original_verify(artifact)
            next(times)

        # The verification hook consumes the first clock value; claim time must use the second.
        times = iter((_T0, _T0 + timedelta(seconds=2)))
        with patch.object(trusted, '_verify_artifact_object', side_effect=verify_then_advance):
            attempt = trusted.begin_attempt(
                run.logical_run_id,
                owner_id='worker',
                lease_seconds=1,
                initial_artifacts={'intent': intent.sha256},
            )
        self.assertEqual(attempt.started_at, _T0 + timedelta(seconds=2))
        self.assertEqual(attempt.lease_expires_at, _T0 + timedelta(seconds=3))

    def test_renewal_samples_trusted_clock_after_waiting_for_write_lock(self) -> None:
        _job_record, run = self._registered_run()
        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker',
            now=_T0,
            lease_seconds=1,
        )
        current = [_T0 + timedelta(microseconds=500_000)]
        clock_sampled = Event()

        def trusted_clock() -> datetime:
            clock_sampled.set()
            return current[0]

        trusted = OperationalStore(self.root, trusted_lease_clock=trusted_clock)
        blocker = sqlite3.connect(self.store.database_path, isolation_level=None)
        blocker.execute('BEGIN IMMEDIATE')

        def renew() -> str:
            try:
                trusted.renew_lease(attempt.attempt_id, owner_id='worker', lease_seconds=2)
                return 'renewed'
            except LeaseConflictError:
                return 'expired'

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(renew)
                clock_sampled.wait(0.05)
                current[0] = _T0 + timedelta(seconds=2)
                blocker.commit()
                self.assertEqual(future.result(), 'expired')
        finally:
            blocker.close()
        self.assertEqual(self.store.get_attempt(attempt.attempt_id).lease_expires_at, attempt.lease_expires_at)

    def test_verify_rejects_events_later_than_its_claimed_verification_time(self) -> None:
        self.store.put_bytes(b'future event', recorded_at=_T0 + timedelta(seconds=1))
        with self.assertRaisesRegex(OperationsIntegrityError, 'verification time predates'):
            self.store.verify(verified_at=_T0)
        self.store.verify(verified_at=_T0 + timedelta(seconds=1))

    def test_concurrent_claims_cannot_exceed_attempt_budget(self) -> None:
        _job_record, run = self._registered_run()

        def claim(round_number: int, ordinal: int):
            try:
                return self.store.begin_attempt(
                    run.logical_run_id,
                    owner_id=f'worker-{round_number}-{ordinal}',
                    now=_T0 + timedelta(minutes=round_number),
                    max_attempts=2,
                )
            except LeaseConflictError:
                return 'lease-conflict'
            except AttemptBudgetExhaustedError:
                return 'budget-exhausted'

        with ThreadPoolExecutor(max_workers=8) as executor:
            first_round = tuple(executor.map(lambda ordinal: claim(1, ordinal), range(8)))
        first_winners = tuple(item for item in first_round if not isinstance(item, str))
        self.assertEqual(len(first_winners), 1)
        first = first_winners[0]
        self.store.fail_attempt(
            first.attempt_id,
            owner_id=first.owner_id,
            terminal_code='retryable',
            now=_T0 + timedelta(minutes=1, seconds=1),
        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            second_round = tuple(executor.map(lambda ordinal: claim(2, ordinal), range(8)))
        second_winners = tuple(item for item in second_round if not isinstance(item, str))
        self.assertEqual(len(second_winners), 1)
        second = second_winners[0]
        self.store.fail_attempt(
            second.attempt_id,
            owner_id=second.owner_id,
            terminal_code='retryable',
            now=_T0 + timedelta(minutes=2, seconds=1),
        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            exhausted_round = tuple(executor.map(lambda ordinal: claim(3, ordinal), range(8)))
        self.assertEqual(exhausted_round, ('budget-exhausted',) * 8)
        self.assertEqual(len(self.store.list_attempts(logical_run_id=run.logical_run_id)), 2)
        self.store.verify()

    def test_lease_owner_expiry_renewal_and_reconciliation(self) -> None:
        _job_record, run = self._registered_run()
        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker-a',
            now=_T0 + timedelta(minutes=1),
            lease_seconds=10,
        )
        with self.assertRaises(LeaseConflictError):
            self.store.renew_lease(
                attempt.attempt_id,
                owner_id='worker-b',
                now=_T0 + timedelta(minutes=1, seconds=1),
                lease_seconds=20,
            )
        renewed = self.store.renew_lease(
            attempt.attempt_id,
            owner_id='worker-a',
            now=_T0 + timedelta(minutes=1, seconds=1),
            lease_seconds=30,
        )
        self.assertEqual(renewed.lease_expires_at, _T0 + timedelta(minutes=1, seconds=31))
        with self.assertRaises(LeaseConflictError):
            self.store.begin_attempt(
                run.logical_run_id,
                owner_id='worker-b',
                now=_T0 + timedelta(minutes=2),
            )
        abandoned = self.store.abandon_expired_attempts(now=_T0 + timedelta(minutes=2))
        self.assertEqual(tuple(item.attempt_id for item in abandoned), (attempt.attempt_id,))
        self.assertEqual(abandoned[0].state, AttemptState.ABANDONED)
        retry = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker-b',
            now=_T0 + timedelta(minutes=2, seconds=1),
        )
        self.assertEqual(retry.attempt_number, 2)
        self.store.verify()

    def test_trusted_clock_rechecks_lease_immediately_before_mutation(self) -> None:
        _job_record, success_run = self._registered_run()
        success_times = iter(
            (
                _T0 + timedelta(seconds=1),
                _T0 + timedelta(seconds=1, microseconds=500_000),
                _T0 + timedelta(seconds=2),
            )
        )
        trusted = OperationalStore(self.root, trusted_lease_clock=lambda: next(success_times))
        success_attempt = trusted.begin_attempt(
            success_run.logical_run_id,
            owner_id='worker',
            lease_seconds=1,
        )
        manifest = self.store.put_bytes(b'manifest', recorded_at=_T0 + timedelta(seconds=1))
        with self.assertRaisesRegex(LeaseConflictError, 'expired'):
            trusted.succeed_attempt(
                success_attempt.attempt_id,
                owner_id='worker',
                run_manifest_sha256=manifest.sha256,
            )
        self.assertEqual(self.store.get_attempt(success_attempt.attempt_id).state, AttemptState.STARTED)
        self.assertEqual(self.store.list_attempt_artifacts(success_attempt.attempt_id), {})

        _job_record, attach_run = self._registered_run(slot=1)
        attach_times = iter(
            (
                _T0 + timedelta(hours=1, seconds=1),
                _T0 + timedelta(hours=1, seconds=1, microseconds=500_000),
                _T0 + timedelta(hours=1, seconds=2),
            )
        )
        trusted = OperationalStore(self.root, trusted_lease_clock=lambda: next(attach_times))
        attach_attempt = trusted.begin_attempt(
            attach_run.logical_run_id,
            owner_id='worker',
            lease_seconds=1,
        )
        with self.assertRaisesRegex(LeaseConflictError, 'expired'):
            trusted.attach_artifact(
                attach_attempt.attempt_id,
                owner_id='worker',
                role='body',
                artifact_sha256=manifest.sha256,
            )
        self.assertEqual(self.store.list_attempt_artifacts(attach_attempt.attempt_id), {})

    def test_concurrent_claim_and_terminal_calls_have_single_winner(self) -> None:
        _job_record, run = self._registered_run()

        def claim(ordinal: int):
            try:
                return self.store.begin_attempt(
                    run.logical_run_id,
                    owner_id=f'worker-{ordinal}',
                    now=_T0 + timedelta(minutes=1),
                )
            except LeaseConflictError:
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            claims = tuple(executor.map(claim, range(8)))
        winners = tuple(claim_item for claim_item in claims if claim_item is not None)
        self.assertEqual(len(winners), 1)
        winner = winners[0]
        artifact = self.store.put_bytes(b'body', recorded_at=_T0 + timedelta(minutes=2))
        self.store.attach_artifact(
            winner.attempt_id,
            owner_id=winner.owner_id,
            role='run-manifest',
            artifact_sha256=artifact.sha256,
            now=_T0 + timedelta(minutes=2),
        )

        def complete(_ordinal: int) -> str:
            try:
                self.store.succeed_attempt(
                    winner.attempt_id,
                    owner_id=winner.owner_id,
                    now=_T0 + timedelta(minutes=2, seconds=1),
                )
                return 'success'
            except AttemptStateError:
                return 'terminal'

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = tuple(executor.map(complete, range(8)))
        self.assertEqual(outcomes.count('success'), 1)
        self.assertEqual(outcomes.count('terminal'), 7)
        self.assertEqual(self.store.verify().attempt_count, 1)

    def test_success_requires_artifact_and_attachment_roles_are_immutable(self) -> None:
        _job_record, run = self._registered_run()
        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker',
            now=_T0 + timedelta(minutes=1),
        )
        with self.assertRaisesRegex(AttemptStateError, 'run-manifest'):
            self.store.succeed_attempt(
                attempt.attempt_id,
                owner_id='worker',
                now=_T0 + timedelta(minutes=1, seconds=1),
            )
        first = self.store.put_bytes(b'first', recorded_at=_T0 + timedelta(minutes=2))
        second = self.store.put_bytes(b'second', recorded_at=_T0 + timedelta(minutes=2))
        self.store.attach_artifact(
            attempt.attempt_id,
            owner_id='worker',
            role='body',
            artifact_sha256=first.sha256,
            now=_T0 + timedelta(minutes=2),
        )
        with self.assertRaisesRegex(AttemptStateError, 'different bytes'):
            self.store.attach_artifact(
                attempt.attempt_id,
                owner_id='worker',
                role='body',
                artifact_sha256=second.sha256,
                now=_T0 + timedelta(minutes=2, seconds=1),
            )

    def test_success_can_atomically_bind_terminal_manifest(self) -> None:
        _job_record, run = self._registered_run()
        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker',
            now=_T0 + timedelta(minutes=1),
        )
        manifest = self.store.put_bytes(b'canonical terminal manifest', recorded_at=_T0 + timedelta(minutes=2))
        succeeded = self.store.succeed_attempt(
            attempt.attempt_id,
            owner_id='worker',
            run_manifest_sha256=manifest.sha256,
            now=_T0 + timedelta(minutes=2, seconds=1),
        )
        self.assertEqual(succeeded.state, AttemptState.SUCCEEDED)
        self.assertEqual(self.store.list_attempt_artifacts(attempt.attempt_id)['run-manifest'], manifest)
        event_types = tuple(event.event_type.value for event in self.store.events())
        self.assertEqual(event_types[-2:], ('attempt_artifact_attached', 'attempt_succeeded'))
        self.store.verify()

    def test_checkpoint_prefix_remains_verifiable_after_later_events(self) -> None:
        first = self.store.put_bytes(b'first', recorded_at=_T0 + timedelta(seconds=1))
        checkpoint = self.store.checkpoint(created_at=_T0 + timedelta(seconds=2))
        self.assertEqual(checkpoint.object_count, 1)
        second = self.store.put_bytes(b'second', recorded_at=_T0 + timedelta(seconds=3))
        self.assertNotEqual(first.sha256, second.sha256)
        self.store.verify_checkpoint(checkpoint)
        self.assertTrue(self.store.verify(checkpoint=checkpoint).checkpoint_verified)

        wrong = checkpoint.model_copy(update={'through_event_sha256': '0' * 64})
        with self.assertRaisesRegex(OperationsIntegrityError, 'prefix'):
            self.store.verify_checkpoint(wrong)

    def test_checkpoint_blocks_writers_between_verification_and_head_selection(self) -> None:
        self.store.put_bytes(b'before', recorded_at=_T0 + timedelta(seconds=1))
        original_verify = self.store.verify
        writer_started = False

        def interposed_verify(*args, **kwargs):
            nonlocal writer_started
            report = original_verify(*args, **kwargs)
            if not writer_started:
                writer_started = True
                executor.submit(
                    self.store.put_bytes,
                    b'after',
                    recorded_at=_T0 + timedelta(seconds=2),
                )
            return report

        with ThreadPoolExecutor(max_workers=1) as executor:
            with patch.object(self.store, 'verify', side_effect=interposed_verify):
                checkpoint = self.store.checkpoint(created_at=_T0 + timedelta(seconds=3))
        self.assertEqual(checkpoint.object_count, 1)
        self.assertEqual(self.store.verify().object_count, 2)
        self.store.verify_checkpoint(checkpoint)

    def test_checkpoint_rejects_any_future_dated_event_not_only_the_head(self) -> None:
        self.store.put_bytes(b'future', recorded_at=_T0 + timedelta(seconds=10))
        self.store.register_job(_job(), registered_at=_T0 + timedelta(seconds=1))
        with self.assertRaisesRegex(OperationsIntegrityError, 'predates an event'):
            self.store.checkpoint(created_at=_T0 + timedelta(seconds=2))

    def test_object_and_event_tampering_fail_closed(self) -> None:
        artifact = self.store.put_bytes(b'original', recorded_at=_T0 + timedelta(seconds=1))
        path = self.store.artifact_path(artifact.sha256)
        path.chmod(0o640)
        path.write_bytes(b'tampered')
        with self.assertRaisesRegex(OperationsIntegrityError, 'digest or size mismatch'):
            self.store.verify()

        # Restore object bytes, then simulate a database administrator defeating the
        # SQL immutability trigger.  Canonical event/hash verification must still fail.
        path.write_bytes(b'original')
        path.chmod(0o440)
        connection = sqlite3.connect(self.store.database_path)
        try:
            connection.execute('DROP TRIGGER events_no_update')
            connection.execute("UPDATE events SET occurred_at = '2030-01-01T00:00:00.000000Z' WHERE sequence = 1")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(OperationsIntegrityError, 'indexed columns'):
            self.store.verify()

    def test_materialized_state_tampering_is_detected_from_ledger(self) -> None:
        _job_record, run = self._registered_run()
        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker',
            now=_T0 + timedelta(minutes=1),
        )
        connection = sqlite3.connect(self.store.database_path)
        try:
            connection.execute(
                "UPDATE attempts SET owner_id = 'attacker' WHERE attempt_id = ?",
                (attempt.attempt_id,),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(OperationsIntegrityError, 'lifecycle ledger'):
            self.store.verify()

    def test_job_and_artifact_timestamp_tampering_is_detected_from_ledger(self) -> None:
        job = self.store.register_job(_job(), registered_at=_T0 + timedelta(seconds=1))
        artifact = self.store.put_bytes(b'payload', recorded_at=_T0 + timedelta(seconds=2))
        connection = sqlite3.connect(self.store.database_path)
        try:
            connection.execute('DROP TRIGGER jobs_no_update')
            connection.execute('DROP TRIGGER artifacts_no_update')
            connection.execute(
                "UPDATE jobs SET registered_at = '2030-01-01T00:00:00.000000Z' WHERE spec_sha256 = ?",
                (job.spec_sha256,),
            )
            connection.execute(
                "UPDATE artifacts SET first_recorded_at = '2030-01-01T00:00:00.000000Z' WHERE sha256 = ?",
                (artifact.sha256,),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(OperationsIntegrityError, 'ledger events'):
            self.store.verify()

    def test_initialize_refuses_replacement_and_open_refuses_symlink(self) -> None:
        with self.assertRaisesRegex(OperationsStoreError, 'already exists'):
            OperationalStore.initialize(self.root)
        alternate = self.root.parent / 'alternate'
        alternate.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(OperationsIntegrityError, 'symbolic link'):
            OperationalStore(alternate)

    def test_initialize_and_open_reject_symlinked_cas_parent_component(self) -> None:
        malicious_root = self.root.parent / 'malicious-root'
        malicious_root.mkdir()
        outside = self.root.parent / 'malicious-outside'
        outside.mkdir()
        (malicious_root / 'objects').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(OperationsIntegrityError, 'CAS objects directory'):
            OperationalStore.initialize(malicious_root)
        self.assertEqual(tuple(outside.iterdir()), ())

        safe_root = self.root.parent / 'safe-then-swapped'
        OperationalStore.initialize(safe_root)
        original_objects = safe_root / 'objects-original'
        (safe_root / 'objects').rename(original_objects)
        (safe_root / 'objects').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(OperationsIntegrityError, 'CAS objects directory'):
            OperationalStore(safe_root)

    def test_concurrent_initialize_has_one_winner_without_deleting_it(self) -> None:
        concurrent_root = self.root.parent / 'concurrent-operations'

        def initialize(_ordinal: int) -> str:
            try:
                OperationalStore.initialize(concurrent_root, created_at=_T0, store_id='b' * 32)
                return 'initialized'
            except OperationsStoreError:
                return 'exists'

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = tuple(executor.map(initialize, range(8)))
        self.assertEqual(outcomes.count('initialized'), 1)
        self.assertEqual(outcomes.count('exists'), 7)
        self.assertEqual(OperationalStore(concurrent_root).verify().event_count, 1)

    def test_invalid_owner_is_rejected_before_attempt_state_is_committed(self) -> None:
        _job_record, run = self._registered_run()
        with self.assertRaisesRegex(ValueError, 'owner_id'):
            self.store.begin_attempt(
                run.logical_run_id,
                owner_id='invalid owner with spaces',
                now=_T0 + timedelta(minutes=1),
            )
        self.assertEqual(self.store.list_attempts(logical_run_id=run.logical_run_id), ())
        self.assertEqual(self.store.get_logical_run(run.logical_run_id).state, LogicalRunState.PENDING)
        self.store.verify()


if __name__ == '__main__':
    unittest.main()
