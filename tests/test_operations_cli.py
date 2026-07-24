from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.cli import main as operations_main
from vaxreplay.operations.collector import (
    STATIC_HTTPS_COLLECTOR_ID,
    StaticCollectionError,
    StaticHttpsArtifactSpec,
    StaticHttpsCollectionPlan,
    static_plan_sha256,
)
from vaxreplay.operations.http_capture import (
    HttpRequestHeader,
    HttpsCaptureReceipt,
    HttpsCaptureRequest,
    NormalizedResponseHeader,
    TemporaryHttpsCapture,
)
from vaxreplay.operations.scheduler import ScheduleHistoryGapError, latest_scheduled_slot
from vaxreplay.operations.schema import CaptureJobSpec, checkpoint_sha256
from vaxreplay.operations.store import OperationalStore, OperationsStoreError

_T0 = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def _run_cli(*arguments: str) -> dict[str, object]:
    stdout = io.StringIO()
    with patch.object(sys, 'argv', ['vaxreplay-ops', *arguments]), redirect_stdout(stdout):
        operations_main()
    payload = stdout.getvalue().encode('utf-8')
    assert payload.endswith(b'\n')
    return json.loads(payload)


def _plan() -> StaticHttpsCollectionPlan:
    return StaticHttpsCollectionPlan(
        plan_id='official-index-v1',
        source_id='publisher:example',
        artifacts=(
            StaticHttpsArtifactSpec(
                artifact_id='index',
                request=HttpsCaptureRequest(
                    url='https://public.example.org/index.html',
                    allowed_host='public.example.org',
                    max_body_bytes=1024,
                ),
            ),
        ),
    )


def _fake_capture(request, *, directory, transport, clock):  # noqa: ANN001, ANN202
    del transport, clock
    body = b'exact fictional page'
    descriptor, name = tempfile.mkstemp(prefix='fake-body-', dir=directory)
    os.close(descriptor)
    path = Path(name)
    path.write_bytes(body)
    now = datetime.now(timezone.utc)
    receipt = HttpsCaptureReceipt(
        requested_url=request.url,
        final_url=request.url,
        request_headers=(
            HttpRequestHeader(name='accept-encoding', value='identity'),
            HttpRequestHeader(name='host', value=request.allowed_host),
            HttpRequestHeader(name='user-agent', value='VaxReplay-Archival-Capture/0.1'),
        ),
        status_code=200,
        response_headers=(NormalizedResponseHeader(name='content-length', values=(str(len(body)),)),),
        body_sha256=hashlib.sha256(body).hexdigest(),
        body_byte_count=len(body),
        started_at=now,
        completed_at=now,
    )
    return TemporaryHttpsCapture(path=path, receipt=receipt)


class OperationsCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.root = self.base / 'operations'
        self.plan = _plan()
        self.plan_path = self.base / 'plan.json'
        self.plan_path.write_bytes(canonical_json_bytes(self.plan))
        self.spec = CaptureJobSpec(
            job_id='official-index-hourly',
            collector_id=STATIC_HTTPS_COLLECTOR_ID,
            schedule_anchor_at=_T0,
            schedule_interval_seconds=3600,
            configuration={
                'catch_up_seconds': 7200,
                'collection_plan_sha256': static_plan_sha256(self.plan),
                'dns_resolution_attempts': 1,
                'dns_resolution_timeout_seconds': 10,
                'lease_seconds': 3600,
                'max_dns_addresses': 16,
                'max_attempts_per_slot': 3,
                'max_slots_per_wake': 10,
                'max_total_body_bytes': 16 * 1024 * 1024,
                'plan_deadline_seconds': 300,
                'request_deadline_seconds': 60,
                'source_id': self.plan.source_id,
            },
        )
        self.spec_path = self.base / 'job.json'
        self.spec_path.write_bytes(canonical_json_bytes(self.spec))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _initialize_and_register(self) -> str:
        initialized = _run_cli('init', '--root', str(self.root))
        self.assertFalse(initialized['tier_a_eligible'])
        registered = _run_cli(
            'register-job',
            '--root',
            str(self.root),
            '--spec',
            str(self.spec_path),
        )
        return str(registered['job_spec_sha256'])

    def test_register_due_status_checkpoint_and_verify(self) -> None:
        spec_sha256 = self._initialize_and_register()
        due = _run_cli(
            'register-due',
            '--root',
            str(self.root),
            '--job-spec-sha256',
            spec_sha256,
            '--window-start',
            _T0.isoformat(),
            '--window-end',
            (_T0 + timedelta(hours=2, minutes=59)).isoformat(),
            '--max-slots',
            '10',
        )
        self.assertEqual(due['run_count'], 3)
        status = _run_cli('status', '--root', str(self.root))
        self.assertTrue(status['local_integrity_verified'])
        self.assertTrue(status['successful_run_semantics_verified'])
        self.assertEqual(status['semantic_verifier_scope'], 'all_successful_runs_only')
        self.assertEqual(status['logical_run_states'], {'pending': 3})
        self.assertFalse(status['external_checkpoint_verified'])
        self.assertFalse(status['tier_a_eligible'])

        checkpoint_path = self.base / 'checkpoint.json'
        checkpoint_result = _run_cli(
            'checkpoint',
            '--root',
            str(self.root),
            '--output',
            str(checkpoint_path),
        )
        checkpoint = json.loads(checkpoint_path.read_bytes())
        self.assertEqual(
            checkpoint_result['checkpoint_sha256'],
            hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(checkpoint_result['checkpoint_sha256'], checkpoint_sha256_from_json(checkpoint))
        self.assertTrue(checkpoint_result['successful_run_semantics_verified'])
        verified = _run_cli(
            'verify',
            '--root',
            str(self.root),
            '--checkpoint',
            str(checkpoint_path),
        )
        self.assertTrue(verified['checkpoint_verified'])
        self.assertTrue(verified['successful_run_semantics_verified'])
        with self.assertRaises(FileExistsError):
            _run_cli(
                'checkpoint',
                '--root',
                str(self.root),
                '--output',
                str(checkpoint_path),
            )

    def test_register_job_rejects_extra_immport_config_without_echo_or_persistence(self) -> None:
        _run_cli('init', '--root', str(self.root))
        secret = 'BEARER-CANARY-MUST-NOT-ECHO-OR-PERSIST'
        malicious_spec = {
            'schema_version': 'vaxreplay.operations-job.v0.1',
            'job_id': 'immport-malformed-registration',
            'collector_id': 'immport-secret-broker-collector',
            'schedule_anchor_at': _T0.isoformat().replace('+00:00', 'Z'),
            'schedule_interval_seconds': 3600,
            'configuration': {
                'collection_plan_sha256': 'a' * 64,
                'source_id': 'immport:test',
                'lease_seconds': 600,
                'max_attempts_per_slot': 1,
                'collector_implementation_sha256': 'b' * 64,
                'collector_execution_environment_sha256': 'c' * 64,
                'diagnostic_label': secret,
            },
        }
        malicious_path = self.base / 'malformed-immport-job.json'
        malicious_path.write_bytes(canonical_json_bytes(malicious_spec))
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(
                sys,
                'argv',
                [
                    'vaxreplay-ops',
                    'register-job',
                    '--root',
                    str(self.root),
                    '--spec',
                    str(malicious_path),
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaisesRegex(ValueError, '^job specification is invalid$') as caught,
        ):
            operations_main()

        self.assertNotIn(secret, stdout.getvalue())
        self.assertNotIn(secret, stderr.getvalue())
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, repr(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        store = OperationalStore(self.root)
        self.assertEqual(store.list_jobs(), ())

        direct_spec = CaptureJobSpec.model_validate_json(canonical_json_bytes(malicious_spec))
        with self.assertRaisesRegex(
            OperationsStoreError,
            '^supported collector job configuration is invalid$',
        ) as direct_caught:
            store.register_job(direct_spec, registered_at=_T0)
        self.assertNotIn(secret, str(direct_caught.exception))
        self.assertNotIn(secret, repr(direct_caught.exception))
        self.assertIsNone(direct_caught.exception.__cause__)
        self.assertIsNone(direct_caught.exception.__context__)
        self.assertEqual(store.list_jobs(), ())
        for stored_path in self.root.rglob('*'):
            if stored_path.is_file():
                self.assertNotIn(secret.encode(), stored_path.read_bytes())

    def test_run_static_slot_uses_durable_worker_path(self) -> None:
        spec_sha256 = self._initialize_and_register()
        due = _run_cli(
            'register-due',
            '--root',
            str(self.root),
            '--job-spec-sha256',
            spec_sha256,
            '--window-start',
            _T0.isoformat(),
            '--window-end',
            _T0.isoformat(),
        )
        run_id = str(due['runs'][0]['logical_run_id'])
        with patch('vaxreplay.operations.collector.capture_https_to_tempfile', side_effect=_fake_capture):
            result = _run_cli(
                'run-static-slot',
                '--root',
                str(self.root),
                '--logical-run-id',
                run_id,
                '--plan',
                str(self.plan_path),
                '--owner-id',
                'worker-cli',
            )
        self.assertEqual(result['status'], 'succeeded')
        self.assertTrue(result['plan_complete'])
        self.assertFalse(result['source_enumeration_complete'])
        status = _run_cli('status', '--root', str(self.root))
        self.assertEqual(status['logical_run_states'], {'succeeded': 1})
        self.assertEqual(status['attempt_states'], {'succeeded': 1})

    def test_reconcile_reports_expired_attempts(self) -> None:
        self.spec = self.spec.model_copy(
            update={
                'configuration': {
                    **self.spec.configuration,
                    'dns_resolution_timeout_seconds': 1,
                    'lease_seconds': 1,
                    'plan_deadline_seconds': 1,
                    'request_deadline_seconds': 1,
                }
            }
        )
        self.spec_path.write_bytes(canonical_json_bytes(self.spec))
        spec_sha256 = self._initialize_and_register()
        due = _run_cli(
            'register-due',
            '--root',
            str(self.root),
            '--job-spec-sha256',
            spec_sha256,
            '--window-start',
            _T0.isoformat(),
            '--window-end',
            _T0.isoformat(),
        )
        from vaxreplay.operations.store import OperationalStore

        store = OperationalStore(self.root, trusted_lease_clock=None)
        run_id = str(due['runs'][0]['logical_run_id'])
        plan_artifact = store.put_bytes(canonical_json_bytes(self.plan), recorded_at=_T0)
        store.begin_attempt(
            run_id,
            owner_id='dead-worker',
            now=_T0,
            initial_artifacts={'collection-plan': plan_artifact.sha256},
        )
        with patch('vaxreplay.operations.cli._clock_utc', return_value=_T0 + timedelta(hours=1, seconds=1)):
            reconciled = _run_cli(
                'reconcile',
                '--root',
                str(self.root),
            )
        self.assertEqual(reconciled['abandoned_attempt_count'], 1)
        self.assertEqual(reconciled['attempts'][0]['state'], 'abandoned')

    def test_semantic_verification_rejects_success_from_unregistered_collector(self) -> None:
        from vaxreplay.operations.store import OperationalStore

        store = OperationalStore.initialize(self.root, created_at=_T0, trusted_lease_clock=None)
        job = store.register_job(
            CaptureJobSpec(
                job_id='unsupported-success',
                collector_id='unsupported-collector-v1',
                schedule_anchor_at=_T0,
                schedule_interval_seconds=3600,
                configuration={},
            ),
            registered_at=_T0,
        )
        run = store.register_logical_run(job.spec_sha256, _T0, registered_at=_T0)
        attempt = store.begin_attempt(run.logical_run_id, owner_id='worker', now=_T0)
        manifest = store.put_bytes(b'opaque manifest', recorded_at=_T0 + timedelta(seconds=1))
        store.succeed_attempt(
            attempt.attempt_id,
            owner_id='worker',
            run_manifest_sha256=manifest.sha256,
            now=_T0 + timedelta(seconds=2),
        )

        for command in ('verify', 'status'):
            with (
                self.subTest(command=command),
                self.assertRaisesRegex(StaticCollectionError, 'no semantic verifier is registered'),
            ):
                _run_cli(command, '--root', str(self.root))
        checkpoint_path = self.base / 'unsupported-checkpoint.json'
        with self.assertRaisesRegex(StaticCollectionError, 'no semantic verifier is registered'):
            _run_cli('checkpoint', '--root', str(self.root), '--output', str(checkpoint_path))
        self.assertFalse(checkpoint_path.exists())

    def test_rejects_future_operator_timestamps_without_mutation(self) -> None:
        spec_sha256 = self._initialize_and_register()
        observed_at = datetime.now(timezone.utc)
        future = observed_at + timedelta(microseconds=1)
        commands = (
            (
                'register-due',
                '--root',
                str(self.root),
                '--job-spec-sha256',
                spec_sha256,
                '--window-start',
                _T0.isoformat(),
                '--window-end',
                future.isoformat(),
            ),
            (
                'run-static-due',
                '--root',
                str(self.root),
                '--job-spec-sha256',
                spec_sha256,
                '--plan',
                str(self.plan_path),
                '--owner-id',
                'worker-cli',
                '--through',
                future.isoformat(),
            ),
        )
        with patch('vaxreplay.operations.cli._clock_utc', return_value=observed_at):
            for command in commands:
                with self.subTest(command=command[0]), self.assertRaisesRegex(ValueError, 'future'):
                    _run_cli(*command)
        status = _run_cli('status', '--root', str(self.root))
        self.assertEqual(status['logical_run_states'], {})
        self.assertEqual(status['attempt_states'], {})

    def test_run_static_slot_rejects_future_registered_slot_before_claim(self) -> None:
        spec_sha256 = self._initialize_and_register()
        from vaxreplay.operations.store import OperationalStore

        store = OperationalStore(self.root)
        observed_at = datetime.now(timezone.utc)
        latest = latest_scheduled_slot(self.spec, observed_at)
        future_slot = (
            self.spec.schedule_anchor_at
            if latest is None
            else latest + timedelta(seconds=self.spec.schedule_interval_seconds)
        )
        run = store.register_logical_run(spec_sha256, future_slot, registered_at=observed_at)
        with (
            patch('vaxreplay.operations.cli._clock_utc', return_value=observed_at),
            patch('vaxreplay.operations.cli.run_static_https_collection') as capture,
            self.assertRaisesRegex(ValueError, 'logical run scheduled_for cannot be in the future'),
        ):
            _run_cli(
                'run-static-slot',
                '--root',
                str(self.root),
                '--logical-run-id',
                run.logical_run_id,
                '--plan',
                str(self.plan_path),
                '--owner-id',
                'worker-cli',
            )
        capture.assert_not_called()
        self.assertEqual(store.list_attempts(logical_run_id=run.logical_run_id), ())

    def test_run_static_due_requires_explicit_backfill_of_older_history(self) -> None:
        spec_sha256 = self._initialize_and_register()
        through = _T0 + timedelta(hours=5)
        with self.assertRaises(ScheduleHistoryGapError) as raised:
            _run_cli(
                'run-static-due',
                '--root',
                str(self.root),
                '--job-spec-sha256',
                spec_sha256,
                '--plan',
                str(self.plan_path),
                '--owner-id',
                'worker-cli',
                '--through',
                through.isoformat(),
            )
        self.assertEqual(raised.exception.missing_slot, _T0)
        self.assertIn('register-due', str(raised.exception))
        self.assertEqual(_run_cli('status', '--root', str(self.root))['logical_run_states'], {})

        backfill = _run_cli(
            'register-due',
            '--root',
            str(self.root),
            '--job-spec-sha256',
            spec_sha256,
            '--window-start',
            _T0.isoformat(),
            '--window-end',
            (_T0 + timedelta(hours=2)).isoformat(),
        )
        self.assertEqual(backfill['run_count'], 3)
        with patch('vaxreplay.operations.collector.capture_https_to_tempfile', side_effect=_fake_capture):
            result = _run_cli(
                'run-static-due',
                '--root',
                str(self.root),
                '--job-spec-sha256',
                spec_sha256,
                '--plan',
                str(self.plan_path),
                '--owner-id',
                'worker-cli',
                '--through',
                through.isoformat(),
            )
        self.assertEqual(result['slot_count'], 3)
        self.assertEqual([item['status'] for item in result['results']], ['succeeded'] * 3)


def checkpoint_sha256_from_json(value: object) -> str:
    from vaxreplay.operations.schema import LedgerCheckpoint

    return checkpoint_sha256(LedgerCheckpoint.model_validate_json(canonical_json_bytes(value)))


if __name__ == '__main__':
    unittest.main()
