from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.test_prospective_attempt_reservation import (
    _authorized_attempt,
    _completion_kwargs,
    _proof,
    _Registry,
)
from tests.test_prospective_release import (
    _case_verifier,
    _source_capture_verifier,
)
from tests.test_prospective_release_seal import (
    _approval_kwargs,
    _release_decision_verifier,
    _timestamp_verifier,
)
from tests.test_prospective_run_seal import _RECEIPT_KEY, _Backend, _response
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.runner.orchestrator import receipt_key_id
from vaxreplay.runner.prospective_attempt_reservation import (
    ProspectiveAttemptCompletionStatus,
    ProspectiveAttemptStartProof,
    ProspectiveAttemptStartTarget,
    build_prospective_attempt_completion,
    prospective_attempt_completion_target_sha256,
    prospective_attempt_start_authorization_manifest_sha256,
)
from vaxreplay.runner.prospective_execution import (
    ProspectiveAttemptExecutionError,
    ProspectiveAttemptExecutionFailureRecord,
    ProspectiveAttemptFailureHandoff,
    ProspectiveAttemptStartConsumer,
    ProspectiveConsumedStartFatalError,
    run_reserved_prospective_attempt,
)
from vaxreplay.runner.schema import IsolationTier

pytestmark = pytest.mark.usefixtures('synthetic_official_replay_patch')


class _Clock(datetime):
    instant: datetime

    @classmethod
    def now(cls, tz=None):  # noqa: ANN001
        value = cls.instant
        return value if tz is None else value.astimezone(tz)


class _TrackingBackend(_Backend):
    def __init__(self, stdout: bytes, tier: IsolationTier = IsolationTier.OFFICIAL) -> None:
        super().__init__(stdout, tier)
        self.prepare_calls = 0

    def prepare(self, system, policy):  # noqa: ANN001
        self.prepare_calls += 1
        return super().prepare(system, policy)


class _ReleaseMutatingBackend(_TrackingBackend):
    def __init__(self, stdout: bytes, release_manifest: Path) -> None:
        super().__init__(stdout)
        self.release_manifest = release_manifest

    def prepare(self, system, policy):  # noqa: ANN001
        prepared = super().prepare(system, policy)
        self.release_manifest.write_bytes(self.release_manifest.read_bytes() + b'\n')
        return prepared


class _StartConsumer:
    """Fictional atomic first-write-wins boundary, separate from proof verification."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.consumed: dict[str, str] = {}
        self.calls = 0

    def consume_start(
        self,
        *,
        start_authorization_manifest_sha256: str,
        target: ProspectiveAttemptStartTarget,
        target_bytes: bytes,
        proof: ProspectiveAttemptStartProof,
        proof_bytes: bytes,
    ) -> bool:
        self.calls += 1
        if target_bytes != canonical_json_bytes(target):
            return False
        if proof.target_sha256 != hashlib.sha256(target_bytes).hexdigest():
            return False
        if proof_bytes != f'start:{proof.receipt_id}'.encode():
            return False
        with self._lock:
            if target.attempt_key_sha256 in self.consumed:
                return False
            self.consumed[target.attempt_key_sha256] = start_authorization_manifest_sha256
            return True


class _PrepareRaisingBackend(_TrackingBackend):
    def prepare(self, system, policy):  # noqa: ANN001
        self.prepare_calls += 1
        raise RuntimeError('fixture backend prepare exploded')


def _execution_kwargs(
    root: Path,
    materials,  # noqa: ANN001
    registry: _Registry,
    backend,  # noqa: ANN001
    *,
    start_consumer: ProspectiveAttemptStartConsumer | None = None,
):
    (
        release,
        opening,
        release_seal,
        system,
        runner_policy,
        attempt_policy,
        reservation,
        start_authorization,
    ) = materials
    return {
        'release_root': release.root,
        'expected_release_sha256': release.release_sha256,
        'release_seal_root': release_seal.root,
        'expected_release_seal_manifest_sha256': release_seal.manifest_sha256,
        'submissions_open_at': opening,
        'reservation_root': reservation.root,
        'expected_reservation_manifest_sha256': reservation.manifest_sha256,
        'start_authorization_root': start_authorization.root,
        'expected_start_authorization_manifest_sha256': start_authorization.manifest_sha256,
        'system': system,
        'runner_policy': runner_policy,
        'attempt_policy': attempt_policy,
        'receipt_key': _RECEIPT_KEY,
        'expected_receipt_key_id': receipt_key_id(_RECEIPT_KEY),
        'output_dir': root / 'run',
        'backend': backend,
        'decision_receipt_verifier': _release_decision_verifier(release),
        'case_universe_seal_verifier': _case_verifier,
        'source_capture_verifier': _source_capture_verifier,
        **_approval_kwargs(release),
        'release_timestamp_verifier': _timestamp_verifier,
        'registry_verifier': registry,
        'start_verifier': registry.verify_start,
        'start_consumer': start_consumer or _StartConsumer(),
    }


class ProspectiveExecutionTest(unittest.TestCase):
    def test_runs_reverified_reservation_and_returns_registry_ready_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            release = materials[0]
            response = _response(release.challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _TrackingBackend(response)
            _Clock.instant = materials[-1].manifest.start_proof.witnessed_at + timedelta(seconds=1)

            with patch('vaxreplay.runner.orchestrator.datetime', _Clock):
                handoff = run_reserved_prospective_attempt(**_execution_kwargs(root, materials, registry, backend))

            self.assertEqual(backend.prepare_calls, 1)
            self.assertEqual(handoff.release, release)
            self.assertEqual(handoff.release_seal, materials[2])
            self.assertEqual(handoff.reservation, materials[-2])
            self.assertEqual(handoff.start_authorization, materials[-1])
            self.assertTrue(handoff.run.receipt.sealed)
            self.assertEqual(handoff.completion_target_bytes, canonical_json_bytes(handoff.completion_target))
            self.assertEqual(
                handoff.completion_target_sha256,
                prospective_attempt_completion_target_sha256(handoff.completion_target),
            )
            self.assertIsNotNone(handoff.completion_target.run)
            assert handoff.completion_target.run is not None
            self.assertEqual(handoff.completion_target.run.run_receipt_sha256, handoff.run.receipt_sha256)

            proof, proof_bytes = _proof(
                handoff.completion_target,
                event_type='completion',
                witnessed_at=handoff.completion_target.terminal_at + timedelta(microseconds=1),
                receipt_id='execution-handoff-completion',
            )
            completion = build_prospective_attempt_completion(
                root / 'completion',
                **_completion_kwargs(materials, registry),
                registry_proof=proof,
                proof_bytes=proof_bytes,
                run=handoff.run,
            )
            self.assertEqual(completion.target, handoff.completion_target)

    def test_stateful_start_consumption_rejects_second_execution_before_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _TrackingBackend(response)
            consumer = _StartConsumer()
            kwargs = _execution_kwargs(
                root,
                materials,
                registry,
                backend,
                start_consumer=consumer,
            )
            _Clock.instant = materials[-1].manifest.start_proof.witnessed_at + timedelta(seconds=1)

            with patch('vaxreplay.runner.orchestrator.datetime', _Clock):
                first = run_reserved_prospective_attempt(**kwargs)
                with self.assertRaisesRegex(
                    ProspectiveConsumedStartFatalError,
                    'already consumed',
                ) as caught:
                    run_reserved_prospective_attempt(**kwargs)

            self.assertFalse(caught.exception.retry_allowed)
            self.assertEqual(
                caught.exception.attempt_key_sha256,
                materials[-2].target.attempt_key_sha256,
            )
            self.assertEqual(consumer.calls, 2)
            self.assertEqual(backend.prepare_calls, 1)
            self.assertNotIsInstance(first, ProspectiveAttemptFailureHandoff)

    def test_rejects_every_out_of_band_identity_before_backend_prepare(self) -> None:
        for field in (
            'expected_release_sha256',
            'expected_release_seal_manifest_sha256',
            'expected_approval_report_sha256',
            'expected_reservation_manifest_sha256',
            'expected_start_authorization_manifest_sha256',
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                registry = _Registry()
                materials = _authorized_attempt(root, registry)
                response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
                backend = _TrackingBackend(response)
                consumer = _StartConsumer()
                kwargs = _execution_kwargs(
                    root,
                    materials,
                    registry,
                    backend,
                    start_consumer=consumer,
                )
                kwargs[field] = '0' * 64

                with self.assertRaises(ProspectiveAttemptExecutionError):
                    run_reserved_prospective_attempt(**kwargs)
                self.assertEqual(backend.prepare_calls, 0)
                self.assertEqual(consumer.calls, 0)
                self.assertFalse((root / 'run').exists())

    def test_rejects_authority_verifier_failure_before_backend_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _TrackingBackend(response)
            kwargs = _execution_kwargs(root, materials, registry, backend)
            kwargs['decision_receipt_verifier'] = lambda _receipt, _proof_bytes: False

            with self.assertRaisesRegex(ProspectiveAttemptExecutionError, 'release context verification failed'):
                run_reserved_prospective_attempt(**kwargs)
            self.assertEqual(backend.prepare_calls, 0)

    def test_pre_open_start_authorization_is_rejected_before_backend_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            authorization = materials[-1]
            response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _TrackingBackend(response)
            pre_open_proof = authorization.manifest.start_proof.model_copy(
                update={'witnessed_at': materials[1] - timedelta(microseconds=1)}
            )
            pre_open_manifest = authorization.manifest.model_copy(update={'start_proof': pre_open_proof})
            (authorization.root / 'start-authorization.json').write_bytes(canonical_json_bytes(pre_open_manifest))
            kwargs = _execution_kwargs(root, materials, registry, backend)
            kwargs['expected_start_authorization_manifest_sha256'] = (
                prospective_attempt_start_authorization_manifest_sha256(pre_open_manifest)
            )

            with self.assertRaisesRegex(ProspectiveAttemptExecutionError, 'start-authorization verification failed'):
                run_reserved_prospective_attempt(**kwargs)
            self.assertEqual(backend.prepare_calls, 0)
            self.assertFalse((root / 'run').exists())

    def test_backend_capability_claims_are_checked_by_the_reserved_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _TrackingBackend(response, IsolationTier.DEVELOPMENT)
            _Clock.instant = materials[-1].manifest.start_proof.witnessed_at + timedelta(seconds=1)

            with (
                patch('vaxreplay.runner.orchestrator.datetime', _Clock),
                patch('vaxreplay.runner.prospective_execution.datetime', _Clock),
            ):
                handoff = run_reserved_prospective_attempt(**_execution_kwargs(root, materials, registry, backend))
            self.assertIsInstance(handoff, ProspectiveAttemptFailureHandoff)
            assert isinstance(handoff, ProspectiveAttemptFailureHandoff)
            self.assertFalse(handoff.retry_allowed)
            self.assertEqual(
                handoff.completion_target.status,
                ProspectiveAttemptCompletionStatus.FAILURE,
            )
            self.assertEqual(handoff.failure.backend_id, 'fictional-official-backend')
            self.assertEqual(handoff.failure_record_bytes, canonical_json_bytes(handoff.failure_record))
            self.assertEqual(
                handoff.completion_target_sha256,
                prospective_attempt_completion_target_sha256(handoff.completion_target),
            )
            self.assertEqual(backend.prepare_calls, 1)
            self.assertFalse((root / 'run').exists())

    def test_prepare_exception_returns_registry_ready_non_retryable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _PrepareRaisingBackend(response)
            _Clock.instant = materials[-1].manifest.start_proof.witnessed_at + timedelta(seconds=1)

            with patch('vaxreplay.runner.prospective_execution.datetime', _Clock):
                handoff = run_reserved_prospective_attempt(**_execution_kwargs(root, materials, registry, backend))

            self.assertIsInstance(handoff, ProspectiveAttemptFailureHandoff)
            assert isinstance(handoff, ProspectiveAttemptFailureHandoff)
            self.assertFalse(handoff.retry_allowed)
            self.assertIsInstance(handoff.failure_record, ProspectiveAttemptExecutionFailureRecord)
            self.assertIn('fixture backend prepare exploded', handoff.failure_record.exception_message)
            proof, proof_bytes = _proof(
                handoff.completion_target,
                event_type='completion',
                witnessed_at=handoff.completion_target.terminal_at + timedelta(microseconds=1),
                receipt_id='execution-failure-completion',
            )
            completion = build_prospective_attempt_completion(
                root / 'failed-completion',
                **_completion_kwargs(materials, registry),
                registry_proof=proof,
                proof_bytes=proof_bytes,
                failure=handoff.failure,
                failure_record=handoff.failure_record_bytes,
            )
            self.assertEqual(completion.target, handoff.completion_target)
            self.assertEqual(backend.prepare_calls, 1)

    def test_launcher_clock_behind_start_witness_still_returns_valid_failure_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _PrepareRaisingBackend(response)
            witnessed_at = materials[-1].manifest.start_proof.witnessed_at
            _Clock.instant = witnessed_at - timedelta(seconds=30)

            with patch('vaxreplay.runner.prospective_execution.datetime', _Clock):
                handoff = run_reserved_prospective_attempt(**_execution_kwargs(root, materials, registry, backend))

            self.assertIsInstance(handoff, ProspectiveAttemptFailureHandoff)
            assert isinstance(handoff, ProspectiveAttemptFailureHandoff)
            self.assertEqual(handoff.failure.started_at, witnessed_at)
            self.assertEqual(handoff.completion_target.failure, handoff.failure)
            self.assertEqual(backend.prepare_calls, 1)

    def test_post_execution_release_mutation_blocks_handoff_and_retains_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _ReleaseMutatingBackend(response, materials[0].root / 'release.json')
            _Clock.instant = materials[-1].manifest.start_proof.witnessed_at + timedelta(seconds=1)

            with (
                patch('vaxreplay.runner.orchestrator.datetime', _Clock),
                self.assertRaisesRegex(ProspectiveConsumedStartFatalError, 'retain it and do not retry') as caught,
            ):
                run_reserved_prospective_attempt(**_execution_kwargs(root, materials, registry, backend))
            self.assertFalse(caught.exception.retry_allowed)
            self.assertEqual(caught.exception.run_receipt_sha256 is not None, True)
            self.assertEqual(backend.prepare_calls, 1)
            self.assertTrue((root / 'run' / 'run.json').is_file())

    def test_rejects_malformed_out_of_band_digest_without_loading_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            response = _response(materials[0].challenge.suite.episodes[0]).model_dump_json().encode()
            backend = _TrackingBackend(response)
            kwargs = _execution_kwargs(root, materials, registry, backend)
            kwargs['expected_release_sha256'] = 'NOT-A-DIGEST'

            with self.assertRaisesRegex(ProspectiveAttemptExecutionError, 'exact lowercase SHA-256'):
                run_reserved_prospective_attempt(**kwargs)
            self.assertEqual(backend.prepare_calls, 0)


if __name__ == '__main__':
    unittest.main()
