from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from tests.test_prospective_release import _case_verifier, _source_capture_verifier
from tests.test_prospective_release_seal import (
    _approval_kwargs,
    _build_seal,
    _release_decision_verifier,
    _timestamp_verifier,
)
from tests.test_prospective_run_seal import _RECEIPT_KEY, _Backend, _response, _system
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.prospective_schema import ProspectiveAttemptPolicy
from vaxreplay.runner.backend import IsolationBackend
from vaxreplay.runner.orchestrator import (
    load_run_artifact,
    receipt_hmac_sha256,
    receipt_key_id,
    run_challenge_bundle,
)
from vaxreplay.runner.prospective_attempt_reservation import (
    ProspectiveAttemptCompletionStatus,
    ProspectiveAttemptIntegrityError,
    ProspectiveAttemptRegistryProof,
    ProspectiveAttemptStartProof,
    ProspectiveExplicitFailure,
    build_prospective_attempt_completion,
    build_prospective_attempt_completion_target,
    build_prospective_attempt_reservation,
    build_prospective_attempt_reservation_target,
    build_prospective_attempt_start_authorization,
    build_prospective_attempt_start_target,
    load_prospective_attempt_completion,
    load_prospective_attempt_reservation,
    load_prospective_attempt_start_authorization,
)
from vaxreplay.runner.schema import IsolationTier, RunnerPolicy
from vaxreplay.temporal_schema import TemporalReceiptAuthority

pytestmark = pytest.mark.usefixtures('synthetic_official_replay_patch')


class _Registry:
    """Fictional idempotent global registry with alias and terminal uniqueness."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.completions: dict[str, str] = {}
        self.starts: dict[str, str] = {}

    def __call__(self, proof: ProspectiveAttemptRegistryProof, raw: bytes) -> bool:
        if raw != f'registry:{proof.event_type}:{proof.receipt_id}'.encode():
            return False
        if proof.event_type == 'reservation':
            prior = self.aliases.setdefault(proof.alias_key_sha256, proof.attempt_key_sha256)
            return prior == proof.attempt_key_sha256
        prior = self.completions.setdefault(proof.attempt_key_sha256, proof.target_sha256)
        return prior == proof.target_sha256

    def verify_start(self, proof: ProspectiveAttemptStartProof, raw: bytes) -> bool:
        if raw != f'start:{proof.receipt_id}'.encode():
            return False
        prior = self.starts.setdefault(proof.attempt_key_sha256, proof.target_sha256)
        return prior == proof.target_sha256


def _proof(
    target,
    *,
    event_type: Literal['reservation', 'completion'],
    witnessed_at,
    receipt_id: str,
):
    raw = f'registry:{event_type}:{receipt_id}'.encode()
    target_bytes = canonical_json_bytes(target)
    proof = ProspectiveAttemptRegistryProof(
        event_type=event_type,
        receipt_id=receipt_id,
        authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
        authority_id='fixture-global-attempt-registry',
        target_schema_version=target.schema_version,
        target_sha256=hashlib.sha256(target_bytes).hexdigest(),
        target_bytes=len(target_bytes),
        canonical_cohort_id=target.canonical_cohort_id,
        attempt_key_sha256=target.attempt_key_sha256,
        alias_key_sha256=target.alias_key_sha256,
        witnessed_at=witnessed_at,
        proof_sha256=hashlib.sha256(raw).hexdigest(),
        proof_bytes=len(raw),
        verification_uri=f'https://registry.invalid/{receipt_id}',
    )
    return proof, raw


def _start_proof(target, *, witnessed_at, receipt_id: str):
    raw = f'start:{receipt_id}'.encode()
    target_bytes = canonical_json_bytes(target)
    proof = ProspectiveAttemptStartProof(
        receipt_id=receipt_id,
        authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
        authority_id='fixture-global-start-authority',
        target_sha256=hashlib.sha256(target_bytes).hexdigest(),
        target_bytes=len(target_bytes),
        prospective_release_sha256=target.prospective_release_sha256,
        canonical_cohort_id=target.canonical_cohort_id,
        attempt_key_sha256=target.attempt_key_sha256,
        alias_key_sha256=target.alias_key_sha256,
        witnessed_at=witnessed_at,
        proof_sha256=hashlib.sha256(raw).hexdigest(),
        proof_bytes=len(raw),
        verification_uri=f'https://start-authority.invalid/{receipt_id}',
    )
    return proof, raw


def _context(root: Path):
    release, opening, release_seal = _build_seal(root)
    system = _system()
    policy = RunnerPolicy(required_isolation=IsolationTier.OFFICIAL)
    attempt_policy = ProspectiveAttemptPolicy()
    return release, opening, release_seal, system, policy, attempt_policy


def _reservation(root: Path, registry: _Registry):
    release, opening, release_seal, system, policy, attempt_policy = _context(root)
    target = build_prospective_attempt_reservation_target(
        release=release,
        release_seal=release_seal,
        system=system,
        runner_policy=policy,
        attempt_policy=attempt_policy,
        canonical_cohort_id='registry:influenza-cohort-2026-sh',
        track_id='harness-plus-model',
        registered_entry_id='entry-001',
        decision_receipt_verifier=_release_decision_verifier(release),
        source_capture_verifier=_source_capture_verifier,
        case_universe_seal_verifier=_case_verifier,
        **_approval_kwargs(release),
        release_timestamp_verifier=_timestamp_verifier,
    )
    proof, raw = _proof(
        target,
        event_type='reservation',
        witnessed_at=opening - timedelta(microseconds=1),
        receipt_id='reservation-001',
    )
    reservation = build_prospective_attempt_reservation(
        root / 'reservation',
        release=release,
        release_seal=release_seal,
        system=system,
        runner_policy=policy,
        attempt_policy=attempt_policy,
        canonical_cohort_id=target.canonical_cohort_id,
        track_id=target.track_id,
        registered_entry_id=target.registered_entry_id,
        registry_proof=proof,
        proof_bytes=raw,
        decision_receipt_verifier=_release_decision_verifier(release),
        source_capture_verifier=_source_capture_verifier,
        case_universe_seal_verifier=_case_verifier,
        **_approval_kwargs(release),
        release_timestamp_verifier=_timestamp_verifier,
        registry_verifier=registry,
    )
    return release, opening, release_seal, system, policy, attempt_policy, reservation


def _authorized_attempt(root: Path, registry: _Registry):
    materials = _reservation(root, registry)
    release, opening, release_seal, system, policy, attempt_policy, reservation = materials
    target = build_prospective_attempt_start_target(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        system=system,
        runner_policy=policy,
        attempt_policy=attempt_policy,
        decision_receipt_verifier=_release_decision_verifier(release),
        source_capture_verifier=_source_capture_verifier,
        case_universe_seal_verifier=_case_verifier,
        **_approval_kwargs(release),
        release_timestamp_verifier=_timestamp_verifier,
        registry_verifier=registry,
    )
    proof, raw = _start_proof(
        target,
        witnessed_at=opening,
        receipt_id='start-001',
    )
    authorization = build_prospective_attempt_start_authorization(
        root / 'start-authorization',
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        system=system,
        runner_policy=policy,
        attempt_policy=attempt_policy,
        start_proof=proof,
        proof_bytes=raw,
        decision_receipt_verifier=_release_decision_verifier(release),
        source_capture_verifier=_source_capture_verifier,
        case_universe_seal_verifier=_case_verifier,
        **_approval_kwargs(release),
        release_timestamp_verifier=_timestamp_verifier,
        registry_verifier=registry,
        start_verifier=registry.verify_start,
    )
    return (*materials, authorization)


def _start_kwargs(materials, registry: _Registry):
    release, _opening, release_seal, system, policy, attempt_policy, reservation = materials
    return {
        'release': release,
        'release_seal': release_seal,
        'reservation': reservation,
        'system': system,
        'runner_policy': policy,
        'attempt_policy': attempt_policy,
        'decision_receipt_verifier': _release_decision_verifier(release),
        'source_capture_verifier': _source_capture_verifier,
        'case_universe_seal_verifier': _case_verifier,
        **_approval_kwargs(release),
        'release_timestamp_verifier': _timestamp_verifier,
        'registry_verifier': registry,
        'start_verifier': registry.verify_start,
    }


def _retimed_run(root: Path, *, release, system, policy, started_at):
    raw_run = run_challenge_bundle(
        release.challenge,
        expected_challenge_sha256=release.challenge.manifest_sha256,
        system=system,
        policy=policy,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=receipt_key_id(_RECEIPT_KEY),
        output_dir=root,
        backend=cast(
            IsolationBackend,
            _Backend(_response(release.challenge.suite.episodes[0]).model_dump_json().encode()),
        ),
    )
    receipt = raw_run.receipt.model_copy(
        update={'started_at': started_at, 'finished_at': started_at + timedelta(seconds=1)}
    )
    (raw_run.root / 'run.json').write_bytes(canonical_json_bytes(receipt))
    (raw_run.root / 'run.hmac').write_text(
        receipt_hmac_sha256(receipt, _RECEIPT_KEY) + '\n',
        encoding='ascii',
    )
    return load_run_artifact(
        raw_run.root,
        challenge=release.challenge,
        system=system,
        policy=policy,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=receipt_key_id(_RECEIPT_KEY),
        require_sealed=True,
    )


def _completion_kwargs(materials, registry: _Registry):
    (
        release,
        _opening,
        release_seal,
        system,
        policy,
        attempt_policy,
        reservation,
        start_authorization,
    ) = materials
    return {
        'release': release,
        'release_seal': release_seal,
        'reservation': reservation,
        'start_authorization': start_authorization,
        'system': system,
        'runner_policy': policy,
        'attempt_policy': attempt_policy,
        'receipt_key': _RECEIPT_KEY,
        'expected_receipt_key_id': receipt_key_id(_RECEIPT_KEY),
        'decision_receipt_verifier': _release_decision_verifier(release),
        'source_capture_verifier': _source_capture_verifier,
        'case_universe_seal_verifier': _case_verifier,
        **_approval_kwargs(release),
        'release_timestamp_verifier': _timestamp_verifier,
        'registry_verifier': registry,
        'start_verifier': registry.verify_start,
        'expected_start_authorization_manifest_sha256': start_authorization.manifest_sha256,
    }


class ProspectiveAttemptReservationTest(unittest.TestCase):
    def test_post_publish_reverification_failure_removes_exact_owned_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / 'reservation'

            class RejectingAfterPublishRegistry(_Registry):
                def __call__(self, proof: ProspectiveAttemptRegistryProof, raw: bytes) -> bool:
                    if proof.event_type == 'reservation' and target.exists():
                        return False
                    return super().__call__(proof, raw)

            with self.assertRaisesRegex(ValueError, 'registry verifier rejected'):
                _reservation(root, RejectingAfterPublishRegistry())

            self.assertFalse(target.exists())
            self.assertFalse(tuple(root.glob('.reservation.vaxreplay-private-*')))

    def test_post_publish_replacement_is_left_untouched_and_cleanup_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / 'reservation'
            displaced = root / 'displaced-owned-reservation'
            marker = b'unrelated replacement content'

            class ReplacingRegistry(_Registry):
                def __call__(self, proof: ProspectiveAttemptRegistryProof, raw: bytes) -> bool:
                    if proof.event_type == 'reservation' and target.exists():
                        os.rename(target, displaced)
                        target.mkdir()
                        (target / 'do-not-delete.txt').write_bytes(marker)
                        return False
                    return super().__call__(proof, raw)

            with self.assertRaisesRegex(ValueError, 'cleanup failed closed'):
                _reservation(root, ReplacingRegistry())

            self.assertEqual((target / 'do-not-delete.txt').read_bytes(), marker)
            self.assertTrue((displaced / 'reservation.json').is_file())

    def test_reserves_exact_system_before_opening_and_reverifies_every_trust_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = _Registry()
            materials = _reservation(Path(temporary_directory), registry)
            reservation = materials[-1]

            self.assertEqual(reservation.target.attempt_number, 1)
            self.assertLess(reservation.manifest.registry_proof.witnessed_at, materials[1])
            self.assertEqual(reservation.target.prospective_release_sha256, materials[0].release_sha256)
            self.assertEqual(
                reservation.target.release_seal_target_sha256,
                hashlib.sha256(canonical_json_bytes(materials[2].target)).hexdigest(),
            )
            self.assertEqual(
                materials[2].target.tier_a_release_approval.approval_report_sha256,
                _approval_kwargs(materials[0])['expected_approval_report_sha256'],
            )
            self.assertEqual(
                {path.name for path in reservation.root.iterdir()},
                {'reservation.json', 'target.json', 'registry-proof.bin'},
            )
            loaded = load_prospective_attempt_reservation(
                reservation.root,
                release=materials[0],
                release_seal=materials[2],
                system=materials[3],
                runner_policy=materials[4],
                attempt_policy=materials[5],
                decision_receipt_verifier=_release_decision_verifier(materials[0]),
                source_capture_verifier=_source_capture_verifier,
                case_universe_seal_verifier=_case_verifier,
                **_approval_kwargs(materials[0]),
                release_timestamp_verifier=_timestamp_verifier,
                registry_verifier=registry,
            )
            self.assertEqual(loaded, reservation)

    def test_alias_key_rejects_renamed_entry_or_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            release, _opening, release_seal, system, policy, attempt_policy, first = _reservation(
                root / 'first', registry
            )
            renamed_system = system.model_copy(
                update={
                    'submission_id': 'renamed-submission',
                    'model_id': 'renamed-model-alias',
                    'harness_id': 'renamed-harness-alias',
                }
            )
            target = build_prospective_attempt_reservation_target(
                release=release,
                release_seal=release_seal,
                system=renamed_system,
                runner_policy=policy,
                attempt_policy=attempt_policy,
                canonical_cohort_id=first.target.canonical_cohort_id,
                track_id=first.target.track_id,
                registered_entry_id='renamed-entry',
                decision_receipt_verifier=_release_decision_verifier(release),
                source_capture_verifier=_source_capture_verifier,
                case_universe_seal_verifier=_case_verifier,
                **_approval_kwargs(release),
                release_timestamp_verifier=_timestamp_verifier,
            )
            self.assertEqual(target.alias_key_sha256, first.target.alias_key_sha256)
            self.assertNotEqual(target.attempt_key_sha256, first.target.attempt_key_sha256)
            proof, raw = _proof(
                target,
                event_type='reservation',
                witnessed_at=target.submissions_open_at - timedelta(microseconds=1),
                receipt_id='renamed-reservation',
            )
            with self.assertRaisesRegex(ValueError, 'registry verifier rejected'):
                build_prospective_attempt_reservation(
                    root / 'renamed',
                    release=release,
                    release_seal=release_seal,
                    system=renamed_system,
                    runner_policy=policy,
                    attempt_policy=attempt_policy,
                    canonical_cohort_id=target.canonical_cohort_id,
                    track_id=target.track_id,
                    registered_entry_id=target.registered_entry_id,
                    registry_proof=proof,
                    proof_bytes=raw,
                    decision_receipt_verifier=_release_decision_verifier(release),
                    source_capture_verifier=_source_capture_verifier,
                    case_universe_seal_verifier=_case_verifier,
                    **_approval_kwargs(release),
                    release_timestamp_verifier=_timestamp_verifier,
                    registry_verifier=registry,
                )

    def test_late_or_self_attested_reservation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            release, opening, release_seal, system, policy, attempt_policy = _context(root)
            target = build_prospective_attempt_reservation_target(
                release=release,
                release_seal=release_seal,
                system=system,
                runner_policy=policy,
                attempt_policy=attempt_policy,
                canonical_cohort_id='registry:cohort',
                track_id='track',
                registered_entry_id='entry',
                decision_receipt_verifier=_release_decision_verifier(release),
                source_capture_verifier=_source_capture_verifier,
                case_universe_seal_verifier=_case_verifier,
                **_approval_kwargs(release),
                release_timestamp_verifier=_timestamp_verifier,
            )
            late = None
            for suffix, witnessed_at in (
                ('equal', opening),
                ('late', opening + timedelta(microseconds=1)),
            ):
                late, raw = _proof(
                    target,
                    event_type='reservation',
                    witnessed_at=witnessed_at,
                    receipt_id=suffix,
                )
                with self.assertRaisesRegex(ValueError, 'allowed deadline'):
                    build_prospective_attempt_reservation(
                        root / suffix,
                        release=release,
                        release_seal=release_seal,
                        system=system,
                        runner_policy=policy,
                        attempt_policy=attempt_policy,
                        canonical_cohort_id=target.canonical_cohort_id,
                        track_id=target.track_id,
                        registered_entry_id=target.registered_entry_id,
                        registry_proof=late,
                        proof_bytes=raw,
                        decision_receipt_verifier=_release_decision_verifier(release),
                        source_capture_verifier=_source_capture_verifier,
                        case_universe_seal_verifier=_case_verifier,
                        **_approval_kwargs(release),
                        release_timestamp_verifier=_timestamp_verifier,
                        registry_verifier=_Registry(),
                    )
            assert late is not None
            with self.assertRaisesRegex(ValidationError, 'RFC 3161 or a public transparency log'):
                ProspectiveAttemptRegistryProof.model_validate(
                    {
                        **late.model_dump(),
                        'authority_type': TemporalReceiptAuthority.ORGANIZER_ATTESTATION,
                    }
                )

    def test_start_authorization_requires_the_open_execution_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _reservation(root, registry)
            kwargs = _start_kwargs(materials, registry)
            target_kwargs = {key: value for key, value in kwargs.items() if key != 'start_verifier'}
            target = build_prospective_attempt_start_target(**target_kwargs)
            for suffix, witnessed_at, message in (
                (
                    'pre-open',
                    target.submissions_open_at - timedelta(microseconds=1),
                    'predates submissions opening',
                ),
                ('at-deadline', target.run_deadline_at, 'at or after the run deadline'),
            ):
                proof, raw = _start_proof(target, witnessed_at=witnessed_at, receipt_id=suffix)
                with self.assertRaisesRegex(ValueError, message):
                    build_prospective_attempt_start_authorization(
                        root / suffix,
                        **kwargs,
                        start_proof=proof,
                        proof_bytes=raw,
                    )

    def test_start_authorization_tampering_and_mismatched_loaded_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            authorization = materials[-1]
            (authorization.root / 'target.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ProspectiveAttemptIntegrityError, 'attempt start target'):
                load_prospective_attempt_start_authorization(
                    authorization.root,
                    **_start_kwargs(materials[:-1], registry),
                    expected_start_authorization_manifest_sha256=authorization.manifest_sha256,
                )

    def test_completion_rejects_terminal_start_before_external_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            reservation_materials = _reservation(root, registry)
            kwargs = _start_kwargs(reservation_materials, registry)
            target_kwargs = {key: value for key, value in kwargs.items() if key != 'start_verifier'}
            start_target = build_prospective_attempt_start_target(**target_kwargs)
            proof, raw = _start_proof(
                start_target,
                witnessed_at=start_target.submissions_open_at + timedelta(seconds=5),
                receipt_id='delayed-start',
            )
            authorization = build_prospective_attempt_start_authorization(
                root / 'delayed-start',
                **kwargs,
                start_proof=proof,
                proof_bytes=raw,
            )
            materials = (*reservation_materials, authorization)
            run = _retimed_run(
                root / 'early-run',
                release=materials[0],
                system=materials[3],
                policy=materials[4],
                started_at=start_target.submissions_open_at + timedelta(seconds=1),
            )
            with self.assertRaisesRegex(ValueError, 'before its external start authorization'):
                build_prospective_attempt_completion_target(
                    **_completion_kwargs(materials, registry),
                    run=run,
                )

            record = b'backend failed before the externally authorized start'
            failure = ProspectiveExplicitFailure(
                failure_code='backend_start_failure',
                backend_id='fixture-official-backend',
                started_at=start_target.submissions_open_at + timedelta(seconds=1),
                failed_at=start_target.submissions_open_at + timedelta(seconds=2),
                failure_record_sha256=hashlib.sha256(record).hexdigest(),
                failure_record_bytes=len(record),
            )
            with self.assertRaisesRegex(ValueError, 'before its external start authorization'):
                build_prospective_attempt_completion_target(
                    **_completion_kwargs(materials, registry),
                    failure=failure,
                    failure_record=record,
                )

    def test_success_completion_binds_exact_run_and_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            run = _retimed_run(
                root / 'run',
                release=materials[0],
                system=materials[3],
                policy=materials[4],
                started_at=materials[-1].manifest.start_proof.witnessed_at + timedelta(seconds=1),
            )
            completion_target = build_prospective_attempt_completion_target(
                **_completion_kwargs(materials, registry),
                run=run,
            )
            proof, raw = _proof(
                completion_target,
                event_type='completion',
                witnessed_at=run.receipt.finished_at + timedelta(seconds=1),
                receipt_id='completion-success',
            )
            completion = build_prospective_attempt_completion(
                root / 'completion',
                **_completion_kwargs(materials, registry),
                registry_proof=proof,
                proof_bytes=raw,
                run=run,
            )
            self.assertEqual(completion.target.status, ProspectiveAttemptCompletionStatus.SUCCESS)
            self.assertEqual(
                completion.target.reservation_target_sha256,
                hashlib.sha256(canonical_json_bytes(materials[-2].target)).hexdigest(),
            )
            self.assertEqual(
                completion.target.start_authorization_manifest_sha256,
                materials[-1].manifest_sha256,
            )
            assert completion.target.run is not None
            self.assertEqual(completion.target.run.responses_sha256, run.receipt.responses_sha256)
            self.assertEqual(completion.run, run)
            loaded = load_prospective_attempt_completion(
                completion.root,
                **_completion_kwargs(materials, registry),
                run=run,
            )
            self.assertEqual(loaded, completion)

    def test_retained_failure_prevents_successful_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            start_authorization = materials[-1]
            record = b'backend failed before an episode response was produced'
            failure = ProspectiveExplicitFailure(
                failure_code='backend_start_failure',
                backend_id='fixture-official-backend',
                started_at=start_authorization.manifest.start_proof.witnessed_at + timedelta(seconds=1),
                failed_at=start_authorization.manifest.start_proof.witnessed_at + timedelta(seconds=2),
                failure_record_sha256=hashlib.sha256(record).hexdigest(),
                failure_record_bytes=len(record),
            )
            failure_target = build_prospective_attempt_completion_target(
                **_completion_kwargs(materials, registry),
                failure=failure,
                failure_record=record,
            )
            proof, raw = _proof(
                failure_target,
                event_type='completion',
                witnessed_at=failure.failed_at,
                receipt_id='completion-failure',
            )
            completion = build_prospective_attempt_completion(
                root / 'failed',
                **_completion_kwargs(materials, registry),
                registry_proof=proof,
                proof_bytes=raw,
                failure=failure,
                failure_record=record,
            )
            self.assertEqual(completion.target.status, ProspectiveAttemptCompletionStatus.FAILURE)

            run = _retimed_run(
                root / 'retry-run',
                release=materials[0],
                system=materials[3],
                policy=materials[4],
                started_at=failure.failed_at + timedelta(seconds=1),
            )
            retry_target = build_prospective_attempt_completion_target(
                **_completion_kwargs(materials, registry),
                run=run,
            )
            retry_proof, retry_raw = _proof(
                retry_target,
                event_type='completion',
                witnessed_at=run.receipt.finished_at,
                receipt_id='completion-retry',
            )
            with self.assertRaisesRegex(ValueError, 'registry verifier rejected'):
                build_prospective_attempt_completion(
                    root / 'retry',
                    **_completion_kwargs(materials, registry),
                    registry_proof=retry_proof,
                    proof_bytes=retry_raw,
                    run=run,
                )

    def test_completion_tampering_and_extra_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = _Registry()
            materials = _authorized_attempt(root, registry)
            run = _retimed_run(
                root / 'run',
                release=materials[0],
                system=materials[3],
                policy=materials[4],
                started_at=materials[-1].manifest.start_proof.witnessed_at + timedelta(seconds=1),
            )
            target = build_prospective_attempt_completion_target(
                **_completion_kwargs(materials, registry),
                run=run,
            )
            proof, raw = _proof(
                target,
                event_type='completion',
                witnessed_at=run.receipt.finished_at,
                receipt_id='completion',
            )
            completion = build_prospective_attempt_completion(
                root / 'completion',
                **_completion_kwargs(materials, registry),
                registry_proof=proof,
                proof_bytes=raw,
                run=run,
            )
            (completion.root / 'unbound.txt').write_bytes(b'unbound')
            with self.assertRaisesRegex(ProspectiveAttemptIntegrityError, 'allowlist mismatch'):
                load_prospective_attempt_completion(
                    completion.root,
                    **_completion_kwargs(materials, registry),
                    run=run,
                )


if __name__ == '__main__':
    unittest.main()
