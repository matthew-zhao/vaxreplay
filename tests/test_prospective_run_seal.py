from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_schema import CandidateForecast, Split
from vaxreplay.prospective import (
    SourceCaptureArtifact,
    build_prospective_decision_package,
    build_prospective_decision_seal,
)
from vaxreplay.prospective_schema import (
    PROSPECTIVE_RESPONSE_PROTOCOL,
    ProspectiveChallengeAdmission,
    ProspectiveSubmission,
    ProspectiveSuiteManifest,
    prospective_suite_manifest_sha256,
)
from vaxreplay.runner.backend import PreparedBackend, RawExecutionResult, RawExecutionStatus
from vaxreplay.runner.orchestrator import receipt_key_id, run_challenge_bundle
from vaxreplay.runner.prospective_challenge import (
    build_prospective_challenge_bundle,
    load_prospective_challenge_bundle,
)
from vaxreplay.runner.prospective_run_seal import (
    ProspectiveAttemptPolicy,
    ProspectiveRunSealIntegrityError,
    ProspectiveRunTimestampProof,
    build_prospective_run_seal,
    build_prospective_run_seal_target,
    load_prospective_run_seal,
    prospective_attempt_policy_sha256,
    prospective_run_seal_target_sha256,
)
from vaxreplay.runner.schema import (
    BackendCapabilities,
    EpisodeRunStatus,
    IsolationTier,
    RunnerPolicy,
    SystemSubmissionManifest,
)
from vaxreplay.temporal_schema import (
    DecisionTimeConfig,
    TemporalArtifactReceipt,
    TemporalArtifactRole,
    TemporalReceiptAuthority,
)


def _fixture() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


_RECEIPT_KEY = bytes(range(32))


def _decision_verifier(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
    return receipt.authority_id == 'fictional-public-log' and proof.startswith(b'decision-proof:')


def _run_verifier(proof: ProspectiveRunTimestampProof, raw: bytes) -> bool:
    return proof.authority_id == 'fictional-public-log' and raw.startswith(b'run-proof:')


def _system() -> SystemSubmissionManifest:
    return SystemSubmissionManifest(
        submission_id='prospective-system-1',
        image_ref='sha256:' + 'a' * 64,
        entrypoint=('/opt/vaxreplay/run',),
        model_id='fictional-model',
        harness_id='fictional-harness',
        response_protocol=PROSPECTIVE_RESPONSE_PROTOCOL,
    )


def _capabilities(tier: IsolationTier) -> BackendCapabilities:
    return BackendCapabilities(
        backend_id='fictional-official-backend',
        backend_version='1',
        isolation_tier=tier,
        network_isolation=True,
        host_filesystem_isolation=True,
        read_only_root=True,
        non_root_user=True,
        capability_drop=True,
        no_new_privileges=True,
        process_limit=True,
        memory_limit=True,
        cpu_limit=True,
        scratch_limit=True,
        fresh_worker_per_episode=True,
    )


class _Backend:
    def __init__(self, stdout: bytes, tier: IsolationTier = IsolationTier.OFFICIAL):
        self.stdout = stdout
        self.tier = tier

    def prepare(self, _system, _policy) -> PreparedBackend:
        return PreparedBackend(
            capabilities=_capabilities(self.tier),
            resolved_image_id='sha256:' + 'b' * 64,
        )

    def run(self, **_kwargs) -> RawExecutionResult:
        return RawExecutionResult(
            status=RawExecutionStatus.EXITED,
            exit_code=0,
            duration_ms=2,
            stdout=self.stdout,
            stderr=b'',
            stdout_truncated=False,
            stderr_truncated=False,
        )


def _response(binding) -> ProspectiveSubmission:
    config = binding.decision_snapshot.config
    return ProspectiveSubmission(
        episode_id=binding.episode_id,
        decision_snapshot_sha256=binding.decision_snapshot_sha256,
        ranking=tuple(config.candidate_ids),
        forecasts=tuple(
            CandidateForecast(
                candidate_id=candidate_id,
                target_id=target.target_id,
                horizon_days=target.horizon_days,
                probability=0.5,
            )
            for candidate_id in config.candidate_ids
            for target in config.forecast_targets
        ),
    )


def _build_materials(
    root: Path,
    *,
    late_deadline: bool = False,
    invalid_first_response: bool = False,
    run_tier: IsolationTier = IsolationTier.OFFICIAL,
):
    bundle = EpisodeBundle.load(_fixture())
    now = datetime.now(UTC)
    decision_at = now - timedelta(hours=4 if late_deadline else 2)
    config = DecisionTimeConfig.from_manifest(
        bundle.manifest.model_copy(
            update={
                'synthetic': False,
                'split': Split.TEST,
                'decision_at': decision_at,
            }
        )
    )
    package = build_prospective_decision_package(
        root / 'package',
        config=config,
        candidates=bundle.candidates,
        evidence=bundle.visible_evidence,
        protocol_artifacts={
            'candidate_set_definition': b'fixed panel inclusion, exclusion, and ordering rules',
            'evidence_acquisition_spec': b'fixed source releases and evidence availability rules',
            'outcome_adjudication_spec': b'fixed endpoint, horizon, censoring, and derivation rules',
        },
        candidate_set_available_at=decision_at - timedelta(days=2),
        source_captures=(
            SourceCaptureArtifact(
                source_id='fictional-source',
                source_release_at=decision_at - timedelta(days=3),
                captured_at=decision_at - timedelta(days=2),
                witnessed_at=decision_at - timedelta(days=1),
                manifest_bytes=canonical_json_bytes(
                    {
                        'schema_version': 'fictional.capture.v1',
                        'source_id': 'fictional-source',
                        'records_sha256': 'c' * 64,
                    }
                ),
            ),
        ),
    )
    receipt_times = {
        TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: (
            package.manifest.episode.decision_snapshot.protocol_commitments.candidate_set_available_at
        ),
        TemporalArtifactRole.EVIDENCE_SNAPSHOT: (package.manifest.episode.decision_snapshot.latest_visible_evidence_at),
        TemporalArtifactRole.DECISION_SNAPSHOT: decision_at,
    }
    receipts = []
    proof_artifacts = {}
    for ordinal, request in enumerate(package.receipt_requests):
        receipt_id = f'decision-receipt-{ordinal}'
        proof = f'decision-proof:{ordinal}'.encode()
        proof_artifacts[receipt_id] = proof
        receipts.append(
            TemporalArtifactReceipt(
                receipt_id=receipt_id,
                role=request.role,
                artifact_schema_version=request.artifact_schema_version,
                artifact_sha256=request.artifact_sha256,
                artifact_bytes=request.artifact_bytes,
                witnessed_at=receipt_times[request.role],
                authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
                authority_id='fictional-public-log',
                receipt_sha256=hashlib.sha256(proof).hexdigest(),
                receipt_bytes=len(proof),
                verification_uri=f'https://log.invalid/{receipt_id}',
            )
        )
    decision_seal = build_prospective_decision_seal(
        root / 'decision-seal',
        package=package,
        receipts=tuple(receipts),
        proof_artifacts=proof_artifacts,
        receipt_verifier=_decision_verifier,
        verified_at=decision_at + timedelta(seconds=1),
    )
    binding = package.manifest.episode
    suite = ProspectiveSuiteManifest(
        suite_id='prospective-suite-1',
        task_type=binding.task_type,
        reward_version=binding.reward_version,
        split=binding.split,
        episodes=(binding,),
    )
    attempt_policy = ProspectiveAttemptPolicy()
    deadline = decision_at + timedelta(hours=1 if late_deadline else 24)
    admission = ProspectiveChallengeAdmission(
        release_id='prospective-release-1',
        suite_sha256=prospective_suite_manifest_sha256(suite),
        split_inventory_sha256='1' * 64,
        case_universe_sha256='2' * 64,
        verifier_policy_sha256='3' * 64,
        source_capture_policy_sha256='5' * 64,
        eligibility_protocol_sha256='4' * 64,
        attempt_policy_sha256=prospective_attempt_policy_sha256(attempt_policy),
        run_deadline_at=deadline,
        episodes=(binding,),
    )
    built_challenge = build_prospective_challenge_bundle(
        root / 'challenge',
        challenge_id='prospective-challenge-1',
        suite_id=suite.suite_id,
        packages=(package,),
        seals=(decision_seal,),
        admission=admission,
    )
    challenge = load_prospective_challenge_bundle(
        built_challenge.root,
        receipt_verifier=_decision_verifier,
    )
    system = _system()
    policy = RunnerPolicy(required_isolation=run_tier)
    stdout = b'not-json' if invalid_first_response else _response(binding).model_dump_json().encode()
    run = run_challenge_bundle(
        challenge,
        expected_challenge_sha256=challenge.manifest_sha256,
        system=system,
        policy=policy,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=receipt_key_id(_RECEIPT_KEY),
        output_dir=root / 'run',
        backend=_Backend(stdout, run_tier),
    )
    return challenge, run, admission, system, policy, attempt_policy


def _timestamp_material(target, *, witnessed_at: datetime | None = None, suffix: str = 'first'):
    raw = f'run-proof:{suffix}'.encode()
    target_bytes = canonical_json_bytes(target)
    proof = ProspectiveRunTimestampProof(
        receipt_id=f'run-receipt-{suffix}',
        authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
        authority_id='fictional-public-log',
        target_sha256=prospective_run_seal_target_sha256(target),
        target_bytes=len(target_bytes),
        attempt_key_sha256=target.attempt_key_sha256,
        witnessed_at=witnessed_at or target.run_finished_at + timedelta(microseconds=1),
        proof_sha256=hashlib.sha256(raw).hexdigest(),
        proof_bytes=len(raw),
        verification_uri=f'https://log.invalid/run/{suffix}',
    )
    return proof, raw


def _load_kwargs(materials):
    challenge, run, admission, system, policy, attempt_policy = materials
    return {
        'challenge': challenge,
        'run': run,
        'admission': admission,
        'system': system,
        'policy': policy,
        'attempt_policy': attempt_policy,
        'timestamp_verifier': _run_verifier,
    }


class ProspectiveRunSealTest(unittest.TestCase):
    def test_builds_canonical_external_seal_over_every_run_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            materials = _build_materials(root)
            target = build_prospective_run_seal_target(
                **{k: v for k, v in _load_kwargs(materials).items() if k != 'timestamp_verifier'}
            )
            proof, raw = _timestamp_material(target)
            seal = build_prospective_run_seal(
                root / 'run-seal',
                **_load_kwargs(materials),
                timestamp_proof=proof,
                proof_bytes=raw,
            )

            self.assertEqual(seal.target, target)
            self.assertEqual(seal.proof_bytes, raw)
            self.assertEqual(seal.target.attempt_number, 1)
            self.assertEqual(seal.target.release_id, materials[2].release_id)
            self.assertEqual(seal.target.run_receipt_sha256, materials[1].receipt_sha256)
            self.assertEqual(seal.target.responses_sha256, materials[1].receipt.responses_sha256)
            self.assertEqual(
                {path.name for path in seal.root.iterdir()},
                {'seal.json', 'target.json', 'timestamp-proof.bin'},
            )
            reloaded = load_prospective_run_seal(seal.root, **_load_kwargs(materials))
            self.assertEqual(reloaded, seal)

    def test_organizer_hmac_is_not_an_external_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            materials = _build_materials(root)
            target = build_prospective_run_seal_target(
                **{k: v for k, v in _load_kwargs(materials).items() if k != 'timestamp_verifier'}
            )
            proof, raw = _timestamp_material(target)
            self.assertTrue((materials[1].root / 'run.hmac').is_file())
            with self.assertRaisesRegex(ValueError, 'independent trusted timestamp verifier'):
                build_prospective_run_seal(
                    root / 'unverified',
                    **{**_load_kwargs(materials), 'timestamp_verifier': None},  # type: ignore[arg-type]
                    timestamp_proof=proof,
                    proof_bytes=raw,
                )
            with self.assertRaisesRegex(ValidationError, 'RFC 3161 or a public transparency log'):
                ProspectiveRunTimestampProof(
                    **{
                        **proof.model_dump(),
                        'authority_type': TemporalReceiptAuthority.ORGANIZER_ATTESTATION,
                    }
                )

    def test_rejects_development_backend_and_late_run_or_witness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            development = _build_materials(root / 'development', run_tier=IsolationTier.DEVELOPMENT)
            with self.assertRaisesRegex(ValueError, 'official runner policy'):
                build_prospective_run_seal_target(
                    **{k: v for k, v in _load_kwargs(development).items() if k != 'timestamp_verifier'}
                )

            late_run = _build_materials(root / 'late-run', late_deadline=True)
            with self.assertRaisesRegex(ValueError, 'finished after'):
                build_prospective_run_seal_target(
                    **{k: v for k, v in _load_kwargs(late_run).items() if k != 'timestamp_verifier'}
                )

            materials = _build_materials(root / 'late-witness')
            target = build_prospective_run_seal_target(
                **{k: v for k, v in _load_kwargs(materials).items() if k != 'timestamp_verifier'}
            )
            proof, raw = _timestamp_material(
                target,
                witnessed_at=target.run_deadline_at + timedelta(seconds=1),
            )
            with self.assertRaisesRegex(ValueError, 'witness arrived after'):
                build_prospective_run_seal(
                    root / 'late-witness-seal',
                    **_load_kwargs(materials),
                    timestamp_proof=proof,
                    proof_bytes=raw,
                )

    def test_load_rejects_changed_system_policy_challenge_or_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            materials = _build_materials(root)
            kwargs = _load_kwargs(materials)
            target = build_prospective_run_seal_target(**{k: v for k, v in kwargs.items() if k != 'timestamp_verifier'})
            proof, raw = _timestamp_material(target)
            seal = build_prospective_run_seal(
                root / 'run-seal',
                **kwargs,
                timestamp_proof=proof,
                proof_bytes=raw,
            )

            changed_cases = {
                'system': {'system': materials[3].model_copy(update={'model_id': 'changed-model'})},
                'policy': {
                    'policy': materials[4].model_copy(
                        update={
                            'limits': materials[4].limits.model_copy(
                                update={'wall_seconds': materials[4].limits.wall_seconds + 1}
                            )
                        }
                    )
                },
                'challenge': {
                    'challenge': replace(materials[0], manifest_sha256='f' * 64),
                },
                'responses': {
                    'run': replace(materials[1], responses=b'null\n', response_records=(b'null\n',)),
                },
            }
            for name, update in changed_cases.items():
                with self.subTest(name=name), self.assertRaises((ValueError, ProspectiveRunSealIntegrityError)):
                    load_prospective_run_seal(seal.root, **{**kwargs, **update})

    def test_exact_proof_bytes_and_file_allowlist_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            materials = _build_materials(root)
            kwargs = _load_kwargs(materials)
            target = build_prospective_run_seal_target(**{k: v for k, v in kwargs.items() if k != 'timestamp_verifier'})
            proof, raw = _timestamp_material(target)
            seal = build_prospective_run_seal(
                root / 'run-seal',
                **kwargs,
                timestamp_proof=proof,
                proof_bytes=raw,
            )
            copied = root / 'tampered-proof'
            shutil.copytree(seal.root, copied)
            (copied / 'timestamp-proof.bin').write_bytes(raw + b'tampered')
            with self.assertRaisesRegex(ProspectiveRunSealIntegrityError, 'proof bytes'):
                load_prospective_run_seal(copied, **kwargs)

            copied = root / 'extra-file'
            shutil.copytree(seal.root, copied)
            (copied / 'organizer-choice.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ProspectiveRunSealIntegrityError, 'allowlist'):
                load_prospective_run_seal(copied, **kwargs)

    def test_failed_first_attempt_is_sealed_and_retry_cannot_be_cherry_picked(self) -> None:
        class FirstTargetPerAttemptKey:
            def __init__(self) -> None:
                self.targets: dict[str, str] = {}

            def __call__(self, proof: ProspectiveRunTimestampProof, raw: bytes) -> bool:
                if not raw.startswith(b'run-proof:'):
                    return False
                prior = self.targets.setdefault(proof.attempt_key_sha256, proof.target_sha256)
                return prior == proof.target_sha256

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            materials = _build_materials(root, invalid_first_response=True)
            challenge, failed_run, admission, system, policy, attempt_policy = materials
            self.assertEqual(failed_run.receipt.episodes[0].status, EpisodeRunStatus.INVALID_JSON)
            verifier = FirstTargetPerAttemptKey()
            common = {
                'challenge': challenge,
                'admission': admission,
                'system': system,
                'policy': policy,
                'attempt_policy': attempt_policy,
            }
            failed_target = build_prospective_run_seal_target(run=failed_run, **common)
            failed_proof, failed_raw = _timestamp_material(failed_target, suffix='failed')
            build_prospective_run_seal(
                root / 'failed-run-seal',
                run=failed_run,
                **common,
                timestamp_proof=failed_proof,
                proof_bytes=failed_raw,
                timestamp_verifier=verifier,
            )

            successful_run = run_challenge_bundle(
                challenge,
                expected_challenge_sha256=challenge.manifest_sha256,
                system=system,
                policy=policy,
                receipt_key=_RECEIPT_KEY,
                expected_receipt_key_id=receipt_key_id(_RECEIPT_KEY),
                output_dir=root / 'retry-run',
                backend=_Backend(_response(challenge.suite.episodes[0]).model_dump_json().encode()),
            )
            with self.assertRaisesRegex(ValueError, 'first and only attempt'):
                build_prospective_run_seal_target(
                    run=successful_run,
                    **common,
                    attempt_number=2,
                )

            # Relabelling the retry as attempt one still reuses the deterministic attempt key.  A
            # trusted transparency-log verifier rejects the conflicting target.
            retry_target = build_prospective_run_seal_target(run=successful_run, **common)
            self.assertEqual(retry_target.attempt_key_sha256, failed_target.attempt_key_sha256)
            retry_proof, retry_raw = _timestamp_material(retry_target, suffix='retry')
            with self.assertRaisesRegex(ValueError, 'trusted external timestamp verifier rejected'):
                build_prospective_run_seal(
                    root / 'retry-seal',
                    run=successful_run,
                    **common,
                    timestamp_proof=retry_proof,
                    proof_bytes=retry_raw,
                    timestamp_verifier=verifier,
                )


if __name__ == '__main__':
    unittest.main()
