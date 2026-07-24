from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.cli import main
from vaxreplay.runner.backend import (
    PreparedBackend,
    RawExecutionResult,
    RawExecutionStatus,
)
from vaxreplay.runner.challenge import build_challenge_bundle
from vaxreplay.runner.orchestrator import (
    RunArtifactIntegrityError,
    load_run_artifact,
    receipt_key_id,
    run_challenge_bundle,
)
from vaxreplay.runner.schema import (
    BackendCapabilities,
    EpisodeRunStatus,
    IsolationTier,
    RunnerPolicy,
    SystemSubmissionManifest,
)


def _fixture() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


_RECEIPT_KEY = bytes(range(32))
_RECEIPT_KEY_ID = receipt_key_id(_RECEIPT_KEY)


def _system() -> SystemSubmissionManifest:
    return SystemSubmissionManifest(
        submission_id='system-1',
        image_ref='sha256:' + 'a' * 64,
        entrypoint=('/opt/vaxreplay/run',),
        model_id='model-1',
        harness_id='harness-1',
    )


def _capabilities(tier: IsolationTier = IsolationTier.DEVELOPMENT) -> BackendCapabilities:
    return BackendCapabilities(
        backend_id='fake-test-backend',
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


class FakeBackend:
    def __init__(self, result: RawExecutionResult, *, tier: IsolationTier = IsolationTier.DEVELOPMENT):
        self.result = result
        self.capabilities = _capabilities(tier)
        self.inputs: list[bytes] = []
        self.prepare_calls = 0

    def prepare(self, system: SystemSubmissionManifest, policy: RunnerPolicy) -> PreparedBackend:
        self.prepare_calls += 1
        return PreparedBackend(capabilities=self.capabilities, resolved_image_id='sha256:' + 'b' * 64)

    def run(self, **kwargs: object) -> RawExecutionResult:
        self.inputs.append(kwargs['input_bytes'])  # type: ignore[arg-type]
        return self.result


class RunnerOrchestratorTest(unittest.TestCase):
    def test_canonicalizes_one_valid_response_and_emits_no_logs(self) -> None:
        bundle = EpisodeBundle.load(_fixture(), include_private=True)
        raw_response = oracle_submission(bundle).model_dump_json(indent=2).encode()
        backend = FakeBackend(
            RawExecutionResult(
                status=RawExecutionStatus.EXITED,
                exit_code=0,
                duration_ms=12,
                stdout=raw_response,
                stderr=b'secret-looking participant log',
                stdout_truncated=False,
                stderr_truncated=False,
            )
        )
        policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            challenge = build_challenge_bundle(
                root / 'challenge',
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            run = run_challenge_bundle(
                challenge,
                expected_challenge_sha256=challenge.manifest_sha256,
                system=_system(),
                policy=policy,
                receipt_key=_RECEIPT_KEY,
                expected_receipt_key_id=_RECEIPT_KEY_ID,
                output_dir=root / 'run',
                backend=backend,
            )

            self.assertEqual(run.receipt.episodes[0].status, EpisodeRunStatus.ACCEPTED)
            self.assertFalse(run.receipt.sealed)
            self.assertEqual(run.responses, canonical_json_bytes(oracle_submission(bundle)) + b'\n')
            self.assertNotIn(b'secret-looking participant log', (root / 'run' / 'run.json').read_bytes())
            self.assertEqual(
                (root / 'run' / 'audit' / '000000.stderr').read_bytes(),
                b'secret-looking participant log',
            )
            self.assertNotIn(b'private', backend.inputs[0])
            self.assertNotIn(b'POST-CUTOFF CANARY', backend.inputs[0])
            self.assertNotIn(b'"relevance_grade"', backend.inputs[0])
            self.assertEqual(
                set(path.name for path in (root / 'run').iterdir()),
                {'responses.jsonl', 'run.json', 'run.hmac', 'audit'},
            )

    def test_runtime_failures_become_null_rows_with_distinct_receipts(self) -> None:
        valid_stdout = (
            oracle_submission(EpisodeBundle.load(_fixture(), include_private=True)).model_dump_json().encode()
        )
        cases = (
            (
                RawExecutionResult(
                    status=RawExecutionStatus.TIMED_OUT,
                    exit_code=None,
                    duration_ms=1000,
                    stdout=b'',
                    stderr=b'',
                    stdout_truncated=False,
                    stderr_truncated=False,
                ),
                EpisodeRunStatus.TIMED_OUT,
            ),
            (
                RawExecutionResult(
                    status=RawExecutionStatus.EXITED,
                    exit_code=0,
                    duration_ms=1,
                    stdout=b'not-json',
                    stderr=b'',
                    stdout_truncated=False,
                    stderr_truncated=False,
                ),
                EpisodeRunStatus.INVALID_JSON,
            ),
            (
                RawExecutionResult(
                    status=RawExecutionStatus.EXITED,
                    exit_code=3,
                    duration_ms=1,
                    stdout=b'{}',
                    stderr=b'',
                    stdout_truncated=False,
                    stderr_truncated=False,
                ),
                EpisodeRunStatus.NONZERO_EXIT,
            ),
            (
                RawExecutionResult(
                    status=RawExecutionStatus.EXITED,
                    exit_code=0,
                    duration_ms=1,
                    stdout=valid_stdout,
                    stderr=b'',
                    stdout_truncated=True,
                    stderr_truncated=False,
                ),
                EpisodeRunStatus.RESPONSE_LIMIT,
            ),
            (
                RawExecutionResult(
                    status=RawExecutionStatus.EXITED,
                    exit_code=0,
                    duration_ms=1,
                    stdout=b'[' * 10_000 + b']' * 10_000,
                    stderr=b'',
                    stdout_truncated=False,
                    stderr_truncated=False,
                ),
                EpisodeRunStatus.INVALID_JSON,
            ),
        )
        policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            challenge = build_challenge_bundle(
                root / 'challenge',
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            for index, (raw, expected_status) in enumerate(cases):
                with self.subTest(status=expected_status):
                    run = run_challenge_bundle(
                        challenge,
                        expected_challenge_sha256=challenge.manifest_sha256,
                        system=_system(),
                        policy=policy,
                        receipt_key=_RECEIPT_KEY,
                        expected_receipt_key_id=_RECEIPT_KEY_ID,
                        output_dir=root / f'run-{index}',
                        backend=FakeBackend(raw),
                    )
                    self.assertEqual(run.responses, b'null\n')
                    self.assertEqual(run.receipt.episodes[0].status, expected_status)

    def test_official_policy_rejects_development_capability_before_execution(self) -> None:
        backend = FakeBackend(
            RawExecutionResult(
                status=RawExecutionStatus.EXITED,
                exit_code=0,
                duration_ms=1,
                stdout=b'{}',
                stderr=b'',
                stdout_truncated=False,
                stderr_truncated=False,
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            challenge = build_challenge_bundle(
                root / 'challenge',
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )

            with self.assertRaisesRegex(ValueError, 'official isolation'):
                run_challenge_bundle(
                    challenge,
                    expected_challenge_sha256=challenge.manifest_sha256,
                    system=_system(),
                    policy=RunnerPolicy(),
                    receipt_key=_RECEIPT_KEY,
                    expected_receipt_key_id=_RECEIPT_KEY_ID,
                    output_dir=root / 'run',
                    backend=backend,
                )
            self.assertEqual(backend.inputs, [])

    def test_wrong_preregistered_receipt_key_id_touches_no_backend(self) -> None:
        backend = FakeBackend(
            RawExecutionResult(
                status=RawExecutionStatus.EXITED,
                exit_code=0,
                duration_ms=1,
                stdout=b'{}',
                stderr=b'',
                stdout_truncated=False,
                stderr_truncated=False,
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            challenge = build_challenge_bundle(
                root / 'challenge',
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )

            with self.assertRaisesRegex(ValueError, 'preregistered organizer key'):
                run_challenge_bundle(
                    challenge,
                    expected_challenge_sha256=challenge.manifest_sha256,
                    system=_system(),
                    policy=RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT),
                    receipt_key=_RECEIPT_KEY,
                    expected_receipt_key_id='f' * 64,
                    output_dir=root / 'run',
                    backend=backend,
                )
            self.assertEqual(backend.prepare_calls, 0)
            self.assertEqual(backend.inputs, [])

    def test_run_artifact_tampering_and_development_scoring_are_fail_closed(self) -> None:
        bundle = EpisodeBundle.load(_fixture(), include_private=True)
        backend = FakeBackend(
            RawExecutionResult(
                status=RawExecutionStatus.EXITED,
                exit_code=0,
                duration_ms=1,
                stdout=oracle_submission(bundle).model_dump_json().encode(),
                stderr=b'',
                stdout_truncated=False,
                stderr_truncated=False,
            )
        )
        policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            challenge = build_challenge_bundle(
                root / 'challenge',
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            run_challenge_bundle(
                challenge,
                expected_challenge_sha256=challenge.manifest_sha256,
                system=_system(),
                policy=policy,
                receipt_key=_RECEIPT_KEY,
                expected_receipt_key_id=_RECEIPT_KEY_ID,
                output_dir=root / 'run',
                backend=backend,
            )

            with self.assertRaisesRegex(RunArtifactIntegrityError, 'development-tier'):
                load_run_artifact(
                    root / 'run',
                    challenge=challenge,
                    system=_system(),
                    policy=policy,
                    receipt_key=_RECEIPT_KEY,
                    expected_receipt_key_id=_RECEIPT_KEY_ID,
                )
            tampered_response = root / 'tampered-response'
            shutil.copytree(root / 'run', tampered_response)
            (tampered_response / 'responses.jsonl').write_bytes(b'null\n')
            with self.assertRaisesRegex(RunArtifactIntegrityError, 'run receipt'):
                load_run_artifact(
                    tampered_response,
                    challenge=challenge,
                    system=_system(),
                    policy=policy,
                    receipt_key=_RECEIPT_KEY,
                    expected_receipt_key_id=_RECEIPT_KEY_ID,
                    require_sealed=False,
                )

            tampered_audit = root / 'tampered-audit'
            shutil.copytree(root / 'run', tampered_audit)
            (tampered_audit / 'audit' / '000000.stderr').write_bytes(b'tampered')
            with self.assertRaisesRegex(RunArtifactIntegrityError, 'private audit'):
                load_run_artifact(
                    tampered_audit,
                    challenge=challenge,
                    system=_system(),
                    policy=policy,
                    receipt_key=_RECEIPT_KEY,
                    expected_receipt_key_id=_RECEIPT_KEY_ID,
                    require_sealed=False,
                )

            forged_official = root / 'forged-official'
            shutil.copytree(root / 'run', forged_official)
            forged_receipt = json.loads((forged_official / 'run.json').read_text(encoding='utf-8'))
            forged_receipt['sealed'] = True
            forged_receipt['capabilities']['isolation_tier'] = 'official'
            (forged_official / 'run.json').write_bytes(canonical_json_bytes(forged_receipt))
            with self.assertRaisesRegex(RunArtifactIntegrityError, 'HMAC'):
                load_run_artifact(
                    forged_official,
                    challenge=challenge,
                    system=_system(),
                    policy=policy,
                    receipt_key=_RECEIPT_KEY,
                    expected_receipt_key_id=_RECEIPT_KEY_ID,
                    require_sealed=False,
                )

            tampered_hmac = root / 'tampered-hmac'
            shutil.copytree(root / 'run', tampered_hmac)
            (tampered_hmac / 'run.hmac').write_text('0' * 64 + '\n', encoding='ascii')
            with self.assertRaisesRegex(RunArtifactIntegrityError, 'HMAC'):
                load_run_artifact(
                    tampered_hmac,
                    challenge=challenge,
                    system=_system(),
                    policy=policy,
                    receipt_key=_RECEIPT_KEY,
                    expected_receipt_key_id=_RECEIPT_KEY_ID,
                    require_sealed=False,
                )

    def test_private_score_run_requires_explicit_development_opt_in(self) -> None:
        bundle = EpisodeBundle.load(_fixture(), include_private=True)
        policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
        system = _system()
        backend = FakeBackend(
            RawExecutionResult(
                status=RawExecutionStatus.EXITED,
                exit_code=0,
                duration_ms=1,
                stdout=oracle_submission(bundle).model_dump_json().encode(),
                stderr=b'',
                stdout_truncated=False,
                stderr_truncated=False,
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            challenge = build_challenge_bundle(
                root / 'challenge',
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            run_challenge_bundle(
                challenge,
                expected_challenge_sha256=challenge.manifest_sha256,
                system=system,
                policy=policy,
                receipt_key=_RECEIPT_KEY,
                expected_receipt_key_id=_RECEIPT_KEY_ID,
                output_dir=root / 'run',
                backend=backend,
            )
            (root / 'system.json').write_bytes(canonical_json_bytes(system))
            (root / 'policy.json').write_bytes(canonical_json_bytes(policy))
            (root / 'receipt-key.hex').write_text(_RECEIPT_KEY.hex() + '\n', encoding='ascii')
            base_argv = [
                'vaxreplay',
                'score-run',
                '--challenge-dir',
                str(root / 'challenge'),
                '--expected-challenge-sha256',
                challenge.manifest_sha256,
                '--run-dir',
                str(root / 'run'),
                '--system-manifest',
                str(root / 'system.json'),
                '--policy',
                str(root / 'policy.json'),
                '--receipt-key',
                str(root / 'receipt-key.hex'),
                '--expected-receipt-key-id',
                _RECEIPT_KEY_ID,
                '--episode-dir',
                str(_fixture()),
            ]
            with patch.object(sys, 'argv', base_argv):
                with self.assertRaisesRegex(RunArtifactIntegrityError, 'development-tier'):
                    main()

            output = io.StringIO()
            with patch.object(sys, 'argv', [*base_argv, '--allow-development-run']):
                with redirect_stdout(output):
                    main()
            score = json.loads(output.getvalue())
            self.assertEqual(score['all_episode_mean_environment_reward'], 1.0)
            self.assertEqual(score['valid_episode_count'], 1)


if __name__ == '__main__':
    unittest.main()
