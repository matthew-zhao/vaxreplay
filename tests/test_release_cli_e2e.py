from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split
from vaxreplay.cli import main as vaxreplay_main
from vaxreplay.iedb.cli import main as iedb_main
from vaxreplay.release import ReleaseIntegrityError, load_release, public_release_sha256
from vaxreplay.runner.backend import PreparedBackend, RawExecutionResult, RawExecutionStatus
from vaxreplay.runner.orchestrator import receipt_key_id, run_challenge_bundle
from vaxreplay.runner.schema import (
    BackendCapabilities,
    IsolationTier,
    RunnerPolicy,
    SystemSubmissionManifest,
)

_LABEL_KEY = bytes(range(32))
_RECEIPT_KEY = bytes(range(32, 64))


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'iedb_fictional_history'


def _development_capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        backend_id='fake-oracle-backend',
        backend_version='1',
        isolation_tier=IsolationTier.DEVELOPMENT,
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


class OracleDevelopmentBackend:
    def __init__(self, response: bytes):
        self.response = response
        self.capabilities = _development_capabilities()
        self.inputs: list[bytes] = []

    def prepare(self, system: SystemSubmissionManifest, policy: RunnerPolicy) -> PreparedBackend:
        return PreparedBackend(
            capabilities=self.capabilities,
            resolved_image_id='sha256:' + 'b' * 64,
        )

    def run(self, **kwargs: object) -> RawExecutionResult:
        self.inputs.append(kwargs['input_bytes'])  # type: ignore[arg-type]
        return RawExecutionResult(
            status=RawExecutionStatus.EXITED,
            exit_code=0,
            duration_ms=1,
            stdout=self.response,
            stderr=b'',
            stdout_truncated=False,
            stderr_truncated=False,
        )


class ReleaseCliEndToEndTest(unittest.TestCase):
    def test_build_synthetic_pilot_run_oracle_and_score_bound_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            receipt_key_path = root / 'receipt-key.hex'
            label_key_path = root / 'label-key.hex'
            receipt_key_path.write_text(_RECEIPT_KEY.hex() + '\n', encoding='ascii')
            label_key_path.write_text(_LABEL_KEY.hex() + '\n', encoding='ascii')
            receipt_key_path.chmod(0o600)
            label_key_path.chmod(0o600)
            public_release = root / 'public-release'
            private_release = root / 'private-release'

            pilot_output = io.StringIO()
            with (
                patch.object(
                    sys,
                    'argv',
                    [
                        'vaxreplay-iedb',
                        'build-synthetic-pilot',
                        '--spec',
                        str(_fixture_root() / 'spec.json'),
                        '--snapshot-dir',
                        str(_fixture_root() / 'snapshot_decision'),
                        '--snapshot-dir',
                        str(_fixture_root() / 'snapshot_outcome'),
                        '--receipt-key',
                        str(receipt_key_path),
                        '--label-commitment-key-file',
                        str(label_key_path),
                        '--release-id',
                        'iedb-cli-e2e-release',
                        '--challenge-id',
                        'iedb-cli-e2e-challenge',
                        '--suite-id',
                        'iedb-cli-e2e-suite',
                        '--public-output-dir',
                        str(public_release),
                        '--private-output-dir',
                        str(private_release),
                    ],
                ),
                redirect_stdout(pilot_output),
            ):
                iedb_main()
            pilot = json.loads(pilot_output.getvalue())

            self.assertEqual(pilot['release_id'], 'iedb-cli-e2e-release')
            self.assertEqual(pilot['source_tier'], 'tier_c')
            self.assertFalse(pilot['sealed_eligible'])
            self.assertFalse(pilot['split_inventory_complete'])
            self.assertEqual(pilot['receipt_key_id'], receipt_key_id(_RECEIPT_KEY))

            release = load_release(
                public_release,
                private_release,
                expected_public_release_sha256=pilot['public_release_sha256'],
            )
            self.assertEqual(public_release_sha256(release.public_manifest), pilot['public_release_sha256'])
            self.assertEqual(release.challenge.manifest_sha256, pilot['challenge_bundle_sha256'])

            system = SystemSubmissionManifest(
                submission_id='iedb-cli-oracle-system',
                image_ref='sha256:' + 'a' * 64,
                entrypoint=('/opt/vaxreplay/run',),
                model_id='fixture-oracle',
                harness_id='fake-development-harness',
            )
            system_path = root / 'system.json'
            system_path.write_bytes(canonical_json_bytes(system))
            sealed_bundle = release.bundles[0]
            train_view = replace(
                sealed_bundle,
                manifest=sealed_bundle.manifest.model_copy(update={'split': Split.TRAIN}),
            )
            oracle = oracle_submission(train_view).model_copy(update={'manifest_sha256': sealed_bundle.manifest_sha256})
            oracle_response = oracle.model_dump_json().encode('utf-8')
            backend = OracleDevelopmentBackend(oracle_response)
            run = run_challenge_bundle(
                release.challenge,
                expected_challenge_sha256=pilot['challenge_bundle_sha256'],
                system=system,
                policy=release.policy,
                receipt_key=_RECEIPT_KEY,
                expected_receipt_key_id=pilot['receipt_key_id'],
                output_dir=root / 'run',
                backend=backend,
            )

            self.assertEqual(run.receipt.challenge_bundle_sha256, release.public_manifest.challenge_bundle_sha256)
            self.assertEqual(run.receipt.admission_sha256, release.public_manifest.admission_sha256)
            self.assertEqual(run.receipt.receipt_key_id, release.public_manifest.receipt_key_id)
            self.assertFalse(run.receipt.sealed)
            self.assertEqual(len(backend.inputs), 1)
            self.assertNotIn(_LABEL_KEY.hex().encode('ascii'), backend.inputs[0])

            score_output = io.StringIO()
            score_argv = [
                'vaxreplay',
                'score-release-run',
                '--public-release-dir',
                str(public_release),
                '--private-release-dir',
                str(private_release),
                '--expected-release-sha256',
                pilot['public_release_sha256'],
                '--run-dir',
                str(root / 'run'),
                '--system-manifest',
                str(system_path),
                '--receipt-key',
                str(receipt_key_path),
            ]
            with patch.object(sys, 'argv', score_argv), redirect_stdout(score_output):
                vaxreplay_main()
            score = json.loads(score_output.getvalue())

            self.assertEqual(score['valid_only_mean_reward'], 1.0)
            self.assertEqual(score['all_episode_mean_environment_reward'], 1.0)
            self.assertEqual(score['valid_episode_count'], 1)
            self.assertEqual(score['missing_episode_ids'], [])

            wrong_release_argv = score_argv.copy()
            wrong_release_argv[wrong_release_argv.index(pilot['public_release_sha256'])] = 'f' * 64
            with patch.object(sys, 'argv', wrong_release_argv):
                with self.assertRaisesRegex(ReleaseIntegrityError, 'preregistered hash'):
                    vaxreplay_main()


if __name__ == '__main__':
    unittest.main()
