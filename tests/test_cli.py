from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vaxreplay.aggregation import make_suite_manifest, suite_manifest_sha256
from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import (
    EpisodeBundle,
    canonical_json_bytes,
    ranking_labels_commitment,
)
from vaxreplay.case_schema import LabelCommitmentScheme, Split
from vaxreplay.cli import main


def _fixture() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


class CliTest(unittest.TestCase):
    def test_make_suite_command_binds_episode_manifest(self) -> None:
        output = io.StringIO()

        with patch.object(
            sys,
            'argv',
            [
                'vaxreplay',
                'make-suite',
                '--suite-id',
                'preclinical-dev',
                '--episode-dir',
                str(_fixture()),
            ],
        ):
            with redirect_stdout(output):
                main()

        result = json.loads(output.getvalue())
        self.assertEqual(result['suite_id'], 'preclinical-dev')
        self.assertEqual(result['task_type'], 'preclinical_candidate_advancement')
        self.assertEqual(result['episodes'][0]['episode_id'], 'synthetic-preclinical-001')

    def test_suite_hash_command_emits_the_precommitment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            bundle = EpisodeBundle.load(_fixture())
            manifest = make_suite_manifest('preclinical-dev', [bundle])
            path = Path(temporary_directory) / 'suite.json'
            path.write_text(manifest.model_dump_json(), encoding='utf-8')
            output = io.StringIO()

            with patch.object(
                sys,
                'argv',
                ['vaxreplay', 'suite-hash', '--suite-manifest', str(path)],
            ):
                with redirect_stdout(output):
                    main()

            self.assertEqual(
                json.loads(output.getvalue())['suite_manifest_sha256'],
                suite_manifest_sha256(manifest),
            )

    def test_score_suite_evaluates_responses_inside_the_private_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle = EpisodeBundle.load(_fixture(), include_private=True)
            manifest = make_suite_manifest('preclinical-dev', [bundle])
            manifest_path = root / 'suite.json'
            responses_path = root / 'responses.jsonl'
            manifest_path.write_text(manifest.model_dump_json(), encoding='utf-8')
            responses_path.write_text(oracle_submission(bundle).model_dump_json() + '\n', encoding='utf-8')
            output = io.StringIO()

            with patch.object(
                sys,
                'argv',
                [
                    'vaxreplay',
                    'score-suite',
                    '--suite-manifest',
                    str(manifest_path),
                    '--expected-suite-sha256',
                    suite_manifest_sha256(manifest),
                    '--episode-dir',
                    str(_fixture()),
                    '--responses-jsonl',
                    str(responses_path),
                ],
            ):
                with redirect_stdout(output):
                    main()

            result = json.loads(output.getvalue())
            self.assertEqual(result['suite_id'], 'preclinical-dev')
            self.assertEqual(result['valid_only_mean_reward'], 1.0)
            self.assertEqual(result['all_episode_mean_environment_reward'], 1.0)
            self.assertEqual(result['missing_episode_ids'], [])

    def test_score_suite_penalizes_a_malformed_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle = EpisodeBundle.load(_fixture(), include_private=True)
            manifest = make_suite_manifest('preclinical-dev', [bundle])
            manifest_path = root / 'suite.json'
            responses_path = root / 'responses.jsonl'
            manifest_path.write_text(manifest.model_dump_json(), encoding='utf-8')
            responses_path.write_text('not-json\n', encoding='utf-8')
            output = io.StringIO()

            with patch.object(
                sys,
                'argv',
                [
                    'vaxreplay',
                    'score-suite',
                    '--suite-manifest',
                    str(manifest_path),
                    '--expected-suite-sha256',
                    suite_manifest_sha256(manifest),
                    '--episode-dir',
                    str(_fixture()),
                    '--responses-jsonl',
                    str(responses_path),
                ],
            ):
                with redirect_stdout(output):
                    main()

            result = json.loads(output.getvalue())
            self.assertEqual(result['all_episode_mean_environment_reward'], -1.0)
            self.assertEqual(result['missing_episode_ids'], ['synthetic-preclinical-001'])

    def test_score_suite_penalizes_a_non_utf8_response_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle = EpisodeBundle.load(_fixture(), include_private=True)
            manifest = make_suite_manifest('preclinical-dev', [bundle])
            manifest_path = root / 'suite.json'
            responses_path = root / 'responses.jsonl'
            manifest_path.write_text(manifest.model_dump_json(), encoding='utf-8')
            responses_path.write_bytes(b'\xff\n')
            output = io.StringIO()

            with patch.object(
                sys,
                'argv',
                [
                    'vaxreplay',
                    'score-suite',
                    '--suite-manifest',
                    str(manifest_path),
                    '--expected-suite-sha256',
                    suite_manifest_sha256(manifest),
                    '--episode-dir',
                    str(_fixture()),
                    '--responses-jsonl',
                    str(responses_path),
                ],
            ):
                with redirect_stdout(output):
                    main()

            result = json.loads(output.getvalue())
            self.assertEqual(result['all_episode_mean_environment_reward'], -1.0)
            self.assertEqual(result['missing_episode_ids'], ['synthetic-preclinical-001'])

    def test_score_suite_is_the_explicit_private_path_for_a_sealed_v1_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode_root = root / 'sealed-v1'
            shutil.copytree(_fixture(), episode_root)
            train_bundle = EpisodeBundle.load(episode_root, include_private=True)
            submission = oracle_submission(train_bundle)
            private_labels = train_bundle.private_labels
            ranking_labels = train_bundle.ranking_labels
            assert private_labels is not None
            assert ranking_labels is not None
            commitment_key = bytes(range(32))
            commitment = ranking_labels_commitment(
                private_labels,
                ranking_labels,
                LabelCommitmentScheme.HMAC_SHA256,
                key=commitment_key,
            )
            test_manifest = train_bundle.manifest.model_copy(
                update={
                    'split': Split.TEST,
                    'labels_sha256': commitment,
                    'label_commitment_scheme': LabelCommitmentScheme.HMAC_SHA256,
                    'label_commitment_key_id': hashlib.sha256(commitment_key).hexdigest(),
                }
            )
            (episode_root / 'manifest.json').write_bytes(canonical_json_bytes(test_manifest))
            (episode_root / 'private' / 'label_commitment_key.hex').write_text(
                commitment_key.hex() + '\n',
                encoding='ascii',
            )
            test_bundle = EpisodeBundle.load(episode_root, include_private=True)
            submission = submission.model_copy(update={'manifest_sha256': test_bundle.manifest_sha256})
            suite = make_suite_manifest('sealed-v1-test', [test_bundle])
            suite_path = root / 'suite.json'
            responses_path = root / 'responses.jsonl'
            suite_path.write_text(suite.model_dump_json(), encoding='utf-8')
            responses_path.write_text(submission.model_dump_json() + '\n', encoding='utf-8')
            output = io.StringIO()

            with patch.object(
                sys,
                'argv',
                [
                    'vaxreplay',
                    'score-suite',
                    '--suite-manifest',
                    str(suite_path),
                    '--expected-suite-sha256',
                    suite_manifest_sha256(suite),
                    '--episode-dir',
                    str(episode_root),
                    '--responses-jsonl',
                    str(responses_path),
                ],
            ):
                with redirect_stdout(output):
                    main()

            result = json.loads(output.getvalue())
            self.assertEqual(result['all_episode_mean_environment_reward'], 1.0)
            self.assertEqual(result['valid_episode_count'], 1)

    def test_score_suite_rejects_a_precommitted_manifest_that_does_not_match_the_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle = EpisodeBundle.load(_fixture(), include_private=True)
            manifest = make_suite_manifest('preclinical-dev', [bundle])
            tampered_binding = manifest.episodes[0].model_copy(update={'manifest_sha256': '0' * 64})
            tampered_manifest = manifest.model_copy(update={'episodes': (tampered_binding,)})
            manifest_path = root / 'suite.json'
            responses_path = root / 'responses.jsonl'
            manifest_path.write_text(tampered_manifest.model_dump_json(), encoding='utf-8')
            responses_path.write_text(oracle_submission(bundle).model_dump_json() + '\n', encoding='utf-8')

            with patch.object(
                sys,
                'argv',
                [
                    'vaxreplay',
                    'score-suite',
                    '--suite-manifest',
                    str(manifest_path),
                    '--expected-suite-sha256',
                    suite_manifest_sha256(tampered_manifest),
                    '--episode-dir',
                    str(_fixture()),
                    '--responses-jsonl',
                    str(responses_path),
                ],
            ):
                with self.assertRaisesRegex(ValueError, 'supplied episode bundles'):
                    main()


if __name__ == '__main__':
    unittest.main()
