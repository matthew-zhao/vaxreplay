from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from vaxreplay.bundle import BundleIntegrityError, EpisodeBundle, labels_sha256
from vaxreplay.prompt import build_episode_prompt, build_system_prompt


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'


class EpisodeBundleTest(unittest.TestCase):
    def test_loads_and_filters_by_public_availability(self) -> None:
        bundle = EpisodeBundle.load(_fixture_root(), include_private=True)

        self.assertEqual(bundle.manifest.episode_id, 'synthetic-antigen-001')
        self.assertEqual(
            [record.evidence_id for record in bundle.visible_evidence],
            [
                'ev-surveillance-1',
                'ev-accessibility-1',
                'ev-functional-1',
            ],
        )
        self.assertEqual(len(bundle.private_labels.outcomes), 3)

    def test_public_prompt_excludes_future_canary_and_private_labels(self) -> None:
        bundle = EpisodeBundle.load(_fixture_root(), include_private=True)

        prompt = build_episode_prompt(bundle)

        self.assertNotIn('POST-CUTOFF CANARY', prompt)
        self.assertNotIn('candidate_utility', prompt)
        self.assertNotIn('revealed_at', prompt)
        self.assertNotIn('disclosed_facts', prompt)
        self.assertNotIn('display_name', prompt)
        self.assertIn('ev-functional-1', prompt)

    def test_manifest_hash_is_deterministic(self) -> None:
        first = EpisodeBundle.load(_fixture_root()).manifest_sha256
        second = EpisodeBundle.load(_fixture_root()).manifest_sha256

        self.assertEqual(first, second)
        self.assertEqual(first, '5d49243cf71a16fd71bc7477abfc33517ef0822ba0b34dfcede8767010c27796')

    def test_v0_label_commitment_and_prompt_bytes_are_golden(self) -> None:
        bundle = EpisodeBundle.load(_fixture_root(), include_private=True)

        self.assertEqual(
            bundle.manifest.labels_sha256,
            'fa8c5af59c3e0242d5891e59922b1379bffcea492a4a89e49c2b05b8b8a29a3d',
        )
        self.assertEqual(
            hashlib.sha256(build_system_prompt(bundle).encode()).hexdigest(),
            '240f277f0dcc3879f74503ed0777885820c97ab6a115dae993e938d9da308dd5',
        )
        self.assertEqual(
            hashlib.sha256(build_episode_prompt(bundle).encode()).hexdigest(),
            '4f85768dbffae4529145f5e72f448dcc0198e66feec00bd7bb56b29e6d23dc18',
        )

    def test_rejects_tampered_evidence_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'episode'
            shutil.copytree(_fixture_root(), root)
            evidence_path = root / 'evidence.jsonl'
            evidence_path.write_text(
                evidence_path.read_text(encoding='utf-8').replace('96 of 100', '95 of 100'),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(BundleIntegrityError, 'evidence snapshot hash'):
                EpisodeBundle.load(root)

    def test_rejects_episode_with_only_censored_forecasts(self) -> None:
        bundle = EpisodeBundle.load(_fixture_root(), include_private=True)
        labels = bundle.private_labels
        assert labels is not None
        censored_labels = labels.model_copy(
            update={
                'outcomes': [
                    outcome.model_copy(update={'outcome': None, 'censor_reason': 'still_active'})
                    for outcome in labels.outcomes
                ]
            }
        )
        censored_manifest = bundle.manifest.model_copy(update={'labels_sha256': labels_sha256(censored_labels)})

        with self.assertRaisesRegex(BundleIntegrityError, 'non-censored forecast'):
            replace(bundle, manifest=censored_manifest, private_labels=censored_labels).validate_integrity()

    def test_rejects_duplicate_gold_evidence(self) -> None:
        bundle = EpisodeBundle.load(_fixture_root(), include_private=True)
        labels = bundle.private_labels
        assert labels is not None
        duplicated_labels = labels.model_copy(
            update={'evidence_gold': [*labels.evidence_gold, labels.evidence_gold[0]]}
        )
        duplicated_manifest = bundle.manifest.model_copy(update={'labels_sha256': labels_sha256(duplicated_labels)})

        with self.assertRaisesRegex(BundleIntegrityError, 'gold evidence records must be unique'):
            replace(bundle, manifest=duplicated_manifest, private_labels=duplicated_labels).validate_integrity()

    def test_private_labels_are_bound_to_manifest_commitment(self) -> None:
        bundle = EpisodeBundle.load(_fixture_root(), include_private=True)
        labels = bundle.private_labels
        assert labels is not None
        mutated_labels = labels.model_copy(
            update={
                'outcomes': [
                    labels.outcomes[0].model_copy(update={'outcome': 0}),
                    *labels.outcomes[1:],
                ]
            }
        )

        with self.assertRaisesRegex(BundleIntegrityError, 'private label hash'):
            replace(bundle, private_labels=mutated_labels).validate_integrity()


if __name__ == '__main__':
    unittest.main()
