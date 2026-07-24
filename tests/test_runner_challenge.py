from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from vaxreplay.aggregation import suite_manifest_sha256
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_schema import Split
from vaxreplay.prompt import (
    PromptVariant,
    build_episode_prompt,
    build_system_prompt,
    model_facing_payload_bytes,
)
from vaxreplay.runner.challenge import (
    ChallengeIntegrityError,
    build_challenge_bundle,
    load_challenge_bundle,
)


def _fixture() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


class RunnerChallengeTest(unittest.TestCase):
    def test_builds_deterministic_public_message_only_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = build_challenge_bundle(
                root / 'first',
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            second = build_challenge_bundle(
                root / 'second',
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            bundle = EpisodeBundle.load(_fixture())

            self.assertEqual(first.manifest_sha256, second.manifest_sha256)
            self.assertEqual(first.manifest.suite_manifest_sha256, suite_manifest_sha256(first.suite))
            self.assertEqual(first.envelopes[0].messages[0].content, build_system_prompt(bundle))
            self.assertEqual(first.envelopes[0].messages[1].content, build_episode_prompt(bundle))
            self.assertEqual(
                {path.relative_to(first.root).as_posix() for path in first.root.rglob('*') if path.is_file()},
                {'challenge.json', 'suite.json', 'episodes/000000.json'},
            )
            all_bytes = b''.join(path.read_bytes() for path in first.root.rglob('*') if path.is_file())
            for forbidden in (
                b'private/',
                b'outcomes.jsonl',
                b'ranking_labels.jsonl',
                b'label_commitment_key.hex',
                b'assessments_gold.jsonl',
                b'POST-CUTOFF CANARY',
                b'"relevance_grade"',
            ):
                self.assertNotIn(forbidden, all_bytes)

    def test_rejects_sha256_commitment_for_any_sealed_test_episode(self) -> None:
        antigen_fixture = Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = root / 'episode'
            shutil.copytree(antigen_fixture, episode)
            bundle = EpisodeBundle.load(episode)
            test_manifest = bundle.manifest.model_copy(update={'split': Split.TEST})
            (episode / 'manifest.json').write_bytes(canonical_json_bytes(test_manifest))

            with self.assertRaisesRegex(ValueError, 'HMAC-SHA256'):
                build_challenge_bundle(
                    root / 'challenge',
                    challenge_id='challenge-1',
                    suite_id='suite-1',
                    episode_dirs=[episode],
                )

    def test_builds_fixed_contamination_sensitivity_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle = EpisodeBundle.load(_fixture())
            full = build_challenge_bundle(
                root / 'full',
                challenge_id='challenge-full',
                suite_id='suite-full',
                episode_dirs=[_fixture()],
            )
            scrubbed = build_challenge_bundle(
                root / 'scrubbed',
                challenge_id='challenge-scrubbed',
                suite_id='suite-scrubbed',
                episode_dirs=[_fixture()],
                prompt_variant=PromptVariant.BIBLIOGRAPHICALLY_SCRUBBED,
            )
            no_evidence = build_challenge_bundle(
                root / 'no-evidence',
                challenge_id='challenge-no-evidence',
                suite_id='suite-no-evidence',
                episode_dirs=[_fixture()],
                prompt_variant=PromptVariant.NO_EVIDENCE,
            )

            original_title = bundle.visible_evidence[0].title
            self.assertIn(original_title, full.envelopes[0].messages[1].content)
            self.assertNotIn(original_title, scrubbed.envelopes[0].messages[1].content)
            self.assertIn('Historical source 1', scrubbed.envelopes[0].messages[1].content)
            self.assertIn('"evidence": []', no_evidence.envelopes[0].messages[1].content)
            self.assertEqual(scrubbed.manifest.prompt_variant, PromptVariant.BIBLIOGRAPHICALLY_SCRUBBED)
            self.assertEqual(no_evidence.envelopes[0].prompt_variant, PromptVariant.NO_EVIDENCE)
            payload = model_facing_payload_bytes(
                bundle,
                variant=PromptVariant.BIBLIOGRAPHICALLY_SCRUBBED,
            )
            parsed_payload = json.loads(payload)
            self.assertEqual(
                [message['content'] for message in parsed_payload['messages']],
                [message.content for message in scrubbed.envelopes[0].messages],
            )

    def test_rejects_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'challenge'
            build_challenge_bundle(
                root,
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            (root / 'private.json').write_text('{}', encoding='utf-8')

            with self.assertRaisesRegex(ChallengeIntegrityError, 'allowlist'):
                load_challenge_bundle(root)

    def test_rejects_tampered_envelope_even_if_it_remains_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'challenge'
            challenge = build_challenge_bundle(
                root,
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            envelope = challenge.envelopes[0].model_copy(update={'sample_index': 99})
            (root / 'episodes' / '000000.json').write_bytes(canonical_json_bytes(envelope))

            with self.assertRaisesRegex(ChallengeIntegrityError, 'hash mismatch'):
                load_challenge_bundle(root)

    def test_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'challenge'
            build_challenge_bundle(
                root,
                challenge_id='challenge-1',
                suite_id='suite-1',
                episode_dirs=[_fixture()],
            )
            (root / 'leak').symlink_to(_fixture() / 'private')

            with self.assertRaisesRegex(ChallengeIntegrityError, 'symlink'):
                load_challenge_bundle(root)


if __name__ == '__main__':
    unittest.main()
