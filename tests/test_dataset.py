from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import RANKING_REWARD_VERSION, Split
from vaxreplay.dataset import (
    SplitAdmissionManifest,
    load_episode_bundles,
    make_split_admission_manifest,
    split_admission_manifest_sha256,
    validate_split_admission_manifest,
    validate_split_admission_subset,
    validate_split_isolation,
)


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0'


def _preclinical_fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EpisodeBundle.load(_fixture_root(), include_private=True)

    def test_loads_published_training_bundle(self) -> None:
        bundles = load_episode_bundles(str(_fixture_root()), Split.TRAIN)

        self.assertEqual([bundle.manifest.episode_id for bundle in bundles], ['synthetic-antigen-001'])

    def test_rejects_duplicate_episode_directories(self) -> None:
        directories = f'{_fixture_root()},{_fixture_root()}'

        with self.assertRaisesRegex(ValueError, 'duplicate episode IDs'):
            load_episode_bundles(directories, Split.TRAIN)

    def test_rejects_wrong_split(self) -> None:
        with self.assertRaisesRegex(ValueError, 'wrong split'):
            load_episode_bundles(str(_fixture_root()), Split.DEV)

    def test_rejects_unexpected_reward_version(self) -> None:
        with self.assertRaisesRegex(ValueError, 'expected v1.0'):
            load_episode_bundles(
                str(_fixture_root()),
                Split.TRAIN,
                RANKING_REWARD_VERSION,
            )

    def test_rejects_episode_id_overlap_between_train_and_dev(self) -> None:
        dev_bundle = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'split': Split.DEV}),
        )

        with self.assertRaisesRegex(ValueError, 'episode IDs overlap'):
            validate_split_isolation([self.bundle], [dev_bundle])

    def test_rejects_lineage_overlap_between_train_and_dev(self) -> None:
        dev_bundle = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(
                update={'episode_id': 'synthetic-antigen-dev', 'split': Split.DEV}
            ),
        )

        with self.assertRaisesRegex(ValueError, 'lineage groups overlap'):
            validate_split_isolation([self.bundle], [dev_bundle])

    def test_rejects_reward_version_mismatch_between_train_and_dev(self) -> None:
        dev_bundle = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(
                update={
                    'episode_id': 'synthetic-antigen-dev',
                    'lineage_group_id': 'synthetic-lineage-dev',
                    'split': Split.DEV,
                    'reward_version': RANKING_REWARD_VERSION,
                }
            ),
        )

        with self.assertRaisesRegex(ValueError, 'reward versions differ'):
            validate_split_isolation([self.bundle], [dev_bundle])

    def test_rejects_cross_task_lineage_overlap_between_train_and_test(self) -> None:
        preclinical = EpisodeBundle.load(_preclinical_fixture_root(), include_private=True)
        test_bundle = replace(
            preclinical,
            manifest=preclinical.manifest.model_copy(
                update={
                    'lineage_group_id': self.bundle.manifest.lineage_group_id,
                    'split': Split.TEST,
                }
            ),
        )

        with self.assertRaisesRegex(ValueError, 'including across task types'):
            validate_split_isolation([self.bundle], [], [test_bundle])

    def test_allows_cross_task_lineage_reuse_within_one_split(self) -> None:
        preclinical = EpisodeBundle.load(_preclinical_fixture_root(), include_private=True)
        train_bundle = replace(
            preclinical,
            manifest=preclinical.manifest.model_copy(
                update={
                    'lineage_group_id': self.bundle.manifest.lineage_group_id,
                    'split': Split.TRAIN,
                }
            ),
        )

        validate_split_isolation([self.bundle, train_bundle], [])
        admission = make_split_admission_manifest('cross-task-train', [train_bundle, self.bundle])

        self.assertEqual(
            tuple(binding.episode_id for binding in admission.episodes),
            ('synthetic-antigen-001', 'synthetic-preclinical-001'),
        )
        self.assertEqual(
            {binding.task_type for binding in admission.episodes},
            {
                'antigen_target_prioritization',
                'preclinical_candidate_advancement',
            },
        )
        self.assertEqual(len(split_admission_manifest_sha256(admission)), 64)

    def test_split_admission_manifest_rejects_tampered_lineage_assignment(self) -> None:
        preclinical = EpisodeBundle.load(_preclinical_fixture_root(), include_private=True)
        admission = make_split_admission_manifest('release', [self.bundle, preclinical])
        leaked_binding = admission.episodes[1].model_copy(
            update={
                'lineage_group_id': admission.episodes[0].lineage_group_id,
                'split': Split.TEST,
            }
        )

        with self.assertRaisesRegex(ValidationError, 'lineage groups overlap'):
            SplitAdmissionManifest(
                admission_id=admission.admission_id,
                episodes=(admission.episodes[0], leaked_binding),
            )

    def test_split_admission_manifest_is_bound_to_episode_manifests(self) -> None:
        admission = make_split_admission_manifest('release', [self.bundle])
        tampered_binding = admission.episodes[0].model_copy(update={'lineage_group_id': 'other-lineage'})
        tampered = SplitAdmissionManifest(
            admission_id=admission.admission_id,
            episodes=(tampered_binding,),
        )

        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_split_admission_manifest(tampered, [self.bundle])

    def test_complete_split_inventory_can_validate_a_selected_subset(self) -> None:
        preclinical = EpisodeBundle.load(_preclinical_fixture_root(), include_private=True)
        admission = make_split_admission_manifest('complete-release-inventory', [self.bundle, preclinical])

        validate_split_admission_subset(admission, [preclinical])

    def test_complete_split_inventory_rejects_an_unknown_or_changed_episode(self) -> None:
        admission = make_split_admission_manifest('complete-release-inventory', [self.bundle])
        changed = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'episode_id': 'not-in-inventory'}),
        )
        with self.subTest(case='unknown'), self.assertRaisesRegex(ValueError, 'absent'):
            validate_split_admission_subset(admission, [changed])

        changed = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'lineage_group_id': 'changed-lineage'}),
        )
        with self.subTest(case='changed'), self.assertRaisesRegex(ValueError, 'disagree'):
            validate_split_admission_subset(admission, [changed])

    def test_split_isolation_rejects_duplicate_episode_within_a_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, 'episode IDs overlap'):
            validate_split_isolation([self.bundle, self.bundle], [])

    def test_split_isolation_rejects_misfiled_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, 'wrong split for dev'):
            validate_split_isolation([], [self.bundle])


if __name__ == '__main__':
    unittest.main()
