"""Dataset loading and split-isolation checks independent of any RL framework."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes, resolve_episode_root
from vaxreplay.case_schema import RewardVersion, Split, StrictModel, TaskType

SPLIT_ADMISSION_MANIFEST_SCHEMA_VERSION = 'vaxreplay.split_admission.v1'


class SplitAdmissionBinding(StrictModel):
    """Public identity and partition binding for one admitted episode."""

    episode_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    lineage_group_id: str = Field(min_length=1)
    task_type: TaskType
    split: Split


class SplitAdmissionManifest(StrictModel):
    """Cross-task release manifest enforcing lineage-level split isolation."""

    schema_version: Literal['vaxreplay.split_admission.v1'] = SPLIT_ADMISSION_MANIFEST_SCHEMA_VERSION
    admission_id: str = Field(min_length=1)
    episodes: tuple[SplitAdmissionBinding, ...] = Field(min_length=1)

    @field_validator('episodes')
    @classmethod
    def validate_episodes(
        cls,
        value: tuple[SplitAdmissionBinding, ...],
    ) -> tuple[SplitAdmissionBinding, ...]:
        episode_ids = tuple(binding.episode_id for binding in value)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('split-admission episode IDs must be unique')
        if episode_ids != tuple(sorted(episode_ids)):
            raise ValueError('split-admission episode bindings must be sorted by episode_id')
        return value

    @model_validator(mode='after')
    def validate_lineage_isolation(self) -> Self:
        _validate_lineage_assignments(self.episodes)
        return self


def load_episode_bundles(
    episode_dirs: str,
    expected_split: Split,
    expected_reward_version: RewardVersion | None = None,
) -> list[EpisodeBundle]:
    roots = [resolve_episode_root(value.strip()) for value in episode_dirs.split(',') if value.strip()]
    if not roots:
        raise ValueError(f'at least one {expected_split.value} episode directory is required')
    bundles = [EpisodeBundle.load(root, include_private=True) for root in roots]
    episode_ids = [bundle.manifest.episode_id for bundle in bundles]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError(f'duplicate episode IDs in {expected_split.value}: {episode_ids}')
    mismatches = [bundle.manifest.episode_id for bundle in bundles if bundle.manifest.split != expected_split]
    if mismatches:
        raise ValueError(f'episodes have the wrong split for {expected_split.value}: {mismatches}')
    reward_versions = {bundle.manifest.reward_version for bundle in bundles}
    if len(reward_versions) != 1:
        raise ValueError(f'episodes mix reward versions for {expected_split.value}: {sorted(reward_versions)}')
    if expected_reward_version is not None and reward_versions != {expected_reward_version}:
        raise ValueError(f'episodes have reward version {sorted(reward_versions)}, expected {expected_reward_version}')
    return bundles


def validate_split_isolation(
    train_bundles: Sequence[EpisodeBundle],
    dev_bundles: Sequence[EpisodeBundle],
    test_bundles: Sequence[EpisodeBundle] = (),
) -> None:
    """Reject episode or lineage reuse across any release partition.

    The check deliberately pools every task type before comparing lineage assignments. This means,
    for example, that an antigen-prioritization train episode and a candidate-advancement test
    episode from the same pathogen/program lineage cannot be admitted independently.
    """

    partitions = {
        Split.TRAIN: train_bundles,
        Split.DEV: dev_bundles,
        Split.TEST: test_bundles,
    }
    for expected_split, bundles in partitions.items():
        mismatches = sorted(bundle.manifest.episode_id for bundle in bundles if bundle.manifest.split != expected_split)
        if mismatches:
            raise ValueError(f'episodes have the wrong split for {expected_split.value}: {mismatches}')

    all_bundles = tuple(bundle for bundles in partitions.values() for bundle in bundles)
    episode_counts: dict[str, int] = defaultdict(int)
    for bundle in all_bundles:
        episode_counts[bundle.manifest.episode_id] += 1
    overlapping_episode_ids = sorted(episode_id for episode_id, count in episode_counts.items() if count > 1)
    if overlapping_episode_ids:
        raise ValueError(f'episode IDs overlap train, dev, or test: {overlapping_episode_ids}')

    _validate_lineage_assignments(
        SplitAdmissionBinding(
            episode_id=bundle.manifest.episode_id,
            manifest_sha256=bundle.manifest_sha256,
            lineage_group_id=bundle.manifest.lineage_group_id,
            task_type=bundle.manifest.task_type,
            split=bundle.manifest.split,
        )
        for bundle in all_bundles
    )

    versions_by_task: dict[TaskType, set[RewardVersion]] = defaultdict(set)
    for bundle in all_bundles:
        versions_by_task[bundle.manifest.task_type].add(bundle.manifest.reward_version)
    mixed_tasks = {
        task_type: sorted(versions_by_task[task_type])
        for task_type in sorted(versions_by_task)
        if len(versions_by_task[task_type]) > 1
    }
    if mixed_tasks:
        raise ValueError(f'reward versions differ across splits for task types: {mixed_tasks}')


def make_split_admission_manifest(
    admission_id: str,
    bundles: Iterable[EpisodeBundle],
) -> SplitAdmissionManifest:
    """Bind a release inventory to episode manifests and validate lineage isolation."""

    ordered_bundles = tuple(sorted(bundles, key=lambda bundle: bundle.manifest.episode_id))
    if not ordered_bundles:
        raise ValueError('cannot create an empty split-admission manifest')
    return SplitAdmissionManifest(
        admission_id=admission_id,
        episodes=tuple(
            SplitAdmissionBinding(
                episode_id=bundle.manifest.episode_id,
                manifest_sha256=bundle.manifest_sha256,
                lineage_group_id=bundle.manifest.lineage_group_id,
                task_type=bundle.manifest.task_type,
                split=bundle.manifest.split,
            )
            for bundle in ordered_bundles
        ),
    )


def split_admission_manifest_sha256(manifest: SplitAdmissionManifest) -> str:
    """Return the public precommitment for a validated admission manifest."""

    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def validate_split_admission_manifest(
    manifest: SplitAdmissionManifest,
    bundles: Iterable[EpisodeBundle],
) -> None:
    """Verify that an admission manifest exactly describes the supplied episode manifests."""

    derived = make_split_admission_manifest(manifest.admission_id, bundles)
    if derived != manifest:
        raise ValueError('split-admission manifest does not match the supplied episode bundles')


def validate_split_admission_subset(
    manifest: SplitAdmissionManifest,
    bundles: Iterable[EpisodeBundle],
) -> None:
    """Verify that every supplied episode is present in a larger frozen split inventory.

    Retrospective and official releases commit the complete train/dev/test inventory while a
    challenge normally contains only the selected test episodes.  The exact validator above is
    still used when the manifest is intended to describe precisely the supplied bundles; this
    validator is the deliberately narrower check for that complete-inventory release case.
    """

    supplied = tuple(bundles)
    if not supplied:
        raise ValueError('cannot validate an empty split-admission subset')
    expected_by_id = {binding.episode_id: binding for binding in manifest.episodes}
    observed = make_split_admission_manifest(manifest.admission_id, supplied)
    missing = sorted(binding.episode_id for binding in observed.episodes if binding.episode_id not in expected_by_id)
    if missing:
        raise ValueError(f'episode bundles are absent from the complete split inventory: {missing}')
    mismatched = sorted(
        binding.episode_id for binding in observed.episodes if expected_by_id.get(binding.episode_id) != binding
    )
    if mismatched:
        raise ValueError(f'episode bundles disagree with the complete split inventory: {mismatched}')


def _validate_lineage_assignments(bindings: Iterable[SplitAdmissionBinding]) -> None:
    lineage_splits: dict[str, set[Split]] = defaultdict(set)
    for binding in bindings:
        lineage_splits[binding.lineage_group_id].add(binding.split)
    overlapping_lineages = {
        lineage_group_id: sorted(split.value for split in lineage_splits[lineage_group_id])
        for lineage_group_id in sorted(lineage_splits)
        if len(lineage_splits[lineage_group_id]) > 1
    }
    if overlapping_lineages:
        raise ValueError(
            f'lineage groups overlap train, dev, or test (including across task types): {overlapping_lineages}'
        )
