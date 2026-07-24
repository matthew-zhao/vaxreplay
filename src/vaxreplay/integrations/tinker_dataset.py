"""Tinker dataset plumbing for published VaxReplay train/dev episodes."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import chz
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_model_attributes
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, Metrics, RLDataset, RLDatasetBuilder, Trajectory
from tinker_cookbook.tokenizer_utils import get_tokenizer

from vaxreplay.bundle import EpisodeBundle
from vaxreplay.case_schema import REWARD_VERSION, RewardVersion, Split
from vaxreplay.dataset import load_episode_bundles, validate_split_isolation
from vaxreplay.integrations.tinker import make_tinker_env
from vaxreplay.prompt import build_episode_prompt, build_system_prompt


class VaxReplayDataset(RLDataset):
    def __init__(
        self,
        bundles: list[EpisodeBundle],
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        max_trajectory_tokens: int,
        max_generation_tokens: int,
        *,
        seed: int = 1234,
        num_epochs: int = 1,
    ):
        if not bundles:
            raise ValueError('VaxReplayDataset requires at least one episode')
        if batch_size <= 0 or group_size <= 0 or num_epochs <= 0:
            raise ValueError('batch_size, group_size, and num_epochs must be positive')
        if any(bundle.manifest.split == Split.TEST for bundle in bundles):
            raise ValueError('sealed test episodes cannot be loaded into an interactive RL dataset')
        for bundle in bundles:
            prompt_length = renderer.build_generation_prompt(
                [
                    {'role': 'system', 'content': build_system_prompt(bundle)},
                    {'role': 'user', 'content': build_episode_prompt(bundle)},
                ]
            ).length
            if prompt_length + max_generation_tokens > max_trajectory_tokens:
                raise ValueError(
                    f'episode {bundle.manifest.episode_id} prompt ({prompt_length}) plus generation reserve '
                    f'({max_generation_tokens}) exceeds max trajectory tokens ({max_trajectory_tokens})'
                )
        self._renderer = renderer
        self._batch_size = batch_size
        self._group_size = group_size
        self._max_trajectory_tokens = max_trajectory_tokens
        self._max_generation_tokens = max_generation_tokens
        self._bundles = bundles * num_epochs
        random.Random(seed).shuffle(self._bundles)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if index < 0 or index >= len(self):
            raise IndexError(f'batch index {index} is out of range')
        return [
            self._make_group_builder(bundle)
            for bundle in self._bundles[index * self._batch_size : (index + 1) * self._batch_size]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self._bundles) / self._batch_size)

    def _make_group_builder(self, bundle: EpisodeBundle) -> EnvGroupBuilder:
        return VaxReplayGroupBuilder(
            env_thunk=partial(
                make_tinker_env,
                renderer=self._renderer,
                bundle=bundle,
                max_trajectory_tokens=self._max_trajectory_tokens,
                max_generation_tokens=self._max_generation_tokens,
            ),
            num_envs=self._group_size,
            episode_id=bundle.manifest.episode_id,
        )


@dataclass(frozen=True)
class VaxReplayGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], Env]
    num_envs: int
    episode_id: str

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        _env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ['vax_replay']


@chz.chz
class VaxReplayDatasetBuilder(RLDatasetBuilder):
    train_episode_dirs: str
    dev_episode_dirs: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    reward_version: RewardVersion = REWARD_VERSION
    seed: int = 1234
    num_epochs: int = 1
    max_trajectory_tokens: int = -1
    max_generation_tokens: int = 2048
    dev_limit: int = -1

    async def __call__(self) -> tuple[VaxReplayDataset, VaxReplayDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train_bundles = load_episode_bundles(
            self.train_episode_dirs,
            Split.TRAIN,
            self.reward_version,
        )
        dev_bundles = load_episode_bundles(
            self.dev_episode_dirs,
            Split.DEV,
            self.reward_version,
        )
        validate_split_isolation(train_bundles, dev_bundles)
        if self.dev_limit > 0:
            dev_bundles = dev_bundles[: self.dev_limit]
        if not dev_bundles:
            raise ValueError('dev_limit removed every dev episode')

        if self.max_trajectory_tokens == -1:
            attributes = get_model_attributes(self.model_name_for_tokenizer)
            max_trajectory_tokens = attributes.context_window if attributes else 32 * 1024
        else:
            max_trajectory_tokens = self.max_trajectory_tokens

        return (
            VaxReplayDataset(
                bundles=train_bundles,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                max_trajectory_tokens=max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                seed=self.seed,
                num_epochs=self.num_epochs,
            ),
            VaxReplayDataset(
                bundles=dev_bundles,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=1,
                max_trajectory_tokens=max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                seed=self.seed,
            ),
        )
