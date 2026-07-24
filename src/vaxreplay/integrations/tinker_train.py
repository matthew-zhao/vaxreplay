"""GRPO entry point for published VaxReplay train/dev episodes."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

from vaxreplay.case_schema import REWARD_VERSION, RewardVersion
from vaxreplay.integrations.tinker_dataset import VaxReplayDatasetBuilder


@chz.chz
class CLIConfig:
    train_episode_dirs: str
    dev_episode_dirs: str
    model_name: str = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
    reward_version: RewardVersion = REWARD_VERSION
    checkpoint_path: str | None = None
    renderer_name: str | None = None
    batch_size: int = 16
    group_size: int = 8
    num_epochs: int = 1
    seed: int = 2
    learning_rate: float = 4e-5
    lora_rank: int = 32
    max_tokens: int = 2048
    max_trajectory_tokens: int = -1
    eval_every: int = 10
    dev_limit: int = -1
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = 'ask'


async def cli_main(config: CLIConfig) -> None:
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    reward_profile = config.reward_version.split('.')[0]
    run_name = config.wandb_name or f'vaxreplay-{reward_profile}-{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
    log_path = config.log_path or f'/tmp/tinker-examples/vax_replay/{run_name}'
    if not Path('/tmp').exists():
        raise ValueError('/tmp does not exist')
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    dataset_builder = VaxReplayDatasetBuilder(
        train_episode_dirs=config.train_episode_dirs,
        dev_episode_dirs=config.dev_episode_dirs,
        batch_size=config.batch_size,
        group_size=config.group_size,
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        reward_version=config.reward_version,
        seed=config.seed,
        num_epochs=config.num_epochs,
        max_trajectory_tokens=config.max_trajectory_tokens,
        max_generation_tokens=config.max_tokens,
        dev_limit=config.dev_limit,
    )
    await train.main(
        train.Config(
            model_name=config.model_name,
            load_checkpoint_path=config.checkpoint_path,
            log_path=log_path,
            dataset_builder=dataset_builder,
            learning_rate=config.learning_rate,
            max_tokens=config.max_tokens,
            renderer_name=renderer_name,
            eval_every=config.eval_every,
            save_every=config.eval_every,
            wandb_project=config.wandb_project,
            wandb_name=run_name,
            lora_rank=config.lora_rank,
        )
    )


if __name__ == '__main__':
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
