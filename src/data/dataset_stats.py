"""Dataset statistics utility for saved NPZ trajectory episodes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data.sequence_dataset import action_shape_from_array, index_episode_files, valid_sequence_starts
from data.trajectory import load_trajectory_npz


@dataclass(frozen=True)
class DatasetStats:
    """Summary statistics for a directory of saved episodes."""

    dataset_dir: str
    sequence_length: int
    episode_files_found: int
    usable_episodes: int
    total_frames: int
    total_valid_sequences: int
    observation_shape: tuple[int, ...] | None
    action_shape: tuple[int, ...] | None
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    episode_length_min: int
    episode_length_mean: float
    episode_length_max: int


def compute_dataset_stats(dataset_dir: Path, sequence_length: int) -> DatasetStats:
    """Compute summary stats for saved NPZ episodes and a sequence length."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    paths = index_episode_files(Path(dataset_dir))
    episode_lengths: list[int] = []
    rewards: list[np.ndarray] = []
    total_frames = 0
    total_valid_sequences = 0
    usable_episodes = 0
    observation_shape: tuple[int, ...] | None = None
    action_shape: tuple[int, ...] | None = None

    for path in paths:
        trajectory = load_trajectory_npz(path)
        num_steps = trajectory.num_steps
        starts = valid_sequence_starts(num_steps, sequence_length)

        episode_lengths.append(num_steps)
        rewards.append(trajectory.rewards)
        total_frames += int(trajectory.images.shape[0])
        total_valid_sequences += len(starts)
        usable_episodes += int(bool(starts))

        if observation_shape is None:
            observation_shape = tuple(int(dim) for dim in trajectory.images.shape[1:])
        if action_shape is None:
            action_shape = action_shape_from_array(trajectory.actions)

    reward_values = np.concatenate(rewards) if rewards else np.asarray([], dtype=np.float32)
    length_values = np.asarray(episode_lengths, dtype=np.float64)

    if reward_values.size:
        reward_mean = float(np.mean(reward_values))
        reward_std = float(np.std(reward_values))
        reward_min = float(np.min(reward_values))
        reward_max = float(np.max(reward_values))
    else:
        reward_mean = reward_std = reward_min = reward_max = float("nan")

    return DatasetStats(
        dataset_dir=str(Path(dataset_dir)),
        sequence_length=sequence_length,
        episode_files_found=len(paths),
        usable_episodes=usable_episodes,
        total_frames=total_frames,
        total_valid_sequences=total_valid_sequences,
        observation_shape=observation_shape,
        action_shape=action_shape,
        reward_mean=reward_mean,
        reward_std=reward_std,
        reward_min=reward_min,
        reward_max=reward_max,
        episode_length_min=int(np.min(length_values)),
        episode_length_mean=float(np.mean(length_values)),
        episode_length_max=int(np.max(length_values)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line parser for dataset statistics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    return parser


def main() -> None:
    """Run the dataset statistics CLI."""

    args = build_arg_parser().parse_args()
    stats = compute_dataset_stats(args.dataset_dir, args.sequence_length)
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
