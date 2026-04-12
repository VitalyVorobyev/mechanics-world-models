"""PyTorch sequence dataset for saved NPZ trajectory episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from data.image_transforms import ImageCropConfig, apply_image_crop_resize
from data.trajectory import Trajectory, load_trajectory_npz

SplitName = Literal["train", "val"]


@dataclass(frozen=True)
class EpisodeInfo:
    """Metadata for one saved trajectory episode."""

    path: Path
    episode_id: int
    num_steps: int
    observation_shape: tuple[int, int, int]
    action_shape: tuple[int, ...]


@dataclass(frozen=True)
class SequenceLocation:
    """Location of one fixed-length sequence within one episode file."""

    episode_index: int
    path: Path
    start: int


def index_episode_files(dataset_dir: Path) -> list[Path]:
    """Return episode NPZ files from a dataset directory in stable order."""

    paths = sorted(dataset_dir.glob("episode_*.npz"))
    if not paths:
        paths = sorted(dataset_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz episode files found in {dataset_dir}")
    return paths


def valid_sequence_starts(num_steps: int, sequence_length: int) -> list[int]:
    """Compute valid start indices for fixed-length transition sequences."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if num_steps < sequence_length:
        return []
    return list(range(num_steps - sequence_length + 1))


def split_episode_files(
    paths: list[Path],
    split: SplitName,
    val_fraction: float,
    seed: int,
) -> list[Path]:
    """Split episode files deterministically by whole episodes."""

    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")

    if not paths:
        return []

    rng = np.random.default_rng(seed)
    ordered_paths = [paths[index] for index in rng.permutation(len(paths))]

    val_count = int(round(len(ordered_paths) * val_fraction))
    if len(ordered_paths) > 1 and val_fraction > 0:
        val_count = max(1, min(val_count, len(ordered_paths) - 1))

    val_paths = sorted(ordered_paths[:val_count])
    train_paths = sorted(ordered_paths[val_count:])
    return train_paths if split == "train" else val_paths


def read_episode_info(path: Path) -> EpisodeInfo:
    """Read basic episode metadata from an NPZ trajectory file."""

    trajectory = load_trajectory_npz(path)
    return EpisodeInfo(
        path=path,
        episode_id=trajectory.episode_id,
        num_steps=trajectory.num_steps,
        observation_shape=tuple(int(dim) for dim in trajectory.images.shape[1:]),
        action_shape=action_shape_from_array(trajectory.actions),
    )


def action_shape_from_array(actions: np.ndarray) -> tuple[int, ...]:
    """Return the per-step action shape used by sequence tensors."""

    if actions.ndim == 1:
        return (1,)
    return tuple(int(dim) for dim in actions.shape[1:])


class EpisodeSequenceDataset(Dataset):
    """Dataset that returns fixed-length sequences from saved NPZ episodes."""

    def __init__(
        self,
        dataset_dir: Path,
        sequence_length: int,
        split: SplitName = "train",
        val_fraction: float = 0.1,
        seed: int = 0,
        max_episodes: int | None = None,
        crop_config: ImageCropConfig | None = None,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if max_episodes is not None and max_episodes < 1:
            raise ValueError("max_episodes must be >= 1 when set")

        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed
        self.max_episodes = max_episodes
        self.crop_config = crop_config or ImageCropConfig()

        all_paths = index_episode_files(self.dataset_dir)
        split_paths = split_episode_files(all_paths, split, val_fraction, seed)
        if max_episodes is not None:
            split_paths = split_paths[:max_episodes]
        if not split_paths:
            raise ValueError(
                f"Split '{split}' contains no episode files. "
                "Use more episodes or a different val_fraction.",
            )

        self._episodes: list[EpisodeInfo] = []
        self._skipped_short_episodes: list[EpisodeInfo] = []
        self._sequence_index: list[SequenceLocation] = []

        for path in split_paths:
            info = read_episode_info(path)
            starts = valid_sequence_starts(info.num_steps, sequence_length)
            if not starts:
                self._skipped_short_episodes.append(info)
                continue

            episode_index = len(self._episodes)
            self._episodes.append(info)
            self._sequence_index.extend(
                SequenceLocation(episode_index=episode_index, path=path, start=start)
                for start in starts
            )

        if not self._sequence_index:
            shortest = min((info.num_steps for info in self._skipped_short_episodes), default=0)
            longest = max((info.num_steps for info in self._skipped_short_episodes), default=0)
            raise ValueError(
                f"No usable episodes in split '{split}' for sequence_length={sequence_length}. "
                f"Skipped {len(self._skipped_short_episodes)} short episode(s); "
                f"available lengths range from {shortest} to {longest}.",
            )

    @property
    def episode_files(self) -> tuple[Path, ...]:
        """Episode files that contributed at least one sequence."""

        return tuple(info.path for info in self._episodes)

    @property
    def skipped_short_episodes(self) -> tuple[EpisodeInfo, ...]:
        """Episodes in this split that were shorter than ``sequence_length``."""

        return tuple(self._skipped_short_episodes)

    @property
    def sequence_index(self) -> tuple[SequenceLocation, ...]:
        """Deterministic index of every sequence returned by this dataset."""

        return tuple(self._sequence_index)

    def __len__(self) -> int:
        return len(self._sequence_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        location = self._sequence_index[index]
        trajectory = load_trajectory_npz(location.path)
        start = location.start
        end = start + self.sequence_length
        return _trajectory_slice_to_tensors(trajectory, start, end, crop_config=self.crop_config)


def _trajectory_slice_to_tensors(
    trajectory: Trajectory,
    start: int,
    end: int,
    crop_config: ImageCropConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a transition-aligned trajectory slice into PyTorch tensors."""

    images = apply_image_crop_resize(trajectory.images[start:end], crop_config)
    next_images = apply_image_crop_resize(trajectory.images[start + 1 : end + 1], crop_config)
    observations_np = images.astype(np.float32) / 255.0
    next_observations_np = next_images.astype(np.float32) / 255.0
    actions_np = np.asarray(trajectory.actions[start:end], dtype=np.float32)
    rewards_np = np.asarray(trajectory.rewards[start:end], dtype=np.float32)
    dones_np = np.asarray(trajectory.dones[start:end], dtype=np.bool_)

    if actions_np.ndim == 1:
        actions_np = actions_np[:, None]

    observations = torch.from_numpy(observations_np).permute(0, 3, 1, 2).contiguous()
    next_observations = torch.from_numpy(next_observations_np).permute(0, 3, 1, 2).contiguous()
    actions = torch.from_numpy(actions_np)
    rewards = torch.from_numpy(rewards_np)
    dones = torch.from_numpy(dones_np)

    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def make_train_val_datasets(
    dataset_dir: Path,
    sequence_length: int,
    val_fraction: float = 0.1,
    seed: int = 0,
    crop_config: ImageCropConfig | None = None,
) -> tuple[EpisodeSequenceDataset, EpisodeSequenceDataset]:
    """Build deterministic train and validation sequence datasets."""

    train = EpisodeSequenceDataset(
        dataset_dir=dataset_dir,
        sequence_length=sequence_length,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
        crop_config=crop_config,
    )
    val = EpisodeSequenceDataset(
        dataset_dir=dataset_dir,
        sequence_length=sequence_length,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
        crop_config=crop_config,
    )
    return train, val
