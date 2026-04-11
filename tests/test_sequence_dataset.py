from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from data import EpisodeSequenceDataset, Trajectory, compute_dataset_stats, save_trajectory_npz


def _write_episode(
    root: Path,
    episode_id: int,
    num_steps: int,
    image_size: int = 8,
    action_dim: int = 2,
    image_value: int | None = None,
) -> Path:
    frame_values = np.arange(num_steps + 1, dtype=np.uint8)
    if image_value is not None:
        frame_values = np.full(num_steps + 1, image_value, dtype=np.uint8)
    images = np.repeat(frame_values[:, None, None, None], image_size, axis=1)
    images = np.repeat(images, image_size, axis=2)
    images = np.repeat(images, 3, axis=3)

    actions = np.full((num_steps, action_dim), episode_id, dtype=np.float32)
    rewards = np.arange(num_steps, dtype=np.float32) + float(episode_id)
    dones = np.zeros(num_steps, dtype=np.bool_)
    if num_steps:
        dones[-1] = True

    trajectory = Trajectory(
        env_name="cartpole-swingup",
        seed=0,
        episode_id=episode_id,
        action_repeat=2,
        image_size=image_size,
        images=images,
        actions=actions,
        rewards=rewards,
        discounts=np.ones(num_steps, dtype=np.float32),
        dones=dones,
        step_indices=np.arange(num_steps, dtype=np.int64),
    )
    return save_trajectory_npz(trajectory, root / f"episode_{episode_id:06d}.npz")


def test_sequence_dataset_returns_expected_shapes_dtypes_and_normalized_images(tmp_path) -> None:
    _write_episode(tmp_path, episode_id=0, num_steps=5, image_value=255)

    dataset = EpisodeSequenceDataset(
        dataset_dir=tmp_path,
        sequence_length=3,
        split="train",
        val_fraction=0.0,
        seed=0,
    )
    sample = dataset[0]

    assert sample["observations"].shape == (3, 3, 8, 8)
    assert sample["actions"].shape == (3, 2)
    assert sample["rewards"].shape == (3,)
    assert sample["dones"].shape == (3,)

    assert sample["observations"].dtype == torch.float32
    assert sample["actions"].dtype == torch.float32
    assert sample["rewards"].dtype == torch.float32
    assert sample["dones"].dtype == torch.bool

    assert torch.all(sample["observations"] >= 0.0)
    assert torch.all(sample["observations"] <= 1.0)
    assert torch.allclose(sample["observations"], torch.ones_like(sample["observations"]))


def test_train_val_split_is_deterministic_for_same_seed(tmp_path) -> None:
    for episode_id in range(8):
        _write_episode(tmp_path, episode_id=episode_id, num_steps=4)

    train_a = EpisodeSequenceDataset(tmp_path, sequence_length=2, split="train", val_fraction=0.25, seed=123)
    train_b = EpisodeSequenceDataset(tmp_path, sequence_length=2, split="train", val_fraction=0.25, seed=123)
    val_a = EpisodeSequenceDataset(tmp_path, sequence_length=2, split="val", val_fraction=0.25, seed=123)
    val_b = EpisodeSequenceDataset(tmp_path, sequence_length=2, split="val", val_fraction=0.25, seed=123)

    assert [path.name for path in train_a.episode_files] == [path.name for path in train_b.episode_files]
    assert [path.name for path in val_a.episode_files] == [path.name for path in val_b.episode_files]
    assert train_a.sequence_index == train_b.sequence_index
    assert val_a.sequence_index == val_b.sequence_index


def test_train_val_split_changes_for_different_seed(tmp_path) -> None:
    for episode_id in range(10):
        _write_episode(tmp_path, episode_id=episode_id, num_steps=4)

    val_a = EpisodeSequenceDataset(tmp_path, sequence_length=2, split="val", val_fraction=0.3, seed=1)
    val_b = EpisodeSequenceDataset(tmp_path, sequence_length=2, split="val", val_fraction=0.3, seed=2)

    assert [path.name for path in val_a.episode_files] != [path.name for path in val_b.episode_files]


def test_sequences_do_not_cross_episode_boundaries(tmp_path) -> None:
    _write_episode(tmp_path, episode_id=10, num_steps=5)
    _write_episode(tmp_path, episode_id=20, num_steps=5)

    dataset = EpisodeSequenceDataset(
        dataset_dir=tmp_path,
        sequence_length=4,
        split="train",
        val_fraction=0.0,
        seed=0,
    )

    assert len(dataset) == 4
    for sample in dataset:
        assert torch.unique(sample["actions"]).numel() == 1
        assert sample["dones"][:-1].sum().item() == 0


def test_max_episodes_limits_split_after_deterministic_split(tmp_path) -> None:
    for episode_id in range(6):
        _write_episode(tmp_path, episode_id=episode_id, num_steps=4)

    full = EpisodeSequenceDataset(
        dataset_dir=tmp_path,
        sequence_length=2,
        split="train",
        val_fraction=0.0,
        seed=0,
    )
    limited = EpisodeSequenceDataset(
        dataset_dir=tmp_path,
        sequence_length=2,
        split="train",
        val_fraction=0.0,
        seed=0,
        max_episodes=2,
    )

    assert len(limited.episode_files) == 2
    assert limited.episode_files == full.episode_files[:2]


def test_short_episodes_are_rejected_explicitly(tmp_path) -> None:
    _write_episode(tmp_path, episode_id=0, num_steps=2)

    with pytest.raises(ValueError, match="No usable episodes"):
        EpisodeSequenceDataset(
            dataset_dir=tmp_path,
            sequence_length=3,
            split="train",
            val_fraction=0.0,
            seed=0,
        )

    stats = compute_dataset_stats(tmp_path, sequence_length=3)
    assert stats.episode_files_found == 1
    assert stats.usable_episodes == 0
    assert stats.total_valid_sequences == 0
    assert stats.observation_shape == (8, 8, 3)
    assert stats.action_shape == (2,)
