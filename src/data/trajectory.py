"""Trajectory dataclass and NumPy NPZ persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trajectory:
    """One episode fragment with images aligned for transition training."""

    env_name: str
    seed: int
    episode_id: int
    action_repeat: int
    image_size: int
    images: NDArray[np.uint8]
    actions: NDArray[np.floating]
    rewards: NDArray[np.float32]
    discounts: NDArray[np.float32]
    dones: NDArray[np.bool_]
    step_indices: NDArray[np.int64]

    @property
    def num_steps(self) -> int:
        """Number of transitions in this trajectory."""

        return int(self.actions.shape[0])

    def validate(self) -> None:
        """Validate basic shape and dtype invariants before writing."""

        if self.actions.ndim < 1:
            raise ValueError("actions must have at least one leading time dimension")
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError("images must have shape (T + 1, H, W, 3)")
        if self.images.dtype != np.uint8:
            raise ValueError("images must have dtype uint8")
        if self.images.shape[1:3] != (self.image_size, self.image_size):
            raise ValueError("images spatial shape must match image_size")
        if self.images.shape[0] != self.num_steps + 1:
            raise ValueError("images must contain one more frame than transition arrays")
        if not np.issubdtype(self.actions.dtype, np.floating):
            raise ValueError("actions must use a floating dtype")

        expected = (self.num_steps,)
        for name, array in (
            ("rewards", self.rewards),
            ("discounts", self.discounts),
            ("dones", self.dones),
            ("step_indices", self.step_indices),
        ):
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {array.shape}")

        if self.rewards.dtype != np.float32:
            raise ValueError("rewards must have dtype float32")
        if self.discounts.dtype != np.float32:
            raise ValueError("discounts must have dtype float32")
        if self.dones.dtype != np.bool_:
            raise ValueError("dones must have dtype bool")
        if self.step_indices.dtype != np.int64:
            raise ValueError("step_indices must have dtype int64")


def save_trajectory_npz(trajectory: Trajectory, path: Path) -> Path:
    """Save a trajectory to a compressed NumPy archive."""

    trajectory.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        env_name=np.asarray(trajectory.env_name),
        seed=np.asarray(trajectory.seed, dtype=np.int64),
        episode_id=np.asarray(trajectory.episode_id, dtype=np.int64),
        action_repeat=np.asarray(trajectory.action_repeat, dtype=np.int64),
        image_size=np.asarray(trajectory.image_size, dtype=np.int64),
        images=trajectory.images,
        actions=trajectory.actions,
        rewards=trajectory.rewards,
        discounts=trajectory.discounts,
        dones=trajectory.dones,
        step_indices=trajectory.step_indices,
    )
    return path


def load_trajectory_npz(path: Path) -> Trajectory:
    """Load a trajectory from a NumPy archive."""

    with np.load(path, allow_pickle=False) as data:
        trajectory = Trajectory(
            env_name=str(data["env_name"].item()),
            seed=int(data["seed"].item()),
            episode_id=int(data["episode_id"].item()),
            action_repeat=int(data["action_repeat"].item()),
            image_size=int(data["image_size"].item()),
            images=np.asarray(data["images"], dtype=np.uint8),
            actions=np.asarray(data["actions"]),
            rewards=np.asarray(data["rewards"], dtype=np.float32),
            discounts=np.asarray(data["discounts"], dtype=np.float32),
            dones=np.asarray(data["dones"], dtype=np.bool_),
            step_indices=np.asarray(data["step_indices"], dtype=np.int64),
        )
    trajectory.validate()
    return trajectory
