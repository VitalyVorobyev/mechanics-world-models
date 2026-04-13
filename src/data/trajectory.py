"""Trajectory dataclass and NumPy NPZ persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trajectory:
    """One episode fragment with images aligned for transition training.

    Optional ``qpos`` / ``qvel`` / ``physics_params`` carry simulator ground
    truth for evaluation diagnostics (linear probes, energy tracking, OOD
    grouping). They are *never* consumed by the training loss — the project
    contract is pixel-only learning, see ``CLAUDE.md`` and the project plan.

    The fields are optional so existing datasets that pre-date physics
    serialization still load through ``load_trajectory_npz``.
    """

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
    qpos: NDArray[np.float64] | None = None
    qvel: NDArray[np.float64] | None = None
    physics_params: dict[str, float] | None = field(default=None)

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

        # Physics ground truth is per-frame (T + 1) so it lines up with images.
        if self.qpos is not None or self.qvel is not None:
            if self.qpos is None or self.qvel is None:
                raise ValueError("qpos and qvel must be set together")
            if self.qpos.ndim != 2 or self.qvel.ndim != 2:
                raise ValueError("qpos/qvel must have shape (T + 1, D_q)")
            if self.qpos.shape[0] != self.num_steps + 1:
                raise ValueError(
                    "qpos must contain one entry per image frame "
                    f"(expected {self.num_steps + 1}, got {self.qpos.shape[0]})",
                )
            if self.qvel.shape[0] != self.num_steps + 1:
                raise ValueError(
                    "qvel must contain one entry per image frame "
                    f"(expected {self.num_steps + 1}, got {self.qvel.shape[0]})",
                )
            if self.qpos.dtype != np.float64 or self.qvel.dtype != np.float64:
                raise ValueError("qpos/qvel must have dtype float64 (MuJoCo native)")

        if self.physics_params is not None:
            for key, value in self.physics_params.items():
                if not isinstance(key, str):
                    raise ValueError("physics_params keys must be strings")
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        "physics_params values must be JSON-serializable scalars",
                    )


def save_trajectory_npz(trajectory: Trajectory, path: Path) -> Path:
    """Save a trajectory to a compressed NumPy archive.

    Optional ``qpos``, ``qvel``, ``physics_params`` are written only when
    present so older readers and older datasets stay compatible.
    """

    trajectory.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, NDArray | str] = {
        "env_name": np.asarray(trajectory.env_name),
        "seed": np.asarray(trajectory.seed, dtype=np.int64),
        "episode_id": np.asarray(trajectory.episode_id, dtype=np.int64),
        "action_repeat": np.asarray(trajectory.action_repeat, dtype=np.int64),
        "image_size": np.asarray(trajectory.image_size, dtype=np.int64),
        "images": trajectory.images,
        "actions": trajectory.actions,
        "rewards": trajectory.rewards,
        "discounts": trajectory.discounts,
        "dones": trajectory.dones,
        "step_indices": trajectory.step_indices,
    }
    if trajectory.qpos is not None and trajectory.qvel is not None:
        payload["qpos"] = trajectory.qpos
        payload["qvel"] = trajectory.qvel
    if trajectory.physics_params is not None:
        payload["physics_params_json"] = np.asarray(
            json.dumps(trajectory.physics_params, sort_keys=True),
        )
    np.savez_compressed(path, **payload)
    return path


def load_trajectory_npz(path: Path) -> Trajectory:
    """Load a trajectory from a NumPy archive.

    Tolerant of both the legacy schema (no physics fields) and the current
    schema (with optional ``qpos``/``qvel``/``physics_params_json``).
    """

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        qpos: NDArray[np.float64] | None = (
            np.asarray(data["qpos"], dtype=np.float64) if "qpos" in keys else None
        )
        qvel: NDArray[np.float64] | None = (
            np.asarray(data["qvel"], dtype=np.float64) if "qvel" in keys else None
        )
        physics_params: dict[str, float] | None = None
        if "physics_params_json" in keys:
            raw = str(data["physics_params_json"].item())
            physics_params = {str(k): float(v) for k, v in json.loads(raw).items()}
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
            qpos=qpos,
            qvel=qvel,
            physics_params=physics_params,
        )
    trajectory.validate()
    return trajectory
