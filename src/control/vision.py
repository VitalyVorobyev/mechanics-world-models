"""The camera path end to end (E-008): frames → keypoint poses → estimated state.

``CameraState`` is a ``state_fn`` for ``control.learned.run_episode``: it
renders camera 0 with the wide view, reads the pose with the keypoint reader,
and returns the fixed-lag estimate ``(q, q̇)``. It records what it read, so
the poses of an episode can become training data.

``observed_episodes`` turns state episodes into what a camera-only learner
sees: every frame rendered and read, θ unwrapped along time, and velocities
by central differences of the read poses (offline, so non-causal is fine).
These replace ``qpos`` / ``qvel`` for training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from control.camera import CameraSpec, render_states
from control.estimator import FixedLagEstimator
from control.keypoints import read_poses
from models.mechanics.keypoints import KeypointEncoder
from oracles.common import StateEpisodes


class CameraState:
    def __init__(self, reader: KeypointEncoder, spec: CameraSpec, estimator: FixedLagEstimator) -> None:
        self.reader, self.spec, self.estimator = reader, spec, estimator
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.poses: list[NDArray[np.float64]] = []
        self.estimates: list[NDArray[np.float64]] = []

    def __call__(self, env: Any, last_action: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.spec.apply(env.physics)
        image = env.physics.render(self.spec.image_size, self.spec.image_size, camera_id=0)
        pose = read_poses(self.reader, image[None], self.spec)[0]
        q, qdot = self.estimator.update(pose, last_action)
        self.poses.append(pose)
        self.estimates.append(np.concatenate([q, qdot]))
        return q, qdot


def unwrap_poses(poses: NDArray[np.float64]) -> NDArray[np.float64]:
    out = poses.copy()
    out[..., 1] = np.unwrap(out[..., 1], axis=-1)
    return out


def central_velocity(poses: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """``poses [E, T, 2]`` (unwrapped) -> velocities: central differences, one-sided at the ends."""

    return np.gradient(poses, dt, axis=1)


def observed_episodes(
    episodes: StateEpisodes, reader: KeypointEncoder, spec: CameraSpec, device: str = "cpu",
) -> StateEpisodes:
    """Render and read every frame; return episodes whose ``qpos`` / ``qvel`` are what the camera gives."""

    e, t = episodes.qpos.shape[:2]
    poses = read_poses(reader, render_states(episodes.qpos.reshape(-1, 2), spec), spec, device).reshape(e, t, 2)
    poses = np.stack([unwrap_poses(p) for p in poses])
    return StateEpisodes(
        qpos=poses,
        qvel=central_velocity(poses, episodes.frame_dt),
        actions=episodes.actions,
        frame_dt=episodes.frame_dt,
    )
