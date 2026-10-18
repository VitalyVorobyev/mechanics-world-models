"""Poses read from rendered pixels by a trained encoder (rung 4 with a supervised front end).

The noisy context oracle uses Gaussian pose noise. A real encoder's errors
are not Gaussian: they depend on the pose (upright poses are rare and badly
read), and frames where the cart or pole tip leaves the image have no
answer at all (A-VIEW-1). This module renders every frame of state
trajectories with the collector's camera, reads ``(x, theta)`` with a pose
encoder from ``oracles.encoder``, and returns the episodes with
``observed_qpos`` (theta unwrapped along time) and ``valid`` (cart and pole
tip in view) filled in. The context oracle then trains, adapts and selects on
these poses exactly as on noisy ones.
"""

from __future__ import annotations

import numpy as np

from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
from oracles.common import StateEpisodes
from oracles.encoder import encode_poses, in_view


def render_states(qpos: np.ndarray, scales: dict[str, float] | None, image_size: int = 84) -> np.ndarray:
    """``qpos [N, 2]`` -> ``uint8 [N, H, W, 3]`` from camera 0 (velocities do not affect the image)."""

    env = DmControlPixelEnv(PixelEnvConfig(physics_param_scales=scales, image_size=image_size))
    physics = env._env.physics
    images = np.empty((qpos.shape[0], image_size, image_size, 3), dtype=np.uint8)
    try:
        for i, q in enumerate(qpos):
            with physics.reset_context():
                physics.data.qpos[:] = q
            images[i] = physics.render(image_size, image_size, camera_id=0)
    finally:
        env.close()
    return images


def with_encoder_poses(
    episodes: StateEpisodes, scales: dict[str, float] | None, encoder, *, device: str = "cpu",
    visible_only: bool = True,
) -> StateEpisodes:
    """Render, encode and attach ``observed_qpos`` (and ``valid`` if ``visible_only``)."""

    e, t = episodes.qpos.shape[:2]
    flat = episodes.qpos.reshape(-1, 2)
    poses = encode_poses(encoder, render_states(flat, scales), device=device).reshape(e, t, 2)
    poses[..., 1] = np.unwrap(poses[..., 1], axis=1)
    valid = in_view(flat).reshape(e, t) if visible_only else None
    return StateEpisodes(
        qpos=episodes.qpos, qvel=episodes.qvel, actions=episodes.actions, frame_dt=episodes.frame_dt,
        observed_qpos=poses, valid=valid,
    )


class PoseCalibration:
    """Oracle calibration: the encoder's median pose error per (theta, x) bin, fitted on true poses.

    Subtracting it removes the smooth, pose-dependent part of the encoder
    error (a distorted chart) and keeps the frame-to-frame residual. A
    diagnostic only: it uses ground truth.
    """

    def __init__(self, theta_bins: int = 24, x_bins: int = 8, x_range: float = 1.8) -> None:
        self.theta_bins, self.x_bins, self.x_range = theta_bins, x_bins, x_range
        self.table = np.zeros((theta_bins, x_bins, 2))

    def _bins(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.angle(np.exp(1j * qpos[..., 1]))
        tb = np.clip(((theta + np.pi) / (2 * np.pi) * self.theta_bins).astype(int), 0, self.theta_bins - 1)
        xb = np.clip(((qpos[..., 0] + self.x_range) / (2 * self.x_range) * self.x_bins).astype(int), 0, self.x_bins - 1)
        return tb, xb

    def fit(self, episodes: StateEpisodes, min_count: int = 10) -> "PoseCalibration":
        err = episodes.observed_qpos - episodes.qpos
        err[..., 1] = np.angle(np.exp(1j * err[..., 1]))
        valid = np.ones(err.shape[:-1], dtype=bool) if episodes.valid is None else episodes.valid
        tb, xb = self._bins(episodes.qpos)
        for i in range(self.theta_bins):
            for j in range(self.x_bins):
                mask = valid & (tb == i) & (xb == j)
                if mask.sum() >= min_count:
                    self.table[i, j] = np.median(err[mask], axis=0)
        return self

    def apply(self, episodes: StateEpisodes) -> StateEpisodes:
        """Episodes with the binned bias (looked up at the *true* pose) removed from ``observed_qpos``."""

        tb, xb = self._bins(episodes.qpos)
        return StateEpisodes(
            qpos=episodes.qpos, qvel=episodes.qvel, actions=episodes.actions, frame_dt=episodes.frame_dt,
            observed_qpos=episodes.observed_qpos - self.table[tb, xb], valid=episodes.valid,
        )
