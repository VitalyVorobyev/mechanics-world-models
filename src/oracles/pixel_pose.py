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
