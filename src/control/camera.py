"""Calibrated camera geometry for cartpole (E-008).

dm_control's camera 0 sits at ``(0, -4, 1)`` and looks along +y, so the pole
plane (y = 0) is 4 m away and the image is an undistorted, axis-aligned view
of it: a world point ``(X, Z)`` lands at

    col = (W - 1) / 2 + f · X / 4,   row = (H - 1) / 2 − f · (Z − 1) / 4,

with focal length ``f = (H / 2) / tan(fovy / 2)`` pixels (25.35 px/m at the
default 45°, 84 px). The keypoints are the cart (pivot, ``(x, 1)``) and the
pole tip ``(x + sin θ, 1 + cos θ)`` (pole length 1 m). ``(x, θ)`` follow from
the two keypoints by this geometry alone, so a keypoint reader needs no
learned calibration. Keypoints use the ``KeypointEncoder`` convention:
``u, v`` in [-1, 1] over pixel centres, ``v`` downwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CAMERA_DISTANCE = 4.0
PIVOT_HEIGHT = 1.0
POLE_LENGTH = 1.0


@dataclass(frozen=True)
class CameraSpec:
    fovy_deg: float = 45.0
    image_size: int = 84

    @classmethod
    def covering(cls, half_width_m: float, image_size: int = 84) -> "CameraSpec":
        """The fovy that shows ``±half_width_m`` of the pole plane around the camera axis."""

        return cls(math.degrees(2.0 * math.atan(half_width_m / CAMERA_DISTANCE)), image_size)

    @property
    def focal_px(self) -> float:
        return (self.image_size / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)

    @property
    def pixels_per_metre(self) -> float:
        return self.focal_px / CAMERA_DISTANCE

    def keypoints(self, qpos: NDArray[np.float64]) -> NDArray[np.float64]:
        """``qpos [..., 2]`` -> keypoints ``[..., 2 (cart, tip), 2 (u, v)]`` in [-1, 1] units."""

        x, theta = qpos[..., 0], qpos[..., 1]
        world_x = np.stack([x, x + POLE_LENGTH * np.sin(theta)], axis=-1)
        world_z = np.stack([np.full_like(x, PIVOT_HEIGHT), PIVOT_HEIGHT + POLE_LENGTH * np.cos(theta)], axis=-1)
        half = (self.image_size - 1) / 2.0
        u = self.pixels_per_metre * world_x / half
        v = -self.pixels_per_metre * (world_z - PIVOT_HEIGHT) / half
        return np.stack([u, v], axis=-1)

    def pose(self, keypoints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Keypoints ``[..., 2, 2]`` -> ``(x, θ) [..., 2]`` with θ in (−π, π]."""

        metres = (self.image_size - 1) / 2.0 / self.pixels_per_metre
        cart, tip = keypoints[..., 0, :] * metres, keypoints[..., 1, :] * metres
        theta = np.arctan2(tip[..., 0] - cart[..., 0], cart[..., 1] - tip[..., 1])
        return np.stack([cart[..., 0], theta], axis=-1)

    def apply(self, physics) -> None:
        """Set camera 0's field of view on a dm_control ``Physics``."""

        physics.model.cam_fovy[0] = self.fovy_deg


# The whole rail (±1.8 m) plus the pole (1 m) plus a margin stays in view.
FULL_VIEW = CameraSpec.covering(2.9)


def render_states(qpos: NDArray[np.float64], spec: CameraSpec) -> NDArray[np.uint8]:
    """``qpos [N, 2]`` -> ``uint8 [N, H, W, 3]`` from camera 0 with ``spec``."""

    from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig

    env = DmControlPixelEnv(PixelEnvConfig(image_size=spec.image_size))
    physics = env.physics
    spec.apply(physics)
    images = np.empty((qpos.shape[0], spec.image_size, spec.image_size, 3), dtype=np.uint8)
    try:
        for i, q in enumerate(qpos):
            with physics.reset_context():
                physics.data.qpos[:] = q
            images[i] = physics.render(spec.image_size, spec.image_size, camera_id=0)
    finally:
        env.close()
    return images
