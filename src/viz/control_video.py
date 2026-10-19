"""Videos of closed-loop episodes: what the controller sees next to the simulation.

Left panel: the 84 px camera frame the keypoint reader gets (upscaled), with
the read cart and pole-tip keypoints drawn on it when the episode came from
the camera pipeline. Right panel: the same state rendered at higher
resolution with the same camera. A caption shows time and the running return.
"""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from control.camera import CameraSpec, render_states


def _to_pixels(keypoints: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    return (keypoints + 1.0) * (size - 1) / 2.0


def episode_frames(
    qpos: NDArray[np.float64],
    spec: CameraSpec,
    *,
    poses: NDArray[np.float64] | None = None,
    rewards: NDArray[np.float64] | None = None,
    title: str = "",
    scale: int = 3,
    hires: int = 256,
) -> list[NDArray[np.uint8]]:
    """``qpos [T, 2]`` (and optional read ``poses [T, 2]``, ``rewards [T]``) -> RGB frames."""

    small = render_states(qpos, spec)
    large = render_states(qpos, CameraSpec(spec.fovy_deg, hires))
    side = spec.image_size * scale
    returns = np.concatenate([[0.0], np.cumsum(rewards)]) if rewards is not None else None
    frames = []
    for t in range(len(qpos)):
        left = Image.fromarray(small[t]).resize((side, side), Image.NEAREST)
        draw = ImageDraw.Draw(left)
        if poses is not None and t < len(poses):
            cart, tip = _to_pixels(spec.keypoints(poses[t]), side)
            draw.line([tuple(cart), tuple(tip)], fill=(80, 255, 120), width=2)
            for point in (cart, tip):
                draw.ellipse([point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4], outline=(80, 255, 120), width=2)
        right = Image.fromarray(large[t]).resize((side, side), Image.BILINEAR)
        canvas = Image.new("RGB", (2 * side, side + 28), (16, 16, 20))
        canvas.paste(left, (0, 28)), canvas.paste(right, (side, 28))
        caption = f"{title}   t = {t * 0.02:5.2f} s"
        if returns is not None:
            caption += f"   return {returns[min(t, len(returns) - 1)]:6.1f}"
        ImageDraw.Draw(canvas).text((8, 8), caption, fill=(230, 230, 230))
        frames.append(np.asarray(canvas))
    return frames


def write_video(frames: list[NDArray[np.uint8]], path: Path, fps: int = 50) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path
