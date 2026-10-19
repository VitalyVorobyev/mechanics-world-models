"""Keypoint pose reader trained on simulator renders (E-008).

A ``KeypointEncoder`` (spatial-softmax heatmaps, fully convolutional) is
supervised directly with the projected cart and pole-tip positions of
frames rendered at uniformly sampled states, so upright poses are as common
in training as hanging ones. ``(x, θ)`` is read from the two keypoints by the
calibrated camera geometry (``control.camera``); nothing else is learned.
One frame gives the pose only; velocities come from the estimator.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray

from control.camera import CameraSpec
from models.mechanics.keypoints import KeypointEncoder


def sample_states(n: int, seed: int, x_limit: float = 1.8) -> NDArray[np.float64]:
    """Uniform ``(x, θ)``: x in ±``x_limit`` m, θ in [−π, π)."""

    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-x_limit, x_limit, n), rng.uniform(-np.pi, np.pi, n)], axis=1)


def _images(batch: NDArray[np.uint8], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(batch, device=device).permute(0, 3, 1, 2).float() / 255.0


def train_reader(
    images: NDArray[np.uint8],
    keypoints: NDArray[np.float64],
    *,
    steps: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[KeypointEncoder, float]:
    """Regress ``keypoints [N, 2, 2]`` from ``images uint8 [N, H, W, 3]``; Adam with cosine decay."""

    torch.manual_seed(seed)
    dev = torch.device(device)
    model = KeypointEncoder(num_keypoints=2).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    targets = torch.as_tensor(keypoints, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    start, loss = time.time(), torch.tensor(float("nan"))
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(images), size=batch_size)
        pred, _, _ = model(_images(images[idx], dev))
        loss = (pred - targets[idx].to(dev)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % max(steps // 10, 1) == 0:
            log(f"keypoints step {step} loss {loss.item():.2e} ({time.time() - start:.0f}s)")
    return model.eval(), float(loss.detach())


@torch.no_grad()
def read_keypoints(model: KeypointEncoder, images: NDArray[np.uint8], device: str = "cpu", batch: int = 512) -> NDArray[np.float64]:
    out = []
    for begin in range(0, len(images), batch):
        kp, _, _ = model(_images(images[begin : begin + batch], torch.device(device)))
        out.append(kp.float().cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def read_poses(model: KeypointEncoder, images: NDArray[np.uint8], spec: CameraSpec, device: str = "cpu") -> NDArray[np.float64]:
    """``images uint8 [N, H, W, 3]`` -> ``(x, θ) [N, 2]`` with θ in (−π, π]."""

    return spec.pose(read_keypoints(model, images, device))


def pose_error_metrics(pred: NDArray[np.float64], true: NDArray[np.float64], spec: CameraSpec) -> dict[str, float]:
    """Wrapped angle errors (degrees) overall and upright (cos θ > 0.9), cart error in pixels."""

    dtheta = np.degrees(np.abs(np.angle(np.exp(1j * (pred[:, 1] - true[:, 1])))))
    dx_px = np.abs(pred[:, 0] - true[:, 0]) * spec.pixels_per_metre
    upright = np.cos(true[:, 1]) > 0.9
    return {
        "theta_median_deg": float(np.median(dtheta)),
        "theta_p90_deg": float(np.percentile(dtheta, 90)),
        "theta_max_deg": float(dtheta.max()),
        "theta_median_deg_upright": float(np.median(dtheta[upright])) if upright.any() else float("nan"),
        "theta_p90_deg_upright": float(np.percentile(dtheta[upright], 90)) if upright.any() else float("nan"),
        "x_median_px": float(np.median(dx_px)),
        "x_p90_px": float(np.percentile(dx_px, 90)),
    }
