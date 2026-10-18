"""Label-free pose front end: classical-CV initial poses, then render-and-compare EM (A-FE-1).

Gradient training of an encoder through the renderer cannot find the pole
angle: the reconstruction loss is flat beyond about 10° of error. A grid
search through a renderer fitted on true poses finds it (0.5° median). This
module removes the true poses from that recipe:

1. ``heuristic_poses``: cart and pole share one colour (orange, R - B > 50)
   that neither the bluish background nor the dark shadows have. The cart is
   the wide part of that mask in its fixed row band; the pole angle is the
   direction from the pivot to the centroid of the remaining mask pixels.
   Scene knowledge used: the pivot row (41.5) and the camera scale
   (25.35 px/m) to express x in metres. No pose labels.
2. EM: fit the geometric renderer on the current poses, re-estimate poses by
   ``search_poses`` (render-and-compare), repeat.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from models.losses import foreground_reconstruction_mask
from oracles.common import Criterion, OracleResult
from oracles.decoder import _PoseDecoder, _to_images
from oracles.encoder import PIXELS_PER_METRE, pose_errors
from oracles.frozen_render import _reconstruction_loss

PIVOT_ROW = 41.5
CART_ROWS = (38, 46)  # row band of the cart body (exclusive end)


def heuristic_poses(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``uint8 [N, H, W, 3]`` -> ``(x metres, theta) [N, 2]`` and a found-mask ``[N]``."""

    rgb = images.astype(np.int16)
    mask = (rgb[..., 0] - rgb[..., 2] > 50) & (rgb[..., 0] > 100)  # cart and pole colour
    n, height, width = mask.shape
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    poses = np.zeros((n, 2))
    found = np.zeros(n, dtype=bool)
    lo, hi = CART_ROWS
    for i in range(n):
        band = mask[i, lo:hi]
        cart_cols = np.nonzero(band.sum(axis=0) >= 4)[0]  # the cart is taller than the 2-px pole
        if len(cart_cols) == 0:
            continue
        u = cart_cols.mean()
        pole = mask[i] & ~((rows >= lo) & (rows < hi) & (np.abs(cols - u) <= 7))
        if pole.sum() < 3:
            continue
        pr, pc = np.nonzero(pole)
        theta = math.atan2(pc.mean() - u, -(pr.mean() - PIVOT_ROW))
        poses[i] = ((u - (width - 1) / 2) / PIXELS_PER_METRE, theta)
        found[i] = True
    return poses, found


def _per_frame_loss(decoder: _PoseDecoder, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = decoder(q)
    se = (pred - target).square()
    mask = foreground_reconstruction_mask(target[:, None])[:, 0]
    return se.mean((1, 2, 3)) + (se * mask).mean((1, 2, 3))


@torch.no_grad()
def search_poses(decoder: _PoseDecoder, images: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Render-and-compare: x (0.05 m grid, 8 coarse angles), then theta (5°), then a local 7×7 refinement."""

    dev = torch.device(device)
    xs = torch.linspace(-2.0, 2.0, 81, device=dev)
    coarse = torch.linspace(-math.pi, math.pi, 9, device=dev)[:-1]
    thetas = torch.linspace(-math.pi, math.pi, 73, device=dev)[:-1]
    grid_x, grid_t = torch.meshgrid(xs, coarse, indexing="ij")
    stage1 = torch.stack([grid_x.flatten(), grid_t.flatten()], -1)
    offsets = torch.stack(
        torch.meshgrid(torch.linspace(-0.06, 0.06, 7, device=dev), torch.linspace(-0.09, 0.09, 7, device=dev), indexing="ij"),
        -1,
    ).reshape(-1, 2)
    out = np.zeros((len(images), 2))
    for i in range(len(images)):
        target = _to_images(images[i : i + 1], dev)
        loss = _per_frame_loss(decoder, stage1, target.expand(len(stage1), -1, -1, -1))
        x0 = xs[loss.reshape(len(xs), len(coarse)).min(1).values.argmin()]
        q = torch.stack([x0.expand(len(thetas)), thetas], -1)
        t0 = thetas[_per_frame_loss(decoder, q, target.expand(len(q), -1, -1, -1)).argmin()]
        q = torch.stack([x0, t0]) + offsets
        out[i] = q[_per_frame_loss(decoder, q, target.expand(len(q), -1, -1, -1)).argmin()].cpu().numpy()
    return out


def fit_renderer(
    images: np.ndarray, poses: np.ndarray, *, steps: int, batch_size: int = 64, lr: float = 1e-3,
    device: str = "cpu", seed: int = 0,
) -> _PoseDecoder:
    torch.manual_seed(seed)
    dev = torch.device(device)
    decoder = _PoseDecoder("render").to(dev)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(images), size=batch_size)
        target = _to_images(images[idx], dev)
        q = torch.as_tensor(poses[idx], dtype=torch.float32, device=dev)
        loss = _reconstruction_loss(decoder(q), target, 1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    decoder.eval()
    return decoder


def _scored(poses: np.ndarray, qpos: np.ndarray) -> dict[str, float]:
    pred = np.stack([poses[:, 0], np.cos(poses[:, 1]), np.sin(poses[:, 1])], axis=-1)
    errors = pose_errors(pred, qpos)
    flipped = np.abs(np.angle(np.exp(1j * (poses[:, 1] - qpos[:, 1])))) > np.pi / 2
    return {**errors, "theta_flipped_fraction": float(flipped.mean())}


def run_bootstrap_oracle(
    train_images: np.ndarray,
    val_images: np.ndarray,
    val_qpos: np.ndarray,
    *,
    em_iterations: int = 2,
    em_frames: int = 3000,
    renderer_steps: int = 3000,
    device: str = "cpu",
    seed: int = 0,
    log=print,
) -> OracleResult:
    """Criterion: after EM, median angle error <= 1° on in-view held-out frames. Labels only score."""

    metrics: dict[str, float] = {}
    start = time.time()
    poses, found = heuristic_poses(train_images)
    val_heuristic, val_found = heuristic_poses(val_images)
    for name, value in _scored(val_heuristic, val_qpos).items():
        metrics[f"heuristic.{name}"] = value
    metrics["heuristic.found_fraction"] = float(val_found.mean())
    log(
        f"heuristic: found {found.mean():.2f} of train frames; held-out in-view θ median "
        f"{metrics['heuristic.visible.theta_err_deg_median']:.2f}° ({time.time() - start:.0f}s)",
    )
    images, current = train_images[found], poses[found]
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(images), size=min(em_frames, len(images)), replace=False)
    for iteration in range(em_iterations + 1):
        decoder = fit_renderer(images, current, steps=renderer_steps, device=device, seed=seed + iteration)
        if iteration == em_iterations:
            break
        searched = search_poses(decoder, images[subset], device=device)
        images, current = images[subset], searched
        subset = np.arange(len(images))
        log(f"EM iteration {iteration + 1}: renderer fitted, {len(images)} poses re-searched ({time.time() - start:.0f}s)")
    val_search = search_poses(decoder, val_images, device=device)
    for name, value in _scored(val_search, val_qpos).items():
        metrics[f"em.{name}"] = value
    log(f"EM: held-out in-view θ median {metrics['em.visible.theta_err_deg_median']:.2f}° ({time.time() - start:.0f}s)")
    return OracleResult(
        name="bootstrap_oracle",
        hypotheses=["A-FE-1"],
        question="Can a label-free front end (CV heuristic + render-and-compare EM) recover the pose?",
        config={
            "em_iterations": em_iterations,
            "em_frames": em_frames,
            "renderer_steps": renderer_steps,
            "train_frames": int(len(train_images)),
            "val_frames": int(len(val_images)),
            "device": device,
            "seed": seed,
        },
        metrics=metrics,
        criteria=[Criterion("em.visible.theta_err_deg_median", "<=", 1.0, "label-free, in-view held-out frames")],
        notes=["Pose labels are used only for scoring; x is metric via the camera scale 25.35 px/m."],
    )
