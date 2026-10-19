"""Rung 1: can a decoder render cartpole frames from the *true* pose?

Trains only a decoder, from ground-truth ``(x, theta)`` (no encoder, no
dynamics), and reports held-out foreground / full-frame reconstruction
error against the trivial mean-image floor. If a decoder cannot render from
the true state, no end-to-end training will make it render from a learned
state.

Decoders:

* ``deconv``: the project's ``ImageDecoder`` fed ``(x, cos theta, sin theta)``.
* ``render``: ``GeometricRenderer`` fed ``(a * x + b, theta)`` with a learned
  affine ``a, b`` from metres to image units (theta needs no map: dm_control
  and the renderer share the convention "positive theta tilts the tip to
  +u", checked 2026-10-18).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from models.decoder import ImageDecoder
from models.losses import foreground_reconstruction_mask
from models.mechanics.keypoints import GeometricRenderer
from oracles.common import Criterion, OracleResult

# Measured on the default camera: 1 m of cart travel ~ 0.62 normalized image units.
METRES_TO_IMAGE_U = 0.62


class _PoseDecoder(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "deconv":
            self.net = ImageDecoder(feature_size=3, output_size=84)
        elif kind == "render":
            self.net = GeometricRenderer(pole_length_init=METRES_TO_IMAGE_U)
            self.scale = nn.Parameter(torch.tensor(METRES_TO_IMAGE_U))
            self.offset = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(kind)

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        """``qpos [N, 2]`` -> images ``[N, 3, 84, 84]`` in ``[0, 1]``."""

        if self.kind == "deconv":
            features = torch.stack([qpos[:, 0], torch.cos(qpos[:, 1]), torch.sin(qpos[:, 1])], dim=-1)
            return self.net(features[:, None])[:, 0]
        q_image = torch.stack([self.scale * qpos[:, 0] + self.offset, qpos[:, 1]], dim=-1)
        return torch.sigmoid(self.net(q_image))


def _to_images(batch_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(batch_uint8, device=device).permute(0, 3, 1, 2).float() / 255.0


def _masked_errors(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    mask = foreground_reconstruction_mask(target[:, None])[:, 0]
    se = (pred - target).square()
    return float((se * mask).mean()), float(se.mean())


def run_decoder_oracle(
    train_images: np.ndarray,
    train_qpos: np.ndarray,
    val_images: np.ndarray,
    val_qpos: np.ndarray,
    *,
    kind: str,
    steps: int = 3000,
    batch_size: int = 64,
    lr: float = 1e-3,
    foreground_weight: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
    log=print,
) -> OracleResult:
    """Fit a pose->image decoder; criterion: held-out fg MSE <= 30% of the mean-image floor."""

    torch.manual_seed(seed)
    dev = torch.device(device)
    model = _PoseDecoder(kind).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    start = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train_images), size=batch_size)
        target = _to_images(train_images[idx], dev)
        qpos = torch.as_tensor(train_qpos[idx], dtype=torch.float32, device=dev)
        pred = model(qpos)
        mask = foreground_reconstruction_mask(target[:, None])[:, 0].detach()
        se = (pred - target).square()
        loss = se.mean() + foreground_weight * (se * mask).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(steps // 10, 1) == 0:
            log(f"[{kind}] step {step} loss {float(loss):.5f} ({time.time() - start:.0f}s)")

    model.eval()
    fg_errors, full_errors, floor_fg, floor_full = [], [], [], []
    mean_image = _to_images(train_images[rng.integers(0, len(train_images), size=512)], dev).mean(0)
    with torch.no_grad():
        for begin in range(0, len(val_images), 128):
            target = _to_images(val_images[begin : begin + 128], dev)
            qpos = torch.as_tensor(val_qpos[begin : begin + 128], dtype=torch.float32, device=dev)
            fg, full = _masked_errors(model(qpos), target)
            fg_errors.append(fg)
            full_errors.append(full)
            fg0, full0 = _masked_errors(mean_image.expand_as(target), target)
            floor_fg.append(fg0)
            floor_full.append(full0)
    metrics = {
        "val_fg_mse": float(np.mean(fg_errors)),
        "val_full_mse": float(np.mean(full_errors)),
        "mean_image_fg_mse": float(np.mean(floor_fg)),
        "mean_image_full_mse": float(np.mean(floor_full)),
    }
    metrics["fg_mse_over_floor"] = metrics["val_fg_mse"] / metrics["mean_image_fg_mse"]
    if kind == "render":
        metrics["learned_scale"] = float(model.scale)
        metrics["learned_offset"] = float(model.offset)
        metrics["learned_pole_length"] = float(model.net.pole_length)
        metrics["learned_cart_v"] = float(model.net.cart_v)
    return OracleResult(
        name=f"decoder_oracle_{kind}",
        hypotheses=["A-DEC-1", "A-FE-1"],
        question=f"Can the {kind} decoder render cartpole frames from the true pose?",
        config={
            "kind": kind,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "foreground_weight": foreground_weight,
            "train_frames": int(len(train_images)),
            "val_frames": int(len(val_images)),
            "device": device,
            "seed": seed,
        },
        metrics=metrics,
        criteria=[
            Criterion(
                "fg_mse_over_floor", "<=", 0.3,
                "held-out foreground error at most 30% of the mean-image floor",
            ),
        ],
        notes=["Foreground mask: background-subtracted saliency mask, computed per batch."],
    )
