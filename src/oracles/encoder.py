"""Rung 2: can an encoder recover the *true* pose from pixels?

Supervised regression from single frames onto ``(x, cos theta, sin theta)``
with an episode-level held-out split. Gives the ceiling for what any
learned latent can carry, and tests whether a given encoder architecture
can localize the cart and pole at all.

Encoders:

* ``conv``: the project's ``ConvEncoder`` trunk + linear head.
* ``keypoint``: ``KeypointEncoder`` -> ``(x_img, theta)`` by the cartpole
  geometry, then a learned affine ``x = (x_img - b) / a`` — i.e. the pose is
  read *only* through the keypoint geometry, exactly as in the keypoint
  world model.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from models.encoder import ConvEncoder
from models.mechanics.keypoints import KeypointEncoder, keypoints_to_cartpole_q
from oracles.common import Criterion, OracleResult, pose_features
from oracles.decoder import METRES_TO_IMAGE_U


class _PoseEncoder(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "conv":
            self.net = ConvEncoder(embedding_size=256)
            self.head = nn.Linear(256, 3)
        elif kind == "keypoint":
            self.net = KeypointEncoder(num_keypoints=2)
            self.scale = nn.Parameter(torch.tensor(METRES_TO_IMAGE_U))
            self.offset = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(kind)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images [N, 3, 84, 84]`` -> ``(x, cos, sin) [N, 3]``."""

        if self.kind == "conv":
            return self.head(self.net(images[:, None])[:, 0])
        keypoints, _, _ = self.net(images)
        q = keypoints_to_cartpole_q(keypoints)
        x = (q[:, 0] - self.offset) / self.scale
        return torch.stack([x, torch.cos(q[:, 1]), torch.sin(q[:, 1])], dim=-1)


def _images(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(batch, device=device).permute(0, 3, 1, 2).float() / 255.0


def _r2(pred: np.ndarray, target: np.ndarray) -> tuple[float, list[float]]:
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    return float(1 - ss_res.sum() / ss_tot.sum()), [float(v) for v in 1 - ss_res / ss_tot]


def run_encoder_oracle(
    train_images: np.ndarray,
    train_qpos: np.ndarray,
    val_images: np.ndarray,
    val_qpos: np.ndarray,
    *,
    kind: str,
    steps: int = 3000,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    log=print,
) -> OracleResult:
    """Fit pixels -> pose; criterion: held-out R² >= 0.95 on (x, cos, sin)."""

    torch.manual_seed(seed)
    dev = torch.device(device)
    model = _PoseEncoder(kind).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    train_targets = pose_features(train_qpos).astype(np.float32)
    start = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train_images), size=batch_size)
        pred = model(_images(train_images[idx], dev))
        loss = (pred - torch.as_tensor(train_targets[idx], device=dev)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(steps // 10, 1) == 0:
            log(f"[{kind}] step {step} loss {float(loss):.5f} ({time.time() - start:.0f}s)")
    model.eval()
    preds = []
    with torch.no_grad():
        for begin in range(0, len(val_images), 256):
            preds.append(model(_images(val_images[begin : begin + 256], dev)).cpu().numpy())
    pred = np.concatenate(preds)
    overall, per_dim = _r2(pred, pose_features(val_qpos))
    metrics = {
        "val_r2_overall": overall,
        "val_r2_x": per_dim[0],
        "val_r2_cos": per_dim[1],
        "val_r2_sin": per_dim[2],
        "final_train_loss": float(loss),
    }
    if kind == "keypoint":
        metrics["learned_scale"] = float(model.scale)
        metrics["inverse_temperature"] = float(model.net.log_inverse_temperature.exp())
    return OracleResult(
        name=f"encoder_oracle_{kind}",
        hypotheses=["A-ENC-1", "A-FE-1"],
        question=f"Can the {kind} encoder regress the true pose (x, cos, sin) from one frame?",
        config={
            "kind": kind,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "train_frames": int(len(train_images)),
            "val_frames": int(len(val_images)),
            "device": device,
            "seed": seed,
        },
        metrics=metrics,
        criteria=[Criterion("val_r2_overall", ">=", 0.95, "held-out pose R² (episode split)")],
    )
