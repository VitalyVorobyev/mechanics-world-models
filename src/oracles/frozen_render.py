"""Rung 2b: can an encoder learn the pose *without labels* through a fixed, correct renderer?

End-to-end keypoint training failed (A-FE-1). Rung 1 showed the geometric
renderer draws frames well from the true pose, and rung 2 showed encoders
can regress the pose with labels. This rung isolates the remaining piece,
self-supervised encoder learning:

1. fit the ``render`` decoder from the true pose (as rung 1);
2. freeze it, and train an encoder on reconstruction loss only (no pose
   labels): ``image -> encoder -> (x, theta) -> frozen renderer -> image``;
3. score the encoder's poses against the truth in degrees and pixels.

Because the frozen renderer is metric (its affine map from metres was
fitted on true poses), a working encoder must produce metric poses; there
is no gauge to fit afterwards. If this fails, joint encoder-renderer
learning cannot succeed either, and the failure is in the reconstruction
gradient (for example, no gradient when the rendered pole misses the real
one), not in the renderer.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from models.losses import foreground_reconstruction_mask
from oracles.common import Criterion, OracleResult
from oracles.decoder import _PoseDecoder, _to_images
from oracles.encoder import _PoseEncoder, pose_errors


def _reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, foreground_weight: float, pool_sizes: tuple[int, ...] = (),
) -> torch.Tensor:
    """Pixel + foreground MSE, plus MSE between average-pooled copies (coarse-to-fine).

    Pooling spreads the thin pole over many coarse pixels, so a rendered pole
    that misses the real one still gets a gradient towards it.
    """

    mask = foreground_reconstruction_mask(target[:, None])[:, 0].detach()
    se = (pred - target).square()
    loss = se.mean() + foreground_weight * (se * mask).mean()
    for size in pool_sizes:
        pooled = lambda img: F.avg_pool2d(img, size, stride=max(size // 2, 1), padding=size // 2)  # noqa: E731
        loss = loss + (pooled(pred) - pooled(target)).square().mean() * size
    return loss


def run_frozen_renderer_oracle(
    train_images: np.ndarray,
    train_qpos: np.ndarray,
    val_images: np.ndarray,
    val_qpos: np.ndarray,
    *,
    kind: str = "keypoint",
    decoder_steps: int = 3000,
    encoder_steps: int = 6000,
    batch_size: int = 64,
    lr: float = 1e-3,
    foreground_weight: float = 1.0,
    pool_sizes: tuple[int, ...] = (),
    device: str = "cpu",
    seed: int = 0,
    log=print,
) -> OracleResult:
    """Criterion: held-out median angle error <= 3° on frames where the system is in view.

    ``pool_sizes`` adds coarse-to-fine terms to the encoder's loss only.
    """

    torch.manual_seed(seed)
    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    decoder = _PoseDecoder("render").to(dev)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    start = time.time()
    for step in range(1, decoder_steps + 1):
        idx = rng.integers(0, len(train_images), size=batch_size)
        target = _to_images(train_images[idx], dev)
        qpos = torch.as_tensor(train_qpos[idx], dtype=torch.float32, device=dev)
        loss = _reconstruction_loss(decoder(qpos), target, foreground_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(decoder_steps // 5, 1) == 0:
            log(f"[decoder] step {step} loss {loss.item():.5f} ({time.time() - start:.0f}s)")
    decoder_loss = loss.item()
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    encoder = _PoseEncoder(kind).to(dev)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, encoder_steps)
    start = time.time()

    def pose(images: torch.Tensor) -> torch.Tensor:
        out = encoder(images)  # (x, cos, sin)
        return torch.stack([out[:, 0], torch.atan2(out[:, 2], out[:, 1])], dim=-1)

    for step in range(1, encoder_steps + 1):
        idx = rng.integers(0, len(train_images), size=batch_size)
        target = _to_images(train_images[idx], dev)
        loss = _reconstruction_loss(decoder(pose(target)), target, foreground_weight, pool_sizes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % max(encoder_steps // 10, 1) == 0:
            log(f"[{kind} via frozen renderer] step {step} loss {loss.item():.5f} ({time.time() - start:.0f}s)")

    encoder.eval()
    preds, recon = [], []
    with torch.no_grad():
        for begin in range(0, len(val_images), 256):
            target = _to_images(val_images[begin : begin + 256], dev)
            q = pose(target)
            preds.append(torch.stack([q[:, 0], torch.cos(q[:, 1]), torch.sin(q[:, 1])], dim=-1).cpu().numpy())
            recon.append(_reconstruction_loss(decoder(q), target, foreground_weight).item())
    pred = np.concatenate(preds)
    errors = pose_errors(pred, val_qpos)
    theta = np.arctan2(pred[:, 2], pred[:, 1])
    flipped = np.abs(np.angle(np.exp(1j * (theta - val_qpos[:, 1])))) > np.pi / 2
    metrics = {
        **errors,
        "theta_flipped_fraction": float(flipped.mean()),
        "x_corr": float(np.corrcoef(pred[:, 0], val_qpos[:, 0])[0, 1]),
        "decoder_final_loss": decoder_loss,
        "val_reconstruction_loss": float(np.mean(recon)),
    }
    return OracleResult(
        name=f"frozen_renderer_oracle_{kind}",
        hypotheses=["A-FE-1"],
        question=f"Can the {kind} encoder learn the pose from reconstruction alone through a fixed renderer?",
        config={
            "kind": kind,
            "decoder_steps": decoder_steps,
            "encoder_steps": encoder_steps,
            "batch_size": batch_size,
            "lr": lr,
            "foreground_weight": foreground_weight,
            "pool_sizes": list(pool_sizes),
            "train_frames": int(len(train_images)),
            "val_frames": int(len(val_images)),
            "device": device,
            "seed": seed,
        },
        metrics=metrics,
        criteria=[
            Criterion(
                "visible.theta_err_deg_median", "<=", 3.0,
                "median angle error on in-view frames, learned without pose labels",
            ),
        ],
    )
