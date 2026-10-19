"""Rung 3 diagnostic: is a learned mass matrix the true one up to scale (A-ACT-1)?

Compares ``M(q)`` of a pretrained Lagrangian (dynamics-oracle cache) with
MuJoCo's ``mj_fullM`` over a grid of poses, fitting one global scale
``alpha``. A relative residual near 0 means the learned terms are physical
(a cart-mass change is then an offset of ``M[0, 0]``); near 1 means the
model found another factorization of ``M^{-1} B``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
from oracles.common import OracleResult
from oracles.dynamics import DynamicsWrapper


def true_mass_matrices(q: np.ndarray, scales: dict[str, float] | None = None) -> np.ndarray:
    """MuJoCo mass matrix ``[N, 2, 2]`` at poses ``q [N, 2]``."""

    import mujoco

    env = DmControlPixelEnv(PixelEnvConfig(physics_param_scales=scales))
    try:
        model, data = env._env.physics.model.ptr, env._env.physics.data.ptr
        out = np.zeros((q.shape[0], 2, 2))
        for i, pose in enumerate(q):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            mujoco.mj_fullM(model, out[i], data.qM)
        return out
    finally:
        env.close()


def mass_gauge(model: DynamicsWrapper, xs: Sequence[float] = (-1.0, 0.0, 1.0), num_angles: int = 73) -> dict:
    thetas = np.linspace(-np.pi, np.pi, num_angles)
    q = np.stack(np.meshgrid(np.asarray(xs), thetas, indexing="ij"), axis=-1).reshape(-1, 2)
    true = true_mass_matrices(q)
    with torch.no_grad():
        learned = model.mass(torch.as_tensor(q, dtype=torch.float32)).double().numpy()
    alpha = float((learned * true).sum() / (true * true).sum())
    metrics = {
        "alpha": alpha,
        "relative_residual": float(np.linalg.norm(learned - alpha * true) / np.linalg.norm(learned)),
    }
    for i, j in ((0, 0), (0, 1), (1, 1)):
        # Least-squares slope of learned vs alpha * true entry (1 = physical).
        t, l = alpha * true[:, i, j], learned[:, i, j]
        metrics[f"slope_M{i}{j}"] = float((l * t).sum() / (t * t).sum())
    metrics["M01_shape_correlation"] = float(np.corrcoef(learned[:, 0, 1], true[:, 0, 1])[0, 1])
    return metrics


def run_mass_gauge(checkpoints: dict[str, tuple[Path, Sequence[int] | None]]) -> OracleResult:
    """``checkpoints[name] = (pretrained_lagrangian.pt, actuated_dims)``."""

    metrics: dict[str, float] = {}
    for name, (path, actuated_dims) in checkpoints.items():
        model = DynamicsWrapper("lagrangian", actuated_dims=actuated_dims)
        model.load_state_dict(torch.load(path, weights_only=True))
        for key, value in mass_gauge(model).items():
            metrics[f"{name}.{key}"] = value
    return OracleResult(
        name="mass_gauge",
        hypotheses=["A-ACT-1", "H1b"],
        question="Is the learned mass matrix the true one up to a global scale?",
        config={name: {"path": str(p), "actuated_dims": a} for name, (p, a) in checkpoints.items()},
        metrics=metrics,
        notes=["relative_residual = |M_learned - alpha M_true| / |M_learned| over x in {-1, 0, 1}, 73 angles."],
    )
