"""Diagnostic linear probes for the mechanics world model.

These metrics are *not* training signals — they exist so we can verify that
the unsupervised factorization actually captures the system's generalized
coordinates, which is the assumption underlying H1/H3. The two probes are:

* **Linear probe**: ordinary least squares from ``encoded_q`` (and
  ``encoded_qdot``) onto the simulator's ``qpos`` (and ``qvel``). We report
  R² so a value near 1 means the encoder's mechanics head is, up to a linear
  reparameterization, the true coordinate.
* **Energy tracking**: compare ``E_learned(t) = T(q_t, qdot_t) + V(q_t)``
  computed by the model's ``MassMatrix`` and ``Potential`` against
  dm_control's ``physics.kinetic_energy() + physics.potential_energy()`` over
  open-loop rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from models.mechanics import MechanicsWorldModel, total_energy


@dataclass(frozen=True)
class LinearProbeResult:
    """OLS fit summary for one probe (q or qdot)."""

    r2_per_dim: NDArray[np.float64]  # [target_dim]
    r2_overall: float
    coef: NDArray[np.float64]  # [target_dim, encoded_dim + 1] (last col = intercept)
    rmse: NDArray[np.float64]  # [target_dim]


def fit_linear_probe(
    encoded: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> LinearProbeResult:
    """Fit a held-out OLS probe ``target ~= W @ encoded + b`` and return R² stats.

    ``encoded`` and ``target`` are flattened across (B, T) so each row is a
    single sample. We split deterministically by row index after a seeded
    shuffle so the train/test split is reproducible across calls.
    """

    if encoded.ndim != 2 or target.ndim != 2:
        raise ValueError("encoded and target must be 2-D [N, D]")
    if encoded.shape[0] != target.shape[0]:
        raise ValueError("encoded and target must have the same N")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must satisfy 0 < f < 1")

    rng = np.random.default_rng(seed)
    n_samples = encoded.shape[0]
    permutation = rng.permutation(n_samples)
    split = int(round(n_samples * train_fraction))
    train_idx = permutation[:split]
    test_idx = permutation[split:]
    if len(train_idx) < encoded.shape[1] + 1 or len(test_idx) < 1:
        raise ValueError(
            "not enough samples for the requested split: "
            f"N={n_samples}, train_fraction={train_fraction}",
        )

    x_train = np.concatenate(
        [encoded[train_idx], np.ones((len(train_idx), 1))], axis=1,
    )  # [N_train, D_enc + 1]
    y_train = target[train_idx]  # [N_train, D_tgt]
    coef, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    coef_t = coef.T  # [D_tgt, D_enc + 1]

    x_test = np.concatenate(
        [encoded[test_idx], np.ones((len(test_idx), 1))], axis=1,
    )
    y_test = target[test_idx]
    predictions = x_test @ coef_t.T  # [N_test, D_tgt]
    residuals = y_test - predictions
    ss_res = (residuals**2).sum(axis=0)
    ss_tot = ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_dim = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)
    r2_overall = float(
        1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12),
    )
    rmse = np.sqrt((residuals**2).mean(axis=0))
    return LinearProbeResult(
        r2_per_dim=r2_per_dim,
        r2_overall=r2_overall,
        coef=coef_t,
        rmse=rmse,
    )


@torch.no_grad()
def encode_dataset(
    model: MechanicsWorldModel,
    observations: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the encoder on a [B, T, 3, H, W] tensor and return ``(q, qdot, z)``.

    All three returned arrays are flattened to ``[B * T, D]`` ``float64``
    NumPy arrays so downstream OLS works without device shuffling.
    """

    if observations.ndim != 5:
        raise ValueError("observations must have shape [B, T, C, H, W]")
    model.eval()
    posterior = model.encoder(observations.to(device))
    q = posterior.q_mean.detach().cpu().double().numpy()
    qdot = posterior.qdot_mean.detach().cpu().double().numpy()
    z = posterior.z_mean.detach().cpu().double().numpy()
    return _flatten(q), _flatten(qdot), _flatten(z)


def _flatten(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr.reshape(-1, arr.shape[-1])


def flatten_target(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten a ``[B, T, D]`` numpy array of physics targets to ``[B*T, D]``."""

    if arr.ndim != 3:
        raise ValueError("target array must be [B, T, D]")
    return arr.reshape(-1, arr.shape[-1])


@torch.no_grad()
def compute_learned_energy(
    model: MechanicsWorldModel,
    q: torch.Tensor,
    qdot: torch.Tensor,
) -> NDArray[np.float64]:
    """Return ``E_learned(t) = T + V`` from the model's mass matrix + potential.

    ``q`` and ``qdot`` are ``[B, T, d]`` posterior or imagined samples.
    Returned array is ``[B, T]`` ``float64``.
    """

    if q.shape != qdot.shape:
        raise ValueError("q and qdot must share shape")
    energy = total_energy(model.mass_matrix_module, model.potential_module, q, qdot)
    return energy.detach().cpu().double().numpy()


def normalized_energy_drift(
    energy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``(E(t) - E(0)) / max(|E(0)|, eps)`` per sequence — diagnostic only."""

    if energy.ndim != 2:
        raise ValueError("energy must be [B, T]")
    reference = np.maximum(np.abs(energy[:, :1]), 1e-3)
    return (energy - energy[:, :1]) / reference
