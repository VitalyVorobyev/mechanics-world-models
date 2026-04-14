"""Factored encoder producing ``(q, qdot, z_nuisance)`` posteriors.

Positions ``q`` are *observed* (emitted by the encoder from the image);
velocities ``q̇`` are *derived* in the forward pass as the finite-difference
time-derivative of ``q``. This hard-codes the kinematic identity
``q̇ = d/dt(q)`` into the model by construction, so the mechanics branch
cannot degenerate into two decoupled per-frame latents the way two
independent encoder heads do. The Lagrangian integrator then operates on
``(q, q̇)`` pairs that are, by design, in a derivative relationship.

Same pattern as LNN/HNN-from-pixels (Cranmer et al. 2020, Greydanus et al.
2019, Toth et al. 2020): encoder extracts the generalized coordinate;
velocity follows by numerical differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.encoder import ConvEncoder


@dataclass(frozen=True)
class FactoredPosterior:
    """Per-timestep posterior parameters for the mechanics+nuisance split.

    Note: ``qdot_mean`` / ``qdot_std`` are *derived* from the finite-difference
    of ``q_mean`` / ``q_std``, not emitted by the encoder. They live on the
    dataclass so downstream code (decoder, losses, integrator) keeps a single
    consumer-facing type.
    """

    q_mean: torch.Tensor  # [B, T, d]
    q_std: torch.Tensor  # [B, T, d]
    qdot_mean: torch.Tensor  # [B, T, d] — derived from q via finite diff
    qdot_std: torch.Tensor  # [B, T, d] — pushforward of q_std through diff
    z_mean: torch.Tensor  # [B, T, k]
    z_std: torch.Tensor  # [B, T, k]


def derive_qdot_from_q(
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(qdot_mean, qdot_std)`` from the kinematic constraint ``q̇ = d/dt(q)``.

    Central difference in the interior, forward diff at ``t=0``, backward diff
    at ``t=T-1``. All three formulas are linear in ``q``, so the pushforward of
    a diagonal Gaussian posterior is again a diagonal Gaussian whose variance
    is the sum of the contributing input variances scaled by the diff stencil
    squared.

    Args:
        q_mean: ``[..., T, d]`` posterior mean of ``q`` along time axis at -2.
        q_std: same shape; posterior std.
        dt: integration step matching the integrator's ``dt``.

    Returns:
        ``(qdot_mean, qdot_std)`` each ``[..., T, d]``.
    """

    if q_mean.shape != q_std.shape:
        raise ValueError(
            f"q_mean and q_std shapes must match; got {q_mean.shape} vs {q_std.shape}",
        )
    if q_mean.shape[-2] < 2:
        raise ValueError(
            f"derive_qdot_from_q requires T >= 2 along axis -2; got T={q_mean.shape[-2]}",
        )
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt}")

    # Central difference for the interior: qdot[t] = (q[t+1] - q[t-1]) / (2dt).
    # Forward / backward at the boundaries: qdot[0] = (q[1] - q[0])/dt, etc.
    qdot_interior = (q_mean[..., 2:, :] - q_mean[..., :-2, :]) / (2.0 * dt)
    qdot_first = (q_mean[..., 1:2, :] - q_mean[..., 0:1, :]) / dt
    qdot_last = (q_mean[..., -1:, :] - q_mean[..., -2:-1, :]) / dt
    qdot_mean = torch.cat([qdot_first, qdot_interior, qdot_last], dim=-2)

    # Pushforward variance: diff is a linear combination of independent
    # Gaussians, so variance is the sum of contributing variances times the
    # coefficient squared. For central diff the coefficients are +/- 1/(2dt).
    q_var = q_std.pow(2)
    qdot_var_interior = (q_var[..., 2:, :] + q_var[..., :-2, :]) / (4.0 * dt * dt)
    qdot_var_first = (q_var[..., 1:2, :] + q_var[..., 0:1, :]) / (dt * dt)
    qdot_var_last = (q_var[..., -1:, :] + q_var[..., -2:-1, :]) / (dt * dt)
    qdot_var = torch.cat([qdot_var_first, qdot_var_interior, qdot_var_last], dim=-2)
    qdot_std = qdot_var.sqrt()
    return qdot_mean, qdot_std


class FactoredEncoder(nn.Module):
    """Shared ConvEncoder trunk + mechanics head (q only) + nuisance head (z).

    The mechanics head emits only ``q`` (mean, raw std); ``q̇`` is computed
    from ``q`` via ``derive_qdot_from_q`` inside ``forward``. The nuisance head
    emits ``(z_mean, z_raw_std)`` exactly as before. Standard deviations are
    positivized via ``softplus + min_std`` to match the RSSM sampling contract.
    """

    def __init__(
        self,
        q_dim: int,
        nuisance_dim: int,
        dt: float,
        embedding_size: int = 256,
        head_hidden_size: int = 128,
        mech_min_std: float = 0.05,
        nuisance_min_std: float = 0.1,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if q_dim < 1:
            raise ValueError("q_dim must be >= 1")
        if nuisance_dim < 1:
            raise ValueError("nuisance_dim must be >= 1")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.q_dim = q_dim
        self.nuisance_dim = nuisance_dim
        self.dt = dt
        self.embedding_size = embedding_size
        self.mech_min_std = mech_min_std
        self.nuisance_min_std = nuisance_min_std

        self.trunk = ConvEncoder(
            embedding_size=embedding_size,
            input_channels=input_channels,
        )
        # Mechanics head now emits only (q_mean, q_raw_std) — two chunks, not
        # four. q̇ is derived from q by finite differencing inside forward().
        self.mech_head = nn.Sequential(
            nn.Linear(embedding_size, head_hidden_size),
            nn.ELU(),
            nn.Linear(head_hidden_size, 2 * q_dim),
        )
        self.nuisance_head = nn.Sequential(
            nn.Linear(embedding_size, head_hidden_size),
            nn.ELU(),
            nn.Linear(head_hidden_size, 2 * nuisance_dim),
        )

    def forward(self, observations: torch.Tensor) -> FactoredPosterior:
        """Encode observations to posterior parameters.

        Requires ``T >= 2`` along the time axis so ``q̇`` can be derived by
        finite differencing. At inference time, callers that operate per-frame
        (e.g. MPC) must buffer at least two consecutive observations.
        """

        if observations.ndim != 5:
            raise ValueError("observations must have shape [B, T, C, H, W]")
        if observations.shape[1] < 2:
            raise ValueError(
                "FactoredEncoder requires T >= 2 so qdot can be derived from "
                f"a finite difference of q; got T={observations.shape[1]}",
            )
        embeddings = self.trunk(observations)  # [B, T, E]
        mech_raw = self.mech_head(embeddings)  # [B, T, 2d]
        nuis_raw = self.nuisance_head(embeddings)  # [B, T, 2k]
        q_mean, q_raw_std = torch.chunk(mech_raw, chunks=2, dim=-1)
        z_mean, z_raw_std = torch.chunk(nuis_raw, chunks=2, dim=-1)
        q_std = F.softplus(q_raw_std) + self.mech_min_std
        z_std = F.softplus(z_raw_std) + self.nuisance_min_std
        qdot_mean, qdot_std = derive_qdot_from_q(q_mean, q_std, dt=self.dt)
        return FactoredPosterior(
            q_mean=q_mean,
            q_std=q_std,
            qdot_mean=qdot_mean,
            qdot_std=qdot_std,
            z_mean=z_mean,
            z_std=z_std,
        )
