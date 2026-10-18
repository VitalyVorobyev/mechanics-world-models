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
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.encoder import ConvEncoder
from models.mechanics.coordinates import validate_angle_dims, wrap_angle_dims

QDOT_STENCILS = ("central", "backward")


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
    stencil: str = "central",
    angle_dims: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(qdot_mean, qdot_std)`` from the kinematic constraint ``q̇ = d/dt(q)``.

    ``stencil="central"`` (legacy): central difference in the interior,
    forward diff at ``t=0``, backward diff at ``t=T-1``. Note that the
    interior value at ``t`` reads ``q_{t+1}``, so a transition prior fed this
    ``q̇_{t-1}`` already sees the frame it is asked to predict.

    ``stencil="backward"``: ``q̇_t = (q_t - q_{t-1}) / dt`` for ``t >= 1`` and
    the forward diff at ``t=0``. Causal for every ``t >= 1`` and exactly the
    velocity convention of the semi-implicit Euler integrator
    (``q_{t+1} = q_t + dt * q̇_{t+1}``).

    Differences on ``angle_dims`` are wrapped to ``[-pi, pi)`` so a crossing of
    the branch cut is not read as a huge velocity. All formulas are linear in
    ``q`` (up to the wrap), so the pushforward of a diagonal Gaussian
    posterior is again diagonal Gaussian.

    Args:
        q_mean: ``[..., T, d]`` posterior mean of ``q`` along time axis at -2.
        q_std: same shape; posterior std.
        dt: integration step matching the integrator's ``dt``.
        stencil: ``"central"`` or ``"backward"``.
        angle_dims: coordinate indices whose differences wrap.

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
    if stencil not in QDOT_STENCILS:
        raise ValueError(f"stencil must be one of {QDOT_STENCILS}; got {stencil!r}")

    q_var = q_std.pow(2)
    if stencil == "backward":
        step_diff = wrap_angle_dims(q_mean[..., 1:, :] - q_mean[..., :-1, :], angle_dims)
        qdot_mean = torch.cat([step_diff[..., :1, :], step_diff], dim=-2) / dt
        step_var = q_var[..., 1:, :] + q_var[..., :-1, :]
        qdot_var = torch.cat([step_var[..., :1, :], step_var], dim=-2) / (dt * dt)
        return qdot_mean, qdot_var.sqrt()

    # Central difference for the interior: qdot[t] = (q[t+1] - q[t-1]) / (2dt).
    # Forward / backward at the boundaries: qdot[0] = (q[1] - q[0])/dt, etc.
    qdot_interior = wrap_angle_dims(
        q_mean[..., 2:, :] - q_mean[..., :-2, :], angle_dims,
    ) / (2.0 * dt)
    qdot_first = wrap_angle_dims(q_mean[..., 1:2, :] - q_mean[..., 0:1, :], angle_dims) / dt
    qdot_last = wrap_angle_dims(q_mean[..., -1:, :] - q_mean[..., -2:-1, :], angle_dims) / dt
    qdot_mean = torch.cat([qdot_first, qdot_interior, qdot_last], dim=-2)

    # Pushforward variance: diff is a linear combination of independent
    # Gaussians, so variance is the sum of contributing variances times the
    # coefficient squared. For central diff the coefficients are +/- 1/(2dt).
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

    For each dim in ``angle_dims`` the head emits one extra output and the
    angle mean is ``atan2(extra, raw_mean)``: the image-to-angle map is then a
    continuous map onto the circle instead of a forced cut of ``S^1`` into
    ``R``. With no angle dims the head layout is the legacy ``2 * q_dim``.
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
        qdot_stencil: str = "central",
        angle_dims: Sequence[int] = (),
    ) -> None:
        super().__init__()
        if q_dim < 1:
            raise ValueError("q_dim must be >= 1")
        if nuisance_dim < 1:
            raise ValueError("nuisance_dim must be >= 1")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if qdot_stencil not in QDOT_STENCILS:
            raise ValueError(f"qdot_stencil must be one of {QDOT_STENCILS}")
        self.q_dim = q_dim
        self.qdot_stencil = qdot_stencil
        self.angle_dims = validate_angle_dims(angle_dims, q_dim)
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
        # four — plus one extra column per angle dim for the atan2 pair.
        # q̇ is derived from q by finite differencing inside forward().
        self.mech_head = nn.Sequential(
            nn.Linear(embedding_size, head_hidden_size),
            nn.ELU(),
            nn.Linear(head_hidden_size, 2 * q_dim + len(self.angle_dims)),
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
        q_mean = mech_raw[..., : self.q_dim]
        q_raw_std = mech_raw[..., self.q_dim : 2 * self.q_dim]
        if self.angle_dims:
            angle_index = list(self.angle_dims)
            angles = torch.atan2(mech_raw[..., 2 * self.q_dim :], q_mean[..., angle_index])
            q_mean = q_mean.clone()
            q_mean[..., angle_index] = angles
        z_mean, z_raw_std = torch.chunk(nuis_raw, chunks=2, dim=-1)
        q_std = F.softplus(q_raw_std) + self.mech_min_std
        z_std = F.softplus(z_raw_std) + self.nuisance_min_std
        qdot_mean, qdot_std = derive_qdot_from_q(
            q_mean, q_std, dt=self.dt, stencil=self.qdot_stencil, angle_dims=self.angle_dims,
        )
        return FactoredPosterior(
            q_mean=q_mean,
            q_std=q_std,
            qdot_mean=qdot_mean,
            qdot_std=qdot_std,
            z_mean=z_mean,
            z_std=z_std,
        )
