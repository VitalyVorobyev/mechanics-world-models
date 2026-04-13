"""Factored transition: Lagrangian step on ``(q, qdot)`` + AR(1) on ``z``.

The mechanics branch is deterministic-mean Gaussian: the integrator produces
``(q_mean, qdot_mean)`` from the previous posterior and the current action,
and a small learnable per-coordinate std parameterizes the prior. The
nuisance branch evolves as ``z_{t+1} = alpha * z_t`` (prior mean) with a
learnable ``alpha`` initialized near ``0.95``, plus a learnable prior std.

No GRU is needed — the mechanics branch carries the full dynamical memory
through ``(q, qdot)`` and the Euler-Lagrange step, while the nuisance branch
is Markov by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.mechanics.integrator import semi_implicit_euler_step
from models.mechanics.lagrangian import LagrangianDynamics


@dataclass(frozen=True)
class MechanicsPrior:
    """Prior distribution parameters for one transition step."""

    q_mean: torch.Tensor
    q_std: torch.Tensor
    qdot_mean: torch.Tensor
    qdot_std: torch.Tensor
    z_mean: torch.Tensor
    z_std: torch.Tensor


class MechanicsTransition(nn.Module):
    """Compose ``LagrangianDynamics`` + integrator + AR(1) nuisance channel.

    The module holds the learnable prior-std parameters and the nuisance
    AR(1) coefficient; everything else is delegated to the injected
    ``LagrangianDynamics`` bundle so tests can swap primitives easily.
    """

    def __init__(
        self,
        dynamics: LagrangianDynamics,
        nuisance_dim: int,
        dt: float,
        n_substeps: int = 1,
        prior_min_std: float = 0.05,
        nuisance_prior_min_std: float = 0.1,
        nuisance_alpha_init: float = 0.95,
    ) -> None:
        super().__init__()
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")
        self.dynamics = dynamics
        self.q_dim = dynamics.mass_matrix.q_dim
        self.action_dim = dynamics.actuation.action_dim
        self.nuisance_dim = nuisance_dim
        self.dt = dt
        self.n_substeps = n_substeps
        self.prior_min_std = prior_min_std
        self.nuisance_prior_min_std = nuisance_prior_min_std
        # Per-coordinate learnable log-stds (held in raw_std via softplus) so
        # the KL terms each branch contributes stay coordinate-scaled.
        self.q_raw_std = nn.Parameter(torch.zeros(self.q_dim))
        self.qdot_raw_std = nn.Parameter(torch.zeros(self.q_dim))
        self.z_raw_std = nn.Parameter(torch.zeros(self.nuisance_dim))
        # alpha is parameterized via a logit so it stays in (0, 1) after sigmoid.
        logit_init = torch.logit(torch.tensor(float(nuisance_alpha_init)))
        self.z_alpha_logit = nn.Parameter(logit_init.clone())

    def prior_stds(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q_std, qdot_std, z_std)`` as ``[d]`` / ``[k]`` tensors."""

        q_std = F.softplus(self.q_raw_std) + self.prior_min_std
        qdot_std = F.softplus(self.qdot_raw_std) + self.prior_min_std
        z_std = F.softplus(self.z_raw_std) + self.nuisance_prior_min_std
        return q_std, qdot_std, z_std

    @property
    def nuisance_alpha(self) -> torch.Tensor:
        """AR(1) coefficient ``alpha`` bounded in ``(0, 1)``."""

        return torch.sigmoid(self.z_alpha_logit)

    def step(
        self,
        q_prev: torch.Tensor,
        qdot_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action: torch.Tensor,
    ) -> MechanicsPrior:
        """Advance one prior step from posterior-sampled ``(q, qdot, z)``.

        Args:
            q_prev: ``[B, d]`` previous posterior q sample.
            qdot_prev: ``[B, d]`` previous posterior qdot sample.
            z_prev: ``[B, k]`` previous posterior z sample.
            action: ``[B, a]`` action applied at this step.

        Returns:
            Prior means/stds shaped ``[B, d]``/``[B, k]`` for the next step.
        """

        if q_prev.shape != qdot_prev.shape:
            raise ValueError("q_prev and qdot_prev shapes must match")
        q_next_mean, qdot_next_mean = semi_implicit_euler_step(
            dynamics=self.dynamics,
            q=q_prev,
            qdot=qdot_prev,
            u=action,
            dt=self.dt,
            n_substeps=self.n_substeps,
        )
        z_next_mean = self.nuisance_alpha * z_prev
        q_std, qdot_std, z_std = self.prior_stds()
        batch_size = q_prev.shape[0]
        q_std_b = q_std.expand(batch_size, -1)
        qdot_std_b = qdot_std.expand(batch_size, -1)
        z_std_b = z_std.expand(batch_size, -1)
        return MechanicsPrior(
            q_mean=q_next_mean,
            q_std=q_std_b,
            qdot_mean=qdot_next_mean,
            qdot_std=qdot_std_b,
            z_mean=z_next_mean,
            z_std=z_std_b,
        )

    def initial_prior(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MechanicsPrior:
        """Unit Gaussian prior at ``t = 0`` (no history to condition on).

        Broadcasts the learnable prior stds across the batch so the KL at
        ``t = 0`` is ``KL(q || N(0, sigma_prior))`` with a non-trivial prior
        std. Mean is zero.
        """

        q_std, qdot_std, z_std = self.prior_stds()
        q_mean = torch.zeros(batch_size, self.q_dim, device=device, dtype=dtype)
        qdot_mean = torch.zeros(batch_size, self.q_dim, device=device, dtype=dtype)
        z_mean = torch.zeros(batch_size, self.nuisance_dim, device=device, dtype=dtype)
        return MechanicsPrior(
            q_mean=q_mean,
            q_std=q_std.expand(batch_size, -1).clone(),
            qdot_mean=qdot_mean,
            qdot_std=qdot_std.expand(batch_size, -1).clone(),
            z_mean=z_mean,
            z_std=z_std.expand(batch_size, -1).clone(),
        )
