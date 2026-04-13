"""Learned Lagrangian primitives: mass matrix, potential, dissipation, actuation.

The mechanics branch of the factored world model applies an Euler-Lagrange
forward dynamics step on ``(q, qdot)`` using the *mechanical form*

    M(q) q_ddot = dL/dq - M_dot(q) qdot - dD/d(qdot) + B(q) u

where the kinetic energy is restricted to ``T = 0.5 qdot^T M(q) qdot``. This
avoids the generic double-Jacobian of a free-form ``L(q, qdot)`` while keeping
the expressiveness that matters for H3 (learned dissipation) and K1 (stable
multi-step prediction).

All modules here are small MLPs over the ``d``-dimensional generalized
coordinate; for cartpole we expect ``d = 2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _mlp(
    input_size: int,
    output_size: int,
    hidden_size: int,
    hidden_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_size
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.SiLU())
        in_features = hidden_size
    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)


class MassMatrix(nn.Module):
    """Positive-definite mass matrix ``M(q)`` via Cholesky parameterization.

    Outputs ``M(q) = L(q) L(q)^T + eps * I`` where ``L`` is lower-triangular
    with ``softplus``-activated diagonal; this guarantees symmetric PD
    regardless of weight values so downstream ``linalg.solve`` never blows up.
    """

    def __init__(
        self,
        q_dim: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.eps = eps
        n_tri = q_dim * (q_dim + 1) // 2
        self._tri_indices = torch.tril_indices(q_dim, q_dim)
        self._diag_mask = (self._tri_indices[0] == self._tri_indices[1])
        self.net = _mlp(
            input_size=q_dim,
            output_size=n_tri,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Return ``M(q)`` shaped ``[..., d, d]``."""

        if q.shape[-1] != self.q_dim:
            raise ValueError(f"expected last dim={self.q_dim}, got {q.shape[-1]}")
        batch_shape = q.shape[:-1]
        flat = q.reshape(-1, self.q_dim)
        raw = self.net(flat)
        diag = self._diag_mask.to(raw.device)
        tri_indices = self._tri_indices.to(raw.device)
        activated = torch.where(diag, F.softplus(raw) + self.eps, raw)
        lower = q.new_zeros(flat.shape[0], self.q_dim, self.q_dim)
        lower[:, tri_indices[0], tri_indices[1]] = activated
        mass = lower @ lower.transpose(-1, -2)
        eye = torch.eye(self.q_dim, device=q.device, dtype=q.dtype)
        mass = mass + self.eps * eye
        return mass.reshape(*batch_shape, self.q_dim, self.q_dim)


class Potential(nn.Module):
    """Learned scalar potential energy ``V(q)``."""

    def __init__(
        self,
        q_dim: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.net = _mlp(
            input_size=q_dim,
            output_size=1,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Return ``V(q)`` shaped ``[...]`` (scalar per batch element)."""

        if q.shape[-1] != self.q_dim:
            raise ValueError(f"expected last dim={self.q_dim}, got {q.shape[-1]}")
        batch_shape = q.shape[:-1]
        flat = q.reshape(-1, self.q_dim)
        return self.net(flat).reshape(*batch_shape)


class Dissipation(nn.Module):
    """Rayleigh dissipation ``D(qdot) = 0.5 * qdot^T C(qdot) qdot`` with PSD ``C``.

    ``C`` is Cholesky-parameterized so ``dD/d(qdot)`` is monotone in damping.
    With ``C`` constant in qdot this reduces to linear damping; letting ``C``
    depend on ``qdot`` keeps the door open to learning non-linear friction
    curves without sacrificing positivity.
    """

    def __init__(
        self,
        q_dim: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.eps = eps
        n_tri = q_dim * (q_dim + 1) // 2
        self._tri_indices = torch.tril_indices(q_dim, q_dim)
        self._diag_mask = (self._tri_indices[0] == self._tri_indices[1])
        self.net = _mlp(
            input_size=q_dim,
            output_size=n_tri,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
        )

    def damping_matrix(self, qdot: torch.Tensor) -> torch.Tensor:
        """Return the PSD damping matrix ``C(qdot)`` shaped ``[..., d, d]``."""

        if qdot.shape[-1] != self.q_dim:
            raise ValueError(f"expected last dim={self.q_dim}, got {qdot.shape[-1]}")
        batch_shape = qdot.shape[:-1]
        flat = qdot.reshape(-1, self.q_dim)
        raw = self.net(flat)
        diag = self._diag_mask.to(raw.device)
        tri_indices = self._tri_indices.to(raw.device)
        activated = torch.where(diag, F.softplus(raw) + self.eps, raw)
        lower = qdot.new_zeros(flat.shape[0], self.q_dim, self.q_dim)
        lower[:, tri_indices[0], tri_indices[1]] = activated
        c = lower @ lower.transpose(-1, -2)
        return c.reshape(*batch_shape, self.q_dim, self.q_dim)

    def forward(self, qdot: torch.Tensor) -> torch.Tensor:
        """Return scalar dissipation ``D(qdot) = 0.5 qdot^T C(qdot) qdot``."""

        c = self.damping_matrix(qdot)
        return 0.5 * torch.einsum("...i,...ij,...j->...", qdot, c, qdot)


class Actuation(nn.Module):
    """Learned input map ``B(q)`` of shape ``[d, a]``.

    Generalized force injected into the Euler-Lagrange equation is ``B(q) u``.
    """

    def __init__(
        self,
        q_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.action_dim = action_dim
        self.net = _mlp(
            input_size=q_dim,
            output_size=q_dim * action_dim,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Return ``B(q)`` shaped ``[..., d, a]``."""

        if q.shape[-1] != self.q_dim:
            raise ValueError(f"expected last dim={self.q_dim}, got {q.shape[-1]}")
        batch_shape = q.shape[:-1]
        flat = q.reshape(-1, self.q_dim)
        raw = self.net(flat)
        return raw.reshape(*batch_shape, self.q_dim, self.action_dim)


@dataclass(frozen=True)
class LagrangianDynamics:
    """Bundle of the four learned primitives used by the EL step."""

    mass_matrix: MassMatrix
    potential: Potential
    dissipation: Dissipation | None
    actuation: Actuation


def forward_acceleration(
    dynamics: LagrangianDynamics,
    q: torch.Tensor,
    qdot: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute ``q_ddot`` from ``(q, qdot, u)`` via the mechanical EL form.

    The closed-form identity used here is

        d/dt(dL/d(qdot)) - dL/dq = F_ext
          <=>  M(q) q_ddot = dL/dq - M_dot(q) qdot + F_ext

    with ``L = 0.5 qdot^T M(q) qdot - V(q)`` (so qdot is treated as fixed when
    taking ``dL/dq``) and ``F_ext = -dD/d(qdot) + B(q) u``. ``M_dot(q) qdot``
    is computed with a forward JVP on ``p(q) = M(q) qdot`` along the direction
    ``qdot`` — equivalent to ``dM/dt * qdot`` because ``qdot`` is fixed.

    Gradients flow back to all module parameters via ``create_graph=True`` on
    the inner autograd calls; outer-graph gradient wrt ``q`` and ``qdot``
    propagates through the enable-grad branches as well, so the full
    imagination rollout stays end-to-end differentiable.
    """

    if q.shape != qdot.shape:
        raise ValueError(f"q and qdot shapes must match; got {q.shape} vs {qdot.shape}")
    if u.shape[:-1] != q.shape[:-1] or u.shape[-1] != dynamics.actuation.action_dim:
        raise ValueError(
            f"u must broadcast-align with q batch dims and have last dim="
            f"{dynamics.actuation.action_dim}; got u.shape={u.shape}",
        )

    # Autograd path: derivatives of T - V wrt q (with qdot held fixed).
    with torch.enable_grad():
        q_grad = q if q.requires_grad else q.detach().requires_grad_(True)
        qdot_ref = qdot.detach() if not qdot.requires_grad else qdot
        mass = dynamics.mass_matrix(q_grad)
        potential = dynamics.potential(q_grad)
        kinetic = 0.5 * torch.einsum("...i,...ij,...j->...", qdot_ref, mass, qdot_ref)
        lagrangian_sum = (kinetic - potential).sum()
        (dL_dq,) = torch.autograd.grad(
            lagrangian_sum,
            q_grad,
            create_graph=torch.is_grad_enabled(),
        )

    # dM/dt * qdot via a forward JVP: direction qdot on p(q) = M(q) qdot.
    def momentum_of_q(q_in: torch.Tensor) -> torch.Tensor:
        m = dynamics.mass_matrix(q_in)
        return torch.einsum("...ij,...j->...i", m, qdot_ref)

    _, m_dot_qdot = torch.autograd.functional.jvp(
        momentum_of_q,
        q if q.requires_grad else q.detach(),
        qdot_ref,
        create_graph=torch.is_grad_enabled(),
    )

    # Dissipation: dD/d(qdot).
    if dynamics.dissipation is not None:
        with torch.enable_grad():
            qdot_grad = qdot if qdot.requires_grad else qdot.detach().requires_grad_(True)
            dissipation_value = dynamics.dissipation(qdot_grad).sum()
            (dD_dqdot,) = torch.autograd.grad(
                dissipation_value,
                qdot_grad,
                create_graph=torch.is_grad_enabled(),
            )
    else:
        dD_dqdot = torch.zeros_like(qdot)

    b_matrix = dynamics.actuation(q)
    tau = torch.einsum("...ij,...j->...i", b_matrix, u)

    rhs = dL_dq - m_dot_qdot - dD_dqdot + tau
    # Recompute mass matrix in the outer graph so linalg.solve differentiates
    # correctly wrt its parameters.
    mass_outer = dynamics.mass_matrix(q)
    q_ddot = torch.linalg.solve(mass_outer, rhs.unsqueeze(-1)).squeeze(-1)
    return q_ddot


def kinetic_energy(
    mass_matrix: MassMatrix,
    q: torch.Tensor,
    qdot: torch.Tensor,
) -> torch.Tensor:
    """Convenience: ``T = 0.5 qdot^T M(q) qdot`` as ``[...]`` scalar."""

    mass = mass_matrix(q)
    return 0.5 * torch.einsum("...i,...ij,...j->...", qdot, mass, qdot)


def total_energy(
    mass_matrix: MassMatrix,
    potential: Potential,
    q: torch.Tensor,
    qdot: torch.Tensor,
) -> torch.Tensor:
    """Return ``T + V`` for diagnostic energy tracking (non-training)."""

    return kinetic_energy(mass_matrix, q, qdot) + potential(q)
