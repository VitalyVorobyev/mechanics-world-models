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


def _bounded_input(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Smoothly bound MLP inputs to ``|x| <= scale`` via ``scale * tanh(x/scale)``.

    At cold start the encoder's ``q`` / ``qdot`` are unbounded linear-head
    outputs; the dynamics MLPs (mass, potential, dissipation, actuation) are
    untrained SiLU stacks. A tail-of-distribution sample then drives an
    unbounded activation pattern, gradients explode, and AdamW's second-moment
    estimates get poisoned — see ``docs/current_status.md`` for the observed
    trace. Squashing each MLP input with ``tanh`` keeps the dynamics-network
    domain compact independent of the encoder's scale, without changing the
    Euler-Lagrange form (the transformation is a smooth, invertible change of
    generalized coordinate on any bounded region of state space — cartpole's
    physical state is bounded, so nothing physical is lost).

    The gradient wrt the raw input is scaled by ``sech^2(x/scale)/scale``, so
    it saturates to zero for ``|x| >> scale``: pathological encoder outputs
    simply stop propagating gradient into the dynamics network.
    """

    return scale * torch.tanh(x / scale)


class MassMatrix(nn.Module):
    """Positive-definite mass matrix ``M(q)`` via Cholesky parameterization.

    Outputs ``M(q) = L(q) L(q)^T + eps * I`` where ``L`` is lower-triangular
    with ``softplus``-activated diagonal; this guarantees symmetric PD
    regardless of weight values so downstream ``linalg.solve`` never blows up.
    The default ridge ``eps=1e-2`` caps ``||M^{-1}||`` at roughly 100, which
    is enough headroom for expressiveness on a rigid-body cartpole while
    preventing the cold-start blow-up observed when the smaller ``eps=1e-3``
    ridge allowed ``||M^{-1}||`` to reach ~1000.
    """

    def __init__(
        self,
        q_dim: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        eps: float = 1e-2,
        input_scale: float = 5.0,
    ) -> None:
        super().__init__()
        if input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")
        self.q_dim = q_dim
        self.eps = eps
        self.input_scale = input_scale
        n_tri = q_dim * (q_dim + 1) // 2
        # ``_tri_indices`` and ``_diag_mask`` are deterministic constants of
        # ``q_dim`` but we register them as buffers so ``model.to(device)``
        # moves them along with the rest of the module — the previous plain-
        # attribute storage forced a ``.to(raw.device)`` host->device transfer
        # on every forward. They carry no gradient and no learnable state.
        tri_indices = torch.tril_indices(q_dim, q_dim)
        self.register_buffer("_tri_indices", tri_indices, persistent=False)
        self.register_buffer(
            "_diag_mask", tri_indices[0] == tri_indices[1], persistent=False,
        )
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
        q_bounded = _bounded_input(q, self.input_scale)
        flat = q_bounded.reshape(-1, self.q_dim)
        raw = self.net(flat)
        activated = torch.where(self._diag_mask, F.softplus(raw) + self.eps, raw)
        lower = q.new_zeros(flat.shape[0], self.q_dim, self.q_dim)
        lower[:, self._tri_indices[0], self._tri_indices[1]] = activated
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
        input_scale: float = 5.0,
    ) -> None:
        super().__init__()
        if input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")
        self.q_dim = q_dim
        self.input_scale = input_scale
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
        q_bounded = _bounded_input(q, self.input_scale)
        flat = q_bounded.reshape(-1, self.q_dim)
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
        input_scale: float = 20.0,
    ) -> None:
        super().__init__()
        if input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")
        self.q_dim = q_dim
        self.eps = eps
        self.input_scale = input_scale
        n_tri = q_dim * (q_dim + 1) // 2
        # Same buffer-vs-attribute reasoning as in ``MassMatrix`` — these
        # tensors travel with the module on ``model.to(device)``.
        tri_indices = torch.tril_indices(q_dim, q_dim)
        self.register_buffer("_tri_indices", tri_indices, persistent=False)
        self.register_buffer(
            "_diag_mask", tri_indices[0] == tri_indices[1], persistent=False,
        )
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
        qdot_bounded = _bounded_input(qdot, self.input_scale)
        flat = qdot_bounded.reshape(-1, self.q_dim)
        raw = self.net(flat)
        activated = torch.where(self._diag_mask, F.softplus(raw) + self.eps, raw)
        lower = qdot.new_zeros(flat.shape[0], self.q_dim, self.q_dim)
        lower[:, self._tri_indices[0], self._tri_indices[1]] = activated
        c = lower @ lower.transpose(-1, -2)
        return c.reshape(*batch_shape, self.q_dim, self.q_dim)

    def forward(self, qdot: torch.Tensor) -> torch.Tensor:
        """Return scalar dissipation ``D(qdot) = 0.5 qdot^T C(qdot) qdot``.

        The quadratic form uses ``qdot`` bounded by ``input_scale`` so the
        Rayleigh functional cannot blow up on tail samples. ``dD/dqdot`` still
        resembles linear damping for ``|qdot| << input_scale``; outside that
        regime the saturation keeps the gradient bounded.
        """

        qdot_bounded = _bounded_input(qdot, self.input_scale)
        c = self.damping_matrix(qdot)
        return 0.5 * torch.einsum("...i,...ij,...j->...", qdot_bounded, c, qdot_bounded)


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
        input_scale: float = 5.0,
    ) -> None:
        super().__init__()
        if input_scale <= 0.0:
            raise ValueError("input_scale must be > 0")
        self.q_dim = q_dim
        self.action_dim = action_dim
        self.input_scale = input_scale
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
        q_bounded = _bounded_input(q, self.input_scale)
        flat = q_bounded.reshape(-1, self.q_dim)
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

    with ``L = 0.5 qdot^T M(q) qdot - V(q)`` (so ``qdot`` is treated as fixed
    when taking ``dL/dq``) and ``F_ext = -dD/d(qdot) + B(q) u``.

    ``M_dot(q) qdot`` is computed as ``J_q(M(q) qdot) @ qdot``. We do this
    with per-component ``autograd.grad`` calls — one per ``q`` dimension —
    rather than ``autograd.functional.jvp``. The JVP API's double-backward
    trick hits a reliably reproducible MPS bug (``AcceleratorError: index
    N is out of bounds``) in PyTorch 2.x; the per-component loop is
    equivalent for small ``d`` (cartpole has ``d = 2``, so two extra grad
    calls per step) and stays on the accelerator.

    Gradients flow back to all module parameters via ``create_graph`` on the
    inner autograd calls; outer-graph gradient wrt ``q`` and ``qdot``
    propagates through the single enable-grad branch, so the full
    imagination rollout stays end-to-end differentiable.
    """

    if q.shape != qdot.shape:
        raise ValueError(f"q and qdot shapes must match; got {q.shape} vs {qdot.shape}")
    if u.shape[:-1] != q.shape[:-1] or u.shape[-1] != dynamics.actuation.action_dim:
        raise ValueError(
            f"u must broadcast-align with q batch dims and have last dim="
            f"{dynamics.actuation.action_dim}; got u.shape={u.shape}",
        )

    needs_outer_grad = torch.is_grad_enabled() and (
        q.requires_grad or qdot.requires_grad or any(
            p.requires_grad for p in _iter_dynamics_parameters(dynamics)
        )
    )

    with torch.enable_grad():
        # Use q/qdot directly when they carry grad so the outer graph stays
        # connected; otherwise attach a local leaf so the inner autograd
        # calls still function (e.g. during eval-mode imagination or CEM).
        q_in = q if q.requires_grad else q.clone().requires_grad_(True)
        qdot_ref = qdot.detach()  # holding qdot fixed when differentiating wrt q
        # F2: bound the qdot used in the kinetic / momentum / M_dot quadratic
        # forms. M, V, B all bound their own inputs; we still need to cap the
        # outer ``qdot^T M qdot``-style quadratics so a pathological encoder
        # output can't blow up q_ddot even when every MLP is well-behaved.
        qdot_bound = dynamics.dissipation.input_scale if dynamics.dissipation is not None else 20.0
        qdot_b = _bounded_input(qdot_ref, qdot_bound)

        mass = dynamics.mass_matrix(q_in)  # [..., d, d]
        potential = dynamics.potential(q_in)  # [...]
        kinetic = 0.5 * torch.einsum("...i,...ij,...j->...", qdot_b, mass, qdot_b)
        lagrangian_sum = (kinetic - potential).sum()
        (dL_dq,) = torch.autograd.grad(
            lagrangian_sum,
            q_in,
            create_graph=needs_outer_grad,
            retain_graph=True,
        )

        q_dim = q_in.shape[-1]
        momentum = torch.einsum("...ij,...j->...i", mass, qdot_b)  # [..., d]
        m_dot_qdot_components: list[torch.Tensor] = []
        for i in range(q_dim):
            is_last = i == q_dim - 1
            (grad_component,) = torch.autograd.grad(
                momentum[..., i].sum(),
                q_in,
                create_graph=needs_outer_grad,
                retain_graph=not is_last or needs_outer_grad,
            )
            m_dot_qdot_components.append((grad_component * qdot_b).sum(dim=-1))
        m_dot_qdot = torch.stack(m_dot_qdot_components, dim=-1)

        if dynamics.dissipation is not None:
            qdot_in = qdot if qdot.requires_grad else qdot.clone().requires_grad_(True)
            dissipation_value = dynamics.dissipation(qdot_in).sum()
            (dD_dqdot,) = torch.autograd.grad(
                dissipation_value,
                qdot_in,
                create_graph=needs_outer_grad,
            )
        else:
            dD_dqdot = torch.zeros_like(qdot)

        b_matrix = dynamics.actuation(q_in)
        tau = torch.einsum("...ij,...j->...i", b_matrix, u)

        rhs = dL_dq - m_dot_qdot - dD_dqdot + tau
        # Prefer ``linalg.inv(M) @ rhs`` over ``linalg.solve`` because
        # ``linalg.solve`` is buggy on Apple MPS inside an autograd-grad-with-
        # create-graph loop: it reliably corrupts kernel dispatch and surfaces
        # later as ``AcceleratorError: index N is out of bounds``. Inversion
        # is numerically fine here — ``M`` is Cholesky-parameterized PD with an
        # ``eps`` ridge, so conditioning is bounded, and the matrix is small
        # (cartpole ``d=2``).
        mass_inv = torch.linalg.inv(mass)
        q_ddot = torch.einsum("...ij,...j->...i", mass_inv, rhs)

    if not needs_outer_grad:
        q_ddot = q_ddot.detach()
    return q_ddot


def _iter_dynamics_parameters(dynamics: LagrangianDynamics):
    """Yield every parameter across the four Lagrangian primitives."""

    for module in (
        dynamics.mass_matrix,
        dynamics.potential,
        dynamics.actuation,
    ):
        yield from module.parameters()
    if dynamics.dissipation is not None:
        yield from dynamics.dissipation.parameters()


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
