"""Symplectic-style integrator for learned Lagrangian dynamics.

Semi-implicit Euler advances ``(q, qdot)`` by first updating ``qdot`` using the
acceleration evaluated at the current ``q``, then updating ``q`` with the new
``qdot``. On a conservative (``D = 0``) Hamiltonian system this is symplectic
up to a bounded modified Hamiltonian, so energy drift stays O(dt^2) rather
than O(t) — the reason we prefer it to explicit Euler for long rollouts.
"""

from __future__ import annotations

import torch

from models.mechanics.lagrangian import LagrangianDynamics, forward_acceleration


def semi_implicit_euler_step(
    dynamics: LagrangianDynamics,
    q: torch.Tensor,
    qdot: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    n_substeps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance ``(q, qdot)`` by ``dt`` using semi-implicit Euler.

    With ``n_substeps > 1`` the same action ``u`` is held constant and the
    integrator is applied ``n_substeps`` times with step ``dt / n_substeps``;
    this adds stiffness margin at the cost of proportional compute.
    """

    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    h = dt / n_substeps
    for _ in range(n_substeps):
        q_ddot = forward_acceleration(dynamics, q, qdot, u)
        qdot = qdot + h * q_ddot
        q = q + h * qdot
    return q, qdot
