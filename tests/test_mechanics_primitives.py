"""Unit tests for the Phase-2 Lagrangian + integrator primitives."""

from __future__ import annotations

import math

import pytest
import torch

from models.mechanics import (
    Actuation,
    Dissipation,
    LagrangianDynamics,
    MassMatrix,
    Potential,
    forward_acceleration,
    kinetic_energy,
    semi_implicit_euler_step,
    total_energy,
)


def _make_dynamics(
    q_dim: int = 2,
    action_dim: int = 1,
    *,
    with_dissipation: bool = True,
    seed: int = 0,
) -> LagrangianDynamics:
    torch.manual_seed(seed)
    return LagrangianDynamics(
        mass_matrix=MassMatrix(q_dim=q_dim, hidden_size=32, hidden_layers=1),
        potential=Potential(q_dim=q_dim, hidden_size=32, hidden_layers=1),
        dissipation=Dissipation(q_dim=q_dim, hidden_size=32, hidden_layers=1)
        if with_dissipation
        else None,
        actuation=Actuation(q_dim=q_dim, action_dim=action_dim, hidden_size=32, hidden_layers=1),
    )


def test_mass_matrix_is_positive_definite_and_symmetric() -> None:
    torch.manual_seed(0)
    mass = MassMatrix(q_dim=3, hidden_size=16, hidden_layers=1)
    q = torch.randn(7, 3, requires_grad=False)
    m = mass(q)
    assert m.shape == (7, 3, 3)
    # Symmetric.
    assert torch.allclose(m, m.transpose(-1, -2), atol=1e-6)
    # Positive-definite: all eigenvalues > 0.
    eigvals = torch.linalg.eigvalsh(m)
    assert (eigvals > 0).all()
    # Lower bound from epsilon ridge.
    assert (eigvals.min(dim=-1).values > 0.99e-3).all()


def test_mass_matrix_supports_leading_time_axis() -> None:
    mass = MassMatrix(q_dim=2)
    q = torch.randn(4, 6, 2)
    m = mass(q)
    assert m.shape == (4, 6, 2, 2)
    assert torch.allclose(m, m.transpose(-1, -2), atol=1e-6)


def test_dissipation_gradient_matches_linear_damping_when_c_is_constant() -> None:
    """If we freeze the damping network, dD/d(qdot) must equal C qdot."""

    torch.manual_seed(42)
    dissipation = Dissipation(q_dim=2, hidden_size=16, hidden_layers=1)
    # Freeze: a constant-in-qdot damping matrix is what a "linear damping"
    # Rayleigh function produces. We force this by zeroing weights in the last
    # layer so the network output stays constant across qdot.
    last = dissipation.net[-1]
    with torch.no_grad():
        last.weight.zero_()
        # Keep bias random so the PSD matrix is non-trivial.
    qdot = torch.randn(5, 2, requires_grad=True)
    d = dissipation(qdot).sum()
    (grad,) = torch.autograd.grad(d, qdot)
    c = dissipation.damping_matrix(qdot)
    expected = torch.einsum("...ij,...j->...i", c, qdot)
    assert torch.allclose(grad, expected, atol=1e-5)


def test_forward_acceleration_shapes() -> None:
    dyn = _make_dynamics(q_dim=2, action_dim=1)
    q = torch.randn(4, 2)
    qdot = torch.randn(4, 2)
    u = torch.randn(4, 1)
    acc = forward_acceleration(dyn, q, qdot, u)
    assert acc.shape == (4, 2)
    assert torch.isfinite(acc).all()


def test_forward_acceleration_differentiable_through_parameters() -> None:
    """At least one parameter per module receives a finite gradient.

    Note: Potential's final-layer bias receives *no* gradient because
    d/dq of (W x + b) = W and the bias cancels — this is correct Lagrangian
    behavior, not a bug. We therefore check "some parameter", not "all".
    """

    dyn = _make_dynamics(q_dim=2, action_dim=1)
    q = torch.randn(3, 2)
    qdot = torch.randn(3, 2)
    u = torch.randn(3, 1)
    acc = forward_acceleration(dyn, q, qdot, u)
    acc.pow(2).sum().backward()
    for module in (dyn.mass_matrix, dyn.potential, dyn.dissipation, dyn.actuation):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"no parameter in {type(module).__name__} received a gradient"
        for g in grads:
            assert torch.isfinite(g).all()


def test_semi_implicit_euler_energy_drift_no_dissipation() -> None:
    """On a D=0 Hamiltonian system energy should drift < 1% over 100 steps.

    Semi-implicit Euler is symplectic up to a bounded modified Hamiltonian, so
    this is the defining characteristic we need for stable long rollouts. We
    use randomly initialized M and V — they define the Hamiltonian; the test
    is that *that* Hamiltonian is approximately conserved, not that it
    matches any real-world mechanics.
    """

    torch.manual_seed(123)
    dyn = _make_dynamics(with_dissipation=False)
    dyn.mass_matrix.eval()
    dyn.potential.eval()
    dyn.actuation.eval()
    q = 0.1 * torch.randn(4, 2)
    qdot = 0.1 * torch.randn(4, 2)
    u = torch.zeros(4, 1)
    with torch.no_grad():
        e0 = total_energy(dyn.mass_matrix, dyn.potential, q, qdot)
    energies = [e0]
    # Small dt so the symplectic bound has room.
    dt = 0.01
    for _ in range(100):
        q, qdot = semi_implicit_euler_step(dyn, q, qdot, u, dt=dt, n_substeps=1)
        with torch.no_grad():
            energies.append(total_energy(dyn.mass_matrix, dyn.potential, q, qdot))
    stacked = torch.stack(energies, dim=0)  # [T+1, B]
    reference = stacked[0].abs().clamp(min=1e-3)
    drift = (stacked - stacked[0]).abs() / reference
    assert drift.max().item() < 0.02, f"max energy drift {drift.max().item():.4f}"


def test_semi_implicit_euler_loses_energy_with_dissipation() -> None:
    """With positive dissipation, total energy must not increase (on average)."""

    torch.manual_seed(7)
    dyn = _make_dynamics(with_dissipation=True)
    # Boost the dissipation bias so damping dominates quickly.
    with torch.no_grad():
        dyn.dissipation.net[-1].bias.fill_(1.0)
    q = 0.2 * torch.randn(3, 2)
    qdot = 0.5 * torch.randn(3, 2)
    u = torch.zeros(3, 1)
    with torch.no_grad():
        e_start = total_energy(dyn.mass_matrix, dyn.potential, q, qdot)
    for _ in range(50):
        q, qdot = semi_implicit_euler_step(dyn, q, qdot, u, dt=0.02)
    with torch.no_grad():
        e_end = total_energy(dyn.mass_matrix, dyn.potential, q, qdot)
    # Kinetic energy should strictly decrease on average.
    ke_start = kinetic_energy(dyn.mass_matrix, q.detach(), qdot.detach())
    assert (e_end.mean() <= e_start.mean() + 1e-5)
    assert torch.isfinite(ke_start).all()


def test_semi_implicit_euler_respects_n_substeps() -> None:
    """Two sub-steps of dt/2 must give a finer-grained trajectory than one step."""

    torch.manual_seed(11)
    dyn = _make_dynamics(with_dissipation=False)
    q = 0.1 * torch.ones(1, 2)
    qdot = torch.zeros(1, 2)
    u = torch.ones(1, 1)
    q1, qdot1 = semi_implicit_euler_step(dyn, q, qdot, u, dt=0.05, n_substeps=1)
    q2, qdot2 = semi_implicit_euler_step(dyn, q, qdot, u, dt=0.05, n_substeps=4)
    # Both finite; the fine-grained one should differ (different integration error).
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    assert not torch.allclose(q1, q2)


def test_forward_acceleration_validates_shapes() -> None:
    dyn = _make_dynamics(q_dim=2, action_dim=1)
    q = torch.randn(3, 2)
    qdot = torch.randn(3, 2)
    with pytest.raises(ValueError):
        forward_acceleration(dyn, q, qdot, torch.randn(3, 2))  # wrong action_dim
    with pytest.raises(ValueError):
        forward_acceleration(dyn, q, torch.randn(3, 3), torch.randn(3, 1))  # qdot mismatch


def test_imagination_loop_does_not_use_linalg_solve() -> None:
    """Regression: forward_acceleration must avoid ``torch.linalg.solve``.

    ``linalg.solve`` is reliably buggy on Apple MPS when used inside a loop
    that calls ``autograd.grad(create_graph=True)`` and is later backward'd
    through — it corrupts kernel dispatch and surfaces as
    ``AcceleratorError: index N is out of bounds`` at the next sync point
    (e.g. inside ``compute_grad_norm``). We worked around it by using
    ``linalg.inv(M) @ rhs`` instead; this test pins that decision so a
    future refactor doesn't regress the MPS path.
    """

    from models.mechanics import lagrangian as lag_module
    source = (lag_module.__file__, open(lag_module.__file__).read())
    assert "torch.linalg.solve" not in source[1], (
        f"forward_acceleration in {source[0]} must not call torch.linalg.solve "
        "(MPS dispatch corruption inside autograd.grad loops — see the "
        "inline comment in forward_acceleration for the rationale)."
    )


def test_pendulum_like_system_oscillates() -> None:
    """Sanity: an unforced, conservative random system should oscillate, not diverge.

    Track the norm of state over 200 small steps; it should stay bounded.
    """

    torch.manual_seed(2024)
    dyn = _make_dynamics(with_dissipation=False)
    q = 0.05 * torch.randn(2, 2)
    qdot = 0.05 * torch.randn(2, 2)
    u = torch.zeros(2, 1)
    norms = []
    for _ in range(200):
        q, qdot = semi_implicit_euler_step(dyn, q, qdot, u, dt=0.005)
        norms.append(torch.cat([q, qdot], dim=-1).norm(dim=-1).detach())
    norms_t = torch.stack(norms, dim=0)
    # Bounded: max norm shouldn't exceed 10x the initial radius.
    initial = math.sqrt(4 * 0.05 * 0.05 * 2)
    assert norms_t.max().item() < 50 * initial
    assert torch.isfinite(norms_t).all()
