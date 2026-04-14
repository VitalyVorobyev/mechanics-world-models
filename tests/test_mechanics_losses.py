"""Tests for the mechanics world-model composite loss + factored KL."""

from __future__ import annotations

import pytest
import torch

from models.mechanics import (
    MechanicsWorldModel,
    compute_factored_kl,
    compute_mechanics_world_model_loss,
    derive_qdot_from_q,
)


def _make_model(reward_head: bool = False) -> MechanicsWorldModel:
    torch.manual_seed(0)
    return MechanicsWorldModel(
        action_dim=1,
        q_dim=2,
        nuisance_dim=4,
        embedding_size=32,
        encoder_head_hidden_size=16,
        mass_matrix_hidden_size=16,
        potential_hidden_size=16,
        dissipation_hidden_size=16,
        actuation_hidden_size=16,
        dynamics_hidden_layers=1,
        image_size=84,
        reward_head=reward_head,
        dt=0.02,
    )


def test_derive_qdot_central_difference_on_linear_trajectory() -> None:
    """For q_t = slope * t, central diff returns exactly the slope."""

    dt = 0.01
    slope = torch.tensor([[[0.4, -0.2]]])  # [1, 1, d]
    t = torch.arange(10, dtype=torch.float32).view(1, -1, 1)  # [1, T, 1]
    q_mean = slope * t * dt  # [1, T, d]
    q_std = torch.full_like(q_mean, 0.1)
    qdot_mean, qdot_std = derive_qdot_from_q(q_mean, q_std, dt=dt)
    assert qdot_mean.shape == q_mean.shape
    # Interior: slope exactly. Boundary: also exact for a linear signal.
    expected = slope.expand_as(qdot_mean)
    torch.testing.assert_close(qdot_mean, expected, atol=1e-5, rtol=0)
    # Variance pushforward sanity: interior std = sqrt(2) * q_std / (2*dt).
    expected_interior = (q_std[:, 0, :].pow(2) + q_std[:, 0, :].pow(2)).sqrt() / (2.0 * dt)
    torch.testing.assert_close(qdot_std[:, 1:-1, :], expected_interior.unsqueeze(1).expand_as(qdot_std[:, 1:-1, :]), atol=1e-5, rtol=0)


def test_derive_qdot_requires_at_least_two_frames() -> None:
    with pytest.raises(ValueError, match="T >= 2"):
        derive_qdot_from_q(torch.zeros(1, 1, 2), torch.ones(1, 1, 2), dt=0.01)


def test_derive_qdot_rejects_nonpositive_dt() -> None:
    with pytest.raises(ValueError, match="dt"):
        derive_qdot_from_q(torch.zeros(1, 5, 2), torch.ones(1, 5, 2), dt=0.0)


def test_factored_kl_sums_to_total() -> None:
    torch.manual_seed(3)
    shape = (2, 4, 2)
    z_shape = (2, 4, 4)
    kw = dict(
        posterior_q_mean=torch.randn(*shape),
        posterior_q_std=torch.rand(*shape) + 0.1,
        posterior_qdot_mean=torch.randn(*shape),
        posterior_qdot_std=torch.rand(*shape) + 0.1,
        posterior_z_mean=torch.randn(*z_shape),
        posterior_z_std=torch.rand(*z_shape) + 0.1,
        prior_q_mean=torch.randn(*shape),
        prior_q_std=torch.rand(*shape) + 0.1,
        prior_qdot_mean=torch.randn(*shape),
        prior_qdot_std=torch.rand(*shape) + 0.1,
        prior_z_mean=torch.randn(*z_shape),
        prior_z_std=torch.rand(*z_shape) + 0.1,
        kl_balance_alpha=0.8,
        kl_free_nats_mech=0.0,
        kl_free_nats_nuisance=0.0,
    )
    parts = compute_factored_kl(**kw)
    assert torch.allclose(
        parts["kl_loss"],
        parts["kl_q_loss"] + parts["kl_qdot_loss"] + parts["kl_z_loss"],
        atol=1e-6,
    )


def test_free_nats_floor_clamps_kl() -> None:
    """A large free-nats floor should force each branch's kl_loss >= floor."""

    torch.manual_seed(4)
    shape = (2, 3, 2)
    z_shape = (2, 3, 4)
    parts = compute_factored_kl(
        posterior_q_mean=torch.zeros(*shape),
        posterior_q_std=torch.ones(*shape),
        posterior_qdot_mean=torch.zeros(*shape),
        posterior_qdot_std=torch.ones(*shape),
        posterior_z_mean=torch.zeros(*z_shape),
        posterior_z_std=torch.ones(*z_shape),
        prior_q_mean=torch.zeros(*shape),
        prior_q_std=torch.ones(*shape),
        prior_qdot_mean=torch.zeros(*shape),
        prior_qdot_std=torch.ones(*shape),
        prior_z_mean=torch.zeros(*z_shape),
        prior_z_std=torch.ones(*z_shape),
        kl_balance_alpha=0.8,
        kl_free_nats_mech=2.0,
        kl_free_nats_nuisance=5.0,
    )
    # With identical posterior=prior the raw KL is 0 but clamped to the floor.
    assert parts["kl_q_loss"].item() == pytest.approx(2.0, rel=1e-5)
    assert parts["kl_qdot_loss"].item() == pytest.approx(2.0, rel=1e-5)
    assert parts["kl_z_loss"].item() == pytest.approx(5.0, rel=1e-5)
    assert parts["kl_q_free_nats_active"].item() == 1.0
    assert parts["kl_z_free_nats_active"].item() == 1.0


def test_mechanics_loss_full_composite_shapes() -> None:
    model = _make_model(reward_head=True)
    obs = torch.rand(2, 8, 3, 84, 84)
    next_obs = torch.rand(2, 8, 3, 84, 84)
    actions = torch.randn(2, 8, 1)
    rewards = torch.randn(2, 8)
    out = model(obs, actions, imagination_context_steps=2, imagination_horizon=5)
    losses = compute_mechanics_world_model_loss(
        outputs=out,
        observations=obs,
        next_observations=next_obs,
        rewards=rewards,
        reconstruction_weight=1.0,
        foreground_reconstruction_weight=0.5,
        imagination_reconstruction_weight=0.5,
        foreground_imagination_reconstruction_weight=1.0,
        kl_weight=1.0,
        kl_free_nats_mech=0.5,
        kl_free_nats_nuisance=3.0,
        reward_weight=0.5,
        imagined_reward_weight=0.2,
    )
    for key in (
        "total_loss",
        "reconstruction_loss",
        "foreground_reconstruction_loss",
        "imagination_reconstruction_loss",
        "foreground_imagination_reconstruction_loss",
        "kl_loss",
        "kl_q_loss",
        "kl_qdot_loss",
        "kl_z_loss",
        "reward_loss",
        "imagined_reward_loss",
    ):
        assert key in losses, f"missing {key}"
        assert losses[key].dim() == 0
        assert torch.isfinite(losses[key])
    losses["total_loss"].backward()
    # Encoder, decoder, dynamics, transition, and reward head should all get
    # some gradient.
    for name, module in (
        ("encoder", model.encoder),
        ("decoder", model.decoder),
        ("mass_matrix", model.mass_matrix_module),
        ("transition", model.transition),
        ("reward_head", model.reward_head),
    ):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"no grads in {name}"
        for g in grads:
            assert torch.isfinite(g).all(), f"non-finite grad in {name}"
