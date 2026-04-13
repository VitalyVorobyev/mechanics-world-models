"""Shape + gradient integration tests for MechanicsWorldModel."""

from __future__ import annotations

import pytest
import torch

from models.mechanics import MechanicsWorldModel


def _make_model(
    *,
    reward_head: bool = False,
    with_dissipation: bool = True,
    q_dim: int = 2,
    nuisance_dim: int = 8,
    action_dim: int = 1,
) -> MechanicsWorldModel:
    torch.manual_seed(0)
    return MechanicsWorldModel(
        action_dim=action_dim,
        q_dim=q_dim,
        nuisance_dim=nuisance_dim,
        embedding_size=32,
        encoder_head_hidden_size=16,
        mass_matrix_hidden_size=16,
        potential_hidden_size=16,
        dissipation_hidden_size=16,
        actuation_hidden_size=16,
        dynamics_hidden_layers=1,
        image_size=84,
        reward_head=reward_head,
        with_dissipation=with_dissipation,
        dt=0.02,
    )


def test_forward_shapes_without_imagination() -> None:
    model = _make_model()
    obs = torch.rand(2, 5, 3, 84, 84)
    actions = torch.randn(2, 5, 1)
    out = model(obs, actions)
    assert out.reconstructions.shape == (2, 5, 3, 84, 84)
    assert out.rollout.posterior.q_mean.shape == (2, 5, 2)
    assert out.rollout.prior_q_mean.shape == (2, 5, 2)
    assert out.rollout.q_sampled.shape == (2, 5, 2)


def test_forward_with_imagination_shapes() -> None:
    model = _make_model(reward_head=True)
    obs = torch.rand(3, 10, 3, 84, 84)
    actions = torch.randn(3, 10, 1)
    out = model(obs, actions, imagination_context_steps=3, imagination_horizon=5)
    assert out.imagined_reconstructions is not None
    assert out.imagined_reconstructions.shape == (3, 5, 3, 84, 84)
    assert out.imagined_features.shape == (3, 5, 2 * 2 + 8)
    assert out.imagined_q.shape == (3, 5, 2)
    assert out.imagined_target_start == 2
    assert out.reward_predictions.shape == (3, 10)
    assert out.imagined_reward_predictions.shape == (3, 5)


def test_imagination_requires_sufficient_sequence_length() -> None:
    model = _make_model()
    obs = torch.rand(2, 3, 3, 84, 84)
    actions = torch.randn(2, 3, 1)
    with pytest.raises(ValueError, match="imagination requires"):
        model(obs, actions, imagination_context_steps=2, imagination_horizon=5)


def test_backward_populates_gradients_across_branches() -> None:
    model = _make_model(reward_head=True)
    obs = torch.rand(2, 6, 3, 84, 84)
    actions = torch.randn(2, 6, 1)
    out = model(obs, actions, imagination_context_steps=2, imagination_horizon=3)
    (out.reconstructions.mean() + out.imagined_reconstructions.mean()).backward()
    for name, module in (
        ("encoder", model.encoder),
        ("decoder", model.decoder),
        ("mass_matrix", model.mass_matrix_module),
        ("actuation", model.actuation_module),
        ("transition", model.transition),
    ):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"no parameter in {name} received a gradient"
        for g in grads:
            assert torch.isfinite(g).all(), f"non-finite grad in {name}"


def test_initial_posterior_shape_matches() -> None:
    model = _make_model(nuisance_dim=4)
    frame = torch.rand(2, 3, 84, 84)
    q0, qdot0, z0 = model.initial_posterior(frame)
    assert q0.shape == (2, 2) and qdot0.shape == (2, 2) and z0.shape == (2, 4)


def test_imagine_rollout_features_differentiable() -> None:
    model = _make_model(nuisance_dim=4)
    q = torch.randn(3, 2, requires_grad=True)
    qdot = torch.randn(3, 2, requires_grad=True)
    z = torch.randn(3, 4, requires_grad=True)
    actions = torch.randn(3, 4, 1)
    feats = model.imagine_rollout_features(q, qdot, z, actions)
    assert feats.shape == (3, 4, 2 * 2 + 4)
    feats.sum().backward()
    assert q.grad is not None
    assert qdot.grad is not None
    assert z.grad is not None


def test_state_dict_captures_dynamics_parameters() -> None:
    model = _make_model()
    state_dict = model.state_dict()
    # All four Lagrangian primitives should appear in the state dict under
    # their registered module names.
    expected_prefixes = {
        "mass_matrix_module.",
        "potential_module.",
        "dissipation_module.",
        "actuation_module.",
        "transition.",
        "encoder.",
        "decoder.",
    }
    present = {key.split(".", maxsplit=1)[0] + "." for key in state_dict}
    for prefix in expected_prefixes:
        assert any(p == prefix for p in present), f"missing {prefix} in state_dict"


def test_no_dissipation_drops_dissipation_module() -> None:
    model = _make_model(with_dissipation=False)
    assert not hasattr(model, "dissipation_module")
    obs = torch.rand(1, 4, 3, 84, 84)
    actions = torch.randn(1, 4, 1)
    out = model(obs, actions, imagination_context_steps=1, imagination_horizon=2)
    assert out.imagined_reconstructions.shape == (1, 2, 3, 84, 84)
