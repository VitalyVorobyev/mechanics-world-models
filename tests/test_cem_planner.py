"""Unit tests for the CEM/MPC planner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models import VisualWorldModel
from planning import CEMPlanner, CEMPlanResult


def _make_model(reward_head: bool = True) -> VisualWorldModel:
    return VisualWorldModel(
        action_dim=2,
        embedding_size=32,
        hidden_size=16,
        latent_size=8,
        reward_head=reward_head,
    )


def test_cem_planner_requires_reward_head() -> None:
    model = _make_model(reward_head=False)
    with pytest.raises(ValueError, match="reward_head"):
        CEMPlanner(
            model,
            horizon=3,
            action_low=np.zeros(2),
            action_high=np.ones(2),
        )


def test_cem_plan_returns_in_bounds_action_sequence() -> None:
    torch.manual_seed(0)
    model = _make_model()
    planner = CEMPlanner(
        model,
        horizon=4,
        action_low=np.asarray([-1.0, -2.0]),
        action_high=np.asarray([1.0, 2.0]),
        n_samples=64,
        n_iters=2,
        elite_frac=0.25,
        seed=0,
    )
    deter = torch.zeros(model.hidden_size)
    stoch = torch.zeros(model.latent_size)

    result = planner.plan_action_sequence(deter, stoch)

    assert isinstance(result, CEMPlanResult)
    assert result.action_sequence.shape == (4, 2)
    assert torch.all(result.action_sequence[:, 0] >= -1.0 - 1e-6)
    assert torch.all(result.action_sequence[:, 0] <= 1.0 + 1e-6)
    assert torch.all(result.action_sequence[:, 1] >= -2.0 - 1e-6)
    assert torch.all(result.action_sequence[:, 1] <= 2.0 + 1e-6)
    assert np.isfinite(result.elite_score_mean)
    assert np.isfinite(result.elite_score_std)


def test_cem_plan_first_action_matches_sequence_head() -> None:
    torch.manual_seed(0)
    model = _make_model()
    planner = CEMPlanner(
        model,
        horizon=3,
        action_low=np.zeros(2),
        action_high=np.ones(2),
        n_samples=32,
        n_iters=2,
        seed=1,
    )
    deter = torch.zeros(model.hidden_size)
    stoch = torch.zeros(model.latent_size)

    sequence = planner.plan_action_sequence(deter, stoch).action_sequence
    # plan() draws fresh CEM samples — to check head equivalence we call it on
    # a fresh planner with the same seed.
    planner_b = CEMPlanner(
        model,
        horizon=3,
        action_low=np.zeros(2),
        action_high=np.ones(2),
        n_samples=32,
        n_iters=2,
        seed=1,
    )
    first_action = planner_b.plan(deter, stoch)
    assert torch.allclose(first_action, sequence[0], atol=1e-6)


def test_cem_converges_toward_a_synthetic_optimum() -> None:
    """With a reward head wired to favor a known feature value, elite score should rise."""

    torch.manual_seed(0)
    model = _make_model()
    # Replace reward head with a deterministic linear head that scores high
    # when the first hidden-state component is large and positive.
    new_head = torch.nn.Linear(model.hidden_size + model.latent_size, 1, bias=False)
    weight = torch.zeros(1, model.hidden_size + model.latent_size)
    weight[0, 0] = 1.0
    with torch.no_grad():
        new_head.weight.copy_(weight)
    model.reward_head = new_head

    planner = CEMPlanner(
        model,
        horizon=4,
        action_low=np.asarray([-1.0, -1.0]),
        action_high=np.asarray([1.0, 1.0]),
        n_samples=128,
        n_iters=4,
        elite_frac=0.125,
        seed=0,
    )
    deter = torch.zeros(model.hidden_size)
    stoch = torch.zeros(model.latent_size)

    # Run CEM once and capture per-iteration elite score by repeated single-iter calls.
    scores: list[float] = []
    for n_iters in (1, 2, 3, 4):
        single_planner = CEMPlanner(
            model,
            horizon=4,
            action_low=np.asarray([-1.0, -1.0]),
            action_high=np.asarray([1.0, 1.0]),
            n_samples=128,
            n_iters=n_iters,
            elite_frac=0.125,
            seed=0,
        )
        scores.append(single_planner.plan_action_sequence(deter, stoch).elite_score_mean)

    # Later iterations should score at least as well as the first (CEM is monotonically
    # non-decreasing in expected elite score for reasonable proposal distributions).
    assert scores[-1] >= scores[0]


def test_cem_planner_validates_action_bounds_shape() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="action_low/action_high"):
        CEMPlanner(
            model,
            horizon=3,
            action_low=np.zeros(3),
            action_high=np.ones(3),
        )
    with pytest.raises(ValueError, match="action_high must exceed"):
        CEMPlanner(
            model,
            horizon=3,
            action_low=np.ones(2),
            action_high=np.zeros(2),
        )


def test_initial_posterior_and_step_produce_consistent_states() -> None:
    """Online MPC helpers: shapes match across the (init → step) loop."""

    torch.manual_seed(0)
    model = _make_model()
    obs = torch.rand(1, 3, 84, 84)
    deter, stoch = model.initial_posterior(obs)
    assert deter.shape == (1, model.hidden_size)
    assert stoch.shape == (1, model.latent_size)
    # Deterministic state at t=0 must be exactly zero.
    assert torch.allclose(deter, torch.zeros_like(deter))

    action = torch.zeros(1, model.action_dim)
    next_obs = torch.rand(1, 3, 84, 84)
    deter_next, stoch_next = model.posterior_step(deter, stoch, action, next_obs)
    assert deter_next.shape == (1, model.hidden_size)
    assert stoch_next.shape == (1, model.latent_size)
    # h_t must change after one transition (GRUCell with non-zero stoch input typically does).
    assert not torch.allclose(deter_next, deter)
