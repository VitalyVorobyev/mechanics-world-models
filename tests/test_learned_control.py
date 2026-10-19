"""Learned-model planning backend, closed-loop episode and training helpers (E-007)."""

from __future__ import annotations

import numpy as np
import torch

from control.learned import LearnedBackend, TrainConfig, episodes_to_states, run_episode, train_model
from control.mppi import MPPI, MPPIConfig
from control.true_model import MujocoBackend, get_state
from oracles.context import simulate_state_episodes
from oracles.dynamics import DynamicsWrapper


def test_learned_backend_matches_model_rollout() -> None:
    torch.manual_seed(0)
    for form in ("lagrangian", "mlp"):
        model = DynamicsWrapper(form, dt=0.02, n_substeps=2, actuated_dims=(0,) if form == "lagrangian" else None)
        q, qdot = np.array([0.1, 3.0]), np.array([0.2, -0.5])
        actions = np.random.default_rng(0).uniform(-1, 1, size=(4, 6, 1))
        _, cart_x = LearnedBackend(model).evaluate((q, qdot), actions)
        with torch.enable_grad():
            expected = model.rollout(
                torch.tensor(q, dtype=torch.float32).expand(4, -1),
                torch.tensor(qdot, dtype=torch.float32).expand(4, -1),
                torch.tensor(actions, dtype=torch.float32),
            ).detach()
        assert np.allclose(cart_x, expected[..., 0].numpy(), atol=1e-5)


def test_run_episode_and_training_round_trip() -> None:
    holder: dict = {}

    def full_state(env, last_action=None):
        holder.setdefault("backend", MujocoBackend(env.physics.model.ptr, 2, 2))
        return get_state(env.physics.model.ptr, env.physics.data.ptr)

    class Lazy:
        def evaluate(self, state, actions):
            return holder["backend"].evaluate(state, actions)

    planner = MPPI(MPPIConfig(horizon=5, samples=8), seed=0)
    episode = run_episode(0, planner, Lazy(), full_state, action_noise=0.1, noise_seed=1)
    assert episode.qpos.shape == (501, 2) and episode.actions.shape == (500, 1) and episode.rewards.shape == (500,)
    assert np.all(np.abs(episode.actions) <= 1.0)

    data = episodes_to_states([episode], frame_dt=0.02)
    random = simulate_state_episodes(None, episodes=1, seed=0)
    assert random.qpos.shape == data.qpos.shape
    model, loss = train_model("lagrangian", data, TrainConfig(steps=3, batch_size=16), seed=0)
    assert np.isfinite(loss)
    reward, _ = LearnedBackend(model).evaluate((episode.qpos[0], episode.qvel[0]), np.zeros((2, 3, 1)))
    assert reward.shape == (2, 3)

