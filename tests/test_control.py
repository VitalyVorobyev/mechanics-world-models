"""Closed-loop control pieces: reward, MPPI, MuJoCo planning backend."""

from __future__ import annotations

import numpy as np
import torch

from control.mppi import MPPI, MPPIConfig
import pytest

from control.reward import poles_reward, rail_penalty, swingup_reward
from control.true_model import MujocoBackend, get_state
from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig


def test_reward_matches_dm_control() -> None:
    env = DmControlPixelEnv(PixelEnvConfig(seed=3, action_repeat=1))
    env.reset_without_render()
    rng = np.random.default_rng(0)
    for _ in range(100):
        action = rng.uniform(-1.0, 1.0, size=1)
        state, reward = env.step_state(action)
        ours = swingup_reward(state["qpos"][0], np.cos(state["qpos"][1]), state["qvel"][1], action[0])
        assert abs(reward - ours) < 1e-12


@pytest.mark.parametrize("task", ["two_poles", "three_poles"])
def test_poles_reward_matches_dm_control(task: str) -> None:
    env = DmControlPixelEnv(PixelEnvConfig(env_name=f"cartpole-{task}", seed=3, action_repeat=1))
    env.reset_without_render()
    rng = np.random.default_rng(0)
    for _ in range(200):
        action = rng.uniform(-1.0, 1.0, size=1)
        state, reward = env.step_state(action)
        ours = poles_reward(state["qpos"][0], state["qpos"][1:], state["qvel"][1:], action[0])
        assert abs(reward - ours) < 1e-12


def test_poles_reward_reduces_to_one_pole_and_accepts_torch() -> None:
    rng = np.random.default_rng(1)
    x, th, thd, u = rng.normal(size=(4, 50))
    assert np.allclose(poles_reward(x, th[:, None], thd[:, None], u), swingup_reward(x, np.cos(th), thd, u))
    angles, vel = rng.normal(size=(2, 50, 3))
    expected = poles_reward(x, angles, vel, u)
    got = poles_reward(*[torch.as_tensor(a) for a in (x, angles, vel, u)]).numpy()
    assert np.allclose(got, expected)


def test_reward_accepts_torch() -> None:
    args = [torch.tensor([0.3]), torch.tensor([0.9]), torch.tensor([1.5]), torch.tensor([-0.4])]
    expected = swingup_reward(*[a.numpy() for a in args])
    assert np.allclose(swingup_reward(*args).numpy(), expected)


def test_rail_penalty_is_zero_inside_limit() -> None:
    assert np.allclose(rail_penalty(np.array([-1.0, 0.0, 1.5, 1.7, -1.9]), 1.6), [0.0, 0.0, 0.0, 0.1, 0.3])


@pytest.mark.parametrize("env_name", ["cartpole-swingup", "cartpole-two_poles"])
def test_mujoco_backend_matches_env_steps(env_name: str) -> None:
    env = DmControlPixelEnv(PixelEnvConfig(env_name=env_name, seed=5, action_repeat=2))
    env.reset_without_render()
    physics = env.physics
    backend = MujocoBackend(physics.model.ptr, action_repeat=2, nthread=2)
    actions = np.random.default_rng(1).uniform(-1.0, 1.0, size=(8, 20, 1))
    reward, cart_x = backend.evaluate(get_state(physics.model.ptr, physics.data.ptr), actions)
    total = 0.0
    for k in range(20):
        state, step_reward = env.step_state(actions[0, k])
        total += step_reward
    assert abs(total - reward[0].sum()) < 1e-9
    assert abs(state["qpos"][0] - cart_x[0, -1]) < 1e-9


class _PointMass:
    """x' = v, v' = u (dt 0.1); reward −(x − 1)² per decision."""

    def evaluate(self, state, actions):
        x, v = np.full(actions.shape[0], state[0]), np.full(actions.shape[0], state[1])
        rewards, xs = [], []
        for k in range(actions.shape[1]):
            v = v + 0.1 * actions[:, k, 0]
            x = x + 0.1 * v
            rewards.append(-(x - 1.0) ** 2)
            xs.append(x)
        return np.stack(rewards, 1), np.stack(xs, 1)


def test_mppi_drives_point_mass_to_target() -> None:
    planner = MPPI(MPPIConfig(horizon=30, samples=256, noise_std=0.5, temperature=0.5, rail_limit=10.0), seed=0)
    state = np.zeros(2)
    for _ in range(80):
        u = planner.plan(_PointMass(), state)[0]
        state = np.array([state[0] + 0.1 * (state[1] + 0.1 * u), state[1] + 0.1 * u])
    assert abs(state[0] - 1.0) < 0.1
