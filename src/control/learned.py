"""Planning with learned dynamics on true state (E-007).

The learned models are the oracle harness's ``DynamicsWrapper`` forms
(learned Lagrangian with ``actuated_dims=(0,)``, or the second-order MLP),
trained by multi-step rollout regression (``oracles.dynamics.fit``) on state
episodes. ``LearnedBackend`` rolls a model out for the MPPI planner;
``run_episode`` is the closed loop shared by every backend; ``collect`` adds
episodes driven by a policy, which is how on-policy data rounds are built.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray

from control.mppi import MPPI, Backend
from control.reward import poles_reward, upright_all
from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
from models.mechanics.lagrangian import forward_acceleration
from oracles.common import StateEpisodes
from oracles.dynamics import DynamicsWrapper, _windows, fit

# Backends take this as the state: the true one wants the full MuJoCo state,
# learned ones want (q, q̇). ``run_episode`` builds it with
# ``state_fn(env, last_action)`` (the camera pipeline needs the action that led
# to the current frame); an optional ``state_fn.reset()`` runs at episode start.
StateFn = Callable[[DmControlPixelEnv, Any], Any]


class LearnedBackend:
    """Rolls ``actions [N, H, A]`` through a learned model from ``(q, q̇)``.

    The model's own integrator substeps (``n_substeps`` of ``dt / n_substeps``)
    stand in for the physics steps, so with ``dt = 0.02`` and 2 substeps the
    reward is summed over the same 0.01 s steps the environment scores.
    """

    def __init__(self, model: DynamicsWrapper) -> None:
        self.model = model.eval()
        self.dynamics = model._dynamics()

    @torch.no_grad()
    def evaluate(
        self, state: tuple[NDArray[np.float64], NDArray[np.float64]], actions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n, horizon, _ = actions.shape
        q = torch.as_tensor(state[0], dtype=torch.float32).expand(n, -1).clone()
        qdot = torch.as_tensor(state[1], dtype=torch.float32).expand(n, -1).clone()
        u_all = torch.as_tensor(actions, dtype=torch.float32)
        h = self.model.dt / self.model.n_substeps
        reward = torch.zeros(n, horizon)
        cart_x = torch.zeros(n, horizon)
        for k in range(horizon):
            u = u_all[:, k]
            for _ in range(self.model.n_substeps):  # same update as semi_implicit_euler_step
                qdot = qdot + h * forward_acceleration(self.dynamics, q, qdot, u)
                q = q + h * qdot
                reward[:, k] += poles_reward(q[:, 0], q[:, 1:], qdot[:, 1:], u[:, 0])
            cart_x[:, k] = q[:, 0]
        return reward.double().numpy(), cart_x.double().numpy()


def true_state(env: DmControlPixelEnv, last_action: Any = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    state = env.physics_state()
    return state["qpos"], state["qvel"]


@dataclass
class Episode:
    qpos: NDArray[np.float64]      # [N+1, 1 + P]
    qvel: NDArray[np.float64]      # [N+1, 1 + P]
    actions: NDArray[np.float64]   # [N, 1]
    rewards: NDArray[np.float64]   # [N]
    plan_times: NDArray[np.float64]

    @property
    def total(self) -> float:
        return float(self.rewards.sum())

    def metrics(self) -> dict[str, float]:
        upright = upright_all(self.qpos[1:, 1:])
        return {
            "return": self.total,
            "decision_time_mean_s": float(self.plan_times.mean()),
            "upright_fraction_second_half": float(np.mean(upright[len(upright) // 2:])),
            "max_abs_x": float(np.abs(self.qpos[:, 0]).max()),
            "rail_fraction": float(np.mean(np.abs(self.qpos[:, 0]) > 1.75)),
        }


def run_episode(
    env_seed: int,
    planner: MPPI,
    backend: Backend,
    state_fn: StateFn,
    *,
    action_repeat: int = 2,
    action_noise: float = 0.0,
    noise_seed: int = 0,
    env_name: str = "cartpole-swingup",
) -> Episode:
    """One closed-loop episode; ``action_noise`` (Gaussian std) is added to executed actions only."""

    env = DmControlPixelEnv(PixelEnvConfig(env_name=env_name, seed=env_seed, action_repeat=action_repeat))
    rng = np.random.default_rng(noise_seed)
    try:
        env.reset_without_render()
        planner.reset()
        if hasattr(state_fn, "reset"):
            state_fn.reset()
        last_action = None
        qpos, qvel = [env.physics.data.qpos.copy()], [env.physics.data.qvel.copy()]
        actions, rewards, plan_times = [], [], []
        while not env.episode_done():
            start = time.perf_counter()
            action = planner.plan(backend, state_fn(env, last_action))
            plan_times.append(time.perf_counter() - start)
            if action_noise > 0.0:
                action = np.clip(action + rng.normal(0.0, action_noise, size=action.shape), -1.0, 1.0)
            state, reward = env.step_state(action)
            last_action = action
            qpos.append(state["qpos"]), qvel.append(state["qvel"])
            actions.append(np.asarray(action, dtype=np.float64)), rewards.append(reward)
    finally:
        env.close()
    return Episode(np.asarray(qpos), np.asarray(qvel), np.asarray(actions), np.asarray(rewards), np.asarray(plan_times))


def episodes_to_states(episodes: list[Episode], frame_dt: float) -> StateEpisodes:
    return StateEpisodes(
        qpos=np.stack([e.qpos for e in episodes]),
        qvel=np.stack([e.qvel for e in episodes]),
        actions=np.stack([e.actions for e in episodes]),
        frame_dt=frame_dt,
    )


def concat_states(parts: list[StateEpisodes]) -> StateEpisodes:
    return StateEpisodes(
        qpos=np.concatenate([p.qpos for p in parts]),
        qvel=np.concatenate([p.qvel for p in parts]),
        actions=np.concatenate([p.actions for p in parts]),
        frame_dt=parts[0].frame_dt,
    )


@dataclass
class TrainConfig:
    steps: int = 15000
    train_horizon: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    hidden: int = 64
    n_substeps: int = 2
    rail_margin: float = 0.2


def make_model(form: str, config: TrainConfig, dt: float, q_dim: int = 2) -> DynamicsWrapper:
    """An untrained model of ``form`` for a cart with ``q_dim - 1`` poles (every pole angle is S¹)."""

    return DynamicsWrapper(
        form, hidden=config.hidden, dt=dt, n_substeps=config.n_substeps,
        actuated_dims=(0,) if form == "lagrangian" else None,
        q_dim=q_dim, angle_dims=tuple(range(1, q_dim)),
    )


def train_model(form: str, data: StateEpisodes, config: TrainConfig, seed: int) -> tuple[DynamicsWrapper, float]:
    """A fresh model of ``form`` fitted to ``data``; returns it and the final batch loss."""

    torch.manual_seed(seed)
    model = make_model(form, config, data.frame_dt, q_dim=data.qpos.shape[-1])
    windows = _windows(data, config.train_horizon, rail_margin=config.rail_margin)
    parameters = [p for name, p in model.named_parameters() if name not in ("mass_offset", "log_gain")]
    loss = fit(
        model, windows, parameters, steps=config.steps, lr=config.lr, batch_size=config.batch_size,
        seed=seed, cosine=True,
    )
    return model, loss


def load_round_model(round_dir) -> DynamicsWrapper:
    """The model a ``learned_control.py`` round trained (``<cache>/<key>/round<r>/model.pt``)."""

    import json
    from pathlib import Path

    round_dir = Path(round_dir)
    config = json.loads((round_dir.parent / "config.json").read_text())
    train = TrainConfig(**config["train"])
    model = make_model(config["form"], train, 0.02, q_dim=config.get("q_dim", 2))
    model.load_state_dict(torch.load(round_dir / "model.pt"))
    return model.eval()
