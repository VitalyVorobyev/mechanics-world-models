"""Causal state estimation from camera poses through a dynamics model (E-008).

A single frame gives the pose ``(x, θ)`` but no velocity. ``FixedLagEstimator``
works with any ``Dynamics`` (a learned model, or the true MuJoCo model) and keeps the last ``window + 1`` observed poses and executed actions, fits the
state at the start of the window by Levenberg–Marquardt shooting through the
model (``oracles.start_state.shoot``), and integrates that state forward to
the current frame. Observations that disagree with the model's one-step
prediction by more than the gate are replaced by the prediction (innovation
gating), so a single misread frame cannot throw the estimate. The gate is
active only once the window is full, and for at most ``max_rejects`` frames
in a row: a gate that keeps rejecting means the estimate, not the camera, is
wrong, and locking out the camera would make that permanent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray

from models.mechanics.integrator import semi_implicit_euler_step
from oracles.dynamics import DynamicsWrapper
from oracles.start_state import ContextWindows, shoot


def integrate(
    model: DynamicsWrapper, q: NDArray[np.float64], qdot: NDArray[np.float64], actions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Roll ``q, qdot [N, 2]`` through ``actions [N, H, A]``; return the final ``(q, qdot)``."""

    dynamics = model._dynamics()
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    q_t, v_t, u = as_t(q), as_t(qdot), as_t(actions)
    with torch.no_grad():
        for k in range(u.shape[1]):
            q_t, v_t = semi_implicit_euler_step(dynamics, q_t, v_t, u[:, k], dt=model.dt, n_substeps=model.n_substeps)
    return q_t.double().numpy(), v_t.double().numpy()


@dataclass
class Dynamics:
    """What the estimator needs from a model: a pose rollout and a state integrator."""

    rollout: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]   # q0, v0 [N,2], actions [N,H,A] -> q [N,H,2]
    integrate: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]  # -> final (q, q̇)
    dt: float


def learned_dynamics(model: DynamicsWrapper) -> Dynamics:
    return Dynamics(model_rollout(model), lambda q, v, a: integrate(model, q, v, a), model.dt)


def mujoco_dynamics(mj_model, action_repeat: int = 2, nthread: int = 1) -> Dynamics:
    """The true model behind the same interface (rail and contacts included)."""

    import mujoco
    import mujoco.rollout

    datas = [mujoco.MjData(mj_model) for _ in range(nthread)]
    roller = mujoco.rollout.Rollout(nthread=nthread)
    nq = mj_model.nq

    def states(q0: np.ndarray, v0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        initial = np.concatenate([np.zeros((q0.shape[0], 1)), q0, v0], axis=1)
        control = np.repeat(actions, action_repeat, axis=1)
        out, _ = roller.rollout(mj_model, datas, initial, control)
        return out[:, action_repeat - 1 :: action_repeat]           # [N, H, nstate] at frame ends

    def rollout(q0, v0, actions):
        return states(q0, v0, actions)[..., 1 : 1 + nq]

    def integrate_true(q0, v0, actions):
        last = states(q0, v0, actions)[:, -1]
        return last[:, 1 : 1 + nq], last[:, 1 + nq : 1 + 2 * nq]

    return Dynamics(rollout, integrate_true, mj_model.opt.timestep * action_repeat)


def model_rollout(model: DynamicsWrapper):
    """``shoot``'s rollout interface: ``q0, qdot0 [N, 2]``, ``actions [N, H, A]`` -> ``q [N, H, 2]``."""

    def rollout(q0: np.ndarray, v0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        with torch.no_grad():
            return model.rollout(as_t(q0), as_t(v0), as_t(actions)).double().numpy()

    return rollout


class FixedLagEstimator:
    def __init__(
        self,
        dynamics: Dynamics,
        *,
        window: int = 12,
        sigma: tuple[float, float] = (0.02, 0.03),
        gate: tuple[float, float] = (0.15, 0.25),
        iterations: int = 8,
        max_rejects: int = 2,
    ) -> None:
        self.dynamics = dynamics
        self.window = window
        self.sigma = sigma
        self.gate = np.asarray(gate)
        self.iterations = iterations
        self.max_rejects = max_rejects
        self.reset()

    def reset(self) -> None:
        self.poses: deque[NDArray[np.float64]] = deque(maxlen=self.window + 1)
        self.actions: deque[NDArray[np.float64]] = deque(maxlen=self.window)
        self.state: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self.rejected = 0
        self._in_a_row = 0

    def update(self, pose: NDArray[np.float64], last_action: NDArray[np.float64] | None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Add the newest pose (θ in any branch) and the action that led to it; return ``(q, q̇)`` now."""

        pose = np.asarray(pose, dtype=np.float64).copy()
        if last_action is not None:
            self.actions.append(np.asarray(last_action, dtype=np.float64))
        if self.poses:
            # Keep θ continuous with the previous frame (the reader returns (−π, π]).
            previous = self.poses[-1]
            pose[1] = previous[1] + np.angle(np.exp(1j * (pose[1] - previous[1])))
        gate_on = len(self.poses) == self.window + 1 and self._in_a_row < self.max_rejects
        if gate_on and self.state is not None and last_action is not None:
            q_pred, _ = self.dynamics.integrate(self.state[0][None], self.state[1][None], np.asarray(last_action)[None, None])
            q_pred = q_pred[0]
            pose[1] = q_pred[1] + np.angle(np.exp(1j * (pose[1] - q_pred[1])))
            if np.any(np.abs(pose - q_pred) > self.gate):
                pose = q_pred
                self.rejected += 1
                self._in_a_row += 1
            else:
                self._in_a_row = 0
        else:
            self._in_a_row = 0
        self.poses.append(pose)

        observed = np.stack(self.poses)
        if len(self.poses) < 3:
            qdot = np.zeros(2) if len(self.poses) == 1 else (observed[-1] - observed[-2]) / self.dynamics.dt
            self.state = (observed[-1], qdot)
            return self.state
        context = len(self.poses) - 1
        actions = np.stack(list(self.actions)[-context:])
        windows = ContextWindows(
            observed=observed[None], actions=actions[None], start_q=observed[:1], start_qdot=np.zeros((1, 2)),
            target=np.zeros((1, 0, 2)), context=context, frame_dt=self.dynamics.dt,
        )
        start = shoot(self.dynamics.rollout, windows, sigma=self.sigma, iterations=self.iterations)
        q, qdot = self.dynamics.integrate(start[:, :2], start[:, 2:], actions[None])
        self.state = (q[0], qdot[0])
        return self.state
