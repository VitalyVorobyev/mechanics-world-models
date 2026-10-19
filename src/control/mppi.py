"""Model-predictive path integral (MPPI) planner over a batched dynamics backend.

At every decision the planner perturbs its mean action sequence with
Gaussian noise, scores the ``samples`` sequences with the backend, and
replaces the mean by the softmax-weighted average (weights
``exp(objective / temperature)``). The first action is executed and the
mean is shifted one step for the next decision (warm start). Sample 0 is
always the unperturbed mean, so a good plan is never lost to noise.

Objective = summed task reward over the horizon − ``rail_weight`` × the
summed distance of the cart beyond ``rail_limit``. The rail term exists
because learned models have no rail; it is applied to every backend so
the planners differ only in their dynamics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from control.reward import rail_penalty


class Backend(Protocol):
    """Scores action sequences from a state.

    ``evaluate(state, actions [N, H, A]) -> (reward [N, H], cart_x [N, H])``:
    reward summed over the physics steps of each decision, and the cart
    position at the end of each decision.
    """

    def evaluate(self, state: Any, actions: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@dataclass
class MPPIConfig:
    horizon: int = 50          # decisions (0.02 s each at action_repeat 2)
    samples: int = 512
    iterations: int = 1        # refinement passes per decision
    noise_std: float = 0.5
    temperature: float = 1.0   # in units of summed reward
    rail_limit: float = 1.6
    rail_weight: float = 10.0
    action_dim: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MPPI:
    def __init__(self, config: MPPIConfig, seed: int = 0) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((config.horizon, config.action_dim))
        self.last_objective = float("nan")

    def reset(self) -> None:
        self.mean = np.zeros_like(self.mean)

    def plan(self, backend: Backend, state: Any) -> NDArray[np.float64]:
        cfg = self.config
        for _ in range(cfg.iterations):
            noise = self.rng.normal(0.0, cfg.noise_std, size=(cfg.samples, cfg.horizon, cfg.action_dim))
            noise[0] = 0.0
            actions = np.clip(self.mean[None] + noise, -1.0, 1.0)
            reward, cart_x = backend.evaluate(state, actions)
            objective = reward.sum(axis=1) - cfg.rail_weight * rail_penalty(cart_x, cfg.rail_limit).sum(axis=1)
            weights = np.exp((objective - objective.max()) / cfg.temperature)
            weights /= weights.sum()
            self.mean = np.einsum("n,nha->ha", weights, actions)
            self.last_objective = float(objective.max())
        action = self.mean[0].copy()
        self.mean = np.concatenate([self.mean[1:], self.mean[-1:]], axis=0)
        return action


# Frozen by E-006 (docs/experiments/E-006-planner-ceiling.md); reused unchanged by every later arm.
E006_PLANNER = MPPIConfig(horizon=75, samples=512, noise_std=0.5, temperature=1.0)

# Frozen by E-009 (docs/experiments/E-009-two-poles-ceiling.md) for the two-pole cartpole.
E009_PLANNER = MPPIConfig(horizon=150, samples=512, noise_std=0.5, temperature=1.0)
E009_CEILING = 422.0  # E-009 mean return over evaluation seeds 0-4 (C-019)
