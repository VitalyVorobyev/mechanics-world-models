"""Rung 0: does a physics perturbation change the trajectories, and by how much?

Simulates the nominal and each perturbed environment from identical initial
states with identical random action sequences (no rendering) and reports
the mean absolute deviation of ``(x, cos theta, sin theta)`` at several
horizons. A perturbation whose deviation is below what the models can
resolve (compare with the open-loop error of the model at that horizon) is
not a usable experimental manipulation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
from oracles.common import Criterion, OracleResult, pose_features


def _rollout(scales: dict[str, float] | None, seed: int, steps: int, action_repeat: int) -> np.ndarray:
    env = DmControlPixelEnv(
        PixelEnvConfig(action_repeat=action_repeat, seed=seed, physics_param_scales=scales),
    )
    try:
        env.reset()  # task RNG seeded by `seed`: same initial state for every perturbation
        rng = np.random.default_rng(seed)
        states = []
        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0, size=(1,))
            states.append(env.step_without_render(action)["qpos"])
        return pose_features(np.stack(states))
    finally:
        env.close()


def run_intervention_effect(
    shifts: dict[str, dict[str, float]],
    *,
    seeds: Sequence[int] = range(8),
    horizons: Sequence[int] = (10, 20, 50),
    action_repeat: int = 2,
    min_effect_h20: float = 0.01,
) -> OracleResult:
    """Return per-shift deviations; criterion: deviation at h20 >= ``min_effect_h20``."""

    steps = max(horizons)
    nominal = {seed: _rollout(None, seed, steps, action_repeat) for seed in seeds}
    metrics: dict[str, float] = {}
    criteria: list[Criterion] = []
    for name, scales in shifts.items():
        deviations = np.stack(
            [np.abs(_rollout(scales, seed, steps, action_repeat) - nominal[seed]) for seed in seeds],
        )  # [S, steps, 3]
        for horizon in horizons:
            metrics[f"{name}.mean_abs_dev_h{horizon}"] = float(deviations[:, horizon - 1].mean())
        if 20 in horizons:
            criteria.append(
                Criterion(
                    f"{name}.mean_abs_dev_h20", ">=", min_effect_h20,
                    "perturbation moves (x, cos, sin) enough to be resolvable",
                ),
            )
    return OracleResult(
        name="intervention_effect",
        hypotheses=["A-OOD-2", "A-OOD-3", "A-OOD-4", "H3"],
        question="Do these physics perturbations measurably change cartpole trajectories?",
        config={
            "shifts": shifts,
            "seeds": list(seeds),
            "horizons": list(horizons),
            "action_repeat": action_repeat,
            "min_effect_h20": min_effect_h20,
        },
        metrics=metrics,
        criteria=criteria,
        notes=[
            "Identical initial states and random actions per seed; no rendering.",
            "Compare deviations with the model's open-loop error in the same units.",
        ],
    )
