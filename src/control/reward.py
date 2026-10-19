"""The dm_control cartpole swing-up reward in closed form.

Mirrors ``dm_control.suite.cartpole.Balance._get_reward`` (smooth variant)
for one pole, evaluated on the state *after* a physics step and the control
applied during it:

    upright        = (cos θ + 1) / 2
    centered       = (1 + tol(x, margin=2)) / 2
    small_control  = (4 + tol(u, margin=1, value_at_margin=0, quadratic)) / 5
    small_velocity = (1 + tol(θ̇, margin=5)) / 2
    reward         = upright · centered · small_control · small_velocity

``tol`` with bounds (0, 0) and the default Gaussian sigmoid is
``exp(-0.5 · (d · s)²)`` with ``s = sqrt(-2 ln 0.1)``; the quadratic one with
value 0 at the margin is ``max(0, 1 - d²)``. Works on NumPy arrays and torch
tensors of any matching shape.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_GAUSS_SCALE = math.sqrt(-2.0 * math.log(0.1))


def _exp(value: Any) -> Any:
    return value.exp() if hasattr(value, "exp") and not isinstance(value, np.ndarray) else np.exp(value)


def _gaussian_tolerance(value: Any, margin: float) -> Any:
    scaled = value / margin * _GAUSS_SCALE
    return _exp(-0.5 * scaled * scaled)


def swingup_reward(x: Any, cos_theta: Any, theta_dot: Any, u: Any) -> Any:
    """Per-physics-step reward; ``u`` must already lie in [-1, 1] (the action bounds)."""

    upright = (cos_theta + 1.0) / 2.0
    centered = (1.0 + _gaussian_tolerance(x, 2.0)) / 2.0
    small_control = (4.0 + (1.0 - u * u)) / 5.0
    small_velocity = (1.0 + _gaussian_tolerance(theta_dot, 5.0)) / 2.0
    return upright * centered * small_control * small_velocity


def rail_penalty(x: Any, limit: float) -> Any:
    """Planner-only cost for the cart beyond ``limit`` (learned models have no rail)."""

    over = abs(x) - limit
    return over * (over > 0)
