"""Start-state estimation from noisy poses: how much of the open-loop error is the start?

A pixel world model never sees the true velocity. It sees noisy poses and
must estimate ``(q, q̇)`` at the last context frame before it can predict.
The noisy context oracle showed that a two-frame backward difference makes
the start-velocity error dominate the h20 error even for the *true*
dynamics. This module compares causal estimators that use the last
``context`` frames (all only from observations up to the start frame):

* ``backward``: ``q_t`` and ``(q_t - q_{t-1}) / dt`` (the mechanics model's
  open-loop start).
* ``quadfit``: least-squares quadratic through the context poses, evaluated
  with its slope at frame ``t``.
* ``shoot``: model-based. Fit the state at frame ``t - context`` so that the
  model's own rollout under the logged actions matches the context poses
  (Gauss–Newton with finite-difference Jacobians, so any rollout function
  works), then keep rolling out past ``t``. This is what a filtering
  posterior in a world model approximates, and a better dynamics model
  gives a better estimate.

Windows here start at frame ``s >= context`` so every estimator sees the
same windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from oracles.ceiling import SLIDER_LIMIT
from oracles.common import StateEpisodes, pose_features

ESTIMATORS = ("backward", "quadfit", "shoot")

# q0 [N, 2], qdot0 [N, 2], actions [N, H, A] (float64 numpy) -> q [N, H, 2]
Rollout = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class ContextWindows:
    """``observed [N, C+1, 2]`` poses at frames ``s-C..s``; ``actions [N, C+H, A]`` from frame ``s-C``;
    ``start_q, start_qdot [N, 2]`` true state at ``s``; ``target [N, H, 2]`` true poses ``s+1..s+H``."""

    observed: np.ndarray
    actions: np.ndarray
    start_q: np.ndarray
    start_qdot: np.ndarray
    target: np.ndarray
    context: int
    frame_dt: float


def context_windows(
    clean: StateEpisodes,
    observed: StateEpisodes,
    *,
    horizon: int,
    context: int,
    stride: int,
    rail_margin: float | None,
) -> ContextWindows:
    """Windows with ``context`` observed frames before the start; the rail mask uses true poses."""

    frames = clean.qpos.shape[1]
    starts = np.arange(context, frames - horizon, stride)
    obs = np.stack([observed.qpos[:, s - context : s + 1] for s in starts], axis=1)
    act = np.stack([clean.actions[:, s - context : s + horizon] for s in starts], axis=1)
    tgt = np.stack([clean.qpos[:, s + 1 : s + 1 + horizon] for s in starts], axis=1)
    q0, v0 = clean.qpos[:, starts], clean.qvel[:, starts]
    flat = lambda a: a.reshape(-1, *a.shape[2:])  # noqa: E731
    obs, act, tgt, q0, v0 = map(flat, (obs, act, tgt, q0, v0))
    if rail_margin is not None:
        reach = np.stack([np.abs(clean.qpos[:, s - context : s + horizon + 1, 0]).max(axis=1) for s in starts], 1)
        keep = reach.reshape(-1) < SLIDER_LIMIT - rail_margin
        obs, act, tgt, q0, v0 = (a[keep] for a in (obs, act, tgt, q0, v0))
    return ContextWindows(obs, act, q0, v0, tgt, context, clean.frame_dt)


def backward_start(w: ContextWindows) -> tuple[np.ndarray, np.ndarray]:
    return w.observed[:, -1], (w.observed[:, -1] - w.observed[:, -2]) / w.frame_dt


def quadfit_start(w: ContextWindows) -> tuple[np.ndarray, np.ndarray]:
    tau = np.arange(-w.context, 1) * w.frame_dt  # frame t at tau = 0
    basis = np.stack([np.ones_like(tau), tau, tau**2], axis=1)  # [C+1, 3]
    coef = np.linalg.lstsq(basis, w.observed.transpose(1, 0, 2).reshape(len(tau), -1), rcond=None)[0]
    coef = coef.reshape(3, *w.observed.shape[::2])  # [3, N, 2]
    return coef[0], coef[1]


def shoot(
    rollout: Rollout,
    w: ContextWindows,
    *,
    sigma: tuple[float, float] = (1.0, 1.0),
    iterations: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """Levenberg–Marquardt fit of the state at frame ``s - C``; returns that state ``[N, 4]`` (q, q̇).

    Per window, a step is kept only if it lowers the weighted residual
    (otherwise the damping grows), so a poor model cannot make the estimate
    diverge.
    """

    obs, c, dt = w.observed, w.context, w.frame_dt
    weight = 1.0 / np.asarray(sigma)
    state = np.concatenate([obs[:, 0], (obs[:, 2] - obs[:, 0]) / (2 * dt)], axis=1)  # init [N, 4]
    actions = w.actions[:, :c]
    n = state.shape[0]

    def residual(s: np.ndarray, reps: int) -> np.ndarray:
        pred = rollout(s[:, :2], s[:, 2:], np.tile(actions, (reps, 1, 1)))  # [reps*N, C, 2]
        full = np.concatenate([s[:, None, :2], pred], axis=1)  # frames s-C..s
        return ((full - np.tile(obs, (reps, 1, 1))) * weight).reshape(s.shape[0], -1)

    def cost(r: np.ndarray) -> np.ndarray:
        value = (r**2).sum(axis=1)
        return np.where(np.isfinite(value), value, np.inf)

    damping = np.full(n, 1e-3)
    for _ in range(iterations):
        batch = np.concatenate([state] + [state + eps * np.eye(4)[j] for j in range(4)])
        r_all = residual(batch, 5).reshape(5, n, -1)
        r0 = r_all[0]
        jac = np.stack([(r_all[j + 1] - r0) / eps for j in range(4)], axis=-1)  # [N, R, 4]
        jac = np.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
        jtj = np.einsum("nri,nrj->nij", jac, jac)
        diag = np.einsum("nii->ni", jtj)
        lhs = jtj + (damping[:, None] * (diag + 1e-9))[:, :, None] * np.eye(4)
        jtr = np.einsum("nri,nr->ni", jac, np.nan_to_num(r0))
        candidate = state - np.linalg.solve(lhs, jtr[..., None])[..., 0]
        better = cost(residual(candidate, 1)) < cost(r0)
        state = np.where(better[:, None], candidate, state)
        damping = np.where(better, damping / 3, damping * 4)
    return state


def evaluate_starts(
    rollout: Rollout,
    w: ContextWindows,
    horizons: tuple[int, ...],
    *,
    sigma: tuple[float, float],
    estimators: tuple[str, ...] = ESTIMATORS,
) -> dict[str, float]:
    """Mean |Δ pose| at each horizon after the start frame, for the true start and each estimator."""

    h_max = max(horizons)
    c = w.context
    preds: dict[str, np.ndarray] = {"true_start": rollout(w.start_q, w.start_qdot, w.actions[:, c : c + h_max])}
    for name in estimators:
        if name == "shoot":
            state = shoot(rollout, w, sigma=sigma)
            preds[name] = rollout(state[:, :2], state[:, 2:], w.actions[:, : c + h_max])[:, c:]
        else:
            q0, v0 = backward_start(w) if name == "backward" else quadfit_start(w)
            preds[name] = rollout(q0, v0, w.actions[:, c : c + h_max])
    out = {}
    target = pose_features(w.target[:, :h_max])
    for name, pred in preds.items():
        dev = np.abs(pose_features(pred) - target)
        for h in horizons:
            out[f"{name}.dev_h{h}"] = float(dev[:, h - 1].mean())
    return out
