"""Rung 3: dynamics classes on ground-truth state (H1b state-first test).

Fits a learned-Lagrangian model and an unstructured second-order MLP model
to *true* cartpole trajectories (``q = (x, theta)``, ``qdot`` from the
simulator), then measures how much shifted-physics data each needs to adapt
(rollout error vs number of shifted segments ``k``). If structure does not
make adaptation cheaper even with perfect state, the pixel version of H1b
cannot succeed; if it does, the gap sizes the effect a pixel model must
resolve.

Adaptation modes:

* ``lag_low``: Lagrangian with every MLP frozen; only 3 interpretable
  parameters adapt — additive diagonal offsets of the mass matrix and a
  scalar actuation gain (a cart-mass change is exactly an offset of
  ``M[0, 0]``; an actuator-gear change is exactly a gain).
* ``lag_full``: all Lagrangian parameters fine-tuned.
* ``mlp_last``: the MLP dynamics' last linear layer fine-tuned.
* ``mlp_full``: all MLP parameters fine-tuned.

Errors are mean ``|Δ (x, cos theta, sin theta)|`` at a horizon, the same
unit as the intervention-effect oracle, so they can be compared with how
far the perturbation itself moves the trajectories.
"""

from __future__ import annotations

import copy
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn

from models.mechanics.integrator import semi_implicit_euler_step
from models.mechanics.lagrangian import (
    Actuation,
    LagrangianDynamics,
    LinearDissipation,
    MassMatrix,
    MLPDynamics,
    Potential,
)
from oracles.common import Criterion, OracleResult, StateEpisodes

ADAPT_MODES = ("lag_low", "lag_full", "mlp_last", "mlp_full")


class DynamicsWrapper(nn.Module):
    """Hold either dynamics form as an ``nn.Module`` with ``step``."""

    def __init__(self, form: str, action_dim: int = 1, hidden: int = 64, dt: float = 0.02) -> None:
        super().__init__()
        self.form = form
        self.dt = dt
        if form == "lagrangian":
            self.mass = MassMatrix(q_dim=2, hidden_size=hidden, hidden_layers=2, angle_dims=(1,))
            self.potential = Potential(q_dim=2, hidden_size=hidden, hidden_layers=2, angle_dims=(1,))
            self.damping = LinearDissipation(q_dim=2)
            self.actuation = Actuation(
                q_dim=2, action_dim=action_dim, hidden_size=hidden, hidden_layers=2, angle_dims=(1,),
            )
            # Low-dimensional adaptation slots (identity at init).
            self.mass_offset = nn.Parameter(torch.zeros(2))
            self.log_gain = nn.Parameter(torch.zeros(()))
        elif form == "mlp":
            self.mlp = MLPDynamics(
                q_dim=2, action_dim=action_dim, hidden_size=2 * hidden, hidden_layers=2, angle_dims=(1,),
            )
        else:
            raise ValueError(f"unknown dynamics form {form!r}")

    def _dynamics(self) -> LagrangianDynamics | MLPDynamics:
        if self.form == "mlp":
            return self.mlp
        return LagrangianDynamics(
            mass_matrix=_OffsetMass(self.mass, self.mass_offset),
            potential=self.potential,
            dissipation=self.damping,
            actuation=_GainActuation(self.actuation, self.log_gain),
        )

    def rollout(self, q: torch.Tensor, qdot: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``q, qdot [N, 2]``, ``actions [N, H, A]`` -> predicted ``q [N, H, 2]``."""

        dynamics = self._dynamics()
        out = []
        for step in range(actions.shape[1]):
            q, qdot = semi_implicit_euler_step(dynamics, q, qdot, actions[:, step], dt=self.dt)
            out.append(q)
        return torch.stack(out, dim=1)

    def adaptation_parameters(self, mode: str) -> list[nn.Parameter]:
        if mode == "lag_low":
            return [self.mass_offset, self.log_gain]
        if mode == "lag_full":
            return list(self.parameters())
        if mode == "mlp_last":
            return list(self.mlp.net[-1].parameters())
        if mode == "mlp_full":
            return list(self.parameters())
        raise ValueError(mode)


class _OffsetMass(nn.Module):
    """``M(q) + diag(offset)`` — exact form of a change in a body's total mass."""

    def __init__(self, base: MassMatrix, offset: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.offset = offset
        self.q_dim = base.q_dim

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.base(q) + torch.diag_embed(self.offset.expand(*q.shape[:-1], -1))


class _GainActuation(nn.Module):
    """``exp(log_gain) * B(q)`` — exact form of an actuator-gear change."""

    def __init__(self, base: Actuation, log_gain: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.log_gain = log_gain
        self.q_dim = base.q_dim
        self.action_dim = base.action_dim

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.log_gain.exp() * self.base(q)


def _pose(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 0], torch.cos(q[..., 1]), torch.sin(q[..., 1])], dim=-1)


def _windows(episodes: StateEpisodes, horizon: int) -> tuple[torch.Tensor, ...]:
    """All windows: start ``q, qdot [N, 2]``, ``actions [N, H, A]``, targets ``[N, H, 2]``."""

    qpos, qvel, actions = episodes.qpos, episodes.qvel, episodes.actions
    num_episodes, frames = qpos.shape[:2]
    starts = np.arange(0, frames - horizon)
    q0 = qpos[:, starts].reshape(-1, 2)
    v0 = qvel[:, starts].reshape(-1, 2)
    act = np.stack([actions[:, s : s + horizon] for s in starts], axis=1).reshape(-1, horizon, actions.shape[-1])
    tgt = np.stack([qpos[:, s + 1 : s + 1 + horizon] for s in starts], axis=1).reshape(-1, horizon, 2)
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    return as_t(q0), as_t(v0), as_t(act), as_t(tgt)


def fit(
    model: DynamicsWrapper,
    data: tuple[torch.Tensor, ...],
    parameters: Sequence[nn.Parameter],
    *,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> float:
    """Multi-step rollout regression on pose features; returns the final batch loss."""

    q0, v0, act, tgt = data
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    target_pose = _pose(tgt)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        idx = torch.randint(0, q0.shape[0], (min(batch_size, q0.shape[0]),), generator=generator)
        pred = model.rollout(q0[idx], v0[idx], act[idx])
        loss = (_pose(pred) - target_pose[idx]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def rollout_error(model: DynamicsWrapper, episodes: StateEpisodes, horizons: Sequence[int]) -> dict[int, float]:
    """Mean |Δ pose| at each horizon over windows starting every 10 frames."""

    horizon = max(horizons)
    q0, v0, act, tgt = _windows(episodes, horizon)
    keep = torch.arange(0, q0.shape[0], 10)
    with torch.enable_grad():  # Lagrangian step uses autograd internally
        pred = model.rollout(q0[keep], v0[keep], act[keep]).detach()
    dev = (_pose(pred) - _pose(tgt[keep])).abs()
    return {h: float(dev[:, h - 1].mean()) for h in horizons}


def run_dynamics_oracle(
    nominal_train: StateEpisodes,
    nominal_val: StateEpisodes,
    shifted: dict[str, tuple[StateEpisodes, StateEpisodes]],
    *,
    ks: Sequence[int] = (1, 2, 5, 10, 20),
    segment_length: int = 50,
    train_horizon: int = 8,
    pretrain_steps: int = 3000,
    adapt_steps: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    adapt_lr: float = 1e-3,
    eval_horizons: Sequence[int] = (10, 20, 50),
    seed: int = 0,
    modes: Sequence[str] = ADAPT_MODES,
    log=print,
) -> OracleResult:
    """Pre-train both forms on nominal truth, adapt to each shift with k segments."""

    torch.manual_seed(seed)
    dt = nominal_train.frame_dt
    metrics: dict[str, float] = {}
    nominal_data = _windows(nominal_train, train_horizon)
    models: dict[str, DynamicsWrapper] = {}
    for form in ("lagrangian", "mlp"):
        start = time.time()
        torch.manual_seed(seed)
        model = DynamicsWrapper(form, action_dim=nominal_train.actions.shape[-1], dt=dt)
        trainable = [p for n, p in model.named_parameters() if n not in ("mass_offset", "log_gain")]
        final = fit(model, nominal_data, trainable, steps=pretrain_steps, lr=lr, batch_size=batch_size, seed=seed)
        models[form] = model
        errors = rollout_error(model, nominal_val, eval_horizons)
        for h, e in errors.items():
            metrics[f"nominal.{form}.dev_h{h}"] = e
        log(f"pretrained {form}: loss {final:.2e} nominal dev {errors} ({time.time() - start:.0f}s)")

    for shift_name, (shift_train, shift_val) in shifted.items():
        for form in ("lagrangian", "mlp"):
            errors = rollout_error(models[form], shift_val, eval_horizons)
            for h, e in errors.items():
                metrics[f"{shift_name}.k0.{form}.dev_h{h}"] = e
        rng = np.random.default_rng(seed)
        segments = []
        for episode in rng.permutation(shift_train.qpos.shape[0]):
            for start in range(0, shift_train.qpos.shape[1] - segment_length, segment_length):
                segments.append((episode, start))
        rng.shuffle(segments)
        for k in ks:
            chosen = segments[:k]
            subset = StateEpisodes(
                qpos=np.stack([shift_train.qpos[e, s : s + segment_length + 1] for e, s in chosen]),
                qvel=np.stack([shift_train.qvel[e, s : s + segment_length + 1] for e, s in chosen]),
                actions=np.stack([shift_train.actions[e, s : s + segment_length] for e, s in chosen]),
                frame_dt=dt,
            )
            data = _windows(subset, train_horizon)
            for mode in modes:
                form = "lagrangian" if mode.startswith("lag") else "mlp"
                adapted = copy.deepcopy(models[form])
                fit(
                    adapted, data, adapted.adaptation_parameters(mode),
                    steps=adapt_steps, lr=adapt_lr, batch_size=batch_size, seed=seed + k,
                )
                errors = rollout_error(adapted, shift_val, eval_horizons)
                for h, e in errors.items():
                    metrics[f"{shift_name}.k{k}.{mode}.dev_h{h}"] = e
            log(
                f"{shift_name} k={k}: "
                + "  ".join(f"{m} {metrics[f'{shift_name}.k{k}.{m}.dev_h20']:.4f}" for m in modes),
            )

    # Headline: at small k, how much better is the best Lagrangian mode than the best MLP mode?
    criteria = []
    for k in [k for k in (1, 5) if k in ks]:
        ratios = []
        for shift_name in shifted:
            best_lag = min(metrics[f"{shift_name}.k{k}.{m}.dev_h20"] for m in modes if m.startswith("lag"))
            best_mlp = min(metrics[f"{shift_name}.k{k}.{m}.dev_h20"] for m in modes if m.startswith("mlp"))
            ratios.append(best_mlp / max(best_lag, 1e-9))
        metrics[f"mlp_over_lag_error_ratio_k{k}_h20"] = float(np.mean(ratios))
        criteria.append(
            Criterion(
                f"mlp_over_lag_error_ratio_k{k}_h20", ">=", 1.3,
                "Lagrangian adapts with >= 30% lower h20 error than the best MLP mode",
            ),
        )
    return OracleResult(
        name="dynamics_oracle",
        hypotheses=["H1b", "A-LAG-1"],
        question="On true state, does Lagrangian structure make few-shot adaptation to changed physics cheaper?",
        config={
            "ks": list(ks),
            "segment_length": segment_length,
            "train_horizon": train_horizon,
            "pretrain_steps": pretrain_steps,
            "adapt_steps": adapt_steps,
            "batch_size": batch_size,
            "lr": lr,
            "adapt_lr": adapt_lr,
            "eval_horizons": list(eval_horizons),
            "seed": seed,
            "modes": list(modes),
            "shifts": list(shifted),
            "frame_dt": dt,
            "nominal_train_episodes": int(nominal_train.qpos.shape[0]),
        },
        metrics=metrics,
        criteria=criteria,
        notes=[
            "True state (x, theta, xdot, thetadot); errors are mean |Δ(x, cos, sin)|.",
            "lag_low adapts 3 parameters (mass-matrix diagonal offsets, actuation gain).",
            "Single seed unless rerun with --seeds; treat as preliminary.",
        ],
    )
