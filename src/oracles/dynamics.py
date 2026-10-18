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
  ``M[0, 0]``; an actuator-gear change is exactly a gain). The offsets are
  expressed as fractions of the nominal mean diagonal of the learned ``M``
  (``slot_scale="diag"``), because the learned ``M`` carries an arbitrary
  global scale (about 360× the true one in v3) that makes raw offsets
  unreachable for the optimizer.
* ``lag_full``: all Lagrangian parameters fine-tuned.
* ``mlp_last``: the MLP dynamics' last linear layer fine-tuned.
* ``mlp_full``: all MLP parameters fine-tuned.

Errors are mean ``|Δ (x, cos theta, sin theta)|`` at a horizon, the same
unit as the intervention-effect oracle, so they can be compared with how
far the perturbation itself moves the trajectories.

Protocol (v2, after the v1 confounds; see ``docs/hypotheses.md`` H1b):

* Windows in which the cart comes within ``rail_margin`` of the ±1.8 m
  slider limit are excluded from training and evaluation. The limit is a
  stiff non-Lagrangian constraint that dominated the v1 errors even for the
  true dynamics (``oracles/ceiling.py``).
* Every mode gets the same adaptation search: a grid of learning rates,
  evaluated after each of ``adapt_checkpoints`` steps (step 0 included, so
  adaptation can always fall back to "no change"), on held-out *selection*
  episodes of the shifted system. Two choices are reported: the best grid
  point per (shift, k) cell, and one recipe per mode chosen by mean
  selection error over all cells (the realistic one; headline).
* Test errors come from a disjoint set of shifted episodes.
* Pretrained models and finished cells are cached under ``cache_dir`` so an
  interrupted run resumes where it stopped.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Sequence

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
from oracles.ceiling import SLIDER_LIMIT
from oracles.common import Criterion, OracleResult, StateEpisodes

ADAPT_MODES = ("lag_low", "lag_full", "mlp_last", "mlp_full")
SLOT_PARAMETERS = ("mass_offset", "log_gain")


class DynamicsWrapper(nn.Module):
    """Hold either dynamics form as an ``nn.Module`` with ``step``."""

    def __init__(
        self,
        form: str,
        action_dim: int = 1,
        hidden: int = 64,
        dt: float = 0.02,
        n_substeps: int = 1,
        actuated_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.form = form
        self.dt = dt
        self.n_substeps = n_substeps
        if form == "lagrangian":
            self.mass = MassMatrix(q_dim=2, hidden_size=hidden, hidden_layers=2, angle_dims=(1,))
            self.potential = Potential(q_dim=2, hidden_size=hidden, hidden_layers=2, angle_dims=(1,))
            self.damping = LinearDissipation(q_dim=2)
            self.actuation = Actuation(
                q_dim=2, action_dim=action_dim, hidden_size=hidden, hidden_layers=2, angle_dims=(1,),
                actuated_dims=actuated_dims,
            )
            # Low-dimensional adaptation slots (identity at init).
            self.mass_offset = nn.Parameter(torch.zeros(2))
            self.log_gain = nn.Parameter(torch.zeros(()))
            self.register_buffer("mass_scale", torch.ones(2), persistent=False)
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
            mass_matrix=_OffsetMass(self.mass, self.mass_offset * self.mass_scale),
            potential=self.potential,
            dissipation=self.damping,
            actuation=_GainActuation(self.actuation, self.log_gain),
        )

    def rollout(self, q: torch.Tensor, qdot: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``q, qdot [N, 2]``, ``actions [N, H, A]`` -> predicted ``q [N, H, 2]``."""

        dynamics = self._dynamics()
        out = []
        for step in range(actions.shape[1]):
            q, qdot = semi_implicit_euler_step(
                dynamics, q, qdot, actions[:, step], dt=self.dt, n_substeps=self.n_substeps,
            )
            out.append(q)
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def calibrate_slots(self, q: torch.Tensor) -> None:
        """Express mass offsets relative to the mean diagonal of ``M`` over ``q [N, 2]``."""

        if self.form == "lagrangian":
            self.mass_scale.copy_(torch.diagonal(self.mass(q), dim1=-2, dim2=-1).mean(dim=0))

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


Windows = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _windows(
    episodes: StateEpisodes,
    horizon: int,
    *,
    stride: int = 1,
    rail_margin: float | None = None,
    rail_reference: np.ndarray | None = None,
) -> Windows:
    """Windows: start ``q, qdot [N, 2]``, ``actions [N, H, A]``, targets ``[N, H, 2]``.

    With ``rail_margin`` set, windows whose cart reaches ``|x| >= 1.8 - margin``
    (start or any target frame) are dropped. ``rail_reference`` (``qpos``-shaped,
    default ``episodes.qpos``) decides that, so noisy and clean views of the
    same episodes keep the same windows.
    """

    qpos, qvel, actions = episodes.qpos, episodes.qvel, episodes.actions
    frames = qpos.shape[1]
    starts = np.arange(0, frames - horizon, stride)
    q0 = qpos[:, starts].reshape(-1, 2)
    v0 = qvel[:, starts].reshape(-1, 2)
    act = np.stack([actions[:, s : s + horizon] for s in starts], axis=1).reshape(-1, horizon, actions.shape[-1])
    tgt = np.stack([qpos[:, s + 1 : s + 1 + horizon] for s in starts], axis=1).reshape(-1, horizon, 2)
    if rail_margin is not None:
        ref = qpos if rail_reference is None else rail_reference
        reach = np.stack([np.abs(ref[:, s : s + horizon + 1, 0]).max(axis=1) for s in starts], axis=1)
        keep = reach.reshape(-1) < SLIDER_LIMIT - rail_margin
        q0, v0, act, tgt = q0[keep], v0[keep], act[keep], tgt[keep]
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    return as_t(q0), as_t(v0), as_t(act), as_t(tgt)


def fit(
    model: DynamicsWrapper,
    data: Windows,
    parameters: Sequence[nn.Parameter],
    *,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    cosine: bool = False,
    checkpoints: Sequence[int] = (),
    on_checkpoint: Callable[[int], None] | None = None,
) -> float:
    """Multi-step rollout regression on pose features; returns the final batch loss.

    ``on_checkpoint(step)`` is called after each step count in ``checkpoints``
    (``0`` = before any update).
    """

    q0, v0, act, tgt = data
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(steps, 1)) if cosine else None
    target_pose = _pose(tgt)
    loss = torch.tensor(float("nan"))
    marks = set(checkpoints)
    if on_checkpoint is not None and 0 in marks:
        on_checkpoint(0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, q0.shape[0], (min(batch_size, q0.shape[0]),), generator=generator)
        pred = model.rollout(q0[idx], v0[idx], act[idx])
        loss = (_pose(pred) - target_pose[idx]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if on_checkpoint is not None and step in marks:
            on_checkpoint(step)
    return float(loss.detach())


@torch.no_grad()
def windows_error(model: DynamicsWrapper, windows: Windows, horizons: Sequence[int]) -> dict[int, float]:
    """Mean |Δ pose| at each horizon over the given windows."""

    q0, v0, act, tgt = windows
    with torch.enable_grad():  # Lagrangian step uses autograd internally
        pred = model.rollout(q0, v0, act).detach()
    dev = (_pose(pred) - _pose(tgt)).abs()
    return {h: float(dev[:, h - 1].mean()) for h in horizons}


def rollout_error(
    model: DynamicsWrapper,
    episodes: StateEpisodes,
    horizons: Sequence[int],
    *,
    stride: int = 10,
    rail_margin: float | None = None,
) -> dict[int, float]:
    """Mean |Δ pose| at each horizon over windows starting every ``stride`` frames."""

    return windows_error(model, _windows(episodes, max(horizons), stride=stride, rail_margin=rail_margin), horizons)


def free_segments(episodes: StateEpisodes, length: int, rail_margin: float | None, seed: int) -> list[tuple[int, int]]:
    """Shuffled non-overlapping ``(episode, start)`` segments that never approach the rail."""

    segments = []
    for episode in range(episodes.qpos.shape[0]):
        for start in range(0, episodes.qpos.shape[1] - length, length):
            x = episodes.qpos[episode, start : start + length + 1, 0]
            if rail_margin is None or np.abs(x).max() < SLIDER_LIMIT - rail_margin:
                segments.append((episode, start))
    np.random.default_rng(seed).shuffle(segments)
    return segments


def _subset(episodes: StateEpisodes, segments: Sequence[tuple[int, int]], length: int) -> StateEpisodes:
    return StateEpisodes(
        qpos=np.stack([episodes.qpos[e, s : s + length + 1] for e, s in segments]),
        qvel=np.stack([episodes.qvel[e, s : s + length + 1] for e, s in segments]),
        actions=np.stack([episodes.actions[e, s : s + length] for e, s in segments]),
        frame_dt=episodes.frame_dt,
    )


def _config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]


class _Cache:
    """Resumable store for pretrained weights and finished adaptation cells."""

    def __init__(self, root: Path | None, config: dict, pretrain_config: dict) -> None:
        self.root = root
        self.pretrain_root = root
        self.cells: dict[str, dict] = {}
        if root is None:
            return
        self.pretrain_root = root / "pretrained" / _config_hash(pretrain_config)
        self.pretrain_root.mkdir(parents=True, exist_ok=True)
        (self.pretrain_root / "config.json").write_text(json.dumps(pretrain_config, indent=2, sort_keys=True, default=str))
        self.root = root / _config_hash(config)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str))
        cells_path = self.root / "cells.json"
        if cells_path.exists():
            self.cells = json.loads(cells_path.read_text())

    def load_model(self, form: str, model: DynamicsWrapper) -> bool:
        if self.pretrain_root is None or not (self.pretrain_root / f"{form}.pt").exists():
            return False
        model.load_state_dict(torch.load(self.pretrain_root / f"{form}.pt", weights_only=True))
        return True

    def save_model(self, form: str, model: DynamicsWrapper) -> None:
        if self.pretrain_root is not None:
            tmp = self.pretrain_root / f"{form}.pt.tmp"
            torch.save(model.state_dict(), tmp)
            tmp.replace(self.pretrain_root / f"{form}.pt")

    def save_cell(self, key: str, value: dict) -> None:
        self.cells[key] = value
        if self.root is not None:
            tmp = self.root / "cells.json.tmp"
            tmp.write_text(json.dumps(self.cells))
            tmp.replace(self.root / "cells.json")


def _adapt_grid(
    base: DynamicsWrapper,
    mode: str,
    data: Windows,
    select: Windows,
    test: Windows,
    *,
    lrs: Sequence[float],
    checkpoints: Sequence[int],
    horizons: Sequence[int],
    select_horizon: int,
    batch_size: int,
    seed: int,
) -> dict:
    """Select/test errors at every (lr, checkpoint) grid point for one mode."""

    grid: dict[str, dict] = {}
    for lr in lrs:
        adapted = copy.deepcopy(base)

        def record(step: int, lr: float = lr, adapted: DynamicsWrapper = adapted) -> None:
            grid[f"{lr:g}@{step}"] = {
                "select": windows_error(adapted, select, (select_horizon,))[select_horizon],
                "test": {str(h): e for h, e in windows_error(adapted, test, horizons).items()},
            }

        fit(
            adapted, data, adapted.adaptation_parameters(mode),
            steps=max(checkpoints), lr=lr, batch_size=batch_size, seed=seed,
            checkpoints=checkpoints, on_checkpoint=record,
        )
    return grid


def run_dynamics_oracle(
    nominal_train: StateEpisodes,
    nominal_val: StateEpisodes,
    shifted: dict[str, tuple[StateEpisodes, StateEpisodes, StateEpisodes]],
    *,
    ks: Sequence[int] = (1, 2, 5, 20),
    segment_length: int = 50,
    train_horizon: int = 8,
    pretrain_steps: int = 10000,
    batch_size: int = 128,
    lr: float = 1e-3,
    adapt_lrs: Sequence[float] = (3e-4, 1e-3, 3e-3, 1e-2),
    adapt_checkpoints: Sequence[int] = (0, 50, 100, 200, 400, 800),
    eval_horizons: Sequence[int] = (10, 20, 50),
    select_horizon: int = 20,
    eval_stride: int = 5,
    n_substeps: int = 2,
    rail_margin: float | None = 0.2,
    actuated_dims: Sequence[int] | None = None,
    slot_scale: str = "diag",
    seed: int = 0,
    modes: Sequence[str] = ADAPT_MODES,
    cache_dir: Path | None = None,
    log=print,
) -> OracleResult:
    """Pre-train both forms on nominal truth, adapt to each shift with k free segments.

    ``shifted[name] = (train, select, test)`` episodes of the shifted system.
    """

    if slot_scale not in ("none", "diag"):
        raise ValueError(f"slot_scale must be 'none' or 'diag', got {slot_scale!r}")
    config = {
        "version": 2,
        "ks": list(ks),
        "segment_length": segment_length,
        "train_horizon": train_horizon,
        "pretrain_steps": pretrain_steps,
        "batch_size": batch_size,
        "lr": lr,
        "adapt_lrs": list(adapt_lrs),
        "adapt_checkpoints": list(adapt_checkpoints),
        "eval_horizons": list(eval_horizons),
        "select_horizon": select_horizon,
        "eval_stride": eval_stride,
        "n_substeps": n_substeps,
        "rail_margin": rail_margin,
        **({"actuated_dims": list(actuated_dims)} if actuated_dims is not None else {}),
        **({"slot_scale": slot_scale} if slot_scale != "none" else {}),
        "seed": seed,
        "modes": list(modes),
        "shifts": list(shifted),
        "frame_dt": nominal_train.frame_dt,
        "nominal_train_episodes": int(nominal_train.qpos.shape[0]),
        "shift_episodes": {n: [int(part.qpos.shape[0]) for part in parts] for n, parts in shifted.items()},
    }
    pretrain_keys = (
        "train_horizon", "pretrain_steps", "batch_size", "lr", "n_substeps", "rail_margin", "actuated_dims",
        "seed", "frame_dt", "nominal_train_episodes",
    )
    cache = _Cache(cache_dir, config, {k: config[k] for k in pretrain_keys if k in config})
    dt = nominal_train.frame_dt
    metrics: dict[str, float] = {}
    nominal_data = _windows(nominal_train, train_horizon, rail_margin=rail_margin)
    nominal_eval = _windows(nominal_val, max(eval_horizons), stride=eval_stride, rail_margin=rail_margin)
    log(f"nominal: {nominal_data[0].shape[0]} train windows, {nominal_eval[0].shape[0]} eval windows")
    models: dict[str, DynamicsWrapper] = {}
    for form in ("lagrangian", "mlp"):
        start = time.time()
        torch.manual_seed(seed)
        model = DynamicsWrapper(
            form, action_dim=nominal_train.actions.shape[-1], dt=dt, n_substeps=n_substeps,
            actuated_dims=actuated_dims,
        )
        if cache.load_model(form, model):
            log(f"pretrained {form}: loaded from cache")
        else:
            trainable = [p for n, p in model.named_parameters() if n not in SLOT_PARAMETERS]
            final = fit(
                model, nominal_data, trainable, steps=pretrain_steps, lr=lr, batch_size=batch_size, seed=seed,
                cosine=True,
            )
            cache.save_model(form, model)
            log(f"pretrained {form}: final loss {final:.2e} ({time.time() - start:.0f}s)")
        if slot_scale == "diag":
            model.calibrate_slots(nominal_data[0])
        models[form] = model
        for h, e in windows_error(model, nominal_eval, eval_horizons).items():
            metrics[f"nominal.{form}.dev_h{h}"] = e
        log(f"  nominal {form} dev: " + "  ".join(f"h{h} {metrics[f'nominal.{form}.dev_h{h}']:.4f}" for h in eval_horizons))

    cells_total = len(shifted) * len(ks) * len(modes)
    for shift_name, (shift_train, shift_select, shift_test) in shifted.items():
        select = _windows(shift_select, select_horizon, stride=eval_stride, rail_margin=rail_margin)
        test = _windows(shift_test, max(eval_horizons), stride=eval_stride, rail_margin=rail_margin)
        for form in ("lagrangian", "mlp"):
            for h, e in windows_error(models[form], test, eval_horizons).items():
                metrics[f"{shift_name}.k0.{form}.dev_h{h}"] = e
        segments = free_segments(shift_train, segment_length, rail_margin, seed)
        if len(segments) < max(ks):
            raise ValueError(f"{shift_name}: only {len(segments)} rail-free segments, need {max(ks)}")
        log(
            f"{shift_name}: {len(segments)} free segments, {select[0].shape[0]} select / {test[0].shape[0]} test "
            f"windows; zero-shot h{select_horizon} lag {metrics[f'{shift_name}.k0.lagrangian.dev_h{select_horizon}']:.4f}"
            f" mlp {metrics[f'{shift_name}.k0.mlp.dev_h{select_horizon}']:.4f}",
        )
        for k in ks:
            data = _windows(_subset(shift_train, segments[:k], segment_length), train_horizon, rail_margin=rail_margin)
            for mode in modes:
                key = f"{shift_name}.k{k}.{mode}"
                if key not in cache.cells:
                    start = time.time()
                    form = "lagrangian" if mode.startswith("lag") else "mlp"
                    grid = _adapt_grid(
                        models[form], mode, data, select, test,
                        lrs=adapt_lrs, checkpoints=adapt_checkpoints, horizons=eval_horizons,
                        select_horizon=select_horizon, batch_size=batch_size, seed=seed + k,
                    )
                    cache.save_cell(key, grid)
                    best = min(grid.values(), key=lambda g: g["select"])
                    log(
                        f"[{len(cache.cells)}/{cells_total}] {key}: best-cell test h{select_horizon} "
                        f"{best['test'][str(select_horizon)]:.4f} ({time.time() - start:.0f}s)",
                    )

    # One recipe per mode: the grid point with the lowest mean selection error over all cells.
    recipes: dict[str, str] = {}
    for mode in modes:
        cells = [cache.cells[f"{s}.k{k}.{mode}"] for s in shifted for k in ks]
        points = list(cells[0])
        recipes[mode] = min(points, key=lambda p: float(np.mean([c[p]["select"] for c in cells])))
    for shift_name in shifted:
        for k in ks:
            for mode in modes:
                grid = cache.cells[f"{shift_name}.k{k}.{mode}"]
                cell_best = min(grid, key=lambda p: grid[p]["select"])
                for h in eval_horizons:
                    metrics[f"{shift_name}.k{k}.{mode}.dev_h{h}"] = grid[recipes[mode]]["test"][str(h)]
                    metrics[f"{shift_name}.k{k}.{mode}.cellbest.dev_h{h}"] = grid[cell_best]["test"][str(h)]

    criteria = []
    effect = min(metrics[f"{s}.k0.lagrangian.dev_h{select_horizon}"] for s in shifted)
    nominal_worst = max(metrics[f"nominal.{form}.dev_h{select_horizon}"] for form in ("lagrangian", "mlp"))
    metrics["prereq_nominal_over_shift_effect"] = nominal_worst / max(effect, 1e-9)
    criteria.append(
        Criterion(
            "prereq_nominal_over_shift_effect", "<=", 0.3,
            "nominal fit error is well below the smallest zero-shot shift error (test can resolve the shift)",
        ),
    )
    for k in [k for k in (1, 5) if k in ks]:
        for suffix in ("", ".cellbest"):
            ratios = []
            for shift_name in shifted:
                lag = min(metrics[f"{shift_name}.k{k}.{m}{suffix}.dev_h{select_horizon}"] for m in modes if m.startswith("lag"))
                mlp = min(metrics[f"{shift_name}.k{k}.{m}{suffix}.dev_h{select_horizon}"] for m in modes if m.startswith("mlp"))
                ratios.append(mlp / max(lag, 1e-9))
            metrics[f"mlp_over_lag_error_ratio{suffix}_k{k}_h{select_horizon}"] = float(np.mean(ratios))
        criteria.append(
            Criterion(
                f"mlp_over_lag_error_ratio_k{k}_h{select_horizon}", ">=", 1.3,
                "best Lagrangian mode adapts with >= 30% lower error than the best MLP mode (one recipe per mode)",
            ),
        )
    return OracleResult(
        name="dynamics_oracle",
        hypotheses=["H1b", "A-LAG-1"],
        question="On true state, does Lagrangian structure make few-shot adaptation to changed physics cheaper?",
        config={**config, "recipes": recipes},
        metrics=metrics,
        criteria=criteria,
        notes=[
            "True state (x, theta, xdot, thetadot); errors are mean |Δ(x, cos, sin)| on rail-free windows.",
            "lag_low adapts 3 parameters (mass-matrix diagonal offsets, actuation gain).",
            "Headline metrics use one adaptation recipe per mode; '.cellbest' picks per (shift, k) cell.",
            "Selection uses held-out shifted episodes disjoint from the test episodes.",
        ],
    )
