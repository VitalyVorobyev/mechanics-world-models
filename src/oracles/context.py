"""Rung 3, matched baseline: multi-environment training with a 3-number context.

The single-environment dynamics oracle gives the Lagrangian structure-given
adaptation slots (mass-matrix offsets, actuation gain) that an MLP lacks.
The fair unstructured competitor (cRSSM-style, Prasanna et al. 2024) is an
MLP that sees several physical settings during training, each with its own
learned context vector, and adapts only that context at test time.

Both forms here are trained on the same set of cart masses, each with a
per-environment 3-number context (environment 0 is nominal and pinned to
zero, which fixes the gauge):

* ``lagrangian``: the context *is* the slot triple — diagonal mass offsets
  (as fractions of the mean learned mass diagonal) and a log actuation gain.
* ``mlp``: the context is an extra input of the second-order MLP.

Adaptation modes: ``lag_ctx`` / ``mlp_ctx`` fit only a new context vector
(3 numbers each); ``lag_full`` / ``mlp_full`` fit everything. The shifted
test systems (cart mass ×0.5, ×2) lie outside the training range, so this
asks whether structure helps the context *extrapolate*.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
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
from oracles.dynamics import (
    _Cache,
    _pose,
    _subset,
    _windows,
    free_segments,
    windows_error,
)

CONTEXT_DIM = 3
CONTEXT_MODES = ("lag_ctx", "lag_full", "mlp_ctx", "mlp_full")


def simulate_state_episodes(
    scales: dict[str, float] | None, *, episodes: int, seed: int, action_repeat: int = 2,
) -> StateEpisodes:
    """Random-policy state trajectories (no rendering), same policy as the collector."""

    env = DmControlPixelEnv(PixelEnvConfig(action_repeat=action_repeat, seed=seed, physics_param_scales=scales))
    rng = np.random.default_rng(seed)
    qpos, qvel, actions = [], [], []
    try:
        for _ in range(episodes):
            env.reset_without_render()
            states, acts = [env.physics_state()], []
            while True:
                action = env.sample_action(rng)
                state = env.step_without_render(action)
                acts.append(np.asarray(action, dtype=np.float64))
                states.append(state)
                if env.episode_done():
                    break
            qpos.append(np.stack([s["qpos"] for s in states]))
            qvel.append(np.stack([s["qvel"] for s in states]))
            actions.append(np.stack(acts))
        frame_dt = env.frame_dt()
    finally:
        env.close()
    return StateEpisodes(qpos=np.stack(qpos), qvel=np.stack(qvel), actions=np.stack(actions), frame_dt=frame_dt)


class _SlotMass(nn.Module):
    def __init__(self, base: MassMatrix, offset: torch.Tensor) -> None:
        super().__init__()
        self.base, self.offset, self.q_dim = base, offset, base.q_dim

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.base(q) + torch.diag_embed(self.offset)  # offset [N, d]


class _SlotActuation(nn.Module):
    def __init__(self, base: Actuation, log_gain: torch.Tensor) -> None:
        super().__init__()
        self.base, self.log_gain = base, log_gain
        self.q_dim, self.action_dim = base.q_dim, base.action_dim

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.log_gain.exp().view(-1, 1, 1) * self.base(q)  # log_gain [N]


class ContextDynamics(nn.Module):
    """Either form with a per-environment context table and one test-time context."""

    def __init__(
        self,
        form: str,
        num_envs: int,
        *,
        action_dim: int = 1,
        hidden: int = 64,
        dt: float = 0.02,
        n_substeps: int = 2,
        actuated_dims: Sequence[int] | None = (0,),
    ) -> None:
        super().__init__()
        self.form, self.dt, self.n_substeps = form, dt, n_substeps
        if form == "lagrangian":
            self.mass = MassMatrix(q_dim=2, hidden_size=hidden, hidden_layers=2, angle_dims=(1,))
            self.potential = Potential(q_dim=2, hidden_size=hidden, hidden_layers=2, angle_dims=(1,))
            self.damping = LinearDissipation(q_dim=2)
            self.actuation = Actuation(
                q_dim=2, action_dim=action_dim, hidden_size=hidden, hidden_layers=2, angle_dims=(1,),
                actuated_dims=actuated_dims,
            )
        elif form == "mlp":
            self.mlp = MLPDynamics(
                q_dim=2, action_dim=action_dim + CONTEXT_DIM, hidden_size=2 * hidden, hidden_layers=2,
                angle_dims=(1,),
            )
        else:
            raise ValueError(f"unknown dynamics form {form!r}")
        # Row 0 (nominal) is pinned to zero in `contexts_for`.
        self.contexts = nn.Parameter(torch.zeros(num_envs, CONTEXT_DIM))
        self.adapted = nn.Parameter(torch.zeros(CONTEXT_DIM))
        self.register_buffer("mass_scale", torch.ones(2), persistent=False)
        self.calibrated = False

    def contexts_for(self, env: torch.Tensor | None, n: int) -> torch.Tensor:
        if env is None:
            return self.adapted.expand(n, CONTEXT_DIM)
        pinned = torch.cat([torch.zeros_like(self.contexts[:1]), self.contexts[1:]], dim=0)
        return pinned[env]

    @torch.no_grad()
    def calibrate_slots(self, q: torch.Tensor) -> None:
        if self.form == "lagrangian":
            self.mass_scale.copy_(torch.diagonal(self.mass(q), dim1=-2, dim2=-1).mean(dim=0))
            self.calibrated = True

    def rollout(
        self, q: torch.Tensor, qdot: torch.Tensor, actions: torch.Tensor, env: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``q, qdot [N, 2]``, ``actions [N, H, A]``, ``env [N]`` (None = adapted) -> ``q [N, H, 2]``."""

        context = self.contexts_for(env, q.shape[0])  # [N, 3]
        if self.form == "lagrangian":
            scale = self.mass_scale
            if not self.calibrated:  # during pretraining: track the current mass scale
                scale = torch.diagonal(self.mass(q), dim1=-2, dim2=-1).mean(dim=0).detach()
            dynamics = LagrangianDynamics(
                mass_matrix=_SlotMass(self.mass, context[:, :2] * scale),
                potential=self.potential,
                dissipation=self.damping,
                actuation=_SlotActuation(self.actuation, context[:, 2]),
            )
        else:
            dynamics = self.mlp
        out = []
        for step in range(actions.shape[1]):
            u = actions[:, step]
            if self.form == "mlp":
                u = torch.cat([u, context], dim=-1)
            q, qdot = semi_implicit_euler_step(dynamics, q, qdot, u, dt=self.dt, n_substeps=self.n_substeps)
            out.append(q)
        return torch.stack(out, dim=1)

    def adaptation_parameters(self, mode: str) -> list[nn.Parameter]:
        if mode.endswith("_ctx"):
            return [self.adapted]
        if mode.endswith("_full"):
            return [p for n, p in self.named_parameters() if n != "contexts"]
        raise ValueError(mode)


class _Adapted(nn.Module):
    """View of a ``ContextDynamics`` that always uses the test-time context."""

    def __init__(self, model: ContextDynamics) -> None:
        super().__init__()
        self.model = model

    def rollout(self, q: torch.Tensor, qdot: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.model.rollout(q, qdot, actions)


def _multi_env_windows(envs: Sequence[StateEpisodes], horizon: int, rail_margin: float | None):
    parts = [_windows(e, horizon, rail_margin=rail_margin) for e in envs]
    env_index = torch.cat([torch.full((p[0].shape[0],), i, dtype=torch.long) for i, p in enumerate(parts)])
    return tuple(torch.cat([p[j] for p in parts]) for j in range(4)) + (env_index,)


def pretrain(
    model: ContextDynamics, data, *, steps: int, lr: float, batch_size: int, seed: int,
) -> float:
    q0, v0, act, tgt, env = data
    generator = torch.Generator().manual_seed(seed)
    parameters = [p for n, p in model.named_parameters() if n != "adapted"]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    target = _pose(tgt)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        idx = torch.randint(0, q0.shape[0], (batch_size,), generator=generator)
        pred = model.rollout(q0[idx], v0[idx], act[idx], env[idx])
        loss = (_pose(pred) - target[idx]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        scheduler.step()
    return float(loss.detach())


def _adapt_grid(base, mode, data, select, test, *, lrs, checkpoints, horizons, select_horizon, batch_size, seed):
    from oracles.dynamics import fit

    grid: dict[str, dict] = {}
    for lr in lrs:
        adapted = copy.deepcopy(base)
        view = _Adapted(adapted)

        def record(step: int, lr: float = lr, view: _Adapted = view) -> None:
            grid[f"{lr:g}@{step}"] = {
                "select": windows_error(view, select, (select_horizon,))[select_horizon],
                "test": {str(h): e for h, e in windows_error(view, test, horizons).items()},
            }

        fit(
            view, data, adapted.adaptation_parameters(mode), steps=max(checkpoints), lr=lr,
            batch_size=batch_size, seed=seed, checkpoints=checkpoints, on_checkpoint=record,
        )
    return grid


def run_context_oracle(
    train_envs: dict[str, StateEpisodes],
    nominal_val: StateEpisodes,
    shifted: dict[str, tuple[StateEpisodes, StateEpisodes, StateEpisodes]],
    *,
    ks: Sequence[int] = (1, 2, 5, 20),
    segment_length: int = 50,
    train_horizon: int = 8,
    pretrain_steps: int = 15000,
    batch_size: int = 128,
    lr: float = 1e-3,
    adapt_lrs: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2),
    adapt_checkpoints: Sequence[int] = (0, 50, 100, 200, 400, 800),
    eval_horizons: Sequence[int] = (10, 20, 50),
    select_horizon: int = 20,
    eval_stride: int = 5,
    n_substeps: int = 2,
    rail_margin: float | None = 0.2,
    seed: int = 0,
    modes: Sequence[str] = CONTEXT_MODES,
    cache_dir: Path | None = None,
    log=print,
) -> OracleResult:
    """Train both forms across ``train_envs`` (first = nominal), adapt a context to each shift."""

    env_names = list(train_envs)
    config = {
        "version": 1,
        "train_envs": env_names,
        "train_episodes": {n: int(e.qpos.shape[0]) for n, e in train_envs.items()},
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
        "actuated_dims": [0],
        "seed": seed,
        "modes": list(modes),
        "shifts": list(shifted),
        "frame_dt": nominal_val.frame_dt,
    }
    pretrain_keys = (
        "train_envs", "train_episodes", "train_horizon", "pretrain_steps", "batch_size", "lr", "n_substeps",
        "rail_margin", "actuated_dims", "seed", "frame_dt",
    )
    cache = _Cache(cache_dir, config, {"oracle": "context", **{k: config[k] for k in pretrain_keys}})
    metrics: dict[str, float] = {}
    data = _multi_env_windows(list(train_envs.values()), train_horizon, rail_margin)
    nominal_eval = _windows(nominal_val, max(eval_horizons), stride=eval_stride, rail_margin=rail_margin)
    log(f"train: {data[0].shape[0]} windows over {len(env_names)} environments {env_names}")
    models: dict[str, ContextDynamics] = {}
    for form in ("lagrangian", "mlp"):
        start = time.time()
        torch.manual_seed(seed)
        model = ContextDynamics(form, len(env_names), dt=nominal_val.frame_dt, n_substeps=n_substeps)
        if cache.load_model(form, model):
            log(f"pretrained {form}: loaded from cache")
        else:
            final = pretrain(model, data, steps=pretrain_steps, lr=lr, batch_size=batch_size, seed=seed)
            cache.save_model(form, model)
            log(f"pretrained {form}: final loss {final:.2e} ({time.time() - start:.0f}s)")
        model.calibrate_slots(data[0][data[4] == 0])
        models[form] = model
        for h, e in windows_error(_Adapted(model), nominal_eval, eval_horizons).items():
            metrics[f"nominal.{form}.dev_h{h}"] = e
        contexts = model.contexts_for(torch.arange(len(env_names)), len(env_names)).detach().numpy().round(3)
        log(
            f"  nominal {form} dev: " + "  ".join(f"h{h} {metrics[f'nominal.{form}.dev_h{h}']:.4f}" for h in eval_horizons)
            + f"\n  learned contexts: {dict(zip(env_names, contexts.tolist()))}",
        )

    cells_total = len(shifted) * len(ks) * len(modes)
    for shift_name, (shift_train, shift_select, shift_test) in shifted.items():
        select = _windows(shift_select, select_horizon, stride=eval_stride, rail_margin=rail_margin)
        test = _windows(shift_test, max(eval_horizons), stride=eval_stride, rail_margin=rail_margin)
        for form in ("lagrangian", "mlp"):
            for h, e in windows_error(_Adapted(models[form]), test, eval_horizons).items():
                metrics[f"{shift_name}.k0.{form}.dev_h{h}"] = e
        segments = free_segments(shift_train, segment_length, rail_margin, seed)
        log(
            f"{shift_name}: zero-shot h{select_horizon} lag {metrics[f'{shift_name}.k0.lagrangian.dev_h{select_horizon}']:.4f}"
            f" mlp {metrics[f'{shift_name}.k0.mlp.dev_h{select_horizon}']:.4f}",
        )
        for k in ks:
            adapt_data = _windows(_subset(shift_train, segments[:k], segment_length), train_horizon, rail_margin=rail_margin)
            for mode in modes:
                key = f"{shift_name}.k{k}.{mode}"
                if key in cache.cells:
                    continue
                start = time.time()
                form = "lagrangian" if mode.startswith("lag") else "mlp"
                grid = _adapt_grid(
                    models[form], mode, adapt_data, select, test, lrs=adapt_lrs, checkpoints=adapt_checkpoints,
                    horizons=eval_horizons, select_horizon=select_horizon, batch_size=batch_size, seed=seed + k,
                )
                cache.save_cell(key, grid)
                best = min(grid.values(), key=lambda g: g["select"])
                log(
                    f"[{len(cache.cells)}/{cells_total}] {key}: best-cell test h{select_horizon} "
                    f"{best['test'][str(select_horizon)]:.4f} ({time.time() - start:.0f}s)",
                )

    recipes: dict[str, str] = {}
    for mode in modes:
        cells = [cache.cells[f"{s}.k{k}.{mode}"] for s in shifted for k in ks]
        recipes[mode] = min(cells[0], key=lambda p: float(np.mean([c[p]["select"] for c in cells])))
    for shift_name in shifted:
        for k in ks:
            for mode in modes:
                grid = cache.cells[f"{shift_name}.k{k}.{mode}"]
                cell_best = min(grid, key=lambda p: grid[p]["select"])
                for h in eval_horizons:
                    metrics[f"{shift_name}.k{k}.{mode}.dev_h{h}"] = grid[recipes[mode]]["test"][str(h)]
                    metrics[f"{shift_name}.k{k}.{mode}.cellbest.dev_h{h}"] = grid[cell_best]["test"][str(h)]
    criteria = []
    for k in [k for k in (1, 5) if k in ks]:
        for suffix in ("", ".cellbest"):
            ratios = [
                metrics[f"{s}.k{k}.mlp_ctx{suffix}.dev_h{select_horizon}"]
                / max(metrics[f"{s}.k{k}.lag_ctx{suffix}.dev_h{select_horizon}"], 1e-9)
                for s in shifted
            ]
            metrics[f"mlp_ctx_over_lag_ctx_ratio{suffix}_k{k}_h{select_horizon}"] = float(np.mean(ratios))
        criteria.append(
            Criterion(
                f"mlp_ctx_over_lag_ctx_ratio_k{k}_h{select_horizon}", ">=", 1.3,
                "with matched multi-environment training and 3 adapted numbers each, the Lagrangian context "
                "extrapolates with >= 30% lower error",
            ),
        )
    return OracleResult(
        name="context_oracle",
        hypotheses=["H1b"],
        question="Against a context-conditioned MLP trained on the same masses, does structure adapt better?",
        config={**config, "recipes": recipes},
        metrics=metrics,
        criteria=criteria,
        notes=[
            "Both forms: per-environment 3-number context; nominal context pinned to zero.",
            "Errors are mean |Δ(x, cos, sin)| on rail-free windows; shifted systems are outside the training range.",
        ],
    )
