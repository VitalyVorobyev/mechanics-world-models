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
from oracles.ceiling import TrueAcceleration, integrate
from oracles.common import Criterion, OracleResult, StateEpisodes, noisy_observation, pose_features
from oracles.start_state import context_windows, evaluate_starts
from oracles.dynamics import (
    _Cache,
    _pose,
    _subset,
    _windows,
    pose_loss,
    free_segments,
    windows_error,
)

CONTEXT_DIM = 3
CONTEXT_MODES = ("lag_ctx", "lag_full", "mlp_ctx", "mlp_full")


def simulate_state_episodes(
    scales: dict[str, float] | None, *, episodes: int, seed: int, action_repeat: int = 2,
    env_name: str = "cartpole-swingup",
) -> StateEpisodes:
    """Random-policy state trajectories (no rendering), same policy as the collector."""

    env = DmControlPixelEnv(
        PixelEnvConfig(env_name=env_name, action_repeat=action_repeat, seed=seed, physics_param_scales=scales),
    )
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


class _SlotPotential(nn.Module):
    def __init__(self, base: Potential, log_scale: torch.Tensor) -> None:
        super().__init__()
        self.base, self.log_scale, self.q_dim = base, log_scale, base.q_dim

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.log_scale.exp() * self.base(q)  # log_scale [N]


class ContextDynamics(nn.Module):
    """Either form with a per-environment context table and one test-time context.

    With ``potential_slot`` the context has a fourth number for both forms; the
    Lagrangian uses it as a log scale on ``V`` (a gravity change scales ``V``).
    """

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
        potential_slot: bool = False,
    ) -> None:
        super().__init__()
        self.form, self.dt, self.n_substeps = form, dt, n_substeps
        self.context_dim = CONTEXT_DIM + int(potential_slot)
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
                q_dim=2, action_dim=action_dim + self.context_dim, hidden_size=2 * hidden, hidden_layers=2,
                angle_dims=(1,),
            )
        else:
            raise ValueError(f"unknown dynamics form {form!r}")
        # Row 0 (nominal) is pinned to zero in `contexts_for`.
        self.contexts = nn.Parameter(torch.zeros(num_envs, self.context_dim))
        self.adapted = nn.Parameter(torch.zeros(self.context_dim))
        self.register_buffer("mass_scale", torch.ones(2), persistent=False)
        self.calibrated = False

    def contexts_for(self, env: torch.Tensor | None, n: int) -> torch.Tensor:
        if env is None:
            return self.adapted.expand(n, self.context_dim)
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

        context = self.contexts_for(env, q.shape[0])  # [N, 3] (or [N, 4] with the potential slot)
        if self.form == "lagrangian":
            scale = self.mass_scale
            if not self.calibrated:  # during pretraining: track the current mass scale
                scale = torch.diagonal(self.mass(q), dim1=-2, dim2=-1).mean(dim=0).detach()
            dynamics = LagrangianDynamics(
                mass_matrix=_SlotMass(self.mass, context[:, :2] * scale),
                potential=_SlotPotential(self.potential, context[:, 3]) if self.context_dim > 3 else self.potential,
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


def _multi_env_windows(
    envs: Sequence[StateEpisodes], horizon: int, rail_margin: float | None,
    references: Sequence[StateEpisodes] | None = None,
):
    references = references or envs
    parts = [
        _windows(e, horizon, rail_margin=rail_margin, rail_reference=r.qpos) for e, r in zip(envs, references)
    ]
    env_index = torch.cat([torch.full((p[0].shape[0],), i, dtype=torch.long) for i, p in enumerate(parts)])
    return tuple(torch.cat([p[j] for p in parts]) for j in range(4)) + (env_index,)


def pretrain(
    model: ContextDynamics, data, *, steps: int, lr: float, batch_size: int, seed: int,
    huber_delta: float | None = None,
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
        loss = pose_loss(_pose(pred), target[idx], huber_delta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        scheduler.step()
    return float(loss.detach())


def _adapt_grid(
    base, mode, data, select, test, *, lrs, checkpoints, horizons, select_horizon, batch_size, seed,
    test_noisy=None, huber_delta=None,
):
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
            if test_noisy is not None:
                grid[f"{lr:g}@{step}"]["test_noisy"] = {
                    str(h): e for h, e in windows_error(view, test_noisy, horizons).items()
                }

        fit(
            view, data, adapted.adaptation_parameters(mode), steps=max(checkpoints), lr=lr,
            batch_size=batch_size, seed=seed, checkpoints=checkpoints, on_checkpoint=record,
            huber_delta=huber_delta,
        )
    return grid


def _torch_rollout(model: ContextDynamics):
    def rollout(q0: np.ndarray, v0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        with torch.enable_grad():  # the Lagrangian step uses autograd internally
            return model.rollout(as_t(q0), as_t(v0), as_t(actions)).detach().double().numpy()

    return rollout


def _true_rollout(accel: TrueAcceleration, frame_dt: float):
    return lambda q0, v0, actions: integrate(accel, q0, v0, actions, frame_dt, 4)


def _start_state_eval(
    models, per_shift, recipes, metrics, cache, *, observe, ks, modes, segment_length, train_horizon,
    rail_margin, eval_horizons, eval_stride, batch_size, seed, start_context, pose_noise, shift_scales,
    frame_dt, log, huber_delta=None,
) -> None:
    """Re-adapt each selected recipe (deterministic) and score start-state estimators."""

    from oracles.dynamics import fit

    horizons = tuple(eval_horizons)
    # Residual weights for `shoot`. Encoder poses come without added noise; weighting by a zero sigma
    # (infinite weights) froze the fit at its initial guess, so fall back to equal 0.01 m / 0.01 rad.
    sigma = tuple(pose_noise) if all(v > 0 for v in pose_noise) else (0.01, 0.01)
    version = "v2" if sigma == tuple(pose_noise) else "v2s"
    for shift_name, (shift_train, observed_train, segments, shift_test, salt) in per_shift.items():
        windows = context_windows(
            shift_test, observe(shift_test, salt + 2, velocity="backward"), horizon=max(horizons),
            context=start_context, stride=eval_stride, rail_margin=rail_margin,
        )
        if shift_scales is not None and shift_name in shift_scales:
            key = f"starts{start_context}{version}.{shift_name}.floor_true"
            if key not in cache.cells:
                accel = TrueAcceleration(shift_scales[shift_name])
                cache.save_cell(key, evaluate_starts(_true_rollout(accel, frame_dt), windows, horizons, sigma=sigma))
                accel.close()
            for name, value in cache.cells[key].items():
                metrics[f"{shift_name}.start_floor.{name}"] = value
        for k in ks:
            adapt_data = _windows(
                _subset(observed_train, segments[:k], segment_length), train_horizon, rail_margin=rail_margin,
                rail_reference=_subset(shift_train, segments[:k], segment_length).qpos,
            )
            for mode in modes:
                key = f"starts{start_context}{version}.{shift_name}.k{k}.{mode}"
                if key not in cache.cells:
                    start = time.time()
                    form = "lagrangian" if mode.startswith("lag") else "mlp"
                    adapted = copy.deepcopy(models[form])
                    lr, steps = recipes[mode].split("@")
                    if int(steps) > 0:
                        fit(
                            _Adapted(adapted), adapt_data, adapted.adaptation_parameters(mode), steps=int(steps),
                            lr=float(lr), batch_size=batch_size, seed=seed + k, huber_delta=huber_delta,
                        )
                    cache.save_cell(
                        key, evaluate_starts(_torch_rollout(adapted), windows, horizons, sigma=sigma),
                    )
                    h = 20 if 20 in horizons else horizons[-1]
                    shoot = cache.cells[key][f"shoot.dev_h{h}"]
                    log(f"start-state {shift_name}.k{k}.{mode}: shoot h{h} {shoot:.4f} ({time.time() - start:.0f}s)")
                for name, value in cache.cells[key].items():
                    metrics[f"{shift_name}.k{k}.{mode}.start.{name}"] = value
    if "lag_ctx" in modes and "mlp_ctx" in modes:
        h = 20 if 20 in horizons else horizons[-1]
        for k in ks:
            for name in ("true_start", "backward", "quadfit", "shoot"):
                ratios = [
                    metrics[f"{s}.k{k}.mlp_ctx.start.{name}.dev_h{h}"] / max(metrics[f"{s}.k{k}.lag_ctx.start.{name}.dev_h{h}"], 1e-9)
                    for s in per_shift
                ]
                metrics[f"start_ratio.{name}_k{k}_h{h}"] = float(np.mean(ratios))


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
    pose_noise: tuple[float, float] = (0.0, 0.0),
    shift_scales: dict[str, dict[str, float] | None] | None = None,
    start_context: int = 0,
    train_velocity: str = "central",
    pose_source: str | None = None,
    potential_slot: bool = False,
    pose_outliers: tuple[float, float, float] = (0.0, 0.0, 0.0),
    huber_delta: float | None = None,
    cache_dir: Path | None = None,
    log=print,
) -> OracleResult:
    """Train both forms across ``train_envs`` (first = nominal), adapt a context to each shift.

    With ``pose_noise = (sigma_x, sigma_theta)`` nonzero, every window the
    models train, adapt or select on is built from noisy poses with
    finite-difference velocities (``noisy_observation``; ``train_velocity``,
    default central differences, the mechanics model's training stencil). Test errors are reported twice: from the true
    start state (``.dev_h*``, dynamics quality) and from a noisy start state
    with a backward-difference velocity (``.noisy_start.dev_h*``, the
    mechanics model's open-loop start), both against true poses. ``shift_scales`` (physics
    scales per shift) adds the true-dynamics floor on the same windows.
    ``start_context > 0`` (noisy runs only) re-adapts each mode's selected
    recipe and compares start-state estimators that use that many context
    frames (``oracles.start_state``), with the true-dynamics floor for each:
    ``{shift}.k{k}.{mode}.start.{estimator}.dev_h*``.
    ``pose_source`` names where ``observed_qpos`` of the given episodes came
    from (e.g. a pixel encoder, ``oracles.pixel_pose``); it switches on the
    observed-pose path even without added noise and enters the cache keys.
    """

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
    noisy = any(s > 0 for s in pose_noise) or pose_source is not None
    pretrain_keys = (
        "train_envs", "train_episodes", "train_horizon", "pretrain_steps", "batch_size", "lr", "n_substeps",
        "rail_margin", "actuated_dims", "seed", "frame_dt",
    )
    if huber_delta is not None:  # only then, so earlier caches keep their hashes
        config["huber_delta"] = huber_delta
        pretrain_keys += ("huber_delta",)
    if potential_slot:  # only then, so earlier caches keep their hashes
        config["potential_slot"] = True
        pretrain_keys += ("potential_slot",)
    if noisy:  # only then, so earlier caches keep their hashes
        config["pose_noise"] = list(pose_noise)
        pretrain_keys += ("pose_noise",)
        if pose_outliers[0] > 0:
            config["pose_outliers"] = list(pose_outliers)
            pretrain_keys += ("pose_outliers",)
        if pose_source is not None:
            config["pose_source"] = pose_source
            pretrain_keys += ("pose_source",)
        if train_velocity != "central":
            config["train_velocity"] = train_velocity
            pretrain_keys += ("train_velocity",)

    def observe(episodes: StateEpisodes, salt: int, velocity: str = train_velocity) -> StateEpisodes:
        if not noisy:
            return episodes
        return noisy_observation(
            episodes, *pose_noise, seed=10_000 * seed + salt, velocity=velocity, outliers=pose_outliers,
        )

    cache = _Cache(cache_dir, config, {"oracle": "context", **{k: config[k] for k in pretrain_keys}})
    if noisy and start_context > 0:  # evaluation-only: added after hashing so finished cells are reused
        config["start_context"] = start_context
    metrics: dict[str, float] = {}
    observed_envs = [observe(e, i) for i, e in enumerate(train_envs.values())]
    data = _multi_env_windows(observed_envs, train_horizon, rail_margin, references=list(train_envs.values()))
    nominal_eval = _windows(nominal_val, max(eval_horizons), stride=eval_stride, rail_margin=rail_margin)
    log(f"train: {data[0].shape[0]} windows over {len(env_names)} environments {env_names}")
    models: dict[str, ContextDynamics] = {}
    for form in ("lagrangian", "mlp"):
        start = time.time()
        torch.manual_seed(seed)
        model = ContextDynamics(
            form, len(env_names), dt=nominal_val.frame_dt, n_substeps=n_substeps, potential_slot=potential_slot,
        )
        if cache.load_model(form, model):
            log(f"pretrained {form}: loaded from cache")
        else:
            final = pretrain(
                model, data, steps=pretrain_steps, lr=lr, batch_size=batch_size, seed=seed, huber_delta=huber_delta,
            )
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
    per_shift: dict[str, tuple] = {}
    for index, (shift_name, (shift_train, shift_select, shift_test)) in enumerate(shifted.items()):
        salt = 100 * (index + 1)
        observed_train = observe(shift_train, salt)
        observed_select = observe(shift_select, salt + 1)
        select = _windows(
            observed_select, select_horizon, stride=eval_stride, rail_margin=rail_margin,
            rail_reference=shift_select.qpos,
        )
        test = _windows(shift_test, max(eval_horizons), stride=eval_stride, rail_margin=rail_margin)
        test_noisy = None
        if noisy:  # noisy start state, true targets
            start = _windows(
                observe(shift_test, salt + 2, velocity="backward"), max(eval_horizons), stride=eval_stride, rail_margin=rail_margin,
                rail_reference=shift_test.qpos,
            )
            test_noisy = (start[0], start[1], test[2], test[3])
        for form in ("lagrangian", "mlp"):
            for h, e in windows_error(_Adapted(models[form]), test, eval_horizons).items():
                metrics[f"{shift_name}.k0.{form}.dev_h{h}"] = e
        if shift_scales is not None and shift_name in shift_scales:
            accel = TrueAcceleration(shift_scales[shift_name])
            for label, windows in (("floor_true", test), ("floor_true.noisy_start", test_noisy)):
                if windows is None:
                    continue
                pred = integrate(
                    accel, windows[0].double().numpy(), windows[1].double().numpy(), windows[2].double().numpy(),
                    nominal_val.frame_dt, 4,
                )
                dev = np.abs(pose_features(pred) - pose_features(windows[3].double().numpy()))
                for h in eval_horizons:
                    metrics[f"{shift_name}.{label}.dev_h{h}"] = float(dev[:, h - 1].mean())
            accel.close()
        segments = free_segments(shift_train, segment_length, rail_margin, seed)
        per_shift[shift_name] = (shift_train, observed_train, segments, shift_test, salt)
        log(
            f"{shift_name}: zero-shot h{select_horizon} lag {metrics[f'{shift_name}.k0.lagrangian.dev_h{select_horizon}']:.4f}"
            f" mlp {metrics[f'{shift_name}.k0.mlp.dev_h{select_horizon}']:.4f}",
        )
        for k in ks:
            adapt_data = _windows(
                _subset(observed_train, segments[:k], segment_length), train_horizon, rail_margin=rail_margin,
                rail_reference=_subset(shift_train, segments[:k], segment_length).qpos,
            )
            for mode in modes:
                key = f"{shift_name}.k{k}.{mode}"
                if key in cache.cells:
                    continue
                start = time.time()
                form = "lagrangian" if mode.startswith("lag") else "mlp"
                grid = _adapt_grid(
                    models[form], mode, adapt_data, select, test, lrs=adapt_lrs, checkpoints=adapt_checkpoints,
                    horizons=eval_horizons, select_horizon=select_horizon, batch_size=batch_size, seed=seed + k,
                    test_noisy=test_noisy, huber_delta=huber_delta,
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
                    if noisy:
                        metrics[f"{shift_name}.k{k}.{mode}.noisy_start.dev_h{h}"] = (
                            grid[recipes[mode]]["test_noisy"][str(h)]
                        )
    if noisy and start_context > 0:
        _start_state_eval(
            models, per_shift, recipes, metrics, cache, observe=observe, ks=ks, modes=modes,
            segment_length=segment_length, train_horizon=train_horizon, rail_margin=rail_margin,
            eval_horizons=eval_horizons, eval_stride=eval_stride, batch_size=batch_size, seed=seed,
            start_context=start_context, pose_noise=pose_noise, shift_scales=shift_scales,
            frame_dt=nominal_val.frame_dt, log=log, huber_delta=huber_delta,
        )
    criteria = []
    for k in [k for k in (1, 5) if k in ks]:
        for suffix in ("", ".cellbest") + ((".noisy_start",) if noisy else ()):
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
