# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Intent

Goal (owner, 2026-10-19): a system that takes **camera images**, **learns the mechanics** (a learned Lagrangian over given generalized coordinates), and **drives the mechanical system** with that learned model. Cartpole swing-up is the showcase; the aim is more complex systems; parameter extrapolation is **not** an aim. Novelty is not an objective: build from standard components and invent only where a measured gap forces it. The active scope, question tree, budget and **stop date** are in `docs/charter.md`; do not work outside its nodes without an owner decision. The main line is E-006 (MPPI ceiling on the true model) → E-007 (MPPI over a learned Lagrangian vs an MLP on true state) → E-008 (sim-trained keypoints + estimator in the loop, then the demo). E-001…E-005 (known-model plan) are abandoned.

The repository also contains two end-to-end pixel world models (RSSM baseline; `MechanicsWorldModel` with a learned Lagrangian) and their trainers/evaluators. They are **frozen**: keep their tests green, do not develop them further unless the charter reopens them. **Do not add actor-critic, replay buffers, zarr, distributed collection, or augmentation pipelines unless a task explicitly asks for them.**

**Research protocol v2 (binding, see `AGENTS.md`):** follow `docs/research_protocol.md`. In short: experiments are records `docs/experiments/E-NNN-*.md`, registered by committing their prediction / primary metric / baselines / decision rule (with a stop branch) *before* running, and run only once the owner has approved them; runs use `--experiment E-NNN` from a clean tree (oracle CLIs refuse dirty trees unless `--allow-dirty`) and append to the committed `results/ledger.jsonl`; tables come from `scripts/research/report.py`, never typed by hand; claims go to `docs/claims.md` with grades A/B/C; three "diagnose further" outcomes in a row on one node force a zoom-out and an owner decision. Run `.venv/bin/python scripts/research/check.py` at the start and end of a session.

Session routine: read `docs/charter.md`, `docs/current_status.md`, `docs/claims.md` and the open experiment records first. At the end, generate reports, update claims and `docs/hypotheses.md`, **rewrite** (do not append to) `docs/current_status.md`, and list the decisions waiting on the owner.

Authoritative docs:
- `docs/charter.md` — goal, root question Q0, question tree, budget, stop date, decision log. Changes only by owner decision.
- `docs/current_status.md` — one page: what is happening now and what waits on the owner.
- `docs/claims.md` — graded facts with ledger evidence ("what do we know").
- `docs/hypotheses.md` — compact index of hypotheses and assumptions.
- `docs/review-2026-10-19.md` — critical review behind the current charter (why the world-model line stopped; process failures found).
- `docs/development.md` — setup, collection, training, evaluation, debugging and research-tooling commands (prefer this over README for day-to-day).
- `docs/models/*.md` — one concise document per trainable model; update it when the model, loss, optimizer, or default training config changes.
- `docs/archive/`, `docs/experiments/archive/` — history (April spec, status log up to 2026-10-19, the full 2026-10-18/19 registry). Read only when a claim cites it.
- `AGENTS.md` — authoritative for agent conventions; this file summarizes.

## Environment and Commands

Python `>=3.10,<3.13`; 3.12 is the practical target (`dm-control` builds are awkward on 3.13). Use `.venv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Entry points are installed as console scripts into `.venv/bin/` (see `[project.scripts]` in `pyproject.toml`): `collect-cartpole`, `dataset-stats`, `train-rssm`, `train-mechanics`, `eval-rssm`, `eval-mechanics`, `eval-control`, `plot-training-history`, `preview-dataset`, `preview-crop`, `debug-reconstruction`, `overfit-reconstruction`, `dataset-image-diagnostics`, `foreground-mask-preview`.

The mechanics trainer / evaluator accept the same data on disk as the RSSM ones and write matching `history.jsonl` / `metrics.json` schemas; when the dataset has `physics_qpos`/`physics_qvel`, both `eval-mechanics` and `eval-rssm` report `state_probes` (linear + MLP probes per latent block onto `[x, cos θ, sin θ]` and `qvel`, episode-level split — the factorization check). `eval-mechanics` also reports the legacy raw-angle `linear_probe` (not a valid gate) and a learned-energy drift. Both report a `copy_last_frame_mse_h*` baseline. `scripts/bench_mechanics_step.py` is a single-step wall-clock benchmark harness used to catch perf regressions in the Lagrangian loop.

Tests:
```bash
.venv/bin/python -m pytest                                  # full suite
.venv/bin/python -m pytest tests/test_world_model.py -k rssm  # single test
```
Tests should stay synthetic/offline where possible. The dm_control render test skips cleanly without working offscreen rendering. On Linux/headless machines set `MUJOCO_GL=egl` for collection.

Concrete command recipes (collection, training with foreground/transition weights, evaluation, crop previews, debugging) live in `docs/development.md`. Don't re-derive them from scratch; copy from there.

## Architecture

Four layers; each lives in its own `src/` subpackage and is exposed via lazy `__init__.py` re-exports.

**Data (`src/data/`)** — random-policy collection to one-episode-per-file `.npz` archives, each with `N + 1` images and `N` transitions so `(obs_t, action_t, obs_{t+1})` never crosses episode boundaries. `sequence_dataset.py` yields fixed-length `[T, ...]` samples; `make_train_val_datasets` splits deterministically by whole episode files (never by frames) and raises `ValueError` if a split has no usable episodes. `image_transforms.ImageCropConfig` crops `uint8` frames and resizes back to the model input size so the trainer input stays `[B, T, 3, 84, 84]`.

**Models (`src/models/`)** — two trainable families share encoder/decoder code:
- **RSSM (`VisualWorldModel`)** composes `ConvEncoder`: `[B, T, 3, 84, 84] -> [B, T, E]`; `RSSM`: GRU-backed action-conditioned dynamics producing deterministic `h_t` (size `H`), stochastic `z_t` (size `Z`), prior `p(z_t | h_t)`, posterior `q(z_t | h_t, embed_t)` with `RSSMOutput.features = cat([h, z], -1)`; `ImageDecoder`: `[B, T, H + Z] -> [B, T, 3, 84, 84]` (sigmoid-normalized); and `latent_predictor`, an MLP from next prior features to encoder embedding space used only when `next_observations` is present. Reward head is opt-in (`--reward-head`).
- **Mechanics (`MechanicsWorldModel`, `src/models/mechanics/`)** composes the shared `ConvEncoder` trunk with a `FactoredEncoder` that emits Gaussian posterior params for `q` (`q_dim=2` for cartpole) and nuisance `z` (`nuisance_dim=4` default on cartpole). **`q̇` is derived, not emitted** — `derive_qdot_from_q(q_mean, q_std, dt)` in `encoder.py` computes `q̇_t = (q_{t+1} − q_{t-1}) / (2·dt)` (central diff interior, forward/backward at the boundaries) and the Gaussian pushforward `q̇_std = sqrt(q_std_{t+1}² + q_std_{t−1}²) / (2·dt)`. Encoder requires `T ≥ 2`; `initial_posterior` / `posterior_step` accept `[B, T≥2, C, H, W]` and return the posterior at the last frame. `lagrangian.py` hosts Cholesky-PD `MassMatrix` (ridge `eps=1e-2`), learned `Potential`, PSD `Dissipation`, learned `Actuation`; all four MLPs tanh-bound their inputs (q scale 5, qdot scale 20) to prevent cold-start dynamics blow-up (F2 defensive fix). `forward_acceleration` implements the mechanical-form Euler-Lagrange step using autograd.grad (per-dim loop avoids the MPS bugs in `torch.autograd.functional.jvp`). `integrator.py` is a semi-implicit symplectic Euler step. `transition.py` composes the integrator on `(q, q̇)` with an AR(1) nuisance channel; its `prior_min_std_qdot` defaults to `prior_min_std / dt` so the integrator's prior q̇-std matches the encoder's pushforward scale at init. By default (legacy `concat` mode) the decoder input is `concat(q, q̇, z)` of size `2 * q_dim + nuisance_dim`. October 2026 options, all defaulting to legacy behaviour (see `docs/models/mechanics_world_model.md`): `--nuisance-mode style` (decoder renders from `q` only, `z` = per-channel colour modulation, reward head reads `(q, q̇)`), `--angle-dims` (S¹ coordinates via atan2 / cos-sin, wrapped differences), `--qdot-stencil backward` (causal q̇), `--dynamics-form mlp` (unstructured-dynamics control), `--dissipation-form linear` (passive constant damping). `--dt` defaults to the dataset frame interval (`data/frame_timing.py`, 0.02 s for cartpole). The open-loop start velocity is always a backward difference of the context frames, and q̇ samples are derived from q samples. Reward head is also opt-in. The imagination API mirrors the RSSM's, but `eval-control` / `CEMPlanner` currently support only `VisualWorldModel`. Critical MPS detail: **do not reintroduce `torch.linalg.solve` inside the imagination loop** — it corrupts kernel dispatch under autograd.grad; a regression test in `tests/test_mechanics_primitives.py` guards this.

**Training and loss (`src/train/`, `src/models/losses.py`, `src/models/mechanics/losses.py`)** — offline only. The RSSM composite loss combines KL (balanced + free-nats), posterior reconstruction, **foreground-masked** reconstruction (background-subtracted saliency mask), dynamic-mask reconstruction, transition reconstruction (decoded prior features at `t` targeted against `obs_{t+1}`), foreground-masked transition reconstruction, **imagination reconstruction** (decoded prior features over `K_ctx + K_imag` horizon), optional reward prediction, and BYOL-style latent consistency. The mechanics loss uses a three-branch balanced KL with asymmetric free-nats (tight on `q`/`q̇`, loose on `z`), posterior / foreground / imagination reconstruction, and optional reward — **no smoothness prior** (the kinematic constraint `q̇ = d/dt(q)` is now structural, not a soft loss term). Each term has its own CLI weight flag. The mechanics trainer also includes a NaN/Inf skip guard (F1 defensive fix) so a pathological batch cannot poison AdamW's running moments. Both trainers use AdamW with linear warmup + cosine LR, write per-epoch and `latest.pt` checkpoints plus JSONL `history.jsonl`, support `--auto-resume`, and log a `val_open_loop_fg_mse_h*` probe every N steps so the K1 metric is observable live. The mechanics history also carries z-channel diagnostics (`z_post_mean_rms`, `kl_z_t0_mean`, `kl_z_transition_mean`, ...) surfaced during the v3 debugging. **Keep JSONL history machine-readable** — terminal formatting (Rich colors, timestamps) must not change the schema.

**Eval (`src/eval/`)** — loads a checkpoint and reports scalar metrics plus videos/contact sheets. Three prediction modes: posterior reconstruction at `t`, one-step prior prediction for `t`, and open-loop (posterior for `K` warmup steps, then prior-only rollout). Uses Gaussian means (deterministic) so a fixed seed gives reproducible metrics. Physics-state probes (`src/eval/probes.py::run_state_probes`) and the learned-energy drift are described above. `eval-control` drives the CEM/MPC planner (RSSM only) against a live dm_control env for in-distribution and OOD settings. OOD physics shifts must use `pole_length=<s>` (MJCF-level) and `body_mass_<i>=<s>` (mass and inertia together); `geom_size_*` edits after compilation change only collision/rendering, not dynamics. If a checkpoint was trained with a crop, eval uses that crop unless overridden with `--crop-mode`.

## Critical Conventions

- **Transition alignment is `obs_t --action_t--> obs_{t+1}`.** Batches expose both `observations[:, t]` and `next_observations[:, t]`; `next_observations[:, t]` is the target for transition/prior decoding and foreground-transition losses. Off-by-one here silently masquerades as weak dynamics. The mechanics imagination contract follows the same convention: `imagined[:, k]` targets `next_observations[:, K_ctx - 1 + k]`.
- Dataloader batch shapes: `observations`/`next_observations` `[B, T, 3, 84, 84]` float32 in `[0, 1]`; `actions` `[B, T, A]`; `rewards` `[B, T]`; `dones` `[B, T]` bool; optional `physics_qpos`/`physics_qvel` `[B, T, D_q]` float64 (present when the dataset was collected after Phase 1.1, consumed only by eval diagnostics — **never** by a training loss). Keep shape comments at RSSM / mechanics rollout sites.
- Don't change the `.npz` storage format or the `N + 1 images / N transitions` alignment without an explicit task. Optional physics fields are backwards-compatible — loaders return `None` for them when absent.
- Do not make research-performance claims in public docs unless backed by code, tests, or logged experiments.
- When adding a trainable model, add a document under `docs/models/` covering purpose, architecture/parameter count, objective, optimizer, and relation to other models; update it whenever the model/loss/optimizer/default config changes.
- Avoid global state in data, models, and training code; prefer small modules and explicit dataclasses.
- On Apple MPS, avoid `torch.linalg.solve` and `torch.autograd.functional.jvp` inside autograd.grad-with-create_graph loops — they corrupt kernel dispatch. Use `torch.linalg.inv` + einsum and a per-dim `autograd.grad` loop instead. Regression tests pin this.
- Checkpoints carry `model_kind` ("rssm" or "mechanics"); each trainer's `load_checkpoint` refuses mismatched kinds so RSSM weights cannot be silently loaded into a mechanics model or vice versa.
