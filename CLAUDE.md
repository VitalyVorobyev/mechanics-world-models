# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Intent

Research code for mechanics-structured visual world models for model-based RL on `dm_control cartpole-swingup` pixels. The repository contains **two trainable world models**: an unstructured RSSM baseline (B1) and a factored `(q, qdot, z_nuisance)` model (B_main) with a learned Lagrangian, Rayleigh dissipation, and a symplectic integrator. Phase-1 scaffolding (reward head, CEM/MPC, physics-perturbation wrapper, visual distractors, `eval-control`) is in place. The ablations B2 (no dissipation) / B3 (no q/z split) / B4 (contrastive) and the OOD evaluation grid are still ahead. **Do not add actor-critic, replay buffers, zarr, distributed collection, or augmentation pipelines unless a task explicitly asks for them.**

Authoritative docs:
- `docs/research_spec.md` — long-term research direction; do not silently rewrite hypotheses, baselines, or kill criteria.
- `docs/development.md` — setup, collection, training, evaluation, debugging commands (prefer this over README for day-to-day).
- `docs/current_status.md` — where the project actually is today: B1 RSSM works (h=10 open-loop FG-MSE ≈ 0.005), Phase 2 mechanics code landed, first `train-mechanics` run diverged at step ~200 (KL blow-up → encoder collapse). Active work is a retry with looser KL pressure and lower LR.
- `docs/models/*.md` — one concise document per trainable model (currently `rssm_visual_world_model.md` and `mechanics_world_model.md`); update it when the model, loss, optimizer, or default training config changes. Add a new doc when introducing a new trainable model.
- `AGENTS.md` — authoritative for agent conventions; this file summarizes.

## Environment and Commands

Python `>=3.10,<3.13`; 3.12 is the practical target (`dm-control` builds are awkward on 3.13). Use `.venv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Entry points are installed as console scripts into `.venv/bin/` (see `[project.scripts]` in `pyproject.toml`): `collect-cartpole`, `dataset-stats`, `train-rssm`, `train-mechanics`, `eval-rssm`, `eval-mechanics`, `eval-control`, `plot-training-history`, `preview-dataset`, `preview-crop`, `debug-reconstruction`, `overfit-reconstruction`, `dataset-image-diagnostics`, `foreground-mask-preview`.

The mechanics trainer / evaluator accept the same data on disk as the RSSM ones and write matching `history.jsonl` / `metrics.json` schemas; `eval-mechanics` additionally reports a linear-probe R² (when the dataset has `physics_qpos`/`physics_qvel`) and a learned-energy drift over the open-loop imagination window. `scripts/bench_mechanics_step.py` is a single-step wall-clock benchmark harness used to catch perf regressions in the Lagrangian loop.

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
- **Mechanics (`MechanicsWorldModel`, `src/models/mechanics/`)** composes the shared `ConvEncoder` trunk with a `FactoredEncoder` that emits Gaussian posterior params for `(q, qdot)` (``q_dim=2`` for cartpole) and nuisance `z` (``nuisance_dim=32`` default). `lagrangian.py` hosts Cholesky-PD `MassMatrix`, learned `Potential`, PSD `Dissipation`, and learned `Actuation`. `forward_acceleration` implements the mechanical-form Euler-Lagrange step using autograd.grad (per-dim loop avoids the MPS bugs in `torch.autograd.functional.jvp`). `integrator.py` is a semi-implicit symplectic Euler step. `transition.py` composes the integrator on `(q, qdot)` with an AR(1) nuisance channel. The decoder input is `concat(q, qdot, z)` of size `2 * q_dim + nuisance_dim`. Reward head is also opt-in. Phase-0 imagination-rollout contract is preserved exactly so `eval-control` / CEM plug in unchanged. Critical MPS detail: **do not reintroduce `torch.linalg.solve` inside the imagination loop** — it corrupts kernel dispatch under autograd.grad; a regression test in `tests/test_mechanics_primitives.py` guards this.

**Training and loss (`src/train/`, `src/models/losses.py`, `src/models/mechanics/losses.py`)** — offline only. The RSSM composite loss combines KL (balanced + free-nats), posterior reconstruction, **foreground-masked** reconstruction (background-subtracted saliency mask), dynamic-mask reconstruction, transition reconstruction (decoded prior features at `t` targeted against `obs_{t+1}`), foreground-masked transition reconstruction, **imagination reconstruction** (decoded prior features over `K_ctx + K_imag` horizon), optional reward prediction, and BYOL-style latent consistency. The mechanics loss uses a three-branch balanced KL with asymmetric free-nats (tight on `q`/`qdot`, loose on `z`) plus a finite-difference smoothness prior that ties the encoder's two mechanics heads into a derivative relationship. Each term has its own CLI weight flag. Both trainers use AdamW with linear warmup + cosine LR, write per-epoch and `latest.pt` checkpoints plus JSONL `history.jsonl`, support `--auto-resume`, and log a `val_open_loop_fg_mse_h*` probe every N steps so the K1 metric is observable live. **Keep JSONL history machine-readable** — terminal formatting (Rich colors, timestamps) must not change the schema.

**Eval (`src/eval/`)** — loads a checkpoint and reports scalar metrics plus videos/contact sheets. Three prediction modes: posterior reconstruction at `t`, one-step prior prediction for `t`, and open-loop (posterior for `K` warmup steps, then prior-only rollout). Uses Gaussian means (deterministic) so a fixed seed gives reproducible metrics. `eval-mechanics` also runs a held-out OLS linear probe from `(encoded_q, encoded_qdot)` onto `(physics_qpos, physics_qvel)` when the dataset carries them (the K2 gate) and a learned-energy-drift diagnostic (for H3). `eval-control` drives the CEM/MPC planner against a live dm_control env for in-distribution and OOD (mass / length / damping / visual) settings. If a checkpoint was trained with a crop, eval uses that crop unless overridden with `--crop-mode`.

## Critical Conventions

- **Transition alignment is `obs_t --action_t--> obs_{t+1}`.** Batches expose both `observations[:, t]` and `next_observations[:, t]`; `next_observations[:, t]` is the target for transition/prior decoding and foreground-transition losses. Off-by-one here silently masquerades as weak dynamics. The mechanics imagination contract follows the same convention: `imagined[:, k]` targets `next_observations[:, K_ctx - 1 + k]`.
- Dataloader batch shapes: `observations`/`next_observations` `[B, T, 3, 84, 84]` float32 in `[0, 1]`; `actions` `[B, T, A]`; `rewards` `[B, T]`; `dones` `[B, T]` bool; optional `physics_qpos`/`physics_qvel` `[B, T, D_q]` float64 (present when the dataset was collected after Phase 1.1, consumed only by eval diagnostics — **never** by a training loss). Keep shape comments at RSSM / mechanics rollout sites.
- Don't change the `.npz` storage format or the `N + 1 images / N transitions` alignment without an explicit task. Optional physics fields are backwards-compatible — loaders return `None` for them when absent.
- Do not make research-performance claims in public docs unless backed by code, tests, or logged experiments.
- When adding a trainable model, add a document under `docs/models/` covering purpose, architecture/parameter count, objective, optimizer, and relation to other models; update it whenever the model/loss/optimizer/default config changes.
- Avoid global state in data, models, and training code; prefer small modules and explicit dataclasses.
- On Apple MPS, avoid `torch.linalg.solve` and `torch.autograd.functional.jvp` inside autograd.grad-with-create_graph loops — they corrupt kernel dispatch. Use `torch.linalg.inv` + einsum and a per-dim `autograd.grad` loop instead. Regression tests pin this.
- Checkpoints carry `model_kind` ("rssm" or "mechanics"); each trainer's `load_checkpoint` refuses mismatched kinds so RSSM weights cannot be silently loaded into a mechanics model or vice versa.
