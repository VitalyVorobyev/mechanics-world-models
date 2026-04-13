# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Intent

Research code for mechanics-structured visual world models for model-based RL. The current code is an unstructured RSSM baseline on `dm_control cartpole-swingup` pixels; the research target is a factored `(q, qdot, z_nuisance)` latent with learned Lagrangian dynamics, Rayleigh dissipation, and MPC evaluation under physical/visual distribution shift. **Do not add actor-critic, replay buffers, zarr, distributed collection, or augmentation pipelines unless a task explicitly asks for them.**

Authoritative docs:
- `docs/research_spec.md` — long-term research direction; do not silently rewrite hypotheses, baselines, or kill criteria.
- `docs/development.md` — setup, collection, training, evaluation, debugging commands (prefer this over README for day-to-day).
- `docs/current_status.md` — current blocker: posterior reconstruction works, but transition-prior decoding is weak and background-dominated. Foreground-weighted transition reconstruction is the active experiment.
- `docs/models/*.md` — one concise document per trainable model; update it when the model, loss, optimizer, or default training config changes. Add a new doc when introducing a new trainable model.
- `AGENTS.md` — authoritative for agent conventions; this file summarizes.

## Environment and Commands

Python `>=3.10,<3.13`; 3.12 is the practical target (`dm-control` builds are awkward on 3.13). Use `.venv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Entry points are installed as console scripts into `.venv/bin/` (see `[project.scripts]` in `pyproject.toml`): `collect-cartpole`, `dataset-stats`, `train-rssm`, `eval-rssm`, `plot-training-history`, `preview-dataset`, `preview-crop`, `debug-reconstruction`, `overfit-reconstruction`, `dataset-image-diagnostics`, `foreground-mask-preview`.

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

**Models (`src/models/`)** — `VisualWorldModel` composes:
- `ConvEncoder`: `[B, T, 3, 84, 84] -> [B, T, E]`
- `RSSM`: GRU-backed action-conditioned dynamics; produces deterministic `h_t` (size `H`), stochastic `z_t` (size `Z`), prior `p(z_t | h_t)`, posterior `q(z_t | h_t, embed_t)`. `RSSMOutput.features = cat([h, z], -1)`.
- `ImageDecoder`: `[B, T, H + Z] -> [B, T, 3, 84, 84]` (sigmoid-normalized).
- `latent_predictor`: MLP from next prior features to encoder embedding space, used only when `next_observations` is present.

**Training and loss (`src/train/`, `src/models/losses.py`)** — offline only. The composite loss combines KL, posterior reconstruction, **foreground-masked** reconstruction (background-subtracted saliency mask), dynamic-mask reconstruction, transition reconstruction (decoded prior features at `t` targeted against `obs_{t+1}`), foreground-masked transition reconstruction, and BYOL-style latent consistency (stop-gradient encoder target). Each term has its own CLI weight flag; default weights are tuned to the current debugging experiment. Trainer writes per-epoch and `latest.pt` checkpoints plus JSONL `history.jsonl`. **Keep JSONL history machine-readable** — terminal formatting (Rich colors, timestamps) must not change the schema.

**Eval (`src/eval/`)** — loads a checkpoint and reports scalar metrics plus videos/contact sheets. Three prediction modes: posterior reconstruction at `t`, one-step prior prediction for `t`, and open-loop (posterior for `K` warmup steps, then prior-only rollout). Uses Gaussian means (deterministic) so a fixed seed gives reproducible metrics. If a checkpoint was trained with a crop, eval uses that crop unless overridden with `--crop-mode`.

## Critical Conventions

- **Transition alignment is `obs_t --action_t--> obs_{t+1}`.** Batches expose both `observations[:, t]` and `next_observations[:, t]`; `next_observations[:, t]` is the target for transition/prior decoding and foreground-transition losses. Off-by-one here silently masquerades as weak dynamics (see `docs/current_status.md`).
- Dataloader batch shapes: `observations`/`next_observations` `[B, T, 3, 84, 84]` float32 in `[0, 1]`; `actions` `[B, T, A]`; `rewards` `[B, T]`; `dones` `[B, T]` bool. Keep shape comments at RSSM rollout sites.
- Don't change the `.npz` storage format or the `N + 1 images / N transitions` alignment without an explicit task.
- Do not make research-performance claims in public docs unless backed by code, tests, or logged experiments.
- When adding a trainable model, add a document under `docs/models/` covering purpose, architecture/parameter count, objective, optimizer, and relation to other models; update it whenever the model/loss/optimizer/default config changes.
- Avoid global state in data, models, and training code; prefer small modules and explicit dataclasses.
