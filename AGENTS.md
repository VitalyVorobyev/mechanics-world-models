# Agent Notes

This is a Python research codebase for mechanics-structured visual world models
for visual model-based RL. The public README is now written for GitHub readers;
keep detailed setup, debugging, and day-to-day commands in `docs/development.md`.
Use `docs/research_spec.md` as the source of truth for the long-term research
direction.

## Project Scope

- Current baseline: random-policy cartpole pixel data, NPZ episode storage,
  PyTorch sequence datasets, and a compact RSSM-style visual world model.
- Research target: mechanics-structured latent dynamics with factored
  `(q, qdot, z_nuisance)`, learned Lagrangian dynamics, Rayleigh dissipation,
  and MPC evaluation under physical and visual distribution shift.
- Do not add actor-critic training, replay buffers, zarr storage, distributed
  collection, or augmentation pipelines unless a task explicitly asks for them.
- Do not make research-performance claims in public docs unless they are backed
  by code, tests, or logged experiments in the repo.

## Documentation

- Keep `README.md` public-facing: concise motivation, current status, quick
  start, data format, and roadmap.
- Keep operational details in `docs/development.md`: setup, collection, dataset
  stats, training, plotting, testing, and debugging commands.
- Keep research framing in `docs/research_spec.md`; do not silently rewrite the
  hypotheses, baselines, or kill criteria without an explicit task.
- Keep one clear, concise document in `docs/models/` for every trainable model.
  Each model document must cover purpose, architecture and parameter count,
  objective function, optimizer, and relation to other models. Add a document
  when introducing a new trainable model and update it when changing the model,
  loss, optimizer, or default training configuration.

## Code Organization

- Use the existing `src/` layout and keep modules small.
- Put environment wrappers under `src/envs/`.
- Put trajectory storage, indexing, sequence datasets, and dataset stats under
  `src/data/`.
- Put encoders, decoders, RSSM dynamics, world-model wrappers, and losses under
  `src/models/`.
- Put offline training entry points under `src/train/`.
- Put checkpoint evaluation and scalar metric code under `src/eval/`.
- Put previews and plotting utilities under `src/viz/`.

## Data And Tensor Conventions

- Prefer simple `.npz` episode files unless a task explicitly changes the
  storage format.
- Preserve the current transition alignment: saved trajectories contain `N + 1`
  images and `N` action/reward/discount/done entries.
- Sequence dataset samples use:
  - `observations`: `[T, C, H, W]`, float32 in `[0, 1]`
  - `next_observations`: `[T, C, H, W]`, float32 in `[0, 1]`, aligned as
    `obs_t --action_t--> obs_{t+1}`
  - `actions`: `[T, A]`, float32
  - `rewards`: `[T]`, float32
  - `dones`: `[T]`, bool
- Dataloader batches use:
  - `observations`: `[B, T, 3, 84, 84]`
  - `next_observations`: `[B, T, 3, 84, 84]`
  - `actions`: `[B, T, A]`
  - `rewards`: `[B, T]`
  - `dones`: `[B, T]`
- Please keep tensor shapes explicit in comments at key interfaces, especially
  in RSSM rollout code.

## Implementation Conventions

- Use `pathlib`, type hints, dataclasses where they improve clarity, and concise
  docstrings for public utilities.
- Prefer small, readable modules over broad abstractions.
- Keep JSONL training history machine-readable; terminal-only formatting such as
  Rich colors or timestamps should not change the history schema.
- Avoid global state in data collection, datasets, models, and training code.

## Testing

- Add or update pytest tests for data-format, shape, dtype, deterministic split,
  logging/history, and model/loss changes.
- Keep tests synthetic/offline when possible so they do not require MuJoCo
  rendering.
- The dm_control render test may skip on machines without working offscreen
  rendering.
- Run `.venv/bin/python -m pytest` after substantive code changes.

## Environment

- Use `.venv` with Python `>=3.10,<3.13`; Python 3.12 is the current practical
  target because `dm-control` pulls dependencies that may not install cleanly on
  Python 3.13 without extra build tools.
- Install with:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```
