# Agent Notes

This is a small Python research codebase for visual model-based RL on DeepMind
Control / MuJoCo environments. Keep changes focused and avoid adding training,
model, replay-buffer, or augmentation code unless the task explicitly asks for it.

## Project Conventions

- Use the existing `src/` layout and keep modules small.
- Put environment wrappers under `src/envs/`, data loading/storage under
  `src/data/`, and visualization utilities under `src/viz/`.
- Use `pathlib`, type hints, dataclasses where they improve clarity, and concise
  docstrings for public utilities.
- Prefer simple `.npz` episode files unless a task explicitly changes the
  storage format.
- Preserve the current transition alignment: saved trajectories contain `N+1`
  images and `N` action/reward/discount/done entries.
- Please keep tensor shapes explicit in comments at key interfaces, especially
  in RSSM rollout code.

## Testing

- Add or update pytest tests for data-format, shape, dtype, and deterministic
  behavior changes.
- Keep tests synthetic/offline when possible so they do not require MuJoCo
  rendering.
- The dm_control render test may skip on machines without working offscreen
  rendering.

## Environment

- Use `.venv` with Python `>=3.10,<3.13`; Python 3.12 is the current practical
  target because `dm-control` pulls dependencies that may not install cleanly on
  Python 3.13 without extra build tools.
- Install with:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```
