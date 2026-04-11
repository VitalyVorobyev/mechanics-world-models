# Mechanics World Models

Minimal research code for mechanics-structured visual world models, starting with
pixel observations from DeepMind Control cartpole-swingup.

## Cartpole pixel data collection

Set up `.venv` with Python 3.12 and install the package in editable mode:

```bash
uv venv --python 3.12 --clear .venv
```

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

The project targets Python `>=3.10,<3.13` because `dm-control` currently pulls
`labmaze`, which does not install cleanly from source on Python 3.13 without
Bazel on macOS ARM.

Collect a random-policy dataset from `cartpole-swingup`:

```bash
.venv/bin/collect-cartpole \
  --output-dir data/cartpole-swingup-random \
  --total-frames 1000 \
  --seed 0 \
  --action-repeat 2 \
  --image-size 84
```

Generate a quick visual sanity check:

```bash
.venv/bin/preview-dataset \
  --dataset-dir data/cartpole-swingup-random \
  --output-dir previews/cartpole-swingup-random
```

The preview command writes `contact_sheet.png` and `preview.mp4`.

On Linux or other headless machines, MuJoCo may need an explicit offscreen backend:

```bash
MUJOCO_GL=egl .venv/bin/collect-cartpole --output-dir data/cartpole-swingup-random --total-frames 1000
```

### Dataset format

Each trajectory is stored as one compressed NumPy archive:

```text
episode_000000.npz
episode_000001.npz
...
```

Each archive stores `N+1` RGB images and `N` transition records so future training
can read `(obs_t, action_t, obs_{t+1})` directly. Arrays and metadata include:
`env_name`, `seed`, `episode_id`, `action_repeat`, `image_size`, `images`,
`step_indices`, `rewards`, `discounts`, `dones`, and `actions`.

### Design choices and extension points

This milestone uses `.npz` because it is easy to inspect, easy to load with NumPy,
and sufficient for small random-rollout baselines. The environment wrapper keeps
dm_control rendering isolated from storage code, and the trajectory dataclass is
the narrow handoff point for future PyTorch datasets or replay buffers.

Future extensions can add a replay sampler, train/validation manifests, policy
collectors, zarr storage for larger runs, frame augmentations, and multi-env
collection without changing the basic transition alignment.

## What I would postpone

Training, replay buffer APIs, zarr/chunked storage, multi-env collection, policy
collectors, distributed collection, and augmentation pipelines are intentionally
out of scope for this first milestone.
