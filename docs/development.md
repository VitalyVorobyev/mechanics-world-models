# Development Guide

This page keeps the operational details out of the public README: environment
setup, collection commands, dataset inspection, training, plotting, and tests.

## Setup

Use Python 3.12 in the local `.venv`:

```bash
uv venv --python 3.12 .venv
```

Install the package and development dependencies:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

The project targets Python `>=3.10,<3.13`. In practice, Python 3.12 is the safer
default today because `dm-control` depends on packages that can be awkward to
build on Python 3.13 on macOS ARM.

On Linux or other headless machines, MuJoCo may need an explicit offscreen
backend:

```bash
MUJOCO_GL=egl .venv/bin/collect-cartpole --output-dir data/cartpole-swingup-random --total-frames 1000
```

## Collect Cartpole Pixel Data

Collect a small smoke-test dataset:

```bash
.venv/bin/collect-cartpole \
  --output-dir data/cartpole-swingup-random \
  --total-frames 1000 \
  --seed 0 \
  --action-repeat 2 \
  --image-size 84
```

Collect a more useful first training dataset:

```bash
.venv/bin/collect-cartpole \
  --output-dir data/cartpole-swingup-random-100k \
  --total-frames 100000 \
  --seed 0 \
  --action-repeat 2 \
  --image-size 84
```

In this collector, `--total-frames` means stored transitions, not raw MuJoCo
physics steps. Each stored transition applies one action for `action_repeat`
dm_control steps. Each episode file stores `N + 1` images and `N` transition
arrays.

For default `cartpole-swingup`, one full dm_control episode is about 1000 action
steps. With `action_repeat=2`, that becomes about 500 stored transitions per full
episode, so `100000` stored frames is roughly 200 episode files. The final
episode can be shorter if the frame budget ends mid-episode.

## Inspect A Dataset

Generate a contact sheet and MP4 preview:

```bash
.venv/bin/preview-dataset \
  --dataset-dir data/cartpole-swingup-random \
  --output-dir previews/cartpole-swingup-random
```

Report sequence-dataset statistics:

```bash
.venv/bin/dataset-stats \
  --dataset-dir data/cartpole-swingup-random \
  --sequence-length 16
```

## Use The Sequence Dataset

```python
from pathlib import Path

from data import make_train_val_datasets

train_dataset, val_dataset = make_train_val_datasets(
    dataset_dir=Path("data/cartpole-swingup-random"),
    sequence_length=16,
    val_fraction=0.1,
    seed=0,
)

sample = train_dataset[0]
observations = sample["observations"]  # [T, C, H, W], float32 in [0, 1]
actions = sample["actions"]            # [T, A], float32
rewards = sample["rewards"]            # [T], float32
dones = sample["dones"]                # [T], bool
```

Splits are deterministic and happen by whole episode files, never by frames.
Episodes shorter than `sequence_length` are skipped; if a split has no usable
episodes, dataset construction raises a clear `ValueError`.

## Train The RSSM Baseline

Train a normal baseline run:

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random-100k \
  --checkpoint-dir checkpoints/rssm-cartpole-random-100k \
  --sequence-length 32 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --epochs 30 \
  --hidden-size 128 \
  --latent-size 32 \
  --embedding-size 256 \
  --kl-weight 1.0 \
  --val-fraction 0.1 \
  --seed 0 \
  --device auto \
  --log-every-steps 100
```

Run a tiny overfit/debug job on a few whole episodes:

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random \
  --checkpoint-dir checkpoints/rssm-overfit \
  --sequence-length 16 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --epochs 50 \
  --hidden-size 128 \
  --latent-size 32 \
  --embedding-size 256 \
  --kl-weight 1.0 \
  --device auto \
  --log-every-steps 10 \
  --max-train-episodes 4 \
  --max-val-episodes 2
```

Do not use `--max-train-episodes` for a real run. It exists to make small
overfit tests fast.

The trainer writes:

```text
checkpoints/<run-name>/latest.pt
checkpoints/<run-name>/epoch_0001.pt
checkpoints/<run-name>/history.jsonl
```

Each checkpoint contains the model state, optimizer state, config, epoch,
`global_step`, latest metrics, and the history path.

Resume from the latest checkpoint:

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random-100k \
  --checkpoint-dir checkpoints/rssm-cartpole-random-100k \
  --resume-from checkpoints/rssm-cartpole-random-100k/latest.pt
```

## Plot Training History

```bash
.venv/bin/plot-training-history \
  --history-path checkpoints/rssm-cartpole-random-100k/history.jsonl \
  --output-path checkpoints/rssm-cartpole-random-100k/loss_history.png
```

The plotter reads the JSONL history and writes a PNG with `total_loss`,
`reconstruction_loss`, `kl_loss`, and `weighted_kl_loss` against optimizer
`global_step`.

## Test

```bash
.venv/bin/python -m pytest
```

The dm_control pixel test skips cleanly if local offscreen rendering is not
available.

## Current Extension Points

The current RSSM is intentionally small. The next research work should add:

- a factored latent state `(q, qdot, z_nuisance)`
- learned Lagrangian dynamics and Rayleigh dissipation
- reward prediction separated from reconstruction loss
- MPC evaluation under physical and visual distribution shift
- OOD dataset/evaluation harnesses for cartpole and acrobot

Replay buffer APIs, zarr storage, distributed collection, actor-critic training,
and augmentation pipelines are useful later, but they should wait until the
structured dynamics baseline is correct.
