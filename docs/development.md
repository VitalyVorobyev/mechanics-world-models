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

Physics state (`qpos`, `qvel`, `physics_params`) is serialized
automatically into each NPZ. The dataloader exposes these as
`physics_qpos` / `physics_qvel` samples, consumed only by
`eval-mechanics` diagnostics (linear probe + energy drift) — never
by a training loss.

Use `--physics-scale KEY=SCALE` (repeatable) to perturb simulator
parameters for OOD collection, e.g. `--physics-scale body_mass_2=0.5`.

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
observations = sample["observations"]            # [T, C, H, W], obs_t, float32 in [0, 1]
next_observations = sample["next_observations"]  # [T, C, H, W], obs_{t+1}, float32 in [0, 1]
actions = sample["actions"]                      # [T, A], action_t, float32
rewards = sample["rewards"]                      # [T], float32
dones = sample["dones"]                          # [T], bool
```

Splits are deterministic and happen by whole episode files, never by frames.
Episodes shorter than `sequence_length` are skipped; if a split has no usable
episodes, dataset construction raises a clear `ValueError`. The sample alignment
is `observations[t] --actions[t]--> next_observations[t]`.

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
  --reconstruction-weight 1.0 \
  --foreground-reconstruction-weight 0.0 \
  --dynamic-reconstruction-weight 0.0 \
  --latent-consistency-weight 0.0 \
  --val-fraction 0.1 \
  --seed 0 \
  --device auto \
  --log-every-steps 100
```

Preview the foreground mask before trusting a weighted run:

```bash
.venv/bin/foreground-mask-preview \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-path eval/foreground-mask-preview.png \
  --sequence-length 32 \
  --batch-size 32 \
  --foreground-mask-floor 0.02 \
  --foreground-mask-kernel-size 7
```

Run the current RSSM with opt-in foreground reconstruction and foreground
transition reconstruction objectives. This keeps the decoder for diagnostics,
heavily downweights full-frame pixel MSE, trains posterior reconstruction on
`obs_t`, and trains the transition-prior decoder directly against
`obs_{t+1}`:

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random-100k \
  --checkpoint-dir checkpoints/rssm-cartpole-foreground-transition \
  --sequence-length 32 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --epochs 30 \
  --hidden-size 128 \
  --latent-size 32 \
  --embedding-size 256 \
  --kl-weight 0.1 \
  --reconstruction-weight 0.05 \
  --foreground-reconstruction-weight 2.0 \
  --foreground-mask-floor 0.02 \
  --foreground-mask-kernel-size 7 \
  --transition-reconstruction-weight 0.0 \
  --foreground-transition-reconstruction-weight 2.0 \
  --dynamic-reconstruction-weight 0.0 \
  --latent-consistency-weight 0.0 \
  --val-fraction 0.1 \
  --seed 0 \
  --device auto \
  --log-every-steps 100
```

This is still the RSSM baseline with auxiliary foreground, transition,
dynamic-reconstruction, and BYOL-style next-latent loss options, not the later
reconstruction-free B4/MuDreamer-style contrastive baseline. The temporal
dynamic reconstruction term remains available for diagnostics, but the
recommended foreground-transition run keeps it disabled.

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

To test whether tighter framing helps the baseline, preview a deterministic
center crop first:

```bash
.venv/bin/preview-crop \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-path eval/crop-preview-center64.png \
  --crop-mode center \
  --crop-height 64 \
  --crop-width 64
```

Then train with the same crop. The dataset loader crops loaded `uint8` frames
and resizes them back to the original model input size, so observations remain
`[B, T, 3, 84, 84]`:

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random-100k \
  --checkpoint-dir checkpoints/rssm-cartpole-center64 \
  --sequence-length 16 \
  --batch-size 32 \
  --epochs 20 \
  --kl-weight 1.0 \
  --crop-mode center \
  --crop-height 64 \
  --crop-width 64
```

Use an explicit rectangle when center crop is not the right framing:

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random-100k \
  --checkpoint-dir checkpoints/rssm-cartpole-rect \
  --sequence-length 16 \
  --crop-mode rect \
  --crop-top 12 \
  --crop-left 12 \
  --crop-height 60 \
  --crop-width 60
```

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

## Train The Mechanics Model

Run the current cartpole config (the one that emerged from the v2/v3/v4/
diag debugging documented in [`current_status.md`](current_status.md)).
`scripts/train_mechanics.sh` holds this verbatim and is what the project
runs by default:

```bash
.venv/bin/train-mechanics \
  --dataset-dir data/cartpole-swingup-random-100k \
  --checkpoint-dir checkpoints/mechanics-phase3-finitediff \
  --sequence-length 32 --batch-size 32 --epochs 10 \
  --q-dim 2 --nuisance-dim 4 --dt 0.01 \
  --imagination-context-steps 4 --imagination-horizon 12 \
  --foreground-imagination-reconstruction-weight 2.0 \
  --kl-weight 0.1 \
  --kl-free-nats-mech 0.5 --kl-free-nats-nuisance 3.0 \
  --nuisance-min-std 1.0 \
  --learning-rate 1e-4 --warmup-steps 2000 \
  --log-every-steps 50 \
  --val-open-loop-every-steps 100 \
  --val-open-loop-horizons 1,5,10,20
```

Notes:

- `q̇` is derived from `q` by finite differencing inside the encoder's
  forward pass (central diff interior, forward / backward at boundaries).
  No `--smoothness-weight` flag — the kinematic constraint is structural,
  not a soft loss term.
- `--nuisance-dim 4` + `--nuisance-min-std 1.0` together control the
  nuisance channel's capacity × KL-stiffness. Larger values on either
  triggered a z-channel KL phase transition at step ~300 in earlier
  iterations; see `current_status.md` for the trace.
- `--log-every-steps 50` still keeps the z-diagnostic metrics
  (`z_post_mean_rms`, `kl_z_t0_mean`, `kl_z_transition_mean`, ...)
  dense enough to catch the earlier failure modes (the v2/v3 z blow-up
  took ~200 steps to manifest) without flooding the terminal. Drop back
  to `10` if you need per-step visibility for a new debug round.

Recollect the training dataset if you need physics state for K2 / H3
diagnostics (the v1 dataset at `data/cartpole-swingup-random-100k` was
collected before automatic physics-state serialization):

```bash
bash scripts/collect_cartpole.sh
```

The script writes to `data/cartpole-swingup-random-100k-v2` with the same
seed / image / action-repeat parameters as v1 — image trajectories are
bit-identical, only the extra `qpos` / `qvel` arrays are new.

Evaluate (matches the same sequence-split and warmup as `eval-rssm` so the
two models are directly comparable):

```bash
.venv/bin/eval-mechanics \
  --checkpoint-path checkpoints/mechanics-phase3-finitediff/latest.pt \
  --dataset-dir data/cartpole-swingup-random-100k-v2 \
  --output-dir eval/mechanics-phase3-finitediff \
  --sequence-length 32 --warmup-length 5 --horizons 1 5 10
```

`metrics.json` additionally carries `linear_probe.q_to_qpos_r2_overall`,
`linear_probe.qdot_to_qvel_r2_overall` (K2 gate, threshold 0.8) and
`energy.mean_abs_drift`, `energy.mean_abs_drift_h10` (H3) when the dataset
carries physics state.

## Plot Training History

```bash
.venv/bin/plot-training-history \
  --history-path checkpoints/rssm-cartpole-random-100k/history.jsonl \
  --output-path checkpoints/rssm-cartpole-random-100k/loss_history.png
```

The plotter reads the JSONL history and writes a PNG with available requested
metrics, including `total_loss`, `reconstruction_loss`,
`weighted_reconstruction_loss`, `foreground_reconstruction_loss`,
`weighted_foreground_reconstruction_loss`, `transition_reconstruction_loss`,
`weighted_transition_reconstruction_loss`,
`foreground_transition_reconstruction_loss`,
`weighted_foreground_transition_reconstruction_loss`,
`dynamic_reconstruction_loss`, `weighted_dynamic_reconstruction_loss`,
`kl_loss`, `weighted_kl_loss`, `latent_consistency_loss`, and
`weighted_latent_consistency_loss` against optimizer `global_step`.

## Evaluate Reconstruction And Prediction

Run evaluation on the deterministic validation split:

```bash
.venv/bin/eval-rssm \
  --checkpoint-path checkpoints/rssm-cartpole-random-100k/latest.pt \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-dir eval/rssm-cartpole-random-100k \
  --sequence-length 32 \
  --batch-size 16 \
  --num-sequences 64 \
  --warmup-length 5 \
  --horizons 1 5 10 \
  --seed 0 \
  --num-visualizations 3
```

If the checkpoint was trained with a crop, evaluation uses that crop
configuration by default. Pass `--crop-mode none` to override it, or pass a new
`--crop-mode center|rect` configuration to test another crop.

Evaluation uses Gaussian means rather than stochastic latent samples, so repeated
runs with the same seed select the same sequences and produce the same scalar
metrics.

For checkpoints that include the latent predictor, evaluation also reports
`latent_consistency_loss` for the transition-aligned objective
`obs_t --action_t--> obs_{t+1}`. Older checkpoints load with a warning and skip
that metric.

Prediction modes:

- Reconstruction: infer posterior latents from the true observation embedding at
  each timestep, decode those posterior features, and compare to the true frame.
- One-step prediction: decode prior features from the same action-conditioned
  recurrent pass before the current observation embedding is used.
- Open-loop prediction: use posterior inference for the first `K` warmup steps,
  then ignore future observations and propagate latent dynamics with only the
  previous latent state, deterministic state, and action sequence.

The evaluator writes:

```text
eval/<run-name>/metrics.json
eval/<run-name>/sequence_0000.mp4
eval/<run-name>/sequence_0000_sheet.png
...
```

Videos show columns in this order: ground truth, posterior reconstruction, and
warmup/open-loop prediction. The contact sheet uses the same column order for
selected timesteps. Video labels mark the prediction panel as `pred ctx` during
the warmup/context frames and `pred open` after future observations are no longer
used. If motion is hard to see, lower the playback rate with `--video-fps 3`.

## Debug Missing Foreground Reconstructions

If reconstructions reproduce the static background while the cartpole disappears,
first separate evaluation bugs from loss/objective imbalance.

Save posterior-only reconstruction grids for one training batch and one
validation batch:

```bash
.venv/bin/debug-reconstruction \
  --checkpoint-path checkpoints/rssm-cartpole-random-100k/latest.pt \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-dir eval/debug-reconstruction \
  --sequence-length 16 \
  --timesteps 0 5 10 15
```

This writes:

```text
eval/debug-reconstruction/train_posterior_reconstruction_grid.png
eval/debug-reconstruction/val_posterior_reconstruction_grid.png
eval/debug-reconstruction/reconstruction_debug.json
```

The grid columns are target frame, posterior reconstruction, and absolute error
heatmap. The code logs that targets and reconstructions are both
`[B, T, 3, H, W]` tensors in `[0, 1]` and aligned at the same timestep.

Run a single-batch overfit probe:

```bash
.venv/bin/overfit-reconstruction \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-dir eval/overfit-one-batch \
  --sequence-length 16 \
  --mode batch \
  --batch-size 4 \
  --steps 500 \
  --save-every 100 \
  --kl-weight 0.0
```

Use `--mode sequence` to overfit one sequence. The utility writes grids such as
`reconstruction_step_000000.png`, `reconstruction_step_000100.png`, and
`reconstruction_step_000500.png`, plus `overfit_history.jsonl` and
`overfit_latest.pt`. Optimization uses the existing training loss; saved grids
use deterministic posterior means to remove sampling noise from the visual
diagnostic.

Check whether the dataset objective is dominated by static pixels:

```bash
.venv/bin/dataset-image-diagnostics \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-dir eval/dataset-images \
  --max-frames 10000
```

This writes `average_image.png`, `variance_image.png`,
`variance_heatmap.png`, and `dataset_image_stats.json`. A near-static average
image with very small foreground variance is a strong hint that plain pixel MSE
is rewarding background reconstruction more than cartpole dynamics.

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
