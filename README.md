# Mechanics World Models

Research code for visual model-based reinforcement learning with mechanical
structure in the latent dynamics.

The question behind this repository is simple: can a world model that knows
something about mechanics generalize better than a generic recurrent latent model
when the physical system or the visual scene changes? In this project, the target
is not just better pixel prediction on the training distribution. The target is
control under shifted mass, length, friction, camera pose, background, and
lighting.

The full research plan is in [`docs/research_spec.md`](docs/research_spec.md).
Trainable model notes live in [`docs/models/`](docs/models/), starting with the
current [minimal RSSM visual world model](docs/models/rssm_visual_world_model.md).
The current baseline status and the foreground-reconstruction debugging problem
are summarized in [`docs/current_status.md`](docs/current_status.md).

## Current Status

This repository is still early. The current code builds the unstructured baseline
needed before adding the mechanics prior:

- random-policy pixel data collection from DeepMind Control `cartpole-swingup`
- one-episode-per-file `.npz` storage with 84x84 RGB observations
- a PyTorch sequence dataset for recurrent world-model training
- a compact RSSM-style visual world model
- an offline trainer with JSONL loss history, checkpoints, loss plotting, and
  an opt-in transition-aligned latent consistency objective
- small tests for the environment wrapper, dataset layer, model shapes, losses,
  and training-history plotting

The next stage is the structured model: a factored latent state with mechanical
coordinates, learned Lagrangian dynamics, Rayleigh dissipation, nuisance factors,
and MPC evaluation under physical and visual distribution shift.

## Why This Exists

Standard visual world models can learn useful latent dynamics, but their latent
state is usually unconstrained. That is convenient, and it often works
in-distribution. It is less clear what happens when the cart is heavier, the pole
is longer, damping changes, or the camera and background shift.

The working hypothesis is that the right mechanical prior can constrain
extrapolation. A learned Lagrangian-plus-dissipation transition should not make a
pixel model magically robust, but it gives the model a different failure mode:
the latent transition is pushed toward physically plausible trajectories instead
of arbitrary recurrent dynamics.

This repo starts with cartpole because the system is small enough to debug
carefully. The plan then moves to acrobot and, if the results justify it, more
complex DeepMind Control tasks.

## Repository Layout

```text
src/envs/     dm_control pixel environment wrapper
src/data/     NPZ trajectory storage, sequence dataset, dataset stats
src/models/   encoder, decoder, RSSM, world-model loss
src/train/    offline RSSM training entry point
src/viz/      dataset previews and training-history plots
tests/        lightweight pytest suite
docs/         research notes and development instructions
```

## Try The Baseline

For setup, data collection, training commands, and debugging notes, start with
[`docs/development.md`](docs/development.md).

The short version is:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Collect a small random cartpole dataset:

```bash
.venv/bin/collect-cartpole \
  --output-dir data/cartpole-swingup-random \
  --total-frames 1000 \
  --seed 0 \
  --action-repeat 2 \
  --image-size 84
```

Train the RSSM baseline:

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random \
  --checkpoint-dir checkpoints/rssm-cartpole \
  --sequence-length 16 \
  --batch-size 32 \
  --epochs 10 \
  --device auto
```

For the current foreground/debugging experiment, keep the decoder for
diagnostics but downweight full-frame pixel MSE, add a posterior foreground
reconstruction term, and train the transition-prior decoder directly against
`obs_{t+1}` with a foreground-weighted next-frame term:

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random \
  --checkpoint-dir checkpoints/rssm-cartpole-foreground-transition \
  --sequence-length 16 \
  --batch-size 32 \
  --epochs 10 \
  --reconstruction-weight 0.05 \
  --foreground-reconstruction-weight 2.0 \
  --foreground-mask-floor 0.02 \
  --foreground-mask-kernel-size 7 \
  --transition-reconstruction-weight 0.0 \
  --foreground-transition-reconstruction-weight 2.0 \
  --dynamic-reconstruction-weight 0.0 \
  --latent-consistency-weight 0.0 \
  --device auto
```

Plot the loss history:

```bash
.venv/bin/plot-training-history \
  --history-path checkpoints/rssm-cartpole/history.jsonl \
  --output-path checkpoints/rssm-cartpole/loss_history.png
```

Evaluate reconstruction and open-loop prediction quality:

```bash
.venv/bin/eval-rssm \
  --checkpoint-path checkpoints/rssm-cartpole/latest.pt \
  --dataset-dir data/cartpole-swingup-random \
  --output-dir eval/rssm-cartpole \
  --sequence-length 16 \
  --warmup-length 5 \
  --horizons 1 5 10
```

Debug missing foreground reconstructions:

```bash
.venv/bin/preview-crop \
  --dataset-dir data/cartpole-swingup-random \
  --output-path eval/crop-preview.png \
  --crop-mode center \
  --crop-height 64 \
  --crop-width 64

.venv/bin/debug-reconstruction \
  --checkpoint-path checkpoints/rssm-cartpole/latest.pt \
  --dataset-dir data/cartpole-swingup-random \
  --output-dir eval/debug-reconstruction \
  --sequence-length 16 \
  --timesteps 0 5 10 15

.venv/bin/overfit-reconstruction \
  --dataset-dir data/cartpole-swingup-random \
  --output-dir eval/overfit-one-batch \
  --sequence-length 16 \
  --mode batch \
  --batch-size 4 \
  --steps 500 \
  --save-every 100 \
  --kl-weight 0.0

.venv/bin/dataset-image-diagnostics \
  --dataset-dir data/cartpole-swingup-random \
  --output-dir eval/dataset-images \
  --max-frames 10000
```

The same crop flags can be passed to `train-rssm` and `eval-rssm`. Evaluation
falls back to the checkpoint crop config when no crop flags are provided.

On Linux or other headless machines, MuJoCo may need an explicit offscreen
backend:

```bash
MUJOCO_GL=egl .venv/bin/collect-cartpole --output-dir data/cartpole-swingup-random --total-frames 1000
```

## Data Format

Each collected episode is stored as one compressed NumPy archive:

```text
episode_000000.npz
episode_000001.npz
...
```

Each archive stores `N + 1` images and `N` transition records so sequence models
can read `(obs_t, action_t, obs_{t+1})` without crossing episode boundaries.
Arrays include `images`, `actions`, `rewards`, `discounts`, `dones`, and
`step_indices`, plus scalar metadata such as `env_name`, `seed`, `episode_id`,
`action_repeat`, and `image_size`.

## Roadmap

- add the mechanics-structured latent dynamics model
- add reward prediction and MPC evaluation
- add physical parameter shifts for cartpole and acrobot
- add visual nuisance shifts for camera/background/lighting
- compare against the RSSM baseline under the same data and compute budget
- keep the implementation inspectable enough to debug failure cases directly

Actor-critic training, replay APIs, zarr storage, and distributed collection are
deliberately out of scope until the small baseline is correct.
