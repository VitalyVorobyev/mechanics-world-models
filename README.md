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
Trainable model notes live in [`docs/models/`](docs/models/):
- [minimal RSSM visual world model](docs/models/rssm_visual_world_model.md)
- [mechanics-structured visual world model](docs/models/mechanics_world_model.md)

The live status of the research — what works, what the open problem is, and
the next experiment to run — is tracked in
[`docs/current_status.md`](docs/current_status.md).

## Current Status

The repository now contains both the unstructured baseline and the factored
mechanics model, plus the scaffolding needed to compare them under physical
and visual distribution shift:

- random-policy pixel data collection from DeepMind Control `cartpole-swingup`,
  with automatic `qpos`/`qvel`/`physics_params` serialization for evaluation
  diagnostics (never consumed by a training loss)
- one-episode-per-file `.npz` storage with 84x84 RGB observations
- a PyTorch sequence dataset for recurrent world-model training
- **two trainable world models**: a compact RSSM-style baseline (B1) and a
  factored `(q, q̇, z_nuisance)` model with a learned Lagrangian
  (Cholesky-PD mass matrix, learned potential, PSD Rayleigh dissipation,
  learned actuation), a semi-implicit symplectic integrator, and a hard
  kinematic constraint `q̇ = d/dt(q)` enforced by finite differencing `q`
  in the encoder's forward pass (LNN/HNN-style)
- offline trainers (`train-rssm`, `train-mechanics`) with JSONL loss history,
  checkpoints, AdamW + cosine LR, auto-resume, NaN/Inf skip guard, and live
  open-loop FG-MSE validation probes
- evaluators (`eval-rssm`, `eval-mechanics`) producing per-horizon MSE
  against a copy-last-frame baseline, videos, and contact sheets, plus
  linear and MLP state probes onto the simulator's ground-truth state (when
  available); `eval-mechanics` also reports a learned-energy drift
- a CEM / MPC planner (`eval-control`, RSSM checkpoints only for now),
  including physics-perturbation and visual-distractor wrappers
- a pytest suite (139 tests at time of writing) covering environment,
  dataset, model shapes, losses, finite-diff kinematics, training-history,
  MPS regression guards, probes, and end-to-end evaluation smoke tests

Phase 0 (RSSM stable) is closed. The October 2026 review of the mechanics
model found that, on the April checkpoints, the pose is carried by the
nuisance channel rather than by `q`, and the RSSM baseline predicts better
on identical sequences. The project is now gated on one experiment that
forces the pose through `q` and compares Lagrangian against MLP dynamics;
see `docs/current_status.md` for the findings, the gate, and the stop
criterion.

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
src/envs/            dm_control pixel environment wrapper + physics-perturbation helpers
src/data/            NPZ trajectory storage (+ optional qpos/qvel/physics_params), sequence dataset, dataset stats
src/models/          shared encoder / decoder / losses
src/models/mechanics Lagrangian primitives, symplectic integrator, factored (q, qdot, z) transition and world model, mechanics-specific loss
src/models/contrastive (planned) reconstruction-free B4 ablation
src/planning/        CEM / MPC planner that consumes either trainable world model
src/train/           offline train-rssm and train-mechanics entry points
src/eval/            eval-rssm, eval-mechanics, eval-control, linear-probe and energy diagnostics
src/viz/             dataset previews, prediction videos/contact sheets, training-history plots
scripts/             convenience shell wrappers + scripts/bench_mechanics_step.py perf harness
tests/               pytest suite (unit + end-to-end smoke)
docs/                research spec, current status, development guide, per-model docs
```

## Try The Baseline

For setup, data collection, training commands, and debugging notes, start with
[`docs/development.md`](docs/development.md).

The short version is:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Collect a small random cartpole dataset. Physics state (`qpos`, `qvel`,
`physics_params`) is serialized automatically — no flag needed. `--physics-
scale KEY=SCALE` is available for OOD perturbation experiments:

```bash
.venv/bin/collect-cartpole \
  --output-dir data/cartpole-swingup-random \
  --total-frames 1000 \
  --seed 0 \
  --action-repeat 2 \
  --image-size 84
```

Train the RSSM baseline (the Phase-0 configuration uses imagination-rollout
loss, KL balancing with free-nats, AdamW + cosine LR, and a live open-loop
FG-MSE probe):

```bash
.venv/bin/train-rssm \
  --dataset-dir data/cartpole-swingup-random \
  --checkpoint-dir checkpoints/rssm-cartpole \
  --sequence-length 32 \
  --batch-size 32 \
  --epochs 10 \
  --foreground-reconstruction-weight 1.0 \
  --foreground-imagination-reconstruction-weight 2.0 \
  --imagination-context-steps 4 --imagination-horizon 12 \
  --kl-weight 1.0 --kl-free-nats 0.5 \
  --val-open-loop-every-steps 500 \
  --val-open-loop-horizons 1,5,10,20 \
  --device auto
```

Train the mechanics model. The recommended cartpole config is the one that
emerged from the v2/v3/v4/diag debugging (full story in
`docs/current_status.md`):

```bash
.venv/bin/train-mechanics \
  --dataset-dir data/cartpole-swingup-random \
  --checkpoint-dir checkpoints/mechanics-cartpole \
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

`q̇` is derived from `q` by central differencing in the encoder's forward
pass — there is no `--smoothness-weight` flag anymore and no soft
kinematic prior to tune. See the model doc
[`docs/models/mechanics_world_model.md`](docs/models/mechanics_world_model.md)
for the reasoning.

Plot the loss history:

```bash
.venv/bin/plot-training-history \
  --history-path checkpoints/rssm-cartpole/history.jsonl \
  --output-path checkpoints/rssm-cartpole/loss_history.png
```

Evaluate reconstruction and open-loop prediction quality. `eval-mechanics`
produces the same metric layout as `eval-rssm` (so the two models are
directly comparable) plus a linear-probe R² and learned-energy drift when
the dataset has `physics_qpos`/`physics_qvel`:

```bash
.venv/bin/eval-rssm \
  --checkpoint-path checkpoints/rssm-cartpole/latest.pt \
  --dataset-dir data/cartpole-swingup-random \
  --output-dir eval/rssm-cartpole \
  --warmup-length 4 --horizons 1 5 10 20

.venv/bin/eval-mechanics \
  --checkpoint-path checkpoints/mechanics-cartpole/latest.pt \
  --dataset-dir data/cartpole-swingup-random \
  --output-dir eval/mechanics-cartpole \
  --warmup-length 4 --horizons 1 5 10 20
```

Drive the trained world model through CEM / MPC on the real environment to
get an episode-return number (requires the checkpoint to have been trained
with `--reward-head`):

```bash
.venv/bin/eval-control \
  --checkpoint-path checkpoints/mechanics-cartpole/latest.pt \
  --output-dir eval/mechanics-control \
  --episodes 5 --horizon 15
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

Done:

- unstructured RSSM baseline (B1) with imagination-rollout loss, KL balancing,
  foreground-masked reconstruction, AdamW + cosine LR, and a live open-loop
  FG-MSE probe; stable training to h=10 FG-MSE ≈ 0.005 on 100k cartpole
- factored `(q, q̇, z_nuisance)` mechanics model with learned Lagrangian,
  Rayleigh dissipation, symplectic integrator, three-branch balanced KL;
  z-channel phase transition diagnosed and fixed (`nuisance_dim=4`,
  `nuisance_min_std=1.0`); `(q, q̇)` factorization hard-wired by finite
  differencing `q̇ = d/dt(q)` inside the encoder
- reward prediction head, CEM / MPC planner, physics-perturbation and
  visual-distractor wrappers, `eval-control` entry point
- state probes (episode-split, linear + MLP) and learned-energy diagnostics
- automatic physics-state, frame-timing and simulator-version
  serialization in the collector
- Phase-3 training and evaluation, joint multi-regime training, RSSM with
  reward head (results and their re-interpretation in
  `docs/current_status.md`)

Ahead:

- gate experiment: `scripts/train_gate.sh lagrangian|mlp`, then
  `scripts/eval_gate.sh` (continue/stop criteria in
  `docs/current_status.md`)
- only if the gate passes: model-agnostic CEM, swing-up-covering data, MPC
  under physically consistent OOD shifts, ablations
- keep the implementation inspectable enough to debug failure cases
  directly

Actor-critic training, replay APIs, zarr storage, and distributed collection
are deliberately out of scope until the main comparison is complete.
