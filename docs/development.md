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
build on Python 3.13 on macOS ARM (as of October 2026 `labmaze` 1.0.6 still
ships wheels only up to cp312).

To reproduce the exact environment the April 2026 datasets and checkpoints
were made with (torch 2.11.0, numpy 2.4.4, dm-control 1.0.38, mujoco 3.6.0),
install from the lock file instead of resolving fresh:

```bash
uv pip sync --python .venv/bin/python requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

`pyproject.toml` only sets lower bounds, so a fresh install pulls newer
mujoco / dm-control, which can change physics or rendered pixels relative
to the stored datasets. New collections record `software_versions` and
`frame_dt` in every episode file. Upgrade torch on a branch: torch 2.14
rewrote the MPS linear-algebra kernels, and the repo's MPS workarounds
(`linalg.inv` instead of `linalg.solve`, per-dim `autograd.grad` instead of
`jvp`) are guarded only by a source-text test, not by an MPS run.

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

## Research Bookkeeping (protocol v2)

See [`research_protocol.md`](research_protocol.md). The day-to-day commands:

```bash
# hygiene check: ledger provenance, pre-registration, claims, charter (also in pytest)
.venv/bin/python scripts/research/check.py

# new experiment record (status draft); fill the protected sections, commit = registration
.venv/bin/python scripts/research/new_experiment.py --slug my-test --title "…" --node Q1 --hypotheses H-CTRL

# run an oracle for a registered, approved experiment (refuses a dirty tree)
.venv/bin/python scripts/oracles/context_oracle.py --experiment E-004 …

# seed-aggregated table into the record + results/E-NNN/summary.json
.venv/bin/python scripts/research/report.py E-004

# ad-hoc aggregation over any result files (exploration / audits)
.venv/bin/python scripts/research/report.py --glob 'runs/oracles/context_oracle/2026*.json' \
    --where 'shifts=["gear_x1.3","gear_x0.5","gear_x2","gear_x3"]' \
    --metric '{shift}.k1.mlp_ctx.dev_h20 / {shift}.k1.lag_ctx.dev_h20'

# import result files that predate the ledger (idempotent)
.venv/bin/python scripts/research/import_legacy.py 'runs/oracles/*/2026*.json'
```

Every oracle CLI accepts `--experiment E-NNN` and `--allow-dirty`. Results
record the full commit, dirty paths, argv, package versions, data and
source fingerprints, and append one line to the committed
`results/ledger.jsonl`. Oracle caches are keyed by the source fingerprint
of `src/oracles/` and `src/models/mechanics/`. Caches written before
2026-10-19 used the config hash alone; `export_demo.py` still falls back
to them.

## Oracle Diagnostics (run before end-to-end training)

The oracle ladder from [`research_protocol.md`](research_protocol.md). Each
script tests one component against simulator ground truth and writes
`runs/oracles/<name>/<timestamp>.json` (plus `latest.json`) with pass/fail
criteria and the registry IDs it informs.

```bash
# Rung 0 (~5 s): how far each physics perturbation moves trajectories
.venv/bin/python scripts/oracles/intervention_effect.py
# Rung 1 (~30 s on MPS): decoder from the TRUE pose (render | deconv)
.venv/bin/python scripts/oracles/decoder_oracle.py --kind render
# Rung 2 (~1 min): encoder regressing the TRUE pose (keypoint | conv)
.venv/bin/python scripts/oracles/encoder_oracle.py --kind keypoint
# Rung 3 ceiling (~3 s): true MuJoCo dynamics under our integrator, with and
# without rail-contact windows (sets the floor any learned dynamics can reach)
.venv/bin/python scripts/oracles/dynamics_ceiling.py \
    --dataset nominal=data/cartpole-swingup-random-100k-v2 \
    --dataset cart_mass_x2=data/cartpole-ood-cartmass20-15k-v2:body_mass_1=2.0
# Rung 3 (~40 min CPU per seed): Lagrangian vs MLP on true state + few-shot
# adaptation, rail-free windows, symmetric adaptation grid
.venv/bin/collect-cartpole --output-dir data/cartpole-ood-cartmass20-15k-v2 \
    --total-frames 15000 --seed 12 --action-repeat 2 --physics-scale body_mass_1=2.0
scripts/oracles/run_dynamics_v2.sh            # seeds 0 1 2 in parallel
tail -f runs/oracles/logs/dynamics_v2_seed0.log   # progress: [cells done/total]
pkill -f dynamics_oracle.py                   # stop; re-run the launcher to resume
```

Use `--actuated-dims 0` for physically meaningful Lagrangian terms
(A-ACT-1); `TAG=dynamics_v3 scripts/oracles/run_dynamics_v2.sh
--actuated-dims 0` names the logs accordingly.

The dynamics oracle caches pretrained models under
`runs/oracles/dynamics_oracle/cache/pretrained/<pretrain-hash>/`, and
finished adaptation cells under `cache/<config-hash>/`. The same command
resumes after an interruption, and adaptation-only flag changes reuse the
pretrained models.

```bash
# Is the learned M the true M up to scale? (A-ACT-1, seconds)
.venv/bin/python scripts/oracles/mass_gauge.py \
    --model cart_only=runs/oracles/dynamics_oracle/cache/pretrained/<hash>/lagrangian.pt:0
# Matched baseline (H1b): context-conditioned MLP vs Lagrangian slots,
# trained on simulated cart masses {1, 0.7, 0.85, 1.2, 1.4}, tested on x0.5 / x2
.venv/bin/python -u scripts/oracles/context_oracle.py --seed 0 \
    >> runs/oracles/logs/context_seed0.log 2>&1
# Animated web demo data (no training): reload the cached seed-0 mass / gravity runs,
# redo the one-clip adaptation, export 2 s rollouts and a rendered front-end episode
.venv/bin/python scripts/oracles/export_demo.py --out runs/demo
# Same, seeds 0-2 in parallel with named logs; extra flags pass through
TAG=context_dose scripts/oracles/run_context.sh \
    --sim-shift cart_mass_x0.4=0.4 --sim-shift cart_mass_x0.3=0.3 --ks 1 5 --modes lag_ctx mlp_ctx
# Pose noise (pixel-encoder stand-in): sigma_x in metres, sigma_theta in degrees.
# Trains/adapts on noisy poses with finite-difference velocities and reports
# errors from the true and from a noisy start state, plus the true-dynamics floor.
TAG=context_noise_0.3px scripts/oracles/run_context.sh --sim-shift cart_mass_x0.3=0.3 --pose-noise 0.01 0.5
# Another physical parameter: actuator gear (or gravity) for training and shifts
TAG=context_gear scripts/oracles/run_context.sh --param actuator_gear_0 --sim-shift gear_x2=2
# Start-state estimators from N noisy context frames (backward / quadfit / model-based
# shooting), evaluation only: re-running a finished noisy config reuses its cells
TAG=context_noise_0.3px scripts/oracles/run_context.sh --sim-shift cart_mass_x0.3=0.3 \
    --pose-noise 0.01 0.5 --start-context 8
# Smoothed (Savitzky-Golay, 9-frame) training velocities for high noise
... --pose-noise 0.033 1.7 --train-velocity savgol
# Poses read from rendered frames by a saved encoder (fully visible windows only)
.venv/bin/python scripts/oracles/encoder_oracle.py --kind conv --steps 20000 --schedule cosine \
    --train-episodes 150 --save runs/oracles/encoders/conv-cosine20k-150ep.pt
TAG=context_encoder_conv scripts/oracles/run_context.sh --sim-shift cart_mass_x0.3=0.3 \
    --encoder runs/oracles/encoders/conv-cosine20k-150ep.pt --start-context 8
# Rung 2b: encoder learned without labels through a frozen renderer (A-FE-1)
.venv/bin/python scripts/oracles/frozen_renderer_oracle.py --kind keypoint
```

The encoder oracle reports the wrapped angle error in degrees and the
equivalent image errors (`x_err_px_rms`, `tip_err_px_rms`; 25.35 px per metre
on camera 0, so one pixel at the pole tip is 2.3°), comparable with the
pose-noise levels above.

The keypoint / renderer front end (`--encoder-type keypoint --decoder-type
render`, cartpole only) and the extra losses (`--latent-dynamics-weight`,
`--latent-dynamics-horizon`, `--keypoint-consistency-weight`,
`--kl-mech-scale`) are described in
[`models/mechanics_world_model.md`](models/mechanics_world_model.md).

## Train The Mechanics Model

### Gate experiment (October 2026, stopped; frozen)

See the review section of [`archive/status_history_2026-04_to_2026-10-19.md`](archive/status_history_2026-04_to_2026-10-19.md) for why
and for the pass criteria. The whole experiment runs as one resumable
pipeline (OOD-v2 collection → both training arms → evaluation → report):

```bash
bash scripts/gate_pipeline.sh start    # background (nohup); logs in runs/gate/
bash scripts/gate_pipeline.sh status   # stages, step/ETA, latest losses and live probe
bash scripts/gate_pipeline.sh stop     # graceful: the trainer writes latest.pt and exits
bash scripts/gate_pipeline.sh start    # continues where it stopped
```

Finished stages are skipped via `runs/gate/<stage>.done` markers. The
trainer writes a resumable mid-epoch `latest.pt` every 250 steps
(`--checkpoint-every-steps`) and whenever it receives SIGINT/SIGTERM, then
exits with code 130. A rerun with `--auto-resume` continues at the next
unseen batch (per-epoch seeded sampler) and drops history rows logged after
the checkpoint; resumed runs are bit-identical to uninterrupted ones on CPU
(`tests/test_train_resumable.py`). Each training run also maintains
`checkpoints/<run>/progress.json` (step, s/step, ETA, latest metrics).
`runs/gate/report.md` applies the gate criteria mechanically
(`scripts/gate_report.py`).

The individual steps can also be run by hand:

```bash
bash scripts/train_gate.sh lagrangian   # B_main with pose forced through q
bash scripts/train_gate.sh mlp          # same encoder/bottleneck, MLP dynamics
bash scripts/collect_cartpole_ood.sh    # optional: physically consistent OOD-v2 sets
bash scripts/eval_gate.sh               # both arms + rssm-phase0-reward, same sequences
```

The gate flags are `--nuisance-mode style --angle-dims 1 --qdot-stencil
backward --dissipation-form linear` and `--dynamics-form lagrangian|mlp`.
`--dt` is omitted so it resolves from the dataset (0.02 s). See
`docs/models/mechanics_world_model.md` for what each flag does. Compare runs
with `state_probes.q->pose.mlp` and `foreground_open_loop_mse_h20` in each
`eval/gate/<arm>/<dataset>/metrics.json`.

### April 2026 configuration (historical)

The config that emerged from the v2/v3/v4/diag debugging documented in
[`archive/status_history_2026-04_to_2026-10-19.md`](archive/status_history_2026-04_to_2026-10-19.md). `scripts/train_mechanics.sh`
holds it verbatim. It passes `--dt 0.01`, which is half the true frame
interval, and uses the `concat` nuisance mode in which `z` ends up carrying
the pose:

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
  iterations; see `archive/status_history_2026-04_to_2026-10-19.md` for the trace.
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

When the dataset carries physics state, `metrics.json` additionally holds
`state_probes` (linear + MLP probes from `q`, `q̇`, `z` and their
concatenations onto `[x, cos θ, sin θ]` and `qvel`, episode-level split —
the factorization check), the legacy `linear_probe` (raw-angle target; not
a meaningful gate) and `energy.*` (learned-energy drift, relative
diagnostic only). Both evaluators also write `copy_last_frame_mse_h*`, and
`eval-rssm` reports `state_probes` for `h`, `z` and the full RSSM feature.
Use `--num-sequences 256` so the probes see enough episodes.

OOD physics shifts: use `--physics-scale pole_length=<s>` (MJCF-level, so
inertia and centre of mass are recompiled) and `body_mass_<i>=<s>` (scales
mass and inertia together). `geom_size_<i>_<j>` only changes collision and
rendering, and in the April scripts `geom_size_2_1` pointed at the rail
`rail2`, not the pole.

### Joint OOD Training (K2 Gauge-Breaking Experiment, historical)

> The April `data/cartpole-ood-*-15k` datasets behind this experiment are
> physically wrong (rail instead of pole length; mass without inertia).
> `collect_cartpole_ood.sh` now writes corrected `*-15k-v2` sets; the
> assemble/train/eval joint scripts still point at the April directories
> and document what was actually run.

When the nominal-only run fails K2 (as Phase-3 finitediff did — see
`docs/archive/status_history_2026-04_to_2026-10-19.md`), the next experiment trains the same mechanics
architecture on a joint dataset spanning nominal + mildly-perturbed
physics. The four prep + run scripts:

```bash
# 1) Collect four mildly-perturbed regimes (~30-50 min on MPS).
bash scripts/collect_cartpole_ood.sh

# 2) Symlink all episode files into a single joint directory (instant).
bash scripts/assemble_joint_dataset.sh

# 3) Train on the joint dataset (~18 h on MPS, fresh run).
bash scripts/train_mechanics_joint.sh

# 4) Evaluate on joint + per-regime (~10 min).
bash scripts/eval_mechanics_joint.sh
```

The joint-assembler preserves the `episode_*.npz` prefix so
`src/data/sequence_dataset.py::index_episode_files` picks up every regime
in one glob; filenames are globally unique (`episode_{regime}_{idx}.npz`)
so the val split mixes regimes deterministically.

### Direct H1/H2 Evaluation via MPC (blocked)

> `eval-control` and `CEMPlanner` only support `VisualWorldModel`, so the
> B_main half of `eval_control_grid.sh` cannot run yet; only the B1 arm
> works. The grid's length points now use `pole_length`.

When K2 stays broken across nominal + joint runs (as documented in
`docs/archive/status_history_2026-04_to_2026-10-19.md`) but the structured model still produces clean
pixel trajectories, the right move is to test H1 / H2 directly via
closed-loop MPC return rather than continue chasing K2. Three scripts:

```bash
# 1) RSSM B1 baseline with reward head (~18 h).
bash scripts/train_rssm_with_reward.sh

# 2) Mechanics B_main with reward head (~18 h, parallelizable with 1).
bash scripts/train_mechanics_with_reward.sh

# 3) eval-control grid: 1 nominal + 5 physical-OOD + 5 visual-OOD,
#    for both checkpoints (~2-4 h total).
bash scripts/eval_control_grid.sh
```

`eval-control` requires a checkpoint trained with `--reward-head`. The
two `*_with_reward.sh` scripts add `--reward-head --reward-weight 1.0
--imagined-reward-weight 1.0` to the existing baseline configs and
write to dedicated checkpoint dirs (`checkpoints/{rssm-phase0,
mechanics-phase3}-reward/`) so they don't collide with the no-reward
baselines used for offline metrics.

The grid script writes `eval/control-grid/{B1,B_main}/{regime}/returns.json`
plus a per-run `summary.csv`. Aggregate via:

```bash
python -c 'import json,glob; [print(p, json.load(open(p)).get("mean_return")) for p in sorted(glob.glob("eval/control-grid/**/returns.json", recursive=True))]'
```

Tunable knobs at the top of `scripts/eval_control_grid.sh` via env vars
(`EPISODES`, `HORIZON`, `N_SAMPLES`, `N_ITERS`); the defaults match the
research-spec CEM/MPC setup (H=15, 512 samples, 3 iters, 10 episodes
per condition).

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
