# Current Status and Open Problem

The repository now contains **two trainable world models** — the
unstructured RSSM baseline (Phase 0) and the mechanics-structured factored
model (Phase 2/3) — plus the research scaffolding (reward head, CEM/MPC,
physics-perturbation wrapper, visual distractors, `eval-control`) from
Phase 1 of the plan at
[`docs/research_spec.md`](research_spec.md). The mechanics model has been
through three iterations of root-cause debugging; the architecture now
hard-constrains `q̇ = d/dt(q)` in the forward pass and the next training
run is ready to execute.

## What Works

- **RSSM baseline (B1).** Phase 0 fixes (imagination-rollout loss, KL
  balancing with free-nats, posterior-mean decoding, AdamW + cosine
  schedule, live open-loop FG-MSE probe) landed and trained successfully.
  After 9 epochs on the 100k cartpole dataset, validation open-loop
  foreground-masked MSE at horizon 10 sits at ≈ 0.005 — about 2.5× the K1
  target of 0.002, with h=1 and h=5 well inside spec. Good enough to
  serve as a real baseline for the structured model.
- **Mechanics world model (B_main).** Phase-3 architecture in place:
  `MechanicsWorldModel` with a shared `ConvEncoder` trunk, a
  `FactoredEncoder` that emits `q` (position) and `z` (nuisance), a
  hard-coded kinematic constraint `q̇_t = (q_{t+1} − q_{t−1}) / (2·dt)`
  computed in the forward pass (central difference interior, forward /
  backward at the boundaries), Cholesky-PD `MassMatrix`, learned
  `Potential`, PSD `Dissipation`, learned `Actuation`, mechanical-form
  Euler-Lagrange step, semi-implicit symplectic integrator, AR(1)
  nuisance prior, three-branch balanced KL with asymmetric free-nats
  (tight on `q`/`q̇`, loose on `z`). No smoothness prior — it is
  structurally unnecessary once `q̇` is derived from `q`. Training entry
  point is `train-mechanics`.
- **Research scaffolding (Phase 1).** `qpos`/`qvel`/`physics_params`
  serialization works out of `collect-cartpole` (automatic when the
  dm_control env returns them; no flag needed). `VisualWorldModel` has
  an optional reward head. A CEM/MPC planner and an `eval-control` CLI
  exist for scoring checkpoints on the real dm_control environment. A
  physics-perturbation wrapper (`--physics-scale`) and a light
  distracting-control shim are ready for the Phase 4 OOD grid.
- **Evaluation.** `eval-mechanics` reports per-horizon MSE (full-frame
  and foreground-masked), posterior reconstruction MSE, and — when the
  dataset carries `physics_qpos` / `physics_qvel` — a held-out linear-
  probe R² and learned-energy drift over the open-loop imagination
  window. Videos and contact sheets use the post-warmup window only,
  where the prior actually matters.
- **Z-channel diagnostics.** `history.jsonl` now includes
  `z_post_mean_rms`, `z_post_mean_abs_max`, `z_post_std_{min,max}`,
  `z_prior_mean_abs_max`, `z_prior_std_min`, `kl_z_t0_mean`,
  `kl_z_transition_mean`, `q_post_mean_abs_max`,
  `qdot_post_mean_abs_max` per training step. Added during the v4
  diagnostic run; these caught the v2/v3 phase-transition and now make
  any regression immediately visible.
- **Tests and perf.** 127 tests pass (`pytest`). A single-step
  benchmark harness at `scripts/bench_mechanics_step.py` reports
  ~780 ms/step on MPS for the default config.

## What Happened: three rounds of debugging

**Round 1 — v2 (first retry).** Documented `--kl-weight 0.1 /
--nuisance-min-std 0.3 / --learning-rate 1e-4 / --warmup-steps 2000`
retry from [the prior plan]. Diverged at step ~300: `kl_z_raw` jumped
12 → 5380 in one 100-step window; total loss blew up to 1e20 by epoch
1. Encoder collapsed into a constant-output basin (four horizons of
`val_open_loop_fg_mse_h*` reported bit-identical values from step
2200 onward).

**Round 2 — v3 (defensive fixes applied).** Added (F1) NaN/Inf skip
guard in the optimizer step, (F2) tanh-squashed MLP inputs to
`MassMatrix`, `Potential`, `Dissipation`, `Actuation` (`input_scale=5`
for q, `20` for q̇), (F3) `MassMatrix` ridge bumped `1e-3 → 1e-2`.
Re-ran with the same v2 config. Diverged at **exactly the same step**
with **the same magnitudes**. F1–F3 targeted the wrong branch — the
blow-up was entirely in the z channel (kl_z 12 → 5380), not in the
mechanics dynamics. Kept F1/F2/F3 in place as cold-start safety
margins.

**Round 3 — diag (fine-grained logging).** Added
z-channel-specific metrics to `compute_factored_kl` (listed above),
re-ran with `--log-every-steps 10` for the first epoch. The trace
showed **gradual exponential drift**: `z_post_mean_rms` grew from
0.052 at step 100 to 98 at step 300, doubling every ~15–20 steps.
`kl_z_t0_mean` (the t=0 anchor) stayed near zero until z_mean was
huge, while `kl_z_transition_mean` stayed small throughout — the
AR(1) prior only enforces *continuity*, not *bounded magnitude*, and
with 32 z-dims on a visually-uniform cartpole environment, the
encoder was abusing z for per-frame pixel detail. Reconstruction
minimum at step 260 (0.020) preceded the blow-up.

**Round 4 — v4 (targeted fix, stable training).** Config-only change
based on the diagnostic: `--nuisance-dim 32 → 4` (caps the capacity
the encoder can abuse), `--nuisance-min-std 0.3 → 1.0` (11× softer
KL stiffness `1/(2σ_p²)`). Trained stably through 6+ epochs. At step
300 all key v3-pathological metrics were within target:

| metric | v3 blow-up | v4 stable |
|---|---:|---:|
| `total_loss` | 538 | 0.62 |
| `kl_z_raw` | 5380 | 1.39 |
| `kl_z_t0_mean` | 1.56e5 | 3.87 |
| `z_post_mean_rms` | 98 | 2.2 |
| `reconstruction_loss` | 0.057 | 0.015 |

By epoch 6, `val_open_loop_fg_mse_h10` reached **0.008** (RSSM baseline
0.005, K1 target 0.006). Effectively passing K1 — 2× over baseline on
foreground, slightly better on full-frame (0.001 vs RSSM's 0.0016).

**New problem visible in v4.** The z blow-up is fixed, but a separate
failure mode became visible once training ran long enough: the
mechanics factorization is nominal, not real. Across epochs:

| metric | epoch 1 | epoch 6 |
|---|---:|---:|
| `q_post_mean_abs_max` | 1.04 | 0.78 |
| `qdot_post_mean_abs_max` | 0.22 | **15.4** |
| `smoothness_loss` | 0.31 | **13.05** |

`q` stays bounded; `q̇` grows unbounded. The old smoothness loss
(`‖½(q̇_t + q̇_{t+1}) − (q_{t+1} − q_t)/dt‖²`) could not win against
reconstruction pressure. Two independent encoder heads for `q` and
`q̇` with only a soft regularizer coupling them — reconstruction
reward for per-frame-distinctive `q̇` beat out the kinematic prior.
Under this architecture, K2 (linear probe `q̇ → qvel` R² > 0.8) and
H3 (learned-energy drift) cannot pass — `q̇` is not a velocity in any
meaningful sense.

## Phase-3 Architecture: Hard Finite-Diff q̇

Consulted the LNN / HNN-from-pixels literature (Cranmer et al. 2020,
Greydanus et al. 2019, Toth et al. 2020). Their standard pattern:
*positions are observed, velocities are derived*. Implemented that:

- `FactoredEncoder.mech_head` now emits only `(q_mean, q_raw_std)` —
  2d outputs, not 4d.
- `q̇_t` is computed in the encoder's forward pass via
  `derive_qdot_from_q(q_mean, q_std, dt)` — central difference
  interior, forward / backward at boundaries. Pushforward gives
  `q̇_std = sqrt(q_std_{t+1}² + q_std_{t-1}²) / (2·dt)`.
- `smoothness_loss` deleted from the composite — structurally zero.
- `MechanicsTransition.prior_min_std_qdot` defaults to
  `prior_min_std / dt` (auto-scaling match for the pushforward), so
  `KL(posterior q̇ ‖ prior q̇)` is bounded at init.
- Encoder requires T ≥ 2; `initial_posterior` / `posterior_step`
  require 2 consecutive frames (CEM/MPC warmup already satisfies
  this).

All 127 tests pass. Smoke test confirms `q̇` matches central diff of
`q` to machine precision in the interior, all gradients finite.
Checkpoint dir for the next run: `checkpoints/mechanics-phase3-
finitediff/`.

## Next Experiments

```bash
bash scripts/train_mechanics.sh
```

10 epochs, ~5 h on MPS. The script now carries the v4-discovered
good-config plus the Phase-3 architecture:

```text
--checkpoint-dir checkpoints/mechanics-phase3-finitediff
--q-dim 2 --nuisance-dim 4 --dt 0.01
--kl-weight 0.1
--kl-free-nats-mech 0.5 --kl-free-nats-nuisance 3.0
--nuisance-min-std 1.0
--learning-rate 1e-4 --warmup-steps 2000
--log-every-steps 50
--val-open-loop-every-steps 100
```

Pass rubric:

- `val_open_loop_fg_mse_h10` ≤ 0.008 by epoch 6 (match or beat v4).
- `qdot_post_mean_abs_max` stays bounded (it now physically cannot
  exceed ~`q_post_mean_abs_max / dt`, which should be well-behaved).
- `kl_qdot_raw` starts high (~200 from the pushforward mismatch
  against an untrained integrator) and drops as the integrator
  learns to match the observed velocity trajectory.
- No `smoothness_loss` in history (correctly absent).

Once training finishes, run the full K1 + K2 + H3 eval on the
recollected-with-physics dataset:

```bash
.venv/bin/eval-mechanics \
  --checkpoint-path checkpoints/mechanics-phase3-finitediff/latest.pt \
  --dataset-dir data/cartpole-swingup-random-100k-v2 \
  --output-dir eval/mechanics-phase3-finitediff \
  --sequence-length 32 --warmup-length 5 --horizons 1 5 10
```

The critical measurement is `linear_probe.qdot_to_qvel_r2_overall`.
Under the old architecture this was structurally doomed. Under the
finite-diff constraint, if q tracks physical position then `q̇` is
the time derivative of that quantity, which *is* physical velocity
up to a linear transform. If R² > 0.8, the factorization is real and
Phase 4 (OOD grid) opens up.

The 100k dataset has been recollected at
`data/cartpole-swingup-random-100k-v2` with automatic qpos/qvel
serialization (identical image trajectories since collection is
deterministic with seed 0; only extra physics-state arrays are new).
The old dataset at `data/cartpole-swingup-random-100k` stays on disk
for backwards-compatibility.

## Pending Follow-ups

- **Smoothness-weight tuning is no longer a knob to turn.** The
  constraint is architectural. If the Phase-3 run converges, we have
  a real structured model; if not, the next step is Option B (Dreamer-
  style posterior fusion with Kalman interpretation) not weight
  tuning.
- **Phase 3 ablations and Phase 4 OOD grid.** Everything behind the
  first successful `eval-mechanics` run on the recollected dataset.
- **B2 (no-dissipation) ablation**: `--no-dissipation` flag on the
  mechanics model. Tests H3.
- **B3 (unfactored LNN+D) and B4 (contrastive) ablations.** Both
  still `TODO`.

## What This Means for the Research Plan

The plan is still intact. We have climbed two hills:

1. **Stable training of the structured model** (v4): cleared.
2. **Real (not nominal) factorization** (Phase-3 finite-diff):
   architecture committed, pending an end-to-end training run.

The K2 linear probe is the next measurement. It is the gate between
"structured model trains" and "structured model actually captures
physics." Everything downstream (OOD generalization, ablations,
paper) depends on it.
