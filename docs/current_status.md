# Current Status and Open Problem

The repository now contains **two trainable world models** — the unstructured
RSSM baseline (Phase 0) and the mechanics-structured factored model (Phase 2)
— plus the research scaffolding (reward head, CEM/MPC, physics-perturbation
wrapper, visual distractors, `eval-control`) from Phase 1 of the plan at
[`docs/research_spec.md`](research_spec.md). The active blocker is mechanics
training stability: the first real `train-mechanics` run diverged around
step 200. Diagnosis and a retry config are below.

## What Works

- **RSSM baseline (B1).** Phase 0 fixes (imagination-rollout loss, KL
  balancing with free-nats, posterior-mean decoding, AdamW + cosine
  schedule, live open-loop FG-MSE probe) landed and trained successfully.
  After 9 epochs on the 100k cartpole dataset, validation open-loop
  foreground-masked MSE at horizon 10 sits at ≈ 0.005 — about 2.5× the K1
  target of 0.002, with h=1 and h=5 well inside spec. Good enough to serve
  as a real baseline for the structured model; [the plan's decision
  rule](../.) accepted this as "Phase 0 closed".
- **Mechanics world model (B_main).** Phase 2 code landed:
  `MechanicsWorldModel` with a shared `ConvEncoder` trunk, two Gaussian
  heads (`q`, `qdot`, `z_nuisance`), a Cholesky-parameterized PD
  `MassMatrix`, a learned `Potential`, a PSD `Dissipation`, a learned
  actuation map, a mechanical-form Euler-Lagrange step, a semi-implicit
  symplectic integrator, an AR(1) nuisance prior, a three-branch balanced
  KL with asymmetric free-nats, and a finite-difference smoothness prior
  on `qdot`. Training entry point is `train-mechanics`.
- **Research scaffolding (Phase 1).** `qpos`/`qvel`/`physics_params` can
  now be serialized inside trajectory NPZ files. `VisualWorldModel` grew
  an optional reward head. A CEM/MPC planner and an `eval-control` CLI
  exist for scoring checkpoints on the real dm_control environment. A
  physics-perturbation wrapper and a light distracting-control shim are
  ready for the Phase 4 OOD grid.
- **Evaluation.** `eval-mechanics` reports per-horizon MSE (full-frame
  and foreground-masked), posterior reconstruction MSE, `qdot`
  smoothness, and — when the dataset carries `physics_qpos` /
  `physics_qvel` — a held-out linear-probe R² and learned-energy drift
  over the open-loop imagination window. Videos and contact sheets use
  the post-warmup window only, where the prior actually matters.
- **Tests and perf.** 124 tests pass (`pytest`). A single-step benchmark
  harness at `scripts/bench_mechanics_step.py` reports ~780 ms/step on
  MPS for the default config, a ~20 % win over the pre-optimization
  implementation (`torch.linalg.solve` → `torch.linalg.inv` under MPS
  autograd-loop bug; batched posterior-prior integrator; fused
  gradient-norm computation).

## The Open Problem: mechanics training diverges early

First `train-mechanics` run on the default config diverged catastrophically
around global step 200:

| step | total_loss | reconstruction_mse | smoothness | kl_q_raw | kl_z_raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 22.2 | 0.043 | 0.002 | 1.7 | 18.6 |
| 200 | 1.2 × 10⁷ | 0.123 | 1.87 | 17 | 1.2 × 10⁷ |
| 300 | 2.3 × 10¹¹ | 0.124 | 3.4 × 10³ | 8.0 × 10⁴ | 2.3 × 10¹¹ |
| 3000 | 4.9 × 10¹⁹ | 0.125 | 4.1 × 10⁹ | 1.5 × 10¹¹ | 4.9 × 10¹⁹ |

Likely failure chain:

1. Step 100 looks healthy — encoder has started learning a low-MSE
   reconstruction and a low-norm `(q, qdot)`.
2. The nuisance posterior `z_std` collapses toward its `min_std` floor
   while `z_mean` drifts away from the AR(1) prior mean `α z_{t-1}`. The
   Gaussian KL `(μ_q − μ_p)² / (2 σ_p²)` then blows up.
3. Huge KL gradient corrupts AdamW's running averages (grad-clip at 100
   caps the step but cannot reverse the damage).
4. Encoder gets pushed into a degenerate basin where every input maps to
   roughly the same posterior. From there the gradient is ~0 everywhere
   and the model cannot escape.

Hard evidence of encoder collapse in `val_open_loop_fg_mse_h*` from step
1000 onward: all four horizons report bit-identical values
(`h1 = h5 = h10 = h20 = 0.15`), which only happens when the decoder is
producing a constant frame independent of input.

The 100k cartpole dataset was collected before Phase 1 added `qpos`/`qvel`
serialization, so the K2 linear-probe gate cannot be evaluated on this
data. A recollect is a prerequisite for judging whether the factorization
actually worked whenever the next training run does converge.

## Next Experiment

Restart (do not resume — the optimizer state is poisoned) with three
conservative knob changes:

```bash
.venv/bin/train-mechanics \
  --dataset-dir data/cartpole-swingup-random-100k \
  --checkpoint-dir checkpoints/mechanics-phase2-v2 \
  --sequence-length 32 --batch-size 32 --epochs 10 \
  --q-dim 2 --nuisance-dim 32 --dt 0.01 \
  --imagination-context-steps 4 --imagination-horizon 12 \
  --foreground-imagination-reconstruction-weight 2.0 \
  --kl-weight 0.1 \
  --kl-free-nats-mech 0.5 --kl-free-nats-nuisance 3.0 \
  --nuisance-min-std 0.3 \
  --smoothness-weight 1.0 \
  --learning-rate 1e-4 --warmup-steps 2000 \
  --val-open-loop-every-steps 200 \
  --val-open-loop-warmup 4 --val-open-loop-horizons 1,5,10,20
```

- `--kl-weight 0.1` (was 1.0): 10× cut on KL gradient magnitude.
- `--nuisance-min-std 0.3` (was 0.1): floors the nuisance posterior std so
  the Gaussian KL denominator cannot get close to zero.
- `--learning-rate 1e-4 --warmup-steps 2000` (was 3e-4 / 1000): slower
  climb to peak LR so the first ~2000 steps cannot blow up the optimizer.

Pass rubric: `val_open_loop_fg_mse_h10` ≤ 0.006 by epoch 6; `kl_z_raw`
stays under 30 nats throughout; `smoothness_loss` drops below 0.01 by
epoch 3. Miss: divergence before step 500 (another fresh start with
`--kl-weight 0.05`), or KL collapse to zero (raise free-nats).

## Pending Follow-ups

- **Re-collect the 100k cartpole dataset with `qpos`/`qvel`.** Required
  for K2 (linear probe) and for energy tracking against the real
  Hamiltonian. The collector already supports it (`--physics-scale`
  flag and backwards-compatible NPZ schema); just re-run
  `collect-cartpole` into a new directory.
- **NaN/Inf guard in the trainers.** A single bad batch should skip its
  optimizer step rather than poisoning AdamW's moving averages. Cheap to
  add; will reduce the blast radius of any future divergence.
- **Categorical latents (Phase 0.4, parked).** Only escalate if the next
  several mechanics runs keep missing K1 by more than 3×.
- **Phase 3 ablations and Phase 4 OOD grid.** Everything behind the
  first successful `train-mechanics` run.

## What This Means for the Research Plan

We are at the top of Phase 2 — the mechanics model exists in code, all
the diagnostics are wired up, but we have not yet demonstrated a single
successful training run. The B1 baseline is real, so the comparison is
meaningful *once* the mechanics model trains. The stability work is
narrow (loss scaling, not architecture), so the expectation is that a
handful of retries lands B_main before the plan has to reach for the
escape hatches (categorical latents, Hamiltonian reformulation,
shallower encoder).
