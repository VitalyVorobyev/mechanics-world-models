# Current Status and Open Problem

## Latest (2026-10-19, overnight): H1b generalizes across parameters; real pose front ends are the bottleneck

Details, predictions written before each run, and tables are in
[`hypotheses.md`](hypotheses.md#h1b--few-shot-adaptation-current-focus).

- **Second and third parameters (true state):**
  - Actuator gear: the same pattern as cart mass. Parity inside the
    trained range, and the advantage over the context MLP grows outside
    it (×2: 1.8×, ×3: 4.0×).
  - Gravity without a matching slot: the Lagrangian is 2–3× *worse*,
    even inside the range.
  - Gravity with a potential-scale slot: up to 5× better. The four-slot
    model (inertia, actuation, potential) keeps the mass result.
  - So the claim needs its qualifier: **structure extrapolates where the
    shifted parameter has a matching slot, and hurts where it does not.**
- **Surprise 1 explained:** the mass and gain slots are a near-gauge.
  Their ratio matches the true inertia ratio within 3%; only the split
  is arbitrary.
- **Pose noise (simulated):**
  - Finite-difference start velocities make noise dominate at once: a
    two-frame start is useless already at 0.08 px.
  - A model-based start state (shooting through the model over 8–16
    context frames) restores the advantage: ratio 1.67 at 0.25 px.
  - At 0.8 px, training itself fails unless training velocities are
    smoothed; with smoothing, 1.54.
- **Real pose front ends (pixels in, poses out, then the context
  oracle):**
  - Supervised keypoint and conv encoders, and a label-free colour/moment
    heuristic, all read typical in-view poses at about 0.1–0.6 px.
  - But the Lagrangian's advantage does not survive them: ratio 1.2
    (keypoint), 0.75 (conv), 0.54 (heuristic). With real errors the
    Lagrangian's nominal fit degrades far more than the MLP's, the
    reverse of the Gaussian case.
  - Removing the smooth per-pose bias does not help. Outlier rejection
    and a synthetic heavy-tail test are running to check the leading
    explanation (q̇² Coriolis terms amplify velocity spikes).
- **The camera does not see the system in 35% of frames** (A-VIEW-1).
  The cart is out of view in 26%, since the frame spans ±1.66 m and the
  rail ±1.8 m. Every pixel model trained on this data was asked to
  predict invisible states.
- **Self-supervised front end (A-FE-1):**
  - Gradient training through even a perfect, frozen renderer cannot
    find the pole angle: the loss is flat beyond about 10°.
  - Render-and-compare search finds it: 0.5° median with a renderer
    fitted on true poses.
  - Label-free, a classical colour-mask and moments heuristic gives
    1.24° median, about as good as the supervised keypoint encoder. EM
    on top does not reduce the bias.
- **Infrastructure:**
  - noisy-pose, start-state, encoder-pose, heuristic front-end,
    frozen-renderer and bootstrap oracles;
  - `--param` for gear and gravity;
  - fixed a result-file collision between parallel seeds and a
    zero-sigma weighting bug in the shooting estimator.
- **Next:** decide from the outlier tests whether robust training is
  enough for real front ends. If it is not, build an explicit
  observation-noise model (a filtering posterior) before any end-to-end
  pixel training.

## Earlier (2026-10-18, night): H1b supported on true state, for parameter extrapolation

Details, predictions and tables are in
[`hypotheses.md`](hypotheses.md#h1b--few-shot-adaptation-current-focus).

- **Rail contacts confounded v1.** About 30% of 20-step windows in
  random-policy data reach the ±1.8 m slider limit, a stiff
  non-Lagrangian constraint. Even the true smooth dynamics err there.
  The oracles now use rail-free windows; there, the cart-mass effect is
  25–40× the ceiling error.
- **Gauge (A-ACT-1).** With an unconstrained actuation `B(q)`, the
  learned `M` is not physical (M11 inflated 71–80×), because `B` absorbs
  the cart-pole coupling. `Actuation(actuated_dims=(0,))` fixes it:
  the learned `M` matches the true `M` up to scale, with residual 0.047.
- **Few-shot adaptation on true state:**
  - Single mass, 3-parameter physical slot against MLP fine-tuning:
    about 1.7× lower h20 error at k = 1–2 segments, parity by k = 5.
  - Matched baseline: a context-conditioned MLP trained on the same five
    masses. Inside the trained mass range the two are at parity. Outside
    it, the Lagrangian's advantage grows with the extrapolation distance
    (cart mass ×0.6: 1.2×; ×0.5: 2.3×; ×0.4: 3.1×; ×0.3: 5.8×), and it
    holds at speeds seen in training.
  - The claim is narrower than "structure helps adaptation": **a
    physically identified Lagrangian lets a 3-number adaptation
    extrapolate in parameter space**.
- **Next:** robustness to pose noise at the encoder-oracle level (a cheap
  intermediate), a second parameter, then the pixel version (A-FE-1
  first).

## Earlier (2026-10-18, evening): pivot to few-shot adaptation, protocol adopted

- **The gate run was stopped** at step 3967 (resumable). A targeted
  literature check and our own oracles showed it would test answered
  questions with a decoder known to fail end to end:
  - Lagrangian vs second-order MLP in distribution: near parity (KeyCLD);
  - pole-mass shifts: dynamically invisible (A-OOD-2);
  - zero-shot OOD: uninformative (H1-zs);
  - random-data swing-up MPC: fails (A-DATA-1).
- **New focus: H1b, few-shot adaptation to changed physics**, tested
  state-first. The v1 dynamics oracle was inconclusive: the MLP overfits
  at k = 1, and the nominal-accuracy prerequisite is not met. See
  [`hypotheses.md`](hypotheses.md#h1b--few-shot-adaptation-current-focus)
  for the result table and the v2 design.
- **Pixel front end:** a keypoint encoder plus geometric renderer is
  implemented. The renderer passes the decoder oracle; the encoder
  localizes x well but the pole tip poorly (A-FE-1, A-ENC-1).
- **Process:** [`research_protocol.md`](research_protocol.md) (oracle
  ladder, predictions before runs, strong inference) is now binding, and
  [`hypotheses.md`](hypotheses.md) is the source of truth for what the
  project believes. The sections below are history.

## October 2026 Review (read this first)

The project was paused from 2026-04-19 to 2026-10-18. A review on resume
(code audit, re-evaluation of the April checkpoints with corrected tooling,
literature check, and an independent review in `docs/chatgpt-review.md`)
changed several conclusions below. The April sections are kept as history;
where they conflict with this section, this section wins.

### Corrected findings

1. **The mechanics branch is bypassed; this is not gauge freedom.** With
   the new episode-split state probes (`eval/review-2026-10/`, 256 val
   windows, Phase-3 checkpoint), the pose `[x, cos θ, sin θ]` is
   recoverable from the *nuisance* channel (`z`: MLP R² 0.89) far better
   than from `q` (MLP 0.53, linear −0.38). Cause: `z`'s 3-nat free-nats
   floor makes information in `z` free (its KL sits under the floor ~99% of
   the time), every nat in `q` is penalized, and the legacy decoder reads
   `concat(q, q̇, z)` with nothing stopping `z` from carrying pose. In
   effect B_main was a small VAE whose pose lived in an action-free AR(1)
   latent. The "gauge freedom" interpretation further down was never
   tested and is superseded.
2. **The K2 linear probe was ill-posed.** Its target is the raw, unbounded
   hinge angle (−31.7…15.6 rad on v2), so no image function can reach
   R² 0.8. It also used a row-level split over overlapping windows. It is
   replaced by `state_probes` (see `docs/models/mechanics_world_model.md`).
3. **K1 comparisons mixed metrics.** On identical sequences (256 val
   windows, warmup 5) the RSSM with reward head beats Phase-3 everywhere:

   | fg open-loop MSE | h1 | h5 | h10 | h20 |
   |---|---:|---:|---:|---:|
   | copy last context frame | 0.00421 | 0.0152 | 0.0178 | 0.0192 |
   | mechanics Phase-3 | 0.00422 | 0.00783 | 0.00911 | 0.0129 |
   | RSSM + reward (`rssm-phase0-reward`) | **0.00230** | **0.00312** | **0.00494** | **0.00917** |

   Full-frame h20: RSSM 0.00102 vs mechanics 0.00316. The mechanics model
   degrades *faster* with horizon (pose in `z` decays as α^h, α ≈ 0.945)
   and only ties copying at h1. The RSSM latent is also *linearly*
   probeable for pose (R² 0.93) and partly for velocity (≈ 0.6); no
   mechanics latent predicts velocity at all. The 2× live-probe vs eval
   gap came from the live probe using the first 32 windows of one episode
   with the pole near rest; the probe now samples random windows.
4. **dt was 2× wrong.** Frames are 0.02 s apart (control timestep 0.01 ×
   action repeat 2); every mechanics run used `--dt 0.01`. `--dt` now
   defaults to the dataset value.
5. **The OOD data was physically wrong.** "Length" runs scaled
   `geom_size_2_1`, which is the static rail `rail2` (the pole is geom 4),
   and post-compile geom edits never change inertia anyway — so length07 /
   length13 changed only a drawn rail. Mass runs scaled the pole mass but
   not its inertia. The joint "5-regime" run therefore saw 3 dynamics
   regimes (two of them inconsistent) plus 2 visual variants.
   `pole_length` (MJCF-level) and inertia-consistent `body_mass_<i>` fix
   this; recollect with `scripts/collect_cartpole_ood.sh` (writes
   `*-15k-v2`).
6. **Path A could not have run.** `eval-control` / `CEMPlanner` only
   support `VisualWorldModel`; the B_main arm of `eval_control_grid.sh`
   would crash. Visual OOD there is colour gain/brightness, not
   Distracting Control. Also, random-policy data has reward > 1 in only
   0.47% of frames and the CEM horizon is 0.3 s, so a swing-up MPC
   comparison on this data would likely be noise against noise.
7. **Smaller issues, now fixed or flagged:** the central-difference `q̇`
   fed the first open-loop target into the imagination start (fixed:
   causal start; negligible effect on Phase-3 numbers); `q̇` samples were
   independent of `q` samples (fixed); the Rayleigh-MLP damping is not
   provably passive (new `--dissipation-form linear`); the reward head read
   `z`, contrary to the spec (fixed in `style` mode); the energy-drift
   metric depends on `V`'s constant and is not compared with true energy
   (documented); the training recipes for B1 and B_main differ (KL 1.0 vs
   0.1, LR 3e-4 vs 1e-4); `eval-rssm` could not load reward-head
   checkpoints (fixed).

### Literature position (late 2026)

The stated novelty ("first systematic OOD evaluation of
mechanics-structured visual world models") no longer holds as phrased.
DreamSAC (CVPR 2026, Hamiltonian world model from pixels with physics and
camera shifts), HaM-World (2026, (q, p, context) + CEM with zero-shot
physics shifts on DMC incl. cartpole, state input) and PH-Dreamer (2026,
port-Hamiltonian with dissipation on visual DMC, no OOD) cover adjacent
ground. Known negative results: Botev et al. 2021 (physics priors add
little on pixels), KeyCLD 2024 (the Lagrangian prior is weak without
constraint/keypoint structure), Chen et al. 2022 (learned state variables
are non-canonical). Baselines a reviewer will require: domain-randomized /
context-conditioned Dreamer (cRSSM, RLC 2024 — vanilla DreamerV3 fails
CartPole pole-length extrapolation), DALI (NeurIPS 2025), TD-MPC2. Pixel
cartpole-swingup is saturated in-distribution (~850+).

There is also a conceptual gap (raised in `docs/chatgpt-review.md`): a
model trained at one mass has no input that tells it the mass changed, so
structure alone cannot make it predict a heavier pole correctly. H1 splits
into **H1a** (structure helps extrapolation to unseen *states* of the same
system) and **H1b** (structure helps *adapt* to new parameters from a few
observations), and H1b needs an explicit adaptation or context mechanism
that the model does not have.

### Verdict and plan

Do not continue the April roadmap (Path A MPC grid, ablations, acrobot).
The core hypothesis was never tested because the mechanics branch was
bypassed, so the idea has not failed either. Run one time-boxed gate
experiment (about two weeks), then either stop and write up the findings
or re-scope to an analysis question along the lines of "when do mechanical
priors help visual model-based control?"

**Gate.** `bash scripts/train_gate.sh lagrangian` and
`bash scripts/train_gate.sh mlp` (pose forced through `q`: `style`
nuisance, angle dim, causal `q̇`, linear damping, dt = 0.02, reward head).
Then optionally `bash scripts/collect_cartpole_ood.sh`, then
`bash scripts/eval_gate.sh`. Continue only if gate-lagrangian:

1. reaches `state_probes.q->pose.mlp` R² > 0.8 on the nominal val split;
2. has `foreground_open_loop_mse_h20` ≤ the RSSM's on the same sequences
   (RSSM: 0.00917); and
3. degrades less than the RSSM at h20 on the OOD-v2 sets,

and beats gate-mlp on 2–3. If the MLP arm matches it, the Lagrangian
structure is not what helps. Only after a pass: a model-agnostic CEM
protocol, a dataset that covers upright states (a noisy energy-shaping
swing-up controller), MPC with true state as an upper bound,
predicted-vs-realized return to detect model exploitation, planning-time
benchmarks, and multiple seeds.

**Stop criterion.** If the gate fails, or the structured model only matches
the MLP arm while costing about 2× per step, stop developing this
architecture. Write up the debugging trail, the `z`-bypass and probe
pitfalls, and the OOD-perturbation bug as a technical note.

---

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
- **Tests and perf.** 139 tests pass (`pytest`, October 2026; 127 in
  April). A single-step benchmark harness at
  `scripts/bench_mechanics_step.py` reports ~780 ms/step on MPS for the
  default config on synthetic tensors; real runs measured 1.1–2.4 s/step
  for mechanics vs ≈ 0.6 s/step for the RSSM (file timestamps).

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
  require 2 consecutive frames. (The April note that "CEM/MPC warmup
  already satisfies this" was wrong: the planner never supported this
  model.)

All 127 tests pass. Smoke test confirms `q̇` matches central diff of
`q` to machine precision in the interior, all gradients finite.
Checkpoint dir for the next run: `checkpoints/mechanics-phase3-
finitediff/`.

## Phase-3 Training: Completed 2026-04-15

`bash scripts/train_mechanics.sh` ran 10 epochs / 26350 steps / ~18 h
wall-clock on MPS and finished cleanly. Checkpoint:
`checkpoints/mechanics-phase3-finitediff/latest.pt`. Training-history
plot: `checkpoints/mechanics-phase3-finitediff/loss_history.png`.

All Phase-3 architectural predictions verified in the training trace:

| Signal | v4 (old arch, e6) | Phase-3 (final, e10) |
|---|---:|---:|
| `val_open_loop_fg_mse_h10` (live probe) | 0.008 | **0.00481** (best 0.00452 at e8) |
| `qdot_post_mean_abs_max` | **15.4** | **1.10** |
| `kl_qdot_raw` trajectory | N/A (architectural) | 174.67 → 0.25 (700× drop within e1) |
| `z_post_mean_rms` (max) | ~1.0 | 2.35 (peak), 1.84 (final) |
| `smoothness_loss` in history | 13.05 | **absent** (structural) |
| NaN/Inf skips | 0 | **0 / 530 logged** |

The hard finite-difference constraint held exactly as designed. No
defensive guardrail (NaN skip, tanh-squashed MLP inputs) tripped.

## Phase-3 Evaluation: K1 Marginal, K2 Hard Fail

`bash scripts/eval_mechanics.sh` on `data/cartpole-swingup-random-100k-v2`
with warmup 5 and 64 random sequences wrote
`eval/mechanics-phase3-finitediff/metrics.json`:

| Gate | Metric | Target | Result |
|---|---|---:|---:|
| K1 | `foreground_open_loop_mse_h10` | ≤ 0.008 | **0.00892** (marginal miss; 0.00481 on the live probe with warmup 4) |
| K1 | `open_loop_mse_h10` (full-frame) | — | **0.00128** (excellent) |
| K1 | `foreground_reconstruction_mse` | — | **0.00288** |
| K2 | `linear_probe.q_to_qpos_r2_overall` | > 0.8 | **0.102** (per-dim 0.110 / 0.064) |
| K2 | `linear_probe.qdot_to_qvel_r2_overall` | > 0.8 | **0.006** (per-dim 0.043 / −0.004) |
| H3 | `energy.mean_abs_drift_h10` | bounded | 0.103 (uninterpretable without B2 no-D comparison) |

**Pixel trajectory is tracked well in open-loop** (contact sheets show
the cart and pole staying on-manifold for 20+ steps), but the encoder's
`q` has essentially **zero linear relation to physical cart position /
pole angle**. The failure is not at the pixel-prediction level; it is
at the coordinate-interpretation level.

## Interpretation: This Is Gauge Freedom, Not Posterior Collapse

> **Superseded (October 2026):** probes on `z` show the pose lives in the
> nuisance channel and `q` has nearly collapsed — an information-routing
> failure, not a chart ambiguity. See the review section at the top.

The Lagrangian framework is **invariant under diffeomorphic change of
coordinates**. For any smooth bijection φ : q → q', the equations of
motion look the same with `M' = J(φ)^T M J(φ)`, `V' = V ∘ φ⁻¹`, etc.
The four learned primitives (`M, V, D, B`) are flexible MLPs and can
absorb any such reparameterization. So the encoder + integrator pair
has a large gauge orbit of pixel-equivalent solutions, and SGD lands
on one that fits the data but does not happen to be the physical one.

The training-side signals — `kl_qdot_raw → 0.25`, `qdot_max` bounded —
confirm the encoder's `q̇` is consistent with its own `q` (that is what
the Phase-3 finite-diff constraint guarantees) and with the learned
integrator. They do **not** constrain which coordinate chart `q` lives
in.

The consequence for the research plan is that **Option B (Kalman-style
posterior fusion) solves posterior collapse, not coordinate
identifiability**. Posterior fusion ties the encoder to the integrator's
preferred coordinates, but the integrator itself has the same gauge
freedom, so the fused system can still settle into any chart. Before
committing to Option B as the fix, we need to either (i) accept that K2
might not be the right paper-gate and reframe around pixel-accuracy +
OOD generalization, or (ii) add a mechanism that *breaks* the
diffeomorphism-invariance of the Lagrangian form.

## What Might Break The Gauge: Candidate Next Steps

The problem is identifiability, so the fix has to inject external
structure about what `q` should mean. Options in rough order of
research-expense and plausibility on cartpole:

1. **Reconstruction-conditioned decoder.** Currently the decoder sees
   `concat(q, q̇, z)`, which lets `z` absorb the absolute pose while
   `q` encodes something else. Forcing the decoder to place the
   cartpole at a position *linearly read from `q`* (e.g., a
   differentiable-renderer or small-CNN decoder that takes `q` as a
   pose parameter) would make q-chart untenable if it is not the
   physical chart. **Most principled but also the most implementation.**
2. **Contrastive keypoint supervision (KeyCLD-style, CoRL 2021).**
   Add a keypoint-detection head trained (self-supervised) to predict
   cartpole keypoints in image space; regress `q` against those
   keypoints. Breaks diffeomorphism because image-space keypoint
   coords have a canonical meaning. **Changes the model but not the
   Lagrangian math.**
3. **OOD-dataset training.** Collect a small perturbed-physics
   dataset and train on the joint. If the encoder is a physical
   coordinate, physics perturbations move `V(q)` / `M(q)` in
   predictable ways; if not, reconstruction on perturbed data breaks.
   This tests H1 *during* training rather than after. Cheapest —
   just a data pipeline change.
4. **Explicit canonical-form constraint.** Penalize deviation from a
   diagonal mass matrix, or fix `B(q) = const` (cartpole's actuator
   is constant), to pin a gauge. Speculative; may over-constrain.
5. **Option B — Dreamer/Kalman-style posterior fusion.** Still worth
   trying *after* a gauge-breaking mechanism is in place, to sharpen
   the posterior. Not a standalone fix for K2 — see interpretation
   above.

## Joint OOD-Dataset Training (Option 3): Completed 2026-04-16

> **Caveat (October 2026):** the length07 / length13 regimes scaled the
> static rail `rail2`, not the pole, and the mass regimes did not scale the
> pole inertia. This run saw 3 dynamics regimes plus 2 visual variants of
> nominal, so it is not evidence about robustness to length changes.

7 epochs of `scripts/train_mechanics_joint.sh` on the 5-regime joint
dataset (`data/cartpole-joint-v3` ≈ 320 episodes / 160k frames),
followed by `scripts/eval_mechanics_joint.sh`. Outcome: **K1 improved
across the board, K2 still cleanly failed**.

| Metric | Phase-3 (nominal-only, e10) | Joint (5 regimes, e7) | Δ |
|---|---:|---:|---|
| `foreground_open_loop_mse_h10` (K1) | 0.00892 | **0.00738** | −17% |
| `reconstruction_mse` | 0.00045 | **0.00028** | −37% |
| `energy.mean_abs_drift_h10` (H3) | 0.103 | **0.057** | −45% |
| `q_to_qpos_r2_overall` (K2) | 0.102 | **0.030** | −70% |
| `qdot_to_qvel_r2_overall` (K2) | 0.006 | **0.081** | +1228% (still ≪ 0.8) |

Per-regime evals (`eval/mechanics-joint-v3-*/metrics.json`):

| Per-regime | `fg_h10` | `q→qpos R²` | `qdot→qvel R²` |
|---|---:|---:|---:|
| nominal v2 | 0.00792 | 0.092 | −0.012 |
| mass07 | 0.00792 | 0.052 | 0.258 |
| mass13 | 0.00875 | 0.127 | 0.026 |
| length07 | 0.00868 | 0.020 | 0.098 |
| length13 | 0.00713 | 0.048 | 0.069 |

No regime has a meaningfully better R² than the others. The strongest
per-dim per-regime fit (`qdot_to_qvel` dim 0 on mass07: 0.31) is still
far from "this is a linear reparameterization of physical velocity."
**The Lagrangian gauge orbit is robust to ±30% physics perturbations**
— the encoder + dynamics pair can absorb the regime variation by
reparameterizing `M, V, B` instead of by adopting a physical `q`.

Joint training was a real improvement on every pixel-level signal —
the model fits 5 physics regimes simultaneously, with sharper
reconstruction and more stable symplectic integration. It just doesn't
do so via physical coordinates.

## Reframe: K2 May Not Be the Right Paper-Gate

The research spec's hypotheses (H1 physical-OOD, H2 visual-OOD) are
about **MPC return under shift**, not about whether the latent is
linearly equivalent to physical coordinates. K2 was adopted as a
stricter proxy on the assumption that "physical coordinates make OOD
generalization easier." But two trained checkpoints now demonstrate
that the structured model produces **excellent pixel trajectories on
multiple physics regimes** — exactly what MPC needs — even with K2
near zero. The next step that actually advances the paper is to
evaluate H1/H2 directly on the closed-loop MPC return, not to keep
chasing K2.

## Next Experiment: Direct H1/H2 via MPC (Path A)

> **Superseded (October 2026):** `eval-control` cannot run the mechanics
> model, and the "length" OOD regimes only rescaled a rail. The RSSM reward
> run finished (≈ 5 h, stopped in epoch 11 of 20); the mechanics reward run
> and the grid never ran. Replaced by the gate experiment at the top.

Three new scripts, syntax-clean, awaiting execution:

| Script | Purpose | Wall-clock |
|---|---|---|
| `scripts/train_rssm_with_reward.sh` | Retrains B1 (RSSM Phase-0) on `data/cartpole-swingup-random-100k-v2` with `--reward-head --reward-weight 1.0 --imagined-reward-weight 1.0`. Output: `checkpoints/rssm-phase0-reward/` | ~18 h |
| `scripts/train_mechanics_with_reward.sh` | Same Phase-3 mechanics architecture + reward head + reward weights, on the v2 dataset. Output: `checkpoints/mechanics-phase3-reward/` | ~18 h |
| `scripts/eval_control_grid.sh` | `eval-control` grid: 1 nominal + 5 physical-OOD (mass×{0.5,1.5,2.0}, length×{0.5,1.5}) + 5 visual-OOD (color gain / brightness) × 2 models. Outputs `eval/control-grid/{B1,B_main}/{regime}/returns.json` | ~2–4 h |

Both training runs can go in parallel if you have spare compute. Both
also support `--auto-resume`.

**Pass / fail logic for the eval-control grid:**

| Result | Verdict | Next |
|---|---|---|
| B_main MPC return ≥ B1 on nominal AND on every physical-OOD point | H1 supported | Move to H2 visual-OOD; if also supported, paper draft |
| B_main wins on nominal but degrades faster than B1 on OOD | H1 falsified for these envs | Quantify how much faster; honest negative result. K2 was a real warning sign |
| B_main loses to B1 even on nominal | Lagrangian prior hurts pixel-MPC throughput | Inspect reward-head training (perhaps reward MLP underfits given the latent structure) |
| B_main wins by a lot OOD but K2 still 0 | Best paper outcome scientifically: "physical coordinates aren't necessary for OOD generalization, but mechanics structure is" | Highlight in paper; explains why K2 is a misleading sanity check |

## Orthogonal Pending Follow-ups

- **B2 (no-dissipation) ablation** for H3, regardless of Path A
  outcome — one `--no-dissipation` training run + eval. Cheapest
  ablation. Should run alongside the reward-head models.
- **B3 (unfactored LNN+D) and B4 (contrastive) ablations.** Still
  TODO. B3 is more interesting now: it shares the same gauge orbit
  as B_main, so if B_main wins H1/H2 and B3 doesn't, the q/z split
  is the operative inductive bias rather than the canonical `q`.
- **K1 eval-vs-probe gap** on Phase-3 (0.0089 eval vs 0.0048 live
  probe at matching horizons) — closed by the joint run (0.0074) but
  the discrepancy is unexplained; worth a quick `--warmup-length 4
  --num-sequences 256` rerun if you want to nail it down.
- **Path B (keypoint contrastive supervision)** stays on the table.
  If Path A's eval-control grid shows B_main losing to B1 even on
  nominal (i.e., the structured model is just bad at MPC), Path B is
  the next move — anchor `q` to image-space keypoints to force
  physical coordinates and retest.

## What This Means for the Research Plan

Three hills climbed; one uncovered; reframe to test the actual
research-spec hypotheses directly.

1. **Stable training of the structured model** (v4): cleared.
2. **Self-consistent factorization** (Phase-3 finite-diff): cleared —
   `q̇ = d/dt(q)` by construction, integrator tracks it, no drift.
3. **Multi-regime pixel prediction** (joint, Option 3): cleared
   convincingly. The structured model handles 5 physics regimes
   simultaneously with better pixel accuracy than nominal-only.
4. **Physical-coordinate factorization** (K2 linear probe): **not
   cleared** by either checkpoint. Shown to be robust to ±30%
   perturbations. May not be the right gate; the research spec's H1/H2
   are about MPC return, not coordinate identifiability.
5. **Direct H1/H2 evaluation** (Path A): scripts in place; awaiting
   user execution. This is the actual research result.
