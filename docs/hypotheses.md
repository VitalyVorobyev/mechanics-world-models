# Hypothesis and Assumption Registry

Source of truth for what the project believes and why (see
[`research_protocol.md`](research_protocol.md)).

**Statuses:**
- `assumed`: relied on, never tested.
- `open`: test planned.
- `supported` / `refuted`: by the cited evidence.
- `superseded`: replaced by another entry.

**Evidence** points to files under `eval/`, `runs/`, `checkpoints/`, or a
citation. Update an entry in the same commit as the code or result that
changes it.

Last full review: 2026-10-18.

## Summary

| ID | Statement (short) | Status |
|---|---|---|
| H1a | Mechanics structure improves extrapolation to unseen *states* of the same system | open |
| H1b | Mechanics structure improves *few-shot adaptation* to changed physical parameters | open: v1 oracle inconclusive (current focus) |
| H1-zs | Structure alone (no adaptation) degrades more gracefully under parameter shift | refuted as a useful question |
| H2 | A `(q, q̇, z)` split isolates visual nuisance and helps under visual shift | assumed |
| H3 | Dissipation matters (conservative-only underperforms on damped systems) | assumed |
| A-DEC-1 | A Linear→deconv decoder can render the cartpole from a 2–3 number pose | refined: renders from true pose; end-to-end failure is joint learning |
| A-FE-1 | Keypoint encoder + geometric renderer learn the pose end to end | open (renderer OK; tip keypoint weak) |
| A-ENC-1 | A per-frame CNN can regress `(x, cos θ, sin θ)` from 84×84 pixels | supported for x, partial for θ |
| A-ROUTE-1 | In `concat` mode the pose goes into `q`, not `z` | refuted |
| A-DT-1 | Model `dt` equals the frame interval | refuted for the April runs, fixed |
| A-OOD-1 | `geom_size_2_1` scales the pole length | refuted (it is a rail) |
| A-OOD-2 | Pole-mass ×0.7…×2 measurably changes the dynamics | refuted |
| A-OOD-3 | Cart-mass shifts measurably change the dynamics | supported |
| A-OOD-4 | Pole-length shifts change the dynamics without confounding appearance | refuted |
| A-DATA-1 | Random-policy data covers the states MPC swing-up needs | refuted |
| A-PROBE-1 | Linear `q → raw qpos` R² > 0.8 is a valid factorization gate | refuted |
| A-METRIC-1 | The live open-loop probe is comparable to the offline eval | refuted for April runs, fixed |
| A-LEAK-1 | The open-loop start state uses only context frames | refuted for April runs, fixed |
| A-DISS-1 | The Rayleigh-MLP damping is passive | refuted (not guaranteed); `linear` form added |
| A-LAG-1 | Unconstrained Lagrangian beats a second-order MLP in distribution | refuted by literature (near parity) |
| A-BASE-1 | The RSSM baseline is a fair stand-in for DreamerV3-class models | assumed |
| A-MPS-1 | `linalg.solve` / `jvp` workarounds are still needed on MPS | assumed (torch 2.11) |
| A-ENV-1 | Results reproduce with the locked environment | supported (CPU tests) |
| D-GAUGE | The April K2 failure was gauge freedom | superseded by A-ROUTE-1 |
| D-K1 | The April mechanics model matched the RSSM in distribution | refuted |

## Research hypotheses

### H1a — extrapolation to unseen states
- **Status:** open.
- **Statement:** Lagrangian structure predicts better than a second-order
  MLP on state regions under-represented in training (high energy, near
  upright).
- **Evidence so far:** KeyCLD (Daems et al. 2024) shows a small gain for an
  unconstrained Lagrangian over a second-order MLP ODE on cartpole (valid
  prediction time 28.0 vs 24.4 frames). Large gains came only from
  holonomic constraints. Gruver et al. 2022: second-order structure
  explains most of the gain.
- **Falsifying test:** dynamics oracle on true states (rung 3). Train on
  low-energy segments, evaluate rollouts on high-energy ones. If the
  Lagrangian shows no lead even with true state, drop H1a.

### H1b — few-shot adaptation (current focus)
- **Status:** open.
- **Statement:** after training at nominal physics, adapting a Lagrangian
  model to changed cart mass, actuator gear or gravity needs fewer shifted
  trajectories (k) than adapting a second-order MLP or an RSSM for the same
  error. The argument is that the change lives in a few identifiable
  terms (M, V, B).
- **Evidence so far:** none for pixels. Both literature searches found no
  from-pixels result (closest: CoDA / LEADS state-based adaptation; Lee et
  al. ICLR 2021 meta-learning Hamiltonians).
- **Falsifying test (state first):** dynamics oracle on true states with
  cart mass ×0.5 / ×2. Plot rollout error vs k ∈ {1, 2, 5, 10, 20}
  trajectories for (a) Lagrangian with only physical heads adapted,
  (b) Lagrangian fully fine-tuned, (c) second-order MLP fine-tuned.
- **Decision rule:** if (a) needs no fewer trajectories than (c) for equal
  error *with true state*, H1b is refuted and the pixel version is
  pointless. Otherwise the gap sizes the effect a pixel model must resolve.
- **Prediction (written 2026-10-18, before running
  `scripts/oracles/dynamics_oracle.py`):**
  - Prerequisite: both forms fit nominal true dynamics to a mean pose
    error at h20 well below the shift's own effect (cart mass ×2 moves
    trajectories by 0.027 at h20). If not, the test cannot resolve
    anything.
  - A cart-mass change is exactly an offset of `M[0, 0]`. A learned `M`
    is only identified up to a global scale shared with `V` and `B`, but
    the offset is still a single scalar in that scale. So I expect
    `lag_low` to adapt from k = 1–2 segments, with h20 error ≥ 30%
    below the best MLP mode at k = 1 (criterion
    `mlp_over_lag_error_ratio_k1_h20 ≥ 1.3`), and the gap to shrink by
    k = 20.
  - Risk: the learned `M` may not separate cart and pole terms the way
    the true one does, so the diagonal offset may not be the right
    knob. That would show up as `lag_low` plateauing above `lag_full`.
  - Caveat to report with any positive result: `lag_low` uses
    structure-given slots that an MLP lacks. The matching unstructured
    baseline is a context-conditioned MLP trained across several masses
    (cRSSM-style), which is a follow-up.
- **Result v1 (2026-10-18, `runs/oracles/dynamics_oracle/20261018-161222.json`,
  1 seed):** inconclusive.

  Mean pose error at h20:

  | | cart mass ×0.5 | cart mass ×2 |
  |---|---|---|
  | k = 0, Lagrangian / MLP | 0.085 / 0.082 | 0.036 / 0.032 |
  | k = 1, best Lagrangian / best MLP | 0.057 / 0.112 | 0.025 / 0.062 |
  | k = 20, best Lagrangian / best MLP | 0.051 / 0.057 | 0.027 / 0.016 |

  - `mlp_over_lag_error_ratio_k1_h20` = 2.2 passes; the k5 ratio fails.
  - The k = 1 advantage is mostly the MLP getting *worse* than zero-shot
    under the fixed adaptation recipe (overfitting), not the Lagrangian
    adapting efficiently.
  - The prerequisite failed: nominal h20 error is 0.027 (Lagrangian) and
    0.021 (MLP), as large as the shift effect. The Lagrangian fits nominal
    *worse* (h50: 0.103 vs 0.068).
  - `lag_low` plateaus above `lag_full`, the risk listed above.
- **Next test (v2):**
  - adaptation learning rate and steps tuned equally per mode (best over
    a grid, evaluated on held-out shifted episodes);
  - meet the nominal-accuracy prerequisite (`n_substeps`, longer
    training);
  - report results with and without windows near the rail limits
    (contacts are not Lagrangian);
  - 3 seeds.

### H1-zs — zero-shot graceful degradation
- **Status:** refuted as a useful question.
- **Statement:** structure alone degrades more gracefully than an RSSM
  under unseen parameters.
- **Evidence:**
  - A model trained at one setting has no input that tells it the
    parameter changed.
  - cRSSM (Prasanna et al. 2024): pixel CartPole pole length, extrapolation
    score 0.059 for single-context Dreamer.
  - DALI 2025: every method collapses out of range.
  - This setup cannot discriminate (A-OOD-2, A-OOD-4).
- **Replacement:** H1b.

### H2 — visual nuisance isolation
- **Status:** assumed.
- Note: in `style` mode `z` is a per-channel gain/offset on the decoder
  output (`--nuisance-mode style`). Untested under actual distractors.
- **Falsifying test:** colour/background shifts with the decoder oracle
  frozen. Does `q` stay stable (probe R²) while `z` absorbs the shift?

### H3 — dissipation matters
- **Status:** assumed.
- Note: nominal cartpole damping is tiny (hinge 2e-6, slider 5e-4); hinge
  damping ×100 changes trajectories by less than 1e-4 over 50 steps
  (2026-10-18 simulation). H3 needs a system or setting with substantial
  damping.
- **Falsifying test:** rung 0 on the damping range first.

## Assumptions

### A-DEC-1 — deconv decoder renders pose
- **Status:** refined. The deconv decoder *can* render from the true pose,
  but end-to-end training fails.
- **Decoder oracle (rung 1), 1500 steps, held-out foreground MSE vs
  mean-image floor 0.0166:**

  | Decoder | fg MSE | fraction of floor |
  |---|---|---|
  | deconv | 0.00425 | 0.26 |
  | geometric renderer | 0.00128 | 0.08 |

  The renderer's fg MSE is below the RSSM's own reconstruction (0.0026).
  (`runs/oracles/decoder_oracle_{deconv,render}/`.)
- So the end-to-end failure below is a *joint* encoder–decoder learning
  problem, not decoder capacity alone. The literature attribution was
  only partly right for our setup.
- **Earlier evidence (still valid):**
- **Evidence:**
  - The `style`-mode gate run (`checkpoints/gate-lagrangian`, stopped at
    step 3967) never rendered the foreground: foreground reconstruction
    0.0145 ≈ open-loop error at every horizon
    (`eval/gate-interim/lagrangian-e1`).
  - Same failure reported in the literature: KeyCLD ("HGN only learns the
    background"), Zhong & Leonard 2020, Jaques et al. 2020. Mechanism:
    CoordConv (Liu et al. 2018).
- **Confirmation still to run:** decoder oracle (rung 1) with true
  `(x, θ)`.

### A-FE-1 — keypoint encoder + geometric renderer
- **Status:** open.
- **Evidence:** overfit check on one batch of 16 real sequences, 600 steps
  (2026-10-18, scratchpad `overfit_kp.py`):
  - foreground reconstruction only reached mean-image level (0.0141);
  - corr(x, x_true) fell from 0.74 to 0.63;
  - cos θ correlation negative.
- **Competing explanations:**
  - (a) the renderer cannot draw the frames (geometry/initialization,
    receptive field);
  - (b) keypoints do not localize (spatial-softmax temperature, no
    heatmap loss — KeyCLD needed one);
  - (c) loss balance (foreground terms tiny);
  - (d) too few steps / too low a learning rate.
- **Rung results (2026-10-18):**
  - (a) eliminated: the renderer renders from the true pose (A-DEC-1).
  - Rung 2, supervised keypoint encoder, 1500 steps: x R² 0.997, but
    cos θ 0.34, sin θ 0.83 (overall 0.92), with inverse temperature 2.6.
    So the tip keypoint is blurry or imprecise. For comparison, the
    supervised conv encoder: x 0.995, cos 0.57, sin 0.88 (overall 0.947).
    (`runs/oracles/encoder_oracle_{keypoint,conv}/`.)
- **Remaining explanations:** (b) tip localization, plus the joint
  co-adaptation of an unsupervised encoder with the renderer.
- **Next tests:**
  - longer and sharper keypoint training (temperature schedule, 84×84
    heatmaps);
  - KeyCLD's heatmap↔blob loss;
  - the overfit test with the renderer frozen from the decoder oracle
    (isolates encoder learning).

### A-ENC-1 — CNN regresses the state
- **Status:** supported for x; partial for θ.
- **Evidence:**
  - Rung 2, conv encoder, 1500 steps: held-out R² x 0.995, cos θ 0.57,
    sin θ 0.88.
  - cos θ has little variance (the pole mostly hangs), so its R² is a
    harsh score.
  - RSSM features give pose R² 0.93 linearly.
- **To do:** longer training and an angle-error metric in degrees.

### A-ROUTE-1 — pose goes into `q`
- **Status:** refuted.
- **Evidence:** Phase-3, episode-split probes (`eval/review-2026-10/mechanics-phase3`):
  - `z → pose` MLP R² 0.89;
  - `q → pose` MLP R² 0.53, linear −0.38.
- **Cause:** free nats 3.0 on `z` vs 0.5 on `q`; the decoder reads `z`.

### A-DT-1 — dt matches frames
- **Status:** refuted for the April runs; fixed.
- **Evidence:** dm_control control timestep 0.01 s × action repeat 2 =
  0.02 s; April runs used `--dt 0.01`. Now resolved from data
  (`data/frame_timing.py`).

### A-OOD-1 / A-OOD-2 / A-OOD-3 / A-OOD-4 — perturbations
Mean |Δ(x, cos θ, sin θ)| against nominal, same random actions, 8 seeds
(2026-10-18 simulation; also the second literature agent's scripts):

| Shift | h20 | h50 |
|---|---|---|
| pole mass ×2 | 0.0023 | 0.0097 |
| pole mass ×0.7 | 0.0007 | 0.0032 |
| cart mass ×2 | 0.027 | 0.060 |
| cart mass ×0.5 | 0.052 | 0.113 |
| pole length ×1.3 (MJCF) | 0.0054 | 0.019 |
| hinge damping ×100 | ≈0 | ≈0 |

- `geom_size_2_1` was the rail `rail2` (A-OOD-1). Post-compile geom edits
  never change inertia.
- Pole mass is a 0.1 kg pole on a 1 kg cart and barely matters (A-OOD-2).
- Cart mass is the informative shift (A-OOD-3).
- Pole length also changes appearance, which confounds the encoder
  (A-OOD-4).

### A-DATA-1 — data coverage for MPC
- **Status:** refuted.
- **Evidence:**
  - Reward > 1 in 0.47% of random-policy frames.
  - PlaNet (Hafner et al. 2019): online collection was "necessary for the
    cartpole".
- **Implication:** any control experiment needs planner-collected or
  swing-up-covering data.

### A-PROBE-1 — linear raw-qpos probe as gate
- **Status:** refuted.
- **Evidence:**
  - The raw hinge angle spans −31.7…15.6 rad on v2, so no image function
    reaches R² 0.8.
  - The row split leaked overlapping windows.
- **Replaced by:** `state_probes`.

### A-METRIC-1 / A-LEAK-1 / A-DISS-1
- **A-METRIC-1:** the live probe used the first 32 windows of one episode
  with the pole near rest, about 2× easier than random windows. Fixed.
- **A-LEAK-1:** the central-difference `q̇` at the imagination start read
  the first target frame. Fixed (causal start; measured effect negligible
  on Phase-3).
- **A-DISS-1:** with velocity-dependent `C(q̇)` and the tanh bound,
  `q̇ᵀ ∂D/∂q̇` can be negative. `--dissipation-form linear` is passive by
  construction (test in `tests/test_mechanics_gate_options.py`).

### A-LAG-1 — Lagrangian vs second-order MLP, in distribution
- **Status:** refuted by literature.
- **Evidence:**
  - KeyCLD: KeyLD 28.0 vs KeyODE2 24.4 frames (1 actuator); "minimal"
    effect without constraints.
  - Gruver et al. 2022.
  - Botev et al. 2021: physics priors are not better than a Neural ODE on
    pixels.
- **Not worth re-testing at h ≤ 20** unless H1a/H1b need it as a control.

### A-BASE-1 — RSSM baseline
- **Status:** assumed.
- Ours is a compact DreamerV2-style RSSM (2.4M parameters), not DreamerV3.
  Fine as an internal baseline; do not present it as DreamerV3.

### A-MPS-1 — MPS workarounds
- **Status:** assumed.
- torch 2.14 rewrote MPS linear algebra.
- **Test:** an MPS smoke test of `forward_acceleration` backward with
  `linalg.solve` before upgrading torch.

### A-ENV-1 — environment reproducibility
- **Status:** supported.
- `requirements.lock.txt` pins the April environment; episode files record
  simulator versions.

## Diagnoses from history

### D-GAUGE — superseded by A-ROUTE-1
The April explanation for the near-zero linear probe ("Lagrangian gauge
freedom") was never tested. It is superseded by A-ROUTE-1 and A-PROBE-1.

### D-K1 — refuted
On identical sequences (256 val windows), foreground open-loop MSE:

| | h1 | h20 |
|---|---|---|
| `rssm-phase0-reward` | 0.0023 | 0.0092 |
| mechanics Phase-3 | 0.0042 | 0.0129 |

The earlier "K1 ≈ passing" compared the mechanics offline eval with RSSM
live-probe numbers.
