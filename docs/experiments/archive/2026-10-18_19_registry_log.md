> **Archived 2026-10-19.** The full hypothesis registry as it stood on 2026-10-19 08:15, including the long H1b lab notebook (predictions and results of the dynamics, context, noise, start-state, encoder, front-end and robust-loss oracles). Kept verbatim. Predictions in this file were committed together with their results, so their pre-registration cannot be verified from git. The compact index is [`../../hypotheses.md`](../../hypotheses.md); graded claims are in [`../../claims.md`](../../claims.md).

# Hypothesis and Assumption Registry

Source of truth for what the project believes and why (see
[`research_protocol.md`](../../research_protocol.md)).

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
| H1b | Mechanics structure improves *few-shot adaptation* to changed physical parameters | supported on true state for parameter *extrapolation* when the parameter has a matching slot: cart mass (up to 7×), actuator gear (4×), gravity with a potential slot (5×) vs a context MLP; parity inside the trained range; a missing slot makes it 2–3× *worse*. Through pixels (two-stage): about 1.5× with a conv encoder plus outlier rejection plus a model-based start state; 1.2–1.3 with label-free heuristic poses; lost without rejection (heavy tails hurt the Lagrangian about 2× more than the MLP) |
| H1-zs | Structure alone (no adaptation) degrades more gracefully under parameter shift | refuted as a useful question |
| H2 | A `(q, q̇, z)` split isolates visual nuisance and helps under visual shift | assumed |
| H3 | Dissipation matters (conservative-only underperforms on damped systems) | assumed |
| A-DEC-1 | A Linear→deconv decoder can render the cartpole from a 2–3 number pose | refined: renders from true pose; end-to-end failure is joint learning |
| A-FE-1 | Keypoint encoder + geometric renderer learn the pose end to end | open: gradient training cannot find θ (flat loss beyond ~10°); render-and-compare search through a fitted renderer reaches 0.5° median; label-free bootstrap untested |
| A-ENC-1 | A per-frame CNN can regress `(x, cos θ, sin θ)` from 84×84 pixels | supported for visible, non-upright poses (0.1–0.4 px); fails on upright (rare) and out-of-view frames (37%) |
| A-ROUTE-1 | In `concat` mode the pose goes into `q`, not `z` | refuted |
| A-DT-1 | Model `dt` equals the frame interval | refuted for the April runs, fixed |
| A-OOD-1 | `geom_size_2_1` scales the pole length | refuted (it is a rail) |
| A-OOD-2 | Pole-mass ×0.7…×2 measurably changes the dynamics | refuted |
| A-OOD-3 | Cart-mass shifts measurably change the dynamics | supported |
| A-OOD-4 | Pole-length shifts change the dynamics without confounding appearance | refuted |
| A-DATA-1 | Random-policy data covers the states MPC swing-up needs | refuted |
| A-VIEW-1 | The camera sees the cart and pole in every frame | refuted: cart out of view in 26% of frames, cart or tip in 37% |
| A-PROBE-1 | Linear `q → raw qpos` R² > 0.8 is a valid factorization gate | refuted |
| A-METRIC-1 | The live open-loop probe is comparable to the offline eval | refuted for April runs, fixed |
| A-LEAK-1 | The open-loop start state uses only context frames | refuted for April runs, fixed |
| A-DISS-1 | The Rayleigh-MLP damping is passive | refuted (not guaranteed); `linear` form added |
| A-LAG-1 | Unconstrained Lagrangian beats a second-order MLP in distribution | refuted by literature (near parity) |
| A-ACT-1 | Learned `M`, `V`, `B` match the true terms up to a common scale | refuted for unconstrained `B`; supported with `actuated_dims=(0,)` |
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
- **Ceiling check (2026-10-18, `runs/oracles/dynamics_ceiling/20261018-162020.json`):**
  explains the v1 prerequisite failure.
  - Setup: MuJoCo's own acceleration, slider limit disabled, integrated by
    our semi-implicit Euler (`scripts/oracles/dynamics_ceiling.py`).
  - Even the *true* smooth dynamics have nominal h20 error 0.031–0.037 on
    all windows, but only 0.0017–0.0070 on windows that stay ≥ 0.2 m from
    the ±1.8 m rail.
  - About 30% of h20 windows touch the rail in random-policy data. The
    rail limit is a stiff, non-Lagrangian constraint, so v1 measured who
    fakes the wall better.
  - Cart mass also changes how often the rail is hit (median first hit at
    frame 94 for ×0.5, 139 nominal, 275 for ×2), which confounds v1.
  - On rail-free windows the shift effect (nominal physics predicting
    shifted data) is 0.060 (×0.5) and 0.036 (×2) at h20. The ceiling (true
    shifted physics, 4 substeps) is 0.0024 and 0.0009, a 25–40× margin.
- **Test v2 (protocol in `src/oracles/dynamics.py`):**
  - Rail-free windows only, for both training and evaluation; adaptation
    segments never approach the rail.
  - 2 integrator substeps.
  - Pretraining uses a cosine learning rate.
  - Every mode gets the same adaptation grid: learning rates
    {3e-4, 1e-3, 3e-3, 1e-2} × checkpoints {0, 50, 100, 200, 400, 800}
    steps. Step 0 means "no adaptation", which removes the v1 overfitting
    confound.
  - Headline: one recipe per mode, chosen by mean selection error over all
    cells, on held-out shifted episodes disjoint from the test episodes.
    The per-cell best is reported alongside.
  - k ∈ {1, 2, 5, 20} segments of 50 frames; seeds 0, 1, 2.
- **Prediction v2 (written 2026-10-18, before running):**
  - Prerequisite (`prereq_nominal_over_shift_effect ≤ 0.3`): both forms
    fit nominal rail-free h20 below about 0.011. I expect this to pass
    (learned error about 0.005–0.010 against a ceiling of 0.0035 at 2
    substeps).
  - k = 0 errors close to the shift effect: about 0.06 (×0.5) and about
    0.036 (×2).
  - A cart-mass change is exactly a `+Δm` on `M[0, 0]`, and the scale gauge
    keeps it a single scalar. So `lag_low` should come within 2× of its
    nominal error at k = 1, provided the learned `M` is accurate.
  - I expect `mlp_over_lag_error_ratio_k1_h20 ≥ 1.3` (about 60%
    confidence) and the k5 ratio ≥ 1.3 (about 40%), because 250 frames
    may be enough for an MLP fine-tune.
- **Decision rule v2** (on the 3-seed mean, with the sign consistent across
  all seeds):
  - *Prerequisite fails:* fix the fit first (longer horizon or training);
    no conclusion about H1b.
  - *k1 and k5 both pass:* H1b is supported on state. Next is the matched
    unstructured baseline (an MLP conditioned on a learned context vector,
    trained across masses; same number of adapted parameters as `lag_low`).
    Pixels come only if structure still wins against that baseline.
  - *Only k1 passes:* the effect is limited to the extreme few-shot regime.
    Run the same matched-baseline check before deciding on pixels.
  - *Neither passes:* H1b is refuted on true state, so the pixel version is
    pointless. Write up the negative result.
- **Result v2 (2026-10-18, `runs/oracles/dynamics_oracle/20261018-1703*.json`
  and `-170402.json`, seeds 0–2):** meets the written decision rule, with
  caveats.
  - Prerequisite: 0.295 ± 0.018 on the mean, borderline (seed 2 alone:
    0.318). Nominal rail-free h20 is 0.008–0.0095 for both forms.
  - `mlp_over_lag_error_ratio_k1_h20` = 1.46 ± 0.21 and `_k5_h20` =
    1.60 ± 0.20. Both are above 1 in every seed; seed 0's k1 is 1.24.
    Per-cell recipe selection gives the same picture (1.44 and 1.56).

    Mean h20 over 3 seeds, one recipe per mode:

    | | k=0 lag / mlp | k=1 lag_low / lag_full / mlp_last / mlp_full | k=5 | k=20 |
    |---|---|---|---|---|
    | ×0.5 | 0.068 / 0.066 | 0.020 / 0.044 / 0.032 / 0.041 | 0.018 / 0.013 / 0.030 / 0.028 | 0.017 / 0.011 / 0.017 / 0.018 |
    | ×2 | 0.030 / 0.030 | 0.0042 / 0.0060 / 0.0053 / 0.0077 | 0.0044 / 0.0036 / 0.0038 / 0.0041 | 0.0043 / 0.0035 / 0.0034 / 0.0036 |

  - The effect is concentrated in ×0.5, where a lighter cart reaches
    faster states than nominal training covered. On ×2 the forms are at
    parity for k ≥ 5. At k ≥ 5 the advantage comes from `lag_full`, which
    points to better *extrapolation* (H1a-like) rather than the
    low-dimensional slot.
  - `lag_low` plateaus, now explained by A-ACT-1 (gauge) and the slot
    scale; see v3.
- **Prediction v3 (written 2026-10-18, before results; same protocol as v2
  plus `--actuated-dims 0`, see A-ACT-1):**
  - The learned `M` becomes proportional to the true `M` (relative residual
    < 0.2).
  - Nominal fit is unchanged within noise.
  - `lag_low` no longer plateaus: with all shifted data it reaches ≤ 1.5×
    nominal error, and at k = 1 it reaches ≤ 0.015 for ×0.5.
  - So the k1 and k5 ratios rise relative to v2.
  - If `M` is still not proportional, there is another gauge freedom
    (the potential or damping absorbing coupling, or `x`-dependence of
    `M`), and the next prior to test is translation invariance (`M`, `V`
    independent of `x`).
- **v3 interim (seed 0, `actuated_dims=(0,)`):**
  - Confirmed: the learned `M` is now proportional to the true `M`
    (relative residual 0.044, scale α ≈ 361).
  - `lag_low` still plateaued (×0.5, all data: 0.021), for a new reason.
    At α ≈ 361 the cart-mass offset needed is about −180. Adam moved it
    only to −11 and took the nearby shortcut of doubling the actuation
    gain.
  - Setting the offset to the analytic value, with no training, gives h20
    0.016 (×0.5) and 0.0048 (×2), against zero-shot 0.067 and 0.032. So
    the slot represents the shift; only its scale was wrong.
  - Fix: offsets are expressed as fractions of the nominal mean diagonal
    of the learned `M` (`slot_scale="diag"`, the oracle default from now
    on).
- **Prediction v3 (amended before the 3-seed run; protocol =
  v2 + `--actuated-dims 0` + `slot_scale="diag"`):**
  - `lag_low` approaches the analytic-offset errors: about 0.016–0.02
    (×0.5) and about 0.005 (×2) at k = 1, flat in k.
  - `mlp_over_lag_error_ratio_k1_h20 ≥ 1.5`.
  - The nominal prerequisite is unchanged from v2 (borderline, about 0.3).
- **Result v3 (2026-10-18, `runs/oracles/dynamics_oracle/20261018-1741*.json`
  and `-1747*.json`, seeds 0–2; gauge check in `runs/oracles/mass_gauge/`):**
  - The gauge is fixed. With `actuated_dims=(0,)` the learned `M` matches
    the true one up to scale (relative residual 0.047; slopes M00 1.00,
    M11 1.3, M01 1.7). Unconstrained models in all 3 v2 seeds have
    residual 0.91–0.93 and M11 inflated 71–80×.
  - `lag_low` now behaves as a physical slot: flat in k at 0.017 (×0.5) and
    0.0035 (×2) from k = 1, close to the analytic-offset values.
  - `mlp_over_lag_error_ratio_k1_h20` = 1.70 ± 0.23, at least 1.5 in every
    seed (prediction met).
  - `_k5_h20` = 1.26 ± 0.25, with seed 0 at 0.93: fails, and the sign is
    inconsistent. By k = 5 an MLP fine-tune is as good. On ×0.5,
    `lag_low` (0.019) is still better than the MLP (0.028–0.030), but
    `lag_full` is no longer better than v2's.
  - Constraining `B` removed `lag_full`'s v2 extrapolation edge (×0.5 k5:
    0.018 vs 0.013). Full fine-tuning had been using the extra freedom.
  - The prerequisite stays borderline (0.308; seed 1: 0.339). It is not a
    clean pass, though the shift is 3× the nominal error.
  - The residual of `lag_low` on ×0.5 (0.017 vs nominal 0.009 and ceiling
    0.0024) is model error on faster states than nominal training covered.
  - **Reading:** a robust but narrow effect. With 1–2 shifted segments
    (50–100 frames) the physical slot gives about 1.7× lower error than
    any MLP fine-tune. With 5 or more, the advantage is within seed noise.
    By the v2 rule this is "only k1 passes", so it goes to the matched
    baseline.
- **Matched baseline: context oracle (`scripts/oracles/context_oracle.py`).**
  - Setup: both forms train on simulated cart masses {1.0, 0.7, 0.85, 1.2,
    1.4} (30 episodes each). Each mass has a learned 3-number context,
    with the nominal one pinned to zero.
    - Lagrangian (with `actuated_dims=(0,)`): the context *is* the slot
      triple.
    - MLP: the context is an extra input.
  - Adaptation fits only a new context (`lag_ctx`, `mlp_ctx`: 3 numbers
    each), or everything (`*_full`). The adaptation grid and selection
    protocol are the same as v2.
  - The test masses ×0.5 and ×2 lie *outside* the training range.
  - Criterion: `mlp_ctx_over_lag_ctx_ratio_k{1,5}_h20 ≥ 1.3`.
- **Prediction (written 2026-10-18, before running):**
  - Within the training range, the context MLP should be fine. Outside it,
    an MLP context has no reason to extrapolate linearly in mass.
  - The Lagrangian slot is exactly linear in mass, so it should
    extrapolate. I expect `lag_ctx` ≈ v3 `lag_low` errors (about 0.016
    for ×0.5, about 0.005 for ×2) and `mlp_ctx` ≥ 1.5× worse at every k,
    because extra data cannot fix a context that has no extrapolating
    direction (about 60% confidence).
  - Zero-shot (k = 0) errors match v2 for both forms.
- **Decision rule:**
  - *Ratio ≥ 1.3 at k1 and k5 (3-seed mean, sign consistent):* H1b is
    supported on state against the matched baseline. The next step is the
    pixel version: keypoint front end, A-FE-1 first.
  - *Ratio < 1.3:* the structural advantage is matched by multi-condition
    training. H1b is reduced to "no advantage over cRSSM-style training".
    Stop H1b and write up.
  - *Mixed:* add an interpolation control (test masses inside the
    training range) before deciding.
- **Result (2026-10-18, `runs/oracles/context_oracle/20261018-18*.json`,
  seeds 0–2):** passes the written rule.
  - `mlp_ctx_over_lag_ctx_ratio_k1_h20` = 1.61 ± 0.12 and `_k5_h20` =
    1.70 ± 0.25; every seed ≥ 1.42.
  - Nominal h20: Lagrangian 0.0091, MLP 0.0080.

    Mean h20 over 3 seeds, one recipe per mode:

    | | k=0 | k=1 lag_ctx / lag_full / mlp_ctx / mlp_full | k=5 | k=20 |
    |---|---|---|---|---|
    | ×0.5 | 0.065 | 0.0100 / 0.019 / 0.023 / 0.040 | 0.0102 / 0.012 / 0.022 / 0.021 | 0.0101 / 0.011 / 0.022 / 0.014 |
    | ×2 | 0.029 | 0.0042 / 0.0066 / 0.0039 / 0.0089 | 0.0035 / 0.0044 / 0.0039 / 0.0042 | 0.0044 / 0.0040 / 0.0038 / 0.0036 |

  - Multi-mass training helped the Lagrangian base: `lag_ctx` on ×0.5 is
    0.010, against v3's single-mass `lag_low` at 0.017.
  - Surprise 1: the learned Lagrangian training contexts are not
    physical. Every non-nominal mass has a positive mass offset and a
    gain of about 1.6×, whereas the MLP contexts are linear in mass.
    Candidate causes: weak identifiability of mass vs gain from
    random-action data; the pinned nominal context plus the shared `M`;
    a mismatch between the pretraining scale and the calibrated scale.
    Not yet tested; it did not prevent good adaptation.
  - Surprise 1, discriminating test (prediction written 2026-10-18 after
    looking at seed 0's contexts only; seeds 1–2 unseen):
    - Explanation (a) predicts a near-gauge: the cart's response depends
      on the gain-to-inertia ratio, so only `(1 + o₀) / exp(g)` is pinned
      down. It should match the true cart-direction inertia ratio
      `(s + 0.1) / 1.1` within 5% for every training mass and seed, while
      `o₀` and `g` separately are arbitrary.
    - (b) and (c) predict no such match.
    - Test: read the learned contexts from the logs of seeds 1–2.
  - Surprise 1, result (2026-10-18, `runs/oracles/logs/context_seed{0,1,2}.log`):
    **explained by (a)**. `(1 + o₀) / exp(g)` matches `(s + 0.1) / 1.1`
    within 1.6–2.8% for all 12 (mass, seed) pairs, always 2% low, while
    the gain drifts to 1.4–1.75 and `o₀` follows it.
    - Why it is a near-symmetry: scaling the cart's mass entry and the
      actuation by the same factor λ leaves the cart's response to the
      motor unchanged. It changes the dynamics only through the coupling
      term `M₀₁² / (M₀₀ M₁₁)` (at most 7% on cartpole) and the pole's
      small reaction force on the cart (pole 0.1 kg vs cart 1 kg).
    - Consequence: the slots identify the cart-direction inertia relative
      to the motor gain, not mass and gain separately. Adaptation moves
      along the same ridge, so this does not affect the H1b result. A
      claim of "physically identified parameters" must say "up to this
      near-gauge".
  - Surprise 2, which matters for the claim: the entire advantage is on
    ×0.5, while ×2 is at parity, even though ×2 is further outside the
    training range (1.4). A lighter cart reaches faster states than
    training covered. So the advantage may be **state-space
    extrapolation** (H1a), not parameter adaptation.
- **Adversarial checks (predictions written 2026-10-18, before running):**
  - *Interpolation control:* test cart masses 0.775 and 1.3, inside the
    training range, simulated. If the mechanism is parameter adaptation,
    the Lagrangian still wins. If it is state extrapolation, they are at
    parity. My prediction: parity (ratio < 1.3, about 70% confidence),
    because ×2 already shows parity.
  - *Velocity split on ×0.5:* the `mlp_ctx` excess error concentrates in
    windows whose speeds exceed the 99th percentile of the training data.
  - *Consequence:* if both hold, the claim becomes "structure helps the
    model extrapolate to unseen *states* that a parameter change exposes".
    This is H1a, demonstrated through H1b. That is narrower than "few-shot
    adaptation" but still testable in pixels, and the paper framing
    changes accordingly.
- **Adversarial check results (2026-10-18):**
  - *Velocity split* (`scripts/oracles/context_velocity_split.py`):
    prediction **refuted**.
    - Only 8–12% of ×0.5 test windows exceed the training data's 99th
      percentile speeds.
    - Inside the training speeds, `mlp_ctx` is still 2.2–2.6× worse
      (0.020–0.024 vs 0.0087–0.0092, seeds 0–2). The gap is not a
      speed-extrapolation artifact.
  - *Interpolation control* (`runs/oracles/context_oracle/`, shifts
    `cart_mass_x0.775`, `cart_mass_x1.3`, simulated): prediction
    **confirmed**.
    - Parity: ratio 1.04 ± 0.08 at k1 and 1.04 ± 0.01 at k5.
    - Both adapt to about 0.009 (0.775) and about 0.006 (1.3) from
      k = 1.
  - *Reading:* the advantage is **parameter extrapolation**. Inside the
    trained parameter range a context MLP is as good. Outside it, the
    Lagrangian slot (exactly linear in mass) keeps working while the MLP
    context saturates (`mlp_ctx` is flat in k at 0.022 on ×0.5).
  - The ×0.5 vs ×2 asymmetry fits the dynamics-relevant coordinate
    1/M₀₀. Training covers 0.67–1.25; ×0.5 lies at 1.67 (0.42 beyond the
    range) and ×2 at 0.48 (0.19 beyond).
- **Dose-response test (prediction written 2026-10-18, before running):**
  - Setup: simulated cart masses ×0.6, ×0.4, ×0.3 (light side) and ×3,
    ×5 (heavy side). Modes `lag_ctx`, `mlp_ctx`; k ∈ {1, 5}; 3 seeds.
  - 1/M₀₀ for these masses: 1.43, 2.0, 2.5, 0.32, 0.20, i.e. 0.18, 0.75,
    1.25, 0.35 and 0.47 beyond the trained range.
  - Prediction: the per-shift ratio `mlp_ctx / lag_ctx` rises with that
    distance. Roughly 1 at ×0.6 (0.18), clearly > 1.5 at ×0.4 and ×0.3.
    The heavy side is above ×2's parity but below the light side at
    equal distance, because a heavier cart also moves less, so the
    absolute errors are small.
  - Refuted if the ratio is flat or non-monotone in the distance.
- **Dose-response result (2026-10-18, `runs/oracles/context_oracle/`,
  shifts `cart_mass_x0.6` … `x5`, seeds 0–2):** **confirmed** on the light
  side; the heavy side sits at the floor.

  | shift | 1/M₀₀ beyond range | ratio k1 | ratio k5 | lag_ctx / mlp_ctx h20 (k1) | zero-shot |
  |---|---|---|---|---|---|
  | ×0.6 | 0.18 | 1.18 ± 0.09 | 1.09 ± 0.07 | 0.011 / 0.013 | 0.047 |
  | ×0.5 (earlier run) | 0.42 | 2.3 | 2.2 | 0.010 / 0.023 | 0.065 |
  | ×0.4 | 0.75 | 3.1 ± 0.7 | 2.8 ± 0.5 | 0.016 / 0.047 | 0.083 |
  | ×0.3 | 1.25 | 5.8 ± 0.4 | 5.6 ± 0.4 | 0.015 / 0.087 | 0.129 |
  | ×3 | 0.35 | 1.2 ± 0.4 | 1.1 ± 0.3 | 0.0036 / 0.0040 | 0.042 |
  | ×5 | 0.47 | 1.2 ± 0.3 | 1.6 ± 0.2 | 0.0032 / 0.0037 | 0.052 |

  - Light side: monotone in distance in every seed. At ×0.3 one 50-frame
    segment cuts the Lagrangian error 8.5× (0.129 → 0.015), while the
    context MLP stays at 0.087.
  - Heavy side: both forms adapt to *below* the nominal error (a heavy
    cart moves slowly), so they are at the floor and cannot discriminate.
    That is consistent with the prediction, but weak evidence.
- **H1b on true state (summary):**
  - With a physically identified Lagrangian (`actuated_dims=(0,)`), a
    3-number adaptation of the mass/gain slots extrapolates to cart masses
    outside the trained range.
  - Its advantage over a context-conditioned MLP trained on the same
    masses grows with the extrapolation distance, from parity inside the
    range to about 6× at 1/M₀₀ 1.25 beyond it.
  - The single-environment version gives about 1.7× at k = 1–2.
  - Caveats:
    - one system (cartpole) and one parameter (cart mass);
    - true state;
    - the prerequisite margin is borderline (nominal error about 0.3 of
      the ×0.5 effect);
    - the learned mass and gain slots are identified only up to a
      near-gauge; their ratio is right within 3% (Surprise 1, explained).
  - **Next per the decision rule:** the pixel version, starting with
    A-FE-1 (front end). A cheaper intermediate is noisy state (simulated
    keypoint noise at the level the encoder oracle achieves): does the
    advantage survive pose noise? Then a second parameter (actuator gear,
    or gravity via the potential) to show it is not specific to the mass
    slot.

- **Noisy-state test (prediction written 2026-10-18, before running):**
  - Question: does the advantage survive the pose noise a pixel encoder
    would add, with velocities from finite differences (as in the
    mechanics world model)?
  - Pixel scale of the 84×84 camera (projection matrix, camera 4 m away,
    fovy 45°): 25.35 px per metre in both directions. One pixel is
    0.039 m of cart position, or 2.3° of pole angle at the tip of the 1 m
    pole. (The first estimate of 30 px/m, used to choose the levels,
    counted the pole's floor shadow.)
  - Noise levels, per frame, independent (run tags in brackets):
    - about 0.08 px: σx 0.0033 m, σθ 0.17° [`0.1px`];
    - about 0.25 px: σx 0.01 m, σθ 0.5° [`0.3px`] (good subpixel
      keypoints);
    - about 0.8 px: σx 0.033 m, σθ 1.7° [`1px`];
    - plus a zero-noise run with the same shifts.
  - Training, adaptation and selection windows use noisy poses with
    central-difference velocities (the model's training stencil). Test
    errors are against true poses, in two ways:
    - from the true start state: did the dynamics come out right?
    - from a noisy start state with a backward-difference velocity: the
      model's open-loop start, with the true-dynamics floor on the same
      windows (`floor_true.noisy_start`).
  - Setup: simulated cart masses ×0.5, ×0.4, ×0.3; modes `lag_ctx`,
    `mlp_ctx`; k ∈ {1, 5}; seeds 0–2; otherwise the context-oracle
    defaults.
  - Finite-difference velocity noise relative to the data's velocity
    spread (std 0.8 m/s for x, 1.95 rad/s for θ):

    | noise | central diff, x / θ | backward diff, x / θ |
    |---|---|---|
    | 0.1 px | 0.12 m/s / 0.10 rad/s | 0.23 / 0.21 |
    | 0.3 px | 0.35 / 0.31 | 0.7 / 0.6 |
    | 1 px | 1.2 / 1.0 | 2.3 / 2.1 |

  - Prediction:
    - **True start state:** the ratio stays ≥ 1.3 at 0.1 and 0.3 px. At
      1 px both forms degrade, because noisy start velocities bias the
      fit (errors in variables, absorbed as extra damping). The ratio
      there is a coin flip.
    - **Noisy start state:** the floor from start-velocity noise
      dominates at 0.3 and 1 px (θ error of about σv × 0.4 s at h20).
      So the ratio falls towards 1 there, whatever the dynamics. That
      makes start-state estimation, not dynamics, the next bottleneck
      for pixels.
  - Decision rule (true-start ratio, mean over the three shifts, k1 and
    k5, 3 seeds):
    - *≥ 1.3 at 0.3 px:* the advantage survives realistic keypoint
      noise. The front-end spec becomes "about 0.3 px", and the pixel
      version goes ahead.
    - *≥ 1.3 only at 0.1 px:* the front end must be near-perfect.
      Before pixels, test better start-state estimation (multi-frame
      fit) and an errors-in-variables-aware fit.
    - *< 1.3 at 0.1 px:* the advantage is fragile. Diagnose before
      going further.

- **Noisy-state interim (2026-10-18, 0.08 and 0.25 px done, 1 px
  running):**
  - From the true start state the ratio holds: 2.2 ± 0.3 (0.08 px) and
    1.8 ± 0.1 (0.25 px, 2 seeds) at k1.
  - But noisy training hurts the MLP's nominal accuracy much more: at
    0.25 px its nominal h20 error is 0.050 against 0.0185 for the
    Lagrangian (0.014 for both at 0.08 px). Part of the ratio at 0.25 px
    is robustness to noisy training data, not adaptation.
  - From a noisy start with a two-frame backward difference, the
    true-dynamics floor is 0.054–0.065 at h20 already at 0.08 px, and
    0.14–0.15 at 0.25 px. Every model sits on that floor, and the ratio
    falls to 1.3 (0.08 px) and 1.0 (0.25 px). The predicted bottleneck
    appears even earlier than predicted.
  - Bug found: parallel seeds finishing in the same second overwrote each
    other's result file (lost: 0.25 px seed 1). Fixed in
    `OracleResult.write`; the run is regenerated from cache.
- **Start-state estimation (prediction written 2026-10-18, before
  running):**
  - Question: with a better causal estimate of the start state from the
    last 8 noisy frames (0.16 s), does the floor drop enough for the
    dynamics advantage to show again?
  - Estimators (`src/oracles/start_state.py`): `backward` (two frames),
    `quadfit` (least-squares quadratic over the context), and `shoot`
    (fit the state at the first context frame so the model's own rollout
    under the logged actions matches the context poses, then continue).
    `shoot` is what a world model's filtering posterior approximates.
  - Same windows for every estimator; the true-dynamics floor is reported
    for each.
  - Prediction:
    - True-dynamics floor at h20 with `shoot`: below 0.02 at 0.25 px,
      i.e. 7× below `backward`. `quadfit` in between.
    - With `shoot`, the learned-model ratio recovers to ≥ 1.3 at 0.08 and
      0.25 px, because the start-state error now depends on the model:
      a wrong model fits the context worse and predicts worse.
    - At 0.8 px even `shoot` leaves a floor comparable to the adapted
      errors; the ratio there is uncertain.
  - Decision rule (k1, mean over three shifts and seeds, `shoot`):
    - *≥ 1.3 at 0.25 px:* the pixel version needs a model-based
      (filtering or shooting) start state and a front end at about
      0.25 px. Both become requirements for the pixel model.
    - *< 1.3 at 0.25 px but ≥ 1.3 at 0.08 px:* the front end must be
      better than 0.1 px. That is likely out of reach at 84×84, so
      consider a higher resolution before pixels.
    - *< 1.3 everywhere:* the advantage does not survive realistic
      observation noise; stop H1b on pixels and write up the state-level
      result.

- **Noisy-state result at 0.8 px (2026-10-18):** training itself fails.
  Both forms reach a nominal h20 error of about 0.16 from the true start
  state (0.009 without noise). Central-difference velocities at 0.8 px
  carry 1.2 m/s and 1.0 rad/s of noise, as large as the signal, so the
  fit is an errors-in-variables problem for both forms. The ratio is 0.8,
  which is meaningless at that error level.
- **Smoothed training velocities (prediction written 2026-10-18, before
  running):**
  - Change: training, adaptation and selection windows use a 9-frame
    Savitzky–Golay slope instead of central differences
    (`--train-velocity savgol`). It is non-causal, which is allowed in
    training, and about 5× less noisy. Test start states as before, plus
    `shoot` with 16 context frames.
  - Runs: 0.25 px and 0.8 px, seeds 0–2, same shifts as the noise test.
  - Prediction:
    - At 0.8 px the nominal h20 error from the true start drops from 0.16
      to below 0.04 for both forms.
    - At 0.25 px the MLP's nominal penalty mostly disappears (0.050 →
      below 0.025).
    - The true-start ratio is ≥ 1.3 at 0.25 px, and ≥ 1.3 at 0.8 px with
      about 55% confidence.
  - Decision rule:
    - *Nominal accuracy restored:* the pixel model must use smoothed
      (non-causal) velocities in training. That is easy: the encoder's
      posterior can smooth over the training window.
    - *Not restored at 0.8 px:* treat 0.8 px as beyond reach, and require
      a front end at or below 0.25 px.

- **Noisy-state and start-state results (2026-10-18,
  `runs/oracles/context_oracle/` with `start_context` 16, seeds 0–2,
  k1, mean over ×0.5/×0.4/×0.3):**

  | noise, training velocity | nominal h20 lag / mlp | true-start ratio | `shoot` ratio | `backward` ratio | `shoot` floor (true dynamics) |
  |---|---|---|---|---|---|
  | 0 | 0.0087 / 0.0077 | 3.8 | n/a | n/a | n/a |
  | 0.08 px, central | 0.014 / 0.014 | 2.15 ± 0.23 | 1.93 ± 0.16 | 1.31 | 0.003–0.005 |
  | 0.25 px, central | 0.018 / 0.050 | 1.86 ± 0.08 | 1.67 ± 0.06 | 1.05 | 0.007–0.008 |
  | 0.25 px, savgol | 0.026 / 0.026 | 1.32 ± 0.09 | 1.32 ± 0.06 | 1.11 | 0.007 |
  | 0.8 px, central | 0.154 / 0.158 | 0.81 | 0.82 | 0.95 | 0.021–0.023 |
  | 0.8 px, savgol | 0.026 / 0.031 | 1.57 ± 0.26 | 1.54 ± 0.26 | n/a | 0.021 |

  - Against the written predictions:
    - *Start-state floor:* with the true dynamics, `shoot` reaches 0.007
      at 0.25 px with 16 frames, 20× below `backward` (0.14–0.15).
      Confirmed; quick check with 8 frames: 0.015.
    - *Ratio recovers with `shoot`:* confirmed at 0.08 and 0.25 px.
      The 8-frame run of the written rule is being repeated with the
      robust (Levenberg–Marquardt) solver.
    - *Smoothed velocities:* at 0.8 px nominal accuracy is restored
      (0.154 → 0.026, predicted below 0.04), and the ratio is ≥ 1.3
      (1.57). Confirmed. At 0.25 px the MLP's penalty falls
      (0.050 → 0.026, predicted below 0.025, so a narrow miss), but the
      Lagrangian gets worse (0.018 → 0.026). That was not predicted.
  - Reading:
    - Random actions change every frame, so velocity is not smooth at
      the frame scale. A 9-frame smoother trades noise for bias, which
      pays off only when the noise is large.
    - With central differences the Lagrangian tolerates noisy training
      data far better than the MLP (0.25 px: 0.018 vs 0.050 nominal).
      That is a second, separate benefit of structure.
    - `quadfit` is useless here (worse than `backward` at 16 frames)
      for the same reason: a quadratic cannot follow random-action
      motion.
    - Learned-model `shoot` errors are 1.5× their true-start errors, while
      the true-dynamics `shoot` floor is near the true-start floor. So
      residual model error, not the estimator, limits the start state.
  - **Requirements for the pixel version:**
    - a model-based start state (filtering or shooting over at least 8
      frames);
    - a front end at or below about 0.25 px, or smoothed training
      velocities at higher noise;
    - windows restricted to frames where the system is in view (A-VIEW-1).
  - The principled replacement for both stencils is to infer the
    training windows' start states jointly with the dynamics, which is
    what a sequence posterior does. Not tested yet.

- **Start-state result under the written rule (8 context frames,
  Levenberg–Marquardt `shoot`, seeds 0–2, central training velocities):**

  | noise | `shoot` ratio k1 / k5 | per-seed k1 | `backward` k1 | `shoot` floor (true dynamics) |
  |---|---|---|---|---|
  | 0.08 px | 1.91 ± 0.16 / 1.87 ± 0.22 | 1.97, 2.07, 1.69 | 1.31 | 0.006 |
  | 0.25 px | 1.67 ± 0.08 / 1.66 ± 0.15 | 1.56, 1.74, 1.69 | 1.06 | 0.015 |
  | 0.8 px | 0.82 / 0.81 | 0.81, 0.84, 0.80 | 0.96 | 0.050 |

  - **Decision (written rule): ≥ 1.3 at 0.25 px.** The pixel version
    needs a model-based start state (filtering or shooting) and a front
    end at about 0.25 px. With smoothed training velocities, 0.8 px also
    works (1.54, 16 frames).
- **Encoder poses instead of simulated noise (prediction written
  2026-10-18, before running):**
  - Setup:
    - A supervised pose encoder (rung 2; conv, and separately keypoint;
      20000 steps, cosine schedule, 150 nominal training episodes) reads
      `(x, θ)` from rendered frames of every training and shift episode
      (`oracles.pixel_pose`, renders match the dataset to 1/255).
    - Only windows whose frames all have the cart and pole tip in view
      are used (A-VIEW-1).
    - Central training velocities; `shoot` with 8 context frames; same
      shifts and seeds as the noise test.
    - Mass and gear do not change the appearance, so the nominal-trained
      encoder applies to every environment.
  - What differs from Gaussian noise: errors depend on the pose. Typical
    visible frames are at 0.1–0.4 px, but a few percent are gross
    outliers (sideways and upright poses, θ errors of tens of degrees).
  - Prediction:
    - True-start ratio ≥ 1.3 (about 70%).
    - `shoot` ratio ≥ 1.3 (about 55%). Outliers in the context frames
      can wreck a least-squares start state, and outliers in the training
      targets hurt the MLP's MSE fit more than the Lagrangian's.
    - The conv encoder (lower median error, worse tail) beats the
      keypoint encoder on nominal accuracy.
  - Decision rule (k1, mean over shifts and seeds, `shoot`):
    - *≥ 1.3 with either encoder:* a two-stage pixel pipeline
      (supervised encoder, then structured dynamics) preserves H1b. Next,
      remove the pose labels: a self-supervised keypoint front end
      (A-FE-1).
    - *< 1.3 with both:* look at the outliers first (robust loss,
      outlier rejection in the shooting fit) before any end-to-end pixel
      training.

- **Encoder poses: result (2026-10-18, `runs/oracles/context_oracle/`
  with `pose_source` `encoder:*`, seeds 0–2, k1, mean over three
  shifts):** **fails the written rule with both encoders.**

  | encoder (held-out visible θ rms) | nominal h20 lag / mlp | true-start ratio | `shoot` ratio (8 frames) | `shoot` floor, true dynamics |
  |---|---|---|---|---|
  | keypoint (1.5°) | 0.036 / 0.038 | 1.26 ± 0.09 | 1.17 ± 0.03 | 0.11 |
  | conv (3.7°) | NaN (seed 0) or about 0.06 / 0.024 | 0.74 (seeds 1–2) | 0.70 | 0.12 |

  - The true-dynamics `shoot` floor is 0.11. Gaussian noise at 0.25 px
    gives 0.015, and 0.8 px gives 0.050. So the encoder's errors hurt
    much more than their size suggests: they must be structured.
  - With conv poses the Lagrangian is *worse* than the MLP, and seed 0's
    pretraining diverged to NaN.
  - Competing explanations:
    - (a) heavy-tailed outliers that least squares cannot reject;
    - (b) a pose-dependent bias, i.e. a distorted chart in which the true
      dynamics are no longer of the assumed Lagrangian form. This would
      hurt the structured model more than the MLP (cf. Wang 2026);
    - (c) temporally correlated errors that bias finite-difference
      velocities;
    - (d) partially clipped frames near the image edge that pass the
      2 px visibility margin.
  - Discriminating test (cheap, no training): decompose the encoder error
    on rendered simulation episodes into a smooth bias in the true pose
    plus a residual. Measure the outlier fraction and the autocorrelation,
    and recompute the true-dynamics `shoot` floor with each component
    alone.

- **Encoder-pose diagnosis (2026-10-18, scratchpad `enc_diag.py`; 10
  rendered ×0.5 episodes, in-view frames):**
  - **Bug found:** with encoder poses no Gaussian noise is added, so
    `shoot` weighted its residuals by 1/σ = ∞ and never left its initial
    guess. The `shoot` columns and floors of the encoder runs above are
    invalid; true-start and `backward` are unaffected. Fixed with
    fallback weights of 0.01 m / 0.01 rad, a guard in `shoot`, and a new
    cache key. Re-running.
  - With correct weights, the true-dynamics `shoot` floor on encoder
    poses is 0.025 (keypoint) and 0.020 (conv), not 0.11. That lies
    between the Gaussian levels of 0.25 px (0.015) and 0.8 px (0.050).

  | | keypoint x / θ (px at tip) | conv x / θ |
  |---|---|---|
  | median error | 0.15 / 0.34 | 0.09 / 0.11 |
  | rms error | 0.40 / 0.60 | 0.41 / 1.17 |
  | smooth bias (median per pose bin), rms | 0.33 / 0.46 | 0.33 / 0.92 |
  | residual rms | 0.22 / 0.41 | 0.26 / 0.69 |
  | lag-1 autocorrelation, raw error | 0.84 / 0.66 | 0.79 / 0.89 |
  | frames with error > 1 px | 3.2% / 9.8% | 1.9% / 1.5% |

  - Reading: (a) outliers are rare; zeroing residuals over 1 px lowers
    the `shoot` floor only from 0.019 to 0.017 (keypoint). The dominant
    part is (b), a smooth pose-dependent bias, together with (c), errors
    correlated in time. The conv encoder has the larger θ bias (0.92 px),
    and its Lagrangian is the worse one, which fits (b): the structured
    model needs a canonical chart, and the MLP does not care.
  - **Prediction for the re-runs (written before running):**
    - With fixed weights, `shoot` ratios come close to the true-start
      ratios: keypoint about 1.2–1.3, conv below 1.
    - With `--train-velocity savgol` (encoder errors are at the level
      where smoothing paid off in the noise test), the keypoint
      Lagrangian's nominal h20 improves from 0.036 to about 0.025, and
      the true-start ratio reaches ≥ 1.3 (about 50%).
    - Conv stays at or below parity, because its chart bias remains
      (about 60%).
  - Decision rule:
    - *Keypoint + savgol ≥ 1.3:* the two-stage pipeline works with a
      geometric (keypoint) front end. The front-end spec gains a bias
      term: pose-dependent bias below about 0.5 px.
    - *Otherwise:* chart bias is the limit. A structured world model
      needs metric, canonical coordinates from its front end, which
      favours a calibrated geometric front end (A-FE-1) over a generic
      regressor. Test this directly: remove the encoder's binned bias
      and re-run.

- **Encoder-pose re-runs (2026-10-18):**

  | encoder, training velocity | nominal h20 lag / mlp | true-start ratio k1 / k5 | `shoot` ratio k1 / k5 | `shoot` floor |
  |---|---|---|---|---|
  | keypoint, central | 0.035 / 0.037 | 1.26 / 1.27 | 1.21 / 1.20 | 0.031 |
  | keypoint, savgol | 0.028 / 0.028 | 1.07 / 1.25 | 1.08 / 1.28 | 0.031 |
  | conv, central | 0.053 / 0.024 (seed 0 NaN) | 0.74 / 1.03 | 0.75 / 0.96 | 0.040 |
  | conv, savgol | 0.039 / 0.032 | 1.12 / 1.26 | 1.04 / 1.14 | 0.040 |

  - Against the predictions:
    - `shoot` ratios near the true-start ratios: confirmed (keypoint
      1.21, conv 0.75).
    - Smoothing improves the keypoint Lagrangian's nominal error (0.036
      → 0.028): confirmed. But the ratio does not reach 1.3, so that
      part is refuted.
    - Conv at or below parity: confirmed.
  - Adaptation on encoder poses gets only part of the way: adapted h20
    is about 0.08, against 0.01–0.016 on true state and 0.035–0.05 with
    0.25 px Gaussian noise.
  - **Decision (written rule): below 1.3 → test the chart-bias
    explanation directly.**
- **Bias-removal test (prediction written 2026-10-18, before running):**
  - Change: `--debias` subtracts the encoder's median error per (θ, x)
    bin. It is fitted on the nominal training episodes with true poses,
    an oracle calibration, and applied to every environment. The
    residual (keypoint θ 0.41 px rms, lag-1 autocorrelation 0.25; conv
    0.69 px, 0.58) stays.
  - Runs: keypoint and conv, central velocities, `shoot` with 8 frames,
    seeds 0–2.
  - Prediction:
    - If the chart bias is the limiting factor, debiasing lifts both
      encoders: the keypoint ratio to ≥ 1.3, and the conv ratio to
      ≥ 1.0 with the larger gain, since conv had the larger bias
      (0.92 px vs 0.46 px). I give this about 50%.
    - The alternative is that the time-correlated residual limits it. In
      that case the ratios stay near 1.2 (keypoint) and 0.8 (conv).
  - Decision rule:
    - *Debiasing lifts the ratio ≥ 1.3:* the front-end requirement is a
      metric, calibrated chart, with pose-dependent bias well below
      0.5 px. That favours geometric (keypoint) decoding with explicit
      calibration.
    - *It does not:* the residual's temporal structure is the problem.
      Model the observation noise explicitly (a filtering posterior with
      a learned noise model) rather than treating poses as clean.

- **Bias-removal result (2026-10-19):** the prediction's alternative
  holds, so chart bias is **not** the limiting factor.
  - Keypoint, debiased: true-start ratio 1.19 / 1.29 (k1 / k5), `shoot`
    1.14 / 1.20. Before debiasing: 1.26 / 1.27 and 1.21 / 1.20.
  - Conv, debiased: 0.76 / 0.84, and the Lagrangian's pretraining is NaN
    again for one seed (seed 1).
  - Per the written rule, the residual is the problem. One concrete,
    cheap-to-test mechanism comes before a full noise model:
    - rare large errors (1–3% of frames above 1 px; θ errors of 10° and
      more) become velocity spikes through finite differences;
    - the Lagrangian's training is destabilized by start velocities far
      outside the physical range (conv: NaN twice, nominal error 2–3× the
      MLP's).
- **Outlier rejection (prediction written 2026-10-19, before running):**
  - Change: a frame is invalid when its pose second difference exceeds
    6× the robust (MAD) scale of second differences in that episode, per
    coordinate (`--reject-jumps 6`). Observable, no labels.
  - Runs: keypoint and conv encoders, central velocities, `shoot` with 8
    frames, seeds 0–2.
  - Prediction:
    - the conv Lagrangian no longer diverges, and its nominal h20 error
      falls to within 1.5× the MLP's;
    - the keypoint ratio reaches ≥ 1.3 (about 45%).
  - Decision rule:
    - *≥ 1.3:* rare outliers were the limit; the pixel pipeline needs
      outlier rejection or robust losses.
    - *Otherwise:* build the explicit noise model (a filtering posterior
      with a learned, correlated observation noise).

- **Outlier-rejection result (2026-10-19, `--reject-jumps 6`, seeds
  0–2):**

  | encoder | nominal h20 lag / mlp | true-start ratio k1 / k5 | `shoot` ratio k1 / k5 (per seed, k1) | `shoot` floor |
  |---|---|---|---|---|
  | conv | 0.022 / 0.021 (was 0.05 / NaN vs 0.024) | 1.73 ± 0.82 / 1.73 ± 0.21 | 1.50 ± 0.52 / 1.58 ± 0.17 (1.09, 1.19, 2.23) | 0.0096 (was 0.040) |
  | keypoint | 0.035 / 0.038 | 1.13 / 1.26 | 1.08 / 1.17 | 0.028 |

  - Rejection removes 2–6% of training windows (conv 41.3k left of 43.9k).
  - Against the predictions:
    - conv Lagrangian stable, with nominal within 1.5× of the MLP:
      **confirmed**. It is now level with the MLP.
    - keypoint ratio ≥ 1.3: **refuted**.
  - Decision (written rule): for conv, rare outliers were the limit, and
    with rejection it passes on the mean (`shoot` k1 1.50, k5 1.58; every
    seed ≥ 1.38 at k5, k1 seed-dependent). For keypoint they were not.
    Its errors are uniformly larger (median 0.7°, smooth bias 0.46 px),
    which puts it near the 0.8 px Gaussian regime.
  - Front-end requirements so far:
    - typical error near 0.1–0.2 px (conv-level);
    - outlier rejection, or a robust loss;
    - a model-based start state.
    With all three, the advantage survives a real pixel front end at
    about 1.5×, against 3.8× on true state.

- **Heavy-tailed noise, a synthetic mechanism test (prediction written
  2026-10-19, before running):**
  - Mechanism: the Lagrangian's Coriolis and centrifugal terms are
    quadratic in q̇ and unbounded, while the MLP's q̇ input is
    tanh-saturated. A velocity spike from a pose outlier therefore pulls
    the Lagrangian's fit much harder. Small Gaussian noise does not, which
    is why the Lagrangian was robust there.
  - Setup: the 0.08 px Gaussian run (σx 0.0033 m, σθ 0.17°) plus 2% of
    frames with heavy errors (σx 0.1 m, σθ 10°)
    (`--pose-outliers 0.02 0.1 10`). Central velocities, `shoot` with 8
    frames, seeds 0–2. Compare with the same run without outliers
    (nominal 0.014 / 0.014, ratio 2.15).
  - Prediction: the Lagrangian's nominal h20 at least doubles (to ≥ 0.03)
    while the MLP's grows less than 1.5×, and the ratio falls below 1.3.
  - Decision rule:
    - *Confirmed:* heavy tails specifically break the structured model.
      Robust training (outlier rejection, Huber loss, or bounded
      velocity features in the Coriolis terms) is a requirement, and the
      rejection test above shows whether it is enough.
    - *Refuted:* outliers alone do not explain the real-front-end
      failure. Look at clipped frames and correlated errors next.
- **Heavy-tail result (2026-10-19, seeds 0–2):** mostly **confirmed**.
  - Nominal h20 without and with 2% outliers: Lagrangian 0.014 → 0.075
    (5.4×); MLP 0.014 → 0.035 (2.5×, against a predicted < 1.5×).
  - Ratios: true start 0.89 ± 0.13, `shoot` 0.86 ± 0.10 (k1), against
    2.15 and 1.91 without outliers.
  - So heavy tails hurt both forms and the Lagrangian about twice as much,
    consistent with unbounded q̇² terms amplifying velocity spikes.
    Together with the conv-encoder rejection result, robust handling of
    outliers is a requirement for the structured model.
  - Not yet tested: whether bounded velocity features in the Coriolis
    term, or a Huber loss, make the Lagrangian itself robust without
    rejecting frames.
- **Heuristic front end with rejection (2026-10-19):** the first attempt
  rejected 80% of frames and crashed on an empty adaptation set. The
  heuristic's cart column is pixel-quantized, so most second differences
  are exactly zero and the MAD scale collapses. Re-running with a floor
  on the scale (`--jump-min-scale 0.004 0.012`, about one frame of
  physical acceleration), which keeps 71% of frames against 77% before
  rejection.
- **Heuristic + floored rejection: result (2026-10-19, seeds 0–2):**
  - The Lagrangian's nominal h20 falls from 0.079–0.166 to 0.029–0.031.
    That is now *better* than the MLP (0.055), so rejection removes the
    instability with a label-free front end as well.
  - Ratios (mean over shifts with valid windows):

    | | seed 0 | seed 1 | seed 2 | mean |
    |---|---|---|---|---|
    | `shoot` k1 | 1.10 | 0.99 | 1.38 | 1.16 |
    | `shoot` k5 | 1.25 | 1.21 | 1.42 | 1.29 |
    | true start k1 | 1.20 | 1.00 | 1.67 | 1.29 |
    | true start k5 | 1.32 | 1.26 | 1.59 | 1.39 |

  - Seed 1 has no fully valid 8-frame context window for ×0.3 after
    rejection (NaN, skipped). `evaluate_starts` now returns NaN for empty
    window sets instead of crashing.
  - Reading: borderline. The label-free pipeline keeps part of the
    advantage, between the keypoint encoder and the conv encoder with
    rejection. The remaining gap fits the heuristic's larger smooth error
    (θ median 1.24°, rms 1.9°, about 0.55–0.8 px at the tip), the regime
    where Gaussian noise also leaves about 1.5× at best.

- **Robust loss instead of frame rejection (prediction written
  2026-10-19, before running):**
  - Change: a Huber loss (δ = 0.02 in pose features, scaled to equal the
    MSE below δ) in pretraining and adaptation (`--huber-delta 0.02`).
    The model is unchanged. `shoot` stays least squares.
  - Runs, seeds 0–2, `shoot` with 8 frames:
    - (a) the synthetic heavy-tail setting (0.08 px plus 2% outliers;
      with MSE: Lagrangian nominal 0.075, ratio 0.86);
    - (b) the heuristic front end *without* rejection (with MSE: 0.54;
      with rejection: 1.16).
  - Prediction:
    - (a) the Lagrangian's nominal error returns to ≤ 0.025 and the
      true-start ratio to ≥ 1.3 (about 60%).
    - (b) the Lagrangian's nominal error is ≤ 0.04, but the `shoot` ratio
      stays below 1.3 (about 65%). The least-squares start-state fit still
      sees the outliers in its context frames, which rejection removed.
  - Decision rule:
    - *Huber matches rejection on training:* use the robust loss
      (simpler, keeps every frame), and make `shoot` robust too.
    - *It does not:* keep frame rejection as part of the pipeline.
- **Robust-loss result (2026-10-19, seeds 0–2):**
  - (a) Synthetic heavy tails: **confirmed, strongly.**
    - Nominal h20: Lagrangian 0.015–0.017 (0.075 with MSE, 0.014 without
      outliers); MLP 0.019–0.021.
    - Ratios: true start 3.3 ± 0.7, `shoot` 2.6 ± 0.5 (k1), even above the
      no-outlier run (2.15 / 1.91).
    - With independent outliers, a Huber loss fully restores the
      structured model.
  - (b) Heuristic front end without rejection: Lagrangian NaN (seed 0) or
    0.059–0.061 nominal, MLP 0.055. Ratios 0.52–0.65 (true start) and
    0.57–0.65 (`shoot`). The nominal prediction (≤ 0.04) is refuted; the
    ratio prediction (< 1.3) is confirmed.
  - Reading: the real front end's bad frames are not isolated spikes.
    Huber handles independent outliers but not this. The likely culprits
    are sustained runs of wrong frames, where the pole is clipped,
    crosses the cart band, or the heuristic misfires on consecutive
    frames. Rejecting a frame together with its neighbours removes those
    (nominal 0.030).
  - **Decision (written rule): keep frame rejection in the pipeline.**
    Add Huber on top for independent outliers. Next diagnostic: measure
    the run-length distribution of the heuristic's large errors.

- **Second parameter, actuator gear (prediction written 2026-10-18,
  before running):**
  - Setup: as the context oracle, but the five training environments vary
    the actuator gear (×1.0, 0.7, 0.85, 1.2, 1.4) instead of the cart
    mass. Simulated test gears: ×1.3 (interpolation), ×0.5, ×2, ×3.
    Modes `lag_ctx`, `mlp_ctx`; k ∈ {1, 5}; seeds 0–2; true state.
  - The Lagrangian's gain slot represents a gear change exactly (log gain
    = log scale); the mass slots are not needed.
  - Prediction:
    - ×1.3: parity (ratio < 1.3), as for mass interpolation.
    - ×2 and ×3: the Lagrangian is better, ratio ≥ 1.5 at ×2 and higher at
      ×3. A context MLP has to learn a context-dependent action response,
      and nothing makes that extrapolate linearly. About 60% confidence:
      the MLP's action pathway may be nearly linear and extrapolate a
      gain by itself.
    - ×0.5: both near the floor (a weak motor moves the cart little), so
      no discrimination.
  - Decision rule (k1 and k5, 3-seed mean):
    - *Ratio ≥ 1.3 at ×2 and ×3, rising from ×2 to ×3, parity at ×1.3:*
      H1b holds for a second parameter on true state; the claim covers
      actuation as well as inertia.
    - *Parity at ×2 and ×3:* the advantage is specific to the mass slot.
      Narrow the claim to inertial parameters, and look at why the MLP
      extrapolates gain (inspect its learned contexts).
    - *Lagrangian worse:* diagnose (slot misuse, recipe selection) before
      drawing conclusions.

- **Second parameter, actuator gear: result (2026-10-18,
  `runs/oracles/context_oracle/`, train envs `gear_x*`, seeds 0–2):**
  **confirmed**, and it passes the written rule.

  | shift | ratio k1 | ratio k5 | lag_ctx / mlp_ctx h20 (k1) | zero-shot |
  |---|---|---|---|---|
  | ×1.3 (inside) | 1.02 ± 0.13 | 0.97 ± 0.04 | 0.0098 / 0.0100 | 0.027 |
  | ×0.5 | 1.15 ± 0.07 | 1.13 ± 0.02 | 0.0035 / 0.0040 | 0.032 |
  | ×2 | 1.84 ± 0.10 | 1.90 ± 0.07 | 0.0155 / 0.0286 | 0.068 |
  | ×3 | 4.0 ± 0.5 | 4.5 ± 0.9 | 0.022 / 0.088 | 0.116 |

  - Every seed has ×3 > ×2 > 1.7. Nominal h20 errors are 0.0082–0.0094
    (Lagrangian) and 0.0077–0.0086 (MLP).
  - The learned Lagrangian gear contexts show the same near-gauge as the
    mass contexts. `o₀` stays near 0.55 while the gain grows, and
    `(1 + o₀) / exp(g)` matches `1 / scale` within 2.5% (seed 0).
  - So H1b on true state holds for two parameters, inertia and actuation,
    with the same pattern: parity inside the range, a growing advantage
    outside it, and no discrimination on the side where the system moves
    less.

- **Third parameter, gravity: a misspecified slot (prediction written
  2026-10-18, before running):**
  - Setup: as the gear test, varying gravity (training ×1.0, 0.7, 0.85,
    1.2, 1.4; tests ×1.3, ×0.5, ×2, ×3). True state, seeds 0–2.
  - The Lagrangian's context has slots for the two diagonal mass entries
    and the actuation gain, but none for the potential. Scaling gravity
    by s is equivalent to scaling the whole of `M` and `B` by 1/s. The
    diagonal offsets cannot scale the coupling `M₀₁`, so the slots can
    only approximate a gravity change.
  - Prediction:
    - inside the range (×1.3): parity;
    - outside it, the advantage is smaller than for mass or gear: ratio
      between 0.8 and 1.5 at ×2 and ×3, with the Lagrangian possibly
      worse at ×3.
  - Decision rule:
    - *Ratio ≥ 1.3 at ×2 and ×3:* the near-equivalence is good enough on
      cartpole, and the advantage does not need an exact slot.
    - *Otherwise:* the advantage needs the shifted parameter to have a
      matching slot. Add a potential-scale slot (context dimension 4 for
      both forms) and re-test, and state the slot requirement in the
      claim.

- **Third parameter, gravity: result (2026-10-18, train envs
  `gravity_x*`, seeds 0–2):** the prediction is **refuted, in the
  unfavourable direction**. The Lagrangian without a potential slot is
  much *worse* than the context MLP, even inside the training range.

  | shift | ratio k1 | ratio k5 | lag_ctx / mlp_ctx h20 (k1) | zero-shot |
  |---|---|---|---|---|
  | ×1.3 (inside) | 0.32 ± 0.08 | 0.54 ± 0.03 | 0.024 / 0.007 | 0.036 |
  | ×0.5 | 0.40 ± 0.12 | 0.31 ± 0.03 | 0.020 / 0.008 | 0.083 |
  | ×2 | 0.46 ± 0.12 | 0.51 ± 0.03 | 0.033 / 0.014 | 0.094 |
  | ×3 | 1.00 ± 0.21 | 1.05 ± 0.20 | 0.044 / 0.042 | 0.128 |

  - The near-equivalence (gravity ×s ≈ `M` and `B` ×1/s) is not good
    enough: the diagonal offsets cannot reproduce it. A misspecified
    slot costs 2–3× in error.
  - **Consequence for the claim:** the advantage requires the shifted
    parameter to enter through a matching slot. Structure helps where it
    is specified correctly and hurts where it is not. The paper's claim
    has to name the slot set (mass, gain, potential scale), and the
    choice of slots becomes part of the method.
- **Potential-scale slot (prediction written 2026-10-18, before
  running):**
  - Change: both forms get a 4-number context
    (`--potential-slot`). The Lagrangian's fourth slot is a log scale on
    `V`, which represents a gravity change exactly; the MLP gets one more
    context input.
  - Runs: (a) the gravity test again; (b) the cart-mass dose test (×0.5,
    ×0.4, ×0.3) with the extra slot, to check that a redundant slot does
    not erode the mass result. Seeds 0–2.
  - Prediction:
    - (a) parity at ×1.3, ratio ≥ 1.3 at ×2 and ×3. ×0.5 near parity (a
      weaker gravity means slower motion).
    - (b) mass ratios within 25% of the 3-slot results (3.8 mean at k1
      in the same-protocol zero-noise run).
  - Decision rule:
    - *(a) and (b) hold:* a Lagrangian with a full physical slot set
      (inertia, actuation, potential) extrapolates in every parameter it
      covers. This becomes the H1b model for pixels.
    - *(a) fails:* the potential slot is not identifiable enough from
      random-action data. Inspect the learned contexts for a gauge, as
      with Surprise 1.

- **Potential-scale slot: result (2026-10-19, `--potential-slot`,
  seeds 0–2):** (a) mostly confirmed, (b) confirmed.

  | shift | ratio k1 | ratio k5 | lag_ctx / mlp_ctx h20 (k1) |
  |---|---|---|---|
  | gravity ×1.3 (inside) | 0.86 ± 0.10 | 1.01 ± 0.04 | 0.0085 / 0.0071 |
  | gravity ×0.5 | 1.22 ± 0.27 | 1.21 ± 0.14 | 0.0060 / 0.0071 |
  | gravity ×2 | 1.68 ± 0.53 | 1.21 ± 0.01 | 0.0093 / 0.0157 |
  | gravity ×3 | 5.3 ± 0.8 | 4.8 ± 0.4 | 0.0100 / 0.053 |
  | cart mass ×0.5 | 1.89 ± 0.35 | 2.69 ± 0.81 | 0.016 / 0.029 |
  | cart mass ×0.4 | 3.7 ± 0.9 | 3.9 ± 0.5 | 0.016 / 0.057 |
  | cart mass ×0.3 | 6.8 ± 1.9 | 7.0 ± 0.5 | 0.015 / 0.096 |

  - Gravity:
    - The Lagrangian adapts to 0.006–0.010 at every gravity, near its
      nominal error (0.009). Without the slot it was at 0.020–0.044.
    - ×2 at k5 (1.21) misses the 1.3 bar narrowly. Everything else
      matches the prediction.
    - The learned potential slot tracks log(scale): −0.30, −0.11, 0.23,
      0.39 against −0.36, −0.16, 0.18, 0.34 (seed 0). The mass and gain
      slots sit at a common offset (0.63, 0.54), the same near-gauge as in
      Surprise 1.
  - Mass with the extra slot: mean k1 ratio 4.1 against 3.8 with three
    slots, so the redundant slot does not erode the result.
  - **Decision (written rule):** a Lagrangian with a full physical slot
    set (inertia, actuation, potential) extrapolates in every parameter
    it covers, and a missing slot makes it worse than an MLP. The
    four-slot model is the H1b model for pixels.

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
- **Rung 2b, frozen renderer (prediction written 2026-10-18, before
  running; `scripts/oracles/frozen_renderer_oracle.py`):**
  - Setup: fit the geometric renderer from true poses (3000 steps), freeze
    it, then train an encoder from reconstruction loss alone (6000 steps,
    cosine schedule). Both keypoint and conv encoders. Poses are scored in
    metric units, because the frozen renderer fixes the gauge.
  - Prediction:
    - x is learned (correlation > 0.95): the cart blob is large and
      overlaps its target from most starting points.
    - θ fails for many frames: more than 20% flipped by over 90°, and a
      median error on in-view frames above 3° (about 60% confidence).
      Mechanism: when the rendered pole does not overlap the real one,
      the reconstruction gradient with respect to θ vanishes, so frames
      that start in the wrong basin stay there.
  - Decision rule:
    - *Flips dominate:* add a coarse-to-fine curriculum (wide blobs or
      blurred targets early) or KeyCLD's heatmap loss, and re-test this
      rung.
    - *Works (median ≤ 3°, few flips):* the end-to-end failure comes from
      joint co-adaptation. Warm-start the renderer, or alternate the
      updates.
- **Rung 2b result (2026-10-18, `runs/oracles/frozen_renderer_oracle_*`):**
  the prediction is **confirmed**, and more starkly than predicted.
  - Keypoint encoder: θ median error 157° on in-view frames (hanging
    160°, sideways 111°, upright 25°), and x fails too (55 px rms).
  - Conv encoder: x is learned (0.15 px rms on hanging frames), but θ
    sits near upright for every frame (hanging 163°, upright 50°).
  - So the encoder's angle stays where it started, upright: there the
    rendered pole never overlaps the hanging real pole, and the gradient
    with respect to θ is zero. The renderer is not at fault (rung 1).
  - Per the rule: a coarse-to-fine loss. Average-pooled copies of the
    rendered and target images (pool 8 and 24) are added to the loss.
    - Prediction: flips (error over 90°) fall below 10%, and the median
      error on in-view frames is ≤ 3° for the conv encoder (about 55%).
    - If that fails, try KeyCLD's heatmap loss.
- **Coarse-to-fine result and diagnosis (2026-10-18):**
  - Pooled loss (pool 8 and 24): still fails. Flips are 85% (keypoint)
    and 92% (conv); conv x stays good (0.85 px), and θ stays near
    upright.
  - Discriminating check (scratchpad `render_sens.py`): the frozen
    renderer *is* θ-sensitive, with loss 0.00039 at the true θ against
    0.0037 at +10°. But the loss is **flat beyond about 10°** (+10°:
    0.0037, +45°: 0.0041, +90°: 0.0037, +180°: 0.0036). At θ = 0.3 for
    hanging targets the gradient points towards the truth for only 22%
    of frames (35% with pooling), and its magnitude is about 1e-5.
  - Conclusion: a sharp global minimum with a flat landscape around it.
    Gradient descent cannot find it, and pooling the whole image does not
    widen the basin, because the thin pole is a tiny fraction of the
    pooled pixels.
  - Next (cheap): render-and-compare. Grid-search θ (and x) through the
    frozen renderer per frame, then use the result as a pseudo-label for
    the encoder (analysis by synthesis, still label-free). The search
    alone is being tested now (scratchpad `render_search.py`).
- **Render-and-compare result (2026-10-19, scratchpad
  `render_search.py`, 1010 held-out frames, 173 s on MPS):** works.
  - Search: x on a 0.05 m grid with 8 coarse angles, then θ on a 5° grid
    at that x, then a local 7×7 refinement. No encoder is involved.

    | frames | θ median | θ rms | x rms |
    |---|---|---|---|
    | in view, all | 0.54° | 1.07° | 0.33 px |
    | in view, hanging | 0.53° | | |
    | in view, sideways | 0.51° | | |
    | in view, upright | 0.96° | | |
    | out of view (median) | 1.8° | | |

    2% of frames are flipped, mostly out-of-view ones.
  - That is better than both supervised encoders, especially on upright
    poses (0.96° here, 4.4° keypoint, 17° conv).
  - **Caveat:** the frozen renderer was fitted on true poses (rung 1).
    This shows the reconstruction likelihood has the right global
    minimum and that search finds it. It does not yet show a label-free
    front end.
  - Next for A-FE-1:
    - a label-free bootstrap that alternates render-and-compare pose
      search with renderer fitting on the searched poses (EM), starting
      from the renderer's analytic geometry;
    - amortize the search into an encoder trained on the searched poses,
      because search costs 0.17 s per frame.
- **Label-free bootstrap (prediction written 2026-10-19, before running;
  `scripts/oracles/bootstrap_oracle.py`):**
  - Step 0, classical CV with no labels:
    - background = per-pixel median over many frames (the camera is
      fixed);
    - foreground = pixels far from the background;
    - cart column = median foreground column in the cart's row band;
    - θ = direction from the pivot to the centroid of the foreground
      pixels outside that band.
    - Scene knowledge used: the pivot row, and 25.35 px/m for reporting
      x in metres. No pose labels.
  - Steps 1–3, EM:
    - fit the geometric renderer on the current poses;
    - re-estimate the poses of a training subset by render-and-compare;
    - repeat twice.
  - Score: held-out in-view frames, poses by search through the final
    renderer.
  - Prediction:
    - the step-0 heuristic reaches a median θ error ≤ 3° in view, worse
      for upright and hanging poses where the pole crosses the cart band;
    - after EM the median is ≤ 1° and the rms ≤ 3°, close to the
      true-pose renderer's 0.54° / 1.07°.
  - Decision rule:
    - *EM ≤ 1° median:* a label-free front end with supervised-level
      accuracy exists. Amortize it (encoder on searched poses) and run
      the encoder-pose context oracle with it.
    - *Heuristic good, EM no better:* the heuristic alone is the front
      end.
    - *Both poor:* back to keypoint losses.
- **Step-0 heuristic result (2026-10-19, 1010 held-out frames):**
  confirmed, and better than predicted. A pose is found in 79% of frames
  (the rest have the cart out of view). On in-view frames:

  | | heuristic, no labels | supervised keypoint encoder | supervised conv encoder |
  |---|---|---|---|
  | θ median | 1.24° | 0.71° | 0.21° |
  | θ rms | 1.87° | 1.55° | 3.7° |
  | upright, median | 2.2° | 4.4° | 17° |
  | x rms | 0.60 px | 0.48 px | 0.67 px |

  The heuristic is a colour mask plus moments, with no learning, and it
  already matches the supervised keypoint encoder.
- **EM result (2026-10-19, `runs/oracles/bootstrap_oracle/20261019-004002.json`,
  two iterations on 3000 frames, 31 min on MPS):** "heuristic good, EM no
  better".
  - EM search on in-view frames: θ median 1.27°, rms 1.86°, x rms 0.60 px.
    Sideways rms improves (1.05° vs 13° for the heuristic where it
    misfires), and every frame gets a pose.
  - Median accuracy does not improve. A renderer fitted on heuristic
    poses learns the heuristic's systematic bias, and EM converges to
    it. With a renderer fitted on true poses, search reached 0.54°.
  - Per the rule, the heuristic (or EM, for coverage) is the label-free
    front end. Getting below 1° needs an independent constraint on the
    bias, such as the dynamics themselves (joint fitting) or a known
    object dimension.
- **Heuristic front end in the context oracle (prediction written
  2026-10-19, before running):**
  - Setup: every training and shift frame is rendered and read by the
    heuristic (`--front-end heuristic`). Valid frames are those where the
    heuristic finds the system, about 76–79%, an observable criterion.
    Central velocities, `shoot` with 8 frames, seeds 0–2.
  - Prediction: the ratio falls between the keypoint encoder's (1.2) and
    the 0.25 px Gaussian result (1.67). The heuristic's errors are
    moment biases where the pole is clipped or crosses the cart band,
    again smooth in the pose. Estimate: about 1.3 (50%).
  - Decision rule: *≥ 1.3*: a fully label-free pixel pipeline (heuristic
    poses, then structured dynamics with model-based start states)
    preserves H1b on cartpole. *< 1.3*: same diagnosis path as the
    encoders (bias removal).
- **Heuristic front-end result (2026-10-19):** **refuted, badly.**
  - Nominal h20: Lagrangian 0.079 / 0.166 / 0.082 (seeds 0–2), MLP about
    0.054.
  - True-start ratio 0.54 ± 0.12 and `shoot` 0.57. The true-dynamics
    `shoot` floor (0.026) is similar to the encoders' floors.
  - Here valid means "found" (76–81% of frames). That includes partly
    clipped frames, where the heuristic's error is larger (median 5° out
    of view).
  - Pattern across all real front ends: the Lagrangian's nominal fit
    degrades much more than the MLP's (keypoint 0.035 vs 0.037, conv
    0.05 vs 0.024, heuristic 0.08–0.17 vs 0.054). Under Gaussian noise of
    a similar rms it was the *more* robust of the two (0.018 vs 0.050 at
    0.25 px). So something non-Gaussian in real pose errors breaks the
    structured model. The outlier-rejection test addresses this, and it
    is now also run on the heuristic poses.

### A-ENC-1 — CNN regresses the state
- **Status:** supported for x; partial for θ.
- **Evidence:**
  - Rung 2, conv encoder, 1500 steps: held-out R² x 0.995, cos θ 0.57,
    sin θ 0.88.
  - cos θ has little variance (the pole mostly hangs), so its R² is a
    harsh score.
  - RSSM features give pose R² 0.93 linearly.
- **To do:** longer training and an angle-error metric in degrees.
- **Angle errors in degrees (2026-10-18, `runs/oracles/logs/encoder_deg.log`,
  held-out episodes, constant lr 1e-3):**

  | encoder, steps | θ median | θ rms | θ p90 | x median | x rms (px) |
  |---|---|---|---|---|---|
  | keypoint, 1500 | 1.9° | 25.8° | 19.3° | 0.034 m | 1.5 |
  | keypoint, 20000 | 7.5° | 33.2° | 49.4° | 0.050 m | 2.7 |
  | conv, 1500 | 1.8° | 23.6° | 18.2° | 0.025 m | 2.1 |
  | conv, 20000 | 1.0° | 20.8° | 13.4° | 0.010 m | 1.0 |

  - Typical frames are good (conv median 1.0°, about 0.4 px at the tip),
    but a heavy tail dominates the rms. Keypoint training is unstable at
    constant lr (loss spikes after step 4000), so it got worse.
  - Prediction for the tail (written before testing): the camera sees
    only ±1.66 m of the ±1.8 m rail (25.35 px/m, 84 px), and a
    horizontal pole reaches 1 m beyond the cart. Gross errors concentrate
    in frames where the cart or the pole tip is out of view. On fully
    visible frames the conv encoder's rms θ error is below 3°.
    Alternatives: errors concentrate at particular angles (pole crossing
    the cart or the rail), or at high angular speed (no blur in MuJoCo,
    so unlikely).
  - Test: split the held-out errors by visibility and by angle; add a
    cosine learning-rate decay to fix the keypoint instability.
- **Result (2026-10-18, `runs/oracles/logs/encoder_cosine.log`, 20000
  steps, cosine decay):** the prediction is **partly refuted**. The tail
  has two sources, and visibility is only one of them.

  | frames (share of held-out) | conv θ median / rms | keypoint θ median / rms | conv / keypoint x rms (px) |
  |---|---|---|---|
  | in view, hanging (52%) | 0.19° / 0.31° | 0.93° / 1.42° | 0.11 / 0.37 |
  | in view, sideways (10%) | 0.52° / 26.8° | 1.15° / 1.45° | 2.9 / 0.60 |
  | in view, upright (1.5%) | 154° / 137° | 101° / 83° | 5.6 / 1.2 |
  | cart or tip out of view (37%) | 0.88° / 21.0° | 2.9° / 21.3° | 0.67 / 1.17 |

  - Overall held-out R²: conv 0.955, keypoint 0.961; both now pass the
    0.95 criterion. Cosine decay removed the keypoint instability.
  - Typical poses are localized at 0.1–0.4 px at the pole tip, which is
    inside the noise range the noisy-state test covers.
  - **Upright poses are wrong by 100–150°.** They are 1.5% of
    random-policy frames, so both encoders regress them towards
    "hanging". This is a data-coverage failure (see A-DATA-1). The
    keypoint geometry does not rescue it.
  - **In 37% of frames the cart or the tip is outside the image.** The
    camera sees ±1.66 m, the rail is ±1.8 m, and the pole reaches 1 m
    beyond the cart. Over the whole training set, 26% of frames have the
    cart itself out of view. Every pixel model trained on this data
    (April runs included) was asked to predict invisible states.
  - Consequences for the pixel version of H1b:
    - restrict windows to fully visible frames, as the rail-free
      protocol does;
    - if upright states matter, collect data with swing-up excitation.

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

### A-VIEW-1 — the camera sees the system
- **Status:** refuted (2026-10-18).
- **Evidence:**
  - Camera 0 (4 m away, fovy 45°, 84 px) gives 25.35 px/m and sees
    x ∈ ±1.66 m. The rail limit is ±1.8 m.
  - On the nominal training set (`data/cartpole-swingup-random-100k-v2`),
    the cart centre is out of view in 26% of frames. The cart or the pole
    tip (2 px margin) is out of view in 35% (65% fully visible); the
    encoder oracle's held-out frames give 37%.
  - The encoder oracle's errors on those frames have an rms of 21°.
- **Implication:**
  - Pixel losses and probes average over frames whose state is
    invisible.
  - Fix in evaluation: restrict to fully visible windows (`oracles.encoder.in_view`).
  - Fix in data: a wider field of view, a tracking camera, or a policy
    that stays away from the rail.

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

### A-ACT-1 — a learned Lagrangian has physically meaningful terms
- **Status:** refuted for unconstrained actuation; supported with
  `actuated_dims=(0,)` (2026-10-18, `runs/oracles/mass_gauge/`).
- **Statement:** after fitting nominal data, the learned `M`, `V`, `B` equal
  the true ones up to a common scale, so a physical change is a change of
  the matching term. H1b's `lag_low` relies on this.
- **Evidence:**
  - Setup: seed 0 of dynamics oracle v2, unconstrained `B(q)`; comparison
    script in the session scratchpad, to be promoted if reused.
  - The learned `M` fits a scaled true `M` with relative residual 0.91.
    `M[0,0] ≈ 10` (true 1.1 × about 9) and `M[0,1]` has the right shape
    (correlation 0.998). But `M[1,1] ≈ 22–26`, against 0.3 for the scaled
    true value.
  - The pole row of the learned `B` drives the pole directly, imitating
    the inertial reaction to cart acceleration. Only `M⁻¹B` is identified,
    so nothing in the loss prevents this.
  - Consequence: `lag_low` fitted on *all* shifted data plateaus (cart
    mass ×0.5: h20 0.0175 against nominal 0.008). The result does not
    depend on the learning rate, so it is representational, not data or
    optimization.
- **Fix under test:** `Actuation(actuated_dims=(0,))`, so the input enters
  only the cart equation (the true cartpole input matrix is `[gear, 0]`).
  This is a mild prior: the actuated coordinate is known, which is standard
  for underactuated systems. In pixels it requires the encoder's first
  coordinate to be the cart (true for the keypoint front end).

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
