# Claims Ledger

What the project currently holds to be true, graded by the strength of its
evidence. This is the file to read for "what do we know". Narrative and
history live in the experiment records
([`experiments/`](experiments/), [`experiments/archive/`](experiments/archive/)).
`scripts/research/check.py` enforces the grade rules.

| Grade | Meaning |
|---|---|
| **A** | Registered experiment (`E-NNN`), clean-tree runs in [`../results/ledger.jsonl`](../results/ledger.jsonl), ≥ 3 seeds, generated table, bracketed by floor and ceiling (and by a known-model baseline where one exists) |
| **B** | Ledgered runs whose recorded commit contains the code they used; narrower evidence (not pre-registered, not bracketed, or 1 seed) |
| **C** | Plausible but not verifiable: run from uncommitted code, scratch scripts, or not ledgered. To upgrade, rerun from a clean tree |

Rules:
- one claim per entry, with its scope stated;
- cite ledger run IDs (`name/YYYYMMDD-HHMMSS`);
- when a claim is refuted or narrowed, keep the entry and mark it **superseded by C-NNN**;
- numbers here are copied from generated reports, never typed from memory.

Last full review: 2026-10-19 (see [`review-2026-10-19.md`](review-2026-10-19.md); no claim is grade A yet).

## Closed-loop control (current charter)

### C-015 — MPPI over the true MuJoCo model solves swing-up from true state
- **Grade:** A
- **Claim:** MPPI (horizon 75 decisions = 1.5 s, 512 samples, noise std
  0.5, temperature 1.0, rail cost beyond 1.6 m) over `mujoco.rollout` with
  the closed-form swing-up reward reaches a dm_control cartpole-swingup
  return of 861 ± 3 (5 episodes, sample std); the pole is upright for the
  whole second half of every episode; 84 ms per decision on 4 CPU threads.
  Floors: zero action 0.005, random 64 ± 38.
- **Scope:** nominal parameters, true state, action repeat 2.
- **Evidence:** `control_episode/20261019-094818`, `control_episode/20261019-094818-1`, `control_episode/20261019-094817`, `control_episode/20261019-094858-1`, `control_episode/20261019-094858`
- **Hypotheses:** H-PLAN

### C-016 — MPPI over a learned Lagrangian or a learned MLP reaches the true-model ceiling on cartpole
- **Grade:** A
- **Claim:** with the E-006 planner, models trained by multi-step rollout
  regression on true states reach, over 3 training seeds × 5 episodes:
  - from 60 random-policy episodes (round 0): MLP 842 ± 3, Lagrangian
    809 ± 70 (seeds 851 / 728 / 847);
  - after 1–3 rounds of 10 on-policy episodes: MLP 853 ± 2, Lagrangian
    856–857 (round 3: 856 ± 3), against 861 ± 3 for the true model (C-015).
  The Lagrangian costs 0.21 s per planning decision against 0.037 s, and
  about 490 s of training against 54 s.
- **Scope:** nominal parameters, true state, `actuated_dims=(0,)`,
  15k training steps, 8-step windows.
- **Evidence:** `learned_control/20261019-102526`, `learned_control/20261019-102520`, `learned_control/20261019-102525`, `learned_control/20261019-121303`, `learned_control/20261019-121227`, `learned_control/20261019-121254`, `learned_control/20261019-101041`, `learned_control/20261019-103007`, `learned_control/20261019-104930`, `learned_control/20261019-102732`, `learned_control/20261019-104654`, `learned_control/20261019-110928`
- **Hypotheses:** H-LAG-CTRL, H-LAG-DATA

### C-017 — A keypoint reader trained on renders of uniformly sampled states reads the pose to 0.2°
- **Grade:** B (one training seed)
- **Claim:** a `KeypointEncoder` (spatial softmax) supervised with the
  projected cart and pole-tip positions of 50k frames rendered at uniform
  states (x in ±1.8 m, θ uniform), camera 0 widened to 71.9° (14.5 px/m at
  84 px), reads θ to 0.18° median (0.19° upright, 0.45° p90, 1.1° max) and
  the cart to 0.04 px median on 5000 held-out frames; (x, θ) comes from the
  keypoints by the camera geometry alone. 20k steps take 10.5 min on MPS.
- **Evidence:** `keypoint_reader/20261019-111217`
- **Hypotheses:** H-KP

### C-018 — Camera images → keypoints → fixed-lag estimator → learned Lagrangian → MPPI swings up and balances cartpole
- **Grade:** A
- **Claim:** with the C-017 reader, a 12-frame fixed-lag estimator through
  the planning model, and the E-006 planner:
  - a Lagrangian learned **only from camera poses** (60 random episodes plus
    3 rounds of 10 own camera episodes) returns 836 ± 9 (3 training seeds ×
    5 episodes), 0.977 of the true-state learned Lagrangian (856 ± 3);
  - the true-state Lagrangians driven from camera images return 839–840;
    the true model driven from camera images 840 ± 4; finite-difference
    velocities instead of the estimator 788 ± 22.
  - A planning decision takes about 0.3 s on the laptop CPU.
- **Scope:** nominal parameters, simulated camera (no lighting or
  background changes), single pole.
- **Evidence:** `learned_control/20261019-141842`, `learned_control/20261019-144851`, `learned_control/20261019-144836`, `camera_control/20261019-113017`, `camera_control/20261019-113108`, `camera_control/20261019-113200`, `camera_control/20261019-113251`, `camera_control/20261019-113341`, `camera_control/20261019-121608`, `camera_control/20261019-122850`, `camera_control/20261019-124131`, `camera_control/20261019-130003`
- **Hypotheses:** H-CAM-LAG, H-KP

### C-019 — On the two-pole cartpole, MPPI over the true model swings up but does not hold balance
- **Grade:** A
- **Claim:** with horizon 150 decisions (3 s), 512 samples, noise std 0.5,
  temperature 1.0 (best of an 8-setting grid plus the registered fix of 3
  more), MPPI over the true MuJoCo model reaches a dm_control
  `cartpole-two_poles` return of 422 ± 44 (5 episodes, sample std; best
  possible about 1000). Both poles are upright for 16–22% of the frames in
  4 of 5 episodes; they fall and are swung up again. Horizon 75 only finds
  a half-solution (first pole hanging, second balanced on it: about 410).
  Floors: zero action 0.6, random 27 ± 13. 0.36 s per decision on 2 CPU
  threads.
- **Scope:** nominal parameters, true state, action repeat 2, this MPPI
  (one temperature, no planner change beyond the grid).
- **Evidence:** `control_episode/20261019-203551-4`, `control_episode/20261019-203551`, `control_episode/20261019-203551-2`, `control_episode/20261019-203551-3`, `control_episode/20261019-203551-1`
- **Hypotheses:** H-PLAN-2P

## Dynamics on true state (H1b, backlog)

### C-001 — Lagrangian slot context extrapolates in cart mass; the context MLP does not
- **Grade:** B
- **Claim:** both models are trained on cart masses {1, 0.7, 0.85, 1.2, 1.4} with a 3-number context. After a k = 1 adaptation (one 50-frame segment), the mean h20 error ratio of `mlp_ctx` to `lag_ctx` over cart mass ×0.5 and ×2 is 1.61 ± 0.15; at k = 5 it is 1.70 ± 0.31 (3 seeds, sample std). The whole effect is on ×0.5: there `lag_ctx` is at 0.0100 and `mlp_ctx` at 0.0229, while ×2 is at parity.
- **Scope:** cartpole, true state, rail-free windows, `actuated_dims=(0,)`, slots designed to match how mass enters.
- **Evidence:** `context_oracle/20261018-182854`, `context_oracle/20261018-183737`, `context_oracle/20261018-183649`
- **Missing brackets:** known-equation least-squares fit and Cramér–Rao bound (E-004); a structure-matched black box (backlog).
- **Hypotheses:** H1b

### C-002 — Same pattern for actuator gear
- **Grade:** B
- **Claim:** trained on gear {1, 0.7, 0.85, 1.2, 1.4}, the k1 ratio is
  - 1.02 ± 0.16 at gear ×1.3 (inside the range);
  - 1.84 ± 0.12 at ×2;
  - 4.04 ± 0.65 at ×3;
  - 1.15 ± 0.09 at ×0.5 (both models near the floor).
- **Scope:** as C-001.
- **Evidence:** `context_oracle/20261018-220444`, `context_oracle/20261018-220442`, `context_oracle/20261018-220443`
- **Missing brackets:** as C-001.
- **Hypotheses:** H1b

### C-003 — A misspecified slot set makes the Lagrangian worse than the MLP
- **Grade:** B
- **Claim:** with gravity varied and no potential slot, the k1 ratio is 0.32–0.47 at gravity ×1.3, ×0.5 and ×2, and 1.00 ± 0.25 at ×3. The Lagrangian is 2–3× *worse*, even inside the training range.
- **Scope:** as C-001.
- **Evidence:** `context_oracle/20261018-233549`, `context_oracle/20261018-233536`, `context_oracle/20261018-233550`
- **Hypotheses:** H1b

### C-004 — The advantage grows with the extrapolation distance and is absent inside the range
- **Grade:** C (simulated-shift runs: their recorded commit `803ac40` lacks `--sim-shift`)
- **Claim:**
  - interpolation masses ×0.775 and ×1.3 give ratios 0.98 and 1.09 at k1;
  - the light side rises monotonically: ×0.6 1.18, ×0.4 3.1, ×0.3 5.8;
  - the heavy side has sign-inconsistent seeds: ×3 gives 0.64 / 1.33 / 1.54.
- **Evidence:** `context_oracle/20261018-191613`, `context_oracle/20261018-193931` (and their seed siblings)
- **Hypotheses:** H1b

### C-005 — A potential-scale slot repairs gravity
- **Grade:** C (recorded commit `1cd6669` lacks `--potential-slot`)
- **Claim:** with a 4th slot, the gravity k1 ratios are 0.86 (×1.3), 1.22 (×0.5), 1.68 (×2, seeds 1.28–2.43) and 5.25 (×3). The cart-mass result is preserved.
- **Evidence:** `context_oracle/20261019-000539`, `context_oracle/20261019-000548`, `context_oracle/20261019-000543`
- **Hypotheses:** H1b

### C-006 — With unconstrained actuation the learned M is not physical; with the actuated coordinate fixed it is, up to scale
- **Grade:** C (recorded commit lacks `src/oracles/gauge.py`; the comparison script was a scratch file)
- **Claim:**
  - unconstrained `B(q)` gives an M residual of 0.91–0.93, with M11 inflated 71–80×;
  - `actuated_dims=(0,)` gives a residual of 0.047;
  - the mass and gain slots form a near-gauge: (1 + o₀)/exp(g) matches the true inertia ratio within 3%.
- **Evidence:** `mass_gauge/20261018-171337`; archived log, A-ACT-1 and "Surprise 1"
- **Hypotheses:** A-ACT-1

### C-007 — Rail contacts and pole mass confound naive tests
- **Grade:** C (recorded commits lack the oracle modules)
- **Claim:**
  - about 30% of 20-step random-policy windows touch the ±1.8 m rail, where even the true smooth dynamics err (h20 0.031–0.037, against 0.002–0.007 rail-free);
  - pole mass ×2 moves trajectories by about 0.002 at h20, while cart mass ×2 moves them by 0.027.
- **Evidence:** `dynamics_ceiling/20261018-162020`, `intervention_effect/20261018-160646`
- **Hypotheses:** A-OOD-2, A-OOD-3

## Perception and estimation (Q2)

### C-008 — Under Gaussian pose noise the Lagrangian fits better than the MLP; under heavy tails it is more fragile
- **Grade:** C (noise, outlier and Huber runs from uncommitted code)
- **Claim:**
  - with 0.25 px Gaussian noise and central-difference training velocities, the nominal h20 error is 0.018 (Lagrangian) against 0.050 (MLP);
  - adding 2% heavy outliers to 0.08 px noise multiplies the Lagrangian's nominal error 5.4× and the MLP's 2.5×;
  - Huber restores the Lagrangian on synthetic independent outliers but not on the heuristic front end.
- **Mechanism (unbounded q̇² Coriolis terms):** untested; E-005.
- **Evidence:** `context_oracle/20261018-214018` and siblings; `context_oracle/20261019-010320` and siblings; `context_oracle/20261019-020839` and siblings
- **Hypotheses:** H1b

### C-009 — The start-state estimate dominates the error under noise; model-based shooting fixes most of it
- **Grade:** C (uncommitted code)
- **Claim:**
  - with true dynamics and 0.25 px noise, a two-frame backward difference gives an h20 floor of 0.14–0.15;
  - shooting over 16 context frames gives 0.007, and over 8 frames 0.015.
- **Evidence:** `context_oracle/20261018-214237` and siblings (start_context 16); archived log, "Start-state estimation"
- **Hypotheses:** H1b

### C-010 — With real pose front ends the advantage shrinks to about 1.2–1.5× and needs outlier rejection
- **Grade:** C (uncommitted code)
- **Claim:** k1 `shoot` ratios:

  | front end | ratio |
  |---|---|
  | conv encoder + jump rejection | 1.50 (seeds 1.09–2.23) |
  | keypoint encoder | 1.1–1.2 |
  | label-free heuristic + floored rejection | 1.16 |
  | heuristic, no rejection | 0.54–0.65 |

- **Evidence:** `context_oracle/20261019-005419`, `context_oracle/20261019-013245` and siblings
- **Hypotheses:** H1b, A-FE-1

### C-011 — A label-free colour/moment heuristic reads the pose about as well as a supervised keypoint encoder
- **Grade:** C (recorded commit lacks `bootstrap.py`)
- **Claim:**
  - on in-view held-out frames: θ median 1.24°, rms 1.87°, x rms 0.60 px (keypoint encoder: 0.71°, 1.55°, 0.48 px);
  - EM through a fitted renderer does not reduce the median.
- **Evidence:** `bootstrap_oracle/20261019-004002`
- **Hypotheses:** A-FE-1

### C-012 — Gradient descent through a correct renderer cannot find the pole angle; render-and-compare search can
- **Grade:** C (scratch scripts `render_sens.py`, `render_search.py`)
- **Claim:**
  - the reconstruction loss is flat beyond about 10° of angle error;
  - grid search through a renderer fitted on true poses reaches a 0.54° median.
- **Evidence:** `frozen_renderer_oracle_conv/20261018-231834`; archived log, A-FE-1
- **Hypotheses:** A-FE-1

### C-013 — The camera does not see the system in about a third of frames
- **Grade:** C (computed in a session, not ledgered)
- **Claim:**
  - the cart centre is out of view in 26% of frames of `cartpole-swingup-random-100k-v2`;
  - the cart or the pole tip is out of view in 35%;
  - 25.35 px/m.
- **Evidence:** archived log, A-VIEW-1
- **Hypotheses:** A-VIEW-1

## End-to-end pixel world models (frozen)

### C-014 — The April mechanics world model stored the pose in the nuisance channel and lost to the RSSM
- **Grade:** C (eval outputs, not ledgered)
- **Claim:**
  - on 256 validation windows the pose probe R² is 0.89 from `z` and 0.53 from `q`;
  - the RSSM beats the mechanics model's foreground open-loop MSE at every horizon (h20: 0.0092 against 0.0129).
- **Evidence:** `eval/review-2026-10/`; archived status log, "October 2026 Review"
- **Hypotheses:** A-ROUTE-1, D-K1
