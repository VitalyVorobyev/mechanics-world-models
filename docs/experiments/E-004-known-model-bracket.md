# E-004 — Bracket H1b with the known cartpole equations and the Cramér–Rao bound

- **Status:** abandoned (never run; superseded by the 2026-10-19 re-charter, see `docs/charter.md`)
- **Question node:** Q4.1
- **Hypotheses:** H1b, A-KNOWN-1, A-CRB-1
- **Budget:** 5 agent-hours, 3 CPU-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** — (owner, date)

## Question
C-001 compares the learned slot Lagrangian only with a context MLP. On
cartpole the equations are known, so the natural upper bracket is the
known model with its physical parameters fitted from the same k segments.
How far is `lag_ctx` from that bracket? And does the comparison change when
the known model is slightly wrong, as real hand-written models are? This
decides whether the learned Lagrangian has any role in the controller
pipeline (Q0), or whether the known model plus a parameter fit is enough.
Lower priority than E-001…E-003: run it only if the owner wants the H1b
line settled, or if the known model turns out to be inadequate in E-001.

## Prediction
- **Sanity:** `known_exact` with the true parameters reproduces the
  context oracle's `floor_true` (the same MuJoCo dynamics) to below 1e-4
  at h20. Otherwise the implementation is wrong. Confidence 95%.
- **True state, k = 1, cart mass ×0.5 / ×0.4 / ×0.3:**
  - `known_exact` (cart mass, pole mass and gear fitted by least squares
    on the k segments, everything else nominal) reaches h20 ≤ 0.002. It
    is limited only by the identifiability of three parameters from one
    50-frame random-action segment. Confidence 85%.
  - `lag_ctx` stays at its C-001 level (about 0.010–0.016).
  - So `lag_ctx / known_exact` ≥ 3. Confidence 80%.
- **Misspecified known model** (`known_pointmass`: the pole is a point
  mass at its centre of mass, so the rod's own inertia I = m l²/3 about
  the centre is dropped; this is the classic textbook simplification):
  - its nominal h20 error is above `lag_ctx`'s, because the omitted
    inertia changes the pole's natural frequency by about 15%;
  - after adaptation it is better than `lag_ctx` on ×0.3, because the
    mass term still enters correctly;
  - `known_pointmass / lag_ctx` at k1 is between 0.5 and 1.5.
    Confidence 50%; this is the informative arm.
- **0.25 px Gaussian pose noise, shooting start (8 frames):** the RMSE of
  the cart-mass estimate from `known_exact` over test segments is within
  1.5× of the Cramér–Rao bound computed from the model's sensitivity
  Jacobian. The reason: least squares is efficient for Gaussian noise when
  the start state is fitted jointly. Confidence 65%.

## Primary metric
Ratio of `lag_ctx` to `known_exact` h20 pose error at k = 1, true state,
computed per seed and averaged over cart mass ×0.5, ×0.4 and ×0.3, then
over seeds 0–2.

Report: `lag_ctx_over_known_exact_ratio_k1_h20`, `{shift}.k1.known_exact.dev_h20`, `{shift}.k1.known_pointmass.dev_h20`, `{shift}.k1.lag_ctx.dev_h20`, `{shift}.k1.mlp_ctx.dev_h20`, `noisy.cart_mass_rmse_over_crb`

## Baselines
- **Floor:** k = 0 for every arm (no adaptation).
- **Black box:** `mlp_ctx` (C-001 protocol).
- **Known model:** `known_exact` (MuJoCo cartpole with its masses and gear
  exposed) and `known_pointmass` (misspecified).
- **Bound / ceiling:** `floor_true` (true parameters); the Cramér–Rao
  bound for the cart mass under 0.25 px noise.
- **Effort symmetry:**
  - the known models adapt 3 parameters, the same count as `lag_ctx`;
  - they fit them with Levenberg–Marquardt, a stronger optimizer than
    Adam. This is an asymmetry in their favour, and it is accepted
    because the known model is the bracket, not a competitor;
  - the learned arms keep the context-oracle grids
    (4 learning rates × 6 checkpoints).

## Decision rule
- **`lag_ctx / known_exact` ≥ 2 and `known_pointmass` ≤ `lag_ctx` at
  k1:** learned structure buys nothing on cartpole when the equations are
  approximately known.
  - Narrow H1b to "the learned Lagrangian approaches a known model only
    when …" and **stop** developing the learned Lagrangian as the tool's
    dynamics.
  - The controller pipeline (E-001…E-003) keeps the known model.
  - Threshold reason: a 2× error is well outside the seed spread (sample
    std of about 10–20%).
- **`lag_ctx / known_exact` ≥ 2 but `known_pointmass` > 1.3 × `lag_ctx`:**
  the learned model beats a plausibly wrong hand model.
  - Keep both as candidate controller models.
  - Register a follow-up on the hybrid question: known model plus learned
    residual, against the Wehenkel et al. 2023 expert-augmentation
    baseline.
- **`lag_ctx / known_exact` < 1.5:** the learned Lagrangian is as good as
  the true equations. This is surprising; check the sanity criterion
  first. If it holds, it is a grade-A claim candidate once a structure-matched
  black box (the parameter entering where it physically enters) is
  tested.
- **CRB ratio > 2:** the estimator is inefficient, so fix it (E-005)
  before any claim based on pixels.
- **Stop:** if the sanity criterion fails and cannot be fixed within the
  budget, stop E-001 and report it. Do not run the comparison on an
  unverified known model.

## Prerequisites
- Context-oracle protocol (C-001): `context_oracle/20261018-182854` and siblings.
- New code, committed and tested before any run:
  - `src/oracles/known_model.py`: a parameterized MuJoCo rollout plus an
    LM fit over the selected parameters;
  - a CRB helper (finite-difference sensitivity Jacobian);
  - `known_*` modes in `context_oracle.py`.
- Retrain `lag_ctx` and `mlp_ctx` from a clean tree. The cached models
  came from uncommitted code (C-004) and are not reused.

## Plan
- `.venv/bin/python scripts/oracles/context_oracle.py --experiment E-004 --sim-shift … --modes lag_ctx mlp_ctx known_exact known_pointmass --seed {0,1,2}`
- The same with `--pose-noise 0.01 0.5 --start-context 8`.
- About 45 min CPU per seed. The three seeds run in parallel.

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update

Abandoned 2026-10-19 before any run: the owner clarified that the aim is to
*learn* the mechanics (cartpole as a showcase), not to control through known
equations with parameter refits. Kept for reference; see the charter backlog.
