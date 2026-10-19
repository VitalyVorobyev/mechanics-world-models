# E-005 — Robust errors-in-variables estimation under real front-end errors (conditional)

- **Status:** abandoned (never run; superseded by the 2026-10-19 re-charter, see `docs/charter.md`)
- **Question node:** Q5
- **Hypotheses:** H-FRAG, H1b-px
- **Budget:** 8 agent-hours, 6 CPU-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** — (owner, date); **run only if** E-002 or E-003 point to pose outliers or start-state errors

## Question
Real front ends produce heavy-tailed, run-correlated errors. These broke
the learned Lagrangian more than the MLP (C-008, C-010), and frame
rejection plus Huber only partly fixed it. Two questions:
- Does a robust joint estimator recover the clean-pose accuracy? It is a
  Student-t likelihood over a window, with the states and parameters
  estimated together and no finite-difference velocities.
- Is the Lagrangian's fragility caused by the unbounded q̇² Coriolis terms
  (H-FRAG)?

## Prediction
- With the heuristic front end and no frame rejection, the Student-t
  joint estimator brings the known-model cart-mass estimate to within 1.5×
  of its error with Gaussian noise of the inlier size. Least squares with
  rejection stays about 2× worse. Confidence 50%.
- **H-FRAG:** bounding the velocity inside the Coriolis term
  (s · tanh(q̇ / s), with s at the 99th percentile of training speed)
  cuts the learned Lagrangian's nominal-error inflation under 2% heavy
  outliers. Today the inflation is 5.4×; with the bound it is 2.5× or
  less, matching the MLP. Confidence 45%. The alternative: the fragility
  comes from the fit through the slots, not the dynamics.

## Primary metric
The ratio of the cart-mass estimate RMSE with heuristic poses (Student-t
joint estimator) to the RMSE with true poses plus inlier-size Gaussian
noise (least squares). Measured on k = 1 segments of shifts ×0.5, ×0.4
and ×0.3, seeds 0–2.

Report: `cart_mass_rmse_ratio.student_t`, `cart_mass_rmse_ratio.ls_reject`, `nominal_inflation.lag_bounded_coriolis`, `nominal_inflation.lag`, `nominal_inflation.mlp`

## Baselines
- **Floor:** least squares, no rejection.
- **Black box:** `mlp_ctx` (H-FRAG arm only).
- **Known model:** the known-model fit; E-004's `known_exact` if E-004
  has run, otherwise the MuJoCo model with the parameters exposed.
- **Bound / ceiling:** the Cramér–Rao bound for the inlier noise; the fit
  on true poses.
- **Effort symmetry:**
  - the Student-t ν and the least-squares rejection threshold each get 4
    settings, tuned on held-out episodes;
  - the bounded-Coriolis change is offered only to the Lagrangian,
    because the MLP's velocity input is already tanh-bounded. The two
    arms then have matched bounds.

## Decision rule
- **Student-t ratio ≤ 1.5:** robust joint estimation is the fix. Feed it
  into the E-002/E-003 pipeline and re-run their primary metric once.
- **Student-t ratio > 2:** the errors are structured beyond any noise
  model (bias, runs). **Stop** estimator work. The remaining lever is the
  front end (resolution, framing, calibration); report to the owner.
- **H-FRAG confirmed:** note it as a claim. It only matters if the
  learned Lagrangian is used (Q4).
- **H-FRAG refuted:** record the eliminated explanation and **stop**. No
  further diagnosis unless Q4 is active.

## Prerequisites
- E-002 or E-003 shows a loss attributable to outliers or start states.
- New code, committed and tested before any run:
  - `src/oracles/robust_fit.py`;
  - the bounded-Coriolis option in `LagrangianDynamics`.

## Plan
- `.venv/bin/python scripts/oracles/context_oracle.py --experiment E-005 --front-end heuristic --estimator student_t …`

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update

Abandoned 2026-10-19 before any run: the owner clarified that the aim is to
*learn* the mechanics (cartpole as a showcase), not to control through known
equations with parameter refits. Kept for reference; see the charter backlog.
