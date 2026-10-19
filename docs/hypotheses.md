# Hypothesis and Assumption Index

A compact index of every hypothesis and assumption the work relies on. One
line each: status, evidence (claim IDs in [`claims.md`](claims.md)), and
the test that would change the status. Keep it short. Predictions,
decision rules and results go into experiment records
([`experiments/`](experiments/)), not here.

The full registry as of 2026-10-19 08:15, with the H1b lab notebook, is
archived verbatim in
[`experiments/archive/2026-10-18_19_registry_log.md`](experiments/archive/2026-10-18_19_registry_log.md).

**Statuses:**
- `assumed`: relied on, never tested;
- `open`: a registered experiment tests it;
- `supported` / `refuted`: by the cited claims;
- `superseded`;
- `out of scope`: by the [charter](charter.md).

## Research hypotheses

| ID | Statement | Status | Evidence | Next test |
|---|---|---|---|---|
| H-PLAN | MPPI over the true MuJoCo model swings up and balances cartpole from true state (return ≥ 800) | supported (861 ± 3) | C-015 | — |
| H-LAG-CTRL | MPPI over a learned Lagrangian, trained from random data plus a few rounds of its own episodes, reaches ≥ 700 | supported (856 ± 3 after round 3; 809 ± 70 from random data alone) | C-016 | — |
| H-LAG-DATA | With random data only, the learned Lagrangian plans better than an MLP (structure carries to unseen upright states) | refuted on cartpole (MLP 842 ± 3 vs Lagrangian 809 ± 70) | C-016 | two poles: H-LAG-2P (E-010) |
| H-KP | A keypoint reader trained on renders of uniformly sampled states reads θ to ≤ 1° median, upright included | supported (0.19° upright) | C-017 | — |
| H-CAM-LAG | Camera states (and a model learned from camera poses) keep ≥ 85% of the true-state return | supported (0.977 for the camera-learned model) | C-018 | — |
| H-PLAN-2P | MPPI over the true model swings up the two-pole cartpole (return ≥ 600) | refuted (422 ± 44: swings up, does not hold balance) | C-019 | a better planner (new record) |
| H-LAG-2P | On two poles, a learned Lagrangian plans better than an MLP from the same small data (≥ 50 points at round 0 or 1) | open | — | E-010 |
| H-LAG-PRED-2P | On two poles, a Lagrangian learned from random data predicts unseen swing-up states at least 2× better than an MLP over 1 s | open | — | E-010 |
| H1a | Structure improves extrapolation to unseen *states* of the same system | not supported in closed loop on cartpole (E-007 round 0) | C-016 | two poles: H-LAG-PRED-2P (E-010) |
| H1b | A physical-slot Lagrangian adapts and extrapolates to changed parameters better than black boxes | backlog (charter): supported on true state against a context MLP for slotted parameters (B); unbracketed | C-001, C-002, C-003, C-004, C-005 | — |
| H1b-px | Parameter adaptation survives real perception errors | backlog | C-008, C-009, C-010 | — |
| H-FRAG | The Lagrangian's fragility to heavy-tailed pose errors is caused by the unbounded q̇² Coriolis terms | backlog; reopens if E-008 fails on outliers | C-008 | — |
| H-CTRL, H-CAM | Known-model MPC with parameter refits; standard front end + EKF for it | out of scope (abandoned E-001/E-002, re-charter 2026-10-19) | — | — |
| H-MIS | A missing slot is detectable from adaptation residuals | backlog | C-003 | — |
| H-RES | Identification accuracy scales with image resolution as the Cramér–Rao bound predicts | backlog | — | — |
| H1-zs | Structure alone degrades more gracefully under unseen parameters | refuted as a useful question (no parameter input) | archived log | — |
| H2 | A `(q, q̇, z)` split isolates visual nuisance | out of scope | — | — |
| H3 | Dissipation matters | out of scope (nominal cartpole damping has no measurable effect) | archived log | — |

## Assumptions

| ID | Statement | Status | Evidence |
|---|---|---|---|
| A-ACT-1 | Learned `M`, `V`, `B` match the true terms up to a common scale | refuted for unconstrained `B`; supported with `actuated_dims=(0,)` (near-gauge between mass and gain) | C-006 |
| A-OOD-2 | Pole-mass shifts measurably change the dynamics | refuted | C-007 |
| A-OOD-3 | Cart-mass shifts measurably change the dynamics | supported | C-007 |
| A-RAIL-1 | Random-policy windows are free of rail contacts | refuted (about 30% touch) | C-007 |
| A-VIEW-1 | The camera sees the cart and pole in every frame | refuted (35% partly out of view) | C-013 |
| A-FE-1 | A label-free front end can read the pose accurately | partly supported: heuristic ≈ supervised keypoint; gradient through a renderer fails, search works | C-011, C-012 |
| A-ENC-1 | A per-frame CNN regresses `(x, cos θ, sin θ)` | supported for visible, non-upright poses | archived log |
| A-KNOWN-1 | The known cartpole model with a few fitted parameters is adequate for control and as an upper bracket | backlog (abandoned E-001, E-004) | — |
| A-CRB-1 | The Cramér–Rao bound for parameters under the pose-noise model is computable and informative | backlog | — |
| A-COV-1 | Random-policy data covers the states needed for balancing | refuted for the data as collected: 0.3% of frames have cos θ > 0.9 (code survey 2026-10-19, not ledgered); E-007 adds on-policy rounds | — |
| A-RAIL-2 | The planner can keep the cart off the rail with a cost alone, although the learned models have no rail | supported for the true model (cart within ±1.14 m); open for learned models | C-015, E-007 |
| A-DEC-1, A-ROUTE-1, A-DT-1, A-OOD-1, A-OOD-4, A-DATA-1, A-PROBE-1, A-METRIC-1, A-LEAK-1, A-DISS-1, A-LAG-1, A-BASE-1, A-MPS-1, A-ENV-1, D-GAUGE, D-K1 | Pixel world-model assumptions and diagnoses (April–October 2026) | resolved or frozen with the pixel world models | archived log, C-014 |
