# Current Status

This file is rewritten at the end of every working session; it is never
appended to. History lives in [`archive/`](archive/).

**As of 2026-10-20, early morning.** Charter: [`charter.md`](charter.md).
Stop date 2026-11-02.

## Where we are
- **Q0 is answered yes for cartpole** (nominal parameters, simulated camera):
  84 px camera frames → keypoint reader → 12-frame fixed-lag estimator →
  learned Lagrangian → MPPI swings up and balances (836 ± 9, C-018).
- **Q4 (does the structure pay on a harder system?) is answered no at the
  registered thresholds** on the two-pole cartpole:
  - true-model MPPI swings both poles up but does not hold them: 422 ± 44
    of about 1000 (C-019);
  - with the same data, the learned Lagrangian plans no better than an MLP
    (round 0: 193 vs 192; round 1: 243 vs 262; round 3: 347 vs 318) and
    costs 7× more per decision (C-020);
  - from random data it predicts the unseen swing-up states *worse* than
    the MLP (1 s error 0.58 vs 0.46, C-021).
- The only point for the Lagrangian is round 3: ahead on all 3 seeds by 29
  points, not significant, and not reflected in prediction error (post hoc
  0.240 vs 0.232).
- All learned-model results on two poles are limited by the planner:
  neither model, nor the true model, keeps both poles up for long.

## Experiments
| ID | Node | What | Status |
|---|---|---|---|
| E-006 | Q1 | MPPI over the true model | done: 861 ± 3 (C-015) |
| E-007 | Q2 | MPPI over a learned Lagrangian vs an MLP, data rounds | done: 856 ± 3 vs 853 ± 2 (C-016) |
| E-008 | Q3 | Keypoints + estimator in the loop; camera-learned model | done: ratio 0.977 (C-017, C-018) |
| E-009 | Q4 | Two-pole cartpole: true-model MPPI ceiling | done: 422 ± 44 (C-019) |
| E-010 | Q4 | Two-pole cartpole: learned Lagrangian vs MLP, true state | done: no advantage (C-020, C-021) |
| E-001…E-005 | — | Known-model plan | abandoned (never run) |

Compute used so far: about 38 CPU process-hours: the first charter
budget about 18, then E-009 about 1 of 1.5 and E-010 about 19 of 25 (within
the Q4 extension of 27), plus 0.2 MPS-hours.

## Waiting on the owner
1. **What next for Q4.** E-010's decision rule ends at an owner decision
   (no branch fits exactly: no advantage, both models between 0.5 and 0.85
   of the ceiling; nearest branch "no advantage"). Options:
   - **(a) stop the structure question** (agent's recommendation): on two
     systems the learned Lagrangian is no better than an MLP and 7× more
     expensive to plan with. Spend the rest of the time to 2026-11-02 on
     the write-up and demo.
   - **(b) fix the planner first** (new record): the true-model ceiling is
     weak because MPPI does not hold the balance (suspected: temperature
     1.0 makes it nearly greedy). Try a temperature/noise sweep or an LQR
     balance controller near upright on the true model (about 1–2
     CPU-hours). Only if that raises the ceiling to ≥ 700, re-evaluate the
     saved round-3 models with the new planner (no retraining, about
     4 CPU-hours). This is the one remaining test where balance matters.
   - (c) `three_poles`: not recommended; the planner would be weaker
     still and the Lagrangian cost grows with the degrees of freedom.
   - (d) acrobot swing-up (2 DoF, no rail): possible, but cartpole (also
     2 DoF) already showed parity.

## Last decisions
- 2026-10-19: E-009 and E-010 approved with the Q4 budget extension; run
  overnight.
- 2026-10-19: D-1 taken early; continue with a harder system (agent's
  choice: two-pole cartpole, node Q4); the owner wants to see whether the
  Lagrangian has an advantage. Pushes and merges have a general green
  light.
- 2026-10-19: E-006…E-008 approved and run; Q1–Q3 closed.
