# Current Status

This file is rewritten at the end of every working session; it is never
appended to. History lives in [`archive/`](archive/).

**As of 2026-10-19, end of day.** Charter: [`charter.md`](charter.md). Stop
date 2026-11-02.

## Where we are
- **Q0 is answered yes for cartpole** (nominal parameters, simulated camera):
  84 px camera frames → keypoint reader → 12-frame fixed-lag estimator →
  learned Lagrangian → MPPI swings up and balances. A Lagrangian learned
  only from camera poses returns 836 ± 9, 0.977 of the same model learned
  from true state (C-018). Nothing had to be invented.
- **What cartpole cannot show:** the Lagrangian structure buys nothing here;
  an MLP dynamics model is as good from the same data and 5.7× cheaper to
  plan with (C-016).
- **Evidence:** C-015 … C-018, grade A except the keypoint reader (B, one
  seed). Older claims stay legacy (B/C).
- Demo: videos via `scripts/demo/episode_video.py`; the demo page is
  published as an artifact (link in the session summary).

## Experiments
| ID | Node | What | Status |
|---|---|---|---|
| E-006 | Q1 | MPPI over the true model | done: 861 ± 3 (C-015) |
| E-007 | Q2 | MPPI over a learned Lagrangian vs an MLP, data rounds | done: 856 ± 3 vs 853 ± 2 (C-016) |
| E-008 | Q3 | Keypoints + estimator in the loop; camera-learned model | done: ratio 0.977 (C-017, C-018) |
| E-009 | Q4 | Two-pole cartpole: true-model MPPI ceiling | registered, awaiting approval |
| E-010 | Q4 | Two-pole cartpole: learned Lagrangian vs MLP, true state | registered, awaiting approval |
| E-001…E-005 | — | Known-model plan | abandoned (never run) |

Compute used so far: about 18 CPU process-hours (E-007 ≈ 8.5 of 8 budgeted, E-008 ≈ 9.5 of 8: both within the 150% limit) and 0.2 MPS-hours. The first charter budget is spent.

## Waiting on the owner
1. Approve E-009 and E-010, and the Q4 budget extension of about
   27 CPU-hours (E-009 1.5, E-010 25; 8–10 hours wall-clock).

## Last decisions
- 2026-10-19: D-1 taken early; continue with a harder system (agent's
  choice: two-pole cartpole, node Q4); the owner wants to see whether the
  Lagrangian has an advantage. Branch `october-2026-review` merged to
  `main` (PR #1); pushes and merges have a general green light.
- 2026-10-19: E-006…E-008 approved and run; Q1–Q3 closed.
- 2026-10-19: re-charter: learn the mechanics; cartpole demo only; keypoints
  trained on renders; E-001…E-005 abandoned.
