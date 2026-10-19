# Current Status

This file is rewritten at the end of every working session; it is never
appended to. History lives in [`archive/`](archive/).

**As of 2026-10-19 (re-charter).** Charter: [`charter.md`](charter.md).
Stop date 2026-11-02.

## Where we are
- **Goal (owner):** learn the mechanics from camera images and drive the
  system with the learned model. Cartpole swing-up is the showcase; the aim
  is more complex systems. Parameter extrapolation is not an aim.
- **Re-charter:** the known-model plan (E-001…E-005) is abandoned before
  any run. The new line is Q1 planner ceiling → Q2 learned Lagrangian on
  true state → Q3 camera in the loop, then the demo.
- **Fact behind it:** no closed loop has ever run with any learned model.
  `CEMPlanner` supports only the RSSM, and no return files exist.
- **Evidence:** the earlier results are legacy (grade B/C), see
  [`claims.md`](claims.md). No claim is grade A yet.

## Experiments
| ID | Node | What | Status |
|---|---|---|---|
| E-006 | Q1 | MPPI over the true MuJoCo model (ceiling) | registered, **awaiting owner approval** |
| E-007 | Q2 | MPPI over a learned Lagrangian vs an MLP, data rounds | registered (needs E-006) |
| E-008 | Q3 | Sim-trained keypoints + estimator in the loop; demo | registered (needs E-007) |
| E-001…E-005 | — | Known-model plan | abandoned (never run) |

Planned compute: about 15 CPU-hours plus 1 MPS-hour over two weeks.

## Waiting on the owner
1. Approve E-006 (4 agent-hours, ≤ 1 CPU-hour), or change it.
2. Approve E-007 and E-008 now (each still starts only after its
   predecessor's decision rule), or one at a time.

## Last decisions
- 2026-10-19: re-charter: learn the mechanics; cartpole demo only; keypoints
  trained on renders; E-001…E-005 abandoned; E-006…E-008 registered.
- 2026-10-19: OOD world-model comparison stopped; charter and protocol v2
  adopted.
- 2026-10-18: pixel gate run stopped; pivot to H1b on true state.
