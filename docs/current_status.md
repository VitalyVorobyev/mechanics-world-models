# Current Status

This file is rewritten at the end of every working session; it is never
appended to. History lives in [`archive/`](archive/).

**As of 2026-10-19.** Charter: [`charter.md`](charter.md). Stop date
2026-11-02.

## Where we are
- **Goal (clarified by the owner):** camera-driven, mechanics-aware
  control of cartpole; later a robot arm. Novelty is not required.
- **Review:** [`review-2026-10-19.md`](review-2026-10-19.md).
  - The original OOD world-model comparison is stopped.
  - The H1b result on true state (C-001 … C-003) is a correct but expected
    grey-box effect.
  - The standard modular pipeline was never tried. It is now the main
    line.
- **Evidence:** 143 earlier results are imported into the ledger as
  `legacy`. At least 87 of them came from uncommitted code, so most
  claims are grade C. No claim is grade A yet.
- **Process v2 is in force:**
  - pre-registration checked by `scripts/research/check.py`;
  - clean-tree runs;
  - generated tables;
  - the chain-depth limit.

## Experiments
| ID | Node | What | Status |
|---|---|---|---|
| E-001 | Q1 | Known-model MPC + online fitting, true state | registered, **awaiting owner approval** |
| E-002 | Q2 | Camera front end + EKF in the loop | registered (needs E-001) |
| E-003 | Q3 | Online identification from camera states | registered (needs E-002) |
| E-004 | Q4 | Known-model bracket for H1b (conditional, low priority) | registered |
| E-005 | Q5 | Robust errors-in-variables estimation (conditional) | registered |

## Waiting on the owner
1. Approve E-001 (6 agent-hours, about 4 CPU-hours), or change it.
2. Confirm the order E-001 → E-002 → E-003. E-004 and E-005 run only on
   their trigger.

## Last decisions
- 2026-10-19: OOD world-model comparison stopped; charter and protocol v2
  adopted; goal clarified (no novelty required).
- 2026-10-18: pixel gate run stopped; pivot to H1b on true state.
