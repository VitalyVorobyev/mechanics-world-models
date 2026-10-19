# Research Charter

The active scope of the project. Change it only by an owner decision,
recorded in the decision log at the bottom. Why it looks like this:
[`review-2026-10-19.md`](review-2026-10-19.md). The April specification is
archived in [`archive/research_spec_2026-04.md`](archive/research_spec_2026-04.md).

- **Adopted:** 2026-10-19
- **Stop date:** 2026-11-02
- **Decision due on the stop date (D-1):** is the camera-driven controller
  working? If not, which measured gap blocks it, and is closing that gap
  worth inventing something for?

## Goal (owner, 2026-10-19)

Build a system that takes **camera images** as input, is **aware of the
mechanics** of what it controls, and **drives a mechanical system**. That
means cartpole first, and later something like a robot arm.

Novelty is **not** an objective. A paper is not an objective. The question
is whether the goal can be reached with existing methods. **The default
answer is yes, and invention is allowed only where a measured gap forces
it.**

Secondary goals:
- deep understanding of what each component contributes;
- a verifiable, agent-driven research process. This project is its
  testbed.

## Root question

**Q0.** Can we drive cartpole swing-up and balance from camera images with
a mechanics-aware pipeline built from standard components? The pipeline
is: calibrated camera → pose → state estimator → known-model MPC → online
parameter identification. Does it keep working when the physical
parameters change? Where exactly does it fall short?

Being "mechanics-aware" is operationalized as: the controller uses an
explicit mechanical model, and after a change in the physical parameters
it recovers by re-identifying a few of them, not by retraining.

## Question tree

Every experiment record names one node. A node is closed by a claim in
[`claims.md`](claims.md), or by an explicit decision to drop it.

| Node | Question | Experiments | State |
|---|---|---|---|
| Q1 | **Control ceiling on true state.** Does known-model MPC with online parameter fitting swing up and balance cartpole, nominal and under parameter shifts? | E-001 | registered |
| Q2 | **Camera to state.** With a field of view that keeps the system in view, does a standard front end plus a state estimator give a state good enough for the Q1 controller, in closed loop? | E-002 | registered |
| Q3 | **Adaptation from vision.** After a parameter change, does online identification from camera-estimated states restore the closed-loop return? | E-003 | registered |
| Q4 | **Conditional: learned structure.** Only if the known model is unavailable or inadequate: does a learned Lagrangian add anything over the known model plus a fit? | E-004 | registered, low priority |
| Q5 | **Conditional: robust estimation.** Only if Q2/Q3 fail because of pose outliers or start-state errors: does a robust (Student-t, errors-in-variables) estimator close the gap? | E-005 | registered, conditional |

**Backlog** (not registered; needs a node and an owner decision):
- detect a misspecified model from residuals;
- scale with resolution against the Cramér–Rao bound;
- a structure-matched black-box control for H1b;
- a second system (acrobot, or a simulated arm with a pendulum payload).

## References for the goal

These are numbers to compare against, not systems to run on a laptop:
- end-to-end pixel agents solve dm_control cartpole-swingup in
  distribution: DreamerV3 and TD-MPC2 report returns of about 800–860;
- they are known to extrapolate poorly in physical parameters (cRSSM,
  Prasanna et al. 2024).

The modular pipeline is the standard engineering alternative. That is
classical adaptive control plus vision, as in UP-OSI (Yu et al. 2017) or
Jaques et al. 2020.

## Not in scope

- **Frozen:** the end-to-end pixel world models (RSSM,
  `MechanicsWorldModel`; 65% of `src/`). They stay in the repo and their
  tests stay green, but there is no new development. They come back only
  if Q2 shows that a standard front end cannot read the state.
- training actor-critic or end-to-end RL agents (reference numbers only);
- visual-nuisance robustness (H2), dissipation (H3).

## Budget

- **Whole charter:** two weeks of calendar time, on laptop CPU/MPS only.
- **Per experiment:** each record states its agent-hours and
  compute-hours. At 150% of either, the experiment stops and goes back to
  the owner.
- **Chain depth:** three consecutive "diagnose further" outcomes on one
  node force a zoom-out note and an owner decision
  ([`research_protocol.md`](research_protocol.md) §4).

## Decision log

| Date | Decision | By | Evidence |
|---|---|---|---|
| 2026-10-18 | Stop the pixel gate run; pivot to H1b on true state | owner + agent | archived status log |
| 2026-10-19 | Stop the OOD world-model comparison; adopt this charter and protocol v2. Goal clarified: camera-driven, mechanics-aware control; novelty not required. E-001…E-005 registered, none approved | owner | [`review-2026-10-19.md`](review-2026-10-19.md) |
