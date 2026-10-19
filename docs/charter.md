# Research Charter

The active scope of the project. Change it only by an owner decision,
recorded in the decision log at the bottom. Background:
[`review-2026-10-19.md`](review-2026-10-19.md). The April specification is
archived in [`archive/research_spec_2026-04.md`](archive/research_spec_2026-04.md).

- **Adopted:** 2026-10-19 (re-chartered the same day, see the decision log)
- **Stop date:** 2026-11-02
- **Decision due on the stop date (D-1):** does the cartpole demo work
  (camera → learned Lagrangian → planner, closed loop)? If yes, is the
  next step a more complex system, and which one? If no, which measured
  gap blocks it? **Taken early on 2026-10-19:** yes; next is the
  two-pole cartpole (Q4). The stop date stays.

## Goal (owner, 2026-10-19)

Build a system that takes **camera images**, **learns the mechanics** of
what it sees, and uses that learned model to **drive the mechanical
system**. Cartpole swing-up is the simplest showcase, not the objective.
The aim is to learn the mechanics of **more complex systems**, whose
equations nobody writes by hand. Extrapolating physical parameters is not
an aim.

Novelty is **not** an objective. A paper is not an objective. Standard
components are the default: invent only where a measured gap forces it.

Secondary goals:
- deep understanding of what each component contributes;
- a verifiable, agent-driven research process. This project is its
  testbed.

## Root question

**Q0.** Can a pipeline that *learns* the mechanics drive cartpole swing-up
in closed loop from camera images, at nominal physical parameters? The
pipeline is: camera → keypoint pose reader (trained on simulator renders)
→ state estimator → learned Lagrangian dynamics over given generalized
coordinates → MPPI planner with the known task reward.

What "learned" means here: the coordinates (x, θ) and which of them is
actuated are given; the mass matrix, potential, dissipation and actuation
are learned from data. The reward is the task specification and is not
learned.

## Question tree

Every experiment record names one node. A node is closed by a claim in
[`claims.md`](claims.md), or by an explicit decision to drop it.

| Node | Question | Experiments | State |
|---|---|---|---|
| Q1 | **Planner ceiling.** Does MPPI over the true MuJoCo model swing up and balance cartpole from true state? | E-006 | closed (C-015) |
| Q2 | **Learned mechanics on true state.** Does MPPI over a learned Lagrangian, trained from random data plus a few rounds of its own episodes, reach the ceiling? How does an MLP dynamics model do with the same budget? | E-007 | closed (C-016) |
| Q3 | **Camera in the loop.** With a sim-trained keypoint reader and a model-based estimator, does the Q2 controller hold its return from camera images, including with a model learned from camera poses? Then the demo. | E-008 | closed (C-017, C-018) |
| Q4 | **Does the structure pay on a harder system?** On the dm_control two-pole cartpole swing-up (3 DoF, coupled configuration-dependent mass matrix), first on true state: the true-model ceiling, then a learned Lagrangian against an MLP with the same data rounds. The camera stage follows only if the structure shows an advantage. | E-009, E-010 | open |

**Backlog** (not registered; needs a node and an owner decision):
- `cartpole.three_poles`, the acrobot swing-up (fallback if two poles
  fails the ceiling), or a simulated arm with a pendulum payload;
- the camera stage on two poles (E-011, after E-010);
- parameter adaptation (H1b; the abandoned E-001, E-003, E-004);
- robust errors-in-variables estimation (abandoned E-005; reopens if Q3
  fails on pose outliers);
- detecting a misspecified model from residuals.

## References for the goal

Numbers to compare against, not systems to run on a laptop:
- end-to-end pixel agents solve dm_control cartpole-swingup in
  distribution: DreamerV3 and TD-MPC2 report returns of about 800–860;
- learned Lagrangians have controlled cartpole and a Furuta pendulum
  from state (DeLaN, Lutter et al. 2019) and a pendulum from pixels
  (Zhong & Leonard 2020). Q0 is a reproduction-level target.

## Not in scope

- **Frozen:** the end-to-end pixel world models (RSSM,
  `MechanicsWorldModel`). They stay in the repo and their tests stay
  green, but there is no new development.
- training actor-critic or end-to-end RL agents (reference numbers only);
- physical-parameter shifts and adaptation (backlog);
- visual-nuisance robustness (H2), dissipation (H3).

## Budget

- **Whole charter:** two weeks of calendar time, laptop CPU/MPS only.
  Planned compute: about 15 CPU-hours plus 1 MPS-hour.
- **Q4 extension (approved by the owner 2026-10-19):** about
  27 CPU-hours more (E-009 1.5, E-010 25); the first charter budget is
  spent (about 18 CPU-hours used).
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
| 2026-10-19 | Re-charter: the aim is to *learn* the mechanics (cartpole is a showcase for more complex systems), not to adapt parameters of known equations. E-001…E-005 abandoned before any run. New tree Q1–Q3 (E-006…E-008). Perception: keypoints trained on simulator renders. Scope: the cartpole demo only; the complex-system step is decided on the stop date | owner | this conversation; plan of 2026-10-19 |
| 2026-10-19 | D-1 taken early: Q0 holds on cartpole; continue with a harder system, chosen by the agent: the dm_control two-pole cartpole (node Q4, E-009, E-010). The owner wants to see whether the Lagrangian structure has an advantage; the records must allow the answer "no" | owner (system chosen by agent) | this conversation |
