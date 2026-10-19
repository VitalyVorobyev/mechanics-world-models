# E-009 — Planner ceiling on the two-pole cartpole

- **Status:** registered (awaiting owner approval)
- **Question node:** Q4
- **Hypotheses:** H-PLAN-2P
- **Budget:** 3 agent-hours, 1.5 CPU-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** —

## Question
Before asking whether a learned Lagrangian beats an MLP on a harder
system, measure what the planner can do with the true model there.
The system is dm_control `cartpole.two_poles`: a cart with two poles in
a chain (3 DoF, one actuator; cart 1 kg, poles 0.1 kg each; physics dt
0.01 s), swing-up from hanging down, dense reward. Its mass matrix
depends on the configuration and couples the two poles, which a
single pole does not. If the true-model planner cannot swing it up, no
learned model can, and the comparison has no room.

## Prediction
- After tuning, MPPI over the true MuJoCo model returns ≥ 600 on
  5 evaluation episodes. Confidence 60%. A double pendulum on a cart is
  a standard MPPI swing-up task, but it needs a longer horizon than one
  pole (both poles must arrive upright together) and the dense reward
  also charges for pole speed during the longer swing.
- The best setting needs a horizon of at least 1.5 s (75 decisions).
  Confidence 75%.
- Cost: 0.15–0.4 s per decision at 1024 × 150, i.e. 1.5–3.5 min per
  episode.

## Primary metric
Mean episode return over evaluation seeds 0–4 with the frozen setting
(500 decisions, action repeat 2, as in E-006).

Report: `return_mean`

## Baselines
- **Floor:** zero action and uniform random actions (seeds 0–4).
- **Known model:** this is the known-model ceiling for E-010.
- **Effort symmetry:** the frozen setting is used unchanged by every
  model in E-010.
- **Tuning grid** (one episode each, tuning seed 101, not counted in the
  result): horizon {75, 150} decisions × samples {512, 1024} × noise
  std {0.5, 0.8}, at most 8 settings. The chosen setting is the best
  return; a cheaper setting within 10 points of the best is preferred
  (as in E-006), and the choice is logged.

## Decision rule
- **≥ 600:** freeze the setting and go to E-010.
- **300–600:** one fix: 2 MPPI iterations per decision with the best
  setting, and horizon 200 (at most 4 more tuning episodes). If the
  result is then ≥ 400, freeze it and go to E-010 with the ceiling as
  measured; the E-010 thresholds are relative to it.
- **< 300 after the fix:** **stop** the two-pole line. Report what fails
  (timing of the swing, rail, reward shape). The owner decides between
  the acrobot swing-up (2 DoF, no cart rail) and stopping.
- Thresholds: 600 is about 70% of the one-pole ceiling (861); 300 is
  roughly "one pole up, one swinging", which is not a usable ceiling.

## Prerequisites
- E-006 code (MPPI, MuJoCo backend), generalized to n poles:
  - the dm_control dense reward for n poles: mean of the per-pole
    uprightness (world-frame cosine), the minimum over joint-velocity
    tolerances, control and centring terms. A test checks it against
    the env reward on random states;
  - `control_episode.py --task two_poles`.

## Plan
- `.venv/bin/python scripts/oracles/control_episode.py --experiment E-009 --task two_poles --dynamics true --seed 101 <grid flags>` (tuning).
- `... --dynamics true --seed {0..4}` with the frozen setting; floors with `--dynamics zero|random`.
- Wall-clock about 40 minutes (4 processes in parallel).

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update
