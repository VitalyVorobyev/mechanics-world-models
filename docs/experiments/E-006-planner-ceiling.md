# E-006 — MPPI planner ceiling on the true model

- **Status:** registered (awaiting owner approval)
- **Question node:** Q1
- **Hypotheses:** H-PLAN
- **Budget:** 4 agent-hours, 1 CPU-hour (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** — (owner, date)

## Question
Before any learned model: does the planner itself solve cartpole
swing-up when it plans with the true MuJoCo dynamics from true state?
This is the ceiling that every learned-model and camera result is
divided by, and it fixes the planner configuration used in E-007 and
E-008.

## Prediction
- MPPI with warm start, horizon about 1–1.5 s (50–75 decisions at
  0.02 s), about 500–1000 samples, rolled out with `mujoco.rollout`,
  reaches a mean dm_control cartpole-swingup return of **at least 800**
  over 5 episodes. Confidence 85%. Reasoning: the dense swing-up reward
  rewards upright, centred, slow states, and sampling MPC with the true
  model and a 1 s horizon is a standard way to solve it; the risk is the
  rail (±1.8 m) and the cost per decision.
- One decision takes 0.05–0.1 s on the laptop CPU, so an episode of 500
  decisions takes about 1 minute. Confidence 70%.

## Primary metric
Mean episode return over 5 evaluation episodes (seeds 0–4, one episode
per run) with the configuration chosen in tuning.

Report: `return_mean`

## Baselines
- **Floor:** zero action and random actions (one run each, 5 episodes).
- **Black box:** none run; published pixel agents reach about 800–860.
- **Known model:** this experiment is the known-model planner.
- **Ceiling:** none above this; it is the ceiling for E-007/E-008.
- **Effort symmetry:** tuning uses at most 6 settings (samples, horizon,
  temperature, noise std), one episode each on seeds 100–105 (not the
  evaluation seeds). The chosen setting is reused unchanged in E-007 and
  E-008, where only the dynamics backend changes.

## Decision rule
- **Return ≥ 800:** Q1 is closed; freeze the planner configuration and
  go to E-007.
- **600 ≤ return < 800:** accept it as the ceiling (E-007/E-008 report
  ratios to it), note the limit, go to E-007. One extra tuning setting
  is allowed only if a single cause (rail hits, horizon) is visible in
  the episodes.
- **Return < 600 after the 6 settings:** **stop** E-006. Report what
  limits it (cost per decision, rail, horizon); the owner decides whether
  to spend more on control or change the planner.
- Thresholds: 800 is the level of published end-to-end agents; 600 is
  where swing-up succeeds but balance is lost often, which would make
  ratios in E-007/E-008 hard to read.

## Prerequisites
New code, committed and tested before any run:
- `src/control/reward.py`: analytic swing-up reward (NumPy and torch),
  tested equal to the env reward;
- `src/control/mppi.py`: MPPI over a batched `rollout(q, q̇, actions)`,
  with a rail cost for |x| > 1.6;
- a MuJoCo rollout backend with `mujoco.rollout`;
- `scripts/oracles/control_episode.py` (begin_run, ledger).

## Plan
- `.venv/bin/python scripts/oracles/control_episode.py --experiment E-006 --dynamics true --seed S`
  for S in 0–4, plus the tuning runs (seeds 100–105) and the floors.
- Pilot rule: if one episode takes more than 2 minutes, report before
  the 5-episode evaluation.

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update
