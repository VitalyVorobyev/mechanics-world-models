# E-001 — Known-model MPC with online parameter fitting on true state

- **Status:** abandoned (never run; superseded by the 2026-10-19 re-charter, see `docs/charter.md`)
- **Question node:** Q1
- **Hypotheses:** A-KNOWN-1
- **Budget:** 6 agent-hours, 4 CPU-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** — (owner, date)

## Question
Before any camera is involved: does the standard mechanics-aware
controller solve the task with the true state? The controller is MPC over
the known cartpole model, with a few physical parameters fitted online
from recent transitions. Does it keep solving the task when cart mass,
pole length or actuator gear change? This is the ceiling for everything
downstream. If it fails, the camera pipeline cannot succeed.

## Prediction
- **Nominal parameters, true state.** MPC over the MuJoCo model reaches a
  mean dm_control cartpole-swingup return of at least 800 over 5
  episodes (about the level reported for SAC from state). The MPC is
  MPPI/CEM, horizon about 1.5 s, using `physics.copy()` rollouts.
  Confidence 75%. The risk is the planning cost per step on a laptop. The
  fallback is energy-shaping swing-up plus an LQR catch, which is also
  model-based and needs the same parameters.
- **Shifted plant, controller still at nominal** (shifts: cart mass ×0.5
  and ×2, MJCF pole length ×0.7 and ×1.3, gear ×0.5): the return drops by
  more than 20% for at least two of the five shifts. Confidence 60%.
  Pole length ×1.3 and gear ×0.5 are the likely worst cases.
- **Shifted plant, online fit** (least squares over the last 2 s of
  transitions for cart mass, pole mass, pole length and gear, refitted
  every 0.5 s): the return comes back to within 5% of the controller
  that knows the true parameters, for every shift. Confidence 70%.

## Primary metric
Mean episode return over 5 episodes per condition (seeds 0–4). The
headline is the **adaptation recovery**:

  (online-fit return − nominal-controller return) /
  (true-parameter-controller return − nominal-controller return),

averaged over the shifts where the nominal-controller drop exceeds 10%.

Report: `return.nominal.true_params`, `return.{shift}.nominal_params`, `return.{shift}.online_fit`, `return.{shift}.true_params`, `adaptation_recovery`

## Baselines
- **Floor:** random actions, and zero action (an episode return of a few
  tens).
- **Black box:** no new run. Published end-to-end pixel agents
  (DreamerV3, TD-MPC2: about 800–860 in distribution) and the cRSSM
  result that pixel Dreamer extrapolates poorly in physical parameters.
- **Known model:** this experiment *is* the known-model controller.
- **Ceiling:** the same controller given the true shifted parameters.
- **Effort symmetry:** one MPC configuration, tuned on nominal only
  (horizon, samples, temperature; at most 6 settings). The same
  configuration is used for every shift and every parameter source.

## Decision rule
- **Nominal ≥ 800 and recovery ≥ 0.8:** Q1 is closed, and the
  state-level answer to Q0 is "no invention needed". Go to E-002 with
  this controller.
- **Nominal ≥ 800 but recovery < 0.5:** the bottleneck is parameter
  identification under closed-loop data. Check excitation, then the
  choice of window. That is at most one diagnostic step, then report to
  the owner.
- **Nominal < 600 after the 6 tuning settings, both MPC and
  energy-shaping:** the controller itself is the problem.
  - **Stop** E-001.
  - Report what limits it (compute per step, or the rail).
  - The owner decides whether to spend more on control engineering.
- The thresholds:
  - 800 is the published level of end-to-end agents;
  - 0.8 recovery means the adapted controller keeps most of what
    knowing the parameters gives.

## Prerequisites
- MJCF-level `pole_length` and inertia-consistent `body_mass_<i>` shifts
  (`src/envs/dm_control_pixels.py`; A-OOD-1 fixed).
- Rung 0 for the shifts: `intervention_effect.py`, with gear and pole
  length added.
- New code, committed and tested before any run:
  - `src/control/mpc.py`: MPPI over a parameterized MuJoCo copy;
  - `src/control/sysid.py`: least squares on transitions;
  - `scripts/oracles/control_ceiling.py`.

## Plan
- `.venv/bin/python scripts/oracles/control_ceiling.py --experiment E-001 --seeds 0-4 --shift …`
- Tune on nominal first (≤ 6 settings, recorded).
- Planning uses batched native rollouts (`mujoco.rollout`), not Python
  stepping.
- Grid: 16 conditions (nominal, plus 5 shifts × 3 parameter sources) × 5
  episodes = 80 episodes of 500 decisions each.
- At ≤ 0.3 s per decision, that is ≤ 3.5 CPU-hours.
- **Pilot rule:** if one nominal episode takes more than 3 minutes,
  report back before running the grid.

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update

Abandoned 2026-10-19 before any run: the owner clarified that the aim is to
*learn* the mechanics (cartpole as a showcase), not to control through known
equations with parameter refits. Kept for reference; see the charter backlog.
