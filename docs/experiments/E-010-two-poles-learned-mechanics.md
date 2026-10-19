# E-010 — Learned Lagrangian vs MLP on the two-pole cartpole, true state

- **Status:** approved
- **Question node:** Q4
- **Hypotheses:** H-LAG-2P, H-LAG-PRED-2P
- **Budget:** 12 agent-hours, 25 CPU-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** owner, 2026-10-19 (with the Q4 budget extension; E-010 starts only after the E-009 decision rule allows it)

## Question
On cartpole the Lagrangian structure bought nothing: an MLP planned as
well from the same data (C-016). Does the structure help on a system
whose mechanics are harder to learn? The two-pole cartpole has a
configuration-dependent mass matrix that couples the poles, a potential
over two angles, and random-policy data that almost never visits the
"both poles up" region. This is where a learned Lagrangian should
carry what it learned to unseen states, and an MLP has no reason to.
The test must be able to say "no".

## Prediction
Models: the E-007 learned Lagrangian generalized to q = (x, θ1, θ2)
(Cholesky mass matrix, potential, linear dissipation,
`actuated_dims=(0,)`, S¹ embedding of both angles, 2 substeps) and the
E-007 MLP dynamics model, both trained by the E-007 multi-step rollout
regression (15k steps), at nominal parameters, 3 training seeds. Same
data rounds as E-007: round 0 = 60 random-policy episodes; rounds 1–3 add
10 MPPI episodes each (action noise std 0.3); retrain from scratch.
Planner: the frozen E-009 setting.

Predictions:
- **Round 0 (random data only): the Lagrangian returns at least 50
  points more than the MLP. Confidence 45%.** On cartpole the MLP won
  round 0 (842 vs 809), so the prior is weak; the reason to expect a
  change is that the two-pole coupling and upright region are much
  further from the random data.
- By round 3, both reach ≥ 0.85 of the E-009 ceiling: Lagrangian 60%,
  MLP 50%.
- Open-loop prediction from held-out states of E-009 true-model episodes
  (swing-up and balance, i.e. states random data does not visit): the
  Lagrangian's 1 s position error is at most half the MLP's, for round-0
  models. Confidence 55%.
- Planning cost: Lagrangian 0.4–0.8 s per decision, MLP about 0.1 s.

## Primary metric
Mean episode return per (form, round), averaged over 3 training seeds,
each evaluated on 5 episodes (seeds 0–4). Headline: Lagrangian minus MLP
at round 0 and at round 1, next to the E-009 ceiling.

Report: `return_mean`

Exploratory (registered here, reported, not used for the headline):
- `pred_error_1s`: median over held-out E-009 windows of the 1 s
  (50-decision) open-loop error in (x, θ1, θ2), per round-0 model;
- a round-0 data-size sweep with 20 and 200 random episodes, per form
  and seed (one evaluation of 5 episodes each).

## Baselines
- **Floor:** the E-009 zero and random floors.
- **Black box:** the MLP dynamics model, same data, same training steps,
  same planner.
- **Known model:** the E-009 true-model planner (ceiling).
- **Effort symmetry:** both forms get the same data rounds, training
  budget, planner setting, and the single allowed fix below. Nothing is
  tuned per form; any asymmetry is logged.

## Decision rule
"Advantage" means: Lagrangian mean − MLP mean ≥ 50 at round 0 or at
round 1, with the Lagrangian ahead for at least 2 of 3 seeds; or, with
no return deficit, `pred_error_1s` at most half the MLP's.
- **Advantage shown and the Lagrangian reaches ≥ 0.85 of the ceiling by
  round 3:** Q4 closes in favour of the structure. Next (to be registered
  as E-011, owner approval): the camera stage on two poles with the
  E-008 pipeline (three keypoints).
- **No advantage, both reach ≥ 0.85:** the structure does not pay on two
  poles either. Report it as the second negative (with cost). The owner
  decides between `three_poles` (the last cheap step up in difficulty)
  and **stopping** the structure question.
- **Both < 0.5 of the ceiling after round 3:** one allowed fix, given to
  both forms: a 5-model ensemble with a disagreement penalty (PETS
  style). If both stay below 0.5, **stop** and report.
- **Only the MLP reaches 0.85:** report it as a negative for the
  structure; **stop** the two-pole line; the owner decides.
- Thresholds: 50 points is far above the seed std of the E-007 models at
  round 3 (about 3) but below the Lagrangian's round-0 seed std on
  cartpole (70), hence the extra 2-of-3-seeds condition; 0.85 matches
  E-008.

**Note before any run (2026-10-19, after E-009):** the E-009 ceiling is
422 (C-019), so 0.85 of it is 358, below the about 410 that the
half-solution (first pole hanging, second balanced) earns. The "reaches
0.85" conditions therefore do not show that a model balances both
poles; results will also report `upright_fraction_second_half` (both
poles upright). The headline (Lagrangian − MLP) and the decision rule are
unchanged. Planner: E009_PLANNER (horizon 150, 512 samples, noise 0.5).
Measured planning cost on random weights: Lagrangian 0.47 s, MLP 0.07 s
per decision (3 threads).

## Prerequisites
- E-009 done (setting frozen, ceiling measured, held-out episodes saved).
- New code, committed and tested before any run:
  - `DynamicsWrapper` with `q_dim` and `angle_dims` arguments (today
    `q_dim=2` is hard-coded); the cartpole defaults unchanged;
  - two-pole state collection (random and MPPI policies);
  - `learned_control.py --task two_poles`, and the prediction-error
    evaluation.

## Plan
- `.venv/bin/python scripts/oracles/learned_control.py --experiment E-010 --task two_poles --form {lagrangian,mlp} --seed {0,1,2} --rounds 3`
- Data-size sweep: `... --rounds 0 --random-episodes {20,200}`.
- Cost estimate: Lagrangian about 5 h per seed (50 episodes × 4–6 min
  plus 4 trainings × 12 min), MLP about 1 h per seed; the sweep about
  6 h; about 25 CPU-hours in all, 7–9 hours wall-clock with 3–4
  processes in parallel.
- Pilot rule: round 0 for one seed of each form first. If a Lagrangian
  episode takes more than 10 minutes, report before continuing.

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update
