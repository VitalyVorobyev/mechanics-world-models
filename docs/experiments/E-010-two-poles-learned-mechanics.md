# E-010 — Learned Lagrangian vs MLP on the two-pole cartpole, true state

- **Status:** done
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
**Condition 1:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="mlp", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=60, `round`=0, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-204328`, `learned_control/20261019-204746`, `learned_control/20261019-204747`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 192 ± 23.2 (3) | 209 / 166 / 202 |
| `upright_fraction_second_half` | 0.004 ± 0.005 (3) | 0.0024 / 0.00e+00 / 0.0096 |
| `decision_time_mean_s` | 0.0776 ± 0.00523 (3) | 0.0836 / 0.0745 / 0.0747 |

**Condition 2:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="mlp", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=60, `round`=1, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-205912`, `learned_control/20261019-205748`, `learned_control/20261019-205751`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 262 ± 19 (3) | 283 / 247 / 256 |
| `upright_fraction_second_half` | 0.00853 ± 0.00201 (3) | 0.0064 / 0.0088 / 0.0104 |
| `decision_time_mean_s` | 0.0894 ± 0.0284 (3) | 0.122 / 0.0729 / 0.0732 |

**Condition 3:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="mlp", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=60, `round`=2, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-211032`, `learned_control/20261019-210816`, `learned_control/20261019-210821`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 319 ± 28.3 (3) | 306 / 299 / 351 |
| `upright_fraction_second_half` | 0.0347 ± 0.0228 (3) | 0.036 / 0.0112 / 0.0568 |
| `decision_time_mean_s` | 0.0781 ± 0.00341 (3) | 0.082 / 0.076 / 0.0762 |

**Condition 4:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=60, `round`=0, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-211517`, `learned_control/20261019-211918`, `learned_control/20261019-211921`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 193 ± 18.6 (3) | 199 / 173 / 208 |
| `upright_fraction_second_half` | 0.00e+00 ± 0.00e+00 (3) | 0.00e+00 / 0.00e+00 / 0.00e+00 |
| `decision_time_mean_s` | 0.621 ± 0.0103 (3) | 0.633 / 0.614 / 0.615 |

**Condition 5:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="mlp", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=60, `round`=3, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-212139`, `learned_control/20261019-211843`, `learned_control/20261019-211851`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 318 ± 19.1 (3) | 329 / 329 / 296 |
| `upright_fraction_second_half` | 0.0245 ± 0.0271 (3) | 0.0536 / 0.02 / 0.00e+00 |
| `decision_time_mean_s` | 0.0769 ± 0.00141 (3) | 0.0785 / 0.076 / 0.0762 |

**Condition 6:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="mlp", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=20, `round`=0, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-212305`, `learned_control/20261019-212711`, `learned_control/20261019-213117`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 185 ± 1.3 (3) | 186 / 185 / 183 |
| `upright_fraction_second_half` | 0.0016 ± 0.00277 (3) | 0.0048 / 0.00e+00 / 0.00e+00 |
| `decision_time_mean_s` | 0.075 ± 1.32e-04 (3) | 0.0749 / 0.0751 / 0.0751 |

**Condition 7:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="mlp", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=200, `round`=0, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-213530`, `learned_control/20261019-213942`, `learned_control/20261019-214354`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 202 ± 10.1 (3) | 212 / 201 / 192 |
| `upright_fraction_second_half` | 0.00587 ± 0.0068 (3) | 0.0136 / 0.0032 / 8.00e-04 |
| `decision_time_mean_s` | 0.0739 ± 1.20e-04 (3) | 0.0739 / 0.074 / 0.0738 |

**Condition 8:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=20, `round`=0, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-215543`, `learned_control/20261019-222818`, `learned_control/20261019-225952`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 189 ± 22.1 (3) | 169 / 184 / 213 |
| `upright_fraction_second_half` | 0.00e+00 ± 0.00e+00 (3) | 0.00e+00 / 0.00e+00 / 0.00e+00 |
| `decision_time_mean_s` | 0.565 ± 0.0252 (3) | 0.592 / 0.56 / 0.543 |

**Condition 9:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=60, `round`=1, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-223747`, `learned_control/20261019-224144`, `learned_control/20261019-224201`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 243 ± 30 (3) | 221 / 231 / 277 |
| `upright_fraction_second_half` | 0.00e+00 ± 0.00e+00 (3) | 0.00e+00 / 0.00e+00 / 0.00e+00 |
| `decision_time_mean_s` | 0.57 ± 0.00221 (3) | 0.571 / 0.567 / 0.572 |

**Condition 10:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=200, `round`=0, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-233219`, `learned_control/20261020-000513`, `learned_control/20261020-003657`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 216 ± 17.6 (3) | 199 / 214 / 234 |
| `upright_fraction_second_half` | 0.00e+00 ± 0.00e+00 (3) | 0.00e+00 / 0.00e+00 / 0.00e+00 |
| `decision_time_mean_s` | 0.551 ± 0.00858 (3) | 0.551 / 0.559 / 0.542 |

**Condition 11:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=60, `round`=2, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261019-235839`, `learned_control/20261020-000244`, `learned_control/20261020-000326`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 323 ± 20.2 (3) | 300 / 332 / 337 |
| `upright_fraction_second_half` | 0.0221 ± 0.0152 (3) | 0.0072 / 0.0376 / 0.0216 |
| `decision_time_mean_s` | 0.572 ± 0.00122 (3) | 0.572 / 0.571 / 0.574 |

**Condition 12:** `collect_noise`=0.3, `episodes`=null, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `horizon`=null, `mppi`={"action_dim": 1, "horizon": 150, "iterations": 1, "noise..., `q_dim`=3, `rail_margin`=null, `random_episodes`=60, `round`=3, `round_episodes`=10, `stride`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep...

Runs (seed order): `learned_control/20261020-011651`, `learned_control/20261020-012041`, `learned_control/20261020-012126`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 347 ± 10.2 (3) | 352 / 353 / 335 |
| `upright_fraction_second_half` | 0.0189 ± 0.00743 (3) | 0.0272 / 0.0168 / 0.0128 |
| `decision_time_mean_s` | 0.535 ± 0.00389 (3) | 0.539 / 0.534 / 0.532 |

**Condition 13:** `collect_noise`=null, `episodes`=["runs/oracles/control_episode/20261019-203551-4.npz", "r..., `eval_seeds`=null, `form`="mlp", `horizon`=50, `mppi`=null, `q_dim`=null, `rail_margin`=0.2, `random_episodes`=60, `round`=0, `round_episodes`=null, `stride`=10, `train`=null

Runs (seed order): `prediction_error/20261019-205351-1`, `prediction_error/20261019-205351-2`, `prediction_error/20261019-205350-1`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `pred_error_1s` | 0.458 ± 0.0398 (3) | 0.503 / 0.443 / 0.428 |
| `pred_error_h10` | 0.0382 ± 0.00193 (3) | 0.0375 / 0.0404 / 0.0368 |
| `pred_error_h25` | 0.158 ± 0.00807 (3) | 0.154 / 0.168 / 0.153 |
| `pred_error_h50_q0` | 0.151 ± 0.0163 (3) | 0.141 / 0.142 / 0.17 |
| `pred_error_h50_q1` | 0.332 ± 0.0284 (3) | 0.323 / 0.309 / 0.364 |
| `pred_error_h50_q2` | 0.611 ± 0.125 (3) | 0.681 / 0.685 / 0.466 |

**Condition 14:** `collect_noise`=null, `episodes`=["runs/oracles/control_episode/20261019-203551-4.npz", "r..., `eval_seeds`=null, `form`="lagrangian", `horizon`=50, `mppi`=null, `q_dim`=null, `rail_margin`=0.2, `random_episodes`=60, `round`=0, `round_episodes`=null, `stride`=10, `train`=null

Runs (seed order): `prediction_error/20261019-205351`, `prediction_error/20261019-205350`, `prediction_error/20261019-205350-2`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `pred_error_1s` | 0.577 ± 0.116 (3) | 0.513 / 0.508 / 0.711 |
| `pred_error_h10` | 0.0503 ± 0.0137 (3) | 0.0514 / 0.0361 / 0.0634 |
| `pred_error_h25` | 0.205 ± 0.057 (3) | 0.185 / 0.161 / 0.27 |
| `pred_error_h50_q0` | 0.161 ± 0.0295 (3) | 0.157 / 0.192 / 0.134 |
| `pred_error_h50_q1` | 0.488 ± 0.123 (3) | 0.4 / 0.436 / 0.628 |
| `pred_error_h50_q2` | 0.644 ± 0.192 (3) | 0.555 / 0.514 / 0.864 |

**Condition 15:** `collect_noise`=null, `episodes`=["runs/oracles/control_episode/20261019-203551-1.npz", "r..., `eval_seeds`=null, `form`="lagrangian", `horizon`=50, `mppi`=null, `q_dim`=null, `rail_margin`=0.2, `random_episodes`=60, `round`=3, `round_episodes`=null, `stride`=10, `train`=null

Runs (seed order): `prediction_error/20261020-012205`, `prediction_error/20261020-012205-1`, `prediction_error/20261020-012205-2`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `pred_error_1s` | 0.24 ± 0.0255 (3) | 0.262 / 0.212 / 0.247 |
| `pred_error_h10` | 0.00782 ± 0.00105 (3) | 0.00891 / 0.00681 / 0.00776 |
| `pred_error_h25` | 0.0499 ± 0.0115 (3) | 0.0624 / 0.0398 / 0.0475 |
| `pred_error_h50_q0` | 0.0296 ± 0.00149 (3) | 0.0309 / 0.028 / 0.0301 |
| `pred_error_h50_q1` | 0.155 ± 0.0146 (3) | 0.168 / 0.139 / 0.158 |
| `pred_error_h50_q2` | 0.451 ± 0.0609 (3) | 0.51 / 0.389 / 0.453 |

**Condition 16:** `collect_noise`=null, `episodes`=["runs/oracles/control_episode/20261019-203551-1.npz", "r..., `eval_seeds`=null, `form`="mlp", `horizon`=50, `mppi`=null, `q_dim`=null, `rail_margin`=0.2, `random_episodes`=60, `round`=3, `round_episodes`=null, `stride`=10, `train`=null

Runs (seed order): `prediction_error/20261020-012205-3`, `prediction_error/20261020-012205-4`, `prediction_error/20261020-012205-5`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `pred_error_1s` | 0.232 ± 0.0149 (3) | 0.247 / 0.217 / 0.231 |
| `pred_error_h10` | 0.00935 ± 9.58e-04 (3) | 0.00943 / 0.0103 / 0.00836 |
| `pred_error_h25` | 0.0561 ± 0.00443 (3) | 0.0585 / 0.0587 / 0.0509 |
| `pred_error_h50_q0` | 0.0872 ± 0.0115 (3) | 0.0939 / 0.0938 / 0.0739 |
| `pred_error_h50_q1` | 0.168 ± 0.0153 (3) | 0.184 / 0.154 / 0.167 |
| `pred_error_h50_q2` | 0.381 ± 0.0326 (3) | 0.405 / 0.344 / 0.395 |

_Generated by `scripts/research/report.py`; std is the sample std over seeds._
<!-- report:end -->

Generated with `report.py E-010 --metric return_mean --metric
upright_fraction_second_half --metric decision_time_mean_s --metric
pred_error_1s --metric pred_error_h10 --metric pred_error_h25 --metric
pred_error_h50_q0 --metric pred_error_h50_q1 --metric pred_error_h50_q2`
(conditions are in run order). Main chains: MLP conditions 1, 2, 3, 5 and
Lagrangian 4, 9, 11, 12 (60 random episodes, rounds 0–3). Data-size sweep:
6–8 and 10. Prediction error: 13–14 (round 0, registered), 15–16 (round
3, **post hoc**, run after the returns were known; not used for the
decision).

Headline (mean over 3 seeds, 5 episodes each; E-009 ceiling 422):

| round | Lagrangian | MLP | Lagrangian − MLP | seeds with Lagrangian ahead |
|---|---|---|---|---|
| 0 | 193 ± 19 | 192 ± 23 | +1 | 1 of 3 |
| 1 | 243 ± 30 | 262 ± 19 | −19 | 1 of 3 |
| 2 | 323 ± 20 | 319 ± 28 | +4 | 2 of 3 |
| 3 | 347 ± 10 (0.82 of the ceiling) | 318 ± 19 (0.75) | +29 | 3 of 3 |

(This table restates the generated numbers above; the difference and the
seed counts are computed from the per-seed values.)

Reading against the predictions:
- **Round 0, Lagrangian ≥ 50 ahead (45%): wrong.** +1. The data-size
  sweep says the same at 20 (189 vs 185) and 200 random episodes (216 vs
  202). From random data alone neither model gets both poles up: the
  Lagrangian never has both upright in the second half, the MLP 0–1% of
  frames.
- **Both ≥ 0.85 of the ceiling by round 3 (Lagrangian 60%, MLP 50%):
  wrong for both.** 0.82 and 0.75. Both are below the half-solution
  (about 410), and both poles are upright in about 2% of the
  second-half frames (0–5% per seed; the true model: 16–22%). Neither learned model
  balances the two poles.
- **Round-0 1 s prediction error, Lagrangian ≤ half the MLP's (55%):
  wrong.** 0.58 ± 0.12 against 0.46 ± 0.04; the Lagrangian is worse at
  every horizon (0.2 s: 0.050 vs 0.038) and on every coordinate, most on
  the first pole (0.49 vs 0.33 rad). The structure does not carry to the
  swing-up states random data lacks. Post hoc, round-3 models: 0.240 vs
  0.232, a tie; both errors halve with the MPPI data.
- **Planning cost: right.** Lagrangian 0.54–0.62 s, MLP 0.074–0.089 s per
  decision: 7× more for the Lagrangian.
- **Late difference.** At round 3 the Lagrangian is ahead on all three
  seeds by 29 points (Welch t ≈ 2.3, about 3 degrees of freedom: not
  significant at 5%). It is outside the registered "advantage" window
  (rounds 0–1), below its 50-point threshold, and the post-hoc prediction
  error does not explain it. It is a hint, not a result.

Decision rule: "advantage" is not shown (round 0 +1, round 1 −19, and the
prediction error is worse, not half). No branch fits exactly: there is no
advantage, and both models end between 0.5 and 0.85 of the ceiling. The
nearest branch is "no advantage": report it as the second negative for the
structure, with its cost, and the owner decides between `three_poles` and
stopping the structure question. The ensemble fix is not triggered (it is
for both < 0.5). This is not a "diagnose further" outcome.

Cost: about 19 CPU process-hours (Lagrangian chains about 13, Lagrangian
sweep about 3, MLP chains and sweep about 2.5, prediction error under
0.1), inside the 25 budgeted; about 5 agent-hours.

## Belief update
- H-LAG-2P refuted at the registered threshold (C-020); H-LAG-PRED-2P
  refuted (C-021).
- Two systems, the same answer: with the same data, the learned
  Lagrangian is no better than an MLP and costs 7× more to plan with. On
  two poles it also extrapolates *worse* from random data, which is the
  case the structure was meant for. Possible reasons, not tested: a mass
  matrix and potential over two coupled angles, fitted only on states far
  from upright, can be wrong there in ways the energy form does not
  prevent; and the Lagrangian's errors grow faster over 1 s (they are
  already larger at 0.2 s).
- Step back: both learned models and the true model are limited by the
  planner (true model 422 of about 1000, half-solution about 410). As long
  as the planner cannot balance two poles, closed-loop returns compare
  "how well a model supports a weak planner", not "how well it knows the
  mechanics". A harder system (`three_poles`) would make the planner
  weaker still. The open-loop prediction error is the planner-free
  measure, and it shows no advantage.
- The late +29 at round 3 (3 of 3 seeds) is the only point in the
  Lagrangian's favour. Resolving it would need more seeds or more rounds
  at 7× the MLP's cost; not proposed without an owner decision.
