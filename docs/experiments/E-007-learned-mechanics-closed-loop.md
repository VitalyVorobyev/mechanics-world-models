# E-007 — Planning with learned mechanics on true state

- **Status:** registered (awaiting owner approval)
- **Question node:** Q2
- **Hypotheses:** H-LAG-CTRL, H-LAG-DATA
- **Budget:** 8 agent-hours, 8 CPU-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** — (owner, date)

## Question
Replace the true dynamics in the E-006 planner with a learned model
trained on true states. Does a learned Lagrangian reach the ceiling? How
much of its own data does it need, and how does an unstructured MLP do
with the same budget? This is Q0 without the camera.

## Prediction
Models: the learned Lagrangian of the oracle harness (Cholesky mass
matrix, potential, linear dissipation, `actuated_dims=(0,)`, S¹ angle
embedding, semi-implicit Euler, 2 substeps of 0.01 s) and the MLP
dynamics model, both trained by multi-step rollout regression with the
existing defaults (15k steps), at nominal parameters, 3 training seeds.

Data rounds:
- round 0: 60 random-policy episodes;
- rounds 1–3: add 10 episodes per round, driven by MPPI over the current
  model with Gaussian action noise (std 0.3), then retrain from scratch.

Predictions:
- Lagrangian mean return ≥ 700 within 3 rounds. Confidence 70%.
- MLP mean return ≥ 700 within 3 rounds. Confidence 55%. A single MLP
  is prone to the planner exploiting its errors away from the data.
- At round 0 (random data: 0.3% of frames near upright), the Lagrangian
  is ahead of the MLP. Confidence 55%. The structure (gravity enters
  only through a potential of θ) should carry to upright states never
  seen; the MLP has no reason to.
- Planning cost: 0.2–0.4 s per decision for the Lagrangian at 1000 × 50,
  about 0.05 s for the MLP (measured on random weights).

## Primary metric
Mean episode return per (model, round), averaged over 3 training seeds,
each evaluated on 5 episodes (seeds 0–4). Headline: the return after
round 3, and the ratio to the E-006 ceiling.

Report: `return_mean`

## Baselines
- **Floor:** zero action; the E-006 floors.
- **Black box:** the MLP dynamics model, same data, same training steps,
  same planner.
- **Known model:** the E-006 true-model planner (ceiling).
- **Effort symmetry:** both models get the same data rounds, training
  budget, planner configuration (frozen in E-006), and the same single
  allowed fix below. Any change to one model's training is applied to
  both or logged as an asymmetry.

## Decision rule
- **Lagrangian ≥ 700 by round 3:** Q2 is closed. Go to E-008 with the
  round-3 Lagrangian.
- **Both models < 500 after round 3:** one allowed fix, the standard
  one: a 5-model ensemble per form, planning on the mean with an
  ensemble-disagreement penalty (PETS style), given to both models.
  If both stay below 500, **stop** E-007 and report; the owner decides.
- **Only the MLP ≥ 700:** report it as a negative for the structure on
  this task; go to E-008 with the MLP only if the owner agrees.
- **Lagrangian between 500 and 700:** one more data round (round 4),
  then accept the result as is and go to E-008.
- Thresholds: 700 is 85–90% of the expected ceiling, the level of a
  demo that visibly swings up and balances; 500 means the pole is
  upright for about half of the episode at best.

## Prerequisites
- E-006 done (planner configuration frozen, ceiling measured).
- New code, committed and tested before any run:
  - a model rollout returning q̇ as well as q, without autograd
    overhead in the planner;
  - a policy hook in the state-episode simulator so MPPI can collect data;
  - `scripts/oracles/learned_control.py` running rounds, cached by
    config + source hash, resumable per round.

## Plan
- `.venv/bin/python scripts/oracles/learned_control.py --experiment E-007 --form {lagrangian,mlp} --seed {0,1,2} --rounds 3`
- Cost: Lagrangian training about 9 min per round, data collection
  10 episodes × 2–3 min, evaluation 5 episodes per round. About 6–8
  CPU-hours in all; about 3 hours of wall-clock with seeds in parallel.
- Pilot rule: round 0 for one seed of each form first. If a Lagrangian
  episode takes more than 5 minutes, report before continuing.

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update
