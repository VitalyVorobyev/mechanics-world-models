# E-007 — Planning with learned mechanics on true state

- **Status:** done
- **Question node:** Q2
- **Hypotheses:** H-LAG-CTRL, H-LAG-DATA
- **Budget:** 8 agent-hours, 8 CPU-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** owner, 2026-10-19 (plan approval; each experiment starts only after its predecessor's decision rule allows it)

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
**Condition 1:** `form`="mlp", `round`=0

Runs (seed order): `learned_control/20261019-101041`, `learned_control/20261019-103007`, `learned_control/20261019-104930`
Superseded: `learned_control/20261019-095525`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 842 ± 3.15 (3) | 843 / 845 / 839 |

**Condition 2:** `form`="lagrangian", `round`=0

Runs (seed order): `learned_control/20261019-102526`, `learned_control/20261019-102520`, `learned_control/20261019-102525`
Superseded: `learned_control/20261019-100803`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 809 ± 70 (3) | 851 / 728 / 847 |

**Condition 3:** `form`="mlp", `round`=1

Runs (seed order): `learned_control/20261019-101616`, `learned_control/20261019-103544`, `learned_control/20261019-105505`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 853 ± 2.31 (3) | 851 / 856 / 853 |

**Condition 4:** `form`="mlp", `round`=2

Runs (seed order): `learned_control/20261019-102151`, `learned_control/20261019-104118`, `learned_control/20261019-110038`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 853 ± 2.06 (3) | 851 / 854 / 854 |

**Condition 5:** `form`="mlp", `round`=3

Runs (seed order): `learned_control/20261019-102732`, `learned_control/20261019-104654`, `learned_control/20261019-110928`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 853 ± 1.59 (3) | 851 / 854 / 853 |

**Condition 6:** `form`="lagrangian", `round`=1

Runs (seed order): `learned_control/20261019-110006`, `learned_control/20261019-105955`, `learned_control/20261019-110009`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 857 ± 1.27 (3) | 856 / 858 / 858 |

**Condition 7:** `form`="lagrangian", `round`=2

Runs (seed order): `learned_control/20261019-113837`, `learned_control/20261019-113820`, `learned_control/20261019-113842`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 856 ± 3.34 (3) | 852 / 858 / 858 |

**Condition 8:** `form`="lagrangian", `round`=3

Runs (seed order): `learned_control/20261019-121303`, `learned_control/20261019-121227`, `learned_control/20261019-121254`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 856 ± 2.78 (3) | 853 / 857 / 858 |

_Generated by `scripts/research/report.py`; std is the sample std over seeds._
<!-- report:end -->

Generated with `scripts/research/report.py E-007 --seeds 0-2` (training
seeds; each run's `return_mean` is the mean over evaluation episodes 0–4).
The round-0 pilot runs of seed 0 have the same config as the full runs
and are superseded by them (identical returns: the runs are deterministic).
Diagnostics from the same runs (`report.py --glob
'runs/oracles/learned_control/2026*.json' --where 'pose_source=null'
--seeds 0-2 --metric ...`): planning costs 0.21 s per decision with the
Lagrangian and 0.037 s with the MLP; training takes 485–500 s against
54 s; the pole is upright for the whole second half of every episode from
round 1 on (round 0, Lagrangian seed 1: 77%).

**Reading against the prediction.**
- Lagrangian ≥ 700 within 3 rounds (70%): **yes**, already at round 0
  (809 ± 70) and at the ceiling from round 1 (857 ± 1; E-006: 861 ± 3).
- MLP ≥ 700 within 3 rounds (55%): **yes**, at round 0 (842 ± 3). The
  feared model exploitation by the planner did not happen on this task.
- Lagrangian ahead at round 0 (55%): **no.** The MLP is at 842 ± 3, the
  Lagrangian at 809 ± 70; one of three Lagrangian seeds (728) is clearly
  below, the other two (851, 847) are level with the MLP. Random-policy
  data, with only 0.3% of frames near upright, is enough for both forms.
- Planning cost (0.2–0.4 s Lagrangian, about 0.05 s MLP): as predicted.

Decision rule: "Lagrangian ≥ 700 by round 3" → Q2 is closed; go to
E-008 with the round-3 Lagrangian.

## Belief update
- **Claims:** C-016 (grade A) added.
- **Hypotheses:** H-LAG-CTRL supported; H-LAG-DATA refuted for cartpole
  (no data-efficiency advantage of the structure; the MLP is as good from
  random data, 5.7× cheaper to plan with and 9× cheaper to train).
- **Step-back:** on cartpole, learning the mechanics is easy for both
  forms, so this task cannot show what the Lagrangian structure is worth.
  That question needs a harder system (more coupled degrees of freedom,
  less coverage), which is the backlog item for the stop-date decision.
  For the cartpole demo the learned Lagrangian works, as the owner asked.
