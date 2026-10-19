# E-003 — Online parameter identification from camera-estimated states

- **Status:** registered
- **Question node:** Q3
- **Hypotheses:** H1b-px
- **Budget:** 6 agent-hours, 4 CPU-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** — (owner, date)

## Question
This is the "mechanics-aware" part of the goal. After the plant changes
(cart mass, pole length, gear), does the E-002 camera pipeline recover
its return by re-fitting a few physical parameters from its own estimated
states? With true state this is E-001; here the fit sees estimator error
and closed-loop (correlated) data.

## Prediction
- With camera-estimated states, the online fit's parameter error is 2–4×
  larger than with true states, because estimator noise enters both the
  regressors and the targets (errors in variables). Confidence 60%.
- Fitting the parameters jointly with the EKF state (an augmented-state
  EKF, a textbook method) halves that error compared with fitting on the
  EKF's state estimates. Confidence 50%.
- Closed loop: the adaptation recovery (as defined in E-001) is at least
  0.7 for every shift where the nominal controller drops by more than
  10%, using the better of the two estimators. Confidence 55%.

## Primary metric
The E-001 adaptation recovery, computed with the camera pipeline (best
E-002 resolution), mean over the qualifying shifts, 5 episodes per
condition (seeds 0–4).

Report: `adaptation_recovery.camera_ls`, `adaptation_recovery.camera_augmented_ekf`, `param_error.{shift}.camera_ls`, `param_error.{shift}.camera_augmented_ekf`, `param_error.{shift}.true_state`

## Baselines
- **Floor:** the nominal controller without adaptation.
- **Black box:** none run. The reference is cRSSM: pixel Dreamer
  extrapolates poorly in physical parameters.
- **Known model:** this experiment is the known-model adaptation.
- **Ceiling:** the E-001 true-state online fit, and the controller given
  the true parameters.
- **Effort symmetry:** both estimators get the same windows, the same
  refit schedule, and the same tuning budget (≤ 4 settings each, on seeds
  100–102).

## Decision rule
- **Recovery ≥ 0.7:** Q3 closed. The goal is met for cartpole with
  standard components. Go to D-1 with "no invention needed". Then the
  owner chooses: harden the pipeline, add a second system (the backlog),
  or stop.
- **Recovery between 0.3 and 0.7:** identify which parameter fails, from
  the per-parameter errors against the true-state fit. Do at most one
  targeted step: more excitation (a probing action dither), or the
  augmented-state EKF if it was not the better arm. Then report.
- **Recovery < 0.3 with both estimators:** identification from camera
  states is the measured gap. **Stop** the pipeline work. This is the
  first place where something non-standard might be justified: robust
  errors-in-variables identification (E-005), or a learned model (E-004
  line). The owner decides.
- Threshold reason: 0.7 keeps most of the benefit of knowing the
  parameters.

## Prerequisites
- E-001 (recovery ≥ 0.8 on true state) and E-002 (ratio ≥ 0.85) done.
- New code, committed and tested before any run: an augmented-state EKF
  option in `src/control/estimator.py`.

## Plan
- `.venv/bin/python scripts/oracles/control_ceiling.py --experiment E-003 --observer heuristic --estimator ekf --sysid ls|augmented --shift … --seeds 0-4`

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update
