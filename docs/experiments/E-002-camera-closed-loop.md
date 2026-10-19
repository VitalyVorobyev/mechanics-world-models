# E-002 — Camera-in-the-loop control with a standard front end and state estimator

- **Status:** registered
- **Question node:** Q2
- **Hypotheses:** A-VIEW-1, A-FE-1
- **Budget:** 8 agent-hours, 4 CPU/MPS-hours (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** — (owner, date)

## Question
Replace the true state in the E-001 controller with what a camera gives.
The front end reads (x, θ) from each rendered frame; a state estimator
turns the pose history into (q, q̇) for the MPC. Does the closed-loop
return hold? This is Q0 at nominal parameters, and the most direct test
of "do we need to invent anything".

## Prediction
- **Field of view.** The default camera sees x ∈ ±1.66 m, while the rail
  spans ±1.8 m and the pole reaches 1 m beyond the cart (A-VIEW-1).
  Pulling the fixed camera back, or widening fovy, so that x ± 1 m stays
  in frame over the whole rail costs resolution: about 25.35 → 18 px/m
  at 84 px.
  - At 84 px the label-free heuristic (C-011) then reads θ at about 1.5–2°
    median in every frame. Confidence 65%.
  - At 128 px it reads about 1°.
- **Estimator.** An EKF over the known model, with the measured action
  and a Huber or gated innovation (outlier rejection), gives a velocity
  estimate within about 0.2 rad/s rms of the truth. Finite differences
  give about 2 rad/s at these noise levels (C-009). Confidence 60%.
- **Closed loop at nominal parameters:**
  - heuristic + EKF + E-001 MPC reaches at least 85% of the E-001
    true-state return at 128 px (confidence 60%), and at least 70% at
    84 px (confidence 50%);
  - finite-difference velocities instead of the EKF give below 50% at
    either resolution, because balancing near upright needs the angular
    velocity (confidence 70%).
  - Upright frames are rare in random data but are where balancing
    happens. The heuristic needs no training data, which avoids the
    supervised encoders' upright failure (A-ENC-1).

## Primary metric
Closed-loop return ratio: the mean return with the camera pipeline
divided by the mean E-001 true-state return at nominal parameters. Five
episodes (seeds 0–4) for each of: heuristic + EKF at 84 px; heuristic +
EKF at 128 px; heuristic + finite differences at 128 px.

Report: `return_ratio.heuristic_ekf_84`, `return_ratio.heuristic_ekf_128`, `return_ratio.heuristic_fd_128`, `pose_error.theta_median_deg_{res}`, `state_error.thetadot_rms_{res}`

## Baselines
- **Floor:** finite-difference velocities (the naive pipeline).
- **Black box:** published end-to-end pixel agents, about 800–860 in
  distribution. These are reference numbers, not run here.
- **Known model:** the E-001 controller and EKF model, at true nominal
  parameters.
- **Ceiling:** the E-001 true-state return; the estimator given true
  poses with added Gaussian noise of the measured size.
- **Effort symmetry:** EKF noise parameters are tuned on 3 nominal
  episodes (seeds 100–102, not the test seeds), with at most 6 settings
  and the same budget for each resolution.

## Decision rule
- **Ratio ≥ 0.85 at 128 px or at 84 px:** Q2 is closed: a standard
  pipeline drives cartpole from pixels. No invention needed. Go to E-003.
- **Ratio between 0.5 and 0.85:** attribute the loss. Swap in true poses
  with matched Gaussian noise, which separates front-end structure from
  noise level. Then do at most one targeted fix:
  - camera framing or resolution, if the loss is in pose noise;
  - E-005 robust estimation, if it is in outliers or runs of bad frames.

  Then report.
- **Ratio < 0.5 even at 128 px with true-pose-plus-noise:** the estimator
  or controller cannot handle realistic noise near upright. **Stop**
  adding front-end work. This is a genuine gap, and the owner decides
  whether it is worth inventing for (for example, an uncertainty-aware
  MPC).
- Threshold reason: 0.85 keeps the pipeline at the published level of
  end-to-end agents (about 0.85 × 900).

## Prerequisites
- E-001 done, with nominal return ≥ 800.
- A camera configuration that keeps the cart and the pole tip in view
  over the whole rail, checked by rung 0: a visibility fraction of 100%
  on 20 random episodes.
- The heuristic front end at the new framing, re-scored on held-out
  frames (rung 2).
- New code, committed and tested before any run:
  - `src/control/estimator.py` (EKF with a gated innovation);
  - the camera option in `src/envs/dm_control_pixels.py`.

## Plan
- `.venv/bin/python scripts/oracles/control_ceiling.py --experiment E-002 --observer heuristic --estimator ekf|fd --resolution 84|128 --seeds 0-4`
- Rendering adds about 2 ms per frame, which is negligible next to
  planning. The estimated total is under 4 hours.

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update
