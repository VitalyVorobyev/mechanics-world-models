# E-008 — Camera in the loop: keypoints, estimator, learned Lagrangian

- **Status:** registered (awaiting owner approval)
- **Question node:** Q3
- **Hypotheses:** H-KP, H-CAM-LAG
- **Budget:** 12 agent-hours, 8 CPU-hours + 1 MPS-hour (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** — (owner, date)

## Question
Replace the true state with what a camera gives. Does the E-007
controller hold its return when its state comes from images, and when
its model is also learned from camera poses instead of true states? A
yes answers Q0 and is the demo.

## Prediction
Setup:
- **Camera:** camera 0 with a wider fovy, so that the whole rail
  (±1.8 m) and the pole (1 m) stay in view: about 18 px/m at 84 px,
  against 25.35 now (today the cart is out of view in 37% of frames).
- **Keypoint reader:** the keypoint encoder architecture of
  `src/oracles/encoder.py`, trained on about 50k frames rendered at
  uniformly sampled states (x in ±1.8 m, θ uniform), labels from the
  simulator; (x, θ) is computed from the cart and pole-tip keypoints.
- **Estimator:** a fixed-lag fit of (q, q̇) over the last 8–16 frames
  through the learned model (`shoot` in `src/oracles/start_state.py`),
  with frames rejected by an innovation gate.

Predictions:
- Keypoint median θ error ≤ 1° on held-out upright frames (cos θ > 0.9)
  and ≤ 1.5° overall. Confidence 70%. Uniform sampling covers the
  upright poses the earlier encoders lacked (4.4° and 17° there).
- **Arm A** (the E-007 model trained on true state, fed camera states):
  return ≥ 0.85 × the E-007 true-state return. Confidence 60%.
- **Arm B** (a model learned from camera poses, same rounds as E-007,
  fed camera states): return ≥ 0.85 × the E-007 return. Confidence 50%.
  Training on estimated poses adds noise to the regression targets
  (C-008: the Lagrangian tolerates Gaussian pose noise, not heavy tails).

## Primary metric
Arm B return ratio: the mean return of arm B divided by the mean E-007
round-3 Lagrangian return (3 training seeds × 5 episodes each). Arm A's
ratio and the keypoint errors are diagnostics.

Report: `return_mean`, `pose_error.theta_median_deg`, `pose_error.theta_median_deg_upright`

## Baselines
- **Floor:** finite-difference velocities from the keypoint poses instead
  of the model-based estimator (arm A model).
- **Black box:** the E-007 MLP, if it reached ≥ 500, in arm A.
- **Known model:** the E-006 true-model planner fed camera states (it
  separates estimation loss from model loss).
- **Ceiling:** the E-007 true-state return.
- **Effort symmetry:** estimator settings (window length, gate) are tuned
  on 3 episodes with seeds 100–102, at most 4 settings, and used
  unchanged in every arm.

## Decision rule
- **Arm B ratio ≥ 0.85:** Q3 and Q0 are closed: the demo works. Build the
  demo page and write the stop-date note.
- **Ratio between 0.5 and 0.85:** one attribution step: true poses with
  Gaussian noise matched to the keypoint error. Then one targeted fix,
  chosen by the attribution: 128 px rendering if the loss is pose noise,
  or a robust (Student-t) estimator if it is outliers. Then report.
- **Ratio < 0.5 even with matched-noise true poses:** **stop** adding
  perception work; the gap is in estimation or control under noise. The
  owner decides at the stop date.
- Threshold: 0.85 keeps the camera controller at about the published
  level of end-to-end pixel agents.

## Prerequisites
- E-007 done with a Lagrangian return ≥ 700 (or an owner decision).
- New code, committed and tested before any run: camera setting in
  `render_states` and the env; uniform-state rendering; keypoint
  training script; the estimator; the camera path in the control script.

## Plan
- Render and train the keypoint reader (about 10 min rendering, about
  1 MPS-hour training).
- Arm A, the known-model and floor runs: 5 episodes per training seed.
- Arm B: E-007 rounds on camera poses, 3 seeds.
- About 6–8 CPU-hours in all.
- Demo: mp4 and an HTML page (camera view with the estimated pose and
  planned trajectories, return over time, next to the true-state run).

## Result
<!-- report:begin -->
<!-- report:end -->

## Belief update
