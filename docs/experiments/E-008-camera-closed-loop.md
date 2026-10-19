# E-008 — Camera in the loop: keypoints, estimator, learned Lagrangian

- **Status:** done
- **Question node:** Q3
- **Hypotheses:** H-KP, H-CAM-LAG
- **Budget:** 12 agent-hours, 8 CPU-hours + 1 MPS-hour (stop and report at 150%)
- **Proposed:** 2026-10-19
- **Approved:** owner, 2026-10-19 (plan approval; each experiment starts only after its predecessor's decision rule allows it)

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
**Condition 1:** `arm`="true", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`=null, `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-113017`, `camera_control/20261019-113108`, `camera_control/20261019-113200`, `camera_control/20261019-113251`, `camera_control/20261019-113341`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 840 ± 3.95 (5) | 841 / 844 / 833 / 840 / 840 |
| `pose_error.theta_median_deg` | 0.21 ± 0.0137 (5) | 0.188 / 0.212 / 0.226 / 0.211 / 0.216 |
| `pose_error.theta_median_deg_upright` | 0.219 ± 0.0224 (5) | 0.184 / 0.211 / 0.239 / 0.231 / 0.232 |

**Condition 2:** `arm`="learned", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`="runs/oracles/learned_control/cache/84c9864c6a/round3", `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-121608`, `camera_control/20261019-121840`, `camera_control/20261019-122112`, `camera_control/20261019-122344`, `camera_control/20261019-122618`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 839 ± 3.01 (5) | 844 / 837 / 838 / 839 / 836 |
| `pose_error.theta_median_deg` | 0.213 ± 0.0114 (5) | 0.2 / 0.229 / 0.204 / 0.219 / 0.212 |
| `pose_error.theta_median_deg_upright` | 0.222 ± 0.013 (5) | 0.22 / 0.238 / 0.204 / 0.23 / 0.217 |

**Condition 3:** `arm`="learned", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`="runs/oracles/learned_control/cache/74b95c49c7/round3", `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-122850`, `camera_control/20261019-123121`, `camera_control/20261019-123356`, `camera_control/20261019-123628`, `camera_control/20261019-123859`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 839 ± 4.33 (5) | 843 / 842 / 836 / 838 / 833 |
| `pose_error.theta_median_deg` | 0.214 ± 0.0117 (5) | 0.226 / 0.224 / 0.216 / 0.201 / 0.203 |
| `pose_error.theta_median_deg_upright` | 0.223 ± 0.0139 (5) | 0.24 / 0.233 / 0.222 / 0.212 / 0.207 |

**Condition 4:** `arm`="learned", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`="runs/oracles/learned_control/cache/c3f4bae45c/round3", `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-124131`, `camera_control/20261019-124406`, `camera_control/20261019-124639`, `camera_control/20261019-124912`, `camera_control/20261019-125148`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 840 ± 4.36 (5) | 842 / 846 / 836 / 840 / 836 |
| `pose_error.theta_median_deg` | 0.216 ± 0.0256 (5) | 0.186 / 0.214 / 0.257 / 0.216 / 0.21 |
| `pose_error.theta_median_deg_upright` | 0.227 ± 0.0222 (5) | 0.199 / 0.219 / 0.26 / 0.223 / 0.235 |

**Condition 5:** `arm`="learned", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`="runs/oracles/learned_control/cache/2feeb78153/round3", `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-125214`, `camera_control/20261019-125240`, `camera_control/20261019-125305`, `camera_control/20261019-125331`, `camera_control/20261019-125358`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 832 ± 3.39 (5) | 837 / 834 / 828 / 831 / 832 |
| `pose_error.theta_median_deg` | 0.23 ± 0.0104 (5) | 0.226 / 0.233 / 0.246 / 0.229 / 0.218 |
| `pose_error.theta_median_deg_upright` | 0.239 ± 0.00909 (5) | 0.238 / 0.242 / 0.249 / 0.243 / 0.225 |

**Condition 6:** `arm`="learned", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`="runs/oracles/learned_control/cache/7f333f4dc8/round3", `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-125424`, `camera_control/20261019-125450`, `camera_control/20261019-125517`, `camera_control/20261019-125543`, `camera_control/20261019-125608`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 831 ± 3.43 (5) | 832 / 833 / 826 / 834 / 827 |
| `pose_error.theta_median_deg` | 0.219 ± 0.0166 (5) | 0.233 / 0.193 / 0.234 / 0.221 / 0.216 |
| `pose_error.theta_median_deg_upright` | 0.229 ± 0.0208 (5) | 0.246 / 0.195 / 0.245 / 0.232 / 0.229 |

**Condition 7:** `arm`="learned", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`="runs/oracles/learned_control/cache/b3dc3eea1e/round3", `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-125634`, `camera_control/20261019-125659`, `camera_control/20261019-125724`, `camera_control/20261019-125749`, `camera_control/20261019-125814`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 831 ± 3.84 (5) | 836 / 832 / 826 / 831 / 828 |
| `pose_error.theta_median_deg` | 0.224 ± 0.00885 (5) | 0.214 / 0.235 / 0.221 / 0.23 / 0.217 |
| `pose_error.theta_median_deg_upright` | 0.235 ± 0.0148 (5) | 0.222 / 0.257 / 0.224 / 0.243 / 0.227 |

**Condition 8:** `arm`="fd", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`="runs/oracles/learned_control/cache/84c9864c6a/round3", `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-130003`, `camera_control/20261019-130151`, `camera_control/20261019-130340`, `camera_control/20261019-130528`, `camera_control/20261019-130730`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 788 ± 22.3 (5) | 794 / 783 / 809 / 752 / 802 |
| `pose_error.theta_median_deg` | 0.224 ± 0.00633 (5) | 0.221 / 0.235 / 0.223 / 0.219 / 0.223 |
| `pose_error.theta_median_deg_upright` | 0.235 ± 0.00658 (5) | 0.244 / 0.238 / 0.226 / 0.233 / 0.235 |

**Condition 9:** `arm`="learned", `batch_size`=null, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84}, `collect_noise`=null, `device`=null, `env`="cartpole-swingup", `estimator`={"gate": [0.15, 0.25], "sigma": [0.02, 0.03], "window": 12}, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=null, `model_round`="runs/oracles/learned_control/cache/bbc9fa6dd4/round3", `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=null, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=null, `round_episodes`=null, `steps`=null, `train`=null, `train_frames`=null, `val_frames`=null

Runs (seed order): `camera_control/20261019-145151`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 838 ± nan (1) | 838 |
| `pose_error.theta_median_deg` | 0.216 ± nan (1) | 0.216 |
| `pose_error.theta_median_deg_upright` | 0.229 ± nan (1) | 0.229 |

**Condition 10:** `arm`=null, `batch_size`=64, `camera`={"fovy_deg": 71.88422374276468, "image_size": 84, "pixels..., `collect_noise`=null, `device`="mps", `env`=null, `estimator`=null, `estimator_window`=null, `eval_seeds`=null, `form`=null, `lr`=0.001, `model_round`=null, `mppi`=null, `pose_source`=null, `random_episodes`=null, `reader`=null, `round`=null, `round_episodes`=null, `steps`=20000, `train`=null, `train_frames`=50000, `val_frames`=5000

Runs (seed order): `keypoint_reader/20261019-111217`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | nan ± nan (0) | nan |
| `pose_error.theta_median_deg` | 0.182 ± nan (1) | 0.182 |
| `pose_error.theta_median_deg_upright` | 0.186 ± nan (1) | 0.186 |

**Condition 11:** `arm`=null, `batch_size`=null, `camera`=null, `collect_noise`=0.3, `device`=null, `env`="cartpole-swingup", `estimator`=null, `estimator_window`=12, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `lr`=null, `model_round`=null, `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=60, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=0, `round_episodes`=10, `steps`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep..., `train_frames`=null, `val_frames`=null

Runs (seed order): `learned_control/20261019-115724`, `learned_control/20261019-123509-1`, `learned_control/20261019-123509`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 700 ± 194 (3) | 834 / 477 / 787 |
| `pose_error.theta_median_deg` | nan ± nan (0) | nan / nan / nan |
| `pose_error.theta_median_deg_upright` | nan ± nan (0) | nan / nan / nan |

**Condition 12:** `arm`=null, `batch_size`=null, `camera`=null, `collect_noise`=0.3, `device`=null, `env`="cartpole-swingup", `estimator`=null, `estimator_window`=12, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `lr`=null, `model_round`=null, `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=60, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=1, `round_episodes`=10, `steps`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep..., `train_frames`=null, `val_frames`=null

Runs (seed order): `learned_control/20261019-124342`, `learned_control/20261019-131919`, `learned_control/20261019-131914`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 828 ± 21.1 (3) | 805 / 845 / 835 |
| `pose_error.theta_median_deg` | nan ± nan (0) | nan / nan / nan |
| `pose_error.theta_median_deg_upright` | nan ± nan (0) | nan / nan / nan |

**Condition 13:** `arm`=null, `batch_size`=null, `camera`=null, `collect_noise`=0.3, `device`=null, `env`="cartpole-swingup", `estimator`=null, `estimator_window`=12, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `lr`=null, `model_round`=null, `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=60, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=2, `round_episodes`=10, `steps`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep..., `train_frames`=null, `val_frames`=null

Runs (seed order): `learned_control/20261019-133045`, `learned_control/20261019-140414`, `learned_control/20261019-140407`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 837 ± 8.25 (3) | 829 / 846 / 835 |
| `pose_error.theta_median_deg` | nan ± nan (0) | nan / nan / nan |
| `pose_error.theta_median_deg_upright` | nan ± nan (0) | nan / nan / nan |

**Condition 14:** `arm`=null, `batch_size`=null, `camera`=null, `collect_noise`=0.3, `device`=null, `env`="cartpole-swingup", `estimator`=null, `estimator_window`=12, `eval_seeds`=[0, 1, 2, 3, 4], `form`="lagrangian", `lr`=null, `model_round`=null, `mppi`={"action_dim": 1, "horizon": 75, "iterations": 1, "noise_..., `pose_source`="camera", `random_episodes`=60, `reader`="runs/oracles/keypoint_reader/models/20261019-111217.pt", `round`=3, `round_episodes`=10, `steps`=null, `train`={"batch_size": 128, "hidden": 64, "lr": 0.001, "n_substep..., `train_frames`=null, `val_frames`=null

Runs (seed order): `learned_control/20261019-141842`, `learned_control/20261019-144851`, `learned_control/20261019-144836`

| metric | mean ± std (n) | per seed |
|---|---|---|
| `return_mean` | 836 ± 9.13 (3) | 829 / 846 / 833 |
| `pose_error.theta_median_deg` | nan ± nan (0) | nan / nan / nan |
| `pose_error.theta_median_deg_upright` | nan ± nan (0) | nan / nan / nan |

_Generated by `scripts/research/report.py`; std is the sample std over seeds._
<!-- report:end -->

Generated with `scripts/research/report.py E-008 --seeds 0-4` (evaluation
seeds; the estimator tuning runs on seeds 100–102 are left out). How to read
the conditions:
- `arm="true"`: the true MuJoCo model in planner and estimator, fed camera
  poses;
- `arm="learned"` with `model_round` `84c9864c6a`, `74b95c49c7`,
  `c3f4bae45c`: **arm A**, the three E-007 round-3 Lagrangians (training
  seeds 0, 1, 2); with `2feeb78153`, `7f333f4dc8`, `b3dc3eea1e`: the three
  E-007 round-3 MLPs; with `bbc9fa6dd4`: one demo episode of the arm-B
  seed-0 model (not part of any comparison);
- `arm="fd"`: two-frame finite-difference velocities (floor);
- `pose_source="camera"` with `round`: **arm B**, the Lagrangian learned
  from camera poses only (training seeds 0, 1, 2; each run is the mean of
  5 episodes);
- the `steps=20000` condition is the keypoint reader itself.

**Reading against the prediction.**
- Keypoint reader (≤ 1° upright, 70%; ≤ 1.5° overall): **yes, far
  better**: 0.19° median upright and 0.18° overall on 5000 held-out
  uniform states, 1.1° worst case, 0.04 px cart error. In the loop it reads
  0.21–0.23° median.
- Arm A ≥ 0.85 (60%): **yes.** The three E-007 Lagrangians reach 839,
  839 and 840 from camera images, a ratio of 0.98 to their true-state
  856 ± 3.
- **Arm B ≥ 0.85 (50%), the primary metric: yes.** The Lagrangian learned
  only from camera poses reaches 836 ± 9 after round 3 (829 / 846 / 833):
  ratio **0.977**. From random data alone (round 0) it is 700 ± 194 (834 /
  477 / 787); one round of its own camera episodes brings every seed to
  805–845.
- Brackets: the true model fed camera states gets 840 ± 4 (the cost of
  estimation alone is about 2.5% of 861); the MLPs fed camera states get
  831–832 (0.975 of their 853); finite-difference velocities give 788 ± 22
  (0.92). The model-based estimator reads θ̇ to 0.03 rad/s median against
  0.17 rad/s for finite differences.

**Estimator tuning (not pre-registered choice):** window 12 against 16 on
seeds 100–102 (model: E-007 seed-0 round 1) gave 845 and 845; window 12 was
kept because it is cheaper and was the default.

**Decision rule:** ratio ≥ 0.85 → Q3 and Q0 are closed; the demo works.

## Belief update
- **Claims:** C-017 (keypoint reader, grade B: one training seed) and
  C-018 (camera closed loop, grade A) added.
- **Hypotheses:** H-KP supported; H-CAM-LAG supported.
- **Step-back:** Q0 is answered yes for cartpole at nominal parameters.
  Nothing had to be invented: keypoints trained on simulator renders, a
  fixed-lag model-based estimator and MPPI are all standard. The camera
  costs about 2–3% of the return. What cartpole cannot show is the value of
  the Lagrangian structure itself (C-016: the MLP is as good). The next
  question for the owner is the harder system.
