"""E-008: closed-loop swing-up from camera images.

Each decision: render camera 0 (wide view) → keypoint reader → pose (x, θ) →
fixed-lag estimator (velocity from the last frames through a model) → MPPI
(planner frozen in E-006) over the planning model → env step. One run is one
episode (``--seed`` = environment seed).

Arms (config ``arm``):
  learned  planner and estimator use a learned model (``--model-round``)
  true     planner and estimator use the true MuJoCo model (estimation loss only)
  fd       learned planner, two-frame finite-difference velocity (floor)
Attribution (``--pose-source noisy-truth``): true poses + Gaussian noise
instead of the camera, to separate reader structure from noise level.

Usage:
  .venv/bin/python scripts/oracles/camera_control.py --experiment E-008 --arm learned \
      --model-round runs/oracles/learned_control/cache/<key>/round3 \
      --reader runs/oracles/keypoint_reader/models/<stamp>.pt --seed 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from control.camera import CameraSpec
from control.estimator import FixedLagEstimator, learned_dynamics, mujoco_dynamics
from control.keypoints import pose_error_metrics, read_poses
from control.learned import LearnedBackend, load_round_model
from control.mppi import E006_PLANNER, MPPI
from control.true_model import MujocoBackend
from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
from models.mechanics.keypoints import KeypointEncoder
from oracles.common import Criterion, OracleResult
from research.provenance import add_run_args, begin_run

NAME = "camera_control"


def load_reader(path: Path) -> tuple[KeypointEncoder, CameraSpec]:
    payload = torch.load(path, map_location="cpu")
    model = KeypointEncoder(num_keypoints=2)
    model.load_state_dict(payload["state_dict"])
    camera = payload["camera"]
    return model.eval(), CameraSpec(camera["fovy_deg"], camera["image_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=["learned", "true", "fd"], required=True)
    parser.add_argument("--model-round", type=Path, help="learned_control round directory (arms learned, fd)")
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pose-source", choices=["camera", "noisy-truth"], default="camera")
    parser.add_argument("--pose-noise", type=float, nargs=2, default=(0.0, 0.0), help="x (m), θ (deg) for noisy-truth")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--sigma", type=float, nargs=2, default=(0.02, 0.03))
    parser.add_argument("--gate", type=float, nargs=2, default=(0.15, 0.25))
    parser.add_argument("--threads", type=int, default=4)
    add_run_args(parser)
    args = parser.parse_args()
    run = begin_run(args)
    torch.set_num_threads(args.threads)

    reader, spec = load_reader(args.reader)
    env = DmControlPixelEnv(PixelEnvConfig(env_name="cartpole-swingup", seed=args.seed, action_repeat=2, image_size=spec.image_size))
    spec.apply(env.physics)
    mj_model = env.physics.model.ptr
    if args.arm == "true":
        backend, dynamics = MujocoBackend(mj_model, 2, args.threads), mujoco_dynamics(mj_model, 2)
        to_backend = lambda q, v: np.concatenate([[0.0], q, v])  # noqa: E731  (full physics state)
    else:
        if args.model_round is None:
            parser.error("--model-round is required for arms learned and fd")
        model = load_round_model(args.model_round)
        backend, dynamics = LearnedBackend(model), learned_dynamics(model)
        to_backend = lambda q, v: (q, v)  # noqa: E731
    estimator = FixedLagEstimator(dynamics, window=args.window, sigma=tuple(args.sigma), gate=tuple(args.gate))
    planner = MPPI(E006_PLANNER, seed=args.seed)
    rng = np.random.default_rng(args.seed + 7)

    env.reset_without_render()
    qpos, qvel, poses, estimates, actions, rewards, times = [], [], [], [], [], [], []
    last_action, previous_pose = None, None
    while True:
        truth = env.physics_state()
        start = time.perf_counter()
        if args.pose_source == "camera":
            pose = read_poses(reader, env.render()[None], spec)[0]
        else:
            pose = truth["qpos"] + rng.normal(0.0, [args.pose_noise[0], np.radians(args.pose_noise[1])])
        if args.arm == "fd":
            if previous_pose is None:
                q, v = pose, np.zeros(2)
            else:
                pose[1] = previous_pose[1] + np.angle(np.exp(1j * (pose[1] - previous_pose[1])))
                q, v = pose, (pose - previous_pose) / dynamics.dt
            previous_pose = pose
        else:
            q, v = estimator.update(pose, last_action)
        if env.episode_done():
            break
        action = planner.plan(backend, to_backend(q, v))
        times.append(time.perf_counter() - start)
        _, reward = env.step_state(action)
        qpos.append(truth["qpos"]), qvel.append(truth["qvel"]), poses.append(pose)
        estimates.append(np.concatenate([q, v])), actions.append(action), rewards.append(reward)
        last_action = action
    env.close()

    qpos_a, qvel_a, est = np.asarray(qpos), np.asarray(qvel), np.asarray(estimates)
    pose_err = pose_error_metrics(np.asarray(poses), qpos_a, spec)
    vel_err = np.abs(est[:, 2:] - qvel_a)
    cos_theta = np.cos(qpos_a[:, 1])
    metrics = {
        "return_mean": float(np.sum(rewards)),
        "decision_time_mean_s": float(np.mean(times)),
        "upright_fraction_second_half": float(np.mean(cos_theta[len(cos_theta) // 2:] > 0.95)),
        "max_abs_x": float(np.abs(qpos_a[:, 0]).max()),
        **{f"pose_error.{k}": val for k, val in pose_err.items()},
        "state_error.xdot_median": float(np.median(vel_err[:, 0])),
        "state_error.thetadot_median": float(np.median(vel_err[:, 1])),
        "state_error.thetadot_p90": float(np.percentile(vel_err[:, 1], 90)),
        "estimator_rejected": estimator.rejected,
    }
    config = {
        "env": "cartpole-swingup",
        "arm": args.arm,
        "seed": args.seed,
        "model_round": str(args.model_round) if args.model_round else None,
        "reader": str(args.reader),
        "camera": {"fovy_deg": spec.fovy_deg, "image_size": spec.image_size},
        "pose_source": args.pose_source,
        "pose_noise": list(args.pose_noise) if args.pose_source == "noisy-truth" else None,
        "estimator": {"window": args.window, "sigma": list(args.sigma), "gate": list(args.gate)},
        "mppi": E006_PLANNER.to_dict(),
    }
    result = OracleResult(
        name=NAME,
        hypotheses=["H-CAM-LAG", "H-KP"],
        question="Closed-loop swing-up return from camera images (keypoints + fixed-lag estimator + MPPI)",
        config=config,
        metrics=metrics,
        criteria=[Criterion("return_mean", ">=", 700.0, "camera demo at a working level")],
    )
    path = result.write(run=run)
    np.savez_compressed(path.with_suffix(".npz"), qpos=qpos_a, qvel=qvel_a, poses=np.asarray(poses),
                        estimates=est, actions=np.asarray(actions), rewards=np.asarray(rewards))
    print(result.summary())
    print(f"return {metrics['return_mean']:.1f}  θ err median {pose_err['theta_median_deg']:.2f}°  "
          f"θ̇ err median {metrics['state_error.thetadot_median']:.3f}  {metrics['decision_time_mean_s'] * 1e3:.0f} ms/decision")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
