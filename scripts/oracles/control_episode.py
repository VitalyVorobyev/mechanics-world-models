"""Closed-loop cartpole swing-up episodes with an MPPI planner (E-006 onward).

One run = one evaluation episode (``--seed`` sets the environment's initial
state and the planner noise). The result records the env's own return,
planning cost and rail use; the trajectory is saved next to it for demos.

Usage:
  .venv/bin/python scripts/oracles/control_episode.py --experiment E-006 --dynamics true --seed 0
  .venv/bin/python scripts/oracles/control_episode.py --experiment E-006 --dynamics zero --seed 0

Writes runs/oracles/control_episode/<timestamp>.json (true model: about 20-60 s per episode).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from control.mppi import MPPI, MPPIConfig
from control.true_model import MujocoBackend, get_state
from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
from oracles.common import Criterion, OracleResult
from research.provenance import add_run_args, begin_run

NAME = "control_episode"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dynamics", choices=["true", "zero", "random"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    defaults = MPPIConfig()
    parser.add_argument("--horizon", type=int, default=defaults.horizon)
    parser.add_argument("--samples", type=int, default=defaults.samples)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--noise-std", type=float, default=defaults.noise_std)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--rail-limit", type=float, default=defaults.rail_limit)
    parser.add_argument("--rail-weight", type=float, default=defaults.rail_weight)
    add_run_args(parser)
    args = parser.parse_args()
    run = begin_run(args)

    env = DmControlPixelEnv(PixelEnvConfig(env_name="cartpole-swingup", seed=args.seed, action_repeat=args.action_repeat))
    env.reset_without_render()
    physics = env.physics
    mppi_config = MPPIConfig(
        horizon=args.horizon, samples=args.samples, iterations=args.iterations, noise_std=args.noise_std,
        temperature=args.temperature, rail_limit=args.rail_limit, rail_weight=args.rail_weight,
    )
    planner = MPPI(mppi_config, seed=args.seed)
    backend = MujocoBackend(physics.model.ptr, args.action_repeat, args.threads) if args.dynamics == "true" else None
    rng = np.random.default_rng(args.seed)

    qpos, qvel, actions, rewards, plan_times = [physics.data.qpos.copy()], [physics.data.qvel.copy()], [], [], []
    while not env.episode_done():
        start = time.perf_counter()
        if args.dynamics == "true":
            action = planner.plan(backend, get_state(physics.model.ptr, physics.data.ptr))
        elif args.dynamics == "zero":
            action = np.zeros(1)
        else:
            action = rng.uniform(-1.0, 1.0, size=1)
        plan_times.append(time.perf_counter() - start)
        state, reward = env.step_state(action)
        qpos.append(state["qpos"]), qvel.append(state["qvel"]), actions.append(action), rewards.append(reward)

    qpos_a, qvel_a = np.asarray(qpos), np.asarray(qvel)
    cos_theta = np.cos(qpos_a[1:, 1])
    second_half = cos_theta[len(cos_theta) // 2:]
    metrics = {
        "return_mean": float(np.sum(rewards)),
        "decisions": len(actions),
        "decision_time_mean_s": float(np.mean(plan_times)),
        "upright_fraction_second_half": float(np.mean(second_half > 0.95)),
        "max_abs_x": float(np.max(np.abs(qpos_a[:, 0]))),
        "rail_fraction": float(np.mean(np.abs(qpos_a[:, 0]) > 1.75)),
    }
    config = {
        "env": "cartpole-swingup",
        "dynamics": args.dynamics,
        "seed": args.seed,
        "action_repeat": args.action_repeat,
        "mppi": mppi_config.to_dict() if args.dynamics == "true" else None,
    }
    result = OracleResult(
        name=NAME,
        hypotheses=["H-PLAN"],
        question="Closed-loop swing-up return of an MPPI planner over the given dynamics backend",
        config=config,
        metrics=metrics,
        criteria=[Criterion("return_mean", ">=", 800.0, "E-006: planner ceiling at the level of published agents")],
    )
    path = result.write(run=run)
    traj = path.with_suffix(".npz")
    np.savez_compressed(traj, qpos=qpos_a, qvel=qvel_a, actions=np.asarray(actions), rewards=np.asarray(rewards))
    print(result.summary())
    print(f"return {metrics['return_mean']:.1f}  {metrics['decision_time_mean_s'] * 1e3:.0f} ms/decision")
    print(f"wrote {path} and {traj.name}")


if __name__ == "__main__":
    main()
