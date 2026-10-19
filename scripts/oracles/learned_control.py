"""E-007, E-010: MPPI over a learned model (Lagrangian or MLP) on true state, with data rounds.

Round 0 trains on random-policy episodes only. After each round the
current model drives ``--round-episodes`` new episodes (MPPI plus Gaussian
action noise), and the next round retrains from scratch on all data so far.
Every round is evaluated on the evaluation seeds with the planner frozen for
the task (E-006 for one pole, E-009 for two), and writes one ledgered result (config ``round`` = r; metric
``return_mean`` = mean over the evaluation episodes).

Resumable: data, models and evaluations are cached per round under
``--cache-dir/<key>/`` (key = config + source fingerprint); rerunning the same
command skips finished work and does not rewrite finished results.

With ``--pose-source camera`` (E-008 arm B) the model never sees the true
state: training data are keypoint poses read from rendered frames (velocities
by central differences of those poses), and every evaluation and collection
episode runs the camera pipeline (render → reader → fixed-lag estimator
through the current model → MPPI).

Usage:
  .venv/bin/python scripts/oracles/learned_control.py --experiment E-007 --form lagrangian --seed 0
  .venv/bin/python scripts/oracles/learned_control.py --experiment E-007 --form mlp --seed 0
  .venv/bin/python scripts/oracles/learned_control.py --experiment E-010 --task two_poles --form lagrangian --seed 0
  .venv/bin/python scripts/oracles/learned_control.py --experiment E-008 --form lagrangian --seed 0 \
      --pose-source camera --reader runs/oracles/keypoint_reader/models/<stamp>.pt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from control.estimator import FixedLagEstimator, learned_dynamics
from control.learned import (
    LearnedBackend, TrainConfig, concat_states, episodes_to_states, make_model, run_episode, train_model, true_state,
)
from control.mppi import E006_PLANNER, E009_CEILING, E009_PLANNER, MPPI
from oracles.common import Criterion, OracleResult, StateEpisodes
from oracles.context import simulate_state_episodes
from oracles.dynamics import _cache_key
from research.provenance import add_run_args, begin_run
from research.report import parse_seeds

NAME = "learned_control"
# Per task: frozen planner, hypotheses, and the return the record's criterion asks for.
TASKS = {
    "swingup": (E006_PLANNER, ["H-LAG-CTRL", "H-LAG-DATA"], 700.0, "E-007: learned model reaches a working demo level"),
    "two_poles": (E009_PLANNER, ["H-LAG-2P"], 0.85 * E009_CEILING, "E-010: 0.85 of the E-009 ceiling"),
}


def _save_states(path: Path, data: StateEpisodes) -> None:
    np.savez_compressed(path, qpos=data.qpos, qvel=data.qvel, actions=data.actions, frame_dt=data.frame_dt)


def _load_states(path: Path) -> StateEpisodes:
    with np.load(path) as z:
        return StateEpisodes(qpos=z["qpos"], qvel=z["qvel"], actions=z["actions"], frame_dt=float(z["frame_dt"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(TASKS), default="swingup", help="dm_control cartpole task")
    parser.add_argument("--form", choices=["lagrangian", "mlp"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3, help="on-policy rounds after round 0")
    parser.add_argument("--random-episodes", type=int, default=60)
    parser.add_argument("--round-episodes", type=int, default=10)
    parser.add_argument("--collect-noise", type=float, default=0.3)
    parser.add_argument("--eval-seeds", default="0-4")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/oracles/learned_control/cache"))
    parser.add_argument("--pose-source", choices=["true", "camera"], default="true")
    parser.add_argument("--reader", type=Path, help="keypoint reader (camera pose source)")
    parser.add_argument("--window", type=int, default=12, help="estimator window (camera pose source)")
    add_run_args(parser)
    args = parser.parse_args()
    run = begin_run(args)
    torch.set_num_threads(args.threads)

    train_config = TrainConfig(steps=args.steps)
    eval_seeds = parse_seeds(args.eval_seeds)
    env_name = f"cartpole-{args.task}"
    planner_config, hypotheses, threshold, reason = TASKS[args.task]
    if args.pose_source == "camera" and args.task != "swingup":
        parser.error("the camera pose source supports one pole only")
    base_config = {
        "env": env_name,
        "form": args.form,
        "seed": args.seed,
        "random_episodes": args.random_episodes,
        "round_episodes": args.round_episodes,
        "collect_noise": args.collect_noise,
        "eval_seeds": eval_seeds,
        "train": asdict(train_config),
        "mppi": planner_config.to_dict(),
    }
    if args.task != "swingup":  # keeps the one-pole config (and cache keys' inputs) as in E-007 / E-008
        base_config["q_dim"] = 1 + {"two_poles": 2}[args.task]
    camera = None
    if args.pose_source == "camera":
        if args.reader is None:
            parser.error("--reader is required with --pose-source camera")
        from camera_control import load_reader  # same directory
        from control.vision import CameraState, observed_episodes

        reader, spec = load_reader(args.reader)
        camera = (reader, spec)
        base_config |= {"pose_source": "camera", "reader": str(args.reader), "estimator_window": args.window}
    cache = args.cache_dir / _cache_key(base_config)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "config.json").write_text(json.dumps(base_config, indent=2, sort_keys=True) + "\n")
    log = lambda m: print(f"[{args.form} s{args.seed}] {m}", flush=True)  # noqa: E731

    random_path = cache / "random.npz"
    if not random_path.exists():
        log(f"simulating {args.random_episodes} random-policy episodes")
        random_data = simulate_state_episodes(
            None, episodes=args.random_episodes, seed=10_000 + args.seed, env_name=env_name,
        )
        if camera is not None:
            random_data = observed_episodes(random_data, *camera)
        _save_states(random_path, random_data)
    parts = [_load_states(random_path)]

    for round_index in range(args.rounds + 1):
        round_dir = cache / f"round{round_index}"
        round_dir.mkdir(exist_ok=True)
        data = concat_states(parts)

        model_path = round_dir / "model.pt"
        if model_path.exists():
            model = make_model(args.form, train_config, data.frame_dt, q_dim=data.qpos.shape[-1])
            model.load_state_dict(torch.load(model_path))
            train_info = json.loads((round_dir / "train.json").read_text())
        else:
            log(f"round {round_index}: training on {data.qpos.shape[0]} episodes")
            start = time.perf_counter()
            model, loss = train_model(args.form, data, train_config, seed=args.seed)
            train_info = {"train_loss": loss, "train_time_s": time.perf_counter() - start, "episodes": data.qpos.shape[0]}
            torch.save(model.state_dict(), model_path)
            (round_dir / "train.json").write_text(json.dumps(train_info) + "\n")
            log(f"round {round_index}: trained in {train_info['train_time_s']:.0f}s, loss {loss:.2e}")
        backend = LearnedBackend(model)
        if camera is None:
            state_fn = true_state
        else:
            state_fn = CameraState(*camera, FixedLagEstimator(learned_dynamics(model), window=args.window))

        if not (round_dir / "result.json").exists():
            episodes = []
            for env_seed in eval_seeds:
                planner = MPPI(planner_config, seed=1000 * args.seed + env_seed)
                episode = run_episode(env_seed, planner, backend, state_fn, env_name=env_name)
                episodes.append(episode)
                log(f"round {round_index}: eval seed {env_seed} return {episode.total:.1f} "
                    f"({episode.plan_times.mean() * 1e3:.0f} ms/decision)")
            per_episode = [e.metrics() for e in episodes]
            returns = [m["return"] for m in per_episode]
            metrics = {
                "return_mean": float(np.mean(returns)),
                "returns": returns,
                **{k: float(np.mean([m[k] for m in per_episode]))
                   for k in ("decision_time_mean_s", "upright_fraction_second_half", "max_abs_x", "rail_fraction")},
                **train_info,
            }
            result = OracleResult(
                name=NAME,
                hypotheses=hypotheses if camera is None else ["H-CAM-LAG"],
                question="Closed-loop swing-up return of MPPI over a learned model, per data round",
                config={**base_config, "round": round_index},
                metrics=metrics,
                criteria=[Criterion("return_mean", ">=", threshold, reason)],
            )
            path = result.write(run=run)
            np.savez_compressed(
                round_dir / "eval_episodes.npz",
                qpos=np.stack([e.qpos for e in episodes]), qvel=np.stack([e.qvel for e in episodes]),
                actions=np.stack([e.actions for e in episodes]), rewards=np.stack([e.rewards for e in episodes]),
            )
            (round_dir / "result.json").write_text(json.dumps({"path": str(path), "return_mean": metrics["return_mean"]}) + "\n")
            log(f"round {round_index}: return {metrics['return_mean']:.1f}; wrote {path}")

        if round_index == args.rounds:
            break
        collected_path = round_dir / "collected.npz"
        if not collected_path.exists():
            episodes = []
            for i in range(args.round_episodes):
                env_seed = 20_000 + 1000 * args.seed + 100 * round_index + i
                planner = MPPI(planner_config, seed=env_seed)
                episode = run_episode(env_seed, planner, backend, state_fn,
                                      action_noise=args.collect_noise, noise_seed=env_seed, env_name=env_name)
                episodes.append(episode)
                log(f"round {round_index}: collected episode {i} return {episode.total:.1f}")
            collected = episodes_to_states(episodes, data.frame_dt)
            _save_states(collected_path, collected if camera is None else observed_episodes(collected, *camera))
        parts.append(_load_states(collected_path))


if __name__ == "__main__":
    main()
