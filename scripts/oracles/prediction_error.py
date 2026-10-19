"""E-010 (exploratory): open-loop prediction error of learned models on held-out episodes.

The held-out episodes are true-model MPPI episodes (``control_episode.py``
trajectories, e.g. the E-009 evaluation runs): swing-up and balance, states
random-policy data does not visit. For each model (a ``learned_control.py``
round), windows start every ``--stride`` frames and roll ``--horizon``
decisions open loop from the true start state under the recorded actions.
The error is the mean over coordinates of ``|Δq|`` (angles wrapped) at the
horizon; ``pred_error_1s`` is its median over windows (horizon 50 = 1 s).

Usage:
  .venv/bin/python scripts/oracles/prediction_error.py --experiment E-010 \
      --cache runs/oracles/learned_control/cache/<key> --round 0 \
      --episodes runs/oracles/control_episode/<stamp>.npz [...]

Writes one runs/oracles/prediction_error/<timestamp>.json per model (seconds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from control.learned import load_round_model
from oracles.common import OracleResult, StateEpisodes
from oracles.dynamics import _windows
from research.provenance import add_run_args, begin_run

NAME = "prediction_error"


def held_out(paths: list[Path], frame_dt: float = 0.02) -> StateEpisodes:
    loaded = [np.load(p) for p in paths]
    return StateEpisodes(
        qpos=np.stack([z["qpos"] for z in loaded]),
        qvel=np.stack([z["qvel"] for z in loaded]),
        actions=np.stack([z["actions"] for z in loaded]).reshape(len(loaded), -1, 1),
        frame_dt=frame_dt,
    )


def wrapped_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``|Δq| [..., D]`` with every coordinate after the first (pole angles) wrapped to (−π, π]."""

    d = pred - target
    angles = torch.atan2(torch.sin(d[..., 1:]), torch.cos(d[..., 1:]))
    return torch.cat([d[..., :1], angles], dim=-1).abs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, nargs="+", required=True, help="learned_control cache directories")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True, help="held-out trajectory .npz files")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--rail-margin", type=float, default=0.2)
    add_run_args(parser)
    args = parser.parse_args()
    run = begin_run(args)

    episodes = held_out(args.episodes)
    q0, v0, act, tgt = _windows(episodes, args.horizon, stride=args.stride, rail_margin=args.rail_margin)
    marks = sorted({10, 25, args.horizon})
    for cache in args.cache:
        config = json.loads((cache / "config.json").read_text())
        model = load_round_model(cache / f"round{args.round}")
        with torch.enable_grad():  # the Lagrangian step differentiates internally
            pred = model.rollout(q0, v0, act).detach()
        err = wrapped_error(pred, tgt)                      # [W, H, D]
        per_window = err.mean(dim=-1)                       # [W, H]
        metrics = {
            "pred_error_1s": float(per_window[:, args.horizon - 1].median()),
            **{f"pred_error_h{h}": float(per_window[:, h - 1].median()) for h in marks},
            **{f"pred_error_h{args.horizon}_q{i}": float(err[:, args.horizon - 1, i].median())
               for i in range(err.shape[-1])},
            "windows": int(q0.shape[0]),
        }
        result = OracleResult(
            name=NAME,
            hypotheses=["H-LAG-PRED-2P"],
            question="Open-loop prediction error on held-out true-model episodes (states random data does not visit)",
            config={
                "env": config["env"], "form": config["form"], "seed": config["seed"],
                "random_episodes": config["random_episodes"], "round": args.round, "cache": cache.name,
                "episodes": [str(p) for p in args.episodes], "horizon": args.horizon, "stride": args.stride,
                "rail_margin": args.rail_margin,
            },
            metrics=metrics,
            criteria=[],
        )
        path = result.write(run=run)
        print(f"{config['form']} s{config['seed']} round {args.round}: "
              f"pred_error_1s {metrics['pred_error_1s']:.4f} ({metrics['windows']} windows) -> {path}")


if __name__ == "__main__":
    main()
