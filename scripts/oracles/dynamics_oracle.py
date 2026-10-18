"""Rung 3 CLI: Lagrangian vs MLP dynamics on true state, few-shot adaptation (H1b).

Usage:
  .venv/bin/python scripts/oracles/dynamics_oracle.py \
      --shift cart_mass_x05=data/cartpole-ood-cartmass05-15k-v2 \
      --shift cart_mass_x2=data/cartpole-ood-cartmass20-15k-v2

Collect a shifted dataset with e.g.
  .venv/bin/collect-cartpole --output-dir data/cartpole-ood-cartmass05-15k-v2 \
      --total-frames 15000 --seed 11 --action-repeat 2 --physics-scale body_mass_1=0.5

Writes runs/oracles/dynamics_oracle/<timestamp>.json (CPU, tens of minutes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from oracles.common import load_state_episodes
from oracles.dynamics import ADAPT_MODES, run_dynamics_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nominal", type=Path, default=Path("data/cartpole-swingup-random-100k-v2"))
    parser.add_argument("--shift", action="append", required=True, help="NAME=DATASET_DIR")
    parser.add_argument("--nominal-train-episodes", type=int, default=60)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--pretrain-steps", type=int, default=3000)
    parser.add_argument("--adapt-steps", type=int, default=300)
    parser.add_argument("--train-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=list(ADAPT_MODES))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    nominal_train = load_state_episodes(
        args.nominal, split="train", max_episodes=args.nominal_train_episodes, seed=args.seed,
    )
    nominal_val = load_state_episodes(args.nominal, split="val", seed=args.seed)
    shifted = {}
    for item in args.shift:
        name, _, directory = item.partition("=")
        shifted[name] = (
            load_state_episodes(Path(directory), split="train", val_fraction=0.2, seed=args.seed),
            load_state_episodes(Path(directory), split="val", val_fraction=0.2, seed=args.seed),
        )
    result = run_dynamics_oracle(
        nominal_train,
        nominal_val,
        shifted,
        ks=args.ks,
        pretrain_steps=args.pretrain_steps,
        adapt_steps=args.adapt_steps,
        train_horizon=args.train_horizon,
        batch_size=args.batch_size,
        seed=args.seed,
        modes=args.modes,
        log=lambda msg: print(msg, flush=True),
    )
    path = result.write()
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
