"""Rung 3 matched baseline CLI: context-conditioned MLP vs Lagrangian slots (H1b).

Both forms train on simulated cart masses (default 1.0 nominal, 0.7, 0.85,
1.2, 1.4; state only, generated in seconds) with a per-mass 3-number
context, then adapt only a new context to cart mass x0.5 and x2, which lie
outside the training range.

Usage:
  .venv/bin/python scripts/oracles/context_oracle.py --seed 0

Writes runs/oracles/context_oracle/<timestamp>.json. Resumable through
--cache-dir like dynamics_oracle.py; progress lines "[cells done/total]".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from oracles.common import StateEpisodes, load_shift_splits, load_state_episodes
from oracles.context import CONTEXT_MODES, run_context_oracle, simulate_state_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nominal", type=Path, default=Path("data/cartpole-swingup-random-100k-v2"))
    parser.add_argument(
        "--shift", action="append", default=None, help="NAME=DATASET_DIR (default: cart mass x0.5 and x2)",
    )
    parser.add_argument(
        "--sim-shift", action="append", default=None,
        help="NAME=CART_MASS_SCALE: simulate a test shift (30 episodes split 15/7/8) instead of datasets",
    )
    parser.add_argument("--train-masses", type=float, nargs="+", default=[1.0, 0.7, 0.85, 1.2, 1.4])
    parser.add_argument("--episodes-per-mass", type=int, default=30)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 20])
    parser.add_argument("--pretrain-steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=list(CONTEXT_MODES))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/oracles/context_oracle/cache"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.train_masses[0] != 1.0:
        parser.error("the first training mass must be the nominal 1.0 (its context is pinned to zero)")

    shift_items = args.shift or [
        "cart_mass_x05=data/cartpole-ood-cartmass05-15k-v2",
        "cart_mass_x2=data/cartpole-ood-cartmass20-15k-v2",
    ]
    train_envs = {
        f"cart_mass_x{m:g}": simulate_state_episodes(
            None if m == 1.0 else {"body_mass_1": m}, episodes=args.episodes_per_mass, seed=1000 * args.seed + i,
        )
        for i, m in enumerate(args.train_masses)
    }
    nominal_val = load_state_episodes(args.nominal, split="val", seed=args.seed)
    shifted = {}
    if args.sim_shift:
        for index, item in enumerate(args.sim_shift):
            name, _, scale = item.partition("=")
            sim = simulate_state_episodes({"body_mass_1": float(scale)}, episodes=30, seed=1000 * args.seed + 500 + index)
            shifted[name] = tuple(
                StateEpisodes(qpos=sim.qpos[part], qvel=sim.qvel[part], actions=sim.actions[part], frame_dt=sim.frame_dt)
                for part in (slice(0, 15), slice(15, 22), slice(22, 30))
            )
    else:
        for item in shift_items:
            name, _, directory = item.partition("=")
            shifted[name] = load_shift_splits(Path(directory), seed=args.seed)
    result = run_context_oracle(
        train_envs, nominal_val, shifted, ks=args.ks, pretrain_steps=args.pretrain_steps, seed=args.seed,
        modes=args.modes, cache_dir=args.cache_dir, log=lambda msg: print(msg, flush=True),
    )
    path = result.write()
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
