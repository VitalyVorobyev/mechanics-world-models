"""Rung 3 CLI: Lagrangian vs MLP dynamics on true state, few-shot adaptation (H1b).

Usage:
  .venv/bin/python scripts/oracles/dynamics_oracle.py \
      --shift cart_mass_x05=data/cartpole-ood-cartmass05-15k-v2 \
      --shift cart_mass_x2=data/cartpole-ood-cartmass20-15k-v2

Collect a shifted dataset with e.g.
  .venv/bin/collect-cartpole --output-dir data/cartpole-ood-cartmass05-15k-v2 \
      --total-frames 15000 --seed 11 --action-repeat 2 --physics-scale body_mass_1=0.5

Shifted datasets are split by episode: half for adaptation segments, a
quarter for selecting each mode's adaptation recipe, a quarter for test.

Writes runs/oracles/dynamics_oracle/<timestamp>.json. Resumable: pretrained
models and finished cells are cached under --cache-dir (keyed by a config
hash), so re-running the same command after an interruption continues.
Progress lines go to stdout ("[cells done/total]").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from oracles.common import load_shift_splits, load_state_episodes
from oracles.dynamics import ADAPT_MODES, run_dynamics_oracle
from research.provenance import add_run_args, begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nominal", type=Path, default=Path("data/cartpole-swingup-random-100k-v2"))
    parser.add_argument("--shift", action="append", required=True, help="NAME=DATASET_DIR")
    parser.add_argument("--nominal-train-episodes", type=int, default=60)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 20])
    parser.add_argument("--pretrain-steps", type=int, default=10000)
    parser.add_argument("--adapt-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--adapt-checkpoints", type=int, nargs="+", default=[0, 50, 100, 200, 400, 800])
    parser.add_argument("--train-horizon", type=int, default=8)
    parser.add_argument("--n-substeps", type=int, default=2)
    parser.add_argument("--rail-margin", type=float, default=0.2, help="negative = keep rail windows")
    parser.add_argument(
        "--actuated-dims", type=int, nargs="+", default=None,
        help="Lagrangian actuation acts only on these coordinates (cartpole: 0 = cart)",
    )
    parser.add_argument("--slot-scale", choices=["none", "diag"], default="diag")
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/oracles/dynamics_oracle/cache"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=list(ADAPT_MODES))
    parser.add_argument("--threads", type=int, default=4)
    add_run_args(parser)
    args = parser.parse_args()
    run = begin_run(args)
    torch.set_num_threads(args.threads)

    nominal_train = load_state_episodes(
        args.nominal, split="train", max_episodes=args.nominal_train_episodes, seed=args.seed,
    )
    nominal_val = load_state_episodes(args.nominal, split="val", seed=args.seed)
    shifted = {}
    for item in args.shift:
        name, _, directory = item.partition("=")
        shifted[name] = load_shift_splits(Path(directory), seed=args.seed)
    result = run_dynamics_oracle(
        nominal_train,
        nominal_val,
        shifted,
        ks=args.ks,
        pretrain_steps=args.pretrain_steps,
        adapt_lrs=args.adapt_lrs,
        adapt_checkpoints=args.adapt_checkpoints,
        train_horizon=args.train_horizon,
        n_substeps=args.n_substeps,
        rail_margin=None if args.rail_margin < 0 else args.rail_margin,
        actuated_dims=args.actuated_dims,
        slot_scale=args.slot_scale,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        modes=args.modes,
        log=lambda msg: print(msg, flush=True),
    )
    path = result.write(run=run)
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
