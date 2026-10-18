"""Rung 3 ceiling CLI: true cartpole dynamics under the models' integrator.

Usage:
  .venv/bin/python scripts/oracles/dynamics_ceiling.py \
      --dataset nominal=data/cartpole-swingup-random-100k-v2 \
      --dataset cart_mass_x05=data/cartpole-ood-cartmass05-15k-v2:body_mass_1=0.5

Writes runs/oracles/dynamics_ceiling/<timestamp>.json (CPU, about a minute).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oracles.ceiling import run_dynamics_ceiling
from oracles.common import load_state_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, help="NAME=DIR[:KEY=SCALE,...]")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--substeps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rail-margin", type=float, default=0.2)
    args = parser.parse_args()

    episodes = {}
    for item in args.dataset:
        name, _, rest = item.partition("=")
        directory, _, scale_text = rest.partition(":")
        scales = None
        if scale_text:
            scales = {k: float(v) for k, v in (pair.split("=") for pair in scale_text.split(","))}
        data = load_state_episodes(Path(directory), split="val", val_fraction=0.2, max_episodes=args.max_episodes)
        episodes[name] = (data, scales)
    result = run_dynamics_ceiling(
        episodes, substeps=args.substeps, rail_margin=args.rail_margin, log=lambda m: print(m, flush=True),
    )
    path = result.write()
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
