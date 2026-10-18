"""Rung 3 diagnostic CLI: learned vs true mass matrix (A-ACT-1).

Usage (paths from runs/oracles/dynamics_oracle/cache/pretrained/<hash>/):
  .venv/bin/python scripts/oracles/mass_gauge.py \
      --model free=runs/oracles/dynamics_oracle/cache/pretrained/02134dd003/lagrangian.pt \
      --model cart_only=runs/oracles/dynamics_oracle/cache/pretrained/958116b8eb/lagrangian.pt:0

The optional ":DIMS" suffix gives the model's actuated dims (comma separated).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oracles.gauge import run_mass_gauge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH[:DIMS]")
    args = parser.parse_args()
    checkpoints = {}
    for item in args.model:
        name, _, rest = item.partition("=")
        path, _, dims = rest.partition(":")
        checkpoints[name] = (Path(path), tuple(int(d) for d in dims.split(",")) if dims else None)
    result = run_mass_gauge(checkpoints)
    path = result.write()
    for key, value in result.metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
