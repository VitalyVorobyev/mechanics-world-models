"""Rung 2b CLI: self-supervised encoder learning through a frozen, correct renderer (A-FE-1).

Usage:
  .venv/bin/python scripts/oracles/frozen_renderer_oracle.py --kind keypoint
  .venv/bin/python scripts/oracles/frozen_renderer_oracle.py --kind conv

Writes runs/oracles/frozen_renderer_oracle_<kind>/<timestamp>.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oracles.common import load_frames_with_state
from oracles.frozen_render import run_frozen_renderer_oracle
from research.provenance import add_run_args, begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/cartpole-swingup-random-100k-v2"))
    parser.add_argument("--kind", choices=("conv", "keypoint"), default="keypoint")
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--decoder-steps", type=int, default=3000)
    parser.add_argument("--encoder-steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pool-sizes", type=int, nargs="*", default=[], help="coarse-to-fine terms, e.g. 8 24")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    add_run_args(parser)
    args = parser.parse_args()
    run = begin_run(args)
    train_images, train_qpos, _ = load_frames_with_state(
        args.dataset, split="train", max_episodes=args.train_episodes, seed=args.seed,
    )
    val_images, val_qpos, _ = load_frames_with_state(
        args.dataset, split="val", max_episodes=args.val_episodes, seed=args.seed, frame_stride=5,
    )
    result = run_frozen_renderer_oracle(
        train_images, train_qpos, val_images, val_qpos, kind=args.kind, decoder_steps=args.decoder_steps,
        encoder_steps=args.encoder_steps, lr=args.lr, pool_sizes=tuple(args.pool_sizes), device=args.device, seed=args.seed,
        log=lambda m: print(m, flush=True),
    )
    path = result.write(run=run)
    print({k: round(v, 4) for k, v in result.metrics.items()})
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
