"""Rung 2 CLI: regress the true pose from pixels; can the encoder localize it?

Usage:
  .venv/bin/python scripts/oracles/encoder_oracle.py --kind keypoint
  .venv/bin/python scripts/oracles/encoder_oracle.py --kind conv

Writes runs/oracles/encoder_oracle_<kind>/<timestamp>.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oracles.common import load_frames_with_state
from oracles.encoder import run_encoder_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/cartpole-swingup-random-100k-v2"))
    parser.add_argument("--kind", choices=("conv", "keypoint"), required=True)
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_images, train_qpos, _ = load_frames_with_state(
        args.dataset, split="train", max_episodes=args.train_episodes, seed=args.seed,
    )
    val_images, val_qpos, _ = load_frames_with_state(
        args.dataset, split="val", max_episodes=args.val_episodes, seed=args.seed, frame_stride=5,
    )
    result = run_encoder_oracle(
        train_images, train_qpos, val_images, val_qpos,
        kind=args.kind, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed, log=lambda m: print(m, flush=True),
    )
    path = result.write()
    print({k: round(v, 5) for k, v in result.metrics.items()})
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
