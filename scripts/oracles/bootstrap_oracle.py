"""Label-free front end CLI: CV heuristic poses, then render-and-compare EM (A-FE-1).

Usage:
  .venv/bin/python scripts/oracles/bootstrap_oracle.py

Writes runs/oracles/bootstrap_oracle/<timestamp>.json. Pose labels are used only to score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oracles.bootstrap import run_bootstrap_oracle
from oracles.common import load_frames_with_state
from research.provenance import add_run_args, begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/cartpole-swingup-random-100k-v2"))
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--em-iterations", type=int, default=2)
    parser.add_argument("--em-frames", type=int, default=3000)
    parser.add_argument("--renderer-steps", type=int, default=3000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    add_run_args(parser)
    args = parser.parse_args()
    run = begin_run(args)
    train_images, _, _ = load_frames_with_state(args.dataset, split="train", max_episodes=args.train_episodes, seed=args.seed)
    val_images, val_qpos, _ = load_frames_with_state(
        args.dataset, split="val", max_episodes=args.val_episodes, seed=args.seed, frame_stride=5,
    )
    result = run_bootstrap_oracle(
        train_images, val_images, val_qpos, em_iterations=args.em_iterations, em_frames=args.em_frames,
        renderer_steps=args.renderer_steps, device=args.device, seed=args.seed, log=lambda m: print(m, flush=True),
    )
    path = result.write(run=run)
    print({k: round(v, 3) for k, v in result.metrics.items() if "visible" in k or "flipped" in k or "found" in k})
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
