"""E-008: train the keypoint pose reader on frames rendered at uniformly sampled states.

Renders ``--train-frames`` + ``--val-frames`` frames from camera 0 with the
wide field of view (``control.camera.FULL_VIEW``), supervises the cart and
pole-tip keypoints with their projections, and reports wrapped angle errors
(overall and upright, cos θ > 0.9) on the held-out frames. The model is saved
under ``runs/oracles/keypoint_reader/models/`` for the closed-loop runs.

Usage:
  .venv/bin/python scripts/oracles/keypoint_reader.py --experiment E-008 --device mps

Rendering takes about 2 ms per frame; 20k steps about 10 minutes on MPS.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from control.camera import FULL_VIEW, CameraSpec, render_states
from control.keypoints import pose_error_metrics, read_poses, sample_states, train_reader
from oracles.common import Criterion, OracleResult
from research.provenance import add_run_args, begin_run

NAME = "keypoint_reader"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-frames", type=int, default=50_000)
    parser.add_argument("--val-frames", type=int, default=5_000)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=FULL_VIEW.image_size)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", type=Path, default=Path(f"runs/oracles/{NAME}/models"))
    add_run_args(parser)
    args = parser.parse_args()
    run = begin_run(args)

    spec = CameraSpec(FULL_VIEW.fovy_deg, args.image_size)
    log = lambda m: print(m, flush=True)  # noqa: E731
    start = time.perf_counter()
    train_q = sample_states(args.train_frames, seed=2 * args.seed)
    val_q = sample_states(args.val_frames, seed=2 * args.seed + 1)
    train_images, val_images = render_states(train_q, spec), render_states(val_q, spec)
    render_s = time.perf_counter() - start
    log(f"rendered {len(train_q) + len(val_q)} frames in {render_s:.0f}s")

    start = time.perf_counter()
    model, loss = train_reader(
        train_images, spec.keypoints(train_q), steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed, log=log,
    )
    train_s = time.perf_counter() - start
    errors = pose_error_metrics(read_poses(model, val_images, spec, args.device), val_q, spec)

    config = {
        "camera": {"fovy_deg": spec.fovy_deg, "image_size": spec.image_size, "pixels_per_metre": spec.pixels_per_metre},
        "train_frames": args.train_frames,
        "val_frames": args.val_frames,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "seed": args.seed,
    }
    result = OracleResult(
        name=NAME,
        hypotheses=["H-KP"],
        question="How accurately does a keypoint reader trained on renders of uniform states read the pose?",
        config=config,
        metrics={**{f"pose_error.{k}": v for k, v in errors.items()},
                 "train_loss": loss, "render_time_s": render_s, "train_time_s": train_s},
        criteria=[
            Criterion("pose_error.theta_median_deg_upright", "<=", 1.0, "E-008 prediction, upright frames"),
            Criterion("pose_error.theta_median_deg", "<=", 1.5, "E-008 prediction, all frames"),
        ],
    )
    path = result.write(run=run)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / f"{path.stem}.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "camera": config["camera"]}, model_path)
    print(result.summary())
    print(f"wrote {path} and {model_path}")


if __name__ == "__main__":
    main()
