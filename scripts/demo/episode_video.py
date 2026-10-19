"""Render a recorded closed-loop episode (``.npz`` next to a control result) to mp4.

Usage:
  .venv/bin/python scripts/demo/episode_video.py runs/oracles/camera_control/<stamp>.npz --title "camera + learned Lagrangian"
  .venv/bin/python scripts/demo/episode_video.py runs/oracles/control_episode/<stamp>.npz --title "true model"

Episodes from the camera pipeline carry ``poses`` (what the reader read); they
are drawn on the controller's view. 500 decisions at 0.02 s play in real time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from control.camera import FULL_VIEW
from viz.control_video import episode_frames, write_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, help="default: next to the episode, .mp4")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    with np.load(args.episode) as z:
        qpos = z["qpos"]
        poses = z["poses"] if "poses" in z else None
        rewards = z["rewards"] if "rewards" in z else None
    frames = episode_frames(qpos, FULL_VIEW, poses=poses, rewards=rewards, title=args.title)
    path = write_video(frames, args.out or args.episode.with_suffix(".mp4"))
    print(f"wrote {path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
