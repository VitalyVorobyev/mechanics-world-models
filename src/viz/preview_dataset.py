"""Create contact sheet and MP4 previews from saved NPZ trajectories."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

from data.trajectory import load_trajectory_npz


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line parser for preview generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--video-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--cols", type=int, default=8)
    return parser


def find_episode_files(dataset_dir: Path) -> list[Path]:
    """Return saved episode archives in deterministic order."""

    paths = sorted(dataset_dir.glob("episode_*.npz"))
    if not paths:
        paths = sorted(dataset_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz trajectory files found in {dataset_dir}")
    return paths


def load_all_frames(dataset_dir: Path) -> NDArray[np.uint8]:
    """Load and concatenate image frames from all trajectories."""

    images = [load_trajectory_npz(path).images for path in find_episode_files(dataset_dir)]
    return np.concatenate(images, axis=0)


def sample_frames(frames: NDArray[np.uint8], count: int) -> NDArray[np.uint8]:
    """Evenly sample up to ``count`` frames."""

    if count < 1:
        raise ValueError("frame sample count must be >= 1")
    if frames.shape[0] == 0:
        raise ValueError("cannot sample from an empty frame array")
    sample_count = min(count, frames.shape[0])
    indices = np.linspace(0, frames.shape[0] - 1, sample_count, dtype=np.int64)
    return frames[indices]


def make_contact_sheet(frames: NDArray[np.uint8], cols: int) -> NDArray[np.uint8]:
    """Tile RGB frames into a contact sheet image."""

    if cols < 1:
        raise ValueError("cols must be >= 1")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (N, H, W, 3)")

    rows = ceil(frames.shape[0] / cols)
    height, width = frames.shape[1:3]
    sheet = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        sheet[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ] = frame

    return sheet


def write_preview_artifacts(
    dataset_dir: Path,
    output_dir: Path,
    num_images: int = 64,
    video_frames: int = 120,
    fps: int = 20,
    cols: int = 8,
) -> tuple[Path, Path]:
    """Write ``contact_sheet.png`` and ``preview.mp4`` for a dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_all_frames(dataset_dir)

    sheet_path = output_dir / "contact_sheet.png"
    sheet = make_contact_sheet(sample_frames(frames, num_images), cols=cols)
    imageio.imwrite(sheet_path, sheet)

    video_path = output_dir / "preview.mp4"
    with imageio.get_writer(video_path, fps=fps, macro_block_size=1) as writer:
        for frame in sample_frames(frames, video_frames):
            writer.append_data(frame)

    return sheet_path, video_path


def main() -> None:
    """Run the dataset preview CLI."""

    args = build_arg_parser().parse_args()
    sheet_path, video_path = write_preview_artifacts(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        video_frames=args.video_frames,
        fps=args.fps,
        cols=args.cols,
    )
    print(f"Wrote {sheet_path}")
    print(f"Wrote {video_path}")


if __name__ == "__main__":
    main()
