"""Preview dataset cropping by comparing original and cropped/resized frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

from data import index_episode_files, load_trajectory_npz
from data.image_transforms import (
    ImageCropConfig,
    add_crop_args,
    apply_image_crop_resize,
    crop_config_from_args,
)
from viz.debug_grids import HEADER_HEIGHT, add_tile_header


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the crop preview CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    add_crop_args(parser, default_mode="none")
    return parser


def write_crop_preview(
    dataset_dir: Path,
    output_path: Path,
    crop_config: ImageCropConfig,
    episode_index: int = 0,
    start_frame: int = 0,
    num_frames: int = 8,
) -> Path:
    """Write a PNG comparing original frames with cropped/resized frames."""

    if episode_index < 0:
        raise ValueError("episode_index must be >= 0")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    episode_files = index_episode_files(dataset_dir)
    if episode_index >= len(episode_files):
        raise ValueError(f"episode_index={episode_index} exceeds {len(episode_files)} episode files")
    frames = load_trajectory_npz(episode_files[episode_index]).images
    end_frame = min(start_frame + num_frames, frames.shape[0])
    if start_frame >= end_frame:
        raise ValueError(f"start_frame={start_frame} exceeds episode frame count {frames.shape[0]}")

    original = frames[start_frame:end_frame]
    cropped = apply_image_crop_resize(original, crop_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, make_crop_preview_grid(original, cropped, start_frame=start_frame))
    return output_path


def make_crop_preview_grid(
    original: NDArray[np.uint8],
    cropped: NDArray[np.uint8],
    start_frame: int = 0,
) -> NDArray[np.uint8]:
    """Make a two-column original/cropped preview grid."""

    if original.shape != cropped.shape:
        raise ValueError(f"original/cropped shape mismatch: {original.shape} != {cropped.shape}")
    if original.ndim != 4 or original.shape[-1] != 3:
        raise ValueError(f"frames must have shape [T, H, W, 3], got {original.shape}")

    height = original.shape[1]
    rows: list[NDArray[np.uint8]] = []
    separator = np.full((height + HEADER_HEIGHT, 2, 3), 255, dtype=np.uint8)
    for index in range(original.shape[0]):
        frame_id = start_frame + index
        original_tile = add_tile_header(original[index], f"frame {frame_id:04d} original")
        cropped_tile = add_tile_header(cropped[index], f"frame {frame_id:04d} crop+resize")
        rows.append(np.concatenate([original_tile, separator, cropped_tile], axis=1))

    row_separator = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    parts: list[NDArray[np.uint8]] = []
    for row in rows:
        if parts:
            parts.append(row_separator)
        parts.append(row)
    return np.concatenate(parts, axis=0)


def main() -> None:
    """Run the crop preview CLI."""

    args = build_arg_parser().parse_args()
    crop_config = crop_config_from_args(args) or ImageCropConfig()
    path = write_crop_preview(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        crop_config=crop_config,
        episode_index=args.episode_index,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
