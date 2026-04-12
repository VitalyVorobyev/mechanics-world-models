"""Preview foreground/background reconstruction masks for saved sequences."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader, Subset

from data import EpisodeSequenceDataset, ImageCropConfig
from data.image_transforms import add_crop_args, crop_config_from_args
from eval.eval_rssm import select_sequence_indices
from models.losses import dynamic_reconstruction_mask, foreground_reconstruction_mask, foreground_saliency
from viz.debug_grids import HEADER_HEIGHT, chw_float_to_uint8, validate_timesteps
from viz.rssm_eval_viz import evenly_spaced_timesteps


SplitName = Literal["train", "val"]


@dataclass(frozen=True)
class ForegroundMaskPreviewConfig:
    """Configuration for foreground-mask contact-sheet diagnostics."""

    dataset_dir: Path
    output_path: Path
    sequence_length: int = 32
    split: SplitName = "val"
    val_fraction: float = 0.1
    seed: int = 0
    batch_size: int = 32
    timesteps: tuple[int, ...] = ()
    sample_index: int = 0
    foreground_mask_floor: float = 0.02
    foreground_mask_kernel_size: int = 7
    include_dynamic_mask: bool = True
    dynamic_mask_floor: float = 0.05
    dynamic_mask_kernel_size: int = 7
    crop_config: ImageCropConfig = field(default_factory=ImageCropConfig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the foreground-mask preview CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, nargs="*", default=[])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--foreground-mask-floor", type=float, default=0.02)
    parser.add_argument("--foreground-mask-kernel-size", type=int, default=7)
    parser.add_argument("--no-dynamic-mask", action="store_true")
    parser.add_argument("--dynamic-mask-floor", type=float, default=0.05)
    parser.add_argument("--dynamic-mask-kernel-size", type=int, default=7)
    add_crop_args(parser, default_mode="none")
    return parser


def config_from_args(args: argparse.Namespace) -> ForegroundMaskPreviewConfig:
    """Convert CLI arguments to a typed config dataclass."""

    return ForegroundMaskPreviewConfig(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        sequence_length=args.sequence_length,
        split=args.split,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        timesteps=tuple(args.timesteps),
        sample_index=args.sample_index,
        foreground_mask_floor=args.foreground_mask_floor,
        foreground_mask_kernel_size=args.foreground_mask_kernel_size,
        include_dynamic_mask=not args.no_dynamic_mask,
        dynamic_mask_floor=args.dynamic_mask_floor,
        dynamic_mask_kernel_size=args.dynamic_mask_kernel_size,
        crop_config=crop_config_from_args(args) or ImageCropConfig(),
    )


def run_foreground_mask_preview(config: ForegroundMaskPreviewConfig) -> dict[str, Any]:
    """Write a foreground-mask diagnostic contact sheet and metadata JSON."""

    validate_config(config)
    dataset = EpisodeSequenceDataset(
        dataset_dir=config.dataset_dir,
        sequence_length=config.sequence_length,
        split=config.split,
        val_fraction=config.val_fraction,
        seed=config.seed,
        crop_config=config.crop_config,
    )
    selected_indices = select_sequence_indices(
        total_sequences=len(dataset),
        num_sequences=min(config.batch_size, len(dataset)),
        seed=config.seed,
    )
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=len(selected_indices),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))
    observations = batch["observations"]
    next_observations = batch["next_observations"]
    timesteps = tuple(config.timesteps) if config.timesteps else tuple(evenly_spaced_timesteps(config.sequence_length, 8))
    grid = make_foreground_mask_grid(
        observations=observations,
        next_observations=next_observations,
        timesteps=timesteps,
        sample_index=config.sample_index,
        foreground_mask_floor=config.foreground_mask_floor,
        foreground_mask_kernel_size=config.foreground_mask_kernel_size,
        include_dynamic_mask=config.include_dynamic_mask,
        dynamic_mask_floor=config.dynamic_mask_floor,
        dynamic_mask_kernel_size=config.dynamic_mask_kernel_size,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(config.output_path, grid)

    metadata = {
        "config": serializable_config(config),
        "selected_indices": selected_indices,
        "rendered_dataset_index": selected_indices[config.sample_index],
        "observation_shape": list(observations.shape),
        "timesteps": list(validate_timesteps(timesteps, observations.shape[1])),
        "output_path": str(config.output_path),
    }
    metadata_path = config.output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def make_foreground_mask_grid(
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    timesteps: Sequence[int],
    sample_index: int,
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
    include_dynamic_mask: bool,
    dynamic_mask_floor: float,
    dynamic_mask_kernel_size: int,
) -> NDArray[np.uint8]:
    """Create a contact sheet for foreground and optional dynamic masks."""

    if observations.shape != next_observations.shape:
        raise ValueError(f"observation shape mismatch: {observations.shape} != {next_observations.shape}")
    if observations.ndim != 5:
        raise ValueError(f"observations must have shape [B, T, C, H, W], got {observations.shape}")
    if sample_index < 0 or sample_index >= observations.shape[0]:
        raise ValueError(f"sample_index must be in [0, {observations.shape[0] - 1}]")

    clean_timesteps = validate_timesteps(timesteps, observations.shape[1])
    saliency = foreground_saliency(observations)
    foreground_mask = foreground_reconstruction_mask(
        observations=observations,
        floor=foreground_mask_floor,
        kernel_size=foreground_mask_kernel_size,
    )
    dynamic_mask = (
        dynamic_reconstruction_mask(
            observations=observations,
            next_observations=next_observations,
            floor=dynamic_mask_floor,
            kernel_size=dynamic_mask_kernel_size,
        )
        if include_dynamic_mask
        else None
    )

    rows: list[NDArray[np.uint8]] = []
    for timestep in clean_timesteps:
        tiles = [
            add_tile_header(chw_float_to_uint8(observations[sample_index, timestep].numpy()), f"truth t={timestep:02d}"),
            add_tile_header(
                scalar_map_to_uint8(saliency[sample_index, timestep, 0].numpy()),
                "bg saliency",
            ),
            add_tile_header(
                scalar_map_to_uint8(foreground_mask[sample_index, timestep, 0].numpy()),
                "fg mask",
            ),
        ]
        if dynamic_mask is not None:
            tiles.append(
                add_tile_header(
                    scalar_map_to_uint8(dynamic_mask[sample_index, timestep, 0].numpy()),
                    "dyn mask",
                ),
            )
        rows.append(concat_tiles(tiles))

    separator = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    parts: list[NDArray[np.uint8]] = []
    for row in rows:
        if parts:
            parts.append(separator)
        parts.append(row)
    return np.concatenate(parts, axis=0)


def scalar_map_to_uint8(values: np.ndarray) -> NDArray[np.uint8]:
    """Render one scalar map as a grayscale RGB tile."""

    if values.ndim != 2:
        raise ValueError(f"expected scalar map shape [H, W], got {values.shape}")
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        scaled = np.zeros_like(values, dtype=np.float32)
    else:
        scaled = (values.astype(np.float32) - low) / (high - low)
    image = np.round(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(image[:, :, None], repeats=3, axis=2)


def concat_tiles(tiles: Sequence[NDArray[np.uint8]]) -> NDArray[np.uint8]:
    """Concatenate labeled RGB tiles with thin white separators."""

    if not tiles:
        raise ValueError("tiles must not be empty")
    height = tiles[0].shape[0]
    separator = np.full((height, 2, 3), 255, dtype=np.uint8)
    parts: list[NDArray[np.uint8]] = []
    for tile in tiles:
        if tile.shape[0] != height:
            raise ValueError("all tiles must have the same height")
        if parts:
            parts.append(separator)
        parts.append(tile)
    return np.concatenate(parts, axis=1)


def add_tile_header(tile: NDArray[np.uint8], label: str) -> NDArray[np.uint8]:
    """Add a compact label above one RGB tile."""

    canvas = np.zeros((tile.shape[0] + HEADER_HEIGHT, tile.shape[1], 3), dtype=np.uint8)
    canvas[HEADER_HEIGHT:] = tile
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((3, 2), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return np.asarray(image, dtype=np.uint8)


def validate_config(config: ForegroundMaskPreviewConfig) -> None:
    """Validate foreground-mask preview settings."""

    if config.sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.sample_index < 0 or config.sample_index >= config.batch_size:
        raise ValueError("sample_index must be within the requested batch")
    if not 0.0 <= config.foreground_mask_floor <= 1.0:
        raise ValueError("foreground_mask_floor must satisfy 0 <= floor <= 1")
    if config.foreground_mask_kernel_size < 1 or config.foreground_mask_kernel_size % 2 == 0:
        raise ValueError("foreground_mask_kernel_size must be a positive odd integer")
    if not 0.0 <= config.dynamic_mask_floor <= 1.0:
        raise ValueError("dynamic_mask_floor must satisfy 0 <= floor <= 1")
    if config.dynamic_mask_kernel_size < 1 or config.dynamic_mask_kernel_size % 2 == 0:
        raise ValueError("dynamic_mask_kernel_size must be a positive odd integer")


def serializable_config(config: ForegroundMaskPreviewConfig) -> dict[str, Any]:
    """Convert config dataclass values to JSON-friendly objects."""

    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def main() -> None:
    """Run the foreground-mask preview CLI."""

    config = config_from_args(build_arg_parser().parse_args())
    metadata = run_foreground_mask_preview(config)
    print(f"Wrote {metadata['output_path']}")


if __name__ == "__main__":
    main()
