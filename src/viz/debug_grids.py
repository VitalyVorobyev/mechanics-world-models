"""Image-grid helpers for RSSM reconstruction debugging."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


HEADER_HEIGHT = 16


def chw_float_to_uint8(image: np.ndarray) -> NDArray[np.uint8]:
    """Convert one ``[3, H, W]`` float image in ``[0, 1]`` to ``[H, W, 3]``."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image shape [3, H, W], got {image.shape}")
    clipped = np.clip(image, 0.0, 1.0)
    return np.round(np.transpose(clipped, (1, 2, 0)) * 255.0).astype(np.uint8)


def absolute_error_heatmap(
    target: np.ndarray,
    reconstruction: np.ndarray,
    scale: float | None = None,
) -> NDArray[np.uint8]:
    """Render mean absolute RGB error as a simple heatmap.

    Args:
        target: Target image shaped ``[3, H, W]`` in ``[0, 1]``.
        reconstruction: Reconstructed image shaped ``[3, H, W]`` in ``[0, 1]``.
        scale: Optional display scale. If omitted, the image max error is used.

    Returns:
        Heatmap image shaped ``[H, W, 3]``.
    """

    if target.shape != reconstruction.shape:
        raise ValueError(
            f"target/reconstruction shape mismatch: {target.shape} != {reconstruction.shape}",
        )
    error = np.abs(target - reconstruction).mean(axis=0)
    display_scale = float(scale) if scale is not None else float(error.max())
    if display_scale <= 0.0:
        display_scale = 1.0
    value = np.clip(error / display_scale, 0.0, 1.0)

    heatmap = np.zeros((*value.shape, 3), dtype=np.float32)
    heatmap[..., 0] = value
    heatmap[..., 1] = np.clip((value - 0.25) / 0.75, 0.0, 1.0)
    heatmap[..., 2] = 0.05 * (1.0 - value)
    return np.round(heatmap * 255.0).astype(np.uint8)


def make_reconstruction_debug_grid(
    targets: np.ndarray,
    reconstructions: np.ndarray,
    timesteps: Sequence[int],
    max_sequences: int = 4,
) -> NDArray[np.uint8]:
    """Create a grid with target, reconstruction, and absolute error columns.

    Args:
        targets: Target observations shaped ``[B, T, 3, H, W]`` in ``[0, 1]``.
        reconstructions: Decoder outputs shaped ``[B, T, 3, H, W]`` in ``[0, 1]``.
        timesteps: Timesteps to include in the grid.
        max_sequences: Maximum batch items to render.

    Returns:
        RGB grid image as ``uint8``.
    """

    if targets.shape != reconstructions.shape:
        raise ValueError(
            f"target/reconstruction shape mismatch: {targets.shape} != {reconstructions.shape}",
        )
    if targets.ndim != 5 or targets.shape[2] != 3:
        raise ValueError(f"expected tensors shaped [B, T, 3, H, W], got {targets.shape}")
    if max_sequences < 1:
        raise ValueError("max_sequences must be >= 1")
    if not timesteps:
        raise ValueError("timesteps must contain at least one index")

    batch_size, sequence_length, _, height, width = targets.shape
    clean_timesteps = validate_timesteps(timesteps, sequence_length)
    rows: list[NDArray[np.uint8]] = []
    for batch_index in range(min(batch_size, max_sequences)):
        for timestep in clean_timesteps:
            target = targets[batch_index, timestep]
            reconstruction = reconstructions[batch_index, timestep]
            error = absolute_error_heatmap(target, reconstruction)
            row_tiles = [
                add_tile_header(chw_float_to_uint8(target), f"b{batch_index} t={timestep:02d} target"),
                add_tile_header(
                    chw_float_to_uint8(reconstruction),
                    f"b{batch_index} t={timestep:02d} recon",
                ),
                add_tile_header(error, f"b{batch_index} t={timestep:02d} abs error"),
            ]
            separator = np.full((height + HEADER_HEIGHT, 2, 3), 255, dtype=np.uint8)
            rows.append(
                np.concatenate([row_tiles[0], separator, row_tiles[1], separator, row_tiles[2]], axis=1),
            )

    row_separator = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    grid_parts: list[NDArray[np.uint8]] = []
    for row in rows:
        if grid_parts:
            grid_parts.append(row_separator)
        grid_parts.append(row)
    return np.concatenate(grid_parts, axis=0)


def write_reconstruction_debug_grid(
    output_path: Path,
    targets: np.ndarray,
    reconstructions: np.ndarray,
    timesteps: Sequence[int],
    max_sequences: int = 4,
) -> Path:
    """Write a posterior-reconstruction debug grid to PNG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_reconstruction_debug_grid(
        targets=targets,
        reconstructions=reconstructions,
        timesteps=timesteps,
        max_sequences=max_sequences,
    )
    imageio.imwrite(output_path, grid)
    return output_path


def add_tile_header(tile: NDArray[np.uint8], label: str) -> NDArray[np.uint8]:
    """Add a compact label above one RGB tile."""

    canvas = np.zeros((tile.shape[0] + HEADER_HEIGHT, tile.shape[1], 3), dtype=np.uint8)
    canvas[HEADER_HEIGHT:] = tile
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((3, 2), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return np.asarray(image, dtype=np.uint8)


def validate_timesteps(timesteps: Sequence[int], sequence_length: int) -> list[int]:
    """Validate and de-duplicate timesteps while preserving order."""

    clean: list[int] = []
    for timestep in timesteps:
        value = int(timestep)
        if value < 0 or value >= sequence_length:
            raise ValueError(f"timestep {value} is outside sequence length {sequence_length}")
        if value not in clean:
            clean.append(value)
    return clean
