"""Visualization helpers for RSSM reconstruction and prediction diagnostics."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


HEADER_HEIGHT = 16
PANEL_LABELS = ("target t+1", "recon t", "pred t+1")


def tensor_sequence_to_uint8(sequence: np.ndarray) -> NDArray[np.uint8]:
    """Convert one image sequence from ``[T, C, H, W]`` float to ``[T, H, W, C]``."""

    if sequence.ndim != 4:
        raise ValueError("sequence must have shape [T, C, H, W]")
    if sequence.shape[1] != 3:
        raise ValueError(f"expected 3 channels, got {sequence.shape[1]}")
    clipped = np.clip(sequence, 0.0, 1.0)
    frames = np.transpose(clipped, (0, 2, 3, 1))
    return np.round(frames * 255.0).astype(np.uint8)


def make_side_by_side_frames(
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    open_loop_prediction: np.ndarray,
    separator_width: int = 2,
    warmup_length: int | None = None,
    add_labels: bool = True,
) -> NDArray[np.uint8]:
    """Create ``[T, H, W_out, 3]`` frames for target/reconstruction/prediction video."""

    gt = tensor_sequence_to_uint8(ground_truth)
    recon = tensor_sequence_to_uint8(reconstruction)
    pred = tensor_sequence_to_uint8(open_loop_prediction)
    if gt.shape != recon.shape or gt.shape != pred.shape:
        raise ValueError("all frame sequences must share shape [T, H, W, 3]")

    separator = np.full((gt.shape[0], gt.shape[1], separator_width, 3), 255, dtype=np.uint8)
    frames = np.concatenate([gt, separator, recon, separator, pred], axis=2)
    if not add_labels:
        return frames
    return np.stack(
        [
            add_frame_header(
                frame=frame,
                timestep=timestep,
                panel_width=gt.shape[2],
                separator_width=separator_width,
                warmup_length=warmup_length,
            )
            for timestep, frame in enumerate(frames)
        ],
        axis=0,
    )


def add_frame_header(
    frame: NDArray[np.uint8],
    timestep: int,
    panel_width: int,
    separator_width: int,
    warmup_length: int | None,
) -> NDArray[np.uint8]:
    """Add compact labels above one side-by-side frame."""

    canvas = np.zeros((frame.shape[0] + HEADER_HEIGHT, frame.shape[1], 3), dtype=np.uint8)
    canvas[HEADER_HEIGHT:] = frame
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    status = ""
    if warmup_length is not None:
        status = "ctx" if timestep < warmup_length - 1 else "open"
    labels = (PANEL_LABELS[0], PANEL_LABELS[1], f"{PANEL_LABELS[2]} {status}".rstrip())
    for panel_index, label in enumerate(labels):
        x = panel_index * (panel_width + separator_width)
        draw.text((x + 3, 2), label, fill=(255, 255, 255), font=font)
    draw.text((frame.shape[1] - 28, 2), f"t={timestep:02d}", fill=(180, 180, 180), font=font)
    return np.asarray(image, dtype=np.uint8)


def write_prediction_video(
    output_path: Path,
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    open_loop_prediction: np.ndarray,
    fps: int = 6,
    warmup_length: int | None = None,
) -> Path:
    """Write an MP4 with columns: target obs_{t+1}, recon obs_t, predicted obs_{t+1}."""

    if fps < 1:
        raise ValueError("fps must be >= 1")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = make_side_by_side_frames(
        ground_truth=ground_truth,
        reconstruction=reconstruction,
        open_loop_prediction=open_loop_prediction,
        warmup_length=warmup_length,
    )
    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    return output_path


def make_prediction_contact_sheet(
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    open_loop_prediction: np.ndarray,
    timesteps: Sequence[int],
    separator_width: int = 2,
    max_cols: int = 4,
    warmup_length: int | None = None,
) -> NDArray[np.uint8]:
    """Tile selected GT/reconstruction/open-loop timestep strips into a contact sheet."""

    if max_cols < 1:
        raise ValueError("max_cols must be >= 1")
    frames = make_side_by_side_frames(
        ground_truth=ground_truth,
        reconstruction=reconstruction,
        open_loop_prediction=open_loop_prediction,
        separator_width=separator_width,
        warmup_length=warmup_length,
    )
    if not timesteps:
        raise ValueError("timesteps must contain at least one index")

    selected = []
    for timestep in timesteps:
        if timestep < 0 or timestep >= frames.shape[0]:
            raise ValueError(f"timestep {timestep} is outside sequence length {frames.shape[0]}")
        selected.append(frames[timestep])

    tiles = np.stack(selected, axis=0)
    cols = min(max_cols, tiles.shape[0])
    rows = ceil(tiles.shape[0] / cols)
    height, width = tiles.shape[1:3]
    sheet = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile
    return sheet


def write_prediction_contact_sheet(
    output_path: Path,
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    open_loop_prediction: np.ndarray,
    timesteps: Sequence[int],
    max_cols: int = 4,
    warmup_length: int | None = None,
) -> Path:
    """Write a PNG contact sheet for selected sequence timesteps."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet = make_prediction_contact_sheet(
        ground_truth=ground_truth,
        reconstruction=reconstruction,
        open_loop_prediction=open_loop_prediction,
        timesteps=timesteps,
        max_cols=max_cols,
        warmup_length=warmup_length,
    )
    imageio.imwrite(output_path, sheet)
    return output_path


def evenly_spaced_timesteps(sequence_length: int, count: int) -> list[int]:
    """Return deterministic timestep indices for image-strip/contact-sheet output."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if count < 1:
        raise ValueError("count must be >= 1")
    sample_count = min(sequence_length, count)
    return np.linspace(0, sequence_length - 1, sample_count, dtype=np.int64).tolist()
