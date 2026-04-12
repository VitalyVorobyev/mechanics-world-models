"""Small image preprocessing helpers for saved trajectory frames."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image


CropMode = Literal["none", "center", "rect"]


@dataclass(frozen=True)
class ImageCropConfig:
    """Optional deterministic crop applied before image normalization.

    Cropping operates on loaded frames shaped ``[T, H, W, 3]`` and resizes the
    result back to the original ``H x W`` so downstream tensors keep shape
    ``[T, C, H, W]``. For the current cartpole pipeline that means ``84 x 84``.
    """

    mode: CropMode = "none"
    top: int | None = None
    left: int | None = None
    height: int | None = None
    width: int | None = None

    @property
    def enabled(self) -> bool:
        """Return whether this config changes input frames."""

        return self.mode != "none"


def add_crop_args(parser: argparse.ArgumentParser, default_mode: CropMode | None = "none") -> None:
    """Add shared image-crop CLI flags to an argument parser."""

    parser.add_argument("--crop-mode", choices=("none", "center", "rect"), default=default_mode)
    parser.add_argument("--crop-top", type=int, default=None)
    parser.add_argument("--crop-left", type=int, default=None)
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--crop-width", type=int, default=None)


def crop_config_from_args(args: argparse.Namespace) -> ImageCropConfig | None:
    """Build a crop config from shared CLI args.

    Returns ``None`` when ``--crop-mode`` was omitted from an eval-style parser
    whose default mode is ``None``; callers can then fall back to checkpoint
    metadata.
    """

    mode = getattr(args, "crop_mode", "none")
    if mode is None:
        return None
    return ImageCropConfig(
        mode=mode,
        top=getattr(args, "crop_top", None),
        left=getattr(args, "crop_left", None),
        height=getattr(args, "crop_height", None),
        width=getattr(args, "crop_width", None),
    )


def crop_config_from_mapping(value: Any) -> ImageCropConfig:
    """Convert checkpoint/config JSON data into an ``ImageCropConfig``."""

    if value is None:
        return ImageCropConfig()
    if isinstance(value, ImageCropConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"expected crop config dict, got {type(value).__name__}")
    return ImageCropConfig(
        mode=value.get("mode", "none"),
        top=value.get("top"),
        left=value.get("left"),
        height=value.get("height"),
        width=value.get("width"),
    )


def apply_image_crop_resize(
    frames: NDArray[np.uint8],
    crop_config: ImageCropConfig | None,
) -> NDArray[np.uint8]:
    """Apply deterministic crop and resize to ``[T, H, W, 3]`` uint8 frames."""

    config = crop_config or ImageCropConfig()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must have shape [T, H, W, 3], got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"frames must have dtype uint8, got {frames.dtype}")
    if not config.enabled:
        return frames

    frame_count, output_height, output_width = frames.shape[:3]
    top, left, crop_height, crop_width = resolve_crop_rectangle(
        image_height=output_height,
        image_width=output_width,
        crop_config=config,
    )
    cropped = frames[:, top : top + crop_height, left : left + crop_width]
    if cropped.shape[1:3] == (output_height, output_width):
        return cropped

    resized = np.empty((frame_count, output_height, output_width, 3), dtype=np.uint8)
    for index, frame in enumerate(cropped):
        image = Image.fromarray(frame)
        resized[index] = np.asarray(
            image.resize((output_width, output_height), resample=Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    return resized


def resolve_crop_rectangle(
    image_height: int,
    image_width: int,
    crop_config: ImageCropConfig,
) -> tuple[int, int, int, int]:
    """Resolve ``(top, left, height, width)`` for a crop config and image size."""

    if crop_config.mode == "none":
        return 0, 0, image_height, image_width

    height = require_positive_int(crop_config.height, "crop_height")
    width = require_positive_int(crop_config.width, "crop_width")
    if height > image_height or width > image_width:
        raise ValueError(
            f"crop size {(height, width)} exceeds image size {(image_height, image_width)}",
        )

    if crop_config.mode == "center":
        top = (image_height - height) // 2
        left = (image_width - width) // 2
    elif crop_config.mode == "rect":
        top = require_nonnegative_int(crop_config.top, "crop_top")
        left = require_nonnegative_int(crop_config.left, "crop_left")
    else:
        raise ValueError(f"unsupported crop mode {crop_config.mode!r}")

    if top + height > image_height or left + width > image_width:
        raise ValueError(
            f"crop rectangle {(top, left, height, width)} exceeds image size {(image_height, image_width)}",
        )
    return top, left, height, width


def require_positive_int(value: int | None, name: str) -> int:
    """Return a required positive integer crop field."""

    if value is None:
        raise ValueError(f"{name} is required for crop mode")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return int(value)


def require_nonnegative_int(value: int | None, name: str) -> int:
    """Return a required nonnegative integer crop field."""

    if value is None:
        raise ValueError(f"{name} is required for rect crop mode")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return int(value)
