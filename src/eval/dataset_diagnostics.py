"""Dataset image diagnostics for foreground/background imbalance checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from data import index_episode_files, load_trajectory_npz
from viz.debug_grids import absolute_error_heatmap


@dataclass(frozen=True)
class DatasetImageDiagnosticsConfig:
    """Configuration for sampled image statistics over saved episodes."""

    dataset_dir: Path
    output_dir: Path
    max_frames: int = 10_000
    seed: int = 0


@dataclass(frozen=True)
class FrameSample:
    """Reservoir-sampled frames and dataset counting metadata."""

    frames: NDArray[np.uint8]
    total_frames_seen: int
    episode_count: int


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the dataset image diagnostics CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> DatasetImageDiagnosticsConfig:
    """Convert CLI arguments to a typed config dataclass."""

    return DatasetImageDiagnosticsConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        seed=args.seed,
    )


def run_dataset_image_diagnostics(
    config: DatasetImageDiagnosticsConfig,
    console: Console | None = None,
) -> dict[str, Any]:
    """Compute and save average/variance images from sampled dataset frames."""

    console = console or Console()
    if config.max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sample = sample_frames(config.dataset_dir, max_frames=config.max_frames, seed=config.seed)
    frames = sample.frames.astype(np.float32) / 255.0
    # frames: [S, H, W, 3] float32 in [0, 1].
    mean_image = frames.mean(axis=0)
    variance_image = frames.var(axis=0)
    variance_display = normalize_image_for_display(variance_image)
    variance_heatmap = absolute_error_heatmap(
        np.zeros_like(np.transpose(variance_image, (2, 0, 1))),
        np.transpose(variance_image, (2, 0, 1)),
    )

    average_path = config.output_dir / "average_image.png"
    variance_path = config.output_dir / "variance_image.png"
    variance_heatmap_path = config.output_dir / "variance_heatmap.png"
    imageio.imwrite(average_path, to_uint8(mean_image))
    imageio.imwrite(variance_path, to_uint8(variance_display))
    imageio.imwrite(variance_heatmap_path, variance_heatmap)

    stats = {
        "config": serializable_config(config),
        "episode_count": sample.episode_count,
        "total_frames_seen": sample.total_frames_seen,
        "sampled_frames": int(sample.frames.shape[0]),
        "frame_shape": list(sample.frames.shape[1:]),
        "mean_rgb": frames.mean(axis=(0, 1, 2)).tolist(),
        "variance_rgb": frames.var(axis=(0, 1, 2)).tolist(),
        "variance_min": float(variance_image.min()),
        "variance_mean": float(variance_image.mean()),
        "variance_max": float(variance_image.max()),
        "variance_p99": float(np.percentile(variance_image, 99.0)),
        "average_image_path": str(average_path),
        "variance_image_path": str(variance_path),
        "variance_heatmap_path": str(variance_heatmap_path),
    }
    stats_path = config.output_dir / "dataset_image_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_stats_table(console, stats)
    console.print(f"[green]wrote[/green] {stats_path}")
    return stats


def sample_frames(dataset_dir: Path, max_frames: int, seed: int) -> FrameSample:
    """Reservoir-sample image frames across episode files deterministically."""

    rng = np.random.default_rng(seed)
    reservoir: list[NDArray[np.uint8]] = []
    total_frames_seen = 0
    episode_paths = index_episode_files(dataset_dir)

    for episode_path in episode_paths:
        trajectory = load_trajectory_npz(episode_path)
        for frame in trajectory.images:
            total_frames_seen += 1
            if len(reservoir) < max_frames:
                reservoir.append(frame.copy())
                continue
            replacement_index = int(rng.integers(0, total_frames_seen))
            if replacement_index < max_frames:
                reservoir[replacement_index] = frame.copy()

    if not reservoir:
        raise ValueError(f"no image frames found in {dataset_dir}")
    return FrameSample(
        frames=np.stack(reservoir, axis=0),
        total_frames_seen=total_frames_seen,
        episode_count=len(episode_paths),
    )


def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize a nonnegative RGB image to ``[0, 1]`` for display."""

    scale = float(np.percentile(image, 99.0))
    if scale <= 0.0:
        scale = float(image.max())
    if scale <= 0.0:
        scale = 1.0
    return np.clip(image / scale, 0.0, 1.0)


def to_uint8(image: np.ndarray) -> NDArray[np.uint8]:
    """Convert an ``[H, W, 3]`` float image in ``[0, 1]`` to ``uint8``."""

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected image shape [H, W, 3], got {image.shape}")
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def serializable_config(config: DatasetImageDiagnosticsConfig) -> dict[str, Any]:
    """Convert config dataclass values to JSON-friendly objects."""

    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def print_stats_table(console: Console, stats: dict[str, Any]) -> None:
    """Print a compact summary of image diagnostics."""

    table = Table(title="Dataset Image Diagnostics", show_header=False)
    table.add_column("key", style="cyan")
    table.add_column("value")
    for key in (
        "episode_count",
        "total_frames_seen",
        "sampled_frames",
        "frame_shape",
        "variance_mean",
        "variance_p99",
        "average_image_path",
        "variance_image_path",
        "variance_heatmap_path",
    ):
        table.add_row(key, str(stats[key]))
    console.print(table)


def main() -> None:
    """Run the dataset image diagnostics CLI."""

    console = Console()
    config = config_from_args(build_arg_parser().parse_args())
    console.rule("[bold cyan]dataset image diagnostics config[/]")
    console.print_json(json.dumps(serializable_config(config), indent=2))
    run_dataset_image_diagnostics(config, console=console)


if __name__ == "__main__":
    main()
