"""Posterior-only RSSM reconstruction diagnostics for train/val batches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from rich.console import Console
from rich.table import Table
import torch
from torch.utils.data import DataLoader, Subset

from data import EpisodeSequenceDataset, ImageCropConfig
from data.image_transforms import add_crop_args, crop_config_from_args, crop_config_from_mapping
from eval.eval_rssm import (
    deterministic_reconstruction,
    load_model_from_checkpoint,
    select_sequence_indices,
)
from train.train_rssm import resolve_device, set_seed
from viz.debug_grids import write_reconstruction_debug_grid
from viz.rssm_eval_viz import evenly_spaced_timesteps


SplitName = Literal["train", "val"]


@dataclass(frozen=True)
class ReconstructionDebugConfig:
    """Configuration for posterior reconstruction debug grids."""

    checkpoint_path: Path
    dataset_dir: Path
    output_dir: Path
    sequence_length: int | None = None
    batch_size: int = 4
    timesteps: tuple[int, ...] = ()
    max_sequences: int = 4
    seed: int = 0
    val_fraction: float | None = None
    device: str = "auto"
    num_workers: int = 0
    crop_config: ImageCropConfig | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the reconstruction debug CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, nargs="*", default=[])
    parser.add_argument("--max-sequences", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    add_crop_args(parser, default_mode=None)
    return parser


def config_from_args(args: argparse.Namespace) -> ReconstructionDebugConfig:
    """Convert CLI arguments to a typed config dataclass."""

    return ReconstructionDebugConfig(
        checkpoint_path=args.checkpoint_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        timesteps=tuple(args.timesteps),
        max_sequences=args.max_sequences,
        seed=args.seed,
        val_fraction=args.val_fraction,
        device=args.device,
        num_workers=args.num_workers,
        crop_config=crop_config_from_args(args),
    )


def run_reconstruction_debug(
    config: ReconstructionDebugConfig,
    console: Console | None = None,
) -> dict[str, Any]:
    """Save posterior-only reconstruction grids for one train and one val batch."""

    console = console or Console()
    validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    checkpoint = torch.load(
        config.checkpoint_path, map_location=device, weights_only=False
    )
    checkpoint_config = dict(checkpoint.get("config", {}))
    sequence_length = config.sequence_length or int(checkpoint_config.get("sequence_length", 16))
    val_fraction = config.val_fraction
    if val_fraction is None:
        val_fraction = float(checkpoint_config.get("val_fraction", 0.1))
    crop_config = config.crop_config or crop_config_from_mapping(checkpoint_config.get("crop_config"))
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be > 0 and < 1 for train/val reconstruction debug")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "train": EpisodeSequenceDataset(
            dataset_dir=config.dataset_dir,
            sequence_length=sequence_length,
            split="train",
            val_fraction=val_fraction,
            seed=config.seed,
            crop_config=crop_config,
        ),
        "val": EpisodeSequenceDataset(
            dataset_dir=config.dataset_dir,
            sequence_length=sequence_length,
            split="val",
            val_fraction=val_fraction,
            seed=config.seed,
            crop_config=crop_config,
        ),
    }

    first_sample = datasets["train"][0]
    action_dim = int(first_sample["actions"].shape[-1])
    image_size = int(first_sample["observations"].shape[-1])
    model = load_model_from_checkpoint(
        checkpoint=checkpoint,
        checkpoint_config=checkpoint_config,
        action_dim=action_dim,
        image_size=image_size,
        device=device,
    )

    console.print(
        "[cyan]posterior reconstruction debug[/cyan]: targets and decoder outputs "
        "are aligned at the same timestep, both shaped [B, T, 3, H, W] in [0, 1]",
    )

    results: dict[str, Any] = {
        "config": serializable_config(config),
        "sequence_length": sequence_length,
        "val_fraction": val_fraction,
        "crop_config": asdict(crop_config),
        "splits": {},
    }
    table = Table(title="Posterior Reconstruction Debug")
    table.add_column("split", style="cyan")
    table.add_column("mse", justify="right")
    table.add_column("target_range", justify="right")
    table.add_column("recon_range", justify="right")
    table.add_column("grid")

    for split, dataset in datasets.items():
        split_result = run_split_reconstruction_debug(
            split=split,  # type: ignore[arg-type]
            dataset=dataset,
            model=model,
            device=device,
            output_dir=config.output_dir,
            batch_size=config.batch_size,
            max_sequences=config.max_sequences,
            timesteps=config.timesteps,
            seed=config.seed,
            num_workers=config.num_workers,
        )
        results["splits"][split] = split_result
        table.add_row(
            split,
            f"{split_result['mse']:.8f}",
            format_range(split_result["target_range"]),
            format_range(split_result["reconstruction_range"]),
            split_result["grid_path"],
        )

    metrics_path = config.output_dir / "reconstruction_debug.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(table)
    console.print(f"[green]wrote[/green] {metrics_path}")
    return results


@torch.no_grad()
def run_split_reconstruction_debug(
    split: SplitName,
    dataset: EpisodeSequenceDataset,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    max_sequences: int,
    timesteps: Sequence[int],
    seed: int,
    num_workers: int,
) -> dict[str, Any]:
    """Run deterministic posterior reconstruction for one split batch."""

    offset = 0 if split == "train" else 1_000_003
    selected_indices = select_sequence_indices(
        total_sequences=len(dataset),
        num_sequences=min(batch_size, len(dataset)),
        seed=seed + offset,
    )
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=len(selected_indices),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    batch = next(iter(loader))
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)

    # observations/reconstructions: [B, T, 3, H, W] float32 in [0, 1].
    # recon_t is decoded from q(z_t | h_t, embed_t), so it should align with
    # observations[:, t] exactly. This script intentionally does no open-loop.
    reconstructions = deterministic_reconstruction(model, observations, actions)
    mse = float(torch.mean((reconstructions - observations) ** 2).detach().cpu())
    clean_timesteps = tuple(timesteps) if timesteps else tuple(evenly_spaced_timesteps(observations.shape[1], 4))
    grid_path = output_dir / f"{split}_posterior_reconstruction_grid.png"
    write_reconstruction_debug_grid(
        output_path=grid_path,
        targets=observations.detach().cpu().numpy(),
        reconstructions=reconstructions.detach().cpu().numpy(),
        timesteps=clean_timesteps,
        max_sequences=max_sequences,
    )

    return {
        "selected_indices": selected_indices,
        "grid_path": str(grid_path),
        "timesteps": list(clean_timesteps),
        "observation_shape": list(observations.shape),
        "reconstruction_shape": list(reconstructions.shape),
        "target_range": tensor_range(observations),
        "reconstruction_range": tensor_range(reconstructions),
        "mse": mse,
    }


def validate_config(config: ReconstructionDebugConfig) -> None:
    """Validate reconstruction debug settings before loading model/data."""

    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.max_sequences < 1:
        raise ValueError("max_sequences must be >= 1")


def tensor_range(tensor: torch.Tensor) -> list[float]:
    """Return ``[min, max]`` for a tensor as Python floats."""

    detached = tensor.detach()
    return [float(detached.amin().cpu()), float(detached.amax().cpu())]


def format_range(values: Sequence[float]) -> str:
    """Format a two-value numeric range for console output."""

    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def serializable_config(config: ReconstructionDebugConfig) -> dict[str, Any]:
    """Convert config dataclass values to JSON-friendly objects."""

    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def main() -> None:
    """Run the posterior reconstruction debug CLI."""

    console = Console()
    config = config_from_args(build_arg_parser().parse_args())
    console.rule("[bold cyan]posterior reconstruction debug config[/]")
    console.print_json(json.dumps(serializable_config(config), indent=2))
    run_reconstruction_debug(config, console=console)


if __name__ == "__main__":
    main()
