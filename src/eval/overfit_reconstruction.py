"""Overfit the RSSM on one batch or one sequence and save reconstruction grids."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from rich.console import Console
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import EpisodeSequenceDataset, ImageCropConfig
from data.image_transforms import add_crop_args, crop_config_from_args
from eval.eval_rssm import (
    deterministic_reconstruction,
    load_model_from_checkpoint,
    select_sequence_indices,
)
from models import VisualWorldModel, compute_world_model_loss
from train.train_rssm import (
    compute_grad_norm,
    resolve_device,
    set_seed,
    tensor_metrics_to_float,
)
from viz.debug_grids import write_reconstruction_debug_grid
from viz.rssm_eval_viz import evenly_spaced_timesteps


OverfitMode = Literal["batch", "sequence"]
SplitName = Literal["train", "val"]


@dataclass(frozen=True)
class OverfitReconstructionConfig:
    """Configuration for single-batch/single-sequence reconstruction overfit."""

    dataset_dir: Path
    output_dir: Path
    checkpoint_path: Path | None = None
    sequence_length: int = 16
    batch_size: int = 4
    mode: OverfitMode = "batch"
    split: SplitName = "train"
    steps: int = 500
    save_every: int = 100
    learning_rate: float = 1e-3
    kl_weight: float = 0.0
    hidden_size: int = 128
    latent_size: int = 32
    embedding_size: int = 256
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "auto"
    grad_clip: float = 100.0
    timesteps: tuple[int, ...] = ()
    crop_config: ImageCropConfig = field(default_factory=ImageCropConfig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the overfit debug CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mode", choices=("batch", "sequence"), default="batch")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--timesteps", type=int, nargs="*", default=[])
    add_crop_args(parser, default_mode="none")
    return parser


def config_from_args(args: argparse.Namespace) -> OverfitReconstructionConfig:
    """Convert CLI arguments to a typed config dataclass."""

    return OverfitReconstructionConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        mode=args.mode,
        split=args.split,
        steps=args.steps,
        save_every=args.save_every,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        embedding_size=args.embedding_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        grad_clip=args.grad_clip,
        timesteps=tuple(args.timesteps),
        crop_config=crop_config_from_args(args) or ImageCropConfig(),
    )


def run_overfit_reconstruction(
    config: OverfitReconstructionConfig,
    console: Console | None = None,
) -> dict[str, Any]:
    """Repeatedly optimize one batch/sequence and save reconstruction grids."""

    console = console or Console()
    validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EpisodeSequenceDataset(
        dataset_dir=config.dataset_dir,
        sequence_length=config.sequence_length,
        split=config.split,
        val_fraction=config.val_fraction,
        seed=config.seed,
        crop_config=config.crop_config,
    )
    effective_batch_size = 1 if config.mode == "sequence" else config.batch_size
    selected_indices = select_sequence_indices(
        total_sequences=len(dataset),
        num_sequences=min(effective_batch_size, len(dataset)),
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
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)

    checkpoint: dict[str, Any] | None = None
    checkpoint_config: dict[str, Any] = {}
    if config.checkpoint_path is not None:
        checkpoint = torch.load(
            config.checkpoint_path, map_location=device, weights_only=False
        )
        checkpoint_config = dict(checkpoint.get("config", {}))

    action_dim = int(actions.shape[-1])
    image_size = int(observations.shape[-1])
    model = build_model(
        config=config,
        checkpoint=checkpoint,
        checkpoint_config=checkpoint_config,
        action_dim=action_dim,
        image_size=image_size,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    console.print(
        "[cyan]single-batch overfit[/cyan]: optimization uses the existing stochastic "
        "RSSM training loss; saved grids use posterior means. Targets and reconstructions "
        "are same-timestep [B, T, 3, H, W] tensors in [0, 1]."
    )
    console.print(
        f"split={config.split} mode={config.mode} selected_indices={selected_indices} "
        f"observations_shape={tuple(observations.shape)} actions_shape={tuple(actions.shape)} "
        f"crop={config.crop_config}"
    )

    history_path = config.output_dir / "overfit_history.jsonl"
    history_path.write_text("", encoding="utf-8")
    timesteps = config.timesteps or tuple(evenly_spaced_timesteps(config.sequence_length, 4))
    saved_grids: list[str] = []

    saved_grids.append(
        str(save_grid(config.output_dir, model, observations, actions, timesteps, step=0)),
    )
    final_metrics: dict[str, float] = {}
    for step in range(1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(observations=observations, actions=actions)
        losses = compute_world_model_loss(outputs, observations, kl_weight=config.kl_weight)
        losses["total_loss"].backward()
        grad_norm = compute_grad_norm(model)
        if config.grad_clip > 0.0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        optimizer.step()

        final_metrics = tensor_metrics_to_float(losses)
        final_metrics["grad_norm"] = grad_norm
        append_history(history_path, step, final_metrics)
        if step == 1 or step % config.save_every == 0 or step == config.steps:
            saved_grids.append(
                str(save_grid(config.output_dir, model, observations, actions, timesteps, step=step)),
            )
            console.print(
                f"step={step:06d} total_loss={final_metrics['total_loss']:.6f} "
                f"reconstruction_loss={final_metrics['reconstruction_loss']:.6f} "
                f"kl_loss={final_metrics['kl_loss']:.6f} grad_norm={grad_norm:.6f}"
            )

    checkpoint_path = config.output_dir / "overfit_latest.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": serializable_config(config),
            "step": config.steps,
            "selected_indices": selected_indices,
            "metrics": final_metrics,
        },
        checkpoint_path,
    )
    summary = {
        "config": serializable_config(config),
        "selected_indices": selected_indices,
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "target_range": tensor_range(observations),
        "saved_grids": saved_grids,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "final_metrics": final_metrics,
    }
    summary_path = config.output_dir / "overfit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(f"[green]wrote[/green] {summary_path}")
    return summary


def build_model(
    config: OverfitReconstructionConfig,
    checkpoint: dict[str, Any] | None,
    checkpoint_config: dict[str, Any],
    action_dim: int,
    image_size: int,
    device: torch.device,
) -> VisualWorldModel:
    """Build a fresh model or load one from a checkpoint."""

    if checkpoint is not None:
        return load_model_from_checkpoint(
            checkpoint=checkpoint,
            checkpoint_config=checkpoint_config,
            action_dim=action_dim,
            image_size=image_size,
            device=device,
        )
    return VisualWorldModel(
        action_dim=action_dim,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
        image_size=image_size,
    ).to(device)


@torch.no_grad()
def save_grid(
    output_dir: Path,
    model: VisualWorldModel,
    observations: torch.Tensor,
    actions: torch.Tensor,
    timesteps: Sequence[int],
    step: int,
) -> Path:
    """Save a deterministic posterior reconstruction grid for the fixed batch."""

    model.eval()
    reconstructions = deterministic_reconstruction(model, observations, actions)
    grid_path = output_dir / f"reconstruction_step_{step:06d}.png"
    write_reconstruction_debug_grid(
        output_path=grid_path,
        targets=observations.detach().cpu().numpy(),
        reconstructions=reconstructions.detach().cpu().numpy(),
        timesteps=timesteps,
        max_sequences=observations.shape[0],
    )
    return grid_path


def append_history(history_path: Path, step: int, metrics: dict[str, float]) -> None:
    """Append one JSONL record for the fixed-batch overfit run."""

    record = {"step": step, **metrics}
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_config(config: OverfitReconstructionConfig) -> None:
    """Validate overfit debug settings before loading data."""

    if config.sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.steps < 1:
        raise ValueError("steps must be >= 1")
    if config.save_every < 1:
        raise ValueError("save_every must be >= 1")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")


def tensor_range(tensor: torch.Tensor) -> list[float]:
    """Return ``[min, max]`` for a tensor as Python floats."""

    detached = tensor.detach()
    return [float(detached.amin().cpu()), float(detached.amax().cpu())]


def serializable_config(config: OverfitReconstructionConfig) -> dict[str, Any]:
    """Convert config dataclass values to JSON-friendly objects."""

    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def main() -> None:
    """Run the single-batch reconstruction overfit CLI."""

    console = Console()
    config = config_from_args(build_arg_parser().parse_args())
    console.rule("[bold cyan]single-batch reconstruction overfit config[/]")
    console.print_json(json.dumps(serializable_config(config), indent=2))
    run_overfit_reconstruction(config, console=console)


if __name__ == "__main__":
    main()
