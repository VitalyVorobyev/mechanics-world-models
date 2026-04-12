"""Offline RSSM world-model training on saved cartpole pixel trajectories."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import EpisodeSequenceDataset, ImageCropConfig
from data.image_transforms import add_crop_args, crop_config_from_args
from models import VisualWorldModel, compute_world_model_loss


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for offline RSSM training."""

    dataset_dir: Path
    checkpoint_dir: Path
    sequence_length: int = 16
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    hidden_size: int = 128
    latent_size: int = 32
    embedding_size: int = 256
    kl_weight: float = 1.0
    reconstruction_weight: float = 1.0
    foreground_reconstruction_weight: float = 0.0
    foreground_mask_floor: float = 0.02
    foreground_mask_kernel_size: int = 7
    transition_reconstruction_weight: float = 0.0
    foreground_transition_reconstruction_weight: float = 0.0
    dynamic_reconstruction_weight: float = 0.0
    dynamic_mask_floor: float = 0.05
    latent_consistency_weight: float = 0.0
    device: str = "auto"
    val_fraction: float = 0.1
    seed: int = 0
    num_workers: int = 0
    grad_clip: float = 100.0
    eval_every: int = 1
    log_every_steps: int = 100
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
    history_path: Path | None = None
    resume_from: Path | None = None
    crop_config: ImageCropConfig = field(default_factory=ImageCropConfig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--foreground-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--foreground-mask-floor", type=float, default=0.02)
    parser.add_argument("--foreground-mask-kernel-size", type=int, default=7)
    parser.add_argument("--transition-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--foreground-transition-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-mask-floor", type=float, default=0.05)
    parser.add_argument("--latent-consistency-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    add_crop_args(parser, default_mode="none")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Convert parsed CLI args to a typed config dataclass."""

    return TrainConfig(
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        embedding_size=args.embedding_size,
        kl_weight=args.kl_weight,
        reconstruction_weight=args.reconstruction_weight,
        foreground_reconstruction_weight=args.foreground_reconstruction_weight,
        foreground_mask_floor=args.foreground_mask_floor,
        foreground_mask_kernel_size=args.foreground_mask_kernel_size,
        transition_reconstruction_weight=args.transition_reconstruction_weight,
        foreground_transition_reconstruction_weight=(
            args.foreground_transition_reconstruction_weight
        ),
        dynamic_reconstruction_weight=args.dynamic_reconstruction_weight,
        dynamic_mask_floor=args.dynamic_mask_floor,
        latent_consistency_weight=args.latent_consistency_weight,
        device=args.device,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        log_every_steps=args.log_every_steps,
        max_train_episodes=args.max_train_episodes,
        max_val_episodes=args.max_val_episodes,
        history_path=args.history_path,
        resume_from=args.resume_from,
        crop_config=crop_config_from_args(args) or ImageCropConfig(),
    )


def train(config: TrainConfig, console: Console | None = None) -> None:
    """Run offline world-model training and save checkpoints."""

    console = console or Console()
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.log_every_steps < 1:
        raise ValueError("log_every_steps must be >= 1")
    if config.foreground_reconstruction_weight < 0.0:
        raise ValueError("foreground_reconstruction_weight must be >= 0")
    if not 0.0 <= config.foreground_mask_floor <= 1.0:
        raise ValueError("foreground_mask_floor must satisfy 0 <= floor <= 1")
    if config.foreground_mask_kernel_size < 1 or config.foreground_mask_kernel_size % 2 == 0:
        raise ValueError("foreground_mask_kernel_size must be a positive odd integer")
    if config.transition_reconstruction_weight < 0.0:
        raise ValueError("transition_reconstruction_weight must be >= 0")
    if config.foreground_transition_reconstruction_weight < 0.0:
        raise ValueError("foreground_transition_reconstruction_weight must be >= 0")
    if config.dynamic_reconstruction_weight < 0.0:
        raise ValueError("dynamic_reconstruction_weight must be >= 0")
    if not 0.0 <= config.dynamic_mask_floor <= 1.0:
        raise ValueError("dynamic_mask_floor must satisfy 0 <= floor <= 1")

    set_seed(config.seed)
    device = resolve_device(config.device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = resolve_history_path(config)

    train_dataset = EpisodeSequenceDataset(
        dataset_dir=config.dataset_dir,
        sequence_length=config.sequence_length,
        split="train",
        val_fraction=config.val_fraction,
        seed=config.seed,
        max_episodes=config.max_train_episodes,
        crop_config=config.crop_config,
    )
    val_dataset: EpisodeSequenceDataset | None = None
    if config.val_fraction > 0.0:
        val_dataset = EpisodeSequenceDataset(
            dataset_dir=config.dataset_dir,
            sequence_length=config.sequence_length,
            split="val",
            val_fraction=config.val_fraction,
            seed=config.seed,
            max_episodes=config.max_val_episodes,
            crop_config=config.crop_config,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    first_sample = train_dataset[0]
    action_dim = int(first_sample["actions"].shape[-1])
    image_size = int(first_sample["observations"].shape[-1])
    model = VisualWorldModel(
        action_dim=action_dim,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
        image_size=image_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch = 1
    global_step = 0

    if config.resume_from is not None:
        checkpoint_epoch, global_step = load_checkpoint(config.resume_from, model, optimizer, device)
        start_epoch = checkpoint_epoch + 1
    prepare_history_file(history_path, resume=config.resume_from is not None)

    print_run_summary(
        console=console,
        device=device,
        history_path=history_path,
        train_sequences=len(train_dataset),
        train_batches=len(train_loader),
        val_sequences=len(val_dataset) if val_dataset is not None else None,
        val_batches=len(val_loader) if val_loader is not None else None,
        model_parameters=count_parameters(model),
        crop_config=config.crop_config,
    )

    for epoch in range(start_epoch, config.epochs + 1):
        console.rule(f"[bold cyan]epoch {epoch:04d}[/]")
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            kl_weight=config.kl_weight,
            reconstruction_weight=config.reconstruction_weight,
            foreground_reconstruction_weight=config.foreground_reconstruction_weight,
            foreground_mask_floor=config.foreground_mask_floor,
            foreground_mask_kernel_size=config.foreground_mask_kernel_size,
            transition_reconstruction_weight=config.transition_reconstruction_weight,
            foreground_transition_reconstruction_weight=(
                config.foreground_transition_reconstruction_weight
            ),
            dynamic_reconstruction_weight=config.dynamic_reconstruction_weight,
            dynamic_mask_floor=config.dynamic_mask_floor,
            latent_consistency_weight=config.latent_consistency_weight,
            grad_clip=config.grad_clip,
            epoch=epoch,
            start_global_step=global_step,
            log_every_steps=config.log_every_steps,
            history_path=history_path,
            console=console,
        )
        emit_metrics("train_epoch", epoch, train_metrics, global_step, history_path, console)

        val_metrics: dict[str, float] | None = None
        if val_loader is not None and epoch % config.eval_every == 0:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                kl_weight=config.kl_weight,
                reconstruction_weight=config.reconstruction_weight,
                foreground_reconstruction_weight=config.foreground_reconstruction_weight,
                foreground_mask_floor=config.foreground_mask_floor,
                foreground_mask_kernel_size=config.foreground_mask_kernel_size,
                transition_reconstruction_weight=config.transition_reconstruction_weight,
                foreground_transition_reconstruction_weight=(
                    config.foreground_transition_reconstruction_weight
                ),
                dynamic_reconstruction_weight=config.dynamic_reconstruction_weight,
                dynamic_mask_floor=config.dynamic_mask_floor,
                latent_consistency_weight=config.latent_consistency_weight,
            )
            emit_metrics("val", epoch, val_metrics, global_step, history_path, console)

        checkpoint_path = save_checkpoint(
            checkpoint_dir=config.checkpoint_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
            metrics=val_metrics or train_metrics,
            global_step=global_step,
            history_path=history_path,
        )
        console.print(
            f"[green]saved checkpoint[/green] "
            f"epoch_path={checkpoint_path} latest_path={config.checkpoint_dir / 'latest.pt'}",
        )


def run_epoch(
    model: VisualWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float,
    reconstruction_weight: float,
    foreground_reconstruction_weight: float,
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
    transition_reconstruction_weight: float,
    foreground_transition_reconstruction_weight: float,
    dynamic_reconstruction_weight: float,
    dynamic_mask_floor: float,
    latent_consistency_weight: float,
    grad_clip: float,
    epoch: int,
    start_global_step: int,
    log_every_steps: int,
    history_path: Path,
    console: Console,
) -> tuple[dict[str, float], int]:
    """Run one training epoch."""

    model.train()
    totals: dict[str, float] = {}
    num_batches = 0
    global_step = start_global_step

    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        needs_transition_outputs = (
            latent_consistency_weight > 0.0
            or transition_reconstruction_weight > 0.0
            or foreground_transition_reconstruction_weight > 0.0
        )
        needs_next_observations = needs_transition_outputs or dynamic_reconstruction_weight > 0.0
        next_observations = (
            batch["next_observations"].to(device)
            if needs_next_observations
            else None
        )
        # observations: [B, T, 3, 84, 84], actions: [B, T, A].
        # When enabled, next_observations[:, t] is the target for
        # obs_t --action_t--> obs_{t+1}.
        outputs = model(
            observations=observations,
            actions=actions,
            next_observations=next_observations if needs_transition_outputs else None,
        )
        losses = compute_world_model_loss(
            outputs,
            observations,
            kl_weight=kl_weight,
            reconstruction_weight=reconstruction_weight,
            foreground_reconstruction_weight=foreground_reconstruction_weight,
            foreground_mask_floor=foreground_mask_floor,
            foreground_mask_kernel_size=foreground_mask_kernel_size,
            transition_reconstruction_weight=transition_reconstruction_weight,
            foreground_transition_reconstruction_weight=foreground_transition_reconstruction_weight,
            dynamic_reconstruction_weight=dynamic_reconstruction_weight,
            dynamic_mask_floor=dynamic_mask_floor,
            latent_consistency_weight=latent_consistency_weight,
            next_observations=next_observations,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        grad_norm = compute_grad_norm(model)
        if grad_clip > 0.0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        optimizer.step()

        batch_metrics = tensor_metrics_to_float(losses)
        batch_metrics["grad_norm"] = grad_norm
        update_totals(totals, batch_metrics)
        num_batches += 1
        global_step += 1
        if global_step % log_every_steps == 0:
            emit_metrics("train", epoch, batch_metrics, global_step, history_path, console)

    return average_totals(totals, num_batches), global_step


@torch.no_grad()
def evaluate(
    model: VisualWorldModel,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    reconstruction_weight: float,
    foreground_reconstruction_weight: float,
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
    transition_reconstruction_weight: float,
    foreground_transition_reconstruction_weight: float,
    dynamic_reconstruction_weight: float,
    dynamic_mask_floor: float,
    latent_consistency_weight: float,
) -> dict[str, float]:
    """Evaluate the world model without optimizer updates."""

    model.eval()
    totals: dict[str, float] = {}
    num_batches = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        needs_transition_outputs = (
            latent_consistency_weight > 0.0
            or transition_reconstruction_weight > 0.0
            or foreground_transition_reconstruction_weight > 0.0
        )
        needs_next_observations = needs_transition_outputs or dynamic_reconstruction_weight > 0.0
        next_observations = (
            batch["next_observations"].to(device)
            if needs_next_observations
            else None
        )
        outputs = model(
            observations=observations,
            actions=actions,
            next_observations=next_observations if needs_transition_outputs else None,
        )
        losses = compute_world_model_loss(
            outputs,
            observations,
            kl_weight=kl_weight,
            reconstruction_weight=reconstruction_weight,
            foreground_reconstruction_weight=foreground_reconstruction_weight,
            foreground_mask_floor=foreground_mask_floor,
            foreground_mask_kernel_size=foreground_mask_kernel_size,
            transition_reconstruction_weight=transition_reconstruction_weight,
            foreground_transition_reconstruction_weight=foreground_transition_reconstruction_weight,
            dynamic_reconstruction_weight=dynamic_reconstruction_weight,
            dynamic_mask_floor=dynamic_mask_floor,
            latent_consistency_weight=latent_consistency_weight,
            next_observations=next_observations,
        )
        update_totals(totals, tensor_metrics_to_float(losses))
        num_batches += 1
    return average_totals(totals, num_batches)


def tensor_metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert scalar tensor metrics to Python floats."""

    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def update_totals(totals: dict[str, float], metrics: dict[str, float]) -> None:
    """Accumulate scalar metrics."""

    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + value


def average_totals(totals: dict[str, float], count: int) -> dict[str, float]:
    """Average metric totals over batches."""

    if count < 1:
        raise ValueError("cannot average metrics over zero batches")
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: VisualWorldModel,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    metrics: dict[str, float],
    global_step: int,
    history_path: Path,
) -> Path:
    """Save model/optimizer state and config for later resume/evaluation."""

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": serializable_config(config),
        "metrics": metrics,
        "history_path": str(history_path),
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    return epoch_path


def load_checkpoint(
    path: Path,
    model: VisualWorldModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    """Load model/optimizer state and return ``(epoch, global_step)``."""

    checkpoint = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed_missing = [name for name in missing if name.startswith("latent_predictor.")]
    disallowed_missing = [name for name in missing if not name.startswith("latent_predictor.")]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"checkpoint model state mismatch: missing={disallowed_missing}, unexpected={unexpected}",
        )
    if allowed_missing:
        print(
            "Warning: checkpoint is missing latent_predictor parameters; "
            "initialized them from the current model defaults.",
        )
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except ValueError:
        if not allowed_missing:
            raise
        print(
            "Warning: checkpoint optimizer state is incompatible with the "
            "new latent_predictor parameters; using a fresh optimizer state.",
        )
    return int(checkpoint["epoch"]), int(checkpoint.get("global_step", 0))


def serializable_config(config: TrainConfig) -> dict[str, Any]:
    """Convert config dataclass values to JSON/checkpoint-friendly objects."""

    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def resolve_history_path(config: TrainConfig) -> Path:
    """Return the JSONL metrics history path for this run."""

    return config.history_path or (config.checkpoint_dir / "history.jsonl")


def prepare_history_file(history_path: Path, resume: bool) -> None:
    """Create or reset the JSONL metrics history file."""

    history_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        history_path.write_text("")
    else:
        history_path.touch(exist_ok=True)


def emit_metrics(
    split: str,
    epoch: int,
    metrics: dict[str, float],
    global_step: int,
    history_path: Path,
    console: Console,
) -> None:
    """Print metrics and append a JSONL history record."""

    log_metrics(split, epoch, metrics, global_step=global_step, console=console)
    append_history_record(history_path, split, epoch, global_step, metrics)


def append_history_record(
    history_path: Path,
    split: str,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    """Append one JSONL metrics record."""

    record = {
        "epoch": epoch,
        "global_step": global_step,
        "split": split,
        **metrics,
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def log_metrics(
    split: str,
    epoch: int,
    metrics: dict[str, float],
    console: Console,
    global_step: int | None = None,
) -> None:
    """Print one compact metrics line."""

    metrics_text = " ".join(f"{key}={value:.6f}" for key, value in sorted(metrics.items()))
    step_text = "" if global_step is None else f" global_step={global_step:07d}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    split_style = {
        "train": "bold green",
        "train_epoch": "bold blue",
        "val": "bold magenta",
    }.get(split, "bold")
    console.print(
        f"[dim]{timestamp}[/dim] "
        f"epoch=[bold]{epoch:04d}[/bold]{step_text} "
        f"split=[{split_style}]{split}[/] "
        f"{metrics_text}",
    )


def print_run_summary(
    console: Console,
    device: torch.device,
    history_path: Path,
    train_sequences: int,
    train_batches: int,
    val_sequences: int | None,
    val_batches: int | None,
    model_parameters: int,
    crop_config: ImageCropConfig,
) -> None:
    """Print one compact Rich summary before training starts."""

    table = Table(title="RSSM Training Run", show_header=False, box=None)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value")
    table.add_row("device", str(device))
    table.add_row("history_path", str(history_path))
    table.add_row("train_sequences", str(train_sequences))
    table.add_row("train_batches_per_epoch", str(train_batches))
    if val_sequences is not None and val_batches is not None:
        table.add_row("val_sequences", str(val_sequences))
        table.add_row("val_batches", str(val_batches))
    table.add_row("crop", str(asdict(crop_config)))
    table.add_row("model_parameters", f"{model_parameters:,}")
    console.print(table)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the global gradient norm before optional clipping."""

    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().data.norm(2)
        squared_norm += float(grad_norm.item() ** 2)
    return squared_norm**0.5


def resolve_device(device_name: str) -> torch.device:
    """Resolve 'auto' to the best available PyTorch device."""

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    """Run the training CLI."""

    console = Console()
    config = config_from_args(build_arg_parser().parse_args())
    console.rule("[bold cyan]RSSM training config[/]")
    console.print_json(json.dumps(serializable_config(config), indent=2))
    train(config, console=console)


if __name__ == "__main__":
    main()
