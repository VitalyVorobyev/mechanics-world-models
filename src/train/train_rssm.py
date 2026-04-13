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
from models.losses import foreground_reconstruction_mask


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for offline RSSM training."""

    dataset_dir: Path
    checkpoint_dir: Path
    sequence_length: int = 16
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"
    epochs: int = 10
    hidden_size: int = 128
    latent_size: int = 32
    embedding_size: int = 256
    kl_weight: float = 1.0
    kl_balance_alpha: float = 0.8
    kl_free_nats: float = 3.0
    reconstruction_weight: float = 1.0
    foreground_reconstruction_weight: float = 0.0
    foreground_mask_floor: float = 0.02
    foreground_mask_kernel_size: int = 7
    transition_reconstruction_weight: float = 0.0
    foreground_transition_reconstruction_weight: float = 0.0
    imagination_reconstruction_weight: float = 0.0
    foreground_imagination_reconstruction_weight: float = 0.0
    imagination_context_steps: int = 4
    imagination_horizon: int = 0
    reward_head: bool = False
    reward_head_hidden_size: int = 200
    reward_weight: float = 0.0
    imagined_reward_weight: float = 0.0
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
    val_open_loop_every_steps: int = 0
    val_open_loop_warmup: int = 4
    val_open_loop_horizons: tuple[int, ...] = (1, 5, 10, 20)
    val_open_loop_sequences: int = 32
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
    history_path: Path | None = None
    resume_from: Path | None = None
    auto_resume: bool = False
    crop_config: ImageCropConfig = field(default_factory=ImageCropConfig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="AdamW weight decay. Tiny by default because the model is small.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Linear LR warmup steps before the main schedule.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help=(
            "LR decay after warmup. 'cosine' decays to 0 at the last step; "
            "'constant' holds the peak LR after warmup."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument(
        "--kl-balance-alpha",
        type=float,
        default=0.8,
        help=(
            "KL balancing coefficient. alpha weights KL(sg(q)||p) (train prior), "
            "1-alpha weights KL(q||sg(p)) (regularize posterior). DreamerV2 default 0.8."
        ),
    )
    parser.add_argument(
        "--kl-free-nats",
        type=float,
        default=3.0,
        help=(
            "Per-sample-per-step KL floor in nats. The balanced KL is clamped at this "
            "value before averaging so the term stops pushing once the information "
            "budget is met. DreamerV2 default 3.0."
        ),
    )
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--foreground-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--foreground-mask-floor", type=float, default=0.02)
    parser.add_argument("--foreground-mask-kernel-size", type=int, default=7)
    parser.add_argument("--transition-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--foreground-transition-reconstruction-weight", type=float, default=0.0)
    parser.add_argument(
        "--imagination-reconstruction-weight",
        type=float,
        default=0.0,
        help=(
            "Multiplier for full-frame MSE on decoded imagined prior rollouts. "
            "Set with --imagination-horizon > 0 to close the loop after a "
            "short posterior context and train the decoder on the multi-step "
            "prior feature distribution used at eval time."
        ),
    )
    parser.add_argument(
        "--foreground-imagination-reconstruction-weight",
        type=float,
        default=0.0,
        help=(
            "Multiplier for foreground-masked MSE on decoded imagined prior "
            "rollouts. Recommended for cartpole because full-frame MSE hides "
            "foreground failure behind background pixels."
        ),
    )
    parser.add_argument(
        "--imagination-context-steps",
        type=int,
        default=4,
        help=(
            "Number of posterior warmup steps before closing the loop for "
            "imagination. Only used when --imagination-horizon > 0."
        ),
    )
    parser.add_argument(
        "--imagination-horizon",
        type=int,
        default=0,
        help=(
            "Number of closed-loop prior steps to imagine and decode. The "
            "training sequence must satisfy T >= context_steps + horizon - 1."
        ),
    )
    parser.add_argument(
        "--reward-head",
        action="store_true",
        help=(
            "Add a reward head MLP from RSSM features to scalar reward. Required "
            "before MPC/CEM evaluation. Backward-compatible with checkpoints "
            "trained without it (the new module is initialized fresh)."
        ),
    )
    parser.add_argument(
        "--reward-head-hidden-size",
        type=int,
        default=200,
        help="Hidden size for the two-layer reward MLP. Only used with --reward-head.",
    )
    parser.add_argument(
        "--reward-weight",
        type=float,
        default=0.0,
        help="MSE weight for posterior reward predictions vs dataloader rewards.",
    )
    parser.add_argument(
        "--imagined-reward-weight",
        type=float,
        default=0.0,
        help=(
            "MSE weight for imagined (multi-step prior) reward predictions. Trains "
            "the head over the same trajectory distribution MPC will roll out."
        ),
    )
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
    parser.add_argument(
        "--val-open-loop-every-steps",
        type=int,
        default=0,
        help=(
            "Run a small deterministic open-loop validation every N optimizer "
            "steps and log foreground-masked MSE per horizon into history.jsonl. "
            "0 disables. This surfaces the K1 acceptance metric during training "
            "instead of only at eval time."
        ),
    )
    parser.add_argument(
        "--val-open-loop-warmup",
        type=int,
        default=4,
        help="Number of posterior warmup steps for the in-training open-loop probe.",
    )
    parser.add_argument(
        "--val-open-loop-horizons",
        type=str,
        default="1,5,10,20",
        help=(
            "Comma-separated list of horizons (in steps) to report for the "
            "in-training open-loop probe."
        ),
    )
    parser.add_argument(
        "--val-open-loop-sequences",
        type=int,
        default=32,
        help="Number of validation sequences used for the in-training open-loop probe.",
    )
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Explicit checkpoint path to resume from. Overrides --auto-resume "
            "when set. Use --auto-resume for the typical 'continue from where "
            "I stopped' workflow."
        ),
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Resume from <checkpoint-dir>/latest.pt when it exists; start fresh "
            "otherwise. Safe to leave on for repeat runs of the same command."
        ),
    )
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
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        embedding_size=args.embedding_size,
        kl_weight=args.kl_weight,
        kl_balance_alpha=args.kl_balance_alpha,
        kl_free_nats=args.kl_free_nats,
        reconstruction_weight=args.reconstruction_weight,
        foreground_reconstruction_weight=args.foreground_reconstruction_weight,
        foreground_mask_floor=args.foreground_mask_floor,
        foreground_mask_kernel_size=args.foreground_mask_kernel_size,
        transition_reconstruction_weight=args.transition_reconstruction_weight,
        foreground_transition_reconstruction_weight=(
            args.foreground_transition_reconstruction_weight
        ),
        imagination_reconstruction_weight=args.imagination_reconstruction_weight,
        foreground_imagination_reconstruction_weight=(
            args.foreground_imagination_reconstruction_weight
        ),
        imagination_context_steps=args.imagination_context_steps,
        imagination_horizon=args.imagination_horizon,
        reward_head=args.reward_head,
        reward_head_hidden_size=args.reward_head_hidden_size,
        reward_weight=args.reward_weight,
        imagined_reward_weight=args.imagined_reward_weight,
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
        val_open_loop_every_steps=args.val_open_loop_every_steps,
        val_open_loop_warmup=args.val_open_loop_warmup,
        val_open_loop_horizons=parse_horizons_arg(args.val_open_loop_horizons),
        val_open_loop_sequences=args.val_open_loop_sequences,
        max_train_episodes=args.max_train_episodes,
        max_val_episodes=args.max_val_episodes,
        history_path=args.history_path,
        resume_from=args.resume_from,
        auto_resume=args.auto_resume,
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
    if config.imagination_reconstruction_weight < 0.0:
        raise ValueError("imagination_reconstruction_weight must be >= 0")
    if config.foreground_imagination_reconstruction_weight < 0.0:
        raise ValueError("foreground_imagination_reconstruction_weight must be >= 0")
    if config.imagination_horizon < 0:
        raise ValueError("imagination_horizon must be >= 0")
    if config.imagination_context_steps < 1:
        raise ValueError("imagination_context_steps must be >= 1")
    imagination_enabled = (
        config.imagination_reconstruction_weight > 0.0
        or config.foreground_imagination_reconstruction_weight > 0.0
        or config.imagined_reward_weight > 0.0
    )
    if imagination_enabled and config.imagination_horizon < 1:
        raise ValueError(
            "imagination_horizon must be >= 1 when any imagination/imagined-reward "
            "weight > 0",
        )
    if config.reward_weight > 0.0 and not config.reward_head:
        raise ValueError(
            "reward_weight > 0 requires --reward-head so the model has a head to train",
        )
    if config.imagined_reward_weight > 0.0 and not config.reward_head:
        raise ValueError(
            "imagined_reward_weight > 0 requires --reward-head",
        )
    if imagination_enabled and config.sequence_length < (
        config.imagination_context_steps + config.imagination_horizon - 1
    ):
        raise ValueError(
            "sequence_length must be >= imagination_context_steps + "
            "imagination_horizon - 1 when imagination is enabled; "
            f"got sequence_length={config.sequence_length}, "
            f"context_steps={config.imagination_context_steps}, "
            f"horizon={config.imagination_horizon}",
        )
    if config.dynamic_reconstruction_weight < 0.0:
        raise ValueError("dynamic_reconstruction_weight must be >= 0")
    if not 0.0 <= config.dynamic_mask_floor <= 1.0:
        raise ValueError("dynamic_mask_floor must satisfy 0 <= floor <= 1")

    set_seed(config.seed)
    device = resolve_device(config.device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = resolve_history_path(config)
    resume_from = resolve_resume_path(config, console)

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
        reward_head=config.reward_head,
        reward_head_hidden_size=config.reward_head_hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(config.epochs * len(train_loader), 1)
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        schedule=config.lr_schedule,
    )
    start_epoch = 1
    global_step = 0

    if resume_from is not None:
        checkpoint_epoch, global_step = load_checkpoint(
            resume_from,
            model,
            optimizer,
            device,
            scheduler=scheduler,
            expected_config=config,
        )
        start_epoch = checkpoint_epoch + 1
        console.print(
            f"[bold yellow]resumed[/bold yellow] from {resume_from} "
            f"epoch={checkpoint_epoch} global_step={global_step:07d} "
            f"will run epochs {start_epoch}..{config.epochs}",
        )
        if start_epoch > config.epochs:
            console.print(
                "[bold red]nothing to do[/bold red]: checkpoint is at or past "
                f"the requested final epoch {config.epochs}; raise --epochs to extend.",
            )
            return
    prepare_history_file(history_path, resume=resume_from is not None)

    val_probe_batch: dict[str, torch.Tensor] | None = None
    if config.val_open_loop_every_steps > 0:
        if val_loader is None:
            raise ValueError(
                "val_open_loop_every_steps > 0 requires val_fraction > 0 so there "
                "is a validation loader to probe",
            )
        max_horizon = max(config.val_open_loop_horizons)
        required = config.val_open_loop_warmup + max_horizon - 1
        if config.sequence_length < required:
            raise ValueError(
                "sequence_length must be >= val_open_loop_warmup + max(horizon) - 1 "
                "when the open-loop probe is enabled; "
                f"got sequence_length={config.sequence_length}, required={required}",
            )
        val_probe_batch = collect_val_probe_batch(
            loader=val_loader,
            num_sequences=config.val_open_loop_sequences,
            device=device,
        )
        if val_probe_batch is None:
            raise ValueError("validation loader is empty; cannot run open-loop probe")

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
            scheduler=scheduler,
            device=device,
            val_probe_batch=val_probe_batch,
            val_open_loop_every_steps=config.val_open_loop_every_steps,
            val_open_loop_warmup=config.val_open_loop_warmup,
            val_open_loop_horizons=config.val_open_loop_horizons,
            kl_weight=config.kl_weight,
            kl_balance_alpha=config.kl_balance_alpha,
            kl_free_nats=config.kl_free_nats,
            imagination_reconstruction_weight=config.imagination_reconstruction_weight,
            foreground_imagination_reconstruction_weight=(
                config.foreground_imagination_reconstruction_weight
            ),
            imagination_context_steps=config.imagination_context_steps,
            imagination_horizon=config.imagination_horizon,
            reward_weight=config.reward_weight,
            imagined_reward_weight=config.imagined_reward_weight,
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
                kl_balance_alpha=config.kl_balance_alpha,
                kl_free_nats=config.kl_free_nats,
                imagination_reconstruction_weight=config.imagination_reconstruction_weight,
                foreground_imagination_reconstruction_weight=(
                    config.foreground_imagination_reconstruction_weight
                ),
                imagination_context_steps=config.imagination_context_steps,
                imagination_horizon=config.imagination_horizon,
                reward_weight=config.reward_weight,
                imagined_reward_weight=config.imagined_reward_weight,
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
            scheduler=scheduler,
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
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    val_probe_batch: dict[str, torch.Tensor] | None,
    val_open_loop_every_steps: int,
    val_open_loop_warmup: int,
    val_open_loop_horizons: tuple[int, ...],
    kl_weight: float,
    kl_balance_alpha: float,
    kl_free_nats: float,
    imagination_reconstruction_weight: float,
    foreground_imagination_reconstruction_weight: float,
    imagination_context_steps: int,
    imagination_horizon: int,
    reward_weight: float,
    imagined_reward_weight: float,
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

    reward_enabled = reward_weight > 0.0 or imagined_reward_weight > 0.0
    imagination_enabled = (
        imagination_reconstruction_weight > 0.0
        or foreground_imagination_reconstruction_weight > 0.0
        or imagined_reward_weight > 0.0
    )
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device) if reward_enabled else None
        needs_transition_outputs = (
            latent_consistency_weight > 0.0
            or transition_reconstruction_weight > 0.0
            or foreground_transition_reconstruction_weight > 0.0
        )
        needs_next_observations = (
            needs_transition_outputs
            or dynamic_reconstruction_weight > 0.0
            or imagination_enabled
        )
        next_observations = (
            batch["next_observations"].to(device)
            if needs_next_observations
            else None
        )
        # observations: [B, T, 3, 84, 84], actions: [B, T, A].
        # When enabled, next_observations[:, t] is the target for
        # obs_t --action_t--> obs_{t+1}. Imagination decodes the prior feature
        # reached after applying actions[K_ctx-1 .. K_ctx-1+K_imag-1] and
        # targets next_observations[:, K_ctx-1 .. K_ctx-1+K_imag-1].
        outputs = model(
            observations=observations,
            actions=actions,
            next_observations=next_observations if needs_transition_outputs else None,
            imagination_context_steps=(
                imagination_context_steps if imagination_enabled else 0
            ),
            imagination_horizon=imagination_horizon if imagination_enabled else 0,
        )
        losses = compute_world_model_loss(
            outputs,
            observations,
            kl_weight=kl_weight,
            kl_balance_alpha=kl_balance_alpha,
            kl_free_nats=kl_free_nats,
            reconstruction_weight=reconstruction_weight,
            foreground_reconstruction_weight=foreground_reconstruction_weight,
            foreground_mask_floor=foreground_mask_floor,
            foreground_mask_kernel_size=foreground_mask_kernel_size,
            transition_reconstruction_weight=transition_reconstruction_weight,
            foreground_transition_reconstruction_weight=foreground_transition_reconstruction_weight,
            imagination_reconstruction_weight=imagination_reconstruction_weight,
            foreground_imagination_reconstruction_weight=(
                foreground_imagination_reconstruction_weight
            ),
            dynamic_reconstruction_weight=dynamic_reconstruction_weight,
            dynamic_mask_floor=dynamic_mask_floor,
            latent_consistency_weight=latent_consistency_weight,
            reward_weight=reward_weight,
            imagined_reward_weight=imagined_reward_weight,
            rewards=rewards,
            next_observations=next_observations,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        # ``clip_grad_norm_`` returns the same total norm we'd report otherwise,
        # so calling ``compute_grad_norm`` first would be a wasted host-device
        # sync per step. Only fall back to the standalone helper when clipping
        # is disabled.
        if grad_clip > 0.0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        else:
            grad_norm = compute_grad_norm(model)
        optimizer.step()
        scheduler.step()

        batch_metrics = tensor_metrics_to_float(losses)
        batch_metrics["grad_norm"] = grad_norm
        batch_metrics["learning_rate"] = float(scheduler.get_last_lr()[0])
        update_totals(totals, batch_metrics)
        num_batches += 1
        global_step += 1
        if global_step % log_every_steps == 0:
            emit_metrics("train", epoch, batch_metrics, global_step, history_path, console)
        if (
            val_probe_batch is not None
            and val_open_loop_every_steps > 0
            and global_step % val_open_loop_every_steps == 0
        ):
            probe_metrics = run_val_open_loop_probe(
                model=model,
                batch=val_probe_batch,
                warmup=val_open_loop_warmup,
                horizons=val_open_loop_horizons,
                foreground_mask_floor=foreground_mask_floor,
                foreground_mask_kernel_size=foreground_mask_kernel_size,
            )
            emit_metrics(
                "val_open_loop", epoch, probe_metrics, global_step, history_path, console,
            )

    return average_totals(totals, num_batches), global_step


@torch.no_grad()
def evaluate(
    model: VisualWorldModel,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    kl_balance_alpha: float,
    kl_free_nats: float,
    imagination_reconstruction_weight: float,
    foreground_imagination_reconstruction_weight: float,
    imagination_context_steps: int,
    imagination_horizon: int,
    reward_weight: float,
    imagined_reward_weight: float,
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
    reward_enabled = reward_weight > 0.0 or imagined_reward_weight > 0.0
    imagination_enabled = (
        imagination_reconstruction_weight > 0.0
        or foreground_imagination_reconstruction_weight > 0.0
        or imagined_reward_weight > 0.0
    )
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device) if reward_enabled else None
        needs_transition_outputs = (
            latent_consistency_weight > 0.0
            or transition_reconstruction_weight > 0.0
            or foreground_transition_reconstruction_weight > 0.0
        )
        needs_next_observations = (
            needs_transition_outputs
            or dynamic_reconstruction_weight > 0.0
            or imagination_reconstruction_weight > 0.0
            or foreground_imagination_reconstruction_weight > 0.0
        )
        next_observations = (
            batch["next_observations"].to(device)
            if needs_next_observations
            else None
        )
        outputs = model(
            observations=observations,
            actions=actions,
            next_observations=next_observations if needs_transition_outputs else None,
            imagination_context_steps=(
                imagination_context_steps if imagination_enabled else 0
            ),
            imagination_horizon=imagination_horizon if imagination_enabled else 0,
        )
        losses = compute_world_model_loss(
            outputs,
            observations,
            kl_weight=kl_weight,
            kl_balance_alpha=kl_balance_alpha,
            kl_free_nats=kl_free_nats,
            reconstruction_weight=reconstruction_weight,
            foreground_reconstruction_weight=foreground_reconstruction_weight,
            foreground_mask_floor=foreground_mask_floor,
            foreground_mask_kernel_size=foreground_mask_kernel_size,
            transition_reconstruction_weight=transition_reconstruction_weight,
            foreground_transition_reconstruction_weight=foreground_transition_reconstruction_weight,
            imagination_reconstruction_weight=imagination_reconstruction_weight,
            foreground_imagination_reconstruction_weight=(
                foreground_imagination_reconstruction_weight
            ),
            dynamic_reconstruction_weight=dynamic_reconstruction_weight,
            dynamic_mask_floor=dynamic_mask_floor,
            latent_consistency_weight=latent_consistency_weight,
            reward_weight=reward_weight,
            imagined_reward_weight=imagined_reward_weight,
            rewards=rewards,
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
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: TrainConfig,
    metrics: dict[str, float],
    global_step: int,
    history_path: Path,
) -> Path:
    """Save model/optimizer/scheduler/RNG state and config for later resume/evaluation."""

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "config": serializable_config(config),
        "metrics": metrics,
        "history_path": str(history_path),
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    return epoch_path


# Keys whose values must match between the live config and the checkpoint
# config or the model architecture / data layout would be inconsistent on
# resume. Optimizer-only knobs (LR, weight decay, schedule, weights) are
# allowed to change so users can tweak them across resumes.
RESUME_GUARDED_CONFIG_KEYS: tuple[str, ...] = (
    "hidden_size",
    "latent_size",
    "embedding_size",
    "sequence_length",
    "imagination_context_steps",
    "imagination_horizon",
    "crop_config",
)


def load_checkpoint(
    path: Path,
    model: VisualWorldModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    expected_config: TrainConfig | None = None,
) -> tuple[int, int]:
    """Load model/optimizer/scheduler/RNG state and return ``(epoch, global_step)``."""

    # weights_only=False because our checkpoints intentionally store NumPy RNG
    # state, scheduler dicts, and the config dataclass alongside the model
    # weights. Safe here because we always load our own checkpoint files.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if expected_config is not None and "config" in checkpoint:
        validate_resume_config(checkpoint["config"], expected_config)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    # Modules added in later phases are allowed to be missing from older
    # checkpoints; we just initialize them from defaults and warn. Add new
    # prefixes here as new optional submodules land.
    OPTIONAL_NEW_MODULE_PREFIXES = ("latent_predictor.", "reward_head.")
    allowed_missing = [
        name
        for name in missing
        if any(name.startswith(prefix) for prefix in OPTIONAL_NEW_MODULE_PREFIXES)
    ]
    disallowed_missing = [
        name
        for name in missing
        if not any(name.startswith(prefix) for prefix in OPTIONAL_NEW_MODULE_PREFIXES)
    ]
    unexpected_disallowed = [
        name
        for name in unexpected
        if not any(name.startswith(prefix) for prefix in OPTIONAL_NEW_MODULE_PREFIXES)
    ]
    if disallowed_missing or unexpected_disallowed:
        raise RuntimeError(
            "checkpoint model state mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected_disallowed}",
        )
    if allowed_missing:
        print(
            f"Warning: checkpoint is missing optional parameters {allowed_missing}; "
            "initialized them from the current model defaults.",
        )
    if [name for name in unexpected if name not in unexpected_disallowed]:
        print(
            "Warning: checkpoint contains optional parameters that the current "
            "model does not use; ignoring them. To use them, enable the "
            "corresponding flag (e.g. --reward-head).",
        )
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except ValueError:
        if not allowed_missing:
            print(
                "Warning: optimizer state in checkpoint is incompatible with the "
                "current optimizer (likely Adam → AdamW); using fresh optimizer state.",
            )
        else:
            print(
                "Warning: checkpoint optimizer state is incompatible with the "
                "new optional submodules; using a fresh optimizer state.",
            )
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]), int(checkpoint.get("global_step", 0))


def validate_resume_config(saved: dict[str, Any], expected: TrainConfig) -> None:
    """Raise if architecture/data-layout fields differ between checkpoint and current config.

    Optimizer / loss-weight knobs are intentionally not guarded so a resume
    can change them (e.g. lower the KL weight after a few epochs).
    """

    expected_dict = serializable_config(expected)
    mismatches: list[str] = []
    for key in RESUME_GUARDED_CONFIG_KEYS:
        saved_value = saved.get(key)
        expected_value = expected_dict.get(key)
        if saved_value != expected_value:
            mismatches.append(f"{key}: checkpoint={saved_value!r} current={expected_value!r}")
    if mismatches:
        raise RuntimeError(
            "cannot resume — checkpoint architecture/data layout does not match "
            "current config:\n  " + "\n  ".join(mismatches),
        )


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python, NumPy, and Torch RNG state so resume is deterministic."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by ``capture_rng_state``.

    ``torch.load(map_location=device)`` relocates every tensor in the
    checkpoint — including the RNG state ByteTensors — onto the training
    device. ``torch.set_rng_state`` and its CUDA/MPS counterparts require
    their inputs on the native RNG device (CPU for the default generator)
    and in ``uint8`` dtype, so we normalize every tensor before restoring.
    """

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(_as_cpu_byte_tensor(state["torch_cpu"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [_as_cpu_byte_tensor(cuda_state) for cuda_state in state["torch_cuda"]]
        )
    if "torch_mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(_as_cpu_byte_tensor(state["torch_mps"]))


def _as_cpu_byte_tensor(tensor: Any) -> torch.Tensor:
    """Coerce an RNG state tensor back onto CPU with ``uint8`` dtype."""

    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().to(device="cpu", dtype=torch.uint8)


def resolve_resume_path(config: TrainConfig, console: Console) -> Path | None:
    """Pick the checkpoint to resume from, honoring --resume-from over --auto-resume."""

    if config.resume_from is not None:
        if not config.resume_from.exists():
            raise FileNotFoundError(
                f"--resume-from path does not exist: {config.resume_from}",
            )
        return config.resume_from
    if config.auto_resume:
        latest = config.checkpoint_dir / "latest.pt"
        if latest.exists():
            console.print(
                f"[bold yellow]auto-resume[/bold yellow]: found {latest}",
            )
            return latest
        console.print(
            "[dim]auto-resume requested but no latest.pt in "
            f"{config.checkpoint_dir} — starting fresh[/dim]",
        )
    return None


def parse_horizons_arg(arg: str) -> tuple[int, ...]:
    """Parse ``"1,5,10,20"`` into a sorted tuple of positive horizons."""

    tokens = [t.strip() for t in arg.split(",") if t.strip()]
    horizons = tuple(sorted({int(t) for t in tokens}))
    if any(h < 1 for h in horizons):
        raise ValueError("val_open_loop_horizons must be positive integers")
    return horizons


@torch.no_grad()
def run_val_open_loop_probe(
    model: VisualWorldModel,
    batch: dict[str, torch.Tensor],
    warmup: int,
    horizons: tuple[int, ...],
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
) -> dict[str, float]:
    """Decode posterior-context + imagined prior rollout and report per-horizon MSE.

    The probe mirrors what ``eval-rssm`` reports at checkpoint time but is
    cheap enough to run every few hundred optimizer steps so the K1 metric is
    observable live in ``history.jsonl``.
    """

    was_training = model.training
    model.eval()
    observations = batch["observations"]
    actions = batch["actions"]
    next_observations = batch["next_observations"]
    sequence_length = observations.shape[1]
    max_horizon = max(horizons)
    required = warmup + max_horizon - 1
    if sequence_length < required:
        raise ValueError(
            "val_open_loop probe requires sequence_length >= warmup + max_horizon - 1; "
            f"got T={sequence_length}, warmup={warmup}, max_horizon={max_horizon}",
        )

    outputs = model(
        observations=observations,
        actions=actions,
        imagination_context_steps=warmup,
        imagination_horizon=max_horizon,
    )
    imagined = outputs.imagined_reconstructions
    assert imagined is not None and outputs.imagined_target_start is not None
    target_start = outputs.imagined_target_start
    target = next_observations[:, target_start : target_start + max_horizon]

    foreground_mask = foreground_reconstruction_mask(
        observations=target,
        floor=foreground_mask_floor,
        kernel_size=foreground_mask_kernel_size,
    )

    metrics: dict[str, float] = {}
    for horizon in horizons:
        squared_error = (imagined[:, horizon - 1] - target[:, horizon - 1]).pow(2)
        mask = foreground_mask[:, horizon - 1]
        metrics[f"val_open_loop_mse_h{horizon}"] = float(squared_error.mean().cpu())
        metrics[f"val_open_loop_fg_mse_h{horizon}"] = float(
            (squared_error * mask).mean().cpu(),
        )

    if was_training:
        model.train()
    return metrics


def collect_val_probe_batch(
    loader: DataLoader,
    num_sequences: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """Build a fixed-size contiguous batch from the validation loader.

    Returns ``None`` when the loader is empty. Sequences are accumulated from
    the start of the loader until the target count is reached and then
    truncated to ``num_sequences``. This keeps the probe deterministic across
    steps because ``val_loader`` is built with ``shuffle=False`` elsewhere.
    """

    collected: list[dict[str, torch.Tensor]] = []
    total = 0
    for batch in loader:
        collected.append(batch)
        total += batch["observations"].shape[0]
        if total >= num_sequences:
            break
    if not collected:
        return None
    merged = {
        key: torch.cat([c[key] for c in collected], dim=0)[:num_sequences].to(device)
        for key in collected[0]
    }
    return merged


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear-warmup + cosine/constant LR scheduler over ``total_steps``.

    ``step(0)`` returns a lr_scale of ``1 / max(1, warmup_steps)`` (the first
    call to ``scheduler.step`` advances to ``1``). With ``warmup_steps=0`` the
    warmup segment is skipped and the schedule starts at the peak LR.
    """

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    # LambdaLR's constructor advances ``last_epoch`` from -1 to 0, so the
    # lambda is first invoked with step=0. Peak LR is reached at
    # step == warmup_steps - 1 (the ``warmup_steps``-th invocation). Cosine
    # decay should reach 0 at step == total_steps - 1.
    effective_warmup = max(warmup_steps, 0)
    decay_steps = max(total_steps - effective_warmup, 1)

    if schedule == "constant":
        def lr_lambda(step: int) -> float:
            if effective_warmup <= 1:
                return 1.0
            return min(1.0, float(step + 1) / float(effective_warmup))
    elif schedule == "cosine":
        import math

        def lr_lambda(step: int) -> float:
            if effective_warmup > 1 and step < effective_warmup - 1:
                return float(step + 1) / float(effective_warmup)
            progress = min(
                1.0,
                float(step - (effective_warmup - 1)) / float(decay_steps),
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"unknown lr_schedule {schedule!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    """Compute the global gradient L2 norm with a single host-device sync.

    The previous implementation called ``.item()`` once per parameter, which
    on MPS forces a pipeline flush per call — dozens of stalls per training
    step for a model with ~80 grad-bearing parameters. We now compute the
    per-parameter norms via a fused kernel (``torch._foreach_norm`` when
    available, falling back to ``torch.stack``) and only call ``.item()``
    once at the very end.

    For convenience, prefer ``torch.nn.utils.clip_grad_norm_`` whenever
    ``grad_clip > 0`` — its return value is the same total norm and it
    is computed in the same fused kernel as the clip itself, so there is
    no reason to call this function separately in that case.
    """

    grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not grads:
        return 0.0
    foreach_norm = getattr(torch, "_foreach_norm", None)
    if foreach_norm is not None:
        per_param_norms = foreach_norm(grads, 2.0)
    else:
        per_param_norms = [g.detach().norm(2) for g in grads]
    total = torch.linalg.vector_norm(torch.stack(per_param_norms))
    return float(total.item())


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
