"""Offline training for the mechanics-structured visual world model.

Reuses the Phase-0 infrastructure (LR schedule, RNG capture, history JSONL,
foreground mask, open-loop probe) from ``train.train_rssm``; only the model
construction, loss, and a few config knobs are mechanics-specific.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import EpisodeSequenceDataset, ImageCropConfig
from data.image_transforms import add_crop_args, crop_config_from_args
from models.losses import foreground_reconstruction_mask
from models.mechanics import (
    MechanicsWorldModel,
    compute_mechanics_world_model_loss,
)
from train.train_rssm import (
    average_totals,
    build_lr_scheduler,
    capture_rng_state,
    collect_val_probe_batch,
    compute_grad_norm,
    count_parameters,
    emit_metrics,
    parse_horizons_arg,
    prepare_history_file,
    resolve_device,
    restore_rng_state,
    set_seed,
    tensor_metrics_to_float,
    update_totals,
)


@dataclass(frozen=True)
class MechanicsTrainConfig:
    """Configuration for offline mechanics world-model training."""

    dataset_dir: Path
    checkpoint_dir: Path
    sequence_length: int = 32
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"
    epochs: int = 10
    q_dim: int = 2
    nuisance_dim: int = 32
    embedding_size: int = 256
    encoder_head_hidden_size: int = 128
    mass_matrix_hidden_size: int = 64
    potential_hidden_size: int = 64
    dissipation_hidden_size: int = 64
    actuation_hidden_size: int = 64
    dynamics_hidden_layers: int = 2
    with_dissipation: bool = True
    dt: float = 0.01
    n_substeps: int = 1
    nuisance_alpha_init: float = 0.95
    mech_min_std: float = 0.05
    nuisance_min_std: float = 0.1
    kl_weight: float = 1.0
    kl_balance_alpha: float = 0.8
    kl_free_nats_mech: float = 0.5
    kl_free_nats_nuisance: float = 3.0
    reconstruction_weight: float = 1.0
    foreground_reconstruction_weight: float = 1.0
    foreground_mask_floor: float = 0.02
    foreground_mask_kernel_size: int = 7
    imagination_reconstruction_weight: float = 0.0
    foreground_imagination_reconstruction_weight: float = 2.0
    imagination_context_steps: int = 4
    imagination_horizon: int = 12
    reward_head: bool = False
    reward_head_hidden_size: int = 200
    reward_weight: float = 0.0
    imagined_reward_weight: float = 0.0
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


# Architecture/data-layout fields whose values must be preserved on resume.
RESUME_GUARDED_CONFIG_KEYS: tuple[str, ...] = (
    "q_dim",
    "nuisance_dim",
    "embedding_size",
    "sequence_length",
    "imagination_context_steps",
    "imagination_horizon",
    "with_dissipation",
    "dt",
    "n_substeps",
    "crop_config",
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the mechanics training CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--q-dim", type=int, default=2)
    parser.add_argument("--nuisance-dim", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--encoder-head-hidden-size", type=int, default=128)
    parser.add_argument("--mass-matrix-hidden-size", type=int, default=64)
    parser.add_argument("--potential-hidden-size", type=int, default=64)
    parser.add_argument("--dissipation-hidden-size", type=int, default=64)
    parser.add_argument("--actuation-hidden-size", type=int, default=64)
    parser.add_argument("--dynamics-hidden-layers", type=int, default=2)
    parser.add_argument(
        "--no-dissipation",
        dest="with_dissipation",
        action="store_false",
        default=True,
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-substeps", type=int, default=1)
    parser.add_argument("--nuisance-alpha-init", type=float, default=0.95)
    parser.add_argument("--mech-min-std", type=float, default=0.05)
    parser.add_argument("--nuisance-min-std", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--kl-balance-alpha", type=float, default=0.8)
    parser.add_argument("--kl-free-nats-mech", type=float, default=0.5)
    parser.add_argument("--kl-free-nats-nuisance", type=float, default=3.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--foreground-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--foreground-mask-floor", type=float, default=0.02)
    parser.add_argument("--foreground-mask-kernel-size", type=int, default=7)
    parser.add_argument("--imagination-reconstruction-weight", type=float, default=0.0)
    parser.add_argument(
        "--foreground-imagination-reconstruction-weight", type=float, default=2.0,
    )
    parser.add_argument("--imagination-context-steps", type=int, default=4)
    parser.add_argument("--imagination-horizon", type=int, default=12)
    parser.add_argument("--reward-head", action="store_true")
    parser.add_argument("--reward-head-hidden-size", type=int, default=200)
    parser.add_argument("--reward-weight", type=float, default=0.0)
    parser.add_argument("--imagined-reward-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--val-open-loop-every-steps", type=int, default=0)
    parser.add_argument("--val-open-loop-warmup", type=int, default=4)
    parser.add_argument(
        "--val-open-loop-horizons",
        type=parse_horizons_arg,
        default="1,5,10,20",
        help="comma-separated positive integers; default 1,5,10,20",
    )
    parser.add_argument("--val-open-loop-sequences", type=int, default=32)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    add_crop_args(parser, default_mode="none")
    return parser


def config_from_args(args: argparse.Namespace) -> MechanicsTrainConfig:
    """Build the trainer config from parsed CLI args."""

    return MechanicsTrainConfig(
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        epochs=args.epochs,
        q_dim=args.q_dim,
        nuisance_dim=args.nuisance_dim,
        embedding_size=args.embedding_size,
        encoder_head_hidden_size=args.encoder_head_hidden_size,
        mass_matrix_hidden_size=args.mass_matrix_hidden_size,
        potential_hidden_size=args.potential_hidden_size,
        dissipation_hidden_size=args.dissipation_hidden_size,
        actuation_hidden_size=args.actuation_hidden_size,
        dynamics_hidden_layers=args.dynamics_hidden_layers,
        with_dissipation=args.with_dissipation,
        dt=args.dt,
        n_substeps=args.n_substeps,
        nuisance_alpha_init=args.nuisance_alpha_init,
        mech_min_std=args.mech_min_std,
        nuisance_min_std=args.nuisance_min_std,
        kl_weight=args.kl_weight,
        kl_balance_alpha=args.kl_balance_alpha,
        kl_free_nats_mech=args.kl_free_nats_mech,
        kl_free_nats_nuisance=args.kl_free_nats_nuisance,
        reconstruction_weight=args.reconstruction_weight,
        foreground_reconstruction_weight=args.foreground_reconstruction_weight,
        foreground_mask_floor=args.foreground_mask_floor,
        foreground_mask_kernel_size=args.foreground_mask_kernel_size,
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
        device=args.device,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        log_every_steps=args.log_every_steps,
        val_open_loop_every_steps=args.val_open_loop_every_steps,
        val_open_loop_warmup=args.val_open_loop_warmup,
        val_open_loop_horizons=args.val_open_loop_horizons,
        val_open_loop_sequences=args.val_open_loop_sequences,
        max_train_episodes=args.max_train_episodes,
        max_val_episodes=args.max_val_episodes,
        history_path=args.history_path,
        resume_from=args.resume_from,
        auto_resume=args.auto_resume,
        crop_config=crop_config_from_args(args) or ImageCropConfig(),
    )


def train(config: MechanicsTrainConfig, console: Console | None = None) -> None:
    """Run offline mechanics world-model training and save checkpoints."""

    console = console or Console()
    validate_config(config)

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
    model = MechanicsWorldModel(
        action_dim=action_dim,
        q_dim=config.q_dim,
        nuisance_dim=config.nuisance_dim,
        embedding_size=config.embedding_size,
        encoder_head_hidden_size=config.encoder_head_hidden_size,
        mass_matrix_hidden_size=config.mass_matrix_hidden_size,
        potential_hidden_size=config.potential_hidden_size,
        dissipation_hidden_size=config.dissipation_hidden_size,
        actuation_hidden_size=config.actuation_hidden_size,
        dynamics_hidden_layers=config.dynamics_hidden_layers,
        image_size=image_size,
        dt=config.dt,
        n_substeps=config.n_substeps,
        with_dissipation=config.with_dissipation,
        mech_min_std=config.mech_min_std,
        nuisance_min_std=config.nuisance_min_std,
        nuisance_alpha_init=config.nuisance_alpha_init,
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
                "val_open_loop_every_steps > 0 requires val_fraction > 0",
            )
        max_horizon = max(config.val_open_loop_horizons)
        required = config.val_open_loop_warmup + max_horizon - 1
        if config.sequence_length < required:
            raise ValueError(
                "sequence_length must be >= val_open_loop_warmup + max(horizon) - 1; "
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
        config=config,
    )

    for epoch in range(start_epoch, config.epochs + 1):
        console.rule(f"[bold cyan]mechanics epoch {epoch:04d}[/]")
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            val_probe_batch=val_probe_batch,
            epoch=epoch,
            start_global_step=global_step,
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
                config=config,
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
    model: MechanicsWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    config: MechanicsTrainConfig,
    val_probe_batch: dict[str, torch.Tensor] | None,
    epoch: int,
    start_global_step: int,
    history_path: Path,
    console: Console,
) -> tuple[dict[str, float], int]:
    """Run one mechanics training epoch."""

    model.train()
    totals: dict[str, float] = {}
    num_batches = 0
    global_step = start_global_step

    reward_enabled = config.reward_weight > 0.0 or config.imagined_reward_weight > 0.0
    imagination_enabled = (
        config.imagination_reconstruction_weight > 0.0
        or config.foreground_imagination_reconstruction_weight > 0.0
        or config.imagined_reward_weight > 0.0
    )
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device) if reward_enabled else None
        next_observations = (
            batch["next_observations"].to(device) if imagination_enabled else None
        )
        outputs = model(
            observations=observations,
            actions=actions,
            imagination_context_steps=(
                config.imagination_context_steps if imagination_enabled else 0
            ),
            imagination_horizon=config.imagination_horizon if imagination_enabled else 0,
        )
        losses = compute_mechanics_world_model_loss(
            outputs=outputs,
            observations=observations,
            next_observations=next_observations,
            rewards=rewards,
            reconstruction_weight=config.reconstruction_weight,
            foreground_reconstruction_weight=config.foreground_reconstruction_weight,
            foreground_mask_floor=config.foreground_mask_floor,
            foreground_mask_kernel_size=config.foreground_mask_kernel_size,
            imagination_reconstruction_weight=config.imagination_reconstruction_weight,
            foreground_imagination_reconstruction_weight=(
                config.foreground_imagination_reconstruction_weight
            ),
            kl_weight=config.kl_weight,
            kl_balance_alpha=config.kl_balance_alpha,
            kl_free_nats_mech=config.kl_free_nats_mech,
            kl_free_nats_nuisance=config.kl_free_nats_nuisance,
            reward_weight=config.reward_weight,
            imagined_reward_weight=config.imagined_reward_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        # ``clip_grad_norm_`` already returns the total norm; calling
        # ``compute_grad_norm`` first would be a wasted host-device sync per
        # step. Only use the standalone helper when clipping is disabled.
        if config.grad_clip > 0.0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        else:
            grad_norm = compute_grad_norm(model)

        loss_value = float(losses["total_loss"].detach())
        step_skipped = not (math.isfinite(loss_value) and math.isfinite(grad_norm))
        if step_skipped:
            # A non-finite loss or gradient would poison AdamW's running
            # moments past recovery (observed in the mechanics-phase2 /
            # -v2 retries at step ~300, where a single bad batch
            # eventually drove the encoder into a constant-output basin).
            # Zero the accumulated grads and skip the weight update;
            # keep the LR schedule advancing so wall-clock ordering of the
            # warmup/cosine schedule is unchanged by rare skips.
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
        scheduler.step()

        batch_metrics = tensor_metrics_to_float(losses)
        batch_metrics["grad_norm"] = grad_norm
        batch_metrics["learning_rate"] = float(scheduler.get_last_lr()[0])
        batch_metrics["nuisance_alpha"] = float(model.transition.nuisance_alpha.detach().cpu())
        batch_metrics["step_skipped"] = float(step_skipped)
        update_totals(totals, batch_metrics)
        num_batches += 1
        global_step += 1
        if global_step % config.log_every_steps == 0:
            emit_metrics("train", epoch, batch_metrics, global_step, history_path, console)
        if (
            val_probe_batch is not None
            and config.val_open_loop_every_steps > 0
            and global_step % config.val_open_loop_every_steps == 0
        ):
            probe_metrics = run_val_open_loop_probe(
                model=model,
                batch=val_probe_batch,
                warmup=config.val_open_loop_warmup,
                horizons=config.val_open_loop_horizons,
                foreground_mask_floor=config.foreground_mask_floor,
                foreground_mask_kernel_size=config.foreground_mask_kernel_size,
            )
            emit_metrics(
                "val_open_loop", epoch, probe_metrics, global_step, history_path, console,
            )

    return average_totals(totals, num_batches), global_step


@torch.no_grad()
def evaluate(
    model: MechanicsWorldModel,
    loader: DataLoader,
    device: torch.device,
    config: MechanicsTrainConfig,
) -> dict[str, float]:
    """Evaluate the mechanics world model without optimizer updates."""

    model.eval()
    totals: dict[str, float] = {}
    num_batches = 0
    reward_enabled = config.reward_weight > 0.0 or config.imagined_reward_weight > 0.0
    imagination_enabled = (
        config.imagination_reconstruction_weight > 0.0
        or config.foreground_imagination_reconstruction_weight > 0.0
        or config.imagined_reward_weight > 0.0
    )
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device) if reward_enabled else None
        next_observations = (
            batch["next_observations"].to(device) if imagination_enabled else None
        )
        outputs = model(
            observations=observations,
            actions=actions,
            imagination_context_steps=(
                config.imagination_context_steps if imagination_enabled else 0
            ),
            imagination_horizon=config.imagination_horizon if imagination_enabled else 0,
        )
        losses = compute_mechanics_world_model_loss(
            outputs=outputs,
            observations=observations,
            next_observations=next_observations,
            rewards=rewards,
            reconstruction_weight=config.reconstruction_weight,
            foreground_reconstruction_weight=config.foreground_reconstruction_weight,
            foreground_mask_floor=config.foreground_mask_floor,
            foreground_mask_kernel_size=config.foreground_mask_kernel_size,
            imagination_reconstruction_weight=config.imagination_reconstruction_weight,
            foreground_imagination_reconstruction_weight=(
                config.foreground_imagination_reconstruction_weight
            ),
            kl_weight=config.kl_weight,
            kl_balance_alpha=config.kl_balance_alpha,
            kl_free_nats_mech=config.kl_free_nats_mech,
            kl_free_nats_nuisance=config.kl_free_nats_nuisance,
            reward_weight=config.reward_weight,
            imagined_reward_weight=config.imagined_reward_weight,
        )
        update_totals(totals, tensor_metrics_to_float(losses))
        num_batches += 1
    return average_totals(totals, num_batches)


@torch.no_grad()
def run_val_open_loop_probe(
    model: MechanicsWorldModel,
    batch: dict[str, torch.Tensor],
    warmup: int,
    horizons: tuple[int, ...],
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
) -> dict[str, float]:
    """Mechanics-model variant of the live K1 probe.

    Mirrors the contract of ``train_rssm.run_val_open_loop_probe`` so the
    ``history.jsonl`` schema is comparable between B1 (RSSM) and B_main
    (mechanics) — same key names, same horizon-major indexing, same
    foreground mask construction.
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

    # Re-enable grad for the inner JVP/autograd calls in forward_acceleration.
    with torch.enable_grad():
        outputs = model(
            observations=observations,
            actions=actions,
            imagination_context_steps=warmup,
            imagination_horizon=max_horizon,
        )
    imagined = outputs.imagined_reconstructions.detach()
    target_start = outputs.imagined_target_start
    assert target_start is not None
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


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: MechanicsWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: MechanicsTrainConfig,
    metrics: dict[str, float],
    global_step: int,
    history_path: Path,
) -> Path:
    """Save model/optimizer/scheduler/RNG state and config."""

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
        "model_kind": "mechanics",
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    return epoch_path


def load_checkpoint(
    path: Path,
    model: MechanicsWorldModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    expected_config: MechanicsTrainConfig | None = None,
) -> tuple[int, int]:
    """Load a mechanics-model checkpoint. Returns ``(epoch, global_step)``."""

    # weights_only=False because we store NumPy RNG / scheduler state /
    # config dataclass alongside the model weights — same rationale as
    # train_rssm.load_checkpoint.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_kind") not in (None, "mechanics"):
        raise RuntimeError(
            "checkpoint model_kind="
            f"{checkpoint.get('model_kind')!r} is not 'mechanics'; "
            "load with the matching trainer (train_rssm for 'rssm').",
        )
    if expected_config is not None and "config" in checkpoint:
        validate_resume_config(checkpoint["config"], expected_config)
    OPTIONAL_NEW_MODULE_PREFIXES = ("reward_head.", "dissipation_module.")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    disallowed_missing = [
        n for n in missing
        if not any(n.startswith(p) for p in OPTIONAL_NEW_MODULE_PREFIXES)
    ]
    unexpected_disallowed = [
        n for n in unexpected
        if not any(n.startswith(p) for p in OPTIONAL_NEW_MODULE_PREFIXES)
    ]
    if disallowed_missing or unexpected_disallowed:
        raise RuntimeError(
            "checkpoint model state mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected_disallowed}",
        )
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except ValueError:
        print(
            "Warning: optimizer state in checkpoint is incompatible with the "
            "current optimizer; using fresh optimizer state.",
        )
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]), int(checkpoint.get("global_step", 0))


def validate_resume_config(
    saved: dict[str, Any], expected: MechanicsTrainConfig,
) -> None:
    """Raise if architecture/data fields differ between checkpoint and current config."""

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


def validate_config(config: MechanicsTrainConfig) -> None:
    """Validate the trainer config before doing any I/O."""

    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.log_every_steps < 1:
        raise ValueError("log_every_steps must be >= 1")
    if config.q_dim < 1:
        raise ValueError("q_dim must be >= 1")
    if config.nuisance_dim < 1:
        raise ValueError("nuisance_dim must be >= 1")
    if config.dt <= 0.0:
        raise ValueError("dt must be > 0")
    if config.n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")
    if config.foreground_reconstruction_weight < 0.0:
        raise ValueError("foreground_reconstruction_weight must be >= 0")
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
            "imagination_horizon must be >= 1 when any imagination weight > 0",
        )
    if config.reward_weight > 0.0 and not config.reward_head:
        raise ValueError("reward_weight > 0 requires --reward-head")
    if config.imagined_reward_weight > 0.0 and not config.reward_head:
        raise ValueError("imagined_reward_weight > 0 requires --reward-head")
    if imagination_enabled and config.sequence_length < (
        config.imagination_context_steps + config.imagination_horizon - 1
    ):
        raise ValueError(
            "sequence_length must be >= imagination_context_steps + "
            "imagination_horizon - 1 when imagination is enabled",
        )


def serializable_config(config: MechanicsTrainConfig) -> dict[str, Any]:
    """Convert config dataclass values to JSON/checkpoint-friendly objects."""

    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def resolve_history_path(config: MechanicsTrainConfig) -> Path:
    """Return the JSONL history path for this run."""

    return config.history_path or (config.checkpoint_dir / "history.jsonl")


def resolve_resume_path(config: MechanicsTrainConfig, console: Console) -> Path | None:
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
            console.print(f"[bold yellow]auto-resume[/bold yellow]: found {latest}")
            return latest
        console.print(
            "[dim]auto-resume requested but no latest.pt in "
            f"{config.checkpoint_dir} — starting fresh[/dim]",
        )
    return None


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
    config: MechanicsTrainConfig,
) -> None:
    """Print a Rich summary of the training configuration."""

    table = Table(title="Mechanics World-Model Training Run", show_header=False, box=None)
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
    table.add_row("q_dim", str(config.q_dim))
    table.add_row("nuisance_dim", str(config.nuisance_dim))
    table.add_row("with_dissipation", str(config.with_dissipation))
    table.add_row("dt", f"{config.dt:.4f}")
    table.add_row("imagination_horizon", str(config.imagination_horizon))
    console.print(table)


def main() -> None:
    """Run the mechanics training CLI."""

    console = Console()
    config = config_from_args(build_arg_parser().parse_args())
    console.rule("[bold cyan]Mechanics training config[/]")
    console.print_json(json.dumps(serializable_config(config), indent=2))
    train(config, console=console)


if __name__ == "__main__":
    main()
